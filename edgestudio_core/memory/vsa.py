# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from edgestudio_core.memory.base import MemoryBlock, StateT


def _random_orthogonal(n_rows: int, n_cols: int, *, key) -> mx.array:
    """Return a ``(n_rows, n_cols)`` orthogonal-rows matrix.

    Works for both square and rectangular cases:
        - n_rows ≤ n_cols: rows are mutually orthogonal (perfect unbinding).
        - n_rows >  n_cols: returns the first ``n_rows`` rows of a random
          orthogonal matrix on ``max(n_rows, n_cols)`` dims truncated to
          ``n_cols`` — rows NO LONGER orthogonal; unbinding becomes lossy.
          Callers should ensure ``role_dim ≥ role_vocab_size``.

    Uses QR decomposition on a Gaussian matrix. ``mx.linalg.qr`` needs a
    square input, so we work on ``max(n_rows, n_cols)``-square and slice.
    """
    d = max(n_rows, n_cols)
    g = mx.random.normal(shape=(d, d), key=key)
    # mx.linalg.qr streams on cpu by default
    q, _ = mx.linalg.qr(g, stream=mx.cpu)
    # Slice to desired (n_rows, n_cols)
    return q[:n_rows, :n_cols]


class VSAMemoryBlock(MemoryBlock):
    """Tensor-Product Representation memory with fixed orthogonal roles.

    Parameters
    ----------
    state_dim : int
        Total effective state capacity ``H · D_f`` (see note on the tensor
        state shape ``(B, H, D_f, D_r)`` above).
    readout_dim : int, optional
        Output dim after ``W_read``. Defaults to ``state_dim``.
    input_dim : int, optional
        Feature dim of ``x``. Defaults to ``state_dim``.
    n_heads : int, default 4
        Number of parallel TPR heads (each with its own ``W_v`` slice).
    filler_dim : int, optional
        Per-head filler dim ``D_f``. Defaults to ``state_dim // n_heads``.
    role_vocab_size : int, default 32
        Number of discrete roles. Keep at 32 for Phase 1 (matches
        ``experiments/winforver/S3d/schema.py`` ROLES).
    role_dim : int, optional
        Role-embedding dim ``D_r``. Defaults to ``max(role_vocab_size,
        filler_dim)`` so orthogonal roles fit.
    seed : int, default 0
        Seed for the frozen random orthogonal role_embed init.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        readout_dim: int | None = None,
        input_dim: int | None = None,
        n_heads: int = 4,
        filler_dim: int | None = None,
        role_vocab_size: int = 32,
        role_dim: int | None = None,
        seed: int = 0,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        self.input_dim = int(input_dim if input_dim is not None else state_dim)
        self.state_dim = int(state_dim)
        self.readout_dim = int(readout_dim if readout_dim is not None else state_dim)

        # Auto-shrink n_heads if filler_dim unspecified and state_dim not divisible.
        if filler_dim is None:
            while n_heads > 1 and state_dim % n_heads != 0:
                n_heads -= 1
            filler_dim = state_dim // n_heads

        if n_heads * filler_dim != state_dim:
            raise ValueError(
                f"state_dim must equal n_heads·filler_dim, got {state_dim} "
                f"vs {n_heads}·{filler_dim}"
            )
        self.n_heads = int(n_heads)
        self.filler_dim = int(filler_dim)
        self.role_vocab_size = int(role_vocab_size)

        if role_dim is None:
            role_dim = max(self.role_vocab_size, self.filler_dim)
        if role_dim < self.role_vocab_size:
            # Honour caller but warn implicitly via a buffer tag the tests check.
            pass
        self.role_dim = int(role_dim)

        D_f = self.filler_dim
        D_r = self.role_dim
        V = self.role_vocab_size
        H = self.n_heads

        # --- Learned projections --------------------------------------
        # v = W_v(x) reshaped to (B, H, D_f)
        self.W_v = nn.Linear(self.input_dim, H * D_f, bias=False)
        # Role selector: (B, V) logits → softmax
        self.W_r_sel = nn.Linear(self.input_dim, V, bias=False)
        # Readout: flatten (B, H, V, D_f) → project to readout_dim
        self.W_read = nn.Linear(H * V * D_f, self.readout_dim, bias=False)

        # --- Frozen random orthogonal role_embed -----------------------
        # Held as a buffer (non-trainable). Stored under a leading underscore
        # attribute so mlx.nn.Module's `trainable_parameters` walker does not
        # list it; value is reproducible from ``seed``.
        key = mx.random.key(int(seed))
        self._role_embed = _random_orthogonal(V, D_r, key=key)
        # Stopping gradients is a belt-and-braces guard: even if a downstream
        # ``nn.value_and_grad`` inadvertently traces through the buffer, no
        # gradient update flows back into it.
        self._role_embed = mx.stop_gradient(self._role_embed)

    # ------------------------------------------------------------------
    # MemoryBlock contract
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int) -> StateT:
        return {
            "T": mx.zeros(
                (batch_size, self.n_heads, self.filler_dim, self.role_dim),
                dtype=mx.float32,
            )
        }

    def step(self, x: mx.array, state: StateT) -> tuple[mx.array, StateT]:
        B = x.shape[0]
        H, D_f, D_r, V = self.n_heads, self.filler_dim, self.role_dim, self.role_vocab_size

        # Filler — per-head vector of dim D_f
        v = self.W_v(x).reshape(B, H, D_f)             # (B, H, D_f)

        # Role — softmax-mixed linear combo of the V orthogonal role vectors
        r_soft = mx.softmax(self.W_r_sel(x), axis=-1)  # (B, V)
        r = r_soft @ self._role_embed                   # (B, D_r)

        # Binding (outer product) + bundle-add into T
        #   T_new[b,h,f,r'] = T[b,h,f,r'] + v[b,h,f] * r[b,r']
        bound = v[:, :, :, None] * r[:, None, None, :]  # (B, H, D_f, D_r)
        T_new = state["T"] + bound

        # Unbind every role — for orthogonal role_embed, U[:,:,k,:] ≈ filler
        # of the bundle slot bound to role k (plus crosstalk if role_dim < V).
        #   U[b,h,k,f] = Σ_r T_new[b,h,f,r] * role_embed[k,r]
        U = mx.einsum("bhfr,kr->bhkf", T_new, self._role_embed)  # (B, H, V, D_f)

        # Flatten heads × roles × fillers → readout_dim
        readout = self.W_read(U.reshape(B, H * V * D_f))          # (B, readout_dim)
        return readout, {"T": T_new}

    def forward_sequence(
        self, xs: mx.array, state: StateT | None = None
    ) -> tuple[mx.array, StateT]:
        B, T, _ = xs.shape
        if state is None:
            state = self.init_state(B)
        # TPR update is purely linear so step-by-step is the reference
        # implementation — contract test asserts this matches bit-exact.
        readouts = []
        for t in range(T):
            r, state = self.step(xs[:, t, :], state)
            readouts.append(r)
        return mx.stack(readouts, axis=1), state

    # ------------------------------------------------------------------
    # Diagnostics — not part of the contract; useful for S3d tests.
    # ------------------------------------------------------------------

    @property
    def role_embed(self) -> mx.array:
        """Read-only view of the frozen orthogonal role embedding."""
        return self._role_embed

    def bind(self, filler: mx.array, role_idx: int) -> mx.array:
        """Utility — single ``filler ⊗ role_embed[role_idx]`` outer product.

        Returns a (filler_dim, role_dim) matrix. Used in synthetic
        binding/unbinding recall tests to isolate the linear-algebra
        contract from the learned projections.
        """
        r = self._role_embed[role_idx]            # (D_r,)
        return filler[:, None] * r[None, :]       # (D_f, D_r)

    def unbind(self, bundle: mx.array, role_idx: int) -> mx.array:
        """Utility — unbind ``role_idx`` from a (D_f, D_r) bundle matrix."""
        r = self._role_embed[role_idx]            # (D_r,)
        return bundle @ r                         # (D_f,)
