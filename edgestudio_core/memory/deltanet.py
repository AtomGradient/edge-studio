# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from edgestudio_core.memory.base import MemoryBlock, StateT


class DeltaNetBlock(MemoryBlock):
    """DeltaNet memory kernel (linear-attention / fast-weight delta rule).

    Parameters
    ----------
    state_dim : int
        Hidden state budget — used to size default per-head dims so that
        ``n_heads · head_dim_{k,v} == state_dim``. Contract-test friendly.
    readout_dim : int, optional
        Output dim of ``W_out``. Defaults to ``state_dim``.
    input_dim : int, optional
        Feature dim of ``x``. Defaults to ``state_dim``.
    n_heads : int, default 4
        Number of parallel delta-rule heads. Auto-shrinks if ``state_dim``
        is not divisible (so small contract-test configs still work).
    head_dim_k, head_dim_v : int, optional
        Per-head key / value dims. Default to ``state_dim // n_heads``.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        readout_dim: int | None = None,
        input_dim: int | None = None,
        n_heads: int = 4,
        head_dim_k: int | None = None,
        head_dim_v: int | None = None,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        # Auto-shrink n_heads so ``state_dim % n_heads == 0`` (mirrors the
        # behaviour in VSAMemoryBlock / RWKVv7Block so contract tests that
        # pass state_dim ∈ {16, 32, 64} don't trip).
        while n_heads > 1 and state_dim % n_heads != 0:
            n_heads -= 1

        default_per_head = state_dim // n_heads
        head_dim_k = int(head_dim_k if head_dim_k is not None else default_per_head)
        head_dim_v = int(head_dim_v if head_dim_v is not None else default_per_head)

        self.input_dim = int(input_dim if input_dim is not None else state_dim)
        self.state_dim = int(state_dim)
        self.readout_dim = int(readout_dim if readout_dim is not None else state_dim)
        self.n_heads = int(n_heads)
        self.d_k = head_dim_k
        self.d_v = head_dim_v

        # --- Learned projections --------------------------------------
        self.W_q = nn.Linear(self.input_dim, self.n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(self.input_dim, self.n_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(self.input_dim, self.n_heads * self.d_v, bias=False)
        # β per-head sigmoid in (0, 1) — start around 0.5 by default.
        self.W_beta = nn.Linear(self.input_dim, self.n_heads, bias=True)
        self.W_out = nn.Linear(self.n_heads * self.d_v, self.readout_dim, bias=False)

    # Key normalisation: **L2 norm** (per head) so ``||k|| = 1``. The delta
    # rule requires unit keys — otherwise ``v_pred = M·k = ||k||² · v``
    # after the first write, and the re-read error is ``(1 − ||k||²)·v``
    # instead of zero (S2c design doc §DeltaNet core, originally called
    # out RMSNorm, but that gives ``||k||² = D_k`` — verified wrong by the
    # ``second_write_with_same_x_has_zero_error`` regression 2026-05-08).

    @staticmethod
    def _l2_norm(k: mx.array) -> mx.array:
        return k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    # ------------------------------------------------------------------
    # MemoryBlock contract
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int) -> StateT:
        # State layout: ``M[b, h, k, v]`` — delta rule maps key components
        # (axis k) to value components (axis v).  Predicted value is the
        # contraction of M over the key axis with the query (see `step`).
        return {
            "M": mx.zeros(
                (batch_size, self.n_heads, self.d_k, self.d_v),
                dtype=mx.float32,
            )
        }

    def step(self, x: mx.array, state: StateT) -> tuple[mx.array, StateT]:
        B = x.shape[0]
        H, D_k, D_v = self.n_heads, self.d_k, self.d_v

        q = self.W_q(x).reshape(B, H, D_k)                     # (B, H, D_k)
        k = self._l2_norm(self.W_k(x).reshape(B, H, D_k))      # unit-norm keys
        v = self.W_v(x).reshape(B, H, D_v)                     # (B, H, D_v)
        # Sigmoid per head; reshape to broadcast over the outer-product axes.
        beta = mx.sigmoid(self.W_beta(x)).reshape(B, H, 1, 1)  # (B, H, 1, 1)

        M = state["M"]                                          # (B, H, D_k, D_v)

        # Delta rule: the current M's prediction given this key.
        v_pred = mx.einsum("bhkv,bhk->bhv", M, k)              # (B, H, D_v)
        err = v - v_pred                                        # (B, H, D_v)

        # Rank-1 update driven by prediction error: ΔM[h,k,v] = β · k[h,k] · err[h,v]
        delta = beta * k[:, :, :, None] * err[:, :, None, :]   # (B, H, D_k, D_v)
        M_new = M + delta

        # Readout via the *updated* M so the current step sees the write.
        y = mx.einsum("bhkv,bhk->bhv", M_new, q)               # (B, H, D_v)
        readout = self.W_out(y.reshape(B, H * D_v))            # (B, readout_dim)
        return readout, {"M": M_new}

    def forward_sequence(
        self, xs: mx.array, state: StateT | None = None
    ) -> tuple[mx.array, StateT]:
        B, T, _ = xs.shape
        if state is None:
            state = self.init_state(B)
        # Step-by-step is the spec for the contract test; the sequential
        # implementation here matches bit-exactly (delta rule is not
        # associatively commutative like RWKV's scan, so a fused kernel
        # would still iterate).
        readouts = []
        cur = state
        for t in range(T):
            r, cur = self.step(xs[:, t, :], cur)
            readouts.append(r)
        return mx.stack(readouts, axis=1), cur
