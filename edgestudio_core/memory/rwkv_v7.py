# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""RWKV v7 ("Goose") MemoryBlock — S2b first concrete SSM backend.

First real subclass of ``edgestudio_core.memory.MemoryBlock`` after the S2a identity
reference. Tracks the upstream reference implementation at
``BlinkDL/RWKV-LM/RWKV-v7/rwkv_v7_numpy.py`` — in particular the v7
**generalized delta rule** state update plus v7's full gating stack (6-way
token shift, two-stage gated decay, in-context learning rate ``a``, kk
split with L2 normalisation, per-head r·k·r_k bonus, output gate ``g``).

Per-head state layout (pytree):
    S     : mx.array (B, H, D_h, D_h)   — matrix-valued recurrent memory
    shift : mx.array (B, input_dim)     — previous input for token shift

Per-step update (one token per batch entry):

    # 6-way token shift (per-channel mixing coefficients μ_r, μ_w, μ_k, μ_v, μ_a, μ_g):
    x_r = x + μ_r · (prev − x);  (similarly for w, k, v, a, g)

    # Standard projections + v7 two-stage gated decay:
    r = W_r · x_r;   k = W_k · x_k;   v = W_v · x_v
    w = exp( −sigmoid(tanh(x_w · W_w1) · W_w2 + b_w) / √e )      # ∈ (exp(−1/√e), 1)
    a = sigmoid(x_a · W_a1 · W_a2 + b_a)                         # in-context LR
    g = sigmoid(x_g · W_g1) · W_g2                               # output gate

    # kk split (generalised delta rule's "erase" direction):
    kk = k · k_k                   # per-channel scale
    k  = k + k · (a − 1) · k_a     # v7 k-mix

    # Reshape to heads (column vectors per head), L2-normalise kk along D_h:
    r, k, v, kk, a, w → (B, H, D_h, 1)
    kk := kk / max(‖kk‖₂, ε)

    # Generalised Delta Rule (v7 core):
    S_new  =  S · diag(w)   −   (S · kk) · (kk · a)ᵀ   +   v · kᵀ
    #           decay            erase direction         write

    # Readout: WKV lookup + per-head r·k·r_k bonus × v, then output gate:
    y      =  S_new · r   +   (r · k · r_k).sum · v
    output =  W_o · (flatten(y) · g)

Design deviations from the full RWKV-7 reference (documented in
``docs/winforver/S2b_rwkv_v7.md`` §2):

1. **No v0 value residual.** The reference mixes the first layer's ``v``
   into every subsequent layer's ``v`` via a sigmoid gate. That is a
   cross-layer coupling not exposed by the ``MemoryBlock`` single-block
   contract. Adding it later requires a multi-block wrapper.

2. **No per-head GroupNorm on the readout.** The reference applies a
   per-head group-norm to ``y`` before the output-gate mix. We leave
   that as a post-projection op (caller can add ``mlx.nn.GroupNorm`` on
   the readout) so the MemoryBlock stays a pure state transducer.

3. **Decay override for multi-freq.** ``MultiFreqMemoryStack`` passes
   ``α ∈ [0, 1)`` via ``decay_override``. We floor the data-dependent
   decay at α:  ``w_eff = α + (1 − α) · w`` — so α=0.999 biases toward
   long-range retention while the per-token residual still modulates
   within ``[α, 1)``. This matches the multi-freq hypothesis in
   ``ultra/03 §1.2`` (α = 0.1 / 0.95 / 0.999 capture single-turn /
   day-scale / weeks-long patterns).

4. **Pure-MLX scan.** ``forward_sequence`` is a Python for-loop over
   ``step``. Step 5 of the S2b plan permits a Metal-kernel fusion once
   training throughput is measured; the contract test
   ``test_forward_sequence_matches_step`` will regress any fused path.
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from edgestudio_core.memory.base import MemoryBlock, StateT


def _swap_last_two(x: mx.array) -> mx.array:
    """Transpose the last two axes of ``x`` (portable across mlx versions)."""
    axes = list(range(x.ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return mx.transpose(x, axes=axes)


class RWKVv7Block(MemoryBlock):
    """RWKV v7 generalised-delta-rule block as a ``MemoryBlock`` implementation.

    Parameters
    ----------
    state_dim : int
        Total recurrent capacity ``D = n_heads · head_dim``.
    readout_dim : int, optional
        Per-step output dim after the final ``W_o`` projection. Defaults to
        ``state_dim``.
    input_dim : int, optional
        Expected feature dim of ``x``. Defaults to ``state_dim`` so the
        contract test factory (``_make_block(name, state_dim=D, readout_dim=D)``)
        works without a third argument.
    n_heads : int, default 4
        Number of heads for the WKV outer product. If ``head_dim`` is not
        supplied and ``state_dim`` is not divisible by ``n_heads``,
        ``n_heads`` is shrunk to the nearest divisor (so ``state_dim=1``
        yields ``n_heads=1``).
    head_dim : int, optional
        Per-head dim ``D_h``. Must satisfy ``state_dim == n_heads · head_dim``.
        Defaults to ``state_dim // n_heads``.
    gate_rank : int, optional
        Low-rank inner dim for the ``w`` / ``a`` / ``g`` two-stage gates.
        Defaults to ``clip(state_dim // 2, 8, 64)``. v7 uses low rank here
        to keep gate-parameter count subquadratic in ``D``.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        readout_dim: int | None = None,
        input_dim: int | None = None,
        n_heads: int = 4,
        head_dim: int | None = None,
        gate_rank: int | None = None,
    ):
        super().__init__()
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")

        self.input_dim = int(input_dim if input_dim is not None else state_dim)
        self.state_dim = int(state_dim)
        self.readout_dim = int(readout_dim if readout_dim is not None else state_dim)

        # Auto-shrink n_heads if head_dim unspecified and state_dim not divisible.
        if head_dim is None:
            while n_heads > 1 and state_dim % n_heads != 0:
                n_heads -= 1
            head_dim = state_dim // n_heads

        if n_heads * head_dim != state_dim:
            raise ValueError(
                f"state_dim must equal n_heads·head_dim, got {state_dim} "
                f"vs {n_heads}·{head_dim}"
            )
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)

        if gate_rank is None:
            gate_rank = max(8, min(64, state_dim // 2 or 8))
        self.gate_rank = int(gate_rank)

        D = self.state_dim

        # -- Standard linear projections (full-rank, no bias per reference) --
        self.W_r = nn.Linear(self.input_dim, D, bias=False)
        self.W_k = nn.Linear(self.input_dim, D, bias=False)
        self.W_v = nn.Linear(self.input_dim, D, bias=False)
        self.W_o = nn.Linear(D, self.readout_dim, bias=False)

        # -- v7 two-stage low-rank gates for decay w, learning-rate a, output g --
        self.W_w1 = nn.Linear(self.input_dim, self.gate_rank, bias=False)
        self.W_w2 = nn.Linear(self.gate_rank, D, bias=False)
        self.w_bias = mx.zeros((D,), dtype=mx.float32)

        self.W_a1 = nn.Linear(self.input_dim, self.gate_rank, bias=False)
        self.W_a2 = nn.Linear(self.gate_rank, D, bias=False)
        self.a_bias = mx.zeros((D,), dtype=mx.float32)

        self.W_g1 = nn.Linear(self.input_dim, self.gate_rank, bias=False)
        self.W_g2 = nn.Linear(self.gate_rank, D, bias=False)

        # -- 6-way token-shift coefficients (per input channel, trainable) --
        init_mix = 0.5  # standard RWKV init — equal weight on current & prev token
        self.mu_r = mx.full((self.input_dim,), init_mix, dtype=mx.float32)
        self.mu_w = mx.full((self.input_dim,), init_mix, dtype=mx.float32)
        self.mu_k = mx.full((self.input_dim,), init_mix, dtype=mx.float32)
        self.mu_v = mx.full((self.input_dim,), init_mix, dtype=mx.float32)
        self.mu_a = mx.full((self.input_dim,), init_mix, dtype=mx.float32)
        self.mu_g = mx.full((self.input_dim,), init_mix, dtype=mx.float32)

        # -- Per-channel scalar params for the k-kk split + per-head bonus --
        # k_k=1 → kk=k at init (identity split).  k_a=0 → k unchanged at init.
        # r_k=0 → zero per-head bonus at init, so the readout reduces to the
        # pure WKV lookup (y = S·r) before training. This matches the reference
        # behaviour when those params start near zero.
        self.k_k = mx.ones((D,), dtype=mx.float32)
        self.k_a = mx.zeros((D,), dtype=mx.float32)
        self.r_k = mx.zeros((self.n_heads, self.head_dim), dtype=mx.float32)

        # 1/√e — scales sigmoid(gated) so w ∈ (exp(-1/√e), 1) ≈ (0.545, 1).
        self._inv_sqrt_e = float(1.0 / math.sqrt(math.e))

    # ------------------------------------------------------------------
    # MemoryBlock API
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int) -> StateT:
        return {
            "S": mx.zeros(
                (batch_size, self.n_heads, self.head_dim, self.head_dim),
                dtype=mx.float32,
            ),
            "shift": mx.zeros((batch_size, self.input_dim), dtype=mx.float32),
        }

    def step(
        self,
        x: mx.array,
        state: StateT,
        *,
        decay_override: float | None = None,
    ) -> tuple[mx.array, StateT]:
        B = x.shape[0]
        H, Dh = self.n_heads, self.head_dim
        D = H * Dh

        prev = state["shift"]

        # 6-way token shift — per reference:  xZ = x + μ_Z · (prev − x)
        x_r = x + self.mu_r * (prev - x)
        x_w = x + self.mu_w * (prev - x)
        x_k = x + self.mu_k * (prev - x)
        x_v = x + self.mu_v * (prev - x)
        x_a = x + self.mu_a * (prev - x)
        x_g = x + self.mu_g * (prev - x)

        # Standard r / k / v projections.
        r = self.W_r(x_r)  # (B, D)
        k = self.W_k(x_k)
        v = self.W_v(x_v)

        # Two-stage gated decay.  w_inner: tanh then low-rank up-project.
        w_raw = self.W_w2(mx.tanh(self.W_w1(x_w))) + self.w_bias   # (B, D)
        w = mx.exp(-mx.sigmoid(w_raw) * self._inv_sqrt_e)          # ∈ (0.545, 1)

        if decay_override is not None:
            alpha = float(decay_override)
            if not 0.0 <= alpha < 1.0:
                raise ValueError(
                    f"decay_override must be in [0, 1), got {alpha}"
                )
            # Floor decay at α — state cannot decay faster than α allows.
            w = alpha + (1.0 - alpha) * w

        # In-context learning rate a ∈ (0, 1) — modulates delta-rule magnitude.
        a = mx.sigmoid(self.W_a2(self.W_a1(x_a)) + self.a_bias)    # (B, D)

        # Output gate g — sigmoid inner, linear outer (per reference).
        g = self.W_g2(mx.sigmoid(self.W_g1(x_g)))                  # (B, D)

        # k-kk split:
        #   kk  : direction to erase in the delta-rule update (L2-normalised per head)
        #   k   : adjusted key written into state — v7 lets ``a`` modulate it
        kk = k * self.k_k
        k = k + k * (a - 1.0) * self.k_a

        # Reshape all per-head tensors to column vectors: (B, H, D_h, 1).
        r_h = r.reshape(B, H, Dh, 1)
        k_h = k.reshape(B, H, Dh, 1)
        v_h = v.reshape(B, H, Dh, 1)
        kk_h = kk.reshape(B, H, Dh, 1)
        a_h = a.reshape(B, H, Dh, 1)
        w_h = w.reshape(B, H, Dh, 1)

        # L2-normalise kk per head along D_h (so kk is a unit erase direction).
        kk_norm = mx.sqrt((kk_h * kk_h).sum(axis=-2, keepdims=True))
        kk_h = kk_h / mx.maximum(kk_norm, mx.array(1e-12, dtype=kk_norm.dtype))

        # --- Generalised Delta Rule ---
        # S_new = S · diag(w)  −  (S · kk) · (kk · a)ᵀ  +  v · kᵀ
        # Shapes:
        #   S_prev          : (B, H, D_h, D_h)
        #   w_col           : (B, H, 1,   D_h)    broadcasts across rows
        #   (S · kk)        : (B, H, D_h, 1)      column vector per head
        #   (kk · a)ᵀ       : (B, H, 1,   D_h)    row vector per head
        #   delta_term      : (B, H, D_h, D_h)    outer product
        #   v · kᵀ          : (B, H, D_h, D_h)    outer product
        S_prev = state["S"]
        w_col = _swap_last_two(w_h)
        S_kk = S_prev @ kk_h
        kk_a_T = _swap_last_two(kk_h * a_h)
        delta_term = S_kk * kk_a_T              # (B,H,Dh,1) · (B,H,1,Dh) → outer
        new_term = v_h * _swap_last_two(k_h)    # (B,H,Dh,1) · (B,H,1,Dh) → outer
        S_new = S_prev * w_col - delta_term + new_term

        # Readout:  y = S_new · r   +   (r · k · r_k).sum_{D_h} · v
        y = S_new @ r_h                                             # (B,H,Dh,1)
        r_k_h = self.r_k.reshape(1, H, Dh, 1)                       # broadcasts B
        bonus = (r_h * k_h * r_k_h).sum(axis=-2, keepdims=True)     # (B,H,1,1)
        y = y + bonus * v_h                                         # (B,H,Dh,1)

        # Flatten heads, apply output gate, project to readout_dim.
        y_flat = y.reshape(B, D)
        readout = self.W_o(y_flat * g)                              # (B, readout_dim)

        return readout, {"S": S_new, "shift": x}

    def forward_sequence(
        self,
        xs: mx.array,
        state: StateT | None = None,
        *,
        decay_override: float | None = None,
    ) -> tuple[mx.array, StateT]:
        # Correctness before speed (S2b §Step 5): simple Python scan. The
        # contract test asserts ``forward_sequence == step`` step-by-step,
        # so any future Metal-fused scan must regress against this path.
        B, T, _ = xs.shape
        if state is None:
            state = self.init_state(B)
        readouts: list[mx.array] = []
        s: Any = state
        for t in range(T):
            r, s = self.step(xs[:, t, :], s, decay_override=decay_override)
            readouts.append(r)
        return mx.stack(readouts, axis=1), s
