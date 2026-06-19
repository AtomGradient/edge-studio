# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from edgestudio_core.memory.base import MemoryBlock, StateT


# Logit value that makes σ(·) essentially 0. Not a literal −∞ to avoid
# producing NaNs through gradients of a nearly-saturated sigmoid.
_INIT_GATE_BIAS = -6.0   # σ(-6) ≈ 2.47e-3 — practically 0 but still differentiable


class MemoryMixer(MemoryBlock):

    def __init__(
        self,
        block_a: MemoryBlock,
        block_b: MemoryBlock,
        *,
        input_dim: int | None = None,
        gate_init_bias: float = _INIT_GATE_BIAS,
    ):
        super().__init__()
        if block_a.readout_dim != block_b.readout_dim:
            raise ValueError(
                f"block readout_dim mismatch: a={block_a.readout_dim} "
                f"b={block_b.readout_dim}"
            )
        self.block_a = block_a
        self.block_b = block_b

        self.readout_dim = block_a.readout_dim
        # state_dim is a book-keeping int — other code uses it as a hint, not
        # a tensor shape. Reporting the sum reflects capacity honestly.
        self.state_dim = int(block_a.state_dim) + int(block_b.state_dim)

        if input_dim is None:
            input_dim = int(getattr(block_a, "input_dim", block_a.state_dim))
        self.input_dim = int(input_dim)

        # Gate linear: input → 1 scalar logit, with init bias pushing σ to 0.
        # bias=True so we can set the initial value directly.
        self.W_mix = nn.Linear(self.input_dim, 1, bias=True)
        # nn.Linear weights default-init via glorot; we only override bias to
        # enforce the "B starts inert" invariant.
        self.W_mix.bias = mx.full((1,), float(gate_init_bias), dtype=mx.float32)

    # ------------------------------------------------------------------
    # MemoryBlock contract
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int) -> StateT:
        return {
            "A": self.block_a.init_state(batch_size),
            "B": self.block_b.init_state(batch_size),
        }

    def _gate(self, x: mx.array) -> mx.array:
        """Compute per-token scalar gate ``g ∈ (0, 1)`` with shape (B, 1)."""
        return mx.sigmoid(self.W_mix(x))

    def step(self, x: mx.array, state: StateT) -> tuple[mx.array, StateT]:
        r_a, ns_a = self.block_a.step(x, state["A"])
        r_b, ns_b = self.block_b.step(x, state["B"])
        g = self._gate(x)                                # (B, 1)
        out = (1.0 - g) * r_a + g * r_b                  # (B, readout_dim)
        return out, {"A": ns_a, "B": ns_b}

    def forward_sequence(
        self, xs: mx.array, state: StateT | None = None
    ) -> tuple[mx.array, StateT]:
        B, T, _ = xs.shape
        if state is None:
            state = self.init_state(B)
        readouts = []
        cur = state
        for t in range(T):
            r, cur = self.step(xs[:, t, :], cur)
            readouts.append(r)
        return mx.stack(readouts, axis=1), cur

    # ------------------------------------------------------------------
    # Diagnostics — NOT part of the contract.
    # ------------------------------------------------------------------

    def gate_at(self, x: mx.array) -> mx.array:
        """Public handle on the gate values for a batch. Useful for training
        instrumentation: watch the gate stats to see if VSA branch is being
        activated over epochs."""
        return self._gate(x)

    # ------------------------------------------------------------------
    # S3d Phase 3b — gate aux loss (decision β in §4.3.1 decision log)
    # ------------------------------------------------------------------

    def gate_trace(self, xs: mx.array) -> mx.array:
        """Per-timestep gate values over a batched sequence.

        Sidesteps the full ``step``/``forward_sequence`` path so callers
        that just want to observe the gate (training instrumentation, aux
        loss) do not pay for block kernel evaluations or touch state. The
        computation is bit-exact equivalent to ``self._gate(xs[:, t, :])``
        applied pointwise — ``nn.Linear`` broadcasts over leading dims.

        Parameters
        ----------
        xs : (B, T, input_dim) input sequence.

        Returns
        -------
        (B, T, 1) gate values in (0, 1).
        """
        return mx.sigmoid(self.W_mix(xs))

    def gate_variance_loss(self, xs: mx.array, lam: float) -> mx.array:
        if lam == 0.0:
            return mx.array(0.0, dtype=mx.float32)
        g = self.gate_trace(xs)                    # (B, T, 1)
        var = g.reshape(-1).var()
        return -float(lam) * var
