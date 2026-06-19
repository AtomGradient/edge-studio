# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Multi-frequency parallel memory stack.

Hypothesis (`ultra/03 §1.2`): different decay rates capture different
temporal scales:

    α ≈ 0.1    — very fast (single turn)
    α ≈ 0.95   — day-scale (one conversation)
    α ≈ 0.999  — long-range (weeks of interaction)

The stack runs N MemoryBlocks in parallel on the same input stream;
each block uses its own decay. The final readout is the concatenation of
all branches along the feature axis.

Concrete SSMs (RWKV v7 / DeltaNet / xLSTM) accept a decay via a
keyword argument on forward_sequence; the exact knob name differs per
SSM (RWKV decay, DeltaNet beta, xLSTM forget gate) — the stack hides
that by passing the alpha through ``decay_override`` which each
concrete subclass interprets appropriately.
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from edgestudio_core.memory.base import MemoryBlock, StateT


class MultiFreqMemoryStack(nn.Module):
    """Run N MemoryBlocks in parallel at different decay rates.

    All sub-blocks must share ``readout_dim``; the stack's effective
    readout dim is ``N × readout_dim``.
    """

    def __init__(
        self,
        blocks: Sequence[MemoryBlock],
        alphas: Sequence[float] = (0.1, 0.95, 0.999),
    ):
        super().__init__()
        if len(blocks) != len(alphas):
            raise ValueError(
                f"len(blocks)={len(blocks)} must match len(alphas)={len(alphas)}"
            )
        if not all(b.readout_dim == blocks[0].readout_dim for b in blocks):
            raise ValueError(
                "All MemoryBlocks in a stack must share readout_dim"
            )
        self.blocks = list(blocks)
        self.alphas = tuple(alphas)

    @property
    def readout_dim(self) -> int:
        return sum(b.readout_dim for b in self.blocks)

    def init_state(self, batch_size: int) -> list[StateT]:
        return [b.init_state(batch_size) for b in self.blocks]

    def forward_sequence(
        self,
        xs: mx.array,
        states: list[StateT] | None = None,
    ) -> tuple[mx.array, list[StateT]]:
        """Run each branch in parallel at its own decay rate.

        Parameters
        ----------
        xs : mx.array, shape ``(B, T, input_dim)``
        states : list of pytrees or ``None``

        Returns
        -------
        readouts : mx.array, shape ``(B, T, N × readout_dim)``
        final_states : list of pytrees, one per branch
        """
        if states is None:
            B = xs.shape[0]
            states = [b.init_state(B) for b in self.blocks]

        branch_readouts: list[mx.array] = []
        new_states: list[StateT] = []
        for block, alpha, s in zip(self.blocks, self.alphas, states):
            # Subclasses opt into decay_override via kwargs; if they don't
            # implement it the alpha is silently ignored (identity block).
            try:
                r, ns = block.forward_sequence(xs, s, decay_override=alpha)
            except TypeError:
                # Identity / back-compat: ignore the knob
                r, ns = block.forward_sequence(xs, s)
            branch_readouts.append(r)
            new_states.append(ns)
        return mx.concatenate(branch_readouts, axis=-1), new_states

    def final_readout(
        self,
        xs: mx.array,
        states: list[StateT] | None = None,
    ) -> tuple[mx.array, list[StateT]]:
        """Convenience: run forward_sequence, return only the LAST step's
        readout ``(B, N × readout_dim)`` plus final state.
        """
        rs, ns = self.forward_sequence(xs, states)
        return rs[:, -1, :], ns
