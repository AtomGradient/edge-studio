# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""MemoryBlock abstract base + IdentityMemoryBlock reference implementation.

All three planned SSM backends (RWKV v7 / DeltaNet / xLSTM) subclass
``MemoryBlock``. The contract test ``tests/winforver/test_memory_block_contract.py``
drives every subclass through identical shape / save-load / batching
invariants so the S2b/c/d ablation stays single-variable (see
``docs/winforver/S2a_memory_block_abstraction.md``).

State is pytree-structured — concrete blocks are free to pack it as
nested dicts / tuples of ``mx.array``; ``save_state`` / ``load_state``
round-trip it through MLX safetensors with required ``format="mlx"``
metadata (per memory ``feedback_mlx_safetensors_metadata``).
"""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


StateT = Any  # pytree: nested dict/tuple/list of mx.array leaves


# Safetensors metadata for MLX — memory feedback_mlx_safetensors_metadata.
_MLX_META = {"format": "mlx"}


class MemoryBlock(nn.Module):
    """Abstract SSM memory kernel.

    Concrete subclasses implement the five abstract methods below. They
    must also set ``state_dim`` and ``readout_dim`` attributes (as class
    or instance constants) before ``__init__`` returns.

    Shape conventions
    -----------------
    ``step`` processes one time-step per batch entry; ``forward_sequence``
    processes the full sequence at once (parallel scan-friendly) and must
    match ``step`` when run token-by-token.
    """

    #: Hidden state dimension; subclass must set.
    state_dim: int
    #: Output (readout) dimension per step; subclass must set.
    readout_dim: int

    # ------------------------------------------------------------------
    # Required API — subclass overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def init_state(self, batch_size: int) -> StateT:
        """Return a fresh zero-initialised state pytree for ``batch_size``.

        Leaves must be ``mx.array``; shapes must be deterministic given
        ``batch_size`` + ``self.state_dim``.
        """

    @abstractmethod
    def step(self, x: mx.array, state: StateT) -> tuple[mx.array, StateT]:
        """Advance one step.

        Parameters
        ----------
        x : mx.array, shape ``(B, input_dim)``
        state : pytree, matches ``init_state(B)``

        Returns
        -------
        readout : mx.array, shape ``(B, readout_dim)``
        new_state : pytree, same structure as ``state``
        """

    @abstractmethod
    def forward_sequence(
        self, xs: mx.array, state: StateT | None = None
    ) -> tuple[mx.array, StateT]:
        """Run the full sequence.

        Parameters
        ----------
        xs : mx.array, shape ``(B, T, input_dim)``
        state : initial state pytree or ``None`` (init from zero)

        Returns
        -------
        readouts : mx.array, shape ``(B, T, readout_dim)``
        final_state : pytree

        The contract test asserts that this is numerically equal (within
        float tolerance) to repeatedly calling ``step`` for each t.
        """

    # ------------------------------------------------------------------
    # Serialisation — default implementation via pytree flattening.
    # Subclasses may override for custom layouts.
    # ------------------------------------------------------------------

    def save_state(self, state: StateT, path: str | Path) -> None:
        """Flatten state and dump to MLX-tagged safetensors.

        Uses ``mlx.utils.tree_flatten`` so arbitrary nested pytree shapes
        survive. Key layout is flat dotted strings, e.g. ``"h.layer0"``.
        Always writes metadata ``format=mlx`` so downstream mlx_vlm /
        mlx-lm readers don't get garbled output (per memory
        ``feedback_mlx_safetensors_metadata``).
        """
        flat = dict(tree_flatten(state))
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(p), flat, metadata=_MLX_META)

    def load_state(self, path: str | Path) -> StateT:
        """Inverse of ``save_state``."""
        flat = mx.load(str(path))
        return tree_unflatten(list(flat.items()))


class IdentityMemoryBlock(MemoryBlock):
    """Reference implementation — state is a single running zero tensor.

    This exists so (a) the contract test has something concrete to run
    against and (b) code that depends on the interface can smoke-test
    before the real SSMs land. Numerically: ``step`` returns ``x`` verbatim
    and ignores the state; ``forward_sequence`` is a no-op copy.

    Do **not** use in production — RWKV v7 / DeltaNet / xLSTM are the
    real backends.
    """

    def __init__(self, state_dim: int = 128, readout_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.readout_dim = readout_dim

    def init_state(self, batch_size: int) -> StateT:
        return {"h": mx.zeros((batch_size, self.state_dim), dtype=mx.float32)}

    def step(self, x: mx.array, state: StateT) -> tuple[mx.array, StateT]:
        # Identity passthrough; state accumulates sum for save/load round-trip
        # coverage (so ``load_state`` has non-zero content to verify).
        h = state["h"] + x[:, : self.state_dim]
        return x[:, : self.readout_dim], {"h": h}

    def forward_sequence(
        self, xs: mx.array, state: StateT | None = None
    ) -> tuple[mx.array, StateT]:
        B, T, _ = xs.shape
        if state is None:
            state = self.init_state(B)
        # Keep this method numerically equivalent to step-by-step iteration
        # (that's what the contract test asserts).
        readouts = xs[..., : self.readout_dim]
        h = state["h"]
        for t in range(T):
            h = h + xs[:, t, : self.state_dim]
        return readouts, {"h": h}
