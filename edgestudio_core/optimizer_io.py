# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Save / load Adam-family optimizer state across training sessions.

Why this exists
---------------
Small trainable heads and profile builders often run in multiple short
sessions. The first-/second-moment buffers (`m`, `v`) — i.e. the EMA of
gradients — encode useful information about the loss landscape. Throwing
them away between iterations is the equivalent of a cold restart.

Storage layout
--------------
    <out_dir>/
        optimizer.safetensors   # all mx.array leaves of optimizer.state
        optimizer.meta.json     # scalar leaves + bookkeeping

Implementation notes
--------------------
- `optimizer.state` is a *tree* (nested dict) whose leaves are either
  `mx.array` (the moment buffers and `step` counter, sometimes also the
  learning rate when a schedule is used) or plain Python scalars.
- We use `mlx.utils.tree_flatten/tree_unflatten` to map the tree to/from
  a flat dict keyed by dotted paths — the same convention used by
  EdgeStudio training utilities for array snapshots.
- `safetensors` is the right wire format for the array leaves and gives us
  zero-copy mmap on load. We MUST set `metadata={"format": "mlx"}` or the
  file will fail to round-trip through `mlx_vlm` and other tooling
  (see memory `feedback_mlx_safetensors_metadata.md`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


_TENSOR_FILE = "optimizer.safetensors"
_META_FILE = "optimizer.meta.json"
_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_state(state: dict) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """Flatten state into (tensor leaves, JSON-serialisable scalar leaves)."""
    tensors: dict[str, mx.array] = {}
    scalars: dict[str, Any] = {}
    for key, value in tree_flatten(state):
        if isinstance(value, mx.array):
            tensors[key] = value
        else:
            scalars[key] = value
    return tensors, scalars


def _deep_merge_preserving_placeholders(base: Any, override: Any) -> Any:
    """Merge ``override`` on top of ``base`` preserving empty-dict placeholders.

    Behaviour differs from ``mlx.utils.tree_merge`` in one critical way:
    ``tree_merge`` coerces ``{}`` / ``[]`` to ``None`` before merging (see
    ``mlx/python/mlx/utils.py::tree_merge``). That wipes the empty-dict
    placeholder that ``Optimizer.init`` seeds on frozen-branch subtrees,
    which is exactly the structure we need to preserve so that the
    subsequent ``tree_map(fn, gradients, params, state)`` call has a
    matching path.

    Rules:
    - Both dicts → recurse into the union of keys.
    - Both lists/tuples → recurse positionally.
    - ``override`` wins at any leaf where both exist.
    - If only one side has a key / position, that side is kept verbatim.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        return {
            k: _deep_merge_preserving_placeholders(
                base.get(k, _MISSING), override.get(k, _MISSING)
            )
            for k in base.keys() | override.keys()
        }
    if isinstance(base, (list, tuple)) and isinstance(override, (list, tuple)):
        n = max(len(base), len(override))
        merged_list = [
            _deep_merge_preserving_placeholders(
                base[i] if i < len(base) else _MISSING,
                override[i] if i < len(override) else _MISSING,
            )
            for i in range(n)
        ]
        return type(base)(merged_list)
    # One side missing or a leaf → override wins when present, else base.
    if override is _MISSING:
        return base
    return override


_MISSING: Any = object()


def _step_int(state: dict) -> int:
    s = state.get("step")
    if s is None:
        return 0
    if isinstance(s, mx.array):
        return int(s.item())
    return int(s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_optimizer_state(optimizer, path: str | Path) -> Path:
    """Persist `optimizer.state` to disk under ``path``.

    Returns the directory that was written. Idempotent — overwrites
    previous artifacts at the same path.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    tensors, scalars = _split_state(optimizer.state)
    step = _step_int(optimizer.state)

    mx.save_safetensors(
        str(out / _TENSOR_FILE),
        tensors,
        metadata={"format": "mlx", "step": str(step)},
    )

    meta = {
        "schema_version": _SCHEMA_VERSION,
        "step": step,
        "tensor_keys": sorted(tensors.keys()),
        "scalar_leaves": scalars,
    }
    with open(out / _META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)
    return out


def load_optimizer_state(
    optimizer,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict:
    """Restore `optimizer.state` from a directory produced by `save_optimizer_state`.

    The optimizer must already have `.init()` been called (i.e. its state
    tree has the right shape) — typically by passing the same model's
    `trainable_parameters()`.

    Parameters
    ----------
    strict : bool
        If True (default), the saved key set must exactly match the current
        optimizer's key set. If False, unknown keys in the snapshot are
        dropped silently and missing keys keep their freshly-initialised
        values — useful for partial resume after a trainable-head shape
        change.

    Returns
    -------
    dict
        ``{"step": int, "loaded_tensor_keys": int, "loaded_scalar_keys": int}``
    """
    src = Path(path)
    tensor_path = src / _TENSOR_FILE
    meta_path = src / _META_FILE
    if not tensor_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"optimizer snapshot incomplete under {src} — "
            f"need {_TENSOR_FILE} and {_META_FILE}"
        )

    flat_tensors = mx.load(str(tensor_path))  # dict[str, mx.array]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scalar_leaves: dict[str, Any] = meta.get("scalar_leaves", {}) or {}

    if meta.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported optimizer snapshot schema_version="
            f"{meta.get('schema_version')!r}; this loader supports "
            f"version {_SCHEMA_VERSION}"
        )

    incoming: dict[str, Any] = {}
    incoming.update(flat_tensors)
    incoming.update(scalar_leaves)

    current_keys = {k for k, _ in tree_flatten(optimizer.state)}
    incoming_keys = set(incoming.keys())

    if strict:
        missing = current_keys - incoming_keys
        extra = incoming_keys - current_keys
        if missing or extra:
            raise ValueError(
                "strict load mismatch — "
                f"missing in snapshot: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}; "
                f"extra in snapshot: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
            )
    else:
        incoming = {k: v for k, v in incoming.items() if k in current_keys}

    new_state = tree_unflatten(list(incoming.items()))
    # Deep-merge rather than shallow `dict.update`.  `Optimizer.init` seeds
    # `opt._state` with empty-dict placeholders (e.g. `{"model":
    # {"embed_tokens": {}}}`) on every subtree that has *no* trainable
    # leaf.  `tree_flatten` drops those placeholders at save time, so
    # `new_state` rebuilt from the saved flat keys also lacks them.  A
    # top-level `state.update(new_state)` would wholesale replace
    # `opt._state["model"]` with a sub-tree missing `"embed_tokens"`,
    # and then the first `apply_gradients` call raises `KeyError` because
    # the gradient tree returned by `nn.value_and_grad` *does* still
    # carry that placeholder (it mirrors `trainable_parameters()`).
    #
    # `_deep_merge_preserving_placeholders` walks both trees: where both
    # sides have a dict it recurses, where only one has a key that side
    # wins verbatim, and at a shared leaf the snapshot wins. Crucially,
    # empty-dict placeholders in `opt.state` survive because the walker
    # recognises them as dicts (unlike `mlx.utils.tree_merge`, which
    # coerces empty containers to `None`).
    merged = _deep_merge_preserving_placeholders(optimizer.state, new_state)
    # Keep `optimizer.state`'s dict identity — some callers hold references
    # to it (e.g. `mx.compile(..., inputs=state)`).
    optimizer.state.clear()
    optimizer.state.update(merged)

    return {
        "step": meta.get("step", 0),
        "loaded_tensor_keys": len(flat_tensors),
        "loaded_scalar_keys": len(scalar_leaves),
    }


def optimizer_state_summary(state: dict) -> dict:
    """Human-friendly snapshot of optimizer.state for logs and debugging."""
    tensors, scalars = _split_state(state)
    return {
        "num_tensor_leaves": len(tensors),
        "num_scalar_leaves": len(scalars),
        "tensor_total_params": int(sum(t.size for t in tensors.values())),
        "scalars": scalars,
    }
