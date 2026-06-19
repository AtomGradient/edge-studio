# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Shared serialization utilities for numpy/dataclass -> JSON-safe dicts.

Replaces the per-file _serialize / _convert helpers that were duplicated
across backend/api/*.py modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert numpy/dataclass objects to JSON-serializable form.

    Handles:
    - numpy arrays  -> list
    - numpy scalars -> Python int/float
    - dataclasses   -> dict (recursive)
    - dict / list / tuple -> recursive descent
    - everything else -> passthrough
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: serialize_for_json(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]
    return obj


def serialize_execution_result(r: Any) -> dict:
    """Serialize an ExecutionResult dataclass to JSON-safe dict.

    Shared by pipeline.py and optimization.py.
    """
    return {
        "operation": r.operation,
        "success": r.success,
        "output_dir": r.output_dir,
        "message": r.message,
        "duration_seconds": r.duration_seconds,
        "original_size_bytes": r.original_size_bytes,
        "result_size_bytes": r.result_size_bytes,
        "saving_bytes": r.saving_bytes,
        "details": r.details if hasattr(r, "details") else {},
    }
