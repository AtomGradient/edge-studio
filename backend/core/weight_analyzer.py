# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Weight statistics and distribution analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

from .weight_loader import (
    TensorMeta,
    WeightIndex,
    find_quant_group,
    is_quantized_weight,
    load_dequantized_tensor,
    load_tensor,
)


@dataclass
class TensorStats:
    """Statistics for a single tensor."""
    name: str
    shape: list[int]
    dtype: str
    num_elements: int
    size_bytes: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None
    sparsity: Optional[float] = None
    histogram_counts: Optional[list[int]] = None
    histogram_edges: Optional[list[float]] = None
    is_quantized: bool = False
    quant_group_size: Optional[int] = None
    quant_bits: Optional[int] = None

    @property
    def has_full_stats(self) -> bool:
        return self.min_val is not None


def metadata_stats(meta: TensorMeta, index: WeightIndex) -> TensorStats:
    """Compute lightweight stats from header metadata only (no tensor loading)."""
    return TensorStats(
        name=meta.name,
        shape=meta.shape,
        dtype=meta.dtype,
        num_elements=meta.num_elements,
        size_bytes=meta.size_bytes,
        is_quantized=is_quantized_weight(meta.name, index),
    )


def full_stats(
    meta: TensorMeta,
    index: WeightIndex,
    quant_config: dict | None = None,
    num_bins: int = 100,
) -> TensorStats:
    """Compute full statistics by loading the tensor via MLX.

    For quantized tensors, dequantizes first using mx.dequantize().
    """
    stats = metadata_stats(meta, index)

    # Determine if we should dequantize
    quant_group = find_quant_group(meta.name, index)
    if quant_group and quant_config and meta.name.endswith(".weight"):
        group_size = quant_config.get("group_size", 64)
        bits = quant_config.get("bits", 4)
        w, s, b = quant_group
        try:
            tensor = load_dequantized_tensor(w, s, b, group_size=group_size, bits=bits)
            stats.is_quantized = True
            stats.quant_group_size = group_size
            stats.quant_bits = bits
        except (TypeError, ValueError, KeyError):
            tensor = load_tensor(meta)
    else:
        tensor = load_tensor(meta)

    # Convert to float32 for statistics via MLX, then to numpy for histogram
    tensor_f32 = tensor.astype(mx.float32)
    mx.eval(tensor_f32)
    arr = np.array(tensor_f32).flatten()

    stats.min_val = float(np.min(arr))
    stats.max_val = float(np.max(arr))
    stats.mean_val = float(np.mean(arr))
    stats.std_val = float(np.std(arr))
    stats.sparsity = float(np.sum(arr == 0) / max(len(arr), 1))

    counts, edges = np.histogram(arr, bins=num_bins)
    stats.histogram_counts = counts.tolist()
    stats.histogram_edges = edges.tolist()

    return stats


def batch_metadata_stats(index: WeightIndex) -> list[TensorStats]:
    """Compute metadata-only stats for all tensors in the index."""
    return [metadata_stats(meta, index) for meta in index.tensors.values()]


def compute_dtype_summary(index: WeightIndex) -> dict[str, dict]:
    """Summarize tensors by dtype."""
    summary: dict[str, dict] = {}
    for meta in index.tensors.values():
        if meta.dtype not in summary:
            summary[meta.dtype] = {"count": 0, "params": 0, "size": 0}
        summary[meta.dtype]["count"] += 1
        summary[meta.dtype]["params"] += meta.num_elements
        summary[meta.dtype]["size"] += meta.size_bytes
    return summary


def compute_component_summary(
    index: WeightIndex, prefixes: list[str]
) -> list[dict]:
    """Summarize params/size grouped by component prefixes."""
    results = []
    for prefix in prefixes:
        tensors = index.tensors_with_prefix(prefix)
        params = sum(t.num_elements for t in tensors.values())
        size = sum(t.size_bytes for t in tensors.values())
        quant_count = sum(1 for t in tensors.values() if is_quantized_weight(t.name, index))
        results.append({
            "prefix": prefix,
            "tensor_count": len(tensors),
            "param_count": params,
            "size_bytes": size,
            "quantized_count": quant_count,
        })
    return results
