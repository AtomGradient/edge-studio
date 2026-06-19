# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Pruning simulator — predict pruning impact without modifying weights."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activation_loader import ActivationProfile


@dataclass
class LayerPruneResult:
    """Pruning simulation result for a single layer."""
    layer_idx: int
    original_size: int
    alive_count: int          # neurons above threshold
    aligned_size: int         # after group_size alignment
    removed_count: int        # original_size - aligned_size
    retention_ratio: float    # aligned_size / original_size
    is_protected: bool = False

    @property
    def reduction_ratio(self) -> float:
        return 1.0 - self.retention_ratio


@dataclass
class PruneSimulationResult:
    """Complete pruning simulation result."""
    threshold: float
    group_size: int
    max_reduction: float
    protected_layers: list[int]
    layers: list[LayerPruneResult] = field(default_factory=list)

    # Model-level estimates
    original_intermediate_total: int = 0
    pruned_intermediate_total: int = 0
    original_mlp_params: int = 0
    pruned_mlp_params: int = 0
    original_mlp_size_bytes: int = 0
    pruned_mlp_size_bytes: int = 0

    @property
    def total_removed_neurons(self) -> int:
        return sum(l.removed_count for l in self.layers)

    @property
    def overall_retention(self) -> float:
        if self.original_intermediate_total == 0:
            return 1.0
        return self.pruned_intermediate_total / self.original_intermediate_total

    @property
    def mlp_size_reduction_bytes(self) -> int:
        return self.original_mlp_size_bytes - self.pruned_mlp_size_bytes

    @property
    def per_layer_sizes(self) -> list[int]:
        return [l.aligned_size for l in self.layers]


def simulate_pruning(
    profile: ActivationProfile,
    threshold: float = 0.1,
    group_size: int = 64,
    max_reduction: float = 0.5,
    min_size: int = 128,
    protected_layers: list[int] | None = None,
    hidden_size: int = 0,
    bits: int = 4,
) -> PruneSimulationResult:
    """Simulate neuron pruning and estimate impact.

    Mirrors the algorithm in prune_neurons.py / prune_gemma_neurons.py:
    1. Count neurons with max_activation >= threshold
    2. Align to group_size boundary
    3. Enforce max_reduction and min_size constraints
    4. Estimate parameter and size changes

    Args:
        profile: Activation profile data
        threshold: Minimum activation to keep a neuron
        group_size: Quantization group alignment (typically 64)
        max_reduction: Max fraction of neurons to remove per layer (0.5 = keep >= 50%)
        min_size: Minimum intermediate size per layer
        protected_layers: Layer indices to skip (keep original size)
        hidden_size: Model hidden_size for param estimation (0 = skip estimation)
        bits: Quantization bit width for size estimation
    """
    if protected_layers is None:
        protected_layers = []

    result = PruneSimulationResult(
        threshold=threshold,
        group_size=group_size,
        max_reduction=max_reduction,
        protected_layers=protected_layers,
    )

    original_total = 0
    pruned_total = 0

    for layer_act in profile.layers:
        idx = layer_act.layer_idx
        orig_size = layer_act.intermediate_size
        original_total += orig_size

        if idx in protected_layers:
            result.layers.append(LayerPruneResult(
                layer_idx=idx,
                original_size=orig_size,
                alive_count=orig_size,
                aligned_size=orig_size,
                removed_count=0,
                retention_ratio=1.0,
                is_protected=True,
            ))
            pruned_total += orig_size
            continue

        # Count alive neurons
        alive = int(np.sum(layer_act.max_activations >= threshold))

        # Align up to group_size
        aligned = _align_up(alive, group_size)

        # Enforce min_size
        aligned = max(aligned, min_size)

        # Enforce max_reduction: keep at least (1 - max_reduction) of original
        min_keep = _align_up(int(orig_size * (1.0 - max_reduction)), group_size)
        aligned = max(aligned, min_keep)

        # Cap at original size
        aligned = min(aligned, orig_size)

        removed = orig_size - aligned
        retention = aligned / orig_size if orig_size > 0 else 1.0

        result.layers.append(LayerPruneResult(
            layer_idx=idx,
            original_size=orig_size,
            alive_count=alive,
            aligned_size=aligned,
            removed_count=removed,
            retention_ratio=retention,
        ))
        pruned_total += aligned

    result.original_intermediate_total = original_total
    result.pruned_intermediate_total = pruned_total

    # Estimate MLP parameter impact
    # Each layer MLP has 3 weight matrices: gate_proj, up_proj [intermediate, hidden], down_proj [hidden, intermediate]
    # Total MLP params per layer = 3 * hidden_size * intermediate_size
    if hidden_size > 0:
        for lr in result.layers:
            orig_params = 3 * hidden_size * lr.original_size
            pruned_params = 3 * hidden_size * lr.aligned_size
            result.original_mlp_params += orig_params
            result.pruned_mlp_params += pruned_params

            # Size estimation: quantized = weight(U32) + scales(BF16) + biases(BF16)
            # For N-bit quant with group_size G: weight bytes = rows * cols * bits / 8
            # scales/biases bytes = rows * (cols / G) * 2 each
            orig_bytes = _estimate_quant_size(hidden_size, lr.original_size, bits, group_size)
            pruned_bytes = _estimate_quant_size(hidden_size, lr.aligned_size, bits, group_size)
            # 3 matrices per layer (gate, up, down — down is transposed but same param count)
            result.original_mlp_size_bytes += orig_bytes * 3
            result.pruned_mlp_size_bytes += pruned_bytes * 3

    return result


def simulate_threshold_sweep(
    profile: ActivationProfile,
    thresholds: list[float] | None = None,
    group_size: int = 64,
    max_reduction: float = 0.5,
    min_size: int = 128,
    protected_layers: list[int] | None = None,
    hidden_size: int = 0,
    bits: int = 4,
) -> list[PruneSimulationResult]:
    """Run pruning simulation across multiple thresholds."""
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    return [
        simulate_pruning(
            profile, t, group_size, max_reduction, min_size,
            protected_layers, hidden_size, bits,
        )
        for t in thresholds
    ]


def _align_up(value: int, alignment: int) -> int:
    """Round up to nearest multiple of alignment."""
    return ((value + alignment - 1) // alignment) * alignment


def _estimate_quant_size(rows: int, cols: int, bits: int, group_size: int) -> int:
    """Estimate bytes for a single quantized weight matrix."""
    # Packed weight: rows * ceil(cols * bits / 32) * 4 bytes (U32)
    weight_bytes = rows * ((cols * bits + 31) // 32) * 4
    # Scales and biases: rows * ceil(cols / group_size) * 2 bytes (BF16) each
    num_groups = (cols + group_size - 1) // group_size
    scales_bytes = rows * num_groups * 2
    biases_bytes = rows * num_groups * 2
    return weight_bytes + scales_bytes + biases_bytes
