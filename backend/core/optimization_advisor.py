# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Smart optimization advisor — analyze model data and generate actionable suggestions.

Examines activation profiles, weight stats, inference traces, and quantization state
to produce prioritized optimization recommendations for edge deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .activation_loader import ActivationProfile
from .architecture import ModelArchitecture, format_param_count, format_size
from .device_profiles import DeviceProfile, DEVICE_PROFILES
from .pruning_detector import PruningTrace
from .pruning_simulator import simulate_pruning, PruneSimulationResult
from .weight_loader import WeightIndex


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptimizationSuggestion:
    """A single optimization recommendation."""
    category: str          # "neuron_pruning" | "layer_pruning" | "quantization" | "vocab_pruning" | "head_pruning"
    priority: str          # "high" | "medium" | "low"
    title: str
    description: str
    estimated_saving: str  # e.g. "~800MB" | "~2 layers"
    risk_level: str        # "low" | "medium" | "high"
    params: dict[str, Any] = field(default_factory=dict)
    applicable: bool = True


@dataclass
class OptimizationReport:
    """Complete optimization analysis report."""
    model_name: str
    model_size_bytes: int
    total_params: int
    suggestions: list[OptimizationSuggestion] = field(default_factory=list)
    device_recommendations: dict[str, list[OptimizationSuggestion]] = field(default_factory=dict)

    @property
    def high_priority(self) -> list[OptimizationSuggestion]:
        return [s for s in self.suggestions if s.priority == "high" and s.applicable]

    @property
    def total_estimated_saving_bytes(self) -> int:
        """Sum up estimated savings from all applicable suggestions."""
        total = 0
        for s in self.suggestions:
            if s.applicable and "saving_bytes" in s.params:
                total += s.params["saving_bytes"]
        return total


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _analyze_dead_neurons(
    profile: ActivationProfile,
    arch: ModelArchitecture,
) -> list[OptimizationSuggestion]:
    """Analyze activation profile for dead neuron pruning opportunities."""
    suggestions = []

    quant = arch.quantization or {}
    bits = quant.get("bits", 16)
    group_size = quant.get("group_size", 64)
    hidden_size = _get_hidden_size(arch)

    # Try multiple thresholds to find the sweet spot
    best_threshold = 0.1
    best_saving = 0

    for threshold in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        sim = simulate_pruning(
            profile,
            threshold=threshold,
            group_size=group_size,
            max_reduction=0.5,
            min_size=128,
            hidden_size=hidden_size,
            bits=bits,
        )
        saving = sim.mlp_size_reduction_bytes
        if saving > best_saving:
            best_saving = saving
            best_threshold = threshold

    # Run with best threshold
    sim = simulate_pruning(
        profile,
        threshold=best_threshold,
        group_size=group_size,
        max_reduction=0.5,
        min_size=128,
        hidden_size=hidden_size,
        bits=bits,
    )

    total_neurons = profile.num_layers * profile.intermediate_size
    dead_count = profile.total_dead_neurons(best_threshold)
    dead_ratio = dead_count / max(total_neurons, 1)

    if dead_ratio > 0.05:  # At least 5% dead neurons
        # Find layers with highest dead ratio
        dead_per_layer = profile.dead_neurons_per_layer(best_threshold)
        high_dead_layers = [
            i for i, d in enumerate(dead_per_layer)
            if d / profile.intermediate_size > 0.3
        ]

        layer_range = ""
        if high_dead_layers:
            layer_range = f"Layer {min(high_dead_layers)}-{max(high_dead_layers)} "

        priority = "high" if dead_ratio > 0.2 else "medium" if dead_ratio > 0.1 else "low"

        suggestions.append(OptimizationSuggestion(
            category="neuron_pruning",
            priority=priority,
            title=f"Neuron Pruning: {dead_ratio:.0%} dead neurons",
            description=(
                f"{layer_range}has {dead_ratio:.1%} dead neurons (threshold={best_threshold}). "
                f"Recommended threshold={best_threshold}, "
                f"can remove {sim.total_removed_neurons:,} neurons, "
                f"save ~{format_size(best_saving)}."
            ),
            estimated_saving=format_size(best_saving),
            risk_level="low" if dead_ratio > 0.3 else "medium",
            params={
                "threshold": best_threshold,
                "saving_bytes": best_saving,
                "removed_neurons": sim.total_removed_neurons,
                "per_layer_sizes": sim.per_layer_sizes,
                "retention": sim.overall_retention,
                "high_dead_layers": high_dead_layers,
            },
        ))

    return suggestions


def _analyze_layer_contributions(
    trace,  # InferenceTrace
) -> list[OptimizationSuggestion]:
    """Analyze inference trace for layer pruning opportunities."""
    suggestions = []

    if not trace or not trace.steps:
        return suggestions

    num_layers = trace.num_layers
    # Aggregate residual norms across all steps
    attn_norms = np.zeros(num_layers)
    mlp_norms = np.zeros(num_layers)
    counts = np.zeros(num_layers)

    for step in trace.steps:
        for lt in step.layers:
            idx = lt.layer_idx
            if idx < num_layers:
                attn_norms[idx] += lt.attn_residual_norm
                mlp_norms[idx] += lt.mlp_residual_norm
                counts[idx] += 1

    # Average norms
    mask = counts > 0
    attn_norms[mask] /= counts[mask]
    mlp_norms[mask] /= counts[mask]
    total_norms = attn_norms + mlp_norms

    # Find low-contribution layers
    if total_norms.max() > 0:
        norm_threshold = np.percentile(total_norms[total_norms > 0], 25)
        low_layers = [int(i) for i in range(num_layers) if 0 < total_norms[i] < norm_threshold]

        if len(low_layers) >= 2:
            # Find contiguous ranges for cleaner suggestion
            ranges = _find_contiguous_ranges(low_layers)
            range_str = ", ".join(
                f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0])
                for r in ranges
            )

            suggestions.append(OptimizationSuggestion(
                category="layer_pruning",
                priority="medium",
                title=f"Layer Pruning: {len(low_layers)} low-contribution layers",
                description=(
                    f"Layers [{range_str}] have residual norm contributions in the bottom 25%. "
                    f"MLP contribution particularly low — consider removing {len(low_layers)} layers. "
                    f"Estimated saving: ~{len(low_layers)} layers worth of parameters."
                ),
                estimated_saving=f"~{len(low_layers)} layers",
                risk_level="medium" if len(low_layers) <= 3 else "high",
                params={
                    "layers_to_remove": low_layers,
                    "avg_norms": {int(i): float(total_norms[i]) for i in low_layers},
                    "norm_threshold": float(norm_threshold),
                },
            ))

    return suggestions


def _analyze_attention_heads(
    trace,  # InferenceTrace
) -> list[OptimizationSuggestion]:
    """Analyze attention patterns for head pruning opportunities."""
    suggestions = []

    if not trace or not trace.steps:
        return suggestions

    num_heads = trace.num_heads
    num_layers = trace.num_layers

    # Track head importance: max attention entropy across steps/layers
    head_importance = np.zeros((num_layers, num_heads))
    head_counts = np.zeros((num_layers, num_heads))

    for step in trace.steps:
        for lt in step.layers:
            attn = lt.attn_weights  # [num_heads, seq_len]
            if attn is None or len(attn.shape) < 2:
                continue
            idx = lt.layer_idx
            if idx >= num_layers:
                continue
            nh = min(attn.shape[0], num_heads)
            for h in range(nh):
                row = attn[h]
                # Compute attention concentration (max attention weight)
                max_attn = float(np.max(row)) if len(row) > 0 else 0
                head_importance[idx, h] += max_attn
                head_counts[idx, h] += 1

    mask = head_counts > 0
    head_importance[mask] /= head_counts[mask]

    # Find consistently low-importance heads
    if head_importance.max() > 0:
        threshold = np.percentile(head_importance[head_importance > 0], 20)
        # Count heads that are low-importance across many layers
        low_heads_per_layer = []
        for layer_idx in range(num_layers):
            low = [int(h) for h in range(num_heads)
                   if 0 < head_importance[layer_idx, h] < threshold]
            low_heads_per_layer.append(low)

        # Find heads that are consistently unimportant
        total_low = sum(len(l) for l in low_heads_per_layer)
        if total_low > num_heads * num_layers * 0.1:
            # Find globally low-importance heads (low in most layers)
            head_low_count = np.zeros(num_heads)
            for layer_low in low_heads_per_layer:
                for h in layer_low:
                    head_low_count[h] += 1

            consistently_low = [
                int(h) for h in range(num_heads)
                if head_low_count[h] > num_layers * 0.5
            ]

            if consistently_low:
                suggestions.append(OptimizationSuggestion(
                    category="head_pruning",
                    priority="low",
                    title=f"Attention Head Analysis: {len(consistently_low)} potentially redundant heads",
                    description=(
                        f"Heads {consistently_low[:5]}{'...' if len(consistently_low) > 5 else ''} "
                        f"show low attention concentration across >50% of layers. "
                        f"Consider GQA or head pruning for further compression."
                    ),
                    estimated_saving="varies",
                    risk_level="high",
                    params={
                        "low_importance_heads": consistently_low,
                        "importance_threshold": float(threshold),
                    },
                ))

    return suggestions


def _analyze_quantization(
    arch: ModelArchitecture,
    weight_index: WeightIndex,
) -> list[OptimizationSuggestion]:
    """Analyze quantization state and recommend further compression."""
    suggestions = []

    quant = arch.quantization or {}
    current_bits = quant.get("bits", 0)
    model_size = arch.root.total_size_bytes
    total_params = arch.root.total_param_count

    if current_bits == 0:
        # Not quantized — recommend quantization
        est_4bit = total_params * 4 / 8  # rough estimate
        saving = model_size - est_4bit

        suggestions.append(OptimizationSuggestion(
            category="quantization",
            priority="high",
            title="Quantization: model is not quantized",
            description=(
                f"Current model is full precision ({format_size(model_size)}). "
                f"4-bit quantization could reduce to ~{format_size(int(est_4bit))}, "
                f"saving ~{format_size(int(saving))}."
            ),
            estimated_saving=format_size(int(saving)),
            risk_level="low",
            params={
                "current_bits": 0,
                "target_bits": 4,
                "saving_bytes": int(saving),
                "group_size": 64,
            },
        ))
    elif current_bits >= 6:
        # Can potentially reduce further
        target_bits = 4
        ratio = target_bits / current_bits
        est_saving = int(model_size * (1 - ratio))

        suggestions.append(OptimizationSuggestion(
            category="quantization",
            priority="medium",
            title=f"Quantization: {current_bits}-bit → {target_bits}-bit",
            description=(
                f"Current {current_bits}-bit quantization ({format_size(model_size)}). "
                f"Reducing to {target_bits}-bit could save ~{format_size(est_saving)}. "
                f"Expected accuracy loss is typically < 2% for well-calibrated quantization."
            ),
            estimated_saving=format_size(est_saving),
            risk_level="low" if current_bits >= 8 else "medium",
            params={
                "current_bits": current_bits,
                "target_bits": target_bits,
                "saving_bytes": est_saving,
                "group_size": quant.get("group_size", 64),
            },
        ))

    return suggestions


def _analyze_weight_sparsity(
    weight_index: WeightIndex,
) -> list[OptimizationSuggestion]:
    """Analyze weight sparsity patterns for structured pruning opportunities."""
    # This is a lightweight analysis based on metadata only
    # Full sparsity analysis requires loading tensors (expensive)
    suggestions = []

    total_tensors = weight_index.tensor_count
    if total_tensors == 0:
        return suggestions

    # Count gate/up/down proj tensors to estimate MLP weight count
    mlp_tensors = [
        name for name in weight_index.tensors
        if any(k in name for k in ["gate_proj", "up_proj", "down_proj", "mlp"])
    ]

    if mlp_tensors:
        suggestions.append(OptimizationSuggestion(
            category="neuron_pruning",
            priority="low",
            title="Weight Sparsity: run activation profiling for detailed analysis",
            description=(
                f"Model has {len(mlp_tensors)} MLP weight tensors across "
                f"{total_tensors} total tensors. "
                f"Generate an activation profile to identify dead neurons for pruning."
            ),
            estimated_saving="requires profiling",
            risk_level="low",
            applicable=False,  # Needs activation profile to be actionable
            params={"mlp_tensor_count": len(mlp_tensors)},
        ))

    return suggestions


def _analyze_device_fit(
    arch: ModelArchitecture,
    device: DeviceProfile,
    existing_suggestions: list[OptimizationSuggestion],
) -> list[OptimizationSuggestion]:
    """Check if model fits target device and recommend actions."""
    suggestions = []

    model_size_gb = arch.root.total_size_bytes / (1024 ** 3)
    max_size_gb = device.max_model_size_gb

    if model_size_gb > max_size_gb:
        gap_gb = model_size_gb - max_size_gb
        gap_bytes = int(gap_gb * 1024 ** 3)

        # Check if existing suggestions cover the gap
        total_available_saving = sum(
            s.params.get("saving_bytes", 0) for s in existing_suggestions if s.applicable
        )

        can_fit = total_available_saving >= gap_bytes

        suggestions.append(OptimizationSuggestion(
            category="device_fit",
            priority="high",
            title=f"Device Fit: {device.name} ({device.ram_gb}GB RAM)",
            description=(
                f"Model is {model_size_gb:.2f}GB, device limit is ~{max_size_gb:.1f}GB. "
                f"Need to reduce by ~{format_size(gap_bytes)}. "
                + (f"Available optimizations can save ~{format_size(total_available_saving)}, "
                   f"which {'should be sufficient' if can_fit else 'may not be enough'}."
                   if total_available_saving > 0 else
                   "Run activation profiling to identify optimization opportunities.")
            ),
            estimated_saving=f"need -{format_size(gap_bytes)}",
            risk_level="medium",
            params={
                "device_name": device.name,
                "model_size_gb": model_size_gb,
                "max_size_gb": max_size_gb,
                "gap_bytes": gap_bytes,
                "can_fit_with_optimizations": can_fit,
            },
        ))
    else:
        headroom_gb = max_size_gb - model_size_gb
        suggestions.append(OptimizationSuggestion(
            category="device_fit",
            priority="low",
            title=f"Device Fit: {device.name} — OK",
            description=(
                f"Model ({model_size_gb:.2f}GB) fits within device limit ({max_size_gb:.1f}GB). "
                f"Headroom: {headroom_gb:.2f}GB."
            ),
            estimated_saving="none needed",
            risk_level="low",
            params={
                "device_name": device.name,
                "model_size_gb": model_size_gb,
                "max_size_gb": max_size_gb,
                "headroom_gb": headroom_gb,
            },
        ))

    return suggestions


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def generate_report(
    arch: ModelArchitecture,
    weight_index: WeightIndex,
    activation_profile: Optional[ActivationProfile] = None,
    inference_trace=None,  # Optional[InferenceTrace]
    target_devices: Optional[list[str]] = None,
) -> OptimizationReport:
    """Generate a complete optimization report.

    Args:
        arch: Model architecture
        weight_index: Weight index for metadata
        activation_profile: Optional activation profile for neuron analysis
        inference_trace: Optional inference trace for layer/head analysis
        target_devices: Optional list of device names to check compatibility
    """
    report = OptimizationReport(
        model_name=arch.model_name,
        model_size_bytes=arch.root.total_size_bytes,
        total_params=arch.root.total_param_count,
    )

    # Quantization analysis (always available)
    report.suggestions.extend(_analyze_quantization(arch, weight_index))

    # Activation-based analysis
    if activation_profile:
        report.suggestions.extend(_analyze_dead_neurons(activation_profile, arch))
    else:
        report.suggestions.extend(_analyze_weight_sparsity(weight_index))

    # Inference trace analysis
    if inference_trace:
        report.suggestions.extend(_analyze_layer_contributions(inference_trace))
        report.suggestions.extend(_analyze_attention_heads(inference_trace))

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    report.suggestions.sort(key=lambda s: priority_order.get(s.priority, 3))

    # Device compatibility
    if target_devices:
        for device_name in target_devices:
            device = DEVICE_PROFILES.get(device_name)
            if device:
                device_suggestions = _analyze_device_fit(arch, device, report.suggestions)
                report.device_recommendations[device_name] = device_suggestions

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_hidden_size(arch: ModelArchitecture) -> int:
    """Extract hidden_size from architecture config."""
    cfg = arch.config
    for key in ["talker_config", "text_config"]:
        sub = cfg.get(key, {})
        if sub.get("hidden_size"):
            return sub["hidden_size"]
    return cfg.get("hidden_size", 0)


def _find_contiguous_ranges(indices: list[int]) -> list[list[int]]:
    """Group sorted indices into contiguous ranges."""
    if not indices:
        return []
    ranges: list[list[int]] = [[indices[0]]]
    for i in indices[1:]:
        if i == ranges[-1][-1] + 1:
            ranges[-1].append(i)
        else:
            ranges.append([i])
    return ranges
