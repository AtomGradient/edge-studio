# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Auto optimizer — search for optimal pruning + quantization combinations.

Given a target device and quality floor, sweeps through neuron pruning thresholds,
layer removal options, and quantization bit widths. Uses fast quality proxy scoring
(not actual PPL) for rapid exploration, with optional validation via real execution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .activation_loader import ActivationProfile
from .architecture import ModelArchitecture
from .device_profiles import DEVICE_PROFILES, DeviceProfile
from .pruning_simulator import simulate_pruning, _estimate_quant_size, _align_up


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchCandidate:
    """A single candidate in the optimization search space."""
    threshold: float
    layers_removed: list[int]
    target_bits: int
    estimated_size_bytes: int
    quality_proxy: float
    neuron_retention: float
    layer_retention: float
    fits_device: bool
    per_layer_sizes: list[int]
    is_pareto: bool = False

    @property
    def estimated_size_gb(self) -> float:
        return self.estimated_size_bytes / (1024 ** 3)

    @property
    def layers_removed_count(self) -> int:
        return len(self.layers_removed)


@dataclass
class SearchResult:
    """Complete search result with candidates and Pareto frontier."""
    candidates: list[SearchCandidate] = field(default_factory=list)
    pareto_frontier: list[SearchCandidate] = field(default_factory=list)
    device_name: str = ""
    device_max_gb: float = 0.0
    model_name: str = ""
    search_time_seconds: float = 0.0
    total_combinations: int = 0

    @property
    def fits_device_count(self) -> int:
        return sum(1 for c in self.candidates if c.fits_device)


@dataclass
class ValidationResult:
    """Result of validating a candidate with actual execution."""
    candidate: SearchCandidate
    actual_ppl: float = 0.0
    actual_size_bytes: int = 0
    success: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

def _estimate_total_size(
    arch: ModelArchitecture,
    per_layer_sizes: list[int],
    num_layers_remaining: int,
    target_bits: int,
    group_size: int = 64,
) -> int:
    """Estimate total model size after pruning and quantization.

    Accounts for embedding, per-layer attention + MLP, norms, and LM head.
    """
    cfg = arch.config
    hidden_size = cfg.get("hidden_size", 0)
    if not hidden_size:
        for sub_key in ["talker_config", "text_config"]:
            sub = cfg.get(sub_key, {})
            if sub.get("hidden_size"):
                hidden_size = sub["hidden_size"]
                break
    if hidden_size == 0:
        return 0

    vocab_size = cfg.get("vocab_size", 151936)
    num_heads = cfg.get("num_attention_heads", 32)
    num_kv_heads = cfg.get("num_key_value_heads", 8)
    head_dim = cfg.get("head_dim", 128)

    total = 0

    # Embedding: vocab_size * hidden_size * 2 bytes (bf16, typically not quantized)
    total += vocab_size * hidden_size * 2

    # Per remaining layer
    for i in range(num_layers_remaining):
        inter_size = per_layer_sizes[i] if i < len(per_layer_sizes) else per_layer_sizes[-1] if per_layer_sizes else 0

        # Attention projections: Q, K, V, O
        q_size = _estimate_quant_size(hidden_size, num_heads * head_dim, target_bits, group_size)
        k_size = _estimate_quant_size(hidden_size, num_kv_heads * head_dim, target_bits, group_size)
        v_size = _estimate_quant_size(hidden_size, num_kv_heads * head_dim, target_bits, group_size)
        o_size = _estimate_quant_size(num_heads * head_dim, hidden_size, target_bits, group_size)
        total += q_size + k_size + v_size + o_size

        # MLP: gate_proj, up_proj [hidden -> inter], down_proj [inter -> hidden]
        gate_size = _estimate_quant_size(hidden_size, inter_size, target_bits, group_size)
        up_size = _estimate_quant_size(hidden_size, inter_size, target_bits, group_size)
        down_size = _estimate_quant_size(inter_size, hidden_size, target_bits, group_size)
        total += gate_size + up_size + down_size

        # Norms: input_layernorm + post_attention_layernorm (hidden_size * 2 bytes each)
        total += hidden_size * 2 * 2

        # QK norm weights (Qwen3): 2 * head_dim * 2 bytes
        total += head_dim * 2 * 2

    # Final norm
    total += hidden_size * 2

    # LM head (often tied to embedding, but count it for safety)
    tie = cfg.get("tie_word_embeddings", True)
    if not tie:
        total += vocab_size * hidden_size * 2

    return total


# ---------------------------------------------------------------------------
# Layer pruning candidates
# ---------------------------------------------------------------------------

def _get_layer_removal_candidates(
    inference_trace,
    max_layers_to_remove: int,
) -> list[list[int]]:
    """Get candidate layer sets to remove, ranked by contribution.

    Uses residual norm from inference trace to find lowest-contribution layers.
    """
    candidates: list[list[int]] = [[]]  # Always include "remove nothing"

    if inference_trace is None or not inference_trace.steps or max_layers_to_remove == 0:
        return candidates

    num_layers = inference_trace.num_layers

    # Aggregate residual norms across all steps
    total_norms = np.zeros(num_layers)
    counts = np.zeros(num_layers)

    for step in inference_trace.steps:
        for lt in step.layers:
            idx = lt.layer_idx
            if idx < num_layers:
                total_norms[idx] += lt.attn_residual_norm + lt.mlp_residual_norm
                counts[idx] += 1

    mask = counts > 0
    total_norms[mask] /= counts[mask]

    # Sort layers by contribution (ascending = least important first)
    # Exclude first and last layers (usually critical)
    removable = list(range(1, num_layers - 1))
    removable.sort(key=lambda i: total_norms[i])

    # Build candidates: remove top-1, top-2, ..., top-N least important
    for n in range(1, min(max_layers_to_remove, len(removable)) + 1):
        candidates.append(sorted(removable[:n]))

    return candidates


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def run_search_sweep(
    arch: ModelArchitecture,
    activation_profile: ActivationProfile,
    device_name: str,
    quality_floor: float = 0.5,
    thresholds: list[float] | None = None,
    target_bits_options: list[int] | None = None,
    max_layers_to_remove: int = 0,
    inference_trace=None,
) -> SearchResult:
    """Sweep through optimization parameter space and find Pareto-optimal candidates.

    Args:
        arch: Model architecture
        activation_profile: Activation profile for neuron pruning simulation
        device_name: Target device name from DEVICE_PROFILES
        quality_floor: Minimum quality_proxy to consider (0.0-1.0)
        thresholds: Neuron pruning thresholds to try
        target_bits_options: Quantization bit widths to try
        max_layers_to_remove: Maximum number of layers to consider removing
        inference_trace: Optional inference trace for layer removal ranking

    Returns:
        SearchResult with all candidates and Pareto frontier
    """
    t0 = time.time()

    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    if target_bits_options is None:
        target_bits_options = [3, 4, 6, 8]

    device = DEVICE_PROFILES.get(device_name)
    if device is None:
        device = list(DEVICE_PROFILES.values())[0]
    device_max_bytes = int(device.max_model_size_gb * 1024 ** 3)

    # Current model info
    quant = arch.quantization or {}
    current_bits = quant.get("bits", 16)
    group_size = quant.get("group_size", 64)
    hidden_size = _get_hidden_size(arch)
    num_layers = activation_profile.num_layers

    # Layer removal candidates
    layer_candidates = _get_layer_removal_candidates(inference_trace, max_layers_to_remove)

    candidates = []
    total_combinations = 0

    for threshold in thresholds:
        # Simulate neuron pruning (instant)
        sim = simulate_pruning(
            activation_profile,
            threshold=threshold,
            group_size=group_size,
            max_reduction=0.5,
            min_size=128,
            hidden_size=hidden_size,
            bits=current_bits,
        )
        neuron_retention = sim.overall_retention

        for layers_to_remove in layer_candidates:
            layers_remaining = num_layers - len(layers_to_remove)
            layer_retention = layers_remaining / num_layers

            # Filter per_layer_sizes to only remaining layers
            remaining_sizes = [
                s for i, s in enumerate(sim.per_layer_sizes)
                if i not in set(layers_to_remove)
            ]

            for target_bits in target_bits_options:
                total_combinations += 1

                # Skip if target bits is higher than current (no point)
                if current_bits > 0 and target_bits > current_bits:
                    continue

                # Estimate total size
                est_size = _estimate_total_size(
                    arch, remaining_sizes, layers_remaining, target_bits, group_size,
                )
                if est_size == 0:
                    continue

                # Quality proxy
                bits_factor = min(target_bits / current_bits, 1.0) if current_bits > 0 else target_bits / 16.0
                quality_proxy = neuron_retention * layer_retention * bits_factor

                fits = est_size <= device_max_bytes

                candidates.append(SearchCandidate(
                    threshold=threshold,
                    layers_removed=layers_to_remove,
                    target_bits=target_bits,
                    estimated_size_bytes=est_size,
                    quality_proxy=quality_proxy,
                    neuron_retention=neuron_retention,
                    layer_retention=layer_retention,
                    fits_device=fits,
                    per_layer_sizes=remaining_sizes,
                ))

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(candidates)
    for c in pareto:
        c.is_pareto = True

    search_time = time.time() - t0

    return SearchResult(
        candidates=candidates,
        pareto_frontier=pareto,
        device_name=device.name,
        device_max_gb=device.max_model_size_gb,
        model_name=arch.model_name,
        search_time_seconds=search_time,
        total_combinations=total_combinations,
    )


def compute_pareto_frontier(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Compute Pareto frontier: minimize size, maximize quality.

    A candidate is Pareto-optimal if no other candidate has both
    smaller size AND higher quality.
    """
    if not candidates:
        return []

    # Sort by quality descending
    sorted_cands = sorted(candidates, key=lambda c: c.quality_proxy, reverse=True)

    frontier = []
    min_size_so_far = float("inf")

    for c in sorted_cands:
        if c.estimated_size_bytes < min_size_so_far:
            frontier.append(c)
            min_size_so_far = c.estimated_size_bytes

    return frontier


# ---------------------------------------------------------------------------
# Validation (optional, slow)
# ---------------------------------------------------------------------------

def validate_candidate(
    candidate: SearchCandidate,
    model_dir: str,
    activation_profile: ActivationProfile | None = None,
) -> ValidationResult:
    """Validate a candidate by actually executing the optimization pipeline.

    This is slow — it runs the actual pruning/quantization scripts.
    """
    from .optimization_executor import (
        PipelineStep,
        execute_pipeline,
    )

    steps = []

    # Neuron pruning (if threshold > 0 and retention < 1)
    if candidate.neuron_retention < 0.999:
        profile_path = activation_profile.source_file if activation_profile else None
        if profile_path:
            steps.append(PipelineStep(
                operation="neuron_pruning",
                params={
                    "threshold": candidate.threshold,
                    "profile_path": profile_path,
                    "max_reduction": 0.5,
                },
            ))

    # Layer pruning
    if candidate.layers_removed:
        steps.append(PipelineStep(
            operation="layer_pruning",
            params={"layers_to_remove": candidate.layers_removed},
        ))

    # Quantization (if changing bits)
    # Only add if we're actually re-quantizing
    steps.append(PipelineStep(
        operation="quantization",
        params={"bits": candidate.target_bits, "group_size": 64},
    ))

    if not steps:
        return ValidationResult(
            candidate=candidate,
            success=False,
            message="No optimization steps to execute",
        )

    try:
        result = execute_pipeline(model_dir, steps)
        actual_size = result.steps[-1].result_size_bytes if result.steps else 0

        return ValidationResult(
            candidate=candidate,
            actual_size_bytes=actual_size,
            success=result.all_success,
            message=f"Pipeline completed. Output: {result.final_output_dir}",
        )
    except Exception as e:
        return ValidationResult(
            candidate=candidate,
            success=False,
            message=f"Validation failed: {e}",
        )


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
