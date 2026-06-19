# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Pruning simulation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.analysis import LayerPruneResult, PruneSimRequest, PruneSimResponse
from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["pruning"])


@router.post("/{model_id}/pruning/simulate", response_model=PruneSimResponse)
def simulate_pruning(model_id: str, req: PruneSimRequest) -> PruneSimResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    profile = manager.get_profile(model_id)
    if not profile:
        raise HTTPException(400, "No activation profile loaded — load one first")

    from backend.core.pruning_simulator import simulate_pruning as _simulate

    # Extract hidden_size and quant config for size estimation
    config = loaded.config
    hidden_size = config.get("hidden_size", 0)
    qcfg = config.get("quantization") or config.get("quantization_config") or {}
    bits = qcfg.get("bits", 4)
    group_size = qcfg.get("group_size", 64)

    result = _simulate(
        profile=profile,
        threshold=req.threshold,
        group_size=group_size,
        max_reduction=req.max_reduction,
        min_size=req.min_intermediate,
        protected_layers=req.protected_layers,
        hidden_size=hidden_size,
        bits=bits,
    )

    layers = []
    for lr in result.layers:
        layers.append(LayerPruneResult(
            layer_idx=lr.layer_idx,
            original_size=lr.original_size,
            alive_count=lr.alive_count,
            aligned_size=lr.aligned_size,
            removed=lr.removed_count,
            retention=lr.retention_ratio,
            is_protected=lr.is_protected,
        ))

    return PruneSimResponse(
        layers=layers,
        total_removed=result.total_removed_neurons,
        total_original=result.original_intermediate_total,
        retention=result.overall_retention,
        mlp_size_saved_bytes=result.mlp_size_reduction_bytes,
        mlp_params_saved=result.original_mlp_params - result.pruned_mlp_params,
        config_preview=result.per_layer_sizes,
    )


@router.post("/{model_id}/pruning/sweep", response_model=list[dict[str, Any]])
def threshold_sweep(model_id: str, req: PruneSimRequest) -> list[dict[str, Any]]:
    """Run pruning simulation across multiple thresholds for the sweep chart."""
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    profile = manager.get_profile(model_id)
    if not profile:
        raise HTTPException(400, "No activation profile loaded — load one first")

    from backend.core.pruning_simulator import simulate_threshold_sweep

    config = loaded.config
    hidden_size = config.get("hidden_size", 0)
    qcfg = config.get("quantization") or config.get("quantization_config") or {}
    bits = qcfg.get("bits", 4)
    group_size = qcfg.get("group_size", 64)

    results = simulate_threshold_sweep(
        profile=profile,
        group_size=group_size,
        max_reduction=req.max_reduction,
        min_size=req.min_intermediate,
        protected_layers=req.protected_layers,
        hidden_size=hidden_size,
        bits=bits,
    )

    return [
        {
            "threshold": r.threshold,
            "retention": r.overall_retention,
            "mlp_size_saved_mb": r.mlp_size_reduction_bytes / (1024 * 1024),
            "total_removed": r.total_removed_neurons,
        }
        for r in results
    ]


@router.post("/{model_id}/pruning/layer-sweep", response_model=list[dict[str, Any]])
def layer_importance_sweep(model_id: str) -> list[dict[str, Any]]:
    """Compute per-layer importance scores based on activation statistics.

    Returns a list of {layer_idx, importance, dead_ratio, mean_activation}
    sorted by layer index. Higher importance = more valuable layer.
    """
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    profile = manager.get_profile(model_id)
    if not profile:
        raise HTTPException(400, "No activation profile loaded — load one first")

    num_layers = len(profile.get("layers", []))
    results = []

    for idx, layer_data in enumerate(profile.get("layers", [])):
        max_activations = layer_data.get("max_activations", [])
        mean_activations = layer_data.get("mean_activations", [])

        if not max_activations:
            results.append({
                "layer_idx": idx,
                "importance": 0.5,
                "dead_ratio": 0.0,
                "mean_activation": 0.0,
            })
            continue

        # Dead neurons: max activation < 0.01
        dead_count = sum(1 for v in max_activations if v < 0.01)
        total = len(max_activations)
        dead_ratio = dead_count / total if total > 0 else 0.0

        mean_act = sum(mean_activations) / len(mean_activations) if mean_activations else 0.0

        # Importance: combines alive ratio + activation magnitude
        alive_ratio = 1.0 - dead_ratio
        importance = alive_ratio * 0.7 + min(mean_act / 10.0, 1.0) * 0.3

        results.append({
            "layer_idx": idx,
            "importance": round(importance, 4),
            "dead_ratio": round(dead_ratio, 4),
            "mean_activation": round(mean_act, 4),
        })

    return results
