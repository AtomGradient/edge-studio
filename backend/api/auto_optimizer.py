# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Auto optimizer search endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.analysis import AutoOptSearchRequest
from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["auto-optimizer"])


def _serialize_candidate(c) -> dict:
    """Serialize a SearchCandidate dataclass."""
    return {
        "threshold": c.threshold,
        "bits": c.target_bits,
        "layers_removed": c.layers_removed_count,
        "layers_removed_list": list(c.layers_removed),
        "estimated_size_gb": c.estimated_size_gb,
        "quality_proxy": c.quality_proxy,
        "neuron_retention": c.neuron_retention,
        "layer_retention": c.layer_retention,
        "fits_device": c.fits_device,
        "is_pareto": c.is_pareto,
        "per_layer_sizes": list(c.per_layer_sizes),
    }


@router.post("/{model_id}/auto-optimize/search", response_model=dict[str, Any])
def search_optimizations(model_id: str, req: AutoOptSearchRequest) -> dict[str, Any]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    profile = manager.get_profile(model_id)
    if not profile:
        raise HTTPException(400, "No activation profile loaded")

    trace = manager.get_trace(model_id)

    from backend.core.auto_optimizer import run_search_sweep

    result = run_search_sweep(
        arch=loaded.architecture,
        activation_profile=profile,
        device_name=req.device_name,
        quality_floor=req.quality_floor,
        target_bits_options=req.target_bits if req.target_bits else None,
        max_layers_to_remove=req.max_layers_remove,
        inference_trace=trace,
    )

    return {
        "candidates": [_serialize_candidate(c) for c in result.candidates],
        "pareto_frontier": [_serialize_candidate(c) for c in result.pareto_frontier],
        "device_name": result.device_name,
        "device_max_gb": result.device_max_gb,
        "model_name": result.model_name,
        "search_time_seconds": result.search_time_seconds,
        "total_combinations": result.total_combinations,
        "fits_device_count": result.fits_device_count,
    }
