# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""MOE expert analysis endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["moe"])


@router.post("/{model_id}/moe/analyze", response_model=dict[str, Any])
def analyze_moe(model_id: str) -> dict[str, Any]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    trace = manager.get_trace(model_id)
    if not trace:
        raise HTTPException(400, "No inference trace — run tracer first")

    from backend.core.moe_analyzer import (
        detect_moe_config,
        analyze_expert_utilization,
        simulate_expert_pruning,
    )

    moe_config = detect_moe_config(loaded.model_dir)
    if moe_config is None:
        raise HTTPException(400, "This model is not a Mixture-of-Experts model")

    num_experts = moe_config["num_experts"]
    num_experts_per_tok = moe_config["num_experts_per_tok"]

    # Get expert traces from the inference trace
    expert_traces = getattr(trace, '_expert_traces', None)
    if expert_traces is None:
        raise HTTPException(400, "No expert routing data — run inference tracer on this MOE model first")

    util = analyze_expert_utilization(expert_traces, num_experts, num_experts_per_tok)

    # Build utilization matrix
    utilization_matrix = []
    layer_stats = []
    for ls in util.layer_stats:
        utilization_matrix.append(ls.token_counts.tolist() if isinstance(ls.token_counts, np.ndarray) else list(ls.token_counts))
        layer_stats.append({
            "layer_idx": ls.layer_idx,
            "expert_counts": ls.token_counts.tolist() if isinstance(ls.token_counts, np.ndarray) else list(ls.token_counts),
            "expert_avg_scores": ls.avg_scores.tolist() if isinstance(ls.avg_scores, np.ndarray) else list(ls.avg_scores),
            "load_balance": float(ls.load_balance_score),
            "cold_experts": ls.cold_experts,
        })

    # Cold expert details
    cold_experts = []
    for ls in util.layer_stats:
        for eidx in ls.cold_experts:
            cold_experts.append({"layer": ls.layer_idx, "expert": eidx})

    return {
        "num_experts": util.num_experts,
        "top_k": util.num_experts_per_tok,
        "avg_load_balance": float(util.overall_balance),
        "total_tokens": util.total_tokens,
        "cold_expert_count": util.cold_expert_count,
        "layer_stats": layer_stats,
        "cold_experts": cold_experts,
        "utilization_matrix": utilization_matrix,
        "global_token_counts": util.global_token_counts.tolist() if isinstance(util.global_token_counts, np.ndarray) else list(util.global_token_counts),
        "global_avg_scores": util.global_avg_scores.tolist() if isinstance(util.global_avg_scores, np.ndarray) else list(util.global_avg_scores),
    }
