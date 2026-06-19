# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Attention pattern analysis endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["attention"])


@router.post("/{model_id}/attention/analyze", response_model=dict[str, Any])
def analyze_attention(model_id: str) -> dict[str, Any]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    trace = manager.get_trace(model_id)
    if not trace:
        raise HTTPException(400, "No inference trace — run tracer with capture_attention=True first")

    from backend.core.attention_analyzer import classify_attention_heads

    result = classify_attention_heads(trace)

    # Serialize the dataclass result
    num_layers = trace.num_layers
    num_heads = trace.num_heads

    pattern_matrix = result.pattern_matrix(num_layers, num_heads)
    confidence_matrix = result.confidence_matrix(num_layers, num_heads)

    classifications = []
    for c in result.classifications:
        classifications.append({
            "layer": c.layer_idx,
            "head": c.head_idx,
            "pattern": c.pattern.value,
            "confidence": float(c.confidence),
        })

    # Build per-layer pattern counts
    pattern_names = ["sink", "local", "global", "sparse"]
    per_layer_summary = []
    for l in range(num_layers):
        counts = {p: 0 for p in pattern_names}
        for c in result.classifications:
            if c.layer_idx == l:
                counts[c.pattern.value] += 1
        dominant = max(counts, key=counts.get)
        per_layer_summary.append({"layer": l, "dominant": dominant, **counts})

    # Pattern matrix as list of lists of strings
    pattern_int_to_name = {0: "SINK", 1: "LOCAL", 2: "GLOBAL", 3: "SPARSE"}
    pattern_matrix_strs = []
    for l in range(num_layers):
        row = []
        for h in range(num_heads):
            row.append(pattern_int_to_name.get(int(pattern_matrix[l, h]), "UNKNOWN"))
        pattern_matrix_strs.append(row)

    return {
        "classifications": classifications,
        "pattern_matrix": pattern_matrix_strs,
        "pattern_counts": result.summary,
        "per_layer_summary": per_layer_summary,
        "suggestions": result.suggestions,
    }
