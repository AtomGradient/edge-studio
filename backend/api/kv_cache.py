# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""KV Cache analysis endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from backend.schemas.analysis import KVReportRequest
from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["kv-cache"])


@router.post("/{model_id}/kv-report", response_model=dict[str, Any])
def get_kv_report(model_id: str, req: KVReportRequest) -> dict[str, Any]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.kv_cache_analyzer import generate_kv_report

    trace = manager.get_trace(model_id)

    report = generate_kv_report(
        arch=loaded.architecture,
        inference_trace=trace,
        target_devices=req.devices if req.devices else None,
    )

    # Serialize the dataclass result
    kv = report.kv_config

    memory_curve = []
    for i, seq_len in enumerate(report.seq_lengths):
        b = report.memory_breakdowns[i]
        memory_curve.append({
            "seq_len": seq_len,
            "model_weights_mb": report.model_weights_bytes / 1e6,
            "kv_cache_mb": b.kv_cache_bytes / 1e6,
            "activation_mb": b.activation_estimate_bytes / 1e6,
            "overhead_mb": b.system_overhead_bytes / 1e6,
            "total_mb": b.total_bytes / 1e6,
        })

    device_capacities = []
    for cap in report.device_capacities:
        device_capacities.append({
            "device_name": cap.device_name,
            "ram_gb": cap.device_ram_gb,
            "available_mb": cap.available_ram_bytes / 1e6,
            "fits": cap.fits,
            "max_seq_len": cap.max_seq_len,
            "kv_at_max_mb": cap.kv_cache_at_max_bytes / 1e6,
            "headroom_mb": cap.headroom_bytes / 1e6,
        })

    trace_steps = report.trace_steps or []

    # DSR retention curves.
    dsr_curves = {}
    for label, curve in report.dsr_curves.items():
        dsr_curves[label] = [
            {
                "seq_len": report.seq_lengths[i],
                "kv_cache_mb": b.kv_cache_bytes / 1e6,
                "total_mb": b.total_bytes / 1e6,
            }
            for i, b in enumerate(curve)
        ]

    return {
        "num_layers": kv.num_layers,
        "num_kv_heads": kv.num_kv_heads,
        "head_dim": kv.head_dim,
        "bytes_per_token": kv.bytes_per_token,
        "model_weights_mb": report.model_weights_bytes / 1e6,
        "memory_curve": memory_curve,
        "device_capacities": device_capacities,
        "trace_steps": trace_steps,
        "dsr_curves": dsr_curves,
    }
