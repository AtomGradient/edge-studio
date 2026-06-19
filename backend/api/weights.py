# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Weight statistics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.schemas.model import (
    DtypeBreakdownResponse,
    DtypeSummary,
    TensorMetaSchema,
    TensorStatsSchema,
    WeightStatsResponse,
)
from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["weights"])


@router.get("/{model_id}/weight-stats", response_model=WeightStatsResponse)
def get_weight_stats(model_id: str) -> WeightStatsResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.weight_loader import is_quantized_weight

    wi = loaded.weight_index
    tensors = []
    quant_count = 0
    for name, meta in wi.tensors.items():
        is_q = is_quantized_weight(name, wi)
        if is_q:
            quant_count += 1
        tensors.append(TensorMetaSchema(
            name=meta.name,
            dtype=meta.dtype,
            shape=meta.shape,
            num_elements=meta.num_elements,
            size_bytes=meta.size_bytes,
            is_quantized=is_q,
            file_path=meta.file_path,
        ))

    # Sort by size descending
    tensors.sort(key=lambda t: t.size_bytes, reverse=True)

    return WeightStatsResponse(
        tensors=tensors,
        total_params=sum(t.num_elements for t in tensors),
        total_size=wi.total_size_bytes,
        quantized_count=quant_count,
    )


@router.get("/{model_id}/weight-stats/dtype", response_model=DtypeBreakdownResponse)
def get_dtype_breakdown(model_id: str) -> DtypeBreakdownResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.weight_analyzer import compute_dtype_summary

    summary = compute_dtype_summary(loaded.weight_index)
    breakdown = [
        DtypeSummary(dtype=dt, count=v["count"], params=v["params"], size=v["size"])
        for dt, v in summary.items()
    ]
    return DtypeBreakdownResponse(breakdown=breakdown)


class FullStatsRequest(BaseModel):
    tensor_name: str


@router.post("/{model_id}/tensor/full-stats", response_model=TensorStatsSchema)
def get_tensor_full_stats(model_id: str, req: FullStatsRequest) -> TensorStatsSchema:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    # GGUF models don't support per-tensor MLX loading
    if loaded.config.get("_source_format") == "gguf":
        raise HTTPException(400, "Full tensor stats not available for GGUF models (quantized binary format)")

    meta = loaded.weight_index.tensors.get(req.tensor_name)
    if not meta:
        raise HTTPException(404, f"Tensor not found: {req.tensor_name}")

    from backend.core.weight_analyzer import full_stats

    qcfg = loaded.config.get("quantization") or loaded.config.get("quantization_config")
    stats = full_stats(meta, loaded.weight_index, qcfg)

    return TensorStatsSchema(
        name=stats.name,
        shape=stats.shape,
        dtype=stats.dtype,
        num_elements=stats.num_elements,
        size_bytes=stats.size_bytes,
        min_val=stats.min_val,
        max_val=stats.max_val,
        mean_val=stats.mean_val,
        std_val=stats.std_val,
        sparsity=stats.sparsity,
        histogram_counts=stats.histogram_counts,
        histogram_edges=stats.histogram_edges,
        is_quantized=stats.is_quantized,
        quant_group_size=stats.quant_group_size,
        quant_bits=stats.quant_bits,
    )
