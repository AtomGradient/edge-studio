# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Architecture tree and pruning trace endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.model import ArchNodeSchema, PruningTraceSchema
from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["architecture"])


def _node_to_schema(node) -> ArchNodeSchema:
    """Recursively convert ArchNode to Pydantic schema."""
    return ArchNodeSchema(
        name=node.name,
        node_type=node.node_type,
        weight_prefix=node.weight_prefix,
        config_params=node.config_params,
        param_count=node.param_count,
        stored_param_count=node.stored_param_count,
        size_bytes=node.size_bytes,
        children=[_node_to_schema(c) for c in node.children],
        pruning_info=node.pruning_info,
        extra=node.extra,
        total_param_count=node.total_param_count,
        total_stored_param_count=node.total_stored_param_count,
        total_size_bytes=node.total_size_bytes,
        is_quantized=node.is_quantized,
    )


@router.get("/{model_id}/architecture", response_model=ArchNodeSchema)
def get_architecture(model_id: str) -> ArchNodeSchema:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")
    return _node_to_schema(loaded.architecture.root)


@router.get("/{model_id}/pruning-traces", response_model=list[PruningTraceSchema])
def get_pruning_traces(model_id: str) -> list[PruningTraceSchema]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")
    return [
        PruningTraceSchema(
            category=t.category,
            description=t.description,
            details=t.details,
            severity=t.severity,
        )
        for t in loaded.pruning_traces
    ]
