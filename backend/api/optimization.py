# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Optimization advisor and execution endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.analysis import ExecuteOptRequest
from backend.schemas.common import CreateTaskResponse
from backend.services.model_manager import manager
from backend.services.serialization import serialize_execution_result
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["optimization"])


def _serialize_suggestion(s) -> dict:
    """Serialize an OptimizationSuggestion dataclass."""
    return {
        "category": s.category,
        "priority": s.priority,
        "title": s.title,
        "description": s.description,
        "estimated_saving": s.estimated_saving,
        "risk_level": s.risk_level,
        "params": s.params if hasattr(s, "params") else {},
        "applicable": s.applicable if hasattr(s, "applicable") else True,
    }


@router.post("/{model_id}/optimize/suggestions", response_model=dict[str, Any])
def get_suggestions(model_id: str) -> dict[str, Any]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.optimization_advisor import generate_report

    profile = manager.get_profile(model_id)
    trace = manager.get_trace(model_id)

    report = generate_report(
        arch=loaded.architecture,
        weight_index=loaded.weight_index,
        activation_profile=profile,
        inference_trace=trace,
    )

    applicable = [s for s in report.suggestions if s.applicable]
    requires_data = [s for s in report.suggestions if not s.applicable]

    return {
        "model_name": report.model_name,
        "model_size_bytes": report.model_size_bytes,
        "total_params": report.total_params,
        "suggestions": [_serialize_suggestion(s) for s in applicable],
        "requires_data": [_serialize_suggestion(s) for s in requires_data],
        "total_estimated_saving_bytes": report.total_estimated_saving_bytes,
    }


@router.post("/{model_id}/optimize/execute", response_model=CreateTaskResponse)
def execute_optimization(model_id: str, req: ExecuteOptRequest):
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.optimization_executor import execute_neuron_pruning, execute_layer_pruning, execute_vocab_pruning, execute_quantization, execute_embedding_quantization

        category = req.category
        params = req.params or {}
        model_dir = loaded.model_dir

        if category == "neuron_pruning":
            result = execute_neuron_pruning(
                model_dir=model_dir,
                threshold=params.get("threshold", 0.1),
                protected_layers=params.get("protected_layers"),
                max_reduction=params.get("max_reduction", 0.5),
                progress_cb=progress_callback,
            )
        elif category == "layer_pruning":
            result = execute_layer_pruning(
                model_dir=model_dir,
                layers_to_remove=params.get("layers_to_remove", []),
                progress_cb=progress_callback,
            )
        elif category == "vocab_pruning":
            result = execute_vocab_pruning(
                model_dir=model_dir,
                progress_cb=progress_callback,
            )
        elif category == "quantization":
            result = execute_quantization(
                model_dir=model_dir,
                bits=params.get("bits", 4),
                group_size=params.get("group_size", 64),
                progress_cb=progress_callback,
            )
        elif category == "embedding_quantization":
            result = execute_embedding_quantization(
                model_dir=model_dir,
                progress_cb=progress_callback,
            )
        else:
            raise ValueError(f"Unknown optimization category: {category}")

        return serialize_execution_result(result)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
