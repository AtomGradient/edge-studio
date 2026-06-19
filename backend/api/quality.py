# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Quality validation endpoints."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException

from backend.schemas.common import CreateTaskResponse
from backend.schemas.inference import GenerateRequest, PPLRequest
from backend.services.model_manager import manager
from backend.services.mlx_runtime_gate import mlx_runtime_gate
from backend.services.serialization import serialize_for_json
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["quality"])
ProgressCallback = Callable[[str, float], None]
QualityTask = Callable[[ProgressCallback | None], Any]


def _run_quality_task_with_gate(
    owner: str,
    task: QualityTask,
    progress_callback: ProgressCallback | None = None,
) -> Any:
    """Run a quality validation task under the shared MLX runtime gate."""
    with mlx_runtime_gate(owner):
        return task(progress_callback)


@router.get("/{model_id}/quality/cached", response_model=dict[str, Any])
def get_cached_quality(model_id: str) -> dict[str, Any]:
    """Return all cached quality results for this model (if any)."""
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")
    return manager.get_all_quality(model_id)


@router.post("/{model_id}/quality/ppl", response_model=CreateTaskResponse)
def compute_ppl(model_id: str, req: PPLRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        def _task(progress_callback=None):
            from backend.core.quality_validator import compute_perplexity

            def _progress(msg: str, pct: float):
                if progress_callback:
                    progress_callback(msg, pct)

            result = compute_perplexity(loaded.model_dir, req.text, progress_callback=_progress)
            serialized = serialize_for_json(result)
            manager.store_quality(model_id, "ppl", serialized)
            return serialized

        return _run_quality_task_with_gate("quality.ppl", _task, progress_callback)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.post("/{model_id}/quality/generate", response_model=CreateTaskResponse)
def run_generation(model_id: str, req: GenerateRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        def _task(progress_callback=None):
            from backend.core.quality_validator import benchmark_generation

            def _progress(msg: str, pct: float):
                if progress_callback:
                    progress_callback(msg, pct)

            samples = benchmark_generation(
                loaded.model_dir,
                prompts=req.prompts,
                max_tokens=req.max_tokens,
                enable_thinking=req.enable_thinking,
                progress_callback=_progress,
            )
            serialized = serialize_for_json(samples)
            manager.store_quality(model_id, "generation", serialized)
            return serialized

        return _run_quality_task_with_gate("quality.generate", _task, progress_callback)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.post("/{model_id}/quality/report", response_model=CreateTaskResponse)
def run_full_report(model_id: str, req: GenerateRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        def _task(progress_callback=None):
            from backend.core.quality_validator import run_quality_report

            def _progress(msg: str, pct: float):
                if progress_callback:
                    progress_callback(msg, pct)

            report = run_quality_report(
                loaded.model_dir,
                max_tokens=req.max_tokens,
                enable_thinking=req.enable_thinking,
                progress_callback=_progress,
            )
            serialized = serialize_for_json(report)
            manager.store_quality(model_id, "report", serialized)
            return serialized

        return _run_quality_task_with_gate("quality.report", _task, progress_callback)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
