# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model merge API routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.common import CreateTaskResponse
from backend.schemas.merge import MergeRequest
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["merge"])


@router.post("/merge", response_model=CreateTaskResponse)
def start_merge(req: MergeRequest):
    """Start a model merge task."""
    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.model_merge import MergeConfig, run_merge

        config = MergeConfig(
            model_dirs=req.model_dirs,
            strategy=req.strategy,
            weights=req.weights,
            base_model_dir=req.base_model_dir,
            density=req.density,
            output_dir=req.output_dir,
        )

        result = run_merge(config, progress_callback=progress_callback)

        return {
            "success": result.success,
            "output_dir": result.output_dir,
            "strategy": result.strategy,
            "model_names": result.model_names,
            "merged_params": result.merged_params,
            "merged_size_bytes": result.merged_size_bytes,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
