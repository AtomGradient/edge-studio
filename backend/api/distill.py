# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Knowledge distillation API routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.common import CreateTaskResponse
from backend.schemas.distillation import DistillRequest
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["distillation"])


@router.post("/distill", response_model=CreateTaskResponse)
def start_distillation(req: DistillRequest):
    """Start a knowledge distillation task (teacher → student)."""
    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.distillation import DistillConfig, run_distillation

        config = DistillConfig(
            teacher_dir=req.teacher_dir,
            student_dir=req.student_dir,
            dataset_path=req.dataset_path,
            mode=req.mode,
            num_epochs=req.num_epochs,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            temperature=req.temperature,
            alpha=req.alpha,
            max_samples=req.max_samples,
            output_dir=req.output_dir,
        )

        result = run_distillation(config, progress_callback=progress_callback)

        return {
            "success": result.success,
            "output_dir": result.output_dir,
            "teacher_name": result.teacher_name,
            "student_name": result.student_name,
            "num_epochs": result.num_epochs,
            "total_steps": result.total_steps,
            "final_loss": result.final_loss,
            "final_kl_loss": result.final_kl_loss,
            "final_ce_loss": result.final_ce_loss,
            "duration_seconds": result.duration_seconds,
            "dataset_samples": result.dataset_samples,
            "error": result.error,
            "warning": result.warning,
            "loss_history": result.loss_history,
        }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
