# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Common response schemas shared across API endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class CreateTaskResponse(BaseModel):
    """Returned by all async task endpoints."""

    task_id: str


class TaskStatusResponse(BaseModel):
    """Returned by GET /api/task/{task_id}."""

    task_id: str
    status: Literal["pending", "running", "complete", "error", "cancelled"]
    progress: float
    message: str
    error: str | None = None
    result: Any = None


class TaskResultResponse(BaseModel):
    """Returned by GET /api/task/{task_id}/result."""

    result: Any


class TaskCancelResponse(BaseModel):
    """Returned by DELETE /api/task/{task_id}."""

    status: str
