# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""WebSocket endpoint for task progress streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend.schemas.common import TaskCancelResponse, TaskResultResponse, TaskStatusResponse
from backend.services.serialization import serialize_for_json
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/task/{task_id}")
async def task_progress(websocket: WebSocket, task_id: str):
    task = task_manager.get_task(task_id)
    if not task:
        await websocket.close(code=4004, reason="Task not found")
        return

    await websocket.accept()

    # If task already finished before WebSocket connected, send final state immediately
    if task.status.value in ("complete", "error", "cancelled"):
        if task.status.value == "complete":
            await websocket.send_text(json.dumps({"type": "complete", "result": "_stored"}))
        elif task.status.value == "error":
            await websocket.send_text(json.dumps({"type": "error", "message": task.error or "Unknown error"}))
        elif task.status.value == "cancelled":
            await websocket.send_text(json.dumps({"type": "cancelled", "message": "Operation cancelled"}))
        return

    queue = task_manager.subscribe(task_id)
    if not queue:
        await websocket.close(code=4004, reason="Task not found")
        return

    # Send current progress if task is already partially done
    if task.progress > 0:
        await websocket.send_text(json.dumps({
            "type": "progress", "message": task.message, "percent": task.progress,
        }))

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_text(json.dumps(event))
                if event.get("type") in ("complete", "error", "cancelled"):
                    break
            except asyncio.TimeoutError:
                # Check if task finished while we were waiting (missed notification)
                task = task_manager.get_task(task_id)
                if task and task.status.value in ("complete", "error", "cancelled"):
                    if task.status.value == "complete":
                        await websocket.send_text(json.dumps({"type": "complete", "result": "_stored"}))
                    elif task.status.value == "error":
                        await websocket.send_text(json.dumps({"type": "error", "message": task.error or "Unknown error"}))
                    break
                # Send keepalive ping
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Task WebSocket error for task %s", task_id)


@router.delete("/api/task/{task_id}", response_model=TaskCancelResponse)
def cancel_task(task_id: str):
    """Cancel a running task."""
    found = task_manager.cancel_task(task_id)
    if not found:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskCancelResponse(status="cancelled")


@router.get("/api/task/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Poll task status (alternative to WebSocket)."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return TaskStatusResponse(
        task_id=task.task_id, status=task.status.value,
        progress=task.progress, message=task.message, error=task.error,
        result=task.result if task.status.value == "complete" else None,
    )


@router.get("/api/task/{task_id}/result", response_model=TaskResultResponse)
def get_task_result(task_id: str):
    """Get task result after completion."""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status.value != "complete":
        raise HTTPException(400, f"Task not complete: {task.status.value}")

    return TaskResultResponse(result=serialize_for_json(task.result))


@router.get("/api/tasks")
def list_tasks(active: bool = False) -> dict:
    """List all known tasks (or only RUNNING/PENDING when active=true).

    Lets the web UI render a global "active jobs" panel — particularly important
    for iOS-initiated training, which the user otherwise can't observe at all.
    Pass `?active=true` to skip completed/error/cancelled tasks.
    """
    return {"tasks": task_manager.list_tasks(active_only=active)}
