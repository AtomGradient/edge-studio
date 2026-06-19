# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for background task lifecycle management."""

from __future__ import annotations

import time

from backend.services.error_mapper import TaskCancelledError
from backend.services.task_manager import TaskManager, TaskStatus


def _wait_for_terminal_state(manager: TaskManager, task_id: str, timeout: float = 2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = manager.get_task(task_id)
        if task and task.status in (TaskStatus.COMPLETE, TaskStatus.ERROR, TaskStatus.CANCELLED):
            return task
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} did not finish")


def test_task_manager_progress_completion_and_subscription_events() -> None:
    manager = TaskManager()
    try:
        task_id = manager.create_task(metadata={"kind": "export"})
        queue = manager.subscribe(task_id)
        assert queue is not None

        manager.mark_running(task_id)
        manager.update_progress(task_id, "Halfway", 0.5)
        manager.complete(task_id, {"zip_path": "/tmp/app.zip"})

        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETE
        assert task.result == {"zip_path": "/tmp/app.zip"}
        assert task.to_summary()["metadata"] == {"kind": "export"}
        assert queue.get_nowait() == {"type": "progress", "message": "Halfway", "percent": 0.5}
        assert queue.get_nowait() == {"type": "complete", "result": "_stored"}
    finally:
        manager._pool.shutdown(wait=True)


def test_task_manager_cancel_causes_progress_to_raise_and_filters_active_tasks() -> None:
    manager = TaskManager()
    try:
        first = manager.create_task()
        second = manager.create_task()
        manager.mark_running(first)
        manager.mark_running(second)

        assert [item["task_id"] for item in manager.list_tasks(active_only=True)] == [second, first]
        assert manager.cancel_task(first) is True

        task = manager.get_task(first)
        assert task is not None
        assert task.status == TaskStatus.CANCELLED
        assert [item["task_id"] for item in manager.list_tasks(active_only=True)] == [second]
        try:
            manager.update_progress(first, "late", 0.9)
        except TaskCancelledError:
            pass
        else:
            raise AssertionError("cancelled task progress did not raise")
    finally:
        manager._pool.shutdown(wait=True)


def test_task_manager_run_in_thread_success_and_error_paths() -> None:
    manager = TaskManager()
    try:
        ok_id = manager.create_task()
        err_id = manager.create_task()

        def successful(*, progress_callback):
            progress_callback("Started", 0.2)
            return {"ok": True}

        def failing(*, progress_callback):
            progress_callback("Started", 0.2)
            raise RuntimeError("boom")

        manager.run_in_thread(ok_id, successful)
        manager.run_in_thread(err_id, failing)

        ok_task = _wait_for_terminal_state(manager, ok_id)
        err_task = _wait_for_terminal_state(manager, err_id)

        assert ok_task.status == TaskStatus.COMPLETE
        assert ok_task.result == {"ok": True}
        assert ok_task.progress == 1.0
        assert err_task.status == TaskStatus.ERROR
        assert err_task.error
    finally:
        manager._pool.shutdown(wait=True)
