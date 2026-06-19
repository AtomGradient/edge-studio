# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Background task lifecycle for long-running operations."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from backend.services.error_mapper import TaskCancelledError, map_error

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _listeners: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _finished_at: float = 0.0
    _started_at: float = 0.0
    # Monotonic counter set when mark_running fires. Used as tiebreaker in
    # list_tasks() sort because time.time() granularity (~1µs on macOS) lets
    # back-to-back mark_running() calls share a timestamp and the sort
    # otherwise picks an unstable order. Also makes ordering deterministic
    # for tests.
    _start_seq: int = 0
    # Free-form metadata for the UI ("kind"="training", "peer_id"=…, etc).
    # Kept on the task itself so /api/tasks can render rich rows without
    # requiring callers to also stuff details into `message`.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict:
        """Serialize for /api/tasks list endpoint (excludes the result blob)."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "metadata": self.metadata,
            "started_at": self._started_at,
            "finished_at": self._finished_at if self._finished_at else None,
        }


_TASK_TTL_SECONDS = 7200  # 2 hours — completed/error tasks auto-removed


class TaskManager:
    """Manages background tasks and streams progress via async queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskState] = {}
        self._next_start_seq = 0
        # Persistent thread pool — MLX TLS destructors crash when threads exit,
        # so we keep worker threads alive across tasks.
        self._pool = ThreadPoolExecutor(max_workers=4)

    def _cleanup_stale_tasks(self) -> None:
        """Remove completed/error/cancelled tasks older than TTL."""
        now = time.time()
        stale = [
            tid for tid, task in self._tasks.items()
            if task.status in (TaskStatus.COMPLETE, TaskStatus.ERROR, TaskStatus.CANCELLED)
            and task._finished_at > 0
            and now - task._finished_at > _TASK_TTL_SECONDS
        ]
        for tid in stale:
            del self._tasks[tid]

    def create_task(self, metadata: dict[str, Any] | None = None) -> str:
        task_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._cleanup_stale_tasks()
            self._tasks[task_id] = TaskState(
                task_id=task_id,
                metadata=dict(metadata or {}),
            )
        return task_id

    def list_tasks(self, *, active_only: bool = False) -> list[dict]:
        """Snapshot of all known tasks (or only RUNNING/PENDING when active_only).
        Newest started first so the UI shows in-progress work at the top."""
        with self._lock:
            self._cleanup_stale_tasks()
            tasks = list(self._tasks.values())
        if active_only:
            tasks = [t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)]
        # Sort by (started_at, start_seq) desc — start_seq tiebreaker handles
        # back-to-back mark_running() calls that share a microsecond timestamp.
        tasks.sort(key=lambda t: (t._started_at or 0.0, t._start_seq), reverse=True)
        return [t.to_summary() for t in tasks]

    # --- Externally-driven task helpers ----------------------------------
    # Used by code paths that don't want to surrender control to
    # `run_in_thread` (e.g. the iOS-initiated training trigger, which is
    # already running in a scheduler-owned thread). Mirror the same
    # progress/complete/fail semantics so the UI sees identical events.

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.RUNNING
            task._started_at = time.time()
            self._next_start_seq += 1
            task._start_seq = self._next_start_seq

    def update_progress(self, task_id: str, message: str, percent: float) -> None:
        """Update task progress and notify subscribers. Raises TaskCancelledError
        if the task has been cancelled — callers should let it propagate so the
        worker unwinds cleanly."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if task.cancelled.is_set():
                raise TaskCancelledError("Operation cancelled by user.")
            task.progress = percent
            task.message = message
        self._notify(task_id, {"type": "progress", "message": message, "percent": percent})

    def complete(self, task_id: str, result: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.COMPLETE
            task.result = result
            task.progress = 1.0
            task._finished_at = time.time()
        self._notify(task_id, {"type": "complete", "result": "_stored"})

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.ERROR
            task.error = error
            task._finished_at = time.time()
        self._notify(task_id, {"type": "error", "message": error})

    def get_task(self, task_id: str) -> TaskState | None:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task. Returns True if the task was found."""
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.cancelled.set()
            task.status = TaskStatus.CANCELLED
            task._finished_at = time.time()
        self._notify(task_id, {"type": "cancelled", "message": "Operation cancelled by user."})
        return True

    def subscribe(self, task_id: str) -> asyncio.Queue | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            q: asyncio.Queue = asyncio.Queue()
            task._listeners.append(q)
            return q

    def _notify(self, task_id: str, event: dict) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            for q in task._listeners:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def run_in_thread(
        self,
        task_id: str,
        fn: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Run *fn* in a background thread. *fn* receives a progress_callback kwarg."""

        def _worker() -> None:
            task = self.get_task(task_id)
            if not task:
                return
            task.status = TaskStatus.RUNNING
            task._started_at = time.time()
            with self._lock:
                self._next_start_seq += 1
                task._start_seq = self._next_start_seq

            def progress_callback(message: str, percent: float) -> None:
                if task.cancelled.is_set():
                    raise TaskCancelledError("Operation cancelled by user.")
                task.progress = percent
                task.message = message
                self._notify(task_id, {
                    "type": "progress",
                    "message": message,
                    "percent": percent,
                })

            try:
                result = fn(*args, progress_callback=progress_callback, **kwargs)
                task.status = TaskStatus.COMPLETE
                task.result = result
                task.progress = 1.0
                task._finished_at = time.time()
                self._notify(task_id, {"type": "complete", "result": "_stored"})
            except TaskCancelledError:
                task.status = TaskStatus.CANCELLED
                task._finished_at = time.time()
                self._notify(task_id, {"type": "cancelled", "message": "Operation cancelled by user."})
            except Exception as exc:
                user_msg, debug_detail = map_error(exc)
                logger.error("Task %s failed: %s\n%s", task_id, debug_detail, traceback.format_exc())
                task.status = TaskStatus.ERROR
                task.error = user_msg
                task._finished_at = time.time()
                self._notify(task_id, {"type": "error", "message": user_msg})

        self._pool.submit(_worker)


# Singleton
task_manager = TaskManager()
