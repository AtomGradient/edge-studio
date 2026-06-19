# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""In-memory pub/sub for mesh lifecycle events.

Purpose: let the Edge Studio web frontend (and any other local subscriber)
receive low-latency notifications about mesh state changes — peer paired /
revoked / deleted / connected / disconnected, and (later) tap-to-pair
requests from iOS — without polling every few seconds.

Design — thread-safe fan-out via per-subscriber `queue.Queue`:

- Anywhere in the backend can call `broadcast({"type": "...", ...})`.
- Each WebSocket connection gets its own bounded queue via `subscribe()`.
- Slow / dead consumers' queues may fill; we drop oldest (never block
  publishers) and keep running.
- Subscribers pull with `next_event()` which plays nicely with `asyncio`.

Event types are strings; the payload schema is owned by the caller. Typical:

    {"type": "peer_paired",       "peer_id": "...", "display_name": "..."}
    {"type": "peer_revoked",      "peer_id": "..."}
    {"type": "peer_deleted",      "peer_id": "..."}
    {"type": "peer_connected",    "peer_id": "..."}
    {"type": "peer_disconnected", "peer_id": "..."}
    {"type": "pair_request",      "peer_id": "...", "display_name": "...",
                                  "pin": "XXXXXX", "nonce": "...", "ttl_seconds": 60}
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

logger = logging.getLogger(__name__)


_QUEUE_MAX = 128  # per-subscriber cap — drops oldest events if frontend is sluggish


class MeshEventBus:
    """Thread-safe multi-subscriber broadcast channel for mesh events."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    # ---------- publish (called from sync or async code) ----------

    def broadcast(self, event: dict[str, Any]) -> None:
        """Fan out a single event to every current subscriber.

        Never blocks. If a subscriber's queue is full, drops the oldest event
        to make room — keeps the newest state reaching the UI faster than a
        backlog of stale events.
        """
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()  # evict one oldest
                    q.put_nowait(event)
                except (queue.Empty, queue.Full):
                    pass  # race — subscriber drained it; best effort

    # ---------- subscribe (WebSocket handler uses this) ----------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subs.remove(q)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    # ---------- async convenience (lets WS handler await next event) ----------

    async def next_event(self, q: queue.Queue, timeout: float = 30.0) -> dict[str, Any] | None:
        """Await the next event on a subscriber queue with a timeout.

        Runs the blocking `q.get(timeout=...)` in the default executor so the
        WebSocket's event loop stays responsive. Returns None on timeout —
        the caller can then send a keepalive frame and loop again.
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: q.get(timeout=timeout))
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_default_bus: MeshEventBus | None = None
_default_bus_lock = threading.Lock()


def get_default_bus() -> MeshEventBus:
    global _default_bus
    with _default_bus_lock:
        if _default_bus is None:
            _default_bus = MeshEventBus()
        return _default_bus
