# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Process-wide gate for MLX runtime sections.

MLX/Metal execution in this backend can be reached through multiple service
paths. A single process-local gate prevents independent per-service locks from
running MLX load or generation concurrently in the same backend process.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Any, Iterator


_MLX_RUNTIME_LOCK = threading.RLock()
_LOCK_STATE = threading.local()
_LOCK_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "EdgeStudio"
    / "mlx_runtime.lock"
)


def get_mlx_runtime_lock() -> Any:
    """Return the shared process-local MLX runtime lock."""
    return _MLX_RUNTIME_LOCK


@contextmanager
def mlx_runtime_gate(owner: str = "") -> Iterator[None]:
    """Serialize an MLX load/generate section across threads and processes."""
    del owner  # Reserved for future logging/metrics without changing callers.
    with _MLX_RUNTIME_LOCK:
        depth = int(getattr(_LOCK_STATE, "depth", 0) or 0)
        lock_file = getattr(_LOCK_STATE, "lock_file", None)
        if depth == 0:
            _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            lock_file = _LOCK_PATH.open("a+")
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                lock_file.close()
                raise
            _LOCK_STATE.lock_file = lock_file
        _LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            next_depth = int(getattr(_LOCK_STATE, "depth", 1) or 1) - 1
            _LOCK_STATE.depth = max(0, next_depth)
            if next_depth <= 0:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_file.close()
                    _LOCK_STATE.lock_file = None
