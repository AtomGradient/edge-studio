# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Shared single-thread MLX worker.

MLX model objects and cache tensors should stay on one long-lived worker
thread. Chat generation and Neural Imprint restore both use this executor so the
restored prompt cache can be reused by later generation without crossing
runtime-thread boundaries.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")

_MLX_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-chat")


def submit_mlx_task(
    fn: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> Future[_T]:
    """Submit work to the shared MLX worker."""

    return _MLX_POOL.submit(fn, *args, **kwargs)


def run_mlx_task(
    fn: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Run work on the shared MLX worker and wait for the result."""

    return submit_mlx_task(fn, *args, **kwargs).result()
