# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Metal command-buffer limits for public MLX runtimes.

Public MLX reads ``MLX_MAX_OPS_PER_BUFFER`` and ``MLX_MAX_MB_PER_BUFFER`` while
the Metal device is initialised. These helpers make that startup-time contract
explicit and expose a capability probe for future public runtime setters.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .exceptions import UnsupportedRuntimeMutation

MAX_OPS_ENV = "MLX_MAX_OPS_PER_BUFFER"
MAX_MB_ENV = "MLX_MAX_MB_PER_BUFFER"


@dataclass(frozen=True)
class MetalLimitCapabilities:
    """Capabilities exposed by the currently installed public MLX package."""

    env_preimport: bool
    runtime_max_ops: bool
    runtime_max_mb: bool

    def as_dict(self) -> dict[str, bool]:
        return {
            "env_preimport": self.env_preimport,
            "runtime_max_ops": self.runtime_max_ops,
            "runtime_max_mb": self.runtime_max_mb,
        }


def _validate_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return ivalue


def configure_preimport(
    *,
    max_ops_per_buffer: int | None = None,
    max_mb_per_buffer: int | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    """Set MLX Metal limit environment variables before MLX initialises.

    This is the release-safe path for public MLX wheels. Call it before the
    first GPU operation in the current process, or pass its result to a worker
    subprocess environment.
    """

    target = os.environ if environ is None else environ
    max_ops = _validate_limit("max_ops_per_buffer", max_ops_per_buffer)
    max_mb = _validate_limit("max_mb_per_buffer", max_mb_per_buffer)
    if max_ops is not None:
        target[MAX_OPS_ENV] = str(max_ops)
    if max_mb is not None:
        target[MAX_MB_ENV] = str(max_mb)
    return dict(target)


def worker_env(
    *,
    max_ops_per_buffer: int | None = None,
    max_mb_per_buffer: int | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an environment suitable for launching an MLX worker subprocess."""

    env = dict(os.environ if base_env is None else base_env)
    configure_preimport(
        max_ops_per_buffer=max_ops_per_buffer,
        max_mb_per_buffer=max_mb_per_buffer,
        environ=env,
    )
    return env


def run_worker(
    args: Sequence[str],
    *,
    max_ops_per_buffer: int | None = None,
    max_mb_per_buffer: int | None = None,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Run a subprocess with MLX Metal limits applied before import time."""

    return subprocess.run(
        list(args),
        env=worker_env(
            max_ops_per_buffer=max_ops_per_buffer,
            max_mb_per_buffer=max_mb_per_buffer,
            base_env=env,
        ),
        **kwargs,
    )


def _metal_module() -> Any | None:
    try:
        import mlx.core as mx
    except Exception:
        return None
    return getattr(mx, "metal", None)


def capabilities() -> MetalLimitCapabilities:
    metal = _metal_module()
    return MetalLimitCapabilities(
        env_preimport=True,
        runtime_max_ops=bool(
            metal is not None
            and hasattr(metal, "set_max_ops_per_buffer")
            and hasattr(metal, "get_max_ops_per_buffer")
        ),
        runtime_max_mb=bool(
            metal is not None
            and hasattr(metal, "set_max_mb_per_buffer")
            and hasattr(metal, "get_max_mb_per_buffer")
        ),
    )


def set_runtime_max_ops_per_buffer(value: int) -> int:
    """Set max ops in-process when the installed public MLX exposes the API."""

    metal = _metal_module()
    if metal is None or not hasattr(metal, "set_max_ops_per_buffer"):
        raise UnsupportedRuntimeMutation(
            "Installed public MLX does not expose mx.metal.set_max_ops_per_buffer; "
            "set MLX_MAX_OPS_PER_BUFFER before import or launch a worker subprocess."
        )
    metal.set_max_ops_per_buffer(_validate_limit("max_ops_per_buffer", value))
    if hasattr(metal, "get_max_ops_per_buffer"):
        return int(metal.get_max_ops_per_buffer())
    return int(value)


def set_runtime_max_mb_per_buffer(value: int) -> int:
    """Set max MB in-process when the installed public MLX exposes the API."""

    metal = _metal_module()
    if metal is None or not hasattr(metal, "set_max_mb_per_buffer"):
        raise UnsupportedRuntimeMutation(
            "Installed public MLX does not expose mx.metal.set_max_mb_per_buffer; "
            "set MLX_MAX_MB_PER_BUFFER before import or launch a worker subprocess."
        )
    metal.set_max_mb_per_buffer(_validate_limit("max_mb_per_buffer", value))
    if hasattr(metal, "get_max_mb_per_buffer"):
        return int(metal.get_max_mb_per_buffer())
    return int(value)
