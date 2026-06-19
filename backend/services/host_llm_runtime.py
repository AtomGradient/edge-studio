# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host LLM runtime facade for local EdgeStudio models.

This module owns the current local MLX-backed runtime details for host-model
assistant calls. The public contract is model selection plus chat generation;
the implementation can keep using ClassifyService internally until a dedicated
host inference runtime replaces it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_HOST_LLM_SERVICE_CACHE: dict[str, Any] = {}
_HOST_LLM_SERVICE_LOCK = threading.Lock()


@dataclass(frozen=True)
class HostLLMResult:
    output: str
    model_id: str
    model_path: str
    elapsed_ms: int
    tokens_generated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "elapsed_ms": self.elapsed_ms,
            "tokens_generated": self.tokens_generated,
        }


class HostLLMRuntimeError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        details: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details


def generate_host_chat(
    *,
    messages: list[dict[str, str]],
    host_model_id: str | None,
    max_tokens: int,
    temperature: float,
) -> HostLLMResult:
    """Generate one chat response with the selected host model or fallback."""
    service = _get_service_for_host_model(host_model_id)
    result = service.generate(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return HostLLMResult(
        output=result.output,
        model_id=host_model_id or result.model_path,
        model_path=result.model_path,
        elapsed_ms=result.elapsed_ms,
        tokens_generated=result.tokens_generated,
    )


def _get_service_for_host_model(host_model_id: str | None) -> Any:
    if not host_model_id:
        from backend.services.classify_service import get_default_service

        return get_default_service()

    from backend.services.model_manager import manager

    loaded = manager.get_model(host_model_id)
    if loaded is not None:
        return _get_local_runtime_service_for_model_path(loaded.model_dir)

    model_path = Path(host_model_id).expanduser()
    if model_path.exists():
        return _get_local_runtime_service_for_model_path(str(model_path))

    raise HostLLMRuntimeError(
        code="host_model_not_loaded",
        message="Selected EdgeStudio host model is not loaded.",
        retryable=False,
        details={"model_id": host_model_id},
    )


def _get_local_runtime_service_for_model_path(model_path: str) -> Any:
    from backend.services.classify_service import (
        DEFAULT_MODEL_PATH,
        ClassifyService,
        get_default_service,
    )

    cache_key = _model_path_cache_key(model_path)
    if cache_key == _model_path_cache_key(DEFAULT_MODEL_PATH):
        return get_default_service()

    with _HOST_LLM_SERVICE_LOCK:
        cached = _HOST_LLM_SERVICE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        service = ClassifyService(model_path=model_path)
        _HOST_LLM_SERVICE_CACHE[cache_key] = service
        return service


def _model_path_cache_key(model_path: str) -> str:
    return str(Path(model_path).expanduser().resolve(strict=False))
