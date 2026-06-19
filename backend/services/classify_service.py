# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from backend.services.mlx_runtime_gate import mlx_runtime_gate

logger = logging.getLogger(__name__)


# Default base model (prefer bf16 for training/inference)
# Qwen3.5-4B-bf16 is the same family as the daemon default client, best prompt compatibility
DEFAULT_MODEL_PATH = os.path.expanduser(
    os.environ.get(
        "EDGE_CLASSIFY_MODEL",
        "~/Documents/mlx-community/Qwen3.5-4B-bf16",
    )
)


@dataclass
class ClassifyResult:

    output: str
    elapsed_ms: int
    tokens_generated: int
    model_path: str


class ClassifyService:

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        # MLX keeps thread-local native state. Running load/generate on request
        # or mesh callback threads lets those threads exit after inference and
        # can crash inside mlx.core TLS teardown. Keep one process-long worker
        # thread per service instance and serialize all MLX runtime access on it.
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="edge-classify-mlx",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with mlx_runtime_gate("classify_service.load"):
            with self._load_lock:
                if self._model is not None:
                    return
                path = Path(self.model_path)
                if not path.exists():
                    raise FileNotFoundError(
                        f"classify model not found at {self.model_path}. "
                        f"Set EDGE_CLASSIFY_MODEL env or place model at default path."
                    )
                logger.info("classify_service: loading model from %s ...", self.model_path)
                t0 = time.time()
                from mlx_lm.utils import load

                self._model, self._tokenizer = load(self.model_path)
                elapsed = time.time() - t0
                logger.info(
                    "classify_service: model ready (%.1fs) — ready to handle classify_request",
                    elapsed,
                )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ClassifyResult:
        worker_messages = [dict(message) for message in messages]
        return self._worker.submit(
            self._generate_on_worker,
            worker_messages,
            max_tokens,
            temperature,
        ).result()

    def _generate_on_worker(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> ClassifyResult:
        """Run the full MLX path on the service-owned persistent worker."""
        self._ensure_loaded()

        # Convert messages to the prompt string expected by the model
        # mlx_lm chat models use tokenizer.apply_chat_template
        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise RuntimeError(
                f"tokenizer at {self.model_path} 不支持 apply_chat_template — "
                "classification 必须用 instruction-tuned 模型 (Memory feedback_it_model_suffix)"
            )
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Memory feedback_qwen3_thinking_disabled_default
        )

        with mlx_runtime_gate("classify_service.generate"), self._gen_lock:
            t0 = time.time()
            from mlx_lm.generate import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=temperature)
            try:
                output = mlx_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            finally:
                _clear_mlx_cache_after_generate()
            elapsed = time.time() - t0

        # Token count estimate (mlx_lm.generate does not return token count directly, infer via tokenizer)
        try:
            tokens = len(self._tokenizer.encode(output))
        except Exception:
            tokens = 0

        elapsed_ms = int(elapsed * 1000)
        logger.info(
            "classify_service: generated %d tokens in %dms (%.1f tps)",
            tokens,
            elapsed_ms,
            tokens / max(0.001, elapsed),
        )

        return ClassifyResult(
            output=output,
            elapsed_ms=elapsed_ms,
            tokens_generated=tokens,
            model_path=self.model_path,
        )


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------

_default_service: Optional[ClassifyService] = None
_default_service_lock = threading.Lock()


def _clear_mlx_cache_after_generate() -> None:
    try:
        import mlx.core as mx

        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("classify_service: mlx cache clear skipped: %s", exc)


def get_default_service() -> ClassifyService:
    global _default_service
    if _default_service is None:
        with _default_service_lock:
            if _default_service is None:
                _default_service = ClassifyService()
    return _default_service
