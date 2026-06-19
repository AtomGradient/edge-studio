# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import logging
import os
import time
import traceback
from typing import Optional

from .classify_service import ClassifyService, get_default_service
from .mesh_transport import MeshTransportServer, PeerContext

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def register(
    server: MeshTransportServer,
    service: Optional[ClassifyService] = None,
    *,
    max_attempts: int | None = None,
    retry_backoff_s: float | None = None,
) -> None:
    classify_service = service or get_default_service()
    attempts = max(
        1,
        max_attempts
        if max_attempts is not None
        else _env_int("EDGE_CLASSIFY_RETRY_ATTEMPTS", 2, minimum=1),
    )
    backoff_s = max(
        0.0,
        retry_backoff_s
        if retry_backoff_s is not None
        else _env_float("EDGE_CLASSIFY_RETRY_BACKOFF_S", 0.25, minimum=0.0),
    )

    def handle_classify_request(payload: dict, ctx: PeerContext) -> dict:
        # Auth (same standard as event_upload)
        if ctx.trusted_peer is None:
            raise PermissionError(
                "classify_request requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"classify_request rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )

        # Extract request_id from payload (iPhone uses it to correlate response)
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("classify_request missing/invalid request_id")

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("classify_request missing/empty messages array")

        max_tokens = int(payload.get("max_tokens", 1024))
        temperature = float(payload.get("temperature", 0.0))

        logger.info(
            "classify_request begin peer=%s request_id=%s messages=%d max_tokens=%d",
            ctx.trusted_peer.peer_id,
            request_id[:16],
            len(messages),
            max_tokens,
        )

        # Run LLM. Error paths return an error field instead of raising — lets iPhone fall back rather than mesh error.
        try:
            result = _generate_with_retry(
                classify_service,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=request_id,
                peer_id=ctx.trusted_peer.peer_id,
                max_attempts=attempts,
                retry_backoff_s=backoff_s,
            )
        except Exception as exc:
            logger.error(
                "classify_request peer=%s request_id=%s failed: %s\n%s",
                ctx.trusted_peer.peer_id,
                request_id[:16],
                exc,
                traceback.format_exc(),
            )
            ctx.trust_store.touch_last_seen(ctx.trusted_peer.peer_id)
            return {
                "op": "classify_response",
                "payload": {
                    "request_id": request_id,
                    "output": "",
                    "elapsed_ms": 0,
                    "tokens_generated": 0,
                    "model_path": "",
                    "error": str(exc),
                },
            }

        logger.info(
            "classify_request done peer=%s request_id=%s elapsed_ms=%d tokens=%d",
            ctx.trusted_peer.peer_id,
            request_id[:16],
            result.elapsed_ms,
            result.tokens_generated,
        )

        # TrustStore auto-updates last_seen (consistent with event_upload)
        ctx.trust_store.touch_last_seen(ctx.trusted_peer.peer_id)

        return {
            "op": "classify_response",
            "payload": {
                "request_id": request_id,
                "output": result.output,
                "elapsed_ms": result.elapsed_ms,
                "tokens_generated": result.tokens_generated,
                "model_path": result.model_path,
                "error": None,
            },
        }

    server.register_handler("classify_request", handle_classify_request)
    logger.info("classify_ingest: registered classify_request handler")


def _generate_with_retry(
    service: ClassifyService,
    *,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    request_id: str,
    peer_id: str,
    max_attempts: int,
    retry_backoff_s: float,
):
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return service.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_attempts:
                break
            logger.warning(
                "classify_request retry peer=%s request_id=%s attempt=%d/%d error=%s",
                peer_id,
                request_id[:16],
                attempt,
                max_attempts,
                exc,
            )
            if retry_backoff_s > 0:
                time.sleep(retry_backoff_s * attempt)
    assert last_exc is not None
    raise last_exc
