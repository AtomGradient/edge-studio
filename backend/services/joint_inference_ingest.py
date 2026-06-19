# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""C1-min EdgeMesh joint inference handler.

This is the controlled single-hop path: a trusted iOS peer can ask the Mac host
to run one text generation turn, and the Mac streams inference events back over
the existing mTLS mesh connection. It is intentionally generic: no
app-specific schema, receipt, merchant, or expense semantics live here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Iterable, Optional, Protocol

from backend.api.chat_llm import _generate_streaming
from backend.api.chat_params import get_generation_params
from backend.services.mlx_worker import submit_mlx_task
from backend.services.model_manager import LoadedModel, manager
from backend.services.neural_imprint_runtime import ensure_neural_imprint_for_loaded_model

from .conversation_store import get_default_store
from .device_learning_snapshot_store import (
    DeviceLearningSnapshotError,
    latest_device_learning_snapshot,
)
from .mesh_transport import MeshTransportServer, PeerContext
from .persona_source_store import latest_persona_source_for_peer

logger = logging.getLogger(__name__)


REQUEST_OP = "joint_inference_request"
EVENT_OP = "joint_inference_event"
CANCEL_OP = "joint_inference_cancel"
REQUEST_SCHEMA_VERSION = "edgestudio.joint_inference_request.v1"
EVENT_SCHEMA_VERSION = "edgestudio.joint_inference_event.v1"
CANCEL_SCHEMA_VERSION = "edgestudio.joint_inference_cancel.v1"
TERMINAL_EVENT_TYPES = {"complete", "error", "cancelled"}
MODEL_TEMPLATE_STOP_MARKERS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
)
JOINT_INFERENCE_HISTORY_SCHEMA_VERSION = "edgestudio.joint_inference_history.v1"
_HISTORY_LIMIT = 100
_TOKEN_FLUSH_MIN_CHARS = 64
_TOKEN_FLUSH_INTERVAL_SECONDS = 0.12
_history_lock = threading.Lock()
_history: deque[dict] = deque(maxlen=_HISTORY_LIMIT)
_history_by_request_id: dict[str, dict] = {}
_active_lock = threading.Lock()
_active_cancels: dict[str, threading.Event] = {}


class JointInferenceEventStreamer(Protocol):
    def stream_events(self, request: dict) -> Iterable[dict]:
        """Yield raw chat events for one normalized request."""


class DefaultJointInferenceService:
    """Run host-side MLX generation and expose chat events synchronously."""

    def stream_events(self, request: dict) -> Iterable[dict]:
        loaded = _select_loaded_model(request.get("model_id"))
        prompt, history = _prompt_and_history(request)
        model_params = get_generation_params(loaded.model_dir)
        neural_imprint_requirements = _neural_imprint_requirements_for_request(request)
        neural_imprint_status = ensure_neural_imprint_for_loaded_model(
            loaded,
            requirements=neural_imprint_requirements,
        )
        use_neural_imprint = neural_imprint_status is not None
        request["use_neural_imprint"] = use_neural_imprint

        max_tokens = _clamp_int(
            request.get("max_tokens"),
            default=min(512, model_params.max_tokens),
            minimum=1,
            maximum=model_params.max_tokens,
        )
        temperature = _clamp_float(
            request.get("temperature"),
            default=model_params.temperature,
            minimum=0.0,
            maximum=2.0,
        )
        top_k = _clamp_int(
            request.get("top_k"),
            default=model_params.top_k,
            minimum=1,
            maximum=1000,
        )
        top_p = _clamp_float(
            request.get("top_p"),
            default=model_params.top_p,
            minimum=0.0,
            maximum=1.0,
        )
        enable_thinking = request.get("enable_thinking")
        if enable_thinking is None and _supports_thinking(loaded.model_dir):
            enable_thinking = False

        yield from _stream_mlx_events(
            loaded=loaded,
            prompt=prompt,
            history=history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            enable_thinking=enable_thinking,
            use_neural_imprint=use_neural_imprint,
            cancel_event=request.get("_cancel_event"),
        )


def register(
    server: MeshTransportServer,
    service: Optional[JointInferenceEventStreamer] = None,
    *,
    run_async: bool = True,
) -> None:
    """Register the joint inference op on an EdgeMesh transport server."""

    inference_service = service or DefaultJointInferenceService()

    def handle_joint_inference_request(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "joint_inference_request requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"joint_inference_request rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )

        request = _normalize_request(payload)
        request_id = request["request_id"]
        peer_id = ctx.trusted_peer.peer_id
        request["peer_id"] = peer_id
        ctx.trust_store.touch_last_seen(peer_id)
        cancel_event = threading.Event()
        request["_cancel_event"] = cancel_event
        _register_active_request(request_id, cancel_event)

        logger.info(
            "joint_inference_request accepted peer=%s request_id=%s messages=%d neural_imprint=%s",
            peer_id,
            request_id[:16],
            len(request["messages"]),
            bool(request.get("use_neural_imprint")),
        )
        _record_history_start(peer_id=peer_id, request=request)

        def runner() -> None:
            _run_stream_to_peer(
                server=server,
                service=inference_service,
                peer_id=peer_id,
                request=request,
            )

        if run_async:
            threading.Thread(
                target=runner,
                name=f"joint-inference-{request_id[:8]}",
                daemon=True,
            ).start()
        else:
            runner()

        return {
            "op": EVENT_OP,
            "payload": _event_payload(
                request_id=request_id,
                event_type="accepted",
                sequence=0,
                message="joint inference accepted",
            ),
        }

    def handle_joint_inference_cancel(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "joint_inference_cancel requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"joint_inference_cancel rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )

        cancel = _normalize_cancel(payload)
        peer_id = ctx.trusted_peer.peer_id
        ctx.trust_store.touch_last_seen(peer_id)
        found = _cancel_active_request(cancel["request_id"])
        logger.info(
            "joint_inference_cancel peer=%s request_id=%s found=%s reason=%s",
            peer_id,
            cancel["request_id"][:16],
            found,
            cancel.get("reason"),
        )
        return {
            "op": EVENT_OP,
            "payload": _event_payload(
                request_id=cancel["request_id"],
                event_type="status",
                sequence=0,
                message="cancellation requested" if found else "request not active",
            ),
        }

    server.register_handler(REQUEST_OP, handle_joint_inference_request)
    server.register_handler(CANCEL_OP, handle_joint_inference_cancel)
    logger.info(
        "joint_inference_ingest: registered joint_inference_request/cancel handlers"
    )


def _run_stream_to_peer(
    *,
    server: MeshTransportServer,
    service: JointInferenceEventStreamer,
    peer_id: str,
    request: dict,
) -> None:
    request_id = request["request_id"]
    sequence = 1
    pending_token_parts: list[str] = []
    pending_token_chars = 0
    pending_token_id = None
    pending_model_id: str | None = None
    pending_model_path: str | None = None
    last_token_flush = time.monotonic()
    saw_template_stop_marker = False
    cancel_event = _request_cancel_event(request)

    def next_sequence() -> int:
        nonlocal sequence
        value = sequence
        sequence += 1
        return value

    def flush_pending_token(*, force: bool) -> bool:
        nonlocal pending_token_parts
        nonlocal pending_token_chars
        nonlocal pending_token_id
        nonlocal pending_model_id
        nonlocal pending_model_path
        nonlocal last_token_flush
        nonlocal saw_template_stop_marker

        if pending_token_chars <= 0:
            return True

        now = time.monotonic()
        token_text, hit_stop_marker = _split_at_template_stop_marker(
            "".join(pending_token_parts)
        )
        if hit_stop_marker:
            saw_template_stop_marker = True
            force = True
        if (
            not force
            and pending_token_chars < _TOKEN_FLUSH_MIN_CHARS
            and now - last_token_flush < _TOKEN_FLUSH_INTERVAL_SECONDS
        ):
            return True

        token_id = pending_token_id
        model_id = pending_model_id
        model_path = pending_model_path
        pending_token_parts = []
        pending_token_chars = 0
        pending_token_id = None
        pending_model_id = None
        pending_model_path = None
        last_token_flush = now
        if not token_text:
            return True

        event = _event_payload(
            request_id=request_id,
            event_type="token",
            sequence=next_sequence(),
            token=token_text,
            token_id=token_id,
            model_id=model_id,
            model_path=model_path,
        )
        return _send_joint_inference_event(
            server=server,
            peer_id=peer_id,
            request_id=request_id,
            event=event,
        )

    def send_cancelled() -> bool:
        if not flush_pending_token(force=True):
            return False
        event = _event_payload(
            request_id=request_id,
            event_type="cancelled",
            sequence=next_sequence(),
            message="cancelled by requester",
        )
        return _send_joint_inference_event(
            server=server,
            peer_id=peer_id,
            request_id=request_id,
            event=event,
        )

    try:
        if cancel_event is not None and cancel_event.is_set():
            send_cancelled()
            return
        for raw_event in service.stream_events(request):
            if cancel_event is not None and cancel_event.is_set():
                send_cancelled()
                return
            raw_type = str(raw_event.get("type") or "status")
            if raw_type == "token":
                if saw_template_stop_marker:
                    continue
                token = str(raw_event.get("token") or "")
                if token:
                    pending_token_parts.append(token)
                    pending_token_chars += len(token)
                pending_token_id = raw_event.get("token_id")
                pending_model_id = _optional_str(
                    raw_event.get("model_id") or request.get("model_id")
                ) or pending_model_id
                pending_model_path = (
                    _optional_str(raw_event.get("model_path")) or pending_model_path
                )
                if _contains_template_stop_marker("".join(pending_token_parts)):
                    if not flush_pending_token(force=True):
                        return
                    continue
                if not flush_pending_token(force=False):
                    return
                continue

            if not flush_pending_token(force=True):
                return

            event = _event_payload_from_chat_event(
                request_id=request_id,
                raw_event=raw_event,
                sequence=next_sequence(),
                model_id=request.get("model_id"),
            )
            if not _send_joint_inference_event(
                server=server,
                peer_id=peer_id,
                request_id=request_id,
                event=event,
            ):
                return
            if event.get("type") in TERMINAL_EVENT_TYPES:
                return
        if cancel_event is not None and cancel_event.is_set():
            send_cancelled()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "joint_inference_request peer=%s request_id=%s failed: %s\n%s",
            peer_id,
            request_id[:16],
            exc,
            traceback.format_exc(),
        )
        flush_pending_token(force=True)
        event = _event_payload(
            request_id=request_id,
            event_type="error",
            sequence=next_sequence(),
            error=str(exc),
        )
        _send_joint_inference_event(
            server=server,
            peer_id=peer_id,
            request_id=request_id,
            event=event,
        )
    finally:
        _unregister_active_request(request_id)


def _send_joint_inference_event(
    *,
    server: MeshTransportServer,
    peer_id: str,
    request_id: str,
    event: dict,
) -> bool:
    if server.send_to_peer(peer_id, EVENT_OP, event):
        _record_history_event(peer_id=peer_id, event=event)
        return True

    logger.warning(
        "joint_inference_event send failed peer=%s request_id=%s type=%s",
        peer_id,
        request_id[:16],
        event.get("type"),
    )
    _record_history_event(
        peer_id=peer_id,
        event={
            **event,
            "type": "error",
            "error": "failed to send event to peer",
        },
    )
    return False


def list_joint_inference_history(*, limit: int = 50, peer_id: str | None = None) -> dict:
    """Return recent host-side joint inference requests for developer observability."""

    safe_limit = max(1, min(int(limit or 50), _HISTORY_LIMIT))
    sessions = get_default_store().list_sessions(
        surface="joint_inference",
        peer_id=peer_id,
        limit=safe_limit,
        include_messages=True,
    )
    records = [_joint_record_from_session(session) for session in sessions]
    records.sort(key=lambda item: float(item.get("accepted_at") or 0), reverse=True)
    return {
        "ok": True,
        "schema_version": JOINT_INFERENCE_HISTORY_SCHEMA_VERSION,
        "count": len(records[:safe_limit]),
        "items": records[:safe_limit],
    }


def get_joint_inference_history_item(request_id: str) -> dict | None:
    """Return one full joint inference history record by request id."""

    session = get_default_store().get_session(str(request_id), include_messages=True)
    if session is not None and session.get("surface") == "joint_inference":
        return _joint_record_from_session(session)
    with _history_lock:
        record = _history_by_request_id.get(str(request_id))
        return dict(record) if record is not None else None


def delete_joint_inference_history_item(request_id: str) -> bool:
    """Delete one persisted joint inference conversation and matching hot history entry."""

    safe_request_id = str(request_id or "").strip()
    if not safe_request_id:
        return False

    deleted = get_default_store().delete_session(safe_request_id)
    with _history_lock:
        record = _history_by_request_id.pop(safe_request_id, None)
        if record is not None:
            try:
                _history.remove(record)
            except ValueError:
                pass
            deleted = True
    return deleted


def stream_joint_inference_continue(
    *,
    parent_request_id: str,
    payload: dict,
    service: Optional[JointInferenceEventStreamer] = None,
) -> Iterable[dict]:
    """Continue a recorded joint inference conversation on the Mac host."""

    parent = get_joint_inference_history_item(parent_request_id)
    if parent is None:
        raise ValueError(f"joint inference request not found: {parent_request_id}")

    request = _normalize_continue_request(parent=parent, payload=payload)
    peer_id = str(parent.get("peer_id") or "web")
    inference_service = service or DefaultJointInferenceService()
    _record_history_start(peer_id=peer_id, request=request)

    sequence = 1
    try:
        for raw_event in inference_service.stream_events(request):
            event = _event_payload_from_chat_event(
                request_id=request["request_id"],
                raw_event=raw_event,
                sequence=sequence,
                model_id=request.get("model_id"),
            )
            sequence += 1
            _record_history_event(peer_id=peer_id, event=event)
            yield event
            if event.get("type") in TERMINAL_EVENT_TYPES:
                return
    except GeneratorExit:
        event = _event_payload(
            request_id=request["request_id"],
            event_type="cancelled",
            sequence=sequence,
            message="client disconnected",
        )
        _record_history_event(peer_id=peer_id, event=event)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "joint_inference_continue parent_request_id=%s request_id=%s failed: %s\n%s",
            parent_request_id[:16],
            request["request_id"][:16],
            exc,
            traceback.format_exc(),
        )
        event = _event_payload(
            request_id=request["request_id"],
            event_type="error",
            sequence=sequence,
            error=str(exc),
        )
        _record_history_event(peer_id=peer_id, event=event)
        yield event


def reset_joint_inference_history() -> None:
    """Clear joint inference history. Intended for tests."""

    with _history_lock:
        _history.clear()
        _history_by_request_id.clear()
    with _active_lock:
        for cancel_event in _active_cancels.values():
            cancel_event.set()
        _active_cancels.clear()
    for session in get_default_store().list_sessions(surface="joint_inference", limit=500):
        get_default_store().delete_session(session["session_id"])


def _register_active_request(request_id: str, cancel_event: threading.Event) -> None:
    with _active_lock:
        _active_cancels[request_id] = cancel_event


def _cancel_active_request(request_id: str) -> bool:
    with _active_lock:
        cancel_event = _active_cancels.get(request_id)
    if cancel_event is None:
        return False
    cancel_event.set()
    return True


def _unregister_active_request(request_id: str) -> None:
    with _active_lock:
        _active_cancels.pop(request_id, None)


def _request_cancel_event(request: dict) -> threading.Event | None:
    cancel_event = request.get("_cancel_event")
    if isinstance(cancel_event, threading.Event):
        return cancel_event
    return None


def _record_history_start(*, peer_id: str, request: dict) -> None:
    accepted_at = time.time()
    request_id = request["request_id"]
    record = {
        "schema_version": JOINT_INFERENCE_HISTORY_SCHEMA_VERSION,
        "request_id": request_id,
        "conversation_id": request.get("conversation_id"),
        "latest_request_id": request_id,
        "parent_request_id": request.get("parent_request_id"),
        "source": request.get("source") or "device",
        "peer_id": peer_id,
        "status": "accepted",
        "accepted_at": accepted_at,
        "completed_at": None,
        "duration_seconds": None,
        "model_id": request.get("model_id"),
        "max_tokens": request.get("max_tokens"),
        "temperature": request.get("temperature"),
        "enable_thinking": request.get("enable_thinking"),
        "use_neural_imprint": bool(request.get("use_neural_imprint")),
        "neural_imprint_artifact_id": request.get("neural_imprint_artifact_id"),
        "neural_imprint_prefix_token_count": request.get("neural_imprint_prefix_token_count"),
        "route_reason": request.get("route_reason"),
        "messages_count": len(request.get("messages") or []),
        "messages": _copy_messages(request.get("messages") or []),
        "prompt_preview": _preview_request_text(request),
        "output_preview": None,
        "full_text": None,
        "error": None,
        "total_tokens": None,
        "tokens_per_sec": None,
        "last_event_type": "accepted",
        "last_sequence": 0,
        "token_events": 0,
    }
    with _history_lock:
        _history.append(record)
        _history_by_request_id[request_id] = record
        _prune_history_index_locked()
    _persist_joint_record(record)


def _record_history_event(*, peer_id: str, event: dict) -> None:
    request_id = str(event.get("request_id") or "")
    if not request_id:
        return
    now = time.time()
    with _history_lock:
        record = _history_by_request_id.get(request_id)
        if record is None:
            record = {
                "schema_version": JOINT_INFERENCE_HISTORY_SCHEMA_VERSION,
                "request_id": request_id,
                "conversation_id": None,
                "latest_request_id": request_id,
                "parent_request_id": None,
                "source": "observed",
                "peer_id": peer_id,
                "status": "observed",
                "accepted_at": now,
                "completed_at": None,
                "duration_seconds": None,
                "model_id": event.get("model_id"),
                "max_tokens": None,
                "temperature": None,
                "enable_thinking": None,
                "use_neural_imprint": None,
                "neural_imprint_artifact_id": None,
                "neural_imprint_prefix_token_count": None,
                "route_reason": None,
                "messages_count": None,
                "messages": [],
                "prompt_preview": None,
                "output_preview": None,
                "full_text": None,
                "error": None,
                "total_tokens": None,
                "tokens_per_sec": None,
                "last_event_type": None,
                "last_sequence": None,
                "token_events": 0,
            }
            _history.append(record)
            _history_by_request_id[request_id] = record
            _prune_history_index_locked()

        event_type = str(event.get("type") or "status")
        record["peer_id"] = peer_id
        record["last_event_type"] = event_type
        record["last_sequence"] = event.get("sequence")
        record["model_id"] = event.get("model_id") or record.get("model_id")
        if event.get("use_neural_imprint") is not None:
            record["use_neural_imprint"] = bool(event.get("use_neural_imprint"))
        if event.get("neural_imprint_artifact_id"):
            record["neural_imprint_artifact_id"] = event.get("neural_imprint_artifact_id")
        if event.get("neural_imprint_prefix_token_count") is not None:
            record["neural_imprint_prefix_token_count"] = event.get(
                "neural_imprint_prefix_token_count"
            )

        if event_type == "token":
            record["status"] = "streaming"
            record["token_events"] = int(record.get("token_events") or 0) + 1
        elif event_type == "complete":
            record["status"] = "complete"
            record["completed_at"] = now
            record["duration_seconds"] = round(now - float(record.get("accepted_at") or now), 3)
            record["output_preview"] = _preview_text(event.get("full_text"))
            record["full_text"] = event.get("full_text")
            record["total_tokens"] = event.get("total_tokens")
            record["tokens_per_sec"] = event.get("tokens_per_sec")
        elif event_type in {"error", "cancelled"}:
            record["status"] = event_type
            record["completed_at"] = now
            record["duration_seconds"] = round(now - float(record.get("accepted_at") or now), 3)
            record["error"] = event.get("error") or event.get("message")
        elif event_type == "queued":
            record["status"] = "queued"
        elif event_type == "status":
            record["status"] = str(event.get("message") or "status")
        snapshot = dict(record)
    _persist_joint_record(snapshot)


def _prune_history_index_locked() -> None:
    live_ids = {str(item.get("request_id")) for item in _history}
    stale = [request_id for request_id in _history_by_request_id if request_id not in live_ids]
    for request_id in stale:
        _history_by_request_id.pop(request_id, None)


def _persist_joint_record(record: dict) -> None:
    request_id = str(record.get("request_id") or "").strip()
    if not request_id:
        return
    session_id = _joint_conversation_session_id(record)
    persisted_record = dict(record)
    persisted_record["conversation_id"] = session_id
    persisted_record["latest_request_id"] = request_id
    store = get_default_store()
    metadata = {"joint_inference_record": _json_safe_record(persisted_record)}
    store.create_session(
        surface="joint_inference",
        session_id=session_id,
        title=record.get("prompt_preview") or request_id,
        model_id=record.get("model_id"),
        peer_id=record.get("peer_id"),
        source=record.get("source"),
        status=str(record.get("status") or "observed"),
        metadata=metadata,
        created_at=float(record.get("accepted_at") or time.time()),
        updated_at=float(record.get("completed_at") or time.time()),
    )
    store.replace_messages(session_id, _conversation_messages_for_joint_record(persisted_record))


def _joint_record_from_session(session: dict) -> dict:
    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    record = metadata.get("joint_inference_record") if isinstance(metadata, dict) else None
    if isinstance(record, dict):
        out = dict(record)
        persisted_messages = [
            {"role": item.get("role"), "content": item.get("content")}
            for item in session.get("messages") or []
            if item.get("role") in {"system", "user", "assistant", "tool"}
        ]
        latest_request_id = out.get("latest_request_id") or out.get("request_id")
        out["conversation_id"] = session.get("session_id")
        out["latest_request_id"] = latest_request_id
        out["request_id"] = session.get("session_id")
        out["status"] = session.get("status") or out.get("status")
        out["model_id"] = session.get("model_id") or out.get("model_id")
        out["peer_id"] = session.get("peer_id") or out.get("peer_id")
        out["source"] = session.get("source") or out.get("source")
        if persisted_messages:
            out["messages"] = persisted_messages
            out["messages_count"] = len(persisted_messages)
        return out
    messages = [
        {"role": item.get("role"), "content": item.get("content")}
        for item in session.get("messages") or []
        if item.get("role") in {"system", "user", "assistant", "tool"}
    ]
    full_text = None
    for item in reversed(messages):
        if item.get("role") == "assistant":
            full_text = item.get("content")
            break
    return {
        "schema_version": JOINT_INFERENCE_HISTORY_SCHEMA_VERSION,
        "request_id": session.get("session_id"),
        "conversation_id": session.get("session_id"),
        "latest_request_id": session.get("session_id"),
        "parent_request_id": None,
        "source": session.get("source") or "persisted",
        "peer_id": session.get("peer_id"),
        "status": session.get("status"),
        "accepted_at": session.get("created_at"),
        "completed_at": session.get("updated_at"),
        "duration_seconds": None,
        "model_id": session.get("model_id"),
        "max_tokens": None,
        "temperature": None,
        "enable_thinking": None,
        "use_neural_imprint": None,
        "neural_imprint_artifact_id": None,
        "neural_imprint_prefix_token_count": None,
        "route_reason": None,
        "messages_count": len(messages),
        "messages": messages,
        "prompt_preview": session.get("title"),
        "output_preview": _preview_text(full_text),
        "full_text": full_text,
        "error": None,
        "total_tokens": None,
        "tokens_per_sec": None,
        "last_event_type": None,
        "last_sequence": None,
        "token_events": None,
    }


def _conversation_messages_for_joint_record(record: dict) -> list[dict]:
    messages = []
    message_prefix = str(record.get("conversation_id") or record.get("request_id") or "joint")
    for idx, item in enumerate(record.get("messages") or []):
        role = item.get("role")
        content = item.get("content")
        if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
            messages.append(
                {
                    "message_id": f"{message_prefix}-input-{idx}",
                    "sequence": idx,
                    "role": role,
                    "content": content,
                    "metadata": {"source": "request"},
                }
            )
    if record.get("full_text"):
        messages.append(
            {
                "message_id": f"{record['request_id']}-assistant",
                "sequence": len(messages),
                "role": "assistant",
                "content": str(record.get("full_text") or ""),
                "metadata": {
                    "source": "response",
                    "total_tokens": record.get("total_tokens"),
                    "tokens_per_sec": record.get("tokens_per_sec"),
                },
            }
        )
    elif record.get("error"):
        messages.append(
            {
                "message_id": f"{record['request_id']}-error",
                "sequence": len(messages),
                "role": "assistant",
                "content": str(record.get("error") or ""),
                "metadata": {"source": "error"},
            }
        )
    return messages


def _joint_conversation_session_id(record: dict) -> str:
    conversation_id = str(record.get("conversation_id") or "").strip()
    if conversation_id:
        return conversation_id
    return str(record.get("request_id") or "").strip()


def _json_safe_record(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if value is None
        or isinstance(value, (str, int, float, bool, list, dict))
    }


def _preview_request_text(request: dict) -> str | None:
    prompt = request.get("prompt")
    if prompt:
        return _preview_text(prompt)
    messages = request.get("messages") or []
    for item in reversed(messages):
        if item.get("role") == "user":
            return _preview_text(item.get("content"))
    if messages:
        return _preview_text(messages[-1].get("content"))
    return None


def _preview_text(value, *, limit: int = 160) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _neural_imprint_requirements_for_request(request: dict) -> dict | None:
    """Return the peer/app learning signature required for host restore.

    Device-initiated joint inference must not silently restore a host artifact
    that merely matches the base model. It has to match the paired device's
    latest learning snapshot/source, especially the tool schema hash that
    controls model-emitted tool_call arguments.
    """

    peer_id = _optional_str(request.get("peer_id"))
    if not peer_id:
        return None

    requirements: dict[str, str | bool] = {"peer_id": peer_id}
    snapshot_record = None
    try:
        snapshot_record = latest_device_learning_snapshot(peer_id)
    except DeviceLearningSnapshotError as exc:
        logger.warning(
            "joint_inference neural imprint snapshot lookup failed peer=%s code=%s",
            peer_id,
            exc.code,
        )
    snapshot = (
        snapshot_record.get("snapshot")
        if isinstance(snapshot_record, dict) and isinstance(snapshot_record.get("snapshot"), dict)
        else {}
    )
    learning = snapshot.get("learning") if isinstance(snapshot.get("learning"), dict) else {}
    identity = snapshot.get("identity") if isinstance(snapshot.get("identity"), dict) else {}

    tool_schema_sha256 = _optional_sha256(learning.get("tool_schema_sha256"))
    if tool_schema_sha256:
        requirements["tool_schema_sha256"] = tool_schema_sha256
    else:
        requirements["strict_no_match"] = True

    app_id = _optional_str(identity.get("app_id") or identity.get("bundle_identifier"))
    source = latest_persona_source_for_peer(peer_id)
    if source is not None:
        requirements["source_id"] = source.source_id
        source_sha256 = _optional_sha256(source.receipt.get("source_sha256"))
        if source_sha256:
            requirements["source_sha256"] = source_sha256
        source_app_id = _optional_str(source.receipt.get("app_id"))
        if source_app_id:
            requirements["app_id"] = source_app_id
        source_kind = _optional_str(source.receipt.get("source_kind"))
        if source_kind:
            requirements["source_kind"] = source_kind
    elif app_id:
        requirements["app_id"] = app_id

    return requirements


def _normalize_request(payload: dict) -> dict:
    schema_version = payload.get("schema_version", REQUEST_SCHEMA_VERSION)
    if schema_version != REQUEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported joint inference schema_version: {schema_version}")

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("joint_inference_request missing/invalid request_id")

    raw_messages = payload.get("messages", [])
    if raw_messages is None:
        raw_messages = []
    if not isinstance(raw_messages, list):
        raise ValueError("joint_inference_request messages must be an array")

    messages: list[dict[str, str]] = []
    for idx, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise ValueError(f"joint_inference_request messages[{idx}] must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"joint_inference_request messages[{idx}] invalid role")
        if not isinstance(content, str):
            raise ValueError(f"joint_inference_request messages[{idx}] invalid content")
        messages.append({"role": role, "content": content})

    prompt = payload.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("joint_inference_request prompt must be a string")
    if not messages and not (prompt or "").strip():
        raise ValueError("joint_inference_request requires messages or prompt")

    return {
        "request_id": request_id.strip(),
        "conversation_id": _optional_str(payload.get("conversation_id")),
        "peer_id": _optional_str(payload.get("peer_id")),
        "model_id": _optional_str(payload.get("model_id")),
        "prompt": prompt,
        "messages": messages,
        "max_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
        "top_k": payload.get("top_k"),
        "top_p": payload.get("top_p"),
        "enable_thinking": _optional_bool(payload.get("enable_thinking")),
        # Host runtime decides Neural Imprint restore from its local registry.
        # Client flags are advisory legacy fields and must not force restore.
        "use_neural_imprint": False,
        "route_reason": _optional_str(payload.get("route_reason")),
        "source": _optional_str(payload.get("source")),
        "parent_request_id": _optional_str(payload.get("parent_request_id")),
    }


def _normalize_cancel(payload: dict) -> dict:
    schema_version = payload.get("schema_version", CANCEL_SCHEMA_VERSION)
    if schema_version != CANCEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported joint inference cancel schema_version: {schema_version}")

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("joint_inference_cancel missing/invalid request_id")

    reason = payload.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("joint_inference_cancel reason must be a string")

    return {
        "request_id": request_id.strip(),
        "peer_id": _optional_str(payload.get("peer_id")),
        "reason": _optional_str(reason),
    }


def _normalize_continue_request(*, parent: dict, payload: dict) -> dict:
    messages = payload.get("messages")
    if messages is None:
        messages = _messages_from_history_record(parent)
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            messages.append({"role": "user", "content": message.strip()})

    enable_thinking = (
        payload.get("enable_thinking")
        if "enable_thinking" in payload
        else parent.get("enable_thinking", False)
    )

    request_payload = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_id": str(uuid.uuid4()),
        "peer_id": parent.get("peer_id"),
        "model_id": payload.get("model_id") or parent.get("model_id"),
        "messages": messages,
        "max_tokens": payload.get("max_tokens") or 2048,
        "temperature": payload.get("temperature") if payload.get("temperature") is not None else 0.2,
        "top_k": payload.get("top_k"),
        "top_p": payload.get("top_p"),
        "enable_thinking": enable_thinking,
        "use_neural_imprint": bool(payload.get(
            "use_neural_imprint",
            parent.get("use_neural_imprint") or False,
        )),
        "route_reason": payload.get("route_reason") or "joint_inference_history_continue",
        "source": "web",
        "conversation_id": parent.get("conversation_id") or parent.get("request_id"),
        "parent_request_id": parent.get("latest_request_id") or parent.get("request_id"),
    }
    return _normalize_request(request_payload)


def _messages_from_history_record(record: dict) -> list[dict[str, str]]:
    messages = _copy_messages(record.get("messages") or [])
    full_text = record.get("full_text")
    if isinstance(full_text, str) and full_text and (
        not messages or messages[-1].get("role") != "assistant"
    ):
        messages.append({"role": "assistant", "content": full_text})
    if not messages:
        prompt = record.get("prompt_preview")
        if isinstance(prompt, str) and prompt:
            messages.append({"role": "user", "content": prompt})
            output = record.get("full_text") or record.get("output_preview")
            if isinstance(output, str) and output:
                messages.append({"role": "assistant", "content": output})
    return messages


def _copy_messages(messages: list[dict]) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
            copied.append({"role": role, "content": content})
    return copied


def _select_loaded_model(model_hint: str | None) -> LoadedModel:
    if model_hint:
        loaded = manager.get_model(model_hint)
        if loaded is not None:
            return loaded

    candidates = [m for m in manager.list_models() if m.category in {"llm", "vlm"}]
    if model_hint:
        for loaded in candidates:
            if Path(loaded.model_dir).name == model_hint:
                return loaded
    if candidates:
        return candidates[0]

    env_model = os.environ.get("EDGE_JOINT_INFERENCE_MODEL")
    if env_model:
        return manager.load_model(env_model)

    raise RuntimeError(
        "No host LLM/VLM model loaded for joint inference. "
        "Load a model in EdgeStudio first or set EDGE_JOINT_INFERENCE_MODEL."
    )


def _prompt_and_history(request: dict) -> tuple[str, list[dict[str, str]]]:
    prompt = str(request.get("prompt") or "").strip()
    messages = list(request.get("messages") or [])
    if prompt:
        return prompt, messages

    if messages and messages[-1].get("role") == "tool":
        return "", messages

    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return messages[idx]["content"], messages[:idx]
    if messages:
        last = messages[-1]
        return last["content"], messages[:-1]
    raise ValueError("joint_inference_request requires a non-empty prompt")


def _stream_mlx_events(
    *,
    loaded: LoadedModel,
    prompt: str,
    history: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    enable_thinking: bool | None,
    use_neural_imprint: bool,
    cancel_event: threading.Event | None = None,
) -> Iterable[dict]:
    loop = asyncio.new_event_loop()
    ready: queue.Queue[asyncio.Queue] = queue.Queue(maxsize=1)
    if cancel_event is None:
        cancel_event = threading.Event()

    def loop_runner() -> None:
        asyncio.set_event_loop(loop)
        ready.put(asyncio.Queue())
        loop.run_forever()

    loop_thread = threading.Thread(
        target=loop_runner,
        name="joint-inference-event-loop",
        daemon=True,
    )
    loop_thread.start()
    event_queue = ready.get(timeout=2)
    future = submit_mlx_task(
        _generate_streaming,
        loaded.model_id,
        loaded.model_dir,
        prompt,
        history,
        max_tokens,
        temperature,
        top_k,
        top_p,
        enable_thinking,
        event_queue,
        loop,
        cancel_event,
        False,
        None,
        use_neural_imprint,
    )

    try:
        while True:
            try:
                event = asyncio.run_coroutine_threadsafe(
                    event_queue.get(),
                    loop,
                ).result(timeout=0.5)
            except FutureTimeoutError:
                if cancel_event.is_set():
                    yield {"type": "cancelled"}
                    return
                if future.done():
                    exc = future.exception()
                    if exc is not None:
                        raise exc
                    return
                continue
            event = dict(event)
            event.setdefault("model_id", loaded.model_id)
            event.setdefault("model_path", loaded.model_dir)
            yield event
            if event.get("type") in TERMINAL_EVENT_TYPES:
                return
    finally:
        cancel_event.set()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)


def _event_payload_from_chat_event(
    *,
    request_id: str,
    raw_event: dict,
    sequence: int,
    model_id: str | None,
) -> dict:
    event_type = str(raw_event.get("type") or "status")
    if event_type == "token":
        return _event_payload(
            request_id=request_id,
            event_type="token",
            sequence=sequence,
            token=str(raw_event.get("token") or ""),
            token_id=raw_event.get("token_id"),
            model_id=_optional_str(raw_event.get("model_id") or model_id),
            model_path=_optional_str(raw_event.get("model_path")),
        )
    if event_type == "complete":
        return _event_payload(
            request_id=request_id,
            event_type="complete",
            sequence=sequence,
            full_text=_sanitize_generated_text(str(raw_event.get("full_text") or "")),
            total_tokens=raw_event.get("total_tokens"),
            tokens_per_sec=raw_event.get("tokens_per_sec"),
            prefill_time=raw_event.get("prefill_time"),
            total_time=raw_event.get("total_time"),
            model_id=_optional_str(raw_event.get("model_id") or model_id),
            model_path=_optional_str(raw_event.get("model_path")),
        )
    if event_type == "error":
        return _event_payload(
            request_id=request_id,
            event_type="error",
            sequence=sequence,
            error=str(raw_event.get("message") or raw_event.get("error") or "unknown error"),
            model_id=_optional_str(raw_event.get("model_id") or model_id),
            model_path=_optional_str(raw_event.get("model_path")),
        )
    if event_type == "cancelled":
        return _event_payload(
            request_id=request_id,
            event_type="cancelled",
            sequence=sequence,
            message="cancelled",
            model_id=_optional_str(raw_event.get("model_id") or model_id),
            model_path=_optional_str(raw_event.get("model_path")),
        )
    return _event_payload(
        request_id=request_id,
        event_type="status",
        sequence=sequence,
        message=str(raw_event.get("message") or event_type),
        model_id=_optional_str(raw_event.get("model_id") or model_id),
        model_path=_optional_str(raw_event.get("model_path")),
        use_neural_imprint=_optional_bool(raw_event.get("use_neural_imprint")),
        neural_imprint_artifact_id=_optional_str(raw_event.get("neural_imprint_artifact_id")),
        neural_imprint_prefix_token_count=_optional_int(
            raw_event.get("neural_imprint_prefix_token_count")
        ),
    )


def _event_payload(
    *,
    request_id: str,
    event_type: str,
    sequence: int,
    message: str | None = None,
    token: str | None = None,
    token_id: int | None = None,
    full_text: str | None = None,
    total_tokens: int | None = None,
    tokens_per_sec: float | None = None,
    prefill_time: float | None = None,
    total_time: float | None = None,
    model_id: str | None = None,
    model_path: str | None = None,
    error: str | None = None,
    use_neural_imprint: bool | None = None,
    neural_imprint_artifact_id: str | None = None,
    neural_imprint_prefix_token_count: int | None = None,
) -> dict:
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "request_id": request_id,
        "type": event_type,
        "sequence": sequence,
    }
    optional = {
        "message": message,
        "token": token,
        "token_id": token_id,
        "full_text": full_text,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "prefill_time": prefill_time,
        "total_time": total_time,
        "model_id": model_id,
        "model_path": model_path,
        "error": error,
        "use_neural_imprint": use_neural_imprint,
        "neural_imprint_artifact_id": neural_imprint_artifact_id,
        "neural_imprint_prefix_token_count": neural_imprint_prefix_token_count,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    return payload


def _sanitize_generated_text(text: str) -> str:
    """Trim template control tokens that can leak from host-side generation."""

    return _split_at_template_stop_marker(text)[0].rstrip()


def _contains_template_stop_marker(text: str) -> bool:
    return any(marker in text for marker in MODEL_TEMPLATE_STOP_MARKERS)


def _split_at_template_stop_marker(text: str) -> tuple[str, bool]:
    first_marker = min(
        (
            idx
            for marker in MODEL_TEMPLATE_STOP_MARKERS
            if (idx := text.find(marker)) >= 0
        ),
        default=-1,
    )
    if first_marker >= 0:
        return text[:first_marker], True
    return text, False


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_sha256(value) -> str | None:
    text = _optional_str(value)
    if text and len(text) == 64 and all(ch in "0123456789abcdef" for ch in text):
        return text
    return None


def _optional_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _clamp_float(value, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _supports_thinking(model_dir: str) -> bool:
    try:
        from backend.core.universal_tracer import detect_thinking_support

        return bool(detect_thinking_support(model_dir))
    except Exception:
        return False
