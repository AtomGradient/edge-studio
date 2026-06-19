# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Conversation session persistence API."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from backend.services.conversation_store import (
    SCHEMA_VERSION,
    get_default_store,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
def list_conversations(
    surface: Optional[str] = Query(default=None),
    peer_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    include_messages: bool = Query(default=False),
) -> dict[str, Any]:
    items = get_default_store().list_sessions(
        surface=surface,
        peer_id=peer_id,
        limit=limit,
        include_messages=include_messages,
    )
    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "count": len(items),
        "items": items,
    }


@router.post("")
async def create_conversation(request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        session = get_default_store().create_session(
            surface=body.get("surface"),
            session_id=body.get("session_id"),
            title=body.get("title"),
            model_id=body.get("model_id"),
            peer_id=body.get("peer_id"),
            source=body.get("source"),
            status=body.get("status") or "active",
            metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "session": session}


@router.get("/{session_id}")
def get_conversation(session_id: str) -> dict[str, Any]:
    session = get_default_store().get_session(session_id, include_messages=True)
    if session is None:
        raise HTTPException(status_code=404, detail="conversation session not found")
    return {"ok": True, "session": session}


@router.patch("/{session_id}")
async def update_conversation(session_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        session = get_default_store().update_session(
            session_id,
            title=body.get("title"),
            model_id=body.get("model_id"),
            peer_id=body.get("peer_id"),
            source=body.get("source"),
            status=body.get("status"),
            metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
            merge_metadata=bool(body.get("merge_metadata", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=404, detail="conversation session not found")
    return {"ok": True, "session": session}


@router.put("/{session_id}/messages")
async def replace_conversation_messages(session_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")
    try:
        messages = get_default_store().replace_messages(session_id, raw_messages)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="conversation session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "count": len(messages), "messages": messages}


@router.post("/{session_id}/messages")
async def append_conversation_message(session_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        message = get_default_store().append_message(
            session_id,
            role=body.get("role"),
            content=body.get("content") or "",
            message_id=body.get("message_id") or body.get("id"),
            sequence=body.get("sequence"),
            metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="conversation session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "message": message}


@router.delete("/{session_id}")
def delete_conversation(session_id: str) -> dict[str, Any]:
    ok = get_default_store().delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation session not found")
    return {"ok": True, "session_id": session_id}
