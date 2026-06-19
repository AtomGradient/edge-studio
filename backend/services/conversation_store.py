# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persistent conversation session store for EdgeStudio developer surfaces.

The store is deliberately generic: it knows about surfaces such as ``chat``,
``neural_imprint_chat`` and ``joint_inference``, but it does not encode app-specific
business semantics.  Each surface can attach structured metadata when it needs
extra audit fields.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .app_dirs import data_path


SCHEMA_VERSION = "edgestudio.conversation_sessions.v1"


def default_db_path() -> Path:
    override = os.environ.get("EDGESTUDIO_CONVERSATION_DB")
    if override:
        return Path(override).expanduser()
    base = data_path("db")
    base.mkdir(parents=True, exist_ok=True)
    return base / "conversations.sqlite"


DDL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id    TEXT PRIMARY KEY,
    surface       TEXT NOT NULL,
    title         TEXT,
    model_id      TEXT,
    peer_id       TEXT,
    source        TEXT,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id    TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    sequence      INTEGER NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(session_id) REFERENCES conversation_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversation_sessions_surface_updated
    ON conversation_sessions(surface, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_peer_updated
    ON conversation_sessions(peer_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_sequence
    ON conversation_messages(session_id, sequence);
CREATE UNIQUE INDEX IF NOT EXISTS uq_conversation_messages_session_sequence
    ON conversation_messages(session_id, sequence);
"""


class ConversationStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(DDL)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def create_session(
        self,
        *,
        surface: str,
        session_id: str | None = None,
        title: str | None = None,
        model_id: str | None = None,
        peer_id: str | None = None,
        source: str | None = None,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> dict[str, Any]:
        sid = _clean_id(session_id) or uuid.uuid4().hex
        now = time.time()
        c_at = float(created_at or now)
        u_at = float(updated_at or c_at)
        clean_surface = _required_text(surface, "surface")
        clean_status = _optional_text(status) or "active"
        metadata_json = _json_dumps(metadata or {})
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversation_sessions (
                    session_id, surface, title, model_id, peer_id, source,
                    status, created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    surface = excluded.surface,
                    title = excluded.title,
                    model_id = excluded.model_id,
                    peer_id = excluded.peer_id,
                    source = excluded.source,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    sid,
                    clean_surface,
                    _optional_text(title),
                    _optional_text(model_id),
                    _optional_text(peer_id),
                    _optional_text(source),
                    clean_status,
                    c_at,
                    u_at,
                    metadata_json,
                ),
            )
        session = self.get_session(sid, include_messages=False)
        if session is None:  # pragma: no cover - insert failure would raise first
            raise RuntimeError(f"failed to create conversation session {sid}")
        return session

    def update_session(
        self,
        session_id: str,
        *,
        title: str | None = None,
        model_id: str | None = None,
        peer_id: str | None = None,
        source: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        updated_at: float | None = None,
        merge_metadata: bool = True,
    ) -> dict[str, Any] | None:
        sid = _required_text(session_id, "session_id")
        existing = self.get_session(sid, include_messages=False)
        if existing is None:
            return None
        next_metadata = existing.get("metadata") or {}
        if metadata is not None:
            next_metadata = {**next_metadata, **metadata} if merge_metadata else metadata
        with self._lock:
            self._conn.execute(
                """
                UPDATE conversation_sessions
                SET title = COALESCE(?, title),
                    model_id = COALESCE(?, model_id),
                    peer_id = COALESCE(?, peer_id),
                    source = COALESCE(?, source),
                    status = COALESCE(?, status),
                    updated_at = ?,
                    metadata_json = ?
                WHERE session_id = ?;
                """,
                (
                    _optional_text(title),
                    _optional_text(model_id),
                    _optional_text(peer_id),
                    _optional_text(source),
                    _optional_text(status),
                    float(updated_at or time.time()),
                    _json_dumps(next_metadata),
                    sid,
                ),
            )
        return self.get_session(sid, include_messages=False)

    def get_session(
        self,
        session_id: str,
        *,
        include_messages: bool = True,
    ) -> dict[str, Any] | None:
        sid = _required_text(session_id, "session_id")
        with self._lock:
            return self._get_session_locked(
                self._conn,
                sid,
                include_messages=include_messages,
            )

    def list_sessions(
        self,
        *,
        surface: str | None = None,
        peer_id: str | None = None,
        limit: int = 50,
        include_messages: bool = False,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 50), 500))
        clauses: list[str] = []
        args: list[Any] = []
        if surface:
            clauses.append("surface = ?")
            args.append(surface)
        if peer_id:
            clauses.append("peer_id = ?")
            args.append(peer_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(safe_limit)
        sql = f"""
            SELECT session_id, surface, title, model_id, peer_id, source,
                   status, created_at, updated_at, metadata_json
            FROM conversation_sessions
            {where}
            ORDER BY updated_at DESC
            LIMIT ?;
        """
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        sessions_by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            session = _row_to_session(row)
            existing = sessions_by_id.get(session["session_id"])
            if existing is None or session["updated_at"] >= existing["updated_at"]:
                sessions_by_id[session["session_id"]] = session
        sessions = sorted(
            sessions_by_id.values(),
            key=lambda item: item["updated_at"],
            reverse=True,
        )[:safe_limit]
        if include_messages:
            for session in sessions:
                full = self.get_session(session["session_id"], include_messages=True)
                session["messages"] = (full or {}).get("messages", [])
        return sessions

    def replace_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        *,
        updated_at: float | None = None,
    ) -> list[dict[str, Any]]:
        sid = _required_text(session_id, "session_id")
        now = float(updated_at or time.time())
        with self._lock:
            if not self._session_exists_locked(sid):
                raise KeyError(sid)
            self._conn.execute("BEGIN;")
            try:
                self._conn.execute("DELETE FROM conversation_messages WHERE session_id = ?;", (sid,))
                for index, message in enumerate(messages):
                    self._insert_message_locked(
                        session_id=sid,
                        sequence=int(message.get("sequence", index)),
                        role=_required_text(message.get("role"), "role"),
                        content=str(message.get("content") or ""),
                        message_id=_clean_id(message.get("message_id")) or _clean_id(message.get("id")),
                        created_at=_optional_float(message.get("created_at")) or now,
                        metadata=message.get("metadata") if isinstance(message.get("metadata"), dict) else {},
                    )
                self._conn.execute(
                    "UPDATE conversation_sessions SET updated_at = ? WHERE session_id = ?;",
                    (now, sid),
                )
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise
            rows = self._conn.execute(
                """
                SELECT message_id, session_id, sequence, role, content, created_at, metadata_json
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY sequence ASC, created_at ASC;
                """,
                (sid,),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        message_id: str | None = None,
        sequence: int | None = None,
        created_at: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sid = _required_text(session_id, "session_id")
        now = time.time()
        with self._lock:
            if not self._session_exists_locked(sid):
                raise KeyError(sid)
            seq = sequence
            if seq is None:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(sequence), -1) + 1 FROM conversation_messages WHERE session_id = ?;",
                    (sid,),
                ).fetchone()
                seq = int(row[0] if row is not None else 0)
            msg_id = self._insert_message_locked(
                session_id=sid,
                sequence=int(seq),
                role=_required_text(role, "role"),
                content=str(content or ""),
                message_id=message_id,
                created_at=float(created_at or now),
                metadata=metadata or {},
            )
            self._conn.execute(
                "UPDATE conversation_sessions SET updated_at = ? WHERE session_id = ?;",
                (now, sid),
            )
            row = self._conn.execute(
                """
                SELECT message_id, session_id, sequence, role, content, created_at, metadata_json
                FROM conversation_messages
                WHERE message_id = ?;
                """,
                (msg_id,),
            ).fetchone()
        return _row_to_message(row)

    def delete_session(self, session_id: str) -> bool:
        sid = _required_text(session_id, "session_id")
        with self._lock:
            cur = self._conn.execute("DELETE FROM conversation_sessions WHERE session_id = ?;", (sid,))
            return cur.rowcount > 0

    def reset(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM conversation_messages;")
            self._conn.execute("DELETE FROM conversation_sessions;")

    def _session_exists_locked(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM conversation_sessions WHERE session_id = ? LIMIT 1;",
            (session_id,),
        ).fetchone()
        return row is not None

    def _get_session_locked(
        self,
        conn: sqlite3.Connection | None,
        session_id: str,
        *,
        include_messages: bool,
    ) -> dict[str, Any] | None:
        if conn is None:
            return None
        row = conn.execute(
            """
            SELECT session_id, surface, title, model_id, peer_id, source,
                   status, created_at, updated_at, metadata_json
            FROM conversation_sessions
            WHERE session_id = ?;
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        session = _row_to_session(row)
        if include_messages:
            rows = conn.execute(
                """
                SELECT message_id, session_id, sequence, role, content, created_at, metadata_json
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY sequence ASC, created_at ASC;
                """,
                (session_id,),
            ).fetchall()
            session["messages"] = [_row_to_message(msg_row) for msg_row in rows]
        return session

    def _insert_message_locked(
        self,
        *,
        session_id: str,
        sequence: int,
        role: str,
        content: str,
        message_id: str | None,
        created_at: float,
        metadata: dict[str, Any],
    ) -> str:
        mid = _clean_id(message_id) or uuid.uuid4().hex
        existing = self._conn.execute(
            "SELECT session_id FROM conversation_messages WHERE message_id = ?;",
            (mid,),
        ).fetchone()
        if existing is not None and existing["session_id"] != session_id:
            mid = f"{session_id}-{mid}-{uuid.uuid4().hex[:8]}"
        self._conn.execute(
            """
            INSERT INTO conversation_messages (
                message_id, session_id, sequence, role, content, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                sequence = excluded.sequence,
                role = excluded.role,
                content = excluded.content,
                created_at = excluded.created_at,
                metadata_json = excluded.metadata_json;
            """,
            (mid, session_id, sequence, role, content, created_at, _json_dumps(metadata)),
        )
        return mid


_default_store: ConversationStore | None = None
_default_store_path_override: Path | None = None
_default_lock = threading.Lock()


def get_default_store() -> ConversationStore:
    global _default_store
    with _default_lock:
        path = _default_store_path_override or default_db_path()
        if _default_store is None or _default_store.db_path != path:
            _default_store = ConversationStore(path)
        return _default_store


def reset_default_store_for_tests(db_path: Path | None = None) -> ConversationStore:
    global _default_store, _default_store_path_override
    with _default_lock:
        _default_store_path_override = db_path
        _default_store = ConversationStore(db_path)
        _default_store.reset()
        return _default_store


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": row["session_id"],
        "surface": row["surface"],
        "title": row["title"],
        "model_id": row["model_id"],
        "peer_id": row["peer_id"],
        "source": row["source"],
        "status": row["status"],
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "metadata": _json_loads(row["metadata_json"]),
    }


def _row_to_message(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:  # pragma: no cover
        raise RuntimeError("message row missing")
    return {
        "message_id": row["message_id"],
        "session_id": row["session_id"],
        "sequence": int(row["sequence"]),
        "role": row["role"],
        "content": row["content"],
        "created_at": float(row["created_at"]),
        "metadata": _json_loads(row["metadata_json"]),
    }


def _json_dumps(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_id(value: Any) -> str | None:
    text = _optional_text(value)
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
