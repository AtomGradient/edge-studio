# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from backend.services.app_dirs import data_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Epoch conversion (Swift default Date <-> Unix ms)
# ---------------------------------------------------------------------------


# Swift/Apple reference date: 2001-01-01 00:00:00 UTC = Unix 978307200 seconds.
# Swift default `JSONEncoder` serializes Date as Double seconds since this epoch.
SWIFT_REFERENCE_EPOCH_SECONDS = 978_307_200.0


def swift_date_to_unix_ms(value: float) -> int:
    return int((value + SWIFT_REFERENCE_EPOCH_SECONDS) * 1000)


def unix_ms_to_swift_date(ms: int) -> float:
    return (ms / 1000.0) - SWIFT_REFERENCE_EPOCH_SECONDS


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_db_path() -> Path:
    base = data_path("db")
    base.mkdir(parents=True, exist_ok=True)
    return base / "events.sqlite"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DataEvent:
    id: str                 # UUID string (Swift uuidString format, uppercase with hyphens)
    timestamp_ms: int       # Unix epoch ms
    app_id: str
    event_type: str
    payload: bytes          # Decoded by business consumers
    tags: list[str]         # sorted raw strings
    source_peer_id: Optional[str] = None   # Not a Swift field; Python-side record of which device uploaded
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_api(self, *, include_payload: bool = False) -> dict:
        obj = {
            "id": self.id,
            "timestamp": self.timestamp_ms / 1000.0,
            "app_id": self.app_id,
            "event_type": self.event_type,
            "tags": list(self.tags),
            "source_peer_id": self.source_peer_id,
            "payload_size": len(self.payload),
        }
        if include_payload:
            import base64
            obj["payload_b64"] = base64.b64encode(self.payload).decode("ascii")
        return obj


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


DDL = """
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    timestamp       INTEGER NOT NULL,
    app_id          TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         BLOB NOT NULL,
    tags            TEXT NOT NULL,        -- JSON array string
    source_peer_id  TEXT,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_app ON events(app_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_peer_id);
"""


class EventStore:

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,           # autocommit; use explicit transactions for BEGIN/COMMIT
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.executescript(DDL)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, event: DataEvent) -> bool:
        return self.insert_batch([event])[1][0]

    def insert_batch(self, events: Iterable[DataEvent]) -> tuple[list[str], list[bool]]:
        received_ids: list[str] = []
        is_new_flags: list[bool] = []
        sql = """
        INSERT OR IGNORE INTO events
            (id, timestamp, app_id, event_type, payload, tags, source_peer_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                for ev in events:
                    tags_json = json.dumps(sorted(set(ev.tags)))
                    cur = self._conn.execute(sql, (
                        ev.id,
                        int(ev.timestamp_ms),
                        ev.app_id,
                        ev.event_type,
                        ev.payload,
                        tags_json,
                        ev.source_peer_id,
                        int(ev.created_at_ms),
                    ))
                    received_ids.append(ev.id)
                    is_new_flags.append(cur.rowcount > 0)
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise
        return received_ids, is_new_flags

    def upsert_batch_by_payload(
        self,
        events: Iterable[DataEvent],
    ) -> tuple[list[str], list[bool], list[bool]]:
        """Insert or replace derived events when their payload changes.

        Raw device events should keep using `insert_batch`. This helper is for
        host-generated derived events, where the same stable event id represents
        the latest reviewed supervision for a source case.
        """

        event_list = list(events)
        received_ids = [ev.id for ev in event_list]
        is_new_flags: list[bool] = []
        updated_flags: list[bool] = []
        if not event_list:
            return received_ids, is_new_flags, updated_flags

        select_sql = "SELECT payload, tags FROM events WHERE id = ?;"
        insert_sql = """
        INSERT INTO events
            (id, timestamp, app_id, event_type, payload, tags, source_peer_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        update_sql = """
        UPDATE events
           SET timestamp = ?,
               app_id = ?,
               event_type = ?,
               payload = ?,
               tags = ?,
               source_peer_id = ?,
               created_at = ?
         WHERE id = ?;
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                for ev in event_list:
                    tags_json = json.dumps(sorted(set(ev.tags)))
                    row = self._conn.execute(select_sql, (ev.id,)).fetchone()
                    if row is None:
                        self._conn.execute(insert_sql, (
                            ev.id,
                            int(ev.timestamp_ms),
                            ev.app_id,
                            ev.event_type,
                            ev.payload,
                            tags_json,
                            ev.source_peer_id,
                            int(ev.created_at_ms),
                        ))
                        is_new_flags.append(True)
                        updated_flags.append(False)
                        continue

                    is_new_flags.append(False)
                    existing_payload, existing_tags = row
                    if existing_payload == ev.payload and existing_tags == tags_json:
                        updated_flags.append(False)
                        continue
                    self._conn.execute(update_sql, (
                        int(ev.timestamp_ms),
                        ev.app_id,
                        ev.event_type,
                        ev.payload,
                        tags_json,
                        ev.source_peer_id,
                        int(ev.created_at_ms),
                        ev.id,
                    ))
                    updated_flags.append(True)
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise
        return received_ids, is_new_flags, updated_flags

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        tags: Optional[list[str]] = None,
        app_id: Optional[str] = None,
        event_type: Optional[str] = None,
        source_peer_id: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[DataEvent]:
        clauses: list[str] = []
        params: list = []

        if app_id:
            clauses.append("app_id = ?")
            params.append(app_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if source_peer_id:
            clauses.append("source_peer_id = ?")
            params.append(source_peer_id)
        if since_ms is not None:
            clauses.append("timestamp >= ?")
            params.append(int(since_ms))
        if until_ms is not None:
            clauses.append("timestamp <= ?")
            params.append(int(until_ms))
        if tags:
            tag_or = " OR ".join("tags LIKE ?" for _ in tags)
            clauses.append(f"({tag_or})")
            for t in tags:
                params.append(f'%"{t}"%')

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT id, timestamp, app_id, event_type, payload, tags, source_peer_id, created_at
          FROM events
          {where_sql}
          ORDER BY timestamp DESC
          LIMIT ? OFFSET ?;
        """
        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))
        params.extend([safe_limit, safe_offset])

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get(self, event_id: str) -> Optional[DataEvent]:
        sql = """
        SELECT id, timestamp, app_id, event_type, payload, tags, source_peer_id, created_at
          FROM events WHERE id = ?;
        """
        with self._lock:
            row = self._conn.execute(sql, (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM events;").fetchone()
        return int(n)

    def stats(self) -> dict:
        with self._lock:
            (total,) = self._conn.execute("SELECT COUNT(*) FROM events;").fetchone()
            (total_bytes,) = self._conn.execute(
                "SELECT COALESCE(SUM(LENGTH(payload)), 0) FROM events;"
            ).fetchone()
            (oldest_ms,) = self._conn.execute(
                "SELECT MIN(timestamp) FROM events;"
            ).fetchone()
            (newest_ms,) = self._conn.execute(
                "SELECT MAX(timestamp) FROM events;"
            ).fetchone()
            per_type_rows = self._conn.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type;"
            ).fetchall()
            per_source_rows = self._conn.execute(
                "SELECT COALESCE(source_peer_id, '<local>'), COUNT(*) "
                "FROM events GROUP BY source_peer_id;"
            ).fetchall()
        return {
            "total_events": int(total),
            "total_bytes": int(total_bytes),
            "oldest_timestamp": (int(oldest_ms) / 1000.0) if oldest_ms else None,
            "newest_timestamp": (int(newest_ms) / 1000.0) if newest_ms else None,
            "per_type": {t: int(n) for (t, n) in per_type_rows},
            "per_source_peer": {s: int(n) for (s, n) in per_source_rows},
        }

    # ------------------------------------------------------------------
    # Retention / purge
    # ------------------------------------------------------------------

    def purge_older_than(self, cutoff_ms: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE timestamp < ?;", (int(cutoff_ms),)
            )
            return cur.rowcount or 0

    def delete_all(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM events;")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_event(row) -> DataEvent:
        id_str, ts_ms, app_id, event_type, payload, tags_json, source_peer, created_at = row
        try:
            tags = list(json.loads(tags_json)) if tags_json else []
        except Exception:  # noqa: BLE001
            tags = []
        return DataEvent(
            id=id_str,
            timestamp_ms=int(ts_ms),
            app_id=app_id,
            event_type=event_type,
            payload=bytes(payload) if payload else b"",
            tags=tags,
            source_peer_id=source_peer,
            created_at_ms=int(created_at) if created_at is not None else 0,
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_default_store: Optional[EventStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> EventStore:
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = EventStore()
        return _default_store


# ---------------------------------------------------------------------------
# Wire decoding helpers (Swift Codable JSON -> DataEvent)
# ---------------------------------------------------------------------------


def decode_wire_event(obj: dict, *, source_peer_id: Optional[str]) -> DataEvent:
    import base64

    raw_id = str(obj["id"]).strip()
    # Validate UUID form but keep the original case / format the client sent.
    uuid.UUID(raw_id)

    ts_swift = float(obj["timestamp"])
    ts_ms = swift_date_to_unix_ms(ts_swift)

    payload_b64 = obj.get("payload", "")
    payload = base64.b64decode(payload_b64) if payload_b64 else b""

    tags_raw = obj.get("tags", [])
    if not isinstance(tags_raw, list):
        raise ValueError("tags must be an array")
    tags = [str(t) for t in tags_raw]

    return DataEvent(
        id=raw_id,
        timestamp_ms=ts_ms,
        app_id=str(obj["appId"]),
        event_type=str(obj["eventType"]),
        payload=payload,
        tags=tags,
        source_peer_id=source_peer_id,
    )
