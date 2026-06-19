# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .app_dirs import data_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_db_path() -> Path:
    base = data_path("db")
    base.mkdir(parents=True, exist_ok=True)
    return base / "trust.sqlite"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TrustedPeer:
    peer_id: str
    display_name: str
    fingerprint: str            # 64-char lowercase hex
    role: str                   # "brain" | "sensor" | "peer"
    paired_at_ms: int           # Unix ms
    last_seen_at_ms: Optional[int]
    revoked: bool
    cert_der: Optional[bytes] = None    # Raw certificate DER, written during pair_hello in P0 phase

    # Stage 3 P1.1: real-time sensor observations from the most recent keepalive ping.
    # last_stats is a JSON dict (PingStats serialized), last_stats_at_ms is the Unix ms when ping was received.
    # nil means never reported (legacy client or not enabled).
    last_stats: Optional[dict] = None
    last_stats_at_ms: Optional[int] = None

    def to_api(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "display_name": self.display_name,
            "fingerprint": self.fingerprint,
            "role": self.role,
            "paired_at": self.paired_at_ms / 1000.0,
            "last_seen_at": (self.last_seen_at_ms / 1000.0) if self.last_seen_at_ms else None,
            "revoked": self.revoked,
            "last_stats": self.last_stats,
            "last_stats_at": (self.last_stats_at_ms / 1000.0) if self.last_stats_at_ms else None,
        }


# ---------------------------------------------------------------------------
# TrustStore
# ---------------------------------------------------------------------------


DDL = """
CREATE TABLE IF NOT EXISTS trusted_peers (
    peer_id           TEXT PRIMARY KEY,
    display_name      TEXT NOT NULL,
    fingerprint       TEXT NOT NULL,
    role              TEXT NOT NULL,
    paired_at         INTEGER NOT NULL,
    last_seen_at      INTEGER,
    revoked           INTEGER NOT NULL DEFAULT 0,
    cert_der          BLOB,
    last_stats_json   TEXT,
    last_stats_at_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_trusted_peers_fingerprint ON trusted_peers(fingerprint);
"""

# Idempotent ALTER list — old DBs (missing last_stats_*) get columns added on startup.
# Must check via PRAGMA table_info before ADD COLUMN; SQLite ALTER does not support column-level IF NOT EXISTS.
_INCREMENTAL_COLUMNS = (
    ("last_stats_json", "TEXT"),
    ("last_stats_at_ms", "INTEGER"),
)


class TrustStore:

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,   # lock ensures single-thread access
            isolation_level=None,       # autocommit (one transaction per statement)
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.executescript(DDL)
        self._migrate_incremental_columns()

    def _migrate_incremental_columns(self) -> None:
        try:
            existing = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(trusted_peers);").fetchall()
            }
            for column, sql_type in _INCREMENTAL_COLUMNS:
                if column not in existing:
                    self._conn.execute(
                        f"ALTER TABLE trusted_peers ADD COLUMN {column} {sql_type};"
                    )
                    logger.info("trust_store migrated: added column %s %s", column, sql_type)
        except sqlite3.Error as exc:  # pragma: no cover — best-effort migration
            logger.warning("trust_store column migration failed: %s", exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def upsert(self, peer: TrustedPeer) -> None:
        sql = """
        INSERT INTO trusted_peers (peer_id, display_name, fingerprint, role,
                                    paired_at, last_seen_at, revoked, cert_der)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(peer_id) DO UPDATE SET
            display_name = excluded.display_name,
            fingerprint  = excluded.fingerprint,
            role         = excluded.role,
            paired_at    = excluded.paired_at,
            last_seen_at = excluded.last_seen_at,
            revoked      = excluded.revoked,
            cert_der     = excluded.cert_der;
        """
        with self._lock:
            self._conn.execute(sql, (
                peer.peer_id,
                peer.display_name,
                peer.fingerprint.lower(),
                peer.role,
                int(peer.paired_at_ms),
                int(peer.last_seen_at_ms) if peer.last_seen_at_ms is not None else None,
                1 if peer.revoked else 0,
                peer.cert_der,
            ))

    # Shared SELECT column list — all SELECTs expand from here to avoid column order drift in _row_to_peer.
    _SELECT_COLUMNS = (
        "peer_id, display_name, fingerprint, role, paired_at, last_seen_at, "
        "revoked, cert_der, last_stats_json, last_stats_at_ms"
    )

    def lookup(self, peer_id: str) -> Optional[TrustedPeer]:
        sql = f"SELECT {self._SELECT_COLUMNS} FROM trusted_peers WHERE peer_id = ?;"
        with self._lock:
            row = self._conn.execute(sql, (peer_id,)).fetchone()
        return self._row_to_peer(row)

    def lookup_by_fingerprint(self, fingerprint: str) -> Optional[TrustedPeer]:
        sql = f"SELECT {self._SELECT_COLUMNS} FROM trusted_peers WHERE fingerprint = ? LIMIT 1;"
        with self._lock:
            row = self._conn.execute(sql, (fingerprint.lower(),)).fetchone()
        return self._row_to_peer(row)

    def list_all(self) -> list[TrustedPeer]:
        sql = f"SELECT {self._SELECT_COLUMNS} FROM trusted_peers ORDER BY paired_at DESC;"
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [p for row in rows if (p := self._row_to_peer(row))]

    def revoke(self, peer_id: str) -> None:
        sql = "UPDATE trusted_peers SET revoked = 1 WHERE peer_id = ?;"
        with self._lock:
            self._conn.execute(sql, (peer_id,))

    def delete(self, peer_id: str) -> None:
        sql = "DELETE FROM trusted_peers WHERE peer_id = ?;"
        with self._lock:
            self._conn.execute(sql, (peer_id,))

    def touch_last_seen(self, peer_id: str, at_ms: Optional[int] = None) -> None:
        ts_ms = at_ms if at_ms is not None else int(time.time() * 1000)
        sql = "UPDATE trusted_peers SET last_seen_at = ? WHERE peer_id = ?;"
        with self._lock:
            self._conn.execute(sql, (ts_ms, peer_id))

    def update_peer_stats(
        self,
        peer_id: str,
        stats: dict[str, Any],
        at_ms: Optional[int] = None,
    ) -> None:
        ts_ms = at_ms if at_ms is not None else int(time.time() * 1000)
        try:
            payload = json.dumps(stats, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            logger.warning("update_peer_stats: stats not JSON-serializable peer=%s: %s", peer_id, exc)
            return
        sql = (
            "UPDATE trusted_peers "
            "SET last_stats_json = ?, last_stats_at_ms = ?, last_seen_at = ? "
            "WHERE peer_id = ?;"
        )
        with self._lock:
            self._conn.execute(sql, (payload, ts_ms, ts_ms, peer_id))

    # ------------------------------------------------------------------
    # Verify (hot path — called on every mTLS handshake)
    # ------------------------------------------------------------------

    def verify(self, fingerprint: str) -> Optional[TrustedPeer]:
        peer = self.lookup_by_fingerprint(fingerprint)
        if peer is None or peer.revoked:
            return None
        return peer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_peer(row) -> Optional[TrustedPeer]:
        if row is None:
            return None
        (
            peer_id,
            display_name,
            fingerprint,
            role,
            paired_at,
            last_seen_at,
            revoked,
            cert_der,
            last_stats_json,
            last_stats_at_ms,
        ) = row
        last_stats: Optional[dict] = None
        if last_stats_json:
            try:
                parsed = json.loads(last_stats_json)
                if isinstance(parsed, dict):
                    last_stats = parsed
            except json.JSONDecodeError as exc:
                logger.debug("last_stats_json corrupt for peer=%s: %s", peer_id, exc)
        return TrustedPeer(
            peer_id=peer_id,
            display_name=display_name,
            fingerprint=fingerprint,
            role=role,
            paired_at_ms=int(paired_at),
            last_seen_at_ms=int(last_seen_at) if last_seen_at is not None else None,
            revoked=bool(revoked),
            cert_der=cert_der,
            last_stats=last_stats,
            last_stats_at_ms=int(last_stats_at_ms) if last_stats_at_ms is not None else None,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Singleton (most callers do not want to hold the DB path)
# ---------------------------------------------------------------------------


_default_store: Optional[TrustStore] = None
_default_store_lock = threading.Lock()


def get_default_store() -> TrustStore:
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = TrustStore()
        return _default_store
