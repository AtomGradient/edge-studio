# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .schema import FactSchema, get_schema

# H1 hard constraint: allowed source_type whitelist (excludes external_doc / shared_user etc.)
ALLOWED_SOURCE_TYPES = frozenset({"user_device"})

# SQL schema version (bump on field changes)
FACT_TABLE_SCHEMA_VERSION = 1


@dataclass
class FactRecord:
    id: str
    schema_name: str
    payload: Dict[str, Any]
    created_at: int  # unix ms
    sensitivity: str = "private"
    ttl_seconds: Optional[int] = None
    derived_from: Optional[str] = None
    source_type: str = "user_device"

    @classmethod
    def new(
        cls,
        schema_name: str,
        payload: Dict[str, Any],
        *,
        created_at: Optional[int] = None,
        sensitivity: str = "private",
        ttl_seconds: Optional[int] = None,
        derived_from: Optional[str] = None,
        source_type: str = "user_device",
        id: Optional[str] = None,
    ) -> "FactRecord":
        return cls(
            id=id or str(uuid.uuid4()),
            schema_name=schema_name,
            payload=payload,
            created_at=created_at if created_at is not None else int(time.time() * 1000),
            sensitivity=sensitivity,
            ttl_seconds=ttl_seconds,
            derived_from=derived_from,
            source_type=source_type,
        )


class FactStore:

    def __init__(self, path: Union[str, Path], *, read_only: bool = False):
        self.path = Path(path)
        self._read_only = read_only

        if read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"read-only but file not found: {self.path}")
            uri = f"file:{self.path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), isolation_level=None)

        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = OFF")  # H3 self-contained
        self._conn.execute("PRAGMA journal_mode = WAL")

        if not read_only:
            self._init_schema()

    # ── Lifecycle ─────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                schema_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                sensitivity TEXT NOT NULL DEFAULT 'private',
                ttl_seconds INTEGER,
                derived_from TEXT,
                source_type TEXT NOT NULL DEFAULT 'user_device',

                idx_amount REAL,
                idx_merchant TEXT,
                idx_category TEXT,
                idx_time INTEGER,
                idx_location TEXT
            );

            CREATE INDEX IF NOT EXISTS ix_facts_schema ON facts(schema_name);
            CREATE INDEX IF NOT EXISTS ix_facts_time ON facts(idx_time);
            CREATE INDEX IF NOT EXISTS ix_facts_merchant ON facts(idx_merchant);
            CREATE INDEX IF NOT EXISTS ix_facts_category ON facts(idx_category);
        """)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(FACT_TABLE_SCHEMA_VERSION)),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "FactStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Write ─────────────────────────────────────────────

    def record(self, fact: FactRecord) -> str:
        if self._read_only:
            raise RuntimeError("FactStore is read-only")

        # H1 hard constraint
        if fact.source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(
                f"source_type {fact.source_type!r} not allowed. "
                f"H1 constraint: only {ALLOWED_SOURCE_TYPES} permitted."
            )

        # Schema validation
        schema = get_schema(fact.schema_name)
        schema.validate_payload(fact.payload)

        # Extract indexed columns
        idx_cols = _extract_indexed_columns(schema, fact.payload)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO facts (
                id, schema_name, payload_json, created_at,
                sensitivity, ttl_seconds, derived_from, source_type,
                idx_amount, idx_merchant, idx_category, idx_time, idx_location
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.id,
                fact.schema_name,
                json.dumps(fact.payload, ensure_ascii=False, sort_keys=True),
                fact.created_at,
                fact.sensitivity,
                fact.ttl_seconds,
                fact.derived_from,
                fact.source_type,
                idx_cols.get("amount"),
                idx_cols.get("merchant"),
                idx_cols.get("category"),
                idx_cols.get("time"),
                idx_cols.get("location"),
            ),
        )
        return fact.id

    def record_many(self, facts: List[FactRecord]) -> Dict[str, int]:
        if self._read_only:
            raise RuntimeError("FactStore is read-only")

        written = 0
        try:
            self._conn.execute("BEGIN")
            for fact in facts:
                self.record(fact)
                written += 1
            self._conn.execute("COMMIT")
            return {"written": written, "failed": 0}
        except Exception:
            self._conn.execute("ROLLBACK")
            return {"written": 0, "failed": len(facts)}

    # ── Read ──────────────────────────────────────────────

    def get(self, fact_id: str) -> Optional[FactRecord]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def query(
        self,
        *,
        schema: Optional[str] = None,
        amount: Optional[float] = None,
        amount_gte: Optional[float] = None,
        amount_lte: Optional[float] = None,
        merchant: Optional[str] = None,
        merchant_like: Optional[str] = None,
        category: Optional[str] = None,
        time_gte: Optional[int] = None,
        time_lte: Optional[int] = None,
        location: Optional[str] = None,
        limit: int = 100,
    ) -> List[FactRecord]:
        conditions: List[str] = []
        params: List[Any] = []

        if schema is not None:
            conditions.append("schema_name = ?"); params.append(schema)
        if amount is not None:
            conditions.append("idx_amount = ?"); params.append(amount)
        if amount_gte is not None:
            conditions.append("idx_amount >= ?"); params.append(amount_gte)
        if amount_lte is not None:
            conditions.append("idx_amount <= ?"); params.append(amount_lte)
        if merchant is not None:
            conditions.append("idx_merchant = ?"); params.append(merchant)
        if merchant_like is not None:
            conditions.append("idx_merchant LIKE ?"); params.append(f"%{merchant_like}%")
        if category is not None:
            conditions.append("idx_category = ?"); params.append(category)
        if time_gte is not None:
            conditions.append("idx_time >= ?"); params.append(time_gte)
        if time_lte is not None:
            conditions.append("idx_time <= ?"); params.append(time_lte)
        if location is not None:
            conditions.append("idx_location = ?"); params.append(location)

        where = " AND ".join(conditions) if conditions else "1"
        sql = f"SELECT * FROM facts WHERE {where} ORDER BY idx_time DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_record(r) for r in rows]

    def count(self, schema: Optional[str] = None) -> int:
        if schema is None:
            return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        return self._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE schema_name = ?", (schema,)
        ).fetchone()[0]

    def aggregate_sum(
        self,
        *,
        schema: str,
        field: str = "amount",
        category: Optional[str] = None,
        merchant: Optional[str] = None,
        merchant_like: Optional[str] = None,
        time_gte: Optional[int] = None,
        time_lte: Optional[int] = None,
    ) -> float:
        schema_def = get_schema(schema)
        if field not in schema_def.fields or schema_def.fields[field].type not in ("int", "float"):
            raise ValueError(f"field {field!r} not numeric in schema {schema!r}")

        col_map = {"amount": "idx_amount"}
        col = col_map.get(field)
        if col is None:
            raise ValueError(f"aggregation on field {field!r} not supported in MVP 0.1")

        conditions = ["schema_name = ?"]
        params: List[Any] = [schema]
        if category is not None:
            conditions.append("idx_category = ?"); params.append(category)
        if merchant is not None:
            conditions.append("idx_merchant = ?"); params.append(merchant)
        if merchant_like is not None:
            conditions.append("idx_merchant LIKE ?"); params.append(f"%{merchant_like}%")
        if time_gte is not None:
            conditions.append("idx_time >= ?"); params.append(time_gte)
        if time_lte is not None:
            conditions.append("idx_time <= ?"); params.append(time_lte)

        sql = f"SELECT COALESCE(SUM({col}), 0) FROM facts WHERE {' AND '.join(conditions)}"
        return float(self._conn.execute(sql, params).fetchone()[0])

    def delete(self, fact_id: str) -> bool:
        if self._read_only:
            raise RuntimeError("FactStore is read-only")
        cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0


# ── Internal utilities ────────────────────────────────────

def _extract_indexed_columns(
    schema: FactSchema,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "amount": payload.get("amount"),
        "merchant": payload.get("merchant"),
        "category": payload.get("category"),
        "time": payload.get("time"),
        "location": payload.get("location"),
    }


def _row_to_record(row: sqlite3.Row) -> FactRecord:
    return FactRecord(
        id=row["id"],
        schema_name=row["schema_name"],
        payload=json.loads(row["payload_json"]),
        created_at=row["created_at"],
        sensitivity=row["sensitivity"],
        ttl_seconds=row["ttl_seconds"],
        derived_from=row["derived_from"],
        source_type=row["source_type"],
    )
