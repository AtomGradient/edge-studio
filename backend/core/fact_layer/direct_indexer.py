# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .fact_store import FactStore, FactRecord
from .schema import get_schema


class DirectIndexer:

    def __init__(self, store: FactStore):
        self._store = store

    def record_fact(
        self,
        *,
        schema: str,
        payload: Dict[str, Any],
        created_at: Optional[int] = None,
        derived_from: Optional[str] = None,
        sensitivity: str = "private",
        ttl_seconds: Optional[int] = None,
        source_type: str = "user_device",
        id: Optional[str] = None,
    ) -> str:
        schema_def = get_schema(schema)

        # Determine created_at
        if created_at is None:
            pt = schema_def.primary_time_field
            if pt and pt in payload:
                created_at = payload[pt]
            else:
                created_at = int(time.time() * 1000)

        record = FactRecord.new(
            schema_name=schema,
            payload=payload,
            created_at=created_at,
            sensitivity=sensitivity,
            ttl_seconds=ttl_seconds,
            derived_from=derived_from,
            source_type=source_type,
            id=id,
        )
        return self._store.record(record)

    def record_batch(self, facts: List[Dict[str, Any]]) -> Dict[str, int]:
        records: List[FactRecord] = []
        for item in facts:
            schema = item["schema"]
            schema_def = get_schema(schema)
            created_at = item.get("created_at")
            if created_at is None:
                pt = schema_def.primary_time_field
                if pt and pt in item["payload"]:
                    created_at = item["payload"][pt]
                else:
                    created_at = int(time.time() * 1000)

            records.append(FactRecord.new(
                schema_name=schema,
                payload=item["payload"],
                created_at=created_at,
                sensitivity=item.get("sensitivity", "private"),
                ttl_seconds=item.get("ttl_seconds"),
                derived_from=item.get("derived_from"),
                source_type=item.get("source_type", "user_device"),
                id=item.get("id"),
            ))

        return self._store.record_many(records)
