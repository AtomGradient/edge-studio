# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from .fact_store import FactStore, FactRecord
from .query_plan import StructuredQueryPlan


@dataclass
class QueryResult:
    kind: Literal["sum", "count", "list", "empty"]
    plan: StructuredQueryPlan
    total: Optional[float] = None          # kind=sum
    count: Optional[int] = None            # kind=count
    records: List[FactRecord] = field(default_factory=list)  # kind=list
    matched_records: int = 0               # Total matched records (for count / list)

    def to_answer_hint(self) -> str:
        if self.kind == "empty":
            return ""

        scope = []
        if self.plan.date:
            scope.append(f"{self.plan.date}")
        if self.plan.month:
            scope.append(f"{self.plan.month} 月")
        if self.plan.date_gte and self.plan.date_lte:
            scope.append(f"{self.plan.date_gte}~{self.plan.date_lte}")
        if self.plan.merchant:
            scope.append(f"商家={self.plan.merchant}")
        if self.plan.category:
            scope.append(f"分类={self.plan.category}")
        scope_str = " ".join(scope) if scope else "全部记录"

        if self.kind == "sum":
            return (
                f"【精确查询】{scope_str} 总支出: ¥{self.total:.2f} "
                f"({self.matched_records} 条记录)"
            )
        if self.kind == "count":
            return f"【精确查询】{scope_str} 共 {self.count} 次"
        if self.kind == "list":
            lines = [f"【精确查询】{scope_str} 最近 {len(self.records)} 条:"]
            for r in self.records[:10]:
                amt = r.payload.get("amount")
                mch = r.payload.get("merchant", "?")
                t_ms = r.payload.get("time", 0)
                t_str = datetime.fromtimestamp(
                    t_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d") if t_ms else "?"
                lines.append(f"  - {t_str} | {mch} | ¥{amt:.2f}")
            return "\n".join(lines)
        return ""


class StructuredIndexer:

    def __init__(self, store: FactStore):
        self._store = store

    # ── Case A: Exact field query ─────────────────────────

    def case_a_exact(
        self,
        *,
        schema: str,
        merchant: Optional[str] = None,
        category: Optional[str] = None,
        date: Optional[str] = None,          # "YYYY-MM-DD"
        month: Optional[str] = None,          # "YYYY-MM"
        amount_gte: Optional[float] = None,
        amount_lte: Optional[float] = None,
        location: Optional[str] = None,
        limit: int = 10,
    ) -> List[FactRecord]:
        time_gte, time_lte = _date_to_ms_range(date=date, month=month)

        return self._store.query(
            schema=schema,
            merchant=merchant,
            category=category,
            time_gte=time_gte,
            time_lte=time_lte,
            amount_gte=amount_gte,
            amount_lte=amount_lte,
            location=location,
            limit=limit,
        )

    # ── Case B: Structured aggregation ────────────────────

    def case_b_sum(
        self,
        *,
        schema: str,
        field: str = "amount",
        category: Optional[str] = None,
        merchant: Optional[str] = None,
        date: Optional[str] = None,
        month: Optional[str] = None,
        date_gte: Optional[str] = None,
        date_lte: Optional[str] = None,
    ) -> float:
        time_gte, time_lte = _date_to_ms_range(date=date, month=month)
        # date_gte/date_lte override month range (week/custom range)
        if date_gte:
            from datetime import datetime as _dt, timezone as _tz
            time_gte = int(_dt.strptime(date_gte, "%Y-%m-%d")
                           .replace(tzinfo=_tz.utc).timestamp() * 1000)
        if date_lte:
            from datetime import datetime as _dt, timezone as _tz
            time_lte = int(_dt.strptime(date_lte, "%Y-%m-%d")
                           .replace(tzinfo=_tz.utc).timestamp() * 1000) + 86_400_000 - 1

        return self._store.aggregate_sum(
            schema=schema,
            field=field,
            category=category,
            merchant_like=merchant,  # LIKE tolerates merchant name variants
            time_gte=time_gte,
            time_lte=time_lte,
        )

    def case_b_count(
        self,
        *,
        schema: str,
        category: Optional[str] = None,
        merchant: Optional[str] = None,
        date: Optional[str] = None,
        month: Optional[str] = None,
        date_gte: Optional[str] = None,
        date_lte: Optional[str] = None,
    ) -> int:
        time_gte, time_lte = _date_to_ms_range(date=date, month=month)
        if date_gte:
            from datetime import datetime as _dt, timezone as _tz
            time_gte = int(_dt.strptime(date_gte, "%Y-%m-%d")
                           .replace(tzinfo=_tz.utc).timestamp() * 1000)
        if date_lte:
            from datetime import datetime as _dt, timezone as _tz
            time_lte = int(_dt.strptime(date_lte, "%Y-%m-%d")
                           .replace(tzinfo=_tz.utc).timestamp() * 1000) + 86_400_000 - 1

        return len(self._store.query(
            schema=schema,
            category=category,
            merchant_like=merchant,
            time_gte=time_gte,
            time_lte=time_lte,
            limit=1_000_000,
        ))

    # ── Case C: Form completion ───────────────────────────

    # ── Phase D: Structured plan dispatch entry ──────

    def execute_plan(
        self,
        plan: StructuredQueryPlan,
        *,
        limit: int = 20,
    ) -> QueryResult:
        if plan.intent == "unknown":
            return QueryResult(kind="empty", plan=plan)

        if plan.intent == "sum":
            total = self.case_b_sum(
                schema=plan.schema,
                category=plan.category,
                merchant=plan.merchant,
                date=plan.date,
                month=plan.month,
                date_gte=plan.date_gte,
                date_lte=plan.date_lte,
            )
            matched = self.case_b_count(
                schema=plan.schema,
                category=plan.category,
                merchant=plan.merchant,
                date=plan.date,
                month=plan.month,
                date_gte=plan.date_gte,
                date_lte=plan.date_lte,
            )
            return QueryResult(kind="sum", plan=plan, total=total,
                               matched_records=matched)

        if plan.intent == "count":
            count = self.case_b_count(
                schema=plan.schema,
                category=plan.category,
                merchant=plan.merchant,
                date=plan.date,
                month=plan.month,
                date_gte=plan.date_gte,
                date_lte=plan.date_lte,
            )
            return QueryResult(kind="count", plan=plan, count=count,
                               matched_records=count)

        # list_recent / merchant_history → case_a_exact
        records = self.case_a_exact(
            schema=plan.schema,
            merchant=plan.merchant,
            category=plan.category,
            date=plan.date,
            month=plan.month,
            limit=limit,
        )
        # Backend currently only supports month/date ranges, week falls back:
        # If week range specified → manually filter records
        if plan.date_gte and plan.date_lte:
            from datetime import datetime as _dt, timezone as _tz
            start_ms = int(_dt.strptime(plan.date_gte, "%Y-%m-%d")
                           .replace(tzinfo=_tz.utc).timestamp() * 1000)
            end_ms = int(_dt.strptime(plan.date_lte, "%Y-%m-%d")
                         .replace(tzinfo=_tz.utc).timestamp() * 1000) + 86400000 - 1
            records = [r for r in records
                       if start_ms <= r.payload.get("time", 0) <= end_ms]
        return QueryResult(kind="list", plan=plan, records=records,
                           matched_records=len(records))

    # ── Case C: Form completion ───────────────────────────

    def case_c_merchant_prefix(
        self,
        *,
        schema: str,
        prefix: str,
        limit: int = 10,
    ) -> List[str]:
        if not prefix:
            return []

        results = self._store.query(
            schema=schema,
            merchant_like=prefix,
            limit=limit * 10,  # Fetch extra then deduplicate
        )
        # Deduplicate by merchant + sort by frequency
        freq: Dict[str, int] = {}
        for r in results:
            m = r.payload.get("merchant")
            if m and m.startswith(prefix):
                freq[m] = freq.get(m, 0) + 1

        return sorted(freq.keys(), key=lambda k: -freq[k])[:limit]


# ── Time conversion utilities ─────────────────────────────

def _date_to_ms_range(
    *,
    date: Optional[str] = None,
    month: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int]]:
    if date:
        dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start_ms = int(dt.timestamp() * 1000)
        end_ms = start_ms + 86_400_000 - 1
        return (start_ms, end_ms)

    if month:
        dt = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
        start_ms = int(dt.timestamp() * 1000)
        # Month end: 1st of next month 00:00 - 1 ms
        year, mon = dt.year, dt.month
        if mon == 12:
            next_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_dt = datetime(year, mon + 1, 1, tzinfo=timezone.utc)
        end_ms = int(next_dt.timestamp() * 1000) - 1
        return (start_ms, end_ms)

    return (None, None)


def iso_to_ms(iso: str) -> int:
    # Python 3.11+ fromisoformat compatibility (Z suffix supported since 3.11)
    iso_normalized = iso.replace("Z", "+00:00") if iso.endswith("Z") else iso
    dt = datetime.fromisoformat(iso_normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
