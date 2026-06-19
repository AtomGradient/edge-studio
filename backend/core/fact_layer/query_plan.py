# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Structured query plan types for Fact Layer execution.

This module intentionally contains no natural-language parsing or keyword
matching rules. Callers that need structured fact queries should construct a
plan from model/tool output and pass it to `StructuredIndexer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


StructuredQueryIntent = Literal[
    "sum",
    "count",
    "list_recent",
    "merchant_history",
    "unknown",
]


@dataclass(frozen=True)
class StructuredQueryPlan:
    """Model/tool-produced structured fact query plan."""

    intent: StructuredQueryIntent
    schema: str = "finance.expense"
    merchant: Optional[str] = None
    category: Optional[str] = None
    month: Optional[str] = None
    date: Optional[str] = None
    date_gte: Optional[str] = None
    date_lte: Optional[str] = None

    def has_narrow_filter(self) -> bool:
        """Return whether the plan has a time or entity constraint."""
        return any([
            self.merchant,
            self.category,
            self.month,
            self.date,
            self.date_gte,
            self.date_lte,
        ])
