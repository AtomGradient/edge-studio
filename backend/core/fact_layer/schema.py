# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# Supported field types. SQLite mapping:
#   str     → TEXT
#   int     → INTEGER
#   float   → REAL
#   datetime→ INTEGER (unix ms)
#   bool    → INTEGER (0/1)
FieldType = Literal["str", "int", "float", "datetime", "bool"]


@dataclass(frozen=True)
class FieldDef:
    type: FieldType
    required: bool = False
    indexed: bool = False          # True → promote to standalone SQL column + create index
    semantic: bool = False         # True → field participates in embedding (P1)
    description: str = ""          # Human-readable description, also used for LLM few-shot

    def validate(self, value: Any) -> None:
        if value is None:
            if self.required:
                raise ValueError("required field cannot be None")
            return

        type_checkers = {
            "str": lambda v: isinstance(v, str),
            "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "datetime": lambda v: isinstance(v, int),  # Already normalized to unix ms
            "bool": lambda v: isinstance(v, bool),
        }
        if not type_checkers[self.type](value):
            raise ValueError(f"expected {self.type}, got {type(value).__name__}: {value!r}")


@dataclass(frozen=True)
class FactSchema:
    name: str
    fields: Dict[str, FieldDef]
    primary_time_field: Optional[str] = None  # Default time field (for time range queries)
    description: str = ""

    def __post_init__(self) -> None:
        if self.primary_time_field is not None:
            pt = self.fields.get(self.primary_time_field)
            if pt is None:
                raise ValueError(
                    f"primary_time_field '{self.primary_time_field}' not in fields"
                )
            if pt.type != "datetime":
                raise ValueError(
                    f"primary_time_field must be datetime, got {pt.type}"
                )

    def validate_payload(self, payload: Dict[str, Any]) -> None:
        unknown = set(payload.keys()) - set(self.fields.keys())
        if unknown:
            raise ValueError(f"unknown fields for schema {self.name!r}: {unknown}")

        for field_name, field_def in self.fields.items():
            field_def.validate(payload.get(field_name))

    def indexed_fields(self) -> List[str]:
        return [n for n, f in self.fields.items() if f.indexed]


# ── Built-in schema: finance.expense ──────────────────────
#
# Fields aligned with iOS seed bundle schema:
#   amount / category / consumptionTime / description / location / id
#
# Design decisions:
#   - amount / category / time / location indexed (high-frequency exact queries)
#   - description not indexed but participates in semantic (P1 embedding)
#   - time uses field name "time" (normalized within schema), migration scripts handle
#     conversion from consumptionTime
FINANCE_EXPENSE = FactSchema(
    name="finance.expense",
    description="个人消费记录（记账条目）",
    primary_time_field="time",
    fields={
        "amount": FieldDef(
            type="float", required=True, indexed=True,
            description="消费金额（本币）",
        ),
        "merchant": FieldDef(
            type="str", indexed=True,
            description="商家名称（如'必胜客'、'邻几便利'）",
        ),
        "category": FieldDef(
            type="str", indexed=True,
            description="消费类别（如'餐饮'、'Transport'、'午餐'）",
        ),
        "time": FieldDef(
            type="datetime", required=True, indexed=True,
            description="消费时间（Unix ms）",
        ),
        "location": FieldDef(
            type="str", indexed=True,
            description="消费地点",
        ),
        "description": FieldDef(
            type="str", semantic=True,
            description="消费描述（自由文本，参与语义检索）",
        ),
    },
)


# ── Schema Registry ────────────────────────────────────
#
# Global registry. All persisted schemas must be registered first.
# To add a schema, just call register_schema(FactSchema(...)).
SCHEMA_REGISTRY: Dict[str, FactSchema] = {
    FINANCE_EXPENSE.name: FINANCE_EXPENSE,
}


def register_schema(schema: FactSchema) -> None:
    SCHEMA_REGISTRY[schema.name] = schema


def get_schema(name: str) -> FactSchema:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(
            f"schema {name!r} not registered. "
            f"Available: {sorted(SCHEMA_REGISTRY.keys())}"
        )
    return SCHEMA_REGISTRY[name]
