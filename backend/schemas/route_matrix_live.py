# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Schemas for controlled route-matrix live audit events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


ROUTE_MATRIX_LIVE_DECISION_AUDIT_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_decision_audit.v0"
)


class RouteMatrixPredictionAudit(BaseModel):
    intent: str
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    threshold_passed: bool
    training_run_id: str | None = None

    @field_validator("intent", "training_run_id")
    @classmethod
    def _trim_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        return text or None

    @field_validator("intent")
    @classmethod
    def _intent_required(cls, value: str | None) -> str:
        if not value:
            raise ValueError("intent must be a non-empty string")
        return value


class RouteMatrixLiveDecisionAudit(BaseModel):
    schema_version: str = ROUTE_MATRIX_LIVE_DECISION_AUDIT_SCHEMA_VERSION
    case_id: str
    matrix_prediction: RouteMatrixPredictionAudit
    matrix_calibrated_confidence: float = Field(ge=0, le=1)
    evidence_available: bool
    evidence_route: dict[str, Any] | None = None
    final_decision_source: Literal["matrix", "evidence", "base"]
    fallback_reason: str | None = None
    shadow_mode_was: bool
    user_correction: dict[str, Any] | None = None

    @field_validator("schema_version", "case_id")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("fallback_reason")
    @classmethod
    def _trim_fallback_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        return text or None

    @model_validator(mode="after")
    def _fallback_reason_matches_source(self) -> "RouteMatrixLiveDecisionAudit":
        if self.final_decision_source == "matrix" and self.fallback_reason:
            raise ValueError("matrix final decisions must not include fallback_reason")
        if self.final_decision_source != "matrix" and not self.fallback_reason:
            raise ValueError("fallback decisions must include fallback_reason")
        return self
