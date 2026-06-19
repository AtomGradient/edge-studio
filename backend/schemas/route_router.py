# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Schemas for R2.1 route-router artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


ROUTE_ROUTER_MANIFEST_SCHEMA_VERSION = "edgestudio.route_router_manifest.v0"
ROUTE_ROUTER_CALIBRATION_SCHEMA_VERSION = "edgestudio.route_router_calibration.v0"
ROUTE_ROUTER_EVAL_REPORT_SCHEMA_VERSION = "edgestudio.route_router_eval_report.v0"


class RouteRouterEncoderSpec(BaseModel):
    kind: str
    hidden_size: int = Field(gt=0)
    layer_index: int
    pooling: str
    base_model_id: str
    tokenizer_sha256: str

    @field_validator("base_model_id", "kind", "pooling", "tokenizer_sha256")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value


class RouteRouterMatrixSpec(BaseModel):
    file: str
    tensor: str
    bias_tensor: str | None = None
    shape: list[int]
    dtype: str

    @field_validator("file", "tensor", "dtype")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("shape")
    @classmethod
    def _valid_shape(cls, value: list[int]) -> list[int]:
        if not value or any(int(dim) <= 0 for dim in value):
            raise ValueError("shape must contain positive dimensions")
        return [int(dim) for dim in value]


class RouteRouterManifest(BaseModel):
    schema_version: str = ROUTE_ROUTER_MANIFEST_SCHEMA_VERSION
    router_type: str
    encoder: RouteRouterEncoderSpec
    intent_vocab: list[str]
    matrices: dict[str, RouteRouterMatrixSpec]
    calibration_file: str
    min_runtime_version: str
    training_run_id: str
    manifest_sha256: str
    fallback_chain: list[str]

    @field_validator(
        "schema_version",
        "router_type",
        "calibration_file",
        "min_runtime_version",
        "training_run_id",
        "manifest_sha256",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("intent_vocab", "fallback_chain")
    @classmethod
    def _non_empty_string_list(cls, value: list[str]) -> list[str]:
        out = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        if not out:
            raise ValueError("must contain at least one non-empty string")
        return out

    @field_validator("manifest_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        value = str(value or "").strip()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("manifest_sha256 must be a lowercase sha256 hex string")
        return value

    @model_validator(mode="after")
    def _valid_intent_matrix(self) -> "RouteRouterManifest":
        intent_matrix = self.matrices.get("intent")
        if intent_matrix is None:
            raise ValueError("matrices.intent is required")
        if intent_matrix.shape != [self.encoder.hidden_size, len(self.intent_vocab)]:
            raise ValueError("matrices.intent.shape must match encoder.hidden_size x intent_vocab")
        return self


class RouteRouterCalibration(BaseModel):
    schema_version: str = ROUTE_ROUTER_CALIBRATION_SCHEMA_VERSION
    intent_temperature: float = Field(gt=0)
    intent_thresholds: dict[str, float]
    tool_threshold_default: float = Field(ge=0, le=1)
    calibration_set_size: int = Field(ge=0)
    calibration_ece: float | None = Field(default=None, ge=0, le=1)

    @field_validator("schema_version")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("intent_thresholds")
    @classmethod
    def _valid_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        out = {}
        for key, raw in value.items():
            text = str(key or "").strip()
            if not text:
                raise ValueError("intent threshold keys must be non-empty")
            threshold = float(raw)
            if threshold < 0 or threshold > 1:
                raise ValueError("intent thresholds must be in [0, 1]")
            out[text] = threshold
        if not out:
            raise ValueError("intent_thresholds must not be empty")
        return out


class RouteRouterEvalReport(BaseModel):
    schema_version: str = ROUTE_ROUTER_EVAL_REPORT_SCHEMA_VERSION
    status: str
    training_run_id: str
    metrics: dict
    leakage_gate: dict

    @field_validator("schema_version", "status", "training_run_id")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value
