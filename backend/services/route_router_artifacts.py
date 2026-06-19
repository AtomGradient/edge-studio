# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""R2.1 route-router artifact contract helpers.

This module does not train a router and does not execute routing. It only
builds the versioned artifact contract used by the future matrix router.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.schemas.route_router import (
    ROUTE_ROUTER_CALIBRATION_SCHEMA_VERSION,
    ROUTE_ROUTER_EVAL_REPORT_SCHEMA_VERSION,
    ROUTE_ROUTER_MANIFEST_SCHEMA_VERSION,
    RouteRouterCalibration,
    RouteRouterEvalReport,
    RouteRouterManifest,
)

ROUTE_ROUTER_MANIFEST_NAME = "route_router_manifest.json"
ROUTE_ROUTER_INTENT_MATRIX_NAME = "route_intent_matrix.safetensors"
ROUTE_ROUTER_CALIBRATION_NAME = "route_calibration.json"
ROUTE_ROUTER_EVAL_REPORT_NAME = "route_eval_report.json"
ROUTE_ROUTER_EVAL_LEAKAGE_REPORT_NAME = "route_eval_leakage_report.json"
ROUTE_ROUTER_CALIBRATION_SET_NAME = "route_calibration_set.jsonl"
ROUTE_ROUTER_TRAINING_SPLITS_NAME = "route_training_splits.json"

DEFAULT_INTENT_VOCAB = [
    "base_chat",
    "exact_fact",
    "aggregate_fact",
    "app_action",
    "user_profile",
    "mixed",
]
DEFAULT_FALLBACK_CHAIN = ["matrix", "evidence_matcher", "base_router"]


def canonical_route_router_base_model_id(base_model_id: str) -> str:
    return str(base_model_id or "").strip().lower()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def route_router_manifest_sha256(manifest_without_sha: dict[str, Any]) -> str:
    body = dict(manifest_without_sha)
    body.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def build_route_router_calibration(
    *,
    intent_thresholds: dict[str, float] | None = None,
    intent_temperature: float = 1.0,
    tool_threshold_default: float = 0.55,
    calibration_set_size: int = 0,
    calibration_ece: float | None = None,
) -> dict[str, Any]:
    thresholds = intent_thresholds or {
        "base_chat": 0.55,
        "exact_fact": 0.60,
        "aggregate_fact": 0.60,
        "app_action": 0.65,
        "user_profile": 0.55,
        "mixed": 0.50,
    }
    calibration = RouteRouterCalibration(
        schema_version=ROUTE_ROUTER_CALIBRATION_SCHEMA_VERSION,
        intent_temperature=intent_temperature,
        intent_thresholds=thresholds,
        tool_threshold_default=tool_threshold_default,
        calibration_set_size=calibration_set_size,
        calibration_ece=calibration_ece,
    )
    return calibration.model_dump(mode="json")


def build_route_router_manifest(
    *,
    training_run_id: str,
    base_model_id: str,
    tokenizer_sha256: str,
    hidden_size: int,
    intent_vocab: list[str] | None = None,
    min_runtime_version: str = "0.9.0",
    encoder_kind: str = "base_model_last_hidden",
    layer_index: int = -1,
    pooling: str = "mean_excluding_special",
    matrix_file: str = ROUTE_ROUTER_INTENT_MATRIX_NAME,
    calibration_file: str = ROUTE_ROUTER_CALIBRATION_NAME,
    dtype: str = "float16",
    fallback_chain: list[str] | None = None,
) -> dict[str, Any]:
    vocab = intent_vocab or list(DEFAULT_INTENT_VOCAB)
    manifest_without_sha = {
        "schema_version": ROUTE_ROUTER_MANIFEST_SCHEMA_VERSION,
        "router_type": "matrix_v0",
        "encoder": {
            "kind": encoder_kind,
            "hidden_size": int(hidden_size),
            "layer_index": int(layer_index),
            "pooling": pooling,
            "base_model_id": canonical_route_router_base_model_id(base_model_id),
            "tokenizer_sha256": tokenizer_sha256,
        },
        "intent_vocab": vocab,
        "matrices": {
            "intent": {
                "file": matrix_file,
                "tensor": "intent_weights",
                "bias_tensor": "intent_bias",
                "shape": [int(hidden_size), len(vocab)],
                "dtype": dtype,
            }
        },
        "calibration_file": calibration_file,
        "min_runtime_version": min_runtime_version,
        "training_run_id": training_run_id,
        "fallback_chain": fallback_chain or list(DEFAULT_FALLBACK_CHAIN),
    }
    manifest = {
        **manifest_without_sha,
        "manifest_sha256": route_router_manifest_sha256(manifest_without_sha),
    }
    return RouteRouterManifest(**manifest).model_dump(mode="json")


def write_route_router_contract_artifacts(
    *,
    output_dir: Path,
    training_run_id: str,
    base_model_id: str,
    tokenizer_sha256: str,
    hidden_size: int,
    layer_index: int = -1,
    intent_matrix_bytes: bytes,
    intent_vocab: list[str] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the minimal runtime route-router artifact contract.

    The caller supplies matrix bytes. In R2.1 Phase 1 tests these bytes are a
    placeholder; real training writes a safetensors payload later.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_payload = calibration or build_route_router_calibration()
    RouteRouterCalibration(**calibration_payload)
    manifest = build_route_router_manifest(
        training_run_id=training_run_id,
        base_model_id=base_model_id,
        tokenizer_sha256=tokenizer_sha256,
        hidden_size=hidden_size,
        layer_index=layer_index,
        intent_vocab=intent_vocab,
    )
    files = {
        ROUTE_ROUTER_MANIFEST_NAME: canonical_json_bytes(manifest),
        ROUTE_ROUTER_CALIBRATION_NAME: canonical_json_bytes(calibration_payload),
        ROUTE_ROUTER_INTENT_MATRIX_NAME: bytes(intent_matrix_bytes),
    }
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    return {
        "ok": True,
        "schema_version": ROUTE_ROUTER_MANIFEST_SCHEMA_VERSION,
        "status": "written",
        "output_dir": str(output_dir),
        "manifest": manifest,
        "files": [
            {
                "name": name,
                "path": str(output_dir / name),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
    }


def build_route_router_eval_report(
    *,
    training_run_id: str,
    status: str,
    metrics: dict[str, Any],
    leakage_gate: dict[str, Any],
) -> dict[str, Any]:
    report = RouteRouterEvalReport(
        schema_version=ROUTE_ROUTER_EVAL_REPORT_SCHEMA_VERSION,
        status=status,
        training_run_id=training_run_id,
        metrics=metrics,
        leakage_gate=leakage_gate,
    )
    return report.model_dump(mode="json")
