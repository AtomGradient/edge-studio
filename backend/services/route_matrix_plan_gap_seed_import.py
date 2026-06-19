# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Materialize reviewed route-matrix plan-gap seeds into learner feedstock.

This module is training-side only. It consumes the explicit dry-run receipt
from `generate_route_matrix_plan_gap_seed_candidates(...)`, stores its already
gated route/action response into an isolated EventStore, then writes the normal
`route_action_policy_dataset.jsonl` learner feed. It never writes runtime
artifacts and never enables matrix routing.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.route_action_training_events import (
    ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
    RouteActionEvalLeakageError,
    store_route_action_training_events,
    write_route_action_learner_dataset_jsonl,
)
from backend.stores.event_store import EventStore


ROUTE_MATRIX_PLAN_GAP_SEED_IMPORT_SCHEMA_VERSION = (
    "edgestudio.route_matrix_plan_gap_seed_import.v0"
)


def import_route_matrix_plan_gap_seed_candidates(
    *,
    plan_gap_generation: dict[str, Any],
    output_dir: Path,
    peer_id: str | None = None,
    event_store_path: Path | None = None,
    event_store: EventStore | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Store reviewed plan-gap seeds and write a gated learner dataset.

    The input must be a successful `plan_gap_seed_candidates_ready` receipt.
    The default EventStore is intentionally not used; callers either pass an
    explicit store or this function creates an isolated SQLite file under
    `output_dir`.
    """

    generated_at = _utc_now()
    try:
        seed_receipt = _ready_seed_receipt(plan_gap_generation)
        route_action_response = _route_action_response(seed_receipt)
        app_id = _required_text(seed_receipt.get("app_id"), "seed_receipt.app_id")
        rpp_run_id = _required_text(
            route_action_response["result"].get("rpp_run_id")
            or seed_receipt.get("seed_run_id"),
            "route_action_response.result.rpp_run_id",
        )
    except (TypeError, ValueError) as exc:
        return _error(
            status="invalid_input",
            code="invalid_plan_gap_seed_import_input",
            message=str(exc),
            details={},
            generated_at=generated_at,
        )

    normalized_output_dir = Path(output_dir)
    normalized_output_dir.mkdir(parents=True, exist_ok=True)
    effective_peer_id = _text(peer_id) or f"route-seed:{app_id}"
    effective_event_store_path = Path(
        event_store_path or (normalized_output_dir / "route_action_seed_events.sqlite")
    )
    owns_store = event_store is None
    store = event_store or EventStore(effective_event_store_path)

    try:
        storage = store_route_action_training_events(
            route_action_response,
            peer_id=effective_peer_id,
            event_store=store,
            timestamp_ms=_timestamp_ms(seed_receipt),
            tool_registry=(
                seed_receipt.get("tool_registry")
                if isinstance(seed_receipt.get("tool_registry"), list)
                else None
            ),
        )
        dataset = write_route_action_learner_dataset_jsonl(
            peer_id=effective_peer_id,
            output_dir=normalized_output_dir,
            rpp_run_id=rpp_run_id,
            event_store=store,
            limit=limit,
        )
    except RouteActionEvalLeakageError as exc:
        return _error(
            status="eval_leakage_blocked",
            code="route_action_eval_leakage_blocked",
            message=str(exc),
            details={"eval_leakage_gate": exc.report},
            generated_at=generated_at,
            app_id=app_id,
            peer_id=effective_peer_id,
            seed_run_id=_text(seed_receipt.get("seed_run_id")),
        )
    except Exception as exc:  # noqa: BLE001
        return _error(
            status="seed_import_failed",
            code="route_matrix_plan_gap_seed_import_failed",
            message=str(exc),
            details={},
            generated_at=generated_at,
            app_id=app_id,
            peer_id=effective_peer_id,
            seed_run_id=_text(seed_receipt.get("seed_run_id")),
        )
    finally:
        if owns_store:
            store.close()

    if not isinstance(dataset, dict):
        return {
            "ok": False,
            "schema_version": ROUTE_MATRIX_PLAN_GAP_SEED_IMPORT_SCHEMA_VERSION,
            "status": "learner_dataset_empty",
            "app_id": app_id,
            "peer_id": effective_peer_id,
            "seed_run_id": _text(seed_receipt.get("seed_run_id")),
            "rpp_run_id": rpp_run_id,
            "storage": storage,
            "learner_dataset": None,
            "error": {
                "code": "learner_dataset_empty",
                "message": "No learner dataset samples were produced from plan-gap seeds.",
                "retryable": False,
                "details": {},
            },
            "audit": _audit(
                generated_at=generated_at,
                output_dir=normalized_output_dir,
                event_store_path=effective_event_store_path,
                seed_receipt=seed_receipt,
                storage=storage,
                dataset=None,
            ),
        }
    if dataset.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": ROUTE_MATRIX_PLAN_GAP_SEED_IMPORT_SCHEMA_VERSION,
            "status": dataset.get("status") or "learner_dataset_blocked",
            "app_id": app_id,
            "peer_id": effective_peer_id,
            "seed_run_id": _text(seed_receipt.get("seed_run_id")),
            "rpp_run_id": rpp_run_id,
            "storage": storage,
            "learner_dataset": dataset,
            "error": {
                "code": dataset.get("status") or "learner_dataset_blocked",
                "message": "Learner dataset gate blocked plan-gap seeds.",
                "retryable": False,
                "details": {
                    "eval_leakage_gate": dataset.get("eval_leakage_gate"),
                },
            },
            "audit": _audit(
                generated_at=generated_at,
                output_dir=normalized_output_dir,
                event_store_path=effective_event_store_path,
                seed_receipt=seed_receipt,
                storage=storage,
                dataset=dataset,
            ),
        }

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_PLAN_GAP_SEED_IMPORT_SCHEMA_VERSION,
        "status": "learner_dataset_written",
        "app_id": app_id,
        "peer_id": effective_peer_id,
        "seed_run_id": _text(seed_receipt.get("seed_run_id")),
        "rpp_run_id": rpp_run_id,
        "storage": storage,
        "learner_dataset": dataset,
        "error": None,
        "audit": _audit(
            generated_at=generated_at,
            output_dir=normalized_output_dir,
            event_store_path=effective_event_store_path,
            seed_receipt=seed_receipt,
            storage=storage,
            dataset=dataset,
        ),
    }


def _ready_seed_receipt(plan_gap_generation: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan_gap_generation, dict):
        raise TypeError("plan_gap_generation must be an object")
    if plan_gap_generation.get("ok") is not True:
        raise ValueError("plan_gap_generation.ok must be true")
    if plan_gap_generation.get("status") != "plan_gap_seed_candidates_ready":
        raise ValueError(
            "plan_gap_generation.status must be plan_gap_seed_candidates_ready"
        )
    result = plan_gap_generation.get("result")
    if not isinstance(result, dict):
        raise ValueError("plan_gap_generation.result must be an object")
    seed_receipt = result.get("route_action_seed_candidates")
    if not isinstance(seed_receipt, dict):
        raise ValueError(
            "plan_gap_generation.result.route_action_seed_candidates missing"
        )
    if seed_receipt.get("ok") is not True:
        raise ValueError("route_action_seed_candidates.ok must be true")
    if seed_receipt.get("status") != "seed_candidates_ready":
        raise ValueError(
            "route_action_seed_candidates.status must be seed_candidates_ready"
        )
    return seed_receipt


def _route_action_response(seed_receipt: dict[str, Any]) -> dict[str, Any]:
    value = seed_receipt.get("route_action_response")
    if not isinstance(value, dict):
        raise ValueError("route_action_seed_candidates.route_action_response missing")
    if value.get("ok") is not True:
        raise ValueError("route_action_response.ok must be true")
    result = value.get("result")
    if not isinstance(result, dict):
        raise ValueError("route_action_response.result must be an object")
    pairs = result.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("route_action_response.result.pairs must be non-empty")
    return value


def _timestamp_ms(seed_receipt: dict[str, Any]) -> int:
    audit = seed_receipt.get("audit") if isinstance(seed_receipt.get("audit"), dict) else {}
    value = audit.get("generated_at_ms")
    if isinstance(value, bool):
        return int(time.time() * 1000)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(time.time() * 1000)
    return parsed if parsed > 0 else int(time.time() * 1000)


def _audit(
    *,
    generated_at: str,
    output_dir: Path,
    event_store_path: Path,
    seed_receipt: dict[str, Any],
    storage: dict[str, Any] | None,
    dataset: dict[str, Any] | None,
) -> dict[str, Any]:
    preview = seed_receipt.get("preview") if isinstance(seed_receipt.get("preview"), dict) else {}
    storage_audit = (
        storage.get("audit")
        if isinstance(storage, dict) and isinstance(storage.get("audit"), dict)
        else {}
    )
    return {
        "schema_version": "edgestudio.route_matrix_plan_gap_seed_import_audit.v0",
        "generated_at": generated_at,
        "training_side_only": True,
        "writes_events": True,
        "writes_runtime_artifacts": False,
        "source_schema_version": seed_receipt.get("schema_version"),
        "source_status": seed_receipt.get("status"),
        "source_fingerprint": _fingerprint(seed_receipt),
        "candidate_count": len(seed_receipt.get("candidates") or []),
        "output_dir": str(output_dir),
        "event_store_path": str(event_store_path),
        "learner_dataset_schema_version": ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
        "preview_event_ids_fingerprint": preview.get("event_ids_fingerprint"),
        "stored_event_ids_fingerprint": storage_audit.get("event_ids_fingerprint"),
        "preview_event_ids_match": (
            preview.get("event_ids_fingerprint") is not None
            and preview.get("event_ids_fingerprint")
            == storage_audit.get("event_ids_fingerprint")
        ),
        "sample_count": (
            dataset.get("sample_count")
            if isinstance(dataset, dict)
            else 0
        ),
    }


def _error(
    *,
    status: str,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
    app_id: str | None = None,
    peer_id: str | None = None,
    seed_run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_PLAN_GAP_SEED_IMPORT_SCHEMA_VERSION,
        "status": status,
        "app_id": app_id,
        "peer_id": peer_id,
        "seed_run_id": seed_run_id,
        "storage": None,
        "learner_dataset": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_plan_gap_seed_import_audit.v0",
            "generated_at": generated_at,
            "training_side_only": True,
            "writes_events": False,
            "writes_runtime_artifacts": False,
        },
    }


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
