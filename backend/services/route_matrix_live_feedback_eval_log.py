# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build live-feedback review feedstock from device eval.log traces."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.services.device_eval_log_parser import (
    build_route_matrix_live_decision_eval_run_from_eval_log,
)
from backend.services.route_matrix_live_audit import (
    build_route_matrix_live_feedback_review_request_from_event_store,
    store_route_matrix_live_decision_events_from_eval_run,
)


ROUTE_MATRIX_LIVE_FEEDBACK_EVAL_LOG_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_eval_log.v0"
)


def build_route_matrix_live_feedback_events_from_eval_log(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Store live-decision audits parsed from eval.log and build a review request.

    This helper is the repeatable host-side bridge from device logs to the
    explicit live-feedback review path. It writes only liveDecisionAudit events;
    it never writes trainingSample tags, learner datasets, or runtime artifacts.
    """

    generated_at_ms = int(time.time() * 1000)
    if not isinstance(request, dict):
        return _error(
            status="invalid_input",
            code="invalid_input",
            message="request must be an object",
            generated_at_ms=generated_at_ms,
            details={"received_type": type(request).__name__},
        )

    try:
        run_id = _required_text(request.get("run_id"), "run_id")
        app_id = _required_text(request.get("app_id"), "app_id")
        peer_id = _required_text(request.get("peer_id"), "peer_id")
        event_store_path = _path(request.get("event_store_path"), "event_store_path")
        event_store_path.parent.mkdir(parents=True, exist_ok=True)
        limit = _positive_int(request.get("limit"), 1000, "limit")
        max_cases = _positive_int(request.get("max_cases"), 50, "max_cases")
        timestamp_ms = _optional_int(request.get("timestamp_ms"))
    except (TypeError, ValueError) as exc:
        return _error(
            status="invalid_input",
            code="invalid_input",
            message=str(exc),
            generated_at_ms=generated_at_ms,
        )

    log_request = {
        "run_id": run_id,
        "subject": {"app_id": app_id, "peer_id": peer_id},
    }
    if request.get("eval_log") is not None:
        log_request["eval_log"] = request.get("eval_log")
    if request.get("lines") is not None:
        log_request["lines"] = request.get("lines")
    if request.get("device_autorun_run_id") or request.get("autorun_run_id"):
        log_request["device_autorun_run_id"] = (
            request.get("device_autorun_run_id") or request.get("autorun_run_id")
        )

    log_ingest = build_route_matrix_live_decision_eval_run_from_eval_log(log_request)
    if log_ingest.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_EVAL_LOG_SCHEMA_VERSION,
            "status": "eval_log_ingest_failed",
            "result": {
                "run_id": run_id,
                "app_id": app_id,
                "peer_id": peer_id,
                "event_store_path": str(event_store_path),
                "log_ingest": log_ingest,
                "store": None,
                "review_request": None,
                "summary": {},
            },
            "error": log_ingest.get("error"),
            "audit": _audit(
                generated_at_ms=generated_at_ms,
                event_count=0,
                inserted_count=0,
            ),
        }

    eval_run = log_ingest["result"]["eval_run"]
    effective_timestamp_ms = timestamp_ms or _stable_timestamp_ms(run_id, eval_run)
    try:
        store_receipt = store_route_matrix_live_decision_events_from_eval_run(
            eval_run,
            event_store_path=event_store_path,
            app_id=app_id,
            peer_id=peer_id,
            timestamp_ms=effective_timestamp_ms,
        )
        review_request = build_route_matrix_live_feedback_review_request_from_event_store(
            run_id=run_id,
            app_id=app_id,
            event_store_path=event_store_path,
            peer_id=peer_id,
            limit=limit,
            max_cases=max_cases,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(
            status="live_feedback_event_store_failed",
            code="live_feedback_event_store_failed",
            message=str(exc),
            generated_at_ms=generated_at_ms,
            details={"event_store_path": str(event_store_path)},
        )

    summary = _summary(log_ingest, store_receipt, review_request)
    status = _status(summary)
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_EVAL_LOG_SCHEMA_VERSION,
        "status": status,
        "result": {
            "run_id": run_id,
            "app_id": app_id,
            "peer_id": peer_id,
            "event_store_path": str(event_store_path),
            "log_ingest": log_ingest,
            "store": store_receipt,
            "review_request": review_request,
            "summary": summary,
        },
        "error": None,
        "audit": _audit(
            generated_at_ms=generated_at_ms,
            event_count=store_receipt.get("event_count", 0),
            inserted_count=store_receipt.get("inserted_count", 0),
            timestamp_ms=effective_timestamp_ms,
        ),
    }


def _summary(
    log_ingest: dict[str, Any],
    store_receipt: dict[str, Any],
    review_request: dict[str, Any],
) -> dict[str, Any]:
    log_summary = (
        log_ingest.get("result", {}).get("summary", {})
        if isinstance(log_ingest.get("result"), dict)
        else {}
    )
    review_summary = (
        review_request.get("result", {}).get("summary", {})
        if isinstance(review_request.get("result"), dict)
        else {}
    )
    return {
        "turn_count": int(log_summary.get("turn_count") or 0),
        "live_decision_count": int(log_summary.get("live_decision_count") or 0),
        "corrected_live_decision_count": int(
            log_summary.get("corrected_live_decision_count") or 0
        ),
        "stored_event_count": int(store_receipt.get("event_count") or 0),
        "inserted_event_count": int(store_receipt.get("inserted_count") or 0),
        "duplicate_event_count": int(store_receipt.get("duplicate_count") or 0),
        "review_case_count": int(review_summary.get("review_case_count") or 0),
        "ready_for_host_model_review": (
            review_summary.get("ready_for_host_model_review") is True
        ),
        "ready_for_learner_dataset": False,
        "ready_for_live_routing": False,
    }


def _status(summary: dict[str, Any]) -> str:
    if summary["ready_for_host_model_review"] is True:
        return "live_feedback_review_request_ready"
    if summary["live_decision_count"] <= 0:
        return "no_live_decision_events"
    if summary["corrected_live_decision_count"] <= 0:
        return "no_corrected_live_decisions"
    return "no_reviewable_live_feedback_corrections"


def _audit(
    *,
    generated_at_ms: int,
    event_count: int,
    inserted_count: int,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    audit = {
        "method": "build_route_matrix_live_feedback_events_from_eval_log",
        "generated_at_ms": generated_at_ms,
        "training_side_only": True,
        "writes_events": event_count > 0,
        "writes_training_sample_tags": False,
        "writes_runtime_artifacts": False,
        "event_count": event_count,
        "inserted_count": inserted_count,
    }
    if timestamp_ms is not None:
        audit["event_timestamp_base_ms"] = timestamp_ms
    return audit


def _error(
    *,
    status: str,
    code: str,
    message: str,
    generated_at_ms: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_EVAL_LOG_SCHEMA_VERSION,
        "status": status,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": dict(details or {}),
        },
        "audit": _audit(
            generated_at_ms=generated_at_ms,
            event_count=0,
            inserted_count=0,
        ),
    }


def _stable_timestamp_ms(run_id: str, eval_run: dict[str, Any]) -> int:
    correction_timestamps: list[int] = []
    for observation in eval_run.get("observations") or []:
        if not isinstance(observation, dict):
            continue
        decision = observation.get("route_matrix_live_decision")
        if not isinstance(decision, dict):
            continue
        correction = decision.get("user_correction")
        if isinstance(correction, dict) and isinstance(correction.get("created_at_ms"), int):
            correction_timestamps.append(int(correction["created_at_ms"]))
    if correction_timestamps:
        return min(correction_timestamps)
    digest = hashlib.sha256(
        json.dumps(
            {
                "run_id": run_id,
                "observations": eval_run.get("observations") or [],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return 1_700_000_000_000 + (int(digest[:10], 16) % 100_000_000_000)


def _path(value: Any, field: str) -> Path:
    text = _required_text(value, field)
    return Path(text).expanduser()


def _positive_int(value: Any, default: int, field: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamp_ms must be an integer") from exc


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text
