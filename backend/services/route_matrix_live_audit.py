# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Validation and EventStore helpers for route-matrix live decisions."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from backend.schemas.route_matrix_live import RouteMatrixLiveDecisionAudit
from backend.stores.event_store import DataEvent, EventStore


ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE = "route_matrix_live_decision"
ROUTE_MATRIX_LIVE_DECISION_EVENT_TAGS = (
    "liveDecisionAudit",
    "route_matrix_live_decision",
    "matrix_router",
)
ROUTE_MATRIX_LIVE_FEEDBACK_IMPORT_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_import.v0"
)
ROUTE_MATRIX_LIVE_FEEDBACK_REVIEW_REQUEST_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_review_request.v0"
)


def validate_route_matrix_live_decision_audit(value: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical JSON-safe audit payload or raise ValueError."""

    audit = RouteMatrixLiveDecisionAudit.model_validate(value)
    return audit.model_dump(mode="json", exclude_none=False)


def build_route_matrix_live_decision_event(
    value: dict[str, Any],
    *,
    app_id: str,
    peer_id: str,
    timestamp_ms: int,
) -> DataEvent:
    """Build an EventStore event for later feedback-loop training."""

    audit = validate_route_matrix_live_decision_audit(value)
    payload_bytes = json.dumps(
        audit,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return DataEvent(
        id=_event_id(
            app_id=app_id,
            peer_id=peer_id,
            timestamp_ms=timestamp_ms,
            payload_bytes=payload_bytes,
        ),
        timestamp_ms=int(timestamp_ms),
        app_id=str(app_id or "").strip(),
        event_type=ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
        payload=payload_bytes,
        tags=list(ROUTE_MATRIX_LIVE_DECISION_EVENT_TAGS),
        source_peer_id=str(peer_id or "").strip() or None,
    )


def build_route_matrix_live_decision_events_from_eval_run(
    eval_run: dict[str, Any],
    *,
    app_id: str | None = None,
    peer_id: str | None = None,
    timestamp_ms: int | None = None,
) -> list[DataEvent]:
    """Build EventStore events from parsed live-decision observations.

    This only preserves live routing facts for later review. It does not turn a
    live decision into a route/action training target.
    """

    if not isinstance(eval_run, dict):
        raise TypeError("eval_run must be an object")
    subject = eval_run.get("subject") if isinstance(eval_run.get("subject"), dict) else {}
    effective_app_id = _required_text(app_id or subject.get("app_id"), "app_id")
    effective_peer_id = _required_text(peer_id or subject.get("peer_id"), "peer_id")
    base_timestamp_ms = int(timestamp_ms if timestamp_ms is not None else time.time() * 1000)
    observations = eval_run.get("observations")
    if not isinstance(observations, list):
        raise TypeError("eval_run.observations must be a list")

    events: list[DataEvent] = []
    for offset, observation in enumerate(observations):
        if not isinstance(observation, dict):
            continue
        audit = observation.get("route_matrix_live_decision")
        if not isinstance(audit, dict):
            continue
        events.append(
            build_route_matrix_live_decision_event(
                audit,
                app_id=effective_app_id,
                peer_id=effective_peer_id,
                timestamp_ms=base_timestamp_ms + offset,
            )
        )
    return events


def store_route_matrix_live_decision_events_from_eval_run(
    eval_run: dict[str, Any],
    *,
    event_store: EventStore | None = None,
    event_store_path: Path | None = None,
    app_id: str | None = None,
    peer_id: str | None = None,
    timestamp_ms: int | None = None,
) -> dict[str, Any]:
    """Store parsed live-decision audit events in an explicit EventStore."""

    owns_store = event_store is None
    store = event_store or EventStore(event_store_path)
    try:
        events = build_route_matrix_live_decision_events_from_eval_run(
            eval_run,
            app_id=app_id,
            peer_id=peer_id,
            timestamp_ms=timestamp_ms,
        )
        received_ids, is_new_flags = store.insert_batch(events)
    finally:
        if owns_store:
            store.close()

    inserted_count = sum(1 for flag in is_new_flags if flag)
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_IMPORT_SCHEMA_VERSION,
        "status": "stored" if events else "no_live_decision_events",
        "event_count": len(events),
        "inserted_count": inserted_count,
        "duplicate_count": len(events) - inserted_count,
        "event_ids": received_ids,
        "audit": {
            "method": "store_route_matrix_live_decision_events_from_eval_run",
            "training_side_only": True,
            "writes_runtime_artifacts": False,
            "event_type": ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
        },
    }


def build_route_matrix_live_feedback_review_request_from_event_store(
    *,
    run_id: str,
    app_id: str,
    event_store: EventStore | None = None,
    event_store_path: Path | None = None,
    peer_id: str | None = None,
    limit: int = 100,
    max_cases: int = 50,
) -> dict[str, Any]:
    """Build a Host Model review request from corrected live decisions.

    Live decisions are audit feedstock only. This helper intentionally does not
    emit route/action training events or learner-dataset rows. A case is
    reviewable only when the live audit carries an explicit user correction
    with the original input text; uncorrected live decisions are skipped.
    """

    effective_run_id = _required_text(run_id, "run_id")
    effective_app_id = _required_text(app_id, "app_id")
    generated_at = int(time.time() * 1000)
    owns_store = event_store is None
    store = event_store or EventStore(event_store_path)
    try:
        events = store.query(
            event_type=ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
            app_id=effective_app_id,
            source_peer_id=_text(peer_id) or None,
            limit=max(1, int(limit)),
        )
    finally:
        if owns_store:
            store.close()

    cases: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    seen_correction_fingerprints: set[str] = set()
    for event in reversed(events):
        try:
            audit = validate_route_matrix_live_decision_audit(
                json.loads(event.payload.decode("utf-8"))
            )
        except Exception:
            _bump(skipped, "invalid_live_decision_audit")
            continue
        correction = audit.get("user_correction")
        if not isinstance(correction, dict) or not correction:
            _bump(skipped, "missing_user_correction")
            continue
        input_text = _correction_source_input_text(correction)
        if not input_text:
            _bump(skipped, "missing_correction_input_text")
            continue
        correction_fingerprint = route_matrix_live_feedback_correction_fingerprint(
            audit=audit,
            correction=correction,
            input_text=input_text,
        )
        if correction_fingerprint in seen_correction_fingerprints:
            _bump(skipped, "duplicate_user_correction")
            continue
        seen_correction_fingerprints.add(correction_fingerprint)
        cases.append(
            _live_feedback_review_case(
                event=event,
                audit=audit,
                correction=correction,
                input_text=input_text,
            )
        )
        if len(cases) >= max(1, int(max_cases)):
            break

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_REVIEW_REQUEST_SCHEMA_VERSION,
        "result": {
            "run_id": effective_run_id,
            "app_id": effective_app_id,
            "summary": {
                "event_count": len(events),
                "review_case_count": len(cases),
                "skipped_counts": dict(sorted(skipped.items())),
                "ready_for_host_model_review": bool(cases),
                "ready_for_learner_dataset": False,
                "ready_for_learner_dataset_reason": (
                    "live_feedback_requires_host_model_review_and_existing_route_action_gates"
                ),
                "ready_for_live_routing": False,
                "ready_for_live_routing_reason": (
                    "live_feedback_review_request_is_training_side_only"
                ),
            },
            "review_cases": cases,
            "host_model_instructions": {
                "review_user_correction_against_live_decision": True,
                "do_not_treat_live_decision_as_ground_truth": True,
                "do_not_replay_source_prompt_as_training_variant": True,
                "tool_call_plan_args_must_match_declared_schema": True,
                "tool_call_plan_args_must_be_entity_free": True,
                "output_must_pass_route_action_seed_candidate_gates": True,
            },
        },
        "error": None,
        "audit": {
            "method": "build_route_matrix_live_feedback_review_request_from_event_store",
            "generated_at_ms": generated_at,
            "training_side_only": True,
            "writes_events": False,
            "writes_runtime_artifacts": False,
            "source_event_type": ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
        },
    }


def _live_feedback_review_case(
    *,
    event: DataEvent,
    audit: dict[str, Any],
    correction: dict[str, Any],
    input_text: str,
) -> dict[str, Any]:
    return {
        "source_event_id": event.id,
        "source_timestamp_ms": event.timestamp_ms,
        "case_id": audit["case_id"],
        "input_text": input_text,
        "correction_text": _correction_text(correction),
        "matrix_prediction": dict(audit["matrix_prediction"]),
        "matrix_calibrated_confidence": audit["matrix_calibrated_confidence"],
        "evidence_available": audit["evidence_available"],
        "evidence_route": audit.get("evidence_route"),
        "final_decision_source": audit["final_decision_source"],
        "fallback_reason": audit.get("fallback_reason"),
        "user_correction": dict(correction),
        "review_goal": "produce_reviewed_route_action_seed_candidate_or_reject",
    }


def _correction_source_input_text(correction: dict[str, Any]) -> str:
    return _text(
        correction.get("source_input_text")
        or correction.get("input_text")
        or correction.get("prompt")
        or correction.get("user_input")
    )


def _correction_text(correction: dict[str, Any]) -> str | None:
    text = _text(
        correction.get("correction_text")
        or correction.get("natural_language_correction")
        or correction.get("corrected_input_text")
    )
    return text or None


def route_matrix_live_feedback_correction_fingerprint(
    *,
    audit: dict[str, Any],
    correction: dict[str, Any],
    input_text: str,
) -> str:
    """Return the stable dedupe key for one explicit live user correction."""

    material = {
        "case_id": audit.get("case_id"),
        "input_text": input_text,
        "correction_text": _correction_text(correction),
        "correction_source": _text(correction.get("correction_source")),
        "is_fixture": correction.get("is_fixture") is True,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _event_id(
    *,
    app_id: str,
    peer_id: str,
    timestamp_ms: int,
    payload_bytes: bytes,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(app_id or "").strip().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(peer_id or "").strip().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(int(timestamp_ms)).encode("ascii"))
    digest.update(b"\x00")
    digest.update(payload_bytes)
    return f"route-matrix-live:{digest.hexdigest()[:32]}"


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
