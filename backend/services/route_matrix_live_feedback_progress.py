# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Progress reporting for route-matrix live-feedback correction accumulation."""

from __future__ import annotations

import json
import math
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from backend.services.route_matrix_live_audit import (
    ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
    route_matrix_live_feedback_correction_fingerprint,
    validate_route_matrix_live_decision_audit,
)
from backend.stores.event_store import EventStore


ROUTE_MATRIX_LIVE_FEEDBACK_PROGRESS_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_progress.v0"
)


def build_route_matrix_live_feedback_progress_report(
    *,
    run_id: str,
    app_id: str,
    event_store: EventStore | None = None,
    event_store_path: Path | None = None,
    peer_id: str | None = None,
    min_real_corrections: int = 30,
    min_distinct_matrix_intents: int = 2,
    max_intent_fraction: float = 0.85,
    max_case_repeat_fraction: float = 0.35,
    limit: int = 10000,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Return a host-side progress report for real user correction accumulation.

    The report is advisory. It never triggers retraining, writes events, writes
    learner feedstock, or writes runtime artifacts. N>=threshold produces a
    human-ack status, not an automatic retrain.
    """

    generated_at_ms = int(time.time() * 1000)
    effective_run_id = _required_text(run_id, "run_id")
    effective_app_id = _required_text(app_id, "app_id")
    effective_peer_id = _text(peer_id) or None
    minimum = max(1, int(min_real_corrections))
    max_examples = max(0, int(max_examples))
    owns_store = event_store is None
    store = event_store or EventStore(event_store_path)
    try:
        events = store.query(
            event_type=ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
            app_id=effective_app_id,
            source_peer_id=effective_peer_id,
            limit=max(1, int(limit)),
        )
    finally:
        if owns_store:
            store.close()

    rows, skipped_counts = _rows_from_events(events)
    unique_real_rows = _unique_real_rows(rows)
    distribution = _distribution(
        unique_real_rows,
        min_real_corrections=minimum,
        min_distinct_matrix_intents=max(1, int(min_distinct_matrix_intents)),
        max_intent_fraction=_fraction(max_intent_fraction, "max_intent_fraction"),
        max_case_repeat_fraction=_fraction(
            max_case_repeat_fraction,
            "max_case_repeat_fraction",
        ),
    )
    threshold_reached = len(unique_real_rows) >= minimum
    distribution_ok = distribution["status"] == "distribution_review_ready"
    status = _status(
        event_count=len(events),
        unique_real_correction_count=len(unique_real_rows),
        threshold_reached=threshold_reached,
        distribution_ok=distribution_ok,
    )
    summary = {
        "event_count": len(events),
        "limit": max(1, int(limit)),
        "limit_reached": len(events) >= max(1, int(limit)),
        "corrected_event_count": sum(1 for row in rows if row["has_correction"]),
        "real_correction_count_raw": sum(1 for row in rows if row["is_real_user"]),
        "real_correction_count_unique": len(unique_real_rows),
        "duplicate_real_correction_count": (
            sum(1 for row in rows if row["is_real_user"]) - len(unique_real_rows)
        ),
        "fixture_correction_count": sum(1 for row in rows if row["is_fixture"]),
        "non_user_correction_count": sum(1 for row in rows if row["is_non_user"]),
        "invalid_event_count": skipped_counts.get("invalid_live_decision_audit", 0),
        "min_real_corrections": minimum,
        "remaining_real_corrections": max(0, minimum - len(unique_real_rows)),
        "threshold_reached": threshold_reached,
        "retrain_requires_explicit_ack": True,
        "auto_retrain_enabled": False,
        "ready_for_paired_eval_cli": threshold_reached and distribution_ok,
        "ready_for_auto_retrain": False,
    }
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_PROGRESS_SCHEMA_VERSION,
        "status": status,
        "result": {
            "run_id": effective_run_id,
            "app_id": effective_app_id,
            "peer_id": effective_peer_id,
            "summary": summary,
            "distribution": distribution,
            "skipped_counts": dict(sorted(skipped_counts.items())),
            "examples": _examples(unique_real_rows, max_examples=max_examples),
        },
        "error": None,
        "audit": {
            "method": "build_route_matrix_live_feedback_progress_report",
            "generated_at_ms": generated_at_ms,
            "training_side_only": True,
            "writes_events": False,
            "writes_training_sample_tags": False,
            "writes_learner_dataset": False,
            "writes_runtime_artifacts": False,
            "retrain_requires_explicit_ack": True,
        },
    }


def _rows_from_events(events: list[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for event in events:
        try:
            audit = validate_route_matrix_live_decision_audit(
                json.loads(event.payload.decode("utf-8"))
            )
        except Exception:
            _bump(skipped, "invalid_live_decision_audit")
            continue
        correction = audit.get("user_correction")
        has_correction = isinstance(correction, dict) and bool(correction)
        correction = correction if isinstance(correction, dict) else {}
        input_text = _correction_input_text(correction)
        source = _correction_source(correction)
        is_fixture = correction.get("is_fixture") is True or source in {
            "fixture",
            "synthetic",
            "test",
        }
        is_real_user = has_correction and not is_fixture and source in {
            "user",
            "human",
            "real_user",
        }
        is_non_user = has_correction and not is_fixture and not is_real_user
        fingerprint = (
            route_matrix_live_feedback_correction_fingerprint(
                audit=audit,
                correction=correction,
                input_text=input_text,
            )
            if has_correction and input_text
            else ""
        )
        rows.append({
            "event_id": event.id,
            "timestamp_ms": event.timestamp_ms,
            "case_id": audit["case_id"],
            "matrix_intent": audit["matrix_prediction"]["intent"],
            "final_decision_source": audit["final_decision_source"],
            "has_correction": has_correction,
            "input_text": input_text,
            "correction_text": _correction_text(correction),
            "correction_source": source,
            "is_fixture": is_fixture,
            "is_real_user": is_real_user,
            "is_non_user": is_non_user,
            "fingerprint": fingerprint,
        })
    return rows, skipped


def _unique_real_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        if row["is_real_user"] is not True:
            continue
        fingerprint = _text(row.get("fingerprint"))
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(row)
    return out


def _distribution(
    rows: list[dict[str, Any]],
    *,
    min_real_corrections: int,
    min_distinct_matrix_intents: int,
    max_intent_fraction: float,
    max_case_repeat_fraction: float,
) -> dict[str, Any]:
    count = len(rows)
    matrix_intents = Counter(row["matrix_intent"] for row in rows)
    cases = Counter(row["case_id"] for row in rows)
    sources = Counter(row["correction_source"] for row in rows)
    final_sources = Counter(row["final_decision_source"] for row in rows)
    input_texts = {_norm(row["input_text"]) for row in rows if row["input_text"]}
    correction_texts = {
        _norm(row["correction_text"])
        for row in rows
        if row["correction_text"]
    }
    max_intent_count = max(matrix_intents.values(), default=0)
    max_case_count = max(cases.values(), default=0)
    max_intent_share = _share(max_intent_count, count)
    max_case_share = _share(max_case_count, count)
    reasons: list[str] = []
    warnings: list[str] = []
    known_matrix_intents = {
        intent
        for intent in matrix_intents
        if intent not in {"", "none", "unknown", "unavailable"}
    }
    if count < min_real_corrections:
        reasons.append("below_min_real_corrections")
    if count >= min_real_corrections and not known_matrix_intents:
        warnings.append("matrix_intent_unavailable_for_distribution")
    if (
        count >= min_real_corrections
        and known_matrix_intents
        and len(known_matrix_intents) < min_distinct_matrix_intents
    ):
        reasons.append("low_matrix_intent_diversity")
    if (
        count >= min_real_corrections
        and known_matrix_intents
        and max_intent_share > max_intent_fraction
    ):
        reasons.append("matrix_intent_concentration_high")
    if count >= min_real_corrections and max_case_share > max_case_repeat_fraction:
        reasons.append("case_repeat_concentration_high")
    if count < min_real_corrections:
        status = "collecting_real_user_corrections"
    elif reasons:
        status = "correction_distribution_insufficient"
    else:
        status = "distribution_review_ready"
    return {
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "matrix_intent_counts": dict(sorted(matrix_intents.items())),
        "final_decision_source_counts": dict(sorted(final_sources.items())),
        "correction_source_counts": dict(sorted(sources.items())),
        "case_counts_top": dict(cases.most_common(10)),
        "distinct_matrix_intent_count": len(matrix_intents),
        "unique_input_text_count": len(input_texts),
        "unique_correction_text_count": len(correction_texts),
        "max_intent_fraction": max_intent_share,
        "max_case_repeat_fraction": max_case_share,
        "thresholds": {
            "min_real_corrections": min_real_corrections,
            "min_distinct_matrix_intents": min_distinct_matrix_intents,
            "max_intent_fraction": max_intent_fraction,
            "max_case_repeat_fraction": max_case_repeat_fraction,
        },
    }


def _status(
    *,
    event_count: int,
    unique_real_correction_count: int,
    threshold_reached: bool,
    distribution_ok: bool,
) -> str:
    if event_count <= 0:
        return "no_live_decision_events"
    if unique_real_correction_count <= 0:
        return "pending_real_user_corrections"
    if not threshold_reached:
        return "collecting_real_user_corrections"
    if not distribution_ok:
        return "correction_distribution_insufficient"
    return "ready_for_alex_retrain_ack"


def _examples(rows: list[dict[str, Any]], *, max_examples: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["timestamp_ms"], reverse=True):
        if len(examples) >= max_examples:
            break
        examples.append({
            "event_id": row["event_id"],
            "timestamp_ms": row["timestamp_ms"],
            "case_id": row["case_id"],
            "matrix_intent": row["matrix_intent"],
            "final_decision_source": row["final_decision_source"],
            "correction_source": row["correction_source"],
            "input_text": row["input_text"],
            "correction_text": row["correction_text"],
        })
    return examples


def _correction_input_text(correction: dict[str, Any]) -> str:
    return _text(
        correction.get("source_input_text")
        or correction.get("input_text")
        or correction.get("prompt")
        or correction.get("user_input")
    )


def _correction_text(correction: dict[str, Any]) -> str:
    return _text(
        correction.get("correction_text")
        or correction.get("natural_language_correction")
        or correction.get("corrected_input_text")
    )


def _correction_source(correction: dict[str, Any]) -> str:
    return _text(correction.get("correction_source") or "none").casefold() or "none"


def _fraction(value: float, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0 or parsed > 1:
        raise ValueError(f"{field} must be in (0, 1]")
    return parsed


def _share(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    return " ".join(text.split())


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
