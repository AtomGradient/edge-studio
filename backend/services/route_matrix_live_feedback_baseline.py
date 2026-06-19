# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-side baseline metrics for route-matrix live feedback.

This module is deliberately training-side only. It reads live-decision audit
events and optional review/eval receipts, then produces a receipt that separates
fixture pipeline proof from real user-correction flywheel proof.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from backend.services.route_matrix_live_audit import (
    ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
    validate_route_matrix_live_decision_audit,
)
from backend.stores.event_store import DataEvent, EventStore


ROUTE_MATRIX_LIVE_FEEDBACK_BASELINE_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_baseline.v0"
)


def build_route_matrix_live_feedback_baseline_from_event_store(
    *,
    run_id: str,
    app_id: str,
    event_store: EventStore | None = None,
    event_store_path: Path | None = None,
    peer_id: str | None = None,
    limit: int = 1000,
    min_evaluable_n: int = 100,
    heldout_cutoff_ms: int | None = None,
    baseline_eval: dict[str, Any] | None = None,
    retrained_eval: dict[str, Any] | None = None,
    host_model_review_receipt: dict[str, Any] | None = None,
    evidence_scope: str = "unspecified",
) -> dict[str, Any]:
    """Build a host-side baseline receipt from route-matrix live audit events."""

    effective_run_id = _required_text(run_id, "run_id")
    effective_app_id = _required_text(app_id, "app_id")
    normalized_evidence_scope = _text(evidence_scope) or "unspecified"
    minimum_n = max(1, int(min_evaluable_n))
    generated_at_ms = int(time.time() * 1000)
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

    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for event in reversed(events):
        try:
            audit = validate_route_matrix_live_decision_audit(
                json.loads(event.payload.decode("utf-8"))
            )
        except Exception:
            _bump(skipped, "invalid_live_decision_audit")
            continue
        rows.append(_baseline_row(event=event, audit=audit))

    metrics = _metrics(rows)
    correction_source_counts = Counter(row["correction_source"] for row in rows)
    user_correction_count = correction_source_counts.get("user", 0)
    fixture_correction_count = correction_source_counts.get("fixture", 0)
    heldout = _heldout_summary(
        rows,
        heldout_cutoff_ms=heldout_cutoff_ms,
        min_evaluable_n=minimum_n,
    )
    retrain = _retrain_comparison(
        baseline_eval=baseline_eval,
        retrained_eval=retrained_eval,
        min_evaluable_n=minimum_n,
    )
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_BASELINE_SCHEMA_VERSION,
        "status": "built",
        "result": {
            "run_id": effective_run_id,
            "app_id": effective_app_id,
            "summary": {
                "event_count": len(events),
                "valid_event_count": len(rows),
                "skipped_counts": dict(sorted(skipped.items())),
                "correction_source_counts": dict(sorted(correction_source_counts.items())),
                "fixture_pipeline_status": (
                    "pipeline_evaluable"
                    if fixture_correction_count > 0
                    else "missing_fixture_corrections"
                ),
                "flywheel_evidence_status": _flywheel_status(
                    user_correction_count=user_correction_count,
                    min_evaluable_n=minimum_n,
                    heldout_status=heldout["status"],
                    retrain_status=retrain["status"],
                ),
                "minimum_evaluable_n": minimum_n,
                "ready_for_learner_dataset": False,
                "ready_for_learner_dataset_reason": (
                    "baseline_receipt_is_metrics_only_host_model_review_and_gates_required"
                ),
                "ready_for_live_routing": False,
                "ready_for_live_routing_reason": (
                    "baseline_receipt_is_training_side_only"
                ),
                "evidence_scope": normalized_evidence_scope,
                "scope_boundaries": {
                    "single_app_evidence_is_not_cross_app_generalization": (
                        normalized_evidence_scope != "cross_app"
                    ),
                },
            },
            "metrics": metrics,
            "host_model_review": _host_model_review_summary(host_model_review_receipt),
            "heldout": heldout,
            "retrain_comparison": retrain,
            "phase_boundaries": {
                "fixture_correction_phase": "3l-alpha_pipeline_proof_only",
                "user_correction_phase": "3l-beta_real_flywheel_evidence",
                "fixture_correction_is_not_flywheel_evidence": True,
            },
        },
        "error": None,
        "audit": {
            "method": "build_route_matrix_live_feedback_baseline_from_event_store",
            "generated_at_ms": generated_at_ms,
            "metrics_location": "edgestudio_host_side_only",
            "training_side_only": True,
            "writes_events": False,
            "writes_runtime_artifacts": False,
            "writes_training_sample_tags": False,
            "source_event_type": ROUTE_MATRIX_LIVE_DECISION_EVENT_TYPE,
        },
    }


def _baseline_row(*, event: DataEvent, audit: dict[str, Any]) -> dict[str, Any]:
    correction = audit.get("user_correction")
    if not isinstance(correction, dict):
        correction = {}
    corrected_intent = _first_text(
        correction,
        "corrected_route_intent",
        "route_intent",
        "intent",
    )
    corrected_tools = _text_list(
        correction.get("corrected_selected_tools")
        or correction.get("selected_tools")
        or correction.get("tools")
    )
    matrix_intent = _text(audit.get("matrix_prediction", {}).get("intent"))
    route_tools = _audit_selected_tools(audit)
    return {
        "source_event_id": event.id,
        "timestamp_ms": event.timestamp_ms,
        "case_id": audit.get("case_id"),
        "matrix_intent": matrix_intent,
        "matrix_probability": audit.get("matrix_prediction", {}).get("probability"),
        "final_decision_source": audit.get("final_decision_source"),
        "fallback_reason": audit.get("fallback_reason"),
        "route_tools": route_tools,
        "has_user_correction": bool(correction),
        "correction_source": _correction_source(correction),
        "corrected_route_intent": corrected_intent or None,
        "corrected_selected_tools": corrected_tools,
        "intent_evaluable": bool(corrected_intent),
        "intent_agrees_with_correction": (
            matrix_intent == corrected_intent if corrected_intent else None
        ),
        "tool_evaluable": bool(corrected_tools and route_tools),
        "tools_agree_with_correction": (
            sorted(corrected_tools) == sorted(route_tools)
            if corrected_tools and route_tools
            else None
        ),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_count = len(rows)
    corrected = [row for row in rows if row["has_user_correction"]]
    user_corrected = [row for row in rows if row["correction_source"] == "user"]
    intent_eval = [row for row in corrected if row["intent_evaluable"]]
    intent_agree = [row for row in intent_eval if row["intent_agrees_with_correction"] is True]
    tool_eval = [row for row in corrected if row["tool_evaluable"]]
    tool_agree = [row for row in tool_eval if row["tools_agree_with_correction"] is True]
    final_source_counts = Counter(_text(row["final_decision_source"]) for row in rows)
    fallback_rows = [row for row in rows if row["final_decision_source"] != "matrix"]
    by_intent: dict[str, dict[str, Any]] = {}
    grouped_by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_by_intent[_text(row["matrix_intent"]) or "<missing>"].append(row)
    for intent, intent_rows in sorted(grouped_by_intent.items()):
        intent_corrected = [row for row in intent_rows if row["intent_evaluable"]]
        intent_agreements = [
            row for row in intent_corrected if row["intent_agrees_with_correction"] is True
        ]
        by_intent[intent] = {
            "event_count": len(intent_rows),
            "fallback_rate": _rate(
                sum(1 for row in intent_rows if row["final_decision_source"] != "matrix"),
                len(intent_rows),
            ),
            "intent_agreement_rate": _rate(len(intent_agreements), len(intent_corrected)),
        }
    by_tool: dict[str, dict[str, Any]] = {}
    grouped_by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for tool in row["route_tools"]:
            grouped_by_tool[tool].append(row)
    for tool, tool_rows in sorted(grouped_by_tool.items()):
        by_tool[tool] = {
            "event_count": len(tool_rows),
            "fallback_rate": _rate(
                sum(1 for row in tool_rows if row["final_decision_source"] != "matrix"),
                len(tool_rows),
            ),
        }
    return {
        "correction_rate": _rate(len(corrected), event_count),
        "real_user_correction_rate": _rate(len(user_corrected), event_count),
        "matrix_intent_agreement_rate": _rate(len(intent_agree), len(intent_eval)),
        "matrix_intent_agreement_evaluable_count": len(intent_eval),
        "matrix_tool_agreement_rate": _rate(len(tool_agree), len(tool_eval)),
        "matrix_tool_agreement_evaluable_count": len(tool_eval),
        "fallback_rate": _rate(len(fallback_rows), event_count),
        "final_decision_source_counts": dict(sorted(final_source_counts.items())),
        "fallback_reason_counts": dict(sorted(Counter(
            _text(row["fallback_reason"]) or "<none>"
            for row in fallback_rows
        ).items())),
        "by_intent": by_intent,
        "by_tool": by_tool,
    }


def _host_model_review_summary(receipt: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        return {"status": "missing", "reviewed_count": 0, "approved_count": 0}
    cases = _review_items(receipt)
    if not cases:
        return {"status": "missing_review_cases", "reviewed_count": 0, "approved_count": 0}
    reviewed = 0
    approved = 0
    for item in cases:
        decision = _review_decision(item)
        if not decision:
            continue
        reviewed += 1
        if decision in {"approve", "approved", "pass", "accepted", "ready"}:
            approved += 1
    return {
        "status": "computed" if reviewed else "missing_review_decisions",
        "reviewed_count": reviewed,
        "approved_count": approved,
        "approval_rate": _rate(approved, reviewed),
    }


def _review_decision(item: dict[str, Any]) -> str:
    for key in (
        "decision",
        "review_decision",
        "status",
        "verdict",
        "label",
        "decision_label",
    ):
        text = _text(item.get(key)).casefold()
        if text:
            return text
    return ""


def _heldout_summary(
    rows: list[dict[str, Any]],
    *,
    heldout_cutoff_ms: int | None,
    min_evaluable_n: int,
) -> dict[str, Any]:
    if heldout_cutoff_ms is None:
        return {
            "status": "missing_heldout_cutoff",
            "policy": "time_cut_required",
            "cutoff_ms": None,
        }
    train_rows = [row for row in rows if int(row["timestamp_ms"]) < int(heldout_cutoff_ms)]
    heldout_rows = [row for row in rows if int(row["timestamp_ms"]) >= int(heldout_cutoff_ms)]
    heldout_corrected = [row for row in heldout_rows if row["has_user_correction"]]
    status = (
        "ready"
        if len(heldout_corrected) >= min_evaluable_n
        else "low_sample_volatile"
    )
    return {
        "status": status,
        "policy": "time_cut",
        "cutoff_ms": int(heldout_cutoff_ms),
        "train_event_count": len(train_rows),
        "heldout_event_count": len(heldout_rows),
        "heldout_corrected_count": len(heldout_corrected),
    }


def _retrain_comparison(
    *,
    baseline_eval: dict[str, Any] | None,
    retrained_eval: dict[str, Any] | None,
    min_evaluable_n: int,
) -> dict[str, Any]:
    baseline = _eval_anchor(baseline_eval)
    retrained = _eval_anchor(retrained_eval)
    if baseline["status"] != "found" or retrained["status"] != "found":
        return {
            "status": "missing_eval_inputs",
            "baseline": baseline,
            "retrained": retrained,
        }
    if baseline["heldout_id"] != retrained["heldout_id"]:
        return {
            "status": "heldout_mismatch",
            "baseline": baseline,
            "retrained": retrained,
        }
    sample_count = min(baseline["sample_count"] or 0, retrained["sample_count"] or 0)
    if sample_count < min_evaluable_n:
        status = "low_sample_volatile"
    else:
        status = "ready"
    delta = (
        retrained["accuracy"] - baseline["accuracy"]
        if baseline["accuracy"] is not None and retrained["accuracy"] is not None
        else None
    )
    return {
        "status": status,
        "baseline": baseline,
        "retrained": retrained,
        "paired_heldout_id": baseline["heldout_id"],
        "accuracy_delta": delta,
        "improved": (delta > 0) if status == "ready" and delta is not None else None,
    }


def _flywheel_status(
    *,
    user_correction_count: int,
    min_evaluable_n: int,
    heldout_status: str,
    retrain_status: str,
) -> str:
    if user_correction_count <= 0:
        return "not_started_no_real_user_corrections"
    if user_correction_count < min_evaluable_n:
        return "low_sample_volatile"
    if heldout_status != "ready":
        return "pending_clean_heldout"
    if retrain_status != "ready":
        return "pending_retrain_comparison"
    return "evaluable"


def _review_items(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
    output_schema = (
        receipt.get("output_schema")
        if isinstance(receipt.get("output_schema"), dict)
        else {}
    )
    output_schema_result = (
        output_schema.get("result")
        if isinstance(output_schema.get("result"), dict)
        else {}
    )
    for container in (result, receipt, output_schema_result, output_schema):
        for key in (
            "review_cases",
            "reviewed_cases",
            "reviews",
            "cases",
            "items",
            "outputs",
            "seed_candidates",
        ):
            value = container.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _eval_anchor(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"status": "missing", "run_id": None, "heldout_id": None}
    summary = value.get("summary") if isinstance(value.get("summary"), dict) else {}
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    result_summary = (
        result.get("summary") if isinstance(result.get("summary"), dict) else {}
    )
    merged = {**summary, **result_summary, **value}
    return {
        "status": "found",
        "run_id": _text(merged.get("run_id") or result.get("run_id")) or None,
        "heldout_id": _text(
            merged.get("heldout_id")
            or merged.get("eval_set_id")
            or merged.get("split_id")
        ) or None,
        "accuracy": _optional_float(
            merged.get("accuracy")
            or merged.get("intent_acc")
            or merged.get("route_accuracy")
        ),
        "sample_count": _optional_int(
            merged.get("sample_count")
            or merged.get("case_count")
            or merged.get("heldout_count")
        ),
    }


def _audit_selected_tools(audit: dict[str, Any]) -> list[str]:
    route = audit.get("evidence_route")
    if not isinstance(route, dict):
        return []
    tools = _text_list(route.get("selected_tools") or route.get("tools"))
    tool_name = _text(route.get("tool_name") or route.get("selected_tool"))
    if tool_name:
        tools.append(tool_name)
    return sorted(set(tools))


def _correction_source(correction: dict[str, Any]) -> str:
    if not correction:
        return "none"
    raw = _text(
        correction.get("correction_source")
        or correction.get("source")
        or correction.get("origin")
    ).casefold()
    if correction.get("is_fixture") is True or raw in {
        "fixture",
        "mock",
        "synthetic",
        "developer_fixture",
    }:
        return "fixture"
    if correction.get("is_fixture") is False or raw in {"user", "human", "real_user"}:
        return "user"
    return "unknown"


def _text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return sorted({
            text
            for item in value
            if (text := _text(item))
        })
    text = _text(value)
    return [text] if text else []


def _first_text(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = _text(value.get(key))
        if text:
            return text
    return ""


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
