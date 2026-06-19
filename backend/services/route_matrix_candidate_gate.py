# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Gate route-matrix shadow candidates before any live routing decision.

This module is review-side only. It merges matrix shadow evidence with
plan-prototype coverage and optional runtime validation receipts. It never
enables routing by itself.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


ROUTE_MATRIX_CANDIDATE_GATE_SCHEMA_VERSION = "edgestudio.route_matrix_candidate_gate.v0"


def build_route_matrix_candidate_gate(request: dict[str, Any]) -> dict[str, Any]:
    generated_at = _utc_now()
    if not isinstance(request, dict):
        return _error(
            code="invalid_input",
            message="request must be an object",
            details={"received_type": type(request).__name__},
            generated_at=generated_at,
        )
    try:
        run_id = _required_text(request.get("run_id"), "run_id")
        shadow_review = _validated_review(request.get("shadow_review"), "shadow_review")
        plan_review = _validated_review(request.get("plan_prototype_review"), "plan_prototype_review")
        runtime_validations = _runtime_validations(request.get("runtime_validations"))
        live_routing_enabled = request.get("live_routing_enabled") is True
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix candidate gate request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    shadow_rows = _shadow_rows_by_case(shadow_review)
    plan_rows = _plan_rows_by_case(plan_review)
    candidate_case_ids = [
        case_id
        for case_id, row in shadow_rows.items()
        if _comparison(row).get("verdict") == "routing_candidate"
    ]
    rows = [
        _gate_candidate(
            shadow_row=shadow_rows[case_id],
            plan_row=plan_rows.get(case_id),
            runtime_validation=runtime_validations.get(case_id),
            live_routing_enabled=live_routing_enabled,
        )
        for case_id in candidate_case_ids
    ]
    summary = _summary(rows, live_routing_enabled=live_routing_enabled)
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_CANDIDATE_GATE_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "summary": summary,
            "candidates": rows,
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_candidate_gate_audit.v0",
            "method": "build_route_matrix_candidate_gate",
            "generated_at": generated_at,
            "input_summary": {
                "shadow_candidate_count": len(candidate_case_ids),
                "plan_candidate_count": len(plan_rows),
                "runtime_validation_count": len(runtime_validations),
                "live_routing_enabled": live_routing_enabled,
            },
        },
    }


def build_route_matrix_candidate_gate_from_files(
    *,
    run_id: str,
    shadow_review_path: Path,
    plan_prototype_review_path: Path,
    runtime_validations_path: Path | None = None,
    live_routing_enabled: bool = False,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "run_id": run_id,
        "shadow_review": _read_json(shadow_review_path),
        "plan_prototype_review": _read_json(plan_prototype_review_path),
        "live_routing_enabled": live_routing_enabled,
    }
    if runtime_validations_path is not None:
        request["runtime_validations"] = _read_json(runtime_validations_path)
    return build_route_matrix_candidate_gate(request)


def _gate_candidate(
    *,
    shadow_row: dict[str, Any],
    plan_row: dict[str, Any] | None,
    runtime_validation: dict[str, Any] | None,
    live_routing_enabled: bool,
) -> dict[str, Any]:
    comparison = _comparison(shadow_row)
    evidence_route = (
        shadow_row.get("evidence_route")
        if isinstance(shadow_row.get("evidence_route"), dict)
        else {}
    )
    base_checks = _base_checks(comparison=comparison, evidence_route=evidence_route)
    status = _status(
        base_checks=base_checks,
        plan_row=plan_row,
        runtime_validation=runtime_validation,
        live_routing_enabled=live_routing_enabled,
    )
    return {
        "case_id": shadow_row.get("case_id"),
        "prompt": shadow_row.get("prompt"),
        "expected_intent": shadow_row.get("expected_intent"),
        "matrix_device": shadow_row.get("matrix_device"),
        "evidence_route": evidence_route,
        "plan_prototype_status": (
            plan_row.get("plan_prototype_status")
            if isinstance(plan_row, dict)
            else "missing_plan_review"
        ),
        "runtime_validation": _public_runtime_validation(runtime_validation),
        "checks": base_checks,
        "gate_status": status,
        "live_routing_candidate": status == "eligible_for_live_routing",
        "shadow_only_candidate": status in {
            "no_tool_shadow_candidate",
            "validated_candidate_shadow_only",
        },
    }


def _status(
    *,
    base_checks: dict[str, bool],
    plan_row: dict[str, Any] | None,
    runtime_validation: dict[str, Any] | None,
    live_routing_enabled: bool,
) -> str:
    if not all(base_checks.values()):
        failed = [key for key, ok in base_checks.items() if not ok]
        if any(key in {"device_completed", "device_no_freeze", "device_no_oom"} for key in failed):
            return "blocked_device_runtime_health"
        return "blocked_shadow_evidence"

    if not isinstance(plan_row, dict):
        return "blocked_missing_plan_review"
    plan_status = _text(plan_row.get("plan_prototype_status"))
    if plan_status == "no_tool_required":
        if not _no_tool_intent_agrees(plan_row):
            return "blocked_intent_disagreement"
        return "no_tool_shadow_candidate"
    if plan_row.get("runtime_executable") is not True:
        if plan_row.get("training_side_only") is True:
            return "blocked_training_side_only_plan"
        return "blocked_missing_runtime_plan"

    if not isinstance(runtime_validation, dict):
        return "blocked_missing_runtime_validation"
    if runtime_validation.get("tool_registry_ok") is not True:
        return "blocked_tool_registry_validation"
    if runtime_validation.get("tool_call_plan_ok") is not True:
        return "blocked_tool_call_plan_validation"
    if runtime_validation.get("schema_validation_ok") is not True:
        return "blocked_schema_validation"
    if live_routing_enabled is not True:
        return "validated_candidate_shadow_only"
    return "eligible_for_live_routing"


def _base_checks(
    *,
    comparison: dict[str, Any],
    evidence_route: dict[str, Any],
) -> dict[str, bool]:
    return {
        "shadow_verdict_candidate": comparison.get("verdict") == "routing_candidate",
        "cli_device_agree": comparison.get("cli_device_agree") is True,
        "expected_match_cli": comparison.get("expected_match_cli") is True,
        "expected_match_device": comparison.get("expected_match_device") is True,
        "threshold_passed_cli": comparison.get("threshold_passed_cli") is True,
        "threshold_passed_device": comparison.get("threshold_passed_device") is True,
        "device_completed": evidence_route.get("completed") is True,
        "device_no_freeze": evidence_route.get("freeze_detected") is not True,
        "device_no_oom": evidence_route.get("oom") is not True,
    }


def _no_tool_intent_agrees(plan_row: dict[str, Any]) -> bool:
    evidence_route = (
        plan_row.get("evidence_route")
        if isinstance(plan_row.get("evidence_route"), dict)
        else {}
    )
    return (
        _text(plan_row.get("expected_intent")) == "base_chat"
        and _text(evidence_route.get("route_intent")) == "base_chat"
    )


def _summary(rows: list[dict[str, Any]], *, live_routing_enabled: bool) -> dict[str, Any]:
    status_counts = Counter(row["gate_status"] for row in rows)
    live_candidates = status_counts.get("eligible_for_live_routing", 0)
    blocked_count = sum(
        count for status, count in status_counts.items() if status.startswith("blocked_")
    )
    no_tool_count = status_counts.get("no_tool_shadow_candidate", 0)
    validated_shadow_only_count = status_counts.get("validated_candidate_shadow_only", 0)
    ready = live_routing_enabled and live_candidates > 0 and blocked_count == 0
    return {
        "candidate_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "blocked_count": blocked_count,
        "no_tool_shadow_candidate_count": no_tool_count,
        "validated_shadow_only_count": validated_shadow_only_count,
        "live_routing_candidate_count": live_candidates,
        "live_routing_enabled": live_routing_enabled,
        "ready_for_live_routing": ready,
        "ready_for_live_routing_reason": _ready_reason(
            ready=ready,
            live_routing_enabled=live_routing_enabled,
            live_candidates=live_candidates,
            blocked_count=blocked_count,
        ),
    }


def _ready_reason(
    *,
    ready: bool,
    live_routing_enabled: bool,
    live_candidates: int,
    blocked_count: int,
) -> str:
    if ready:
        return "all_review_candidates_passed_shadow_plan_and_runtime_validation"
    if blocked_count > 0:
        return "one_or_more_candidates_blocked_by_gate"
    if live_routing_enabled is not True:
        return "live_routing_disabled_shadow_review_only"
    if live_candidates <= 0:
        return "no_validated_live_routing_candidates"
    return "not_ready"


def _shadow_rows_by_case(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    rows = result.get("cases")
    if not isinstance(rows, list):
        raise ValueError("shadow_review.result.cases must be a list")
    return _rows_by_case(rows, "shadow_review.result.cases")


def _plan_rows_by_case(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    rows = result.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("plan_prototype_review.result.candidates must be a list")
    return _rows_by_case(rows, "plan_prototype_review.result.candidates")


def _rows_by_case(rows: list[Any], field: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"{field}[{index}] must be an object")
        case_id = _required_text(row.get("case_id"), f"{field}[{index}].case_id")
        if case_id in out:
            raise ValueError(f"duplicate case_id in {field}: {case_id}")
        out[case_id] = row
    return out


def _runtime_validations(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    rows: list[Any]
    if isinstance(value, dict):
        result = value.get("result") if isinstance(value.get("result"), dict) else {}
        raw_rows = value.get("validations") or result.get("validations")
        if isinstance(raw_rows, list):
            rows = raw_rows
        else:
            rows = []
            for case_id, row in value.items():
                if isinstance(row, dict):
                    copied = dict(row)
                    copied.setdefault("case_id", case_id)
                    rows.append(copied)
    elif isinstance(value, list):
        rows = value
    else:
        raise TypeError("runtime_validations must be a list or object")
    out: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"runtime_validations[{index}] must be an object")
        case_id = _required_text(row.get("case_id"), f"runtime_validations[{index}].case_id")
        if case_id in out:
            raise ValueError(f"duplicate runtime validation case_id: {case_id}")
        out[case_id] = row
    return out


def _public_runtime_validation(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "tool_registry_ok": value.get("tool_registry_ok") is True,
        "tool_call_plan_ok": value.get("tool_call_plan_ok") is True,
        "schema_validation_ok": value.get("schema_validation_ok") is True,
        "selected_tools": _string_list(value.get("selected_tools")),
        "tool_call_plan_count": len(_list_of_dicts(value.get("tool_call_plan"))),
    }


def _comparison(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("comparison")
    return value if isinstance(value, dict) else {}


def _validated_review(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    if value.get("ok") is not True:
        raise ValueError(f"{field}.ok must be true")
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_CANDIDATE_GATE_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_candidate_gate_audit.v0",
            "method": "build_route_matrix_candidate_gate",
            "generated_at": generated_at,
            "input_summary": {},
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


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
