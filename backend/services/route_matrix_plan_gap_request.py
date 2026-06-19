# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build Host Model requests for route-matrix plan gaps.

The matrix router can identify an intent, but tool-requiring intents still need
app-owned tool selection, toolCallPlan, and schema validation. This module turns
candidate-gate blockers into a dry-run Host Model request. It does not call the
Host Model and does not emit runtime artifacts.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

ROUTE_MATRIX_PLAN_GAP_REQUEST_SCHEMA_VERSION = "edgestudio.route_matrix_plan_gap_request.v0"
ROUTE_MATRIX_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.route_matrix_runtime_validation_receipt.v0"
)

_PLAN_SEED_STATUSES = {
    "blocked_missing_runtime_plan",
    "blocked_training_side_only_plan",
}
_RUNTIME_VALIDATION_STATUSES = {
    "blocked_missing_runtime_validation",
    "blocked_tool_registry_validation",
    "blocked_tool_call_plan_validation",
    "blocked_schema_validation",
}


def build_route_matrix_plan_gap_request(request: dict[str, Any]) -> dict[str, Any]:
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
        app_id = _required_text(request.get("app_id"), "app_id")
        candidate_gate = _candidate_gate(request.get("candidate_gate"))
        tool_registry = _tool_registry(request.get("tool_registry"))
        target_seed_count = max(1, int(request.get("target_seed_count") or 24))
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix plan gap request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    candidates = _candidate_rows(candidate_gate)
    plan_seed_cases = [
        _golden_case(row)
        for row in candidates
        if _text(row.get("gate_status")) in _PLAN_SEED_STATUSES
    ]
    runtime_validation_cases = [
        _runtime_validation_case(row)
        for row in candidates
        if _text(row.get("gate_status")) in _RUNTIME_VALIDATION_STATUSES
    ]
    ignored_status_counts = Counter(
        _text(row.get("gate_status"))
        for row in candidates
        if _text(row.get("gate_status")) not in _PLAN_SEED_STATUSES
        and _text(row.get("gate_status")) not in _RUNTIME_VALIDATION_STATUSES
    )
    route_action_request = {
        "schema_version": "edgestudio.route_action_seed_candidates_request.v0",
        "app_id": app_id,
        "tool_registry": tool_registry,
        "golden_cases": plan_seed_cases,
        "target_seed_count": max(target_seed_count, len(plan_seed_cases)),
        "seed_run_id": f"{run_id}:plan-gap",
    }
    runtime_receipt_schema = _runtime_validation_receipt_schema()

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_PLAN_GAP_REQUEST_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "app_id": app_id,
            "summary": {
                "candidate_count": len(candidates),
                "plan_seed_case_count": len(plan_seed_cases),
                "runtime_validation_case_count": len(runtime_validation_cases),
                "ignored_status_counts": dict(sorted(ignored_status_counts.items())),
                "ready_for_host_model_seed_generation": bool(plan_seed_cases),
                "ready_for_live_routing": False,
                "ready_for_live_routing_reason": (
                    "plan_gap_request_is_training_side_only_and_requires_host_model_output_gates"
                ),
            },
            "route_action_seed_candidates_request": route_action_request,
            "runtime_validation_receipt_schema": runtime_receipt_schema,
            "runtime_validation_required_cases": runtime_validation_cases,
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_plan_gap_request_audit.v0",
            "method": "build_route_matrix_plan_gap_request",
            "generated_at": generated_at,
            "input_summary": {
                "candidate_count": len(candidates),
                "tool_count": len(tool_registry),
                "target_seed_count": target_seed_count,
            },
        },
    }


def build_route_matrix_plan_gap_request_from_files(
    *,
    run_id: str,
    app_id: str,
    candidate_gate_path: Path,
    tool_registry_path: Path,
    target_seed_count: int = 24,
) -> dict[str, Any]:
    return build_route_matrix_plan_gap_request({
        "run_id": run_id,
        "app_id": app_id,
        "candidate_gate": _read_json(candidate_gate_path),
        "tool_registry": _read_json(tool_registry_path),
        "target_seed_count": target_seed_count,
    })


def _golden_case(row: dict[str, Any]) -> dict[str, Any]:
    case_id = _required_text(row.get("case_id"), "candidate.case_id")
    prompt = _required_text(row.get("prompt"), f"{case_id}.prompt")
    expected_intent = _required_text(row.get("expected_intent"), f"{case_id}.expected_intent")
    return {
        "case_id": case_id,
        "prompt": prompt,
        "expected_route_intent": expected_intent,
        "blocked_status": _text(row.get("gate_status")),
        "plan_prototype_status": _text(row.get("plan_prototype_status")),
        "matrix_predicted_intent": _matrix_intent(row),
        "evidence_route_intent": _evidence_intent(row),
        "instructions": {
            "source_prompt_is_eval_only": True,
            "generate_prompt_variants": True,
            "do_not_replay_source_prompt": True,
            "selected_tools_must_exist_in_tool_registry": True,
            "tool_call_plan_args_must_match_declared_schema": True,
            "do_not_include_user_fact_values_in_variants": True,
        },
    }


def _runtime_validation_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "prompt": row.get("prompt"),
        "expected_route_intent": row.get("expected_intent"),
        "blocked_status": row.get("gate_status"),
        "plan_prototype_status": row.get("plan_prototype_status"),
        "required_receipt_schema_version": ROUTE_MATRIX_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "required_checks": [
            "tool_registry_ok",
            "tool_call_plan_ok",
            "schema_validation_ok",
        ],
    }


def _runtime_validation_receipt_schema() -> dict[str, Any]:
    return {
        "schema_version": ROUTE_MATRIX_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "validations": [
            {
                "case_id": "must match candidate case_id",
                "selected_tools": ["registered tool names"],
                "tool_call_plan": [
                    {
                        "toolName": "registered tool name",
                        "arguments": "object validated against the app tool schema",
                    }
                ],
                "tool_registry_ok": "boolean",
                "tool_call_plan_ok": "boolean",
                "schema_validation_ok": "boolean",
                "notes": "optional audit string",
            }
        ],
    }


def _candidate_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    rows = result.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("candidate_gate.result.candidates must be a list")
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"candidate_gate.result.candidates[{index}] must be an object")
        out.append(row)
    return out


def _candidate_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("candidate_gate must be an object")
    if value.get("ok") is not True:
        raise ValueError("candidate_gate.ok must be true")
    return value


def _tool_registry(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("tool_registry must be a list")
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise TypeError(f"tool_registry[{index}] must be an object")
        name = _required_text(
            raw.get("name") or raw.get("tool_name") or raw.get("toolName"),
            f"tool_registry[{index}].name",
        )
        if name in seen:
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(name)
        tool = dict(raw)
        tool["name"] = name
        tools.append(tool)
    if not tools:
        raise ValueError("tool_registry must contain at least one tool")
    return tools


def _matrix_intent(row: dict[str, Any]) -> str:
    matrix = row.get("matrix_device") if isinstance(row.get("matrix_device"), dict) else {}
    return _text(matrix.get("predicted_intent"))


def _evidence_intent(row: dict[str, Any]) -> str:
    evidence = row.get("evidence_route") if isinstance(row.get("evidence_route"), dict) else {}
    return _text(evidence.get("route_intent"))


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
        "schema_version": ROUTE_MATRIX_PLAN_GAP_REQUEST_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_plan_gap_request_audit.v0",
            "method": "build_route_matrix_plan_gap_request",
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
