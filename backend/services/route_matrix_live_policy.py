# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build a controlled-live policy from route-matrix candidate gate output.

This module is review-side only. It does not enable runtime routing by itself.
It turns validated shadow candidates into an explicit allowlist artifact that a
runtime/live switch can consume later.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json


ROUTE_MATRIX_LIVE_POLICY_SCHEMA_VERSION = "edgestudio.route_matrix_live_policy.v0"

_ACCEPTED_GATE_STATUSES = {
    "validated_candidate_shadow_only",
    "eligible_for_live_routing",
    "no_tool_shadow_candidate",
}

_LIVE_DECISION_AUDIT_FIELDS = (
    "case_id",
    "matrix_prediction",
    "matrix_calibrated_confidence",
    "evidence_available",
    "evidence_route",
    "final_decision_source",
    "fallback_reason",
    "shadow_mode_was",
    "user_correction",
)


def build_route_matrix_live_policy(request: dict[str, Any]) -> dict[str, Any]:
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
        adapter_id = _required_text(request.get("adapter_id"), "adapter_id")
        candidate_gate = _validated_candidate_gate(request.get("candidate_gate"))
        controls = _controls(request.get("controls"))
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix live policy request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    candidates = _candidate_rows(candidate_gate)
    global_status = _global_status(
        app_id=app_id,
        adapter_id=adapter_id,
        controls=controls,
    )
    rows = [
        _policy_row(
            row,
            app_id=app_id,
            adapter_id=adapter_id,
            controls=controls,
            global_status=global_status,
        )
        for row in candidates
    ]
    summary = _summary(rows, controls=controls, global_status=global_status)
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_POLICY_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "app_id": app_id,
            "adapter_id": adapter_id,
            "controls": _public_controls(controls),
            "summary": summary,
            "policy_rows": rows,
            "live_decision_audit_schema": {
                "schema_version": "edgestudio.route_matrix_live_decision_audit.v0",
                "required_fields": list(_LIVE_DECISION_AUDIT_FIELDS),
            },
            "circuit_breaker": _public_circuit_breaker(controls),
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_live_policy_audit.v0",
            "method": "build_route_matrix_live_policy",
            "generated_at": generated_at,
            "input_summary": {
                "candidate_count": len(candidates),
                "enabled": controls["enabled"],
                "allowed_app_count": len(controls["allowed_app_ids"]),
                "allowed_adapter_count": len(controls["allowed_adapter_ids"]),
                "allowed_input_sha256_count": len(controls["allowed_input_sha256s"]),
                "allowed_intent_count": len(controls["allowed_intents"]),
                "allowed_tool_count": len(controls["allowed_tools"]),
            },
        },
    }


def build_route_matrix_live_policy_from_files(
    *,
    run_id: str,
    app_id: str,
    adapter_id: str,
    candidate_gate_path: Path,
    controls_path: Path,
) -> dict[str, Any]:
    return build_route_matrix_live_policy({
        "run_id": run_id,
        "app_id": app_id,
        "adapter_id": adapter_id,
        "candidate_gate": _read_json(candidate_gate_path),
        "controls": _read_json(controls_path),
    })


def _policy_row(
    row: dict[str, Any],
    *,
    app_id: str,
    adapter_id: str,
    controls: dict[str, Any],
    global_status: str,
) -> dict[str, Any]:
    intent = _candidate_intent(row)
    selected_tools = _selected_tools(row)
    input_sha256 = _matrix_input_sha256(row)
    status = _row_status(
        row,
        intent=intent,
        input_sha256=input_sha256,
        selected_tools=selected_tools,
        app_id=app_id,
        adapter_id=adapter_id,
        controls=controls,
        global_status=global_status,
    )
    return {
        "case_id": row.get("case_id"),
        "prompt": row.get("prompt"),
        "intent": intent,
        "input_sha256": input_sha256,
        "selected_tools": selected_tools,
        "candidate_gate_status": row.get("gate_status"),
        "live_policy_status": status,
        "live_routing_candidate": status == "eligible_for_controlled_live",
        "final_decision_source_if_live": (
            "matrix" if status == "eligible_for_controlled_live" else None
        ),
        "fallback_reason_if_not_live": (
            None if status == "eligible_for_controlled_live" else status
        ),
    }


def _row_status(
    row: dict[str, Any],
    *,
    intent: str,
    input_sha256: str,
    selected_tools: list[str],
    app_id: str,
    adapter_id: str,
    controls: dict[str, Any],
    global_status: str,
) -> str:
    if controls["enabled"] is not True:
        return "excluded_live_disabled"
    if global_status != "ready":
        return global_status
    if app_id not in controls["allowed_app_ids"]:
        return "excluded_app_not_enabled"
    if adapter_id not in controls["allowed_adapter_ids"]:
        return "excluded_adapter_not_enabled"
    allowed_input_sha256s = controls["allowed_input_sha256s"]
    if allowed_input_sha256s and input_sha256 not in allowed_input_sha256s:
        return "excluded_input_not_enabled"
    if intent not in controls["allowed_intents"]:
        return "excluded_intent_not_enabled"
    if selected_tools:
        allowed_tools = controls["allowed_tools"]
        if not allowed_tools:
            return "blocked_missing_tool_allowlist"
        if any(tool not in allowed_tools for tool in selected_tools):
            return "excluded_tool_not_enabled"
    if _matrix_threshold_passed(row) is not True:
        return "blocked_low_confidence_matrix_prediction"
    if _text(row.get("gate_status")) not in _ACCEPTED_GATE_STATUSES:
        return "blocked_candidate_gate"
    return "eligible_for_controlled_live"


def _global_status(
    *,
    app_id: str,
    adapter_id: str,
    controls: dict[str, Any],
) -> str:
    if controls["enabled"] is not True:
        return "excluded_live_disabled"
    if not controls["allowed_app_ids"]:
        return "blocked_missing_app_allowlist"
    if not controls["allowed_adapter_ids"]:
        return "blocked_missing_adapter_allowlist"
    if not controls["allowed_intents"]:
        return "blocked_missing_intent_allowlist"
    if app_id not in controls["allowed_app_ids"]:
        return "excluded_app_not_enabled"
    if adapter_id not in controls["allowed_adapter_ids"]:
        return "excluded_adapter_not_enabled"
    circuit_status = _circuit_breaker_status(controls["circuit_breaker"])
    if circuit_status != "ready":
        return circuit_status
    return "ready"


def _summary(
    rows: list[dict[str, Any]],
    *,
    controls: dict[str, Any],
    global_status: str,
) -> dict[str, Any]:
    status_counts = Counter(row["live_policy_status"] for row in rows)
    eligible_count = status_counts.get("eligible_for_controlled_live", 0)
    hard_blocked_count = sum(
        count for status, count in status_counts.items() if status.startswith("blocked_")
    )
    ready = (
        controls["enabled"] is True
        and global_status == "ready"
        and eligible_count > 0
        and hard_blocked_count == 0
    )
    return {
        "candidate_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "eligible_live_candidate_count": eligible_count,
        "hard_blocked_count": hard_blocked_count,
        "live_routing_enabled": controls["enabled"],
        "ready_for_live_routing": ready,
        "ready_for_live_routing_reason": _ready_reason(
            ready=ready,
            enabled=controls["enabled"],
            global_status=global_status,
            eligible_count=eligible_count,
            hard_blocked_count=hard_blocked_count,
        ),
    }


def _ready_reason(
    *,
    ready: bool,
    enabled: bool,
    global_status: str,
    eligible_count: int,
    hard_blocked_count: int,
) -> str:
    if ready:
        return "controlled_live_policy_has_eligible_candidates"
    if enabled is not True:
        return "live_routing_disabled"
    if global_status != "ready":
        return global_status
    if hard_blocked_count > 0:
        return "one_or_more_enabled_candidates_blocked"
    if eligible_count <= 0:
        return "no_candidates_enabled_by_policy"
    return "not_ready"


def _circuit_breaker_status(value: dict[str, Any]) -> str:
    if value["enabled"] is not True:
        return "ready"
    if value["tripped"] is True:
        return "blocked_circuit_breaker_tripped"
    sample_count = value["sample_count"]
    min_sample_count = value["min_sample_count"]
    if sample_count < min_sample_count:
        return "ready"
    fallback_rate = value["fallback_rate"]
    max_fallback_rate = value["max_fallback_rate"]
    if fallback_rate is not None and fallback_rate > max_fallback_rate:
        return "blocked_circuit_breaker_fallback_rate"
    error_rate = value["error_rate"]
    max_error_rate = value["max_error_rate"]
    if error_rate is not None and error_rate > max_error_rate:
        return "blocked_circuit_breaker_error_rate"
    return "ready"


def _public_circuit_breaker(controls: dict[str, Any]) -> dict[str, Any]:
    circuit = controls["circuit_breaker"]
    return {
        "enabled": circuit["enabled"],
        "sample_count": circuit["sample_count"],
        "min_sample_count": circuit["min_sample_count"],
        "fallback_rate": circuit["fallback_rate"],
        "max_fallback_rate": circuit["max_fallback_rate"],
        "error_rate": circuit["error_rate"],
        "max_error_rate": circuit["max_error_rate"],
        "status": _circuit_breaker_status(circuit),
    }


def _public_controls(controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": controls["enabled"],
        "allowed_app_ids": sorted(controls["allowed_app_ids"]),
        "allowed_adapter_ids": sorted(controls["allowed_adapter_ids"]),
        "allowed_input_sha256s": sorted(controls["allowed_input_sha256s"]),
        "allowed_intents": sorted(controls["allowed_intents"]),
        "allowed_tools": sorted(controls["allowed_tools"]),
        "circuit_breaker": _public_circuit_breaker(controls),
    }


def _controls(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("controls must be an object")
    controls = {
        "enabled": value.get("enabled") is True,
        "allowed_app_ids": _text_set(value.get("allowed_app_ids")),
        "allowed_adapter_ids": _text_set(value.get("allowed_adapter_ids")),
        "allowed_input_sha256s": _lower_text_set(value.get("allowed_input_sha256s")),
        "allowed_intents": _text_set(value.get("allowed_intents")),
        "allowed_tools": _text_set(value.get("allowed_tools")),
        "circuit_breaker": _circuit_breaker(value.get("circuit_breaker")),
    }
    if controls["enabled"] is True and controls["circuit_breaker"]["enabled"] is not True:
        raise ValueError("controls.enabled=True requires circuit_breaker.enabled=True")
    if controls["enabled"] is True and not controls["allowed_input_sha256s"]:
        raise ValueError("controls.enabled=True requires non-empty allowed_input_sha256s")
    return controls


def _circuit_breaker(value: Any) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError("controls.circuit_breaker must be an object")
    return {
        "enabled": value.get("enabled") is not False,
        "tripped": value.get("tripped") is True,
        "sample_count": _non_negative_int(value.get("sample_count"), default=0),
        "min_sample_count": _non_negative_int(value.get("min_sample_count"), default=100),
        "fallback_rate": _optional_rate(value.get("fallback_rate")),
        "max_fallback_rate": _rate(value.get("max_fallback_rate"), default=0.30),
        "error_rate": _optional_rate(value.get("error_rate")),
        "max_error_rate": _rate(value.get("max_error_rate"), default=0.05),
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


def _validated_candidate_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("candidate_gate must be an object")
    if value.get("ok") is not True:
        raise ValueError("candidate_gate.ok must be true")
    return value


def _candidate_intent(row: dict[str, Any]) -> str:
    value = _text(row.get("expected_intent"))
    if value:
        return value
    matrix_device = row.get("matrix_device") if isinstance(row.get("matrix_device"), dict) else {}
    value = _text(matrix_device.get("predicted_intent"))
    if value:
        return value
    evidence_route = row.get("evidence_route") if isinstance(row.get("evidence_route"), dict) else {}
    return _text(evidence_route.get("route_intent"))


def _matrix_threshold_passed(row: dict[str, Any]) -> bool:
    matrix_device = row.get("matrix_device") if isinstance(row.get("matrix_device"), dict) else {}
    return matrix_device.get("threshold_passed") is True


def _matrix_input_sha256(row: dict[str, Any]) -> str:
    matrix_device = row.get("matrix_device") if isinstance(row.get("matrix_device"), dict) else {}
    return _text(matrix_device.get("input_sha256")).lower()


def _selected_tools(row: dict[str, Any]) -> list[str]:
    runtime_validation = (
        row.get("runtime_validation") if isinstance(row.get("runtime_validation"), dict) else {}
    )
    values = _text_list(runtime_validation.get("selected_tools"))
    if values:
        return values
    evidence_route = row.get("evidence_route") if isinstance(row.get("evidence_route"), dict) else {}
    return _text_list(evidence_route.get("selected_tools"))


def _text_set(value: Any) -> set[str]:
    return set(_text_list(value))


def _lower_text_set(value: Any) -> set[str]:
    return {item.lower() for item in _text_list(value)}


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("expected a list of strings")
    out: list[str] = []
    for item in value:
        text = _text(item)
        if text:
            out.append(text)
    return sorted(set(out))


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _non_negative_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected non-negative integer") from exc
    if parsed < 0:
        raise ValueError("expected non-negative integer")
    return parsed


def _rate(value: Any, *, default: float) -> float:
    if value is None:
        return default
    parsed = _optional_rate(value)
    if parsed is None:
        return default
    return parsed


def _optional_rate(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected rate between 0 and 1") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("expected rate between 0 and 1")
    return parsed


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_LIVE_POLICY_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_live_policy_audit.v0",
            "method": "build_route_matrix_live_policy",
            "generated_at": generated_at,
        },
    }
