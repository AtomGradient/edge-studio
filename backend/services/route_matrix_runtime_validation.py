# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build runtime validation receipts for route-matrix tool candidates."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from backend.services.route_matrix_plan_gap_request import (
    ROUTE_MATRIX_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
)


def build_route_matrix_runtime_validation_receipt(request: dict[str, Any]) -> dict[str, Any]:
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
        candidate_gate = _candidate_gate(request.get("candidate_gate"))
        registry = _registry_by_name(request.get("tool_registry"))
        observed = _observed_tool_calls(request.get("observed_tool_calls"))
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix runtime validation request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    rows = [
        _validate_candidate(row, registry=registry, observed=observed)
        for row in _candidate_rows(candidate_gate)
        if _text(row.get("gate_status")) in {
            "blocked_missing_runtime_validation",
            "blocked_tool_registry_validation",
            "blocked_tool_call_plan_validation",
            "blocked_schema_validation",
            "validated_candidate_shadow_only",
            "eligible_for_live_routing",
        }
    ]
    summary = {
        "validation_count": len(rows),
        "all_valid": bool(rows) and all(
            row["tool_registry_ok"]
            and row["tool_call_plan_ok"]
            and row["schema_validation_ok"]
            for row in rows
        ),
        "ready_for_live_routing": False,
        "ready_for_live_routing_reason": (
            "runtime_validation_receipt_is_input_to_candidate_gate_not_a_release_switch"
        ),
    }
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "summary": summary,
            "validations": rows,
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_runtime_validation_audit.v0",
            "method": "build_route_matrix_runtime_validation_receipt",
            "generated_at": generated_at,
            "input_summary": {
                "candidate_count": len(_candidate_rows(candidate_gate)),
                "tool_count": len(registry),
                "observed_case_count": len(observed),
            },
        },
    }


def build_route_matrix_runtime_validation_receipt_from_files(
    *,
    run_id: str,
    candidate_gate_path: Path,
    tool_registry_path: Path,
    observed_tool_calls_path: Path | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "run_id": run_id,
        "candidate_gate": _read_json(candidate_gate_path),
        "tool_registry": _read_json(tool_registry_path),
    }
    if observed_tool_calls_path is not None:
        request["observed_tool_calls"] = _read_json(observed_tool_calls_path)
    return build_route_matrix_runtime_validation_receipt(request)


def _validate_candidate(
    row: dict[str, Any],
    *,
    registry: dict[str, dict[str, Any]],
    observed: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    case_id = _required_text(row.get("case_id"), "candidate.case_id")
    evidence = row.get("evidence_route") if isinstance(row.get("evidence_route"), dict) else {}
    runtime_validation = (
        evidence.get("route_matrix_runtime_validation")
        if isinstance(evidence.get("route_matrix_runtime_validation"), dict)
        else {}
    )
    selected_tools = _string_list(evidence.get("selected_tools")) or _string_list(
        runtime_validation.get("selected_tools")
    )
    tool_calls = (
        observed.get(case_id)
        or _list_of_dicts(evidence.get("tool_calls"))
        or _tool_calls_from_runtime_validation(runtime_validation)
    )
    tool_registry_ok = bool(selected_tools) and all(tool in registry for tool in selected_tools)
    tool_call_plan_ok = bool(tool_calls) and all(
        _tool_name(call) in selected_tools for call in tool_calls
    )
    schema_errors = _schema_errors(tool_calls=tool_calls, registry=registry)
    return {
        "case_id": case_id,
        "selected_tools": selected_tools,
        "tool_call_plan": tool_calls,
        "tool_registry_ok": tool_registry_ok,
        "tool_call_plan_ok": tool_call_plan_ok,
        "schema_validation_ok": not schema_errors and tool_call_plan_ok,
        "schema_errors": schema_errors,
        "source": "observed_tool_calls" if case_id in observed else "candidate_gate",
    }


def _tool_calls_from_runtime_validation(value: dict[str, Any]) -> list[dict[str, Any]]:
    tool_name = _text(value.get("tool_plan_name"))
    if not tool_name:
        return []
    arguments = value.get("arguments") if isinstance(value.get("arguments"), dict) else {}
    return [{"name": tool_name, "arguments": dict(arguments)}]


def _schema_errors(
    *,
    tool_calls: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        tool_name = _tool_name(call)
        if not tool_name or tool_name not in registry:
            errors.append({
                "plan_index": index,
                "tool_name": tool_name,
                "error": "tool_not_registered",
            })
            continue
        allowed = _allowed_arg_names(registry[tool_name])
        args = _tool_args(call)
        if allowed is None:
            continue
        unknown = sorted(key for key in args if key not in allowed)
        if unknown:
            errors.append({
                "plan_index": index,
                "tool_name": tool_name,
                "error": "unknown_arguments",
                "unknown_args": unknown,
            })
    return errors


def _candidate_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("candidate_gate must be an object")
    if value.get("ok") is not True:
        raise ValueError("candidate_gate.ok must be true")
    return value


def _candidate_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    rows = result.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("candidate_gate.result.candidates must be a list")
    return [row for row in rows if isinstance(row, dict)]


def _registry_by_name(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("tool_registry must be a list")
    out: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise TypeError(f"tool_registry[{index}] must be an object")
        name = _required_text(
            raw.get("name") or raw.get("tool_name") or raw.get("toolName"),
            f"tool_registry[{index}].name",
        )
        if name in out:
            raise ValueError(f"duplicate tool name: {name}")
        tool = dict(raw)
        tool["name"] = name
        out[name] = tool
    if not out:
        raise ValueError("tool_registry must contain at least one tool")
    return out


def _observed_tool_calls(value: Any) -> dict[str, list[dict[str, Any]]]:
    if value is None:
        return {}
    if isinstance(value, dict) and isinstance(value.get("observations"), list):
        value = value["observations"]
    if isinstance(value, dict):
        return {
            _required_text(case_id, "observed_tool_calls.case_id"): _list_of_dicts(tool_calls)
            for case_id, tool_calls in value.items()
        }
    if not isinstance(value, list):
        raise TypeError("observed_tool_calls must be a list or object")
    out: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise TypeError(f"observed_tool_calls[{index}] must be an object")
        case_id = _required_text(row.get("case_id"), f"observed_tool_calls[{index}].case_id")
        out[case_id] = _list_of_dicts(row.get("tool_calls") or row.get("tool_call_plan"))
    return out


def _allowed_arg_names(tool: dict[str, Any]) -> set[str] | None:
    for key in ("args_schema", "arguments_schema", "schema"):
        raw = tool.get(key)
        schema = raw if isinstance(raw, dict) else None
        if schema is None:
            continue
        properties = schema.get("properties")
        if isinstance(properties, dict):
            return {str(name) for name in properties}
    return None


def _tool_name(entry: dict[str, Any]) -> str:
    for key in ("tool_name", "toolName", "tool", "name"):
        text = _text(entry.get(key))
        if text:
            return text
    return ""


def _tool_args(entry: dict[str, Any]) -> dict[str, Any]:
    for key in ("args", "arguments"):
        value = entry.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


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
        "schema_version": ROUTE_MATRIX_RUNTIME_VALIDATION_RECEIPT_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_runtime_validation_audit.v0",
            "method": "build_route_matrix_runtime_validation_receipt",
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
