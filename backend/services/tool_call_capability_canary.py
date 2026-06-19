# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Raw-free tool-call capability canary result builder and validator."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping


TOOL_CALL_CAPABILITY_CANARY_SCHEMA_VERSION = (
    "edgestudio.tool_call_capability_canary.v1"
)
ALLOWED_MODES = {"real_model", "parser_fixture"}
ALLOWED_SURFACE_KINDS = {"host_cli", "device_app"}
RAW_TEXT_KEYS = {
    "answer",
    "assistant_response",
    "expected_text",
    "expected_tool",
    "generated_text",
    "messages",
    "prompt",
    "question",
    "raw_text",
    "response",
    "text",
    "tool_arguments",
    "transcript",
    "user_text",
}


def build_tool_call_capability_canary_result(
    *,
    run_id: str,
    mode: str,
    surface_kind: str,
    model_fingerprint: str,
    tool_schema_sha256: str,
    prompt_set_sha256: str,
    case_refs: list[Mapping[str, Any]],
    blockers: list[str] | None = None,
    planned_tool_calls_used: bool = False,
    learning_triggered: bool = False,
    runtime_artifacts_mutated: bool = False,
    raw_text_included: bool = False,
    claims_behavior_improved: bool = False,
    claims_model_quality_improved: bool = False,
    claims_router_quality_improved: bool = False,
    claims_production_learning_shipped: bool = False,
) -> dict[str, Any]:
    """Build a canary result from already raw-free case observations."""

    normalized_cases = [_case_ref(item) for item in case_refs]
    counts = _counts(normalized_cases)
    all_blockers = sorted(
        set(_text_list(blockers) + _derived_blockers(
            mode=mode,
            surface_kind=surface_kind,
            counts=counts,
            case_refs=normalized_cases,
            planned_tool_calls_used=planned_tool_calls_used,
            learning_triggered=learning_triggered,
            runtime_artifacts_mutated=runtime_artifacts_mutated,
            raw_text_included=raw_text_included,
            claims_behavior_improved=claims_behavior_improved,
            claims_model_quality_improved=claims_model_quality_improved,
            claims_router_quality_improved=claims_router_quality_improved,
            claims_production_learning_shipped=claims_production_learning_shipped,
        ))
    )
    status = "passed" if not all_blockers else "blocked"

    return {
        "schema_version": TOOL_CALL_CAPABILITY_CANARY_SCHEMA_VERSION,
        "run_id": _required_text(run_id, "run_id"),
        "generated_at": _utc_now(),
        "status": status,
        "mode": _required_text(mode, "mode"),
        "surface_kind": _required_text(surface_kind, "surface_kind"),
        "model_fingerprint": _normalize_sha256(model_fingerprint),
        "tool_schema_sha256": _normalize_sha256(tool_schema_sha256),
        "prompt_set_sha256": _normalize_sha256(prompt_set_sha256),
        "case_count": len(normalized_cases),
        "completed_count": counts["completed_count"],
        "valid_tool_call_count": counts["valid_tool_call_count"],
        "invalid_tool_call_count": counts["invalid_tool_call_count"],
        "no_tool_call_count": counts["no_tool_call_count"],
        "disallowed_tool_call_count": counts["disallowed_tool_call_count"],
        "raw_text_included": raw_text_included is True,
        "planned_tool_calls_used": planned_tool_calls_used is True,
        "learning_triggered": learning_triggered is True,
        "runtime_artifacts_mutated": runtime_artifacts_mutated is True,
        "claims_behavior_improved": claims_behavior_improved is True,
        "claims_model_quality_improved": claims_model_quality_improved is True,
        "claims_router_quality_improved": claims_router_quality_improved is True,
        "claims_production_learning_shipped": claims_production_learning_shipped is True,
        "case_refs": normalized_cases,
        "blockers": all_blockers,
    }


def validate_tool_call_capability_canary_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canary result stays raw-free, app-neutral, and fail-closed."""

    errors: list[dict[str, str]] = []
    if result.get("schema_version") != TOOL_CALL_CAPABILITY_CANARY_SCHEMA_VERSION:
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if result.get("status") not in {"passed", "blocked"}:
        errors.append({"code": "invalid_status", "field": "status"})
    if result.get("mode") not in ALLOWED_MODES:
        errors.append({"code": "invalid_mode", "field": "mode"})
    if result.get("surface_kind") not in ALLOWED_SURFACE_KINDS:
        errors.append({"code": "invalid_surface_kind", "field": "surface_kind"})
    for field in ("run_id", "generated_at"):
        if not _text(result.get(field)):
            errors.append({"code": f"missing_{field}", "field": field})
    for field in (
        "model_fingerprint",
        "tool_schema_sha256",
        "prompt_set_sha256",
    ):
        if not _valid_sha256(result.get(field)):
            errors.append({"code": f"invalid_{field}", "field": field})
    for field in (
        "raw_text_included",
        "planned_tool_calls_used",
        "learning_triggered",
        "runtime_artifacts_mutated",
        "claims_behavior_improved",
        "claims_model_quality_improved",
        "claims_router_quality_improved",
        "claims_production_learning_shipped",
    ):
        if result.get(field) is not False:
            errors.append({"code": field, "field": field})
    for field in _raw_text_fields(result):
        errors.append({"code": "raw_text_field_present", "field": field})

    case_refs = result.get("case_refs")
    if not isinstance(case_refs, list):
        errors.append({"code": "missing_case_refs", "field": "case_refs"})
        case_refs = []
    normalized_cases = [_case_ref(item) for item in case_refs if isinstance(item, Mapping)]
    counts = _counts(normalized_cases)
    expected_counts = {
        "case_count": len(normalized_cases),
        "completed_count": counts["completed_count"],
        "valid_tool_call_count": counts["valid_tool_call_count"],
        "invalid_tool_call_count": counts["invalid_tool_call_count"],
        "no_tool_call_count": counts["no_tool_call_count"],
        "disallowed_tool_call_count": counts["disallowed_tool_call_count"],
    }
    for field, expected in expected_counts.items():
        if _optional_int(result.get(field)) != expected:
            errors.append({"code": f"{field}_mismatch", "field": field})

    for index, item in enumerate(case_refs):
        if not isinstance(item, Mapping):
            errors.append({"code": "invalid_case_ref", "field": f"case_refs[{index}]"})
            continue
        if not _text(item.get("case_id")):
            errors.append({"code": "missing_case_id", "field": f"case_refs[{index}].case_id"})
        for field in ("prompt_sha256", "answer_sha256"):
            if not _valid_sha256(item.get(field)):
                errors.append(
                    {"code": f"invalid_{field}", "field": f"case_refs[{index}].{field}"}
                )
        for field in (
            "selected_tool_count",
            "tool_call_count",
            "valid_tool_call_count",
            "invalid_tool_call_count",
            "no_tool_call_count",
            "disallowed_tool_call_count",
        ):
            value = _optional_int(item.get(field))
            if value is None or value < 0:
                errors.append(
                    {"code": f"invalid_{field}", "field": f"case_refs[{index}].{field}"}
                )
        if item.get("completed") is not None and not isinstance(item.get("completed"), bool):
            errors.append({"code": "invalid_completed", "field": f"case_refs[{index}].completed"})
        call_total = (
            (_optional_int(item.get("valid_tool_call_count")) or 0)
            + (_optional_int(item.get("invalid_tool_call_count")) or 0)
            + (_optional_int(item.get("disallowed_tool_call_count")) or 0)
        )
        if _optional_int(item.get("tool_call_count")) != call_total:
            errors.append(
                {"code": "tool_call_count_mismatch", "field": f"case_refs[{index}]"}
            )

    blockers = result.get("blockers")
    if not isinstance(blockers, list) or not all(_text(item) for item in blockers):
        errors.append({"code": "invalid_blockers", "field": "blockers"})
        blockers = []

    if result.get("status") == "passed":
        if blockers:
            errors.append({"code": "passed_has_blockers", "field": "blockers"})
        if result.get("mode") != "real_model":
            errors.append({"code": "passed_without_real_model", "field": "mode"})
        if counts["valid_tool_call_count"] <= 0:
            errors.append({"code": "passed_without_valid_tool_call", "field": "valid_tool_call_count"})
        if counts["completed_count"] != len(normalized_cases):
            errors.append({"code": "passed_with_incomplete_cases", "field": "completed_count"})
        if (
            counts["invalid_tool_call_count"] > 0
            or counts["no_tool_call_count"] > 0
            or counts["disallowed_tool_call_count"] > 0
        ):
            errors.append({"code": "passed_with_blocking_counts", "field": "status"})
    if result.get("status") == "blocked" and not blockers:
        errors.append({"code": "blocked_without_blockers", "field": "blockers"})
    return {"ok": not errors, "errors": errors}


def _case_ref(item: Mapping[str, Any]) -> dict[str, Any]:
    valid_count = _non_negative_int(item.get("valid_tool_call_count"))
    invalid_count = _non_negative_int(item.get("invalid_tool_call_count"))
    disallowed_count = _non_negative_int(item.get("disallowed_tool_call_count"))
    default_tool_count = valid_count + invalid_count + disallowed_count
    return {
        "case_id": _text(item.get("case_id")),
        "prompt_sha256": _normalize_sha256(item.get("prompt_sha256")),
        "answer_sha256": _normalize_sha256(item.get("answer_sha256")),
        "completed": item.get("completed") is not False,
        "selected_tool_count": _non_negative_int(item.get("selected_tool_count")),
        "tool_call_count": _non_negative_int(item.get("tool_call_count"), default_tool_count),
        "valid_tool_call_count": valid_count,
        "invalid_tool_call_count": invalid_count,
        "no_tool_call_count": _non_negative_int(item.get("no_tool_call_count")),
        "disallowed_tool_call_count": disallowed_count,
    }


def _counts(case_refs: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "completed_count": sum(1 for item in case_refs if item.get("completed") is True),
        "valid_tool_call_count": sum(
            _non_negative_int(item.get("valid_tool_call_count")) for item in case_refs
        ),
        "invalid_tool_call_count": sum(
            _non_negative_int(item.get("invalid_tool_call_count")) for item in case_refs
        ),
        "no_tool_call_count": sum(
            _non_negative_int(item.get("no_tool_call_count")) for item in case_refs
        ),
        "disallowed_tool_call_count": sum(
            _non_negative_int(item.get("disallowed_tool_call_count")) for item in case_refs
        ),
    }


def _derived_blockers(
    *,
    mode: str,
    surface_kind: str,
    counts: Mapping[str, int],
    case_refs: list[Mapping[str, Any]],
    planned_tool_calls_used: bool,
    learning_triggered: bool,
    runtime_artifacts_mutated: bool,
    raw_text_included: bool,
    claims_behavior_improved: bool,
    claims_model_quality_improved: bool,
    claims_router_quality_improved: bool,
    claims_production_learning_shipped: bool,
) -> list[str]:
    blockers: list[str] = []
    if mode not in ALLOWED_MODES:
        blockers.append("invalid_mode")
    elif mode != "real_model":
        blockers.append("real_model_required")
    if surface_kind not in ALLOWED_SURFACE_KINDS:
        blockers.append("invalid_surface_kind")
    if not case_refs:
        blockers.append("missing_cases")
    if case_refs and any((_optional_int(item.get("selected_tool_count")) or 0) <= 0 for item in case_refs):
        blockers.append("explicit_tool_schema_required")
    if counts["completed_count"] != len(case_refs):
        blockers.append("case_not_completed")
    if counts["valid_tool_call_count"] <= 0:
        blockers.append("no_valid_tool_calls")
    if counts["invalid_tool_call_count"] > 0:
        blockers.append("invalid_tool_calls")
    if counts["no_tool_call_count"] > 0:
        blockers.append("missing_model_tool_call")
    if counts["disallowed_tool_call_count"] > 0:
        blockers.append("disallowed_tool_calls")
    for flag, code in (
        (planned_tool_calls_used, "planned_tool_calls_used"),
        (learning_triggered, "learning_triggered"),
        (runtime_artifacts_mutated, "runtime_artifacts_mutated"),
        (raw_text_included, "raw_text_included"),
        (claims_behavior_improved, "claims_behavior_improved"),
        (claims_model_quality_improved, "claims_model_quality_improved"),
        (claims_router_quality_improved, "claims_router_quality_improved"),
        (claims_production_learning_shipped, "claims_production_learning_shipped"),
    ):
        if flag:
            blockers.append(code)
    return blockers


def _raw_text_fields(value: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in RAW_TEXT_KEYS:
                fields.append(path)
            fields.extend(_raw_text_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            fields.extend(_raw_text_fields(item, f"{prefix}[{index}]"))
    return fields


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _non_negative_int(value: Any, default: int = 0) -> int:
    number = _optional_int(value)
    if number is None or number < 0:
        return default
    return number


def _normalize_sha256(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    lower = text.lower()
    if lower.startswith("sha256:"):
        lower = lower.removeprefix("sha256:")
    if len(lower) == 64 and all(char in "0123456789abcdef" for char in lower):
        return f"sha256:{lower}"
    return None


def _valid_sha256(value: Any) -> bool:
    return _normalize_sha256(value) is not None


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
