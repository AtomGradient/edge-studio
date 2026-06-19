# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Raw-free heldout runtime diagnostic for Learning Flywheel receipts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping


HELDOUT_RUNTIME_DIAGNOSTIC_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.heldout_runtime_diagnostic.v1"
)
HELDOUT_AUTORUN_RESULT_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.heldout_eval_autorun_result.v0"
)
TOOL_CALL_CAPABILITY_CANARY_SCHEMA_VERSION = (
    "edgestudio.tool_call_capability_canary.v1"
)

RAW_TEXT_KEYS = {
    "answer",
    "assistant_response",
    "correction_text",
    "expected_text",
    "expected_tool",
    "generated_text",
    "golden_answer",
    "messages",
    "prompt",
    "question",
    "rationale",
    "raw_text",
    "reference_answer",
    "response",
    "selected_tools",
    "text",
    "tool_arguments",
    "tool_calls",
    "tool_name",
    "transcript",
    "user_text",
}

NO_CLAIM_FLAGS = (
    "writes_runtime_artifacts",
    "runs_device_harness",
    "invokes_host_model_judge",
    "triggers_learning",
    "mutates_artifacts",
    "claims_behavior_improved",
    "claims_model_quality_improved",
    "claims_router_quality_improved",
    "claims_production_learning_shipped",
    "raw_text_included",
    "legacy_expected_hint_fields_included",
)


def build_learning_flywheel_heldout_runtime_diagnostic(
    *,
    run_id: str,
    evidence_scope: str,
    before_heldout_result: Mapping[str, Any],
    after_heldout_result: Mapping[str, Any],
    tool_call_canary_results: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a read-only diagnostic from existing hash-only runtime receipts."""

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    canaries = tool_call_canary_results or []
    before_summary = _heldout_summary(before_heldout_result)
    after_summary = _heldout_summary(after_heldout_result)
    before_cases = _heldout_case_map(before_heldout_result)
    after_cases = _heldout_case_map(after_heldout_result)
    case_ids = sorted(set(before_cases) | set(after_cases))
    case_diagnostics = [
        _case_diagnostic(case_id, before_cases.get(case_id), after_cases.get(case_id))
        for case_id in case_ids
    ]
    canary_summary = _canary_summary(canaries)
    diagnostic_codes = _diagnostic_codes(
        before_summary=before_summary,
        after_summary=after_summary,
        canary_summary=canary_summary,
    )
    evidence_blockers = _evidence_blockers(diagnostic_codes)
    input_errors = _input_errors(
        before_heldout_result=before_heldout_result,
        after_heldout_result=after_heldout_result,
        tool_call_canary_results=canaries,
    )
    status = "diagnosed" if not input_errors else "blocked"
    claimability = {
        "fact_tool_behavior_quality_claimable": False,
        "answer_quality_claimable": False,
        "paired_delta_computable": False,
        "behavior_improvement_claimable": False,
        "model_quality_improvement_claimable": False,
        "router_quality_improvement_claimable": False,
        "production_learning_shipped_claimable": False,
    }
    payload_without_hash = {
        "ok": status == "diagnosed",
        "schema_version": HELDOUT_RUNTIME_DIAGNOSTIC_SCHEMA_VERSION,
        "status": status,
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "evidence_scope": normalized_scope,
        "writes_runtime_artifacts": False,
        "runs_device_harness": False,
        "invokes_host_model_judge": False,
        "triggers_learning": False,
        "mutates_artifacts": False,
        "claims_behavior_improved": False,
        "claims_model_quality_improved": False,
        "claims_router_quality_improved": False,
        "claims_production_learning_shipped": False,
        "raw_text_included": False,
        "legacy_expected_hint_fields_included": False,
        "sources": {
            "before_heldout_result": _source_ref(before_heldout_result),
            "after_heldout_result": _source_ref(after_heldout_result),
            "tool_call_canary_results": [_source_ref(item) for item in canaries],
        },
        "summary": {
            "before": before_summary,
            "after": after_summary,
            "paired_shape": _paired_shape_summary(
                before_summary=before_summary,
                after_summary=after_summary,
            ),
            "tool_call_canaries": canary_summary,
        },
        "claimability": claimability,
        "diagnostic_codes": diagnostic_codes,
        "blockers": evidence_blockers,
        "input_errors": input_errors,
        "case_diagnostics_sha256": _sha256_json(case_diagnostics),
        "case_diagnostics": case_diagnostics,
    }
    return {
        **payload_without_hash,
        "diagnostic_sha256": _sha256_json(payload_without_hash),
    }


def validate_learning_flywheel_heldout_runtime_diagnostic(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the diagnostic stays raw-free, read-only, and no-claim."""

    errors: list[dict[str, str]] = []
    if diagnostic.get("schema_version") != HELDOUT_RUNTIME_DIAGNOSTIC_SCHEMA_VERSION:
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if diagnostic.get("status") not in {"diagnosed", "blocked"}:
        errors.append({"code": "invalid_status", "field": "status"})
    for field in NO_CLAIM_FLAGS:
        if diagnostic.get(field) is not False:
            errors.append({"code": field, "field": field})
    for field in _raw_text_fields(diagnostic):
        errors.append({"code": "raw_text_field_present", "field": field})

    claimability = _mapping(diagnostic.get("claimability"))
    for field in (
        "fact_tool_behavior_quality_claimable",
        "answer_quality_claimable",
        "paired_delta_computable",
        "behavior_improvement_claimable",
        "model_quality_improvement_claimable",
        "router_quality_improvement_claimable",
        "production_learning_shipped_claimable",
    ):
        if claimability.get(field) is not False:
            errors.append({"code": f"claimability_{field}", "field": f"claimability.{field}"})

    case_diagnostics = diagnostic.get("case_diagnostics")
    if not isinstance(case_diagnostics, list):
        errors.append({"code": "missing_case_diagnostics", "field": "case_diagnostics"})
        case_diagnostics = []
    elif _sha256_json(case_diagnostics) != diagnostic.get("case_diagnostics_sha256"):
        errors.append(
            {"code": "case_diagnostics_hash_mismatch", "field": "case_diagnostics_sha256"}
        )

    for index, item in enumerate(case_diagnostics):
        if not isinstance(item, Mapping):
            errors.append(
                {"code": "invalid_case_diagnostic", "field": f"case_diagnostics[{index}]"}
            )
            continue
        if not _text(item.get("case_id")):
            errors.append(
                {"code": "missing_case_id", "field": f"case_diagnostics[{index}].case_id"}
            )
        for phase in ("before", "after"):
            phase_item = item.get(phase) if isinstance(item.get(phase), Mapping) else {}
            for field in ("prompt_sha256", "answer_sha256"):
                value = phase_item.get(field)
                if value is not None and not _valid_sha256(value):
                    errors.append(
                        {
                            "code": f"invalid_{field}",
                            "field": f"case_diagnostics[{index}].{phase}.{field}",
                        }
                    )

    diagnostic_codes = _text_list(diagnostic.get("diagnostic_codes"))
    if not diagnostic_codes:
        errors.append({"code": "missing_diagnostic_codes", "field": "diagnostic_codes"})
    if (
        "all_after_case_tool_call_count_zero" in diagnostic_codes
        and "behavior_improvement_not_claimable" not in diagnostic_codes
    ):
        errors.append(
            {
                "code": "tool_call_zero_without_no_improvement_claim",
                "field": "diagnostic_codes",
            }
        )

    blockers = diagnostic.get("blockers")
    if not isinstance(blockers, list):
        errors.append({"code": "invalid_blockers", "field": "blockers"})
        blockers = []
    else:
        last_priority = 0
        for index, blocker in enumerate(blockers):
            if not isinstance(blocker, Mapping):
                errors.append({"code": "invalid_blocker", "field": f"blockers[{index}]"})
                continue
            priority = _optional_int(blocker.get("priority"))
            if priority is None or priority <= 0:
                errors.append(
                    {"code": "invalid_blocker_priority", "field": f"blockers[{index}].priority"}
                )
            elif priority < last_priority:
                errors.append({"code": "blockers_not_ordered", "field": "blockers"})
            else:
                last_priority = priority
            if not _text(blocker.get("code")):
                errors.append({"code": "missing_blocker_code", "field": f"blockers[{index}].code"})

    if not _valid_sha256(diagnostic.get("diagnostic_sha256")):
        errors.append({"code": "invalid_diagnostic_sha256", "field": "diagnostic_sha256"})
    else:
        expected = _sha256_json(
            {key: value for key, value in diagnostic.items() if key != "diagnostic_sha256"}
        )
        if diagnostic.get("diagnostic_sha256") != expected:
            errors.append({"code": "diagnostic_hash_mismatch", "field": "diagnostic_sha256"})

    if diagnostic.get("status") == "diagnosed":
        if diagnostic.get("ok") is not True:
            errors.append({"code": "diagnosed_not_ok", "field": "ok"})
        if diagnostic.get("input_errors") not in ([], ()):
            errors.append({"code": "diagnosed_has_input_errors", "field": "input_errors"})
    if diagnostic.get("status") == "blocked" and diagnostic.get("ok") is not False:
        errors.append({"code": "blocked_ok", "field": "ok"})
    return {"ok": not errors, "errors": errors}


def _heldout_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    cases = _object_list(payload.get("case_refs"))
    fact_cases = [item for item in cases if _text(item.get("primary_class")) == "fact_tool_behavior"]
    return {
        "schema_version": _text(payload.get("schema_version")),
        "status": _text(payload.get("status")),
        "phase": _text(payload.get("phase")),
        "case_count": _optional_int(payload.get("case_count")),
        "started_count": _optional_int(payload.get("started_count")),
        "completed_count": _optional_int(payload.get("completed_count")),
        "matched_count": _optional_int(payload.get("matched_count")),
        "error_count": _optional_int(payload.get("error_count")),
        "host_model_invoked": payload.get("host_model_invoked") is True,
        "learning_triggered": payload.get("learning_triggered") is True,
        "runtime_artifacts_mutated": payload.get("runtime_artifacts_mutated") is True,
        "raw_text_included": payload.get("raw_text_included") is True,
        "neural_imprint_active": payload.get("neural_imprint_active") is True,
        "normal_app_runtime_path": payload.get("normal_app_runtime_path") is True,
        "session_mode": _text(payload.get("session_mode")),
        "case_refs_neural_imprint_active_count": sum(
            1 for item in cases if item.get("neural_imprint_active") is True
        ),
        "selected_tool_count_total": sum(
            _non_negative_int(item.get("selected_tool_count")) for item in cases
        ),
        "tool_call_count_total": sum(
            _non_negative_int(item.get("tool_call_count")) for item in cases
        ),
        "fact_tool_behavior_case_count": len(fact_cases),
        "fact_tool_behavior_tool_call_count": sum(
            _non_negative_int(item.get("tool_call_count")) for item in fact_cases
        ),
        "primary_class_distribution": _primary_class_distribution(cases),
    }


def _heldout_case_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _object_list(payload.get("case_refs")):
        case_id = _text(row.get("case_id"))
        if not case_id:
            continue
        out[case_id] = {
            "case_id": case_id,
            "primary_class": _text(row.get("primary_class")),
            "prompt_sha256": _normalize_sha256(row.get("prompt_sha256")),
            "answer_sha256": _normalize_sha256(row.get("answer_sha256")),
            "completed": row.get("completed") is True,
            "fact_tool_called": row.get("fact_tool_called") is True,
            "selected_tool_count": _non_negative_int(row.get("selected_tool_count")),
            "tool_call_count": _non_negative_int(row.get("tool_call_count")),
            "neural_imprint_active": row.get("neural_imprint_active") is True,
        }
    return out


def _case_diagnostic(
    case_id: str,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    primary_class = _text(_mapping(after).get("primary_class")) or _text(
        _mapping(before).get("primary_class")
    )
    return {
        "case_id": case_id,
        "primary_class": primary_class,
        "before": _phase_case(before),
        "after": _phase_case(after),
    }


def _phase_case(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = _mapping(row)
    return {
        "prompt_sha256": _normalize_sha256(row.get("prompt_sha256")),
        "answer_sha256": _normalize_sha256(row.get("answer_sha256")),
        "completed": row.get("completed") is True,
        "fact_tool_called": row.get("fact_tool_called") is True,
        "selected_tool_count": _non_negative_int(row.get("selected_tool_count")),
        "tool_call_count": _non_negative_int(row.get("tool_call_count")),
        "neural_imprint_active": row.get("neural_imprint_active") is True,
    }


def _canary_summary(canaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "result_count": len(canaries),
        "passed_count": sum(1 for item in canaries if item.get("status") == "passed"),
        "blocked_count": sum(1 for item in canaries if item.get("status") == "blocked"),
        "real_model_result_count": sum(1 for item in canaries if item.get("mode") == "real_model"),
        "device_app_result_count": sum(
            1 for item in canaries if item.get("surface_kind") == "device_app"
        ),
        "case_count": sum(_non_negative_int(item.get("case_count")) for item in canaries),
        "completed_count": sum(
            _non_negative_int(item.get("completed_count")) for item in canaries
        ),
        "valid_tool_call_count": sum(
            _non_negative_int(item.get("valid_tool_call_count")) for item in canaries
        ),
        "invalid_tool_call_count": sum(
            _non_negative_int(item.get("invalid_tool_call_count")) for item in canaries
        ),
        "no_tool_call_count": sum(
            _non_negative_int(item.get("no_tool_call_count")) for item in canaries
        ),
        "disallowed_tool_call_count": sum(
            _non_negative_int(item.get("disallowed_tool_call_count")) for item in canaries
        ),
    }


def _paired_shape_summary(
    *,
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
) -> dict[str, Any]:
    before_distribution = _mapping(before_summary.get("primary_class_distribution"))
    after_distribution = _mapping(after_summary.get("primary_class_distribution"))
    return {
        "before_neural_imprint_active": before_summary.get("neural_imprint_active") is True,
        "after_neural_imprint_active": after_summary.get("neural_imprint_active") is True,
        "case_count_before": _optional_int(before_summary.get("case_count")),
        "case_count_after": _optional_int(after_summary.get("case_count")),
        "case_count_match": before_summary.get("case_count") == after_summary.get("case_count"),
        "primary_class_distribution_before": before_distribution,
        "primary_class_distribution_after": after_distribution,
        "primary_class_distribution_match": before_distribution == after_distribution,
    }


def _diagnostic_codes(
    *,
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
    canary_summary: Mapping[str, Any],
) -> list[str]:
    codes: set[str] = set()
    if before_summary.get("status") == "passed":
        codes.add("before_heldout_result_passed")
    if after_summary.get("status") == "passed":
        codes.add("after_heldout_result_passed")
    if after_summary.get("neural_imprint_active") is True:
        codes.add("after_neural_imprint_active")
        codes.add("active_ni_after_receipt_structural_only")
    after_case_count = _optional_int(after_summary.get("case_count")) or 0
    after_tool_count = _optional_int(after_summary.get("tool_call_count_total")) or 0
    if after_case_count > 0 and after_tool_count == 0:
        codes.add("all_after_case_tool_call_count_zero")
    fact_case_count = _optional_int(after_summary.get("fact_tool_behavior_case_count")) or 0
    fact_tool_count = (
        _optional_int(after_summary.get("fact_tool_behavior_tool_call_count")) or 0
    )
    if fact_case_count > 0 and fact_tool_count == 0:
        codes.add("fact_tool_behavior_tool_calls_absent")
    if (
        (_optional_int(canary_summary.get("passed_count")) or 0) > 0
        and (_optional_int(canary_summary.get("valid_tool_call_count")) or 0) > 0
        and after_case_count > 0
        and after_tool_count == 0
    ):
        codes.add("canary_passed_but_learning_flywheel_heldout_tool_calls_absent")
    if before_summary.get("case_count") != after_summary.get("case_count"):
        codes.add("before_after_case_count_mismatch")
    if _mapping(before_summary.get("primary_class_distribution")) != _mapping(
        after_summary.get("primary_class_distribution")
    ):
        codes.add("before_after_primary_class_distribution_mismatch")
    codes.add("host_answer_quality_not_evaluated")
    codes.add("paired_delta_not_computable")
    codes.add("behavior_improvement_not_claimable")
    return sorted(codes)


def _evidence_blockers(diagnostic_codes: list[str]) -> list[dict[str, Any]]:
    ordered = [
        (
            "fact_tool_behavior_tool_calls_absent",
            "Learning Flywheel heldout fact-tool cases did not emit model tool calls.",
        ),
        (
            "all_after_case_tool_call_count_zero",
            "Every after-phase heldout case reported zero tool calls.",
        ),
        (
            "host_answer_quality_not_evaluated",
            "No Host Model answer-quality gate has judged the heldout answers.",
        ),
        (
            "paired_delta_not_computable",
            "Before/after behavior delta is not computable without answer-quality review.",
        ),
        (
            "canary_passed_but_learning_flywheel_heldout_tool_calls_absent",
            "Tool-call canary proves capability only, not Learning Flywheel heldout behavior.",
        ),
        (
            "behavior_improvement_not_claimable",
            "The current evidence cannot claim Learning Flywheel behavior improvement.",
        ),
    ]
    code_set = set(diagnostic_codes)
    return [
        {
            "priority": index + 1,
            "code": code,
            "status": "open",
            "detail": detail,
        }
        for index, (code, detail) in enumerate(ordered)
        if code in code_set
    ]


def _input_errors(
    *,
    before_heldout_result: Mapping[str, Any],
    after_heldout_result: Mapping[str, Any],
    tool_call_canary_results: list[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for label, payload, expected_schema in (
        ("before_heldout_result", before_heldout_result, HELDOUT_AUTORUN_RESULT_SCHEMA_VERSION),
        ("after_heldout_result", after_heldout_result, HELDOUT_AUTORUN_RESULT_SCHEMA_VERSION),
    ):
        if payload.get("schema_version") != expected_schema:
            errors.append(f"{label}_schema_version_mismatch")
        if label.startswith("before") and payload.get("phase") != "before":
            errors.append("before_heldout_result_phase_mismatch")
        if label.startswith("after") and payload.get("phase") != "after":
            errors.append("after_heldout_result_phase_mismatch")
        if payload.get("raw_text_included") is True:
            errors.append(f"{label}_raw_text_included")
        for field in _raw_text_fields(payload):
            errors.append(f"{label}_raw_text_field_present:{field}")
    for index, payload in enumerate(tool_call_canary_results):
        if payload.get("schema_version") != TOOL_CALL_CAPABILITY_CANARY_SCHEMA_VERSION:
            errors.append(f"tool_call_canary_results[{index}]_schema_version_mismatch")
        if payload.get("raw_text_included") is True:
            errors.append(f"tool_call_canary_results[{index}]_raw_text_included")
        for field in _raw_text_fields(payload):
            errors.append(f"tool_call_canary_results[{index}]_raw_text_field_present:{field}")
    return sorted(set(errors))


def _source_ref(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _text(payload.get("schema_version")),
        "status": _text(payload.get("status")),
        "phase": _text(payload.get("phase")),
        "content_sha256": _sha256_json(payload),
    }


def _primary_class_distribution(cases: list[Mapping[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for item in cases:
        key = _text(item.get("primary_class")) or "unspecified"
        distribution[key] = distribution.get(key, 0) + 1
    return dict(sorted(distribution.items()))


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _object_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


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
