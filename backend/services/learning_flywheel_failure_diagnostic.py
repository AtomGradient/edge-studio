# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Raw-free Learning Flywheel paired-eval failure diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping


LEARNING_FLYWHEEL_FAILURE_DIAGNOSTIC_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.failure_diagnostic.v1"
)
RAW_TEXT_KEYS = {
    "answer",
    "assistant_response",
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
    "text",
    "transcript",
    "user_text",
}


def build_learning_flywheel_failure_diagnostic(
    *,
    run_id: str,
    evidence_scope: str,
    before_review: Mapping[str, Any],
    after_review: Mapping[str, Any],
    host_answer_quality_gate: Mapping[str, Any],
    paired_delta_receipt: Mapping[str, Any],
    paired_eval_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw-free diagnostic from existing paired-eval gate artifacts."""

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    before_summary = _review_summary(before_review)
    after_summary = _review_summary(after_review)
    before_cases = _case_map(before_review)
    after_cases = _case_map(after_review)
    paired_cases = _paired_case_map(paired_delta_receipt)
    host_gate_summary = _mapping(host_answer_quality_gate.get("summary"))
    paired_summary = _paired_summary(paired_delta_receipt)
    plan_blockers = _text_list(_mapping(paired_eval_plan).get("blockers"))
    case_ids = sorted(set(before_cases) | set(after_cases))
    case_diagnostics = [
        _case_diagnostic(
            case_id,
            before_cases.get(case_id),
            after_cases.get(case_id),
            paired_cases.get(case_id),
        )
        for case_id in case_ids
    ]
    diagnostic_codes = _diagnostic_codes(
        before_summary=before_summary,
        after_summary=after_summary,
        host_answer_quality_gate=host_answer_quality_gate,
        paired_delta_receipt=paired_delta_receipt,
        paired_summary=paired_summary,
        plan_blockers=plan_blockers,
        case_diagnostics=case_diagnostics,
    )
    blockers = _blockers(
        before_review=before_review,
        after_review=after_review,
        host_answer_quality_gate=host_answer_quality_gate,
        paired_delta_receipt=paired_delta_receipt,
        diagnostic_codes=diagnostic_codes,
    )
    status = "diagnosed" if not blockers and diagnostic_codes else "blocked"
    payload_without_hash = {
        "ok": status == "diagnosed",
        "schema_version": LEARNING_FLYWHEEL_FAILURE_DIAGNOSTIC_SCHEMA_VERSION,
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
            "before_host_answer_quality_review": _source_ref(before_review),
            "after_host_answer_quality_review": _source_ref(after_review),
            "host_answer_quality_gate": _source_ref(host_answer_quality_gate),
            "paired_delta_receipt": _source_ref(paired_delta_receipt),
            "paired_eval_plan": _source_ref(paired_eval_plan),
        },
        "summary": {
            "before": before_summary,
            "after": after_summary,
            "host_answer_quality_gate": {
                "status": _text(host_answer_quality_gate.get("status")),
                "decision": _text(host_gate_summary.get("decision")),
                "source_decision": _text(host_gate_summary.get("source_decision")),
                "pass_count": _optional_int(host_gate_summary.get("pass_count")),
                "fail_count": _optional_int(host_gate_summary.get("fail_count")),
                "reviewed_count": _optional_int(host_gate_summary.get("reviewed_count")),
            },
            "paired_delta": paired_summary,
            "paired_eval_plan_blockers": plan_blockers,
        },
        "diagnostic_codes": diagnostic_codes,
        "case_diagnostics_sha256": _sha256_json(case_diagnostics),
        "case_diagnostics": case_diagnostics,
        "blockers": blockers,
    }
    return {
        **payload_without_hash,
        "diagnostic_sha256": _sha256_json(payload_without_hash),
    }


def validate_learning_flywheel_failure_diagnostic(
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the diagnostic stays raw-free and fail-closed."""

    errors: list[dict[str, str]] = []
    if diagnostic.get("schema_version") != LEARNING_FLYWHEEL_FAILURE_DIAGNOSTIC_SCHEMA_VERSION:
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if diagnostic.get("status") not in {"diagnosed", "blocked"}:
        errors.append({"code": "invalid_status", "field": "status"})
    for field in (
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
    ):
        if diagnostic.get(field) is not False:
            errors.append({"code": field, "field": field})
    for field in _raw_text_fields(diagnostic):
        errors.append({"code": "raw_text_field_present", "field": field})

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
        paired_item = item.get("paired_delta") if isinstance(item.get("paired_delta"), Mapping) else {}
        delta = _text(paired_item.get("delta"))
        if delta is not None and delta not in {"improved", "regressed", "unchanged"}:
            errors.append(
                {
                    "code": "invalid_paired_delta",
                    "field": f"case_diagnostics[{index}].paired_delta.delta",
                }
            )
        for field in ("before_answer_sha256", "after_answer_sha256"):
            value = paired_item.get(field)
            if value is not None and not _valid_sha256(value):
                errors.append(
                    {
                        "code": f"invalid_{field}",
                        "field": f"case_diagnostics[{index}].paired_delta.{field}",
                    }
                )

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
        if diagnostic.get("blockers") not in ([], ()):
            errors.append({"code": "diagnosed_has_blockers", "field": "blockers"})
        if not diagnostic.get("diagnostic_codes"):
            errors.append({"code": "diagnosed_without_codes", "field": "diagnostic_codes"})
    if diagnostic.get("status") == "blocked" and diagnostic.get("ok") is not False:
        errors.append({"code": "blocked_ok", "field": "ok"})
    return {"ok": not errors, "errors": errors}


def _review_summary(review: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(_mapping(review.get("result")).get("summary"))
    return {
        "status": _text(review.get("status")),
        "ok": review.get("ok") is True,
        "decision": _text(summary.get("decision")),
        "case_count": _optional_int(summary.get("case_count")),
        "reviewed_count": _optional_int(summary.get("reviewed_count")),
        "pass_count": _optional_int(summary.get("pass_count")),
        "fail_count": _optional_int(summary.get("fail_count")),
        "needs_human_review_count": _optional_int(summary.get("needs_human_review_count")),
        "answer_quality_evidence_ready": summary.get("answer_quality_evidence_ready") is True,
    }


def _case_map(review: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = _mapping(review.get("result"))
    refs = {
        _text(row.get("case_id")): _mapping(row.get("structural_evidence"))
        for row in _object_list(result.get("review_case_refs"))
        if _text(row.get("case_id"))
    }
    out: dict[str, dict[str, Any]] = {}
    for row in _object_list(result.get("review_case_refs")):
        case_id = _text(row.get("case_id"))
        if not case_id:
            continue
        structural = refs.get(case_id) or _mapping(row.get("structural_evidence"))
        out[case_id] = {
            "case_id": case_id,
            "prompt_sha256": _normalize_sha256(row.get("prompt_sha256")),
            "answer_sha256": _normalize_sha256(row.get("answer_sha256")),
            "completed": structural.get("completed") is True,
            "fact_tool_called": structural.get("fact_tool_called") is True,
            "selected_tool_count": len(_text_list(structural.get("selected_tools"))),
            "tool_call_count": _optional_int(structural.get("tool_call_count")) or 0,
        }
    for row in _object_list(result.get("reviews")):
        case_id = _text(row.get("case_id"))
        if not case_id:
            continue
        current = out.setdefault(case_id, {"case_id": case_id})
        current["verdict"] = _text(row.get("verdict"))
        current["answer_quality_passed"] = row.get("answer_quality_passed") is True
        current["failure_tags"] = sorted(set(_text_list(row.get("failure_tags"))))
    return out


def _case_diagnostic(
    case_id: str,
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
    paired: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "before": _phase_case(before),
        "after": _phase_case(after),
        "paired_delta": _paired_case(paired),
    }


def _phase_case(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = _mapping(row)
    return {
        "prompt_sha256": _normalize_sha256(row.get("prompt_sha256")),
        "answer_sha256": _normalize_sha256(row.get("answer_sha256")),
        "completed": row.get("completed") is True,
        "verdict": _text(row.get("verdict")),
        "answer_quality_passed": row.get("answer_quality_passed") is True,
        "failure_tags": sorted(set(_text_list(row.get("failure_tags")))),
        "fact_tool_called": row.get("fact_tool_called") is True,
        "selected_tool_count": _optional_int(row.get("selected_tool_count")) or 0,
        "tool_call_count": _optional_int(row.get("tool_call_count")) or 0,
    }


def _paired_case(row: Mapping[str, Any] | None) -> dict[str, Any]:
    row = _mapping(row)
    return {
        "delta": _text(row.get("delta")),
        "before_verdict": _text(row.get("before_verdict")),
        "after_verdict": _text(row.get("after_verdict")),
        "before_answer_sha256": _normalize_sha256(row.get("before_answer_sha256")),
        "after_answer_sha256": _normalize_sha256(row.get("after_answer_sha256")),
    }


def _paired_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": _text(receipt.get("status")),
        "ok": receipt.get("ok") is True,
        "reviewed_pair_count": _optional_int(receipt.get("reviewed_pair_count")),
        "improved_count": _optional_int(receipt.get("improved_count")),
        "regressed_count": _optional_int(receipt.get("regressed_count")),
        "unchanged_count": _optional_int(receipt.get("unchanged_count")),
    }


def _paired_case_map(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        _text(row.get("case_id")): row
        for row in _object_list(receipt.get("paired_case_refs"))
        if _text(row.get("case_id"))
    }


def _diagnostic_codes(
    *,
    before_summary: Mapping[str, Any],
    after_summary: Mapping[str, Any],
    host_answer_quality_gate: Mapping[str, Any],
    paired_delta_receipt: Mapping[str, Any],
    paired_summary: Mapping[str, Any],
    plan_blockers: list[str],
    case_diagnostics: list[Mapping[str, Any]],
) -> list[str]:
    codes: set[str] = set()
    if before_summary.get("decision") != "pass":
        codes.add("before_answer_quality_not_passed")
    if after_summary.get("decision") != "pass":
        codes.add("after_answer_quality_not_passed")
    if host_answer_quality_gate.get("status") != "passed":
        codes.add("host_answer_quality_gate_not_passed")
    if paired_delta_receipt.get("status") != "improved":
        codes.add("paired_delta_not_improved")
    if (paired_summary.get("improved_count") or 0) == 0:
        codes.add("no_improved_pairs")
    reviewed_pair_count = paired_summary.get("reviewed_pair_count") or 0
    improved_count = paired_summary.get("improved_count") or 0
    regressed_count = paired_summary.get("regressed_count") or 0
    unchanged_count = paired_summary.get("unchanged_count") or 0
    if (
        reviewed_pair_count > 0
        and unchanged_count == reviewed_pair_count
        and improved_count == 0
        and regressed_count == 0
    ):
        codes.add("all_reviewed_pairs_unchanged")
    if (
        before_summary.get("decision") == "pass"
        and after_summary.get("decision") == "pass"
        and host_answer_quality_gate.get("status") == "passed"
        and improved_count == 0
    ):
        codes.add("quality_passed_but_no_paired_improvement")
    if "host_answer_quality_gate_not_passed" in plan_blockers:
        codes.add("planner_blocks_on_host_answer_quality_gate")
    if "paired_delta_not_improved" in plan_blockers:
        codes.add("planner_blocks_on_paired_delta")

    after_cases = [_mapping(item.get("after")) for item in case_diagnostics]
    if any("incomplete" in _text_list(item.get("failure_tags")) for item in after_cases):
        codes.add("after_incomplete_answer")
    if any("incorrect_answer" in _text_list(item.get("failure_tags")) for item in after_cases):
        codes.add("after_incorrect_answer")
    selected_cases = [
        item
        for item in after_cases
        if (_optional_int(item.get("selected_tool_count")) or 0) > 0
    ]
    if after_summary.get("decision") != "pass" and selected_cases and all(
        (_optional_int(item.get("tool_call_count")) or 0) == 0 for item in selected_cases
    ):
        codes.add("tools_selected_but_no_tool_calls")
    if any(item.get("completed") is not True for item in after_cases):
        codes.add("after_structural_incomplete")
    return sorted(codes)


def _blockers(
    *,
    before_review: Mapping[str, Any],
    after_review: Mapping[str, Any],
    host_answer_quality_gate: Mapping[str, Any],
    paired_delta_receipt: Mapping[str, Any],
    diagnostic_codes: list[str],
) -> list[str]:
    blockers: list[str] = []
    for label, payload in (
        ("before_review", before_review),
        ("after_review", after_review),
    ):
        if payload.get("status") != "host_answer_quality_reviewed":
            blockers.append(f"{label}_not_reviewed")
    if host_answer_quality_gate.get("status") not in {"passed", "blocked"}:
        blockers.append("host_answer_quality_gate_invalid_status")
    if paired_delta_receipt.get("status") not in {"improved", "blocked"}:
        blockers.append("paired_delta_invalid_status")
    if not diagnostic_codes:
        blockers.append("no_failure_codes")
    return blockers


def _source_ref(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"status": "missing", "content_sha256": None}
    return {
        "schema_version": _text(payload.get("schema_version")),
        "status": _text(payload.get("status")),
        "ok": payload.get("ok") is True,
        "content_sha256": _sha256_json(payload),
    }


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
    return [str(item).strip() for item in value if str(item).strip()]


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
