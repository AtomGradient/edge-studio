# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Raw-free host answer-quality review gate projection.

This module converts the full host-review receipt into a portable gate receipt
for Learning Flywheel evidence. It never invokes the Host Model and never writes
files; callers decide where to persist the returned projection.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping


LEARNING_FLYWHEEL_HOST_REVIEW_GATE_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.host_answer_quality_gate.v1"
)
HOST_ANSWER_QUALITY_REVIEW_SCHEMA_VERSION = "edgestudio.host_answer_quality_review.v1"

READY_SOURCE_STATUS = "host_answer_quality_reviewed"
PASSED_DECISION = "pass"
ALLOWED_VERDICTS = frozenset({"pass", "fail", "needs_human_review"})
ALLOWED_FAILURE_TAGS = frozenset(
    {
        "incorrect_answer",
        "missing_answer",
        "tool_not_called",
        "wrong_tool",
        "tool_misuse",
        "hallucination",
        "refusal",
        "format_error",
        "incomplete",
        "unsafe",
        "insufficient_context",
        "runtime_error",
        "other",
    }
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
    "selected_tools",
    "structural_evidence",
    "text",
    "tool_arguments",
    "tool_call_names",
    "tool_calls",
    "tool_name",
    "transcript",
    "user_text",
}
SOURCE_DIRECT_RAW_KEYS = RAW_TEXT_KEYS - {"rationale", "structural_evidence"}


def project_learning_flywheel_host_review_gate(
    receipt: Mapping[str, Any],
    *,
    run_id: str,
    evidence_scope: str,
) -> dict[str, Any]:
    """Project a host-review receipt into a raw-free fail-closed gate."""

    if not isinstance(receipt, Mapping):
        raise ValueError("receipt must be an object")
    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    result = receipt.get("result") if isinstance(receipt.get("result"), Mapping) else {}
    source_status = _text(receipt.get("status"))
    source_ok = receipt.get("ok") is True
    source_schema = _text(receipt.get("schema_version"))
    source_summary = _safe_summary(result.get("summary"))
    blockers: list[dict[str, Any]] = []

    if source_schema != HOST_ANSWER_QUALITY_REVIEW_SCHEMA_VERSION:
        blockers.append({"code": "source_schema_version_mismatch"})
    if not source_ok or source_status != READY_SOURCE_STATUS:
        blockers.append(
            {
                "code": "source_review_not_ready",
                "source_status": source_status or "<missing>",
            }
        )
    if source_summary["decision"] != PASSED_DECISION:
        blockers.append(
            {
                "code": "source_decision_not_pass",
                "decision": source_summary["decision"] or "<missing>",
            }
        )
    if source_summary["answer_quality_evidence_ready"] is not True:
        blockers.append({"code": "source_answer_quality_evidence_not_ready"})

    case_refs, case_ref_blockers = _case_refs(result.get("review_case_refs"))
    review_refs, review_blockers = _review_refs(
        result.get("reviews"),
        case_hashes={item["case_id"]: item for item in case_refs},
    )
    blockers.extend(case_ref_blockers)
    blockers.extend(review_blockers)

    counts = _review_counts(review_refs)
    status = "passed" if not blockers else "blocked"
    gate_without_hash = {
        "ok": status == "passed",
        "schema_version": LEARNING_FLYWHEEL_HOST_REVIEW_GATE_SCHEMA_VERSION,
        "status": status,
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "evidence_scope": normalized_scope,
        "writes_runtime_artifacts": False,
        "runs_device_harness": False,
        "invokes_host_model_judge": False,
        "raw_text_included": False,
        "legacy_expected_hint_fields_included": False,
        "source": {
            "schema_version": source_schema,
            "status": source_status,
            "ok": source_ok,
            "receipt_content_sha256": _sha256_json(receipt),
            "summary_decision": source_summary["decision"],
        },
        "summary": {
            "decision": "pass" if status == "passed" else "blocked",
            "source_decision": source_summary["decision"],
            "case_count": len(case_refs),
            "reviewed_count": len(review_refs),
            "pass_count": counts["pass"],
            "fail_count": counts["fail"],
            "needs_human_review_count": counts["needs_human_review"],
            "answer_quality_evidence_ready": status == "passed",
        },
        "blockers": blockers,
        "case_refs_sha256": _sha256_json(review_refs),
        "case_refs": review_refs,
    }
    return {
        **gate_without_hash,
        "gate_sha256": _sha256_json(gate_without_hash),
    }


def validate_learning_flywheel_host_review_gate(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a host-review gate projection is raw-free and fail-closed."""

    errors: list[dict[str, str]] = []
    if gate.get("schema_version") != LEARNING_FLYWHEEL_HOST_REVIEW_GATE_SCHEMA_VERSION:
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if gate.get("status") not in {"passed", "blocked"}:
        errors.append({"code": "invalid_status", "field": "status"})
    if gate.get("raw_text_included") is not False:
        errors.append({"code": "raw_text_included", "field": "raw_text_included"})
    if gate.get("legacy_expected_hint_fields_included") is not False:
        errors.append(
            {
                "code": "legacy_expected_hint_fields_included",
                "field": "legacy_expected_hint_fields_included",
            }
        )
    if gate.get("writes_runtime_artifacts") is not False:
        errors.append(
            {"code": "writes_runtime_artifacts", "field": "writes_runtime_artifacts"}
        )
    if gate.get("runs_device_harness") is not False:
        errors.append({"code": "runs_device_harness", "field": "runs_device_harness"})
    if gate.get("invokes_host_model_judge") is not False:
        errors.append(
            {"code": "invokes_host_model_judge", "field": "invokes_host_model_judge"}
        )
    for field in _raw_text_fields(gate):
        errors.append({"code": "raw_text_field_present", "field": field})

    source = gate.get("source") if isinstance(gate.get("source"), Mapping) else {}
    for field, value in (
        ("source.receipt_content_sha256", source.get("receipt_content_sha256")),
        ("case_refs_sha256", gate.get("case_refs_sha256")),
        ("gate_sha256", gate.get("gate_sha256")),
    ):
        if not _valid_sha256(value):
            errors.append({"code": "invalid_sha256", "field": field})

    refs = gate.get("case_refs")
    if not isinstance(refs, list):
        errors.append({"code": "missing_case_refs", "field": "case_refs"})
        refs = []
    elif _sha256_json(refs) != gate.get("case_refs_sha256"):
        errors.append({"code": "case_refs_hash_mismatch", "field": "case_refs_sha256"})

    for index, item in enumerate(refs):
        if not isinstance(item, Mapping):
            errors.append({"code": "invalid_case_ref", "field": f"case_refs[{index}]"})
            continue
        for field in ("case_id", "verdict"):
            if not _text(item.get(field)):
                errors.append(
                    {"code": f"missing_{field}", "field": f"case_refs[{index}].{field}"}
                )
        for field in ("prompt_sha256", "answer_sha256"):
            if not _valid_sha256(item.get(field)):
                errors.append(
                    {
                        "code": f"invalid_{field}",
                        "field": f"case_refs[{index}].{field}",
                    }
                )
        verdict = _text(item.get("verdict")).casefold()
        if verdict not in ALLOWED_VERDICTS:
            errors.append(
                {"code": "invalid_verdict", "field": f"case_refs[{index}].verdict"}
            )
        for tag in item.get("failure_tags") or []:
            if _text(tag) not in ALLOWED_FAILURE_TAGS:
                errors.append(
                    {
                        "code": "invalid_failure_tag",
                        "field": f"case_refs[{index}].failure_tags",
                    }
                )
        confidence = item.get("confidence")
        if confidence is not None and not _valid_confidence(confidence):
            errors.append(
                {"code": "invalid_confidence", "field": f"case_refs[{index}].confidence"}
            )

    if gate.get("status") == "passed":
        if gate.get("ok") is not True:
            errors.append({"code": "passed_gate_not_ok", "field": "ok"})
        if gate.get("blockers") not in ([], ()):
            errors.append({"code": "passed_gate_has_blockers", "field": "blockers"})
        if not refs:
            errors.append({"code": "passed_gate_without_case_refs", "field": "case_refs"})
        summary = gate.get("summary") if isinstance(gate.get("summary"), Mapping) else {}
        if summary.get("source_decision") != PASSED_DECISION:
            errors.append(
                {"code": "passed_gate_source_decision_not_pass", "field": "summary"}
            )
    if gate.get("status") == "blocked" and gate.get("ok") is not False:
        errors.append({"code": "blocked_gate_ok", "field": "ok"})

    expected_gate_hash = _sha256_json(
        {key: value for key, value in gate.items() if key != "gate_sha256"}
    )
    if gate.get("gate_sha256") != expected_gate_hash:
        errors.append({"code": "gate_hash_mismatch", "field": "gate_sha256"})
    return {"ok": not errors, "errors": errors}


def _case_refs(value: Any) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    if not isinstance(value, list) or not value:
        return [], [{"code": "missing_review_case_refs"}]

    refs: dict[str, dict[str, str]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            blockers.append({"code": "invalid_review_case_ref", "index": index})
            continue
        if _source_direct_raw_fields(item):
            blockers.append(
                {
                    "code": "raw_source_field_present",
                    "source_section": "review_case_refs",
                    "index": index,
                }
            )
        case_id = _text(item.get("case_id"))
        if not case_id:
            blockers.append({"code": "missing_case_id", "field": f"review_case_refs[{index}]"})
            continue
        if case_id in refs:
            blockers.append({"code": "duplicate_review_case_ref", "case_id": case_id})
            continue
        prompt_hash = _normalize_sha256(item.get("prompt_sha256"))
        answer_hash = _normalize_sha256(item.get("answer_sha256"))
        if prompt_hash is None:
            blockers.append({"code": "invalid_prompt_sha256", "case_id": case_id})
        if answer_hash is None:
            blockers.append({"code": "invalid_answer_sha256", "case_id": case_id})
        if prompt_hash is not None and answer_hash is not None:
            refs[case_id] = {
                "case_id": case_id,
                "prompt_sha256": prompt_hash,
                "answer_sha256": answer_hash,
            }
    return sorted(refs.values(), key=lambda item: item["case_id"]), blockers


def _review_refs(
    value: Any,
    *,
    case_hashes: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    if not isinstance(value, list) or not value:
        return [], [{"code": "missing_reviews"}]

    refs_by_case: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            blockers.append({"code": "invalid_review", "index": index})
            continue
        if _source_direct_raw_fields(item):
            blockers.append(
                {
                    "code": "raw_source_field_present",
                    "source_section": "reviews",
                    "index": index,
                }
            )
        case_id = _text(item.get("case_id"))
        if not case_id:
            blockers.append({"code": "missing_case_id", "field": f"reviews[{index}]"})
            continue
        if case_id in refs_by_case:
            blockers.append({"code": "duplicate_review", "case_id": case_id})
            continue
        verdict = _text(item.get("verdict")).casefold()
        if verdict not in ALLOWED_VERDICTS:
            blockers.append({"code": "invalid_verdict", "case_id": case_id})
            continue
        hashes = case_hashes.get(case_id)
        if hashes is None:
            blockers.append({"code": "missing_review_case_ref", "case_id": case_id})
            continue
        for hash_field in ("prompt_sha256", "answer_sha256"):
            review_hash = _normalize_sha256(item.get(hash_field))
            if review_hash is not None and review_hash != hashes[hash_field]:
                blockers.append(
                    {
                        "code": "prompt_answer_hash_mismatch",
                        "case_id": case_id,
                        "field": hash_field,
                    }
                )
        tags = _failure_tags(item.get("failure_tags"), blockers=blockers, case_id=case_id)
        ref: dict[str, Any] = {
            "case_id": case_id,
            "prompt_sha256": hashes["prompt_sha256"],
            "answer_sha256": hashes["answer_sha256"],
            "verdict": verdict,
            "answer_quality_passed": verdict == PASSED_DECISION,
            "failure_tags": tags,
            "failure_tag_count": len(tags),
        }
        confidence = item.get("confidence")
        if confidence is not None:
            if _valid_confidence(confidence):
                ref["confidence"] = float(confidence)
            else:
                blockers.append({"code": "invalid_confidence", "case_id": case_id})
        refs_by_case[case_id] = ref
    return sorted(refs_by_case.values(), key=lambda item: item["case_id"]), blockers


def _failure_tags(
    value: Any,
    *,
    blockers: list[dict[str, Any]],
    case_id: str,
) -> list[str]:
    if value in (None, [], ()):
        return []
    if not isinstance(value, list):
        blockers.append({"code": "invalid_failure_tags", "case_id": case_id})
        return []
    tags: list[str] = []
    for tag in value:
        normalized = _text(tag).casefold()
        if normalized not in ALLOWED_FAILURE_TAGS:
            blockers.append(
                {
                    "code": "invalid_failure_tag",
                    "case_id": case_id,
                    "tag_sha256": _sha256_text(_text(tag)),
                }
            )
            continue
        tags.append(normalized)
    return sorted(set(tags))


def _review_counts(refs: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {verdict: 0 for verdict in ALLOWED_VERDICTS}
    for item in refs:
        verdict = _text(item.get("verdict")).casefold()
        if verdict in counts:
            counts[verdict] += 1
    return counts


def _safe_summary(value: Any) -> dict[str, Any]:
    summary = value if isinstance(value, Mapping) else {}
    return {
        "decision": _text(summary.get("decision")).casefold(),
        "answer_quality_evidence_ready": summary.get("answer_quality_evidence_ready") is True,
    }


def _source_direct_raw_fields(payload: Mapping[str, Any]) -> list[str]:
    return sorted(str(key) for key in payload if str(key) in SOURCE_DIRECT_RAW_KEYS)


def _raw_text_fields(payload: Any, prefix: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in RAW_TEXT_KEYS:
                fields.append(field)
            fields.extend(_raw_text_fields(value, field))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            field = f"{prefix}[{index}]" if prefix else f"[{index}]"
            fields.extend(_raw_text_fields(item, field))
    return fields


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_sha256(value: Any) -> str | None:
    if not _valid_sha256(value):
        return None
    raw = str(value)
    return raw if raw.startswith("sha256:") else "sha256:" + raw


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value[7:] if value.startswith("sha256:") else value
    return len(raw) == 64 and all(char in "0123456789abcdef" for char in raw.lower())


def _valid_confidence(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
