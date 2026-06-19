# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Paired before/after delta receipt for Learning Flywheel evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


LEARNING_FLYWHEEL_PAIRED_DELTA_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.paired_delta.v1"
)

RAW_TEXT_KEYS = {
    "answer",
    "assistant_response",
    "expected_text",
    "expected_tool",
    "golden_answer",
    "messages",
    "prompt",
    "question",
    "rationale",
    "raw_text",
    "reference_answer",
    "response",
    "transcript",
    "user_text",
}


def build_learning_flywheel_paired_delta_receipt(
    *,
    before_review: Mapping[str, Any],
    after_review: Mapping[str, Any],
    run_id: str,
    evidence_scope: str,
    eval_prompt_set_hash: str,
    min_reviewed_pairs: int = 1,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a raw-free paired delta from host answer-quality review receipts."""

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    normalized_prompt_set_hash = _required_sha256(
        eval_prompt_set_hash,
        "eval_prompt_set_hash",
    )
    min_pairs = max(1, int(min_reviewed_pairs))
    before = _review_refs(before_review, phase="before")
    after = _review_refs(after_review, phase="after")
    paired, blockers = _paired_refs(before=before, after=after)

    reviewed_pair_count = len(paired)
    improved_count = sum(1 for item in paired if item["delta"] == "improved")
    regressed_count = sum(1 for item in paired if item["delta"] == "regressed")
    unchanged_count = sum(1 for item in paired if item["delta"] == "unchanged")
    if before["status"] != "host_answer_quality_reviewed":
        blockers.append(
            {
                "code": "before_review_not_ready",
                "status": before["status"],
            }
        )
    if after["status"] != "host_answer_quality_reviewed":
        blockers.append(
            {
                "code": "after_review_not_ready",
                "status": after["status"],
            }
        )
    if reviewed_pair_count < min_pairs:
        blockers.append(
            {
                "code": "insufficient_reviewed_pairs",
                "reviewed_pair_count": reviewed_pair_count,
                "min_reviewed_pairs": min_pairs,
            }
        )
    if regressed_count:
        blockers.append(
            {
                "code": "regression_detected",
                "regressed_count": regressed_count,
            }
        )
    if not improved_count:
        blockers.append({"code": "no_improved_pairs"})

    status = "improved" if not blockers else "blocked"
    manifest_without_hash = {
        "ok": status == "improved",
        "schema_version": LEARNING_FLYWHEEL_PAIRED_DELTA_SCHEMA_VERSION,
        "status": status,
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "evidence_scope": normalized_scope,
        "eval_prompt_set_hash": normalized_prompt_set_hash,
        "writes_runtime_artifacts": False,
        "runs_device_harness": False,
        "invokes_host_model_judge": False,
        "raw_text_included": False,
        "legacy_expected_hint_fields_included": False,
        "min_reviewed_pairs": min_pairs,
        "reviewed_pair_count": reviewed_pair_count,
        "improved_count": improved_count,
        "regressed_count": regressed_count,
        "unchanged_count": unchanged_count,
        "missing_before_case_ids": blockers_by_code(blockers, "missing_before_case"),
        "missing_after_case_ids": blockers_by_code(blockers, "missing_after_case"),
        "blockers": blockers,
        "paired_case_refs": paired,
        "before_review_summary": before["summary"],
        "after_review_summary": after["summary"],
        "paired_material_sha256": _sha256_json(
            {
                "eval_prompt_set_hash": normalized_prompt_set_hash,
                "paired_case_refs": paired,
                "before_review_refs_sha256": before["review_refs_sha256"],
                "after_review_refs_sha256": after["review_refs_sha256"],
            }
        ),
    }
    manifest = {
        **manifest_without_hash,
        "receipt_sha256": _sha256_json(manifest_without_hash),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def validate_learning_flywheel_paired_delta_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a paired delta receipt is raw-free and fail-closed."""

    errors: list[dict[str, str]] = []
    if receipt.get("schema_version") != LEARNING_FLYWHEEL_PAIRED_DELTA_SCHEMA_VERSION:
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if receipt.get("status") not in {"improved", "blocked"}:
        errors.append({"code": "invalid_status", "field": "status"})
    if receipt.get("raw_text_included") is not False:
        errors.append({"code": "raw_text_included", "field": "raw_text_included"})
    if receipt.get("legacy_expected_hint_fields_included") is not False:
        errors.append(
            {
                "code": "legacy_expected_hint_fields_included",
                "field": "legacy_expected_hint_fields_included",
            }
        )
    for field in _raw_text_fields(receipt):
        errors.append({"code": "raw_text_field_present", "field": field})
    for field in ("eval_prompt_set_hash", "paired_material_sha256", "receipt_sha256"):
        if not _valid_sha256(receipt.get(field)):
            errors.append({"code": "invalid_sha256", "field": field})
    if receipt.get("status") == "improved":
        if receipt.get("blockers") not in ([], ()):
            errors.append({"code": "improved_receipt_has_blockers", "field": "blockers"})
        if not _positive_int(receipt.get("improved_count")):
            errors.append({"code": "improved_receipt_without_improvement", "field": "improved_count"})
        if int(receipt.get("regressed_count") or 0) != 0:
            errors.append({"code": "improved_receipt_has_regression", "field": "regressed_count"})
    return {"ok": not errors, "errors": errors}


def blockers_by_code(blockers: list[dict[str, Any]], code: str) -> list[str]:
    return sorted(
        _text(blocker.get("case_id"))
        for blocker in blockers
        if blocker.get("code") == code and _text(blocker.get("case_id"))
    )


def _review_refs(review: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    if not isinstance(review, Mapping):
        raise ValueError(f"{phase}_review must be an object")
    result = review.get("result") if isinstance(review.get("result"), Mapping) else {}
    reviews = result.get("reviews")
    case_refs = result.get("review_case_refs")
    if not isinstance(reviews, list):
        reviews = []
    if not isinstance(case_refs, list):
        case_refs = []
    refs_by_case: dict[str, dict[str, Any]] = {}
    case_hashes = {
        _required_text(item.get("case_id"), "case_id"): {
            "prompt_sha256": _required_sha256(item.get("prompt_sha256"), "prompt_sha256"),
            "answer_sha256": _required_sha256(item.get("answer_sha256"), "answer_sha256"),
        }
        for item in case_refs
        if isinstance(item, Mapping) and _text(item.get("case_id"))
    }
    for item in reviews:
        if not isinstance(item, Mapping):
            continue
        case_id = _required_text(item.get("case_id"), "case_id")
        verdict = _required_text(item.get("verdict"), "verdict")
        if verdict not in {"pass", "fail", "needs_human_review"}:
            raise ValueError(f"unsupported review verdict: {verdict}")
        hashes = case_hashes.get(case_id, {})
        refs_by_case[case_id] = {
            "case_id": case_id,
            "verdict": verdict,
            "answer_quality_passed": verdict == "pass",
            "prompt_sha256": hashes.get("prompt_sha256"),
            "answer_sha256": hashes.get("answer_sha256"),
        }
    refs = sorted(refs_by_case.values(), key=lambda item: item["case_id"])
    return {
        "status": _text(review.get("status")),
        "summary": _safe_review_summary(result.get("summary")),
        "refs_by_case": refs_by_case,
        "review_refs": refs,
        "review_refs_sha256": _sha256_json(refs),
    }


def _paired_refs(
    *,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before_cases = before["refs_by_case"]
    after_cases = after["refs_by_case"]
    case_ids = sorted(set(before_cases) | set(after_cases))
    paired: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for case_id in case_ids:
        before_ref = before_cases.get(case_id)
        after_ref = after_cases.get(case_id)
        if before_ref is None:
            blockers.append({"code": "missing_before_case", "case_id": case_id})
            continue
        if after_ref is None:
            blockers.append({"code": "missing_after_case", "case_id": case_id})
            continue
        if before_ref.get("prompt_sha256") != after_ref.get("prompt_sha256"):
            blockers.append({"code": "prompt_hash_mismatch", "case_id": case_id})
            continue
        before_passed = before_ref["answer_quality_passed"] is True
        after_passed = after_ref["answer_quality_passed"] is True
        if after_passed and not before_passed:
            delta = "improved"
        elif before_passed and not after_passed:
            delta = "regressed"
        else:
            delta = "unchanged"
        paired.append(
            {
                "case_id": case_id,
                "prompt_sha256": before_ref.get("prompt_sha256"),
                "before_answer_sha256": before_ref.get("answer_sha256"),
                "after_answer_sha256": after_ref.get("answer_sha256"),
                "before_verdict": before_ref["verdict"],
                "after_verdict": after_ref["verdict"],
                "delta": delta,
            }
        )
    return paired, blockers


def _safe_review_summary(value: Any) -> dict[str, Any]:
    summary = value if isinstance(value, Mapping) else {}
    return {
        "decision": _text(summary.get("decision")),
        "case_count": _int(summary.get("case_count")),
        "reviewed_count": _int(summary.get("reviewed_count")),
        "pass_count": _int(summary.get("pass_count")),
        "fail_count": _int(summary.get("fail_count")),
        "needs_human_review_count": _int(summary.get("needs_human_review_count")),
        "answer_quality_evidence_ready": summary.get("answer_quality_evidence_ready") is True,
    }


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


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _required_sha256(value: Any, field: str) -> str:
    if not _valid_sha256(value):
        raise ValueError(f"{field} must be a sha256 hash")
    raw = str(value)
    return raw if raw.startswith("sha256:") else "sha256:" + raw


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value[7:] if value.startswith("sha256:") else value
    return len(raw) == 64 and all(char in "0123456789abcdef" for char in raw.lower())


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
