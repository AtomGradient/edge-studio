# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Validation for Learning Flywheel evidence receipts.

The receipt is evidence metadata only. It must not carry user text, run a
host-model judge, mutate runtime artifacts, or convert data plumbing into a
behavior-improvement claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse


LEARNING_FLYWHEEL_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.receipt.v1"
)

ALLOWED_MODES = {"receipt_only", "real_device_paired_eval"}
ALLOWED_STATUSES = {"passed", "failed", "blocked", "not_run"}
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "run_id",
    "mode",
    "status",
    "surface",
    "device",
    "model",
    "artifact",
    "corrections",
    "evaluation",
    "evidence",
    "privacy",
    "claims",
)
HASH_RE = re.compile(r"^(sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
RAW_TEXT_KEYS = {
    "answer",
    "after_answer",
    "assistant_response",
    "before_answer",
    "chosen",
    "correction",
    "correction_text",
    "expected_text",
    "expected_tool",
    "generated_text",
    "messages",
    "prompt",
    "question",
    "raw_text",
    "rejected",
    "response",
    "sample_text",
    "transcript",
    "user_text",
}
NO_CHEAT_FLAGS = (
    "no_keyword_rule_fallback",
    "no_nl2query_rule_fallback",
    "no_prompt_stuffing",
    "no_visible_prompt_patch",
    "facts_answered_by_tools",
    "normal_app_runtime_path",
)
ALWAYS_FALSE_CLAIMS = (
    "model_quality_improved",
    "router_quality_improved",
    "production_learning_shipped",
)


@dataclass(frozen=True)
class LearningFlywheelReceiptIssue:
    level: str
    code: str
    message: str
    field: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True)
class LearningFlywheelReceiptValidation:
    ok: bool
    errors: list[LearningFlywheelReceiptIssue]
    warnings: list[LearningFlywheelReceiptIssue]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }


def learning_flywheel_receipt_schema() -> dict[str, Any]:
    """Return the minimal schema shape expected from receipt producers."""

    return {
        "schema_version": LEARNING_FLYWHEEL_RECEIPT_SCHEMA_VERSION,
        "modes": sorted(ALLOWED_MODES),
        "statuses": sorted(ALLOWED_STATUSES),
        "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
        "no_cheat_flags": list(NO_CHEAT_FLAGS),
        "claim_gate": {
            "receipt_only_must_not_claim_behavior_improved": True,
            "behavior_improved_requires_real_device_paired_eval": True,
            "behavior_improved_requires_eval_prompt_set_hash": True,
            "behavior_improved_requires_host_answer_quality_pass": True,
            "behavior_improved_requires_heldout_leakage_pass": True,
            "model_quality_improved_must_remain_false": True,
            "router_quality_improved_must_remain_false": True,
            "production_learning_shipped_must_remain_false": True,
        },
    }


def validate_learning_flywheel_receipt(
    receipt: Mapping[str, Any],
) -> LearningFlywheelReceiptValidation:
    """Validate a Learning Flywheel receipt without executing learning."""

    errors: list[LearningFlywheelReceiptIssue] = []
    warnings: list[LearningFlywheelReceiptIssue] = []

    if not isinstance(receipt, Mapping):
        return LearningFlywheelReceiptValidation(
            ok=False,
            errors=[
                LearningFlywheelReceiptIssue(
                    "error",
                    "invalid_receipt",
                    "receipt root must be an object",
                )
            ],
            warnings=[],
        )

    schema = receipt.get("schema_version")
    if schema != LEARNING_FLYWHEEL_RECEIPT_SCHEMA_VERSION:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "schema_version_mismatch",
                f"schema_version must be {LEARNING_FLYWHEEL_RECEIPT_SCHEMA_VERSION}",
                "schema_version",
            )
        )

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in receipt:
            errors.append(
                LearningFlywheelReceiptIssue(
                    "error",
                    "missing_required_field",
                    f"missing required field: {field}",
                    field,
                )
            )

    mode = _text(receipt.get("mode"))
    status = _text(receipt.get("status"))
    if mode not in ALLOWED_MODES:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "invalid_mode",
                f"mode must be one of {sorted(ALLOWED_MODES)}",
                "mode",
            )
        )
    if status not in ALLOWED_STATUSES:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "invalid_status",
                f"status must be one of {sorted(ALLOWED_STATUSES)}",
                "status",
            )
        )

    surface = _mapping(receipt.get("surface"))
    device = _mapping(receipt.get("device"))
    model = _mapping(receipt.get("model"))
    artifact = _mapping(receipt.get("artifact"))
    corrections = _mapping(receipt.get("corrections"))
    evaluation = _mapping(receipt.get("evaluation"))
    evidence = _mapping(receipt.get("evidence"))
    privacy = _mapping(receipt.get("privacy"))
    claims = _mapping(receipt.get("claims"))

    _require_text(surface, "kind", "surface.kind", errors)
    _require_text(surface, "app_id", "surface.app_id", errors)
    _require_text(model, "model_id", "model.model_id", errors)
    _require_hash(model, "model_fingerprint", "model.model_fingerprint", errors)
    _validate_artifact(artifact, errors)
    _validate_corrections(corrections, mode=mode, status=status, errors=errors)
    _validate_privacy(receipt, privacy, errors)
    _validate_claims(claims, mode=mode, status=status, errors=errors)

    if mode == "receipt_only" and status == "passed":
        _validate_receipt_only_evidence(evidence, errors)
    if mode == "real_device_paired_eval" and status == "passed":
        _validate_real_device_paired_eval(
            device=device,
            artifact=artifact,
            corrections=corrections,
            evaluation=evaluation,
            evidence=evidence,
            privacy=privacy,
            claims=claims,
            errors=errors,
            warnings=warnings,
        )
    elif claims.get("behavior_improved") is True:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "behavior_improved_without_passed_paired_eval",
                (
                    "behavior_improved requires mode=real_device_paired_eval "
                    "and status=passed"
                ),
                "claims.behavior_improved",
            )
        )

    return LearningFlywheelReceiptValidation(
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


def _validate_artifact(
    artifact: Mapping[str, Any],
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    generated = artifact.get("generated") is True
    if generated:
        _require_hash(artifact, "artifact_sha256", "artifact.artifact_sha256", errors)
        _require_hash(artifact, "metadata_sha256", "artifact.metadata_sha256", errors)
        _require_hash(artifact, "lineage_sha256", "artifact.lineage_sha256", errors)
        _require_non_negative_int(
            artifact,
            "prefix_token_count",
            "artifact.prefix_token_count",
            errors,
        )
    else:
        _optional_hash(artifact, "artifact_sha256", "artifact.artifact_sha256", errors)
        _optional_hash(artifact, "metadata_sha256", "artifact.metadata_sha256", errors)
        _optional_hash(artifact, "lineage_sha256", "artifact.lineage_sha256", errors)
        _optional_non_negative_int(
            artifact,
            "prefix_token_count",
            "artifact.prefix_token_count",
            errors,
        )


def _validate_corrections(
    corrections: Mapping[str, Any],
    *,
    mode: str,
    status: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    if mode == "receipt_only" and status == "passed":
        _require_non_negative_int(
            corrections,
            "correction_fingerprint_count",
            "corrections.correction_fingerprint_count",
            errors,
        )
    else:
        _optional_non_negative_int(
            corrections,
            "correction_fingerprint_count",
            "corrections.correction_fingerprint_count",
            errors,
        )
    _optional_hash(
        corrections,
        "correction_fingerprints_sha256",
        "corrections.correction_fingerprints_sha256",
        errors,
    )
    if corrections.get("raw_correction_text_included") is not False:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "raw_correction_text_included",
                "corrections.raw_correction_text_included must be false",
                "corrections.raw_correction_text_included",
            )
        )


def _validate_receipt_only_evidence(
    evidence: Mapping[str, Any],
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    if evidence.get("data_plumbing_audited") is not True:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "missing_data_plumbing_audit",
                "receipt_only passed receipts require evidence.data_plumbing_audited=true",
                "evidence.data_plumbing_audited",
            )
        )


def _validate_real_device_paired_eval(
    *,
    device: Mapping[str, Any],
    artifact: Mapping[str, Any],
    corrections: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    privacy: Mapping[str, Any],
    claims: Mapping[str, Any],
    errors: list[LearningFlywheelReceiptIssue],
    warnings: list[LearningFlywheelReceiptIssue],
) -> None:
    _require_text(device, "name", "device.name", errors)
    _require_text(device, "os_build", "device.os_build", errors)
    _require_text(device, "device_fingerprint", "device.device_fingerprint", errors)
    if artifact.get("generated") is not True:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "missing_generated_artifact",
                "paired eval requires a generated Neural Imprint artifact",
                "artifact.generated",
            )
        )
    _require_non_negative_int(
        corrections,
        "correction_fingerprint_count",
        "corrections.correction_fingerprint_count",
        errors,
    )
    correction_count = corrections.get("correction_fingerprint_count")
    if isinstance(correction_count, int) and not isinstance(correction_count, bool):
        has_corrections = correction_count > 0
    else:
        has_corrections = False
    if not has_corrections:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "missing_correction_fingerprints",
                "paired eval requires at least one correction fingerprint",
                "corrections.correction_fingerprint_count",
            )
        )
    _require_hash(
        corrections,
        "correction_fingerprints_sha256",
        "corrections.correction_fingerprints_sha256",
        errors,
    )

    _require_hash(
        evaluation,
        "eval_prompt_set_hash",
        "evaluation.eval_prompt_set_hash",
        errors,
    )
    _require_hash(
        evaluation,
        "before_observations_sha256",
        "evaluation.before_observations_sha256",
        errors,
    )
    _require_hash(
        evaluation,
        "after_observations_sha256",
        "evaluation.after_observations_sha256",
        errors,
    )
    _validate_gate(
        evaluation,
        key="heldout_leakage_report",
        expected_status="passed",
        field="evaluation.heldout_leakage_report",
        code="missing_or_failed_heldout_leakage_report",
        errors=errors,
    )
    _validate_gate(
        evaluation,
        key="host_answer_quality_review",
        expected_status="passed",
        field="evaluation.host_answer_quality_review",
        code="missing_or_failed_host_answer_quality_review",
        errors=errors,
    )
    _validate_gate(
        evaluation,
        key="paired_delta",
        expected_status="improved",
        field="evaluation.paired_delta",
        code="missing_or_non_improved_paired_delta",
        errors=errors,
    )
    for key in NO_CHEAT_FLAGS:
        if privacy.get(key) is not True:
            errors.append(
                LearningFlywheelReceiptIssue(
                    "error",
                    "missing_no_cheat_flag",
                    f"privacy.{key} must be true for paired behavior evidence",
                    f"privacy.{key}",
                )
            )
    if evidence.get("data_plumbing_audited") is not True:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "missing_data_plumbing_audit",
                "paired eval requires evidence.data_plumbing_audited=true",
                "evidence.data_plumbing_audited",
            )
        )
    if claims.get("behavior_improved") is not True:
        warnings.append(
            LearningFlywheelReceiptIssue(
                "warning",
                "passed_paired_eval_without_behavior_claim",
                "paired eval passed but does not claim behavior_improved",
                "claims.behavior_improved",
            )
        )


def _validate_gate(
    evaluation: Mapping[str, Any],
    *,
    key: str,
    expected_status: str,
    field: str,
    code: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    gate = _mapping(evaluation.get(key))
    if gate.get("status") != expected_status:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                code,
                f"{field}.status must be {expected_status}",
                f"{field}.status",
            )
        )
    _optional_hash(gate, "receipt_sha256", f"{field}.receipt_sha256", errors)


def _validate_privacy(
    receipt: Mapping[str, Any],
    privacy: Mapping[str, Any],
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    if privacy.get("raw_text_included") is not False:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "raw_text_included",
                "privacy.raw_text_included must be false",
                "privacy.raw_text_included",
            )
        )
    if privacy.get("external_network_used") is not False:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "external_network_used",
                "privacy.external_network_used must be false",
                "privacy.external_network_used",
            )
        )
    for field in _raw_text_fields(receipt):
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "raw_text_field_present",
                "receipt contains a raw-text field",
                field,
            )
        )
    for field, url in _external_urls(receipt):
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "external_url_present",
                f"receipt contains a non-localhost URL: {url}",
                field,
            )
        )


def _validate_claims(
    claims: Mapping[str, Any],
    *,
    mode: str,
    status: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    for key in ALWAYS_FALSE_CLAIMS:
        if claims.get(key) is not False:
            errors.append(
                LearningFlywheelReceiptIssue(
                    "error",
                    f"{key}_claim",
                    f"{key} must remain false in Learning Flywheel receipts",
                    f"claims.{key}",
                )
            )
    if mode == "receipt_only" and claims.get("behavior_improved") is True:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "receipt_only_behavior_improved_claim",
                "receipt_only mode cannot claim behavior_improved",
                "claims.behavior_improved",
            )
        )
    if status != "passed" and claims.get("behavior_improved") is True:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "behavior_improved_without_passed_status",
                "behavior_improved requires status=passed",
                "claims.behavior_improved",
            )
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _require_text(
    payload: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    if not _text(payload.get(key)):
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "missing_text_field",
                f"{field} is required",
                field,
            )
        )


def _require_hash(
    payload: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    value = payload.get(key)
    if not isinstance(value, str) or not HASH_RE.match(value):
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "invalid_sha256",
                f"{field} must be a sha256 hash",
                field,
            )
        )


def _optional_hash(
    payload: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    value = payload.get(key)
    if value in (None, ""):
        return
    if not isinstance(value, str) or not HASH_RE.match(value):
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "invalid_sha256",
                f"{field} must be a sha256 hash when present",
                field,
            )
        )


def _require_non_negative_int(
    payload: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "invalid_non_negative_int",
                f"{field} must be a non-negative integer",
                field,
            )
        )


def _optional_non_negative_int(
    payload: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[LearningFlywheelReceiptIssue],
) -> None:
    value = payload.get(key)
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(
            LearningFlywheelReceiptIssue(
                "error",
                "invalid_non_negative_int",
                f"{field} must be a non-negative integer when present",
                field,
            )
        )


def _raw_text_fields(value: Any, *, path: str = "") -> list[str]:
    fields: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if key in RAW_TEXT_KEYS and child not in (None, "", [], {}):
                fields.append(child_path)
            fields.extend(_raw_text_fields(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            fields.extend(_raw_text_fields(child, path=child_path))
    return fields


def _external_urls(value: Any, *, path: str = "") -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            urls.extend(_external_urls(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            urls.extend(_external_urls(child, path=child_path))
    elif isinstance(value, str):
        for match in URL_RE.finditer(value):
            url = match.group(0)
            parsed = urlparse(url)
            if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
                urls.append((path, url))
    return urls
