# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hash-only heldout prompt-set contract for Learning Flywheel evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEARNING_FLYWHEEL_HELDOUT_PROMPT_SET_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.heldout_prompt_set.v1"
)

RAW_TEXT_KEYS = {
    "answer",
    "expected_text",
    "expected_tool",
    "golden_answer",
    "messages",
    "prompt",
    "question",
    "raw_text",
    "text",
    "transcript",
    "user_text",
}
LEGACY_HINT_KEYS = {"expected_tool", "expected_text", "golden_answer", "expected_answer"}
HELDOUT_PRIMARY_CLASS_VALUES = frozenset(
    {
        "preference_behavior",
        "fact_tool_behavior",
        "refusal_uncertainty_behavior",
    }
)


def build_learning_flywheel_heldout_prompt_set_manifest(
    *,
    cases: Iterable[Mapping[str, Any]],
    run_id: str,
    evidence_scope: str,
    output_path: Path | None = None,
    require_primary_class: bool = False,
) -> dict[str, Any]:
    """Build a portable manifest that contains prompt hashes, not prompts."""

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    primary_class_required = bool(require_primary_class)
    case_refs = _case_refs(
        cases,
        require_primary_class=primary_class_required,
    )
    prompt_set_hash = _sha256_json(
        {
            "schema_version": LEARNING_FLYWHEEL_HELDOUT_PROMPT_SET_SCHEMA_VERSION,
            "evidence_scope": normalized_scope,
            "primary_class_required": primary_class_required,
            "case_refs": case_refs,
        }
    )
    manifest_without_hash = {
        "ok": True,
        "schema_version": LEARNING_FLYWHEEL_HELDOUT_PROMPT_SET_SCHEMA_VERSION,
        "status": "written",
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "evidence_scope": normalized_scope,
        "training_side_only": False,
        "writes_runtime_artifacts": False,
        "runs_device_harness": False,
        "invokes_host_model_judge": False,
        "raw_text_included": False,
        "legacy_expected_hint_fields_included": False,
        "primary_class_required": primary_class_required,
        "allowed_primary_classes": sorted(HELDOUT_PRIMARY_CLASS_VALUES),
        "prompt_count": len(case_refs),
        "prompt_set_hash": prompt_set_hash,
        "case_refs": case_refs,
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": _sha256_json(manifest_without_hash),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def validate_learning_flywheel_heldout_prompt_set_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a heldout prompt-set manifest is portable and hash-only."""

    errors: list[dict[str, str]] = []
    if manifest.get("schema_version") != LEARNING_FLYWHEEL_HELDOUT_PROMPT_SET_SCHEMA_VERSION:
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if manifest.get("raw_text_included") is not False:
        errors.append({"code": "raw_text_included", "field": "raw_text_included"})
    if manifest.get("legacy_expected_hint_fields_included") is not False:
        errors.append(
            {
                "code": "legacy_expected_hint_fields_included",
                "field": "legacy_expected_hint_fields_included",
            }
        )
    for field in _raw_text_fields(manifest):
        errors.append({"code": "raw_text_field_present", "field": field})

    primary_class_required = manifest.get("primary_class_required") is True
    case_refs = manifest.get("case_refs")
    if not isinstance(case_refs, list) or not case_refs:
        errors.append({"code": "missing_case_refs", "field": "case_refs"})
    else:
        seen: set[str] = set()
        for index, item in enumerate(case_refs):
            if not isinstance(item, Mapping):
                errors.append({"code": "invalid_case_ref", "field": f"case_refs[{index}]"})
                continue
            case_id = _text(item.get("case_id"))
            if not case_id:
                errors.append(
                    {"code": "missing_case_id", "field": f"case_refs[{index}].case_id"}
                )
            elif case_id in seen:
                errors.append(
                    {"code": "duplicate_case_id", "field": f"case_refs[{index}].case_id"}
                )
            seen.add(case_id)
            if not _valid_sha256(item.get("prompt_sha256")):
                errors.append(
                    {
                        "code": "invalid_prompt_sha256",
                        "field": f"case_refs[{index}].prompt_sha256",
                    }
                )
            primary_class = item.get("primary_class")
            if primary_class_required and not _text(primary_class):
                errors.append(
                    {
                        "code": "missing_primary_class",
                        "field": f"case_refs[{index}].primary_class",
                    }
                )
            elif primary_class is not None and primary_class not in HELDOUT_PRIMARY_CLASS_VALUES:
                errors.append(
                    {
                        "code": "invalid_primary_class",
                        "field": f"case_refs[{index}].primary_class",
                    }
                )
    if not _valid_sha256(manifest.get("prompt_set_hash")):
        errors.append({"code": "invalid_prompt_set_hash", "field": "prompt_set_hash"})
    if not _valid_sha256(manifest.get("manifest_sha256")):
        errors.append({"code": "invalid_manifest_sha256", "field": "manifest_sha256"})
    return {"ok": not errors, "errors": errors}


def normalize_learning_flywheel_heldout_cases(
    value: Any,
    *,
    require_primary_class: bool = False,
) -> list[dict[str, Any]]:
    """Return local heldout cases with raw prompts for host-side eval tools."""

    raw_cases = value.get("cases") if isinstance(value, Mapping) else value
    if not isinstance(raw_cases, list):
        if isinstance(raw_cases, Iterable) and not isinstance(raw_cases, (str, bytes)):
            raw_cases = list(raw_cases)
        else:
            raise TypeError("cases JSON must be a list or an object with a cases list")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, Mapping):
            raise TypeError(f"cases[{index}] must be an object")
        legacy_keys = sorted(key for key in LEGACY_HINT_KEYS if key in item)
        if legacy_keys:
            raise ValueError(
                f"cases[{index}] contains removed legacy hint keys: {legacy_keys}"
            )
        case_id = _required_text(item.get("case_id") or item.get("id"), "case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        prompt = _required_text(item.get("prompt") or item.get("text"), "prompt")
        case: dict[str, Any] = {
            "case_id": case_id,
            "prompt": prompt,
        }
        tags = _text_list(item.get("tags"))
        if tags:
            case["tags"] = tags
        primary_class = _text(item.get("primary_class"))
        if require_primary_class and not primary_class:
            raise ValueError(f"cases[{index}] missing required primary_class")
        if primary_class:
            if primary_class not in HELDOUT_PRIMARY_CLASS_VALUES:
                allowed = ", ".join(sorted(HELDOUT_PRIMARY_CLASS_VALUES))
                raise ValueError(
                    f"cases[{index}] invalid primary_class: {primary_class!r}; "
                    f"allowed values: {allowed}"
                )
            case["primary_class"] = primary_class
        cases.append(case)
    if not cases:
        raise ValueError("cases must contain at least one heldout prompt")
    return cases


def _case_refs(
    cases: Iterable[Mapping[str, Any]],
    *,
    require_primary_class: bool,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for case in normalize_learning_flywheel_heldout_cases(
        cases,
        require_primary_class=require_primary_class,
    ):
        case_id = case["case_id"]
        prompt = case["prompt"]
        tags = _text_list(case.get("tags"))
        ref: dict[str, Any] = {
            "case_id": case_id,
            "prompt_sha256": _sha256_text(prompt),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        }
        if tags:
            ref["tags"] = tags
        primary_class = _text(case.get("primary_class"))
        if primary_class:
            ref["primary_class"] = primary_class
        refs.append(ref)
    return sorted(refs, key=lambda item: item["case_id"])


def _raw_text_fields(payload: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in RAW_TEXT_KEYS:
                yield field
            yield from _raw_text_fields(value, field)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            field = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _raw_text_fields(item, field)


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({_text(item) for item in value if _text(item)})


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value[7:] if value.startswith("sha256:") else value
    return len(raw) == 64 and all(char in "0123456789abcdef" for char in raw.lower())


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
