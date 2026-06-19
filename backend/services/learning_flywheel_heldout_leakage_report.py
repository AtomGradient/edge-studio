# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hash-only heldout leakage report for Learning Flywheel paired evals."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


LEARNING_FLYWHEEL_HELDOUT_LEAKAGE_REPORT_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.heldout_leakage_report.v1"
)

REQUIRED_SOURCE_KINDS = (
    "correction_inputs",
    "rpp_inputs",
    "training_rows",
    "prompt_patches",
    "eval_fixtures",
)

RAW_TEXT_KEYS = {
    "answer",
    "correct_response",
    "correction",
    "correction_text",
    "expected_text",
    "expected_tool",
    "golden_answer",
    "input_text",
    "messages",
    "prompt",
    "question",
    "raw_text",
    "reference_answer",
    "response",
    "selected_tools",
    "text",
    "tool_calls",
    "transcript",
    "user_text",
}
LEGACY_HINT_KEYS = {
    "correct_response",
    "expected_text",
    "expected_tool",
    "golden_answer",
    "reference_answer",
}


def build_learning_flywheel_heldout_leakage_report(
    *,
    heldout_manifest: Mapping[str, Any],
    leakage_sources: Iterable[Mapping[str, Any]],
    run_id: str,
    evidence_scope: str,
    required_source_kinds: Iterable[str] = REQUIRED_SOURCE_KINDS,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare heldout case ids and prompt hashes against local input sources."""

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    heldout = _heldout_refs(heldout_manifest)
    required_kinds = sorted({_required_text(kind, "required_source_kind") for kind in required_source_kinds})
    source_refs = _source_refs(leakage_sources)
    present_kinds = {source["source_kind"] for source in source_refs}
    missing_required_source_kinds = [
        kind for kind in required_kinds if kind not in present_kinds
    ]

    heldout_case_ids = set(heldout["case_ids"])
    heldout_prompt_hashes = set(heldout["prompt_hashes"])
    overlaps: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    checked_item_count = 0
    for source in source_refs:
        for item in source["item_refs"]:
            has_case_id = bool(item.get("case_id"))
            has_prompt_hash = bool(item.get("prompt_sha256"))
            if not has_case_id and not has_prompt_hash:
                unverifiable.append(
                    {
                        "source_id": source["source_id"],
                        "source_kind": source["source_kind"],
                        "item_id": item["item_id"],
                        "reason": "missing_case_id_and_prompt_hash",
                    }
                )
                continue
            checked_item_count += 1
            overlap_types: list[str] = []
            if has_case_id and item["case_id"] in heldout_case_ids:
                overlap_types.append("case_id")
            if has_prompt_hash and item["prompt_sha256"] in heldout_prompt_hashes:
                overlap_types.append("prompt_sha256")
            if overlap_types:
                overlaps.append(
                    {
                        "source_id": source["source_id"],
                        "source_kind": source["source_kind"],
                        "item_id": item["item_id"],
                        "overlap_types": overlap_types,
                        "case_id": item.get("case_id"),
                        "prompt_sha256": item.get("prompt_sha256"),
                    }
                )

    status = (
        "passed"
        if not overlaps and not unverifiable and not missing_required_source_kinds
        else "leakage_blocked"
    )
    checked_material = {
        "eval_prompt_set_hash": heldout["eval_prompt_set_hash"],
        "heldout_case_ids": heldout["case_ids"],
        "heldout_prompt_hashes": heldout["prompt_hashes"],
        "required_source_kinds": required_kinds,
        "source_refs": source_refs,
    }
    manifest_without_hash = {
        "ok": status == "passed",
        "schema_version": LEARNING_FLYWHEEL_HELDOUT_LEAKAGE_REPORT_SCHEMA_VERSION,
        "status": status,
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "evidence_scope": normalized_scope,
        "eval_prompt_set_hash": heldout["eval_prompt_set_hash"],
        "writes_runtime_artifacts": False,
        "runs_device_harness": False,
        "invokes_host_model_judge": False,
        "raw_text_included": False,
        "legacy_expected_hint_fields_included": False,
        "required_source_kinds": required_kinds,
        "missing_required_source_kinds": missing_required_source_kinds,
        "heldout_case_count": len(heldout_case_ids),
        "heldout_prompt_hash_count": len(heldout_prompt_hashes),
        "source_count": len(source_refs),
        "checked_item_count": checked_item_count,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "unverifiable_count": len(unverifiable),
        "unverifiable": unverifiable,
        "source_summaries": [
            {
                "source_id": source["source_id"],
                "source_kind": source["source_kind"],
                "item_count": len(source["item_refs"]),
            }
            for source in source_refs
        ],
        "checked_material_sha256": _sha256_json(checked_material),
    }
    manifest = {
        **manifest_without_hash,
        "report_sha256": _sha256_json(manifest_without_hash),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def validate_learning_flywheel_heldout_leakage_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a heldout leakage report is portable and raw-free."""

    errors: list[dict[str, str]] = []
    if (
        report.get("schema_version")
        != LEARNING_FLYWHEEL_HELDOUT_LEAKAGE_REPORT_SCHEMA_VERSION
    ):
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    status = report.get("status")
    if status not in {"passed", "leakage_blocked"}:
        errors.append({"code": "invalid_status", "field": "status"})
    if report.get("raw_text_included") is not False:
        errors.append({"code": "raw_text_included", "field": "raw_text_included"})
    if report.get("legacy_expected_hint_fields_included") is not False:
        errors.append(
            {
                "code": "legacy_expected_hint_fields_included",
                "field": "legacy_expected_hint_fields_included",
            }
        )
    for field in _raw_text_fields(report):
        errors.append({"code": "raw_text_field_present", "field": field})
    for field in ("eval_prompt_set_hash", "checked_material_sha256", "report_sha256"):
        if not _valid_sha256(report.get(field)):
            errors.append({"code": "invalid_sha256", "field": field})

    if report.get("ok") is True and status != "passed":
        errors.append({"code": "ok_true_without_passed_status", "field": "ok"})
    if status == "passed":
        for field in (
            "missing_required_source_kinds",
            "overlaps",
            "unverifiable",
        ):
            value = report.get(field)
            if value not in ([], ()):
                errors.append({"code": "passed_report_has_blockers", "field": field})
    return {"ok": not errors, "errors": errors}


def _heldout_refs(heldout_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(heldout_manifest, Mapping):
        raise ValueError("heldout_manifest must be an object")
    case_refs = heldout_manifest.get("case_refs")
    if not isinstance(case_refs, list) or not case_refs:
        raise ValueError("heldout_manifest.case_refs must be a non-empty list")
    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    for index, item in enumerate(case_refs):
        if not isinstance(item, Mapping):
            raise ValueError(f"heldout_manifest.case_refs[{index}] must be an object")
        case_ids.add(_required_text(item.get("case_id"), "case_id"))
        prompt_hashes.add(_required_sha256(item.get("prompt_sha256"), "prompt_sha256"))
    eval_prompt_set_hash = _required_sha256(
        heldout_manifest.get("prompt_set_hash")
        or heldout_manifest.get("eval_prompt_set_hash"),
        "eval_prompt_set_hash",
    )
    return {
        "eval_prompt_set_hash": eval_prompt_set_hash,
        "case_ids": sorted(case_ids),
        "prompt_hashes": sorted(prompt_hashes),
    }


def _source_refs(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"leakage_sources[{index}] must be an object")
        source_kind = _required_text(source.get("source_kind") or source.get("kind"), "source_kind")
        source_id = _required_text(
            source.get("source_id") or source.get("id") or source_kind,
            "source_id",
        )
        source_key = f"{source_kind}:{source_id}"
        if source_key in seen_sources:
            raise ValueError(f"duplicate source: {source_key}")
        seen_sources.add(source_key)
        items = source.get("items")
        if not isinstance(items, list):
            raise ValueError(f"leakage_sources[{index}].items must be a list")
        refs.append(
            {
                "source_id": source_id,
                "source_kind": source_kind,
                "item_refs": _source_item_refs(source_id=source_id, items=items),
            }
        )
    return sorted(refs, key=lambda item: (item["source_kind"], item["source_id"]))


def _source_item_refs(*, source_id: str, items: list[Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"{source_id}.items[{index}] must be an object")
        legacy_keys = sorted(key for key in LEGACY_HINT_KEYS if key in item)
        if legacy_keys:
            raise ValueError(
                f"{source_id}.items[{index}] contains removed legacy hint keys: {legacy_keys}"
            )
        item_id = _text(item.get("item_id") or item.get("id")) or f"{source_id}:{index + 1}"
        if item_id in seen_items:
            raise ValueError(f"duplicate item_id in {source_id}: {item_id}")
        seen_items.add(item_id)
        ref: dict[str, Any] = {"item_id": item_id}
        case_id = _text(item.get("case_id"))
        if case_id:
            ref["case_id"] = case_id
        prompt_hash = _prompt_hash(item)
        if prompt_hash:
            ref["prompt_sha256"] = prompt_hash
        content_hash = _content_hash(item)
        if content_hash:
            ref["content_sha256"] = content_hash
        refs.append(ref)
    return sorted(refs, key=lambda item: item["item_id"])


def _prompt_hash(item: Mapping[str, Any]) -> str | None:
    prompt_hash = _text(item.get("prompt_sha256") or item.get("prompt_hash"))
    if prompt_hash:
        return _required_sha256(prompt_hash, "prompt_sha256")
    prompt = (
        _text(item.get("prompt"))
        or _text(item.get("text"))
        or _text(item.get("input_text"))
        or _text(item.get("question"))
    )
    return _sha256_text(prompt) if prompt else None


def _content_hash(item: Mapping[str, Any]) -> str | None:
    content_hash = _text(item.get("content_sha256") or item.get("content_hash"))
    if content_hash:
        return _required_sha256(content_hash, "content_sha256")
    raw = (
        _text(item.get("correction_text"))
        or _text(item.get("raw_text"))
        or _text(item.get("transcript"))
    )
    return _sha256_text(raw) if raw else None


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


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
