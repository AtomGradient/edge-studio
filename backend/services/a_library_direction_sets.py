# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Validation and storage helpers for A-library direction-set YAML files."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .app_dirs import data_path


DIRECTION_SET_SOURCE_SCHEMA_VERSION = "edgestudio.a_library_direction_set_source.v1"
DIRECTION_SET_VALIDATION_SCHEMA_VERSION = "edgestudio.a_library_direction_set_validation.v1"
ALLOWED_DIRECTION_SOURCE_TYPES = frozenset({"host_model_seed", "claude_authored", "manual"})
MIN_DIRECTIONS = 10
MIN_EXAMPLES_PER_SIDE = 5
MIN_LENGTH_BALANCE = 0.7
MAX_DIRECTION_SET_ID_LENGTH = 80


class DirectionSetValidationError(ValueError):
    def __init__(self, report: dict[str, Any]):
        super().__init__("Invalid A-library direction-set YAML.")
        self.report = report


def a_library_source_root() -> Path:
    configured = os.environ.get("EDGESTUDIO_A_LIBRARY_SOURCE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return data_path("a_library_sources")


def sanitize_direction_set_id(value: str | None) -> str:
    raw = (value or "directions_a").strip()
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    raw = raw.strip("._-")
    if not raw:
        raw = "directions_a"
    return raw[:MAX_DIRECTION_SET_ID_LENGTH]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_direction_source_type(value: str | None) -> str:
    source_type = (value or "manual").strip()
    if source_type not in ALLOWED_DIRECTION_SOURCE_TYPES:
        raise ValueError(f"Unsupported direction source_type: {source_type}")
    return source_type


def validate_direction_yaml_file(
    path: str | Path,
    *,
    direction_set_id: str | None = None,
) -> dict[str, Any]:
    text = Path(path).expanduser().resolve().read_text(encoding="utf-8")
    return validate_direction_yaml_text(text, direction_set_id=direction_set_id)


def validate_direction_yaml_text(
    text: str,
    *,
    direction_set_id: str | None = None,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    yaml_sha256 = sha256_text(text)
    payload: Any
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        payload = None
        errors.append({
            "code": "invalid_yaml",
            "message": str(exc),
        })

    if not isinstance(payload, dict):
        payload = {}
        if not errors:
            errors.append({
                "code": "invalid_yaml_root",
                "message": "YAML root must be an object.",
            })

    schema_version = _string(payload.get("schema_version")) or _string(payload.get("version"))
    if not schema_version:
        errors.append({
            "code": "missing_schema_version",
            "message": "Direction-set YAML must include schema_version or version.",
        })

    resolved_id = sanitize_direction_set_id(
        direction_set_id
        or _string(payload.get("direction_set_id"))
        or _string(payload.get("id"))
        or "directions_a"
    )
    directions = payload.get("directions")
    direction_rows: list[dict[str, Any]] = []
    domains: Counter[str] = Counter()
    seen_names: set[str] = set()
    seen_texts: dict[str, str] = {}
    duplicate_texts: list[dict[str, str]] = []
    positive_count = 0
    negative_count = 0

    if not isinstance(directions, list):
        errors.append({
            "code": "directions_not_list",
            "message": "directions must be a list.",
        })
        directions = []
    elif len(directions) < MIN_DIRECTIONS:
        errors.append({
            "code": "too_few_directions",
            "message": f"Direction count must be >= {MIN_DIRECTIONS}.",
            "actual": len(directions),
            "minimum": MIN_DIRECTIONS,
        })

    for idx, raw_direction in enumerate(directions):
        label = f"directions[{idx}]"
        if not isinstance(raw_direction, dict):
            errors.append({
                "code": "direction_not_object",
                "message": f"{label} must be an object.",
                "direction_index": idx,
            })
            continue
        name = _string(raw_direction.get("name")) or f"direction_{idx}"
        if name in seen_names:
            original_name = name
            suffix = 2
            while f"{name}_{suffix}" in seen_names:
                suffix += 1
            name = f"{name}_{suffix}"
            errors.append({
                "code": "duplicate_direction_name_auto_fixed",
                "message": f"Duplicate name '{original_name}' renamed to '{name}'. Consider giving each direction a unique descriptive name.",
                "direction_index": idx,
                "severity": "warning",
            })
        seen_names.add(name)
        domain = _string(raw_direction.get("domain")) or "uncategorized"
        description = _string(raw_direction.get("description")) or ""
        positive = _text_list(raw_direction.get("positive"))
        negative = _text_list(raw_direction.get("negative"))
        raw_positive_count = len(positive)
        raw_negative_count = len(negative)
        positive_count += raw_positive_count
        negative_count += raw_negative_count
        domains[domain] += 1

        if raw_positive_count < MIN_EXAMPLES_PER_SIDE:
            errors.append({
                "code": "too_few_positive_examples",
                "message": f"{name} positive examples must be >= {MIN_EXAMPLES_PER_SIDE}.",
                "direction": name,
                "actual": raw_positive_count,
                "minimum": MIN_EXAMPLES_PER_SIDE,
            })
        if raw_negative_count < MIN_EXAMPLES_PER_SIDE:
            errors.append({
                "code": "too_few_negative_examples",
                "message": f"{name} negative examples must be >= {MIN_EXAMPLES_PER_SIDE}.",
                "direction": name,
                "actual": raw_negative_count,
                "minimum": MIN_EXAMPLES_PER_SIDE,
            })
        for side, values in (("positive", positive), ("negative", negative)):
            for item_idx, value in enumerate(values):
                if not value.strip():
                    errors.append({
                        "code": "empty_text",
                        "message": f"{name}.{side}[{item_idx}] is empty.",
                        "direction": name,
                        "side": side,
                        "index": item_idx,
                    })
                    continue
                normalized = _normalize_text(value)
                previous = seen_texts.get(normalized)
                current = f"{name}.{side}[{item_idx}]"
                if previous is not None:
                    duplicate_texts.append({"first": previous, "duplicate": current})
                else:
                    seen_texts[normalized] = current

        for pair_idx, (pos, neg) in enumerate(zip(positive, negative)):
            max_len = max(len(pos), len(neg))
            min_len = min(len(pos), len(neg))
            if max_len > 0 and min_len / max_len <= MIN_LENGTH_BALANCE:
                errors.append({
                    "code": "length_balance_failed",
                    "message": f"{name} pair {pair_idx} length balance must be > {MIN_LENGTH_BALANCE}.",
                    "direction": name,
                    "pair_index": pair_idx,
                    "balance": round(min_len / max_len, 4),
                    "minimum_exclusive": MIN_LENGTH_BALANCE,
                })
        direction_rows.append({
            "name": name,
            "description": description,
            "domain": domain,
            "positive_count": raw_positive_count,
            "negative_count": raw_negative_count,
        })

    if duplicate_texts:
        errors.append({
            "code": "duplicate_text",
            "message": "Direction-set YAML contains duplicate text.",
            "duplicates": duplicate_texts[:20],
            "duplicate_count": len(duplicate_texts),
        })

    hard_errors = [e for e in errors if e.get("severity") != "warning"]
    return {
        "ok": not hard_errors,
        "schema_version": DIRECTION_SET_VALIDATION_SCHEMA_VERSION,
        "direction_set_id": resolved_id,
        "source_schema_version": schema_version,
        "yaml_sha256": yaml_sha256,
        "errors": errors,
        "coverage": {
            "direction_count": len(direction_rows),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "sentence_count": positive_count + negative_count,
            "domains": dict(sorted(domains.items())),
            "directions": direction_rows,
        },
    }


def store_direction_yaml(
    text: str,
    *,
    direction_set_id: str | None = None,
) -> dict[str, Any]:
    report = validate_direction_yaml_text(text, direction_set_id=direction_set_id)
    if not report["ok"]:
        raise DirectionSetValidationError(report)
    text, report = _canonicalize_direction_yaml_for_storage(
        text,
        report=report,
        direction_set_id=direction_set_id,
    )
    set_id = sanitize_direction_set_id(report["direction_set_id"])
    root = a_library_source_root() / set_id
    root.mkdir(parents=True, exist_ok=True)
    yaml_path = root / "directions.yaml"
    yaml_path.write_text(text, encoding="utf-8")
    report_path = root / "validation_report.json"
    stored_report = report | {
        "stored_path": str(yaml_path),
        "validation_report_path": str(report_path),
    }
    report_path.write_text(
        json.dumps(stored_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stored_report


def _canonicalize_direction_yaml_for_storage(
    text: str,
    *,
    report: dict[str, Any],
    direction_set_id: str | None,
) -> tuple[str, dict[str, Any]]:
    warnings = [error for error in report.get("errors", []) if error.get("severity") == "warning"]
    if not any(error.get("code") == "duplicate_direction_name_auto_fixed" for error in warnings):
        return text, report
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict) or not isinstance(payload.get("directions"), list):
        return text, report
    canonical_names = [
        str(row.get("name"))
        for row in report.get("coverage", {}).get("directions", [])
        if isinstance(row, dict) and row.get("name")
    ]
    if len(canonical_names) != len(payload["directions"]):
        return text, report
    for raw_direction, name in zip(payload["directions"], canonical_names):
        if isinstance(raw_direction, dict):
            raw_direction["name"] = name
    payload["direction_set_id"] = sanitize_direction_set_id(report.get("direction_set_id"))
    canonical_text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    canonical_report = validate_direction_yaml_text(canonical_text, direction_set_id=direction_set_id)
    if not canonical_report["ok"]:
        raise DirectionSetValidationError(canonical_report)
    canonical_report["errors"] = warnings + [
        error for error in canonical_report.get("errors", []) if error.get("severity") != "warning"
    ]
    return canonical_text, canonical_report


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item if isinstance(item, str) else "" for item in value]


def _string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())
