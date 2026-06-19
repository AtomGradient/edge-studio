# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Utilities for training-side route/action learner datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUTE_ACTION_POLICY_DATASET_MERGE_SCHEMA_VERSION = (
    "edgestudio.route_action_policy_dataset_merge.v0"
)


def merge_route_action_policy_datasets(
    *,
    input_paths: list[Path],
    output_path: Path,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Merge learner JSONL datasets with stable training-material dedupe."""

    if not input_paths:
        raise ValueError("input_paths must not be empty")
    inputs = [Path(path) for path in input_paths]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(str(path))

    output = Path(output_path)
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    seen_material_keys: set[str] = set()
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    duplicate_sample_id_count = 0
    duplicate_material_count = 0
    for path in inputs:
        source_count = 0
        for row in _read_jsonl(path):
            sample_id_key = _sample_id_key(row)
            material_key = _material_key(row)
            if sample_id_key is not None and sample_id_key in seen_sample_ids:
                duplicate_count += 1
                duplicate_sample_id_count += 1
                continue
            if material_key in seen_material_keys:
                duplicate_count += 1
                duplicate_material_count += 1
                continue
            if sample_id_key is not None:
                seen_sample_ids.add(sample_id_key)
            seen_material_keys.add(material_key)
            rows.append(row)
            source_count += 1
        source_counts[str(path)] = source_count

    output.parent.mkdir(parents=True, exist_ok=True)
    data = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in rows
        )
        + ("\n" if rows else "")
    ).encode("utf-8")
    output.write_bytes(data)

    summary = {
        "ok": True,
        "schema_version": ROUTE_ACTION_POLICY_DATASET_MERGE_SCHEMA_VERSION,
        "status": "written",
        "generated_at": _utc_now(),
        "training_side_only": True,
        "writes_runtime_artifacts": False,
        "input_paths": [str(path) for path in inputs],
        "output_path": str(output),
        "summary_path": str(summary_path) if summary_path else None,
        "input_count": len(inputs),
        "sample_count": len(rows),
        "duplicate_count": duplicate_count,
        "duplicate_sample_id_count": duplicate_sample_id_count,
        "duplicate_material_count": duplicate_material_count,
        "source_sample_counts": source_counts,
        "sha256": hashlib.sha256(data).hexdigest(),
        "intent_counts": dict(Counter(_intent(row) for row in rows)),
        "tool_counts": dict(Counter(_tools_key(row) for row in rows)),
    }
    if summary_path is not None:
        summary_output = Path(summary_path)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary["summary_path"] = str(summary_output)
        summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        rows.append(value)
    return rows


def _sample_id_key(row: dict[str, Any]) -> str | None:
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return f"sample_id:{sample_id.strip()}"
    return None


def _material_key(row: dict[str, Any]) -> str:
    input_value = row.get("input") if isinstance(row.get("input"), dict) else {}
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    return "material:" + hashlib.sha256(
        json.dumps(
            {
                "input_text": _normalize_text(input_value.get("text")),
                "route_intent": _normalize_scalar(target.get("route_intent")),
                "selected_tools": _normalize_tools(target.get("selected_tools")),
                "tool_call_plan": _normalize_tool_call_plan(target.get("tool_call_plan")),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _intent(row: dict[str, Any]) -> str:
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    value = target.get("route_intent")
    return str(value or "<missing>").strip() or "<missing>"


def _tools_key(row: dict[str, Any]) -> str:
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    tools = target.get("selected_tools")
    if not isinstance(tools, list) or not tools:
        return "<none>"
    return ",".join(str(tool).strip() for tool in tools if str(tool).strip()) or "<none>"


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _normalize_scalar(value: Any) -> str:
    return str(value or "").strip()


def _normalize_tools(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tools = [str(tool).strip() for tool in value if str(tool).strip()]
    return sorted(tools)


def _normalize_tool_call_plan(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    plans: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("toolName")
        if not isinstance(name, str) or not name.strip():
            name = item.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            name = item.get("tool")
        arguments = item.get("arguments")
        if not isinstance(arguments, dict):
            arguments = item.get("args")
        plans.append(
            {
                "tool_name": str(name or "").strip(),
                "arguments": _normalize_json_value(arguments if isinstance(arguments, dict) else {}),
            }
        )
    return plans


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
