# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persist Halo capsule automation run receipts.

This store records bounded automation run results for local audit/debugging.
It does not schedule work, retry transfers, push capsules, or restore artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_dirs import data_path


AUTOMATION_RUN_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.halo_capsule_automation_run_receipt.v1"
)


@dataclass
class HaloCapsuleAutomationRunStoreError(ValueError):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": False,
            "details": self.details or {},
        }


def default_halo_capsule_automation_run_root() -> Path:
    configured = os.environ.get("EDGE_HALO_CAPSULE_AUTOMATION_RUN_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("db", "capsule_automation_runs")


def store_halo_capsule_automation_run(
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    source: str = "api",
    root: Path | None = None,
) -> dict[str, Any]:
    clean_request = _clean_dict(request, "request")
    clean_response = _clean_dict(response, "response")
    source_clean = _safe_source(source)
    received_at_ms = int(time.time() * 1000)
    payload = {
        "request": clean_request,
        "response": clean_response,
    }
    run_sha256 = _canonical_sha256(payload)
    run_id = f"halo-auto-run-{received_at_ms}-{run_sha256[:12]}"
    receipt = {
        "schema_version": AUTOMATION_RUN_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source_clean,
        "received_at": received_at_ms / 1000.0,
        "run_sha256": run_sha256,
        "dry_run": bool(clean_response.get("dry_run")),
        "attempted_count": _optional_int(clean_response.get("attempted_count")) or 0,
        "pushed_count": _optional_int(clean_response.get("pushed_count")) or 0,
        "peer_ids": _clean_peer_ids(clean_request.get("peer_ids")),
    }
    record = {
        "receipt": receipt,
        "request": clean_request,
        "response": clean_response,
    }

    base = (root or default_halo_capsule_automation_run_root()).expanduser().resolve()
    history_dir = base / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    _write_json(base / "latest.json", record)
    _write_json(history_dir / f"{received_at_ms}-{run_sha256[:12]}.json", record)
    return receipt


def latest_halo_capsule_automation_run(
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    path = (root or default_halo_capsule_automation_run_root()).expanduser().resolve() / "latest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HaloCapsuleAutomationRunStoreError(
            "automation_run_store_corrupt",
            "failed to read latest Halo automation run",
            {"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("receipt"), dict):
        raise HaloCapsuleAutomationRunStoreError(
            "automation_run_store_corrupt",
            "latest Halo automation run record is invalid",
            {"path": str(path)},
        )
    return data


def _clean_dict(value: dict[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HaloCapsuleAutomationRunStoreError(
            "invalid_record",
            f"{name} must be a JSON object",
            {"field": name, "type": type(value).__name__},
        )
    return value


def _clean_peer_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    peer_ids: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", text):
            peer_ids.append(text)
    return peer_ids


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _safe_source(value: str) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"api", "scheduler", "test"} else "api"


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
