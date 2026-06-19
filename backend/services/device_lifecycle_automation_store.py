# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persist audit-only device lifecycle automation decisions.

This store records what the host-side lifecycle coordinator would do for a
device snapshot. It does not execute pushes, run regeneration, schedule work,
or mutate device state beyond the local audit receipt.
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


DEVICE_LIFECYCLE_AUTOMATION_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.device_lifecycle_automation_decision_receipt.v1"
)


@dataclass
class DeviceLifecycleAutomationStoreError(ValueError):
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


def default_device_lifecycle_automation_root() -> Path:
    configured = os.environ.get("EDGE_DEVICE_LIFECYCLE_AUTOMATION_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("db", "device_lifecycle_automation")


def store_device_lifecycle_automation_decision(
    decision: dict[str, Any],
    *,
    source: str = "api",
    root: Path | None = None,
) -> dict[str, Any]:
    """Persist one decision receipt, deduping by decision_key."""

    clean_decision = _clean_decision(decision)
    source_clean = _safe_source(source)
    peer_id = clean_decision["peer_id"]
    decision_key = clean_decision["decision_key"]
    base = (root or default_device_lifecycle_automation_root()).expanduser().resolve()
    peer_dir = base / _path_component(peer_id)
    key_path = peer_dir / "decisions" / f"{decision_key}.json"

    existing = _read_record(key_path)
    if existing is not None:
        receipt = dict(existing["receipt"])
        receipt["deduped"] = True
        receipt["duplicate_of"] = receipt.get("decision_id")
        return receipt

    received_at_ms = int(time.time() * 1000)
    decision_sha256 = _canonical_sha256(clean_decision)
    lifecycle = _dict(clean_decision.get("lifecycle"))
    action = _dict(clean_decision.get("plan_action"))
    candidate = _dict(clean_decision.get("candidate"))
    effects = _dict(clean_decision.get("effects"))
    receipt = {
        "schema_version": DEVICE_LIFECYCLE_AUTOMATION_RECEIPT_SCHEMA_VERSION,
        "decision_id": f"device-lifecycle-auto-{received_at_ms}-{decision_key[:12]}",
        "source": source_clean,
        "peer_id": peer_id,
        "received_at": received_at_ms / 1000.0,
        "decision_key": decision_key,
        "decision_sha256": decision_sha256,
        "snapshot_sha256": _optional_text(clean_decision.get("snapshot_sha256")),
        "lifecycle_phase": _optional_text(lifecycle.get("phase")),
        "lifecycle_phase_label": _optional_text(lifecycle.get("phase_label")),
        "action_kind": _optional_text(action.get("kind")) or "unknown",
        "candidate_kind": _optional_text(candidate.get("kind")) or "none",
        "policy_status": _optional_text(candidate.get("policy_status")) or "not_applicable",
        "side_effects_executed": effects.get("side_effects_executed") is True,
        "automatic_push": effects.get("capsule_push_executed") is True,
        "automatic_regen": effects.get("neural_imprint_regen_triggered") is True,
        "deduped": False,
    }
    record = {
        "receipt": receipt,
        "decision": clean_decision,
    }

    history_path = peer_dir / "history" / f"{received_at_ms}-{decision_key[:12]}.json"
    _write_json(key_path, record)
    _write_json(peer_dir / "latest.json", record)
    _write_json(history_path, record)
    return receipt


def latest_device_lifecycle_automation_decision(
    peer_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    clean_peer_id = _required_id(peer_id, "peer_id")
    path = (
        (root or default_device_lifecycle_automation_root()).expanduser().resolve()
        / _path_component(clean_peer_id)
        / "latest.json"
    )
    return _read_record(path)


def _clean_decision(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeviceLifecycleAutomationStoreError(
            "invalid_decision",
            "decision must be a JSON object",
            {"type": type(value).__name__},
        )
    peer_id = _required_id(value.get("peer_id"), "peer_id")
    decision_key = _required_sha256(value.get("decision_key"), "decision_key")
    clean = dict(value)
    clean["peer_id"] = peer_id
    clean["decision_key"] = decision_key
    return clean


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeviceLifecycleAutomationStoreError(
            "automation_decision_store_corrupt",
            "failed to read lifecycle automation decision",
            {"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("receipt"), dict):
        raise DeviceLifecycleAutomationStoreError(
            "automation_decision_store_corrupt",
            "lifecycle automation decision record is invalid",
            {"path": str(path)},
        )
    return data


def _required_id(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", text):
        raise DeviceLifecycleAutomationStoreError(
            "invalid_id",
            f"{field} must be 1-128 chars of A-Z a-z 0-9 . _ : -",
            {"field": field, "value": text},
        )
    return text


def _required_sha256(value: Any, field: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", text):
        raise DeviceLifecycleAutomationStoreError(
            "invalid_decision_key",
            f"{field} must be a sha256 hex digest",
            {"field": field},
        )
    return text


def _safe_source(value: str) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"api", "mesh", "test"} else "api"


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._:-]", "_", value)[:128]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
