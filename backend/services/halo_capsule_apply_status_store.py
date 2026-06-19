# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persist device-originated Halo capsule apply status receipts.

This store is deliberately status-only. It records whether a trusted device
received/applied/failed one capsule transfer, but it never stores artifact
contents or user data.
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


APPLY_STATUS_SCHEMA_VERSION = "edgestudio.halo_capsule_apply_status.v1"
APPLY_STATUS_RECEIPT_SCHEMA_VERSION = "edgestudio.halo_capsule_apply_status_receipt.v1"
ALLOWED_APPLY_STATUSES = {"received", "applied", "failed"}


@dataclass
class HaloCapsuleApplyStatusError(ValueError):
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


def default_halo_capsule_apply_status_root() -> Path:
    configured = os.environ.get("EDGE_HALO_CAPSULE_APPLY_STATUS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("db", "capsule_apply_status")


def store_halo_capsule_apply_status(
    peer_id: str,
    payload: dict[str, Any],
    *,
    source: str = "mesh",
    root: Path | None = None,
) -> dict[str, Any]:
    clean_peer_id = _required_id(peer_id, "peer_id")
    clean_payload = _clean_apply_status_payload(payload)
    source_clean = _safe_source(source)

    received_at_ms = int(time.time() * 1000)
    payload_bytes = json.dumps(
        clean_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    apply_status_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    receipt = {
        "schema_version": APPLY_STATUS_RECEIPT_SCHEMA_VERSION,
        "peer_id": clean_peer_id,
        "source": source_clean,
        "received_at": received_at_ms / 1000.0,
        "apply_status_sha256": apply_status_sha256,
        "transfer_id": clean_payload["transfer_id"],
        "capsule_id": clean_payload["capsule_id"],
        "status": clean_payload["status"],
        "artifact_sha256": clean_payload.get("artifact_sha256"),
        "canonical_sha256": clean_payload.get("canonical_sha256"),
        "runtime_version": clean_payload.get("runtime_version"),
        "prefix_token_count": clean_payload.get("prefix_token_count"),
        "applied_at_unix_seconds": clean_payload.get("applied_at_unix_seconds"),
        "error_code": clean_payload.get("error_code"),
    }
    record = {
        "receipt": receipt,
        "payload": clean_payload,
    }

    base = (root or default_halo_capsule_apply_status_root()).expanduser().resolve()
    peer_dir = base / _path_component(clean_peer_id)
    status_dir = (
        peer_dir
        / _path_component(clean_payload["transfer_id"])
        / _path_component(clean_payload["capsule_id"])
    )
    history_dir = status_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    _write_json(status_dir / "latest.json", record)
    _write_json(peer_dir / "latest.json", record)
    _write_json(
        history_dir / f"{received_at_ms}-{apply_status_sha256[:12]}.json",
        record,
    )
    return receipt


def latest_halo_capsule_apply_status(
    peer_id: str,
    *,
    transfer_id: str | None = None,
    capsule_id: str | None = None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    clean_peer_id = _required_id(peer_id, "peer_id")
    base = (root or default_halo_capsule_apply_status_root()).expanduser().resolve()
    if transfer_id is not None or capsule_id is not None:
        clean_transfer_id = _required_id(transfer_id, "transfer_id")
        clean_capsule_id = _required_id(capsule_id, "capsule_id")
        path = (
            base
            / _path_component(clean_peer_id)
            / _path_component(clean_transfer_id)
            / _path_component(clean_capsule_id)
            / "latest.json"
        )
    else:
        path = base / _path_component(clean_peer_id) / "latest.json"
    return _read_record(path)


def _clean_apply_status_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HaloCapsuleApplyStatusError(
            "invalid_payload",
            "payload must be a JSON object",
            {"type": type(payload).__name__},
        )

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != APPLY_STATUS_SCHEMA_VERSION:
        raise HaloCapsuleApplyStatusError(
            "unsupported_schema_version",
            f"unsupported schema_version: {schema_version}",
            {"expected": APPLY_STATUS_SCHEMA_VERSION},
        )

    status = str(payload.get("status") or "").strip()
    if status not in ALLOWED_APPLY_STATUSES:
        raise HaloCapsuleApplyStatusError(
            "invalid_status",
            f"unsupported apply status: {status}",
            {"allowed": sorted(ALLOWED_APPLY_STATUSES)},
        )

    clean: dict[str, Any] = {
        "schema_version": APPLY_STATUS_SCHEMA_VERSION,
        "transfer_id": _required_id(payload.get("transfer_id"), "transfer_id"),
        "capsule_id": _required_id(payload.get("capsule_id"), "capsule_id"),
        "status": status,
    }

    for key in ("artifact_sha256", "canonical_sha256"):
        value = _optional_sha256(payload.get(key), key)
        if value is not None:
            clean[key] = value

    runtime_version = _optional_text(payload.get("runtime_version"), "runtime_version")
    if runtime_version is not None:
        clean["runtime_version"] = runtime_version

    prefix_token_count = _optional_non_negative_int(
        payload.get("prefix_token_count"),
        "prefix_token_count",
    )
    if prefix_token_count is not None:
        clean["prefix_token_count"] = prefix_token_count

    applied_at = _optional_float(
        payload.get("applied_at_unix_seconds"),
        "applied_at_unix_seconds",
    )
    if applied_at is not None:
        clean["applied_at_unix_seconds"] = applied_at

    error_code = _optional_text(payload.get("error_code"), "error_code")
    if error_code is not None:
        clean["error_code"] = error_code

    error_message = _optional_text(
        payload.get("error_message"),
        "error_message",
        max_len=1024,
    )
    if error_message is not None:
        clean["error_message"] = error_message

    return clean


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HaloCapsuleApplyStatusError(
            "apply_status_store_corrupt",
            "failed to read latest Halo capsule apply status",
            {"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("receipt"), dict):
        raise HaloCapsuleApplyStatusError(
            "apply_status_store_corrupt",
            "latest Halo capsule apply status record is invalid",
            {"path": str(path)},
        )
    return data


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _required_id(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HaloCapsuleApplyStatusError(
            "missing_required_id",
            f"{name} is required",
            {"field": name},
        )
    if len(text) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", text):
        raise HaloCapsuleApplyStatusError(
            "invalid_id",
            f"{name} contains unsupported characters",
            {"field": name},
        )
    return text


def _path_component(value: str) -> str:
    return value.replace("/", "_")


def _safe_source(value: str) -> str:
    source = str(value or "").strip().lower()
    return source if source in {"mesh", "api", "test"} else "api"


def _optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not re.fullmatch(r"[a-f0-9]{64}", text):
        raise HaloCapsuleApplyStatusError(
            "invalid_sha256",
            f"{name} must be a 64-character lowercase sha256 hex string",
            {"field": name},
        )
    return text


def _optional_text(value: Any, name: str, *, max_len: int = 128) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len:
        raise HaloCapsuleApplyStatusError(
            "text_too_long",
            f"{name} is too long",
            {"field": name, "max_len": max_len},
        )
    return text


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise HaloCapsuleApplyStatusError(
            "invalid_int",
            f"{name} must be an integer",
            {"field": name},
        ) from exc
    if number < 0:
        raise HaloCapsuleApplyStatusError(
            "invalid_int",
            f"{name} must be non-negative",
            {"field": name},
        )
    return number


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HaloCapsuleApplyStatusError(
            "invalid_float",
            f"{name} must be a number",
            {"field": name},
        ) from exc
