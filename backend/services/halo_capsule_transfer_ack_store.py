# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persist device-originated Halo capsule transfer ACK receipts.

This store records offer/complete ACK metadata only. It intentionally does not
store artifact contents or user data.
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


TRANSFER_ACK_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.halo_capsule_transfer_ack_receipt.v1"
)
ALLOWED_ACK_KINDS = {"offer_ack", "complete_ack"}


@dataclass
class HaloCapsuleTransferAckError(ValueError):
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


def default_halo_capsule_transfer_ack_root() -> Path:
    configured = os.environ.get("EDGE_HALO_CAPSULE_TRANSFER_ACK_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("db", "capsule_transfer_ack")


def store_halo_capsule_transfer_ack(
    peer_id: str,
    payload: dict[str, Any],
    *,
    ack_kind: str = "offer_ack",
    source: str = "mesh",
    root: Path | None = None,
) -> dict[str, Any]:
    clean_peer_id = _required_id(peer_id, "peer_id")
    clean_ack_kind = _clean_ack_kind(ack_kind)
    clean_payload = _clean_transfer_ack_payload(payload)
    source_clean = _safe_source(source)

    received_at_ms = int(time.time() * 1000)
    payload_bytes = json.dumps(
        clean_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    ack_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    receipt = {
        "schema_version": TRANSFER_ACK_RECEIPT_SCHEMA_VERSION,
        "peer_id": clean_peer_id,
        "source": source_clean,
        "received_at": received_at_ms / 1000.0,
        "ack_sha256": ack_sha256,
        "ack_kind": clean_ack_kind,
        "transfer_id": clean_payload["transfer_id"],
        "accepted": clean_payload["accepted"],
        "reason": clean_payload.get("reason"),
        "canonical_sha256": clean_payload.get("canonical_sha256"),
    }
    record = {
        "receipt": receipt,
        "payload": clean_payload,
    }

    base = (root or default_halo_capsule_transfer_ack_root()).expanduser().resolve()
    peer_dir = base / _path_component(clean_peer_id)
    transfer_dir = peer_dir / _path_component(clean_payload["transfer_id"])
    history_dir = transfer_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    _write_json(transfer_dir / "latest.json", record)
    _write_json(peer_dir / "latest.json", record)
    _write_json(
        history_dir / f"{received_at_ms}-{ack_sha256[:12]}.json",
        record,
    )
    return receipt


def latest_halo_capsule_transfer_ack(
    peer_id: str,
    *,
    transfer_id: str | None = None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    clean_peer_id = _required_id(peer_id, "peer_id")
    base = (root or default_halo_capsule_transfer_ack_root()).expanduser().resolve()
    if transfer_id is not None:
        clean_transfer_id = _required_id(transfer_id, "transfer_id")
        path = (
            base
            / _path_component(clean_peer_id)
            / _path_component(clean_transfer_id)
            / "latest.json"
        )
    else:
        path = base / _path_component(clean_peer_id) / "latest.json"
    return _read_record(path)


def _clean_transfer_ack_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HaloCapsuleTransferAckError(
            "invalid_payload",
            "payload must be a JSON object",
            {"type": type(payload).__name__},
        )

    accepted = payload.get("accepted")
    if not isinstance(accepted, bool):
        raise HaloCapsuleTransferAckError(
            "invalid_accepted",
            "accepted must be a boolean",
            {"type": type(accepted).__name__},
        )

    clean: dict[str, Any] = {
        "transfer_id": _required_id(payload.get("transfer_id"), "transfer_id"),
        "accepted": accepted,
    }
    reason = _optional_text(payload.get("reason"), "reason", max_len=1024)
    if reason is not None:
        clean["reason"] = reason
    canonical = _optional_sha256(payload.get("canonical_sha256"), "canonical_sha256")
    if canonical is not None:
        clean["canonical_sha256"] = canonical
    return clean


def _clean_ack_kind(value: str) -> str:
    text = str(value or "").strip().lower()
    if text not in ALLOWED_ACK_KINDS:
        raise HaloCapsuleTransferAckError(
            "invalid_ack_kind",
            f"unsupported ack kind: {text}",
            {"allowed": sorted(ALLOWED_ACK_KINDS)},
        )
    return text


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HaloCapsuleTransferAckError(
            "transfer_ack_store_corrupt",
            "failed to read latest Halo capsule transfer ACK",
            {"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("receipt"), dict):
        raise HaloCapsuleTransferAckError(
            "transfer_ack_store_corrupt",
            "latest Halo capsule transfer ACK record is invalid",
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
        raise HaloCapsuleTransferAckError(
            "missing_required_id",
            f"{name} is required",
            {"field": name},
        )
    if len(text) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", text):
        raise HaloCapsuleTransferAckError(
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
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise HaloCapsuleTransferAckError(
            "invalid_sha256",
            f"{name} must be a lowercase sha256 hex string",
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
        raise HaloCapsuleTransferAckError(
            "invalid_text",
            f"{name} is too long",
            {"field": name, "max_len": max_len},
        )
    return text
