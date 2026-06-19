# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Local store for device-originated Persona source uploads.

Persona source uploads are the host-side input material for A3 generation:
tool schema exports and optional RPP profile bodies. The store validates hashes,
persists files under the local EdgeStudio data directory, and
returns an auditable receipt. It does not generate Neural Imprint artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_dirs import data_path


SOURCE_SCHEMA_VERSION = "edgestudio.persona_source_upload.v1"
RECEIPT_SCHEMA_VERSION = "edgestudio.persona_source_receipt.v1"

SOURCE_KINDS = {
    "tool_schema_only",
    "device_rpp_profile",
    "host_rpp_profile",
}


@dataclass
class PersonaSourceStoreError(ValueError):
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


@dataclass(frozen=True)
class StoredPersonaSource:
    peer_id: str
    source_id: str
    path: Path
    received_at_ms: int
    receipt: dict[str, Any]
    payload: dict[str, Any]


def default_persona_source_root() -> Path:
    configured = os.environ.get("EDGE_PERSONA_SOURCE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("persona", "sources")


def store_persona_source_upload(
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate and persist one Persona source upload."""

    clean_payload = _validate_payload(payload)
    source_bytes = _canonical_json_bytes(clean_payload)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    source_id = f"persona_source-{source_sha256[:16]}"
    peer_id = clean_payload["peer_id"]

    base = (root or default_persona_source_root()).expanduser().resolve()
    peer_dir = base / _path_component(peer_id, "peer_id")
    source_dir = peer_dir / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    payload_path = source_dir / "payload.json"
    tool_specs_path = source_dir / "tool_specs.json"
    profile_body_path: Path | None = None

    payload_path.write_bytes(_pretty_json_bytes(clean_payload))
    tool_specs_path.write_bytes(
        _pretty_json_bytes(clean_payload["tool_schema_export"])
    )

    profile_body = clean_payload.get("profile_body")
    if isinstance(profile_body, str):
        profile_body_path = source_dir / "profile_body.txt"
        profile_body_path.write_text(profile_body, encoding="utf-8")

    received_at_ms = int(time.time() * 1000)
    receipt = {
        "ok": True,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": peer_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "source_kind": clean_payload["source_kind"],
        "app_id": clean_payload["app_id"],
        "base_model_id": clean_payload["base_model_id"],
        "tool_schema_sha256": clean_payload["tool_schema_sha256"],
        "profile_body_sha256": clean_payload.get("profile_body_sha256"),
        "rpp_run_id": clean_payload.get("rpp_run_id"),
        "created_at": clean_payload["created_at"],
        "received_at_ms": received_at_ms,
        "storage_path": str(source_dir),
        "payload_path": str(payload_path),
        "tool_specs_path": str(tool_specs_path),
        "profile_body_path": str(profile_body_path) if profile_body_path else None,
    }
    if "lineage" in clean_payload:
        receipt["lineage"] = clean_payload["lineage"]

    receipt_path = source_dir / "persona_source_receipt.json"
    receipt_path.write_bytes(_pretty_json_bytes(receipt))
    latest_path = peer_dir / "latest.json"
    latest_path.write_bytes(_pretty_json_bytes(receipt))
    return receipt


def latest_persona_source_for_peer(
    peer_id: str,
    *,
    root: Path | None = None,
) -> StoredPersonaSource | None:
    safe_peer_id = _path_component(str(peer_id).strip(), "peer_id")
    base = (root or default_persona_source_root()).expanduser().resolve()
    latest_path = base / safe_peer_id / "latest.json"
    if not latest_path.is_file():
        return None
    try:
        receipt = _read_json_object(latest_path)
        if receipt.get("peer_id") != safe_peer_id:
            return None
        peer_dir = base / safe_peer_id
        payload_path = (
            Path(str(receipt.get("payload_path") or "")).expanduser().resolve()
        )
        payload = _read_json_object(payload_path)
        source_dir = (
            Path(str(receipt.get("storage_path") or "")).expanduser().resolve()
        )
        if not _is_relative_to(source_dir, peer_dir):
            return None
        if not _is_relative_to(payload_path, source_dir):
            return None
        return StoredPersonaSource(
            peer_id=safe_peer_id,
            source_id=str(receipt.get("source_id") or ""),
            path=source_dir,
            received_at_ms=int(receipt.get("received_at_ms") or 0),
            receipt=receipt,
            payload=payload,
        )
    except Exception:  # noqa: BLE001
        return None
    return None


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PersonaSourceStoreError(
            "invalid_input",
            "payload must be a JSON object",
            {"type": type(payload).__name__},
        )

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != SOURCE_SCHEMA_VERSION:
        raise PersonaSourceStoreError(
            "unsupported_schema_version",
            f"unsupported schema_version: {schema_version}",
            {"expected": SOURCE_SCHEMA_VERSION},
        )

    peer_id = _required_id(payload, "peer_id")
    app_id = _required_id(payload, "app_id")
    base_model_id = _required_id(payload, "base_model_id")
    source_kind = _required_id(payload, "source_kind")
    if source_kind not in SOURCE_KINDS:
        raise PersonaSourceStoreError(
            "unsupported_source_kind",
            f"unsupported source_kind: {source_kind}",
            {"allowed": sorted(SOURCE_KINDS)},
        )

    tool_schema_export = payload.get("tool_schema_export")
    if not isinstance(tool_schema_export, dict):
        raise PersonaSourceStoreError(
            "invalid_tool_schema_export",
            "tool_schema_export must be an object",
            {"type": type(tool_schema_export).__name__},
        )
    tool_schema_sha256 = _required_hash(payload, "tool_schema_sha256")
    actual_tool_schema_sha256 = hashlib.sha256(
        _canonical_json_bytes(tool_schema_export)
    ).hexdigest()
    if tool_schema_sha256 != actual_tool_schema_sha256:
        raise PersonaSourceStoreError(
            "tool_schema_sha256_mismatch",
            "tool_schema_sha256 does not match tool_schema_export",
            {"expected": tool_schema_sha256, "actual": actual_tool_schema_sha256},
        )

    created_at = payload.get("created_at")
    if (
        isinstance(created_at, bool)
        or not isinstance(created_at, (int, float))
        or not math.isfinite(float(created_at))
    ):
        raise PersonaSourceStoreError(
            "invalid_created_at",
            "created_at must be a finite Unix timestamp number",
            {"type": type(created_at).__name__},
        )

    profile_body = payload.get("profile_body")
    profile_body_sha256 = _optional_hash(payload.get("profile_body_sha256"))
    if profile_body is not None and not isinstance(profile_body, str):
        raise PersonaSourceStoreError(
            "invalid_profile_body",
            "profile_body must be a string when provided",
            {"type": type(profile_body).__name__},
        )
    if isinstance(profile_body, str):
        actual_profile_sha256 = hashlib.sha256(
            profile_body.encode("utf-8")
        ).hexdigest()
        if not profile_body_sha256:
            raise PersonaSourceStoreError(
                "missing_profile_body_sha256",
                "profile_body_sha256 is required when profile_body is provided",
                {},
            )
        if profile_body_sha256 != actual_profile_sha256:
            raise PersonaSourceStoreError(
                "profile_body_sha256_mismatch",
                "profile_body_sha256 does not match profile_body",
                {"expected": profile_body_sha256, "actual": actual_profile_sha256},
            )
    elif profile_body_sha256:
        raise PersonaSourceStoreError(
            "orphan_profile_body_sha256",
            "profile_body_sha256 requires profile_body",
            {},
        )

    if source_kind in {"device_rpp_profile", "host_rpp_profile"} and not profile_body:
        raise PersonaSourceStoreError(
            "missing_profile_body",
            f"profile_body is required for source_kind={source_kind}",
            {},
        )

    clean_payload: dict[str, Any] = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "peer_id": peer_id,
        "app_id": app_id,
        "base_model_id": base_model_id,
        "tool_schema_export": tool_schema_export,
        "tool_schema_sha256": tool_schema_sha256,
        "source_kind": source_kind,
        "created_at": float(created_at),
    }
    if isinstance(profile_body, str):
        clean_payload["profile_body"] = profile_body
        clean_payload["profile_body_sha256"] = profile_body_sha256
    rpp_run_id = _optional_id(payload.get("rpp_run_id"), "rpp_run_id")
    if rpp_run_id:
        clean_payload["rpp_run_id"] = rpp_run_id
    lineage = payload.get("lineage")
    if lineage is not None:
        if not isinstance(lineage, dict):
            raise PersonaSourceStoreError(
                "invalid_lineage",
                "lineage must be an object when provided",
                {"type": type(lineage).__name__},
            )
        clean_payload["lineage"] = _validate_lineage(lineage)
    return clean_payload


def _validate_lineage(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "input_id",
        "input_sha256",
        "records_sha256",
        "text_sha256",
        "record_count",
        "total_text_chars",
        "correction_context_schema_version",
        "correction_compiler_schema_version",
        "correction_fingerprints",
        "correction_overlay_sha256",
        "correction_counts",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PersonaSourceStoreError(
            "unknown_lineage_fields",
            "lineage contains unsupported fields",
            {"fields": unknown},
        )

    clean: dict[str, Any] = {}
    for key in ("input_id", "input_sha256", "records_sha256", "text_sha256"):
        raw = value.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        clean[key] = (
            _optional_hash(text)
            if key.endswith("sha256")
            else _path_component(text, key)
        )

    for key in ("record_count", "total_text_chars"):
        raw = value.get(key)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise PersonaSourceStoreError(
                "invalid_lineage_count",
                f"lineage.{key} must be a non-negative integer",
                {"field": key, "value": raw},
            )
        clean[key] = raw

    schema_version = str(value.get("correction_context_schema_version") or "").strip()
    if schema_version:
        clean["correction_context_schema_version"] = schema_version

    compiler_schema_version = str(value.get("correction_compiler_schema_version") or "").strip()
    if compiler_schema_version:
        clean["correction_compiler_schema_version"] = compiler_schema_version

    overlay_sha = _optional_hash(value.get("correction_overlay_sha256"))
    if overlay_sha:
        clean["correction_overlay_sha256"] = overlay_sha

    fingerprints = value.get("correction_fingerprints")
    if fingerprints is not None:
        if not isinstance(fingerprints, list):
            raise PersonaSourceStoreError(
                "invalid_lineage_correction_fingerprints",
                "lineage.correction_fingerprints must be a list",
                {"type": type(fingerprints).__name__},
            )
        clean["correction_fingerprints"] = [
            item
            for item in (_optional_hash(raw) for raw in fingerprints)
            if item
        ]

    counts = value.get("correction_counts")
    if counts is not None:
        if not isinstance(counts, dict):
            raise PersonaSourceStoreError(
                "invalid_lineage_correction_counts",
                "lineage.correction_counts must be an object",
                {"type": type(counts).__name__},
            )
        clean_counts: dict[str, int] = {}
        for raw_key, raw_value in counts.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise PersonaSourceStoreError(
                    "invalid_lineage_correction_count",
                    "lineage.correction_counts values must be non-negative integers",
                    {"field": key, "value": raw_value},
                )
            clean_counts[key] = raw_value
        clean["correction_counts"] = clean_counts
    return clean


def _required_id(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise PersonaSourceStoreError(
            "missing_required_field",
            f"{key} is required",
            {"field": key},
        )
    return _path_component(value, key)


def _optional_id(value: Any, key: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _path_component(text, key)


def _required_hash(payload: dict[str, Any], key: str) -> str:
    value = _optional_hash(payload.get(key))
    if not value:
        raise PersonaSourceStoreError(
            "missing_required_field",
            f"{key} is required",
            {"field": key},
        )
    return value


def _optional_hash(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        text = text.removeprefix("sha256:")
    if not text:
        return None
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise PersonaSourceStoreError(
            "invalid_sha256",
            "sha256 fields must be 64 lowercase hex characters",
            {"value": text},
        )
    return text


def _path_component(value: str, field: str) -> str:
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in value:
        raise PersonaSourceStoreError(
            "invalid_path_component",
            f"{field} must not contain path separators or '..'",
            {"field": field, "value": value},
        )
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PersonaSourceStoreError(
            "invalid_stored_json",
            f"{path.name} must be a JSON object",
            {"path": str(path)},
        )
    return data


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
