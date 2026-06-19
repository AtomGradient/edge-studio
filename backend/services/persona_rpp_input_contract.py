# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Canonical Persona/RPP input contract store.

This is the A3.1a boundary between vertical apps and EdgeStudio/EdgeHalo.
Apps may export normalized learning records, but EdgeStudio does not parse
domain-specific payloads such as expenses, receipts, merchants, or categories.
It only validates the canonical contract, hashes it, persists it, and returns
an auditable receipt.
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


INPUT_SCHEMA_VERSION = "edgestudio.persona_rpp_input.v1"
RECEIPT_SCHEMA_VERSION = "edgestudio.persona_rpp_input_receipt.v1"

SOURCE_KINDS = {
    "app_facts",
    "imported_facts",
    "correction_overlay",
}

TOP_LEVEL_FIELDS = {
    "schema_version",
    "peer_id",
    "app_id",
    "base_model_id",
    "source_kind",
    "created_at",
    "records",
    "records_sha256",
    "input_note",
}

RECORD_FIELDS = {
    "record_id",
    "kind",
    "text",
    "created_at",
    "weight",
    "tags",
    "metadata",
}


@dataclass
class PersonaRPPInputContractError(ValueError):
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
class StoredPersonaRPPInput:
    peer_id: str
    input_id: str
    path: Path
    received_at_ms: int
    receipt: dict[str, Any]
    payload: dict[str, Any]


def default_persona_rpp_input_root() -> Path:
    configured = os.environ.get("EDGE_PERSONA_RPP_INPUT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("persona", "rpp_inputs")


def store_persona_rpp_input_contract(
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate and persist one canonical Persona/RPP input contract."""

    clean_payload = validate_persona_rpp_input_contract(payload)
    input_sha256 = persona_rpp_input_contract_sha256(clean_payload)
    input_id = f"persona_rpp_input-{input_sha256[:16]}"
    peer_id = clean_payload["peer_id"]

    base = (root or default_persona_rpp_input_root()).expanduser().resolve()
    peer_dir = base / _path_component(peer_id, "peer_id")
    input_dir = peer_dir / input_id
    input_dir.mkdir(parents=True, exist_ok=True)

    payload_path = input_dir / "payload.json"
    payload_path.write_bytes(_pretty_json_bytes(clean_payload))

    received_at_ms = int(time.time() * 1000)
    text_sha256 = hashlib.sha256(
        _canonical_json_bytes([record["text"] for record in clean_payload["records"]])
    ).hexdigest()
    receipt = {
        "ok": True,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": peer_id,
        "input_id": input_id,
        "input_sha256": input_sha256,
        "records_sha256": clean_payload["records_sha256"],
        "text_sha256": text_sha256,
        "record_count": len(clean_payload["records"]),
        "total_text_chars": sum(
            len(record["text"]) for record in clean_payload["records"]
        ),
        "source_kind": clean_payload["source_kind"],
        "app_id": clean_payload["app_id"],
        "base_model_id": clean_payload["base_model_id"],
        "created_at": clean_payload["created_at"],
        "received_at_ms": received_at_ms,
        "storage_path": str(input_dir),
        "payload_path": str(payload_path),
    }

    (input_dir / "persona_rpp_input_receipt.json").write_bytes(
        _pretty_json_bytes(receipt)
    )
    (peer_dir / "latest.json").write_bytes(_pretty_json_bytes(receipt))
    return receipt


def latest_persona_rpp_input_for_peer(
    peer_id: str,
    *,
    root: Path | None = None,
) -> StoredPersonaRPPInput | None:
    safe_peer_id = _path_component(str(peer_id).strip(), "peer_id")
    base = (root or default_persona_rpp_input_root()).expanduser().resolve()
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
        input_dir = Path(str(receipt.get("storage_path") or "")).expanduser().resolve()
        if not _is_relative_to(input_dir, peer_dir):
            return None
        if not _is_relative_to(payload_path, input_dir):
            return None
        return StoredPersonaRPPInput(
            peer_id=safe_peer_id,
            input_id=str(receipt.get("input_id") or ""),
            path=input_dir,
            received_at_ms=int(receipt.get("received_at_ms") or 0),
            receipt=receipt,
            payload=payload,
        )
    except Exception:  # noqa: BLE001
        return None


def validate_persona_rpp_input_contract(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PersonaRPPInputContractError(
            "invalid_input",
            "payload must be a JSON object",
            {"type": type(payload).__name__},
        )

    unknown = sorted(set(payload) - TOP_LEVEL_FIELDS)
    if unknown:
        raise PersonaRPPInputContractError(
            "unknown_top_level_fields",
            "persona RPP input must use the canonical contract, not app-specific top-level fields",
            {"fields": unknown},
        )

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != INPUT_SCHEMA_VERSION:
        raise PersonaRPPInputContractError(
            "unsupported_schema_version",
            f"unsupported schema_version: {schema_version}",
            {"expected": INPUT_SCHEMA_VERSION},
        )

    peer_id = _required_id(payload, "peer_id")
    app_id = _required_id(payload, "app_id")
    base_model_id = _required_id(payload, "base_model_id")
    source_kind = _required_id(payload, "source_kind")
    if source_kind not in SOURCE_KINDS:
        raise PersonaRPPInputContractError(
            "unsupported_source_kind",
            f"unsupported source_kind: {source_kind}",
            {"allowed": sorted(SOURCE_KINDS)},
        )

    created_at = _finite_number(payload.get("created_at"), "created_at")
    records = _validate_records(payload.get("records"))
    records_sha256 = _required_hash(payload, "records_sha256")
    actual_records_sha256 = hashlib.sha256(_canonical_json_bytes(records)).hexdigest()
    if records_sha256 != actual_records_sha256:
        raise PersonaRPPInputContractError(
            "records_sha256_mismatch",
            "records_sha256 does not match canonical records",
            {"expected": records_sha256, "actual": actual_records_sha256},
        )

    clean_payload: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "peer_id": peer_id,
        "app_id": app_id,
        "base_model_id": base_model_id,
        "source_kind": source_kind,
        "created_at": float(created_at),
        "records": records,
        "records_sha256": records_sha256,
    }
    input_note = payload.get("input_note")
    if input_note is not None:
        if not isinstance(input_note, str):
            raise PersonaRPPInputContractError(
                "invalid_input_note",
                "input_note must be a string when provided",
                {"type": type(input_note).__name__},
            )
        note = input_note.strip()
        if note:
            clean_payload["input_note"] = note
    return clean_payload


def persona_rpp_input_contract_sha256(payload: dict[str, Any]) -> str:
    """Return the canonical SHA256 for a validated contract payload."""

    clean_payload = validate_persona_rpp_input_contract(payload)
    return hashlib.sha256(_canonical_json_bytes(clean_payload)).hexdigest()


def records_sha256(records: list[dict[str, Any]]) -> str:
    """Return the records hash producers must place in the contract."""

    clean_records = _validate_records(records)
    return hashlib.sha256(_canonical_json_bytes(clean_records)).hexdigest()


def _validate_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PersonaRPPInputContractError(
            "invalid_records",
            "records must be a non-empty list",
            {"type": type(value).__name__},
        )

    clean_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise PersonaRPPInputContractError(
                "invalid_record",
                "each record must be an object",
                {"index": index, "type": type(record).__name__},
            )
        unknown = sorted(set(record) - RECORD_FIELDS)
        if unknown:
            raise PersonaRPPInputContractError(
                "unknown_record_fields",
                "record contains fields outside the canonical contract",
                {"index": index, "fields": unknown},
            )

        record_id = _required_id(record, "record_id")
        if record_id in seen_ids:
            raise PersonaRPPInputContractError(
                "duplicate_record_id",
                "record_id must be unique within a contract",
                {"record_id": record_id},
            )
        seen_ids.add(record_id)

        text_value = record.get("text")
        if not isinstance(text_value, str) or not text_value.strip():
            raise PersonaRPPInputContractError(
                "invalid_record_text",
                "record.text must be a non-empty string",
                {"index": index, "type": type(text_value).__name__},
            )
        clean_record: dict[str, Any] = {
            "record_id": record_id,
            "text": text_value.strip(),
        }

        kind = _optional_id(record.get("kind"), "kind")
        if kind:
            clean_record["kind"] = kind
        if "created_at" in record:
            clean_record["created_at"] = float(
                _finite_number(record.get("created_at"), "created_at", index=index)
            )
        if "weight" in record:
            weight = _finite_number(record.get("weight"), "weight", index=index)
            if weight <= 0 or weight > 10:
                raise PersonaRPPInputContractError(
                    "invalid_record_weight",
                    "record.weight must be in the range (0, 10]",
                    {"index": index, "value": weight},
                )
            clean_record["weight"] = float(weight)
        tags = _validate_tags(record.get("tags"), index=index)
        if tags:
            clean_record["tags"] = tags
        metadata = record.get("metadata")
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise PersonaRPPInputContractError(
                    "invalid_record_metadata",
                    "record.metadata must be an object when provided",
                    {"index": index, "type": type(metadata).__name__},
                )
            clean_record["metadata"] = _clean_json_value(
                metadata,
                field="metadata",
                index=index,
            )
        clean_records.append(clean_record)
    return clean_records


def _validate_tags(value: Any, *, index: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PersonaRPPInputContractError(
            "invalid_record_tags",
            "record.tags must be a list of strings",
            {"index": index, "type": type(value).__name__},
        )
    tags: list[str] = []
    for tag_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise PersonaRPPInputContractError(
                "invalid_record_tag",
                "record.tags entries must be non-empty strings",
                {"index": index, "tag_index": tag_index},
            )
        tags.append(item.strip())
    return tags


def _clean_json_value(value: Any, *, field: str, index: int | None = None) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise PersonaRPPInputContractError(
                "invalid_json_number",
                f"{field} contains a non-finite number",
                {"index": index},
            )
        return value
    if isinstance(value, list):
        return [
            _clean_json_value(item, field=field, index=index) for item in value
        ]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or not key.strip():
                raise PersonaRPPInputContractError(
                    "invalid_json_key",
                    f"{field} keys must be non-empty strings",
                    {"index": index},
                )
            clean[key.strip()] = _clean_json_value(child, field=field, index=index)
        return clean
    raise PersonaRPPInputContractError(
        "invalid_json_value",
        f"{field} contains unsupported JSON value",
        {"index": index, "type": type(value).__name__},
    )


def _required_id(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise PersonaRPPInputContractError(
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
        raise PersonaRPPInputContractError(
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
        raise PersonaRPPInputContractError(
            "invalid_sha256",
            "sha256 fields must be 64 lowercase hex characters",
            {"value": text},
        )
    return text


def _finite_number(value: Any, field: str, *, index: int | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PersonaRPPInputContractError(
            f"invalid_{field}",
            f"{field} must be a finite number",
            {"index": index, "type": type(value).__name__},
        )
    return float(value)


def _path_component(value: str, field: str) -> str:
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in value:
        raise PersonaRPPInputContractError(
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
        raise PersonaRPPInputContractError(
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
