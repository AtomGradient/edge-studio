# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host bridge from canonical Persona/RPP input to Persona source."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

from .neural_imprint_generation import NeuralImprintGenerationError, _fact_tool_schema_export
from .persona_rpp_input_contract import latest_persona_rpp_input_for_peer
from .persona_source_store import (
    SOURCE_SCHEMA_VERSION,
    latest_persona_source_for_peer,
    store_persona_source_upload,
)

PROCESSOR_RECEIPT_SCHEMA_VERSION = "edgestudio.host_rpp_processor_receipt.v1"


@dataclass
class HostRPPProcessorError(ValueError):
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


def process_canonical_rpp_input_to_persona_source(
    *,
    peer_id: str,
    tool_schema_export: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    base_model_id: str | None = None,
    profile_body_suffix: str | None = None,
    lineage_extra: Mapping[str, Any] | None = None,
    input_root: Path | None = None,
    source_root: Path | None = None,
    created_at: float | None = None,
) -> dict[str, Any]:
    source = latest_persona_rpp_input_for_peer(peer_id, root=input_root)
    if source is None:
        raise HostRPPProcessorError(
            "persona_rpp_input_not_found",
            f"persona RPP input for peer {peer_id} not found",
            {"peer_id": peer_id},
        )

    payload = source.payload
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise HostRPPProcessorError(
            "missing_records",
            "canonical input records must be a non-empty list",
            {"peer_id": peer_id},
        )

    profile_body = build_profile_body(records)
    if profile_body_suffix and profile_body_suffix.strip():
        profile_body = profile_body.rstrip() + "\n\n" + profile_body_suffix.strip() + "\n"
    profile_body_sha256 = _sha256_text(profile_body)
    try:
        tool_schema = _fact_tool_schema_export(_normalize_tool_schema(tool_schema_export))
    except NeuralImprintGenerationError as exc:
        raise HostRPPProcessorError(
            exc.code,
            exc.message,
            exc.details,
        ) from exc
    tool_schema_sha256 = _sha256_json(tool_schema)
    source_payload = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "peer_id": source.peer_id,
        "app_id": str(payload.get("app_id") or ""),
        "base_model_id": str(base_model_id or payload.get("base_model_id") or ""),
        "source_kind": "host_rpp_profile",
        "tool_schema_export": tool_schema,
        "tool_schema_sha256": tool_schema_sha256,
        "profile_body": profile_body,
        "profile_body_sha256": profile_body_sha256,
        "created_at": _created_at(created_at),
        "lineage": _lineage_from_receipt(source.receipt, lineage_extra=lineage_extra),
    }
    persona_source_receipt = store_persona_source_upload(
        source_payload,
        root=source_root,
    )
    stored = latest_persona_source_for_peer(source.peer_id, root=source_root)

    return {
        "ok": True,
        "schema_version": PROCESSOR_RECEIPT_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": source.peer_id,
        "source_id": persona_source_receipt["source_id"],
        "source_kind": persona_source_receipt["source_kind"],
        "profile_body_sha256": profile_body_sha256,
        "tool_schema_sha256": tool_schema_sha256,
        "lineage": source_payload["lineage"],
        "persona_source_receipt": persona_source_receipt,
        "persona_source": stored.payload if stored else None,
    }


def load_tool_schema_export_from_model_dir(model_dir: str | Path) -> dict[str, Any]:
    root = Path(str(model_dir)).expanduser().resolve()
    if not root.exists():
        raise HostRPPProcessorError(
            "model_dir_not_found",
            f"model_dir not found: {root}",
            {"model_dir": str(root)},
        )
    if not root.is_dir():
        raise HostRPPProcessorError(
            "model_dir_not_directory",
            f"model_dir is not a directory: {root}",
            {"model_dir": str(root)},
        )

    candidates = [
        root / "tool_specs.json",
        root / "neural_imprint" / "tool_specs.json",
    ]
    for path in candidates:
        if path.is_file():
            return _normalize_tool_schema(_read_json(path))

    raise HostRPPProcessorError(
        "tool_schema_not_found",
        "tool schema not found in model_dir",
        {"candidates": [str(path) for path in candidates]},
    )


def build_profile_body(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise HostRPPProcessorError(
                "invalid_record",
                "canonical input records must contain objects",
                {"index": index, "type": type(record).__name__},
            )
        record_id = str(record.get("record_id") or "").strip()
        text = str(record.get("text") or "").strip()
        if not record_id or not text:
            raise HostRPPProcessorError(
                "invalid_record",
                "record_id and text are required",
                {"index": index},
            )
        kind = str(record.get("kind") or "record").strip() or "record"
        normalized.append({"kind": kind, "record_id": record_id, "text": text})

    lines: list[str] = []
    ordered = sorted(normalized, key=lambda item: (item["kind"], item["record_id"]))
    for kind, group in groupby(ordered, key=lambda item: item["kind"]):
        if lines:
            lines.append("")
        lines.append(f"[{kind}]")
        lines.extend(item["text"] for item in group)
    return "\n".join(lines).strip() + "\n"


def _normalize_tool_schema(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        tools = value.get("tools")
        schema_version = str(
            value.get("schema_version") or "edgestudio.tool_schema_export.v1"
        )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        tools = list(value)
        schema_version = "edgestudio.tool_schema_export.v1"
    else:
        raise HostRPPProcessorError(
            "invalid_tool_schema_export",
            "tool schema export must be an object or a list",
            {"type": type(value).__name__},
        )

    if (
        not isinstance(tools, list)
        or not tools
        or not all(isinstance(item, Mapping) for item in tools)
    ):
        raise HostRPPProcessorError(
            "invalid_tool_schema_export",
            "tool schema export must contain a non-empty tools list",
            {},
        )
    return {
        "schema_version": schema_version,
        "tools": [dict(item) for item in tools],
    }


def _lineage_from_receipt(
    receipt: Mapping[str, Any],
    *,
    lineage_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    lineage = {
        "input_id": str(receipt.get("input_id") or ""),
        "input_sha256": str(receipt.get("input_sha256") or ""),
        "records_sha256": str(receipt.get("records_sha256") or ""),
        "text_sha256": str(receipt.get("text_sha256") or ""),
        "record_count": int(receipt.get("record_count") or 0),
        "total_text_chars": int(receipt.get("total_text_chars") or 0),
    }
    if lineage_extra:
        lineage.update(dict(lineage_extra))
    return lineage


def _created_at(value: float | None) -> float:
    if value is None:
        return time.time()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise HostRPPProcessorError(
            "invalid_created_at",
            "created_at must be a finite Unix timestamp number",
            {"type": type(value).__name__},
        )
    return float(value)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HostRPPProcessorError(
            "tool_schema_read_failed",
            f"failed to read tool schema: {path}",
            {"path": str(path), "error": str(exc)},
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
