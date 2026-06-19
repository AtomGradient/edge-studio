# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Inspection helpers for locally stored RPP artifact runs.

The inspector is a host-local visualization surface. It summarizes RPP outputs
and artifact headers without returning raw model responses or loading tensor
payloads.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.services.rpp_artifact_store import (
    StoredRPPArtifactRun,
    latest_rpp_artifact_run_for_peer,
)


INSPECTION_SCHEMA_VERSION = "edgestudio.rpp_artifact_inspection.v0"
MAX_JSON_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024


@dataclass
class RPPArtifactInspectionError(ValueError):
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


def inspect_latest_rpp_artifact_run_for_peer(
    peer_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return a UI-ready inspection payload for the latest RPP run."""

    run = latest_rpp_artifact_run_for_peer(peer_id, root=root)
    if run is None:
        return {
            "ok": True,
            "schema_version": INSPECTION_SCHEMA_VERSION,
            "status": "missing",
            "peer_id": peer_id,
            "rpp_run_id": None,
            "received_at_ms": None,
            "storage_path": None,
            "summary": {},
            "dataset_summary": {},
            "profile": {},
            "directions": [],
            "artifacts": [],
            "warnings": ["missing_rpp_artifact_run"],
        }
    return inspect_rpp_artifact_run(run)


def inspect_rpp_artifact_run(run: StoredRPPArtifactRun) -> dict[str, Any]:
    """Inspect a stored RPP run without loading safetensors tensor payloads."""

    receipt = run.receipt if isinstance(run.receipt, dict) else {}
    warnings: list[str] = []
    artifact_infos = [_artifact_info(artifact) for artifact in _receipt_artifacts(receipt)]

    last_run = _read_json_artifact(run, role="rpp_last_run", warnings=warnings)
    if last_run is not None and not isinstance(last_run, dict):
        raise RPPArtifactInspectionError(
            "invalid_rpp_last_run",
            "Stored rpp_last_run.json must be a JSON object.",
            {"peer_id": run.peer_id, "rpp_run_id": run.rpp_run_id},
        )
    rpp_output = last_run or {}

    naming_payload = _read_json_artifact(run, role="rpp_b_naming", warnings=warnings)
    naming_entries = _normalize_naming_entries(naming_payload)
    b_directions_header = _read_safetensors_artifact_header(
        run,
        role="rpp_b_directions",
        warnings=warnings,
    )

    dataset_summary = _object_or_empty(
        _first_present(rpp_output, ("dataset_summary",)),
        _first_present(receipt, ("dataset_summary",)),
    )
    summary = _summary(run=run, receipt=receipt, rpp_output=rpp_output, dataset_summary=dataset_summary)
    profile = _profile(rpp_output=rpp_output, naming_entries=naming_entries)
    directions = _directions(
        rpp_output=rpp_output,
        naming_entries=naming_entries,
    )

    if not directions:
        warnings.append("no_rpp_directions_found")
    if not profile.get("narrative"):
        warnings.append("missing_profile_narrative")

    return {
        "ok": True,
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "status": "found",
        "peer_id": run.peer_id,
        "rpp_run_id": run.rpp_run_id,
        "received_at_ms": run.received_at_ms,
        "storage_path": str(run.path),
        "summary": summary,
        "dataset_summary": dataset_summary,
        "profile": profile,
        "directions": directions,
        "b_directions_header": b_directions_header,
        "artifacts": artifact_infos,
        "warnings": sorted(set(warnings)),
        "audit": {
            "raw_response_returned": False,
            "safetensors_payload_loaded": False,
            "artifact_count": len(artifact_infos),
        },
    }


def _summary(
    *,
    run: StoredRPPArtifactRun,
    receipt: dict[str, Any],
    rpp_output: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    layer_id = _first_present(rpp_output, ("target_layer", "layer_id", "layer_idx"))
    if layer_id is None:
        layer_id = receipt.get("layer_id")
    direction_count = len(_list_of_dicts(rpp_output.get("directions")))
    return {
        "peer_id": run.peer_id,
        "rpp_run_id": run.rpp_run_id,
        "base_model_id": _first_present(
            rpp_output,
            ("base_model_id", "model_id", "model_name"),
        ) or receipt.get("base_model_id"),
        "layer_id": _coerce_int(layer_id),
        "a_version": _first_present(rpp_output, ("a_version",)) or receipt.get("a_version"),
        "a_hash": _first_present(rpp_output, ("a_hash",)) or receipt.get("a_hash"),
        "n_transactions": _coerce_int(
            _first_present(
                rpp_output,
                ("n_transactions", "event_count", "sample_count"),
            )
            or _first_present(
                dataset_summary,
                ("n_transactions", "total_count", "event_count", "sample_count"),
            )
        ),
        "k_selected": _coerce_int(
            _first_present(
                rpp_output,
                ("k_selected", "k_selected_after_fallback", "selected_count"),
            )
            or dataset_summary.get("k_selected")
        ),
        "direction_count": direction_count,
        "total_elapsed_seconds": _coerce_float(
            _first_present(
                rpp_output,
                ("total_elapsed_seconds", "elapsed_seconds", "duration_seconds"),
            )
        ),
    }


def _profile(
    *,
    rpp_output: dict[str, Any],
    naming_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    first_naming = naming_entries[0] if naming_entries else {}
    return {
        "name": _coerce_str(
            _first_present(rpp_output, ("profile_name",))
            or first_naming.get("profile_name")
        ),
        "summary": _coerce_str(
            _first_present(rpp_output, ("profile_summary",))
            or first_naming.get("profile_summary")
        ),
        "narrative": _coerce_str(
            _first_present(
                rpp_output,
                ("profile_narrative", "profile_body", "narrative"),
            )
            or first_naming.get("profile_summary")
        ),
    }


def _directions(
    *,
    rpp_output: dict[str, Any],
    naming_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_directions = _list_of_dicts(rpp_output.get("directions"))
    bootstrap_verdicts = _list_of_dicts(
        _first_present(rpp_output, ("bootstrap_verdicts",))
        or _nested_get(rpp_output, ("bootstrap", "verdict"))
    )
    by_idx, by_id = _naming_maps(naming_entries)

    if not raw_directions and naming_entries:
        raw_directions = naming_entries

    normalized: list[dict[str, Any]] = []
    for index, direction in enumerate(raw_directions):
        direction_idx = _coerce_int(
            _first_present(direction, ("direction_idx", "component_idx", "idx", "index"))
        )
        if direction_idx is None:
            direction_idx = index + 1
        direction_id = _coerce_str(
            _first_present(direction, ("direction_id", "direction_key", "id"))
        ) or f"u_{direction_idx}"
        naming = by_id.get(direction_id) or by_idx.get(direction_idx) or {}
        bootstrap = _bootstrap_for_direction(
            bootstrap_verdicts,
            index=index,
            direction_idx=direction_idx,
        )
        projection_stats = _projection_stats(direction.get("projection_stats"))
        normalized.append(
            {
                "direction_idx": direction_idx,
                "direction_id": direction_id,
                "name": _coerce_str(
                    naming.get("name")
                    or _first_present(direction, ("name", "llm_name", "label"))
                )
                or direction_id,
                "reason": _coerce_str(
                    naming.get("reason")
                    or _first_present(direction, ("reason", "llm_reason", "description"))
                ),
                "confidence": _coerce_float(naming.get("confidence")),
                "bootstrap_pass": _coerce_bool(
                    _first_present(bootstrap, ("pass", "passed", "verdict"))
                ),
                "mean_similarity": _coerce_float(
                    _first_present(bootstrap, ("mean_similarity", "mean_sim"))
                ),
                "std_similarity": _coerce_float(
                    _first_present(bootstrap, ("std_similarity", "std_sim"))
                ),
                "projection_stats": projection_stats,
                "top_positive_count": len(_list_of_dicts(direction.get("top_positive"))),
                "top_negative_count": len(_list_of_dicts(direction.get("top_negative"))),
            }
        )
    return normalized


def _read_json_artifact(
    run: StoredRPPArtifactRun,
    *,
    role: str,
    warnings: list[str],
) -> Any | None:
    artifact = _stored_artifact_for_role(run, role)
    if artifact is None:
        warnings.append(f"missing_{role}")
        return None
    path = _artifact_path(run, artifact)
    if path is None:
        warnings.append(f"invalid_{role}_path")
        return None
    size = path.stat().st_size
    if size > MAX_JSON_ARTIFACT_BYTES:
        raise RPPArtifactInspectionError(
            "json_artifact_too_large",
            f"{path.name} is too large to inspect.",
            {"name": path.name, "size_bytes": size},
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RPPArtifactInspectionError(
            "invalid_json_artifact",
            f"{path.name} is not valid JSON.",
            {"name": path.name},
        ) from exc


def _read_safetensors_artifact_header(
    run: StoredRPPArtifactRun,
    *,
    role: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    artifact = _stored_artifact_for_role(run, role)
    if artifact is None:
        warnings.append(f"missing_{role}")
        return None
    path = _artifact_path(run, artifact)
    if path is None:
        warnings.append(f"invalid_{role}_path")
        return None
    with path.open("rb") as handle:
        first = handle.read(8)
        if len(first) != 8:
            raise RPPArtifactInspectionError(
                "invalid_safetensors_artifact",
                f"{path.name} is too small to be a safetensors file.",
                {"name": path.name},
            )
        header_len = struct.unpack("<Q", first)[0]
        if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise RPPArtifactInspectionError(
                "invalid_safetensors_header",
                f"{path.name} has invalid safetensors header length.",
                {"name": path.name, "header_size_bytes": header_len},
            )
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise RPPArtifactInspectionError(
                "truncated_safetensors_header",
                f"{path.name} ended before the safetensors header completed.",
                {"name": path.name},
            )
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RPPArtifactInspectionError(
            "invalid_safetensors_header_json",
            f"{path.name} has invalid safetensors header JSON.",
            {"name": path.name},
        ) from exc
    if not isinstance(header, dict):
        raise RPPArtifactInspectionError(
            "invalid_safetensors_header",
            f"{path.name} safetensors header must be a JSON object.",
            {"name": path.name},
        )
    tensors = []
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        data_offsets = [int(v) for v in offsets] if isinstance(offsets, list) else []
        byte_count = None
        if len(data_offsets) == 2:
            byte_count = max(0, data_offsets[1] - data_offsets[0])
        shape = meta.get("shape")
        tensors.append(
            {
                "name": name,
                "dtype": _coerce_str(meta.get("dtype")),
                "shape": [int(v) for v in shape] if isinstance(shape, list) else [],
                "byte_count": byte_count,
            }
        )
    return {
        "name": path.name,
        "header_size_bytes": header_len,
        "tensor_count": len(tensors),
        "metadata": _object_or_empty(header.get("__metadata__")),
        "tensors": sorted(tensors, key=lambda item: item["name"]),
    }


def _stored_artifact_for_role(
    run: StoredRPPArtifactRun,
    role: str,
) -> dict[str, Any] | None:
    for artifact in run.stored_artifacts:
        if artifact.get("role") == role and artifact.get("stored") is True:
            return artifact
    return None


def _artifact_path(run: StoredRPPArtifactRun, artifact: dict[str, Any]) -> Path | None:
    name = _coerce_str(artifact.get("name"))
    raw_path = _coerce_str(artifact.get("path"))
    if not name or not raw_path:
        return None
    path = Path(raw_path).expanduser().resolve()
    if path.name != name or not path.is_file():
        return None
    try:
        path.relative_to(run.path.resolve())
    except ValueError:
        return None
    return path


def _artifact_info(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _coerce_str(artifact.get("name")),
        "role": _coerce_str(artifact.get("role")),
        "stored": artifact.get("stored") is True,
        "size_bytes": _coerce_int(artifact.get("size_bytes")),
        "sha256": _coerce_str(artifact.get("sha256")),
    }


def _receipt_artifacts(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = receipt.get("artifacts")
    return _list_of_dicts(artifacts)


def _normalize_naming_entries(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _list_of_dicts(payload)
    if isinstance(payload, dict):
        entries = payload.get("directions")
        if isinstance(entries, list):
            return _list_of_dicts(entries)
        return [payload]
    return []


def _naming_maps(
    naming_entries: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_idx: dict[int, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for entry in naming_entries:
        idx = _coerce_int(entry.get("direction_idx"))
        direction_id = _coerce_str(entry.get("direction_id"))
        if idx is not None:
            by_idx[idx] = entry
        if direction_id:
            by_id[direction_id] = entry
    return by_idx, by_id


def _bootstrap_for_direction(
    verdicts: list[dict[str, Any]],
    *,
    index: int,
    direction_idx: int,
) -> dict[str, Any]:
    for verdict in verdicts:
        component_idx = _coerce_int(verdict.get("component_idx"))
        if component_idx is not None and (
            component_idx == index or component_idx == direction_idx
        ):
            return verdict
    if index < len(verdicts):
        return verdicts[index]
    return {}


def _projection_stats(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "min",
        "max",
        "mean",
        "std",
        "median",
        "p05",
        "p95",
        "positive_count",
        "negative_count",
    }
    return {
        str(key): child
        for key, child in value.items()
        if str(key) in allowed and isinstance(child, (int, float, str, bool))
    }


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return None


def _nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, dict):
            return None
        cursor = cursor.get(key)
    return cursor


def _object_or_empty(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "pass", "passed", "yes", "1"}:
            return True
        if lowered in {"false", "fail", "failed", "no", "0"}:
            return False
    return None
