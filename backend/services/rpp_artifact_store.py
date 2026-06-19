# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Local store for device-originated RPP artifact uploads.

RPP artifacts are user-derived personalization signals. This module only
persists uploads on the user's EdgeStudio host and returns an auditable receipt;
it does not trigger training or make routing/profile decisions.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_dirs import data_path


UPLOAD_SCHEMA_VERSION = "edgestudio.rpp_artifact_upload.v0"
RECEIPT_SCHEMA_VERSION = "edgestudio.rpp_artifact_receipt.v0"

ALLOWED_ARTIFACT_ROLES = {
    "rpp_last_run",
    "rpp_b_directions",
    "rpp_b_naming",
    "rpp_directions_a",
    "rpp_projection_eval",
}


@dataclass
class RPPArtifactUploadError(ValueError):
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
class StoredRPPArtifactRun:
    peer_id: str
    rpp_run_id: str
    path: Path
    received_at_ms: int
    receipt: dict[str, Any]
    stored_artifacts: tuple[dict[str, Any], ...]


def default_rpp_artifact_root() -> Path:
    configured = os.environ.get("EDGE_RPP_ARTIFACT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("persona", "rpp_artifacts")


def store_rpp_artifact_upload(
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate and persist one device RPP artifact upload."""

    if not isinstance(payload, dict):
        raise RPPArtifactUploadError(
            "invalid_input",
            "payload must be a JSON object",
            {"type": type(payload).__name__},
        )

    schema_version = str(payload.get("schema_version") or UPLOAD_SCHEMA_VERSION)
    if schema_version != UPLOAD_SCHEMA_VERSION:
        raise RPPArtifactUploadError(
            "unsupported_schema_version",
            f"unsupported schema_version: {schema_version}",
            {"expected": UPLOAD_SCHEMA_VERSION},
        )

    peer_id = _required_id(payload, "peer_id")
    rpp_run_id = _required_id(payload, "rpp_run_id")
    artifact_specs = _optional_list(payload.get("artifacts"), "artifacts")
    rpp_last_run = payload.get("rpp_last_run")
    if rpp_last_run is not None and not isinstance(rpp_last_run, dict):
        raise RPPArtifactUploadError(
            "invalid_rpp_last_run",
            "rpp_last_run must be an object when provided",
            {"type": type(rpp_last_run).__name__},
        )
    if rpp_last_run is not None:
        embedded_run_id = str(rpp_last_run.get("rpp_run_id") or "").strip()
        if embedded_run_id and embedded_run_id != rpp_run_id:
            raise RPPArtifactUploadError(
                "rpp_run_id_mismatch",
                "rpp_last_run.rpp_run_id does not match top-level rpp_run_id",
                {"expected": rpp_run_id, "actual": embedded_run_id},
            )

    if rpp_last_run is None and not artifact_specs:
        raise RPPArtifactUploadError(
            "empty_artifact_upload",
            "upload must include rpp_last_run or at least one artifact",
            {},
        )

    base = (root or default_rpp_artifact_root()).expanduser().resolve()
    run_dir = base / _path_component(peer_id, "peer_id") / _path_component(
        rpp_run_id, "rpp_run_id"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    existing_receipt = _existing_receipt_for_run(
        run_dir=run_dir,
        peer_id=peer_id,
        rpp_run_id=rpp_run_id,
    )

    received_at_ms = int(time.time() * 1000)
    stored_files: list[dict[str, Any]] = []

    if rpp_last_run is not None:
        data = json.dumps(rpp_last_run, ensure_ascii=False, indent=2, sort_keys=True)
        last_run_path = run_dir / "rpp_last_run.json"
        last_run_path.write_text(data, encoding="utf-8")
        encoded = data.encode("utf-8")
        stored_files.append(
            _file_receipt(
                name="rpp_last_run.json",
                role="rpp_last_run",
                path=last_run_path,
                data=encoded,
                source="rpp_last_run",
            )
        )

    metadata_only = 0
    for raw_spec in artifact_specs:
        if not isinstance(raw_spec, dict):
            raise RPPArtifactUploadError(
                "invalid_artifact",
                "each artifact must be an object",
                {"type": type(raw_spec).__name__},
            )
        spec = dict(raw_spec)
        name = _safe_artifact_name(spec.get("name"))
        role = str(spec.get("role") or "").strip()
        if role not in ALLOWED_ARTIFACT_ROLES:
            raise RPPArtifactUploadError(
                "invalid_artifact_role",
                f"unsupported artifact role: {role}",
                {"allowed_roles": sorted(ALLOWED_ARTIFACT_ROLES), "name": name},
            )

        content_b64 = spec.get("content_base64")
        if content_b64 is None:
            metadata_only += 1
            stored_files.append(
                _metadata_only_receipt(name=name, role=role, spec=spec)
            )
            continue

        data = _decode_base64(content_b64, name)
        expected_size = spec.get("size_bytes")
        if expected_size is not None and int(expected_size) != len(data):
            raise RPPArtifactUploadError(
                "artifact_size_mismatch",
                "artifact size_bytes does not match decoded content",
                {"name": name, "expected": int(expected_size), "actual": len(data)},
            )

        sha = hashlib.sha256(data).hexdigest()
        expected_sha = str(spec.get("sha256") or "").strip()
        if expected_sha and expected_sha != sha:
            raise RPPArtifactUploadError(
                "artifact_sha256_mismatch",
                "artifact sha256 does not match decoded content",
                {"name": name, "expected": expected_sha, "actual": sha},
            )

        artifact_path = run_dir / name
        artifact_path.write_bytes(data)
        stored_files.append(
            _file_receipt(
                name=name,
                role=role,
                path=artifact_path,
                data=data,
                source="content_base64",
            )
        )

    artifacts = _merge_artifact_receipts(
        existing_receipt.get("artifacts") if existing_receipt else None,
        stored_files,
        run_dir=run_dir,
    )
    receipt = {
        "ok": True,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": peer_id,
        "rpp_run_id": rpp_run_id,
        "storage_path": str(run_dir),
        "received_at_ms": received_at_ms,
        "base_model_id": _optional_str(payload.get("base_model_id"))
        or _existing_optional_str(existing_receipt, "base_model_id"),
        "layer_id": payload.get("layer_id")
        if payload.get("layer_id") is not None
        else _existing_optional_value(existing_receipt, "layer_id"),
        "a_version": _optional_str(payload.get("a_version"))
        or _existing_optional_str(existing_receipt, "a_version"),
        "a_hash": _optional_str(payload.get("a_hash"))
        or _existing_optional_str(existing_receipt, "a_hash"),
        "dataset_summary": payload.get("dataset_summary")
        or _existing_optional_value(existing_receipt, "dataset_summary")
        or {},
        "artifacts": artifacts,
        "audit": {
            "schema_version": "edgestudio.rpp_artifact_audit.v0",
            "artifact_count": len(artifacts),
            "metadata_only_artifact_count": sum(
                1 for artifact in artifacts if artifact.get("stored") is not True
            ),
            "new_artifact_count": len(stored_files),
            "merged_existing_artifact_count": max(
                0,
                len(artifacts) - len(stored_files),
            ),
            "transport": "https_json_v0",
            "trust_boundary": "local_edgestudio_host_pending_peer_auth",
        },
    }

    manifest_path = run_dir / "upload_receipt.json"
    manifest_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def _existing_receipt_for_run(
    *,
    run_dir: Path,
    peer_id: str,
    rpp_run_id: str,
) -> dict[str, Any] | None:
    receipt_path = run_dir / "upload_receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(receipt, dict):
        return None
    if receipt.get("peer_id") != peer_id or receipt.get("rpp_run_id") != rpp_run_id:
        return None
    return receipt


def _merge_artifact_receipts(
    existing_artifacts: Any,
    new_artifacts: list[dict[str, Any]],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    if isinstance(existing_artifacts, list):
        for artifact in existing_artifacts:
            if not isinstance(artifact, dict):
                continue
            name = _optional_str(artifact.get("name"))
            if not name or not _artifact_receipt_is_current(artifact, run_dir):
                continue
            merged[name] = dict(artifact)

    for artifact in new_artifacts:
        name = _optional_str(artifact.get("name"))
        if name:
            merged[name] = artifact

    return [merged[name] for name in sorted(merged)]


def _artifact_receipt_is_current(artifact: dict[str, Any], run_dir: Path) -> bool:
    if artifact.get("stored") is not True:
        return True
    name = _optional_str(artifact.get("name"))
    path = _optional_str(artifact.get("path"))
    if not name or not path:
        return False
    artifact_path = Path(path).expanduser().resolve()
    if artifact_path.name != name or not artifact_path.is_file():
        return False
    try:
        artifact_path.relative_to(run_dir.resolve())
    except ValueError:
        return False
    return True


def _existing_optional_value(receipt: dict[str, Any] | None, key: str) -> Any | None:
    if not isinstance(receipt, dict):
        return None
    return receipt.get(key)


def _existing_optional_str(receipt: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(receipt, dict):
        return None
    return _optional_str(receipt.get(key))


def latest_rpp_artifact_run_for_peer(
    peer_id: str,
    *,
    root: Path | None = None,
) -> StoredRPPArtifactRun | None:
    """Return the newest stored RPP run for a peer, if any."""

    safe_peer_id = _path_component(str(peer_id).strip(), "peer_id")
    base = (root or default_rpp_artifact_root()).expanduser().resolve()
    peer_dir = base / safe_peer_id
    if not peer_dir.is_dir():
        return None

    candidates: list[StoredRPPArtifactRun] = []
    for receipt_path in peer_dir.glob("*/upload_receipt.json"):
        parsed = _read_stored_run(receipt_path)
        if parsed is not None and parsed.peer_id == safe_peer_id:
            candidates.append(parsed)

    if not candidates:
        return None
    return max(candidates, key=lambda run: (run.received_at_ms, run.rpp_run_id))


def attach_latest_rpp_artifacts_to_adapter(
    *,
    peer_id: str,
    adapter_dir: Path,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Copy the latest stored RPP run's files into an adapter package directory.

    The AdapterDistributor already treats these filenames as optional package
    members and will include them in the distribution manifest. We clear known
    RPP files first so a reused adapter output directory cannot accidentally
    ship stale artifacts.
    """

    adapter_dir = adapter_dir.expanduser().resolve()
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cleared = _clear_adapter_rpp_files(adapter_dir)

    run = latest_rpp_artifact_run_for_peer(peer_id, root=root)
    if run is None:
        return None

    copied: list[dict[str, Any]] = []
    for artifact in run.stored_artifacts:
        if artifact.get("stored") is not True:
            continue
        name = _safe_artifact_name(artifact.get("name"))
        source_path = Path(str(artifact.get("path") or "")).expanduser().resolve()
        if source_path.name != name or not source_path.is_file():
            continue
        try:
            source_path.relative_to(run.path.resolve())
        except ValueError:
            continue

        target_path = adapter_dir / name
        shutil.copy2(source_path, target_path)
        data = target_path.read_bytes()
        copied.append(
            _file_receipt(
                name=name,
                role=str(artifact.get("role") or ""),
                path=target_path,
                data=data,
                source="attached_from_rpp_artifact_store",
            )
        )

    if not copied:
        return None

    return {
        "rpp_run_id": run.rpp_run_id,
        "a_version": _optional_str(run.receipt.get("a_version")),
        "a_hash": _optional_str(run.receipt.get("a_hash")),
        "source_path": str(run.path),
        "copied_artifacts": copied,
        "cleared_stale_artifacts": cleared,
    }


def _read_stored_run(receipt_path: Path) -> StoredRPPArtifactRun | None:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        peer_id = str(receipt.get("peer_id") or "").strip()
        rpp_run_id = str(receipt.get("rpp_run_id") or "").strip()
        if not peer_id or not rpp_run_id:
            return None
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, list):
            return None
        stored_artifacts = tuple(
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict) and artifact.get("stored") is True
        )
        return StoredRPPArtifactRun(
            peer_id=peer_id,
            rpp_run_id=rpp_run_id,
            path=receipt_path.parent.resolve(),
            received_at_ms=int(receipt.get("received_at_ms") or 0),
            receipt=receipt,
            stored_artifacts=stored_artifacts,
        )
    except Exception:  # noqa: BLE001
        return None


def _clear_adapter_rpp_files(adapter_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in ("rpp_last_run.json",):
        path = adapter_dir / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    for pattern in (
        "B_directions_layer_*.safetensors",
        "B_naming_layer_*.json",
        "directions_a_layer_*.safetensors",
    ):
        for path in adapter_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    return sorted(removed)


def _required_id(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise RPPArtifactUploadError(
            "missing_required_field",
            f"{key} is required",
            {"field": key},
        )
    return value


def _path_component(value: str, field: str) -> str:
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in value:
        raise RPPArtifactUploadError(
            "invalid_path_component",
            f"{field} must not contain path separators or '..'",
            {"field": field, "value": value},
        )
    return value


def _safe_artifact_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        raise RPPArtifactUploadError(
            "invalid_artifact_name",
            "artifact name is required",
            {},
        )
    if Path(name).name != name or "/" in name or "\\" in name or ".." in name:
        raise RPPArtifactUploadError(
            "invalid_artifact_name",
            "artifact name must be a plain file name",
            {"name": name},
        )
    return name


def _optional_list(value: Any, field: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RPPArtifactUploadError(
            "invalid_input",
            f"{field} must be a list when provided",
            {"field": field, "type": type(value).__name__},
        )
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _decode_base64(value: Any, name: str) -> bytes:
    if not isinstance(value, str):
        raise RPPArtifactUploadError(
            "invalid_artifact_content",
            "content_base64 must be a string",
            {"name": name, "type": type(value).__name__},
        )
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise RPPArtifactUploadError(
            "invalid_artifact_content",
            "content_base64 is not valid base64",
            {"name": name},
        ) from exc


def _file_receipt(
    *,
    name: str,
    role: str,
    path: Path,
    data: bytes,
    source: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "path": str(path),
        "stored": True,
        "source": source,
    }


def _metadata_only_receipt(
    *,
    name: str,
    role: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "role": role,
        "size_bytes": spec.get("size_bytes"),
        "sha256": spec.get("sha256"),
        "path": None,
        "stored": False,
        "source": "metadata_only",
    }
