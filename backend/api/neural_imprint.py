# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Neural Imprint artifact inspection endpoints.

The inspector is intentionally header-only: it reads the safetensors JSON
header plus the optional sidecar metadata, never tensor payloads.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.config import BROWSE_ROOTS
from backend.services.neural_imprint_artifact_registry import (
    NeuralImprintArtifactRegistryError,
    find_neural_imprint_artifact,
    list_neural_imprint_artifacts,
)
from backend.services.neural_imprint_generation import (
    NeuralImprintGenerationError,
    enqueue_neural_imprint_generation,
    get_neural_imprint_generation_job,
)
from backend.services.host_rpp_processor import (
    HostRPPProcessorError,
    load_tool_schema_export_from_model_dir,
    process_canonical_rpp_input_to_persona_source,
)
from backend.services.correction_regen_coordinator import (
    CorrectionRegenError,
    regenerate_neural_imprint_from_corrections,
)
from backend.services.neural_imprint_runtime import (
    NeuralImprintRuntimeError,
    get_neural_imprint_status,
    restore_neural_imprint_for_model,
    unload_neural_imprint,
)
from backend.services.persona_source_store import latest_persona_source_for_peer
from backend.services.persona_rpp_input_contract import latest_persona_rpp_input_for_peer

router = APIRouter(prefix="/api/neural_imprint", tags=["neural-imprint"])
neural_imprint_router = router

MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
NEURAL_IMPRINT_METADATA_NAME = "neural_imprint_metadata.json"


class NeuralImprintTensorInfo(BaseModel):
    name: str
    dtype: str | None = None
    shape: list[int] = Field(default_factory=list)
    data_offsets: list[int] = Field(default_factory=list)
    byte_count: int | None = None


class NeuralImprintHashEntry(BaseModel):
    name: str
    value: str
    source: str


class NeuralImprintSummary(BaseModel):
    prefix_token_count: int | None = None
    model_id: str | None = None
    created_at: str | int | float | None = None
    profile_body: Any | None = None
    tool_schema: Any | None = None
    hashes: list[NeuralImprintHashEntry] = Field(default_factory=list)


class NeuralImprintCompatibilityCheck(BaseModel):
    name: str
    expected: str | None = None
    actual: str | None = None
    matched: bool | None = None
    reason: str | None = None


class NeuralImprintCompatibility(BaseModel):
    status: str
    checks: list[NeuralImprintCompatibilityCheck] = Field(default_factory=list)
    message: str


class NeuralImprintInspectResponse(BaseModel):
    ok: bool = True
    artifact_path: str | None = None
    artifact_name: str
    artifact_size_bytes: int | None = None
    header_size_bytes: int
    safetensors_metadata: dict[str, Any] = Field(default_factory=dict)
    tensor_count: int
    tensors: list[NeuralImprintTensorInfo] = Field(default_factory=list)
    sidecar_found: bool
    sidecar_path: str | None = None
    sidecar_metadata: dict[str, Any] = Field(default_factory=dict)
    summary: NeuralImprintSummary
    compatibility: NeuralImprintCompatibility


class NeuralImprintArtifactRegistryResponse(BaseModel):
    ok: bool = True
    schema_version: str
    roots: list[str]
    artifact_count: int
    artifacts: list[dict[str, Any]]


class NeuralImprintArtifactSourceResponse(BaseModel):
    ok: bool = True
    artifact: dict[str, Any]


class PersonaSourceLatestResponse(BaseModel):
    ok: bool = True
    receipt: dict[str, Any]
    payload: dict[str, Any]


class PersonaRPPInputLatestResponse(BaseModel):
    ok: bool = True
    receipt: dict[str, Any]
    payload: dict[str, Any]


class NeuralImprintGenerateRequest(BaseModel):
    peer_id: str = Field(..., min_length=1)
    model_dir: str = Field(..., min_length=1)
    model_id: str | None = None
    validate_restore: bool = False


class PersonaRPPInputProcessRequest(BaseModel):
    peer_id: str = Field(..., min_length=1)
    model_dir: str = Field(..., min_length=1)
    base_model_id: str | None = None


class NeuralImprintCorrectionRegenRequest(BaseModel):
    peer_id: str = Field(..., min_length=1)
    model_dir: str = Field(..., min_length=1)
    model_id: str | None = None
    base_model_id: str | None = None
    validate_restore: bool = False
    include_statuses: list[str] | None = None
    tool_schema_export: dict[str, Any] | None = None


class NeuralImprintGenerateJobResponse(BaseModel):
    ok: bool = True
    job: dict[str, Any]


class PersonaRPPInputProcessResponse(BaseModel):
    ok: bool = True
    receipt: dict[str, Any]


class NeuralImprintCorrectionRegenResponse(BaseModel):
    ok: bool = True
    receipt: dict[str, Any]


class NeuralImprintRestoreRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    artifact_id: str | None = None
    artifact_path: str | None = None
    sidecar_path: str | None = None


class NeuralImprintUnloadRequest(BaseModel):
    model_id: str | None = None


class NeuralImprintRuntimeStatusResponse(BaseModel):
    ok: bool = True
    active: bool
    model_id: str | None = None
    model_dir: str | None = None
    artifact_id: str | None = None
    artifact_path: str | None = None
    sidecar_path: str | None = None
    prefix_token_count: int | None = None
    base_model_id: str | None = None
    model_architecture: str | None = None
    hidden_size: int | None = None
    layer_count: int | None = None
    loaded_at: float | None = None


def _safe_file_path(path: str, *, suffixes: tuple[str, ...] | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise HTTPException(404, "Path not found")
    if not target.is_file():
        raise HTTPException(400, "Path is not a file")
    real = str(target)
    if not any(os.path.commonpath([real, root]) == root for root in BROWSE_ROOTS):
        raise HTTPException(403, "Access denied")
    if suffixes and target.suffix not in suffixes:
        raise HTTPException(400, f"Expected file suffix: {', '.join(suffixes)}")
    return target


def _runtime_status_response(status: Any) -> NeuralImprintRuntimeStatusResponse:
    return NeuralImprintRuntimeStatusResponse(**status.as_dict())


def _resolve_restore_artifact(
    request: NeuralImprintRestoreRequest,
) -> tuple[Path, Path | None, str | None]:
    artifact_id = request.artifact_id.strip() if request.artifact_id else None
    if artifact_id:
        try:
            artifact = find_neural_imprint_artifact(artifact_id)
        except NeuralImprintArtifactRegistryError as exc:
            status = 404 if exc.code == "artifact_not_found" else 400
            raise HTTPException(status_code=status, detail=exc.to_error()) from exc
        artifact_path = str(artifact.get("artifact_path") or "")
        sidecar_path = str(artifact.get("sidecar_path") or "")
        if not artifact_path:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_artifact_record",
                    "message": "Neural Imprint artifact record has no artifact_path",
                    "retryable": False,
                    "details": {"artifact_id": artifact_id},
                },
            )
        return Path(artifact_path).expanduser().resolve(), (
            Path(sidecar_path).expanduser().resolve() if sidecar_path else None
        ), artifact_id

    if not request.artifact_path:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "missing_artifact",
                "message": "artifact_id or artifact_path is required",
                "retryable": False,
                "details": {},
            },
        )
    artifact_path = _safe_file_path(request.artifact_path, suffixes=(".safetensors",))
    sidecar_path = _safe_file_path(request.sidecar_path, suffixes=(".json",)) if request.sidecar_path else None
    return artifact_path, sidecar_path, None


def _read_safetensors_header(stream: BinaryIO, label: str) -> tuple[dict[str, Any], int]:
    first = stream.read(8)
    if len(first) != 8:
        raise HTTPException(400, f"{label} is too small to be a safetensors file")
    header_len = struct.unpack("<Q", first)[0]
    if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise HTTPException(400, f"{label} has invalid safetensors header length")
    raw_header = stream.read(header_len)
    if len(raw_header) != header_len:
        raise HTTPException(400, f"{label} ended before safetensors header completed")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, f"{label} has invalid safetensors header JSON") from exc
    if not isinstance(header, dict):
        raise HTTPException(400, f"{label} safetensors header must be an object")
    return header, header_len


def _parse_safetensors_file(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as handle:
        return _read_safetensors_header(handle, path.name)


def _tensor_infos(header: dict[str, Any]) -> list[NeuralImprintTensorInfo]:
    tensors: list[NeuralImprintTensorInfo] = []
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        data_offsets = [int(v) for v in offsets] if isinstance(offsets, list) else []
        byte_count: int | None = None
        if len(data_offsets) == 2:
            byte_count = max(0, data_offsets[1] - data_offsets[0])
        shape = meta.get("shape")
        tensors.append(NeuralImprintTensorInfo(
            name=name,
            dtype=str(meta.get("dtype")) if meta.get("dtype") is not None else None,
            shape=[int(v) for v in shape] if isinstance(shape, list) else [],
            data_offsets=data_offsets,
            byte_count=byte_count,
        ))
    return sorted(tensors, key=lambda item: item.name)


def _try_read_sidecar(path: Path | None) -> tuple[bool, str | None, dict[str, Any]]:
    if path is None:
        return False, None, {}
    if not path.exists():
        return False, str(path), {}
    if not path.is_file():
        raise HTTPException(400, "Sidecar path is not a file")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Sidecar metadata is not valid JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "Sidecar metadata must be a JSON object")
    return True, str(path), data


def _metadata_from_header(header: dict[str, Any]) -> dict[str, Any]:
    metadata = header.get("__metadata__")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _first_value(*values: Any) -> Any | None:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _lookup(data: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return None


HASH_KEY_RE = re.compile(r"(^|[_\-.])(sha256|hash|fingerprint)([_\-.]|$)|sha256$", re.IGNORECASE)


def _collect_hashes(value: Any, *, source: str, prefix: str = "") -> list[NeuralImprintHashEntry]:
    hashes: list[NeuralImprintHashEntry] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, str) and HASH_KEY_RE.search(str(key)):
                hashes.append(NeuralImprintHashEntry(name=child_prefix, value=child, source=source))
            elif isinstance(child, (dict, list)):
                hashes.extend(_collect_hashes(child, source=source, prefix=child_prefix))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            if isinstance(child, (dict, list)):
                hashes.extend(_collect_hashes(child, source=source, prefix=child_prefix))
    return hashes


def _find_hash(hashes: list[NeuralImprintHashEntry], *candidates: str) -> str | None:
    lowered = [candidate.lower() for candidate in candidates]
    for entry in hashes:
        name = entry.name.lower()
        if any(candidate in name for candidate in lowered):
            return entry.value
    return None


def _normalize_hash(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text.startswith("sha256:"):
        return text.removeprefix("sha256:")
    return text


def _build_summary(header_metadata: dict[str, Any], sidecar: dict[str, Any]) -> NeuralImprintSummary:
    hashes = [
        *_collect_hashes(header_metadata, source="safetensors_header"),
        *_collect_hashes(sidecar, source="sidecar"),
    ]
    return NeuralImprintSummary(
        prefix_token_count=_coerce_int(_first_value(
            _lookup(sidecar, "prefix_token_count", "prefix_tokens", "token_count"),
            _lookup(header_metadata, "prefix_token_count", "prefix_tokens", "token_count"),
        )),
        model_id=_coerce_str(_first_value(
            _lookup(sidecar, "model_id", "base_model_id", "model", "model_name"),
            _lookup(header_metadata, "model_id", "base_model_id", "model", "model_name"),
        )),
        created_at=_first_value(
            _lookup(sidecar, "created_at", "createdAt", "created_at_ms"),
            _lookup(header_metadata, "created_at", "createdAt", "created_at_ms"),
        ),
        profile_body=_first_value(
            _lookup(sidecar, "profile_body", "profileBody", "profile", "profile_narrative"),
            _lookup(header_metadata, "profile_body", "profileBody", "profile", "profile_narrative"),
        ),
        tool_schema=_first_value(
            _lookup(sidecar, "tool_schema", "toolSchema", "tool_schema_snapshot", "tools"),
            _lookup(header_metadata, "tool_schema", "toolSchema", "tool_schema_snapshot", "tools"),
        ),
        hashes=hashes,
    )


def _coerce_int(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _compatibility(
    summary: NeuralImprintSummary,
    *,
    current_model_id: str | None,
    current_model_hash: str | None,
    tokenizer_hash: str | None,
    tool_schema_hash: str | None,
    cache_topology_sha256: str | None,
) -> NeuralImprintCompatibility:
    checks: list[NeuralImprintCompatibilityCheck] = []

    def add_check(name: str, expected: str | None, actual: str | None) -> None:
        if not expected:
            return
        if not actual:
            checks.append(NeuralImprintCompatibilityCheck(
                name=name,
                expected=expected,
                actual=None,
                matched=None,
                reason="artifact field missing",
            ))
            return
        matched = _normalize_hash(expected) == _normalize_hash(actual)
        checks.append(NeuralImprintCompatibilityCheck(
            name=name,
            expected=expected,
            actual=actual,
            matched=matched,
        ))

    add_check("model_id", current_model_id, summary.model_id)
    add_check("model_hash", current_model_hash, _find_hash(
        summary.hashes,
        "model_hash",
        "model_sha256",
        "model_weights_sha256",
        "weights_sha256",
        "base_model_sha256",
    ))
    add_check("tokenizer_hash", tokenizer_hash, _find_hash(
        summary.hashes,
        "tokenizer_hash",
        "tokenizer_sha256",
    ))
    add_check("tool_schema_hash", tool_schema_hash, _find_hash(
        summary.hashes,
        "tool_schema_hash",
        "tool_schema_sha256",
    ))
    add_check("cache_topology_sha256", cache_topology_sha256, _find_hash(
        summary.hashes,
        "cache_topology_sha256",
        "cache_topology_hash",
    ))

    if not checks:
        return NeuralImprintCompatibility(
            status="unknown",
            message="No current model hashes were supplied for comparison.",
        )
    if any(check.matched is False for check in checks):
        return NeuralImprintCompatibility(
            status="incompatible",
            checks=checks,
            message="At least one supplied model/tool hash does not match the artifact.",
        )
    if any(check.matched is None for check in checks):
        return NeuralImprintCompatibility(
            status="unknown",
            checks=checks,
            message="Some requested compatibility fields are missing from the artifact.",
        )
    return NeuralImprintCompatibility(
        status="compatible",
        checks=checks,
        message="All supplied compatibility fields match.",
    )


def inspect_neural_imprint_artifact(
    artifact_path: Path,
    *,
    sidecar_path: Path | None = None,
    current_model_id: str | None = None,
    current_model_hash: str | None = None,
    tokenizer_hash: str | None = None,
    tool_schema_hash: str | None = None,
    cache_topology_sha256: str | None = None,
) -> NeuralImprintInspectResponse:
    header, header_len = _parse_safetensors_file(artifact_path)
    header_metadata = _metadata_from_header(header)
    candidate = sidecar_path if sidecar_path is not None else _default_sidecar_path(artifact_path)
    sidecar_found, resolved_sidecar_path, sidecar = _try_read_sidecar(candidate)
    summary = _build_summary(header_metadata, sidecar)
    tensors = _tensor_infos(header)
    return NeuralImprintInspectResponse(
        artifact_path=str(artifact_path),
        artifact_name=artifact_path.name,
        artifact_size_bytes=artifact_path.stat().st_size,
        header_size_bytes=header_len,
        safetensors_metadata=header_metadata,
        tensor_count=len(tensors),
        tensors=tensors,
        sidecar_found=sidecar_found,
        sidecar_path=resolved_sidecar_path,
        sidecar_metadata=sidecar,
        summary=summary,
        compatibility=_compatibility(
            summary,
            current_model_id=current_model_id,
            current_model_hash=current_model_hash,
            tokenizer_hash=tokenizer_hash,
            tool_schema_hash=tool_schema_hash,
            cache_topology_sha256=cache_topology_sha256,
        ),
    )


def _default_sidecar_path(artifact_path: Path) -> Path:
    return artifact_path.parent / NEURAL_IMPRINT_METADATA_NAME


@router.get("/artifacts", response_model=NeuralImprintArtifactRegistryResponse)
def list_neural_imprint_artifact_sources(
    include_invalid: bool = Query(False),
) -> NeuralImprintArtifactRegistryResponse:
    """List valid local Neural Imprint directories that can be pushed as capsules."""

    registry = list_neural_imprint_artifacts(include_invalid=include_invalid)
    return NeuralImprintArtifactRegistryResponse(**registry)


@router.get("/artifacts/{artifact_id}", response_model=NeuralImprintArtifactSourceResponse)
def get_neural_imprint_artifact_source(artifact_id: str) -> NeuralImprintArtifactSourceResponse:
    """Resolve one local Neural Imprint artifact source by registry id."""

    try:
        artifact = find_neural_imprint_artifact(artifact_id)
    except NeuralImprintArtifactRegistryError as exc:
        status = 404 if exc.code == "artifact_not_found" else 400
        raise HTTPException(status_code=status, detail=exc.to_error()) from exc
    return NeuralImprintArtifactSourceResponse(artifact=artifact)


@router.get("/sources/{peer_id}/latest", response_model=PersonaSourceLatestResponse)
def get_latest_persona_source(peer_id: str) -> PersonaSourceLatestResponse:
    """Return the latest host-stored Persona source upload for one mesh peer."""

    source = latest_persona_source_for_peer(peer_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "persona_source_not_found",
                "message": f"persona source for peer {peer_id} not found",
            },
        )
    return PersonaSourceLatestResponse(receipt=source.receipt, payload=source.payload)


@router.get("/rpp_inputs/{peer_id}/latest", response_model=PersonaRPPInputLatestResponse)
def get_latest_persona_rpp_input(peer_id: str) -> PersonaRPPInputLatestResponse:
    """Return the latest canonical Persona/RPP input contract for one mesh peer."""

    source = latest_persona_rpp_input_for_peer(peer_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "persona_rpp_input_not_found",
                "message": f"persona RPP input for peer {peer_id} not found",
            },
        )
    return PersonaRPPInputLatestResponse(receipt=source.receipt, payload=source.payload)


@router.post("/process_rpp_input", response_model=PersonaRPPInputProcessResponse)
def process_persona_rpp_input(
    request: PersonaRPPInputProcessRequest,
) -> PersonaRPPInputProcessResponse:
    """Convert latest canonical Persona/RPP input into a host Persona source."""

    try:
        tool_schema_export = load_tool_schema_export_from_model_dir(request.model_dir)
        receipt = process_canonical_rpp_input_to_persona_source(
            peer_id=request.peer_id,
            tool_schema_export=tool_schema_export,
            base_model_id=request.base_model_id,
        )
    except HostRPPProcessorError as exc:
        status = 404 if exc.code in {
            "persona_rpp_input_not_found",
            "model_dir_not_found",
            "tool_schema_not_found",
        } else 400
        raise HTTPException(status_code=status, detail=exc.to_error()) from exc
    return PersonaRPPInputProcessResponse(receipt=receipt)


@router.post("/regen_from_corrections", response_model=NeuralImprintCorrectionRegenResponse)
def regenerate_neural_imprint_from_correction_context(
    request: NeuralImprintCorrectionRegenRequest,
) -> NeuralImprintCorrectionRegenResponse:
    """Queue Neural Imprint regeneration from explicit correction ledger context."""

    try:
        receipt = regenerate_neural_imprint_from_corrections(
            peer_id=request.peer_id,
            model_dir=request.model_dir,
            model_id=request.model_id,
            base_model_id=request.base_model_id,
            validate_restore=request.validate_restore,
            include_statuses=request.include_statuses,
            tool_schema_export=request.tool_schema_export,
        )
    except CorrectionRegenError as exc:
        status = 404 if exc.code in {
            "persona_rpp_input_not_found",
            "model_dir_not_found",
            "tool_schema_not_found",
            "persona_source_not_found",
        } else 400
        raise HTTPException(status_code=status, detail=exc.to_error()) from exc
    return NeuralImprintCorrectionRegenResponse(receipt=receipt)


@router.post("/generate", response_model=NeuralImprintGenerateJobResponse)
def generate_neural_imprint(
    request: NeuralImprintGenerateRequest,
) -> NeuralImprintGenerateJobResponse:
    """Queue a Mac-side Neural Imprint generation job.

    The job writes into the local registry root and deliberately does not push
    the artifact to any device.
    """

    try:
        job = enqueue_neural_imprint_generation(
            peer_id=request.peer_id,
            model_dir=request.model_dir,
            model_id=request.model_id,
            validate_restore=request.validate_restore,
        )
    except NeuralImprintGenerationError as exc:
        status = 404 if exc.code in {"persona_source_not_found", "model_dir_not_found"} else 400
        raise HTTPException(status_code=status, detail=exc.to_error()) from exc
    return NeuralImprintGenerateJobResponse(job=job)


@router.get("/generate/{job_id}", response_model=NeuralImprintGenerateJobResponse)
def get_neural_imprint_generation(job_id: str) -> NeuralImprintGenerateJobResponse:
    """Poll one Mac-side Neural Imprint generation job."""

    job = get_neural_imprint_generation_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "neural_imprint_generation_job_not_found",
                "message": f"Neural Imprint generation job not found: {job_id}",
            },
        )
    return NeuralImprintGenerateJobResponse(job=job)


@router.post("/restore", response_model=NeuralImprintRuntimeStatusResponse)
def restore_neural_imprint(
    request: NeuralImprintRestoreRequest,
) -> NeuralImprintRuntimeStatusResponse:
    """Restore one Neural Imprint full-cache artifact for Mac-side chat preview."""

    artifact_path, sidecar_path, artifact_id = _resolve_restore_artifact(request)
    try:
        status = restore_neural_imprint_for_model(
            model_id=request.model_id,
            artifact_path=artifact_path,
            sidecar_path=sidecar_path,
            artifact_id=artifact_id,
        )
    except NeuralImprintRuntimeError as exc:
        http_status = 404 if exc.code in {"model_not_loaded", "sidecar_not_found"} else 400
        raise HTTPException(status_code=http_status, detail=exc.to_error()) from exc
    return _runtime_status_response(status)


@router.post("/unload", response_model=NeuralImprintRuntimeStatusResponse)
def unload_neural_imprint_endpoint(
    request: NeuralImprintUnloadRequest,
) -> NeuralImprintRuntimeStatusResponse:
    """Clear Neural Imprint preview state."""

    status = unload_neural_imprint(request.model_id)
    return _runtime_status_response(status)


@router.get("/status", response_model=NeuralImprintRuntimeStatusResponse)
def get_neural_imprint_runtime_status(
    model_id: str | None = Query(None),
) -> NeuralImprintRuntimeStatusResponse:
    """Return the active Neural Imprint preview state."""

    status = get_neural_imprint_status(model_id)
    return _runtime_status_response(status)


@router.get("/parse", response_model=NeuralImprintInspectResponse)
def parse_neural_imprint(
    path: str = Query(..., description="Local .safetensors artifact path"),
    sidecar_path: str | None = Query(None, description="Optional neural_imprint_metadata.json path"),
    current_model_id: str | None = Query(None),
    current_model_hash: str | None = Query(None),
    tokenizer_hash: str | None = Query(None),
    tool_schema_hash: str | None = Query(None),
    cache_topology_sha256: str | None = Query(None),
) -> NeuralImprintInspectResponse:
    artifact = _safe_file_path(path, suffixes=(".safetensors",))
    sidecar = _safe_file_path(sidecar_path, suffixes=(".json",)) if sidecar_path else None
    return inspect_neural_imprint_artifact(
        artifact,
        sidecar_path=sidecar,
        current_model_id=current_model_id,
        current_model_hash=current_model_hash,
        tokenizer_hash=tokenizer_hash,
        tool_schema_hash=tool_schema_hash,
        cache_topology_sha256=cache_topology_sha256,
    )


@router.post("/parse", response_model=NeuralImprintInspectResponse)
async def upload_neural_imprint(
    artifact: UploadFile = File(...),
    sidecar: UploadFile | None = File(None),
    current_model_id: str | None = Query(None),
    current_model_hash: str | None = Query(None),
    tokenizer_hash: str | None = Query(None),
    tool_schema_hash: str | None = Query(None),
    cache_topology_sha256: str | None = Query(None),
) -> NeuralImprintInspectResponse:
    if not artifact.filename or not artifact.filename.endswith(".safetensors"):
        raise HTTPException(400, "Expected a .safetensors artifact upload")
    header, header_len = _read_safetensors_header(artifact.file, artifact.filename)
    header_metadata = _metadata_from_header(header)

    sidecar_metadata: dict[str, Any] = {}
    sidecar_found = False
    sidecar_name: str | None = None
    if sidecar is not None:
        sidecar_name = sidecar.filename
        try:
            parsed = json.loads((await sidecar.read()).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, "Sidecar metadata is not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Sidecar metadata must be a JSON object")
        sidecar_metadata = parsed
        sidecar_found = True

    summary = _build_summary(header_metadata, sidecar_metadata)
    tensors = _tensor_infos(header)
    return NeuralImprintInspectResponse(
        artifact_path=None,
        artifact_name=artifact.filename,
        artifact_size_bytes=None,
        header_size_bytes=header_len,
        safetensors_metadata=header_metadata,
        tensor_count=len(tensors),
        tensors=tensors,
        sidecar_found=sidecar_found,
        sidecar_path=sidecar_name,
        sidecar_metadata=sidecar_metadata,
        summary=summary,
        compatibility=_compatibility(
            summary,
            current_model_id=current_model_id,
            current_model_hash=current_model_hash,
            tokenizer_hash=tokenizer_hash,
            tool_schema_hash=tool_schema_hash,
            cache_topology_sha256=cache_topology_sha256,
        ),
    )
