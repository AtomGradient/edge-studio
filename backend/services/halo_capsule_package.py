# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-side Halo capsule package construction for Neural Imprint artifacts.

This module prepares the Python / EdgeStudio side of A3 distribution. It does
not pair devices, restore caches, or generate Neural Imprint; it only turns an
existing local artifact directory into the same offer/chunk/complete frame
sequence that edge-kit's Swift receiver understands.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote


HALO_MESSAGE_SCHEMA_VERSION = "edgestudio.halo_capsule_mesh_message.v1"
HALO_MESSAGE_KIND = "halo_capsule_offer"
HALO_DESCRIPTOR_SCHEMA_VERSION = "edgestudio.halo_capsule_descriptor.v1"
HALO_ARTIFACT_MEDIA_TYPE = "application/vnd.edgestudio.halo-capsule"
HALO_OP_OFFER = HALO_MESSAGE_KIND
HALO_OP_CHUNK = "halo_capsule_chunk"
HALO_OP_COMPLETE = "halo_capsule_complete"
DEFAULT_CHUNK_SIZE = 1 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
NEURAL_IMPRINT_ARTIFACT_NAME = "neural_imprint.safetensors"
NEURAL_IMPRINT_METADATA_NAME = "neural_imprint_metadata.json"
REQUIRED_PREFIX_RENDERER_VERSION = "edgestudio.neural_imprint.renderer.v1"
SUPPORTED_PREFIX_RENDERER_VERSIONS = {
    REQUIRED_PREFIX_RENDERER_VERSION,
}

_APPLE_REFERENCE_DATE = datetime(2001, 1, 1, tzinfo=UTC)


class HaloCapsulePackageError(ValueError):
    pass


@dataclass(frozen=True)
class HaloCapsulePackage:
    package_directory: Path
    message: dict[str, Any]
    files: tuple[dict[str, Any], ...]

    @property
    def transfer_id(self) -> str:
        return str(self.message["transfer_id"])


def build_neural_imprint_halo_package(
    neural_imprint_dir: Path,
    *,
    min_runtime_version: str,
    transfer_id: str | None = None,
    capsule_id: str | None = None,
    created_at: datetime | None = None,
) -> HaloCapsulePackage:
    """Build a Halo capsule package descriptor for one Neural Imprint directory."""

    root = Path(neural_imprint_dir).expanduser().resolve()
    artifact_path = root / NEURAL_IMPRINT_ARTIFACT_NAME
    sidecar_path = root / NEURAL_IMPRINT_METADATA_NAME
    if not artifact_path.exists():
        raise HaloCapsulePackageError(f"Neural Imprint artifact not found: {artifact_path}")
    if not sidecar_path.exists():
        raise HaloCapsulePackageError(f"Neural Imprint sidecar not found: {sidecar_path}")
    sidecar = _read_json_object(sidecar_path, sidecar_path.name)
    header = _read_safetensors_header(artifact_path)
    header_metadata = _metadata_from_header(header)
    _validate_prefix_renderer_version(header_metadata)

    created = created_at or _created_at(header_metadata, sidecar)
    files = _package_files(root)
    artifact_sha256 = _sha256_concat(root / item["name"] for item in files)
    requirements = _requirements(
        sidecar=sidecar,
        header_metadata=header_metadata,
        cache_topology_sha256=_cache_topology_sha256(sidecar),
    )
    cache_snapshot = _cache_snapshot(
        sidecar=sidecar,
        header=header,
        created_at=created,
        requirements=requirements,
    )
    artifact = {
        "artifact_id": f"neural-imprint-{artifact_sha256[:16]}",
        "media_type": HALO_ARTIFACT_MEDIA_TYPE,
        "total_bytes": sum(int(item["byte_count"]) for item in files),
        "sha256": artifact_sha256,
        "files": list(files),
    }
    descriptor = {
        "schema_version": HALO_DESCRIPTOR_SCHEMA_VERSION,
        "capsule_id": capsule_id or f"neural-imprint-{artifact_sha256[:12]}",
        "created_at": _swift_date(created),
        "base_model_id": _required_str(_lookup(sidecar, "model", "id"), "model.id"),
        "min_runtime_version": _required_non_empty(min_runtime_version, "min_runtime_version"),
        "requirements_sha256": _canonical_sha256(requirements),
        "requirements": requirements,
        "cache_snapshot": cache_snapshot,
        "artifact": artifact,
    }
    message = {
        "schema_version": HALO_MESSAGE_SCHEMA_VERSION,
        "kind": HALO_MESSAGE_KIND,
        "transfer_id": transfer_id or f"transfer-{artifact_sha256[:16]}",
        "capsule": descriptor,
    }
    return HaloCapsulePackage(package_directory=root, message=message, files=tuple(files))




def iter_halo_capsule_transfer_frames(
    package: HaloCapsulePackage,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[bytes]:
    """Yield offer JSON, chunk-header JSON, raw binary chunks, and complete JSON."""

    if chunk_size <= 0:
        raise HaloCapsulePackageError("chunk_size must be positive")
    yield _transfer_envelope(HALO_OP_OFFER, package.message)

    for file_spec in package.files:
        name = _safe_file_name(str(file_spec["name"]))
        file_path = package.package_directory / name
        total_bytes = int(file_spec["byte_count"])
        offset = 0
        with file_path.open("rb") as handle:
            while offset < total_bytes:
                chunk = handle.read(min(chunk_size, total_bytes - offset))
                if not chunk:
                    raise HaloCapsulePackageError(
                        f"artifact file ended early: {name} at {offset}/{total_bytes}"
                    )
                header = {
                    "transfer_id": package.transfer_id,
                    "file_name": name,
                    "offset": offset,
                    "byte_count": len(chunk),
                    "sha256": _sha256_bytes(chunk),
                }
                yield _transfer_envelope(HALO_OP_CHUNK, header)
                yield chunk
                offset += len(chunk)

    yield _transfer_envelope(
        HALO_OP_COMPLETE,
        {"transfer_id": package.transfer_id},
    )


def push_halo_capsule_package_to_peer(
    server: Any,
    peer_id: str,
    package: HaloCapsulePackage,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Push package frames through `MeshTransportServer.send_frame_to_peer`.

    This is intentionally synchronous and fail-fast. A caller that needs retry
    or background scheduling should wrap this primitive at a higher layer.
    """

    sent = 0
    bytes_sent = 0
    for frame in iter_halo_capsule_transfer_frames(package, chunk_size=chunk_size):
        ok = server.send_frame_to_peer(
            peer_id,
            frame,
            op_label="halo_capsule_package",
        )
        if not ok:
            raise HaloCapsulePackageError(
                f"peer {peer_id} is not connected or frame send failed"
            )
        sent += 1
        bytes_sent += len(frame)
    return {
        "ok": True,
        "peer_id": peer_id,
        "transfer_id": package.transfer_id,
        "frame_count": sent,
        "payload_bytes": bytes_sent,
    }


def package_with_download_urls(
    package: HaloCapsulePackage,
    *,
    base_url: str,
    download_token: str | None = None,
) -> HaloCapsulePackage:
    """Return a copy of `package` whose artifact files carry HTTP download URLs."""

    normalized_base = base_url.rstrip("/")
    if not normalized_base:
        raise HaloCapsulePackageError("download base_url must be non-empty")

    message = deepcopy(package.message)
    files = message["capsule"]["artifact"]["files"]
    for file_spec in files:
        name = _safe_file_name(str(file_spec["name"]))
        url = (
            f"{normalized_base}/"
            f"{quote(package.transfer_id, safe='')}/"
            f"{quote(name, safe='')}"
        )
        if download_token:
            url = f"{url}?token={quote(download_token, safe='')}"
        file_spec["download_url"] = url
    return HaloCapsulePackage(
        package_directory=package.package_directory,
        message=message,
        files=package.files,
    )


def push_halo_capsule_download_offer_to_peer(
    server: Any,
    peer_id: str,
    package: HaloCapsulePackage,
) -> dict[str, Any]:
    """Send only the Halo capsule offer frame.

    Bulk artifact bytes are fetched by the device through the per-file
    `download_url` fields in the offer. This mirrors the old LoRA path: mTLS is
    used as a small authenticated control channel, not as the large-file pipe.
    """

    frame = _transfer_envelope(HALO_OP_OFFER, package.message)
    ok = server.send_frame_to_peer(
        peer_id,
        frame,
        op_label="halo_capsule_download_offer",
    )
    if not ok:
        raise HaloCapsulePackageError(
            f"peer {peer_id} is not connected or offer send failed"
        )
    return {
        "ok": True,
        "peer_id": peer_id,
        "transfer_id": package.transfer_id,
        "frame_count": 1,
        "payload_bytes": len(frame),
    }


def _package_files(root: Path) -> list[dict[str, Any]]:
    names = [
        NEURAL_IMPRINT_ARTIFACT_NAME,
        NEURAL_IMPRINT_METADATA_NAME,
        "profile_body.txt",
        "tool_specs.json",
    ]
    files: list[dict[str, Any]] = []
    for name in names:
        path = root / name
        if not path.exists():
            if name in {
                NEURAL_IMPRINT_ARTIFACT_NAME,
                NEURAL_IMPRINT_METADATA_NAME,
            }:
                _required_file(path)
            continue
        safe_name = _safe_file_name(name)
        if not path.is_file():
            raise HaloCapsulePackageError(f"{safe_name} is not a regular file")
        files.append({
            "name": safe_name,
            "byte_count": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    return files


def _model_family_from_model_id(model_id: str, *, fallback: str) -> str:
    """Return the product model family used for cross-model fail-closed checks."""

    clean = str(model_id or "").strip()
    if not clean:
        return fallback
    name = clean.rsplit("/", 1)[-1].strip().lower()
    if not name:
        return fallback
    # Qwen3.5-9B-4bit -> qwen3.5-9b. Keep MoE shape labels such as
    # Qwen3.6-35B-A3B-8bit -> qwen3.6-35b-a3b.
    name = re.sub(r"-(?:[0-9]+bit|bf16|fp16|f16|float16|fp32|f32)$", "", name)
    return name or fallback


def _requirements(
    *,
    sidecar: dict[str, Any],
    header_metadata: dict[str, Any],
    cache_topology_sha256: str,
) -> dict[str, Any]:
    model = _required_dict(sidecar.get("model"), "model")
    tokenizer = _required_dict(sidecar.get("tokenizer"), "tokenizer")
    prefix = _required_dict(sidecar.get("prefix"), "prefix")
    source = _required_dict(sidecar.get("source"), "source")
    return {
        "model_family": _model_family_from_model_id(
            _required_str(model.get("id"), "model.id"),
            fallback=_required_str(model.get("architecture"), "model.architecture"),
        ),
        "hidden_size": _required_int(model.get("hidden_size"), "model.hidden_size"),
        "layer_count": _required_int(model.get("num_layers"), "model.num_layers"),
        "model_config_sha256": _required_str(
            header_metadata.get("model_config_sha256"),
            "safetensors.__metadata__.model_config_sha256",
        ),
        "model_weights_sha256": _required_str(
            header_metadata.get("model_weights_fingerprint"),
            "safetensors.__metadata__.model_weights_fingerprint",
        ),
        "tokenizer_json_sha256": _required_str(
            tokenizer.get("tokenizer_json_sha256"),
            "tokenizer.tokenizer_json_sha256",
        ),
        "tokenizer_config_sha256": _required_str(
            tokenizer.get("tokenizer_config_sha256"),
            "tokenizer.tokenizer_config_sha256",
        ),
        "chat_template_sha256": _required_str(
            tokenizer.get("chat_template_sha256"),
            "tokenizer.chat_template_sha256",
        ),
        "system_prompt_sha256": _required_str(
            header_metadata.get("system_prompt_sha256"),
            "safetensors.__metadata__.system_prompt_sha256",
        ),
        "rendered_prefix_sha256": _required_str(
            prefix.get("rendered_prefix_sha256"),
            "prefix.rendered_prefix_sha256",
        ),
        "prefix_token_count": _required_int(prefix.get("token_count"), "prefix.token_count"),
        "tool_schema_sha256": _required_str(
            source.get("tool_schema_sha256"),
            "source.tool_schema_sha256",
        ),
        "profile_body_sha256": _required_str(
            source.get("profile_body_sha256"),
            "source.profile_body_sha256",
        ),
        "enable_thinking": bool(tokenizer.get("enable_thinking", False)),
        "cache_backend": _required_str(
            header_metadata.get("cache_backend"),
            "safetensors.__metadata__.cache_backend",
        ),
        "cache_backend_version": _required_str(
            header_metadata.get("cache_backend_version"),
            "safetensors.__metadata__.cache_backend_version",
        ),
        "cache_topology_sha256": cache_topology_sha256,
    }


def _cache_snapshot(
    *,
    sidecar: dict[str, Any],
    header: dict[str, Any],
    created_at: datetime,
    requirements: dict[str, Any],
) -> dict[str, Any]:
    prefix = _required_dict(sidecar.get("prefix"), "prefix")
    tensors = _tensor_references(
        cache_manifest=_required_dict(sidecar.get("cache_manifest"), "cache_manifest"),
        header=header,
    )
    return {
        "snapshot_id": "neural-imprint-" + _required_str(
            sidecar.get("artifact_sha256"),
            "artifact_sha256",
        )[:16],
        "created_at": _swift_date(created_at),
        "token_count": requirements["prefix_token_count"],
        "token_ids_sha256": _required_str(
            prefix.get("token_ids_sha256"),
            "prefix.token_ids_sha256",
        ),
        "cache_backend": requirements["cache_backend"],
        "cache_backend_version": requirements["cache_backend_version"],
        "tensors": tensors,
    }


def _tensor_references(
    *,
    cache_manifest: dict[str, Any],
    header: dict[str, Any],
) -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    raw_layers = cache_manifest.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise HaloCapsulePackageError("cache_manifest.layers must be a non-empty list")
    for layer in raw_layers:
        if not isinstance(layer, dict):
            raise HaloCapsulePackageError("cache_manifest.layers[] must be an object")
        states = layer.get("states")
        if not isinstance(states, list) or not states:
            raise HaloCapsulePackageError("cache_manifest.layers[].states must be non-empty")
        for state in states:
            if not isinstance(state, dict):
                raise HaloCapsulePackageError("cache state must be an object")
            name = _required_str(state.get("name"), "cache state name")
            tensor_header = header.get(name)
            byte_count = _tensor_byte_count(tensor_header)
            if byte_count is None:
                byte_count = _estimated_byte_count(
                    _required_int_list(state.get("shape"), f"{name}.shape"),
                    _required_str(state.get("dtype"), f"{name}.dtype"),
                )
            tensors.append({
                "name": name,
                "shape": _required_int_list(state.get("shape"), f"{name}.shape"),
                "dtype": _required_str(state.get("dtype"), f"{name}.dtype"),
                "byte_count": byte_count,
            })
    return tensors


def _cache_topology_sha256(sidecar: dict[str, Any]) -> str:
    cache_manifest = _required_dict(sidecar.get("cache_manifest"), "cache_manifest")
    layers = cache_manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise HaloCapsulePackageError("cache_manifest.layers must be a non-empty list")
    normalized_layers: list[dict[str, Any]] = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise HaloCapsulePackageError("cache_manifest.layers[] must be an object")
        states = layer.get("states")
        if not isinstance(states, list) or not states:
            raise HaloCapsulePackageError("cache_manifest.layers[].states must be non-empty")
        normalized_states = []
        for state in states:
            if not isinstance(state, dict):
                raise HaloCapsulePackageError("cache state must be an object")
            normalized_states.append({
                "name": _required_str(state.get("name"), "cache state name"),
                "shape": _required_int_list(
                    state.get("shape"),
                    f"{state.get('name', 'cache state')}.shape",
                ),
                "dtype": _required_str(
                    state.get("dtype"),
                    f"{state.get('name', 'cache state')}.dtype",
                ),
            })
        normalized_layers.append({"states": normalized_states})
    return _canonical_sha256({"layers": normalized_layers})


def _transfer_envelope(op: str, payload: dict[str, Any]) -> bytes:
    return _json_bytes({"op": op, "payload": payload})


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        length_data = handle.read(8)
        if len(length_data) != 8:
            raise HaloCapsulePackageError(f"{path.name} is too small")
        header_len = struct.unpack("<Q", length_data)[0]
        if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise HaloCapsulePackageError(f"{path.name} has invalid header length")
        raw_header = handle.read(header_len)
        if len(raw_header) != header_len:
            raise HaloCapsulePackageError(f"{path.name} ended before header completed")
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HaloCapsulePackageError(f"{path.name} has invalid header JSON") from exc
    if not isinstance(header, dict):
        raise HaloCapsulePackageError(f"{path.name} header must be an object")
    return header


def _metadata_from_header(header: dict[str, Any]) -> dict[str, Any]:
    metadata = header.get("__metadata__")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _validate_prefix_renderer_version(header_metadata: dict[str, Any]) -> None:
    actual = str(header_metadata.get("prefix_renderer_version") or "").strip()
    if actual not in SUPPORTED_PREFIX_RENDERER_VERSIONS:
        raise HaloCapsulePackageError(
            "unsupported prefix_renderer_version"
            if actual
            else "missing prefix_renderer_version"
        )


def _created_at(header_metadata: dict[str, Any], sidecar: dict[str, Any]) -> datetime:
    raw = header_metadata.get("created_at") or sidecar.get("created_at")
    if isinstance(raw, str):
        text = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromtimestamp(time.time(), tz=UTC)


def _swift_date(value: datetime) -> float:
    normalized = value if value.tzinfo else value.replace(tzinfo=UTC)
    return (normalized.astimezone(UTC) - _APPLE_REFERENCE_DATE).total_seconds()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HaloCapsulePackageError(f"{label} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise HaloCapsulePackageError(f"{label} must be a JSON object")
    return data


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise HaloCapsulePackageError(f"missing required file: {path.name}")
    if not path.is_file():
        raise HaloCapsulePackageError(f"not a file: {path.name}")
    return path


def _required_first_existing(root: Path, *names: str) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return _required_file(root / names[0])


def _safe_file_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise HaloCapsulePackageError("artifact file name is empty or unsafe")
    if "/" in name or "\\" in name or os.path.basename(name) != name:
        raise HaloCapsulePackageError(f"unsafe artifact file name: {name}")
    return name


def _lookup(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _required_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HaloCapsulePackageError(f"{field} must be an object")
    return value


def _required_str(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise HaloCapsulePackageError(f"{field} must be a non-empty string")
    return text


def _required_non_empty(value: str, field: str) -> str:
    return _required_str(value, field)


def _required_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise HaloCapsulePackageError(f"{field} must be an integer") from exc
    if number <= 0:
        raise HaloCapsulePackageError(f"{field} must be positive")
    return number


def _required_int_list(value: Any, field: str) -> list[int]:
    if not isinstance(value, list):
        raise HaloCapsulePackageError(f"{field} must be a list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise HaloCapsulePackageError(f"{field} must contain integers") from exc


def _tensor_byte_count(tensor_header: Any) -> int | None:
    if not isinstance(tensor_header, dict):
        return None
    offsets = tensor_header.get("data_offsets")
    if not isinstance(offsets, list) or len(offsets) != 2:
        return None
    return max(0, int(offsets[1]) - int(offsets[0]))


def _estimated_byte_count(shape: list[int], dtype: str) -> int:
    elements = 1
    for dim in shape:
        elements *= int(dim)
    lower = dtype.lower()
    if lower in {"mlx.core.float32", "float32", "f32"}:
        bytes_per_element = 4
    elif lower in {
        "mlx.core.float16",
        "mlx.core.bfloat16",
        "float16",
        "bfloat16",
        "f16",
        "bf16",
    }:
        bytes_per_element = 2
    else:
        bytes_per_element = 0
    return elements * bytes_per_element


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_concat(paths: Iterator[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(value, sort_keys=True))


def _json_bytes(value: dict[str, Any], *, sort_keys: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    ).encode("utf-8")
