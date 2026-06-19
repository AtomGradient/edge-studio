# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Read-only registry for host-side Neural Imprint artifacts.

The registry answers one question for Phase B distribution: which local
Neural Imprint directories are valid capsule sources? It does not generate,
mutate, push, or restore artifacts.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_dirs import data_path, unique_roots
from .halo_capsule_package import (
    HaloCapsulePackageError,
    NEURAL_IMPRINT_ARTIFACT_NAME,
    NEURAL_IMPRINT_METADATA_NAME,
    build_neural_imprint_halo_package,
)


REGISTRY_SCHEMA_VERSION = "edgestudio.neural_imprint_artifact_registry.v2"
DEFAULT_SCAN_DEPTH = 8
_VALIDATION_RUNTIME = "0.0.0-registry"
NEURAL_IMPRINT_GENERATION_RECEIPT_NAME = "neural_imprint_generation_receipt.json"


@dataclass
class NeuralImprintArtifactRegistryError(ValueError):
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


def default_neural_imprint_artifact_roots() -> list[Path]:
    configured = os.environ.get("EDGE_NEURAL_IMPRINT_ARTIFACT_ROOTS", "").strip()
    if configured:
        return [_expand_root(item) for item in _split_roots(configured)]
    return unique_roots(
        data_path("persona", "neural_imprint_artifacts"),
    )


def list_neural_imprint_artifacts(
    *,
    roots: list[Path] | None = None,
    include_invalid: bool = False,
    max_depth: int = DEFAULT_SCAN_DEPTH,
) -> dict[str, Any]:
    scan_roots = roots if roots is not None else default_neural_imprint_artifact_roots()
    artifacts: list[dict[str, Any]] = []
    for root in scan_roots:
        for artifact_path in _iter_neural_imprint_files(root, max_depth=max_depth):
            item = _artifact_item(artifact_path.parent)
            if item.get("valid") is True or include_invalid:
                artifacts.append(item)
    artifacts.sort(key=lambda item: (str(item.get("base_model_id") or ""), str(item.get("artifact_id") or "")))
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "roots": [str(Path(root).expanduser()) for root in scan_roots],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def find_neural_imprint_artifact(
    artifact_id: str,
    *,
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    needle = _required_lookup_id(artifact_id)
    registry = list_neural_imprint_artifacts(roots=roots)
    matches = [
        artifact
        for artifact in registry["artifacts"]
        if needle
        in {
            str(artifact.get("artifact_id") or ""),
            str(artifact.get("capsule_id") or ""),
            str(artifact.get("artifact_sha256") or ""),
            str(artifact.get("neural_imprint_sha256") or ""),
        }
    ]
    if not matches:
        raise NeuralImprintArtifactRegistryError(
            "artifact_not_found",
            f"Neural Imprint artifact not found: {needle}",
            {"artifact_id": needle},
        )
    if len(matches) > 1:
        raise NeuralImprintArtifactRegistryError(
            "ambiguous_artifact_id",
            f"Neural Imprint artifact id is ambiguous: {needle}",
            {"artifact_id": needle, "match_count": len(matches)},
        )
    return matches[0]


def find_neural_imprint_artifact_for_loaded_model(
    loaded_model: Any,
    *,
    roots: list[Path] | None = None,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the newest artifact that strictly matches a loaded host model.

    The host runtime must fail closed: a Neural Imprint generated for a
    different model id, architecture, hidden size, or caller-supplied learning
    signature is not eligible for automatic restore.
    """

    expected_model_id = _loaded_base_model_id(loaded_model)
    expected_hidden = _loaded_hidden_size(loaded_model)
    expected_architecture = _loaded_architecture(loaded_model)
    if not expected_model_id or expected_hidden is None or not expected_architecture:
        return None

    candidates: list[dict[str, Any]] = []
    for artifact in list_neural_imprint_artifacts(roots=roots)["artifacts"]:
        if artifact.get("valid") is not True:
            continue
        if str(artifact.get("base_model_id") or "") != expected_model_id:
            continue
        if _coerce_int(artifact.get("hidden_size")) != expected_hidden:
            continue
        if str(artifact.get("model_architecture") or "") != expected_architecture:
            continue
        if not _artifact_matches_requirements(artifact, requirements):
            continue
        candidates.append(artifact)

    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            _coerce_float(item.get("mtime")) or 0.0,
            str(item.get("artifact_id") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def resolve_neural_imprint_artifact_dir(artifact_id: str) -> Path:
    artifact = find_neural_imprint_artifact(artifact_id)
    directory = str(artifact.get("artifact_dir") or "").strip()
    if not directory:
        raise NeuralImprintArtifactRegistryError(
            "invalid_artifact_record",
            "Neural Imprint artifact record has no artifact_dir",
            {"artifact_id": artifact_id},
        )
    return Path(directory)


def _artifact_item(directory: Path) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    artifact_path = root / NEURAL_IMPRINT_ARTIFACT_NAME
    sidecar_path = root / NEURAL_IMPRINT_METADATA_NAME
    base = {
        "artifact_dir": str(root),
        "artifact_path": str(artifact_path),
        "sidecar_path": str(sidecar_path),
    }
    try:
        package = build_neural_imprint_halo_package(
            root,
            min_runtime_version=_VALIDATION_RUNTIME,
        )
    except HaloCapsulePackageError as exc:
        return {
            **base,
            "valid": False,
            "error": str(exc),
        }

    capsule = package.message["capsule"]
    artifact = capsule["artifact"]
    requirements = capsule["requirements"]
    cache_snapshot = capsule["cache_snapshot"]
    generation_receipt = _generation_receipt(root)
    source_receipt = _dict(generation_receipt.get("source_receipt"))
    return {
        **base,
        "valid": True,
        "artifact_id": artifact["artifact_id"],
        "capsule_id": capsule["capsule_id"],
        "transfer_id_hint": package.transfer_id,
        "base_model_id": capsule["base_model_id"],
        "model_architecture": _sidecar_model_value(sidecar_path, "architecture"),
        "model_family": requirements.get("model_family"),
        "hidden_size": requirements.get("hidden_size"),
        "layer_count": requirements.get("layer_count"),
        "prefix_token_count": requirements.get("prefix_token_count"),
        "tool_schema_sha256": requirements.get("tool_schema_sha256"),
        "artifact_tool_schema_sha256": _first_text("artifact_tool_schema_sha256", generation_receipt)
        or requirements.get("tool_schema_sha256"),
        "source_tool_schema_sha256": _first_text(
            "source_tool_schema_sha256",
            generation_receipt,
            source_receipt,
        ),
        "profile_body_sha256": requirements.get("profile_body_sha256"),
        "source_id": _first_text("source_id", generation_receipt, source_receipt),
        "source_sha256": _first_text("source_sha256", generation_receipt, source_receipt),
        "source_kind": _first_text("source_kind", generation_receipt, source_receipt),
        "source_peer_id": _first_text("peer_id", generation_receipt, source_receipt),
        "peer_id": _first_text("peer_id", generation_receipt, source_receipt),
        "app_id": _first_text("app_id", generation_receipt, source_receipt),
        "cache_backend": requirements.get("cache_backend"),
        "cache_backend_version": requirements.get("cache_backend_version"),
        "cache_topology_sha256": requirements.get("cache_topology_sha256"),
        "neural_imprint_sha256": _sidecar_artifact_sha256(sidecar_path),
        "artifact_sha256": artifact["sha256"],
        "total_bytes": artifact["total_bytes"],
        "file_count": len(artifact["files"]),
        "created_at": cache_snapshot.get("created_at"),
        "mtime": artifact_path.stat().st_mtime if artifact_path.exists() else None,
    }


def _artifact_matches_requirements(
    artifact: dict[str, Any],
    requirements: dict[str, Any] | None,
) -> bool:
    if not requirements:
        return True
    if requirements.get("strict_no_match") is True:
        return False
    fields = (
        "tool_schema_sha256",
        "profile_body_sha256",
        "source_id",
        "source_sha256",
        "source_kind",
        "app_id",
        "peer_id",
    )
    for field in fields:
        expected = _optional_text(requirements.get(field))
        if not expected:
            continue
        if field == "tool_schema_sha256":
            actual_values = {
                _optional_text(artifact.get("tool_schema_sha256")),
                _optional_text(artifact.get("artifact_tool_schema_sha256")),
                _optional_text(artifact.get("source_tool_schema_sha256")),
            }
            actual_values.discard(None)
            if expected not in actual_values:
                return False
            continue
        if _optional_text(artifact.get(field)) != expected:
            return False
    return True


def _iter_neural_imprint_files(root: Path, *, max_depth: int) -> list[Path]:
    base = Path(root).expanduser()
    if not base.exists():
        return []
    if base.is_file():
        return [base.resolve()] if base.name in _ARTIFACT_FILE_NAMES else []
    if not base.is_dir():
        return []
    resolved = base.resolve()
    found_by_directory: dict[Path, Path] = {}
    for current, dirs, files in os.walk(resolved):
        rel = Path(current).relative_to(resolved)
        depth = 0 if str(rel) == "." else len(rel.parts)
        dirs[:] = [item for item in dirs if item not in {".git", "node_modules", "__pycache__"}]
        if depth >= max_depth:
            dirs[:] = []
        current_path = Path(current)
        if NEURAL_IMPRINT_ARTIFACT_NAME in files:
            found_by_directory[current_path] = current_path / NEURAL_IMPRINT_ARTIFACT_NAME
    return sorted(found_by_directory.values())


_ARTIFACT_FILE_NAMES = {NEURAL_IMPRINT_ARTIFACT_NAME}


def _first_existing(root: Path, preferred: str, legacy: str) -> Path:
    preferred_path = root / preferred
    if preferred_path.exists():
        return preferred_path
    return root / legacy


def _sidecar_artifact_sha256(path: Path) -> str | None:
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    value = str(data.get("artifact_sha256") or "").strip().lower()
    return value if re.fullmatch(r"[a-f0-9]{64}", value) else None


def _generation_receipt(root: Path) -> dict[str, Any]:
    for name in (NEURAL_IMPRINT_GENERATION_RECEIPT_NAME,):
        path = root / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            return data
    return {}


def _first_text(key: str, *sources: dict[str, Any]) -> str | None:
    for source in sources:
        value = _optional_text(source.get(key))
        if value:
            return value
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sidecar_model_value(path: Path, key: str) -> str | None:
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    if not isinstance(model, dict):
        return None
    value = model.get(key)
    text = str(value).strip() if value is not None else ""
    return text or None


def _split_roots(value: str) -> list[str]:
    chunks: list[str] = []
    for part in value.split(os.pathsep):
        chunks.extend(part.split(","))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _expand_root(value: str) -> Path:
    return Path(value).expanduser()


def _loaded_base_model_id(loaded_model: Any) -> str | None:
    model_dir = getattr(loaded_model, "model_dir", None)
    if model_dir:
        name = Path(str(model_dir)).name.strip()
        if name:
            return name
    model_id = getattr(loaded_model, "model_id", None)
    text = str(model_id).strip() if model_id is not None else ""
    return text or None


def _loaded_hidden_size(loaded_model: Any) -> int | None:
    return _config_int(_loaded_config(loaded_model), "hidden_size")


def _loaded_architecture(loaded_model: Any) -> str | None:
    return _config_str(_loaded_config(loaded_model), "model_type", "architectures")


def _loaded_config(loaded_model: Any) -> dict[str, Any]:
    config = getattr(loaded_model, "config", None)
    return config if isinstance(config, dict) else {}


def _config_int(config: dict[str, Any], *keys: str) -> int | None:
    text_config = config.get("text_config")
    configs = [text_config, config] if isinstance(text_config, dict) else [config]
    for cfg in configs:
        for key in keys:
            coerced = _coerce_int(cfg.get(key))
            if coerced is not None:
                return coerced
    return None


def _config_str(config: dict[str, Any], *keys: str) -> str | None:
    text_config = config.get("text_config")
    configs = [config, text_config] if isinstance(text_config, dict) else [config]
    for cfg in configs:
        for key in keys:
            value = cfg.get(key)
            if isinstance(value, list) and value:
                value = value[0]
            text = str(value).strip() if value is not None else ""
            if text:
                return text
    return None


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


def _required_lookup_id(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise NeuralImprintArtifactRegistryError(
            "missing_artifact_id",
            "artifact_id is required",
            {},
        )
    return clean
