# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Runtime Neural Imprint restore state for Mac-side chat preview."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.api.chat_loaders import _get_or_load_mlx_model
from backend.services.mlx_runtime_gate import mlx_runtime_gate
from backend.services.mlx_worker import run_mlx_task
from backend.services.model_manager import manager


_COMPATIBILITY_METADATA_FIELDS = (
    "model_architecture",
    "model_config_sha256",
    "model_weights_fingerprint",
    "tokenizer_json_sha256",
    "tokenizer_config_sha256",
    "chat_template_sha256",
    "rendered_prefix_sha256",
    "prefix_token_ids_sha256",
    "enable_thinking",
    "cache_backend",
    "cache_backend_version",
)
NEURAL_IMPRINT_METADATA_NAME = "neural_imprint_metadata.json"
logger = logging.getLogger(__name__)


@dataclass
class NeuralImprintRuntimeError(ValueError):
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
class NeuralImprintRuntimeStatus:
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
    tool_schema_sha256: str | None = None
    artifact_tool_schema_sha256: str | None = None
    source_tool_schema_sha256: str | None = None
    profile_body_sha256: str | None = None
    source_id: str | None = None
    source_sha256: str | None = None
    source_kind: str | None = None
    source_peer_id: str | None = None
    app_id: str | None = None
    loaded_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "model_id": self.model_id,
            "model_dir": self.model_dir,
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "sidecar_path": self.sidecar_path,
            "prefix_token_count": self.prefix_token_count,
            "base_model_id": self.base_model_id,
            "model_architecture": self.model_architecture,
            "hidden_size": self.hidden_size,
            "layer_count": self.layer_count,
            "tool_schema_sha256": self.tool_schema_sha256,
            "artifact_tool_schema_sha256": self.artifact_tool_schema_sha256,
            "source_tool_schema_sha256": self.source_tool_schema_sha256,
            "profile_body_sha256": self.profile_body_sha256,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "source_kind": self.source_kind,
            "source_peer_id": self.source_peer_id,
            "app_id": self.app_id,
            "loaded_at": self.loaded_at,
        }


@dataclass
class _NeuralImprintRuntimeState:
    status: NeuralImprintRuntimeStatus
    cache: Any


_lock = threading.Lock()
_states: dict[str, _NeuralImprintRuntimeState] = {}


def restore_neural_imprint_for_model(
    *,
    model_id: str,
    artifact_path: Path,
    sidecar_path: Path | None = None,
    artifact_id: str | None = None,
) -> NeuralImprintRuntimeStatus:
    """Restore a Neural Imprint artifact into the shared MLX runtime."""

    loaded = manager.get_model(model_id)
    if loaded is None:
        raise NeuralImprintRuntimeError(
            "model_not_loaded",
            "Load the base model before restoring Neural Imprint.",
            {"model_id": model_id},
        )
    if loaded.category not in {"llm", "vlm"}:
        raise NeuralImprintRuntimeError(
            "unsupported_model_category",
            "Neural Imprint restore only supports LLM/VLM text models.",
            {"model_id": model_id, "category": loaded.category},
        )

    artifact = artifact_path.expanduser().resolve()
    sidecar = (
        sidecar_path.expanduser().resolve()
        if sidecar_path is not None
        else _default_sidecar_path(artifact)
    )
    sidecar_metadata = _read_sidecar(sidecar)
    _validate_sidecar_matches_loaded(loaded, sidecar_metadata)
    generation_receipt = _read_generation_receipt(sidecar.parent)

    def _restore_on_worker() -> tuple[Any, NeuralImprintRuntimeStatus]:
        _ensure_edgestudio_core_importable()
        from edgestudio_core.halo_capsule import full_cache

        with mlx_runtime_gate("neural_imprint.restore"):
            model, _tokenizer = _get_or_load_mlx_model(loaded.model_dir)
            header = full_cache.load_safetensors_metadata(artifact)
            expected_metadata = _expected_metadata_from_header(header)
            cache = full_cache.restore_full_cache(
                model,
                artifact,
                metadata_path=sidecar,
                expected_metadata=expected_metadata,
            )
            _eval_cache_state(cache)

        model_info = _mapping(sidecar_metadata.get("model"))
        prefix_info = _mapping(sidecar_metadata.get("prefix"))
        source_info = _mapping(sidecar_metadata.get("source"))
        source_receipt = _mapping(generation_receipt.get("source_receipt"))
        status = NeuralImprintRuntimeStatus(
            active=True,
            model_id=model_id,
            model_dir=loaded.model_dir,
            artifact_id=artifact_id,
            artifact_path=str(artifact),
            sidecar_path=str(sidecar),
            prefix_token_count=_coerce_int(prefix_info.get("token_count")),
            base_model_id=_coerce_str(model_info.get("id")),
            model_architecture=_coerce_str(model_info.get("architecture")),
            hidden_size=_coerce_int(model_info.get("hidden_size")),
            layer_count=_coerce_int(model_info.get("num_layers")),
            tool_schema_sha256=_first_text(
                source_info,
                generation_receipt,
                source_receipt,
                key="tool_schema_sha256",
            )
            or _first_text(generation_receipt, key="artifact_tool_schema_sha256"),
            artifact_tool_schema_sha256=_first_text(
                source_info,
                generation_receipt,
                key="tool_schema_sha256",
            )
            or _first_text(generation_receipt, key="artifact_tool_schema_sha256"),
            source_tool_schema_sha256=_first_text(
                generation_receipt,
                source_receipt,
                key="source_tool_schema_sha256",
            )
            or _first_text(source_receipt, key="tool_schema_sha256"),
            profile_body_sha256=_first_text(
                source_info,
                generation_receipt,
                source_receipt,
                key="profile_body_sha256",
            ),
            source_id=_first_text(generation_receipt, source_receipt, key="source_id"),
            source_sha256=_first_text(generation_receipt, source_receipt, key="source_sha256"),
            source_kind=_first_text(generation_receipt, source_receipt, key="source_kind"),
            source_peer_id=_first_text(generation_receipt, source_receipt, key="peer_id"),
            app_id=_first_text(generation_receipt, source_receipt, key="app_id"),
            loaded_at=time.time(),
        )
        return cache, status

    cache, status = run_mlx_task(_restore_on_worker)
    with _lock:
        _states[model_id] = _NeuralImprintRuntimeState(status=status, cache=cache)
    return status


def ensure_neural_imprint_for_loaded_model(
    loaded: Any,
    *,
    requirements: Mapping[str, Any] | None = None,
) -> NeuralImprintRuntimeStatus | None:
    """Restore the newest matching Neural Imprint for a loaded model, if present.

    This helper is intentionally fail-closed. If no artifact exactly matches the
    loaded model metadata, or if restore validation rejects the artifact, callers
    get ``None`` and should continue with the base model.
    """

    if getattr(loaded, "category", None) not in {"llm", "vlm"}:
        return None

    current = get_neural_imprint_status(getattr(loaded, "model_id", None))
    if (
        current.active
        and current.model_dir == getattr(loaded, "model_dir", None)
        and _status_matches_requirements(current, requirements)
    ):
        return current

    from backend.services.neural_imprint_artifact_registry import (
        NeuralImprintArtifactRegistryError,
        find_neural_imprint_artifact_for_loaded_model,
    )

    try:
        artifact = find_neural_imprint_artifact_for_loaded_model(
            loaded,
            requirements=dict(requirements or {}),
        )
    except NeuralImprintArtifactRegistryError:
        return None
    if artifact is None:
        return None

    artifact_path = _coerce_str(artifact.get("artifact_path"))
    sidecar_path = _coerce_str(artifact.get("sidecar_path"))
    if not artifact_path or not sidecar_path:
        return None
    try:
        return restore_neural_imprint_for_model(
            model_id=str(loaded.model_id),
            artifact_path=Path(artifact_path),
            sidecar_path=Path(sidecar_path),
            artifact_id=_coerce_str(artifact.get("artifact_id")),
        )
    except NeuralImprintRuntimeError as exc:
        logger.warning(
            "Neural Imprint auto-restore skipped for model=%s: %s",
            getattr(loaded, "model_id", None),
            exc.code,
        )
        return None


def unload_neural_imprint(model_id: str | None = None) -> NeuralImprintRuntimeStatus:
    """Clear one model's Neural Imprint state, or all state when model_id is omitted."""

    with _lock:
        if model_id:
            previous = _states.pop(model_id, None)
        else:
            previous = None
            _states.clear()
    if previous is not None:
        return NeuralImprintRuntimeStatus(
            active=False,
            model_id=previous.status.model_id,
            model_dir=previous.status.model_dir,
        )
    return NeuralImprintRuntimeStatus(active=False, model_id=model_id)


def get_neural_imprint_status(model_id: str | None = None) -> NeuralImprintRuntimeStatus:
    with _lock:
        if model_id:
            state = _states.get(model_id)
            return (
                state.status
                if state is not None
                else NeuralImprintRuntimeStatus(active=False, model_id=model_id)
            )
        if not _states:
            return NeuralImprintRuntimeStatus(active=False)
        latest = max(
            _states.values(),
            key=lambda item: item.status.loaded_at or 0,
        )
        return latest.status


def clone_neural_imprint_cache_for_model(
    *,
    model: Any,
    model_id: str,
    model_dir: str,
) -> tuple[Any | None, NeuralImprintRuntimeStatus | None]:
    """Return a fresh cache clone seeded with the active Neural Imprint, if any."""

    with _lock:
        state = _states.get(model_id)
    if state is None or state.status.model_dir != model_dir:
        return None, None

    _ensure_edgestudio_core_importable()
    from edgestudio_core.halo_capsule import full_cache

    clone = model.make_cache()
    for index, src in enumerate(state.cache):
        items = full_cache.state_items(src)
        src_type = type(src).__name__
        if src_type == "KVCache":
            clone[index].state = tuple(items)
        elif src_type == "ArraysCache":
            clone[index].state = list(items)
        else:
            raise NeuralImprintRuntimeError(
                "unsupported_cache_class",
                "Neural Imprint cache contains an unsupported cache class.",
                {"cache_class": src_type, "layer": index},
            )
    return clone, state.status


def _default_sidecar_path(artifact: Path) -> Path:
    return artifact.with_name(NEURAL_IMPRINT_METADATA_NAME)


def _read_sidecar(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NeuralImprintRuntimeError(
            "sidecar_not_found",
            "Neural Imprint sidecar metadata is missing.",
            {"sidecar_path": str(path)},
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NeuralImprintRuntimeError(
            "invalid_sidecar_json",
            "Neural Imprint sidecar metadata is not valid JSON.",
            {"sidecar_path": str(path)},
        ) from exc
    if not isinstance(data, dict):
        raise NeuralImprintRuntimeError(
            "invalid_sidecar_json",
            "Neural Imprint sidecar metadata must be a JSON object.",
            {"sidecar_path": str(path)},
        )
    return data


def _status_matches_requirements(
    status: NeuralImprintRuntimeStatus,
    requirements: Mapping[str, Any] | None,
) -> bool:
    if not requirements:
        return True
    if requirements.get("strict_no_match") is True:
        return False
    for field in (
        "tool_schema_sha256",
        "profile_body_sha256",
        "source_id",
        "source_sha256",
        "source_kind",
        "app_id",
        "peer_id",
    ):
        expected = _coerce_str(requirements.get(field))
        if not expected:
            continue
        status_field = "source_peer_id" if field == "peer_id" else field
        if field == "tool_schema_sha256":
            actual_values = {
                _coerce_str(status.tool_schema_sha256),
                _coerce_str(status.artifact_tool_schema_sha256),
                _coerce_str(status.source_tool_schema_sha256),
            }
            actual_values.discard(None)
            if expected not in actual_values:
                return False
            continue
        actual = _coerce_str(getattr(status, status_field, None))
        if actual != expected:
            return False
    return True


def _read_generation_receipt(root: Path) -> dict[str, Any]:
    for name in ("neural_imprint_generation_receipt.json",):
        path = root / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, dict):
            return data
    return {}


def _first_text(*sources: Mapping[str, Any], key: str) -> str | None:
    for source in sources:
        value = _coerce_str(source.get(key))
        if value:
            return value
    return None


def _validate_sidecar_matches_loaded(loaded: Any, sidecar: Mapping[str, Any]) -> None:
    model_info = _mapping(sidecar.get("model"))
    if not model_info:
        raise NeuralImprintRuntimeError(
            "missing_model_metadata",
            "Neural Imprint sidecar has no model metadata.",
            {},
        )

    artifact_model_id = _coerce_str(model_info.get("id"))
    current_model_names = {
        loaded.model_id,
        Path(loaded.model_dir).name,
        _coerce_str(getattr(loaded.architecture, "model_name", None)),
    }
    current_model_names.discard(None)
    if artifact_model_id and artifact_model_id not in current_model_names:
        raise NeuralImprintRuntimeError(
            "model_id_mismatch",
            "Neural Imprint artifact was generated for a different base model.",
            {
                "artifact_model_id": artifact_model_id,
                "loaded_model_id": loaded.model_id,
                "loaded_model_dir_name": Path(loaded.model_dir).name,
            },
        )

    artifact_architecture = _coerce_str(model_info.get("architecture"))
    loaded_architecture = _config_str(loaded.config, "model_type", "architectures")
    if (
        artifact_architecture is not None
        and loaded_architecture is not None
        and artifact_architecture != loaded_architecture
    ):
        raise NeuralImprintRuntimeError(
            "model_architecture_mismatch",
            "Neural Imprint model architecture does not match the loaded model.",
            {
                "artifact_model_architecture": artifact_architecture,
                "loaded_model_architecture": loaded_architecture,
            },
        )

    artifact_hidden = _coerce_int(model_info.get("hidden_size"))
    loaded_hidden = _config_int(loaded.config, "hidden_size")
    if artifact_hidden is not None and loaded_hidden is not None and artifact_hidden != loaded_hidden:
        raise NeuralImprintRuntimeError(
            "hidden_size_mismatch",
            "Neural Imprint hidden_size does not match the loaded model.",
            {"artifact_hidden_size": artifact_hidden, "loaded_hidden_size": loaded_hidden},
        )

    artifact_layers = _coerce_int(model_info.get("num_layers"))
    loaded_layers = _config_int(loaded.config, "num_hidden_layers", "num_layers", "n_layer")
    if artifact_layers is not None and loaded_layers is not None and artifact_layers != loaded_layers:
        raise NeuralImprintRuntimeError(
            "layer_count_mismatch",
            "Neural Imprint layer count does not match the loaded model.",
            {"artifact_layer_count": artifact_layers, "loaded_layer_count": loaded_layers},
        )


def _expected_metadata_from_header(header: Mapping[str, Any]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for field in _COMPATIBILITY_METADATA_FIELDS:
        value = header.get(field)
        if not isinstance(value, str) or not value:
            raise NeuralImprintRuntimeError(
                "artifact_metadata_missing",
                "Neural Imprint artifact is missing required compatibility metadata.",
                {"field": field},
            )
        expected[field] = value
    return expected


def _eval_cache_state(cache: Any) -> None:
    try:
        import mlx.core as mx
        from edgestudio_core.halo_capsule import full_cache

        arrays = [
            item
            for layer in cache
            for item in full_cache.state_items(layer)
        ]
        if arrays:
            mx.eval(*arrays)
    except Exception:
        # Restore compatibility is validated before this point. Evaluation only
        # warms the runtime; failures can surface during generation.
        return


def _ensure_edgestudio_core_importable() -> None:
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config_int(config: Mapping[str, Any], *keys: str) -> int | None:
    text_config = config.get("text_config")
    configs = [text_config, config] if isinstance(text_config, Mapping) else [config]
    for cfg in configs:
        for key in keys:
            value = cfg.get(key)
            coerced = _coerce_int(value)
            if coerced is not None:
                return coerced
    return None


def _config_str(config: Mapping[str, Any], *keys: str) -> str | None:
    text_config = config.get("text_config")
    configs = [config, text_config] if isinstance(text_config, Mapping) else [config]
    for cfg in configs:
        for key in keys:
            value = cfg.get(key)
            if isinstance(value, list) and value:
                value = value[0]
            coerced = _coerce_str(value)
            if coerced is not None:
                return coerced
    return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
