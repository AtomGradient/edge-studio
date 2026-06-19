# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persist and restore Neural Imprint full-cache state for Halo Capsule artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mlx.core as mx


FULL_CACHE_ARTIFACT_SCHEMA = "edgestudio.neural_imprint.full_cache.v2"
FULL_CACHE_METADATA_SCHEMA = "edgestudio.neural_imprint.full_cache_metadata.v2"
FULL_CACHE_ARTIFACT_TYPE = "neural_imprint"
FULL_CACHE_ARTIFACT_VERSION = "2"
PREFIX_RENDERER_VERSION = "edgestudio.neural_imprint.renderer.v1"
NEURAL_IMPRINT_METADATA_FILE_NAME = "neural_imprint_metadata.json"

REQUIRED_SAFETENSORS_METADATA_FIELDS = (
    "format",
    "artifact_type",
    "artifact_version",
    "cache_schema",
    "model_id",
    "model_architecture",
    "model_config_sha256",
    "model_weights_fingerprint",
    "tokenizer_json_sha256",
    "tokenizer_config_sha256",
    "chat_template_sha256",
    "system_prompt_sha256",
    "rendered_prefix_sha256",
    "prefix_token_ids_sha256",
    "prefix_token_count",
    "prefix_renderer_version",
    "tool_schema_sha256",
    "profile_body_sha256",
    "enable_thinking",
    "cache_backend",
    "cache_backend_version",
    "created_at",
    "created_by",
    "writer_version",
    "min_reader_version",
)

COMPATIBILITY_METADATA_FIELDS = (
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


class FullCacheCompatibilityError(ValueError):
    """Raised when a Neural Imprint artifact fails fail-closed compatibility checks."""


def state_items(cache_obj: Any) -> list[Any]:
    """Return cache state tensors while preserving the original container type."""

    state = cache_obj.state
    if isinstance(state, tuple):
        return list(state)
    if isinstance(state, list):
        return list(state)
    raise TypeError(f"unsupported cache state type {type(state).__name__}")


def _state_container(cache_obj: Any) -> str:
    cache_class = type(cache_obj).__name__
    if cache_class == "KVCache":
        return "tuple"
    if cache_class == "ArraysCache":
        return "list"
    try:
        state = cache_obj.state
    except AttributeError as exc:
        raise TypeError(f"unsupported cache state type for {cache_class}") from exc
    if isinstance(state, tuple):
        return "tuple"
    if isinstance(state, list):
        return "list"
    raise TypeError(f"unsupported cache state type {type(state).__name__}")


def _shape(tensor: Any) -> list[int]:
    return [int(dim) for dim in getattr(tensor, "shape", ())]


def _dtype(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", type(tensor).__name__))


def tensor_name(layer: int, state_index: int) -> str:
    return f"layer_{layer:02d}.state_{state_index}"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safetensors_metadata(path: Path | str) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        return dict(handle.metadata() or {})


def full_cache_manifest(cache: Sequence[Any]) -> dict[str, Any]:
    layers = []
    for layer_index, cache_obj in enumerate(cache):
        items = state_items(cache_obj)
        layers.append(
            {
                "layer": layer_index,
                "cache_class": type(cache_obj).__name__,
                "state_container": _state_container(cache_obj),
                "state_count": len(items),
                "states": [
                    {
                        "name": tensor_name(layer_index, state_index),
                        "shape": _shape(item),
                        "dtype": _dtype(item),
                    }
                    for state_index, item in enumerate(items)
                ],
                "meta_state": getattr(cache_obj, "meta_state", ""),
                "offset": getattr(cache_obj, "offset", None),
            }
        )
    return {
        "layer_count": len(layers),
        "layers": layers,
    }


def capture_full_cache(
    model: Any,
    prefix_ids: Sequence[int],
    *,
    forward_fn: Callable[..., Any],
) -> Any:
    """Create an empty model cache and prefill it with prefix tokens."""

    cache = model.make_cache()
    forward_fn(model, list(prefix_ids), cache=cache)
    return cache


def safetensors_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    merged = {
        "format": "mlx",
        "artifact_type": FULL_CACHE_ARTIFACT_TYPE,
        "artifact_version": FULL_CACHE_ARTIFACT_VERSION,
        "cache_schema": FULL_CACHE_ARTIFACT_SCHEMA,
        **dict(metadata),
    }
    _validate_safetensors_metadata(merged)
    return merged


def _validate_safetensors_metadata(metadata: Mapping[str, Any]) -> None:
    for field in REQUIRED_SAFETENSORS_METADATA_FIELDS:
        if field not in metadata:
            raise FullCacheCompatibilityError(
                f"missing safetensors metadata field: {field}"
            )
        value = metadata[field]
        if not isinstance(value, str):
            raise FullCacheCompatibilityError(
                f"safetensors metadata field must be a string: {field}"
            )
        if not value:
            raise FullCacheCompatibilityError(
                f"empty safetensors metadata field: {field}"
            )
    if metadata["format"] != "mlx":
        raise FullCacheCompatibilityError("unsupported safetensors format")
    if metadata["artifact_type"] != FULL_CACHE_ARTIFACT_TYPE:
        raise FullCacheCompatibilityError("unsupported artifact_type")
    if metadata["artifact_version"] != FULL_CACHE_ARTIFACT_VERSION:
        raise FullCacheCompatibilityError("unsupported artifact_version")
    if metadata["cache_schema"] != FULL_CACHE_ARTIFACT_SCHEMA:
        raise FullCacheCompatibilityError("unsupported cache_schema")
    if metadata["prefix_renderer_version"] != PREFIX_RENDERER_VERSION:
        raise FullCacheCompatibilityError("unsupported prefix_renderer_version")


def _require_fields(name: str, payload: Mapping[str, Any], fields: Sequence[str]) -> None:
    for field in fields:
        if field not in payload:
            raise ValueError(f"missing {name}.{field}")
        if isinstance(payload[field], str) and not payload[field]:
            raise ValueError(f"empty {name}.{field}")


def _require_equal(left_name: str, left: Any, right_name: str, right: Any) -> None:
    if left != right:
        raise ValueError(f"{left_name} does not match {right_name}")


def save_full_cache(
    *,
    artifact: Path | str,
    cache: Sequence[Any],
    metadata: Mapping[str, str],
    source: Mapping[str, Any],
    model_info: Mapping[str, Any],
    tokenizer_info: Mapping[str, Any],
    prefix_info: Mapping[str, Any],
    metadata_path: Path | str | None = None,
    save_safetensors: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    artifact = Path(artifact)
    metadata_file = Path(metadata_path) if metadata_path is not None else None
    manifest = full_cache_manifest(cache)
    tensors = {
        state["name"]: state_items(cache[layer["layer"]])[state_index]
        for layer in manifest["layers"]
        for state_index, state in enumerate(layer["states"])
    }
    header_metadata = safetensors_metadata(metadata)
    _require_fields(
        "source",
        source,
        (
            "profile_body_path",
            "profile_body_sha256",
            "tool_specs_path",
            "tool_schema_sha256",
        ),
    )
    _require_fields(
        "model",
        model_info,
        ("id", "architecture", "hidden_size", "num_layers", "quantization", "layer_types"),
    )
    _require_fields(
        "tokenizer",
        tokenizer_info,
        (
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "enable_thinking",
        ),
    )
    _require_fields(
        "prefix",
        prefix_info,
        ("token_count", "rendered_prefix_sha256", "token_ids_sha256"),
    )
    _require_equal(
        "metadata.model_id",
        header_metadata["model_id"],
        "model.id",
        model_info["id"],
    )
    _require_equal(
        "metadata.model_architecture",
        header_metadata["model_architecture"],
        "model.architecture",
        model_info["architecture"],
    )
    _require_equal(
        "metadata.tokenizer_json_sha256",
        header_metadata["tokenizer_json_sha256"],
        "tokenizer.tokenizer_json_sha256",
        tokenizer_info["tokenizer_json_sha256"],
    )
    _require_equal(
        "metadata.tokenizer_config_sha256",
        header_metadata["tokenizer_config_sha256"],
        "tokenizer.tokenizer_config_sha256",
        tokenizer_info["tokenizer_config_sha256"],
    )
    _require_equal(
        "metadata.chat_template_sha256",
        header_metadata["chat_template_sha256"],
        "tokenizer.chat_template_sha256",
        tokenizer_info["chat_template_sha256"],
    )
    _require_equal(
        "metadata.enable_thinking",
        header_metadata["enable_thinking"],
        "tokenizer.enable_thinking",
        str(tokenizer_info["enable_thinking"]).lower(),
    )
    _require_equal(
        "metadata.profile_body_sha256",
        header_metadata["profile_body_sha256"],
        "source.profile_body_sha256",
        source["profile_body_sha256"],
    )
    _require_equal(
        "metadata.tool_schema_sha256",
        header_metadata["tool_schema_sha256"],
        "source.tool_schema_sha256",
        source["tool_schema_sha256"],
    )
    _require_equal(
        "metadata.rendered_prefix_sha256",
        header_metadata["rendered_prefix_sha256"],
        "prefix.rendered_prefix_sha256",
        prefix_info["rendered_prefix_sha256"],
    )
    _require_equal(
        "metadata.prefix_token_ids_sha256",
        header_metadata["prefix_token_ids_sha256"],
        "prefix.token_ids_sha256",
        prefix_info["token_ids_sha256"],
    )
    _require_equal(
        "metadata.prefix_token_count",
        header_metadata["prefix_token_count"],
        "prefix.token_count",
        str(prefix_info["token_count"]),
    )

    artifact.parent.mkdir(parents=True, exist_ok=True)
    save = save_safetensors or mx.save_safetensors
    save(
        str(artifact),
        tensors,
        metadata=header_metadata,
    )
    artifact_hash = sha256_file(artifact)

    receipt: dict[str, Any] = {
        "schema": FULL_CACHE_METADATA_SCHEMA,
        "artifact": str(artifact),
        "artifact_sha256": artifact_hash,
        "source": dict(source),
        "model": dict(model_info),
        "tokenizer": dict(tokenizer_info),
        "prefix": dict(prefix_info),
        "cache_manifest": manifest,
    }
    receipt["artifact_size_bytes"] = artifact.stat().st_size

    if metadata_file is not None:
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return receipt


def validate_compatibility(
    *,
    cache: Sequence[Any],
    tensors: Mapping[str, Any],
    safetensors_header: Mapping[str, Any],
    sidecar_metadata: Mapping[str, Any],
    expected_metadata: Mapping[str, str],
) -> None:
    _validate_safetensors_metadata(safetensors_header)
    for field in COMPATIBILITY_METADATA_FIELDS:
        if field not in expected_metadata:
            raise FullCacheCompatibilityError(f"missing expected metadata: {field}")
        if not isinstance(expected_metadata[field], str) or not expected_metadata[field]:
            raise FullCacheCompatibilityError(f"invalid expected metadata: {field}")
        if safetensors_header[field] != expected_metadata[field]:
            raise FullCacheCompatibilityError(f"metadata mismatch: {field}")

    if sidecar_metadata.get("schema") != FULL_CACHE_METADATA_SCHEMA:
        raise FullCacheCompatibilityError("unsupported sidecar schema")
    _validate_sidecar_metadata(sidecar_metadata, safetensors_header)
    manifest = sidecar_metadata.get("cache_manifest")
    if not isinstance(manifest, Mapping):
        raise FullCacheCompatibilityError("missing cache_manifest")
    _validate_cache_manifest(cache, tensors, manifest)


def _validate_sidecar_metadata(
    sidecar_metadata: Mapping[str, Any],
    safetensors_header: Mapping[str, Any],
) -> None:
    _require_fields("sidecar", sidecar_metadata, ("artifact", "artifact_sha256"))
    source = sidecar_metadata.get("source")
    model_info = sidecar_metadata.get("model")
    tokenizer_info = sidecar_metadata.get("tokenizer")
    prefix_info = sidecar_metadata.get("prefix")
    if not isinstance(source, Mapping):
        raise FullCacheCompatibilityError("missing source metadata")
    if not isinstance(model_info, Mapping):
        raise FullCacheCompatibilityError("missing model metadata")
    if not isinstance(tokenizer_info, Mapping):
        raise FullCacheCompatibilityError("missing tokenizer metadata")
    if not isinstance(prefix_info, Mapping):
        raise FullCacheCompatibilityError("missing prefix metadata")
    _require_fields(
        "source",
        source,
        (
            "profile_body_path",
            "profile_body_sha256",
            "tool_specs_path",
            "tool_schema_sha256",
        ),
    )
    _require_fields(
        "model",
        model_info,
        ("id", "architecture", "hidden_size", "num_layers", "quantization", "layer_types"),
    )
    _require_fields(
        "tokenizer",
        tokenizer_info,
        (
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "enable_thinking",
        ),
    )
    _require_fields(
        "prefix",
        prefix_info,
        ("token_count", "rendered_prefix_sha256", "token_ids_sha256"),
    )

    comparisons = (
        ("model_id", model_info["id"]),
        ("model_architecture", model_info["architecture"]),
        ("tokenizer_json_sha256", tokenizer_info["tokenizer_json_sha256"]),
        ("tokenizer_config_sha256", tokenizer_info["tokenizer_config_sha256"]),
        ("chat_template_sha256", tokenizer_info["chat_template_sha256"]),
        ("enable_thinking", str(tokenizer_info["enable_thinking"]).lower()),
        ("profile_body_sha256", source["profile_body_sha256"]),
        ("tool_schema_sha256", source["tool_schema_sha256"]),
        ("rendered_prefix_sha256", prefix_info["rendered_prefix_sha256"]),
        ("prefix_token_ids_sha256", prefix_info["token_ids_sha256"]),
        ("prefix_token_count", str(prefix_info["token_count"])),
    )
    for field, value in comparisons:
        if safetensors_header[field] != value:
            raise FullCacheCompatibilityError(f"sidecar/header mismatch: {field}")


def _validate_cache_manifest(
    cache: Sequence[Any],
    tensors: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise FullCacheCompatibilityError("cache_manifest.layers must be a list")
    if manifest.get("layer_count") != len(cache) or len(layers) != len(cache):
        raise FullCacheCompatibilityError("cache layer_count mismatch")

    for layer_index, cache_obj in enumerate(cache):
        layer = layers[layer_index]
        if layer.get("layer") != layer_index:
            raise FullCacheCompatibilityError(f"cache layer index mismatch: {layer_index}")
        cache_class = type(cache_obj).__name__
        if layer.get("cache_class") != cache_class:
            raise FullCacheCompatibilityError(f"cache class mismatch: layer {layer_index}")
        state_container = _state_container(cache_obj)
        if layer.get("state_container") != state_container:
            raise FullCacheCompatibilityError(
                f"cache state_container mismatch: layer {layer_index}"
            )
        states = layer.get("states")
        if not isinstance(states, list) or not states:
            raise FullCacheCompatibilityError(f"missing states for layer {layer_index}")
        if layer.get("state_count") != len(states):
            raise FullCacheCompatibilityError(f"state_count mismatch: layer {layer_index}")
        for state_index, state in enumerate(states):
            name = tensor_name(layer_index, state_index)
            if state.get("name") != name:
                raise FullCacheCompatibilityError(f"tensor name mismatch: {name}")
            if name not in tensors:
                raise FullCacheCompatibilityError(f"missing tensor: {name}")
            tensor = tensors[name]
            if _shape(tensor) != state.get("shape"):
                raise FullCacheCompatibilityError(f"tensor shape mismatch: {name}")
            if _dtype(tensor) != state.get("dtype"):
                raise FullCacheCompatibilityError(f"tensor dtype mismatch: {name}")


def restore_cache_state(
    cache: Sequence[Any],
    tensors: Mapping[str, Any],
    cache_manifest: Mapping[str, Any],
) -> Sequence[Any]:
    _validate_cache_manifest(cache, tensors, cache_manifest)
    for layer_index, cache_obj in enumerate(cache):
        layer = cache_manifest["layers"][layer_index]
        items = [tensors[state["name"]] for state in layer["states"]]
        if layer["state_container"] == "tuple":
            cache_obj.state = tuple(items)
        elif layer["state_container"] == "list":
            cache_obj.state = list(items)
        else:
            raise FullCacheCompatibilityError(
                f"unsupported state_container: {layer['state_container']}"
            )
    return cache


def restore_full_cache(
    model: Any,
    artifact: Path | str,
    *,
    sidecar_metadata: Mapping[str, Any] | None = None,
    metadata_path: Path | str | None = None,
    expected_metadata: Mapping[str, str],
    load_safetensors: Callable[..., Mapping[str, Any]] | None = None,
    load_metadata: Callable[[Path | str], Mapping[str, str]] | None = None,
) -> Any:
    load = load_safetensors or mx.load
    load_header = load_metadata or load_safetensors_metadata
    header = load_header(artifact)
    if sidecar_metadata is None:
        if metadata_path is None:
            metadata_path = _default_metadata_path_for_artifact(Path(artifact))
        sidecar_metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if sidecar_metadata.get("artifact_sha256") != sha256_file(artifact):
        raise FullCacheCompatibilityError("artifact_sha256 mismatch")
    tensors = load(str(artifact))
    cache = model.make_cache()
    validate_compatibility(
        cache=cache,
        tensors=tensors,
        safetensors_header=header,
        sidecar_metadata=sidecar_metadata,
        expected_metadata=expected_metadata,
    )
    restore_cache_state(cache, tensors, sidecar_metadata["cache_manifest"])
    return cache


def _default_metadata_path_for_artifact(artifact: Path) -> Path:
    return artifact.with_name(NEURAL_IMPRINT_METADATA_FILE_NAME)
