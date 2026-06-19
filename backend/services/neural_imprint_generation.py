# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-side Neural Imprint generation jobs.

This service is the Mac workbench half of A3.2: take a stored persona source
upload, render the combined persona/tool prefix, capture the MLX full cache via
edgestudio_core, and write the resulting artifact into the local registry root.
It intentionally does not push capsules to devices.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import logging
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.services.mlx_runtime_gate import mlx_runtime_gate
from backend.services.neural_imprint_artifact_registry import (
    default_neural_imprint_artifact_roots,
)
from backend.services.persona_source_store import (
    StoredPersonaSource,
    latest_persona_source_for_peer,
)

logger = logging.getLogger(__name__)

GENERATION_JOB_SCHEMA_VERSION = "edgestudio.neural_imprint_generation_job.v2"
GENERATION_RESULT_SCHEMA_VERSION = "edgestudio.neural_imprint_generation_result.v2"
GENERATION_RECEIPT_SCHEMA_VERSION = "edgestudio.neural_imprint_generation_receipt.v2"
PREFIX_RENDERER_VERSION = "edgestudio.neural_imprint.renderer.v1"
NEURAL_IMPRINT_ARTIFACT_NAME = "neural_imprint.safetensors"
NEURAL_IMPRINT_METADATA_NAME = "neural_imprint_metadata.json"
NEURAL_IMPRINT_GENERATION_RECEIPT_NAME = "neural_imprint_generation_receipt.json"
NEURAL_IMPRINT_FACT_INTENT_TAGS = frozenset({"exact_fact", "aggregate_fact"})
NEURAL_IMPRINT_FACT_PERMISSIONS = frozenset({"read_facts"})
NEURAL_IMPRINT_WRITE_PERMISSIONS = frozenset({"write_facts", "write_profile"})
NEURAL_IMPRINT_WRITE_NAME_PREFIXES = (
    "create_",
    "delete_",
    "insert_",
    "set_",
    "update_",
    "upsert_",
)
_JOB_TTL_SECONDS = 6 * 60 * 60

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


@dataclass
class NeuralImprintGenerationError(ValueError):
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
class NeuralImprintGenerationSpec:
    job_id: str
    peer_id: str
    model_dir: Path
    model_id: str
    source: StoredPersonaSource
    output_dir: Path
    validate_restore: bool = False


@dataclass(frozen=True)
class NeuralImprintGenerationResult:
    artifact_dir: Path
    artifact_path: Path
    metadata_path: Path
    receipt_path: Path
    peer_id: str
    source_id: str
    model_id: str
    prefix_token_count: int
    artifact_sha256: str
    tool_schema_sha256: str
    profile_body_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_RESULT_SCHEMA_VERSION,
            "artifact_dir": str(self.artifact_dir),
            "artifact_path": str(self.artifact_path),
            "metadata_path": str(self.metadata_path),
            "receipt_path": str(self.receipt_path),
            "peer_id": self.peer_id,
            "source_id": self.source_id,
            "model_id": self.model_id,
            "prefix_token_count": self.prefix_token_count,
            "artifact_sha256": self.artifact_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "profile_body_sha256": self.profile_body_sha256,
        }


@dataclass
class _GenerationJob:
    job_id: str
    peer_id: str
    source_id: str
    model_dir: Path
    model_id: str
    output_dir: Path
    validate_restore: bool
    status: str = "queued"
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GENERATION_JOB_SCHEMA_VERSION,
            "job_id": self.job_id,
            "status": self.status,
            "peer_id": self.peer_id,
            "source_id": self.source_id,
            "model_dir": str(self.model_dir),
            "model_id": self.model_id,
            "output_dir": str(self.output_dir),
            "validate_restore": self.validate_restore,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": copy.deepcopy(self.result),
            "error": copy.deepcopy(self.error),
        }


class _NeuralImprintGenerationJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, _GenerationJob] = {}
        # MLX runtime access must be serialized and run on a persistent worker
        # thread to avoid overlapping Metal work and MLX TLS teardown crashes.
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="neural-imprint-generation",
        )

    def enqueue(self, spec: NeuralImprintGenerationSpec) -> dict[str, Any]:
        now = time.time()
        job = _GenerationJob(
            job_id=spec.job_id,
            peer_id=spec.peer_id,
            source_id=spec.source.source_id,
            model_dir=spec.model_dir,
            model_id=spec.model_id,
            output_dir=spec.output_dir,
            validate_restore=spec.validate_restore,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._cleanup_locked(now)
            self._jobs[job.job_id] = job
        self._worker.submit(self._run, spec)
        return job.to_dict()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(str(job_id).strip())
            return job.to_dict() if job else None

    def reset_for_tests(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _run(self, spec: NeuralImprintGenerationSpec) -> None:
        self._mark_running(spec.job_id)
        try:
            result = generate_neural_imprint_artifact(spec)
            self._complete(spec.job_id, result.to_dict())
        except NeuralImprintGenerationError as exc:
            self._fail(spec.job_id, exc.to_error())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Neural Imprint generation job failed: %s", spec.job_id)
            self._fail(
                spec.job_id,
                {
                    "code": "generation_failed",
                    "message": str(exc),
                    "retryable": False,
                    "details": {},
                },
            )

    def _mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            now = time.time()
            job.status = "running"
            job.started_at = now
            job.updated_at = now

    def _complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            now = time.time()
            job.status = "succeeded"
            job.result = result
            job.completed_at = now
            job.updated_at = now

    def _fail(self, job_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            now = time.time()
            job.status = "failed"
            job.error = error
            job.completed_at = now
            job.updated_at = now

    def _cleanup_locked(self, now: float) -> None:
        stale = [
            job_id
            for job_id, job in self._jobs.items()
            if job.completed_at is not None and now - job.completed_at > _JOB_TTL_SECONDS
        ]
        for job_id in stale:
            del self._jobs[job_id]


_JOB_MANAGER = _NeuralImprintGenerationJobManager()


def enqueue_neural_imprint_generation(
    *,
    peer_id: str,
    model_dir: str | Path,
    model_id: str | None = None,
    validate_restore: bool = False,
) -> dict[str, Any]:
    clean_peer_id = _required_text(peer_id, "peer_id")
    source = latest_persona_source_for_peer(clean_peer_id)
    if source is None:
        raise NeuralImprintGenerationError(
            "persona_source_not_found",
            f"persona source for peer {clean_peer_id} not found",
            {"peer_id": clean_peer_id},
        )
    _validate_generation_source(source)

    resolved_model_dir = _resolve_model_dir(model_dir)
    clean_model_id = _required_text(model_id, "model_id") if model_id else (
        str(source.payload.get("base_model_id") or "").strip() or resolved_model_dir.name
    )
    job_id = f"neural_imprint_gen_{uuid.uuid4().hex[:16]}"
    output_dir = _default_output_dir(
        model_id=clean_model_id,
        source_id=source.source_id,
        job_id=job_id,
    )
    spec = NeuralImprintGenerationSpec(
        job_id=job_id,
        peer_id=clean_peer_id,
        model_dir=resolved_model_dir,
        model_id=clean_model_id,
        source=source,
        output_dir=output_dir,
        validate_restore=bool(validate_restore),
    )
    return _JOB_MANAGER.enqueue(spec)


def get_neural_imprint_generation_job(job_id: str) -> dict[str, Any] | None:
    return _JOB_MANAGER.get(job_id)


def reset_neural_imprint_generation_jobs_for_tests() -> None:
    _JOB_MANAGER.reset_for_tests()


def generate_neural_imprint_artifact(
    spec: NeuralImprintGenerationSpec,
) -> NeuralImprintGenerationResult:
    _ensure_edgestudio_core_importable()

    from edgestudio_core.halo_capsule import full_cache
    from mlx_lm.utils import load

    payload = spec.source.payload
    profile_body = _profile_body(payload)
    source_tool_schema_sha256 = _required_hash(payload, "tool_schema_sha256")
    tool_schema_export = _neural_imprint_tool_schema_export(payload)
    tools = _tools_list(tool_schema_export)
    tool_schema_sha256 = _sha256_json(tool_schema_export)
    profile_body_sha256 = _required_hash(payload, "profile_body_sha256")

    output_dir = spec.output_dir.expanduser().resolve()
    artifact_path = output_dir / NEURAL_IMPRINT_ARTIFACT_NAME
    metadata_path = output_dir / NEURAL_IMPRINT_METADATA_NAME
    generation_receipt_path = output_dir / NEURAL_IMPRINT_GENERATION_RECEIPT_NAME
    profile_body_path = output_dir / "profile_body.txt"
    tool_specs_path = output_dir / "tool_specs.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_body_path.write_text(profile_body, encoding="utf-8")
    tool_specs_path.write_text(_pretty_json(tool_schema_export), encoding="utf-8")

    created_at = datetime.now(UTC).isoformat()
    model: Any | None = None
    tokenizer: Any | None = None
    cache: Any | None = None
    try:
        with mlx_runtime_gate("neural_imprint_generation.capture"):
            model, tokenizer = load(str(spec.model_dir))
            system_prompt = _build_system_prompt(profile_body)
            rendered_prefix, prefix_token_ids = _render_combined_prefix(
                tokenizer,
                tools=tools,
                profile_body=profile_body,
            )
            cache = full_cache.capture_full_cache(
                model,
                prefix_token_ids,
                forward_fn=_forward_last_logits,
            )
            metadata, source, model_info, tokenizer_info, prefix_info = _contract_inputs(
                model_dir=spec.model_dir,
                tokenizer=tokenizer,
                model_id=spec.model_id,
                profile_body_sha256=profile_body_sha256,
                tool_schema_sha256=tool_schema_sha256,
                system_prompt=system_prompt,
                rendered_prefix=rendered_prefix,
                prefix_token_ids=prefix_token_ids,
                created_at=created_at,
            )
            receipt = full_cache.save_full_cache(
                artifact=artifact_path,
                cache=cache,
                metadata=metadata,
                source=source,
                model_info=model_info,
                tokenizer_info=tokenizer_info,
                prefix_info=prefix_info,
                metadata_path=metadata_path,
            )
            if spec.validate_restore:
                full_cache.restore_full_cache(
                    model,
                    artifact_path,
                    metadata_path=metadata_path,
                    expected_metadata=_expected_metadata(metadata),
                )
    finally:
        del cache
        del tokenizer
        del model
        _clear_mlx_cache()

    generation_receipt = {
        "schema_version": GENERATION_RECEIPT_SCHEMA_VERSION,
        "job_id": spec.job_id,
        "peer_id": spec.peer_id,
        "source_id": spec.source.source_id,
        "source_sha256": spec.source.receipt.get("source_sha256"),
        "source_kind": spec.source.payload.get("source_kind"),
        "app_id": spec.source.receipt.get("app_id"),
        "source_receipt": _source_receipt_summary(spec.source.receipt),
        "model_id": spec.model_id,
        "model_dir": str(spec.model_dir),
        "output_dir": str(output_dir),
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "created_at": created_at,
        "validate_restore": spec.validate_restore,
        "prefix": prefix_info,
        "artifact_sha256": receipt["artifact_sha256"],
        "artifact_size_bytes": receipt["artifact_size_bytes"],
        "tool_schema_mode": "fact_tools_no_profile_tool",
        "source_tool_schema_sha256": source_tool_schema_sha256,
        "artifact_tool_schema_sha256": tool_schema_sha256,
        "boundary": "generated locally on Mac; not pushed automatically",
    }
    generation_receipt_path.write_text(
        _pretty_json(generation_receipt),
        encoding="utf-8",
    )

    return NeuralImprintGenerationResult(
        artifact_dir=output_dir,
        artifact_path=artifact_path,
        metadata_path=metadata_path,
        receipt_path=generation_receipt_path,
        peer_id=spec.peer_id,
        source_id=spec.source.source_id,
        model_id=spec.model_id,
        prefix_token_count=int(prefix_info["token_count"]),
        artifact_sha256=str(receipt["artifact_sha256"]),
        tool_schema_sha256=tool_schema_sha256,
        profile_body_sha256=profile_body_sha256,
    )


def _validate_generation_source(source: StoredPersonaSource) -> None:
    payload = source.payload
    _profile_body(payload)
    _required_hash(payload, "tool_schema_sha256")
    _tools_list(_neural_imprint_tool_schema_export(payload))
    _required_hash(payload, "profile_body_sha256")


def _resolve_model_dir(model_dir: str | Path) -> Path:
    raw = _required_text(str(model_dir), "model_dir")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise NeuralImprintGenerationError(
            "model_dir_not_found",
            f"model_dir not found: {path}",
            {"model_dir": str(path)},
        )
    if not path.is_dir():
        raise NeuralImprintGenerationError(
            "model_dir_not_directory",
            f"model_dir is not a directory: {path}",
            {"model_dir": str(path)},
        )
    return path


def _default_output_dir(*, model_id: str, source_id: str, job_id: str) -> Path:
    roots = default_neural_imprint_artifact_roots()
    if not roots:
        raise NeuralImprintGenerationError(
            "artifact_registry_root_missing",
            "Neural Imprint artifact registry root is not configured",
            {},
        )
    return (
        roots[0].expanduser().resolve()
        / _path_component(model_id)
        / _path_component(source_id)
        / _path_component(job_id)
    )


def _build_system_prompt(profile_body: str) -> str:
    return profile_body.strip()


def _render_combined_prefix(
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]],
    profile_body: str,
) -> tuple[str, list[int]]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise NeuralImprintGenerationError(
            "tokenizer_missing_chat_template",
            "tokenizer does not support apply_chat_template",
            {},
        )
    messages = [
        {"role": "system", "content": _build_system_prompt(profile_body)},
        {"role": "user", "content": "__PERSONA_TOOLS_SPLIT_SENTINEL__"},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
    if not isinstance(rendered, str) or not rendered:
        raise NeuralImprintGenerationError(
            "prefix_render_failed",
            "chat template did not return a rendered string",
            {},
        )
    marker = "<|im_start|>user\n"
    marker_index = rendered.find(marker)
    if marker_index < 0:
        raise NeuralImprintGenerationError(
            "prefix_split_marker_missing",
            "rendered chat template did not contain the Qwen user marker",
            {"marker": marker},
        )
    prefix = rendered[:marker_index]
    token_ids = _encode_text(tokenizer, prefix)
    if not token_ids:
        raise NeuralImprintGenerationError(
            "prefix_token_ids_empty",
            "rendered prefix produced no token ids",
            {},
        )
    return prefix, token_ids


def _forward_last_logits(model: Any, token_ids: Sequence[int], cache: Any = None) -> Any:
    import mlx.core as mx

    arr = mx.array(list(token_ids), dtype=mx.int32)[None, :]
    out = model(arr, cache=cache) if cache is not None else model(arr)
    logits = out[0] if isinstance(out, tuple) else out
    last = logits[:, -1, :].astype(mx.float32)
    mx.eval(last)
    return last


def _contract_inputs(
    *,
    model_dir: Path,
    tokenizer: Any,
    model_id: str,
    profile_body_sha256: str,
    tool_schema_sha256: str,
    system_prompt: str,
    rendered_prefix: str,
    prefix_token_ids: Sequence[int],
    created_at: str,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_info = _model_contract_info(model_dir, model_id=model_id)
    tokenizer_info = {
        "tokenizer_json_sha256": _sha256_file(model_dir / "tokenizer.json"),
        "tokenizer_config_sha256": _sha256_file(model_dir / "tokenizer_config.json"),
        "chat_template_sha256": _sha256_text(_chat_template_text(tokenizer, model_dir)),
        "enable_thinking": False,
    }
    source = {
        "profile_body_path": "profile_body.txt",
        "profile_body_sha256": profile_body_sha256,
        "tool_specs_path": "tool_specs.json",
        "tool_schema_sha256": tool_schema_sha256,
    }
    prefix = {
        "token_count": len(prefix_token_ids),
        "rendered_prefix_sha256": _sha256_text(rendered_prefix),
        "token_ids_sha256": _token_ids_sha256(prefix_token_ids),
    }
    metadata = {
        "model_id": model_info["id"],
        "model_architecture": str(model_info["architecture"]),
        "model_config_sha256": _sha256_file(model_dir / "config.json"),
        "model_weights_fingerprint": _model_weights_fingerprint(model_dir),
        "tokenizer_json_sha256": tokenizer_info["tokenizer_json_sha256"],
        "tokenizer_config_sha256": tokenizer_info["tokenizer_config_sha256"],
        "chat_template_sha256": tokenizer_info["chat_template_sha256"],
        "system_prompt_sha256": _sha256_text(system_prompt),
        "rendered_prefix_sha256": prefix["rendered_prefix_sha256"],
        "prefix_token_ids_sha256": prefix["token_ids_sha256"],
        "prefix_token_count": str(prefix["token_count"]),
        "prefix_renderer_version": PREFIX_RENDERER_VERSION,
        "tool_schema_sha256": source["tool_schema_sha256"],
        "profile_body_sha256": source["profile_body_sha256"],
        "enable_thinking": "false",
        "cache_backend": "mlx-lm",
        "cache_backend_version": _cache_backend_version(),
        "created_at": created_at,
                "created_by": "EdgeStudio neural_imprint_generation",
        "writer_version": f"edgestudio_core {_edgestudio_core_version()}",
        "min_reader_version": f"edgestudio_core {_edgestudio_core_version()}",
    }
    return metadata, source, model_info, tokenizer_info, prefix


def _model_contract_info(model_dir: Path, *, model_id: str) -> dict[str, Any]:
    config = _read_json_object(model_dir / "config.json")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    hidden_size = text_config.get("hidden_size") or config.get("hidden_size")
    num_layers = (
        text_config.get("num_hidden_layers")
        or config.get("num_hidden_layers")
        or config.get("n_layers")
    )
    if hidden_size is None or num_layers is None:
        raise NeuralImprintGenerationError(
            "invalid_model_config",
            "config.json must include hidden_size and num_hidden_layers",
            {"model_dir": str(model_dir)},
        )
    layer_types = text_config.get("layer_types") or config.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        layer_types = ["unknown"] * int(num_layers)
    return {
        "id": model_id,
        "architecture": config.get("model_type") or text_config.get("model_type") or "unknown",
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "quantization": config.get("quantization") or config.get("quantization_config") or {},
        "layer_types": list(layer_types),
    }


def _model_weight_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_json_object(index_path)
        names = sorted(set(index.get("weight_map", {}).values()))
        return [model_dir / str(name) for name in names]
    return sorted(model_dir.glob("model*.safetensors"))


def _model_weights_fingerprint(model_dir: Path) -> str:
    index_path = model_dir / "model.safetensors.index.json"
    payload = {
        "index_sha256": _sha256_file(index_path) if index_path.exists() else None,
        "shards": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in _model_weight_files(model_dir)
        ],
    }
    return "sha256:" + _sha256_json(payload)


def _chat_template_text(tokenizer: Any, model_dir: Path) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return str(template)
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = _read_json_object(tokenizer_config_path)
        config_template = tokenizer_config.get("chat_template")
        if isinstance(config_template, str) and config_template:
            return config_template
    template_path = model_dir / "chat_template.jinja"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    raise NeuralImprintGenerationError(
        "chat_template_not_found",
        "chat template not found on tokenizer or model dir",
        {"model_dir": str(model_dir)},
    )


def _expected_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    return {field: metadata[field] for field in _COMPATIBILITY_METADATA_FIELDS}


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    return [int(token) for token in tokens]


def _tool_schema_export(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("tool_schema_export")
    if not isinstance(value, dict):
        raise NeuralImprintGenerationError(
            "invalid_tool_schema_export",
            "tool_schema_export must be an object",
            {"type": type(value).__name__},
        )
    return value


def _neural_imprint_tool_schema_export(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _fact_tool_schema_export(_tool_schema_export(payload))


def _fact_tool_schema_export(tool_schema_export: Mapping[str, Any]) -> dict[str, Any]:
    tools = _tools_list(tool_schema_export)
    selected = [
        tool
        for tool in tools
        if _is_neural_imprint_fact_tool(tool)
    ]
    if not selected:
        raise NeuralImprintGenerationError(
            "missing_fact_tools",
            "Neural Imprint generation requires at least one fact tool",
            {
                "accepted_permissions": sorted(NEURAL_IMPRINT_FACT_PERMISSIONS),
                "accepted_intent_tags": sorted(NEURAL_IMPRINT_FACT_INTENT_TAGS),
            },
        )
    return {
        "schema_version": str(
            tool_schema_export.get("schema_version")
            or "edgestudio.tool_schema_export.v1"
        ),
        "tools": [dict(tool) for tool in selected],
    }


def _is_neural_imprint_fact_tool(tool: Mapping[str, Any]) -> bool:
    name = _tool_name(tool)
    if not name:
        return False
    lowered_name = name.lower()
    permissions = _text_set(tool.get("permissions"))
    if permissions & NEURAL_IMPRINT_WRITE_PERMISSIONS:
        return False
    if any(lowered_name.startswith(prefix) for prefix in NEURAL_IMPRINT_WRITE_NAME_PREFIXES):
        return False

    intent_tags = _text_set(tool.get("intentTags")) | _text_set(
        tool.get("intent_tags")
    )
    if permissions & NEURAL_IMPRINT_FACT_PERMISSIONS:
        return True
    if intent_tags & NEURAL_IMPRINT_FACT_INTENT_TAGS:
        return True
    return False


def _tool_name(tool: Mapping[str, Any]) -> str | None:
    name = tool.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    function = tool.get("function")
    if isinstance(function, Mapping):
        function_name = function.get("name")
        if isinstance(function_name, str) and function_name.strip():
            return function_name.strip()
    return None


def _tools_list(tool_schema_export: Mapping[str, Any]) -> list[dict[str, Any]]:
    tools = tool_schema_export.get("tools")
    if not isinstance(tools, list) or not tools:
        raise NeuralImprintGenerationError(
            "missing_tools",
            "tool_schema_export.tools must be a non-empty list",
            {},
        )
    if not all(isinstance(item, dict) for item in tools):
        raise NeuralImprintGenerationError(
            "invalid_tools",
            "tool_schema_export.tools must contain objects",
            {},
        )
    return [dict(item) for item in tools]


def _text_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if not isinstance(value, Sequence) or isinstance(
        value, (bytes, bytearray)
    ):
        return set()
    return {str(item).strip() for item in value if str(item).strip()}


def _profile_body(payload: Mapping[str, Any]) -> str:
    value = payload.get("profile_body")
    if not isinstance(value, str) or not value.strip():
        raise NeuralImprintGenerationError(
            "missing_profile_body",
            "profile_body is required to generate a full Neural Imprint artifact",
            {},
        )
    return value


def _source_receipt_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "status",
        "peer_id",
        "source_id",
        "source_sha256",
        "source_kind",
        "app_id",
        "base_model_id",
        "tool_schema_sha256",
        "profile_body_sha256",
        "rpp_run_id",
        "created_at",
        "received_at_ms",
    )
    return {key: receipt.get(key) for key in keys if key in receipt}


def _ensure_edgestudio_core_importable() -> None:
    return None


def _edgestudio_core_version() -> str:
    _ensure_edgestudio_core_importable()
    try:
        import edgestudio_core

        return str(getattr(edgestudio_core, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def _cache_backend_version() -> str:
    try:
        return importlib.metadata.version("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Neural Imprint generation: mlx cache clear skipped: %s", exc)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NeuralImprintGenerationError(
            "required_file_missing",
            f"required file missing: {path.name}",
            {"path": str(path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise NeuralImprintGenerationError(
            "invalid_json_file",
            f"invalid JSON file: {path.name}",
            {"path": str(path)},
        ) from exc
    if not isinstance(value, dict):
        raise NeuralImprintGenerationError(
            "invalid_json_file",
            f"JSON file must contain an object: {path.name}",
            {"path": str(path)},
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise NeuralImprintGenerationError(
            "required_file_missing",
            f"required file missing: {path.name}",
            {"path": str(path)},
        ) from exc
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return _sha256_json([int(token) for token in token_ids])


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise NeuralImprintGenerationError(
            "missing_required_field",
            f"{name} is required",
            {"field": name},
        )
    return text


def _required_hash(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise NeuralImprintGenerationError(
            "invalid_hash",
            f"{key} must be a sha256 hex string",
            {"field": key},
        )
    return value


def _path_component(value: str) -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not safe:
        raise NeuralImprintGenerationError(
            "invalid_path_component",
            "path component is empty after sanitization",
            {"value": text},
        )
    return safe[:120]
