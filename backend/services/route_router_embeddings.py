# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Frozen embedding extraction helpers for R2.1 route-router training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.services.route_router_training import (
    PrecomputedRouteRouterEmbeddingProvider,
    RouteRouterEmbeddingProvider,
    RouteRouterTrainConfig,
    load_route_action_policy_dataset_jsonl,
    train_route_router_matrix_artifacts,
)

ROUTE_ROUTER_EMBEDDING_CACHE_NAME = "route_router_embeddings.jsonl"
ROUTE_ROUTER_EMBEDDING_CACHE_SCHEMA_VERSION = (
    "edgestudio.route_router_embedding_cache.v0"
)


@dataclass(frozen=True)
class RouteRouterEmbeddingCacheConfig:
    dataset_path: Path
    output_dir: Path
    hidden_size: int
    training_run_id: str
    base_model_id: str
    tokenizer_sha256: str
    layer_index: int = -1
    tool_descriptors: dict[str, str] | None = None


class MlxLastHiddenEmbeddingProvider:
    """Route-router embedding provider backed by an MLX LM backbone.

    The loaded model must expose the transformer backbone as `model.model`.
    Calling the top-level model usually returns logits, which are rejected by
    the hidden-size check below.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        hidden_size: int,
        layer_index: int = -1,
        mx_module: Any | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = int(hidden_size)
        self.layer_index = int(layer_index)
        self._mx_module = mx_module

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str,
        *,
        hidden_size: int,
        layer_index: int = -1,
    ) -> "MlxLastHiddenEmbeddingProvider":
        from mlx_lm.utils import load

        model, tokenizer = load(str(model_dir))
        return cls(
            model=model,
            tokenizer=tokenizer,
            hidden_size=hidden_size,
            layer_index=layer_index,
        )

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        import numpy as np

        tokens = _encode_tokens(self.tokenizer, text)
        if not tokens:
            raise ValueError("cannot embed empty token sequence")
        mx = self._mx()
        dtype = getattr(mx, "int32", None)
        input_ids = (
            mx.array([tokens], dtype=dtype)
            if dtype is not None
            else mx.array([tokens])
        )
        hidden = self._last_hidden(input_ids)
        float32 = getattr(mx, "float32", None)
        if float32 is not None and hasattr(hidden, "astype"):
            hidden = hidden.astype(float32)
        if hasattr(mx, "eval"):
            mx.eval(hidden)
        array = np.asarray(hidden, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            raise ValueError(f"last hidden state must be rank-2/3, got {array.ndim}")
        if array.shape[-1] != self.hidden_size:
            raise ValueError(
                "last hidden size mismatch: "
                f"expected {self.hidden_size}, got {array.shape[-1]}"
            )
        mask = _non_special_token_mask(tokens, self.tokenizer)
        if len(mask) != array.shape[0] or not mask.any():
            mask = np.ones((array.shape[0],), dtype=bool)
        pooled = array[mask].mean(axis=0)
        return [float(value) for value in pooled.tolist()]

    def _last_hidden(self, input_ids: Any) -> Any:
        backbone = self._backbone()
        if backbone is None or not callable(backbone):
            raise ValueError(
                "mlx model does not expose a callable hidden-state backbone"
            )
        if self.layer_index >= 0:
            return self._layer_hidden(backbone, input_ids, self.layer_index)
        output = backbone(input_ids)
        if hasattr(output, "last_hidden_state"):
            output = output.last_hidden_state
        elif hasattr(output, "hidden_states"):
            output = output.hidden_states[-1]
        elif isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def _layer_hidden(self, backbone: Any, input_ids: Any, layer_index: int) -> Any:
        from backend.core.hidden_states import HiddenStatesUnavailable, layer_hidden

        try:
            return layer_hidden(backbone, input_ids, layer_index)
        except HiddenStatesUnavailable as exc:
            raise ValueError(str(exc)) from exc

    def _backbone(self) -> Any:
        for path in (
            ("model",),
            ("language_model", "model"),
            ("language_model",),
            ("transformer",),
        ):
            candidate = self.model
            for attr in path:
                candidate = getattr(candidate, attr, None)
                if candidate is None:
                    break
            if callable(candidate):
                return candidate
        return None

    def _mx(self) -> Any:
        if self._mx_module is not None:
            return self._mx_module
        import mlx.core as mx

        return mx


def extract_route_router_embedding_cache(
    *,
    config: RouteRouterEmbeddingCacheConfig,
    embedding_provider: RouteRouterEmbeddingProvider,
) -> dict[str, Any]:
    samples = load_route_action_policy_dataset_jsonl(config.dataset_path)
    texts = collect_route_router_embedding_texts(
        samples=samples,
        tool_descriptors=config.tool_descriptors or {},
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / ROUTE_ROUTER_EMBEDDING_CACHE_NAME
    return write_route_router_embedding_cache(
        path=cache_path,
        texts=texts,
        embedding_provider=embedding_provider,
        hidden_size=config.hidden_size,
        metadata={
            "training_run_id": config.training_run_id,
            "base_model_id": config.base_model_id,
            "tokenizer_sha256": config.tokenizer_sha256,
            "layer_index": config.layer_index,
        },
    )


def extract_and_train_route_router_matrix_artifacts(
    *,
    dataset_path: Path,
    adapter_dir: Path,
    config: RouteRouterTrainConfig,
    embedding_provider: RouteRouterEmbeddingProvider,
    tool_descriptors: dict[str, str] | None = None,
    tool_route_intents: dict[str, list[str]] | None = None,
    write_runtime_contract: bool = False,
) -> dict[str, Any]:
    training_dir = Path(adapter_dir) / "_training"
    cache_receipt = extract_route_router_embedding_cache(
        config=RouteRouterEmbeddingCacheConfig(
            dataset_path=dataset_path,
            output_dir=training_dir,
            hidden_size=config.hidden_size,
            training_run_id=config.training_run_id,
            base_model_id=config.base_model_id,
            tokenizer_sha256=config.tokenizer_sha256,
            layer_index=config.encoder_layer_index,
            tool_descriptors=tool_descriptors or {},
        ),
        embedding_provider=embedding_provider,
    )
    cache_provider = PrecomputedRouteRouterEmbeddingProvider.from_jsonl(
        Path(cache_receipt["path"]),
        hidden_size=config.hidden_size,
    )
    train_receipt = train_route_router_matrix_artifacts(
        dataset_path=dataset_path,
        adapter_dir=adapter_dir,
        config=config,
        embedding_provider=cache_provider,
        tool_descriptors=tool_descriptors or {},
        tool_route_intents=tool_route_intents or {},
        write_runtime_contract=write_runtime_contract,
    )
    return {
        "ok": cache_receipt["ok"] is True and train_receipt["ok"] is True,
        "schema_version": "edgestudio.route_router_extract_and_train.v0",
        "status": train_receipt["status"],
        "training_run_id": config.training_run_id,
        "embedding_cache": cache_receipt,
        "training": train_receipt,
    }


def collect_route_router_embedding_texts(
    *,
    samples: list[dict[str, Any]],
    tool_descriptors: dict[str, str] | None = None,
) -> list[str]:
    texts = {
        _sample_text(sample)
        for sample in samples
    }
    texts.update(
        str(description).strip()
        for description in (tool_descriptors or {}).values()
        if str(description or "").strip()
    )
    return sorted(texts, key=lambda text: (_sha256(text), text))


def build_route_router_tool_descriptors(
    *,
    tool_registry: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]] | None = None,
    max_examples_per_tool: int = 3,
) -> dict[str, str]:
    """Build stable open-vocab tool descriptor text for matrix routing.

    Tool descriptions are app-provided runtime metadata. Golden examples should
    be developer-reviewed, entity-free prompts, not harvested user facts.
    """

    examples_by_tool = _golden_examples_by_tool(
        golden_cases or [],
        max_examples_per_tool=max_examples_per_tool,
    )
    descriptors: dict[str, str] = {}
    for tool in sorted(
        tool_registry,
        key=lambda item: _tool_name(item) if isinstance(item, dict) else "",
    ):
        if not isinstance(tool, dict):
            continue
        name = _tool_name(tool)
        if not name:
            continue
        sections = [f"tool: {name}"]
        description = _first_text(
            tool,
            ("description", "summary", "purpose"),
        )
        if description:
            sections.append(f"description: {description}")
        arg_summary = _tool_arg_summary(tool)
        if arg_summary:
            sections.append(f"arguments: {arg_summary}")
        examples = _tool_registry_examples(tool)
        examples.extend(examples_by_tool.get(name, []))
        examples = _dedupe_strings(examples)[:max(0, int(max_examples_per_tool))]
        if examples:
            sections.append("examples: " + " | ".join(examples))
        descriptors[name] = "\n".join(sections)
    return descriptors


def build_route_router_tool_route_intents(
    *,
    tool_registry: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Build tool -> allowed route-intent labels from app registry metadata."""

    out: dict[str, list[str]] = {}
    for tool in sorted(
        tool_registry,
        key=lambda item: _tool_name(item) if isinstance(item, dict) else "",
    ):
        if not isinstance(tool, dict):
            continue
        name = _tool_name(tool)
        if not name:
            continue
        intents = _tool_route_intents(tool)
        if intents:
            out[name] = intents
    return out


def write_route_router_embedding_cache(
    *,
    path: Path,
    texts: list[str],
    embedding_provider: RouteRouterEmbeddingProvider,
    hidden_size: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unique_texts = sorted(set(texts), key=lambda text: (_sha256(text), text))
    embeddings = embedding_provider.encode_texts(unique_texts)
    if len(embeddings) != len(unique_texts):
        raise ValueError(
            "route-router embedding count mismatch: "
            f"expected={len(unique_texts)} got={len(embeddings)}"
        )
    rows = []
    for text, embedding in zip(unique_texts, embeddings):
        vector = [float(value) for value in embedding]
        if len(vector) != int(hidden_size):
            raise ValueError(
                "route-router embedding shape mismatch: "
                f"text={text!r} expected={hidden_size} got={len(vector)}"
            )
        rows.append({
            "schema_version": ROUTE_ROUTER_EMBEDDING_CACHE_SCHEMA_VERSION,
            "text": text,
            "embedding": vector,
            "embedding_sha256": _embedding_sha256(vector),
            "metadata": dict(metadata or {}),
        })
    data = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in rows
        )
        + ("\n" if rows else "")
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "ok": True,
        "schema_version": ROUTE_ROUTER_EMBEDDING_CACHE_SCHEMA_VERSION,
        "status": "written",
        "path": str(path),
        "text_count": len(unique_texts),
        "hidden_size": int(hidden_size),
        "sha256": hashlib.sha256(data).hexdigest(),
        "metadata": dict(metadata or {}),
    }


def _encode_tokens(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    else:
        raw = getattr(tokenizer, "_tokenizer", tokenizer)
        tokens = raw.encode(text)
    if hasattr(tokens, "ids"):
        tokens = tokens.ids
    return [int(token) for token in tokens]


def _non_special_token_mask(tokens: list[int], tokenizer: Any) -> Any:
    import numpy as np

    special_ids = {
        int(token_id)
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id")
        for token_id in [getattr(tokenizer, attr, None)]
        if token_id is not None
    }
    if not special_ids:
        return np.ones((len(tokens),), dtype=bool)
    return np.asarray(
        [int(token) not in special_ids for token in tokens],
        dtype=bool,
    )


def _sample_text(sample: dict[str, Any]) -> str:
    payload = sample.get("input") if isinstance(sample.get("input"), dict) else {}
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("route-router sample input.text must be non-empty")
    return text


def _golden_examples_by_tool(
    cases: list[dict[str, Any]],
    *,
    max_examples_per_tool: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        prompt = _first_text(case, ("prompt", "source_prompt", "query"))
        if not prompt:
            continue
        for tool in _case_selected_tools(case):
            bucket = out.setdefault(tool, [])
            if len(bucket) < max(0, int(max_examples_per_tool)):
                bucket.append(prompt)
    return {
        tool: _dedupe_strings(examples)[:max(0, int(max_examples_per_tool))]
        for tool, examples in out.items()
    }


def _case_selected_tools(case: dict[str, Any]) -> list[str]:
    tools = _string_list(
        case.get("selected_tools")
        or case.get("selectedTools")
        or case.get("tools")
    )
    expectations = case.get("expectations")
    if isinstance(expectations, dict):
        tools.extend(_string_list(expectations.get("selected_tools_exact")))
        tools.extend(_string_list(expectations.get("selected_tools_include")))
    return _dedupe_strings(tools)


def _tool_registry_examples(tool: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("example_prompts", "examples"):
        raw = tool.get(key)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    values.append(_first_text(item, ("prompt", "query", "text")))
                else:
                    values.append(_compact_text(item))
        elif isinstance(raw, str):
            values.append(_compact_text(raw))
    return _dedupe_strings(values)


def _tool_route_intents(tool: dict[str, Any]) -> list[str]:
    for key in ("route_intents", "routeIntents", "route_intent", "routeIntent"):
        raw = tool.get(key)
        if isinstance(raw, list):
            return _string_list(raw)
        text = _compact_text(raw)
        if text:
            return [text]
    return []


def _tool_arg_summary(tool: dict[str, Any]) -> str:
    schema = _tool_schema(tool)
    if schema is not None:
        properties = schema.get("properties")
        if isinstance(properties, dict):
            args = [
                _schema_property_summary(name, value)
                for name, value in sorted(properties.items())
            ]
            return "; ".join(arg for arg in args if arg)
    parameters = tool.get("parameters")
    if isinstance(parameters, list):
        args = [
            _parameter_summary(item)
            for item in parameters
            if isinstance(item, dict)
        ]
        return "; ".join(arg for arg in args if arg)
    return ""


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("args_schema", "arguments_schema", "schema"):
        raw = tool.get(key)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _schema_property_summary(name: Any, value: Any) -> str:
    arg_name = _compact_text(name)
    if not arg_name:
        return ""
    if not isinstance(value, dict):
        return arg_name
    parts = [arg_name]
    arg_type = value.get("type")
    if isinstance(arg_type, list):
        arg_type_text = "|".join(str(item) for item in arg_type if str(item).strip())
    else:
        arg_type_text = _compact_text(arg_type)
    if arg_type_text:
        parts.append(f"type={arg_type_text}")
    description = _compact_text(value.get("description"))
    if description:
        parts.append(description)
    enum = value.get("enum")
    if isinstance(enum, list) and enum:
        enum_text = ",".join(_compact_text(item) for item in enum[:8])
        if enum_text:
            parts.append(f"enum={enum_text}")
    return " ".join(parts)


def _parameter_summary(item: dict[str, Any]) -> str:
    name = _first_text(item, ("name", "key"))
    if not name:
        return ""
    parts = [name]
    param_type = _first_text(item, ("type", "kind"))
    if param_type:
        parts.append(f"type={param_type}")
    description = _first_text(item, ("description", "summary"))
    if description:
        parts.append(description)
    return " ".join(parts)


def _tool_name(tool: dict[str, Any]) -> str | None:
    for key in ("name", "tool_name", "toolName"):
        text = _compact_text(tool.get(key))
        if text:
            return text
    return None


def _first_text(value: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _compact_text(value.get(key))
        if text:
            return text
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_strings(_compact_text(item) for item in value)


def _dedupe_strings(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _compact_text(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _compact_text(value: Any, *, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _embedding_sha256(embedding: list[float]) -> str:
    data = json.dumps(
        embedding,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
