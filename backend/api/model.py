# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model loading and info endpoints."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.model import LoadModelRequest, ModelInfo, QuantizationInfo
from backend.services.model_manager import manager

router = APIRouter(prefix="/api/model", tags=["model"])


def _arch_to_model_info(loaded) -> ModelInfo:
    """Convert LoadedModel to ModelInfo response."""
    arch = loaded.architecture
    wi = loaded.weight_index
    cfg = arch.config

    quant = None
    qcfg = cfg.get("quantization") or cfg.get("quantization_config")
    if qcfg:
        quant = QuantizationInfo(
            bits=qcfg.get("bits"),
            group_size=qcfg.get("group_size"),
            mode=qcfg.get("mode"),
        )

    # Handle nested text_config (VLMs: gemma3, qwen3_5, llama4, mistral3, pixtral, …)
    tc = cfg.get("text_config") or cfg.get("talker_config") or cfg

    def _pick(*keys, fallback=0):
        for k in keys:
            v = tc.get(k) or cfg.get(k)
            if v:
                return v
        return fallback

    has_moe = bool(
        _pick("num_local_experts", "num_experts", "n_routed_experts")
    )
    has_vision = "vision_config" in cfg

    num_layers          = _pick("num_hidden_layers", "n_layer", "num_layers",
                                "encoder_layers", "n_audio_layer")
    hidden_size         = _pick("hidden_size", "d_model", "n_audio_state")
    intermediate_size   = _pick("intermediate_size", "d_ff")
    num_attention_heads = _pick("num_attention_heads", "n_head",
                                "encoder_attention_heads", "n_audio_head")
    num_kv_heads        = _pick("num_key_value_heads", "num_kv_heads") or num_attention_heads
    # Whisper derives intermediate_size from hidden_size * 4
    if not intermediate_size and hidden_size:
        if cfg.get("model_type", "").startswith("whisper") or "n_audio_state" in cfg or "encoder_layers" in cfg:
            intermediate_size = hidden_size * 4

    # Detect thinking support (Qwen3.5, QwQ, DeepSeek-V3, etc.)
    # Note: Qwen3.5 is a VLM but supports thinking mode — check both LLM and VLM
    supports_thinking = False
    if loaded.category in ("llm", "vlm"):
        try:
            from backend.core.universal_tracer import detect_thinking_support
            supports_thinking = detect_thinking_support(arch.model_dir)
        except Exception:
            pass

    # Detect GGUF source
    is_gguf = cfg.get("_source_format") == "gguf"
    source_format = "gguf" if is_gguf else "safetensors"

    # Strip internal/large keys (e.g. _gguf_raw_metadata can contain
    # the entire tokenizer vocabulary — 150K+ strings — and would exceed
    # localStorage quota if persisted by the frontend)
    frontend_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}

    return ModelInfo(
        model_id=loaded.model_id,
        model_type=arch.model_type,
        model_name=arch.model_name,
        model_dir=arch.model_dir,
        total_params=arch.total_params,
        total_stored_params=arch.total_stored_params,
        total_size_bytes=arch.total_size_bytes,
        tensor_count=wi.tensor_count,
        quantization=quant,
        config=frontend_cfg,
        has_moe=has_moe,
        has_vision=has_vision,
        num_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        supports_thinking=supports_thinking,
        source_format=source_format,
        is_gguf=is_gguf,
        model_category=loaded.category,
    )


@router.post("/load", response_model=ModelInfo)
def load_model(req: LoadModelRequest) -> ModelInfo:
    # Expand ~ to actual home path
    model_dir = os.path.expanduser(req.model_dir)
    try:
        loaded = manager.load_model(model_dir)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Model directory not found")
    except (MemoryError, RuntimeError, ValueError, OSError) as exc:
        raise HTTPException(400, f"Failed to load model: {type(exc).__name__}")
    return _arch_to_model_info(loaded)


@router.get("/{model_id}/info", response_model=ModelInfo)
def model_info(model_id: str) -> ModelInfo:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")
    return _arch_to_model_info(loaded)


@router.delete("/{model_id}", response_model=dict[str, str])
def unload_model(model_id: str) -> dict[str, str]:
    manager.unload_model(model_id)
    return {"status": "ok"}


@router.get("/loaded", response_model=list[ModelInfo])
def list_loaded_models() -> list[ModelInfo]:
    return [_arch_to_model_info(m) for m in manager.list_models()]


@router.get("/{model_id}/session", response_model=dict[str, Any])
def get_session(model_id: str) -> dict[str, Any]:
    """Return summary of cached analysis results for this model."""
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")
    return manager.get_session_summary(model_id)
