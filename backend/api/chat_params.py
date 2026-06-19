# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model-aware generation parameter defaults.

Centralizes parameter decisions based on model metadata (category, size, family).
Each model type and size gets appropriate defaults — small models get tighter limits
and stronger repetition penalties; large models get more freedom.

Usage in chat handlers:
    params = get_generation_params(model_dir)
    effective_max = min(client_max_tokens, params["max_tokens"])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationParams:
    """Recommended generation parameters for a specific model."""
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float      # 1.0 = disabled, >1.0 = penalize repeats
    repetition_context_size: int
    # Safety net: n-gram repetition detection
    rep_ngram: int                 # n-gram size to monitor
    rep_threshold: int             # how many consecutive repeats before stopping
    # EOS tokens (extracted from config, cached)
    eos_token_ids: tuple[int, ...] = (2,)


def _estimate_params_b(config: dict) -> float:
    """Estimate model parameters (billions) from config.json.

    Handles nested configs: VLM/multimodal models often put text model
    dimensions under 'text_config' rather than top-level.
    """
    # Some configs have it directly
    if "num_params" in config:
        return config["num_params"] / 1e9

    # Resolve the text/LLM config — may be nested for VLM/multimodal models
    cfg = config
    for nested_key in ("text_config", "language_config", "llm_config"):
        if nested_key in config and isinstance(config[nested_key], dict):
            nested = config[nested_key]
            if "hidden_size" in nested:
                cfg = nested
                break

    hidden = cfg.get("hidden_size", 0)
    layers = cfg.get("num_hidden_layers", 0)
    intermediate = cfg.get("intermediate_size", hidden * 4)
    vocab = cfg.get("vocab_size", config.get("vocab_size", 32000))

    if hidden == 0 or layers == 0:
        return 0

    # Rough formula: embedding + layers * (attn + ffn)
    embed_params = vocab * hidden * 2  # input + output embeddings
    attn_params = layers * (4 * hidden * hidden)  # Q, K, V, O
    ffn_params = layers * (2 * hidden * intermediate)  # gate + up, down
    total = embed_params + attn_params + ffn_params
    return total / 1e9


def _detect_family(config: dict, model_dir: str) -> str:
    """Detect model family from config or directory name."""
    model_type = config.get("model_type", "").lower()
    dir_lower = os.path.basename(model_dir).lower()

    if "qwen" in model_type or "qwen" in dir_lower:
        return "qwen"
    if "gemma" in model_type or "gemma" in dir_lower:
        return "gemma"
    if "llama" in model_type or "llama" in dir_lower:
        return "llama"
    if "mistral" in model_type or "mistral" in dir_lower:
        return "mistral"
    return model_type or "unknown"


def _detect_category(config: dict, model_dir: str) -> str:
    """Detect model category: llm, vlm, tts, asr."""
    if "vision_config" in config:
        return "vlm"
    # Check for TTS/ASR markers
    model_type = config.get("model_type", "").lower()
    dir_lower = os.path.basename(model_dir).lower()
    if any(k in model_type or k in dir_lower for k in ("tts", "speech_synthesis")):
        return "tts"
    if any(k in model_type or k in dir_lower for k in ("asr", "whisper", "stt", "sensevoice", "parakeet")):
        return "asr"
    return "llm"


@lru_cache(maxsize=64)
def get_generation_params(model_dir: str) -> GenerationParams:
    """Return recommended generation parameters based on model metadata.

    Decision factors:
    - Model category (LLM/VLM/TTS/ASR)
    - Model size (params_b)
    - Model family (Qwen, Gemma, etc.)
    """
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        config = {}

    params_b = _estimate_params_b(config)
    family = _detect_family(config, model_dir)
    category = _detect_category(config, model_dir)

    # Extract EOS token IDs
    raw_eos = config.get("eos_token_id", 2)
    eos_ids = tuple(raw_eos) if isinstance(raw_eos, list) else (raw_eos,)

    logger.debug(
        "Model params: dir=%s, category=%s, family=%s, params_b=%.1f",
        os.path.basename(model_dir), category, family, params_b,
    )

    # ── TTS / ASR: not token-based, return generous defaults ──
    if category == "tts":
        return GenerationParams(
            max_tokens=4096,  # TTS internally manages chunk sizes
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0,
            repetition_context_size=20,
            rep_ngram=6,
            rep_threshold=4,
            eos_token_ids=eos_ids,
        )
    if category == "asr":
        return GenerationParams(
            max_tokens=8192,  # ASR needs many tokens for long audio
            temperature=0.0,  # greedy for transcription accuracy
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.0,
            repetition_context_size=20,
            rep_ngram=6,
            rep_threshold=4,
            eos_token_ids=eos_ids,
        )

    # ── LLM / VLM: max_tokens from model config ──

    # Prefer the model's declared max_position_embeddings (e.g. 256K),
    # instead of guessing from params_b. Default output cap is context_length/4 or 8192,
    # whichever is smaller, as a safety ceiling (prevents unbounded generation).
    cfg_for_ctx = config
    for nested_key in ("text_config", "language_config", "llm_config"):
        if nested_key in config and isinstance(config[nested_key], dict):
            nested = config[nested_key]
            if "max_position_embeddings" in nested:
                cfg_for_ctx = nested
                break
    model_context = cfg_for_ctx.get("max_position_embeddings", 0)
    if model_context > 0:
        max_tokens = min(model_context // 4, 65536)
    elif params_b <= 1.5:
        max_tokens = 1024
    elif params_b <= 5.0:
        max_tokens = 2048
    elif params_b <= 35.0:
        max_tokens = 4096
    else:
        max_tokens = 8192

    # temperature: small models benefit from slightly higher temp to avoid loops
    if params_b <= 3.0:
        temperature = 0.8
    elif params_b <= 10.0:
        temperature = 0.7
    else:
        temperature = 0.6  # large models: more focused

    # repetition_penalty: critical for small models, unnecessary for large
    if params_b <= 2.0:
        rep_penalty = 1.2
        rep_context = 40
    elif params_b <= 5.0:
        rep_penalty = 1.1
        rep_context = 40
    elif params_b <= 10.0:
        rep_penalty = 1.05
        rep_context = 30
    else:
        rep_penalty = 1.0  # large models: trust the model
        rep_context = 20

    # top_k / top_p: tighter for small models
    if params_b <= 3.0:
        top_k = 30
        top_p = 0.85
    else:
        top_k = 50
        top_p = 0.9

    # Safety net thresholds: more aggressive for small models
    if params_b <= 3.0:
        rep_ngram = 4
        rep_threshold = 3
    else:
        rep_ngram = 6
        rep_threshold = 4

    return GenerationParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=rep_penalty,
        repetition_context_size=rep_context,
        rep_ngram=rep_ngram,
        rep_threshold=rep_threshold,
        eos_token_ids=eos_ids,
    )
