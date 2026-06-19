# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model loading and caching for chat endpoints."""

from __future__ import annotations

import threading
from typing import Any

import logging
import gc
import os

from backend.services.mlx_runtime_gate import mlx_runtime_gate

logger = logging.getLogger(__name__)

# Cache for mlx-lm models (text-only, keyed by model_dir)
_model_cache: dict[str, tuple[Any, Any]] = {}
_cache_lock = threading.Lock()

# Cache for mlx-vlm models (vision, keyed by model_dir)
_vlm_model_cache: dict[str, tuple[Any, Any]] = {}
_vlm_cache_lock = threading.Lock()

# Cache for mlx-audio TTS models (keyed by model_dir)
_tts_model_cache: dict[str, Any] = {}
_tts_cache_lock = threading.Lock()

# Cache for mlx-audio STT models (keyed by model_dir)
_stt_model_cache: dict[str, Any] = {}
_stt_cache_lock = threading.Lock()


def _canonical_model_dir(model_dir: str) -> str:
    return os.path.abspath(os.path.expanduser(model_dir))


def _model_dir_keys(model_dir: str) -> set[str]:
    canonical = _canonical_model_dir(model_dir)
    expanded = os.path.expanduser(model_dir)
    return {model_dir, expanded, os.path.abspath(expanded), canonical}


def clear_model_cache(model_dir: str) -> int:
    """Remove cached inference objects for a model directory.

    Returns the number of cache entries removed across all runtime caches.
    """
    keys = _model_dir_keys(model_dir)
    removed = 0

    with _cache_lock:
        for key in keys:
            removed += 1 if _model_cache.pop(key, None) is not None else 0
    with _vlm_cache_lock:
        for key in keys:
            removed += 1 if _vlm_model_cache.pop(key, None) is not None else 0
    with _tts_cache_lock:
        for key in keys:
            removed += 1 if _tts_model_cache.pop(key, None) is not None else 0
    with _stt_cache_lock:
        for key in keys:
            removed += 1 if _stt_model_cache.pop(key, None) is not None else 0

    if removed:
        _clear_runtime_memory_cache()
    return removed


def _clear_runtime_memory_cache() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
        metal = getattr(mx, "metal", None)
        metal_clear_cache = getattr(metal, "clear_cache", None) if metal is not None else None
        if callable(metal_clear_cache):
            metal_clear_cache()
    except Exception as exc:
        logger.debug("MLX cache clear skipped: %s", exc)


def _get_or_load_mlx_model(model_dir: str):
    """Load or retrieve cached mlx-lm model and tokenizer (text-only models)."""
    model_dir = _canonical_model_dir(model_dir)
    with _cache_lock:
        if model_dir in _model_cache:
            return _model_cache[model_dir]

    with mlx_runtime_gate("chat_loaders.lm_load"):
        with _cache_lock:
            if model_dir in _model_cache:
                return _model_cache[model_dir]

        from mlx_lm.utils import load as lm_load
        model, tokenizer = lm_load(model_dir)

        with _cache_lock:
            _model_cache[model_dir] = (model, tokenizer)

    return model, tokenizer


def _get_or_load_vlm_model(model_dir: str):
    """Load or retrieve cached mlx-vlm model and processor."""
    model_dir = _canonical_model_dir(model_dir)
    with _vlm_cache_lock:
        if model_dir in _vlm_model_cache:
            return _vlm_model_cache[model_dir]

    with mlx_runtime_gate("chat_loaders.vlm_load"):
        with _vlm_cache_lock:
            if model_dir in _vlm_model_cache:
                return _vlm_model_cache[model_dir]

        from mlx_vlm import load as vlm_load
        model, processor = vlm_load(model_dir)

        with _vlm_cache_lock:
            _vlm_model_cache[model_dir] = (model, processor)

    return model, processor


def _infer_tts_model_type(model_dir: str) -> str | None:
    """Infer TTS model_type from config when it's missing.

    mlx_audio relies on model_type or directory-name heuristics to find
    the right architecture module.  Some models (e.g. Kokoro) ship without
    a model_type field, causing a load failure when the directory name
    doesn't split neatly into a recognized part.

    Returns the inferred model_type string, or None if unneeded / unknown.
    """
    import json, os
    cfg_path = os.path.join(model_dir, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if cfg.get("model_type"):
        return None  # already present — let mlx_audio handle it

    # Kokoro: has istftnet + plbert, no model_type
    if "istftnet" in cfg and "plbert" in cfg:
        return "kokoro"

    return None


def _get_or_load_tts_model(model_dir: str):
    """Load or retrieve cached mlx-audio TTS model."""
    model_dir = _canonical_model_dir(model_dir)
    with _tts_cache_lock:
        if model_dir in _tts_model_cache:
            return _tts_model_cache[model_dir]

    # Ensure tokenizer.json exists (needed by Swift's swift-transformers,
    # also good practice for consistency)
    from backend.core.model_category import ensure_tokenizer_json
    ensure_tokenizer_json(model_dir)

    # If config.json is missing model_type, inject it so mlx_audio can
    # find the right architecture module regardless of directory naming.
    import json, os
    inferred = _infer_tts_model_type(model_dir)
    patched = False
    cfg_path = os.path.join(model_dir, "config.json")
    if inferred:
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["model_type"] = inferred
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            patched = True
            logger.info("Injected model_type=%r into %s", inferred, cfg_path)
        except (OSError, json.JSONDecodeError):
            pass

    with mlx_runtime_gate("chat_loaders.tts_load"):
        with _tts_cache_lock:
            if model_dir in _tts_model_cache:
                return _tts_model_cache[model_dir]

        from mlx_audio.tts.utils import load as tts_load

        # Workaround for mlx_audio Kokoro bug: the sanitize method assumes
        # PyTorch weight layout and transposes conv weights, but quantized
        # MLX models already have the correct layout. Detect this and skip
        # sanitize to avoid shape mismatches.
        _skip_sanitize = False
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                _cfg = json.load(f)
            if _cfg.get("model_type") == "kokoro" and "quantization" in _cfg:
                _skip_sanitize = True
        except (OSError, json.JSONDecodeError):
            pass

        if _skip_sanitize:
            # Replicate base_load_model but skip sanitize for pre-converted weights
            from mlx_audio.utils import get_model_class, load_config, load_weights
            from mlx_audio.tts.utils import MODEL_REMAPPING
            from pathlib import Path

            model_path = Path(model_dir)
            model_name = model_path.name.lower().split("-")
            config = load_config(model_path)
            config["model_path"] = str(model_path)
            model_type_val = config.get("model_type", model_name[0].lower())

            model_class, model_type_val = get_model_class(
                model_type=model_type_val, model_name=model_name,
                category="tts", model_remapping=MODEL_REMAPPING,
            )
            model_config = (
                model_class.ModelConfig.from_dict(config)
                if hasattr(model_class, "ModelConfig") else config
            )
            model = model_class.Model(model_config)
            weights = load_weights(model_path)
            # Skip sanitize — weights are already in MLX format
            quantization = config.get("quantization", None)
            if quantization:
                from mlx_audio.utils import apply_quantization
                apply_quantization(model, config, weights)
            model.load_weights(list(weights.items()), strict=False)
            if not False:  # lazy=False → eval
                import mlx.core as mx
                mx.eval(model.parameters())
            model.eval()
        else:
            model = tts_load(model_dir)

        with _tts_cache_lock:
            _tts_model_cache[model_dir] = model

    return model


def _get_or_load_stt_model(model_dir: str):
    """Load or retrieve cached mlx-audio STT model."""
    model_dir = _canonical_model_dir(model_dir)
    with _stt_cache_lock:
        if model_dir in _stt_model_cache:
            return _stt_model_cache[model_dir]

    with mlx_runtime_gate("chat_loaders.stt_load"):
        with _stt_cache_lock:
            if model_dir in _stt_model_cache:
                return _stt_model_cache[model_dir]

        from mlx_audio.stt.utils import load as stt_load
        model = stt_load(model_dir)

        with _stt_cache_lock:
            _stt_model_cache[model_dir] = model

    return model
