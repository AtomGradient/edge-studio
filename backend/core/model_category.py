# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model category detection — LLM / VLM / TTS / STT."""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ModelCategory(str, Enum):
    LLM = "llm"      # Text generation (mlx-lm)
    VLM = "vlm"      # Vision + Language (mlx-vlm)
    TTS = "tts"      # Text-to-Speech (mlx-audio)
    STT = "stt"      # Speech-to-Text (mlx-audio)


def ensure_tokenizer_json(model_dir: str) -> bool:
    """Ensure tokenizer.json exists in model_dir.

    Swift's swift-transformers requires tokenizer.json, but some models
    (e.g. Qwen3-TTS) only ship with vocab.json + tokenizer_config.json.
    This function generates tokenizer.json from those files using
    HuggingFace's AutoTokenizer.

    Returns True if tokenizer.json already existed or was successfully created.
    """
    tokenizer_json = os.path.join(model_dir, "tokenizer.json")
    if os.path.isfile(tokenizer_json):
        return True

    # Need vocab.json or tokenizer_config.json to generate from
    if not os.path.isfile(os.path.join(model_dir, "tokenizer_config.json")):
        return False

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        logger.info("Generated tokenizer.json for %s", model_dir)
        return True
    except Exception as e:
        logger.warning("Failed to generate tokenizer.json: %s", e)
        return False


def detect_model_category(config: dict[str, Any]) -> ModelCategory:
    """Detect model category from config.json fields.

    - Has `vision_config` → VLM
    - `model_type` contains "tts" or has `talker_config` → TTS
    - `model_type` contains "asr" or equals "sensevoice" → STT
    - Otherwise → LLM
    """
    # VLM: vision models
    if "vision_config" in config:
        return ModelCategory.VLM

    model_type = config.get("model_type", "")
    model_type_lower = model_type.lower()

    # TTS: speech synthesis models
    if "tts" in model_type_lower or "talker_config" in config:
        return ModelCategory.TTS

    # STT: speech-to-text / ASR models
    if "asr" in model_type_lower or model_type_lower == "sensevoice":
        return ModelCategory.STT

    # Default: LLM
    return ModelCategory.LLM
