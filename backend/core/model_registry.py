# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model type detection and factory for loading models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.services.app_dirs import data_path

from .architecture import ModelArchitecture
from .config_parser import ConfigParser, load_config
from .config_gemma3 import Gemma3ConfigParser
from .config_generic import GenericConfigParser
from .config_qwen3 import Qwen3ConfigParser
from .config_qwen3_tts import Qwen3TTSConfigParser
from .gguf_loader import is_gguf_file, is_gguf_directory, find_gguf_file, load_gguf_metadata, load_gguf_weight_index
from .model_category import ModelCategory, detect_model_category
from .pruning_detector import PruningTrace, detect_pruning
from .weight_loader import WeightIndex, load_weight_index


# Registry: model_type string -> parser class
PARSER_REGISTRY: dict[str, type[ConfigParser]] = {
    "qwen3": Qwen3ConfigParser,
    "qwen3_tts": Qwen3TTSConfigParser,
    "gemma3": Gemma3ConfigParser,
}

# Common mlx-lm model directories to scan
DEFAULT_SEARCH_PATHS = [
    "~/.cache/huggingface/hub",
    "~/mlx-community",
    "~/Documents/mlx-community",
]

# Preset models — auto-discovered from standard directories.
# Override via EDGE_STUDIO_PRESETS env var (JSON) or platform data presets.json.
def _load_preset_models() -> dict[str, str]:
    """Load preset models: env var > config file > auto-scan standard dirs."""
    import json as _json

    # 1. Environment variable (JSON dict)
    env_presets = os.environ.get("EDGE_STUDIO_PRESETS")
    if env_presets:
        try:
            return _json.loads(env_presets)
        except (ValueError, TypeError):
            pass

    # 2. Config file
    config_path = data_path("presets.json")
    if config_path.exists():
        try:
            return _json.loads(config_path.read_text())
        except (ValueError, TypeError, OSError):
            pass

    # 3. Auto-scan common model directories
    presets: dict[str, str] = {}
    for search_dir in DEFAULT_SEARCH_PATHS:
        d = Path(search_dir).expanduser()
        if d.is_dir():
            for model_dir in sorted(d.iterdir()):
                if (model_dir / "config.json").exists():
                    presets[model_dir.name] = str(model_dir)
    return presets


PRESET_MODELS = _load_preset_models()


def detect_model_type(config: dict[str, Any]) -> str:
    """Detect model type from config.json's model_type field.

    Returns a parser registry key. Falls back to 'generic' for unknown types.
    """
    model_type = config.get("model_type", "")
    if model_type in PARSER_REGISTRY:
        return model_type
    # Fuzzy matching against known parsers, but require exact prefix match
    # to avoid e.g. "qwen3" matching "qwen3_5" (different architecture)
    lower = model_type.lower()
    for key in sorted(PARSER_REGISTRY.keys(), key=len, reverse=True):
        # Only match if the key IS the model_type (not a substring of a longer name)
        if lower == key:
            return key
    # Fallback to generic parser
    return "generic"


def discover_local_models(search_paths: list[str] | None = None) -> dict[str, str]:
    """Scan local directories for MLX model directories.

    Returns dict of {display_name: path} for directories containing config.json
    and at least one .safetensors file.
    """
    if search_paths is None:
        search_paths = DEFAULT_SEARCH_PATHS

    discovered = {}
    for search_path in search_paths:
        base = Path(search_path).expanduser()
        if not base.exists():
            continue

        # Search up to 2 levels deep
        for depth_pattern in ["*", "*/*"]:
            for candidate in base.glob(depth_pattern):
                if not candidate.is_dir():
                    continue
                config_file = candidate / "config.json"
                if not config_file.exists():
                    continue
                # Check for safetensors or GGUF files
                safetensors = list(candidate.glob("*.safetensors"))
                gguf_files = list(candidate.glob("*.gguf"))
                if not safetensors and not gguf_files:
                    continue
                # Use directory name as display name
                display_name = f"[Local] {candidate.name}"
                discovered[display_name] = str(candidate)

    return discovered


def load_model(model_dir: str) -> tuple[ModelArchitecture, WeightIndex, list[PruningTrace], ModelCategory]:
    """Load and analyze a model from its directory or GGUF file.

    Returns (architecture, weight_index, pruning_traces).
    Supports all mlx-lm model types via generic fallback parser,
    and single-file GGUF models.
    """
    # Check for GGUF: either a .gguf file path or a directory containing .gguf
    gguf_path = find_gguf_file(model_dir)
    p = Path(model_dir)
    # Use parent directory as model_dir when a file path is given
    effective_dir = str(p.parent) if p.is_file() else model_dir
    if gguf_path and not Path(effective_dir).joinpath("config.json").exists():
        # GGUF path — extract metadata and tensor index from the .gguf file
        config = load_gguf_metadata(gguf_path)
        model_type = detect_model_type(config)
        weight_index = load_gguf_weight_index(gguf_path)

        parser_cls = PARSER_REGISTRY.get(model_type, GenericConfigParser)
        parser = parser_cls(effective_dir, config, weight_index)
        architecture = parser.parse()

        category = detect_model_category(config)
        # No pruning traces for GGUF models
        return architecture, weight_index, [], category

    # Standard safetensors path
    config = load_config(model_dir)
    model_type = detect_model_type(config)

    weight_index = load_weight_index(model_dir)

    if model_type in PARSER_REGISTRY:
        parser_cls = PARSER_REGISTRY[model_type]
    else:
        parser_cls = GenericConfigParser

    parser = parser_cls(model_dir, config, weight_index)
    architecture = parser.parse()

    pruning_traces = detect_pruning(config)
    category = detect_model_category(config)

    return architecture, weight_index, pruning_traces, category
