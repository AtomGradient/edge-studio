# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Shared model path helpers for Edge developer CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from backend.core.model_registry import DEFAULT_SEARCH_PATHS, discover_local_models


def model_cache_roots(env: Mapping[str, str] | None = None) -> list[Path]:
    """Return model cache roots, including explicit user overrides first."""

    values = env or os.environ
    raw = values.get("EDGESTUDIO_MODEL_ROOTS", "").strip()
    roots: list[Path] = []
    if raw:
        separators = "," if "," in raw else os.pathsep
        roots.extend(Path(part).expanduser() for part in raw.split(separators) if part.strip())

    home = Path.home()
    roots.extend(Path(path).expanduser() for path in DEFAULT_SEARCH_PATHS)
    roots.append(home / "Library" / "Caches" / "huggingface" / "hub")
    return unique_paths(roots)


def unique_paths(paths: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result


def detect_model_dirs(root: Path) -> list[str]:
    """Detect shallow model-like directories under a cache root."""

    detected: list[str] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)[:200]
    except OSError:
        return detected
    for child in children:
        if not child.is_dir():
            continue
        if child.name.startswith("models--"):
            detected.append(str(child))
            continue
        try:
            names = {entry.name for entry in child.iterdir()}
        except OSError:
            continue
        if {"config.json", "tokenizer.json"} & names or any(name.endswith(".safetensors") for name in names):
            detected.append(str(child))
    return detected


def discover_local_model_paths(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Discover complete local model directories using the core registry."""

    roots = model_cache_roots(env)
    return discover_local_models([str(root) for root in roots])
