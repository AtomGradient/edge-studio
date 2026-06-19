# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Platform-native EdgeStudio data/cache directory helpers."""

from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir


APP_NAME = "edgestudio"


def data_dir() -> Path:
    """Return the canonical persistent data directory for EdgeStudio."""

    override = os.environ.get("EDGESTUDIO_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_data_dir(APP_NAME)).expanduser().resolve()


def cache_dir() -> Path:
    """Return the canonical cache directory for EdgeStudio."""

    override = os.environ.get("EDGESTUDIO_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_cache_dir(APP_NAME)).expanduser().resolve()


def data_path(*parts: str) -> Path:
    return data_dir().joinpath(*parts)


def cache_path(*parts: str) -> Path:
    return cache_dir().joinpath(*parts)


def unique_roots(*roots: Path) -> list[Path]:
    """Return roots in order, removing duplicates after expansion."""

    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        expanded = Path(root).expanduser().resolve()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        result.append(expanded)
    return result
