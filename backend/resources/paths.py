# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Resolve source-tree and wheel-installed Edge Studio resources."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
RESOURCE_DIR = BACKEND_DIR / "resources"


def script_path(name: str) -> Path:
    """Return the best available packaged helper script path."""
    candidates = [
        RESOURCE_DIR / "scripts" / name,
        REPO_ROOT / "scripts" / name,
        Path(sys.prefix) / "scripts" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def frontend_dist_candidates() -> list[Path]:
    """Return frontend dist locations in source and wheel install order."""
    return [
        REPO_ROOT / "frontend" / "dist",
        RESOURCE_DIR / "frontend" / "dist",
        Path(sys.prefix) / "frontend" / "dist",
    ]
