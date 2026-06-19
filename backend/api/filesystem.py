# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Filesystem browsing endpoints."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.config import BROWSE_ROOTS
from backend.schemas.filesystem import BrowseResponse, FileEntry

router = APIRouter(prefix="/api/fs", tags=["filesystem"])


def _safe_path(path: str) -> Path:
    """Resolve, validate, and enforce whitelist on path."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(404, "Path not found")
    real = str(p)
    if not any(
        os.path.commonpath([real, root]) == root
        for root in BROWSE_ROOTS
    ):
        raise HTTPException(403, "Access denied")
    return p


@router.get("/home", response_model=dict[str, str])
def get_home() -> dict[str, str]:
    return {"path": str(Path.home())}


@router.get("/browse", response_model=BrowseResponse)
def browse_directory(path: str | None = Query(None)) -> BrowseResponse:
    if path is None:
        target = Path.home()
    else:
        target = _safe_path(path)

    if not target.is_dir():
        raise HTTPException(400, "Not a directory")

    entries: list[FileEntry] = []
    has_config = False
    has_safetensors = False
    has_gguf = False

    try:
        for item in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if item.name.startswith("."):
                continue
            is_dir = item.is_dir()
            size = None
            if not is_dir:
                try:
                    size = item.stat().st_size
                except OSError:
                    pass
            entries.append(FileEntry(
                name=item.name,
                path=str(item),
                is_dir=is_dir,
                size=size,
            ))
            if item.name == "config.json":
                has_config = True
            if item.name.endswith(".safetensors"):
                has_safetensors = True
            if item.name.endswith(".gguf"):
                has_gguf = True
    except PermissionError:
        raise HTTPException(403, "Permission denied")

    parent = str(target.parent) if target != target.parent else None

    return BrowseResponse(
        current_path=str(target),
        parent_path=parent,
        entries=entries,
        has_config_json=has_config,
        has_safetensors=has_safetensors,
        has_gguf=has_gguf,
    )


@router.get("/list-profiles", response_model=dict[str, list[str]])
def list_profiles(model_dir: str = Query(...)) -> dict[str, list[str]]:
    """List activation profile files in model directory."""
    from backend.core.activation_loader import find_profile_files

    profiles = find_profile_files(model_dir)
    return {"profiles": profiles}
