# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Filesystem browsing schemas."""

from __future__ import annotations

from pydantic import BaseModel


class FileEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class BrowseRequest(BaseModel):
    path: str | None = None


class BrowseResponse(BaseModel):
    current_path: str
    parent_path: str | None = None
    entries: list[FileEntry]
    has_config_json: bool = False
    has_safetensors: bool = False
    has_gguf: bool = False
