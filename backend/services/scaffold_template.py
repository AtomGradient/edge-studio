# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Resolve the EdgeScaffold template used by Studio export."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from .app_dirs import cache_path


EDGE_SCAFFOLD_REPOSITORY = "https://github.com/AtomGradient/edge-scaffold"
EDGE_SCAFFOLD_TEMPLATE_REF = "3be2ff826b1a5c9f67ff5af25f66bccaebc20164"
EDGE_SCAFFOLD_DIR_ENV = "EDGE_SCAFFOLD_DIR"


class ScaffoldTemplateError(Exception):
    pass


def source_tree_scaffold_source() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / "edge-scaffold"
    if _is_scaffold_template(candidate):
        return candidate
    return None


def resolve_scaffold_source() -> Path:
    configured = os.environ.get(EDGE_SCAFFOLD_DIR_ENV, "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not _is_scaffold_template(path):
            raise ScaffoldTemplateError(
                f"EDGE_SCAFFOLD_DIR does not point to a valid EdgeScaffold template: {path}"
            )
        return path

    source_tree = source_tree_scaffold_source()
    if source_tree is not None:
        return source_tree

    return ensure_cached_scaffold_template()


def ensure_cached_scaffold_template() -> Path:
    cache_root = cache_path("scaffold_templates")
    target = cache_root / f"edge-scaffold-{EDGE_SCAFFOLD_TEMPLATE_REF[:12]}"
    if _is_scaffold_template(target):
        return target

    archive_url = f"{EDGE_SCAFFOLD_REPOSITORY}/archive/{EDGE_SCAFFOLD_TEMPLATE_REF}.zip"
    cache_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="edgestudio_scaffold_template_") as tmp:
        tmp_root = Path(tmp)
        archive_path = tmp_root / "edge-scaffold.zip"
        try:
            with urllib.request.urlopen(archive_url, timeout=120) as response:
                with archive_path.open("wb") as f:
                    shutil.copyfileobj(response, f)
        except (OSError, urllib.error.URLError) as exc:
            raise ScaffoldTemplateError(
                "EdgeScaffold template is not available locally and could not be "
                f"downloaded from {archive_url}. Check your network connection, or "
                f"set {EDGE_SCAFFOLD_DIR_ENV} to a local edge-scaffold checkout."
            ) from exc

        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(tmp_root)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ScaffoldTemplateError(
                f"Downloaded EdgeScaffold template archive is invalid: {archive_url}"
            ) from exc

        candidates = [
            item for item in tmp_root.iterdir()
            if item.is_dir() and item.name.startswith("edge-scaffold-")
        ]
        if len(candidates) != 1 or not _is_scaffold_template(candidates[0]):
            raise ScaffoldTemplateError(
                f"Downloaded EdgeScaffold template archive does not contain a valid template: {archive_url}"
            )

        tmp_target = cache_root / f".edge-scaffold-{EDGE_SCAFFOLD_TEMPLATE_REF[:12]}.download"
        if tmp_target.exists():
            shutil.rmtree(tmp_target)
        shutil.copytree(
            candidates[0],
            tmp_target,
            ignore=shutil.ignore_patterns(".git", ".github", ".ai-mailbox", ".claude", ".pytest_cache", "build"),
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp_target, target)

    if not _is_scaffold_template(target):
        raise ScaffoldTemplateError(f"Cached EdgeScaffold template is invalid: {target}")
    return target


def _is_scaffold_template(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / ".scaffold_version").is_file()
        and (path / ".min_runtime_version").is_file()
        and (path / "project.yml").is_file()
        and (path / "EdgeScaffold" / "App" / "ScaffoldConfig.swift").is_file()
    )
