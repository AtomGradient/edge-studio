# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""File and directory fingerprint helpers for Edge developer CLI receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DIRECTORY_MANIFEST_SCOPE = "directory_manifest_v1"
CONTENT_SHA256_SCOPE = "content_sha256_v1"
IGNORE_SUFFIXES = (".aria2",)


def is_complete_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any(
        path.glob(pattern)
        for pattern in ("*.safetensors", "*.gguf", "*.npz")
    )
    return has_config and has_weights


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for file_path in path.rglob("*"):
        if not _is_receipt_file(file_path):
            continue
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


def directory_manifest_hash(path: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    if path.exists():
        for file_path in sorted(
            (p for p in path.rglob("*") if _is_receipt_file(p)),
            key=lambda p: str(p.relative_to(path)),
        ):
            try:
                stat = file_path.stat()
            except OSError:
                continue
            entries.append({
                "path": str(file_path.relative_to(path)),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    digest = hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
    return {
        "sha256": f"sha256:{digest}",
        "manifest_sha256": f"sha256:{digest}",
        "sha256_scope": DIRECTORY_MANIFEST_SCOPE,
        "manifest_file_count": len(entries),
    }


def path_sha256(path: Path) -> dict[str, object]:
    if path.is_dir():
        payload = directory_manifest_hash(path)
        payload["size_bytes"] = dir_size_bytes(path)
        return payload
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError:
        digest.update(b"")
    return {
        "sha256": f"sha256:{digest.hexdigest()}",
        "sha256_scope": CONTENT_SHA256_SCOPE,
        "size_bytes": size,
    }


def pretty_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _is_receipt_file(path: Path) -> bool:
    return path.is_file() and not path.name.endswith(IGNORE_SUFFIXES)
