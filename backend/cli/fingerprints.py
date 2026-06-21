# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""File and directory fingerprint helpers for Edge developer CLI receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DIRECTORY_MANIFEST_SCOPE = "directory_manifest_v1"
CONTENT_SHA256_SCOPE = "content_sha256_v1"
IGNORE_SUFFIXES = (".aria2",)


@dataclass(frozen=True)
class ModelDirIntegrity:
    complete: bool
    issues: tuple[str, ...]


def is_complete_model_dir(path: Path) -> bool:
    return model_dir_integrity(path).complete


def model_dir_integrity(path: Path, *, expected_size_bytes: int | None = None) -> ModelDirIntegrity:
    issues: list[str] = []
    if not path.is_dir():
        return ModelDirIntegrity(False, ("not_a_directory",))

    if not (path / "config.json").exists():
        issues.append("missing_config_json")

    temp_files = sorted(p for p in path.rglob("*") if p.is_file() and p.name.endswith(IGNORE_SUFFIXES))
    if temp_files:
        issues.append("partial_download_files_present")

    weight_files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix in {".safetensors", ".gguf", ".npz"}
    )
    if not weight_files:
        issues.append("missing_weight_files")

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        issues.extend(_safetensors_index_issues(index_path))

    for weight_path in weight_files:
        if weight_path.suffix == ".safetensors":
            issue = _safetensors_issue(weight_path)
            if issue:
                issues.append(issue)
        elif weight_path.stat().st_size <= 0:
            issues.append(f"empty_weight_file:{weight_path.relative_to(path)}")

    if expected_size_bytes is not None and expected_size_bytes > 0:
        actual_size = dir_size_bytes(path)
        floor = int(expected_size_bytes * 0.70)
        if actual_size < floor:
            issues.append(f"size_below_expected:{actual_size}<{floor}")

    return ModelDirIntegrity(not issues, tuple(issues))


def _safetensors_index_issues(index_path: Path) -> list[str]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"invalid_safetensors_index:{index_path.name}"]

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        return [f"invalid_safetensors_index:{index_path.name}"]

    missing: list[str] = []
    for shard_name in sorted({str(value) for value in weight_map.values()}):
        shard_path = index_path.parent / shard_name
        if not shard_path.is_file():
            missing.append(shard_name)

    return [f"missing_safetensors_shard:{name}" for name in missing[:20]]


def _safetensors_issue(path: Path) -> str | None:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="np") as handle:
            keys = list(handle.keys())
        if not keys:
            return f"empty_safetensors:{path.name}"
        return None
    except Exception as exc:  # noqa: BLE001
        return f"invalid_safetensors:{path.name}:{str(exc)[:160]}"


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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_prefixed(data: bytes) -> str:
    return f"sha256:{sha256_hex(data)}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_receipt_file(path: Path) -> bool:
    return path.is_file() and not path.name.endswith(IGNORE_SUFFIXES)
