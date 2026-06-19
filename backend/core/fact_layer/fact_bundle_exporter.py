# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .fact_store import FactStore

# ── Hard constraint constants ────────────────────────────

# H3: keys that must never appear in manifest (violation = regression to account system)
FORBIDDEN_MANIFEST_KEYS = frozenset({
    "auth_token",
    "server_endpoint",
    "account_id",
    "sync_url",
    "external_db_url",
    "cloud_sync_endpoint",
    "api_key",
    "refresh_token",
    "user_password",
})

# Bundle version
BUNDLE_FORMAT_VERSION = 2

# Fixed paths inside zip
PATH_MANIFEST = "manifest.json"
PATH_FACTS_SQLITE = "facts.bin/structured.sqlite3"
# Post-extraction runtime layout equivalent (02 §3.4)
# apps/<bundleId>/<userId>/data.sqlite3  ← conceptual runtime reference point


@dataclass
class ExportResult:
    zip_path: Path
    manifest: Dict[str, Any]
    fact_count: int
    bundle_size_bytes: int
    checksum: str              # Joint SHA256 of all packaged files


# ── FactBundleExporter ──────────────────────────────────

class FactBundleExporter:

    def export(
        self,
        *,
        facts_sqlite_path: Union[str, Path],
        out_zip_path: Union[str, Path],
        base_model_ref: str,
        training_timestamp: Optional[int] = None,
        extra_manifest: Optional[Dict[str, Any]] = None,
    ) -> ExportResult:
        facts_sqlite_path = Path(facts_sqlite_path)
        out_zip_path = Path(out_zip_path)
        out_zip_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 1. Source file validation ─────────────────
        if not facts_sqlite_path.exists():
            raise FileNotFoundError(f"facts sqlite not found: {facts_sqlite_path}")
        if not facts_sqlite_path.is_file():
            raise ValueError(f"facts sqlite path is not a regular file: {facts_sqlite_path}")

        # ── 2. Fact stats (with H1 re-verify) ────────
        fact_stats = _collect_fact_stats(facts_sqlite_path)
        if fact_stats["source_types"] - {"user_device"}:
            raise ValueError(
                f"H1 violated: facts contain non-user_device source_types: "
                f"{fact_stats['source_types']}"
            )

        # ── 3. Build manifest ─────────────────────────
        now_ms = int(time.time() * 1000)
        manifest: Dict[str, Any] = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "version": 2,
            "created_at": now_ms,
            "training_timestamp": training_timestamp or now_ms,
            "base_model_ref": base_model_ref,
            "fact_stats": {
                "total_count": fact_stats["total_count"],
                "schemas_included": sorted(fact_stats["schemas"]),
                "source_types": sorted(fact_stats["source_types"]),
            },
        }
        if extra_manifest:
            # H3 guard: extra must not contain forbidden keys
            bad = FORBIDDEN_MANIFEST_KEYS & set(extra_manifest.keys())
            if bad:
                raise ValueError(
                    f"H3 violated: extra_manifest contains forbidden keys: {bad}"
                )
            manifest.update(extra_manifest)

        # Full re-check (guard against nested/misused keys in extra, though only top-level is checked)
        bad_top = FORBIDDEN_MANIFEST_KEYS & set(manifest.keys())
        if bad_top:
            raise ValueError(
                f"H3 violated: manifest top-level contains forbidden keys: {bad_top}"
            )

        # ── 4. Compute checksum (before write) ────────
        payload_files: List[tuple[str, Path]] = [
            (PATH_FACTS_SQLITE, facts_sqlite_path),
        ]

        checksum = _joint_checksum(payload_files)
        manifest["checksum"] = checksum

        # ── 5. Write zip ──────────────────────────────
        tmp_path = out_zip_path.with_suffix(out_zip_path.suffix + ".tmp")
        try:
            with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                # manifest
                zf.writestr(
                    PATH_MANIFEST,
                    json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2),
                )
                # facts sqlite
                zf.write(facts_sqlite_path, arcname=PATH_FACTS_SQLITE)

            # Atomic replace (H3: avoid corrupt bundle on partial failure)
            os.replace(tmp_path, out_zip_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        # ── 6. Post-write contract self-verify (H2/H3) ─
        _verify_bundle_contract(out_zip_path)

        return ExportResult(
            zip_path=out_zip_path,
            manifest=manifest,
            fact_count=fact_stats["total_count"],
            bundle_size_bytes=out_zip_path.stat().st_size,
            checksum=checksum,
        )


# ── Internal utilities ────────────────────────────────────

def _collect_fact_stats(sqlite_path: Path) -> Dict[str, Any]:
    store = FactStore(sqlite_path, read_only=True)
    try:
        conn = store._conn  # reuse already-opened read-only connection
        total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        schemas = {r[0] for r in conn.execute(
            "SELECT DISTINCT schema_name FROM facts"
        ).fetchall()}
        source_types = {r[0] for r in conn.execute(
            "SELECT DISTINCT source_type FROM facts"
        ).fetchall()}
        return {
            "total_count": total,
            "schemas": schemas,
            "source_types": source_types,
        }
    finally:
        store.close()


def _joint_checksum(files: Sequence[tuple[str, Path]]) -> str:
    h = hashlib.sha256()
    for arcname, path in sorted(files, key=lambda x: x[0]):
        h.update(arcname.encode("utf-8"))
        h.update(b"\x00")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def _verify_bundle_contract(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        # H2: reject symlinks (unix zip external_attr high 4 bits = 0xA for symlink)
        for info in zf.infolist():
            mode = info.external_attr >> 16
            if (mode & 0o170000) == 0o120000:  # S_IFLNK
                raise RuntimeError(
                    f"H2 violated: symlink entry found in bundle: {info.filename}"
                )

        # Required paths must exist
        names = set(zf.namelist())
        if PATH_MANIFEST not in names:
            raise RuntimeError("manifest missing")
        if PATH_FACTS_SQLITE not in names:
            raise RuntimeError(f"{PATH_FACTS_SQLITE} missing")

        # H3: manifest must not contain forbidden keys
        with zf.open(PATH_MANIFEST) as f:
            manifest = json.loads(f.read())
        bad = FORBIDDEN_MANIFEST_KEYS & set(manifest.keys())
        if bad:
            raise RuntimeError(
                f"H3 violated post-write: manifest has forbidden keys: {bad}"
            )
