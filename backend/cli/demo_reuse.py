# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Artifact reuse smoke for the Edge demo CLI.

B7 is intentionally a receipt/manifest simulation. It does not copy artifacts,
restore Neural Imprint caches, load models, generate text, or use the network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.cli.demo_receipts import DEMO_RECEIPT_SCHEMA_VERSION, inspect_demo_receipt
from backend.cli.fingerprints import pretty_json
from backend.services.app_dirs import data_path


REUSE_REPORT_SCHEMA_VERSION = "edge.demo.reuse.report.v1"
REUSE_MANIFEST_SCHEMA_VERSION = "edge.demo.reuse.manifest.v1"
REUSE_RECEIPT_SCHEMA_VERSION = "edge.demo.reuse.receipt.v1"
DEFAULT_REUSE_APPS = "notes,finance"
APP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class DemoReuseOptions:
    run_id: str | None = None
    receipt_path: Path | None = None
    artifact_path: Path | None = None
    apps: str = DEFAULT_REUSE_APPS


@dataclass(frozen=True)
class DemoReuseResult:
    ok: bool
    exit_code: int
    report: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2)


def run_demo_reuse(*, options: DemoReuseOptions) -> DemoReuseResult:
    try:
        apps = _parse_apps(options.apps)
    except ValueError as exc:
        return _reuse_error("invalid_apps", str(exc))
    if not apps:
        return _reuse_error("invalid_apps", "At least one synthetic app id is required.")

    source = _load_source(options)
    if isinstance(source, DemoReuseResult):
        return source

    receipt_path, receipt = source
    validation = _validate_source_receipt(receipt, receipt_path=receipt_path, artifact_path=options.artifact_path)
    if validation:
        return validation

    run_id = str(receipt["run_id"])
    source_summary = _source_summary(receipt, receipt_path)
    app_reports: list[dict[str, Any]] = []
    created_at = _utc_now_iso()

    for app_id in apps:
        app_root = data_path("demo_runs", run_id, "reuse", app_id)
        app_root.mkdir(parents=True, exist_ok=True)
        manifest_path = app_root / "reuse_manifest.json"
        app_receipt_path = app_root / "reuse_receipt.json"

        manifest = _reuse_manifest(
            app_id=app_id,
            source=source_summary,
            manifest_path=manifest_path,
            created_at=created_at,
        )
        app_receipt = _reuse_receipt(
            app_id=app_id,
            source=source_summary,
            manifest_path=manifest_path,
            receipt_path=app_receipt_path,
            created_at=created_at,
        )
        manifest_path.write_text(pretty_json(manifest), encoding="utf-8")
        app_receipt_path.write_text(pretty_json(app_receipt), encoding="utf-8")
        app_reports.append(
            {
                "app_id": app_id,
                "state_root": str(app_root),
                "manifest_path": str(manifest_path),
                "receipt_path": str(app_receipt_path),
                "compatibility_ok": True,
                "status": "completed",
            }
        )

    report = {
        "schema_version": REUSE_REPORT_SCHEMA_VERSION,
        "ok": True,
        "status": "completed",
        "source": source_summary,
        "apps": app_reports,
        "app_count": len(app_reports),
        "raw_text_included": False,
        "network_used_during_reuse": False,
        "artifact_copied": False,
        "artifact_restored": False,
        "model_loaded": False,
        "created_at": created_at,
    }
    return DemoReuseResult(True, 0, report)


def format_demo_reuse(result: DemoReuseResult) -> str:
    r = result.report
    lines = [
        f"Edge demo reuse ({r.get('schema_version')})",
        f"status: {r.get('status')}",
    ]
    source = r.get("source")
    if isinstance(source, dict):
        lines.append(f"source run: {source.get('run_id')}")
        lines.append(f"source receipt: {source.get('receipt_path')}")
        lines.append(f"artifact_sha256: {source.get('artifact_sha256')}")
    for app in r.get("apps", []) if isinstance(r.get("apps"), list) else []:
        if isinstance(app, dict):
            lines.append(f"app {app.get('app_id')}: {app.get('status')} manifest={app.get('manifest_path')}")
    error = r.get("error")
    if isinstance(error, dict):
        lines.append(f"error: {error.get('code')}: {error.get('message')}")
        if error.get("detail"):
            lines.append(f"detail: {error['detail']}")
    return "\n".join(lines)


def _load_source(options: DemoReuseOptions) -> tuple[Path, dict[str, Any]] | DemoReuseResult:
    selected = [options.run_id is not None, options.receipt_path is not None, options.artifact_path is not None]
    if sum(1 for item in selected if item) != 1:
        return _reuse_error("invalid_source", "Specify exactly one of --run, --path, or --artifact.")

    if options.artifact_path is not None:
        artifact_path = options.artifact_path.expanduser()
        receipt_path = artifact_path.parent / "receipt.json"
        result = inspect_demo_receipt(path=receipt_path)
    elif options.receipt_path is not None:
        result = inspect_demo_receipt(path=options.receipt_path.expanduser())
    else:
        result = inspect_demo_receipt(run_id=options.run_id)

    if not result.ok or result.receipt is None:
        return _reuse_error(
            "receipt_invalid",
            "Source receipt could not be read or did not pass local-only validation.",
            receipt_path=result.receipt_path,
            detail=result.error,
        )
    return result.receipt_path or Path(""), result.receipt


def _validate_source_receipt(
    receipt: Mapping[str, Any],
    *,
    receipt_path: Path,
    artifact_path: Path | None,
) -> DemoReuseResult | None:
    if receipt.get("schema_version") != DEMO_RECEIPT_SCHEMA_VERSION:
        return _reuse_error("schema_version_mismatch", f"Expected {DEMO_RECEIPT_SCHEMA_VERSION}.", receipt_path=receipt_path)
    if receipt.get("status") != "completed":
        return _reuse_error("receipt_not_completed", "Only completed Neural Imprint demo receipts can be reused.", receipt_path=receipt_path)

    required = (
        "run_id",
        "model_sha256",
        "sample_sha256",
        "artifact_id",
        "artifact_path",
        "metadata_path",
        "artifact_sha256",
        "metadata_sha256",
        "prefix_tokens",
    )
    missing = [field for field in required if field not in receipt]
    if missing:
        return _reuse_error("source_incomplete", "Source receipt is missing artifact reuse fields.", receipt_path=receipt_path, detail=", ".join(missing))
    if not isinstance(receipt.get("prefix_tokens"), int):
        return _reuse_error("source_invalid", "Source receipt prefix_tokens must be an integer.", receipt_path=receipt_path, detail="prefix_tokens")

    source_artifact = Path(str(receipt["artifact_path"])).expanduser()
    source_metadata = Path(str(receipt["metadata_path"])).expanduser()
    if artifact_path is not None and source_artifact.resolve() != artifact_path.expanduser().resolve():
        return _reuse_error("artifact_mismatch", "Artifact path does not match the co-located source receipt.", receipt_path=receipt_path)
    if not source_artifact.is_file():
        return _reuse_error("artifact_missing", "Source artifact file does not exist.", receipt_path=receipt_path, detail=str(source_artifact))
    if not source_metadata.is_file():
        return _reuse_error("metadata_missing", "Source metadata file does not exist.", receipt_path=receipt_path, detail=str(source_metadata))
    return None


def _source_summary(receipt: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    return {
        "run_id": receipt["run_id"],
        "receipt_path": str(receipt_path),
        "receipt_schema_version": receipt["schema_version"],
        "artifact_id": receipt["artifact_id"],
        "artifact_path": receipt["artifact_path"],
        "metadata_path": receipt["metadata_path"],
        "model_sha256": receipt["model_sha256"],
        "sample_id": receipt.get("sample_id"),
        "sample_sha256": receipt["sample_sha256"],
        "artifact_sha256": receipt["artifact_sha256"],
        "metadata_sha256": receipt["metadata_sha256"],
        "prefix_tokens": receipt["prefix_tokens"],
    }


def _reuse_manifest(
    *,
    app_id: str,
    source: Mapping[str, Any],
    manifest_path: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": REUSE_MANIFEST_SCHEMA_VERSION,
        "app_id": app_id,
        "source": dict(source),
        "manifest_path": str(manifest_path),
        "compatibility": {
            "ok": True,
            "checks": [
                {"name": "receipt_schema", "status": "passed", "expected": DEMO_RECEIPT_SCHEMA_VERSION},
                {"name": "artifact_hash_present", "status": "passed"},
                {"name": "metadata_hash_present", "status": "passed"},
                {"name": "model_hash_present", "status": "passed"},
                {"name": "prefix_tokens_present", "status": "passed"},
            ],
        },
        "local_only": True,
        "network_used_during_reuse": False,
        "artifact_copied": False,
        "artifact_restored": False,
        "model_loaded": False,
        "raw_text_included": False,
        "created_at": created_at,
    }


def _reuse_receipt(
    *,
    app_id: str,
    source: Mapping[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": REUSE_RECEIPT_SCHEMA_VERSION,
        "app_id": app_id,
        "source_run_id": source["run_id"],
        "source_receipt_path": source["receipt_path"],
        "source_receipt_schema_version": source["receipt_schema_version"],
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "artifact_id": source["artifact_id"],
        "artifact_path": source["artifact_path"],
        "metadata_path": source["metadata_path"],
        "model_sha256": source["model_sha256"],
        "sample_sha256": source["sample_sha256"],
        "artifact_sha256": source["artifact_sha256"],
        "metadata_sha256": source["metadata_sha256"],
        "prefix_tokens": source["prefix_tokens"],
        "compatibility_ok": True,
        "raw_text_included": False,
        "network_used_during_reuse": False,
        "artifact_copied": False,
        "artifact_restored": False,
        "model_loaded": False,
        "status": "completed",
        "created_at": created_at,
    }


def _parse_apps(value: str) -> list[str]:
    apps: list[str] = []
    for raw in value.split(","):
        app = raw.strip()
        if not app or not APP_ID_RE.match(app):
            raise ValueError(f"Invalid synthetic app id: {raw.strip() or '<empty>'}.")
        if app in apps:
            continue
        apps.append(app)
    return apps


def _reuse_error(
    code: str,
    message: str,
    *,
    receipt_path: Path | None = None,
    detail: str | None = None,
) -> DemoReuseResult:
    error: dict[str, str] = {"code": code, "message": message}
    if detail:
        error["detail"] = detail
    return DemoReuseResult(
        False,
        1,
        {
            "schema_version": REUSE_REPORT_SCHEMA_VERSION,
            "ok": False,
            "status": code,
            "receipt_path": str(receipt_path) if receipt_path else None,
            "error": error,
            "raw_text_included": False,
            "network_used_during_reuse": False,
            "artifact_copied": False,
            "artifact_restored": False,
            "model_loaded": False,
        },
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
