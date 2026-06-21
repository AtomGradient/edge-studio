# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Code-agent friendly EdgeScaffold ZIP export command."""

from __future__ import annotations

import json
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from backend.cli.models import ModelWhereReport, where_model
from backend.core.scaffold_zip_export import ScaffoldZipResult, export_scaffold_zip


EXPORT_SCAFFOLD_SCHEMA_VERSION = "edge.export.scaffold.report.v1"
DEFAULT_EXPORT_MODEL_REF = "qwen3.5-9b-4bit"

ScaffoldExporter = Callable[..., ScaffoldZipResult]


@dataclass(frozen=True)
class ExportScaffoldOptions:
    model_ref: str = "auto"
    app_name: str = "MyApp"
    system_prompt: str = "You are a helpful assistant."
    bundle_id: str | None = None
    team_id: str | None = None
    direction_set_id: str | None = None
    output_path: Path | None = None
    enable_dsr: bool = True
    dsr_budget: int | None = None


@dataclass(frozen=True)
class ExportScaffoldResult:
    ok: bool
    exit_code: int
    report: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2)


def run_export_scaffold(
    *,
    options: ExportScaffoldOptions,
    env: Mapping[str, str] | None = None,
    exporter: ScaffoldExporter | None = None,
) -> ExportScaffoldResult:
    started = time.time()
    model_ref = DEFAULT_EXPORT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    if local_match is None:
        remediation = where.fetch_command or f"edge models fetch {model_ref}"
        return _error_result(
            code="missing_model",
            message="A complete local model is required before exporting an Edge Scaffold app.",
            options=options,
            model_ref=model_ref,
            where=where,
            remediation=remediation,
            started=started,
        )

    model_path = Path(local_match.path)
    _progress(f"[export:scaffold] model={model_ref} path={model_path}")
    export_fn = exporter or export_scaffold_zip
    core_result = export_fn(
        str(model_path),
        app_name=options.app_name,
        system_prompt=options.system_prompt,
        bundle_id=options.bundle_id,
        team_id=options.team_id,
        direction_set_id=options.direction_set_id,
        enable_dsr=options.enable_dsr,
        dsr_budget=options.dsr_budget,
        progress_callback=lambda message, fraction: _progress(
            f"[export:scaffold] {round(float(fraction) * 100):>3}% {message}"
        ),
    )
    if not core_result.success:
        return _error_result(
            code="export_failed",
            message=core_result.error or "Edge Scaffold export failed.",
            options=options,
            model_ref=model_ref,
            where=where,
            remediation=_remediation_for_export_error(core_result.error),
            started=started,
        )

    source_zip = Path(core_result.zip_path).expanduser()
    if not source_zip.is_file():
        return _error_result(
            code="zip_missing",
            message=f"Export reported success but ZIP was not found: {source_zip}",
            options=options,
            model_ref=model_ref,
            where=where,
            started=started,
        )

    output_zip = _copy_to_stable_output(source_zip, options.output_path)
    _cleanup_temp_zip(source_zip, output_zip)
    elapsed = round(time.time() - started, 2)
    report: dict[str, object] = {
        "schema_version": EXPORT_SCAFFOLD_SCHEMA_VERSION,
        "ok": True,
        "status": "completed",
        "app_name": core_result.app_name,
        "model_ref": model_ref,
        "model_name": core_result.model_name,
        "model_dir": core_result.model_dir,
        "direction_set_id": core_result.direction_set_id or options.direction_set_id or "finance_consumer",
        "zip_path": str(output_zip),
        "zip_size_bytes": output_zip.stat().st_size,
        "elapsed_seconds": elapsed,
        "model_download_used_during_export": False,
        "next_steps": [
            f"unzip {shlex.quote(str(output_zip))}",
            "Open the generated .xcodeproj in Xcode.",
            "Select a real iOS device and configure signing if needed.",
        ],
    }
    return ExportScaffoldResult(ok=True, exit_code=0, report=report)


def format_export_scaffold(result: ExportScaffoldResult) -> str:
    report = result.report
    lines = [
        f"Edge export scaffold ({report.get('schema_version')})",
        f"status: {report.get('status')}",
    ]
    if not result.ok:
        error = report.get("error")
        if isinstance(error, dict):
            lines.append(f"error: {error.get('code')}: {error.get('message')}")
            if error.get("remediation"):
                lines.append(f"remediation: {error['remediation']}")
        return "\n".join(lines)

    lines.extend(
        [
            f"app: {report.get('app_name')}",
            f"model: {report.get('model_ref')}",
            f"model_path: {report.get('model_dir')}",
            f"direction_set_id: {report.get('direction_set_id')}",
            f"zip: {report.get('zip_path')}",
        ]
    )
    next_steps = report.get("next_steps")
    if isinstance(next_steps, list) and next_steps:
        lines.append("next:")
        lines.extend(f"- {step}" for step in next_steps)
    return "\n".join(lines)


def _error_result(
    *,
    code: str,
    message: str,
    options: ExportScaffoldOptions,
    model_ref: str,
    where: ModelWhereReport | None = None,
    remediation: str | None = None,
    started: float | None = None,
) -> ExportScaffoldResult:
    error: dict[str, str] = {"code": code, "message": message}
    if remediation:
        error["remediation"] = remediation
    report: dict[str, object] = {
        "schema_version": EXPORT_SCAFFOLD_SCHEMA_VERSION,
        "ok": False,
        "status": code,
        "app_name": options.app_name,
        "model_ref": model_ref,
        "zip_path": None,
        "model_download_used_during_export": False,
        "error": error,
    }
    if where is not None:
        report["model_where"] = where.as_dict()
    if started is not None:
        report["elapsed_seconds"] = round(time.time() - started, 2)
    return ExportScaffoldResult(ok=False, exit_code=1, report=report)


def _first_complete_match(where: ModelWhereReport):
    for match in where.local_matches:
        if match.complete:
            return match
    return None


def _copy_to_stable_output(source_zip: Path, output_path: Path | None) -> Path:
    target = _resolve_output_zip(source_zip.name, output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_zip.resolve() != target.resolve():
        shutil.copy2(source_zip, target)
    return target.resolve()


def _resolve_output_zip(default_name: str, output_path: Path | None) -> Path:
    if output_path is None:
        return Path.cwd() / default_name
    raw = output_path.expanduser()
    path = raw if raw.is_absolute() else Path.cwd() / raw
    if path.exists() and path.is_dir():
        return path / default_name
    if path.suffix.lower() == ".zip":
        return path
    return path / default_name


def _cleanup_temp_zip(source_zip: Path, output_zip: Path) -> None:
    try:
        if source_zip.resolve() == output_zip.resolve():
            return
        source_zip.unlink(missing_ok=True)
        source_zip.parent.rmdir()
    except OSError:
        pass


def _remediation_for_export_error(error: str | None) -> str | None:
    message = error or ""
    if "xcodegen" in message.lower():
        return "brew install xcodegen"
    if "edgescaffold" in message.lower() or "scaffold template" in message.lower():
        return "Set EDGE_SCAFFOLD_DIR to a local edge-scaffold checkout or retry when network access is available."
    return None


def _progress(message: str) -> None:
    print(message, file=sys.stderr)
