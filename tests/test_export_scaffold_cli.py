# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Tests for the agent-first scaffold export CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cli import export_scaffold, main as cli_main
from backend.cli.models import CatalogResolution, LocalModel, ModelWhereReport
from backend.core.scaffold_zip_export import ScaffoldZipResult


def _resolution() -> CatalogResolution:
    return CatalogResolution(
        status="resolved",
        input="qwen3.5-9b-4bit",
        model_id="qwen3.5-9b-4bit",
        name="Qwen3.5-9B-4bit",
        download_hint="mlx-community/Qwen3.5-9B-4bit",
        category="llm",
        size_gb=5.0,
        catalog_source="test",
        catalog_version="test",
        matched_by="id",
        alternates=[],
    )


def _where_ok(model_dir: Path) -> ModelWhereReport:
    return ModelWhereReport(
        schema_version="edge.models.where.report.v1",
        status="ok",
        resolution=_resolution(),
        local_matches=[
            LocalModel(
                name="Qwen3.5-9B-4bit",
                path=str(model_dir),
                size_bytes=123,
                complete=True,
            )
        ],
        fetch_command=None,
    )


def _where_missing() -> ModelWhereReport:
    return ModelWhereReport(
        schema_version="edge.models.where.report.v1",
        status="missing",
        resolution=_resolution(),
        local_matches=[],
        fetch_command="edge models fetch qwen3.5-9b-4bit",
    )


def test_export_scaffold_missing_model_returns_fetch_remediation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_scaffold, "where_model", lambda *_args, **_kwargs: _where_missing())

    result = export_scaffold.run_export_scaffold(
        options=export_scaffold.ExportScaffoldOptions(model_ref="qwen3.5-9b-4bit")
    )

    assert result.ok is False
    assert result.exit_code == 1
    assert result.report["status"] == "missing_model"
    error = result.report["error"]
    assert isinstance(error, dict)
    assert error["remediation"] == "edge models fetch qwen3.5-9b-4bit"
    assert result.report["zip_path"] is None


def test_export_scaffold_writes_stable_output_zip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "Qwen3.5-9B-4bit"
    model_dir.mkdir(parents=True)
    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    temp_zip = temp_dir / "FinanceAgent.zip"
    temp_zip.write_bytes(b"zip")
    calls: dict[str, object] = {}

    monkeypatch.setattr(export_scaffold, "where_model", lambda *_args, **_kwargs: _where_ok(model_dir))

    def fake_exporter(model_dir_arg: str, **kwargs) -> ScaffoldZipResult:
        calls["model_dir"] = model_dir_arg
        calls.update(kwargs)
        return ScaffoldZipResult(
            success=True,
            zip_path=str(temp_zip),
            zip_size_bytes=temp_zip.stat().st_size,
            app_name=str(kwargs["app_name"]),
            model_name=model_dir.name,
            model_dir=model_dir_arg,
            direction_set_id="finance_consumer",
        )

    output_dir = tmp_path / "exports"
    result = export_scaffold.run_export_scaffold(
        options=export_scaffold.ExportScaffoldOptions(
            model_ref="qwen3.5-9b-4bit",
            app_name="FinanceAgent",
            system_prompt="Protect cash flow.",
            bundle_id="com.example.financeagent",
            team_id="TEAMID",
            output_path=output_dir,
            dsr_budget=128,
        ),
        exporter=fake_exporter,
    )

    stable_zip = output_dir / "FinanceAgent.zip"
    assert result.ok is True
    assert result.exit_code == 0
    assert stable_zip.read_bytes() == b"zip"
    assert result.report["zip_path"] == str(stable_zip.resolve())
    assert result.report["zip_size_bytes"] == 3
    assert result.report["model_dir"] == str(model_dir)
    assert result.report["direction_set_id"] == "finance_consumer"
    assert calls["model_dir"] == str(model_dir)
    assert calls["app_name"] == "FinanceAgent"
    assert calls["system_prompt"] == "Protect cash flow."
    assert calls["bundle_id"] == "com.example.financeagent"
    assert calls["team_id"] == "TEAMID"
    assert calls["direction_set_id"] is None
    assert calls["enable_dsr"] is True
    assert calls["dsr_budget"] == 128
    assert not temp_zip.exists()


def test_export_scaffold_core_failure_is_reported_with_remediation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "Qwen3.5-9B-4bit"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(export_scaffold, "where_model", lambda *_args, **_kwargs: _where_ok(model_dir))

    def fake_exporter(*_args, **_kwargs) -> ScaffoldZipResult:
        return ScaffoldZipResult(success=False, error="xcodegen not found. Install it: brew install xcodegen")

    result = export_scaffold.run_export_scaffold(
        options=export_scaffold.ExportScaffoldOptions(model_ref="qwen3.5-9b-4bit"),
        exporter=fake_exporter,
    )

    assert result.ok is False
    assert result.exit_code == 1
    assert result.report["status"] == "export_failed"
    error = result.report["error"]
    assert isinstance(error, dict)
    assert error["message"] == "xcodegen not found. Install it: brew install xcodegen"
    assert error["remediation"] == "brew install xcodegen"


def test_edge_export_scaffold_dispatches_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*, options: export_scaffold.ExportScaffoldOptions):
        captured["options"] = options
        return export_scaffold.ExportScaffoldResult(
            ok=True,
            exit_code=0,
            report={
                "schema_version": export_scaffold.EXPORT_SCAFFOLD_SCHEMA_VERSION,
                "ok": True,
                "status": "completed",
                "zip_path": str(tmp_path / "exports" / "FinanceAgent.zip"),
            },
        )

    monkeypatch.setattr(cli_main, "run_export_scaffold", fake_run)

    exit_code = cli_main.main(
        [
            "export",
            "scaffold",
            "--model",
            "qwen3.5-9b-4bit",
            "--app-name",
            "FinanceAgent",
            "--system-prompt",
            "Protect cash flow.",
            "--bundle-id",
            "com.example.financeagent",
            "--team-id",
            "TEAMID",
            "--direction-set-id",
            "finance_consumer",
            "--output",
            str(tmp_path / "exports"),
            "--no-dsr",
            "--dsr-budget",
            "64",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    options = captured["options"]
    assert isinstance(options, export_scaffold.ExportScaffoldOptions)
    assert options.model_ref == "qwen3.5-9b-4bit"
    assert options.app_name == "FinanceAgent"
    assert options.system_prompt == "Protect cash flow."
    assert options.bundle_id == "com.example.financeagent"
    assert options.team_id == "TEAMID"
    assert options.direction_set_id == "finance_consumer"
    assert options.output_path == tmp_path / "exports"
    assert options.enable_dsr is False
    assert options.dsr_budget == 64
