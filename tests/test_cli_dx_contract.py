# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Developer-facing CLI contracts caught by cold-start DX testing."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.cli import doctor, main, model_fetch, models
from backend.cli.fingerprints import model_dir_integrity
from backend.cli.model_fetch import CommandResult, FetchOptions
from backend.cli.models import CatalogResolution


def test_edge_version_flag_prints_distribution_name(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main.main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("edge-studio ")


def test_doctor_checks_public_distribution_name(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_version(package: str) -> str:
        assert package != "edgestudio"
        seen.append(package)
        return "1.0"

    monkeypatch.setattr(doctor.importlib.metadata, "version", fake_version)

    result = doctor._check_python_packages()

    assert result.status == "ok"
    assert "edge-studio" in seen
    assert result.details["missing"] == []


def test_model_integrity_rejects_partial_download_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3.5-9B-4bit"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"not a real safetensors file")
    (model_dir / "model.safetensors.aria2").write_text("partial", encoding="utf-8")

    integrity = model_dir_integrity(model_dir)

    assert integrity.complete is False
    assert "partial_download_files_present" in integrity.issues
    assert any(issue.startswith("invalid_safetensors:model.safetensors") for issue in integrity.issues)


def test_where_marks_catalog_match_incomplete_when_size_is_too_small(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "Qwen3.5-9B-4bit"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"not a real safetensors file")

    resolution = CatalogResolution(
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
    monkeypatch.setattr(models, "resolve_model_reference", lambda _model_ref: resolution)
    monkeypatch.setattr(
        models,
        "discover_local_model_paths",
        lambda _env=None: {"[Local] Qwen3.5-9B-4bit": str(model_dir)},
    )

    report = models.where_model("qwen3.5-9b-4bit")

    assert report.status == "incomplete"
    assert report.fetch_command == "edge models fetch qwen3.5-9b-4bit"
    assert report.local_matches[0].complete is False
    assert any(issue.startswith("size_below_expected:") for issue in report.local_matches[0].issues)


def test_fetch_success_with_bad_local_dir_returns_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = CatalogResolution(
        status="resolved",
        input="qwen3.5-9b-4bit",
        model_id="qwen3.5-9b-4bit",
        name="Qwen3.5-9B-4bit",
        download_hint="mlx-community/Qwen3.5-9B-4bit",
        category="llm",
        size_gb=None,
        catalog_source="test",
        catalog_version="test",
        matched_by="id",
        alternates=[],
    )

    class _Where:
        status = "missing"
        local_matches: list[object] = []

    def fake_runner(_args, _env, _timeout):
        return CommandResult(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(model_fetch, "resolve_model_reference", lambda _model_ref: resolution)
    monkeypatch.setattr(model_fetch, "where_model", lambda *_args, **_kwargs: _Where())

    result = model_fetch.fetch_model(
        "qwen3.5-9b-4bit",
        options=FetchOptions(source="huggingface", download_dir=tmp_path),
        runner=fake_runner,
    )

    assert result.ok is False
    assert result.status == "download_incomplete"
    assert result.exit_code == 1
    retry = str(result.receipt["retry_command"])
    assert retry.startswith("edge models fetch qwen3.5-9b-4bit --source huggingface --retry")
    assert "--download-dir" in retry
