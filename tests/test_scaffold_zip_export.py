# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for scaffold ZIP export without XcodeGen or real model files."""

from __future__ import annotations

import json
import struct
import zipfile as zipfile_module
import zipfile
from pathlib import Path

import pytest

from backend.core import scaffold_zip_export as scaffold
from backend.services import scaffold_template


def _write_safetensors(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode("utf-8")
    max_end = max(
        (info.get("data_offsets", [0, 0])[1] for name, info in header.items() if name != "__metadata__"),
        default=0,
    )
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + (b"\0" * max_end))


def _write_minimal_scaffold(root: Path) -> None:
    (root / ".scaffold_version").write_text("3\n", encoding="utf-8")
    (root / ".min_runtime_version").write_text("1.2.3\n", encoding="utf-8")
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".codex").mkdir()
    (root / "EdgeScaffold/App").mkdir(parents=True)
    (root / "EdgeScaffold/AI").mkdir(parents=True)
    (root / "Resources/RPP").mkdir(parents=True)
    (root / "archive").mkdir()
    (root / "docs/eval").mkdir(parents=True)

    (root / ".github/workflows/ci.yml").write_text("name: ci\n", encoding="utf-8")
    (root / ".codex/config.json").write_text("{}\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("agent notes\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("review notes\n", encoding="utf-8")
    (root / "archive/old.txt").write_text("old\n", encoding="utf-8")
    (root / "docs/eval/run.log").write_text("eval log\n", encoding="utf-8")
    (root / "EdgeScaffold/App/EdgeScaffoldApp.swift").write_text(
        "import SwiftUI\nstruct EdgeScaffoldApp: App { var body: some Scene { WindowGroup { Text(\"hi\") } } }\n",
        encoding="utf-8",
    )
    (root / "EdgeScaffold/App/ScaffoldConfig.swift").write_text(
        """
enum ModelCategory { case llm, vlm, tts, asr }
struct ScaffoldConfig {
    static let appName = "EdgeScaffold"
    static let defaultSystemPrompt = "Default"
    static let modelID: String = "old-model"
    static let modelDisplayName: String = "old display"
    static let modelSizeGB: Double = 0.0
    static let bundleModelName: String? = nil
    static let modelCategory: ModelCategory = .llm
    static let rppModelFamily: String = ""
    static let rppHiddenSize: Int = 0
    static let rppLayerCount: Int = 0
    static let rppDirectionsAResourceName: String = ""
    static let rppTargetLayer: Int = -1
    static let defaultSampleDomainID: String = "finance"
    static let enableDSR: Bool = true
    static let dsrBudget: Int? = nil
}
""",
        encoding="utf-8",
    )
    (root / "EdgeScaffold/AI/AIManager.swift").write_text(
        "final class AIManager { let bundleModelName = ScaffoldConfig.bundleModelName }\n",
        encoding="utf-8",
    )
    (root / "EdgeScaffold/Info.plist").write_text("<string>EdgeScaffold</string>\n", encoding="utf-8")
    (root / "EdgeScaffold/EdgeScaffold.entitlements").write_text("{}\n", encoding="utf-8")
    (root / "edgescaffolding_model_config").write_text("", encoding="utf-8")
    (root / "project.yml").write_text(
        """
name: EdgeScaffold
packages:
  EdgeKit:
    path: ../edge-kit
targets:
  EdgeScaffold:
    type: application
    platform: iOS
    settings:
      PRODUCT_BUNDLE_IDENTIFIER: com.atomgradient.EdgeScaffold
      INFOPLIST_KEY_CFBundleDisplayName: EdgeScaffold
      DEVELOPMENT_TEAM: ABC123
    postBuildScripts:
      - script: |
          source edgescaffolding_model_config
        name: Copy Local Model
    dependencies:
    entitlements:
      path: EdgeScaffold/EdgeScaffold.entitlements
""",
        encoding="utf-8",
    )
    for idx in range(8):
        (root / "EdgeScaffold" / f"Extra{idx}.swift").write_text(f"// {idx}\n", encoding="utf-8")


def _write_scaffold_archive(archive_path: Path, scaffold_root: Path) -> None:
    with zipfile_module.ZipFile(archive_path, "w") as zf:
        for path in scaffold_root.rglob("*"):
            if path.is_file():
                rel = path.relative_to(scaffold_root)
                zf.write(path, f"edge-scaffold-{scaffold_template.EDGE_SCAFFOLD_TEMPLATE_REF}/{rel}")


def test_export_scaffold_zip_creates_expected_archive_with_mocked_xcodegen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models" / "Qwen3-Test-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    _write_safetensors(
        model_dir / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
    )

    scaffold_src = tmp_path / "edge-scaffold"
    scaffold_src.mkdir()
    _write_minimal_scaffold(scaffold_src)
    monkeypatch.setattr(scaffold, "SCAFFOLD_SRC", str(scaffold_src))

    artifact = tmp_path / "directions_a_layer_3.safetensors"
    report = tmp_path / "directions_a_layer_3_report.json"
    artifact.write_bytes(b"fake-rpp")
    report.write_text('{"ok": true}', encoding="utf-8")
    monkeypatch.setattr(
        scaffold,
        "_select_rpp_a_library",
        lambda *_args, **_kwargs: {
            "library_id": "lib-qwen-finance",
            "direction_set_id": "finance_consumer",
            "model_family": "qwen",
            "hidden_size": 4096,
            "layer_count": 32,
            "target_layer": 3,
            "artifact": artifact.name,
            "artifact_path": str(artifact),
            "health_report": report.name,
            "health_report_path": str(report),
        },
    )

    def fake_xcodegen(scaffold_dest: str, safe_name: str) -> None:
        project_dir = Path(scaffold_dest) / f"{safe_name}.xcodeproj"
        project_dir.mkdir()
        (project_dir / "project.pbxproj").write_text("// generated\n", encoding="utf-8")

    def fake_odr_patch(scaffold_dest: str, safe_name: str, _model_dir: str, model_name: str) -> None:
        pbxproj = Path(scaffold_dest) / f"{safe_name}.xcodeproj" / "project.pbxproj"
        pbxproj.write_text(f"KnownAssetTags\n{model_name}\n", encoding="utf-8")

    monkeypatch.setattr(scaffold, "_run_xcodegen", fake_xcodegen)
    monkeypatch.setattr(scaffold, "_patch_xcodeproj_odr", fake_odr_patch)

    progress: list[tuple[str, float]] = []
    result = scaffold.export_scaffold_zip(
        str(model_dir),
        app_name="test app!",
        system_prompt='Say "hi"\nnow',
        bundle_id="com.example.testapp",
        team_id="TEAMID",
        direction_set_id="finance_consumer",
        dsr_budget=128,
        progress_callback=lambda message, percent: progress.append((message, percent)),
    )

    assert result.success is True, result.error
    assert result.app_name == "test app!"
    assert result.model_name == "Qwen3-Test-4bit"
    assert result.direction_set_id == "finance_consumer"
    assert progress[-1] == ("Export complete!", 1.0)

    with zipfile.ZipFile(result.zip_path) as zf:
        names = set(zf.namelist())
        assert "TestApp/project.yml" in names
        assert "TestApp/TestApp/App/TestAppApp.swift" in names
        assert "TestApp/TestApp/App/ScaffoldConfig.swift" in names
        assert "TestApp/TestApp.xcodeproj/project.pbxproj" in names
        assert "TestApp/README.md" in names
        assert "TestApp/Resources/RPP/finance_consumer_qwen_layer_3.safetensors" in names
        assert all("Qwen3-Test-4bit/model.safetensors" not in name for name in names)
        assert "TestApp/AGENTS.md" not in names
        assert "TestApp/CLAUDE.md" not in names
        assert all(not name.startswith("TestApp/.github/") for name in names)
        assert all(not name.startswith("TestApp/.codex/") for name in names)
        assert all(not name.startswith("TestApp/archive/") for name in names)
        assert all(not name.startswith("TestApp/docs/") for name in names)

        config = zf.read("TestApp/TestApp/App/ScaffoldConfig.swift").decode("utf-8")
        assert 'static let appName = "test app!"' in config
        assert 'static let modelID: String = "Qwen3-Test-4bit"' in config
        assert 'static let bundleModelName: String? = "Qwen3-Test-4bit"' in config
        assert 'static let rppDirectionsAResourceName: String = "finance_consumer_qwen_layer_3"' in config
        assert "Say \\\"hi\\\"\\nnow" in config

        project_yml = zf.read("TestApp/project.yml").decode("utf-8")
        assert "name: TestApp" in project_yml
        assert "PRODUCT_BUNDLE_IDENTIFIER: com.example.testapp" in project_yml
        assert "DEVELOPMENT_TEAM: TEAMID" in project_yml
        assert "url: https://github.com/AtomGradient/edge-kit.git" in project_yml
        assert "git@github.com:AtomGradient/edge-kit.git" not in project_yml
        assert "exactVersion: 1.2.3" in project_yml

        readme = zf.read("TestApp/README.md").decode("utf-8")
        assert "Generated by **Edge Studio** from the public" in readme
        assert "This README is for this exported app instance" in readme
        assert "- App display name: `test app!`" in readme
        assert "- Xcode target: `TestApp`" in readme
        assert "- Model: `Qwen3-Test-4bit`" in readme
        assert "- ODR tag: `model`" in readme
        assert "The model weights are not included in this ZIP" in readme
        assert "`TestApp_model_config`" in readme
        assert "`TestApp/App/ScaffoldConfig.swift`" in readme
        assert "EdgeHalo binary package" in readme
        assert "`Resources/RPP` is empty" in readme
        assert "## What This Repo Is" not in readme
        assert "Current minimum EdgeKit version" not in readme


def test_scaffold_template_downloads_fixed_public_archive_when_not_local(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "archive-source"
    source.mkdir()
    _write_minimal_scaffold(source)
    archive = tmp_path / "edge-scaffold.zip"
    _write_scaffold_archive(archive, source)

    class FakeResponse:
        def __enter__(self):
            self._file = archive.open("rb")
            return self._file

        def __exit__(self, *_exc):
            self._file.close()
            return False

    monkeypatch.delenv(scaffold_template.EDGE_SCAFFOLD_DIR_ENV, raising=False)
    monkeypatch.setenv("EDGESTUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(scaffold_template, "source_tree_scaffold_source", lambda: None)
    monkeypatch.setattr(scaffold_template.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    resolved = scaffold_template.resolve_scaffold_source()

    assert resolved.is_dir()
    assert resolved.name == f"edge-scaffold-{scaffold_template.EDGE_SCAFFOLD_TEMPLATE_REF[:12]}"
    assert (resolved / "project.yml").is_file()
    assert (resolved / ".min_runtime_version").read_text(encoding="utf-8").strip() == "1.2.3"


def test_export_scaffold_zip_uses_cached_public_template_without_monorepo_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models" / "Qwen3-Test-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    _write_safetensors(
        model_dir / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
    )

    cached = tmp_path / "cache" / "scaffold_templates" / f"edge-scaffold-{scaffold_template.EDGE_SCAFFOLD_TEMPLATE_REF[:12]}"
    cached.mkdir(parents=True)
    _write_minimal_scaffold(cached)

    artifact = tmp_path / "directions_a_layer_3.safetensors"
    report = tmp_path / "directions_a_layer_3_report.json"
    artifact.write_bytes(b"fake-rpp")
    report.write_text('{"ok": true}', encoding="utf-8")

    monkeypatch.delenv(scaffold_template.EDGE_SCAFFOLD_DIR_ENV, raising=False)
    monkeypatch.setenv("EDGESTUDIO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(scaffold_template, "source_tree_scaffold_source", lambda: None)
    monkeypatch.setattr(scaffold, "SCAFFOLD_SRC", "")
    monkeypatch.setattr(
        scaffold,
        "_select_rpp_a_library",
        lambda *_args, **_kwargs: {
            "library_id": "lib-qwen-finance",
            "direction_set_id": "finance_consumer",
            "model_family": "qwen",
            "hidden_size": 4096,
            "layer_count": 32,
            "target_layer": 3,
            "artifact": artifact.name,
            "artifact_path": str(artifact),
            "health_report": report.name,
            "health_report_path": str(report),
        },
    )

    monkeypatch.setattr(scaffold, "_run_xcodegen", lambda dest, name: (Path(dest) / f"{name}.xcodeproj").mkdir())
    monkeypatch.setattr(
        scaffold,
        "_patch_xcodeproj_odr",
        lambda dest, name, _model_dir, model_name: (
            Path(dest) / f"{name}.xcodeproj" / "project.pbxproj"
        ).write_text(f"KnownAssetTags\n{model_name}\n", encoding="utf-8"),
    )

    result = scaffold.export_scaffold_zip(str(model_dir), app_name="Pip User App")

    assert result.success is True, result.error
    with zipfile.ZipFile(result.zip_path) as zf:
        project_yml = zf.read("PipUserApp/project.yml").decode("utf-8")
        assert "url: https://github.com/AtomGradient/edge-kit.git" in project_yml
        assert "path: ../edge-kit" not in project_yml


def test_export_scaffold_zip_without_a_library_still_creates_app_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models" / "Qwen3-Test-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    _write_safetensors(
        model_dir / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
    )

    scaffold_src = tmp_path / "edge-scaffold"
    scaffold_src.mkdir()
    _write_minimal_scaffold(scaffold_src)
    monkeypatch.setattr(scaffold, "SCAFFOLD_SRC", str(scaffold_src))
    monkeypatch.setattr(scaffold, "_select_rpp_a_library", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(scaffold, "_run_xcodegen", lambda dest, name: (Path(dest) / f"{name}.xcodeproj").mkdir())
    monkeypatch.setattr(
        scaffold,
        "_patch_xcodeproj_odr",
        lambda dest, name, _model_dir, model_name: (
            Path(dest) / f"{name}.xcodeproj" / "project.pbxproj"
        ).write_text(f"KnownAssetTags\n{model_name}\n", encoding="utf-8"),
    )

    result = scaffold.export_scaffold_zip(str(model_dir), app_name="No RPP App")

    assert result.success is True, result.error
    assert result.direction_set_id == "finance_consumer"
    with zipfile.ZipFile(result.zip_path) as zf:
        config = zf.read("NoRPPApp/NoRPPApp/App/ScaffoldConfig.swift").decode("utf-8")
        assert 'static let rppModelFamily: String = ""' in config
        assert "static let rppHiddenSize: Int = 0" in config
        assert "static let rppLayerCount: Int = 0" in config
        assert 'static let rppDirectionsAResourceName: String = ""' in config
        assert "static let rppTargetLayer: Int = -1" in config
        assert "NoRPPApp/Resources/RPP/rpp_a_library_manifest.json" not in zf.namelist()


def test_export_scaffold_zip_explicit_a_library_request_still_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "models" / "Qwen3-Test-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    _write_safetensors(
        model_dir / "model.safetensors",
        {"model.embed_tokens.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}},
    )

    scaffold_src = tmp_path / "edge-scaffold"
    scaffold_src.mkdir()
    _write_minimal_scaffold(scaffold_src)
    monkeypatch.setenv("EDGESTUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv(scaffold_template.EDGE_SCAFFOLD_DIR_ENV, str(scaffold_src))
    monkeypatch.setattr(scaffold, "SCAFFOLD_SRC", str(scaffold_src))

    result = scaffold.export_scaffold_zip(
        str(model_dir),
        app_name="Needs RPP App",
        direction_set_id="finance_consumer",
    )

    assert result.success is False
    assert "[a-library] No matching RPP A-library" in result.error


def test_export_scaffold_zip_reports_missing_model_dir(tmp_path: Path) -> None:
    result = scaffold.export_scaffold_zip(str(tmp_path / "missing"), app_name="Missing")

    assert result.success is False
    assert "Model directory not found" in result.error
