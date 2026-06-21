# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Scaffold App ZIP export.

Generates a lightweight ZIP with:
  - {AppName}/ — iOS App (from EdgeScaffold template)
  - README.md

EdgeKit and the binary EdgeHalo package are referenced via GitHub URL (SPM), not
embedded as source.
Model files are NOT included (too large) — referenced in .xcodeproj via
absolute path with ODR resource tags, so the existing EdgeScaffold
loading strategy (Documents → Cache → Bundle → ODR → HuggingFace) works
out of the box.
"""

from __future__ import annotations

import os
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from typing import Callable

from backend.services.scaffold_template import (
    EDGE_SCAFFOLD_DIR_ENV,
    ScaffoldTemplateError,
    resolve_scaffold_source,
    source_tree_scaffold_source,
)

# ─────────────────────────────────────────────
# Source paths
# ─────────────────────────────────────────────
_SOURCE_TREE_SCAFFOLD = source_tree_scaffold_source()
SCAFFOLD_SRC = str(_SOURCE_TREE_SCAFFOLD) if _SOURCE_TREE_SCAFFOLD is not None else ""

_export_lock = threading.Lock()

_EXCLUDE_DIRS = {
    ".github",
    ".git",
    ".build",
    ".ai-mailbox",
    ".claude",
    ".codex",
    ".vscode",
    ".pytest_cache",
    "archive",
    "DerivedData",
    "build",
    "docs",
    "xcuserdata",
}
_EXCLUDE_FILES = {".DS_Store", "AGENTS.md", "CLAUDE.md", "Package.resolved"}
_MIN_SCAFFOLD_VERSION = 3
_DEFAULT_SCAFFOLD_DIRECTION_SET_ID = "finance_consumer"
_DIRECTION_SET_TO_SAMPLE_DOMAIN = {
    "finance_consumer": "finance",
    "health_fitness": "health",
    "reading_learning": "reading",
    "journal_reflection": "journal",
    "travel_explorer": "travel",
    "cooking_kitchen": "cooking",
    "music_media": "music",
    "work_productivity": "work",
}

ProgressCallback = Callable[[str, float], None] | None


class ScaffoldExportError(Exception):
    """Raised when a specific export step fails with a clear diagnostic."""
    pass


def _read_min_runtime_version(scaffold_src: str) -> str:
    """Read minimum EdgeKit version from EdgeScaffold's version contract file."""
    ver_file = os.path.join(scaffold_src, ".min_runtime_version")
    if not os.path.isfile(ver_file):
        raise ScaffoldExportError(
            "[version-contract] .min_runtime_version not found in EdgeScaffold. "
            f"Set {EDGE_SCAFFOLD_DIR_ENV} to a valid edge-scaffold checkout or retry the template download."
        )
    with open(ver_file) as f:
        ver = f.read().strip()
    if not ver:
        raise ScaffoldExportError("[version-contract] .min_runtime_version is empty")
    return ver


def _assert_file(path: str, step: str):
    if not os.path.isfile(path):
        raise ScaffoldExportError(f"[{step}] Expected file not found: {path}")


def _assert_dir(path: str, step: str):
    if not os.path.isdir(path):
        raise ScaffoldExportError(f"[{step}] Expected directory not found: {path}")


def _assert_contains(path: str, keyword: str, step: str):
    with open(path) as f:
        if keyword not in f.read():
            raise ScaffoldExportError(f"[{step}] {os.path.basename(path)} missing expected content: '{keyword}'")


@dataclass
class ScaffoldZipResult:
    success: bool = False
    zip_path: str = ""
    zip_size_bytes: int = 0
    app_name: str = ""
    model_name: str = ""
    model_dir: str = ""
    model_tier: str = ""
    direction_set_id: str = ""
    error: str = ""


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def export_scaffold_zip(
    model_dir: str,
    app_name: str = "MyApp",
    system_prompt: str = "You are a helpful assistant.",
    model_tier: str = "",
    progress_callback: ProgressCallback = None,
    enable_dsr: bool = True,
    dsr_budget: int | None = None,
    bundle_id: str | None = None,
    team_id: str | None = None,
    direction_set_id: str | None = None,
    enable_h2o: bool | None = None,
    h2o_budget: int | None = None,
) -> ScaffoldZipResult:
    """Generate a self-contained Scaffold App ZIP.

    Args:
        model_dir: Path to the model directory (not included in ZIP, referenced
                   in .xcodeproj with ODR resource tags).
        app_name: Display name for the iOS app.
        system_prompt: Default system prompt baked into ScaffoldConfig.
        model_tier: Deprecated, kept for API compatibility. Ignored.
        progress_callback: (message, fraction) callback.
    """
    if enable_h2o is not None:
        enable_dsr = enable_h2o
    if h2o_budget is not None:
        dsr_budget = h2o_budget

    if not _export_lock.acquire(timeout=5):
        return ScaffoldZipResult(success=False, error="Another export is in progress. Please wait.")

    result = ScaffoldZipResult(app_name=app_name, model_dir=model_dir)

    def _p(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)

    tmp_dir = None
    try:
        # ── Step 1: Validate ──────────────────────
        _p("Validating inputs...", 0.05)
        if not os.path.isdir(model_dir):
            result.error = f"Model directory not found: {model_dir}"
            return result
        try:
            scaffold_src = _resolve_scaffold_src()
        except ScaffoldTemplateError as e:
            result.error = str(e)
            return result
        if not os.path.isdir(scaffold_src):
            result.error = f"EdgeScaffold not found: {scaffold_src}"
            return result

        # ── Step 1.5: Scaffold version check ─────
        version_file = os.path.join(scaffold_src, ".scaffold_version")
        if os.path.isfile(version_file):
            with open(version_file) as f:
                ver = int(f.read().strip())
            if ver < _MIN_SCAFFOLD_VERSION:
                result.error = (
                    f"EdgeScaffold template too old (v{ver}, need v{_MIN_SCAFFOLD_VERSION}). "
                    f"Please update {scaffold_src} or clear the cached template."
                )
                return result
        else:
            result.error = (
                f"EdgeScaffold missing .scaffold_version file. "
                f"Please update {scaffold_src} or clear the cached template."
            )
            return result

        # Derive names
        safe_name = _sanitize_app_name(app_name)
        model_name = os.path.basename(model_dir.rstrip("/"))
        result.model_name = model_name
        result.model_tier = "single"  # Single model, no tier selection
        rpp_library = _select_rpp_a_library(
            model_dir,
            direction_set_id=direction_set_id,
            required=bool((direction_set_id or "").strip()),
        )
        if rpp_library:
            result.direction_set_id = str(rpp_library.get("direction_set_id") or "")
            rpp_profile = _rpp_a_library_profile(rpp_library)
        else:
            result.direction_set_id = (direction_set_id or _DEFAULT_SCAFFOLD_DIRECTION_SET_ID).strip()
            rpp_profile = _empty_rpp_a_library_profile(result.direction_set_id)

        # ── Step 2: Create temp dir ──────────────
        _p("Creating workspace...", 0.10)
        tmp_dir = tempfile.mkdtemp(prefix="scaffold_")

        # ── Step 3: Copy EdgeScaffold ───────────
        _p("Copying EdgeScaffold template...", 0.20)
        scaffold_dest = os.path.join(tmp_dir, safe_name)
        _copytree_filtered(scaffold_src, scaffold_dest, _EXCLUDE_DIRS, _EXCLUDE_FILES)
        _assert_dir(scaffold_dest, "Step 3: Copy EdgeScaffold")
        _assert_file(os.path.join(scaffold_dest, "project.yml"), "Step 3: Copy EdgeScaffold")
        if rpp_library:
            _install_selected_a_library(scaffold_dest, rpp_library)

        # ── Step 4: Rename project dirs + files ──
        _p("Renaming project...", 0.35)
        _rename_project(scaffold_dest, safe_name)
        _assert_dir(os.path.join(scaffold_dest, safe_name), "Step 4: Rename project")
        _assert_file(
            os.path.join(scaffold_dest, safe_name, "App", f"{safe_name}App.swift"),
            "Step 4: Rename project",
        )

        # ── Step 5: Customize ScaffoldConfig.swift
        _p("Customizing ScaffoldConfig...", 0.45)
        # Detect model category from config.json
        from backend.core.model_category import detect_model_category, ensure_tokenizer_json
        try:
            import json as _json
            with open(os.path.join(model_dir, "config.json")) as _cf:
                _model_cfg = _json.load(_cf)
            _model_category = detect_model_category(_model_cfg).value
        except (ImportError, OSError, ValueError, KeyError):
            _model_category = "llm"
        # TTS models need tokenizer.json for Swift's swift-transformers
        if _model_category == "tts":
            ensure_tokenizer_json(model_dir)
        _customize_scaffold_config(
            scaffold_dest,
            safe_name,
            app_name,
            system_prompt,
            model_name,
            model_dir,
            _model_category,
            enable_dsr,
            dsr_budget,
            rpp_profile,
        )
        config_swift = os.path.join(scaffold_dest, safe_name, "App", "ScaffoldConfig.swift")
        _assert_contains(config_swift, app_name, "Step 5: ScaffoldConfig")
        _assert_contains(config_swift, model_name, "Step 5: ScaffoldConfig bundleModelName")

        # ── Step 5b: Patch AIManager if needed ────
        _inject_bundle_strategy(scaffold_dest, safe_name)

        # ── Step 5c: Patch Info.plist bundle names ────
        _patch_info_plist(scaffold_dest, safe_name, app_name)

        # ── Step 6: Customize model_config ────────
        _p("Customizing model config...", 0.50)
        _customize_model_config(scaffold_dest, safe_name, model_dir, model_name)
        _assert_file(
            os.path.join(scaffold_dest, f"{safe_name}_model_config"),
            "Step 6: model_config",
        )

        # ── Step 7: Customize project.yml ────────
        _p("Customizing project.yml...", 0.55)
        _customize_project_yml(
            scaffold_dest,
            safe_name,
            app_name,
            model_name,
            model_dir,
            bundle_id,
            team_id,
        )
        yml_path = os.path.join(scaffold_dest, "project.yml")
        _assert_contains(yml_path, f"name: {safe_name}", "Step 7: project.yml target name")
        _assert_contains(yml_path, f"{safe_name}_model_config", "Step 7: project.yml model config ref")

        # ── Step 8: Rename Swift entry struct ────
        _p("Updating Swift entry point...", 0.65)
        _rename_entry_struct(scaffold_dest, safe_name)
        _assert_contains(
            os.path.join(scaffold_dest, safe_name, "App", f"{safe_name}App.swift"),
            f"struct {safe_name}App:",
            "Step 8: Entry struct rename",
        )

        # ── Step 9: Run xcodegen ────────────────
        _p("Generating Xcode project...", 0.73)
        _run_xcodegen(scaffold_dest, safe_name)
        _assert_dir(
            os.path.join(scaffold_dest, f"{safe_name}.xcodeproj"),
            "Step 9: xcodegen — .xcodeproj not generated (is xcodegen installed?)",
        )

        # ── Step 10: Patch .xcodeproj with ODR ───
        _p("Configuring ODR resource tags...", 0.82)
        _patch_xcodeproj_odr(scaffold_dest, safe_name, model_dir, model_name)
        pbxproj = os.path.join(scaffold_dest, f"{safe_name}.xcodeproj", "project.pbxproj")
        _assert_contains(pbxproj, "KnownAssetTags", "Step 10: ODR KnownAssetTags")
        _assert_contains(pbxproj, model_name, "Step 10: ODR model reference")

        # ── Step 11: Generate README ─────────────
        _p("Generating README...", 0.90)
        _generate_readme(scaffold_dest, safe_name, app_name, model_name, model_dir)

        # ── Step 12: Create ZIP ──────────────────
        _p("Creating ZIP archive...", 0.95)
        zip_base = os.path.join(tmp_dir, safe_name)
        zip_path = shutil.make_archive(zip_base, "zip", tmp_dir, safe_name)

        # ── Step 13: Validate ZIP ─────────────────
        _p("Validating ZIP...", 0.97)
        _validate_zip(zip_path, safe_name)

        result.zip_path = zip_path
        result.zip_size_bytes = os.path.getsize(zip_path)

        _p("Export complete!", 1.0)
        result.success = True

    except ScaffoldExportError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"Unexpected error: {type(e).__name__}: {e}"

    finally:
        # Clean up the staging directory but keep the ZIP
        if tmp_dir:
            scaffold_dest = os.path.join(tmp_dir, _sanitize_app_name(app_name))
            if os.path.isdir(scaffold_dest):
                shutil.rmtree(scaffold_dest, ignore_errors=True)
        _export_lock.release()

    return result


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _sanitize_app_name(name: str) -> str:
    """Sanitize app name to a valid Swift identifier / directory name."""
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", name).strip()
    if not cleaned:
        cleaned = "MyApp"
    return "".join(w[:1].upper() + w[1:] for w in cleaned.split())


def _resolve_scaffold_src() -> str:
    if os.environ.get(EDGE_SCAFFOLD_DIR_ENV, "").strip():
        return str(resolve_scaffold_source())
    if SCAFFOLD_SRC and os.path.isdir(SCAFFOLD_SRC):
        return SCAFFOLD_SRC
    return str(resolve_scaffold_source())


def _copytree_filtered(src: str, dst: str, exclude_dirs: set, exclude_files: set):
    """Copy directory tree, skipping excluded dirs and files."""
    def _ignore(directory: str, contents: list[str]) -> list[str]:
        ignored = []
        for item in contents:
            if item in exclude_dirs:
                ignored.append(item)
            elif item in exclude_files:
                ignored.append(item)
            elif item.endswith(".xcodeproj"):
                ignored.append(item)
        return ignored

    shutil.copytree(src, dst, ignore=_ignore)


def _rename_project(scaffold_dest: str, safe_name: str):
    """Rename EdgeScaffold dirs/files to {safe_name}."""
    inner_old = os.path.join(scaffold_dest, "EdgeScaffold")
    inner_new = os.path.join(scaffold_dest, safe_name)
    if os.path.isdir(inner_old):
        os.rename(inner_old, inner_new)
    elif not os.path.isdir(inner_new):
        legacy_inner_old = os.path.join(scaffold_dest, "EdgeScaffolding")
        if os.path.isdir(legacy_inner_old):
            os.rename(legacy_inner_old, inner_new)

    app_swift_old = os.path.join(inner_new, "App", "EdgeScaffoldApp.swift")
    app_swift_new = os.path.join(inner_new, "App", f"{safe_name}App.swift")
    if os.path.isfile(app_swift_old):
        os.rename(app_swift_old, app_swift_new)
    else:
        legacy_app_swift_old = os.path.join(inner_new, "App", "EdgeScaffoldingApp.swift")
        if os.path.isfile(legacy_app_swift_old):
            os.rename(legacy_app_swift_old, app_swift_new)

    entitlements_old = os.path.join(inner_new, "EdgeScaffold.entitlements")
    entitlements_new = os.path.join(inner_new, f"{safe_name}.entitlements")
    if os.path.isfile(entitlements_old):
        os.rename(entitlements_old, entitlements_new)
    else:
        legacy_entitlements_old = os.path.join(inner_new, "EdgeScaffolding.entitlements")
        if os.path.isfile(legacy_entitlements_old):
            os.rename(legacy_entitlements_old, entitlements_new)

    # Rename model config file
    config_old = os.path.join(scaffold_dest, "edgescaffolding_model_config")
    config_new = os.path.join(scaffold_dest, f"{safe_name}_model_config")
    if os.path.isfile(config_old):
        os.rename(config_old, config_new)


def _inject_bundle_strategy(scaffold_dest: str, safe_name: str):
    """Inject Bundle-embedded model strategy into AIManager if not present."""
    ai_path = os.path.join(scaffold_dest, safe_name, "AI", "AIManager.swift")
    if not os.path.isfile(ai_path):
        return

    with open(ai_path) as f:
        content = f.read()

    if "bundleModelName" in content:
        return  # Already has the new strategies

    # Inject before "// Strategy 2: ODR" or "// Strategy 3: ODR"
    bundle_strategy = '''
        // Strategy 2: Bundle-embedded model (Build Phase copy)
        if let bundleName = ScaffoldConfig.bundleModelName,
           let bundleURL = Bundle.main.url(forResource: bundleName, withExtension: nil) {
            do {
                try await engine.loadLocal(directory: bundleURL) { [weak self] p in
                    Task { @MainActor [weak self] in
                        self?.loadingProgress = p
                    }
                }
                isModelLoaded = true
                stateManager.setLastLoadedModel(config.modelID)
                return
            } catch {
                debugPrint("[AIManager] Bundle model load failed: \\(error)")
            }
        }

'''
    # Find ODR strategy marker
    for marker in ["// Strategy 2: ODR", "// Strategy 3: ODR"]:
        if marker in content:
            content = content.replace(marker, bundle_strategy.rstrip() + "\n\n        " + marker)
            break

    with open(ai_path, "w") as f:
        f.write(content)


def _model_display_name(model_name: str) -> str:
    """Derive a human-readable display name from model directory name.

    e.g. "Qwen3.5-4B-4bit" → "Qwen3.5 4B 4bit"
    """
    return model_name.replace("-", " ").replace("_", " ")


def _model_size_gb(model_dir: str) -> float:
    """Calculate total safetensors size in GB."""
    total = 0
    for dirpath, _, filenames in os.walk(model_dir):
        for f in filenames:
            if f.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(dirpath, f))
    return round(total / (1024 ** 3), 1)


def _rpp_a_library_profile(selected: dict) -> dict[str, int | str]:
    """Return a config profile from a selected A-library manifest item."""
    direction_set_id = str(selected.get("direction_set_id") or _DEFAULT_SCAFFOLD_DIRECTION_SET_ID)
    artifact_name = _export_a_library_resource_name(selected, "artifact") or str(selected.get("artifact") or "")
    return {
        "model_family": selected.get("model_family", ""),
        "hidden_size": int(selected.get("hidden_size") or 0),
        "layer_count": int(selected.get("layer_count") or 0),
        "resource_name": os.path.splitext(artifact_name)[0],
        "target_layer": int(selected.get("target_layer") or -1),
        "default_sample_domain_id": _sample_domain_id_for_direction_set(direction_set_id),
    }


def _empty_rpp_a_library_profile(direction_set_id: str) -> dict[str, int | str]:
    return {
        "model_family": "",
        "hidden_size": 0,
        "layer_count": 0,
        "resource_name": "",
        "target_layer": -1,
        "default_sample_domain_id": _sample_domain_id_for_direction_set(direction_set_id),
    }


def _select_rpp_a_library(
    model_dir: str,
    *,
    direction_set_id: str | None = None,
    required: bool = False,
) -> dict | None:
    from backend.services.a_library_registry import select_a_library_for_model_dir

    explicit_direction_set_id = (direction_set_id or "").strip()
    requested_direction_set_id = explicit_direction_set_id or _DEFAULT_SCAFFOLD_DIRECTION_SET_ID
    selection = select_a_library_for_model_dir(
        model_dir,
        direction_set_id=requested_direction_set_id,
    )
    if not selection.get("ok") and not explicit_direction_set_id:
        fallback_selection = select_a_library_for_model_dir(
            model_dir,
            direction_set_id="directions_a",
        )
        if fallback_selection.get("ok"):
            selection = fallback_selection
            requested_direction_set_id = "directions_a"
    if not selection.get("ok"):
        if not required:
            return None
        reasons = ", ".join(selection.get("reasons") or ["unknown"])
        raise ScaffoldExportError(
            "[a-library] No matching RPP A-library for this model. "
            f"Run Training → A-library generation first. "
            f"direction_set_id={requested_direction_set_id}; reasons={reasons}"
        )
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ScaffoldExportError("[a-library] Registry returned an invalid selected library")
    return selected


def _sample_domain_id_for_direction_set(direction_set_id: str) -> str:
    return _DIRECTION_SET_TO_SAMPLE_DOMAIN.get(direction_set_id, "finance")


def _install_selected_a_library(scaffold_dest: str, selected: dict) -> None:
    """Copy the selected A-library into the exported app Resources/RPP bundle."""
    rpp_dir = os.path.join(scaffold_dest, "Resources", "RPP")
    os.makedirs(rpp_dir, exist_ok=True)

    artifact_name = _export_a_library_resource_name(selected, "artifact")
    report_name = _export_a_library_resource_name(selected, "health_report")
    export_selected = dict(selected)
    export_selected["artifact"] = artifact_name
    export_selected["health_report"] = report_name

    for key, dest_name in (("artifact", artifact_name), ("health_report", report_name)):
        name = selected.get(key)
        source_key = f"{key}_path" if key == "artifact" else "health_report_path"
        source = selected.get(source_key)
        if not name or not dest_name or not source or not os.path.isfile(str(source)):
            raise ScaffoldExportError(f"[a-library] Missing selected {key}: {name}")
        dest = os.path.join(rpp_dir, dest_name)
        if os.path.realpath(str(source)) != os.path.realpath(dest):
            shutil.copy2(str(source), dest)

    manifest_path = os.path.join(rpp_dir, "rpp_a_library_manifest.json")
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError):
        manifest = {
            "schema_version": "edgestudio.rpp_a_library_manifest.v1",
            "default_library_id": selected.get("library_id", ""),
            "libraries": [],
        }
    libraries = [
        item for item in manifest.get("libraries", [])
        if item.get("library_id") != export_selected.get("library_id")
    ]
    clean = {
        key: value
        for key, value in export_selected.items()
        if key not in {
            "source_manifest",
            "artifact_path",
            "health_report_path",
            "artifact_exists",
            "health_report_exists",
            "artifact_sha256_actual",
            "artifact_sha256_ok",
            "match_reasons",
        }
    }
    libraries.append(clean)
    manifest["libraries"] = libraries
    manifest["default_library_id"] = export_selected.get("library_id", manifest.get("default_library_id", ""))
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _export_a_library_resource_name(selected: dict, key: str) -> str:
    original = os.path.basename(str(selected.get(key) or ""))
    if not original:
        return ""
    direction_set_id = _safe_resource_component(str(selected.get("direction_set_id") or "directions_a"))
    if direction_set_id == "directions_a" or not original.startswith("directions_a_layer_"):
        return original

    model_family = _safe_resource_component(str(selected.get("model_family") or "model"))
    target_layer = int(selected.get("target_layer") or 0)
    if key == "health_report":
        return f"{direction_set_id}_{model_family}_layer_{target_layer}_report.json"
    return f"{direction_set_id}_{model_family}_layer_{target_layer}.safetensors"


def _safe_resource_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _customize_scaffold_config(
    scaffold_dest: str,
    safe_name: str,
    app_name: str,
    system_prompt: str,
    model_name: str,
    model_dir: str,
    model_category: str = "llm",
    enable_dsr: bool = True,
    dsr_budget: int | None = None,
    rpp_profile: dict[str, int | str] | None = None,
):
    """Patch ScaffoldConfig.swift — appName, systemPrompt, modelID, modelCategory."""
    config_path = os.path.join(scaffold_dest, safe_name, "App", "ScaffoldConfig.swift")
    if not os.path.isfile(config_path):
        return

    with open(config_path) as f:
        content = f.read()

    # Update appName
    content = re.sub(
        r'(static let appName = )"[^"]*"',
        rf'\1"{app_name}"',
        content,
    )

    # Update system prompt
    escaped_prompt = (system_prompt
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\0", ""))
    content = re.sub(
        r'(static let defaultSystemPrompt = )"[^"]*"',
        lambda match: f'{match.group(1)}"{escaped_prompt}"',
        content,
    )

    # Update modelID — use exact model folder name (case-sensitive on device).
    model_id = model_name
    content = re.sub(
        r'(static let modelID: String = )"[^"]*"',
        rf'\1"{model_id}"',
        content,
    )

    # Update model display metadata
    display_name = _model_display_name(model_name)
    content = re.sub(
        r'(static let modelDisplayName: String = )"[^"]*"',
        rf'\1"{display_name}"',
        content,
    )
    size_gb = _model_size_gb(model_dir)
    content = re.sub(
        r'(static let modelSizeGB: Double = )[\d.]+',
        rf'\g<1>{size_gb}',
        content,
    )

    # Bind RPP when a model-matched A-library exists. First-run scaffold export can
    # still produce a usable app without RPP assets; RPP features fail closed in-app.
    rpp_profile = rpp_profile or _rpp_a_library_profile({})
    content = re.sub(
        r'(static let rppModelFamily: String = )"[^"]*"',
        rf'\1"{rpp_profile["model_family"]}"',
        content,
    )
    content = re.sub(
        r'(static let rppHiddenSize: Int = )-?\d+',
        rf'\g<1>{rpp_profile["hidden_size"]}',
        content,
    )
    content = re.sub(
        r'(static let rppLayerCount: Int = )-?\d+',
        rf'\g<1>{rpp_profile["layer_count"]}',
        content,
    )
    content = re.sub(
        r'(static let rppDirectionsAResourceName: String = )"[^"]*"',
        rf'\1"{rpp_profile["resource_name"]}"',
        content,
    )
    content = re.sub(
        r'(static let rppTargetLayer: Int = )-?\d+',
        rf'\g<1>{rpp_profile["target_layer"]}',
        content,
    )
    content = re.sub(
        r'(static let defaultSampleDomainID: String = )"[^"]*"',
        rf'\1"{rpp_profile["default_sample_domain_id"]}"',
        content,
    )

    # Set bundleModelName for Strategy 2 (Bundle-embedded model)
    if "bundleModelName" in content:
        content = content.replace(
            "static let bundleModelName: String? = nil",
            f'static let bundleModelName: String? = "{model_name}"',
        )
    else:
        # Old template — inject bundleModelName before enableSustainability
        content = content.replace(
            "    // 功能开关",
            f'    static let bundleModelName: String? = "{model_name}"\n\n    // 功能开关',
        )

    # Set modelCategory for multimodal routing
    if model_category and model_category != "llm":
        content = re.sub(
            r'(static let modelCategory: ModelCategory = )\.\w+',
            rf'\1.{model_category}',
            content,
        )

    # DSR cache retention. Current templates route DSR through EdgeGenerateParameters;
    # keep legacy H2O patches for older templates that still expose these fields.
    if "enableDSR" in content:
        content = re.sub(
            r'(static let enableDSR: Bool = )(true|false)',
            rf'\g<1>{str(enable_dsr).lower()}',
            content,
        )
    if "dsrBudget" in content and dsr_budget is not None:
        content = re.sub(
            r'(static let dsrBudget: Int\? = )(nil|\d+)',
            rf'\g<1>{dsr_budget}',
            content,
        )

    if "enableH2OEviction" in content:
        if not enable_dsr:
            content = content.replace(
                "static let enableH2OEviction: Bool = true",
                "static let enableH2OEviction: Bool = false",
            )
        if dsr_budget is not None:
            content = content.replace(
                "static let h2oBudget: Int? = nil",
                f"static let h2oBudget: Int? = {dsr_budget}",
            )

    with open(config_path, "w") as f:
        f.write(content)


def _patch_info_plist(scaffold_dest: str, safe_name: str, app_name: str):
    """Patch Info.plist — replace hardcoded bundle names with app_name."""
    plist_path = os.path.join(scaffold_dest, safe_name, "Info.plist")
    if not os.path.isfile(plist_path):
        return

    with open(plist_path) as f:
        content = f.read()

    content = content.replace("<string>EdgeScaffold</string>", f"<string>{app_name}</string>")
    content = content.replace("<string>EdgeScaffolding</string>", f"<string>{app_name}</string>")

    with open(plist_path, "w") as f:
        f.write(content)


def _customize_model_config(scaffold_dest: str, safe_name: str, model_dir: str, model_name: str):
    """Update {safe_name}_model_config with actual model info and enable copy."""
    models_source_dir = os.path.dirname(model_dir.rstrip("/"))

    config_content = f'MODEL_NAME={shlex.quote(model_name)}\nMODELS_SOURCE_DIR={shlex.quote(models_source_dir)}\nMODEL_COPY="true"\n'
    config_path = os.path.join(scaffold_dest, f"{safe_name}_model_config")
    with open(config_path, "w") as f:
        f.write(config_content)


def _yaml_quote(value: str) -> str:
    return json.dumps(str(value))


def _ensure_project_yml_odr(content: str, *, model_dir: str, model_name: str) -> str:
    if "KnownAssetTags:" not in content:
        known_tags = "attributes:\n  KnownAssetTags:\n    - model\n"
        if re.search(r"^attributes:\n", content, flags=re.MULTILINE):
            content = re.sub(r"^attributes:\n", known_tags, content, count=1, flags=re.MULTILINE)
        else:
            content = re.sub(r"^(name: .+\n)", r"\1" + known_tags, content, count=1, flags=re.MULTILINE)

    if model_dir not in content or "resourceTags:" not in content:
        if "\n    sources:\n" not in content:
            content = re.sub(
                r"(\n    platform: [^\n]+\n)",
                r"\1    sources:\n",
                content,
                count=1,
            )
        odr_source = (
            f"      - path: {_yaml_quote(model_dir)}\n"
            f"        name: {_yaml_quote(model_name)}\n"
            "        buildPhase: resources\n"
            "        resourceTags:\n"
            "          - model\n"
        )
        for marker in ("\n    settings:\n", "\n    dependencies:\n", "\n    entitlements:\n"):
            if marker in content:
                return content.replace(marker, "\n" + odr_source + marker, 1)
        raise ScaffoldExportError("project.yml target section not found for ODR source insertion")

    return content


def _customize_project_yml(
    scaffold_dest: str,
    safe_name: str,
    app_name: str,
    model_name: str,
    model_dir: str,
    bundle_id: str | None = None,
    team_id: str | None = None,
):
    """Patch project.yml with new app name, paths, and config file reference."""
    yml_path = os.path.join(scaffold_dest, "project.yml")
    _assert_file(yml_path, "project.yml")

    with open(yml_path) as f:
        content = f.read()

    safe_id = re.sub(r"[^a-z0-9]", "", app_name.lower())
    if not safe_id:
        safe_id = "myapp"

    # Replace template app references with the new app name.
    content = content.replace("EdgeScaffolding", safe_name)
    content = content.replace("EdgeScaffold", safe_name)

    # Default bundle ID uses a generic prefix to avoid conflicts with com.atomgradient.
    # Users can override via the bundle_id parameter or edit project.yml after export.
    content = re.sub(
        r"PRODUCT_BUNDLE_IDENTIFIER: .*",
        f"PRODUCT_BUNDLE_IDENTIFIER: com.edgestudio.{safe_id}",
        content,
    )
    content = re.sub(
        r"INFOPLIST_KEY_CFBundleDisplayName: .*",
        f"INFOPLIST_KEY_CFBundleDisplayName: {app_name}",
        content,
    )
    content = re.sub(
        r"DEVELOPMENT_TEAM: .*",
        'DEVELOPMENT_TEAM: ""',
        content,
    )

    # Override bundle ID and team ID if provided
    if bundle_id:
        content = re.sub(r'PRODUCT_BUNDLE_IDENTIFIER: .*', f'PRODUCT_BUNDLE_IDENTIFIER: {bundle_id}', content)
    if team_id:
        content = re.sub(r'DEVELOPMENT_TEAM: .*', f'DEVELOPMENT_TEAM: {team_id}', content)

    content = _ensure_project_yml_odr(content, model_dir=model_dir, model_name=model_name)

    # Update Build Phase script config file reference (lowercase variant)
    if "postBuildScripts" in content:
        content = content.replace("edgescaffolding_model_config", f"{safe_name}_model_config")
    else:
        # Old template without Build Phase — inject it
        post_build_block = _build_post_build_script(safe_name)
        content = content.replace(
            "    dependencies:\n",
            post_build_block + "    dependencies:\n",
        )

    # Ensure EdgeKit uses the template's version contract.
    min_ver = _read_min_runtime_version(scaffold_dest)
    content = re.sub(
        r'  (?:EdgeRuntime|EdgeKit|edge-kit):\n    (?:path: [^\n]+|url: [^\n]+(?:\n    (?:from|exactVersion): [^\n]+)?)',
        f'  EdgeKit:\n    url: https://github.com/AtomGradient/edge-kit.git\n    exactVersion: {min_ver}',
        content,
    )
    # Remove legacy mlx-swift-lm references; EdgeKit owns runtime dependencies.
    content = re.sub(
        r'  mlx-swift-lm:\n    (?:path|url): [^\n]+(?:\n    (?:from|exactVersion): [^\n]+)?\n',
        '',
        content,
    )

    # Ensure required EdgeKit products are linked.
    content = content.replace("      - package: EdgeRuntime\n", "      - package: EdgeKit\n")
    content = content.replace("      - package: edge-kit\n", "      - package: EdgeKit\n")
    if '      - package: EdgeKit\n        product: EdgeInference' not in content:
        content = content.replace(
            '    dependencies:\n',
            '    dependencies:\n'
            '      - package: EdgeKit\n        product: EdgeInference\n',
        )
    for product in ("EdgeModelKit", "EdgeMesh", "EdgeData", "EdgeSession"):
        dep = f'      - package: EdgeKit\n        product: {product}'
        if dep not in content:
            content = content.replace(
                '      - package: EdgeKit\n        product: EdgeInference\n',
                '      - package: EdgeKit\n        product: EdgeInference\n'
                f'{dep}\n',
                1,
            )

    if "  swift-async-algorithms:\n" not in content:
        content = content.replace(
            "targets:\n",
            "  swift-async-algorithms:\n"
            "    url: https://github.com/apple/swift-async-algorithms.git\n"
            "    from: 1.1.3\n"
            "  swift-markdown-ui:\n"
            "    url: https://github.com/gonzalezreal/swift-markdown-ui\n"
            "    from: 2.4.1\n\n"
            "targets:\n",
            1,
        )
    elif "  swift-markdown-ui:\n" not in content:
        content = content.replace(
            "targets:\n",
            "  swift-markdown-ui:\n"
            "    url: https://github.com/gonzalezreal/swift-markdown-ui\n"
            "    from: 2.4.1\n\n"
            "targets:\n",
            1,
        )

    async_dep = '      - package: swift-async-algorithms\n        product: AsyncAlgorithms'
    markdown_dep = '      - package: swift-markdown-ui\n        product: MarkdownUI'
    if async_dep not in content:
        content = content.replace('    entitlements:\n', f'{async_dep}\n    entitlements:\n', 1)
    if markdown_dep not in content:
        content = content.replace('    entitlements:\n', f'{markdown_dep}\n    entitlements:\n', 1)

    with open(yml_path, "w") as f:
        f.write(content)


def _build_post_build_script(safe_name: str) -> str:
    """Generate postBuildScripts YAML block for old templates without built-in Build Phase."""
    indent = "          "
    lines = [
        '#!/bin/bash',
        '# Copy local model to app bundle (Debug builds only)',
        '',
        'if [ "${CONFIGURATION}" != "Debug" ]; then',
        '    echo "Skipping model copy (Release build)"',
        '    exit 0',
        'fi',
        '',
        'CONFIG_FILE="${SRCROOT}/' + safe_name + '_model_config"',
        'if [ -f "${CONFIG_FILE}" ]; then',
        '    source "${CONFIG_FILE}"',
        'fi',
        '',
        'MODEL_COPY="${MODEL_COPY:-false}"',
        'MODEL_NAME="${MODEL_NAME:-}"',
        'MODELS_SOURCE_DIR="${MODELS_SOURCE_DIR:-}"',
        '',
        'if [ "${MODEL_COPY}" = "0" ] || [ "${MODEL_COPY}" = "false" ]; then',
        '    exit 0',
        'fi',
        '',
        'if [ -z "${MODEL_NAME}" ] || [ -z "${MODELS_SOURCE_DIR}" ]; then',
        '    exit 0',
        'fi',
        '',
        'MODEL_SRC="${MODELS_SOURCE_DIR}/${MODEL_NAME}"',
        'if [ ! -d "${MODEL_SRC}" ]; then',
        '    echo "Model not found: ${MODEL_SRC}"',
        '    exit 1',
        'fi',
        '',
        'DEST="${BUILT_PRODUCTS_DIR}/${EXECUTABLE_FOLDER_PATH}/${MODEL_NAME}"',
        'if [ -d "${DEST}" ]; then',
        '    SRC_SIZE=$(du -s "${MODEL_SRC}" | cut -f1)',
        '    DST_SIZE=$(du -s "${DEST}" | cut -f1)',
        '    if [ "${SRC_SIZE}" = "${DST_SIZE}" ]; then',
        '        echo "Model already in bundle (unchanged)"',
        '        exit 0',
        '    fi',
        'fi',
        '',
        'echo "Copying model: ${MODEL_NAME}"',
        'rm -rf "${DEST}"',
        'cp -r "${MODEL_SRC}" "${DEST}"',
    ]
    script_body = "\n".join(indent + line for line in lines) + "\n"
    return (
        f"    postBuildScripts:\n"
        f"      - script: |\n"
        f"{script_body}"
        f"        name: Copy Local Model\n"
        f"        basedOnDependencyAnalysis: false\n"
    )


def _rename_entry_struct(scaffold_dest: str, safe_name: str):
    """Rename template app entry struct to struct {safe_name}App."""
    entry_path = os.path.join(scaffold_dest, safe_name, "App", f"{safe_name}App.swift")
    if not os.path.isfile(entry_path):
        return

    with open(entry_path) as f:
        content = f.read()

    content = content.replace("struct EdgeScaffoldApp:", f"struct {safe_name}App:")
    content = content.replace("EdgeScaffoldApp", f"{safe_name}App")
    content = content.replace("struct EdgeScaffoldingApp:", f"struct {safe_name}App:")
    content = content.replace("EdgeScaffoldingApp", f"{safe_name}App")

    with open(entry_path, "w") as f:
        f.write(content)


def _run_xcodegen(scaffold_dest: str, safe_name: str):
    """Run xcodegen to generate .xcodeproj."""
    xcodegen = shutil.which("xcodegen")
    if not xcodegen and os.path.isfile("/opt/homebrew/bin/xcodegen"):
        xcodegen = "/opt/homebrew/bin/xcodegen"
    if not xcodegen:
        raise ScaffoldExportError(
            "xcodegen not found. Install it: brew install xcodegen"
        )

    try:
        r = subprocess.run(
            [xcodegen, "generate"],
            cwd=scaffold_dest,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            raise ScaffoldExportError(
                f"xcodegen failed (exit {r.returncode}): {r.stderr.strip() or r.stdout.strip()}"
            )
    except FileNotFoundError:
        raise ScaffoldExportError("xcodegen not found. Install it: brew install xcodegen")


# ─────────────────────────────────────────────
# ODR patching — using pbxproj library
# ─────────────────────────────────────────────

def _patch_xcodeproj_odr(
    scaffold_dest: str,
    safe_name: str,
    model_dir: str,
    model_name: str,
):
    """Patch generated .xcodeproj with ODR resource tags using pbxproj library.

    Adds:
    1. PBXFileReference  — folder reference pointing to model on disk
    2. PBXBuildFile       — with ASSET_TAGS = ("model")
    3. PBXResourcesBuildPhase — includes build file in Resources (auto-created)
    4. Root PBXGroup      — adds file ref to project navigator
    5. KnownAssetTags     — declares "model" tag in project attributes
    """
    from pbxproj import XcodeProject, PBXGenericObject
    from pbxproj.pbxextensions.ProjectFiles import FileOptions

    pbxproj_path = os.path.join(
        scaffold_dest, f"{safe_name}.xcodeproj", "project.pbxproj"
    )
    _assert_file(pbxproj_path, "ODR patching — project.pbxproj")

    project = XcodeProject.load(pbxproj_path)

    # Already patched?
    if project.get_files_by_name(model_name):
        project.save()
        return

    odr_tag = "model"

    # ── 1-4. Add file reference + build file + resources phase + group ──
    build_files = project.add_file(
        model_dir,
        parent=None,
        tree="<absolute>",
        file_options=FileOptions(create_build_files=True),
    )

    if not build_files:
        raise ScaffoldExportError(
            f"pbxproj failed to add model reference: {model_name}"
        )

    # Set ASSET_TAGS on each build file
    for bf in build_files:
        bf["settings"] = PBXGenericObject().parse({"ASSET_TAGS": [odr_tag]})

    # ── 5. KnownAssetTags in project attributes ─
    root_obj = project.objects[project.rootObject]
    attrs = root_obj.get("attributes", {})
    existing_tags = attrs.get("KnownAssetTags", None)
    if isinstance(existing_tags, list):
        known_tags = list(existing_tags)
    elif existing_tags:
        known_tags = [str(existing_tags)]
    else:
        known_tags = []
    if odr_tag not in known_tags:
        known_tags.append(odr_tag)

    if isinstance(attrs, PBXGenericObject):
        attrs.KnownAssetTags = known_tags
    else:
        attrs = PBXGenericObject().parse({**attrs, "KnownAssetTags": known_tags})
    root_obj["attributes"] = attrs

    project.save()


def _generate_readme(
    root_dir: str,
    safe_name: str,
    app_name: str,
    model_name: str,
    model_dir: str,
):
    """Generate an instance-specific README for the exported app."""
    readme = f"""# {app_name}

Generated by **Edge Studio** from the public
[edge-scaffold](https://github.com/AtomGradient/edge-scaffold) template.

This README is for this exported app instance. The template repository README
explains how the scaffold itself is maintained; this file explains how to run
and continue this app.

## Quick Start

1. Open `{safe_name}.xcodeproj` in Xcode 16 or newer.
2. Wait for Swift Package Manager to resolve dependencies.
3. Select your development team in Signing & Capabilities.
4. Build on a physical iOS 18+ device. Simulator is not supported for MLX
   runtime validation.

If `{safe_name}.xcodeproj` is missing, install XcodeGen and regenerate it:

```bash
brew install xcodegen
cd {safe_name}
xcodegen generate
```

For a command-line build check without copying model weights:

```bash
xcodebuild -project {safe_name}.xcodeproj \\
  -scheme {safe_name} \\
  -configuration Release \\
  -destination 'generic/platform=iOS' \\
  SKIP_MODEL_COPY=1 \\
  build
```

## This Export

- App display name: `{app_name}`
- Xcode target: `{safe_name}`
- Model: `{model_name}`
- Model source path on this Mac: `{model_dir}`
- ODR tag: `model`

The model weights are not included in this ZIP. The project references the
local model path above, and the build phase reads `{safe_name}_model_config`
when model copying is enabled.

## Project Structure

```text
{safe_name}/
+-- {safe_name}.xcodeproj/
+-- project.yml
+-- {safe_name}_model_config
+-- README.md
+-- Resources/
|   +-- SampleData/
|   +-- RPP/
+-- {safe_name}/
    +-- App/
    |   +-- ScaffoldConfig.swift
    +-- AI/
    |   +-- AIManager.swift
    |   +-- EdgeDataBootstrap.swift
    |   +-- PersonalizationManager.swift
    +-- Chat/
    |   +-- DemoChatView*.swift
    +-- Settings/
    +-- Business/
```

## Key Files

- `{safe_name}/App/ScaffoldConfig.swift`: app name, bundle model name, system
  prompt, model category, Neural Imprint and RPP runtime settings.
- `{safe_name}_model_config`: local model path and model copy settings used by
  the Xcode build phase.
- `{safe_name}/AI/AIManager.swift`: model loading and streaming generation.
- `{safe_name}/Chat/DemoChatView*.swift`: text, vision, speech, and audio demo
  surfaces.
- `project.yml`: XcodeGen project spec and Swift Package Manager dependencies.

## Model Loading

The exported app uses the scaffold's four-tier loading strategy:

1. Local app cache.
2. Model copied into the app bundle by the build phase.
3. On-Demand Resources using the `model` tag.
4. Remote fallback when the app is configured to allow it.

For local development, edit `{safe_name}_model_config`:

```bash
MODEL_NAME={model_name}
MODELS_SOURCE_DIR={os.path.dirname(model_dir.rstrip(os.sep)) or model_dir}
MODEL_COPY="true"
```

Set `SKIP_MODEL_COPY=1` for CI or release build checks when the model is not on
the build machine.

## Dependencies

Swift Package Manager resolves the runtime packages from public GitHub URLs:

- `edge-kit` from `https://github.com/AtomGradient/edge-kit.git`
- `edge-engine` through the EdgeKit package graph
- `edge-halo-binary` from `https://github.com/AtomGradient/edge-halo-binary`

EdgeHalo source is private. This app consumes the EdgeHalo binary package.

## Neural Imprint And RPP

This app is wired for Neural Imprint restore hooks and RPP-aware demo flows. If
`Resources/RPP` is empty, the app keeps RPP-specific behavior fail-closed. If
Edge Studio exported a model-matched A-library, its manifest and assets are
under `Resources/RPP`.

## Continue Development

Good first edits:

- Change product behavior in `{safe_name}/Business/HomeView.swift`.
- Change chat behavior in `{safe_name}/Chat/DemoChatView*.swift`.
- Change app configuration in `{safe_name}/App/ScaffoldConfig.swift`.
- Change model loading in `{safe_name}/AI/AIManager.swift`.
- Add app-specific facts, schemas, or sample data under `Resources/SampleData`.

When using an AI coding agent on this app, ask it to read these files first:

1. `README.md`
2. `project.yml`
3. `{safe_name}/App/ScaffoldConfig.swift`
4. `{safe_name}/AI/AIManager.swift`
5. `{safe_name}/Chat/DemoChatView+LLM.swift`

## Troubleshooting

SPM cache looks stale:

```bash
rm -rf .build
rm -rf ~/Library/Developer/Xcode/DerivedData
xcodebuild -resolvePackageDependencies -project {safe_name}.xcodeproj
```

Signing fails:

- Select a team in Xcode.
- Confirm the bundle identifier is unique.
- Build on a physical device.

Model is not copied:

- Check `{safe_name}_model_config`.
- Confirm `MODEL_COPY="true"` for Debug builds.
- Confirm `MODELS_SOURCE_DIR/MODEL_NAME` exists.
- For CI, set `SKIP_MODEL_COPY=1`.
"""
    with open(os.path.join(root_dir, "README.md"), "w") as f:
        f.write(readme)


def _validate_zip(zip_path: str, safe_name: str):
    """Validate ZIP integrity and completeness after creation.

    Checks:
    1. ZIP CRC integrity (detects corrupt entries)
    2. Minimum file count (valid scaffold has 15+ source files)
    3. Critical files exist (project.yml, App entry, ScaffoldConfig, pbxproj, model_config)
    """
    _MIN_FILE_COUNT = 10

    # Critical files that MUST be in the ZIP (paths relative to archive root)
    required = [
        f"{safe_name}/project.yml",
        f"{safe_name}/{safe_name}/App/{safe_name}App.swift",
        f"{safe_name}/{safe_name}/App/ScaffoldConfig.swift",
        f"{safe_name}/{safe_name}.xcodeproj/project.pbxproj",
        f"{safe_name}/{safe_name}_model_config",
        f"{safe_name}/README.md",
    ]

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # 1. CRC integrity
            bad = zf.testzip()
            if bad:
                raise ScaffoldExportError(f"[ZIP validation] Corrupt entry: {bad}")

            names = set(zf.namelist())
            file_count = sum(1 for n in names if not n.endswith("/"))

            # 2. Minimum file count
            if file_count < _MIN_FILE_COUNT:
                raise ScaffoldExportError(
                    f"[ZIP validation] Only {file_count} file{'s' if file_count != 1 else ''}, "
                    f"expected ≥{_MIN_FILE_COUNT}. Export incomplete — please retry."
                )

            # 3. Critical files
            missing = [p for p in required if p not in names]
            if missing:
                short_names = [os.path.basename(p) for p in missing]
                raise ScaffoldExportError(
                    f"[ZIP validation] Missing critical files: {', '.join(short_names)} "
                    f"(ZIP has {file_count} files). Please retry."
                )

    except zipfile.BadZipFile:
        raise ScaffoldExportError("[ZIP validation] Generated ZIP is corrupt (BadZipFile)")
