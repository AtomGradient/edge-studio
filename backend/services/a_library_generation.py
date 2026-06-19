# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-side A-library generation job wrapper."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.services.a_library_inspector import inspect_a_library_directory
from backend.services.a_library_direction_sets import (
    sanitize_direction_set_id,
    validate_direction_yaml_file,
    validate_direction_source_type,
)
from backend.services.a_library_registry import (
    generated_a_library_root,
    manifest_item_for_generated_library,
    model_metadata_from_dir,
    write_manifest,
)


ProgressCallback = Callable[[str, float], None] | None

DEFAULT_DIRECTIONS_YAML = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "rpp"
    / "data"
    / "caa_pairs_seed"
    / "directions_a.yaml"
)
DEFAULT_SWEEP_LAYERS = [11, 19, 23, 27, 6, 14, 22, 30]


class ALibraryGenerationError(ValueError):
    pass


def generate_a_library(
    *,
    model_path: str | Path,
    yaml_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    layers: list[int] | None = None,
    sweep: bool = True,
    pooling: str = "last_real",
    direction_set_id: str | None = None,
    source_type: str = "manual",
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    """Generate an RPP A-library and return an inspection-ready result."""

    def progress(message: str, percent: float) -> None:
        if progress_callback is not None:
            progress_callback(message, percent)

    model_root = Path(model_path).expanduser().resolve()
    if not model_root.is_dir():
        raise ALibraryGenerationError(f"Model directory not found: {model_root}")

    metadata = model_metadata_from_dir(model_root)
    model_family = metadata.get("model_family")
    hidden_size = metadata.get("hidden_size")
    layer_count = metadata.get("layer_count")
    if not isinstance(model_family, str) or not model_family.startswith(("qwen3.5", "qwen3.6")):
        raise ALibraryGenerationError(
            "A-library generation currently supports Qwen3.5/Qwen3.6 models only."
        )
    if metadata.get("is_moe"):
        raise ALibraryGenerationError(
            "MoE A-library generation is tracked as a follow-up; dense Qwen3.5/Qwen3.6 is supported first."
        )
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ALibraryGenerationError("Model config must expose hidden_size for A-library validation.")
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ALibraryGenerationError("Model config must expose num_hidden_layers for A-library validation.")

    selected_layers = _normalize_layers(layers, sweep=sweep)
    directions_yaml = Path(yaml_path or DEFAULT_DIRECTIONS_YAML).expanduser().resolve()
    if not directions_yaml.is_file():
        raise ALibraryGenerationError(f"Direction-set YAML not found: {directions_yaml}")
    try:
        direction_report = validate_direction_yaml_file(directions_yaml, direction_set_id=direction_set_id)
    except (OSError, UnicodeDecodeError) as exc:
        raise ALibraryGenerationError(f"Unable to read direction-set YAML: {directions_yaml}") from exc
    if not direction_report.get("ok"):
        raise ALibraryGenerationError(
            "Invalid direction-set YAML: "
            + ", ".join(error.get("code", "unknown") for error in direction_report.get("errors", []))
        )
    resolved_direction_set_id = sanitize_direction_set_id(direction_report.get("direction_set_id"))
    yaml_sha256 = str(direction_report.get("yaml_sha256") or "")
    source_schema_version = str(direction_report.get("source_schema_version") or "")
    resolved_source_type = validate_direction_source_type(source_type)
    out_dir = _default_output_dir(metadata) if output_dir is None else Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    progress("Loading model and extracting A-library directions...", 0.05)
    from docs.research.rpp.src.directions.build_a import build_library_a

    reports = build_library_a(
        yaml_path=directions_yaml,
        model_path=str(model_root),
        layer_indices=selected_layers,
        pooling=pooling,
        output_dir=out_dir,
    )

    progress("Converting A-library directions to safetensors...", 0.78)
    from docs.research.rpp.src.training.export_safetensors import npz_a_to_safetensors

    libraries: list[dict[str, Any]] = []
    for layer_idx, report in sorted(reports.items()):
        npz_path = out_dir / f"directions_a_layer_{layer_idx}.npz"
        st_path = out_dir / f"directions_a_layer_{layer_idx}.safetensors"
        if npz_path.is_file():
            npz_a_to_safetensors(npz_path, st_path)
        enriched = _enrich_report(
            report=report,
            model_family=model_family,
            hidden_size=hidden_size,
            layer_count=layer_count,
            layer_idx=layer_idx,
            npz_path=npz_path,
            direction_set_id=resolved_direction_set_id,
            yaml_sha256=yaml_sha256,
            source_type=resolved_source_type,
            source_schema_version=source_schema_version,
        )
        report_path = out_dir / f"directions_a_layer_{layer_idx}_report.json"
        report_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        libraries.append(
            manifest_item_for_generated_library(
                output_dir=out_dir,
                model_family=model_family,
                hidden_size=hidden_size,
                layer_count=layer_count,
                target_layer=layer_idx,
                n_directions=enriched["n_directions"],
                artifact_name=st_path.name,
                health_report_name=report_path.name,
                health_verdict=enriched["verdict"],
                pooling=pooling,
                direction_set_id=resolved_direction_set_id,
                yaml_sha256=yaml_sha256,
                source_type=resolved_source_type,
                source_schema_version=source_schema_version,
            )
        )

    progress("Writing A-library manifest...", 0.92)
    selected = _select_best_library(libraries, reports)
    manifest_path = write_manifest(
        out_dir,
        libraries,
        default_library_id=selected["library_id"] if selected else "",
    )

    inspection = inspect_a_library_directory(out_dir)
    progress("A-library generation complete.", 1.0)
    return {
        "ok": True,
        "schema_version": "edgestudio.a_library_generation.v1",
        "status": "complete",
        "output_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "model": metadata,
        "layers": selected_layers,
        "direction_set_id": resolved_direction_set_id,
        "yaml_sha256": yaml_sha256,
        "source_type": resolved_source_type,
        "source_schema_version": source_schema_version,
        "selected_library_id": selected["library_id"] if selected else None,
        "inspection": inspection,
    }


def _normalize_layers(layers: list[int] | None, *, sweep: bool) -> list[int]:
    values = [int(layer) for layer in (layers or []) if int(layer) >= 0]
    if values:
        return sorted(dict.fromkeys(values))
    if sweep:
        return DEFAULT_SWEEP_LAYERS
    return [23]


def _default_output_dir(metadata: dict[str, Any]) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = str(metadata.get("model_name") or "model").replace("/", "_")
    return generated_a_library_root() / f"{model_name}_{stamp}"


def _enrich_report(
    *,
    report: dict[str, Any],
    model_family: str,
    hidden_size: int,
    layer_count: int,
    layer_idx: int,
    npz_path: Path,
    direction_set_id: str,
    yaml_sha256: str,
    source_type: str,
    source_schema_version: str,
) -> dict[str, Any]:
    n_directions = int(report.get("n_directions") or _npz_direction_count(npz_path) or 0)
    orthogonality = report.get("health_check_1_orthogonality") or {}
    signal = report.get("health_check_2_signal_strength") or {}
    verdict = (
        "pass"
        if orthogonality.get("max_pass") is True
        and orthogonality.get("mean_pass") is True
        and signal.get("n_pass") == signal.get("n_total")
        else "fail"
    )
    enriched = dict(report)
    enriched.update({
        "library_kind": "rpp_user_profile",
        "model_family": model_family,
        "hidden_size": hidden_size,
        "layer_count": layer_count,
        "layer_idx": layer_idx,
        "direction_set_id": direction_set_id,
        "yaml_sha256": yaml_sha256,
        "source_type": source_type,
        "source_schema_version": source_schema_version,
        "n_directions": n_directions,
        "verdict": verdict,
    })
    return enriched


def _npz_direction_count(path: Path) -> int | None:
    try:
        with np.load(path) as data:
            return len(data.keys())
    except Exception:
        return None


def _select_best_library(
    libraries: list[dict[str, Any]],
    reports: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    passing = [item for item in libraries if item.get("health_verdict") == "pass"]
    candidates = passing or libraries
    if not candidates:
        return None

    def score(item: dict[str, Any]) -> tuple[bool, float, float]:
        layer_idx = int(item.get("target_layer") or -1)
        report = reports.get(layer_idx) or {}
        o = report.get("health_check_1_orthogonality") or {}
        s = report.get("health_check_2_signal_strength") or {}
        return (
            item.get("health_verdict") == "pass",
            float(s.get("median_signal_strength") or 0.0),
            -float(o.get("max_abs_cos_sim") or 99.0),
        )

    return max(candidates, key=score)
