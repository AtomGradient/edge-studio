# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Read-only inspection helpers for RPP A-library artifacts.

The A-library inspector is a host-side management surface. It reads layer
health reports and tensor headers so EdgeStudio can show which model/layer a
library belongs to without loading tensor payloads or running generation.
"""

from __future__ import annotations

import ast
import json
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


A_LIBRARY_INSPECTION_SCHEMA_VERSION = "edgestudio.a_library_inspection.v0"
MAX_JSON_REPORT_BYTES = 4 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
MAX_REPORTS = 128
MAX_ARTIFACTS = 128


@dataclass
class ALibraryInspectionError(ValueError):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": False,
            "details": self.details or {},
        }


def inspect_a_library_directory(library_dir: Path) -> dict[str, Any]:
    """Inspect an A-library directory using reports and artifact headers only."""

    root = library_dir.expanduser().resolve()
    if not root.exists():
        raise ALibraryInspectionError(
            "a_library_path_not_found",
            "A-library path does not exist.",
            {"path": str(root)},
        )
    if not root.is_dir():
        raise ALibraryInspectionError(
            "a_library_path_not_directory",
            "A-library path must be a directory.",
            {"path": str(root)},
        )

    warnings: list[str] = []
    report_paths = _limited_paths(root.rglob("directions_a_layer_*_report.json"), MAX_REPORTS)
    sweep_paths = _limited_paths(root.rglob("sweep_summary.json"), MAX_REPORTS)
    artifact_paths = _limited_paths(
        (
            path
            for path in root.rglob("directions_a_layer_*")
            if path.suffix in {".safetensors", ".npz"} and path.is_file()
        ),
        MAX_ARTIFACTS,
    )

    reports = [_read_layer_report(path, root, warnings) for path in report_paths]
    reports = [report for report in reports if report is not None]
    sweep_summaries = [_read_sweep_summary(path, root, warnings) for path in sweep_paths]
    sweep_summaries = [summary for summary in sweep_summaries if summary is not None]
    artifacts = [_artifact_info(path, root, warnings) for path in artifact_paths]
    artifacts = [artifact for artifact in artifacts if artifact is not None]

    _attach_artifacts_to_reports(reports, artifacts)
    summary = _summary(root=root, reports=reports, artifacts=artifacts, sweep_summaries=sweep_summaries)
    manifest = _manifest_summary(summary=summary, reports=reports)

    if not reports:
        warnings.append("missing_a_library_health_reports")
    if not artifacts:
        warnings.append("missing_a_library_artifacts")
    if summary.get("health_status") != "pass":
        warnings.append("a_library_health_not_fully_passing")

    return {
        "ok": True,
        "schema_version": A_LIBRARY_INSPECTION_SCHEMA_VERSION,
        "status": "found",
        "library_path": str(root),
        "summary": summary,
        "manifest": manifest,
        "health_reports": sorted(reports, key=lambda item: (item.get("layer_idx") is None, item.get("layer_idx") or 0)),
        "sweep_summaries": sweep_summaries,
        "artifacts": sorted(artifacts, key=lambda item: (item.get("layer_idx") is None, item.get("layer_idx") or 0, item.get("name") or "")),
        "warnings": sorted(set(warnings)),
        "audit": {
            "json_report_count": len(reports),
            "sweep_summary_count": len(sweep_summaries),
            "artifact_count": len(artifacts),
            "safetensors_payload_loaded": False,
            "npz_payload_loaded": False,
        },
    }


def _limited_paths(paths: Any, limit: int) -> list[Path]:
    items = sorted((path for path in paths if isinstance(path, Path) and path.is_file()), key=lambda p: str(p))
    return items[:limit]


def _read_layer_report(path: Path, root: Path, warnings: list[str]) -> dict[str, Any] | None:
    payload = _read_json(path, warnings)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        warnings.append(f"invalid_report_json:{_relative(path, root)}")
        return None

    orthogonality = _object_or_empty(payload.get("health_check_1_orthogonality"))
    signal = _object_or_empty(payload.get("health_check_2_signal_strength"))
    layer_idx = _coerce_int(payload.get("layer_idx"))
    if layer_idx is None:
        layer_idx = _layer_from_name(path.name)
    model_path = _coerce_str(payload.get("model_path"))
    max_abs_cos_sim = _coerce_float(orthogonality.get("max_abs_cos_sim"))
    mean_abs_cos_sim = _coerce_float(orthogonality.get("mean_abs_cos_sim"))
    max_threshold = _coerce_float(orthogonality.get("threshold_max_lt")) or 0.4
    mean_threshold = _coerce_float(orthogonality.get("threshold_mean_lt")) or 0.15
    max_pass = _coerce_bool(orthogonality.get("max_pass"))
    if max_pass is None and max_abs_cos_sim is not None:
        max_pass = max_abs_cos_sim < max_threshold
    mean_pass = _coerce_bool(orthogonality.get("mean_pass"))
    if mean_pass is None and mean_abs_cos_sim is not None:
        mean_pass = mean_abs_cos_sim < mean_threshold

    n_pass = _coerce_int(signal.get("n_pass"))
    n_total = _coerce_int(signal.get("n_total")) or _coerce_int(payload.get("n_directions"))
    signal_pass = n_pass is not None and n_total is not None and n_pass >= n_total
    verdict = "pass" if max_pass is True and mean_pass is True and signal_pass else "fail"

    return {
        "layer_idx": layer_idx,
        "layer_type": _coerce_str(payload.get("layer_type")),
        "library_kind": "rpp_user_profile",
        "model_family": _infer_model_family(model_path or str(path)),
        "model_path": model_path,
        "pooling": _coerce_str(payload.get("pooling")),
        "direction_set_id": _coerce_str(payload.get("direction_set_id")) or _direction_set_id(payload.get("yaml_path")),
        "yaml_sha256": _coerce_str(payload.get("yaml_sha256")),
        "source_type": _coerce_str(payload.get("source_type")),
        "source_schema_version": _coerce_str(payload.get("source_schema_version")),
        "n_directions": _coerce_int(payload.get("n_directions")),
        "n_sentences_total": _coerce_int(payload.get("n_sentences_total")),
        "extract_seconds": _coerce_float(payload.get("extract_seconds")),
        "ms_per_sentence": _coerce_float(payload.get("ms_per_sentence")),
        "max_abs_cos_sim": max_abs_cos_sim,
        "mean_abs_cos_sim": mean_abs_cos_sim,
        "threshold_max_lt": max_threshold,
        "threshold_mean_lt": mean_threshold,
        "max_pass": max_pass,
        "mean_pass": mean_pass,
        "worst_pair": orthogonality.get("worst_pair") if isinstance(orthogonality.get("worst_pair"), list) else None,
        "min_signal_strength": _coerce_float(signal.get("min_signal_strength")),
        "median_signal_strength": _coerce_float(signal.get("median_signal_strength")),
        "max_signal_strength": _coerce_float(signal.get("max_signal_strength")),
        "signal_threshold": _coerce_float(signal.get("threshold")),
        "n_pass": n_pass,
        "n_total": n_total,
        "signal_pass": signal_pass,
        "verdict": verdict,
        "report_path": _relative(path, root),
        "artifact_names": [],
    }


def _read_sweep_summary(path: Path, root: Path, warnings: list[str]) -> dict[str, Any] | None:
    payload = _read_json(path, warnings)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        warnings.append(f"invalid_sweep_summary_json:{_relative(path, root)}")
        return None
    per_layer = payload.get("per_layer")
    per_layer_rows: list[dict[str, Any]] = []
    if isinstance(per_layer, dict):
        for key, raw in per_layer.items():
            if not isinstance(raw, dict):
                continue
            row = _summary_layer_row(key, raw)
            if row is not None:
                per_layer_rows.append(row)

    return {
        "path": _relative(path, root),
        "model_path": _coerce_str(payload.get("model_path")),
        "model_family": _infer_model_family(_coerce_str(payload.get("model_path")) or str(path)),
        "pooling": _coerce_str(payload.get("pooling")),
        "layers_swept": [_coerce_int(item) for item in _list_or_empty(payload.get("layers_swept")) if _coerce_int(item) is not None],
        "per_layer": sorted(per_layer_rows, key=lambda item: item.get("layer_idx") or 0),
    }


def _summary_layer_row(key: Any, raw: dict[str, Any]) -> dict[str, Any] | None:
    layer_idx = _coerce_int(key)
    if layer_idx is None:
        return None
    max_abs = _coerce_float(raw.get("max_abs_cos_sim"))
    mean_abs = _coerce_float(raw.get("mean_abs_cos_sim"))
    n_pass = _coerce_int(raw.get("n_pass"))
    n_total = _coerce_int(raw.get("n_total"))
    max_pass = max_abs is not None and max_abs < 0.4
    mean_pass = mean_abs is not None and mean_abs < 0.15
    signal_pass = n_pass is not None and n_total is not None and n_pass >= n_total
    return {
        "layer_idx": layer_idx,
        "layer_type": _coerce_str(raw.get("layer_type")),
        "max_abs_cos_sim": max_abs,
        "mean_abs_cos_sim": mean_abs,
        "min_signal_strength": _coerce_float(raw.get("min_signal_strength")),
        "median_signal_strength": _coerce_float(raw.get("median_signal_strength")),
        "n_pass": n_pass,
        "n_total": n_total,
        "verdict": "pass" if max_pass and mean_pass and signal_pass else "fail",
    }


def _artifact_info(path: Path, root: Path, warnings: list[str]) -> dict[str, Any] | None:
    layer_idx = _layer_from_name(path.name)
    info: dict[str, Any] = {
        "name": path.name,
        "path": _relative(path, root),
        "kind": "safetensors" if path.suffix == ".safetensors" else "npz",
        "layer_idx": layer_idx,
        "size_bytes": _file_size(path),
        "model_family": _infer_model_family(str(path)),
        "hidden_size": None,
        "direction_count": None,
        "dtype": None,
    }
    if path.suffix == ".safetensors":
        info.update(_read_safetensors_summary(path, root, warnings))
    elif path.suffix == ".npz":
        info.update(_read_npz_summary(path, root, warnings))
    if not info.get("model_family") and info.get("hidden_size"):
        info["model_family"] = _infer_model_family_from_hidden_size(info["hidden_size"])
    return info


def _read_safetensors_summary(path: Path, root: Path, warnings: list[str]) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            first = handle.read(8)
            if len(first) != 8:
                warnings.append(f"invalid_safetensors_too_small:{_relative(path, root)}")
                return {}
            header_len = struct.unpack("<Q", first)[0]
            if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
                warnings.append(f"invalid_safetensors_header_length:{_relative(path, root)}")
                return {}
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                warnings.append(f"truncated_safetensors_header:{_relative(path, root)}")
                return {}
            header = json.loads(raw_header.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"invalid_safetensors_header:{_relative(path, root)}:{exc.__class__.__name__}")
        return {}
    if not isinstance(header, dict):
        warnings.append(f"invalid_safetensors_header_object:{_relative(path, root)}")
        return {}
    tensors = [
        (name, meta)
        for name, meta in header.items()
        if name != "__metadata__" and isinstance(meta, dict)
    ]
    shapes = [meta.get("shape") for _, meta in tensors if isinstance(meta.get("shape"), list)]
    hidden_size = _common_vector_width(shapes)
    dtype = _common_value([_coerce_str(meta.get("dtype")) for _, meta in tensors])
    return {
        "header_size_bytes": int(header_len),
        "tensor_count": len(tensors),
        "direction_count": len(tensors),
        "hidden_size": hidden_size,
        "dtype": dtype,
        "metadata": _object_or_empty(header.get("__metadata__")),
        "tensors_preview": [
            {
                "name": name,
                "dtype": _coerce_str(meta.get("dtype")),
                "shape": [_coerce_int(item) for item in _list_or_empty(meta.get("shape"))],
                "byte_count": _byte_count_from_offsets(meta.get("data_offsets")),
            }
            for name, meta in tensors[:8]
        ],
    }


def _read_npz_summary(path: Path, root: Path, warnings: list[str]) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".npy")]
            parsed_members: list[tuple[str, list[int], str]] = []
            for name in members[:MAX_ARTIFACTS]:
                try:
                    parsed = _read_npy_header(archive, name)
                except (UnicodeDecodeError, ValueError, SyntaxError, struct.error):
                    parsed = None
                if parsed is None:
                    continue
                shape, dtype = parsed
                parsed_members.append((name, shape, dtype))
    except (OSError, zipfile.BadZipFile) as exc:
        warnings.append(f"invalid_npz_header:{_relative(path, root)}:{exc.__class__.__name__}")
        return {}
    shapes = [shape for _, shape, _ in parsed_members]
    dtypes = [dtype for _, _, dtype in parsed_members]
    hidden_size = _common_vector_width(shapes)
    return {
        "tensor_count": len(members),
        "direction_count": len(members),
        "hidden_size": hidden_size,
        "dtype": _common_value(dtypes),
        "tensors_preview": [
            {"name": name, "shape": shape, "dtype": dtype}
            for name, shape, dtype in parsed_members[:8]
        ],
    }


def _read_npy_header(archive: zipfile.ZipFile, name: str) -> tuple[list[int], str] | None:
    with archive.open(name) as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            return None
        version = handle.read(2)
        if len(version) != 2:
            return None
        if version == b"\x01\x00":
            header_len_raw = handle.read(2)
            if len(header_len_raw) != 2:
                return None
            header_len = struct.unpack("<H", header_len_raw)[0]
        else:
            header_len_raw = handle.read(4)
            if len(header_len_raw) != 4:
                return None
            header_len = struct.unpack("<I", header_len_raw)[0]
        header = handle.read(header_len).decode("latin1").strip()
    payload = ast.literal_eval(header)
    if not isinstance(payload, dict):
        return None
    shape = [_coerce_int(item) for item in _list_or_empty(payload.get("shape"))]
    shape = [item for item in shape if item is not None]
    return shape, _coerce_str(payload.get("descr")) or ""


def _attach_artifacts_to_reports(reports: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> None:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        layer_idx = _coerce_int(artifact.get("layer_idx"))
        if layer_idx is None:
            continue
        by_layer.setdefault(layer_idx, []).append(artifact)
    for report in reports:
        layer_idx = _coerce_int(report.get("layer_idx"))
        if layer_idx is None:
            continue
        attached = by_layer.get(layer_idx, [])
        report["artifact_names"] = [artifact.get("name") for artifact in attached if artifact.get("name")]
        if report.get("hidden_size") is None:
            report["hidden_size"] = _common_value([artifact.get("hidden_size") for artifact in attached])


def _summary(
    *,
    root: Path,
    reports: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    sweep_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = _selected_artifact(artifacts, reports)
    selected_report = _report_for_layer(reports, _coerce_int(selected.get("layer_idx")) if selected else None)
    report_source = selected_report or _first_passing_report(reports) or (reports[0] if reports else {})
    hidden_size = _coerce_int((selected or {}).get("hidden_size")) or _coerce_int(report_source.get("hidden_size"))
    model_family = (
        _coerce_str((selected or {}).get("model_family"))
        or _coerce_str(report_source.get("model_family"))
        or _infer_model_family(str(root))
        or _infer_model_family_from_hidden_size(hidden_size)
    )
    passing_reports = [report for report in reports if report.get("verdict") == "pass"]
    health_status = "unknown"
    if reports:
        health_status = "pass" if passing_reports else "fail"

    return {
        "library_kind": "rpp_user_profile",
        "model_family": model_family,
        "hidden_size": hidden_size,
        "target_layer": _coerce_int((selected or {}).get("layer_idx")) or _coerce_int(report_source.get("layer_idx")),
        "direction_set_id": _coerce_str(report_source.get("direction_set_id")) or "directions_a",
        "yaml_sha256": _coerce_str(report_source.get("yaml_sha256")),
        "source_type": _coerce_str(report_source.get("source_type")),
        "source_schema_version": _coerce_str(report_source.get("source_schema_version")),
        "health_status": health_status,
        "health_report_path": report_source.get("report_path"),
        "selected_artifact": (selected or {}).get("name"),
        "selected_reason": _selected_reason(selected, reports),
        "report_count": len(reports),
        "artifact_count": len(artifacts),
        "sweep_summary_count": len(sweep_summaries),
        "pooling": _coerce_str(report_source.get("pooling")) or _common_value([item.get("pooling") for item in sweep_summaries]),
        "model_path": _coerce_str(report_source.get("model_path")) or _common_value([item.get("model_path") for item in sweep_summaries]),
        "n_directions": _coerce_int((selected or {}).get("direction_count")) or _coerce_int(report_source.get("n_directions")),
    }


def _manifest_summary(*, summary: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "library_kind": summary.get("library_kind"),
        "model_family": summary.get("model_family"),
        "hidden_size": summary.get("hidden_size"),
        "target_layer": summary.get("target_layer"),
        "direction_set_id": summary.get("direction_set_id"),
        "health_report": summary.get("health_report_path"),
    }
    checks = [
        {"name": key, "present": value not in (None, "", []), "value": value}
        for key, value in required.items()
    ]
    target_report = _report_for_layer(reports, _coerce_int(summary.get("target_layer")))
    checks.append({
        "name": "health_verdict",
        "present": target_report is not None,
        "value": (target_report or {}).get("verdict"),
        "passed": (target_report or {}).get("verdict") == "pass",
    })
    ready = all(check.get("present") for check in checks) and (target_report or {}).get("verdict") == "pass"
    return {
        "ready": ready,
        "required_keys": required,
        "checks": checks,
    }


def _selected_artifact(artifacts: list[dict[str, Any]], reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    safetensors = [artifact for artifact in artifacts if artifact.get("kind") == "safetensors"]
    if len(safetensors) == 1:
        return safetensors[0]
    passing_layers = {_coerce_int(report.get("layer_idx")) for report in reports if report.get("verdict") == "pass"}
    candidates = [
        artifact
        for artifact in safetensors or artifacts
        if _coerce_int(artifact.get("layer_idx")) in passing_layers
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def _selected_reason(selected: dict[str, Any] | None, reports: list[dict[str, Any]]) -> str:
    if selected is None:
        return "ambiguous_or_missing_exported_artifact"
    layer_idx = _coerce_int(selected.get("layer_idx"))
    report = _report_for_layer(reports, layer_idx)
    if selected.get("kind") == "safetensors" and report and report.get("verdict") == "pass":
        return "single_exported_safetensors_with_passing_health_report"
    if selected.get("kind") == "safetensors":
        return "single_exported_safetensors"
    return "single_matching_artifact"


def _report_for_layer(reports: list[dict[str, Any]], layer_idx: int | None) -> dict[str, Any] | None:
    if layer_idx is None:
        return None
    for report in reports:
        if _coerce_int(report.get("layer_idx")) == layer_idx:
            return report
    return None


def _first_passing_report(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    for report in sorted(reports, key=lambda item: item.get("layer_idx") or 0):
        if report.get("verdict") == "pass":
            return report
    return None


def _read_json(path: Path, warnings: list[str]) -> Any | None:
    try:
        if path.stat().st_size > MAX_JSON_REPORT_BYTES:
            warnings.append(f"json_report_too_large:{path.name}")
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"invalid_json:{path.name}:{exc.__class__.__name__}")
        return None


def _common_vector_width(shapes: list[Any]) -> int | None:
    widths: list[int] = []
    for shape in shapes:
        values = [_coerce_int(item) for item in _list_or_empty(shape)]
        values = [value for value in values if value is not None]
        if not values:
            continue
        widths.append(values[-1])
    return _common_value(widths)


def _common_value(values: list[Any]) -> Any | None:
    clean = [value for value in values if value not in (None, "")]
    if not clean:
        return None
    first = clean[0]
    if all(value == first for value in clean):
        return first
    return None


def _layer_from_name(name: str) -> int | None:
    match = re.search(r"directions_a_layer_(\d+)", name)
    if not match:
        return None
    return _coerce_int(match.group(1))


def _direction_set_id(yaml_path: Any) -> str:
    value = _coerce_str(yaml_path)
    if not value:
        return "directions_a"
    return Path(value).stem or "directions_a"


def _infer_model_family(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.lower().replace("_", "-")
    if "qwen3.5-9b" in normalized or "qwen35-9b" in normalized or "qwen3-5-9b" in normalized:
        return "qwen3.5-9b"
    if "qwen3.5-4b" in normalized or "qwen35-4b" in normalized or "qwen3-5-4b" in normalized:
        return "qwen3.5-4b"
    if "qwen3.6" in normalized or "qwen36" in normalized:
        return "qwen3.6"
    if "qwen3.5" in normalized or "qwen35" in normalized:
        return "qwen3.5"
    return None


def _infer_model_family_from_hidden_size(hidden_size: Any) -> str | None:
    size = _coerce_int(hidden_size)
    if size == 2560:
        return "qwen3.5-4b"
    if size == 4096:
        return "qwen3.5-9b"
    return None


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _byte_count_from_offsets(value: Any) -> int | None:
    offsets = [_coerce_int(item) for item in _list_or_empty(value)]
    offsets = [item for item in offsets if item is not None]
    if len(offsets) != 2:
        return None
    return max(0, offsets[1] - offsets[0])


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _object_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _coerce_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
