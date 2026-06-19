# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Registry and selector helpers for RPP A-library artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .app_dirs import data_path, unique_roots


A_LIBRARY_MANIFEST_VERSION = "edgestudio.rpp_a_library_manifest.v1"
A_LIBRARY_SELECTION_SCHEMA_VERSION = "edgestudio.a_library_selection.v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def scaffold_rpp_dir() -> Path:
    default = repo_root() / "edge-scaffold"
    scaffold = Path(os.environ.get("EDGE_SCAFFOLD_DIR", str(default))).expanduser()
    return scaffold.resolve() / "Resources" / "RPP"


def generated_a_library_root() -> Path:
    configured = os.environ.get("EDGESTUDIO_A_LIBRARY_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return data_path("a_libraries")


def generated_a_library_history_roots() -> list[Path]:
    return [generated_a_library_root()]


def a_library_search_roots() -> list[Path]:
    return unique_roots(*generated_a_library_history_roots(), scaffold_rpp_dir())


def model_metadata_from_dir(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    config = _read_json(root / "config.json")
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    model_name = root.name
    hidden_size = _coerce_int(config.get("hidden_size")) or _coerce_int(text_config.get("hidden_size"))
    layer_count = (
        _coerce_int(config.get("num_hidden_layers"))
        or _coerce_int(config.get("n_layer"))
        or _coerce_int(config.get("num_layers"))
        or _coerce_int(text_config.get("num_hidden_layers"))
        or _coerce_int(text_config.get("n_layer"))
        or _coerce_int(text_config.get("num_layers"))
    )
    return {
        "model_name": model_name,
        "model_dir": str(root),
        "model_family": infer_model_family(model_name, config=config, hidden_size=hidden_size),
        "hidden_size": hidden_size,
        "layer_count": layer_count,
        "is_moe": _is_moe_config(config, model_name),
    }


def infer_model_family(
    value: str | None,
    *,
    config: dict[str, Any] | None = None,
    hidden_size: int | None = None,
) -> str | None:
    text = " ".join(
        item
        for item in [
            value or "",
            str((config or {}).get("model_type") or ""),
            str((config or {}).get("architectures") or ""),
        ]
        if item
    ).lower().replace("_", "-")
    if "qwen3.6" in text or "qwen36" in text:
        if "9b" in text:
            return "qwen3.6-9b"
        if "4b" in text:
            return "qwen3.6-4b"
        if hidden_size:
            return _qwen_family_from_hidden_size("qwen3.6", hidden_size)
        return "qwen3.6"
    if "qwen3.5" in text or "qwen35" in text or "qwen3-5" in text:
        if "9b" in text:
            return "qwen3.5-9b"
        if "4b" in text:
            return "qwen3.5-4b"
        if hidden_size:
            return _qwen_family_from_hidden_size("qwen3.5", hidden_size)
        return "qwen3.5"
    if hidden_size == 2560:
        return "qwen3.5-4b"
    if hidden_size == 4096:
        return "qwen3.5-9b"
    return None


def discover_a_libraries(search_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    roots = search_roots or a_library_search_roots()
    entries: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        manifests = [root / "rpp_a_library_manifest.json"] if (root / "rpp_a_library_manifest.json").is_file() else []
        manifests.extend(path for path in root.rglob("rpp_a_library_manifest.json") if path not in manifests)
        for manifest_path in manifests:
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != A_LIBRARY_MANIFEST_VERSION:
                continue
            libraries = manifest.get("libraries")
            if not isinstance(libraries, list):
                continue
            for item in libraries:
                if isinstance(item, dict):
                    entries.append(_normalize_manifest_item(item, manifest_path.parent))
    return entries


def select_a_library_for_model_dir(
    model_dir: str | Path,
    *,
    direction_set_id: str | None = None,
    search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    return select_a_library_for_model(
        model_metadata_from_dir(model_dir),
        direction_set_id=direction_set_id,
        search_roots=search_roots,
    )


def select_a_library_for_model(
    model: dict[str, Any],
    *,
    direction_set_id: str | None = None,
    search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    model_family = _coerce_str(model.get("model_family"))
    hidden_size = _coerce_int(model.get("hidden_size"))
    layer_count = _coerce_int(model.get("layer_count"))
    is_moe = bool(model.get("is_moe"))
    requested_direction_set_id = _coerce_str(direction_set_id)
    candidates = discover_a_libraries(search_roots)
    scored = [
        _score_candidate(
            candidate,
            model_family,
            hidden_size,
            layer_count,
            is_moe,
            requested_direction_set_id,
        )
        for candidate in candidates
    ]
    matches = [row for row in scored if row["match"]]
    selected = matches[0]["candidate"] if matches else None
    reasons = _selection_reasons(
        model_family,
        hidden_size,
        is_moe,
        requested_direction_set_id,
        scored,
    )
    return {
        "ok": selected is not None,
        "schema_version": A_LIBRARY_SELECTION_SCHEMA_VERSION,
        "status": "matched" if selected else "missing",
        "model": {
            "model_name": model.get("model_name"),
            "model_dir": model.get("model_dir"),
            "model_family": model_family,
            "hidden_size": hidden_size,
            "layer_count": layer_count,
            "is_moe": is_moe,
        },
        "direction_set_id": requested_direction_set_id,
        "selected": selected,
        "candidates": [row["candidate"] | {"match_reasons": row["reasons"]} for row in scored],
        "reasons": reasons,
        "recommended_action": (
            "use_selected_a_library"
            if selected
            else "moe_a_library_research_required_before_rpp_or_scaffold_export"
            if is_moe
            else "generate_a_library_for_model_before_rpp_or_scaffold_export"
        ),
    }


def manifest_item_for_generated_library(
    *,
    output_dir: Path,
    model_family: str,
    hidden_size: int,
    layer_count: int,
    target_layer: int,
    n_directions: int,
    artifact_name: str,
    health_report_name: str,
    health_verdict: str,
    pooling: str,
    direction_set_id: str = "directions_a",
    yaml_sha256: str | None = None,
    source_type: str | None = None,
    source_schema_version: str | None = None,
) -> dict[str, Any]:
    artifact_path = output_dir / artifact_name
    return {
        "library_id": f"{model_family}/rpp_user_profile/{direction_set_id}/layer_{target_layer}",
        "library_kind": "rpp_user_profile",
        "model_family": model_family,
        "hidden_size": hidden_size,
        "layer_count": layer_count,
        "target_layer": target_layer,
        "direction_set_id": direction_set_id,
        "yaml_sha256": yaml_sha256 or "",
        "source_type": source_type or "",
        "source_schema_version": source_schema_version or "",
        "artifact": artifact_name,
        "artifact_sha256": _sha256_file(artifact_path) if artifact_path.is_file() else "",
        "health_report": health_report_name,
        "health_verdict": health_verdict,
        "pooling": pooling,
        "n_directions": n_directions,
    }


def write_manifest(output_dir: Path, libraries: list[dict[str, Any]], default_library_id: str) -> Path:
    manifest_path = output_dir / "rpp_a_library_manifest.json"
    payload = {
        "schema_version": A_LIBRARY_MANIFEST_VERSION,
        "default_library_id": default_library_id,
        "libraries": libraries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def _score_candidate(
    candidate: dict[str, Any],
    model_family: str | None,
    hidden_size: int | None,
    layer_count: int | None,
    is_moe: bool,
    direction_set_id: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    match = True
    if is_moe:
        match = False
        reasons.append("moe_a_library_unsupported")
    if not model_family:
        match = False
        reasons.append("model_family_unknown")
    elif candidate.get("model_family") != model_family:
        match = False
        reasons.append("model_family_mismatch")
    if hidden_size is None:
        match = False
        reasons.append("model_hidden_size_unknown")
    elif candidate.get("hidden_size") != hidden_size:
        match = False
        reasons.append("hidden_size_mismatch")
    if layer_count is None:
        match = False
        reasons.append("model_layer_count_unknown")
    elif candidate.get("layer_count") not in (None, 0, layer_count):
        match = False
        reasons.append("layer_count_mismatch")
    if direction_set_id and candidate.get("direction_set_id") != direction_set_id:
        match = False
        reasons.append("direction_set_id_mismatch")
    if candidate.get("library_kind") != "rpp_user_profile":
        match = False
        reasons.append("library_kind_mismatch")
    if candidate.get("health_verdict") != "pass":
        match = False
        reasons.append("health_not_pass")
    if not candidate.get("artifact_exists"):
        match = False
        reasons.append("artifact_missing")
    if not candidate.get("health_report_exists"):
        match = False
        reasons.append("health_report_missing")
    if candidate.get("artifact_sha256_ok") is False:
        match = False
        reasons.append("artifact_sha256_mismatch")
    if match:
        reasons.append("matched")
    return {"candidate": candidate, "match": match, "reasons": reasons}


def _selection_reasons(
    model_family: str | None,
    hidden_size: int | None,
    is_moe: bool,
    direction_set_id: str | None,
    scored: list[dict[str, Any]],
) -> list[str]:
    if is_moe:
        return ["moe_a_library_unsupported"]
    if not model_family:
        return ["model_family_unknown"]
    if hidden_size is None:
        return ["model_hidden_size_unknown"]
    if not scored:
        return ["no_a_library_manifests_found"]
    if not any(row["candidate"].get("model_family") == model_family for row in scored):
        return [f"no_candidate_for_model_family:{model_family}"]
    if hidden_size is not None and not any(
        row["candidate"].get("model_family") == model_family
        and row["candidate"].get("hidden_size") == hidden_size
        for row in scored
    ):
        return [f"no_candidate_for_hidden_size:{hidden_size}"]
    if direction_set_id and not any(
        row["candidate"].get("model_family") == model_family
        and row["candidate"].get("hidden_size") == hidden_size
        and row["candidate"].get("direction_set_id") == direction_set_id
        for row in scored
    ):
        return [f"no_candidate_for_direction_set_id:{direction_set_id}"]
    failures = sorted({reason for row in scored for reason in row["reasons"] if reason != "matched"})
    return failures or ["no_matching_candidate"]


def _normalize_manifest_item(item: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    artifact = _coerce_str(item.get("artifact")) or ""
    report = _coerce_str(item.get("health_report")) or ""
    artifact_path = (base_dir / artifact).resolve() if artifact else None
    report_path = (base_dir / report).resolve() if report else None
    expected_sha = _coerce_str(item.get("artifact_sha256"))
    actual_sha = _sha256_file(artifact_path) if artifact_path and artifact_path.is_file() else None
    normalized = {
        key: value
        for key, value in item.items()
        if key not in {"artifact_path", "health_report_path", "source_manifest"}
    }
    normalized.update({
        "source_manifest": str((base_dir / "rpp_a_library_manifest.json").resolve()),
        "artifact_path": str(artifact_path) if artifact_path else None,
        "health_report_path": str(report_path) if report_path else None,
        "artifact_exists": bool(artifact_path and artifact_path.is_file()),
        "health_report_exists": bool(report_path and report_path.is_file()),
        "artifact_sha256_actual": actual_sha,
        "artifact_sha256_ok": None if not expected_sha or actual_sha is None else expected_sha == actual_sha,
    })
    return normalized


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _qwen_family_from_hidden_size(prefix: str, hidden_size: int) -> str:
    if hidden_size == 2560:
        return f"{prefix}-4b"
    if hidden_size == 4096:
        return f"{prefix}-9b"
    return prefix


def _is_moe_config(config: dict[str, Any], model_name: str) -> bool:
    text = model_name.lower()
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return (
        "moe" in text
        or "a3b" in text
        or "a10b" in text
        or _coerce_int(config.get("num_experts")) not in (None, 0)
        or _coerce_int(config.get("num_routed_experts")) not in (None, 0)
        or _coerce_int(text_config.get("num_experts")) not in (None, 0)
        or _coerce_int(text_config.get("num_routed_experts")) not in (None, 0)
    )


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


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
