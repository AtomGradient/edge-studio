# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Read-only model discovery commands for the ``edge models`` CLI group."""

from __future__ import annotations

import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.cli.fingerprints import dir_size_bytes, model_dir_integrity
from backend.cli.model_paths import discover_local_model_paths, model_cache_roots
from backend.resources.paths import script_path


LIST_SCHEMA_VERSION = "edge.models.list.report.v1"
WHERE_SCHEMA_VERSION = "edge.models.where.report.v1"
DOCTOR_SCHEMA_VERSION = "edge.models.doctor.report.v1"


@dataclass(frozen=True)
class LocalModel:
    name: str
    path: str
    size_bytes: int
    complete: bool
    issues: tuple[str, ...] = ()
    expected_size_bytes: int | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "complete": self.complete,
            "issues": list(self.issues),
        }
        if self.expected_size_bytes is not None:
            payload["expected_size_bytes"] = self.expected_size_bytes
        return payload


@dataclass(frozen=True)
class CatalogResolution:
    status: str
    input: str
    model_id: str | None
    name: str | None
    download_hint: str | None
    category: str | None
    size_gb: float | None
    catalog_source: str
    catalog_version: str
    matched_by: str | None
    alternates: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "input": self.input,
            "model_id": self.model_id,
            "name": self.name,
            "download_hint": self.download_hint,
            "category": self.category,
            "size_gb": self.size_gb,
            "catalog_source": self.catalog_source,
            "catalog_version": self.catalog_version,
            "matched_by": self.matched_by,
            "alternates": self.alternates,
        }


@dataclass(frozen=True)
class ModelsListReport:
    schema_version: str
    catalog: dict[str, object]
    roots: list[str]
    local_models: list[LocalModel]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "catalog": self.catalog,
            "roots": self.roots,
            "local_models": [model.as_dict() for model in self.local_models],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ModelWhereReport:
    schema_version: str
    status: str
    resolution: CatalogResolution
    local_matches: list[LocalModel]
    fetch_command: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "resolution": self.resolution.as_dict(),
            "local_matches": [match.as_dict() for match in self.local_matches],
            "fetch_command": self.fetch_command,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ModelDoctorReport:
    schema_version: str
    overall_status: str
    where: ModelWhereReport
    downloader_status: dict[str, object]
    source_plan: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "overall_status": self.overall_status,
            "where": self.where.as_dict(),
            "downloader_status": self.downloader_status,
            "source_plan": self.source_plan,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


def list_models(*, env: Mapping[str, str] | None = None) -> ModelsListReport:
    data, source, _signature = _load_catalog_data()
    models = _discover_local_models(env)
    roots = [str(root) for root in model_cache_roots(env)]
    return ModelsListReport(
        schema_version=LIST_SCHEMA_VERSION,
        catalog={
            "source": source,
            "version": _catalog_version(data),
            "total_models": len(data.get("models", [])),
        },
        roots=roots,
        local_models=models,
    )


def where_model(model_ref: str, *, env: Mapping[str, str] | None = None) -> ModelWhereReport:
    resolution = resolve_model_reference(model_ref)
    local_matches = _matching_local_models(resolution, env)
    expected_size_bytes = _expected_size_bytes(resolution)
    if expected_size_bytes is not None:
        local_matches = [_with_expected_integrity(match, expected_size_bytes) for match in local_matches]
    if resolution.status == "unknown":
        status = "unknown"
    elif any(match.complete for match in local_matches):
        status = "ok"
    elif local_matches:
        status = "incomplete"
    else:
        status = "missing"
    fetch_command = None if status == "ok" or resolution.status == "unknown" else f"edge models fetch {model_ref}"
    return ModelWhereReport(
        schema_version=WHERE_SCHEMA_VERSION,
        status=status,
        resolution=resolution,
        local_matches=local_matches,
        fetch_command=fetch_command,
    )


def doctor_model(model_ref: str, *, env: Mapping[str, str] | None = None) -> ModelDoctorReport:
    where = where_model(model_ref, env=env)
    if where.status == "unknown":
        overall = "fail"
    elif where.status == "ok":
        overall = "ok"
    else:
        overall = "warn"
    return ModelDoctorReport(
        schema_version=DOCTOR_SCHEMA_VERSION,
        overall_status=overall,
        where=where,
        downloader_status=_downloader_status(),
        source_plan=_source_plan(),
    )


def resolve_model_reference(model_ref: str) -> CatalogResolution:
    data, source, _signature = _load_catalog_data()
    catalog_version = _catalog_version(data)
    models = [model for model in data.get("models", []) if isinstance(model, dict)]
    matches = _catalog_matches(model_ref, models)
    if matches:
        best = matches[0]
        entry = best["model"]
        alternates = [
            _catalog_entry_summary(match["model"])
            for match in matches[1:6]
            if match["model"].get("download_hint") != entry.get("download_hint")
        ]
        return CatalogResolution(
            status="resolved",
            input=model_ref,
            model_id=_optional_str(entry.get("id")),
            name=_optional_str(entry.get("name")),
            download_hint=_optional_str(entry.get("download_hint")),
            category=_optional_str(entry.get("category")),
            size_gb=_optional_float(entry.get("size_gb")),
            catalog_source=source,
            catalog_version=catalog_version,
            matched_by=_optional_str(best.get("matched_by")),
            alternates=alternates,
        )

    if "/" in model_ref.strip():
        return CatalogResolution(
            status="external",
            input=model_ref,
            model_id=None,
            name=model_ref.strip(),
            download_hint=model_ref.strip(),
            category=None,
            size_gb=None,
            catalog_source=source,
            catalog_version=catalog_version,
            matched_by="repo_id",
            alternates=[],
        )

    return CatalogResolution(
        status="unknown",
        input=model_ref,
        model_id=None,
        name=None,
        download_hint=None,
        category=None,
        size_gb=None,
        catalog_source=source,
        catalog_version=catalog_version,
        matched_by=None,
        alternates=[],
    )


def format_models_list(report: ModelsListReport) -> str:
    lines = [
        f"Edge models ({report.schema_version})",
        (
            "catalog: "
            f"{report.catalog.get('source')} "
            f"version={report.catalog.get('version')} "
            f"models={report.catalog.get('total_models')}"
        ),
        f"local models: {len(report.local_models)}",
    ]
    for model in report.local_models[:20]:
        complete = "complete" if model.complete else "incomplete"
        lines.append(f"- {model.name} ({complete})")
        lines.append(f"  path: {model.path}")
        if model.issues:
            lines.append(f"  issues: {', '.join(model.issues[:3])}")
    if len(report.local_models) > 20:
        lines.append(f"... {len(report.local_models) - 20} more local model(s)")
    if not report.local_models:
        lines.append("No local models found in configured roots.")
    return "\n".join(lines)


def format_model_where(report: ModelWhereReport) -> str:
    resolution = report.resolution
    lines = [
        f"Edge models where ({report.schema_version})",
        f"status: {report.status}",
        f"input: {resolution.input}",
    ]
    if resolution.status == "unknown":
        lines.append("catalog: no matching model entry")
        return "\n".join(lines)
    lines.append(f"catalog: {resolution.name or resolution.download_hint}")
    if resolution.model_id:
        lines.append(f"model id: {resolution.model_id}")
    if resolution.download_hint:
        lines.append(f"download hint: {resolution.download_hint}")
    if report.local_matches:
        for match in report.local_matches:
            complete = "complete" if match.complete else "incomplete"
            lines.append(f"path ({complete}): {match.path}")
            if match.issues:
                lines.append(f"issues: {', '.join(match.issues[:5])}")
    else:
        lines.append("path: not installed")
    if report.fetch_command:
        lines.append(f"next: {report.fetch_command}")
    if resolution.alternates:
        lines.append(f"alternates: {len(resolution.alternates)} additional catalog match(es)")
    return "\n".join(lines)


def format_model_doctor(report: ModelDoctorReport) -> str:
    lines = [
        f"Edge models doctor ({report.schema_version})",
        f"overall: {report.overall_status}",
        f"model status: {report.where.status}",
    ]
    resolution = report.where.resolution
    if resolution.download_hint:
        lines.append(f"download hint: {resolution.download_hint}")
    if report.where.local_matches:
        for match in report.where.local_matches:
            complete = "complete" if match.complete else "incomplete"
            lines.append(f"local ({complete}): {match.path}")
            if match.issues:
                lines.append(f"local issues: {', '.join(match.issues[:5])}")
    elif report.where.fetch_command:
        lines.append(f"local: missing; run `{report.where.fetch_command}`")
    else:
        lines.append("local: no matching model")

    downloader = report.downloader_status
    lines.append(
        "downloaders: "
        f"msd.sh={'yes' if downloader.get('msd_script') else 'no'}, "
        f"hfd.sh={'yes' if downloader.get('hfd_script') else 'no'}, "
        f"aria2c={'yes' if downloader.get('aria2c') else 'no'}, "
        f"modelscope={'yes' if downloader.get('modelscope_python') else 'no'}"
    )
    plan = report.source_plan
    lines.append("source plan: no network probe in B2a")
    lines.append(f"- mainland: {' -> '.join(plan['china_mainland_order'])}")
    lines.append(f"- international: {' -> '.join(plan['international_order'])}")
    return "\n".join(lines)


def where_exit_code(report: ModelWhereReport) -> int:
    return 0 if report.status == "ok" else 1


def doctor_exit_code(report: ModelDoctorReport) -> int:
    return 1 if report.overall_status == "fail" else 0


def _load_catalog_data() -> tuple[dict[str, Any], str, str]:
    from backend.core.model_catalog_sync import load_effective_catalog

    return load_effective_catalog(start_background_refresh=False)


def _catalog_version(data: dict[str, Any]) -> str:
    meta = data.get("_meta", {})
    return str(meta.get("version", "unknown")) if isinstance(meta, dict) else "unknown"


def _catalog_matches(model_ref: str, models: Sequence[dict[str, Any]]) -> list[dict[str, object]]:
    terms = _query_terms(model_ref)
    matches: list[dict[str, object]] = []
    for index, model in enumerate(models):
        best_score: int | None = None
        matched_by: str | None = None
        for label, alias in _model_aliases(model):
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue
            for term in terms:
                term_norm = _normalize(term)
                if not term_norm:
                    continue
                score: int | None = None
                if alias.lower() == term.lower():
                    score = 0
                elif alias_norm == term_norm:
                    score = 1
                elif alias_norm.startswith(term_norm):
                    score = 2
                elif term_norm in alias_norm:
                    score = 3
                if score is not None and (best_score is None or score < best_score):
                    best_score = score
                    matched_by = label
        if best_score is not None:
            matches.append({"score": best_score, "index": index, "model": model, "matched_by": matched_by})
    matches.sort(key=lambda item: (item["score"], item["index"]))
    return matches


def _query_terms(model_ref: str) -> list[str]:
    value = model_ref.strip()
    terms = [value]
    if "/" in value:
        terms.append(value.split("/")[-1])
    return [term for term in terms if term]


def _model_aliases(model: dict[str, Any]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for key in ("id", "name", "download_hint", "family"):
        value = model.get(key)
        if value:
            aliases.append((key, str(value)))
    hint = str(model.get("download_hint", ""))
    if "/" in hint:
        aliases.append(("repo_name", hint.split("/")[-1]))
    return aliases


def _matching_local_models(
    resolution: CatalogResolution,
    env: Mapping[str, str] | None,
) -> list[LocalModel]:
    if resolution.status == "unknown":
        return []
    needles = [
        value
        for value in (
            resolution.model_id,
            resolution.name,
            resolution.download_hint,
            resolution.download_hint.split("/")[-1] if resolution.download_hint and "/" in resolution.download_hint else None,
        )
        if value
    ]
    normalized_needles = [_normalize(value) for value in needles]
    matches: list[LocalModel] = []
    for model in _discover_local_models(env):
        haystack = _normalize(f"{model.name} {model.path}")
        if any(needle and needle in haystack for needle in normalized_needles):
            matches.append(model)
    return matches


def _discover_local_models(env: Mapping[str, str] | None) -> list[LocalModel]:
    discovered = discover_local_model_paths(env)
    models: list[LocalModel] = []
    for name, path in sorted(discovered.items(), key=lambda item: item[0].lower()):
        model_path = Path(path)
        integrity = model_dir_integrity(model_path)
        models.append(
            LocalModel(
                name=name,
                path=path,
                size_bytes=dir_size_bytes(model_path),
                complete=integrity.complete,
                issues=integrity.issues,
            )
        )
    return models


def _with_expected_integrity(model: LocalModel, expected_size_bytes: int) -> LocalModel:
    integrity = model_dir_integrity(Path(model.path), expected_size_bytes=expected_size_bytes)
    return LocalModel(
        name=model.name,
        path=model.path,
        size_bytes=model.size_bytes,
        complete=integrity.complete,
        issues=integrity.issues,
        expected_size_bytes=expected_size_bytes,
    )


def _expected_size_bytes(resolution: CatalogResolution) -> int | None:
    if resolution.size_gb is None or resolution.size_gb <= 0:
        return None
    return int(resolution.size_gb * 1_000_000_000)


def _downloader_status() -> dict[str, object]:
    hfd_script = script_path("hfd.sh")
    msd_script = script_path("msd.sh")
    return {
        "hfd_script": hfd_script.exists(),
        "hfd_path": str(hfd_script),
        "msd_script": msd_script.exists(),
        "msd_path": str(msd_script),
        "aria2c": shutil.which("aria2c") is not None,
        "modelscope_cli": shutil.which("modelscope") is not None,
        "modelscope_python": importlib.util.find_spec("modelscope") is not None,
        "network_probe": "not_run",
    }


def _source_plan() -> dict[str, object]:
    return {
        "strategy": "auto_planned_no_network_probe",
        "network_probe": "not_run",
        "explicit_sources": ["auto", "modelscope", "huggingface", "hf-mirror"],
        "china_mainland_order": ["modelscope", "hf-mirror", "huggingface"],
        "international_order": ["huggingface", "hf-mirror", "modelscope"],
        "note": "B2a is read-only. B2b fetch will run network probes and write fetch receipts.",
    }


def _catalog_entry_summary(model: dict[str, Any]) -> dict[str, object]:
    return {
        "id": model.get("id"),
        "name": model.get("name"),
        "download_hint": model.get("download_hint"),
        "category": model.get("category"),
        "size_gb": model.get("size_gb"),
    }


def _normalize(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
