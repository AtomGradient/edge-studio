# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""A-library inspection endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.config import BROWSE_ROOTS
from backend.schemas.common import CreateTaskResponse
from backend.services.a_library_inspector import (
    ALibraryInspectionError,
    inspect_a_library_directory,
)
from backend.services.a_library_generation import (
    ALibraryGenerationError,
    generate_a_library as run_a_library_generation,
)
from backend.services.a_library_direction_sets import (
    DirectionSetValidationError,
    store_direction_yaml,
    validate_direction_yaml_text,
)
from backend.services.a_library_registry import (
    generated_a_library_history_roots,
    select_a_library_for_model_dir,
)
from backend.services.a_library_suggestion_service import (
    ALibrarySuggestionServiceError,
    clean_refined_description,
    direction_repair_instruction,
    json_object_from_text,
    parse_direction_suggestion_output,
    refine_domain_description as run_domain_description_refinement,
    serialize_direction_suggestions_to_yaml,
    suggest_a_library_directions as run_direction_suggestion,
)
from backend.services.model_manager import manager
from backend.services.task_manager import task_manager


router = APIRouter(prefix="/api/a_library", tags=["a-library"])


class ALibraryGenerateRequest(BaseModel):
    model_path: str = Field(..., min_length=1)
    yaml_path: str | None = None
    output_dir: str | None = None
    direction_set_id: str | None = None
    layers: list[int] | None = None
    sweep: bool = True
    pooling: Literal["last_real", "mean"] = "last_real"
    source_type: Literal["host_model_seed", "claude_authored", "manual"] = "manual"


class ALibraryValidateYAMLRequest(BaseModel):
    content: str = Field(..., min_length=1)
    direction_set_id: str | None = None
    persist: bool = True


class ALibraryDirectionRepairContext(BaseModel):
    worst_pairs: list[list[str]] = Field(default_factory=list)
    max_abs_cos: float | None = None
    mean_abs_cos: float | None = None
    signal_pass: bool | None = None
    validation_error_codes: list[str] = Field(default_factory=list)
    prev_direction_set_id: str | None = None
    reason: str | None = None


class ALibrarySuggestDirectionsRequest(BaseModel):
    domain_description: str = Field(..., min_length=2, max_length=1200)
    target_count: int = Field(default=10, ge=10, le=20)
    model_id: str | None = None
    repair_context: ALibraryDirectionRepairContext | None = None


class ALibraryRefineDomainDescriptionRequest(BaseModel):
    domain_description: str = Field(..., min_length=2, max_length=1200)
    model_id: str | None = None


class ALibraryDirectionSuggestionItem(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = ""
    domain: str = "custom"
    positive: list[str] = Field(default_factory=list)
    negative: list[str] = Field(default_factory=list)


class ALibraryGenerateFromSuggestionsRequest(BaseModel):
    model_path: str = Field(..., min_length=1)
    direction_set_id: str = Field(..., min_length=1, max_length=80)
    directions: list[ALibraryDirectionSuggestionItem] = Field(..., min_length=10, max_length=20)
    output_dir: str | None = None
    layers: list[int] | None = None
    sweep: bool = True
    pooling: Literal["last_real", "mean"] = "last_real"
    source_type: Literal["host_model_seed", "claude_authored", "manual"] = "host_model_seed"


def _safe_directory_path(path: str) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise HTTPException(404, "Path not found")
    if not target.is_dir():
        raise HTTPException(400, "Path is not a directory")
    real = str(target)
    if not any(os.path.commonpath([real, root]) == root for root in BROWSE_ROOTS):
        raise HTTPException(403, "Access denied")
    return target


@router.get("/inspect")
def inspect_a_library(path: str = Query(...)) -> dict:
    """Inspect a local A-library directory using manifest/report/header data."""

    try:
        return inspect_a_library_directory(_safe_directory_path(path))
    except ALibraryInspectionError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.a_library_inspection.v0",
                "status": "error",
                "error": exc.to_error(),
            },
        ) from exc


@router.get("/select")
def select_a_library(
    model_id: str | None = Query(default=None),
    model_path: str | None = Query(default=None),
    direction_set_id: str | None = Query(default=None),
) -> dict:
    """Select a model-matched A-library from bundled/generated registries."""

    if model_id:
        loaded = manager.get_model(model_id)
        if not loaded:
            raise HTTPException(404, "Model not loaded")
        return select_a_library_for_model_dir(loaded.model_dir, direction_set_id=direction_set_id)
    if model_path:
        return select_a_library_for_model_dir(
            _safe_directory_path_for_model(model_path),
            direction_set_id=direction_set_id,
        )
    raise HTTPException(400, "Provide model_id or model_path")


@router.get("/history")
def list_a_library_history(limit: int = Query(default=50, ge=1, le=200)) -> dict:
    """List generated A-library directories from the local EdgeStudio registry."""

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in generated_a_library_history_roots():
        for candidate in _generated_library_dirs(root):
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            item = _history_item_for_dir(candidate, root)
            if item is not None:
                items.append(item)
    items.sort(key=lambda row: row.get("created_at_unix") or 0, reverse=True)
    return {
        "ok": True,
        "schema_version": "edgestudio.a_library_history.v1",
        "roots": [str(root) for root in generated_a_library_history_roots()],
        "items": items[:limit],
    }


@router.post("/generate", response_model=CreateTaskResponse)
def generate_a_library(req: ALibraryGenerateRequest) -> CreateTaskResponse:
    """Run host-side A-library generation as a background task."""

    model_path = _safe_directory_path_for_model(req.model_path)
    yaml_path = _safe_file_path(req.yaml_path) if req.yaml_path else None
    output_dir = _safe_output_dir(req.output_dir) if req.output_dir else None
    task_id = task_manager.create_task(
        metadata={
            "kind": "a_library_generation",
            "model_path": str(model_path),
            "output_dir": str(output_dir) if output_dir else None,
            "direction_set_id": req.direction_set_id or "directions_a",
        }
    )

    def _run(progress_callback=None):
        try:
            return run_a_library_generation(
                model_path=model_path,
                yaml_path=yaml_path,
                output_dir=output_dir,
                layers=req.layers,
                sweep=req.sweep,
                pooling=req.pooling,
                direction_set_id=req.direction_set_id,
                source_type=req.source_type,
                progress_callback=progress_callback,
            )
        except ALibraryGenerationError:
            raise

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.post("/generate_from_suggestions", response_model=CreateTaskResponse)
def generate_a_library_from_suggestions(req: ALibraryGenerateFromSuggestionsRequest) -> CreateTaskResponse:
    """Validate suggested directions, persist YAML, then build an A-library task."""

    model_path = _safe_directory_path_for_model(req.model_path)
    output_dir = _safe_output_dir(req.output_dir) if req.output_dir else None
    yaml_text = serialize_direction_suggestions_to_yaml(
        direction_set_id=req.direction_set_id,
        directions=[_model_to_dict(item) for item in req.directions],
    )
    try:
        stored = store_direction_yaml(yaml_text, direction_set_id=req.direction_set_id)
    except DirectionSetValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.a_library_generate_from_suggestions.v1",
                "status": "invalid_direction_set",
                "validation": exc.report,
            },
        ) from exc

    yaml_path = Path(stored["stored_path"]).expanduser().resolve()
    task_id = task_manager.create_task(
        metadata={
            "kind": "a_library_generation_from_suggestions",
            "model_path": str(model_path),
            "output_dir": str(output_dir) if output_dir else None,
            "direction_set_id": stored.get("direction_set_id") or req.direction_set_id,
            "source_type": req.source_type,
            "yaml_path": str(yaml_path),
        }
    )

    def _run(progress_callback=None):
        result = run_a_library_generation(
            model_path=model_path,
            yaml_path=yaml_path,
            output_dir=output_dir,
            layers=req.layers,
            sweep=req.sweep,
            pooling=req.pooling,
            direction_set_id=stored.get("direction_set_id") or req.direction_set_id,
            source_type=req.source_type,
            progress_callback=progress_callback,
        )
        return result | {
            "source_type": req.source_type,
            "stored_direction_yaml": stored,
        }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.post("/validate_yaml")
def validate_a_library_yaml(req: ALibraryValidateYAMLRequest) -> dict:
    """Validate and optionally persist a custom A-library direction-set YAML."""

    report = validate_direction_yaml_text(req.content, direction_set_id=req.direction_set_id)
    if not report.get("ok") or not req.persist:
        return report
    try:
        return store_direction_yaml(req.content, direction_set_id=report.get("direction_set_id"))
    except DirectionSetValidationError as exc:
        return exc.report


@router.post("/suggest_directions")
def suggest_a_library_directions(req: ALibrarySuggestDirectionsRequest) -> dict:
    """Use the currently loaded host model to draft editable direction candidates."""

    loaded = manager.get_model(req.model_id) if req.model_id else _first_loaded_text_model()
    if not loaded:
        raise HTTPException(404, "Load a text model before requesting direction suggestions.")
    if loaded.category not in ("llm", "vlm"):
        raise HTTPException(400, "A-library direction suggestions require a loaded text LLM or VLM.")
    try:
        return run_direction_suggestion(
            host_model_id=loaded.model_id,
            model_name=Path(loaded.model_dir).name,
            domain_description=req.domain_description,
            target_count=req.target_count,
            repair_context=_model_to_dict(req.repair_context) if req.repair_context else None,
        )
    except ALibrarySuggestionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_response()) from exc


@router.post("/refine_domain_description")
def refine_a_library_domain_description(req: ALibraryRefineDomainDescriptionRequest) -> dict:
    """Use the currently loaded host model to rewrite rough ideas for the editor."""

    loaded = manager.get_model(req.model_id) if req.model_id else _first_loaded_text_model()
    if not loaded:
        raise HTTPException(404, "Load a text model before refining the domain description.")
    if loaded.category not in ("llm", "vlm"):
        raise HTTPException(400, "A-library description refinement requires a loaded text LLM or VLM.")
    try:
        return run_domain_description_refinement(
            host_model_id=loaded.model_id,
            model_name=Path(loaded.model_dir).name,
            domain_description=req.domain_description,
        )
    except ALibrarySuggestionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_response()) from exc


def _safe_directory_path_for_model(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise HTTPException(404, "Model path not found")
    if not target.is_dir():
        raise HTTPException(400, "Model path is not a directory")
    if not (target / "config.json").is_file():
        raise HTTPException(400, "Model path must contain config.json")
    _assert_allowed(target)
    return target


def _safe_file_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise HTTPException(404, "File not found")
    if not target.is_file():
        raise HTTPException(400, "Path is not a file")
    _assert_allowed(target)
    return target


def _safe_output_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists() and not target.is_dir():
        raise HTTPException(400, "Output path is not a directory")
    _assert_allowed(target.parent if not target.exists() else target)
    return target


def _assert_allowed(target: Path) -> None:
    real = str(target)
    if not any(os.path.commonpath([real, root]) == root for root in BROWSE_ROOTS):
        raise HTTPException(403, "Access denied")


def _generated_library_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    candidates: list[Path] = []
    if _looks_like_a_library_dir(root):
        candidates.append(root)
    try:
        children = sorted((child for child in root.iterdir() if child.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return candidates
    candidates.extend(child for child in children if _looks_like_a_library_dir(child))
    return candidates


def _looks_like_a_library_dir(path: Path) -> bool:
    return (
        (path / "rpp_a_library_manifest.json").is_file()
        or any(path.glob("directions_a_layer_*_report.json"))
        or any(path.glob("directions_a_layer_*.safetensors"))
    )


def _history_item_for_dir(path: Path, root: Path) -> dict[str, Any] | None:
    try:
        inspection = inspect_a_library_directory(path)
        stat = path.stat()
    except (OSError, ALibraryInspectionError):
        return None
    summary = inspection.get("summary") if isinstance(inspection.get("summary"), dict) else {}
    manifest = inspection.get("manifest") if isinstance(inspection.get("manifest"), dict) else {}
    return {
        "path": str(path),
        "root": str(root),
        "model_name": path.name.split("_20", 1)[0] or path.name,
        "created_at_unix": stat.st_mtime,
        "direction_set_id": summary.get("direction_set_id"),
        "target_layer": summary.get("target_layer"),
        "health_status": summary.get("health_status"),
        "health_verdict": summary.get("health_status"),
        "library_kind": summary.get("library_kind"),
        "model_family": summary.get("model_family"),
        "hidden_size": summary.get("hidden_size"),
        "n_directions": summary.get("n_directions"),
        "ready": bool(manifest.get("ready")),
        "warnings": inspection.get("warnings") or [],
    }


def _first_loaded_text_model():
    for loaded in manager.list_models():
        if loaded.category in ("llm", "vlm"):
            return loaded
    return None


def _parse_direction_suggestion_output(payload: dict[str, Any], target_count: int) -> list[dict[str, Any]]:
    try:
        return parse_direction_suggestion_output(payload, target_count)
    except ALibrarySuggestionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _json_object_from_text(text: str) -> tuple[dict[str, Any], str]:
    try:
        return json_object_from_text(text)
    except ALibrarySuggestionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _clean_refined_description(text: str) -> str:
    try:
        return clean_refined_description(text)
    except ALibrarySuggestionServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


def _direction_repair_instruction(repair_context: ALibraryDirectionRepairContext | None) -> str:
    return direction_repair_instruction(_model_to_dict(repair_context) if repair_context else None)


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {}
