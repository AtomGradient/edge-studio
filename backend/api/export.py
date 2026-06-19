# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Export endpoints (GGUF, CoreML, Swift, Scaffold ZIP)."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.schemas.common import CreateTaskResponse
from backend.schemas.export import (
    CoreMLExportRequest,
    EdgeRuntimeExportRequest,
    EdgeRuntimeExportResponse,
    GGUFExportRequest,
    ScaffoldExportRequest,
    ScaffoldZipExportRequest,
    SwiftCodeRequest,
    SwiftCodeResponse,
)
from backend.services.model_manager import manager
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["export"])


def _serialize_export_result(result) -> dict:
    """Serialize a GGUFExportResult or CoreMLExportResult."""
    return {
        "success": result.success,
        "output_path": result.output_path,
        "output_size_bytes": result.output_size_bytes,
        "duration_seconds": result.duration_seconds,
        "error_message": getattr(result, "error_message", ""),
    }


@router.post("/{model_id}/export/gguf", response_model=CreateTaskResponse)
def export_gguf(model_id: str, req: GGUFExportRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.export_gguf import export_to_gguf

        result = export_to_gguf(
            model_dir=loaded.model_dir,
            quantization=req.quant_type,
            output_path=req.output_path,
            progress_callback=progress_callback,
        )
        return _serialize_export_result(result)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.post("/{model_id}/export/coreml", response_model=CreateTaskResponse)
def export_coreml(model_id: str, req: CoreMLExportRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.export_coreml import export_to_coreml

        result = export_to_coreml(
            model_dir=loaded.model_dir,
            compute_units=req.compute_units,
            max_seq_len=req.max_seq_length,
            progress_callback=progress_callback,
        )
        return _serialize_export_result(result)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.post("/{model_id}/export/swift", response_model=SwiftCodeResponse)
def generate_swift(model_id: str, req: SwiftCodeRequest) -> SwiftCodeResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.swift_codegen import generate_swift_code

    code = generate_swift_code(
        model_dir=loaded.model_dir,
        model_package_name=req.package_name,
        max_tokens=req.default_max_tokens,
    )
    return SwiftCodeResponse(code=code, filename=f"{req.package_name}.swift")


@router.post("/{model_id}/export/edge-runtime", response_model=EdgeRuntimeExportResponse)
def generate_edge_runtime(model_id: str, req: EdgeRuntimeExportRequest) -> EdgeRuntimeExportResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.edge_runtime_codegen import generate_edge_runtime_project

    result = generate_edge_runtime_project(
        model_dir=loaded.model_dir,
        optimized_dir=req.optimized_dir,
    )
    return EdgeRuntimeExportResponse(**result)


def _scaffold_export_disabled() -> None:
    raise HTTPException(
        status_code=410,
        detail=(
            "iOS scaffold export is not included in the public edgestudio "
            "package. Use GGUF, CoreML, or EdgeRuntime Swift export in this release."
        ),
    )


@router.post("/{model_id}/export/scaffold", response_model=CreateTaskResponse)
def export_scaffold(model_id: str, req: ScaffoldExportRequest) -> CreateTaskResponse:
    """iOS scaffold export is disabled in the public pip release."""
    _scaffold_export_disabled()


@router.post("/{model_id}/export/scaffold-zip", response_model=CreateTaskResponse)
def export_scaffold_zip(model_id: str, req: ScaffoldZipExportRequest) -> CreateTaskResponse:
    """Export a self-contained EdgeScaffold App ZIP."""
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.scaffold_zip_export import export_scaffold_zip as do_export

        result = do_export(
            model_dir=loaded.model_dir,
            app_name=req.app_name or "MyApp",
            system_prompt=req.system_prompt or "You are a helpful assistant.",
            model_tier=req.model_tier or "",
            enable_dsr=req.enable_dsr,
            dsr_budget=req.dsr_budget,
            bundle_id=req.bundle_id,
            team_id=req.team_id,
            direction_set_id=req.direction_set_id,
            progress_callback=progress_callback,
        )
        return {
            "success": result.success,
            "zip_path": result.zip_path,
            "zip_size_bytes": result.zip_size_bytes,
            "app_name": result.app_name,
            "model_name": result.model_name,
            "model_dir": result.model_dir,
            "model_tier": result.model_tier,
            "direction_set_id": result.direction_set_id or req.direction_set_id or "finance_consumer",
            "error": result.error,
        }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.get("/export/scaffold-zip/download")
def download_scaffold_zip(path: str = Query(..., description="Path to the ZIP file")):
    """Download a generated EdgeScaffold App ZIP."""
    if not os.path.isfile(path) or not path.endswith(".zip"):
        raise HTTPException(404, "ZIP not found")
    return FileResponse(
        path,
        media_type="application/zip",
        filename=os.path.basename(path),
    )
