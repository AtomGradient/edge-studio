# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Export-related schemas."""

from __future__ import annotations

from pydantic import BaseModel


class GGUFExportRequest(BaseModel):
    quant_type: str = "q4_k_m"
    output_path: str | None = None


class CoreMLExportRequest(BaseModel):
    compute_units: str = "ALL"
    max_seq_length: int = 512


class SwiftCodeRequest(BaseModel):
    package_name: str = "LLMModel"
    default_max_tokens: int = 256


class ExportResultResponse(BaseModel):
    success: bool
    output_path: str | None = None
    size_bytes: int | None = None
    duration_ms: float | None = None
    error: str | None = None


class SwiftCodeResponse(BaseModel):
    code: str
    filename: str


class EdgeRuntimeExportRequest(BaseModel):
    optimized_dir: str = ""


class EdgeRuntimeExportResponse(BaseModel):
    package_swift: str
    main_swift: str
    readme: str
    run_command: str
    model_name: str
    is_optimized: bool
    optimization_summary: str


class ScaffoldExportRequest(BaseModel):
    odr_tag: str = "model-custom"
    app_name: str = ""
    system_prompt: str = ""
    direction_set_id: str | None = None


class ScaffoldZipExportRequest(BaseModel):
    app_name: str = "MyApp"
    system_prompt: str = "You are a helpful assistant."
    model_tier: str = ""  # empty = auto-detect
    enable_dsr: bool = True  # DSR intelligent cache retention
    dsr_budget: int | None = None  # None = auto (runtime decides)
    bundle_id: str | None = None
    team_id: str | None = None
    direction_set_id: str | None = None
