# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Schemas for recommendation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendModelsRequest(BaseModel):
    device_name: str = Field("", description="Device name from device profiles. Empty = auto-detect.")
    use_case: str = Field("chat", description="Use case: chat, coding, reasoning, translation, multimodal")
    max_results: int = Field(50, ge=1, le=200)
    tts_variant: str = Field("", description="Optional TTS variant filter: customvoice | voicedesign")


class ModelRecommendationSchema(BaseModel):
    name: str
    description: str
    estimated_size_gb: float
    fits_device: bool
    headroom_gb: float
    quality_tier: str
    download_hint: str


class RecommendOptimizationRequest(BaseModel):
    model_size_gb: float = Field(..., gt=0)
    device_name: str = Field(..., description="Target device name from device profiles")
    current_bits: int = Field(0, ge=0, le=32, description="Current quantization bits (0 = unknown/float)")


class OptimizationRecommendationSchema(BaseModel):
    strategy_name: str
    description: str
    estimated_final_size_gb: float
    fits_device: bool
    steps: list[str]
    risk_level: str
    quality_impact: str


class CatalogStatusSchema(BaseModel):
    enabled: bool
    source: str
    version: str
    generated_at: str | None = None
    total_models: int
    cache_path: str
    cache_exists: bool
    active_path: str
    refreshing: bool
    stale: bool
    last_error: str | None = None
    runtime_only_download_hints: list[str] = []


class CatalogRefreshResponseSchema(CatalogStatusSchema):
    started: bool
