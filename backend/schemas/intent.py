# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Schemas for intent-driven semantic search."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IntentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       description="Natural language query describing the desired use case")
    device_name: str = Field("", description="Target device. Empty = auto-detect.")
    max_results: int = Field(50, ge=1, le=200)
    tts_variant: str = Field("", description="Optional TTS variant filter: customvoice | voicedesign")


class IntentSearchResultSchema(BaseModel):
    name: str
    description: str
    estimated_size_gb: float
    fits_device: bool
    headroom_gb: float
    quality_tier: str
    download_hint: str
    category: str = "llm"
    family: str = ""
    params_b: float = 0
    context_k: int = 0
    semantic_score: float = 0.0


class IntentSearchResponse(BaseModel):
    """Wrapper response for intent search — includes results + detected device context."""
    results: list[IntentSearchResultSchema]
    detected_device: str | None = Field(None, description="Device detected from query (e.g. 'on phone' → 'iPhone 15 Pro')")
    detected_max_size_gb: float | None = Field(None, description="Max model size for detected device")


class EmbeddingStatusSchema(BaseModel):
    ready: bool
    model_repo: str | None = None
    region: str
    catalog_version: str | None = None
    dependency_ready: bool = True
    downloading: bool = False
    task_id: str | None = None
