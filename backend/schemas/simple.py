# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Schemas for Simple mode API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Device Profile
# ---------------------------------------------------------------------------

class DeviceProfileResponse(BaseModel):
    """AI capability profile for the current Mac."""
    chip: str
    ram_gb: float
    gpu_cores: int
    max_model_size_gb: float
    ai_rating: str = Field(description="air | standard | pro | max | ultra")
    ai_rating_label: str = Field(description="Human-readable rating label")
    ai_rating_stars: int = Field(description="Star count 1-5")
    available_tiers: list[str]
    recommended_tier: str


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

class PackagesRequest(BaseModel):
    focus: str = Field(..., description="chat | coding | vision | asr | tts | voice_duplex")
    device_name: str = Field("", description="Device name override. Empty = auto-detect.")
    ram_gb: float = Field(0, description="RAM in GB. If >0, skip system detection.")
    tts_variant: str = Field("", description="TTS variant filter: customvoice | voicedesign")


class PackageModel(BaseModel):
    """Model info within a package."""
    catalog_id: str
    display_name: str
    family: str
    category: str
    params_b: float
    size_gb: float
    quant: str = ""
    download_hint: str


class Package(BaseModel):
    """A tier-level package for a given focus."""
    tier: str
    tier_label: str
    available: bool
    unavailable_reason: str = ""
    download_size_gb: float
    setup_time_hint: str
    capabilities: list[str]
    model: PackageModel | None = None
    secondary_model: PackageModel | None = None   # TTS in duplex mode
    tertiary_model: PackageModel | None = None     # ASR in duplex mode


class PackagesResponse(BaseModel):
    packages: list[Package]
    recommended_tier: str


# ---------------------------------------------------------------------------
# Setup (download + load orchestration)
# ---------------------------------------------------------------------------

class SetupRequest(BaseModel):
    focus: str = Field(..., description="chat | coding | vision | asr | tts | voice_duplex")
    tier: str = Field("", description="Tier name. Empty if custom_model_id is set.")
    custom_model_id: str = Field("", description="Catalog model ID for manual selection.")


class SetupResponse(BaseModel):
    task_id: str = ""
    model_display_name: str = ""
    download_hint: str = ""
    size_gb: float = 0
    already_downloaded: bool = False
    local_dir: str = ""


# ---------------------------------------------------------------------------
# Export Check
# ---------------------------------------------------------------------------

class ExportCheckRequest(BaseModel):
    target_device: str = Field(..., description="iphone | ipad or specific device name")
    focus: str = Field("chat", description="Current focus for downgrade suggestions.")
    current_model_id: str = Field("", description="Catalog model ID of current model.")
    current_model_size_gb: float = Field(0, description="Fallback size if ID not in catalog.")


class ExportCheckResponse(BaseModel):
    fits: bool
    suggestion: str = Field(description="direct | downgrade | change_focus")
    suggested_tier: str = ""
    reason: str = ""
    needs_download: bool = False
    download_size_gb: float = 0
