# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Recommendation API — model and optimization recommendations for wizard."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.recommend import (
    RecommendModelsRequest,
    ModelRecommendationSchema,
    RecommendOptimizationRequest,
    OptimizationRecommendationSchema,
    CatalogStatusSchema,
    CatalogRefreshResponseSchema,
)
from backend.core.device_profiles import get_device, all_devices
from backend.core.recommendation_engine import (
    recommend_models,
    recommend_optimization,
    USE_CASE_CATEGORIES,
)
from backend.core.model_catalog_sync import (
    get_catalog_status,
    refresh_catalog_background,
)
from backend.core.auto_tune import _detect_device_name
from backend.api.system_info import _match_device_profile

router = APIRouter(prefix="/api/recommend")


def _detect_current_ram_gb() -> float:
    import platform, subprocess

    if platform.system() != "Darwin":
        return 0.0
    try:
        r = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, timeout=5, capture_output=True)
        return round(int(r.stdout.strip()) / (1024**3), 1)
    except Exception:
        return 0.0


def _resolve_device(device_name: str):
    """Resolve device name to DeviceProfile, auto-detecting if empty."""
    # Auto-detect chip + RAM, use same matching as system-info endpoint.
    # The Simple wizard passes the detected chip string (e.g. "Apple M4 Pro"),
    # while the catalog API historically accepted exact profile names. Support
    # both so hardware filtering stays aligned with the device profile page.
    ram_gb = _detect_current_ram_gb()

    if not device_name:
        chip = _detect_device_name()
        matched = _match_device_profile(chip, ram_gb)
        if matched:
            return matched
        # Fallback to a reasonable default
        return get_device("MacBook Air M5 (16GB)")

    device = get_device(device_name)
    if device:
        return device

    matched = _match_device_profile(device_name, ram_gb)
    if matched:
        return matched

    raise HTTPException(status_code=404, detail=f"Unknown device: {device_name}")


@router.post("/models", response_model=list[ModelRecommendationSchema])
def recommend_models_endpoint(req: RecommendModelsRequest) -> list[ModelRecommendationSchema]:
    """Recommend models for a device and use case."""
    device = _resolve_device(req.device_name)

    if req.use_case not in USE_CASE_CATEGORIES and req.use_case != "all":
        raise HTTPException(status_code=400, detail=f"Unknown use case: {req.use_case}")

    recs = recommend_models(
        device,
        use_case=req.use_case,
        max_results=req.max_results,
        tts_variant=req.tts_variant,
    )

    return [
        ModelRecommendationSchema(
            name=r.name,
            description=r.description,
            estimated_size_gb=r.estimated_size_gb,
            fits_device=r.fits_device,
            headroom_gb=round(r.headroom_gb, 1),
            quality_tier=r.quality_tier,
            download_hint=r.download_hint,
        )
        for r in recs
    ]


@router.post("/optimization", response_model=OptimizationRecommendationSchema)
def recommend_optimization_endpoint(req: RecommendOptimizationRequest) -> OptimizationRecommendationSchema:
    """Recommend optimization strategy for a model + device."""
    device = _resolve_device(req.device_name)

    rec = recommend_optimization(req.model_size_gb, device, req.current_bits)

    return OptimizationRecommendationSchema(
        strategy_name=rec.strategy_name,
        description=rec.description,
        estimated_final_size_gb=round(rec.estimated_final_size_gb, 1),
        fits_device=rec.fits_device,
        steps=rec.steps,
        risk_level=rec.risk_level,
        quality_impact=rec.quality_impact,
    )


@router.get("/use-cases", response_model=dict[str, str])
def list_use_cases() -> dict[str, str]:
    """Return available use cases for model recommendation."""
    return {
        "chat": "General chat and Q&A",
        "coding": "Code generation and debugging",
        "reasoning": "Complex reasoning and analysis",
        "translation": "Translation and multilingual tasks",
        "multimodal": "Image understanding + text",
        "asr": "Speech recognition",
        "tts": "Text to speech",
    }


@router.get("/catalog-status", response_model=CatalogStatusSchema)
def model_catalog_status() -> CatalogStatusSchema:
    """Return active model catalog freshness and refresh status."""
    return CatalogStatusSchema(**get_catalog_status())


@router.post("/catalog-refresh", response_model=CatalogRefreshResponseSchema)
def trigger_model_catalog_refresh() -> CatalogRefreshResponseSchema:
    """Refresh mlx-community model metadata in the background."""
    started = refresh_catalog_background(force=False)
    status = get_catalog_status()
    return CatalogRefreshResponseSchema(**status, started=started)
