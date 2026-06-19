# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Recommendation engine — device-aware model and optimization recommendations.

Loads model catalog from JSON, filters by device capability and use case,
ranks by relevance + quality tier + benchmark scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .device_profiles import DeviceProfile, DEVICE_PROFILES
from .model_filters import matches_tts_variant

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

QUALITY_TIER_RANK = {"premium": 0, "high": 1, "balanced": 2, "entry": 3}

# Map use_case to relevant model categories
USE_CASE_CATEGORIES: dict[str, set[str]] = {
    "chat": {"llm", "vlm"},
    "coding": {"llm"},
    "reasoning": {"llm"},
    "translation": {"llm"},
    "multimodal": {"vlm"},
    "asr": {"asr"},
    "tts": {"tts"},
}


@dataclass
class ModelRecommendation:
    """A recommended model for a specific device and use case."""
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
    benchmarks: dict[str, float] | None = None


@dataclass
class OptimizationRecommendation:
    """Recommended optimization strategy for a model + device combo."""
    strategy_name: str
    description: str
    estimated_final_size_gb: float
    fits_device: bool
    steps: list[str]
    risk_level: str
    quality_impact: str


# ---------------------------------------------------------------------------
# Catalog loader
# ---------------------------------------------------------------------------

_catalog_cache: list[dict[str, Any]] | None = None
_catalog_signature: str | None = None


def _load_catalog() -> list[dict[str, Any]]:
    """Load the freshest available model catalog.

    The bundled JSON is the offline fallback.  A runtime cache under
    the platform cache catalog can supersede it after background refresh.
    """
    global _catalog_cache, _catalog_signature
    from .model_catalog_sync import load_effective_catalog

    data, _source, signature = load_effective_catalog(start_background_refresh=False)
    if _catalog_cache is not None and _catalog_signature == signature:
        return _catalog_cache
    _catalog_cache = data["models"]
    _catalog_signature = signature
    return _catalog_cache


def reload_catalog() -> None:
    """Force reload catalog (e.g. after editing JSON)."""
    global _catalog_cache, _catalog_signature
    _catalog_cache = None
    _catalog_signature = None
    _load_catalog()


def get_catalog() -> list[dict[str, Any]]:
    """Return the full model catalog."""
    return _load_catalog()


# ---------------------------------------------------------------------------
# Recommendation functions
# ---------------------------------------------------------------------------

def recommend_models(
    device: DeviceProfile,
    use_case: str = "chat",
    max_results: int = 8,
    category_filter: str | None = None,
    tts_variant: str = "",
) -> list[ModelRecommendation]:
    """Recommend models for a device + use case.

    Scoring:
    1. Filter by category relevance (LLM for coding, VLM for multimodal, etc.)
    2. Filter by MLX availability (only recommend models we can actually run)
    3. Score = relevance (use_case match) + quality_tier + size efficiency
    4. Sort: fits_device first, then by composite score
    """
    catalog = _load_catalog()
    max_size = device.max_model_size_gb

    # Determine which categories are relevant
    relevant_cats = USE_CASE_CATEGORIES.get(use_case, {"llm"})
    if category_filter:
        relevant_cats = {category_filter}

    scored: list[tuple[float, ModelRecommendation]] = []

    for m in catalog:
        cat = m.get("category", "llm")
        # Skip irrelevant categories
        if cat not in relevant_cats:
            continue
        # Skip non-MLX models (we can't run them in Edge Studio)
        if not m.get("mlx", False):
            continue
        if not matches_tts_variant(m, tts_variant):
            continue

        size = m["size_gb"]
        fits = size <= max_size
        headroom = max_size - size
        strengths = m.get("strengths", [])

        # --- Relevance score (lower = better) ---
        if use_case in strengths:
            relevance = strengths.index(use_case)  # 0 = primary strength
        else:
            relevance = 10  # not a declared strength

        # --- Quality tier score ---
        tier_score = QUALITY_TIER_RANK.get(m.get("quality_tier", "balanced"), 2)

        # --- Size efficiency: prefer the biggest model that fits ---
        # Normalize to 0-1 range where bigger (relative to device) = better
        size_score = 1.0 - (size / max(max_size, 0.1)) if fits else 2.0

        # --- Composite score (lower = better) ---
        # Weights: relevance matters most, then quality, then size efficiency
        score = relevance * 3.0 + tier_score * 2.0 + size_score

        rec = ModelRecommendation(
            name=m["name"],
            description=m.get("description", ""),
            estimated_size_gb=size,
            fits_device=fits,
            headroom_gb=max(0, headroom),
            quality_tier=m.get("quality_tier", "balanced"),
            download_hint=m.get("download_hint", ""),
            category=cat,
            family=m.get("family", ""),
            params_b=m.get("params_b", 0),
            context_k=m.get("context_k", 0),
            benchmarks=m.get("benchmarks"),
        )

        scored.append((score, rec))

    # Sort: fitting models first, then by score (lower = better)
    scored.sort(key=lambda pair: (not pair[1].fits_device, pair[0]))

    return [rec for _, rec in scored[:max_results]]


def recommend_optimization(
    model_size_gb: float,
    device: DeviceProfile,
    current_bits: int = 0,
) -> OptimizationRecommendation:
    """Recommend the best optimization strategy to fit a model on a device."""
    target_size = device.max_model_size_gb
    ratio_needed = model_size_gb / target_size if target_size > 0 else float("inf")

    if model_size_gb <= target_size:
        return OptimizationRecommendation(
            strategy_name="No optimization needed",
            description=f"Model already fits on {device.name} ({model_size_gb:.1f} GB < {target_size:.1f} GB limit).",
            estimated_final_size_gb=model_size_gb,
            fits_device=True,
            steps=["Model is ready to deploy."],
            risk_level="low",
            quality_impact="None",
        )

    steps = []
    est_size = model_size_gb
    risk = "low"
    quality_impact_parts = []

    # Strategy 1: Quantize to lower bits
    if current_bits == 0 or current_bits > 4:
        target_bits = 4
        if ratio_needed > 2:
            target_bits = 3
        reduction = est_size * (1 - target_bits / max(current_bits if current_bits > 0 else 16, target_bits))
        est_size -= reduction
        steps.append(f"Quantize to {target_bits}-bit (saves ~{reduction:.1f} GB)")
        quality_impact_parts.append(f"{target_bits}-bit quantization: minimal impact on most tasks")
        if target_bits <= 3:
            risk = "medium"
            quality_impact_parts[-1] = f"{target_bits}-bit quantization: may reduce accuracy on math/reasoning"

    # Strategy 2: Neuron pruning
    if est_size > target_size:
        prune_ratio = min(0.3, (est_size - target_size) / est_size)
        reduction = est_size * prune_ratio * 0.5
        est_size -= reduction
        steps.append(f"Prune ~{prune_ratio:.0%} of inactive neurons (saves ~{reduction:.1f} GB)")
        quality_impact_parts.append(f"Neuron pruning at {prune_ratio:.0%}: low impact if guided by activation data")
        risk = "medium"

    # Strategy 3: Layer removal (last resort)
    if est_size > target_size:
        layers_to_remove = max(1, int((est_size - target_size) / (est_size / 32)))
        reduction = est_size * layers_to_remove / 32
        est_size -= reduction
        steps.append(f"Remove {layers_to_remove} least-important layer(s) (saves ~{reduction:.1f} GB)")
        quality_impact_parts.append("Layer removal: moderate impact, validate with perplexity check")
        risk = "high"

    fits = est_size <= target_size

    return OptimizationRecommendation(
        strategy_name="Multi-step optimization",
        description=f"Reduce {model_size_gb:.1f} GB to ~{est_size:.1f} GB for {device.name}.",
        estimated_final_size_gb=est_size,
        fits_device=fits,
        steps=steps,
        risk_level=risk,
        quality_impact="; ".join(quality_impact_parts),
    )


def get_device_recommendations() -> dict[str, list[ModelRecommendation]]:
    """Get model recommendations for all devices."""
    result = {}
    for name, device in DEVICE_PROFILES.items():
        result[name] = recommend_models(device)
    return result
