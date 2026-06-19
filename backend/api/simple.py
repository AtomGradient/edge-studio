# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Simple mode API — device profile, package selection, setup orchestration, export check.

Provides the backend for the "2 clicks + auto download" Simple mode experience.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.simple import (
    DeviceProfileResponse,
    ExportCheckRequest,
    ExportCheckResponse,
    Package,
    PackageModel,
    PackagesRequest,
    PackagesResponse,
    SetupRequest,
    SetupResponse,
)

router = APIRouter(prefix="/api/simple", tags=["simple"])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUALITY_TIER_RANK = {"premium": 0, "high": 1, "balanced": 2, "entry": 3}

# Focus → catalog categories
FOCUS_CATEGORIES: dict[str, list[str]] = {
    "chat": ["llm"],
    "coding": ["llm"],
    "vision": ["vlm"],
    "asr": ["asr"],
    "tts": ["tts"],
    "voice_duplex": ["llm"],   # primary model is LLM
    "voice": ["asr"],          # backward-compat alias → asr
}

# Preferred model families per focus (for LLM/VLM primary selection)
PREFERRED_FAMILIES: dict[str, list[str]] = {
    "chat": ["Qwen3.5", "Gemma4"],
    "coding": ["Qwen3.5", "Gemma4"],
    "vision": ["Qwen3.5", "Gemma4"],
    "asr": ["Qwen3"],
    "tts": ["Qwen3"],
    "voice_duplex": ["Qwen3.5", "Gemma4"],
}

# Preferred families when selecting via category_override (audio models in duplex)
PREFERRED_FAMILIES_AUDIO: dict[str, list[str]] = {
    "asr": ["Qwen3"],
    "tts": ["Qwen3"],
}

# Tier → (min_params_b, max_params_b) — left inclusive, right exclusive
TIER_PARAMS_LLM: dict[str, tuple[float, float]] = {
    "standard": (0, 2.0),
    "pro": (2.0, 9.0),
    "max": (9.0, 35.0),
    "ultra": (35.0, float("inf")),
}

# Audio models have a completely different size distribution:
#   47 ASR models: 44 < 2B, 3 in 2-9B, 0 in 9B+
#   41 TTS models: 35 < 2B, 6 in 2-9B, 0 in 9B+
TIER_PARAMS_AUDIO: dict[str, tuple[float, float]] = {
    "standard": (0, 0.5),
    "pro": (0.5, 2.0),
    "max": (2.0, float("inf")),
}

# Quick lookup: category → which tier params to use
TIER_PARAMS_BY_CATEGORY: dict[str, dict[str, tuple[float, float]]] = {
    "llm": TIER_PARAMS_LLM,
    "vlm": TIER_PARAMS_LLM,
    "asr": TIER_PARAMS_AUDIO,
    "tts": TIER_PARAMS_AUDIO,
}

# Memory headroom reserved for runtime overhead in duplex mode
DUPLEX_HEADROOM_GB = 0.5

# Tier display info
TIER_INFO: dict[str, dict[str, Any]] = {
    "standard": {
        "label": "Standard",
        "capabilities_chat": [
            "simple.tier.standard.cap1",
            "simple.tier.standard.cap2",
        ],
        "capabilities_coding": [
            "simple.tier.standard.cap1",
            "simple.tier.standard.cap2",
        ],
        "capabilities_vision": [
            "simple.tier.standard.capVision1",
            "simple.tier.standard.capVision2",
        ],
        "capabilities_asr": [
            "simple.tier.standard.capASR1",
        ],
        "capabilities_tts": [
            "simple.tier.standard.capTTS1",
        ],
        "capabilities_voice_duplex": [
            "simple.tier.standard.capDuplex1",
        ],
    },
    "pro": {
        "label": "Pro",
        "capabilities_chat": [
            "simple.tier.pro.cap1",
            "simple.tier.pro.cap2",
            "simple.tier.pro.cap3",
        ],
        "capabilities_coding": [
            "simple.tier.pro.capCode1",
            "simple.tier.pro.capCode2",
            "simple.tier.pro.capCode3",
        ],
        "capabilities_vision": [
            "simple.tier.pro.capVision1",
            "simple.tier.pro.capVision2",
        ],
        "capabilities_asr": [
            "simple.tier.pro.capASR1",
            "simple.tier.pro.capASR2",
        ],
        "capabilities_tts": [
            "simple.tier.pro.capTTS1",
            "simple.tier.pro.capTTS2",
        ],
        "capabilities_voice_duplex": [
            "simple.tier.pro.capDuplex1",
            "simple.tier.pro.capDuplex2",
        ],
    },
    "max": {
        "label": "Max",
        "capabilities_chat": [
            "simple.tier.max.cap1",
            "simple.tier.max.cap2",
            "simple.tier.max.cap3",
        ],
        "capabilities_coding": [
            "simple.tier.max.capCode1",
            "simple.tier.max.capCode2",
            "simple.tier.max.capCode3",
        ],
        "capabilities_vision": [
            "simple.tier.max.capVision1",
            "simple.tier.max.capVision2",
        ],
        "capabilities_asr": [
            "simple.tier.max.capASR1",
            "simple.tier.max.capASR2",
        ],
        "capabilities_tts": [
            "simple.tier.max.capTTS1",
            "simple.tier.max.capTTS2",
        ],
        "capabilities_voice_duplex": [
            "simple.tier.max.capDuplex1",
            "simple.tier.max.capDuplex2",
        ],
    },
    "ultra": {
        "label": "Ultra",
        "capabilities_chat": [
            "simple.tier.ultra.cap1",
            "simple.tier.ultra.cap2",
            "simple.tier.ultra.cap3",
        ],
        "capabilities_coding": [
            "simple.tier.ultra.capCode1",
            "simple.tier.ultra.capCode2",
            "simple.tier.ultra.capCode3",
        ],
        "capabilities_vision": [
            "simple.tier.ultra.capVision1",
            "simple.tier.ultra.capVision2",
        ],
        # No audio capabilities at Ultra tier
        "capabilities_asr": [],
        "capabilities_tts": [],
        "capabilities_voice_duplex": [],
    },
}

# AI rating tiers
AI_RATINGS = [
    (8, "air", "simple.rating.air", 1),
    (24, "standard", "simple.rating.standard", 2),
    (128, "pro", "simple.rating.pro", 3),
    (256, "max", "simple.rating.max", 4),
    (float("inf"), "ultra", "simple.rating.ultra", 5),
]

# iOS export device capacity (Jetsam-aware)
IOS_DEVICE_CAPACITY: dict[str, float] = {
    "iphone": 3.0,    # conservative: 8GB * 0.55 * 0.85 ≈ 3.7, use 3.0
    "ipad": 6.0,      # conservative: 8-16GB range, use 6.0 baseline
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_system_info() -> dict[str, Any]:
    """Fetch system info using the existing system_info module."""
    from backend.api.system_info import get_system_info
    return get_system_info()


def _load_catalog() -> list[dict[str, Any]]:
    """Load model catalog (cached)."""
    from backend.core.recommendation_engine import get_catalog
    return get_catalog()


def _device_rating(ram_gb: float) -> tuple[str, str, int]:
    """Return (rating_key, label_i18n_key, stars) for a given RAM size."""
    for threshold, key, label, stars in AI_RATINGS:
        if ram_gb <= threshold:
            return key, label, stars
    return "ultra", "simple.rating.ultra", 5


def _recommend_tier(ram_gb: float) -> str:
    """Return the recommended tier based on device RAM."""
    if ram_gb <= 8:
        return "standard"
    elif ram_gb <= 24:
        return "pro"
    elif ram_gb <= 128:
        return "max"
    else:
        return "ultra"


def _available_tiers(ram_gb: float, focus: str = "") -> list[str]:
    """Return which tiers are usable on this device for a given focus."""
    max_size = ram_gb * 0.85
    catalog = _load_catalog()

    # Determine which tier params and category to check
    primary_category = FOCUS_CATEGORIES.get(focus, ["llm"])[0]
    tier_params = TIER_PARAMS_BY_CATEGORY.get(primary_category, TIER_PARAMS_LLM)

    tiers = []
    for tier_name in tier_params:
        min_p, max_p = tier_params[tier_name]
        has_model = any(
            m.get("mlx", False)
            and m.get("category") == primary_category
            and min_p <= m.get("params_b", 0) < max_p
            and m.get("size_gb", 999) <= max_size
            for m in catalog
        )
        if has_model:
            tiers.append(tier_name)

    return tiers if tiers else ["standard"]


def _estimate_setup_time(size_gb: float, network_speed_mbps: float = 50) -> str:
    """Estimate download + load time. Returns an i18n key with interpolation."""
    download_s = (size_gb * 1024) / (network_speed_mbps / 8)
    load_s = size_gb * 3
    total_min = (download_s + load_s) / 60

    if total_min < 2:
        return "simple.time.instant"
    elif total_min < 30:
        return f"simple.time.minutes:{round(total_min)}"
    else:
        return f"simple.time.long:{round(total_min)}"


def _select_model(
    focus: str,
    tier: str,
    max_size_gb: float,
    category_override: str | None = None,
    name_filter: str | None = None,
) -> dict[str, Any] | None:
    """Select the best model for focus + tier + device capacity.

    Algorithm: determine tier params from the effective category (audio vs LLM),
    filter by tier param range & size, then sort by
    preferred family → quality tier → larger params → 4bit preferred.
    """
    catalog = _load_catalog()

    # Choose tier params and families based on the actual category being searched
    effective_category = category_override or FOCUS_CATEGORIES.get(focus, ["llm"])[0]
    tier_params = TIER_PARAMS_BY_CATEGORY.get(effective_category, TIER_PARAMS_LLM)
    min_p, max_p = tier_params.get(tier, (0, float("inf")))

    categories = [category_override] if category_override else FOCUS_CATEGORIES.get(focus, ["llm"])

    if category_override:
        families = PREFERRED_FAMILIES_AUDIO.get(category_override, [])
    else:
        families = PREFERRED_FAMILIES.get(focus, [])

    candidates = [
        m for m in catalog
        if m.get("mlx", False)
        and m.get("category") in categories
        and min_p <= m.get("params_b", 0) < max_p
        and m.get("size_gb", 999) <= max_size_gb
    ]

    if name_filter:
        candidates = [m for m in candidates if name_filter.lower() in m.get("download_hint", "").lower()]

    if not candidates:
        return None

    def sort_key(m: dict) -> tuple:
        fam = m.get("family", "")
        family_rank = families.index(fam) if fam in families else 99
        tier_rank = QUALITY_TIER_RANK.get(m.get("quality_tier", "balanced"), 99)
        quant = m.get("quant", "")
        quant_rank = 0 if quant == "4bit" else (1 if quant == "8bit" else 2)
        return (family_rank, tier_rank, -m.get("params_b", 0), quant_rank, m.get("size_gb", 0))

    candidates.sort(key=sort_key)
    return candidates[0]


def _build_duplex_package(
    tier: str,
    max_size: float,
    tts_variant: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Select the three models for Voice Duplex mode.

    Strategy: ASR/TTS fixed to "pro" Audio tier (best small model),
    LLM uses remaining memory budget at user-selected tier.

    Returns (llm, tts, asr) — any may be None if no model fits.
    """
    # 1. ASR: pick best small model (Audio "pro" = 0.5-2B)
    asr = _select_model("voice_duplex", "pro", max_size, category_override="asr")
    if not asr:
        asr = _select_model("voice_duplex", "standard", max_size, category_override="asr")

    # 2. TTS: same strategy, optionally filtered by variant name
    tts_filter = tts_variant if tts_variant else None
    tts = _select_model("voice_duplex", "pro", max_size, category_override="tts", name_filter=tts_filter)
    if not tts:
        tts = _select_model("voice_duplex", "standard", max_size, category_override="tts", name_filter=tts_filter)

    # 3. LLM: budget = total - ASR - TTS - headroom
    asr_size = asr["size_gb"] if asr else 0
    tts_size = tts["size_gb"] if tts else 0
    llm_budget = max_size - asr_size - tts_size - DUPLEX_HEADROOM_GB

    if llm_budget <= 0:
        return None, tts, asr

    llm = _select_model("voice_duplex", tier, llm_budget)
    return llm, tts, asr


def _model_to_package_model(m: dict[str, Any]) -> PackageModel:
    """Convert a catalog entry to PackageModel schema."""
    family = str(m.get("family", "")).strip()
    name = str(m.get("name", "")).strip()
    if family and not name.lower().startswith(family.lower()):
        display_name = f"{family} {name}".strip()
    else:
        display_name = name or family

    return PackageModel(
        catalog_id=m.get("id", ""),
        display_name=display_name,
        family=family,
        category=m.get("category", "llm"),
        params_b=m.get("params_b", 0),
        size_gb=m.get("size_gb", 0),
        quant=m.get("quant", ""),
        download_hint=m.get("download_hint", ""),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/device-profile", response_model=DeviceProfileResponse)
def get_device_profile():
    """Detect hardware and return AI capability profile + rating."""
    info = _get_system_info()

    ram_gb = info.get("ram_gb") or info.get("total_memory_gb", 8)
    chip = info.get("chip", "Unknown")
    gpu_cores = info.get("gpu_cores", 0)
    max_model = round(ram_gb * 0.85, 1)

    rating_key, rating_label, stars = _device_rating(ram_gb)
    rec_tier = _recommend_tier(ram_gb)
    tiers = _available_tiers(ram_gb)

    return DeviceProfileResponse(
        chip=chip,
        ram_gb=ram_gb,
        gpu_cores=gpu_cores,
        max_model_size_gb=max_model,
        ai_rating=rating_key,
        ai_rating_label=rating_label,
        ai_rating_stars=stars,
        available_tiers=tiers,
        recommended_tier=rec_tier,
    )


@router.post("/packages", response_model=PackagesResponse)
def get_packages(req: PackagesRequest):
    """Return tier packages for a focus, ranked and filtered by device capability."""
    if req.ram_gb > 0:
        ram_gb = req.ram_gb
    else:
        info = _get_system_info()
        ram_gb = info.get("ram_gb") or info.get("total_memory_gb", 8)
    max_size = ram_gb * 0.85

    focus = req.focus
    if focus not in FOCUS_CATEGORIES:
        raise HTTPException(400, f"Invalid focus: {focus}. Must be one of {list(FOCUS_CATEGORIES.keys())}")

    rec_tier = _recommend_tier(ram_gb)
    packages: list[Package] = []

    # Determine tier list based on focus category
    primary_category = FOCUS_CATEGORIES[focus][0]
    if focus == "voice_duplex":
        # Duplex uses LLM tiers but no Ultra (ASR+TTS overhead)
        tier_names = ["standard", "pro", "max"]
    elif primary_category in ("asr", "tts"):
        # Audio models: only standard/pro/max
        tier_names = list(TIER_PARAMS_AUDIO.keys())
    else:
        # LLM/VLM: all four tiers
        tier_names = list(TIER_PARAMS_LLM.keys())

    for tier_name in tier_names:
        tier_meta = TIER_INFO[tier_name]
        cap_key = f"capabilities_{focus}"
        capabilities = tier_meta.get(cap_key, tier_meta.get("capabilities_chat", []))

        if focus == "voice_duplex":
            # === Duplex: three-model package with memory budget ===
            llm, tts, asr = _build_duplex_package(tier_name, max_size, tts_variant=req.tts_variant)

            if llm:
                total_size = llm["size_gb"]
                total_size += tts["size_gb"] if tts else 0
                total_size += asr["size_gb"] if asr else 0
                packages.append(Package(
                    tier=tier_name,
                    tier_label=tier_meta["label"],
                    available=True,
                    download_size_gb=round(total_size, 1),
                    setup_time_hint=_estimate_setup_time(total_size),
                    capabilities=capabilities,
                    model=_model_to_package_model(llm),
                    secondary_model=_model_to_package_model(tts) if tts else None,
                    tertiary_model=_model_to_package_model(asr) if asr else None,
                ))
            else:
                packages.append(Package(
                    tier=tier_name,
                    tier_label=tier_meta["label"],
                    available=False,
                    unavailable_reason="simple.tier.unavailable.tooLarge",
                    download_size_gb=0,
                    setup_time_hint="",
                    capabilities=capabilities,
                ))

        else:
            # === Single-model: LLM, VLM, ASR, or TTS ===
            primary = _select_model(focus, tier_name, max_size)
            available = primary is not None
            packages.append(Package(
                tier=tier_name,
                tier_label=tier_meta["label"],
                available=available,
                unavailable_reason="" if available else "simple.tier.unavailable.tooLarge",
                download_size_gb=round(primary["size_gb"], 1) if primary else 0,
                setup_time_hint=_estimate_setup_time(primary["size_gb"]) if primary else "",
                capabilities=capabilities,
                model=_model_to_package_model(primary) if primary else None,
            ))

    return PackagesResponse(
        packages=packages,
        recommended_tier=rec_tier,
    )


@router.post("/setup", response_model=SetupResponse)
def setup_model(req: SetupRequest):
    """Resolve model for download + load. Frontend orchestrates via existing endpoints.

    Returns model info so the frontend can:
    1. POST /api/hf/download with download_hint
    2. POST /api/model/load with model_dir
    """
    info = _get_system_info()
    ram_gb = info.get("ram_gb") or info.get("total_memory_gb", 8)
    max_size = ram_gb * 0.85

    # Resolve which model to use
    if req.custom_model_id:
        catalog = _load_catalog()
        cid = req.custom_model_id
        # Match by catalog id OR download_hint (browse panel passes download_hint)
        model_entry = next(
            (m for m in catalog if m["id"] == cid or m.get("download_hint") == cid),
            None,
        )
        if not model_entry:
            from backend.core.model_catalog_sync import fetch_remote_model_search
            query = cid.split("/")[-1]
            try:
                remote = fetch_remote_model_search(query, timeout=8.0, limit=20)
                model_entry = next(
                    (
                        m for m in remote.get("models", [])
                        if m.get("id") == cid or m.get("download_hint") == cid
                    ),
                    None,
                )
            except Exception as exc:
                logger.info("Remote custom model resolve failed for %s: %s", cid, exc)
        if not model_entry:
            raise HTTPException(404, f"Model '{cid}' not found in catalog.")
    else:
        focus = req.focus
        tier = req.tier
        if focus not in FOCUS_CATEGORIES:
            raise HTTPException(400, f"Invalid focus: {focus}")
        # Validate tier against the appropriate tier params for this focus
        primary_category = FOCUS_CATEGORIES[focus][0]
        valid_tiers = TIER_PARAMS_BY_CATEGORY.get(primary_category, TIER_PARAMS_LLM)
        if tier not in valid_tiers:
            raise HTTPException(400, f"Invalid tier: {tier} for focus: {focus}")

        model_entry = _select_model(focus, tier, max_size)
        if not model_entry:
            raise HTTPException(404, f"No model available for focus={focus}, tier={tier} on this device.")

    download_hint = model_entry.get("download_hint", "")
    family = str(model_entry.get("family", "")).strip()
    name = str(model_entry.get("name", "")).strip()
    display_name = name if name.lower().startswith(family.lower()) else f"{family} {name}".strip()
    size_gb = model_entry.get("size_gb", 0)

    if not download_hint:
        raise HTTPException(400, "Model has no download_hint in catalog.")

    # Check if already downloaded
    default_dir = str(Path.home() / "mlx-community")
    local_name = download_hint.replace("/", "_")
    local_dir = os.path.join(default_dir, local_name)
    already_downloaded = (
        os.path.isdir(local_dir)
        and os.path.exists(os.path.join(local_dir, "config.json"))
        and any(f.endswith(".safetensors") for f in os.listdir(local_dir))
    )

    return SetupResponse(
        task_id="",  # No background task — frontend orchestrates
        model_display_name=display_name,
        download_hint=download_hint,
        size_gb=size_gb,
        already_downloaded=already_downloaded,
        local_dir=local_dir if already_downloaded else "",
    )


@router.post("/export-check", response_model=ExportCheckResponse)
def check_export(req: ExportCheckRequest):
    """Check if the current model fits on the target iOS device for export."""
    target = req.target_device.lower()

    # Resolve target device max model size
    if target in IOS_DEVICE_CAPACITY:
        target_max_gb = IOS_DEVICE_CAPACITY[target]
    else:
        # Try to find in device profiles
        from backend.core.device_profiles import DEVICE_PROFILES
        device = DEVICE_PROFILES.get(req.target_device)
        if device:
            target_max_gb = device.max_model_size_gb
        elif "iphone" in target:
            target_max_gb = IOS_DEVICE_CAPACITY["iphone"]
        elif "ipad" in target:
            target_max_gb = IOS_DEVICE_CAPACITY["ipad"]
        else:
            target_max_gb = 3.0  # ultra conservative fallback

    # Resolve current model size
    model_size = req.current_model_size_gb
    if req.current_model_id:
        catalog = _load_catalog()
        entry = next((m for m in catalog if m["id"] == req.current_model_id), None)
        if entry:
            model_size = entry.get("size_gb", model_size)

    if model_size <= 0:
        raise HTTPException(400, "Cannot determine model size. Provide current_model_size_gb or valid current_model_id.")

    fits = model_size <= target_max_gb

    if fits:
        return ExportCheckResponse(
            fits=True,
            suggestion="direct",
            reason="simple.export.fits",
            needs_download=False,
        )

    # Suggest downgrade: find the largest tier model that fits target
    focus = req.focus if req.focus in FOCUS_CATEGORIES else "chat"
    for tier_name in ["pro", "standard"]:
        candidate = _select_model(focus, tier_name, target_max_gb)
        if candidate:
            return ExportCheckResponse(
                fits=False,
                suggestion="downgrade",
                suggested_tier=tier_name,
                reason="simple.export.tooLarge",
                needs_download=True,
                download_size_gb=candidate.get("size_gb", 0),
            )

    return ExportCheckResponse(
        fits=False,
        suggestion="change_focus",
        reason="simple.export.noFit",
        needs_download=False,
    )
