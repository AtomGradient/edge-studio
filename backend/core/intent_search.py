# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Intent-driven semantic search — embedding-based model recommendation.

Downloads a lightweight embedding model on first use (60-93MB), precomputes
catalog vectors, then matches user natural language queries to the best models
via cosine similarity combined with device-fit scoring.

Region-aware: mainland China users get bge-small-zh-v1.5, others get gte-small.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .model_filters import matches_tts_variant
from backend.services.app_dirs import cache_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBEDDING_HOME = cache_path("models", "embedding")

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "china": {
        "repo_id": "BAAI/bge-small-zh-v1.5",
        "size_mb": 93,
        "dim": 512,
    },
    "international": {
        "repo_id": "thenlper/gte-small",
        "size_mb": 67,
        "dim": 384,
    },
}

VECTORS_FILENAME = "catalog_vectors.npz"

# ---------------------------------------------------------------------------
# Keyword config — loaded from intent_search_keywords.json at module level.
# The JSON file is the single source of truth; editing keywords no longer
# requires a code change.
# ---------------------------------------------------------------------------

_KEYWORDS_JSON = Path(__file__).with_name("intent_search_keywords.json")


def _load_keyword_config() -> dict:
    """Load keyword/device-hint config from the co-located JSON file."""
    with open(_KEYWORDS_JSON, encoding="utf-8") as f:
        return json.load(f)


_keyword_config = _load_keyword_config()

_KEYWORD_MAP: dict[str, str] = _keyword_config["keyword_map"]

# Category expansion keywords — injected into _build_model_text() to help
# the embedding model connect short category labels to user queries.
_CATEGORY_KEYWORDS: dict[str, str] = _keyword_config["category_keywords"]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EmbeddingNotReadyError(Exception):
    """Raised when the embedding model is not yet downloaded."""
    pass


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_encoder = None  # Lazy-loaded SentenceTransformer
_encoder_dir: str | None = None


def is_embedding_dependency_ready() -> bool:
    """Return True when the sentence-transformers runtime is installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Region detection
# ---------------------------------------------------------------------------

def detect_region() -> str:
    """Detect if user is in mainland China by probing HuggingFace.

    Returns 'china' if HF is unreachable, 'international' otherwise.
    """
    try:
        import httpx
        resp = httpx.head("https://huggingface.co", timeout=3.0, follow_redirects=True)
        return "international" if resp.status_code < 500 else "china"
    except Exception:
        return "china"


# ---------------------------------------------------------------------------
# Embedding model lifecycle
# ---------------------------------------------------------------------------

def get_embedding_dir(region: str | None = None) -> Path:
    """Return the local directory for the embedding model."""
    if region is None:
        region = detect_region()
    return _embedding_dir(region, EMBEDDING_HOME)


def _embedding_dir(region: str, root: Path) -> Path:
    model_info = EMBEDDING_MODELS[region]
    repo_name = model_info["repo_id"].replace("/", "_")
    return root / repo_name


def _new_embedding_dir(region: str) -> Path:
    return _embedding_dir(region, EMBEDDING_HOME)


def _embedding_dir_ready(path: Path) -> bool:
    return path.exists() and (
        any(path.glob("*.bin"))
        or any(path.glob("*.safetensors"))
        or any(path.glob("model.safetensors"))
    )


def is_embedding_ready(region: str | None = None) -> dict[str, Any]:
    """Check embedding model status.

    Returns dict with: ready, model_repo, region, model_dir.
    """
    if region is None:
        # Check both regions — one might already be downloaded
        for r in ("china", "international"):
            d = get_embedding_dir(r)
            if _embedding_dir_ready(d):
                info = EMBEDDING_MODELS[r]
                return {"ready": True, "model_repo": info["repo_id"], "region": r, "model_dir": str(d)}
        # Neither downloaded — detect region for recommendation
        region = detect_region()

    d = get_embedding_dir(region)
    info = EMBEDDING_MODELS[region]
    ready = _embedding_dir_ready(d)
    return {
        "ready": ready,
        "model_repo": info["repo_id"] if ready else None,
        "region": region,
        "model_dir": str(d) if ready else None,
    }


def download_embedding_model(
    region: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> str:
    """Download the embedding model for the given region.

    Returns the local directory path.
    Skips download if the model is already present locally.

    Download strategy (China):
      1. ModelScope (most reliable in mainland)
      2. HuggingFace via hf-mirror
      3. local_files_only fallback (HF cache)

    Download strategy (international):
      1. HuggingFace direct
      2. local_files_only fallback
    """
    if region is None:
        region = detect_region()

    # Skip if already downloaded
    status = is_embedding_ready(region)
    if status["ready"]:
        if progress_callback:
            progress_callback("Embedding model already downloaded", 1.0)
        return status["model_dir"]

    model_info = EMBEDDING_MODELS[region]
    repo_id = model_info["repo_id"]
    local_dir = _new_embedding_dir(region)

    if progress_callback:
        progress_callback(f"Downloading {repo_id} ({model_info['size_mb']}MB)...", 0.1)

    # China: try ModelScope first (most reliable in mainland)
    if region == "china":
        try:
            from modelscope import snapshot_download as ms_download
            logger.info("Downloading embedding model from ModelScope: %s", repo_id)
            result_path = ms_download(repo_id, local_dir=str(local_dir))
            if progress_callback:
                progress_callback("Embedding model downloaded", 1.0)
            return result_path
        except ImportError:
            logger.info("modelscope not installed, trying hf-mirror")
        except Exception as e:
            logger.warning("ModelScope download failed: %s, trying hf-mirror", e)

    # HuggingFace (with hf-mirror for China)
    from huggingface_hub import snapshot_download

    old_endpoint = os.environ.get("HF_ENDPOINT")
    if region == "china":
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        result_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
        )
    except Exception as download_err:
        # Network failure — try local_files_only as fallback (HF cache may have it)
        try:
            result_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_files_only=True,
            )
            logger.info("Using cached embedding model (offline fallback)")
        except Exception:
            raise download_err  # Re-raise original network error
    finally:
        if region == "china":
            if old_endpoint is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = old_endpoint

    if progress_callback:
        progress_callback("Embedding model downloaded", 1.0)

    return result_path


# ---------------------------------------------------------------------------
# Encoder management
# ---------------------------------------------------------------------------

def _get_encoder(model_dir: str):
    """Lazily load the SentenceTransformer encoder."""
    global _encoder, _encoder_dir
    if _encoder is not None and _encoder_dir == model_dir:
        return _encoder

    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model from %s", model_dir)
    _encoder = SentenceTransformer(model_dir, device="cpu")
    _encoder_dir = model_dir
    return _encoder


# ---------------------------------------------------------------------------
# Catalog vector computation
# ---------------------------------------------------------------------------

# Query → device hint mapping: natural language device mentions → profile name
# Uses conservative estimates (8GB iPhone) so results are safe for most users.
# Loaded from intent_search_keywords.json; longer patterns first (checked in
# order, first match wins).  Names must match keys in
# device_profiles.DEVICE_PROFILES exactly.
_DEVICE_HINT_MAP: list[tuple[str, str]] = [
    (pair[0], pair[1]) for pair in _keyword_config["device_hint_map"]
]

_REMOTE_LOOKUP_CACHE: dict[str, list[dict]] = {}


def _detect_device_from_query(query_lower: str) -> str | None:
    """Extract device intent from natural language query.

    Returns a device profile name if a device hint is found, None otherwise.
    """
    for pattern, profile in _DEVICE_HINT_MAP:
        if pattern in query_lower:
            return profile
    return None


def _build_model_text(model: dict) -> str:
    """Build searchable text representation for a catalog model.

    Prefers the rich `search_text` field (generated from model properties).
    Falls back to name+category+description+strengths for models without it.
    """
    # Prefer pre-built search_text (rich, natural language, lives in data not code)
    search_text = model.get("search_text", "")
    if search_text:
        return f"{model.get('name', '')} {search_text}"

    # Fallback for models without search_text
    cat = model.get("category", "")
    parts = [
        model.get("name", ""),
        cat,
        model.get("description", ""),
        " ".join(model.get("strengths", [])),
        _CATEGORY_KEYWORDS.get(cat, ""),
    ]
    return " ".join(p for p in parts if p)


def _normalize_model_lookup(value: str) -> str:
    """Normalize model names for exact-ish user lookup.

    Users often type release names like "Qwen3.6" while catalog entries contain
    punctuation variants such as "Qwen3.6-35B-A3B-8bit".  Keep only
    alphanumerics so this path is robust without needing embeddings.
    """
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _extract_model_lookup_terms(query: str) -> list[str]:
    """Extract likely model-family tokens from free text.

    This keeps natural-language queries semantic, while letting explicit model
    names ("Qwen3.6", "llama-4", "Qwen3.6-35B") take a fast exact-lookup path.
    """
    terms: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*", query):
        norm = _normalize_model_lookup(raw)
        if len(norm) >= 4 and any(ch.isdigit() for ch in norm) and norm not in terms:
            terms.append(norm)
    return terms


def _extract_model_lookup_raw_terms(query: str) -> list[str]:
    raw_terms: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*", query):
        cleaned = raw.strip("._-")
        norm = _normalize_model_lookup(cleaned)
        if len(norm) >= 4 and any(ch.isdigit() for ch in norm):
            raw_terms.append(cleaned)
    return raw_terms


def _matches_model_lookup(query: str, model: dict) -> bool:
    terms = _extract_model_lookup_terms(query)
    if not terms:
        return False
    haystack = " ".join(
        str(model.get(key, ""))
        for key in ("name", "download_hint", "family", "id")
    )
    haystack_norm = _normalize_model_lookup(haystack)
    return any(term in haystack_norm for term in terms)


def _format_search_result(model: dict, max_size: float) -> dict:
    size = model.get("size_gb", 0)
    fits = size <= max_size
    headroom = max(0, max_size - size)
    return {
        "name": model["name"],
        "description": model.get("description", ""),
        "estimated_size_gb": size,
        "fits_device": fits,
        "headroom_gb": round(headroom, 1),
        "quality_tier": model.get("quality_tier", "balanced"),
        "download_hint": model.get("download_hint", ""),
        "category": model.get("category", "llm"),
        "family": model.get("family", ""),
        "params_b": model.get("params_b", 0),
        "context_k": model.get("context_k", 0),
        "semantic_score": 0.0,
    }


def _resolve_search_device(query_lower: str, device_name: str):
    from .device_profiles import get_device

    device_from_query = _detect_device_from_query(query_lower)
    device = get_device(device_from_query) if device_from_query else None
    if device is None and device_name:
        device = get_device(device_name)
    if device is None:
        from backend.api.system_info import _match_device_profile
        from backend.core.auto_tune import _detect_device_name
        import platform, subprocess
        chip = device_name or _detect_device_name()
        ram_gb = 0.0
        if platform.system() == "Darwin":
            try:
                r = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, timeout=5, capture_output=True)
                ram_gb = round(int(r.stdout.strip()) / (1024**3), 1)
            except Exception:
                pass
        device = _match_device_profile(chip, ram_gb)
    if device is None:
        device = get_device("MacBook Air M5 (16GB)")
    return device, device_from_query


def model_lookup_search(
    query: str,
    device_name: str = "",
    max_results: int = 50,
    tts_variant: str = "",
) -> dict[str, Any] | None:
    """Fast model-name lookup across local catalog plus remote search slice."""
    terms = _extract_model_lookup_terms(query)
    if not terms:
        return None

    from .recommendation_engine import _load_catalog
    from .model_catalog_sync import fetch_remote_model_search

    query_lower = query.lower()
    device, device_from_query = _resolve_search_device(query_lower, device_name)
    max_size = device.max_model_size_gb

    by_hint: dict[str, dict] = {}
    for model in _load_catalog():
        if model.get("mlx", False) and _matches_model_lookup(query, model) and matches_tts_variant(model, tts_variant):
            by_hint[model.get("download_hint", "")] = _format_search_result(model, max_size)

    if not by_hint:
        raw_terms = _extract_model_lookup_raw_terms(query)
        remote_query = raw_terms[0] if raw_terms else terms[0]
        remote_cache_key = _normalize_model_lookup(remote_query)
        if remote_cache_key not in _REMOTE_LOOKUP_CACHE:
            try:
                remote_catalog = fetch_remote_model_search(remote_query, timeout=8.0, limit=max(max_results, 50))
                _REMOTE_LOOKUP_CACHE[remote_cache_key] = remote_catalog.get("models", [])
            except Exception as exc:
                logger.info("Remote model lookup failed for %r: %s", query, exc)
                _REMOTE_LOOKUP_CACHE[remote_cache_key] = []

        for model in _REMOTE_LOOKUP_CACHE.get(remote_cache_key, []):
            if model.get("mlx", False) and _matches_model_lookup(query, model) and matches_tts_variant(model, tts_variant):
                by_hint[model.get("download_hint", "")] = _format_search_result(model, max_size)

    if not by_hint:
        return None

    results = list(by_hint.values())
    results.sort(
        key=lambda r: (
            not r["fits_device"],
            -r.get("params_b", 0),
            r["estimated_size_gb"],
            r["name"],
        )
    )
    return {
        "results": results[:max_results],
        "detected_device": device.name if device_from_query else None,
        "detected_max_size_gb": round(device.max_model_size_gb, 1) if device_from_query else None,
    }


def _get_catalog_version() -> str:
    """Read the active catalog version from runtime cache or bundled JSON."""
    from .model_catalog_sync import load_effective_catalog

    data, _source, _signature = load_effective_catalog(start_background_refresh=False)
    return data.get("_meta", {}).get("version", "unknown")


def _vectors_cache_path() -> Path:
    """Path to cached catalog vectors file."""
    return EMBEDDING_HOME / VECTORS_FILENAME


def _load_cached_vectors(catalog_version: str) -> np.ndarray | None:
    """Load cached vectors if version matches. Returns None if stale/missing."""
    cache_path = _vectors_cache_path()
    if not cache_path.exists():
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        if data.get("version", None) is not None and str(data["version"]) == catalog_version:
            return data["vectors"]
    except Exception as e:
        logger.warning("Failed to load cached vectors: %s", e)

    return None


def compute_catalog_vectors(
    catalog: list[dict],
    model_dir: str,
    force: bool = False,
) -> np.ndarray:
    """Compute and cache embeddings for all catalog models.

    Returns shape (N, dim) float32 array.
    """
    catalog_version = _get_catalog_version()

    if not force:
        cached = _load_cached_vectors(catalog_version)
        if cached is not None:
            logger.info("Using cached catalog vectors (version %s, %d models)", catalog_version, len(cached))
            return cached

    encoder = _get_encoder(model_dir)
    texts = [_build_model_text(m) for m in catalog]
    logger.info("Computing embeddings for %d models...", len(texts))
    vectors = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    # Cache
    cache_path = _vectors_cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, vectors=vectors, version=catalog_version)
    logger.info("Cached catalog vectors to %s", cache_path)

    return vectors


# ---------------------------------------------------------------------------
# Intent search
# ---------------------------------------------------------------------------

def intent_search(
    query: str,
    device_name: str = "",
    max_results: int = 50,
    alpha: float = 0.6,
    tts_variant: str = "",
) -> list[dict]:
    from .recommendation_engine import _load_catalog, QUALITY_TIER_RANK
    from .device_profiles import get_device, all_devices

    # Check embedding readiness
    status = is_embedding_ready()
    if not status["ready"]:
        raise EmbeddingNotReadyError("Embedding model not downloaded yet")

    model_dir = status["model_dir"]
    catalog = _load_catalog()

    # Compute/load catalog vectors
    catalog_vectors = compute_catalog_vectors(catalog, model_dir)

    # Encode query
    encoder = _get_encoder(model_dir)
    query_vec = encoder.encode([query], normalize_embeddings=True)[0]

    # Cosine similarity (vectors are already normalized)
    similarities = catalog_vectors @ query_vec

    # Intent-category boosting: detect categories from keywords, boost matching models
    query_lower = query.lower()
    intent_categories: set[str] = set()
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)
    matched_spans: list[tuple[int, int]] = []
    for keyword in sorted_keywords:
        pos = query_lower.find(keyword.lower())
        if pos == -1:
            continue
        span = (pos, pos + len(keyword))
        if any(s[0] <= span[0] and s[1] >= span[1] for s in matched_spans):
            continue
        matched_spans.append(span)
        use_case = _KEYWORD_MAP[keyword]
        # Map use_case to intent categories
        if use_case == "voice":
            intent_categories.update(["asr", "tts"])  # "voice" implies both input and output
        elif use_case == "asr":
            intent_categories.add("asr")
        elif use_case == "tts":
            intent_categories.add("tts")
        elif use_case == "multimodal":
            intent_categories.add("vlm")
        elif use_case == "translation":
            intent_categories.add("translation")  # strength, not category

    # Resolve device for fit scoring
    # Step 1: detect device intent from query (e.g. "on phone" → iPhone 8GB)
    _device_from_query = _detect_device_from_query(query_lower)
    device = None
    if _device_from_query:
        device = get_device(_device_from_query)
    if device is None and device_name:
        device = get_device(device_name)
    if device is None:
        # Auto-detect from current hardware
        from backend.api.system_info import _match_device_profile
        from backend.core.auto_tune import _detect_device_name
        import platform, subprocess
        chip = device_name or _detect_device_name()
        ram_gb = 0.0
        if platform.system() == "Darwin":
            try:
                r = subprocess.run(["sysctl", "-n", "hw.memsize"], text=True, timeout=5, capture_output=True)
                ram_gb = round(int(r.stdout.strip()) / (1024**3), 1)
            except Exception:
                pass
        device = _match_device_profile(chip, ram_gb)
    if device is None:
        device = get_device("MacBook Air M5 (16GB)")

    max_size = device.max_model_size_gb

    # Build scored results
    results = []
    for i, m in enumerate(catalog):
        if not m.get("mlx", False):
            continue
        if not matches_tts_variant(m, tts_variant):
            continue

        semantic = float(similarities[i])
        size = m["size_gb"]
        fits = size <= max_size
        headroom = max(0, max_size - size)

        # --- Intent matching (computed first — influences device_fit) ---
        # Category match = model's category is in detected intents (strong signal)
        # Strength match = model has a matching strength (weaker signal)
        if intent_categories:
            model_cat = m.get("category", "")
            model_strengths = set(m.get("strengths", []))
            category_match = model_cat in intent_categories
            strength_match = bool(intent_categories & model_strengths)
            if category_match:
                intent_mult = 1.5   # strong boost: right model type
            elif strength_match:
                intent_mult = 1.2   # moderate boost: has the skill
            else:
                intent_mult = 0.6   # penalize non-matching models
        else:
            category_match = False
            intent_mult = 1.0

        # --- Device-fit score ---
        if fits:
            if category_match and model_cat in ("tts", "asr"):
                # TTS/ASR: flat fit — size is a speed/quality tradeoff, not "bigger = better".
                # Let semantic similarity decide which models surface, not size.
                fit_score = 0.7
            elif category_match:
                # Other category matches (VLM, etc): dampened size preference
                fit_score = 0.5 + 0.5 * (size / max(max_size, 0.1))
            else:
                fit_score = size / max(max_size, 0.1)  # general: bigger = better
            tier_bonus = (3 - QUALITY_TIER_RANK.get(m.get("quality_tier", "balanced"), 2)) / 3
            device_fit = fit_score * 0.7 + tier_bonus * 0.3
        else:
            device_fit = 0.0

        final_score = (alpha * semantic + (1 - alpha) * device_fit) * intent_mult

        results.append({
            "name": m["name"],
            "description": m.get("description", ""),
            "estimated_size_gb": size,
            "fits_device": fits,
            "headroom_gb": round(headroom, 1),
            "quality_tier": m.get("quality_tier", "balanced"),
            "download_hint": m.get("download_hint", ""),
            "category": m.get("category", "llm"),
            "family": m.get("family", ""),
            "params_b": m.get("params_b", 0),
            "context_k": m.get("context_k", 0),
            "semantic_score": round(semantic, 4),
            "_final_score": final_score,
        })

    # Exact/substring name match boost: if query looks like a model name,
    # prioritize models whose name contains the query (or vice versa).
    lookup_terms = _extract_model_lookup_terms(query_lower)
    for r in results:
        name_norm = _normalize_model_lookup(r["name"])
        hint_norm = _normalize_model_lookup(r["download_hint"])
        if lookup_terms and any(term in name_norm or term in hint_norm for term in lookup_terms):
            r["_final_score"] += 10.0  # strong boost to surface exact matches

    # Sort: fits_device first, then by final_score descending
    results.sort(key=lambda r: (not r["fits_device"], -r["_final_score"]))

    # Remove internal field
    for r in results:
        del r["_final_score"]

    # Return results + metadata about detected device intent
    return {
        "results": results[:max_results],
        "detected_device": device.name if _device_from_query else None,
        "detected_max_size_gb": round(device.max_model_size_gb, 1) if _device_from_query else None,
    }


# ---------------------------------------------------------------------------
# Tag-based fallback
# ---------------------------------------------------------------------------

def tag_based_fallback(
    query: str,
    device_name: str = "",
    max_results: int = 8,
    tts_variant: str = "",
) -> list[dict]:
    """Keyword-based fallback when embedding model is unavailable.

    Maps query keywords to use_case categories, then delegates to
    the existing recommend_models() function.
    """
    from .recommendation_engine import recommend_models

    query_lower = query.lower()

    # Find matching use_cases from keywords (longer keywords first to avoid
    # "voice" matching before "voice synthesis")
    matched_cases: list[str] = []
    matched_spans: list[tuple[int, int]] = []
    sorted_keywords = sorted(_KEYWORD_MAP.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        kw_lower = keyword.lower()
        pos = query_lower.find(kw_lower)
        if pos == -1:
            continue
        # Skip if this span is already covered by a longer keyword
        span = (pos, pos + len(kw_lower))
        if any(s[0] <= span[0] and s[1] >= span[1] for s in matched_spans):
            continue
        matched_spans.append(span)
        use_case = _KEYWORD_MAP[keyword]
        # Expand composite "voice" → both "asr" and "tts"
        if use_case == "voice":
            for uc in ("asr", "tts"):
                if uc not in matched_cases:
                    matched_cases.append(uc)
        elif use_case not in matched_cases:
            matched_cases.append(use_case)

    if not matched_cases:
        matched_cases = ["chat"]  # Default

    # Model-name lookup path: when the user types "Qwen3.6" or a repo id,
    # return matching catalog entries directly. This must work even before the
    # embedding model is downloaded; otherwise exact release-name searches
    # degrade to generic "chat" recommendations.
    lookup = model_lookup_search(query, device_name, max_results, tts_variant=tts_variant)
    if lookup:
        return lookup["results"]

    device, _device_from_query = _resolve_search_device(query_lower, device_name)

    # Collect results from all matched use_cases (fair quota per category)
    all_results: list[dict] = []
    seen_hints: set[str] = set()
    per_case_limit = max(max_results // len(matched_cases), 2)

    for uc in matched_cases:
        recs = recommend_models(device, use_case=uc, max_results=per_case_limit, tts_variant=tts_variant)
        for rec in recs:
            if rec.download_hint not in seen_hints:
                seen_hints.add(rec.download_hint)
                all_results.append({
                    "name": rec.name,
                    "description": rec.description,
                    "estimated_size_gb": rec.estimated_size_gb,
                    "fits_device": rec.fits_device,
                    "headroom_gb": round(rec.headroom_gb, 1),
                    "quality_tier": rec.quality_tier,
                    "download_hint": rec.download_hint,
                    "category": rec.category,
                    "family": rec.family,
                    "params_b": rec.params_b,
                    "context_k": rec.context_k,
                    "semantic_score": 0.0,
                })

    return all_results[:max_results]
