# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Runtime model catalog refresh for the packaged Edge Studio app.

The wheel ships with a bundled ``model_catalog.json`` so the app works
offline. At runtime we keep a user-local cache under the platform cache
directory and refresh it from public mlx-community model metadata when stale.
No user query text or local model metadata is sent during this refresh.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from backend.services.app_dirs import cache_path

logger = logging.getLogger(__name__)

BUNDLED_CATALOG_PATH = Path(__file__).parent / "model_catalog.json"
CATALOG_CACHE_DIR = Path(
    os.environ.get("EDGESTUDIO_CATALOG_CACHE_DIR", str(cache_path("catalog")))
).expanduser().resolve()
CATALOG_CACHE_PATH = CATALOG_CACHE_DIR / "model_catalog.json"
REFRESH_TTL_SECONDS = int(os.environ.get("EDGESTUDIO_CATALOG_REFRESH_TTL_SECONDS", "86400"))
REFRESH_MAX_PAGES = int(os.environ.get("EDGESTUDIO_CATALOG_REFRESH_MAX_PAGES", "1"))
DISABLE_REFRESH = os.environ.get("EDGESTUDIO_DISABLE_CATALOG_REFRESH", "").lower() in {
    "1",
    "true",
    "yes",
}

HF_ENDPOINTS = [
    "https://huggingface.co/api/models",
    "https://hf-mirror.com/api/models",
]

PIPELINE_TO_CATEGORY = {
    "text-generation": "llm",
    "image-text-to-text": "vlm",
    "video-text-to-text": "vlm",
    "automatic-speech-recognition": "asr",
    "text-to-speech": "tts",
    "audio-text-to-text": "asr",
}

CATEGORY_STRENGTHS = {
    "llm": ["chat"],
    "vlm": ["chat", "multimodal"],
    "asr": ["asr"],
    "tts": ["tts"],
}

CATEGORY_SEARCH_TEXT = {
    "llm": "Text generation and language model. General conversation, Q&A, chatbot, assistant.",
    "vlm": "Vision language model. Understand images, photos, documents, screenshots, visual analysis.",
    "asr": "Automatic speech recognition. Convert voice and spoken audio to text. Transcription, dictation, meeting notes, subtitles.",
    "tts": "Text to speech synthesis. Generate natural human-like voice from text. Narration, read aloud, voiceover.",
}

NAME_PATTERNS: list[tuple[str, str]] = [
    (r"whisper", "asr"),
    (r"sensevoice", "asr"),
    (r"parakeet", "asr"),
    (r"moonshine", "asr"),
    (r"\basr\b", "asr"),
    (r"\btts\b", "tts"),
    (r"kokoro", "tts"),
    (r"soprano", "tts"),
    (r"voxtral.*tts", "tts"),
    (r"fish.*audio", "tts"),
    (r"\bvlm\b", "vlm"),
    (r"qwen.*vl\b", "vlm"),
    (r"internvl", "vlm"),
    (r"minicpm.*v", "vlm"),
    (r"llava", "vlm"),
    (r"gemma.*it.*qat", "vlm"),
    (r"pixtral", "vlm"),
    (r"phi.*vision", "vlm"),
]

STRENGTH_PATTERNS: list[tuple[str, str]] = [
    (r"code|coder|starcoder|codellama|deepseek.*coder", "coding"),
    (r"reason|think|r1|qwq|math", "reasoning"),
    (r"translat", "translation"),
    (r"multilingual|201.lang|119.lang", "translation"),
]

ALLOWED_QUANTS = {"3bit", "4bit", "6bit", "8bit", "bf16"}

_KNOWN_PARAMS: dict[str, float] = {
    "whisper-tiny": 0.039,
    "whisper-base": 0.074,
    "whisper-small": 0.244,
    "whisper-medium": 0.769,
    "whisper-large": 1.55,
    "kokoro": 0.082,
    "soprano": 0.08,
    "moonshine-tiny": 0.027,
}

_QUANT_PATTERNS = [
    (re.compile(r"(?:^|[-_. ])3[ -]?bit", re.I), "3bit"),
    (re.compile(r"(?:^|[-_. ])4[ -]?bit", re.I), "4bit"),
    (re.compile(r"(?:^|[-_. ])6[ -]?bit", re.I), "6bit"),
    (re.compile(r"(?:^|[-_. ])8[ -]?bit", re.I), "8bit"),
    (re.compile(r"(?:^|[-_.])q3", re.I), "3bit"),
    (re.compile(r"(?:^|[-_.])q4", re.I), "4bit"),
    (re.compile(r"(?:^|[-_.])q6", re.I), "6bit"),
    (re.compile(r"(?:^|[-_.])q8", re.I), "8bit"),
    (re.compile(r"(?:^|[-_. ])w4a16", re.I), "4bit"),
    (re.compile(r"(?:^|[-_. ])int4", re.I), "4bit"),
    (re.compile(r"(?:^|[-_. ])int8", re.I), "8bit"),
    (re.compile(r"(?:^|[-_. ])bf16", re.I), "bf16"),
    (re.compile(r"(?:^|[-_. ])fp16", re.I), "bf16"),
    (re.compile(r"(?:^|[-_. ])f16", re.I), "bf16"),
]

_refresh_lock = threading.Lock()
_refreshing = False
_last_refresh_error: str | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("Failed to read catalog %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        logger.warning("Ignoring invalid catalog shape at %s", path)
        return None
    return data


def _version_tuple(value: str | None) -> tuple[int, int, int]:
    if not value:
        return (0, 0, 0)
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _catalog_version(data: dict[str, Any] | None) -> str:
    if not data:
        return "unknown"
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    return str(meta.get("version") or "unknown")


def _catalog_generated_at(data: dict[str, Any] | None) -> str | None:
    if not data:
        return None
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    generated_at = meta.get("generated_at")
    return str(generated_at) if generated_at else None


def _cache_is_stale() -> bool:
    if not CATALOG_CACHE_PATH.exists():
        return True
    try:
        age = datetime.now(timezone.utc).timestamp() - CATALOG_CACHE_PATH.stat().st_mtime
    except OSError:
        return True
    return age > REFRESH_TTL_SECONDS


def _select_effective_catalog() -> tuple[dict[str, Any], str, Path]:
    bundled = _read_json(BUNDLED_CATALOG_PATH)
    if bundled is None:
        bundled = {"_meta": {"version": "unknown", "total_models": 0}, "models": []}

    cached = _read_json(CATALOG_CACHE_PATH)
    if cached is None:
        return bundled, "bundled", BUNDLED_CATALOG_PATH

    cached_version = _version_tuple(_catalog_version(cached))
    bundled_version = _version_tuple(_catalog_version(bundled))
    if cached_version >= bundled_version and len(cached.get("models", [])) > 0:
        return cached, "remote_cache", CATALOG_CACHE_PATH
    return bundled, "bundled", BUNDLED_CATALOG_PATH


def load_effective_catalog(
    *,
    start_background_refresh: bool = True,
) -> tuple[dict[str, Any], str, str]:
    """Return the active catalog plus source and reload signature."""
    if start_background_refresh and not DISABLE_REFRESH and _cache_is_stale():
        refresh_catalog_background()

    data, source, path = _select_effective_catalog()
    signature = f"{source}:{_catalog_version(data)}:{_catalog_generated_at(data) or ''}:{path}"
    return data, source, signature


def get_catalog_status() -> dict[str, Any]:
    data, source, path = _select_effective_catalog()
    bundled = _read_json(BUNDLED_CATALOG_PATH) or {"models": []}
    bundled_hints = {
        str(m.get("download_hint"))
        for m in bundled.get("models", [])
        if isinstance(m, dict) and m.get("download_hint")
    }
    runtime_only_hints = [
        str(m.get("download_hint"))
        for m in data.get("models", [])
        if isinstance(m, dict)
        and m.get("download_hint")
        and str(m.get("download_hint")) not in bundled_hints
    ]
    return {
        "enabled": not DISABLE_REFRESH,
        "source": source,
        "version": _catalog_version(data),
        "generated_at": _catalog_generated_at(data),
        "total_models": len(data.get("models", [])),
        "cache_path": str(CATALOG_CACHE_PATH),
        "cache_exists": CATALOG_CACHE_PATH.exists(),
        "active_path": str(path),
        "refreshing": _refreshing,
        "stale": (not DISABLE_REFRESH) and _cache_is_stale(),
        "last_error": _last_refresh_error,
        "runtime_only_download_hints": runtime_only_hints if source == "remote_cache" else [],
    }


def estimate_quality_tier(params_b: float, quant: str) -> str:
    if params_b >= 30:
        return "premium"
    if params_b >= 7:
        return "high"
    if params_b >= 2:
        return "balanced"
    return "entry"


def estimate_size_gb(name: str, params_b: float) -> float:
    quant = extract_quant(name)
    multipliers = {
        "3bit": 0.45,
        "4bit": 0.6,
        "6bit": 0.85,
        "8bit": 1.1,
        "bf16": 2.0,
    }
    return round(params_b * multipliers.get(quant, 0.6), 1)


def extract_params(name: str) -> float:
    name_lower = name.lower()
    match = re.search(r"(\d+\.?\d*)\s*b(?:illion)?(?:\s|$|-|_)", name_lower)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+\.?\d*)\s*m(?:illion)?(?:\s|$|-|_)", name_lower)
    if match:
        return round(float(match.group(1)) / 1000, 3)
    for pattern, params in _KNOWN_PARAMS.items():
        if pattern in name_lower:
            return params
    return 0


def extract_quant(name: str) -> str:
    for pattern, quant in _QUANT_PATTERNS:
        if pattern.search(name):
            return quant
    return "4bit"


def detect_family(name: str) -> str:
    name_lower = name.lower()
    families = [
        ("qwen3.6", "Qwen3.6"),
        ("qwen3.5", "Qwen3.5"),
        ("qwen3", "Qwen3"),
        ("qwen2.5", "Qwen2.5"),
        ("qwen2", "Qwen2"),
        ("llama-4", "Llama4"),
        ("llama-3.3", "Llama3.3"),
        ("llama-3.2", "Llama3.2"),
        ("llama-3.1", "Llama3.1"),
        ("llama", "Llama"),
        ("gemma-4", "Gemma4"),
        ("gemma-3n", "Gemma3n"),
        ("gemma-3", "Gemma3"),
        ("gemma-2", "Gemma2"),
        ("gemma", "Gemma"),
        ("phi-4", "Phi4"),
        ("phi-3", "Phi3"),
        ("phi", "Phi"),
        ("mistral", "Mistral"),
        ("ministral", "Ministral"),
        ("deepseek", "DeepSeek"),
        ("internvl", "InternVL"),
        ("internlm", "InternLM"),
        ("minicpm", "MiniCPM"),
        ("whisper", "Whisper"),
        ("sensevoice", "SenseVoice"),
        ("moonshine", "Moonshine"),
        ("parakeet", "Parakeet"),
        ("kokoro", "Kokoro"),
        ("soprano", "Soprano"),
        ("voxtral", "Voxtral"),
        ("starcoder", "StarCoder"),
        ("codellama", "CodeLlama"),
        ("vicuna", "Vicuna"),
        ("yi", "Yi"),
        ("glm", "GLM"),
        ("exaone", "EXAONE"),
        ("olmo", "OLMo"),
        ("smollm", "SmolLM"),
    ]
    for pattern, family in families:
        if pattern in name_lower:
            return family
    return name.split("/")[-1].split("-")[0]


def classify_model(model_id: str, pipeline_tag: str | None, tags: list[str]) -> str | None:
    if pipeline_tag and pipeline_tag in PIPELINE_TO_CATEGORY:
        return PIPELINE_TO_CATEGORY[pipeline_tag]

    name_lower = model_id.lower()
    for pattern, category in NAME_PATTERNS:
        if re.search(pattern, name_lower):
            return category

    tag_set = {str(t).lower() for t in tags}
    if "text-generation" in tag_set:
        return "llm"
    if "image-text-to-text" in tag_set:
        return "vlm"

    if "mlx" in tag_set and ("transformers" in tag_set or "safetensors" in tag_set):
        skip_tags = {
            "feature-extraction",
            "fill-mask",
            "sentence-similarity",
            "text-classification",
            "token-classification",
            "image-classification",
            "image-to-image",
            "text-to-image",
            "image-to-video",
            "text-to-video",
            "video-to-video",
            "image-to-3d",
            "text-to-3d",
            "robotics",
            "mask-generation",
            "image-segmentation",
            "depth-estimation",
            "object-detection",
            "zero-shot-image-classification",
            "zero-shot-classification",
            "image-feature-extraction",
            "time-series-forecasting",
            "text-to-audio",
            "audio-to-audio",
            "voice-activity-detection",
            "visual-document-retrieval",
            "text-ranking",
            "question-answering",
            "image-text-to-image",
            "image-text-to-video",
            "audio-classification",
            "video-classification",
        }
        if pipeline_tag and pipeline_tag in skip_tags:
            return None
        return "llm"

    return None


def detect_strengths(name: str, category: str, tags: list[str]) -> list[str]:
    strengths = list(CATEGORY_STRENGTHS.get(category, []))
    name_lower = name.lower()

    for pattern, strength in STRENGTH_PATTERNS:
        if re.search(pattern, name_lower) and strength not in strengths:
            strengths.append(strength)

    if "translation" not in strengths:
        if any(kw in name_lower for kw in ["multilingual", "translate"]):
            strengths.append("translation")

    tag_set = {str(t).lower() for t in tags}
    if "code" in tag_set and "coding" not in strengths:
        strengths.append("coding")
    return strengths


def friendly_name(model_id: str) -> str:
    name = model_id.split("/")[-1]
    name = name.replace("-MLX", "").replace("-mlx", "")
    name = name.replace("-4bit", " 4-bit").replace("-8bit", " 8-bit")
    name = name.replace("-3bit", " 3-bit").replace("-bf16", " bf16").replace("-fp16", " fp16")
    name = name.replace("-Instruct", " Instruct").replace("-instruct", " Instruct")
    name = re.sub(r"-+", " ", name)
    return name.strip()


def generate_search_text(model: dict[str, Any]) -> str:
    category = model["category"]
    parts = [CATEGORY_SEARCH_TEXT.get(category, ""), model.get("description", "")]

    for strength in model.get("strengths", []):
        if strength == "coding":
            parts.append("Code generation, programming, debugging, software development.")
        elif strength == "reasoning":
            parts.append("Complex reasoning, analysis, math, logic, chain of thought.")
        elif strength == "translation":
            parts.append("Language translation, multilingual communication, localization.")

    family = model.get("family", "")
    if "Whisper" in family:
        parts.append("OpenAI Whisper. Industry standard speech recognition. Multilingual.")
    elif "Kokoro" in family:
        parts.append("#1 TTS Arena. Ultra-efficient text to speech.")
    elif "DeepSeek" in family and "R1" in model.get("name", ""):
        parts.append("Deep reasoning and chain of thought specialist.")

    size = model.get("size_gb", 0)
    if size < 1:
        parts.append("Ultra-compact, runs on any Apple device.")
    elif size < 3:
        parts.append("Compact, suitable for iPhone and iPad.")
    elif size < 6:
        parts.append("Medium size, runs on 8GB+ devices.")
    elif size < 12:
        parts.append("Large model, needs 16GB+ RAM.")
    else:
        parts.append("Very large, needs Mac with 32GB+ RAM.")

    return " ".join(p for p in parts if p)


def build_catalog_from_hf_models(raw_models: list[dict[str, Any]]) -> dict[str, Any]:
    catalog_models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw in raw_models:
        model_id = str(raw.get("id") or "")
        if not model_id.startswith("mlx-community/") or model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        category = classify_model(model_id, raw.get("pipeline_tag"), tags)
        if category is None:
            continue

        params_b = extract_params(model_id)
        quant = extract_quant(model_id)
        if quant not in ALLOWED_QUANTS:
            continue

        entry = {
            "id": model_id.split("/")[-1].lower(),
            "name": friendly_name(model_id),
            "family": detect_family(model_id),
            "params_b": params_b,
            "quant": quant,
            "size_gb": estimate_size_gb(model_id, params_b) if params_b > 0 else 0,
            "context_k": 0,
            "category": category,
            "strengths": detect_strengths(model_id, category, tags),
            "quality_tier": estimate_quality_tier(params_b, quant),
            "description": "",
            "download_hint": model_id,
            "mlx": True,
            "_downloads": int(raw.get("downloads") or 0),
        }
        entry["search_text"] = generate_search_text(entry)
        catalog_models.append(entry)

    catalog_models.sort(key=lambda m: (m["category"], -m["_downloads"], m["name"]))
    for entry in catalog_models:
        del entry["_downloads"]

    now = datetime.now(timezone.utc)
    return {
        "_meta": {
            "version": now.strftime("%Y-%m-%d"),
            "source": "mlx-community (HuggingFace runtime refresh)",
            "generated_at": now.isoformat(),
            "total_models": len(catalog_models),
            "sources": [
                "https://huggingface.co/mlx-community",
                "https://hf-mirror.com/mlx-community",
                "https://www.modelscope.cn/organization/mlx-community",
            ],
            "allowed_quants": sorted(ALLOWED_QUANTS),
        },
        "models": catalog_models,
    }


def fetch_remote_catalog(*, timeout: float = 30.0) -> dict[str, Any]:
    raw_models: list[dict[str, Any]] = []
    errors: list[str] = []

    for endpoint in HF_ENDPOINTS:
        try:
            url = endpoint
            params: dict[str, Any] = {
                "author": "mlx-community",
                "limit": 1000,
                "sort": "downloads",
                "direction": "-1",
            }
            pages = 0
            while True:
                resp = httpx.get(url, params=params, timeout=timeout)
                resp.raise_for_status()
                batch = resp.json()
                if not isinstance(batch, list):
                    raise RuntimeError(f"Unexpected catalog response from {endpoint}")
                raw_models.extend(batch)
                pages += 1
                link = resp.headers.get("link", "")
                if 'rel="next"' in link and (REFRESH_MAX_PAGES <= 0 or pages < REFRESH_MAX_PAGES):
                    url = link.split(";")[0].strip("<> ")
                    params = {}
                else:
                    break
            if raw_models:
                break
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
            raw_models = []

    if not raw_models:
        raise RuntimeError("; ".join(errors) or "No models returned")

    catalog = build_catalog_from_hf_models(raw_models)
    if not catalog["models"]:
        raise RuntimeError("Remote catalog refresh produced no usable models")
    return catalog


def fetch_remote_model_search(
    query: str,
    *,
    timeout: float = 8.0,
    limit: int = 50,
) -> dict[str, Any]:
    """Fetch a small remote catalog slice for an explicit model-name query.

    Full mlx-community catalog refresh can take many pages.  When a user types
    a concrete release/family name such as "Qwen3.6", ask HuggingFace for just
    that search term so the simple-mode browser can surface new releases
    without waiting for full background sync.
    """
    query = query.strip()
    if not query:
        return build_catalog_from_hf_models([])

    errors: list[str] = []
    for endpoint in HF_ENDPOINTS:
        try:
            resp = httpx.get(
                endpoint,
                params={
                    "author": "mlx-community",
                    "search": query,
                    "limit": max(1, min(limit, 100)),
                    "sort": "downloads",
                    "direction": "-1",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw_models = resp.json()
            if not isinstance(raw_models, list):
                raise RuntimeError(f"Unexpected catalog search response from {endpoint}")
            return build_catalog_from_hf_models(raw_models)
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")

    raise RuntimeError("; ".join(errors) or "No models returned")


def refresh_catalog(*, force: bool = False) -> dict[str, Any]:
    """Refresh the user-local catalog cache synchronously."""
    global _last_refresh_error
    if DISABLE_REFRESH:
        return get_catalog_status()
    if not force and not _cache_is_stale():
        return get_catalog_status()

    try:
        catalog = fetch_remote_catalog()
        CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = CATALOG_CACHE_PATH.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, CATALOG_CACHE_PATH)
        _last_refresh_error = None
    except Exception as exc:
        _last_refresh_error = str(exc)
        logger.warning("Model catalog refresh failed: %s", exc)
    return get_catalog_status()


def refresh_catalog_background(*, force: bool = False) -> bool:
    """Start a single background refresh. Returns True if a new thread started."""
    global _refreshing
    if DISABLE_REFRESH:
        return False
    if not force and not _cache_is_stale():
        return False

    with _refresh_lock:
        if _refreshing:
            return False
        _refreshing = True

    def _run() -> None:
        global _refreshing
        try:
            refresh_catalog(force=force)
        finally:
            with _refresh_lock:
                _refreshing = False

    thread = threading.Thread(target=_run, name="model_catalog_refresh", daemon=True)
    thread.start()
    return True
