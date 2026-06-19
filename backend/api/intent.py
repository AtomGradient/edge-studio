# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Intent-driven search API — semantic model recommendation via embedding."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter

from backend.schemas.common import CreateTaskResponse
from backend.schemas.intent import (
    IntentSearchRequest,
    IntentSearchResultSchema,
    IntentSearchResponse,
    EmbeddingStatusSchema,
)
from backend.services.task_manager import task_manager
from backend.core.intent_search import (
    is_embedding_ready,
    is_embedding_dependency_ready,
    detect_region,
    download_embedding_model,
    intent_search,
    model_lookup_search,
    tag_based_fallback,
    EmbeddingNotReadyError,
    _get_catalog_version,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommend")

# Guard: only one auto-download at a time
_embedding_download_lock = threading.Lock()
_embedding_downloading = False
_embedding_task_id: str | None = None


def _start_embedding_download_task() -> str:
    """Start the embedding download once and return the active task id."""
    global _embedding_downloading, _embedding_task_id
    with _embedding_download_lock:
        if _embedding_downloading:
            if _embedding_task_id:
                return _embedding_task_id
            _embedding_downloading = False
        task_id = task_manager.create_task(metadata={"kind": "embedding_download"})
        _embedding_task_id = task_id
        _embedding_downloading = True

    def _run(progress_callback=None):
        global _embedding_downloading
        try:
            region = detect_region()
            if progress_callback:
                progress_callback(f"Detected region: {region}", 0.05)
            path = download_embedding_model(region=region, progress_callback=progress_callback)
            return {"path": path, "region": region}
        finally:
            with _embedding_download_lock:
                _embedding_downloading = False

    try:
        task_manager.run_in_thread(task_id, _run)
    except Exception:
        with _embedding_download_lock:
            _embedding_downloading = False
            _embedding_task_id = None
        raise
    return task_id


def _auto_download_embedding():
    """Trigger embedding model download in background (fire-and-forget)."""
    try:
        _start_embedding_download_task()
    except Exception as e:
        logger.warning("Embedding auto-download failed: %s", e)


@router.get("/embedding-status", response_model=EmbeddingStatusSchema)
def embedding_status():
    """Check if the embedding model is downloaded and ready."""
    status = is_embedding_ready()
    dependency_ready = is_embedding_dependency_ready()
    return EmbeddingStatusSchema(
        ready=status["ready"] and dependency_ready,
        model_repo=status.get("model_repo"),
        region=status["region"],
        catalog_version=_get_catalog_version() if status["ready"] and dependency_ready else None,
        dependency_ready=dependency_ready,
        downloading=_embedding_downloading,
        task_id=_embedding_task_id,
    )


@router.post("/embedding-download", response_model=CreateTaskResponse)
def trigger_embedding_download() -> CreateTaskResponse:
    """Download the embedding model (auto-detects region). Returns task_id."""
    task_id = _start_embedding_download_task()
    return CreateTaskResponse(task_id=task_id)


@router.post("/intent-search", response_model=IntentSearchResponse)
def intent_search_endpoint(req: IntentSearchRequest):
    """Semantic model search: natural language query → ranked recommendations.

    Automatically falls back to keyword-based search if embedding is unavailable.
    Returns results + detected device context (if query mentions a device like "phone").
    """
    lookup_data = model_lookup_search(
        query=req.query,
        device_name=req.device_name,
        max_results=req.max_results,
        tts_variant=req.tts_variant,
    )
    if lookup_data:
        data = lookup_data
    else:
        try:
            data = intent_search(
                query=req.query,
                device_name=req.device_name,
                max_results=req.max_results,
                tts_variant=req.tts_variant,
            )
        except EmbeddingNotReadyError:
            logger.info("Embedding not ready, falling back to tag-based search")
            results = tag_based_fallback(
                query=req.query,
                device_name=req.device_name,
                max_results=req.max_results,
                tts_variant=req.tts_variant,
            )
            data = {"results": results, "detected_device": None, "detected_max_size_gb": None}
            # Auto-download embedding model in background so next search is semantic
            _auto_download_embedding()
        except ImportError:
            logger.warning("sentence-transformers not installed, using tag-based fallback")
            results = tag_based_fallback(
                query=req.query,
                device_name=req.device_name,
                max_results=req.max_results,
                tts_variant=req.tts_variant,
            )
            data = {"results": results, "detected_device": None, "detected_max_size_gb": None}

    return IntentSearchResponse(
        results=[IntentSearchResultSchema(**r) for r in data["results"]],
        detected_device=data.get("detected_device"),
        detected_max_size_gb=data.get("detected_max_size_gb"),
    )
