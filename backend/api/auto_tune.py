# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Auto-tune benchmark API routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.auto_tune import AutoTuneRequest
from backend.schemas.common import CreateTaskResponse
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["auto_tune"])


@router.post("/auto-tune", response_model=CreateTaskResponse)
def start_auto_tune(req: AutoTuneRequest):
    """Start an auto-tune benchmark task."""
    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.auto_tune import TuneConfig, run_auto_tune, _load_cache, _detect_device_name
        import os

        device = req.device_name or _detect_device_name()

        # Check cache unless force_rerun
        if not req.force_rerun:
            cached = _load_cache(req.model_dir, device)
            if cached:
                from dataclasses import asdict
                if progress_callback:
                    progress_callback("Loaded from cache", 1.0)
                return {
                    "success": True,
                    "model_name": cached.model_name,
                    "device_name": cached.device_name,
                    "best": asdict(cached.best) if cached.best else None,
                    "all_candidates": [asdict(c) for c in cached.all_candidates],
                    "search_time_seconds": cached.search_time_seconds,
                    "total_configs_tested": cached.total_configs_tested,
                    "cached": True,
                    "cache_path": cached.cache_path,
                    "error": "",
                }

        config = TuneConfig(
            model_dir=req.model_dir,
            device_name=device,
            max_tokens=req.max_tokens,
            num_runs=req.num_runs,
            search_temperatures=req.search_temperatures,
            search_kv_cache_sizes=req.search_kv_cache_sizes,
        )

        result = run_auto_tune(config, progress_callback=progress_callback)

        from dataclasses import asdict
        return {
            "success": result.success,
            "model_name": result.model_name,
            "device_name": result.device_name,
            "best": asdict(result.best) if result.best else None,
            "all_candidates": [asdict(c) for c in result.all_candidates],
            "search_time_seconds": result.search_time_seconds,
            "total_configs_tested": result.total_configs_tested,
            "cached": result.cached,
            "cache_path": result.cache_path,
            "error": result.error,
        }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
