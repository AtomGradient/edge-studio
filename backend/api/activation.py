# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Activation profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.analysis import (
    ActivationHeatmapData,
    GenerateProfileRequest,
    LoadProfileRequest,
    ProfileSummary,
)
from backend.schemas.common import CreateTaskResponse
from backend.services.model_manager import manager
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["activation"])


@router.get("/{model_id}/profiles", response_model=dict[str, list[str]])
def list_profiles(model_id: str) -> dict[str, list[str]]:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.activation_loader import find_profile_files

    files = find_profile_files(loaded.model_dir)
    return {"profiles": files}


@router.post("/{model_id}/profile/load", response_model=ProfileSummary)
def load_profile(model_id: str, req: LoadProfileRequest) -> ProfileSummary:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.activation_loader import load_profile_json, load_profile_npz

    path = req.profile_path
    try:
        if path.endswith(".npz"):
            profile = load_profile_npz(path)
        else:
            profile = load_profile_json(path)
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(400, f"Failed to load profile: {exc}")

    manager.store_profile(model_id, profile)

    dead_01 = profile.total_dead_neurons(0.1)
    total = profile.layers[0].max_activations.shape[0] * len(profile.layers) if profile.layers else 0

    return ProfileSummary(
        intermediate_size=profile.layers[0].max_activations.shape[0] if profile.layers else 0,
        num_layers=len(profile.layers),
        run_count=getattr(profile, "run_count", 1),
        total_dead_at_01=dead_01,
        dead_ratio_at_01=dead_01 / total if total else 0.0,
    )


@router.get("/{model_id}/activation/heatmap", response_model=ActivationHeatmapData)
def get_heatmap_data(model_id: str, threshold: float = 0.1) -> ActivationHeatmapData:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    profile = manager.get_profile(model_id)
    if not profile:
        raise HTTPException(400, "No activation profile loaded")

    import numpy as np

    max_mat = profile.max_acts_matrix
    mean_mat = profile.mean_acts_matrix
    dead_per_layer = profile.dead_neurons_per_layer(threshold)

    return ActivationHeatmapData(
        max_matrix=np.nan_to_num(max_mat, nan=0.0).tolist(),
        mean_matrix=np.nan_to_num(mean_mat, nan=0.0).tolist(),
        num_layers=max_mat.shape[0],
        neurons_per_layer=max_mat.shape[1] if max_mat.ndim > 1 else 0,
        dead_per_layer=dead_per_layer,
        threshold=threshold,
    )


@router.post("/{model_id}/profile/generate", response_model=CreateTaskResponse)
def generate_profile(model_id: str, req: GenerateProfileRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    from backend.core.activation_profiler import profile_model

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        # Build a progress adapter: core expects (current_run, total_runs, message)
        # task_manager expects (message: str, percent: float)
        def _progress(current: int, total: int, msg: str):
            if progress_callback:
                progress_callback(msg, current / max(total, 1))

        result = profile_model(
            model_dir=loaded.model_dir,
            config=loaded.config,
            model_type=loaded.architecture.model_type if loaded.architecture else "generic",
            num_runs=req.num_runs,
            progress_callback=_progress,
        )
        manager.store_profile(model_id, result)

        # Return profile summary
        dead_01 = result.total_dead_neurons(0.1)
        total = result.layers[0].max_activations.shape[0] * len(result.layers) if result.layers else 0
        return {
            "intermediate_size": result.layers[0].max_activations.shape[0] if result.layers else 0,
            "num_layers": len(result.layers),
            "run_count": getattr(result, "run_count", req.num_runs),
            "total_dead_at_01": dead_01,
            "dead_ratio_at_01": dead_01 / total if total else 0.0,
        }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
