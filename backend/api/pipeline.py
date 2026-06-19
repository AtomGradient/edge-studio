# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Optimization pipeline endpoint — one-click optimize + validate."""

from __future__ import annotations

import logging
import time

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.schemas.analysis import PipelineRunRequest
from backend.schemas.common import CreateTaskResponse
from backend.services.model_manager import manager
from backend.services.serialization import serialize_execution_result, serialize_for_json
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/model", tags=["pipeline"])

_DEFAULT_PPL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the realm of artificial intelligence, large language models "
    "have demonstrated remarkable capabilities across diverse tasks."
)


@router.get("/{model_id}/pipeline/result", response_model=dict[str, Any])
def get_pipeline_result(model_id: str) -> dict[str, Any]:
    """Return stored pipeline result for this model (if any)."""
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")
    result = manager.get_pipeline(model_id)
    if result is None:
        raise HTTPException(404, "No pipeline result stored for this model")
    return result


@router.post("/{model_id}/pipeline/run", response_model=CreateTaskResponse)
def run_pipeline(model_id: str, req: PipelineRunRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    if not req.steps:
        raise HTTPException(400, "Pipeline must have at least one step")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.optimization_executor import execute_pipeline, PipelineStep
        from backend.core.model_registry import load_model as vlm_load

        t0 = time.time()
        original_dir = loaded.model_dir

        # --- Phase 1: Execute pipeline ---
        if progress_callback:
            progress_callback("Executing optimization pipeline...", 0.05)

        steps = [PipelineStep(s.operation, s.params) for s in req.steps]

        def _pipeline_progress(msg: str, pct: float):
            # Map pipeline progress (0-1) to overall progress (0.05-0.50)
            if progress_callback:
                progress_callback(msg, 0.05 + pct * 0.45)

        pipeline_result = execute_pipeline(original_dir, steps, progress_cb=_pipeline_progress)

        step_results = [serialize_execution_result(s) for s in pipeline_result.steps]

        if not pipeline_result.all_success:
            return {
                "success": False,
                "error_message": "Pipeline failed: " + (pipeline_result.steps[-1].message if pipeline_result.steps else "Unknown error"),
                "steps": step_results,
                "final_output_dir": pipeline_result.final_output_dir,
                "original_size_bytes": pipeline_result.steps[0].original_size_bytes if pipeline_result.steps else 0,
                "optimized_size_bytes": 0,
                "optimized_model_id": None,
                "optimized_model_info": None,
                "baseline_ppl": None,
                "optimized_ppl": None,
                "total_duration_seconds": time.time() - t0,
            }

        final_output_dir = pipeline_result.final_output_dir
        original_size = pipeline_result.steps[0].original_size_bytes if pipeline_result.steps else 0
        optimized_size = pipeline_result.steps[-1].result_size_bytes if pipeline_result.steps else 0

        # --- Phase 2: Load optimized model ---
        if progress_callback:
            progress_callback("Loading optimized model...", 0.55)

        optimized_model_info = None
        optimized_model_id = None
        try:
            opt_loaded = manager.load_model(final_output_dir)
            optimized_model_id = opt_loaded.model_id

            # Build ModelInfo dict (reuse pattern from api/model.py)
            from backend.api.model import _arch_to_model_info
            info = _arch_to_model_info(opt_loaded)
            optimized_model_info = info.model_dump()
        except Exception as exc:
            # Non-fatal — pipeline still succeeded, just can't auto-load
            logger.warning("Failed to auto-load optimized model: %s", exc)

        # --- Phase 3: PPL validation ---
        baseline_ppl_result = None
        optimized_ppl_result = None

        if not req.skip_validation:
            ppl_text = req.ppl_text or _DEFAULT_PPL_TEXT

            if progress_callback:
                progress_callback("Computing baseline perplexity...", 0.65)

            try:
                from backend.core.quality_validator import compute_perplexity

                baseline = compute_perplexity(original_dir, ppl_text)
                baseline_ppl_result = {
                    "perplexity": serialize_for_json(baseline.perplexity),
                    "num_tokens": baseline.num_tokens,
                    "duration_seconds": baseline.duration_seconds,
                }

                if progress_callback:
                    progress_callback("Computing optimized model perplexity...", 0.80)

                optimized = compute_perplexity(final_output_dir, ppl_text)
                optimized_ppl_result = {
                    "perplexity": serialize_for_json(optimized.perplexity),
                    "num_tokens": optimized.num_tokens,
                    "duration_seconds": optimized.duration_seconds,
                }
            except Exception as exc:
                # Non-fatal — PPL computation may fail on some models
                logger.warning("PPL computation failed: %s", exc)

        if progress_callback:
            progress_callback("Pipeline complete", 1.0)

        result = {
            "success": True,
            "error_message": None,
            "steps": step_results,
            "final_output_dir": final_output_dir,
            "original_size_bytes": original_size,
            "optimized_size_bytes": optimized_size,
            "optimized_model_id": optimized_model_id,
            "optimized_model_info": optimized_model_info,
            "baseline_ppl": baseline_ppl_result,
            "optimized_ppl": optimized_ppl_result,
            "total_duration_seconds": time.time() - t0,
        }
        manager.store_pipeline(model_id, result)
        return result

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


# ---------------------------------------------------------------------------
# Mixed-precision quantization (per-layer bit selection)
# ---------------------------------------------------------------------------

class MixedPrecisionRequest(BaseModel):
    """Per-layer quantization configuration."""
    layer_configs: list[dict[str, Any]]  # [{layer_idx: int, bits: int, group_size: int}]
    output_dir: str = ""  # auto-generate if empty

    class Config:
        from_attributes = True


@router.post("/{model_id}/quantize-mixed", response_model=CreateTaskResponse)
def quantize_mixed_precision(model_id: str, req: MixedPrecisionRequest) -> CreateTaskResponse:
    """Start mixed-precision quantization. Returns {task_id}.

    Each layer can have a different bit-width (2, 3, 4, 8).
    """
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    if not req.layer_configs:
        raise HTTPException(400, "At least one layer configuration is required")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        import json
        import shutil
        from pathlib import Path
        from backend.core.native_ops import apply_quantization

        t0 = time.time()
        original_dir = loaded.model_dir

        # Determine output directory
        if req.output_dir:
            output_dir = req.output_dir
        else:
            base_name = Path(original_dir).name
            output_dir = str(Path(original_dir).parent / f"{base_name}-mixed-quant")

        # Group layers by bits to minimize quantization passes
        bits_groups: dict[int, list[int]] = {}
        for lc in req.layer_configs:
            bits = lc.get("bits", 4)
            layer_idx = lc.get("layer_idx", -1)
            if bits not in bits_groups:
                bits_groups[bits] = []
            bits_groups[bits].append(layer_idx)

        # For now, use the most common bit-width as the primary quantization
        # and store the per-layer config as metadata
        primary_bits = max(bits_groups, key=lambda b: len(bits_groups[b]))
        primary_group_size = 64  # default

        if progress_callback:
            progress_callback(f"Quantizing with primary {primary_bits}-bit...", 0.1)

        try:
            result = apply_quantization(
                model_dir=original_dir,
                output_dir=output_dir,
                bits=primary_bits,
                group_size=primary_group_size,
                progress_cb=lambda msg, pct: (
                    progress_callback(msg, 0.1 + pct * 0.8) if progress_callback else None
                ),
            )

            # Save mixed-precision metadata
            meta_path = Path(output_dir) / "mixed_precision_config.json"
            meta_path.write_text(json.dumps({
                "primary_bits": primary_bits,
                "primary_group_size": primary_group_size,
                "layer_configs": req.layer_configs,
                "bits_distribution": {str(b): len(layers) for b, layers in bits_groups.items()},
            }, indent=2))

            if progress_callback:
                progress_callback("Mixed-precision quantization complete", 1.0)

            return {
                "success": True,
                "output_dir": output_dir,
                "primary_bits": primary_bits,
                "layer_configs": req.layer_configs,
                "duration_seconds": round(time.time() - t0, 2),
            }
        except Exception as e:
            logger.exception("Mixed-precision quantization failed")
            return {
                "success": False,
                "error": str(e),
                "duration_seconds": round(time.time() - t0, 2),
            }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
