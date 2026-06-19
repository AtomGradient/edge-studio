# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Benchmark API — one-click baseline vs optimized comparison."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.schemas.common import CreateTaskResponse
from backend.services.task_manager import task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class BenchmarkRequest(BaseModel):
    baseline_dir: str
    compare_dir: str | None = None
    num_tokens: int = 100
    num_ppl_texts: int = 3
    enable_dsr: bool = False
    dsr_budget: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_guidance(
    disk_reduction_pct: float,
    memory_reduction_pct: float,
    speed_improvement_pct: float,
    ppl_delta: float,
    baseline_ppl: float,
) -> dict:
    """Generate verdict + actionable guidance from comparison metrics."""
    ppl_change_pct = (ppl_delta / baseline_ppl * 100) if baseline_ppl > 0 else 0.0

    if ppl_change_pct > 15:
        verdict = "danger"
        title = "Quality degradation too high"
        message = (
            f"PPL increased by {ppl_change_pct:.1f}% (>{15}% threshold). "
            "This model may produce noticeably worse outputs. "
            "Try: reduce pruning threshold, protect more layers, or use fewer optimization steps."
        )
    elif ppl_change_pct > 5:
        verdict = "warning"
        title = "Moderate quality loss — review before deploying"
        message = (
            f"PPL increased by {ppl_change_pct:.1f}% (5–15% range). "
            "Acceptable for most use cases but test on your target domain. "
            "If quality is critical, try reducing neuron pruning threshold."
        )
    elif disk_reduction_pct > 40 and speed_improvement_pct > 50:
        verdict = "success"
        title = "Excellent optimization"
        message = (
            f"Disk −{disk_reduction_pct:.0f}%, Memory −{memory_reduction_pct:.0f}%, "
            f"Speed +{speed_improvement_pct:.0f}%, PPL Δ{ppl_delta:+.2f}. "
            "This model is ready for edge deployment."
        )
    else:
        verdict = "success"
        title = "Good optimization"
        message = (
            f"Disk −{disk_reduction_pct:.0f}%, Memory −{memory_reduction_pct:.0f}%, "
            f"Speed +{speed_improvement_pct:.0f}%, PPL Δ{ppl_delta:+.2f}. "
            "Balanced compression with minimal quality impact."
        )

    return {"verdict": verdict, "title": title, "message": message, "ppl_change_pct": ppl_change_pct}


def _serialize_result(r) -> dict:
    from dataclasses import asdict
    return asdict(r)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/run", response_model=CreateTaskResponse)
def run_benchmark(req: BenchmarkRequest) -> CreateTaskResponse:
    """Start a benchmark task. Returns {task_id}."""
    if not Path(req.baseline_dir).exists():
        raise HTTPException(400, "Baseline directory not found")
    if req.compare_dir and not Path(req.compare_dir).exists():
        raise HTTPException(400, "Compare directory not found")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.benchmark import benchmark_model, benchmark_gguf_model, ComparisonResult

        def _bench(model_path: str, label: str = "model"):
            """Route to GGUF or MLX benchmark based on path."""
            p = Path(model_path)
            is_gguf = (p.is_file() and p.suffix.lower() == ".gguf")
            if is_gguf:
                return benchmark_gguf_model(model_path, verbose=False)
            return benchmark_model(
                model_path,
                n_perplexity_texts=req.num_ppl_texts,
                generation_tokens=req.num_tokens,
                verbose=False,
                enable_dsr=req.enable_dsr,
                dsr_budget=req.dsr_budget,
            )

        if progress_callback:
            progress_callback("Benchmarking baseline model...", 0.05)

        baseline = _bench(req.baseline_dir, "baseline")

        if req.compare_dir:
            if progress_callback:
                progress_callback("Benchmarking optimized model...", 0.55)

            optimized = _bench(req.compare_dir, "optimized")

            cmp = ComparisonResult(baseline=baseline, optimized=optimized)
            guidance = _generate_guidance(
                disk_reduction_pct=cmp.disk_reduction_pct,
                memory_reduction_pct=cmp.memory_reduction_pct,
                speed_improvement_pct=cmp.speed_improvement_pct,
                ppl_delta=cmp.perplexity_delta,
                baseline_ppl=baseline.perplexity,
            )

            if progress_callback:
                progress_callback("Done", 1.0)

            return {
                "mode": "comparison",
                "baseline": _serialize_result(baseline),
                "optimized": _serialize_result(optimized),
                "comparison": {
                    "disk_reduction_pct": round(cmp.disk_reduction_pct, 1),
                    "memory_reduction_pct": round(cmp.memory_reduction_pct, 1),
                    "speed_improvement_pct": round(cmp.speed_improvement_pct, 1),
                    "perplexity_delta": round(cmp.perplexity_delta, 3),
                    "guidance": guidance,
                },
            }
        else:
            if progress_callback:
                progress_callback("Done", 1.0)
            return {
                "mode": "single",
                "baseline": _serialize_result(baseline),
                "optimized": None,
                "comparison": None,
            }

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


# ---------------------------------------------------------------------------
# Batch benchmark
# ---------------------------------------------------------------------------

class BatchBenchmarkItem(BaseModel):
    model_dir: str
    label: str = ""
    num_tokens: int = 100
    num_ppl_texts: int = 3


class BatchBenchmarkRequest(BaseModel):
    items: list[BatchBenchmarkItem]


@router.post("/batch", response_model=CreateTaskResponse)
def run_batch_benchmark(req: BatchBenchmarkRequest) -> CreateTaskResponse:
    """Run benchmark on multiple models in sequence. Returns {task_id}."""
    if not req.items:
        raise HTTPException(400, "At least one benchmark item is required")

    for item in req.items:
        if not Path(item.model_dir).exists():
            raise HTTPException(400, "Model directory not found")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.benchmark import benchmark_model, benchmark_gguf_model

        results = []
        total = len(req.items)

        for i, item in enumerate(req.items):
            label = item.label or Path(item.model_dir).name

            if progress_callback:
                progress_callback(f"Benchmarking {label} ({i + 1}/{total})...", i / total)

            try:
                p = Path(item.model_dir)
                is_gguf = p.is_file() and p.suffix.lower() == ".gguf"
                if is_gguf:
                    result = benchmark_gguf_model(item.model_dir, verbose=False)
                else:
                    result = benchmark_model(
                        item.model_dir,
                        n_perplexity_texts=item.num_ppl_texts,
                        generation_tokens=item.num_tokens,
                        verbose=False,
                    )
                results.append({
                    "label": label,
                    "model_dir": item.model_dir,
                    "success": True,
                    "result": _serialize_result(result),
                })
            except Exception as e:
                logger.warning("Benchmark item %s failed: %s", label, e)
                results.append({
                    "label": label,
                    "model_dir": item.model_dir,
                    "success": False,
                    "error": str(e),
                })

        if progress_callback:
            progress_callback("Batch benchmark complete", 1.0)

        return {"results": results, "total": total}
