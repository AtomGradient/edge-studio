# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model comparison endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schemas.common import CreateTaskResponse
from backend.schemas.inference import CompareRequest
from backend.services.model_manager import manager
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api", tags=["comparison"])


def _serialize_comparison(result) -> dict:
    """Serialize a ModelComparisonResult dataclass."""
    out: dict = {}

    # Architecture diff
    if result.arch_diff:
        diff = result.arch_diff
        out["arch_diff"] = {
            "model_a_name": diff.model_a_name,
            "model_b_name": diff.model_b_name,
            "rows": [
                {
                    "field_name": r.field_name,
                    "model_a_value": r.model_a_value,
                    "model_b_value": r.model_b_value,
                    "is_different": r.is_different,
                }
                for r in diff.rows
            ],
        }
    else:
        out["arch_diff"] = None

    # Latency profiles
    for key, profile in [("latency_a", result.latency_a), ("latency_b", result.latency_b)]:
        if profile:
            out[key] = {
                "model_name": profile.model_name,
                "prefill_layer_attn_ms": list(profile.prefill_layer_attn_ms),
                "prefill_layer_mlp_ms": list(profile.prefill_layer_mlp_ms),
                "prefill_total_ms": profile.prefill_total_ms,
                "decode_layer_attn_ms": list(profile.decode_layer_attn_ms),
                "decode_layer_mlp_ms": list(profile.decode_layer_mlp_ms),
                "decode_total_ms": profile.decode_total_ms,
                "decode_steps": profile.decode_steps,
                "tokens_per_second": profile.tokens_per_second,
            }
        else:
            out[key] = None

    # Bottleneck layers
    for key, bns in [("bottlenecks_a", result.bottlenecks_a), ("bottlenecks_b", result.bottlenecks_b)]:
        out[key] = [
            {
                "layer_idx": b.layer_idx,
                "attn_ms": b.attn_ms,
                "mlp_ms": b.mlp_ms,
                "total_ms": b.total_ms,
                "pct_of_total": b.pct_of_total,
                "bottleneck_type": b.bottleneck_type,
            }
            for b in bns
        ]

    return out


@router.post("/compare", response_model=CreateTaskResponse)
def compare_models(req: CompareRequest):
    a = manager.get_model(req.model_id_a)
    b = manager.get_model(req.model_id_b)
    if not a:
        raise HTTPException(404, f"Model A not loaded: {req.model_id_a}")
    if not b:
        raise HTTPException(404, f"Model B not loaded: {req.model_id_b}")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        from backend.core.universal_tracer import run_universal_trace
        from backend.core.model_comparator import run_comparison

        # Run trace A
        if progress_callback:
            progress_callback("Running inference on Model A...", 0.0)

        def _prog_a(step, total, msg):
            if progress_callback:
                progress_callback(f"Model A: {msg}", step / max(total, 1) * 0.4)

        trace_a = run_universal_trace(
            model_path=a.model_dir,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            enable_timing=req.enable_timing,
            progress_callback=_prog_a,
        )
        manager.store_trace(req.model_id_a, trace_a)

        # Run trace B
        if progress_callback:
            progress_callback("Running inference on Model B...", 0.45)

        def _prog_b(step, total, msg):
            if progress_callback:
                progress_callback(f"Model B: {msg}", 0.45 + step / max(total, 1) * 0.4)

        trace_b = run_universal_trace(
            model_path=b.model_dir,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            enable_timing=req.enable_timing,
            progress_callback=_prog_b,
        )
        manager.store_trace(req.model_id_b, trace_b)

        # Compare
        if progress_callback:
            progress_callback("Computing comparison...", 0.9)

        result = run_comparison(
            arch_a=a.architecture,
            arch_b=b.architecture,
            trace_a=trace_a,
            trace_b=trace_b,
        )

        return _serialize_comparison(result)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
