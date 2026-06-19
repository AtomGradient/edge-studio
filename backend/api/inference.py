# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Inference tracing endpoints."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.schemas.common import CreateTaskResponse
from backend.schemas.inference import TraceRequest
from backend.services.model_manager import manager
from backend.services.mlx_runtime_gate import mlx_runtime_gate
from backend.services.serialization import serialize_for_json
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/model", tags=["inference"])


def _is_vlm(model_dir: str) -> bool:
    """Return True if model has vision_config (is a VLM)."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        return "vision_config" in cfg
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def _run_vlm_trace(
    model_dir: str,
    prompt: str,
    image_b64: str | None,
    max_tokens: int,
    temperature: float,
    progress_callback=None,
) -> dict:
    """Simplified VLM trace using mlx_vlm.

    Returns a dict compatible with the TraceResponse format.
    Layer-level analysis is not available for VLM models — steps contain
    generated tokens only (probability = 1.0 placeholder).
    """
    import base64
    import tempfile

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    if progress_callback:
        progress_callback("Loading VLM model...", 0.05)

    # Import here to avoid loading at module level
    from mlx_vlm import load as vlm_load
    model, processor = vlm_load(model_dir)
    config = load_config(model_dir)

    model_name = os.path.basename(model_dir.rstrip("/"))

    # Prepare image
    image_path = None
    images: list[str] = []
    num_images = 0
    if image_b64:
        img_data = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            image_path = tmp.name
        images = [image_path]
        num_images = 1

    formatted_prompt = apply_chat_template(
        processor, config, prompt, num_images=num_images
    )

    if progress_callback:
        progress_callback("Generating (VLM simplified trace)...", 0.2)

    t_start = time.time()
    tokens: list[str] = []
    full_text = ""

    # Try streaming for token-level granularity
    try:
        from mlx_vlm import stream_generate
        for i, chunk in enumerate(stream_generate(
            model, processor, formatted_prompt, images,
            max_tokens=max_tokens, temperature=temperature,
        )):
            tok = chunk.text if hasattr(chunk, "text") else str(chunk)
            tokens.append(tok)
            full_text += tok
            if progress_callback and i % 10 == 0:
                pct = min(0.2 + (i / max(max_tokens, 1)) * 0.7, 0.9)
                progress_callback(f"Generating token {i}...", pct)
    except (ImportError, AttributeError, TypeError):
        # Fallback: non-streaming
        from mlx_vlm import generate as vlm_generate
        result = vlm_generate(
            model, processor, formatted_prompt, images,
            max_tokens=max_tokens, temperature=temperature, verbose=False,
        )
        full_text = result if isinstance(result, str) else str(result)
        # Split on spaces as rough token approximation
        tokens = full_text.split()

    t_total = time.time() - t_start

    # Cleanup temp image
    if image_path:
        try:
            os.unlink(image_path)
        except OSError:
            pass

    if progress_callback:
        progress_callback("Done", 1.0)

    # Build steps: one entry per token (no layer analysis)
    steps = [
        {
            "step_idx": i,
            "token_id": i,
            "token_str": tok,
            "top_k_token_ids": [i],
            "top_k_probs": [1.0],
            "top_k_token_strs": [tok],
            "chosen_rank": 0,
            "chosen_prob": 1.0,
            "final_hidden_norm": 0.0,
            "layers": [],
        }
        for i, tok in enumerate(tokens)
    ]

    return {
        "prompt": prompt,
        "prompt_token_ids": [],
        "prompt_tokens": ["[VLM — layer tracing not available]"],
        "temperature": temperature,
        "top_k": 0,
        "top_p": 1.0,
        "model_dir": model_dir,
        "model_name": model_name,
        "num_layers": 0,
        "num_heads": 0,
        "hidden_size": 0,
        "steps": steps,
        "generated_text": full_text,
        "total_time_seconds": t_total,
        "prefill_time_seconds": 0.0,
        "prefill_layer_traces": [],
        "enable_timing": False,
    }


@router.post("/{model_id}/trace", response_model=CreateTaskResponse)
def run_trace(model_id: str, req: TraceRequest) -> CreateTaskResponse:
    loaded = manager.get_model(model_id)
    if not loaded:
        raise HTTPException(404, "Model not loaded")

    task_id = task_manager.create_task()

    def _run(progress_callback=None):
        with mlx_runtime_gate("inference.trace"):
            # Adapter: core uses (step, total, msg), TaskManager provides (msg, percent)
            def _progress(step: int, total: int, msg: str):
                if progress_callback:
                    progress_callback(msg, step / max(total, 1))

            # VLM path: only when user actually provides an image
            if req.image_b64:
                from backend.core.universal_tracer import run_vlm_universal_trace

                result = run_vlm_universal_trace(
                    model_path=loaded.model_dir,
                    prompt=req.prompt,
                    image_b64=req.image_b64,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    enable_thinking=req.enable_thinking,
                    enable_timing=req.enable_timing,
                    capture_attention=req.capture_attention,
                    capture_moe_routing=req.capture_moe_routing,
                    progress_callback=_progress,
                )
                manager.store_trace(model_id, result)
                return _serialize_trace(result)

            if req.use_legacy_tracer:
                from backend.core.inference_tracer import run_inference_trace

                result = run_inference_trace(
                    model_dir=loaded.model_dir,
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    enable_thinking=req.enable_thinking,
                    enable_timing=req.enable_timing,
                    progress_callback=_progress,
                )
            else:
                from backend.core.universal_tracer import run_universal_trace

                result = run_universal_trace(
                    model_path=loaded.model_dir,
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    enable_thinking=req.enable_thinking,
                    enable_timing=req.enable_timing,
                    capture_attention=req.capture_attention,
                    capture_moe_routing=req.capture_moe_routing,
                    progress_callback=_progress,
                )

            manager.store_trace(model_id, result)
            return _serialize_trace(result)

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)


@router.get("/{model_id}/trace/result", response_model=dict[str, Any])
def get_trace_result(model_id: str) -> dict[str, Any]:
    trace = manager.get_trace(model_id)
    if not trace:
        raise HTTPException(404, "No trace available — run inference first")
    return _serialize_trace(trace)


def _serialize_trace(trace) -> dict:
    """Convert InferenceTrace dataclass or dict to JSON-safe dict."""
    return serialize_for_json(trace)
