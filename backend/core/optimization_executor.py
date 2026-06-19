# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Optimization executor — unified API for pruning, quantization, and vocab reduction.

All operations are implemented natively via native_ops.py (Python + MLX + safetensors).
No external script dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of a single optimization operation."""
    operation: str
    success: bool
    output_dir: str
    message: str
    duration_seconds: float = 0.0
    original_size_bytes: int = 0
    result_size_bytes: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def saving_bytes(self) -> int:
        return max(0, self.original_size_bytes - self.result_size_bytes)


@dataclass
class PipelineResult:
    """Result of a multi-step optimization pipeline."""
    steps: list[ExecutionResult] = field(default_factory=list)
    final_output_dir: str = ""

    @property
    def all_success(self) -> bool:
        return all(s.success for s in self.steps)

    @property
    def total_saving_bytes(self) -> int:
        if not self.steps:
            return 0
        return max(0, self.steps[0].original_size_bytes - self.steps[-1].result_size_bytes)


ProgressCallback = Callable[[str, float], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_size(model_dir: str) -> int:
    """Total size of safetensors files in a model directory."""
    total = 0
    for p in Path(model_dir).glob("*.safetensors"):
        total += p.stat().st_size
    return total


def _make_output_dir(model_dir: str, suffix: str) -> str:
    """Build an output directory path from a model dir and suffix.

    Uses HuggingFace-style hyphen separator (e.g. Qwen3-0.6B-4bit).
    If the path already exists, appends a timestamp to avoid conflicts.
    """
    p = Path(model_dir)
    base = p.parent / f"{p.name}-{suffix}"
    if base.exists():
        ts = time.strftime("%Y%m%dT%H%M%S")
        return str(p.parent / f"{p.name}-{suffix}-{ts}")
    return str(base)


# ---------------------------------------------------------------------------
# Public execution functions
# ---------------------------------------------------------------------------

def execute_neuron_pruning(
    model_dir: str,
    threshold: float = 0.1,
    profile_path: Optional[str] = None,
    protected_layers: Optional[list[int]] = None,
    max_reduction: float = 0.5,
    group_size: int = 64,
    output_suffix: str = "neuron-pruned",
    progress_cb: Optional[ProgressCallback] = None,
) -> ExecutionResult:
    """Prune MLP neurons per layer using activation profile."""
    from .activation_loader import find_profile_files, load_profile
    from .native_ops import apply_neuron_pruning

    t0 = time.time()
    output_dir = _make_output_dir(model_dir, output_suffix)

    # Auto-detect profile
    if not profile_path:
        profiles = find_profile_files(model_dir)
        if not profiles:
            return ExecutionResult(
                operation="neuron_pruning", success=False,
                output_dir="",
                message="No activation profile found. Run activation profiling first.",
            )
        profile_path = profiles[0]

    try:
        profile = load_profile(profile_path)
    except Exception as e:
        return ExecutionResult(
            operation="neuron_pruning", success=False,
            output_dir="", message=f"Failed to load profile: {e}",
        )

    orig_size = _model_size(model_dir)
    success, msg, per_layer_sizes = apply_neuron_pruning(
        model_dir=model_dir,
        output_dir=output_dir,
        activation_profile=profile,
        threshold=threshold,
        max_reduction=max_reduction,
        group_size=group_size,
        protected_layers=protected_layers or [],
        progress_cb=progress_cb,
    )
    result_size = _model_size(output_dir) if success and Path(output_dir).exists() else 0

    return ExecutionResult(
        operation="neuron_pruning",
        success=success,
        output_dir=output_dir if success else "",
        message=msg,
        duration_seconds=time.time() - t0,
        original_size_bytes=orig_size,
        result_size_bytes=result_size,
        details={
            "threshold": threshold,
            "max_reduction": max_reduction,
            "protected_layers": protected_layers or [],
            "per_layer_intermediate_sizes": per_layer_sizes,
        },
    )


def execute_layer_pruning(
    model_dir: str,
    layers_to_remove: list[int],
    component: str = "text",  # kept for API compatibility
    output_suffix: str = "layer-pruned",
    progress_cb: Optional[ProgressCallback] = None,
) -> ExecutionResult:
    """Remove entire transformer layers and renumber remaining ones."""
    from .native_ops import apply_layer_pruning

    t0 = time.time()
    output_dir = _make_output_dir(model_dir, output_suffix)
    orig_size = _model_size(model_dir)

    success, msg = apply_layer_pruning(
        model_dir=model_dir,
        output_dir=output_dir,
        layers_to_remove=layers_to_remove,
        progress_cb=progress_cb,
    )
    result_size = _model_size(output_dir) if success and Path(output_dir).exists() else 0

    return ExecutionResult(
        operation="layer_pruning",
        success=success,
        output_dir=output_dir if success else "",
        message=msg,
        duration_seconds=time.time() - t0,
        original_size_bytes=orig_size,
        result_size_bytes=result_size,
        details={"layers_removed": layers_to_remove, "component": component},
    )


def execute_vocab_pruning(
    model_dir: str,
    output_suffix: str = "vocab-pruned",
    progress_cb: Optional[ProgressCallback] = None,
) -> ExecutionResult:
    """Prune embedding and lm_head to the tokenizer's actual vocabulary size."""
    from .native_ops import apply_vocab_pruning

    t0 = time.time()
    output_dir = _make_output_dir(model_dir, output_suffix)
    orig_size = _model_size(model_dir)

    success, msg = apply_vocab_pruning(
        model_dir=model_dir,
        output_dir=output_dir,
        progress_cb=progress_cb,
    )
    result_size = _model_size(output_dir) if success and Path(output_dir).exists() else 0

    return ExecutionResult(
        operation="vocab_pruning",
        success=success,
        output_dir=output_dir if success else "",
        message=msg,
        duration_seconds=time.time() - t0,
        original_size_bytes=orig_size,
        result_size_bytes=result_size,
    )


def execute_quantization(
    model_dir: str,
    bits: int = 4,
    group_size: int = 64,
    quantize_embeddings: bool = False,  # kept for API compatibility
    output_suffix: str = "",
    progress_cb: Optional[ProgressCallback] = None,
) -> ExecutionResult:
    """Quantize model weights using mlx_lm.convert."""
    from .native_ops import apply_quantization

    t0 = time.time()
    # Auto-generate HuggingFace-style suffix: e.g. "4bit", "8bit-g128"
    if not output_suffix:
        output_suffix = f"{bits}bit" if group_size == 64 else f"{bits}bit-g{group_size}"
    output_dir = _make_output_dir(model_dir, output_suffix)
    orig_size = _model_size(model_dir)

    success, msg = apply_quantization(
        model_dir=model_dir,
        output_dir=output_dir,
        bits=bits,
        group_size=group_size,
        progress_cb=progress_cb,
    )
    result_size = _model_size(output_dir) if success and Path(output_dir).exists() else 0

    return ExecutionResult(
        operation="quantization",
        success=success,
        output_dir=output_dir if success else "",
        message=msg,
        duration_seconds=time.time() - t0,
        original_size_bytes=orig_size,
        result_size_bytes=result_size,
        details={"bits": bits, "group_size": group_size},
    )


def execute_embedding_quantization(
    model_dir: str,
    output_suffix: str = "emb-quant",
    progress_cb: Optional[ProgressCallback] = None,
) -> ExecutionResult:
    """Embedding-only quantization — delegated to full quantize with embed flag."""
    # mlx_lm.convert quantizes embeddings by default; this is an alias.
    return execute_quantization(
        model_dir=model_dir,
        bits=4,
        group_size=64,
        output_suffix=output_suffix,
        progress_cb=progress_cb,
    )


# ---------------------------------------------------------------------------
# Pipeline executor
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    """A single step in an optimization pipeline."""
    operation: str
    params: dict[str, Any] = field(default_factory=dict)


def execute_pipeline(
    model_dir: str,
    steps: list[PipelineStep],
    progress_cb: Optional[ProgressCallback] = None,
) -> PipelineResult:
    """Execute a sequence of optimization steps, chaining outputs."""
    result = PipelineResult()
    current_dir = model_dir

    executors = {
        "neuron_pruning": execute_neuron_pruning,
        "layer_pruning": execute_layer_pruning,
        "vocab_pruning": execute_vocab_pruning,
        "quantization": execute_quantization,
        "embedding_quantization": execute_embedding_quantization,
    }

    for i, step in enumerate(steps):
        executor = executors.get(step.operation)
        if not executor:
            result.steps.append(ExecutionResult(
                operation=step.operation, success=False,
                output_dir=current_dir,
                message=f"Unknown operation: {step.operation}",
            ))
            break

        if progress_cb:
            progress_cb(f"Step {i+1}/{len(steps)}: {step.operation}", i / len(steps))

        kwargs = dict(step.params)
        kwargs["model_dir"] = current_dir
        kwargs["progress_cb"] = progress_cb

        step_result = executor(**kwargs)
        result.steps.append(step_result)

        if not step_result.success:
            break

        current_dir = step_result.output_dir

    result.final_output_dir = current_dir

    if progress_cb:
        progress_cb("Pipeline complete", 1.0)

    return result


# ---------------------------------------------------------------------------
# Utility: list available operations for a model
# ---------------------------------------------------------------------------

def available_operations(model_dir: str) -> list[dict[str, Any]]:
    """List operations available for a given model (all native, always available)."""
    import json
    from pathlib import Path

    ops = ["neuron_pruning", "layer_pruning", "vocab_pruning", "quantization", "embedding_quantization"]
    results = []

    # neuron_pruning requires an activation profile
    from .activation_loader import find_profile_files
    profiles = find_profile_files(model_dir)

    for op in ops:
        available = True
        note = ""
        if op == "neuron_pruning" and not profiles:
            available = False
            note = "Requires activation profile (run profiling first)"
        results.append({
            "operation": op,
            "available": available,
            "note": note,
            "implementation": "native_mlx",
        })

    return results
