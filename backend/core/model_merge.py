# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model merge engine — combine multiple models into one.

Supports:
- Linear: weighted average of parameters
- SLERP: spherical linear interpolation (2 models)
- TIES: sparsification + sign consensus (multi-model)
- Task Arithmetic: base + λ × Σ(task vectors)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class MergeConfig:
    model_dirs: list[str]
    strategy: str = "linear"
    weights: list[float] = field(default_factory=list)
    base_model_dir: str = ""
    density: float = 0.5
    output_dir: str = ""


@dataclass
class MergeResult:
    success: bool
    output_dir: str = ""
    strategy: str = ""
    model_names: list[str] = field(default_factory=list)
    merged_params: int = 0
    merged_size_bytes: int = 0
    duration_seconds: float = 0.0
    error: str = ""


def run_merge(
    config: MergeConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> MergeResult:
    """Merge multiple models into one using the specified strategy."""
    t0 = time.time()

    try:
        import mlx.core as mx
    except ImportError:
        return MergeResult(success=False, error="MLX required: pip install mlx")

    # Validate inputs
    if len(config.model_dirs) < 2:
        return MergeResult(success=False, error="At least 2 models required for merging")

    if config.strategy == "slerp" and len(config.model_dirs) != 2:
        return MergeResult(success=False, error="SLERP requires exactly 2 models")

    if config.strategy == "task_arithmetic" and not config.base_model_dir:
        return MergeResult(success=False, error="Task arithmetic requires a base_model_dir")

    # Resolve weights
    weights = config.weights
    if not weights:
        weights = [1.0 / len(config.model_dirs)] * len(config.model_dirs)
    elif len(weights) != len(config.model_dirs):
        return MergeResult(success=False, error="Number of weights must match number of models")

    # Normalize weights
    w_sum = sum(weights)
    if w_sum > 0:
        weights = [w / w_sum for w in weights]

    model_names = [Path(d).name for d in config.model_dirs]

    # Resolve output dir
    output_dir = config.output_dir
    if not output_dir:
        base = Path(config.model_dirs[0]).parent
        output_dir = str(base / f"merged-{config.strategy}-{'_'.join(model_names[:2])}")

    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback("Loading model weights...", 0.1)

    # Load all model safetensors
    try:
        from .weight_loader import load_safetensors_index

        model_weights: list[dict[str, mx.array]] = []
        for i, model_dir in enumerate(config.model_dirs):
            if progress_callback:
                progress_callback(
                    f"Loading model {i+1}/{len(config.model_dirs)}: {model_names[i]}",
                    0.1 + 0.3 * (i / len(config.model_dirs)),
                )

            # Load safetensors weights
            wt = _load_model_weights(model_dir)
            model_weights.append(wt)

    except Exception as e:
        return MergeResult(success=False, error=f"Failed to load models: {e}")

    if progress_callback:
        progress_callback("Merging weights...", 0.5)

    # Merge based on strategy
    try:
        if config.strategy == "linear":
            merged = _merge_linear(model_weights, weights, progress_callback)
        elif config.strategy == "slerp":
            merged = _merge_slerp(model_weights[0], model_weights[1], weights[1], progress_callback)
        elif config.strategy == "ties":
            merged = _merge_ties(model_weights, weights, config.density, progress_callback)
        elif config.strategy == "task_arithmetic":
            base_weights = _load_model_weights(config.base_model_dir)
            task_vectors = [
                {k: w[k] - base_weights[k] for k in w if k in base_weights}
                for w in model_weights
            ]
            merged = _merge_task_arithmetic(base_weights, task_vectors, weights, progress_callback)
        else:
            return MergeResult(success=False, error=f"Unknown strategy: {config.strategy}")
    except Exception as e:
        return MergeResult(success=False, error=f"Merge failed: {e}")

    # Save merged weights
    if progress_callback:
        progress_callback("Saving merged model...", 0.9)

    try:
        mx.save_safetensors(os.path.join(output_dir, "model.safetensors"), merged)

        # Copy config and tokenizer from first model
        for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "tokenizer.model"):
            src = os.path.join(config.model_dirs[0], fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

        # Write merge metadata
        meta = {
            "merge": {
                "strategy": config.strategy,
                "models": model_names,
                "weights": weights,
                "density": config.density if config.strategy == "ties" else None,
            }
        }
        with open(os.path.join(output_dir, "merge_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    except Exception as e:
        return MergeResult(success=False, error=f"Failed to save merged model: {e}")

    # Calculate stats
    total_params = sum(v.size for v in merged.values())
    total_bytes = sum(v.nbytes for v in merged.values())

    duration = time.time() - t0
    if progress_callback:
        progress_callback("Merge complete!", 1.0)

    return MergeResult(
        success=True,
        output_dir=output_dir,
        strategy=config.strategy,
        model_names=model_names,
        merged_params=total_params,
        merged_size_bytes=total_bytes,
        duration_seconds=round(duration, 2),
    )


def _load_model_weights(model_dir: str) -> dict:
    """Load all safetensors weights from a model directory."""
    import mlx.core as mx

    weights = {}
    model_path = Path(model_dir)

    # Find all safetensors files
    st_files = sorted(model_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    for sf in st_files:
        w = mx.load(str(sf))
        weights.update(w)

    return weights


def _merge_linear(
    models: list[dict],
    weights: list[float],
    progress_callback: Callable | None = None,
) -> dict:
    """Weighted average of model parameters."""
    import mlx.core as mx

    # Use first model's keys as reference
    keys = list(models[0].keys())
    merged = {}
    total = len(keys)

    for i, key in enumerate(keys):
        tensors = [m[key] for m in models if key in m]
        ws = weights[:len(tensors)]

        if len(tensors) == 1:
            merged[key] = tensors[0]
        else:
            result = ws[0] * tensors[0]
            for w, t in zip(ws[1:], tensors[1:]):
                result = result + w * t
            merged[key] = result

        if progress_callback and (i + 1) % max(1, total // 20) == 0:
            progress_callback(
                f"Merging: {i+1}/{total} tensors",
                0.5 + 0.35 * ((i + 1) / total),
            )

    return merged


def _merge_slerp(
    model_a: dict,
    model_b: dict,
    t: float,
    progress_callback: Callable | None = None,
) -> dict:
    """Spherical linear interpolation between two models."""
    import mlx.core as mx

    keys = list(model_a.keys())
    merged = {}
    total = len(keys)

    for i, key in enumerate(keys):
        if key not in model_b:
            merged[key] = model_a[key]
            continue

        a = model_a[key].reshape(-1).astype(mx.float32)
        b = model_b[key].reshape(-1).astype(mx.float32)

        # Normalize
        a_norm = mx.sqrt(mx.sum(a * a))
        b_norm = mx.sqrt(mx.sum(b * b))

        if a_norm.item() < 1e-8 or b_norm.item() < 1e-8:
            merged[key] = ((1 - t) * model_a[key] + t * model_b[key])
            continue

        a_unit = a / a_norm
        b_unit = b / b_norm

        # Cosine similarity
        cos_sim = mx.clip(mx.sum(a_unit * b_unit), -1.0, 1.0)
        omega = mx.arccos(cos_sim)

        sin_omega = mx.sin(omega)
        if sin_omega.item() < 1e-8:
            # Vectors nearly parallel — linear interpolation
            merged[key] = ((1 - t) * model_a[key] + t * model_b[key])
        else:
            # SLERP
            scale = (a_norm * (1 - t) + b_norm * t)
            result = (mx.sin((1 - t) * omega) / sin_omega) * a_unit + \
                     (mx.sin(t * omega) / sin_omega) * b_unit
            merged[key] = (result * scale).reshape(model_a[key].shape).astype(model_a[key].dtype)

        if progress_callback and (i + 1) % max(1, total // 20) == 0:
            progress_callback(
                f"SLERP: {i+1}/{total} tensors",
                0.5 + 0.35 * ((i + 1) / total),
            )

    return merged


def _merge_ties(
    models: list[dict],
    weights: list[float],
    density: float,
    progress_callback: Callable | None = None,
) -> dict:
    """TIES merging: Trim + Elect Sign + Disjoint Merge."""
    import mlx.core as mx

    keys = list(models[0].keys())
    merged = {}
    total = len(keys)

    for i, key in enumerate(keys):
        tensors = [m[key] for m in models if key in m]

        if len(tensors) == 1:
            merged[key] = tensors[0]
            continue

        # Step 1: Compute task vectors (delta from mean)
        mean_t = sum(t for t in tensors) * (1.0 / len(tensors))
        deltas = [t - mean_t for t in tensors]

        # Step 2: Trim (sparsify) — keep only top-k% by magnitude
        trimmed = []
        for d in deltas:
            flat = mx.abs(d.reshape(-1))
            k = max(1, int(flat.size * density))
            threshold = mx.sort(flat)[-k]
            mask = mx.abs(d) >= threshold
            trimmed.append(d * mask)

        # Step 3: Elect sign — majority vote (exclude zeros to avoid bias)
        signs = mx.stack([mx.sign(t) for t in trimmed])
        nonzero_mask = (signs != 0).astype(signs.dtype)
        sign_sum = mx.sum(signs * nonzero_mask, axis=0)
        elected_sign = mx.sign(sign_sum)

        # Step 4: Disjoint merge — average magnitudes that agree with elected sign
        result = mx.zeros_like(mean_t)
        count = mx.zeros_like(mean_t)

        for j, t in enumerate(trimmed):
            agree = (mx.sign(t) == elected_sign)
            result = result + mx.where(agree, t * weights[j], mx.zeros_like(t))
            count = count + mx.where(agree, mx.ones_like(t) * weights[j], mx.zeros_like(t))

        safe_count = mx.where(count > 0, count, mx.ones_like(count))
        merged[key] = mean_t + result / safe_count

        if progress_callback and (i + 1) % max(1, total // 20) == 0:
            progress_callback(
                f"TIES: {i+1}/{total} tensors",
                0.5 + 0.35 * ((i + 1) / total),
            )

    return merged


def _merge_task_arithmetic(
    base: dict,
    task_vectors: list[dict],
    weights: list[float],
    progress_callback: Callable | None = None,
) -> dict:
    """Task Arithmetic: base + Σ(weight_i × task_vector_i)."""
    import mlx.core as mx

    keys = list(base.keys())
    merged = {}
    total = len(keys)

    for i, key in enumerate(keys):
        result = base[key].astype(mx.float32)

        for j, tv in enumerate(task_vectors):
            if key in tv:
                result = result + weights[j] * tv[key].astype(mx.float32)

        merged[key] = result.astype(base[key].dtype)

        if progress_callback and (i + 1) % max(1, total // 20) == 0:
            progress_callback(
                f"Task Arithmetic: {i+1}/{total} tensors",
                0.5 + 0.35 * ((i + 1) / total),
            )

    return merged
