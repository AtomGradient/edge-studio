# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Knowledge distillation engine — teacher → student knowledge transfer.

Supports:
- Offline distillation with KL divergence + cross-entropy loss
- TAID (Time-Adaptive Interpolation Distillation, ICLR 2025 Spotlight)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class DistillConfig:
    teacher_dir: str
    student_dir: str
    dataset_path: str
    output_dir: str = ""
    mode: str = "offline"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    temperature: float = 2.0
    alpha: float = 0.5
    max_samples: int = 0


@dataclass
class DistillResult:
    success: bool
    output_dir: str = ""
    teacher_name: str = ""
    student_name: str = ""
    num_epochs: int = 0
    total_steps: int = 0
    final_loss: float = 0.0
    final_kl_loss: float = 0.0
    final_ce_loss: float = 0.0
    duration_seconds: float = 0.0
    dataset_samples: int = 0
    error: str = ""
    warning: str = ""
    loss_history: list[dict] = field(default_factory=list)


def _load_dataset(path: str, max_samples: int = 0) -> list[dict]:
    """Load JSONL or Parquet dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[dict] = []

    if p.suffix == ".jsonl":
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    elif p.suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(str(p))
            samples = table.to_pylist()
        except ImportError:
            raise ImportError("pyarrow required for parquet datasets: pip install pyarrow")
    else:
        raise ValueError(f"Unsupported dataset format: {p.suffix} (use .jsonl or .parquet)")

    if max_samples > 0:
        samples = samples[:max_samples]

    return samples


def _get_text_field(sample: dict) -> str:
    """Extract text from a dataset sample (supports common formats)."""
    for key in ("text", "content", "instruction", "prompt", "input"):
        if key in sample and sample[key]:
            return str(sample[key])
    # Fallback: concatenate all string values
    parts = [str(v) for v in sample.values() if isinstance(v, str) and v]
    return " ".join(parts) if parts else ""


def _run_data_distillation(
    config: DistillConfig,
    teacher_model, teacher_tokenizer,
    student_model, student_tokenizer,
    samples: list[dict],
    t0: float,
    progress_callback: Callable[[str, float], None] | None = None,
) -> DistillResult:
    """Data distillation: teacher generates text → student learns from teacher's output.

    Used when teacher and student have different vocab/tokenizer (cross-family).
    No logit comparison needed — student trains on teacher's generated text with CE loss.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import generate as mlx_generate

    teacher_name = Path(config.teacher_dir).name
    student_name = Path(config.student_dir).name
    output_dir = _resolve_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Teacher generates responses for each sample
    if progress_callback:
        progress_callback("Data distillation: teacher generating responses...", 0.1)

    training_pairs: list[tuple[mx.array, mx.array]] = []
    for i, sample in enumerate(samples):
        prompt = _get_text_field(sample)
        if not prompt:
            continue

        if progress_callback:
            progress_callback(
                f"Teacher generating {i+1}/{len(samples)}...",
                0.1 + 0.3 * (i / len(samples)),
            )

        # Teacher generates response
        try:
            import inspect
            gen_params = inspect.signature(mlx_generate).parameters
            temp_kwarg = "temp" if "temp" in gen_params else "temperature"
            teacher_response = mlx_generate(
                teacher_model, teacher_tokenizer,
                prompt=prompt,
                max_tokens=256,
                verbose=False,
                **{temp_kwarg: 0.7},
            )
            if not isinstance(teacher_response, str):
                teacher_response = getattr(teacher_response, 'text', str(teacher_response))
        except Exception as e:
            logger.warning(f"Teacher generation failed for sample {i}: {e}")
            continue

        # Tokenize teacher's response with student's tokenizer
        full_text = prompt + " " + teacher_response
        tokens = student_tokenizer.encode(full_text)
        if len(tokens) > 3:
            tokens = tokens[:512]
            input_ids = mx.array(tokens[:-1])
            target_ids = mx.array(tokens[1:])
            training_pairs.append((input_ids, target_ids))

    if not training_pairs:
        return DistillResult(success=False, error="Teacher generated no valid training data")

    if progress_callback:
        progress_callback(f"Training student on {len(training_pairs)} teacher outputs...", 0.45)

    # Phase 2: Train student on teacher's outputs (standard CE loss)
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    total_steps = config.num_epochs * ((len(training_pairs) + config.batch_size - 1) // config.batch_size)
    step = 0
    loss_history: list[dict] = []
    final_loss = 0.0

    def ce_loss(student_model, input_ids, target_ids):
        logits = student_model(input_ids[None])
        return mx.mean(nn.losses.cross_entropy(logits[0], target_ids))

    loss_and_grad = nn.value_and_grad(student_model, lambda m, x, y: ce_loss(m, x, y))

    for epoch in range(config.num_epochs):
        epoch_losses = []
        for i in range(0, len(training_pairs), config.batch_size):
            batch = training_pairs[i : i + config.batch_size]
            batch_loss = 0.0

            for input_ids, target_ids in batch:
                loss, grads = loss_and_grad(student_model, input_ids, target_ids)
                optimizer.update(student_model, grads)
                mx.eval(student_model.parameters(), optimizer.state)
                batch_loss += loss.item()

            avg_loss = batch_loss / len(batch)
            epoch_losses.append(avg_loss)
            step += 1
            pct = 0.45 + 0.45 * (step / max(total_steps, 1))

            loss_history.append({
                "epoch": epoch + 1,
                "step": step,
                "loss": round(avg_loss, 4),
                "kl_loss": 0.0,
                "ce_loss": round(avg_loss, 4),
            })

            if progress_callback:
                progress_callback(
                    f"Epoch {epoch+1}/{config.num_epochs}, Step {step}/{total_steps}, CE Loss: {avg_loss:.4f}",
                    pct,
                )

        final_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

    # Save
    if progress_callback:
        progress_callback("Saving distilled model...", 0.92)

    try:
        student_model.save_weights(os.path.join(output_dir, "model.safetensors"))
        import shutil
        for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "tokenizer.model"):
            src = os.path.join(config.student_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

        meta = {
            "distillation": {
                "teacher": teacher_name,
                "student": student_name,
                "mode": "data",
                "epochs": config.num_epochs,
                "dataset_samples": len(training_pairs),
                "final_loss": round(final_loss, 4),
                "note": "Cross-family data distillation (teacher generates text, student learns CE)",
            }
        }
        with open(os.path.join(output_dir, "distillation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        return DistillResult(success=False, error=f"Failed to save: {e}")

    duration = time.time() - t0
    if progress_callback:
        progress_callback("Data distillation complete!", 1.0)

    return DistillResult(
        success=True,
        output_dir=output_dir,
        teacher_name=teacher_name,
        student_name=student_name,
        num_epochs=config.num_epochs,
        total_steps=step,
        final_loss=round(final_loss, 4),
        final_kl_loss=0.0,
        final_ce_loss=round(final_loss, 4),
        duration_seconds=round(duration, 2),
        dataset_samples=len(training_pairs),
        warning=f"Cross-family distillation: vocab mismatch detected, used data distillation mode "
                f"(teacher generates → student learns text). Quality is good but slower than same-family KL distillation.",
        loss_history=loss_history,
    )


def _resolve_output_dir(config: DistillConfig) -> str:
    """Resolve output directory for distilled model."""
    if config.output_dir:
        return config.output_dir

    student_name = Path(config.student_dir).name
    teacher_name = Path(config.teacher_dir).name
    base = Path(config.student_dir).parent
    output = base / f"{student_name}-distilled-from-{teacher_name}"
    return str(output)


def run_distillation(
    config: DistillConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> DistillResult:
    """Run knowledge distillation from teacher to student model.

    Offline mode: pre-cache teacher logits → train student with KL divergence.
    """
    t0 = time.time()

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError:
        return DistillResult(
            success=False,
            error="MLX required: pip install mlx",
        )

    if progress_callback:
        progress_callback("Loading dataset...", 0.0)

    # Load dataset
    try:
        samples = _load_dataset(config.dataset_path, config.max_samples)
    except Exception as e:
        return DistillResult(success=False, error=str(e))

    if not samples:
        return DistillResult(success=False, error="Dataset is empty")

    if progress_callback:
        progress_callback(f"Loaded {len(samples)} samples", 0.05)

    # Load teacher & student models
    try:
        from mlx_lm import load as mlx_load

        if progress_callback:
            progress_callback("Loading teacher model...", 0.1)
        teacher_model, teacher_tokenizer = mlx_load(config.teacher_dir)

        if progress_callback:
            progress_callback("Loading student model...", 0.2)
        student_model, student_tokenizer = mlx_load(config.student_dir)
    except Exception as e:
        return DistillResult(success=False, error=f"Failed to load models: {e}")

    # Freeze teacher
    teacher_model.freeze()

    # Check vocab size compatibility
    teacher_vocab = teacher_model.model.embed_tokens.weight.shape[0] if hasattr(teacher_model, 'model') else None
    student_vocab = student_model.model.embed_tokens.weight.shape[0] if hasattr(student_model, 'model') else None
    vocab_mismatch = teacher_vocab and student_vocab and teacher_vocab != student_vocab

    # Cross-family: auto-switch to data distillation
    if vocab_mismatch:
        logger.info(
            f"Vocab mismatch (teacher={teacher_vocab}, student={student_vocab}). "
            f"Switching to data distillation mode."
        )
        return _run_data_distillation(
            config, teacher_model, teacher_tokenizer,
            student_model, student_tokenizer,
            samples, t0, progress_callback,
        )

    # Prepare output
    output_dir = _resolve_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)

    teacher_name = Path(config.teacher_dir).name
    student_name = Path(config.student_dir).name

    # Tokenize dataset
    if progress_callback:
        progress_callback("Tokenizing dataset...", 0.25)

    tokenized = []
    for sample in samples:
        text = _get_text_field(sample)
        if not text:
            continue
        tokens = student_tokenizer.encode(text)
        if len(tokens) > 2:  # Skip very short samples
            tokenized.append(mx.array(tokens[:512]))  # Cap at 512 tokens

    if not tokenized:
        return DistillResult(success=False, error="No valid text in dataset after tokenization")

    # Route to TAID mode if requested
    if config.mode == "taid":
        return _run_taid_distillation(
            config, teacher_model, teacher_tokenizer,
            student_model, student_tokenizer,
            tokenized, output_dir, teacher_name, student_name,
            t0, progress_callback,
        )

    # Training loop (standard offline KL+CE)
    T = config.temperature
    alpha = config.alpha
    total_steps = config.num_epochs * ((len(tokenized) + config.batch_size - 1) // config.batch_size)
    step = 0
    loss_history: list[dict] = []

    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Loss function combining KL divergence and cross-entropy
    def distill_loss(student_model, input_ids):
        # Student forward pass
        student_logits = student_model(input_ids[:-1][None])
        # Teacher forward pass (no grad)
        teacher_logits = mx.stop_gradient(teacher_model(input_ids[:-1][None]))

        # Handle vocab size mismatch: truncate to smaller vocab
        s_vocab = student_logits.shape[-1]
        t_vocab = teacher_logits.shape[-1]
        if s_vocab != t_vocab:
            min_vocab = min(s_vocab, t_vocab)
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]

        # Targets (next token) — clamp to valid range
        targets = input_ids[1:]
        if s_vocab != t_vocab:
            targets = mx.clip(targets, 0, min(s_vocab, t_vocab) - 1)

        # Temperature-scaled softmax
        student_soft = mx.softmax(student_logits[0] / T, axis=-1)
        teacher_soft = mx.softmax(teacher_logits[0] / T, axis=-1)

        # KL divergence: KL(teacher || student) = sum(teacher * log(teacher / student))
        # Clamp for numerical stability
        eps = 1e-8
        kl = mx.sum(teacher_soft * (mx.log(teacher_soft + eps) - mx.log(student_soft + eps)), axis=-1)
        kl_loss = mx.mean(kl) * (T * T)

        # Cross-entropy loss with hard targets
        ce_loss = mx.mean(nn.losses.cross_entropy(student_logits[0], targets))

        # Combined loss
        total = alpha * kl_loss + (1.0 - alpha) * ce_loss
        return total, (kl_loss, ce_loss)

    loss_and_grad = nn.value_and_grad(student_model, lambda m, x: distill_loss(m, x))

    final_loss = 0.0
    final_kl = 0.0
    final_ce = 0.0

    for epoch in range(config.num_epochs):
        epoch_losses = []

        for i in range(0, len(tokenized), config.batch_size):
            batch = tokenized[i : i + config.batch_size]

            batch_loss = 0.0
            batch_kl = 0.0
            batch_ce = 0.0

            for input_ids in batch:
                (loss, (kl_loss, ce_loss)), grads = loss_and_grad(student_model, input_ids)
                optimizer.update(student_model, grads)
                mx.eval(student_model.parameters(), optimizer.state)

                batch_loss += loss.item()
                batch_kl += kl_loss.item()
                batch_ce += ce_loss.item()

            n = len(batch)
            avg_loss = batch_loss / n
            avg_kl = batch_kl / n
            avg_ce = batch_ce / n
            epoch_losses.append(avg_loss)

            step += 1
            pct = 0.3 + 0.65 * (step / max(total_steps, 1))

            loss_record = {
                "epoch": epoch + 1,
                "step": step,
                "loss": round(avg_loss, 4),
                "kl_loss": round(avg_kl, 4),
                "ce_loss": round(avg_ce, 4),
            }
            loss_history.append(loss_record)

            if progress_callback:
                progress_callback(
                    f"Epoch {epoch+1}/{config.num_epochs}, "
                    f"Step {step}/{total_steps}, "
                    f"Loss: {avg_loss:.4f} (KL: {avg_kl:.4f}, CE: {avg_ce:.4f})",
                    pct,
                )

        final_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        final_kl = avg_kl
        final_ce = avg_ce

    # Save distilled student model
    if progress_callback:
        progress_callback("Saving distilled model...", 0.95)

    try:
        from mlx_lm import save as mlx_save

        # Save model weights
        student_model.save_weights(os.path.join(output_dir, "model.safetensors"))

        # Copy config and tokenizer files from student
        import shutil
        for fname in ("config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "tokenizer.model"):
            src = os.path.join(config.student_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

        # Write distillation metadata
        meta = {
            "distillation": {
                "teacher": teacher_name,
                "student": student_name,
                "mode": config.mode,
                "temperature": config.temperature,
                "alpha": config.alpha,
                "epochs": config.num_epochs,
                "dataset_samples": len(tokenized),
                "final_loss": round(final_loss, 4),
            }
        }
        with open(os.path.join(output_dir, "distillation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    except Exception as e:
        return DistillResult(
            success=False,
            error=f"Failed to save model: {e}",
            duration_seconds=time.time() - t0,
        )

    duration = time.time() - t0

    if progress_callback:
        progress_callback("Distillation complete!", 1.0)

    warning = ""
    if vocab_mismatch:
        warning = (
            f"Vocab size mismatch (teacher={teacher_vocab}, student={student_vocab}). "
            f"Cross-family distillation — KL divergence quality is degraded. "
            f"Recommend same-family models (e.g., Qwen3.5-4B → Qwen3.5-0.8B)."
        )

    return DistillResult(
        success=True,
        output_dir=output_dir,
        teacher_name=teacher_name,
        student_name=student_name,
        num_epochs=config.num_epochs,
        total_steps=step,
        final_loss=round(final_loss, 4),
        final_kl_loss=round(final_kl, 4),
        final_ce_loss=round(final_ce, 4),
        duration_seconds=round(duration, 2),
        dataset_samples=len(tokenized),
        warning=warning,
        loss_history=loss_history,
    )


def _run_taid_distillation(
    config: DistillConfig,
    teacher_model,
    teacher_tokenizer,
    student_model,
    student_tokenizer,
    tokenized: list,
    output_dir: str,
    teacher_name: str,
    student_name: str,
    t0: float,
    progress_callback=None,
) -> DistillResult:
    import math

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    T = config.temperature
    alpha_max = min(config.alpha + 0.3, 0.9)  # initial alpha (higher)
    alpha_min = max(config.alpha - 0.3, 0.1)  # final alpha (lower)

    total_steps = config.num_epochs * (
        (len(tokenized) + config.batch_size - 1) // config.batch_size
    )
    step = 0
    loss_history: list[dict] = []
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    def taid_alpha(current_step: int) -> float:
        """Cosine schedule from alpha_max to alpha_min."""
        t = current_step / max(total_steps, 1)
        return alpha_min + (alpha_max - alpha_min) * math.cos(math.pi * t / 2)

    def taid_loss(student_model, input_ids, alpha):
        """TAID loss with time-adaptive alpha."""
        student_logits = student_model(input_ids[:-1][None])
        teacher_logits = mx.stop_gradient(teacher_model(input_ids[:-1][None]))

        # Handle vocab mismatch
        s_vocab = student_logits.shape[-1]
        t_vocab = teacher_logits.shape[-1]
        if s_vocab != t_vocab:
            min_vocab = min(s_vocab, t_vocab)
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]

        targets = input_ids[1:]
        if s_vocab != t_vocab:
            targets = mx.clip(targets, 0, min(s_vocab, t_vocab) - 1)

        # Temperature-scaled softmax
        student_soft = mx.softmax(student_logits[0] / T, axis=-1)
        teacher_soft = mx.softmax(teacher_logits[0] / T, axis=-1)

        # TAID interpolated target: blend teacher and student predictions
        # This helps the student gradually develop independence
        interpolated = alpha * teacher_soft + (1.0 - alpha) * mx.stop_gradient(student_soft)

        # KL divergence against interpolated target
        eps = 1e-8
        kl = mx.sum(
            interpolated * (mx.log(interpolated + eps) - mx.log(student_soft + eps)),
            axis=-1,
        )
        kl_loss = mx.mean(kl) * (T * T)

        # Cross-entropy with hard targets
        ce_loss = mx.mean(nn.losses.cross_entropy(student_logits[0], targets))

        # Combined loss with current alpha
        total = alpha * kl_loss + (1.0 - alpha) * ce_loss
        return total, (kl_loss, ce_loss)

    final_loss = 0.0
    final_kl = 0.0
    final_ce = 0.0

    for epoch in range(config.num_epochs):
        epoch_losses = []

        for i in range(0, len(tokenized), config.batch_size):
            batch = tokenized[i : i + config.batch_size]
            current_alpha = taid_alpha(step)

            # Create loss function with current alpha
            loss_and_grad = nn.value_and_grad(
                student_model,
                lambda m, x: taid_loss(m, x, current_alpha),
            )

            batch_loss = 0.0
            batch_kl = 0.0
            batch_ce = 0.0

            for input_ids in batch:
                (loss, (kl_loss, ce_loss)), grads = loss_and_grad(
                    student_model, input_ids
                )
                optimizer.update(student_model, grads)
                mx.eval(student_model.parameters(), optimizer.state)

                batch_loss += loss.item()
                batch_kl += kl_loss.item()
                batch_ce += ce_loss.item()

            n = len(batch)
            avg_loss = batch_loss / n
            avg_kl = batch_kl / n
            avg_ce = batch_ce / n
            epoch_losses.append(avg_loss)

            step += 1
            pct = 0.3 + 0.65 * (step / max(total_steps, 1))

            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    "loss": round(avg_loss, 4),
                    "kl_loss": round(avg_kl, 4),
                    "ce_loss": round(avg_ce, 4),
                    "alpha": round(current_alpha, 4),
                }
            )

            if progress_callback:
                progress_callback(
                    f"TAID Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Step {step}/{total_steps}, "
                    f"Loss: {avg_loss:.4f} (α={current_alpha:.3f})",
                    pct,
                )

        final_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        final_kl = avg_kl
        final_ce = avg_ce

    # Save distilled model
    if progress_callback:
        progress_callback("Saving TAID distilled model...", 0.95)

    try:
        student_model.save_weights(os.path.join(output_dir, "model.safetensors"))

        import shutil

        for fname in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ):
            src = os.path.join(config.student_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))

        meta = {
            "distillation": {
                "teacher": teacher_name,
                "student": student_name,
                "mode": "taid",
                "temperature": config.temperature,
                "alpha_schedule": f"cos({alpha_max:.2f} → {alpha_min:.2f})",
                "epochs": config.num_epochs,
                "dataset_samples": len(tokenized),
                "final_loss": round(final_loss, 4),
            }
        }
        with open(os.path.join(output_dir, "distillation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    except Exception as e:
        return DistillResult(
            success=False,
            error=f"Failed to save TAID model: {e}",
            duration_seconds=time.time() - t0,
        )

    duration = time.time() - t0

    if progress_callback:
        progress_callback("TAID distillation complete!", 1.0)

    return DistillResult(
        success=True,
        output_dir=output_dir,
        teacher_name=teacher_name,
        student_name=student_name,
        num_epochs=config.num_epochs,
        total_steps=step,
        final_loss=round(final_loss, 4),
        final_kl_loss=round(final_kl, 4),
        final_ce_loss=round(final_ce, 4),
        duration_seconds=round(duration, 2),
        dataset_samples=len(tokenized),
        loss_history=loss_history,
    )
