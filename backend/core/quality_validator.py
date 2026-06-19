# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Quality validator — perplexity computation and multi-prompt benchmarking.

Uses mlx-lm's model loading API for universal model support (117+ architectures).
Computes perplexity via direct model forward pass and compares generation quality
across models using the universal tracer.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PerplexityResult:
    """Result of perplexity computation on a text."""
    text: str
    num_tokens: int
    total_log_prob: float
    perplexity: float
    per_token_log_probs: list[float] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class GenerationSample:
    """A single prompt→generation result for quality comparison."""
    prompt: str
    generated_text: str
    num_tokens: int
    avg_prob: float
    tokens_per_second: float
    duration_seconds: float


@dataclass
class QualityReport:
    """Complete quality validation report."""
    model_dir: str
    model_name: str
    perplexity_results: list[PerplexityResult] = field(default_factory=list)
    generation_samples: list[GenerationSample] = field(default_factory=list)
    avg_perplexity: float = 0.0
    total_duration_seconds: float = 0.0

    def compute_averages(self):
        if self.perplexity_results:
            self.avg_perplexity = math.exp(
                sum(r.total_log_prob for r in self.perplexity_results)
                / max(sum(r.num_tokens for r in self.perplexity_results), 1)
            )


@dataclass
class ComparisonResult:
    """Side-by-side comparison between two models."""
    prompt: str
    model_a_text: str
    model_b_text: str
    model_a_avg_prob: float
    model_b_avg_prob: float
    model_a_tps: float
    model_b_tps: float


# ---------------------------------------------------------------------------
# Default benchmark prompts
# ---------------------------------------------------------------------------

DEFAULT_PPL_TEXTS = [
    "The transformer architecture was introduced in the paper 'Attention Is All You Need' "
    "by Vaswani et al. in 2017. It replaced recurrent neural networks with self-attention "
    "mechanisms, enabling much more efficient parallel training on modern hardware.",

    "Large language models are trained on massive datasets of text from the internet. "
    "During training, the model learns to predict the next token in a sequence, which "
    "gives it broad knowledge of language, facts, and reasoning patterns.",

    "Quantization reduces the precision of model weights from floating point to lower "
    "bit representations. This significantly reduces memory usage and can improve "
    "inference speed, with minimal impact on model quality when done carefully.",
]

DEFAULT_GENERATION_PROMPTS = [
    "Hi How are you?",
    "Explain what a neural network is in one sentence.",
    "What is the capital of Japan?",
    "Write a short poem about coding.",
    "Translate 'hello world' to French, German, and Spanish.",
]


# ---------------------------------------------------------------------------
# Perplexity computation (universal — uses mlx-lm model loading)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_dir: str):
    """Load model and tokenizer via mlx-lm. Cached internally."""
    from mlx_lm.utils import load as lm_load
    return lm_load(model_dir)


def compute_perplexity(
    model_dir: str,
    text: str,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PerplexityResult:
    """Compute perplexity of a text under the model.

    PPL = exp(-1/N * sum(log P(token_i | context)))
    Works with any mlx-lm compatible model.
    """
    t0 = time.time()

    if progress_callback:
        progress_callback("Loading model...", 0.1)

    model, tokenizer = _load_model_and_tokenizer(model_dir)

    # Tokenize
    tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
    if hasattr(tok, 'encode'):
        encoding = tok.encode(text)
        token_ids = encoding.ids if hasattr(encoding, 'ids') else list(encoding)
    else:
        token_ids = list(tokenizer.encode(text))

    if len(token_ids) < 2:
        return PerplexityResult(
            text=text, num_tokens=0, total_log_prob=0.0,
            perplexity=float("inf"), duration_seconds=0.0,
        )

    if progress_callback:
        progress_callback("Computing logits...", 0.3)

    # Process in chunks to avoid OOM on long texts
    max_chunk = 512
    all_log_probs: list[float] = []

    for start in range(0, len(token_ids) - 1, max_chunk):
        end = min(start + max_chunk + 1, len(token_ids))
        chunk_ids = token_ids[start:end]

        # Forward pass using mlx-lm model
        input_ids = mx.array([chunk_ids])  # [1, L]
        logits = model(input_ids)           # [1, L, vocab_size]
        mx.eval(logits)

        # logits[0, i] predicts token[i+1]
        logits_2d = logits[0]  # [L, vocab_size]
        log_probs = logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)
        log_probs_np = np.array(log_probs.astype(mx.float32))

        for i in range(len(chunk_ids) - 1):
            target_id = chunk_ids[i + 1]
            lp = float(log_probs_np[i, target_id])
            all_log_probs.append(lp)

        if progress_callback:
            pct = 0.3 + 0.6 * (start + len(chunk_ids)) / len(token_ids)
            progress_callback(f"Processing tokens {start}-{end}...", pct)

    total_log_prob = sum(all_log_probs)
    n = len(all_log_probs)
    ppl = math.exp(-total_log_prob / n) if n > 0 else float("inf")

    if progress_callback:
        progress_callback("Done", 1.0)

    return PerplexityResult(
        text=text,
        num_tokens=n,
        total_log_prob=total_log_prob,
        perplexity=ppl,
        per_token_log_probs=all_log_probs,
        duration_seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Generation quality benchmark
# ---------------------------------------------------------------------------

def benchmark_generation(
    model_dir: str,
    prompts: list[str] | None = None,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    enable_thinking: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[GenerationSample]:
    """Run generation on multiple prompts and collect quality metrics.

    Uses universal tracer for any mlx-lm model, falls back to Qwen3 legacy
    tracer if the model is Qwen3.
    """
    if prompts is None:
        prompts = DEFAULT_GENERATION_PROMPTS

    samples = []
    for i, prompt in enumerate(prompts):
        if progress_callback:
            progress_callback(f"Prompt {i+1}/{len(prompts)}: {prompt[:30]}...", i / len(prompts))

        t0 = time.time()
        try:
            trace = _run_trace(
                model_dir=model_dir,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )

            num_gen = len(trace.steps)
            decode_time = trace.total_time_seconds - trace.prefill_time_seconds
            tps = num_gen / decode_time if decode_time > 0 else 0
            avg_prob = (
                sum(s.chosen_prob for s in trace.steps) / num_gen
                if num_gen > 0 else 0
            )

            samples.append(GenerationSample(
                prompt=prompt,
                generated_text=trace.generated_text,
                num_tokens=num_gen,
                avg_prob=avg_prob,
                tokens_per_second=tps,
                duration_seconds=time.time() - t0,
            ))
        except Exception as e:
            samples.append(GenerationSample(
                prompt=prompt,
                generated_text=f"[ERROR: {e}]",
                num_tokens=0,
                avg_prob=0,
                tokens_per_second=0,
                duration_seconds=time.time() - t0,
            ))

    if progress_callback:
        progress_callback("Done", 1.0)

    return samples


def _run_trace(model_dir: str, prompt: str, max_tokens: int = 50,
               temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
               enable_thinking: bool = False, enable_timing: bool = False):
    """Run inference trace using universal tracer (supports all models)."""
    from .universal_tracer import run_universal_trace
    return run_universal_trace(
        model_path=model_dir,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        enable_thinking=enable_thinking,
        enable_timing=enable_timing,
    )


# ---------------------------------------------------------------------------
# Full quality report
# ---------------------------------------------------------------------------

def run_quality_report(
    model_dir: str,
    ppl_texts: list[str] | None = None,
    generation_prompts: list[str] | None = None,
    max_tokens: int = 50,
    enable_thinking: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> QualityReport:
    """Run complete quality validation: perplexity + generation benchmark."""
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_name = config.get("architectures", ["Unknown"])[0]
    report = QualityReport(model_dir=model_dir, model_name=model_name)

    t0 = time.time()

    # Perplexity
    if ppl_texts is None:
        ppl_texts = DEFAULT_PPL_TEXTS

    for i, text in enumerate(ppl_texts):
        if progress_callback:
            progress_callback(f"PPL text {i+1}/{len(ppl_texts)}", i / (len(ppl_texts) + 1))
        result = compute_perplexity(model_dir, text)
        report.perplexity_results.append(result)

    # Generation benchmark
    if progress_callback:
        progress_callback("Running generation benchmark...", 0.6)

    report.generation_samples = benchmark_generation(
        model_dir=model_dir,
        prompts=generation_prompts,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )

    report.total_duration_seconds = time.time() - t0
    report.compute_averages()

    if progress_callback:
        progress_callback("Report complete", 1.0)

    return report


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    model_a_dir: str,
    model_b_dir: str,
    prompts: list[str] | None = None,
    max_tokens: int = 50,
    enable_thinking: bool = False,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[ComparisonResult]:
    """Compare generation quality between two models on the same prompts.

    Works with any mlx-lm compatible model (not limited to Qwen3).
    """
    if prompts is None:
        prompts = DEFAULT_GENERATION_PROMPTS

    results = []
    total = len(prompts) * 2

    for i, prompt in enumerate(prompts):
        if progress_callback:
            progress_callback(f"Model A — prompt {i+1}/{len(prompts)}", (i * 2) / total)

        try:
            trace_a = _run_trace(
                model_dir=model_a_dir, prompt=prompt, max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            text_a = trace_a.generated_text
            prob_a = sum(s.chosen_prob for s in trace_a.steps) / max(len(trace_a.steps), 1)
            dt_a = trace_a.total_time_seconds - trace_a.prefill_time_seconds
            tps_a = len(trace_a.steps) / dt_a if dt_a > 0 else 0
        except Exception as e:
            text_a, prob_a, tps_a = f"[ERROR: {e}]", 0, 0

        if progress_callback:
            progress_callback(f"Model B — prompt {i+1}/{len(prompts)}", (i * 2 + 1) / total)

        try:
            trace_b = _run_trace(
                model_dir=model_b_dir, prompt=prompt, max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )
            text_b = trace_b.generated_text
            prob_b = sum(s.chosen_prob for s in trace_b.steps) / max(len(trace_b.steps), 1)
            dt_b = trace_b.total_time_seconds - trace_b.prefill_time_seconds
            tps_b = len(trace_b.steps) / dt_b if dt_b > 0 else 0
        except Exception as e:
            text_b, prob_b, tps_b = f"[ERROR: {e}]", 0, 0

        results.append(ComparisonResult(
            prompt=prompt,
            model_a_text=text_a,
            model_b_text=text_b,
            model_a_avg_prob=prob_a,
            model_b_avg_prob=prob_b,
            model_a_tps=tps_a,
            model_b_tps=tps_b,
        ))

    if progress_callback:
        progress_callback("Comparison complete", 1.0)

    return results
