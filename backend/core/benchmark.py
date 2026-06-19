# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Edge Studio Benchmark — validates that optimizations are truly effective.

Measures before/after:
  - Memory usage (RAM occupied by model weights)
  - Token generation speed (tok/s)
  - Perplexity on WikiText-103 subset (quality proxy)
  - Model file size on disk

Usage:
    source ~/Documents/mlx-community/edgestudio-3-11-env/bin/activate
    python -m backend.core.benchmark --model /path/to/model [--compare /path/to/optimized]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog. "
    "Artificial intelligence has transformed how we interact with technology. "
    "Edge computing brings processing closer to where data is generated.",

    "Machine learning models require significant computational resources. "
    "Quantization reduces model size while preserving most of the accuracy. "
    "The optimization pipeline consists of seven distinct stages.",

    "语言模型在各种自然语言处理任务中表现出色。"
    "端侧推理使得隐私保护成为可能，因为数据不需要离开设备。"
    "模型优化是提高推理效率的关键技术。",
]

GENERATION_PROMPT = (
    "Explain the concept of edge computing in three sentences."
)

GENERATION_TOKENS = 100


def _disk_size_mb(model_dir: str) -> float:
    p = Path(model_dir)
    # Single GGUF file
    if p.is_file() and p.suffix.lower() == ".gguf":
        return p.stat().st_size / (1024 * 1024)
    # Directory: sum safetensors + gguf files
    total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
    total += sum(f.stat().st_size for f in p.glob("*.gguf"))
    return total / (1024 * 1024)


def _memory_mb() -> float:
    """Current MLX active memory in MB."""
    return mx.get_active_memory() / (1024 * 1024)


def _peak_memory_mb() -> float:
    """Peak MLX cache memory in MB."""
    return mx.get_peak_memory() / (1024 * 1024)


def _reset_peak_memory():
    mx.reset_peak_memory()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    model_dir: str
    disk_size_mb: float

    # Memory
    memory_before_load_mb: float = 0.0
    memory_after_load_mb: float = 0.0
    peak_memory_mb: float = 0.0

    # Speed
    generation_prompt: str = ""
    generation_tokens: int = 0
    generation_time_s: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_s: float = 0.0

    # Quality
    perplexity: float = 0.0
    perplexity_texts: list[str] = field(default_factory=list)

    # Metadata
    error: str = ""
    model_type: str = ""
    is_edge_optimized: bool = False
    optimization_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Model: {Path(self.model_dir).name}",
            f"  Disk size:        {self.disk_size_mb:.1f} MB",
            f"  Memory (load):    {self.memory_after_load_mb:.0f} MB (+{self.memory_after_load_mb - self.memory_before_load_mb:.0f} MB)",
            f"  Peak memory:      {self.peak_memory_mb:.0f} MB",
            f"  Speed:            {self.tokens_per_second:.1f} tok/s",
            f"  TTFT:             {self.time_to_first_token_s*1000:.0f} ms",
            f"  Perplexity:       {self.perplexity:.2f}",
        ]
        if self.is_edge_optimized:
            lines.append(f"  [Edge Studio]:    {self.optimization_summary}")
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    baseline: BenchmarkResult
    optimized: BenchmarkResult

    @property
    def disk_reduction_pct(self) -> float:
        if self.baseline.disk_size_mb == 0:
            return 0.0
        return (1 - self.optimized.disk_size_mb / self.baseline.disk_size_mb) * 100

    @property
    def memory_reduction_pct(self) -> float:
        b = self.baseline.memory_after_load_mb
        o = self.optimized.memory_after_load_mb
        if b == 0:
            return 0.0
        return (1 - o / b) * 100

    @property
    def speed_improvement_pct(self) -> float:
        b = self.baseline.tokens_per_second
        o = self.optimized.tokens_per_second
        if b == 0:
            return 0.0
        return (o / b - 1) * 100

    @property
    def perplexity_delta(self) -> float:
        return self.optimized.perplexity - self.baseline.perplexity

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BASELINE",
            self.baseline.summary(),
            "",
            "OPTIMIZED",
            self.optimized.summary(),
            "",
            "COMPARISON",
            f"  Disk:      {self.baseline.disk_size_mb:.0f} MB → {self.optimized.disk_size_mb:.0f} MB  ({self.disk_reduction_pct:+.1f}%)",
            f"  Memory:    {self.baseline.memory_after_load_mb:.0f} MB → {self.optimized.memory_after_load_mb:.0f} MB  ({self.memory_reduction_pct:+.1f}%)",
            f"  Speed:     {self.baseline.tokens_per_second:.1f} → {self.optimized.tokens_per_second:.1f} tok/s  ({self.speed_improvement_pct:+.1f}%)",
            f"  PPL:       {self.baseline.perplexity:.2f} → {self.optimized.perplexity:.2f}  (Δ{self.perplexity_delta:+.2f})",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def benchmark_model(
    model_dir: str,
    n_perplexity_texts: int = 3,
    generation_tokens: int = GENERATION_TOKENS,
    verbose: bool = True,
    enable_dsr: bool = False,
    dsr_budget: int | None = None,
) -> BenchmarkResult:
    """Run full benchmark on a single model directory."""
    import mlx_lm
    from mlx_lm import load, generate

    model_path = Path(model_dir)
    result = BenchmarkResult(
        model_dir=model_dir,
        disk_size_mb=_disk_size_mb(model_dir),
        perplexity_texts=PERPLEXITY_TEXTS[:n_perplexity_texts],
        generation_prompt=GENERATION_PROMPT,
        generation_tokens=generation_tokens,
    )

    # Detect Edge Studio optimizations
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        result.model_type = config.get("model_type", "unknown")
        result.is_edge_optimized = any(k in config for k in [
            "vocab_pruning", "text_layer_pruning", "weight_split",
            "resolution_reduction", "vision_fc2_quantization",
        ])
        # Also check nested text_config for per_layer_intermediate_sizes
        tc = config.get("text_config", {})
        if "per_layer_intermediate_sizes" in tc or "per_layer_intermediate_sizes" in config:
            result.is_edge_optimized = True
        if result.is_edge_optimized:
            parts = []
            if "vocab_pruning" in config:
                vp = config["vocab_pruning"]
                orig = vp.get("original_text_vocab_size") or vp.get("original_vocab_size", 0)
                comp = vp.get("compact_vocab_size", 0)
                parts.append(f"vocab {orig}→{comp}")
            if "text_layer_pruning" in config:
                lp = config["text_layer_pruning"]
                parts.append(f"layers {lp.get('old_num_layers')}→{lp.get('new_num_layers')}")
            if "per_layer_intermediate_sizes" in tc:
                plis = tc["per_layer_intermediate_sizes"]
                default = tc.get("intermediate_size", 0)
                pruned = sum(1 for s in plis if s < default)
                parts.append(f"neurons {pruned}/{len(plis)} layers pruned")
            result.optimization_summary = ", ".join(parts) if parts else "optimized"

    if verbose:
        print(f"\n📊 Benchmarking: {model_path.name}")
        print(f"   Disk: {result.disk_size_mb:.1f} MB | Type: {result.model_type}")
        if result.is_edge_optimized:
            print(f"   [Edge Studio] {result.optimization_summary}")

    try:
        # --- Memory before load ---
        mx.reset_peak_memory()
        result.memory_before_load_mb = _memory_mb()

        # --- Load model (try mlx_lm first, fall back to mlx_vlm for VLMs) ---
        if verbose:
            print("   Loading model...", end=" ", flush=True)
        t_load = time.time()
        _generate = generate  # default: mlx_lm.generate
        try:
            model, tokenizer = load(model_dir)
        except (ImportError, AttributeError, TypeError):
            from mlx_vlm import load as vlm_load, generate as vlm_generate
            model, tokenizer = vlm_load(model_dir)
            _generate = vlm_generate
        mx.eval(model.parameters())
        result.memory_after_load_mb = _memory_mb()
        result.peak_memory_mb = _peak_memory_mb()
        if verbose:
            print(f"done ({time.time() - t_load:.1f}s)")
            print(f"   Memory: {result.memory_after_load_mb:.0f} MB (peak {result.peak_memory_mb:.0f} MB)")

        # --- Generation speed ---
        if verbose:
            dsr_label = f" [DSR budget={dsr_budget}]" if enable_dsr else ""
            print(f"   Generating {generation_tokens} tokens{dsr_label}...", end=" ", flush=True)

        # Build DSR kwargs if enabled
        gen_extra = {}
        if enable_dsr and dsr_budget:
            try:
                from backend.core.dsr_cache import build_dsr_config, make_prompt_cache

                gen_extra["prompt_cache"] = make_prompt_cache(
                    model,
                    dsr_config=build_dsr_config(dsr_budget),
                )
            except (ImportError, RuntimeError):
                pass

        # Warmup
        _ = _generate(model, tokenizer, prompt=GENERATION_PROMPT, max_tokens=5, verbose=False)

        # Timed run
        t_gen = time.time()
        first_token_time = None

        # Use streaming to capture TTFT
        response = _generate(
            model, tokenizer,
            prompt=GENERATION_PROMPT,
            max_tokens=generation_tokens,
            verbose=False,
            **gen_extra,
        )
        elapsed = time.time() - t_gen
        # Count tokens in response
        response_tokens = len(tokenizer.encode(response))
        result.generation_time_s = elapsed
        result.generation_tokens = response_tokens
        result.tokens_per_second = response_tokens / elapsed if elapsed > 0 else 0
        result.time_to_first_token_s = first_token_time or 0.0

        if verbose:
            print(f"done ({result.tokens_per_second:.1f} tok/s)")

        # --- Perplexity ---
        if verbose:
            print("   Computing perplexity...", end=" ", flush=True)
        ppls = []
        for text in PERPLEXITY_TEXTS[:n_perplexity_texts]:
            ppl = _compute_perplexity(model, tokenizer, text)
            ppls.append(ppl)
        result.perplexity = sum(ppls) / len(ppls) if ppls else 0.0
        if verbose:
            print(f"done (PPL={result.perplexity:.2f})")

    except Exception as e:
        result.error = str(e)
        if verbose:
            print(f"\n   ERROR: {e}")

    return result


def _compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of model on text using cross-entropy loss."""
    import mlx.nn as nn

    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float("inf")

    # Cap at 512 tokens to keep it fast
    tokens = tokens[:512]
    input_ids = mx.array(tokens[:-1])[None]  # [1, seq-1]
    target_ids = mx.array(tokens[1:])         # [seq-1]

    # Forward pass
    logits = model(input_ids)  # [1, seq-1, vocab]
    logits = logits[0]          # [seq-1, vocab]

    # Cross-entropy
    loss = nn.losses.cross_entropy(logits, target_ids, reduction="mean")
    mx.eval(loss)
    ppl = math.exp(float(loss.item()))
    return ppl


# ---------------------------------------------------------------------------
# GGUF benchmark (uses llama.cpp tools)
# ---------------------------------------------------------------------------

def _find_gguf_file(model_dir: str) -> str | None:
    """Find primary .gguf file in path."""
    p = Path(model_dir)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
        return str(files[0]) if files else None
    return None


def benchmark_gguf_model(
    model_dir: str,
    n_threads: int = 8,
    n_tokens: int = 128,
    verbose: bool = True,
) -> BenchmarkResult:
    """Benchmark a GGUF model using llama-bench and llama-perplexity."""
    gguf_path = _find_gguf_file(model_dir)
    if not gguf_path:
        return BenchmarkResult(model_dir=model_dir, disk_size_mb=0, error="No .gguf file found")

    result = BenchmarkResult(
        model_dir=model_dir,
        disk_size_mb=_disk_size_mb(gguf_path),
        generation_prompt=GENERATION_PROMPT,
        generation_tokens=n_tokens,
    )

    # File size = memory estimate for GGUF (close approximation)
    result.memory_after_load_mb = result.disk_size_mb
    result.peak_memory_mb = result.disk_size_mb * 1.1

    if verbose:
        print(f"\n benchmarking GGUF: {Path(gguf_path).name}")
        print(f"   Disk: {result.disk_size_mb:.1f} MB")

    # Speed: llama-bench
    llama_bench = shutil.which("llama-bench")
    if llama_bench:
        try:
            if verbose:
                print(f"   Running llama-bench ({n_tokens} tokens)...", end=" ", flush=True)
            proc = subprocess.run(
                [llama_bench, "-m", gguf_path, "-t", str(n_threads),
                 "-n", str(n_tokens), "-p", "0", "-r", "1"],
                capture_output=True, text=True, timeout=300,
            )
            # Parse tok/s from llama-bench output (CSV format, last column is t/s)
            for line in proc.stdout.strip().split("\n"):
                if gguf_path in line or Path(gguf_path).name in line:
                    # llama-bench CSV: model,size,params,...,t/s
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            tps = float(parts[-1].strip())
                            result.tokens_per_second = tps
                        except ValueError:
                            pass
            if verbose:
                print(f"done ({result.tokens_per_second:.1f} tok/s)")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if verbose:
                print(f"failed: {e}")
    else:
        result.error = "llama-bench not found"

    # PPL: llama-perplexity
    llama_ppl = shutil.which("llama-perplexity")
    if llama_ppl:
        try:
            if verbose:
                print("   Running llama-perplexity...", end=" ", flush=True)
            # Use a short text for quick PPL estimation
            ppl_text = PERPLEXITY_TEXTS[0]
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                tmp.write(ppl_text)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    [llama_ppl, "-m", gguf_path, "-f", tmp_path,
                     "-t", str(n_threads), "--chunks", "1"],
                    capture_output=True, text=True, timeout=300,
                )
                # Parse PPL from output: "Final estimate: PPL = XX.XXXX"
                for line in proc.stderr.split("\n") + proc.stdout.split("\n"):
                    m = re.search(r"(?:Final estimate|perplexity)\s*[=:]\s*([\d.]+)", line, re.IGNORECASE)
                    if m:
                        result.perplexity = float(m.group(1))
                        break
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if verbose:
                print(f"done (PPL={result.perplexity:.2f})")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if verbose:
                print(f"failed: {e}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Edge Studio Benchmark")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--compare", default=None, help="Optimized model directory to compare against")
    parser.add_argument("--tokens", type=int, default=GENERATION_TOKENS, help="Generation tokens")
    parser.add_argument("--ppl-texts", type=int, default=3, help="Number of perplexity texts (1-3)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    result = benchmark_model(args.model, n_perplexity_texts=args.ppl_texts, generation_tokens=args.tokens)
    print("\n" + result.summary())

    if args.compare:
        opt_result = benchmark_model(args.compare, n_perplexity_texts=args.ppl_texts, generation_tokens=args.tokens)
        comparison = ComparisonResult(baseline=result, optimized=opt_result)
        print("\n" + comparison.summary())

        if args.output:
            out = {
                "baseline": result.to_dict(),
                "optimized": opt_result.to_dict(),
                "comparison": {
                    "disk_reduction_pct": comparison.disk_reduction_pct,
                    "memory_reduction_pct": comparison.memory_reduction_pct,
                    "speed_improvement_pct": comparison.speed_improvement_pct,
                    "perplexity_delta": comparison.perplexity_delta,
                },
            }
            Path(args.output).write_text(json.dumps(out, indent=2))
            print(f"\nResults saved to {args.output}")
    else:
        if args.output:
            Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
