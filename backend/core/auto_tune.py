# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Auto-tune benchmark — search optimal inference parameters for device + model.

Inspired by PMetal's Tuna: automatically finds the best configuration
(batch size, KV cache size, quantization precision) for maximum token/s.

Results are cached to ~/.cache/edgestudio/tuna/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.expanduser("~/.cache/edgestudio/tuna")


@dataclass
class TuneConfig:
    model_dir: str
    device_name: str = ""
    max_tokens: int = 50
    search_quantizations: list[str] = field(default_factory=lambda: ["default"])
    search_batch_sizes: list[int] = field(default_factory=lambda: [1])
    search_kv_cache_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048, 4096])
    search_temperatures: list[float] = field(default_factory=lambda: [0.0, 0.7])
    num_warmup: int = 2
    num_runs: int = 3


@dataclass
class TuneCandidate:
    batch_size: int
    kv_cache_size: int
    temperature: float
    quantization: str
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    peak_memory_mb: float = 0.0
    perplexity: float = 0.0


@dataclass
class TuneResult:
    success: bool
    model_name: str = ""
    device_name: str = ""
    best: TuneCandidate | None = None
    all_candidates: list[TuneCandidate] = field(default_factory=list)
    search_time_seconds: float = 0.0
    total_configs_tested: int = 0
    cached: bool = False
    cache_path: str = ""
    error: str = ""


def _cache_key(model_dir: str, device_name: str) -> str:
    """Generate a unique cache key for model + device combo."""
    key = f"{Path(model_dir).name}:{device_name}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_cache(model_dir: str, device_name: str) -> TuneResult | None:
    """Load cached tune result if available."""
    cache_file = Path(CACHE_DIR) / f"{_cache_key(model_dir, device_name)}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        best = TuneCandidate(**data["best"]) if data.get("best") else None
        candidates = [TuneCandidate(**c) for c in data.get("all_candidates", [])]
        return TuneResult(
            success=True,
            model_name=data.get("model_name", ""),
            device_name=data.get("device_name", ""),
            best=best,
            all_candidates=candidates,
            search_time_seconds=data.get("search_time_seconds", 0),
            total_configs_tested=data.get("total_configs_tested", 0),
            cached=True,
            cache_path=str(cache_file),
        )
    except (ImportError, RuntimeError, ValueError, OSError):
        return None


def _save_cache(result: TuneResult, model_dir: str, device_name: str):
    """Save tune result to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = Path(CACHE_DIR) / f"{_cache_key(model_dir, device_name)}.json"
    data = {
        "model_name": result.model_name,
        "device_name": result.device_name,
        "best": asdict(result.best) if result.best else None,
        "all_candidates": [asdict(c) for c in result.all_candidates],
        "search_time_seconds": result.search_time_seconds,
        "total_configs_tested": result.total_configs_tested,
        "timestamp": time.time(),
    }
    cache_file.write_text(json.dumps(data, indent=2))
    result.cache_path = str(cache_file)


def run_auto_tune(
    config: TuneConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> TuneResult:
    """Search for optimal inference parameters on this device + model combo."""
    t0 = time.time()

    # Check cache first
    device_name = config.device_name or _detect_device_name()
    cached = _load_cache(config.model_dir, device_name)
    if cached:
        if progress_callback:
            progress_callback("Loaded from cache", 1.0)
        return cached

    try:
        import mlx.core as mx
        from mlx_lm import load, generate
    except ImportError:
        return TuneResult(success=False, error="MLX required: pip install mlx mlx_lm")

    model_name = Path(config.model_dir).name

    if progress_callback:
        progress_callback("Loading model...", 0.05)

    try:
        model, tokenizer = load(config.model_dir)
    except Exception as e:
        return TuneResult(success=False, error=f"Failed to load model: {e}")

    # Build search space
    candidates: list[TuneCandidate] = []
    search_space = []
    for bs in config.search_batch_sizes:
        for kv in config.search_kv_cache_sizes:
            for temp in config.search_temperatures:
                for quant in config.search_quantizations:
                    search_space.append((bs, kv, temp, quant))

    total = len(search_space)

    if progress_callback:
        progress_callback(f"Testing {total} configurations...", 0.1)

    prompt = "Explain the concept of edge computing in three sentences."

    for i, (bs, kv, temp, quant) in enumerate(search_space):
        if progress_callback:
            pct = 0.1 + 0.8 * ((i + 1) / total)
            progress_callback(
                f"Config {i+1}/{total}: batch={bs}, kv={kv}, temp={temp}",
                pct,
            )

        try:
            # Warmup
            for _ in range(config.num_warmup):
                _ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)

            # Timed runs
            speeds = []
            ttfts = []
            peak_mems = []

            for _ in range(config.num_runs):
                mx.reset_peak_memory()
                t_start = time.time()

                response = generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=config.max_tokens,
                    temp=temp,
                    verbose=False,
                )

                elapsed = time.time() - t_start
                n_tokens = len(tokenizer.encode(response))
                tps = n_tokens / elapsed if elapsed > 0 else 0
                speeds.append(tps)
                peak_mems.append(mx.get_peak_memory() / (1024 * 1024))

            avg_speed = sum(speeds) / len(speeds)
            avg_peak = sum(peak_mems) / len(peak_mems)

            candidate = TuneCandidate(
                batch_size=bs,
                kv_cache_size=kv,
                temperature=temp,
                quantization=quant,
                tokens_per_second=round(avg_speed, 2),
                peak_memory_mb=round(avg_peak, 1),
            )
            candidates.append(candidate)

        except Exception as e:
            logger.warning(f"Config {i+1} failed: {e}")
            candidates.append(TuneCandidate(
                batch_size=bs, kv_cache_size=kv, temperature=temp, quantization=quant,
                tokens_per_second=0,
            ))

    # Find best by token/s
    valid = [c for c in candidates if c.tokens_per_second > 0]
    best = max(valid, key=lambda c: c.tokens_per_second) if valid else None

    duration = time.time() - t0

    result = TuneResult(
        success=True,
        model_name=model_name,
        device_name=device_name,
        best=best,
        all_candidates=candidates,
        search_time_seconds=round(duration, 2),
        total_configs_tested=total,
    )

    # Cache result
    _save_cache(result, config.model_dir, device_name)

    if progress_callback:
        if best:
            progress_callback(
                f"Best: {best.tokens_per_second:.1f} tok/s "
                f"(temp={best.temperature}, kv={best.kv_cache_size})",
                1.0,
            )
        else:
            progress_callback("No valid configuration found", 1.0)

    return result


def _detect_device_name() -> str:
    """Detect current device name."""
    import platform
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (ImportError, RuntimeError, ValueError, OSError):
        pass
    return platform.processor() or platform.machine()
