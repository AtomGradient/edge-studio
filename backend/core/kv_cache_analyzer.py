# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""KV Cache analyzer — memory estimation and device capacity planning.

Computes KV cache memory usage per token, peak memory projections,
and maximum conversation length for target devices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .architecture import ModelArchitecture, format_size
from .device_profiles import DeviceProfile, DEVICE_PROFILES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KVCacheConfig:
    """KV cache configuration derived from model config."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # bf16 = 2 bytes per element

    @property
    def bytes_per_token(self) -> int:
        """Memory for one token in KV cache (both K and V, all layers).

        Per layer: 2 (K+V) * num_kv_heads * head_dim * dtype_bytes
        Total: num_layers * per_layer
        """
        per_layer = 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes
        return self.num_layers * per_layer

    @property
    def bytes_per_token_per_layer(self) -> int:
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def cache_size(self, seq_len: int) -> int:
        """Total KV cache size in bytes for a given sequence length."""
        return self.bytes_per_token * seq_len


@dataclass
class MemoryBreakdown:
    """Memory usage breakdown for a given configuration."""
    model_weights_bytes: int
    kv_cache_bytes: int
    activation_estimate_bytes: int
    system_overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return (self.model_weights_bytes + self.kv_cache_bytes
                + self.activation_estimate_bytes + self.system_overhead_bytes)

    def as_dict(self) -> dict[str, int]:
        return {
            "Model Weights": self.model_weights_bytes,
            "KV Cache": self.kv_cache_bytes,
            "Activations": self.activation_estimate_bytes,
            "System Overhead": self.system_overhead_bytes,
        }


@dataclass
class DeviceCapacity:
    """What a device can handle with this model."""
    device_name: str
    device_ram_gb: float
    available_ram_bytes: int
    model_weights_bytes: int
    max_seq_len: int
    kv_cache_at_max_bytes: int
    fits: bool
    headroom_bytes: int


@dataclass
class KVCacheReport:
    """Complete KV cache analysis report."""
    model_name: str
    model_dir: str
    kv_config: KVCacheConfig
    model_weights_bytes: int
    # Precomputed data points for plotting
    seq_lengths: list[int] = field(default_factory=list)
    memory_breakdowns: list[MemoryBreakdown] = field(default_factory=list)
    device_capacities: list[DeviceCapacity] = field(default_factory=list)
    # From inference trace if available
    trace_steps: list[dict] = field(default_factory=list)
    # DSR retention curves (optional)
    dsr_curves: dict[str, list[MemoryBreakdown]] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def extract_kv_config(model_dir: str) -> KVCacheConfig:
    """Extract KV cache configuration from model config.json.

    Handles standard MHA/GQA/MQA as well as MLA (Multi-head Latent Attention)
    used by DeepSeek V3 where the KV cache stores compressed latent vectors.
    """
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_layers = config.get("num_hidden_layers", 0)
    num_heads = config.get("num_attention_heads", 0)
    num_kv_heads = config.get("num_key_value_heads", num_heads)

    # MLA detection (DeepSeek V3): uses compressed KV
    kv_lora_rank = config.get("kv_lora_rank", 0)
    if kv_lora_rank > 0:
        # MLA: KV cache stores compressed latent of size kv_lora_rank + qk_rope_head_dim
        qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        head_dim = kv_lora_rank + qk_rope_head_dim
        num_kv_heads = 1  # single compressed KV per layer
    else:
        head_dim = config.get("head_dim", config.get("hidden_size", 0) // num_heads if num_heads else 128)

    # For quantized models, KV cache is still in bf16 during inference
    dtype_bytes = 2

    return KVCacheConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )


def estimate_activation_memory(config: KVCacheConfig, batch_size: int = 1) -> int:
    """Estimate peak activation memory during a single forward pass.

    This is a rough estimate: activations for one layer at a time
    (attention scores, intermediate MLP activations, etc.)
    """
    # Attention scores: batch * num_heads * seq_len * seq_len (peak, not cached)
    # But for single-token decode, this is small
    # MLP intermediate: batch * intermediate_size * dtype
    # Rough heuristic: ~2x one layer's KV contribution
    return config.bytes_per_token_per_layer * batch_size * 4


def compute_memory_breakdown(
    kv_config: KVCacheConfig,
    model_weights_bytes: int,
    seq_len: int,
) -> MemoryBreakdown:
    """Compute memory breakdown for a given sequence length."""
    kv_bytes = kv_config.cache_size(seq_len)
    activation_bytes = estimate_activation_memory(kv_config)
    # System overhead: ~200MB for MLX runtime, tokenizer, etc.
    overhead = 200 * 1024 * 1024

    return MemoryBreakdown(
        model_weights_bytes=model_weights_bytes,
        kv_cache_bytes=kv_bytes,
        activation_estimate_bytes=activation_bytes,
        system_overhead_bytes=overhead,
    )


def compute_dsr_kv_bytes(
    kv_config: KVCacheConfig,
    seq_len: int,
    dsr_budget: int,
    kv_quant_bits: int | None = None,
) -> int:
    """Compute KV cache bytes with DSR retention (and optional quantization).

    DSR caps the cache at dsr_budget tokens — once seq_len exceeds the budget,
    cache size stays flat instead of growing linearly.

    With kv_quant_bits, stored tokens use ~(bits/16) of the FP16 size.
    """
    effective_tokens = min(seq_len, dsr_budget)
    base_bytes = kv_config.bytes_per_token * effective_tokens
    if kv_quant_bits and kv_quant_bits < 16:
        # Quantized KV: roughly bits/16 of FP16 size
        return int(base_bytes * kv_quant_bits / 16)
    return base_bytes


def compute_max_seq_len(
    kv_config: KVCacheConfig,
    model_weights_bytes: int,
    available_ram_bytes: int,
) -> int:
    """Compute maximum sequence length that fits in available RAM."""
    activation_bytes = estimate_activation_memory(kv_config)
    overhead = 200 * 1024 * 1024

    remaining = available_ram_bytes - model_weights_bytes - activation_bytes - overhead
    if remaining <= 0:
        return 0

    bytes_per_token = kv_config.bytes_per_token
    if bytes_per_token <= 0:
        return 0

    return remaining // bytes_per_token


def analyze_device_capacity(
    kv_config: KVCacheConfig,
    model_weights_bytes: int,
    device: DeviceProfile,
) -> DeviceCapacity:
    """Analyze how much conversation a device can handle."""
    available_bytes = int(device.available_ram_gb * 1024 ** 3)
    max_seq = compute_max_seq_len(kv_config, model_weights_bytes, available_bytes)
    kv_at_max = kv_config.cache_size(max_seq) if max_seq > 0 else 0
    fits = model_weights_bytes < available_bytes

    breakdown = compute_memory_breakdown(kv_config, model_weights_bytes, max_seq)
    headroom = available_bytes - breakdown.total_bytes

    return DeviceCapacity(
        device_name=device.name,
        device_ram_gb=device.ram_gb,
        available_ram_bytes=available_bytes,
        model_weights_bytes=model_weights_bytes,
        max_seq_len=max_seq,
        kv_cache_at_max_bytes=kv_at_max,
        fits=fits,
        headroom_bytes=max(0, headroom),
    )


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def generate_kv_report(
    arch: ModelArchitecture,
    inference_trace=None,
    target_devices: Optional[list[str]] = None,
) -> KVCacheReport:
    """Generate complete KV cache analysis report.

    Args:
        arch: Model architecture
        inference_trace: Optional InferenceTrace with actual KV cache data
        target_devices: Device names to analyze (defaults to all)
    """
    kv_config = extract_kv_config(arch.model_dir)
    model_weights = arch.root.total_size_bytes

    report = KVCacheReport(
        model_name=arch.model_name,
        model_dir=arch.model_dir,
        kv_config=kv_config,
        model_weights_bytes=model_weights,
    )

    # Precompute memory curves for various sequence lengths
    seq_points = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    for sl in seq_points:
        breakdown = compute_memory_breakdown(kv_config, model_weights, sl)
        report.seq_lengths.append(sl)
        report.memory_breakdowns.append(breakdown)

    # DSR retention curves: show memory savings at various budgets
    dsr_budgets = {"DSR 2K": 2048, "DSR 4K": 4096, "DSR 4K+INT4": 4096}
    for label, budget in dsr_budgets.items():
        quant_bits = 4 if "INT4" in label else None
        curve = []
        for sl in seq_points:
            dsr_kv = compute_dsr_kv_bytes(kv_config, sl, budget, quant_bits)
            activation_bytes = estimate_activation_memory(kv_config)
            overhead = 200 * 1024 * 1024
            curve.append(MemoryBreakdown(
                model_weights_bytes=model_weights,
                kv_cache_bytes=dsr_kv,
                activation_estimate_bytes=activation_bytes,
                system_overhead_bytes=overhead,
            ))
        report.dsr_curves[label] = curve

    # Device capacities
    if target_devices is None:
        target_devices = list(DEVICE_PROFILES.keys())

    for device_name in target_devices:
        device = DEVICE_PROFILES.get(device_name)
        if device:
            cap = analyze_device_capacity(kv_config, model_weights, device)
            report.device_capacities.append(cap)

    # Extract actual KV cache growth from inference trace
    if inference_trace and hasattr(inference_trace, "steps"):
        prompt_len = len(inference_trace.prompt_token_ids)
        for i, step in enumerate(inference_trace.steps):
            current_seq = prompt_len + i + 1
            kv_bytes = kv_config.cache_size(current_seq)
            report.trace_steps.append({
                "step": i,
                "seq_len": current_seq,
                "kv_cache_bytes": kv_bytes,
                "token": step.token_str,
            })

    return report
