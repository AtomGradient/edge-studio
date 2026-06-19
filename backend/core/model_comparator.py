# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model comparison logic — arch diff, latency profiling, bottleneck detection.

Compares any two mlx-lm compatible models across architecture, per-layer latency,
and generation quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import os

from .architecture import ModelArchitecture, format_param_count, format_size
from .inference_tracer import InferenceTrace, LayerTrace


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ArchDiffRow:
    field_name: str
    model_a_value: str
    model_b_value: str
    is_different: bool


@dataclass
class ArchDiff:
    model_a_name: str
    model_b_name: str
    rows: list[ArchDiffRow] = field(default_factory=list)


@dataclass
class LatencyProfile:
    model_dir: str
    model_name: str
    prefill_layer_attn_ms: list[float]   # per-layer attention latency (prefill)
    prefill_layer_mlp_ms: list[float]    # per-layer MLP latency (prefill)
    prefill_total_ms: float
    decode_layer_attn_ms: list[float]    # per-layer attention latency (decode avg)
    decode_layer_mlp_ms: list[float]     # per-layer MLP latency (decode avg)
    decode_total_ms: float               # average per-token decode time
    decode_steps: int
    tokens_per_second: float


@dataclass
class BottleneckLayer:
    layer_idx: int
    attn_ms: float
    mlp_ms: float
    total_ms: float
    pct_of_total: float
    bottleneck_type: str  # "attn", "mlp", "both"


@dataclass
class ModelComparisonResult:
    arch_diff: ArchDiff | None
    latency_a: LatencyProfile | None
    latency_b: LatencyProfile | None
    bottlenecks_a: list[BottleneckLayer]
    bottlenecks_b: list[BottleneckLayer]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def _resolve_text_config(config: dict) -> dict:
    """Resolve nested text_config / model_config for models like Qwen3.5, VLMs, etc.

    Many models nest architecture params under text_config or model_config.
    This merges the sub-config into a flat dict so field lookups work uniformly.
    Root-level keys take precedence (e.g. model_type stays from root).
    """
    for sub_key in ("text_config", "model_config", "language_config"):
        sub = config.get(sub_key)
        if isinstance(sub, dict):
            merged = dict(sub)
            merged.update({k: v for k, v in config.items() if k != sub_key})
            return merged
    return config


def compute_arch_diff(arch_a: ModelArchitecture, arch_b: ModelArchitecture) -> ArchDiff:
    """Compare two model architectures field-by-field."""
    cfg_a = _resolve_text_config(arch_a.config)
    cfg_b = _resolve_text_config(arch_b.config)

    name_a = os.path.basename(arch_a.model_dir.rstrip("/"))
    name_b = os.path.basename(arch_b.model_dir.rstrip("/"))

    fields = [
        ("Model Name", name_a, name_b),
        ("Model Type", cfg_a.get("model_type", "N/A"), cfg_b.get("model_type", "N/A")),
        ("Hidden Size", str(cfg_a.get("hidden_size", "N/A")), str(cfg_b.get("hidden_size", "N/A"))),
        ("Num Layers", str(cfg_a.get("num_hidden_layers", "N/A")), str(cfg_b.get("num_hidden_layers", "N/A"))),
        ("Intermediate Size", str(cfg_a.get("intermediate_size", "N/A")), str(cfg_b.get("intermediate_size", "N/A"))),
        ("Attention Heads", str(cfg_a.get("num_attention_heads", "N/A")), str(cfg_b.get("num_attention_heads", "N/A"))),
        ("KV Heads", str(cfg_a.get("num_key_value_heads", "N/A")), str(cfg_b.get("num_key_value_heads", "N/A"))),
        ("Head Dim", str(cfg_a.get("head_dim", 128)), str(cfg_b.get("head_dim", 128))),
        ("Vocab Size", str(cfg_a.get("vocab_size", "N/A")), str(cfg_b.get("vocab_size", "N/A"))),
        ("Total Params", format_param_count(arch_a.total_params), format_param_count(arch_b.total_params)),
        ("Disk Size", format_size(arch_a.total_size_bytes), format_size(arch_b.total_size_bytes)),
    ]

    # Quantization info
    quant_a = cfg_a.get("quantization") or cfg_a.get("quantization_config") or {}
    quant_b = cfg_b.get("quantization") or cfg_b.get("quantization_config") or {}
    bits_a = str(quant_a.get("bits", "none"))
    bits_b = str(quant_b.get("bits", "none"))
    fields.append(("Quantization Bits", bits_a, bits_b))

    group_a = str(quant_a.get("group_size", "N/A"))
    group_b = str(quant_b.get("group_size", "N/A"))
    fields.append(("Group Size", group_a, group_b))

    rows = [
        ArchDiffRow(name, val_a, val_b, val_a != val_b)
        for name, val_a, val_b in fields
    ]

    return ArchDiff(
        model_a_name=name_a,
        model_b_name=name_b,
        rows=rows,
    )


def compute_latency_profile(trace: InferenceTrace) -> LatencyProfile | None:
    """Extract per-layer latency profile from an InferenceTrace.

    Returns None if timing data is not available.
    """
    if not trace.enable_timing:
        return None

    num_layers = trace.num_layers

    # Prefill latencies
    prefill_attn = [0.0] * num_layers
    prefill_mlp = [0.0] * num_layers
    if trace.prefill_layer_traces:
        for lt in trace.prefill_layer_traces:
            if lt.layer_idx < num_layers:
                prefill_attn[lt.layer_idx] = lt.attn_latency_ms
                prefill_mlp[lt.layer_idx] = lt.mlp_latency_ms
    prefill_total = sum(prefill_attn) + sum(prefill_mlp)

    # Decode latencies (average across steps)
    decode_attn = [0.0] * num_layers
    decode_mlp = [0.0] * num_layers
    decode_count = 0

    for step in trace.steps:
        if not step.layers:
            continue
        decode_count += 1
        for lt in step.layers:
            if lt.layer_idx < num_layers:
                decode_attn[lt.layer_idx] += lt.attn_latency_ms
                decode_mlp[lt.layer_idx] += lt.mlp_latency_ms

    if decode_count > 0:
        decode_attn = [v / decode_count for v in decode_attn]
        decode_mlp = [v / decode_count for v in decode_mlp]

    decode_total = sum(decode_attn) + sum(decode_mlp)

    decode_time = trace.total_time_seconds - trace.prefill_time_seconds
    num_gen = len(trace.steps)
    tps = num_gen / decode_time if decode_time > 0 else 0.0

    return LatencyProfile(
        model_dir=trace.model_dir,
        model_name=trace.model_name,
        prefill_layer_attn_ms=prefill_attn,
        prefill_layer_mlp_ms=prefill_mlp,
        prefill_total_ms=prefill_total,
        decode_layer_attn_ms=decode_attn,
        decode_layer_mlp_ms=decode_mlp,
        decode_total_ms=decode_total,
        decode_steps=decode_count,
        tokens_per_second=tps,
    )


def identify_bottleneck_layers(
    profile: LatencyProfile,
    top_n: int = 5,
) -> list[BottleneckLayer]:
    """Identify the slowest layers from a latency profile."""
    if not profile.decode_layer_attn_ms:
        return []

    total_decode = profile.decode_total_ms
    if total_decode <= 0:
        return []

    layers = []
    for i in range(len(profile.decode_layer_attn_ms)):
        attn = profile.decode_layer_attn_ms[i]
        mlp = profile.decode_layer_mlp_ms[i]
        total = attn + mlp
        pct = total / total_decode * 100

        if attn > mlp * 1.5:
            btype = "attn"
        elif mlp > attn * 1.5:
            btype = "mlp"
        else:
            btype = "both"

        layers.append(BottleneckLayer(
            layer_idx=i,
            attn_ms=attn,
            mlp_ms=mlp,
            total_ms=total,
            pct_of_total=pct,
            bottleneck_type=btype,
        ))

    layers.sort(key=lambda b: b.total_ms, reverse=True)
    return layers[:top_n]


def run_comparison(
    arch_a: ModelArchitecture,
    arch_b: ModelArchitecture,
    trace_a: InferenceTrace | None = None,
    trace_b: InferenceTrace | None = None,
) -> ModelComparisonResult:
    """Run full model comparison combining arch diff and latency analysis."""
    arch_diff = compute_arch_diff(arch_a, arch_b)

    latency_a = compute_latency_profile(trace_a) if trace_a else None
    latency_b = compute_latency_profile(trace_b) if trace_b else None

    bottlenecks_a = identify_bottleneck_layers(latency_a) if latency_a else []
    bottlenecks_b = identify_bottleneck_layers(latency_b) if latency_b else []

    return ModelComparisonResult(
        arch_diff=arch_diff,
        latency_a=latency_a,
        latency_b=latency_b,
        bottlenecks_a=bottlenecks_a,
        bottlenecks_b=bottlenecks_b,
    )
