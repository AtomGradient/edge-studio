# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""MOE (Mixture of Experts) analyzer — expert utilization tracking and pruning simulation.

Captures routing decisions from MOE gate layers, computes per-expert utilization
metrics, load balance scores, and simulates expert pruning.

Supports: Llama 4 Scout/Maverick, DeepSeek V3, Mixtral, Qwen3-MOE.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpertTrace:
    """Routing decision for one token at one layer."""
    layer_idx: int
    expert_indices: np.ndarray     # [top_k] — selected expert indices
    expert_scores: np.ndarray      # [top_k] — routing scores for selected experts
    gate_logits: np.ndarray | None  # [num_experts] — raw gate logits (optional)
    routing_entropy: float          # entropy of routing distribution


@dataclass
class LayerExpertStats:
    """Aggregated expert statistics for one layer."""
    layer_idx: int
    num_experts: int
    token_counts: np.ndarray        # [num_experts] — tokens routed to each expert
    avg_scores: np.ndarray          # [num_experts] — average routing score per expert
    total_tokens: int
    load_balance_score: float       # 1.0 = perfect balance, lower = less balanced
    routing_entropy_mean: float     # mean entropy across tokens
    cold_experts: list[int]         # experts that received 0 tokens


@dataclass
class ExpertUtilization:
    """Full MOE utilization analysis."""
    num_layers: int
    num_experts: int
    num_experts_per_tok: int
    total_tokens: int
    layer_stats: list[LayerExpertStats]
    # Aggregated
    global_token_counts: np.ndarray  # [num_experts] — total across all layers
    global_avg_scores: np.ndarray    # [num_experts] — average across all layers
    overall_balance: float
    cold_expert_count: int           # experts with 0 tokens in ALL layers


@dataclass
class ExpertPruningResult:
    """Result of simulating expert pruning."""
    threshold: float                 # utilization threshold
    experts_to_remove: list[tuple[int, int]]  # [(layer, expert_idx)]
    experts_remaining: int
    total_experts: int
    estimated_size_saved_bytes: int
    retention_ratio: float


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_expert_utilization(
    traces: list[list[ExpertTrace]],  # [steps][layers]
    num_experts: int,
    num_experts_per_tok: int,
) -> ExpertUtilization:
    """Analyze expert utilization from collected routing traces.

    Args:
        traces: list of per-step trace lists (each step has one ExpertTrace per MOE layer)
        num_experts: total number of experts
        num_experts_per_tok: experts activated per token (top-K)

    Returns:
        ExpertUtilization with per-layer and global statistics.
    """
    if not traces:
        return ExpertUtilization(
            num_layers=0, num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            total_tokens=0, layer_stats=[],
            global_token_counts=np.zeros(num_experts),
            global_avg_scores=np.zeros(num_experts),
            overall_balance=0.0, cold_expert_count=num_experts,
        )

    # Group by layer
    layer_traces: dict[int, list[ExpertTrace]] = {}
    for step_traces in traces:
        for et in step_traces:
            layer_traces.setdefault(et.layer_idx, []).append(et)

    num_layers = len(layer_traces)
    total_tokens = len(traces)

    layer_stats = []
    global_counts = np.zeros(num_experts)
    global_score_sums = np.zeros(num_experts)
    global_score_counts = np.zeros(num_experts)

    for layer_idx in sorted(layer_traces.keys()):
        lt = layer_traces[layer_idx]
        counts = np.zeros(num_experts)
        score_sums = np.zeros(num_experts)
        entropies = []

        for et in lt:
            for i, eidx in enumerate(et.expert_indices):
                if 0 <= eidx < num_experts:
                    counts[eidx] += 1
                    score_sums[eidx] += et.expert_scores[i]
            entropies.append(et.routing_entropy)

        avg_scores = np.divide(score_sums, counts, where=counts > 0, out=np.zeros(num_experts))

        # Load balance score: (num_experts * sum(fi * Pi)) where fi = fraction of tokens
        # Perfect balance = 1.0
        fi = counts / max(counts.sum(), 1)
        pi = avg_scores / max(avg_scores.sum(), 1e-10)
        lb_score = float(num_experts * np.sum(fi * pi))

        cold = [int(i) for i in range(num_experts) if counts[i] == 0]

        layer_stats.append(LayerExpertStats(
            layer_idx=layer_idx,
            num_experts=num_experts,
            token_counts=counts,
            avg_scores=avg_scores,
            total_tokens=total_tokens,
            load_balance_score=lb_score,
            routing_entropy_mean=float(np.mean(entropies)) if entropies else 0.0,
            cold_experts=cold,
        ))

        global_counts += counts
        global_score_sums += score_sums
        global_score_counts += counts

    global_avg_scores = np.divide(
        global_score_sums, global_score_counts,
        where=global_score_counts > 0, out=np.zeros(num_experts),
    )

    # Overall balance across all layers
    total_fi = global_counts / max(global_counts.sum(), 1)
    total_pi = global_avg_scores / max(global_avg_scores.sum(), 1e-10)
    overall_balance = float(num_experts * np.sum(total_fi * total_pi))

    # Cold experts: experts with 0 tokens across ALL layers
    cold_count = int(np.sum(global_counts == 0))

    return ExpertUtilization(
        num_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        total_tokens=total_tokens,
        layer_stats=layer_stats,
        global_token_counts=global_counts,
        global_avg_scores=global_avg_scores,
        overall_balance=overall_balance,
        cold_expert_count=cold_count,
    )


def simulate_expert_pruning(
    utilization: ExpertUtilization,
    threshold: float = 0.05,
    model_size_bytes: int = 0,
) -> ExpertPruningResult:
    """Simulate removing experts below a utilization threshold.

    Args:
        utilization: expert utilization analysis
        threshold: fraction threshold — experts receiving fewer than
                   threshold * (total_tokens / num_experts) tokens are marked for removal
        model_size_bytes: total model size for savings estimation

    Returns:
        ExpertPruningResult with removal candidates and estimated savings.
    """
    expected_per_expert = utilization.total_tokens / max(utilization.num_experts, 1)
    cutoff = threshold * expected_per_expert

    to_remove = []
    for ls in utilization.layer_stats:
        for eidx in range(ls.num_experts):
            if ls.token_counts[eidx] < cutoff:
                to_remove.append((ls.layer_idx, eidx))

    total_experts = utilization.num_layers * utilization.num_experts
    remaining = total_experts - len(to_remove)
    retention = remaining / max(total_experts, 1)

    # Estimate size savings: each expert is roughly model_size / (num_layers * num_experts) of MLP
    # MLP is roughly 2/3 of each layer's parameters
    if model_size_bytes > 0 and total_experts > 0:
        mlp_fraction = 0.66
        per_expert_bytes = int(model_size_bytes * mlp_fraction / total_experts)
        saved = len(to_remove) * per_expert_bytes
    else:
        saved = 0

    return ExpertPruningResult(
        threshold=threshold,
        experts_to_remove=to_remove,
        experts_remaining=remaining,
        total_experts=total_experts,
        estimated_size_saved_bytes=saved,
        retention_ratio=retention,
    )


def compute_routing_entropy(scores: np.ndarray) -> float:
    """Compute entropy of routing probability distribution."""
    # Ensure valid probability distribution
    p = np.clip(scores, 1e-10, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def detect_moe_config(model_dir: str) -> dict[str, Any] | None:
    """Detect MOE configuration from model config.json.

    Handles both flat configs (most LLMs) and nested configs where MoE
    fields live under ``text_config`` (VLM-shaped MoEs like Qwen3.5-MoE).
    See Memory ``feedback_nested_config`` for the canonical pattern.

    Returns dict with num_experts, num_experts_per_tok, or None if not MOE.
    """
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    # MoE fields may live under text_config for VLM-shaped MoEs.
    text_config = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}

    def _pick(*keys: str, default: Any = None) -> Any:
        for k in keys:
            if k in config and config[k] not in (None, 0):
                return config[k]
            if k in text_config and text_config[k] not in (None, 0):
                return text_config[k]
        return default

    num_experts = _pick("num_local_experts", "num_experts", "n_routed_experts", default=0)
    if not num_experts:
        return None

    num_experts_per_tok = _pick("num_experts_per_tok", "n_group_top_k", "top_k", default=2)
    num_layers = _pick("num_hidden_layers", default=0)
    model_type = config.get("model_type") or text_config.get("model_type", "unknown")

    return {
        "num_experts": int(num_experts),
        "num_experts_per_tok": int(num_experts_per_tok),
        "num_layers": int(num_layers),
        "model_type": model_type,
    }
