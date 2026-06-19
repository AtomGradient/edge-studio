# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Attention pattern analyzer — classify head attention patterns from inference traces.

Analyzes per-head attention weight distributions to identify SINK, LOCAL, GLOBAL,
and SPARSE patterns. Uses majority voting across decode steps for robust classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class HeadPattern(Enum):
    SINK = "sink"
    LOCAL = "local"
    GLOBAL = "global"
    SPARSE = "sparse"


PATTERN_COLORS = {
    HeadPattern.SINK: "#FF9800",
    HeadPattern.LOCAL: "#2196F3",
    HeadPattern.GLOBAL: "#4CAF50",
    HeadPattern.SPARSE: "#9E9E9E",
}

PATTERN_DESCRIPTIONS = {
    HeadPattern.SINK: "Attention concentrated on first few positions (attention sink)",
    HeadPattern.LOCAL: "Attention focused on recent/nearby tokens",
    HeadPattern.GLOBAL: "Attention distributed evenly across all positions",
    HeadPattern.SPARSE: "Attention concentrated on specific scattered positions",
}


@dataclass
class HeadClassification:
    """Classification result for a single attention head."""
    layer_idx: int
    head_idx: int
    pattern: HeadPattern
    confidence: float
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionAnalysisResult:
    """Complete attention pattern analysis result."""
    classifications: list[HeadClassification] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    suggestions: list[dict[str, Any]] = field(default_factory=list)

    def pattern_matrix(self, num_layers: int, num_heads: int) -> np.ndarray:
        """Return [num_layers, num_heads] matrix of pattern enum values (0-3)."""
        matrix = np.full((num_layers, num_heads), -1, dtype=int)
        pattern_to_int = {
            HeadPattern.SINK: 0,
            HeadPattern.LOCAL: 1,
            HeadPattern.GLOBAL: 2,
            HeadPattern.SPARSE: 3,
        }
        for c in self.classifications:
            if c.layer_idx < num_layers and c.head_idx < num_heads:
                matrix[c.layer_idx, c.head_idx] = pattern_to_int[c.pattern]
        return matrix

    def confidence_matrix(self, num_layers: int, num_heads: int) -> np.ndarray:
        """Return [num_layers, num_heads] matrix of confidence values."""
        matrix = np.zeros((num_layers, num_heads))
        for c in self.classifications:
            if c.layer_idx < num_layers and c.head_idx < num_heads:
                matrix[c.layer_idx, c.head_idx] = c.confidence
        return matrix


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_single_head(
    attn_weights: np.ndarray,
    seq_len: int,
) -> tuple[HeadPattern, float, dict]:
    """Classify a single head's attention pattern from one step.

    Args:
        attn_weights: [seq_len] attention weights for last token over all positions
        seq_len: sequence length

    Returns:
        (pattern, confidence, stats_dict)
    """
    if len(attn_weights) == 0:
        return HeadPattern.SPARSE, 0.0, {}

    w = attn_weights.astype(np.float64)
    w = np.clip(w, 0, None)
    total = w.sum()
    if total < 1e-12:
        return HeadPattern.SPARSE, 0.0, {}
    w = w / total

    n = len(w)
    stats: dict[str, Any] = {}

    # SINK: first 3 positions hold >50% attention
    sink_positions = min(3, n)
    sink_mass = float(w[:sink_positions].sum())
    stats["sink_mass"] = sink_mass

    # LOCAL: last min(20, n//4) positions hold >60% attention
    local_window = min(20, max(1, n // 4))
    local_mass = float(w[-local_window:].sum())
    stats["local_mass"] = local_mass
    stats["local_window"] = local_window

    # GLOBAL: normalized entropy > 0.8
    # Entropy normalized by log(n)
    eps = 1e-12
    entropy = -float(np.sum(w * np.log(w + eps)))
    max_entropy = np.log(n + eps)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    stats["norm_entropy"] = norm_entropy

    # Classification with priority: SINK > LOCAL > GLOBAL > SPARSE
    if sink_mass > 0.5:
        return HeadPattern.SINK, sink_mass, stats
    if local_mass > 0.6:
        return HeadPattern.LOCAL, local_mass, stats
    if norm_entropy > 0.8:
        return HeadPattern.GLOBAL, norm_entropy, stats

    return HeadPattern.SPARSE, 1.0 - norm_entropy, stats


def classify_attention_heads(trace) -> AttentionAnalysisResult:
    """Classify all attention heads using majority voting across decode steps.

    Args:
        trace: InferenceTrace with step-level attention weights

    Returns:
        AttentionAnalysisResult with per-head classifications and suggestions
    """
    num_layers = trace.num_layers
    num_heads = trace.num_heads

    # Collect per-step classifications for each (layer, head)
    # votes[layer][head] = list of HeadPattern
    votes: dict[tuple[int, int], list[HeadPattern]] = {}
    all_stats: dict[tuple[int, int], list[dict]] = {}

    for step in trace.steps:
        if not step.layers:
            continue
        for lt in step.layers:
            attn = lt.attn_weights  # [num_heads, seq_len]
            if attn is None or len(attn.shape) < 2:
                continue
            idx = lt.layer_idx
            if idx >= num_layers:
                continue
            nh = min(attn.shape[0], num_heads)
            seq_len = attn.shape[1]
            for h in range(nh):
                key = (idx, h)
                pattern, conf, stats = classify_single_head(attn[h], seq_len)
                votes.setdefault(key, []).append(pattern)
                all_stats.setdefault(key, []).append(stats)

    # Majority voting
    classifications = []
    pattern_counts = {p: 0 for p in HeadPattern}

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            key = (layer_idx, head_idx)
            head_votes = votes.get(key, [])
            if not head_votes:
                classifications.append(HeadClassification(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    pattern=HeadPattern.SPARSE,
                    confidence=0.0,
                ))
                pattern_counts[HeadPattern.SPARSE] += 1
                continue

            # Count votes per pattern
            counts = {}
            for v in head_votes:
                counts[v] = counts.get(v, 0) + 1

            winner = max(counts, key=counts.get)
            confidence = counts[winner] / len(head_votes)

            # Average stats for the winning pattern
            head_stats_list = all_stats.get(key, [])
            avg_stats = {}
            if head_stats_list:
                all_keys = head_stats_list[0].keys()
                for k in all_keys:
                    vals = [s[k] for s in head_stats_list if k in s]
                    if vals:
                        avg_stats[k] = float(np.mean(vals))

            classifications.append(HeadClassification(
                layer_idx=layer_idx,
                head_idx=head_idx,
                pattern=winner,
                confidence=confidence,
                stats=avg_stats,
            ))
            pattern_counts[winner] += 1

    # Summary
    total_heads = num_layers * num_heads
    summary = {
        "total_heads": total_heads,
        "num_layers": num_layers,
        "num_heads_per_layer": num_heads,
    }
    for p in HeadPattern:
        count = pattern_counts[p]
        summary[p.value] = count
        summary[f"{p.value}_pct"] = count / max(total_heads, 1) * 100

    # Generate suggestions
    suggestions = _generate_suggestions(pattern_counts, total_heads, num_layers, num_heads, classifications)

    return AttentionAnalysisResult(
        classifications=classifications,
        summary=summary,
        suggestions=suggestions,
    )


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------

def _generate_suggestions(
    pattern_counts: dict[HeadPattern, int],
    total_heads: int,
    num_layers: int,
    num_heads: int,
    classifications: list[HeadClassification],
) -> list[dict[str, Any]]:
    """Generate optimization suggestions based on attention pattern distribution."""
    suggestions = []

    sink_pct = pattern_counts[HeadPattern.SINK] / max(total_heads, 1)
    local_pct = pattern_counts[HeadPattern.LOCAL] / max(total_heads, 1)
    sparse_pct = pattern_counts[HeadPattern.SPARSE] / max(total_heads, 1)

    if sink_pct > 0.2:
        suggestions.append({
            "title": "Attention Sink Optimization",
            "description": (
                f"{pattern_counts[HeadPattern.SINK]} heads ({sink_pct:.0%}) exhibit attention sink pattern. "
                f"Consider StreamingLLM-style attention sink token retention for long-context inference, "
                f"or sink-aware KV cache eviction to reduce memory usage."
            ),
            "priority": "medium",
            "category": "attention_sink",
        })

    if local_pct > 0.3:
        suggestions.append({
            "title": "Sliding Window Attention",
            "description": (
                f"{pattern_counts[HeadPattern.LOCAL]} heads ({local_pct:.0%}) focus on local/recent tokens. "
                f"A sliding window attention mechanism could reduce KV cache memory without quality loss. "
                f"Consider replacing full attention with local attention for these heads."
            ),
            "priority": "medium",
            "category": "sliding_window",
        })

    if sparse_pct > 0.3:
        # Find layers with many sparse heads — candidates for head pruning
        sparse_per_layer = {}
        for c in classifications:
            if c.pattern == HeadPattern.SPARSE:
                sparse_per_layer.setdefault(c.layer_idx, []).append(c.head_idx)
        high_sparse_layers = [
            (l, heads) for l, heads in sparse_per_layer.items()
            if len(heads) > num_heads * 0.5
        ]

        layer_info = ""
        if high_sparse_layers:
            layer_nums = [str(l) for l, _ in high_sparse_layers[:5]]
            layer_info = f" Layers with most sparse heads: {', '.join(layer_nums)}."

        suggestions.append({
            "title": "Attention Head Pruning Candidate",
            "description": (
                f"{pattern_counts[HeadPattern.SPARSE]} heads ({sparse_pct:.0%}) show sparse attention patterns. "
                f"These heads may be candidates for pruning or GQA grouping.{layer_info}"
            ),
            "priority": "low",
            "category": "head_pruning",
        })

    # Dominant pattern info
    dominant = max(pattern_counts, key=pattern_counts.get)
    if pattern_counts[dominant] > total_heads * 0.5:
        suggestions.append({
            "title": f"Dominant Pattern: {dominant.value.upper()}",
            "description": (
                f"Over {pattern_counts[dominant] / total_heads:.0%} of attention heads "
                f"exhibit {dominant.value} pattern. {PATTERN_DESCRIPTIONS[dominant]}"
            ),
            "priority": "low",
            "category": "info",
        })

    return suggestions
