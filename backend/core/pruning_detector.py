# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Detect pruning and optimization traces in model configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PruningTrace:
    """A detected pruning or optimization modification."""
    category: str  # "vocab_pruning", "layer_pruning", "dimension_pruning", etc.
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # "info", "minor", "major"


def detect_pruning(config: dict[str, Any]) -> list[PruningTrace]:
    """Detect all pruning traces in a model config."""
    traces: list[PruningTrace] = []

    # Vocab pruning
    vp = config.get("vocab_pruning")
    if vp:
        orig = vp.get("original_text_vocab_size") or vp.get("original_vocab_size", 0)
        compact = vp.get("compact_vocab_size", 0)
        reduction = (1 - compact / orig) * 100 if orig > 0 else 0
        traces.append(PruningTrace(
            category="vocab_pruning",
            description=f"Vocabulary pruned: {orig:,} -> {compact:,} ({reduction:.1f}% reduction)",
            details=vp,
            severity="major",
        ))

    # Per-layer intermediate sizes (dimension pruning)
    for sub_key in ["text_config", "talker_config", ""]:
        sub_cfg = config.get(sub_key, config) if sub_key else config
        if not isinstance(sub_cfg, dict):
            continue
        plis = sub_cfg.get("per_layer_intermediate_sizes")
        if plis:
            default_size = sub_cfg.get("intermediate_size", 0)
            pruned_layers = [i for i, s in enumerate(plis) if s < default_size]
            if pruned_layers:
                traces.append(PruningTrace(
                    category="dimension_pruning",
                    description=f"Per-layer intermediate sizes: {len(pruned_layers)} layers pruned "
                                f"(default {default_size}, min {min(plis)})",
                    details={
                        "per_layer_sizes": plis,
                        "default_size": default_size,
                        "pruned_layers": pruned_layers,
                        "source": sub_key or "root",
                    },
                    severity="major",
                ))

    # Text layer pruning
    tlp = config.get("text_layer_pruning")
    if tlp:
        removed = tlp.get("removed_layers", [])
        old_num = tlp.get("old_num_layers", 0)
        new_num = tlp.get("new_num_layers", 0)
        traces.append(PruningTrace(
            category="layer_pruning",
            description=f"Text layers pruned: {old_num} -> {new_num} "
                        f"(removed layers: {removed})",
            details=tlp,
            severity="major",
        ))

    # Resolution reduction
    rr = config.get("resolution_reduction")
    if rr:
        old_size = rr.get("original_image_size", 0)
        new_size = rr.get("target_image_size", 0)
        traces.append(PruningTrace(
            category="resolution_reduction",
            description=f"Image resolution reduced: {old_size} -> {new_size}",
            details=rr,
            severity="minor",
        ))

    # Vision FC2 quantization
    vfq = config.get("vision_fc2_quantization")
    if vfq:
        traces.append(PruningTrace(
            category="vision_quantization",
            description=f"Vision FC2 layers quantized: {vfq.get('bits', '?')}-bit, "
                        f"saved {vfq.get('saved_mb', '?')} MB",
            details=vfq,
            severity="minor",
        ))

    # Weight split
    ws = config.get("weight_split")
    if ws and ws.get("enabled"):
        traces.append(PruningTrace(
            category="weight_split",
            description=f"Weights split: LM {ws.get('language_model_size_mb', '?')} MB, "
                        f"Vision {ws.get('vision_model_size_mb', '?')} MB",
            details=ws,
            severity="info",
        ))

    # Quantization (general)
    quant = config.get("quantization") or config.get("quantization_config")
    if quant:
        bits = quant.get("bits", "?")
        group_size = quant.get("group_size", "?")
        mode = quant.get("mode", "")
        traces.append(PruningTrace(
            category="quantization",
            description=f"Model quantized: {bits}-bit, group_size={group_size}"
                        + (f", mode={mode}" if mode else ""),
            details=quant,
            severity="info",
        ))

    return traces
