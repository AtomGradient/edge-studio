# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""VLM overlays for expert-routing analysis."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np


def install_qwen35_moe_routing_capture() -> None:
    """Install expert-routing capture on public ``mlx-vlm`` Qwen3.5-MoE."""

    try:
        import mlx_vlm.models.qwen3_5_moe.language as language
    except Exception:
        return

    cls = getattr(language, "Qwen3_5MoeSparseMoeBlock", None)
    if cls is None or getattr(cls, "_edgestudio_core_routing_capture", False):
        return

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._capture_routing = False
        self._captured_routing = []

    def __call__(self, x):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        top_k = self.top_k
        inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        if getattr(self, "_capture_routing", False):
            inds_f = inds.astype(mx.int32)
            scores_f = scores.astype(mx.float32)
            gates_f = gates.astype(mx.float32)
            mx.eval(inds_f, scores_f, gates_f)
            gates_np = np.asarray(gates_f).astype(np.float64)
            p = np.clip(gates_np, 1e-10, 1.0)
            entropies = -np.sum(p * np.log2(p), axis=-1)
            self._captured_routing.append(
                {
                    "inds": np.asarray(inds_f),
                    "scores": np.asarray(scores_f),
                    "entropies": entropies.astype(np.float32),
                }
            )

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls._edgestudio_core_routing_capture = True
    cls._edgestudio_core_original_init = original_init
    cls._edgestudio_core_original_call = original_call


def _iter_modules(root: Any):
    if root is None:
        return
    yield root
    if hasattr(root, "named_modules"):
        try:
            for _, module in root.named_modules():
                yield module
            return
        except Exception:
            pass
    for value in getattr(root, "__dict__", {}).values():
        if isinstance(value, (list, tuple)):
            for item in value:
                if hasattr(item, "__dict__"):
                    yield from _iter_modules(item)
        elif hasattr(value, "__dict__") and value is not root:
            yield from _iter_modules(value)


def enable_routing_capture_hooks(model: Any) -> int:
    """Ensure loaded MoE blocks expose capture fields.

    Returns the number of blocks prepared for tracing.
    """

    install_qwen35_moe_routing_capture()
    count = 0
    for module in _iter_modules(model):
        if module.__class__.__name__ == "Qwen3_5MoeSparseMoeBlock":
            if not hasattr(module, "_capture_routing"):
                module._capture_routing = False
            if not hasattr(module, "_captured_routing"):
                module._captured_routing = []
            count += 1
    return count


__all__ = ["enable_routing_capture_hooks", "install_qwen35_moe_routing_capture"]
