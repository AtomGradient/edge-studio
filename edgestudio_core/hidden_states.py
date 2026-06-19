# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hidden-state extraction helpers for public ``mlx-lm`` models."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


class HiddenStatesUnavailable(RuntimeError):
    """Raised when a model does not expose a supported hidden-state path."""


def _unwrap_backbone(backbone: Any) -> Any:
    for path in (
        (),
        ("model",),
        ("language_model", "model"),
        ("language_model",),
    ):
        candidate = backbone
        for attr in path:
            candidate = getattr(candidate, attr, None)
            if candidate is None:
                break
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    return backbone


def forward_layer_hidden(
    backbone: Any,
    input_ids: Any,
    *,
    layer_index: int,
    input_embeddings: Any | None = None,
) -> Any:
    """Return the post-block hidden state for one transformer layer.

    This covers the Qwen3.5 public ``mlx-lm`` backbone that EdgeStudio uses for
    route-router embedding extraction. It does not modify model weights.
    """

    model = _unwrap_backbone(backbone)
    layers = getattr(model, "layers", None)
    if layers is None:
        return _forward_legacy_capture_hook(backbone, input_ids, layer_index)
    if layer_index < 0 or layer_index >= len(layers):
        raise HiddenStatesUnavailable(
            f"layer_index {layer_index} outside range 0..{len(layers) - 1}"
        )

    if input_embeddings is not None:
        hidden_states = input_embeddings
    elif hasattr(model, "embed_tokens"):
        hidden_states = model.embed_tokens(input_ids)
    else:
        raise HiddenStatesUnavailable(
            f"{type(model).__name__} does not expose embed_tokens"
        )

    cache = [None] * len(layers)
    try:
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        fa_idx = getattr(model, "fa_idx", 0)
        ssm_idx = getattr(model, "ssm_idx", 0)
        fa_mask = create_attention_mask(hidden_states, cache[fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[ssm_idx])
    except Exception:
        fa_mask = None
        ssm_mask = None

    for index, layer in enumerate(layers):
        mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=cache[index])
        if index == layer_index:
            return mx.stop_gradient(hidden_states)

    raise HiddenStatesUnavailable(f"Layer {layer_index} was not reached")


def _forward_legacy_capture_hook(backbone: Any, input_ids: Any, layer_index: int) -> Any:
    try:
        output = backbone(input_ids, return_hidden_states=[layer_index])
    except TypeError as exc:
        raise HiddenStatesUnavailable(
            f"{type(backbone).__name__} does not expose a layers attribute"
        ) from exc

    captured = None
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        captured = output[1]
    elif hasattr(output, "hidden_states"):
        hidden_states = output.hidden_states
        if isinstance(hidden_states, dict):
            captured = hidden_states
        elif len(hidden_states) > layer_index:
            return hidden_states[layer_index]
    if not isinstance(captured, dict) or layer_index not in captured:
        raise HiddenStatesUnavailable(
            f"Backbone did not return hidden state for layer {layer_index}"
        )
    return captured[layer_index]
