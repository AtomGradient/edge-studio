# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hidden-state extraction boundary for the bundled edgestudio_core runtime."""

from __future__ import annotations

from typing import Any


class HiddenStatesUnavailable(RuntimeError):
    """Raised when a layer hidden state was requested but no backend supports it."""


def layer_hidden(backbone: Any, input_ids: Any, layer_index: int) -> Any:
    """Return one hidden-state layer from an MLX backbone.

    Public ``mlx-lm`` does not expose EdgeStudio's historical
    ``return_hidden_states`` hook. Prefer the bundled ``edgestudio_core``
    runtime and keep the old fork keyword as a temporary compatibility path.
    """
    try:
        from edgestudio_core.hidden_states import forward_layer_hidden
    except ImportError:
        forward_layer_hidden = None

    if forward_layer_hidden is not None:
        return forward_layer_hidden(
            backbone,
            input_ids,
            layer_index=layer_index,
        )

    try:
        output = backbone(input_ids, return_hidden_states=[layer_index])
    except TypeError as exc:
        raise HiddenStatesUnavailable(
            "Layer hidden-state extraction requires the bundled edgestudio_core "
            "runtime. Reinstall edgestudio or use layer_index=-1."
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
            f"MLX backbone did not return hidden states for layer {layer_index}"
        )
    return captured[layer_index]
