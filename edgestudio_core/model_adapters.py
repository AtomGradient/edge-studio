# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Runtime adapters for public ``mlx-lm`` model classes."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def install_qwen35_rpp_adapters() -> None:
    """Install Qwen3.5 hidden-state and residual hooks on public ``mlx-lm``.

    The adapter mirrors the historical fork API for callers that directly pass
    ``return_hidden_states`` or ``stack_residuals`` to Qwen3.5 models. It is
    intentionally opt-in; EdgeStudio can also use ``hidden_states.forward_layer_hidden``
    without global patching.
    """

    try:
        import mlx_lm.models.qwen3_5 as qwen35
    except Exception:
        return

    text_cls = getattr(qwen35, "Qwen3_5TextModel", None)
    wrapper_cls = getattr(qwen35, "TextModel", None)
    model_cls = getattr(qwen35, "Model", None)
    if text_cls is not None and not getattr(text_cls, "_edgestudio_core_rpp", False):
        _patch_qwen35_text_model(text_cls)
    if wrapper_cls is not None and not getattr(wrapper_cls, "_edgestudio_core_rpp", False):
        _patch_qwen35_text_wrapper(wrapper_cls)
    if model_cls is not None and not getattr(model_cls, "_edgestudio_core_rpp", False):
        _patch_qwen35_model(model_cls)


def _patch_qwen35_text_model(cls: type) -> None:
    original = cls.__call__

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        return_hidden_states=None,
        stack_residuals=None,
    ):
        if return_hidden_states is None and stack_residuals is None:
            return original(
                self,
                inputs,
                cache=cache,
                input_embeddings=input_embeddings,
            )

        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)

        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        fa_mask = create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])

        if stack_residuals is not None and len(stack_residuals) != len(self.layers):
            raise ValueError(
                f"stack_residuals length {len(stack_residuals)} != "
                f"num_hidden_layers {len(self.layers)}"
            )

        captured = {} if return_hidden_states is not None else None
        capture_set = set(return_hidden_states or [])
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
            if stack_residuals is not None:
                residual = stack_residuals[index]
                if residual is not None:
                    hidden_states = hidden_states + residual
            if captured is not None and index in capture_set:
                captured[index] = mx.stop_gradient(hidden_states)

        out = self.norm(hidden_states)
        return (out, captured) if captured is not None else out

    cls.__call__ = __call__
    cls._edgestudio_core_rpp = True
    cls._edgestudio_core_original_call = original


def _patch_qwen35_text_wrapper(cls: type) -> None:
    original = cls.__call__

    def __call__(self, inputs, cache=None, input_embeddings=None, stack_residuals=None):
        if stack_residuals is None:
            return original(
                self,
                inputs,
                cache=cache,
                input_embeddings=input_embeddings,
            )
        out = self.model(
            inputs,
            cache,
            input_embeddings=input_embeddings,
            stack_residuals=stack_residuals,
        )
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    cls.__call__ = __call__
    cls._edgestudio_core_rpp = True
    cls._edgestudio_core_original_call = original


def _patch_qwen35_model(cls: type) -> None:
    original = cls.__call__

    def __call__(self, inputs, cache=None, input_embeddings=None, stack_residuals=None):
        if stack_residuals is None:
            return original(
                self,
                inputs,
                cache=cache,
                input_embeddings=input_embeddings,
            )
        return self.language_model(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            stack_residuals=stack_residuals,
        )

    cls.__call__ = __call__
    cls._edgestudio_core_rpp = True
    cls._edgestudio_core_original_call = original


def install_all_model_adapters() -> None:
    install_qwen35_rpp_adapters()


__all__ = ["install_all_model_adapters", "install_qwen35_rpp_adapters"]
