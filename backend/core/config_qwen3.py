# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Config parser for standard Qwen3 causal language models."""

from __future__ import annotations

from typing import Any

from .architecture import ArchNode, ModelArchitecture
from .config_parser import ConfigParser
from .weight_loader import WeightIndex


class Qwen3ConfigParser(ConfigParser):
    """Parse standard Qwen3 (CausalLM) model configuration.

    Flat structure: single Transformer stack with embedding + layers + norm + lm_head.
    """

    def model_type_name(self) -> str:
        return "Qwen3"

    def parse(self) -> ModelArchitecture:
        cfg = self.config
        num_layers = cfg.get("num_hidden_layers", 36)
        hidden_size = cfg.get("hidden_size", 2560)
        intermediate_size = cfg.get("intermediate_size", 9728)
        num_heads = cfg.get("num_attention_heads", 32)
        num_kv_heads = cfg.get("num_key_value_heads", 8)
        head_dim = cfg.get("head_dim", 128)
        vocab_size = cfg.get("vocab_size", 151936)

        per_layer_sizes = cfg.get("per_layer_intermediate_sizes")

        layer_config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_act": cfg.get("hidden_act", "silu"),
        }

        layers_group = self._make_layer_group(
            name="Transformer Layers",
            prefix="model.layers",
            num_layers=num_layers,
            layer_config=layer_config,
            per_layer_intermediate_sizes=per_layer_sizes,
        )

        embed_node = self._make_prefix_node("Embeddings", "model.embed_tokens")
        norm_node = self._make_misc_node("Final Norm", ["model.norm.weight"], "model.norm")

        lm_logical, lm_stored, lm_head_size = self._count_params_for_prefix("lm_head")
        tied = cfg.get("tie_word_embeddings", False)
        lm_head_node = ArchNode(
            name="LM Head" + (" (tied)" if tied and lm_logical == 0 else ""),
            node_type="module",
            weight_prefix="lm_head",
            param_count=lm_logical,
            stored_param_count=lm_stored,
            size_bytes=lm_head_size,
            extra={"tied": tied and lm_logical == 0},
        )

        root = ArchNode(
            name=cfg.get("architectures", ["Qwen3ForCausalLM"])[0],
            node_type="model",
            weight_prefix="",
            config_params={
                "model_type": cfg.get("model_type"),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "intermediate_size": intermediate_size,
                "head_dim": head_dim,
                "vocab_size": vocab_size,
                "hidden_act": cfg.get("hidden_act", "silu"),
                "max_position_embeddings": cfg.get("max_position_embeddings"),
                "rope_theta": cfg.get("rope_theta"),
                "tie_word_embeddings": tied,
            },
        )
        root.children = [embed_node, layers_group, norm_node, lm_head_node]

        quant = cfg.get("quantization") or cfg.get("quantization_config")

        return ModelArchitecture(
            model_type="qwen3",
            model_name=cfg.get("architectures", ["Qwen3ForCausalLM"])[0],
            model_dir=self.model_dir,
            root=root,
            config=cfg,
            quantization=quant,
            total_params=root.total_param_count,
            total_stored_params=root.total_stored_param_count,
            total_size_bytes=root.total_size_bytes,
        )

    def _make_prefix_node(self, name: str, prefix: str) -> ArchNode:
        logical, stored, size = self._count_params_for_prefix(prefix)
        return ArchNode(name=name, node_type="module", weight_prefix=prefix,
                        param_count=logical, stored_param_count=stored, size_bytes=size)

    def _make_misc_node(self, name: str, tensor_names: list[str], prefix: str = "") -> ArchNode:
        logical, stored, size = self._count_exact_tensors(tensor_names)
        return ArchNode(name=name, node_type="module", weight_prefix=prefix,
                        param_count=logical, stored_param_count=stored, size_bytes=size)
