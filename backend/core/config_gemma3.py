# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Config parser for Gemma 3 multimodal models."""

from __future__ import annotations

from typing import Any

from .architecture import ArchNode, ModelArchitecture
from .config_parser import ConfigParser
from .weight_loader import WeightIndex


class Gemma3ConfigParser(ConfigParser):
    """Parse Gemma 3 model configuration.

    Handles the multimodal structure:
    - Root: Gemma3ForConditionalGeneration
      - Language Model: 34-layer Transformer (GELU, GQA 8Q/4KV)
      - Vision Tower: 27-layer SigLIP Vision Model
      - Multi-Modal Projector
    """

    def model_type_name(self) -> str:
        return "Gemma 3"

    def parse(self) -> ModelArchitecture:
        text_cfg = self.config.get("text_config", {})
        vision_cfg = self.config.get("vision_config", {})

        root = ArchNode(
            name=self.config.get("architectures", ["Gemma3"])[0],
            node_type="model",
            weight_prefix="",
            config_params={
                "model_type": self.config.get("model_type"),
                "tie_word_embeddings": self.config.get("tie_word_embeddings", False),
                "mm_tokens_per_image": self.config.get("mm_tokens_per_image", 256),
                "image_token_index": self.config.get("image_token_index"),
            },
        )

        # ---- Language Model ----
        lm_node = self._build_language_model(text_cfg)
        root.children.append(lm_node)

        # ---- Vision Tower ----
        vt_node = self._build_vision_tower(vision_cfg)
        root.children.append(vt_node)

        # ---- Multi-Modal Projector ----
        proj_node = self._make_prefix_node("Multi-Modal Projector", "multi_modal_projector")
        root.children.append(proj_node)

        quant = self.config.get("quantization") or self.config.get("quantization_config")

        arch = ModelArchitecture(
            model_type="gemma3",
            model_name=self.config.get("architectures", ["Gemma3"])[0],
            model_dir=self.model_dir,
            root=root,
            config=self.config,
            quantization=quant,
            total_params=root.total_param_count,
            total_stored_params=root.total_stored_param_count,
            total_size_bytes=root.total_size_bytes,
        )
        return arch

    def _build_language_model(self, text_cfg: dict) -> ArchNode:
        num_layers = text_cfg.get("num_hidden_layers", 34)
        hidden_size = text_cfg.get("hidden_size", 2560)
        intermediate_size = text_cfg.get("intermediate_size", 10240)
        num_heads = text_cfg.get("num_attention_heads", 8)
        num_kv_heads = text_cfg.get("num_key_value_heads", 4)
        head_dim = text_cfg.get("head_dim", 256)
        vocab_size = text_cfg.get("vocab_size", 262208)

        per_layer_sizes = text_cfg.get("per_layer_intermediate_sizes") or self.config.get("per_layer_intermediate_sizes")

        layer_config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_activation": text_cfg.get("hidden_activation", "gelu_pytorch_tanh"),
            "sliding_window": text_cfg.get("sliding_window", 1024),
            "sliding_window_pattern": text_cfg.get("sliding_window_pattern", 6),
        }

        layers_group = self._make_layer_group(
            name="Transformer Layers",
            prefix="language_model.model.layers",
            num_layers=num_layers,
            layer_config=layer_config,
            per_layer_intermediate_sizes=per_layer_sizes,
        )

        embed_node = self._make_prefix_node("Embeddings", "language_model.model.embed_tokens")
        norm_node = self._make_misc_node(
            "Final Norm",
            ["language_model.model.norm.weight"],
            "language_model.model.norm",
        )

        # Check for separate lm_head or tied embeddings
        lm_logical, lm_stored, lm_head_size = self._count_params_for_prefix("language_model.lm_head")
        lm_head_node = ArchNode(
            name="LM Head" + (" (tied)" if lm_logical == 0 else ""),
            node_type="module",
            weight_prefix="language_model.lm_head",
            param_count=lm_logical,
            stored_param_count=lm_stored,
            size_bytes=lm_head_size,
            extra={"tied": lm_logical == 0},
        )

        lm_node = ArchNode(
            name="Language Model",
            node_type="submodel",
            weight_prefix="language_model",
            config_params={
                "model_type": text_cfg.get("model_type", "gemma3_text"),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "intermediate_size": intermediate_size,
                "head_dim": head_dim,
                "vocab_size": vocab_size,
                "hidden_activation": text_cfg.get("hidden_activation", "gelu_pytorch_tanh"),
                "max_position_embeddings": text_cfg.get("max_position_embeddings", 131072),
                "rope_theta": text_cfg.get("rope_theta", 1000000),
                "sliding_window": text_cfg.get("sliding_window", 1024),
                "query_pre_attn_scalar": text_cfg.get("query_pre_attn_scalar", 256),
                "tie_word_embeddings": text_cfg.get("tie_word_embeddings", True),
            },
        )
        lm_node.children = [embed_node, layers_group, norm_node, lm_head_node]

        return lm_node

    def _build_vision_tower(self, vision_cfg: dict) -> ArchNode:
        num_layers = vision_cfg.get("num_hidden_layers", 27)
        hidden_size = vision_cfg.get("hidden_size", 1152)
        intermediate_size = vision_cfg.get("intermediate_size", 4304)
        num_heads = vision_cfg.get("num_attention_heads", 16)
        image_size = vision_cfg.get("image_size", 896)
        patch_size = vision_cfg.get("patch_size", 14)

        layer_config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "hidden_act": vision_cfg.get("hidden_act", "gelu_pytorch_tanh"),
        }

        layers_group = self._make_layer_group(
            name="Encoder Layers",
            prefix="vision_tower.vision_model.encoder.layers",
            num_layers=num_layers,
            layer_config=layer_config,
        )

        embed_node = self._make_prefix_node(
            "Patch Embeddings",
            "vision_tower.vision_model.embeddings",
        )
        post_ln = self._make_misc_node(
            "Post LayerNorm",
            ["vision_tower.vision_model.post_layernorm.weight",
             "vision_tower.vision_model.post_layernorm.bias"],
            "vision_tower.vision_model.post_layernorm",
        )

        vt_node = ArchNode(
            name="Vision Tower (SigLIP)",
            node_type="submodel",
            weight_prefix="vision_tower",
            config_params={
                "model_type": vision_cfg.get("model_type", "siglip_vision_model"),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "intermediate_size": intermediate_size,
                "image_size": image_size,
                "patch_size": patch_size,
                "num_channels": vision_cfg.get("num_channels", 3),
                "num_patches": (image_size // patch_size) ** 2,
            },
        )
        vt_node.children = [embed_node, layers_group, post_ln]

        return vt_node

    def _make_prefix_node(self, name: str, prefix: str) -> ArchNode:
        logical, stored, size = self._count_params_for_prefix(prefix)
        return ArchNode(
            name=name,
            node_type="module",
            weight_prefix=prefix,
            param_count=logical,
            stored_param_count=stored,
            size_bytes=size,
        )

    def _make_misc_node(self, name: str, tensor_names: list[str], prefix: str = "") -> ArchNode:
        logical, stored, size = self._count_exact_tensors(tensor_names)
        return ArchNode(
            name=name,
            node_type="module",
            weight_prefix=prefix,
            param_count=logical,
            stored_param_count=stored,
            size_bytes=size,
        )
