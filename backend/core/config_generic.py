# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Generic config parser — auto-detects architecture from config.json + tensor names.

Works with any mlx-lm compatible model. Infers layer structure from tensor name
prefixes and config.json fields. Supports MOE models (num_experts, num_local_experts).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .architecture import ArchNode, ModelArchitecture
from .config_parser import ConfigParser
from .weight_loader import WeightIndex


class GenericConfigParser(ConfigParser):
    """Generic config parser for any mlx-lm compatible model.

    Infers architecture from:
    1. config.json standard fields (num_hidden_layers, hidden_size, etc.)
    2. Tensor name prefixes (model.layers.X.self_attn, model.layers.X.mlp, etc.)
    3. MOE indicators (num_experts, num_local_experts, block_sparse_moe)
    """

    def model_type_name(self) -> str:
        model_type = self.config.get("model_type", "unknown")
        # Capitalize first letter of each word
        return model_type.replace("_", " ").title()

    def parse(self) -> ModelArchitecture:
        cfg = self.config
        # Handle nested configs: VLMs use text_config, TTS uses talker_config
        tc = cfg.get("text_config") or cfg.get("talker_config") or cfg

        def _get(key: str, fallback=0):
            return tc.get(key) or cfg.get(key) or fallback

        # Standard transformer fields
        num_layers        = _get("num_hidden_layers") or _get("num_layers")
        hidden_size       = _get("hidden_size") or _get("d_model")
        intermediate_size = _get("intermediate_size") or _get("d_ff")
        num_heads         = _get("num_attention_heads") or _get("n_head")
        num_kv_heads      = _get("num_key_value_heads") or _get("num_kv_heads") or num_heads
        head_dim          = _get("head_dim") or (hidden_size // num_heads if num_heads else 128)
        vocab_size        = _get("vocab_size")

        # Whisper / audio model fallbacks (HuggingFace and MLX-native formats)
        # encoder_layers + d_model → num_layers + hidden_size
        if not num_layers:
            num_layers = _get("encoder_layers") or _get("n_audio_layer")
        if not hidden_size:
            hidden_size = _get("n_audio_state")
        if not intermediate_size and hidden_size:
            # Whisper: n_state * 4; also check decoder
            if _get("n_audio_state") or _get("encoder_layers"):
                intermediate_size = hidden_size * 4
        if not num_heads:
            num_heads = _get("encoder_attention_heads") or _get("n_audio_head")
        if not num_kv_heads:
            num_kv_heads = num_heads

        # MOE detection (check both levels)
        num_experts = (
            tc.get("num_local_experts") or tc.get("num_experts") or tc.get("n_routed_experts")
            or cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts")
            or 0
        )
        num_experts_per_tok = (
            (tc.get("num_experts_per_tok") or tc.get("top_k")
             or cfg.get("num_experts_per_tok") or cfg.get("top_k") or 0)
            if num_experts > 0 else 0
        )
        is_moe = num_experts > 0

        per_layer_sizes = tc.get("per_layer_intermediate_sizes") or cfg.get("per_layer_intermediate_sizes")

        layer_config: dict[str, Any] = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_act": tc.get("hidden_act") or tc.get("activation_function")
                          or cfg.get("hidden_act") or cfg.get("activation_function") or "silu",
        }
        if is_moe:
            layer_config["num_experts"] = num_experts
            layer_config["num_experts_per_tok"] = num_experts_per_tok

        # Detect layer prefix from tensor names
        layer_prefix = self._detect_layer_prefix()

        # Build children nodes
        children = []

        # Embeddings
        embed_prefix = self._detect_embed_prefix()
        if embed_prefix:
            children.append(self._make_prefix_node("Embeddings", embed_prefix))

        # Transformer layers
        if num_layers > 0 and layer_prefix:
            layers_group = self._make_layer_group(
                name="Transformer Layers",
                prefix=layer_prefix,
                num_layers=num_layers,
                layer_config=layer_config,
                per_layer_intermediate_sizes=per_layer_sizes,
            )
            children.append(layers_group)

        # Final norm
        norm_prefix = self._detect_norm_prefix()
        if norm_prefix:
            norm_tensors = [
                n for n in self.weight_index.tensors
                if n.startswith(norm_prefix) and "layers" not in n
            ]
            if norm_tensors:
                children.append(self._make_exact_node("Final Norm", norm_tensors, norm_prefix))

        # LM head
        lm_logical, lm_stored, lm_size = self._count_params_for_prefix("lm_head")
        tied = tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
        lm_head_node = ArchNode(
            name="LM Head" + (" (tied)" if tied and lm_logical == 0 else ""),
            node_type="module",
            weight_prefix="lm_head",
            param_count=lm_logical,
            stored_param_count=lm_stored,
            size_bytes=lm_size,
            extra={"tied": tied and lm_logical == 0},
        )
        children.append(lm_head_node)

        # Detect any remaining top-level tensor groups not yet covered
        covered_prefixes = {c.weight_prefix for c in children if c.weight_prefix}
        remaining = self._find_uncovered_prefixes(covered_prefixes, num_layers, layer_prefix)
        for name, prefix in remaining:
            children.append(self._make_prefix_node(name, prefix))

        # Architecture name
        arch_name = cfg.get("architectures", [self.model_type_name()])[0]

        root_config: dict[str, Any] = {
            "model_type": cfg.get("model_type"),
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "intermediate_size": intermediate_size,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "hidden_act": layer_config["hidden_act"],
            "max_position_embeddings": tc.get("max_position_embeddings") or cfg.get("max_position_embeddings"),
            "rope_theta": tc.get("rope_theta") or cfg.get("rope_theta"),
            "tie_word_embeddings": tied,
        }
        if is_moe:
            root_config["num_experts"] = num_experts
            root_config["num_experts_per_tok"] = num_experts_per_tok

        root = ArchNode(
            name=arch_name,
            node_type="model",
            weight_prefix="",
            config_params=root_config,
        )
        root.children = children

        quant = cfg.get("quantization") or cfg.get("quantization_config")
        model_type = cfg.get("model_type") or tc.get("model_type") or "generic"

        return ModelArchitecture(
            model_type=model_type,
            model_name=arch_name,
            model_dir=self.model_dir,
            root=root,
            config=cfg,
            quantization=quant,
            total_params=root.total_param_count,
            total_stored_params=root.total_stored_param_count,
            total_size_bytes=root.total_size_bytes,
        )

    # ------------------------------------------------------------------
    # Auto-detection helpers
    # ------------------------------------------------------------------

    def _detect_layer_prefix(self) -> str:
        """Detect the prefix for transformer layers from tensor names."""
        # Common patterns: model.layers, transformer.h, decoder.layers, etc.
        patterns = [
            r"^(model\.layers)\.\d+\.",
            r"^(transformer\.h)\.\d+\.",
            r"^(transformer\.layers)\.\d+\.",
            r"^(decoder\.layers)\.\d+\.",
            r"^(encoder\.layers)\.\d+\.",
            r"^(gpt_neox\.layers)\.\d+\.",
            r"^(transformer\.blocks)\.\d+\.",
            r"^(model\.decoder\.layers)\.\d+\.",
            r"^(language_model\.model\.layers)\.\d+\.",
            # Whisper: encoder.blocks.{i} / decoder.blocks.{i}
            r"^(encoder\.blocks)\.\d+\.",
            r"^(decoder\.blocks)\.\d+\.",
            # Parakeet / Conformer: encoder.layers.{i}
            r"^(encoder\.encoder\.layers)\.\d+\.",
        ]
        for pattern in patterns:
            for name in self.weight_index.tensors:
                m = re.match(pattern, name)
                if m:
                    return m.group(1)
        return "model.layers"

    def _detect_embed_prefix(self) -> str:
        """Detect the embedding tensor prefix."""
        candidates = [
            "model.embed_tokens",
            "transformer.wte",
            "gpt_neox.embed_in",
            "transformer.word_embeddings",
            "model.decoder.embed_tokens",
            "language_model.model.embed_tokens",
        ]
        for prefix in candidates:
            for name in self.weight_index.tensors:
                if name.startswith(prefix):
                    return prefix
        # Fallback: look for any embed tensor
        for name in self.weight_index.tensors:
            if "embed" in name.lower() and "layer" not in name:
                parts = name.rsplit(".", 1)
                if len(parts) > 1:
                    return parts[0]
                return name
        return ""

    def _detect_norm_prefix(self) -> str:
        """Detect the final layer norm prefix."""
        candidates = [
            "model.norm",
            "transformer.ln_f",
            "gpt_neox.final_layer_norm",
            "transformer.norm",
            "model.decoder.final_layer_norm",
            "language_model.model.norm",
        ]
        for prefix in candidates:
            for name in self.weight_index.tensors:
                if name.startswith(prefix):
                    return prefix
        return ""

    def _find_uncovered_prefixes(
        self,
        covered: set[str],
        num_layers: int,
        layer_prefix: str,
    ) -> list[tuple[str, str]]:
        """Find top-level tensor groups not covered by known components."""
        # Group tensors by their top-level prefix (depth 2)
        groups: dict[str, int] = defaultdict(int)
        for name in self.weight_index.tensors:
            parts = name.split(".")
            if len(parts) >= 2:
                prefix = ".".join(parts[:2])
            else:
                prefix = parts[0]

            # Skip if covered by any known prefix
            skip = False
            for cp in covered:
                if name.startswith(cp) or prefix.startswith(cp):
                    skip = True
                    break
            if skip:
                continue

            # Skip layer tensors
            if layer_prefix and name.startswith(layer_prefix + "."):
                continue

            # Skip lm_head (already handled)
            if name.startswith("lm_head"):
                continue

            groups[prefix] += 1

        # Return significant groups
        result = []
        for prefix, count in sorted(groups.items(), key=lambda x: -x[1]):
            if count > 0:
                # Create a human-readable name
                name = prefix.replace("model.", "").replace("_", " ").title()
                result.append((name, prefix))

        return result

    def _make_prefix_node(self, name: str, prefix: str) -> ArchNode:
        logical, stored, size = self._count_params_for_prefix(prefix)
        return ArchNode(
            name=name, node_type="module", weight_prefix=prefix,
            param_count=logical, stored_param_count=stored, size_bytes=size,
        )

    def _make_exact_node(self, name: str, tensor_names: list[str], prefix: str = "") -> ArchNode:
        logical, stored, size = self._count_exact_tensors(tensor_names)
        return ArchNode(
            name=name, node_type="module", weight_prefix=prefix,
            param_count=logical, stored_param_count=stored, size_bytes=size,
        )
