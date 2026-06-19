# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Config parser for Qwen3 TTS models."""

from __future__ import annotations

from typing import Any

from .architecture import ArchNode, ModelArchitecture
from .config_parser import ConfigParser, load_sub_config
from .weight_loader import WeightIndex


class Qwen3TTSConfigParser(ConfigParser):
    """Parse Qwen3 TTS model configuration.

    Handles the nested structure:
    - Root: Qwen3TTSForConditionalGeneration
      - Talker: 28-layer Transformer (SiLU, GQA 16Q/8KV)
        - CodePredictor: 5-layer sub-model
      - SpeechTokenizer: Encoder (8 layers) + Decoder (8 layers)
    """

    def model_type_name(self) -> str:
        return "Qwen3 TTS"

    def parse(self) -> ModelArchitecture:
        talker_cfg = self.config.get("talker_config", {})
        code_pred_cfg = talker_cfg.get("code_predictor_config", {})
        speech_tok_cfg = load_sub_config(self.model_dir, "speech_tokenizer")

        root = ArchNode(
            name=self.config.get("architectures", ["Qwen3TTS"])[0],
            node_type="model",
            weight_prefix="",
            config_params={
                "model_type": self.config.get("model_type"),
                "tts_model_size": self.config.get("tts_model_size"),
                "tts_model_type": self.config.get("tts_model_type"),
                "tokenizer_type": self.config.get("tokenizer_type"),
            },
        )

        # ---- Talker ----
        talker_node = self._build_talker(talker_cfg, code_pred_cfg)
        root.children.append(talker_node)

        # ---- Speech Tokenizer ----
        if speech_tok_cfg:
            st_node = self._build_speech_tokenizer(speech_tok_cfg)
            root.children.append(st_node)

        # Compute totals
        quant = self.config.get("quantization") or self.config.get("quantization_config")

        arch = ModelArchitecture(
            model_type="qwen3_tts",
            model_name=self.config.get("architectures", ["Qwen3TTS"])[0],
            model_dir=self.model_dir,
            root=root,
            config=self.config,
            quantization=quant,
            total_params=root.total_param_count,
            total_stored_params=root.total_stored_param_count,
            total_size_bytes=root.total_size_bytes,
        )
        return arch

    def _build_talker(self, talker_cfg: dict, code_pred_cfg: dict) -> ArchNode:
        num_layers = talker_cfg.get("num_hidden_layers", 28)
        hidden_size = talker_cfg.get("hidden_size", 1024)
        intermediate_size = talker_cfg.get("intermediate_size", 3072)
        num_heads = talker_cfg.get("num_attention_heads", 16)
        num_kv_heads = talker_cfg.get("num_key_value_heads", 8)
        head_dim = talker_cfg.get("head_dim", 128)

        per_layer_sizes = talker_cfg.get("per_layer_intermediate_sizes")

        layer_config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_act": talker_cfg.get("hidden_act", "silu"),
        }

        # Talker layers
        layers_group = self._make_layer_group(
            name="Transformer Layers",
            prefix="talker.model.layers",
            num_layers=num_layers,
            layer_config=layer_config,
            per_layer_intermediate_sizes=per_layer_sizes,
        )

        # Talker embeddings and norms
        embed_node = self._make_misc_node(
            "Embeddings",
            ["talker.model.embed_tokens.weight", "talker.audio_embed.weight",
             "talker.text_embed.weight", "talker.text_embed_norm.weight"],
            "talker.model.embed",
        )

        norm_node = self._make_misc_node(
            "Final Norm",
            ["talker.model.norm.weight"],
            "talker.model.norm",
        )

        lm_head_node = self._make_prefix_node("LM Head", "talker.lm_head")

        # Code predictor
        cp_node = self._build_code_predictor(code_pred_cfg)

        talker_node = ArchNode(
            name="Talker",
            node_type="submodel",
            weight_prefix="talker",
            config_params={
                "model_type": talker_cfg.get("model_type", "qwen3_tts_talker"),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "intermediate_size": intermediate_size,
                "head_dim": head_dim,
                "hidden_act": talker_cfg.get("hidden_act", "silu"),
                "vocab_size": talker_cfg.get("vocab_size", 3072),
                "text_vocab_size": talker_cfg.get("text_vocab_size", 151936),
                "text_hidden_size": talker_cfg.get("text_hidden_size", 2048),
                "max_position_embeddings": talker_cfg.get("max_position_embeddings", 32768),
                "rope_theta": talker_cfg.get("rope_theta", 1000000),
            },
        )
        talker_node.children = [embed_node, layers_group, norm_node, lm_head_node, cp_node]

        return talker_node

    def _build_code_predictor(self, cp_cfg: dict) -> ArchNode:
        num_layers = cp_cfg.get("num_hidden_layers", 5)
        hidden_size = cp_cfg.get("hidden_size", 1024)
        intermediate_size = cp_cfg.get("intermediate_size", 3072)
        num_heads = cp_cfg.get("num_attention_heads", 16)
        num_kv_heads = cp_cfg.get("num_key_value_heads", 8)

        layer_config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "hidden_act": cp_cfg.get("hidden_act", "silu"),
        }

        layers_group = self._make_layer_group(
            name="Transformer Layers",
            prefix="talker.code_predictor.model.layers",
            num_layers=num_layers,
            layer_config=layer_config,
        )

        embed_node = self._make_prefix_node("Embeddings", "talker.code_predictor.model.embed_tokens")
        norm_node = self._make_misc_node(
            "Final Norm",
            ["talker.code_predictor.model.norm.weight"],
            "talker.code_predictor.model.norm",
        )
        lm_head_node = self._make_prefix_node("LM Heads", "talker.code_predictor.lm_head")

        cp_node = ArchNode(
            name="Code Predictor",
            node_type="submodel",
            weight_prefix="talker.code_predictor",
            config_params={
                "model_type": cp_cfg.get("model_type", "qwen3_tts_talker_code_predictor"),
                "hidden_size": hidden_size,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "intermediate_size": intermediate_size,
                "vocab_size": cp_cfg.get("vocab_size", 2048),
                "num_code_groups": cp_cfg.get("num_code_groups", 16),
            },
        )
        cp_node.children = [embed_node, layers_group, norm_node, lm_head_node]

        return cp_node

    def _build_speech_tokenizer(self, st_cfg: dict) -> ArchNode:
        enc_cfg = st_cfg.get("encoder_config", {})
        dec_cfg = st_cfg.get("decoder_config", {})

        # Encoder
        enc_layers = enc_cfg.get("num_hidden_layers", 8)
        enc_layer_config = {
            "hidden_size": enc_cfg.get("hidden_size", 512),
            "intermediate_size": enc_cfg.get("intermediate_size", 2048),
            "num_attention_heads": enc_cfg.get("num_attention_heads", 8),
            "num_key_value_heads": enc_cfg.get("num_key_value_heads", 8),
            "hidden_act": enc_cfg.get("hidden_act", "gelu"),
        }

        enc_transformer = self._make_layer_group(
            name="Transformer Layers",
            prefix="encoder.transformer.layers",
            num_layers=enc_layers,
            layer_config=enc_layer_config,
        )
        enc_other = self._make_prefix_node("Conv/VQ Modules", "encoder")
        # Exclude transformer layers from enc_other count
        enc_t_logical, enc_t_stored, enc_t_size = self._count_params_for_prefix("encoder.transformer.layers.")
        enc_all_logical, enc_all_stored, enc_all_size = self._count_params_for_prefix("encoder.")
        enc_other.param_count = enc_all_logical - enc_t_logical
        enc_other.stored_param_count = enc_all_stored - enc_t_stored
        enc_other.size_bytes = enc_all_size - enc_t_size

        encoder_node = ArchNode(
            name="Encoder",
            node_type="submodel",
            weight_prefix="encoder",
            config_params={
                "hidden_size": enc_cfg.get("hidden_size", 512),
                "num_hidden_layers": enc_layers,
                "codebook_size": enc_cfg.get("codebook_size", 2048),
                "num_quantizers": enc_cfg.get("num_quantizers", 32),
                "sampling_rate": enc_cfg.get("sampling_rate", 24000),
                "upsampling_ratios": enc_cfg.get("upsampling_ratios"),
            },
        )
        encoder_node.children = [enc_other, enc_transformer]

        # Decoder
        dec_layers = dec_cfg.get("num_hidden_layers", 8)
        dec_layer_config = {
            "hidden_size": dec_cfg.get("hidden_size", 512),
            "intermediate_size": dec_cfg.get("intermediate_size", 1024),
            "num_attention_heads": dec_cfg.get("num_attention_heads", 16),
            "num_key_value_heads": dec_cfg.get("num_key_value_heads", 16),
            "hidden_act": dec_cfg.get("hidden_act", "silu"),
            "decoder_dim": dec_cfg.get("decoder_dim", 1536),
        }

        dec_transformer = self._make_layer_group(
            name="Transformer Layers",
            prefix="decoder.transformer.layers",
            num_layers=dec_layers,
            layer_config=dec_layer_config,
        )
        dec_other = self._make_prefix_node("Conv/VQ Modules", "decoder")
        dec_t_logical, dec_t_stored, dec_t_size = self._count_params_for_prefix("decoder.transformer.layers.")
        dec_all_logical, dec_all_stored, dec_all_size = self._count_params_for_prefix("decoder.")
        dec_other.param_count = dec_all_logical - dec_t_logical
        dec_other.stored_param_count = dec_all_stored - dec_t_stored
        dec_other.size_bytes = dec_all_size - dec_t_size

        decoder_node = ArchNode(
            name="Decoder",
            node_type="submodel",
            weight_prefix="decoder",
            config_params={
                "hidden_size": dec_cfg.get("hidden_size", 512),
                "num_hidden_layers": dec_layers,
                "codebook_size": dec_cfg.get("codebook_size", 2048),
                "decoder_dim": dec_cfg.get("decoder_dim", 1536),
                "latent_dim": dec_cfg.get("latent_dim", 1024),
                "upsample_rates": dec_cfg.get("upsample_rates"),
            },
        )
        decoder_node.children = [dec_other, dec_transformer]

        st_node = ArchNode(
            name="Speech Tokenizer",
            node_type="submodel",
            weight_prefix="speech_tokenizer",
            config_params={
                "model_type": st_cfg.get("model_type"),
                "input_sample_rate": st_cfg.get("input_sample_rate", 24000),
                "output_sample_rate": st_cfg.get("output_sample_rate", 24000),
                "encoder_valid_num_quantizers": st_cfg.get("encoder_valid_num_quantizers", 16),
            },
        )
        st_node.children = [encoder_node, decoder_node]

        return st_node

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
