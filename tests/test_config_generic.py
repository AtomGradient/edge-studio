# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for the generic model config parser."""

from __future__ import annotations

from backend.core.config_generic import GenericConfigParser
from backend.core.weight_loader import TensorMeta, WeightIndex


def _tensor(name: str, shape: list[int], dtype: str = "F16") -> TensorMeta:
    return TensorMeta(
        name=name,
        dtype=dtype,
        shape=shape,
        offset_start=0,
        offset_end=1,
        file_path="model.safetensors",
    )


def test_generic_parser_builds_transformer_tree_from_nested_text_config() -> None:
    weights = WeightIndex(
        model_dir="/fake/model",
        tensors={
            "model.embed_tokens.weight": _tensor("model.embed_tokens.weight", [16, 8]),
            "model.layers.0.self_attn.q_proj.weight": _tensor("model.layers.0.self_attn.q_proj.weight", [8, 8]),
            "model.layers.0.mlp.down_proj.weight": _tensor("model.layers.0.mlp.down_proj.weight", [8, 16]),
            "model.layers.1.self_attn.q_proj.weight": _tensor("model.layers.1.self_attn.q_proj.weight", [8, 8]),
            "model.layers.1.mlp.down_proj.weight": _tensor("model.layers.1.mlp.down_proj.weight", [8, 12]),
            "model.norm.weight": _tensor("model.norm.weight", [8]),
            "vision_tower.patch.weight": _tensor("vision_tower.patch.weight", [4, 4]),
        },
    )
    parser = GenericConfigParser(
        "/fake/model",
        {
            "model_type": "vlm_wrapper",
            "architectures": ["FakeVLM"],
            "text_config": {
                "num_hidden_layers": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "tie_word_embeddings": True,
            },
        },
        weights,
    )

    arch = parser.parse()
    flat = arch.root.flatten()

    assert arch.model_name == "FakeVLM"
    assert arch.model_type == "vlm_wrapper"
    assert arch.root.config_params["hidden_size"] == 8
    assert arch.root.find_by_prefix("model.layers") is not None
    assert [node.name for node in arch.root.children][:3] == [
        "Embeddings",
        "Transformer Layers",
        "Final Norm",
    ]
    layer_group = arch.root.find_by_prefix("model.layers")
    assert layer_group is not None
    assert len(layer_group.children) == 2
    assert layer_group.children[1].config_params["intermediate_size"] == 16
    assert any(node.weight_prefix == "vision_tower.patch" for node in flat)
    assert arch.total_params > 0


def test_generic_parser_audio_fallbacks_detect_encoder_blocks() -> None:
    weights = WeightIndex(
        model_dir="/fake/asr",
        tensors={
            "encoder.blocks.0.attn.query.weight": _tensor("encoder.blocks.0.attn.query.weight", [4, 4]),
            "encoder.blocks.1.attn.query.weight": _tensor("encoder.blocks.1.attn.query.weight", [4, 4]),
        },
    )
    parser = GenericConfigParser(
        "/fake/asr",
        {
            "model_type": "whisper",
            "encoder_layers": 2,
            "n_audio_state": 4,
            "n_audio_head": 2,
        },
        weights,
    )

    arch = parser.parse()
    layer_group = arch.root.find_by_prefix("encoder.blocks")

    assert layer_group is not None
    assert len(layer_group.children) == 2
    assert layer_group.children[0].config_params["hidden_size"] == 4
    assert layer_group.children[0].config_params["intermediate_size"] == 16
