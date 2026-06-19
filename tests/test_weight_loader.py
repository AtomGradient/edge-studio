# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for safetensors header parsing and weight indexing."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from backend.core import weight_loader


def _write_safetensors(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode("utf-8")
    max_end = max(
        (info.get("data_offsets", [0, 0])[1] for name, info in header.items() if name != "__metadata__"),
        default=0,
    )
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + (b"\0" * max_end))


def test_parse_safetensors_header_and_weight_index_include_nested_and_split_files(tmp_path: Path) -> None:
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "__metadata__": {"format": "mlx"},
            "model.embed_tokens.weight": {"dtype": "F16", "shape": [2, 4], "data_offsets": [0, 16]},
            "model.layers.0.mlp.weight": {"dtype": "U32", "shape": [2, 1], "data_offsets": [16, 24]},
            "model.layers.0.mlp.scales": {"dtype": "F16", "shape": [2, 2], "data_offsets": [24, 32]},
            "model.layers.0.mlp.biases": {"dtype": "F16", "shape": [2, 2], "data_offsets": [32, 40]},
        },
    )
    subdir = tmp_path / "speech_tokenizer"
    subdir.mkdir()
    _write_safetensors(
        subdir / "model.safetensors",
        {"speech.encoder.weight": {"dtype": "F32", "shape": [1, 3], "data_offsets": [0, 12]}},
    )
    _write_safetensors(
        tmp_path / "language_model.safetensors",
        {"language.extra.weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}},
    )

    header, data_offset = weight_loader.parse_safetensors_header(str(tmp_path / "model.safetensors"))
    index = weight_loader.load_weight_index(str(tmp_path))

    assert "__metadata__" in header
    assert data_offset > 8
    assert index.metadata == {"format": "mlx"}
    assert index.tensor_count == 6
    assert index.tensors["model.embed_tokens.weight"].size_bytes == 16
    assert index.tensors["language.extra.weight"].size_bytes == 4
    assert index.tensors_with_prefix("model.layers.0").keys() >= {
        "model.layers.0.mlp.weight",
        "model.layers.0.mlp.scales",
        "model.layers.0.mlp.biases",
    }
    assert "model.layers" in index.unique_prefixes(depth=2)

    quant_group = weight_loader.find_quant_group("model.layers.0.mlp.weight", index)
    assert quant_group is not None
    assert [item.name for item in quant_group] == [
        "model.layers.0.mlp.weight",
        "model.layers.0.mlp.scales",
        "model.layers.0.mlp.biases",
    ]
    assert weight_loader.is_quantized_weight("model.layers.0.mlp.scales", index) is True


def test_parse_safetensors_header_rejects_truncated_and_oversized_files(tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(b"short")
    huge_header = tmp_path / "huge.safetensors"
    huge_header.write_bytes(struct.pack("<Q", 101 * 1024 * 1024))

    with pytest.raises(ValueError, match="File too small"):
        weight_loader.parse_safetensors_header(str(truncated))
    with pytest.raises(ValueError, match="seems too large"):
        weight_loader.parse_safetensors_header(str(huge_header))
