# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Safetensors weight loader with header-only parsing and on-demand tensor loading.

Uses MLX native APIs for tensor loading on Apple Silicon.
"""

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np


# dtype string -> element size in bytes
DTYPE_SIZE = {
    "F32": 4, "F16": 2, "BF16": 2,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "U8": 1, "U16": 2, "U32": 4, "U64": 8,
    "BOOL": 1,
}


@dataclass
class TensorMeta:
    """Metadata for a single tensor extracted from safetensors header."""
    name: str
    dtype: str
    shape: list[int]
    offset_start: int
    offset_end: int
    file_path: str

    @property
    def num_elements(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def element_size(self) -> int:
        return DTYPE_SIZE.get(self.dtype, 0)

    @property
    def size_bytes(self) -> int:
        sz = self.num_elements * self.element_size
        # GGUF quantized types have element_size 0 — fall back to offset range
        if sz == 0 and self.offset_end > self.offset_start:
            return self.offset_end - self.offset_start
        return sz

    @property
    def is_quantized(self) -> bool:
        if self.name.endswith((".scales", ".biases")):
            return True
        # GGUF quantized dtypes start with Q or IQ
        if self.dtype.startswith(("Q", "IQ")):
            return True
        return False

    @property
    def is_quant_weight(self) -> bool:
        return self.dtype == "U32"


@dataclass
class WeightIndex:
    """Index of all tensors in a model directory."""
    model_dir: str
    tensors: dict[str, TensorMeta] = field(default_factory=dict)
    file_headers: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def total_size_bytes(self) -> int:
        return sum(t.size_bytes for t in self.tensors.values())

    def tensors_with_prefix(self, prefix: str) -> dict[str, TensorMeta]:
        return {k: v for k, v in self.tensors.items() if k.startswith(prefix)}

    def unique_prefixes(self, depth: int = 2) -> list[str]:
        prefixes = set()
        for name in self.tensors:
            parts = name.split(".")
            if len(parts) >= depth:
                prefixes.add(".".join(parts[:depth]))
        return sorted(prefixes)


def parse_safetensors_header(file_path: str) -> tuple[dict, int]:
    """Parse safetensors file header without loading weights.

    Returns (header_dict, header_size) where header_dict maps
    tensor_name -> {"dtype": str, "shape": list, "data_offsets": [start, end]}.
    """
    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError(f"File too small to be safetensors: {file_path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]

        if header_len > 100 * 1024 * 1024:
            raise ValueError(f"Header length {header_len} seems too large: {file_path}")

        header_json = f.read(header_len)
        header = json.loads(header_json)

    return header, 8 + header_len


def _index_single_file(file_path: str, tensors: dict[str, TensorMeta],
                        file_headers: dict[str, dict]) -> dict:
    """Index tensors from a single safetensors file."""
    header, data_offset = parse_safetensors_header(file_path)
    metadata = header.pop("__metadata__", {})
    file_headers[file_path] = header

    for tensor_name, info in header.items():
        tensors[tensor_name] = TensorMeta(
            name=tensor_name,
            dtype=info["dtype"],
            shape=info["shape"],
            offset_start=data_offset + info["data_offsets"][0],
            offset_end=data_offset + info["data_offsets"][1],
            file_path=file_path,
        )

    return metadata


def load_weight_index(model_dir: str) -> WeightIndex:
    """Load weight index from a model directory.

    Supports:
    1. Single model.safetensors
    2. Sharded model-NNNNN-of-NNNNN.safetensors with index JSON
    3. Subdirectories (e.g., speech_tokenizer/model.safetensors)
    4. Split weight files (language_model.safetensors, etc.)
    """
    model_path = Path(model_dir)
    index = WeightIndex(model_dir=model_dir)
    indexed_files: set[str] = set()

    def _index_if_new(path: Path):
        p = str(path)
        if p not in indexed_files and path.exists():
            indexed_files.add(p)
            meta = _index_single_file(p, index.tensors, index.file_headers)
            if meta:
                index.metadata.update(meta)

    # Check for index.json (sharded model)
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        index.metadata = index_data.get("metadata", {})
        weight_map = index_data.get("weight_map", {})

        shard_files = set(weight_map.values())
        existing_shards = [s for s in shard_files if (model_path / s).exists()]

        if existing_shards:
            for shard_name in sorted(existing_shards):
                _index_if_new(model_path / shard_name)
        else:
            _index_if_new(model_path / "model.safetensors")
    else:
        _index_if_new(model_path / "model.safetensors")

    # Check subdirectories for additional safetensors
    for subdir in sorted(model_path.iterdir()):
        if subdir.is_dir() and subdir.name not in (".", "..", "__pycache__"):
            _index_if_new(subdir / "model.safetensors")

    # Split weight files (language_model.safetensors, vision_model.safetensors, etc.)
    for split_file in sorted(model_path.glob("*.safetensors")):
        if split_file.name.startswith("model"):
            continue
        _index_if_new(split_file)

    return index


def load_tensor(meta: TensorMeta) -> mx.array:
    """Load a single tensor using MLX (native safetensors support)."""
    weights = mx.load(meta.file_path)
    return weights[meta.name]


def load_dequantized_tensor(
    weight_meta: TensorMeta,
    scales_meta: TensorMeta,
    biases_meta: TensorMeta,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Load and dequantize a quantized tensor (weight + scales + biases)."""
    # Load all three from the same file if possible to avoid triple I/O
    if weight_meta.file_path == scales_meta.file_path == biases_meta.file_path:
        weights = mx.load(weight_meta.file_path)
        w = weights[weight_meta.name]
        s = weights[scales_meta.name]
        b = weights[biases_meta.name]
    else:
        w = load_tensor(weight_meta)
        s = load_tensor(scales_meta)
        b = load_tensor(biases_meta)
    return mx.dequantize(w, s, b, group_size=group_size, bits=bits)


def find_quant_group(name: str, index: WeightIndex) -> Optional[tuple[TensorMeta, TensorMeta, TensorMeta]]:
    """Find the (weight, scales, biases) triple for a quantized tensor."""
    for suffix in (".weight", ".scales", ".biases"):
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            break
    else:
        base = name

    w = index.tensors.get(f"{base}.weight")
    s = index.tensors.get(f"{base}.scales")
    b = index.tensors.get(f"{base}.biases")

    if w and s and b:
        return (w, s, b)
    return None


def is_quantized_weight(name: str, index: WeightIndex) -> bool:
    """Check if a tensor name belongs to a quantized weight group."""
    for suffix in (".weight", ".scales", ".biases"):
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            return (f"{base}.weight" in index.tensors
                    and f"{base}.scales" in index.tensors
                    and f"{base}.biases" in index.tensors)
    return False
