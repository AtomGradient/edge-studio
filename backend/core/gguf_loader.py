# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""GGUF model loader — parse metadata and tensor index from .gguf files.

Maps GGUF metadata keys to HuggingFace config.json format so the rest of
Edge Studio's analysis pipeline works transparently with GGUF models.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .weight_loader import TensorMeta, WeightIndex


# ---------------------------------------------------------------------------
# GGUF magic and constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" as bytes 47 47 55 46, read as uint32 LE

# GGUF value types
GGUF_TYPE_UINT8    = 0
GGUF_TYPE_INT8     = 1
GGUF_TYPE_UINT16   = 2
GGUF_TYPE_INT16    = 3
GGUF_TYPE_UINT32   = 4
GGUF_TYPE_INT32    = 5
GGUF_TYPE_FLOAT32  = 6
GGUF_TYPE_BOOL     = 7
GGUF_TYPE_STRING   = 8
GGUF_TYPE_ARRAY    = 9
GGUF_TYPE_UINT64   = 10
GGUF_TYPE_INT64    = 11
GGUF_TYPE_FLOAT64  = 12

# GGML tensor types (subset covering common quantization formats)
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M",
    30: "BF16",
}

# Approximate bytes per element for GGML quantized types
# For block-quantized types, this is (block_size_bytes / elements_per_block)
GGML_TYPE_SIZE = {
    "F32": 4.0, "F16": 2.0, "BF16": 2.0, "F64": 8.0,
    "I8": 1.0, "I16": 2.0, "I32": 4.0, "I64": 8.0,
    "Q4_0": 0.5625, "Q4_1": 0.625,  # 18/32, 20/32
    "Q5_0": 0.6875, "Q5_1": 0.75,   # 22/32, 24/32
    "Q8_0": 1.0625, "Q8_1": 1.125,  # 34/32, 36/32
    "Q2_K": 0.3125, "Q3_K": 0.4375, "Q4_K": 0.5625,
    "Q5_K": 0.6875, "Q6_K": 0.8125, "Q8_K": 1.0625,
    "IQ2_XXS": 0.25, "IQ2_XS": 0.3125, "IQ2_S": 0.3125,
    "IQ3_XXS": 0.375, "IQ3_S": 0.4375,
    "IQ4_NL": 0.5, "IQ4_XS": 0.5,
    "IQ1_S": 0.1875, "IQ1_M": 0.21875,
}

# GGUF metadata key -> config.json key mapping
GGUF_META_MAP = {
    "llama.block_count":          "num_hidden_layers",
    "llama.embedding_length":     "hidden_size",
    "llama.feed_forward_length":  "intermediate_size",
    "llama.attention.head_count":       "num_attention_heads",
    "llama.attention.head_count_kv":    "num_key_value_heads",
    "llama.context_length":       "max_position_embeddings",
    "llama.rope.freq_base":       "rope_theta",
    "llama.vocab_size":           "vocab_size",
    # Qwen
    "qwen2.block_count":          "num_hidden_layers",
    "qwen2.embedding_length":     "hidden_size",
    "qwen2.feed_forward_length":  "intermediate_size",
    "qwen2.attention.head_count":       "num_attention_heads",
    "qwen2.attention.head_count_kv":    "num_key_value_heads",
    "qwen2.context_length":       "max_position_embeddings",
    "qwen2.vocab_size":           "vocab_size",
    # Qwen3.5 (uses qwen35 arch prefix)
    "qwen35.block_count":         "num_hidden_layers",
    "qwen35.embedding_length":    "hidden_size",
    "qwen35.feed_forward_length": "intermediate_size",
    "qwen35.attention.head_count":      "num_attention_heads",
    "qwen35.attention.head_count_kv":   "num_key_value_heads",
    "qwen35.context_length":      "max_position_embeddings",
    "qwen35.vocab_size":          "vocab_size",
    # Gemma
    "gemma.block_count":          "num_hidden_layers",
    "gemma.embedding_length":     "hidden_size",
    "gemma.feed_forward_length":  "intermediate_size",
    "gemma.attention.head_count":       "num_attention_heads",
    "gemma.attention.head_count_kv":    "num_key_value_heads",
    "gemma.context_length":       "max_position_embeddings",
    "gemma.vocab_size":           "vocab_size",
}

# GGUF tensor name -> HF tensor name mapping patterns
GGUF_TENSOR_MAP = [
    ("token_embd.weight",                       "model.embed_tokens.weight"),
    ("output_norm.weight",                      "model.norm.weight"),
    ("output.weight",                           "lm_head.weight"),
    # Per-layer patterns (use {layer} placeholder)
    ("blk.{layer}.attn_norm.weight",            "model.layers.{layer}.input_layernorm.weight"),
    ("blk.{layer}.ffn_norm.weight",             "model.layers.{layer}.post_attention_layernorm.weight"),
    ("blk.{layer}.attn_q.weight",               "model.layers.{layer}.self_attn.q_proj.weight"),
    ("blk.{layer}.attn_k.weight",               "model.layers.{layer}.self_attn.k_proj.weight"),
    ("blk.{layer}.attn_v.weight",               "model.layers.{layer}.self_attn.v_proj.weight"),
    ("blk.{layer}.attn_output.weight",          "model.layers.{layer}.self_attn.o_proj.weight"),
    ("blk.{layer}.ffn_gate.weight",             "model.layers.{layer}.mlp.gate_proj.weight"),
    ("blk.{layer}.ffn_up.weight",               "model.layers.{layer}.mlp.up_proj.weight"),
    ("blk.{layer}.ffn_down.weight",             "model.layers.{layer}.mlp.down_proj.weight"),
    # Biases (some models)
    ("blk.{layer}.attn_q.bias",                 "model.layers.{layer}.self_attn.q_proj.bias"),
    ("blk.{layer}.attn_k.bias",                 "model.layers.{layer}.self_attn.k_proj.bias"),
    ("blk.{layer}.attn_v.bias",                 "model.layers.{layer}.self_attn.v_proj.bias"),
    ("blk.{layer}.attn_output.bias",            "model.layers.{layer}.self_attn.o_proj.bias"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_gguf_file(path: str) -> bool:
    """Check if path points to a single .gguf file."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() == ".gguf"


def is_gguf_directory(path: str) -> bool:
    """Check if a directory contains a .gguf file (but no safetensors)."""
    p = Path(path)
    if not p.is_dir():
        return False
    has_gguf = any(p.glob("*.gguf"))
    has_safetensors = any(p.glob("*.safetensors"))
    return has_gguf and not has_safetensors


def find_gguf_file(path: str) -> str | None:
    """Find the primary .gguf file given a path (file or directory)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return str(p)
    if p.is_dir():
        gguf_files = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
        if gguf_files:
            return str(gguf_files[0])
    return None


def load_gguf_metadata(gguf_path: str) -> dict[str, Any]:
    """Parse GGUF file and return config dict in HuggingFace format.

    The returned dict includes:
    - Standard HF config keys (num_hidden_layers, hidden_size, etc.)
    - _source_format: "gguf"
    - _gguf_file: absolute path to the .gguf file
    - _gguf_raw_metadata: original GGUF key-value pairs
    """
    raw_meta = _read_gguf_header_metadata(gguf_path)

    config: dict[str, Any] = {
        "_source_format": "gguf",
        "_gguf_file": str(Path(gguf_path).resolve()),
        "_gguf_raw_metadata": raw_meta,
    }

    # Map GGUF keys to HF config keys (explicit map)
    for gguf_key, hf_key in GGUF_META_MAP.items():
        if gguf_key in raw_meta:
            config[hf_key] = raw_meta[gguf_key]

    # Fallback: auto-detect arch prefix and map common suffixes
    arch_prefix = raw_meta.get("general.architecture", "")
    if arch_prefix:
        _suffix_map = {
            ".block_count": "num_hidden_layers",
            ".embedding_length": "hidden_size",
            ".feed_forward_length": "intermediate_size",
            ".attention.head_count": "num_attention_heads",
            ".attention.head_count_kv": "num_key_value_heads",
            ".context_length": "max_position_embeddings",
            ".vocab_size": "vocab_size",
        }
        for suffix, hf_key in _suffix_map.items():
            full_key = arch_prefix + suffix
            if full_key in raw_meta and hf_key not in config:
                config[hf_key] = raw_meta[full_key]

    # Detect model_type from general.architecture
    arch = raw_meta.get("general.architecture", "")
    name = raw_meta.get("general.name", "")
    config["model_type"] = _detect_model_type_from_gguf(arch, name)

    # Detect quantization from general.file_type or tensor types
    file_type = raw_meta.get("general.file_type", "")
    quant_desc = raw_meta.get("general.quantization_version", "")
    if file_type or quant_desc:
        config["_gguf_file_type"] = file_type
        config["_gguf_quant_version"] = quant_desc

    return config


def load_gguf_weight_index(gguf_path: str) -> WeightIndex:
    """Parse GGUF tensor info and return a WeightIndex compatible with the rest of Edge Studio."""
    tensors_raw = _read_gguf_tensor_info(gguf_path)

    gguf_path_str = str(Path(gguf_path).resolve())
    parent_dir = str(Path(gguf_path).parent)
    index = WeightIndex(model_dir=parent_dir)

    for raw in tensors_raw:
        gguf_name = raw["name"]
        hf_name = _map_tensor_name(gguf_name)
        dtype_str = GGML_TYPE_NAMES.get(raw["type"], f"GGML_{raw['type']}")
        shape = raw["dimensions"]
        n_elements = 1
        for d in shape:
            n_elements *= d

        # Calculate size from type
        bytes_per_elem = GGML_TYPE_SIZE.get(dtype_str, 1.0)
        estimated_size = int(n_elements * bytes_per_elem)

        index.tensors[hf_name] = TensorMeta(
            name=hf_name,
            dtype=dtype_str,
            shape=shape,
            offset_start=raw["offset"],
            offset_end=raw["offset"] + estimated_size,
            file_path=gguf_path_str,
        )

    return index


# ---------------------------------------------------------------------------
# Internal parsing
# ---------------------------------------------------------------------------

def _read_gguf_header_metadata(path: str) -> dict[str, Any]:
    """Read GGUF file header and extract all metadata key-value pairs."""
    metadata = {}
    with open(path, "rb") as f:
        # Magic
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#x}, expected {GGUF_MAGIC:#x})")

        # Version
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Counts
        n_tensors = struct.unpack("<Q" if version >= 3 else "<I", f.read(8 if version >= 3 else 4))[0]
        n_kv = struct.unpack("<Q" if version >= 3 else "<I", f.read(8 if version >= 3 else 4))[0]

        # Read KV pairs
        for _ in range(n_kv):
            key = _read_string(f, version)
            value_type = struct.unpack("<I", f.read(4))[0]
            value = _read_value(f, value_type, version)
            metadata[key] = value

    return metadata


def _read_gguf_tensor_info(path: str) -> list[dict]:
    """Read GGUF tensor info entries (name, dimensions, type, offset)."""
    tensors = []
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError("Not a GGUF file")

        version = struct.unpack("<I", f.read(4))[0]

        if version >= 3:
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
        else:
            n_tensors = struct.unpack("<I", f.read(4))[0]
            n_kv = struct.unpack("<I", f.read(4))[0]

        # Skip KV pairs
        for _ in range(n_kv):
            _read_string(f, version)
            value_type = struct.unpack("<I", f.read(4))[0]
            _read_value(f, value_type, version)

        # Read tensor info
        for _ in range(n_tensors):
            name = _read_string(f, version)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack("<Q" if version >= 3 else "<I",
                                         f.read(8 if version >= 3 else 4))[0])
            tensor_type = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]

            tensors.append({
                "name": name,
                "dimensions": dims,
                "type": tensor_type,
                "offset": offset,
            })

    return tensors


def _read_string(f, version: int) -> str:
    """Read a GGUF string (length-prefixed)."""
    length = struct.unpack("<Q" if version >= 3 else "<I",
                           f.read(8 if version >= 3 else 4))[0]
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f, value_type: int, version: int) -> Any:
    """Read a GGUF typed value."""
    if value_type == GGUF_TYPE_UINT8:
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == GGUF_TYPE_INT8:
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == GGUF_TYPE_UINT16:
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == GGUF_TYPE_INT16:
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == GGUF_TYPE_BOOL:
        return bool(struct.unpack("<B", f.read(1))[0])
    elif value_type == GGUF_TYPE_STRING:
        return _read_string(f, version)
    elif value_type == GGUF_TYPE_ARRAY:
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q" if version >= 3 else "<I",
                              f.read(8 if version >= 3 else 4))[0]
        return [_read_value(f, elem_type, version) for _ in range(count)]
    elif value_type == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == GGUF_TYPE_INT64:
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == GGUF_TYPE_FLOAT64:
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {value_type}")


def _detect_model_type_from_gguf(arch: str, name: str) -> str:
    """Map GGUF general.architecture to a model_type string."""
    arch_lower = arch.lower()
    name_lower = name.lower()
    # Qwen family (uses qwen2 or llama arch in GGUF)
    if "qwen" in arch_lower or "qwen" in name_lower:
        return "qwen3"
    if "llama" in arch_lower:
        return "llama"
    if "gemma" in arch_lower:
        return "gemma3"
    if "phi" in arch_lower:
        return "phi"
    if "mistral" in arch_lower:
        return "mistral"
    if "starcoder" in arch_lower:
        return "starcoder"
    return arch_lower or "unknown"


def _map_tensor_name(gguf_name: str) -> str:
    """Map a GGUF tensor name to HuggingFace format."""
    import re

    for gguf_pattern, hf_pattern in GGUF_TENSOR_MAP:
        if "{layer}" in gguf_pattern:
            # Extract layer number
            regex = gguf_pattern.replace("{layer}", r"(\d+)")
            regex = regex.replace(".", r"\.")
            m = re.match(regex, gguf_name)
            if m:
                layer = m.group(1)
                return hf_pattern.replace("{layer}", layer)
        else:
            if gguf_name == gguf_pattern:
                return hf_pattern

    # Fallback: return original name with gguf_ prefix to mark unmapped
    return gguf_name
