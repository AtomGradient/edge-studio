# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model-related schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LoadModelRequest(BaseModel):
    model_dir: str


class ArchNodeSchema(BaseModel):
    name: str
    node_type: str
    weight_prefix: str = ""
    config_params: dict[str, Any] = {}
    param_count: int = 0
    stored_param_count: int = 0
    size_bytes: int = 0
    children: list[ArchNodeSchema] = []
    pruning_info: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    total_param_count: int = 0
    total_stored_param_count: int = 0
    total_size_bytes: int = 0
    is_quantized: bool = False


class PruningTraceSchema(BaseModel):
    category: str
    description: str
    details: dict[str, Any] = {}
    severity: str = "info"


class QuantizationInfo(BaseModel):
    bits: int | None = None
    group_size: int | None = None
    mode: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    model_name: str
    model_dir: str
    total_params: int = 0
    total_stored_params: int = 0
    total_size_bytes: int = 0
    tensor_count: int = 0
    quantization: QuantizationInfo | None = None
    config: dict[str, Any] = {}
    has_moe: bool = False
    supports_thinking: bool = False
    has_vision: bool = False
    num_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_attention_heads: int = 0
    num_kv_heads: int = 0
    source_format: str = "safetensors"
    is_gguf: bool = False
    model_category: str = "llm"  # "llm" | "vlm" | "tts" | "stt"


class TensorMetaSchema(BaseModel):
    name: str
    dtype: str
    shape: list[int]
    num_elements: int
    size_bytes: int
    is_quantized: bool = False
    file_path: str = ""


class TensorStatsSchema(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    num_elements: int
    size_bytes: int
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    std_val: float | None = None
    sparsity: float | None = None
    histogram_counts: list[int] | None = None
    histogram_edges: list[float] | None = None
    is_quantized: bool = False
    quant_group_size: int | None = None
    quant_bits: int | None = None


class DtypeSummary(BaseModel):
    dtype: str
    count: int
    params: int
    size: int


class WeightStatsResponse(BaseModel):
    tensors: list[TensorMetaSchema]
    total_params: int
    total_size: int
    quantized_count: int


class DtypeBreakdownResponse(BaseModel):
    breakdown: list[DtypeSummary]
