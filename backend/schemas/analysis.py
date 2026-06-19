# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Analysis-related schemas (activation, pruning, KV cache, optimization)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# --- Activation Profile ---

class ProfileSummary(BaseModel):
    intermediate_size: int
    num_layers: int
    run_count: int
    total_dead_at_01: int
    dead_ratio_at_01: float


class ActivationHeatmapData(BaseModel):
    max_matrix: list[list[float]]
    mean_matrix: list[list[float]]
    num_layers: int
    neurons_per_layer: int
    dead_per_layer: list[int]
    threshold: float


class LoadProfileRequest(BaseModel):
    profile_path: str


class GenerateProfileRequest(BaseModel):
    num_runs: int = 5


# --- Pruning Simulation ---

class PruneSimRequest(BaseModel):
    threshold: float = 0.1
    max_reduction: float = 0.5
    min_intermediate: int = 128
    protected_layers: list[int] = []


class LayerPruneResult(BaseModel):
    layer_idx: int
    original_size: int
    alive_count: int
    aligned_size: int
    removed: int
    retention: float
    is_protected: bool


class PruneSimResponse(BaseModel):
    layers: list[LayerPruneResult]
    total_removed: int
    total_original: int
    retention: float
    mlp_size_saved_bytes: int
    mlp_params_saved: int
    config_preview: list[int]


# --- KV Cache ---

class DeviceCapacity(BaseModel):
    device_name: str
    ram_gb: float
    available_gb: float
    fits: bool
    max_context_length: int
    kv_at_max_context_mb: float


class KVReportResponse(BaseModel):
    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_token: int
    memory_curve: list[dict[str, Any]]
    device_capacities: list[DeviceCapacity]


class KVReportRequest(BaseModel):
    devices: list[str] = ["iPhone 17 Pro", "iPad Pro M5 (16GB)", "MacBook Air M5 (16GB)"]
    seq_lengths: list[int] | None = None


# --- Optimization Advisor ---

class SuggestionSchema(BaseModel):
    category: str
    priority: str
    title: str
    description: str
    estimated_saving_bytes: int
    risk: str
    params: dict[str, Any] = {}
    requires: list[str] = []


class OptSuggestionsResponse(BaseModel):
    suggestions: list[SuggestionSchema]
    requires_data: list[SuggestionSchema] = []


class ExecuteOptRequest(BaseModel):
    category: str
    params: dict[str, Any] = {}


# --- Auto Optimizer ---

class AutoOptSearchRequest(BaseModel):
    device_name: str
    quality_floor: float = 0.5
    target_bits: list[int] = [4]
    max_layers_remove: int = 0


class CandidateSchema(BaseModel):
    threshold: float
    bits: int
    layers_removed: int
    estimated_size_gb: float
    quality_proxy: float
    fits_device: bool
    is_pareto: bool


class AutoOptSearchResponse(BaseModel):
    candidates: list[CandidateSchema]
    pareto_frontier: list[CandidateSchema]
    total_candidates: int
    fits_device_count: int
    search_time_ms: float


# --- Device Profiles ---

class DeviceProfileSchema(BaseModel):
    name: str
    category: str
    ram_gb: float
    available_ram_gb: float
    neural_engine_tops: float
    gpu_cores: int
    chip: str
    max_model_size_gb: float


# --- Optimization Pipeline ---

class PipelineStepRequest(BaseModel):
    operation: str  # "neuron_pruning" | "layer_pruning" | "quantization" | etc.
    params: dict[str, Any] = {}


class PipelineRunRequest(BaseModel):
    steps: list[PipelineStepRequest]
    ppl_text: str = ""  # empty = use default text
    skip_validation: bool = False
