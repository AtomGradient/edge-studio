# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Inference and trace-related schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class TraceRequest(BaseModel):
    prompt: str = "Hi, how are you?"
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    enable_thinking: bool = True
    enable_timing: bool = False
    capture_attention: bool = False
    capture_moe_routing: bool = False  # MoE expert routing (per-token inds + scores)
    use_legacy_tracer: bool = False  # Qwen3 expert tracer
    image_b64: str | None = None


class StepSchema(BaseModel):
    step: int
    token_id: int
    token_text: str
    prob: float
    rank: int
    top_logits: list[dict[str, Any]]
    hidden_norm: float | None = None


class LayerTraceSchema(BaseModel):
    layer_idx: int
    attn_residual_norm: float | None = None
    mlp_residual_norm: float | None = None
    attn_latency_ms: float | None = None
    mlp_latency_ms: float | None = None
    attn_weights: list[list[float]] | None = None  # [num_heads, seq_len]


class TraceResponse(BaseModel):
    generated_text: str
    num_tokens: int
    tokens_per_sec: float
    prefill_time_ms: float
    total_time_ms: float
    steps: list[StepSchema]
    layer_traces: list[list[LayerTraceSchema]]  # per-step, per-layer
    has_attention: bool = False
    has_timing: bool = False


# --- Quality Validator ---

class PPLRequest(BaseModel):
    text: str


class PPLResponse(BaseModel):
    perplexity: float
    num_tokens: int
    duration_ms: float
    token_log_probs: list[float] | None = None


class GenerateRequest(BaseModel):
    prompts: list[str]
    max_tokens: int = 512
    enable_thinking: bool = True


class GenerationSample(BaseModel):
    prompt: str
    generated_text: str
    avg_prob: float
    tokens_per_sec: float
    duration_ms: float


class QualityReportResponse(BaseModel):
    perplexity_results: list[PPLResponse] = []
    generation_samples: list[GenerationSample] = []
    avg_perplexity: float | None = None
    avg_tokens_per_sec: float | None = None


# --- Attention Analysis ---

class HeadClassification(BaseModel):
    layer: int
    head: int
    pattern: str  # SINK / LOCAL / GLOBAL / SPARSE
    confidence: float


class AttentionAnalysisResponse(BaseModel):
    classifications: list[HeadClassification]
    pattern_matrix: list[list[str]]  # [num_layers][num_heads]
    pattern_counts: dict[str, int]
    per_layer_summary: list[dict[str, Any]]
    suggestions: list[str]


# --- MOE Analysis ---

class ExpertLayerStats(BaseModel):
    layer_idx: int
    expert_counts: list[int]
    expert_avg_scores: list[float]
    load_balance: float


class MOEAnalysisResponse(BaseModel):
    num_experts: int
    top_k: int
    avg_load_balance: float
    layer_stats: list[ExpertLayerStats]
    cold_experts: list[dict[str, Any]]
    utilization_matrix: list[list[int]]  # [num_layers][num_experts]


# --- Model Comparison ---

class CompareRequest(BaseModel):
    model_id_a: str
    model_id_b: str
    prompt: str = "Hi, how are you?"
    max_tokens: int = 512
    enable_timing: bool = True
