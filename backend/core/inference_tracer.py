# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Inference tracer — autoregressive generation with full trace capture.

Implements manual forward pass with RoPE, KV cache, and attention weight
capture for Qwen3 CausalLM models. Records per-step token probabilities,
attention patterns, MLP activations, and hidden state norms.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .activation_profiler import (
    _dequantize,
    _get_weight,
    _linear,
    _mlp_forward,
    _rms_norm,
    _silu,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerTrace:
    layer_idx: int
    attn_weights: np.ndarray          # [num_heads, seq_len]
    mlp_act_mean: float
    mlp_act_max: float
    mlp_act_top_indices: np.ndarray   # [16]
    mlp_act_top_values: np.ndarray    # [16]
    attn_residual_norm: float
    mlp_residual_norm: float
    norm_after_attn: float
    norm_after_mlp: float
    attn_latency_ms: float = 0.0     # per-layer attention latency (ms)
    mlp_latency_ms: float = 0.0      # per-layer MLP latency (ms)


@dataclass
class StepTrace:
    step_idx: int
    token_id: int
    token_str: str
    top_k_token_ids: np.ndarray       # [K]
    top_k_probs: np.ndarray           # [K]
    top_k_token_strs: list[str]
    chosen_rank: int
    chosen_prob: float
    layers: list[LayerTrace]
    final_hidden_norm: float


@dataclass
class InferenceTrace:
    prompt: str
    prompt_token_ids: list[int]
    prompt_tokens: list[str]
    temperature: float
    top_k: int
    top_p: float
    model_dir: str
    model_name: str
    num_layers: int
    num_heads: int
    hidden_size: int
    steps: list[StepTrace]
    generated_text: str
    total_time_seconds: float
    prefill_time_seconds: float
    prefill_layer_traces: list[LayerTrace] = field(default_factory=list)
    enable_timing: bool = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(model_dir: str):
    """Load tokenizer.json using the tokenizers library."""
    from tokenizers import Tokenizer
    tok_path = Path(model_dir) / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    return Tokenizer.from_file(str(tok_path))


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _rope_freqs(head_dim: int, theta: float = 1_000_000.0) -> mx.array:
    """Precompute RoPE inverse frequency table. Returns [head_dim/2]."""
    dim_pairs = head_dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, dim_pairs).astype(mx.float32) * 2.0 / head_dim))
    return freqs


def _apply_rope(x: mx.array, freqs: mx.array, offset: int) -> mx.array:
    """Apply RoPE to x: [B, num_heads, L, head_dim].

    positions = [offset, offset+1, ..., offset+L-1]
    """
    L = x.shape[2]
    positions = mx.arange(offset, offset + L).astype(mx.float32)  # [L]
    # angles: [L, head_dim/2]
    angles = positions[:, None] * freqs[None, :]
    cos_vals = mx.cos(angles)  # [L, head_dim/2]
    sin_vals = mx.sin(angles)  # [L, head_dim/2]

    # Split x into two halves along last dimension
    half = x.shape[-1] // 2
    x1 = x[..., :half]   # [B, H, L, half]
    x2 = x[..., half:]   # [B, H, L, half]

    # Broadcast: cos/sin are [L, half], need to be [1, 1, L, half]
    cos_vals = cos_vals[None, None, :, :]
    sin_vals = sin_vals[None, None, :, :]

    # Apply rotation
    out1 = x1 * cos_vals - x2 * sin_vals
    out2 = x2 * cos_vals + x1 * sin_vals
    return mx.concatenate([out1, out2], axis=-1)


# ---------------------------------------------------------------------------
# Attention with KV Cache
# ---------------------------------------------------------------------------

def _attention_with_cache(
    x: mx.array,
    weights: dict,
    prefix: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_freqs_table: mx.array,
    kv_cache: tuple[mx.array, mx.array] | None,
    group_size: int = 64,
    bits: int = 4,
) -> tuple[mx.array, mx.array, tuple[mx.array, mx.array]]:
    """Attention with RoPE + KV cache + attention weight capture.

    Returns (output, attn_weights_last_token, new_kv_cache).
    attn_weights_last_token: [num_heads, total_seq_len] — attention for last token.
    """
    B, L, _ = x.shape

    q = _linear(x, weights, f"{prefix}.self_attn.q_proj", group_size, bits)
    k = _linear(x, weights, f"{prefix}.self_attn.k_proj", group_size, bits)
    v = _linear(x, weights, f"{prefix}.self_attn.v_proj", group_size, bits)

    q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # QK norm (Qwen3 has this)
    q_norm_key = f"{prefix}.self_attn.q_norm.weight"
    k_norm_key = f"{prefix}.self_attn.k_norm.weight"
    if q_norm_key in weights:
        q = _rms_norm(q, weights[q_norm_key])
    if k_norm_key in weights:
        k = _rms_norm(k, weights[k_norm_key])

    # RoPE
    offset = 0 if kv_cache is None else kv_cache[0].shape[2]
    q = _apply_rope(q, rope_freqs_table, offset)
    k = _apply_rope(k, rope_freqs_table, offset)

    # KV cache update
    if kv_cache is not None:
        k = mx.concatenate([kv_cache[0], k], axis=2)
        v = mx.concatenate([kv_cache[1], v], axis=2)
    new_cache = (k, v)

    # GQA: repeat KV heads
    if num_kv_heads < num_heads:
        repeats = num_heads // num_kv_heads
        k_exp = mx.repeat(k, repeats, axis=1)
        v_exp = mx.repeat(v, repeats, axis=1)
    else:
        k_exp = k
        v_exp = v

    scale = head_dim ** -0.5
    # q: [B, num_heads, L, head_dim], k_exp: [B, num_heads, S, head_dim]
    scores = (q @ k_exp.transpose(0, 1, 3, 2)) * scale  # [B, H, L, S]

    # Causal mask
    S = k_exp.shape[2]
    if L > 1:
        # Prefill: full causal mask
        mask = mx.full((L, S), -1e9)
        for i in range(L):
            # position in full sequence: offset + i
            # can attend to positions 0..offset+i
            mask[i, :offset + i + 1] = 0.0
        scores = scores + mask[None, None, :, :]
    # For L==1 (decode), no mask needed — can attend to all previous positions

    attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)

    # Capture attention weights for last token only: [B, num_heads, S]
    attn_last = attn[:, :, -1, :]  # [B, num_heads, S]

    out = (attn @ v_exp).transpose(0, 2, 1, 3).reshape(B, L, -1)
    out = _linear(out, weights, f"{prefix}.self_attn.o_proj", group_size, bits)

    # Convert to numpy: [num_heads, S]
    attn_weights_np = np.array(attn_last[0].astype(mx.float32))

    return out, attn_weights_np, new_cache


# ---------------------------------------------------------------------------
# Token sampling
# ---------------------------------------------------------------------------

def _sample_token(
    logits: mx.array,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> tuple[int, mx.array]:
    """Sample a token from logits [vocab_size]. Returns (token_id, full_probs).

    Uses mlx_lm.sample_utils for battle-tested sampling.
    """
    from mlx_lm.sample_utils import make_sampler

    logprobs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    sampler = make_sampler(temp=temperature, top_k=top_k, top_p=top_p)
    token = sampler(logprobs)
    mx.eval(token)
    token_id = int(token.item())

    # Full probability distribution (unfiltered) for trace recording
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    return token_id, probs


# ---------------------------------------------------------------------------
# Single step forward
# ---------------------------------------------------------------------------

def _forward_one_step(
    token_ids: mx.array,
    weights: dict,
    config: dict,
    rope_freqs_table: mx.array,
    kv_caches: list[tuple[mx.array, mx.array] | None],
    embed_weight: mx.array,
    group_size: int,
    bits: int,
    enable_timing: bool = False,
) -> tuple[mx.array, list[LayerTrace], float, list[tuple[mx.array, mx.array]]]:
    """Forward pass for one step (prefill or single-token decode).

    Args:
        token_ids: [1, L] token indices
        weights, config: model weights and config
        rope_freqs_table: precomputed RoPE freqs
        kv_caches: list of (K, V) per layer, or None for first step
        embed_weight: embedding weight (dequantized)
        group_size, bits: quantization params

    Returns:
        (logits, layer_traces, final_hidden_norm, new_kv_caches)
        logits: [vocab_size] for last token
    """
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", 128)
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)

    # Embedding lookup
    x = embed_weight[token_ids]  # [1, L, hidden_size]

    layer_traces = []
    new_kv_caches = []

    for i in range(num_layers):
        prefix = f"model.layers.{i}"

        # Sync before timing if needed
        if enable_timing:
            mx.eval(x)
            t0 = time.perf_counter()

        # Pre-attention norm
        norm_w = weights[f"{prefix}.input_layernorm.weight"]
        h = _rms_norm(x, norm_w, rms_norm_eps)

        # Attention with cache
        attn_out, attn_weights_np, new_cache = _attention_with_cache(
            h, weights, prefix, num_heads, num_kv_heads, head_dim,
            rope_freqs_table, kv_caches[i], group_size, bits,
        )
        new_kv_caches.append(new_cache)

        # Attention residual
        attn_residual_norm = float(mx.sqrt(mx.sum(attn_out[0, -1] ** 2)).item())
        x = x + attn_out
        norm_after_attn = float(mx.sqrt(mx.sum(x[0, -1] ** 2)).item())

        if enable_timing:
            mx.eval(x)
            t_attn = time.perf_counter()

        # Post-attention norm
        norm_w2 = weights[f"{prefix}.post_attention_layernorm.weight"]
        h = _rms_norm(x, norm_w2, rms_norm_eps)

        # MLP
        mlp_out, gated_acts = _mlp_forward(h, weights, prefix, _silu, group_size, bits)

        # MLP residual
        mlp_residual_norm = float(mx.sqrt(mx.sum(mlp_out[0, -1] ** 2)).item())
        x = x + mlp_out
        norm_after_mlp = float(mx.sqrt(mx.sum(x[0, -1] ** 2)).item())

        if enable_timing:
            mx.eval(x)
            t_mlp = time.perf_counter()
            attn_ms = (t_attn - t0) * 1000
            mlp_ms = (t_mlp - t_attn) * 1000
        else:
            attn_ms = 0.0
            mlp_ms = 0.0

        # MLP activation stats (last token only)
        last_acts = gated_acts[0, -1]  # [intermediate_size]
        abs_acts = mx.abs(last_acts)
        act_mean = float(mx.mean(abs_acts).item())
        act_max = float(mx.max(abs_acts).item())

        # Top-16 activations
        k_top = min(16, abs_acts.shape[0])
        top_indices = mx.argpartition(abs_acts, kth=abs_acts.shape[0] - k_top)[-k_top:]
        top_values = last_acts[top_indices]
        # Sort by absolute value descending
        sort_order = mx.argsort(mx.abs(top_values))[::-1]
        top_indices = top_indices[sort_order]
        top_values = top_values[sort_order]

        layer_traces.append(LayerTrace(
            layer_idx=i,
            attn_weights=attn_weights_np,
            mlp_act_mean=act_mean,
            mlp_act_max=act_max,
            mlp_act_top_indices=np.array(top_indices),
            mlp_act_top_values=np.array(top_values.astype(mx.float32)),
            attn_residual_norm=attn_residual_norm,
            mlp_residual_norm=mlp_residual_norm,
            norm_after_attn=norm_after_attn,
            norm_after_mlp=norm_after_mlp,
            attn_latency_ms=attn_ms,
            mlp_latency_ms=mlp_ms,
        ))

        if not enable_timing:
            mx.eval(x)

    # Final norm
    final_norm_w = weights["model.norm.weight"]
    x = _rms_norm(x, final_norm_w, rms_norm_eps)
    final_hidden_norm = float(mx.sqrt(mx.sum(x[0, -1] ** 2)).item())

    # LM head
    tie_embeddings = config.get("tie_word_embeddings", True)
    if tie_embeddings or "lm_head.weight" not in weights:
        lm_weight = embed_weight
    else:
        lm_weight = _get_weight(weights, "lm_head", group_size, bits)

    # logits = x[-1] @ lm_weight^T
    logits = (x[0, -1:, :] @ lm_weight.T)[0]  # [vocab_size]

    return logits, layer_traces, final_hidden_norm, new_kv_caches


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def _build_chat_prompt(prompt: str, enable_thinking: bool = True) -> str:
    """Build ChatML prompt for Qwen3.

    Qwen3 defaults to thinking mode. To disable, append /no_think to user message,
    and prefill assistant with empty think block.
    """
    if enable_thinking:
        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def run_inference_trace(
    model_dir: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    trace_top_k: int = 20,
    enable_thinking: bool = True,
    enable_timing: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> InferenceTrace:
    """Run autoregressive generation and capture full trace.

    Always uses ChatML template (Qwen3 default). Thinking mode can be toggled.

    Args:
        model_dir: path to Qwen3 model directory
        prompt: input text (user message content)
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k for sampling
        top_p: top-p (nucleus) for sampling
        trace_top_k: number of top tokens to record per step
        enable_thinking: enable Qwen3 thinking mode (default True)
        enable_timing: enable per-layer latency profiling (adds mx.eval sync overhead)
        progress_callback: (step, total, message) progress function

    Returns:
        InferenceTrace with all recorded data.
    """
    t_start = time.time()

    # Load config
    config_path = Path(model_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    hidden_size = config["hidden_size"]
    head_dim = config.get("head_dim", 128)
    rope_theta = config.get("rope_theta", 1_000_000.0)
    quant_cfg = config.get("quantization") or config.get("quantization_config") or {}
    group_size = quant_cfg.get("group_size", 64)
    bits = quant_cfg.get("bits", 4)

    # EOS token(s) — Qwen3 can have multiple
    raw_eos = config.get("eos_token_id", 151645)
    if isinstance(raw_eos, list):
        eos_token_ids = set(raw_eos)
    else:
        eos_token_ids = {raw_eos}

    # Load tokenizer
    tokenizer = load_tokenizer(model_dir)

    # Always use ChatML template
    full_prompt = _build_chat_prompt(prompt, enable_thinking=enable_thinking)
    encoding = tokenizer.encode(full_prompt)
    prompt_ids = encoding.ids
    prompt_tokens_str = [tokenizer.decode([tid]) for tid in prompt_ids]

    # Load weights
    model_path = Path(model_dir)
    weights = {}
    for sf in sorted(model_path.glob("*.safetensors")):
        weights.update(mx.load(str(sf)))

    # Dequantize embedding for lookup
    embed_weight = _get_weight(weights, "model.embed_tokens", group_size, bits)
    mx.eval(embed_weight)

    # RoPE frequencies
    rope_freqs_table = _rope_freqs(head_dim, rope_theta)

    # Initialize KV caches
    kv_caches: list[tuple[mx.array, mx.array] | None] = [None] * num_layers

    # --- Prefill ---
    if progress_callback:
        progress_callback(0, max_tokens, "Prefilling...")

    t_prefill_start = time.time()
    input_ids = mx.array([prompt_ids])  # [1, prompt_len]

    logits, prefill_layer_traces, _, kv_caches = _forward_one_step(
        input_ids, weights, config, rope_freqs_table, kv_caches,
        embed_weight, group_size, bits, enable_timing=enable_timing,
    )
    mx.eval(logits)
    for cache in kv_caches:
        if cache is not None:
            mx.eval(cache[0], cache[1])
    t_prefill = time.time() - t_prefill_start

    # --- Decode loop ---
    steps: list[StepTrace] = []
    generated_ids: list[int] = []

    for step in range(max_tokens):
        if progress_callback:
            progress_callback(step, max_tokens, f"Generating token {step+1}/{max_tokens}")

        # Sample
        token_id, probs = _sample_token(logits, temperature, top_k, top_p)
        mx.eval(probs)

        # Top-K trace
        k_trace = min(trace_top_k, probs.shape[0])
        top_k_indices = mx.argpartition(probs, kth=probs.shape[0] - k_trace)[-k_trace:]
        top_k_probs_vals = probs[top_k_indices]
        # Sort descending
        sort_order = mx.argsort(top_k_probs_vals)[::-1]
        top_k_indices = top_k_indices[sort_order]
        top_k_probs_vals = top_k_probs_vals[sort_order]

        top_k_ids_np = np.array(top_k_indices)
        top_k_probs_np = np.array(top_k_probs_vals.astype(mx.float32))
        top_k_strs = [tokenizer.decode([int(tid)]) for tid in top_k_ids_np]

        chosen_prob = float(probs[token_id].item())
        # Find rank
        chosen_rank = 0
        for ri, tid in enumerate(top_k_ids_np):
            if int(tid) == token_id:
                chosen_rank = ri
                break
        else:
            chosen_rank = k_trace  # not in top-K

        token_str = tokenizer.decode([token_id])
        generated_ids.append(token_id)

        # Check EOS
        if token_id in eos_token_ids:
            steps.append(StepTrace(
                step_idx=step,
                token_id=token_id,
                token_str=token_str,
                top_k_token_ids=top_k_ids_np,
                top_k_probs=top_k_probs_np,
                top_k_token_strs=top_k_strs,
                chosen_rank=chosen_rank,
                chosen_prob=chosen_prob,
                layers=[],
                final_hidden_norm=0.0,
            ))
            break

        # Forward next token and capture trace
        next_input = mx.array([[token_id]])
        logits, layer_traces, final_hidden_norm, kv_caches = _forward_one_step(
            next_input, weights, config, rope_freqs_table, kv_caches,
            embed_weight, group_size, bits, enable_timing=enable_timing,
        )
        mx.eval(logits)
        for cache in kv_caches:
            if cache is not None:
                mx.eval(cache[0], cache[1])

        steps.append(StepTrace(
            step_idx=step,
            token_id=token_id,
            token_str=token_str,
            top_k_token_ids=top_k_ids_np,
            top_k_probs=top_k_probs_np,
            top_k_token_strs=top_k_strs,
            chosen_rank=chosen_rank,
            chosen_prob=chosen_prob,
            layers=layer_traces,
            final_hidden_norm=final_hidden_norm,
        ))

    t_total = time.time() - t_start
    generated_text = tokenizer.decode(generated_ids)
    model_name = config.get("architectures", ["Qwen3ForCausalLM"])[0]

    return InferenceTrace(
        prompt=prompt,
        prompt_token_ids=prompt_ids,
        prompt_tokens=prompt_tokens_str,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        model_dir=model_dir,
        model_name=model_name,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        steps=steps,
        generated_text=generated_text,
        total_time_seconds=t_total,
        prefill_time_seconds=t_prefill,
        prefill_layer_traces=prefill_layer_traces,
        enable_timing=enable_timing,
    )
