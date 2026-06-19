# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Activation profiler — run forward passes and record MLP neuron activations.

Implements manual forward pass (no nn.Module) to capture gated MLP activations.
Generic path auto-detects config layout, embed key, layer prefix, MLP structure,
and activation function. Special paths only for TTS (dual embedding) and Whisper
(encoder-only).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from .activation_loader import ActivationProfile, LayerActivation


# ---- Activation functions ----

def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _gelu_pytorch_tanh(x: mx.array) -> mx.array:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(c * (x + 0.044715 * x * x * x)))


# ---- Low-level ops ----

def _dequantize(weights: dict, prefix: str, group_size: int = 64, bits: int = 4) -> mx.array:
    w = weights[f"{prefix}.weight"]
    s = weights[f"{prefix}.scales"]
    b = weights[f"{prefix}.biases"]
    return mx.dequantize(w, s, b, group_size=group_size, bits=bits)


def _linear(x: mx.array, weights: dict, prefix: str,
            group_size: int = 64, bits: int = 4) -> mx.array:
    """Compute x @ W^T (+ bias if present). Handles both quantized and raw weights."""
    if f"{prefix}.scales" in weights:
        w = _dequantize(weights, prefix, group_size, bits)
    else:
        w = weights[f"{prefix}.weight"]
    out = x @ w.T
    bias_key = f"{prefix}.bias"
    if bias_key in weights:
        out = out + weights[bias_key]
    return out


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(variance + eps) * weight


def _get_weight(weights: dict, key: str, group_size: int = 64, bits: int = 4) -> mx.array:
    """Get a weight, dequantizing if quantized. Tries key.weight fallback."""
    if f"{key}.scales" in weights:
        return _dequantize(weights, key, group_size, bits)
    if key in weights:
        return weights[key]
    return weights[f"{key}.weight"]


# ---- MLP forward (shared) ----

def _mlp_forward(
    x: mx.array,
    weights: dict,
    prefix: str,
    act_fn: Callable,
    group_size: int = 64,
    bits: int = 4,
) -> tuple[mx.array, mx.array]:
    """Gated MLP forward. Returns (output, gated_activations)."""
    gate = _linear(x, weights, f"{prefix}.mlp.gate_proj", group_size, bits)
    up = _linear(x, weights, f"{prefix}.mlp.up_proj", group_size, bits)
    gated = act_fn(gate) * up
    out = _linear(gated, weights, f"{prefix}.mlp.down_proj", group_size, bits)
    return out, gated


# ---- Attention forward (simplified, no RoPE needed for activation profiling) ----

def _attention_forward(
    x: mx.array,
    weights: dict,
    prefix: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group_size: int = 64,
    bits: int = 4,
    has_qk_norm: bool = False,
) -> mx.array:
    B, L, _ = x.shape
    q = _linear(x, weights, f"{prefix}.self_attn.q_proj", group_size, bits)
    k = _linear(x, weights, f"{prefix}.self_attn.k_proj", group_size, bits)
    v = _linear(x, weights, f"{prefix}.self_attn.v_proj", group_size, bits)

    # Handle fused q_proj (Qwen3.5: q_proj outputs query + attention-output-gate)
    expected_q_size = num_heads * head_dim
    if q.shape[-1] == expected_q_size * 2:
        q = q[..., :expected_q_size]

    q = q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    if has_qk_norm:
        q_norm_w = weights.get(f"{prefix}.self_attn.q_norm.weight")
        k_norm_w = weights.get(f"{prefix}.self_attn.k_norm.weight")
        if q_norm_w is not None:
            q = _rms_norm(q, q_norm_w)
        if k_norm_w is not None:
            k = _rms_norm(k, k_norm_w)

    # GQA: repeat KV heads
    if num_kv_heads < num_heads:
        repeats = num_heads // num_kv_heads
        k = mx.repeat(k, repeats, axis=1)
        v = mx.repeat(v, repeats, axis=1)

    scale = head_dim ** -0.5
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale

    # Causal mask
    mask = mx.triu(mx.full((L, L), -1e9), k=1)
    scores = scores + mask

    attn = mx.softmax(scores, axis=-1)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
    out = _linear(out, weights, f"{prefix}.self_attn.o_proj", group_size, bits)
    return out


# ---- Model-specific full forward ----

def _qwen3_tts_forward(
    weights: dict,
    config: dict,
    num_runs: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ActivationProfile:
    """Profile Qwen3 TTS talker activations."""
    tc = config["talker_config"]
    hidden_size = tc["hidden_size"]
    intermediate_size = tc["intermediate_size"]
    num_layers = tc["num_hidden_layers"]
    num_heads = tc["num_attention_heads"]
    num_kv_heads = tc["num_key_value_heads"]
    head_dim = tc.get("head_dim", hidden_size // num_heads)
    group_size = config.get("quantization", {}).get("group_size", 64)
    bits = config.get("quantization", {}).get("bits", 4)

    # Handle different embedding key names across model versions
    # Quantized: talker.model.embed_tokens (.weight/.scales/.biases)
    # bf16: talker.model.text_embedding.weight
    for embed_key in ["talker.model.embed_tokens", "talker.model.text_embedding"]:
        try:
            text_embed = _get_weight(weights, embed_key, group_size, bits)
            break
        except KeyError:
            continue
    else:
        raise KeyError("Cannot find text embedding: tried talker.model.embed_tokens and talker.model.text_embedding")

    text_vocab_size = tc.get("text_vocab_size", text_embed.shape[0])
    text_hidden_size = tc.get("text_hidden_size", text_embed.shape[1])
    is_quantized = "quantization" in config

    global_max_acts = np.zeros((num_layers, intermediate_size), dtype=np.float32)
    global_mean_acts = np.zeros((num_layers, intermediate_size), dtype=np.float32)

    for run in range(num_runs):
        if progress_callback:
            progress_callback(run, num_runs, f"Run {run+1}/{num_runs}")

        seq_len = np.random.randint(20, 60)
        token_ids = mx.array(np.random.randint(100, min(50000, text_vocab_size), size=(1, seq_len)))
        text_embeds = text_embed[token_ids]  # [1, L, text_hidden]

        # Project text embeddings to hidden_size if dimensions differ
        if text_hidden_size != hidden_size:
            # Try different projection key patterns
            if f"talker.text_projection.linear_fc1.weight" in weights:
                text_embeds = _linear(text_embeds, weights, "talker.text_projection.linear_fc1", group_size, bits)
                text_embeds = _silu(text_embeds)
                text_embeds = _linear(text_embeds, weights, "talker.text_projection.linear_fc2", group_size, bits)
            else:
                text_embeds = _linear(text_embeds, weights, "talker.text_embed", group_size, bits)
                text_embeds = _silu(text_embeds)
                if "talker.text_embed_norm.weight" in weights:
                    text_embeds = _rms_norm(text_embeds, weights["talker.text_embed_norm.weight"])

        x = text_embeds

        for i in range(num_layers):
            prefix = f"talker.model.layers.{i}"
            norm_w = weights[f"{prefix}.input_layernorm.weight"]
            h = _rms_norm(x, norm_w)
            h = _attention_forward(h, weights, prefix, num_heads, num_kv_heads, head_dim, group_size, bits)
            x = x + h

            norm_w2 = weights[f"{prefix}.post_attention_layernorm.weight"]
            h = _rms_norm(x, norm_w2)
            mlp_out, gated_acts = _mlp_forward(h, weights, prefix, _silu, group_size, bits)
            x = x + mlp_out

            # Record activations
            acts_np = np.array(mx.max(mx.abs(gated_acts[0]), axis=0).astype(mx.float32))
            global_max_acts[i] = np.maximum(global_max_acts[i], acts_np)
            global_mean_acts[i] += acts_np / num_runs

            mx.eval(x)

    layers = []
    for i in range(num_layers):
        layers.append(LayerActivation(
            layer_idx=i,
            max_activations=global_max_acts[i],
            mean_activations=global_mean_acts[i],
        ))

    return ActivationProfile(
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        run_count=num_runs,
        layers=layers,
    )


# ---- Generic helpers ----

def _detect_layer_prefix_from_weights(weights: dict) -> tuple[str, int]:
    """Return (layer_prefix, num_layers) by scanning weight keys."""
    pattern = re.compile(r"^(.*\.layers)\.(\d+)\.")
    prefix_sets: dict[str, set[int]] = {}
    for key in weights:
        m = pattern.match(key)
        if m:
            pfx, idx = m.group(1), int(m.group(2))
            prefix_sets.setdefault(pfx, set()).add(idx)
    if not prefix_sets:
        return "model.layers", 0
    prefix = max(prefix_sets, key=lambda p: len(prefix_sets[p]))
    return prefix, max(prefix_sets[prefix]) + 1


def _find_embed_key(weights: dict) -> str:
    """Find embedding table key by trying known candidates."""
    candidates = [
        "model.embed_tokens",
        "model.language_model.embed_tokens",
        "language_model.model.embed_tokens",
        "talker.model.embed_tokens",
        "talker.model.text_embedding",
        "transformer.wte",
        "gpt_neox.embed_in",
        "transformer.word_embeddings",
        "embed_tokens",
    ]
    for cand in candidates:
        if f"{cand}.weight" in weights or f"{cand}.scales" in weights:
            return cand
    # Fuzzy: any key ending with embed_tokens.weight
    for key in weights:
        if key.endswith("embed_tokens.weight") or key.endswith("embed_in.weight"):
            return key[: -len(".weight")]
    return ""


def _get_act_fn(name: str) -> Callable:
    """Map hidden_act name string to activation function."""
    name = (name or "silu").lower()
    if name in ("gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh", "gelu_approx"):
        return _gelu_pytorch_tanh
    if name == "relu":
        return lambda x: mx.maximum(x, 0)
    return _silu  # default: silu / swish


def _try_mlp_forward(
    h: mx.array,
    weights: dict,
    layer_prefix: str,
    act_fn: Callable,
    group_size: int = 64,
    bits: int = 4,
) -> tuple[mx.array | None, mx.array | None]:
    """Try multiple MLP weight layouts. Returns (output, gated_acts) or (None, None)."""
    lp = f"{layer_prefix}.mlp"

    def _has(key: str) -> bool:
        return f"{key}.weight" in weights or f"{key}.scales" in weights

    # Standard gated: gate_proj / up_proj / down_proj (llama, qwen, mistral, …)
    if _has(f"{lp}.gate_proj"):
        return _mlp_forward(h, weights, layer_prefix, act_fn, group_size, bits)

    # InternLM2 / some Qwen variants: w1(gate) / w3(up) / w2(down)
    if _has(f"{lp}.w1"):
        gate = _linear(h, weights, f"{lp}.w1", group_size, bits)
        up   = _linear(h, weights, f"{lp}.w3", group_size, bits)
        gated = act_fn(gate) * up
        out = _linear(gated, weights, f"{lp}.w2", group_size, bits)
        return out, gated

    # Phi-style: fc1 / fc2 (non-gated; fc1 may be 2× wide for gated variant)
    if _has(f"{lp}.fc1"):
        h_mid = _linear(h, weights, f"{lp}.fc1", group_size, bits)
        gated = act_fn(h_mid)
        out = _linear(gated, weights, f"{lp}.fc2", group_size, bits)
        return out, gated

    # GPT-2 style at layer level: c_fc / c_proj
    if _has(f"{layer_prefix}.mlp.c_fc"):
        h_mid = _linear(h, weights, f"{layer_prefix}.mlp.c_fc", group_size, bits)
        gated = act_fn(h_mid)
        out = _linear(gated, weights, f"{layer_prefix}.mlp.c_proj", group_size, bits)
        return out, gated

    return None, None  # Unknown MLP layout


def _generic_forward(
    weights: dict,
    config: dict,
    num_runs: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ActivationProfile:
    """Universal activation profiler for any standard transformer model.

    Handles:
    - Flat configs (Llama, Qwen2/3, Mistral, DeepSeek, Cohere, OLMo, …)
    - Nested text_config (Gemma3, Qwen3.5, Llama4, Mistral3, pixtral, …)
    - Multiple MLP layouts: gate/up/down, w1/w2/w3, fc1/fc2
    - Graceful skip when attention keys are missing
    - Gemma-style pre/post_feedforward_layernorm
    """
    # Resolve nested config
    tc = config.get("text_config") or config.get("talker_config") or config

    def _get(key, fallback=0):
        return tc.get(key) or config.get(key) or fallback

    hidden_size      = _get("hidden_size")
    intermediate_size = _get("intermediate_size")
    num_layers       = _get("num_hidden_layers") or _get("num_layers")
    num_heads        = _get("num_attention_heads") or _get("n_head")
    num_kv_heads     = _get("num_key_value_heads") or _get("num_kv_heads") or num_heads
    head_dim         = _get("head_dim") or (hidden_size // num_heads if num_heads else 64)
    vocab_size       = _get("vocab_size") or 128256

    quant_cfg  = config.get("quantization", {})
    group_size = quant_cfg.get("group_size", 64)
    bits       = quant_cfg.get("bits", 4)

    act_name = tc.get("hidden_act") or tc.get("activation_function") or config.get("hidden_act", "silu")
    act_fn = _get_act_fn(act_name)

    # Auto-detect layer prefix and embed key
    layer_prefix, detected_n = _detect_layer_prefix_from_weights(weights)
    if detected_n > 0:
        num_layers = min(num_layers, detected_n) if num_layers > 0 else detected_n

    embed_key = _find_embed_key(weights)
    if not embed_key:
        raise ValueError("Cannot find embedding table in weights")
    if not intermediate_size or not num_layers:
        raise ValueError(
            f"Cannot determine model dimensions: "
            f"intermediate_size={intermediate_size}, num_layers={num_layers}"
        )

    has_qk_norm = f"{layer_prefix}.0.self_attn.q_norm.weight" in weights

    embed = _get_weight(weights, embed_key, group_size, bits)

    global_max_acts  = np.zeros((num_layers, intermediate_size), dtype=np.float32)
    global_mean_acts = np.zeros((num_layers, intermediate_size), dtype=np.float32)

    for run in range(num_runs):
        if progress_callback:
            progress_callback(run, num_runs, f"Run {run+1}/{num_runs}")

        seq_len   = np.random.randint(20, 60)
        token_ids = mx.array(np.random.randint(100, min(50000, vocab_size), size=(1, seq_len)))
        x = embed[token_ids]

        for i in range(num_layers):
            prefix = f"{layer_prefix}.{i}"

            # --- Attention sub-layer ---
            pre_attn_norm = None
            for nk in (f"{prefix}.input_layernorm.weight", f"{prefix}.ln_1.weight"):
                if nk in weights:
                    pre_attn_norm = weights[nk]
                    break

            if pre_attn_norm is not None:
                h = _rms_norm(x, pre_attn_norm)
                try:
                    h = _attention_forward(
                        h, weights, prefix,
                        num_heads, num_kv_heads, head_dim,
                        group_size, bits, has_qk_norm=has_qk_norm,
                    )
                    x = x + h
                except (KeyError, Exception):
                    pass  # Skip attention on missing keys (linear_attn, MoE-only, etc.)

            # --- MLP sub-layer ---
            # Gemma-style pre_feedforward_layernorm takes priority
            pre_ff = f"{prefix}.pre_feedforward_layernorm.weight"
            if pre_ff in weights:
                h = _rms_norm(x, weights[pre_ff])
            else:
                h = x
                for nk in (
                    f"{prefix}.post_attention_layernorm.weight",
                    f"{prefix}.ln_2.weight",
                ):
                    if nk in weights:
                        h = _rms_norm(x, weights[nk])
                        break

            mlp_out, gated_acts = _try_mlp_forward(h, weights, prefix, act_fn, group_size, bits)

            if mlp_out is not None:
                # Gemma-style post_feedforward_layernorm
                post_ff = f"{prefix}.post_feedforward_layernorm.weight"
                if post_ff in weights:
                    mlp_out = _rms_norm(mlp_out, weights[post_ff])

                x = x + mlp_out
                acts_np = np.array(mx.max(mx.abs(gated_acts[0]), axis=0).astype(mx.float32))
                global_max_acts[i]  = np.maximum(global_max_acts[i], acts_np)
                global_mean_acts[i] += acts_np / num_runs

            mx.eval(x)

    layers = [
        LayerActivation(
            layer_idx=i,
            max_activations=global_max_acts[i],
            mean_activations=global_mean_acts[i],
        )
        for i in range(num_layers)
    ]
    return ActivationProfile(
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        run_count=num_runs,
        layers=layers,
    )


# ---- Audio model forward (Whisper encoder) ----

def _whisper_forward(
    weights: dict,
    config: dict,
    num_runs: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ActivationProfile:
    """Profile Whisper encoder block MLP activations.

    Supports both HuggingFace config format (d_model, encoder_layers) and
    MLX-native format (n_audio_state, n_audio_layer).

    MLP structure per block:
        x = x + mlp2(gelu(mlp1(mlp_ln(x))))
    where mlp1: n_state → n_state*4, mlp2: n_state*4 → n_state

    Weight keys:
        encoder.blocks.{i}.mlp1.weight
        encoder.blocks.{i}.mlp2.weight
        encoder.blocks.{i}.mlp_ln.weight / bias
    """
    # Resolve config — HF format vs MLX-native format
    if "d_model" in config or "encoder_layers" in config:
        n_state = config.get("d_model", 1280)
        n_layers = config.get("encoder_layers", 32)
        n_heads = config.get("encoder_attention_heads", 20)
        n_ctx = config.get("max_source_positions", 1500)
        n_mels = config.get("num_mel_bins", 128)
    else:
        n_state = config.get("n_audio_state", 1280)
        n_layers = config.get("n_audio_layer", 32)
        n_heads = config.get("n_audio_head", 20)
        n_ctx = config.get("n_audio_ctx", 1500)
        n_mels = config.get("n_mels", 128)

    # Override from weights if possible
    detected_n = sum(1 for k in weights if k.startswith("encoder.blocks.") and k.endswith(".mlp1.weight"))
    if detected_n > 0:
        n_layers = detected_n

    n_mlp = n_state * 4  # Whisper always uses 4x multiplier

    global_max_acts  = np.zeros((n_layers, n_mlp), dtype=np.float32)
    global_mean_acts = np.zeros((n_layers, n_mlp), dtype=np.float32)

    # Check if quantized
    is_quant = any(k.endswith(".scales") for k in weights)
    group_size = config.get("quantization", {}).get("group_size", 64) if isinstance(config.get("quantization"), dict) else 64
    bits = config.get("quantization", {}).get("bits", 4) if isinstance(config.get("quantization"), dict) else 4

    for run in range(num_runs):
        if progress_callback:
            progress_callback(run, num_runs, f"Run {run+1}/{num_runs}")

        # Random mel-feature input to encoder: [1, seq_len, n_state]
        seq_len = np.random.randint(10, min(50, n_ctx))
        x = mx.array(np.random.randn(1, seq_len, n_state).astype(np.float32) * 0.1)

        # Apply conv projection if present (skip for profiling — just use random hidden states)
        for i in range(n_layers):
            bp = f"encoder.blocks.{i}"

            # --- Self-attention (skip errors gracefully) ---
            attn_norm_key = f"{bp}.attn_ln.weight"
            if attn_norm_key in weights:
                h = _rms_norm(x, weights[attn_norm_key])
                try:
                    h = _attention_forward(h, weights, bp, n_heads, n_heads, n_state // n_heads, group_size, bits)
                    x = x + h
                except (KeyError, Exception):
                    pass

            # --- MLP ---
            mlp_ln_w = weights.get(f"{bp}.mlp_ln.weight")
            if mlp_ln_w is None:
                continue

            # Layer norm (Whisper uses LayerNorm with bias)
            mlp_ln_b = weights.get(f"{bp}.mlp_ln.bias")
            mean = mx.mean(x, axis=-1, keepdims=True)
            var  = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
            h = (x - mean) * mx.rsqrt(var + 1e-5) * mlp_ln_w
            if mlp_ln_b is not None:
                h = h + mlp_ln_b

            # mlp1 → gelu → mlp2
            h_mid = _linear(h, weights, f"{bp}.mlp1", group_size, bits)
            gated  = _gelu_pytorch_tanh(h_mid)          # Whisper uses gelu
            out    = _linear(gated, weights, f"{bp}.mlp2", group_size, bits)
            x = x + out

            acts_np = np.array(mx.max(mx.abs(gated[0]), axis=0).astype(mx.float32))
            global_max_acts[i]  = np.maximum(global_max_acts[i], acts_np)
            global_mean_acts[i] += acts_np / num_runs

            mx.eval(x)

    layers = [
        LayerActivation(
            layer_idx=i,
            max_activations=global_max_acts[i],
            mean_activations=global_mean_acts[i],
        )
        for i in range(n_layers)
    ]
    return ActivationProfile(
        intermediate_size=n_mlp,
        num_layers=n_layers,
        run_count=num_runs,
        layers=layers,
    )


# ---- Public API ----

def profile_model(
    model_dir: str,
    config: dict,
    model_type: str,
    num_runs: int = 20,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ActivationProfile:
    """Run activation profiling on a model.

    Loads weights, runs forward passes, and returns an ActivationProfile.
    """
    # Load all weights
    model_path = Path(model_dir)
    weights = {}
    for sf in sorted(model_path.glob("**/*.safetensors")):
        weights.update(mx.load(str(sf)))
    # Also try .npz (some audio models)
    if not weights:
        for npz in sorted(model_path.glob("*.npz")):
            weights.update({k: mx.array(v) for k, v in np.load(str(npz)).items()})

    if model_type == "qwen3_tts":
        profile = _qwen3_tts_forward(weights, config, num_runs, progress_callback)
    elif model_type == "whisper":
        profile = _whisper_forward(weights, config, num_runs, progress_callback)
    else:
        # Generic: auto-detects config layout, embed key, layer prefix,
        # MLP structure, activation function, and fused q_proj.
        # Covers llama, qwen2/3/3.5, gemma3, mistral, deepseek, phi,
        # internlm2, olmo, falcon, cohere, gpt2, …
        profile = _generic_forward(weights, config, num_runs, progress_callback)

    profile.source_file = str(model_path / "activation_profile.json")
    return profile


def save_profile_json(profile: ActivationProfile, output_path: str):
    """Save activation profile to JSON (compatible with prune_neurons.py)."""
    data = {
        "intermediate_size": profile.intermediate_size,
        "num_layers": profile.num_layers,
        "run_count": profile.run_count,
        "layers": [
            {
                "layer": l.layer_idx,
                "max_activations": l.max_activations.tolist(),
                "mean_activations": l.mean_activations.tolist(),
            }
            for l in profile.layers
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f)
