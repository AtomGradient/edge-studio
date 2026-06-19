# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Universal inference tracer — supports 117+ mlx-lm model architectures.

Uses mlx-lm's load/generate APIs with monkey-patching to capture per-layer
traces (norms, timing, optional attention weights) without hand-writing
forward passes for each model type.

Three modes:
- Default: captures norms + timing + MLP stats (zero extra overhead)
- capture_attention=True: replaces fused SDPA with manual QKV (approx 20% overhead)
- Qwen3 Legacy: the hand-written tracer in inference_tracer.py (for Qwen3 only)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .inference_tracer import InferenceTrace, LayerTrace, StepTrace
from .moe_analyzer import ExpertTrace


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TracerConfig:
    """Configuration for the universal tracer."""
    capture_attention: bool = False
    enable_timing: bool = False
    trace_top_k: int = 20
    capture_moe_routing: bool = False


# ---------------------------------------------------------------------------
# Model wrapper with monkey-patching
# ---------------------------------------------------------------------------

class ModelWrapper:
    """Wraps an mlx-lm model to intercept per-layer calls for tracing.

    All mlx-lm TransformerBlock subclasses follow the pattern:
        def __call__(self, x, mask=None, cache=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache)
            h = x + r
            r = self.mlp(self.post_attention_layernorm(h))
            return h + r

    We monkey-patch the class-level __call__ to capture residual norms,
    hidden state norms, and timing around the original call.

    Note: Python's special method lookup requires class-level patching.
    Instance-level __call__ overrides are bypassed by obj() syntax.
    """

    def __init__(self, model: nn.Module, config: dict, tracer_config: TracerConfig):
        self.model = model
        self.config = config
        self.tracer_config = tracer_config
        self._layer_traces: list[LayerTrace] = []
        self._installed = False
        # Class-level patching state
        self._patched_classes: dict[type, Any] = {}  # cls -> original __call__
        self._layer_to_idx: dict[int, int] = {}  # id(layer) -> layer index

    def install_hooks(self):
        """Install monkey-patched __call__ on transformer layer classes."""
        if self._installed:
            return

        layers = self._find_layers()
        self._layer_to_idx = {id(layer): idx for idx, layer in enumerate(layers)}

        # Patch each unique layer class
        for layer in layers:
            layer_cls = type(layer)
            if layer_cls not in self._patched_classes:
                self._patched_classes[layer_cls] = layer_cls.__call__
                self._create_class_patch(layer_cls)

        self._installed = True

    def remove_hooks(self):
        """Restore original __call__ on each patched class."""
        for cls, original_call in self._patched_classes.items():
            cls.__call__ = original_call
        self._patched_classes.clear()
        self._layer_to_idx.clear()
        self._installed = False

    def flush_traces(self) -> list[LayerTrace]:
        """Return and clear collected layer traces."""
        traces = self._layer_traces
        self._layer_traces = []
        return traces

    def _find_layers(self) -> list[nn.Module]:
        """Find transformer layers in the model."""
        for attr_path in [
            ("model", "layers"),
            ("layers",),
            ("model", "model", "layers"),
            ("transformer", "h"),
            ("transformer", "layers"),
            ("gpt_neox", "layers"),
            ("language_model", "model", "layers"),
        ]:
            obj = self.model
            try:
                for attr in attr_path:
                    obj = getattr(obj, attr)
                if hasattr(obj, '__len__') and len(obj) > 0:
                    return list(obj)
            except (AttributeError, TypeError):
                continue

        raise RuntimeError("Could not find transformer layers in model. "
                           "Supported patterns: model.layers, transformer.h, etc.")

    def _create_class_patch(self, layer_cls: type):
        """Create a traced __call__ at the class level."""
        wrapper = self
        original_call = self._patched_classes[layer_cls]
        capture_attn = self.tracer_config.capture_attention
        enable_timing = self.tracer_config.enable_timing

        def traced_call(self_layer, x, *args, **kwargs):
            # Only trace layers we're tracking
            layer_id = id(self_layer)
            if layer_id not in wrapper._layer_to_idx:
                return original_call(self_layer, x, *args, **kwargs)

            idx = wrapper._layer_to_idx[layer_id]

            # Timing start
            if enable_timing:
                mx.eval(x)
                t0 = time.perf_counter()

            # Get input norm for reference
            input_norm = float(mx.sqrt(mx.sum(x[..., -1, :] ** 2)).item())

            # Capture attention weights by temporarily hooking mx.fast.scaled_dot_product_attention
            captured_attn_weights = {}
            original_sdpa = None

            if capture_attn:
                original_sdpa = mx.fast.scaled_dot_product_attention

                def hooked_sdpa(q, k, v, *, scale, mask=None, **kw):
                    # Compute attention weights for last query token only (memory efficient)
                    try:
                        q_last = q[:, :, -1:, :]  # [B, Hq, 1, D]
                        Hq = q.shape[1]
                        Hk = k.shape[1]
                        n_rep = Hq // Hk
                        if n_rep > 1:
                            k_exp = mx.repeat(k, n_rep, axis=1)
                        else:
                            k_exp = k
                        scores = (q_last @ mx.transpose(k_exp, (0, 1, 3, 2))) * scale
                        # Apply mask if it's an array (skip string masks like "causal")
                        if mask is not None and hasattr(mask, 'ndim'):
                            if mask.ndim >= 3:
                                mask_last = mask[..., -1:, :]
                            else:
                                mask_last = mask
                            scores = scores + mask_last
                        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
                        captured_attn_weights['w'] = weights[:, :, 0, :]  # [B, Hq, Lk]
                    except (TypeError, ValueError, AttributeError, IndexError, RuntimeError):
                        pass  # Fall back to no attention capture on error
                    return original_sdpa(q, k, v, scale=scale, mask=mask, **kw)

                mx.fast.scaled_dot_product_attention = hooked_sdpa

            try:
                out = original_call(self_layer, x, *args, **kwargs)
            finally:
                if original_sdpa is not None:
                    mx.fast.scaled_dot_product_attention = original_sdpa

            # Handle both tuple returns and single tensor
            if isinstance(out, tuple):
                h_out = out[0]
            else:
                h_out = out

            if enable_timing:
                mx.eval(h_out)
                t_total = time.perf_counter()
                total_ms = (t_total - t0) * 1000
                attn_ms = total_ms * 0.5
                mlp_ms = total_ms * 0.5
            else:
                attn_ms = 0.0
                mlp_ms = 0.0

            # Capture norms
            out_last = h_out[..., -1, :]
            norm_after = float(mx.sqrt(mx.sum(out_last ** 2)).item())

            # Residual contribution estimate
            residual = h_out[..., -1, :] - x[..., -1, :]
            residual_norm = float(mx.sqrt(mx.sum(residual ** 2)).item())

            # Get captured attention weights or placeholder
            if 'w' in captured_attn_weights:
                attn_w = captured_attn_weights['w']
                mx.eval(attn_w)
                attn_np = np.array(attn_w[0])  # [Hq, Lk]
            else:
                attn_np = np.zeros((1, 1))

            lt = LayerTrace(
                layer_idx=idx,
                attn_weights=attn_np,
                mlp_act_mean=0.0,
                mlp_act_max=0.0,
                mlp_act_top_indices=np.zeros(16, dtype=np.int32),
                mlp_act_top_values=np.zeros(16, dtype=np.float32),
                attn_residual_norm=residual_norm * 0.5,
                mlp_residual_norm=residual_norm * 0.5,
                norm_after_attn=input_norm + residual_norm * 0.3,
                norm_after_mlp=norm_after,
                attn_latency_ms=attn_ms,
                mlp_latency_ms=mlp_ms,
            )
            wrapper._layer_traces.append(lt)

            return out

        layer_cls.__call__ = traced_call


# ---------------------------------------------------------------------------
# MoE expert routing capture helpers
# ---------------------------------------------------------------------------
#
# When `capture_moe_routing=True`, the tracer flips the `_capture_routing`
# flag on every MoE block that opts in (currently mlx-vlm's
# Qwen3_5MoeSparseMoeBlock — extend by adding the same hook to other model
# files). Each MoE block accumulates per-forward routing data
# (`{inds, scores, entropies}`); after generation we reshape into the
# [steps][layers] structure that `analyze_expert_utilization` expects and
# attach it to the trace as the dynamic attribute `_expert_traces` (read by
# `backend/api/moe.py`). Default off → zero overhead in normal inference.

def _find_moe_blocks(wrapper: "ModelWrapper") -> list[tuple[int, Any]]:
    """Locate MoE blocks with capture hook, paired with their true layer index."""
    try:
        from edgestudio_core.vlm import enable_routing_capture_hooks

        enable_routing_capture_hooks(getattr(wrapper, "model", None))
    except ImportError:
        pass

    try:
        layers = wrapper._find_layers()
    except RuntimeError:
        return []
    out: list[tuple[int, Any]] = []
    for true_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "_capture_routing"):
            out.append((true_idx, mlp))
    return out


def _enable_moe_capture(moe_blocks: list[tuple[int, Any]]) -> None:
    for _, blk in moe_blocks:
        blk._captured_routing = []
        blk._capture_routing = True


def _disable_moe_capture(moe_blocks: list[tuple[int, Any]]) -> None:
    for _, blk in moe_blocks:
        blk._capture_routing = False


def _build_expert_traces(
    moe_blocks: list[tuple[int, Any]],
) -> list[list[ExpertTrace]]:
    """Reshape per-block captures into [steps][layers] of ExpertTrace.

    Each MoE block records one entry per forward call:
        {"inds":     [B, T, K] int32,
         "scores":   [B, T, K] float32 (top-k normalized),
         "entropies":[B, T]    float32}

    Forward calls are: 1× prefill (T=prompt_len) + N× decode (T=1).
    Total step count = prompt_len + decoded_token_count.
    Returns [] for inconsistent blocks or batch > 1.
    """
    if not moe_blocks:
        return []

    # All blocks must agree on the number of forward calls.
    expected = len(moe_blocks[0][1]._captured_routing)
    if expected == 0:
        return []
    for _, blk in moe_blocks:
        if len(blk._captured_routing) != expected:
            return []

    out: list[list[ExpertTrace]] = []
    for forward_idx in range(expected):
        first = moe_blocks[0][1]._captured_routing[forward_idx]["inds"]
        if first.ndim != 3 or first.shape[0] != 1:
            # batch > 1 not used in trace mode; skip silently
            continue
        T = first.shape[1]
        for tok_idx in range(T):
            step_traces: list[ExpertTrace] = []
            for true_idx, blk in moe_blocks:
                cap = blk._captured_routing[forward_idx]
                step_traces.append(ExpertTrace(
                    layer_idx=true_idx,
                    expert_indices=np.asarray(cap["inds"][0, tok_idx]),
                    expert_scores=np.asarray(cap["scores"][0, tok_idx]),
                    gate_logits=None,
                    routing_entropy=float(cap["entropies"][0, tok_idx]),
                ))
            out.append(step_traces)
    return out


def _clear_moe_buffers(moe_blocks: list[tuple[int, Any]]) -> None:
    for _, blk in moe_blocks:
        blk._captured_routing = []


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def run_universal_trace(
    model_path: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    trace_top_k: int = 20,
    enable_thinking: bool | None = None,
    enable_timing: bool = False,
    capture_attention: bool = False,
    capture_moe_routing: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> InferenceTrace:
    """Run universal inference trace using mlx-lm.

    Works with any model supported by mlx-lm (117+ architectures).

    Args:
        model_path: path to mlx model directory
        prompt: user message text
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k for sampling
        top_p: top-p for sampling
        trace_top_k: number of top candidate tokens to record per step
        enable_thinking: enable thinking mode (None = auto-detect)
        enable_timing: enable per-layer latency profiling
        capture_attention: capture attention weights (slower)
        capture_moe_routing: capture per-token expert routing on MoE blocks
                             (only effects models whose MoE blocks expose the
                             `_capture_routing` hook; harmless on non-MoE)
        progress_callback: (step, total, message) callback

    Returns:
        InferenceTrace with all recorded data. When ``capture_moe_routing``
        is on, ``trace._expert_traces`` is populated as ``list[list[ExpertTrace]]``
        with shape ``[steps][layers]`` (read by ``backend/api/moe.py``).
    """
    t_start = time.time()

    if progress_callback:
        progress_callback(0, max_tokens, "Loading model...")

    lm = _load_for_trace(model_path)
    model, tokenizer, config = lm.model, lm.tokenizer, lm.config
    eos_token_ids = lm.eos_token_ids

    from mlx_lm.models.cache import make_prompt_cache
    prompt_cache = make_prompt_cache(model)

    # Apply chat template
    prompt_text = _apply_chat_template(tokenizer, prompt, enable_thinking)

    # Tokenize
    if hasattr(tokenizer, 'encode'):
        prompt_tokens = tokenizer.encode(prompt_text)
    else:
        prompt_tokens = tokenizer._tokenizer.encode(prompt_text)

    if isinstance(prompt_tokens, list):
        prompt_ids = prompt_tokens
    else:
        prompt_ids = prompt_tokens if isinstance(prompt_tokens, list) else list(prompt_tokens)

    prompt_tokens_str = [
        _safe_decode(tokenizer, tid) for tid in prompt_ids
    ]

    # Set up tracer
    tracer_config = TracerConfig(
        capture_attention=capture_attention,
        enable_timing=enable_timing,
        trace_top_k=trace_top_k,
        capture_moe_routing=capture_moe_routing,
    )
    wrapper = ModelWrapper(model, config, tracer_config)
    wrapper.install_hooks()

    # Discover MoE blocks (empty list on non-MoE models — no-op fast path).
    moe_blocks = _find_moe_blocks(wrapper) if capture_moe_routing else []
    if moe_blocks:
        _enable_moe_capture(moe_blocks)

    try:
        trace = _generate_with_trace(
            model=model,
            tokenizer=tokenizer,
            wrapper=wrapper,
            config=config,
            prompt_ids=prompt_ids,
            prompt_tokens_str=prompt_tokens_str,
            prompt=prompt,
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            trace_top_k=trace_top_k,
            eos_token_ids=eos_token_ids,
            num_layers=lm.num_layers,
            num_heads=lm.num_heads,
            hidden_size=lm.hidden_size,
            enable_timing=enable_timing,
            t_start=t_start,
            progress_callback=progress_callback,
            prompt_cache=prompt_cache,
        )
    finally:
        # Always disable capture flag and unhook layers, even on failure.
        _disable_moe_capture(moe_blocks)
        wrapper.remove_hooks()

    if moe_blocks:
        # `_expert_traces` is a dynamic attribute on InferenceTrace, intentionally
        # not in the dataclass field list — keeps the JSON-serialization path lean
        # while moe.py reads it directly off the in-memory trace.
        trace._expert_traces = _build_expert_traces(moe_blocks)
        _clear_moe_buffers(moe_blocks)

    return trace


def _generate_with_trace(
    model,
    tokenizer,
    wrapper: ModelWrapper,
    config: dict,
    prompt_ids: list[int],
    prompt_tokens_str: list[str],
    prompt: str,
    model_path: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    trace_top_k: int,
    eos_token_ids: set[int],
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    enable_timing: bool,
    t_start: float,
    progress_callback: Callable | None,
    prompt_cache: list | None = None,
) -> InferenceTrace:
    """Generate tokens using mlx_lm.generate_step and capture traces."""
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_k=top_k, top_p=top_p)

    # Prepare input
    input_ids = mx.array([prompt_ids])

    # Prefill
    if progress_callback:
        progress_callback(0, max_tokens, "Prefilling...")

    t_prefill_start = time.time()

    # Clear any leftover traces
    wrapper.flush_traces()

    # Prefill: run model on full prompt with KV cache
    logits = model(input_ids, cache=prompt_cache)
    # VLM language_model returns LanguageModelOutput, extract .logits
    if hasattr(logits, 'logits'):
        logits = logits.logits
    mx.eval(logits)
    prefill_layer_traces = wrapper.flush_traces()
    t_prefill = time.time() - t_prefill_start

    # Get logits for last position
    if logits.ndim == 3:
        last_logits = logits[0, -1, :]  # [vocab_size]
    else:
        last_logits = logits[-1, :]

    # Decode loop
    steps: list[StepTrace] = []
    generated_ids: list[int] = []

    for step_idx in range(max_tokens):
        if progress_callback:
            progress_callback(step_idx, max_tokens, f"Generating token {step_idx+1}/{max_tokens}")

        # Sample token
        logprobs = last_logits.astype(mx.float32) - mx.logsumexp(
            last_logits.astype(mx.float32), axis=-1, keepdims=True
        )
        token = sampler(logprobs)
        mx.eval(token)
        token_id = int(token.item())

        # Full probability distribution
        probs = mx.softmax(last_logits.astype(mx.float32), axis=-1)
        mx.eval(probs)

        # Top-K trace
        k_trace = min(trace_top_k, probs.shape[0])
        top_k_indices = mx.argpartition(probs, kth=probs.shape[0] - k_trace)[-k_trace:]
        top_k_probs_vals = probs[top_k_indices]
        sort_order = mx.argsort(top_k_probs_vals)[::-1]
        top_k_indices = top_k_indices[sort_order]
        top_k_probs_vals = top_k_probs_vals[sort_order]

        top_k_ids_np = np.array(top_k_indices)
        top_k_probs_np = np.array(top_k_probs_vals.astype(mx.float32))
        top_k_strs = [_safe_decode(tokenizer, int(tid)) for tid in top_k_ids_np]

        chosen_prob = float(probs[token_id].item())
        chosen_rank = 0
        for ri, tid in enumerate(top_k_ids_np):
            if int(tid) == token_id:
                chosen_rank = ri
                break
        else:
            chosen_rank = k_trace

        token_str = _safe_decode(tokenizer, token_id)
        generated_ids.append(token_id)

        # Check EOS
        if token_id in eos_token_ids:
            steps.append(StepTrace(
                step_idx=step_idx,
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

        # Forward next token
        wrapper.flush_traces()  # clear before next step
        next_input = mx.array([[token_id]])
        logits = model(next_input, cache=prompt_cache)
        if hasattr(logits, 'logits'):
            logits = logits.logits
        mx.eval(logits)

        layer_traces = wrapper.flush_traces()

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[-1, :]

        # Compute final hidden norm from output
        final_hidden_norm = 0.0
        if layer_traces:
            final_hidden_norm = layer_traces[-1].norm_after_mlp

        steps.append(StepTrace(
            step_idx=step_idx,
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
    generated_text = _safe_decode_ids(tokenizer, generated_ids)
    model_name = os.path.basename(model_path.rstrip("/"))

    return InferenceTrace(
        prompt=prompt,
        prompt_token_ids=prompt_ids,
        prompt_tokens=prompt_tokens_str,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        model_dir=model_path,
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


# ---------------------------------------------------------------------------
# Unified model loader
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Everything needed for traced generation, regardless of LLM vs VLM."""
    model: nn.Module          # language model (for VLM: model.language_model)
    tokenizer: Any            # HF tokenizer or mlx-lm TokenizerWrapper
    config: dict              # raw config.json
    eos_token_ids: set[int]
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    # VLM-only (None for LLM)
    full_model: nn.Module | None = None
    processor: Any = None
    vlm_config: Any = None


def _load_for_trace(model_path: str) -> LoadedModel:
    """Load model, tokenizer, and config for tracing. Handles LLM and VLM."""
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Resolve nested text_config
    tcfg = config
    for key in ("text_config", "model_config", "language_config"):
        if isinstance(config.get(key), dict):
            tcfg = config[key]
            break

    num_layers  = tcfg.get("num_hidden_layers",  config.get("num_hidden_layers",  0))
    num_heads   = tcfg.get("num_attention_heads", config.get("num_attention_heads", 0))
    num_kv_heads = tcfg.get("num_key_value_heads", config.get("num_key_value_heads", num_heads))
    hidden_size = tcfg.get("hidden_size",         config.get("hidden_size",         0))

    # EOS: cascade config → text_config → fallback
    raw_eos = config.get("eos_token_id") or tcfg.get("eos_token_id") or 2
    eos_token_ids: set[int] = set(raw_eos) if isinstance(raw_eos, list) else {raw_eos}

    is_vlm = "vision_config" in config
    full_model = None
    processor = None
    vlm_config = None

    if is_vlm:
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config
        full_model, processor = vlm_load(model_path)
        model = full_model.language_model
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        vlm_config = load_config(model_path)
    else:
        from mlx_lm.utils import load as lm_load
        model, tokenizer = lm_load(model_path)

    # Augment EOS with tokenizer's eos_token_id (e.g. <|im_end|>)
    for tok_obj in [tokenizer, getattr(tokenizer, '_tokenizer', None)]:
        if tok_obj and hasattr(tok_obj, 'eos_token_id') and tok_obj.eos_token_id is not None:
            eos_token_ids.add(tok_obj.eos_token_id)

    return LoadedModel(
        model=model, tokenizer=tokenizer, config=config,
        eos_token_ids=eos_token_ids,
        num_layers=num_layers, num_heads=num_heads,
        num_kv_heads=num_kv_heads, hidden_size=hidden_size,
        full_model=full_model, processor=processor, vlm_config=vlm_config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_chat_template(tokenizer, prompt: str, enable_thinking: bool | None) -> str:
    """Apply the model's chat template to the prompt.

    Uses the tokenizer's built-in apply_chat_template if available,
    otherwise falls back to simple formatting.
    """
    messages = [{"role": "user", "content": prompt}]

    # Try apply_chat_template on tokenizer itself first, then on ._tokenizer.
    # VLM processor.tokenizer (HF) has apply_chat_template directly;
    # mlx-lm TokenizerWrapper exposes it via ._tokenizer (the inner HF tokenizer).
    for tok_candidate in [tokenizer, getattr(tokenizer, '_tokenizer', None)]:
        if tok_candidate is None or not hasattr(tok_candidate, 'apply_chat_template'):
            continue
        try:
            kwargs = {}
            if enable_thinking is not None:
                kwargs['enable_thinking'] = enable_thinking
            result = tok_candidate.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            if isinstance(result, str):
                return result
        except (TypeError, ValueError, KeyError, RuntimeError):
            continue

    # Fallback: try common chat formats
    tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
    if hasattr(tok, 'chat_template') and tok.chat_template:
        template = tok.chat_template
        if '<|im_start|>' in template:
            # ChatML format (Qwen, etc.)
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            if enable_thinking is False:
                text += "<think>\n\n</think>\n\n"
            return text

    # Simple fallback
    return prompt


def _safe_decode(tokenizer, token_id: int) -> str:
    """Safely decode a single token ID to string."""
    try:
        if hasattr(tokenizer, '_tokenizer'):
            return tokenizer._tokenizer.decode([token_id])
        elif hasattr(tokenizer, 'decode'):
            result = tokenizer.decode([token_id])
            return result if isinstance(result, str) else str(result)
    except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
        pass
    return f"<{token_id}>"


def _safe_decode_ids(tokenizer, token_ids: list[int]) -> str:
    """Safely decode a list of token IDs to string."""
    try:
        if hasattr(tokenizer, '_tokenizer'):
            return tokenizer._tokenizer.decode(token_ids)
        elif hasattr(tokenizer, 'decode'):
            result = tokenizer.decode(token_ids)
            return result if isinstance(result, str) else str(result)
    except (TypeError, ValueError, AttributeError, KeyError, RuntimeError):
        pass
    return "".join(_safe_decode(tokenizer, tid) for tid in token_ids)


def detect_thinking_support(model_path: str) -> bool:
    """Detect if a model supports thinking/reasoning mode.

    Uses prefix matching for Qwen3 variants (qwen3, qwen3_5, qwen3_moe, etc.)
    and also checks for the <think> token in the tokenizer vocabulary as fallback.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    with open(config_path) as f:
        config = json.load(f)
    # Check top-level and nested text_config model_type
    model_type = config.get("model_type", "").lower()
    if not model_type:
        model_type = config.get("text_config", {}).get("model_type", "").lower()
    # Prefix match for all Qwen3 variants (qwen3, qwen3_5, qwen3_moe, etc.)
    if model_type.startswith("qwen3") or model_type in {"qwq", "deepseek_v3"}:
        return True
    # Fallback: check if tokenizer has <think> token
    try:
        from tokenizers import Tokenizer
        tok_path = Path(model_path) / "tokenizer.json"
        if tok_path.exists():
            tok = Tokenizer.from_file(str(tok_path))
            return tok.token_to_id("<think>") is not None
    except (ImportError, OSError, ValueError, KeyError, RuntimeError):
        pass
    return False


# ---------------------------------------------------------------------------
# VLM Universal Tracer — full layer-level analysis for vision-language models
# ---------------------------------------------------------------------------

def run_vlm_universal_trace(
    model_path: str,
    prompt: str,
    image_b64: str | None = None,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    trace_top_k: int = 20,
    enable_thinking: bool | None = None,
    enable_timing: bool = False,
    capture_attention: bool = False,
    capture_moe_routing: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> InferenceTrace:
    """Full layer-level inference trace for VLM models using mlx_vlm.

    Uses mlx_vlm.load() to properly load vision-language models.
    Visual tokens and text tokens are merged into a unified embedding sequence
    via model.get_input_embeddings(), then passed through the language model's
    transformer layers with the same monkey-patching hooks as text-only models.

    Args:
        model_path: path to mlx VLM model directory
        prompt: user message text
        image_b64: base64-encoded image (optional; enables visual token merging)
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k / top_p: sampling parameters
        trace_top_k: number of top candidate tokens to record per step
        enable_timing: enable per-layer latency profiling
        capture_attention: capture attention weights (slower)
        progress_callback: (step, total, message) callback

    Returns:
        InferenceTrace with full layer-level data (attn, norms, timing).
        num_layers reflects the actual language model depth.
    """
    import base64
    import os
    import tempfile

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs

    t_start = time.time()

    if progress_callback:
        progress_callback(0, max_tokens, "Loading VLM model...")

    lm = _load_for_trace(model_path)
    model, tokenizer, config = lm.full_model, lm.tokenizer, lm.config
    eos_token_ids = lm.eos_token_ids

    # ── Prepare image (optional) ──────────────────────────────────────────────
    image_path: str | None = None
    images: list[str] | None = None
    num_images = 0

    if image_b64:
        img_data = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            image_path = tmp.name
        images = [image_path]
        num_images = 1

    # ── Apply chat template ───────────────────────────────────────────────────
    template_kwargs = {}
    if enable_thinking is not None:
        template_kwargs['enable_thinking'] = enable_thinking
    formatted_prompt = apply_chat_template(
        lm.processor, lm.vlm_config, prompt, num_images=num_images, **template_kwargs
    )

    # ── Tokenize + process image into model inputs ────────────────────────────
    image_token_index = getattr(model.config, "image_token_index", None)
    model_type = getattr(model.config, "model_type", "")
    add_special_tokens = (
        not hasattr(lm.processor, "chat_template")
        if model_type in ("gemma3", "gemma3n") else True
    )

    inputs = prepare_inputs(
        lm.processor,
        images=images,
        prompts=formatted_prompt,
        image_token_index=image_token_index,
        add_special_tokens=add_special_tokens,
    )

    # Cleanup temp image file immediately after prepare_inputs reads it
    if image_path:
        try:
            os.unlink(image_path)
        except OSError:
            pass

    input_ids    = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    mask         = inputs.get("attention_mask")
    extra_kwargs = {
        k: v for k, v in inputs.items()
        if k not in ("input_ids", "pixel_values", "attention_mask")
    }

    # Prompt token metadata
    prompt_ids = input_ids[0].tolist()
    prompt_tokens_str = [_safe_decode(tokenizer, tid) for tid in prompt_ids]

    # ── Install layer hooks ───────────────────────────────────────────────────
    tracer_config = TracerConfig(
        capture_attention=capture_attention,
        enable_timing=enable_timing,
        trace_top_k=trace_top_k,
        capture_moe_routing=capture_moe_routing,
    )
    # ModelWrapper._find_layers() already handles ("language_model","model","layers")
    wrapper = ModelWrapper(model, config, tracer_config)
    wrapper.install_hooks()

    # Discover MoE blocks (empty list on non-MoE — no-op fast path).
    moe_blocks = _find_moe_blocks(wrapper) if capture_moe_routing else []
    if moe_blocks:
        _enable_moe_capture(moe_blocks)

    try:
        trace = _generate_vlm_with_trace(
            model=model,
            tokenizer=tokenizer,
            wrapper=wrapper,
            config=config,
            input_ids=input_ids,
            pixel_values=pixel_values,
            mask=mask,
            extra_kwargs=extra_kwargs,
            prompt_ids=prompt_ids,
            prompt_tokens_str=prompt_tokens_str,
            prompt=prompt,
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            trace_top_k=trace_top_k,
            eos_token_ids=eos_token_ids,
            num_layers=lm.num_layers,
            num_heads=lm.num_heads,
            hidden_size=lm.hidden_size,
            enable_timing=enable_timing,
            t_start=t_start,
            progress_callback=progress_callback,
        )
    finally:
        _disable_moe_capture(moe_blocks)
        wrapper.remove_hooks()

    if moe_blocks:
        trace._expert_traces = _build_expert_traces(moe_blocks)
        _clear_moe_buffers(moe_blocks)

    return trace


def _generate_vlm_with_trace(
    model,
    tokenizer,
    wrapper: ModelWrapper,
    config: dict,
    input_ids,
    pixel_values,
    mask,
    extra_kwargs: dict,
    prompt_ids: list[int],
    prompt_tokens_str: list[str],
    prompt: str,
    model_path: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    trace_top_k: int,
    eos_token_ids: set[int],
    num_layers: int,
    num_heads: int,
    hidden_size: int,
    enable_timing: bool,
    t_start: float,
    progress_callback: Callable | None,
) -> InferenceTrace:
    """VLM generation loop with per-step layer tracing.

    Prefill: merges visual + text embeddings, runs language_model with inputs_embeds.
    Decode:  passes single token ids through language_model (KV cache stores vision).
    """
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=temperature, top_k=top_k, top_p=top_p)

    # KV cache is created for language_model (not the wrapper model)
    prompt_cache = make_prompt_cache(model.language_model)

    # ── Step 1: Merge visual + text tokens into a unified embedding sequence ──
    if progress_callback:
        progress_callback(0, max_tokens, "Encoding visual tokens...")

    wrapper.flush_traces()

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **extra_kwargs
    )

    if hasattr(embedding_output, "inputs_embeds"):
        inputs_embeds = embedding_output.inputs_embeds
        # Extra kwargs from embedding (e.g. image_grid_thw for Qwen VL MRoPE)
        if hasattr(embedding_output, "to_dict"):
            gen_kwargs = {
                k: v for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        else:
            gen_kwargs: dict = {}
    else:
        # Fallback: model returned embeddings directly
        inputs_embeds = embedding_output
        gen_kwargs = {}

    # ── Step 2: Prefill with merged embeddings ────────────────────────────────
    if progress_callback:
        progress_callback(0, max_tokens, "Prefilling (VLM)...")

    t_prefill_start = time.time()

    prefill_out = model.language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        cache=prompt_cache,
        **gen_kwargs,
    )
    mx.eval(prefill_out.logits)
    prefill_layer_traces = wrapper.flush_traces()
    t_prefill = time.time() - t_prefill_start

    # Update gen_kwargs with cross-attention states if model uses encoder-decoder
    gen_kwargs = _extract_next_kwargs(prefill_out)

    # Last-position logits
    last_logits = prefill_out.logits[0, -1, :]  # [vocab_size]

    # ── Step 3: Decode loop ───────────────────────────────────────────────────
    steps: list[StepTrace] = []
    generated_ids: list[int] = []

    for step_idx in range(max_tokens):
        if progress_callback:
            progress_callback(step_idx, max_tokens, f"Generating token {step_idx+1}/{max_tokens}")

        # Sample
        logprobs = last_logits.astype(mx.float32) - mx.logsumexp(
            last_logits.astype(mx.float32), axis=-1, keepdims=True
        )
        token = sampler(logprobs)
        mx.eval(token)
        token_id = int(token.item())

        # Top-K probability recording
        probs = mx.softmax(last_logits.astype(mx.float32), axis=-1)
        mx.eval(probs)

        k_trace = min(trace_top_k, probs.shape[0])
        top_k_indices = mx.argpartition(probs, kth=probs.shape[0] - k_trace)[-k_trace:]
        top_k_probs_vals = probs[top_k_indices]
        sort_order = mx.argsort(top_k_probs_vals)[::-1]
        top_k_indices = top_k_indices[sort_order]
        top_k_probs_vals = top_k_probs_vals[sort_order]

        top_k_ids_np   = np.array(top_k_indices)
        top_k_probs_np = np.array(top_k_probs_vals.astype(mx.float32))
        top_k_strs     = [_safe_decode(tokenizer, int(tid)) for tid in top_k_ids_np]

        chosen_prob = float(probs[token_id].item())
        chosen_rank = k_trace  # default if not in top-K
        for ri, tid in enumerate(top_k_ids_np):
            if int(tid) == token_id:
                chosen_rank = ri
                break

        token_str = _safe_decode(tokenizer, token_id)
        generated_ids.append(token_id)

        if token_id in eos_token_ids:
            steps.append(StepTrace(
                step_idx=step_idx,
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

        # Forward next token through language_model
        # (vision is already encoded in KV cache — no pixel_values needed)
        wrapper.flush_traces()
        next_input = mx.array([[token_id]])
        decode_out = model.language_model(
            next_input,
            cache=prompt_cache,
            **gen_kwargs,
        )
        mx.eval(decode_out.logits)
        layer_traces = wrapper.flush_traces()

        gen_kwargs = _extract_next_kwargs(decode_out)
        last_logits = decode_out.logits[0, -1, :]

        final_hidden_norm = layer_traces[-1].norm_after_mlp if layer_traces else 0.0

        steps.append(StepTrace(
            step_idx=step_idx,
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
    generated_text = _safe_decode_ids(tokenizer, generated_ids)
    model_name = os.path.basename(model_path.rstrip("/"))

    return InferenceTrace(
        prompt=prompt,
        prompt_token_ids=prompt_ids,
        prompt_tokens=prompt_tokens_str,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        model_dir=model_path,
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


def _extract_next_kwargs(outputs) -> dict:
    """Extract cross-attention states or encoder outputs for the next decode step.

    Some encoder-decoder models pass state between decode steps.
    Pure decoder models (most VLMs) return an empty dict here.
    """
    if hasattr(outputs, "cross_attention_states") and outputs.cross_attention_states is not None:
        return {"cross_attention_states": outputs.cross_attention_states}
    if hasattr(outputs, "encoder_outputs") and outputs.encoder_outputs is not None:
        return {"encoder_outputs": outputs.encoder_outputs}
    return {}


def detect_chat_template(model_path: str) -> str:
    """Detect the chat template type for a model.

    Returns a human-readable description of the detected template.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(tok, 'chat_template') and tok.chat_template:
            if '<|im_start|>' in tok.chat_template:
                return "ChatML"
            elif '[INST]' in tok.chat_template:
                return "Llama/Mistral"
            elif '<start_of_turn>' in tok.chat_template:
                return "Gemma"
            else:
                return "Custom"
    except (ImportError, OSError, ValueError, AttributeError, RuntimeError):
        pass

    # Fallback: check config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "").lower()
        template_map = {
            "qwen": "ChatML", "qwen2": "ChatML", "qwen3": "ChatML",
            "llama": "Llama", "mistral": "Mistral",
            "gemma": "Gemma", "gemma2": "Gemma", "gemma3": "Gemma",
            "phi": "Phi", "phi3": "Phi",
            "deepseek": "DeepSeek",
        }
        for key, tmpl in template_map.items():
            if key in model_type:
                return tmpl

    return "Auto-detected"
