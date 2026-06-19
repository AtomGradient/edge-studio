# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""LLM text generation for chat endpoints."""

from __future__ import annotations

import json
import os
import time
import asyncio
import threading

from backend.services.error_mapper import map_error
from backend.api.chat_loaders import _get_or_load_mlx_model
from backend.services.neural_imprint_runtime import clone_neural_imprint_cache_for_model
from backend.services.mlx_runtime_gate import get_mlx_runtime_lock

import logging

logger = logging.getLogger(__name__)


def _has_vision_config(model_dir: str) -> bool:
    """Check if model is a VLM by inspecting config.json."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        return "vision_config" in cfg
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def _apply_chat_template_from_messages(
    tokenizer,
    messages: list[dict],
    enable_thinking: bool | None,
) -> str:
    """Apply chat template to already-formed chat messages."""

    tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
    if hasattr(tok, 'apply_chat_template'):
        try:
            kwargs = {}
            if enable_thinking is not None:
                kwargs['enable_thinking'] = enable_thinking
            result = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            if isinstance(result, str):
                # For distilled models, apply_chat_template may accept
                # enable_thinking=False but not actually add the prefill.
                # Force-add the empty thinking block so the model skips <think>.
                if enable_thinking is False and '<think>' not in result:
                    result += '<think>\n\n</think>\n\n'
                return result
        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
            logger.debug("apply_chat_template failed, trying fallback: %s", exc)

    # Fallback: ChatML
    if hasattr(tok, 'chat_template') and tok.chat_template and '<|im_start|>' in tok.chat_template:
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        text = "\n".join(parts)
        if enable_thinking is False:
            text += "<think>\n\n</think>\n\n"
        return text

    # Simple fallback
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(messages[-1].get("content", "")) if messages else ""


def _apply_chat_template(tokenizer, prompt: str, history: list[dict], enable_thinking: bool | None) -> str:
    """Apply chat template with conversation history."""
    if not prompt and history:
        return _apply_chat_template_from_messages(tokenizer, list(history), enable_thinking)
    messages = list(history) + [{"role": "user", "content": prompt}]
    return _apply_chat_template_from_messages(tokenizer, messages, enable_thinking)


def _apply_neural_imprint_turn_template(
    tokenizer,
    prompt: str,
    history: list[dict],
    enable_thinking: bool | None,
) -> str:
    """Build only the continuation after an already-restored Neural Imprint prefix."""

    tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
    template = getattr(tok, "chat_template", None)
    continuation_messages = [
        {"role": msg.get("role"), "content": msg.get("content", "")}
        for msg in history
        if msg.get("role") in {"user", "assistant", "tool"}
    ]
    if prompt:
        continuation_messages.append({"role": "user", "content": prompt})
    if template and "<|im_start|>" not in str(template):
        try:
            kwargs = {}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            result = tok.apply_chat_template(
                continuation_messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            if isinstance(result, str):
                return result
        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
            logger.debug("persona continuation chat template failed, using ChatML: %s", exc)

    parts: list[str] = []
    for msg in continuation_messages:
        role = msg.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        content = str(msg.get("content", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if enable_thinking is False:
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


import re

_THINK_BLOCK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    # Full blocks
    text = _THINK_BLOCK_RE.sub('', text)
    # Unclosed <think> at the end (model didn't finish thinking)
    idx = text.find('<think>')
    if idx != -1:
        text = text[:idx]
    return text.strip()


def _get_think_token_ids(tokenizer) -> tuple[int | None, int | None]:
    """Get token IDs for <think> and </think> special tokens."""
    tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
    start_id = end_id = None
    try:
        if hasattr(tok, 'token_to_id'):
            start_id = tok.token_to_id('<think>')
            end_id = tok.token_to_id('</think>')
        elif hasattr(tok, 'convert_tokens_to_ids'):
            start_id = tok.convert_tokens_to_ids('<think>')
            end_id = tok.convert_tokens_to_ids('</think>')
    except Exception:
        pass
    return start_id, end_id


def _get_special_token_ids(tokenizer, tokens: list[str]) -> set[int]:
    """Resolve tokenizer special token strings to ids when model config is incomplete."""
    tok = tokenizer._tokenizer if hasattr(tokenizer, '_tokenizer') else tokenizer
    ids: set[int] = set()
    for token in tokens:
        token_id = None
        try:
            if hasattr(tok, 'token_to_id'):
                token_id = tok.token_to_id(token)
            elif hasattr(tok, 'convert_tokens_to_ids'):
                token_id = tok.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return ids


def _safe_decode(tokenizer, token_id: int) -> str:
    try:
        if hasattr(tokenizer, '_tokenizer'):
            return tokenizer._tokenizer.decode([token_id])
        elif hasattr(tokenizer, 'decode'):
            result = tokenizer.decode([token_id])
            return result if isinstance(result, str) else str(result)
    except (TypeError, ValueError, AttributeError, IndexError):
        pass
    return f"<{token_id}>"


def _generate_streaming(
    model_id: str,
    model_dir: str,
    prompt: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    enable_thinking: bool | None,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
    enable_dsr: bool = False,
    dsr_budget: int | None = None,
    use_neural_imprint: bool = False,
):
    """Run token generation in a thread, pushing events to the async queue."""
    import mlx.core as mx

    def _send(event: dict):
        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop)

    runtime_gate = get_mlx_runtime_lock()
    runtime_gate.acquire()
    try:
        _send({"type": "status", "message": "Loading model..."})
        model, tokenizer = _get_or_load_mlx_model(model_dir)

        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        from backend.api.chat_params import get_generation_params
        from backend.core.dsr_cache import build_dsr_config, make_prompt_cache

        # Model-aware parameters (max_tokens already capped by chat.py)
        model_params = get_generation_params(model_dir)

        # Repetition penalty processor
        rep_processor = None
        if model_params.repetition_penalty > 1.0:
            rep_processor = make_repetition_penalty(
                model_params.repetition_penalty,
                context_size=model_params.repetition_context_size,
            )

        persona_status = None
        if use_neural_imprint:
            persona_cache, persona_status = clone_neural_imprint_cache_for_model(
                model=model,
                model_id=model_id,
                model_dir=model_dir,
            )
            if persona_cache is None:
                _send({
                    "type": "error",
                    "message": "Neural Imprint is not loaded for this model.",
                })
                return
            prompt_cache = persona_cache
            _send({
                "type": "status",
                "message": (
                    f"Neural Imprint active"
                    f" ({persona_status.prefix_token_count} prefix tokens)"
                    if persona_status and persona_status.prefix_token_count is not None
                    else "Neural Imprint active"
                ),
                "use_neural_imprint": True,
                "neural_imprint_artifact_id": (
                    persona_status.artifact_id if persona_status else None
                ),
                "neural_imprint_prefix_token_count": (
                    persona_status.prefix_token_count if persona_status else None
                ),
            })
        else:
            dsr_config = build_dsr_config(dsr_budget) if enable_dsr else None
            prompt_cache = make_prompt_cache(model, dsr_config=dsr_config)
        sampler = make_sampler(temp=temperature, top_k=top_k, top_p=top_p)

        # Build prompt text
        if use_neural_imprint:
            prompt_text = _apply_neural_imprint_turn_template(
                tokenizer,
                prompt,
                history,
                enable_thinking,
            )
        else:
            prompt_text = _apply_chat_template(tokenizer, prompt, history, enable_thinking)
        logger.debug("enable_thinking=%s, prompt tail: %r", enable_thinking,
                      prompt_text[-120:] if len(prompt_text) > 120 else prompt_text)

        # Tokenize
        if hasattr(tokenizer, 'encode'):
            prompt_ids = tokenizer.encode(prompt_text)
        else:
            prompt_ids = tokenizer._tokenizer.encode(prompt_text)
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        # EOS tokens (cached in model_params)
        eos_token_ids = set(model_params.eos_token_ids)
        eos_token_ids.update(
            _get_special_token_ids(
                tokenizer,
                ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
            )
        )

        _send({"type": "status", "message": "Prefilling..."})

        # Prefill
        input_ids = mx.array([prompt_ids])
        t_start = time.time()
        logits = model(input_ids, cache=prompt_cache)
        mx.eval(logits)
        t_prefill = time.time() - t_start

        if logits.ndim == 3:
            last_logits = logits[0, -1, :]
        else:
            last_logits = logits[-1, :]

        _send({"type": "status", "message": "Generating..."})

        # Token-level thinking filter: suppress <think>...</think> during streaming
        think_start_id, think_end_id = _get_think_token_ids(tokenizer)
        in_thinking = False

        # Decode loop
        generated_ids: list[int] = []
        t_decode_start = time.time()

        # Repetition detection (safety net): stop if same n-gram repeats too many times
        _REP_NGRAM = model_params.rep_ngram
        _REP_THRESHOLD = model_params.rep_threshold
        repetition_stopped = False

        for step_idx in range(max_tokens):
            if cancel_event.is_set():
                _send({"type": "cancelled"})
                return

            # Apply repetition penalty to logits before sampling
            if rep_processor is not None and generated_ids:
                last_logits = rep_processor(
                    mx.array(generated_ids[-40:]),
                    last_logits[None, :] if last_logits.ndim == 1 else last_logits,
                )
                if last_logits.ndim == 2:
                    last_logits = last_logits[0]

            # Sample
            logprobs = last_logits.astype(mx.float32) - mx.logsumexp(
                last_logits.astype(mx.float32), axis=-1, keepdims=True
            )
            token = sampler(logprobs)
            mx.eval(token)
            token_id = int(token.item())

            # Check EOS before appending — don't include EOS in decoded output
            if token_id in eos_token_ids:
                break

            token_str = _safe_decode(tokenizer, token_id)
            generated_ids.append(token_id)

            # Check repetition: if the last n-gram has repeated too many times, stop
            n = _REP_NGRAM
            total = len(generated_ids)
            if total >= n * _REP_THRESHOLD:
                pattern = generated_ids[-n:]
                repeats = 0
                for offset in range(1, _REP_THRESHOLD + 1):
                    start = total - n * (offset + 1)
                    if start < 0:
                        break
                    if generated_ids[start:start + n] == pattern:
                        repeats += 1
                    else:
                        break
                if repeats >= _REP_THRESHOLD - 1:
                    logger.info("Repetition detected at step %d, stopping generation", step_idx)
                    repetition_stopped = True
                    break

            # Token-level thinking filter: don't send thinking tokens to frontend
            should_send = True
            if enable_thinking is False and think_start_id is not None:
                if token_id == think_start_id:
                    in_thinking = True
                    should_send = False
                elif token_id == think_end_id:
                    in_thinking = False
                    should_send = False
                elif in_thinking:
                    should_send = False

            # Send token event (only non-thinking tokens)
            if should_send:
                _send({
                    "type": "token",
                    "token": token_str,
                    "token_id": token_id,
                })

            # Forward next token
            next_input = mx.array([[token_id]])
            logits = model(next_input, cache=prompt_cache)
            mx.eval(logits)

            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]

        t_total = time.time() - t_start
        t_decode = time.time() - t_decode_start
        tokens_per_sec = len(generated_ids) / t_decode if t_decode > 0 else 0

        # If repetition stopped, trim trailing repeated tokens
        if repetition_stopped:
            n = _REP_NGRAM
            pattern = generated_ids[-n:]
            # Walk backwards to find where the repetition started
            trim_to = len(generated_ids) - n
            while trim_to >= n and generated_ids[trim_to - n:trim_to] == pattern:
                trim_to -= n
            generated_ids = generated_ids[:trim_to + n]  # keep one instance

        # Decode full text
        try:
            if hasattr(tokenizer, '_tokenizer'):
                full_text = tokenizer._tokenizer.decode(generated_ids)
            elif hasattr(tokenizer, 'decode'):
                full_text = tokenizer.decode(generated_ids)
            else:
                full_text = "".join(_safe_decode(tokenizer, tid) for tid in generated_ids)
        except (TypeError, ValueError, AttributeError):
            full_text = "".join(_safe_decode(tokenizer, tid) for tid in generated_ids)

        # Strip any thinking blocks the model generated despite disable
        full_text = _strip_thinking(full_text)

        _send({
            "type": "complete",
            "full_text": full_text,
            "total_tokens": len(generated_ids),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "prefill_time": round(t_prefill, 3),
            "total_time": round(t_total, 3),
        })

    except Exception as exc:
        logger.exception("LLM generation failed")
        user_msg, _ = map_error(exc)
        _send({"type": "error", "message": user_msg})
    finally:
        runtime_gate.release()
