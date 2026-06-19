# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Duplex mode: interleaved LLM token generation + TTS audio streaming.

Runs LLM and TTS alternately within a single pool task so the user
hears audio while the LLM is still generating text.  Flow:

    LLM tokens → accumulate → sentence boundary detected
      → pause LLM → run TTS on sentence → stream audio_chunk
      → resume LLM → next sentence …

Both models share the same thread — no concurrency issues with MLX.
"""

from __future__ import annotations

import re
import time
import asyncio
import threading

import numpy as np

from backend.services.error_mapper import map_error
from backend.api.chat_loaders import _get_or_load_mlx_model, _get_or_load_tts_model
from backend.api.chat_llm import _apply_chat_template, _safe_decode, _strip_thinking, _get_think_token_ids
from backend.api.chat_tts import _audio_to_wav_b64, _is_instruct_tts

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence boundary detection for TTS chunking
# ---------------------------------------------------------------------------

_SENT_END_RE = re.compile(r'[。！？\n]|[.!?](?:\s|$)')
_CLAUSE_END_RE = re.compile(r'[，、,;：:]')


def _extract_tts_chunks(
    buffer: str,
    min_chars: int = 20,
    max_chars: int = 80,
) -> tuple[list[str], str]:
    """Extract complete sentences/clauses from *buffer* for TTS.

    Returns ``(sentences, remaining_buffer)``.
    Splits on sentence-ending punctuation first (。！？.!?\\n),
    falls back to clause boundaries (，、,;) for very long runs.
    """
    sentences: list[str] = []
    while len(buffer) >= min_chars:
        best: int | None = None

        # 1. Try sentence-ending punctuation
        for m in _SENT_END_RE.finditer(buffer):
            if m.start() >= min_chars - 1:
                best = m.end()
                break

        # 2. Fallback: clause boundary for long buffers
        if best is None and len(buffer) >= max_chars:
            for m in _CLAUSE_END_RE.finditer(buffer):
                if m.start() >= min_chars - 1:
                    best = m.end()
                    break

        if best is None:
            break

        sentence = buffer[:best].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[best:].lstrip()

    return sentences, buffer


# ---------------------------------------------------------------------------
# Duplex generation
# ---------------------------------------------------------------------------

def _generate_duplex_llm_tts(
    model_dir: str,
    tts_model_dir: str,
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
    voice: str | None = None,
    instruct: str | None = None,
):
    """Interleaved LLM + TTS generation for duplex voice mode.

    Streams ``token`` events (text) and ``audio_chunk`` events (WAV audio)
    through the same event queue / WebSocket connection.
    """
    import mlx.core as mx

    def _send(event: dict):
        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop)

    try:
        # ── Load both models ──────────────────────────────────────────
        _send({"type": "status", "message": "Loading models..."})

        model, tokenizer = _get_or_load_mlx_model(model_dir)
        tts_model = _get_or_load_tts_model(tts_model_dir)

        # TTS voice / instruct setup
        # Priority: explicit voice > explicit instruct > auto-detect speakers > instruct fallback
        tts_kwargs: dict = {"stream": True}
        if voice:
            tts_kwargs["voice"] = voice
            logger.info("Duplex TTS: using explicit voice=%s", voice)
        elif instruct:
            tts_kwargs["instruct"] = instruct
            logger.info("Duplex TTS: using explicit instruct")
        elif hasattr(tts_model, "get_supported_speakers"):
            speakers = tts_model.get_supported_speakers()
            if speakers:
                tts_kwargs["voice"] = list(speakers)[0]
                logger.info("Duplex TTS: auto-selected speaker=%s", tts_kwargs["voice"])
            elif _is_instruct_tts(tts_model_dir):
                tts_kwargs["instruct"] = "A natural, friendly voice with moderate pace"
                logger.info("Duplex TTS: auto-using default instruct")
        elif _is_instruct_tts(tts_model_dir):
            tts_kwargs["instruct"] = "A natural, friendly voice with moderate pace"
            logger.info("Duplex TTS: auto-using default instruct (no speakers)")

        # ── LLM setup ────────────────────────────────────────────────
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        from backend.api.chat_params import get_generation_params

        model_params = get_generation_params(model_dir)

        rep_processor = None
        if model_params.repetition_penalty > 1.0:
            rep_processor = make_repetition_penalty(
                model_params.repetition_penalty,
                context_size=model_params.repetition_context_size,
            )

        prompt_cache = make_prompt_cache(model)
        sampler = make_sampler(temp=temperature, top_k=top_k, top_p=top_p)

        prompt_text = _apply_chat_template(tokenizer, prompt, history, enable_thinking)

        if hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt_text)
        else:
            prompt_ids = tokenizer._tokenizer.encode(prompt_text)
        if not isinstance(prompt_ids, list):
            prompt_ids = list(prompt_ids)

        eos_token_ids = set(model_params.eos_token_ids)

        _send({"type": "status", "message": "Prefilling..."})

        # Prefill
        input_ids = mx.array([prompt_ids])
        t_start = time.time()
        logits = model(input_ids, cache=prompt_cache)
        mx.eval(logits)
        t_prefill = time.time() - t_start

        last_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]

        _send({"type": "status", "message": "Generating..."})

        # ── Decode loop with TTS interleaving ────────────────────────
        generated_ids: list[int] = []
        tts_buffer = ""
        all_audio_chunks: list[np.ndarray] = []
        sample_rate = 24000
        total_tts_tokens = 0
        t_decode_start = time.time()

        # Voice consistency: use the FIRST chunk's audio as a fixed anchor
        # reference so all subsequent TTS calls produce the same voice.
        # This prevents the "telephone game" drift where each chunk copies
        # the previous one and gradually loses the original characteristics.
        _anchor_audio: mx.array | None = None   # first chunk — never overwritten
        _anchor_text: str | None = None
        _REF_DURATION_S = 6  # seconds of reference audio to keep

        _REP_NGRAM = model_params.rep_ngram
        _REP_THRESHOLD = model_params.rep_threshold
        repetition_stopped = False

        # Thinking suppression: detect <think>/<​/think> tokens to skip
        # thinking content from TTS and frontend display
        think_start_id, think_end_id = _get_think_token_ids(tokenizer)
        in_thinking = False

        def _run_tts_for_text(text: str):
            """Run TTS on a sentence, accumulate all chunks, send as one piece."""
            nonlocal sample_rate, total_tts_tokens, _anchor_audio, _anchor_text
            sentence_chunks: list[np.ndarray] = []

            # Build per-call kwargs: always reference the anchor (first chunk)
            call_kwargs = dict(tts_kwargs)
            if _anchor_audio is not None and _anchor_text is not None:
                call_kwargs["ref_audio"] = _anchor_audio
                call_kwargs["ref_text"] = _anchor_text

            for result in tts_model.generate(text, **call_kwargs):
                if cancel_event.is_set():
                    return
                sample_rate = result.sample_rate
                total_tts_tokens += result.token_count
                audio_np = np.array(result.audio, dtype=np.float32)
                sentence_chunks.append(audio_np)
            # Send entire sentence as a single audio chunk — avoids choppy playback
            if sentence_chunks:
                sentence_audio = np.concatenate(sentence_chunks) if len(sentence_chunks) > 1 else sentence_chunks[0]
                all_audio_chunks.append(sentence_audio)

                # Save the FIRST chunk as anchor — all future chunks reference it
                if _anchor_audio is None:
                    max_ref_samples = sample_rate * _REF_DURATION_S
                    ref = sentence_audio[-max_ref_samples:] if len(sentence_audio) > max_ref_samples else sentence_audio
                    _anchor_audio = mx.array(ref)
                    _anchor_text = text

                chunk_b64 = _audio_to_wav_b64(sentence_audio, sample_rate)
                _send({
                    "type": "audio_chunk",
                    "audio_b64": chunk_b64,
                    "sample_rate": sample_rate,
                })
            # Free TTS intermediate tensors before resuming LLM
            try:
                mx.clear_cache()
            except (ImportError, RuntimeError):
                pass

        for step_idx in range(max_tokens):
            if cancel_event.is_set():
                _send({"type": "cancelled"})
                return

            # Apply repetition penalty
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

            # ── Thinking suppression ──
            # Track <think>/<​/think> boundaries — suppress thinking content
            # from both TTS input and frontend display
            if think_start_id is not None and token_id == think_start_id:
                in_thinking = True
                # Continue generating (LLM needs to finish thinking) but don't
                # forward tokens to TTS or frontend
            if think_end_id is not None and token_id == think_end_id:
                in_thinking = False
                # Skip the </think> token itself too — go to next token
                # (fall through to next-token forward below)

            if not in_thinking and token_id != think_start_id and token_id != think_end_id:
                # Repetition detection (only for visible tokens)
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
                        logger.info("Duplex: repetition at step %d, stopping", step_idx)
                        repetition_stopped = True
                        break

                # Send token event (text streaming continues in frontend)
                _send({"type": "token", "token": token_str, "token_id": token_id})

                # Accumulate for TTS
                tts_buffer += token_str

                # ── TTS interleave: sentence ready? ──
                sentences, tts_buffer = _extract_tts_chunks(tts_buffer)
                for sentence in sentences:
                    _run_tts_for_text(sentence)
                    if cancel_event.is_set():
                        _send({"type": "cancelled"})
                        return

            # Forward next token (LLM continues)
            next_input = mx.array([[token_id]])
            logits = model(next_input, cache=prompt_cache)
            mx.eval(logits)

            last_logits = logits[0, -1, :] if logits.ndim == 3 else logits[-1, :]

        # ── LLM done — TTS remaining buffer ─────────────────────────
        remaining = tts_buffer.strip()
        if remaining:
            _run_tts_for_text(remaining)

        # ── Build final response ─────────────────────────────────────
        t_total = time.time() - t_start
        t_decode = time.time() - t_decode_start
        tokens_per_sec = len(generated_ids) / t_decode if t_decode > 0 else 0

        # Trim repetition
        if repetition_stopped:
            n = _REP_NGRAM
            pattern = generated_ids[-n:]
            trim_to = len(generated_ids) - n
            while trim_to >= n and generated_ids[trim_to - n:trim_to] == pattern:
                trim_to -= n
            generated_ids = generated_ids[:trim_to + n]

        # Decode full text
        try:
            if hasattr(tokenizer, "_tokenizer"):
                full_text = tokenizer._tokenizer.decode(generated_ids)
            elif hasattr(tokenizer, "decode"):
                full_text = tokenizer.decode(generated_ids)
            else:
                full_text = "".join(_safe_decode(tokenizer, tid) for tid in generated_ids)
        except (TypeError, ValueError, AttributeError):
            full_text = "".join(_safe_decode(tokenizer, tid) for tid in generated_ids)

        # Strip any thinking blocks from final text
        full_text = _strip_thinking(full_text)

        # Concatenate all audio for replay
        audio_b64 = None
        audio_duration = 0.0
        if all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            audio_duration = len(full_audio) / sample_rate
            audio_b64 = _audio_to_wav_b64(full_audio, sample_rate)

        _send({
            "type": "complete",
            "full_text": full_text,
            "total_tokens": len(generated_ids),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "prefill_time": round(t_prefill, 3),
            "total_time": round(t_total, 3),
            "audio_b64": audio_b64,
            "sample_rate": sample_rate,
            "audio_duration": round(audio_duration, 2),
        })

    except Exception as exc:
        logger.exception("Duplex generation failed")
        user_msg, _ = map_error(exc)
        _send({"type": "error", "message": user_msg})
