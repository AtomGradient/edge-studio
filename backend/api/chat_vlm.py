# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""VLM (vision-language model) generation for chat endpoints."""

from __future__ import annotations

import os
import time
import asyncio
import threading

import numpy as np

from backend.services.error_mapper import map_error
from backend.api.chat_loaders import _get_or_load_vlm_model, _get_or_load_tts_model
from backend.api.chat_llm import _strip_thinking
from backend.api.chat_tts import _audio_to_wav_b64, _is_instruct_tts
from backend.api.chat_duplex import _extract_tts_chunks

import logging

logger = logging.getLogger(__name__)


def _generate_streaming_vlm(
    model_dir: str,
    prompt: str,
    image_b64: str | None,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
    enable_dsr: bool = False,
    dsr_budget: int | None = None,
    enable_thinking: bool | None = None,
    tts_model_dir: str | None = None,
    voice: str | None = None,
    instruct: str | None = None,
):
    """VLM generation (text-only or image+text) in a background thread.

    Uses mlx_vlm so vision tower weights are properly recognised.
    Supports streaming via mlx_vlm.stream_generate when available.
    """
    image_path = None

    def _send(event: dict):
        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop)

    try:
        _send({"type": "status", "message": "Loading vision model..."})

        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        vlm_model, vlm_processor = _get_or_load_vlm_model(model_dir)
        config = load_config(model_dir)

        # ── TTS duplex setup (if tts_model_dir provided) ──
        tts_model = None
        tts_kwargs: dict = {}
        all_audio_chunks: list[np.ndarray] = []
        tts_sample_rate = 24000
        total_tts_tokens = 0
        if tts_model_dir:
            tts_model = _get_or_load_tts_model(tts_model_dir)
            tts_kwargs = {"stream": True}
            # Priority: explicit voice > explicit instruct > auto-detect speakers > instruct fallback
            if voice:
                tts_kwargs["voice"] = voice
            elif instruct:
                tts_kwargs["instruct"] = instruct
            elif hasattr(tts_model, "get_supported_speakers"):
                speakers = tts_model.get_supported_speakers()
                if speakers:
                    tts_kwargs["voice"] = list(speakers)[0]
                elif _is_instruct_tts(tts_model_dir):
                    tts_kwargs["instruct"] = "A natural, friendly voice with moderate pace"
            elif _is_instruct_tts(tts_model_dir):
                tts_kwargs["instruct"] = "A natural, friendly voice with moderate pace"

        # Voice consistency: use the FIRST chunk's audio as a fixed anchor
        # reference so all subsequent TTS calls produce the same voice.
        # This prevents the "telephone game" drift where each chunk copies
        # the previous one and gradually loses the original characteristics.
        _anchor_audio = None  # mx.array | None
        _anchor_text: str | None = None
        _REF_DURATION_S = 6  # seconds of reference audio to keep

        def _run_tts_for_text(text: str):
            """Run TTS on a sentence, send as single audio chunk."""
            nonlocal tts_sample_rate, total_tts_tokens, _anchor_audio, _anchor_text
            if not tts_model:
                return
            import mlx.core as mx
            sentence_chunks: list[np.ndarray] = []

            # Build per-call kwargs: always reference the anchor (first chunk)
            call_kwargs = dict(tts_kwargs)
            if _anchor_audio is not None and _anchor_text is not None:
                call_kwargs["ref_audio"] = _anchor_audio
                call_kwargs["ref_text"] = _anchor_text

            for result in tts_model.generate(text, **call_kwargs):
                if cancel_event.is_set():
                    return
                tts_sample_rate = result.sample_rate
                total_tts_tokens += result.token_count
                audio_np = np.array(result.audio, dtype=np.float32)
                sentence_chunks.append(audio_np)
            if sentence_chunks:
                sentence_audio = np.concatenate(sentence_chunks) if len(sentence_chunks) > 1 else sentence_chunks[0]
                all_audio_chunks.append(sentence_audio)

                # Save the FIRST chunk as anchor — all future chunks reference it
                if _anchor_audio is None:
                    max_ref_samples = tts_sample_rate * _REF_DURATION_S
                    ref = sentence_audio[-max_ref_samples:] if len(sentence_audio) > max_ref_samples else sentence_audio
                    _anchor_audio = mx.array(ref)
                    _anchor_text = text

                chunk_b64 = _audio_to_wav_b64(sentence_audio, tts_sample_rate)
                _send({
                    "type": "audio_chunk",
                    "audio_b64": chunk_b64,
                    "sample_rate": tts_sample_rate,
                })
            try:
                mx.clear_cache()
            except (ImportError, RuntimeError):
                pass

        # Prepare image list
        images: list[str] = []
        num_images = 0
        if image_b64:
            import base64
            import tempfile
            img_data = base64.b64decode(image_b64)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_data)
                image_path = tmp.name
            images = [image_path]
            num_images = 1

        # Build formatted prompt
        # Unified path: tokenizer.apply_chat_template handles history, images,
        # and thinking in one call.  When images are present, use multimodal
        # content format [{"type":"image"}, {"type":"text","text":...}] so the
        # template inserts <|image_pad|> tokens correctly.
        thinking_kwargs = {}
        if enable_thinking is not None:
            thinking_kwargs['enable_thinking'] = enable_thinking

        try:
            tok = getattr(vlm_processor, 'tokenizer', vlm_processor)
            if not hasattr(tok, 'apply_chat_template'):
                raise AttributeError("tokenizer has no apply_chat_template")

            # Build user content: multimodal list when images, plain string otherwise
            if num_images > 0:
                user_content = [{"type": "image"}] * num_images + [{"type": "text", "text": prompt}]
            else:
                user_content = prompt

            msgs = (list(history) if history else []) + [{"role": "user", "content": user_content}]
            formatted_prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                **thinking_kwargs,
            )
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            logger.debug("VLM tokenizer chat template failed, using mlx_vlm fallback: %s", exc)
            formatted_prompt = apply_chat_template(
                vlm_processor, config, prompt, num_images=num_images
            )

        # Force-add empty thinking block if tokenizer didn't include it
        if enable_thinking is False and '<think>' not in formatted_prompt:
            formatted_prompt += '<think>\n\n</think>\n\n'

        logger.debug("VLM enable_thinking=%s, prompt tail: %r", enable_thinking,
                      formatted_prompt[-120:] if len(formatted_prompt) > 120 else formatted_prompt)

        _send({"type": "status", "message": "Generating..."})
        t_start = time.time()
        full_text = ""
        streamed = False
        generation_tokens = 0

        # Build DSR cache for VLM if requested
        vlm_dsr_cache = None
        if enable_dsr and dsr_budget:
            try:
                from backend.core.dsr_cache import build_dsr_config, make_prompt_cache

                vlm_dsr_cache = make_prompt_cache(
                    vlm_model.language_model,
                    dsr_config=build_dsr_config(dsr_budget),
                )
            except (ImportError, ValueError, RuntimeError, AttributeError) as exc:
                logger.debug("VLM DSR cache init failed, proceeding without: %s", exc)

        # Token-level thinking filter for VLM streaming
        # mlx-vlm stream_generate yields text chunks (not token IDs),
        # so we track thinking state via text markers.
        in_thinking = False
        tts_buffer = ""  # TTS sentence accumulation buffer

        # Try streaming generation (mlx_vlm >= 0.1.x)
        try:
            from mlx_vlm import stream_generate
            generation_tokens = 0
            vlm_kwargs = {"max_tokens": max_tokens, "temperature": temperature}
            if vlm_dsr_cache is not None:
                vlm_kwargs["prompt_cache"] = vlm_dsr_cache

            # Repetition detection for VLM streaming
            _REP_WINDOW = 50   # check last N chars for repeating pattern
            _REP_MIN_PAT = 10  # minimum pattern length to detect
            _REP_COUNT = 4     # stop after this many repeats

            # Buffer for text-level thinking detection
            _text_buf = ""

            for chunk in stream_generate(
                vlm_model,
                vlm_processor,
                formatted_prompt,
                images,
                **vlm_kwargs,
            ):
                if cancel_event.is_set():
                    _send({"type": "cancelled"})
                    return
                # chunk.text is detokenizer.last_segment — already incremental
                token_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
                if token_text:
                    full_text += token_text
                generation_tokens += 1

                # Text-level thinking filter for VLM (no token IDs available)
                should_send = True
                if enable_thinking is False:
                    _text_buf += token_text
                    if not in_thinking and '<think>' in _text_buf:
                        in_thinking = True
                        should_send = False
                        _text_buf = _text_buf[_text_buf.index('<think>') + 7:]
                    elif in_thinking and '</think>' in _text_buf:
                        in_thinking = False
                        should_send = False
                        _text_buf = _text_buf[_text_buf.index('</think>') + 8:]
                        # Send remaining text after </think> if any
                        remaining = _text_buf.lstrip('\n')
                        if remaining:
                            _send({"type": "token", "token": remaining})
                            tts_buffer += remaining
                        _text_buf = ""
                    elif in_thinking:
                        should_send = False
                        _text_buf = _text_buf[-20:]  # keep tail for marker detection

                if should_send:
                    _send({"type": "token", "token": token_text})
                    tts_buffer += token_text

                # ── TTS duplex interleave: sentence ready? ──
                if tts_model and not in_thinking:
                    sentences, tts_buffer = _extract_tts_chunks(tts_buffer)
                    for sentence in sentences:
                        _run_tts_for_text(sentence)
                        if cancel_event.is_set():
                            _send({"type": "cancelled"})
                            return

                # Check for text-level repetition
                if len(full_text) > _REP_WINDOW * 2:
                    tail = full_text[-_REP_WINDOW:]
                    for plen in range(_REP_MIN_PAT, _REP_WINDOW // _REP_COUNT + 1):
                        pattern = tail[-plen:]
                        repeats = 0
                        for i in range(1, _REP_COUNT + 1):
                            start = len(tail) - plen * (i + 1)
                            if start < 0:
                                break
                            if tail[start:start + plen] == pattern:
                                repeats += 1
                            else:
                                break
                        if repeats >= _REP_COUNT - 1:
                            logger.info("VLM repetition detected at token %d, stopping", generation_tokens)
                            cancel_event.set()
                            break
            streamed = True
        except (ImportError, AttributeError, TypeError):
            pass

        if not streamed:
            # Fallback: non-streaming generate (correct arg order: prompt, images)
            from mlx_vlm import generate as vlm_generate
            result = vlm_generate(
                vlm_model,
                vlm_processor,
                formatted_prompt,
                images,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            )
            full_text = result if isinstance(result, str) else str(result)

        # ── TTS remaining buffer ──
        remaining_tts = tts_buffer.strip()
        if tts_model and remaining_tts:
            _run_tts_for_text(remaining_tts)

        # Strip thinking blocks from full text (safety net)
        if enable_thinking is False:
            full_text = _strip_thinking(full_text)

        t_total = time.time() - t_start
        num_tokens = generation_tokens if streamed and generation_tokens > 0 else max(len(full_text.split()), 1)
        tokens_per_sec = round(num_tokens / max(t_total, 0.001), 1)

        # Build complete audio for replay
        audio_b64 = None
        audio_duration = 0.0
        if all_audio_chunks:
            full_audio = np.concatenate(all_audio_chunks)
            audio_duration = len(full_audio) / tts_sample_rate
            audio_b64 = _audio_to_wav_b64(full_audio, tts_sample_rate)

        _send({
            "type": "complete",
            "full_text": full_text,
            "total_tokens": num_tokens,
            "tokens_per_sec": tokens_per_sec,
            "prefill_time": 0,
            "total_time": round(t_total, 3),
            "audio_b64": audio_b64,
            "sample_rate": tts_sample_rate if audio_b64 else None,
            "audio_duration": round(audio_duration, 2) if audio_b64 else None,
        })

    except Exception as exc:
        logger.exception("VLM generation failed")
        user_msg, _ = map_error(exc)
        _send({"type": "error", "message": user_msg})

    finally:
        if image_path:
            try:
                os.unlink(image_path)
            except OSError:
                pass
