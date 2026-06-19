# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""TTS (text-to-speech) generation for chat endpoints."""

from __future__ import annotations

import json
import os
import time
import asyncio
import threading

from backend.services.error_mapper import map_error
from backend.api.chat_loaders import _get_or_load_tts_model

import logging

logger = logging.getLogger(__name__)


def _is_tts_model(model_dir: str) -> bool:
    """Check if model is a TTS model by inspecting config files."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if "tts" in model_type.lower() or "talker_config" in cfg:
            return True
        # Kokoro-style TTS: has istftnet / n_mels but no model_type
        if "istftnet" in cfg or ("n_mels" in cfg and "style_dim" in cfg):
            return True
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    # Fallback: check configuration.json (Kokoro uses task="text-to-speech")
    try:
        with open(os.path.join(model_dir, "configuration.json")) as f:
            meta = json.load(f)
        if "speech" in meta.get("task", "").lower():
            return True
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return False


def _get_tts_voices(model_dir: str) -> list[str]:
    """Get available voice names for a TTS model."""
    try:
        model = _get_or_load_tts_model(model_dir)
        if hasattr(model, 'get_supported_speakers'):
            speakers = model.get_supported_speakers()
            if speakers:
                return list(speakers)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Failed to get TTS voices: %s", exc)
    return []


def _is_instruct_tts(model_dir: str) -> bool:
    """Check if TTS model uses 'instruct' param for voice description (VoiceDesign models)."""
    try:
        model = _get_or_load_tts_model(model_dir)
        import inspect
        sig = inspect.signature(model.generate)
        return "instruct" in sig.parameters
    except Exception:
        return False


def _audio_to_wav_b64(audio_np, sample_rate: int) -> str:
    """Convert float32 numpy audio to base64-encoded WAV string."""
    import base64
    import io
    import struct

    # Clamp to [-1, 1] and convert to int16
    import numpy as np
    audio_clamped = np.clip(audio_np.flatten(), -1.0, 1.0)
    audio_int16 = (audio_clamped * 32767).astype(np.int16)

    # Write WAV manually (avoid scipy dependency)
    buf = io.BytesIO()
    num_samples = len(audio_int16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    # WAV header
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))  # chunk size
    buf.write(struct.pack('<H', 1))   # PCM format
    buf.write(struct.pack('<H', 1))   # mono
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))  # byte rate
    buf.write(struct.pack('<H', 2))   # block align
    buf.write(struct.pack('<H', 16))  # bits per sample
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(audio_int16.tobytes())

    return base64.b64encode(buf.getvalue()).decode('ascii')


def _generate_streaming_tts(
    model_dir: str,
    prompt: str,
    voice: str | None,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
    instruct: str | None = None,
):
    """TTS generation in a background thread — streams audio chunks via WebSocket.

    Uses mlx-audio's GenerationResult which has:
      .audio (mx.array), .sample_rate, .is_streaming_chunk, .is_final_chunk,
      .token_count, .audio_duration, .processing_time_seconds
    """
    import numpy as np

    def _send(event: dict):
        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop)

    try:
        _send({"type": "status", "message": "Loading TTS model..."})

        tts_model = _get_or_load_tts_model(model_dir)

        _send({"type": "status", "message": "Generating speech..."})
        t_start = time.time()

        chunks_collected = []
        sample_rate = 24000
        total_tokens = 0

        gen_kwargs = {"stream": True}
        # Priority: explicit voice > explicit instruct > auto-detect speakers > instruct fallback
        if voice:
            gen_kwargs["voice"] = voice
            logger.info("TTS using explicit voice: %s", voice)
        elif instruct:
            gen_kwargs["instruct"] = instruct
            logger.info("TTS using explicit instruct: %s", instruct[:50])
        elif hasattr(tts_model, 'get_supported_speakers'):
            speakers = tts_model.get_supported_speakers()
            if speakers:
                first_voice = list(speakers)[0]
                gen_kwargs["voice"] = first_voice
                logger.info("TTS auto-selected voice: %s", first_voice)
            elif _is_instruct_tts(model_dir):
                gen_kwargs["instruct"] = "A natural, friendly voice with moderate pace"
                logger.info("TTS auto-using default instruct (no speakers)")
        elif _is_instruct_tts(model_dir):
            gen_kwargs["instruct"] = "A natural, friendly voice with moderate pace"
            logger.info("TTS auto-using default instruct")
        for result in tts_model.generate(prompt, **gen_kwargs):
            if cancel_event.is_set():
                _send({"type": "cancelled"})
                return

            sample_rate = result.sample_rate
            total_tokens += result.token_count
            audio_np = np.array(result.audio, dtype=np.float32)
            chunks_collected.append(audio_np)

            # Send streaming chunk
            chunk_b64 = _audio_to_wav_b64(audio_np, sample_rate)
            _send({
                "type": "audio_chunk",
                "audio_b64": chunk_b64,
                "sample_rate": sample_rate,
            })

        t_total = time.time() - t_start

        if chunks_collected:
            full_audio = np.concatenate(chunks_collected)
            duration = len(full_audio) / sample_rate
            audio_b64 = _audio_to_wav_b64(full_audio, sample_rate)
            _send({
                "type": "complete",
                "audio_b64": audio_b64,
                "sample_rate": sample_rate,
                "duration": round(duration, 2),
                "total_time": round(t_total, 3),
                "full_text": f"[Audio: {duration:.1f}s]",
                "total_tokens": total_tokens,
                "tokens_per_sec": round(total_tokens / max(t_total, 0.001), 1),
            })
        else:
            _send({"type": "error", "message": "TTS generated empty audio"})

    except ImportError as exc:
        logger.exception("TTS import error")
        # Distinguish missing mlx-audio from missing model-specific deps (e.g. misaki for Kokoro)
        missing = getattr(exc, 'name', '') or str(exc)
        if 'mlx_audio' in missing:
            _send({"type": "error", "message": "mlx-audio not installed. Run: pip install mlx-audio"})
        else:
            _send({"type": "error", "message": f"Missing dependency: {missing}. Run: pip install {missing}"})
    except (ValueError, RuntimeError, TypeError) as exc:
        err_str = str(exc)
        logger.exception("TTS generation failed")
        # mlx_audio wraps ImportError as ValueError with "Missing dependency" message
        if "missing dependency" in err_str.lower():
            _send({"type": "error", "message": err_str})
        elif "not supported" in err_str.lower():
            _send({"type": "error", "message": f"{err_str}. Try: pip install -U mlx-audio"})
        else:
            user_msg, _ = map_error(exc)
            _send({"type": "error", "message": user_msg})
    except Exception as exc:
        logger.exception("TTS generation failed")
        user_msg, _ = map_error(exc)
        _send({"type": "error", "message": user_msg})
