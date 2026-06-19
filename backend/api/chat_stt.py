# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""STT (speech-to-text) transcription for chat endpoints."""

from __future__ import annotations

import json
import os
import time
import asyncio
import threading
from typing import Any

from backend.services.error_mapper import map_error
from backend.api.chat_loaders import _get_or_load_stt_model

import logging

logger = logging.getLogger(__name__)

# Temporary storage for uploaded audio files (file_id -> (path, timestamp))
_uploaded_audio: dict[str, tuple[str, float]] = {}
_uploaded_audio_lock = threading.Lock()
_UPLOAD_TTL_SECONDS = 3600  # 1 hour


def _cleanup_expired_uploads() -> None:
    """Remove expired uploaded audio files. Must be called with lock held."""
    now = time.time()
    expired = [k for k, (_, ts) in _uploaded_audio.items()
               if now - ts > _UPLOAD_TTL_SECONDS]
    for k in expired:
        path, _ = _uploaded_audio.pop(k)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _detect_audio_suffix(file_name: str, audio_data: bytes) -> str:
    """Detect audio file suffix from filename or magic bytes."""
    if file_name:
        ext = os.path.splitext(file_name)[1].lower()
        if ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".opus"):
            return ext

    # Detect from magic bytes
    if audio_data[:4] == b"RIFF":
        return ".wav"
    if audio_data[:3] == b"ID3" or audio_data[:2] == b"\xff\xfb":
        return ".mp3"
    if audio_data[:4] == b"fLaC":
        return ".flac"
    if audio_data[:4] == b"OggS":
        return ".ogg"
    if len(audio_data) >= 8 and audio_data[4:8] == b"ftyp":
        return ".m4a"
    if audio_data[:4] == b"\x1aE\xdf\xa3":
        return ".webm"

    return ".wav"  # fallback


_MINIAUDIO_SUPPORTED = {".wav", ".mp3", ".flac", ".ogg"}


def _ensure_wav(audio_path: str) -> str:
    """Convert audio to WAV if the format is not supported by miniaudio.

    Browser MediaRecorder typically produces webm/opus which miniaudio cannot
    decode. Uses ffmpeg to transcode to 16kHz mono WAV (optimal for STT).
    Returns the original path if already supported, otherwise a new WAV path
    (and deletes the original).
    """
    import subprocess as sp

    ext = os.path.splitext(audio_path)[1].lower()
    if ext in _MINIAUDIO_SUPPORTED:
        return audio_path

    wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
    try:
        sp.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True,
            timeout=30,
            check=True,
        )
    except (sp.CalledProcessError, FileNotFoundError, sp.TimeoutExpired) as exc:
        logger.warning("ffmpeg conversion failed for %s: %s", audio_path, exc)
        return audio_path  # fallback: let downstream report the error

    # Clean up original non-WAV file
    try:
        os.unlink(audio_path)
    except OSError:
        pass
    return wav_path


def _is_stt_model(model_dir: str) -> bool:
    """Check if model is an STT/ASR model by inspecting config.json."""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "").lower()
        return "asr" in model_type or model_type == "sensevoice"
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def _generate_streaming_stt(
    model_dir: str,
    audio_b64: str | None,
    file_name: str,
    language: str | None,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
    file_id: str | None = None,
):
    """STT transcription in a background thread — streams tokens via WebSocket.

    Supports streaming for Qwen3-ASR (stream=True), fallback to batch for SenseVoice.
    Audio source: either file_id (pre-uploaded) or audio_b64 (inline base64).
    """
    import base64
    import tempfile

    audio_path = None
    is_uploaded_file = False

    def _send(event: dict):
        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop)

    try:
        _send({"type": "status", "message": "Loading STT model..."})

        if file_id:
            # Use pre-uploaded file
            with _uploaded_audio_lock:
                entry = _uploaded_audio.pop(file_id, None)
                audio_path = entry[0] if entry else None
            if not audio_path or not os.path.isfile(audio_path):
                _send({"type": "error", "message": "Uploaded audio file not found or expired"})
                return
            is_uploaded_file = True
        elif audio_b64:
            audio_data = base64.b64decode(audio_b64)
            suffix = _detect_audio_suffix(file_name, audio_data)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_data)
                audio_path = tmp.name
        else:
            _send({"type": "error", "message": "No audio data"})
            return

        # Convert non-WAV formats (webm/opus from browser MediaRecorder) to WAV
        # miniaudio doesn't support webm/opus — ffmpeg handles the conversion
        audio_path = _ensure_wav(audio_path)

        stt_model = _get_or_load_stt_model(model_dir)

        _send({"type": "status", "message": "Transcribing..."})
        t_start = time.time()

        gen_kwargs: dict[str, Any] = {}
        if language and language.strip():
            gen_kwargs["language"] = language.strip()

        # Unified chunking: ALL models go through external chunking for long audio.
        # Each chunk (~120s) generates ~400 tokens, well under default max_tokens=8192.
        # This ensures bounded memory and supports infinite-length audio.
        import inspect
        from backend.core.audio_chunker import chunk_audio, needs_chunking

        sig = inspect.signature(stt_model.generate)
        supports_stream = "stream" in sig.parameters

        full_text = ""
        total_tokens = 0

        if needs_chunking(audio_path):
            chunks = chunk_audio(audio_path)
            if not chunks:
                chunks = None
        else:
            chunks = None

        if chunks:
            _send({"type": "status", "message": f"Splitting audio into {len(chunks)} segments..."})
            import tempfile
            import subprocess as sp
            for i, audio_chunk in enumerate(chunks):
                if cancel_event.is_set():
                    _send({"type": "cancelled"})
                    return
                _send({"type": "status", "message": f"Transcribing segment {i+1}/{len(chunks)} ({audio_chunk.start_time:.0f}s-{audio_chunk.end_time:.0f}s)..."})
                _chunk_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                chunk_path = _chunk_tmp.name
                _chunk_tmp.close()
                try:
                    sp.run([
                        "ffmpeg", "-y", "-ss", str(audio_chunk.start_time),
                        "-i", audio_path, "-t", str(audio_chunk.duration),
                        "-ar", "16000", "-ac", "1", chunk_path,
                    ], capture_output=True, timeout=60)

                    if supports_stream:
                        # Qwen3-ASR: stream tokens per chunk
                        for token_chunk in stt_model.generate(chunk_path, stream=True, **gen_kwargs):
                            if cancel_event.is_set():
                                _send({"type": "cancelled"})
                                return
                            token_text = token_chunk.text if hasattr(token_chunk, "text") else str(token_chunk)
                            if token_text:
                                _send({"type": "token", "token": token_text})
                                full_text += token_text
                            total_tokens += 1
                    else:
                        # SenseVoice: batch per chunk
                        result = stt_model.generate(chunk_path, **gen_kwargs)
                        chunk_text = result.text.strip()
                        if chunk_text:
                            if full_text:
                                chunk_text = " " + chunk_text
                            _send({"type": "token", "token": chunk_text})
                            full_text += chunk_text
                        total_tokens += result.generation_tokens or max(len(chunk_text.split()), 1)
                finally:
                    try:
                        os.unlink(chunk_path)
                    except OSError:
                        pass
                    # Release MLX GPU cache after each chunk — prevents
                    # memory accumulation across chunks (SenseVoice: 2.3GB/chunk)
                    try:
                        import mlx.core as mx
                        mx.clear_cache()
                    except (ImportError, RuntimeError):
                        pass
        else:
            # Short audio — process directly (single chunk, well under max_tokens)
            if supports_stream:
                for token_chunk in stt_model.generate(audio_path, stream=True, **gen_kwargs):
                    if cancel_event.is_set():
                        _send({"type": "cancelled"})
                        return
                    token_text = token_chunk.text if hasattr(token_chunk, "text") else str(token_chunk)
                    if token_text:
                        _send({"type": "token", "token": token_text})
                        full_text += token_text
                    total_tokens += 1
            else:
                result = stt_model.generate(audio_path, **gen_kwargs)
                full_text = result.text
                total_tokens = result.generation_tokens or max(len(full_text.split()), 1)

        t_total = time.time() - t_start
        tokens_per_sec = round(total_tokens / max(t_total, 0.001), 1)

        _send({
            "type": "complete",
            "full_text": full_text,
            "total_tokens": total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "total_time": round(t_total, 3),
        })

    except ImportError:
        _send({"type": "error", "message": "mlx-audio not installed. Run: pip install mlx-audio"})
    except Exception as exc:
        logger.exception("STT generation failed")
        user_msg, _ = map_error(exc)
        _send({"type": "error", "message": user_msg})
    finally:
        if audio_path:
            try:
                os.unlink(audio_path)
            except OSError:
                pass
