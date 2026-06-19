# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Streaming chat endpoint — token-by-token generation via WebSocket."""

from __future__ import annotations

import json
import os
import time
import asyncio
import threading
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse

from backend.services.model_manager import manager
from backend.services.error_mapper import map_error
from backend.services.mlx_runtime_gate import mlx_runtime_gate
from backend.services.mlx_worker import submit_mlx_task

from backend.api.chat_loaders import _get_or_load_stt_model
from backend.api.chat_llm import _has_vision_config, _generate_streaming
from backend.api.chat_vlm import _generate_streaming_vlm
from backend.api.chat_tts import _is_tts_model, _get_tts_voices, _is_instruct_tts, _generate_streaming_tts
from backend.api.chat_stt import (
    _uploaded_audio, _uploaded_audio_lock, _cleanup_expired_uploads,
    _is_stt_model, _detect_audio_suffix, _generate_streaming_stt,
)
from backend.api.chat_gguf import _is_gguf_model, _generate_streaming_gguf
from backend.api.chat_duplex import _generate_duplex_llm_tts
from backend.api.chat_params import get_generation_params

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

def _run_chat_generation_with_gate(
    gen_fn,
    gen_args: tuple,
    gen_kwargs: dict,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run one chat generation task under the shared MLX runtime gate."""
    with mlx_runtime_gate("chat.generate"):
        try:
            gen_fn(*gen_args, **gen_kwargs)
        except Exception as exc:
            logger.exception("Generation failed")
            user_msg, _ = map_error(exc)
            asyncio.run_coroutine_threadsafe(
                event_queue.put({"type": "error", "message": user_msg}), loop
            )


@router.get("/api/chat/{model_id}/params", response_model=dict[str, Any])
def get_chat_params(model_id: str) -> dict[str, Any]:
    """Return model-aware generation parameter defaults."""
    loaded = manager.get_model(model_id)
    if not loaded:
        return {"error": "Model not loaded"}
    from dataclasses import asdict
    params = get_generation_params(loaded.model_dir)
    return asdict(params)


@router.get("/api/chat/{model_id}/tts-voices", response_model=dict[str, Any])
def get_tts_voices(model_id: str) -> dict[str, Any]:
    """Return available TTS voice names for a model."""
    loaded = manager.get_model(model_id)
    if not loaded:
        return {"voices": []}
    if not _is_tts_model(loaded.model_dir):
        return {"voices": []}
    try:
        voices = _get_tts_voices(loaded.model_dir)
        instruct_mode = _is_instruct_tts(loaded.model_dir) if not voices else False
    except Exception as exc:
        logger.warning("Failed to get TTS voices for %s: %s", model_id, exc)
        return {"voices": [], "instruct_mode": False, "error": str(exc)}
    return {"voices": voices, "instruct_mode": instruct_mode}


@router.post("/api/chat/upload-audio", response_model=dict[str, str])
async def upload_audio(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload audio file for STT transcription.

    Returns {"file_id": "...", "file_name": "..."} to use with WebSocket STT.
    Supports large files without base64 overhead.
    """
    import uuid
    import tempfile

    file_id = uuid.uuid4().hex
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = os.path.join(tempfile.gettempdir(), f"stt_upload_{file_id}{suffix}")

    content = await file.read()
    # Also detect from magic bytes if extension is generic
    if suffix in (".bin", ".dat", ""):
        suffix = _detect_audio_suffix(file.filename or "", content)
        tmp_path = os.path.join(tempfile.gettempdir(), f"stt_upload_{file_id}{suffix}")

    with open(tmp_path, "wb") as f:
        f.write(content)

    with _uploaded_audio_lock:
        _cleanup_expired_uploads()
        _uploaded_audio[file_id] = (tmp_path, time.time())

    return {"file_id": file_id, "file_name": file.filename or f"upload{suffix}"}


@router.post("/api/chat/{model_id}/transcribe", response_model=dict[str, Any])
async def transcribe_audio(model_id: str, request: Request) -> dict[str, Any]:
    """Transcribe audio using STT model.

    Accepts JSON: {"audio_b64": "<base64-encoded audio>", "language": "auto"}
    Returns: {"text": "...", "segments": [...], "language": "...", "total_time": ..., ...}
    """
    loaded = manager.get_model(model_id)
    if not loaded:
        return JSONResponse(status_code=404, content={"error": "Model not loaded"})

    if not _is_stt_model(loaded.model_dir):
        return JSONResponse(status_code=400, content={"error": "Model is not an STT model"})

    body = await request.json()
    audio_b64 = body.get("audio_b64", "")
    language = body.get("language", None)
    file_name = body.get("file_name", "")

    if not audio_b64:
        return JSONResponse(status_code=400, content={"error": "No audio data"})

    import base64
    import tempfile

    # Decode audio to temp file — preserve original extension for codec detection
    audio_data = base64.b64decode(audio_b64)
    suffix = _detect_audio_suffix(file_name, audio_data)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        audio_path = tmp.name

    try:
        with mlx_runtime_gate("chat.transcribe"):
            stt_model = _get_or_load_stt_model(loaded.model_dir)

            gen_kwargs: dict[str, Any] = {}
            if language and language.strip():
                gen_kwargs["language"] = language.strip()

            result = stt_model.generate(audio_path, **gen_kwargs)

        return {
            "text": result.text,
            "segments": result.segments,
            "language": result.language,
            "total_time": result.total_time,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "total_tokens": result.total_tokens,
            "generation_tps": result.generation_tps,
        }
    except ImportError:
        return JSONResponse(status_code=500, content={"error": "mlx-audio not installed. Run: pip install mlx-audio"})
    except Exception as exc:
        logger.exception("Transcribe failed")
        user_msg, _ = map_error(exc)
        return JSONResponse(status_code=500, content={"error": user_msg})
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


@router.websocket("/ws/chat/{model_id}")
async def chat_stream(websocket: WebSocket, model_id: str):
    """Streaming chat WebSocket.

    Client sends: {"prompt": "...", "max_tokens": 2048, "temperature": 0.7, ...}
    Server streams: {"type": "token/status/complete/error", ...}
    Client can send: {"type": "cancel"} to stop generation.

    Architecture: generation runs on a global ThreadPoolExecutor (max_workers=1).
    Pool threads persist for the process lifetime — this prevents GIL crashes caused
    by MLX's TLS destructors running after Python thread state is destroyed.
    Serial execution (1 worker) ensures Metal operations never overlap.
    """
    loaded = manager.get_model(model_id)
    if not loaded:
        await websocket.close(code=4004, reason="Model not loaded")
        return

    # Detect model type once per connection
    is_stt = _is_stt_model(loaded.model_dir)
    is_tts = not is_stt and _is_tts_model(loaded.model_dir)
    is_vlm = not is_tts and not is_stt and _has_vision_config(loaded.model_dir)
    is_gguf = not is_tts and not is_vlm and not is_stt and _is_gguf_model(loaded.model_dir)

    # Detect thinking support once per connection (for default disable)
    # Note: Qwen3.5 is a VLM but supports thinking — check both LLM and VLM
    is_llm = not is_stt and not is_tts and not is_vlm and not is_gguf
    supports_thinking = False
    if is_llm or is_vlm:
        try:
            from backend.core.universal_tracer import detect_thinking_support
            supports_thinking = detect_thinking_support(loaded.model_dir)
        except Exception:
            pass

    # Model-aware parameter defaults
    model_params = get_generation_params(loaded.model_dir)

    await websocket.accept()

    # ── Submit generation tasks to the global MLX worker pool ──
    # Pool threads persist for the process lifetime → no TLS destructor crash.
    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Track the current generation future and cancel event
    current_future = None
    current_cancel: threading.Event | None = None

    try:
        while True:
            # Wait for a chat message from client
            raw = await websocket.receive_text()
            if len(raw) > 10 * 1024 * 1024:  # 10 MB limit
                await websocket.send_text(json.dumps({"type": "error", "message": "Message too large"}))
                continue
            data = json.loads(raw)

            # Handle cancel
            if data.get("type") == "cancel":
                continue

            prompt = data.get("prompt", "")
            audio_b64_ws = data.get("audio_b64", None)

            # STT requires audio_b64; others require prompt
            if not is_stt and not prompt.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Empty prompt",
                }))
                continue
            if is_stt and not audio_b64_ws and not data.get("file_id"):
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No audio data",
                }))
                continue

            history = data.get("history", [])
            # Model-aware defaults; client can request fewer tokens but not exceed ceiling
            max_tokens = max(1, min(
                int(data.get("max_tokens", model_params.max_tokens)),
                model_params.max_tokens,  # safety ceiling
            ))
            # Temperature: use model default if client doesn't specify
            temperature = max(0.0, min(
                float(data.get("temperature", model_params.temperature)),
                2.0,
            ))
            top_k_val = max(1, min(
                int(data.get("top_k", model_params.top_k)),
                1000,
            ))
            top_p_val = max(0.0, min(
                float(data.get("top_p", model_params.top_p)),
                1.0,
            ))
            enable_thinking = data.get("enable_thinking", None)
            # Default: disable thinking for models that support it (Qwen3.5, QwQ, etc.)
            if enable_thinking is None and supports_thinking:
                enable_thinking = False
            image_b64 = data.get("image_b64", None)
            enable_dsr = data.get("enable_dsr", False)
            dsr_budget = data.get("dsr_budget", None)
            cancel_event = threading.Event()
            current_cancel = cancel_event  # expose to finally for cleanup

            voice = data.get("voice", None)
            instruct = data.get("instruct", None)
            stt_file_name = data.get("file_name", "")
            stt_language = data.get("language", None)
            stt_file_id = data.get("file_id", None)

            # Build generation request for the worker thread
            if is_stt:
                gen_fn = _generate_streaming_stt
                gen_args = (loaded.model_dir, audio_b64_ws, stt_file_name, stt_language,
                            event_queue, loop, cancel_event)
                gen_kwargs = {"file_id": stt_file_id}
            elif is_tts:
                gen_fn = _generate_streaming_tts
                gen_args = (loaded.model_dir, prompt, voice, event_queue, loop, cancel_event)
                gen_kwargs = {"instruct": instruct}
            elif is_gguf:
                gen_fn = _generate_streaming_gguf
                gen_args = (loaded.model_dir, prompt, history, max_tokens, temperature,
                            event_queue, loop, cancel_event)
                gen_kwargs = {}
            elif is_vlm:
                gen_fn = _generate_streaming_vlm
                gen_args = (loaded.model_dir, prompt, image_b64, history, max_tokens,
                            temperature, event_queue, loop, cancel_event)
                # Duplex TTS interleaving for VLM (e.g. Qwen3.5 has vision_config but catalog says LLM)
                tts_model_id_vlm = data.get("tts_model_id", None)
                tts_loaded_vlm = manager.get_model(tts_model_id_vlm) if tts_model_id_vlm else None
                gen_kwargs = {"enable_dsr": enable_dsr, "dsr_budget": dsr_budget,
                              "enable_thinking": enable_thinking,
                              "tts_model_dir": tts_loaded_vlm.model_dir if tts_loaded_vlm else None,
                              "voice": voice, "instruct": instruct}
            else:
                # Duplex mode: interleave LLM + TTS when tts_model_id is provided
                tts_model_id = data.get("tts_model_id", None)
                tts_loaded = manager.get_model(tts_model_id) if tts_model_id else None
                if tts_loaded:
                    gen_fn = _generate_duplex_llm_tts
                    gen_args = (loaded.model_dir, tts_loaded.model_dir,
                                prompt, history, max_tokens, temperature,
                                top_k_val, top_p_val, enable_thinking,
                                event_queue, loop, cancel_event)
                    gen_kwargs = {"voice": voice, "instruct": instruct}
                else:
                    gen_fn = _generate_streaming
                    gen_args = (loaded.model_id, loaded.model_dir, prompt, history, max_tokens, temperature,
                                top_k_val, top_p_val, enable_thinking,
                                event_queue, loop, cancel_event)
                    gen_kwargs = {"enable_dsr": enable_dsr, "dsr_budget": dsr_budget}

            # Dispatch to pool — the pool thread persists (no TLS crash).
            current_future = submit_mlx_task(
                _run_chat_generation_with_gate,
                gen_fn,
                gen_args,
                gen_kwargs,
                event_queue,
                loop,
            )

            # Stream events to client
            done = False
            while not done:
                try:
                    # Check for client cancel messages
                    try:
                        client_msg = await asyncio.wait_for(
                            websocket.receive_text(), timeout=0.01,
                        )
                        client_data = json.loads(client_msg)
                        if client_data.get("type") == "cancel":
                            cancel_event.set()
                    except asyncio.TimeoutError:
                        pass

                    # Get next event from worker
                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        await websocket.send_text(json.dumps(event))
                        if event.get("type") in ("complete", "error", "cancelled"):
                            done = True
                    except asyncio.TimeoutError:
                        if current_future.done():
                            done = True
                except (WebSocketDisconnect, RuntimeError):
                    cancel_event.set()
                    done = True

    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.exception("WebSocket chat session error for model %s", model_id)
    finally:
        # Cancel any in-progress generation. The pool thread stays alive
        # (no TLS destructor crash) — it just stops the current task.
        if current_cancel is not None:
            current_cancel.set()
        # Wait briefly for the generation to finish (cancel is fast)
        if current_future is not None and not current_future.done():
            try:
                current_future.result(timeout=5.0)
            except Exception:
                pass


async def _neural_imprint_chat_stream(websocket: WebSocket, model_id: str):
    """Streaming chat endpoint with Neural Imprint active."""

    loaded = manager.get_model(model_id)
    if not loaded:
        await websocket.close(code=4004, reason="Model not loaded")
        return
    if loaded.category not in {"llm", "vlm"}:
        await websocket.close(code=4003, reason="Neural Imprint chat requires an LLM/VLM model")
        return

    await websocket.accept()
    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    model_params = get_generation_params(loaded.model_dir)
    supports_thinking = False
    try:
        from backend.core.universal_tracer import detect_thinking_support
        supports_thinking = detect_thinking_support(loaded.model_dir)
    except Exception:
        pass

    current_future = None
    current_cancel: threading.Event | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw) > 10 * 1024 * 1024:
                await websocket.send_text(json.dumps({"type": "error", "message": "Message too large"}))
                continue
            data = json.loads(raw)
            if data.get("type") == "cancel":
                continue

            prompt = data.get("prompt", "")
            if not prompt.strip():
                await websocket.send_text(json.dumps({"type": "error", "message": "Empty prompt"}))
                continue

            history = data.get("history", [])
            max_tokens = max(1, min(
                int(data.get("max_tokens", model_params.max_tokens)),
                model_params.max_tokens,
            ))
            temperature = max(0.0, min(float(data.get("temperature", model_params.temperature)), 2.0))
            top_k_val = max(1, min(int(data.get("top_k", model_params.top_k)), 1000))
            top_p_val = max(0.0, min(float(data.get("top_p", model_params.top_p)), 1.0))
            enable_thinking = data.get("enable_thinking", None)
            if enable_thinking is None and supports_thinking:
                enable_thinking = False

            cancel_event = threading.Event()
            current_cancel = cancel_event
            current_future = submit_mlx_task(
                _run_chat_generation_with_gate,
                _generate_streaming,
                (
                    loaded.model_id,
                    loaded.model_dir,
                    prompt,
                    history,
                    max_tokens,
                    temperature,
                    top_k_val,
                    top_p_val,
                    enable_thinking,
                    event_queue,
                    loop,
                    cancel_event,
                ),
                {"use_neural_imprint": True},
                event_queue,
                loop,
            )

            done = False
            while not done:
                try:
                    try:
                        client_msg = await asyncio.wait_for(
                            websocket.receive_text(), timeout=0.01,
                        )
                        client_data = json.loads(client_msg)
                        if client_data.get("type") == "cancel":
                            cancel_event.set()
                    except asyncio.TimeoutError:
                        pass

                    try:
                        event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                        await websocket.send_text(json.dumps(event))
                        if event.get("type") in ("complete", "error", "cancelled"):
                            done = True
                    except asyncio.TimeoutError:
                        if current_future.done():
                            done = True
                except (WebSocketDisconnect, RuntimeError):
                    cancel_event.set()
                    done = True
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        logger.exception("Neural Imprint chat session error for model %s", model_id)
    finally:
        if current_cancel is not None:
            current_cancel.set()
        if current_future is not None and not current_future.done():
            try:
                current_future.result(timeout=5.0)
            except Exception:
                pass


@router.websocket("/ws/neural-imprint-chat/{model_id}")
async def neural_imprint_chat_stream(websocket: WebSocket, model_id: str):
    await _neural_imprint_chat_stream(websocket, model_id)
