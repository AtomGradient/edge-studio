# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""GGUF chat via llama-server for chat endpoints."""

from __future__ import annotations

import json
import subprocess
import time
import asyncio
import threading

import logging

logger = logging.getLogger(__name__)


class LlamaServerManager:
    """Manage a llama-server process lifecycle for GGUF chat."""

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._gguf_path: str | None = None
        self._port: int = 8090
        self._lock = threading.Lock()

    def start(self, gguf_path: str, port: int = 8090) -> None:
        with self._lock:
            if self._process and self._process.poll() is None:
                if self._gguf_path == gguf_path:
                    return  # Already running with same model
                self._stop_unlocked()

            self._gguf_path = gguf_path
            self._port = port
            self._process = subprocess.Popen(
                ["llama-server", "-m", gguf_path,
                 "--port", str(port), "--host", "127.0.0.1",
                 "-ngl", "99",
                 "-c", "4096",       # limit context to 4K (default 262K is overkill for chat)
                 "--no-warmup"],      # skip warmup to speed up startup
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            # Wait for server to be ready (up to 60 seconds for large models)
            import httpx
            for i in range(120):
                # Check if process died
                if self._process.poll() is not None:
                    stderr = self._process.stdout.read().decode(errors="replace")[-500:] if self._process.stdout else ""
                    raise RuntimeError(f"llama-server exited with code {self._process.returncode}: {stderr}")
                time.sleep(0.5)
                try:
                    r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
                    if r.status_code == 200:
                        return
                except (httpx.HTTPError, OSError):
                    pass
            # Timeout — kill and report
            self._stop_unlocked()
            raise RuntimeError("llama-server did not become ready within 60 seconds")

    def _stop_unlocked(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._gguf_path = None

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def port(self) -> int:
        return self._port


_llama_server = LlamaServerManager()


def _is_gguf_model(model_dir: str) -> bool:
    """Check if model_dir points to or contains a GGUF model."""
    from pathlib import Path
    p = Path(model_dir)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return True
    if p.is_dir():
        return any(p.glob("*.gguf")) and not any(p.glob("*.safetensors"))
    return False


def _find_gguf_path(model_dir: str) -> str | None:
    """Find primary .gguf file."""
    from pathlib import Path
    p = Path(model_dir)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return str(p)
    if p.is_dir():
        files = sorted(p.glob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
        return str(files[0]) if files else None
    return None


def _generate_streaming_gguf(
    model_dir: str,
    prompt: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    cancel_event: threading.Event,
):
    """GGUF chat via llama-server's OpenAI-compatible API."""
    import httpx

    def _send(event: dict):
        asyncio.run_coroutine_threadsafe(event_queue.put(event), loop)

    try:
        gguf_path = _find_gguf_path(model_dir)
        if not gguf_path:
            _send({"type": "error", "message": "No .gguf file found"})
            return

        _send({"type": "status", "message": "Starting llama-server..."})
        try:
            _llama_server.start(gguf_path)
        except RuntimeError as e:
            err_msg = str(e)
            # Provide user-friendly message for common errors
            if "wrong shape" in err_msg or "check_tensor_dims" in err_msg:
                _send({"type": "error", "message": (
                    "GGUF tensor维度不匹配。"
                    "如果是Edge Studio旧版导出的GGUF，请到Export页面重新导出（已修复）。"
                    "如果是第三方GGUF，请尝试重新下载或转换。"
                )})
            elif "failed to load model" in err_msg:
                _send({"type": "error", "message": f"llama-server failed to load model: {err_msg[-300:]}"})
            else:
                _send({"type": "error", "message": f"llama-server error: {err_msg[-300:]}"})
            return

        _send({"type": "status", "message": "Generating..."})

        messages = list(history) + [{"role": "user", "content": prompt}]
        t_start = time.time()

        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{_llama_server.port}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
            },
            timeout=120,
        ) as resp:
            full_text = ""
            for line in resp.iter_lines():
                if cancel_event.is_set():
                    _send({"type": "cancelled"})
                    return
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        _send({"type": "token", "token": content})
                        full_text += content
                except json.JSONDecodeError:
                    continue

        t_total = time.time() - t_start
        # Rough token count
        num_tokens = max(len(full_text.split()), 1)
        tokens_per_sec = round(num_tokens / max(t_total, 0.001), 1)

        _send({
            "type": "complete",
            "full_text": full_text,
            "total_tokens": num_tokens,
            "tokens_per_sec": tokens_per_sec,
            "prefill_time": 0,
            "total_time": round(t_total, 3),
        })

    except Exception as exc:
        logger.exception("GGUF generation failed")
        _send({"type": "error", "message": str(exc)})
