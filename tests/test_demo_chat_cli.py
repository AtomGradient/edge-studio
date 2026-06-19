# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Tests for the local demo chat CLI."""

from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest

from backend.cli import demo_chat
from backend.cli import main as cli_main
from backend.cli.models import CatalogResolution, LocalModel, ModelWhereReport


def test_studio_command_dispatches_local_server(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_studio_server(*, host: str | None = None, port: int | None = None) -> int:
        captured["host"] = host
        captured["port"] = port
        return 0

    monkeypatch.setattr(cli_main, "run_studio_server", fake_run_studio_server)

    exit_code = cli_main.main(["studio", "--host", "127.0.0.1", "--port", "18842"])

    assert exit_code == 0
    assert captured == {"host": "127.0.0.1", "port": 18842}


def test_demo_chat_requires_prompt_unless_interactive() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["demo", "chat"])

    assert exc.value.code == 2


def test_demo_chat_interactive_rejects_json() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main.main(["demo", "chat", "--interactive", "--json"])

    assert exc.value.code == 2


def test_demo_chat_interactive_dispatches_without_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(*, options: demo_chat.ChatRunOptions):
        captured["options"] = options
        return demo_chat.ChatInteractiveResult(
            ok=True,
            exit_code=0,
            session_id="edge-chat-session-test",
            turn_count=0,
            receipt_paths=[],
        )

    monkeypatch.setattr(cli_main, "run_demo_chat_interactive", fake_run)

    exit_code = cli_main.main(["demo", "chat", "--interactive", "--model", "qwen3.5-9b-4bit"])

    assert exit_code == 0
    options = captured["options"]
    assert isinstance(options, demo_chat.ChatRunOptions)
    assert options.interactive is True
    assert options.model_ref == "qwen3.5-9b-4bit"
    assert options.max_tokens is None


def test_demo_chat_resolves_max_tokens_from_generation_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "generation_config.json").write_text('{"max_new_tokens": 768}', encoding="utf-8")

    value = demo_chat._resolve_max_tokens(demo_chat.ChatRunOptions(max_tokens=None), model_dir)

    assert value == 768


def test_demo_chat_resolves_max_tokens_with_fallback_and_explicit_override(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fallback = demo_chat._resolve_max_tokens(demo_chat.ChatRunOptions(max_tokens=None), model_dir)
    explicit = demo_chat._resolve_max_tokens(demo_chat.ChatRunOptions(max_tokens=64), model_dir)

    assert fallback == demo_chat.DEFAULT_MAX_TOKENS
    assert explicit == 64


def test_demo_chat_resolves_max_tokens_from_model_params_cap(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"max_position_embeddings": 32768, "hidden_size": 4096, '
        '"num_hidden_layers": 32, "intermediate_size": 11008, "vocab_size": 151936}',
        encoding="utf-8",
    )

    value = demo_chat._resolve_max_tokens(demo_chat.ChatRunOptions(max_tokens=None), model_dir)

    assert value == demo_chat.MAX_CONFIGURED_TOKENS


def test_demo_chat_interactive_streams_tokens_and_writes_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "Qwen3.5-9B-4bit"
    model_dir.mkdir()
    receipts: list[dict] = []
    stream_calls: list[dict[str, object]] = []

    fake_where = ModelWhereReport(
        schema_version="edge.models.where.report.v1",
        status="ok",
        resolution=CatalogResolution(
            status="resolved",
            input="qwen3.5-9b-4bit",
            model_id="qwen3.5-9b-4bit",
            name="Qwen3.5-9B-4bit",
            download_hint="mlx-community/Qwen3.5-9B-4bit",
            category="llm",
            size_gb=5.0,
            catalog_source="test",
            catalog_version="test",
            matched_by="id",
            alternates=[],
        ),
        local_matches=[
            LocalModel(
                name="Qwen3.5-9B-4bit",
                path=str(model_dir),
                size_bytes=0,
                complete=True,
            )
        ],
        fetch_command=None,
    )

    monkeypatch.setattr(demo_chat, "where_model", lambda *_args, **_kwargs: fake_where)
    monkeypatch.setattr(demo_chat, "_run_mlx_sync", lambda fn, *args: fn(*args))
    monkeypatch.setattr(demo_chat, "_get_or_load_mlx_model", lambda _path: (object(), object()))
    monkeypatch.setattr(demo_chat, "directory_manifest_hash", lambda _path: {"sha256": "sha256:model"})

    def fake_streaming(
        model_ref: str,
        model_dir: str,
        prompt: str,
        history: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        enable_thinking: bool | None,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        cancel_event,
        **_kwargs,
    ) -> None:
        del cancel_event
        call_index = len(stream_calls) + 1
        answer = f"answer-{call_index}"
        stream_calls.append(
            {
                "model_ref": model_ref,
                "model_dir": model_dir,
                "prompt": prompt,
                "history": [dict(turn) for turn in history],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "enable_thinking": enable_thinking,
            }
        )
        for event in (
            {"type": "token", "token": "answer-"},
            {"type": "token", "token": str(call_index)},
            {"type": "complete", "full_text": answer, "total_tokens": 2, "total_time": 0.01},
        ):
            asyncio.run_coroutine_threadsafe(event_queue.put(event), loop).result(timeout=1.0)

    def fake_write(receipt: dict, *, run_id: str | None = None, path: Path | None = None) -> Path:
        receipts.append(dict(receipt))
        return tmp_path / (run_id or "receipt") / "chat_receipt.json"

    monkeypatch.setattr(demo_chat, "_generate_streaming", fake_streaming)
    monkeypatch.setattr(demo_chat, "write_chat_receipt", fake_write)

    output = io.StringIO()
    result = demo_chat.run_demo_chat_interactive(
        options=demo_chat.ChatRunOptions(model_ref="qwen3.5-9b-4bit", max_tokens=8, interactive=True),
        input_stream=io.StringIO("hello\nagain\n/exit\n"),
        output_stream=output,
    )

    assert result.ok is True
    assert result.exit_code == 0
    assert result.turn_count == 2
    assert len(result.receipt_paths) == 2
    assert [call["prompt"] for call in stream_calls] == ["hello", "again"]
    assert stream_calls[0]["history"] == []
    assert stream_calls[1]["history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer-1"},
    ]
    assert [call["max_tokens"] for call in stream_calls] == [8, 8]
    assert [call["enable_thinking"] for call in stream_calls] == [False, False]
    assert "assistant> answer-1" in output.getvalue()
    assert "assistant> answer-2" in output.getvalue()
    assert [receipt["turn_index"] for receipt in receipts] == [1, 2]
    assert [receipt["history_turn_count"] for receipt in receipts] == [1, 2]
    assert [receipt["answer_tokens"] for receipt in receipts] == [2, 2]
    assert all(receipt["raw_text_included"] is False for receipt in receipts)


def test_demo_chat_uses_vlm_streaming_for_vision_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "Qwen3.5-9B-4bit"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"vision_config": {}}', encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fail_llm(*_args, **_kwargs) -> None:
        raise AssertionError("LLM streaming should not be used for vision-config models")

    def fake_vlm_streaming(
        model_dir_arg: str,
        prompt: str,
        image_b64: str | None,
        history: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        cancel_event,
        **kwargs,
    ) -> None:
        del cancel_event
        calls.append(
            {
                "model_dir": model_dir_arg,
                "prompt": prompt,
                "image_b64": image_b64,
                "history": history,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )
        for event in (
            {"type": "token", "token": "ok"},
            {"type": "complete", "full_text": "ok", "total_tokens": 1, "total_time": 0.01},
        ):
            asyncio.run_coroutine_threadsafe(event_queue.put(event), loop).result(timeout=1.0)

    monkeypatch.setattr(demo_chat, "_generate_streaming", fail_llm)
    monkeypatch.setattr(demo_chat, "_generate_streaming_vlm", fake_vlm_streaming)

    answer = demo_chat._generate_streamed_answer(
        model_ref="qwen3.5-9b-4bit",
        model_path=model_dir,
        prompt="hello",
        history=[],
        max_tokens=8,
    )

    assert answer["text"] == "ok"
    assert calls == [
        {
            "model_dir": str(model_dir),
            "prompt": "hello",
            "image_b64": None,
            "history": [],
            "max_tokens": 8,
            "temperature": 0.8,
            "kwargs": {"enable_thinking": False},
        }
    ]
