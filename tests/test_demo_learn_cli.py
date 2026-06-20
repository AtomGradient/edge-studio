# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Tests for the local learning demo CLI output."""

from __future__ import annotations

from pathlib import Path

from backend.cli import demo_imprint, demo_learn


def test_format_learn_run_prints_imprint_paths_and_next_command() -> None:
    result = demo_learn.LearnRunResult(
        ok=True,
        exit_code=0,
        report={
            "schema_version": demo_learn.LEARN_RUN_SCHEMA_VERSION,
            "status": "completed",
            "model": {"model_ref": "qwen3.5-9b-4bit"},
            "model_prepare": {"status": "skipped_existing"},
            "sample": {"sample_id": "synthetic_profile_correction_v1"},
            "state": {"root": "/state"},
            "generation": {
                "job_id": "job-1",
                "artifact_path": "/state/neural_imprint_full_cache.safetensors",
                "metadata_path": "/state/neural_imprint_metadata.json",
            },
            "comparison": {
                "before_answer_sha256": "sha256:before",
                "after_answer_sha256": "sha256:after",
                "answers_differ": True,
                "before_answer": "before text",
                "after_answer": "after text",
            },
            "receipt_path": "/state/learn_receipt.json",
            "raw_text_included": True,
        },
    )

    output = demo_learn.format_learn_run(result)

    assert "artifact: /state/neural_imprint_full_cache.safetensors" in output
    assert "metadata: /state/neural_imprint_metadata.json" in output
    assert "receipt: /state/learn_receipt.json" in output
    assert (
        'next: edge demo chat --model qwen3.5-9b-4bit --interactive '
        '--with-imprint "/state/learn_receipt.json"'
    ) in output
    assert "[Before]\nbefore text" in output
    assert "[After]\nafter text" in output


def test_imprint_generate_answer_reuses_streaming_wrapper(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_generate_streamed_answer(**kwargs):
        calls.append(dict(kwargs))
        return {
            "text": "streamed answer",
            "token_count": 3,
            "elapsed_seconds": 0.25,
        }

    monkeypatch.setattr(demo_imprint, "_generate_streamed_answer", fake_generate_streamed_answer)

    result = demo_imprint._generate_answer(
        model_id="model-id",
        model_path=Path("/models/qwen"),
        prompt="hello",
        max_tokens=12,
        use_neural_imprint=True,
    )

    assert result["text"] == "streamed answer"
    assert result["token_count"] == 3
    assert result["elapsed_seconds"] == 0.25
    assert calls == [
        {
            "model_id": "model-id",
            "model_path": Path("/models/qwen"),
            "prompt": "hello",
            "history": [],
            "max_tokens": 12,
            "use_neural_imprint": True,
        }
    ]
