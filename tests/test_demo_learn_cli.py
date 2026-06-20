# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Tests for the local learning demo CLI output."""

from __future__ import annotations

from pathlib import Path

from backend.cli import demo_imprint, demo_learn
from backend.cli.demo_samples import resolve_learn_demo_sample


def test_default_learn_sample_is_finance_conservative_cashflow() -> None:
    assert demo_learn.DEFAULT_LEARN_SAMPLE_ID == "finance_conservative_cashflow_v1"


def test_finance_conservative_cashflow_sample_is_inspectable() -> None:
    sample = resolve_learn_demo_sample("finance_conservative_cashflow_v1")

    assert sample.sample_id == "finance_conservative_cashflow_v1"
    assert sample.peer_id == "edge-demo-learn-peer"
    assert sample.app_id == "com.atomgradient.edge.demo.finance"
    assert sample.question == "I have $800 left after bills this month. What should I do with it?"
    assert [record["record_id"] for record in sample.records] == [
        "finance-001",
        "finance-002",
        "finance-003",
    ]
    assert sample.records[0]["kind"] == "explicit_preference"
    assert "high-risk recommendations" in sample.records[0]["text"]
    assert "$800 left after bills" in sample.records[1]["text"]
    assert "unsupported return claims" in sample.records[2]["text"]
    assert sample.corrections[0]["target"] == {"profile_field": "financial_guidance_style"}
    overlay = sample.corrections[0]["correction"]["profile_overlay"]
    assert overlay["risk_style"] == "avoid high-risk or leveraged recommendations"
    assert overlay["priority"] == "cash-flow stability first"
    assert overlay["boundary"] == "no financial return claims without user-provided facts"
    assert [tool["name"] for tool in sample.tool_schema_export["tools"]] == [
        "sample_finance_facts_lookup",
        "sample_finance_cashflow_summary",
    ]
    policy = sample.expected_tool_policy
    assert policy["description"] == "Deterministic tool-use policy learned from this sample"
    assert [tool["name"] for tool in policy["tools_available"]] == [
        "sample_finance_facts_lookup",
        "sample_finance_cashflow_summary",
    ]
    assert "Do not call tools that require network access" in policy["negative_policy"]
    assert sample.expected_tool_policy_sha256.startswith("sha256:")


def test_finance_sample_tool_policy_matches_schema_tools() -> None:
    sample = resolve_learn_demo_sample("finance_conservative_cashflow_v1")

    schema_tools = {tool["name"] for tool in sample.tool_schema_export["tools"]}
    policy_tools = {tool["name"] for tool in sample.expected_tool_policy["tools_available"]}

    assert policy_tools == schema_tools
    assert sample.as_plan_summary()["expected_tool_policy_sha256"] == sample.expected_tool_policy_sha256


def test_finance_sample_dry_run_includes_raw_text_when_requested(monkeypatch) -> None:
    class _Resolution:
        def as_dict(self) -> dict[str, str]:
            return {"model_ref": "qwen3.5-9b-4bit"}

    class _Match:
        complete = True
        path = "/models/Qwen3.5-9B-4bit"
        size_bytes = 123

    class _Where:
        resolution = _Resolution()
        status = "ready"
        fetch_command = "edge models fetch qwen3.5-9b-4bit"
        local_matches = [_Match()]

    monkeypatch.setattr(demo_learn, "where_model", lambda *_args, **_kwargs: _Where())
    monkeypatch.setattr(demo_learn, "directory_manifest_hash", lambda _path: {"sha256": "sha256:model"})

    result = demo_learn.plan_learn_run(
        options=demo_learn.LearnRunOptions(
            sample_id="finance_conservative_cashflow_v1",
            model_ref="qwen3.5-9b-4bit",
            dry_run=True,
            include_text=True,
        )
    )

    assert isinstance(result, demo_learn.LearnPlanResult)
    assert result.ok is True
    assert result.plan["sample"]["sample_id"] == "finance_conservative_cashflow_v1"
    assert result.plan["sample"]["expected_tool_policy_sha256"].startswith("sha256:")
    assert result.plan["tool_learning"]["policy_kind"] == "deterministic_preview"
    assert result.plan["tool_learning"]["actual_tool_calls"] is False
    assert [tool["name"] for tool in result.plan["tool_learning"]["expected_tool_policy"]["tools_available"]] == [
        "sample_finance_facts_lookup",
        "sample_finance_cashflow_summary",
    ]
    assert result.plan["question"] == "I have $800 left after bills this month. What should I do with it?"
    sample_text = result.plan["sample_text"]
    assert sample_text["records"][0]["kind"] == "explicit_preference"
    assert "cash-flow-aware guidance" in sample_text["records"][0]["text"]
    assert sample_text["corrections"][0]["target"]["profile_field"] == "financial_guidance_style"
    assert sample_text["tool_schema_export"]["schema_version"] == "edgestudio.tool_schema_export.v1"
    assert sample_text["expected_tool_policy"]["tools_available"][1]["name"] == "sample_finance_cashflow_summary"


def test_write_learn_receipt_preserves_expected_tool_policy(tmp_path: Path) -> None:
    sample = resolve_learn_demo_sample("finance_conservative_cashflow_v1")
    receipt = {
        "schema_version": demo_learn.LEARN_RECEIPT_SCHEMA_VERSION,
        "run_id": "learn-test",
        "expected_tool_policy_sha256": sample.expected_tool_policy_sha256,
        "expected_tool_policy": sample.expected_tool_policy,
    }

    path = demo_learn.write_learn_receipt(receipt, path=tmp_path / "learn_receipt.json")

    text = path.read_text(encoding="utf-8")
    assert "expected_tool_policy" in text
    assert "sample_finance_cashflow_summary" in text


def test_format_learn_run_prints_imprint_paths_and_next_command() -> None:
    result = demo_learn.LearnRunResult(
        ok=True,
        exit_code=0,
        report={
            "schema_version": demo_learn.LEARN_RUN_SCHEMA_VERSION,
            "status": "completed",
            "model": {"model_ref": "qwen3.5-9b-4bit"},
            "model_prepare": {"status": "skipped_existing"},
            "sample": {"sample_id": "finance_conservative_cashflow_v1"},
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

    assert "sample: finance_conservative_cashflow_v1" in output
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
