# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hash-only before/after observation envelopes for Learning Flywheel evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


LEARNING_FLYWHEEL_OBSERVATION_ENVELOPE_SCHEMA_VERSION = (
    "edgestudio.learning_flywheel.observation_envelope.v1"
)

ALLOWED_PHASES = {"before", "after"}
# Observation envelopes scan a wider key set than heldout manifests because
# runtime rows can carry answer text and tool-call text in addition to prompts.
RAW_TEXT_KEYS = {
    "answer",
    "assistant_response",
    "generated_text",
    "messages",
    "prompt",
    "question",
    "raw_text",
    "response",
    "selected_tools",
    "text",
    "tool_arguments",
    "tool_calls",
    "tool_name",
    "transcript",
    "user_text",
}
LEGACY_HINT_KEYS = {
    "expected_text",
    "expected_tool",
    "golden_answer",
}


def build_learning_flywheel_observation_envelope(
    *,
    observations: Iterable[Mapping[str, Any]],
    run_id: str,
    phase: str,
    evidence_scope: str,
    eval_prompt_set_hash: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a portable observation envelope without raw prompt/answer text."""

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_phase = _phase(phase)
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    normalized_prompt_set_hash = _required_sha256(
        eval_prompt_set_hash,
        "eval_prompt_set_hash",
    )
    refs, local_rows = _observation_refs(observations)
    observation_refs_sha256 = _sha256_json(refs)
    local_observations_sha256 = _sha256_jsonl(local_rows)
    manifest_without_hash = {
        "ok": True,
        "schema_version": LEARNING_FLYWHEEL_OBSERVATION_ENVELOPE_SCHEMA_VERSION,
        "status": "written",
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "phase": normalized_phase,
        "evidence_scope": normalized_scope,
        "eval_prompt_set_hash": normalized_prompt_set_hash,
        "writes_runtime_artifacts": False,
        "runs_device_harness": False,
        "invokes_host_model_judge": False,
        "raw_text_included": False,
        "legacy_expected_hint_fields_included": False,
        "observation_count": len(refs),
        "local_observations_sha256": local_observations_sha256,
        "observation_refs_sha256": observation_refs_sha256,
        "observation_refs": refs,
    }
    manifest = {
        **manifest_without_hash,
        "manifest_sha256": _sha256_json(manifest_without_hash),
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest


def validate_learning_flywheel_observation_envelope(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that an observation envelope is portable and hash-only."""

    errors: list[dict[str, str]] = []
    if (
        manifest.get("schema_version")
        != LEARNING_FLYWHEEL_OBSERVATION_ENVELOPE_SCHEMA_VERSION
    ):
        errors.append({"code": "schema_version_mismatch", "field": "schema_version"})
    if manifest.get("phase") not in ALLOWED_PHASES:
        errors.append({"code": "invalid_phase", "field": "phase"})
    if manifest.get("raw_text_included") is not False:
        errors.append({"code": "raw_text_included", "field": "raw_text_included"})
    if manifest.get("legacy_expected_hint_fields_included") is not False:
        errors.append(
            {
                "code": "legacy_expected_hint_fields_included",
                "field": "legacy_expected_hint_fields_included",
            }
        )
    for field in _raw_text_fields(manifest):
        errors.append({"code": "raw_text_field_present", "field": field})
    for field in (
        "eval_prompt_set_hash",
        "local_observations_sha256",
        "observation_refs_sha256",
        "manifest_sha256",
    ):
        if not _valid_sha256(manifest.get(field)):
            errors.append({"code": "invalid_sha256", "field": field})

    refs = manifest.get("observation_refs")
    if not isinstance(refs, list) or not refs:
        errors.append({"code": "missing_observation_refs", "field": "observation_refs"})
    else:
        seen: set[str] = set()
        for index, item in enumerate(refs):
            if not isinstance(item, Mapping):
                errors.append(
                    {"code": "invalid_observation_ref", "field": f"observation_refs[{index}]"}
                )
                continue
            case_id = _text(item.get("case_id"))
            if not case_id:
                errors.append(
                    {
                        "code": "missing_case_id",
                        "field": f"observation_refs[{index}].case_id",
                    }
                )
            elif case_id in seen:
                errors.append(
                    {
                        "code": "duplicate_case_id",
                        "field": f"observation_refs[{index}].case_id",
                    }
                )
            seen.add(case_id)
            if not _valid_sha256(item.get("prompt_sha256")):
                errors.append(
                    {
                        "code": "invalid_prompt_sha256",
                        "field": f"observation_refs[{index}].prompt_sha256",
                    }
                )
            answer_hash = item.get("answer_sha256")
            if answer_hash is not None and not _valid_sha256(answer_hash):
                errors.append(
                    {
                        "code": "invalid_answer_sha256",
                        "field": f"observation_refs[{index}].answer_sha256",
                    }
                )
    return {"ok": not errors, "errors": errors}


def _observation_refs(
    observations: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    refs: list[dict[str, Any]] = []
    local_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise ValueError(f"observations[{index}] must be an object")
        legacy_keys = sorted(key for key in LEGACY_HINT_KEYS if key in observation)
        if legacy_keys:
            raise ValueError(
                f"observations[{index}] contains removed legacy hint keys: {legacy_keys}"
            )
        case_id = _required_text(
            observation.get("case_id") or observation.get("id"),
            "case_id",
        )
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        prompt_hash = _prompt_hash(observation)
        answer_hash = _answer_hash(observation)
        runtime = _runtime_summary(observation)
        ref: dict[str, Any] = {
            "case_id": case_id,
            "prompt_sha256": prompt_hash,
            "completed": observation.get("completed") is True,
            "freeze_detected": observation.get("freeze_detected") is True,
            "oom": observation.get("oom") is True,
            "fact_tool_called": observation.get("fact_tool_called") is True,
            "tool_call_count": runtime["tool_call_count"],
            "selected_tool_count": runtime["selected_tool_count"],
        }
        if answer_hash is not None:
            ref["answer_sha256"] = answer_hash
        if isinstance(runtime.get("duration_ms"), int):
            ref["duration_ms"] = runtime["duration_ms"]
        if isinstance(runtime.get("max_memory_mb"), int | float):
            ref["max_memory_mb"] = runtime["max_memory_mb"]
        refs.append(ref)
        local_rows.append(_local_hash_row(observation, prompt_hash, answer_hash, runtime))
    if not refs:
        raise ValueError("observations must contain at least one observation")
    return (
        sorted(refs, key=lambda item: item["case_id"]),
        sorted(local_rows, key=lambda item: item["case_id"]),
    )


def _prompt_hash(observation: Mapping[str, Any]) -> str:
    prompt_hash = _text(observation.get("prompt_sha256"))
    if prompt_hash:
        return _required_sha256(prompt_hash, "prompt_sha256")
    prompt = _text(observation.get("prompt")) or _text(observation.get("text"))
    if not prompt:
        raise ValueError("prompt_sha256 or local prompt text is required")
    return _sha256_text(prompt)


def _answer_hash(observation: Mapping[str, Any]) -> str | None:
    answer_hash = _text(observation.get("answer_sha256"))
    if answer_hash:
        return _required_sha256(answer_hash, "answer_sha256")
    answer = (
        _text(observation.get("answer"))
        or _text(observation.get("response"))
        or _text(observation.get("generated_text"))
    )
    return _sha256_text(answer) if answer else None


def _runtime_summary(observation: Mapping[str, Any]) -> dict[str, Any]:
    selected_tools = _object_list(observation.get("selected_tools"))
    tool_calls = _object_list(observation.get("tool_calls"))
    summary: dict[str, Any] = {
        "selected_tool_count": len(selected_tools),
        "tool_call_count": len(tool_calls),
    }
    duration = observation.get("duration_ms")
    if isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0:
        summary["duration_ms"] = duration
    max_memory = observation.get("max_memory_mb")
    if (
        isinstance(max_memory, int | float)
        and not isinstance(max_memory, bool)
        and max_memory >= 0
    ):
        summary["max_memory_mb"] = max_memory
    return summary


def _local_hash_row(
    observation: Mapping[str, Any],
    prompt_hash: str,
    answer_hash: str | None,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": _required_text(
            observation.get("case_id") or observation.get("id"),
            "case_id",
        ),
        "prompt_sha256": prompt_hash,
        "answer_sha256": answer_hash,
        "local_prompt": _text(observation.get("prompt")) or _text(observation.get("text")),
        "local_answer": (
            _text(observation.get("answer"))
            or _text(observation.get("response"))
            or _text(observation.get("generated_text"))
        ),
        "selected_tools": _object_list(observation.get("selected_tools")),
        "tool_calls": _object_list(observation.get("tool_calls")),
        "completed": observation.get("completed") is True,
        "freeze_detected": observation.get("freeze_detected") is True,
        "oom": observation.get("oom") is True,
        "fact_tool_called": observation.get("fact_tool_called") is True,
        "runtime": dict(runtime),
    }
    error_hash = _text(observation.get("error_sha256"))
    if error_hash:
        row["error_sha256"] = _required_sha256(error_hash, "error_sha256")
    elif _text(observation.get("error")):
        row["error_sha256"] = _sha256_text(_text(observation.get("error")))
    return row


def _phase(value: str) -> str:
    normalized = _required_text(value, "phase")
    if normalized not in ALLOWED_PHASES:
        raise ValueError(f"phase must be one of {sorted(ALLOWED_PHASES)}")
    return normalized


def _raw_text_fields(payload: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in RAW_TEXT_KEYS:
                yield field
            yield from _raw_text_fields(value, field)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            field = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _raw_text_fields(item, field)


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _object_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, (str, Mapping))]


def _required_sha256(value: Any, field: str) -> str:
    if not _valid_sha256(value):
        raise ValueError(f"{field} must be a sha256 hash")
    raw = str(value)
    return raw if raw.startswith("sha256:") else "sha256:" + raw


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    raw = value[7:] if value.startswith("sha256:") else value
    return len(raw) == 64 and all(char in "0123456789abcdef" for char in raw.lower())


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _sha256_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    )
    return _sha256_text(payload + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
