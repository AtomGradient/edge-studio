# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-model hard-fact leakage review for generated training SFT.

This service is a narrow bridge from `train.jsonl` to
`host_model_assistant.review_hard_fact_leakage`. It intentionally stores and
returns only compact review summaries for orchestration metadata; raw sample text
and caller-supplied forbidden entities stay out of train/distribute metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from backend.services.host_model_assistant import (
    HOST_MODEL_PROVIDER,
    review_hard_fact_leakage,
)


TRAINING_LEAKAGE_REVIEW_SCHEMA_VERSION = "edgestudio.training_leakage_review.v0"
DEFAULT_REVIEW_BATCH_SIZE = int(
    os.environ.get("EDGE_TRAIN_HARD_FACT_REVIEW_BATCH_SIZE", "16")
)


def build_training_leakage_review_samples(
    sft_jsonl_path: Path | str,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Build host-model leakage review samples from an SFT JSONL file."""

    path = Path(sft_jsonl_path)
    samples: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if max_samples is not None and len(samples) >= max_samples:
            break
        text = raw_line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no} is not valid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_no} is {type(row).__name__}, not object")
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"line {line_no} missing messages list")

        normalized_messages: list[dict[str, str]] = []
        for idx, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"line {line_no} messages[{idx}] is "
                    f"{type(message).__name__}, not object"
                )
            role = str(message.get("role") or "").strip()
            content = message.get("content")
            if not role:
                raise ValueError(f"line {line_no} messages[{idx}] missing role")
            if not isinstance(content, str):
                content = json.dumps(
                    content,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=repr,
                )
            normalized_messages.append({"role": role, "content": content})

        row_fingerprint = _fingerprint(row)
        sample_id = _sample_id(row, line_no=line_no, row_fingerprint=row_fingerprint)
        samples.append(
            {
                "sample_id": sample_id,
                "messages": normalized_messages,
                "source": {
                    "kind": "training_sft_jsonl",
                    "line": line_no,
                    "file_name": path.name,
                    "sample_fingerprint": row_fingerprint,
                },
            }
        )
    return samples


def review_training_sft_hard_fact_leakage(
    sft_jsonl_path: Path | str,
    forbidden_entities: list[str],
    *,
    host_model_id: str | None = None,
    provider: str | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Review a generated SFT file through the host-model leakage contract."""

    try:
        samples = build_training_leakage_review_samples(
            sft_jsonl_path,
            max_samples=max_samples,
        )
    except (OSError, ValueError) as exc:
        return {
            "ok": False,
            "schema_version": TRAINING_LEAKAGE_REVIEW_SCHEMA_VERSION,
            "result": None,
            "error": {
                "code": "invalid_training_sft",
                "message": "Training SFT could not be converted to leakage review samples.",
                "retryable": False,
                "details": {"reason": str(exc)},
            },
            "audit": {
                "method": "review_training_sft_hard_fact_leakage",
                "status": "error",
                "input_summary": {
                    "file_name": Path(sft_jsonl_path).name,
                    "sample_count": None,
                    "sft_fingerprint": None,
                },
                "warnings": [],
            },
        }

    effective_provider = provider or HOST_MODEL_PROVIDER
    effective_batch_size = max(1, int(batch_size or DEFAULT_REVIEW_BATCH_SIZE))
    if len(samples) <= effective_batch_size:
        response = review_hard_fact_leakage(
            samples,
            forbidden_entities,
            host_model_id=host_model_id,
            provider=effective_provider,
        )
        if isinstance(response, dict) and response.get("ok") is True:
            return response
        if len(samples) <= 1:
            return response
        responses, failed_response = _split_sample_chunk_with_retry(
            samples=samples,
            forbidden_entities=forbidden_entities,
            host_model_id=host_model_id,
            provider=effective_provider,
        )
        if failed_response is not None:
            return _chunked_review_error(
                failed_response,
                chunk_index=0,
                chunk_count=1,
                total_sample_count=len(samples),
                provider=effective_provider,
                host_model_id=host_model_id,
            )
        return _combine_chunked_review_responses(
            responses,
            total_sample_count=len(samples),
            forbidden_entity_count=len(forbidden_entities),
            provider=effective_provider,
            host_model_id=host_model_id,
        )

    responses: list[dict[str, Any]] = []
    for start in range(0, len(samples), effective_batch_size):
        chunk = samples[start:start + effective_batch_size]
        chunk_responses, failed_response = _review_sample_chunk_with_retry(
            samples=chunk,
            forbidden_entities=forbidden_entities,
            host_model_id=host_model_id,
            provider=effective_provider,
        )
        if failed_response is not None:
            return _chunked_review_error(
                failed_response,
                chunk_index=len(responses),
                chunk_count=(len(samples) + effective_batch_size - 1)
                // effective_batch_size,
                total_sample_count=len(samples),
                provider=effective_provider,
                host_model_id=host_model_id,
            )
        responses.extend(chunk_responses)

    return _combine_chunked_review_responses(
        responses,
        total_sample_count=len(samples),
        forbidden_entity_count=len(forbidden_entities),
        provider=effective_provider,
        host_model_id=host_model_id,
    )


def prune_training_sft_by_leakage_review(
    sft_jsonl_path: Path | str,
    review_summary: dict[str, Any],
) -> dict[str, Any]:
    """Drop SFT rows the host-model review marked as hard-fact leakage.

    The input review is expected to be the raw-free summary produced by
    `summarize_training_leakage_review`; pruning is driven only by line numbers
    in `review_items_safe_refs`, not by raw text or matched entity values.
    """

    path = Path(sft_jsonl_path)
    leaked_lines = _leaked_review_line_numbers(review_summary)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    original_count = sum(1 for line in raw_lines if line.strip())
    if not leaked_lines:
        return {
            "schema_version": "edgestudio.training_leakage_prune.v0",
            "ok": True,
            "status": "no_leaked_rows",
            "original_count": original_count,
            "dropped_count": 0,
            "kept_count": original_count,
            "source_review_fingerprint": _optional_str(
                review_summary.get("response_fingerprint")
            ),
            "warnings": [],
        }

    kept_lines: list[str] = []
    dropped_count = 0
    for line_no, raw_line in enumerate(raw_lines, 1):
        if not raw_line.strip():
            kept_lines.append(raw_line)
            continue
        if line_no in leaked_lines:
            dropped_count += 1
            continue
        kept_lines.append(raw_line)

    tmp_path = path.with_name(f"{path.name}.pruned.tmp")
    text = "\n".join(kept_lines)
    tmp_path.write_text((text + "\n") if text else "", encoding="utf-8")
    tmp_path.replace(path)

    warnings: list[str] = []
    missing_line_count = len(leaked_lines) - dropped_count
    if missing_line_count > 0:
        warnings.append("leakage_prune_missing_line_refs")

    return {
        "schema_version": "edgestudio.training_leakage_prune.v0",
        "ok": True,
        "status": "pruned",
        "original_count": original_count,
        "dropped_count": dropped_count,
        "kept_count": original_count - dropped_count,
        "missing_line_ref_count": max(0, missing_line_count),
        "source_review_fingerprint": _optional_str(
            review_summary.get("response_fingerprint")
        ),
        "warnings": warnings,
    }


def _review_sample_chunk_with_retry(
    *,
    samples: list[dict[str, Any]],
    forbidden_entities: list[str],
    host_model_id: str | None,
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    response = review_hard_fact_leakage(
        samples,
        forbidden_entities,
        host_model_id=host_model_id,
        provider=provider,
    )
    if isinstance(response, dict) and response.get("ok") is True:
        return [response], None
    if len(samples) <= 1:
        return [], response if isinstance(response, dict) else {
            "ok": False,
            "error": {
                "code": "invalid_host_model_response",
                "message": "Host-model review did not return an envelope.",
                "retryable": True,
            },
        }
    return _split_sample_chunk_with_retry(
        samples=samples,
        forbidden_entities=forbidden_entities,
        host_model_id=host_model_id,
        provider=provider,
    )


def _split_sample_chunk_with_retry(
    *,
    samples: list[dict[str, Any]],
    forbidden_entities: list[str],
    host_model_id: str | None,
    provider: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:

    midpoint = max(1, len(samples) // 2)
    left_responses, left_error = _review_sample_chunk_with_retry(
        samples=samples[:midpoint],
        forbidden_entities=forbidden_entities,
        host_model_id=host_model_id,
        provider=provider,
    )
    if left_error is not None:
        return [], left_error
    right_responses, right_error = _review_sample_chunk_with_retry(
        samples=samples[midpoint:],
        forbidden_entities=forbidden_entities,
        host_model_id=host_model_id,
        provider=provider,
    )
    if right_error is not None:
        return [], right_error
    return [*left_responses, *right_responses], None


def _chunked_review_error(
    response: Any,
    *,
    chunk_index: int,
    chunk_count: int,
    total_sample_count: int,
    provider: str,
    host_model_id: str | None,
) -> dict[str, Any]:
    error = response.get("error") if isinstance(response, dict) else None
    code = _optional_str(error.get("code")) if isinstance(error, dict) else None
    message = _optional_str(error.get("message")) if isinstance(error, dict) else None
    return {
        "ok": False,
        "schema_version": TRAINING_LEAKAGE_REVIEW_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": "chunked_review_failed",
            "message": message or "A hard-fact leakage review chunk failed.",
            "retryable": bool(error.get("retryable")) if isinstance(error, dict) else True,
            "details": {
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "chunk_error_code": code,
                "response_fingerprint": _fingerprint(response),
            },
        },
        "audit": {
            "schema_version": "edgestudio.host_model.audit.v0",
            "method": "review_training_sft_hard_fact_leakage",
            "provider": provider,
            "host_model": {
                "enabled": provider == HOST_MODEL_PROVIDER,
                "model_id": None,
                "selected_model_id": host_model_id,
            },
            "status": "error",
            "input_summary": {
                "sample_count": total_sample_count,
                "chunk_count": chunk_count,
                "failed_chunk_index": chunk_index,
            },
            "warnings": ["chunked_hard_fact_review_failed"],
        },
    }


def _combine_chunked_review_responses(
    responses: list[dict[str, Any]],
    *,
    total_sample_count: int,
    forbidden_entity_count: int,
    provider: str,
    host_model_id: str | None,
) -> dict[str, Any]:
    review_items: list[dict[str, Any]] = []
    chunk_decisions: list[str] = []
    leakage_count = 0
    reviewed_count = 0
    unverifiable_count = 0
    chunk_fingerprints: list[str] = []
    host_model: dict[str, Any] = {
        "enabled": provider == HOST_MODEL_PROVIDER,
        "model_id": None,
        "selected_model_id": host_model_id,
    }
    warnings: list[str] = ["chunked_hard_fact_review"]

    for response in responses:
        chunk_fingerprints.append(_fingerprint(response))
        result = response.get("result") if isinstance(response, dict) else None
        if not isinstance(result, dict):
            unverifiable_count += 1
            continue
        chunk_unverifiable = False
        items = result.get("review_items")
        item_reviewed_count: int | None = None
        item_leakage_count: int | None = None
        if isinstance(items, list):
            chunk_items = [item for item in items if isinstance(item, dict)]
            review_items.extend(chunk_items)
            item_reviewed_count = len(chunk_items)
            item_leakage_count = sum(
                1 for item in chunk_items if item.get("leakage_detected") is True
            )
        else:
            chunk_unverifiable = True
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else None
        reviewed_raw = summary.get("reviewed_count") if isinstance(summary, dict) else None
        leakage_raw = summary.get("leakage_count") if isinstance(summary, dict) else None
        reviewed_value = _optional_nonnegative_int(reviewed_raw)
        leakage_value = _optional_nonnegative_int(leakage_raw)
        if reviewed_value is None or leakage_value is None:
            chunk_unverifiable = True
        if item_reviewed_count is not None and item_leakage_count is not None:
            reviewed_count += item_reviewed_count
            leakage_count += item_leakage_count
            if (
                reviewed_value is not None
                and reviewed_value != item_reviewed_count
            ) or (
                leakage_value is not None
                and leakage_value != item_leakage_count
            ):
                chunk_unverifiable = True
                if "chunked_hard_fact_review_count_mismatch" not in warnings:
                    warnings.append("chunked_hard_fact_review_count_mismatch")
        elif reviewed_value is not None and leakage_value is not None:
            reviewed_count += reviewed_value
            leakage_count += leakage_value
            chunk_unverifiable = True
        else:
            chunk_unverifiable = True
        decision = _optional_str(result.get("decision"))
        if decision in {"pass", "fail", "needs_human_review"}:
            chunk_decisions.append(decision)
        else:
            chunk_unverifiable = True
        if chunk_unverifiable:
            unverifiable_count += 1

        audit = response.get("audit") if isinstance(response.get("audit"), dict) else {}
        audit_host = audit.get("host_model") if isinstance(audit.get("host_model"), dict) else {}
        model_id = _optional_str(audit_host.get("model_id"))
        if model_id:
            host_model["model_id"] = model_id
        for warning in audit.get("warnings") or []:
            text = _optional_str(warning)
            if text and text not in warnings:
                warnings.append(text)

    if unverifiable_count and "chunked_hard_fact_review_unverifiable" not in warnings:
        warnings.append("chunked_hard_fact_review_unverifiable")

    if any(decision == "fail" for decision in chunk_decisions) or leakage_count:
        decision = "fail"
    elif (
        any(decision == "needs_human_review" for decision in chunk_decisions)
        or unverifiable_count
    ):
        decision = "needs_human_review"
    else:
        decision = "pass"

    return {
        "ok": True,
        "schema_version": "edgestudio.host_model.hard_fact_leakage_review.v0",
        "result": {
            "status": "host_model_reviewed",
            "decision": decision,
            "review_items": review_items,
            "summary": {
                "sample_count": total_sample_count,
                "reviewed_count": reviewed_count or len(review_items),
                "forbidden_entity_count": forbidden_entity_count,
                "leakage_count": leakage_count,
                "unverifiable_count": unverifiable_count,
                "chunk_count": len(responses),
            },
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.host_model.audit.v0",
            "assistant_version": "host_model_assistant.v0",
            "method": "review_hard_fact_leakage",
            "provider": provider,
            "host_model": host_model,
            "status": "ok",
            "input_summary": {
                "sample_count": total_sample_count,
                "chunk_count": len(responses),
                "forbidden_entity_count": forbidden_entity_count,
                "chunk_response_fingerprints": chunk_fingerprints,
            },
            "warnings": warnings,
            "log_target": "response.audit",
        },
    }


def summarize_training_leakage_review(response: dict[str, Any]) -> dict[str, Any]:
    """Return a raw-free summary safe for task results and adapter metadata."""

    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, dict):
        result = {}
    summary = result.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    return {
        "schema_version": TRAINING_LEAKAGE_REVIEW_SCHEMA_VERSION,
        "ok": response.get("ok") is True,
        "host_response_schema_version": response.get("schema_version"),
        "status": _optional_str(result.get("status")),
        "decision": _optional_str(result.get("decision")),
        "summary": _compact_dict(summary),
        "review_items_safe_refs": _review_items_safe_refs(
            result.get("review_items")
        ),
        "error": _safe_error(response.get("error")),
        "audit": _safe_audit(response.get("audit")),
        "response_fingerprint": _fingerprint(response),
    }


def is_training_leakage_review_pass(review: dict[str, Any] | None) -> bool:
    """Only a completed host-model pass unlocks opt-in training."""

    if not isinstance(review, dict) or review.get("ok") is not True:
        return False
    result = review.get("result")
    if isinstance(result, dict):
        status = result.get("status")
        decision = result.get("decision")
    else:
        status = review.get("status")
        decision = review.get("decision")
    return status == "host_model_reviewed" and decision == "pass"


def training_leakage_review_reason(review: dict[str, Any] | None) -> str:
    if not isinstance(review, dict):
        return "host-model hard-fact leakage review did not return a valid envelope"
    error = review.get("error")
    if isinstance(error, dict) and error.get("code"):
        return f"host-model hard-fact leakage review failed: {error['code']}"
    status = review.get("status")
    decision = review.get("decision")
    return (
        "host-model hard-fact leakage review did not pass: "
        f"status={status or 'unknown'} decision={decision or 'unknown'}"
    )


def _sample_id(row: dict[str, Any], *, line_no: int, row_fingerprint: str) -> str:
    for key in ("sample_id", "id", "key"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"sft:{line_no}:{row_fingerprint.removeprefix('sha256:')[:12]}"


def _safe_error(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    details = value.get("details")
    return {
        "code": _optional_str(value.get("code")),
        "message": _optional_str(value.get("message")),
        "retryable": bool(value.get("retryable")),
        "details": _safe_details(details),
    }


def _safe_details(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "expected_type",
        "received_type",
        "reason",
        "model_id",
        "raw_output_fingerprint",
        "provider",
        "supported",
    }
    return {
        str(key): _safe_scalar(val)
        for key, val in value.items()
        if key in allowed
    }


def _safe_audit(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {}
    for key in (
        "schema_version",
        "assistant_version",
        "method",
        "provider",
        "host_model",
        "status",
        "generated_at",
        "input_fingerprint",
        "input_summary",
        "warnings",
        "log_target",
    ):
        if key in value:
            out[key] = value[key]
    return out


def _compact_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_scalar(val) for key, val in value.items()}


def _review_items_safe_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        sample_id = _optional_str(item.get("sample_id"))
        rationale = _optional_str(item.get("rationale"))
        matched_entities = item.get("matched_entities")
        matched_entity_count = (
            len(matched_entities) if isinstance(matched_entities, list) else 0
        )
        ref: dict[str, Any] = {
            "sample_idx": _optional_nonnegative_int(item.get("sample_idx")),
            "sample_id": sample_id,
            "line_no": (
                _optional_nonnegative_int(source.get("line_no"))
                or _optional_nonnegative_int(source.get("line"))
                or _line_no_from_sample_id(sample_id)
            ),
            "severity": _safe_severity(item.get("severity")),
            "leakage_detected": bool(item.get("leakage_detected")),
            "matched_entity_count": matched_entity_count,
            "rationale_fingerprint": _fingerprint(rationale) if rationale else None,
            "sample_fingerprint": _optional_str(source.get("sample_fingerprint")),
        }
        refs.append({key: val for key, val in ref.items() if val is not None})
    return refs


def _leaked_review_line_numbers(review_summary: dict[str, Any]) -> set[int]:
    refs = review_summary.get("review_items_safe_refs")
    if not isinstance(refs, list):
        return set()
    line_numbers: set[int] = set()
    for ref in refs:
        if not isinstance(ref, dict) or ref.get("leakage_detected") is not True:
            continue
        line_no = _optional_nonnegative_int(ref.get("line_no"))
        if line_no and line_no > 0:
            line_numbers.add(line_no)
    return line_numbers


def _line_no_from_sample_id(sample_id: str | None) -> int | None:
    if not sample_id:
        return None
    parts = sample_id.split(":")
    if len(parts) >= 3 and parts[0] == "sft":
        return int(parts[1]) if parts[1].isdigit() else None
    return None


def _safe_severity(value: Any) -> str:
    text = _optional_str(value)
    if text in {"none", "low", "medium", "high"}:
        return text
    return "none"


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_safe_scalar(item) for item in value]
    if isinstance(value, dict):
        return _compact_dict(value)
    return repr(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        converted = int(value)
        return converted if converted >= 0 else None
    return None


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
