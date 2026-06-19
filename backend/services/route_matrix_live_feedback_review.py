# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host Model review generation for route-matrix live feedback corrections."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from backend.services.host_model_assistant import (
    HOST_MODEL_PROVIDER,
    HOST_MODEL_PROVIDERS,
    PROVIDER,
    SUPPORTED_PROVIDERS,
    HostModelAssistantError,
    HostModelGenerate,
    _call_host_model,
    _extract_json_block,
    _host_model_id,
    _host_model_output_text,
)
from backend.services.route_action_tool_contracts import route_action_tool_contracts_for_prompt


ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_host_model_review.v0"
)

_APPROVED_DECISIONS = frozenset({"approve", "approved", "accept", "accepted", "pass", "ok"})


def generate_route_matrix_live_feedback_review(
    *,
    review_request: dict[str, Any],
    tool_registry: list[dict[str, Any]],
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: HostModelGenerate | None = None,
    max_tokens: int | None = None,
    review_chunk_size: int | None = None,
) -> dict[str, Any]:
    """Ask the Host Model to review corrections and emit reviewed pair payloads."""

    generated_at_ms = int(time.time() * 1000)
    selected_provider = provider or HOST_MODEL_PROVIDER
    try:
        request_result = _review_request_result(review_request)
        app_id = _required_text(request_result.get("app_id"), "review_request.result.app_id")
        run_id = _required_text(request_result.get("run_id"), "review_request.result.run_id")
        cases = _review_cases(request_result)
        tools = _normalize_tool_registry(tool_registry)
    except (TypeError, ValueError) as exc:
        return _error(
            status="invalid_input",
            code="invalid_live_feedback_review_input",
            message=str(exc),
            details={},
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    if not isinstance(selected_provider, str) or selected_provider not in SUPPORTED_PROVIDERS:
        return _error(
            status="invalid_provider",
            code="invalid_provider",
            message="Unsupported live-feedback review provider.",
            details={
                "provider": selected_provider,
                "supported": sorted(SUPPORTED_PROVIDERS),
            },
            app_id=app_id,
            run_id=run_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    if review_chunk_size is not None and review_chunk_size < 1:
        return _error(
            status="invalid_input",
            code="invalid_live_feedback_review_input",
            message="review_chunk_size must be greater than zero.",
            details={"review_chunk_size": review_chunk_size},
            app_id=app_id,
            run_id=run_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    if not cases:
        return {
            "ok": True,
            "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION,
            "status": "no_review_cases",
            "app_id": app_id,
            "run_id": run_id,
            "result": {"reviews": [], "summary": _review_summary([])},
            "error": None,
            "audit": _audit(
                provider=selected_provider,
                host_model={"enabled": False, "model_id": None, "selected_model_id": host_model_id},
                generated_at_ms=generated_at_ms,
                review_request=review_request,
                tool_count=len(tools),
                review_case_count=0,
                status="no_review_cases",
                warnings=[],
            ),
        }

    if selected_provider not in HOST_MODEL_PROVIDERS:
        return {
            "ok": True,
            "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION,
            "status": "pending_host_model",
            "app_id": app_id,
            "run_id": run_id,
            "result": {"reviews": [], "summary": _review_summary([])},
            "error": None,
            "audit": _audit(
                provider=selected_provider,
                host_model={"enabled": False, "model_id": None, "selected_model_id": host_model_id},
                generated_at_ms=generated_at_ms,
                review_request=review_request,
                tool_count=len(tools),
                review_case_count=len(cases),
                status="pending_host_model",
                warnings=["route_matrix_live_feedback_review_pending_host_model"],
            ),
        }

    if review_chunk_size is not None:
        return _generate_chunked_review_response(
            review_request=review_request,
            app_id=app_id,
            run_id=run_id,
            tool_registry=tools,
            review_cases=cases,
            provider=selected_provider,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
            review_chunk_size=review_chunk_size,
            generated_at_ms=generated_at_ms,
        )

    try:
        payload, host_model = _generate_review_payload_with_host_model(
            review_request=review_request,
            app_id=app_id,
            run_id=run_id,
            tool_registry=tools,
            review_cases=cases,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
        )
        reviews = _reviews(payload)
        summary = _review_summary(reviews, review_cases=cases)
        warnings = _review_warnings(summary)
        if summary["approved_missing_pair_payload_count"] > 0:
            return _error(
                status="host_model_review_contract_failed",
                code="host_model_review_contract_failed",
                message=(
                    "Approved live-feedback review items must include a "
                    "reviewed route_action_pair payload."
                ),
                details={"summary": summary},
                app_id=app_id,
                run_id=run_id,
                provider=selected_provider,
                host_model_id=host_model_id,
                host_model=host_model,
                generated_at_ms=generated_at_ms,
            )
    except HostModelAssistantError as exc:
        return _error(
            status="host_model_failed",
            code=exc.code,
            message=exc.message,
            details=exc.details,
            app_id=app_id,
            run_id=run_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return _error(
            status="host_model_schema_error",
            code="host_model_schema_error",
            message=str(exc),
            details={},
            app_id=app_id,
            run_id=run_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION,
        "status": "reviewed_pair_payloads_ready",
        "app_id": app_id,
        "run_id": run_id,
        "result": {"reviews": reviews, "summary": summary},
        "error": None,
        "audit": _audit(
            provider=selected_provider,
            host_model=host_model,
            generated_at_ms=generated_at_ms,
            review_request=review_request,
            tool_count=len(tools),
            review_case_count=len(cases),
            status="reviewed_pair_payloads_ready",
            warnings=warnings,
        ),
    }


def _generate_chunked_review_response(
    *,
    review_request: dict[str, Any],
    app_id: str,
    run_id: str,
    tool_registry: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
    provider: str,
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
    max_tokens: int | None,
    review_chunk_size: int,
    generated_at_ms: int,
) -> dict[str, Any]:
    input_completeness = _review_case_input_completeness(review_cases)
    if not input_completeness["complete"]:
        return _error(
            status="invalid_input",
            code="invalid_live_feedback_review_input",
            message="Chunked Host Model review input cases must have stable unique case identities.",
            details={"review_case_completeness": input_completeness},
            app_id=app_id,
            run_id=run_id,
            provider=provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )
    chunks = _review_case_chunks(review_cases, review_chunk_size)
    reviews: list[dict[str, Any]] = []
    chunk_summaries: list[dict[str, Any]] = []
    host_model_chunks: list[dict[str, Any]] = []
    pending_chunks = list(chunks)
    adaptive_splits: list[dict[str, Any]] = []
    host_model_call_index = 0
    while pending_chunks:
        chunk_cases = pending_chunks.pop(0)
        host_model_call_index += 1
        attempt = _generate_review_chunk_attempt(
            review_request=review_request,
            app_id=app_id,
            run_id=run_id,
            tool_registry=tool_registry,
            review_cases=chunk_cases,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
        )
        if attempt["ok"] is True:
            chunk_reviews = attempt["reviews"]
            chunk_summary = attempt["summary"]
            reviews.extend(chunk_reviews)
            chunk_summaries.append(
                {
                    "chunk_index": len(chunk_summaries) + 1,
                    "host_model_call_index": host_model_call_index,
                    "input_case_count": len(chunk_cases),
                    "reviewed_count": len(chunk_reviews),
                    "case_ids": _case_ids(chunk_cases),
                    "summary": chunk_summary,
                }
            )
            host_model_chunks.append(attempt["host_model"])
            continue
        if len(chunk_cases) > 1:
            split_chunks = _split_review_case_chunk(chunk_cases)
            adaptive_splits.append(
                {
                    "failed_host_model_call_index": host_model_call_index,
                    "input_case_count": len(chunk_cases),
                    "case_ids": _case_ids(chunk_cases),
                    "reason_code": attempt["code"],
                    "split_case_counts": [len(chunk) for chunk in split_chunks],
                }
            )
            pending_chunks = split_chunks + pending_chunks
            continue
        if attempt["status"] == "host_model_failed":
            exc = attempt["exception"]
            return _error(
                status="host_model_failed",
                code=exc.code,
                message=exc.message,
                details={
                    **exc.details,
                    "chunk_index": len(chunk_summaries) + 1,
                    "chunk_count": len(chunks),
                    "host_model_call_index": host_model_call_index,
                    "chunk_case_ids": _case_ids(chunk_cases),
                    "adaptive_splits": adaptive_splits,
                },
                app_id=app_id,
                run_id=run_id,
                provider=provider,
                host_model_id=host_model_id,
                host_model=_chunked_host_model_summary(
                    host_model_id=host_model_id,
                    chunks=host_model_chunks,
                ),
                generated_at_ms=generated_at_ms,
            )
        return _error(
            status=attempt["status"],
            code=attempt["code"],
            message=attempt["message"],
            details={
                **attempt["details"],
                "chunk_index": len(chunk_summaries) + 1,
                "chunk_count": len(chunks),
                "host_model_call_index": host_model_call_index,
                "chunk_case_ids": _case_ids(chunk_cases),
                "adaptive_splits": adaptive_splits,
            },
            app_id=app_id,
            run_id=run_id,
            provider=provider,
            host_model_id=host_model_id,
            host_model=_chunked_host_model_summary(
                host_model_id=host_model_id,
                chunks=host_model_chunks,
            ),
            generated_at_ms=generated_at_ms,
        )

    completeness = _review_case_completeness(reviews, review_cases)
    if not completeness["complete"]:
        return _error(
            status="host_model_review_contract_failed",
            code="host_model_review_contract_failed",
            message="Chunked Host Model review merge must preserve every input case exactly once.",
            details={"review_case_completeness": completeness},
            app_id=app_id,
            run_id=run_id,
            provider=provider,
            host_model_id=host_model_id,
            host_model=_chunked_host_model_summary(
                host_model_id=host_model_id,
                chunks=host_model_chunks,
            ),
            generated_at_ms=generated_at_ms,
        )

    summary = _review_summary(reviews, review_cases=review_cases)
    warnings = _review_warnings(summary)
    if summary["approved_missing_pair_payload_count"] > 0:
        return _error(
            status="host_model_review_contract_failed",
            code="host_model_review_contract_failed",
            message=(
                "Approved live-feedback review items must include a "
                "reviewed route_action_pair payload."
            ),
            details={"summary": summary},
            app_id=app_id,
            run_id=run_id,
            provider=provider,
            host_model_id=host_model_id,
            host_model=_chunked_host_model_summary(
                host_model_id=host_model_id,
                chunks=host_model_chunks,
            ),
            generated_at_ms=generated_at_ms,
        )

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION,
        "status": "reviewed_pair_payloads_ready",
        "app_id": app_id,
        "run_id": run_id,
        "result": {
            "reviews": reviews,
            "summary": summary,
            "review_chunks": chunk_summaries,
        },
        "error": None,
        "audit": {
            **_audit(
                provider=provider,
                host_model=_chunked_host_model_summary(
                    host_model_id=host_model_id,
                    chunks=host_model_chunks,
                ),
                generated_at_ms=generated_at_ms,
                review_request=review_request,
                tool_count=len(tool_registry),
                review_case_count=len(review_cases),
                status="reviewed_pair_payloads_ready",
                warnings=warnings,
            ),
            "review_chunking": {
                "enabled": True,
                "chunk_size": review_chunk_size,
                "initial_chunk_count": len(chunks),
                "initial_chunk_case_counts": [len(chunk) for chunk in chunks],
                "chunk_count": len(chunk_summaries),
                "chunk_case_counts": [
                    int(chunk["input_case_count"]) for chunk in chunk_summaries
                ],
                "host_model_call_count": host_model_call_index,
                "adaptive_split_count": len(adaptive_splits),
                "adaptive_splits": adaptive_splits,
                "completeness": completeness,
            },
        },
    }


def _generate_review_chunk_attempt(
    *,
    review_request: dict[str, Any],
    app_id: str,
    run_id: str,
    tool_registry: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    try:
        payload, host_model = _generate_review_payload_with_host_model(
            review_request=review_request,
            app_id=app_id,
            run_id=run_id,
            tool_registry=tool_registry,
            review_cases=review_cases,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
        )
        chunk_reviews = _reviews(payload)
        completeness = _review_case_completeness(chunk_reviews, review_cases)
        if not completeness["complete"]:
            return {
                "ok": False,
                "status": "host_model_review_contract_failed",
                "code": "host_model_review_contract_failed",
                "message": (
                    "Chunked Host Model review must return exactly one review "
                    "per input case."
                ),
                "details": {"review_case_completeness": completeness},
            }
        chunk_summary = _review_summary(chunk_reviews, review_cases=review_cases)
        if chunk_summary["approved_missing_pair_payload_count"] > 0:
            return {
                "ok": False,
                "status": "host_model_review_contract_failed",
                "code": "host_model_review_contract_failed",
                "message": (
                    "Approved live-feedback review items must include a "
                    "reviewed route_action_pair payload."
                ),
                "details": {"summary": chunk_summary},
            }
        return {
            "ok": True,
            "reviews": chunk_reviews,
            "summary": chunk_summary,
            "host_model": host_model,
        }
    except HostModelAssistantError as exc:
        return {
            "ok": False,
            "status": "host_model_failed",
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "exception": exc,
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "status": "host_model_schema_error",
            "code": "host_model_schema_error",
            "message": str(exc),
            "details": {},
        }


def _generate_review_payload_with_host_model(
    *,
    review_request: dict[str, Any],
    app_id: str,
    run_id: str,
    tool_registry: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
    max_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = _build_review_messages(
        app_id=app_id,
        run_id=run_id,
        tool_registry=tool_registry,
        review_cases=review_cases,
    )
    token_budget = max_tokens or _review_token_budget(len(review_cases))
    try:
        raw = (
            host_model_generate(messages, token_budget, 0.0)
            if host_model_generate is not None
            else _call_host_model(
                messages,
                host_model_id=host_model_id,
                host_model_generate=None,
                max_tokens=token_budget,
                temperature=0.0,
            )
        )
    except HostModelAssistantError:
        raise
    except Exception as exc:
        raise HostModelAssistantError(
            code="host_model_call_failed",
            message="Host Model live-feedback review call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host Model returned an empty live-feedback review response.",
            retryable=True,
            details={"model_id": model_id},
        )
    payload = json.loads(_extract_json_block(raw_output))
    if not isinstance(payload, dict):
        raise TypeError("Host Model live-feedback review output must be an object")
    return payload, {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _fingerprint(raw_output),
    }


def _build_review_messages(
    *,
    app_id: str,
    run_id: str,
    tool_registry: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
) -> list[dict[str, str]]:
    request = {
        "task": "review_route_matrix_live_feedback_and_generate_reviewed_pairs",
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION,
        "app_id": app_id,
        "run_id": run_id,
        "tool_registry": tool_registry,
        "tool_contracts": route_action_tool_contracts_for_prompt(
            [tool["name"] for tool in tool_registry],
            tool_registry=tool_registry,
        ),
        "language_policy": {
            "prompt_variants_preserve_review_case_language": True,
            "source_language_fields": [
                "review_cases[].input_text",
                "review_cases[].correction_text",
                "review_cases[].user_correction",
            ],
            "instruction": (
                "Maintain the source language of the user's input and "
                "correction when producing prompt_variants and rationales. "
                "Do not translate to English unless the user input is in English."
            ),
        },
        "output_constraints": {
            "valid_json_only": True,
            "compact_output": True,
            "no_fields_outside_output_schema": True,
            "max_rationale_characters": 160,
            "no_markdown_or_backticks_in_string_values": True,
            "no_unescaped_quote_characters_inside_string_values": True,
        },
        "review_cases": review_cases,
        "output_schema": {
            "reviews": [
                {
                    "source_event_id": "copy from review_cases[]",
                    "case_id": "copy from review_cases[]",
                    "verdict": "approved|reject",
                    "rationale": "short review rationale",
                    "route_action_pair": {
                        "route_intent": "one corrected route intent",
                        "prompt_variants": [
                            (
                                "same-language entity-free training paraphrase; "
                                "do not replay input_text"
                            )
                        ],
                        "selected_tools": ["tool names from tool_registry only"],
                        "tool_call_plan": [
                            {
                                "tool_name": "one selected tool name",
                                "args": "object using declared schema keys only",
                                "reason": "why this call is required",
                            }
                        ],
                        "answer_discipline": "how to answer without memorizing exact facts",
                        "rationale": "why the corrected route/tool target is right",
                        "confidence": "0.0-1.0",
                    },
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio Host Model for live-feedback route "
                "learning. Review each explicit user correction against the "
                "live decision, then emit a reviewed route_action_pair payload "
                "only when the correction is structurally usable. Do not treat "
                "the original live decision as ground truth. Do not replay "
                "input_text as a training prompt variant. Do not invent tool "
                "names or argument keys. Tool call args must match the app "
                "tool registry schema and must be entity-free: use enum values, "
                "bounded numbers, or parser-owned placeholders instead of "
                "one-off user facts, dates, amounts, names, or IDs. Training "
                "prompt_variants must be complete natural-language examples; "
                "write each prompt_variant in the same language and locale as "
                "that review case's input_text and user correction. Do not "
                "translate to English unless the user input is in English. "
                "Return compact valid JSON only. Do not add fields outside the "
                "requested output_schema. Keep every rationale string under "
                "160 characters. Do not use markdown, backticks, or quotation "
                "marks inside string values. "
                "do not use unresolved placeholders such as N, X, <title>, "
                "{date}, or bracketed variable text. If the correction implies "
                "a concrete bounded value such as 'three', the prompt variant "
                "and tool_call_plan must agree on that concrete value. Reject "
                "cases that cannot be safely converted. Do not copy output "
                "schema example phrases literally; every string field in an "
                "approved route_action_pair must be case-specific. Return only "
                "one JSON object matching the requested schema."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, sort_keys=True),
        },
    ]


def _review_request_result(review_request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(review_request, dict):
        raise TypeError("review_request must be an object")
    if review_request.get("ok") is not True:
        raise ValueError("review_request.ok must be true")
    result = review_request.get("result")
    if not isinstance(result, dict):
        raise ValueError("review_request.result must be an object")
    return result


def _review_cases(result: dict[str, Any]) -> list[dict[str, Any]]:
    value = result.get("review_cases")
    if not isinstance(value, list):
        raise ValueError("review_request.result.review_cases must be a list")
    return [item for item in value if isinstance(item, dict)]


def _reviews(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    output_schema = (
        payload.get("output_schema")
        if isinstance(payload.get("output_schema"), dict)
        else {}
    )
    output_schema_result = (
        output_schema.get("result")
        if isinstance(output_schema.get("result"), dict)
        else {}
    )
    for container in (result, payload, output_schema_result, output_schema):
        value = container.get("reviews")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError("Host Model live-feedback review output missing reviews list")


def _review_summary(
    reviews: list[dict[str, Any]],
    *,
    review_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cases_by_event_id, cases_by_case_id = _case_indexes(review_cases or [])
    approved_count = 0
    rejected_count = 0
    approved_missing_pair_payload_count = 0
    approved_language_mismatch_count = 0
    for item in reviews:
        if _is_approved(item):
            approved_count += 1
            pair_payload = item.get("route_action_pair")
            if not _pair_has_required_keys(pair_payload):
                approved_missing_pair_payload_count += 1
            elif not _pair_preserves_case_language(
                pair_payload,
                case=_matching_case(
                    item,
                    cases_by_event_id=cases_by_event_id,
                    cases_by_case_id=cases_by_case_id,
                ),
            ):
                approved_language_mismatch_count += 1
        else:
            rejected_count += 1
    return {
        "reviewed_count": len(reviews),
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "approved_missing_pair_payload_count": approved_missing_pair_payload_count,
        "approved_language_mismatch_count": approved_language_mismatch_count,
        "ready_for_live_feedback_import": (
            approved_count > 0
            and approved_missing_pair_payload_count == 0
        ),
    }


def _review_case_chunks(
    review_cases: list[dict[str, Any]],
    chunk_size: int,
) -> list[list[dict[str, Any]]]:
    return [
        review_cases[index : index + chunk_size]
        for index in range(0, len(review_cases), chunk_size)
    ]


def _split_review_case_chunk(review_cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    midpoint = max(1, len(review_cases) // 2)
    return [
        chunk
        for chunk in (review_cases[:midpoint], review_cases[midpoint:])
        if chunk
    ]


def _review_case_completeness(
    reviews: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    input_completeness = _review_case_input_completeness(review_cases)
    expected_set = set(input_completeness["case_ids"])
    matched: list[str] = []
    extras: list[str] = []
    for item in reviews:
        identity = _review_identity(item)
        if identity in expected_set:
            matched.append(identity)
        else:
            extras.append(identity or "<missing-case-id>")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for identity in matched:
        if identity in seen:
            duplicates.add(identity)
        seen.add(identity)
    missing = sorted(expected_set - seen)
    input_duplicate_case_ids = input_completeness["input_duplicate_case_ids"]
    missing_input_identity_count = input_completeness["missing_input_identity_count"]
    return {
        "complete": (
            input_completeness["complete"]
            and not missing
            and not extras
            and not duplicates
        ),
        "expected_count": len(expected_set),
        "reviewed_count": len(reviews),
        "matched_count": len(seen),
        "missing_case_ids": missing,
        "extra_case_ids": sorted(set(extras)),
        "duplicate_case_ids": sorted(duplicates),
        "input_duplicate_case_ids": input_duplicate_case_ids,
        "missing_input_identity_count": missing_input_identity_count,
    }


def _review_case_input_completeness(review_cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = [_case_identity(case) for case in review_cases]
    non_empty_case_ids = [case_id for case_id in case_ids if case_id]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case_id in non_empty_case_ids:
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    missing_count = len(case_ids) - len(non_empty_case_ids)
    return {
        "complete": missing_count == 0 and not duplicates,
        "case_ids": non_empty_case_ids,
        "expected_input_count": len(case_ids),
        "unique_input_count": len(seen),
        "input_duplicate_case_ids": sorted(duplicates),
        "missing_input_identity_count": missing_count,
    }


def _case_identity(case: dict[str, Any]) -> str:
    return _text(case.get("case_id")) or _text(case.get("source_event_id"))


def _review_identity(item: dict[str, Any]) -> str:
    return _text(item.get("case_id")) or _text(item.get("source_event_id") or item.get("event_id"))


def _case_ids(cases: list[dict[str, Any]]) -> list[str]:
    return [_case_identity(case) for case in cases if _case_identity(case)]


def _chunked_host_model_summary(
    *,
    host_model_id: str | None,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    model_ids = sorted(
        {
            _text(chunk.get("model_id"))
            for chunk in chunks
            if _text(chunk.get("model_id"))
        }
    )
    return {
        "enabled": True,
        "model_id": model_ids[0] if len(model_ids) == 1 else None,
        "selected_model_id": host_model_id,
        "chunk_count": len(chunks),
        "chunk_model_ids": model_ids,
        "raw_output_fingerprints": [
            chunk["raw_output_fingerprint"]
            for chunk in chunks
            if _text(chunk.get("raw_output_fingerprint"))
        ],
    }


def _review_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if int(summary.get("approved_language_mismatch_count") or 0) > 0:
        warnings.append("route_matrix_live_feedback_prompt_variant_language_drift")
    return warnings


def _case_indexes(
    cases: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_event_id = {
        _text(case.get("source_event_id")): case
        for case in cases
        if _text(case.get("source_event_id"))
    }
    by_case_id = {
        _text(case.get("case_id")): case
        for case in cases
        if _text(case.get("case_id"))
    }
    return by_event_id, by_case_id


def _matching_case(
    item: dict[str, Any],
    *,
    cases_by_event_id: dict[str, dict[str, Any]],
    cases_by_case_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    source_event_id = _text(item.get("source_event_id") or item.get("event_id"))
    if source_event_id and source_event_id in cases_by_event_id:
        return cases_by_event_id[source_event_id]
    case_id = _text(item.get("case_id"))
    if case_id and case_id in cases_by_case_id:
        return cases_by_case_id[case_id]
    return None


def _pair_preserves_case_language(value: Any, *, case: dict[str, Any] | None) -> bool:
    if not isinstance(value, dict) or not isinstance(case, dict):
        return True
    source_text = _case_language_source_text(case)
    if not _has_cjk(source_text):
        return True
    prompt_variants = (
        value.get("prompt_variants")
        or value.get("training_prompts")
        or value.get("variants")
    )
    return all(_has_cjk(variant) for variant in _text_list(prompt_variants))


def _case_language_source_text(case: dict[str, Any]) -> str:
    correction = (
        case.get("user_correction")
        if isinstance(case.get("user_correction"), dict)
        else {}
    )
    parts = [
        case.get("input_text"),
        case.get("correction_text"),
        correction.get("source_input_text"),
        correction.get("input_text"),
        correction.get("correction_text"),
        correction.get("natural_language_correction"),
        correction.get("corrected_input_text"),
    ]
    return "\n".join(_text(part) for part in parts if _text(part))


def _pair_has_required_keys(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if _contains_schema_placeholder_text(value):
        return False
    route_intent = _first_text(
        value,
        ("route_intent", "expected_route_intent", "intent", "routeIntent"),
    )
    if not route_intent:
        return False
    prompt_variants = (
        value.get("prompt_variants")
        or value.get("training_prompts")
        or value.get("variants")
    )
    if not _non_empty_text_list(prompt_variants):
        return False
    if _text_list_has_unresolved_placeholders(prompt_variants):
        return False
    if not _has_list_field(value, ("selected_tools", "expected_selected_tools")):
        return False
    if not _has_list_field(value, ("tool_call_plan", "toolCallPlan", "plan")):
        return False
    if not _first_text(value, ("answer_discipline", "answer_policy", "answerPolicy")):
        return False
    if not _first_text(value, ("rationale", "reason", "explanation")):
        return False
    return True


def _first_text(value: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _text(value.get(key))
        if text:
            return text
    return ""


def _non_empty_text_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return any(_text(item) for item in value)


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _text_list_has_unresolved_placeholders(value: Any) -> bool:
    if not isinstance(value, list):
        return True
    return any(_has_unresolved_placeholder(_text(item)) for item in value)


def _has_unresolved_placeholder(text: str) -> bool:
    if not text:
        return False
    if re.search(r"<[^>]+>", text):
        return True
    if re.search(r"\{[^{}]+\}", text):
        return True
    if re.search(r"\[[^\[\]]+\]", text):
        return True
    if re.search(r"\b[XYZ]\b", text):
        return True
    if re.search(r"\bN\b", text):
        return True
    if "..." in text or "…" in text:
        return True
    return False


_SCHEMA_PLACEHOLDER_TEXTS = {
    "copy from review_cases[]",
    "short review rationale",
    "one corrected route intent",
    "entity-free training paraphrase; do not replay input_text",
    "same-language entity-free training paraphrase; do not replay input_text",
    "tool names from tool_registry only",
    "one selected tool name",
    "object using declared schema keys only",
    "why this call is required",
    "how to answer without memorizing exact facts",
    "why the corrected route/tool target is right",
    "0.0-1.0",
}


def _contains_schema_placeholder_text(value: Any) -> bool:
    if isinstance(value, str):
        normalized = " ".join(value.strip().casefold().split())
        return normalized in _SCHEMA_PLACEHOLDER_TEXTS
    if isinstance(value, list):
        return any(_contains_schema_placeholder_text(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_schema_placeholder_text(item) for item in value.values())
    return False


def _has_list_field(value: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key in value:
            return isinstance(value.get(key), list)
    return False


def _normalize_tool_registry(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("tool_registry must be a list")
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise TypeError(f"tool_registry[{index}] must be an object")
        name = _required_text(
            raw.get("name") or raw.get("tool_name") or raw.get("toolName"),
            f"tool_registry[{index}].name",
        )
        if name in seen:
            raise ValueError("tool_registry must not contain duplicate tool names")
        seen.add(name)
        tool = dict(raw)
        tool["name"] = name
        tools.append(tool)
    if not tools:
        raise ValueError("tool_registry must contain at least one tool")
    return tools


def _is_approved(item: dict[str, Any]) -> bool:
    for key in ("decision", "review_decision", "status", "verdict", "label", "decision_label"):
        text = _text(item.get(key)).casefold()
        if text:
            return text in _APPROVED_DECISIONS
    return False


def _review_token_budget(case_count: int) -> int:
    return min(12000, max(2000, 900 + max(1, int(case_count)) * 900))


def _audit(
    *,
    provider: str,
    host_model: dict[str, Any],
    generated_at_ms: int,
    review_request: dict[str, Any] | None = None,
    tool_count: int = 0,
    review_case_count: int = 0,
    status: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "edgestudio.route_matrix_live_feedback_host_model_review_audit.v0",
        "method": "generate_route_matrix_live_feedback_review",
        "provider": provider,
        "host_model": host_model,
        "status": status,
        "generated_at_ms": generated_at_ms,
        "generated_at": _utc_from_ms(generated_at_ms),
        "input_fingerprint": _fingerprint(review_request) if review_request is not None else None,
        "input_summary": {
            "tool_count": tool_count,
            "review_case_count": review_case_count,
        },
        "warnings": list(warnings),
        "training_side_only": True,
        "writes_events": False,
        "writes_runtime_artifacts": False,
        "writes_training_sample_tags": False,
    }


def _error(
    *,
    status: str,
    code: str,
    message: str,
    details: dict[str, Any],
    provider: str,
    host_model_id: str | None,
    generated_at_ms: int,
    app_id: str | None = None,
    run_id: str | None = None,
    host_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_HOST_MODEL_REVIEW_SCHEMA_VERSION,
        "status": status,
        "app_id": app_id,
        "run_id": run_id,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": _audit(
            provider=provider,
            host_model=host_model
            or {"enabled": provider in HOST_MODEL_PROVIDERS, "model_id": None, "selected_model_id": host_model_id},
            generated_at_ms=generated_at_ms,
            status=status,
            warnings=[],
        ),
    }


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_cjk(value: Any) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", _text(value)))


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
