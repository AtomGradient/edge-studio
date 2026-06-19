# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-model answer-quality review contract for eval receipts.

This service is intentionally evaluation-side only. It asks the Host Model to
judge answer quality from the user prompt, model answer, and runtime structure,
without passing legacy hint fields as grading ground truth.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Sequence

from backend.services.host_model_assistant import (
    HOST_MODEL_PROVIDERS,
    HOST_MODEL_PROVIDER,
    PROVIDER,
    SUPPORTED_PROVIDERS,
    HostModelAssistantError,
    HostModelGenerate,
    _call_host_model,
    _extract_json_block,
    _host_model_id,
    _host_model_output_text,
)


HOST_ANSWER_QUALITY_REVIEW_SCHEMA_VERSION = (
    "edgestudio.host_answer_quality_review.v1"
)
HOST_ANSWER_QUALITY_REVIEW_REQUEST_SCHEMA_VERSION = (
    "edgestudio.host_answer_quality_review_request.v1"
)

_ALLOWED_VERDICTS = frozenset({"pass", "fail", "needs_human_review"})
_ALLOWED_FAILURE_TAGS = frozenset(
    {
        "incorrect_answer",
        "missing_answer",
        "tool_not_called",
        "wrong_tool",
        "tool_misuse",
        "hallucination",
        "refusal",
        "format_error",
        "incomplete",
        "unsafe",
        "insufficient_context",
        "runtime_error",
        "other",
    }
)


def review_host_answer_quality(
    *,
    eval_run: Mapping[str, Any] | None = None,
    cases: Sequence[Mapping[str, Any]] | None = None,
    observations: Sequence[Mapping[str, Any]] | None = None,
    run_id: str | None = None,
    app_id: str | None = None,
    provider: str = PROVIDER,
    host_model_id: str | None = None,
    host_model_generate: HostModelGenerate | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Review model answers without using legacy hint scoring."""

    generated_at_ms = int(time.time() * 1000)
    selected_provider = provider or PROVIDER
    try:
        source_cases = (
            list(cases)
            if cases is not None
            else list(_object(eval_run).get("cases") or [])
        )
        source_observations = (
            list(observations)
            if observations is not None
            else list(_object(eval_run).get("observations") or [])
        )
        effective_run_id = (
            _text(run_id)
            or _text(_object(eval_run).get("run_id"))
            or "host-answer-quality-review"
        )
        subject = _object(_object(eval_run).get("subject"))
        effective_app_id = _text(app_id) or _text(subject.get("app_id"))
        review_cases, skipped = _build_review_cases(
            cases=source_cases,
            observations=source_observations,
        )
    except (TypeError, ValueError) as exc:
        return _error(
            status="invalid_input",
            code="invalid_answer_quality_review_input",
            message=str(exc),
            details={},
            run_id=run_id,
            app_id=app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    if selected_provider not in SUPPORTED_PROVIDERS:
        return _error(
            status="invalid_provider",
            code="invalid_provider",
            message="Unsupported answer-quality review provider.",
            details={
                "provider": selected_provider,
                "supported": sorted(SUPPORTED_PROVIDERS),
            },
            run_id=effective_run_id,
            app_id=effective_app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    if not review_cases:
        return _response(
            status="no_review_cases",
            run_id=effective_run_id,
            app_id=effective_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
            host_model=None,
            reviews=[],
            review_cases=review_cases,
            skipped=skipped,
            warnings=[],
        )

    if selected_provider not in HOST_MODEL_PROVIDERS:
        return _response(
            status="pending_host_model_review",
            run_id=effective_run_id,
            app_id=effective_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
            host_model=None,
            reviews=[],
            review_cases=review_cases,
            skipped=skipped,
            warnings=["answer_quality_review_pending_host_model"],
        )

    try:
        payload, host_model = _generate_host_review(
            run_id=effective_run_id,
            app_id=effective_app_id,
            review_cases=review_cases,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
        )
        raw_reviews = _reviews(payload)
        validation = _validate_reviews(raw_reviews, review_cases)
        if validation["contract_failed"]:
            return _error(
                status="host_model_review_contract_failed",
                code="host_model_review_contract_failed",
                message=(
                    "Host answer-quality review must return exactly one valid "
                    "verdict per review case."
                ),
                details={"review_summary": validation},
                run_id=effective_run_id,
                app_id=effective_app_id,
                provider=selected_provider,
                host_model_id=host_model_id,
                host_model=host_model,
                host_model_review=payload,
                generated_at_ms=generated_at_ms,
            )
        reviews = _normalize_reviews(raw_reviews)
    except HostModelAssistantError as exc:
        return _error(
            status="host_model_failed",
            code=exc.code,
            message=exc.message,
            details=exc.details,
            run_id=effective_run_id,
            app_id=effective_app_id,
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
            run_id=effective_run_id,
            app_id=effective_app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    return _response(
        status="host_answer_quality_reviewed",
        run_id=effective_run_id,
        app_id=effective_app_id,
        generated_at_ms=generated_at_ms,
        provider=selected_provider,
        host_model_id=host_model_id,
        host_model=host_model,
        reviews=reviews,
        review_cases=review_cases,
        skipped=skipped,
        warnings=[],
    )


def _build_review_cases(
    *,
    cases: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_case_id = {
        _required_text(case.get("case_id"), "case.case_id"): case
        for case in cases
        if isinstance(case, Mapping) and _text(case.get("case_id"))
    }
    review_cases: list[dict[str, Any]] = []
    skipped = {"missing_case_id": 0, "missing_prompt": 0, "missing_observation": 0}
    for observation in observations:
        if not isinstance(observation, Mapping):
            skipped["missing_observation"] += 1
            continue
        case_id = _text(observation.get("case_id"))
        if not case_id:
            skipped["missing_case_id"] += 1
            continue
        case = by_case_id.get(case_id, {})
        prompt = _text(case.get("prompt")) or _text(observation.get("prompt"))
        if not prompt:
            skipped["missing_prompt"] += 1
            continue
        answer = _text(observation.get("answer"))
        structural_evidence = _structural_evidence(observation)
        review_cases.append(
            {
                "case_id": case_id,
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "answer": answer,
                "answer_sha256": _sha256_text(answer),
                "structural_evidence": structural_evidence,
                "structural_checks": _structural_checks_for_case(
                    structural_evidence=structural_evidence,
                ),
            }
        )
    return review_cases, skipped


def _structural_evidence(observation: Mapping[str, Any]) -> dict[str, Any]:
    tool_calls = [
        item for item in observation.get("tool_calls") or [] if isinstance(item, Mapping)
    ]
    return {
        "completed": observation.get("completed") is True,
        "freeze_detected": observation.get("freeze_detected") is True,
        "oom": observation.get("oom") is True,
        "has_error": bool(_text(observation.get("error"))),
        "route_intent": _text(observation.get("route_intent")),
        "selected_tools": _text_list(observation.get("selected_tools")),
        "tool_call_count": len(tool_calls),
        "tool_call_names": sorted(
            {
                _text(item.get("name") or item.get("tool_name"))
                for item in tool_calls
                if _text(item.get("name") or item.get("tool_name"))
            }
        ),
        "fact_tool_called": observation.get("fact_tool_called") is True,
    }


def _structural_checks_for_case(
    *,
    structural_evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "completed": structural_evidence["completed"],
        "no_runtime_error": not structural_evidence["has_error"],
        "no_freeze": not structural_evidence["freeze_detected"],
        "no_oom": not structural_evidence["oom"],
        "tool_call_count": structural_evidence["tool_call_count"],
    }


def _generate_host_review(
    *,
    run_id: str,
    app_id: str,
    review_cases: list[dict[str, Any]],
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
    max_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = _review_messages(run_id=run_id, app_id=app_id, review_cases=review_cases)
    token_budget = max_tokens or max(2_000, 700 + len(review_cases) * 600)
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
            message="Host Model answer-quality review call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host Model returned an empty answer-quality review response.",
            retryable=True,
            details={"model_id": model_id},
        )
    payload = json.loads(_extract_json_block(raw_output))
    if not isinstance(payload, dict):
        raise TypeError("Host Model answer-quality review output must be an object")
    return payload, {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _sha256_text(raw_output),
    }


def _review_messages(
    *,
    run_id: str,
    app_id: str,
    review_cases: list[dict[str, Any]],
) -> list[dict[str, str]]:
    request = {
        "task": "review_model_answer_quality_without_golden_answer_lookup",
        "schema_version": HOST_ANSWER_QUALITY_REVIEW_REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "app_id": app_id,
        "review_cases": _host_review_cases(review_cases),
        "input_constraints": {
            "no_legacy_hint_fields": True,
            "judge_from_prompt_answer_and_runtime_evidence_only": True,
            "structural_evidence_is_not_accuracy_evidence": True,
        },
        "allowed_failure_tags": sorted(_ALLOWED_FAILURE_TAGS),
        "output_schema": {
            "reviews": [
                {
                    "case_id": "copy from review_cases[]",
                    "verdict": "pass|fail|needs_human_review",
                    "rationale": "short reason based on prompt, answer, and structural evidence",
                    "failure_tags": ["items from allowed_failure_tags only"],
                    "confidence": "0.0-1.0",
                }
            ],
            "summary": {
                "decision": "pass|fail|needs_human_review",
                "reason": "short aggregate reason",
            },
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio Host Model for non-cheating answer "
                "evaluation. Judge each model answer from the user's prompt, "
                "the answer, and structural runtime evidence. Do not ask for "
                "or use fixed golden answers, golden tool labels, keyword "
                "matches, or substring matches. If the prompt "
                "or answer lacks enough context for a fair quality judgment, "
                "use needs_human_review. Return compact valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, sort_keys=True),
        },
    ]


def _host_review_cases(review_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "prompt": case["prompt"],
            "prompt_sha256": case["prompt_sha256"],
            "answer": case["answer"],
            "answer_sha256": case["answer_sha256"],
            "structural_evidence": case["structural_evidence"],
        }
        for case in review_cases
    ]


def _reviews(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    for container in (result, payload):
        value = container.get("reviews")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError("Host Model answer-quality review output missing reviews list")


def _validate_reviews(
    reviews: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {_text(case.get("case_id")) for case in review_cases}
    matched: list[str] = []
    extras: list[str] = []
    invalid_verdicts: list[str] = []
    invalid_failure_tags: list[dict[str, str]] = []
    missing_rationale: list[str] = []
    for item in reviews:
        case_id = _text(item.get("case_id"))
        if case_id in expected:
            matched.append(case_id)
        else:
            extras.append(case_id or "<missing-case-id>")
        verdict = _text(item.get("verdict")).casefold()
        if verdict not in _ALLOWED_VERDICTS:
            invalid_verdicts.append(case_id or "<missing-case-id>")
        for tag in _text_list(item.get("failure_tags")):
            normalized = tag.casefold()
            if normalized not in _ALLOWED_FAILURE_TAGS:
                invalid_failure_tags.append(
                    {
                        "case_id": case_id or "<missing-case-id>",
                        "tag": tag,
                    }
                )
        if not _text(item.get("rationale")):
            missing_rationale.append(case_id or "<missing-case-id>")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case_id in matched:
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
    missing = sorted(expected - seen)
    return {
        "reviewed_count": len(reviews),
        "expected_count": len(expected),
        "matched_count": len(seen),
        "missing_case_ids": missing,
        "extra_case_ids": sorted(set(extras)),
        "duplicate_case_ids": sorted(duplicates),
        "invalid_verdict_case_ids": sorted(set(invalid_verdicts)),
        "invalid_failure_tags": invalid_failure_tags,
        "missing_rationale_case_ids": sorted(set(missing_rationale)),
        "contract_failed": bool(
            missing
            or extras
            or duplicates
            or invalid_verdicts
            or invalid_failure_tags
            or missing_rationale
        ),
    }


def _normalize_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in reviews:
        verdict = _text(item.get("verdict")).casefold()
        normalized.append(
            {
                "case_id": _text(item.get("case_id")),
                "verdict": verdict,
                "answer_quality_passed": verdict == "pass",
                "rationale": _text(item.get("rationale")),
                "failure_tags": [
                    tag.casefold() for tag in _text_list(item.get("failure_tags"))
                ],
                "confidence": _optional_float(item.get("confidence")),
            }
        )
    return normalized


def _response(
    *,
    status: str,
    run_id: str,
    app_id: str,
    generated_at_ms: int,
    provider: str,
    host_model_id: str | None,
    host_model: dict[str, Any] | None,
    reviews: list[dict[str, Any]],
    review_cases: list[dict[str, Any]],
    skipped: dict[str, int],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "ok": True,
        "schema_version": HOST_ANSWER_QUALITY_REVIEW_SCHEMA_VERSION,
        "status": status,
        "run_id": run_id,
        "app_id": app_id,
        "generated_at_ms": generated_at_ms,
        "result": {
            "summary": _summary(reviews, review_cases=review_cases),
            "structural_checks": _structural_checks(review_cases),
            "host_review_verdict": _host_review_verdict(
                reviews,
                review_cases=review_cases,
            ),
            "reviews": reviews,
            "review_case_refs": _review_case_refs(review_cases),
            "skipped_counts": dict(sorted(skipped.items())),
        },
        "error": None,
        "audit": _audit(
            provider=provider,
            host_model_id=host_model_id,
            host_model=host_model,
            review_cases=review_cases,
            status=status,
            warnings=warnings,
        ),
    }


def _error(
    *,
    status: str,
    code: str,
    message: str,
    details: dict[str, Any],
    run_id: str | None,
    app_id: str | None,
    provider: str,
    host_model_id: str | None,
    generated_at_ms: int,
    host_model: dict[str, Any] | None = None,
    host_model_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": HOST_ANSWER_QUALITY_REVIEW_SCHEMA_VERSION,
        "status": status,
        "run_id": _text(run_id),
        "app_id": _text(app_id),
        "generated_at_ms": generated_at_ms,
        "result": None,
        "host_model_review": (
            {"payload": host_model_review, "host_model": host_model}
            if host_model_review is not None and host_model is not None
            else None
        ),
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": _audit(
            provider=provider,
            host_model_id=host_model_id,
            host_model=host_model,
            review_cases=[],
            status=status,
            warnings=[],
        ),
    }


def _summary(
    reviews: list[dict[str, Any]],
    *,
    review_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {verdict: 0 for verdict in _ALLOWED_VERDICTS}
    for item in reviews:
        verdict = _text(item.get("verdict")).casefold()
        if verdict in counts:
            counts[verdict] += 1
    if not reviews:
        decision = "not_reviewed"
    elif counts["fail"] > 0:
        decision = "fail"
    elif counts["needs_human_review"] > 0:
        decision = "needs_human_review"
    else:
        decision = "pass"
    return {
        "case_count": len(review_cases),
        "reviewed_count": len(reviews),
        "pass_count": counts["pass"],
        "fail_count": counts["fail"],
        "needs_human_review_count": counts["needs_human_review"],
        "decision": decision,
        "answer_quality_evidence_ready": bool(reviews),
        "answer_quality_source": "host_review_verdict" if reviews else "not_reviewed",
        "structural_checks_are_accuracy_evidence": False,
        "production_model_improved": False,
        "ready_for_live_routing": False,
    }


def _structural_checks(review_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "prompt_sha256": case["prompt_sha256"],
            "answer_sha256": case["answer_sha256"],
            "checks": case["structural_checks"],
        }
        for case in review_cases
    ]


def _host_review_verdict(
    reviews: list[dict[str, Any]],
    *,
    review_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _summary(reviews, review_cases=review_cases)
    return {
        "status": "host_model_reviewed" if reviews else "not_reviewed",
        "decision": summary["decision"],
        "answer_quality_evidence_ready": summary["answer_quality_evidence_ready"],
        "reviews": reviews,
    }


def _review_case_refs(review_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "prompt_sha256": case["prompt_sha256"],
            "answer_sha256": case["answer_sha256"],
            "structural_evidence": case["structural_evidence"],
        }
        for case in review_cases
    ]


def _audit(
    *,
    provider: str,
    host_model_id: str | None,
    host_model: dict[str, Any] | None,
    review_cases: list[dict[str, Any]],
    status: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "method": "review_host_answer_quality",
        "provider": provider,
        "host_model": host_model
        or {
            "enabled": provider == HOST_MODEL_PROVIDER or provider in HOST_MODEL_PROVIDERS,
            "model_id": None,
            "selected_model_id": host_model_id,
        },
        "status": status,
        "training_side_only": True,
        "writes_events": False,
        "writes_runtime_artifacts": False,
        "writes_training_sample_tags": False,
        "triggers_rpp": False,
        "triggers_neural_imprint_regen": False,
        "triggers_capsule_push": False,
        "automatic_push": False,
        "no_keyword_or_legacy_hint_scoring": True,
        "legacy_hint_fields_sent_to_host_model": False,
        "input_summary": {
            "review_case_count": len(review_cases),
            "review_case_fingerprints": [
                _sha256_json(
                    {
                        "case_id": case["case_id"],
                        "prompt_sha256": case["prompt_sha256"],
                        "answer_sha256": case["answer_sha256"],
                    }
                )
                for case in review_cases
            ],
        },
        "warnings": warnings,
    }


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return sorted({text for item in value if (text := _text(item))})
    text = _text(value)
    return [text] if text else []


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
