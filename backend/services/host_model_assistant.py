# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-model assistant contracts for personalization pipeline v0.

This module intentionally keeps v0 narrow: it exposes profile naming,
route/action pair, and leakage-review contracts, with a deterministic
placeholder provider and an optional EdgeStudio host-model provider. Local v0
placeholders do not perform deterministic leakage blocking or redaction; they
only fix the API contracts until the host model is invoked.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from backend.services.route_action_tool_contracts import (
    route_action_tool_contracts_for_prompt,
    validate_route_action_tool_contracts,
)


PROFILE_NAMING_SCHEMA_VERSION = "edgestudio.host_model.profile_naming.v0"
ROUTE_ACTION_SCHEMA_VERSION = "edgestudio.host_model.route_action_pairs.v0"
LEAKAGE_REVIEW_SCHEMA_VERSION = "edgestudio.host_model.hard_fact_leakage_review.v0"
AUDIT_SCHEMA_VERSION = "edgestudio.host_model.audit.v0"
HOST_MODEL_ASSISTANT_VERSION = "host_model_assistant.v0"
PROVIDER = "local_deterministic_stub"
HOST_MODEL_PROVIDER = "edgestudio_host_runtime"
LEGACY_HOST_MODEL_PROVIDER = "classify_service_host_model"
HOST_MODEL_PROVIDERS = frozenset({HOST_MODEL_PROVIDER, LEGACY_HOST_MODEL_PROVIDER})
HostModelGenerate = Callable[[list[dict[str, str]], int, float], Any]
SUPPORTED_PROVIDERS = frozenset({PROVIDER, *HOST_MODEL_PROVIDERS})
HOST_MODEL_MAX_OUTPUT_TOKENS = int(
    os.environ.get("EDGE_HOST_MODEL_MAX_OUTPUT_TOKENS", "8192")
)
ROUTE_ACTION_ALLOWED_INTENTS = (
    "base_chat",
    "user_profile",
    "exact_fact",
    "aggregate_fact",
    "app_action",
    "mixed",
)
_ROUTE_ACTION_ALLOWED_INTENT_SET = frozenset(ROUTE_ACTION_ALLOWED_INTENTS)


class HostModelAssistantError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        details: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details


def generate_profile_naming(
    rpp_output: dict,
    forbidden_entities: list[str] | None = None,
    *,
    host_model_id: str | None = None,
    provider: str = PROVIDER,
    host_model_generate: HostModelGenerate | None = None,
) -> dict:
    """Generate a v0 profile-naming envelope from an RPP output dict.

    The default v0 provider does not infer semantic labels. It extracts stable
    direction identities when present and returns placeholder names with fixed
    schema, error shape, and audit metadata. `HOST_MODEL_PROVIDER` delegates
    naming to the selected host model. `forbidden_entities` is not a sanitizer;
    hard-fact leakage review is intentionally a separate stage.
    """

    generated_at = _utc_now()
    fingerprint = _input_fingerprint(
        rpp_output,
        forbidden_entities,
        provider=provider,
        host_model_id=host_model_id,
    )

    if not isinstance(rpp_output, dict):
        return _response(
            ok=False,
            result=None,
            error={
                "code": "invalid_input",
                "message": "rpp_output must be a dict.",
                "retryable": False,
                "details": {
                    "expected_type": "dict",
                    "received_type": type(rpp_output).__name__,
                },
            },
            audit=_audit(
                rpp_output,
                forbidden_entities=forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )

    try:
        _normalize_forbidden_entities(forbidden_entities)
    except TypeError as exc:
        return _response(
            ok=False,
            result=None,
            error={
                "code": "invalid_input",
                "message": "forbidden_entities must be a list of strings when provided.",
                "retryable": False,
                "details": {
                    "expected_type": "list[str] | None",
                    "received_type": type(forbidden_entities).__name__,
                    "reason": str(exc),
                },
            },
            audit=_audit(
                rpp_output,
                forbidden_entities=forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )

    try:
        sources = _extract_direction_sources(rpp_output)
    except (TypeError, ValueError) as exc:
        return _response(
            ok=False,
            result=None,
            error={
                "code": "unsupported_rpp_shape",
                "message": "rpp_output has an unsupported v0 shape.",
                "retryable": False,
                "details": {
                    "reason": str(exc),
                },
            },
            audit=_audit(
                rpp_output,
                forbidden_entities=forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )
    warnings: list[str] = []
    if not sources:
        warnings.append("no_stable_directions_found")

    if not isinstance(provider, str) or provider not in {PROVIDER, *HOST_MODEL_PROVIDERS}:
        return _response(
            ok=False,
            result=None,
            error={
                "code": "invalid_provider",
                "message": "Unsupported profile naming provider.",
                "retryable": False,
                "details": {
                    "provider": provider,
                    "received_type": type(provider).__name__,
                    "supported": [
                        PROVIDER,
                        HOST_MODEL_PROVIDER,
                        LEGACY_HOST_MODEL_PROVIDER,
                    ],
                },
            },
            audit=_audit(
                rpp_output,
                forbidden_entities=forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=False,
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=warnings,
            ),
        )

    host_model: dict[str, Any] | None = None
    if _is_host_model_provider(provider):
        try:
            result, host_model = _generate_profile_naming_with_host_model(
                rpp_output=rpp_output,
                forbidden_entities=_normalize_forbidden_entities(forbidden_entities),
                sources=sources,
                host_model_id=host_model_id,
                host_model_generate=host_model_generate,
            )
        except HostModelAssistantError as exc:
            return _response(
                ok=False,
                result=None,
                error={
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "details": exc.details,
                },
                audit=_audit(
                    rpp_output,
                    forbidden_entities=forbidden_entities,
                    provider=provider,
                    host_model={
                        "enabled": True,
                        "model_id": exc.details.get("model_id"),
                        "selected_model_id": host_model_id,
                    },
                    generated_at=generated_at,
                    fingerprint=fingerprint,
                    status="error",
                    warnings=warnings,
                ),
            )
    else:
        result = _generate_profile_naming_locally(rpp_output, sources)

    return _response(
        ok=True,
        result=result,
        error=None,
        audit=_audit(
            rpp_output,
            forbidden_entities=forbidden_entities,
            provider=provider,
            host_model=host_model,
            generated_at=generated_at,
            fingerprint=fingerprint,
            status="ok",
            warnings=warnings,
        ),
    )


def generate_route_action_pairs(
    rpp_output: dict,
    eval_cases: list[dict],
    *,
    tool_registry: list[dict[str, Any]] | None = None,
    host_model_id: str | None = None,
    provider: str = PROVIDER,
    host_model_generate: HostModelGenerate | None = None,
) -> dict:
    """Generate a v0 route/action pair envelope from RPP output and eval cases.

    The local v0 provider fixes the contract but does not invent route labels,
    tool plans, or answer policy. Host-model providers may return generated
    pairs that are normalized into this envelope.
    """
    generated_at = _utc_now()
    fingerprint = _route_action_input_fingerprint(
        rpp_output,
        eval_cases,
        tool_registry=tool_registry,
        provider=provider,
        host_model_id=host_model_id,
    )

    if not isinstance(rpp_output, dict):
        return _route_action_response(
            ok=False,
            result=None,
            error={
                "code": "invalid_input",
                "message": "rpp_output must be a dict.",
                "retryable": False,
                "details": {
                    "expected_type": "dict",
                    "received_type": type(rpp_output).__name__,
                },
            },
            audit=_route_action_audit(
                rpp_output,
                eval_cases,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )

    try:
        case_sources = _extract_eval_case_sources(eval_cases)
    except TypeError as exc:
        return _route_action_response(
            ok=False,
            result=None,
            error={
                "code": "invalid_input",
                "message": "eval_cases must be a list of dicts.",
                "retryable": False,
                "details": {
                    "expected_type": "list[dict]",
                    "received_type": type(eval_cases).__name__,
                    "reason": str(exc),
                },
            },
            audit=_route_action_audit(
                rpp_output,
                eval_cases,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )

    warnings: list[str] = []
    if not case_sources:
        warnings.append("no_eval_cases_provided")

    if not isinstance(provider, str) or provider not in SUPPORTED_PROVIDERS:
        return _route_action_response(
            ok=False,
            result=None,
            error={
                "code": "invalid_provider",
                "message": "Unsupported route/action provider.",
                "retryable": False,
                "details": {
                    "provider": provider,
                    "received_type": type(provider).__name__,
                    "supported": _supported_providers_list(),
                },
            },
            audit=_route_action_audit(
                rpp_output,
                eval_cases,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=False,
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=warnings,
            ),
        )

    host_model: dict[str, Any] | None = None
    if _is_host_model_provider(provider):
        try:
            result, host_model = _generate_route_action_pairs_with_host_model(
                rpp_output=rpp_output,
                eval_cases=eval_cases,
                case_sources=case_sources,
                tool_registry=tool_registry,
                host_model_id=host_model_id,
                host_model_generate=host_model_generate,
            )
        except HostModelAssistantError as exc:
            return _route_action_response(
                ok=False,
                result=None,
                error={
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "details": exc.details,
                },
                audit=_route_action_audit(
                    rpp_output,
                    eval_cases,
                    provider=provider,
                    host_model={
                        "enabled": True,
                        "model_id": exc.details.get("model_id"),
                        "selected_model_id": host_model_id,
                    },
                    generated_at=generated_at,
                    fingerprint=fingerprint,
                    status="error",
                    warnings=warnings,
                ),
            )
    else:
        result = _generate_route_action_pairs_locally(rpp_output, case_sources)
        if case_sources:
            warnings.append("route_action_pairs_stub_pending_host_model")

    return _route_action_response(
        ok=True,
        result=result,
        error=None,
        audit=_route_action_audit(
            rpp_output,
            eval_cases,
            provider=provider,
            host_model=host_model,
            generated_at=generated_at,
            fingerprint=fingerprint,
            status="ok",
            warnings=warnings,
        ),
    )


def review_hard_fact_leakage(
    samples: list[dict],
    forbidden_entities: list[str],
    *,
    host_model_id: str | None = None,
    provider: str = PROVIDER,
    host_model_generate: HostModelGenerate | None = None,
) -> dict:
    """Review candidate samples for hard-fact leakage.

    The local v0 provider does not judge or redact samples. It returns a pending
    envelope so callers cannot mistake the stub for a safety gate.
    """
    generated_at = _utc_now()
    fingerprint = _leakage_review_input_fingerprint(
        samples,
        forbidden_entities,
        provider=provider,
        host_model_id=host_model_id,
    )

    try:
        sample_sources = _extract_sample_sources(samples)
    except TypeError as exc:
        return _leakage_review_response(
            ok=False,
            result=None,
            error={
                "code": "invalid_input",
                "message": "samples must be a list of dicts.",
                "retryable": False,
                "details": {
                    "expected_type": "list[dict]",
                    "received_type": type(samples).__name__,
                    "reason": str(exc),
                },
            },
            audit=_leakage_review_audit(
                samples,
                forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )

    try:
        normalized_forbidden_entities = _normalize_forbidden_entities(forbidden_entities)
    except TypeError as exc:
        return _leakage_review_response(
            ok=False,
            result=None,
            error={
                "code": "invalid_input",
                "message": "forbidden_entities must be a list of strings.",
                "retryable": False,
                "details": {
                    "expected_type": "list[str]",
                    "received_type": type(forbidden_entities).__name__,
                    "reason": str(exc),
                },
            },
            audit=_leakage_review_audit(
                samples,
                forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=_is_host_model_provider(provider),
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=[],
            ),
        )

    warnings: list[str] = []
    if not sample_sources:
        warnings.append("no_samples_provided")
    if not normalized_forbidden_entities:
        warnings.append("no_forbidden_entities_provided")

    if not isinstance(provider, str) or provider not in SUPPORTED_PROVIDERS:
        return _leakage_review_response(
            ok=False,
            result=None,
            error={
                "code": "invalid_provider",
                "message": "Unsupported leakage review provider.",
                "retryable": False,
                "details": {
                    "provider": provider,
                    "received_type": type(provider).__name__,
                    "supported": _supported_providers_list(),
                },
            },
            audit=_leakage_review_audit(
                samples,
                forbidden_entities,
                provider=provider,
                host_model=_host_model_audit(
                    enabled=False,
                    selected_model_id=host_model_id,
                ),
                generated_at=generated_at,
                fingerprint=fingerprint,
                status="error",
                warnings=warnings,
            ),
        )

    host_model: dict[str, Any] | None = None
    if _is_host_model_provider(provider):
        try:
            result, host_model = _review_hard_fact_leakage_with_host_model(
                samples=samples,
                forbidden_entities=normalized_forbidden_entities,
                sample_sources=sample_sources,
                host_model_id=host_model_id,
                host_model_generate=host_model_generate,
            )
        except HostModelAssistantError as exc:
            return _leakage_review_response(
                ok=False,
                result=None,
                error={
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "details": exc.details,
                },
                audit=_leakage_review_audit(
                    samples,
                    forbidden_entities,
                    provider=provider,
                    host_model={
                        "enabled": True,
                        "model_id": exc.details.get("model_id"),
                        "selected_model_id": host_model_id,
                    },
                    generated_at=generated_at,
                    fingerprint=fingerprint,
                    status="error",
                    warnings=warnings,
                ),
            )
    else:
        result = _review_hard_fact_leakage_locally(sample_sources, normalized_forbidden_entities)
        if sample_sources:
            warnings.append("hard_fact_leakage_review_stub_pending_host_model")

    return _leakage_review_response(
        ok=True,
        result=result,
        error=None,
        audit=_leakage_review_audit(
            samples,
            forbidden_entities,
            provider=provider,
            host_model=host_model,
            generated_at=generated_at,
            fingerprint=fingerprint,
            status="ok",
            warnings=warnings,
        ),
    )


def _generate_profile_naming_locally(rpp_output: dict, sources: list[dict]) -> dict:
    directions = [
        {
            "direction_idx": source["direction_idx"],
            "direction_id": source["direction_id"],
            "name": f"rpp_direction_{source['direction_idx']}",
            "reason": (
                "Placeholder only: host model naming is not connected in v0, "
                "so no semantic profile target was generated."
            ),
            "confidence": 0.0,
            "evidence_refs": [],
            "source": source,
        }
        for source in sources
    ]

    return {
        "rpp_run_id": _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id"))),
        "profile_name": "rpp_profile_pending_host_model",
        "profile_summary": (
            "Placeholder only. Connect the EdgeStudio host model before using "
            "this output as a profile naming target."
        ),
        "directions": directions,
        "status": "stub_pending_host_model",
    }


def _generate_route_action_pairs_locally(rpp_output: dict, case_sources: list[dict]) -> dict:
    return {
        "rpp_run_id": _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id"))),
        "status": "stub_pending_host_model",
        "pairs": [],
        "coverage": {
            "eval_case_count": len(case_sources),
            "pair_count": 0,
        },
        "note": (
            "Placeholder only. Route/action pairs must be generated by the "
            "EdgeStudio host model before use as training data."
        ),
    }


def _review_hard_fact_leakage_locally(
    sample_sources: list[dict],
    forbidden_entities: list[str],
) -> dict:
    return {
        "status": "stub_pending_host_model",
        "decision": "not_reviewed",
        "review_items": [],
        "summary": {
            "sample_count": len(sample_sources),
            "reviewed_count": 0,
            "forbidden_entity_count": len(forbidden_entities),
            "leakage_count": None,
        },
        "note": (
            "Placeholder only. Hard-fact leakage review must be completed by "
            "the EdgeStudio host model and deterministic gates before use."
        ),
    }


def _is_host_model_provider(provider: Any) -> bool:
    return isinstance(provider, str) and provider in HOST_MODEL_PROVIDERS


def _supported_providers_list() -> list[str]:
    return [PROVIDER, HOST_MODEL_PROVIDER, LEGACY_HOST_MODEL_PROVIDER]


def _generate_profile_naming_with_host_model(
    *,
    rpp_output: dict,
    forbidden_entities: list[str],
    sources: list[dict],
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
) -> tuple[dict, dict[str, Any]]:
    messages = _build_profile_naming_messages(
        rpp_output=rpp_output,
        forbidden_entities=forbidden_entities,
        sources=sources,
    )

    try:
        raw = _call_host_model(
            messages,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
        )
    except HostModelAssistantError:
        raise
    except Exception as exc:
        raise HostModelAssistantError(
            code="host_model_call_failed",
            message="Host model profile naming call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host model returned an empty profile naming response.",
            retryable=True,
            details={"model_id": model_id},
        )

    try:
        payload = json.loads(_extract_json_block(raw_output))
    except json.JSONDecodeError as exc:
        raise HostModelAssistantError(
            code="host_model_parse_error",
            message="Host model profile naming response was not valid JSON.",
            retryable=False,
            details={
                "model_id": model_id,
                "reason": str(exc),
                "raw_output_fingerprint": _fingerprint(raw_output),
            },
        ) from exc

    result = _normalize_host_model_profile_naming(payload, rpp_output, sources)
    host_model = {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _fingerprint(raw_output),
    }
    return result, host_model


def _generate_route_action_pairs_with_host_model(
    *,
    rpp_output: dict,
    eval_cases: list[dict],
    case_sources: list[dict],
    tool_registry: list[dict[str, Any]] | None,
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
) -> tuple[dict, dict[str, Any]]:
    messages = _build_route_action_messages(
        rpp_output=rpp_output,
        eval_cases=eval_cases,
        case_sources=case_sources,
        tool_registry=tool_registry,
    )

    try:
        raw = _call_host_model(
            messages,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=_route_action_output_token_budget(case_sources),
            temperature=0.0,
        )
    except HostModelAssistantError:
        raise
    except Exception as exc:
        raise HostModelAssistantError(
            code="host_model_call_failed",
            message="Host model route/action pair call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host model returned an empty route/action response.",
            retryable=True,
            details={"model_id": model_id},
        )

    try:
        payload = json.loads(_extract_json_block(raw_output))
    except json.JSONDecodeError as exc:
        raise HostModelAssistantError(
            code="host_model_parse_error",
            message="Host model route/action response was not valid JSON.",
            retryable=False,
            details={
                "model_id": model_id,
                "reason": str(exc),
                "raw_output_fingerprint": _fingerprint(raw_output),
            },
        ) from exc

    result = _normalize_host_model_route_action_pairs(
        payload,
        rpp_output,
        case_sources,
        tool_registry=tool_registry,
    )
    host_model = {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _fingerprint(raw_output),
    }
    return result, host_model


def _review_hard_fact_leakage_with_host_model(
    *,
    samples: list[dict],
    forbidden_entities: list[str],
    sample_sources: list[dict],
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
) -> tuple[dict, dict[str, Any]]:
    messages = _build_leakage_review_messages(
        samples=samples,
        forbidden_entities=forbidden_entities,
        sample_sources=sample_sources,
    )

    try:
        raw = _call_host_model(
            messages,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=_leakage_review_output_token_budget(sample_sources),
            temperature=0.0,
        )
    except HostModelAssistantError:
        raise
    except Exception as exc:
        raise HostModelAssistantError(
            code="host_model_call_failed",
            message="Host model hard-fact leakage review call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host model returned an empty leakage review response.",
            retryable=True,
            details={"model_id": model_id},
        )

    try:
        payload = json.loads(_extract_json_block(raw_output))
    except json.JSONDecodeError as exc:
        raise HostModelAssistantError(
            code="host_model_parse_error",
            message="Host model leakage review response was not valid JSON.",
            retryable=False,
            details={
                "model_id": model_id,
                "reason": str(exc),
                "raw_output_fingerprint": _fingerprint(raw_output),
            },
        ) from exc

    result = _normalize_host_model_leakage_review(payload, sample_sources, forbidden_entities)
    host_model = {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _fingerprint(raw_output),
    }
    return result, host_model


def _build_profile_naming_messages(
    *,
    rpp_output: dict,
    forbidden_entities: list[str],
    sources: list[dict],
) -> list[dict[str, str]]:
    request = {
        "task": "generate_profile_naming",
        "schema_version": PROFILE_NAMING_SCHEMA_VERSION,
        "rpp_run_id": _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id"))),
        "sources": sources,
        "rpp_output": rpp_output,
        "forbidden_entities": forbidden_entities,
        "output_schema": {
            "profile_name": "short stable label, not a hard fact",
            "profile_summary": "brief stable profile interpretation",
            "directions": [
                {
                    "direction_idx": "int",
                    "direction_id": "string",
                    "name": "short semantic label",
                    "reason": "why the label fits stable RPP evidence",
                    "confidence": "0.0-1.0",
                    "evidence_refs": [],
                }
            ],
        },
    }
    user_content = json.dumps(request, ensure_ascii=False, sort_keys=True, default=repr)
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio host model assistant for personalization. "
                "Name stable RPP profile directions from the provided RPP output. "
                "Do not memorize or reveal exact transactions, exact amounts, dates, "
                "or forbidden entities. Return only one JSON object matching the "
                "requested schema."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _build_route_action_messages(
    *,
    rpp_output: dict,
    eval_cases: list[dict],
    case_sources: list[dict],
    tool_registry: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    selected_tool_names = _route_action_selected_tool_names(case_sources)
    request = {
        "task": "generate_route_action_pairs",
        "schema_version": ROUTE_ACTION_SCHEMA_VERSION,
        "rpp_run_id": _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id"))),
        "case_sources": case_sources,
        "eval_cases": eval_cases,
        "rpp_output": rpp_output,
        "available_tool_contracts": route_action_tool_contracts_for_prompt(
            selected_tool_names or None,
            tool_registry=tool_registry,
        ),
        "route_intent_allowed_values": list(ROUTE_ACTION_ALLOWED_INTENTS),
        "route_intent_guidance": (
            "Use exactly one allowed route_intent value. Tool argument "
            "correctness belongs in selected_tools/tool_call_plan, not in a "
            "new route_intent label."
        ),
        "tool_contract_guidance": (
            "Respect case_sources[*].tool_expectations. selected_tools must "
            "match selected_tools_exact when present, include every "
            "selected_tools_include/tool_name value, and exclude "
            "selected_tools_exclude values. For tools listed in "
            "available_tool_contracts, tool_call_plan args may only use the "
            "declared allowed_args and enum values. Do not invent keys such as "
            "aggregate/include_narrative; counts come from tool results."
        ),
        "output_schema": {
            "pairs": [
                {
                    "case_id": "stable eval case id",
                    "route_intent": "one of route_intent_allowed_values",
                    "prompt_variants": [
                        (
                            "entity-free paraphrase prompts for training; do not "
                            "repeat eval prompt facts such as merchant, amount, "
                            "date/time, order/account IDs, or receipt details"
                        )
                    ],
                    "selected_tools": ["tool names, may be empty"],
                    "tool_call_plan": [
                        {
                            "tool_name": "one selected tool name",
                            "args": "object using only declared allowed_args",
                            "reason": "why this tool call is needed",
                        }
                    ],
                    "answer_discipline": "how to answer while leaving exact facts to tools",
                    "rationale": "why this route/action pair fits",
                    "confidence": "0.0-1.0",
                }
            ]
        },
    }
    user_content = json.dumps(request, ensure_ascii=False, sort_keys=True, default=repr)
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio host model assistant for personalization. "
                "Generate route/action training pairs for eval cases using stable "
                "RPP evidence. Do not invent exact facts; precise factual values "
                "must remain delegated to deterministic Fact/tool execution. "
                "Source eval prompts may contain user-specific hard facts. "
                "Do not copy merchant names, exact amounts, exact dates/times, "
                "order/account IDs, or one-off receipt details into prompt_variants; "
                "preserve only the route/action signal with generic wording. "
                "Use only the requested tool expectations and provided tool "
                "contracts; do not invent tool argument keys. "
                "Return only one JSON object matching the requested schema."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _build_leakage_review_messages(
    *,
    samples: list[dict],
    forbidden_entities: list[str],
    sample_sources: list[dict],
) -> list[dict[str, str]]:
    request = {
        "task": "review_hard_fact_leakage",
        "schema_version": LEAKAGE_REVIEW_SCHEMA_VERSION,
        "sample_sources": sample_sources,
        "samples": samples,
        "forbidden_entities": forbidden_entities,
        "output_schema": {
            "decision": "pass | fail | needs_human_review",
            "review_items": [
                {
                    "sample_id": "stable sample id",
                    "leakage_detected": "bool",
                    "severity": "none | low | medium | high",
                    "matched_entities": [
                        "specific forbidden entities or named hard facts only"
                    ],
                    "rationale": (
                        "why this sample is safe or unsafe under the hard-fact "
                        "definition"
                    ),
                }
            ],
            "summary": {
                "sample_count": "int",
                "reviewed_count": "int",
                "forbidden_entity_count": "int",
                "leakage_count": "int",
            },
        },
    }
    user_content = json.dumps(request, ensure_ascii=False, sort_keys=True, default=repr)
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio host model assistant for personalization. "
                "Review candidate training samples for hard-fact leakage against "
                "the provided forbidden entities. If no forbidden entities are "
                "provided, still flag only named hard facts: specific merchant or "
                "place names, exact amounts, exact dates, account/order ids, or "
                "one-off receipt details. Do not flag broad categories, stable "
                "persona descriptors, route intents, tool names, or tool argument "
                "shapes by themselves. Do not rewrite or redact samples; return a "
                "structured review only. Return one JSON object matching the "
                "requested schema."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def _call_host_model(
    messages: list[dict[str, str]],
    *,
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
    max_tokens: int = 700,
    temperature: float = 0.0,
) -> Any:
    if host_model_generate is not None:
        return host_model_generate(messages, max_tokens, temperature)

    from backend.services.host_llm_runtime import HostLLMRuntimeError, generate_host_chat

    try:
        return generate_host_chat(
            messages=messages,
            host_model_id=host_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        ).to_dict()
    except HostLLMRuntimeError as exc:
        raise HostModelAssistantError(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            details=exc.details,
        ) from exc


def _route_action_output_token_budget(case_sources: list[dict]) -> int:
    # One pair contains a prompt variant, route/tool plan, answer discipline,
    # rationale, and provenance. Keep the lower bound for old tests/simple
    # cases, then scale with the number of eval cases.
    return _bounded_output_token_budget(
        item_count=len(case_sources),
        base_tokens=600,
        tokens_per_item=450,
        floor=1200,
    )


def _leakage_review_output_token_budget(sample_sources: list[dict]) -> int:
    # Review output is O(samples): each item needs sample_id, verdict, severity,
    # matched_entities, and a short rationale. This is an output budget, not the
    # model context window; callers still batch large review sets.
    return _bounded_output_token_budget(
        item_count=len(sample_sources),
        base_tokens=400,
        tokens_per_item=280,
        floor=1200,
    )


def _bounded_output_token_budget(
    *,
    item_count: int,
    base_tokens: int,
    tokens_per_item: int,
    floor: int,
) -> int:
    requested = base_tokens + max(0, item_count) * tokens_per_item
    return min(HOST_MODEL_MAX_OUTPUT_TOKENS, max(floor, requested))


def _host_model_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        output = value.get("output")
        return output.strip() if isinstance(output, str) else ""
    output = getattr(value, "output", "")
    return output.strip() if isinstance(output, str) else ""


def _host_model_id(value: Any) -> str | None:
    if isinstance(value, dict):
        return _optional_str(value.get("model_id") or value.get("model_path"))
    return _optional_str(getattr(value, "model_path", None) or getattr(value, "model_id", None))


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return stripped[start_obj:end_obj + 1]
    return stripped


def _normalize_host_model_profile_naming(
    payload: Any,
    rpp_output: dict,
    sources: list[dict],
) -> dict:
    if not isinstance(payload, dict):
        raise HostModelAssistantError(
            code="host_model_schema_error",
            message="Host model profile naming JSON must be an object.",
            retryable=False,
            details={"received_type": type(payload).__name__, "model_id": None},
        )

    direction_payloads = payload.get("directions")
    if not isinstance(direction_payloads, list):
        raise HostModelAssistantError(
            code="host_model_schema_error",
            message="Host model profile naming JSON missing directions list.",
            retryable=False,
            details={"field": "directions", "model_id": None},
        )

    by_idx: dict[int, dict] = {}
    for entry in direction_payloads:
        if not isinstance(entry, dict):
            continue
        idx = _optional_int(entry.get("direction_idx"))
        if idx is not None:
            by_idx[idx] = entry

    directions: list[dict] = []
    for index, source in enumerate(sources):
        entry = by_idx.get(source["direction_idx"])
        if entry is None and index < len(direction_payloads) and isinstance(direction_payloads[index], dict):
            entry = direction_payloads[index]
        if entry is None:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model omitted a required RPP direction.",
                retryable=False,
                details={"direction_idx": source["direction_idx"], "model_id": None},
            )

        name = _optional_str(entry.get("name"))
        reason = _optional_str(entry.get("reason"))
        if not name or not reason:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model direction must include non-empty name and reason.",
                retryable=False,
                details={"direction_idx": source["direction_idx"], "model_id": None},
            )

        directions.append({
            "direction_idx": source["direction_idx"],
            "direction_id": _optional_str(entry.get("direction_id")) or source["direction_id"],
            "name": name,
            "reason": reason,
            "confidence": _confidence(entry.get("confidence")),
            "evidence_refs": entry.get("evidence_refs") if isinstance(entry.get("evidence_refs"), list) else [],
            "source": source,
        })

    return {
        "rpp_run_id": _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id"))),
        "profile_name": _optional_str(payload.get("profile_name")) or "rpp_profile_named_by_host_model",
        "profile_summary": _optional_str(payload.get("profile_summary")) or "",
        "directions": directions,
        "status": "host_model_generated",
    }


def _normalize_host_model_route_action_pairs(
    payload: Any,
    rpp_output: dict,
    case_sources: list[dict],
    *,
    tool_registry: list[dict[str, Any]] | None = None,
) -> dict:
    if not isinstance(payload, dict):
        raise HostModelAssistantError(
            code="host_model_schema_error",
            message="Host model route/action JSON must be an object.",
            retryable=False,
            details={"received_type": type(payload).__name__, "model_id": None},
        )

    pair_payloads = payload.get("pairs")
    if not isinstance(pair_payloads, list):
        raise HostModelAssistantError(
            code="host_model_schema_error",
            message="Host model route/action JSON missing pairs list.",
            retryable=False,
            details={"field": "pairs", "model_id": None},
        )

    by_case_id = {
        case_id: entry
        for entry in pair_payloads
        if isinstance(entry, dict)
        for case_id in [_optional_str(entry.get("case_id"))]
        if case_id
    }

    pairs: list[dict] = []
    for index, source in enumerate(case_sources):
        entry = by_case_id.get(source["case_id"])
        if entry is None and index < len(pair_payloads) and isinstance(pair_payloads[index], dict):
            entry = pair_payloads[index]
        if entry is None:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model omitted a required eval case pair.",
                retryable=False,
                details={"case_id": source["case_id"], "model_id": None},
            )

        route_intent = _optional_str(_first_present(entry, ("route_intent", "intent")))
        answer_discipline = _optional_str(entry.get("answer_discipline"))
        rationale = _optional_str(_first_present(entry, ("rationale", "reason")))
        if not route_intent or not answer_discipline or not rationale:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message=(
                    "Host model route/action pair must include route_intent, "
                    "answer_discipline, and rationale."
                ),
                retryable=False,
                details={"case_id": source["case_id"], "model_id": None},
            )
        canonical_route_intent = _canonical_route_action_intent(route_intent)
        if canonical_route_intent is None:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message=(
                    "Host model route/action pair route_intent must be one of "
                    "route_intent_allowed_values."
                ),
                retryable=False,
                details={
                    "case_id": source["case_id"],
                    "route_intent": route_intent,
                    "allowed_route_intents": list(ROUTE_ACTION_ALLOWED_INTENTS),
                    "model_id": None,
                },
            )

        training_prompts = _route_action_training_prompts(entry, source["prompt"])
        if not training_prompts:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message=(
                    "Host model route/action pair must include at least one "
                    "prompt_variant that does not exactly replay the eval prompt."
                ),
                retryable=False,
                details={"case_id": source["case_id"], "model_id": None},
            )

        selected_tools = _string_list(entry.get("selected_tools"))
        tool_call_plan = _list_of_dicts(entry.get("tool_call_plan"))
        tool_expectations = (
            source.get("tool_expectations")
            if isinstance(source.get("tool_expectations"), dict)
            else {}
        )
        try:
            validate_route_action_tool_contracts(
                selected_tools=selected_tools,
                tool_call_plan=tool_call_plan,
                case_id=source["case_id"],
                required_tools=tool_expectations.get("required_tools"),
                exact_tools=(
                    tool_expectations.get("exact_tools")
                    if tool_expectations.get("has_exact_tools") is True
                    else None
                ),
                excluded_tools=tool_expectations.get("excluded_tools"),
                tool_registry=tool_registry,
            )
        except ValueError as exc:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model route/action pair violated tool contract.",
                retryable=False,
                details={
                    "case_id": source["case_id"],
                    "reason": str(exc),
                    "model_id": None,
                },
            ) from exc

        pairs.append({
            "case_idx": source["case_idx"],
            "case_id": _optional_str(entry.get("case_id")) or source["case_id"],
            "prompt": training_prompts[0],
            "route_intent": canonical_route_intent,
            "prompt_variants": training_prompts[1:],
            "selected_tools": selected_tools,
            "tool_call_plan": tool_call_plan,
            "answer_discipline": answer_discipline,
            "rationale": rationale,
            "confidence": _confidence(entry.get("confidence")),
            "source": source,
        })

    return {
        "rpp_run_id": _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id"))),
        "status": "host_model_generated",
        "pairs": pairs,
        "coverage": {
            "eval_case_count": len(case_sources),
            "pair_count": len(pairs),
        },
    }


def _normalize_host_model_leakage_review(
    payload: Any,
    sample_sources: list[dict],
    forbidden_entities: list[str],
) -> dict:
    if not isinstance(payload, dict):
        raise HostModelAssistantError(
            code="host_model_schema_error",
            message="Host model leakage review JSON must be an object.",
            retryable=False,
            details={"received_type": type(payload).__name__, "model_id": None},
        )

    review_payloads = payload.get("review_items")
    if not isinstance(review_payloads, list):
        raise HostModelAssistantError(
            code="host_model_schema_error",
            message="Host model leakage review JSON missing review_items list.",
            retryable=False,
            details={"field": "review_items", "model_id": None},
        )

    by_sample_id: dict[str, dict] = {}
    for entry in review_payloads:
        if not isinstance(entry, dict):
            continue
        sample_id = _optional_str(entry.get("sample_id"))
        if not sample_id:
            continue
        if sample_id in by_sample_id:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model leakage review duplicated a sample_id.",
                retryable=False,
                details={"sample_id": sample_id, "model_id": None},
            )
        by_sample_id[sample_id] = entry

    review_items: list[dict] = []
    for index, source in enumerate(sample_sources):
        entry = by_sample_id.get(source["sample_id"])
        if entry is None and index < len(review_payloads) and isinstance(review_payloads[index], dict):
            candidate = review_payloads[index]
            candidate_sample_id = _optional_str(candidate.get("sample_id"))
            if candidate_sample_id and candidate_sample_id != source["sample_id"]:
                raise HostModelAssistantError(
                    code="host_model_schema_error",
                    message="Host model leakage review sample_id did not match the requested sample.",
                    retryable=False,
                    details={"sample_id": source["sample_id"], "model_id": None},
                )
            entry = candidate
        if entry is None:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model omitted a required leakage review item.",
                retryable=False,
                details={"sample_id": source["sample_id"], "model_id": None},
            )

        rationale = _optional_str(_first_present(entry, ("rationale", "reason")))
        if not rationale:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model leakage review item must include rationale.",
                retryable=False,
                details={"sample_id": source["sample_id"], "model_id": None},
            )

        entry_sample_id = _optional_str(entry.get("sample_id"))
        if entry_sample_id and entry_sample_id != source["sample_id"]:
            raise HostModelAssistantError(
                code="host_model_schema_error",
                message="Host model leakage review sample_id did not match the requested sample.",
                retryable=False,
                details={"sample_id": source["sample_id"], "model_id": None},
            )

        leakage_detected = bool(entry.get("leakage_detected"))
        severity = _severity(entry.get("severity"), leakage_detected=leakage_detected)
        review_items.append({
            "sample_idx": source["sample_idx"],
            "sample_id": source["sample_id"],
            "leakage_detected": leakage_detected,
            "severity": severity,
            "matched_entities": _string_list(entry.get("matched_entities")),
            "rationale": rationale,
            "source": source,
        })

    leakage_count = sum(1 for item in review_items if item["leakage_detected"])
    decision = _decision(payload.get("decision"), leakage_count=leakage_count)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "status": "host_model_reviewed",
        "decision": decision,
        "review_items": review_items,
        "summary": {
            "sample_count": len(sample_sources),
            "reviewed_count": len(review_items),
            "forbidden_entity_count": len(forbidden_entities),
            "leakage_count": leakage_count,
        },
    }


def _response(
    *,
    ok: bool,
    result: dict | None,
    error: dict | None,
    audit: dict,
    schema_version: str = PROFILE_NAMING_SCHEMA_VERSION,
) -> dict:
    return {
        "ok": ok,
        "schema_version": schema_version,
        "result": result,
        "error": error,
        "audit": audit,
    }


def _route_action_response(
    *,
    ok: bool,
    result: dict | None,
    error: dict | None,
    audit: dict,
) -> dict:
    return _response(
        ok=ok,
        result=result,
        error=error,
        audit=audit,
        schema_version=ROUTE_ACTION_SCHEMA_VERSION,
    )


def _leakage_review_response(
    *,
    ok: bool,
    result: dict | None,
    error: dict | None,
    audit: dict,
) -> dict:
    return _response(
        ok=ok,
        result=result,
        error=error,
        audit=audit,
        schema_version=LEAKAGE_REVIEW_SCHEMA_VERSION,
    )


def _audit(
    value: Any,
    *,
    forbidden_entities: Any,
    provider: str,
    host_model: dict[str, Any] | None = None,
    generated_at: str,
    fingerprint: str,
    status: str,
    warnings: list[str],
    method: str = "generate_profile_naming",
    input_summary: dict | None = None,
) -> dict:
    host_model_info = host_model or {
        "enabled": _is_host_model_provider(provider),
        "model_id": None,
    }

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "assistant_version": HOST_MODEL_ASSISTANT_VERSION,
        "method": method,
        "provider": provider,
        "host_model": host_model_info,
        "status": status,
        "generated_at": generated_at,
        "input_fingerprint": fingerprint,
        "input_summary": input_summary or _input_summary(value, forbidden_entities=forbidden_entities),
        "warnings": list(warnings),
        "log_target": "response.audit",
    }


def _route_action_audit(
    rpp_output: Any,
    eval_cases: Any,
    *,
    provider: str,
    host_model: dict[str, Any] | None = None,
    generated_at: str,
    fingerprint: str,
    status: str,
    warnings: list[str],
) -> dict:
    return _audit(
        rpp_output,
        forbidden_entities=None,
        provider=provider,
        host_model=host_model,
        generated_at=generated_at,
        fingerprint=fingerprint,
        status=status,
        warnings=warnings,
        method="generate_route_action_pairs",
        input_summary=_route_action_input_summary(rpp_output, eval_cases),
    )


def _leakage_review_audit(
    samples: Any,
    forbidden_entities: Any,
    *,
    provider: str,
    host_model: dict[str, Any] | None = None,
    generated_at: str,
    fingerprint: str,
    status: str,
    warnings: list[str],
) -> dict:
    return _audit(
        samples,
        forbidden_entities=forbidden_entities,
        provider=provider,
        host_model=host_model,
        generated_at=generated_at,
        fingerprint=fingerprint,
        status=status,
        warnings=warnings,
        method="review_hard_fact_leakage",
        input_summary=_leakage_review_input_summary(samples, forbidden_entities),
    )


def _host_model_audit(*, enabled: bool, selected_model_id: str | None) -> dict[str, Any]:
    audit = {
        "enabled": enabled,
        "model_id": None,
    }
    if selected_model_id is not None:
        audit["selected_model_id"] = selected_model_id
    return audit


def _input_summary(value: Any, *, forbidden_entities: Any) -> dict:
    forbidden_summary = _forbidden_entities_summary(forbidden_entities)

    if not isinstance(value, dict):
        return {
            "top_level_keys": [],
            "rpp_run_id": None,
            "base_model_id": None,
            "layer_id": None,
            "direction_count": 0,
            "dataset_count": None,
            "forbidden_entities": forbidden_summary,
        }

    try:
        direction_count = len(_extract_direction_sources(value))
    except (TypeError, ValueError):
        direction_count = 0

    return {
        "top_level_keys": sorted(str(key) for key in value.keys()),
        "rpp_run_id": _optional_str(_first_present(value, ("rpp_run_id", "run_id"))),
        "base_model_id": _optional_str(_first_present(value, ("base_model_id", "model_id", "model_path"))),
        "layer_id": _optional_int(_first_present(value, ("layer_id", "layer_idx"))),
        "direction_count": direction_count,
        "dataset_count": _dataset_count(value),
        "forbidden_entities": forbidden_summary,
    }


def _route_action_input_summary(rpp_output: Any, eval_cases: Any) -> dict:
    if isinstance(rpp_output, dict):
        try:
            direction_count = len(_extract_direction_sources(rpp_output))
        except (TypeError, ValueError):
            direction_count = 0
        top_level_keys = sorted(str(key) for key in rpp_output.keys())
        rpp_run_id = _optional_str(_first_present(rpp_output, ("rpp_run_id", "run_id")))
        base_model_id = _optional_str(_first_present(rpp_output, ("base_model_id", "model_id", "model_path")))
        layer_id = _optional_int(_first_present(rpp_output, ("layer_id", "layer_idx")))
    else:
        direction_count = 0
        top_level_keys = []
        rpp_run_id = None
        base_model_id = None
        layer_id = None

    try:
        eval_case_count = len(_extract_eval_case_sources(eval_cases))
        eval_cases_valid = True
    except TypeError:
        eval_case_count = 0
        eval_cases_valid = False

    return {
        "top_level_keys": top_level_keys,
        "rpp_run_id": rpp_run_id,
        "base_model_id": base_model_id,
        "layer_id": layer_id,
        "direction_count": direction_count,
        "eval_case_count": eval_case_count,
        "eval_cases_valid": eval_cases_valid,
        "eval_cases_fingerprint": _fingerprint(eval_cases) if eval_cases_valid else None,
    }


def _extract_eval_case_sources(eval_cases: Any) -> list[dict]:
    if not isinstance(eval_cases, list):
        raise TypeError("eval_cases is not a list")

    sources: list[dict] = []
    for index, case in enumerate(eval_cases):
        if not isinstance(case, dict):
            raise TypeError(f"eval_cases[{index}] is {type(case).__name__}, not dict")

        case_id = _optional_str(_first_present(case, ("case_id", "id", "key")))
        prompt = _optional_str(_first_present(case, ("prompt", "query", "user_message", "input")))
        source = {
            "case_idx": index + 1,
            "case_id": case_id or f"eval_case_{index + 1}",
            "prompt": prompt or "",
            "input_path": f"eval_cases[{index}]",
        }
        expectations = (
            case.get("expectations")
            if isinstance(case.get("expectations"), dict)
            else {}
        )
        tool_expectations = _route_action_tool_expectations(expectations)
        if tool_expectations:
            source["tool_expectations"] = tool_expectations
        sources.append(source)
    return sources


def _route_action_tool_expectations(expectations: dict[str, Any]) -> dict[str, Any]:
    required = _string_list(expectations.get("selected_tools_include"))
    exact_raw = expectations.get("selected_tools_exact")
    has_exact = isinstance(exact_raw, list)
    exact = _string_list(exact_raw) if has_exact else []
    excluded = _string_list(expectations.get("selected_tools_exclude"))
    tool_name = _optional_str(expectations.get("tool_name"))
    if tool_name:
        required.append(tool_name)

    required = _dedupe_strings(required)
    exact = _dedupe_strings(exact)
    excluded = _dedupe_strings(excluded)

    out: dict[str, Any] = {}
    if required:
        out["required_tools"] = required
    if has_exact:
        out["has_exact_tools"] = True
        out["exact_tools"] = exact
    if excluded:
        out["excluded_tools"] = excluded
    return out


def _route_action_selected_tool_names(case_sources: list[dict]) -> list[str]:
    names: list[str] = []
    for source in case_sources:
        expectations = (
            source.get("tool_expectations")
            if isinstance(source.get("tool_expectations"), dict)
            else {}
        )
        names.extend(_string_list(expectations.get("required_tools")))
        names.extend(_string_list(expectations.get("exact_tools")))
    return _dedupe_strings(names)


def _leakage_review_input_summary(samples: Any, forbidden_entities: Any) -> dict:
    try:
        sample_count = len(_extract_sample_sources(samples))
        samples_valid = True
    except TypeError:
        sample_count = 0
        samples_valid = False

    try:
        entities = _normalize_forbidden_entities(forbidden_entities)
        entities_valid = True
    except TypeError:
        entities = []
        entities_valid = False

    return {
        "sample_count": sample_count,
        "samples_valid": samples_valid,
        "samples_fingerprint": _fingerprint(samples) if samples_valid else None,
        "forbidden_entities": {
            "provided": forbidden_entities is not None,
            "valid": entities_valid,
            "count": len(entities) if entities_valid else None,
            "fingerprint": _fingerprint(entities) if entities else None,
        },
    }


def _extract_sample_sources(samples: Any) -> list[dict]:
    if not isinstance(samples, list):
        raise TypeError("samples is not a list")

    sources: list[dict] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"samples[{index}] is {type(sample).__name__}, not dict")

        sample_id = _optional_str(_first_present(sample, ("sample_id", "id", "key")))
        source = sample.get("source") if isinstance(sample.get("source"), dict) else {}
        item = {
            "sample_idx": index + 1,
            "sample_id": sample_id or f"sample_{index + 1}",
            "input_path": f"samples[{index}]",
        }
        line = _optional_int(source.get("line"))
        if line is not None and line > 0:
            item["line_no"] = line
        file_name = _optional_str(source.get("file_name"))
        if file_name:
            item["file_name"] = file_name
        sample_fingerprint = _optional_str(source.get("sample_fingerprint"))
        if sample_fingerprint:
            item["sample_fingerprint"] = sample_fingerprint
        sources.append(item)
    return sources


def _extract_direction_sources(rpp_output: dict) -> list[dict]:
    explicit = _first_present(rpp_output, ("directions", "components", "B_naming", "b_naming"))
    if isinstance(explicit, list):
        sources = [
            _source_from_entry(entry, index)
            for index, entry in enumerate(explicit)
        ]
        return _limit_by_k_selected(rpp_output, sources)

    bootstrap = rpp_output.get("bootstrap", {})
    if bootstrap is not None and not isinstance(bootstrap, dict):
        raise TypeError("bootstrap must be a dict when provided.")

    verdict = (bootstrap or {}).get("verdict")
    if isinstance(verdict, list):
        sources = [
            _source_from_entry(entry, index, input_path=f"bootstrap.verdict[{index}]")
            for index, entry in enumerate(verdict)
        ]
        return _limit_by_k_selected(rpp_output, sources)
    if verdict is not None:
        raise TypeError("bootstrap.verdict must be a list when provided.")

    k_selected = _optional_int(rpp_output.get("k_selected_after_fallback"))
    if k_selected is None or k_selected <= 0:
        return []

    return [
        {
            "direction_idx": index + 1,
            "direction_id": f"u_{index + 1}",
            "component_idx": index,
            "input_path": "k_selected_after_fallback",
        }
        for index in range(k_selected)
    ]


def _source_from_entry(entry: Any, index: int, *, input_path: str | None = None) -> dict:
    if isinstance(entry, dict):
        if "direction_idx" in entry:
            direction_idx = _normalize_one_based_idx(entry["direction_idx"], fallback=index + 1)
        elif "component_idx" in entry:
            component_idx = _optional_int(entry["component_idx"])
            direction_idx = (component_idx + 1) if component_idx is not None and component_idx >= 0 else index + 1
        else:
            raw_idx = _first_present(entry, ("idx", "index"))
            direction_idx = _normalize_one_based_idx(raw_idx, fallback=index + 1)
        direction_id = _optional_str(_first_present(entry, ("direction_id", "id", "key", "name")))
        return {
            "direction_idx": direction_idx,
            "direction_id": direction_id or f"u_{direction_idx}",
            "component_idx": _optional_int(entry.get("component_idx")),
            "input_path": input_path or f"directions[{index}]",
        }

    direction_idx = index + 1
    return {
        "direction_idx": direction_idx,
        "direction_id": f"u_{direction_idx}",
        "component_idx": index,
        "input_path": input_path or f"directions[{index}]",
    }


def _normalize_one_based_idx(value: Any, *, fallback: int) -> int:
    parsed = _optional_int(value)
    if parsed is None:
        return fallback
    if isinstance(value, str) and value.strip().startswith("u_"):
        return parsed
    if parsed <= 0:
        return fallback
    return parsed


def _limit_by_k_selected(rpp_output: dict, sources: list[dict]) -> list[dict]:
    k_selected = _optional_int(rpp_output.get("k_selected_after_fallback"))
    if k_selected is None or k_selected < 0:
        return sources
    return sources[:k_selected]


def _dataset_count(rpp_output: dict) -> int | None:
    for key in ("n_transactions", "event_count", "sample_count"):
        value = _optional_int(rpp_output.get(key))
        if value is not None:
            return value

    dataset_summary = rpp_output.get("dataset_summary")
    if isinstance(dataset_summary, dict):
        for key in ("event_count", "sample_count", "n_transactions", "count"):
            value = _optional_int(dataset_summary.get(key))
            if value is not None:
                return value

    return None


def _input_fingerprint(
    rpp_output: Any,
    forbidden_entities: Any,
    *,
    provider: Any,
    host_model_id: Any,
) -> str:
    return _fingerprint({
        "rpp_output": rpp_output,
        "forbidden_entities": _fingerprintable_forbidden_entities(forbidden_entities),
        "provider": provider,
        "host_model_id": host_model_id,
    })


def _route_action_input_fingerprint(
    rpp_output: Any,
    eval_cases: Any,
    *,
    tool_registry: Any,
    provider: Any,
    host_model_id: Any,
) -> str:
    return _fingerprint({
        "rpp_output": rpp_output,
        "eval_cases": eval_cases,
        "tool_registry": tool_registry,
        "provider": provider,
        "host_model_id": host_model_id,
    })


def _leakage_review_input_fingerprint(
    samples: Any,
    forbidden_entities: Any,
    *,
    provider: Any,
    host_model_id: Any,
) -> str:
    return _fingerprint({
        "samples": samples,
        "forbidden_entities": _fingerprintable_forbidden_entities(forbidden_entities),
        "provider": provider,
        "host_model_id": host_model_id,
    })


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_forbidden_entities(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("not a list")

    entities: list[str] = []
    for index, entity in enumerate(value):
        if not isinstance(entity, str):
            raise TypeError(f"entry {index} is {type(entity).__name__}, not str")
        text = entity.strip()
        if text:
            entities.append(text)
    return entities


def _fingerprintable_forbidden_entities(value: Any) -> Any:
    try:
        return _normalize_forbidden_entities(value)
    except TypeError:
        return value


def _forbidden_entities_summary(value: Any) -> dict:
    try:
        entities = _normalize_forbidden_entities(value)
    except TypeError:
        return {
            "provided": value is not None,
            "valid": False,
            "count": None,
            "fingerprint": None,
        }

    return {
        "provided": value is not None,
        "valid": True,
        "count": len(entities),
        "fingerprint": _fingerprint(entities) if entities else None,
    }


def _first_present(mapping: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("u_"):
            stripped = stripped[2:]
        if stripped.isdigit():
            return int(stripped)
    return None


def _canonical_route_action_intent(value: Any) -> str | None:
    normalized = (
        str(value or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return normalized if normalized in _ROUTE_ACTION_ALLOWED_INTENT_SET else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            return 0.0
    return 0.0


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in (_optional_str(item) for item in value) if item]


def _dedupe_strings(value: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _prompt_variants(value: Any, prompt: str) -> list[str]:
    variants: list[str] = []
    seen = {_route_prompt_key(prompt)}
    for item in _string_list(value):
        key = _route_prompt_key(item)
        if key in seen:
            continue
        variants.append(item)
        seen.add(key)
    return variants


def _route_action_training_prompts(entry: dict, source_prompt: str) -> list[str]:
    """Host model may propose paraphrases; source eval prompt stays audit-only."""

    return _prompt_variants(entry.get("prompt_variants"), source_prompt)


def _route_prompt_key(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split()).rstrip("?!。！？.")


def _list_of_dicts(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _severity(value: Any, *, leakage_detected: bool) -> str:
    severity = _optional_str(value)
    if severity in {"none", "low", "medium", "high"}:
        return severity
    return "medium" if leakage_detected else "none"


def _decision(value: Any, *, leakage_count: int) -> str:
    decision = _optional_str(value)
    if decision in {"pass", "fail", "needs_human_review"}:
        return decision
    return "fail" if leakage_count > 0 else "pass"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
