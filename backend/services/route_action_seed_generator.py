# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-model route/action seed generation for learned-router data coverage.

This module is training-side only. It helps EdgeStudio propose route/action
seed candidates from a developer-provided tool registry plus a small set of
golden cases, then pushes the proposal through the existing route/action
normalization and leakage gates. It never writes events or runtime artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from backend.services.host_model_assistant import (
    HOST_MODEL_PROVIDER,
    HOST_MODEL_PROVIDERS,
    PROVIDER,
    ROUTE_ACTION_ALLOWED_INTENTS,
    SUPPORTED_PROVIDERS,
    HostModelAssistantError,
    HostModelGenerate,
    _call_host_model,
    _extract_json_block,
    _host_model_id,
    _host_model_output_text,
    generate_route_action_pairs,
)
from backend.services.route_action_training_events import (
    ROUTE_ACTION_EVENT_TYPE,
    build_route_action_training_events,
)
from backend.services.route_action_tool_contracts import route_action_tool_contracts_for_prompt


ROUTE_ACTION_SEED_CANDIDATES_SCHEMA_VERSION = (
    "edgestudio.route_action_seed_candidates.v0"
)

ROUTE_ACTION_INTENT_GUIDANCE: dict[str, str] = {
    "base_chat": "General assistant response that does not need app facts or app actions.",
    "exact_fact": "Concrete stored records or exact factual lookup from app data.",
    "aggregate_fact": "Totals, counts, rankings, trends, comparisons, or grouped summaries computed from app records.",
    "app_action": "A user request to mutate app state or perform an app-defined command.",
    "user_profile": "Stable preferences, habits, style, or profile summaries; not exact totals, rankings, or record breakdowns.",
    "mixed": "Multiple route types are genuinely required and cannot be represented by one more specific intent.",
}


class RouteActionSeedGeneratorError(ValueError):
    """Raised when a host-model seed proposal fails validation."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


def generate_route_action_seed_candidates(
    *,
    app_id: str,
    tool_registry: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    rpp_output: dict[str, Any] | None = None,
    target_seed_count: int = 24,
    seed_run_id: str | None = None,
    peer_id: str | None = None,
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: HostModelGenerate | None = None,
) -> dict[str, Any]:
    """Generate dry-run route/action seed candidates.

    The host model proposes candidate route/action pairs, but this function
    validates them against the app tool registry, then reuses
    `generate_route_action_pairs(...)` and `build_route_action_training_events`
    so eval-replay leakage and existing tool-contract gates stay in one place.
    """

    generated_at_ms = int(time.time() * 1000)
    selected_provider = provider or PROVIDER
    try:
        normalized_app_id = _required_str(app_id, "app_id")
        tools = _normalize_tool_registry(tool_registry)
        cases = _normalize_golden_cases(golden_cases)
    except RouteActionSeedGeneratorError as exc:
        return _error_envelope(
            status="invalid_seed_request",
            code=exc.code,
            message=str(exc),
            details=exc.details,
            app_id=str(app_id or "").strip(),
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    effective_seed_run_id = _seed_run_id(
        seed_run_id=seed_run_id,
        app_id=normalized_app_id,
        tool_registry=tools,
        golden_cases=cases,
    )
    effective_rpp_output = dict(rpp_output or {})
    if not _optional_str(
        effective_rpp_output.get("rpp_run_id")
        or effective_rpp_output.get("run_id")
    ):
        effective_rpp_output["rpp_run_id"] = effective_seed_run_id

    if not isinstance(selected_provider, str) or selected_provider not in SUPPORTED_PROVIDERS:
        return _error_envelope(
            status="invalid_provider",
            code="invalid_provider",
            message="Unsupported route/action seed provider.",
            details={
                "provider": selected_provider,
                "supported": sorted(SUPPORTED_PROVIDERS),
            },
            app_id=normalized_app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    if selected_provider not in HOST_MODEL_PROVIDERS:
        return {
            "ok": True,
            "schema_version": ROUTE_ACTION_SEED_CANDIDATES_SCHEMA_VERSION,
            "status": "stub_pending_host_model",
            "app_id": normalized_app_id,
            "seed_run_id": effective_seed_run_id,
            "route_action_response": None,
            "preview": None,
            "candidates": [],
            "audit": _audit(
                app_id=normalized_app_id,
                provider=selected_provider,
                host_model={
                    "enabled": False,
                    "model_id": None,
                },
                generated_at_ms=generated_at_ms,
                tool_count=len(tools),
                golden_case_count=len(cases),
                candidate_count=0,
                warnings=["route_action_seed_candidates_stub_pending_host_model"],
            ),
        }

    try:
        seed_payload, host_model = _generate_seed_payload_with_host_model(
            app_id=normalized_app_id,
            tool_registry=tools,
            golden_cases=cases,
            target_seed_count=target_seed_count,
            seed_run_id=effective_seed_run_id,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
        )
        eval_cases, route_pairs, duplicate_candidate_count = (
            _seed_payload_to_route_action_inputs(
                seed_payload,
                tool_registry=tools,
                golden_cases=cases,
            )
        )
    except HostModelAssistantError as exc:
        return _error_envelope(
            status="host_model_failed",
            code=exc.code,
            message=exc.message,
            details=exc.details,
            app_id=normalized_app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )
    except RouteActionSeedGeneratorError as exc:
        return _error_envelope(
            status="host_model_schema_error",
            code=exc.code,
            message=str(exc),
            details=exc.details,
            app_id=normalized_app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    route_action_response = generate_route_action_pairs(
        effective_rpp_output,
        eval_cases,
        tool_registry=tools,
        host_model_id=host_model_id,
        provider=HOST_MODEL_PROVIDER,
        host_model_generate=lambda _messages, _max_tokens, _temperature: {
            "model_id": host_model.get("model_id"),
            "output": json.dumps({"pairs": route_pairs}, ensure_ascii=False),
        },
    )
    if route_action_response.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": ROUTE_ACTION_SEED_CANDIDATES_SCHEMA_VERSION,
            "status": "route_action_normalization_failed",
            "app_id": normalized_app_id,
            "seed_run_id": effective_seed_run_id,
            "route_action_response": route_action_response,
            "preview": None,
            "candidates": [],
            "error": route_action_response.get("error"),
            "audit": _audit(
                app_id=normalized_app_id,
                provider=selected_provider,
                host_model=host_model,
                generated_at_ms=generated_at_ms,
                tool_count=len(tools),
                golden_case_count=len(cases),
                candidate_count=0,
                duplicate_candidate_count=duplicate_candidate_count,
                warnings=["route_action_normalization_failed"],
            ),
        }

    try:
        events = build_route_action_training_events(
            route_action_response,
            peer_id=_optional_str(peer_id) or f"route-seed:{normalized_app_id}",
            timestamp_ms=generated_at_ms,
            tool_registry=tools,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_envelope(
            status="route_action_gate_failed",
            code="route_action_gate_failed",
            message=str(exc),
            details={},
            app_id=normalized_app_id,
            provider=selected_provider,
            host_model_id=host_model_id,
            generated_at_ms=generated_at_ms,
        )

    candidates = route_action_response["result"]["pairs"]
    event_ids = [event.id for event in events]
    return {
        "ok": True,
        "schema_version": ROUTE_ACTION_SEED_CANDIDATES_SCHEMA_VERSION,
        "status": "seed_candidates_ready",
        "app_id": normalized_app_id,
        "seed_run_id": effective_seed_run_id,
        "route_action_response": route_action_response,
        "tool_registry": tools,
        "preview": {
            "event_type": ROUTE_ACTION_EVENT_TYPE,
            "event_count": len(events),
            "would_store": False,
            "event_ids_fingerprint": _fingerprint(event_ids),
        },
        "candidates": candidates,
        "audit": _audit(
            app_id=normalized_app_id,
            provider=selected_provider,
            host_model=host_model,
            generated_at_ms=generated_at_ms,
            tool_count=len(tools),
            golden_case_count=len(cases),
            candidate_count=len(candidates),
            duplicate_candidate_count=duplicate_candidate_count,
            warnings=(
                ["route_action_seed_duplicate_candidates_dropped"]
                if duplicate_candidate_count
                else []
            ),
        ),
    }


def _generate_seed_payload_with_host_model(
    *,
    app_id: str,
    tool_registry: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    target_seed_count: int,
    seed_run_id: str,
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = _build_seed_messages(
        app_id=app_id,
        tool_registry=tool_registry,
        golden_cases=golden_cases,
        target_seed_count=target_seed_count,
        seed_run_id=seed_run_id,
    )
    try:
        raw = (
            host_model_generate(messages, _seed_output_token_budget(target_seed_count), 0.0)
            if host_model_generate is not None
            else _call_host_model(
                messages,
                host_model_id=host_model_id,
                host_model_generate=None,
                max_tokens=_seed_output_token_budget(target_seed_count),
                temperature=0.0,
            )
        )
    except HostModelAssistantError:
        raise
    except Exception as exc:
        raise HostModelAssistantError(
            code="host_model_call_failed",
            message="Host model route/action seed call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host model returned an empty route/action seed response.",
            retryable=True,
            details={"model_id": model_id},
        )
    try:
        payload = json.loads(_extract_json_block(raw_output))
    except json.JSONDecodeError as exc:
        raise HostModelAssistantError(
            code="host_model_parse_error",
            message="Host model route/action seed response was not valid JSON.",
            retryable=False,
            details={
                "model_id": model_id,
                "reason": str(exc),
                "raw_output_fingerprint": _fingerprint(raw_output),
            },
        ) from exc
    return payload, {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _fingerprint(raw_output),
    }


def _build_seed_messages(
    *,
    app_id: str,
    tool_registry: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
    target_seed_count: int,
    seed_run_id: str,
) -> list[dict[str, str]]:
    request = {
        "task": "generate_route_action_seed_candidates",
        "schema_version": ROUTE_ACTION_SEED_CANDIDATES_SCHEMA_VERSION,
        "app_id": app_id,
        "seed_run_id": seed_run_id,
        "target_seed_count": max(1, int(target_seed_count)),
        "tool_registry": tool_registry,
        "tool_contracts": _tool_contracts_for_seed_prompt(tool_registry),
        "golden_cases": golden_cases,
        "route_intent_allowed_values": list(ROUTE_ACTION_ALLOWED_INTENTS),
        "route_intent_guidance": ROUTE_ACTION_INTENT_GUIDANCE,
        "output_schema": {
            "seed_cases": [
                {
                    "case_id": "stable id unique within this response",
                    "source_prompt": "audit/eval prompt, not used for training",
                    "route_intent": "one of route_intent_allowed_values",
                    "prompt_variants": [
                        "entity-free training paraphrases in the same language/locale as source_prompt that do not replay source_prompt"
                    ],
                    "selected_tools": ["tool names from tool_registry only"],
                    "tool_call_plan": [
                        {
                            "tool_name": "one selected tool name",
                            "args": "object using only that tool's declared schema keys",
                            "reason": "why this tool call is needed",
                        }
                    ],
                    "answer_discipline": "how to answer without memorizing exact facts",
                    "rationale": "why the route/tool target is correct",
                    "confidence": "0.0-1.0",
                }
            ]
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio host model assistant for developer app "
                "route learning. Generate route/action seed candidates from the "
                "developer-provided tool registry and golden cases. Do not invent "
                "tool names. Do not invent argument keys. Do not include concrete "
                "user facts, amounts, dates, account IDs, or one-off entity values "
                "in prompt_variants or tool_call_plan args. Tool call args must be "
                "schema structure, enum choices, bounded numbers, or placeholders "
                "for parser-owned values, never learned user fact values. If "
                "tool_contracts lists enum_args, integer_ranges, or usage_notes for "
                "a selected tool, obey those constraints exactly. "
                "For every seed_case, copy source_prompt exactly from one "
                "golden_cases[].prompt; do not paraphrase source_prompt or invent "
                "new source prompts. Cover every golden case at least once before "
                "adding extra candidates. "
                "Use route_intent_guidance to keep prompt_variants inside the requested "
                "intent boundary; for example, user_profile variants must not drift into "
                "aggregate rankings, totals, trends, or record breakdowns. "
                "Write prompt_variants in the same language and locale as the "
                "source_prompt unless a golden case explicitly requests another language. "
                "Return only one JSON object matching the requested schema."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, ensure_ascii=False, sort_keys=True),
        },
    ]


def _seed_payload_to_route_action_inputs(
    payload: dict[str, Any],
    *,
    tool_registry: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not isinstance(payload, dict):
        raise RouteActionSeedGeneratorError(
            code="host_model_schema_error",
            message="Host model route/action seed JSON must be an object.",
        )
    raw_cases = payload.get("seed_cases")
    if not isinstance(raw_cases, list):
        raise RouteActionSeedGeneratorError(
            code="host_model_schema_error",
            message="Host model route/action seed JSON missing seed_cases list.",
            details={"field": "seed_cases"},
        )

    registry_by_name = {tool["name"]: tool for tool in tool_registry}
    golden_by_prompt_key = {_prompt_key(golden["prompt"]): golden for golden in golden_cases}
    covered_golden_prompt_keys: set[str] = set()
    eval_cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_candidate_keys: set[str] = set()
    duplicate_candidate_count = 0
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise RouteActionSeedGeneratorError(
                code="host_model_schema_error",
                message=f"seed_cases[{index}] must be an object.",
            )
        case_id = _required_str(raw.get("case_id"), f"seed_cases[{index}].case_id")
        if case_id in seen_case_ids:
            raise RouteActionSeedGeneratorError(
                code="host_model_schema_error",
                message="Host model route/action seed duplicated a case_id.",
                details={"case_id": case_id},
            )
        seen_case_ids.add(case_id)
        source_prompt = _required_str(
            raw.get("source_prompt") or raw.get("eval_prompt") or raw.get("prompt"),
            f"seed_cases[{index}].source_prompt",
        )
        source_prompt_key = _prompt_key(source_prompt)
        if source_prompt_key not in golden_by_prompt_key:
            raise RouteActionSeedGeneratorError(
                code="host_model_schema_error",
                message="Host model route/action seed source_prompt must match a golden case prompt.",
                details={
                    "case_id": case_id,
                    "source_prompt": source_prompt,
                },
            )
        selected_tools = _string_list(raw.get("selected_tools"))
        _validate_selected_tools(
            case_id=case_id,
            selected_tools=selected_tools,
            registry_by_name=registry_by_name,
        )
        tool_call_plan = _sanitize_tool_call_plan_values(
            _list_of_dicts(raw.get("tool_call_plan")),
            registry_by_name=registry_by_name,
        )
        _validate_tool_call_plan(
            case_id=case_id,
            selected_tools=selected_tools,
            tool_call_plan=tool_call_plan,
            registry_by_name=registry_by_name,
        )
        route_intent = _seed_route_intent(raw, golden_cases)
        prompt_variants = _string_list(raw.get("prompt_variants"))
        candidate_key = _seed_candidate_key(
            route_intent=route_intent,
            prompt_variants=prompt_variants,
            selected_tools=selected_tools,
            tool_call_plan=tool_call_plan,
        )
        if candidate_key in seen_candidate_keys:
            duplicate_candidate_count += 1
            continue
        seen_candidate_keys.add(candidate_key)
        covered_golden_prompt_keys.add(source_prompt_key)
        eval_cases.append({
            "case_id": case_id,
            "prompt": source_prompt,
            "expectations": {
                "selected_tools_exact": selected_tools,
            },
        })
        selected_tool_text = ", ".join(selected_tools) if selected_tools else "no tool"
        pairs.append({
            "case_id": case_id,
            "route_intent": route_intent,
            "prompt_variants": prompt_variants,
            "selected_tools": selected_tools,
            "tool_call_plan": tool_call_plan,
            "answer_discipline": (
                raw.get("answer_discipline")
                or raw.get("answer_policy")
                or raw.get("answerPolicy")
                or f"Follow the {route_intent} route; use {selected_tool_text} when needed and do not invent stored facts."
            ),
            "rationale": (
                raw.get("rationale")
                or raw.get("reason")
                or raw.get("explanation")
                or f"Derived from a plan-gap golden case for {route_intent}."
            ),
            "confidence": raw.get("confidence"),
        })
    if raw_cases and not pairs:
        raise RouteActionSeedGeneratorError(
            code="host_model_schema_error",
            message="Host model route/action seed produced only duplicate candidates.",
        )
    missing_golden_cases = [
        {
            "case_id": _optional_str(golden.get("case_id")),
            "prompt": golden.get("prompt"),
        }
        for key, golden in golden_by_prompt_key.items()
        if key not in covered_golden_prompt_keys
    ]
    if missing_golden_cases:
        raise RouteActionSeedGeneratorError(
            code="host_model_schema_error",
            message="Host model route/action seed did not cover every golden case.",
            details={"missing_golden_cases": missing_golden_cases},
        )
    return eval_cases, pairs, duplicate_candidate_count


def _seed_candidate_key(
    *,
    route_intent: Any,
    prompt_variants: list[str],
    selected_tools: list[str],
    tool_call_plan: list[dict[str, Any]],
) -> str:
    primary_prompt = prompt_variants[0] if prompt_variants else ""
    return _fingerprint({
        "prompt": _prompt_key(primary_prompt),
        "route_intent": _optional_str(route_intent),
        "selected_tools": selected_tools,
        "tool_call_plan": [
            {
                "tool_name": _tool_name(entry),
                "args": _tool_args(entry),
            }
            for entry in tool_call_plan
        ],
    })


def _seed_route_intent(raw: dict[str, Any], golden_cases: list[dict[str, Any]]) -> Any:
    for key in ("route_intent", "expected_route_intent", "intent", "routeIntent"):
        value = _optional_str(raw.get(key))
        if value:
            return value
    source_prompt = _optional_str(
        raw.get("source_prompt") or raw.get("eval_prompt") or raw.get("prompt")
    )
    case_id = _optional_str(raw.get("case_id"))
    for golden in golden_cases:
        golden_prompt = _optional_str(golden.get("prompt"))
        golden_case_id = _optional_str(golden.get("case_id"))
        if source_prompt and golden_prompt and source_prompt == golden_prompt:
            return golden.get("expected_route_intent") or golden.get("route_intent")
        if case_id and golden_case_id and case_id.startswith(golden_case_id):
            return golden.get("expected_route_intent") or golden.get("route_intent")
    return raw.get("route_intent")


def _normalize_tool_registry(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RouteActionSeedGeneratorError(
            code="invalid_tool_registry",
            message="tool_registry must be a list.",
        )
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise RouteActionSeedGeneratorError(
                code="invalid_tool_registry",
                message=f"tool_registry[{index}] must be an object.",
            )
        name = _required_str(
            raw.get("name") or raw.get("tool_name") or raw.get("toolName"),
            f"tool_registry[{index}].name",
        )
        if name in seen:
            raise RouteActionSeedGeneratorError(
                code="invalid_tool_registry",
                message="tool_registry must not contain duplicate tool names.",
                details={"tool_name": name},
            )
        seen.add(name)
        tool = dict(raw)
        tool["name"] = name
        tools.append(tool)
    if not tools:
        raise RouteActionSeedGeneratorError(
            code="invalid_tool_registry",
            message="tool_registry must contain at least one tool.",
        )
    return tools


def _normalize_golden_cases(value: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RouteActionSeedGeneratorError(
            code="invalid_golden_cases",
            message="golden_cases must be a list.",
        )
    cases: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise RouteActionSeedGeneratorError(
                code="invalid_golden_cases",
                message=f"golden_cases[{index}] must be an object.",
            )
        prompt = _required_str(
            raw.get("prompt") or raw.get("source_prompt") or raw.get("query"),
            f"golden_cases[{index}].prompt",
        )
        case = dict(raw)
        case["prompt"] = prompt
        case["case_id"] = _optional_str(
            raw.get("case_id") or raw.get("id")
        ) or f"golden_case_{index + 1}"
        cases.append(case)
    if not cases:
        raise RouteActionSeedGeneratorError(
            code="invalid_golden_cases",
            message="golden_cases must contain at least one case.",
        )
    return cases


def _validate_selected_tools(
    *,
    case_id: str,
    selected_tools: list[str],
    registry_by_name: dict[str, dict[str, Any]],
) -> None:
    unknown = [tool for tool in selected_tools if tool not in registry_by_name]
    if unknown:
        raise RouteActionSeedGeneratorError(
            code="host_model_schema_error",
            message="Host model selected a tool outside the app registry.",
            details={"case_id": case_id, "unknown_tools": unknown},
        )


def _validate_tool_call_plan(
    *,
    case_id: str,
    selected_tools: list[str],
    tool_call_plan: list[dict[str, Any]],
    registry_by_name: dict[str, dict[str, Any]],
) -> None:
    selected = set(selected_tools)
    for index, entry in enumerate(tool_call_plan):
        tool_name = _tool_name(entry)
        if not tool_name:
            raise RouteActionSeedGeneratorError(
                code="host_model_schema_error",
                message="tool_call_plan entries must include tool_name or tool.",
                details={"case_id": case_id, "plan_index": index},
            )
        if tool_name not in selected:
            raise RouteActionSeedGeneratorError(
                code="host_model_schema_error",
                message="tool_call_plan tool must be present in selected_tools.",
                details={
                    "case_id": case_id,
                    "plan_index": index,
                    "tool_name": tool_name,
                },
            )
        allowed_args = _allowed_arg_names(registry_by_name.get(tool_name, {}))
        args = _tool_args(entry)
        if allowed_args is not None:
            unknown = sorted(key for key in args if key not in allowed_args)
            if unknown:
                raise RouteActionSeedGeneratorError(
                    code="host_model_schema_error",
                    message="tool_call_plan arguments contain keys outside the app schema.",
                    details={
                        "case_id": case_id,
                        "plan_index": index,
                        "tool_name": tool_name,
                        "unknown_args": unknown,
                    },
                )


def _sanitize_tool_call_plan_values(
    plan: list[dict[str, Any]],
    *,
    registry_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for entry in plan:
        item = dict(entry)
        tool_name = _tool_name(item) or ""
        args = _tool_args(item)
        if args:
            cleaned = dict(args)
            parser_owned_args = _parser_owned_arg_names(
                registry_by_name.get(tool_name, {})
            )
            for key, value in list(cleaned.items()):
                if not isinstance(value, str) or not value:
                    continue
                if key in parser_owned_args:
                    cleaned[key] = f"<{key}>"
                    continue
                if _is_date_arg_name(key) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                    cleaned[key] = f"<{key}>"
            if "args" in item:
                item["args"] = cleaned
            elif "arguments" in item:
                item["arguments"] = cleaned
            else:
                item["args"] = cleaned
        sanitized.append(item)
    return sanitized


def _parser_owned_arg_names(tool: dict[str, Any]) -> set[str]:
    names = set(_string_list(tool.get("parser_owned_args")))
    names.update(_string_list(tool.get("parserOwnedArgs")))
    value_policy = tool.get("value_policy")
    if isinstance(value_policy, dict):
        names.update(_string_list(value_policy.get("parser_owned_args")))
        names.update(_string_list(value_policy.get("parserOwnedArgs")))
    schema = None
    for key in ("args_schema", "arguments_schema", "schema"):
        schema = _json_object(tool.get(key))
        if schema is not None:
            break
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, raw_property in properties.items():
            if not isinstance(raw_property, dict):
                continue
            if (
                raw_property.get("x-parser-owned") is True
                or raw_property.get("x_parser_owned") is True
                or raw_property.get("parser_owned") is True
            ):
                text = str(name).strip()
                if text:
                    names.add(text)
    return names


def _is_date_arg_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return normalized in {"date", "startdate", "enddate", "fromdate", "todate"}


def _allowed_arg_names(tool: dict[str, Any]) -> set[str] | None:
    for key in ("args_schema", "arguments_schema", "schema"):
        raw = tool.get(key)
        schema = _json_object(raw)
        if schema is None:
            continue
        properties = schema.get("properties")
        if isinstance(properties, dict):
            return {str(name) for name in properties}
    parameters = tool.get("parameters")
    if isinstance(parameters, list):
        names = {
            str(item.get("name")).strip()
            for item in parameters
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        }
        return names
    return None


def _tool_contracts_for_seed_prompt(tool_registry: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declared_args_by_tool = {
        tool["name"]: _allowed_arg_names(tool)
        for tool in tool_registry
        if isinstance(tool.get("name"), str)
    }
    contract_by_name = {
        str(contract.get("name") or ""): dict(contract)
        for contract in route_action_tool_contracts_for_prompt(
            declared_args_by_tool.keys(),
            tool_registry=tool_registry,
        )
    }
    narrowed: list[dict[str, Any]] = []
    for tool in tool_registry:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        item = dict(contract_by_name.get(name, {"name": name}))
        if tool.get("description") and not item.get("description"):
            item["description"] = tool.get("description")
        registry_usage_notes = _optional_str(
            tool.get("usage_notes") or tool.get("usageNotes")
        )
        if registry_usage_notes and not _optional_str(item.get("usage_notes")):
            item["usage_notes"] = registry_usage_notes
        examples = _string_list(tool.get("examples"))
        if examples:
            item["examples"] = examples
        route_intents = _string_list(tool.get("route_intents") or tool.get("routeIntents"))
        if route_intents:
            item["route_intents"] = route_intents
        parser_owned_args = sorted(_parser_owned_arg_names(tool))
        if parser_owned_args:
            item["parser_owned_args"] = parser_owned_args
        allowed = declared_args_by_tool.get(name)
        if allowed is not None:
            existing_allowed = [arg for arg in item.get("allowed_args", []) if arg in allowed]
            item["allowed_args"] = sorted(existing_allowed or allowed)
            enum_args = item.get("enum_args") if isinstance(item.get("enum_args"), dict) else {}
            item["enum_args"] = {
                key: values for key, values in enum_args.items() if key in allowed
            }
            integer_ranges = (
                item.get("integer_ranges")
                if isinstance(item.get("integer_ranges"), dict)
                else {}
            )
            item["integer_ranges"] = {
                key: bounds for key, bounds in integer_ranges.items() if key in allowed
            }
        narrowed.append(item)
    return narrowed


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _tool_name(entry: dict[str, Any]) -> str | None:
    for key in ("tool_name", "toolName", "tool", "name"):
        text = _optional_str(entry.get(key))
        if text:
            return text
    return None


def _tool_args(entry: dict[str, Any]) -> dict[str, Any]:
    for key in ("args", "arguments"):
        value = entry.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise RouteActionSeedGeneratorError(
                code="host_model_schema_error",
                message=f"tool_call_plan {key!r} must be an object.",
            )
        return value
    return {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RouteActionSeedGeneratorError(
            code="host_model_schema_error",
            message="tool_call_plan must be a list.",
        )
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _optional_str(item)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _seed_run_id(
    *,
    seed_run_id: str | None,
    app_id: str,
    tool_registry: list[dict[str, Any]],
    golden_cases: list[dict[str, Any]],
) -> str:
    explicit = _optional_str(seed_run_id)
    if explicit:
        return explicit
    return "route_seed_" + hashlib.sha256(
        json.dumps(
            {
                "app_id": app_id,
                "tool_registry": tool_registry,
                "golden_cases": golden_cases,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]


def _seed_output_token_budget(target_seed_count: int) -> int:
    return min(12000, max(2000, int(target_seed_count) * 450))


def _audit(
    *,
    app_id: str,
    provider: str,
    host_model: dict[str, Any],
    generated_at_ms: int,
    tool_count: int,
    golden_case_count: int,
    candidate_count: int,
    warnings: list[str],
    duplicate_candidate_count: int = 0,
) -> dict[str, Any]:
    return {
        "app_id": app_id,
        "provider": provider,
        "host_model": host_model,
        "generated_at_ms": generated_at_ms,
        "tool_count": tool_count,
        "golden_case_count": golden_case_count,
        "candidate_count": candidate_count,
        "duplicate_candidate_count": duplicate_candidate_count,
        "warnings": warnings,
        "training_side_only": True,
        "golden_case_labels_are_training_supervision": True,
        "golden_case_labels_are_answer_quality_evidence": False,
        "golden_case_labels_are_runtime_routing_rules": False,
        "writes_events": False,
        "writes_runtime_artifacts": False,
    }


def _error_envelope(
    *,
    status: str,
    code: str,
    message: str,
    details: dict[str, Any],
    app_id: str,
    provider: str,
    host_model_id: str | None,
    generated_at_ms: int,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_ACTION_SEED_CANDIDATES_SCHEMA_VERSION,
        "status": status,
        "app_id": app_id,
        "route_action_response": None,
        "preview": None,
        "candidates": [],
        "error": {
            "code": code,
            "message": message,
            "retryable": code in {"host_model_call_failed", "host_model_empty_output"},
            "details": details,
        },
        "audit": _audit(
            app_id=app_id,
            provider=provider,
            host_model={
                "enabled": provider in HOST_MODEL_PROVIDERS,
                "model_id": None,
                "selected_model_id": host_model_id,
            },
            generated_at_ms=generated_at_ms,
            tool_count=0,
            golden_case_count=0,
            candidate_count=0,
            warnings=[status],
        ),
    }


def _required_str(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RouteActionSeedGeneratorError(
            code="invalid_seed_request",
            message=f"{field} must be a non-empty string.",
        )
    return text


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _prompt_key(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _fingerprint(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()
