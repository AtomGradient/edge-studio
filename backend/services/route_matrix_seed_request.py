# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build Host Model seed requests from route-matrix shadow gaps.

This module is training-side only. It turns app-provided matrix eval cases plus
shadow scoring output into golden cases for the existing route/action seed
generator. It does not call the Host Model and does not emit runtime artifacts.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re


ROUTE_MATRIX_SEED_REQUEST_SCHEMA_VERSION = "edgestudio.route_matrix_seed_request.v0"


def build_route_matrix_seed_request(request: dict[str, Any]) -> dict[str, Any]:
    generated_at = _utc_now()
    if not isinstance(request, dict):
        return _error(
            code="invalid_input",
            message="request must be an object",
            details={"received_type": type(request).__name__},
            generated_at=generated_at,
        )
    try:
        run_id = _required_text(request.get("run_id"), "run_id")
        app_id = _required_text(request.get("app_id"), "app_id")
        cases = _case_rows(request.get("cases"))
        cli_report = _cli_report(request.get("cli_report"))
        tool_registry = _tool_registry(request.get("tool_registry"))
        target_seed_count = max(1, int(request.get("target_seed_count") or 24))
        include_intents = _optional_set(request.get("include_intents"))
        max_cases = int(request.get("max_cases") or 0)
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix seed request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    turns_by_prompt = {
        _normalize_prompt(_text(turn.get("prompt"))): turn
        for turn in _cli_turns(cli_report)
        if _normalize_prompt(_text(turn.get("prompt")))
    }
    seed_cases: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for row in cases:
        case = _golden_case_from_shadow_gap(row, turns_by_prompt)
        status = _text(case.get("shadow_gap_status"))
        if include_intents and case.get("expected_route_intent") not in include_intents:
            skipped_counts["intent_filter"] += 1
            continue
        if status == "shadow_passed":
            skipped_counts["shadow_passed"] += 1
            continue
        status_counts[status] += 1
        seed_cases.append(case)
        if max_cases > 0 and len(seed_cases) >= max_cases:
            break

    route_action_request = {
        "schema_version": "edgestudio.route_action_seed_candidates_request.v0",
        "app_id": app_id,
        "tool_registry": tool_registry,
        "golden_cases": seed_cases,
        "target_seed_count": max(target_seed_count, len(seed_cases)),
        "seed_run_id": f"{run_id}:matrix-shadow-gaps",
    }
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_SEED_REQUEST_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "app_id": app_id,
            "summary": {
                "case_count": len(cases),
                "seed_case_count": len(seed_cases),
                "shadow_gap_status_counts": dict(sorted(status_counts.items())),
                "skipped_counts": dict(sorted(skipped_counts.items())),
                "intent_counts": dict(
                    sorted(Counter(case["expected_route_intent"] for case in seed_cases).items())
                ),
                "ready_for_host_model_seed_generation": bool(seed_cases),
                "ready_for_live_routing": False,
                "ready_for_live_routing_reason": (
                    "matrix_seed_request_is_training_side_only_and_requires_host_model_output_gates"
                ),
            },
            "route_action_seed_candidates_request": route_action_request,
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_seed_request_audit.v0",
            "method": "build_route_matrix_seed_request",
            "generated_at": generated_at,
            "training_side_only": True,
            "golden_case_labels_are_training_supervision": True,
            "golden_case_labels_are_answer_quality_evidence": False,
            "golden_case_labels_are_runtime_routing_rules": False,
            "writes_events": False,
            "writes_runtime_artifacts": False,
            "input_summary": {
                "tool_count": len(tool_registry),
                "target_seed_count": target_seed_count,
                "include_intents": sorted(include_intents),
            },
        },
    }


def build_route_matrix_seed_request_from_files(
    *,
    run_id: str,
    app_id: str,
    cases_path: Path,
    cli_report_path: Path,
    tool_registry_path: Path,
    target_seed_count: int = 24,
    include_intents: list[str] | None = None,
    max_cases: int = 0,
) -> dict[str, Any]:
    return build_route_matrix_seed_request({
        "run_id": run_id,
        "app_id": app_id,
        "cases": _read_json(cases_path),
        "cli_report": _read_json(cli_report_path),
        "tool_registry": _read_json(tool_registry_path),
        "target_seed_count": target_seed_count,
        "include_intents": include_intents or [],
        "max_cases": max_cases,
    })


def _golden_case_from_shadow_gap(
    row: dict[str, Any],
    turns_by_prompt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = _required_text(row.get("case_id") or row.get("id"), "case.case_id")
    prompt = _required_text(row.get("prompt"), f"{case_id}.prompt")
    expected_intent = _required_text(
        row.get("expected_intent")
        or row.get("expected_route_intent")
        or row.get("route_intent"),
        f"{case_id}.expected_intent",
    )
    turn = turns_by_prompt.get(_normalize_prompt(prompt), {})
    predicted_intent = _text(turn.get("predictedIntent") or turn.get("predicted_intent"))
    threshold_passed = turn.get("thresholdPassed")
    if not turn:
        status = "missing_shadow_score"
    elif predicted_intent != expected_intent:
        status = "wrong_intent"
    elif threshold_passed is not True:
        status = "below_threshold"
    else:
        status = "shadow_passed"
    return {
        "case_id": case_id,
        "prompt": prompt,
        "expected_route_intent": expected_intent,
        "selected_tools": _string_list(row.get("selected_tools")),
        "shadow_gap_status": status,
        "matrix_predicted_intent": predicted_intent,
        "matrix_predicted_probability": _number_or_none(
            turn.get("predictedProbability") or turn.get("predicted_probability")
        ),
        "matrix_predicted_threshold": _number_or_none(
            turn.get("predictedThreshold") or turn.get("predicted_threshold")
        ),
        "instructions": {
            "source_prompt_is_eval_only": True,
            "generate_prompt_variants": True,
            "do_not_replay_source_prompt": True,
            "selected_tools_must_exist_in_tool_registry": True,
            "tool_call_plan_args_must_match_declared_schema": True,
            "tool_call_plan_args_must_be_entity_free": True,
            "do_not_include_user_fact_values_in_variants": True,
            "repair_shadow_gap_status": status,
            "intent_boundary": _intent_boundary(expected_intent),
        },
    }


def _case_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("cases must be a list")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise TypeError(f"cases[{index}] must be an object")
        rows.append(dict(row))
    if not rows:
        raise ValueError("cases must not be empty")
    return rows


def _cli_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("cli_report must be an object")
    return value


def _cli_turns(report: dict[str, Any]) -> list[dict[str, Any]]:
    turns = report.get("turns")
    if not isinstance(turns, list):
        return []
    return [dict(turn) for turn in turns if isinstance(turn, dict)]


def _tool_registry(value: Any) -> list[dict[str, Any]]:
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
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(name)
        tool = dict(raw)
        tool["name"] = name
        tools.append(tool)
    if not tools:
        raise ValueError("tool_registry must contain at least one tool")
    return tools


def _optional_set(value: Any) -> set[str]:
    return set(_string_list(value))


def _intent_boundary(intent: str) -> str:
    boundaries = {
        "base_chat": (
            "Keep variants as general no-tool chat; do not ask for stored app "
            "facts, profile facts, or app state changes."
        ),
        "exact_fact": (
            "Keep variants as concrete record or exact fact lookup; do not ask "
            "for totals, rankings, trends, or habit summaries."
        ),
        "aggregate_fact": (
            "Keep variants as totals, counts, rankings, trends, comparisons, or "
            "grouped summaries over records; do not turn them into stable habit "
            "or preference summaries."
        ),
        "app_action": (
            "Keep variants as app state mutations or commands; tool arguments "
            "for parser-owned values must stay as slots, not learned constants."
        ),
        "user_profile": (
            "Keep variants as stable habits, preferences, style, or profile "
            "summaries; avoid exact totals, biggest/top rankings, trends, record "
            "lists, or category breakdown calculations."
        ),
        "mixed": "Use only when a single more specific intent cannot represent the route.",
    }
    return boundaries.get(intent, "")


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _text(item)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _normalize_prompt(value: str) -> str:
    return " ".join(
        token.strip(" \t\r\n.,!?;:'\"()[]{}，。！？；：、“”‘’（）【】")
        for token in re.split(r"\s+", value.casefold())
        if token.strip()
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_SEED_REQUEST_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_seed_request_audit.v0",
            "method": "build_route_matrix_seed_request",
            "generated_at": generated_at,
            "training_side_only": True,
            "golden_case_labels_are_training_supervision": True,
            "golden_case_labels_are_answer_quality_evidence": False,
            "golden_case_labels_are_runtime_routing_rules": False,
            "writes_events": False,
            "writes_runtime_artifacts": False,
        },
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
