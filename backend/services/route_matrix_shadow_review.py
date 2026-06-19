# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Review route-matrix shadow results against expected routes.

This module is deliberately review-only. It compares offline CLI scoring and
true-device shadow audit logs, then labels each case as a routing candidate or
as needing more evidence. It never changes runtime routing behavior.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.device_eval_log_parser import build_device_eval_run_from_eval_log


ROUTE_MATRIX_SHADOW_REVIEW_SCHEMA_VERSION = "edgestudio.route_matrix_shadow_review.v0"

_SHADOW_READY_STATUS = "scored_not_applied"


def build_route_matrix_shadow_review(request: dict[str, Any]) -> dict[str, Any]:
    """Return a review report for route-matrix shadow CLI/device evidence."""

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
        cases = _cases(request.get("cases"))
        device_autorun_run_id = _optional_text(request.get("device_autorun_run_id"))
        cli_turns = _cli_turns(request.get("cli_report"))
        device_turns = _device_turns(
            run_id=run_id,
            cases=cases,
            eval_log=request.get("device_eval_log"),
            lines=request.get("device_eval_log_lines"),
            device_autorun_run_id=device_autorun_run_id,
        )
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix shadow review request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    rows = [
        _review_case(case, cli_turns.get(case["prompt_key"]), device_turns.get(case["case_id"]))
        for case in cases
    ]
    summary = _summary(rows)

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_SHADOW_REVIEW_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "summary": summary,
            "cases": rows,
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_shadow_review_audit.v0",
            "method": "build_route_matrix_shadow_review",
            "generated_at": generated_at,
            "input_summary": {
                "case_count": len(cases),
                "cli_turn_count": len(cli_turns),
                "device_turn_count": len(device_turns),
                "device_autorun_run_id": device_autorun_run_id,
            },
        },
    }


def build_route_matrix_shadow_review_from_files(
    *,
    run_id: str,
    cases_path: Path,
    cli_report_path: Path | None = None,
    device_eval_log_path: Path | None = None,
    device_autorun_run_id: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "run_id": run_id,
        "cases": _read_json(cases_path),
    }
    if device_autorun_run_id:
        request["device_autorun_run_id"] = device_autorun_run_id
    if cli_report_path is not None:
        request["cli_report"] = _read_json(cli_report_path)
    if device_eval_log_path is not None:
        request["device_eval_log"] = device_eval_log_path.read_text(encoding="utf-8")
    return build_route_matrix_shadow_review(request)


def _review_case(
    case: dict[str, Any],
    cli_turn: dict[str, Any] | None,
    device_turn: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_intent = case["expected_intent"]
    cli = _matrix_observation(cli_turn)
    device = _matrix_observation(
        device_turn.get("route_matrix_shadow") if isinstance(device_turn, dict) else None
    )
    evidence = _evidence_observation(device_turn)
    verdict, reasons = _verdict(expected_intent=expected_intent, cli=cli, device=device)
    execution_stage = _execution_stage(
        expected_intent=expected_intent,
        verdict=verdict,
        evidence=evidence,
    )

    return {
        "case_id": case["case_id"],
        "prompt": case["prompt"],
        "lane": case["lane"],
        "expected_intent": expected_intent,
        "evidence_route": evidence,
        "matrix_cli": cli,
        "matrix_device": device,
        "comparison": {
            "cli_device_agree": _same_non_empty(
                cli.get("predicted_intent"),
                device.get("predicted_intent"),
            ),
            "expected_match_cli": cli.get("predicted_intent") == expected_intent,
            "expected_match_device": device.get("predicted_intent") == expected_intent,
            "evidence_expected_match": evidence.get("route_intent") == expected_intent,
            "threshold_passed_cli": cli.get("threshold_passed") is True,
            "threshold_passed_device": device.get("threshold_passed") is True,
            "execution_stage": execution_stage,
            "verdict": verdict,
            "reasons": reasons,
        },
    }


def _verdict(
    *,
    expected_intent: str,
    cli: dict[str, Any],
    device: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    cli_status = _text(cli.get("status"))
    device_status = _text(device.get("status"))
    if cli_status != _SHADOW_READY_STATUS:
        reasons.append(f"cli_status:{cli_status or 'missing'}")
    if device_status != _SHADOW_READY_STATUS:
        reasons.append(f"device_status:{device_status or 'missing'}")
    if reasons:
        return "insufficient_shadow_evidence", reasons

    cli_intent = _text(cli.get("predicted_intent"))
    device_intent = _text(device.get("predicted_intent"))
    if cli_intent != device_intent:
        return "runtime_parity_gap", [
            f"cli_intent:{cli_intent or 'missing'}",
            f"device_intent:{device_intent or 'missing'}",
        ]

    if cli_intent != expected_intent or device_intent != expected_intent:
        return "needs_data_or_intent_schema", [
            f"expected:{expected_intent}",
            f"predicted:{cli_intent or device_intent or 'missing'}",
        ]

    if cli.get("threshold_passed") is True and device.get("threshold_passed") is True:
        return "routing_candidate", []

    return "needs_threshold_or_calibration", [
        f"cli_threshold_passed:{cli.get('threshold_passed')}",
        f"device_threshold_passed:{device.get('threshold_passed')}",
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts = Counter(row["comparison"]["verdict"] for row in rows)
    lane_counts: dict[str, Counter[str]] = {}
    execution_stage_counts: Counter[str] = Counter()
    candidate_lanes: set[str] = set()
    candidate_evidence_gap_cases: list[str] = []
    for row in rows:
        lane = row["lane"]
        verdict = row["comparison"]["verdict"]
        execution_stage_counts[row["comparison"]["execution_stage"]] += 1
        lane_counts.setdefault(lane, Counter())[verdict] += 1
        if verdict == "routing_candidate":
            candidate_lanes.add(lane)
            if row["comparison"]["evidence_expected_match"] is not True:
                candidate_evidence_gap_cases.append(row["case_id"])
    return {
        "case_count": len(rows),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "by_lane": {
            lane: dict(sorted(counts.items()))
            for lane, counts in sorted(lane_counts.items())
        },
        "routing_candidate_count": verdict_counts.get("routing_candidate", 0),
        "routing_candidate_lanes": sorted(candidate_lanes),
        "routing_candidate_evidence_gap_count": len(candidate_evidence_gap_cases),
        "routing_candidate_evidence_gap_cases": candidate_evidence_gap_cases,
        "routing_candidate_execution_stage_counts": dict(sorted(execution_stage_counts.items())),
        "executable_candidate_count": (
            execution_stage_counts.get("executable_with_validated_plan", 0)
            + execution_stage_counts.get("executable_with_existing_plan", 0)
        ),
        "intent_only_candidate_count": execution_stage_counts.get("intent_only_missing_tool_plan", 0),
        "runtime_validation_failed_candidate_count": execution_stage_counts.get("runtime_validation_failed", 0),
        "runtime_validation_unknown_candidate_count": execution_stage_counts.get("runtime_validation_unknown", 0),
        "ready_for_live_routing": False,
        "ready_for_live_routing_reason": "shadow_review_only_requires_separate_release_decision",
    }


def _cli_turns(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("cli_report must be an object")
    raw_turns = value.get("turns")
    if not isinstance(raw_turns, list):
        raise TypeError("cli_report.turns must be a list")
    out: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_turns):
        if not isinstance(row, dict):
            raise TypeError(f"cli_report.turns[{index}] must be an object")
        prompt = _required_text(_first(row, "prompt", "text"), f"cli_report.turns[{index}].prompt")
        out[_prompt_key(prompt)] = row
    return out


def _execution_stage(
    *,
    expected_intent: str,
    verdict: str,
    evidence: dict[str, Any],
) -> str:
    if verdict != "routing_candidate":
        return "not_candidate"
    if expected_intent == "base_chat":
        return "intent_ready_no_tool_required"
    selected_tools = evidence.get("selected_tools")
    if isinstance(selected_tools, list) and selected_tools:
        runtime_validation = evidence.get("route_matrix_runtime_validation")
        if isinstance(runtime_validation, dict):
            if (
                runtime_validation.get("status") == "validated"
                and runtime_validation.get("tool_call_plan_ok") is not False
                and runtime_validation.get("tool_registry_ok") is not False
                and runtime_validation.get("schema_validation_ok") is not False
            ):
                return "executable_with_validated_plan"
            return "runtime_validation_failed"
        return "runtime_validation_unknown"
    return "intent_only_missing_tool_plan"


def _device_turns(
    *,
    run_id: str,
    cases: list[dict[str, Any]],
    eval_log: Any,
    lines: Any,
    device_autorun_run_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    if eval_log is None and lines is None:
        return {}
    request: dict[str, Any] = {
        "run_id": f"{run_id}-device-log",
        "cases": [
            {
                "case_id": case["case_id"],
                "prompt": case["prompt"],
            }
            for case in cases
        ],
    }
    if device_autorun_run_id:
        request["device_autorun_run_id"] = device_autorun_run_id
    if eval_log is not None:
        if not isinstance(eval_log, str):
            raise TypeError("device_eval_log must be a string")
        request["eval_log"] = eval_log
    if lines is not None:
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise TypeError("device_eval_log_lines must be list[str]")
        request["lines"] = lines
    built = build_device_eval_run_from_eval_log(request)
    if built.get("ok") is not True:
        details = built.get("error", {}).get("details", {})
        raise ValueError(f"device eval log parse failed: {details}")
    observations = built["result"]["eval_run"]["observations"]
    return {
        _text(row.get("case_id")): row
        for row in observations
        if isinstance(row, dict) and _text(row.get("case_id"))
    }


def _matrix_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"status": "missing"}
    status = _text(_first(value, "status"))
    out: dict[str, Any] = {
        "status": status or "missing",
    }
    for out_key, *keys in (
        ("reason", "reason"),
        ("predicted_intent", "predicted_intent", "predictedIntent"),
        ("input_sha256", "input_sha256", "inputSHA256", "inputSha256"),
        ("error", "error"),
    ):
        found = _text(_first(value, *keys))
        if found and found != "nil":
            out[out_key] = found
    for out_key, *keys in (
        ("predicted_probability", "predicted_probability", "predictedProbability"),
        ("predicted_threshold", "predicted_threshold", "predictedThreshold"),
        ("latency_ms", "latency_ms", "latencyMs", "latencyMS"),
    ):
        number = _optional_float(_first(value, *keys))
        if number is not None:
            out[out_key] = number
    threshold = _optional_bool(_first(value, "threshold_passed", "thresholdPassed"))
    if threshold is not None:
        out["threshold_passed"] = threshold
    probabilities = _first(value, "probabilities_by_intent", "probabilitiesByIntent")
    if isinstance(probabilities, dict):
        out["probabilities_by_intent"] = {
            str(key): float(val)
            for key, val in probabilities.items()
            if isinstance(val, int | float)
        }
    return out


def _evidence_observation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"matched": False}
    out = {
        "matched": True,
        "route_intent": _text(value.get("route_intent")),
        "selected_tools": list(value.get("selected_tools") or []),
        "tool_calls": list(value.get("tool_calls") or []),
        "completed": value.get("completed") is True,
        "freeze_detected": value.get("freeze_detected") is True,
        "oom": value.get("oom") is True,
    }
    runtime_validation = value.get("route_matrix_runtime_validation")
    if isinstance(runtime_validation, dict):
        out["route_matrix_runtime_validation"] = dict(runtime_validation)
    return out


def _cases(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("cases must be a list")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"cases[{index}] must be an object")
        case_id = _required_text(item.get("case_id"), f"cases[{index}].case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        prompt = _required_text(item.get("prompt"), f"cases[{index}].prompt")
        expected_intent = _required_text(
            _first(item, "expected_intent", "expectedIntent"),
            f"cases[{index}].expected_intent",
        )
        lane = _text(item.get("lane")) or expected_intent
        out.append({
            **item,
            "case_id": case_id,
            "prompt": prompt,
            "prompt_key": _prompt_key(prompt),
            "expected_intent": expected_intent,
            "lane": lane,
        })
    return out


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
        "schema_version": ROUTE_MATRIX_SHADOW_REVIEW_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_shadow_review_audit.v0",
            "method": "build_route_matrix_shadow_review",
            "generated_at": generated_at,
            "input_summary": {},
        },
    }


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _prompt_key(value: Any) -> str:
    return " ".join(_text(value).split()).casefold()


def _same_non_empty(left: Any, right: Any) -> bool:
    left_text = _text(left)
    right_text = _text(right)
    return bool(left_text and right_text and left_text == right_text)


def _first(value: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(value, dict) and key in value:
            return value[key]
    return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _optional_float(value: Any) -> float | None:
    try:
        return float(_text(value))
    except ValueError:
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
