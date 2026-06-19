# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build canonical device eval observations from app eval.log traces.

This module intentionally does not infer business expectations. Callers provide
the eval cases when they need case matching; the parser only extracts observed
runtime evidence and matches it back to prompts or case IDs.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Any


DEVICE_EVAL_LOG_SCHEMA_VERSION = "edgestudio.device_eval_log.v0"
ROUTE_MATRIX_LIVE_DECISION_EVAL_LOG_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_decision_eval_log.v0"
)


def build_device_eval_run_from_eval_log(request: dict[str, Any]) -> dict[str, Any]:
    """Return a device eval run populated with observations from eval.log."""

    generated_at = _utc_now()
    fingerprint = _fingerprint(request)
    if not isinstance(request, dict):
        return _error(
            code="invalid_input",
            message="request must be an object",
            details={"received_type": type(request).__name__},
            generated_at=generated_at,
            fingerprint=fingerprint,
        )

    try:
        run_id = _required_text(request.get("run_id"), "run_id")
        subject = _dict_or_empty(request.get("subject"))
        cases = _cases(request.get("cases"))
        lines = _log_lines(request)
        autorun_run_id_filter = _optional_text(
            request.get("device_autorun_run_id") or request.get("autorun_run_id")
        )
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="device eval log request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
            fingerprint=fingerprint,
        )

    turns, parse_warnings = _parse_turns(
        lines,
        autorun_run_id_filter=autorun_run_id_filter,
    )
    observations: list[dict[str, Any]] = []
    warnings = list(parse_warnings)
    used_turns: set[int] = set()
    for case in cases:
        prompt = _text(case.get("prompt"))
        if not prompt:
            warnings.append(f"case_missing_prompt:{case['case_id']}")
            continue
        match = _matching_turn(case["case_id"], prompt, turns, used_turns)
        if match is None:
            warnings.append(f"case_unmatched:{case['case_id']}")
            continue
        used_turns.add(match["index"])
        observations.append(_observation_from_turn(case["case_id"], match))

    summary = {
        "case_count": len(cases),
        "turn_count": len(turns),
        "matched_count": len(observations),
        "unmatched_count": len(cases) - len(observations),
        "warning_count": len(warnings),
        "warnings": warnings,
    }
    if autorun_run_id_filter:
        summary["device_autorun_run_id"] = autorun_run_id_filter

    eval_run = {
        "schema_version": "edgestudio.device_eval_run.v0",
        "run_id": run_id,
        "subject": subject,
        "cases": cases,
        "observations": observations,
    }
    return {
        "ok": True,
        "schema_version": DEVICE_EVAL_LOG_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "eval_run": eval_run,
            "summary": summary,
        },
        "error": None,
        "audit": _audit(
            generated_at=generated_at,
            fingerprint=fingerprint,
            status="ok",
            line_count=len(lines),
            turn_count=len(turns),
            warnings=warnings,
            autorun_run_id_filter=autorun_run_id_filter,
        ),
    }


def build_route_matrix_live_decision_eval_run_from_eval_log(
    request: dict[str, Any],
) -> dict[str, Any]:
    """Return an eval run containing all route-matrix live decisions in eval.log.

    This path is for live-feedback accumulation. It does not infer expected
    routes, tool names, or app semantics; it only preserves observed live audit
    facts that were already emitted by the device.
    """

    generated_at = _utc_now()
    fingerprint = _fingerprint(request)
    method = "build_route_matrix_live_decision_eval_run_from_eval_log"
    if not isinstance(request, dict):
        return _error(
            code="invalid_input",
            message="request must be an object",
            details={"received_type": type(request).__name__},
            generated_at=generated_at,
            fingerprint=fingerprint,
            method=method,
        )

    try:
        run_id = _required_text(request.get("run_id"), "run_id")
        subject = _dict_or_empty(request.get("subject"))
        lines = _log_lines(request)
        autorun_run_id_filter = _optional_text(
            request.get("device_autorun_run_id") or request.get("autorun_run_id")
        )
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix live eval log request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
            fingerprint=fingerprint,
            method=method,
        )

    turns, parse_warnings = _parse_turns(
        lines,
        autorun_run_id_filter=autorun_run_id_filter,
    )
    observations: list[dict[str, Any]] = []
    corrected_count = 0
    for turn in turns:
        decision = turn.get("route_matrix_live_decision")
        if not isinstance(decision, dict):
            continue
        case_id = (
            _text(decision.get("case_id"))
            or _text(turn.get("case_id"))
            or f"live:{_text(turn.get('prompt_sha256'))[:12]}"
        )
        observation = _observation_from_turn(case_id, turn)
        observations.append(observation)
        if isinstance(decision.get("user_correction"), dict) and decision["user_correction"]:
            corrected_count += 1

    summary = {
        "turn_count": len(turns),
        "live_decision_count": len(observations),
        "corrected_live_decision_count": corrected_count,
        "warning_count": len(parse_warnings),
        "warnings": list(parse_warnings),
    }
    if autorun_run_id_filter:
        summary["device_autorun_run_id"] = autorun_run_id_filter

    eval_run = {
        "schema_version": "edgestudio.device_eval_run.v0",
        "run_id": run_id,
        "subject": subject,
        "cases": [],
        "observations": observations,
    }
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_DECISION_EVAL_LOG_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "eval_run": eval_run,
            "summary": summary,
        },
        "error": None,
        "audit": _audit(
            generated_at=generated_at,
            fingerprint=fingerprint,
            status="ok",
            line_count=len(lines),
            turn_count=len(turns),
            warnings=list(parse_warnings),
            autorun_run_id_filter=autorun_run_id_filter,
            method=method,
        ),
    }


def _parse_turns(
    lines: list[str],
    *,
    autorun_run_id_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    turns: list[dict[str, Any]] = []
    warnings: list[str] = []
    current: dict[str, Any] | None = None

    for raw in lines:
        event = _eval_event(raw)
        if not event:
            continue
        if (
            event.startswith("llm_start_enter ")
            or event.startswith("ws_chat_question ")
            or event.startswith("route_matrix_autorun_prompt ")
        ):
            if current is not None:
                turns.append(current)
            fields = _fields(event)
            current = _new_turn(len(turns), fields.get("text"))
            case_id = _text(fields.get("case_id"))
            if case_id:
                current["case_id"] = case_id
            autorun_run_id = _text(fields.get("run_id"))
            if autorun_run_id:
                current["autorun_run_id"] = autorun_run_id
            autorun_index = _optional_int(fields.get("index"))
            if autorun_index is not None:
                current["autorun_index"] = autorun_index
            continue
        if current is None:
            warnings.append("eval_event_before_turn_ignored")
            continue

        if event.startswith("router_decision "):
            fields = _fields(event)
            current["route_intent"] = fields.get("intent") or current.get("route_intent")
            current["effective_intent"] = fields.get("effective_intent")
            current["selected_tools"] = _csv(fields.get("selected_tools"))
            current["router_reason"] = fields.get("reason")
            current["router_confidence"] = _optional_float(fields.get("confidence"))
        elif event.startswith("route_matrix_autorun_decision "):
            fields = _fields(event)
            current["route_intent"] = fields.get("intent") or current.get("route_intent")
            current["effective_intent"] = fields.get("intent") or current.get("effective_intent")
            current["selected_tools"] = _csv(fields.get("selected_tools"))
            current["router_reason"] = fields.get("reason")
            current["router_confidence"] = _optional_float(fields.get("confidence"))
            current["completed"] = True
        elif event.startswith("route_matrix_shadow "):
            shadow = _route_matrix_shadow_event(event)
            matched = _attach_route_matrix_shadow_by_input_hash(
                turns=turns,
                current=current,
                shadow=shadow,
                autorun_run_id_filter=autorun_run_id_filter,
            )
            if not matched:
                current["route_matrix_shadow"] = shadow
        elif event.startswith("route_matrix_live_decision "):
            decision = _route_matrix_live_decision_event(event)
            pending = current.get("route_matrix_live_user_correction")
            if isinstance(pending, dict) and _correction_matches_decision(
                correction=pending,
                decision=decision,
            ):
                decision["user_correction"] = dict(pending["user_correction"])
            current["route_matrix_live_decision"] = decision
        elif event.startswith("route_matrix_live_user_correction "):
            correction = _route_matrix_live_user_correction_event(event)
            matched = _attach_route_matrix_live_user_correction(
                turns=turns,
                current=current,
                correction=correction,
            )
            if not matched:
                current["route_matrix_live_user_correction"] = correction
        elif event.startswith("route_matrix_live "):
            current["route_matrix_live"] = _route_matrix_live_event(event)
        elif event.startswith("route_matrix_runtime_validation "):
            current["route_matrix_runtime_validation"] = _route_matrix_runtime_validation_event(event)
        elif event.startswith("tool_call "):
            fields = _fields(event)
            current["tool_calls"].append({
                "name": fields.get("name") or "",
                "arguments": _parse_tool_args(_field_to_end(event, "args")),
            })
        elif event.startswith("direct_tool_summary "):
            fields = _fields(event)
            name = fields.get("name")
            if name:
                current["tool_calls"].append({"name": name, "arguments": {}})
        elif event.startswith("tool_result ") or event.startswith("direct_tool_summary_done "):
            current["fact_tool_called"] = True
        elif event.startswith("chat_assistant_reply "):
            current["answer"] = _field(event, "text")
        elif event.startswith("chat_generation done "):
            fields = _fields(event)
            current["completed"] = True
            current["duration_ms"] = _elapsed_ms(fields.get("elapsed"))
        elif event.startswith("chat_generation_error "):
            current["completed"] = False
            current["error"] = _field(event, "error")
        elif event.startswith("phys_footprint "):
            _record_footprint(current, event)

    if current is not None:
        turns.append(current)
    for turn in turns:
        if turn.get("completed") is None:
            turn["completed"] = False
            turn["freeze_detected"] = True
        else:
            turn["freeze_detected"] = turn.get("completed") is not True
        error = _text(turn.get("error"))
        turn["oom"] = "oom" in error.lower() or "out of memory" in error.lower()
    if autorun_run_id_filter:
        turns = [
            turn
            for turn in turns
            if _text(turn.get("autorun_run_id")) == autorun_run_id_filter
        ]
        if not turns:
            warnings.append(f"autorun_run_id_unmatched:{autorun_run_id_filter}")
    return turns, _dedupe(warnings)


def _new_turn(index: int, prompt: str | None) -> dict[str, Any]:
    prompt_text = _text(prompt)
    return {
        "index": index,
        "case_id": "",
        "prompt": prompt_text,
        "prompt_sha256": _sha256_text(prompt_text) if prompt_text else "",
        "answer": "",
        "route_intent": "",
        "effective_intent": "",
        "selected_tools": [],
        "tool_calls": [],
        "fact_tool_called": False,
        "completed": None,
        "duration_ms": None,
        "max_memory_mb": None,
        "error": "",
    }


def _observation_from_turn(case_id: str, turn: dict[str, Any]) -> dict[str, Any]:
    observation = {
        "case_id": case_id,
        "answer": _text(turn.get("answer")),
        "route_intent": _text(turn.get("effective_intent")) or _text(turn.get("route_intent")),
        "selected_tools": list(turn.get("selected_tools") or []),
        "fact_tool_called": bool(turn.get("fact_tool_called") or turn.get("tool_calls")),
        "tool_calls": list(turn.get("tool_calls") or []),
        "completed": turn.get("completed") is True,
        "freeze_detected": turn.get("freeze_detected") is True,
        "oom": turn.get("oom") is True,
        "error": _text(turn.get("error")),
    }
    if isinstance(turn.get("route_matrix_shadow"), dict):
        observation["route_matrix_shadow"] = dict(turn["route_matrix_shadow"])
    if isinstance(turn.get("route_matrix_live"), dict):
        observation["route_matrix_live"] = dict(turn["route_matrix_live"])
    if isinstance(turn.get("route_matrix_live_decision"), dict):
        observation["route_matrix_live_decision"] = dict(
            turn["route_matrix_live_decision"]
        )
    if isinstance(turn.get("route_matrix_runtime_validation"), dict):
        observation["route_matrix_runtime_validation"] = dict(
            turn["route_matrix_runtime_validation"]
        )
    if turn.get("duration_ms") is not None:
        observation["duration_ms"] = turn["duration_ms"]
    if turn.get("max_memory_mb") is not None:
        observation["max_memory_mb"] = turn["max_memory_mb"]
    return observation


def _eval_event(line: str) -> str | None:
    marker = "[Eval]"
    if marker not in line:
        return None
    return line.split(marker, 1)[1].strip()


def _fields(event: str) -> dict[str, str]:
    return {key: _unquote(value) for key, value in _FIELD_RE.findall(event)}


def _field(event: str, name: str) -> str | None:
    fields = _fields(event)
    return fields.get(name)


_FIELD_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\"(?:[^\"\\]|\\.)*\"|[^ ]*)")


def _route_matrix_shadow_event(event: str) -> dict[str, Any]:
    fields = _fields(event)
    out: dict[str, Any] = {}
    for key in (
        "status",
        "reason",
        "training_run_id",
        "input_sha256",
        "predicted_intent",
        "threshold_passed",
    ):
        value = _text(fields.get(key))
        if value and value != "nil":
            out[key] = value
    for key in ("predicted_probability", "predicted_threshold", "latency_ms"):
        value = _optional_float(fields.get(key))
        if value is not None:
            out[key] = value
    return out


def _route_matrix_runtime_validation_event(event: str) -> dict[str, Any]:
    fields = _fields(event)
    out: dict[str, Any] = {
        "status": _text(fields.get("status")),
        "run_id": _text(fields.get("run_id")),
        "selected_tools": _csv(fields.get("selected_tools")),
        "tool_plan_name": _text(fields.get("tool_plan_name")),
        "tool_registry_ok": _optional_bool(fields.get("tool_registry_ok")),
        "tool_call_plan_ok": _optional_bool(fields.get("tool_call_plan_ok")),
        "schema_validation_ok": _optional_bool(fields.get("schema_validation_ok")),
    }
    args = _text(fields.get("args"))
    if args:
        out["arguments"] = _parse_tool_args(args)
    reason = _text(fields.get("reason"))
    if reason:
        out["reason"] = reason
    return out


def _route_matrix_live_event(event: str) -> dict[str, Any]:
    fields = _fields(event)
    out: dict[str, Any] = {}
    for key in (
        "status",
        "reason",
        "final_source",
        "fallback_reason",
        "training_run_id",
        "matrix_intent",
        "threshold_passed",
    ):
        value = _nil_text(fields.get(key))
        if value is not None:
            out[key] = value
    for key in ("matrix_probability", "matrix_threshold"):
        value = _optional_float(fields.get(key))
        if value is not None:
            out[key] = value
    return out


def _route_matrix_live_decision_event(event: str) -> dict[str, Any]:
    fields = _fields(event)
    probability = _optional_float(fields.get("matrix_probability"))
    threshold = _optional_float(fields.get("threshold"))
    threshold_passed = _optional_bool(fields.get("threshold_passed"))
    return {
        "case_id": _text(fields.get("case_id")),
        "matrix_prediction": {
            "intent": _text(fields.get("matrix_intent")),
            "probability": probability,
            "threshold": threshold,
            "threshold_passed": threshold_passed,
        },
        "matrix_calibrated_confidence": probability,
        "evidence_available": _optional_bool(fields.get("evidence_available")),
        "evidence_route": None,
        "final_decision_source": _text(fields.get("final_source")),
        "fallback_reason": _nil_text(fields.get("fallback_reason")),
        "shadow_mode_was": False,
        "user_correction": None,
    }


def _route_matrix_live_user_correction_event(event: str) -> dict[str, Any]:
    fields = _fields(event)
    payload: dict[str, Any] = {
        "schema_version": _text(
            fields.get("schema_version")
            or "edgestudio.route_matrix_user_correction.v0"
        ),
        "source_input_text": _text(fields.get("source_input_text")),
        "correction_text": _text(fields.get("correction_text")),
        "correction_source": _text(fields.get("correction_source") or "user"),
        "is_fixture": _optional_bool(fields.get("is_fixture")) is True,
    }
    created_at_ms = _optional_int(fields.get("created_at_ms"))
    if created_at_ms is not None:
        payload["created_at_ms"] = created_at_ms
    return {
        "case_id": _text(fields.get("case_id")),
        "user_correction": payload,
    }


def _attach_route_matrix_live_user_correction(
    *,
    turns: list[dict[str, Any]],
    current: dict[str, Any],
    correction: dict[str, Any],
) -> bool:
    for turn in [current, *reversed(turns)]:
        decision = turn.get("route_matrix_live_decision")
        if not isinstance(decision, dict):
            continue
        if not _correction_matches_decision(correction=correction, decision=decision):
            continue
        decision["user_correction"] = dict(correction["user_correction"])
        return True
    return False


def _correction_matches_decision(
    *,
    correction: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    correction_case_id = _text(correction.get("case_id"))
    decision_case_id = _text(decision.get("case_id"))
    return not correction_case_id or correction_case_id == decision_case_id


def _attach_route_matrix_shadow_by_input_hash(
    *,
    turns: list[dict[str, Any]],
    current: dict[str, Any],
    shadow: dict[str, Any],
    autorun_run_id_filter: str | None = None,
) -> bool:
    input_sha256 = _text(shadow.get("input_sha256"))
    if not input_sha256:
        return False
    if (
        autorun_run_id_filter
        and _text(current.get("autorun_run_id"))
        and _text(current.get("autorun_run_id")) != autorun_run_id_filter
    ):
        return False
    for turn in [current, *reversed(turns)]:
        if autorun_run_id_filter and _text(turn.get("autorun_run_id")) != autorun_run_id_filter:
            continue
        if _text(turn.get("prompt_sha256")) == input_sha256:
            turn["route_matrix_shadow"] = shadow
            return True
    return False


def _field_to_end(event: str, name: str) -> str | None:
    match = re.search(rf"\b{re.escape(name)}=", event)
    if not match:
        return None
    return event[match.end():].strip()


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    text = _text(raw)
    if not text:
        return {}
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    body = text.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]
    out: dict[str, Any] = {}
    for part in _split_top_level(body):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = _unquote(key.strip())
        if not key:
            continue
        out[key] = _parse_scalar(value.strip())
    return out


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    in_quote = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if char == "," and not in_quote:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(value: str) -> Any:
    text = _unquote(value)
    lowered = text.lower()
    if lowered in {"nil", "none", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _record_footprint(turn: dict[str, Any], event: str) -> None:
    match = re.search(r"=([0-9]+(?:\.[0-9]+)?)MB", event)
    if not match:
        return
    value = float(match.group(1))
    current = turn.get("max_memory_mb")
    if current is None or value > float(current):
        turn["max_memory_mb"] = value


def _matching_turn(
    case_id: str,
    prompt: str,
    turns: list[dict[str, Any]],
    used_turns: set[int],
) -> dict[str, Any] | None:
    normalized_case_id = _text(case_id)
    if normalized_case_id:
        for turn in reversed(turns):
            if turn["index"] in used_turns:
                continue
            if _text(turn.get("case_id")) == normalized_case_id:
                return turn
    key = _norm(prompt)
    for turn in reversed(turns):
        if turn["index"] in used_turns:
            continue
        if _norm(turn.get("prompt")) == key:
            return turn
    return None


def _log_lines(request: dict[str, Any]) -> list[str]:
    lines = request.get("lines")
    if isinstance(lines, list):
        if not all(isinstance(line, str) for line in lines):
            raise TypeError("lines must be list[str]")
        return lines
    text = request.get("eval_log")
    if isinstance(text, str) and text.strip():
        return text.splitlines()
    raise ValueError("eval_log or lines is required")


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
        out.append(dict(item))
    return out


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
    fingerprint: str,
    method: str = "build_device_eval_run_from_eval_log",
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": DEVICE_EVAL_LOG_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": _audit(
            generated_at=generated_at,
            fingerprint=fingerprint,
            status="error",
            line_count=None,
            turn_count=None,
            warnings=[],
            autorun_run_id_filter=None,
            method=method,
        ),
    }


def _audit(
    *,
    generated_at: str,
    fingerprint: str,
    status: str,
    line_count: int | None,
    turn_count: int | None,
    warnings: list[str],
    autorun_run_id_filter: str | None,
    method: str = "build_device_eval_run_from_eval_log",
) -> dict[str, Any]:
    input_summary: dict[str, Any] = {
        "line_count": line_count,
        "turn_count": turn_count,
    }
    if autorun_run_id_filter:
        input_summary["device_autorun_run_id"] = autorun_run_id_filter

    return {
        "schema_version": "edgestudio.device_eval_log_audit.v0",
        "method": method,
        "generated_at": generated_at,
        "status": status,
        "input_fingerprint": fingerprint,
        "input_summary": input_summary,
        "warnings": list(warnings),
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("subject must be an object")
    return dict(value)


def _optional_text(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _csv(value: str | None) -> list[str]:
    return [item.strip() for item in _text(value).split(",") if item.strip()]


def _elapsed_ms(value: str | None) -> int | None:
    text = _text(value)
    if text.endswith("s"):
        text = text[:-1]
    try:
        return int(round(float(text) * 1000))
    except ValueError:
        return None


def _optional_int(value: str | None) -> int | None:
    try:
        return int(_text(value))
    except ValueError:
        return None


def _optional_float(value: str | None) -> float | None:
    try:
        return float(_text(value))
    except ValueError:
        return None


def _optional_bool(value: str | None) -> bool | None:
    text = _text(value).lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _text(value: Any) -> str:
    return str(value or "").strip()


def _nil_text(value: Any) -> str | None:
    text = _text(value)
    if not text or text.lower() in {"nil", "none", "null"}:
        return None
    return text


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    text = " ".join(text.split())
    return _TRAILING_PROMPT_PUNCT_RE.sub("", text)


_TRAILING_PROMPT_PUNCT_RE = re.compile(r"[\s.!?,;:。！？、，；：]+$")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _fingerprint(value: Any) -> str:
    material = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
