# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-model suggestion service for editable RPP A-library direction sets."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

import yaml

from backend.services.a_library_direction_sets import (
    DIRECTION_SET_SOURCE_SCHEMA_VERSION,
    sanitize_direction_set_id,
)
from backend.services.host_model_assistant import (
    HOST_MODEL_PROVIDER,
    HostModelAssistantError,
    HostModelGenerate,
    _call_host_model,
    _host_model_output_text,
    _host_model_id,
)


DIRECTION_SUGGESTION_SCHEMA_VERSION = "edgestudio.a_library_direction_suggestions.v1"
DIRECTION_SUGGESTION_AUDIT_SCHEMA_VERSION = "edgestudio.a_library_direction_suggestion_audit.v1"
DOMAIN_DESCRIPTION_REFINEMENT_SCHEMA_VERSION = "edgestudio.a_library_domain_description_refinement.v1"


class ALibrarySuggestionServiceError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        status_code: int,
        details: dict[str, Any] | None = None,
        audit: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        self.details = details or {}
        self.audit = audit or {}

    def to_response(self) -> dict[str, Any]:
        return {
            "ok": False,
            "schema_version": DIRECTION_SUGGESTION_SCHEMA_VERSION,
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            },
            "audit": self.audit,
        }


def suggest_a_library_directions(
    *,
    host_model_id: str,
    model_name: str,
    domain_description: str,
    target_count: int,
    repair_context: dict[str, Any] | None = None,
    host_model_generate: HostModelGenerate | None = None,
) -> dict[str, Any]:
    generated_at = _utc_now()
    messages = build_direction_suggestion_messages(
        domain_description=domain_description,
        target_count=target_count,
        repair_context=repair_context,
    )
    prompt_fingerprint = _fingerprint(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    audit_base = _audit(
        task="suggest_directions",
        status="started",
        generated_at=generated_at,
        host_model_id=host_model_id,
        model_name=model_name,
        target_count=target_count,
        domain_description=domain_description,
        prompt_fingerprint=prompt_fingerprint,
        repair_context=repair_context,
        host_model=None,
        warnings=[],
    )

    raw = _call_suggestion_host_model(
        messages=messages,
        host_model_id=host_model_id,
        max_tokens=_direction_output_token_budget(target_count),
        temperature=0.2,
        audit=audit_base,
        host_model_generate=host_model_generate,
    )
    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise ALibrarySuggestionServiceError(
            code="host_model_empty_output",
            message="Host model returned an empty direction suggestion response.",
            retryable=True,
            status_code=502,
            details={"model_id": model_id},
            audit=audit_base | {
                "status": "error",
                "host_model": _host_model_audit(model_id=model_id, raw_output=None),
            },
        )

    try:
        parsed_json, repair_strategy = json_object_from_text(raw_output)
        candidates = parse_direction_suggestion_output(parsed_json, target_count)
    except ALibrarySuggestionServiceError as exc:
        exc.audit = audit_base | {
            "status": "error",
            "host_model": _host_model_audit(model_id=model_id, raw_output=raw_output),
        }
        raise

    return {
        "ok": True,
        "schema_version": DIRECTION_SUGGESTION_SCHEMA_VERSION,
        "status": "complete",
        "model_id": host_model_id,
        "model_name": model_name,
        "domain_description": domain_description,
        "target_count": target_count,
        "directions": candidates[:target_count],
        "repair_strategy": repair_strategy,
        "repaired": repair_strategy != "direct",
        "raw_output": raw_output,
        "audit": audit_base | {
            "status": "complete",
            "host_model": _host_model_audit(model_id=model_id, raw_output=raw_output),
            "parse": {
                "repair_strategy": repair_strategy,
                "repaired": repair_strategy != "direct",
                "candidate_count": len(candidates),
            },
        },
    }


def refine_domain_description(
    *,
    host_model_id: str,
    model_name: str,
    domain_description: str,
    host_model_generate: HostModelGenerate | None = None,
) -> dict[str, Any]:
    generated_at = _utc_now()
    messages = build_domain_description_refinement_messages(domain_description)
    prompt_fingerprint = _fingerprint(json.dumps(messages, ensure_ascii=False, sort_keys=True))
    audit_base = _audit(
        task="refine_domain_description",
        status="started",
        generated_at=generated_at,
        host_model_id=host_model_id,
        model_name=model_name,
        target_count=None,
        domain_description=domain_description,
        prompt_fingerprint=prompt_fingerprint,
        repair_context=None,
        host_model=None,
        warnings=[],
    )
    raw = _call_suggestion_host_model(
        messages=messages,
        host_model_id=host_model_id,
        max_tokens=768,
        temperature=0.2,
        audit=audit_base,
        host_model_generate=host_model_generate,
    )
    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    try:
        refined = clean_refined_description(raw_output)
    except ALibrarySuggestionServiceError as exc:
        exc.audit = audit_base | {
            "status": "error",
            "host_model": _host_model_audit(model_id=model_id, raw_output=raw_output),
        }
        raise
    return {
        "ok": True,
        "schema_version": DOMAIN_DESCRIPTION_REFINEMENT_SCHEMA_VERSION,
        "status": "complete",
        "model_id": host_model_id,
        "model_name": model_name,
        "original_description": domain_description,
        "refined_description": refined,
        "audit": audit_base | {
            "status": "complete",
            "host_model": _host_model_audit(model_id=model_id, raw_output=raw_output),
        },
    }


def build_direction_suggestion_messages(
    *,
    domain_description: str,
    target_count: int,
    repair_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    has_cjk = _has_cjk(domain_description)
    lang_instruction = (
        "IMPORTANT: The positive and negative examples MUST be complete sentences written in Chinese (中文), "
        "matching the user's language. Each sentence must describe a specific observable behavior. "
        "Do NOT use English keywords, short labels, or a fixed repeated prefix in the examples."
        if has_cjk
        else
        "Each positive/negative example must be a complete sentence describing observable user behavior. "
        "Do NOT use single keywords, short labels, or a fixed repeated prefix."
    )
    repair_instruction = direction_repair_instruction(repair_context)
    return [
        {
            "role": "system",
            "content": (
                "You design editable RPP A-library direction-set seeds for developers. "
                "Return strict JSON only. Do not use markdown. "
                "The JSON shape is {\"directions\":[{\"name\":\"snake_case\","
                "\"description\":\"...\",\"domain\":\"short_domain\","
                "\"positive\":[5 strings],\"negative\":[5 strings]}]}. "
                "CRITICAL RULES: "
                "0) The directions array length MUST match the requested target count exactly; fewer directions is an API failure. "
                "1) Every direction name MUST be unique — no two directions can share the same name. "
                "2) Every positive and negative array MUST have exactly 5 items. "
                "3) The directions are RPP Basis observation axes, not Interest/Steering controls. "
                "Each direction MUST be a mutually independent observable axis, not a paraphrase, subtopic, "
                "or sibling label of one narrow theme/value. "
                "4) If the domain description is narrow, expand it into independent behavior axes "
                "(motivation, trade-off, context, cadence, evidence use, correction preference) rather than "
                "repeating the same concern. "
                "5) Close all JSON arrays with ] before closing objects with }. "
                f"{repair_instruction}"
                f"{lang_instruction}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Domain description: {domain_description}\n"
                f"Generate exactly {target_count} directions. "
                "Each direction must have a UNIQUE snake_case name (e.g. emotional_support, academic_pressure, peer_relations — all different), "
                "a practical description, exactly 5 positive and 5 negative example sentences, and a clearly independent observation target."
            ),
        },
    ]


def build_domain_description_refinement_messages(domain_description: str) -> list[dict[str, str]]:
    language_instruction = "Write the refined paragraph in Chinese." if _has_cjk(domain_description) else "Write the refined paragraph in English."
    return [
        {
            "role": "system",
            "content": (
                "You refine rough app-domain ideas for an RPP A-library Direction Set Editor. "
                "Output one concise editable paragraph only: no markdown, no bullets, no JSON. "
                "The paragraph should describe the target app/user domain and suggest several mutually independent "
                "observable behavior dimensions that are suitable as RPP Basis axes. "
                "Do not write Interest/Steering instructions and do not repeat one narrow theme with synonyms. "
                f"{language_instruction}"
            ),
        },
        {
            "role": "user",
            "content": (
                "Rough developer idea:\n"
                f"{domain_description.strip()}\n\n"
                "Rewrite it into a stronger domain description for generating independent RPP Basis observation axes."
            ),
        },
    ]


def direction_repair_instruction(repair_context: dict[str, Any] | None) -> str:
    if repair_context is None:
        return ""
    lines: list[str] = [
        "REPAIR MODE: The previous direction set did not pass validation or A-library health checks. ",
        "Treat the following as failure evidence, not as names to preserve. Generate a fresh full set of independent RPP Basis axes. ",
    ]
    prev_direction_set_id = _str(repair_context.get("prev_direction_set_id"))
    reason = _str(repair_context.get("reason"))
    if prev_direction_set_id:
        lines.append(f"Previous direction_set_id: {prev_direction_set_id}. ")
    if reason:
        lines.append(f"Failure reason: {reason[:400]}. ")
    worst_pairs = repair_context.get("worst_pairs")
    if isinstance(worst_pairs, list):
        pairs = [
            " / ".join(str(item).strip() for item in pair[:2] if str(item).strip())
            for pair in worst_pairs[:6]
            if isinstance(pair, list)
        ]
        pairs = [pair for pair in pairs if pair]
        if pairs:
            lines.append(
                "Collapsed pairs from the previous run: "
                + "; ".join(pairs)
                + ". Do not merely rename these; spread the next set across broader, mutually independent observation axes. "
            )
    metrics: list[str] = []
    max_abs_cos = _float_or_none(repair_context.get("max_abs_cos"))
    mean_abs_cos = _float_or_none(repair_context.get("mean_abs_cos"))
    signal_pass = repair_context.get("signal_pass")
    if max_abs_cos is not None:
        metrics.append(f"max_abs_cos={max_abs_cos:.3f}")
    if mean_abs_cos is not None:
        metrics.append(f"mean_abs_cos={mean_abs_cos:.3f}")
    if isinstance(signal_pass, bool):
        metrics.append(f"signal_pass={signal_pass}")
    if metrics:
        lines.append("Previous health summary: " + ", ".join(metrics) + ". ")
    validation_error_codes = repair_context.get("validation_error_codes")
    if isinstance(validation_error_codes, list):
        codes = ", ".join(str(code)[:80] for code in validation_error_codes[:12])
        if codes:
            lines.append(f"Previous validation errors: {codes}. ")
    return "".join(lines)


def json_object_from_text(text: str) -> tuple[dict[str, Any], str]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ALibrarySuggestionServiceError(
            code="host_model_no_json",
            message="Host model did not return JSON.",
            retryable=True,
            status_code=502,
        )
    fragment = cleaned[start:]
    strategies: list[tuple[str, Callable[[str], dict[str, Any] | None]]] = [
        ("direct", _try_parse),
        ("fix_unclosed_arrays", _fix_unclosed_arrays),
        ("repair_truncated", _repair_truncated_json),
    ]
    for name, attempt in strategies:
        result = attempt(fragment)
        if result is not None:
            return result, name
    raise ALibrarySuggestionServiceError(
        code="host_model_parse_error",
        message="Host model returned unparseable JSON.",
        retryable=False,
        status_code=502,
    )


def parse_direction_suggestion_output(payload: dict[str, Any], target_count: int) -> list[dict[str, Any]]:
    directions = payload.get("directions")
    if not isinstance(directions, list):
        raise ALibrarySuggestionServiceError(
            code="host_model_missing_directions",
            message="Host model did not return a directions list.",
            retryable=True,
            status_code=502,
        )
    parsed: list[dict[str, Any]] = []
    for idx, raw in enumerate(directions):
        if not isinstance(raw, dict):
            continue
        positive = _string_list(raw.get("positive"))[:5]
        negative = _string_list(raw.get("negative"))[:5]
        if len(positive) < 2 or len(negative) < 2:
            continue
        parsed.append({
            "name": safe_direction_name(str(raw.get("name") or f"direction_{idx + 1}")),
            "description": str(raw.get("description") or ""),
            "domain": safe_direction_name(str(raw.get("domain") or "custom")),
            "positive": positive + [""] * max(0, 5 - len(positive)),
            "negative": negative + [""] * max(0, 5 - len(negative)),
        })
    if not parsed:
        raise ALibrarySuggestionServiceError(
            code="host_model_no_usable_directions",
            message="Host model returned no usable direction candidates.",
            retryable=True,
            status_code=502,
        )
    if len(parsed) < target_count:
        raise ALibrarySuggestionServiceError(
            code="host_model_too_few_directions",
            message=f"Host model returned only {len(parsed)}/{target_count} directions. Try a larger model or simpler domain description.",
            retryable=True,
            status_code=502,
            details={"actual": len(parsed), "expected": target_count},
        )
    return parsed


def serialize_direction_suggestions_to_yaml(
    *,
    direction_set_id: str,
    directions: list[dict[str, Any]],
) -> str:
    set_id = sanitize_direction_set_id(direction_set_id)
    payload = {
        "schema_version": DIRECTION_SET_SOURCE_SCHEMA_VERSION,
        "direction_set_id": set_id,
        "directions": [
            {
                "name": safe_direction_name(str(direction.get("name") or f"direction_{idx + 1}")),
                "description": str(direction.get("description") or ""),
                "domain": safe_direction_name(str(direction.get("domain") or "custom")),
                "positive": _string_list(direction.get("positive"))[:5],
                "negative": _string_list(direction.get("negative"))[:5],
            }
            for idx, direction in enumerate(directions)
            if isinstance(direction, dict)
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def clean_refined_description(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip()
    if not cleaned:
        raise ALibrarySuggestionServiceError(
            code="host_model_empty_refined_description",
            message="Host model returned an empty refined description.",
            retryable=True,
            status_code=502,
        )
    return cleaned


def safe_direction_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip().lower()).strip("._-")
    return safe or "custom"


def _call_suggestion_host_model(
    *,
    messages: list[dict[str, str]],
    host_model_id: str,
    max_tokens: int,
    temperature: float,
    audit: dict[str, Any],
    host_model_generate: HostModelGenerate | None,
) -> Any:
    try:
        return _call_host_model(
            messages,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except HostModelAssistantError as exc:
        raise ALibrarySuggestionServiceError(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            status_code=502 if exc.retryable else 400,
            details=exc.details,
            audit=audit | {
                "status": "error",
                "host_model": {
                    "enabled": True,
                    "provider": HOST_MODEL_PROVIDER,
                    "selected_model_id": host_model_id,
                    "model_id": exc.details.get("model_id"),
                },
            },
        ) from exc
    except Exception as exc:
        raise ALibrarySuggestionServiceError(
            code="host_model_call_failed",
            message="Host model A-library direction suggestion call failed.",
            retryable=True,
            status_code=502,
            details={"reason": str(exc), "model_id": None},
            audit=audit | {
                "status": "error",
                "host_model": {
                    "enabled": True,
                    "provider": HOST_MODEL_PROVIDER,
                    "selected_model_id": host_model_id,
                    "model_id": None,
                },
            },
        ) from exc


def _try_parse(fragment: str) -> dict[str, Any] | None:
    end = fragment.rfind("}")
    if end <= 0:
        return None
    try:
        payload = json.loads(fragment[: end + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _fix_unclosed_arrays(fragment: str) -> dict[str, Any] | None:
    fixed, changed = _insert_missing_array_closers_before_objects(fragment)
    if not changed:
        return None
    return _try_parse(fixed)


def _insert_missing_array_closers_before_objects(fragment: str) -> tuple[str, bool]:
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False
    changed = False
    for ch in fragment:
        if escape:
            out.append(ch)
            escape = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escape = True
            continue
        if ch == '"':
            out.append(ch)
            in_string = not in_string
            continue
        if in_string:
            out.append(ch)
            continue
        if ch in ("{", "["):
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
            continue
        if ch in ("}", "]"):
            if ch == "}":
                while stack and stack[-1] == "]":
                    out.append("]")
                    stack.pop()
                    changed = True
            if stack and stack[-1] == ch:
                stack.pop()
            out.append(ch)
            continue
        out.append(ch)
    return "".join(out), changed


def _repair_truncated_json(fragment: str) -> dict[str, Any] | None:
    trimmed = fragment.rstrip()
    while trimmed and trimmed[-1] in (",", ":", '"'):
        if trimmed[-1] == '"':
            trimmed = trimmed[:-1]
            last_quote = trimmed.rfind('"')
            if last_quote >= 0:
                trimmed = trimmed[:last_quote]
            break
        trimmed = trimmed[:-1].rstrip()
    opens: list[str] = []
    in_string = False
    escape = False
    for ch in trimmed:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            opens.append("}" if ch == "{" else "]")
        elif ch in ("}", "]") and opens and opens[-1] == ch:
            opens.pop()
    repaired = trimmed + "".join(reversed(opens))
    try:
        payload = json.loads(repaired)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _direction_output_token_budget(target_count: int) -> int:
    return min(8192, max(1800, target_count * 700))


def _audit(
    *,
    task: str,
    status: str,
    generated_at: str,
    host_model_id: str,
    model_name: str,
    target_count: int | None,
    domain_description: str,
    prompt_fingerprint: str,
    repair_context: dict[str, Any] | None,
    host_model: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": DIRECTION_SUGGESTION_AUDIT_SCHEMA_VERSION,
        "task": task,
        "status": status,
        "provider": HOST_MODEL_PROVIDER,
        "generated_at": generated_at,
        "host_model": host_model,
        "input_summary": {
            "model_id": host_model_id,
            "model_name": model_name,
            "target_count": target_count,
            "domain_description_fingerprint": _fingerprint(domain_description),
            "domain_description_length": len(domain_description),
            "repair_context_present": repair_context is not None,
        },
        "prompt_fingerprint": prompt_fingerprint,
        "warnings": warnings,
    }


def _host_model_audit(*, model_id: str | None, raw_output: str | None) -> dict[str, Any]:
    return {
        "enabled": True,
        "provider": HOST_MODEL_PROVIDER,
        "model_id": model_id,
        "raw_output_fingerprint": _fingerprint(raw_output) if raw_output else None,
    }


def _has_cjk(value: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fingerprint(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
