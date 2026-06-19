# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Audit plan-prototype coverage for route-matrix shadow candidates.

This is a training/review-side helper. It does not emit runtime artifacts and
does not make matrix predictions executable. Its purpose is to make the next
blocker explicit: matrix v0 predicts intent, while tool-requiring intents still
need selected tools and validated toolCallPlan data.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUTE_MATRIX_PLAN_PROTOTYPE_REVIEW_SCHEMA_VERSION = (
    "edgestudio.route_matrix_plan_prototype_review.v0"
)


def build_route_matrix_plan_prototype_review(request: dict[str, Any]) -> dict[str, Any]:
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
        shadow_review = _shadow_review(request.get("shadow_review"))
        samples = _samples(request.get("learner_dataset_samples"))
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix plan prototype review request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    catalog = _prototype_catalog(samples)
    rows = [_review_candidate(row, catalog) for row in _candidate_rows(shadow_review)]
    summary = _summary(rows, sample_count=len(samples), catalog=catalog)
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_PLAN_PROTOTYPE_REVIEW_SCHEMA_VERSION,
        "result": {
            "run_id": run_id,
            "summary": summary,
            "candidates": rows,
        },
        "error": None,
        "audit": {
            "schema_version": "edgestudio.route_matrix_plan_prototype_review_audit.v0",
            "method": "build_route_matrix_plan_prototype_review",
            "generated_at": generated_at,
            "input_summary": {
                "shadow_candidate_count": len(rows),
                "learner_sample_count": len(samples),
            },
        },
    }


def build_route_matrix_plan_prototype_review_from_files(
    *,
    run_id: str,
    shadow_review_path: Path,
    learner_dataset_path: Path,
) -> dict[str, Any]:
    return build_route_matrix_plan_prototype_review({
        "run_id": run_id,
        "shadow_review": _read_json(shadow_review_path),
        "learner_dataset_samples": _read_jsonl(learner_dataset_path),
    })


def _review_candidate(row: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    expected_intent = _text(row.get("expected_intent"))
    prompt = _text(row.get("prompt"))
    prompt_key = _prompt_key(prompt)
    warnings: list[str] = []
    evidence_route = row.get("evidence_route") if isinstance(row.get("evidence_route"), dict) else {}
    evidence_tools = _string_list(evidence_route.get("selected_tools"))
    if expected_intent == "base_chat":
        status = "no_tool_required"
        prototype = None
    elif evidence_tools:
        status = "already_executable_by_evidence"
        prototype = {
            "selected_tools": evidence_tools,
            "source": "evidence_route",
        }
    else:
        exact = catalog["exact"].get((prompt_key, expected_intent))
        intent_prototypes = catalog["by_intent"].get(expected_intent, [])
        with_plan = [item for item in intent_prototypes if item["has_tool_call_plan"]]
        if exact and exact["has_tool_call_plan"]:
            status = "exact_prompt_plan_available_training_side_only"
            prototype = exact
        elif with_plan:
            status = "same_intent_plan_available_training_side_only"
            prototype = with_plan[0]
            warnings.append("same_intent_plan_substituted")
            if exact and exact["selected_tools"]:
                warnings.append(
                    "same_intent_plan_substituted_for_exact_prompt_selected_tool_only"
                )
        elif exact and exact["selected_tools"]:
            status = "exact_prompt_selected_tool_without_plan"
            prototype = exact
        elif intent_prototypes:
            status = "same_intent_selected_tool_without_plan"
            prototype = intent_prototypes[0]
        else:
            status = "no_plan_prototype"
            prototype = None
    return {
        "case_id": row.get("case_id"),
        "prompt": prompt,
        "expected_intent": expected_intent,
        "matrix_device": row.get("matrix_device"),
        "evidence_route": evidence_route,
        "plan_prototype_status": status,
        "prototype": _public_prototype(prototype),
        "runtime_executable": status in {
            "no_tool_required",
            "already_executable_by_evidence",
        },
        "training_side_only": status.endswith("_training_side_only"),
        "warnings": warnings,
    }


def _summary(
    rows: list[dict[str, Any]],
    *,
    sample_count: int,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(row["plan_prototype_status"] for row in rows)
    warning_counts = Counter(
        warning
        for row in rows
        for warning in (
            row.get("warnings") if isinstance(row.get("warnings"), list) else []
        )
        if isinstance(warning, str) and warning
    )
    return {
        "candidate_count": len(rows),
        "learner_sample_count": sample_count,
        "prototype_count": len(catalog["prototypes"]),
        "status_counts": dict(sorted(status_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "same_intent_plan_substituted_count": warning_counts.get(
            "same_intent_plan_substituted",
            0,
        ),
        "runtime_executable_count": sum(1 for row in rows if row["runtime_executable"]),
        "training_side_only_count": sum(1 for row in rows if row["training_side_only"]),
        "missing_runtime_plan_count": sum(
            1
            for row in rows
            if not row["runtime_executable"]
        ),
        "ready_for_live_routing": False,
        "ready_for_live_routing_reason": (
            "plan_prototypes_are_training_side_only_and_runtime_tool_calls_still_need_registry_schema_validation"
        ),
    }


def _prototype_catalog(samples: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[tuple[str, str], dict[str, Any]] = {}
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prototypes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        input_payload = sample.get("input") if isinstance(sample.get("input"), dict) else {}
        target = sample.get("target") if isinstance(sample.get("target"), dict) else {}
        prompt = _text(input_payload.get("text"))
        route_intent = _text(target.get("route_intent"))
        if not prompt or not route_intent:
            continue
        selected_tools = _string_list(target.get("selected_tools"))
        tool_call_plan = _list_of_dicts(target.get("tool_call_plan"))
        fingerprint = json.dumps(
            {
                "route_intent": route_intent,
                "selected_tools": selected_tools,
                "tool_call_plan": tool_call_plan,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        prototype = {
            "route_intent": route_intent,
            "selected_tools": selected_tools,
            "tool_call_plan": tool_call_plan,
            "has_tool_call_plan": bool(tool_call_plan),
            "sample_prompt": prompt,
            "sample_id": _text(sample.get("sample_id")),
            "source_case_id": _text(
                sample.get("source", {}).get("case_id")
                if isinstance(sample.get("source"), dict)
                else None
            ),
        }
        exact.setdefault((_prompt_key(prompt), route_intent), prototype)
        if fingerprint not in seen:
            seen.add(fingerprint)
            prototypes.append(prototype)
            by_intent[route_intent].append(prototype)
    return {
        "exact": exact,
        "by_intent": by_intent,
        "prototypes": prototypes,
    }


def _public_prototype(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "route_intent": value.get("route_intent"),
        "selected_tools": value.get("selected_tools"),
        "tool_call_plan": value.get("tool_call_plan"),
        "has_tool_call_plan": value.get("has_tool_call_plan") is True,
        "sample_prompt": value.get("sample_prompt"),
        "sample_id": value.get("sample_id"),
        "source_case_id": value.get("source_case_id"),
    }


def _candidate_rows(shadow_review: dict[str, Any]) -> list[dict[str, Any]]:
    result = shadow_review.get("result") if isinstance(shadow_review.get("result"), dict) else {}
    rows = result.get("cases")
    if not isinstance(rows, list):
        raise ValueError("shadow_review.result.cases must be a list")
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"shadow_review.result.cases[{index}] must be an object")
        comparison = row.get("comparison") if isinstance(row.get("comparison"), dict) else {}
        if comparison.get("verdict") == "routing_candidate":
            candidates.append(row)
    return candidates


def _shadow_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("shadow_review must be an object")
    if value.get("ok") is not True:
        raise ValueError("shadow_review.ok must be true")
    return value


def _samples(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("learner_dataset_samples must be a list")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"learner_dataset_samples[{index}] must be an object")
        out.append(item)
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} must be an object")
            rows.append(value)
    return rows


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_PLAN_PROTOTYPE_REVIEW_SCHEMA_VERSION,
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.route_matrix_plan_prototype_review_audit.v0",
            "method": "build_route_matrix_plan_prototype_review",
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


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
