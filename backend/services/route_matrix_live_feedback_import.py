# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Import reviewed live-feedback corrections into route/action learner feedstock.

This is the explicit second-loop bridge:

live decision audit -> user correction -> Host Model review -> route/action gates
-> learner dataset.

The raw live audit event is never a training sample. Fixture corrections are
blocked by default, and every imported pair is normalized through the existing
route/action contract and eval-leakage gates before any learner JSONL is
written.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.host_model_assistant import (
    HOST_MODEL_PROVIDER,
    generate_route_action_pairs,
)
from backend.services.route_action_training_events import (
    ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
    RouteActionEvalLeakageError,
    store_route_action_training_events,
    write_route_action_learner_dataset_jsonl,
)
from backend.stores.event_store import EventStore


ROUTE_MATRIX_LIVE_FEEDBACK_REVIEW_IMPORT_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_review_import.v0"
)

_APPROVED_DECISIONS = frozenset({"approve", "approved", "accept", "accepted", "pass", "ok"})
_LANGUAGE_DRIFT_WARNING = "route_matrix_live_feedback_prompt_variant_language_drift"
_LANGUAGE_DRIFT_DETECTION_DIRECTIONS = ["cjk_to_non_cjk"]
_LANGUAGE_DRIFT_DETECTION_STRATEGY = "source_has_any_cjk_requires_each_variant_has_any_cjk"


def import_route_matrix_live_feedback_review(
    *,
    review_request: dict[str, Any],
    host_model_review_receipt: dict[str, Any],
    tool_registry: list[dict[str, Any]] | None,
    output_dir: Path,
    peer_id: str | None = None,
    event_store_path: Path | None = None,
    event_store: EventStore | None = None,
    limit: int = 1000,
    allow_fixture_corrections: bool = False,
) -> dict[str, Any]:
    """Store reviewed live corrections and write a gated learner dataset."""

    generated_at = _utc_now()
    try:
        request_result = _review_request_result(review_request)
        app_id = _required_text(request_result.get("app_id"), "review_request.result.app_id")
        run_id = _required_text(request_result.get("run_id"), "review_request.result.run_id")
        cases = _review_request_cases(request_result)
        if _is_fixture_review(host_model_review_receipt) and not allow_fixture_corrections:
            return _error(
                status="fixture_review_blocked",
                code="fixture_review_blocked",
                message="Fixture Host Model reviews are not learner feedstock.",
                details={},
                generated_at=generated_at,
                app_id=app_id,
                peer_id=peer_id,
                run_id=run_id,
            )
        pairs, eval_cases, summary = _route_action_inputs_from_review(
            cases=cases,
            host_model_review_receipt=host_model_review_receipt,
            allow_fixture_corrections=allow_fixture_corrections,
        )
        if not pairs:
            status = (
                "no_real_user_corrections"
                if summary["skipped_counts"].get("fixture_correction", 0)
                else "no_importable_reviewed_pairs"
            )
            return _error(
                status=status,
                code=status,
                message="No reviewed live-feedback corrections were importable.",
                details={"import_summary": summary},
                generated_at=generated_at,
                app_id=app_id,
                peer_id=peer_id,
                run_id=run_id,
            )
        rpp_run_id = f"route-matrix-live-feedback:{run_id}"
        route_action_response = generate_route_action_pairs(
            {"rpp_run_id": rpp_run_id},
            eval_cases,
            tool_registry=tool_registry,
            provider=HOST_MODEL_PROVIDER,
            host_model_id=_review_model_id(host_model_review_receipt),
            host_model_generate=lambda _messages, _max_tokens, _temperature: {
                "model_id": _review_model_id(host_model_review_receipt),
                "output": json.dumps({"pairs": pairs}, ensure_ascii=False),
            },
        )
    except (TypeError, ValueError) as exc:
        return _error(
            status="invalid_input",
            code="invalid_live_feedback_review_import_input",
            message=str(exc),
            details={},
            generated_at=generated_at,
            peer_id=peer_id,
        )

    if route_action_response.get("ok") is not True:
        return _error(
            status="route_action_normalization_failed",
            code="route_action_normalization_failed",
            message="Reviewed live feedback did not pass route/action normalization.",
            details={
                "route_action_error": route_action_response.get("error"),
                "import_summary": summary,
            },
            generated_at=generated_at,
            app_id=app_id,
            peer_id=peer_id,
            run_id=run_id,
        )

    normalized_output_dir = Path(output_dir)
    normalized_output_dir.mkdir(parents=True, exist_ok=True)
    effective_peer_id = _text(peer_id) or f"route-live-feedback:{app_id}"
    effective_event_store_path = Path(
        event_store_path
        or (normalized_output_dir / "route_matrix_live_feedback_events.sqlite")
    )
    owns_store = event_store is None
    store = event_store or EventStore(effective_event_store_path)
    try:
        storage = store_route_action_training_events(
            route_action_response,
            peer_id=effective_peer_id,
            event_store=store,
            timestamp_ms=_timestamp_ms(host_model_review_receipt),
            tool_registry=tool_registry,
        )
        dataset = write_route_action_learner_dataset_jsonl(
            peer_id=effective_peer_id,
            output_dir=normalized_output_dir,
            rpp_run_id=rpp_run_id,
            event_store=store,
            limit=limit,
        )
    except RouteActionEvalLeakageError as exc:
        return _error(
            status="eval_leakage_blocked",
            code="route_action_eval_leakage_blocked",
            message=str(exc),
            details={
                "eval_leakage_gate": exc.report,
                "import_summary": summary,
            },
            generated_at=generated_at,
            app_id=app_id,
            peer_id=effective_peer_id,
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(
            status="live_feedback_import_failed",
            code="route_matrix_live_feedback_import_failed",
            message=str(exc),
            details={"import_summary": summary},
            generated_at=generated_at,
            app_id=app_id,
            peer_id=effective_peer_id,
            run_id=run_id,
        )
    finally:
        if owns_store:
            store.close()

    if not isinstance(dataset, dict):
        return _error(
            status="learner_dataset_empty",
            code="learner_dataset_empty",
            message="No learner dataset samples were produced from reviewed live feedback.",
            details={"import_summary": summary},
            generated_at=generated_at,
            app_id=app_id,
            peer_id=effective_peer_id,
            run_id=run_id,
            storage=storage,
        )
    if dataset.get("ok") is not True:
        return _error(
            status=dataset.get("status") or "learner_dataset_blocked",
            code=dataset.get("status") or "learner_dataset_blocked",
            message="Learner dataset gate blocked reviewed live feedback.",
            details={
                "eval_leakage_gate": dataset.get("eval_leakage_gate"),
                "import_summary": summary,
            },
            generated_at=generated_at,
            app_id=app_id,
            peer_id=effective_peer_id,
            run_id=run_id,
            storage=storage,
            learner_dataset=dataset,
        )

    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_REVIEW_IMPORT_SCHEMA_VERSION,
        "status": "learner_dataset_written",
        "app_id": app_id,
        "peer_id": effective_peer_id,
        "run_id": run_id,
        "rpp_run_id": rpp_run_id,
        "storage": storage,
        "learner_dataset": dataset,
        "route_action_response": route_action_response,
        "import_summary": summary,
        "error": None,
        "audit": _audit(
            generated_at=generated_at,
            output_dir=normalized_output_dir,
            event_store_path=effective_event_store_path,
            review_request=review_request,
            host_model_review_receipt=host_model_review_receipt,
            storage=storage,
            dataset=dataset,
            import_summary=summary,
            writes_events=True,
        ),
    }


def _route_action_inputs_from_review(
    *,
    cases: list[dict[str, Any]],
    host_model_review_receipt: dict[str, Any],
    allow_fixture_corrections: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    review_items = _review_items(host_model_review_receipt)
    cases_by_event_id = {
        _text(case.get("source_event_id")): case
        for case in cases
        if _text(case.get("source_event_id"))
    }
    cases_by_case_id = {
        _text(case.get("case_id")): case
        for case in cases
        if _text(case.get("case_id"))
    }
    skipped: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    eval_cases: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    language_drift_cases: list[dict[str, Any]] = []
    for index, item in enumerate(review_items):
        if not _is_approved(item):
            _bump(skipped, "not_approved")
            continue
        case = _matching_case(
            item,
            cases_by_event_id=cases_by_event_id,
            cases_by_case_id=cases_by_case_id,
        )
        if case is None:
            _bump(skipped, "unmatched_review_case")
            continue
        case_key = _text(case.get("source_event_id")) or _text(case.get("case_id"))
        if case_key in seen_cases:
            _bump(skipped, "duplicate_review_case")
            continue
        correction = case.get("user_correction") if isinstance(case.get("user_correction"), dict) else {}
        if _is_fixture_correction(correction) and not allow_fixture_corrections:
            _bump(skipped, "fixture_correction")
            continue
        pair = _reviewed_pair(item, case=case, index=index)
        language_drift = _prompt_variant_language_drift(case=case, pair=pair)
        if language_drift:
            language_drift_cases.append(language_drift)
        pairs.append(pair)
        eval_cases.append(_eval_case(pair, case=case))
        seen_cases.add(case_key)

    language_drift_count = len(language_drift_cases)
    return pairs, eval_cases, {
        "review_item_count": len(review_items),
        "review_case_count": len(cases),
        "approved_count": sum(1 for item in review_items if _is_approved(item)),
        "importable_count": len(pairs),
        "skipped_counts": dict(sorted(skipped.items())),
        "source_language_preservation_required": True,
        "language_drift_detected": language_drift_count > 0,
        "language_drift_detected_count": language_drift_count,
        "language_drift_detection_directions": list(
            _LANGUAGE_DRIFT_DETECTION_DIRECTIONS
        ),
        "language_drift_detection_strategy": _LANGUAGE_DRIFT_DETECTION_STRATEGY,
        "cjk_ratio_threshold_applied": False,
        "language_drift_cases": language_drift_cases,
    }


def _reviewed_pair(item: dict[str, Any], *, case: dict[str, Any], index: int) -> dict[str, Any]:
    payload = _pair_payload(item)
    correction = case.get("user_correction") if isinstance(case.get("user_correction"), dict) else {}
    case_id = _text(item.get("case_id")) or _required_text(case.get("case_id"), "review_case.case_id")
    route_intent = _first_text(
        payload,
        ("route_intent", "expected_route_intent", "intent", "routeIntent"),
    ) or _text(correction.get("corrected_route_intent"))
    if not route_intent:
        raise ValueError(f"review item for {case_id} missing route_intent")
    prompt_variants = _text_list(
        payload.get("prompt_variants")
        or payload.get("training_prompts")
        or payload.get("variants")
    )
    if not prompt_variants:
        raise ValueError(f"review item for {case_id} missing prompt_variants")
    selected_tools = _text_list(
        payload.get("selected_tools")
        or payload.get("expected_selected_tools")
        or correction.get("corrected_selected_tools")
    )
    tool_call_plan = _list_of_dicts(
        payload.get("tool_call_plan")
        or payload.get("toolCallPlan")
        or payload.get("plan")
    )
    answer_discipline = _first_text(
        payload,
        ("answer_discipline", "answer_policy", "answerPolicy"),
    )
    rationale = _first_text(payload, ("rationale", "reason", "explanation"))
    if not answer_discipline:
        raise ValueError(f"review item for {case_id} missing answer_discipline")
    if not rationale:
        raise ValueError(f"review item for {case_id} missing rationale")
    return {
        "case_id": case_id,
        "case_idx": index + 1,
        "route_intent": route_intent,
        "prompt_variants": prompt_variants,
        "selected_tools": selected_tools,
        "tool_call_plan": tool_call_plan,
        "answer_discipline": answer_discipline,
        "rationale": rationale,
        "confidence": payload.get("confidence") or item.get("confidence"),
    }


def _eval_case(pair: dict[str, Any], *, case: dict[str, Any]) -> dict[str, Any]:
    selected_tools = _text_list(pair.get("selected_tools"))
    return {
        "case_id": pair["case_id"],
        "prompt": _required_text(case.get("input_text"), "review_case.input_text"),
        "expectations": {
            "selected_tools_exact": selected_tools,
        },
    }


def _prompt_variant_language_drift(
    *,
    case: dict[str, Any],
    pair: dict[str, Any],
) -> dict[str, Any] | None:
    source_text = _case_language_source_text(case)
    if not _has_cjk(source_text):
        return None
    prompt_variants = _text_list(pair.get("prompt_variants"))
    non_cjk_count = sum(1 for variant in prompt_variants if not _has_cjk(variant))
    if non_cjk_count <= 0:
        return None
    return {
        "case_id": _text(pair.get("case_id")) or _text(case.get("case_id")),
        "source_event_id": _text(case.get("source_event_id")) or None,
        "source_language": "cjk",
        "prompt_variant_count": len(prompt_variants),
        "non_matching_prompt_variant_count": non_cjk_count,
    }


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


def _pair_payload(item: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "route_action_pair",
        "route_action_seed",
        "seed_case",
        "pair",
        "candidate",
    ):
        value = item.get(key)
        if isinstance(value, dict):
            return value
    return item


def _review_request_result(review_request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(review_request, dict):
        raise TypeError("review_request must be an object")
    if review_request.get("ok") is not True:
        raise ValueError("review_request.ok must be true")
    result = review_request.get("result")
    if not isinstance(result, dict):
        raise ValueError("review_request.result must be an object")
    return result


def _review_request_cases(result: dict[str, Any]) -> list[dict[str, Any]]:
    value = result.get("review_cases")
    if not isinstance(value, list):
        raise ValueError("review_request.result.review_cases must be a list")
    return [item for item in value if isinstance(item, dict)]


def _review_items(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(receipt, dict):
        raise TypeError("host_model_review_receipt must be an object")
    result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
    output_schema = (
        receipt.get("output_schema")
        if isinstance(receipt.get("output_schema"), dict)
        else {}
    )
    output_schema_result = (
        output_schema.get("result")
        if isinstance(output_schema.get("result"), dict)
        else {}
    )
    for container in (result, receipt, output_schema_result, output_schema):
        for key in (
            "review_cases",
            "reviewed_cases",
            "reviews",
            "cases",
            "items",
            "outputs",
            "seed_candidates",
        ):
            value = container.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


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


def _is_fixture_review(receipt: dict[str, Any]) -> bool:
    audit = receipt.get("audit") if isinstance(receipt.get("audit"), dict) else {}
    return (
        audit.get("fixture_review_only") is True
        or audit.get("not_real_user_flywheel_evidence") is True
    )


def _is_fixture_correction(correction: dict[str, Any]) -> bool:
    return (
        correction.get("is_fixture") is True
        or _text(correction.get("correction_source")).casefold() == "fixture"
    )


def _is_approved(item: dict[str, Any]) -> bool:
    for key in ("decision", "review_decision", "status", "verdict", "label", "decision_label"):
        decision = _text(item.get(key)).casefold()
        if decision:
            return decision in _APPROVED_DECISIONS
    return False


def _review_model_id(receipt: dict[str, Any]) -> str | None:
    audit = receipt.get("audit") if isinstance(receipt.get("audit"), dict) else {}
    model_id = _text(audit.get("model_id") or audit.get("selected_model_id"))
    return model_id or None


def _timestamp_ms(receipt: dict[str, Any]) -> int:
    audit = receipt.get("audit") if isinstance(receipt.get("audit"), dict) else {}
    for value in (
        audit.get("generated_at_ms"),
        receipt.get("generated_at_ms"),
        receipt.get("created_at_ms"),
    ):
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return int(time.time() * 1000)


def _audit(
    *,
    generated_at: str,
    output_dir: Path | None = None,
    event_store_path: Path | None = None,
    review_request: dict[str, Any] | None = None,
    host_model_review_receipt: dict[str, Any] | None = None,
    storage: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    import_summary: dict[str, Any] | None = None,
    writes_events: bool,
) -> dict[str, Any]:
    storage_audit = (
        storage.get("audit")
        if isinstance(storage, dict) and isinstance(storage.get("audit"), dict)
        else {}
    )
    return {
        "schema_version": "edgestudio.route_matrix_live_feedback_review_import_audit.v0",
        "generated_at": generated_at,
        "training_side_only": True,
        "writes_events": bool(writes_events),
        "writes_runtime_artifacts": False,
        "writes_training_sample_tags": bool(writes_events),
        "warnings": _audit_warnings(import_summary),
        "source_review_request_fingerprint": (
            _fingerprint(review_request) if review_request is not None else None
        ),
        "host_model_review_fingerprint": (
            _fingerprint(host_model_review_receipt)
            if host_model_review_receipt is not None
            else None
        ),
        "import_summary": import_summary or {},
        "output_dir": str(output_dir) if output_dir is not None else None,
        "event_store_path": str(event_store_path) if event_store_path is not None else None,
        "learner_dataset_schema_version": ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
        "stored_event_ids_fingerprint": storage_audit.get("event_ids_fingerprint"),
        "sample_count": dataset.get("sample_count") if isinstance(dataset, dict) else 0,
    }


def _audit_warnings(import_summary: dict[str, Any] | None) -> list[str]:
    if (
        isinstance(import_summary, dict)
        and import_summary.get("language_drift_detected") is True
    ):
        return [_LANGUAGE_DRIFT_WARNING]
    return []


def _error(
    *,
    status: str,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
    app_id: str | None = None,
    peer_id: str | None = None,
    run_id: str | None = None,
    storage: dict[str, Any] | None = None,
    learner_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_REVIEW_IMPORT_SCHEMA_VERSION,
        "status": status,
        "app_id": app_id,
        "peer_id": peer_id,
        "run_id": run_id,
        "storage": storage,
        "learner_dataset": learner_dataset,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": _audit(
            generated_at=generated_at,
            import_summary=details.get("import_summary") if isinstance(details, dict) else None,
            writes_events=storage is not None,
        ),
    }


def _first_text(value: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _text(value.get(key))
        if text:
            return text
    return ""


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        text = _text(value)
        return [text] if text else []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _text(item)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


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
    return any("\u3400" <= char <= "\u9fff" for char in _text(value))


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
