# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Bridge exported chat corrections into the structured correction ledger.

This module is deliberately narrow. It reviews exported ChatCorrection DPO
records with the host model, writes approved profile corrections to the local
correction ledger, and stops there. It does not regenerate Neural Imprint
artifacts or push anything to devices.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

from backend.services.correction_ledger import (
    CorrectionLedgerError,
    record_correction_entry,
)
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


CHAT_CORRECTION_FEEDBACK_BRIDGE_SCHEMA_VERSION = (
    "edgestudio.chat_correction_feedback_bridge.v0"
)
CHAT_CORRECTION_HOST_REVIEW_SCHEMA_VERSION = (
    "edgestudio.chat_correction_host_review.v0"
)

_APPROVED_DECISIONS = frozenset(
    {"approve", "approved", "accept", "accepted", "pass", "ok"}
)


def run_chat_correction_feedback_bridge(
    *,
    peer_id: str,
    app_id: str,
    preferences_jsonl_path: str | Path | None = None,
    records: Sequence[dict[str, Any]] | None = None,
    ledger_root: Path | None = None,
    provider: str = PROVIDER,
    host_model_id: str | None = None,
    host_model_generate: HostModelGenerate | None = None,
    max_tokens: int | None = None,
    allow_fixture_corrections: bool = False,
) -> dict[str, Any]:
    """Review exported ChatCorrection DPO records and write ledger entries."""

    generated_at_ms = int(time.time() * 1000)
    clean_peer_id = _required_text(peer_id, "peer_id")
    clean_app_id = _text(app_id)
    selected_provider = provider or PROVIDER
    if selected_provider not in SUPPORTED_PROVIDERS:
        return _error(
            status="invalid_provider",
            code="invalid_provider",
            message="Unsupported chat-correction feedback provider.",
            details={
                "provider": selected_provider,
                "supported": sorted(SUPPORTED_PROVIDERS),
            },
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
        )

    try:
        source_records = list(records) if records is not None else _read_jsonl(
            preferences_jsonl_path
        )
        eligible, skipped = _eligible_cases(
            source_records,
            allow_fixture_corrections=allow_fixture_corrections,
        )
    except (OSError, TypeError, ValueError) as exc:
        return _error(
            status="invalid_input",
            code="invalid_chat_correction_feedback_input",
            message=str(exc),
            details={},
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
        )

    if not eligible:
        return _response(
            status="no_eligible_chat_corrections",
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
            review_request=_review_request(clean_peer_id, clean_app_id, eligible),
            host_model_review=None,
            ledger_entries=[],
            counts=_counts(source_records, eligible, skipped, written=0),
            warnings=[],
            host_model_called=False,
            writes_ledger=False,
        )

    review_request = _review_request(clean_peer_id, clean_app_id, eligible)
    if selected_provider not in HOST_MODEL_PROVIDERS:
        return _response(
            status="pending_host_model_review",
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
            review_request=review_request,
            host_model_review=None,
            ledger_entries=[],
            counts=_counts(source_records, eligible, skipped, written=0),
            warnings=["chat_correction_feedback_pending_host_model_review"],
            host_model_called=False,
            writes_ledger=False,
        )

    try:
        review_payload, host_model = _generate_host_review(
            review_request=review_request,
            host_model_id=host_model_id,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
        )
        reviews = _reviews(review_payload)
        validation = _validate_reviews(reviews, eligible)
        if (
            validation["approved_invalid_count"]
            or validation["approved_missing_case_id_count"]
            or validation["unknown_case_ids"]
            or validation["approved_correction_id_mismatches"]
        ):
            return _error(
                status="host_model_review_contract_failed",
                code="host_model_review_contract_failed",
                message=(
                    "Approved chat-correction reviews must include a complete "
                    "profile_correction target/profile_overlay and must match "
                    "an input review case/correction id."
                ),
                details={"review_summary": validation},
                peer_id=clean_peer_id,
                app_id=clean_app_id,
                generated_at_ms=generated_at_ms,
                provider=selected_provider,
                host_model_id=host_model_id,
                host_model=host_model,
                review_request=review_request,
                host_model_review=review_payload,
            )
        written_entries = _write_approved_profile_corrections(
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            cases=eligible,
            reviews=reviews,
            ledger_root=ledger_root,
            received_at_ms=generated_at_ms,
        )
    except HostModelAssistantError as exc:
        return _error(
            status="host_model_failed",
            code=exc.code,
            message=exc.message,
            details=exc.details,
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
        )
    except (CorrectionLedgerError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _error(
            status="chat_correction_feedback_bridge_failed",
            code=getattr(exc, "code", "chat_correction_feedback_bridge_failed"),
            message=getattr(exc, "message", str(exc)),
            details=getattr(exc, "details", {}) if hasattr(exc, "details") else {},
            peer_id=clean_peer_id,
            app_id=clean_app_id,
            generated_at_ms=generated_at_ms,
            provider=selected_provider,
            host_model_id=host_model_id,
        )

    return _response(
        status="structured_profile_corrections_recorded"
        if written_entries
        else "no_approved_profile_corrections",
        peer_id=clean_peer_id,
        app_id=clean_app_id,
        generated_at_ms=generated_at_ms,
        provider=selected_provider,
        host_model_id=host_model_id,
        review_request=review_request,
        host_model_review={
            "payload": review_payload,
            "host_model": host_model,
            "summary": _review_summary(reviews),
        },
        ledger_entries=written_entries,
        counts=_counts(source_records, eligible, skipped, written=len(written_entries)),
        warnings=[],
        host_model_called=True,
        writes_ledger=bool(written_entries),
    )


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        raise ValueError("preferences_jsonl_path or records is required")
    source = Path(path).expanduser()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        records.append(value)
    return records


def _eligible_cases(
    records: Sequence[dict[str, Any]],
    *,
    allow_fixture_corrections: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[dict[str, Any]] = []
    skipped = {
        "not_chat_correction_dpo": 0,
        "fixture_correction": 0,
        "missing_required_text": 0,
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            skipped["not_chat_correction_dpo"] += 1
            continue
        if record.get("record_type") != "dpo_pair" or record.get("source") != "chat_correction":
            skipped["not_chat_correction_dpo"] += 1
            continue
        if record.get("is_fixture") is not False and not allow_fixture_corrections:
            skipped["fixture_correction"] += 1
            continue
        source_input = _message_text(record.get("prompt"), preferred_role="user")
        chosen = _message_text(record.get("chosen"), preferred_role="assistant")
        rejected = _message_text(record.get("rejected"), preferred_role="assistant")
        if not source_input or not chosen or not rejected:
            skipped["missing_required_text"] += 1
            continue
        record_fingerprint = _sha256_json(
            {
                "prompt": source_input,
                "chosen": chosen,
                "rejected": rejected,
                "correction_id": _text(record.get("correction_id")),
            }
        )
        correction_id = _text(record.get("correction_id")) or f"chat-corr-{record_fingerprint[:16]}"
        eligible.append(
            {
                "case_id": f"chat-correction-{record_fingerprint[:16]}",
                "correction_id": correction_id,
                "record_index": index,
                "record_fingerprint": record_fingerprint,
                "source_input": source_input,
                "rejected_answer": rejected,
                "corrected_answer": chosen,
                "correction_source": _text(record.get("correction_source")),
                "created_at": _text(record.get("created_at")),
            }
        )
    return eligible, skipped


def _review_request(peer_id: str, app_id: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": CHAT_CORRECTION_HOST_REVIEW_SCHEMA_VERSION,
        "task": "review_chat_corrections_and_emit_profile_corrections",
        "peer_id": peer_id,
        "app_id": app_id,
        "review_cases": cases,
        "output_constraints": {
            "valid_json_only": True,
            "approved_items_require_profile_correction": True,
            "profile_correction_only": True,
            "no_fact_correction_without_fact_id": True,
            "max_rationale_characters": 160,
        },
        "output_schema": {
            "reviews": [
                {
                    "case_id": "copy from review_cases[]",
                    "correction_id": "copy from review_cases[]",
                    "verdict": "approved|reject",
                    "rationale": "short review rationale",
                    "profile_correction": {
                        "target": {
                            "profile_field": "stable profile dimension",
                        },
                        "profile_overlay": {
                            "preference": "stable user preference or behavior",
                            "rationale": "why this correction should affect profile",
                        },
                    },
                }
            ]
        },
    }


def _generate_host_review(
    *,
    review_request: dict[str, Any],
    host_model_id: str | None,
    host_model_generate: HostModelGenerate | None,
    max_tokens: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = _review_messages(review_request)
    token_budget = max_tokens or max(2_000, 900 + len(review_request["review_cases"]) * 500)
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
            message="Host Model chat-correction review call failed.",
            retryable=True,
            details={"reason": str(exc), "model_id": None},
        ) from exc

    raw_output = _host_model_output_text(raw)
    model_id = _host_model_id(raw)
    if not raw_output:
        raise HostModelAssistantError(
            code="host_model_empty_output",
            message="Host Model returned an empty chat-correction review response.",
            retryable=True,
            details={"model_id": model_id},
        )
    payload = json.loads(_extract_json_block(raw_output))
    if not isinstance(payload, dict):
        raise TypeError("Host Model chat-correction review output must be an object")
    return payload, {
        "enabled": True,
        "model_id": model_id,
        "selected_model_id": host_model_id,
        "raw_output_fingerprint": _sha256_text(raw_output),
    }


def _review_messages(review_request: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the EdgeStudio Host Model for feedback learning. "
                "Review explicit user chat corrections and emit a structured "
                "profile_correction only when the correction reveals a stable "
                "user preference, work pattern, or profile-level behavior. "
                "Do not emit fact_correction entries unless a stable fact ID "
                "and schema are present; this bridge only accepts "
                "profile_correction output. Reject one-off answer mistakes, "
                "ambiguous corrections, or corrections that only restate the "
                "desired answer. Return compact valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(review_request, ensure_ascii=False, sort_keys=True),
        },
    ]


def _reviews(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    for container in (result, payload):
        value = container.get("reviews")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError("Host Model chat-correction review output missing reviews list")


def _validate_reviews(
    reviews: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    cases_by_case_id = {case["case_id"]: case for case in cases}
    approved = 0
    rejected = 0
    invalid = 0
    missing_case_id = 0
    unknown_case_ids: list[str] = []
    correction_id_mismatches: list[dict[str, str]] = []
    for item in reviews:
        case_id = _text(item.get("case_id"))
        if not _is_approved(item):
            rejected += 1
            continue
        approved += 1
        case = cases_by_case_id.get(case_id)
        if not case_id:
            missing_case_id += 1
        elif case is None:
            unknown_case_ids.append(case_id)
        elif _text(item.get("correction_id")) != case["correction_id"]:
            correction_id_mismatches.append(
                {
                    "case_id": case_id,
                    "expected": case["correction_id"],
                    "actual": _text(item.get("correction_id")),
                }
            )
        if not _complete_profile_correction(item.get("profile_correction")):
            invalid += 1
    return {
        "reviewed_count": len(reviews),
        "approved_count": approved,
        "rejected_count": rejected,
        "approved_invalid_count": invalid,
        "approved_missing_case_id_count": missing_case_id,
        "unknown_case_ids": sorted(set(unknown_case_ids)),
        "approved_correction_id_mismatches": correction_id_mismatches,
        "ready_for_ledger_write": (
            approved > 0
            and invalid == 0
            and missing_case_id == 0
            and not unknown_case_ids
            and not correction_id_mismatches
        ),
    }


def _write_approved_profile_corrections(
    *,
    peer_id: str,
    app_id: str,
    cases: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    ledger_root: Path | None,
    received_at_ms: int,
) -> list[dict[str, Any]]:
    cases_by_case_id = {case["case_id"]: case for case in cases}
    written: list[dict[str, Any]] = []
    for item in reviews:
        if not _is_approved(item):
            continue
        case = cases_by_case_id.get(_text(item.get("case_id")))
        profile_correction = _object(item.get("profile_correction"))
        if case is None or not _complete_profile_correction(profile_correction):
            continue
        target = _object(profile_correction.get("target"))
        overlay = _object(profile_correction.get("profile_overlay"))
        receipt = record_correction_entry(
            {
                "peer_id": peer_id,
                "app_id": app_id,
                "correction_type": "profile_correction",
                "source": {
                    "source": "chat_correction_feedback_bridge",
                    "chat_correction_id": case["correction_id"],
                    "record_fingerprint": case["record_fingerprint"],
                    "host_review_case_id": case["case_id"],
                },
                "target": target,
                "correction": {"profile_overlay": overlay},
            },
            root=ledger_root,
            received_at_ms=received_at_ms,
        )
        written.append(
            {
                "correction_id": receipt["correction_id"],
                "correction_fingerprint": receipt["entry"]["correction_fingerprint"],
                "status": receipt["status"],
                "ledger_entry": receipt["entry"],
            }
        )
    return written


def _response(
    *,
    status: str,
    peer_id: str,
    app_id: str,
    generated_at_ms: int,
    provider: str,
    host_model_id: str | None,
    review_request: dict[str, Any],
    host_model_review: dict[str, Any] | None,
    ledger_entries: list[dict[str, Any]],
    counts: dict[str, int],
    warnings: list[str],
    host_model_called: bool,
    writes_ledger: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        "schema_version": CHAT_CORRECTION_FEEDBACK_BRIDGE_SCHEMA_VERSION,
        "status": status,
        "peer_id": peer_id,
        "app_id": app_id,
        "generated_at_ms": generated_at_ms,
        "review_request": review_request,
        "host_model_review": host_model_review,
        "ledger_entries": ledger_entries,
        "counts": counts,
        "warnings": warnings,
        "error": None,
        "audit": _audit(
            provider=provider,
            host_model_id=host_model_id,
            host_model_called=host_model_called,
            writes_ledger=writes_ledger,
            warnings=warnings,
        ),
    }


def _error(
    *,
    status: str,
    code: str,
    message: str,
    details: dict[str, Any],
    peer_id: str,
    app_id: str,
    generated_at_ms: int,
    provider: str,
    host_model_id: str | None,
    host_model: dict[str, Any] | None = None,
    review_request: dict[str, Any] | None = None,
    host_model_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structured_host_model_review = host_model_review
    if host_model_review is not None and host_model is not None:
        structured_host_model_review = {
            "payload": host_model_review,
            "host_model": host_model,
        }
    return {
        "ok": False,
        "schema_version": CHAT_CORRECTION_FEEDBACK_BRIDGE_SCHEMA_VERSION,
        "status": status,
        "peer_id": peer_id,
        "app_id": app_id,
        "generated_at_ms": generated_at_ms,
        "review_request": review_request,
        "host_model_review": structured_host_model_review,
        "ledger_entries": [],
        "counts": {},
        "warnings": [],
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": _audit(
            provider=provider,
            host_model_id=host_model_id,
            host_model_called=bool(host_model),
            writes_ledger=False,
            warnings=[],
        ),
    }


def _audit(
    *,
    provider: str,
    host_model_id: str | None,
    host_model_called: bool,
    writes_ledger: bool,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "provider": provider,
        "host_model": {
            "enabled": provider == HOST_MODEL_PROVIDER or provider in HOST_MODEL_PROVIDERS,
            "selected_model_id": host_model_id,
            "called": host_model_called,
        },
        "training_side_only": True,
        "writes_correction_ledger": writes_ledger,
        "writes_runtime_artifacts": False,
        "triggers_rpp": False,
        "triggers_neural_imprint_regen": False,
        "triggers_capsule_push": False,
        "automatic_push": False,
        "output_type": "profile_correction",
        "fixture_records_blocked_by_default": True,
        "warnings": warnings,
    }


def _counts(
    records: Sequence[dict[str, Any]],
    eligible: Sequence[dict[str, Any]],
    skipped: dict[str, int],
    *,
    written: int,
) -> dict[str, int]:
    return {
        "input_records": len(records),
        "eligible_chat_corrections": len(eligible),
        "ledger_entries_written": written,
        **skipped,
    }


def _review_summary(reviews: list[dict[str, Any]]) -> dict[str, int]:
    approved = sum(1 for item in reviews if _is_approved(item))
    return {
        "reviewed_count": len(reviews),
        "approved_count": approved,
        "rejected_count": len(reviews) - approved,
    }


def _complete_profile_correction(value: Any) -> bool:
    item = _object(value)
    target = _object(item.get("target"))
    overlay = _object(item.get("profile_overlay"))
    return bool(
        overlay
        and (_text(target.get("direction_id")) or _text(target.get("profile_field")))
    )


def _is_approved(value: dict[str, Any]) -> bool:
    return _text(value.get("verdict")).casefold() in _APPROVED_DECISIONS


def _message_text(value: Any, *, preferred_role: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            if _text(item.get("role")).casefold() == preferred_role:
                return _text(item.get("content"))
        for item in value:
            if isinstance(item, dict) and _text(item.get("content")):
                return _text(item.get("content"))
    return ""


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
