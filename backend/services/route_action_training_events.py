# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Bridge host-model route/action output into standard learning events.

The host model owns the route intent and tool/action semantics. This module
only turns a reviewed `generate_route_action_pairs(...)` envelope into local
EventStore rows and runtime artifacts that the device can consume.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from backend.services.route_action_tool_contracts import (
    validate_route_action_tool_contracts,
)
from backend.stores.event_store import DataEvent, EventStore, get_default_store


ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION = (
    "edgestudio.route_action_training_events.v0"
)
ROUTE_ACTION_EVAL_LEAKAGE_GATE_SCHEMA_VERSION = (
    "edgestudio.route_action_eval_leakage_gate.v0"
)
ROUTE_ACTION_APP_ID = "com.edgestudio.host_model_assistant"
ROUTE_ACTION_EVENT_TYPE = "route_action_pair"
ROUTE_ACTION_EVENT_TAGS = (
    "trainingSample",
    "route_action_pair",
    "host_model_generated",
)
ROUTE_TRAINING_PAIRS_ARTIFACT_SCHEMA_VERSION = (
    "edgeruntime.route_training_pairs_artifact.v0"
)
ROUTE_TRAINING_PAIRS_ARTIFACT_NAME = "route_training_pairs.json"
ROUTE_TRAINING_PAIRS_HARD_FACT_GATE_SCHEMA_VERSION = (
    "edgestudio.route_training_pairs_hard_fact_leakage_gate.v0"
)
ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION = (
    "edgestudio.route_action_learner_dataset.v0"
)
ROUTE_ACTION_LEARNER_SAMPLE_SCHEMA_VERSION = (
    "edgestudio.route_action_learner_sample.v0"
)
ROUTE_ACTION_LEARNER_DATASET_NAME = "route_action_policy_dataset.jsonl"


class RouteActionEvalLeakageError(ValueError):
    """Raised when route/action training would exactly replay eval prompts."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(
            "route/action eval leakage gate blocked training data: "
            f"overlap_count={report.get('overlap_count', 0)} "
            f"unverifiable_count={report.get('unverifiable_count', 0)}"
        )


def build_route_action_training_events(
    route_action_response: dict[str, Any],
    *,
    peer_id: str,
    timestamp_ms: int | None = None,
    app_id: str = ROUTE_ACTION_APP_ID,
    tool_registry: list[dict[str, Any]] | None = None,
    tool_contracts: list[dict[str, Any]] | None = None,
) -> list[DataEvent]:
    """Build deterministic `route_action_pair` events from a host-model envelope."""

    normalized_peer_id = _required_str(peer_id, "peer_id")
    result = _route_action_result(route_action_response)
    rpp_run_id = _required_str(result.get("rpp_run_id"), "result.rpp_run_id")
    pairs = result.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("result.pairs must be a list")
    _assert_route_action_eval_leakage_gate(
        pairs,
        context="route_action_training_events",
    )

    effective_timestamp_ms = int(timestamp_ms or time.time() * 1000)
    events: list[DataEvent] = []
    for index, raw_pair in enumerate(pairs):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"result.pairs[{index}] must be an object")
        payload = _payload_from_pair(
            raw_pair,
            rpp_run_id=rpp_run_id,
            tool_registry=tool_registry,
            tool_contracts=tool_contracts,
        )
        payload_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        events.append(
            DataEvent(
                id=_event_id(
                    peer_id=normalized_peer_id,
                    rpp_run_id=rpp_run_id,
                    pair=payload,
                ),
                timestamp_ms=effective_timestamp_ms,
                app_id=app_id,
                event_type=ROUTE_ACTION_EVENT_TYPE,
                payload=payload_bytes,
                tags=list(ROUTE_ACTION_EVENT_TAGS),
                source_peer_id=normalized_peer_id,
            )
        )
    return events


def store_route_action_training_events(
    route_action_response: dict[str, Any],
    *,
    peer_id: str,
    event_store: EventStore | None = None,
    timestamp_ms: int | None = None,
    tool_registry: list[dict[str, Any]] | None = None,
    tool_contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and insert route/action training events into EventStore."""

    events = build_route_action_training_events(
        route_action_response,
        peer_id=peer_id,
        timestamp_ms=timestamp_ms,
        tool_registry=tool_registry,
        tool_contracts=tool_contracts,
    )
    store = event_store or get_default_store()
    received_ids, is_new_flags, updated_flags = store.upsert_batch_by_payload(events)
    return {
        "ok": True,
        "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": peer_id,
        "event_type": ROUTE_ACTION_EVENT_TYPE,
        "event_count": len(events),
        "inserted_count": sum(1 for flag in is_new_flags if flag),
        "updated_count": sum(1 for flag in updated_flags if flag),
        "received_ids": received_ids,
        "audit": {
            "tags": list(ROUTE_ACTION_EVENT_TAGS),
            "event_ids_fingerprint": _fingerprint(received_ids),
            "eval_leakage_gate": evaluate_route_action_eval_leakage(
                route_action_response
            ),
        },
    }


def build_route_training_pairs_artifact(
    *,
    peer_id: str,
    rpp_run_id: str | None = None,
    event_store: EventStore | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Build EdgeRuntime `RouteTrainingPair` JSON from stored route/action events.

    This is a data bridge only: host-model route/action outputs remain the source
    of truth. Python maps the host-model route labels onto the EdgeRuntime
    contract and preserves the prompt/rationale/audit metadata for the device.
    Eval leakage failures are returned as an envelope so callers do not need to
    know a separate exception contract for artifact construction.
    """

    normalized_peer_id = _required_str(peer_id, "peer_id")
    store = event_store or get_default_store()
    events = store.query(
        event_type=ROUTE_ACTION_EVENT_TYPE,
        source_peer_id=normalized_peer_id,
        limit=limit,
    )

    pairs: list[dict[str, Any]] = []
    gate_pairs: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for event in events:
        payload = _json_event_payload(event)
        if payload is None:
            continue
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        event_rpp_run_id = _optional_str(source.get("rpp_run_id"))
        if rpp_run_id and event_rpp_run_id != rpp_run_id:
            continue

        event_pairs = _route_training_pairs_from_payload(
            payload,
            event=event,
            event_rpp_run_id=event_rpp_run_id,
        )
        for pair in event_pairs:
            pair_prompt = _required_str(
                pair.get("input", {}).get("text")
                if isinstance(pair.get("input"), dict)
                else None,
                "route_training_pair.input.text",
            )
            pair_prompt_key = _route_training_prompt_key(pair_prompt)
            if pair_prompt_key in seen_prompts:
                continue
            pairs.append(pair)
            gate_pairs.append(
                _eval_gate_pair_from_route_training_pair(
                    payload=payload,
                    pair=pair,
                    event_id=event.id,
                )
            )
            seen_prompts.add(pair_prompt_key)
    gate_report = _route_action_eval_leakage_report(
        gate_pairs,
        context="route_training_pairs_artifact",
    )
    if gate_report["ok"] is not True:
        return _route_training_pairs_artifact_envelope(
            ok=False,
            status="eval_leakage_blocked",
            pairs=[],
            rpp_run_id=rpp_run_id,
            eval_leakage_gate=gate_report,
        )
    return _route_training_pairs_artifact_envelope(
        ok=True,
        status="built" if pairs else "empty",
        pairs=pairs,
        rpp_run_id=rpp_run_id,
        eval_leakage_gate=gate_report,
    )


def write_route_training_pairs_artifact_to_directory(
    *,
    peer_id: str,
    artifact_dir: Path,
    rpp_run_id: str | None = None,
    event_store: EventStore | None = None,
    limit: int = 1000,
    require_hard_fact_review: bool = False,
    hard_fact_forbidden_entities: list[str] | None = None,
    hard_fact_review_host_model_id: str | None = None,
    hard_fact_review_provider: str | None = None,
    hard_fact_review_generate: Any | None = None,
) -> dict[str, Any] | None:
    """Write `route_training_pairs.json` into a runtime artifact dir when pairs exist."""

    artifact = build_route_training_pairs_artifact(
        peer_id=peer_id,
        rpp_run_id=rpp_run_id,
        event_store=event_store,
        limit=limit,
    )
    if artifact.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": ROUTE_TRAINING_PAIRS_ARTIFACT_SCHEMA_VERSION,
            "status": artifact.get("status") or "build_failed",
            "name": ROUTE_TRAINING_PAIRS_ARTIFACT_NAME,
            "path": None,
            "pair_count": 0,
            "rpp_run_id": rpp_run_id,
            "eval_leakage_gate": artifact.get("eval_leakage_gate"),
        }
    pairs = artifact.get("pairs") if isinstance(artifact.get("pairs"), list) else []
    if not pairs:
        return None

    hard_fact_gate: dict[str, Any] | None = None
    if require_hard_fact_review:
        hard_fact_gate = review_route_training_pairs_hard_fact_leakage(
            pairs,
            hard_fact_forbidden_entities or [],
            host_model_id=hard_fact_review_host_model_id,
            provider=hard_fact_review_provider,
            host_model_generate=hard_fact_review_generate,
        )
        if hard_fact_gate.get("ok") is not True:
            return {
                "ok": False,
                "schema_version": ROUTE_TRAINING_PAIRS_ARTIFACT_SCHEMA_VERSION,
                "status": "hard_fact_leakage_blocked",
                "name": ROUTE_TRAINING_PAIRS_ARTIFACT_NAME,
                "path": None,
                "pair_count": 0,
                "rpp_run_id": rpp_run_id,
                "eval_leakage_gate": artifact.get("eval_leakage_gate"),
                "hard_fact_leakage_gate": hard_fact_gate,
            }

    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / ROUTE_TRAINING_PAIRS_ARTIFACT_NAME
    data = json.dumps(
        pairs,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    artifact_path.write_bytes(data)
    return {
        "ok": True,
        "schema_version": ROUTE_TRAINING_PAIRS_ARTIFACT_SCHEMA_VERSION,
        "status": "written",
        "name": ROUTE_TRAINING_PAIRS_ARTIFACT_NAME,
        "path": str(artifact_path),
        "pair_count": len(pairs),
        "rpp_run_id": rpp_run_id,
        "sha256": hashlib.sha256(data).hexdigest(),
        "hard_fact_leakage_gate": hard_fact_gate,
    }


def build_route_training_pairs_hard_fact_review_samples(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert final runtime route pairs into host-model leakage-review samples."""

    samples: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        pair_input = pair.get("input") if isinstance(pair.get("input"), dict) else {}
        text = _optional_str(pair_input.get("text"))
        if not text:
            continue
        metadata = pair.get("metadata") if isinstance(pair.get("metadata"), dict) else {}
        sample_payload = {
            "input_text": text,
            "expected_intent_tag": _optional_str(pair.get("expectedIntentTag")),
            "selected_tool_names": _string_list(pair.get("selectedToolNames")),
            "tool_call_plan": _edge_runtime_tool_call_plan(pair.get("toolCallPlan")),
        }
        case_id = _optional_str(metadata.get("case_id"))
        rpp_run_id = _optional_str(metadata.get("rpp_run_id"))
        sample_fingerprint = _fingerprint(sample_payload)
        sample_source: dict[str, Any] = {
            "kind": "route_training_pair_runtime_artifact",
            "line": index + 1,
            "file_name": ROUTE_TRAINING_PAIRS_ARTIFACT_NAME,
            "sample_fingerprint": sample_fingerprint,
        }
        if case_id:
            sample_source["case_id"] = case_id
        if rpp_run_id:
            sample_source["rpp_run_id"] = rpp_run_id
        samples.append(
            {
                "sample_id": f"route_pair:{sample_fingerprint.removeprefix('sha256:')[:16]}",
                "messages": [
                    {"role": "user", "content": text},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "route": sample_payload["expected_intent_tag"],
                                "selected_tools": sample_payload["selected_tool_names"],
                                "tool_call_plan": sample_payload["tool_call_plan"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
                "source": sample_source,
            }
        )
    return samples


def review_route_training_pairs_hard_fact_leakage(
    pairs: list[dict[str, Any]],
    forbidden_entities: list[str],
    *,
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: Any | None = None,
) -> dict[str, Any]:
    """Review final runtime route pairs before shipping them in an artifact."""

    samples = build_route_training_pairs_hard_fact_review_samples(pairs)
    effective_provider = provider
    if effective_provider is None:
        from backend.services.host_model_assistant import HOST_MODEL_PROVIDER

        effective_provider = HOST_MODEL_PROVIDER
    from backend.services.host_model_assistant import review_hard_fact_leakage

    response = review_hard_fact_leakage(
        samples,
        forbidden_entities,
        host_model_id=host_model_id,
        provider=effective_provider,
        host_model_generate=host_model_generate,
    )
    return _route_training_pairs_hard_fact_gate_from_review(
        response,
        sample_count=len(samples),
    )


def build_route_action_learner_dataset(
    *,
    peer_id: str,
    rpp_run_id: str | None = None,
    event_store: EventStore | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Build a gated route/action learner dataset from stored evidence.

    This is the feedstock for the learned-router/distillation path. It is
    deliberately separate from `route_training_pairs.json`, which remains a
    runtime evidence replay artifact and safety fallback.
    """

    normalized_peer_id = _required_str(peer_id, "peer_id")
    store = event_store or get_default_store()
    events = store.query(
        event_type=ROUTE_ACTION_EVENT_TYPE,
        source_peer_id=normalized_peer_id,
        limit=limit,
    )

    samples: list[dict[str, Any]] = []
    gate_pairs: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for event in events:
        payload = _json_event_payload(event)
        if payload is None:
            continue
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        event_rpp_run_id = _optional_str(source.get("rpp_run_id"))
        if rpp_run_id and event_rpp_run_id != rpp_run_id:
            continue

        event_pairs = _route_training_pairs_from_payload(
            payload,
            event=event,
            event_rpp_run_id=event_rpp_run_id,
        )
        for pair in event_pairs:
            pair_prompt = _required_str(
                pair.get("input", {}).get("text")
                if isinstance(pair.get("input"), dict)
                else None,
                "route_action_learner_sample.input.text",
            )
            pair_prompt_key = _route_training_prompt_key(pair_prompt)
            if pair_prompt_key in seen_prompts:
                continue
            samples.append(
                _route_action_learner_sample_from_pair(
                    payload=payload,
                    pair=pair,
                    event=event,
                    event_rpp_run_id=event_rpp_run_id,
                )
            )
            gate_pairs.append(
                _eval_gate_pair_from_route_training_pair(
                    payload=payload,
                    pair=pair,
                    event_id=event.id,
                )
            )
            seen_prompts.add(pair_prompt_key)

    gate_report = _route_action_eval_leakage_report(
        gate_pairs,
        context="route_action_learner_dataset",
    )
    if gate_report["ok"] is not True:
        return _route_action_learner_dataset_envelope(
            ok=False,
            status="eval_leakage_blocked",
            samples=[],
            rpp_run_id=rpp_run_id,
            eval_leakage_gate=gate_report,
        )
    return _route_action_learner_dataset_envelope(
        ok=True,
        status="built" if samples else "empty",
        samples=samples,
        rpp_run_id=rpp_run_id,
        eval_leakage_gate=gate_report,
    )


def write_route_action_learner_dataset_jsonl(
    *,
    peer_id: str,
    output_dir: Path,
    rpp_run_id: str | None = None,
    event_store: EventStore | None = None,
    limit: int = 1000,
) -> dict[str, Any] | None:
    """Write a route/action learner JSONL dataset when gated samples exist."""

    dataset = build_route_action_learner_dataset(
        peer_id=peer_id,
        rpp_run_id=rpp_run_id,
        event_store=event_store,
        limit=limit,
    )
    if dataset.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
            "status": dataset.get("status") or "build_failed",
            "name": ROUTE_ACTION_LEARNER_DATASET_NAME,
            "path": None,
            "sample_count": 0,
            "rpp_run_id": rpp_run_id,
            "eval_leakage_gate": dataset.get("eval_leakage_gate"),
        }
    samples = dataset.get("samples") if isinstance(dataset.get("samples"), list) else []
    if not samples:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / ROUTE_ACTION_LEARNER_DATASET_NAME
    data = (
        "\n".join(
            json.dumps(sample, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for sample in samples
        )
        + "\n"
    ).encode("utf-8")
    dataset_path.write_bytes(data)
    return {
        "ok": True,
        "schema_version": ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
        "status": "written",
        "name": ROUTE_ACTION_LEARNER_DATASET_NAME,
        "path": str(dataset_path),
        "sample_count": len(samples),
        "rpp_run_id": rpp_run_id,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _route_training_pairs_artifact_envelope(
    *,
    ok: bool,
    status: str,
    pairs: list[dict[str, Any]],
    rpp_run_id: str | None,
    eval_leakage_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "schema_version": ROUTE_TRAINING_PAIRS_ARTIFACT_SCHEMA_VERSION,
        "status": status,
        "name": ROUTE_TRAINING_PAIRS_ARTIFACT_NAME,
        "path": None,
        "pair_count": len(pairs),
        "rpp_run_id": rpp_run_id,
        "pairs": pairs,
        "eval_leakage_gate": eval_leakage_gate,
    }


def _route_training_pairs_hard_fact_gate_from_review(
    response: dict[str, Any],
    *,
    sample_count: int,
) -> dict[str, Any]:
    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, dict):
        result = {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    status = _optional_str(result.get("status"))
    decision = _optional_str(result.get("decision"))
    leakage_count = _optional_int(summary.get("leakage_count"))
    passed = (
        isinstance(response, dict)
        and response.get("ok") is True
        and status == "host_model_reviewed"
        and decision == "pass"
        and (leakage_count or 0) == 0
    )
    if passed:
        gate_status = "passed"
    elif isinstance(response, dict) and response.get("ok") is True:
        gate_status = "hard_fact_leakage_blocked"
    else:
        gate_status = "hard_fact_leakage_review_failed"

    return {
        "ok": passed,
        "schema_version": ROUTE_TRAINING_PAIRS_HARD_FACT_GATE_SCHEMA_VERSION,
        "status": gate_status,
        "host_response_schema_version": (
            response.get("schema_version") if isinstance(response, dict) else None
        ),
        "decision": decision,
        "summary": {
            "sample_count": sample_count,
            "reviewed_count": _optional_int(summary.get("reviewed_count")),
            "forbidden_entity_count": _optional_int(
                summary.get("forbidden_entity_count")
            ),
            "leakage_count": leakage_count,
            "unverifiable_count": _optional_int(summary.get("unverifiable_count")),
        },
        "review_items_safe_refs": _route_training_pairs_review_items_safe_refs(
            result.get("review_items")
        ),
        "error": _route_training_pairs_safe_error(
            response.get("error") if isinstance(response, dict) else None
        ),
        "response_fingerprint": _fingerprint(response),
    }


def _route_training_pairs_review_items_safe_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        matched_entities = item.get("matched_entities")
        refs.append({
            "sample_idx": _optional_int(item.get("sample_idx")),
            "sample_id": _optional_str(item.get("sample_id")),
            "line_no": _optional_int(source.get("line_no")),
            "file_name": _optional_str(source.get("file_name")),
            "sample_fingerprint": _optional_str(source.get("sample_fingerprint")),
            "leakage_detected": bool(item.get("leakage_detected")),
            "severity": _optional_str(item.get("severity")),
            "matched_entity_count": (
                len(matched_entities) if isinstance(matched_entities, list) else 0
            ),
        })
    return refs


def _route_training_pairs_safe_error(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    details = value.get("details")
    return {
        "code": _optional_str(value.get("code")),
        "message": _optional_str(value.get("message")),
        "retryable": bool(value.get("retryable")),
        "details": details if isinstance(details, dict) else {},
    }


def _route_action_learner_dataset_envelope(
    *,
    ok: bool,
    status: str,
    samples: list[dict[str, Any]],
    rpp_run_id: str | None,
    eval_leakage_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "schema_version": ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
        "status": status,
        "name": ROUTE_ACTION_LEARNER_DATASET_NAME,
        "path": None,
        "sample_count": len(samples),
        "rpp_run_id": rpp_run_id,
        "samples": samples,
        "eval_leakage_gate": eval_leakage_gate,
        "audit": {
            "sample_schema_version": ROUTE_ACTION_LEARNER_SAMPLE_SCHEMA_VERSION,
            "source_event_type": ROUTE_ACTION_EVENT_TYPE,
            "runtime_artifact": ROUTE_TRAINING_PAIRS_ARTIFACT_NAME,
            "note": (
                "Learner dataset only; route_training_pairs.json remains "
                "runtime evidence replay/fallback."
            ),
        },
    }


def generate_and_store_route_action_training_events(
    *,
    rpp_output: dict[str, Any],
    eval_cases: list[dict[str, Any]],
    peer_id: str,
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: Any | None = None,
    event_store: EventStore | None = None,
    timestamp_ms: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate route/action pairs, then optionally store them as events.

    Host-model failures are returned as-is in `route_action_response` so callers
    can surface the real model/runtime error instead of silently falling back to
    handwritten route labels.
    """

    from backend.services.host_model_assistant import (
        HOST_MODEL_PROVIDER,
        generate_route_action_pairs,
    )

    route_action_response = generate_route_action_pairs(
        rpp_output,
        eval_cases,
        host_model_id=host_model_id,
        provider=provider or HOST_MODEL_PROVIDER,
        host_model_generate=host_model_generate,
    )
    if route_action_response.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
            "status": "route_action_generation_failed",
            "peer_id": peer_id,
            "route_action_response": route_action_response,
            "storage": None,
            "preview": None,
        }

    try:
        if dry_run:
            events = build_route_action_training_events(
                route_action_response,
                peer_id=peer_id,
                timestamp_ms=timestamp_ms,
            )
            return {
                "ok": True,
                "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
                "status": "previewed",
                "peer_id": peer_id,
                "dry_run": True,
                "route_action_response": route_action_response,
                "storage": None,
                "preview": _route_action_training_events_preview(
                    events,
                    route_action_response=route_action_response,
                ),
            }
        storage = store_route_action_training_events(
            route_action_response,
            peer_id=peer_id,
            event_store=event_store,
            timestamp_ms=timestamp_ms,
        )
    except RouteActionEvalLeakageError as exc:
        return {
            "ok": False,
            "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
            "status": "route_action_eval_leakage_blocked",
            "peer_id": peer_id,
            "route_action_response": route_action_response,
            "storage": None,
            "preview": None,
            "eval_leakage_gate": exc.report,
        }
    return {
        "ok": True,
        "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": peer_id,
        "dry_run": False,
        "route_action_response": route_action_response,
        "storage": storage,
        "preview": None,
    }


def _route_action_training_events_preview(
    events: list[DataEvent],
    *,
    route_action_response: dict[str, Any],
) -> dict[str, Any]:
    event_ids = [event.id for event in events]
    return {
        "event_type": ROUTE_ACTION_EVENT_TYPE,
        "event_count": len(events),
        "would_store": False,
        "event_ids_fingerprint": _fingerprint(event_ids),
        "eval_leakage_gate": evaluate_route_action_eval_leakage(
            route_action_response
        ),
    }


def _route_action_result(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("route_action_response must be an object")
    if response.get("ok") is not True:
        raise ValueError("route_action_response.ok must be true")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("route_action_response.result must be an object")
    return result


def evaluate_route_action_eval_leakage(
    route_action_response: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic exact-replay gate report for route/action output."""

    result = _route_action_result(route_action_response)
    pairs = result.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("result.pairs must be a list")
    return _route_action_eval_leakage_report(
        pairs,
        context="route_action_response",
    )


def _assert_route_action_eval_leakage_gate(
    pairs: list[dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    report = _route_action_eval_leakage_report(pairs, context=context)
    if report["ok"] is not True:
        raise RouteActionEvalLeakageError(report)
    return report


def _route_action_eval_leakage_report(
    pairs: list[dict[str, Any]],
    *,
    context: str,
) -> dict[str, Any]:
    overlaps: list[dict[str, Any]] = []
    unverifiable: list[dict[str, Any]] = []
    warnings: list[str] = []
    checked_pair_count = 0
    checked_prompt_count = 0
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            continue
        checked_pair_count += 1
        source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
        source_prompt = _optional_str(
            source.get("prompt")
            or source.get("eval_prompt")
            or source.get("source_prompt")
        )
        if not source_prompt:
            _append_warning(warnings, "route_action_pair_missing_source_prompt")
            unverifiable.append({
                "pair_idx": index + 1,
                "case_id": _optional_str(pair.get("case_id")),
                "reason": "missing_source_prompt",
                "prompt_fingerprint": (
                    _fingerprint(pair.get("prompt"))
                    if _optional_str(pair.get("prompt"))
                    else None
                ),
            })
            continue
        source_key = _route_training_prompt_key(source_prompt)
        prompts = [("prompt", _optional_str(pair.get("prompt")))]
        prompts.extend(
            (f"prompt_variants[{variant_index}]", variant)
            for variant_index, variant in enumerate(
                _prompt_variants(pair.get("prompt_variants"), "")
            )
        )
        for field, prompt in prompts:
            if not prompt:
                continue
            checked_prompt_count += 1
            if _route_training_prompt_key(prompt) != source_key:
                continue
            overlaps.append({
                "pair_idx": index + 1,
                "case_id": _optional_str(pair.get("case_id")),
                "field": field,
                "source_input_path": _optional_str(source.get("input_path")),
                "prompt_fingerprint": _fingerprint(prompt),
                "source_prompt_fingerprint": _fingerprint(source_prompt),
            })
    return {
        "ok": not overlaps and not unverifiable,
        "schema_version": ROUTE_ACTION_EVAL_LEAKAGE_GATE_SCHEMA_VERSION,
        "status": (
            "passed"
            if not overlaps and not unverifiable
            else "eval_leakage_blocked"
        ),
        "context": context,
        "checked_pair_count": checked_pair_count,
        "checked_prompt_count": checked_prompt_count,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "unverifiable_count": len(unverifiable),
        "unverifiable": unverifiable,
        "warnings": warnings,
    }




def _payload_from_pair(
    pair: dict[str, Any],
    *,
    rpp_run_id: str,
    tool_registry: list[dict[str, Any]] | None = None,
    tool_contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt = _required_str(pair.get("prompt"), "pair.prompt")
    route_intent = _required_str(pair.get("route_intent"), "pair.route_intent")
    answer_discipline = _required_str(
        pair.get("answer_discipline"),
        "pair.answer_discipline",
    )
    rationale = _required_str(pair.get("rationale"), "pair.rationale")
    case_id = _required_str(pair.get("case_id"), "pair.case_id")
    selected_tools = _string_list(pair.get("selected_tools"))
    tool_call_plan = _list_of_dicts(pair.get("tool_call_plan"))
    source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
    tool_expectations = (
        source.get("tool_expectations")
        if isinstance(source.get("tool_expectations"), dict)
        else {}
    )
    validate_route_action_tool_contracts(
        selected_tools=selected_tools,
        tool_call_plan=tool_call_plan,
        case_id=case_id,
        required_tools=tool_expectations.get("required_tools"),
        exact_tools=(
            tool_expectations.get("exact_tools")
            if tool_expectations.get("has_exact_tools") is True
            else None
        ),
        excluded_tools=tool_expectations.get("excluded_tools"),
        tool_registry=tool_registry,
        tool_contracts=tool_contracts,
    )
    return {
        "prompt": prompt,
        "routeIntent": route_intent,
        "promptVariants": _prompt_variants(pair.get("prompt_variants"), prompt),
        "selectedTools": selected_tools,
        "toolCallPlan": tool_call_plan,
        "answerDiscipline": answer_discipline,
        "rationale": rationale,
        "caseId": case_id,
        "source": {
            "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
            "rpp_run_id": rpp_run_id,
            "case_idx": pair.get("case_idx"),
            "confidence": pair.get("confidence"),
            "host_model_source": source,
        },
        "userTags": ["route_action_pair", "host_model_generated"],
    }


def _event_id(*, peer_id: str, rpp_run_id: str, pair: dict[str, Any]) -> str:
    source = pair.get("source") if isinstance(pair.get("source"), dict) else {}
    host_model_source = (
        source.get("host_model_source")
        if isinstance(source.get("host_model_source"), dict)
        else {}
    )
    source_identity = {
        "case_id": pair.get("caseId"),
        "source_prompt": host_model_source.get("prompt") or pair.get("prompt"),
    }
    identity = {
        "peer_id": peer_id,
        "rpp_run_id": rpp_run_id,
        "source": source_identity,
    }
    material = json.dumps(identity, ensure_ascii=False, sort_keys=True)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, material)).upper()


def _json_event_payload(event: DataEvent) -> dict[str, Any] | None:
    try:
        payload = json.loads(event.payload.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _route_training_pair_from_payload(
    payload: dict[str, Any],
    *,
    event: DataEvent,
    event_rpp_run_id: str | None,
) -> dict[str, Any] | None:
    route_intent = _required_str(payload.get("routeIntent"), "route_action_pair.routeIntent")
    expected_intent = _route_intent_to_personal_intent_tag(route_intent)
    if expected_intent is None:
        return None

    prompt = _required_str(payload.get("prompt"), "route_action_pair.prompt")
    confidence = _optional_float(
        (payload.get("source") if isinstance(payload.get("source"), dict) else {}).get("confidence")
    )
    case_id = _optional_str(payload.get("caseId"))
    metadata: dict[str, Any] = {
        "event_id": event.id,
        "route_intent": route_intent,
        "route_action_schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
    }
    if case_id:
        metadata["case_id"] = case_id
    if event_rpp_run_id:
        metadata["rpp_run_id"] = event_rpp_run_id
    answer_discipline = _optional_str(payload.get("answerDiscipline"))
    if answer_discipline:
        metadata["answer_discipline"] = answer_discipline

    return {
        "input": {
            "text": prompt,
            "appContext": {},
            "personaSignals": [],
        },
        "expectedIntentTag": expected_intent,
        "selectedToolNames": _string_list(payload.get("selectedTools")),
        "toolCallPlan": _edge_runtime_tool_call_plan(payload.get("toolCallPlan")),
        "confidence": confidence if confidence is not None else 1.0,
        "source": "runtime_audit",
        "rationale": _optional_str(payload.get("rationale")),
        "createdAtMs": event.timestamp_ms,
        "metadata": metadata,
    }


def _route_training_pairs_from_payload(
    payload: dict[str, Any],
    *,
    event: DataEvent,
    event_rpp_run_id: str | None,
) -> list[dict[str, Any]]:
    primary = _route_training_pair_from_payload(
        payload,
        event=event,
        event_rpp_run_id=event_rpp_run_id,
    )
    if primary is None:
        return []

    pairs = [primary]
    primary_prompt = primary["input"]["text"]
    prompt_variants = _prompt_variants(payload.get("promptVariants"), primary_prompt)
    for index, prompt in enumerate(prompt_variants, start=1):
        variant = json.loads(json.dumps(primary, ensure_ascii=False))
        variant["input"]["text"] = prompt
        variant["metadata"]["prompt_variant_of"] = primary_prompt
        variant["metadata"]["prompt_variant_idx"] = index
        pairs.append(variant)
    return pairs


def _route_action_learner_sample_from_pair(
    *,
    payload: dict[str, Any],
    pair: dict[str, Any],
    event: DataEvent,
    event_rpp_run_id: str | None,
) -> dict[str, Any]:
    metadata = pair.get("metadata") if isinstance(pair.get("metadata"), dict) else {}
    pair_input = pair.get("input") if isinstance(pair.get("input"), dict) else {}
    prompt = _required_str(pair_input.get("text"), "route_action_learner_sample.input.text")
    expected_intent = _required_str(
        pair.get("expectedIntentTag"),
        "route_action_learner_sample.target.route_intent",
    )
    selected_tools = _string_list(pair.get("selectedToolNames"))
    tool_call_plan = _edge_runtime_tool_call_plan(pair.get("toolCallPlan"))
    answer_discipline = _optional_str(metadata.get("answer_discipline"))
    confidence = _optional_float(pair.get("confidence"))
    case_id = _optional_str(metadata.get("case_id") or payload.get("caseId"))

    source: dict[str, Any] = {
        "event_id": event.id,
        "event_type": event.event_type,
        "created_at_ms": event.timestamp_ms,
        "route_intent_raw": _optional_str(metadata.get("route_intent")),
    }
    if case_id:
        source["case_id"] = case_id
    if event_rpp_run_id:
        source["rpp_run_id"] = event_rpp_run_id
    prompt_variant_of = _optional_str(metadata.get("prompt_variant_of"))
    if prompt_variant_of:
        source["prompt_variant_of"] = prompt_variant_of
    if "prompt_variant_idx" in metadata:
        source["prompt_variant_idx"] = metadata["prompt_variant_idx"]

    target = {
        "route_intent": expected_intent,
        "selected_tools": selected_tools,
        "tool_call_plan": tool_call_plan,
    }
    if answer_discipline:
        target["answer_discipline"] = answer_discipline

    identity = {
        "peer_id": event.source_peer_id,
        "event_id": event.id,
        "prompt": prompt,
        "target": target,
    }
    sample: dict[str, Any] = {
        "schema_version": ROUTE_ACTION_LEARNER_SAMPLE_SCHEMA_VERSION,
        "sample_id": _fingerprint(identity),
        "input": {
            "text": prompt,
        },
        "target": target,
        "source": source,
        "weight": confidence if confidence is not None else 1.0,
    }
    rationale = _optional_str(pair.get("rationale"))
    if rationale:
        sample["rationale"] = rationale
    return sample


def _edge_runtime_tool_call_plan(value: Any) -> list[dict[str, Any]]:
    entries = value if isinstance(value, list) else []
    plans: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tool_name = _optional_str(
            entry.get("toolName")
            or entry.get("tool_name")
            or entry.get("tool")
        )
        if not tool_name:
            continue
        arguments = entry.get("arguments")
        if arguments is None:
            arguments = entry.get("args")
        if not isinstance(arguments, dict):
            arguments = {}
        plan: dict[str, Any] = {
            "toolName": tool_name,
            "arguments": {
                str(key): value
                for key, value in arguments.items()
                if isinstance(key, str)
            },
        }
        reason = _optional_str(entry.get("reason") or entry.get("purpose"))
        if reason:
            plan["reason"] = reason
        plans.append(plan)
    return plans


def _eval_gate_pair_from_route_training_pair(
    *,
    payload: dict[str, Any],
    pair: dict[str, Any],
    event_id: str,
) -> dict[str, Any]:
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    host_model_source = (
        source.get("host_model_source")
        if isinstance(source.get("host_model_source"), dict)
        else {}
    )
    metadata = pair.get("metadata") if isinstance(pair.get("metadata"), dict) else {}
    pair_input = pair.get("input") if isinstance(pair.get("input"), dict) else {}
    return {
        "case_id": _optional_str(payload.get("caseId") or metadata.get("case_id")),
        "prompt": _optional_str(pair_input.get("text")),
        "prompt_variants": [],
        "source": {
            "prompt": _optional_str(host_model_source.get("prompt")),
            "input_path": f"event:{event_id}",
        },
    }


def _route_intent_to_personal_intent_tag(route_intent: str) -> str | None:
    normalized = (
        str(route_intent or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if normalized in {"base_chat", "general_knowledge", "general", "plain_chat"}:
        return "base_chat"
    if normalized in {
        "user_profile",
        "persona",
        "profile",
        "personal_profile_with_fact_boundary",
    }:
        return "user_profile"
    if normalized in {"exact_fact", "fact"}:
        return "exact_fact"
    if normalized == "aggregate_fact":
        return "aggregate_fact"
    if normalized in {"app_action", "action"}:
        return "app_action"
    if normalized == "mixed":
        return "mixed"
    return None


def _route_training_prompt_key(prompt: str) -> str:
    return " ".join(str(prompt or "").strip().lower().split()).rstrip("?!。！？.")


def _required_str(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _prompt_variants(value: Any, prompt: str) -> list[str]:
    variants: list[str] = []
    seen = {_route_training_prompt_key(prompt)}
    for item in _string_list(value):
        key = _route_training_prompt_key(item)
        if key in seen:
            continue
        variants.append(item)
        seen.add(key)
    return variants


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _append_warning(warnings: list[str], value: str) -> None:
    if value not in warnings:
        warnings.append(value)
