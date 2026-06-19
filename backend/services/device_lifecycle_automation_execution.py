# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Policy gate for executing device lifecycle automation decisions.

The lifecycle decision builder is audit-only. This module is the next narrow
step: it can execute one explicitly allowed capsule push through an injected
executor, while every other path fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


DEVICE_LIFECYCLE_AUTOMATION_EXECUTION_SCHEMA_VERSION = (
    "edgestudio.device_lifecycle_automation_execution.v1"
)

CapsulePushExecutor = Callable[[str, str], dict[str, Any] | None]


def execute_device_lifecycle_automation_decision(
    decision: dict[str, Any],
    *,
    allow_side_effects: bool = False,
    allowed_peer_ids: Iterable[str] | str | None = None,
    enabled_candidate_kinds: Iterable[str] | str | None = None,
    capsule_push_executor: CapsulePushExecutor | None = None,
) -> dict[str, Any]:
    """Execute one lifecycle decision only when explicit policy allows it."""

    candidate = _dict(decision.get("candidate"))
    target = _dict(candidate.get("target"))
    peer_id = _text(decision.get("peer_id")) or _text(target.get("peer_id"))
    candidate_kind = _text(candidate.get("kind")) or "none"
    artifact_id = _text(target.get("artifact_id"))
    allowed_peers = _string_set(allowed_peer_ids)
    enabled_kinds = _string_set(enabled_candidate_kinds)

    execution = _base_execution(
        decision=decision,
        peer_id=peer_id,
        candidate_kind=candidate_kind,
        target=target,
        allow_side_effects=allow_side_effects,
        allowed_peer_ids=sorted(allowed_peers),
        enabled_candidate_kinds=sorted(enabled_kinds),
        executor_injected=capsule_push_executor is not None,
    )

    if candidate_kind == "none" or candidate.get("policy_status") == "not_applicable":
        return _finish(
            execution,
            status="not_applicable",
            reason="no lifecycle automation candidate is available",
        )

    if not allow_side_effects:
        return _finish(
            execution,
            status="blocked_by_policy",
            reason="side effects are disabled by policy",
        )

    if not peer_id or peer_id not in allowed_peers:
        return _finish(
            execution,
            status="blocked_by_policy",
            reason="peer_id is not explicitly allowed",
        )

    if candidate_kind not in enabled_kinds:
        return _finish(
            execution,
            status="blocked_by_policy",
            reason="candidate kind is not explicitly enabled",
        )

    if candidate_kind != "capsule_push":
        return _finish(
            execution,
            status="unsupported",
            reason=f"{candidate_kind} execution is not implemented in this slice",
        )

    if not artifact_id:
        return _finish(
            execution,
            status="invalid_candidate",
            reason="capsule_push candidate is missing artifact_id",
        )

    if capsule_push_executor is None:
        return _finish(
            execution,
            status="executor_unavailable",
            reason="capsule_push executor is required",
        )

    execution["attempted"] = True
    try:
        result = capsule_push_executor(peer_id, artifact_id) or {}
    except Exception as exc:  # pragma: no cover - exact executor types are injected.
        return _finish(
            execution,
            status="failed",
            reason="capsule_push executor failed",
            error=str(exc),
        )

    execution["result"] = result
    execution["effects"]["side_effects_executed"] = True
    execution["effects"]["capsule_push_executed"] = True
    return _finish(
        execution,
        status="executed",
        reason="capsule_push executed by explicit policy",
    )


def _base_execution(
    *,
    decision: dict[str, Any],
    peer_id: str | None,
    candidate_kind: str,
    target: dict[str, Any],
    allow_side_effects: bool,
    allowed_peer_ids: list[str],
    enabled_candidate_kinds: list[str],
    executor_injected: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        "schema_version": DEVICE_LIFECYCLE_AUTOMATION_EXECUTION_SCHEMA_VERSION,
        "decision_key": _text(decision.get("decision_key")),
        "peer_id": peer_id,
        "candidate_kind": candidate_kind,
        "target": target,
        "attempted": False,
        "side_effects_executed": False,
        "effects": {
            "side_effects_executed": False,
            "capsule_push_executed": False,
            "neural_imprint_regen_triggered": False,
            "model_push_executed": False,
            "background_scheduler_triggered": False,
        },
        "policy": {
            "allow_side_effects": bool(allow_side_effects),
            "allowed_peer_ids": allowed_peer_ids,
            "enabled_candidate_kinds": enabled_candidate_kinds,
        },
        "audit": {
            "policy_gated": True,
            "explicit_side_effects_required": True,
            "explicit_peer_required": True,
            "explicit_candidate_kind_required": True,
            "executor_injected": executor_injected,
            "background_scheduler": False,
            "broad_fanout": False,
            "automatic_default_execution": False,
        },
        "result": None,
        "error": None,
    }


def _finish(
    execution: dict[str, Any],
    *,
    status: str,
    reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    execution["status"] = status
    execution["reason"] = reason
    execution["error"] = error
    execution["side_effects_executed"] = bool(
        execution["effects"]["side_effects_executed"]
    )
    if status == "failed":
        execution["ok"] = False
    return execution


def _string_set(value: Iterable[str] | str | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        return {text} if text else set()
    result: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if text:
            result.add(text)
    return result


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
