# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Audit-only coordinator for device lifecycle automation decisions.

This module turns the latest S0-S5 lifecycle snapshot plus the existing Halo
capsule plan into an explainable side-effect candidate. It deliberately does
not push capsules, run Neural Imprint regeneration, push models, or schedule
background work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .device_learning_snapshot_store import latest_device_learning_snapshot
from .device_lifecycle_automation_store import (
    store_device_lifecycle_automation_decision,
)
from .halo_capsule_coordinator import build_halo_capsule_coordinator_plan


DEVICE_LIFECYCLE_AUTOMATION_DECISION_SCHEMA_VERSION = (
    "edgestudio.device_lifecycle_automation_decision.v1"
)


def build_device_lifecycle_automation_decision(
    peer_id: str,
    *,
    connected: bool,
    snapshot_root: Path | None = None,
    apply_status_root: Path | None = None,
    transfer_ack_root: Path | None = None,
) -> dict[str, Any]:
    """Return an audit-only lifecycle automation decision for one peer."""

    clean_peer_id = str(peer_id or "").strip()
    snapshot_record = latest_device_learning_snapshot(clean_peer_id, root=snapshot_root)
    snapshot_receipt = (
        _dict(snapshot_record.get("receipt"))
        if isinstance(snapshot_record, dict)
        else {}
    )
    lifecycle = _dict(snapshot_receipt.get("lifecycle"))
    plan = build_halo_capsule_coordinator_plan(
        clean_peer_id,
        connected=connected,
        snapshot_root=snapshot_root,
        apply_status_root=apply_status_root,
        transfer_ack_root=transfer_ack_root,
    )
    action = _dict(plan.get("action"))
    candidate = _candidate_from_plan(
        lifecycle=lifecycle,
        action=action,
        selected_model_id=_optional_text(plan.get("selected_model_id")),
    )
    effects = {
        "side_effects_executed": False,
        "capsule_push_executed": False,
        "neural_imprint_regen_triggered": False,
        "model_push_executed": False,
        "background_scheduler_triggered": False,
    }
    decision_base = {
        "peer_id": clean_peer_id,
        "snapshot_sha256": _optional_text(snapshot_receipt.get("snapshot_sha256")),
        "connected": bool(connected),
        "lifecycle": lifecycle or None,
        "plan_action": action,
        "candidate": candidate,
        "effects": effects,
        "audit": {
            "policy_gated": True,
            "audit_only": True,
            "automatic_push": False,
            "automatic_regen": False,
            "background_scheduler": False,
            "uses_latest_snapshot": snapshot_record is not None,
            "uses_halo_plan": True,
        },
    }
    decision = {
        "ok": True,
        "schema_version": DEVICE_LIFECYCLE_AUTOMATION_DECISION_SCHEMA_VERSION,
        **decision_base,
        "decision_key": _decision_key(decision_base),
    }
    return decision


def build_and_store_device_lifecycle_automation_decision(
    peer_id: str,
    *,
    connected: bool,
    source: str = "api",
    root: Path | None = None,
    snapshot_root: Path | None = None,
    apply_status_root: Path | None = None,
    transfer_ack_root: Path | None = None,
) -> dict[str, Any]:
    """Build and persist an audit-only lifecycle automation decision."""

    decision = build_device_lifecycle_automation_decision(
        peer_id,
        connected=connected,
        snapshot_root=snapshot_root,
        apply_status_root=apply_status_root,
        transfer_ack_root=transfer_ack_root,
    )
    receipt = store_device_lifecycle_automation_decision(
        decision,
        source=source,
        root=root,
    )
    return {
        "ok": True,
        "receipt": receipt,
        "decision": decision,
    }


def _candidate_from_plan(
    *,
    lifecycle: dict[str, Any],
    action: dict[str, Any],
    selected_model_id: str | None,
) -> dict[str, Any]:
    action_kind = str(action.get("kind") or "").strip()
    push_request = _dict(action.get("push_request"))
    artifact_id = _optional_text(push_request.get("artifact_id"))

    if action.get("can_push") is True and artifact_id:
        return {
            "kind": "capsule_push",
            "eligible": True,
            "policy_status": "blocked_by_policy",
            "blocked_reason": "automatic capsule push is disabled in this audit-only slice",
            "target": {
                "peer_id": _optional_text(push_request.get("peer_id")),
                "artifact_id": artifact_id,
            },
        }

    if action_kind == "regenerate_neural_imprint_first":
        return {
            "kind": "neural_imprint_regen",
            "eligible": True,
            "policy_status": "blocked_by_policy",
            "blocked_reason": "automatic Neural Imprint regeneration is disabled in this audit-only slice",
            "target": {
                "selected_model_id": selected_model_id,
            },
        }

    if action_kind == "install_model_first" or lifecycle.get("phase") == "S0":
        return {
            "kind": "model_push",
            "eligible": False,
            "policy_status": "unsupported",
            "blocked_reason": "automatic model distribution is not implemented in this slice",
            "target": {
                "selected_model_id": selected_model_id,
            },
        }

    return {
        "kind": "none",
        "eligible": False,
        "policy_status": "not_applicable",
        "blocked_reason": None,
        "target": {},
    }


def _decision_key(value: dict[str, Any]) -> str:
    action = _dict(value.get("plan_action"))
    candidate = _dict(value.get("candidate"))
    target = _dict(candidate.get("target"))
    lifecycle = _dict(value.get("lifecycle"))
    material = {
        "peer_id": value.get("peer_id"),
        "snapshot_sha256": value.get("snapshot_sha256"),
        "lifecycle_phase": lifecycle.get("phase"),
        "action_kind": action.get("kind"),
        "candidate_kind": candidate.get("kind"),
        "artifact_id": target.get("artifact_id"),
        "selected_model_id": target.get("selected_model_id"),
    }
    payload = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
