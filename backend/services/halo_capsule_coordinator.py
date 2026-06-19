# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Plan-only coordinator for Neural Imprint Halo capsule distribution.

This module is intentionally read-side only. It combines the latest device
snapshot, host-side artifact registry, and device apply status into an
explainable next action. It never pushes capsules, retries transfers, restores
artifacts, or mutates device state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .device_learning_snapshot_store import latest_device_learning_snapshot
from .halo_capsule_apply_status_store import latest_halo_capsule_apply_status
from .halo_capsule_transfer_ack_store import latest_halo_capsule_transfer_ack
from .neural_imprint_artifact_registry import list_neural_imprint_artifacts


HALO_CAPSULE_COORDINATOR_PLAN_SCHEMA_VERSION = (
    "edgestudio.halo_capsule_coordinator_plan.v1"
)


def build_halo_capsule_coordinator_plan(
    peer_id: str,
    *,
    connected: bool,
    snapshot_root: Path | None = None,
    apply_status_root: Path | None = None,
    transfer_ack_root: Path | None = None,
) -> dict[str, Any]:
    """Return an explainable plan for one peer without side effects."""

    snapshot_record = latest_device_learning_snapshot(peer_id, root=snapshot_root)
    if snapshot_record is None:
        return _plan(
            peer_id,
            connected=connected,
            action=_action(
                "wait_for_snapshot",
                can_push=False,
                label="Wait for device snapshot",
                reasons=["no latest device learning snapshot is available"],
            ),
        )

    snapshot = _dict(snapshot_record.get("snapshot"))
    receipt = _dict(snapshot_record.get("receipt"))
    lifecycle = _dict(receipt.get("lifecycle"))
    model = _dict(snapshot.get("model"))
    learning = _dict(snapshot.get("learning"))
    corrections = _dict(snapshot.get("corrections"))

    selected_model_id = _selected_model_id(model)
    artifacts = [
        artifact
        for artifact in list_neural_imprint_artifacts().get("artifacts", [])
        if artifact.get("valid") is True
    ]
    matched_artifact = _select_artifact_for_model(artifacts, selected_model_id)
    last_apply_record = latest_halo_capsule_apply_status(peer_id, root=apply_status_root)
    last_apply_receipt = (
        _dict(last_apply_record.get("receipt"))
        if isinstance(last_apply_record, dict)
        else None
    )
    last_ack_record = latest_halo_capsule_transfer_ack(peer_id, root=transfer_ack_root)
    last_ack_receipt = (
        _dict(last_ack_record.get("receipt"))
        if isinstance(last_ack_record, dict)
        else None
    )

    context = {
        "snapshot_sha256": receipt.get("snapshot_sha256"),
        "lifecycle": lifecycle or None,
        "selected_model_id": selected_model_id,
        "load_state": str(model.get("load_state") or "").strip() or None,
        "data_readiness": _dict(snapshot.get("data")).get("readiness"),
        "learning": _learning_summary(learning),
        "artifact_count": len(artifacts),
        "matched_artifact": _artifact_summary(matched_artifact),
        "last_apply_status": last_apply_receipt,
        "last_transfer_ack": last_ack_receipt,
    }

    action = _decide_action(
        peer_id=peer_id,
        connected=connected,
        lifecycle=lifecycle,
        learning=learning,
        corrections=corrections,
        selected_model_id=selected_model_id,
        matched_artifact=matched_artifact,
        last_apply_receipt=last_apply_receipt,
    )
    return _plan(peer_id, connected=connected, action=action, **context)


def _decide_action(
    *,
    peer_id: str,
    connected: bool,
    lifecycle: dict[str, Any],
    learning: dict[str, Any],
    corrections: dict[str, Any],
    selected_model_id: str | None,
    matched_artifact: dict[str, Any] | None,
    last_apply_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    phase = str(lifecycle.get("phase") or "").strip()
    phase_label = str(lifecycle.get("phase_label") or "").strip()
    neural_imprint = _learning_neural_imprint(learning)
    neural_imprint_status = _status(neural_imprint)
    needs_regen = corrections.get("needs_regen") is True or neural_imprint_status in {
        "stale",
        "incompatible",
    }

    if phase == "S0":
        return _action(
            "install_model_first",
            can_push=False,
            label="Install model first",
            reasons=list(lifecycle.get("reasons") or ["device reports no installed model"]),
        )

    if phase == "S1" and phase_label == "model_present_not_loaded":
        return _action(
            "load_model_first",
            can_push=False,
            label="Load model first",
            reasons=list(lifecycle.get("reasons") or ["model is present but not loaded"]),
        )

    if phase == "S1" and phase_label == "model_ready_no_data":
        return _action(
            "wait_for_data",
            can_push=False,
            label="Wait for enough user data",
            reasons=list(lifecycle.get("reasons") or ["device data is below the learning threshold"]),
        )

    if needs_regen:
        return _action(
            "regenerate_neural_imprint_first",
            can_push=False,
            label="Regenerate Neural Imprint first",
                reasons=list(
                    lifecycle.get("reasons")
                    or ["corrections or artifact drift require regeneration"]
                ),
        )

    if neural_imprint_status == "active" and not needs_regen:
        active_artifact_sha = _optional_text(neural_imprint.get("artifact_sha256"))
        matched_neural_imprint_sha = (
            _artifact_neural_imprint_sha256(matched_artifact)
            if matched_artifact
            else None
        )
        if (
            active_artifact_sha
            and matched_neural_imprint_sha
            and active_artifact_sha != matched_neural_imprint_sha
        ):
            if not connected:
                return _action(
                    "connect_device_first",
                    can_push=False,
                    label="Reconnect device before push",
                    reasons=[
                        "host has a newer matching Neural Imprint artifact but peer is not connected"
                    ],
                )
            return _action(
                "newer_artifact_available",
                can_push=True,
                label="Newer Neural Imprint artifact available",
                reasons=[
                    "device active Neural Imprint sha256 differs from host registry latest matching artifact"
                ],
                push_request={
                    "peer_id": peer_id,
                    "artifact_id": matched_artifact.get("artifact_id")
                    if matched_artifact
                    else None,
                },
            )
        return _action(
            "neural_imprint_active_no_push_needed",
            can_push=False,
            label="Neural Imprint is already active",
            reasons=["device reports active Neural Imprint"],
        )

    if neural_imprint_status == "present_inactive" and not needs_regen:
        artifact_sha = _optional_text(neural_imprint.get("artifact_sha256"))
        if (
            artifact_sha
            and matched_artifact
            and artifact_sha == _artifact_neural_imprint_sha256(matched_artifact)
        ):
            return _action(
                "activate_neural_imprint_first",
                can_push=False,
                label="Activate existing Neural Imprint on device",
                reasons=["device already has the matching Neural Imprint artifact but it is inactive"],
            )

    if not selected_model_id:
        return _action(
            "select_model_first",
            can_push=False,
            label="Select model first",
            reasons=["snapshot does not report a selected or loaded model id"],
        )

    if matched_artifact is None:
        return _action(
            "no_matching_artifact",
            can_push=False,
            label="No matching Neural Imprint artifact",
            reasons=[f"host registry has no valid artifact for {selected_model_id}"],
        )

    artifact_sha = str(matched_artifact.get("artifact_sha256") or "")
    if last_apply_receipt and last_apply_receipt.get("artifact_sha256") == artifact_sha:
        status = str(last_apply_receipt.get("status") or "")
        if status == "applied":
            return _action(
                "already_applied",
                can_push=False,
                label="Latest matching capsule is already applied",
                reasons=["device reported applied for the matching artifact sha256"],
            )
        if status == "received":
            return _action(
                "apply_pending",
                can_push=False,
                label="Device received capsule; waiting for apply",
                reasons=["device has received the matching artifact but has not reported applied yet"],
            )
        if status == "failed":
            if not connected:
                return _action(
                    "connect_device_first",
                    can_push=False,
                    label="Reconnect device before retry",
                    reasons=["last apply failed, but peer is not connected"],
                )
            return _action(
                "retry_manual_push_available",
                can_push=True,
                label="Retry manual push",
                reasons=[
                    last_apply_receipt.get("error_code")
                    or "last apply for matching artifact failed"
                ],
                push_request={
                    "peer_id": peer_id,
                    "artifact_id": matched_artifact.get("artifact_id"),
                },
            )

    if not connected:
        return _action(
            "connect_device_first",
            can_push=False,
            label="Reconnect device before push",
            reasons=["peer is trusted but not currently connected"],
        )

    return _action(
        "manual_push_available",
        can_push=True,
        label="Manual Neural Imprint push is available",
        reasons=["device is connected and host has a valid artifact for the selected model"],
        push_request={
            "peer_id": peer_id,
            "artifact_id": matched_artifact.get("artifact_id"),
        },
    )


def _plan(
    peer_id: str,
    *,
    connected: bool,
    action: dict[str, Any],
    **context: Any,
) -> dict[str, Any]:
    return {
        "ok": True,
        "schema_version": HALO_CAPSULE_COORDINATOR_PLAN_SCHEMA_VERSION,
        "peer_id": peer_id,
        "connected": connected,
        "action": action,
        **context,
    }


def _action(
    kind: str,
    *,
    can_push: bool,
    label: str,
    reasons: list[str],
    push_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "kind": kind,
        "can_push": can_push,
        "requires_user_confirmation": can_push,
        "label": label,
        "reasons": [str(reason) for reason in reasons if str(reason).strip()],
    }
    if push_request:
        action["push_request"] = {
            key: value for key, value in push_request.items() if value
        }
    return action


def _selected_model_id(model: dict[str, Any]) -> str | None:
    for key in ("loaded_model_id", "selected_model_id"):
        value = _optional_text(model.get(key))
        if value:
            return value
    installed = model.get("installed_models")
    if isinstance(installed, list):
        for item in installed:
            if isinstance(item, dict):
                value = _optional_text(item.get("model_id"))
                if value:
                    return value
    return None


def _select_artifact_for_model(
    artifacts: list[dict[str, Any]],
    selected_model_id: str | None,
) -> dict[str, Any] | None:
    if not selected_model_id:
        return None
    matches = [
        artifact
        for artifact in artifacts
        if str(artifact.get("base_model_id") or "") == selected_model_id
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda artifact: str(artifact.get("created_at") or ""),
        reverse=True,
    )[0]


def _artifact_summary(artifact: dict[str, Any] | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    keys = (
        "artifact_id",
        "capsule_id",
        "base_model_id",
        "model_family",
        "artifact_sha256",
        "neural_imprint_sha256",
        "total_bytes",
        "prefix_token_count",
        "tool_schema_sha256",
        "profile_body_sha256",
        "created_at",
    )
    return {key: artifact.get(key) for key in keys if artifact.get(key) is not None}


def _artifact_neural_imprint_sha256(artifact: dict[str, Any] | None) -> str | None:
    if not artifact:
        return None
    return _optional_text(
        artifact.get("neural_imprint_sha256")
        or artifact.get("artifact_sha256")
    )


def _learning_summary(learning: dict[str, Any]) -> dict[str, Any]:
    return {
        "tools_only": _artifact_summary_from_learning(_dict(learning.get("tools_only"))),
        "rpp": _artifact_summary_from_learning(_dict(learning.get("rpp"))),
        "neural_imprint": _artifact_summary_from_learning(_learning_neural_imprint(learning)),
        "target_layer": learning.get("target_layer"),
        "a_library_id": learning.get("a_library_id"),
        "a_library_sha256": learning.get("a_library_sha256"),
        "tool_schema_sha256": learning.get("tool_schema_sha256"),
    }


def _artifact_summary_from_learning(value: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "artifact_sha256",
        "prefix_token_count",
        "run_id",
        "error_code",
    )
    return {key: value.get(key) for key in keys if value.get(key) is not None}


def _learning_neural_imprint(learning: dict[str, Any]) -> dict[str, Any]:
    return _dict(learning.get("neural_imprint"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _status(value: dict[str, Any]) -> str:
    return str(value.get("status") or "unknown").strip().lower()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
