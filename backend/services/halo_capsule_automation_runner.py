# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Bounded Halo capsule automation runner.

This service is intentionally a behavior-preserving extraction from the mesh
API endpoint. It does not schedule background work. Callers must provide the
peer list, connection predicate, and capsule-push executor explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .device_learning_snapshot_store import DeviceLearningSnapshotError
from .device_lifecycle_automation import build_device_lifecycle_automation_decision
from .device_lifecycle_automation_execution import (
    execute_device_lifecycle_automation_decision,
)
from .halo_capsule_apply_status_store import HaloCapsuleApplyStatusError
from .halo_capsule_automation import (
    HaloCapsuleAutomationPeer,
    build_halo_capsule_automation_preview,
)
from .halo_capsule_automation_run_store import store_halo_capsule_automation_run
from .halo_capsule_package import HaloCapsulePackageError


PushCandidate = Callable[[dict[str, Any], str, str, int], dict[str, Any]]


@dataclass
class HaloCapsuleAutomationRunnerError(ValueError):
    status_code: int
    message: str
    details: dict[str, Any] | None = None

    def to_error(self) -> dict[str, Any]:
        return {
            "code": "halo_capsule_automation_runner_error",
            "message": self.message,
            "retryable": False,
            "details": self.details or {},
        }


def run_halo_capsule_automation_once(
    *,
    dry_run: bool = True,
    peer_ids: Iterable[str] | None = None,
    max_pushes: int = 1,
    chunk_size: int = 1024 * 1024,
    peers: Iterable[HaloCapsuleAutomationPeer],
    is_peer_connected: Callable[[str], bool],
    push_candidate: PushCandidate,
    source: str = "api",
    request_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded automation pass.

    Real pushes require explicit ``peer_ids``. This function preserves the API
    endpoint's fail-closed behavior and writes the same run receipt schema.
    """

    requested_peer_ids = [peer_id.strip() for peer_id in (peer_ids or []) if peer_id.strip()]
    if not dry_run and not requested_peer_ids:
        raise HaloCapsuleAutomationRunnerError(
            status_code=400,
            message="peer_ids is required when dry_run=false",
        )

    selected_peers = list(peers)
    if requested_peer_ids:
        by_id = {peer.peer_id: peer for peer in selected_peers}
        missing = [peer_id for peer_id in requested_peer_ids if peer_id not in by_id]
        if missing:
            raise HaloCapsuleAutomationRunnerError(
                status_code=404,
                message=f"peer {missing[0]} not found",
                details={"peer_id": missing[0]},
            )
        selected_peers = [by_id[peer_id] for peer_id in requested_peer_ids]

    preview = build_halo_capsule_automation_preview(
        selected_peers,
        is_peer_connected=is_peer_connected,
    )

    results: list[dict[str, Any]] = []
    pushed_count = 0
    attempted_count = 0
    for entry in preview["entries"]:
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        action_kind = str(action.get("kind") or "unknown")
        result: dict[str, Any] = {
            "peer_id": entry["peer_id"],
            "display_name": entry.get("display_name"),
            "dry_run": dry_run,
            "would_push": entry.get("would_push") is True,
            "action_kind": action_kind,
            "status": "candidate" if entry.get("would_push") is True else "skipped",
        }
        if dry_run or entry.get("would_push") is not True:
            results.append(result)
            continue
        if pushed_count >= max_pushes:
            result["status"] = "skipped"
            result["error"] = "max_pushes reached"
            results.append(result)
            continue

        attempted_count += 1
        push_response: dict[str, Any] | None = None

        def execute_push(peer_id: str, artifact_id: str) -> dict[str, Any]:
            nonlocal push_response
            _validate_automation_candidate(entry, peer_id, artifact_id)
            push_response = push_candidate(entry, peer_id, artifact_id, chunk_size)
            return push_response

        try:
            decision = build_device_lifecycle_automation_decision(
                entry["peer_id"],
                connected=bool(entry.get("connected")),
            )
            execution = execute_device_lifecycle_automation_decision(
                decision,
                allow_side_effects=True,
                allowed_peer_ids=requested_peer_ids,
                enabled_candidate_kinds=["capsule_push"],
                capsule_push_executor=execute_push,
            )
        except (
            DeviceLearningSnapshotError,
            HaloCapsuleApplyStatusError,
            HaloCapsulePackageError,
        ) as exc:
            result["status"] = "failed"
            result["error"] = _exception_message(exc)
        else:
            if execution["status"] == "executed" and push_response is not None:
                pushed_count += 1
                result["status"] = "pushed"
                result["push"] = push_response
            else:
                result["status"] = "failed"
                result["error"] = _automation_execution_error_message(execution)
        results.append(result)

    response_data = {
        "dry_run": dry_run,
        "attempted_count": attempted_count,
        "pushed_count": pushed_count,
        "preview": preview,
        "results": results,
    }
    request_data = request_payload if request_payload is not None else {
        "dry_run": dry_run,
        "peer_ids": requested_peer_ids,
        "max_pushes": max_pushes,
        "chunk_size": chunk_size,
    }
    receipt = store_halo_capsule_automation_run(
        request=request_data,
        response=response_data,
        source=source,
    )
    response_data["receipt"] = receipt
    return response_data


def _validate_automation_candidate(
    entry: dict[str, Any],
    peer_id: str,
    artifact_id: str,
) -> None:
    entry_peer_id = str(entry.get("peer_id") or "").strip()
    action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
    push_request = (
        action.get("push_request")
        if isinstance(action.get("push_request"), dict)
        else {}
    )
    entry_artifact_id = str(push_request.get("artifact_id") or "").strip()
    if peer_id != entry_peer_id or artifact_id != entry_artifact_id:
        raise HaloCapsulePackageError(
            "automation candidate no longer matches lifecycle decision"
        )


def _automation_execution_error_message(execution: dict[str, Any]) -> str:
    error = str(execution.get("error") or "").strip()
    if error:
        return error
    reason = str(execution.get("reason") or "").strip()
    if reason:
        return reason
    status = str(execution.get("status") or "failed").strip()
    return f"automation execution {status}"


def _exception_message(exc: Exception) -> str:
    to_error = getattr(exc, "to_error", None)
    if callable(to_error):
        error = to_error()
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            if message:
                return message
    return str(exc)
