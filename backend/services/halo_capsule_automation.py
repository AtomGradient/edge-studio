# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Dry-run automation preview for Halo capsule distribution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from .halo_capsule_coordinator import build_halo_capsule_coordinator_plan


HALO_CAPSULE_AUTOMATION_PREVIEW_SCHEMA_VERSION = (
    "edgestudio.halo_capsule_automation_preview.v1"
)


class HaloCapsuleAutomationPeer(Protocol):
    peer_id: str
    display_name: str
    revoked: bool


def build_halo_capsule_automation_preview(
    peers: Iterable[HaloCapsuleAutomationPeer],
    *,
    is_peer_connected: Callable[[str], bool],
) -> dict[str, Any]:
    """Return a dry-run preview of host-driven Halo capsule automation.

    This function deliberately has no side effects. It does not push capsules,
    retry failed transfers, restore artifacts, or write automation state.
    """

    entries: list[dict[str, Any]] = []
    candidate_count = 0
    skipped_revoked = 0
    for peer in peers:
        if peer.revoked:
            skipped_revoked += 1
            continue
        connected = bool(is_peer_connected(peer.peer_id))
        plan = build_halo_capsule_coordinator_plan(peer.peer_id, connected=connected)
        action = plan.get("action") if isinstance(plan.get("action"), dict) else {}
        would_push = action.get("can_push") is True
        if would_push:
            candidate_count += 1
        entries.append({
            "peer_id": peer.peer_id,
            "display_name": peer.display_name,
            "connected": connected,
            "would_push": would_push,
            "action": action,
            "plan": plan,
        })

    return {
        "ok": True,
        "schema_version": HALO_CAPSULE_AUTOMATION_PREVIEW_SCHEMA_VERSION,
        "dry_run": True,
        "peer_count": len(entries),
        "candidate_count": candidate_count,
        "skipped_revoked_count": skipped_revoked,
        "entries": entries,
    }
