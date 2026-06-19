# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Register EdgeMesh `device_state_snapshot` ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.services.device_lifecycle_automation import (
    build_and_store_device_lifecycle_automation_decision,
)
from backend.services.mesh_events import get_default_bus

from .device_learning_snapshot_store import store_device_learning_snapshot
from .mesh_transport import MeshTransportServer, PeerContext

logger = logging.getLogger(__name__)


def register(
    server: MeshTransportServer,
    *,
    root: Optional[Path] = None,
    automation_root: Optional[Path] = None,
) -> None:
    """Register the authenticated device learning snapshot handler."""

    def handle_device_state_snapshot(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "device_state_snapshot requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"device_state_snapshot rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )
        if not isinstance(payload, dict):
            raise ValueError("device_state_snapshot payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        receipt = store_device_learning_snapshot(
            peer_id,
            payload,
            source="mesh",
            root=root,
        )
        ctx.trust_store.touch_last_seen(peer_id)
        get_default_bus().broadcast({
            "type": "device_state_snapshot",
            "peer_id": peer_id,
            "phase": receipt["lifecycle"]["phase"],
            "phase_label": receipt["lifecycle"]["phase_label"],
            "snapshot_sha256": receipt["snapshot_sha256"],
        })
        automation = _record_lifecycle_automation_decision(
            server,
            peer_id,
            snapshot_root=root,
            root=automation_root or _automation_root_for_snapshot_root(root),
        )
        if automation is not None:
            get_default_bus().broadcast({
                "type": "device_lifecycle_automation_decision",
                "peer_id": peer_id,
                "decision_id": automation["receipt"]["decision_id"],
                "decision_key": automation["receipt"]["decision_key"],
                "candidate_kind": automation["receipt"]["candidate_kind"],
                "policy_status": automation["receipt"]["policy_status"],
                "side_effects_executed": automation["receipt"]["side_effects_executed"],
            })

        logger.info(
            "device_state_snapshot done peer=%s phase=%s sha=%s",
            peer_id,
            receipt["lifecycle"]["phase_label"],
            receipt["snapshot_sha256"][:12],
        )
        return {
            "op": "device_state_snapshot_ack",
            "payload": receipt,
        }

    server.register_handler("device_state_snapshot", handle_device_state_snapshot)
    logger.info("device_snapshot_ingest: registered device_state_snapshot handler")


def _record_lifecycle_automation_decision(
    server: MeshTransportServer,
    peer_id: str,
    *,
    snapshot_root: Optional[Path],
    root: Optional[Path],
) -> dict | None:
    try:
        is_connected = getattr(server, "is_peer_connected", None)
        connected = bool(is_connected(peer_id)) if callable(is_connected) else False
        return build_and_store_device_lifecycle_automation_decision(
            peer_id,
            connected=connected,
            source="mesh",
            root=root,
            snapshot_root=snapshot_root,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "device_lifecycle_automation audit skipped peer=%s: %s",
            peer_id,
            exc,
        )
        return None


def _automation_root_for_snapshot_root(root: Optional[Path]) -> Optional[Path]:
    if root is None:
        return None
    return Path(root).expanduser().resolve().parent / "lifecycle-automation"
