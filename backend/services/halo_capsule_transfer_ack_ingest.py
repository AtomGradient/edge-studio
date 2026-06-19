# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Register EdgeMesh Halo capsule transfer ACK handlers.

When a device receives an offer, it sends back an offer_ack indicating
accept/reject. This module handles those ACKs: logs, broadcasts to
WebSocket clients for live UI updates, and responds to the device.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .mesh_events import get_default_bus
from .mesh_transport import MeshTransportServer, PeerContext
from .halo_capsule_transfer_ack_store import store_halo_capsule_transfer_ack

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer, *, root: Path | None = None) -> None:
    """Register handlers for device-side Halo capsule transfer ACKs."""

    def handle_ack(payload: dict, ctx: PeerContext, *, ack_kind: str) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError("halo capsule ACK requires a trusted peer")
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"halo capsule ACK rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )
        if not isinstance(payload, dict):
            raise ValueError("halo capsule ACK payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        transfer_id = str(payload.get("transfer_id") or "")
        accepted = payload.get("accepted")
        reason = str(payload.get("reason") or "")
        canonical_sha256 = str(payload.get("canonical_sha256") or "")
        ctx.trust_store.touch_last_seen(peer_id)
        receipt = store_halo_capsule_transfer_ack(
            peer_id,
            payload,
            ack_kind=ack_kind,
            source="mesh",
            root=root,
        )
        logger.info(
            "halo_capsule_transfer_ack peer=%s transfer=%s accepted=%s reason=%s",
            peer_id,
            transfer_id,
            accepted,
            reason,
        )

        # Broadcast to WebSocket clients for live UI feedback
        get_default_bus().broadcast({
            "type": "halo_capsule_offer_ack",
            "peer_id": peer_id,
            "transfer_id": transfer_id,
            "accepted": bool(accepted) if accepted is not None else None,
            "reason": reason or None,
            "canonical_sha256": canonical_sha256 or None,
            "ack_kind": ack_kind,
            "ack_sha256": receipt.get("ack_sha256"),
        })

        return {
            "op": "halo_capsule_ack_received",
            "payload": {
                "transfer_id": transfer_id,
                "accepted": bool(accepted) if accepted is not None else None,
            },
        }

    server.register_handler(
        "halo_capsule_offer_ack",
        lambda payload, ctx: handle_ack(payload, ctx, ack_kind="offer_ack"),
    )
    server.register_handler(
        "halo_capsule_complete_ack",
        lambda payload, ctx: handle_ack(payload, ctx, ack_kind="complete_ack"),
    )
    logger.info("halo_capsule_transfer_ack_ingest: registered ACK handlers")
