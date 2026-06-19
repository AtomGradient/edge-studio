# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Register EdgeMesh peer trust unlink ingestion."""

from __future__ import annotations

import logging
from typing import Any

from backend.services.mesh_events import get_default_bus

from .mesh_transport import MeshTransportServer, PeerContext

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer) -> None:
    """Register the authenticated peer trust delete handler."""

    def handle_peer_trust_deleted(payload: dict[str, Any], ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "peer_trust_deleted requires a trusted peer (complete pairing first)"
            )
        if not isinstance(payload, dict):
            raise ValueError("peer_trust_deleted payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        claimed_peer_id = str(payload.get("peer_id", "")).strip()
        if claimed_peer_id and claimed_peer_id != peer_id:
            raise PermissionError(
                f"peer_trust_deleted peer_id mismatch: claim={claimed_peer_id} tls={peer_id}"
            )

        was_known = ctx.trust_store.lookup(peer_id) is not None
        ctx.trust_store.delete(peer_id)
        get_default_bus().broadcast({
            "type": "peer_deleted",
            "peer_id": peer_id,
            "source": "peer_trust_deleted",
        })

        logger.info(
            "peer_trust_deleted done peer=%s was_known=%s reason=%s",
            peer_id,
            was_known,
            payload.get("reason"),
        )
        return {
            "op": "peer_trust_deleted_ack",
            "payload": {
                "ok": True,
                "peer_id": peer_id,
                "was_known": was_known,
            },
        }

    server.register_handler("peer_trust_deleted", handle_peer_trust_deleted)
    logger.info("peer_trust_ingest: registered peer_trust_deleted handler")
