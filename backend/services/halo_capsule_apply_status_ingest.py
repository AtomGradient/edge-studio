# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Register EdgeMesh `halo_capsule_apply_status` ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from backend.services.mesh_events import get_default_bus

from .halo_capsule_apply_status_store import store_halo_capsule_apply_status
from .mesh_transport import MeshTransportServer, PeerContext

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer, *, root: Optional[Path] = None) -> None:
    """Register the authenticated Halo capsule apply-status handler."""

    def handle_halo_capsule_apply_status(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "halo_capsule_apply_status requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"halo_capsule_apply_status rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )
        if not isinstance(payload, dict):
            raise ValueError("halo_capsule_apply_status payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        receipt = store_halo_capsule_apply_status(
            peer_id,
            payload,
            source="mesh",
            root=root,
        )
        ctx.trust_store.touch_last_seen(peer_id)
        get_default_bus().broadcast({
            "type": "halo_capsule_apply_status",
            "peer_id": peer_id,
            "transfer_id": receipt["transfer_id"],
            "capsule_id": receipt["capsule_id"],
            "status": receipt["status"],
            "artifact_sha256": receipt.get("artifact_sha256"),
            "canonical_sha256": receipt.get("canonical_sha256"),
            "runtime_version": receipt.get("runtime_version"),
            "prefix_token_count": receipt.get("prefix_token_count"),
            "error_code": receipt.get("error_code"),
        })

        logger.info(
            "halo_capsule_apply_status done peer=%s transfer=%s capsule=%s status=%s",
            peer_id,
            receipt["transfer_id"],
            receipt["capsule_id"],
            receipt["status"],
        )
        return {
            "op": "halo_capsule_apply_status_ack",
            "payload": receipt,
        }

    server.register_handler("halo_capsule_apply_status", handle_halo_capsule_apply_status)
    logger.info("halo_capsule_apply_status_ingest: registered handler")
