# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hook up the `persona_source_upload` wire op to the local source store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .mesh_transport import MeshTransportServer, PeerContext
from .persona_source_store import store_persona_source_upload

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer, *, root: Optional[Path] = None) -> None:
    """Register the mTLS Persona source upload handler."""

    def handle_persona_source_upload(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "persona_source_upload requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"persona_source_upload rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )
        if not isinstance(payload, dict):
            raise ValueError("persona_source_upload payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        supplied_peer_id = str(payload.get("peer_id") or "").strip()
        if supplied_peer_id and supplied_peer_id != peer_id:
            raise PermissionError(
                "persona_source_upload peer_id mismatch: "
                f"trusted={peer_id} supplied={supplied_peer_id}"
            )

        upload_payload = dict(payload)
        upload_payload["peer_id"] = peer_id
        receipt = store_persona_source_upload(upload_payload, root=root)

        logger.info(
            "persona_source_upload done peer=%s source_id=%s kind=%s",
            peer_id,
            receipt.get("source_id"),
            receipt.get("source_kind"),
        )
        ctx.trust_store.touch_last_seen(peer_id)

        return {
            "op": "persona_source_upload_ack",
            "payload": receipt,
        }

    server.register_handler("persona_source_upload", handle_persona_source_upload)
    logger.info("persona_source_ingest: registered persona_source_upload handler")
