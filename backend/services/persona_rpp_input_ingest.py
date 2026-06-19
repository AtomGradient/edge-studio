# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hook up canonical Persona/RPP input uploads to the local contract store."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .mesh_transport import MeshTransportServer, PeerContext
from .persona_rpp_input_contract import store_persona_rpp_input_contract

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer, *, root: Optional[Path] = None) -> None:
    """Register the mTLS Persona/RPP input upload handler."""

    def handle_persona_rpp_input_upload(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "persona_rpp_input_upload requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"persona_rpp_input_upload rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )
        if not isinstance(payload, dict):
            raise ValueError("persona_rpp_input_upload payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        supplied_peer_id = str(payload.get("peer_id") or "").strip()
        if supplied_peer_id and supplied_peer_id != peer_id:
            raise PermissionError(
                "persona_rpp_input_upload peer_id mismatch: "
                f"trusted={peer_id} supplied={supplied_peer_id}"
            )

        upload_payload = dict(payload)
        upload_payload["peer_id"] = peer_id
        receipt = store_persona_rpp_input_contract(upload_payload, root=root)

        logger.info(
            "persona_rpp_input_upload done peer=%s input_id=%s kind=%s records=%s",
            peer_id,
            receipt.get("input_id"),
            receipt.get("source_kind"),
            receipt.get("record_count"),
        )
        ctx.trust_store.touch_last_seen(peer_id)

        return {
            "op": "persona_rpp_input_upload_ack",
            "payload": receipt,
        }

    server.register_handler("persona_rpp_input_upload", handle_persona_rpp_input_upload)
    logger.info("persona_rpp_input_ingest: registered persona_rpp_input_upload handler")
