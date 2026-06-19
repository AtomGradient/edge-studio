# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Hook up the `rpp_artifact_upload` wire op to the local RPP artifact store.

The HTTP endpoint remains useful for local tooling, but device-originated
uploads over EdgeMesh must use the already-authenticated mTLS connection. The
server binds `peer_id` to the trusted certificate record instead of trusting
client-supplied identity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .mesh_transport import MeshTransportServer, PeerContext
from .rpp_artifact_store import store_rpp_artifact_upload

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer, *, root: Optional[Path] = None) -> None:
    """Register the mTLS RPP artifact upload handler."""

    def handle_rpp_artifact_upload(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "rpp_artifact_upload requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"rpp_artifact_upload rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )
        if not isinstance(payload, dict):
            raise ValueError("rpp_artifact_upload payload must be an object")

        peer_id = ctx.trusted_peer.peer_id
        supplied_peer_id = str(payload.get("peer_id") or "").strip()
        if supplied_peer_id and supplied_peer_id != peer_id:
            raise PermissionError(
                "rpp_artifact_upload peer_id mismatch: "
                f"trusted={peer_id} supplied={supplied_peer_id}"
            )

        upload_payload = dict(payload)
        upload_payload["peer_id"] = peer_id
        receipt = store_rpp_artifact_upload(upload_payload, root=root)

        logger.info(
            "rpp_artifact_upload done peer=%s rpp_run_id=%s artifacts=%d",
            peer_id,
            receipt.get("rpp_run_id"),
            len(receipt.get("artifacts") or []),
        )
        ctx.trust_store.touch_last_seen(peer_id)

        return {
            "op": "rpp_artifact_upload_ack",
            "payload": receipt,
        }

    server.register_handler("rpp_artifact_upload", handle_rpp_artifact_upload)
    logger.info("rpp_artifact_ingest: registered rpp_artifact_upload handler")
