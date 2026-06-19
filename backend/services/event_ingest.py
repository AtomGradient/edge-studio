# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import logging
from typing import Optional

from backend.stores.event_store import EventStore, decode_wire_event, get_default_store

from .mesh_transport import MeshTransportServer, PeerContext

logger = logging.getLogger(__name__)


def register(server: MeshTransportServer, store: Optional[EventStore] = None) -> None:
    event_store = store or get_default_store()

    def handle_event_upload(payload: dict, ctx: PeerContext) -> dict:
        if ctx.trusted_peer is None:
            raise PermissionError(
                "event_upload requires a trusted peer (complete pairing first)"
            )
        if ctx.trusted_peer.revoked:
            raise PermissionError(
                f"event_upload rejected: peer {ctx.trusted_peer.peer_id} revoked"
            )

        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("events must be an array")

        # Log count + avg payload size on receive, helps quickly identify batch scale
        avg_payload_bytes = 0
        if raw_events:
            payload_lens = [len(r.get("payload", "") or "") for r in raw_events if isinstance(r, dict)]
            avg_payload_bytes = sum(payload_lens) // max(1, len(payload_lens))
        logger.info(
            "event_upload begin peer=%s incoming=%d avg_payload_b64_bytes=%d",
            ctx.trusted_peer.peer_id, len(raw_events), avg_payload_bytes,
        )

        decoded = []
        for idx, raw in enumerate(raw_events):
            if not isinstance(raw, dict):
                raise ValueError(f"event entry [{idx}] must be an object")
            try:
                decoded.append(
                    decode_wire_event(raw, source_peer_id=ctx.trusted_peer.peer_id)
                )
            except Exception as exc:
                # On decode failure show raw content + idx, helps debug cross-language contract errors
                # (e.g. Swift dateEncodingStrategy=.iso8601 misuse → timestamp string)
                logger.error(
                    "event_upload decode failed at idx=%d: %s | raw=%r",
                    idx, exc, raw,
                )
                raise

        received_ids, is_new_flags = event_store.insert_batch(decoded)
        new_count = sum(1 for flag in is_new_flags if flag)
        # Log tags / event_type distribution — confirms whether trainingSample / userCorrection etc.
        # are actually carrying training signals via mesh sync (key observation point)
        tag_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for ev in decoded:
            for t in ev.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
            type_counts[ev.event_type] = type_counts.get(ev.event_type, 0) + 1
        logger.info(
            "event_upload done peer=%s count=%d new=%d duplicated=%d tags=%s types=%s",
            ctx.trusted_peer.peer_id,
            len(received_ids),
            new_count,
            len(received_ids) - new_count,
            tag_counts,
            type_counts,
        )

        # TrustStore auto-updates last_seen
        ctx.trust_store.touch_last_seen(ctx.trusted_peer.peer_id)

        return {
            "op": "event_upload_ack",
            "payload": {"receivedIds": received_ids},
        }

    server.register_handler("event_upload", handle_event_upload)
    logger.info("event_upload handler registered on MeshTransportServer")
