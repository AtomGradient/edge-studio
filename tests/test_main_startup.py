# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

from __future__ import annotations

import errno
import logging
from types import SimpleNamespace

from backend import main as backend_main
from backend.services import (
    classify_ingest,
    device_snapshot_ingest,
    event_ingest,
    halo_capsule_apply_status_ingest,
    halo_capsule_transfer_ack_ingest,
    joint_inference_ingest,
    peer_trust_ingest,
    persona_rpp_input_ingest,
    persona_source_ingest,
    rpp_artifact_ingest,
)
from backend.services import certificate_manager, mesh_transport


def test_edgemesh_port_conflict_logs_actionable_message_without_traceback(monkeypatch, caplog):
    class BusyServer:
        port = 18843

        def register_handler(self, _op, _handler) -> None:
            return None

        def start(self) -> None:
            raise OSError(errno.EADDRINUSE, "Address already in use")

    for module in (
        classify_ingest,
        device_snapshot_ingest,
        event_ingest,
        halo_capsule_apply_status_ingest,
        halo_capsule_transfer_ack_ingest,
        joint_inference_ingest,
        peer_trust_ingest,
        persona_rpp_input_ingest,
        persona_source_ingest,
        rpp_artifact_ingest,
    ):
        monkeypatch.setattr(module, "register", lambda _server, *args, **kwargs: None)

    monkeypatch.setattr(
        certificate_manager,
        "load_or_create",
        lambda: SimpleNamespace(
            peer_id="peer",
            display_name="Edge Studio",
            fingerprint="fp",
        ),
    )
    monkeypatch.setattr(mesh_transport, "get_default_server", lambda: BusyServer())

    with caplog.at_level(logging.WARNING, logger="backend.main"):
        backend_main._auto_start_edgemesh()

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "EdgeMesh is already running on port 18843; Studio UI is still available at http://127.0.0.1:18842"
    ]
    assert all(record.exc_info is None for record in caplog.records)
