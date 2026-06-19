# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import logging
import secrets
import time
from pathlib import Path
from typing import Optional

import asyncio
import json

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from backend.schemas.mesh import (
    DeleteResponse,
    DevicesListResponse,
    DeviceSnapshotIngestRequest,
    DeviceSnapshotIngestResponse,
    DeviceSnapshotLatestResponse,
    EventRecord,
    EventStatsResponse,
    EventsListResponse,
    HaloCapsuleAutomationPreviewResponse,
    HaloCapsuleAutomationRunLatestResponse,
    HaloCapsuleAutomationRunRequest,
    HaloCapsuleAutomationRunResponse,
    HaloCapsuleApplyStatusIngestRequest,
    HaloCapsuleApplyStatusIngestResponse,
    HaloCapsuleApplyStatusLatestResponse,
    HaloCapsuleCoordinatorPlanResponse,
    HaloCapsulePushRequest,
    HaloCapsulePushResponse,
    MeshStatusResponse,
    PairRequestBody,
    PairRequestResponse,
    PeerStatusQuery,
    PeerStatusResponse,
    PinExchangeRequest,
    QRPairingEndpoint,
    QRPairingPayloadModel,
    QRPairingResponse,
    RevokeResponse,
    TrustedPeerModel,
)
from backend.services import mesh_discovery
from backend.services.certificate_manager import load_or_create
from backend.services.mesh_discovery import SERVICE_TYPE as EDGEMESH_SERVICE_TYPE
from backend.services.mesh_events import get_default_bus
from backend.services.mesh_transport import get_default_server
from backend.services.pairing_manager import (
    DEFAULT_EXPIRY_SECONDS,
    PairingPayload,
    get_default_manager,
)
from backend.services.trust_store import get_default_store as get_trust_store
from backend.stores.event_store import get_default_store as get_event_store
from backend.services.device_learning_snapshot_store import (
    DeviceLearningSnapshotError,
    latest_device_learning_snapshot,
    store_device_learning_snapshot,
)
from backend.services.device_lifecycle_automation import (
    build_and_store_device_lifecycle_automation_decision,
)
from backend.services.halo_capsule_package import (
    HaloCapsulePackageError,
    build_neural_imprint_halo_package,
    package_with_download_urls,
    push_halo_capsule_download_offer_to_peer,
)
from backend.services.halo_capsule_apply_status_store import (
    HaloCapsuleApplyStatusError,
    latest_halo_capsule_apply_status,
    store_halo_capsule_apply_status,
)
from backend.services.neural_imprint_artifact_registry import (
    NeuralImprintArtifactRegistryError,
    resolve_neural_imprint_artifact_dir,
)
from backend.services.halo_capsule_coordinator import build_halo_capsule_coordinator_plan
from backend.services.halo_capsule_automation import build_halo_capsule_automation_preview
from backend.services.halo_capsule_automation_run_store import (
    HaloCapsuleAutomationRunStoreError,
    latest_halo_capsule_automation_run,
)
from backend.services.halo_capsule_automation_runner import (
    HaloCapsuleAutomationRunnerError,
    run_halo_capsule_automation_once,
)
from backend.services.joint_inference_ingest import (
    delete_joint_inference_history_item,
    get_joint_inference_history_item,
    list_joint_inference_history,
    stream_joint_inference_continue,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mesh", tags=["mesh"])

_HALO_CAPSULE_DOWNLOAD_TTL_SECONDS = 60 * 60
_halo_capsule_downloads: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_to_model(p: PairingPayload) -> QRPairingPayloadModel:
    return QRPairingPayloadModel(
        version=p.version,
        peerId=p.peer_id,
        displayName=p.display_name,
        role=p.role,
        endpoint=QRPairingEndpoint(
            serviceType=p.service_type,
            serviceName=p.service_name,
            ipv4=p.ipv4,
            port=p.port,
        ),
        certFingerprint=p.cert_fingerprint,
        nonce=p.nonce,
        expiresAt=p.expires_at,
    )


def _mesh_port() -> int:
    return get_default_server().port


def _http_port() -> int:
    from backend.config import PORT
    return PORT


def _snapshot_peer_id(snapshot: dict) -> str:
    identity = snapshot.get("identity") if isinstance(snapshot, dict) else None
    if not isinstance(identity, dict):
        return ""
    return str(identity.get("peer_id") or "").strip()


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@router.post("/pair/qr", response_model=QRPairingResponse)
def create_pairing_qr() -> QRPairingResponse:
    identity = load_or_create()
    ipv4 = mesh_discovery.get_primary_ipv4() or None

    # Service name matches MeshDiscovery broadcast, iOS NWBrowser/NWConnection can use this to discover
    broadcaster = _current_broadcaster()
    service_name = broadcaster.status().get("service_name", "") if broadcaster else ""

    payload = get_default_manager().create_session(
        peer_id=identity.peer_id,
        display_name=identity.display_name,
        role="brain",
        service_type=EDGEMESH_SERVICE_TYPE.rstrip("."),
        service_name=service_name,
        ipv4=ipv4,
        port=_mesh_port(),
        cert_fingerprint=identity.fingerprint,
        ttl_seconds=DEFAULT_EXPIRY_SECONDS,
    )
    return QRPairingResponse(
        payload=_payload_to_model(payload),
        pin=payload.pin,
        ttl_seconds=DEFAULT_EXPIRY_SECONDS,
    )


@router.post("/pair/request", response_model=PairRequestResponse)
def pair_request(body: PairRequestBody, request: Request) -> PairRequestResponse:
    client_ip = request.client.host if request.client else "unknown"
    identity = load_or_create()
    ipv4 = mesh_discovery.get_primary_ipv4() or None
    broadcaster = _current_broadcaster()
    service_name = broadcaster.status().get("service_name", "") if broadcaster else ""

    payload = get_default_manager().create_session(
        peer_id=identity.peer_id,
        display_name=identity.display_name,
        role="brain",
        service_type=EDGEMESH_SERVICE_TYPE.rstrip("."),
        service_name=service_name,
        ipv4=ipv4,
        port=_mesh_port(),
        cert_fingerprint=identity.fingerprint,
        ttl_seconds=DEFAULT_EXPIRY_SECONDS,
        # iOS-initiated sessions require the Mac user to visually verify the PIN
        # and click Approve before PIN exchange / pair_hello can succeed.
        approved=False,
    )

    get_default_bus().broadcast({
        "type": "pair_request",
        "requester_peer_id": body.peer_id,
        "requester_display_name": body.display_name,
        "requester_fingerprint": body.fingerprint,
        "pin": payload.pin,
        "nonce": payload.nonce,
        "ttl_seconds": DEFAULT_EXPIRY_SECONDS,
        "from_ip": client_ip,
    })

    logger.info(
        "pair_request from peer=%s (%s) ip=%s pin=%s****",
        body.peer_id, body.display_name, client_ip, payload.pin[:2],
    )
    return PairRequestResponse(
        pin=payload.pin,
        nonce=payload.nonce,
        ttl_seconds=DEFAULT_EXPIRY_SECONDS,
    )


@router.post("/pair/approve/{nonce}")
def approve_pair_request(nonce: str) -> dict:
    ok = get_default_manager().approve_session(nonce)
    if not ok:
        raise HTTPException(status_code=404, detail=f"nonce {nonce} not found or expired")
    get_default_bus().broadcast({"type": "pair_approved", "nonce": nonce})
    logger.info("Pair session %s approved by Mac user", nonce[:8])
    return {"ok": True, "nonce": nonce}


@router.get("/pair/status/{nonce}")
def pair_status(nonce: str) -> dict:
    mgr = get_default_manager()
    session = mgr.get_session(nonce)
    if session is None:
        return {"state": "unknown"}
    if session.payload.is_expired():
        return {"state": "expired"}
    return {"state": "approved" if session.approved else "pending"}


@router.post("/pair/pin", response_model=QRPairingPayloadModel)
def exchange_pin(body: PinExchangeRequest, request: Request) -> QRPairingPayloadModel:
    client_ip = request.client.host if request.client else "unknown"
    try:
        payload = get_default_manager().lookup_by_pin(body.pin, client_ip)
    except PermissionError as exc:
        logger.warning("pair/pin lockout for %s: %s", client_ip, exc)
        raise HTTPException(status_code=429, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _payload_to_model(payload)


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


@router.get("/devices", response_model=DevicesListResponse)
def list_devices() -> DevicesListResponse:
    identity = load_or_create()
    store = get_trust_store()
    peers = [
        TrustedPeerModel(**p.to_api())
        for p in store.list_all()
    ]
    pending = [
        _payload_to_model(p) for p in get_default_manager().list_pending()
    ]
    return DevicesListResponse(
        local={
            "peer_id": identity.peer_id,
            "display_name": identity.display_name,
            "fingerprint": identity.fingerprint,
            "mesh_port": _mesh_port(),
            "http_port": _http_port(),
            "ipv4": mesh_discovery.get_primary_ipv4() or None,
        },
        peers=peers,
        pending=pending,
    )


@router.post("/devices/{peer_id}/revoke", response_model=RevokeResponse)
def revoke_device(peer_id: str) -> RevokeResponse:
    store = get_trust_store()
    if store.lookup(peer_id) is None:
        raise HTTPException(status_code=404, detail=f"peer {peer_id} not found")
    store.revoke(peer_id)
    # Push-disconnect any currently-active mTLS session so the peer knows instantly.
    # Falls through cleanly if no active connection (most trusted peers sit idle).
    try:
        closed = get_default_server().disconnect_peer(peer_id, reason="revoked")
    except Exception as exc:  # noqa: BLE001
        logger.warning("disconnect_peer on revoke failed: %s", exc)
        closed = False
    get_default_bus().broadcast({"type": "peer_revoked", "peer_id": peer_id})
    logger.info("Peer %s revoked via API (live_conn_closed=%s)", peer_id, closed)
    return RevokeResponse(peer_id=peer_id)


@router.delete("/devices/{peer_id}", response_model=DeleteResponse)
def delete_device(peer_id: str) -> DeleteResponse:
    store = get_trust_store()
    if store.lookup(peer_id) is None:
        raise HTTPException(status_code=404, detail=f"peer {peer_id} not found")
    store.delete(peer_id)
    try:
        closed = get_default_server().disconnect_peer(peer_id, reason="deleted")
    except Exception as exc:  # noqa: BLE001
        logger.warning("disconnect_peer on delete failed: %s", exc)
        closed = False
    get_default_bus().broadcast({"type": "peer_deleted", "peer_id": peer_id})
    logger.info("Peer %s deleted via API (live_conn_closed=%s)", peer_id, closed)
    return DeleteResponse(peer_id=peer_id)


@router.post("/peer_status", response_model=PeerStatusResponse)
def peer_status(body: PeerStatusQuery) -> PeerStatusResponse:
    if not (body.peer_id or body.fingerprint):
        raise HTTPException(status_code=400, detail="must provide peer_id or fingerprint")
    store = get_trust_store()
    peer = None
    if body.fingerprint:
        peer = store.lookup_by_fingerprint(body.fingerprint)
    if peer is None and body.peer_id:
        peer = store.lookup(body.peer_id)
    if peer is None:
        return PeerStatusResponse(
            known=False, trusted=False, revoked=False,
            peer_id=body.peer_id, display_name=None,
        )
    return PeerStatusResponse(
        known=True,
        trusted=not peer.revoked,
        revoked=peer.revoked,
        peer_id=peer.peer_id,
        display_name=peer.display_name,
    )


# ---------------------------------------------------------------------------
# Device learning snapshots
# ---------------------------------------------------------------------------


@router.post("/device_snapshot", response_model=DeviceSnapshotIngestResponse)
def ingest_device_snapshot(body: DeviceSnapshotIngestRequest) -> DeviceSnapshotIngestResponse:
    """Store a device learning snapshot via local HTTP tooling.

    Real paired devices normally send the same payload over the authenticated
    `device_state_snapshot` mesh op. This endpoint is the read-side/test mirror
    used by EdgeStudio and local tools; it does not trigger learning or pushes.
    """
    peer_id = body.peer_id or _snapshot_peer_id(body.snapshot)
    if not peer_id:
        raise HTTPException(
            status_code=400,
            detail="peer_id is required when snapshot.identity.peer_id is absent",
        )
    try:
        receipt = store_device_learning_snapshot(
            peer_id,
            body.snapshot,
            source=body.source,
        )
    except DeviceLearningSnapshotError as exc:
        raise HTTPException(status_code=400, detail=exc.to_error()) from exc
    get_default_bus().broadcast({
        "type": "device_state_snapshot",
        "peer_id": receipt["peer_id"],
        "phase": receipt["lifecycle"]["phase"],
        "phase_label": receipt["lifecycle"]["phase_label"],
        "snapshot_sha256": receipt["snapshot_sha256"],
    })
    automation = _record_lifecycle_automation_decision(receipt["peer_id"], source="api")
    if automation is not None:
        get_default_bus().broadcast({
            "type": "device_lifecycle_automation_decision",
            "peer_id": receipt["peer_id"],
            "decision_id": automation["receipt"]["decision_id"],
            "decision_key": automation["receipt"]["decision_key"],
            "candidate_kind": automation["receipt"]["candidate_kind"],
            "policy_status": automation["receipt"]["policy_status"],
            "side_effects_executed": automation["receipt"]["side_effects_executed"],
        })
    return DeviceSnapshotIngestResponse(receipt=receipt)


@router.get(
    "/devices/{peer_id}/snapshot",
    response_model=DeviceSnapshotLatestResponse,
)
def get_device_snapshot(peer_id: str) -> DeviceSnapshotLatestResponse:
    """Return the latest stored learning snapshot for a paired device."""
    try:
        record = latest_device_learning_snapshot(peer_id)
    except DeviceLearningSnapshotError as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"snapshot for peer {peer_id} not found")
    return DeviceSnapshotLatestResponse(
        receipt=record["receipt"],
        snapshot=record["snapshot"],
    )


@router.get("/devices/{peer_id}/lifecycle")
def get_device_lifecycle(peer_id: str) -> dict:
    """Return only the derived lifecycle status for a paired device."""
    try:
        record = latest_device_learning_snapshot(peer_id)
    except DeviceLearningSnapshotError as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"snapshot for peer {peer_id} not found")
    return {
        "ok": True,
        "peer_id": peer_id,
        "receipt": record["receipt"],
        "lifecycle": record["receipt"]["lifecycle"],
    }


def _record_lifecycle_automation_decision(peer_id: str, *, source: str) -> dict | None:
    try:
        connected = get_default_server().is_peer_connected(peer_id)
        return build_and_store_device_lifecycle_automation_decision(
            peer_id,
            connected=connected,
            source=source,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "device_lifecycle_automation audit skipped peer=%s: %s",
            peer_id,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Halo capsule push
# ---------------------------------------------------------------------------


@router.get("/halo_capsules/download/{transfer_id}/{file_name}")
def download_halo_capsule_file(
    transfer_id: str,
    file_name: str,
    token: str = Query(..., min_length=16),
) -> FileResponse:
    """Serve one file from a host-offered Halo capsule package.

    The mesh control frame is authenticated; the file endpoint is intentionally
    narrow and transfer-scoped so a device can fetch large Neural Imprint artifacts
    with URLSession instead of keeping a long mTLS bulk stream open.
    """

    _prune_expired_halo_capsule_downloads()
    record = _halo_capsule_downloads.get(transfer_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"transfer {transfer_id} not found")
    if str(record.get("token") or "") != token:
        raise HTTPException(status_code=403, detail="invalid download token")
    files = record.get("files") if isinstance(record, dict) else None
    if not isinstance(files, dict) or file_name not in files:
        raise HTTPException(status_code=404, detail=f"file {file_name} not found")
    path = Path(files[file_name])
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"file {file_name} not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=file_name,
    )


@router.post("/halo_capsules/push", response_model=HaloCapsulePushResponse)
def push_halo_capsule(body: HaloCapsulePushRequest) -> HaloCapsulePushResponse:
    """Push an existing Neural Imprint Halo capsule package to a connected peer.

    This endpoint is deliberately only the host-side downlink trigger. Device
    restore/apply status is a separate Phase B ack path.
    """

    peer_id = body.peer_id.strip()
    peer = get_trust_store().lookup(peer_id)
    if peer is None:
        raise HTTPException(status_code=404, detail=f"peer {peer_id} not found")
    if getattr(peer, "revoked", False):
        raise HTTPException(status_code=403, detail=f"peer {peer_id} is revoked")

    server = get_default_server()
    if not server.is_peer_connected(peer_id):
        raise HTTPException(status_code=409, detail=f"peer {peer_id} is not connected")

    min_runtime_version = (
        body.min_runtime_version.strip()
        if body.min_runtime_version
        else _min_runtime_version_from_latest_snapshot(peer_id)
    )
    try:
        neural_imprint_dir = _neural_imprint_dir_for_push(body)
        package = _package_with_registered_downloads(build_neural_imprint_halo_package(
            neural_imprint_dir,
            min_runtime_version=min_runtime_version,
            transfer_id=body.transfer_id,
            capsule_id=body.capsule_id,
        ))
    except HaloCapsulePackageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        receipt = push_halo_capsule_download_offer_to_peer(
            server,
            peer_id,
            package,
        )
    except HaloCapsulePackageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    capsule = package.message["capsule"]
    artifact = capsule["artifact"]
    response = HaloCapsulePushResponse(
        peer_id=peer_id,
        transfer_id=package.transfer_id,
        capsule_id=capsule["capsule_id"],
        base_model_id=capsule["base_model_id"],
        min_runtime_version=min_runtime_version,
        artifact_sha256=artifact["sha256"],
        artifact_file_count=len(artifact["files"]),
        frame_count=int(receipt["frame_count"]),
        payload_bytes=int(receipt["payload_bytes"]),
        chunk_size=body.chunk_size,
    )
    _broadcast_halo_capsule_push(response)
    logger.info(
        "Halo capsule pushed to peer=%s transfer=%s frames=%d bytes=%d",
        response.peer_id,
        response.transfer_id,
        response.frame_count,
        response.payload_bytes,
    )
    return response


def _broadcast_halo_capsule_push(response: HaloCapsulePushResponse) -> None:
    get_default_bus().broadcast({
        "type": "halo_capsule_push",
        "peer_id": response.peer_id,
        "transfer_id": response.transfer_id,
        "capsule_id": response.capsule_id,
        "artifact_sha256": response.artifact_sha256,
        "frame_count": response.frame_count,
        "payload_bytes": response.payload_bytes,
    })


def _package_with_registered_downloads(package):
    token = _register_halo_capsule_download(package)
    return package_with_download_urls(
        package,
        base_url=_halo_capsule_download_base_url(),
        download_token=token,
    )


def _halo_capsule_download_base_url() -> str:
    host = mesh_discovery.get_primary_ipv4() or "127.0.0.1"
    return f"http://{host}:{_http_port()}/api/mesh/halo_capsules/download"


def _register_halo_capsule_download(package) -> str:
    _prune_expired_halo_capsule_downloads()
    token = secrets.token_urlsafe(32)
    files = {
        str(file_spec["name"]): package.package_directory / str(file_spec["name"])
        for file_spec in package.files
    }
    _halo_capsule_downloads[package.transfer_id] = {
        "expires_at": time.time() + _HALO_CAPSULE_DOWNLOAD_TTL_SECONDS,
        "files": files,
        "token": token,
    }
    return token


def _prune_expired_halo_capsule_downloads() -> None:
    now = time.time()
    expired = [
        transfer_id
        for transfer_id, record in _halo_capsule_downloads.items()
        if float(record.get("expires_at", 0)) < now
    ]
    for transfer_id in expired:
        _halo_capsule_downloads.pop(transfer_id, None)


def _neural_imprint_dir_for_push(body: HaloCapsulePushRequest) -> Path:
    has_neural_imprint_path = bool(
        getattr(body, "neural_imprint_dir", None) and body.neural_imprint_dir.strip()
    )
    has_artifact_id = bool(body.artifact_id and body.artifact_id.strip())
    if has_neural_imprint_path == has_artifact_id:
        raise HTTPException(
            status_code=400,
            detail="exactly one of neural_imprint_dir or artifact_id is required",
        )
    if has_neural_imprint_path:
        return Path(str(body.neural_imprint_dir))
    try:
        return resolve_neural_imprint_artifact_dir(str(body.artifact_id))
    except NeuralImprintArtifactRegistryError as exc:
        status = 404 if exc.code == "artifact_not_found" else 400
        raise HTTPException(status_code=status, detail=exc.to_error()) from exc


def _min_runtime_version_from_latest_snapshot(peer_id: str) -> str:
    try:
        record = latest_device_learning_snapshot(peer_id)
    except DeviceLearningSnapshotError as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc
    if record is None:
        raise HTTPException(
            status_code=400,
            detail="min_runtime_version is required when no latest device snapshot exists",
        )
    snapshot = record.get("snapshot")
    identity = snapshot.get("identity") if isinstance(snapshot, dict) else None
    version = str(identity.get("edge_kit_version") or "").strip() if isinstance(identity, dict) else ""
    if not version:
        raise HTTPException(
            status_code=400,
            detail="min_runtime_version is required when latest snapshot has no edge_kit_version",
        )
    return version


# ---------------------------------------------------------------------------
# Halo capsule apply status
# ---------------------------------------------------------------------------


@router.post(
    "/halo_capsules/apply_status",
    response_model=HaloCapsuleApplyStatusIngestResponse,
)
def ingest_halo_capsule_apply_status(
    body: HaloCapsuleApplyStatusIngestRequest,
) -> HaloCapsuleApplyStatusIngestResponse:
    """Store a Halo capsule apply status via local HTTP tooling.

    Real devices normally report the same payload over the authenticated
    `halo_capsule_apply_status` mesh op. This endpoint is the read-side/test
    mirror and does not trigger retries or restore orchestration.
    """

    try:
        receipt = store_halo_capsule_apply_status(
            body.peer_id,
            body.payload,
            source=body.source,
        )
    except HaloCapsuleApplyStatusError as exc:
        raise HTTPException(status_code=400, detail=exc.to_error()) from exc
    get_default_bus().broadcast({
        "type": "halo_capsule_apply_status",
        "peer_id": receipt["peer_id"],
        "transfer_id": receipt["transfer_id"],
        "capsule_id": receipt["capsule_id"],
        "status": receipt["status"],
        "artifact_sha256": receipt.get("artifact_sha256"),
        "canonical_sha256": receipt.get("canonical_sha256"),
        "runtime_version": receipt.get("runtime_version"),
        "prefix_token_count": receipt.get("prefix_token_count"),
        "error_code": receipt.get("error_code"),
    })
    return HaloCapsuleApplyStatusIngestResponse(receipt=receipt)


@router.get(
    "/devices/{peer_id}/halo_capsules/apply_status/latest",
    response_model=HaloCapsuleApplyStatusLatestResponse,
)
def get_halo_capsule_apply_status(
    peer_id: str,
    transfer_id: Optional[str] = Query(default=None),
    capsule_id: Optional[str] = Query(default=None),
) -> HaloCapsuleApplyStatusLatestResponse:
    """Return the latest stored Halo capsule apply status for one peer."""

    if (transfer_id is None) != (capsule_id is None):
        raise HTTPException(
            status_code=400,
            detail="transfer_id and capsule_id must be supplied together",
        )
    try:
        record = latest_halo_capsule_apply_status(
            peer_id,
            transfer_id=transfer_id,
            capsule_id=capsule_id,
        )
    except HaloCapsuleApplyStatusError as exc:
        status_code = 400 if exc.code in {"missing_required_id", "invalid_id"} else 500
        raise HTTPException(status_code=status_code, detail=exc.to_error()) from exc
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Halo capsule apply status for peer {peer_id} not found",
        )
    return HaloCapsuleApplyStatusLatestResponse(
        receipt=record["receipt"],
        payload=record["payload"],
    )


@router.get(
    "/devices/{peer_id}/halo_capsules/plan",
    response_model=HaloCapsuleCoordinatorPlanResponse,
)
def get_halo_capsule_coordinator_plan(peer_id: str) -> HaloCapsuleCoordinatorPlanResponse:
    """Return a read-only Halo capsule distribution plan for one peer.

    This endpoint bridges the manual push panel and future automation. It only
    explains the next action; it does not push, retry, or mutate device state.
    """

    peer = get_trust_store().lookup(peer_id)
    if peer is None:
        raise HTTPException(status_code=404, detail=f"peer {peer_id} not found")
    if getattr(peer, "revoked", False):
        raise HTTPException(status_code=403, detail=f"peer {peer_id} is revoked")
    connected = get_default_server().is_peer_connected(peer_id)
    try:
        plan = build_halo_capsule_coordinator_plan(peer_id, connected=connected)
    except (DeviceLearningSnapshotError, HaloCapsuleApplyStatusError) as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc
    return HaloCapsuleCoordinatorPlanResponse(**plan)


@router.get(
    "/halo_capsules/automation/preview",
    response_model=HaloCapsuleAutomationPreviewResponse,
)
def get_halo_capsule_automation_preview() -> HaloCapsuleAutomationPreviewResponse:
    """Return a dry-run preview of host-driven Halo capsule automation."""

    store = get_trust_store()
    server = get_default_server()
    try:
        preview = build_halo_capsule_automation_preview(
            store.list_all(),
            is_peer_connected=server.is_peer_connected,
        )
    except (DeviceLearningSnapshotError, HaloCapsuleApplyStatusError) as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc
    return HaloCapsuleAutomationPreviewResponse(**preview)


@router.post(
    "/halo_capsules/automation/run",
    response_model=HaloCapsuleAutomationRunResponse,
)
def run_halo_capsule_automation(
    body: HaloCapsuleAutomationRunRequest,
) -> HaloCapsuleAutomationRunResponse:
    """Run one bounded Halo capsule automation pass.

    Defaults to dry-run. Real pushes require explicit peer_ids, so this endpoint
    cannot silently fan out to every paired device.
    """

    store = get_trust_store()
    server = get_default_server()
    def push_candidate(
        entry: dict,
        peer_id: str,
        artifact_id: str,
        chunk_size: int,
    ) -> dict:
        _ = (peer_id, artifact_id)
        return _model_dump(_execute_halo_capsule_automation_push(
            server,
            entry,
            chunk_size=chunk_size,
        ))

    try:
        response_data = run_halo_capsule_automation_once(
            dry_run=body.dry_run,
            peer_ids=body.peer_ids,
            max_pushes=body.max_pushes,
            chunk_size=body.chunk_size,
            peers=store.list_all(),
            is_peer_connected=server.is_peer_connected,
            push_candidate=push_candidate,
            source="api",
            request_payload=_model_dump(body),
        )
    except HaloCapsuleAutomationRunnerError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except (DeviceLearningSnapshotError, HaloCapsuleApplyStatusError) as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc

    receipt = response_data["receipt"]
    get_default_bus().broadcast({
        "type": "halo_capsule_automation_run",
        "run_id": receipt["run_id"],
        "dry_run": receipt["dry_run"],
        "attempted_count": receipt["attempted_count"],
        "pushed_count": receipt["pushed_count"],
        "peer_ids": receipt["peer_ids"],
    })
    return HaloCapsuleAutomationRunResponse(**response_data)


@router.get(
    "/halo_capsules/automation/runs/latest",
    response_model=HaloCapsuleAutomationRunLatestResponse,
)
def get_latest_halo_capsule_automation_run() -> HaloCapsuleAutomationRunLatestResponse:
    """Return the latest persisted Halo capsule automation run receipt."""

    try:
        record = latest_halo_capsule_automation_run()
    except HaloCapsuleAutomationRunStoreError as exc:
        raise HTTPException(status_code=500, detail=exc.to_error()) from exc
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Halo capsule automation run not found",
        )
    return HaloCapsuleAutomationRunLatestResponse(
        receipt=record["receipt"],
        request=record["request"],
        response=record["response"],
    )


def _execute_halo_capsule_automation_push(
    server,
    entry: dict,
    *,
    chunk_size: int,
) -> HaloCapsulePushResponse:
    peer_id = str(entry.get("peer_id") or "").strip()
    action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
    push_request = action.get("push_request") if isinstance(action.get("push_request"), dict) else {}
    artifact_id = str(push_request.get("artifact_id") or "").strip()
    if not peer_id or not artifact_id:
        raise HaloCapsulePackageError("automation candidate is missing peer_id or artifact_id")
    if not server.is_peer_connected(peer_id):
        raise HaloCapsulePackageError(f"peer {peer_id} is not connected")

    min_runtime_version = _min_runtime_version_from_latest_snapshot(peer_id)
    neural_imprint_dir = resolve_neural_imprint_artifact_dir(artifact_id)
    package = _package_with_registered_downloads(build_neural_imprint_halo_package(
        neural_imprint_dir,
        min_runtime_version=min_runtime_version,
    ))
    receipt = push_halo_capsule_download_offer_to_peer(
        server,
        peer_id,
        package,
    )
    capsule = package.message["capsule"]
    artifact = capsule["artifact"]
    response = HaloCapsulePushResponse(
        peer_id=peer_id,
        transfer_id=package.transfer_id,
        capsule_id=capsule["capsule_id"],
        base_model_id=capsule["base_model_id"],
        min_runtime_version=min_runtime_version,
        artifact_sha256=artifact["sha256"],
        artifact_file_count=len(artifact["files"]),
        frame_count=int(receipt["frame_count"]),
        payload_bytes=int(receipt["payload_bytes"]),
        chunk_size=chunk_size,
    )
    _broadcast_halo_capsule_push(response)
    return response


def _model_dump(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@router.get("/status", response_model=MeshStatusResponse)
def mesh_status() -> MeshStatusResponse:
    identity = load_or_create()
    server = get_default_server()
    store = get_trust_store()
    broadcaster = _current_broadcaster()
    return MeshStatusResponse(
        transport_running=server.is_running(),
        discovery_running=broadcaster.is_running() if broadcaster else False,
        peer_id=identity.peer_id,
        fingerprint=identity.fingerprint,
        mesh_port=server.port,
        http_port=_http_port(),
        ipv4=mesh_discovery.get_primary_ipv4() or None,
        peers_count=len(store.list_all()),
        pending_count=len(get_default_manager().list_pending()),
    )


# ---------------------------------------------------------------------------
# Events (training pipeline consumption entry)
# ---------------------------------------------------------------------------


@router.get("/events", response_model=EventsListResponse)
def list_events(
    tags: Optional[list[str]] = Query(
        default=None,
        description="tag 过滤（OR 语义，多个 ?tags=a&tags=b）",
    ),
    app_id: Optional[str] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    source_peer_id: Optional[str] = Query(default=None),
    since: Optional[float] = Query(
        default=None, description="Unix epoch seconds, 下界（含）"
    ),
    until: Optional[float] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    include_payload: bool = Query(
        default=False,
        description="true → 返回 payload_b64，false → 仅返回元信息",
    ),
) -> EventsListResponse:
    store = get_event_store()
    since_ms = int(since * 1000) if since is not None else None
    until_ms = int(until * 1000) if until is not None else None

    events = store.query(
        tags=tags,
        app_id=app_id,
        event_type=event_type,
        source_peer_id=source_peer_id,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=limit,
        offset=offset,
    )

    total = store.count() if not any([tags, app_id, event_type, source_peer_id, since_ms, until_ms]) else len(events) + offset
    return EventsListResponse(
        total=total,
        returned=len(events),
        events=[
            EventRecord(**e.to_api(include_payload=include_payload))
            for e in events
        ],
    )


@router.get("/events/stats", response_model=EventStatsResponse)
def events_stats() -> EventStatsResponse:
    stats = get_event_store().stats()
    return EventStatsResponse(**stats)


@router.get("/events/{event_id}", response_model=EventRecord)
def get_event(event_id: str, include_payload: bool = Query(default=True)) -> EventRecord:
    event = get_event_store().get(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"event {event_id} not found")
    return EventRecord(**event.to_api(include_payload=include_payload))


@router.get("/joint_inference/history")
def get_joint_inference_history(
    peer_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict:
    """Return recent C1 joint inference requests observed by this Mac host."""

    return list_joint_inference_history(limit=limit, peer_id=peer_id)


@router.get("/joint_inference/history/{request_id}")
def get_joint_inference_history_detail(request_id: str) -> dict:
    """Return a full joint inference request record for drill-down UI."""

    record = get_joint_inference_history_item(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="joint inference request not found")
    return {"ok": True, "item": record}


@router.delete("/joint_inference/history/{request_id}")
def delete_joint_inference_history_detail(request_id: str) -> dict:
    """Delete one joint inference history conversation."""

    ok = delete_joint_inference_history_item(request_id)
    if not ok:
        raise HTTPException(status_code=404, detail="joint inference request not found")
    return {"ok": True, "request_id": request_id}


@router.post("/joint_inference/history/{request_id}/continue")
async def continue_joint_inference_history(request_id: str, request: Request) -> StreamingResponse:
    """Continue a recorded joint inference turn on the Mac host.

    Response is newline-delimited JSON events using the same event schema as the
    mTLS C1 path, so the frontend can render token streaming without opening a
    WebSocket.
    """

    payload = await request.json()
    if get_joint_inference_history_item(request_id) is None:
        raise HTTPException(status_code=404, detail="joint inference request not found")

    def iter_events():
        for event in stream_joint_inference_continue(
            parent_request_id=request_id,
            payload=payload,
        ):
            yield json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"

    return StreamingResponse(iter_events(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# WebSocket — live mesh lifecycle event stream (frontend subscribes here)
# ---------------------------------------------------------------------------


@router.websocket("/events/stream")
async def mesh_events_stream(ws: WebSocket) -> None:
    """Low-latency push channel for mesh state changes.

    Frontend subscribes once on DevicesPage mount; receives JSON events for:
      - peer_paired / peer_revoked / peer_deleted
      - peer_connected / peer_disconnected  (mTLS session lifecycle)
      - pair_request    (iOS tap-to-pair — C1 territory)

    Replaces polling-based UI refreshes with push (lower latency + less CPU).
    The frontend still keeps a periodic refetch as a safety net; WS just
    makes reactive updates instant.

    Keepalive: if the bus is idle >30s, we send a `{"type":"keepalive"}` frame
    so proxies (nginx et al.) don't close the connection.
    """
    await ws.accept()
    bus = get_default_bus()
    q = bus.subscribe()
    logger.info("mesh events WS connected (total subs=%d)", bus.subscriber_count())
    try:
        while True:
            event = await bus.next_event(q, timeout=30.0)
            if event is None:
                # Idle — send keepalive so intermediaries don't drop the socket
                await ws.send_text(json.dumps({"type": "keepalive"}))
                continue
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("mesh events WS error: %s", exc)
    finally:
        bus.unsubscribe(q)
        logger.info("mesh events WS disconnected (total subs=%d)", bus.subscriber_count())


# ---------------------------------------------------------------------------
# Internal — broadcaster singleton accessor (created lazily in main.py startup)
# ---------------------------------------------------------------------------


def _current_broadcaster() -> Optional[object]:
    # mesh_discovery._default_broadcaster is private — exists as long as it has been started
    return getattr(mesh_discovery, "_default_broadcaster", None)
