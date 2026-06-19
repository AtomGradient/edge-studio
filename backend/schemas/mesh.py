# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Pydantic schemas for /api/mesh/* endpoints (P0 transport security + event stream)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# QR payload  (mirrors Swift `QRPairingPayload`)
# ---------------------------------------------------------------------------


class QRPairingEndpoint(BaseModel):
    serviceType: str
    serviceName: str
    ipv4: Optional[str] = None
    port: int


class QRPairingPayloadModel(BaseModel):
    version: int = 1
    peerId: str
    displayName: str
    role: str = Field(..., description='"brain" | "sensor" | "peer"')
    endpoint: QRPairingEndpoint
    certFingerprint: str = Field(..., min_length=64, max_length=64)
    nonce: str
    expiresAt: int = Field(..., description="Unix seconds")


class QRPairingResponse(BaseModel):
    payload: QRPairingPayloadModel
    pin: str = Field(..., description="6-char base32 code displayed to user")
    ttl_seconds: int


class PinExchangeRequest(BaseModel):
    pin: str = Field(..., min_length=1, max_length=16)


class PairRequestBody(BaseModel):
    peer_id: str = Field(..., min_length=1, max_length=128)
    display_name: str = Field(..., min_length=1, max_length=128)
    fingerprint: str = Field(..., min_length=64, max_length=64)


class PairRequestResponse(BaseModel):
    pin: str
    nonce: str
    ttl_seconds: int


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


class TrustedPeerModel(BaseModel):
    peer_id: str
    display_name: str
    fingerprint: str
    role: str
    paired_at: float
    last_seen_at: Optional[float] = None
    revoked: bool
    # Stage 3 P1.1: real-time sensor observations carried in keepalive ping (PingStats JSON dict).
    # nil = peer is a legacy client / has never reported. Fields include appId / eventStoreTotal /
    # factsClassified / factsRawUnclassified.
    last_stats: Optional[dict] = None
    last_stats_at: Optional[float] = None


class DevicesListResponse(BaseModel):
    local: dict = Field(..., description="This Mac's identity (peer_id, display_name, fingerprint, mesh_port, http_port, ipv4)")
    peers: list[TrustedPeerModel]
    pending: list[QRPairingPayloadModel]


class RevokeResponse(BaseModel):
    ok: bool = True
    peer_id: str


class DeleteResponse(BaseModel):
    ok: bool = True
    peer_id: str


class PeerStatusQuery(BaseModel):
    peer_id: Optional[str] = None
    fingerprint: Optional[str] = None


class PeerStatusResponse(BaseModel):
    known: bool
    trusted: bool
    revoked: bool
    peer_id: Optional[str] = None
    display_name: Optional[str] = None


class DeviceSnapshotIngestRequest(BaseModel):
    """HTTP mirror of the EdgeMesh `device_state_snapshot` op."""

    peer_id: Optional[str] = Field(
        default=None,
        description="Trusted peer id. Optional when snapshot.identity.peer_id is present.",
    )
    snapshot: dict[str, Any] = Field(..., description="DeviceLearningSnapshot JSON payload")
    source: str = Field(default="api", description="api | mesh | test")


class DeviceLifecycleStatus(BaseModel):
    schema_version: str
    phase: str
    phase_label: str
    ready_for_persona_chat: bool
    recommended_actions: list[str]
    reasons: list[str]


class DeviceSnapshotReceipt(BaseModel):
    schema_version: str
    peer_id: str
    source: str
    received_at: float
    snapshot_sha256: str
    lifecycle: DeviceLifecycleStatus


class DeviceSnapshotIngestResponse(BaseModel):
    ok: bool = True
    receipt: DeviceSnapshotReceipt


class DeviceSnapshotLatestResponse(BaseModel):
    ok: bool = True
    receipt: DeviceSnapshotReceipt
    snapshot: dict[str, Any]


class HaloCapsulePushRequest(BaseModel):
    """Request body for host-initiated Halo capsule package transfer."""

    peer_id: str = Field(..., min_length=1, max_length=128)
    neural_imprint_dir: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Local directory containing neural_imprint.safetensors and sidecar files.",
    )
    artifact_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Neural Imprint artifact registry id. Alternative to neural_imprint_dir.",
    )
    min_runtime_version: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Minimum EdgeKit runtime version. Defaults to latest snapshot edge_kit_version.",
    )
    transfer_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    capsule_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    chunk_size: int = Field(
        default=1024 * 1024,
        ge=1,
        le=8 * 1024 * 1024,
        description="Binary chunk size for mesh transfer frames.",
    )


class HaloCapsulePushResponse(BaseModel):
    ok: bool = True
    peer_id: str
    transfer_id: str
    capsule_id: str
    base_model_id: str
    min_runtime_version: str
    artifact_sha256: str
    artifact_file_count: int
    frame_count: int
    payload_bytes: int
    chunk_size: int


class HaloCapsuleApplyStatusIngestRequest(BaseModel):
    """HTTP mirror of the EdgeMesh `halo_capsule_apply_status` op."""

    peer_id: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any] = Field(..., description="HaloCapsuleApplyStatusPayload JSON")
    source: str = Field(default="api", description="api | mesh | test")


class HaloCapsuleApplyStatusReceipt(BaseModel):
    schema_version: str
    peer_id: str
    source: str
    received_at: float
    apply_status_sha256: str
    transfer_id: str
    capsule_id: str
    status: str
    artifact_sha256: Optional[str] = None
    canonical_sha256: Optional[str] = None
    runtime_version: Optional[str] = None
    prefix_token_count: Optional[int] = None
    applied_at_unix_seconds: Optional[float] = None
    error_code: Optional[str] = None


class HaloCapsuleTransferAckReceipt(BaseModel):
    schema_version: str
    peer_id: str
    source: str
    received_at: float
    ack_sha256: str
    ack_kind: str
    transfer_id: str
    accepted: bool
    reason: Optional[str] = None
    canonical_sha256: Optional[str] = None


class HaloCapsuleApplyStatusIngestResponse(BaseModel):
    ok: bool = True
    receipt: HaloCapsuleApplyStatusReceipt


class HaloCapsuleApplyStatusLatestResponse(BaseModel):
    ok: bool = True
    receipt: HaloCapsuleApplyStatusReceipt
    payload: dict[str, Any]


class HaloCapsuleCoordinatorPlanResponse(BaseModel):
    ok: bool = True
    schema_version: str
    peer_id: str
    connected: bool
    action: dict[str, Any]
    snapshot_sha256: Optional[str] = None
    lifecycle: Optional[DeviceLifecycleStatus] = None
    selected_model_id: Optional[str] = None
    load_state: Optional[str] = None
    data_readiness: Optional[str] = None
    learning: Optional[dict[str, Any]] = None
    artifact_count: Optional[int] = None
    matched_artifact: Optional[dict[str, Any]] = None
    last_apply_status: Optional[HaloCapsuleApplyStatusReceipt] = None
    last_transfer_ack: Optional[HaloCapsuleTransferAckReceipt] = None


class HaloCapsuleAutomationPreviewEntry(BaseModel):
    peer_id: str
    display_name: str
    connected: bool
    would_push: bool
    action: dict[str, Any]
    plan: dict[str, Any]


class HaloCapsuleAutomationPreviewResponse(BaseModel):
    ok: bool = True
    schema_version: str
    dry_run: bool
    peer_count: int
    candidate_count: int
    skipped_revoked_count: int
    entries: list[HaloCapsuleAutomationPreviewEntry]


class HaloCapsuleAutomationRunRequest(BaseModel):
    dry_run: bool = Field(
        default=True,
        description="true previews only; false pushes selected candidates once.",
    )
    peer_ids: list[str] = Field(
        default_factory=list,
        description="Required when dry_run=false. Limits execution to explicit peers.",
    )
    max_pushes: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Upper bound for one run. Defaults to one device.",
    )
    chunk_size: int = Field(
        default=1024 * 1024,
        ge=1,
        le=8 * 1024 * 1024,
        description="Binary chunk size for mesh transfer frames.",
    )


class HaloCapsuleAutomationRunResult(BaseModel):
    peer_id: str
    display_name: Optional[str] = None
    dry_run: bool
    would_push: bool
    action_kind: str
    status: str
    push: Optional[HaloCapsulePushResponse] = None
    error: Optional[str] = None


class HaloCapsuleAutomationRunReceipt(BaseModel):
    schema_version: str
    run_id: str
    source: str
    received_at: float
    run_sha256: str
    dry_run: bool
    attempted_count: int
    pushed_count: int
    peer_ids: list[str]


class HaloCapsuleAutomationRunResponse(BaseModel):
    ok: bool = True
    dry_run: bool
    attempted_count: int
    pushed_count: int
    preview: dict[str, Any]
    results: list[HaloCapsuleAutomationRunResult]
    receipt: Optional[HaloCapsuleAutomationRunReceipt] = None


class HaloCapsuleAutomationRunLatestResponse(BaseModel):
    ok: bool = True
    receipt: HaloCapsuleAutomationRunReceipt
    request: dict[str, Any]
    response: dict[str, Any]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class MeshStatusResponse(BaseModel):
    transport_running: bool
    discovery_running: bool
    peer_id: str
    fingerprint: str
    mesh_port: int
    http_port: int
    ipv4: Optional[str] = None
    peers_count: int
    pending_count: int


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventRecord(BaseModel):
    id: str
    timestamp: float                      # Unix seconds (more convenient for frontend / training pipeline)
    app_id: str
    event_type: str
    tags: list[str]
    source_peer_id: Optional[str] = None
    payload_size: int
    payload_b64: Optional[str] = None


class EventsListResponse(BaseModel):
    total: int                            # Total matches for current filter
    returned: int                         # Count returned in this response (subject to limit/offset)
    events: list[EventRecord]


class EventStatsResponse(BaseModel):
    total_events: int
    total_bytes: int
    oldest_timestamp: Optional[float] = None
    newest_timestamp: Optional[float] = None
    per_type: dict[str, int]
    per_source_peer: dict[str, int]
