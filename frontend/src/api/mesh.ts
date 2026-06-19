// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * EdgeMesh REST API client (P0 transport security + event stream).
 *
 * Mirrors backend/schemas/mesh.py. Types use camelCase fields from Swift
 * `QRPairingPayload` where relevant, snake_case for local-only structures.
 */

import client from './client';

// ---------------------------------------------------------------------------
// QR pairing payload (wire-compat with Swift QRPairingPayload)
// ---------------------------------------------------------------------------

export interface QRPairingEndpoint {
  serviceType: string;
  serviceName: string;
  ipv4: string | null;
  port: number;
}

export interface QRPairingPayload {
  version: number;
  peerId: string;
  displayName: string;
  role: 'brain' | 'sensor' | 'peer';
  endpoint: QRPairingEndpoint;
  certFingerprint: string;
  nonce: string;
  expiresAt: number;          // Unix seconds
}

export interface QRPairingResponse {
  payload: QRPairingPayload;
  pin: string;                // 6-char base32
  ttl_seconds: number;
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

export interface LocalIdentity {
  peer_id: string;
  display_name: string;
  fingerprint: string;
  mesh_port: number;
  http_port: number;
  ipv4: string | null;
}

export interface TrustedPeer {
  peer_id: string;
  display_name: string;
  fingerprint: string;
  role: string;
  paired_at: number;            // Unix seconds
  last_seen_at: number | null;
  revoked: boolean;
}

export interface DevicesListResponse {
  local: LocalIdentity;
  peers: TrustedPeer[];
  pending: QRPairingPayload[];
}

export interface MeshStatus {
  transport_running: boolean;
  discovery_running: boolean;
  peer_id: string;
  fingerprint: string;
  mesh_port: number;
  http_port: number;
  ipv4: string | null;
  peers_count: number;
  pending_count: number;
}

// ---------------------------------------------------------------------------
// Events (training pipeline view)
// ---------------------------------------------------------------------------

export interface EventRecord {
  id: string;
  timestamp: number;             // Unix seconds
  app_id: string;
  event_type: string;
  tags: string[];
  source_peer_id: string | null;
  payload_size: number;
  payload_b64?: string;
}

export interface EventsListResponse {
  total: number;
  returned: number;
  events: EventRecord[];
}

export interface EventStats {
  total_events: number;
  total_bytes: number;
  oldest_timestamp: number | null;
  newest_timestamp: number | null;
  per_type: Record<string, number>;
  per_source_peer: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Halo capsule / Neural Imprint artifact distribution
// ---------------------------------------------------------------------------

export interface NeuralImprintArtifactSource {
  valid: boolean;
  artifact_id?: string;
  capsule_id?: string;
  transfer_id_hint?: string;
  artifact_dir: string;
  artifact_path: string;
  sidecar_path: string;
  base_model_id?: string;
  model_architecture?: string;
  model_family?: string;
  hidden_size?: number;
  layer_count?: number;
  prefix_token_count?: number;
  tool_schema_sha256?: string;
  profile_body_sha256?: string;
  cache_backend?: string;
  cache_backend_version?: string;
  cache_topology_sha256?: string;
  neural_imprint_sha256?: string | null;
  artifact_sha256?: string;
  total_bytes?: number;
  file_count?: number;
  created_at?: number | string | null;
  mtime?: number | null;
  error?: string;
}

export interface NeuralImprintArtifactRegistryResponse {
  ok: boolean;
  schema_version: string;
  roots: string[];
  artifact_count: number;
  artifacts: NeuralImprintArtifactSource[];
}

export interface PersonaSourceLatestResponse {
  ok: boolean;
  receipt: {
    peer_id: string;
    source_id: string;
    source_kind: string;
    base_model_id: string;
    tool_schema_sha256: string;
    profile_body_sha256?: string | null;
    rpp_run_id?: string | null;
    received_at_ms?: number | null;
  };
  payload: Record<string, unknown>;
}

export interface NeuralImprintGenerateRequest {
  peer_id: string;
  model_dir: string;
  model_id?: string;
  validate_restore?: boolean;
}

export interface NeuralImprintGenerationJob {
  schema_version: string;
  job_id: string;
  status: 'queued' | 'running' | 'succeeded' | 'failed' | string;
  peer_id: string;
  source_id: string;
  model_dir: string;
  model_id: string;
  output_dir: string;
  validate_restore: boolean;
  created_at: number;
  updated_at: number;
  started_at?: number | null;
  completed_at?: number | null;
  result?: {
    schema_version: string;
    artifact_dir: string;
    artifact_path: string;
    metadata_path: string;
    receipt_path: string;
    peer_id: string;
    source_id: string;
    model_id: string;
    prefix_token_count: number;
    artifact_sha256: string;
    tool_schema_sha256: string;
    profile_body_sha256: string;
  } | null;
  error?: {
    code: string;
    message: string;
    retryable?: boolean;
    details?: Record<string, unknown>;
  } | null;
}

export interface NeuralImprintGenerateJobResponse {
  ok: boolean;
  job: NeuralImprintGenerationJob;
}


export interface HaloCapsulePushRequest {
  peer_id: string;
  neural_imprint_dir?: string;
  artifact_id?: string;
  min_runtime_version?: string;
  transfer_id?: string;
  capsule_id?: string;
  chunk_size?: number;
}

export interface HaloCapsulePushResponse {
  ok: boolean;
  peer_id: string;
  transfer_id: string;
  capsule_id: string;
  base_model_id: string;
  min_runtime_version: string;
  artifact_sha256: string;
  artifact_file_count: number;
  frame_count: number;
  payload_bytes: number;
  chunk_size: number;
}

export interface HaloCapsuleApplyStatusReceipt {
  schema_version: string;
  peer_id: string;
  source: string;
  received_at: number;
  apply_status_sha256: string;
  transfer_id: string;
  capsule_id: string;
  status: 'received' | 'applied' | 'failed' | string;
  artifact_sha256?: string | null;
  canonical_sha256?: string | null;
  runtime_version?: string | null;
  prefix_token_count?: number | null;
  applied_at_unix_seconds?: number | null;
  error_code?: string | null;
}

export interface HaloCapsuleTransferAckReceipt {
  schema_version: string;
  peer_id: string;
  source: string;
  received_at: number;
  ack_sha256: string;
  ack_kind: string;
  transfer_id: string;
  accepted: boolean;
  reason?: string | null;
  canonical_sha256?: string | null;
}

export interface HaloCapsuleApplyStatusLatestResponse {
  ok: boolean;
  receipt: HaloCapsuleApplyStatusReceipt;
  payload: Record<string, unknown>;
}

export interface HaloCapsuleCoordinatorAction {
  kind: string;
  can_push: boolean;
  requires_user_confirmation: boolean;
  label: string;
  reasons: string[];
  push_request?: HaloCapsulePushRequest;
}

export interface HaloCapsuleCoordinatorPlanResponse {
  ok: boolean;
  schema_version: string;
  peer_id: string;
  connected: boolean;
  action: HaloCapsuleCoordinatorAction;
  snapshot_sha256?: string | null;
  lifecycle?: Record<string, unknown> | null;
  selected_model_id?: string | null;
  load_state?: string | null;
  data_readiness?: string | null;
  learning?: Record<string, unknown> | null;
  artifact_count?: number | null;
  matched_artifact?: NeuralImprintArtifactSource | null;
  last_apply_status?: HaloCapsuleApplyStatusReceipt | null;
  last_transfer_ack?: HaloCapsuleTransferAckReceipt | null;
}

export interface HaloCapsuleAutomationPreviewEntry {
  peer_id: string;
  display_name: string;
  connected: boolean;
  would_push: boolean;
  action: HaloCapsuleCoordinatorAction;
  plan: HaloCapsuleCoordinatorPlanResponse;
}

export interface HaloCapsuleAutomationPreviewResponse {
  ok: boolean;
  schema_version: string;
  dry_run: boolean;
  peer_count: number;
  candidate_count: number;
  skipped_revoked_count: number;
  entries: HaloCapsuleAutomationPreviewEntry[];
}

export interface HaloCapsuleAutomationRunRequest {
  dry_run?: boolean;
  peer_ids?: string[];
  max_pushes?: number;
  chunk_size?: number;
}

export interface HaloCapsuleAutomationRunResult {
  peer_id: string;
  display_name?: string | null;
  dry_run: boolean;
  would_push: boolean;
  action_kind: string;
  status: 'candidate' | 'skipped' | 'pushed' | 'failed' | string;
  push?: HaloCapsulePushResponse | null;
  error?: string | null;
}

export interface HaloCapsuleAutomationRunResponse {
  ok: boolean;
  dry_run: boolean;
  attempted_count: number;
  pushed_count: number;
  preview: HaloCapsuleAutomationPreviewResponse;
  results: HaloCapsuleAutomationRunResult[];
}

export interface JointInferenceHistoryItem {
  schema_version: string;
  request_id: string;
  conversation_id?: string | null;
  latest_request_id?: string | null;
  parent_request_id?: string | null;
  source?: string | null;
  peer_id: string;
  status: string;
  accepted_at: number;
  completed_at?: number | null;
  duration_seconds?: number | null;
  model_id?: string | null;
  max_tokens?: number | null;
  temperature?: number | null;
  enable_thinking?: boolean | null;
  use_neural_imprint?: boolean | null;
  neural_imprint_artifact_id?: string | null;
  neural_imprint_prefix_token_count?: number | null;
  route_reason?: string | null;
  messages_count?: number | null;
  messages?: Array<{ role: string; content: string }>;
  prompt_preview?: string | null;
  output_preview?: string | null;
  full_text?: string | null;
  error?: string | null;
  total_tokens?: number | null;
  tokens_per_sec?: number | null;
  last_event_type?: string | null;
  last_sequence?: number | null;
  token_events?: number | null;
}

export interface JointInferenceHistoryResponse {
  ok: boolean;
  schema_version: string;
  count: number;
  items: JointInferenceHistoryItem[];
}

export interface JointInferenceHistoryDetailResponse {
  ok: boolean;
  item: JointInferenceHistoryItem;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const getMeshStatus = () =>
  client.get<MeshStatus>('/mesh/status').then((r) => r.data);

export const listDevices = () =>
  client.get<DevicesListResponse>('/mesh/devices').then((r) => r.data);

export const createPairing = () =>
  client.post<QRPairingResponse>('/mesh/pair/qr').then((r) => r.data);

export const revokePeer = (peerId: string) =>
  client.post<{ ok: boolean; peer_id: string }>(
    `/mesh/devices/${encodeURIComponent(peerId)}/revoke`
  ).then((r) => r.data);

export const deletePeer = (peerId: string) =>
  client.delete<{ ok: boolean; peer_id: string }>(
    `/mesh/devices/${encodeURIComponent(peerId)}`
  ).then((r) => r.data);

export const approvePairRequest = (nonce: string) =>
  client.post<{ ok: boolean; nonce: string }>(
    `/mesh/pair/approve/${encodeURIComponent(nonce)}`
  ).then((r) => r.data);

export const listEvents = (params?: {
  tags?: string[];
  app_id?: string;
  event_type?: string;
  source_peer_id?: string;
  since?: number;
  until?: number;
  limit?: number;
  offset?: number;
  include_payload?: boolean;
}) =>
  client
    .get<EventsListResponse>('/mesh/events', { params })
    .then((r) => r.data);

export const getEventStats = () =>
  client.get<EventStats>('/mesh/events/stats').then((r) => r.data);

export const getJointInferenceHistory = (params?: { peer_id?: string; limit?: number }) =>
  client
    .get<JointInferenceHistoryResponse>('/mesh/joint_inference/history', { params })
    .then((r) => r.data);

export const getJointInferenceHistoryItem = (requestId: string) =>
  client
    .get<JointInferenceHistoryDetailResponse>(`/mesh/joint_inference/history/${requestId}`)
    .then((r) => r.data);

export const deleteJointInferenceHistoryItem = (requestId: string) =>
  client
    .delete<{ ok: boolean; request_id: string }>(`/mesh/joint_inference/history/${encodeURIComponent(requestId)}`)
    .then((r) => r.data);

export type JointInferenceStreamEvent = {
  schema_version: string;
  request_id: string;
  type: 'accepted' | 'status' | 'token' | 'complete' | 'error' | 'cancelled';
  sequence?: number;
  message?: string;
  token?: string;
  full_text?: string;
  total_tokens?: number;
  tokens_per_sec?: number;
  error?: string;
  use_neural_imprint?: boolean;
  neural_imprint_artifact_id?: string;
  neural_imprint_prefix_token_count?: number;
};

export async function streamJointInferenceContinue(
  requestId: string,
  payload: {
    messages?: Array<{ role: string; content: string }>;
    message?: string;
    max_tokens?: number;
    temperature?: number;
    enable_thinking?: boolean;
    use_neural_imprint?: boolean;
  },
  onEvent: (event: JointInferenceStreamEvent) => void,
) {
  const response = await fetch(`/api/mesh/joint_inference/history/${requestId}/continue`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  if (!response.body) {
    throw new Error('Streaming response body is unavailable');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      onEvent(JSON.parse(trimmed) as JointInferenceStreamEvent);
    }
  }
  buffer += decoder.decode();
  const trimmed = buffer.trim();
  if (trimmed) {
    onEvent(JSON.parse(trimmed) as JointInferenceStreamEvent);
  }
}

export const listNeuralImprintArtifacts = (includeInvalid = false) =>
  client
    .get<NeuralImprintArtifactRegistryResponse>('/neural_imprint/artifacts', {
      params: { include_invalid: includeInvalid },
    })
    .then((r) => r.data);

export const getLatestPersonaSource = (peerId: string) =>
  client
    .get<PersonaSourceLatestResponse>(
      `/neural_imprint/sources/${encodeURIComponent(peerId)}/latest`,
    )
    .then((r) => r.data);

export const generateNeuralImprint = (request: NeuralImprintGenerateRequest) =>
  client.post<NeuralImprintGenerateJobResponse>('/neural_imprint/generate', request).then((r) => r.data);

export const getNeuralImprintGenerationJob = (jobId: string) =>
  client
    .get<NeuralImprintGenerateJobResponse>(`/neural_imprint/generate/${encodeURIComponent(jobId)}`)
    .then((r) => r.data);


export const pushHaloCapsule = (request: HaloCapsulePushRequest) =>
  client.post<HaloCapsulePushResponse>('/mesh/halo_capsules/push', request).then((r) => r.data);

export const getHaloCapsuleApplyStatus = (peerId: string) =>
  client
    .get<HaloCapsuleApplyStatusLatestResponse>(
      `/mesh/devices/${encodeURIComponent(peerId)}/halo_capsules/apply_status/latest`,
    )
    .then((r) => r.data);

export const getHaloCapsulePlan = (peerId: string) =>
  client
    .get<HaloCapsuleCoordinatorPlanResponse>(
      `/mesh/devices/${encodeURIComponent(peerId)}/halo_capsules/plan`,
    )
    .then((r) => r.data);

export const getHaloCapsuleAutomationPreview = () =>
  client
    .get<HaloCapsuleAutomationPreviewResponse>('/mesh/halo_capsules/automation/preview')
    .then((r) => r.data);

export const runHaloCapsuleAutomation = (request: HaloCapsuleAutomationRunRequest) =>
  client
    .post<HaloCapsuleAutomationRunResponse>('/mesh/halo_capsules/automation/run', request)
    .then((r) => r.data);
