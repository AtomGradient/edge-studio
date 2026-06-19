// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import client from './client';
import type {
  ArchNode,
  BrowseResponse,
  DeviceProfile,
  DtypeSummary,
  EdgeRuntimeExportResult,
  ModelInfo,
  PipelineRunResult,
  PruningTrace,
  SystemInfo,
  TensorStats,
  WeightStatsResponse,
  ProfileSummary,
  ActivationHeatmapData,
  PruneSimResponse,
  ThresholdSweepPoint,
  TraceResponse,
  AttentionAnalysisResponse,
  KVReportResponse,
  MOEAnalysisResponse,
  OptimizationReport,
  SearchResult,
  TaskStatus,
  NeuralImprintInspectResponse,
  NeuralImprintRuntimeStatusResponse,
  RPPResultsInspectResponse,
  ALibraryGenerateFromSuggestionsRequest,
  ALibraryHistoryResponse,
  ALibraryInspectResponse,
  ALibraryGenerateRequest,
  ALibraryRefineDomainDescriptionRequest,
  ALibraryRefineDomainDescriptionResponse,
  ALibrarySuggestDirectionsRequest,
  ALibrarySuggestDirectionsResponse,
  ALibraryValidateYamlRequest,
  ALibraryValidateYamlResponse,
  ALibrarySelectionResponse,
} from './types';

// ---- Filesystem ----

export const getHome = () =>
  client.get<{ path: string }>('/fs/home').then(r => r.data);

export const browsePath = (path?: string) =>
  client.get<BrowseResponse>('/fs/browse', { params: path ? { path } : {} }).then(r => r.data);

// ---- Model ----

export const loadModel = (model_dir: string) =>
  client.post<ModelInfo>('/model/load', { model_dir }).then(r => r.data);

export const getModelInfo = (modelId: string) =>
  client.get<ModelInfo>(`/model/${modelId}/info`).then(r => r.data);

export const unloadModel = (modelId: string) =>
  client.delete(`/model/${modelId}`).then(r => r.data);

export const listLoadedModels = () =>
  client.get<ModelInfo[]>('/model/loaded').then(r => r.data);

export interface SessionSummary {
  has_trace: boolean;
  has_profile: boolean;
  has_ppl: boolean;
  has_report: boolean;
  has_generation: boolean;
  has_pipeline: boolean;
}

export const getSession = (modelId: string) =>
  client.get<SessionSummary>(`/model/${modelId}/session`).then(r => r.data);

// ---- Architecture ----

export const getArchitecture = (modelId: string) =>
  client.get<ArchNode>(`/model/${modelId}/architecture`).then(r => r.data);

export const getPruningTraces = (modelId: string) =>
  client.get<PruningTrace[]>(`/model/${modelId}/pruning-traces`).then(r => r.data);

// ---- Weights ----

export const getWeightStats = (modelId: string) =>
  client.get<WeightStatsResponse>(`/model/${modelId}/weight-stats`).then(r => r.data);

export const getDtypeBreakdown = (modelId: string) =>
  client.get<{ breakdown: DtypeSummary[] }>(`/model/${modelId}/weight-stats/dtype`).then(r => r.data);

export const getTensorFullStats = (modelId: string, tensorName: string) =>
  client.post<TensorStats>(`/model/${modelId}/tensor/full-stats`, { tensor_name: tensorName }).then(r => r.data);

// ---- Activation Profile ----

export const listProfiles = (modelId: string) =>
  client.get<{ profiles: string[] }>(`/model/${modelId}/profiles`).then(r => r.data);

export const loadProfile = (modelId: string, profilePath: string) =>
  client.post<ProfileSummary>(`/model/${modelId}/profile/load`, { profile_path: profilePath }).then(r => r.data);

export const generateProfile = (modelId: string, numRuns: number) =>
  client.post<{ task_id: string }>(`/model/${modelId}/profile/generate`, { num_runs: numRuns }).then(r => r.data);

export const getHeatmapData = (modelId: string, threshold: number = 0.1) =>
  client.get<ActivationHeatmapData>(`/model/${modelId}/activation/heatmap`, { params: { threshold } }).then(r => r.data);

// ---- Pruning Simulation ----

export const simulatePruning = (modelId: string, params: {
  threshold?: number;
  max_reduction?: number;
  min_intermediate?: number;
  protected_layers?: number[];
}) =>
  client.post<PruneSimResponse>(`/model/${modelId}/pruning/simulate`, params).then(r => r.data);

export const thresholdSweep = (modelId: string, params: {
  max_reduction?: number;
  min_intermediate?: number;
  protected_layers?: number[];
}) =>
  client.post<ThresholdSweepPoint[]>(`/model/${modelId}/pruning/sweep`, params).then(r => r.data);

// ---- Inference ----

export const runTrace = (modelId: string, params: {
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  enable_thinking?: boolean;
  enable_timing?: boolean;
  capture_attention?: boolean;
  capture_moe_routing?: boolean;
  use_legacy_tracer?: boolean;
  image_b64?: string;
}) =>
  client.post<{ task_id: string }>(`/model/${modelId}/trace`, params).then(r => r.data);

export const getTraceResult = (modelId: string) =>
  client.get<TraceResponse>(`/model/${modelId}/trace/result`).then(r => r.data);

// ---- Quality ----

export const getCachedQuality = (modelId: string) =>
  client.get<Record<string, unknown>>(`/model/${modelId}/quality/cached`).then(r => r.data);

export const computePPL = (modelId: string, text: string) =>
  client.post<{ task_id: string }>(`/model/${modelId}/quality/ppl`, { text }).then(r => r.data);

export const runGeneration = (modelId: string, prompts: string[], maxTokens: number = 50, enableThinking: boolean = false) =>
  client.post<{ task_id: string }>(`/model/${modelId}/quality/generate`, {
    prompts, max_tokens: maxTokens, enable_thinking: enableThinking,
  }).then(r => r.data);

export const runFullReport = (modelId: string, maxTokens: number = 50, enableThinking: boolean = false) =>
  client.post<{ task_id: string }>(`/model/${modelId}/quality/report`, {
    prompts: [], max_tokens: maxTokens, enable_thinking: enableThinking,
  }).then(r => r.data);

// ---- KV Cache ----

export const getKVReport = (modelId: string, devices: string[]) =>
  client.post<KVReportResponse>(`/model/${modelId}/kv-report`, { devices }).then(r => r.data);

// ---- Optimization ----

export const getOptSuggestions = (modelId: string) =>
  client.post<OptimizationReport>(`/model/${modelId}/optimize/suggestions`).then(r => r.data);

export const executeOptimization = (modelId: string, category: string, params: Record<string, unknown>) =>
  client.post<{ task_id: string }>(`/model/${modelId}/optimize/execute`, { category, params }).then(r => r.data);

// ---- Pipeline ----

export const runPipeline = (modelId: string, steps: Array<{ operation: string; params: Record<string, unknown> }>, pplText: string = '', skipValidation: boolean = false) =>
  client.post<{ task_id: string }>(`/model/${modelId}/pipeline/run`, {
    steps, ppl_text: pplText, skip_validation: skipValidation,
  }).then(r => r.data);

export const getPipelineResult = (modelId: string) =>
  client.get<PipelineRunResult>(`/model/${modelId}/pipeline/result`).then(r => r.data);

// ---- Auto Optimizer ----

export const searchOptimizations = (modelId: string, params: {
  device_name: string;
  quality_floor?: number;
  target_bits?: number[];
  max_layers_remove?: number;
}) =>
  client.post<SearchResult>(`/model/${modelId}/auto-optimize/search`, params).then(r => r.data);

// ---- Attention ----

export const analyzeAttention = (modelId: string) =>
  client.post<AttentionAnalysisResponse>(`/model/${modelId}/attention/analyze`).then(r => r.data);

// ---- MOE ----

export const analyzeMOE = (modelId: string) =>
  client.post<MOEAnalysisResponse>(`/model/${modelId}/moe/analyze`).then(r => r.data);

// ---- Comparison ----

export const compareModels = (modelIdA: string, modelIdB: string, prompt: string, maxTokens: number = 50) =>
  client.post<{ task_id: string }>('/compare', {
    model_id_a: modelIdA,
    model_id_b: modelIdB,
    prompt,
    max_tokens: maxTokens,
  }).then(r => r.data);

// ---- System Info ----

export const getSystemInfo = () =>
  client.get<SystemInfo>('/system-info').then(r => r.data);

// ---- Devices ----

export const getDevices = () =>
  client.get<DeviceProfile[]>('/devices').then(r => r.data);

// ---- Export ----

export const exportGGUF = (modelId: string, quantType: string = 'q4_k_m') =>
  client.post<{ task_id: string }>(`/model/${modelId}/export/gguf`, { quant_type: quantType }).then(r => r.data);

export const exportCoreML = (modelId: string, computeUnits: string = 'ALL', maxSeqLength: number = 512) =>
  client.post<{ task_id: string }>(`/model/${modelId}/export/coreml`, {
    compute_units: computeUnits,
    max_seq_length: maxSeqLength,
  }).then(r => r.data);

export const generateSwift = (modelId: string, packageName: string, defaultMaxTokens: number = 256) =>
  client.post<{ code: string; filename: string }>(`/model/${modelId}/export/swift`, {
    package_name: packageName,
    default_max_tokens: defaultMaxTokens,
  }).then(r => r.data);

export const generateEdgeRuntime = (modelId: string, optimizedDir?: string) =>
  client.post<EdgeRuntimeExportResult>(`/model/${modelId}/export/edge-runtime`, {
    optimized_dir: optimizedDir ?? '',
  }).then(r => r.data);

export const exportScaffoldZip = (
  modelId: string, appName: string, systemPrompt: string, modelTier: string,
  enableH2O: boolean = true, h2oBudget: number | null = null,
  directionSetId: string | null = null
) =>
  client.post<{ task_id: string }>(`/model/${modelId}/export/scaffold-zip`, {
    app_name: appName,
    system_prompt: systemPrompt,
    model_tier: modelTier,
    enable_dsr: enableH2O,
    dsr_budget: h2oBudget,
    direction_set_id: directionSetId,
  }).then(r => r.data);

export const downloadScaffoldZip = (zipPath: string) => {
  window.open(`/api/model/export/scaffold-zip/download?path=${encodeURIComponent(zipPath)}`, '_blank');
};

// ---- HuggingFace ----

export interface HFModel {
  id: string;
  author: string | null;
  downloads: number;
  likes: number;
  tags: string[];
  pipeline_tag: string | null;
  last_modified: string | null;
}

export interface LocalModel {
  name: string;
  path: string;
  size_bytes?: number;
}

export const searchHFModels = (query: string, limit: number = 20) =>
  client.get<{ models: HFModel[] }>('/hf/search', { params: { query, limit } }).then(r => r.data);

export const listLocalModels = () =>
  client.get<{ models: LocalModel[] }>('/hf/local').then(r => r.data);

export interface PathCheckResult {
  exists: boolean;
  complete: boolean;
  has_config: boolean;
  has_weights: boolean;
  size_bytes: number;
  path?: string;
}

export const checkLocalPath = (path: string) =>
  client.get<PathCheckResult>('/hf/check-path', { params: { path } }).then(r => r.data);

export const deleteLocalModel = (path: string) =>
  client.delete<{ status: string; path: string; freed_bytes: number }>('/hf/local', { data: { path } }).then(r => r.data);

export const downloadHFModel = (repo_id: string, download_dir?: string, mirror?: string) =>
  client.post<{ task_id: string }>('/hf/download', { repo_id, download_dir, mirror }).then(r => r.data);

export const probeHFNetwork = () =>
  client.get<{ reachable: boolean; latency_ms: number; suggestion?: string }>('/hf/probe').then(r => r.data);

// ---- Distillation ----

export const startDistillation = (params: {
  teacher_dir: string;
  student_dir: string;
  dataset_path: string;
  mode?: string;
  num_epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  temperature?: number;
  alpha?: number;
  max_samples?: number;
  output_dir?: string;
}) =>
  client.post<{ task_id: string }>('/model/distill', params).then(r => r.data);

// ---- Model Merge ----

export const startMerge = (params: {
  model_dirs: string[];
  strategy?: string;
  weights?: number[];
  base_model_dir?: string;
  density?: number;
  output_dir?: string;
}) =>
  client.post<{ task_id: string }>('/model/merge', params).then(r => r.data);

// ---- Auto Tune ----

export const startAutoTune = (params: {
  model_dir: string;
  device_name?: string;
  max_tokens?: number;
  num_runs?: number;
  search_temperatures?: number[];
  search_kv_cache_sizes?: number[];
  force_rerun?: boolean;
}) =>
  client.post<{ task_id: string }>('/model/auto-tune', params).then(r => r.data);

// ---- Benchmark ----

export const runBenchmark = (baselineDir: string, compareDir?: string, numTokens: number = 100) =>
  client.post<{ task_id: string }>('/benchmark/run', {
    baseline_dir: baselineDir,
    compare_dir: compareDir ?? null,
    num_tokens: numTokens,
  }).then(r => r.data);

// ---- Neural Imprint artifacts ----

export const parseNeuralImprintArtifact = (params: {
  path: string;
  sidecar_path?: string;
  current_model_id?: string;
  current_model_hash?: string;
  tokenizer_hash?: string;
  tool_schema_hash?: string;
  cache_topology_sha256?: string;
}) =>
  client.get<NeuralImprintInspectResponse>('/neural_imprint/parse', {
    params: Object.fromEntries(
      Object.entries(params).filter(([, value]) => value !== undefined && value !== ''),
    ),
  }).then(r => r.data);

export const getNeuralImprintRuntimeStatus = (modelId?: string) =>
  client.get<NeuralImprintRuntimeStatusResponse>('/neural_imprint/status', {
    params: modelId ? { model_id: modelId } : undefined,
  }).then(r => r.data);

export const restoreNeuralImprint = (params: {
  model_id: string;
  artifact_id?: string;
  artifact_path?: string;
  sidecar_path?: string;
}) =>
  client.post<NeuralImprintRuntimeStatusResponse>('/neural_imprint/restore', params).then(r => r.data);

export const unloadNeuralImprint = (modelId?: string) =>
  client.post<NeuralImprintRuntimeStatusResponse>('/neural_imprint/unload', {
    model_id: modelId ?? null,
  }).then(r => r.data);


export const inspectLatestRPPArtifacts = (peerId: string) =>
  client.get<RPPResultsInspectResponse>('/personal/rpp/artifacts/latest/inspect', {
    params: { peer_id: peerId },
  }).then(r => r.data);

export const inspectALibrary = (path: string) =>
  client.get<ALibraryInspectResponse>('/a_library/inspect', {
    params: { path },
  }).then(r => r.data);

export const listALibraryHistory = (limit: number = 50) =>
  client.get<ALibraryHistoryResponse>('/a_library/history', { params: { limit } }).then(r => r.data);

export const selectALibrary = (params: { model_id?: string; model_path?: string; direction_set_id?: string }) =>
  client.get<ALibrarySelectionResponse>('/a_library/select', { params }).then(r => r.data);

export const generateALibrary = (params: ALibraryGenerateRequest) =>
  client.post<{ task_id: string }>('/a_library/generate', params).then(r => r.data);

export const generateALibraryFromSuggestions = (params: ALibraryGenerateFromSuggestionsRequest) =>
  client.post<{ task_id: string }>('/a_library/generate_from_suggestions', params).then(r => r.data);

export const validateALibraryYaml = (params: ALibraryValidateYamlRequest) =>
  client.post<ALibraryValidateYamlResponse>('/a_library/validate_yaml', params).then(r => r.data);

export const suggestALibraryDirections = (params: ALibrarySuggestDirectionsRequest) =>
  client.post<ALibrarySuggestDirectionsResponse>('/a_library/suggest_directions', params).then(r => r.data);

export const refineALibraryDomainDescription = (params: ALibraryRefineDomainDescriptionRequest) =>
  client.post<ALibraryRefineDomainDescriptionResponse>('/a_library/refine_domain_description', params).then(r => r.data);

// ---- Task polling ----

export const getTaskStatus = (taskId: string) =>
  client.get<TaskStatus>(`/task/${taskId}`).then(r => r.data);

export const getTaskResult = (taskId: string) =>
  client.get<{ result: unknown }>(`/task/${taskId}/result`).then(r => r.data);

export const cancelTask = (taskId: string) =>
  client.delete<{ status: string }>(`/task/${taskId}`).then(r => r.data);

// ---- Terminal ----

export interface CreateTerminalParams {
  cols?: number;
  rows?: number;
  cmd?: string[];
  cwd?: string;
  env?: Record<string, string>;
}

export const createTerminal = (params: CreateTerminalParams = {}) =>
  client.post<{ session_id: string }>('/terminal/create', params).then(r => r.data);

export const closeTerminal = (sessionId: string) =>
  client.delete(`/terminal/${sessionId}`).then(r => r.data);

export const runTerminalCommand = (cmd: string[], cwd?: string, env?: Record<string, string>) =>
  client.post<{ session_id: string }>('/terminal/run', { cmd, cwd, env }).then(r => r.data);
