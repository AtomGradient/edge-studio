// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

// ---- Filesystem ----

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
}

export interface BrowseResponse {
  current_path: string;
  parent_path: string | null;
  entries: FileEntry[];
  has_config_json: boolean;
  has_safetensors: boolean;
  has_gguf: boolean;
}

// ---- Model ----

export interface QuantizationInfo {
  bits: number;
  group_size: number;
  mode: string | null;
  quantized_count: number;
  total_count: number;
}

export interface ModelInfo {
  model_id: string;
  model_type: string;
  model_name: string;
  model_dir: string;
  total_params: number;
  total_stored_params: number;
  total_size_bytes: number;
  tensor_count: number;
  quantization: QuantizationInfo;
  config: Record<string, unknown>;
  has_moe: boolean;
  supports_thinking: boolean;
  has_vision: boolean;
  num_layers: number;
  hidden_size: number;
  intermediate_size: number;
  num_attention_heads: number;
  num_kv_heads: number;
  source_format: string;
  is_gguf: boolean;
  model_category: string; // "llm" | "vlm" | "tts" | "stt"
}

// ---- Architecture ----

export interface ArchNode {
  name: string;
  node_type: string;
  weight_prefix: string;
  config_params: Record<string, unknown>;
  param_count: number;
  stored_param_count: number;
  size_bytes: number;
  children: ArchNode[];
  pruning_info: Record<string, unknown>;
  extra: Record<string, unknown>;
  total_param_count: number;
  total_stored_param_count: number;
  total_size_bytes: number;
  is_quantized: boolean;
}

export interface PruningTrace {
  category: string;
  description: string;
  details: Record<string, unknown>;
  severity: string;
}

// ---- Weights ----

export interface TensorMeta {
  name: string;
  dtype: string;
  shape: number[];
  num_elements: number;
  size_bytes: number;
  is_quantized: boolean;
  file_path: string;
}

export interface TensorStats {
  name: string;
  shape: number[];
  dtype: string;
  num_elements: number;
  size_bytes: number;
  min_val: number | null;
  max_val: number | null;
  mean_val: number | null;
  std_val: number | null;
  sparsity: number | null;
  histogram_counts: number[] | null;
  histogram_edges: number[] | null;
  is_quantized: boolean;
  quant_group_size: number | null;
  quant_bits: number | null;
}

export interface DtypeSummary {
  dtype: string;
  count: number;
  params: number;
  size: number;
}

export interface WeightStatsResponse {
  tensors: TensorMeta[];
  total_params: number;
  total_size: number;
  quantized_count: number;
}

// ---- Neural Imprint Inspector ----

export interface NeuralImprintTensorInfo {
  name: string;
  dtype?: string | null;
  shape: number[];
  data_offsets: number[];
  byte_count?: number | null;
}

export interface NeuralImprintHashEntry {
  name: string;
  value: string;
  source: string;
}

export interface NeuralImprintSummary {
  prefix_token_count?: number | null;
  model_id?: string | null;
  created_at?: string | number | null;
  profile_body?: unknown;
  tool_schema?: unknown;
  hashes: NeuralImprintHashEntry[];
}

export interface NeuralImprintCompatibilityCheck {
  name: string;
  expected?: string | null;
  actual?: string | null;
  matched?: boolean | null;
  reason?: string | null;
}

export interface NeuralImprintCompatibility {
  status: 'compatible' | 'incompatible' | 'unknown' | string;
  checks: NeuralImprintCompatibilityCheck[];
  message: string;
}

export interface NeuralImprintInspectResponse {
  ok: boolean;
  artifact_path?: string | null;
  artifact_name: string;
  artifact_size_bytes?: number | null;
  header_size_bytes: number;
  safetensors_metadata: Record<string, unknown>;
  tensor_count: number;
  tensors: NeuralImprintTensorInfo[];
  sidecar_found: boolean;
  sidecar_path?: string | null;
  sidecar_metadata: Record<string, unknown>;
  summary: NeuralImprintSummary;
  compatibility: NeuralImprintCompatibility;
}

export interface NeuralImprintRuntimeStatusResponse {
  ok: boolean;
  active: boolean;
  model_id?: string | null;
  model_dir?: string | null;
  artifact_id?: string | null;
  artifact_path?: string | null;
  sidecar_path?: string | null;
  prefix_token_count?: number | null;
  base_model_id?: string | null;
  model_architecture?: string | null;
  hidden_size?: number | null;
  layer_count?: number | null;
  loaded_at?: number | null;
}


// ---- RPP Results Inspector ----

export interface RPPArtifactInfo {
  name?: string | null;
  role?: string | null;
  stored: boolean;
  size_bytes?: number | null;
  sha256?: string | null;
}

export interface RPPDirectionSummary {
  direction_idx: number;
  direction_id: string;
  name: string;
  reason?: string | null;
  confidence?: number | null;
  bootstrap_pass?: boolean | null;
  mean_similarity?: number | null;
  std_similarity?: number | null;
  projection_stats?: Record<string, unknown>;
  top_positive_count?: number;
  top_negative_count?: number;
}

export interface RPPResultsInspectResponse {
  ok: boolean;
  schema_version: string;
  status: 'found' | 'missing' | string;
  peer_id: string;
  rpp_run_id?: string | null;
  received_at_ms?: number | null;
  storage_path?: string | null;
  summary: {
    peer_id?: string;
    rpp_run_id?: string;
    base_model_id?: string | null;
    layer_id?: number | null;
    a_version?: string | null;
    a_hash?: string | null;
    n_transactions?: number | null;
    k_selected?: number | null;
    direction_count?: number | null;
    total_elapsed_seconds?: number | null;
  };
  dataset_summary: Record<string, unknown>;
  profile: {
    name?: string | null;
    summary?: string | null;
    narrative?: string | null;
  };
  directions: RPPDirectionSummary[];
  b_directions_header?: {
    name?: string;
    header_size_bytes?: number;
    tensor_count?: number;
    metadata?: Record<string, unknown>;
    tensors?: Array<{
      name: string;
      dtype?: string | null;
      shape: number[];
      byte_count?: number | null;
    }>;
  } | null;
  artifacts: RPPArtifactInfo[];
  warnings: string[];
  audit?: Record<string, unknown>;
}

// ---- A-library Inspector ----

export interface ALibraryHealthReport {
  layer_idx?: number | null;
  layer_type?: string | null;
  library_kind?: string | null;
  model_family?: string | null;
  model_path?: string | null;
  pooling?: string | null;
  direction_set_id?: string | null;
  yaml_sha256?: string | null;
  source_type?: string | null;
  source_schema_version?: string | null;
  hidden_size?: number | null;
  n_directions?: number | null;
  n_sentences_total?: number | null;
  max_abs_cos_sim?: number | null;
  mean_abs_cos_sim?: number | null;
  threshold_max_lt?: number | null;
  threshold_mean_lt?: number | null;
  max_pass?: boolean | null;
  mean_pass?: boolean | null;
  worst_pair?: string[] | null;
  min_signal_strength?: number | null;
  median_signal_strength?: number | null;
  n_pass?: number | null;
  n_total?: number | null;
  signal_pass?: boolean | null;
  verdict?: 'pass' | 'fail' | string;
  report_path?: string | null;
  artifact_names?: string[];
}

export interface ALibraryArtifactInfo {
  name: string;
  path: string;
  kind: 'safetensors' | 'npz' | string;
  layer_idx?: number | null;
  size_bytes?: number | null;
  model_family?: string | null;
  hidden_size?: number | null;
  direction_count?: number | null;
  tensor_count?: number | null;
  dtype?: string | null;
  header_size_bytes?: number | null;
  metadata?: Record<string, unknown>;
  tensors_preview?: Array<{
    name: string;
    dtype?: string | null;
    shape?: Array<number | null>;
    byte_count?: number | null;
  }>;
}

export interface ALibrarySweepSummary {
  path: string;
  model_path?: string | null;
  model_family?: string | null;
  pooling?: string | null;
  layers_swept: number[];
  per_layer: ALibraryHealthReport[];
}

export interface ALibraryInspectResponse {
  ok: boolean;
  schema_version: string;
  status: 'found' | string;
  library_path: string;
  summary: {
    library_kind?: string | null;
    model_family?: string | null;
    hidden_size?: number | null;
    target_layer?: number | null;
    direction_set_id?: string | null;
    yaml_sha256?: string | null;
    source_type?: string | null;
    source_schema_version?: string | null;
    health_status?: 'pass' | 'fail' | 'unknown' | string;
    health_report_path?: string | null;
    selected_artifact?: string | null;
    selected_reason?: string | null;
    report_count?: number;
    artifact_count?: number;
    sweep_summary_count?: number;
    pooling?: string | null;
    model_path?: string | null;
    n_directions?: number | null;
  };
  manifest: {
    ready: boolean;
    required_keys: Record<string, unknown>;
    checks: Array<{
      name: string;
      present?: boolean;
      value?: unknown;
      passed?: boolean;
    }>;
  };
  health_reports: ALibraryHealthReport[];
  sweep_summaries: ALibrarySweepSummary[];
  artifacts: ALibraryArtifactInfo[];
  warnings: string[];
  audit?: Record<string, unknown>;
}

export interface ALibrarySelectionCandidate {
  library_id?: string;
  library_kind?: string;
  model_family?: string;
  hidden_size?: number;
  layer_count?: number;
  target_layer?: number;
  direction_set_id?: string;
  yaml_sha256?: string;
  source_type?: string;
  source_schema_version?: string;
  artifact?: string;
  artifact_sha256?: string;
  health_report?: string;
  health_verdict?: string;
  pooling?: string;
  n_directions?: number;
  source_manifest?: string;
  artifact_path?: string | null;
  health_report_path?: string | null;
  artifact_exists?: boolean;
  health_report_exists?: boolean;
  artifact_sha256_ok?: boolean | null;
  match_reasons?: string[];
}

export interface ALibrarySelectionResponse {
  ok: boolean;
  schema_version: string;
  status: 'matched' | 'missing' | string;
  model: {
    model_name?: string | null;
    model_dir?: string | null;
    model_family?: string | null;
    hidden_size?: number | null;
    layer_count?: number | null;
    is_moe?: boolean;
  };
  direction_set_id?: string | null;
  selected?: ALibrarySelectionCandidate | null;
  candidates: ALibrarySelectionCandidate[];
  reasons: string[];
  recommended_action: string;
}

export interface ALibraryGenerateRequest {
  model_path: string;
  yaml_path?: string | null;
  output_dir?: string | null;
  direction_set_id?: string | null;
  layers?: number[] | null;
  sweep?: boolean;
  pooling?: 'last_real' | 'mean';
  source_type?: 'host_model_seed' | 'claude_authored' | 'manual';
}

export interface ALibraryHistoryItem {
  path: string;
  root: string;
  model_name?: string | null;
  created_at_unix?: number | null;
  direction_set_id?: string | null;
  target_layer?: number | null;
  health_status?: string | null;
  health_verdict?: string | null;
  library_kind?: string | null;
  model_family?: string | null;
  hidden_size?: number | null;
  n_directions?: number | null;
  ready?: boolean;
  warnings?: string[];
}

export interface ALibraryHistoryResponse {
  ok: boolean;
  schema_version: string;
  roots: string[];
  items: ALibraryHistoryItem[];
}

export interface ALibraryValidateYamlRequest {
  content: string;
  direction_set_id?: string | null;
  persist?: boolean;
}

export interface ALibraryDirectionPreview {
  name: string;
  description?: string | null;
  domain: string;
  positive_count: number;
  negative_count: number;
}

export interface ALibraryValidateYamlResponse {
  ok: boolean;
  schema_version: string;
  direction_set_id: string;
  source_schema_version?: string | null;
  yaml_sha256: string;
  errors: Array<Record<string, unknown>>;
  coverage: {
    direction_count: number;
    positive_count: number;
    negative_count: number;
    sentence_count: number;
    domains: Record<string, number>;
    directions: ALibraryDirectionPreview[];
  };
  stored_path?: string | null;
  validation_report_path?: string | null;
}

export interface ALibraryDirectionSuggestion {
  name: string;
  description: string;
  domain: string;
  positive: string[];
  negative: string[];
}

export interface ALibraryDirectionRepairContext {
  worst_pairs?: string[][];
  max_abs_cos?: number | null;
  mean_abs_cos?: number | null;
  signal_pass?: boolean | null;
  validation_error_codes?: string[];
  prev_direction_set_id?: string | null;
  reason?: string | null;
}

export interface ALibrarySuggestDirectionsRequest {
  domain_description: string;
  target_count?: number;
  model_id?: string | null;
  repair_context?: ALibraryDirectionRepairContext | null;
}

export interface ALibrarySuggestDirectionsResponse {
  ok: boolean;
  schema_version: string;
  model_id: string;
  model_name: string;
  domain_description: string;
  target_count: number;
  directions: ALibraryDirectionSuggestion[];
  repair_strategy?: 'direct' | 'fix_unclosed_arrays' | 'repair_truncated' | string;
  repaired?: boolean;
  raw_output?: string;
  audit?: Record<string, unknown>;
}

export interface ALibraryGenerateFromSuggestionsRequest {
  model_path: string;
  direction_set_id: string;
  directions: ALibraryDirectionSuggestion[];
  output_dir?: string | null;
  layers?: number[] | null;
  sweep?: boolean;
  pooling?: 'last_real' | 'mean';
  source_type?: 'host_model_seed' | 'claude_authored' | 'manual';
}

export interface ALibraryRefineDomainDescriptionRequest {
  domain_description: string;
  model_id?: string | null;
}

export interface ALibraryRefineDomainDescriptionResponse {
  ok: boolean;
  schema_version: string;
  model_id: string;
  model_name: string;
  original_description: string;
  refined_description: string;
  audit?: Record<string, unknown>;
}

// ---- Activation ----

export interface ProfileSummary {
  intermediate_size: number;
  num_layers: number;
  run_count: number;
  total_dead_at_01: number;
  dead_ratio_at_01: number;
}

export interface ActivationHeatmapData {
  max_matrix: number[][];
  mean_matrix: number[][];
  num_layers: number;
  neurons_per_layer: number;
  dead_per_layer: number[];
  threshold: number;
}

// ---- Pruning Simulation ----

export interface LayerPruneResult {
  layer_idx: number;
  original_size: number;
  alive_count: number;
  aligned_size: number;
  removed: number;
  retention: number;
  is_protected: boolean;
}

export interface PruneSimResponse {
  layers: LayerPruneResult[];
  total_removed: number;
  total_original: number;
  retention: number;
  mlp_size_saved_bytes: number;
  mlp_params_saved: number;
  config_preview: number[];
}

// ---- Threshold Sweep ----

export interface ThresholdSweepPoint {
  threshold: number;
  retention: number;
  mlp_size_saved_mb: number;
  total_removed: number;
}

// ---- Inference Trace ----

export interface TopLogit {
  token_id: number;
  token_str: string;
  prob: number;
}

export interface StepData {
  step_idx: number;
  token_id: number;
  token_str: string;
  top_k_token_ids: number[];
  top_k_probs: number[];
  top_k_token_strs: string[];
  chosen_rank: number;
  chosen_prob: number;
  final_hidden_norm: number;
  layers: LayerTraceData[];
}

export interface LayerTraceData {
  layer_idx: number;
  attn_weights: number[][] | null; // [num_heads, seq_len]
  mlp_act_mean: number;
  mlp_act_max: number;
  mlp_act_top_indices: number[];
  mlp_act_top_values: number[];
  attn_residual_norm: number;
  mlp_residual_norm: number;
  norm_after_attn: number;
  norm_after_mlp: number;
  attn_latency_ms: number;
  mlp_latency_ms: number;
}

export interface TraceResponse {
  prompt: string;
  prompt_token_ids: number[];
  prompt_tokens: string[];
  temperature: number;
  top_k: number;
  top_p: number;
  model_dir: string;
  model_name: string;
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  steps: StepData[];
  generated_text: string;
  total_time_seconds: number;
  prefill_time_seconds: number;
  prefill_layer_traces: LayerTraceData[];
  enable_timing: boolean;
}

// ---- Attention Analysis ----

export interface HeadClassification {
  layer: number;
  head: number;
  pattern: string; // SINK / LOCAL / GLOBAL / SPARSE
  confidence: number;
}

export interface AttentionAnalysisResponse {
  classifications: HeadClassification[];
  pattern_matrix: string[][]; // [num_layers][num_heads]
  pattern_counts: Record<string, number>;
  per_layer_summary: Array<{
    layer: number;
    dominant: string;
    sink: number;
    local: number;
    global: number;
    sparse: number;
  }>;
  suggestions: Array<{
    title: string;
    description: string;
    priority?: string;
    category?: string;
  }>;
}

// ---- Quality Validation ----

export interface PerplexityResult {
  text: string;
  num_tokens: number;
  total_log_prob: number;
  perplexity: number;
  per_token_log_probs: number[];
  duration_seconds: number;
}

export interface GenerationSample {
  prompt: string;
  generated_text: string;
  num_tokens: number;
  avg_prob: number;
  tokens_per_second: number;
  duration_seconds: number;
}

export interface QualityReportResult {
  model_dir: string;
  model_name: string;
  perplexity?: PerplexityResult;
  perplexity_results: PerplexityResult[];
  generation_samples: GenerationSample[];
  avg_perplexity: number;
  total_duration_seconds: number;
}

// ---- KV Cache ----

export interface KVMemoryCurvePoint {
  seq_len: number;
  model_weights_mb: number;
  kv_cache_mb: number;
  activation_mb: number;
  overhead_mb: number;
  total_mb: number;
}

export interface KVDeviceCapacity {
  device_name: string;
  ram_gb: number;
  available_mb: number;
  fits: boolean;
  max_seq_len: number;
  kv_at_max_mb: number;
  headroom_mb: number;
}

export interface KVTraceStep {
  step: number;
  seq_len: number;
  kv_cache_bytes: number;
  token: string;
}

export interface DSRCurvePoint {
  seq_len: number;
  kv_cache_mb: number;
  total_mb: number;
}

export interface KVReportResponse {
  num_layers: number;
  num_kv_heads: number;
  head_dim: number;
  bytes_per_token: number;
  model_weights_mb: number;
  memory_curve: KVMemoryCurvePoint[];
  device_capacities: KVDeviceCapacity[];
  trace_steps: KVTraceStep[];
  dsr_curves: Record<string, DSRCurvePoint[]>;
}

// ---- MOE Analysis ----

export interface MOELayerStats {
  layer_idx: number;
  expert_counts: number[];
  expert_avg_scores: number[];
  load_balance: number;
  cold_experts: number[];
}

export interface MOEAnalysisResponse {
  num_experts: number;
  top_k: number;
  avg_load_balance: number;
  total_tokens: number;
  cold_expert_count: number;
  layer_stats: MOELayerStats[];
  cold_experts: Array<{ layer: number; expert: number }>;
  utilization_matrix: number[][];
  global_token_counts: number[];
  global_avg_scores: number[];
}

// ---- Optimization Advisor ----

export interface OptimizationSuggestion {
  category: string;
  priority: string; // 'high' | 'medium' | 'low'
  title: string;
  description: string;
  estimated_saving: string;
  risk_level: string; // 'low' | 'medium' | 'high'
  params: Record<string, unknown>;
  applicable: boolean;
}

export interface OptimizationReport {
  model_name: string;
  model_size_bytes: number;
  total_params: number;
  suggestions: OptimizationSuggestion[];
  requires_data: OptimizationSuggestion[];
  total_estimated_saving_bytes: number;
}

export interface ExecutionResult {
  operation: string;
  success: boolean;
  output_dir: string;
  message: string;
  duration_seconds: number;
  original_size_bytes: number;
  result_size_bytes: number;
  saving_bytes: number;
  details: Record<string, unknown>;
}

// ---- Auto Optimizer ----

export interface SearchCandidate {
  label?: string;
  threshold: number;
  bits: number;
  layers_removed: number;
  layers_removed_list: number[];
  estimated_size_gb: number;
  quality_proxy: number;
  neuron_retention: number;
  layer_retention: number;
  fits_device: boolean;
  is_pareto: boolean;
  per_layer_sizes: number[];
  speedup_pct?: number;
  size_reduction_pct?: number;
  quality_pct?: number;
}

export interface SearchResult {
  candidates: SearchCandidate[];
  pareto_frontier: SearchCandidate[];
  device_name: string;
  device_max_gb: number;
  model_name: string;
  search_time_seconds: number;
  total_combinations: number;
  fits_device_count: number;
}

// ---- Model Comparison ----

export interface ArchDiffRow {
  field_name: string;
  model_a_value: string;
  model_b_value: string;
  is_different: boolean;
}

export interface ArchDiff {
  model_a_name: string;
  model_b_name: string;
  rows: ArchDiffRow[];
}

export interface LatencyProfile {
  model_name: string;
  prefill_layer_attn_ms: number[];
  prefill_layer_mlp_ms: number[];
  prefill_total_ms: number;
  decode_layer_attn_ms: number[];
  decode_layer_mlp_ms: number[];
  decode_total_ms: number;
  decode_steps: number;
  tokens_per_second: number;
}

export interface BottleneckLayer {
  layer_idx: number;
  attn_ms: number;
  mlp_ms: number;
  total_ms: number;
  pct_of_total: number;
  bottleneck_type: string;
}

export interface ComparisonResult {
  arch_diff: ArchDiff | null;
  latency_a: LatencyProfile | null;
  latency_b: LatencyProfile | null;
  bottlenecks_a: BottleneckLayer[];
  bottlenecks_b: BottleneckLayer[];
}

// ---- Export ----

export interface ExportResult {
  success: boolean;
  output_path: string;
  output_size_bytes: number;
  duration_seconds: number;
  error_message: string;
}

export interface EdgeRuntimeExportResult {
  package_swift: string;
  main_swift: string;
  readme: string;
  run_command: string;
  model_name: string;
  is_optimized: boolean;
  optimization_summary: string;
}

// ---- Device ----

export interface DeviceProfile {
  name: string;
  category: string;
  ram_gb: number;
  available_ram_gb: number;
  neural_engine_tops: number;
  gpu_cores: number;
  chip: string;
  max_model_size_gb: number;
}

// ---- Pipeline ----

export interface PipelineStepResult {
  operation: string;
  success: boolean;
  output_dir: string;
  message: string;
  duration_seconds: number;
  original_size_bytes: number;
  result_size_bytes: number;
  saving_bytes: number;
  details: Record<string, unknown>;
}

export interface PPLResult {
  perplexity: number;
  num_tokens: number;
  duration_seconds: number;
}

export interface PipelineRunResult {
  steps: PipelineStepResult[];
  final_output_dir: string;
  original_size_bytes: number;
  optimized_size_bytes: number;
  optimized_model_id: string | null;
  optimized_model_info: ModelInfo | null;
  baseline_ppl: PPLResult | null;
  optimized_ppl: PPLResult | null;
  total_duration_seconds: number;
  success: boolean;
  error_message: string | null;
}

// ---- Benchmark ----

export interface BenchmarkModelResult {
  model_dir: string;
  disk_size_mb: number;
  memory_before_load_mb: number;
  memory_after_load_mb: number;
  peak_memory_mb: number;
  tokens_per_second: number;
  time_to_first_token_s: number;
  generation_tokens: number;
  perplexity: number;
  model_type: string;
  is_edge_optimized: boolean;
  optimization_summary: string;
  error: string;
}

export interface BenchmarkGuidance {
  verdict: 'success' | 'warning' | 'danger';
  title: string;
  message: string;
  ppl_change_pct: number;
}

export interface BenchmarkComparison {
  disk_reduction_pct: number;
  memory_reduction_pct: number;
  speed_improvement_pct: number;
  perplexity_delta: number;
  guidance: BenchmarkGuidance;
}

export interface BenchmarkRunResult {
  mode: 'single' | 'comparison';
  baseline: BenchmarkModelResult;
  optimized: BenchmarkModelResult | null;
  comparison: BenchmarkComparison | null;
}

// ---- Distillation ----

export interface DistillResult {
  success: boolean;
  output_dir: string;
  teacher_name: string;
  student_name: string;
  num_epochs: number;
  total_steps: number;
  final_loss: number;
  final_kl_loss: number;
  final_ce_loss: number;
  duration_seconds: number;
  dataset_samples: number;
  error: string;
  warning: string;
  loss_history: Array<{
    epoch: number;
    step: number;
    loss: number;
    kl_loss: number;
    ce_loss: number;
  }>;
}

// ---- Model Merge ----

export interface MergeResult {
  success: boolean;
  output_dir: string;
  strategy: string;
  model_names: string[];
  merged_params: number;
  merged_size_bytes: number;
  duration_seconds: number;
  error: string;
}

// ---- Auto Tune ----

export interface TuneCandidate {
  label?: string;
  batch_size: number;
  kv_cache_size: number;
  temperature: number;
  quantization: string;
  tokens_per_second: number;
  time_to_first_token_ms: number;
  peak_memory_mb: number;
  perplexity: number;
}

export interface AutoTuneResult {
  success: boolean;
  model_name: string;
  device_name: string;
  best: TuneCandidate | null;
  candidates?: TuneCandidate[];
  all_candidates: TuneCandidate[];
  search_time_seconds: number;
  total_configs_tested: number;
  cached: boolean;
  cache_path: string;
  error: string;
}

// ---- System Info ----

export interface SystemInfo {
  device_name: string;
  chip: string;
  total_memory_gb: number;
  available_memory_gb: number;
}

// ---- Task (long-running ops) ----

export interface TaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'complete' | 'error' | 'cancelled';
  progress: number;
  message: string;
  error: string | null;
}

export interface TaskEvent {
  type: 'progress' | 'complete' | 'error' | 'cancelled' | 'ping';
  message?: string;
  percent?: number;
  result?: unknown;
}

export interface ScaffoldZipExportResult {
  success: boolean;
  zip_path: string;
  zip_size_bytes: number;
  app_name: string;
  model_name: string;
  model_dir: string;
  model_tier: string;
  direction_set_id?: string | null;
  error: string;
}
