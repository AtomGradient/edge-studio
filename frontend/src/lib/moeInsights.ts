// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * moeInsights — derived expert-routing cohort + chat helpers for /moe
 * (MOEAnalyzer.tsx).
 *
 * The MOE analyzer turns a captured inference trace into per-(layer, expert)
 * token counts:
 *  - utilization_matrix[L][E] = how many top-k slots got routed to expert E
 *    on layer L across all observed tokens.
 *  - layer_stats[L].load_balance ≈ N · Σ fᵢ·pᵢ — 1.0 = perfectly balanced.
 *  - global_token_counts[E] = total across all layers.
 *
 * §9.1 N-cell cohort (L × E grid, 13th instance after attention's L × H).
 *   Per-layer entropy on expert_counts → diversity score; cross-layer
 *   intersection of top-utilization experts → "always-hot" cohort.
 *
 * §10.3 naming: capability uses `runPhase` not `status`.
 *
 *   surfaced inline (4-card identity strip + brain narrating gating)
 *   instead of returning EmptyState early.
 *
 *   measures within-layer diversity (max = log2(num_experts)); cross-layer
 *   entropy on dominant-expert distribution measures structural diversity.
 *
 * Sovereignty (§9.2 mandatory): expert routing is captured locally during
 * inference, analyzed locally; zero cloud calls.
 */
import type { ModelInfo, MOEAnalysisResponse } from '@/api/types';

type Locale = 'en' | 'zh';

export type MoePhase = 'noModel' | 'notMoe' | 'noTrace' | 'idle' | 'analyzing' | 'analyzed';

/** Expert utilization buckets (global). */
export type ExpertBucket = 'hot' | 'warm' | 'tepid' | 'cold';

/** Per-layer routing pattern. */
export type LayerPattern =
  | 'balanced'        // entropy ≥ 0.85 · log2(K), low CV
  | 'mild_skew'       // entropy 0.6-0.85, dispersed but skewed
  | 'monopolized'     // top-1 expert > 50% of layer's traffic
  | 'hot_cold_split'  // ≥ 30% experts cold AND a small set carrying ≥ 60%
  | 'sparse_sample';  // total routings < num_experts (sample is too small)

/** Macro-shape across the whole model. */
export type MoeShape =
  | 'balanced'        // most layers balanced
  | 'mostly_skew'     // most layers mild_skew
  | 'monopoly_heavy'  // ≥ 25% layers monopolized
  | 'hot_cold_heavy'  // ≥ 25% layers hot_cold_split
  | 'sample_starved'  // ≥ 50% layers sparse_sample (need longer trace)
  | 'mixed';

export interface MoeCapabilities {
  // ── Inputs ─────────────────────────────────────────────────────────────
  result: MOEAnalysisResponse | null;
  loading: boolean;
  hasTrace: boolean;
  brain: ModelInfo | null;

  // ── Gating ─────────────────────────────────────────────────────────────
  hasModel: boolean;
  hasMoeArch: boolean;
  hasResult: boolean;
  runPhase: MoePhase;

  // ── Architecture facts ─────────────────────────────────────────────────
  numExperts: number;
  topK: number;
  numLayers: number;
  totalSlots: number;        // numLayers · numExperts
  totalRoutings: number;     // total_tokens · top_k · num_layers (post-trace)
  totalTokens: number;       // tokens routed (prompt + decoded)

  // ── Cohort: per-expert (global) ────────────────────────────────────────
  /** Hot experts: top 5% by global token count. */
  hotExperts: number[];
  /** Cold experts: 0 routings globally. */
  coldExpertsGlobal: number[];
  /** Warm: 5-25% percentile. Tepid: 25-95%. */
  warmExperts: number[];
  tepidExperts: number[];
  /** Highest single-expert traffic count. */
  maxGlobalCount: number;
  /** Top-1 expert id (most utilized globally). */
  topExpertId: number | null;
  /** Top-1 fraction of total routings. */
  topExpertShare: number;
  /** Coefficient of variation across global_token_counts. */
  globalCV: number;

  // ── Cohort: per-layer ──────────────────────────────────────────────────
  /** Per-layer entropy normalized to [0, 1] (entropy / log2(K)). */
  layerEntropies: number[];
  /** Per-layer Pattern classification. */
  layerPatterns: LayerPattern[];
  /** Layers tagged 'monopolized'. */
  monopolizedLayers: number[];
  /** Layers tagged 'hot_cold_split'. */
  hotColdLayers: number[];
  /** Layers tagged 'balanced'. */
  balancedLayers: number[];
  /** Layers with sample too small (< num_experts routings). */
  sampleStarvedLayers: number[];
  /** Mean layer load_balance score (1 = perfect). */
  meanLoadBalance: number;
  /** Layer with worst load_balance. */
  worstLayerIdx: number | null;
  worstLayerBalance: number;

  // ── Cross-layer dominance ──────────────────────────────────────────────
  /** Experts dominating in ≥ N layers (top-1 in that layer). */
  alwaysHotExperts: number[];
  /** Diversity entropy on cross-layer dominant-expert distribution (bits). */
  layerDominanceDiversity: number;

  // ── Macro ──────────────────────────────────────────────────────────────
  shape: MoeShape;
  /** Fraction of experts with 0 global routings. */
  coldRatio: number;
}

// ── Math helpers ─────────────────────────────────────────────────────────

function shannonEntropy(counts: readonly number[]): number {
  let total = 0;
  for (const c of counts) if (c > 0) total += c;
  if (total === 0) return 0;
  let h = 0;
  for (const c of counts) {
    if (c <= 0) continue;
    const p = c / total;
    h -= p * Math.log2(p);
  }
  return h;
}

function meanAndStd(arr: readonly number[]): { mean: number; std: number } {
  if (arr.length === 0) return { mean: 0, std: 0 };
  let m = 0;
  for (const v of arr) m += v;
  m /= arr.length;
  let s2 = 0;
  for (const v of arr) s2 += (v - m) ** 2;
  return { mean: m, std: Math.sqrt(s2 / arr.length) };
}

function classifyLayer(
  counts: readonly number[],
  numExperts: number,
  topK: number,
  loadBalance: number,
  coldExperts: readonly number[],
): { pattern: LayerPattern; entropyNorm: number } {
  const total = counts.reduce((a, b) => a + b, 0);
  // Aggregated layer entropy can reach log2(num_experts) at full uniformity
  // (since different tokens can route to different K experts); normalize by
  // that not log2(topK).
  const maxEntropy = Math.log2(Math.max(numExperts, 2));
  const ent = shannonEntropy(counts);
  const entropyNorm = maxEntropy > 0 ? ent / maxEntropy : 0;

  // Sample is too small to draw conclusions when total routings < num_experts.
  if (total < numExperts) {
    return { pattern: 'sparse_sample', entropyNorm };
  }
  // Top-1 expert share within this layer
  const max = Math.max(...counts);
  const topShare = total > 0 ? max / total : 0;
  // Cold ratio within this layer
  const coldRatio = coldExperts.length / Math.max(numExperts, 1);
  // Top-K share: how concentrated is the top K of experts
  const sorted = [...counts].sort((a, b) => b - a);
  const topKShare = total > 0
    ? sorted.slice(0, topK).reduce((a, b) => a + b, 0) / total
    : 0;

  if (topShare > 0.5) {
    return { pattern: 'monopolized', entropyNorm };
  }
  if (coldRatio > 0.3 && topKShare > 0.6) {
    return { pattern: 'hot_cold_split', entropyNorm };
  }
  // Use load_balance as a corroborating signal: > 0.85 of perfect = balanced.
  if (entropyNorm >= 0.85 && loadBalance >= 0.7) {
    return { pattern: 'balanced', entropyNorm };
  }
  return { pattern: 'mild_skew', entropyNorm };
}

function detectShape(
  patterns: readonly LayerPattern[],
): MoeShape {
  if (patterns.length === 0) return 'mixed';
  const counts: Record<LayerPattern, number> = {
    balanced: 0, mild_skew: 0, monopolized: 0, hot_cold_split: 0, sparse_sample: 0,
  };
  for (const p of patterns) counts[p] += 1;
  const n = patterns.length;
  if (counts.sparse_sample / n >= 0.5) return 'sample_starved';
  if (counts.monopolized / n >= 0.25) return 'monopoly_heavy';
  if (counts.hot_cold_split / n >= 0.25) return 'hot_cold_heavy';
  if (counts.balanced / n >= 0.6) return 'balanced';
  if (counts.mild_skew / n >= 0.6) return 'mostly_skew';
  return 'mixed';
}

// ── Main capability deriver ──────────────────────────────────────────────

export function deriveMoeCapabilities(
  result: MOEAnalysisResponse | null,
  loading: boolean,
  hasTrace: boolean,
  brain: ModelInfo | null,
): MoeCapabilities {
  const hasModel = !!brain;
  const hasMoeArch = !!brain?.has_moe;
  const hasResult = !!result;

  let runPhase: MoePhase;
  if (!hasModel) runPhase = 'noModel';
  else if (!hasMoeArch) runPhase = 'notMoe';
  else if (!hasTrace) runPhase = 'noTrace';
  else if (loading) runPhase = 'analyzing';
  else if (hasResult) runPhase = 'analyzed';
  else runPhase = 'idle';

  const numExperts = result?.num_experts ?? 0;
  const topK = result?.top_k ?? 0;
  const numLayers = result?.layer_stats?.length ?? brain?.num_layers ?? 0;
  const totalSlots = numLayers * numExperts;
  const totalTokens = result?.total_tokens ?? 0;
  const totalRoutings = totalTokens * topK * numLayers;

  // ── Per-expert global cohort ─────────────────────────────────────────
  const counts = result?.global_token_counts ?? [];
  const sortedDesc = counts.length > 0
    ? counts.map((c, i) => [c, i] as [number, number]).sort((a, b) => b[0] - a[0])
    : [];
  const totalGlobal = counts.reduce((a, b) => a + b, 0);
  const maxGlobalCount = counts.length > 0 ? Math.max(...counts) : 0;
  const topExpertId = sortedDesc.length > 0 ? sortedDesc[0][1] : null;
  const topExpertShare = totalGlobal > 0 && topExpertId !== null
    ? counts[topExpertId] / totalGlobal
    : 0;

  // Buckets via percentile cutoffs on the descending sorted list.
  const hotCutIdx = Math.max(1, Math.floor(numExperts * 0.05));
  const warmCutIdx = Math.max(1, Math.floor(numExperts * 0.25));
  const tepidCutIdx = Math.max(1, Math.floor(numExperts * 0.95));
  const hotExperts: number[] = [];
  const warmExperts: number[] = [];
  const tepidExperts: number[] = [];
  const coldExpertsGlobal: number[] = [];
  for (let rank = 0; rank < sortedDesc.length; rank++) {
    const [cnt, eid] = sortedDesc[rank];
    if (cnt === 0) {
      coldExpertsGlobal.push(eid);
    } else if (rank < hotCutIdx) {
      hotExperts.push(eid);
    } else if (rank < warmCutIdx) {
      warmExperts.push(eid);
    } else if (rank < tepidCutIdx) {
      tepidExperts.push(eid);
    } else {
      coldExpertsGlobal.push(eid);
    }
  }

  // CV = std / mean over global counts (helps catch unevenness even when
  // no expert is fully cold).
  const { mean: gMean, std: gStd } = meanAndStd(counts);
  const globalCV = gMean > 0 ? gStd / gMean : 0;

  // ── Per-layer cohort ─────────────────────────────────────────────────
  const layerEntropies: number[] = [];
  const layerPatterns: LayerPattern[] = [];
  const monopolizedLayers: number[] = [];
  const hotColdLayers: number[] = [];
  const balancedLayers: number[] = [];
  const sampleStarvedLayers: number[] = [];
  const layerLoadBalances: number[] = [];
  const dominantPerLayer: Record<number, number> = {};

  if (result?.layer_stats) {
    for (const ls of result.layer_stats) {
      const c = ls.expert_counts ?? [];
      const cls = classifyLayer(
        c, numExperts, topK, ls.load_balance ?? 0, ls.cold_experts ?? [],
      );
      layerEntropies.push(cls.entropyNorm);
      layerPatterns.push(cls.pattern);
      layerLoadBalances.push(ls.load_balance ?? 0);

      if (cls.pattern === 'monopolized') monopolizedLayers.push(ls.layer_idx);
      if (cls.pattern === 'hot_cold_split') hotColdLayers.push(ls.layer_idx);
      if (cls.pattern === 'balanced') balancedLayers.push(ls.layer_idx);
      if (cls.pattern === 'sparse_sample') sampleStarvedLayers.push(ls.layer_idx);

      // Track dominant expert per layer for cross-layer cohort
      if (c.length > 0) {
        let maxC = -1;
        let maxIdx = -1;
        for (let i = 0; i < c.length; i++) {
          if (c[i] > maxC) { maxC = c[i]; maxIdx = i; }
        }
        if (maxIdx >= 0) {
          dominantPerLayer[maxIdx] = (dominantPerLayer[maxIdx] ?? 0) + 1;
        }
      }
    }
  }
  const meanLoadBalance = layerLoadBalances.length > 0
    ? layerLoadBalances.reduce((a, b) => a + b, 0) / layerLoadBalances.length
    : 0;
  let worstLayerIdx: number | null = null;
  let worstLayerBalance = Infinity;
  for (const ls of result?.layer_stats ?? []) {
    if ((ls.load_balance ?? 0) < worstLayerBalance) {
      worstLayerBalance = ls.load_balance;
      worstLayerIdx = ls.layer_idx;
    }
  }
  if (worstLayerIdx === null) worstLayerBalance = 0;

  // Always-hot experts: dominant in ≥ ⌈numLayers · 0.25⌉ layers
  const dominanceThreshold = Math.max(1, Math.ceil(numLayers * 0.25));
  const alwaysHotExperts = Object.entries(dominantPerLayer)
    .filter(([, n]) => n >= dominanceThreshold)
    .map(([eid]) => parseInt(eid, 10));
  const dominanceCounts = Object.values(dominantPerLayer);
  const layerDominanceDiversity = shannonEntropy(dominanceCounts);

  const shape = detectShape(layerPatterns);
  const coldRatio = numExperts > 0 ? coldExpertsGlobal.length / numExperts : 0;

  return {
    result,
    loading,
    hasTrace,
    brain,
    hasModel,
    hasMoeArch,
    hasResult,
    runPhase,
    numExperts,
    topK,
    numLayers,
    totalSlots,
    totalRoutings,
    totalTokens,
    hotExperts,
    coldExpertsGlobal,
    warmExperts,
    tepidExperts,
    maxGlobalCount,
    topExpertId,
    topExpertShare,
    globalCV,
    layerEntropies,
    layerPatterns,
    monopolizedLayers,
    hotColdLayers,
    balancedLayers,
    sampleStarvedLayers,
    meanLoadBalance,
    worstLayerIdx,
    worstLayerBalance,
    alwaysHotExperts,
    layerDominanceDiversity,
    shape,
    coldRatio,
  };
}

// ── Risk ──────────────────────────────────────────────────────────────────

export type MoeRiskLevel = 'safe' | 'caution' | 'danger';
export interface MoeRisk {
  level: MoeRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Risk hierarchy:
 *  - safe: no model / not MoE (informative, not blocking)
 *  - danger: trace missing — must capture routing first
 *  - caution: > 50% layers sample-starved (trace too short)
 *  - caution: > 30% experts globally cold (consider expert pruning)
 *  - caution: monopoly heavy (> 25% layers monopolized)
 *  - caution: mean load_balance < 0.6
 *  - safe: balanced
 */
export function assessMoe(caps: MoeCapabilities): MoeRisk {
  if (caps.runPhase === 'noModel') {
    return { level: 'safe', reason: 'No model loaded.', reasonZh: '尚未加载模型.' };
  }
  if (caps.runPhase === 'notMoe') {
    return {
      level: 'safe',
      reason: 'Loaded model is not Mixture-of-Experts. Load a model with has_moe=true to use this analyzer.',
      reasonZh: '当前模型不是 MoE 架构. 加载有 has_moe=true 的模型才能用这个分析器.',
    };
  }
  if (caps.runPhase === 'noTrace') {
    return {
      level: 'danger',
      reason: 'No inference trace. Go to Inference Tracer, enable "Capture MoE expert routing", run a representative prompt, then return.',
      reasonZh: '没 inference trace. 去 Inference Tracer 勾选 "Capture MoE expert routing", 跑一个代表性 prompt, 再回来.',
    };
  }
  if (!caps.hasResult) {
    return {
      level: 'safe',
      reason: 'Trace captured. Click "Analyze" to compute expert utilization.',
      reasonZh: 'Trace 已就绪. 点 "Analyze" 开始计算 expert 利用率.',
    };
  }
  if (caps.shape === 'sample_starved') {
    return {
      level: 'caution',
      reason: `${caps.sampleStarvedLayers.length}/${caps.numLayers} layers have fewer routings than experts (sample too thin). Re-trace with longer prompt or higher max_tokens — at least ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens recommended.`,
      reasonZh: `${caps.sampleStarvedLayers.length}/${caps.numLayers} 层 routing 数 < expert 数 (采样不足). 用更长 prompt 或更大 max_tokens 重 trace — 推荐至少 ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens.`,
    };
  }
  if (caps.coldRatio > 0.3) {
    return {
      level: 'caution',
      reason: `${(caps.coldRatio * 100).toFixed(1)}% experts (${caps.coldExpertsGlobal.length}/${caps.numExperts}) saw zero routings. Either the trace is unrepresentative or these experts are pruning candidates.`,
      reasonZh: `${(caps.coldRatio * 100).toFixed(1)}% expert (${caps.coldExpertsGlobal.length}/${caps.numExperts}) 完全没被路由. 要么 trace 不代表性, 要么这些 expert 可以裁剪.`,
    };
  }
  if (caps.shape === 'monopoly_heavy') {
    return {
      level: 'caution',
      reason: `${caps.monopolizedLayers.length}/${caps.numLayers} layers monopolized (top-1 expert > 50% of layer traffic). Routing collapsed — model relies heavily on a few experts; check expert balance loss in training.`,
      reasonZh: `${caps.monopolizedLayers.length}/${caps.numLayers} 层 monopolized (top-1 expert 拿 > 50% 流量). routing collapse — 模型严重依赖少数几个 expert; 检查训练时的 balance loss.`,
    };
  }
  if (caps.shape === 'hot_cold_heavy') {
    return {
      level: 'caution',
      reason: `${caps.hotColdLayers.length}/${caps.numLayers} layers show hot/cold split (≥30% cold + concentrated traffic). Top experts could be promoted to dense; cold experts could be pruned.`,
      reasonZh: `${caps.hotColdLayers.length}/${caps.numLayers} 层是 hot/cold 分化 (≥30% cold + 流量集中). 热门 expert 可考虑 promote 成 dense, cold expert 可考虑裁剪.`,
    };
  }
  if (caps.meanLoadBalance < 0.6) {
    return {
      level: 'caution',
      reason: `Mean load_balance ${caps.meanLoadBalance.toFixed(3)} (worst L${caps.worstLayerIdx ?? '?'}=${caps.worstLayerBalance.toFixed(3)}). Routing is uneven — investigate or trace with more diverse prompts.`,
      reasonZh: `平均 load_balance ${caps.meanLoadBalance.toFixed(3)} (最差 L${caps.worstLayerIdx ?? '?'}=${caps.worstLayerBalance.toFixed(3)}). routing 不均衡 — 用更多样 prompt 重 trace 或检查问题.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.numExperts}×${caps.numLayers} grid, shape ${caps.shape}, ${caps.balancedLayers.length} balanced layers, mean load_balance ${caps.meanLoadBalance.toFixed(3)}, layer-dominance diversity ${caps.layerDominanceDiversity.toFixed(2)} bits.`,
    reasonZh: `${caps.numExperts}×${caps.numLayers} 网格, 模式 ${caps.shape}, ${caps.balancedLayers.length} 层均衡, 平均 load_balance ${caps.meanLoadBalance.toFixed(3)}, 跨层 dominance 多样性 ${caps.layerDominanceDiversity.toFixed(2)} bits.`,
  };
}

// ── Labels ────────────────────────────────────────────────────────────────

export function shapeLabel(s: MoeShape, locale: Locale): string {
  const map: Record<MoeShape, [string, string]> = {
    balanced: ['Balanced (健康路由)', '均衡 (健康路由)'],
    mostly_skew: ['Mostly skewed', '大多偏斜'],
    monopoly_heavy: ['Monopoly-heavy', '寡头主导'],
    hot_cold_heavy: ['Hot/cold split', '冷热两极分化'],
    sample_starved: ['Sample-starved', '采样不足'],
    mixed: ['Mixed', '混合'],
  };
  return locale === 'zh' ? map[s][1] : map[s][0];
}

export function bucketLabel(b: ExpertBucket, locale: Locale): string {
  const map: Record<ExpertBucket, [string, string]> = {
    hot: ['Hot (top 5%)', '热门 (前 5%)'],
    warm: ['Warm (5-25%)', '次热 (5-25%)'],
    tepid: ['Tepid (25-95%)', '常规 (25-95%)'],
    cold: ['Cold (0 routings)', '冷门 (零路由)'],
  };
  return locale === 'zh' ? map[b][1] : map[b][0];
}

// ── Brain context snippet ─────────────────────────────────────────────────

export function buildMoeContextSnippet(
  caps: MoeCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的 MoE 路由分析 (Mixture-of-Experts)` : `## YOUR MoE ROUTING ANALYSIS`,
    locale === 'zh'
      ? `这条流程把你 (${caps.brain?.model_name ?? '已加载的 MoE 模型'}) 每次 forward 时每个 token 选了哪 ${caps.topK} 个 expert (在 ${caps.numExperts} 个里) 的路由数据汇总, 让你能看到自己内部专家的负载分布. 你是 brain, 用第一人称解释这些路由模式 + MoE pruning/promote 含义.`
      : `This flow aggregates which ${caps.topK} of your ${caps.numExperts} experts each token routed to (${caps.brain?.model_name ?? 'the loaded MoE model'}) so you can see your own internal expert utilisation. You are the brain — explain the routing patterns + MoE pruning/promote implications in first person.`,
    `- Run phase: ${caps.runPhase}`,
    `- Architecture: ${caps.numLayers} layers × ${caps.numExperts} experts × top-${caps.topK} = ${caps.totalSlots} expert slots`,
  ];

  if (caps.runPhase === 'noModel') {
    lines.push(
      locale === 'zh' ? `- ⚠ 还没加载模型; 推荐先到 Models 页面加载一个 MoE 模型.` : `- ⚠ no model loaded; recommend loading a MoE model from the Models page.`,
    );
  } else if (caps.runPhase === 'notMoe') {
    lines.push(
      locale === 'zh' ? `- ⚠ 当前模型 (${caps.brain?.model_name}) 不是 MoE 架构 — 没有 expert 路由可分析. 这个页面只对 has_moe=true 的模型有意义.`
        : `- ⚠ current model (${caps.brain?.model_name}) is not MoE — no expert routing exists. This page only matters for has_moe=true models.`,
    );
  } else if (caps.runPhase === 'noTrace') {
    lines.push(
      locale === 'zh'
        ? `- ⚠ 没 inference trace — 必须先去 Inference Tracer 勾选 "Capture MoE expert routing" 跑一次 (推荐至少 ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens 才足够看 routing 模式).`
        : `- ⚠ no inference trace — must first run Inference Tracer with "Capture MoE expert routing" enabled (recommend at least ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens for routing patterns to be statistically meaningful).`,
    );
  } else if (caps.hasResult) {
    lines.push(
      ``,
      locale === 'zh' ? `### 实测路由数据:` : `### Measured routing data:`,
      `- Total tokens routed: ${caps.totalTokens}`,
      `- Active routings: ${caps.totalTokens * caps.topK} per layer (= total ${caps.totalTokens * caps.topK * caps.numLayers} across layers)`,
      `- Mean load balance: ${caps.meanLoadBalance.toFixed(3)} (1.0 = perfect)`,
      `- Macro shape: ${caps.shape}`,
      ``,
      locale === 'zh' ? `### 专家分桶 (全局):` : `### Expert buckets (global):`,
      `- Hot (top 5%): ${caps.hotExperts.length}, top-1 expert E${caps.topExpertId ?? '?'} (${(caps.topExpertShare * 100).toFixed(1)}% of all routings)`,
      `- Warm (5-25%): ${caps.warmExperts.length}`,
      `- Tepid (25-95%): ${caps.tepidExperts.length}`,
      `- Cold (zero routings): ${caps.coldExpertsGlobal.length} (${(caps.coldRatio * 100).toFixed(1)}%)`,
      `- Global expert traffic CV: ${caps.globalCV.toFixed(3)} (lower = more uniform)`,
      ``,
      locale === 'zh' ? `### 层级模式:` : `### Layer patterns:`,
      `- Balanced: ${caps.balancedLayers.length} layers${caps.balancedLayers.length > 0 ? ` [${caps.balancedLayers.slice(0, 8).join(', ')}${caps.balancedLayers.length > 8 ? '…' : ''}]` : ''}`,
      `- Monopolized (top-1 > 50%): ${caps.monopolizedLayers.length} layers${caps.monopolizedLayers.length > 0 ? ` [${caps.monopolizedLayers.slice(0, 8).join(', ')}${caps.monopolizedLayers.length > 8 ? '…' : ''}]` : ''}`,
      `- Hot/cold split: ${caps.hotColdLayers.length} layers${caps.hotColdLayers.length > 0 ? ` [${caps.hotColdLayers.slice(0, 8).join(', ')}${caps.hotColdLayers.length > 8 ? '…' : ''}]` : ''}`,
      `- Sample-starved: ${caps.sampleStarvedLayers.length} layers (need longer trace)`,
      `- Worst load_balance: layer ${caps.worstLayerIdx ?? '?'} = ${caps.worstLayerBalance.toFixed(3)}`,
      ``,
      locale === 'zh' ? `### 跨层 dominance:` : `### Cross-layer dominance:`,
      `- Always-hot experts (top-1 in ≥${Math.ceil(caps.numLayers * 0.25)} layers): ${caps.alwaysHotExperts.length}${caps.alwaysHotExperts.length > 0 ? ` [${caps.alwaysHotExperts.slice(0, 8).join(', ')}${caps.alwaysHotExperts.length > 8 ? '…' : ''}]` : ''}`,
      `- Layer-dominance diversity: ${caps.layerDominanceDiversity.toFixed(2)} bits (higher = different experts dominate different layers)`,
    );
  }

  lines.push(
    ``,
    locale === 'zh' ? `### 路由概念 (引用时用):` : `### Concept refs (cite when explaining):`,
    locale === 'zh'
      ? `- top-K routing: 每个 token 走 softmax(gate) → 取最高 K 个 expert, scores normalize 后加权和.`
      : `- top-K routing: each token does softmax(gate), picks top K experts, sums their outputs weighted by normalized scores.`,
    locale === 'zh'
      ? `- Load balance ≈ N · Σ fᵢ pᵢ; 1.0 = 完美均衡, 越低越偏斜.`
      : `- Load balance ≈ N · Σ fᵢ pᵢ; 1.0 = perfectly balanced, lower = more skewed.`,
    locale === 'zh'
      ? `- Cold expert: 0 路由 — 可考虑 prune (优化目标), 但短 trace 也会假阳性.`
      : `- Cold expert: zero routings — pruning candidate, but short traces can produce false positives.`,
    locale === 'zh'
      ? `- Monopolized layer: top-1 expert > 50% — 通常说明 routing collapse 或训练 balance loss 过弱.`
      : `- Monopolized layer: top-1 expert > 50% — often indicates routing collapse or weak balance loss in training.`,
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: 路由捕获在 forward 时本机, 分析也在本机, 0 次云调用.`
      : `### North-star §1 sovereignty: routing capture happens locally during forward, analysis local, zero cloud calls.`,
  );
  return lines.filter(Boolean).join('\n');
}

// ── Auto brief — 6 states × 2 locales ───────────────────────────────────

export function buildMoeAutoBrief(
  caps: MoeCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (caps.runPhase === 'noModel') {
      return `还没加载模型. 用 1-2 句作为 brain 介绍 MoE 路由分析是干啥, 推荐去 Models 加载一个 has_moe=true 的模型.`;
    }
    if (caps.runPhase === 'notMoe') {
      return `当前模型 (${caps.brain?.model_name ?? '?'}) 不是 MoE. 用 2-3 句作为 brain 解释 MoE 路由 vs dense FFN 的差别, 列 1-2 个值得加载的 MoE 模型 (Mixtral / Qwen3.6-A3B / DeepSeek-V3).`;
    }
    if (caps.runPhase === 'noTrace') {
      return `MoE 模型 (${caps.numExperts} experts × top-${caps.topK} × ${caps.numLayers} layers) 已加载, 但还没 routing trace. 用 2-3 句作为 brain 解释怎么生成 trace (Inference Tracer 勾选 "Capture MoE expert routing", 推荐至少 ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens).`;
    }
    if (caps.runPhase === 'idle') {
      return `Trace 已就绪. 用 1-2 句作为 brain 简介接下来分析会做什么 (聚合 ${caps.totalTokens} tokens 的路由 → ${caps.numLayers}×${caps.numExperts} 利用率矩阵), 推荐点 Analyze.`;
    }
    if (caps.runPhase === 'analyzing') {
      return `Analyzer 在跑. 用 1-2 句作为 brain 解释正在算什么 (per-layer entropy + load balance + cold/hot 分桶 + 跨层 dominance).`;
    }
    return `分析完成: ${caps.numExperts}×${caps.numLayers} 网格, 模式 ${caps.shape}, ${caps.balancedLayers.length} 层均衡 / ${caps.monopolizedLayers.length} 层寡头 / ${caps.hotColdLayers.length} 层冷热分化, 全局 cold ${(caps.coldRatio * 100).toFixed(0)}%, 平均 load_balance ${caps.meanLoadBalance.toFixed(3)}. 用 2-3 句作为 brain 解读路由健康度 + 推荐 prune/promote/重 trace 之一. 第一人称, 引用具体数字.`;
  }

  if (caps.runPhase === 'noModel') {
    return `No model loaded. In 1-2 sentences as brain, introduce MoE routing analysis and recommend loading a has_moe=true model from the Models page.`;
  }
  if (caps.runPhase === 'notMoe') {
    return `Current model (${caps.brain?.model_name ?? '?'}) is not MoE. In 2-3 sentences as brain, explain MoE routing vs dense FFN, list 1-2 worthwhile MoE models (Mixtral / Qwen3.6-A3B / DeepSeek-V3).`;
  }
  if (caps.runPhase === 'noTrace') {
    return `MoE model (${caps.numExperts} experts × top-${caps.topK} × ${caps.numLayers} layers) loaded, but no routing trace. In 2-3 sentences as brain, explain how to capture one (Inference Tracer with "Capture MoE expert routing", at least ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens recommended).`;
  }
  if (caps.runPhase === 'idle') {
    return `Trace ready. In 1-2 sentences as brain, briefly preview what Analyze will do (aggregate ${caps.totalTokens} tokens' routing into a ${caps.numLayers}×${caps.numExperts} utilisation matrix). Recommend clicking Analyze.`;
  }
  if (caps.runPhase === 'analyzing') {
    return `Analyzer running. In 1-2 sentences as brain, explain what's being computed (per-layer entropy + load balance + cold/hot bucketing + cross-layer dominance).`;
  }
  return `Analyzed: ${caps.numExperts}×${caps.numLayers} grid, shape ${caps.shape}, ${caps.balancedLayers.length} balanced / ${caps.monopolizedLayers.length} monopolized / ${caps.hotColdLayers.length} hot-cold layers, global cold ${(caps.coldRatio * 100).toFixed(0)}%, mean load_balance ${caps.meanLoadBalance.toFixed(3)}. In 2-3 sentences as brain, interpret routing health + recommend one of prune / promote / re-trace. First person, cite numbers.`;
}

// ── Suggested prompts (4 per state per locale) ───────────────────────────

export function getMoeSuggestedPrompts(
  caps: MoeCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.runPhase === 'noModel') {
      return [
        { label: '🧠 MoE 是什么', prompt: `用 2-3 句解释 Mixture-of-Experts 架构 vs dense FFN, 为什么 inference 时只激活一小部分参数.` },
        { label: '📦 推荐 MoE 模型', prompt: `给 1-2 个值得加载的 MoE 模型 (Mixtral 8x7B / Qwen3.6-35B-A3B / DeepSeek-V3 等), 指出每个的特点.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: MoE 路由分析跑在本机, 0 次云调用.` },
      ];
    }
    if (caps.runPhase === 'notMoe') {
      return [
        { label: '🧠 MoE vs dense', prompt: `用 2-3 句解释 MoE vs dense FFN 在推理 / 训练 / 内存上的差别.` },
        { label: '🎯 我适合 MoE 吗', prompt: `对一个像我这样的 ${caps.brain?.num_layers ?? '?'}-layer dense 模型, 切到 MoE 在哪些场景能赚 (大 batch / 高 throughput / 低延迟)?` },
      ];
    }
    if (caps.runPhase === 'noTrace') {
      return [
        { label: '📊 怎么 trace', prompt: `用 2-3 句解释怎么 trace MoE 路由: Inference Tracer + capture_moe_routing=true, 至少 ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens (因为 ${caps.numExperts} experts × top-${caps.topK} 需要最少这么多采样才有 1× 覆盖).` },
        { label: '🎯 用什么 prompt', prompt: `推荐用什么 prompt 才能看到代表性 routing (代码 / 数学 / 自然语言混合, 长度 ≥ ${Math.max(256, caps.numExperts)} tokens).` },
        { label: '⚠️ 短 trace 陷阱', prompt: `如果只跑 50 tokens 而我有 ${caps.numExperts} experts, cold expert 数会有多假? 用 2-3 句解释采样 vs 真实分布.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 路由捕获在 forward 时本机, 分析也本机, 0 次云调用.` },
      ];
    }
    if (caps.runPhase === 'idle') {
      return [
        { label: '🎯 现在该分析吗', prompt: `Trace 已就绪 (${caps.totalTokens} tokens), ${caps.numExperts}×${caps.numLayers} 网格等聚合. 用 2-3 句作为 brain 简述点 Analyze 后会出什么 (heatmap + load balance bar + global expert bar + 层表 + cold list).` },
        { label: '📊 期望什么', prompt: `${caps.totalTokens} tokens × top-${caps.topK} = ${caps.totalTokens * caps.topK} routings/layer 对 ${caps.numExperts} experts. 用 2-3 句给一个先验: 应该看到多少 cold? 多少 balanced layer?` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 整条流程 (capture → analyze → 显示) 全在本机.` },
      ];
    }
    if (caps.runPhase === 'analyzing') {
      return [
        { label: '⏱️ 在算什么', prompt: `Analyzer 正在做什么具体计算 (per-layer entropy / load balance / cold experts / 跨层 dominance)? 用 2-3 句解释.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 这个聚合计算正在本机内存里跑.` },
      ];
    }
    return [
      { label: '📊 这套路由健康吗', prompt: `${caps.numExperts}×${caps.numLayers} 网格, 模式 ${caps.shape}, ${caps.balancedLayers.length} 层均衡, ${caps.monopolizedLayers.length} 层寡头, ${caps.hotColdLayers.length} 层冷热分化, 全局 cold ${(caps.coldRatio * 100).toFixed(0)}%. 用 2-3 句作为 brain 评估这是健康的 MoE 路由吗.` },
      { label: '✂️ 该 prune 哪些 expert', prompt: `Cold experts ${caps.coldExpertsGlobal.length} 个 (${(caps.coldRatio * 100).toFixed(1)}%)${caps.alwaysHotExperts.length > 0 ? `, always-hot experts ${caps.alwaysHotExperts.length} 个 (E${caps.alwaysHotExperts[0]} 等)` : ''}. 用 2-3 句给具体 prune/promote 推荐: 哪些 cold 真能 prune, 哪些 hot 值得 promote 成 dense?` },
      { label: '⚠️ 异常层', prompt: `${caps.monopolizedLayers.length > 0 ? `Monopolized 层: ${caps.monopolizedLayers.slice(0, 5).join(', ')}.` : '没 monopolized 层.'}${caps.worstLayerIdx !== null ? ` 最差 load_balance: L${caps.worstLayerIdx}=${caps.worstLayerBalance.toFixed(3)}.` : ''} 用 2-3 句解读: 这是 routing collapse 还是 trace 不代表性?` },
      { label: '📦 接下来', prompt: `基于这次分析, 下一步推荐 (再 trace 几个长 prompt 验证 / prune cold experts / 跑量化 / 跳到 Pruning 页面)? 用 2-3 句给具体步骤.` },
    ];
  }

  // English
  if (caps.runPhase === 'noModel') {
    return [
      { label: '🧠 What is MoE', prompt: `In 2-3 sentences, explain Mixture-of-Experts vs dense FFN — why inference activates only a fraction of parameters.` },
      { label: '📦 Recommended MoE models', prompt: `Suggest 1-2 worthwhile MoE models to load (Mixtral 8x7B / Qwen3.6-35B-A3B / DeepSeek-V3) and call out their distinctive traits.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: MoE routing analysis runs locally, zero cloud calls.` },
    ];
  }
  if (caps.runPhase === 'notMoe') {
    return [
      { label: '🧠 MoE vs dense', prompt: `In 2-3 sentences, explain MoE vs dense FFN tradeoffs across inference / training / memory.` },
      { label: '🎯 Should I switch to MoE', prompt: `For a ${caps.brain?.num_layers ?? '?'}-layer dense model like me, when is switching to MoE worthwhile (large batch / high throughput / low latency)?` },
    ];
  }
  if (caps.runPhase === 'noTrace') {
    return [
      { label: '📊 How to trace', prompt: `In 2-3 sentences, explain how to capture MoE routing: Inference Tracer + capture_moe_routing=true, at least ${Math.ceil(caps.numExperts / Math.max(caps.topK, 1))} tokens (because ${caps.numExperts} experts × top-${caps.topK} needs at least that many for 1× coverage).` },
      { label: '🎯 Which prompts', prompt: `What prompts produce representative routing (code / math / natural-language mix, length ≥ ${Math.max(256, caps.numExperts)} tokens)?` },
      { label: '⚠️ Short-trace pitfall', prompt: `If I only trace 50 tokens with ${caps.numExperts} experts, how false will the cold count be? In 2-3 sentences, explain sampling vs true distribution.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: routing capture during forward is local, analysis is local, zero cloud calls.` },
    ];
  }
  if (caps.runPhase === 'idle') {
    return [
      { label: '🎯 Should I click Analyze', prompt: `Trace ready (${caps.totalTokens} tokens), ${caps.numExperts}×${caps.numLayers} grid awaits aggregation. In 2-3 sentences as brain, briefly preview what Analyze produces (heatmap + load balance bar + global expert bar + layer table + cold list).` },
      { label: '📊 What to expect', prompt: `${caps.totalTokens} tokens × top-${caps.topK} = ${caps.totalTokens * caps.topK} routings/layer over ${caps.numExperts} experts. In 2-3 sentences, give a prior: how many cold experts should I expect, how many balanced layers?` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: the whole flow (capture → analyze → display) runs locally.` },
    ];
  }
  if (caps.runPhase === 'analyzing') {
    return [
      { label: '⏱️ What is the analyzer doing', prompt: `What does the analyzer compute (per-layer entropy / load balance / cold experts / cross-layer dominance)? In 2-3 sentences.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this aggregation runs in local memory, zero cloud calls.` },
    ];
  }
  return [
    { label: '📊 Is this routing healthy', prompt: `${caps.numExperts}×${caps.numLayers} grid, shape ${caps.shape}, ${caps.balancedLayers.length} balanced layers, ${caps.monopolizedLayers.length} monopolized, ${caps.hotColdLayers.length} hot/cold split, global cold ${(caps.coldRatio * 100).toFixed(0)}%. In 2-3 sentences as brain, assess: is this healthy MoE routing.` },
    { label: '✂️ Which experts to prune', prompt: `Cold experts ${caps.coldExpertsGlobal.length} (${(caps.coldRatio * 100).toFixed(1)}%)${caps.alwaysHotExperts.length > 0 ? `, always-hot experts ${caps.alwaysHotExperts.length} (E${caps.alwaysHotExperts[0]} etc.)` : ''}. In 2-3 sentences, give a concrete prune/promote rec: which cold are real prune candidates, which hot deserve promotion to dense?` },
    { label: '⚠️ Anomalous layers', prompt: `${caps.monopolizedLayers.length > 0 ? `Monopolized layers: ${caps.monopolizedLayers.slice(0, 5).join(', ')}.` : 'No monopolized layers.'}${caps.worstLayerIdx !== null ? ` Worst load_balance: L${caps.worstLayerIdx}=${caps.worstLayerBalance.toFixed(3)}.` : ''} In 2-3 sentences, interpret: is this routing collapse or unrepresentative trace?` },
    { label: '📦 Next', prompt: `Based on this analysis, what's next (re-trace with more long prompts / prune cold experts / run quantization / go to Pruning page)? In 2-3 sentences with concrete steps.` },
  ];
}
