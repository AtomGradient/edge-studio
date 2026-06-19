// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * activationInsights — derived activation-heatmap state + chat helpers
 * for the /activation page (ActivationHeatmap.tsx).
 *
 * Activation heatmap visualises MLP intermediate-layer activations sampled
 * during an activation profile run. The page exposes:
 *  - 2D heatmap (layers × neurons), max or mean
 *  - 3D surface / scatter views
 *  - Threshold slider (defines "dead" cutoff)
 *  - Multi-threshold compare table (dead@0.01 / 0.05 / 0.1 / 0.5)
 *  - Per-layer focus
 *
 * §9.1 N-layer cohort: every layer is a slot; capability =
 *   profile metadata + per-layer dead/alive + threshold-sweep dead counts
 *   + variance signal across layers.
 *
 * §10.3 naming: capability uses `runPhase` not `status`.
 *
 * Sovereignty (§9.2 mandatory): heatmap data is materialised from local
 * profile arrays; no inference traffic to outside services.
 */
import type { ModelInfo, ActivationHeatmapData, ProfileSummary } from '@/api/types';

type Locale = 'en' | 'zh';

export type ActivationPhase = 'noModel' | 'noProfile' | 'loading' | 'hasData';

export type DeadRatioBucket =
  | 'healthy'    // < 5% dead
  | 'normal'     // 5..15%
  | 'sparse'     // 15..30%
  | 'pathological'; // > 30%

export type VarianceBucket = 'low' | 'moderate' | 'high';

/** Per-layer summary derived from a single matrix row. */
export interface LayerActivitySummary {
  layerIdx: number;
  alive: number;
  dead: number;
  total: number;
  deadRatio: number;
  maxActivation: number;
  meanActivation: number;
}

export interface ThresholdSweepRow {
  threshold: number;
  dead: number;
  total: number;
  ratio: number;
}

export interface ActivationCapabilities {
  // ── Inputs ─────────────────────────────────────────────────────────────
  profileSummary: ProfileSummary | null;
  data: ActivationHeatmapData | null;
  metric: 'max' | 'mean';
  threshold: number;
  selectedLayer: number;
  loading: boolean;
  brain: ModelInfo | null;

  // ── Gating ─────────────────────────────────────────────────────────────
  hasModel: boolean;
  hasProfile: boolean;
  hasData: boolean;
  runPhase: ActivationPhase;

  // ── Profile-level (works even without heatmap) ─────────────────────────
  numLayers: number;
  neuronsPerLayer: number;
  totalNeurons: number;
  totalDeadAtCurrentThreshold: number;
  globalDeadRatio: number;
  deadRatioBucket: DeadRatioBucket;

  // ── Per-layer cohort (only when data present) ──────────────────────────
  layerSummaries: LayerActivitySummary[];
  /** Layer with highest dead ratio (most inactive). */
  deadestLayer: LayerActivitySummary | null;
  /** Layer with lowest dead ratio (most active). */
  liveliestLayer: LayerActivitySummary | null;
  /** Variance of dead ratios across layers (signals bad profile if high). */
  deadRatioVariance: number;
  varianceBucket: VarianceBucket;
  /** Selected layer focus (mirrors UI state). */
  selectedSummary: LayerActivitySummary | null;

  // ── Threshold sweep ────────────────────────────────────────────────────
  thresholdSweep: ThresholdSweepRow[];
}

const COMPARE_THRESHOLDS = [0.01, 0.05, 0.1, 0.5] as const;

function bucketDeadRatio(r: number): DeadRatioBucket {
  if (r < 0.05) return 'healthy';
  if (r < 0.15) return 'normal';
  if (r < 0.30) return 'sparse';
  return 'pathological';
}

function bucketVariance(v: number): VarianceBucket {
  // Variance of dead ratios (numbers in [0,1]). Practical scale:
  //   0.001 = layers nearly identical (low)
  //   0.005 = moderate (real diversity)
  //   0.02+ = high (likely pathological / pathological prompts)
  if (v < 0.002) return 'low';
  if (v < 0.01) return 'moderate';
  return 'high';
}

export function deriveActivationCapabilities(
  profileSummary: ProfileSummary | null,
  data: ActivationHeatmapData | null,
  metric: 'max' | 'mean',
  threshold: number,
  selectedLayer: number,
  loading: boolean,
  brain: ModelInfo | null,
): ActivationCapabilities {
  const hasModel = !!brain;
  const hasProfile = !!profileSummary;
  const hasData = !!data;

  let runPhase: ActivationPhase;
  if (!hasModel) runPhase = 'noModel';
  else if (!hasProfile) runPhase = 'noProfile';
  else if (loading || !hasData) runPhase = 'loading';
  else runPhase = 'hasData';

  const numLayers = profileSummary?.num_layers ?? 0;
  const neuronsPerLayer = profileSummary?.intermediate_size ?? 0;
  const totalNeurons = numLayers * neuronsPerLayer;

  // Per-layer summaries (require both profile + heatmap data)
  const layerSummaries: LayerActivitySummary[] = [];
  if (data) {
    const matrix = metric === 'max' ? data.max_matrix : data.mean_matrix;
    for (let i = 0; i < matrix.length; i++) {
      const row = matrix[i];
      let alive = 0;
      let max = 0;
      let sum = 0;
      for (const v of row) {
        if (v >= threshold) alive++;
        if (v > max) max = v;
        sum += v;
      }
      layerSummaries.push({
        layerIdx: i,
        alive,
        dead: row.length - alive,
        total: row.length,
        deadRatio: row.length > 0 ? (row.length - alive) / row.length : 0,
        maxActivation: max,
        meanActivation: row.length > 0 ? sum / row.length : 0,
      });
    }
  }

  let deadestLayer: LayerActivitySummary | null = null;
  let liveliestLayer: LayerActivitySummary | null = null;
  for (const s of layerSummaries) {
    if (deadestLayer === null || s.deadRatio > deadestLayer.deadRatio) deadestLayer = s;
    if (liveliestLayer === null || s.deadRatio < liveliestLayer.deadRatio) liveliestLayer = s;
  }

  // Variance of dead ratios
  let variance = 0;
  if (layerSummaries.length > 0) {
    const mean = layerSummaries.reduce((a, s) => a + s.deadRatio, 0) / layerSummaries.length;
    variance = layerSummaries.reduce((a, s) => a + (s.deadRatio - mean) ** 2, 0) / layerSummaries.length;
  }

  const totalDeadAtCurrentThreshold = layerSummaries.reduce((a, s) => a + s.dead, 0);
  const globalDeadRatio = totalNeurons > 0
    ? totalDeadAtCurrentThreshold / totalNeurons
    : (profileSummary?.dead_ratio_at_01 ?? 0);

  // Threshold sweep
  const thresholdSweep: ThresholdSweepRow[] = [];
  if (data) {
    const matrix = data.max_matrix; // sweep always uses max
    for (const t of COMPARE_THRESHOLDS) {
      let dead = 0;
      let total = 0;
      for (const row of matrix) {
        for (const val of row) {
          total++;
          if (val < t) dead++;
        }
      }
      thresholdSweep.push({
        threshold: t,
        dead,
        total,
        ratio: total > 0 ? dead / total : 0,
      });
    }
  }

  return {
    profileSummary,
    data,
    metric,
    threshold,
    selectedLayer,
    loading,
    brain,
    hasModel,
    hasProfile,
    hasData,
    runPhase,
    numLayers,
    neuronsPerLayer,
    totalNeurons,
    totalDeadAtCurrentThreshold,
    globalDeadRatio,
    deadRatioBucket: bucketDeadRatio(globalDeadRatio),
    layerSummaries,
    deadestLayer,
    liveliestLayer,
    deadRatioVariance: variance,
    varianceBucket: bucketVariance(variance),
    selectedSummary: layerSummaries.find((s) => s.layerIdx === selectedLayer) ?? null,
    thresholdSweep,
  };
}

export type ActRiskLevel = 'safe' | 'caution' | 'danger';
export interface ActRisk {
  level: ActRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Risk hierarchy:
 *  - danger: no profile (cannot view)
 *  - caution: globalDeadRatio > 30% (pathological — most neurons inactive)
 *  - caution: variance > 0.01 (layers wildly inconsistent — may indicate
 *    profile generated from too few or atypical prompts)
 *  - caution: layer 0 or last layer dead ratio > 50% (sensitive boundaries)
 *  - safe: healthy / normal / sparse with reasonable variance
 */
export function assessActivation(caps: ActivationCapabilities): ActRisk {
  if (!caps.hasModel) {
    return { level: 'safe', reason: 'No model loaded.', reasonZh: '尚未加载模型.' };
  }
  if (!caps.hasProfile) {
    return {
      level: 'danger',
      reason: 'No activation profile loaded. Heatmap shows nothing — go to the dashboard, run an activation profile on representative prompts (≥ 10 runs).',
      reasonZh: '没加载激活 profile. heatmap 显示不出 — 去 dashboard 用代表性 prompts 跑 (>= 10 次).',
    };
  }
  if (caps.deadRatioBucket === 'pathological') {
    return {
      level: 'caution',
      reason: `Global dead ratio ${(caps.globalDeadRatio * 100).toFixed(1)}% — most neurons inactive. Either prompts were too narrow, or the model is genuinely sparse (some MoE/sparse arch). Re-profile with diverse prompts before pruning.`,
      reasonZh: `全局失活率 ${(caps.globalDeadRatio * 100).toFixed(1)}% — 大部分神经元不动. 可能 prompts 太窄, 或模型本身就是稀疏 (MoE/稀疏架构). 用多样 prompts 重新跑 profile 再剪枝.`,
    };
  }
  if (caps.varianceBucket === 'high') {
    return {
      level: 'caution',
      reason: `High inter-layer dead-ratio variance (${caps.deadRatioVariance.toFixed(4)}). Some layers are much sparser than others — usually a sign the profile reflects atypical prompts. Diversify the profile corpus.`,
      reasonZh: `层间失活率方差 ${caps.deadRatioVariance.toFixed(4)} 偏高. 部分层比其他稀疏很多 — 通常说明 profile 用的 prompts 不够代表. 加大语料多样性重跑.`,
    };
  }
  if (caps.layerSummaries.length > 0) {
    const first = caps.layerSummaries[0];
    const last = caps.layerSummaries[caps.layerSummaries.length - 1];
    if (first.deadRatio > 0.5) {
      return {
        level: 'caution',
        reason: `Layer 0 has ${(first.deadRatio * 100).toFixed(1)}% dead. The first layer is embedding-adjacent — high dead is unusual and suggests profile/threshold issues.`,
        reasonZh: `Layer 0 失活率 ${(first.deadRatio * 100).toFixed(1)}%. 首层紧邻 embedding — 高失活异常, 通常说明 profile 或阈值有问题.`,
      };
    }
    if (last.deadRatio > 0.5) {
      return {
        level: 'caution',
        reason: `Last layer has ${(last.deadRatio * 100).toFixed(1)}% dead. Output projection is lm_head-adjacent — high dead suggests profile/threshold issues.`,
        reasonZh: `末层失活率 ${(last.deadRatio * 100).toFixed(1)}%. 输出投影紧邻 lm_head — 高失活通常是 profile 或阈值问题.`,
      };
    }
  }
  return {
    level: 'safe',
    reason: caps.hasData
      ? `${caps.numLayers} layers, ${(caps.globalDeadRatio * 100).toFixed(1)}% dead @ θ=${caps.threshold}, variance ${caps.deadRatioVariance.toFixed(4)} (${caps.varianceBucket}).`
      : `${caps.numLayers} layers, ${(caps.globalDeadRatio * 100).toFixed(1)}% baseline dead.`,
    reasonZh: caps.hasData
      ? `${caps.numLayers} 层, θ=${caps.threshold} 下失活 ${(caps.globalDeadRatio * 100).toFixed(1)}%, 方差 ${caps.deadRatioVariance.toFixed(4)} (${caps.varianceBucket}).`
      : `${caps.numLayers} 层, 基线失活 ${(caps.globalDeadRatio * 100).toFixed(1)}%.`,
  };
}

export function deadBucketLabel(b: DeadRatioBucket, locale: Locale): string {
  const map: Record<DeadRatioBucket, [string, string]> = {
    healthy: ['Healthy', '健康'],
    normal: ['Normal', '正常'],
    sparse: ['Sparse', '稀疏'],
    pathological: ['Pathological', '病态'],
  };
  return locale === 'zh' ? map[b][1] : map[b][0];
}

export function varianceBucketLabel(b: VarianceBucket, locale: Locale): string {
  const map: Record<VarianceBucket, [string, string]> = {
    low: ['Low', '低'],
    moderate: ['Moderate', '适中'],
    high: ['High', '高'],
  };
  return locale === 'zh' ? map[b][1] : map[b][0];
}

export function buildActivationContextSnippet(
  caps: ActivationCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的激活热图 (Activation Heatmap)` : `## YOUR ACTIVATION HEATMAP`,
    locale === 'zh'
      ? `这条流程从激活 profile 渲染 layer × neuron 的活跃度图. 你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 是 brain, 用第一人称解释这些活跃模式 + 风险信号.`
      : `This flow renders a layer × neuron activity map from the activation profile. You (${caps.brain?.model_name ?? 'the loaded LLM'}) are the brain; explain the activity patterns + risk signals in first person.`,
    `- Run phase: ${caps.runPhase}`,
    !caps.hasProfile
      ? (locale === 'zh' ? `- ⚠ 还没有 profile — 必须先 dashboard 跑 activation profile.` : `- ⚠ no profile yet — user must first run an activation profile from the dashboard.`)
      : `- Profile: ${caps.numLayers} layers, ${caps.neuronsPerLayer} neurons/layer (${caps.totalNeurons.toLocaleString()} total)`,
  ];

  if (caps.hasData) {
    lines.push(
      `- Metric: ${caps.metric} · threshold θ=${caps.threshold}`,
      `- Global dead @θ=${caps.threshold}: ${caps.totalDeadAtCurrentThreshold.toLocaleString()} (${(caps.globalDeadRatio * 100).toFixed(2)}%, bucket=${caps.deadRatioBucket})`,
      `- Inter-layer dead-ratio variance: ${caps.deadRatioVariance.toFixed(4)} (${caps.varianceBucket})`,
      caps.deadestLayer
        ? `- Most-dead layer: L${caps.deadestLayer.layerIdx} (${(caps.deadestLayer.deadRatio * 100).toFixed(1)}% dead)`
        : '',
      caps.liveliestLayer
        ? `- Most-alive layer: L${caps.liveliestLayer.layerIdx} (${(caps.liveliestLayer.deadRatio * 100).toFixed(1)}% dead)`
        : '',
      caps.selectedSummary
        ? `- Selected L${caps.selectedSummary.layerIdx}: alive ${caps.selectedSummary.alive}/${caps.selectedSummary.total} (${(caps.selectedSummary.deadRatio * 100).toFixed(1)}% dead, max=${caps.selectedSummary.maxActivation.toFixed(3)}, mean=${caps.selectedSummary.meanActivation.toFixed(4)})`
        : '',
      ``,
      locale === 'zh' ? `### 阈值 sweep:` : `### Threshold sweep:`,
      ...caps.thresholdSweep.map((r) => `- @θ=${r.threshold}: ${r.dead.toLocaleString()} dead (${(r.ratio * 100).toFixed(2)}%)`),
    );
  }
  lines.push(
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: 热图渲染纯本地 profile 数据, 0 次云调用.`
      : `### North-star §1 sovereignty: heatmap renders pure local profile arrays, zero cloud calls.`,
  );
  return lines.filter(Boolean).join('\n');
}

export function buildActivationAutoBrief(
  caps: ActivationCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (!caps.hasModel) {
      return `还没加载模型. 用 1-2 句作为 brain 介绍 activation heatmap 是干啥. 第一人称.`;
    }
    if (!caps.hasProfile) {
      return `还没有激活 profile. 用 2-3 句作为 brain 解释 heatmap 需要什么数据 (per-layer per-neuron activations from inference), 推荐用户去 dashboard 跑一次. 第一人称.`;
    }
    if (caps.runPhase === 'loading') {
      return `正在拉 heatmap 数据 (θ=${caps.threshold}). 用 1-2 句简述 heatmap 在解释什么 (哪些神经元在动 / 哪些 dead) + 等的时候该想什么. 第一人称.`;
    }
    return `Heatmap 出来了: ${caps.numLayers} 层, θ=${caps.threshold} 下全局失活 ${(caps.globalDeadRatio * 100).toFixed(1)}% (${caps.deadRatioBucket}), 层间方差 ${caps.varianceBucket}. 用 2-3 句作为 brain 解读: 这套 profile 健康吗, 有没有异常层, 推荐什么 prune 起手. 第一人称, 引用具体数字.`;
  }
  if (!caps.hasModel) {
    return `No model loaded. In 1-2 sentences as brain, introduce the activation heatmap. First person.`;
  }
  if (!caps.hasProfile) {
    return `No activation profile yet. In 2-3 sentences as brain, explain what the heatmap needs (per-layer per-neuron activations from inference), and recommend running a profile from the dashboard. First person.`;
  }
  if (caps.runPhase === 'loading') {
    return `Fetching heatmap data (θ=${caps.threshold}). In 1-2 sentences as brain, briefly explain what the heatmap reveals (which neurons fire vs are dead) + what to think about while waiting. First person.`;
  }
  return `Heatmap rendered: ${caps.numLayers} layers, ${(caps.globalDeadRatio * 100).toFixed(1)}% dead @θ=${caps.threshold} (${caps.deadRatioBucket}), variance ${caps.varianceBucket}. In 2-3 sentences as brain, interpret: is this profile healthy, any anomalous layers, what pruning starting point. First person, cite numbers.`;
}

export function getActivationSuggestedPrompts(
  caps: ActivationCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (!caps.hasProfile) {
      return [
        { label: '🎯 heatmap 是干啥', prompt: `用 2-3 句作为 brain 解释 activation heatmap (每层 × 每神经元的活跃度), 与单一 dead ratio 数字相比能看到什么额外信息.` },
        { label: '📊 怎么跑 profile', prompt: `用 2-3 句解释 activation profile 怎么跑: 用什么 prompts (代表性), 多少轮 (>= 10), 哪些指标关键.` },
        { label: '🔍 看出什么', prompt: `从 heatmap 能看出哪些信号 (统一 dead / 单层 dead / 中间峰)? 各对应什么模型问题.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: profile 跑在本机, heatmap 渲染也在本机, 0 次云调用.` },
      ];
    }
    if (caps.runPhase === 'loading') {
      return [
        { label: '⏱️ 为什么慢', prompt: `Heatmap 数据通常多大, 为什么会慢 (大模型 N×M 矩阵 N=32 M=14336 = 几 MB 传输 + 渲染). 用 2-3 句解释.` },
        { label: '🎯 等的时候想啥', prompt: `等待时, 应该提前思考什么 (期望的 dead 比例 / 哪些层应该最活跃 / 最值得 protect 的层)? 用 2-3 句作为 brain 主动给思考材料.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调当前 heatmap 数据正在本机内存里组装, 0 次云调用.` },
      ];
    }
    return [
      { label: '📊 这套 profile 健康吗', prompt: `全局失活 ${(caps.globalDeadRatio * 100).toFixed(1)}% (${caps.deadRatioBucket}), 方差 ${caps.varianceBucket}. 用 2-3 句作为 brain 评估: 这个 profile 能用来 prune 吗, 还是要先重新跑.` },
      { label: '🔍 异常层', prompt: `${caps.deadestLayer ? `最 dead 是 L${caps.deadestLayer.layerIdx} (${(caps.deadestLayer.deadRatio * 100).toFixed(1)}%)` : ''}${caps.liveliestLayer ? `, 最 alive 是 L${caps.liveliestLayer.layerIdx} (${(caps.liveliestLayer.deadRatio * 100).toFixed(1)}%)` : ''}. 这个差距正常吗, 给一个解读.` },
      { label: '⚙️ 阈值 sweep 解读', prompt: `阈值 sweep 显示 @θ=0.01/0.05/0.1/0.5 时 dead 比例不同. 这个曲线是什么形状告诉你模型 health (陡升 = 阈值边缘多 / 平稳 = 真 dead / 全 dead = 病态)?` },
      { label: '✂️ prune 起手建议', prompt: `基于这个 heatmap, 推荐 prune 的起始 threshold 和保护层. 用 2-3 句给具体数字.` },
    ];
  }
  // English
  if (!caps.hasProfile) {
    return [
      { label: '🎯 What is the heatmap', prompt: `In 2-3 sentences as brain, explain the activation heatmap (per-layer × per-neuron activity), and what extra info it gives over a single dead-ratio number.` },
      { label: '📊 How to profile', prompt: `In 2-3 sentences, explain how to run an activation profile: which prompts (representative), how many runs (>=10), which metrics matter.` },
      { label: '🔍 What to look for', prompt: `What patterns to look for (uniform dead / single-layer dead / middle peak)? What model problems do they suggest.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: the profile runs locally, the heatmap renders locally, zero cloud calls.` },
    ];
  }
  if (caps.runPhase === 'loading') {
    return [
      { label: '⏱️ Why so slow', prompt: `Roughly how big is the heatmap data and why is it slow to render (e.g. 32 layers × 14336 neurons = a few MB transfer + render). In 2-3 sentences.` },
      { label: '🎯 What to think while waiting', prompt: `While waiting, what should the user think about (expected dead ratio / which layers should be most active / which to protect)? In 2-3 sentences as brain.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: the heatmap data is being assembled in local memory, zero cloud calls.` },
    ];
  }
  return [
    { label: '📊 Is this profile healthy', prompt: `Global ${(caps.globalDeadRatio * 100).toFixed(1)}% dead (${caps.deadRatioBucket}), variance ${caps.varianceBucket}. In 2-3 sentences as brain, assess: is this profile fit for pruning or should it be re-run.` },
    { label: '🔍 Anomalous layers', prompt: `${caps.deadestLayer ? `Most dead is L${caps.deadestLayer.layerIdx} (${(caps.deadestLayer.deadRatio * 100).toFixed(1)}%)` : ''}${caps.liveliestLayer ? `, most alive is L${caps.liveliestLayer.layerIdx} (${(caps.liveliestLayer.deadRatio * 100).toFixed(1)}%)` : ''}. Is this gap normal — give an interpretation.` },
    { label: '⚙️ Threshold sweep read', prompt: `Sweep shows different dead ratios at θ=0.01/0.05/0.1/0.5. What shape does this curve tell you about model health (steep = border-zone / flat = truly dead / all dead = pathological)?` },
    { label: '✂️ Pruning starter', prompt: `Based on this heatmap, recommend a starting threshold and protected layers for pruning. In 2-3 sentences with concrete numbers.` },
  ];
}
