// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * pruningInsights — derived pruning-config + per-layer cohort state + chat
 * helpers for the /pruning page (PruningSimulator.tsx).
 *
 * Pruning takes an activation profile (`ProfileSummary` from the dashboard)
 * and proposes which FFN neurons to drop. The page:
 *  - Threshold slider — neurons with mean activation below this are "dead"
 *  - Max reduction — cap per-layer pruning ratio
 *  - Min intermediate — never prune below this size
 *  - Protected layers — explicit layer indices to leave untouched
 *
 * §9.1 N-layer cohort: every layer is a slot; aggregate capability =
 *  retention histogram + cliff detection on the threshold sweep.
 *
 * §10.3 naming: capability uses `runPhase` not `status`.
 *
 * Sovereignty (§9.2 mandatory): pruning simulation is pure local activation
 * profile arithmetic — no inference traffic to outside services.
 */
import type { ModelInfo, PruneSimResponse, ThresholdSweepPoint, ProfileSummary } from '@/api/types';

type Locale = 'en' | 'zh';

export type PrunePhase = 'noProfile' | 'noModel' | 'idle' | 'computed';

export type RetentionBucket =
  | 'extreme'      // < 30% kept — model gutted
  | 'aggressive'   // 30..60%
  | 'moderate'     // 60..85%
  | 'conservative'; // >= 85%

export type SavingsBucket = 'none' | 'trim' | 'modest' | 'strong' | 'extreme';

/** Macro shape of per-layer retention. */
export type RetentionPattern =
  | 'uniform'        // all layers within ±5%
  | 'cliff'          // sharp drop after threshold
  | 'edges_protected' // first/last layers >> middle
  | 'mixed';

export interface PruningCapabilities {
  // ── Inputs ─────────────────────────────────────────────────────────────
  threshold: number;
  maxReduction: number;
  minSize: number;
  protectedLayers: number[];
  profileSummary: ProfileSummary | null;
  result: PruneSimResponse | null;
  sweepData: ThresholdSweepPoint[];
  brain: ModelInfo | null;

  // ── Derived gating ─────────────────────────────────────────────────────
  hasProfile: boolean;
  hasModel: boolean;
  hasResult: boolean;
  runPhase: PrunePhase;

  // ── Profile metadata (when present) ────────────────────────────────────
  numLayers: number;
  intermediateSize: number;
  deadRatio: number; // baseline dead neuron ratio at θ=0.1
  protectedCount: number;
  effectiveLayers: number; // numLayers - protectedCount

  // ── Cohort derived (after sim runs) ────────────────────────────────────
  /** Overall retention 0..1. */
  globalRetention: number;
  retentionBucket: RetentionBucket;
  savingsRatio: number; // 0..1, fraction of MLP saved
  savingsBucket: SavingsBucket;
  /** Layer with highest retention (likely protected or stable). */
  bestRetentionLayer: number | null;
  /** Layer with lowest retention (most pruned). */
  worstRetentionLayer: number | null;
  /** Number of layers where the cap (max_reduction) hit. */
  cappedLayerCount: number;
  pattern: RetentionPattern;

  // ── Sweep analysis ─────────────────────────────────────────────────────
  /** Threshold where retention drops below 50% (the "cliff"), null if none. */
  cliffThreshold: number | null;
  /** Whether current threshold is past the cliff. */
  pastCliff: boolean;
}

function detectPattern(layers: PruneSimResponse['layers']): RetentionPattern {
  if (layers.length === 0) return 'uniform';
  const retentions = layers.map((l) => l.retention);
  const min = Math.min(...retentions);
  const max = Math.max(...retentions);
  if (max - min < 0.05) return 'uniform';

  // edges_protected: layer 0 and last layer have retention >= avg + 10% margin
  const n = retentions.length;
  const avg = retentions.reduce((a, b) => a + b, 0) / n;
  const first = retentions[0];
  const last = retentions[n - 1];
  if (first - avg > 0.1 && last - avg > 0.1) {
    const mid = retentions.slice(Math.floor(n * 0.25), Math.ceil(n * 0.75));
    const midAvg = mid.reduce((a, b) => a + b, 0) / mid.length;
    if (first > midAvg && last > midAvg) return 'edges_protected';
  }

  // cliff: > 25% spread between adjacent layers
  for (let i = 1; i < n; i++) {
    if (Math.abs(retentions[i] - retentions[i - 1]) > 0.25) return 'cliff';
  }

  return 'mixed';
}

function bucketRetention(r: number): RetentionBucket {
  if (r < 0.3) return 'extreme';
  if (r < 0.6) return 'aggressive';
  if (r < 0.85) return 'moderate';
  return 'conservative';
}

function bucketSavings(ratio: number): SavingsBucket {
  if (ratio <= 0) return 'none';
  if (ratio < 0.2) return 'trim';
  if (ratio < 0.4) return 'modest';
  if (ratio < 0.6) return 'strong';
  return 'extreme';
}

export function derivePruningCapabilities(
  threshold: number,
  maxReduction: number,
  minSize: number,
  protectedLayers: number[],
  profileSummary: ProfileSummary | null,
  result: PruneSimResponse | null,
  sweepData: ThresholdSweepPoint[],
  brain: ModelInfo | null,
): PruningCapabilities {
  const hasProfile = !!profileSummary;
  const hasModel = !!brain;
  const hasResult = !!result;

  let runPhase: PrunePhase;
  if (!hasModel) runPhase = 'noModel';
  else if (!hasProfile) runPhase = 'noProfile';
  else if (!hasResult) runPhase = 'idle';
  else runPhase = 'computed';

  const numLayers = profileSummary?.num_layers ?? 0;
  const intermediateSize = profileSummary?.intermediate_size ?? 0;
  const deadRatio = profileSummary?.dead_ratio_at_01 ?? 0;
  const protectedCount = protectedLayers.length;
  const effectiveLayers = Math.max(0, numLayers - protectedCount);

  const globalRetention = result?.retention ?? 1.0;
  const totalOriginal = result?.total_original ?? 1;
  const totalRemoved = result?.total_removed ?? 0;
  const savingsRatio = totalOriginal > 0 ? totalRemoved / totalOriginal : 0;

  let bestRetentionLayer: number | null = null;
  let worstRetentionLayer: number | null = null;
  let bestRet = -Infinity;
  let worstRet = Infinity;
  let cappedLayerCount = 0;
  if (result) {
    for (const l of result.layers) {
      // detect cap-hit: removed/original close to maxReduction
      const reductionRatio = l.original_size > 0 ? l.removed / l.original_size : 0;
      if (Math.abs(reductionRatio - maxReduction) < 0.01) cappedLayerCount += 1;
      if (l.retention > bestRet && !l.is_protected) {
        bestRet = l.retention;
        bestRetentionLayer = l.layer_idx;
      }
      if (l.retention < worstRet && !l.is_protected) {
        worstRet = l.retention;
        worstRetentionLayer = l.layer_idx;
      }
    }
  }

  // Sweep cliff: threshold where retention first drops below 0.5
  let cliffThreshold: number | null = null;
  if (sweepData.length > 0) {
    const sorted = [...sweepData].sort((a, b) => a.threshold - b.threshold);
    for (const p of sorted) {
      if (p.retention < 0.5) {
        cliffThreshold = p.threshold;
        break;
      }
    }
  }
  const pastCliff = cliffThreshold !== null && threshold > cliffThreshold;

  return {
    threshold,
    maxReduction,
    minSize,
    protectedLayers,
    profileSummary,
    result,
    sweepData,
    brain,
    hasProfile,
    hasModel,
    hasResult,
    runPhase,
    numLayers,
    intermediateSize,
    deadRatio,
    protectedCount,
    effectiveLayers,
    globalRetention,
    retentionBucket: bucketRetention(globalRetention),
    savingsRatio,
    savingsBucket: bucketSavings(savingsRatio),
    bestRetentionLayer,
    worstRetentionLayer,
    cappedLayerCount,
    pattern: result ? detectPattern(result.layers) : 'uniform',
    cliffThreshold,
    pastCliff,
  };
}

export type PruneRiskLevel = 'safe' | 'caution' | 'danger';
export interface PruneRisk {
  level: PruneRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Risk hierarchy:
 *  - danger: no profile (cannot simulate)
 *  - danger: maxReduction > 0.75 (gut model)
 *  - danger: retention < 30% (output likely broken)
 *  - caution: pastCliff (threshold past sharp drop)
 *  - caution: protectedCount == 0 in deep model (>20 layers)
 *  - caution: minSize > intermediate_size/2 (defeats the purpose)
 *  - safe: balanced config
 */
export function assessPruning(caps: PruningCapabilities): PruneRisk {
  if (!caps.hasModel) {
    return { level: 'safe', reason: 'No model loaded.', reasonZh: '尚未加载模型.' };
  }
  if (!caps.hasProfile) {
    return {
      level: 'danger',
      reason: 'No activation profile loaded. Pruning cannot run without one — go to the dashboard, run an activation profile on representative prompts, then return.',
      reasonZh: '没加载激活 profile. 没有 profile 不能 prune — 去 dashboard 用代表性 prompts 跑一次, 再回来.',
    };
  }
  if (caps.maxReduction > 0.75) {
    return {
      level: 'danger',
      reason: `Max reduction ${(caps.maxReduction * 100).toFixed(0)}% per layer is gut-the-model territory. Drop to ≤ 60% unless you've validated.`,
      reasonZh: `单层最大裁剪 ${(caps.maxReduction * 100).toFixed(0)}% 是掏空模型级别. 除非已验证, 降到 ≤ 60%.`,
    };
  }
  if (caps.hasResult && caps.globalRetention < 0.3) {
    return {
      level: 'danger',
      reason: `Global retention ${(caps.globalRetention * 100).toFixed(1)}% is below 30% — output will likely be incoherent. Lower threshold or raise minSize.`,
      reasonZh: `全局保留率 ${(caps.globalRetention * 100).toFixed(1)}% 低于 30% — 输出大概率不连贯. 降阈值或提高 minSize.`,
    };
  }
  if (caps.pastCliff && caps.cliffThreshold !== null) {
    return {
      level: 'caution',
      reason: `Threshold ${caps.threshold.toFixed(3)} sits past the retention cliff at θ=${caps.cliffThreshold.toFixed(3)}. Sweep curve shows retention drops below 50% beyond this point — pull threshold back.`,
      reasonZh: `阈值 ${caps.threshold.toFixed(3)} 已越过保留率断崖 (θ=${caps.cliffThreshold.toFixed(3)}). Sweep 曲线显示越过此点保留率跌破 50% — 把阈值往回调.`,
    };
  }
  if (caps.protectedCount === 0 && caps.numLayers > 20) {
    return {
      level: 'caution',
      reason: `${caps.numLayers}-layer model with no protected layers. Embedding-adjacent (layer 0) and lm_head-adjacent (layer ${caps.numLayers - 1}) are usually worth protecting — try "0,${caps.numLayers - 1}".`,
      reasonZh: `${caps.numLayers} 层模型且没有保护层. 紧邻 embedding (Layer 0) 和 lm_head (Layer ${caps.numLayers - 1}) 通常值得保护 — 试试 "0,${caps.numLayers - 1}".`,
    };
  }
  if (caps.intermediateSize > 0 && caps.minSize > caps.intermediateSize * 0.5) {
    return {
      level: 'caution',
      reason: `minSize ${caps.minSize} is > 50% of intermediate ${caps.intermediateSize}. Cap binds early — most layers won't get pruned. Drop minSize to see savings.`,
      reasonZh: `minSize ${caps.minSize} 大于中间维 ${caps.intermediateSize} 的 50%. 提前命中下限, 大多数层不会裁 — 降低 minSize 才能看出收益.`,
    };
  }
  if (caps.hasResult && caps.cappedLayerCount > caps.numLayers * 0.5) {
    return {
      level: 'caution',
      reason: `${caps.cappedLayerCount}/${caps.numLayers} layers hit the max-reduction cap — config is bound by the cap, not data. Raise maxReduction if you want more savings, or accept that data limits prevent it.`,
      reasonZh: `${caps.cappedLayerCount}/${caps.numLayers} 层撞到 max-reduction 上限 — 配置被上限锁住, 不是数据本身. 想多省就抬 maxReduction, 或者接受数据限制.`,
    };
  }
  return {
    level: 'safe',
    reason: caps.hasResult
      ? `Retention ${(caps.globalRetention * 100).toFixed(1)}%, saved ${(caps.savingsRatio * 100).toFixed(1)}%, pattern ${caps.pattern}.`
      : `Profile loaded (${caps.numLayers} layers, ${(caps.deadRatio * 100).toFixed(1)}% dead at θ=0.1). Adjust controls to simulate.`,
    reasonZh: caps.hasResult
      ? `保留 ${(caps.globalRetention * 100).toFixed(1)}%, 省 ${(caps.savingsRatio * 100).toFixed(1)}%, 模式 ${caps.pattern}.`
      : `Profile 已加载 (${caps.numLayers} 层, θ=0.1 下 ${(caps.deadRatio * 100).toFixed(1)}% 失活). 调控件开始模拟.`,
  };
}

export function patternLabel(p: RetentionPattern, locale: Locale): string {
  const map: Record<RetentionPattern, [string, string]> = {
    uniform: ['Uniform', '均匀'],
    cliff: ['Cliff', '断崖'],
    edges_protected: ['Edges-protected', '两端保护'],
    mixed: ['Mixed', '混合'],
  };
  return locale === 'zh' ? map[p][1] : map[p][0];
}

export function retentionBucketLabel(b: RetentionBucket, locale: Locale): string {
  const map: Record<RetentionBucket, [string, string]> = {
    extreme: ['Extreme', '极端'],
    aggressive: ['Aggressive', '激进'],
    moderate: ['Moderate', '适度'],
    conservative: ['Conservative', '保守'],
  };
  return locale === 'zh' ? map[b][1] : map[b][0];
}

export function buildPruningContextSnippet(
  caps: PruningCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的剪枝模拟器 (Pruning Simulator)` : `## YOUR PRUNING SIMULATOR`,
    locale === 'zh'
      ? `这条流程基于激活 profile 模拟 FFN 神经元剪枝. 你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 是 brain, 用第一人称解释这套配置 + 风险 + 收益.`
      : `This flow simulates FFN neuron pruning from an activation profile. You (${caps.brain?.model_name ?? 'the loaded LLM'}) are the brain; explain the config + risks + gains in first person.`,
    `- Run phase: ${caps.runPhase}`,
    !caps.hasProfile
      ? (locale === 'zh' ? `- ⚠ 还没有 profile — 必须先 dashboard 跑 activation profile.` : `- ⚠ no profile yet — user must first run an activation profile from the dashboard.`)
      : `- Profile: ${caps.numLayers} layers, intermediate ${caps.intermediateSize}, dead ratio ${(caps.deadRatio * 100).toFixed(1)}% @ θ=0.1`,
    `- Config: threshold=${caps.threshold.toFixed(3)} · max_reduction=${(caps.maxReduction * 100).toFixed(0)}% · minSize=${caps.minSize} · protected=[${caps.protectedLayers.join(', ') || '(none)'}]`,
  ];

  if (caps.hasResult && caps.result) {
    lines.push(
      ``,
      locale === 'zh' ? `### 当前模拟结果 (cite when explaining):` : `### Current simulation result (cite when explaining):`,
      `- Global retention: ${(caps.globalRetention * 100).toFixed(2)}% (${caps.retentionBucket})`,
      `- Total removed: ${caps.result.total_removed.toLocaleString()} of ${caps.result.total_original.toLocaleString()} neurons`,
      `- MLP size saved: ${(caps.result.mlp_size_saved_bytes / 1e6).toFixed(1)} MB (${(caps.savingsRatio * 100).toFixed(1)}% of MLP)`,
      `- Pattern: ${caps.pattern}`,
      caps.cappedLayerCount > 0 ? `- Capped layers (hit max_reduction): ${caps.cappedLayerCount}` : '',
      caps.bestRetentionLayer !== null ? `- Best-retained non-protected layer: L${caps.bestRetentionLayer}` : '',
      caps.worstRetentionLayer !== null ? `- Worst-retained non-protected layer: L${caps.worstRetentionLayer}` : '',
    );
  }
  if (caps.cliffThreshold !== null) {
    lines.push(
      ``,
      locale === 'zh'
        ? `### 阈值 sweep 分析:`
        : `### Threshold sweep analysis:`,
      locale === 'zh'
        ? `- 断崖在 θ=${caps.cliffThreshold.toFixed(3)} (保留率首次跌破 50%)`
        : `- Retention cliff at θ=${caps.cliffThreshold.toFixed(3)} (first drop below 50%)`,
      caps.pastCliff
        ? (locale === 'zh' ? `- ⚠ 当前阈值已越过断崖` : `- ⚠ current threshold past the cliff`)
        : (locale === 'zh' ? `- 当前阈值在断崖之前 (安全侧)` : `- Current threshold before the cliff (safe side)`),
    );
  }
  lines.push(
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: 剪枝模拟纯本地 profile 运算, 0 次云调用.`
      : `### North-star §1 sovereignty: pruning simulation is pure local profile arithmetic, zero cloud calls.`,
  );
  return lines.filter(Boolean).join('\n');
}

export function buildPruningAutoBrief(
  caps: PruningCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (!caps.hasModel) {
      return `还没加载模型. 用 1-2 句作为 brain 介绍剪枝是干啥. 第一人称.`;
    }
    if (!caps.hasProfile) {
      return `还没有激活 profile. 用 2-3 句作为 brain 解释为什么 prune 必须有 profile (activation 数据决定哪个神经元死/活), 推荐用户去 dashboard 跑一次代表性 prompts. 第一人称.`;
    }
    if (caps.hasResult) {
      return `模拟跑出来 ${(caps.globalRetention * 100).toFixed(1)}% 保留率, 省 ${(caps.savingsRatio * 100).toFixed(1)}%, 模式 ${caps.pattern}. 用 2-3 句作为 brain 评估这次配置 (gut check + 模型层级影响 + 推荐). 第一人称, 引用具体数字.`;
    }
    return `Profile 已加载 (${caps.numLayers} 层, ${(caps.deadRatio * 100).toFixed(1)}% 失活). 用 2-3 句作为 brain 推荐如何起手 (起始阈值 / 保护层 / 期望收益). 第一人称.`;
  }
  if (!caps.hasModel) {
    return `No model loaded. In 1-2 sentences as brain, introduce pruning. First person.`;
  }
  if (!caps.hasProfile) {
    return `No activation profile yet. In 2-3 sentences as brain, explain why pruning needs a profile (activation data identifies dead vs alive neurons), and tell the user to run one from the dashboard with representative prompts. First person.`;
  }
  if (caps.hasResult) {
    return `Simulation gave ${(caps.globalRetention * 100).toFixed(1)}% retention, ${(caps.savingsRatio * 100).toFixed(1)}% saved, pattern ${caps.pattern}. In 2-3 sentences as brain, assess (gut check + which layers feel impact + recommendation). First person, cite numbers.`;
  }
  return `Profile loaded (${caps.numLayers} layers, ${(caps.deadRatio * 100).toFixed(1)}% dead). In 2-3 sentences as brain, recommend starting points (initial threshold / protected layers / expected savings). First person.`;
}

export function getPruningSuggestedPrompts(
  caps: PruningCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (!caps.hasProfile) {
      return [
        { label: '🎯 剪枝是干啥', prompt: `用 2-3 句作为 brain 解释 FFN 剪枝 (drop 低活跃神经元), 与量化的区别 (前者改架构, 后者改权重精度), 适合什么场景.` },
        { label: '📊 profile 怎么跑', prompt: `用 2-3 句解释怎么跑 activation profile: 用什么 prompts (代表性场景), 跑多少轮 (>=10 才稳), 哪些指标重要.` },
        { label: '🛡️ 哪些层不该剪', prompt: `通常哪几层应该 protect (Layer 0 紧邻 embedding, last layer 紧邻 lm_head)? 给一个保护层默认值建议.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 整个 profile 跑在本机, 剪枝模拟也在本机, 0 次云调用.` },
      ];
    }
    if (caps.hasResult) {
      return [
        { label: '📊 这次合理吗', prompt: `保留 ${(caps.globalRetention * 100).toFixed(1)}%, 省 ${(caps.savingsRatio * 100).toFixed(1)}%, 模式 ${caps.pattern}. 用 2-3 句作为 brain 评估: 这套配置出来的模型还能用吗, 哪些任务可能退化.` },
        { label: '🔍 模式解读', prompt: `检测到 ${caps.pattern} 模式. ${caps.pattern === 'edges_protected' ? '两端高 中间低 — 标准做法, 解释为啥这样好.' : caps.pattern === 'cliff' ? '某层断崖 — 这是哪些层在出问题, 为什么.' : caps.pattern === 'uniform' ? '层间均匀 — 这是好事还是说明配置太保守.' : '混合 — 各层独立调整.'} 用 2-3 句解读.` },
        { label: '⚖️ 试一试更激进', prompt: `如果让 maxReduction 从 ${(caps.maxReduction * 100).toFixed(0)}% 提到 60%, 估计会怎样 (省得多少, 保留率怎么变)?` },
        { label: '📦 下一步', prompt: `想把这套剪枝 apply 到模型, 用 2-3 句告诉用户接下来怎么做 (用 pipeline 应用 / benchmark 验证 / device test).` },
      ];
    }
    return [
      { label: '🎯 起手怎么调', prompt: `Profile 显示 ${caps.numLayers} 层, ${(caps.deadRatio * 100).toFixed(1)}% 失活. 起始阈值/maxReduction/protected 怎么设? 用 2-3 句给具体起手值.` },
      { label: '🛡️ 保护层默认', prompt: `${caps.numLayers} 层模型, 当前 ${caps.protectedCount === 0 ? '没保护层' : `保护了 ${caps.protectedLayers.join(',')}`}. 推荐保护清单 + 理由.` },
      { label: '📊 怎么读 sweep', prompt: `阈值 sweep 怎么读? 哪个 (threshold, retention) 点是甜点, 怎么找断崖.` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 这套剪枝模拟全程在本机, 0 次云调用 — 与"上传到云端推理优化"的对比.` },
    ];
  }
  // English
  if (!caps.hasProfile) {
    return [
      { label: '🎯 What is pruning', prompt: `In 2-3 sentences as brain, explain FFN pruning (drop low-activation neurons), vs quantization (architecture vs precision), and the sweet spot.` },
      { label: '📊 How to profile', prompt: `In 2-3 sentences, explain how to run an activation profile: which prompts (representative scenarios), how many runs (>=10 for stability), which metrics matter.` },
      { label: '🛡️ Which layers to protect', prompt: `Which layers should typically be protected (layer 0 near embeddings, last layer near lm_head)? Give a default protected-layer suggestion.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: the profile runs locally, the simulation runs locally, zero cloud calls.` },
    ];
  }
  if (caps.hasResult) {
    return [
      { label: '📊 Is this sane', prompt: `Retention ${(caps.globalRetention * 100).toFixed(1)}%, saved ${(caps.savingsRatio * 100).toFixed(1)}%, pattern ${caps.pattern}. In 2-3 sentences as brain, assess: is this still usable, which tasks may degrade.` },
      { label: '🔍 Pattern read', prompt: `Detected ${caps.pattern} pattern. ${caps.pattern === 'edges_protected' ? 'High at ends, low in middle — explain why this is good.' : caps.pattern === 'cliff' ? 'Sharp drop at some layer — which ones, why.' : caps.pattern === 'uniform' ? 'Even across layers — is this good or too conservative.' : 'Mixed across layers.'} In 2-3 sentences.` },
      { label: '⚖️ Try more aggressive', prompt: `If I bumped maxReduction from ${(caps.maxReduction * 100).toFixed(0)}% to 60%, what would happen (more savings, retention impact)?` },
      { label: '📦 Next step', prompt: `Want to apply this pruning to the model. In 2-3 sentences, tell the user what to do (apply via pipeline / benchmark / device test).` },
    ];
  }
  return [
    { label: '🎯 Starting points', prompt: `Profile shows ${caps.numLayers} layers, ${(caps.deadRatio * 100).toFixed(1)}% dead. What threshold/maxReduction/protected to start with? Give concrete starter values in 2-3 sentences.` },
    { label: '🛡️ Default protected', prompt: `${caps.numLayers}-layer model, currently ${caps.protectedCount === 0 ? 'no protected layers' : `protecting ${caps.protectedLayers.join(',')}`}. Recommend a protected list with reasoning.` },
    { label: '📊 How to read sweep', prompt: `How to read the threshold sweep? Which (threshold, retention) point is the sweet spot, how to find the cliff.` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this entire pruning simulation runs locally, zero cloud calls — vs uploading to a cloud optimization service.` },
  ];
}
