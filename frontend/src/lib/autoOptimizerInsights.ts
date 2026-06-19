// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * autoOptimizerInsights — derived search-space + Pareto-result helpers for
 * the /auto-optimizer page (automatic optimization parameter sweep).
 *
 * Differs from /pipeline (which is manual stack building):
 *  - /pipeline = user picks ops + params, we assess risk, run as configured
 *  - /auto-optimizer = user picks search SPACE (bits × layers × quality),
 *    backend sweeps Pareto frontier, returns N candidates ranked by Pareto
 *    optimality; we narrate the result + recommend.
 *
 * Both share §9.1 multi-component pattern; this one has 3 search dims +
 * a result-set summary.
 *
 * Sovereignty (§9.2 mandatory): the search runs locally — no cloud HPO
 * service, no telemetry on which candidates were tried.
 */
import type { ModelInfo, SearchResult, SearchCandidate } from '@/api/types';

type Locale = 'en' | 'zh';

export type SearchSizeBucket = 'tiny' | 'small' | 'medium' | 'large' | 'huge';

export interface AutoOptCapabilities {
  /** Selected bit widths (subset of [3,4,6,8]). */
  targetBits: number[];
  /** 0..N — max layers we let the search consider removing. */
  maxLayersRemove: number;
  /** 0..1 — minimum quality preservation we accept. */
  qualityFloor: number;
  /** Target device name (drives memory fit). */
  deviceName: string;
  /** Whether the user has activation profile loaded (page enforces this). */
  hasProfile: boolean;
  /** Loaded brain LLM. */
  brain: ModelInfo | null;
  /** Last search result (null = not yet run). */
  result: SearchResult | null;
  /** Currently-selected candidate. */
  selectedCandidate: SearchCandidate | null;

  // ── Derived ────────────────────────────────────────────────────────────
  /** Approximate search-space size = |bits| × (max_layers + 1). */
  searchSpaceSize: number;
  searchSizeBucket: SearchSizeBucket;
  /** Whether targetBits contains anything <= 3. */
  hasAggressiveQuant: boolean;
  /** Whether qualityFloor is very strict (>= 0.95). */
  isStrictQuality: boolean;
  /** Empty / running / done. */
  runPhase: 'noResult' | 'hasResult';
  /** Best-speedup candidate (max speedup_pct on the Pareto frontier). */
  bestSpeedup: SearchCandidate | null;
  /** Best-size candidate (max size_reduction_pct on Pareto). */
  bestSize: SearchCandidate | null;
  /** Highest-quality candidate (max quality_pct on Pareto). */
  bestQuality: SearchCandidate | null;
  /** Pareto count / fits-device count. */
  paretoCount: number;
  fitsDeviceCount: number;
  candidateCount: number;
}

function bucketSize(n: number): SearchSizeBucket {
  if (n <= 3) return 'tiny';
  if (n <= 10) return 'small';
  if (n <= 30) return 'medium';
  if (n <= 80) return 'large';
  return 'huge';
}

export function deriveAutoOptCapabilities(
  targetBits: number[],
  maxLayersRemove: number,
  qualityFloor: number,
  deviceName: string,
  hasProfile: boolean,
  brain: ModelInfo | null,
  result: SearchResult | null,
  selectedCandidate: SearchCandidate | null,
): AutoOptCapabilities {
  const searchSpaceSize = Math.max(0, targetBits.length) * Math.max(1, maxLayersRemove + 1);

  const pareto = result?.pareto_frontier ?? [];
  const bestSpeedup = pareto.reduce<SearchCandidate | null>(
    (acc, c) => (!acc || (c.speedup_pct ?? 0) > (acc.speedup_pct ?? 0) ? c : acc),
    null,
  );
  const bestSize = pareto.reduce<SearchCandidate | null>(
    (acc, c) => (!acc || (c.size_reduction_pct ?? 0) > (acc.size_reduction_pct ?? 0) ? c : acc),
    null,
  );
  const bestQuality = pareto.reduce<SearchCandidate | null>(
    (acc, c) => (!acc || (c.quality_pct ?? 0) > (acc.quality_pct ?? 0) ? c : acc),
    null,
  );

  return {
    targetBits,
    maxLayersRemove,
    qualityFloor,
    deviceName,
    hasProfile,
    brain,
    result,
    selectedCandidate,
    searchSpaceSize,
    searchSizeBucket: bucketSize(searchSpaceSize),
    hasAggressiveQuant: targetBits.some((b) => b <= 3),
    isStrictQuality: qualityFloor >= 0.95,
    runPhase: result ? 'hasResult' : 'noResult',
    bestSpeedup,
    bestSize,
    bestQuality,
    paretoCount: pareto.length,
    fitsDeviceCount: result?.fits_device_count ?? 0,
    candidateCount: result?.candidates.length ?? 0,
  };
}

export type AutoOptRiskLevel = 'safe' | 'caution' | 'danger';
export interface AutoOptRisk {
  level: AutoOptRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Per-(search config) risk:
 *  - caution: targetBits empty (search will return nothing)
 *  - caution: includes 3-bit (likely outside Pareto frontier — wastes time)
 *  - caution: search space huge (>80) — sweep takes long
 *  - caution: qualityFloor >= 0.95 (very strict, risk empty Pareto)
 *  - safe:    sane bit selection, reasonable space, soft quality floor
 */
export function assessAutoOptConfig(caps: AutoOptCapabilities): AutoOptRisk {
  if (caps.targetBits.length === 0) {
    return {
      level: 'caution',
      reason: 'No target bits selected — pick at least one (4 / 6 / 8).',
      reasonZh: '没选目标位宽 — 至少选一个 (4 / 6 / 8).',
    };
  }
  if (caps.hasAggressiveQuant) {
    return {
      level: 'caution',
      reason: '3-bit included — usually below the Pareto frontier (heavy quality loss for marginal size win). Consider 4-bit minimum.',
      reasonZh: '勾了 3-bit — 通常在 Pareto 前沿之下 (质量损失大, 体积收益小). 推荐最低 4-bit.',
    };
  }
  if (caps.searchSpaceSize > 80) {
    return {
      level: 'caution',
      reason: `Search space ~${caps.searchSpaceSize} candidates is huge — sweep can take minutes.`,
      reasonZh: `搜索空间 ~${caps.searchSpaceSize} 个候选偏大 — 可能要几分钟.`,
    };
  }
  if (caps.isStrictQuality) {
    return {
      level: 'caution',
      reason: `qualityFloor=${caps.qualityFloor.toFixed(2)} is very strict — Pareto frontier may be empty.`,
      reasonZh: `qualityFloor=${caps.qualityFloor.toFixed(2)} 很严 — 可能 Pareto 前沿为空.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.searchSpaceSize} candidates · floor ${(caps.qualityFloor * 100).toFixed(0)}% · ready to sweep.`,
    reasonZh: `${caps.searchSpaceSize} 个候选 · 质量底线 ${(caps.qualityFloor * 100).toFixed(0)}% · 可以扫描.`,
  };
}

export function buildAutoOptContextSnippet(
  caps: AutoOptCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的 Auto Optimizer 搜索` : `## YOUR AUTO-OPTIMIZER SWEEP`,
    locale === 'zh'
      ? `用户在用 Pareto 搜索找最佳量化+剪枝组合. 你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 是 brain, 帮用户解读 Pareto 前沿 + 推荐. 第一人称.`
      : `User is sweeping Pareto frontier of quant+prune combos. You (${caps.brain?.model_name ?? 'the loaded LLM'}) are the brain; explain results + recommend. First person.`,
    `- Target device: ${caps.deviceName}`,
    `- Search space: bits=[${caps.targetBits.join(', ')}] × max_layers_remove=${caps.maxLayersRemove} → ~${caps.searchSpaceSize} candidates (${caps.searchSizeBucket})`,
    `- Quality floor: ${(caps.qualityFloor * 100).toFixed(0)}%${caps.isStrictQuality ? ' (strict)' : ''}`,
    `- Aggressive quant (≤3-bit): ${caps.hasAggressiveQuant ? 'YES' : 'NO'}`,
    caps.runPhase === 'hasResult' && caps.result
      ? `- Result: ${caps.candidateCount} candidates · ${caps.paretoCount} on Pareto · ${caps.fitsDeviceCount} fit ${caps.deviceName}`
      : `- Result: not yet run`,
  ];
  if (caps.runPhase === 'hasResult' && caps.result) {
    if (caps.bestSpeedup) lines.push(`- Best speedup: ${caps.bestSpeedup.label} (+${(caps.bestSpeedup.speedup_pct ?? 0).toFixed(0)}% vs base, quality ${((caps.bestSpeedup.quality_pct ?? 0) * 100).toFixed(0)}%)`);
    if (caps.bestSize) lines.push(`- Smallest: ${caps.bestSize.label} (-${(caps.bestSize.size_reduction_pct ?? 0).toFixed(0)}% size, quality ${((caps.bestSize.quality_pct ?? 0) * 100).toFixed(0)}%)`);
    if (caps.bestQuality) lines.push(`- Highest quality: ${caps.bestQuality.label} (quality ${((caps.bestQuality.quality_pct ?? 0) * 100).toFixed(0)}%)`);
  }
  if (caps.selectedCandidate) {
    lines.push('', `### User-selected candidate: ${caps.selectedCandidate.label}`);
    lines.push(`- bits=${caps.selectedCandidate.bits ?? '?'} · layers_removed=${caps.selectedCandidate.layers_removed ?? 0}`);
    lines.push(`- speedup ${(caps.selectedCandidate.speedup_pct ?? 0).toFixed(0)}% · size -${(caps.selectedCandidate.size_reduction_pct ?? 0).toFixed(0)}% · quality ${((caps.selectedCandidate.quality_pct ?? 0) * 100).toFixed(0)}%`);
  }
  lines.push('');
  lines.push(locale === 'zh'
    ? `### 北极星 §1 主权: 整个搜索本地跑, 不调任何云端 HPO / AutoML 服务.`
    : `### North-star §1 sovereignty: full sweep local, no cloud HPO / AutoML service.`);
  return lines.filter(Boolean).join('\n');
}

export function buildAutoOptAutoBrief(
  caps: AutoOptCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (caps.runPhase === 'noResult') {
      return `用户在 Auto Optimizer 配搜索空间 (bits=[${caps.targetBits.join(', ')}], max_layers=${caps.maxLayersRemove}, floor=${(caps.qualityFloor * 100).toFixed(0)}%). 用 2-3 句作为 brain 评估这个空间合不合理 (太大/太小/有 3-bit/质量底线严不严), 第一人称.`;
    }
    if (caps.paretoCount === 0) {
      return `搜索完成但 Pareto 前沿为空 (${caps.candidateCount} 个候选都不过质量底线 ${(caps.qualityFloor * 100).toFixed(0)}%). 用 2-3 句作为 brain 解释为什么 + 推荐怎么调 (降 floor / 加 4-bit / 减 max_layers).`;
    }
    return `Pareto 前沿 ${caps.paretoCount} 个 (${caps.fitsDeviceCount} 个装得下 ${caps.deviceName}). 用 2-3 句作为 brain 推荐用户该选哪个 (best-speedup / best-size / best-quality 三选一根据用户场景), 引用具体数字.`;
  }
  if (caps.runPhase === 'noResult') {
    return `User configuring search space (bits=[${caps.targetBits.join(', ')}], max_layers=${caps.maxLayersRemove}, floor=${(caps.qualityFloor * 100).toFixed(0)}%). In 2-3 sentences as brain, assess if this space is sane (too big/small/3-bit/strict floor). First person.`;
  }
  if (caps.paretoCount === 0) {
    return `Sweep done but Pareto empty (${caps.candidateCount} candidates failed floor ${(caps.qualityFloor * 100).toFixed(0)}%). In 2-3 sentences as brain, explain why + recommend tuning (lower floor / add 4-bit / reduce max_layers).`;
  }
  return `Pareto frontier has ${caps.paretoCount} candidates (${caps.fitsDeviceCount} fit ${caps.deviceName}). In 2-3 sentences as brain, recommend which to pick (best-speedup / best-size / best-quality) by use case. Cite numbers.`;
}

export function getAutoOptSuggestedPrompts(
  caps: AutoOptCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.runPhase === 'noResult') {
      return [
        { label: '🎯 配多大空间合理', prompt: `当前搜索空间 ~${caps.searchSpaceSize} 个候选. 多大算合理 (扫得过来 + 不漏 sweet spot)? 推荐 bits 和 max_layers 各取多少, 第一人称给具体数字.` },
        { label: '⚖️ 质量底线怎么设', prompt: `用 2-3 句话解释 qualityFloor 是什么 (PPL 保留率 / similarity / token-level recall 之类), 当前 ${(caps.qualityFloor * 100).toFixed(0)}% 严不严, 推荐起步用什么值.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句话作为 brain 解释: 为什么这条 sweep 走本地比 OptunaCloud / W&B Sweep 这种云端 HPO 更值得 (隐私 / 速度 / 数据不出 device).` },
        { label: '⚡ 自动 vs 手动 pipeline', prompt: `用 2-3 句话对比 Auto Optimizer vs /pipeline 手动配置: 各自适合什么场景, 新手该用哪个.` },
      ];
    }
    if (caps.paretoCount === 0) {
      return [
        { label: '🚨 为什么前沿是空', prompt: `${caps.candidateCount} 个候选, Pareto 0 个. 用 2-3 句作为 brain 直接告诉用户最可能的根因 (floor 太严 / max_layers 太狠 / 3-bit 拖后腿) + 给一个具体调整方案.` },
        { label: '🔧 怎么调能出结果', prompt: `给一个具体的"调 3 个参数中的 1 个"实验路径: 第一步先把 floor 从 ${(caps.qualityFloor * 100).toFixed(0)}% 降到多少 / 还是先去 3-bit / 还是先减 max_layers. 推荐顺序.` },
        { label: '⚠️ 这个模型能不能这么压', prompt: `${caps.brain?.model_name || '我'} 是不是 architecturally 就不适合这种激进压缩 (例如 hybrid attention / GQA / 已经 4-bit 了再压会崩). 第一人称承认.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句话强调: 即使搜索失败, 我们也不会上报"哪些 config 失败"到云端. 这条数据不出 Mac.` },
      ];
    }
    return [
      { label: '🏆 选哪个最好', prompt: `Pareto 前沿 ${caps.paretoCount} 个候选. ${caps.bestSpeedup ? `Best-speedup ${caps.bestSpeedup.label} +${(caps.bestSpeedup.speedup_pct ?? 0).toFixed(0)}%, ` : ''}${caps.bestSize ? `Best-size ${caps.bestSize.label} -${(caps.bestSize.size_reduction_pct ?? 0).toFixed(0)}%, ` : ''}${caps.bestQuality ? `Best-quality ${caps.bestQuality.label} ${((caps.bestQuality.quality_pct ?? 0) * 100).toFixed(0)}%` : ''}. 作为 brain 推荐: 用户在 ${caps.deviceName} 上做日常聊天, 该选哪个? 直接给答案.` },
      { label: '⚖️ Pareto 解读', prompt: `用 2-3 句话向新手解释 Pareto 前沿是什么 (没有任何一个候选能在 3 个维度都优于另一个), 以及为什么我们只看前沿不看所有 ${caps.candidateCount} 个.` },
      { label: '📦 选完怎么落地', prompt: `用户选好 ${caps.selectedCandidate?.label || 'candidate'} 之后, 用 2-3 句话告诉他们落地路径 (导出 candidate config → 调 /pipeline 跑 → benchmark verify → 推 iPhone).` },
      { label: '🎯 为什么不 fit 设备', prompt: caps.fitsDeviceCount < caps.paretoCount
        ? `${caps.paretoCount - caps.fitsDeviceCount} 个 Pareto 候选装不下 ${caps.deviceName}. 用 2-3 句作为 brain 解释为什么 (RAM 限制 / KV cache 没算够). 推荐换设备还是改 config.`
        : `所有 ${caps.paretoCount} 个 Pareto 候选都装得下 ${caps.deviceName}. 用 2-3 句话作为 brain 解释这意味着什么 (设备选得宽裕 / 还能往更大压缩比走).` },
    ];
  }
  // English
  if (caps.runPhase === 'noResult') {
    return [
      { label: '🎯 How big should the space be', prompt: `Current ~${caps.searchSpaceSize} candidates. What size is sensible (sweep finishes + still finds sweet spot)? Recommend bits and max_layers values. First person, numbers.` },
      { label: '⚖️ How to set quality floor', prompt: `In 2-3 sentences, explain what qualityFloor measures (PPL retention / similarity / token recall), is ${(caps.qualityFloor * 100).toFixed(0)}% strict, what's a good starting value.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences as brain, explain why this sweep being local beats cloud HPO (Optuna Cloud / W&B Sweep) — privacy / speed / data stays on device.` },
      { label: '⚡ Auto vs manual', prompt: `In 2-3 sentences, compare Auto Optimizer vs /pipeline manual config: which fits which scenario, beginner pick.` },
    ];
  }
  if (caps.paretoCount === 0) {
    return [
      { label: '🚨 Why empty', prompt: `${caps.candidateCount} candidates, 0 on Pareto. In 2-3 sentences as brain, give the most likely root cause (floor too strict / max_layers too aggressive / 3-bit dragging) + concrete tuning.` },
      { label: '🔧 Tuning path', prompt: `Give a concrete 1-knob-at-a-time experiment plan: first lower floor from ${(caps.qualityFloor * 100).toFixed(0)}% to what / or drop 3-bit / or reduce max_layers. Order matters.` },
      { label: '⚠️ Architecture limit', prompt: `Is ${caps.brain?.model_name || 'me'} architecturally unsuited to this aggressive squeeze (hybrid attention / GQA / already 4-bit)? Admit honestly in first person.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: even on failure, we don't report "which configs failed" to cloud — that data stays local.` },
    ];
  }
  return [
    { label: '🏆 Which to pick', prompt: `Pareto has ${caps.paretoCount} candidates. ${caps.bestSpeedup ? `Best-speedup ${caps.bestSpeedup.label} +${(caps.bestSpeedup.speedup_pct ?? 0).toFixed(0)}%, ` : ''}${caps.bestSize ? `Best-size ${caps.bestSize.label} -${(caps.bestSize.size_reduction_pct ?? 0).toFixed(0)}%, ` : ''}${caps.bestQuality ? `Best-quality ${caps.bestQuality.label} ${((caps.bestQuality.quality_pct ?? 0) * 100).toFixed(0)}%` : ''}. As brain, for daily chat on ${caps.deviceName}, which? Direct answer.` },
    { label: '⚖️ Pareto explanation', prompt: `In 2-3 sentences, explain to a beginner what Pareto frontier is (no candidate dominates another on all 3 dims), and why we only look at frontier vs all ${caps.candidateCount}.` },
    { label: '📦 How to ship', prompt: `Once user picks ${caps.selectedCandidate?.label || 'candidate'}, in 2-3 sentences tell them the path (export config → run /pipeline → benchmark verify → push to iPhone).` },
    { label: '🎯 Device fit', prompt: caps.fitsDeviceCount < caps.paretoCount
      ? `${caps.paretoCount - caps.fitsDeviceCount} Pareto candidates don't fit ${caps.deviceName}. In 2-3 sentences as brain, explain why (RAM cap / underestimated KV) and recommend bigger device or tweaked config.`
      : `All ${caps.paretoCount} Pareto candidates fit ${caps.deviceName}. In 2-3 sentences as brain, explain what that means (room to push compression further).` },
  ];
}
