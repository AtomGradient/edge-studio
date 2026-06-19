// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * attentionInsights — derived attention-pattern cohort + chat helpers
 * for the /attention page (AttentionPatterns.tsx).
 *
 * The attention analyzer classifies each (layer, head) into one of four
 * pattern archetypes:
 *  - SINK   : attention concentrates on a few "sink" tokens (BOS, etc).
 *  - LOCAL  : attention stays within a small window (sliding-window).
 *  - GLOBAL : attention spreads across many tokens.
 *  - SPARSE : irregular / very low-entropy without clear structure.
 *
 * Backend: POST /api/model/{id}/attention/analyze, returns a
 * `pattern_matrix[L][H]` of strings + `per_layer_summary` counts.
 *
 * §9.1 N-cell cohort (L × H grid): every cell is a slot; capability
 * aggregates pattern histogram + per-layer dominance + variance signals.
 *
 * §10.3 naming: capability uses `runPhase` not `status`.
 *
 * Sovereignty (§9.2 mandatory): pattern classification is pure local
 * post-processing of the captured attention trace; zero cloud calls.
 */
import type { ModelInfo, AttentionAnalysisResponse } from '@/api/types';

type Locale = 'en' | 'zh';

export type AttentionPhase = 'noModel' | 'noTrace' | 'idle' | 'analyzing' | 'analyzed';

export type Pattern = 'sink' | 'local' | 'global' | 'sparse';
export const PATTERNS: Pattern[] = ['sink', 'local', 'global', 'sparse'];

/** Macro-shape of pattern distribution. */
export type AttentionPatternShape =
  | 'balanced'        // all 4 patterns present, no dominant
  | 'sink_heavy'      // SINK > 50%
  | 'local_heavy'     // LOCAL > 50%
  | 'global_heavy'    // GLOBAL > 50%
  | 'sparse_heavy'    // SPARSE > 50% — usually concerning
  | 'mixed';          // none dominant but not balanced

export interface AttentionCapabilities {
  // ── Inputs ─────────────────────────────────────────────────────────────
  result: AttentionAnalysisResponse | null;
  loading: boolean;
  hasTrace: boolean;
  hasAttentionTrace: boolean;
  brain: ModelInfo | null;
  numLayers: number;
  numHeads: number;

  // ── Gating ─────────────────────────────────────────────────────────────
  hasModel: boolean;
  hasResult: boolean;
  runPhase: AttentionPhase;

  // ── Cohort derived ─────────────────────────────────────────────────────
  /** Total head count (numLayers × numHeads). */
  totalHeads: number;
  /** Per-pattern global count + ratio. */
  histogram: Record<Pattern, { count: number; ratio: number }>;
  /** Most-common pattern overall. */
  dominantPattern: Pattern | null;
  /** Macro shape. */
  shape: AttentionPatternShape;
  /** Layers where SINK + LOCAL together account for > 80% (good for DSR). */
  dsrFriendlyLayers: number[];
  /** Layers where SPARSE > 50% (questionable heads). */
  sparseHeavyLayers: number[];
  /** Layers where GLOBAL > 50% (need full attention). */
  globalHeavyLayers: number[];
  /** Layer-level dominant-pattern entropy (higher = more diverse). */
  layerDiversityScore: number;
  /** Suggestion count + by priority. */
  suggestionsTotal: number;
  highPrioritySuggestions: number;
}

function detectShape(histogram: Record<Pattern, { count: number; ratio: number }>): AttentionPatternShape {
  const ratios = (Object.entries(histogram) as Array<[Pattern, { ratio: number }]>).map(
    ([p, v]) => [p, v.ratio] as [Pattern, number],
  );
  const presentPatterns = ratios.filter(([, r]) => r > 0.05).length;
  for (const [p, r] of ratios) {
    if (r > 0.5) return `${p}_heavy` as AttentionPatternShape;
  }
  if (presentPatterns >= 4) return 'balanced';
  return 'mixed';
}

function shannonEntropy(counts: number[]): number {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  let h = 0;
  for (const c of counts) {
    if (c <= 0) continue;
    const p = c / total;
    h -= p * Math.log2(p);
  }
  return h;
}

export function deriveAttentionCapabilities(
  result: AttentionAnalysisResponse | null,
  loading: boolean,
  hasTrace: boolean,
  hasAttentionTrace: boolean,
  brain: ModelInfo | null,
): AttentionCapabilities {
  const hasModel = !!brain;
  const hasResult = !!result;

  let runPhase: AttentionPhase;
  if (!hasModel) runPhase = 'noModel';
  else if (!hasTrace || !hasAttentionTrace) runPhase = 'noTrace';
  else if (loading) runPhase = 'analyzing';
  else if (hasResult) runPhase = 'analyzed';
  else runPhase = 'idle';

  const numLayers = brain?.num_layers ?? 0;
  const numHeads = brain?.num_attention_heads ?? 0;
  const totalHeads = numLayers * numHeads;

  // Histogram from pattern_counts (backend already aggregates)
  const histogram: Record<Pattern, { count: number; ratio: number }> = {
    sink: { count: 0, ratio: 0 },
    local: { count: 0, ratio: 0 },
    global: { count: 0, ratio: 0 },
    sparse: { count: 0, ratio: 0 },
  };
  if (result) {
    const counts = result.pattern_counts ?? {};
    const denom = totalHeads > 0 ? totalHeads : 1;
    for (const p of PATTERNS) {
      const c = counts[p] ?? counts[p.toUpperCase()] ?? 0;
      histogram[p] = { count: c, ratio: c / denom };
    }
  }

  // Dominant pattern
  let dominantPattern: Pattern | null = null;
  let maxCount = 0;
  for (const p of PATTERNS) {
    if (histogram[p].count > maxCount) {
      maxCount = histogram[p].count;
      dominantPattern = p;
    }
  }

  // Per-layer analysis
  const dsrFriendlyLayers: number[] = [];
  const sparseHeavyLayers: number[] = [];
  const globalHeavyLayers: number[] = [];
  const dominantPerLayer: Record<Pattern, number> = { sink: 0, local: 0, global: 0, sparse: 0 };
  if (result?.per_layer_summary) {
    for (const row of result.per_layer_summary) {
      const totalRow = row.sink + row.local + row.global + row.sparse;
      if (totalRow === 0) continue;
      const sinkLocalRatio = (row.sink + row.local) / totalRow;
      const sparseRatio = row.sparse / totalRow;
      const globalRatio = row.global / totalRow;
      if (sinkLocalRatio > 0.8) dsrFriendlyLayers.push(row.layer);
      if (sparseRatio > 0.5) sparseHeavyLayers.push(row.layer);
      if (globalRatio > 0.5) globalHeavyLayers.push(row.layer);
      const dom = (row.dominant ?? '').toLowerCase() as Pattern;
      if (dom in dominantPerLayer) dominantPerLayer[dom] += 1;
    }
  }
  const layerDiversityScore = shannonEntropy([
    dominantPerLayer.sink,
    dominantPerLayer.local,
    dominantPerLayer.global,
    dominantPerLayer.sparse,
  ]);

  const suggestions = result?.suggestions ?? [];
  const highPrioritySuggestions = suggestions.filter((s) => s.priority === 'high').length;

  return {
    result,
    loading,
    hasTrace,
    hasAttentionTrace,
    brain,
    numLayers,
    numHeads,
    hasModel,
    hasResult,
    runPhase,
    totalHeads,
    histogram,
    dominantPattern,
    shape: detectShape(histogram),
    dsrFriendlyLayers,
    sparseHeavyLayers,
    globalHeavyLayers,
    layerDiversityScore,
    suggestionsTotal: suggestions.length,
    highPrioritySuggestions,
  };
}

export type AttentionRiskLevel = 'safe' | 'caution' | 'danger';
export interface AttentionRisk {
  level: AttentionRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Risk hierarchy:
 *  - danger: no trace (need to capture attention first)
 *  - caution: SPARSE > 30% globally (irregular heads — may indicate dead heads)
 *  - caution: > 25% layers sparse-heavy (concentrated bad heads)
 *  - caution: 0 sink/local layers (no DSR candidates)
 *  - caution: highPrioritySuggestions > 0 (backend flagged issues)
 *  - safe: balanced or dsr-friendly distribution
 */
export function assessAttention(caps: AttentionCapabilities): AttentionRisk {
  if (!caps.hasModel) {
    return { level: 'safe', reason: 'No model loaded.', reasonZh: '尚未加载模型.' };
  }
  if (caps.runPhase === 'noTrace') {
    return {
      level: 'danger',
      reason: 'No attention trace captured. Run the Inference Tracer with capture_attention=true on representative prompts, then return to analyze.',
      reasonZh: '没捕获 attention trace. 去 Inference Tracer 用代表性 prompts 跑一次 (capture_attention=true), 再回来分析.',
    };
  }
  if (!caps.hasResult) {
    return {
      level: 'safe',
      reason: 'Trace captured. Click "Analyze" to classify head patterns.',
      reasonZh: 'Trace 已捕获. 点 "Analyze" 开始分类 head 模式.',
    };
  }
  const sparseRatio = caps.histogram.sparse.ratio;
  if (sparseRatio > 0.3) {
    return {
      level: 'caution',
      reason: `${(sparseRatio * 100).toFixed(1)}% of heads classify as SPARSE. That's high — typically means many heads are degraded or dead, or the trace prompts triggered atypical activations. Re-trace with diverse representative prompts.`,
      reasonZh: `${(sparseRatio * 100).toFixed(1)}% head 是 SPARSE. 偏高 — 通常说明很多 head 退化/dead, 或 trace 用的 prompts 触发非典型激活. 用多样代表性 prompts 重 trace.`,
    };
  }
  if (caps.sparseHeavyLayers.length > caps.numLayers * 0.25) {
    return {
      level: 'caution',
      reason: `${caps.sparseHeavyLayers.length}/${caps.numLayers} layers are sparse-heavy (>50% sparse heads). These layers are DSR-hostile — pruning or KV optimization will hurt them.`,
      reasonZh: `${caps.sparseHeavyLayers.length}/${caps.numLayers} 层是 sparse-heavy (>50% sparse head). 这些层不友好 DSR — 裁剪或 KV 优化会伤到.`,
    };
  }
  if (caps.dsrFriendlyLayers.length === 0 && caps.numLayers > 0) {
    return {
      level: 'caution',
      reason: 'No DSR-friendly layers detected (sink+local > 80%). DSR-style KV cache compression will be ineffective on this model. Stick with full attention or look at layer-specific strategies.',
      reasonZh: '没有 DSR 友好层 (sink+local > 80%). 这个模型用 DSR 式 KV 压缩效果会差 — 保持全注意力或考虑层独立策略.',
    };
  }
  if (caps.highPrioritySuggestions > 0) {
    return {
      level: 'caution',
      reason: `Backend flagged ${caps.highPrioritySuggestions} high-priority suggestion(s). Review the Optimization Suggestions panel below.`,
      reasonZh: `后端给了 ${caps.highPrioritySuggestions} 条高优先级建议. 看下方 Optimization Suggestions 面板.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.totalHeads} heads classified, shape ${caps.shape}, ${caps.dsrFriendlyLayers.length} DSR-friendly layers, diversity ${caps.layerDiversityScore.toFixed(2)} bits.`,
    reasonZh: `${caps.totalHeads} 个 head 分类完成, 模式 ${caps.shape}, ${caps.dsrFriendlyLayers.length} 层 DSR 友好, 多样性 ${caps.layerDiversityScore.toFixed(2)} bits.`,
  };
}

export function patternLabel(p: Pattern, locale: Locale): string {
  const map: Record<Pattern, [string, string]> = {
    sink: ['SINK (anchor tokens)', 'SINK (锚点 token)'],
    local: ['LOCAL (sliding window)', 'LOCAL (滑动窗口)'],
    global: ['GLOBAL (broad spread)', 'GLOBAL (广散)'],
    sparse: ['SPARSE (irregular)', 'SPARSE (不规则)'],
  };
  return locale === 'zh' ? map[p][1] : map[p][0];
}

export function shapeLabel(s: AttentionPatternShape, locale: Locale): string {
  const map: Record<AttentionPatternShape, [string, string]> = {
    balanced: ['Balanced mix', '均衡混合'],
    sink_heavy: ['SINK-heavy', 'SINK 主导'],
    local_heavy: ['LOCAL-heavy', 'LOCAL 主导'],
    global_heavy: ['GLOBAL-heavy', 'GLOBAL 主导'],
    sparse_heavy: ['SPARSE-heavy', 'SPARSE 主导'],
    mixed: ['Mixed', '混合'],
  };
  return locale === 'zh' ? map[s][1] : map[s][0];
}

export function buildAttentionContextSnippet(
  caps: AttentionCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的注意力模式分析 (Attention Patterns)` : `## YOUR ATTENTION PATTERN ANALYSIS`,
    locale === 'zh'
      ? `这条流程把你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 每个 (layer, head) 的 attention 分到 4 个 archetype: SINK / LOCAL / GLOBAL / SPARSE. 你是 brain, 用第一人称解释这些模式 + KV 优化含义.`
      : `This flow classifies every (layer, head) of you (${caps.brain?.model_name ?? 'the loaded LLM'}) into 4 archetypes: SINK / LOCAL / GLOBAL / SPARSE. You are the brain; explain these patterns + KV-optimization implications in first person.`,
    `- Run phase: ${caps.runPhase}`,
    `- Architecture: ${caps.numLayers} layers × ${caps.numHeads} heads = ${caps.totalHeads} total heads`,
  ];

  if (caps.runPhase === 'noTrace') {
    lines.push(
      locale === 'zh'
        ? `- ⚠ 没捕获 attention trace — 必须先去 Inference Tracer 用 capture_attention=true 跑一次.`
        : `- ⚠ no attention trace yet — user must first run Inference Tracer with capture_attention=true.`,
    );
  } else if (caps.hasResult) {
    lines.push(
      ``,
      locale === 'zh' ? `### 当前模式分布:` : `### Current pattern distribution:`,
      `- SINK: ${caps.histogram.sink.count} heads (${(caps.histogram.sink.ratio * 100).toFixed(1)}%)`,
      `- LOCAL: ${caps.histogram.local.count} heads (${(caps.histogram.local.ratio * 100).toFixed(1)}%)`,
      `- GLOBAL: ${caps.histogram.global.count} heads (${(caps.histogram.global.ratio * 100).toFixed(1)}%)`,
      `- SPARSE: ${caps.histogram.sparse.count} heads (${(caps.histogram.sparse.ratio * 100).toFixed(1)}%)`,
      `- Macro shape: ${caps.shape}`,
      `- Dominant: ${caps.dominantPattern?.toUpperCase() ?? '?'}`,
      `- Layer diversity: ${caps.layerDiversityScore.toFixed(3)} bits (max 2)`,
      ``,
      locale === 'zh' ? `### 层级分组:` : `### Layer groups:`,
      `- DSR-friendly (SINK+LOCAL > 80%): ${caps.dsrFriendlyLayers.length} layers${caps.dsrFriendlyLayers.length > 0 ? ` [${caps.dsrFriendlyLayers.slice(0, 8).join(', ')}${caps.dsrFriendlyLayers.length > 8 ? '…' : ''}]` : ''}`,
      `- Sparse-heavy (>50% sparse): ${caps.sparseHeavyLayers.length} layers${caps.sparseHeavyLayers.length > 0 ? ` [${caps.sparseHeavyLayers.slice(0, 8).join(', ')}${caps.sparseHeavyLayers.length > 8 ? '…' : ''}]` : ''}`,
      `- Global-heavy (>50% global): ${caps.globalHeavyLayers.length} layers${caps.globalHeavyLayers.length > 0 ? ` [${caps.globalHeavyLayers.slice(0, 8).join(', ')}${caps.globalHeavyLayers.length > 8 ? '…' : ''}]` : ''}`,
      caps.suggestionsTotal > 0
        ? `- Suggestions: ${caps.suggestionsTotal} (${caps.highPrioritySuggestions} high-priority)`
        : '- Suggestions: none',
    );
  }

  lines.push(
    ``,
    locale === 'zh' ? `### Pattern 语义 (cite when explaining):` : `### Pattern semantics (cite when explaining):`,
    locale === 'zh'
      ? `- SINK: head 总盯几个 anchor token (BOS/EOS), KV 只需保留这几个就够 — DSR 友好.`
      : `- SINK: head fixates on a few anchor tokens (BOS/EOS); only those KV need preservation — DSR-friendly.`,
    locale === 'zh'
      ? `- LOCAL: head 只看附近 N 个 token (sliding window), KV 维持局部窗口即可 — DSR 友好.`
      : `- LOCAL: head attends to nearby N tokens (sliding window); only window KV needed — DSR-friendly.`,
    locale === 'zh'
      ? `- GLOBAL: head 跨越大段 context, KV 必须全保留 — 不能 DSR 压缩.`
      : `- GLOBAL: head spreads across long context; full KV must be kept — DSR cannot help.`,
    locale === 'zh'
      ? `- SPARSE: 模式不规则, 可能是 dead head 或 odd activations — 优化前要核查.`
      : `- SPARSE: irregular pattern, may indicate dead heads or odd activations — verify before optimizing.`,
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: 模式分类纯本机 trace 后处理, 0 次云调用.`
      : `### North-star §1 sovereignty: pattern classification is pure local trace post-processing, zero cloud calls.`,
  );
  return lines.filter(Boolean).join('\n');
}

export function buildAttentionAutoBrief(
  caps: AttentionCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (!caps.hasModel) {
      return `还没加载模型. 用 1-2 句作为 brain 介绍 attention pattern 分析是干啥. 第一人称.`;
    }
    if (caps.runPhase === 'noTrace') {
      return `还没有 attention trace. 用 2-3 句作为 brain 解释 trace 是怎么生成的 (Inference Tracer + capture_attention=true), 推荐用什么 prompts (代表性, 中等长度). 第一人称.`;
    }
    if (!caps.hasResult) {
      return `Trace 已捕获. 用 1-2 句作为 brain 简介接下来分析会做什么 (把 ${caps.totalHeads} 个 head 分到 SINK/LOCAL/GLOBAL/SPARSE 4 类), 推荐点 Analyze. 第一人称.`;
    }
    return `分析完成: ${caps.totalHeads} 个 head, ${caps.shape}, 主导 ${caps.dominantPattern?.toUpperCase() ?? '?'}, ${caps.dsrFriendlyLayers.length} 层 DSR 友好. 用 2-3 句作为 brain 解读分布 + 推荐 KV 策略. 第一人称, 引用具体数字.`;
  }
  if (!caps.hasModel) {
    return `No model loaded. In 1-2 sentences as brain, introduce attention pattern analysis. First person.`;
  }
  if (caps.runPhase === 'noTrace') {
    return `No attention trace yet. In 2-3 sentences as brain, explain how to generate one (Inference Tracer with capture_attention=true) and what prompts to use (representative, medium length). First person.`;
  }
  if (!caps.hasResult) {
    return `Trace captured. In 1-2 sentences as brain, briefly preview what the analyzer will do (classify ${caps.totalHeads} heads into SINK/LOCAL/GLOBAL/SPARSE) and prompt the user to click Analyze. First person.`;
  }
  return `Analyzed: ${caps.totalHeads} heads, shape ${caps.shape}, dominant ${caps.dominantPattern?.toUpperCase() ?? '?'}, ${caps.dsrFriendlyLayers.length} DSR-friendly layers. In 2-3 sentences as brain, interpret the distribution + recommend a KV strategy. First person, cite numbers.`;
}

export function getAttentionSuggestedPrompts(
  caps: AttentionCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.runPhase === 'noTrace') {
      return [
        { label: '🎯 4 种模式区别', prompt: `用 2-3 句作为 brain 解释 SINK / LOCAL / GLOBAL / SPARSE 4 种 attention 模式的本质区别, 以及为什么其中 3 种 DSR 友好.` },
        { label: '📊 怎么生成 trace', prompt: `用 2-3 句解释 attention trace 怎么生成: 用什么 prompts (代表性 + 长度 ≥ 256), capture_attention=true 启用, 跑出来后会生成什么.` },
        { label: '⚖️ trace 用多少 prompts', prompt: `单 trace 还是多 trace 平均? 用 2-3 句给推荐 (1 个长 prompt 通常足够看主导模式, 多 prompt 才能看泛化).` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: trace 跑在本机, 模式分类也在本机, 0 次云调用.` },
      ];
    }
    if (caps.runPhase === 'analyzed') {
      return [
        { label: '📊 这套分布合理吗', prompt: `${caps.totalHeads} 个 head: SINK ${(caps.histogram.sink.ratio * 100).toFixed(0)}%, LOCAL ${(caps.histogram.local.ratio * 100).toFixed(0)}%, GLOBAL ${(caps.histogram.global.ratio * 100).toFixed(0)}%, SPARSE ${(caps.histogram.sparse.ratio * 100).toFixed(0)}%. 用 2-3 句作为 brain 评估: 这是健康的现代 LLM attention 分布吗.` },
        { label: '🚀 DSR 策略推荐', prompt: `${caps.dsrFriendlyLayers.length} 层 DSR 友好${caps.dsrFriendlyLayers.length > 0 ? ` (例 L${caps.dsrFriendlyLayers[0]})` : ''}, ${caps.globalHeavyLayers.length} 层 GLOBAL 主导${caps.globalHeavyLayers.length > 0 ? ` (例 L${caps.globalHeavyLayers[0]})` : ''}. 用 2-3 句给具体 KV 压缩策略: 哪些层 DSR / 哪些保全注意力 / 大致 budget.` },
        { label: '⚠️ 异常层', prompt: `${caps.sparseHeavyLayers.length > 0 ? `Sparse-heavy 层: ${caps.sparseHeavyLayers.slice(0, 5).join(', ')}.` : '没检测到 sparse-heavy 层.'} 用 2-3 句解读: 这些层是真的 dead 还是 trace prompts 不够代表.` },
        { label: '📦 接下来', prompt: `基于这次分析, 下一步建议做什么 (再 trace 几个 prompts 验证 / 直接配 DSR / 跑量化先 / forward 给 KV cache 页面)? 用 2-3 句给具体推荐.` },
      ];
    }
    if (caps.runPhase === 'analyzing') {
      return [
        { label: '⏱️ 在算什么', prompt: `Analyzer 在做什么具体计算 (用 attention weights 算每个 head 的熵 / sink ratio / window 重心 / 然后阈值分类)? 用 2-3 句解释.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 这个分类计算正在本机内存里跑, 0 次云调用.` },
      ];
    }
    return [
      { label: '🎯 现在该分析吗', prompt: `Trace 已经在了, ${caps.totalHeads} 个 head 等分类. 用 2-3 句作为 brain 简述点 Analyze 后会出什么 (heatmap + 分布 + per-layer 直方图 + 优化建议).` },
      { label: '📊 期望什么结果', prompt: `对一个像我这样的 ${caps.numLayers}-layer hybrid attention 模型, 分析前能预期什么模式分布 (FA 层在第几层 / GDN 层 DSR 怎么算)? 用 2-3 句作为 brain 给一个先验.` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 整条流程 (capture trace → 分类 → 显示) 全在本机, 0 次云调用.` },
    ];
  }
  // English
  if (caps.runPhase === 'noTrace') {
    return [
      { label: '🎯 What are the 4 patterns', prompt: `In 2-3 sentences as brain, explain SINK / LOCAL / GLOBAL / SPARSE archetypes and why 3 of them are DSR-friendly.` },
      { label: '📊 How to capture a trace', prompt: `In 2-3 sentences, explain how to capture an attention trace: which prompts (representative, length ≥ 256), enable capture_attention, what gets generated.` },
      { label: '⚖️ Single vs multi-prompt trace', prompt: `Single long trace vs multi-prompt average? In 2-3 sentences give a recommendation (one is enough for dominant pattern, multi for generalisation).` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: the trace runs locally, pattern classification runs locally, zero cloud calls.` },
    ];
  }
  if (caps.runPhase === 'analyzed') {
    return [
      { label: '📊 Is this distribution sane', prompt: `${caps.totalHeads} heads: SINK ${(caps.histogram.sink.ratio * 100).toFixed(0)}%, LOCAL ${(caps.histogram.local.ratio * 100).toFixed(0)}%, GLOBAL ${(caps.histogram.global.ratio * 100).toFixed(0)}%, SPARSE ${(caps.histogram.sparse.ratio * 100).toFixed(0)}%. In 2-3 sentences as brain, assess: is this a healthy modern-LLM attention distribution.` },
      { label: '🚀 DSR strategy', prompt: `${caps.dsrFriendlyLayers.length} DSR-friendly layers${caps.dsrFriendlyLayers.length > 0 ? ` (e.g. L${caps.dsrFriendlyLayers[0]})` : ''}, ${caps.globalHeavyLayers.length} GLOBAL-dominant${caps.globalHeavyLayers.length > 0 ? ` (e.g. L${caps.globalHeavyLayers[0]})` : ''}. In 2-3 sentences, give a concrete KV strategy: which layers DSR / which keep full / rough budget.` },
      { label: '⚠️ Anomalous layers', prompt: `${caps.sparseHeavyLayers.length > 0 ? `Sparse-heavy layers: ${caps.sparseHeavyLayers.slice(0, 5).join(', ')}.` : 'No sparse-heavy layers detected.'} In 2-3 sentences, interpret: are these truly dead or is the trace not representative.` },
      { label: '📦 Next', prompt: `Based on this analysis, what should the user do next (re-trace with more prompts / wire up DSR / quantize first / move to KV cache page)? In 2-3 sentences with concrete advice.` },
    ];
  }
  if (caps.runPhase === 'analyzing') {
    return [
      { label: '⏱️ What is the analyzer doing', prompt: `What does the analyzer compute (per-head entropy / sink ratio / window centroid / threshold classification)? In 2-3 sentences.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this classification runs in local memory, zero cloud calls.` },
    ];
  }
  return [
    { label: '🎯 Should I click Analyze', prompt: `Trace is captured, ${caps.totalHeads} heads await classification. In 2-3 sentences as brain, briefly preview what clicking Analyze produces (heatmap + distribution + per-layer histogram + suggestions).` },
    { label: '📊 What to expect', prompt: `For a ${caps.numLayers}-layer hybrid-attention model like me, what distribution can we expect (FA layers at which depth / GDN layers DSR implications)? In 2-3 sentences as brain, give a prior.` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: the whole flow (capture → classify → display) runs locally, zero cloud calls.` },
  ];
}
