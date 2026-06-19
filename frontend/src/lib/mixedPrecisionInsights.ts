// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * mixedPrecisionInsights — derived per-layer bit-config state + chat helpers
 * for the /mixed-precision page (MixedPrecisionPanel.tsx).
 *
 * Mixed-precision quantization assigns a bit-width per transformer layer
 * (commonly 2/3/4/8) instead of a single uniform width. The sweet spot:
 *  - Sandwich: layer 0 + last layer at 8-bit (embeddings + lm_head sensitive)
 *    middle layers at 3-4 bit. Compression win without PPL collapse.
 *  - Conservative-uniform: every layer at 4-bit (no mixed-precision benefit).
 *  - Aggressive-low: any 2-bit layer triggers PPL collapse.
 *
 * §9.1 multi-component: each layer is a "slot"; aggregate capability =
 *  histogram + avg + pattern + risky-layer detection. §10.6 bucket idiom.
 *
 * §10.3 naming: capability uses `runPhase` not `status`.
 *
 * Sovereignty (§9.2 mandatory): mixed-precision quantization is a pure local
 * weight transform. The Sovereignty card asserts "0 cloud calls".
 */
import type { ModelInfo } from '@/api/types';

type Locale = 'en' | 'zh';

export interface LayerConfig {
  layer_idx: number;
  bits: number;
  group_size: number;
}

export interface MixedPrecisionResult {
  success: boolean;
  output_dir?: string;
  duration_seconds?: number;
  error?: string;
}

export type MixPhase = 'idle' | 'running' | 'success' | 'failed';

/** Macro shape of the bit allocation. */
export type BitPattern =
  | 'uniform'        // all layers same bit
  | 'sandwich'       // high at both ends, low in middle (typical sweet spot)
  | 'inverted'       // low at both ends, high in middle (suspicious)
  | 'monotonic_up'   // gradient low → high
  | 'monotonic_down' // gradient high → low
  | 'mixed';         // none of the above

export type AvgBitsBucket = 'aggressive' | 'compressed' | 'balanced' | 'conservative';

export interface MixedPrecisionCapabilities {
  // ── Inputs ─────────────────────────────────────────────────────────────
  layerConfigs: LayerConfig[];
  numLayers: number;
  modelSizeBytes: number;
  baselineBits: number;
  brain: ModelInfo | null;
  result: MixedPrecisionResult | null;
  running: boolean;

  // ── Aggregate derived ──────────────────────────────────────────────────
  /** {bits: count} */
  bitHistogram: Record<number, number>;
  /** Distinct bit widths in use (sorted ascending). */
  bitsInUse: number[];
  /** Average across all layers. */
  avgBits: number;
  /** Estimated size after applying configs (relative to baselineBits). */
  estimatedSize: number;
  /** Estimated savings vs current model size, 0..1. */
  savingsRatio: number;
  /** Layers running below 4-bit (vulnerability flag). */
  riskyLayers: LayerConfig[];
  /** Number of 2-bit layers (highest danger tier). */
  bits2Count: number;
  /** Number of 3-bit layers (caution tier). */
  bits3Count: number;
  /** Detected macro pattern. */
  pattern: BitPattern;
  /** Bucket for avg bit width. */
  avgBitsBucket: AvgBitsBucket;
  /** Run lifecycle. */
  runPhase: MixPhase;
}

function detectPattern(configs: LayerConfig[]): BitPattern {
  if (configs.length === 0) return 'uniform';
  const bits = configs.map((c) => c.bits);
  if (bits.every((b) => b === bits[0])) return 'uniform';

  const n = bits.length;
  const first = bits[0];
  const last = bits[n - 1];
  const mid = bits[Math.floor(n / 2)];

  // Sandwich: edges high, middle low (allow ±1 wiggle)
  if (first >= 4 && last >= 4 && mid <= Math.min(first, last) - 1) {
    // Check whether the middle band stays low
    const midBand = bits.slice(Math.floor(n * 0.25), Math.ceil(n * 0.75));
    const midAvg = midBand.reduce((a, b) => a + b, 0) / midBand.length;
    if (midAvg < (first + last) / 2) return 'sandwich';
  }

  // Inverted sandwich: edges low, middle high
  if (first <= 3 && last <= 3 && mid >= Math.max(first, last) + 1) {
    const midBand = bits.slice(Math.floor(n * 0.25), Math.ceil(n * 0.75));
    const midAvg = midBand.reduce((a, b) => a + b, 0) / midBand.length;
    if (midAvg > (first + last) / 2) return 'inverted';
  }

  // Monotonic: > 75% steps in one direction
  let upSteps = 0;
  let downSteps = 0;
  for (let i = 1; i < n; i++) {
    if (bits[i] > bits[i - 1]) upSteps++;
    else if (bits[i] < bits[i - 1]) downSteps++;
  }
  if (upSteps > 0.5 * (n - 1) && downSteps === 0) return 'monotonic_up';
  if (downSteps > 0.5 * (n - 1) && upSteps === 0) return 'monotonic_down';

  return 'mixed';
}

function bucketAvgBits(avg: number): AvgBitsBucket {
  if (avg < 2.75) return 'aggressive';
  if (avg < 3.5) return 'compressed';
  if (avg < 5) return 'balanced';
  return 'conservative';
}

export function deriveMixedPrecisionCapabilities(
  layerConfigs: LayerConfig[],
  modelSizeBytes: number,
  baselineBits: number,
  brain: ModelInfo | null,
  result: MixedPrecisionResult | null,
  running: boolean,
): MixedPrecisionCapabilities {
  const numLayers = layerConfigs.length;
  const bitHistogram: Record<number, number> = {};
  for (const lc of layerConfigs) {
    bitHistogram[lc.bits] = (bitHistogram[lc.bits] || 0) + 1;
  }
  const bitsInUse = Object.keys(bitHistogram).map(Number).sort((a, b) => a - b);
  const avgBits = numLayers > 0
    ? layerConfigs.reduce((sum, lc) => sum + lc.bits, 0) / numLayers
    : 0;

  const estimatedSize = modelSizeBytes > 0 && baselineBits > 0
    ? modelSizeBytes * (avgBits / baselineBits)
    : 0;
  const savingsRatio = modelSizeBytes > 0 && estimatedSize > 0
    ? Math.max(0, (modelSizeBytes - estimatedSize) / modelSizeBytes)
    : 0;

  const riskyLayers = layerConfigs.filter((lc) => lc.bits <= 3);
  const bits2Count = layerConfigs.filter((lc) => lc.bits <= 2).length;
  const bits3Count = layerConfigs.filter((lc) => lc.bits === 3).length;

  let runPhase: MixPhase;
  if (running) runPhase = 'running';
  else if (result?.success) runPhase = 'success';
  else if (result && !result.success) runPhase = 'failed';
  else runPhase = 'idle';

  return {
    layerConfigs,
    numLayers,
    modelSizeBytes,
    baselineBits,
    brain,
    result,
    running,
    bitHistogram,
    bitsInUse,
    avgBits,
    estimatedSize,
    savingsRatio,
    riskyLayers,
    bits2Count,
    bits3Count,
    pattern: detectPattern(layerConfigs),
    avgBitsBucket: bucketAvgBits(avgBits),
    runPhase,
  };
}

export type MixRiskLevel = 'safe' | 'caution' | 'danger';
export interface MixRisk {
  level: MixRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Risk hierarchy (most dangerous first):
 *  - danger: any 2-bit layer (PPL typically collapses, no backend guardrail)
 *  - danger: avgBits < 2.75 (whole-model aggressive)
 *  - caution: > 40% of layers at 3-bit (cumulative quality risk)
 *  - caution: inverted sandwich (suspect schedule)
 *  - caution: uniform (mixed-precision UI pointless — use simple quant)
 *  - caution: layer 0 OR last layer below 4-bit (embeddings/lm_head sensitive)
 *  - safe: balanced mix
 */
export function assessMixedPrecision(caps: MixedPrecisionCapabilities): MixRisk {
  if (caps.numLayers === 0) {
    return { level: 'safe', reason: 'No layers configured.', reasonZh: '尚未配置层.' };
  }
  if (caps.bits2Count > 0) {
    return {
      level: 'danger',
      reason: `${caps.bits2Count} layer(s) at 2-bit. PPL typically collapses below 4-bit; no backend guardrail. Bump to 3-bit minimum, ideally 4-bit.`,
      reasonZh: `${caps.bits2Count} 层用 2-bit 量化. 低于 4-bit PPL 通常会崩, 后端不拦. 至少改成 3-bit, 建议 4-bit.`,
    };
  }
  if (caps.avgBits < 2.75) {
    return {
      level: 'danger',
      reason: `Average ${caps.avgBits.toFixed(2)} bits is too aggressive — model will likely break. Push avg ≥ 3.5 unless you've validated the architecture.`,
      reasonZh: `平均 ${caps.avgBits.toFixed(2)}-bit 过激 — 模型大概率崩. 平均至少 3.5-bit, 除非已实测验证.`,
    };
  }
  if (caps.bits3Count / caps.numLayers > 0.4) {
    return {
      level: 'caution',
      reason: `${caps.bits3Count}/${caps.numLayers} layers at 3-bit (${(caps.bits3Count / caps.numLayers * 100).toFixed(0)}%). 3-bit is borderline; aggregating that many in one model multiplies failure risk.`,
      reasonZh: `${caps.bits3Count}/${caps.numLayers} 层用 3-bit (${(caps.bits3Count / caps.numLayers * 100).toFixed(0)}%). 3-bit 边缘可用; 太多层堆在一起放大失败风险.`,
    };
  }
  // Layer 0 / last layer guard — embeddings + lm_head are precision-sensitive
  const last = caps.layerConfigs[caps.numLayers - 1];
  const first = caps.layerConfigs[0];
  if (first && first.bits < 4) {
    return {
      level: 'caution',
      reason: `Layer 0 at ${first.bits}-bit. The first layer is close to embeddings — keep at ≥ 4-bit unless you know the model handles low-bit input projection.`,
      reasonZh: `Layer 0 用 ${first.bits}-bit. 首层紧邻 embedding — 除非你已实测过, 至少保持 4-bit.`,
    };
  }
  if (last && last.bits < 4) {
    return {
      level: 'caution',
      reason: `Last layer at ${last.bits}-bit. The output projection feeds lm_head — keep at ≥ 4-bit to protect logit quality.`,
      reasonZh: `末层用 ${last.bits}-bit. 输出投影喂 lm_head — 至少保持 4-bit 保住 logit 质量.`,
    };
  }
  if (caps.pattern === 'inverted') {
    return {
      level: 'caution',
      reason: 'Inverted sandwich (low edges, high middle). Sensitivity research shows the opposite is usually better — verify before running.',
      reasonZh: '反三明治模式 (两端低 中间高). 敏感性研究通常推荐相反 — 跑之前先核对.',
    };
  }
  if (caps.pattern === 'uniform') {
    return {
      level: 'caution',
      reason: `All layers at ${caps.avgBits}-bit — that's just uniform quantization, not mixed-precision. Use the simpler quantize page or vary at least one layer.`,
      reasonZh: `所有层都是 ${caps.avgBits}-bit — 这就是均匀量化, 不是混精. 用普通量化页面, 或至少调一层.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.numLayers} layers, avg ${caps.avgBits.toFixed(2)} bits, pattern ${caps.pattern}. Estimated saving ${(caps.savingsRatio * 100).toFixed(1)}%.`,
    reasonZh: `${caps.numLayers} 层, 平均 ${caps.avgBits.toFixed(2)}-bit, 模式 ${caps.pattern}. 估省 ${(caps.savingsRatio * 100).toFixed(1)}%.`,
  };
}

export function patternLabel(p: BitPattern, locale: Locale): string {
  const map: Record<BitPattern, [string, string]> = {
    uniform: ['Uniform', '均匀'],
    sandwich: ['Sandwich', '三明治'],
    inverted: ['Inverted', '反三明治'],
    monotonic_up: ['Ascending', '渐升'],
    monotonic_down: ['Descending', '渐降'],
    mixed: ['Mixed', '混合'],
  };
  return locale === 'zh' ? map[p][1] : map[p][0];
}

export function avgBitsBucketLabel(b: AvgBitsBucket, locale: Locale): string {
  const map: Record<AvgBitsBucket, [string, string]> = {
    aggressive: ['Aggressive', '激进'],
    compressed: ['Compressed', '紧凑'],
    balanced: ['Balanced', '平衡'],
    conservative: ['Conservative', '保守'],
  };
  return locale === 'zh' ? map[b][1] : map[b][0];
}

export function buildMixedPrecisionContextSnippet(
  caps: MixedPrecisionCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的混合精度量化 (Mixed-Precision)` : `## YOUR MIXED-PRECISION QUANTIZATION`,
    locale === 'zh'
      ? `这条流程对每一层独立选 bit 宽度 (2/3/4/8). 你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 是 brain, 用第一人称解释这套配置选择 + PPL 风险 + 预期收益.`
      : `This flow assigns a bit-width per layer (2/3/4/8). You (${caps.brain?.model_name ?? 'the loaded LLM'}) are the brain; explain this config + PPL risks + expected gains in first person.`,
    `- Layers: ${caps.numLayers} · baseline ${caps.baselineBits}-bit`,
    `- Bit histogram: ${Object.entries(caps.bitHistogram).sort(([a], [b]) => Number(a) - Number(b)).map(([b, c]) => `${b}-bit×${c}`).join(' · ') || '(none)'}`,
    `- Average bits: ${caps.avgBits.toFixed(2)} (${caps.avgBitsBucket})`,
    `- Pattern: ${caps.pattern}`,
    `- Estimated size: ${(caps.estimatedSize / 1e9).toFixed(2)} GB (${(caps.savingsRatio * 100).toFixed(1)}% saved vs baseline)`,
    caps.bits2Count > 0 ? `- ⚠ ${caps.bits2Count} layer(s) at 2-bit (PPL collapse risk)` : '',
    caps.bits3Count > 0 ? `- ${caps.bits3Count} layer(s) at 3-bit (borderline)` : '',
    caps.runPhase !== 'idle' ? `- Run phase: ${caps.runPhase}${caps.result?.success ? ` · output ${caps.result.output_dir}` : ''}` : '',
    ``,
    locale === 'zh' ? `### 模式语义 (cite when explaining):` : `### Pattern semantics (cite when explaining):`,
    locale === 'zh'
      ? `- sandwich = 两端高 中间低 (常见甜点: 第 0 层 + 末层 8-bit, 中间 3-4 bit). 保 embedding + lm_head 精度.`
      : `- sandwich = high at edges, low in middle (typical sweet spot: layer 0 + last at 8-bit, middle 3-4 bit). Preserves embedding + lm_head precision.`,
    locale === 'zh'
      ? `- inverted = 两端低 中间高. 通常不推荐, 因为 embeddings/lm_head 比中间 FFN 对 quant 更敏感.`
      : `- inverted = low edges, high middle. Usually not recommended — embeddings/lm_head are more sensitive than middle FFN to quantization.`,
    locale === 'zh'
      ? `- uniform = 全部相同 bit. 失去混精意义, 等价于普通量化.`
      : `- uniform = all same bit. Loses the mixed-precision benefit, equivalent to plain quantization.`,
    locale === 'zh'
      ? `- monotonic = 单调升/降. 偶尔有用 (例如自下而上 attention quality budget).`
      : `- monotonic = ascending/descending gradient. Occasionally useful (e.g. depth-budgeted attention quality).`,
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: 混精量化是纯本地权重运算, 0 次云调用.`
      : `### North-star §1 sovereignty: mixed-precision quantization is pure local weight arithmetic, zero cloud calls.`,
  ];
  return lines.filter(Boolean).join('\n');
}

export function buildMixedPrecisionAutoBrief(
  caps: MixedPrecisionCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (caps.numLayers === 0) {
      return `还没加载模型 / 没有层信息. 用 2-3 句作为 brain 介绍混合精度量化是干啥 + 何时该用. 第一人称.`;
    }
    if (caps.runPhase === 'success') {
      return `混精量化完成! 输出 ${caps.result?.output_dir ?? ''}. 用 2-3 句作为 brain 评估这次结果 (平均 ${caps.avgBits.toFixed(2)}-bit, ${caps.pattern} 模式, 估省 ${(caps.savingsRatio * 100).toFixed(1)}%), 推荐 benchmark. 第一人称, 引用具体数字.`;
    }
    if (caps.runPhase === 'failed') {
      return `混精量化失败 (${caps.result?.error ?? '未知错误'}). 用 2-3 句作为 brain 推断最可能根因 + 第一步排查. 第一人称.`;
    }
    if (caps.runPhase === 'running') {
      return `正在跑混精量化, 配置: ${caps.numLayers} 层平均 ${caps.avgBits.toFixed(2)}-bit. 用 2-3 句简述配置意图 + 等结果时该看哪些指标. 第一人称.`;
    }
    return `当前配置: ${caps.numLayers} 层, 平均 ${caps.avgBits.toFixed(2)}-bit, 模式 ${caps.pattern}, 预估 ${(caps.savingsRatio * 100).toFixed(1)}% 压缩. 用 2-3 句作为 brain 评估这套混精方案 (gut check + 风险点 + 推荐). 第一人称.`;
  }
  if (caps.numLayers === 0) {
    return `No layers configured yet. In 2-3 sentences as brain, explain what mixed-precision quantization is and when to use it. First person.`;
  }
  if (caps.runPhase === 'success') {
    return `Mixed-precision quant complete! Output ${caps.result?.output_dir ?? ''}. In 2-3 sentences as brain, assess (avg ${caps.avgBits.toFixed(2)} bits, ${caps.pattern} pattern, ~${(caps.savingsRatio * 100).toFixed(1)}% saved), recommend benchmarking. First person, cite numbers.`;
  }
  if (caps.runPhase === 'failed') {
    return `Mixed-precision quant failed (${caps.result?.error ?? 'unknown error'}). In 2-3 sentences as brain, infer the most likely root cause + the first triage step. First person.`;
  }
  if (caps.runPhase === 'running') {
    return `Running mixed-precision quant — config: ${caps.numLayers} layers averaging ${caps.avgBits.toFixed(2)} bits. In 2-3 sentences as brain, summarize the config intent and which metrics to watch in the result. First person.`;
  }
  return `Current config: ${caps.numLayers} layers, avg ${caps.avgBits.toFixed(2)} bits, pattern ${caps.pattern}, estimated ${(caps.savingsRatio * 100).toFixed(1)}% compression. In 2-3 sentences as brain, assess this mixed-precision plan (gut check + risks + recommendation). First person.`;
}

export function getMixedPrecisionSuggestedPrompts(
  caps: MixedPrecisionCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.numLayers === 0) {
      return [
        { label: '🎯 混精是干啥', prompt: `用 2-3 句作为 brain 解释混合精度量化 (per-layer bit width) 与普通量化的区别, 以及哪些场景应该用 (压缩极限 + 保关键层精度).` },
        { label: '🥪 sandwich 模式', prompt: `什么是 sandwich pattern? 为什么首末层一般要 8-bit, 中间可以 3-bit? 给一个可视化的解释.` },
        { label: '⚖️ avg bits 推荐', prompt: `iPhone (8GB)/iPad (16GB)/Mac (32GB+) 各推荐什么平均 bit 宽度? 给一个甜点区域和该区域常见 pattern.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 混精量化全程在本机进行, 0 次云调用 — 与"上传到 SaaS"对比.` },
      ];
    }
    if (caps.runPhase === 'success') {
      return [
        { label: '📊 这次合理吗', prompt: `跑出来 ${caps.numLayers} 层平均 ${caps.avgBits.toFixed(2)}-bit, 估省 ${(caps.savingsRatio * 100).toFixed(1)}%. 用 2-3 句作为 brain 评估: 这套配置出来的模型适合什么场景, PPL 大概在什么水平.` },
        { label: '🔬 怎么 verify', prompt: `给一个 4 步验证清单 (PPL benchmark / smoke test prompts / 长文本 coherence / 速度对比) 帮用户判断这次混精值不值.` },
        { label: '⚖️ vs uniform 4-bit', prompt: `用 2-3 句对比这次混精方案 vs 简单全部 4-bit: 哪个更小, 哪个更稳, 用户应该选哪个.` },
        { label: '📦 接下来', prompt: `输出在 ${caps.result?.output_dir ?? '/tmp/...'}. 用 2-3 句告诉用户接下来怎么做 (benchmark → device test → export).` },
      ];
    }
    if (caps.runPhase === 'failed') {
      return [
        { label: '🔥 失败根因', prompt: `失败信息 "${caps.result?.error ?? '未知'}". 用 2-3 句作为 brain 推断最可能根因 + 第一步排查命令.` },
        { label: '🛠️ retry 策略', prompt: `这套配置改哪几个参数可能让它跑通? (例如 2-bit 升 3-bit / 减少低 bit 层数 / 改 group_size). 给具体建议.` },
        { label: '⚙️ 系统级 vs 配置级', prompt: `这个失败是配置问题 (bits 选错) 还是系统问题 (内存不够 / mlx 版本)? 给一个分流诊断.` },
        { label: '🌍 端侧主权', prompt: `即使失败, 整个尝试也在本机. 用 2-3 句强调本机 fail-fast 的优势.` },
      ];
    }
    if (caps.runPhase === 'running') {
      return [
        { label: '⏱️ 还要多久', prompt: `${caps.numLayers} 层平均 ${caps.avgBits.toFixed(2)}-bit 量化, 估计剩余时间. 给一个 estimate + 等结果时建议看什么.` },
        { label: '🎯 现在该看什么', prompt: `跑量化时, 用户能监控什么 (内存占用 / 进度 / 中间产物)? 用 2-3 句给一个 status checklist.` },
        { label: '⚠️ 何时该停', prompt: `如果跑了很久没结束, 什么信号表示该 abort 重新配? 给具体判断.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调当前正在本机进行 ${caps.numLayers} 层量化, 0 网络调用.` },
      ];
    }
    return [
      { label: '⚙️ 这套配置怎么样', prompt: `${caps.numLayers} 层, 平均 ${caps.avgBits.toFixed(2)}-bit, ${caps.pattern} 模式, 估省 ${(caps.savingsRatio * 100).toFixed(1)}%. 用 2-3 句作为 brain 评估这套配置的合理性 + 风险点.` },
      { label: '🥪 改 sandwich', prompt: `如果让我改成 sandwich pattern (首末 8-bit, 中间 3-bit), 估省多少, PPL 风险怎么变化?` },
      { label: '⚖️ avg bit 怎么选', prompt: `当前平均 ${caps.avgBits.toFixed(2)} 在 ${caps.avgBitsBucket} 区. 这个 bucket 适合什么部署目标 (iPhone / iPad / Mac)?` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 这套混精方案全程在本机, 0 云调用.` },
    ];
  }
  // English
  if (caps.numLayers === 0) {
    return [
      { label: '🎯 What is mixed-precision', prompt: `In 2-3 sentences as brain, contrast mixed-precision (per-layer bit width) with plain quantization, and when to prefer it (push compression limits while protecting sensitive layers).` },
      { label: '🥪 Sandwich pattern', prompt: `What is the sandwich pattern? Why are first/last layers usually 8-bit while middle can be 3-bit? Give a visual mental model.` },
      { label: '⚖️ Avg bits recommendation', prompt: `For iPhone (8GB) / iPad (16GB) / Mac (32GB+), what avg bit width is the sweet spot? Give a band and the typical pattern in that band.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: mixed-precision quantization runs entirely locally, zero cloud calls — vs uploading to a SaaS.` },
    ];
  }
  if (caps.runPhase === 'success') {
    return [
      { label: '📊 Is this sane', prompt: `Came out with ${caps.numLayers} layers averaging ${caps.avgBits.toFixed(2)} bits, ~${(caps.savingsRatio * 100).toFixed(1)}% saved. In 2-3 sentences as brain, assess: which deployment target this fits, what PPL ballpark to expect.` },
      { label: '🔬 How to verify', prompt: `Give a 4-step verification checklist (PPL bench / smoke prompts / long-context coherence / speed compare) to know if this mixed-precision was worth it.` },
      { label: '⚖️ vs uniform 4-bit', prompt: `In 2-3 sentences, contrast this mixed plan vs plain all-4-bit: which is smaller, which is safer, which should the user pick.` },
      { label: '📦 Next steps', prompt: `Output at ${caps.result?.output_dir ?? '/tmp/...'}. In 2-3 sentences, tell the user what's next (benchmark → device test → export).` },
    ];
  }
  if (caps.runPhase === 'failed') {
    return [
      { label: '🔥 Root cause', prompt: `Error: "${caps.result?.error ?? 'unknown'}". In 2-3 sentences as brain, infer the most likely root cause + the first triage command.` },
      { label: '🛠️ Retry strategy', prompt: `Which params would I change to make this pass? (e.g. 2-bit → 3-bit, fewer low-bit layers, larger group_size). Give concrete advice.` },
      { label: '⚙️ System vs config', prompt: `Is this a config issue (bad bit choice) or a system issue (OOM / mlx version)? Give a triage decision tree.` },
      { label: '🌍 Edge sovereignty', prompt: `Even on failure, the attempt stayed local. In 2-3 sentences, emphasise the fail-fast advantage of local execution.` },
    ];
  }
  if (caps.runPhase === 'running') {
    return [
      { label: '⏱️ How much longer', prompt: `Quantizing ${caps.numLayers} layers at avg ${caps.avgBits.toFixed(2)} bits — estimate remaining time + suggest what to watch.` },
      { label: '🎯 What to monitor', prompt: `While quant runs, what can the user observe (memory / progress / partial outputs)? In 2-3 sentences give a status checklist.` },
      { label: '⚠️ When to stop', prompt: `If it runs unreasonably long, what signals it's time to abort and reconfigure? Give a concrete decision rule.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: quantization is running locally on this machine across ${caps.numLayers} layers, zero network calls.` },
    ];
  }
  return [
    { label: '⚙️ Is this config sane', prompt: `${caps.numLayers} layers, avg ${caps.avgBits.toFixed(2)} bits, ${caps.pattern} pattern, estimated ${(caps.savingsRatio * 100).toFixed(1)}% savings. In 2-3 sentences as brain, assess this config + risks.` },
    { label: '🥪 Try sandwich', prompt: `If I switched to a sandwich pattern (first + last 8-bit, middle 3-bit), how much would I save, and how would PPL risk change?` },
    { label: '⚖️ Pick the right avg', prompt: `Current avg ${caps.avgBits.toFixed(2)} sits in the ${caps.avgBitsBucket} band. What deployment target does this band fit (iPhone / iPad / Mac)?` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this entire mixed-precision plan runs locally on this Mac, zero cloud calls.` },
  ];
}
