// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * comparisonInsights — derived deltas + chat helpers for /comparison page.
 *
 * Backend's /api/compare returns rich data (arch diff + per-layer latency +
 * bottlenecks) but the page renders it as neutral tables. This file:
 *  - parses arch_diff into a structured delta map (size / params / GQA / quant)
 *  - aggregates bottleneck profile (attn-bound vs mlp-bound)
 *  - composes a model-aware "compare A vs B" system snippet
 *  - generates result-aware suggested questions
 */
import type { ModelInfo, ComparisonResult } from '@/api/types';
import { formatSize, formatParamCount } from '@/lib/utils';
import { deriveModelFacts } from '@/lib/chatPrompts';

type Locale = 'en' | 'zh';

export interface ComparisonDeltas {
  /** A divided by B for major numeric fields. >1 means A is bigger. */
  paramRatio: number;
  sizeRatio: number;
  contextRatio: number;
  /** A's TPS / B's TPS (1.0 = same speed). */
  tpsRatio: number;
  /** A's prefill_ms / B's prefill_ms. */
  prefillRatio: number;
  /** Pretty difference summary (e.g. "A is 2.4× larger") for one-line UI. */
  oneLineSize: string;
  oneLineSpeed: string;
  /** Per-side bottleneck profile from bottlenecks list. */
  profileA: BottleneckProfile;
  profileB: BottleneckProfile;
  /** True if both share the same model_type (same family). */
  sameFamily: boolean;
  quantBitsA: number;
  quantBitsB: number;
  quantDelta: number;
}

export interface BottleneckProfile {
  attnLayers: number;
  mlpLayers: number;
  bothLayers: number;
  dominant: 'attn' | 'mlp' | 'both' | 'none';
  totalMs: number;
}

function summarizeBottlenecks(layers: ComparisonResult['bottlenecks_a']): BottleneckProfile {
  const counts = { attn: 0, mlp: 0, both: 0 };
  let totalMs = 0;
  for (const b of layers) {
    totalMs += b.total_ms;
    const t = b.bottleneck_type.toLowerCase();
    if (t === 'attn') counts.attn++;
    else if (t === 'mlp') counts.mlp++;
    else counts.both++;
  }
  const max = Math.max(counts.attn, counts.mlp, counts.both);
  const dominant: BottleneckProfile['dominant'] =
    max === 0 ? 'none'
    : counts.attn === max ? 'attn'
    : counts.mlp === max ? 'mlp'
    : 'both';
  return { attnLayers: counts.attn, mlpLayers: counts.mlp, bothLayers: counts.both, dominant, totalMs };
}

function fmtRatio(r: number, locale: Locale): string {
  if (!isFinite(r) || r <= 0) return '—';
  if (r >= 1) {
    return locale === 'zh' ? `A 比 B 大 ${((r - 1) * 100).toFixed(0)}%` : `A is ${((r - 1) * 100).toFixed(0)}% larger`;
  }
  return locale === 'zh' ? `B 比 A 大 ${((1 / r - 1) * 100).toFixed(0)}%` : `B is ${((1 / r - 1) * 100).toFixed(0)}% larger`;
}

function fmtSpeedRatio(r: number, locale: Locale): string {
  if (!isFinite(r) || r <= 0) return '—';
  if (Math.abs(r - 1) < 0.05) return locale === 'zh' ? '速度相当' : 'similar speed';
  if (r >= 1) {
    return locale === 'zh' ? `A 快 ${((r - 1) * 100).toFixed(0)}%` : `A is ${((r - 1) * 100).toFixed(0)}% faster`;
  }
  return locale === 'zh' ? `B 快 ${((1 / r - 1) * 100).toFixed(0)}%` : `B is ${((1 / r - 1) * 100).toFixed(0)}% faster`;
}

export function deriveComparisonDeltas(
  model: ModelInfo | null,
  comp: ModelInfo | null,
  result: ComparisonResult | null,
  locale: Locale,
): ComparisonDeltas | null {
  if (!model || !comp) return null;
  const fA = deriveModelFacts(model);
  const fB = deriveModelFacts(comp);
  const paramRatio = fB.totalParams > 0 ? fA.totalParams / fB.totalParams : 1;
  const sizeRatio = fB.totalSizeBytes > 0 ? fA.totalSizeBytes / fB.totalSizeBytes : 1;
  const contextRatio = fB.maxCtx > 0 ? fA.maxCtx / fB.maxCtx : 1;
  const tpsRatio = result?.latency_a && result?.latency_b && result.latency_b.tokens_per_second > 0
    ? result.latency_a.tokens_per_second / result.latency_b.tokens_per_second
    : 1;
  const prefillRatio = result?.latency_a && result?.latency_b && result.latency_b.prefill_total_ms > 0
    ? result.latency_a.prefill_total_ms / result.latency_b.prefill_total_ms
    : 1;
  return {
    paramRatio, sizeRatio, contextRatio, tpsRatio, prefillRatio,
    oneLineSize: fmtRatio(sizeRatio, locale),
    oneLineSpeed: result?.latency_a && result?.latency_b ? fmtSpeedRatio(tpsRatio, locale) : (locale === 'zh' ? '需先 Run Comparison' : 'run comparison first'),
    profileA: summarizeBottlenecks(result?.bottlenecks_a ?? []),
    profileB: summarizeBottlenecks(result?.bottlenecks_b ?? []),
    sameFamily: model.model_type === comp.model_type,
    quantBitsA: fA.quantBits,
    quantBitsB: fB.quantBits,
    quantDelta: fA.quantBits - fB.quantBits,
  };
}

export function buildComparisonContextSnippet(
  model: ModelInfo,
  comp: ModelInfo,
  result: ComparisonResult | null,
): string {
  const fA = deriveModelFacts(model);
  const fB = deriveModelFacts(comp);
  const lines: string[] = [
    `## YOUR COMPARISON CONTEXT`,
    `You are speaking AS ${model.model_name} (the primary). The user is comparing you against ${comp.model_name}.`,
    ``,
    `### Side-by-side facts`,
    `| Field | ${model.model_name} | ${comp.model_name} |`,
    `|---|---|---|`,
    `| Family | ${fA.family} | ${fB.family} |`,
    `| Logical params | ${formatParamCount(fA.totalParams)} | ${formatParamCount(fB.totalParams)} |`,
    `| Disk size | ${formatSize(fA.totalSizeBytes)} | ${formatSize(fB.totalSizeBytes)} |`,
    `| Quantization | ${fA.quantBits}-bit (g${fA.groupSize}) | ${fB.quantBits}-bit (g${fB.groupSize}) |`,
    `| Layers | ${fA.numLayers} | ${fB.numLayers} |`,
    `| Hidden | ${fA.hiddenSize} | ${fB.hiddenSize} |`,
    `| FFN | ${fA.ffnSize} | ${fB.ffnSize} |`,
    `| Q heads / KV heads | ${fA.numHeads}/${fA.numKVHeads} (GQA ${fA.gqaRatio}:1) | ${fB.numHeads}/${fB.numKVHeads} (GQA ${fB.gqaRatio}:1) |`,
    `| Max context | ${fA.maxCtx.toLocaleString()} | ${fB.maxCtx.toLocaleString()} |`,
    `| Vocabulary | ${fA.vocabSize.toLocaleString()} | ${fB.vocabSize.toLocaleString()} |`,
  ];
  if (result?.latency_a && result?.latency_b) {
    const la = result.latency_a;
    const lb = result.latency_b;
    lines.push(
      ``,
      `### Measured latency on the same prompt (Edge Studio just ran it)`,
      `- ${la.model_name}: ${la.tokens_per_second.toFixed(2)} tok/s, prefill ${la.prefill_total_ms.toFixed(1)} ms, ${la.decode_steps} decode steps`,
      `- ${lb.model_name}: ${lb.tokens_per_second.toFixed(2)} tok/s, prefill ${lb.prefill_total_ms.toFixed(1)} ms, ${lb.decode_steps} decode steps`,
      ``,
      `### Top bottleneck layers`,
    );
    if (result.bottlenecks_a.length > 0) {
      lines.push(`${la.model_name}: ` + result.bottlenecks_a.slice(0, 3).map((b) =>
        `L${b.layer_idx} (${b.bottleneck_type} ${b.total_ms.toFixed(1)}ms)`,
      ).join(', '));
    }
    if (result.bottlenecks_b.length > 0) {
      lines.push(`${lb.model_name}: ` + result.bottlenecks_b.slice(0, 3).map((b) =>
        `L${b.layer_idx} (${b.bottleneck_type} ${b.total_ms.toFixed(1)}ms)`,
      ).join(', '));
    }
  } else {
    lines.push(``, `### Latency: not yet measured (user has not clicked Run Comparison).`);
  }
  return lines.join('\n');
}

export function getComparisonSuggestedPrompts(
  model: ModelInfo | null,
  comp: ModelInfo | null,
  result: ComparisonResult | null,
  deltas: ComparisonDeltas | null,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (!model || !comp || !deltas) return [];
  const aName = model.model_name;
  const bName = comp.model_name;
  if (locale === 'zh') {
    return [
      { label: '🎯 哪个更适合我', prompt: result
        ? `基于上面的所有 side-by-side 数据 + 实测延迟 (我=${result.latency_a?.tokens_per_second.toFixed(1) || '?'} tok/s, ${bName}=${result.latency_b?.tokens_per_second.toFixed(1) || '?'} tok/s), 直白告诉用户:如果目标是 iPhone 17 Pro 流畅日常对话, 应该选我还是 ${bName}? 给出明确推荐 + 一句理由。`
        : `从架构和参数维度看, 如果目标是 iPhone 17 Pro 日常对话, 你 (${aName}) 和 ${bName} 哪个更合适? 给出推荐 + 一句理由。` },
      { label: '⚡ 速度差距',  prompt: result?.latency_a && result?.latency_b
        ? `${deltas.oneLineSpeed}. 解释这个 TPS 差距背后最可能的 1-2 个根因 (例如: ${deltas.quantDelta !== 0 ? `量化差 ${Math.abs(deltas.quantDelta)} bits` : '相同量化但'}${deltas.profileA.dominant !== deltas.profileB.dominant ? `, bottleneck 类型不同 (我 ${deltas.profileA.dominant} vs ${bName} ${deltas.profileB.dominant})` : ''})。`
        : `根据规模差异, 你预计在 iPhone 上比 ${bName} 快还是慢? 列 1-2 条推断依据。` },
      { label: '🧬 架构差异', prompt: deltas.sameFamily
        ? `你和 ${bName} 同属 ${model.model_type} family. 主要区别是${deltas.paramRatio !== 1 ? ` 参数量 (${deltas.oneLineSize})` : ''}${deltas.quantDelta !== 0 ? ` 和 量化等级 (我 ${deltas.quantBitsA}-bit vs B ${deltas.quantBitsB}-bit)` : ''}. 解释这种"同 family 不同 size/quant"的取舍。`
        : `你属 ${model.model_type}, ${bName} 属 ${comp.model_type} — 跨 family. 解释这两类架构的核心设计差异 (attention 实现 / MoE 与否 / hybrid 与否) 对端侧推理的影响。` },
      { label: '🪞 自评劣势', prompt: `诚实告诉用户: 在哪 1-2 个具体场景下 ${bName} 会比你做得更好? 不要客套, 用第一人称承认弱点。` },
    ];
  }
  return [
    { label: '🎯 Which to choose', prompt: result
      ? `Based on all the side-by-side data + measured latency (me=${result.latency_a?.tokens_per_second.toFixed(1) || '?'} tok/s, ${bName}=${result.latency_b?.tokens_per_second.toFixed(1) || '?'} tok/s), tell the user plainly: for smooth daily chat on an iPhone 17 Pro, should they pick me or ${bName}? Give a clear recommendation + one-line reason.`
      : `From architecture and parameters alone, for daily chat on iPhone 17 Pro, would I (${aName}) or ${bName} be a better fit? Recommend + one-line reason.` },
    { label: '⚡ Speed gap', prompt: result?.latency_a && result?.latency_b
      ? `${deltas.oneLineSpeed}. Explain the 1-2 most likely root causes for this TPS gap (e.g., ${deltas.quantDelta !== 0 ? `${Math.abs(deltas.quantDelta)}-bit quantization difference` : 'same quant but'}${deltas.profileA.dominant !== deltas.profileB.dominant ? `, different bottleneck profile (I'm ${deltas.profileA.dominant}-bound vs ${bName} is ${deltas.profileB.dominant}-bound)` : ''}).`
      : `From the scale difference, would you expect to run faster or slower than ${bName} on an iPhone? Give 1-2 lines of reasoning.` },
    { label: '🧬 Arch differences', prompt: deltas.sameFamily
      ? `${bName} and I share the ${model.model_type} family. The main differences are${deltas.paramRatio !== 1 ? ` parameter count (${deltas.oneLineSize})` : ''}${deltas.quantDelta !== 0 ? ` and quantization (me ${deltas.quantBitsA}-bit vs ${bName} ${deltas.quantBitsB}-bit)` : ''}. Explain the trade-offs of "same family, different size/quant".`
      : `I'm ${model.model_type}, ${bName} is ${comp.model_type} — cross-family. Explain the core design differences (attention impl / MoE / hybrid) and what each means for on-device inference.` },
    { label: '🪞 Honest weakness', prompt: `Honestly, in which 1-2 concrete scenarios would ${bName} outperform me? No flattery — admit weaknesses in first person.` },
  ];
}

export function buildComparisonAutoBrief(
  model: ModelInfo,
  comp: ModelInfo,
  result: ComparisonResult | null,
  deltas: ComparisonDeltas,
  locale: Locale,
): string {
  void model;
  if (locale === 'zh') {
    if (result?.latency_a && result?.latency_b) {
      return `用 2-3 句话总结你和 ${comp.model_name} 这次对比的核心结论 (规模 ${deltas.oneLineSize}, 速度 ${deltas.oneLineSpeed}, 是否同 family). 结尾邀请用户点 suggested 问题深入 (尤其哪个更适合特定设备)。不要列项。`;
    }
    return `用 2-3 句话总结你和 ${comp.model_name} 在架构和规模上的核心差异 (${deltas.oneLineSize}, ${deltas.sameFamily ? '同 family' : '跨 family'}). 提示用户点 Run Comparison 获得实测延迟对比。不要列项。`;
  }
  if (result?.latency_a && result?.latency_b) {
    return `In 2-3 sentences, summarize the headline of this comparison vs ${comp.model_name}: scale (${deltas.oneLineSize}), speed (${deltas.oneLineSpeed}), and whether you share a family. End by inviting the user to click a suggested question (especially "which fits which device"). No bullets.`;
  }
  return `In 2-3 sentences, summarize the structural difference between you and ${comp.model_name} (${deltas.oneLineSize}, ${deltas.sameFamily ? 'same family' : 'cross family'}). Hint that clicking Run Comparison gives measured latency. No bullets.`;
}

export function deviceFitVerdict(
  fA: ReturnType<typeof deriveModelFacts>,
  fB: ReturnType<typeof deriveModelFacts>,
  iphoneAvailableMB = 6500,
): { aFits: boolean; bFits: boolean; aMargin: number; bMargin: number } {
  // very rough: weights + KV @4K + 20% activation overhead
  const aTotalMB = fA.totalSizeBytes / 1e6 + fA.kvAt4kBytes / 1e6 + (fA.totalSizeBytes / 1e6) * 0.2;
  const bTotalMB = fB.totalSizeBytes / 1e6 + fB.kvAt4kBytes / 1e6 + (fB.totalSizeBytes / 1e6) * 0.2;
  return {
    aFits: aTotalMB <= iphoneAvailableMB,
    bFits: bTotalMB <= iphoneAvailableMB,
    aMargin: iphoneAvailableMB - aTotalMB,
    bMargin: iphoneAvailableMB - bTotalMB,
  };
}
