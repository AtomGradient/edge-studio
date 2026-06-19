// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * optimizationInsights — chat helpers + derived metrics for /optimization page.
 *
 * Edge Studio's advisor returns two buckets:
 *  - suggestions[] (applicable now, can execute)
 *  - requires_data[] (need profile/trace first)
 *
 * This file:
 *  - composes a model-aware "explain my optimization budget" system snippet
 *  - generates suggested questions tied to the current report (e.g.
 *    "you have N applicable suggestions, but X require profile — which one
 *    should I run first?")
 */
import type { OptimizationReport, ModelInfo } from '@/api/types';
import { formatSize } from '@/lib/utils';

type Locale = 'en' | 'zh';

export interface OptCounts {
  applicable: number;
  requiresData: number;
  byCategory: Record<string, number>;
  byPriority: Record<string, number>;
  totalSavingBytes: number;
}

export function deriveOptCounts(report: OptimizationReport): OptCounts {
  const counts: OptCounts = {
    applicable: report.suggestions.length,
    requiresData: report.requires_data.length,
    byCategory: {},
    byPriority: {},
    totalSavingBytes: report.total_estimated_saving_bytes,
  };
  for (const s of [...report.suggestions, ...report.requires_data]) {
    counts.byCategory[s.category] = (counts.byCategory[s.category] ?? 0) + 1;
    counts.byPriority[s.priority] = (counts.byPriority[s.priority] ?? 0) + 1;
  }
  return counts;
}

export function buildOptimizationContextSnippet(
  report: OptimizationReport,
  model: ModelInfo,
): string {
  const counts = deriveOptCounts(report);
  const lines: string[] = [
    `## YOUR OPTIMIZATION BUDGET (Edge Studio just generated it)`,
    `- Current size: ${formatSize(report.model_size_bytes)} (${model.quantization?.bits ?? '?'}-bit)`,
    `- ${counts.applicable} immediately applicable suggestions, ${counts.requiresData} need profile/trace first`,
    `- Total estimated saving (applicable only): ${formatSize(counts.totalSavingBytes)}`,
  ];
  if (report.suggestions.length > 0) {
    lines.push(``, `### Applicable suggestions:`);
    for (const s of report.suggestions.slice(0, 6)) {
      lines.push(`- [${s.priority.toUpperCase()}] ${s.title} — ${s.estimated_saving}, risk=${s.risk_level} (\`${s.category}\`)`);
    }
  }
  if (report.requires_data.length > 0) {
    lines.push(``, `### Requires more data (run profile / trace first):`);
    for (const s of report.requires_data.slice(0, 6)) {
      lines.push(`- ${s.title} (\`${s.category}\`)`);
    }
  }
  return lines.join('\n');
}

export function getOptimizationSuggestedPrompts(
  report: OptimizationReport | null,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (!report) {
    if (locale === 'zh') {
      return [
        { label: '🪞 自我评估', prompt: '在不查看具体建议的情况下, 用 3-4 句话讲讲你自己最值得优化的方向 (基于参数量 / 量化覆盖 / 架构特性)。' },
        { label: '⚖️ 量化策略', prompt: '解释 4-bit / 8-bit / int4+fp16 mixed 三种量化的 trade-off, 哪种最适合 iPhone 部署。' },
        { label: '✂️ 剪枝可行性', prompt: '神经元剪枝 / 层剪枝 / vocab 剪枝, 这三种各自适合什么场景, 用 1-2 句话各说一下。' },
      ];
    }
    return [
      { label: '🪞 Self-assessment', prompt: 'Without looking at specific suggestions, in 3-4 sentences describe what optimization opportunities your own architecture has (params / quant coverage / arch traits).' },
      { label: '⚖️ Quant strategy', prompt: 'Explain the trade-offs between 4-bit / 8-bit / int4+fp16 mixed precision. Which fits iPhone deployment best?' },
      { label: '✂️ Pruning fit', prompt: 'Briefly cover when neuron pruning / layer pruning / vocab pruning each makes sense.' },
    ];
  }
  const counts = deriveOptCounts(report);
  if (locale === 'zh') {
    return [
      { label: '🎯 先做哪一个', prompt: counts.applicable > 0
        ? `Edge Studio 给了 ${counts.applicable} 个可立即执行的建议. 按 ROI / 风险综合排序, 你建议我先做哪个? 解释理由。`
        : `所有建议都需要先生成 profile / trace. 你建议我先去 /activation 还是 /inference, 为什么?` },
      { label: '⚠️ 风险评估', prompt: '从你自己的视角看, 哪些建议如果激进执行会显著伤害质量? (例如剪过多 head, 量化 norm)' },
      { label: '📱 部署目标', prompt: `如果目标是 iPhone 17 Pro 流畅跑你, 当前 ${formatSize(report.model_size_bytes)} 需要压缩到多少? 给出一个建议路径。` },
      { label: '🪞 自评建议', prompt: 'Edge Studio 给的建议里, 有没有你觉得"漏掉的"或"应该提示但没提示的"? 用第一人称讲一下。' },
    ];
  }
  return [
    { label: '🎯 What to do first', prompt: counts.applicable > 0
      ? `Edge Studio surfaced ${counts.applicable} immediately-applicable suggestions. Rank them by ROI vs risk and tell me which to run first, with reasoning.`
      : `All suggestions require profile / trace first. Should I head to /activation or /inference first, and why?` },
    { label: '⚠️ Risk review', prompt: 'From your own perspective, which of these suggestions would noticeably hurt quality if pushed too aggressively? (e.g. over-pruning heads, quantizing norms)' },
    { label: '📱 Deployment target', prompt: `If the goal is smooth playback on iPhone 17 Pro, how much do I need to compress your current ${formatSize(report.model_size_bytes)}? Suggest a concrete path.` },
    { label: '🪞 Missing suggestions', prompt: 'Are there optimizations you think Edge Studio missed or under-prioritized? Speak in first person.' },
  ];
}

export function buildOptimizationAutoBrief(
  report: OptimizationReport,
  locale: Locale,
): string {
  const counts = deriveOptCounts(report);
  if (locale === 'zh') {
    return `用 2-3 句话总结你看到的优化机会: 你目前 ${formatSize(report.model_size_bytes)}${counts.applicable > 0 ? `, 有 ${counts.applicable} 个可立即执行的建议 (估计能省 ${formatSize(counts.totalSavingBytes)})` : ', 但所有建议都需要先生成 profile/trace'}. 邀请用户点 suggested 问题深入。不要列项。`;
  }
  return `In 2-3 sentences, summarize the optimization landscape you see: you are currently ${formatSize(report.model_size_bytes)}${counts.applicable > 0 ? `, with ${counts.applicable} immediately-applicable suggestions (could save ${formatSize(counts.totalSavingBytes)})` : ', but all suggestions need profile/trace first'}. End by inviting the user to click a suggested question. No bullets.`;
}
