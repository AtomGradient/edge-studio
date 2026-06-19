// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * inferenceInsights — derived metrics + chat prompts for /inference page.
 *
 * Consumes a completed TraceResponse and surfaces "so what" insights:
 *  - aggregated chosen-token confidence (mean/min)
 *  - low-confidence step indices (where the model "hesitated")
 *  - bottleneck layers (top-K by total decode latency)
 *  - per-layer attn/mlp split for the model-self chat prompt
 *
 * Suggested prompts are tuned to drive interesting trace investigations:
 * "describe yourself", "summarize what you generated", "explain your slowest layer".
 */
import type { TraceResponse, ModelInfo } from '@/api/types';
import { formatParamCount } from '@/lib/utils';
import { deriveModelFacts } from '@/lib/chatPrompts';

type Locale = 'en' | 'zh';

export interface TraceMetrics {
  numTokensGenerated: number;
  totalTimeS: number;
  prefillTimeS: number;
  decodeTimeS: number;
  tokPerSec: number;
  prefillTPS: number;
  meanProb: number;
  minProb: number;
  /** Step indices where chosen_prob < 0.05 (model was "uncertain"). */
  uncertainStepIdxs: number[];
  /** Most-confident token (max chosen_prob). */
  mostConfidentToken: { idx: number; tok: string; prob: number } | null;
  /** Least-confident token (min chosen_prob). */
  leastConfidentToken: { idx: number; tok: string; prob: number } | null;
  /** Layers ranked by avg decode latency (top 5). Empty if !enable_timing. */
  decodeBottlenecks: Array<{ layer: number; attnMs: number; mlpMs: number; totalMs: number; bottleneckType: 'ATTN' | 'MLP' | 'BOTH' }>;
  /** Total decode-per-token latency derived from per-layer timings. */
  decodePerTokenMs: number;
  prefillTotalMs: number;
}

export function deriveTraceMetrics(trace: TraceResponse): TraceMetrics {
  const numTokens = trace.steps.length;
  const decodeTimeS = trace.total_time_seconds - trace.prefill_time_seconds;
  const tokPerSec = decodeTimeS > 0 ? numTokens / decodeTimeS : 0;
  const promptLen = trace.prompt_token_ids.length;
  const prefillTPS = trace.prefill_time_seconds > 0 ? promptLen / trace.prefill_time_seconds : 0;

  // Probability stats
  let meanProb = 0;
  let minProb = 1;
  let mostConf: { idx: number; tok: string; prob: number } | null = null;
  let leastConf: { idx: number; tok: string; prob: number } | null = null;
  const uncertainStepIdxs: number[] = [];
  for (const s of trace.steps) {
    meanProb += s.chosen_prob;
    if (s.chosen_prob < minProb) {
      minProb = s.chosen_prob;
      leastConf = { idx: s.step_idx, tok: s.token_str, prob: s.chosen_prob };
    }
    if (!mostConf || s.chosen_prob > mostConf.prob) {
      mostConf = { idx: s.step_idx, tok: s.token_str, prob: s.chosen_prob };
    }
    if (s.chosen_prob < 0.05) uncertainStepIdxs.push(s.step_idx);
  }
  meanProb = numTokens > 0 ? meanProb / numTokens : 0;

  // Latency aggregations (per-layer averages → bottleneck top-5)
  const numLayers = trace.num_layers;
  const decodeAttn = new Array(numLayers).fill(0);
  const decodeMlp = new Array(numLayers).fill(0);
  const decodeCounts = new Array(numLayers).fill(0);
  for (const step of trace.steps) {
    if (!step.layers) continue;
    for (const lt of step.layers) {
      if (lt.layer_idx < numLayers) {
        decodeAttn[lt.layer_idx] += lt.attn_latency_ms;
        decodeMlp[lt.layer_idx] += lt.mlp_latency_ms;
        decodeCounts[lt.layer_idx]++;
      }
    }
  }
  for (let i = 0; i < numLayers; i++) {
    if (decodeCounts[i] > 0) {
      decodeAttn[i] /= decodeCounts[i];
      decodeMlp[i] /= decodeCounts[i];
    }
  }
  const decodeBottlenecks = Array.from({ length: numLayers }, (_, i) => {
    const attnMs = decodeAttn[i] || 0;
    const mlpMs = decodeMlp[i] || 0;
    const totalMs = attnMs + mlpMs;
    const bottleneckType: 'ATTN' | 'MLP' | 'BOTH' =
      attnMs > mlpMs * 1.5 ? 'ATTN' : mlpMs > attnMs * 1.5 ? 'MLP' : 'BOTH';
    return { layer: i, attnMs, mlpMs, totalMs, bottleneckType };
  })
  .filter((b) => b.totalMs > 0)
  .sort((a, b) => b.totalMs - a.totalMs)
  .slice(0, 5);

  const decodePerTokenMs = decodeBottlenecks.reduce((s, b) => s + b.totalMs, 0)
    + decodeAttn.slice(decodeBottlenecks.length).reduce((s, a) => s + a, 0)
    + decodeMlp.slice(decodeBottlenecks.length).reduce((s, a) => s + a, 0);

  let prefillAttn = 0;
  let prefillMlp = 0;
  if (trace.prefill_layer_traces) {
    for (const lt of trace.prefill_layer_traces) {
      prefillAttn += lt.attn_latency_ms;
      prefillMlp += lt.mlp_latency_ms;
    }
  }
  const prefillTotalMs = prefillAttn + prefillMlp;

  return {
    numTokensGenerated: numTokens,
    totalTimeS: trace.total_time_seconds,
    prefillTimeS: trace.prefill_time_seconds,
    decodeTimeS,
    tokPerSec,
    prefillTPS,
    meanProb,
    minProb,
    uncertainStepIdxs,
    mostConfidentToken: mostConf,
    leastConfidentToken: leastConf,
    decodeBottlenecks,
    decodePerTokenMs,
    prefillTotalMs,
  };
}

export function buildTraceContextSnippet(trace: TraceResponse, m: TraceMetrics): string {
  const lines: string[] = [
    `## YOUR LATEST TRACE (Edge Studio captured it)`,
    `- Prompt: ${JSON.stringify(trace.prompt.slice(0, 200))}${trace.prompt.length > 200 ? '… (truncated)' : ''}`,
    `- Generated text: ${JSON.stringify(trace.generated_text.slice(0, 240))}${trace.generated_text.length > 240 ? '… (truncated)' : ''}`,
    `- ${m.numTokensGenerated} tokens generated in ${m.totalTimeS.toFixed(2)}s (${m.tokPerSec.toFixed(1)} tok/s decode)`,
    `- Prefill: ${m.prefillTimeS.toFixed(2)}s for ${trace.prompt_token_ids.length} prompt tokens (${m.prefillTPS.toFixed(0)} tok/s)`,
    `- Mean chosen-token probability: ${(m.meanProb * 100).toFixed(1)}%, min ${(m.minProb * 100).toFixed(1)}%`,
  ];
  if (m.mostConfidentToken) {
    lines.push(`- Most confident pick: ${JSON.stringify(m.mostConfidentToken.tok)} at step ${m.mostConfidentToken.idx} (p=${m.mostConfidentToken.prob.toFixed(3)})`);
  }
  if (m.leastConfidentToken) {
    lines.push(`- Least confident pick: ${JSON.stringify(m.leastConfidentToken.tok)} at step ${m.leastConfidentToken.idx} (p=${m.leastConfidentToken.prob.toFixed(3)})`);
  }
  if (m.uncertainStepIdxs.length > 0) {
    lines.push(`- ${m.uncertainStepIdxs.length} uncertain steps (p<0.05) at indices: ${m.uncertainStepIdxs.slice(0, 10).join(', ')}${m.uncertainStepIdxs.length > 10 ? '…' : ''}`);
  }
  if (m.decodeBottlenecks.length > 0) {
    lines.push(``, `### Decode bottlenecks (top by avg ms/layer):`);
    for (const b of m.decodeBottlenecks) {
      lines.push(`- Layer ${b.layer}: ${b.totalMs.toFixed(2)} ms/token (${b.bottleneckType.toLowerCase()}: attn ${b.attnMs.toFixed(2)} + mlp ${b.mlpMs.toFixed(2)})`);
    }
  } else if (!trace.enable_timing) {
    lines.push(``, `### Per-layer timing was NOT captured in this trace.`);
  }
  return lines.join('\n');
}

export function getInferenceSuggestedPrompts(
  trace: TraceResponse | null,
  m: TraceMetrics | null,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (!trace || !m) {
      return [
        { label: '🪞 介绍自己', prompt: '用 3-4 句话介绍你自己的架构, 端侧部署最适合什么任务。' },
        { label: '🧪 写代码', prompt: '用 Swift 写一个二叉树中序遍历的递归实现, 加注释。' },
        { label: '✍️ 创意写作', prompt: '用第一人称写 3 段话, 描述一个端侧 AI 模型在 iPhone 里观察世界的奇妙感受。' },
        { label: '🧮 推理题', prompt: '一个房间里有 3 盏灯和 3 个开关, 开关在房间外, 你只能进房间一次, 怎么确定每个开关对应哪盏灯？逐步推理。' },
      ];
    }
    return [
      { label: '🔬 解释生成', prompt: `用一段话解释你刚才生成的回答 (${trace.generated_text.slice(0, 60)}...) 的核心思路, 以及哪一步你最不确定 (token "${m.leastConfidentToken?.tok}" 概率仅 ${(m.minProb * 100).toFixed(1)}%) 为什么。` },
      { label: '⚡ 性能解读', prompt: `这次推理 prefill ${m.prefillTimeS.toFixed(2)}s + decode ${m.tokPerSec.toFixed(1)} tok/s. ${m.decodeBottlenecks.length > 0 ? `最慢的是 layer ${m.decodeBottlenecks[0].layer} (${m.decodeBottlenecks[0].totalMs.toFixed(1)} ms/token, 主要是 ${m.decodeBottlenecks[0].bottleneckType}). ` : ''}解释为什么这种 split 是合理的, 以及 iPhone 17 Pro 上预计的 tok/s。` },
      { label: '🎯 不确定性', prompt: m.uncertainStepIdxs.length > 0
        ? `检测到 ${m.uncertainStepIdxs.length} 步 chosen_prob < 0.05. 解释为什么会有这么多"不确定时刻", 以及它们通常出现在什么位置 (e.g., 句子开头/标点之后/技术术语前)。`
        : `这次回答的 mean prob ${(m.meanProb * 100).toFixed(1)}%, 整体很自信. 解释什么样的 prompt 会让你陷入"不确定"。` },
      { label: '🪞 自评质量', prompt: `从 LLM 自评的角度看, 你刚才的回答 ${trace.generated_text.length} 字, 是否充分回答了用户问题 ${JSON.stringify(trace.prompt.slice(0, 80))}? 给一个 1-10 分评分, 列出 1 个优点和 1 个改进点。` },
    ];
  }
  if (!trace || !m) {
    return [
      { label: '🪞 Introduce yourself', prompt: 'In 3-4 sentences, describe your architecture and what kind of task you are best at on-device.' },
      { label: '🧪 Write code', prompt: 'In Swift, write a recursive in-order traversal of a binary tree with comments.' },
      { label: '✍️ Creative writing', prompt: 'In first person, write 3 paragraphs describing the surreal experience of being an on-device AI model observing the world through an iPhone.' },
      { label: '🧮 Reasoning puzzle', prompt: 'A room has 3 lamps and 3 switches outside it. You can enter the room only once. How do you figure out which switch controls which lamp? Reason step by step.' },
    ];
  }
  return [
    { label: '🔬 Explain generation', prompt: `Walk me through your recent answer "${trace.generated_text.slice(0, 60)}…" — what was the core thought, and where were you least confident (token "${m.leastConfidentToken?.tok}" had only ${(m.minProb * 100).toFixed(1)}% probability) and why.` },
    { label: '⚡ Perf analysis', prompt: `This run: prefill ${m.prefillTimeS.toFixed(2)}s + decode ${m.tokPerSec.toFixed(1)} tok/s. ${m.decodeBottlenecks.length > 0 ? `Slowest layer was layer ${m.decodeBottlenecks[0].layer} (${m.decodeBottlenecks[0].totalMs.toFixed(1)} ms/token, mostly ${m.decodeBottlenecks[0].bottleneckType.toLowerCase()}). ` : ''}Explain why this split is reasonable and what tok/s I'd expect on an iPhone 17 Pro.` },
    { label: '🎯 Uncertainty', prompt: m.uncertainStepIdxs.length > 0
      ? `${m.uncertainStepIdxs.length} steps had chosen_prob < 0.05. Explain why these "uncertain moments" happen and where they typically appear (e.g., sentence starts, after punctuation, before technical terms).`
      : `This answer had a mean probability of ${(m.meanProb * 100).toFixed(1)}% — very confident overall. Explain what kinds of prompts would push you into uncertainty.` },
    { label: '🪞 Self-review', prompt: `From an LLM self-evaluation lens, did your ${trace.generated_text.length}-char answer fully address the prompt ${JSON.stringify(trace.prompt.slice(0, 80))}? Give a 1-10 score plus one strength and one weakness.` },
  ];
}

export function buildInferenceAutoBrief(
  trace: TraceResponse,
  m: TraceMetrics,
  locale: Locale,
): string {
  void trace;
  if (locale === 'zh') {
    return `用 2-3 句话总结这次推理: 你生成了 ${m.numTokensGenerated} 个 token (${m.tokPerSec.toFixed(1)} tok/s), 平均自信度 ${(m.meanProb * 100).toFixed(1)}%${m.uncertainStepIdxs.length > 0 ? `, 有 ${m.uncertainStepIdxs.length} 个不确定时刻` : ''}${m.decodeBottlenecks.length > 0 ? `, 最慢的 layer 是 ${m.decodeBottlenecks[0].layer}` : ''}. 邀请用户点 suggested 问题深入。不要列项。`;
  }
  return `In 2-3 sentences, summarize this run: you generated ${m.numTokensGenerated} tokens (${m.tokPerSec.toFixed(1)} tok/s), mean confidence ${(m.meanProb * 100).toFixed(1)}%${m.uncertainStepIdxs.length > 0 ? `, with ${m.uncertainStepIdxs.length} uncertain moments` : ''}${m.decodeBottlenecks.length > 0 ? `, and your slowest layer was layer ${m.decodeBottlenecks[0].layer}` : ''}. End by inviting the user to click a suggested question. No bullets.`;
}

/** Suggested example prompts to seed the input box on first load. */
export function getInferencePromptExamples(model: ModelInfo, locale: Locale): { label: string; prompt: string }[] {
  const f = deriveModelFacts(model);
  void f;
  if (locale === 'zh') {
    return [
      { label: '🪞 自介绍', prompt: '用一段话介绍你自己的架构和最适合的任务。' },
      { label: '🧪 代码', prompt: '用 Swift 实现快速排序, 带 docstring。' },
      { label: '🧮 推理', prompt: '7 个人坐成一圈, A 在 B 的左边, C 在 D 的右边, E 在 F 的对面, G 在 A 和 C 之间, 写出每个人的位置。' },
      { label: '✍️ 写诗', prompt: '用古典风格写一首关于"端侧 AI"的七言绝句, 押韵。' },
      { label: '💬 闲聊', prompt: '今天心情不太好, 帮我提一个让自己开心起来的小建议。' },
    ];
  }
  return [
    { label: '🪞 Introduce yourself', prompt: 'In one paragraph, introduce your architecture and the tasks you excel at.' },
    { label: '🧪 Code', prompt: 'In Swift, implement quicksort with a docstring and a usage example.' },
    { label: '🧮 Reasoning', prompt: 'Seven people sit in a circle: A is left of B, C is right of D, E faces F, G is between A and C. Write out each person\'s seat.' },
    { label: '✍️ Poem', prompt: 'Write a haiku about on-device AI, then briefly explain the imagery.' },
    { label: '💬 Casual', prompt: 'I\'m having a rough day. Suggest one small thing I can do right now to feel better.' },
  ];
}

export function formatTokens(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return String(n);
}

export function summaryLabel(model: ModelInfo): string {
  return `${model.model_name} · ${formatParamCount(model.total_params)} · ${model.quantization?.bits ?? '?'}-bit`;
}
