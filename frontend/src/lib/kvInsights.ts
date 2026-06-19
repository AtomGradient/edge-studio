// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * kvInsights — derived KV cache insights for the /kv-cache page.
 *
 * Backend's /kv-report assumes uniform full attention across all layers.
 * For hybrid models (Qwen3.5 = 8 FA + 24 GDN, Gemma3 = sliding window) the
 * reported KV is wildly overestimated. This file:
 *  - detects hybrid layouts from model.config (layer_types / sliding_window)
 *  - corrects the reported KV by the actual full-attention ratio
 *  - derives GQA savings, @N context cards, DSR budget recommendations
 *  - builds chat system prompt + suggested questions specialized for KV story
 *
 * No new backend; everything is computed from values already in
 * ModelInfo + KVReportResponse.
 */
import type { ModelInfo, KVReportResponse } from '@/api/types';
import { deriveModelFacts } from '@/lib/chatPrompts';
import { formatSize } from '@/lib/utils';

type Locale = 'en' | 'zh';

function unwrapTextConfig(c: Record<string, unknown> | null | undefined): Record<string, unknown> {
  if (!c) return {};
  const tc = c['text_config'];
  return tc && typeof tc === 'object' ? (tc as Record<string, unknown>) : c;
}

export interface AttentionLayout {
  /** "uniform" = standard full-attention every layer.
   *  "hybrid"  = mix of full + linear/SSM (Qwen3.5 etc).
   *  "sliding" = full but sliding window cap (Gemma3 etc). */
  kind: 'uniform' | 'hybrid' | 'sliding';
  totalLayers: number;
  /** Number of layers that contribute to KV growth. */
  fullAttnLayers: number;
  /** If sliding, the window in tokens (else 0). */
  slidingWindow: number;
  /** Reported KV ÷ corrected KV (≥1). 1 means no correction. */
  overestimateRatio: number;
  /** Friendly label for display. */
  label: string;
  /** Histogram of layer types if mixed. */
  layerTypeCounts: Record<string, number>;
}

/** Fallback KV bytes/token computation when backend fails to parse nested config.
 *  Standard formula: 2 (K+V) × num_kv_heads × head_dim × 2 (fp16) × num_layers.
 *  This is the *raw, uniform-attention* number; hybrid correction is applied
 *  separately by the caller via overestimateRatio. */
export function computeBytesPerTokenFallback(model: ModelInfo): number {
  const cfg = unwrapTextConfig(model.config as Record<string, unknown>);
  const numLayers = (cfg.num_hidden_layers as number) || model.num_layers || 0;
  const numKVHeads = (cfg.num_key_value_heads as number) || model.num_kv_heads || 0;
  const numHeads = (cfg.num_attention_heads as number) || model.num_attention_heads || 0;
  const hiddenSize = (cfg.hidden_size as number) || model.hidden_size || 0;
  const headDim = (cfg.head_dim as number) || (hiddenSize && numHeads ? hiddenSize / numHeads : 0);
  return 2 * numKVHeads * headDim * 2 * numLayers;  // fp16
}

/** Derive a usable bytes/token: prefer backend report, fall back to model.config. */
export function effectiveBytesPerToken(report: { bytes_per_token: number } | null, model: ModelInfo): number {
  if (report && report.bytes_per_token > 0) return report.bytes_per_token;
  return computeBytesPerTokenFallback(model);
}

export function detectAttentionLayout(model: ModelInfo): AttentionLayout {
  const cfg = unwrapTextConfig(model.config as Record<string, unknown>);
  const reportedLayers = (cfg.num_hidden_layers as number) || model.num_layers || 0;
  const layerTypes = Array.isArray(cfg.layer_types) ? (cfg.layer_types as string[]) : [];
  const slidingWindow = (cfg.sliding_window as number) || 0;

  const layerTypeCounts: Record<string, number> = {};
  if (layerTypes.length > 0) {
    for (const lt of layerTypes) layerTypeCounts[lt] = (layerTypeCounts[lt] ?? 0) + 1;
    const fullKeys = Object.keys(layerTypeCounts).filter((k) => /full|sdpa|attention$|^attn$/i.test(k) && !/linear|gdn|mamba|ssm/i.test(k));
    let fullCount = 0;
    if (fullKeys.length > 0) {
      for (const k of fullKeys) fullCount += layerTypeCounts[k];
    } else {
      // Fallback: anything not mentioning linear/gdn/mamba/ssm is full
      for (const [k, n] of Object.entries(layerTypeCounts)) {
        if (!/linear|gdn|mamba|ssm|conv/i.test(k)) fullCount += n;
      }
    }
    if (fullCount > 0 && fullCount < reportedLayers) {
      return {
        kind: 'hybrid',
        totalLayers: reportedLayers,
        fullAttnLayers: fullCount,
        slidingWindow: 0,
        overestimateRatio: reportedLayers / fullCount,
        label: `Hybrid: ${fullCount} full-attn / ${reportedLayers - fullCount} linear/GDN`,
        layerTypeCounts,
      };
    }
  }

  if (slidingWindow > 0) {
    return {
      kind: 'sliding',
      totalLayers: reportedLayers,
      fullAttnLayers: reportedLayers,
      slidingWindow,
      overestimateRatio: 1,
      label: `Sliding window ${slidingWindow.toLocaleString()} tok cap`,
      layerTypeCounts,
    };
  }

  return {
    kind: 'uniform',
    totalLayers: reportedLayers,
    fullAttnLayers: reportedLayers,
    slidingWindow: 0,
    overestimateRatio: 1,
    label: 'Uniform full-attention',
    layerTypeCounts,
  };
}

export interface ContextCard {
  seqLen: number;
  label: string;             // "1K", "4K", "32K", "max"
  totalMB: number;           // raw report total
  totalCorrectedMB: number;  // adjusted for hybrid/sliding
  kvMB: number;              // raw kv only
  kvCorrectedMB: number;     // adjusted
  fittingDeviceCount: number;
  smallestFitName: string | null;
}

const STANDARD_BUCKETS_TOKEN: { len: number; label: string }[] = [
  { len: 1024,   label: '1K' },
  { len: 4096,   label: '4K' },
  { len: 8192,   label: '8K' },
  { len: 32768,  label: '32K' },
  { len: 131072, label: '128K' },
];

/** Pick context buckets that fit within the model's actual max_position_embeddings. */
export function buildContextCards(
  report: KVReportResponse,
  model: ModelInfo,
  layout: AttentionLayout,
): ContextCard[] {
  const cfg = unwrapTextConfig(model.config as Record<string, unknown>);
  const maxCtx = (cfg.max_position_embeddings as number) || 0;

  // Effective KV per token after hybrid correction (with fallback if backend parsing failed)
  const rawBytesPerToken = effectiveBytesPerToken(report, model);
  const effectiveKvBytesPerToken = rawBytesPerToken / layout.overestimateRatio;
  const slidingCap = layout.slidingWindow;

  // Find closest curve point
  function pointAt(seqLen: number) {
    let closest = report.memory_curve[0];
    let minDist = Math.abs(closest.seq_len - seqLen);
    for (const p of report.memory_curve) {
      const d = Math.abs(p.seq_len - seqLen);
      if (d < minDist) { minDist = d; closest = p; }
    }
    return closest;
  }

  const buckets = STANDARD_BUCKETS_TOKEN.filter((b) => b.len <= maxCtx);
  if (maxCtx > 0 && (buckets.length === 0 || buckets[buckets.length - 1].len < maxCtx)) {
    // Always include actual max as the last bucket
    const maxLabel = maxCtx >= 1024 ? `${(maxCtx / 1024).toFixed(0)}K (max)` : `${maxCtx}`;
    buckets.push({ len: maxCtx, label: maxLabel });
  }

  return buckets.map((b) => {
    const p = pointAt(b.len);
    // For sliding window, KV is capped at window size
    const effectiveLen = slidingCap > 0 ? Math.min(b.len, slidingCap) : b.len;
    const kvCorrectedMB = (effectiveKvBytesPerToken * effectiveLen) / 1e6;
    // Note: report.activation/overhead may also be 0 when backend failed parsing.
    // We at least surface model_weights + KV; if backend gave 0 for activation,
    // estimate ~5% of weights as a baseline.
    const baselineActivation = p.activation_mb > 0 ? p.activation_mb : p.model_weights_mb * 0.05;
    const baselineOverhead = p.overhead_mb > 0 ? p.overhead_mb : 100;
    const totalCorrectedMB = p.model_weights_mb + kvCorrectedMB + baselineActivation + baselineOverhead;

    // Device fit using corrected total
    const fittingDevices = report.device_capacities.filter((d) => d.available_mb >= totalCorrectedMB);
    fittingDevices.sort((a, b2) => a.ram_gb - b2.ram_gb);

    return {
      seqLen: b.len,
      label: b.label,
      totalMB: p.total_mb,
      totalCorrectedMB,
      kvMB: p.kv_cache_mb,
      kvCorrectedMB,
      fittingDeviceCount: fittingDevices.length,
      smallestFitName: fittingDevices[0]?.device_name ?? null,
    };
  });
}

export interface DSRBudgetRec {
  device_name: string;
  ram_gb: number;
  available_mb: number;
  recommendedBudget: number;        // tokens
  budgetReason: string;
}

/** Recommend DSR eviction budgets per device given the corrected KV cost per token. */
export function recommendDSRBudgets(
  report: KVReportResponse,
  layout: AttentionLayout,
  model: ModelInfo,
): DSRBudgetRec[] {
  const rawBytesPerToken = effectiveBytesPerToken(report, model);
  const effectiveKvBytesPerToken = rawBytesPerToken / layout.overestimateRatio;
  const recs: DSRBudgetRec[] = [];
  const TARGET_KV_FRACTION = 0.35;  // leave headroom for activations + scratch
  const MIN_BUDGET = 1024;
  const MAX_BUDGET = 32768;

  for (const d of report.device_capacities) {
    const usableForKV = (d.available_mb - report.model_weights_mb) * 1e6 * TARGET_KV_FRACTION;
    let budget = effectiveKvBytesPerToken > 0
      ? Math.floor(usableForKV / effectiveKvBytesPerToken)
      : 0;
    let reason: string;
    if (budget < MIN_BUDGET) {
      budget = MIN_BUDGET;
      reason = 'Floor — device is tight, expect aggressive eviction';
    } else if (budget > MAX_BUDGET) {
      budget = MAX_BUDGET;
      reason = 'Plenty of headroom — cap at 32K, no need to evict';
    } else {
      // round to nearest 1024
      budget = Math.round(budget / 1024) * 1024;
      reason = `${(TARGET_KV_FRACTION * 100).toFixed(0)}% of available memory dedicated to KV`;
    }
    recs.push({
      device_name: d.device_name,
      ram_gb: d.ram_gb,
      available_mb: d.available_mb,
      recommendedBudget: budget,
      budgetReason: reason,
    });
  }
  return recs;
}

/** Pure GQA savings vs MHA equivalent (% reduction in KV cache). */
export function gqaSavingPct(model: ModelInfo): number {
  const f = deriveModelFacts(model);
  return f.gqaSavingPct;
}

/** Build chat system prompt with KV-specific knowledge appended. */
export function buildKVContextSnippet(
  model: ModelInfo,
  report: KVReportResponse | null,
  layout: AttentionLayout,
): string {
  const lines: string[] = [
    `## YOUR KV-CACHE PROFILE (Edge Studio computed it)`,
  ];
  if (report) {
    const rawBytes = effectiveBytesPerToken(report, model);
    const usedFallback = rawBytes !== report.bytes_per_token;
    const facts = deriveModelFacts(model);
    lines.push(
      `- ${facts.numLayers} layers, ${facts.numKVHeads} KV heads, head_dim=${facts.headDim}`,
      `- Raw KV/token: ${(rawBytes / 1024).toFixed(2)} KB (assuming uniform full-attn)${usedFallback ? ' [computed from model.config — backend report missing]' : ''}`,
    );
  }
  if (layout.kind === 'hybrid') {
    const realKvBytes = report ? report.bytes_per_token / layout.overestimateRatio : 0;
    lines.push(
      `- ⚠️  HYBRID ATTENTION: only ${layout.fullAttnLayers}/${layout.totalLayers} layers do full self-attn (rest are linear / GDN with constant-size SSM state).`,
      `- Effective KV/token ≈ ${(realKvBytes / 1024).toFixed(2)} KB — backend's reported number is ${layout.overestimateRatio.toFixed(2)}× too high.`,
      `- Layer-type histogram: ${Object.entries(layout.layerTypeCounts).map(([k, n]) => `${n}× ${k}`).join(', ')}`,
    );
  } else if (layout.kind === 'sliding') {
    lines.push(
      `- ⚠️  SLIDING WINDOW: KV is capped at ${layout.slidingWindow.toLocaleString()} tokens regardless of how long the conversation grows.`,
    );
  } else {
    lines.push(`- Uniform full-attention every layer (KV grows linearly).`);
  }
  if (report) {
    const cards = buildContextCards(report, model, layout);
    if (cards.length > 0) {
      lines.push(``, `### Total memory at common context lengths:`);
      for (const c of cards) {
        lines.push(`- ${c.label} ctx: ~${c.totalCorrectedMB.toFixed(0)} MB (KV alone ${c.kvCorrectedMB.toFixed(0)} MB)${c.smallestFitName ? `, smallest device that fits = ${c.smallestFitName}` : ', NO listed device fits'}`);
      }
    }
  }
  return lines.join('\n');
}

export function getKVSuggestedPrompts(
  model: ModelInfo,
  report: KVReportResponse | null,
  layout: AttentionLayout,
  locale: Locale,
): { label: string; prompt: string }[] {
  const f = deriveModelFacts(model);
  if (locale === 'zh') {
    return [
      { label: '🪞 自己讲讲 KV', prompt: '用 3-4 句话讲讲你自己的 KV cache 设计：每 token 多少 KB、GQA 节省了多少、context 多长会触发设备瓶颈。' },
      { label: '⚖️ GQA 收益', prompt: `你有 ${f.numHeads} 个 query head 但只有 ${f.numKVHeads} 个 KV head (GQA ${f.gqaRatio}:1)。详细解释这种 ratio 对推理速度和内存的影响，以及当年为什么从 MHA 改 GQA。` },
      { label: layout.kind === 'hybrid' ? '🧬 Hybrid 真相' : '📈 长上下文', prompt: layout.kind === 'hybrid'
        ? `你是 hybrid attention 模型 (${layout.fullAttnLayers}/${layout.totalLayers} 层做 full attention, 剩下走 linear/GDN)。解释为什么只有部分层贡献 KV、剩下层用什么状态机制，对长上下文有什么优势。`
        : `你能撑到 ${(f.maxCtx / 1024).toFixed(0)}K context, 解释长上下文对 KV 的实际增长曲线，以及哪些技巧 (sliding window / sink token / DSR / quantize KV) 能延伸有效长度。` },
      { label: '📱 真机部署', prompt: report ? `iPhone 17 Pro 上跑你能开多大 context (考虑 weights + KV + activations)？给具体数字。` : 'iPhone 17 Pro 上跑你大概能开多大 context？' },
    ];
  }
  return [
    { label: '🪞 Walk through your KV', prompt: 'In 3-4 sentences, describe your KV cache design: KB per token, GQA savings, what context length triggers device bottleneck.' },
    { label: '⚖️ GQA payoff', prompt: `You have ${f.numHeads} query heads but only ${f.numKVHeads} KV heads (GQA ${f.gqaRatio}:1). Explain in detail how this ratio affects inference speed + memory, and why MHA was abandoned in favor of GQA.` },
    { label: layout.kind === 'hybrid' ? '🧬 Hybrid truth' : '📈 Long context', prompt: layout.kind === 'hybrid'
      ? `You are a hybrid-attention model (${layout.fullAttnLayers}/${layout.totalLayers} layers run full self-attn, the rest run linear/GDN). Explain why only some layers contribute to KV, what state mechanism the rest use, and the long-context advantage.`
      : `You can stretch to ${(f.maxCtx / 1024).toFixed(0)}K context. Explain the actual KV growth curve and which tricks (sliding window / sink token / DSR / KV quantization) extend usable length.` },
    { label: '📱 Real device fit', prompt: report ? `What max context can I sustain on an iPhone 17 Pro accounting for weights + KV + activations? Give concrete numbers.` : 'Roughly what max context can I sustain on an iPhone 17 Pro?' },
  ];
}

export function buildKVAutoBrief(
  model: ModelInfo,
  layout: AttentionLayout,
  locale: Locale,
): string {
  void model;
  if (locale === 'zh') {
    if (layout.kind === 'hybrid') {
      return `用 2-3 句直白话指出你 KV cache 最关键的特征 (强调你是 hybrid attention，只有 ${layout.fullAttnLayers}/${layout.totalLayers} 层贡献 KV，所以后端 KV 报告高估了 ${layout.overestimateRatio.toFixed(1)} 倍)。结尾邀请用户点 suggested 问题深入。不要列项。`;
    }
    if (layout.kind === 'sliding') {
      return `用 2-3 句话解释你的 sliding-window KV 策略 (window=${layout.slidingWindow})，以及对长对话的好处。不要列项。`;
    }
    return `用 2-3 句话总结你的 KV cache 在端侧部署上的两个关键事实 (例如 GQA 节省比、KB/token、长 context 内存增长)。不要列项。`;
  }
  if (layout.kind === 'hybrid') {
    return `In 2-3 plain sentences, surface your most important KV-cache fact (emphasize that you are a hybrid-attention model, only ${layout.fullAttnLayers}/${layout.totalLayers} layers contribute to KV, so the backend's KV report overestimates by ${layout.overestimateRatio.toFixed(1)}×). End by inviting the user to click a suggested question. No bullet points.`;
  }
  if (layout.kind === 'sliding') {
    return `In 2-3 sentences, explain your sliding-window KV strategy (window=${layout.slidingWindow}) and what it means for long conversations. No bullets.`;
  }
  return `In 2-3 sentences, surface the 2 most important on-device deployment facts about your KV cache (e.g., GQA savings, KB/token, long-context memory growth). No bullets.`;
}

export function formatKVPerToken(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB/tok`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB/tok`;
  return `${bytes} B/tok`;
}

export function pickPrimaryDevice(report: KVReportResponse): { closest: KVReportResponse['device_capacities'][number] | null } {
  // Default highlight = largest device that fits, or the smallest device that almost fits.
  const fitting = report.device_capacities.filter((d) => d.fits);
  if (fitting.length > 0) {
    fitting.sort((a, b) => a.ram_gb - b.ram_gb);
    return { closest: fitting[0] };
  }
  // None fit — pick the largest one
  const sorted = [...report.device_capacities].sort((a, b) => b.ram_gb - a.ram_gb);
  return { closest: sorted[0] ?? null };
}

/** Quick formatter for the device-fit dot in @N cards. */
export function deviceFitTone(card: ContextCard): 'emerald' | 'amber' | 'red' {
  if (card.fittingDeviceCount >= 3) return 'emerald';
  if (card.fittingDeviceCount >= 1) return 'amber';
  return 'red';
}

// re-export for callers
export { formatSize };
