// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * weightInsights — pure-frontend aggregations over `TensorMeta[]` from /weight-stats.
 *
 * Backend's per-tensor endpoint is cheap (just safetensors header), so we can
 * derive the "X-ray" insights without any new backend work:
 *  - classify each tensor by module class (attn/mlp/norm/embed/head/expert/...)
 *  - aggregate quant coverage + effective bits/param + size share per class
 *  - identify per-layer size profile + outlier layers
 *  - top-N memory hogs with cumulative %
 *
 * No mapping table for "the model itself" — only for tensor NAME prefixes,
 * which is a structural fact about safetensors layout, not a model behavior.
 */
import type { TensorMeta } from '@/api/types';

export type ModuleClass =
  | 'embed' | 'lm_head' | 'attn' | 'mlp' | 'norm'
  | 'expert' | 'router' | 'vision' | 'audio' | 'other';

const CLASS_ORDER: ModuleClass[] = [
  'embed', 'attn', 'mlp', 'norm', 'expert', 'router', 'vision', 'audio', 'lm_head', 'other',
];

/** Friendly human label per class. */
export const CLASS_LABEL: Record<ModuleClass, string> = {
  embed: 'Embeddings',
  lm_head: 'LM Head',
  attn: 'Attention',
  mlp: 'Feed-Forward',
  norm: 'Norms',
  expert: 'MoE Experts',
  router: 'MoE Router',
  vision: 'Vision Tower',
  audio: 'Audio Tower',
  other: 'Other',
};

/** Plotly-friendly hex color per class — distinct in light + dark. */
export const CLASS_COLOR: Record<ModuleClass, string> = {
  embed:   '#0ea5e9',  // sky
  lm_head: '#14b8a6',  // teal
  attn:    '#6366f1',  // indigo
  mlp:     '#f97316',  // orange
  norm:    '#84cc16',  // lime
  expert:  '#a855f7',  // purple
  router:  '#ec4899',  // pink
  vision:  '#8b5cf6',  // violet
  audio:   '#06b6d4',  // cyan
  other:   '#9ca3af',  // gray
};

/**
 * Pure structural classification by tensor name.
 * Order matters: more-specific patterns first (lm_head before generic head).
 */
export function classifyTensorByName(name: string): ModuleClass {
  const n = name.toLowerCase();
  if (n.includes('lm_head') || n === 'lm_head' || n.endsWith('.lm_head.weight')) return 'lm_head';
  if (n.includes('embed_tokens') || n.includes('.embed.') || /\bembed/.test(n)) return 'embed';
  if (n.includes('vision') || n.includes('visual') || n.includes('vit.') || n.includes('image_')) return 'vision';
  if (n.includes('audio') || n.includes('encoder.audio') || n.includes('whisper')) return 'audio';
  if (n.includes('experts.') || n.includes('.experts')) return 'expert';
  if (n.includes('router') || n.includes('gate.weight') && n.includes('moe')) return 'router';
  if (n.includes('self_attn') || n.includes('.attention.') || /\battn/.test(n)) return 'attn';
  if (n.includes('mlp') || n.includes('feed_forward') || n.includes('ffn') || n.includes('gate_proj') || n.includes('up_proj') || n.includes('down_proj')) return 'mlp';
  if (n.includes('norm') || n.includes('layernorm') || n.includes('rmsnorm') || n.endsWith('.bias') && n.includes('norm')) return 'norm';
  return 'other';
}

/** Extract layer index from a tensor name like "model.layers.7.self_attn.q_proj.weight". */
export function extractLayerIndex(name: string): number | null {
  const m = name.match(/\.layers?\.(\d+)\./);
  return m ? parseInt(m[1], 10) : null;
}

export interface ClassAgg {
  cls: ModuleClass;
  label: string;
  color: string;
  count: number;
  params: number;
  sizeBytes: number;
  quantizedCount: number;
  /** weighted-avg bits/param (size_bytes*8 / num_elements). */
  bitsPerParam: number;
  /** % of total size that this class occupies. */
  shareOfTotal: number;
}

export interface TopHog {
  name: string;
  sizeBytes: number;
  share: number;     // 0-1
  cumShare: number;  // running cumulative share
  cls: ModuleClass;
  bitsPerParam: number;
  isQuantized: boolean;
}

export interface LayerProfile {
  layerIdx: number;
  totalBytes: number;
  attnBytes: number;
  mlpBytes: number;
  otherBytes: number;
  tensorCount: number;
}

export interface WeightAggregates {
  byClass: ClassAgg[];                   // ordered by CLASS_ORDER, only present classes
  topHogs: TopHog[];                     // top N by size desc
  layerProfile: LayerProfile[];          // sorted by layerIdx
  totalSize: number;
  /** sum of TensorMeta.num_elements — physical elements after quant packing. */
  totalStoredElements: number;
  totalTensors: number;
  quantizedCount: number;
  /** size_bytes × 8 / totalStoredElements — bits per packed element (~16 for fp16). */
  avgBitsPerStored: number;
  /** size_bytes × 8 / logicalParamCount — what the user actually wants to see.
   *  Falls back to avgBitsPerStored if logicalParamCount is 0/missing. */
  avgBitsPerLogical: number;
  logicalParams: number;                 // mirror of model.total_params for convenience
  /** Layers whose total size deviates >20% from the median (for outlier flag). */
  outlierLayers: { layerIdx: number; totalBytes: number; deviationPct: number }[];
  /** Whether quantization is "uniform" (all big tensors quantized) or "selective". */
  embedQuantized: boolean;
  lmHeadQuantized: boolean;
  normsQuantized: boolean;
}

function effectiveBits(t: TensorMeta): number {
  return t.num_elements > 0 ? (t.size_bytes * 8) / t.num_elements : 0;
}

export function aggregateWeights(
  tensors: TensorMeta[],
  topN = 10,
  logicalParams = 0,
): WeightAggregates {
  const byClassMap = new Map<ModuleClass, ClassAgg>();
  const layerMap = new Map<number, LayerProfile>();
  let totalSize = 0;
  let totalStoredElements = 0;
  let quantizedCount = 0;

  for (const t of tensors) {
    const cls = classifyTensorByName(t.name);
    totalSize += t.size_bytes;
    totalStoredElements += t.num_elements;
    if (t.is_quantized) quantizedCount++;

    let agg = byClassMap.get(cls);
    if (!agg) {
      agg = {
        cls,
        label: CLASS_LABEL[cls],
        color: CLASS_COLOR[cls],
        count: 0, params: 0, sizeBytes: 0, quantizedCount: 0,
        bitsPerParam: 0, shareOfTotal: 0,
      };
      byClassMap.set(cls, agg);
    }
    agg.count++;
    agg.params += t.num_elements;
    agg.sizeBytes += t.size_bytes;
    if (t.is_quantized) agg.quantizedCount++;

    const li = extractLayerIndex(t.name);
    if (li != null) {
      let lp = layerMap.get(li);
      if (!lp) {
        lp = { layerIdx: li, totalBytes: 0, attnBytes: 0, mlpBytes: 0, otherBytes: 0, tensorCount: 0 };
        layerMap.set(li, lp);
      }
      lp.totalBytes += t.size_bytes;
      lp.tensorCount++;
      if (cls === 'attn') lp.attnBytes += t.size_bytes;
      else if (cls === 'mlp') lp.mlpBytes += t.size_bytes;
      else lp.otherBytes += t.size_bytes;
    }
  }

  // Finalize per-class derived stats
  const byClass: ClassAgg[] = [];
  for (const cls of CLASS_ORDER) {
    const a = byClassMap.get(cls);
    if (a && a.count > 0) {
      a.bitsPerParam = a.params > 0 ? (a.sizeBytes * 8) / a.params : 0;
      a.shareOfTotal = totalSize > 0 ? a.sizeBytes / totalSize : 0;
      byClass.push(a);
    }
  }

  // Top hogs
  const sorted = [...tensors].sort((a, b) => b.size_bytes - a.size_bytes).slice(0, topN);
  const topHogs: TopHog[] = [];
  let running = 0;
  for (const t of sorted) {
    const share = totalSize > 0 ? t.size_bytes / totalSize : 0;
    running += share;
    topHogs.push({
      name: t.name,
      sizeBytes: t.size_bytes,
      share,
      cumShare: running,
      cls: classifyTensorByName(t.name),
      bitsPerParam: effectiveBits(t),
      isQuantized: t.is_quantized,
    });
  }

  const layerProfile = Array.from(layerMap.values()).sort((a, b) => a.layerIdx - b.layerIdx);

  // Outlier layers (>20% deviation from median total size)
  const outlierLayers: { layerIdx: number; totalBytes: number; deviationPct: number }[] = [];
  if (layerProfile.length >= 4) {
    const sizes = layerProfile.map((l) => l.totalBytes).sort((a, b) => a - b);
    const median = sizes[Math.floor(sizes.length / 2)];
    if (median > 0) {
      for (const lp of layerProfile) {
        const dev = (lp.totalBytes - median) / median;
        if (Math.abs(dev) > 0.20) {
          outlierLayers.push({ layerIdx: lp.layerIdx, totalBytes: lp.totalBytes, deviationPct: dev * 100 });
        }
      }
    }
  }

  // Per-class quant flags (for Insight panel)
  const embedAgg = byClassMap.get('embed');
  const headAgg = byClassMap.get('lm_head');
  const normAgg = byClassMap.get('norm');
  const embedQuantized = !!embedAgg && embedAgg.quantizedCount === embedAgg.count && embedAgg.count > 0;
  const lmHeadQuantized = !!headAgg && headAgg.quantizedCount === headAgg.count && headAgg.count > 0;
  const normsQuantized = !!normAgg && normAgg.quantizedCount > 0;

  const avgBitsPerStored = totalStoredElements > 0 ? (totalSize * 8) / totalStoredElements : 0;
  const avgBitsPerLogical = logicalParams > 0 ? (totalSize * 8) / logicalParams : avgBitsPerStored;

  return {
    byClass, topHogs, layerProfile,
    totalSize, totalStoredElements, totalTensors: tensors.length, quantizedCount,
    avgBitsPerStored, avgBitsPerLogical, logicalParams,
    outlierLayers,
    embedQuantized, lmHeadQuantized, normsQuantized,
  };
}

/** Compact diagnostic appendix for an LLM "explain my weights" system prompt. */
export function buildWeightContextSnippet(agg: WeightAggregates): string {
  const lines: string[] = [
    `## YOUR WEIGHT DISTRIBUTION (Edge Studio just analyzed it)`,
    `- ${agg.totalTensors} tensors, ${agg.quantizedCount} quantized (${(agg.quantizedCount / Math.max(agg.totalTensors, 1) * 100).toFixed(0)}%)`,
    `- Average ${agg.avgBitsPerLogical.toFixed(2)} bits/logical-param across all weights`,
    `  (≈ ${agg.avgBitsPerStored.toFixed(1)} bits per packed/stored element)`,
    ``,
    `### By module class:`,
  ];
  for (const c of agg.byClass) {
    const sizeMB = (c.sizeBytes / 1e6).toFixed(1);
    const sharePct = (c.shareOfTotal * 100).toFixed(1);
    const quantPct = c.count > 0 ? (c.quantizedCount / c.count * 100).toFixed(0) : '0';
    lines.push(
      `- **${c.label}**: ${sizeMB} MB (${sharePct}%), ${c.count} tensors, ${quantPct}% quantized, ~${c.bitsPerParam.toFixed(1)} bits/param`,
    );
  }
  lines.push(``, `### Top 3 memory hogs:`);
  for (const h of agg.topHogs.slice(0, 3)) {
    lines.push(`- \`${h.name}\` ${(h.sizeBytes / 1e6).toFixed(1)} MB (${(h.share * 100).toFixed(1)}%)`);
  }
  if (agg.outlierLayers.length > 0) {
    lines.push(``, `### Layer-size anomalies:`);
    for (const o of agg.outlierLayers.slice(0, 5)) {
      lines.push(`- Layer ${o.layerIdx}: ${o.deviationPct > 0 ? '+' : ''}${o.deviationPct.toFixed(0)}% vs median`);
    }
  }
  if (!agg.embedQuantized) lines.push(``, `> Note: token embeddings remain at full precision — common pattern, but they are usually the single biggest tensor.`);
  if (!agg.normsQuantized) lines.push(`> Note: normalization layers are at full precision (standard practice — RMSNorm precision matters more than its tiny size).`);
  return lines.join('\n');
}

/** Per-page suggested questions tailored to weight-analysis context. */
export function getWeightSuggestedPrompts(agg: WeightAggregates | null, locale: 'en' | 'zh'): { label: string; prompt: string }[] {
  if (!agg) return [];
  const topHog = agg.topHogs[0];
  const biggestClass = agg.byClass.slice().sort((a, b) => b.sizeBytes - a.sizeBytes)[0];
  if (locale === 'zh') {
    return [
      { label: '🪞 自我画像', prompt: `用 3-4 句话描述你自己的权重分布特征：每个 module class 大概占多少，平均 bits/param 是多少，最适合做哪种部署。` },
      { label: '🐘 最大消耗', prompt: topHog ? `你最大的单 tensor 是 \`${topHog.name}\`，占整体 ${(topHog.share * 100).toFixed(1)}%。解释它是什么、为什么这么大、能不能进一步压缩。` : '解释你最大的几个 tensor 分别承担什么职责。' },
      { label: '⚖️ 量化覆盖', prompt: agg.embedQuantized ? '你的 embedding 也被量化了，解释为什么这种激进量化在你这能 work、有什么风险。' : `你的 embedding 没被量化（保持 full precision），但 ${biggestClass?.label || 'Feed-Forward'} 走 ${biggestClass?.bitsPerParam.toFixed(1) || '4'} bits/param。解释这种"分层量化策略"的取舍。` },
      { label: '📐 层均匀性', prompt: agg.outlierLayers.length > 0
          ? `检测到 ${agg.outlierLayers.length} 个 layer 的总大小偏离中位数 >20% (例如 layer ${agg.outlierLayers[0].layerIdx} 偏 ${agg.outlierLayers[0].deviationPct.toFixed(0)}%)。解释为什么会有这种异常。`
          : '你所有 transformer layer 的大小都很接近，解释这种"结构均匀性"对剪枝/量化策略的意义。' },
    ];
  }
  return [
    { label: '🪞 Self portrait', prompt: 'In 3-4 sentences, describe your weight distribution: roughly how each module class is sized, your average bits/param, and what deployment style fits you best.' },
    { label: '🐘 Biggest hog', prompt: topHog
        ? `Your single biggest tensor is \`${topHog.name}\` taking ${(topHog.share * 100).toFixed(1)}% of all weights. Explain what it is, why it is so large, and whether it can be compressed further.`
        : 'Explain what your top few largest tensors do.' },
    { label: '⚖️ Quant coverage', prompt: agg.embedQuantized
        ? 'Your embeddings are quantized too — explain when this aggressive quant is safe and what the risks are.'
        : `Your embeddings stay full precision but ${biggestClass?.label || 'Feed-Forward'} runs at ${biggestClass?.bitsPerParam.toFixed(1) || '4'} bits/param. Explain the trade-off of this "selective quantization" strategy.` },
    { label: '📐 Layer uniformity', prompt: agg.outlierLayers.length > 0
        ? `${agg.outlierLayers.length} layers deviate >20% from the median size (e.g. layer ${agg.outlierLayers[0].layerIdx} is ${agg.outlierLayers[0].deviationPct > 0 ? '+' : ''}${agg.outlierLayers[0].deviationPct.toFixed(0)}%). Explain why this anomaly exists.`
        : 'All your transformer layers are nearly identical in size. Explain what this structural uniformity means for pruning and quantization strategies.' },
  ];
}

/** Auto-fired short brief once weights are aggregated. */
export function buildWeightAutoBrief(agg: WeightAggregates, locale: 'en' | 'zh'): string {
  const top = agg.byClass.slice().sort((a, b) => b.sizeBytes - a.sizeBytes).slice(0, 2);
  if (locale === 'zh') {
    return `用 2-3 句话总结你的权重分布最值得注意的一两个特征 (比如哪个 module class 最大、量化覆盖率、是否有不寻常的层)。结尾邀请用户点 Suggested 问题或选 tensor 看 detail。不要列项。`;
  }
  void top;
  return `In 2-3 natural sentences, surface the 1-2 most interesting things about your weight distribution (e.g. which module class dominates, quant coverage, any unusual layers). End by inviting the user to click a suggested question or select a tensor for details. No bullets.`;
}

/** Per-tensor explain prompt (after user clicks a tensor in the table). */
export function buildTensorExplainPrompt(tensorName: string, agg: WeightAggregates | null, locale: 'en' | 'zh'): string {
  const cls = classifyTensorByName(tensorName);
  const label = CLASS_LABEL[cls];
  const layer = extractLayerIndex(tensorName);
  const layerStr = layer != null ? ` (layer ${layer})` : '';
  let sharePct = '?';
  if (agg) {
    const found = agg.topHogs.find((h) => h.name === tensorName);
    if (found) sharePct = (found.share * 100).toFixed(2);
  }
  if (locale === 'zh') {
    return `用户在 Weight Analysis 页面选中了 \`${tensorName}\`${layerStr}。这是一个 **${label}** 类的 tensor，约占整模型的 ${sharePct}% 权重。简明解释：(1) 它在 forward pass 中做什么，(2) 它的 shape 形状怎么读，(3) 量化它对模型质量的影响，(4) 是否有更激进的优化空间。引用 detail panel 里的 min/max/mean/std/sparsity 数字（如果用户已经加载）。`;
  }
  return `The user just selected \`${tensorName}\`${layerStr} in the Weight Analysis page. This is a **${label}** tensor, ~${sharePct}% of total model weight. Briefly explain: (1) what it does in the forward pass, (2) how to read its shape, (3) the impact of quantizing it on model quality, (4) whether more aggressive optimization is possible. Reference the min/max/mean/std/sparsity numbers from the detail panel if available.`;
}
