// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Architecture Browser — "model X-ray" for on-device LLM optimization.
 *
 * Top:    Model Overview — identity, attention config, compression, KV cache
 * Middle: Architecture tree with quantization color-coding + share bars
 * Right:  Detail panel with structured config + tensor table
 * Alt:    Treemap/3D visualization modes
 */

import { useState, useEffect, useMemo, useCallback, useRef, lazy, Suspense } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { getArchitecture, getPruningTraces, getWeightStats } from '@/api/endpoints';
import type { ArchNode, PruningTrace, TensorMeta } from '@/api/types';
import { EmptyState } from '@/components/common/EmptyState';
import { ChartToggle } from '@/components/charts/ChartToggle';
import { InsightPanel } from '@/components/common/InsightPanel';
import { SkeletonList } from '@/components/common/Skeleton';
import { useArchitectureInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import type { VisualClickPayload } from '@/components/charts/Treemap';
import { formatParamCount, formatSize, cn } from '@/lib/utils';
import {
  ChevronRight, ChevronDown, Loader2, PanelRightClose, PanelRightOpen,
  Hash, HardDrive, Layers, Cpu, Eye, Zap, Database, Box, Info,
  Lightbulb, Sparkles, Send, X, RotateCcw,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';

const Treemap = lazy(() => import('@/components/charts/Treemap').then(m => ({ default: m.Treemap })));
const ForceGraph3D = lazy(() => import('@/components/charts/ForceGraph3D').then(m => ({ default: m.ForceGraph3D })));

type ViewMode = 'tree' | 'treemap' | '3d';

const VIEW_OPTIONS = [
  { value: 'tree', label: 'Tree' },
  { value: 'treemap', label: 'Treemap' },
  { value: '3d', label: '3D' },
];

// ─── Model Overview Cards ───────────────────────────────────────────────────

interface OverviewConfig {
  modelType: string;
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  maxCtx: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  intermediateSize: number;
  ropeTheta: number;
  tiedEmbeddings: boolean;
  bitsPerParam: number;
  quantBits: number;
  groupSize: number;
  quantizedCount: number;
  totalTensorCount: number;
  totalSizeBytes: number;
  totalParams: number;
  totalStoredParams: number;
  hasMoe: boolean;
  hasVision: boolean;
  modelCategory: string;
}

function extractOverview(
  archRoot: ArchNode,
  model: { total_params: number; total_stored_params: number; total_size_bytes: number; quantization: { bits: number; group_size: number; quantized_count: number; total_count: number }; has_moe: boolean; has_vision: boolean; model_category: string; model_type: string; num_layers: number; num_attention_heads: number; num_kv_heads: number },
): OverviewConfig {
  const cfg = archRoot.config_params;
  return {
    modelType: (cfg.model_type as string) || model.model_type || 'unknown',
    hiddenSize: (cfg.hidden_size as number) || 0,
    numLayers: (cfg.num_hidden_layers as number) || model.num_layers || 0,
    vocabSize: (cfg.vocab_size as number) || 0,
    maxCtx: (cfg.max_position_embeddings as number) || 0,
    numHeads: (cfg.num_attention_heads as number) || model.num_attention_heads || 0,
    numKVHeads: (cfg.num_key_value_heads as number) || model.num_kv_heads || 0,
    headDim: (cfg.head_dim as number) || 0,
    intermediateSize: (cfg.intermediate_size as number) || 0,
    ropeTheta: (cfg.rope_theta as number) || 0,
    tiedEmbeddings: (cfg.tie_word_embeddings as boolean) ?? false,
    bitsPerParam: model.total_size_bytes * 8 / model.total_params,
    quantBits: model.quantization.bits,
    groupSize: model.quantization.group_size,
    quantizedCount: model.quantization.quantized_count,
    totalTensorCount: model.quantization.total_count,
    totalSizeBytes: model.total_size_bytes,
    totalParams: model.total_params,
    totalStoredParams: model.total_stored_params,
    hasMoe: model.has_moe,
    hasVision: model.has_vision,
    modelCategory: model.model_category || 'LLM',
  };
}

/** KV cache size in bytes for one token across all layers.
 *  Formula: 2(K+V) × num_kv_heads × head_dim × 2 (fp16 bytes) × num_layers */
function kvCachePerToken(o: OverviewConfig): number {
  return 2 * o.numKVHeads * o.headDim * 2 * o.numLayers;
}

function OverviewCards({ o, t }: { o: OverviewConfig; t: (k: string, p?: Record<string, string | number>) => string }) {
  const gqaRatio = o.numKVHeads > 0 ? o.numHeads / o.numKVHeads : 0;
  const compressionRatio = o.totalStoredParams > 0 ? o.totalParams / o.totalStoredParams : 1;
  const quantPct = o.totalTensorCount > 0 ? (o.quantizedCount / o.totalTensorCount * 100) : 0;
  const kvPerToken = kvCachePerToken(o);
  const kvAt4k = kvPerToken * 4096;
  const kvAt8k = kvPerToken * 8192;
  const kvAt16k = kvPerToken * 16384;

  const typeBadgeColor: Record<string, string> = {
    LLM: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    VLM: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
    ASR: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    TTS: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
  };

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4 mb-4">
      {/* Card 1: Identity */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="flex items-center gap-2 mb-3">
          <Box size={14} className="text-gray-400 dark:text-stone-500" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-stone-400">{t('arch.card.identity')}</h3>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <span className={cn('rounded px-1.5 py-0.5 text-[10px] font-bold uppercase', typeBadgeColor[o.modelCategory] || typeBadgeColor.LLM)}>
              {o.modelCategory}
            </span>
            <span className="text-sm font-semibold text-gray-900 dark:text-stone-100">{o.modelType}</span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">Hidden</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.hiddenSize.toLocaleString()}</span></div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">FFN</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.intermediateSize.toLocaleString()}</span></div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">Vocab</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.vocabSize.toLocaleString()}</span></div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">Max ctx</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.maxCtx > 0 ? `${(o.maxCtx / 1024).toFixed(0)}K` : '—'}</span></div>
          </div>
          {(o.tiedEmbeddings || o.hasMoe || o.hasVision) && (
            <div className="flex flex-wrap gap-1 pt-1">
              {o.tiedEmbeddings && <span className="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500 dark:bg-stone-800 dark:text-stone-400">tied embeddings</span>}
              {o.hasMoe && <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">MoE</span>}
              {o.hasVision && <span className="rounded bg-purple-100 px-1.5 py-0.5 text-[10px] text-purple-700 dark:bg-purple-900/30 dark:text-purple-400">Vision</span>}
            </div>
          )}
        </div>
      </div>

      {/* Card 2: Attention */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="flex items-center gap-2 mb-3">
          <Eye size={14} className="text-gray-400 dark:text-stone-500" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-stone-400">{t('arch.card.attention')}</h3>
        </div>
        <div className="space-y-2">
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-gray-900 dark:text-stone-100">{o.numHeads}</span>
            <span className="text-xs text-gray-400 dark:text-stone-500">heads</span>
            <span className="text-gray-300 dark:text-stone-600 mx-1">×</span>
            <span className="text-2xl font-bold text-gray-900 dark:text-stone-100">{o.headDim}</span>
            <span className="text-xs text-gray-400 dark:text-stone-500">dim</span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">KV heads</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.numKVHeads}</span></div>
            <div className="flex justify-between">
              <span className="text-gray-400 dark:text-stone-500">GQA</span>
              <span className={cn('font-medium', gqaRatio > 1 ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-700 dark:text-stone-300')}>
                {gqaRatio > 1 ? `${gqaRatio}:1` : 'MHA'}
              </span>
            </div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">Layers</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.numLayers}</span></div>
            {o.ropeTheta > 0 && (
              <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">RoPE θ</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.ropeTheta >= 1e6 ? `${(o.ropeTheta / 1e6).toFixed(0)}M` : o.ropeTheta.toLocaleString()}</span></div>
            )}
          </div>
          {gqaRatio > 1 && (
            <p className="text-[10px] text-emerald-600 dark:text-emerald-400">
              {t('arch.gqa.savings', { pct: Math.round((1 - 1 / gqaRatio) * 100) })}
            </p>
          )}
        </div>
      </div>

      {/* Card 3: Compression */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="flex items-center gap-2 mb-3">
          <Zap size={14} className="text-gray-400 dark:text-stone-500" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-stone-400">{t('arch.card.compression')}</h3>
        </div>
        <div className="space-y-2">
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-gray-900 dark:text-stone-100">{o.bitsPerParam.toFixed(1)}</span>
            <span className="text-xs text-gray-400 dark:text-stone-500">bits/param</span>
          </div>
          {/* Quantization coverage bar */}
          <div>
            <div className="flex justify-between text-[10px] mb-0.5">
              <span className="text-gray-400 dark:text-stone-500">{t('arch.quantCoverage')}</span>
              <span className="font-medium text-gray-600 dark:text-stone-300">{quantPct.toFixed(0)}% ({o.quantizedCount}/{o.totalTensorCount})</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-gray-100 dark:bg-stone-800">
              <div
                className={cn('h-1.5 rounded-full', quantPct > 90 ? 'bg-emerald-500' : quantPct > 50 ? 'bg-amber-500' : 'bg-red-400')}
                style={{ width: `${Math.min(quantPct, 100)}%` }}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">{t('arch.quantBits')}</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.quantBits > 0 ? `${o.quantBits}-bit` : 'none'}</span></div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">{t('arch.groupSize')}</span><span className="font-medium text-gray-700 dark:text-stone-300">{o.groupSize > 0 ? o.groupSize : '—'}</span></div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">{t('arch.ratio')}</span><span className="font-medium text-gray-700 dark:text-stone-300">{compressionRatio.toFixed(1)}×</span></div>
            <div className="flex justify-between"><span className="text-gray-400 dark:text-stone-500">{t('arch.diskSize')}</span><span className="font-medium text-gray-700 dark:text-stone-300">{formatSize(o.totalSizeBytes)}</span></div>
          </div>
        </div>
      </div>

      {/* Card 4: KV Cache / Runtime Memory */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="flex items-center gap-2 mb-3">
          <Database size={14} className="text-gray-400 dark:text-stone-500" />
          <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-stone-400">{t('arch.card.kvCache')}</h3>
        </div>
        <div className="space-y-2">
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold text-gray-900 dark:text-stone-100">{(kvPerToken / 1024).toFixed(1)}</span>
            <span className="text-xs text-gray-400 dark:text-stone-500">KB/token</span>
          </div>
          <div className="space-y-1">
            {([
              { ctx: '4K', bytes: kvAt4k, total: kvAt4k + o.totalSizeBytes },
              { ctx: '8K', bytes: kvAt8k, total: kvAt8k + o.totalSizeBytes },
              { ctx: '16K', bytes: kvAt16k, total: kvAt16k + o.totalSizeBytes },
            ]).map(({ ctx, bytes, total }) => (
              <div key={ctx} className="flex items-center justify-between text-xs">
                <span className="text-gray-400 dark:text-stone-500">@ {ctx}</span>
                <div className="flex items-center gap-2">
                  <span className="text-gray-500 dark:text-stone-400">KV {formatSize(bytes)}</span>
                  <span className="text-gray-300 dark:text-stone-600">→</span>
                  <span className={cn(
                    'font-medium',
                    total > 8e9 ? 'text-red-500' : total > 4e9 ? 'text-amber-600 dark:text-amber-400' : 'text-emerald-600 dark:text-emerald-400',
                  )}>
                    {formatSize(total)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-gray-400 dark:text-stone-500 flex items-center gap-1">
            <Info size={9} />
            {t('arch.kvNote')}
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Enhanced Tree Node ─────────────────────────────────────────────────────

function TreeNode({ node, depth = 0, selectedPrefix, onSelect, rootSize }: {
  node: ArchNode;
  depth?: number;
  selectedPrefix: string | null;
  onSelect: (node: ArchNode) => void;
  rootSize: number;
}) {
  const [expanded, setExpanded] = useState(depth < 2);
  const hasChildren = node.children.length > 0;
  const isSelected = selectedPrefix === node.weight_prefix;
  const shareOfTotal = rootSize > 0 ? (node.total_size_bytes / rootSize) * 100 : 0;
  const isQuantized = node.is_quantized;

  return (
    <div>
      <button
        onClick={() => {
          if (hasChildren) setExpanded(!expanded);
          onSelect(node);
        }}
        className={cn(
          'group flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-left text-sm transition-colors hover:bg-gray-50 dark:hover:bg-stone-800',
          isSelected && 'bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400',
        )}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
      >
        {hasChildren ? (
          expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />
        ) : (
          <span className="w-3.5" />
        )}
        {/* Quantization indicator dot */}
        <span className={cn(
          'h-1.5 w-1.5 rounded-full shrink-0',
          isQuantized ? 'bg-emerald-500' : 'bg-gray-300 dark:bg-stone-600',
        )} title={isQuantized ? 'Quantized' : 'Full precision'} />
        <span className="flex-1 truncate font-medium">{node.name}</span>
        {/* Share bar */}
        {depth > 0 && shareOfTotal >= 0.5 && (
          <div className="hidden sm:flex items-center gap-1 mr-1" title={`${shareOfTotal.toFixed(1)}% of model`}>
            <div className="h-1 w-12 rounded-full bg-gray-100 dark:bg-stone-800">
              <div
                className="h-1 rounded-full bg-indigo-400/60 dark:bg-indigo-500/40"
                style={{ width: `${Math.min(shareOfTotal, 100)}%` }}
              />
            </div>
            <span className="text-[10px] text-gray-300 dark:text-stone-600 w-7 text-right">{shareOfTotal.toFixed(0)}%</span>
          </div>
        )}
        <span className="text-xs text-gray-400 dark:text-stone-500 tabular-nums">{formatParamCount(node.total_param_count)}</span>
        <span className="ml-2 text-xs text-gray-400 dark:text-stone-500 tabular-nums w-16 text-right">{formatSize(node.total_size_bytes)}</span>
      </button>
      {expanded && hasChildren && (
        <div>
          {node.children.map((child, i) => (
            <TreeNode
              key={`${child.weight_prefix || child.name}-${i}`}
              node={child}
              depth={depth + 1}
              selectedPrefix={selectedPrefix}
              onSelect={onSelect}
              rootSize={rootSize}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function findNodeByPrefix(node: ArchNode, prefix: string): ArchNode | null {
  if (node.weight_prefix === prefix) return node;
  for (const child of node.children) {
    const found = findNodeByPrefix(child, prefix);
    if (found) return found;
  }
  return null;
}

// ─── AI Prompt Builder ─────────────────────────────────────────────────────

interface ModelInfoLite {
  model_name: string;
  total_size_bytes: number;
  num_layers: number;
}

function buildExplainPrompt(node: ArchNode, root: ArchNode, model: ModelInfoLite): string {
  const isRoot = node === root || node.weight_prefix === root.weight_prefix;
  if (isRoot) {
    return `Give me a complete architectural overview of yourself (${model.model_name}). ` +
      `Walk through the major components in order: token embedding → ${model.num_layers}× transformer layers (attention + FFN + norms) → final norm → LM head. ` +
      `For each, mention its size, role, and any notable design choices. ` +
      `End with the top 2 things a developer should know before deploying you on iPhone.`;
  }

  const sharePct = root.total_size_bytes > 0
    ? (node.total_size_bytes / root.total_size_bytes * 100).toFixed(2)
    : '?';

  // Detect what kind of component this is to ask the right questions
  const lower = node.name.toLowerCase();
  const isLayer = lower.startsWith('layer') || node.node_type === 'layer';
  const isAttn = lower.includes('attn') || lower.includes('attention');
  const isFFN = lower.includes('mlp') || lower.includes('feed_forward') || lower.includes('ffn');
  const isNorm = lower.includes('norm');
  const isEmbed = lower.includes('embed');
  const isHead = lower.includes('head') || lower === 'lm_head';

  let focus = '';
  if (isLayer) {
    focus = `Walk through what happens inside this layer when a token passes through: which sub-modules (self_attn, MLP, norms) ` +
      `are called in what order, what each computes, how much of this layer's ${(node.total_size_bytes / 1e6).toFixed(1)} MB goes to attention vs FFN, ` +
      `and whether this layer is identical to the others (so any insight applies to all ${model.num_layers} layers).`;
  } else if (isAttn) {
    focus = `Explain what self-attention does in this layer, how my GQA configuration shapes the Q/K/V/O projection sizes ` +
      `(K and V should be smaller than Q and O if GQA is active — check the actual numbers above), and what fraction of inference time/memory this consumes.`;
  } else if (isFFN) {
    focus = `Explain the Gate/Up/Down projection structure (SwiGLU-style), why FFN is typically the largest component per layer ` +
      `(${(node.total_size_bytes / 1e6).toFixed(1)} MB here = ${sharePct}% of total model), and what optimization options exist (pruning, lower-bit quantization).`;
  } else if (isNorm) {
    focus = `Explain what RMSNorm/LayerNorm does here, why it's negligibly small compared to projections, ` +
      `and why it's typically NOT quantized (precision matters more than size for normalization).`;
  } else if (isEmbed) {
    focus = `Explain why token embeddings take so much space (${(node.total_size_bytes / 1e6).toFixed(1)} MB = ${sharePct}% of model), ` +
      `the vocab × hidden_dim math, and whether tied embeddings could save weight here.`;
  } else if (isHead) {
    focus = `Explain the LM head's role (project hidden state → vocabulary logits), whether it's tied to embeddings (size 0 if so), ` +
      `and the precision tradeoff at the output layer.`;
  } else {
    focus = `Explain what this component does inside the model, why it's sized the way it is, and what optimization opportunities exist.`;
  }

  return `The user just selected **${node.name}** in the architecture tree (${formatSize(node.total_size_bytes)}, ${sharePct}% of total weight). ` +
    `${focus} ` +
    `Be specific — cite the actual sizes/shapes from the tensor breakdown above. End with one concrete optimization tip relevant to on-device deployment.`;
}

/**
 * Prompt for clicks in Treemap/3D visual modes (aggregated synthetic nodes).
 * Reads from VisualClickPayload which includes label, scope, module/sub names.
 */
function buildVisualExplainPrompt(
  vs: { label: string; sizeBytes: number; params: number; kind: string; numLayers?: number; module?: string; sub?: string },
  model: ModelInfoLite,
): string {
  const sharePct = model.total_size_bytes > 0
    ? (vs.sizeBytes / model.total_size_bytes * 100).toFixed(2)
    : '?';
  const perLayer = vs.numLayers ? vs.sizeBytes / vs.numLayers : vs.sizeBytes;

  if (vs.kind === 'aggregated') {
    const isSubcomp = !!vs.sub;
    if (isSubcomp) {
      // e.g., "Q Projection" across 36 layers
      return `The user clicked **${vs.label}** in the Treemap. This is the AGGREGATE of all ${vs.numLayers} layers' \`${vs.module}.${vs.sub}\` weights. ` +
        `Total size: ${formatSize(vs.sizeBytes)} (${sharePct}% of model), or ${formatSize(perLayer)} per layer × ${vs.numLayers} layers. ` +
        `Explain: (1) what \`${vs.sub}\` does inside the ${vs.module} module, (2) WHY this projection is the size it is (reference hidden_size and KV head count where relevant — for K/V projections under GQA they are smaller than Q/O), ` +
        `(3) the tradeoff if we quantize/prune it further. ` +
        `Be specific with numbers.`;
    }
    // module-level aggregation (e.g., "Attention" or "Feed-Forward" across 36 layers)
    return `The user clicked **${vs.label}** in the Treemap — the AGGREGATE of all ${vs.numLayers} layers' \`${vs.module}\` modules. ` +
      `Total: ${formatSize(vs.sizeBytes)} (${sharePct}% of model) = ${formatSize(perLayer)} per layer × ${vs.numLayers} layers. ` +
      `Explain: (1) what this module does in each layer, (2) what fraction of total inference compute it accounts for, (3) which sub-components inside it are the largest, ` +
      `(4) the most impactful optimization for this part of the model. ` +
      `Use the actual numbers.`;
  }

  // Default visualSelection (whole model, before any user click)
  if (vs.label === model.model_name) {
    return `The user just opened the visualization view of you (${model.model_name}) but hasn't clicked any specific component yet. ` +
      `Give a useful overview: ` +
      `(1) what the user is looking at in this Treemap/3D view (largest blocks = ${'biggest weight consumers'}), ` +
      `(2) which 1-2 components dominate your weight budget (be specific with %), ` +
      `(3) what the user should click on if they want to understand the biggest optimization opportunities. ` +
      `Encourage them to click on a Treemap region or 3D surface point to drill in.`;
  }
  // Single component (3D click on a specific Layer/Component cell)
  return `The user clicked **${vs.label}** in the 3D Memory Landscape. ` +
    `Size: ${formatSize(vs.sizeBytes)} (${sharePct}% of model). ` +
    `Explain what this specific component does, why it has this size, and how it compares to the same component in other layers (uniform vs varying). ` +
    `Cite numbers.`;
}

// ─── Enhanced Detail Panel ──────────────────────────────────────────────────

function DetailPanel({ node, tensors, rootSize, t, onClose }: {
  node: ArchNode | null;
  tensors: TensorMeta[];
  rootSize: number;
  t: (k: string, p?: Record<string, string | number>) => string;
  onClose: () => void;
}) {
  if (!node) return null;

  const shareOfTotal = rootSize > 0 ? (node.total_size_bytes / rootSize * 100) : 0;
  const compressionRatio = node.total_stored_param_count > 0 ? node.total_param_count / node.total_stored_param_count : 1;
  const cfgKeys = Object.keys(node.config_params);
  const pruningKeys = Object.keys(node.pruning_info);
  const extraKeys = Object.keys(node.extra);

  return (
    <div className="flex h-full flex-col border-l border-gray-200 bg-white dark:border-stone-700 dark:bg-stone-900">
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3 dark:border-stone-800">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-gray-900 truncate dark:text-stone-100">{node.name}</h3>
          <p className="text-[10px] text-gray-400 dark:text-stone-500">{node.node_type}</p>
        </div>
        <button
          onClick={onClose}
          className="shrink-0 rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
        >
          <PanelRightClose size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Metrics grid */}
        <div className="grid grid-cols-2 gap-2.5">
          {([
            { label: t('arch.detail.params'), value: formatParamCount(node.total_param_count) },
            { label: t('arch.detail.stored'), value: formatParamCount(node.total_stored_param_count) },
            { label: t('arch.detail.size'), value: formatSize(node.total_size_bytes) },
            { label: t('arch.detail.share'), value: `${shareOfTotal.toFixed(1)}%` },
            { label: t('arch.detail.quantized'), value: node.is_quantized ? `Yes (${compressionRatio.toFixed(1)}×)` : 'No' },
            { label: t('arch.detail.children'), value: node.children.length > 0 ? `${node.children.length}` : '—' },
          ]).map(({ label, value }) => (
            <div key={label} className="rounded-lg bg-gray-50 px-2.5 py-1.5 dark:bg-stone-800">
              <p className="text-[10px] text-gray-400 dark:text-stone-500">{label}</p>
              <p className="text-xs font-medium text-gray-800 dark:text-stone-200">{value}</p>
            </div>
          ))}
        </div>

        {/* Config — structured grid instead of JSON dump */}
        {cfgKeys.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs font-medium text-gray-600 dark:text-stone-300">{t('arch.detail.config')}</p>
            <div className="rounded-lg bg-gray-50 dark:bg-stone-800 divide-y divide-gray-100 dark:divide-stone-700">
              {cfgKeys.map((k) => (
                <div key={k} className="flex justify-between px-2.5 py-1 text-xs">
                  <span className="text-gray-400 dark:text-stone-500 font-mono">{k}</span>
                  <span className="font-medium text-gray-700 dark:text-stone-300 ml-2 text-right truncate max-w-[50%]">
                    {String(node.config_params[k])}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pruning info */}
        {pruningKeys.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs font-medium text-amber-600 dark:text-amber-400">{t('arch.detail.pruning')}</p>
            <div className="rounded-lg bg-amber-50 dark:bg-amber-900/10 divide-y divide-amber-100 dark:divide-amber-900/20">
              {pruningKeys.map((k) => (
                <div key={k} className="flex justify-between px-2.5 py-1 text-xs">
                  <span className="text-amber-500 dark:text-amber-400/60 font-mono">{k}</span>
                  <span className="font-medium text-amber-700 dark:text-amber-300 ml-2 text-right truncate max-w-[50%]">
                    {String(node.pruning_info[k])}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Extra metadata */}
        {extraKeys.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs font-medium text-gray-600 dark:text-stone-300">{t('arch.detail.extra')}</p>
            <div className="max-h-24 overflow-y-auto rounded-lg bg-gray-50 p-2.5 dark:bg-stone-800">
              <pre className="text-[10px] text-gray-500 dark:text-stone-400">{JSON.stringify(node.extra, null, 2)}</pre>
            </div>
          </div>
        )}

        {/* Tensor table — enhanced with dtype + quantized */}
        {tensors.length > 0 && (
          <div>
            <p className="mb-1.5 text-xs font-medium text-gray-600 dark:text-stone-300">
              {t('arch.detail.tensors', { count: tensors.length })}
            </p>
            <div className="max-h-72 overflow-y-auto rounded-lg border border-gray-100 dark:border-stone-700">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-gray-50 dark:bg-stone-800">
                  <tr className="text-left text-gray-500 dark:text-stone-400">
                    <th className="px-2 py-1.5 font-medium">Name</th>
                    <th className="px-2 py-1.5 font-medium">Shape</th>
                    <th className="px-2 py-1.5 font-medium">Type</th>
                    <th className="px-2 py-1.5 font-medium text-right">Size</th>
                  </tr>
                </thead>
                <tbody>
                  {tensors.map((tensor) => (
                    <tr key={tensor.name} className="border-t border-gray-50 hover:bg-gray-50 dark:border-stone-800 dark:hover:bg-stone-800">
                      <td className="max-w-[120px] truncate px-2 py-1 font-mono text-gray-800 dark:text-stone-300" title={tensor.name}>
                        {tensor.name.replace(node.weight_prefix + '.', '')}
                      </td>
                      <td className="px-2 py-1 text-gray-500 dark:text-stone-400 tabular-nums whitespace-nowrap">
                        [{tensor.shape.join('×')}]
                      </td>
                      <td className="px-2 py-1 whitespace-nowrap">
                        <span className={cn(
                          'rounded px-1 py-0.5 text-[9px] font-medium',
                          tensor.is_quantized
                            ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                            : 'bg-gray-100 text-gray-500 dark:bg-stone-800 dark:text-stone-400',
                        )}>
                          {tensor.dtype}
                        </span>
                      </td>
                      <td className="px-2 py-1 text-right text-gray-500 dark:text-stone-400 tabular-nums">
                        {formatSize(tensor.size_bytes)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

export default function ArchitectureBrowser() {
  const model = useModelStore((s) => s.currentModel);
  const hasProfile = !!useModelStore((s) => s.profileSummary);
  const [archTree, setArchTree] = useState<ArchNode | null>(null);
  const [pruningTraces, setPruningTraces] = useState<PruningTrace[]>([]);
  const [allTensors, setAllTensors] = useState<TensorMeta[]>([]);
  const [selectedNode, setSelectedNode] = useState<ArchNode | null>(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('tree');
  const [detailOpen, setDetailOpen] = useState(true);
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const [visualSelection, setVisualSelection] = useState<VisualClickPayload | null>(null);
  const lastAskedPrefixRef = useRef<string | null>(null);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale);
  const archInsights = useArchitectureInsights(t, model, hasProfile);

  // ── System prompt: model overview + selected component context ──
  const archSystemPrompt = useMemo(() => {
    if (!model || !archTree) return '';
    const cfg = archTree.config_params;
    const numHeads = model.num_attention_heads;
    const numKV = model.num_kv_heads;
    const headDim = (cfg.head_dim as number) || 0;
    const gqaRatio = numKV > 0 ? Math.round(numHeads / numKV) : 1;
    const numLayers = model.num_layers;
    const hiddenSize = (cfg.hidden_size as number) || model.hidden_size;
    const ffnSize = (cfg.intermediate_size as number) || model.intermediate_size;
    const vocabSize = (cfg.vocab_size as number) || 0;
    const maxCtx = (cfg.max_position_embeddings as number) || 0;

    // Compute KV cache estimates (fp16 = 2 bytes/elem)
    const kvPerToken = 2 * numKV * headDim * 2 * numLayers;
    const kvAt4k = kvPerToken * 4096;
    const kvAt8k = kvPerToken * 8192;
    const compressionRatio = model.total_stored_params > 0
      ? (model.total_params / model.total_stored_params).toFixed(1)
      : '1.0';

    const lines: string[] = [
      `# CONTEXT — Edge Studio Architecture Browser`,
      ``,
      `You are speaking AS THIS MODEL — a ${model.model_type || 'transformer'} loaded in Edge Studio, an on-device LLM optimization workbench for Apple Silicon.`,
      `When the user asks about "you", they mean this model. Speak in first person where natural ("My architecture has...", "I use GQA to...").`,
      ``,
      `## YOUR IDENTITY`,
      `- Name: ${model.model_name}`,
      `- Family: ${model.model_type || 'unknown'}${model.has_vision ? ' (Vision-Language Model)' : ''}${model.has_moe ? ' (Mixture of Experts)' : ''}${model.supports_thinking ? ' (supports <think> mode)' : ''}`,
      `- Category: ${model.model_category || 'LLM'}`,
      `- Source format: ${model.source_format || 'safetensors'}`,
      ``,
      `## PARAMETER PROFILE`,
      `- Logical parameters: ${formatParamCount(model.total_params)} (${model.total_params.toLocaleString()})`,
      `- Stored elements: ${formatParamCount(model.total_stored_params)} (compression ${compressionRatio}× via ${model.quantization.bits > 0 ? model.quantization.bits + '-bit quantization' : 'no quantization'})`,
      `- Disk size: ${formatSize(model.total_size_bytes)} = ${(model.total_size_bytes * 8 / model.total_params).toFixed(2)} bits/param average`,
      `- Quantization coverage: ${model.quantization.quantized_count}/${model.quantization.total_count} tensors quantized${model.quantization.group_size > 0 ? `, group size ${model.quantization.group_size}` : ''}`,
      ``,
      `## ARCHITECTURE DIMENSIONS`,
      `- ${numLayers} transformer layers`,
      `- Hidden size: ${hiddenSize} (residual stream width)`,
      `- FFN intermediate size: ${ffnSize} (${(ffnSize / hiddenSize).toFixed(1)}× hidden)`,
      `- Vocabulary: ${vocabSize.toLocaleString()} tokens`,
      `- Max context: ${maxCtx.toLocaleString()} tokens`,
      cfg.tie_word_embeddings ? `- Tied embeddings: YES (lm_head shares weights with embed_tokens, saves ~${formatSize((vocabSize * hiddenSize * 2))})` : `- Tied embeddings: NO (separate lm_head)`,
      ``,
      `## ATTENTION CONFIG`,
      `- ${numHeads} query heads × ${headDim} dim/head`,
      `- ${numKV} KV heads (${gqaRatio > 1 ? `GQA ${gqaRatio}:1 — saves ${Math.round((1 - 1 / gqaRatio) * 100)}% KV cache vs MHA` : 'MHA — no GQA'})`,
      cfg.rope_theta ? `- RoPE θ: ${cfg.rope_theta}${(cfg.rope_theta as number) >= 1e6 ? ' (long-context optimized)' : ''}` : '',
      cfg.sliding_window ? `- Sliding window: ${cfg.sliding_window}` : '',
      `- KV cache per token: ${(kvPerToken / 1024).toFixed(1)} KB across all layers (fp16)`,
      ``,
      `## RUNTIME MEMORY ESTIMATES`,
      `- Weights only: ${formatSize(model.total_size_bytes)}`,
      `- @ 4K context: weights + ${formatSize(kvAt4k)} KV = ${formatSize(model.total_size_bytes + kvAt4k)}`,
      `- @ 8K context: weights + ${formatSize(kvAt8k)} KV = ${formatSize(model.total_size_bytes + kvAt8k)}`,
      `- Real peak ~1.2-1.5× this (activations, scratch buffers)`,
      ``,
      `## DEVICE FIT REFERENCE (iOS 26 increased memory limit ~85% RAM)`,
      `- iPhone 15 Pro / 16 Pro (8GB): usable ~6.8 GB`,
      `- iPad M2/M3/M4 (8-16GB): usable ~6.8-13.6 GB`,
      `- M1 Max MacBook (32GB), M2 Ultra Mac Studio (192GB)`,
    ];

    // Prefer visualSelection (richer aggregated info) over plain selectedNode
    if (visualSelection && visualSelection.kind === 'aggregated') {
      const vs = visualSelection;
      const share = archTree.total_size_bytes > 0
        ? (vs.sizeBytes / archTree.total_size_bytes * 100).toFixed(2)
        : '?';
      const bpp = vs.params > 0 ? (vs.sizeBytes * 8 / vs.params).toFixed(1) : '—';
      const perLayer = vs.numLayers ? vs.sizeBytes / vs.numLayers : vs.sizeBytes;

      lines.push(
        ``,
        `## SELECTED COMPONENT (user clicked in Treemap/3D — AGGREGATED across layers)`,
        `- Display name: **${vs.label}**`,
        vs.module ? `- Module: \`${vs.module}\`${vs.sub ? ` / sub: \`${vs.sub}\`` : ''}` : '',
        `- Aggregation scope: SUMMED across all ${vs.numLayers} transformer layers`,
        `- Total size: ${formatSize(vs.sizeBytes)} (${share}% of model)`,
        `- Per-layer size: ${formatSize(perLayer)} × ${vs.numLayers} layers`,
        `- Parameters: ${formatParamCount(vs.params)} total`,
        `- Avg precision: ${bpp} bits/param`,
        ``,
        `IMPORTANT: When discussing "${vs.label}", explain it as the AGGREGATE of all ${vs.numLayers} layers' ${vs.sub || vs.module || 'components'}. ` +
        `Each individual ${vs.sub || vs.module || 'instance'} is ${formatSize(perLayer)} per layer.`,
      );
    } else if (selectedNode && selectedNode !== archTree) {
      const sn = selectedNode;
      const share = archTree.total_size_bytes > 0
        ? (sn.total_size_bytes / archTree.total_size_bytes * 100).toFixed(2)
        : '?';
      const ownBpp = sn.total_param_count > 0
        ? (sn.total_size_bytes * 8 / sn.total_param_count).toFixed(1)
        : '—';
      const componentTensorList = allTensors.filter(t =>
        t.name.startsWith(sn.weight_prefix + '.') || t.name === sn.weight_prefix
      );
      const tensorBreakdown = componentTensorList.slice(0, 12).map(t =>
        `  - ${t.name.replace(sn.weight_prefix + '.', '')}: shape=[${t.shape.join('×')}] ${t.dtype} ${formatSize(t.size_bytes)}${t.is_quantized ? ' (quantized)' : ''}`
      ).join('\n');

      lines.push(
        ``,
        `## SELECTED COMPONENT (user is currently inspecting)`,
        `- Path: \`${sn.weight_prefix}\``,
        `- Name: ${sn.name}`,
        `- Node type: ${sn.node_type}`,
        `- Size: ${formatSize(sn.total_size_bytes)} (${share}% of total model)`,
        `- Parameters: ${formatParamCount(sn.total_param_count)} logical, ${formatParamCount(sn.total_stored_param_count)} stored`,
        `- Avg precision: ${ownBpp} bits/param`,
        `- Quantized: ${sn.is_quantized ? 'Yes' : 'No'}`,
        sn.children.length > 0
          ? `- Direct sub-components: ${sn.children.map(c => `${c.name}(${formatSize(c.total_size_bytes)})`).join(', ')}`
          : '',
        Object.keys(sn.config_params).length > 0
          ? `- Config: ${Object.entries(sn.config_params).map(([k, v]) => `${k}=${v}`).join(', ')}`
          : '',
        componentTensorList.length > 0
          ? `- Tensor breakdown (${componentTensorList.length} tensors):\n${tensorBreakdown}${componentTensorList.length > 12 ? `\n  ...and ${componentTensorList.length - 12} more` : ''}`
          : '',
      );
    }

    const langInstruction = locale === 'zh'
      ? `- **必须用简体中文回复**, 即使用户用英文提问也用中文回答 (UI 当前是中文)。技术术语 (q_proj/GQA/RMSNorm 等) 保留英文.`
      : `- **MUST reply in English**, even if the user asks in another language (UI is currently English).`;

    lines.push(
      ``,
      `## RESPONSE STYLE`,
      langInstruction,
      `- Be specific and QUANTITATIVE — cite actual numbers from the data above (sizes, params, GB, %).`,
      `- Focus on actionable insights for ON-DEVICE DEPLOYMENT (memory budget, latency, device fit).`,
      `- When discussing the selected component, refer to it by name and use the actual data provided.`,
      `- Keep answers concise (3-6 sentences unless deep dive requested).`,
      `- Use markdown for emphasis (**bold** for key numbers, bullet lists for breakdowns).`,
    );

    return lines.filter(Boolean).join('\n');
  }, [model, archTree, selectedNode, visualSelection, allTensors, locale]);

  // Single chat instance — reused for all interactions
  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: archSystemPrompt,
    maxTokens: 1200,
    temperature: 0.65,
  });

  useEffect(() => {
    if (!model) return;
    chat.reset();
    setAskOpen(false);
    setVisualSelection(null);
    lastAskedPrefixRef.current = null;
    setLoading(true);
    Promise.all([
      getArchitecture(model.model_id),
      getPruningTraces(model.model_id),
      getWeightStats(model.model_id),
    ]).then(([arch, traces, weightData]) => {
      setArchTree(arch);
      setPruningTraces(traces);
      setAllTensors(weightData.tensors);
      setSelectedNode(arch);
      // If we're already in visual mode (e.g., user reloaded), seed default visualSelection
      if (viewMode !== 'tree') {
        setVisualSelection({
          prefix: arch.weight_prefix,
          label: model.model_name,
          sizeBytes: model.total_size_bytes,
          params: model.total_params,
          kind: 'single',
        });
      }
    }).finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  const overview = useMemo(() => {
    if (!archTree || !model) return null;
    return extractOverview(archTree, model);
  }, [archTree, model]);

  const componentTensors = useMemo(() => {
    if (!selectedNode || !allTensors.length) return [];
    const prefix = selectedNode.weight_prefix;
    if (!prefix) return [];
    return allTensors.filter((t) => t.name.startsWith(prefix + '.') || t.name === prefix);
  }, [selectedNode, allTensors]);

  const handleVisualizationNodeClick = useCallback((payload: VisualClickPayload) => {
    if (!archTree) return;
    setVisualSelection(payload);
    const node = findNodeByPrefix(archTree, payload.prefix);
    if (node) {
      setSelectedNode(node);
      setDetailOpen(true);
    }
  }, [archTree]);

  // ── Data-driven insights for visual modes ──
  const visualInsights = useMemo(() => {
    if (!archTree || allTensors.length === 0) return [];
    const lg = archTree.children.find(c => c.node_type === 'layer_group');
    if (!lg || lg.children.length === 0) return [];

    const nL = lg.children.length;
    const totalBytes = archTree.total_size_bytes;

    const layer0 = lg.children[0];
    const modSizes = new Map<string, number>();
    for (const t of allTensors) {
      if (!t.name.startsWith(layer0.weight_prefix + '.')) continue;
      const rel = t.name.slice(layer0.weight_prefix.length + 1);
      const mod = rel.split('.')[0];
      modSizes.set(mod, (modSizes.get(mod) || 0) + t.size_bytes);
    }

    const insights: string[] = [];
    const sortedMods = Array.from(modSizes.entries()).sort((a, b) => b[1] - a[1]);
    const perLayerTotal = sortedMods.reduce((s, [, v]) => s + v, 0);

    if (sortedMods.length > 0) {
      const [topMod, topBytes] = sortedMods[0];
      const topPct = ((topBytes * nL) / totalBytes * 100).toFixed(0);
      const friendlyMod = topMod === 'mlp' ? 'Feed-Forward (MLP)' : topMod === 'self_attn' ? 'Attention' : topMod;
      insights.push(`${friendlyMod} uses ${topPct}% of total weight (${formatSize(topBytes * nL)}) — the biggest optimization target`);
    }

    const getSubSize = (mod: string, sub: string) => {
      let total = 0;
      const pfx = `${layer0.weight_prefix}.${mod}.${sub}.`;
      for (const t of allTensors) { if (t.name.startsWith(pfx)) total += t.size_bytes; }
      return total;
    };
    const qSize = getSubSize('self_attn', 'q_proj');
    const kSize = getSubSize('self_attn', 'k_proj');
    if (qSize > 0 && kSize > 0 && qSize > kSize * 1.5) {
      const ratio = (qSize / kSize).toFixed(0);
      insights.push(`GQA active: K/V projections are ${ratio}× smaller than Q — saves ${formatSize((qSize - kSize) * 2 * nL)} across all layers`);
    }

    if (nL > 1) {
      const layer1Size = lg.children[1]?.total_size_bytes ?? 0;
      if (Math.abs(layer0.total_size_bytes - layer1Size) < 100) {
        insights.push(`All ${nL} layers are structurally identical (${formatSize(perLayerTotal)}/layer) — pruning any layer saves the same amount`);
      }
    }

    if (overview) {
      const kvPerToken = 2 * overview.numKVHeads * overview.headDim * 2 * overview.numLayers;
      const kvAt4k = kvPerToken * 4096;
      insights.push(`Runtime memory at 4K context: weights ${formatSize(totalBytes)} + KV cache ${formatSize(kvAt4k)} = ${formatSize(totalBytes + kvAt4k)}`);
    }

    return insights;
  }, [archTree, allTensors, overview]);

  // ── Early returns AFTER all hooks ──

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to browse its architecture" />;
  }

  if (loading || !archTree) {
    return <SkeletonList count={8} className="py-4" />;
  }

  const rootSize = archTree.total_size_bytes;
  const bitsPerParam = model.total_size_bytes * 8 / model.total_params;
  const isVisualMode = viewMode !== 'tree';

  return (
    <div className={cn(isVisualMode ? '-m-6' : '')}>
      {/* Compact header bar */}
      <div className={cn(
        'flex items-center justify-between gap-4',
        isVisualMode ? 'border-b border-gray-100 px-6 py-2.5 dark:border-stone-800' : 'mb-4',
      )}>
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-sm font-semibold text-gray-900 truncate dark:text-stone-100">
            {model.model_name}
          </h1>
          <div className="hidden sm:flex items-center gap-2">
            <span className="flex items-center gap-1.5 rounded-lg bg-gray-100 px-2.5 py-1 dark:bg-stone-800" title="Parameters">
              <Hash size={11} className="text-gray-400 dark:text-stone-500" />
              <span className="text-xs font-medium text-gray-700 dark:text-stone-300">{formatParamCount(model.total_params)}</span>
            </span>
            <span className="flex items-center gap-1.5 rounded-lg bg-gray-100 px-2.5 py-1 dark:bg-stone-800" title="Size">
              <HardDrive size={11} className="text-gray-400 dark:text-stone-500" />
              <span className="text-xs font-medium text-gray-700 dark:text-stone-300">{formatSize(model.total_size_bytes)}</span>
            </span>
            <span className="flex items-center gap-1.5 rounded-lg bg-gray-100 px-2.5 py-1 dark:bg-stone-800" title="Layers">
              <Layers size={11} className="text-gray-400 dark:text-stone-500" />
              <span className="text-xs font-medium text-gray-700 dark:text-stone-300">{model.num_layers}L</span>
            </span>
            <span className="flex items-center gap-1.5 rounded-lg bg-gray-100 px-2.5 py-1 dark:bg-stone-800" title="Bits/param">
              <Cpu size={11} className="text-gray-400 dark:text-stone-500" />
              <span className="text-xs font-medium text-gray-700 dark:text-stone-300">{bitsPerParam.toFixed(1)}b</span>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <ChartToggle
            mode={viewMode}
            options={VIEW_OPTIONS}
            onChange={(m) => {
              const newMode = m as ViewMode;
              setViewMode(newMode);
              chat.reset();
              lastAskedPrefixRef.current = null;
              // Set a default visualSelection for visual modes so FAB always has context.
              // Treemap/3D click handlers will overwrite this on user interaction.
              if (newMode !== 'tree' && model && archTree) {
                setVisualSelection({
                  prefix: archTree.weight_prefix,
                  label: model.model_name,
                  sizeBytes: model.total_size_bytes,
                  params: model.total_params,
                  kind: 'single',
                });
              } else {
                setVisualSelection(null);
              }
            }}
          />
          {isVisualMode && selectedNode && (
            <button
              onClick={() => setDetailOpen(!detailOpen)}
              className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              title={detailOpen ? 'Hide detail' : 'Show detail'}
            >
              {detailOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
            </button>
          )}
        </div>
      </div>

      {/* Model Overview Cards — tree mode only */}
      {!isVisualMode && overview && <OverviewCards o={overview} t={t} />}

      {/* Insight panel — tree mode only */}
      {!isVisualMode && <InsightPanel insights={archInsights} />}

      {/* Visual mode insight bar */}
      {isVisualMode && visualInsights.length > 0 && (
        <div className="flex items-start gap-2 border-b border-gray-100 px-6 py-2 dark:border-stone-800">
          <Lightbulb size={14} className="mt-0.5 shrink-0 text-amber-500" />
          <div className="flex flex-wrap gap-x-4 gap-y-0.5">
            {visualInsights.map((insight, i) => (
              <span key={i} className="text-xs text-gray-600 dark:text-stone-300">
                {insight}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Main content */}
      {isVisualMode ? (
        /* Visual mode: full-height visualization + side detail panel */
        <div className="flex h-[calc(100vh-var(--header-height,49px)-49px-36px)]">
          <div className="flex-1 overflow-hidden bg-white dark:bg-stone-950">
            <Suspense fallback={<div className="flex items-center justify-center h-full"><Loader2 className="animate-spin text-gray-400 dark:text-stone-500" size={24} /></div>}>
              {viewMode === 'treemap' && (
                <Treemap root={archTree} tensors={allTensors} onNodeClick={handleVisualizationNodeClick} />
              )}
              {viewMode === '3d' && (
                <ForceGraph3D root={archTree} tensors={allTensors} onNodeClick={handleVisualizationNodeClick} />
              )}
            </Suspense>
          </div>
          {detailOpen && selectedNode && (
            <div className="w-80 shrink-0">
              <DetailPanel node={selectedNode} tensors={componentTensors} rootSize={rootSize} t={t} onClose={() => setDetailOpen(false)} />
            </div>
          )}
        </div>
      ) : (
        /* Tree mode: side-by-side tree + detail */
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
          {/* Tree — 3/5 width */}
          <div className="lg:col-span-3 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            {/* Legend */}
            <div className="flex items-center gap-4 mb-2 text-[10px] text-gray-400 dark:text-stone-500">
              <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />{t('arch.legend.quantized')}</span>
              <span className="flex items-center gap-1"><span className="h-1.5 w-1.5 rounded-full bg-gray-300 dark:bg-stone-600" />{t('arch.legend.fullPrec')}</span>
              <span className="flex items-center gap-1"><span className="h-1 w-4 rounded-full bg-indigo-400/60" />{t('arch.legend.share')}</span>
            </div>
            <div className="max-h-[calc(100vh-26rem)] overflow-y-auto">
              <TreeNode
                node={archTree}
                selectedPrefix={selectedNode?.weight_prefix ?? null}
                onSelect={(n) => { setSelectedNode(n); setDetailOpen(true); }}
                rootSize={rootSize}
              />
            </div>
          </div>

          {/* Detail — 2/5 width */}
          <div className="lg:col-span-2 rounded-xl border border-gray-200 dark:border-stone-700 overflow-hidden">
            {selectedNode ? (
              <DetailPanel
                node={selectedNode}
                tensors={componentTensors}
                rootSize={rootSize}
                t={t}
                onClose={() => setSelectedNode(null)}
              />
            ) : (
              <div className="flex items-center justify-center h-40 text-sm text-gray-400 dark:text-stone-500">
                {t('arch.selectComponent')}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Pruning details — tree mode only */}
      {!isVisualMode && pruningTraces.length > 0 && (
        <div className="mt-4 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
          <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('arch.pruningDetails')}</h3>
          <div className="space-y-2">
            {pruningTraces.map((trace, i) => (
              <details key={i} className="rounded-lg border border-gray-100 dark:border-stone-700">
                <summary className="cursor-pointer px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 dark:text-stone-300 dark:hover:bg-stone-800">
                  {trace.severity === 'major' ? '🔴' : trace.severity === 'minor' ? '🟡' : '🔵'}{' '}
                  {trace.category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}: {trace.description}
                </summary>
                <div className="border-t border-gray-100 px-3 py-2 dark:border-stone-700">
                  <pre className="overflow-x-auto text-xs text-gray-600 dark:text-stone-400">
                    {JSON.stringify(trace.details, null, 2)}
                  </pre>
                </div>
              </details>
            ))}
          </div>
        </div>
      )}

      {/* ── Floating "Ask Model" FAB + Drawer ── */}
      {/* Visual mode w/ detail panel open: shift left by 320px (detail panel width) + 20px margin */}
      {model && archTree && (
        <>
          {!askOpen && (
            <button
              onClick={() => {
                // Determine target: visualSelection (Treemap/3D) > selectedNode (Tree)
                const targetKey = visualSelection
                  ? `vs:${visualSelection.label}`
                  : selectedNode && selectedNode !== archTree
                    ? `node:${selectedNode.weight_prefix}`
                    : null;
                if (!chat.text && !chat.streaming && targetKey) {
                  lastAskedPrefixRef.current = targetKey;
                  if (visualSelection) {
                    chat.send(buildVisualExplainPrompt(visualSelection, model));
                  } else if (selectedNode) {
                    chat.send(buildExplainPrompt(selectedNode, archTree, model));
                  }
                }
                setAskOpen(true);
              }}
              className={cn(
                'fixed bottom-6 z-50 flex items-center gap-2 rounded-full bg-indigo-600 px-4 py-3 text-sm font-medium text-white shadow-lg shadow-indigo-600/25 transition-all hover:bg-indigo-700 hover:shadow-xl active:scale-95',
                isVisualMode && detailOpen && selectedNode ? 'right-[340px]' : 'right-6',
              )}
            >
              <Sparkles size={16} />
              <span>{t('arch.ai.askButton')}</span>
              {(visualSelection || (selectedNode && selectedNode !== archTree)) && (
                <span className="max-w-[160px] truncate rounded bg-indigo-500/50 px-1.5 py-0.5 text-[11px]">
                  {visualSelection?.label || selectedNode?.name}
                </span>
              )}
            </button>
          )}

          {askOpen && (
            <div className={cn(
              'fixed bottom-6 z-50 flex w-[400px] max-h-[70vh] flex-col rounded-2xl border border-gray-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900',
              isVisualMode && detailOpen && selectedNode ? 'right-[340px]' : 'right-6',
            )}>
              <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3 dark:border-stone-800">
                <div className="flex items-center gap-2 min-w-0">
                  <Sparkles size={14} className="shrink-0 text-indigo-500" />
                  <h3 className="text-sm font-semibold text-gray-900 dark:text-stone-100 truncate">{t('arch.ai.askTitle')}</h3>
                </div>
                <div className="flex items-center gap-1">
                  <button onClick={() => { chat.reset(); lastAskedPrefixRef.current = null; }} className="rounded p-1 text-gray-400 hover:text-gray-600 dark:text-stone-500 dark:hover:text-stone-300" title="New conversation"><RotateCcw size={13} /></button>
                  <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:text-gray-600 dark:text-stone-500 dark:hover:text-stone-300"><X size={14} /></button>
                </div>
              </div>
              {(visualSelection || (selectedNode && selectedNode !== archTree)) && (() => {
                const ctxLabel = visualSelection?.label || selectedNode?.name || '';
                const ctxSize = visualSelection?.sizeBytes ?? selectedNode?.total_size_bytes ?? 0;
                const ctxKey = visualSelection ? `vs:${visualSelection.label}` : `node:${selectedNode?.weight_prefix}`;
                const isAggregated = visualSelection?.kind === 'aggregated';
                return (
                  <div className="flex items-center gap-2 border-b border-gray-50 px-4 py-2 dark:border-stone-800">
                    <span className="text-[10px] text-gray-400 dark:text-stone-500">{t('arch.ai.context')}</span>
                    <span className="rounded bg-indigo-100 px-1.5 py-0.5 text-[11px] font-medium text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-300">{ctxLabel}</span>
                    <span className="text-[10px] text-gray-400 dark:text-stone-500">{formatSize(ctxSize)}</span>
                    {isAggregated && visualSelection?.numLayers && (
                      <span className="text-[10px] text-indigo-500 dark:text-indigo-400">×{visualSelection.numLayers}L</span>
                    )}
                    {lastAskedPrefixRef.current !== ctxKey && (
                      <button onClick={() => {
                        lastAskedPrefixRef.current = ctxKey;
                        chat.reset();
                        setTimeout(() => {
                          if (visualSelection) chat.send(buildVisualExplainPrompt(visualSelection, model));
                          else if (selectedNode) chat.send(buildExplainPrompt(selectedNode, archTree, model));
                        }, 50);
                      }}
                        className="ml-auto rounded bg-indigo-600 px-2 py-0.5 text-[10px] font-medium text-white hover:bg-indigo-700">{t('arch.ai.explain')}</button>
                    )}
                  </div>
                );
              })()}
              <div className="flex-1 overflow-y-auto px-4 py-3">
                {chat.status && (
                  <div className="mb-2 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-700 dark:bg-amber-900/20 dark:text-amber-400">
                    Status: {chat.status}
                  </div>
                )}
                {!chat.text && !chat.streaming && (
                  <div className="space-y-2">
                    <p className="text-xs text-gray-400 dark:text-stone-500">{t('arch.ai.hint')}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {[t('arch.ai.q1'), t('arch.ai.q2'), t('arch.ai.q3')].map((q, i) => (
                        <button key={i} onClick={() => chat.send(q)} className="rounded-lg border border-gray-200 px-2 py-1 text-[11px] text-gray-600 hover:bg-gray-50 dark:border-stone-700 dark:text-stone-400 dark:hover:bg-stone-800">{q}</button>
                      ))}
                    </div>
                  </div>
                )}
                {(chat.text || chat.streaming) && (
                  <div className="text-sm leading-relaxed text-gray-700 dark:text-stone-300">
                    <MarkdownContent content={chat.text} />
                    {chat.streaming && <Loader2 size={12} className="inline ml-1 animate-spin text-indigo-400" />}
                  </div>
                )}
              </div>
              <form onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
                className="border-t border-gray-100 px-4 py-3 dark:border-stone-800">
                <div className="flex gap-2">
                  <input type="text" value={askInput} onChange={(e) => setAskInput(e.target.value)} placeholder={t('arch.ai.placeholder')} disabled={chat.streaming}
                    className="flex-1 rounded-lg border border-gray-200 px-3 py-2 text-sm outline-none focus:border-indigo-400 focus:ring-1 focus:ring-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-100 dark:placeholder-stone-500" />
                  {chat.streaming ? (
                    <button type="button" onClick={chat.cancel} className="rounded-lg bg-red-500 px-3 py-2 text-white hover:bg-red-600"><X size={14} /></button>
                  ) : (
                    <button type="submit" disabled={!askInput.trim()} className="rounded-lg bg-indigo-600 px-3 py-2 text-white hover:bg-indigo-700 disabled:opacity-40"><Send size={14} /></button>
                  )}
                </div>
              </form>
            </div>
          )}
        </>
      )}
    </div>
  );
}
