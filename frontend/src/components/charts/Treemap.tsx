// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Architecture Treemap — "Weight Budget"
 *
 * Reconstructs sub-component structure from tensor names and shows
 * a functional decomposition: Attention / FFN / Norms / Embeddings / Head.
 *
 * Area = bytes. Color = functional group (high-contrast saturated palette).
 */

import { useMemo, useRef, useCallback, useEffect } from 'react';
import Plot from 'react-plotly.js';
import type { ArchNode, TensorMeta } from '@/api/types';
import { formatParamCount, formatSize } from '@/lib/utils';

/** Payload sent on click. Aggregated nodes (synthetic) have label/scope fields. */
export interface VisualClickPayload {
  /** weight prefix of an ArchNode (Layer 0 for aggregated nodes), or '' for synthetic */
  prefix: string;
  /** display name (e.g., "Q Projection", "Feed-Forward") */
  label: string;
  /** total bytes for this aggregated/single component */
  sizeBytes: number;
  /** total params/elements */
  params: number;
  /** "single" = direct ArchNode, "aggregated" = sum across N layers */
  kind: 'single' | 'aggregated';
  /** if aggregated, how many layers were summed */
  numLayers?: number;
  /** module name for aggregated (self_attn, mlp, ...) */
  module?: string;
  /** sub-name for aggregated (q_proj, gate_proj, ...) */
  sub?: string;
}

interface TreemapProps {
  root: ArchNode;
  tensors?: TensorMeta[];
  onNodeClick?: (payload: VisualClickPayload) => void;
}

// ─── High-contrast palette: all dark enough for white text ──────────────────

function colorFor(name: string): string {
  const lower = name.toLowerCase();
  // Attention — deep blue family
  if (lower === 'self_attn' || lower === 'attention') return '#1e40af';
  if (lower === 'q_proj') return '#1d4ed8';
  if (lower === 'k_proj') return '#2563eb';
  if (lower === 'v_proj') return '#1e3a8a';
  if (lower === 'o_proj') return '#1e40af';
  if (lower.includes('q_norm') || lower.includes('k_norm')) return '#3b82f6';
  if (lower.includes('attn') || lower.includes('attention')) return '#1e40af';
  // FFN — deep purple family
  if (lower === 'mlp' || lower === 'feed_forward') return '#6d28d9';
  if (lower === 'gate_proj' || lower === 'w1') return '#7c3aed';
  if (lower === 'up_proj' || lower === 'w3') return '#6d28d9';
  if (lower === 'down_proj' || lower === 'w2') return '#5b21b6';
  if (lower.includes('mlp')) return '#6d28d9';
  // Norms — dark teal
  if (lower.includes('norm')) return '#0f766e';
  // Embeddings — dark amber
  if (lower.includes('embed')) return '#b45309';
  // Head — dark rose
  if (lower.includes('head')) return '#9f1239';
  return '#475569'; // slate-600
}

const FRIENDLY: Record<string, string> = {
  self_attn: 'Attention', attention: 'Attention',
  mlp: 'Feed-Forward', feed_forward: 'Feed-Forward',
  q_proj: 'Q Projection', k_proj: 'K Projection',
  v_proj: 'V Projection', o_proj: 'O Projection',
  q_norm: 'Q Norm', k_norm: 'K Norm',
  gate_proj: 'Gate', up_proj: 'Up', down_proj: 'Down',
  w1: 'Gate (W1)', w2: 'Down (W2)', w3: 'Up (W3)',
  input_layernorm: 'Pre-Attn Norm', post_attention_layernorm: 'Post-Attn Norm',
  embed_tokens: 'Token Embeddings', lm_head: 'Output Head',
  norm: 'Final Norm', model_norm: 'Final Norm',
};
function friendly(n: string) { return FRIENDLY[n] || n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); }

// ─── Tensor-based decomposition ─────────────────────────────────────────────

function decomposeLayer(tensors: TensorMeta[], prefix: string) {
  const map = new Map<string, { bytes: number; elems: number; module: string; sub: string }>();
  for (const t of tensors) {
    if (!t.name.startsWith(prefix + '.')) continue;
    const parts = t.name.slice(prefix.length + 1).split('.');
    let module: string, sub: string, key: string;
    if (parts.length >= 3) { module = parts[0]; sub = parts[1]; key = `${module}.${sub}`; }
    else { module = parts[0]; sub = ''; key = module; }
    const b = map.get(key) || { bytes: 0, elems: 0, module, sub };
    b.bytes += t.size_bytes;
    b.elems += t.num_elements;
    map.set(key, b);
  }
  return map;
}

interface FNode {
  id: string; label: string; parentId: string;
  sizeBytes: number; params: number; color: string;
  hoverExtra: string;
  /** Click payload to send back when this node is selected */
  payload: VisualClickPayload;
}

function buildFunctionalData(root: ArchNode, tensors: TensorMeta[]): FNode[] {
  const nodes: FNode[] = [];
  const lg = root.children.find(c => c.node_type === 'layer_group');
  const others = root.children.filter(c => c !== lg);
  const nL = lg?.children.length ?? 0;

  // Root
  nodes.push({
    id: root.name, label: root.name, parentId: '',
    sizeBytes: root.total_size_bytes, params: root.total_param_count,
    color: '#334155', hoverExtra: '',
    payload: {
      prefix: root.weight_prefix, label: root.name,
      sizeBytes: root.total_size_bytes, params: root.total_param_count,
      kind: 'single',
    },
  });

  // Non-layer children (real ArchNodes)
  for (const ch of others) {
    nodes.push({
      id: `${root.name}/${ch.name}`, label: friendly(ch.name),
      parentId: root.name, sizeBytes: ch.total_size_bytes, params: ch.total_param_count,
      color: colorFor(ch.name), hoverExtra: '',
      payload: {
        prefix: ch.weight_prefix, label: friendly(ch.name),
        sizeBytes: ch.total_size_bytes, params: ch.total_param_count,
        kind: 'single',
      },
    });
  }

  if (!lg || nL === 0 || tensors.length === 0) return nodes;

  // Aggregate across layers
  const modAgg = new Map<string, { bytes: number; elems: number;
    subs: Map<string, { bytes: number; elems: number }> }>();

  for (const layer of lg.children) {
    for (const [, b] of decomposeLayer(tensors, layer.weight_prefix)) {
      let mod = modAgg.get(b.module);
      if (!mod) { mod = { bytes: 0, elems: 0, subs: new Map() }; modAgg.set(b.module, mod); }
      mod.bytes += b.bytes; mod.elems += b.elems;
      if (b.sub) {
        let sub = mod.subs.get(b.sub);
        if (!sub) { sub = { bytes: 0, elems: 0 }; mod.subs.set(b.sub, sub); }
        sub.bytes += b.bytes; sub.elems += b.elems;
      }
    }
  }

  // Aggregated module nodes (Attention, Feed-Forward, ...)
  for (const [modName, mod] of modAgg) {
    const gId = `${root.name}/${modName}`;
    nodes.push({
      id: gId, label: friendly(modName), parentId: root.name,
      sizeBytes: mod.bytes, params: mod.elems, color: colorFor(modName),
      hoverExtra: `${nL} layers × ${formatSize(mod.bytes / nL)}/layer`,
      payload: {
        prefix: lg.children[0].weight_prefix,
        label: friendly(modName),
        sizeBytes: mod.bytes, params: mod.elems,
        kind: 'aggregated',
        numLayers: nL,
        module: modName,
      },
    });

    // Sub-components (Q Proj, K Proj, ...)
    for (const [subName, sub] of mod.subs) {
      nodes.push({
        id: `${gId}/${subName}`, label: friendly(subName),
        parentId: gId, sizeBytes: sub.bytes, params: sub.elems,
        color: colorFor(subName),
        hoverExtra: `${nL}L × ${formatSize(sub.bytes / nL)}/layer`,
        payload: {
          prefix: lg.children[0].weight_prefix,
          label: friendly(subName),
          sizeBytes: sub.bytes, params: sub.elems,
          kind: 'aggregated',
          numLayers: nL,
          module: modName,
          sub: subName,
        },
      });
    }
  }
  return nodes;
}

// ─── Component ──────────────────────────────────────────────────────────────

export function Treemap({ root, tensors = [], onNodeClick }: TreemapProps) {
  const payloadRef = useRef<VisualClickPayload[]>([]);
  const idsMapRef = useRef<Map<string, number>>(new Map());
  const onClickRef = useRef(onNodeClick);

  const trace = useMemo(() => {
    const fn = buildFunctionalData(root, tensors);
    return {
      ids: fn.map(n => n.id), labels: fn.map(n => n.label),
      parents: fn.map(n => n.parentId), values: fn.map(n => Math.max(n.sizeBytes, 1)),
      colors: fn.map(n => n.color),
      payloads: fn.map(n => n.payload),
      idsMap: new Map(fn.map((n, i) => [n.id, i])),
      cd: fn.map(n => {
        const bpp = n.params > 0 ? (n.sizeBytes * 8 / n.params).toFixed(1) + 'b' : '—';
        const pct = root.total_size_bytes > 0 ? (n.sizeBytes / root.total_size_bytes * 100).toFixed(1) + '%' : '—';
        return [formatSize(n.sizeBytes), formatParamCount(n.params), bpp, pct, n.hoverExtra || ''];
      }),
    };
  }, [root, tensors]);

  useEffect(() => {
    onClickRef.current = onNodeClick;
    payloadRef.current = trace.payloads;
    idsMapRef.current = trace.idsMap;
  }, [onNodeClick, trace.idsMap, trace.payloads]);

  // Resolve click event → payload, called by both onClick prop and direct gd listener
  const handlePoint = useCallback((point: { pointNumber?: number; id?: string }) => {
    let p = point.pointNumber !== undefined ? payloadRef.current[point.pointNumber] : null;
    if (!p && point.id) {
      const idx = idsMapRef.current.get(point.id);
      if (idx !== undefined) p = payloadRef.current[idx];
    }
    if (p) onClickRef.current?.(p);
  }, []);

  // Backup onClick prop for React-Plotly
  const onClick = useCallback((e: Plotly.PlotMouseEvent) => {
    if (!e.points?.[0]) return;
    handlePoint(e.points[0] as { pointNumber?: number; id?: string });
  }, [handlePoint]);

  // Primary handler: attach via onInitialized to the graph div directly.
  // This works around plotly.js's flaky first-click on treemap.
  const onInitialized = useCallback((_figure: unknown, gd: HTMLElement) => {
    const gdAny = gd as HTMLElement & { on?: (ev: string, cb: (e: { points?: object[] }) => void) => void };
    if (gdAny.on) {
      gdAny.on('plotly_treemapclick', (ev) => {
        const pt = ev.points?.[0] as { pointNumber?: number; id?: string } | undefined;
        if (pt) handlePoint(pt);
      });
      gdAny.on('plotly_click', (ev) => {
        const pt = ev.points?.[0] as { pointNumber?: number; id?: string } | undefined;
        if (pt) handlePoint(pt);
      });
    }
  }, [handlePoint]);

  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');

  return (
    <Plot
      data={[{
        type: 'treemap', ids: trace.ids, labels: trace.labels,
        parents: trace.parents, values: trace.values, branchvalues: 'total',
        customdata: trace.cd,
        marker: { colors: trace.colors, line: { width: 2, color: dk ? '#0c0a09' : '#f5f5f4' } },
        texttemplate: '<b>%{label}</b><br>%{customdata[0]}<br>%{customdata[3]} of model',
        textfont: { size: 13, color: '#f5f5f4', family: 'system-ui, -apple-system, sans-serif' },
        hovertemplate:
          '<b>%{label}</b><br>───────────────' +
          '<br>Size: %{customdata[0]}  (%{customdata[3]} of model)' +
          '<br>Params: %{customdata[1]}  ·  Precision: %{customdata[2]}' +
          '<br>%{customdata[4]}<extra></extra>',
        pathbar: { visible: true, thickness: 28,
          textfont: { size: 12, color: dk ? '#d6d3d1' : '#292524' }, side: 'top' },
        tiling: { packing: 'squarify', pad: 3 },
      } as unknown as Plotly.Data]}
      layout={{
        margin: { t: 36, l: 4, r: 4, b: 4 },
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'system-ui, -apple-system, sans-serif', color: dk ? '#a8a29e' : '#57534e' },
        autosize: true,
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      onClick={onClick}
      onInitialized={onInitialized}
    />
  );
}
