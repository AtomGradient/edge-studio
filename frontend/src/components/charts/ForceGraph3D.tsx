// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * 3D Architecture — "Memory Landscape"
 *
 * Surface plot: X=Layer, Y=Component, Z=Size(MB).
 * High-contrast "turbo" colorscale that reads well in both light/dark.
 */

import { useMemo, useCallback, useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';
import type { ArchNode, TensorMeta } from '@/api/types';
import { formatSize } from '@/lib/utils';
import type { VisualClickPayload } from './Treemap';

interface ForceGraph3DProps {
  root: ArchNode;
  tensors?: TensorMeta[];
  onNodeClick?: (payload: VisualClickPayload) => void;
}

const SHORT: Record<string, string> = {
  q_proj: 'Q', k_proj: 'K', v_proj: 'V', o_proj: 'O', qkv_proj: 'QKV',
  q_norm: 'QNorm', k_norm: 'KNorm',
  gate_proj: 'Gate', up_proj: 'Up', down_proj: 'Down',
  w1: 'W1', w2: 'W2', w3: 'W3',
  input_layernorm: 'Norm₁', post_attention_layernorm: 'Norm₂',
};

function decomposeLayer(tensors: TensorMeta[], prefix: string) {
  const map = new Map<string, number>();
  for (const t of tensors) {
    if (!t.name.startsWith(prefix + '.')) continue;
    const parts = t.name.slice(prefix.length + 1).split('.');
    const key = parts.length > 1 ? parts.slice(0, -1).join('.') : parts[0];
    map.set(key, (map.get(key) || 0) + t.size_bytes);
  }
  return map;
}

function buildSurface(root: ArchNode, tensors: TensorMeta[]) {
  const lg = root.children.find(c => c.node_type === 'layer_group');
  if (!lg || lg.children.length === 0 || tensors.length === 0) return null;

  const layers = lg.children;
  const nL = layers.length;
  const tpl = decomposeLayer(tensors, layers[0].weight_prefix);
  const keys = Array.from(tpl.keys()).sort();
  const labels = keys.map(k => { const p = k.split('.').pop() || k; return SHORT[p] || p; });
  if (keys.length === 0) return null;

  const z: number[][] = [];
  const hover: string[][] = [];
  for (let i = 0; i < nL; i++) {
    const m = decomposeLayer(tensors, layers[i].weight_prefix);
    const zr: number[] = [], hr: string[] = [];
    for (let j = 0; j < keys.length; j++) {
      const b = m.get(keys[j]) || 0;
      zr.push(b / 1e6);
      hr.push(`<b>Layer ${i} · ${labels[j]}</b><br>${formatSize(b)}`);
    }
    z.push(zr); hover.push(hr);
  }
  return { z, hover, labels, nL, nC: keys.length };
}

export function ForceGraph3D({ root, tensors = [], onNodeClick }: ForceGraph3DProps) {
  const data = useMemo(() => buildSurface(root, tensors), [root, tensors]);
  const onClickRef = useRef(onNodeClick);
  const dataRef = useRef(data);

  useEffect(() => {
    onClickRef.current = onNodeClick;
    dataRef.current = data;
  }, [data, onNodeClick]);

  const handlePoint = useCallback((point: { x?: number; y?: number; pointNumber?: number | number[] }) => {
    const d = dataRef.current;
    const cb = onClickRef.current;
    if (!cb || !d) return;
    const lg = root.children.find(c => c.node_type === 'layer_group');
    if (!lg) return;
    let xi = 0, yi = 0;
    if (Array.isArray(point.pointNumber)) {
      yi = point.pointNumber[0];
      xi = point.pointNumber[1];
    } else if (typeof point.x === 'number' && typeof point.y === 'number') {
      xi = point.x;
      yi = point.y;
    }
    if (xi < 0 || xi >= lg.children.length || yi < 0 || yi >= d.labels.length) return;
    const layer = lg.children[xi];
    const componentLabel = d.labels[yi] || '?';
    const componentBytes = d.z[xi]?.[yi] ? d.z[xi][yi] * 1e6 : 0;
    cb({
      prefix: layer.weight_prefix,
      label: `Layer ${xi} · ${componentLabel}`,
      sizeBytes: componentBytes,
      params: 0,
      kind: 'single',
      module: componentLabel,
    });
  }, [root]);

  const onClick = useCallback((e: Plotly.PlotMouseEvent) => {
    if (e.points?.[0]) handlePoint(e.points[0] as { x?: number; y?: number; pointNumber?: number | number[] });
  }, [handlePoint]);

  const onInitialized = useCallback((_figure: unknown, gd: HTMLElement) => {
    const gdAny = gd as HTMLElement & { on?: (ev: string, cb: (e: { points?: object[] }) => void) => void };
    if (gdAny.on) {
      gdAny.on('plotly_click', (ev) => {
        const pt = ev.points?.[0] as { x?: number; y?: number; pointNumber?: number | number[] } | undefined;
        if (pt) handlePoint(pt);
      });
    }
  }, [handlePoint]);

  if (!data) return <div className="flex items-center justify-center h-full text-sm text-gray-400">No layer tensor data</div>;

  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  const bg = dk ? '#0c0a09' : '#fafaf9';
  const tx = dk ? '#a8a29e' : '#57534e';
  const grid = dk ? 'rgba(168,162,158,0.12)' : 'rgba(87,83,78,0.08)';
  const maxZ = Math.max(...data.z.flat(), 1);

  // High-contrast "turbo-like" colorscale: dark blue → cyan → yellow → red
  const cs: [number, string][] = [
    [0, dk ? '#172554' : '#dbeafe'],
    [0.2, '#2563eb'],
    [0.4, '#06b6d4'],
    [0.6, '#eab308'],
    [0.8, '#f97316'],
    [1.0, '#dc2626'],
  ];

  return (
    <Plot
      data={[{
        type: 'surface',
        z: data.z, x: Array.from({ length: data.nL }, (_, i) => i),
        y: data.labels.map((_, i) => i),
        text: data.hover, hoverinfo: 'text',
        colorscale: cs, cmin: 0, cmax: maxZ,
        colorbar: {
          title: { text: 'MB', font: { size: 10, color: tx } },
          thickness: 12, len: 0.4, y: 0.5,
          tickfont: { size: 9, color: tx }, outlinewidth: 0,
        },
        contours: { z: { show: true, usecolormap: true, highlightcolor: dk ? '#fafaf9' : '#1c1917', project: { z: false } } },
        lighting: { ambient: 0.75, diffuse: 0.5, specular: 0.3, roughness: 0.5 },
        opacity: 0.95,
      } as unknown as Plotly.Data]}
      layout={{
        scene: {
          xaxis: { title: { text: 'Layer', font: { size: 11, color: tx } },
            showgrid: true, gridcolor: grid, tickfont: { size: 9, color: tx },
            dtick: data.nL > 20 ? 5 : data.nL > 10 ? 2 : 1,
            showspikes: false, backgroundcolor: bg },
          yaxis: { title: { text: 'Component', font: { size: 11, color: tx } },
            showgrid: true, gridcolor: grid, tickfont: { size: 9, color: tx },
            tickvals: data.labels.map((_, i) => i), ticktext: data.labels,
            showspikes: false, backgroundcolor: bg },
          zaxis: { title: { text: 'MB', font: { size: 11, color: tx } },
            showgrid: true, gridcolor: grid, tickfont: { size: 9, color: tx },
            showspikes: false, backgroundcolor: bg },
          camera: { eye: { x: 1.6, y: -2.0, z: 1.0 }, up: { x: 0, y: 0, z: 1 } },
          aspectmode: 'manual', aspectratio: { x: 2, y: 1.2, z: 0.8 },
        },
        margin: { t: 0, l: 0, r: 0, b: 0 },
        paper_bgcolor: bg, showlegend: false, autosize: true,
        font: { family: 'system-ui, -apple-system, sans-serif', color: tx },
      }}
      config={{ responsive: true, displayModeBar: true, displaylogo: false,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'] }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      onClick={onClick}
      onInitialized={onInitialized}
    />
  );
}
