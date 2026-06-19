// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * WeightAnalysis — "Weight X-ray" for the loaded model.
 *
 * Optimization layers (page-optimization-playbook §1):
 *  A. Information archaeology: classified tensors by module class, derived
 *     effective bits/param, surfaced top-N memory hogs + layer-size outliers
 *     — all WITHOUT any new backend (cheap header endpoint already returns
 *     enough).
 *  B. Information design: 4-card identity strip (avg bits, quant coverage,
 *     top hog, layer uniformity), module-class breakdown table, top-hog bar.
 *  C. Visualization: per-layer stacked bar (attn vs mlp vs other) with
 *     outlier flagging; histogram in tensor-detail panel.
 *  D. Model-as-interpreter: auto AI Brief on first render, "Ask the model"
 *     FAB with weight-aware system prompt + per-tensor explain shortcut.
 */
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useModelStore } from '@/stores/modelStore';
import { getWeightStats, getDtypeBreakdown, getTensorFullStats } from '@/api/endpoints';
import type { TensorMeta, TensorStats, DtypeSummary } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { EmptyState } from '@/components/common/EmptyState';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { formatParamCount, formatSize, cn } from '@/lib/utils';
import { NextStepBanner } from '@/components/common/NextStepBanner';
import {
  Loader2, Search, Scissors, MessageCircle, Sparkles, Send, X, Wand2,
  Cpu, Layers, Award, Activity,
} from 'lucide-react';
import Plot from 'react-plotly.js';
import MarkdownContent from '@/components/MarkdownContent';
import {
  aggregateWeights, classifyTensorByName,
  CLASS_LABEL, CLASS_COLOR, buildWeightContextSnippet,
  buildWeightAutoBrief, buildTensorExplainPrompt, getWeightSuggestedPrompts,
  type WeightAggregates,
} from '@/lib/weightInsights';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';

// ───── Module-class breakdown table ─────
function ModuleClassTable({ agg }: { agg: WeightAggregates }) {
  const t = useT();
  return (
    <div className="rounded-xl border border-gray-200 bg-white dark:border-stone-700 dark:bg-stone-900 overflow-hidden">
      <div className="px-4 py-2.5 border-b border-gray-100 dark:border-stone-800 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('weights.moduleClass.title')}</h3>
        <span className="text-[10px] text-gray-400 dark:text-stone-500">{t('weights.moduleClass.subtitle')}</span>
      </div>
      <table className="w-full text-xs">
        <thead className="bg-gray-50 dark:bg-stone-800/40 text-gray-500 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 text-left font-medium">{t('weights.moduleClass.colClass')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('weights.moduleClass.colTensors')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('weights.moduleClass.colSize')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('weights.moduleClass.colShare')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('weights.moduleClass.colQuant')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('weights.moduleClass.colBits')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 dark:divide-stone-800">
          {agg.byClass.map((c) => {
            const quantPct = c.count > 0 ? (c.quantizedCount / c.count) * 100 : 0;
            return (
              <tr key={c.cls} className="hover:bg-gray-50/60 dark:hover:bg-stone-800/40">
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block w-2 h-2 rounded-sm" style={{ background: c.color }} />
                    <span className="font-medium text-gray-700 dark:text-stone-300">{c.label}</span>
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums text-gray-600 dark:text-stone-400">{c.count}</td>
                <td className="px-3 py-1.5 text-right tabular-nums text-gray-600 dark:text-stone-400">{formatSize(c.sizeBytes)}</td>
                <td className="px-3 py-1.5 text-right">
                  <div className="inline-flex items-center gap-1.5">
                    <div className="relative w-16 h-1.5 bg-gray-100 dark:bg-stone-800 rounded overflow-hidden">
                      <div className="absolute inset-y-0 left-0 rounded" style={{ width: `${(c.shareOfTotal * 100).toFixed(1)}%`, background: c.color }} />
                    </div>
                    <span className="tabular-nums text-gray-600 dark:text-stone-400 w-12 text-right">{(c.shareOfTotal * 100).toFixed(1)}%</span>
                  </div>
                </td>
                <td className="px-3 py-1.5 text-right">
                  <span className={cn('tabular-nums font-medium',
                    quantPct === 100 ? 'text-emerald-600 dark:text-emerald-400'
                    : quantPct > 0 ? 'text-amber-600 dark:text-amber-400'
                    : 'text-gray-400 dark:text-stone-500')}>
                    {quantPct.toFixed(0)}%
                  </span>
                  <span className="text-[10px] text-gray-400 dark:text-stone-500 ml-1">({c.quantizedCount}/{c.count})</span>
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums font-mono text-gray-700 dark:text-stone-300">{c.bitsPerParam.toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ───── Top hogs horizontal bar ─────
function TopHogsChart({ agg }: { agg: WeightAggregates }) {
  const t = useT();
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  const data = agg.topHogs.slice(0, 10);
  if (data.length === 0) return null;
  return (
    <div className="rounded-xl border border-gray-200 bg-white dark:border-stone-700 dark:bg-stone-900 p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('weights.topHogs.title')}</h3>
        <span className="text-[10px] text-gray-400 dark:text-stone-500">{t('weights.topHogs.subtitle', { pct: (data[data.length - 1].cumShare * 100).toFixed(1) })}</span>
      </div>
      <Plot
        data={[{
          type: 'bar',
          orientation: 'h',
          y: data.map((h) => h.name.length > 38 ? '…' + h.name.slice(-37) : h.name).reverse(),
          x: data.map((h) => h.sizeBytes / 1e6).reverse(),
          marker: { color: data.map((h) => CLASS_COLOR[h.cls]).reverse() },
          text: data.map((h) => `${(h.share * 100).toFixed(1)}%`).reverse(),
          textposition: 'outside',
          hovertemplate: '%{y}<br>%{x:.1f} MB · %{text}<extra></extra>',
        }]}
        layout={{
          height: Math.max(220, 28 * data.length + 60),
          margin: { t: 8, b: 35, l: 280, r: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          xaxis: { title: { text: 'MB', font: { size: 10 } }, gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
          yaxis: { tickfont: { size: 10, color: dk ? '#d6d3d1' : '#374151', family: 'ui-monospace, SFMono-Regular, monospace' }, automargin: false },
          font: { color: dk ? '#d6d3d1' : '#374151' },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}

// ───── Per-layer stacked bar profile ─────
function LayerProfileChart({ agg }: { agg: WeightAggregates }) {
  const t = useT();
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  if (agg.layerProfile.length === 0) return null;
  const x = agg.layerProfile.map((l) => l.layerIdx);
  const outlierSet = new Set(agg.outlierLayers.map((o) => o.layerIdx));
  const outlinedColor = (cls: 'attn' | 'mlp' | 'other') => agg.layerProfile.map((l) =>
    outlierSet.has(l.layerIdx) ? CLASS_COLOR[cls] : CLASS_COLOR[cls]);
  return (
    <div className="rounded-xl border border-gray-200 bg-white dark:border-stone-700 dark:bg-stone-900 p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('weights.layerProfile.title')}</h3>
        <span className="text-[10px] text-gray-400 dark:text-stone-500">
          {agg.outlierLayers.length > 0
            ? t('weights.layerProfile.outliers', { n: agg.outlierLayers.length })
            : t('weights.layerProfile.uniform')}
        </span>
      </div>
      <Plot
        data={[
          {
            type: 'bar', name: CLASS_LABEL.attn, x, y: agg.layerProfile.map((l) => l.attnBytes / 1e6),
            marker: { color: outlinedColor('attn'), line: { width: agg.layerProfile.map((l) => outlierSet.has(l.layerIdx) ? 1.5 : 0), color: '#dc2626' } },
            hovertemplate: 'L%{x} attn: %{y:.1f} MB<extra></extra>',
          },
          {
            type: 'bar', name: CLASS_LABEL.mlp, x, y: agg.layerProfile.map((l) => l.mlpBytes / 1e6),
            marker: { color: outlinedColor('mlp'), line: { width: agg.layerProfile.map((l) => outlierSet.has(l.layerIdx) ? 1.5 : 0), color: '#dc2626' } },
            hovertemplate: 'L%{x} mlp: %{y:.1f} MB<extra></extra>',
          },
          {
            type: 'bar', name: 'Other', x, y: agg.layerProfile.map((l) => l.otherBytes / 1e6),
            marker: { color: outlinedColor('other'), line: { width: agg.layerProfile.map((l) => outlierSet.has(l.layerIdx) ? 1.5 : 0), color: '#dc2626' } },
            hovertemplate: 'L%{x} other: %{y:.1f} MB<extra></extra>',
          },
        ]}
        layout={{
          barmode: 'stack', height: 240,
          margin: { t: 8, b: 35, l: 50, r: 10 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          legend: { orientation: 'h', y: -0.18, font: { size: 10, color: dk ? '#a8a29e' : '#6b7280' } },
          xaxis: { title: { text: 'Layer index', font: { size: 10 } }, gridcolor: dk ? '#292524' : '#f3f4f6', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
          yaxis: { title: { text: 'MB', font: { size: 10 } }, gridcolor: dk ? '#292524' : '#f3f4f6', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
          font: { color: dk ? '#d6d3d1' : '#374151' },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  );
}

// ───── Main ─────

export default function WeightAnalysis() {
  const model = useModelStore((s) => s.currentModel);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [tensors, setTensors] = useState<TensorMeta[]>([]);
  const [dtypes, setDtypes] = useState<DtypeSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const [nameFilter, setNameFilter] = useState('');
  const [dtypeFilter, setDtypeFilter] = useState('');
  const [classFilter, setClassFilter] = useState<string>('');

  const [selectedTensor, setSelectedTensor] = useState<string | null>(null);
  const [tensorStats, setTensorStats] = useState<TensorStats | null>(null);
  const [loadingStats, setLoadingStats] = useState(false);

  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  const tableContainerRef = useRef<HTMLDivElement>(null);

  // ── Aggregations (pure, derived) ──
  // Pass model.total_params (logical) so avg-bits-per-param uses the meaningful denominator.
  const agg = useMemo(
    () => (tensors.length > 0 ? aggregateWeights(tensors, 10, model?.total_params ?? 0) : null),
    [tensors, model?.total_params],
  );

  // ── Filtering ──
  const filteredTensors = useMemo(() => {
    return tensors.filter((tn) => {
      if (nameFilter && !tn.name.toLowerCase().includes(nameFilter.toLowerCase())) return false;
      if (dtypeFilter && tn.dtype !== dtypeFilter) return false;
      if (classFilter && classifyTensorByName(tn.name) !== classFilter) return false;
      return true;
    });
  }, [tensors, nameFilter, dtypeFilter, classFilter]);

  const uniqueDtypes = useMemo(() => Array.from(new Set(tensors.map((tn) => tn.dtype))).sort(), [tensors]);

  const rowVirtualizer = useVirtualizer({
    count: filteredTensors.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 36,
    overscan: 24,
  });

  // ── System prompt for chat (model self + weight context) ──
  const weightSystemPrompt = useMemo(() => {
    if (!model) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    if (!agg) return base;
    return base + '\n\n' + buildWeightContextSnippet(agg);
  }, [model, locale, agg]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: weightSystemPrompt,
    maxTokens: 800,
    temperature: 0.55,
  });

  const suggestedPrompts = useMemo(() => getWeightSuggestedPrompts(agg, locale), [agg, locale]);

  // ── Effects ──
  useEffect(() => {
    if (!model) return;
    setLoading(true);
    setSelectedTensor(null);
    setTensorStats(null);
    setNameFilter(''); setDtypeFilter(''); setClassFilter('');
    briefFiredForRef.current = null;
    chat.reset();
    Promise.all([
      getWeightStats(model.model_id),
      getDtypeBreakdown(model.model_id),
    ]).then(([ws, dt]) => {
      setTensors(ws.tensors);
      setDtypes(dt.breakdown);
    }).finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire weight-aware brief once per model (after agg is ready)
  useEffect(() => {
    if (!model || !agg) return;
    if (briefFiredForRef.current === model.model_id) return;
    if (chat.streaming) return;
    briefFiredForRef.current = model.model_id;
    const id = window.setTimeout(() => {
      chat.send(buildWeightAutoBrief(agg, locale));
    }, 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, !!agg, locale]);

  // ── Callbacks ──
  const loadFullStats = useCallback(async (name: string) => {
    if (!model) return;
    setSelectedTensor(name);
    setLoadingStats(true);
    setTensorStats(null);
    try {
      const stats = await getTensorFullStats(model.model_id, name);
      setTensorStats(stats);
    } finally {
      setLoadingStats(false);
    }
  }, [model]);

  const handleExplainTensor = useCallback(() => {
    if (!selectedTensor) return;
    setAskOpen(true);
    chat.send(buildTensorExplainPrompt(selectedTensor, agg, locale));
  }, [selectedTensor, agg, locale, chat]);

  const handleSuggested = useCallback((q: string) => {
    setAskOpen(true);
    chat.send(q);
  }, [chat]);

  // ── Early return after all hooks ──
  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');

  return (
    <div className="space-y-5 pb-12 relative">
      <PageHeader title={t('weights.title')} description={model.model_name} />

      {loading || !agg ? (
        <div className="flex items-center justify-center py-16 text-gray-400">
          <Loader2 className="animate-spin mr-2" size={20} /> {t('common.loading')}
        </div>
      ) : (
        <>
          {/* 4-card identity strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <IdentityCard
              icon={<Cpu size={16} />}
              label={t('weights.idAvgBits')}
              value={`${agg.avgBitsPerLogical.toFixed(2)} bits/param`}
              hint={`${formatSize(agg.totalSize)} / ${formatParamCount(agg.logicalParams || agg.totalStoredElements)} logical params (stored ${formatParamCount(agg.totalStoredElements)} elem)`}
              tone="indigo"
            />
            <IdentityCard
              icon={<Award size={16} />}
              label={t('weights.idQuantCoverage')}
              value={`${agg.quantizedCount}/${agg.totalTensors} (${(agg.quantizedCount / Math.max(agg.totalTensors, 1) * 100).toFixed(0)}%)`}
              hint={agg.embedQuantized ? 'Embeddings quantized too' : 'Embeddings stay full precision'}
              tone={agg.quantizedCount / Math.max(agg.totalTensors, 1) > 0.8 ? 'emerald' : 'amber'}
            />
            <IdentityCard
              icon={<Activity size={16} />}
              label={t('weights.idTopHog')}
              value={agg.topHogs[0] ? `${(agg.topHogs[0].share * 100).toFixed(1)}%` : '—'}
              hint={agg.topHogs[0]?.name}
              tone={agg.topHogs[0] && agg.topHogs[0].share > 0.20 ? 'amber' : 'neutral'}
            />
            <IdentityCard
              icon={<Layers size={16} />}
              label={t('weights.idLayerUniform')}
              value={agg.layerProfile.length === 0
                ? '—'
                : agg.outlierLayers.length === 0
                  ? t('weights.layerUniform')
                  : t('weights.layerOutlierCount', { n: agg.outlierLayers.length })}
              hint={`${agg.layerProfile.length} transformer layers`}
              tone={agg.outlierLayers.length === 0 ? 'emerald' : 'amber'}
            />
          </div>

          {/* AI Brief (model talking to user about its own weights) */}
          <ModelBriefCard
            label={t('weights.briefLabel')}
            status={chat.status}
            text={chat.text}
            streaming={chat.streaming}
            emptyText={t('weights.briefPending')}
            refreshTitle={t('weights.briefRefresh')}
            prompts={suggestedPrompts}
            onRefresh={() => { briefFiredForRef.current = null; chat.reset(); if (agg) chat.send(buildWeightAutoBrief(agg, locale)); }}
            onPrompt={handleSuggested}
          />

          {/* Module-class table + Top hogs (2-col on lg) */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <ModuleClassTable agg={agg} />
            <TopHogsChart agg={agg} />
          </div>

          {/* Per-layer profile (full width) */}
          <LayerProfileChart agg={agg} />

          {/* dtype breakdown */}
          {dtypes.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('weights.dtypeBreakdown')}</h3>
              <table className="w-full text-xs">
                <thead className="text-gray-500 dark:text-stone-400">
                  <tr className="border-b border-gray-100 dark:border-stone-800 text-left">
                    <th className="pb-2 font-medium">dtype</th>
                    <th className="pb-2 font-medium text-right">Count</th>
                    <th className="pb-2 font-medium text-right">Params</th>
                    <th className="pb-2 font-medium text-right">Size</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-stone-800">
                  {dtypes.map((d) => (
                    <tr key={d.dtype}>
                      <td className="py-1.5 font-mono">{d.dtype}</td>
                      <td className="py-1.5 text-right tabular-nums text-gray-600 dark:text-stone-400">{d.count}</td>
                      <td className="py-1.5 text-right tabular-nums text-gray-600 dark:text-stone-400">{formatParamCount(d.params)}</td>
                      <td className="py-1.5 text-right tabular-nums text-gray-600 dark:text-stone-400">{formatSize(d.size)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Filters + tensor table + detail panel */}
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400" />
                <input
                  value={nameFilter}
                  onChange={(e) => setNameFilter(e.target.value)}
                  placeholder={t('weights.searchTensor')}
                  className="rounded-lg border border-gray-200 py-1.5 pl-7 pr-3 text-xs w-56 focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
                />
              </div>
              <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)} className="rounded-lg border border-gray-200 px-2 py-1.5 text-xs dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200">
                <option value="">{t('weights.filterAllClasses')}</option>
                {agg.byClass.map((c) => <option key={c.cls} value={c.cls}>{c.label}</option>)}
              </select>
              <select value={dtypeFilter} onChange={(e) => setDtypeFilter(e.target.value)} className="rounded-lg border border-gray-200 px-2 py-1.5 text-xs dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200">
                <option value="">{t('weights.filterAllDtypes')}</option>
                {uniqueDtypes.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
              <span className="text-[11px] text-gray-400 dark:text-stone-500 ml-auto">
                {filteredTensors.length} / {tensors.length} {t('weights.tensorsLabel')}
              </span>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 rounded-xl border border-gray-200 bg-white dark:border-stone-700 dark:bg-stone-900 overflow-hidden">
                <div className="grid grid-cols-[3fr_1.4fr_0.7fr_0.7fr_0.9fr_0.9fr] px-3 py-2 text-[11px] uppercase tracking-wider text-gray-400 dark:text-stone-500 bg-gray-50 dark:bg-stone-800/40 border-b border-gray-100 dark:border-stone-800">
                  <span>{t('weights.colName')}</span>
                  <span>{t('weights.colShape')}</span>
                  <span>{t('weights.colClass')}</span>
                  <span>{t('weights.colDtype')}</span>
                  <span className="text-right">{t('weights.colSize')}</span>
                  <span className="text-right">{t('weights.colBitsPP')}</span>
                </div>
                <div ref={tableContainerRef} className="max-h-[480px] overflow-y-auto">
                  <div style={{ height: `${rowVirtualizer.getTotalSize()}px`, position: 'relative' }}>
                    {rowVirtualizer.getVirtualItems().map((vr) => {
                      const tn = filteredTensors[vr.index];
                      const cls = classifyTensorByName(tn.name);
                      const bits = tn.num_elements > 0 ? (tn.size_bytes * 8) / tn.num_elements : 0;
                      return (
                        <div
                          key={tn.name}
                          onClick={() => loadFullStats(tn.name)}
                          className={cn(
                            'absolute left-0 w-full grid grid-cols-[3fr_1.4fr_0.7fr_0.7fr_0.9fr_0.9fr] px-3 py-1.5 text-xs cursor-pointer border-b border-gray-50 dark:border-stone-800/60 hover:bg-gray-50 dark:hover:bg-stone-800/60',
                            selectedTensor === tn.name && 'bg-indigo-50 dark:bg-indigo-500/10',
                          )}
                          style={{ height: `${vr.size}px`, transform: `translateY(${vr.start}px)` }}
                        >
                          <span className="font-mono truncate text-gray-700 dark:text-stone-300" title={tn.name}>{tn.name}</span>
                          <span className="text-gray-500 dark:text-stone-400 truncate">[{tn.shape.join('×')}]</span>
                          <span className="text-[10px]">
                            <span className="inline-flex items-center gap-1">
                              <span className="inline-block w-1.5 h-1.5 rounded-sm" style={{ background: CLASS_COLOR[cls] }} />
                              <span className="text-gray-500 dark:text-stone-400">{cls}</span>
                            </span>
                          </span>
                          <span>
                            <span className={cn('rounded px-1 py-0.5 text-[9px] font-medium',
                              tn.is_quantized ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                                              : 'bg-gray-100 text-gray-500 dark:bg-stone-800 dark:text-stone-400',
                            )}>{tn.dtype}</span>
                          </span>
                          <span className="text-right tabular-nums text-gray-600 dark:text-stone-400">{formatSize(tn.size_bytes)}</span>
                          <span className="text-right tabular-nums font-mono text-gray-500 dark:text-stone-400">{bits.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Detail panel */}
              <div className="rounded-xl border border-gray-200 bg-white dark:border-stone-700 dark:bg-stone-900 p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('weights.tensorDetail')}</h3>
                  {selectedTensor && (
                    <button
                      onClick={handleExplainTensor}
                      disabled={chat.streaming}
                      className="flex items-center gap-1 rounded-lg bg-indigo-500/10 px-2 py-1 text-[11px] text-indigo-600 hover:bg-indigo-500/20 dark:text-indigo-400 disabled:opacity-50"
                      title={t('weights.askAboutTensor')}
                    >
                      <Wand2 size={11} />
                      {t('weights.explainTensor')}
                    </button>
                  )}
                </div>
                {!selectedTensor && (
                  <p className="text-xs text-gray-400 dark:text-stone-500">{t('weights.tensorDetailHint')}</p>
                )}
                {loadingStats && (
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    <Loader2 size={12} className="animate-spin" /> {t('weights.loadingStats')}
                  </div>
                )}
                {tensorStats && (
                  <div className="space-y-3">
                    <p className="break-all font-mono text-[10px] text-gray-600 dark:text-stone-400">{tensorStats.name}</p>
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <div><span className="text-gray-500 dark:text-stone-500">Min:</span> <span className="font-mono text-gray-700 dark:text-stone-300">{tensorStats.min_val?.toFixed(4) ?? '—'}</span></div>
                      <div><span className="text-gray-500 dark:text-stone-500">Max:</span> <span className="font-mono text-gray-700 dark:text-stone-300">{tensorStats.max_val?.toFixed(4) ?? '—'}</span></div>
                      <div><span className="text-gray-500 dark:text-stone-500">Mean:</span> <span className="font-mono text-gray-700 dark:text-stone-300">{tensorStats.mean_val?.toFixed(4) ?? '—'}</span></div>
                      <div><span className="text-gray-500 dark:text-stone-500">Std:</span> <span className="font-mono text-gray-700 dark:text-stone-300">{tensorStats.std_val?.toFixed(4) ?? '—'}</span></div>
                      <div><span className="text-gray-500 dark:text-stone-500">{t('weights.sparsity')}:</span> <span className="font-mono text-gray-700 dark:text-stone-300">{((tensorStats.sparsity ?? 0) * 100).toFixed(1)}%</span></div>
                      {tensorStats.is_quantized && (
                        <div><span className="text-gray-500 dark:text-stone-500">Quant:</span> <span className="font-mono text-gray-700 dark:text-stone-300">{tensorStats.quant_bits}b g{tensorStats.quant_group_size}</span></div>
                      )}
                    </div>
                    {tensorStats.histogram_counts && tensorStats.histogram_edges && (
                      <Plot
                        data={[{
                          type: 'bar',
                          x: tensorStats.histogram_edges.slice(0, -1).map((e, i) =>
                            (e + (tensorStats.histogram_edges![i + 1] ?? e)) / 2,
                          ),
                          y: tensorStats.histogram_counts,
                          marker: { color: CLASS_COLOR[classifyTensorByName(tensorStats.name)] },
                        }]}
                        layout={{
                          height: 180,
                          margin: { t: 8, b: 30, l: 40, r: 10 },
                          paper_bgcolor: 'transparent',
                          plot_bgcolor: 'transparent',
                          xaxis: { title: { text: 'Value', font: { size: 9 } }, tickfont: { size: 9, color: dk ? '#a8a29e' : '#6b7280' } },
                          yaxis: { tickfont: { size: 9, color: dk ? '#a8a29e' : '#6b7280' }, type: 'log' },
                          font: { color: dk ? '#d6d3d1' : '#374151' },
                        }}
                        config={{ displayModeBar: false, responsive: true }}
                        style={{ width: '100%' }}
                      />
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <NextStepBanner steps={[
            { label: t('nextSteps.pruning'), description: t('nextSteps.pruning.desc'), path: '/pruning', icon: <Scissors size={16} /> },
            { label: t('nextSteps.chat'), description: t('nextSteps.chat.desc'), path: '/chat', icon: <MessageCircle size={16} /> },
          ]} />
        </>
      )}

      {/* Floating "Ask Model" FAB + drawer */}
      {agg && (
        <>
          {!askOpen && (
            <button
              onClick={() => setAskOpen(true)}
              className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600 dark:bg-indigo-500 dark:hover:bg-indigo-400"
              title={t('weights.askModel')}
            >
              <Sparkles size={16} />
              {selectedTensor ? t('weights.askAboutSelected') : t('weights.askModel')}
            </button>
          )}
          {askOpen && (
            <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
                <div className="flex items-center gap-2">
                  <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
                  <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('weights.askModel')}</span>
                  {selectedTensor && <span className="text-[10px] text-gray-400 dark:text-stone-500 truncate max-w-[180px]">{selectedTensor}</span>}
                </div>
                <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
                  <X size={14} />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 text-sm">
                {chat.text ? (
                  <MarkdownContent content={chat.text} />
                ) : (
                  <p className="text-gray-400 dark:text-stone-500 text-xs">{t('weights.askEmpty')}</p>
                )}
                {chat.streaming && <Loader2 size={14} className="animate-spin text-indigo-500 mt-2" />}
              </div>
              <div className="border-t border-gray-100 dark:border-stone-800 p-3 space-y-2">
                {selectedTensor && (
                  <button
                    onClick={handleExplainTensor}
                    disabled={chat.streaming}
                    className="w-full text-left text-[11px] rounded-lg border border-indigo-200 px-2 py-1.5 text-indigo-700 hover:bg-indigo-50 dark:border-indigo-500/30 dark:text-indigo-300 dark:hover:bg-indigo-500/10 disabled:opacity-50"
                  >
                    <Wand2 size={11} className="inline mr-1" />
                    {t('weights.explainSelected')}
                  </button>
                )}
                <form
                  onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
                  className="flex gap-2"
                >
                  <input
                    value={askInput}
                    onChange={(e) => setAskInput(e.target.value)}
                    placeholder={t('weights.askPlaceholder')}
                    disabled={chat.streaming}
                    className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
                  />
                  <button
                    type="submit"
                    disabled={!askInput.trim() || chat.streaming}
                    className="rounded-lg bg-indigo-500 px-3 py-1.5 text-white hover:bg-indigo-600 disabled:opacity-50"
                  >
                    <Send size={12} />
                  </button>
                </form>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
