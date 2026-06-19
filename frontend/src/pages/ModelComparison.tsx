// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * ModelComparison — A vs B X-ray.
 *
 * Optimization layers (page-optimization-playbook §1):
 *  A. Information archaeology: backend has rich data (arch_diff + per-layer
 *     latency + bottleneck profile), frontend was rendering it as neutral
 *     tables with NO derived insight (InsightPanel was 2 hardcoded strings).
 *  B. Information design: 4-card delta strip (size / speed / context / arch)
 *     + AI verdict surfacing "which to pick for which device".
 *  C. Visualization: keep latency comparison + prefill/decode + bottleneck
 *     tables, add dark mode + tone arch diff rows by importance.
 *  D. Model-as-interpreter: the primary model speaks in first person,
 *     comparing itself to the comparison model with concrete numbers.
 */
import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import { useModelStore } from '@/stores/modelStore';
import { useUIStore } from '@/stores/uiStore';
import { compareModels } from '@/api/endpoints';
import type { ComparisonResult, LatencyProfile, BottleneckLayer, ArchDiff } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { IdentityCard } from '@/components/common/IdentityCard';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { cn, formatSize, formatParamCount } from '@/lib/utils';
import {
  GitCompare, Sparkles, Send, X, RotateCcw, Cpu, Zap, Layers as LayersIcon, Smartphone,
  Loader2, RefreshCw,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt, deriveModelFacts } from '@/lib/chatPrompts';
import {
  deriveComparisonDeltas, buildComparisonContextSnippet, buildComparisonAutoBrief,
  getComparisonSuggestedPrompts, deviceFitVerdict,
} from '@/lib/comparisonInsights';

export default function ModelComparison() {
  const model = useModelStore((s) => s.currentModel);
  const compModel = useModelStore((s) => s.comparisonModel);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [prompt, setPrompt] = useState('Hi, how are you?');
  const [maxTokens, setMaxTokens] = useState(50);
  const [enableTiming, setEnableTiming] = useState(true);

  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // AI brief / Ask
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  const { setFileBrowserOpen, sidebarOpen, toggleSidebar } = useUIStore();

  const handleLoadComparisonModel = useCallback(() => {
    if (!sidebarOpen) toggleSidebar();
    setFileBrowserOpen(true);
  }, [sidebarOpen, toggleSidebar, setFileBrowserOpen]);

  // Derived deltas
  const deltas = useMemo(
    () => deriveComparisonDeltas(model, compModel, result, locale),
    [model, compModel, result, locale],
  );
  const factsA = useMemo(() => (model ? deriveModelFacts(model) : null), [model]);
  const factsB = useMemo(() => (compModel ? deriveModelFacts(compModel) : null), [compModel]);
  const fit = useMemo(
    () => (factsA && factsB ? deviceFitVerdict(factsA, factsB) : null),
    [factsA, factsB],
  );

  // System prompt (model self + comparison context)
  const systemPrompt = useMemo(() => {
    if (!model || !compModel) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    return base + '\n\n' + buildComparisonContextSnippet(model, compModel, result);
  }, [model, compModel, locale, result]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt,
    maxTokens: 800,
    temperature: 0.55,
  });

  const suggestedPrompts = useMemo(
    () => (model && compModel && deltas ? getComparisonSuggestedPrompts(model, compModel, result, deltas, locale) : []),
    [model, compModel, result, deltas, locale],
  );

  // Reset chat when either model changes
  useEffect(() => {
    setResult(null);
    setAskOpen(false);
    briefFiredForRef.current = null;
    chat.reset();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, compModel?.model_id]);

  // Auto-fire brief once both models are present (and refire when result arrives)
  useEffect(() => {
    if (!model || !compModel || !deltas) return;
    const key = `${model.model_id}:${compModel.model_id}:${result ? 'with-latency' : 'no-latency'}`;
    if (briefFiredForRef.current === key) return;
    if (chat.streaming) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => {
      chat.send(buildComparisonAutoBrief(model, compModel, result, deltas, locale));
    }, 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, compModel?.model_id, !!result, locale]);

  const handleCompare = useCallback(async () => {
    if (!model || !compModel) return;
    setError(null);
    try {
      const { task_id } = await compareModels(
        model.model_id,
        compModel.model_id,
        prompt,
        maxTokens,
      );
      setTaskId(task_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Comparison failed');
    }
  }, [model, compModel, prompt, maxTokens]);

  const handleSuggested = useCallback((q: string) => {
    setAskOpen(true);
    chat.send(q);
  }, [chat]);

  if (!model) {
    return (
      <EmptyState
        icon={<GitCompare size={48} />}
        title={t('compare.noModelTitle')}
        description={t('compare.noModelDesc')}
      />
    );
  }

  if (!compModel) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="mb-4 text-gray-400 dark:text-stone-500">
          <GitCompare size={48} />
        </div>
        <h3 className="mb-2 text-lg font-semibold text-gray-700 dark:text-stone-200">{t('compare.emptyTitle')}</h3>
        <p className="mb-4 max-w-md text-sm text-gray-500 dark:text-stone-400">{t('compare.emptyDesc')}</p>
        <div className="mb-6 max-w-md space-y-2 text-left text-sm text-gray-600 dark:text-stone-300">
          <p>{t('compare.step1')}</p>
          <p>{t('compare.step2')}</p>
          <p>{t('compare.step3')}</p>
          <p>{t('compare.step4')}</p>
        </div>
        <button
          onClick={handleLoadComparisonModel}
          className="rounded-lg bg-indigo-500 px-5 py-2 text-sm font-medium text-white hover:bg-indigo-600"
        >
          {t('compare.loadModel')}
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-5 pb-12 relative">
      <PageHeader
        title={t('compare.title')}
        description={`${model.model_name} vs ${compModel.model_name}`}
      />

      {/* 4-card delta strip */}
      {deltas && factsA && factsB && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <IdentityCard
            icon={<Cpu size={16} />}
            label={t('compare.idSize')}
            value={`${formatSize(factsA.totalSizeBytes)} ↔ ${formatSize(factsB.totalSizeBytes)}`}
            hint={deltas.oneLineSize}
            tone={Math.abs(deltas.sizeRatio - 1) > 0.5 ? 'amber' : 'neutral'}
          />
          <IdentityCard
            icon={<Zap size={16} />}
            label={t('compare.idSpeed')}
            value={result?.latency_a && result?.latency_b
              ? `${result.latency_a.tokens_per_second.toFixed(1)} ↔ ${result.latency_b.tokens_per_second.toFixed(1)} tok/s`
              : '—'}
            hint={deltas.oneLineSpeed}
            tone={result?.latency_a ? (Math.abs(deltas.tpsRatio - 1) > 0.2 ? 'amber' : 'emerald') : 'neutral'}
          />
          <IdentityCard
            icon={<LayersIcon size={16} />}
            label={t('compare.idArch')}
            value={deltas.sameFamily ? t('compare.sameFamily') : t('compare.crossFamily')}
            hint={`${factsA.family} vs ${factsB.family}, GQA ${factsA.gqaRatio}:1 vs ${factsB.gqaRatio}:1`}
            tone={deltas.sameFamily ? 'emerald' : 'indigo'}
          />
          <IdentityCard
            icon={<Smartphone size={16} />}
            label={t('compare.idIphone')}
            value={fit
              ? `${fit.aFits ? '✓' : '✗'} ↔ ${fit.bFits ? '✓' : '✗'}`
              : '—'}
            hint={fit
              ? `Margin (8GB): A ${fit.aMargin >= 0 ? '+' : ''}${fit.aMargin.toFixed(0)} MB, B ${fit.bMargin >= 0 ? '+' : ''}${fit.bMargin.toFixed(0)} MB`
              : ''}
            tone={fit && fit.aFits && fit.bFits ? 'emerald' : fit && (fit.aFits || fit.bFits) ? 'amber' : 'red'}
          />
        </div>
      )}

      {/* AI Brief */}
      <div className="rounded-xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
              {t('compare.briefLabel')}
            </span>
            {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
          </div>
          {chat.text && !chat.streaming && deltas && (
            <button
              onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildComparisonAutoBrief(model, compModel, result, deltas, locale)); }}
              className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
              title={t('weights.briefRefresh')}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
        <div className="text-sm text-gray-700 dark:text-stone-300">
          {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
          {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('compare.briefPending')}</span>}
          {chat.streaming && chat.text && <span className="inline-block w-1 h-3.5 ml-0.5 bg-indigo-500 animate-pulse rounded-sm" />}
        </div>
        {suggestedPrompts.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {suggestedPrompts.map((sp) => (
              <button
                key={sp.label}
                onClick={() => handleSuggested(sp.prompt)}
                disabled={chat.streaming}
                className="rounded-full border border-indigo-200 bg-white/60 px-2.5 py-0.5 text-[11px] text-indigo-700 hover:bg-white dark:border-indigo-500/30 dark:bg-stone-900/50 dark:text-indigo-300 dark:hover:bg-stone-900 disabled:opacity-50"
              >
                {sp.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-3 dark:border-stone-700 dark:bg-stone-900">
        <div>
          <label className="mb-1 block text-xs text-gray-500 dark:text-stone-400">{t('compare.sharedPrompt')}</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200"
          />
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <div className="max-w-[150px]">
            <label className="mb-1 block text-xs text-gray-500 dark:text-stone-400">{t('compare.maxTokens')}</label>
            <input
              type="number" min={10} max={200} value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200"
            />
          </div>
          <label className="flex items-center gap-2 pt-4 text-sm text-gray-600 dark:text-stone-400">
            <input type="checkbox" checked={enableTiming} onChange={(e) => setEnableTiming(e.target.checked)} className="rounded accent-indigo-500" />
            {t('compare.enableTiming')}
          </label>
          <button
            onClick={handleCompare}
            disabled={!!taskId}
            className={cn(
              'ml-auto flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-xs font-medium transition-colors',
              taskId
                ? 'bg-gray-100 text-gray-400 dark:bg-stone-800 dark:text-stone-500'
                : 'bg-indigo-500 text-white hover:bg-indigo-600',
            )}
          >
            <RefreshCw size={11} className={taskId ? 'animate-spin' : ''} />
            {result ? t('compare.rerun') : t('compare.run')}
          </button>
        </div>
        {error && <p className="text-sm text-red-500">{error}</p>}
      </div>

      {/* Progress overlay */}
      {taskId && (
        <ProgressOverlay
          taskId={taskId}
          title={t('compare.running')}
          onComplete={(r) => { setResult(r as ComparisonResult); setTaskId(null); }}
          onError={() => setTaskId(null)}
          onClose={() => setTaskId(null)}
        />
      )}

      {result && (
        <>
          {/* Architecture diff */}
          {result.arch_diff && (
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('compare.archDiffTitle')}</h3>
              <ArchDiffTable diff={result.arch_diff} />
            </div>
          )}

          {/* Latency comparison + Prefill/Decode */}
          {result.latency_a && result.latency_b && (
            <>
              <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
                <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('compare.perLayerLatency')}</h3>
                <LatencyComparisonChart a={result.latency_a} b={result.latency_b} />
              </div>

              <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
                <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('compare.prefillDecode')}</h3>
                <PrefillDecodeBar a={result.latency_a} b={result.latency_b} />
              </div>

              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
                  <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-stone-300">
                    {t('compare.bottlenecksA')} — {result.latency_a.model_name}
                  </h3>
                  <BottleneckTable layers={result.bottlenecks_a} />
                </div>
                <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
                  <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-stone-300">
                    {t('compare.bottlenecksB')} — {result.latency_b.model_name}
                  </h3>
                  <BottleneckTable layers={result.bottlenecks_b} />
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* FAB */}
      {!askOpen && (
        <button
          onClick={() => setAskOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
          title={t('compare.askModel')}
        >
          <Sparkles size={16} /> {t('compare.askModel')}
        </button>
      )}
      {askOpen && (
        <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('compare.askModel')}</span>
            </div>
            <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
              <X size={14} />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-sm">
            {chat.text ? <MarkdownContent content={chat.text} /> : (
              <p className="text-gray-400 dark:text-stone-500 text-xs">{t('compare.askEmpty')}</p>
            )}
          </div>
          <form
            onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
            className="flex gap-2 border-t border-gray-100 dark:border-stone-800 p-3"
          >
            <input
              value={askInput}
              onChange={(e) => setAskInput(e.target.value)}
              placeholder={t('compare.askPlaceholder')}
              disabled={chat.streaming}
              className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
            />
            <button type="submit" disabled={!askInput.trim() || chat.streaming} className="rounded-lg bg-indigo-500 px-3 py-1.5 text-white hover:bg-indigo-600 disabled:opacity-50">
              <Send size={12} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

// ───── Sub-components (kept from legacy, dark-mode adapted) ─────

function ArchDiffTable({ diff }: { diff: ArchDiff }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 dark:bg-stone-800/40 text-gray-500 dark:text-stone-400">
          <tr className="text-left">
            <th className="px-3 py-2 font-medium">Field</th>
            <th className="px-3 py-2 font-medium">{diff.model_a_name}</th>
            <th className="px-3 py-2 font-medium">{diff.model_b_name}</th>
          </tr>
        </thead>
        <tbody>
          {diff.rows.map((r) => (
            <tr key={r.field_name}
              className={cn('border-t border-gray-100 dark:border-stone-800',
                r.is_different ? 'bg-amber-50/60 dark:bg-amber-500/5' : 'hover:bg-gray-50/60 dark:hover:bg-stone-800/30')}>
              <td className="px-3 py-2 font-medium text-gray-700 dark:text-stone-300">{r.field_name}</td>
              <td className={cn('px-3 py-2 dark:text-stone-300', r.is_different && 'text-amber-700 dark:text-amber-400 font-medium')}>{r.model_a_value}</td>
              <td className={cn('px-3 py-2 dark:text-stone-300', r.is_different && 'text-amber-700 dark:text-amber-400 font-medium')}>{r.model_b_value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LatencyComparisonChart({ a, b }: { a: LatencyProfile; b: LatencyProfile }) {
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  const maxLayers = Math.max(a.decode_layer_attn_ms.length, b.decode_layer_attn_ms.length);
  const labels = Array.from({ length: maxLayers }, (_, i) => `L${i}`);
  return (
    <Plot
      data={[
        { x: labels.slice(0, a.decode_layer_attn_ms.length), y: a.decode_layer_attn_ms, name: `${a.model_name} Attn`, type: 'bar', marker: { color: '#6366f1' } },
        { x: labels.slice(0, a.decode_layer_mlp_ms.length), y: a.decode_layer_mlp_ms, name: `${a.model_name} MLP`, type: 'bar', marker: { color: '#a855f7' } },
        { x: labels.slice(0, b.decode_layer_attn_ms.length), y: b.decode_layer_attn_ms, name: `${b.model_name} Attn`, type: 'bar', marker: { color: '#f59e0b' } },
        { x: labels.slice(0, b.decode_layer_mlp_ms.length), y: b.decode_layer_mlp_ms, name: `${b.model_name} MLP`, type: 'bar', marker: { color: '#ef4444' } },
      ]}
      layout={{
        barmode: 'group',
        yaxis: { title: { text: 'Latency (ms)' }, gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        xaxis: { gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        height: 350, margin: { t: 10, l: 60, r: 20, b: 40 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center', font: { size: 10, color: dk ? '#a8a29e' : '#6b7280' } },
        font: { color: dk ? '#d6d3d1' : '#374151' },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

function PrefillDecodeBar({ a, b }: { a: LatencyProfile; b: LatencyProfile }) {
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  return (
    <Plot
      data={[
        { x: [a.model_name, b.model_name], y: [a.prefill_total_ms, b.prefill_total_ms], name: 'Prefill', type: 'bar', marker: { color: '#6366f1' } },
        { x: [a.model_name, b.model_name], y: [a.decode_total_ms * a.decode_steps, b.decode_total_ms * b.decode_steps], name: 'Decode', type: 'bar', marker: { color: '#a855f7' } },
      ]}
      layout={{
        barmode: 'stack',
        yaxis: { title: { text: 'Time (ms)' }, gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        xaxis: { tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        height: 280, margin: { t: 10, l: 60, r: 20, b: 40 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', font: { size: 10, color: dk ? '#a8a29e' : '#6b7280' } },
        font: { color: dk ? '#d6d3d1' : '#374151' },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

function BottleneckTable({ layers }: { layers: BottleneckLayer[] }) {
  if (layers.length === 0) {
    return <p className="py-4 text-center text-xs text-gray-400 dark:text-stone-500">No timing data available</p>;
  }
  return (
    <table className="w-full text-xs">
      <thead className="bg-gray-50 dark:bg-stone-800/40 text-gray-500 dark:text-stone-400">
        <tr className="text-left">
          <th className="px-2 py-1.5 font-medium">Layer</th>
          <th className="px-2 py-1.5 font-medium text-right">Attn</th>
          <th className="px-2 py-1.5 font-medium text-right">MLP</th>
          <th className="px-2 py-1.5 font-medium text-right">Total</th>
          <th className="px-2 py-1.5 font-medium text-right">% Total</th>
          <th className="px-2 py-1.5 font-medium">Type</th>
        </tr>
      </thead>
      <tbody>
        {layers.map((b) => (
          <tr key={b.layer_idx} className="border-t border-gray-50 dark:border-stone-800 hover:bg-gray-50/60 dark:hover:bg-stone-800/40">
            <td className="px-2 py-1.5 font-mono dark:text-stone-300">L{b.layer_idx}</td>
            <td className="px-2 py-1.5 text-right tabular-nums dark:text-stone-400">{b.attn_ms.toFixed(2)}</td>
            <td className="px-2 py-1.5 text-right tabular-nums dark:text-stone-400">{b.mlp_ms.toFixed(2)}</td>
            <td className="px-2 py-1.5 text-right tabular-nums font-medium dark:text-stone-200">{b.total_ms.toFixed(2)}</td>
            <td className="px-2 py-1.5 text-right tabular-nums dark:text-stone-400">{(b.pct_of_total * 100).toFixed(1)}%</td>
            <td className="px-2 py-1.5">
              <span className={cn('rounded px-1.5 py-0.5 text-xs font-medium',
                b.bottleneck_type === 'attn' ? 'bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-300'
                : b.bottleneck_type === 'mlp' ? 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300'
                : 'bg-orange-100 text-orange-700 dark:bg-orange-500/20 dark:text-orange-300')}>
                {b.bottleneck_type.toUpperCase()}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

void formatParamCount;  // kept for future use
