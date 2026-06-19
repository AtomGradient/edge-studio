// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import Plot from 'react-plotly.js';
import { Layers, Activity, Shield, Sparkles, Send, Square, X, Scissors } from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { simulatePruning, thresholdSweep } from '@/api/endpoints';
import type { PruneSimResponse, ThresholdSweepPoint } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { usePruningInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { MetricCards } from '@/components/data/MetricCards';
import { EmptyState } from '@/components/common/EmptyState';
import { formatParamCount, formatSize, formatPercent } from '@/lib/utils';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  derivePruningCapabilities,
  assessPruning,
  buildPruningContextSnippet,
  buildPruningAutoBrief,
  getPruningSuggestedPrompts,
  patternLabel as prunePatternLabel,
  retentionBucketLabel,
} from '@/lib/pruningInsights';

/** Parse protected layers string like "0-4,27" into number array */
function parseProtectedLayers(input: string): number[] {
  if (!input.trim()) return [];
  const result: number[] = [];
  for (const part of input.split(',')) {
    const trimmed = part.trim();
    if (trimmed.includes('-')) {
      const [startStr, endStr] = trimmed.split('-');
      const start = parseInt(startStr);
      const end = parseInt(endStr);
      if (!isNaN(start) && !isNaN(end)) {
        for (let i = start; i <= end; i++) result.push(i);
      }
    } else {
      const n = parseInt(trimmed);
      if (!isNaN(n)) result.push(n);
    }
  }
  return result;
}

export default function PruningSimulator() {
  const model = useModelStore((s) => s.currentModel);
  const profileSummary = useModelStore((s) => s.profileSummary);
  const hasProfile = !!profileSummary;
  const t = useT();
  const pruningInsights = usePruningInsights(t, model, hasProfile);

  // Controls
  const [threshold, setThreshold] = useState(0.1);
  const [maxReduction, setMaxReduction] = useState(0.5);
  const [minSize, setMinSize] = useState(128);
  const [protectedStr, setProtectedStr] = useState('');

  // Results
  const [result, setResult] = useState<PruneSimResponse | null>(null);
  const [sweepData, setSweepData] = useState<ThresholdSweepPoint[]>([]);
  const [loading, setLoading] = useState(false);

  const protectedLayers = parseProtectedLayers(protectedStr);

  const runSimulation = useCallback(async () => {
    if (!model || !profileSummary) return;
    setLoading(true);
    try {
      const [simResult, sweep] = await Promise.all([
        simulatePruning(model.model_id, {
          threshold,
          max_reduction: maxReduction,
          min_intermediate: minSize,
          protected_layers: protectedLayers,
        }),
        thresholdSweep(model.model_id, {
          max_reduction: maxReduction,
          min_intermediate: minSize,
          protected_layers: protectedLayers,
        }),
      ]);
      setResult(simResult);
      setSweepData(sweep);
    } finally {
      setLoading(false);
    }
  }, [model?.model_id, profileSummary, threshold, maxReduction, minSize, protectedStr]);

  // Run simulation when controls change (debounced via useEffect)
  useEffect(() => {
    if (!model || !profileSummary) return;
    const timer = setTimeout(runSimulation, 300);
    return () => clearTimeout(timer);
  }, [runSimulation, model, profileSummary]);

  // ── §9.1 capability + §9.2 risk + AI brief — hooks BEFORE early returns ──
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const brain = model;
  const pruneCaps = useMemo(
    () => derivePruningCapabilities(threshold, maxReduction, minSize, protectedLayers, profileSummary, result, sweepData, brain),
    [threshold, maxReduction, minSize, protectedLayers, profileSummary, result, sweepData, brain],
  );
  const pruneRisk = useMemo(() => assessPruning(pruneCaps), [pruneCaps]);

  const pruneSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildPruningContextSnippet(pruneCaps, locale);
  }, [brain, pruneCaps, locale]);

  const briefChat = useModelChat({
    modelId: brain?.model_id || null,
    systemPrompt: pruneSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const prunePrompts = useMemo(
    () => getPruningSuggestedPrompts(pruneCaps, locale),
    [pruneCaps, locale],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${pruneCaps.runPhase}:${pruneCaps.pattern}:${pruneCaps.globalRetention.toFixed(3)}:${pruneRisk.level}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildPruningAutoBrief(pruneCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, pruneCaps.runPhase, pruneCaps.pattern, pruneCaps.globalRetention, pruneRisk.level, locale]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to simulate pruning" />;
  }

  // Note: we no longer return early on !profileSummary — instead we render a
  // "no profile" state inside the main layout so the brain can still narrate
  // the gating + recommend the next step.

  const RISK_BANNER_CLASS: Record<typeof pruneRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const retTone: 'neutral' | 'emerald' | 'amber' | 'red' =
    !pruneCaps.hasResult ? 'neutral'
    : pruneCaps.retentionBucket === 'extreme' ? 'red'
    : pruneCaps.retentionBucket === 'aggressive' ? 'amber'
    : pruneCaps.retentionBucket === 'moderate' ? 'emerald'
    : 'emerald';

  const patternTone: 'neutral' | 'emerald' | 'amber' | 'indigo' =
    !pruneCaps.hasResult ? 'neutral'
    : pruneCaps.pattern === 'edges_protected' ? 'emerald'
    : pruneCaps.pattern === 'cliff' ? 'amber'
    : pruneCaps.pattern === 'uniform' ? 'indigo'
    : 'indigo';

  const layerLabels = result ? result.layers.map((l) => `L${l.layer_idx}`) : [];

  return (
    <div>
      <PageHeader
        title="Pruning Simulator"
        description="Interactive neuron pruning threshold simulator"
      />

      <InsightPanel insights={pruningInsights} />

      {/* §9.1 4-card identity strip — Profile / Retention / Pattern / Sovereignty */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('pruning.cardProfile')}
          value={
            !pruneCaps.hasProfile
              ? (locale === 'zh' ? '未加载' : 'none')
              : `${pruneCaps.numLayers} ${locale === 'zh' ? '层' : 'layers'}`
          }
          hint={
            !pruneCaps.hasProfile
              ? t('pruning.profileNeeded')
              : `${(pruneCaps.deadRatio * 100).toFixed(1)}% ${locale === 'zh' ? '失活 @θ=0.1' : 'dead @θ=0.1'}`
          }
          tone={pruneCaps.hasProfile ? 'emerald' : 'red'}
        />
        <IdentityCard
          icon={<Activity size={16} />}
          label={t('pruning.cardRetention')}
          value={pruneCaps.hasResult ? `${(pruneCaps.globalRetention * 100).toFixed(1)}%` : '—'}
          hint={
            pruneCaps.hasResult
              ? `${retentionBucketLabel(pruneCaps.retentionBucket, locale)} · ${locale === 'zh' ? '省 ' : 'saved '}${(pruneCaps.savingsRatio * 100).toFixed(1)}%`
              : t('pruning.retentionHint')
          }
          tone={retTone}
        />
        <IdentityCard
          icon={<Scissors size={16} />}
          label={t('pruning.cardPattern')}
          value={pruneCaps.hasResult ? prunePatternLabel(pruneCaps.pattern, locale) : '—'}
          hint={
            pruneCaps.hasResult
              ? `${pruneCaps.protectedCount > 0 ? `${pruneCaps.protectedCount} ${locale === 'zh' ? '保护层' : 'protected'} · ` : ''}${pruneCaps.cappedLayerCount > 0 ? `${pruneCaps.cappedLayerCount} ${locale === 'zh' ? '撞上限' : 'capped'}` : (locale === 'zh' ? '无层撞上限' : 'no caps hit')}`
              : t('pruning.patternHint')
          }
          tone={patternTone}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('pruning.cardSovereignty')}
          value={t('pruning.zeroCloud')}
          hint={t('pruning.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {pruneRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[pruneRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {pruneRisk.level === 'danger' ? t('pruning.riskDanger') : t('pruning.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? pruneRisk.reasonZh : pruneRisk.reason}
        </div>
      )}

      {brain && (
        <ModelBriefCard
          className="mb-6"
          label={t('pruning.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('pruning.briefEmpty')}
          streamingText={t('pruning.briefThinking')}
          refreshTitle={t('pruning.briefRefire')}
          prompts={prunePrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildPruningAutoBrief(pruneCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {!profileSummary && (
        <EmptyState
          title={t('pruning.noProfileTitle')}
          description={t('pruning.noProfileDesc')}
        />
      )}

      {profileSummary && <>

      {/* Controls */}
      <div className="mb-6 rounded-xl border border-gray-200 bg-white px-4 py-4">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div>
            <label className="mb-1 block text-xs text-gray-500">
              Activation threshold: {threshold.toFixed(3)}
            </label>
            <input
              type="range"
              min={0.001}
              max={2.0}
              step={0.001}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs text-gray-500">
              Max reduction per layer: {(maxReduction * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={maxReduction}
              onChange={(e) => setMaxReduction(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs text-gray-500">Min intermediate size</label>
            <input
              type="number"
              min={64}
              max={profileSummary.intermediate_size}
              value={minSize}
              onChange={(e) => setMinSize(Number(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs text-gray-500">Protected layers</label>
            <input
              type="text"
              value={protectedStr}
              onChange={(e) => setProtectedStr(e.target.value)}
              placeholder="e.g. 0-4,27"
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            />
          </div>
        </div>
      </div>

      {loading && !result && (
        <div className="flex items-center justify-center py-16 text-gray-400">Running simulation...</div>
      )}

      {result && (
        <>
          {/* Summary metrics */}
          <MetricCards
            metrics={[
              { label: 'Neurons Removed', value: formatParamCount(result.total_removed) },
              { label: 'Retention', value: formatPercent(result.retention) },
              { label: 'MLP Size Saved', value: formatSize(result.mlp_size_saved_bytes) },
              { label: 'MLP Params Saved', value: formatParamCount(result.mlp_params_saved) },
            ]}
            className="mb-6"
          />

          {/* Per-layer bar chart */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-700">Per-Layer Intermediate Size</h3>
            <Plot
              data={[
                {
                  x: layerLabels,
                  y: result.layers.map((l) => l.original_size),
                  name: 'Original',
                  type: 'bar',
                  marker: { color: '#ccc', opacity: 0.5 },
                },
                {
                  x: layerLabels,
                  y: result.layers.map((l) => l.aligned_size),
                  name: 'After Pruning',
                  type: 'bar',
                  marker: { color: '#4CAF50' },
                },
              ]}
              layout={{
                barmode: 'overlay',
                xaxis: { title: { text: 'Layer' } },
                yaxis: { title: { text: 'Intermediate Size' } },
                height: 350,
                margin: { t: 10, l: 60, r: 20, b: 40 },
                legend: { orientation: 'h', y: 1.05 },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Per-layer detail table */}
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Per-Layer Detail</h3>
              <div className="max-h-[400px] overflow-y-auto">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-gray-50">
                    <tr className="text-left text-gray-500">
                      <th className="px-2 py-1.5 font-medium">Layer</th>
                      <th className="px-2 py-1.5 font-medium text-right">Original</th>
                      <th className="px-2 py-1.5 font-medium text-right">Alive</th>
                      <th className="px-2 py-1.5 font-medium text-right">Aligned</th>
                      <th className="px-2 py-1.5 font-medium text-right">Removed</th>
                      <th className="px-2 py-1.5 font-medium text-right">Retention</th>
                      <th className="px-2 py-1.5 font-medium">Prot.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.layers.map((l) => (
                      <tr key={l.layer_idx} className="border-t border-gray-50 hover:bg-gray-50">
                        <td className="px-2 py-1.5 font-mono">L{l.layer_idx}</td>
                        <td className="px-2 py-1.5 text-right">{l.original_size.toLocaleString()}</td>
                        <td className="px-2 py-1.5 text-right text-green-600">{l.alive_count.toLocaleString()}</td>
                        <td className="px-2 py-1.5 text-right">{l.aligned_size.toLocaleString()}</td>
                        <td className="px-2 py-1.5 text-right text-red-600">{l.removed.toLocaleString()}</td>
                        <td className="px-2 py-1.5 text-right">{formatPercent(l.retention)}</td>
                        <td className="px-2 py-1.5">{l.is_protected ? 'Yes' : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Threshold sweep chart */}
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Threshold Sweep</h3>
              {sweepData.length > 0 ? (
                <Plot
                  data={[
                    {
                      x: sweepData.map((d) => d.threshold),
                      y: sweepData.map((d) => d.retention * 100),
                      name: 'Retention (%)',
                      type: 'scatter',
                      mode: 'lines+markers',
                      line: { color: '#4CAF50', width: 2 },
                      marker: { size: 6 },
                      yaxis: 'y',
                    },
                    {
                      x: sweepData.map((d) => d.threshold),
                      y: sweepData.map((d) => d.mlp_size_saved_mb),
                      name: 'Size Saved (MB)',
                      type: 'scatter',
                      mode: 'lines+markers',
                      line: { color: '#f44336', width: 2 },
                      marker: { size: 6 },
                      yaxis: 'y2',
                    },
                  ]}
                  layout={{
                    xaxis: { title: { text: 'Threshold' } },
                    yaxis: { title: { text: 'Retention (%)' }, side: 'left', rangemode: 'tozero' },
                    yaxis2: {
                      title: { text: 'Size Saved (MB)' },
                      overlaying: 'y',
                      side: 'right',
                      rangemode: 'tozero',
                    },
                    shapes: [
                      {
                        type: 'line',
                        x0: threshold, x1: threshold,
                        y0: 0, y1: 1,
                        yref: 'paper',
                        line: { color: '#9c27b0', width: 2, dash: 'dash' },
                      },
                    ],
                    height: 350,
                    margin: { t: 10, l: 50, r: 50, b: 40 },
                    legend: { orientation: 'h', y: 1.1 },
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              ) : (
                <p className="py-8 text-center text-sm text-gray-400">Loading sweep data...</p>
              )}
            </div>
          </div>

          {/* Config preview */}
          <div className="mt-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-gray-700">Config Preview</h3>
            <p className="mb-2 text-xs text-gray-500">
              per_layer_intermediate_sizes to write into config.json:
            </p>
            <div className="max-h-40 overflow-y-auto rounded-lg bg-gray-50 p-3">
              <pre className="text-xs text-gray-700">
                {JSON.stringify(result.config_preview, null, 2)}
              </pre>
            </div>
          </div>
        </>
      )}
      </>}

      {/* Ask Model FAB drawer */}
      {brain && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('pruning.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brain.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Scissors size={13} className="text-indigo-500" />
                  {t('pruning.askDrawerTitle')}
                  <span className="text-[10px] font-normal text-stone-400">[{brain.model_name}]</span>
                </div>
                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-200"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="max-h-[50vh] min-h-[8rem] overflow-y-auto px-3 py-3 text-sm leading-relaxed text-stone-700 dark:text-stone-200">
                {briefChat.text ? (
                  <div className="whitespace-pre-wrap">{briefChat.text}</div>
                ) : (
                  <p className="text-xs text-stone-400">{t('pruning.askDrawerHint')}</p>
                )}
                {briefChat.streaming && <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-indigo-400" />}
              </div>
              <div className="border-t border-stone-200 p-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={drawerInput}
                    onChange={(e) => setDrawerInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSendBriefDrawer(); } }}
                    placeholder={t('pruning.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('pruning.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('pruning.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {prunePrompts.slice(0, 4).map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => { briefChat.reset(); briefChat.send(p.prompt); }}
                      disabled={briefChat.streaming}
                      className="rounded-full border border-stone-200 bg-stone-50 px-2 py-0.5 text-[10px] font-medium text-stone-600 hover:bg-stone-100 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300 dark:hover:bg-stone-700"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
