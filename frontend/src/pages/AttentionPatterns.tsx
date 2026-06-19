// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Layers, Activity, Shield, Sparkles, Send, Square, X, Eye } from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { analyzeAttention } from '@/api/endpoints';
import type { AttentionAnalysisResponse } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useAttentionInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { MetricCards } from '@/components/data/MetricCards';
import { EmptyState } from '@/components/common/EmptyState';
import { formatPercent } from '@/lib/utils';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveAttentionCapabilities,
  assessAttention,
  buildAttentionContextSnippet,
  buildAttentionAutoBrief,
  getAttentionSuggestedPrompts,
  shapeLabel,
} from '@/lib/attentionInsights';

const PATTERN_COLORS: Record<string, string> = {
  SINK: '#FF9800',
  LOCAL: '#2196F3',
  GLOBAL: '#4CAF50',
  SPARSE: '#9E9E9E',
};

const PRIORITY_ICONS: Record<string, string> = {
  high: 'HIGH',
  medium: 'MED',
  low: 'LOW',
};

export default function AttentionPatterns() {
  const model = useModelStore((s) => s.currentModel);
  const hasTrace = useModelStore((s) => s.hasTrace);
  const hasAttentionTrace = useModelStore((s) => s.hasAttentionTrace);
  const t = useT();
  const attentionInsights = useAttentionInsights(t, model, hasAttentionTrace);

  const [result, setResult] = useState<AttentionAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── §9.1 capability + §9.2 risk + AI brief — hooks BEFORE early returns ──
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const brain = model;
  const attCaps = useMemo(
    () => deriveAttentionCapabilities(result, loading, hasTrace, hasAttentionTrace, brain),
    [result, loading, hasTrace, hasAttentionTrace, brain],
  );
  const attRisk = useMemo(() => assessAttention(attCaps), [attCaps]);

  const attSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildAttentionContextSnippet(attCaps, locale);
  }, [brain, attCaps, locale]);

  const briefChat = useModelChat({
    modelId: brain?.model_id || null,
    systemPrompt: attSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const attPrompts = useMemo(
    () => getAttentionSuggestedPrompts(attCaps, locale),
    [attCaps, locale],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${attCaps.runPhase}:${attCaps.shape}:${attCaps.dominantPattern ?? 'none'}:${attRisk.level}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildAttentionAutoBrief(attCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, attCaps.runPhase, attCaps.shape, attCaps.dominantPattern, attRisk.level, locale]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to analyze attention patterns" />;
  }

  // Note: no longer return early on missing trace — render inline gating so
  // brain can narrate the capture step.

  const RISK_BANNER_CLASS: Record<typeof attRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeAttention(model.model_id);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const numLayers = model.num_layers;
  const numHeads = model.num_attention_heads;
  const counts = result?.pattern_counts ?? {};
  const total = numLayers * numHeads || 1;

  return (
    <div>
      <PageHeader
        title="Attention Patterns"
        description="Classify attention head patterns (SINK/LOCAL/GLOBAL/SPARSE)"
      />

      <InsightPanel insights={attentionInsights} />

      {/* §9.1 4-card identity strip — Trace / Shape / DSR fit / Sovereignty */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('attention.cardTrace')}
          value={
            attCaps.runPhase === 'noTrace'
              ? (locale === 'zh' ? '未捕获' : 'none')
              : `${numLayers}×${numHeads}`
          }
          hint={
            attCaps.runPhase === 'noTrace'
              ? t('attention.traceNeeded')
              : `${attCaps.totalHeads} ${locale === 'zh' ? 'heads 待分类' : 'heads to classify'}`
          }
          tone={attCaps.runPhase === 'noTrace' ? 'red' : 'emerald'}
        />
        <IdentityCard
          icon={<Eye size={16} />}
          label={t('attention.cardShape')}
          value={attCaps.hasResult ? shapeLabel(attCaps.shape, locale) : '—'}
          hint={
            attCaps.hasResult
              ? `${locale === 'zh' ? '主导 ' : 'dominant '}${attCaps.dominantPattern?.toUpperCase() ?? '?'} (${((attCaps.dominantPattern ? attCaps.histogram[attCaps.dominantPattern].ratio : 0) * 100).toFixed(0)}%)`
              : t('attention.shapeHint')
          }
          tone={
            !attCaps.hasResult ? 'neutral'
            : attCaps.shape === 'sparse_heavy' ? 'red'
            : attCaps.shape === 'balanced' ? 'emerald'
            : 'indigo'
          }
        />
        <IdentityCard
          icon={<Activity size={16} />}
          label={t('attention.cardDSR')}
          value={
            attCaps.hasResult
              ? `${attCaps.dsrFriendlyLayers.length}/${numLayers}`
              : '—'
          }
          hint={
            attCaps.hasResult
              ? attCaps.dsrFriendlyLayers.length === 0
                ? (locale === 'zh' ? '无 DSR 友好层' : 'no DSR-friendly layers')
                : `${locale === 'zh' ? 'SINK+LOCAL > 80%' : 'SINK+LOCAL > 80%'} · ${attCaps.globalHeavyLayers.length} GLOBAL`
              : t('attention.dsrHint')
          }
          tone={
            !attCaps.hasResult ? 'neutral'
            : attCaps.dsrFriendlyLayers.length === 0 ? 'amber'
            : attCaps.dsrFriendlyLayers.length > numLayers * 0.5 ? 'emerald'
            : 'indigo'
          }
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('attention.cardSovereignty')}
          value={t('attention.zeroCloud')}
          hint={t('attention.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {attRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[attRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {attRisk.level === 'danger' ? t('attention.riskDanger') : t('attention.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? attRisk.reasonZh : attRisk.reason}
        </div>
      )}

      {brain && (
        <ModelBriefCard
          className="mb-6"
          label={t('attention.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('attention.briefEmpty')}
          streamingText={t('attention.briefThinking')}
          refreshTitle={t('attention.briefRefire')}
          prompts={attPrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildAttentionAutoBrief(attCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {(!hasTrace || !hasAttentionTrace) && (
        <EmptyState
          title={t('attention.noTraceTitle')}
          description={t('attention.noTraceDesc')}
        />
      )}

      {hasTrace && hasAttentionTrace && (
      <div className="mb-6">
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          {loading ? 'Analyzing...' : 'Analyze Attention Patterns'}
        </button>
        {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
      </div>
      )}

      {result && (
        <>
          {/* Summary metrics */}
          <MetricCards
            metrics={[
              { label: 'SINK', value: `${counts.sink ?? 0}`, subtitle: formatPercent((counts.sink ?? 0) / total) },
              { label: 'LOCAL', value: `${counts.local ?? 0}`, subtitle: formatPercent((counts.local ?? 0) / total) },
              { label: 'GLOBAL', value: `${counts.global ?? 0}`, subtitle: formatPercent((counts.global ?? 0) / total) },
              { label: 'SPARSE', value: `${counts.sparse ?? 0}`, subtitle: formatPercent((counts.sparse ?? 0) / total) },
            ]}
            className="mb-6"
          />

          {/* Pattern Heatmap */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-700">Pattern Heatmap</h3>
            <p className="mb-3 text-xs text-gray-400">
              Each cell shows the dominant attention pattern for a (layer, head) pair.
            </p>
            <PatternHeatmap matrix={result.pattern_matrix} numLayers={numLayers} numHeads={numHeads} />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Pie chart */}
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-2 text-sm font-semibold text-gray-700">Pattern Distribution</h3>
              <PatternPie counts={counts} />
            </div>

            {/* Stacked bar */}
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-2 text-sm font-semibold text-gray-700">Per-Layer Pattern Breakdown</h3>
              <PerLayerBar summary={result.per_layer_summary} />
            </div>
          </div>

          {/* Layer detail table */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-gray-700">Layer Detail</h3>
            <div className="max-h-[400px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-50">
                  <tr className="text-left text-gray-500">
                    <th className="px-2 py-1.5 font-medium">Layer</th>
                    <th className="px-2 py-1.5 font-medium text-right">SINK</th>
                    <th className="px-2 py-1.5 font-medium text-right">LOCAL</th>
                    <th className="px-2 py-1.5 font-medium text-right">GLOBAL</th>
                    <th className="px-2 py-1.5 font-medium text-right">SPARSE</th>
                    <th className="px-2 py-1.5 font-medium">Dominant</th>
                  </tr>
                </thead>
                <tbody>
                  {result.per_layer_summary.map((row) => (
                    <tr key={row.layer} className="border-t border-gray-50 hover:bg-gray-50">
                      <td className="px-2 py-1.5 font-mono">L{row.layer}</td>
                      <td className="px-2 py-1.5 text-right">{row.sink}</td>
                      <td className="px-2 py-1.5 text-right">{row.local}</td>
                      <td className="px-2 py-1.5 text-right">{row.global}</td>
                      <td className="px-2 py-1.5 text-right">{row.sparse}</td>
                      <td className="px-2 py-1.5">
                        <span
                          className="rounded px-1.5 py-0.5 text-xs font-medium text-white"
                          style={{ backgroundColor: PATTERN_COLORS[row.dominant.toUpperCase()] || '#888' }}
                        >
                          {row.dominant.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Suggestions */}
          {result.suggestions.length > 0 && (
            <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Optimization Suggestions</h3>
              <div className="space-y-2">
                {result.suggestions.map((s, i) => (
                  <div key={i} className="rounded-lg border border-gray-100 p-3">
                    <div className="flex items-center gap-2">
                      <span className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                        s.priority === 'high' ? 'bg-red-100 text-red-700' :
                        s.priority === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-green-100 text-green-700'
                      }`}>
                        {PRIORITY_ICONS[s.priority ?? 'low']}
                      </span>
                      <span className="text-sm font-medium text-gray-800">{s.title}</span>
                    </div>
                    <p className="mt-1 text-xs text-gray-500">{s.description}</p>
                    {s.category && (
                      <p className="mt-1 text-xs text-gray-400">Category: {s.category}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Ask Model FAB drawer — must be inside main fn scope (§10.5) */}
      {brain && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('attention.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brain.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Eye size={13} className="text-indigo-500" />
                  {t('attention.askDrawerTitle')}
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
                  <p className="text-xs text-stone-400">{t('attention.askDrawerHint')}</p>
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
                    placeholder={t('attention.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('attention.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('attention.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {attPrompts.slice(0, 4).map((p) => (
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

function PatternHeatmap({
  matrix,
  numLayers,
  numHeads,
}: {
  matrix: string[][];
  numLayers: number;
  numHeads: number;
}) {
  const patternToInt: Record<string, number> = { SINK: 0, LOCAL: 1, GLOBAL: 2, SPARSE: 3 };

  const z = matrix.map((row) => row.map((p) => (patternToInt[p] ?? 3) / 3));

  const hoverText = matrix.map((row, l) =>
    row.map((p, h) => `L${l} H${h}<br>Pattern: ${p}`),
  );

  const colorscale: Array<[number, string]> = [
    [0.0, '#FF9800'], [0.25, '#FF9800'],
    [0.25, '#2196F3'], [0.5, '#2196F3'],
    [0.5, '#4CAF50'], [0.75, '#4CAF50'],
    [0.75, '#9E9E9E'], [1.0, '#9E9E9E'],
  ];

  return (
    <Plot
      data={[
        {
          z,
          x: Array.from({ length: numHeads }, (_, h) => `H${h}`),
          y: Array.from({ length: numLayers }, (_, l) => `L${l}`),
          type: 'heatmap',
          colorscale,
          showscale: false,
          hovertext: hoverText as unknown as string[],
          hoverinfo: 'text' as const,
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Head' } },
        yaxis: { title: { text: 'Layer' }, autorange: 'reversed' },
        height: Math.max(300, numLayers * 18 + 80),
        margin: { t: 10, l: 60, r: 20, b: 60 },
        annotations: [
          { x: 0.1, y: -0.12, xref: 'paper', yref: 'paper', text: '<b style="color:#FF9800">SINK</b>', showarrow: false, font: { size: 11 } },
          { x: 0.35, y: -0.12, xref: 'paper', yref: 'paper', text: '<b style="color:#2196F3">LOCAL</b>', showarrow: false, font: { size: 11 } },
          { x: 0.6, y: -0.12, xref: 'paper', yref: 'paper', text: '<b style="color:#4CAF50">GLOBAL</b>', showarrow: false, font: { size: 11 } },
          { x: 0.85, y: -0.12, xref: 'paper', yref: 'paper', text: '<b style="color:#9E9E9E">SPARSE</b>', showarrow: false, font: { size: 11 } },
        ],
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function PatternPie({ counts }: { counts: Record<string, number> }) {
  const patterns = ['sink', 'local', 'global', 'sparse'];
  const labels = patterns.filter((p) => (counts[p] ?? 0) > 0).map((p) => p.toUpperCase());
  const values = patterns.filter((p) => (counts[p] ?? 0) > 0).map((p) => counts[p]);
  const colors = labels.map((l) => PATTERN_COLORS[l]);

  return (
    <Plot
      data={[
        {
          labels,
          values,
          type: 'pie',
          hole: 0.4,
          marker: { colors },
          textinfo: 'label+percent',
          hovertemplate: '%{label}: %{value} heads (%{percent})<extra></extra>',
        },
      ]}
      layout={{
        height: 300,
        margin: { t: 10, l: 20, r: 20, b: 20 },
        showlegend: false,
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function PerLayerBar({
  summary,
}: {
  summary: AttentionAnalysisResponse['per_layer_summary'];
}) {
  const labels = summary.map((r) => `L${r.layer}`);
  const patterns = ['sink', 'local', 'global', 'sparse'] as const;

  return (
    <Plot
      data={patterns.map((p) => ({
        x: labels,
        y: summary.map((r) => r[p]),
        name: p.toUpperCase(),
        type: 'bar' as const,
        marker: { color: PATTERN_COLORS[p.toUpperCase()] },
        hovertemplate: `${p.toUpperCase()}: %{y}<extra></extra>`,
      }))}
      layout={{
        barmode: 'stack',
        yaxis: { title: { text: 'Head Count' } },
        height: 300,
        margin: { t: 10, l: 50, r: 20, b: 40 },
        legend: { orientation: 'h', y: 1.08, x: 0.5, xanchor: 'center' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}
