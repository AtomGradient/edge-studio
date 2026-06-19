// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Layers, Activity, Shield, Sparkles, Send, Square, X, BarChart3 } from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { getHeatmapData } from '@/api/endpoints';
import type { ActivationHeatmapData } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useActivationInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { MetricCards } from '@/components/data/MetricCards';
import { ChartToggle } from '@/components/charts/ChartToggle';
import { EmptyState } from '@/components/common/EmptyState';
import { formatParamCount, formatPercent } from '@/lib/utils';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveActivationCapabilities,
  assessActivation,
  buildActivationContextSnippet,
  buildActivationAutoBrief,
  getActivationSuggestedPrompts,
  deadBucketLabel,
  varianceBucketLabel,
} from '@/lib/activationInsights';

type ViewMode = '2d' | '3d-surface' | '3d-scatter';

const VIEW_OPTIONS = [
  { value: '2d', label: '2D Heatmap' },
  { value: '3d-surface', label: '3D Surface' },
  { value: '3d-scatter', label: '3D Scatter' },
];

const COMPARE_THRESHOLDS = [0.01, 0.05, 0.1, 0.5];

export default function ActivationHeatmap() {
  const model = useModelStore((s) => s.currentModel);
  const profileSummary = useModelStore((s) => s.profileSummary);
  const hasProfile = !!profileSummary;
  const t = useT();
  const activationInsights = useActivationInsights(t, model, hasProfile, profileSummary?.dead_ratio_at_01);

  const [data, setData] = useState<ActivationHeatmapData | null>(null);
  const [loading, setLoading] = useState(false);
  const [metric, setMetric] = useState<'max' | 'mean'>('max');
  const [logScale, setLogScale] = useState(true);
  const [threshold, setThreshold] = useState(0.1);
  const [viewMode, setViewMode] = useState<ViewMode>('2d');
  const [selectedLayer, setSelectedLayer] = useState(0);

  useEffect(() => {
    if (!model || !profileSummary) return;
    setLoading(true);
    getHeatmapData(model.model_id, threshold)
      .then(setData)
      .finally(() => setLoading(false));
  }, [model?.model_id, profileSummary, threshold]);

  // Compute dead neuron counts for comparison thresholds
  const deadByThreshold = useMemo(() => {
    if (!data) return [];
    const matrix = data.max_matrix;
    return COMPARE_THRESHOLDS.map((t) => {
      let dead = 0;
      let total = 0;
      for (const row of matrix) {
        for (const val of row) {
          total++;
          if (val < t) dead++;
        }
      }
      return { threshold: t, dead, total, ratio: total > 0 ? dead / total : 0 };
    });
  }, [data]);

  // ── §9.1 capability + §9.2 risk + AI brief — hooks BEFORE early returns ──
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const brain = model;
  const actCaps = useMemo(
    () => deriveActivationCapabilities(profileSummary, data, metric, threshold, selectedLayer, loading, brain),
    [profileSummary, data, metric, threshold, selectedLayer, loading, brain],
  );
  const actRisk = useMemo(() => assessActivation(actCaps), [actCaps]);

  const actSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildActivationContextSnippet(actCaps, locale);
  }, [brain, actCaps, locale]);

  const briefChat = useModelChat({
    modelId: brain?.model_id || null,
    systemPrompt: actSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const actPrompts = useMemo(
    () => getActivationSuggestedPrompts(actCaps, locale),
    [actCaps, locale],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${actCaps.runPhase}:${actCaps.deadRatioBucket}:${actCaps.varianceBucket}:${actRisk.level}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildActivationAutoBrief(actCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, actCaps.runPhase, actCaps.deadRatioBucket, actCaps.varianceBucket, actRisk.level, locale]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to view activation analysis" />;
  }

  // Note: we do NOT early-return on !profileSummary or loading anymore — the
  // identity strip + AI brief + sovereignty card render even without data so
  // the brain can narrate the gating state.

  const RISK_BANNER_CLASS: Record<typeof actRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const deadTone: 'neutral' | 'emerald' | 'amber' | 'red' =
    !actCaps.hasProfile ? 'red'
    : actCaps.deadRatioBucket === 'healthy' ? 'emerald'
    : actCaps.deadRatioBucket === 'normal' ? 'emerald'
    : actCaps.deadRatioBucket === 'sparse' ? 'amber'
    : 'red';

  const varianceTone: 'neutral' | 'emerald' | 'amber' =
    !actCaps.hasData ? 'neutral'
    : actCaps.varianceBucket === 'low' ? 'emerald'
    : actCaps.varianceBucket === 'moderate' ? 'emerald'
    : 'amber';

  // Data-dependent values — guarded for null `data`.
  const num_layers = data?.num_layers ?? 0;
  const neurons_per_layer = data?.neurons_per_layer ?? 0;
  const dead_per_layer = data?.dead_per_layer ?? [];
  const totalNeurons = num_layers * neurons_per_layer;
  const totalDead = dead_per_layer.reduce((s, d) => s + d, 0);

  const matrix = data ? (metric === 'max' ? data.max_matrix : data.mean_matrix) : [];
  const layerLabels = Array.from({ length: num_layers }, (_, i) => `L${i}`);

  // Apply log scale (only meaningful with data).
  const displayMatrix = logScale && data
    ? matrix.map((row) => row.map((v) => Math.log10(Math.max(v, 1e-10))))
    : matrix;

  // Single layer data
  const layerRow = matrix[selectedLayer] || [];
  const layerAlive = layerRow.filter((v) => v >= threshold).length;
  const layerDead = layerRow.length - layerAlive;
  const layerMax = layerRow.length > 0 ? Math.max(...layerRow, 0) : 0;
  const layerMean = layerRow.length > 0 ? layerRow.reduce((s, v) => s + v, 0) / layerRow.length : 0;

  return (
    <div>
      <PageHeader
        title="Activation Heatmap"
        description="MLP neuron activation patterns and dead neuron analysis"
        actions={
          <ChartToggle
            mode={viewMode}
            options={VIEW_OPTIONS}
            onChange={(m) => setViewMode(m as ViewMode)}
          />
        }
      />

      <InsightPanel insights={activationInsights} />

      {/* §9.1 4-card identity strip — Profile / Dead / Variance / Sovereignty */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('activation.cardProfile')}
          value={
            !actCaps.hasProfile
              ? (locale === 'zh' ? '未加载' : 'none')
              : `${actCaps.numLayers} ${locale === 'zh' ? '层' : 'layers'}`
          }
          hint={
            !actCaps.hasProfile
              ? t('activation.profileNeeded')
              : `${actCaps.neuronsPerLayer.toLocaleString()} ${locale === 'zh' ? '神经元/层' : 'per layer'}`
          }
          tone={actCaps.hasProfile ? 'emerald' : 'red'}
        />
        <IdentityCard
          icon={<Activity size={16} />}
          label={t('activation.cardDead')}
          value={actCaps.hasData ? `${(actCaps.globalDeadRatio * 100).toFixed(1)}%` : '—'}
          hint={
            actCaps.hasData
              ? `${deadBucketLabel(actCaps.deadRatioBucket, locale)} @θ=${actCaps.threshold}`
              : t('activation.deadHint')
          }
          tone={deadTone}
        />
        <IdentityCard
          icon={<BarChart3 size={16} />}
          label={t('activation.cardVariance')}
          value={
            actCaps.hasData
              ? varianceBucketLabel(actCaps.varianceBucket, locale)
              : '—'
          }
          hint={
            actCaps.hasData
              ? `${actCaps.deadRatioVariance.toFixed(4)} ${locale === 'zh' ? '方差' : 'variance'}${actCaps.deadestLayer ? ` · ${locale === 'zh' ? '最 dead' : 'most dead'} L${actCaps.deadestLayer.layerIdx}` : ''}`
              : t('activation.varianceHint')
          }
          tone={varianceTone}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('activation.cardSovereignty')}
          value={t('activation.zeroCloud')}
          hint={t('activation.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {actRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[actRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {actRisk.level === 'danger' ? t('activation.riskDanger') : t('activation.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? actRisk.reasonZh : actRisk.reason}
        </div>
      )}

      {brain && (
        <ModelBriefCard
          className="mb-6"
          label={t('activation.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('activation.briefEmpty')}
          streamingText={t('activation.briefThinking')}
          refreshTitle={t('activation.briefRefire')}
          prompts={actPrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildActivationAutoBrief(actCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {!profileSummary && (
        <EmptyState
          title={t('activation.noProfileTitle')}
          description={t('activation.noProfileDesc')}
        />
      )}

      {profileSummary && (loading || !data) && (
        <div className="flex items-center justify-center py-16 text-gray-400">
          {t('activation.loadingHeatmap')}
        </div>
      )}

      {profileSummary && data && <>

      <MetricCards
        metrics={[
          { label: 'Layers', value: num_layers },
          { label: 'Neurons / Layer', value: formatParamCount(neurons_per_layer) },
          { label: 'Total Neurons', value: formatParamCount(totalNeurons) },
          { label: `Dead @ ${threshold}`, value: formatParamCount(totalDead), subtitle: formatPercent(totalDead / totalNeurons) },
        ]}
        className="mb-6"
      />

      {/* Controls */}
      <div className="mb-6 flex flex-wrap items-center gap-4 rounded-xl border border-gray-200 bg-white px-4 py-3">
        <div>
          <label className="mb-1 block text-xs text-gray-500">Activation metric</label>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as 'max' | 'mean')}
            className="rounded-lg border border-gray-300 px-2 py-1 text-sm"
          >
            <option value="max">Max Activation</option>
            <option value="mean">Mean Activation</option>
          </select>
        </div>

        <div className="flex items-center gap-2 pt-4">
          <input
            type="checkbox"
            id="logScale"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
            className="rounded"
          />
          <label htmlFor="logScale" className="text-sm text-gray-600">Log scale</label>
        </div>

        <div className="flex-1 min-w-[200px]">
          <label className="mb-1 block text-xs text-gray-500">
            Dead neuron threshold: {threshold.toFixed(3)}
          </label>
          <input
            type="range"
            min={0.001}
            max={1.0}
            step={0.001}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>

      {/* Main visualization */}
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
        {viewMode === '2d' && (() => {
          // Subsample neurons for Plotly performance (cap at 1024 columns)
          const maxCols = 1024;
          const step2d = Math.max(1, Math.floor(neurons_per_layer / maxCols));
          const zDisplay2d = step2d > 1
            ? displayMatrix.map((row) => row.filter((_, i) => i % step2d === 0))
            : displayMatrix;
          return (
            <Plot
              data={[
                {
                  z: zDisplay2d,
                  type: 'heatmap',
                  colorscale: 'Viridis',
                  y: layerLabels,
                  x: step2d > 1
                    ? Array.from({ length: Math.ceil(neurons_per_layer / step2d) }, (_, i) => i * step2d)
                    : undefined,
                  colorbar: {
                    title: { text: logScale
                      ? `log10(${metric} activation)`
                      : `${metric} activation` },
                  },
                  hovertemplate: 'Layer %{y}<br>Neuron %{x}<br>Value: %{z:.4f}<extra></extra>',
                },
              ]}
              layout={{
                title: { text: `Neuron Activation Heatmap (${metric})${step2d > 1 ? ` [1:${step2d} sampled]` : ''}` },
                xaxis: { title: { text: 'Neuron Index' } },
                yaxis: { title: { text: 'Layer' }, autorange: 'reversed' },
                height: Math.max(400, num_layers * 20),
                margin: { t: 40, l: 60, r: 20, b: 40 },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          );
        })()}

        {viewMode === '3d-surface' && (() => {
          // Subsample neurons for performance
          const step = Math.max(1, Math.floor(neurons_per_layer / 512));
          const zDisplay = displayMatrix.map((row) => row.filter((_, i) => i % step === 0));
          const xDisplay = Array.from({ length: Math.ceil(neurons_per_layer / step) }, (_, i) => i * step);
          return (
            <Plot
              data={[
                {
                  z: zDisplay,
                  x: xDisplay,
                  y: Array.from({ length: num_layers }, (_, i) => i),
                  type: 'surface',
                  colorscale: 'Viridis',
                  colorbar: { title: { text: logScale ? `log10(${metric})` : metric } },
                  hovertemplate: 'Neuron %{x}<br>Layer %{y}<br>Value: %{z:.4f}<extra></extra>',
                } as unknown as Plotly.Data,
              ]}
              layout={{
                title: { text: '3D Neuron Activation Surface' },
                scene: {
                  xaxis: { title: { text: 'Neuron Index' } },
                  yaxis: { title: { text: 'Layer' } },
                  zaxis: { title: { text: logScale ? `log10(${metric})` : metric } },
                  camera: { eye: { x: 1.5, y: -1.5, z: 1.2 } },
                },
                height: 600,
                margin: { t: 40, l: 0, r: 0, b: 0 },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          );
        })()}

        {viewMode === '3d-scatter' && (() => {
          const step = Math.max(1, Math.floor(neurons_per_layer / 256));
          const aliveX: number[] = [], aliveY: number[] = [], aliveZ: number[] = [];
          const deadX: number[] = [], deadY: number[] = [], deadZ: number[] = [];

          for (let layer = 0; layer < num_layers; layer++) {
            for (let n = 0; n < neurons_per_layer; n += step) {
              const val = matrix[layer][n];
              if (val >= threshold) {
                aliveX.push(n); aliveY.push(layer); aliveZ.push(val);
              } else {
                deadX.push(n); deadY.push(layer); deadZ.push(val);
              }
            }
          }

          return (
            <Plot
              data={[
                ...(aliveX.length > 0 ? [{
                  x: aliveX, y: aliveY, z: aliveZ,
                  mode: 'markers' as const, name: 'Alive', type: 'scatter3d' as const,
                  marker: { size: 2, color: '#4CAF50', opacity: 0.5 },
                  hovertemplate: 'Neuron %{x}<br>Layer %{y}<br>Act: %{z:.4f}<extra></extra>',
                }] : []),
                ...(deadX.length > 0 ? [{
                  x: deadX, y: deadY, z: deadZ,
                  mode: 'markers' as const, name: 'Dead', type: 'scatter3d' as const,
                  marker: { size: 2, color: '#f44336', opacity: 0.7 },
                  hovertemplate: 'Neuron %{x}<br>Layer %{y}<br>Act: %{z:.4f}<extra></extra>',
                }] : []),
              ]}
              layout={{
                title: { text: `3D Dead vs Alive Neurons (threshold=${threshold})` },
                scene: {
                  xaxis: { title: { text: 'Neuron Index' } },
                  yaxis: { title: { text: 'Layer' } },
                  zaxis: { title: { text: 'Max Activation' } },
                  camera: { eye: { x: 1.5, y: -1.5, z: 1.0 } },
                },
                height: 600,
                margin: { t: 40, l: 0, r: 0, b: 0 },
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          );
        })()}
      </div>

      {/* Dead neurons bar chart */}
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
        <h3 className="mb-2 text-sm font-semibold text-gray-700">Dead Neurons per Layer</h3>
        <Plot
          data={[
            {
              x: layerLabels,
              y: dead_per_layer.map((d) => neurons_per_layer - d),
              name: 'Alive',
              type: 'bar',
              marker: { color: '#4CAF50' },
            },
            {
              x: layerLabels,
              y: dead_per_layer,
              name: 'Dead',
              type: 'bar',
              marker: { color: '#f44336' },
            },
          ]}
          layout={{
            barmode: 'stack',
            xaxis: { title: { text: 'Layer' } },
            yaxis: { title: { text: 'Neuron Count' } },
            height: 300,
            margin: { t: 10, l: 60, r: 20, b: 40 },
            legend: { orientation: 'h', y: 1.05 },
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Threshold comparison table */}
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-gray-700">Dead Neuron Ratio by Threshold</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="px-2 py-2 font-medium">Threshold</th>
                <th className="px-2 py-2 font-medium text-right">Dead Neurons</th>
                <th className="px-2 py-2 font-medium text-right">Ratio</th>
              </tr>
            </thead>
            <tbody>
              {deadByThreshold.map((row) => (
                <tr
                  key={row.threshold}
                  className={`border-b border-gray-50 ${row.threshold === threshold ? 'bg-indigo-50 font-medium' : ''}`}
                >
                  <td className="px-2 py-2">{row.threshold}</td>
                  <td className="px-2 py-2 text-right">{row.dead.toLocaleString()}</td>
                  <td className="px-2 py-2 text-right">{formatPercent(row.ratio)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Single layer detail */}
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Layer Detail</h3>
            <select
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(Number(e.target.value))}
              className="rounded-lg border border-gray-300 px-2 py-1 text-sm"
            >
              {layerLabels.map((label, i) => (
                <option key={i} value={i}>{label}</option>
              ))}
            </select>
          </div>

          <div className="mb-4 grid grid-cols-2 gap-3 text-sm">
            <div className="rounded-lg bg-green-50 px-3 py-2">
              <p className="text-xs text-gray-500">Alive</p>
              <p className="font-semibold text-green-700">{layerAlive.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-red-50 px-3 py-2">
              <p className="text-xs text-gray-500">Dead</p>
              <p className="font-semibold text-red-700">{layerDead.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-gray-50 px-3 py-2">
              <p className="text-xs text-gray-500">Max Activation</p>
              <p className="font-semibold">{layerMax.toFixed(4)}</p>
            </div>
            <div className="rounded-lg bg-gray-50 px-3 py-2">
              <p className="text-xs text-gray-500">Mean Activation</p>
              <p className="font-semibold">{layerMean.toFixed(4)}</p>
            </div>
          </div>

          {/* Layer activation histogram */}
          <Plot
            data={[
              {
                x: layerRow,
                type: 'histogram',
                nbinsx: 50,
                marker: { color: '#7c4dff', opacity: 0.8 },
                hovertemplate: 'Value: %{x:.4f}<br>Count: %{y}<extra></extra>',
              } as unknown as Plotly.Data,
            ]}
            layout={{
              shapes: [
                {
                  type: 'line',
                  x0: threshold, x1: threshold,
                  y0: 0, y1: 1,
                  yref: 'paper',
                  line: { color: '#f44336', width: 2, dash: 'dash' },
                },
              ],
              xaxis: { title: { text: 'Activation Value' } },
              yaxis: { title: { text: 'Count' } },
              height: 200,
              margin: { t: 10, l: 50, r: 10, b: 40 },
            }}
            config={{ responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
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
              {t('activation.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brain.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <BarChart3 size={13} className="text-indigo-500" />
                  {t('activation.askDrawerTitle')}
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
                  <p className="text-xs text-stone-400">{t('activation.askDrawerHint')}</p>
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
                    placeholder={t('activation.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('activation.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('activation.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {actPrompts.slice(0, 4).map((p) => (
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
