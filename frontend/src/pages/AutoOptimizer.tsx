// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useModelStore } from '@/stores/modelStore';
import { searchOptimizations, getDevices } from '@/api/endpoints';
import type { SearchResult, SearchCandidate, DeviceProfile } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useAutoOptInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { MetricCards } from '@/components/data/MetricCards';
import { EmptyState } from '@/components/common/EmptyState';
import { Search, Trophy, Cpu, Shield, Sparkles, Send, Square, X as XIcon } from 'lucide-react';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveAutoOptCapabilities,
  assessAutoOptConfig,
  buildAutoOptContextSnippet,
  buildAutoOptAutoBrief,
  getAutoOptSuggestedPrompts,
} from '@/lib/autoOptimizerInsights';

const BIT_OPTIONS = [3, 4, 6, 8];

export default function AutoOptimizer() {
  const model = useModelStore((s) => s.currentModel);
  const profileSummary = useModelStore((s) => s.profileSummary);
  const hasProfile = !!profileSummary;
  const t = useT();
  const autoOptInsights = useAutoOptInsights(t, model, hasProfile);

  const [devices, setDevices] = useState<DeviceProfile[]>([]);
  const [deviceName, setDeviceName] = useState('MacBook Air M5 (16GB)');
  const [qualityFloor, setQualityFloor] = useState(0.5);
  const [targetBits, setTargetBits] = useState<number[]>([4]);
  const [maxLayersRemove, setMaxLayersRemove] = useState(0);

  const [result, setResult] = useState<SearchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  useEffect(() => {
    getDevices()
      .then(setDevices)
      .catch(() => console.warn('[AutoOptimizer] Failed to load device list'));
  }, []);

  // ── §9.1 hooks must precede early returns (React Hooks rule) ─────────
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const selectedCandidate =
    selectedIdx !== null && result ? result.pareto_frontier[selectedIdx] ?? null : null;
  const autoCaps = useMemo(
    () => deriveAutoOptCapabilities(targetBits, maxLayersRemove, qualityFloor, deviceName, hasProfile, model, result, selectedCandidate),
    [targetBits, maxLayersRemove, qualityFloor, deviceName, hasProfile, model, result, selectedCandidate],
  );
  const autoRisk = useMemo(() => assessAutoOptConfig(autoCaps), [autoCaps]);
  const autoSystemPrompt = useMemo(() => {
    if (!model) return '';
    return buildModelSelfSystemPrompt(model, locale) + '\n\n' + buildAutoOptContextSnippet(autoCaps, locale);
  }, [model, autoCaps, locale]);
  const briefChat = useModelChat({
    modelId: model?.model_id || null,
    systemPrompt: autoSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });
  const autoPrompts = useMemo(() => getAutoOptSuggestedPrompts(autoCaps, locale), [autoCaps, locale]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);
  useEffect(() => {
    if (!model) return;
    if (briefChat.streaming) return;
    const sig = `${model.model_id}:${autoCaps.runPhase}:${autoCaps.paretoCount}:${autoCaps.searchSizeBucket}:${selectedCandidate?.label || ''}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildAutoOptAutoBrief(autoCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, autoCaps.runPhase, autoCaps.paretoCount, autoCaps.searchSizeBucket, selectedCandidate?.label, locale]);
  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const AUTO_RISK_BANNER_CLASS: Record<typeof autoRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to run auto optimization search" />;
  }

  if (!profileSummary) {
    return (
      <EmptyState
        title="No Activation Profile"
        description="Load an activation profile first, then return here to search the optimization space"
      />
    );
  }

  const handleSearch = async () => {
    setLoading(true);
    setError(null);
    setSelectedIdx(null);
    try {
      const data = await searchOptimizations(model.model_id, {
        device_name: deviceName,
        quality_floor: qualityFloor,
        target_bits: targetBits,
        max_layers_remove: maxLayersRemove,
      });
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  const toggleBit = (b: number) => {
    setTargetBits((prev) =>
      prev.includes(b) ? prev.filter((x) => x !== b) : [...prev, b].sort(),
    );
  };

  return (
    <div>
      <PageHeader
        title="Auto Optimizer"
        description="Search optimization parameter space for Pareto frontier"
      />

      <InsightPanel insights={autoOptInsights} />

      {/* 4-card identity strip */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Search size={16} />}
          label={t('autoOpt.cardSpace')}
          value={`~${autoCaps.searchSpaceSize}`}
          hint={`bits=[${targetBits.join(',') || '—'}] · layers≤${maxLayersRemove} · ${autoCaps.searchSizeBucket}`}
          tone={autoCaps.searchSizeBucket === 'huge' ? 'amber' : autoCaps.searchSizeBucket === 'tiny' ? 'amber' : 'indigo'}
        />
        <IdentityCard
          icon={<Cpu size={16} />}
          label={t('autoOpt.cardDevice')}
          value={deviceName}
          hint={autoCaps.runPhase === 'hasResult' ? `${autoCaps.fitsDeviceCount} / ${autoCaps.paretoCount} ${t('autoOpt.fits')}` : t('autoOpt.notRun')}
          tone={autoCaps.runPhase === 'hasResult' && autoCaps.paretoCount > 0 && autoCaps.fitsDeviceCount === 0 ? 'red' : 'indigo'}
        />
        <IdentityCard
          icon={<Trophy size={16} />}
          label={t('autoOpt.cardPareto')}
          value={autoCaps.runPhase === 'hasResult' ? `${autoCaps.paretoCount} / ${autoCaps.candidateCount}` : '—'}
          hint={autoCaps.runPhase === 'hasResult' && autoCaps.bestSpeedup
            ? `+${(autoCaps.bestSpeedup.speedup_pct ?? 0).toFixed(0)}% ${t('autoOpt.peakSpeedup')}`
            : t('autoOpt.runFirst')}
          tone={autoCaps.runPhase === 'hasResult' && autoCaps.paretoCount > 0 ? 'emerald' : autoCaps.runPhase === 'hasResult' ? 'amber' : 'neutral'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('autoOpt.cardSovereignty')}
          value={t('autoOpt.zeroCloud')}
          hint={t('autoOpt.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {autoRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${AUTO_RISK_BANNER_CLASS[autoRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {autoRisk.level === 'danger' ? t('autoOpt.riskDanger') : t('autoOpt.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? autoRisk.reasonZh : autoRisk.reason}
        </div>
      )}

      {model && (
        <ModelBriefCard
          className="mb-6"
          label={t('autoOpt.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('autoOpt.briefEmpty')}
          streamingText={t('autoOpt.briefThinking')}
          refreshTitle={t('autoOpt.briefRefire')}
          prompts={autoPrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildAutoOptAutoBrief(autoCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {/* Search controls */}
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* Device selector */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">Target device</label>
            <select
              value={deviceName}
              onChange={(e) => setDeviceName(e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            >
              {devices.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} ({d.ram_gb} GB)
                </option>
              ))}
            </select>
          </div>

          {/* Quality floor */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">
              Quality floor: {qualityFloor.toFixed(2)}
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={qualityFloor}
              onChange={(e) => setQualityFloor(Number(e.target.value))}
              className="w-full"
            />
          </div>

          {/* Target bits */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">Target bit widths</label>
            <div className="flex gap-2">
              {BIT_OPTIONS.map((b) => (
                <button
                  key={b}
                  onClick={() => toggleBit(b)}
                  className={`rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                    targetBits.includes(b)
                      ? 'bg-indigo-100 text-indigo-700'
                      : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                  }`}
                >
                  {b}-bit
                </button>
              ))}
            </div>
          </div>

          {/* Max layers remove */}
          <div>
            <label className="mb-1 block text-xs text-gray-500">Max layers to remove</label>
            <input
              type="number"
              min={0}
              max={10}
              value={maxLayersRemove}
              onChange={(e) => setMaxLayersRemove(Number(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            />
          </div>
        </div>

        <button
          onClick={handleSearch}
          disabled={loading || targetBits.length === 0}
          className="mt-4 rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          {loading ? 'Searching...' : 'Run Search'}
        </button>
        {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
      </div>

      {result && (
        <>
          {/* Summary metrics */}
          <MetricCards
            metrics={[
              { label: 'Candidates', value: result.candidates.length },
              { label: 'Pareto Frontier', value: result.pareto_frontier.length },
              { label: 'Fits Device', value: result.fits_device_count },
              {
                label: 'Search Time',
                value: `${(result.search_time_seconds * 1000).toFixed(0)}ms`,
              },
            ]}
            className="mb-6"
          />

          {/* Pareto scatter */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-700">Pareto Frontier</h3>
            <ParetoScatter
              result={result}
              qualityFloor={qualityFloor}
              onSelect={setSelectedIdx}
            />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Candidate table */}
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Candidate Rankings</h3>
              <CandidateTable
                result={result}
                selectedIdx={selectedIdx}
                onSelect={setSelectedIdx}
              />
            </div>

            {/* Candidate detail */}
            <div className="rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Candidate Detail</h3>
              {selectedCandidate ? (
                <CandidateDetail candidate={selectedCandidate} />
              ) : (
                <p className="py-8 text-center text-sm text-gray-400">
                  Select a Pareto candidate to view details
                </p>
              )}
            </div>
          </div>
        </>
      )}

      {/* Ask Model FAB — brain reads Pareto results (playbook §1-D) */}
      {!drawerOpen && (
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
        >
          <Sparkles size={14} />
          {t('autoOpt.askFab')}
          <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
            [{model.model_name}]
          </span>
        </button>
      )}
      {drawerOpen && (
        <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
          <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
              <Search size={13} className="text-indigo-500" />
              {t('autoOpt.askDrawerTitle')}
              <span className="text-[10px] font-normal text-stone-400">[{model.model_name}]</span>
            </div>
            <button
              type="button"
              onClick={() => setDrawerOpen(false)}
              className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-200"
            >
              <XIcon size={14} />
            </button>
          </div>
          <div className="max-h-[50vh] min-h-[8rem] overflow-y-auto px-3 py-3 text-sm leading-relaxed text-stone-700 dark:text-stone-200">
            {briefChat.text ? (
              <div className="whitespace-pre-wrap">{briefChat.text}</div>
            ) : (
              <p className="text-xs text-stone-400">{t('autoOpt.askDrawerHint')}</p>
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
                placeholder={t('autoOpt.askDrawerPlaceholder')}
                disabled={briefChat.streaming}
                className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
              />
              {briefChat.streaming ? (
                <button
                  type="button"
                  onClick={() => briefChat.cancel()}
                  className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                >
                  <Square size={12} /> {t('autoOpt.askDrawerStop')}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleSendBriefDrawer}
                  disabled={!drawerInput.trim()}
                  className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                >
                  <Send size={12} /> {t('autoOpt.askDrawerSend')}
                </button>
              )}
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {autoPrompts.slice(0, 4).map((p) => (
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
    </div>
  );
}

function ParetoScatter({
  result,
  qualityFloor,
  onSelect,
}: {
  result: SearchResult;
  qualityFloor: number;
  onSelect: (idx: number | null) => void;
}) {
  const nonPareto = result.candidates.filter((c) => !c.is_pareto);
  const pareto = result.pareto_frontier;

  return (
    <Plot
      data={[
        // Non-pareto candidates
        {
          x: nonPareto.map((c) => c.estimated_size_gb),
          y: nonPareto.map((c) => c.quality_proxy),
          mode: 'markers',
          type: 'scatter',
          name: 'Candidates',
          marker: {
            color: nonPareto.map((c) => (c.fits_device ? '#22c55e' : '#ef4444')),
            size: 6,
            opacity: 0.5,
          },
          hovertemplate:
            'Size: %{x:.2f} GB<br>Quality: %{y:.3f}<br>' +
            'Threshold: %{customdata[0]:.3f}<br>Bits: %{customdata[1]}<extra></extra>',
          customdata: nonPareto.map((c) => [c.threshold, c.bits]),
        },
        // Pareto frontier
        {
          x: pareto.map((c) => c.estimated_size_gb),
          y: pareto.map((c) => c.quality_proxy),
          mode: 'lines+markers',
          type: 'scatter',
          name: 'Pareto Frontier',
          marker: { color: '#f59e0b', size: 12, symbol: 'star' },
          line: { color: '#f59e0b', width: 1, dash: 'dot' },
          hovertemplate:
            'Size: %{x:.2f} GB<br>Quality: %{y:.3f}<br>' +
            'Threshold: %{customdata[0]:.3f}<br>Bits: %{customdata[1]}<extra>Pareto</extra>',
          customdata: pareto.map((c) => [c.threshold, c.bits]),
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Estimated Size (GB)' } },
        yaxis: { title: { text: 'Quality Proxy' }, range: [0, 1.05] },
        height: 350,
        margin: { t: 10, l: 60, r: 20, b: 50 },
        legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' },
        shapes: [
          // Quality floor line
          {
            type: 'line',
            x0: 0,
            x1: 1,
            xref: 'paper',
            y0: qualityFloor,
            y1: qualityFloor,
            line: { dash: 'dash', color: '#a855f7', width: 1 },
          },
          // Device limit line
          ...(result.device_max_gb > 0
            ? [
                {
                  type: 'line' as const,
                  y0: 0,
                  y1: 1,
                  yref: 'paper' as const,
                  x0: result.device_max_gb,
                  x1: result.device_max_gb,
                  line: { dash: 'dash' as const, color: '#f97316', width: 1 },
                },
              ]
            : []),
        ],
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
      onClick={(event) => {
        if (event.points?.[0]?.curveNumber === 1) {
          onSelect(event.points[0].pointIndex);
        }
      }}
    />
  );
}

function CandidateTable({
  result,
  selectedIdx,
  onSelect,
}: {
  result: SearchResult;
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
}) {
  // Show pareto first, then rest sorted by quality
  const sorted = [...result.candidates].sort((a, b) => {
    if (a.is_pareto !== b.is_pareto) return a.is_pareto ? -1 : 1;
    return b.quality_proxy - a.quality_proxy;
  });

  return (
    <div className="max-h-[400px] overflow-y-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-gray-50">
          <tr className="text-left text-gray-500">
            <th className="px-2 py-1.5 font-medium">Thr</th>
            <th className="px-2 py-1.5 font-medium">Bits</th>
            <th className="px-2 py-1.5 font-medium text-right">Layers Rm</th>
            <th className="px-2 py-1.5 font-medium text-right">Size (GB)</th>
            <th className="px-2 py-1.5 font-medium text-right">Quality</th>
            <th className="px-2 py-1.5 font-medium text-center">Fits</th>
            <th className="px-2 py-1.5 font-medium text-center">Pareto</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c, i) => {
            const paretoIdx = result.pareto_frontier.indexOf(c);
            const isSelected = paretoIdx >= 0 && paretoIdx === selectedIdx;
            return (
              <tr
                key={i}
                className={`border-t border-gray-50 cursor-pointer hover:bg-gray-50 ${
                  isSelected ? 'bg-indigo-50' : ''
                }`}
                onClick={() => {
                  if (paretoIdx >= 0) onSelect(paretoIdx);
                }}
              >
                <td className="px-2 py-1.5 font-mono">{c.threshold.toFixed(3)}</td>
                <td className="px-2 py-1.5">{c.bits}</td>
                <td className="px-2 py-1.5 text-right">{c.layers_removed}</td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {c.estimated_size_gb.toFixed(2)}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {c.quality_proxy.toFixed(3)}
                </td>
                <td className="px-2 py-1.5 text-center">
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      c.fits_device ? 'bg-green-500' : 'bg-red-500'
                    }`}
                  />
                </td>
                <td className="px-2 py-1.5 text-center">
                  {c.is_pareto && <span className="text-yellow-500">★</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CandidateDetail({ candidate: c }: { candidate: SearchCandidate }) {
  return (
    <div className="space-y-4">
      {/* Parameters */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-gray-500">Threshold</p>
          <p className="font-semibold">{c.threshold.toFixed(3)}</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-gray-500">Bits</p>
          <p className="font-semibold">{c.bits}</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-gray-500">Layers Removed</p>
          <p className="font-semibold">{c.layers_removed}</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-gray-500">Est. Size</p>
          <p className="font-semibold">{c.estimated_size_gb.toFixed(2)} GB</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-gray-500">Quality Proxy</p>
          <p className="font-semibold">{c.quality_proxy.toFixed(3)}</p>
        </div>
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-gray-500">Neuron Retention</p>
          <p className="font-semibold">{(c.neuron_retention * 100).toFixed(1)}%</p>
        </div>
      </div>

      {/* Per-layer sizes chart */}
      {c.per_layer_sizes.length > 0 && (
        <div>
          <p className="mb-2 text-xs text-gray-500">Per-Layer Intermediate Sizes</p>
          <Plot
            data={[
              {
                x: c.per_layer_sizes.map((_, i) => `L${i}`),
                y: c.per_layer_sizes,
                type: 'bar',
                marker: { color: '#22c55e' },
                hovertemplate: 'L%{x}: %{y}<extra></extra>',
              },
            ]}
            layout={{
              yaxis: { title: { text: 'Intermediate Size' } },
              height: 200,
              margin: { t: 5, l: 50, r: 10, b: 30 },
            }}
            config={{ responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      )}
    </div>
  );
}
