// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * KVCacheAnalysis — "KV cache X-ray" with hybrid-attention correction.
 *
 * Optimization layers (page-optimization-playbook §1):
 *  A. Information archaeology: backend assumes uniform full-attn → wrong for
 *     Qwen3.5 (8 FA + 24 GDN), Gemma3 (sliding window). Detect from
 *     model.config.layer_types / sliding_window and surface the correction.
 *  B. Information design: 4-card identity strip + @N context grid + DSR budget
 *     recommendations per device + GQA savings call-out.
 *  C. Visualization: keep stacked memory-curve + breakdown pie + device table,
 *     overlay corrected KV when hybrid model detected.
 *  D. Model-as-interpreter: auto AI brief explaining its own KV strategy
 *     (mentions hybrid layout / sliding window when applicable) + Ask FAB.
 */
import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useModelStore } from '@/stores/modelStore';
import { getKVReport, getDevices } from '@/api/endpoints';
import type { KVReportResponse, DeviceProfile } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { EmptyState } from '@/components/common/EmptyState';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { formatSize, cn, formatParamCount } from '@/lib/utils';
import {
  Loader2, Sparkles, X, Send, RotateCcw, AlertTriangle, ChevronDown, ChevronUp,
  Cpu, Layers, Zap, Smartphone,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt, deriveModelFacts } from '@/lib/chatPrompts';
import {
  detectAttentionLayout, buildContextCards, recommendDSRBudgets,
  buildKVContextSnippet, buildKVAutoBrief, getKVSuggestedPrompts,
  formatKVPerToken, deviceFitTone, effectiveBytesPerToken,
} from '@/lib/kvInsights';
import { IdentityCard } from '@/components/common/IdentityCard';

const DEFAULT_DEVICES = [
  'iPhone 17 Pro',
  'iPad Pro M5 (16GB)',
  'MacBook Air M5 (16GB)',
  'MacBook Pro M5 Max (48GB)',
  'Mac Studio M3 Ultra (256GB)',
];

export default function KVCacheAnalysis() {
  const model = useModelStore((s) => s.currentModel);
  const hasTrace = useModelStore((s) => s.hasTrace);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [selectedDevices, setSelectedDevices] = useState<string[]>(DEFAULT_DEVICES);
  const [allDevices, setAllDevices] = useState<DeviceProfile[] | null>(null);
  const [report, setReport] = useState<KVReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [breakdownSeqLen, setBreakdownSeqLen] = useState(4096);
  const [devicePickerOpen, setDevicePickerOpen] = useState(false);
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // Layout detection (model-only, no API needed)
  const layout = useMemo(() => (model ? detectAttentionLayout(model) : null), [model]);
  const facts = useMemo(() => (model ? deriveModelFacts(model) : null), [model]);

  // Derived insights
  const contextCards = useMemo(
    () => (report && model && layout ? buildContextCards(report, model, layout) : []),
    [report, model, layout],
  );
  const dsrRecs = useMemo(
    () => (report && layout && model ? recommendDSRBudgets(report, layout, model) : []),
    [report, layout, model],
  );

  // Effective bytes/token (backend value, falling back to model.config-derived)
  const effectiveKVBytes = useMemo(
    () => (report && model ? effectiveBytesPerToken(report, model) : 0),
    [report, model],
  );
  const usedFallback = !!report && !!model && report.bytes_per_token === 0 && effectiveKVBytes > 0;

  // Chat / brief
  const kvSystemPrompt = useMemo(() => {
    if (!model) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    if (!report || !layout) return base;
    return base + '\n\n' + buildKVContextSnippet(model, report, layout);
  }, [model, locale, report, layout]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: kvSystemPrompt,
    maxTokens: 700,
    temperature: 0.55,
  });

  const suggestedPrompts = useMemo(
    () => (model && layout ? getKVSuggestedPrompts(model, report, layout, locale) : []),
    [model, report, layout, locale],
  );

  // ── Effects ──
  useEffect(() => {
    setReport(null);
    setError(null);
    setAskOpen(false);
    briefFiredForRef.current = null;
    chat.reset();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fetch on model load
  useEffect(() => {
    if (!model) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([
      getDevices().catch(() => null),
      getKVReport(model.model_id, selectedDevices),
    ]).then(([devices, data]) => {
      if (cancelled) return;
      if (devices) setAllDevices(devices);
      setReport(data);
    }).catch((err) => {
      if (cancelled) return;
      setError(err instanceof Error ? err.message : 'Analysis failed');
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire brief (after layout known + chat ready)
  useEffect(() => {
    if (!model || !layout || !report) return;
    if (briefFiredForRef.current === model.model_id) return;
    if (chat.streaming) return;
    briefFiredForRef.current = model.model_id;
    const id = window.setTimeout(() => {
      chat.send(buildKVAutoBrief(model, layout, locale));
    }, 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, !!layout, !!report, locale]);

  // ── Callbacks ──
  const reanalyze = useCallback(async () => {
    if (!model) return;
    setLoading(true); setError(null);
    try {
      const data = await getKVReport(model.model_id, selectedDevices);
      setReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [model, selectedDevices]);

  const toggleDevice = useCallback((name: string) => {
    setSelectedDevices((prev) => prev.includes(name) ? prev.filter((d) => d !== name) : [...prev, name]);
  }, []);

  const handleSuggested = useCallback((q: string) => {
    setAskOpen(true);
    chat.send(q);
  }, [chat]);

  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  return (
    <div className="space-y-5 pb-12 relative">
      <PageHeader title={t('kv.title')} description={model.model_name} />

      {/* 4-card Identity Strip */}
      {layout && facts && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <IdentityCard
            icon={<Cpu size={16} />}
            label={t('kv.idKvPerTok')}
            value={effectiveKVBytes > 0 ? formatKVPerToken(effectiveKVBytes / layout.overestimateRatio) : '—'}
            hint={
              usedFallback
                ? `Backend report missing — computed from model.config: ${formatKVPerToken(effectiveKVBytes)} raw`
                : layout.overestimateRatio > 1
                  ? `Backend reports ${formatKVPerToken(report?.bytes_per_token || 0)} (uniform-attn assumption)`
                  : '2 × num_kv_heads × head_dim × 2 (fp16) × num_layers'
            }
            tone={layout.overestimateRatio > 1 || usedFallback ? 'amber' : 'indigo'}
          />
          <IdentityCard
            icon={<Zap size={16} />}
            label={t('kv.idGqaSaving')}
            value={facts.gqaRatio > 1 ? `−${facts.gqaSavingPct}% vs MHA` : 'MHA (no GQA)'}
            hint={`${facts.numHeads} Q heads / ${facts.numKVHeads} KV heads (${facts.gqaRatio}:1)`}
            tone={facts.gqaSavingPct >= 50 ? 'emerald' : facts.gqaRatio > 1 ? 'indigo' : 'neutral'}
          />
          <IdentityCard
            icon={<Layers size={16} />}
            label={t('kv.idAttnLayout')}
            value={layout.kind === 'uniform' ? t('kv.layoutUniform')
                  : layout.kind === 'hybrid' ? `${layout.fullAttnLayers}/${layout.totalLayers} ${t('kv.layoutHybrid')}`
                  : t('kv.layoutSliding', { n: layout.slidingWindow })}
            hint={layout.label}
            tone={layout.kind === 'hybrid' ? 'amber' : layout.kind === 'sliding' ? 'amber' : 'neutral'}
          />
          <IdentityCard
            icon={<Smartphone size={16} />}
            label={t('kv.idDeviceFit')}
            value={contextCards.length > 0
              ? `${contextCards[Math.floor(contextCards.length / 2)].fittingDeviceCount}/${(report?.device_capacities.length ?? 0)} ${t('kv.deviceFitDevices')}`
              : '—'}
            hint={`@ ${contextCards[Math.floor(contextCards.length / 2)]?.label || '—'} context, corrected for layout`}
          />
        </div>
      )}

      {/* Hybrid / sliding window correction banner */}
      {layout && layout.kind !== 'uniform' && (
        <div className={cn(
          'rounded-xl border-l-4 px-4 py-3',
          layout.kind === 'hybrid'
            ? 'border-amber-500 bg-amber-50 dark:bg-amber-500/10'
            : 'border-cyan-500 bg-cyan-50 dark:bg-cyan-500/10',
        )}>
          <div className="flex items-start gap-2.5">
            <AlertTriangle size={16} className={layout.kind === 'hybrid' ? 'text-amber-500 dark:text-amber-400' : 'text-cyan-500 dark:text-cyan-400'} />
            <div className="flex-1 text-xs">
              <div className={cn('font-semibold mb-1',
                layout.kind === 'hybrid' ? 'text-amber-800 dark:text-amber-300' : 'text-cyan-800 dark:text-cyan-300',
              )}>
                {layout.kind === 'hybrid' ? t('kv.hybridBannerTitle') : t('kv.slidingBannerTitle')}
              </div>
              <div className="text-gray-700 dark:text-stone-300 leading-relaxed">
                {layout.kind === 'hybrid'
                  ? t('kv.hybridBannerDesc', { full: layout.fullAttnLayers, total: layout.totalLayers, ratio: layout.overestimateRatio.toFixed(2) })
                  : t('kv.slidingBannerDesc', { window: layout.slidingWindow.toLocaleString() })}
              </div>
              {layout.kind === 'hybrid' && Object.keys(layout.layerTypeCounts).length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {Object.entries(layout.layerTypeCounts).map(([k, n]) => (
                    <span key={k} className="rounded-full px-2 py-0.5 text-[10px] font-medium bg-white/60 text-amber-700 dark:bg-stone-900/60 dark:text-amber-300 border border-amber-200 dark:border-amber-700/30">
                      {n}× {k}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Brief */}
      <div className="rounded-xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
              {t('kv.briefLabel')}
            </span>
            {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
          </div>
          {chat.text && !chat.streaming && layout && (
            <button
              onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildKVAutoBrief(model, layout, locale)); }}
              className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
              title={t('weights.briefRefresh')}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
        <div className="text-sm text-gray-700 dark:text-stone-300">
          {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
          {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('kv.briefPending')}</span>}
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

      {/* Loading / error */}
      {loading && (
        <div className="flex items-center justify-center py-8 text-gray-400">
          <Loader2 className="animate-spin mr-2" size={18} /> {t('kv.analyzing')}
        </div>
      )}
      {error && (
        <div className="rounded-lg bg-red-50 dark:bg-red-500/10 px-4 py-3 text-sm text-red-700 dark:text-red-400">{error}</div>
      )}

      {report && layout && (
        <>
          {/* @N Context cards grid */}
          {contextCards.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300 mb-2">{t('kv.contextCards.title')}</h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
                {contextCards.map((c) => {
                  const tone = deviceFitTone(c);
                  const dotClass = { emerald: 'bg-emerald-500', amber: 'bg-amber-500', red: 'bg-red-500' }[tone];
                  const corrected = layout.overestimateRatio > 1 || layout.kind === 'sliding';
                  return (
                    <div key={c.label} className="rounded-xl border border-gray-200 bg-white p-3 dark:border-stone-700 dark:bg-stone-900">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-600 dark:text-stone-400">{c.label}</span>
                        <span className={cn('inline-block w-2 h-2 rounded-full', dotClass)} title={`${c.fittingDeviceCount} devices fit`} />
                      </div>
                      <div className="text-lg font-semibold tabular-nums text-gray-800 dark:text-stone-200 mt-1">
                        {c.totalCorrectedMB >= 1024
                          ? `${(c.totalCorrectedMB / 1024).toFixed(1)} GB`
                          : `${c.totalCorrectedMB.toFixed(0)} MB`}
                      </div>
                      <div className="text-[10px] text-gray-400 dark:text-stone-500 mt-0.5">
                        KV: {c.kvCorrectedMB >= 1024 ? `${(c.kvCorrectedMB / 1024).toFixed(1)} GB` : `${c.kvCorrectedMB.toFixed(0)} MB`}
                        {corrected && c.totalMB > c.totalCorrectedMB && (
                          <span className="text-amber-500 ml-1" title="raw report value">(raw: {(c.totalMB / 1024 > 1 ? `${(c.totalMB / 1024).toFixed(1)} GB` : `${c.totalMB.toFixed(0)} MB`)})</span>
                        )}
                      </div>
                      <div className="text-[10px] mt-1 text-gray-500 dark:text-stone-500 truncate" title={c.smallestFitName ?? ''}>
                        {c.smallestFitName ? `📱 ${c.smallestFitName}` : '⚠️ no listed device fits'}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Memory curve */}
          <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('kv.memoryCurve.title')}</h3>
              {usedFallback && (
                <span className="text-[10px] text-amber-600 dark:text-amber-400 italic">{t('kv.fallbackChartHint')}</span>
              )}
            </div>
            <MemoryCurveChart report={report} overestimateRatio={layout.overestimateRatio} />
          </div>

          {/* Breakdown + Device Table */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('kv.breakdown.title')}</h3>
              <div className="mb-3">
                <label className="mb-1 block text-xs text-gray-500 dark:text-stone-400">
                  {t('kv.breakdown.seqLen')}: <span className="font-mono">{breakdownSeqLen.toLocaleString()}</span>
                </label>
                <input type="range" min={64} max={Math.min(facts?.maxCtx || 8192, 32768)} step={64}
                  value={breakdownSeqLen} onChange={(e) => setBreakdownSeqLen(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>
              <BreakdownPie report={report} seqLen={breakdownSeqLen} overestimateRatio={layout.overestimateRatio} />
            </div>
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('kv.deviceTable.title')}</h3>
                <button
                  onClick={() => setDevicePickerOpen((v) => !v)}
                  className="text-[11px] text-indigo-500 hover:text-indigo-700 dark:text-indigo-400 flex items-center gap-1"
                >
                  {devicePickerOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  {t('kv.deviceTable.choose')}
                </button>
              </div>
              {devicePickerOpen && (
                <div className="mb-3 max-h-32 overflow-y-auto rounded border border-gray-100 dark:border-stone-800 p-2 flex flex-wrap gap-1.5">
                  {(allDevices ?? []).map((d) => (
                    <button
                      key={d.name}
                      onClick={() => toggleDevice(d.name)}
                      className={cn(
                        'rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors',
                        selectedDevices.includes(d.name)
                          ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/15 dark:text-indigo-300'
                          : 'bg-gray-100 text-gray-500 hover:bg-gray-200 dark:bg-stone-800 dark:text-stone-400',
                      )}
                    >
                      {d.name}
                    </button>
                  ))}
                  <button
                    onClick={reanalyze}
                    className="ml-auto rounded-lg bg-indigo-500 px-2 py-0.5 text-[10px] font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
                    disabled={loading}
                  >
                    {t('kv.deviceTable.refresh')}
                  </button>
                </div>
              )}
              <DeviceTable report={report} overestimateRatio={layout.overestimateRatio} />
            </div>
          </div>

          {/* DSR Budget recommendations */}
          {dsrRecs.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('kv.dsr.title')}</h3>
                <span className="text-[10px] text-amber-600 dark:text-amber-400 italic">{t('kv.dsr.callout')}</span>
              </div>
              <table className="w-full text-xs">
                <thead className="text-gray-500 dark:text-stone-400">
                  <tr className="border-b border-gray-100 dark:border-stone-800 text-left">
                    <th className="py-1.5 font-medium">{t('kv.dsr.colDevice')}</th>
                    <th className="py-1.5 font-medium text-right">{t('kv.dsr.colAvail')}</th>
                    <th className="py-1.5 font-medium text-right">{t('kv.dsr.colBudget')}</th>
                    <th className="py-1.5 font-medium">{t('kv.dsr.colReason')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-stone-800">
                  {dsrRecs.map((r) => (
                    <tr key={r.device_name}>
                      <td className="py-1.5 font-medium text-gray-700 dark:text-stone-300">{r.device_name}</td>
                      <td className="py-1.5 text-right tabular-nums text-gray-600 dark:text-stone-400">{r.available_mb.toFixed(0)} MB</td>
                      <td className="py-1.5 text-right tabular-nums font-mono">
                        <span className={cn(
                          r.recommendedBudget <= 1024 ? 'text-red-500'
                          : r.recommendedBudget <= 4096 ? 'text-amber-500'
                          : 'text-emerald-500',
                          'font-semibold',
                        )}>
                          {r.recommendedBudget.toLocaleString()}
                        </span> tok
                      </td>
                      <td className="py-1.5 text-[10px] text-gray-500 dark:text-stone-400">{r.budgetReason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Trace growth (if available) */}
          {hasTrace && report.trace_steps.length > 0 && (
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-300">{t('kv.traceGrowth.title')}</h3>
              <p className="mb-2 text-[11px] text-gray-400 dark:text-stone-500">{t('kv.traceGrowth.subtitle')}</p>
              <TraceGrowthChart steps={report.trace_steps} />
            </div>
          )}
        </>
      )}

      {/* FAB */}
      {report && (
        <>
          {!askOpen && (
            <button
              onClick={() => setAskOpen(true)}
              className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
              title={t('kv.askModel')}
            >
              <Sparkles size={16} /> {t('kv.askModel')}
            </button>
          )}
          {askOpen && (
            <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
                <div className="flex items-center gap-2">
                  <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
                  <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('kv.askModel')}</span>
                </div>
                <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
                  <X size={14} />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 text-sm">
                {chat.text ? <MarkdownContent content={chat.text} /> : (
                  <p className="text-gray-400 dark:text-stone-500 text-xs">{t('kv.askEmpty')}</p>
                )}
                {chat.streaming && <Loader2 size={14} className="animate-spin text-indigo-500 mt-2" />}
              </div>
              <form
                onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
                className="flex gap-2 border-t border-gray-100 dark:border-stone-800 p-3"
              >
                <input
                  value={askInput}
                  onChange={(e) => setAskInput(e.target.value)}
                  placeholder={t('kv.askPlaceholder')}
                  disabled={chat.streaming}
                  className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
                />
                <button type="submit" disabled={!askInput.trim() || chat.streaming} className="rounded-lg bg-indigo-500 px-3 py-1.5 text-white hover:bg-indigo-600 disabled:opacity-50">
                  <Send size={12} />
                </button>
              </form>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ───── Charts (kept from legacy, with hybrid-correction overlay) ─────

const DSR_COLORS: Record<string, string> = {
  'DSR 2K': '#14b8a6',
  'DSR 4K': '#06b6d4',
  'DSR 4K+INT4': '#10b981',
};

function MemoryCurveChart({ report, overestimateRatio }: { report: KVReportResponse; overestimateRatio: number }) {
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  const curve = report.memory_curve;
  const seqLens = curve.map((p) => p.seq_len);
  const correctedKv = curve.map((p) => p.kv_cache_mb / overestimateRatio);

  const dsrCurves = report.dsr_curves || {};
  const dsrTraces = Object.entries(dsrCurves).map(([label, points]) => ({
    x: points.map((p) => p.seq_len),
    y: points.map((p) => p.total_mb),
    name: label,
    type: 'scatter' as const,
    mode: 'lines' as const,
    line: { color: DSR_COLORS[label] || '#14b8a6', width: 2.5, dash: 'dot' as const },
    hovertemplate: `${label}<br>Seq %{x}<br>Total: %{y:.1f} MB<extra></extra>`,
  }));

  const traces: Plotly.Data[] = [
    {
      x: seqLens, y: curve.map((p) => p.model_weights_mb),
      name: 'Model Weights', type: 'scatter', mode: 'lines', stackgroup: 'one',
      line: { color: '#6366f1' },
      hovertemplate: 'Seq %{x}<br>Weights: %{y:.1f} MB<extra></extra>',
    },
    {
      x: seqLens, y: overestimateRatio > 1 ? correctedKv : curve.map((p) => p.kv_cache_mb),
      name: overestimateRatio > 1 ? 'KV Cache (corrected)' : 'KV Cache',
      type: 'scatter', mode: 'lines', stackgroup: 'one',
      line: { color: '#a855f7' },
      hovertemplate: 'Seq %{x}<br>KV: %{y:.1f} MB<extra></extra>',
    },
    {
      x: seqLens, y: curve.map((p) => p.activation_mb),
      name: 'Activations', type: 'scatter', mode: 'lines', stackgroup: 'one',
      line: { color: '#f59e0b' },
      hovertemplate: 'Seq %{x}<br>Activation: %{y:.1f} MB<extra></extra>',
    },
    {
      x: seqLens, y: curve.map((p) => p.overhead_mb),
      name: 'Overhead', type: 'scatter', mode: 'lines', stackgroup: 'one',
      line: { color: '#ef4444' },
      hovertemplate: 'Seq %{x}<br>Overhead: %{y:.1f} MB<extra></extra>',
    },
    ...dsrTraces,
    ...report.device_capacities.map((d) => ({
      x: [seqLens[0], seqLens[seqLens.length - 1]],
      y: [d.available_mb, d.available_mb],
      name: d.device_name, type: 'scatter' as const, mode: 'lines' as const,
      line: { dash: 'dash' as const, width: 1 },
      hovertemplate: `${d.device_name}: %{y:.0f} MB<extra></extra>`,
    })),
  ];

  // Optional: dotted "raw" KV overlay so the user sees the correction
  if (overestimateRatio > 1) {
    traces.push({
      x: seqLens, y: curve.map((p) => p.kv_cache_mb),
      name: 'KV (raw report)', type: 'scatter' as const, mode: 'lines' as const,
      line: { color: '#a855f7', dash: 'dashdot' as const, width: 1 },
      opacity: 0.45,
      hovertemplate: 'Seq %{x}<br>KV raw: %{y:.1f} MB<extra></extra>',
    });
  }

  return (
    <Plot
      data={traces}
      layout={{
        xaxis: { title: { text: 'Sequence Length' }, type: 'log', gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        yaxis: { title: { text: 'Memory (MB)' }, gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        height: 400, margin: { t: 10, l: 60, r: 20, b: 50 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        legend: { orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center', font: { size: 10, color: dk ? '#a8a29e' : '#6b7280' } },
        font: { color: dk ? '#d6d3d1' : '#374151' },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

function BreakdownPie({ report, seqLen, overestimateRatio }: { report: KVReportResponse; seqLen: number; overestimateRatio: number }) {
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  const curve = report.memory_curve;
  let closest = curve[0];
  let minDist = Math.abs(curve[0].seq_len - seqLen);
  for (const p of curve) {
    const d = Math.abs(p.seq_len - seqLen);
    if (d < minDist) { minDist = d; closest = p; }
  }
  const labels = ['Model Weights', 'KV Cache', 'Activations', 'Overhead'];
  const values = [
    closest.model_weights_mb,
    closest.kv_cache_mb / overestimateRatio,
    closest.activation_mb,
    closest.overhead_mb,
  ];
  const colors = ['#6366f1', '#a855f7', '#f59e0b', '#ef4444'];
  return (
    <Plot
      data={[{
        labels, values, type: 'pie', hole: 0.4, marker: { colors },
        textinfo: 'label+percent',
        hovertemplate: '%{label}: %{value:.1f} MB (%{percent})<extra></extra>',
        textfont: { color: dk ? '#fafaf9' : '#1f2937' },
      }]}
      layout={{
        height: 280, margin: { t: 10, l: 20, r: 20, b: 20 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        showlegend: false,
        font: { color: dk ? '#d6d3d1' : '#374151' },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
}

function DeviceTable({ report, overestimateRatio }: { report: KVReportResponse; overestimateRatio: number }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead className="text-gray-500 dark:text-stone-400">
          <tr className="border-b border-gray-100 dark:border-stone-800 text-left">
            <th className="py-1.5 font-medium">Device</th>
            <th className="py-1.5 font-medium text-right">RAM</th>
            <th className="py-1.5 font-medium text-right">Avail</th>
            <th className="py-1.5 font-medium text-center">Fits</th>
            <th className="py-1.5 font-medium text-right">Max Ctx</th>
            <th className="py-1.5 font-medium text-right">KV @ Max</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100 dark:divide-stone-800">
          {report.device_capacities.map((d) => {
            // Adjust max_seq_len for hybrid if backend reported it
            const correctedMaxCtx = overestimateRatio > 1 && d.max_seq_len > 0
              ? Math.floor(d.max_seq_len * overestimateRatio)
              : d.max_seq_len;
            const correctedKvAtMax = d.kv_at_max_mb > 0 ? d.kv_at_max_mb / overestimateRatio : 0;
            return (
              <tr key={d.device_name} className="hover:bg-gray-50/60 dark:hover:bg-stone-800/40">
                <td className="py-1.5 font-medium text-gray-700 dark:text-stone-300">{d.device_name}</td>
                <td className="py-1.5 text-right text-gray-600 dark:text-stone-400">{d.ram_gb} GB</td>
                <td className="py-1.5 text-right text-gray-600 dark:text-stone-400">{formatSize(d.available_mb * 1e6)}</td>
                <td className="py-1.5 text-center">
                  <span className={cn('inline-block h-2 w-2 rounded-full', d.fits ? 'bg-emerald-500' : 'bg-red-500')} />
                </td>
                <td className="py-1.5 text-right font-mono text-gray-700 dark:text-stone-300">
                  {correctedMaxCtx > 0 ? formatParamCount(correctedMaxCtx) : '—'}
                </td>
                <td className="py-1.5 text-right text-gray-600 dark:text-stone-400">
                  {correctedKvAtMax > 0 ? `${correctedKvAtMax.toFixed(0)} MB` : '—'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TraceGrowthChart({
  steps,
}: {
  steps: Array<{ step: number; seq_len: number; kv_cache_bytes: number; token: string }>;
}) {
  const dk = typeof document !== 'undefined' && document.documentElement.classList.contains('dark');
  return (
    <Plot
      data={[{
        x: steps.map((s) => s.step),
        y: steps.map((s) => s.kv_cache_bytes / 1e6),
        type: 'scatter', mode: 'lines+markers',
        line: { color: '#a855f7', width: 2 },
        marker: { size: 4 },
        hovertemplate: 'Step %{x}<br>KV: %{y:.2f} MB<br>Token: %{text}<extra></extra>',
        text: steps.map((s) => s.token),
      }]}
      layout={{
        xaxis: { title: { text: 'Generation Step' }, gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        yaxis: { title: { text: 'KV Cache (MB)' }, gridcolor: dk ? '#292524' : '#e5e7eb', tickfont: { color: dk ? '#a8a29e' : '#6b7280' } },
        height: 250, margin: { t: 10, l: 60, r: 20, b: 40 },
        paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
        font: { color: dk ? '#d6d3d1' : '#374151' },
      }}
      config={{ responsive: true, displayModeBar: false }}
      style={{ width: '100%' }}
    />
  );
}

// Wrap chart trace types loosely (Plotly types)
import type * as Plotly from 'plotly.js';
