// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * MixedPrecisionPanel — per-layer bit-width selection for quantization.
 * Shows a table where each layer's quantization bits can be individually set.
 */

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { Loader2, Play, Layers, Shield, Sparkles, Send, Square, X, Activity } from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useMixedPrecisionInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { cn, formatSize } from '@/lib/utils';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveMixedPrecisionCapabilities,
  assessMixedPrecision,
  buildMixedPrecisionContextSnippet,
  buildMixedPrecisionAutoBrief,
  getMixedPrecisionSuggestedPrompts,
  patternLabel,
  avgBitsBucketLabel,
  type MixedPrecisionResult,
} from '@/lib/mixedPrecisionInsights';
import axios from 'axios';

interface LayerConfig {
  layer_idx: number;
  bits: number;
  group_size: number;
}

const BIT_OPTIONS = [2, 3, 4, 8] as const;

const BIT_COLORS: Record<number, string> = {
  2: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
  3: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400',
  4: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
  8: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400',
};

export default function MixedPrecisionPanel() {
  const t = useT();
  const model = useModelStore((s) => s.currentModel);
  const mixedPrecisionInsights = useMixedPrecisionInsights(t, model);

  const numLayers = Number(model?.config?.num_hidden_layers || model?.config?.num_layers || 32);
  const modelSizeBytes = model?.total_size_bytes || 0;

  const [layerConfigs, setLayerConfigs] = useState<LayerConfig[]>(() =>
    Array.from({ length: numLayers }, (_, i) => ({
      layer_idx: i,
      bits: 4,
      group_size: 64,
    }))
  );
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<MixedPrecisionResult | null>(null);

  // Estimate size based on bit selections
  const baselineBits = Number((model?.config?.quantization as { bits?: number } | undefined)?.bits || 4);
  const estimatedSize = useMemo(() => {
    if (!modelSizeBytes) return 0;
    const avgBits = layerConfigs.reduce((sum, lc) => sum + lc.bits, 0) / layerConfigs.length;
    return modelSizeBytes * (avgBits / baselineBits);
  }, [layerConfigs, modelSizeBytes, baselineBits]);

  const setBitsForAll = (bits: number) => {
    setLayerConfigs((prev) => prev.map((lc) => ({ ...lc, bits })));
  };

  const setBitsForLayer = (idx: number, bits: number) => {
    setLayerConfigs((prev) =>
      prev.map((lc) => (lc.layer_idx === idx ? { ...lc, bits } : lc))
    );
  };

  const handleRun = async () => {
    if (!model) return;
    setRunning(true);
    setResult(null);
    try {
      const res = await axios.post(`/api/model/${model.model_id}/quantize-mixed`, {
        layer_configs: layerConfigs,
      });
      const taskId = res.data.task_id;
      let done = false;
      while (!done) {
        await new Promise((r) => setTimeout(r, 1000));
        const statusRes = await axios.get(`/api/task/${taskId}`);
        if (statusRes.data.status === 'complete') {
          done = true;
          setResult(statusRes.data.result as MixedPrecisionResult);
        } else if (statusRes.data.status === 'error') {
          throw new Error(statusRes.data.error || 'Quantization failed');
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed';
      setResult({ success: false, error: msg });
    } finally {
      setRunning(false);
    }
  };

  // Bit distribution summary
  const distribution = useMemo(() => {
    const counts: Record<number, number> = {};
    for (const lc of layerConfigs) {
      counts[lc.bits] = (counts[lc.bits] || 0) + 1;
    }
    return counts;
  }, [layerConfigs]);

  // ── §9.1 capability + §9.2 risk + AI brief — hooks BEFORE early return ──
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const brain = model;
  const mixCaps = useMemo(
    () => deriveMixedPrecisionCapabilities(layerConfigs, modelSizeBytes, baselineBits, brain, result, running),
    [layerConfigs, modelSizeBytes, baselineBits, brain, result, running],
  );
  const mixRisk = useMemo(() => assessMixedPrecision(mixCaps), [mixCaps]);

  const mixSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildMixedPrecisionContextSnippet(mixCaps, locale);
  }, [brain, mixCaps, locale]);

  const briefChat = useModelChat({
    modelId: brain?.model_id || null,
    systemPrompt: mixSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const mixPrompts = useMemo(
    () => getMixedPrecisionSuggestedPrompts(mixCaps, locale),
    [mixCaps, locale],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${mixCaps.runPhase}:${mixCaps.pattern}:${mixCaps.avgBits.toFixed(2)}:${mixRisk.level}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildMixedPrecisionAutoBrief(mixCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, mixCaps.runPhase, mixCaps.pattern, mixCaps.avgBits, mixRisk.level, locale]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  if (!model) {
    return (
      <div className="p-8 text-center text-gray-400 dark:text-gray-500">
        {t('common.noModel')}
      </div>
    );
  }

  const RISK_BANNER_CLASS: Record<typeof mixRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const avgTone: 'neutral' | 'emerald' | 'amber' | 'red' =
    mixCaps.avgBitsBucket === 'aggressive' ? 'red'
    : mixCaps.avgBitsBucket === 'compressed' ? 'amber'
    : mixCaps.avgBitsBucket === 'balanced' ? 'emerald'
    : 'neutral';

  const patternTone: 'neutral' | 'emerald' | 'amber' | 'indigo' =
    mixCaps.pattern === 'sandwich' ? 'emerald'
    : mixCaps.pattern === 'inverted' ? 'amber'
    : mixCaps.pattern === 'uniform' ? 'amber'
    : 'indigo';

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title={t('pro.mixedPrecision.title')}
        description={t('pro.mixedPrecision.desc')}
      />

      <InsightPanel insights={mixedPrecisionInsights} />

      {/* §9.1 4-card identity strip — Layers / Avg / Pattern / Sovereignty */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('pro.mixedPrecision.cardLayers')}
          value={`${mixCaps.numLayers}`}
          hint={
            mixCaps.bitsInUse.length === 0
              ? '—'
              : mixCaps.bitsInUse.map((b) => `${b}-bit×${mixCaps.bitHistogram[b]}`).join(' · ')
          }
          tone={mixCaps.bits2Count > 0 ? 'red' : mixCaps.bits3Count > mixCaps.numLayers * 0.4 ? 'amber' : 'indigo'}
        />
        <IdentityCard
          icon={<Activity size={16} />}
          label={t('pro.mixedPrecision.cardAvg')}
          value={`${mixCaps.avgBits.toFixed(2)}-bit`}
          hint={`${avgBitsBucketLabel(mixCaps.avgBitsBucket, locale)} · ${(mixCaps.savingsRatio * 100).toFixed(1)}% ${locale === 'zh' ? '估省' : 'saved'}`}
          tone={avgTone}
        />
        <IdentityCard
          icon={<Activity size={16} />}
          label={t('pro.mixedPrecision.cardPattern')}
          value={patternLabel(mixCaps.pattern, locale)}
          hint={
            mixCaps.runPhase === 'success'
              ? `${locale === 'zh' ? '已生成 ' : 'output '}${mixCaps.result?.output_dir?.split('/').pop() ?? ''}`
              : mixCaps.runPhase === 'failed'
                ? `${locale === 'zh' ? '失败' : 'failed'}: ${mixCaps.result?.error ?? '?'}`
                : mixCaps.runPhase === 'running'
                  ? (locale === 'zh' ? '运行中…' : 'running…')
                  : t('pro.mixedPrecision.patternHint')
          }
          tone={patternTone}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('pro.mixedPrecision.cardSovereignty')}
          value={t('pro.mixedPrecision.zeroCloud')}
          hint={t('pro.mixedPrecision.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {mixRisk.level !== 'safe' && mixCaps.numLayers > 0 && (
        <div className={`rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[mixRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {mixRisk.level === 'danger' ? t('pro.mixedPrecision.riskDanger') : t('pro.mixedPrecision.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? mixRisk.reasonZh : mixRisk.reason}
        </div>
      )}

      {brain && (
        <ModelBriefCard
          label={t('pro.mixedPrecision.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('pro.mixedPrecision.briefEmpty')}
          streamingText={t('pro.mixedPrecision.briefThinking')}
          refreshTitle={t('pro.mixedPrecision.briefRefire')}
          prompts={mixPrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildMixedPrecisionAutoBrief(mixCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {/* Summary bar */}
      <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-gray-900">
        <div>
          <span className="text-xs text-gray-500 dark:text-gray-400">{t('pro.mixedPrecision.layers')}</span>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{numLayers}</p>
        </div>
        <div className="h-8 w-px bg-gray-200 dark:bg-gray-700" />
        <div>
          <span className="text-xs text-gray-500 dark:text-gray-400">{t('pro.mixedPrecision.estimatedSize')}</span>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">{formatSize(estimatedSize)}</p>
        </div>
        <div className="h-8 w-px bg-gray-200 dark:bg-gray-700" />
        <div className="flex gap-2">
          {Object.entries(distribution).sort(([a], [b]) => Number(a) - Number(b)).map(([bits, count]) => (
            <span key={bits} className={cn('rounded-full px-2.5 py-0.5 text-xs font-medium', BIT_COLORS[Number(bits)])}>
              {bits}-bit: {count}
            </span>
          ))}
        </div>
        <div className="ml-auto flex gap-2">
          {BIT_OPTIONS.map((bits) => (
            <button
              key={bits}
              type="button"
              onClick={() => setBitsForAll(bits)}
              className="rounded-lg border border-gray-200 px-3 py-1 text-xs font-medium text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
            >
              All {bits}-bit
            </button>
          ))}
        </div>
      </div>

      {/* Layer table */}
      <div className="overflow-hidden rounded-2xl border border-gray-200 dark:border-gray-800">
        <div className="max-h-[480px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-2.5 text-left font-medium text-gray-600 dark:text-gray-400">Layer</th>
                <th className="px-4 py-2.5 text-center font-medium text-gray-600 dark:text-gray-400">Bits</th>
                <th className="px-4 py-2.5 text-center font-medium text-gray-600 dark:text-gray-400">Visual</th>
              </tr>
            </thead>
            <tbody>
              {layerConfigs.map((lc) => (
                <tr key={lc.layer_idx} className="border-t border-gray-100 dark:border-gray-800/60">
                  <td className="px-4 py-1.5 font-mono text-xs text-gray-700 dark:text-gray-300">
                    Layer {lc.layer_idx}
                  </td>
                  <td className="px-4 py-1.5 text-center">
                    <div className="inline-flex gap-1">
                      {BIT_OPTIONS.map((bits) => (
                        <button
                          key={bits}
                          type="button"
                          onClick={() => setBitsForLayer(lc.layer_idx, bits)}
                          className={cn(
                            'h-7 w-8 rounded text-xs font-medium transition-all duration-150',
                            lc.bits === bits
                              ? BIT_COLORS[bits]
                              : 'bg-gray-100 text-gray-500 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700',
                          )}
                        >
                          {bits}
                        </button>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-1.5">
                    <div className="h-3 w-full overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
                      <div
                        className={cn(
                          'h-full rounded-full transition-all duration-200',
                          lc.bits <= 2 ? 'bg-red-400' :
                          lc.bits <= 3 ? 'bg-amber-400' :
                          lc.bits <= 4 ? 'bg-blue-400' : 'bg-green-400',
                        )}
                        style={{ width: `${(lc.bits / 8) * 100}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={handleRun}
          disabled={running || mixRisk.level === 'danger'}
          className="flex items-center gap-2 rounded-xl bg-gray-900 px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:bg-gray-800 disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-200"
        >
          {running ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
          {running ? t('common.loading') : t('pro.mixedPrecision.run')}
        </button>
        {mixRisk.level === 'danger' && (
          <span className="text-xs text-red-600 dark:text-red-400">
            {t('pro.mixedPrecision.runBlocked')}
          </span>
        )}
      </div>

      {/* Result */}
      {result && (
        <div className={cn(
          'rounded-2xl border p-4',
          result.success
            ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20'
            : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20',
        )}>
          {result.success ? (
            <div>
              <p className="text-sm font-medium text-green-800 dark:text-green-300">
                {t('pro.mixedPrecision.success')}
              </p>
              <p className="mt-1 text-xs text-green-600 dark:text-green-400">
                Output: {result.output_dir} ({result.duration_seconds}s)
              </p>
            </div>
          ) : (
            <p className="text-sm text-red-700 dark:text-red-400">{result.error}</p>
          )}
        </div>
      )}

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
              {t('pro.mixedPrecision.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brain.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Layers size={13} className="text-indigo-500" />
                  {t('pro.mixedPrecision.askDrawerTitle')}
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
                  <p className="text-xs text-stone-400">{t('pro.mixedPrecision.askDrawerHint')}</p>
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
                    placeholder={t('pro.mixedPrecision.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('pro.mixedPrecision.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('pro.mixedPrecision.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {mixPrompts.slice(0, 4).map((p) => (
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
