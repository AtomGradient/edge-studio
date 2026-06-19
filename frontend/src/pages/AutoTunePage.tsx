// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { useAutoTuneInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { startAutoTune } from '@/api/endpoints';
import { useToastStore } from '@/stores/toastStore';
import { useT, useLocaleStore } from '@/i18n';
import { Gauge, Loader2, Trophy, Activity, Shield, Sparkles, Send, Square, X as XIcon } from 'lucide-react';
import type { AutoTuneResult } from '@/api/types';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveAutoTuneCapabilities,
  assessAutoTuneConfig,
  buildAutoTuneContextSnippet,
  buildAutoTuneAutoBrief,
  getAutoTuneSuggestedPrompts,
} from '@/lib/autoTuneInsights';

export default function AutoTunePage() {
  const t = useT();
  const model = useModelStore((s) => s.currentModel);
  const addToast = useToastStore((s) => s.addToast);
  const autoTuneInsights = useAutoTuneInsights(t, model);
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [maxTokens, setMaxTokens] = useState(50);
  const [numRuns, setNumRuns] = useState(3);
  const [forceRerun, setForceRerun] = useState(false);

  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<AutoTuneResult | null>(null);

  // ── Hooks must precede early return ───────────────────────────────
  const tuneCaps = useMemo(
    () => deriveAutoTuneCapabilities(model, maxTokens, numRuns, forceRerun, result),
    [model, maxTokens, numRuns, forceRerun, result],
  );
  const tuneRisk = useMemo(() => assessAutoTuneConfig(tuneCaps), [tuneCaps]);
  const tuneSystemPrompt = useMemo(() => {
    if (!model) return '';
    return buildModelSelfSystemPrompt(model, locale) + '\n\n' + buildAutoTuneContextSnippet(tuneCaps, locale);
  }, [model, tuneCaps, locale]);
  const briefChat = useModelChat({
    modelId: model?.model_id || null,
    systemPrompt: tuneSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });
  const tunePrompts = useMemo(() => getAutoTuneSuggestedPrompts(tuneCaps, locale), [tuneCaps, locale]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);
  useEffect(() => {
    if (!model) return;
    if (briefChat.streaming) return;
    const sig = `${model.model_id}:${tuneCaps.runPhase}:${tuneCaps.bestTPSBucket}:${tuneCaps.candidateCount}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildAutoTuneAutoBrief(tuneCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, tuneCaps.runPhase, tuneCaps.bestTPSBucket, tuneCaps.candidateCount, locale]);
  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const TUNE_RISK_BANNER_CLASS: Record<typeof tuneRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  const handleStart = async () => {
    try {
      const resp = await startAutoTune({
        model_dir: model.model_dir,
        max_tokens: maxTokens,
        num_runs: numRuns,
        force_rerun: forceRerun,
      });
      setTaskId(resp.task_id);
    } catch {
      addToast('Failed to start auto-tune', 'error');
    }
  };

  const handleComplete = (raw: unknown) => {
    const r = raw as AutoTuneResult;
    setResult(r);
    if (r?.success && r.best) {
      addToast(`Best: ${r.best.tokens_per_second} tok/s${r.cached ? ' (cached)' : ''}`, 'success');
    }
  };

  return (
    <div>
      <PageHeader
        title={t('page.autoTune')}
        description={t('autoTune.desc')}
      />

      <InsightPanel insights={autoTuneInsights} />

      {/* 4-card identity strip */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Gauge size={16} />}
          label={t('autoTune.cardConfig')}
          value={`${numRuns}× ${maxTokens}t`}
          hint={`${tuneCaps.runsBucket} runs${forceRerun ? ' · force rerun' : ''}`}
          tone="indigo"
        />
        <IdentityCard
          icon={<Trophy size={16} />}
          label={t('autoTune.cardBestTPS')}
          value={tuneCaps.runPhase === 'hasResult' && tuneCaps.bestTPS > 0
            ? `${tuneCaps.bestTPS.toFixed(1)} tok/s`
            : '—'}
          hint={tuneCaps.runPhase === 'hasResult'
            ? `${tuneCaps.candidateCount} ${t('autoTune.configsTried')}${tuneCaps.isCached ? ' · cached' : ''}`
            : t('autoTune.runFirst')}
          tone={tuneCaps.bestTPSBucket === 'blazing' || tuneCaps.bestTPSBucket === 'fast'
            ? 'emerald'
            : tuneCaps.bestTPSBucket === 'ok'
              ? 'indigo'
              : tuneCaps.bestTPSBucket === 'slow'
                ? 'amber'
                : 'neutral'}
        />
        <IdentityCard
          icon={<Activity size={16} />}
          label={t('autoTune.cardVariability')}
          value={tuneCaps.runPhase === 'hasResult' ? `${(tuneCaps.variability * 100).toFixed(0)}%` : '—'}
          hint={tuneCaps.runPhase === 'hasResult'
            ? (tuneCaps.variability < 0.1 ? t('autoTune.stable') : tuneCaps.variability < 0.3 ? t('autoTune.acceptable') : t('autoTune.unstable'))
            : t('autoTune.needsRun')}
          tone={tuneCaps.runPhase === 'hasResult'
            ? (tuneCaps.variability < 0.1 ? 'emerald' : tuneCaps.variability < 0.3 ? 'indigo' : 'amber')
            : 'neutral'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('autoTune.cardSovereignty')}
          value={t('autoTune.zeroCloud')}
          hint={t('autoTune.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {tuneRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${TUNE_RISK_BANNER_CLASS[tuneRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {tuneRisk.level === 'danger' ? t('autoTune.riskDanger') : t('autoTune.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? tuneRisk.reasonZh : tuneRisk.reason}
        </div>
      )}

      {model && (
        <ModelBriefCard
          className="mb-6"
          label={t('autoTune.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('autoTune.briefEmpty')}
          streamingText={t('autoTune.briefThinking')}
          refreshTitle={t('autoTune.briefRefire')}
          prompts={tunePrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildAutoTuneAutoBrief(tuneCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Config */}
        <div className="space-y-4">
          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="mb-4 text-sm font-semibold text-gray-700 uppercase tracking-wide">Configuration</h3>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Model</label>
                <div className="rounded-lg bg-gray-50 px-3 py-2 text-sm text-gray-700">{model.model_name}</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Max Tokens</label>
                  <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)}
                    min={10} max={200} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Runs per Config</label>
                  <input type="number" value={numRuns} onChange={(e) => setNumRuns(+e.target.value)}
                    min={1} max={10} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={forceRerun} onChange={(e) => setForceRerun(e.target.checked)}
                  className="rounded border-gray-300" />
                Force re-run (ignore cache)
              </label>
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={!!taskId}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
          >
            {taskId ? <Loader2 size={16} className="animate-spin" /> : <Gauge size={16} />}
            {taskId ? 'Running...' : 'Start Auto-Tune'}
          </button>
        </div>

        {/* Right: Result */}
        <div>
          {result ? (
            <div className="space-y-4">
              {/* Best config */}
              {result.best && (
                <div className="rounded-xl border-2 border-green-200 bg-green-50 p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-green-800 uppercase tracking-wide">Best Configuration</h3>
                    {result.cached && (
                      <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded-full">cached</span>
                    )}
                  </div>
                  <div className="text-3xl font-bold text-green-700 mb-2">
                    {result.best.tokens_per_second} tok/s
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm text-green-800">
                    <p>Temperature: {result.best.temperature}</p>
                    <p>KV Cache: {result.best.kv_cache_size}</p>
                    <p>Peak Memory: {result.best.peak_memory_mb} MB</p>
                    <p>Device: {result.device_name}</p>
                  </div>
                </div>
              )}

              {/* All candidates */}
              {result.all_candidates.length > 0 && (
                <div className="rounded-xl border border-gray-200 bg-white p-5">
                  <h3 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">
                    All Configurations ({result.total_configs_tested} tested, {result.search_time_seconds}s)
                  </h3>
                  <div className="max-h-64 overflow-y-auto">
                    <table className="w-full text-xs">
                      <thead className="text-gray-500 sticky top-0 bg-white">
                        <tr>
                          <th className="text-left py-1">Temp</th>
                          <th className="text-left py-1">KV Size</th>
                          <th className="text-right py-1">tok/s</th>
                          <th className="text-right py-1">Memory MB</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.all_candidates
                          .sort((a, b) => b.tokens_per_second - a.tokens_per_second)
                          .map((c, i) => (
                          <tr key={i} className={`border-t border-gray-100 ${i === 0 ? 'bg-green-50 font-medium' : ''}`}>
                            <td className="py-1">{c.temperature}</td>
                            <td className="py-1">{c.kv_cache_size}</td>
                            <td className="text-right py-1">{c.tokens_per_second}</td>
                            <td className="text-right py-1">{c.peak_memory_mb}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {result.error && (
                <p className="text-sm text-red-600">{result.error}</p>
              )}
            </div>
          ) : (
            <EmptyState
              icon={<Gauge size={40} />}
              title="Auto-Tune Benchmark"
              description="Automatically find the optimal inference parameters for this model on your device. Results are cached for instant recall."
            />
          )}
        </div>
      </div>

      <ProgressOverlay
        taskId={taskId}
        title="Auto-Tune Benchmark"
        onComplete={handleComplete}
        onError={(err) => addToast(err, 'error')}
        onClose={() => setTaskId(null)}
      />

      {/* Ask Model FAB */}
      {!drawerOpen && (
        <button
          type="button"
          onClick={() => setDrawerOpen(true)}
          className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
        >
          <Sparkles size={14} />
          {t('autoTune.askFab')}
          <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
            [{model.model_name}]
          </span>
        </button>
      )}
      {drawerOpen && (
        <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
          <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
              <Gauge size={13} className="text-indigo-500" />
              {t('autoTune.askDrawerTitle')}
              <span className="text-[10px] font-normal text-stone-400">[{model.model_name}]</span>
            </div>
            <button type="button" onClick={() => setDrawerOpen(false)} className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-200">
              <XIcon size={14} />
            </button>
          </div>
          <div className="max-h-[50vh] min-h-[8rem] overflow-y-auto px-3 py-3 text-sm leading-relaxed text-stone-700 dark:text-stone-200">
            {briefChat.text ? (<div className="whitespace-pre-wrap">{briefChat.text}</div>) : (<p className="text-xs text-stone-400">{t('autoTune.askDrawerHint')}</p>)}
            {briefChat.streaming && <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-indigo-400" />}
          </div>
          <div className="border-t border-stone-200 p-2 dark:border-stone-700">
            <div className="flex items-center gap-1.5">
              <input type="text" value={drawerInput} onChange={(e) => setDrawerInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSendBriefDrawer(); } }}
                placeholder={t('autoTune.askDrawerPlaceholder')} disabled={briefChat.streaming}
                className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500" />
              {briefChat.streaming ? (
                <button type="button" onClick={() => briefChat.cancel()} className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600">
                  <Square size={12} /> {t('autoTune.askDrawerStop')}
                </button>
              ) : (
                <button type="button" onClick={handleSendBriefDrawer} disabled={!drawerInput.trim()} className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600">
                  <Send size={12} /> {t('autoTune.askDrawerSend')}
                </button>
              )}
            </div>
            <div className="mt-2 flex flex-wrap gap-1">
              {tunePrompts.slice(0, 4).map((p) => (
                <button key={p.label} type="button" onClick={() => { briefChat.reset(); briefChat.send(p.prompt); }} disabled={briefChat.streaming}
                  className="rounded-full border border-stone-200 bg-stone-50 px-2 py-0.5 text-[10px] font-medium text-stone-600 hover:bg-stone-100 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300 dark:hover:bg-stone-700">
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
