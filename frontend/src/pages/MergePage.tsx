// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { useMergeInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { startMerge } from '@/api/endpoints';
import { useModelStore } from '@/stores/modelStore';
import { useToastStore } from '@/stores/toastStore';
import { useT, useLocaleStore } from '@/i18n';
import { FolderOpen, GitMerge, Loader2, Plus, X, Layers, Scale, Shield, Sparkles, Send, Square } from 'lucide-react';
import { FileBrowser } from '@/components/model/FileBrowser';
import { formatSize, formatParamCount } from '@/lib/utils';
import type { MergeResult } from '@/api/types';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveMergeCapabilities,
  assessMergeConfig,
  buildMergeContextSnippet,
  buildMergeAutoBrief,
  getMergeSuggestedPrompts,
  strategyLabel as mergeStrategyLabel,
} from '@/lib/mergeInsights';

type Strategy = 'linear' | 'slerp' | 'ties' | 'task_arithmetic';

const STRATEGIES: { value: Strategy; label: string; desc: string }[] = [
  { value: 'linear', label: 'Linear', desc: 'Weighted average of parameters' },
  { value: 'slerp', label: 'SLERP', desc: 'Spherical interpolation (2 models)' },
  { value: 'ties', label: 'TIES', desc: 'Sparse + sign consensus (multi-model)' },
  { value: 'task_arithmetic', label: 'Task Arithmetic', desc: 'Base + task vectors' },
];

export default function MergePage() {
  const t = useT();
  const addToast = useToastStore((s) => s.addToast);
  const mergeInsights = useMergeInsights(t);

  const [modelDirs, setModelDirs] = useState<string[]>(['', '']);
  const [strategy, setStrategy] = useState<Strategy>('linear');
  const [weights, setWeights] = useState<string>('');
  const [baseModelDir, setBaseModelDir] = useState('');
  const [density, setDensity] = useState(0.5);

  const [pickingIdx, setPickingIdx] = useState<number | null>(null);
  const [pickingBase, setPickingBase] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<MergeResult | null>(null);

  const validModels = modelDirs.filter(d => d.trim());
  const canStart = validModels.length >= 2 &&
    (strategy !== 'task_arithmetic' || baseModelDir) &&
    (strategy !== 'slerp' || validModels.length === 2);

  // ── §9.1 multi-component (variable N) + risk + AI brief ────────────────
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const brain = useModelStore((s) => s.currentModel);
  const mergeCaps = useMemo(
    () => deriveMergeCapabilities(modelDirs, baseModelDir, strategy, weights, density, brain, result),
    [modelDirs, baseModelDir, strategy, weights, density, brain, result],
  );
  const mergeRisk = useMemo(() => assessMergeConfig(mergeCaps), [mergeCaps]);

  const mergeSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildMergeContextSnippet(mergeCaps, locale);
  }, [brain, mergeCaps, locale]);

  const briefChat = useModelChat({
    modelId: brain?.model_id || null,
    systemPrompt: mergeSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const mergePrompts = useMemo(() => getMergeSuggestedPrompts(mergeCaps, locale), [mergeCaps, locale]);

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${mergeCaps.strategy}:${mergeCaps.validCount}:${mergeCaps.status}:${mergeRisk.level}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildMergeAutoBrief(mergeCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, mergeCaps.strategy, mergeCaps.validCount, mergeCaps.status, mergeRisk.level, locale]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const MERGE_RISK_BANNER_CLASS: Record<typeof mergeRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const addModelSlot = () => setModelDirs([...modelDirs, '']);
  const removeModelSlot = (idx: number) => {
    if (modelDirs.length <= 2) return;
    setModelDirs(modelDirs.filter((_, i) => i !== idx));
  };
  const updateModelDir = (idx: number, val: string) => {
    const copy = [...modelDirs];
    copy[idx] = val;
    setModelDirs(copy);
  };

  const handleStart = async () => {
    const parsedWeights = weights.trim()
      ? weights.split(',').map(Number).filter(n => !isNaN(n))
      : [];

    try {
      const resp = await startMerge({
        model_dirs: validModels,
        strategy,
        weights: parsedWeights,
        base_model_dir: baseModelDir,
        density,
      });
      setTaskId(resp.task_id);
    } catch {
      addToast('Failed to start merge', 'error');
    }
  };

  const handleComplete = (raw: unknown) => {
    const r = raw as MergeResult;
    setResult(r);
    if (r?.success) {
      addToast(`Merge complete: ${r.output_dir}`, 'success');
    }
  };

  const handlePick = (path: string) => {
    if (pickingBase) {
      setBaseModelDir(path);
      setPickingBase(false);
    } else if (pickingIdx !== null) {
      updateModelDir(pickingIdx, path);
      setPickingIdx(null);
    }
  };

  return (
    <div>
      <PageHeader title={t('page.merge')} description={t('merge.desc')} />

      <InsightPanel insights={mergeInsights} />

      {/* 4-card identity strip — N-source / strategy / weights / sovereignty */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('merge.cardSources')}
          value={`${mergeCaps.validCount}${mergeCaps.validCount > 0 ? ` / ${modelDirs.length}` : ''}`}
          hint={mergeCaps.validCount === 0
            ? t('merge.noSources')
            : mergeCaps.sourceNames.filter(Boolean).slice(0, 2).join(' + ') + (mergeCaps.validCount > 2 ? ` + ${mergeCaps.validCount - 2}` : '')}
          tone={mergeCaps.meetsCount ? 'emerald' : 'amber'}
        />
        <IdentityCard
          icon={<GitMerge size={16} />}
          label={t('merge.cardStrategy')}
          value={mergeStrategyLabel(strategy, locale)}
          hint={strategy === 'task_arithmetic' && !mergeCaps.hasBase
            ? t('merge.needsBase')
            : strategy === 'slerp' && mergeCaps.validCount !== 2
              ? t('merge.slerpExact2')
              : t('merge.strategyOk')}
          tone={mergeRisk.level === 'danger' ? 'red' : 'indigo'}
        />
        <IdentityCard
          icon={<Scale size={16} />}
          label={t('merge.cardWeights')}
          value={mergeCaps.weights.length === 0
            ? t('merge.uniform')
            : mergeCaps.weightsAligned
              ? `Σ=${mergeCaps.weightsSum.toFixed(2)}`
              : t('merge.mismatch')}
          hint={mergeCaps.weights.length === 0
            ? `${(1 / Math.max(1, mergeCaps.validCount)).toFixed(3)} ${t('merge.eachUniform')}`
            : `[${mergeCaps.weights.slice(0, 3).join(', ')}${mergeCaps.weights.length > 3 ? '…' : ''}]`}
          tone={mergeCaps.weights.length === 0
            ? 'neutral'
            : (mergeCaps.weightsAligned && (strategy !== 'linear' || Math.abs(mergeCaps.weightsSum - 1) <= 0.05))
              ? 'emerald'
              : 'amber'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('merge.cardSovereignty')}
          value={t('merge.zeroCloud')}
          hint={t('merge.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {mergeRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${MERGE_RISK_BANNER_CLASS[mergeRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {mergeRisk.level === 'danger' ? t('merge.riskDanger') : t('merge.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? mergeRisk.reasonZh : mergeRisk.reason}
        </div>
      )}

      {brain && (
        <ModelBriefCard
          className="mb-6"
          label={t('merge.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('merge.briefEmpty')}
          streamingText={t('merge.briefThinking')}
          refreshTitle={t('merge.briefRefire')}
          prompts={mergePrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildMergeAutoBrief(mergeCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Config */}
        <div className="space-y-4">
          {/* Strategy selector */}
          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">Strategy</h3>
            <div className="grid grid-cols-2 gap-2">
              {STRATEGIES.map((s) => (
                <button
                  key={s.value}
                  onClick={() => setStrategy(s.value)}
                  className={`rounded-lg border p-3 text-left transition-colors ${
                    strategy === s.value
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="text-sm font-medium">{s.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{s.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Model inputs */}
          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Models</h3>
              <button
                onClick={addModelSlot}
                className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-700"
              >
                <Plus size={14} /> Add Model
              </button>
            </div>

            <div className="space-y-2">
              {modelDirs.map((dir, idx) => (
                <div key={idx} className="flex gap-2">
                  <input
                    value={dir}
                    onChange={(e) => updateModelDir(idx, e.target.value)}
                    placeholder={`Model ${idx + 1} path`}
                    className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                  />
                  <button
                    onClick={() => setPickingIdx(idx)}
                    className="rounded-lg border border-gray-300 px-3 py-2 text-gray-500 hover:bg-gray-50"
                  >
                    <FolderOpen size={16} />
                  </button>
                  {modelDirs.length > 2 && (
                    <button
                      onClick={() => removeModelSlot(idx)}
                      className="rounded-lg border border-gray-300 px-2 py-2 text-gray-400 hover:text-red-500 hover:bg-red-50"
                    >
                      <X size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>

            {strategy === 'task_arithmetic' && (
              <div className="mt-3">
                <label className="block text-xs font-medium text-gray-600 mb-1">Base Model (required)</label>
                <div className="flex gap-2">
                  <input
                    value={baseModelDir}
                    onChange={(e) => setBaseModelDir(e.target.value)}
                    placeholder="/path/to/base-model"
                    className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
                  />
                  <button
                    onClick={() => setPickingBase(true)}
                    className="rounded-lg border border-gray-300 px-3 py-2 text-gray-500 hover:bg-gray-50"
                  >
                    <FolderOpen size={16} />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Advanced params */}
          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">Parameters</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Weights (comma-separated, empty = equal)
                </label>
                <input
                  value={weights}
                  onChange={(e) => setWeights(e.target.value)}
                  placeholder="e.g. 0.7, 0.3"
                  className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
                />
              </div>
              {strategy === 'ties' && (
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    Density (sparsification): {density}
                  </label>
                  <input
                    type="range" min={0.1} max={1} step={0.05}
                    value={density}
                    onChange={(e) => setDensity(+e.target.value)}
                    className="w-full"
                  />
                </div>
              )}
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={!canStart || !!taskId}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
          >
            {taskId ? <Loader2 size={16} className="animate-spin" /> : <GitMerge size={16} />}
            {taskId ? 'Merging...' : 'Start Merge'}
          </button>
        </div>

        {/* Right: Result */}
        <div>
          {result ? (
            <div className="rounded-xl border border-gray-200 bg-white p-5">
              <h3 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">Result</h3>
              {result.success ? (
                <div className="space-y-2 text-sm">
                  <p><span className="text-gray-500">Strategy:</span> {result.strategy}</p>
                  <p><span className="text-gray-500">Models:</span> {result.model_names.join(' + ')}</p>
                  <p><span className="text-gray-500">Output:</span> <code className="text-xs bg-gray-100 px-1 rounded">{result.output_dir}</code></p>
                  <p><span className="text-gray-500">Parameters:</span> {formatParamCount(result.merged_params)}</p>
                  <p><span className="text-gray-500">Size:</span> {formatSize(result.merged_size_bytes)}</p>
                  <p><span className="text-gray-500">Duration:</span> {result.duration_seconds}s</p>
                </div>
              ) : (
                <p className="text-sm text-red-600">{result.error}</p>
              )}
            </div>
          ) : (
            <EmptyState
              icon={<GitMerge size={40} />}
              title="Model Merge"
              description="Combine multiple models into one. Supports Linear, SLERP, TIES, and Task Arithmetic strategies."
            />
          )}
        </div>
      </div>

      {/* File picker */}
      {(pickingIdx !== null || pickingBase) && (
        <FileBrowser
          onSelect={handlePick}
          onCancel={() => { setPickingIdx(null); setPickingBase(false); }}
        />
      )}

      {/* Progress overlay */}
      <ProgressOverlay
        taskId={taskId}
        title="Model Merge"
        onComplete={handleComplete}
        onError={(err) => addToast(err, 'error')}
        onClose={() => setTaskId(null)}
      />

      {/* Ask Model FAB */}
      {brain && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('merge.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brain.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <GitMerge size={13} className="text-indigo-500" />
                  {t('merge.askDrawerTitle')}
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
                  <p className="text-xs text-stone-400">{t('merge.askDrawerHint')}</p>
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
                    placeholder={t('merge.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('merge.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('merge.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {mergePrompts.slice(0, 4).map((p) => (
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
