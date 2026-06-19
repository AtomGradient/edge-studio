// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * BatchOperations — multi-model batch optimization with results table.
 *
 * §9.1 multi-component (variable-N queue + cohort) + §9.2 Sovereignty.
 * Brain LLM narrates queue health, predicts duration, post-mortems failures.
 */

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, FolderOpen, Gauge, ListChecks, Loader2, Play, Plus, Sparkles, Trash2, Zap, Layers, Wrench, Shield, Send, Square, X } from 'lucide-react';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { WorkflowGuide } from '@/components/common/WorkflowGuide';
import { AskModelFab } from '@/components/common/AskModelFab';
import { useBatchInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useModelStore } from '@/stores/modelStore';
import { FileBrowser } from '@/components/model/FileBrowser';
import { useT, useLocaleStore } from '@/i18n';
import { cn, formatSize } from '@/lib/utils';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveBatchCapabilities,
  assessBatchConfig,
  buildBatchContextSnippet,
  buildBatchAutoBrief,
  getBatchSuggestedPrompts,
  operationLabel,
  type BatchItem,
  type BatchResult,
} from '@/lib/batchInsights';
import axios from 'axios';

const OPERATIONS = [
  { value: 'quantization', label: 'Quantize' },
  { value: 'neuron_pruning', label: 'Neuron Prune' },
  { value: 'vocab_pruning', label: 'Vocab Prune' },
];

const SAMPLE_BATCH_ITEMS: BatchItem[] = [
  {
    model_dir: 'sample://batch/edgedemo-3b-chat',
    label: 'EdgeDemo 3B chat',
    bits: 4,
    operation: 'quantization',
  },
  {
    model_dir: 'sample://batch/edgedemo-7b-chat',
    label: 'EdgeDemo 7B chat',
    bits: 4,
    operation: 'quantization',
  },
  {
    model_dir: 'sample://batch/edgedemo-14b-distill',
    label: 'EdgeDemo 14B distill',
    bits: 4,
    operation: 'quantization',
  },
];

const SAMPLE_BATCH_RESULTS: BatchResult[] = [
  {
    label: 'EdgeDemo 3B chat',
    success: true,
    output_dir: 'sample://batch/outputs/edgedemo-3b-chat-q4',
    original_size: 6_200_000_000,
    result_size: 2_050_000_000,
    duration_seconds: 72.4,
  },
  {
    label: 'EdgeDemo 7B chat',
    success: true,
    output_dir: 'sample://batch/outputs/edgedemo-7b-chat-q4',
    original_size: 15_200_000_000,
    result_size: 4_260_000_000,
    duration_seconds: 158.5,
  },
  {
    label: 'EdgeDemo 14B distill',
    success: true,
    output_dir: 'sample://batch/outputs/edgedemo-14b-distill-q4',
    original_size: 28_400_000_000,
    result_size: 8_100_000_000,
    duration_seconds: 312.0,
  },
];

interface TaskStatusPayload {
  status: 'pending' | 'running' | 'complete' | 'error' | 'cancelled';
  progress?: number;
  message?: string;
  error?: string | null;
  result?: {
    success?: boolean;
    final_output_dir?: string;
    original_size_bytes?: number;
    optimized_size_bytes?: number;
    total_duration_seconds?: number;
  };
}

function getErrorMessage(err: unknown, fallback: string): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => (typeof item?.msg === 'string' ? item.msg : String(item)))
        .join('; ');
    }
    return err.message || fallback;
  }
  if (err instanceof Error) return err.message;
  return fallback;
}

export default function BatchOperations() {
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const brain = useModelStore((s) => s.currentModel);
  const batchInsights = useBatchInsights(t);

  const [items, setItems] = useState<BatchItem[]>([]);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [progressPercent, setProgressPercent] = useState(0);
  const [runError, setRunError] = useState('');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [samplePreview, setSamplePreview] = useState(false);

  // ── §9.1 capability + §9.2 risk + AI brief ─────────────────────────────
  const batchCaps = useMemo(
    () => deriveBatchCapabilities(items, results, running, brain),
    [items, results, running, brain],
  );
  const batchRisk = useMemo(() => assessBatchConfig(batchCaps), [batchCaps]);

  const batchSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildBatchContextSnippet(batchCaps, locale, { fixture: samplePreview });
  }, [brain, batchCaps, locale, samplePreview]);

  const briefChat = useModelChat({
    modelId: brain?.model_id || null,
    systemPrompt: batchSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const batchPrompts = useMemo(
    () => getBatchSuggestedPrompts(batchCaps, locale, { fixture: samplePreview }),
    [batchCaps, locale, samplePreview],
  );
  const loadedModelQueued = useMemo(
    () => !!brain && items.some((item) => item.model_dir === brain.model_dir),
    [brain, items],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${batchCaps.runPhase}:${batchCaps.itemCount}:${batchCaps.completedCount}:${batchRisk.level}:${locale}:${samplePreview ? 'fixture' : 'real'}`;
    if (briefFiredForRef.current === sig) return;
    const id = window.setTimeout(() => {
      briefFiredForRef.current = sig;
      briefChat.send(buildBatchAutoBrief(batchCaps, locale, { fixture: samplePreview }));
    }, 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, batchCaps.runPhase, batchCaps.itemCount, batchCaps.completedCount, batchRisk.level, locale, samplePreview]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const RISK_BANNER_CLASS: Record<typeof batchRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  // ── Queue actions ──────────────────────────────────────────────────────
  const loadSamplePreview = () => {
    setSamplePreview(true);
    setItems(SAMPLE_BATCH_ITEMS);
    setResults(SAMPLE_BATCH_RESULTS);
    setRunning(false);
    setProgress('');
    setProgressPercent(0);
    setRunError('');
    briefFiredForRef.current = null;
    briefChat.reset();
  };

  const clearSamplePreview = () => {
    setSamplePreview(false);
    setItems(brain ? [{ model_dir: brain.model_dir, label: brain.model_name, bits: 4, operation: 'quantization' }] : []);
    setResults([]);
    setProgress('');
    setProgressPercent(0);
    setRunError('');
    briefFiredForRef.current = null;
    briefChat.reset();
  };

  const addModel = (path: string) => {
    setPickerOpen(false);
    if (samplePreview) {
      setSamplePreview(false);
      setResults([]);
      setProgress('');
      setProgressPercent(0);
      setRunError('');
    }
    const label = path.split('/').pop() || path;
    setItems((prev) => (
      (samplePreview ? [] : prev).some((item) => item.model_dir === path)
        ? prev
        : [...(samplePreview ? [] : prev), { model_dir: path, label, bits: 4, operation: 'quantization' }]
    ));
  };

  const addLoadedModel = () => {
    if (!brain) return;
    if (samplePreview) {
      setSamplePreview(false);
      setResults([]);
      setProgress('');
      setProgressPercent(0);
      setRunError('');
      setItems([{ model_dir: brain.model_dir, label: brain.model_name, bits: 4, operation: 'quantization' }]);
      return;
    }
    setItems((prev) => (
      prev.some((item) => item.model_dir === brain.model_dir)
        ? prev
        : [...prev, { model_dir: brain.model_dir, label: brain.model_name, bits: 4, operation: 'quantization' }]
    ));
  };

  const removeItem = (idx: number) => {
    if (samplePreview) {
      clearSamplePreview();
      return;
    }
    setItems((prev) => prev.filter((_, i) => i !== idx));
  };

  const updateItem = (idx: number, updates: Partial<BatchItem>) => {
    if (samplePreview) return;
    setItems((prev) => prev.map((item, i) => (i === idx ? { ...item, ...updates } : item)));
  };

  const handleRun = async () => {
    if (items.length === 0 || samplePreview) return;
    setRunning(true);
    setResults([]);
    setProgress('');
    setProgressPercent(0);
    setRunError('');

    const batchResults: BatchResult[] = [];

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      setProgress(t('pro.batch.progressItem', { label: item.label, current: i + 1, total: items.length }));
      setProgressPercent(Math.round((i / items.length) * 100));

      try {
        const loadRes = await axios.post('/api/model/load', { model_dir: item.model_dir });
        const modelId = loadRes.data.model_id;

        const pipeRes = await axios.post(`/api/model/${modelId}/pipeline/run`, {
          steps: [{ operation: item.operation, params: { bits: item.bits } }],
          skip_validation: true,
        });

        const taskId = pipeRes.data.task_id;

        let done = false;
        while (!done) {
          await new Promise((r) => setTimeout(r, 1500));
          const statusRes = await axios.get<TaskStatusPayload>(`/api/task/${taskId}`);
          const status = statusRes.data;
          if (typeof status.progress === 'number') {
            const totalProgress = ((i + Math.max(0, Math.min(1, status.progress))) / items.length) * 100;
            setProgressPercent(Math.round(totalProgress));
          }
          if (status.message) {
            setProgress(`${item.label}: ${status.message}`);
          }
          if (status.status === 'complete') {
            done = true;
            const r = status.result || {};
            batchResults.push({
              label: item.label,
              success: !!r.success,
              output_dir: r.final_output_dir,
              original_size: r.original_size_bytes,
              result_size: r.optimized_size_bytes,
              duration_seconds: r.total_duration_seconds,
            });
          } else if (status.status === 'error' || status.status === 'cancelled') {
            throw new Error(status.error || status.message || t('pro.batch.progressFailed'));
          }
        }
      } catch (err: unknown) {
        const msg = getErrorMessage(err, t('pro.batch.progressFailed'));
        setRunError(msg);
        batchResults.push({ label: item.label, success: false, error: msg });
      }

      setResults([...batchResults]);
      setProgressPercent(Math.round(((i + 1) / items.length) * 100));
    }

    setProgress('');
    setProgressPercent(100);
    setRunning(false);
  };

  // ── Identity strip ─────────────────────────────────────────────────────
  const queueTone: 'neutral' | 'emerald' | 'amber' | 'red' =
    batchCaps.itemCount === 0
      ? 'neutral'
      : batchCaps.duplicateLabels.length > 0
        ? 'amber'
        : 'emerald';

  const opTone: 'neutral' | 'indigo' | 'amber' =
    batchCaps.itemCount === 0
      ? 'neutral'
      : batchCaps.opMix.length === 1
        ? 'indigo'
        : 'amber';

  const phaseTone: 'neutral' | 'emerald' | 'amber' | 'red' = (() => {
    switch (batchCaps.runPhase) {
      case 'idle': return 'neutral';
      case 'running': return 'amber';
      case 'complete': return 'emerald';
      case 'partial': return 'amber';
      case 'allFailed': return 'red';
      default: return 'neutral';
    }
  })();

  const phaseLabel = (() => {
    if (locale === 'zh') {
      switch (batchCaps.runPhase) {
        case 'idle': return batchCaps.itemCount === 0 ? '空队列' : '待运行';
        case 'running': return `运行中 ${batchCaps.completedCount}/${batchCaps.itemCount}`;
        case 'complete': return `全部完成 ${batchCaps.successCount}/${batchCaps.itemCount}`;
        case 'partial': return `部分完成 ${batchCaps.successCount}+${batchCaps.failedCount}`;
        case 'allFailed': return `全部失败 ${batchCaps.failedCount}`;
        default: return '';
      }
    }
    switch (batchCaps.runPhase) {
      case 'idle': return batchCaps.itemCount === 0 ? 'Empty queue' : 'Ready';
      case 'running': return `Running ${batchCaps.completedCount}/${batchCaps.itemCount}`;
      case 'complete': return `Complete ${batchCaps.successCount}/${batchCaps.itemCount}`;
      case 'partial': return `Partial ${batchCaps.successCount}+${batchCaps.failedCount}`;
      case 'allFailed': return `All failed ${batchCaps.failedCount}`;
      default: return '';
    }
  })();

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title={t('pro.batch.title')}
        description={t('pro.batch.desc')}
      />

      <InsightPanel insights={batchInsights} />

      {/* §9.1 4-card identity strip — Queue / Operations / Phase / Sovereignty */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('pro.batch.cardQueue')}
          value={`${batchCaps.itemCount} ${locale === 'zh' ? '项' : batchCaps.itemCount === 1 ? 'item' : 'items'}`}
          hint={
            batchCaps.itemCount === 0
              ? t('pro.batch.noItems')
              : batchCaps.duplicateLabels.length > 0
                ? t('pro.batch.duplicatesHint').replace('{n}', String(batchCaps.duplicateLabels.length))
                : `${batchCaps.uniquePathCount} ${locale === 'zh' ? '个唯一路径' : 'unique paths'}`
          }
          tone={queueTone}
        />
        <IdentityCard
          icon={<Wrench size={16} />}
          label={t('pro.batch.cardOps')}
          value={
            batchCaps.opMix.length === 0
              ? '—'
              : batchCaps.opMix.length === 1
                ? operationLabel(batchCaps.opMix[0], locale)
                : `${batchCaps.opMix.length} ${locale === 'zh' ? '种' : 'mixed'}`
          }
          hint={
            batchCaps.opMix.length === 0
              ? t('pro.batch.opsNone')
              : batchCaps.opMix
                  .map((op) => `${operationLabel(op, locale)}×${batchCaps.opHistogram[op]}`)
                  .join(' · ')
          }
          tone={opTone}
        />
        <IdentityCard
          icon={<Zap size={16} />}
          label={t('pro.batch.cardPhase')}
          value={phaseLabel}
          hint={
            batchCaps.runPhase === 'complete' && batchCaps.totalSavedBytes > 0
              ? `${formatSize(batchCaps.totalSavedBytes)} ${locale === 'zh' ? '已省 · 平均 ' : 'saved · avg '}${batchCaps.avgSavingsPct.toFixed(1)}%`
              : batchCaps.bitsRange.values.length > 0
                ? `${locale === 'zh' ? '量化 ' : 'quant '}${batchCaps.bitsRange.min === batchCaps.bitsRange.max ? `${batchCaps.bitsRange.min}-bit` : `${batchCaps.bitsRange.min}–${batchCaps.bitsRange.max} bit`}`
                : t('pro.batch.phaseHint')
          }
          tone={phaseTone}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('pro.batch.cardSovereignty')}
          value={t('pro.batch.zeroCloud')}
          hint={t('pro.batch.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {batchRisk.level !== 'safe' && batchCaps.itemCount > 0 && (
        <div className={`rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[batchRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {batchRisk.level === 'danger' ? t('pro.batch.riskDanger') : t('pro.batch.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? batchRisk.reasonZh : batchRisk.reason}
        </div>
      )}

      <WorkflowGuide
        title={t('pro.batch.workflowTitle')}
        description={t('pro.batch.workflowDesc')}
        badges={[
          { label: t('pro.batch.defaultSettings') },
          { label: t('pro.batch.queueCount', { count: items.length }), tone: 'indigo' },
        ]}
        steps={[
          {
            icon: <FolderOpen size={16} />,
            title: t('pro.batch.stepAddTitle'),
            description: t('pro.batch.stepAddDesc'),
          },
          {
            icon: <Gauge size={16} />,
            title: t('pro.batch.stepConfigTitle'),
            description: t('pro.batch.stepConfigDesc'),
          },
          {
            icon: <ListChecks size={16} />,
            title: t('pro.batch.stepRunTitle'),
            description: t('pro.batch.stepRunDesc'),
          },
        ]}
      >
        <div className="flex flex-col gap-3 rounded-lg border border-indigo-100 bg-indigo-50/60 px-3 py-3 dark:border-indigo-400/30 dark:bg-indigo-950/50 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 gap-2">
            <Sparkles size={16} className="mt-0.5 shrink-0 text-indigo-600 dark:text-indigo-300" />
            <div className="min-w-0">
              <p className="text-xs font-semibold text-indigo-900 dark:text-indigo-100">{t('pro.batch.sampleTitle')}</p>
              <p className="mt-1 text-xs leading-relaxed text-indigo-700 dark:text-indigo-200">{t('pro.batch.sampleDesc')}</p>
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            <button
              type="button"
              onClick={loadSamplePreview}
              disabled={running}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-indigo-400 dark:text-gray-950 dark:hover:bg-indigo-300"
            >
              <Sparkles size={13} />
              {samplePreview ? t('pro.batch.sampleReload') : t('pro.batch.sampleLoad')}
            </button>
            {samplePreview && (
              <button
                type="button"
                onClick={clearSamplePreview}
                className="inline-flex items-center justify-center rounded-lg border border-indigo-200 bg-white/70 px-3 py-2 text-xs font-medium text-indigo-700 transition-colors hover:bg-white dark:border-indigo-400/40 dark:bg-stone-950/80 dark:text-indigo-200 dark:hover:bg-stone-900"
              >
                {t('pro.batch.sampleClear')}
              </button>
            )}
          </div>
        </div>
      </WorkflowGuide>

      {samplePreview && (
        <div className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs leading-relaxed text-sky-800 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-200">
          <span className="font-semibold">{t('pro.batch.sampleBannerTitle')}</span>
          {' '}{t('pro.batch.sampleBannerDesc')}
        </div>
      )}

      {/* AI Brief — brain LLM narrates the queue */}
      {brain && (
        <ModelBriefCard
          label={t('pro.batch.briefTitle')}
          status={briefChat.status || brain.model_name}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('pro.batch.briefEmpty')}
          streamingText={t('pro.batch.briefThinking')}
          refreshTitle={t('pro.batch.briefRefire')}
          prompts={batchPrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildBatchAutoBrief(batchCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {/* Items queue */}
      <div className="space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">{t('pro.batch.queueTitle')}</h2>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{t('pro.batch.queueHint')}</p>
          </div>
          {brain && !loadedModelQueued && (
            <button
              type="button"
              onClick={addLoadedModel}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-100 dark:border-indigo-500/30 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
            >
              <Layers size={14} />
              {t('pro.batch.addLoadedModel')}
            </button>
          )}
        </div>

        {items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 px-5 py-6 text-center dark:border-gray-700 dark:bg-gray-900/50">
            <FolderOpen size={20} className="mx-auto text-gray-400 dark:text-gray-500" />
            <p className="mt-2 text-sm font-medium text-gray-800 dark:text-gray-100">{t('pro.batch.emptyQueueTitle')}</p>
            <p className="mx-auto mt-1 max-w-2xl text-xs leading-relaxed text-gray-500 dark:text-gray-400">
              {t('pro.batch.emptyQueueDesc')}
            </p>
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="mt-4 inline-flex items-center justify-center gap-2 rounded-lg bg-gray-900 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-200"
            >
              <Plus size={14} />
              {t('pro.batch.addModel')}
            </button>
            <button
              type="button"
              onClick={loadSamplePreview}
              className="ml-2 mt-4 inline-flex items-center justify-center gap-2 rounded-lg border border-indigo-200 bg-white px-3 py-2 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-50 dark:border-indigo-500/30 dark:bg-gray-900 dark:text-indigo-300 dark:hover:bg-indigo-500/10"
            >
              <Sparkles size={14} />
              {t('pro.batch.sampleLoad')}
            </button>
          </div>
        ) : (
          <>
            {items.map((item, idx) => {
              const isDangerous = item.operation === 'quantization' && item.bits <= 3;
              const isDup = batchCaps.duplicateLabels.includes(item.label);
              return (
                <div
                  key={`${item.model_dir}:${idx}`}
                  className={cn(
                    'flex items-center gap-3 rounded-xl border bg-white px-5 py-3 dark:bg-gray-900',
                    isDangerous
                      ? 'border-red-300 dark:border-red-900/60'
                      : isDup
                        ? 'border-amber-300 dark:border-amber-900/60'
                        : 'border-gray-200 dark:border-gray-800',
                  )}
                >
                  <Zap size={16} className="shrink-0 text-amber-500 dark:text-amber-400" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">{item.label}</p>
                    {(isDangerous || isDup) && (
                      <p className="truncate text-[10px] uppercase tracking-wider text-red-600 dark:text-red-400">
                        {isDangerous && (locale === 'zh' ? `⚠ ${item.bits}-bit 量化 PPL 有崩盘风险` : `⚠ ${item.bits}-bit quant — PPL collapse risk`)}
                        {isDangerous && isDup ? ' · ' : ''}
                        {isDup && (locale === 'zh' ? '⚠ 路径重复' : '⚠ duplicate path')}
                      </p>
                    )}
                  </div>
                  <select
                    value={item.operation}
                    onChange={(e) => updateItem(idx, { operation: e.target.value })}
                    disabled={samplePreview}
                    className="rounded-lg border border-gray-200 bg-gray-50 px-2.5 py-1 text-xs text-gray-700 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200"
                  >
                    {OPERATIONS.map((op) => (
                      <option key={op.value} value={op.value}>{op.label}</option>
                    ))}
                  </select>
                  {item.operation === 'quantization' && (
                    <select
                      value={item.bits}
                      onChange={(e) => updateItem(idx, { bits: Number(e.target.value) })}
                      disabled={samplePreview}
                      className={cn(
                        'rounded-lg border bg-gray-50 px-2.5 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-60 dark:bg-gray-800',
                        item.bits <= 3
                          ? 'border-red-300 text-red-700 dark:border-red-900/60 dark:text-red-300'
                          : 'border-gray-200 text-gray-700 dark:border-gray-700 dark:text-gray-200',
                      )}
                    >
                      {[2, 3, 4, 8].map((b) => (
                        <option key={b} value={b}>{b}-bit</option>
                      ))}
                    </select>
                  )}
                  <button
                    type="button"
                    onClick={() => removeItem(idx)}
                    className="rounded-lg p-1 text-gray-400 transition-colors hover:bg-gray-100 hover:text-red-500 dark:hover:bg-gray-800"
                    aria-label={t('pro.batch.removeModel')}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            })}
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-gray-300 py-3 text-sm text-gray-500 transition-colors hover:border-gray-400 hover:text-gray-700 dark:border-gray-700 dark:hover:border-gray-600 dark:hover:text-gray-300"
            >
              <Plus size={16} />
              {t('pro.batch.addModel')}
            </button>
          </>
        )}
      </div>

      {/* Run */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <button
            type="button"
            onClick={handleRun}
            disabled={running || items.length === 0 || batchRisk.level === 'danger' || samplePreview}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-gray-900 px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-200"
          >
            {running ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {running ? t('common.loading') : t('pro.batch.runAll')}
          </button>
          <p className={cn(
            'text-xs',
            batchRisk.level === 'danger' && batchCaps.itemCount > 0
              ? 'text-red-600 dark:text-red-400'
              : 'text-gray-500 dark:text-gray-400',
          )}>
            {samplePreview
              ? t('pro.batch.sampleRunHint')
              : batchRisk.level === 'danger' && batchCaps.itemCount > 0
              ? t('pro.batch.runBlocked')
              : items.length === 0
                ? t('pro.batch.runDisabledHint')
                : t('pro.batch.runReadyHint')}
          </p>
        </div>

        {running && (
          <div className="mt-4">
            <div className="mb-1 flex items-center justify-between gap-3 text-xs text-gray-500 dark:text-gray-400">
              <span className="truncate">{progress || t('common.loading')}</span>
              <span className="shrink-0 font-mono">{progressPercent}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        )}

        {runError && (
          <div className="mt-4 flex gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">{t('pro.batch.errorTitle')}</p>
              <p className="mt-0.5">{runError}</p>
            </div>
          </div>
        )}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="overflow-hidden rounded-2xl border border-gray-200 dark:border-gray-800">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400">{t('pro.batch.colModel')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('pro.batch.colOriginal')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('pro.batch.colResult')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('pro.batch.colSavings')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('pro.batch.colTime')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('pro.batch.colStatus')}</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => {
                const savings = r.original_size && r.result_size
                  ? ((r.original_size - r.result_size) / r.original_size * 100).toFixed(1)
                  : null;
                const bucket = batchCaps.savingsBuckets[idx];
                const savingsClass =
                  bucket === 'extreme' ? 'text-emerald-600 dark:text-emerald-400 font-semibold'
                  : bucket === 'strong' ? 'text-emerald-600 dark:text-emerald-400'
                  : bucket === 'modest' ? 'text-green-600 dark:text-green-400'
                  : bucket === 'trim' ? 'text-amber-600 dark:text-amber-400'
                  : 'text-gray-400 dark:text-gray-500';
                const isBest = batchCaps.bestSavingsItem === r;
                const isWorst = batchCaps.worstSavingsItem === r && batchCaps.worstSavingsItem !== batchCaps.bestSavingsItem;
                return (
                  <tr key={idx} className="border-t border-gray-100 dark:border-gray-800/60">
                    <td className="px-4 py-2.5 font-medium text-gray-900 dark:text-gray-100">
                      {r.label}
                      {isBest && <span className="ml-2 rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300">{locale === 'zh' ? '最优' : 'best'}</span>}
                      {isWorst && <span className="ml-2 rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold text-amber-700 dark:bg-amber-900/40 dark:text-amber-300">{locale === 'zh' ? '最弱' : 'worst'}</span>}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                      {r.original_size ? formatSize(r.original_size) : '--'}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                      {r.result_size ? formatSize(r.result_size) : '--'}
                    </td>
                    <td className={cn('px-4 py-2.5 text-right font-mono', savingsClass)}>
                      {savings ? `−${savings}%` : '--'}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                      {r.duration_seconds ? `${r.duration_seconds.toFixed(1)}s` : '--'}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <span className={cn(
                        'rounded-full px-2.5 py-0.5 text-xs font-medium',
                        r.success
                          ? 'bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-400',
                      )}>
                        {r.success ? t('pro.batch.statusOk') : t('pro.batch.statusError')}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
            {batchCaps.successCount > 0 && (
              <tfoot className="bg-gray-50 dark:bg-gray-900/50">
                <tr className="border-t border-gray-200 dark:border-gray-800">
                  <td className="px-4 py-2.5 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    {t('pro.batch.totalRow')}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-700 dark:text-gray-300">
                    {formatSize(batchCaps.totalOriginalBytes)}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-700 dark:text-gray-300">
                    {formatSize(batchCaps.totalResultBytes)}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                    −{batchCaps.avgSavingsPct.toFixed(1)}% avg
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-700 dark:text-gray-300">
                    {batchCaps.avgDurationSec.toFixed(1)}s avg
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-700 dark:text-gray-300">
                    {batchCaps.successCount}/{batchCaps.completedCount}
                  </td>
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      )}

      {pickerOpen && (
        <FileBrowser
          onSelect={addModel}
          onCancel={() => setPickerOpen(false)}
        />
      )}

      {/* Ask Model FAB drawer */}
      {brain && (
        <>
          {!drawerOpen && (
            <AskModelFab
              label={t('pro.batch.askFab')}
              modelName={brain.model_name}
              icon={<Layers size={15} />}
              onClick={() => setDrawerOpen(true)}
            />
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Layers size={13} className="text-indigo-500" />
                  {t('pro.batch.askDrawerTitle')}
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
                  <p className="text-xs text-stone-400">{t('pro.batch.askDrawerHint')}</p>
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
                    placeholder={t('pro.batch.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('pro.batch.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('pro.batch.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {batchPrompts.slice(0, 4).map((p) => (
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
