// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * BenchmarkDashboard — batch benchmark with side-by-side comparison.
 * Supports multiple models, CSV export, and Plotly charts.
 */

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { AlertCircle, BarChart3, Brain, Download, FolderOpen, Gauge, ListChecks, Loader2, Play, Plus, Shield, Sparkles, Square, Send, Trash2, Trophy, X as XIcon, Zap } from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { WorkflowGuide } from '@/components/common/WorkflowGuide';
import { AskModelFab } from '@/components/common/AskModelFab';
import { useBenchmarkInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { FileBrowser } from '@/components/model/FileBrowser';
import { useT, useLocaleStore } from '@/i18n';
import { formatSize } from '@/lib/utils';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  summarizeBenchmark,
  assessBenchmarkCohort,
  buildBenchmarkContextSnippet,
  buildBenchmarkAutoBrief,
  getBenchmarkSuggestedPrompts,
  bucketLabel,
} from '@/lib/benchmarkInsights';
import axios from 'axios';

interface BenchmarkItem {
  model_dir: string;
  label: string;
}

interface BenchmarkResult {
  label: string;
  model_dir: string;
  success: boolean;
  result?: {
    disk_size_bytes: number;
    memory_peak_mb: number;
    tokens_per_second: number;
    time_to_first_token_ms: number;
    perplexity: number;
  };
  error?: string;
}

const SAMPLE_BENCHMARK_ITEMS: BenchmarkItem[] = [
  {
    label: 'EdgeDemo 7B FP16 baseline',
    model_dir: 'sample://benchmark/edgedemo-7b-fp16',
  },
  {
    label: 'EdgeDemo 7B INT8 export',
    model_dir: 'sample://benchmark/edgedemo-7b-int8',
  },
  {
    label: 'EdgeDemo 7B Q4 mobile',
    model_dir: 'sample://benchmark/edgedemo-7b-q4-mobile',
  },
];

const SAMPLE_BENCHMARK_RESULTS: BenchmarkResult[] = [
  {
    label: 'EdgeDemo 7B FP16 baseline',
    model_dir: 'sample://benchmark/edgedemo-7b-fp16',
    success: true,
    result: {
      disk_size_bytes: 15_200_000_000,
      memory_peak_mb: 8420,
      tokens_per_second: 26.4,
      time_to_first_token_ms: 420,
      perplexity: 6.82,
    },
  },
  {
    label: 'EdgeDemo 7B INT8 export',
    model_dir: 'sample://benchmark/edgedemo-7b-int8',
    success: true,
    result: {
      disk_size_bytes: 7_880_000_000,
      memory_peak_mb: 4760,
      tokens_per_second: 36.8,
      time_to_first_token_ms: 310,
      perplexity: 7.08,
    },
  },
  {
    label: 'EdgeDemo 7B Q4 mobile',
    model_dir: 'sample://benchmark/edgedemo-7b-q4-mobile',
    success: true,
    result: {
      disk_size_bytes: 4_260_000_000,
      memory_peak_mb: 2920,
      tokens_per_second: 48.9,
      time_to_first_token_ms: 235,
      perplexity: 9.36,
    },
  },
];

interface TaskStatusPayload {
  status: 'pending' | 'running' | 'complete' | 'error' | 'cancelled';
  progress?: number;
  message?: string;
  error?: string | null;
  result?: {
    results?: BenchmarkResult[];
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

export default function BenchmarkDashboard() {
  const t = useT();
  const model = useModelStore((s) => s.currentModel);
  const benchmarkInsights = useBenchmarkInsights(t);

  const [items, setItems] = useState<BenchmarkItem[]>(() => {
    const initial: BenchmarkItem[] = [];
    if (model) {
      initial.push({ model_dir: model.model_dir, label: model.model_name });
    }
    return initial;
  });
  const [results, setResults] = useState<BenchmarkResult[]>([]);
  const [running, setRunning] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const [runError, setRunError] = useState('');
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerIndex, setPickerIndex] = useState(-1);
  const [samplePreview, setSamplePreview] = useState(false);

  // ── i18n + locale ───────────────────────────────────────────────────────
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  // ── Cohort summary + risk + chat (§9.1 capability + §9.2 sovereignty) ───
  const summary = useMemo(() => summarizeBenchmark(results), [results]);
  const cohortRisk = useMemo(() => assessBenchmarkCohort(summary), [summary]);

  const benchmarkSystemPrompt = useMemo(() => {
    if (!model) return '';
    return (
      buildModelSelfSystemPrompt(model, locale) +
      '\n\n' +
      buildBenchmarkContextSnippet(results, summary, model, locale, { fixture: samplePreview })
    );
  }, [model, results, summary, locale, samplePreview]);

  const briefChat = useModelChat({
    modelId: model?.model_id || null,
    systemPrompt: benchmarkSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const cohortPrompts = useMemo(
    () => getBenchmarkSuggestedPrompts(summary, model, locale, { fixture: samplePreview }),
    [summary, model, locale, samplePreview],
  );
  const loadedModelQueued = useMemo(
    () => !!model && items.some((item) => item.model_dir === model.model_dir),
    [items, model],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // Fire-once brief; refire on (model + cohort signature) change.
  useEffect(() => {
    if (!model) return;
    if (briefChat.streaming) return;
    const sig = `${model.model_id}:${summary.totalRuns}:${summary.successCount}:${summary.brokenPPL.length}:${summary.slowOutliers.length}:${locale}:${samplePreview ? 'fixture' : 'real'}`;
    if (briefFiredForRef.current === sig) return;
    const id = window.setTimeout(() => {
      // React StrictMode cancels the first mount timer in dev. Mark fired only
      // when the send actually runs, otherwise the empty brief can get stuck.
      briefFiredForRef.current = sig;
      briefChat.send(buildBenchmarkAutoBrief(summary, locale, { fixture: samplePreview }));
    }, 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, summary.totalRuns, summary.successCount, summary.brokenPPL.length, summary.slowOutliers.length, locale, samplePreview]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const COHORT_RISK_BANNER_CLASS: Record<typeof cohortRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const addItem = () => {
    if (samplePreview) {
      setSamplePreview(false);
      setResults([]);
      setRunError('');
      setItems(model ? [{ model_dir: model.model_dir, label: model.model_name }] : []);
      setPickerIndex(model ? 1 : 0);
    } else {
      setPickerIndex(items.length);
    }
    setPickerOpen(true);
  };

  const addLoadedModel = () => {
    if (!model) return;
    if (samplePreview) {
      setSamplePreview(false);
      setResults([]);
      setRunError('');
      setItems([{ model_dir: model.model_dir, label: model.model_name }]);
      return;
    }
    setItems((prev) => (
      prev.some((item) => item.model_dir === model.model_dir)
        ? prev
        : [...prev, { model_dir: model.model_dir, label: model.model_name }]
    ));
  };

  const handlePickModel = (path: string) => {
    if (samplePreview) {
      setSamplePreview(false);
      setResults([]);
      setRunError('');
    }
    setPickerOpen(false);
    const label = path.split('/').pop() || path;
    if (pickerIndex >= items.length) {
      setItems((prev) => (
        prev.some((item) => item.model_dir === path)
          ? prev
          : [...prev, { model_dir: path, label }]
      ));
    } else {
      setItems((prev) => prev.map((item, i) => (i === pickerIndex ? { model_dir: path, label } : item)));
    }
  };

  const removeItem = (idx: number) => {
    if (samplePreview) {
      setSamplePreview(false);
      setResults([]);
      setRunError('');
    }
    setItems((prev) => prev.filter((_, i) => i !== idx));
  };

  const loadSamplePreview = () => {
    setSamplePreview(true);
    setItems(SAMPLE_BENCHMARK_ITEMS);
    setResults(SAMPLE_BENCHMARK_RESULTS);
    setRunning(false);
    setRunError('');
    setProgressPercent(0);
    setProgressMessage('');
    briefFiredForRef.current = null;
    briefChat.reset();
  };

  const clearSamplePreview = () => {
    setSamplePreview(false);
    setResults([]);
    setRunError('');
    setProgressPercent(0);
    setProgressMessage('');
    setItems(model ? [{ model_dir: model.model_dir, label: model.model_name }] : []);
    briefFiredForRef.current = null;
    briefChat.reset();
  };

  const handleRun = async () => {
    if (items.length === 0) return;
    setSamplePreview(false);
    setRunning(true);
    setResults([]);
    setRunError('');
    setProgressPercent(0);
    setProgressMessage(t('benchmark.progressQueued'));
    try {
      const res = await axios.post('/api/benchmark/batch', {
        items: items.map((item) => ({
          model_dir: item.model_dir,
          label: item.label,
          num_tokens: 100,
          num_ppl_texts: 3,
        })),
      });
      const taskId = res.data.task_id;

      let done = false;
      while (!done) {
        await new Promise((r) => setTimeout(r, 2000));
        const statusRes = await axios.get<TaskStatusPayload>(`/api/task/${taskId}`);
        const status = statusRes.data;
        if (typeof status.progress === 'number') {
          setProgressPercent(Math.round(Math.max(0, Math.min(1, status.progress)) * 100));
        }
        if (status.message) {
          setProgressMessage(status.message);
        }
        if (status.status === 'complete') {
          done = true;
          setProgressPercent(100);
          setProgressMessage(status.message || t('benchmark.progressComplete'));
          setResults(status.result?.results || []);
        } else if (status.status === 'error' || status.status === 'cancelled') {
          throw new Error(status.error || status.message || t('benchmark.progressFailed'));
        }
      }
    } catch (err: unknown) {
      console.error('Batch benchmark failed:', err);
      setRunError(getErrorMessage(err, t('benchmark.progressFailed')));
      setProgressMessage('');
    } finally {
      setRunning(false);
    }
  };

  const exportCSV = () => {
    if (results.length === 0) return;
    const headers = ['Model', 'Disk Size', 'Peak Memory (MB)', 'Tokens/s', 'TTFT (ms)', 'Perplexity'];
    const rows = results
      .filter((r) => r.success && r.result)
      .map((r) => [
        r.label,
        r.result!.disk_size_bytes,
        r.result!.memory_peak_mb?.toFixed(1) || '',
        r.result!.tokens_per_second?.toFixed(2) || '',
        r.result!.time_to_first_token_ms?.toFixed(1) || '',
        r.result!.perplexity?.toFixed(3) || '',
      ]);
    const csv = [headers, ...rows].map((row) => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = samplePreview ? 'benchmark_sample_preview.csv' : 'benchmark_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 p-6">
      <PageHeader
        title={t('pro.benchmark.title')}
        description={t('pro.benchmark.desc')}
      />

      <InsightPanel insights={benchmarkInsights} />

      {/* 4-card identity strip — cohort overview (playbook §7.2 / §9.1 / §9.2) */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<BarChart3 size={16} />}
          label={t('benchmark.cardRuns')}
          value={summary.totalRuns === 0 ? t('benchmark.noRuns') : `${summary.successCount} / ${summary.totalRuns}`}
          hint={summary.totalRuns === 0
            ? t('benchmark.runsHint')
            : `${summary.modelCount} ${t('benchmark.distinctModels')}${summary.errorCount > 0 ? ` · ${summary.errorCount} ${t('benchmark.errors')}` : ''}`}
          tone={summary.totalRuns === 0 ? 'neutral' : summary.errorCount > 0 ? 'amber' : 'indigo'}
        />
        <IdentityCard
          icon={<Zap size={16} />}
          label={t('benchmark.cardPeakTPS')}
          value={summary.bestTPS > 0 ? `${summary.bestTPS.toFixed(1)} tok/s` : '—'}
          hint={summary.bestTPS > 0
            ? `${summary.bestTPSLabel} · ${bucketLabel(summary.bestTPSBucket, locale)}`
            : t('benchmark.noTPS')}
          tone={summary.bestTPSBucket === 'blazing' || summary.bestTPSBucket === 'fast'
            ? 'emerald'
            : summary.bestTPSBucket === 'ok'
              ? 'indigo'
              : summary.bestTPSBucket === 'slow'
                ? 'amber'
                : 'neutral'}
        />
        <IdentityCard
          icon={<Trophy size={16} />}
          label={t('benchmark.cardBestPPL')}
          value={summary.bestPPL > 0 ? summary.bestPPL.toFixed(2) : '—'}
          hint={summary.bestPPL > 0
            ? `${summary.bestPPLLabel} · ${bucketLabel(summary.bestPPLBucket, locale)}`
            : t('benchmark.noPPL')}
          tone={summary.bestPPLBucket === 'excellent'
            ? 'emerald'
            : summary.bestPPLBucket === 'good'
              ? 'indigo'
              : summary.bestPPLBucket === 'concerning'
                ? 'amber'
                : summary.bestPPLBucket === 'broken'
                  ? 'red'
                  : 'neutral'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('benchmark.cardSovereignty')}
          value={t('benchmark.zeroTelemetry')}
          hint={t('benchmark.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {/* Cohort risk banner — playbook §8.1 risk surface */}
      {cohortRisk.level !== 'safe' && (
        <div className={`rounded-lg border px-3 py-2 text-xs ${COHORT_RISK_BANNER_CLASS[cohortRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {cohortRisk.level === 'danger' ? t('benchmark.riskDanger') : t('benchmark.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? cohortRisk.reasonZh : cohortRisk.reason}
        </div>
      )}

      <WorkflowGuide
        title={t('benchmark.workflowTitle')}
        description={t('benchmark.workflowDesc')}
        badges={[
          { label: t('benchmark.runSettings') },
          { label: t('benchmark.queueCount', { count: items.length }), tone: 'indigo' },
        ]}
        steps={[
          {
            icon: <FolderOpen size={16} />,
            title: t('benchmark.stepAddTitle'),
            description: t('benchmark.stepAddDesc'),
          },
          {
            icon: <Gauge size={16} />,
            title: t('benchmark.stepRunTitle'),
            description: t('benchmark.stepRunDesc'),
          },
          {
            icon: <ListChecks size={16} />,
            title: t('benchmark.stepReadTitle'),
            description: t('benchmark.stepReadDesc'),
          },
        ]}
        footerItems={[
          t('benchmark.metricTps'),
          t('benchmark.metricTtft'),
          t('benchmark.metricPpl'),
        ]}
      >
        <div className="flex flex-col gap-3 rounded-lg border border-indigo-100 bg-indigo-50/60 px-3 py-3 dark:border-indigo-400/30 dark:bg-indigo-950/50 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 gap-2">
            <Sparkles size={16} className="mt-0.5 shrink-0 text-indigo-600 dark:text-indigo-300" />
            <div className="min-w-0">
              <p className="text-xs font-semibold text-indigo-900 dark:text-indigo-100">{t('benchmark.sampleTitle')}</p>
              <p className="mt-1 text-xs leading-relaxed text-indigo-700 dark:text-indigo-200">{t('benchmark.sampleDesc')}</p>
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
              {samplePreview ? t('benchmark.sampleReload') : t('benchmark.sampleLoad')}
            </button>
            {samplePreview && (
              <button
                type="button"
                onClick={clearSamplePreview}
                className="inline-flex items-center justify-center rounded-lg border border-indigo-200 bg-white/70 px-3 py-2 text-xs font-medium text-indigo-700 transition-colors hover:bg-white dark:border-indigo-400/40 dark:bg-stone-950/80 dark:text-indigo-200 dark:hover:bg-stone-900"
              >
                {t('benchmark.sampleClear')}
              </button>
            )}
          </div>
        </div>
      </WorkflowGuide>

      {samplePreview && (
        <div className="rounded-lg border border-sky-200 bg-sky-50 px-3 py-2 text-xs leading-relaxed text-sky-800 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-200">
          <span className="font-semibold">{t('benchmark.sampleBannerTitle')}</span>
          {' '}{t('benchmark.sampleBannerDesc')}
        </div>
      )}

      {/* AI Brief — brain LLM speaks as itself about the cohort */}
      {model && (
        <ModelBriefCard
          label={t('benchmark.briefTitle')}
          status={briefChat.status || model.model_name}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('benchmark.briefEmpty')}
          streamingText={t('benchmark.briefThinking')}
          refreshTitle={t('benchmark.briefRefire')}
          prompts={cohortPrompts}
          onRefresh={() => {
            briefFiredForRef.current = null;
            briefChat.reset();
            briefChat.send(buildBenchmarkAutoBrief(summary, locale));
          }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {/* Model list */}
      <div className="space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">{t('benchmark.queueTitle')}</h2>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{t('benchmark.queueHint')}</p>
          </div>
          {model && !loadedModelQueued && (
            <button
              type="button"
              onClick={addLoadedModel}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-100 dark:border-indigo-500/30 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
            >
              <Brain size={14} />
              {t('benchmark.addLoadedModel')}
            </button>
          )}
        </div>

        {items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 px-5 py-6 text-center dark:border-gray-700 dark:bg-gray-900/50">
            <FolderOpen size={20} className="mx-auto text-gray-400 dark:text-gray-500" />
            <p className="mt-2 text-sm font-medium text-gray-800 dark:text-gray-100">{t('benchmark.emptyQueueTitle')}</p>
            <p className="mx-auto mt-1 max-w-2xl text-xs leading-relaxed text-gray-500 dark:text-gray-400">
              {t('benchmark.emptyQueueDesc')}
            </p>
            <button
              type="button"
              onClick={addItem}
              className="mt-4 inline-flex items-center justify-center gap-2 rounded-lg bg-gray-900 px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-200"
            >
              <Plus size={14} />
              {t('pro.benchmark.addModel')}
            </button>
            <button
              type="button"
              onClick={loadSamplePreview}
              className="ml-2 mt-4 inline-flex items-center justify-center gap-2 rounded-lg border border-indigo-200 bg-white px-3 py-2 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-50 dark:border-indigo-500/30 dark:bg-gray-900 dark:text-indigo-300 dark:hover:bg-indigo-500/10"
            >
              <Sparkles size={14} />
              {t('benchmark.sampleLoad')}
            </button>
          </div>
        ) : (
          <>
            {items.map((item, idx) => (
              <div key={item.model_dir} className="flex items-center gap-3 rounded-xl border border-gray-200 bg-white px-5 py-3 dark:border-gray-800 dark:bg-gray-900">
                <BarChart3 size={16} className="shrink-0 text-gray-400 dark:text-gray-500" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">{item.label}</p>
                  <p className="truncate text-xs text-gray-500 dark:text-gray-400">{item.model_dir}</p>
                </div>
                <button
                  type="button"
                  onClick={() => removeItem(idx)}
                  className="shrink-0 rounded-lg p-1.5 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-600 dark:hover:bg-gray-800 dark:hover:text-gray-300"
                  aria-label={t('benchmark.removeModel')}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={addItem}
              className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-gray-300 py-3 text-sm text-gray-500 transition-colors hover:border-gray-400 hover:text-gray-700 dark:border-gray-700 dark:hover:border-gray-600 dark:hover:text-gray-300"
            >
              <Plus size={16} />
              {t('pro.benchmark.addModel')}
            </button>
          </>
        )}
      </div>

      {/* Actions */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <button
            type="button"
            onClick={handleRun}
            disabled={running || items.length === 0 || samplePreview}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-gray-900 px-5 py-2.5 text-sm font-medium text-white transition-all duration-200 hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-200"
          >
            {running ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
            {running ? t('pro.benchmark.running') : t('pro.benchmark.run')}
          </button>
          {results.length > 0 && (
            <button
              type="button"
              onClick={exportCSV}
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-200 px-4 py-2.5 text-sm text-gray-700 transition-colors hover:bg-gray-50 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
            >
              <Download size={16} />
              CSV
            </button>
          )}
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {samplePreview
              ? t('benchmark.sampleRunHint')
              : items.length === 0
                ? t('benchmark.runDisabledHint')
                : t('benchmark.runReadyHint')}
          </p>
        </div>

        {running && (
          <div className="mt-4">
            <div className="mb-1 flex items-center justify-between gap-3 text-xs text-gray-500 dark:text-gray-400">
              <span className="truncate">{progressMessage || t('pro.benchmark.running')}</span>
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
              <p className="font-semibold">{t('benchmark.errorTitle')}</p>
              <p className="mt-0.5">{runError}</p>
            </div>
          </div>
        )}
      </div>

      {/* Results table */}
      {results.length > 0 && (
        <div className="overflow-hidden rounded-2xl border border-gray-200 dark:border-gray-800">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400">{t('benchmark.tableModel')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('benchmark.tableSize')}</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">Tokens/s</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">TTFT</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">PPL</th>
                <th className="px-4 py-3 text-right font-medium text-gray-600 dark:text-gray-400">{t('benchmark.tableStatus')}</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => (
                <tr key={idx} className="border-t border-gray-100 dark:border-gray-800/60">
                  <td className="px-4 py-2.5 font-medium text-gray-900 dark:text-gray-100">{r.label}</td>
                  {r.success && r.result ? (
                    <>
                      <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                        {formatSize(r.result.disk_size_bytes)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                        {r.result.tokens_per_second?.toFixed(1) || '--'}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                        {r.result.time_to_first_token_ms?.toFixed(0) || '--'} ms
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">
                        {r.result.perplexity?.toFixed(2) || '--'}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <span className="rounded-full bg-green-50 px-2.5 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/30 dark:text-green-400">
                          OK
                        </span>
                      </td>
                    </>
                  ) : (
                    <>
                      <td colSpan={4} className="px-4 py-2.5 text-gray-500 dark:text-gray-400">{r.error || 'Failed'}</td>
                      <td className="px-4 py-2.5 text-right">
                        <span className="rounded-full bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700 dark:bg-red-900/30 dark:text-red-400">
                          Error
                        </span>
                      </td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {pickerOpen && (
        <FileBrowser
          onSelect={handlePickModel}
          onCancel={() => setPickerOpen(false)}
        />
      )}

      {/* Ask Model FAB — brain speaks about cohort */}
      {model && (
        <>
          {!drawerOpen && (
            <AskModelFab
              label={t('benchmark.askFab')}
              modelName={model.model_name}
              icon={<Brain size={15} />}
              onClick={() => setDrawerOpen(true)}
            />
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Brain size={13} className="text-indigo-500" />
                  {t('benchmark.askDrawerTitle')}
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
                  <p className="text-xs text-stone-400">{t('benchmark.askDrawerHint')}</p>
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
                    placeholder={t('benchmark.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('benchmark.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('benchmark.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {cohortPrompts.slice(0, 4).map((p) => (
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
