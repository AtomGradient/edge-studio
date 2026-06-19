// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { getOptSuggestions, executeOptimization, getDevices } from '@/api/endpoints';
import type {
  OptimizationReport,
  OptimizationSuggestion,
  ExecutionResult,
  DeviceProfile,
} from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { IdentityCard } from '@/components/common/IdentityCard';
import { cn, formatSize, formatParamCount } from '@/lib/utils';
import {
  Sparkles, Send, X, RotateCcw, RefreshCw, Cpu, Layers, Lightbulb, AlertTriangle,
  Loader2,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  buildOptimizationContextSnippet, buildOptimizationAutoBrief,
  getOptimizationSuggestedPrompts, deriveOptCounts,
} from '@/lib/optimizationInsights';

const PRIORITY_STYLES: Record<string, string> = {
  high: 'bg-red-100 text-red-700',
  medium: 'bg-yellow-100 text-yellow-700',
  low: 'bg-green-100 text-green-700',
};

const RISK_STYLES: Record<string, string> = {
  high: 'bg-red-100 text-red-700',
  medium: 'bg-yellow-100 text-yellow-700',
  low: 'bg-green-100 text-green-700',
};

const CATEGORY_LABELS: Record<string, string> = {
  neuron_pruning: 'Neuron Pruning',
  layer_pruning: 'Layer Pruning',
  quantization: 'Quantization',
  vocab_pruning: 'Vocabulary Pruning',
  head_pruning: 'Head Pruning',
  embedding_quantization: 'Embedding Quantization',
  device_compatibility: 'Device Compatibility',
};

export default function OptimizationAdvisor() {
  const model = useModelStore((s) => s.currentModel);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [report, setReport] = useState<OptimizationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Execution state
  const [execTaskId, setExecTaskId] = useState<string | null>(null);
  const [execResults, setExecResults] = useState<ExecutionResult[]>([]);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  // Device compatibility
  const [devices, setDevices] = useState<DeviceProfile[] | null>(null);
  const [selectedDevices, setSelectedDevices] = useState<string[]>([
    'iPhone 17 Pro',
    'iPad Pro M5 (16GB)',
    'MacBook Air M5 (16GB)',
  ]);

  // AI Brief / Ask FAB
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // Derived
  const counts = useMemo(() => (report ? deriveOptCounts(report) : null), [report]);
  const suggestedPrompts = useMemo(
    () => getOptimizationSuggestedPrompts(report, locale),
    [report, locale],
  );

  // System prompt (model-self + report context)
  const systemPrompt = useMemo(() => {
    if (!model) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    if (!report) return base;
    return base + '\n\n' + buildOptimizationContextSnippet(report, model);
  }, [model, locale, report]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt,
    maxTokens: 700,
    temperature: 0.55,
  });

  const handleLoadSuggestions = useCallback(async () => {
    if (!model) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getOptSuggestions(model.model_id);
      setReport(data);
      if (!devices) {
        try {
          const d = await getDevices();
          setDevices(d);
        } catch { /* ignore */ }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load suggestions');
    } finally {
      setLoading(false);
    }
  }, [model, devices]);

  // Reset state when model switches
  useEffect(() => {
    setReport(null);
    setExecResults([]);
    setExpanded(new Set());
    setAskOpen(false);
    briefFiredForRef.current = null;
    chat.reset();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fetch on model load (no more "Analyze Model" button required)
  useEffect(() => {
    if (!model || report || loading) return;
    handleLoadSuggestions();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire AI Brief once report is ready
  useEffect(() => {
    if (!model || !report) return;
    if (briefFiredForRef.current === model.model_id) return;
    if (chat.streaming) return;
    briefFiredForRef.current = model.model_id;
    const id = window.setTimeout(() => chat.send(buildOptimizationAutoBrief(report, locale)), 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, !!report, locale]);

  const handleExecute = useCallback(async (suggestion: OptimizationSuggestion) => {
    if (!model) return;
    try {
      const { task_id } = await executeOptimization(
        model.model_id,
        suggestion.category,
        suggestion.params,
      );
      setExecTaskId(task_id);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Execution failed');
    }
  }, [model]);

  const toggleExpand = useCallback((i: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });
  }, []);

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
    <div className="relative pb-12">
      <PageHeader
        title={t('opt.title')}
        description={model.model_name}
      />

      {/* 4-card identity strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <IdentityCard
          icon={<Cpu size={16} />}
          label={t('opt.idScale')}
          value={`${formatParamCount(model.total_params)} · ${model.quantization?.bits ?? '?'}-bit`}
          hint={`${formatSize(report?.model_size_bytes ?? model.total_size_bytes)} on disk`}
          tone="indigo"
        />
        <IdentityCard
          icon={<Lightbulb size={16} />}
          label={t('opt.idApplicable')}
          value={counts ? `${counts.applicable}` : '—'}
          hint={counts ? `${counts.requiresData} require profile/trace` : 'Computing…'}
          tone={counts && counts.applicable > 0 ? 'emerald' : 'neutral'}
        />
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('opt.idEstSaving')}
          value={counts ? formatSize(counts.totalSavingBytes) : '—'}
          hint="Sum of estimated savings across applicable suggestions only"
          tone={counts && counts.totalSavingBytes > 0 ? 'amber' : 'neutral'}
        />
        <IdentityCard
          icon={<AlertTriangle size={16} />}
          label={t('opt.idHighPrio')}
          value={counts ? String((counts.byPriority['high'] ?? 0)) : '—'}
          hint="HIGH priority suggestions across both buckets"
          tone={counts && (counts.byPriority['high'] ?? 0) > 0 ? 'red' : 'neutral'}
        />
      </div>

      {/* AI Brief */}
      {report && (
        <div className="mb-4 rounded-xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
                {t('opt.briefLabel')}
              </span>
              {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
            </div>
            {chat.text && !chat.streaming && (
              <button
                onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildOptimizationAutoBrief(report, locale)); }}
                className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
                title={t('weights.briefRefresh')}
              >
                <RotateCcw size={12} />
              </button>
            )}
          </div>
          <div className="text-sm text-gray-700 dark:text-stone-300">
            {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
            {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('opt.briefPending')}</span>}
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
      )}

      {/* Refresh */}
      <div className="mb-4 flex items-center justify-between">
        <p className="text-xs text-gray-400 dark:text-stone-500">
          {loading ? t('opt.analyzing') : report ? t('opt.upToDate') : ''}
        </p>
        <button
          onClick={handleLoadSuggestions}
          disabled={loading}
          className={cn(
            'flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
            loading
              ? 'bg-gray-100 text-gray-400 dark:bg-stone-800 dark:text-stone-500'
              : 'border border-indigo-200 text-indigo-600 hover:bg-indigo-50 dark:border-indigo-500/30 dark:text-indigo-400 dark:hover:bg-indigo-500/10',
          )}
        >
          <RefreshCw size={11} className={loading ? 'animate-spin' : ''} />
          {report ? t('opt.refresh') : t('opt.analyze')}
        </button>
      </div>
      {error && <p className="mb-3 text-sm text-red-500">{error}</p>}

      {/* Progress overlay for execution */}
      {execTaskId && (
        <ProgressOverlay
          taskId={execTaskId}
          title="Executing Optimization"
          onComplete={(result) => {
            if (result) setExecResults((prev) => [...prev, result as ExecutionResult]);
            setExecTaskId(null);
          }}
          onError={() => setExecTaskId(null)}
          onClose={() => setExecTaskId(null)}
        />
      )}

      {report && (
        <>
          {/* Applicable suggestions */}
          {report.suggestions.length > 0 && (
            <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">
                Optimization Suggestions ({report.suggestions.length})
              </h3>
              <div className="space-y-2">
                {report.suggestions.map((s, i) => (
                  <SuggestionCard
                    key={i}
                    suggestion={s}
                    isExpanded={expanded.has(i)}
                    onToggle={() => toggleExpand(i)}
                    onExecute={() => handleExecute(s)}
                    isExecuting={!!execTaskId}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Suggestions requiring more data */}
          {report.requires_data.length > 0 && (
            <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">
                Requires Additional Data ({report.requires_data.length})
              </h3>
              <p className="mb-3 text-xs text-gray-400">
                Load activation profile or run inference tracer to unlock these suggestions.
              </p>
              <div className="space-y-2">
                {report.requires_data.map((s, i) => (
                  <div
                    key={i}
                    className="rounded-lg border border-gray-100 p-3 opacity-60"
                  >
                    <div className="flex items-center gap-2">
                      <PriorityBadge priority={s.priority} />
                      <span className="text-sm font-medium text-gray-700">{s.title}</span>
                      <CategoryBadge category={s.category} />
                    </div>
                    <p className="mt-1 text-xs text-gray-500">{s.description}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Device Compatibility */}
          {devices && devices.length > 0 && (
            <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
              <h3 className="mb-3 text-sm font-semibold text-gray-700">Device Compatibility</h3>
              <div className="mb-3 flex flex-wrap gap-2">
                {devices.map((d) => (
                  <button
                    key={d.name}
                    onClick={() => toggleDevice(d.name)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                      selectedDevices.includes(d.name)
                        ? 'bg-indigo-100 text-indigo-700'
                        : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                    }`}
                  >
                    {d.name}
                  </button>
                ))}
              </div>
              <DeviceTable
                devices={devices.filter((d) => selectedDevices.includes(d.name))}
                modelSizeBytes={report.model_size_bytes}
              />
            </div>
          )}

          {/* Before/After Comparison */}
          {execResults.length > 0 && (
            <BeforeAfterPanel results={execResults} originalBytes={report.model_size_bytes} />
          )}
        </>
      )}

      {/* Ask Model FAB */}
      {report && (
        <>
          {!askOpen && (
            <button
              onClick={() => setAskOpen(true)}
              className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
              title={t('opt.askModel')}
            >
              <Sparkles size={16} /> {t('opt.askModel')}
            </button>
          )}
          {askOpen && (
            <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
                <div className="flex items-center gap-2">
                  <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
                  <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('opt.askModel')}</span>
                </div>
                <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
                  <X size={14} />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 text-sm">
                {chat.text ? <MarkdownContent content={chat.text} /> : (
                  <p className="text-gray-400 dark:text-stone-500 text-xs">{t('opt.askEmpty')}</p>
                )}
              </div>
              <form
                onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
                className="flex gap-2 border-t border-gray-100 dark:border-stone-800 p-3"
              >
                <input
                  value={askInput}
                  onChange={(e) => setAskInput(e.target.value)}
                  placeholder={t('opt.askPlaceholder')}
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

function SuggestionCard({
  suggestion: s,
  isExpanded,
  onToggle,
  onExecute,
  isExecuting,
}: {
  suggestion: OptimizationSuggestion;
  isExpanded: boolean;
  onToggle: () => void;
  onExecute: () => void;
  isExecuting: boolean;
}) {
  return (
    <div className="rounded-lg border border-gray-100">
      <button
        onClick={onToggle}
        className="flex w-full items-center justify-between px-3 py-2.5 text-left hover:bg-gray-50"
      >
        <div className="flex items-center gap-2">
          <PriorityBadge priority={s.priority} />
          <span className="text-sm font-medium text-gray-800">{s.title}</span>
          <CategoryBadge category={s.category} />
        </div>
        <span className="text-xs text-gray-400">{isExpanded ? '▲' : '▼'}</span>
      </button>
      {isExpanded && (
        <div className="border-t border-gray-100 px-3 py-3 space-y-3">
          <p className="text-xs text-gray-600">{s.description}</p>
          <div className="flex flex-wrap gap-3 text-xs">
            <div>
              <span className="text-gray-400">Estimated saving: </span>
              <span className="font-medium text-gray-700">{s.estimated_saving}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-400">Risk: </span>
              <span
                className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                  RISK_STYLES[s.risk_level] || 'bg-gray-100 text-gray-600'
                }`}
              >
                {s.risk_level.toUpperCase()}
              </span>
            </div>
          </div>
          {Object.keys(s.params).length > 0 && (
            <div className="rounded-lg bg-gray-50 p-2">
              <p className="mb-1 text-xs text-gray-400">Parameters</p>
              <div className="grid grid-cols-2 gap-1 text-xs">
                {Object.entries(s.params).map(([k, v]) => (
                  <div key={k}>
                    <span className="text-gray-500">{k}: </span>
                    <span className="font-mono text-gray-700">{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <button
            onClick={onExecute}
            disabled={isExecuting}
            className="rounded-lg bg-indigo-500 px-4 py-1.5 text-xs font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
          >
            Execute
          </button>
        </div>
      )}
    </div>
  );
}

function PriorityBadge({ priority }: { priority: string }) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-xs font-bold ${
        PRIORITY_STYLES[priority] || 'bg-gray-100 text-gray-600'
      }`}
    >
      {priority === 'high' ? 'HIGH' : priority === 'medium' ? 'MED' : 'LOW'}
    </span>
  );
}

function CategoryBadge({ category }: { category: string }) {
  return (
    <span className="rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-500">
      {CATEGORY_LABELS[category] || category}
    </span>
  );
}

function DeviceTable({
  devices,
  modelSizeBytes,
}: {
  devices: DeviceProfile[];
  modelSizeBytes: number;
}) {
  const modelGB = modelSizeBytes / 1e9;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50">
          <tr className="text-left text-gray-500">
            <th className="px-3 py-2 font-medium">Device</th>
            <th className="px-3 py-2 font-medium">Chip</th>
            <th className="px-3 py-2 font-medium text-right">RAM</th>
            <th className="px-3 py-2 font-medium text-right">Available</th>
            <th className="px-3 py-2 font-medium text-right">Max Model</th>
            <th className="px-3 py-2 font-medium text-center">Fits</th>
            <th className="px-3 py-2 font-medium text-right">Headroom</th>
          </tr>
        </thead>
        <tbody>
          {devices.map((d) => {
            const fits = modelGB <= d.max_model_size_gb;
            const headroom = d.max_model_size_gb - modelGB;
            return (
              <tr key={d.name} className="border-t border-gray-100 hover:bg-gray-50">
                <td className="px-3 py-2 font-medium text-gray-700">{d.name}</td>
                <td className="px-3 py-2 text-gray-500">{d.chip}</td>
                <td className="px-3 py-2 text-right">{d.ram_gb} GB</td>
                <td className="px-3 py-2 text-right">{d.available_ram_gb.toFixed(1)} GB</td>
                <td className="px-3 py-2 text-right">{d.max_model_size_gb.toFixed(1)} GB</td>
                <td className="px-3 py-2 text-center">
                  <span
                    className={`inline-block h-2.5 w-2.5 rounded-full ${
                      fits ? 'bg-green-500' : 'bg-red-500'
                    }`}
                  />
                </td>
                <td
                  className={`px-3 py-2 text-right font-medium ${
                    fits ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {headroom >= 0 ? '+' : ''}
                  {headroom.toFixed(1)} GB
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BeforeAfterPanel({
  results,
  originalBytes,
}: {
  results: ExecutionResult[];
  originalBytes: number;
}) {
  const totalSaving = results.reduce((s, r) => s + r.saving_bytes, 0);
  const latestSize = results.length > 0 ? results[results.length - 1].result_size_bytes : originalBytes;

  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">Before / After Comparison</h3>
      <div className="mb-4 grid grid-cols-4 gap-4">
        <div className="rounded-lg bg-gray-50 p-3">
          <p className="text-xs text-gray-500">Original Size</p>
          <p className="text-lg font-bold">{formatSize(originalBytes)}</p>
        </div>
        <div className="rounded-lg bg-gray-50 p-3">
          <p className="text-xs text-gray-500">Current Size</p>
          <p className="text-lg font-bold text-indigo-600">{formatSize(latestSize)}</p>
        </div>
        <div className="rounded-lg bg-gray-50 p-3">
          <p className="text-xs text-gray-500">Total Saved</p>
          <p className="text-lg font-bold text-green-600">{formatSize(totalSaving)}</p>
        </div>
        <div className="rounded-lg bg-gray-50 p-3">
          <p className="text-xs text-gray-500">Reduction</p>
          <p className="text-lg font-bold text-green-600">
            {((totalSaving / Math.max(originalBytes, 1)) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Steps table */}
      <table className="w-full text-xs">
        <thead className="bg-gray-50">
          <tr className="text-left text-gray-500">
            <th className="px-3 py-1.5 font-medium">Operation</th>
            <th className="px-3 py-1.5 font-medium text-center">Status</th>
            <th className="px-3 py-1.5 font-medium text-right">Before</th>
            <th className="px-3 py-1.5 font-medium text-right">After</th>
            <th className="px-3 py-1.5 font-medium text-right">Saved</th>
            <th className="px-3 py-1.5 font-medium text-right">Duration</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className="border-t border-gray-100">
              <td className="px-3 py-1.5 font-medium text-gray-700">
                {CATEGORY_LABELS[r.operation] || r.operation}
              </td>
              <td className="px-3 py-1.5 text-center">
                {r.success ? (
                  <span className="text-green-600 font-medium">OK</span>
                ) : (
                  <span className="text-red-600 font-medium">FAIL</span>
                )}
              </td>
              <td className="px-3 py-1.5 text-right">{formatSize(r.original_size_bytes)}</td>
              <td className="px-3 py-1.5 text-right">{formatSize(r.result_size_bytes)}</td>
              <td className="px-3 py-1.5 text-right text-green-600">
                {formatSize(r.saving_bytes)}
              </td>
              <td className="px-3 py-1.5 text-right">{r.duration_seconds.toFixed(1)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
