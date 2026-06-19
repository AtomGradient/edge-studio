// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { runPipeline, getOptSuggestions, runBenchmark } from '@/api/endpoints';
import type {
  PipelineRunResult, OptimizationSuggestion, ModelInfo,
  BenchmarkRunResult, BenchmarkGuidance,
} from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { MetricCards } from '@/components/data/MetricCards';
import { EmptyState } from '@/components/common/EmptyState';
import { NextStepBanner } from '@/components/common/NextStepBanner';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { IdentityCard } from '@/components/common/IdentityCard';
import { useToastStore } from '@/stores/toastStore';
import { cn, formatSize, formatParamCount } from '@/lib/utils';
import {
  Plus, Trash2, ChevronUp, ChevronDown, Play, Import, FolderOpen,
  BarChart2, CheckCircle, AlertTriangle, XCircle, Download, MessageCircle,
  Sparkles, Send, X, RotateCcw, Loader2, ShieldCheck, Workflow as WorkflowIcon,
  Cpu, Layers,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt, deriveModelFacts } from '@/lib/chatPrompts';
import {
  summarizePipeline, buildPipelineContextSnippet, buildPipelineAutoBrief,
  getPipelineSuggestedPrompts, RISK_CHIP_CLASS, RISK_BORDER_CLASS,
  type OperationRisk, type PipelineSummary,
} from '@/lib/pipelineInsights';

// --- Constants ---

const OPERATIONS = [
  { value: 'neuron_pruning', label: 'Neuron Pruning' },
  { value: 'layer_pruning', label: 'Layer Pruning' },
  { value: 'quantization', label: 'Quantization' },
  { value: 'vocab_pruning', label: 'Vocab Pruning' },
  { value: 'embedding_quantization', label: 'Embedding Quantization' },
] as const;

const OP_LABELS: Record<string, string> = Object.fromEntries(
  OPERATIONS.map((o) => [o.value, o.label]),
);

interface ParamDef {
  key: string;
  label: string;
  type: 'number' | 'int_array';
  default: number | number[];
  min?: number;
  max?: number;
  step?: number;
}

const OP_PARAMS: Record<string, ParamDef[]> = {
  neuron_pruning: [
    { key: 'threshold', label: 'Threshold', type: 'number', default: 0.1, min: 0, max: 1, step: 0.01 },
    { key: 'max_reduction', label: 'Max Reduction', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 },
    { key: 'protected_layers', label: 'Protected Layers', type: 'int_array', default: [] },
  ],
  layer_pruning: [
    { key: 'layers_to_remove', label: 'Layers to Remove', type: 'int_array', default: [] },
  ],
  quantization: [
    { key: 'bits', label: 'Bits', type: 'number', default: 4, min: 2, max: 8, step: 1 },
    { key: 'group_size', label: 'Group Size', type: 'number', default: 64, min: 32, max: 128, step: 32 },
  ],
  vocab_pruning: [],
  embedding_quantization: [],
};

// --- Types ---

interface PipelineStepConfig {
  id: string;
  operation: string;
  params: Record<string, unknown>;
}

let _nextId = 0;
function makeId() {
  return `step_${++_nextId}_${Date.now()}`;
}

function createStep(operation: string): PipelineStepConfig {
  const defs = OP_PARAMS[operation] || [];
  const params: Record<string, unknown> = {};
  for (const d of defs) {
    params[d.key] = d.default;
  }
  return { id: makeId(), operation, params };
}

// --- Component ---

export default function OptimizationPipeline() {
  const model = useModelStore((s) => s.currentModel);
  const setCurrentModel = useModelStore((s) => s.setCurrentModel);
  const profileSummary = useModelStore((s) => s.profileSummary);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const addToast = useToastStore((s) => s.addToast);
  const hasProfile = !!profileSummary;

  const [steps, setSteps] = useState<PipelineStepConfig[]>([]);
  const [skipValidation, setSkipValidation] = useState(false);
  const [pplText, setPplText] = useState('');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<PipelineRunResult | null>(null);
  const [importLoading, setImportLoading] = useState(false);

  // AI brief / Ask FAB
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // Risk assessment
  const summary: PipelineSummary | null = useMemo(
    () => (model ? summarizePipeline(steps.map((s) => ({ operation: s.operation, params: s.params })), model, hasProfile) : null),
    [model, steps, hasProfile],
  );
  const facts = useMemo(() => (model ? deriveModelFacts(model) : null), [model]);

  // System prompt (model self + pipeline state context)
  const systemPrompt = useMemo(() => {
    if (!model) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    return base + '\n\n' + buildPipelineContextSnippet(model,
      steps.map((s) => ({ operation: s.operation, params: s.params })),
      summary, hasProfile);
  }, [model, locale, steps, summary, hasProfile]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt,
    maxTokens: 700,
    temperature: 0.55,
  });

  const suggestedPrompts = useMemo(
    () => (model ? getPipelineSuggestedPrompts(model,
      steps.map((s) => ({ operation: s.operation, params: s.params })),
      summary, hasProfile, locale) : []),
    [model, steps, summary, hasProfile, locale],
  );

  // --- Step management ---

  const addStep = useCallback((op: string) => {
    setSteps((prev) => [...prev, createStep(op)]);
  }, []);

  const removeStep = useCallback((id: string) => {
    setSteps((prev) => prev.filter((s) => s.id !== id));
  }, []);

  const moveStep = useCallback((id: string, dir: -1 | 1) => {
    setSteps((prev) => {
      const idx = prev.findIndex((s) => s.id === id);
      if (idx < 0) return prev;
      const next = [...prev];
      const target = idx + dir;
      if (target < 0 || target >= next.length) return prev;
      [next[idx], next[target]] = [next[target], next[idx]];
      return next;
    });
  }, []);

  const updateParam = useCallback((id: string, key: string, value: unknown) => {
    setSteps((prev) =>
      prev.map((s) => (s.id === id ? { ...s, params: { ...s.params, [key]: value } } : s)),
    );
  }, []);

  // --- Import from Advisor ---

  const handleImportFromAdvisor = useCallback(async () => {
    if (!model) return;
    setImportLoading(true);
    try {
      const report = await getOptSuggestions(model.model_id);
      const applicable = report.suggestions.filter(
        (s: OptimizationSuggestion) =>
          s.applicable &&
          ['neuron_pruning', 'layer_pruning', 'quantization', 'vocab_pruning', 'embedding_quantization'].includes(s.category),
      );
      const newSteps = applicable.map((s: OptimizationSuggestion) =>
        createStep(s.category),
      );
      if (newSteps.length > 0) {
        setSteps((prev) => [...prev, ...newSteps]);
      }
    } catch {
      addToast('Failed to import optimization suggestions.', 'error');
    } finally {
      setImportLoading(false);
    }
  }, [model, addToast]);

  // --- Execute ---

  const handleRun = useCallback(async () => {
    if (!model || steps.length === 0) return;
    setResult(null);
    try {
      const { task_id } = await runPipeline(
        model.model_id,
        steps.map((s) => ({ operation: s.operation, params: s.params })),
        pplText,
        skipValidation,
      );
      setTaskId(task_id);
    } catch {
      // error handling via ProgressOverlay
    }
  }, [model, steps, pplText, skipValidation]);

  const handleComplete = useCallback((raw: unknown) => {
    setTaskId(null);
    if (raw && typeof raw === 'object') {
      setResult(raw as PipelineRunResult);
    }
  }, []);

  // --- Load optimized model ---

  const handleLoadOptimized = useCallback(() => {
    if (result?.optimized_model_info) {
      setCurrentModel(result.optimized_model_info as ModelInfo);
    }
  }, [result, setCurrentModel]);

  // Reset chat / brief on model change
  useEffect(() => {
    briefFiredForRef.current = null;
    chat.reset();
    setAskOpen(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire brief when (model + step config + summary) changes meaningfully
  useEffect(() => {
    if (!model || !summary) return;
    const key = `${model.model_id}:${steps.length}:${summary.totalRiskScore}:${summary.orderWarnings.length}`;
    if (briefFiredForRef.current === key) return;
    if (chat.streaming) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => {
      chat.send(buildPipelineAutoBrief(model,
        steps.map((s) => ({ operation: s.operation, params: s.params })),
        summary, hasProfile, locale));
    }, 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, steps.length, summary?.totalRiskScore, summary?.orderWarnings.length, locale]);

  const handleSuggested = useCallback((q: string) => {
    setAskOpen(true);
    chat.send(q);
  }, [chat]);

  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  return (
    <div className="space-y-5 pb-12 relative">
      <PageHeader
        title={t('pipeline.title')}
        description={model.model_name}
      />

      {/* 4-card identity strip */}
      {facts && summary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <IdentityCard
            icon={<Cpu size={16} />}
            label={t('pipeline.idScale')}
            value={`${formatParamCount(facts.totalParams)} · ${facts.quantBits || '?'}-bit`}
            hint={`${formatSize(facts.totalSizeBytes)} on disk`}
            tone="indigo"
          />
          <IdentityCard
            icon={<ShieldCheck size={16} />}
            label={t('pipeline.idProfile')}
            value={hasProfile ? t('pipeline.profileReady') : t('pipeline.profileMissing')}
            hint={hasProfile ? 'neuron_pruning unlocked' : 'neuron_pruning will prune random neurons — run /activation first'}
            tone={hasProfile ? 'emerald' : 'amber'}
          />
          <IdentityCard
            icon={<WorkflowIcon size={16} />}
            label={t('pipeline.idSteps')}
            value={`${steps.length}`}
            hint={steps.length === 0 ? 'No steps yet' : `${summary.perStepRisk.filter(p => p.risk.level === 'safe').length} safe, ${summary.perStepRisk.filter(p => p.risk.level === 'caution').length} caution, ${summary.perStepRisk.filter(p => p.risk.level === 'danger').length} danger`}
            tone={summary.hasDanger ? 'red' : summary.hasCaution ? 'amber' : (steps.length > 0 ? 'emerald' : 'neutral')}
          />
          <IdentityCard
            icon={<Layers size={16} />}
            label={t('pipeline.idOrder')}
            value={summary.orderWarnings.length === 0 ? (steps.length > 0 ? t('pipeline.orderOk') : '—') : t('pipeline.orderWarnings', { n: summary.orderWarnings.length })}
            hint={summary.orderWarnings.join(' / ') || 'Step ordering looks fine'}
            tone={summary.orderWarnings.length > 0 ? 'amber' : 'neutral'}
          />
        </div>
      )}

      {/* AI Brief — model recommends pipeline based on its state */}
      <div className="rounded-xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
              {t('pipeline.briefLabel')}
            </span>
            {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
          </div>
          {chat.text && !chat.streaming && summary && (
            <button
              onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildPipelineAutoBrief(model, steps.map(s => ({ operation: s.operation, params: s.params })), summary, hasProfile, locale)); }}
              className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
              title={t('pipeline.briefRefresh')}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
        <div className="text-sm text-gray-700 dark:text-stone-300">
          {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
          {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('pipeline.briefPending')}</span>}
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

      {/* Order warnings */}
      {summary && summary.orderWarnings.length > 0 && (
        <div className="rounded-xl border-l-4 border-amber-500 bg-amber-50 dark:bg-amber-500/10 px-4 py-3">
          <div className="flex items-start gap-2.5">
            <AlertTriangle size={16} className="text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" />
            <div className="flex-1 text-xs">
              <div className="font-semibold mb-1 text-amber-800 dark:text-amber-300">{t('pipeline.orderWarningTitle')}</div>
              <ul className="space-y-0.5 text-gray-700 dark:text-stone-300 list-disc pl-4">
                {summary.orderWarnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ============ Section 1: Configure ============ */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 dark:border-stone-700 dark:bg-stone-900">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-stone-100">{t('pipeline.stepsTitle')}</h3>
          <div className="flex items-center gap-2">
            <button
              onClick={handleImportFromAdvisor}
              disabled={importLoading}
              className="flex items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-800"
            >
              <Import size={14} />
              {importLoading ? t('pipeline.loading') : t('pipeline.importAdvisor')}
            </button>
            <AddStepDropdown onAdd={addStep} />
          </div>
        </div>

        {steps.length === 0 ? (
          <div className="rounded-lg border-2 border-dashed border-gray-200 py-10 text-center text-sm text-gray-400 dark:border-stone-700 dark:text-stone-500">
            {t('pipeline.empty')}
          </div>
        ) : (
          <div className="space-y-3">
            {steps.map((step, idx) => (
              <StepCard
                key={step.id}
                step={step}
                index={idx}
                total={steps.length}
                risk={summary?.perStepRisk[idx]?.risk ?? null}
                locale={locale}
                onRemove={removeStep}
                onMove={moveStep}
                onUpdateParam={updateParam}
              />
            ))}
          </div>
        )}

        {/* Validation toggle */}
        <div className="mt-5 space-y-3 border-t border-gray-100 pt-4 dark:border-stone-800">
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-stone-300">
            <input
              type="checkbox"
              checked={!skipValidation}
              onChange={(e) => setSkipValidation(!e.target.checked)}
              className="rounded accent-indigo-500"
            />
            {t('pipeline.computePPL')}
          </label>
          {!skipValidation && (
            <div>
              <label className="block text-xs font-medium text-gray-500 mb-1 dark:text-stone-400">
                {t('pipeline.customPPL')}
              </label>
              <textarea
                value={pplText}
                onChange={(e) => setPplText(e.target.value)}
                placeholder="The quick brown fox jumps over the lazy dog..."
                rows={2}
                className="w-full rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-700 placeholder:text-gray-300 focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:placeholder:text-stone-600"
              />
            </div>
          )}
        </div>
      </div>

      {/* ============ Section 2: Execute ============ */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 dark:border-stone-700 dark:bg-stone-900">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-stone-100">{t('pipeline.executeTitle')}</h3>
          <button
            onClick={handleRun}
            disabled={steps.length === 0 || !!taskId}
            className={cn(
              'flex items-center gap-2 rounded-lg px-5 py-2 text-sm font-medium transition-colors',
              summary?.hasDanger
                ? 'bg-red-500 text-white hover:bg-red-600 disabled:opacity-50'
                : summary?.hasCaution
                  ? 'bg-amber-500 text-white hover:bg-amber-600 disabled:opacity-50'
                  : 'bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-50',
            )}
            title={summary?.hasDanger ? t('pipeline.runDangerHint') : ''}
          >
            <Play size={16} />
            {summary?.hasDanger ? t('pipeline.runAnyway') : t('pipeline.run')}
          </button>
        </div>
        <p className="mt-1 text-xs text-gray-400 dark:text-stone-500">
          {steps.length} {steps.length !== 1 ? t('pipeline.stepsPlural') : t('pipeline.stepSingular')}
          {!skipValidation ? ` + ${t('pipeline.pplValidation')}` : ''}
        </p>
      </div>

      {/* ============ Section 3: Results ============ */}
      {result && (
        <ResultsDashboard
          result={result}
          baselineDir={model.model_dir}
          optimizedDir={result.final_output_dir}
          onLoadOptimized={handleLoadOptimized}
        />
      )}

      {/* Progress overlay */}
      <ProgressOverlay
        taskId={taskId}
        title={t('pipeline.runningTitle')}
        onComplete={handleComplete}
        onError={() => setTaskId(null)}
        onClose={() => setTaskId(null)}
      />

      {/* Ask Model FAB */}
      {!askOpen && (
        <button
          onClick={() => setAskOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
          title={t('pipeline.askModel')}
        >
          <Sparkles size={16} /> {t('pipeline.askModel')}
        </button>
      )}
      {askOpen && (
        <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('pipeline.askModel')}</span>
            </div>
            <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
              <X size={14} />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-sm">
            {chat.text ? <MarkdownContent content={chat.text} /> : (
              <p className="text-gray-400 dark:text-stone-500 text-xs">{t('pipeline.askEmpty')}</p>
            )}
          </div>
          <form
            onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
            className="flex gap-2 border-t border-gray-100 dark:border-stone-800 p-3"
          >
            <input
              value={askInput}
              onChange={(e) => setAskInput(e.target.value)}
              placeholder={t('pipeline.askPlaceholder')}
              disabled={chat.streaming}
              className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
            />
            <button type="submit" disabled={!askInput.trim() || chat.streaming} className="rounded-lg bg-indigo-500 px-3 py-1.5 text-white hover:bg-indigo-600 disabled:opacity-50">
              <Send size={12} />
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

// --- Sub-components ---

function AddStepDropdown({ onAdd }: { onAdd: (op: string) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-600"
      >
        <Plus size={14} />
        Add Step
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute right-0 top-full z-50 mt-1 w-52 rounded-lg border border-gray-200 bg-white py-1 shadow-lg dark:border-stone-700 dark:bg-stone-900">
            {OPERATIONS.map((op) => (
              <button
                key={op.value}
                onClick={() => { onAdd(op.value); setOpen(false); }}
                className="block w-full px-3 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 dark:text-stone-300 dark:hover:bg-stone-800"
              >
                {op.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function StepCard({
  step, index, total, risk, locale, onRemove, onMove, onUpdateParam,
}: {
  step: PipelineStepConfig;
  index: number;
  total: number;
  risk: OperationRisk | null;
  locale: 'en' | 'zh';
  onRemove: (id: string) => void;
  onMove: (id: string, dir: -1 | 1) => void;
  onUpdateParam: (id: string, key: string, value: unknown) => void;
}) {
  const defs = OP_PARAMS[step.operation] || [];
  const borderClass = risk ? RISK_BORDER_CLASS[risk.level] : 'border-l-gray-300 dark:border-l-stone-700';
  const chipClass = risk ? RISK_CHIP_CLASS[risk.level] : '';

  return (
    <div className={cn('flex items-start gap-3 rounded-lg border border-l-4 bg-gray-50 p-4 dark:bg-stone-800/50 border-gray-200 dark:border-stone-700', borderClass)}>
      {/* Reorder buttons */}
      <div className="flex flex-col gap-0.5 pt-1">
        <button
          onClick={() => onMove(step.id, -1)}
          disabled={index === 0}
          className="rounded p-0.5 text-gray-400 hover:text-gray-600 disabled:opacity-30 dark:text-stone-500 dark:hover:text-stone-300"
        >
          <ChevronUp size={14} />
        </button>
        <button
          onClick={() => onMove(step.id, 1)}
          disabled={index === total - 1}
          className="rounded p-0.5 text-gray-400 hover:text-gray-600 disabled:opacity-30 dark:text-stone-500 dark:hover:text-stone-300"
        >
          <ChevronDown size={14} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-100 text-[10px] font-bold text-indigo-600 dark:bg-indigo-500/20 dark:text-indigo-400 shrink-0">
              {index + 1}
            </span>
            <span className="text-sm font-medium text-gray-800 dark:text-stone-200 truncate">
              {OP_LABELS[step.operation] || step.operation}
            </span>
            {risk && (
              <span className={cn('rounded-full px-2 py-0.5 text-[10px] font-medium uppercase shrink-0', chipClass)}>
                {risk.level}
              </span>
            )}
          </div>
          <button
            onClick={() => onRemove(step.id)}
            className="rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 dark:hover:bg-red-500/10"
          >
            <Trash2 size={14} />
          </button>
        </div>

        {/* Risk reason — surface inline so user sees it without hover */}
        {risk && risk.level !== 'safe' && (
          <p className={cn('mt-1.5 text-[11px] leading-relaxed',
            risk.level === 'danger' ? 'text-red-700 dark:text-red-400' : 'text-amber-700 dark:text-amber-400')}>
            {locale === 'zh' ? risk.reasonZh : risk.reason}
          </p>
        )}

        {defs.length > 0 && (
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
            {defs.map((d) => (
              <ParamInput
                key={d.key}
                def={d}
                value={step.params[d.key]}
                onChange={(v) => onUpdateParam(step.id, d.key, v)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ParamInput({
  def, value, onChange,
}: {
  def: ParamDef;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (def.type === 'int_array') {
    const str = Array.isArray(value) ? (value as number[]).join(', ') : '';
    return (
      <div>
        <label className="block text-[11px] font-medium text-gray-500 dark:text-stone-400 mb-0.5">{def.label}</label>
        <input
          type="text"
          value={str}
          onChange={(e) => {
            const nums = e.target.value
              .split(',')
              .map((s) => parseInt(s.trim(), 10))
              .filter((n) => !isNaN(n));
            onChange(nums);
          }}
          placeholder="e.g. 0, 1, 2"
          className="w-full rounded border border-gray-200 px-2 py-1 text-xs text-gray-700 focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
        />
      </div>
    );
  }

  return (
    <div>
      <label className="block text-[11px] font-medium text-gray-500 dark:text-stone-400 mb-0.5">{def.label}</label>
      <input
        type="number"
        value={typeof value === 'number' ? value : def.default as number}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={def.min}
        max={def.max}
        step={def.step}
        className="w-full rounded border border-gray-200 px-2 py-1 text-xs text-gray-700 focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
      />
    </div>
  );
}

function GuidancePanel({ guidance }: { guidance: BenchmarkGuidance }) {
  const cfg = {
    success: { Icon: CheckCircle, bg: 'bg-green-50', border: 'border-green-200', title: 'text-green-800', text: 'text-green-700', ic: 'text-green-500' },
    warning: { Icon: AlertTriangle, bg: 'bg-yellow-50', border: 'border-yellow-200', title: 'text-yellow-800', text: 'text-yellow-700', ic: 'text-yellow-500' },
    danger:  { Icon: XCircle, bg: 'bg-red-50', border: 'border-red-200', title: 'text-red-800', text: 'text-red-700', ic: 'text-red-500' },
  }[guidance.verdict];
  const { Icon } = cfg;
  return (
    <div className={`rounded-xl border ${cfg.border} ${cfg.bg} p-4`}>
      <div className="flex items-start gap-3">
        <Icon size={20} className={`shrink-0 mt-0.5 ${cfg.ic}`} />
        <div>
          <p className={`text-sm font-semibold ${cfg.title}`}>{guidance.title}</p>
          <p className={`mt-1 text-xs ${cfg.text}`}>{guidance.message}</p>
        </div>
      </div>
    </div>
  );
}

function ResultsDashboard({
  result, baselineDir, optimizedDir, onLoadOptimized,
}: {
  result: PipelineRunResult;
  baselineDir: string;
  optimizedDir: string;
  onLoadOptimized: () => void;
}) {
  const t = useT();
  const [benchTaskId, setBenchTaskId] = useState<string | null>(null);
  const [benchResult, setBenchResult] = useState<BenchmarkRunResult | null>(null);

  const handleRunBenchmark = async () => {
    try {
      const { task_id } = await runBenchmark(baselineDir, optimizedDir);
      setBenchTaskId(task_id);
    } catch {
      // ignore
    }
  };

  const handleBenchComplete = (raw: unknown) => {
    setBenchTaskId(null);
    if (raw && typeof raw === 'object') {
      setBenchResult(raw as BenchmarkRunResult);
    }
  };

  const saving = result.original_size_bytes - result.optimized_size_bytes;
  const savingPct = result.original_size_bytes > 0
    ? ((saving / result.original_size_bytes) * 100).toFixed(1)
    : '0.0';

  const pplChange =
    result.baseline_ppl && result.optimized_ppl
      ? ((result.optimized_ppl.perplexity - result.baseline_ppl.perplexity) / result.baseline_ppl.perplexity) * 100
      : null;

  const pplColor =
    pplChange === null ? undefined
    : pplChange < 5 ? '#16a34a'      // green
    : pplChange < 15 ? '#ca8a04'     // yellow
    : '#dc2626';                      // red

  return (
    <div className="space-y-4">
      {/* Success / Error banner */}
      {!result.success && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4">
          <p className="text-sm font-medium text-red-700">Pipeline failed</p>
          <p className="mt-1 text-xs text-red-600">{result.error_message}</p>
        </div>
      )}

      {/* Size metrics */}
      <MetricCards
        metrics={[
          { label: 'Original Size', value: formatSize(result.original_size_bytes) },
          { label: 'Optimized Size', value: formatSize(result.optimized_size_bytes), color: '#4f46e5' },
          { label: 'Saved', value: formatSize(saving), color: '#16a34a' },
          { label: 'Reduction', value: `${savingPct}%`, color: '#16a34a' },
        ]}
      />

      {/* PPL metrics */}
      {(result.baseline_ppl || result.optimized_ppl) && (
        <MetricCards
          metrics={[
            {
              label: 'Baseline PPL',
              value: result.baseline_ppl ? result.baseline_ppl.perplexity.toFixed(2) : 'N/A',
              subtitle: result.baseline_ppl ? `${result.baseline_ppl.num_tokens} tokens` : undefined,
            },
            {
              label: 'Optimized PPL',
              value: result.optimized_ppl ? result.optimized_ppl.perplexity.toFixed(2) : 'N/A',
              subtitle: result.optimized_ppl ? `${result.optimized_ppl.num_tokens} tokens` : undefined,
              color: pplColor,
            },
            {
              label: 'PPL Change',
              value: pplChange !== null ? `${pplChange > 0 ? '+' : ''}${pplChange.toFixed(1)}%` : 'N/A',
              color: pplColor,
            },
            {
              label: 'Total Duration',
              value: `${result.total_duration_seconds.toFixed(1)}s`,
            },
          ]}
        />
      )}

      {/* Per-step breakdown */}
      {result.steps.length > 0 && (
        <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
          <h4 className="mb-3 text-sm font-semibold text-gray-900">Step Breakdown</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr className="text-left text-gray-500">
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Operation</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium text-right">Input Size</th>
                  <th className="px-3 py-2 font-medium text-right">Output Size</th>
                  <th className="px-3 py-2 font-medium text-right">Saved</th>
                  <th className="px-3 py-2 font-medium text-right">Duration</th>
                </tr>
              </thead>
              <tbody>
                {result.steps.map((s, i) => (
                  <tr key={i} className="border-t border-gray-100 hover:bg-gray-50">
                    <td className="px-3 py-2 text-gray-500">{i + 1}</td>
                    <td className="px-3 py-2 font-medium text-gray-700">
                      {OP_LABELS[s.operation] || s.operation}
                    </td>
                    <td className="px-3 py-2">
                      <span className={`rounded px-1.5 py-0.5 text-xs font-bold ${s.success ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {s.success ? 'OK' : 'FAIL'}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right text-gray-600">{formatSize(s.original_size_bytes)}</td>
                    <td className="px-3 py-2 text-right text-gray-600">{formatSize(s.result_size_bytes)}</td>
                    <td className="px-3 py-2 text-right text-green-600">{formatSize(s.saving_bytes)}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{s.duration_seconds.toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Output dir + Load button */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="flex items-center justify-between">
          <div className="min-w-0">
            <p className="text-xs font-medium text-gray-500 dark:text-stone-400">Output Directory</p>
            <p className="mt-0.5 truncate text-sm font-mono text-gray-700 dark:text-stone-300">{result.final_output_dir}</p>
          </div>
          {result.optimized_model_info && (
            <button
              onClick={onLoadOptimized}
              className="ml-4 flex shrink-0 items-center gap-2 rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-600"
            >
              <FolderOpen size={16} />
              Load Optimized Model
            </button>
          )}
        </div>
      </div>

      {/* ============ Benchmark Section ============ */}
      <div className="rounded-xl border border-gray-200 bg-white p-5 dark:border-stone-700 dark:bg-stone-900">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h4 className="text-sm font-semibold text-gray-900 dark:text-stone-100">Full Benchmark</h4>
            <p className="mt-0.5 text-xs text-gray-400 dark:text-stone-500">
              Measures disk, memory, speed (tok/s), and perplexity — baseline vs. optimized.
            </p>
          </div>
          <button
            onClick={handleRunBenchmark}
            disabled={!!benchTaskId}
            className="flex items-center gap-2 rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
          >
            <BarChart2 size={16} />
            Run Benchmark
          </button>
        </div>

        {benchResult && benchResult.comparison && (
          <div className="space-y-4">
            <GuidancePanel guidance={benchResult.comparison.guidance} />

            <MetricCards
              metrics={[
                {
                  label: 'Disk Reduction',
                  value: `${benchResult.comparison.disk_reduction_pct.toFixed(1)}%`,
                  color: '#16a34a',
                },
                {
                  label: 'Memory Reduction',
                  value: `${benchResult.comparison.memory_reduction_pct.toFixed(1)}%`,
                  color: '#16a34a',
                },
                {
                  label: 'Speed Change',
                  value: `${benchResult.comparison.speed_improvement_pct >= 0 ? '+' : ''}${benchResult.comparison.speed_improvement_pct.toFixed(1)}%`,
                  color: benchResult.comparison.speed_improvement_pct >= 0 ? '#16a34a' : '#dc2626',
                },
                {
                  label: 'PPL Delta',
                  value: `${benchResult.comparison.perplexity_delta >= 0 ? '+' : ''}${benchResult.comparison.perplexity_delta.toFixed(2)}`,
                  color: benchResult.comparison.perplexity_delta <= 0.5 ? '#16a34a' : benchResult.comparison.perplexity_delta <= 2 ? '#ca8a04' : '#dc2626',
                },
              ]}
            />

            <div className="overflow-x-auto rounded-lg border border-gray-100">
              <table className="w-full text-xs">
                <thead className="bg-gray-50">
                  <tr className="text-left text-gray-500">
                    <th className="px-3 py-2 font-medium">Metric</th>
                    <th className="px-3 py-2 font-medium text-right">Baseline</th>
                    <th className="px-3 py-2 font-medium text-right">Optimized</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Disk Size', `${benchResult.baseline.disk_size_mb.toFixed(0)} MB`, `${benchResult.optimized!.disk_size_mb.toFixed(0)} MB`],
                    ['RAM (loaded)', `${benchResult.baseline.memory_after_load_mb.toFixed(0)} MB`, `${benchResult.optimized!.memory_after_load_mb.toFixed(0)} MB`],
                    ['Speed', `${benchResult.baseline.tokens_per_second.toFixed(1)} tok/s`, `${benchResult.optimized!.tokens_per_second.toFixed(1)} tok/s`],
                    ['Perplexity', benchResult.baseline.perplexity.toFixed(2), benchResult.optimized!.perplexity.toFixed(2)],
                  ].map(([label, base, opt]) => (
                    <tr key={label} className="border-t border-gray-100">
                      <td className="px-3 py-2 text-gray-500">{label}</td>
                      <td className="px-3 py-2 text-right text-gray-700">{base}</td>
                      <td className="px-3 py-2 text-right font-medium text-indigo-600">{opt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      {/* Benchmark progress overlay */}
      <ProgressOverlay
        taskId={benchTaskId}
        title="Running Benchmark"
        onComplete={handleBenchComplete}
        onError={() => setBenchTaskId(null)}
        onClose={() => setBenchTaskId(null)}
      />

      <NextStepBanner steps={[
        { label: t('nextSteps.chat'), description: t('nextSteps.chatOptimized.desc'), path: '/chat', icon: <MessageCircle size={16} /> },
        { label: t('nextSteps.export'), description: t('nextSteps.export.desc'), path: '/export', icon: <Download size={16} /> },
      ]} />
    </div>
  );
}
