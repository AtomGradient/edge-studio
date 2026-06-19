// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useModelStore } from '@/stores/modelStore';
import { runTrace, getTraceResult } from '@/api/endpoints';
import type { TraceResponse, StepData } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { MetricCards } from '@/components/data/MetricCards';
import { ChartToggle } from '@/components/charts/ChartToggle';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { useToastStore } from '@/stores/toastStore';
import {
  Image as ImageIcon, X, Sparkles, Send, RotateCcw, Cpu, Layers, Zap, Target,
} from 'lucide-react';
import { formatParamCount } from '@/lib/utils';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveTraceMetrics, buildTraceContextSnippet, buildInferenceAutoBrief,
  getInferenceSuggestedPrompts, getInferencePromptExamples,
} from '@/lib/inferenceInsights';
import { IdentityCard } from '@/components/common/IdentityCard';

/** Map probability [0,1] to green→yellow→red color */
function probToColor(prob: number): string {
  if (prob > 0.8) return '#4CAF50';
  if (prob > 0.5) return '#8BC34A';
  if (prob > 0.2) return '#FFC107';
  if (prob > 0.05) return '#FF9800';
  return '#f44336';
}

/** Escape HTML entities for safe rendering */
function escapeHtml(str: string): string {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

type StepViewMode = 'contributions' | 'hidden-norm';

const STEP_VIEW_OPTIONS = [
  { value: 'contributions', label: 'Layer Contributions' },
  { value: 'hidden-norm', label: 'Hidden State Norm' },
];

export default function InferenceTracer() {
  const model = useModelStore((s) => s.currentModel);
  const setTraceState = useModelStore((s) => s.setTraceState);
  const addToast = useToastStore((s) => s.addToast);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  // Image upload state (for vision models)
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageThumb, setImageThumb] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      setImageB64(dataUrl.split(',')[1]);
      setImageThumb(dataUrl);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }, []);

  // Input controls
  const [prompt, setPrompt] = useState('Hi How are you?');
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [enableThinking, setEnableThinking] = useState(true);
  const [enableTiming, setEnableTiming] = useState(false);
  const [captureAttention, setCaptureAttention] = useState(false);
  // MoE expert routing capture — auto-enabled when the loaded model is MoE so
  // /moe/analyze gets data on the very first trace (zero-friction). Users can
  // still uncheck it; non-MoE models hide the toggle entirely.
  const [captureMoeRouting, setCaptureMoeRouting] = useState(model?.has_moe ?? false);
  useEffect(() => {
    setCaptureMoeRouting(model?.has_moe ?? false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Results
  const [trace, setTrace] = useState<TraceResponse | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Step explorer
  const [stepIdx, setStepIdx] = useState(0);
  const [attnLayerIdx, setAttnLayerIdx] = useState(0);
  const [stepViewMode, setStepViewMode] = useState<StepViewMode>('contributions');

  // AI brief / Ask
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // Derived metrics from trace
  const traceMetrics = useMemo(() => (trace ? deriveTraceMetrics(trace) : null), [trace]);
  const promptExamples = useMemo(() => (model ? getInferencePromptExamples(model, locale) : []), [model, locale]);
  const suggestedPrompts = useMemo(
    () => (trace && traceMetrics ? getInferenceSuggestedPrompts(trace, traceMetrics, locale) : []),
    [trace, traceMetrics, locale],
  );

  // System prompt for chat (model self + trace context if available)
  const inferenceSystemPrompt = useMemo(() => {
    if (!model) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    if (!trace || !traceMetrics) return base;
    return base + '\n\n' + buildTraceContextSnippet(trace, traceMetrics);
  }, [model, locale, trace, traceMetrics]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: inferenceSystemPrompt,
    maxTokens: 700,
    temperature: 0.55,
  });

  // Restore cached trace on mount
  useEffect(() => {
    if (!model || trace) return;
    getTraceResult(model.model_id)
      .then((result) => {
        setTrace(result);
        const hasAttn = result.steps.some(
          (s) => s.layers?.some((l) => l.attn_weights && l.attn_weights.length > 0),
        );
        setTraceState(true, hasAttn, result.enable_timing);
      })
      .catch(() => { /* no cached trace */ });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Reset chat / brief when model switches
  useEffect(() => {
    briefFiredForRef.current = null;
    chat.reset();
    setAskOpen(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire AI brief once a trace is loaded (per model+trace combination)
  useEffect(() => {
    if (!model || !trace || !traceMetrics) return;
    const key = `${model.model_id}:${trace.steps.length}:${trace.total_time_seconds.toFixed(2)}`;
    if (briefFiredForRef.current === key) return;
    if (chat.streaming) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => {
      chat.send(buildInferenceAutoBrief(trace, traceMetrics, locale));
    }, 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, trace?.steps.length, trace?.total_time_seconds, locale]);

  const handleGenerate = async () => {
    if (!model) return;
    setLoading(true);
    setTrace(null); // clear stale results immediately
    setStepIdx(0);
    setAttnLayerIdx(0);
    try {
      const { task_id } = await runTrace(model.model_id, {
        prompt,
        max_tokens: maxTokens,
        temperature,
        top_k: topK,
        top_p: topP,
        enable_thinking: enableThinking,
        enable_timing: enableTiming,
        capture_attention: captureAttention,
        capture_moe_routing: captureMoeRouting,
        image_b64: imageB64 ?? undefined,
      });
      setTaskId(task_id);
    } catch {
      addToast('Failed to start inference trace.', 'error');
      setLoading(false);
    }
  };

  const handleTraceComplete = async () => {
    if (!model) return;
    setTaskId(null);
    setLoading(false);
    try {
      const result = await getTraceResult(model.model_id);
      setTrace(result);
      setStepIdx(0);
      setAttnLayerIdx(0);
      // Update global trace state
      const hasAttn = result.steps.some(
        (s) => s.layers?.some((l) => l.attn_weights && l.attn_weights.length > 0),
      );
      setTraceState(true, hasAttn, result.enable_timing);
    } catch {
      addToast('Failed to load trace results.', 'error');
    }
  };

  const step: StepData | null = trace && trace.steps.length > 0 ? trace.steps[stepIdx] : null;

  // Check if attention data is available
  const hasAttnData = useMemo(() => {
    if (!step || !step.layers || step.layers.length === 0) return false;
    const attn = step.layers[0]?.attn_weights;
    if (!attn || attn.length === 0) return false;
    // Check not all zeros
    return attn.some((row) => row.some((v) => v !== 0));
  }, [step]);

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to run inference tracing" />;
  }

  // Compute summary metrics
  const numGen = trace ? trace.steps.length : 0;
  const decodeTime = trace ? trace.total_time_seconds - trace.prefill_time_seconds : 0;
  const tokPerSec = decodeTime > 0 ? numGen / decodeTime : 0;

  return (
    <div>
      <PageHeader
        title={t('inference.pageTitle')}
        description={model.model_name}
      />

      {/* 4-card identity strip — model facts always; trace-derived facts when available */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
        <IdentityCard
          icon={<Cpu size={16} />}
          label={t('inference.idScale')}
          value={`${formatParamCount(model.total_params)} · ${model.quantization?.bits ?? '?'}-bit`}
          hint={`${model.num_layers}L hidden ${model.hidden_size}, ${model.num_attention_heads}/${model.num_kv_heads} GQA`}
          tone="indigo"
        />
        <IdentityCard
          icon={<Layers size={16} />}
          label={t('inference.idGenerated')}
          value={traceMetrics ? `${traceMetrics.numTokensGenerated} tok` : '—'}
          hint={trace ? `Prompt ${trace.prompt_token_ids.length} → ${trace.steps.length} generated` : 'Run a trace to populate'}
          tone={traceMetrics ? 'emerald' : 'neutral'}
        />
        <IdentityCard
          icon={<Zap size={16} />}
          label={t('inference.idSpeed')}
          value={traceMetrics
            ? `${traceMetrics.tokPerSec.toFixed(1)} tok/s`
            : '—'}
          hint={traceMetrics ? `Prefill ${traceMetrics.prefillTimeS.toFixed(2)}s · Decode ${traceMetrics.decodeTimeS.toFixed(2)}s` : ''}
          tone={traceMetrics ? 'emerald' : 'neutral'}
        />
        <IdentityCard
          icon={<Target size={16} />}
          label={t('inference.idConfidence')}
          value={traceMetrics
            ? `${(traceMetrics.meanProb * 100).toFixed(1)}%${traceMetrics.uncertainStepIdxs.length > 0 ? ` · ${traceMetrics.uncertainStepIdxs.length} ↘` : ''}`
            : '—'}
          hint={traceMetrics?.leastConfidentToken ? `Least-confident token: ${JSON.stringify(traceMetrics.leastConfidentToken.tok)} (p=${traceMetrics.leastConfidentToken.prob.toFixed(3)})` : 'Mean chosen-token probability'}
          tone={traceMetrics ? (traceMetrics.meanProb > 0.6 ? 'emerald' : traceMetrics.meanProb > 0.3 ? 'amber' : 'red') : 'neutral'}
        />
      </div>

      {/* AI Brief — auto-fired after trace, plus suggested follow-up prompts */}
      {trace && (
        <div className="mb-4 rounded-xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
                {t('inference.briefLabel')}
              </span>
              {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
            </div>
            {chat.text && !chat.streaming && traceMetrics && (
              <button
                onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildInferenceAutoBrief(trace, traceMetrics, locale)); }}
                className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
                title={t('weights.briefRefresh')}
              >
                <RotateCcw size={12} />
              </button>
            )}
          </div>
          <div className="text-sm text-gray-700 dark:text-stone-300">
            {chat.streaming && !chat.text && <span className="text-gray-400">…</span>}
            {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('inference.briefPending')}</span>}
            {chat.streaming && chat.text && <span className="inline-block w-1 h-3.5 ml-0.5 bg-indigo-500 animate-pulse rounded-sm" />}
          </div>
          {suggestedPrompts.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {suggestedPrompts.map((sp) => (
                <button
                  key={sp.label}
                  onClick={() => { setAskOpen(true); chat.send(sp.prompt); }}
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

      {/* VLM notice */}
      {model.has_vision && (
        <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50 px-4 py-2.5 text-sm text-blue-700 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-300">
          {t('inference.vlmNotice')}
        </div>
      )}

      {/* Input Controls */}
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Left: prompt + options */}
          <div className="lg:col-span-2 space-y-3">
            {model.has_vision && (
              <div>
                <label className="mb-1 block text-xs text-gray-500 dark:text-stone-400">{t('inference.imageOptional')}</label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleImageSelect}
                  className="hidden"
                />
                {imageThumb ? (
                  <div className="flex items-center gap-2">
                    <img
                      src={imageThumb}
                      alt="selected"
                      className="h-20 rounded-lg object-contain border border-gray-200 dark:border-stone-700"
                    />
                    <button
                      onClick={() => { setImageB64(null); setImageThumb(null); }}
                      className="rounded-full bg-gray-200 p-1 text-gray-600 hover:bg-gray-300 dark:bg-stone-800 dark:text-stone-400"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex items-center gap-2 rounded-lg border border-dashed border-gray-300 px-4 py-2 text-sm text-gray-500 hover:border-indigo-400 hover:text-indigo-500 dark:border-stone-600 dark:text-stone-400"
                  >
                    <ImageIcon size={16} />
                    {t('inference.uploadImage')}
                  </button>
                )}
              </div>
            )}
            <div>
              <div className="mb-1 flex items-center justify-between">
                <label className="block text-xs text-gray-500 dark:text-stone-400">{t('inference.prompt')}</label>
                {promptExamples.length > 0 && (
                  <span className="text-[10px] text-gray-400 dark:text-stone-500">{t('inference.tryExample')}</span>
                )}
              </div>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200"
              />
              {promptExamples.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {promptExamples.map((p) => (
                    <button
                      key={p.label}
                      onClick={() => setPrompt(p.prompt)}
                      disabled={loading}
                      className="rounded-full border border-gray-200 bg-gray-50 px-2 py-0.5 text-[10px] text-gray-600 hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-600 dark:border-stone-700 dark:bg-stone-800/50 dark:text-stone-400 dark:hover:border-indigo-500/30 dark:hover:bg-indigo-500/10 dark:hover:text-indigo-400 disabled:opacity-50"
                      title={p.prompt}
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="flex flex-wrap gap-4">
              {model.supports_thinking && (
                <label className="flex items-center gap-2 text-sm text-gray-600">
                  <input
                    type="checkbox"
                    checked={enableThinking}
                    onChange={(e) => setEnableThinking(e.target.checked)}
                    className="rounded"
                  />
                  Enable thinking mode
                </label>
              )}
              <label className="flex items-center gap-2 text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={enableTiming}
                  onChange={(e) => setEnableTiming(e.target.checked)}
                  className="rounded"
                />
                Enable per-layer timing
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={captureAttention}
                  onChange={(e) => setCaptureAttention(e.target.checked)}
                  className="rounded"
                />
                Capture attention weights
              </label>
              {model.has_moe && (
                <label className="flex items-center gap-2 text-sm text-gray-600" title="Capture per-token expert routing — required for /moe/analyze">
                  <input
                    type="checkbox"
                    checked={captureMoeRouting}
                    onChange={(e) => setCaptureMoeRouting(e.target.checked)}
                    className="rounded"
                  />
                  Capture MoE expert routing
                </label>
              )}
            </div>

            {(enableTiming && captureAttention) && (
              <p className="text-xs text-amber-600 bg-amber-50 rounded px-2 py-1">
                ⚠️ Per-layer timing + attention weights both enabled — response data will be large, expect slower results
              </p>
            )}
          </div>

          {/* Right: generation params */}
          <div className="space-y-2">
            <div>
              <label className="mb-1 block text-xs text-gray-500">Max tokens</label>
              <input
                type="number"
                min={1}
                max={2048}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-gray-500">
                Temperature: {temperature.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="mb-1 block text-xs text-gray-500">Top-K</label>
                <input
                  type="number"
                  min={0}
                  max={200}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-gray-500">
                  Top-P: {topP.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={topP}
                  onChange={(e) => setTopP(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4">
          <button
            onClick={handleGenerate}
            disabled={loading}
            className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
          >
            {loading ? 'Running...' : 'Generate'}
          </button>
        </div>
      </div>

      {/* Progress overlay */}
      {taskId && (
        <ProgressOverlay
          taskId={taskId}
          title="Running Inference Trace"
          onComplete={handleTraceComplete}
          onError={(err) => {
            setTaskId(null);
            setLoading(false);
            console.error('Trace error:', err);
          }}
          onClose={() => {
            setTaskId(null);
            setLoading(false);
          }}
        />
      )}

      {/* Results */}
      {trace && (
        <>
          {/* Summary metrics */}
          <MetricCards
            metrics={[
              { label: 'Generated Tokens', value: numGen },
              { label: 'Tokens/sec', value: tokPerSec.toFixed(1) },
              { label: 'Prefill Time', value: `${trace.prefill_time_seconds.toFixed(2)}s` },
              { label: 'Total Time', value: `${trace.total_time_seconds.toFixed(2)}s` },
            ]}
            className="mb-6"
          />

          {/* Colored token output */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-700">Generated Text</h3>
            <div
              className="overflow-x-auto rounded-lg bg-gray-900 p-4 font-mono text-sm leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: renderColoredTokens(trace),
              }}
            />
            <p className="mt-2 text-xs text-gray-400">
              Token colors: green=high prob, red=low prob. Hover for details.
            </p>
          </div>

          {/* Probability timeline */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
            <h3 className="mb-2 text-sm font-semibold text-gray-700">Token Probability Timeline</h3>
            <ProbabilityTimeline trace={trace} onStepClick={(i) => setStepIdx(i)} />
          </div>

          {/* Step Explorer */}
          {step && (
            <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
              <div className="mb-4 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-700">Step Explorer</h3>
                <ChartToggle
                  mode={stepViewMode}
                  options={STEP_VIEW_OPTIONS}
                  onChange={(m) => setStepViewMode(m as StepViewMode)}
                />
              </div>

              {/* Step slider */}
              <div className="mb-4">
                <label className="mb-1 block text-xs text-gray-500">
                  Step {stepIdx} of {trace.steps.length - 1}
                </label>
                <input
                  type="range"
                  min={0}
                  max={trace.steps.length - 1}
                  value={stepIdx}
                  onChange={(e) => setStepIdx(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {/* Step metrics */}
              <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-xs text-gray-500">Token</p>
                  <p className="font-mono text-sm font-semibold">{JSON.stringify(step.token_str)}</p>
                </div>
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-xs text-gray-500">Probability</p>
                  <p className="font-semibold text-sm">{step.chosen_prob.toFixed(4)}</p>
                </div>
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-xs text-gray-500">Rank</p>
                  <p className="font-semibold text-sm">{step.chosen_rank}</p>
                </div>
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-xs text-gray-500">Hidden Norm</p>
                  <p className="font-semibold text-sm">
                    {step.final_hidden_norm ? step.final_hidden_norm.toFixed(2) : 'N/A'}
                  </p>
                </div>
              </div>

              {/* Top-K Candidates */}
              <div className="mb-6">
                <h4 className="mb-2 text-xs font-semibold text-gray-600">Top-K Candidates</h4>
                <TopKCandidates step={step} />
              </div>

              {/* Attention Heatmap */}
              <div className="mb-6">
                <h4 className="mb-2 text-xs font-semibold text-gray-600">Attention Heatmap</h4>
                {hasAttnData ? (
                  <div>
                    <div className="mb-2">
                      <select
                        value={attnLayerIdx}
                        onChange={(e) => setAttnLayerIdx(Number(e.target.value))}
                        className="rounded-lg border border-gray-300 px-2 py-1 text-sm"
                      >
                        {step.layers.map((l, i) => (
                          <option key={i} value={i}>
                            Layer {l.layer_idx}
                          </option>
                        ))}
                      </select>
                    </div>
                    <AttentionHeatmap step={step} trace={trace} layerIdx={attnLayerIdx} />
                  </div>
                ) : (
                  <p className="py-4 text-center text-sm text-gray-400">
                    Attention weights not available. Enable &quot;Capture attention weights&quot; and
                    re-run inference.
                  </p>
                )}
              </div>

              {/* Layer Contributions / Hidden State Norm */}
              {stepViewMode === 'contributions' && step.layers && step.layers.length > 0 && (
                <div>
                  <h4 className="mb-2 text-xs font-semibold text-gray-600">
                    Layer Residual Contributions
                  </h4>
                  <LayerContributions step={step} />
                </div>
              )}

              {stepViewMode === 'hidden-norm' && step.layers && step.layers.length > 0 && (
                <div>
                  <h4 className="mb-2 text-xs font-semibold text-gray-600">
                    Hidden State Norm Flow
                  </h4>
                  <HiddenStateFlow step={step} />
                </div>
              )}
            </div>
          )}

          {/* Per-Layer Latency Profile */}
          {trace.enable_timing && <LatencyProfile trace={trace} />}
        </>
      )}

      {/* Ask Model FAB + drawer */}
      {trace && (
        <>
          {!askOpen && (
            <button
              onClick={() => setAskOpen(true)}
              className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
              title={t('inference.askModel')}
            >
              <Sparkles size={16} /> {t('inference.askModel')}
            </button>
          )}
          {askOpen && (
            <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
                <div className="flex items-center gap-2">
                  <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
                  <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('inference.askModel')}</span>
                </div>
                <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
                  <X size={14} />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto p-4 text-sm">
                {chat.text ? <MarkdownContent content={chat.text} /> : (
                  <p className="text-gray-400 dark:text-stone-500 text-xs">{t('inference.askEmpty')}</p>
                )}
              </div>
              {/* Quick re-shoot suggestions */}
              {suggestedPrompts.length > 0 && (
                <div className="border-t border-gray-100 dark:border-stone-800 px-3 py-2 flex flex-wrap gap-1.5">
                  {suggestedPrompts.slice(0, 3).map((sp) => (
                    <button
                      key={sp.label}
                      onClick={() => chat.send(sp.prompt)}
                      disabled={chat.streaming}
                      className="rounded-full border border-indigo-200 px-2 py-0.5 text-[10px] text-indigo-700 hover:bg-indigo-50 dark:border-indigo-500/30 dark:text-indigo-300 dark:hover:bg-indigo-500/10 disabled:opacity-50"
                    >
                      {sp.label}
                    </button>
                  ))}
                </div>
              )}
              <form
                onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
                className="flex gap-2 border-t border-gray-100 dark:border-stone-800 p-3"
              >
                <input
                  value={askInput}
                  onChange={(e) => setAskInput(e.target.value)}
                  placeholder={t('inference.askPlaceholder')}
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

// ---- Helper components ----

function renderColoredTokens(trace: TraceResponse): string {
  const parts: string[] = [];
  // Prompt tokens (gray)
  parts.push('<span style="color:#666;">');
  for (const tok of trace.prompt_tokens) {
    parts.push(escapeHtml(tok));
  }
  parts.push('</span>');
  // Generated tokens (colored by probability)
  for (const step of trace.steps) {
    const color = probToColor(step.chosen_prob);
    const escaped = escapeHtml(step.token_str);
    const title = `Step ${step.step_idx}: p=${step.chosen_prob.toFixed(3)}, rank=${step.chosen_rank}`;
    parts.push(
      `<span style="color:${color}; cursor:pointer;" title="${title}">${escaped}</span>`,
    );
  }
  return parts.join('');
}

function ProbabilityTimeline({
  trace,
  onStepClick,
}: {
  trace: TraceResponse;
  onStepClick: (i: number) => void;
}) {
  if (!trace.steps.length) return null;
  const steps = trace.steps.map((s) => s.step_idx);
  const probs = trace.steps.map((s) => s.chosen_prob);
  const tokens = trace.steps.map((s) => s.token_str);
  const ranks = trace.steps.map((s) => s.chosen_rank);
  const colors = probs.map(probToColor);

  return (
    <Plot
      data={[
        {
          x: steps,
          y: probs,
          mode: 'lines+markers',
          type: 'scatter',
          marker: { size: 8, color: colors },
          line: { color: '#00d4ff', width: 1.5 },
          text: tokens.map((t, i) => `${t} (rank=${ranks[i]})`),
          hovertemplate:
            'Step %{x}<br>Token: %{text}<br>Prob: %{y:.4f}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Generation Step' } },
        yaxis: { title: { text: 'Chosen Token Probability' }, range: [0, Math.max(...probs) * 1.1 + 0.01] },
        height: 300,
        margin: { t: 10, l: 50, r: 20, b: 40 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
      onClick={(data) => {
        if (data.points && data.points.length > 0) {
          onStepClick(data.points[0].pointIndex);
        }
      }}
    />
  );
}

function TopKCandidates({ step }: { step: StepData }) {
  const labels = step.top_k_token_strs.slice(0, 15).map((s, i) => {
    let display = s.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
    if (display.length > 20) display = display.slice(0, 17) + '...';
    if (step.top_k_token_ids[i] === step.token_id) display = `>>> ${display}`;
    return display;
  });

  const colors = step.top_k_token_ids.slice(0, 15).map((id) =>
    id === step.token_id ? '#4CAF50' : '#7c4dff',
  );

  return (
    <Plot
      data={[
        {
          x: step.top_k_probs.slice(0, 15) as number[],
          y: labels,
          orientation: 'h',
          type: 'bar',
          marker: { color: colors },
          hovertemplate: 'Token: %{y}<br>Prob: %{x:.4f}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Probability' } },
        height: Math.max(250, labels.length * 25),
        margin: { t: 10, l: 150, r: 20, b: 40 },
        yaxis: { autorange: 'reversed' as const },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function AttentionHeatmap({
  step,
  trace,
  layerIdx,
}: {
  step: StepData;
  trace: TraceResponse;
  layerIdx: number;
}) {
  const layer = step.layers[layerIdx];
  const attn = layer?.attn_weights;
  if (!attn || attn.length === 0) return null;

  const numHeads = attn.length;
  const seqLen = attn[0].length;
  const promptLen = trace.prompt_token_ids.length;

  const xLabels: string[] = [];
  for (let i = 0; i < seqLen; i++) {
    if (i < promptLen) {
      const tok = i < trace.prompt_tokens.length ? trace.prompt_tokens[i] : `p${i}`;
      xLabels.push(`${i}:${tok.replace(/\n/g, '\\n').slice(0, 10)}`);
    } else {
      const genIdx = i - promptLen;
      if (genIdx < trace.steps.length) {
        xLabels.push(`${i}:${trace.steps[genIdx].token_str.replace(/\n/g, '\\n').slice(0, 10)}`);
      } else {
        xLabels.push(`${i}`);
      }
    }
  }

  const yLabels = Array.from({ length: numHeads }, (_, h) => `H${h}`);

  return (
    <Plot
      data={[
        {
          z: attn,
          x: xLabels,
          y: yLabels,
          type: 'heatmap',
          colorscale: 'Blues',
          hovertemplate:
            'Head %{y}<br>Position %{x}<br>Attention: %{z:.4f}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Sequence Position' }, tickangle: -45 },
        yaxis: { title: { text: 'Attention Head' } },
        height: Math.max(300, numHeads * 15),
        margin: { t: 10, l: 60, r: 20, b: 80 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function LayerContributions({ step }: { step: StepData }) {
  const layers = step.layers;
  const labels = layers.map((l) => `L${l.layer_idx}`);

  return (
    <Plot
      data={[
        {
          x: labels,
          y: layers.map((l) => l.attn_residual_norm),
          name: 'Attention',
          type: 'bar',
          marker: { color: '#42A5F5' },
          hovertemplate: 'Layer %{x}<br>Attn Residual Norm: %{y:.2f}<extra></extra>',
        },
        {
          x: labels,
          y: layers.map((l) => l.mlp_residual_norm),
          name: 'MLP',
          type: 'bar',
          marker: { color: '#AB47BC' },
          hovertemplate: 'Layer %{x}<br>MLP Residual Norm: %{y:.2f}<extra></extra>',
        },
      ]}
      layout={{
        barmode: 'group',
        xaxis: { title: { text: 'Layer' } },
        yaxis: { title: { text: 'Residual Norm' } },
        height: 300,
        margin: { t: 10, l: 50, r: 20, b: 40 },
        legend: { orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function HiddenStateFlow({ step }: { step: StepData }) {
  const layers = step.layers;
  const labels = layers.map((l) => `L${l.layer_idx}`);

  return (
    <Plot
      data={[
        {
          x: labels,
          y: layers.map((l) => l.norm_after_attn),
          mode: 'lines+markers',
          type: 'scatter',
          name: 'After Attention',
          line: { color: '#42A5F5' },
          marker: { size: 5 },
          hovertemplate: 'Layer %{x}<br>Norm: %{y:.2f}<extra></extra>',
        },
        {
          x: labels,
          y: layers.map((l) => l.norm_after_mlp),
          mode: 'lines+markers',
          type: 'scatter',
          name: 'After MLP',
          line: { color: '#AB47BC' },
          marker: { size: 5 },
          hovertemplate: 'Layer %{x}<br>Norm: %{y:.2f}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Layer' } },
        yaxis: { title: { text: 'Hidden State Norm' } },
        height: 300,
        margin: { t: 10, l: 50, r: 20, b: 40 },
        legend: { orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function LatencyProfile({ trace }: { trace: TraceResponse }) {
  // Compute latency profile from trace data
  const numLayers = trace.num_layers;
  const labels = Array.from({ length: numLayers }, (_, i) => `L${i}`);

  // Prefill layer latencies
  const prefillAttn: number[] = new Array(numLayers).fill(0);
  const prefillMlp: number[] = new Array(numLayers).fill(0);
  if (trace.prefill_layer_traces && trace.prefill_layer_traces.length > 0) {
    for (const lt of trace.prefill_layer_traces) {
      if (lt.layer_idx < numLayers) {
        prefillAttn[lt.layer_idx] = lt.attn_latency_ms;
        prefillMlp[lt.layer_idx] = lt.mlp_latency_ms;
      }
    }
  }

  // Decode average latencies per layer
  const decodeAttn: number[] = new Array(numLayers).fill(0);
  const decodeMlp: number[] = new Array(numLayers).fill(0);
  const decodeCounts: number[] = new Array(numLayers).fill(0);

  for (const step of trace.steps) {
    if (!step.layers) continue;
    for (const lt of step.layers) {
      if (lt.layer_idx < numLayers) {
        decodeAttn[lt.layer_idx] += lt.attn_latency_ms;
        decodeMlp[lt.layer_idx] += lt.mlp_latency_ms;
        decodeCounts[lt.layer_idx]++;
      }
    }
  }
  for (let i = 0; i < numLayers; i++) {
    if (decodeCounts[i] > 0) {
      decodeAttn[i] /= decodeCounts[i];
      decodeMlp[i] /= decodeCounts[i];
    }
  }

  const prefillTotal = prefillAttn.reduce((a, b) => a + b, 0) + prefillMlp.reduce((a, b) => a + b, 0);
  const decodePerToken = decodeAttn.reduce((a, b) => a + b, 0) + decodeMlp.reduce((a, b) => a + b, 0);

  // Bottleneck layers (top 5 by decode total)
  const bottlenecks = labels
    .map((_, i) => ({
      layer: i,
      attn: decodeAttn[i],
      mlp: decodeMlp[i],
      total: decodeAttn[i] + decodeMlp[i],
    }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 5);

  const totalDecodeMs = bottlenecks.reduce((s, b) => s + b.total, 0) || 1;

  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">Per-Layer Latency Profile</h3>

      <MetricCards
        metrics={[
          { label: 'Prefill Total', value: `${prefillTotal.toFixed(1)} ms` },
          { label: 'Decode / Token', value: `${decodePerToken.toFixed(1)} ms` },
          { label: 'Decode Steps', value: trace.steps.length },
        ]}
        className="mb-4"
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Prefill latency */}
        <div>
          <h4 className="mb-2 text-xs font-semibold text-gray-600">Prefill Per-Layer Latency</h4>
          <Plot
            data={[
              {
                x: labels,
                y: prefillAttn,
                name: 'Attention',
                type: 'bar',
                marker: { color: '#42A5F5' },
                hovertemplate: 'Layer %{x}<br>Attn: %{y:.2f} ms<extra></extra>',
              },
              {
                x: labels,
                y: prefillMlp,
                name: 'MLP',
                type: 'bar',
                marker: { color: '#AB47BC' },
                hovertemplate: 'Layer %{x}<br>MLP: %{y:.2f} ms<extra></extra>',
              },
            ]}
            layout={{
              barmode: 'group',
              xaxis: { title: { text: 'Layer' } },
              yaxis: { title: { text: 'Latency (ms)' } },
              height: 300,
              margin: { t: 10, l: 50, r: 20, b: 40 },
              legend: { orientation: 'h', y: 1.05 },
            }}
            config={{ responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Decode average latency */}
        <div>
          <h4 className="mb-2 text-xs font-semibold text-gray-600">
            Decode Average Per-Layer Latency
          </h4>
          <Plot
            data={[
              {
                x: labels,
                y: decodeAttn,
                name: 'Attention',
                type: 'bar',
                marker: { color: '#42A5F5' },
                hovertemplate: 'Layer %{x}<br>Attn: %{y:.2f} ms<extra></extra>',
              },
              {
                x: labels,
                y: decodeMlp,
                name: 'MLP',
                type: 'bar',
                marker: { color: '#AB47BC' },
                hovertemplate: 'Layer %{x}<br>MLP: %{y:.2f} ms<extra></extra>',
              },
            ]}
            layout={{
              barmode: 'group',
              xaxis: { title: { text: 'Layer' } },
              yaxis: { title: { text: 'Latency (ms)' } },
              height: 300,
              margin: { t: 10, l: 50, r: 20, b: 40 },
              legend: { orientation: 'h', y: 1.05 },
            }}
            config={{ responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>

      {/* Bottleneck layers table */}
      {bottlenecks.length > 0 && (
        <div className="mt-4">
          <h4 className="mb-2 text-xs font-semibold text-gray-600">Bottleneck Layers (Decode)</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-gray-50">
                <tr className="text-left text-gray-500">
                  <th className="px-2 py-1.5 font-medium">Layer</th>
                  <th className="px-2 py-1.5 font-medium text-right">Attn (ms)</th>
                  <th className="px-2 py-1.5 font-medium text-right">MLP (ms)</th>
                  <th className="px-2 py-1.5 font-medium text-right">Total (ms)</th>
                  <th className="px-2 py-1.5 font-medium text-right">% of Top-5</th>
                  <th className="px-2 py-1.5 font-medium">Bottleneck</th>
                </tr>
              </thead>
              <tbody>
                {bottlenecks.map((b) => {
                  const bnType =
                    b.attn > b.mlp * 1.5
                      ? 'ATTN'
                      : b.mlp > b.attn * 1.5
                        ? 'MLP'
                        : 'BOTH';
                  return (
                    <tr key={b.layer} className="border-t border-gray-50">
                      <td className="px-2 py-1.5 font-mono">Layer {b.layer}</td>
                      <td className="px-2 py-1.5 text-right">{b.attn.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right">{b.mlp.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right font-medium">{b.total.toFixed(2)}</td>
                      <td className="px-2 py-1.5 text-right">
                        {((b.total / totalDecodeMs) * 100).toFixed(1)}%
                      </td>
                      <td className="px-2 py-1.5">
                        <span
                          className={`rounded px-1.5 py-0.5 text-xs font-medium ${
                            bnType === 'ATTN'
                              ? 'bg-blue-100 text-blue-700'
                              : bnType === 'MLP'
                                ? 'bg-purple-100 text-purple-700'
                                : 'bg-gray-100 text-gray-700'
                          }`}
                        >
                          {bnType}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
