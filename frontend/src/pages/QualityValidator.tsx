// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { useModelStore } from '@/stores/modelStore';
import { computePPL, runGeneration, runFullReport, getCachedQuality } from '@/api/endpoints';
import type { PerplexityResult, GenerationSample, QualityReportResult } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { MetricCards } from '@/components/data/MetricCards';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { IdentityCard } from '@/components/common/IdentityCard';
import {
  Sparkles, Send, X, RotateCcw, Loader2, Cpu, Activity, Target, BookOpen,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { formatParamCount } from '@/lib/utils';
import { buildModelSelfSystemPrompt, deriveModelFacts } from '@/lib/chatPrompts';

type Mode = 'quick-ppl' | 'full-report' | 'custom-prompts';

const MODE_OPTIONS: { value: Mode; label: string }[] = [
  { value: 'quick-ppl', label: 'Quick PPL' },
  { value: 'full-report', label: 'Full Report' },
  { value: 'custom-prompts', label: 'Custom Prompts' },
];

const DEFAULT_PPL_TEXT =
  "The transformer architecture was introduced in the paper 'Attention Is All You Need' " +
  "by Vaswani et al. in 2017. It replaced recurrent neural networks with self-attention " +
  "mechanisms, enabling much more efficient parallel training on modern hardware.";

const DEFAULT_PROMPTS = [
  'Hi How are you?',
  'Explain what a neural network is in one sentence.',
  'What is the capital of Japan?',
  'Write a short poem about coding.',
  "Translate 'hello world' to French, German, and Spanish.",
];

export default function QualityValidator() {
  const model = useModelStore((s) => s.currentModel);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [mode, setMode] = useState<Mode>('quick-ppl');
  const [enableThinking, setEnableThinking] = useState(false);
  const [maxTokens, setMaxTokens] = useState(50);

  // Quick PPL
  const [pplText, setPplText] = useState(DEFAULT_PPL_TEXT);
  const [pplResult, setPplResult] = useState<PerplexityResult | null>(null);
  const [pplTaskId, setPplTaskId] = useState<string | null>(null);

  // Full Report
  const [report, setReport] = useState<QualityReportResult | null>(null);
  const [reportTaskId, setReportTaskId] = useState<string | null>(null);

  // Custom Prompts
  const [customPromptsText, setCustomPromptsText] = useState(DEFAULT_PROMPTS.join('\n'));
  const [customSamples, setCustomSamples] = useState<GenerationSample[] | null>(null);
  const [customTaskId, setCustomTaskId] = useState<string | null>(null);

  // AI Brief / Ask
  const [askOpen, setAskOpen] = useState(false);
  const [askInput, setAskInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // Derived: PPL bucket for AI commentary
  const pplBucket = useMemo(() => {
    if (!pplResult) return 'unmeasured';
    const p = pplResult.perplexity;
    if (p < 5) return 'excellent';
    if (p < 10) return 'good';
    if (p < 20) return 'acceptable';
    if (p < 50) return 'concerning';
    return 'broken';
  }, [pplResult]);

  const facts = useMemo(() => (model ? deriveModelFacts(model) : null), [model]);

  // System prompt with quality context
  const qualitySystemPrompt = useMemo(() => {
    if (!model) return '';
    const base = buildModelSelfSystemPrompt(model, locale);
    const lines: string[] = [base, '', '## YOUR QUALITY VALIDATION STATE'];
    if (pplResult) {
      lines.push(`- Most recent PPL: ${pplResult.perplexity.toFixed(2)} on ${pplResult.num_tokens} tokens (bucket: ${pplBucket})`);
    } else {
      lines.push('- No PPL measured yet — user is on Quality Validator page deciding what to measure');
    }
    if (report) {
      lines.push(`- Full report ran: PPL=${report.perplexity?.perplexity?.toFixed(2)}, ${(report.generation_samples ?? []).length} generation samples`);
    }
    if (customSamples && customSamples.length > 0) {
      lines.push(`- Custom prompts: ${customSamples.length} samples generated, avg ${(customSamples.reduce((s, c) => s + (c.tokens_per_second ?? 0), 0) / customSamples.length).toFixed(1)} tok/s`);
    }
    return lines.join('\n');
  }, [model, locale, pplResult, report, customSamples, pplBucket]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: qualitySystemPrompt,
    maxTokens: 600,
    temperature: 0.55,
  });

  const buildBrief = useCallback((m: typeof model, p: typeof pplResult, loc: 'en' | 'zh'): string => {
    if (!m) return '';
    if (loc === 'zh') {
      if (!p) {
        return `用 2-3 句话介绍 Quality Validator 三种模式 (Quick PPL / Full Report / Custom Prompts) 各适合什么场景, 推荐用户先跑哪个验证我的质量。第一人称, 不要列项。`;
      }
      return `用 2-3 句话评价我刚测的 PPL=${p.perplexity.toFixed(2)} (在 ${p.num_tokens} token 文本上) — 是好是差? 对比同规模模型基线 (一般 Qwen/Llama 4-bit 的 PPL 在 5-10 区间). 第一人称, 给用户一个 GO/CAUTION/STOP 的质量判断。`;
    }
    if (!p) {
      return `In 2-3 sentences, introduce the 3 validation modes (Quick PPL / Full Report / Custom Prompts), what each is best for, and recommend which to run first to check my quality. First person, no bullets.`;
    }
    return `In 2-3 sentences, evaluate my recent PPL=${p.perplexity.toFixed(2)} (on ${p.num_tokens} tokens) — is it good or bad? Reference typical baselines (Qwen/Llama 4-bit usually land 5-10). First person, give the user a GO/CAUTION/STOP quality verdict.`;
  }, []);

  const suggestedPrompts = useMemo(() => {
    if (!model) return [];
    if (locale === 'zh') {
      return [
        { label: '🪞 自评质量', prompt: pplResult
          ? `PPL=${pplResult.perplexity.toFixed(2)} 是好是差? 对比典型 4-bit baseline 给我评分 + 1 个改进方向。`
          : `如果用户只能跑一种验证, 推荐我做 Quick PPL / Full Report / Custom Prompts 中的哪个? 解释原因。` },
        { label: '📐 PPL 基线', prompt: `不同规模 (1B/4B/8B/13B) 和不同量化 (fp16/8-bit/4-bit/2-bit) 的 PPL 大致区间是多少? 帮用户建立判断基准。` },
        { label: '🧪 测试文本', prompt: `当前默认 PPL 文本是关于 transformer 的英文段落. 这个文本是否能反映我的真实质量? 推荐 1-2 类更全面的测试文本 (代码/中文/数学/对话).` },
        { label: '⚠️ PPL 误导', prompt: `什么情况下 PPL 数字看起来好但实际生成质量差? 列 1-2 个常见陷阱 (overfit / domain mismatch / 评测文本太简单).` },
      ];
    }
    return [
      { label: '🪞 Self-grade', prompt: pplResult
        ? `Is PPL=${pplResult.perplexity.toFixed(2)} good or bad? Compare to typical 4-bit baselines, give me a grade + one improvement direction.`
        : `If the user can only run one validation, which of Quick PPL / Full Report / Custom Prompts should I do? Explain why.` },
      { label: '📐 PPL baselines', prompt: `What are typical PPL ranges for different scales (1B/4B/8B/13B) × different quants (fp16/8-bit/4-bit/2-bit)? Help the user calibrate.` },
      { label: '🧪 Test text', prompt: `The default PPL text is an English paragraph about transformers. Does it really reflect my quality? Suggest 1-2 broader test text categories (code / Chinese / math / dialog).` },
      { label: '⚠️ PPL traps', prompt: `When does PPL look good but real generation quality is bad? List 1-2 common pitfalls (overfit / domain mismatch / too-easy text).` },
    ];
  }, [model, pplResult, locale]);

  // Restore cached quality results on mount
  useEffect(() => {
    if (!model) return;
    getCachedQuality(model.model_id)
      .then((cached) => {
        if (cached.ppl && !pplResult) setPplResult(cached.ppl as PerplexityResult);
        if (cached.report && !report) setReport(cached.report as QualityReportResult);
        if (cached.generation && !customSamples) setCustomSamples(cached.generation as GenerationSample[]);
      })
      .catch(() => { /* no cached results */ });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Reset chat / brief on model switch
  useEffect(() => {
    briefFiredForRef.current = null;
    chat.reset();
    setAskOpen(false);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire brief — refire when ppl bucket changes
  useEffect(() => {
    if (!model) return;
    const key = `${model.model_id}:${pplBucket}`;
    if (briefFiredForRef.current === key) return;
    if (chat.streaming) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => chat.send(buildBrief(model, pplResult, locale)), 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, pplBucket, locale]);

  const handleSuggested = useCallback((q: string) => {
    setAskOpen(true);
    chat.send(q);
  }, [chat]);

  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  const handleQuickPPL = async () => {
    const { task_id } = await computePPL(model.model_id, pplText);
    setPplTaskId(task_id);
  };

  const handleFullReport = async () => {
    const { task_id } = await runFullReport(model.model_id, maxTokens, enableThinking);
    setReportTaskId(task_id);
  };

  const handleCustomPrompts = async () => {
    const prompts = customPromptsText
      .split('\n')
      .map((p) => p.trim())
      .filter(Boolean);
    if (prompts.length === 0) return;
    const { task_id } = await runGeneration(model.model_id, prompts, maxTokens, enableThinking);
    setCustomTaskId(task_id);
  };

  return (
    <div className="space-y-5 pb-12 relative">
      <PageHeader
        title={t('quality.title')}
        description={model.model_name}
      />

      {/* 4-card identity strip */}
      {facts && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <IdentityCard
            icon={<Cpu size={16} />}
            label={t('quality.idScale')}
            value={`${formatParamCount(facts.totalParams)} · ${facts.quantBits || '?'}-bit`}
            hint={`${facts.numLayers}L · ${facts.numHeads}/${facts.numKVHeads} GQA`}
            tone="indigo"
          />
          <IdentityCard
            icon={<Activity size={16} />}
            label={t('quality.idPPL')}
            value={pplResult ? pplResult.perplexity.toFixed(2) : '—'}
            hint={pplResult ? `On ${pplResult.num_tokens} tokens · bucket: ${pplBucket}` : 'Run Quick PPL to populate'}
            tone={pplBucket === 'excellent' || pplBucket === 'good' ? 'emerald'
              : pplBucket === 'acceptable' ? 'neutral'
              : pplBucket === 'concerning' ? 'amber'
              : pplBucket === 'broken' ? 'red' : 'neutral'}
          />
          <IdentityCard
            icon={<Target size={16} />}
            label={t('quality.idSamples')}
            value={customSamples ? `${customSamples.length}` : (report?.generation_samples ? `${report.generation_samples.length}` : '—')}
            hint={customSamples
              ? `Custom prompts ran (avg ${(customSamples.reduce((s, c) => s + (c.tokens_per_second ?? 0), 0) / Math.max(customSamples.length, 1)).toFixed(1)} tok/s)`
              : 'Run Custom Prompts to populate'}
            tone={(customSamples && customSamples.length > 0) || (report?.generation_samples && report.generation_samples.length > 0) ? 'emerald' : 'neutral'}
          />
          <IdentityCard
            icon={<BookOpen size={16} />}
            label={t('quality.idReport')}
            value={report ? t('quality.reportReady') : '—'}
            hint={report ? 'Full quality report cached' : 'Run Full Report to populate'}
            tone={report ? 'emerald' : 'neutral'}
          />
        </div>
      )}

      {/* AI Brief */}
      <div className="rounded-xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
            <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
              {t('quality.briefLabel')}
            </span>
            {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
          </div>
          {chat.text && !chat.streaming && (
            <button
              onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildBrief(model, pplResult, locale)); }}
              className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
              title={t('quality.briefRefresh')}
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
        <div className="text-sm text-gray-700 dark:text-stone-300">
          {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
          {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('quality.briefPending')}</span>}
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

      {/* Mode selector + controls */}
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
        <div className="mb-4 flex flex-wrap items-center gap-4">
          <div>
            <label className="mb-1 block text-xs text-gray-500">Validation mode</label>
            <div className="inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5">
              {MODE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setMode(opt.value)}
                  className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    mode === opt.value
                      ? 'bg-white text-indigo-700 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {model.supports_thinking && (
            <label className="flex items-center gap-2 pt-4 text-sm text-gray-600">
              <input
                type="checkbox"
                checked={enableThinking}
                onChange={(e) => setEnableThinking(e.target.checked)}
                className="rounded"
              />
              Enable thinking mode
            </label>
          )}
        </div>

        {/* Quick PPL */}
        {mode === 'quick-ppl' && (
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-gray-500">Text for perplexity</label>
              <textarea
                value={pplText}
                onChange={(e) => setPplText(e.target.value)}
                rows={4}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <button
              onClick={handleQuickPPL}
              className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600"
            >
              Compute Perplexity
            </button>
          </div>
        )}

        {/* Full Report */}
        {mode === 'full-report' && (
          <div className="space-y-3">
            <div className="max-w-xs">
              <label className="mb-1 block text-xs text-gray-500">Max tokens per generation</label>
              <input
                type="number"
                min={10}
                max={200}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
              />
            </div>
            <button
              onClick={handleFullReport}
              className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600"
            >
              Run Full Report
            </button>
          </div>
        )}

        {/* Custom Prompts */}
        {mode === 'custom-prompts' && (
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-gray-500">Prompts (one per line)</label>
              <textarea
                value={customPromptsText}
                onChange={(e) => setCustomPromptsText(e.target.value)}
                rows={6}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <div className="max-w-xs">
              <label className="mb-1 block text-xs text-gray-500">Max tokens</label>
              <input
                type="number"
                min={10}
                max={200}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
              />
            </div>
            <button
              onClick={handleCustomPrompts}
              className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600"
            >
              Run Benchmark
            </button>
          </div>
        )}
      </div>

      {/* Progress overlays */}
      {pplTaskId && (
        <ProgressOverlay
          taskId={pplTaskId}
          title="Computing Perplexity"
          onComplete={(result) => {
            setPplResult(result as PerplexityResult);
            setPplTaskId(null);
          }}
          onError={() => setPplTaskId(null)}
          onClose={() => setPplTaskId(null)}
        />
      )}
      {reportTaskId && (
        <ProgressOverlay
          taskId={reportTaskId}
          title="Running Quality Report"
          onComplete={(result) => {
            setReport(result as QualityReportResult);
            setReportTaskId(null);
          }}
          onError={() => setReportTaskId(null)}
          onClose={() => setReportTaskId(null)}
        />
      )}
      {customTaskId && (
        <ProgressOverlay
          taskId={customTaskId}
          title="Running Generation Benchmark"
          onComplete={(result) => {
            setCustomSamples(result as GenerationSample[]);
            setCustomTaskId(null);
          }}
          onError={() => setCustomTaskId(null)}
          onClose={() => setCustomTaskId(null)}
        />
      )}

      {/* Quick PPL results */}
      {mode === 'quick-ppl' && pplResult && (
        <>
          <MetricCards
            metrics={[
              { label: 'Perplexity', value: pplResult.perplexity.toFixed(2) },
              { label: 'Tokens', value: pplResult.num_tokens },
              { label: 'Duration', value: `${pplResult.duration_seconds.toFixed(1)}s` },
            ]}
            className="mb-6"
          />
          <PPLChart result={pplResult} />
        </>
      )}

      {/* Full Report results */}
      {mode === 'full-report' && report && (
        <>
          <MetricCards
            metrics={[
              { label: 'Avg Perplexity', value: report.avg_perplexity.toFixed(2) },
              { label: 'PPL Texts', value: report.perplexity_results.length },
              { label: 'Gen Prompts', value: report.generation_samples.length },
              { label: 'Total Time', value: `${report.total_duration_seconds.toFixed(1)}s` },
            ]}
            className="mb-6"
          />

          {/* PPL results table */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            <h3 className="mb-3 text-sm font-semibold text-gray-700">Perplexity Results</h3>
            <PPLTable results={report.perplexity_results} />
          </div>

          {/* Generation samples */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            <h3 className="mb-3 text-sm font-semibold text-gray-700">Generation Samples</h3>
            <SamplesList samples={report.generation_samples} />
          </div>
        </>
      )}

      {/* Custom Prompts results */}
      {mode === 'custom-prompts' && customSamples && (
        <>
          <MetricCards
            metrics={[
              { label: 'Prompts', value: customSamples.length },
              {
                label: 'Avg Tok/s',
                value: (
                  customSamples.reduce((s, x) => s + x.tokens_per_second, 0) /
                  Math.max(customSamples.length, 1)
                ).toFixed(1),
              },
              {
                label: 'Avg Prob',
                value: (
                  customSamples.reduce((s, x) => s + x.avg_prob, 0) /
                  Math.max(customSamples.length, 1)
                ).toFixed(4),
              },
            ]}
            className="mb-6"
          />
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-stone-300">Generation Samples</h3>
            <SamplesList samples={customSamples} />
          </div>
        </>
      )}

      {/* Ask Model FAB */}
      {!askOpen && (
        <button
          onClick={() => setAskOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
          title={t('quality.askModel')}
        >
          <Sparkles size={16} /> {t('quality.askModel')}
        </button>
      )}
      {askOpen && (
        <div className="fixed bottom-0 right-0 z-40 w-full max-w-md h-[70vh] bg-white dark:bg-stone-950 border-l border-t border-gray-200 dark:border-stone-700 rounded-tl-2xl shadow-2xl flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-stone-800">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('quality.askModel')}</span>
            </div>
            <button onClick={() => setAskOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
              <X size={14} />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-sm">
            {chat.text ? <MarkdownContent content={chat.text} /> : (
              <p className="text-gray-400 dark:text-stone-500 text-xs">{t('quality.askEmpty')}</p>
            )}
          </div>
          <form
            onSubmit={(e) => { e.preventDefault(); if (askInput.trim() && !chat.streaming) { chat.send(askInput.trim()); setAskInput(''); } }}
            className="flex gap-2 border-t border-gray-100 dark:border-stone-800 p-3"
          >
            <input
              value={askInput}
              onChange={(e) => setAskInput(e.target.value)}
              placeholder={t('quality.askPlaceholder')}
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

function PPLChart({ result }: { result: PerplexityResult }) {
  if (!result.per_token_log_probs || result.per_token_log_probs.length === 0) return null;

  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
      <h3 className="mb-2 text-sm font-semibold text-gray-700">Token Log-Probabilities</h3>
      <Plot
        data={[
          {
            y: result.per_token_log_probs,
            mode: 'lines',
            type: 'scatter',
            line: { color: '#7c4dff', width: 1 },
            hovertemplate: 'Token %{x}<br>Log prob: %{y:.3f}<extra></extra>',
          },
        ]}
        layout={{
          xaxis: { title: { text: 'Token Position' } },
          yaxis: { title: { text: 'Log Probability' } },
          height: 250,
          margin: { t: 10, l: 50, r: 20, b: 40 },
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    </div>
  );
}

function PPLTable({ results }: { results: PerplexityResult[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50">
          <tr className="text-left text-gray-500">
            <th className="px-3 py-2 font-medium">Text</th>
            <th className="px-3 py-2 font-medium text-right">Tokens</th>
            <th className="px-3 py-2 font-medium text-right">PPL</th>
            <th className="px-3 py-2 font-medium text-right">Duration</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className="border-t border-gray-100 hover:bg-gray-50">
              <td className="max-w-xs truncate px-3 py-2 text-gray-700">
                {r.text.length > 60 ? r.text.slice(0, 60) + '...' : r.text}
              </td>
              <td className="px-3 py-2 text-right">{r.num_tokens}</td>
              <td className="px-3 py-2 text-right font-medium">{r.perplexity.toFixed(2)}</td>
              <td className="px-3 py-2 text-right">{r.duration_seconds.toFixed(1)}s</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SamplesList({ samples }: { samples: GenerationSample[] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const toggle = (i: number) => {
    const next = new Set(expanded);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    setExpanded(next);
  };

  return (
    <div className="space-y-2">
      {samples.map((s, i) => (
        <div key={i} className="rounded-lg border border-gray-100">
          <button
            onClick={() => toggle(i)}
            className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-gray-50"
          >
            <span className="font-medium text-gray-700">{s.prompt}</span>
            <span className="text-xs text-gray-400">
              {s.num_tokens} tok &middot; {s.tokens_per_second.toFixed(1)} tok/s
            </span>
          </button>
          {expanded.has(i) && (
            <div className="border-t border-gray-100 px-3 py-3">
              <pre className="mb-3 whitespace-pre-wrap rounded-lg bg-gray-50 p-3 text-xs text-gray-700">
                {s.generated_text}
              </pre>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-gray-500">Avg Prob</p>
                  <p className="font-semibold">{s.avg_prob.toFixed(4)}</p>
                </div>
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-gray-500">Tok/s</p>
                  <p className="font-semibold">{s.tokens_per_second.toFixed(1)}</p>
                </div>
                <div className="rounded-lg bg-gray-50 px-3 py-2">
                  <p className="text-gray-500">Duration</p>
                  <p className="font-semibold">{s.duration_seconds.toFixed(1)}s</p>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
