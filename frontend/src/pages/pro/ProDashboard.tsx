// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * ProDashboard — progress-aware model cockpit.
 *
 * Shows different content based on workflow stage:
 * - just_loaded: explore architecture + generate profile CTA
 * - profiled: profile summary + optimize CTA
 * - optimized: optimization comparison card + export CTA
 *
 * Also shows: metrics row, quick actions, progress timeline.
 */

import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Boxes, Scale, Scissors, MessageCircle, Workflow,
  Download, ArrowRight, Sparkles, CheckCircle, AlertTriangle,
  Cpu, HardDrive, Layers, Hash, BarChart3,
  TrendingDown, Zap, Eye, RotateCcw, Loader2,
} from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { useProgressStore } from '@/stores/progressStore';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { RecommendedPaths, type RecommendedPath } from '@/components/common/RecommendedPaths';
import { cn, formatParamCount, formatSize } from '@/lib/utils';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt, deriveModelFacts } from '@/lib/chatPrompts';

// ---- Types ----

interface MetricProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub?: string;
}

interface ActionCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  path: string;
  available: boolean;
  badge?: string;
  highlight?: boolean;
}

interface SuggestionProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  action: string;
  path: string;
  variant: 'info' | 'success' | 'warning';
}

// ---- Sub-components ----

function MetricCard({ icon, label, value, sub }: MetricProps) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-gray-200 bg-white px-5 py-4 dark:border-stone-700 dark:bg-stone-900">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gray-50 text-gray-400 dark:bg-stone-800 dark:text-stone-500">
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-xs text-gray-500 dark:text-stone-400">{label}</p>
        <p className="truncate text-lg font-semibold text-gray-900 dark:text-stone-100">{value}</p>
        {sub && <p className="truncate text-xs text-gray-400 dark:text-stone-500">{sub}</p>}
      </div>
    </div>
  );
}

function ActionCard({ icon, title, description, path, available, badge, highlight }: ActionCardProps) {
  const navigate = useNavigate();
  return (
    <button
      type="button"
      onClick={() => available && navigate(path)}
      disabled={!available}
      className={cn(
        'group relative flex flex-col items-start gap-3 rounded-2xl border p-5 text-left transition-all duration-200',
        available
          ? highlight
            ? 'border-indigo-200 bg-indigo-50/50 hover:border-indigo-300 hover:shadow-md cursor-pointer dark:border-indigo-500/30 dark:bg-indigo-950/30 dark:hover:border-indigo-500/50'
            : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md cursor-pointer dark:border-stone-700 dark:bg-stone-900 dark:hover:border-stone-600'
          : 'border-gray-100 bg-gray-50 opacity-60 cursor-not-allowed dark:border-stone-800 dark:bg-stone-900/50',
      )}
    >
      <div className="flex w-full items-center justify-between">
        <div className={cn(
          'flex h-10 w-10 items-center justify-center rounded-xl transition-colors',
          available
            ? highlight
              ? 'bg-indigo-100 text-indigo-600 group-hover:bg-indigo-600 group-hover:text-white dark:bg-indigo-500/20 dark:text-indigo-400 dark:group-hover:bg-indigo-500 dark:group-hover:text-white'
              : 'bg-gray-100 text-gray-600 group-hover:bg-gray-900 group-hover:text-white dark:bg-stone-800 dark:text-stone-400 dark:group-hover:bg-stone-100 dark:group-hover:text-stone-900'
            : 'bg-gray-100 text-gray-400 dark:bg-stone-800 dark:text-stone-600',
        )}>
          {icon}
        </div>
        {badge && (
          <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
            {badge}
          </span>
        )}
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-900 dark:text-stone-100">{title}</h3>
        <p className="mt-0.5 text-xs text-gray-500 leading-relaxed dark:text-stone-400">{description}</p>
      </div>
      {available && (
        <ArrowRight size={14} className="absolute bottom-5 right-5 text-gray-300 transition-all group-hover:translate-x-0.5 group-hover:text-gray-600 dark:text-stone-600 dark:group-hover:text-stone-300" />
      )}
    </button>
  );
}

function SuggestionCard({ icon, title, description, action, path, variant }: SuggestionProps) {
  const navigate = useNavigate();
  const colors = {
    info: 'border-blue-100 bg-blue-50/50 dark:border-blue-500/20 dark:bg-blue-950/30',
    success: 'border-green-100 bg-green-50/50 dark:border-green-500/20 dark:bg-green-950/30',
    warning: 'border-amber-100 bg-amber-50/50 dark:border-amber-500/20 dark:bg-amber-950/30',
  };
  const textColors = {
    info: 'text-blue-700 dark:text-blue-400',
    success: 'text-green-700 dark:text-green-400',
    warning: 'text-amber-700 dark:text-amber-400',
  };
  return (
    <div className={cn('flex items-center gap-4 rounded-2xl border p-4', colors[variant])}>
      <div className={cn('shrink-0', textColors[variant])}>{icon}</div>
      <div className="min-w-0 flex-1">
        <p className={cn('text-sm font-medium', textColors[variant])}>{title}</p>
        <p className="mt-0.5 text-xs text-gray-600 dark:text-stone-400">{description}</p>
      </div>
      <button
        type="button"
        onClick={() => navigate(path)}
        className={cn(
          'shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
          variant === 'info' && 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-500/20 dark:text-blue-300 dark:hover:bg-blue-500/30',
          variant === 'success' && 'bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-500/20 dark:text-green-300 dark:hover:bg-green-500/30',
          variant === 'warning' && 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-500/20 dark:text-amber-300 dark:hover:bg-amber-500/30',
        )}
      >
        {action}
      </button>
    </div>
  );
}

// ---- Progress Timeline ----

interface TimelineStep {
  label: string;
  done: boolean;
  active: boolean;
}

function ProgressTimeline({ steps }: { steps: TimelineStep[] }) {
  return (
    <div className="flex items-center gap-1">
      {steps.map((step, i) => (
        <div key={i} className="flex items-center gap-1">
          <div className={cn(
            'flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors',
            step.done
              ? 'bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-400'
              : step.active
                ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-400'
                : 'bg-gray-100 text-gray-400 dark:bg-stone-800 dark:text-stone-500',
          )}>
            {step.done ? <CheckCircle size={12} /> : step.active ? <Zap size={12} /> : null}
            {step.label}
          </div>
          {i < steps.length - 1 && (
            <div className={cn(
              'h-px w-6',
              step.done ? 'bg-green-300 dark:bg-green-500/40' : 'bg-gray-200 dark:bg-stone-700',
            )} />
          )}
        </div>
      ))}
    </div>
  );
}

// ---- Optimization Summary Card ----

function OptimizationSummaryCard() {
  const t = useT();
  const navigate = useNavigate();
  const pipelineResult = useProgressStore((s) => s.pipelineResult);

  if (!pipelineResult?.success) return null;

  const originalGB = (pipelineResult.original_size_bytes / (1024 ** 3)).toFixed(1);
  const optimizedGB = (pipelineResult.optimized_size_bytes / (1024 ** 3)).toFixed(1);
  const savingPct = pipelineResult.original_size_bytes > 0
    ? ((1 - pipelineResult.optimized_size_bytes / pipelineResult.original_size_bytes) * 100).toFixed(0)
    : '0';

  const baselinePPL = pipelineResult.baseline_ppl?.perplexity;
  const optimizedPPL = pipelineResult.optimized_ppl?.perplexity;
  const pplDelta = baselinePPL && optimizedPPL
    ? ((optimizedPPL - baselinePPL) / baselinePPL * 100).toFixed(1)
    : null;

  return (
    <div className="rounded-2xl border border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 p-6 dark:border-green-500/20 dark:from-green-950/30 dark:to-emerald-950/20">
      <div className="flex items-center gap-2 mb-4">
        <CheckCircle size={18} className="text-green-600 dark:text-green-400" />
        <h3 className="text-sm font-semibold text-green-800 dark:text-green-300">
          {t('dashboard.optimized.title')}
        </h3>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="text-xs text-green-600/70 dark:text-green-400/70">{t('dashboard.optimized.original')}</p>
          <p className="text-lg font-semibold text-green-900 dark:text-green-200">{originalGB} GB</p>
        </div>
        <div>
          <p className="text-xs text-green-600/70 dark:text-green-400/70">{t('dashboard.optimized.optimized')}</p>
          <p className="text-lg font-semibold text-green-900 dark:text-green-200">{optimizedGB} GB</p>
        </div>
        <div>
          <p className="text-xs text-green-600/70 dark:text-green-400/70">{t('dashboard.optimized.saving')}</p>
          <p className="text-lg font-semibold text-green-900 dark:text-green-200">
            <TrendingDown size={14} className="inline mr-1" />{savingPct}%
          </p>
        </div>
        {pplDelta !== null && (
          <div>
            <p className="text-xs text-green-600/70 dark:text-green-400/70">{t('dashboard.optimized.quality')}</p>
            <p className={cn(
              'text-lg font-semibold',
              Number(pplDelta) <= 15 ? 'text-green-900 dark:text-green-200' : 'text-amber-700 dark:text-amber-400',
            )}>
              {Number(pplDelta) > 0 ? '+' : ''}{pplDelta}%
            </p>
          </div>
        )}
      </div>

      <div className="mt-4 flex gap-2">
        <button
          type="button"
          onClick={() => navigate('/export')}
          className="rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 transition-colors dark:bg-green-500 dark:hover:bg-green-600"
        >
          {t('dashboard.optimized.exportNow')}
        </button>
        <button
          type="button"
          onClick={() => navigate('/benchmark-dashboard')}
          className="rounded-lg border border-green-300 px-4 py-2 text-sm font-medium text-green-700 hover:bg-green-100 transition-colors dark:border-green-500/30 dark:text-green-400 dark:hover:bg-green-500/10"
        >
          {t('dashboard.optimized.benchmark')}
        </button>
      </div>
    </div>
  );
}

// ---- Main ----

export default function ProDashboard() {
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const model = useModelStore((s) => s.currentModel);
  const profileSummary = useModelStore((s) => s.profileSummary);

  const { stage, fetchProgress } = useProgressStore();
  const pipelineResult = useProgressStore((s) => s.pipelineResult);

  // ─── AI Brief (model-as-interpreter for the dashboard view) ───
  const briefFiredForRef = useRef<string | null>(null);

  const stageLabel = useMemo(() => {
    if (!stage) return 'just_loaded';
    return stage;
  }, [stage]);

  // Stage-aware "what to do next" prompt
  const buildDashboardBrief = useCallback((m: typeof model, st: string, loc: 'en' | 'zh') => {
    if (!m) return '';
    const f = deriveModelFacts(m);
    if (loc === 'zh') {
      const stageNudge = st === 'just_loaded'
        ? `用户刚加载你, 还没生成 profile. 提示用户先去 /architecture 看你的结构, 再去 /activation 跑一次 profile 解锁个性化优化建议。`
        : st === 'profiled'
          ? `用户已生成 profile. 提示用户去 /pipeline 跑优化, 重点提醒哪种优化适合你 (基于 ${f.quantBits}-bit 量化 + ${f.numLayers} 层 + ${f.gqaRatio}:1 GQA)。`
          : st === 'optimized'
            ? `用户已优化. 提示用户去 /export 导出 iOS App, 或者 /benchmark-dashboard 看真机性能。`
            : `用户已导出. 邀请用户去 /chat 实测对话效果, 或加载新模型对比。`;
      return `用 2-3 句话向用户介绍你自己 (名字 ${f.name}, 规模 ${formatParamCount(f.totalParams)}, ${f.quantBits}-bit), 然后基于当前阶段给出 1 个具体下一步建议: ${stageNudge} 不要列项, 写成自然的一段话。`;
    }
    const stageNudge = st === 'just_loaded'
      ? `The user just loaded you, no profile yet. Nudge them to /architecture first to see your structure, then /activation to run a profile that unlocks personalized optimization advice.`
      : st === 'profiled'
        ? `User has profiled you. Nudge them to /pipeline for optimization, with a hint at which strategy fits you best (you're ${f.quantBits}-bit, ${f.numLayers} layers, ${f.gqaRatio}:1 GQA).`
        : st === 'optimized'
          ? `User has optimized you. Nudge them to /export for the iOS app or /benchmark-dashboard for real-device perf.`
          : `User has exported. Invite them to /chat to test conversation, or load another model to compare.`;
    return `In 2-3 natural sentences, introduce yourself to the user (name ${f.name}, scale ${formatParamCount(f.totalParams)}, ${f.quantBits}-bit), then give one concrete next-step suggestion based on the current stage: ${stageNudge} No bullets, write it as one paragraph.`;
  }, []);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: model ? buildModelSelfSystemPrompt(model, locale) : '',
    maxTokens: 350,
    temperature: 0.6,
  });

  // Fetch progress when model changes
  useEffect(() => {
    if (model?.model_id) {
      fetchProgress(model.model_id);
    }
  }, [model?.model_id, fetchProgress]);

  // Reset brief tracking when model switches
  useEffect(() => {
    briefFiredForRef.current = null;
    chat.reset();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire AI brief when model + stage are known (one fire per model+stage combo)
  useEffect(() => {
    if (!model) return;
    const key = `${model.model_id}:${stageLabel}`;
    if (briefFiredForRef.current === key) return;
    if (chat.streaming) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => {
      chat.send(buildDashboardBrief(model, stageLabel, locale));
    }, 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, stageLabel, locale]);

  if (!model) {
    return null;
  }

  const isGGUF = model.is_gguf;
  const qBits = model.quantization?.bits;
  const sizeGB = (model.total_size_bytes / (1024 ** 3)).toFixed(1);

  const categoryLabels: Record<string, string> = {
    llm: 'Language Model',
    vlm: 'Vision-Language',
    tts: 'Text-to-Speech',
    stt: 'Speech-to-Text',
  };
  const categoryLabel = categoryLabels[model.model_category] || model.model_category.toUpperCase();

  // Timeline steps
  const timelineSteps: TimelineStep[] = [
    { label: t('dashboard.stage.loaded'), done: true, active: stage === 'just_loaded' },
    { label: t('dashboard.stage.profiled'), done: stage === 'profiled' || stage === 'optimized' || stage === 'exported', active: stage === 'just_loaded' && !isGGUF },
    { label: t('dashboard.stage.optimized'), done: stage === 'optimized' || stage === 'exported', active: stage === 'profiled' },
    { label: t('dashboard.stage.exported'), done: stage === 'exported', active: stage === 'optimized' },
  ];

  // Build stage-aware suggestions
  const suggestions: SuggestionProps[] = [];

  if (stage === 'just_loaded') {
    // Priority: explore architecture, then generate profile
    suggestions.push({
      icon: <Eye size={18} />,
      title: t('dashboard.suggest.explore.title'),
      description: t('dashboard.suggest.explore.desc'),
      action: t('dashboard.suggest.explore.action'),
      path: '/architecture',
      variant: 'info',
    });

    if (!isGGUF) {
      suggestions.push({
        icon: <Sparkles size={18} />,
        title: t('dashboard.suggest.profile.title'),
        description: t('dashboard.suggest.profile.desc'),
        action: t('dashboard.suggest.profile.action'),
        path: '/activation',
        variant: 'info',
      });
    }
  }

  if (stage === 'profiled') {
    // Profile loaded — show dead neuron stats if available, suggest optimize
    const deadRatio = profileSummary?.dead_ratio_at_01;
    const deadDesc = deadRatio !== undefined && deadRatio > 0
      ? t('dashboard.suggest.optimizeWithStats.desc', { pct: (deadRatio * 100).toFixed(0), size: sizeGB })
      : t('dashboard.suggest.optimize.desc');

    suggestions.push({
      icon: <Scissors size={18} />,
      title: t('dashboard.suggest.optimize.title'),
      description: deadDesc,
      action: t('dashboard.suggest.optimize.action'),
      path: '/pipeline',
      variant: 'success',
    });
  }

  if (stage === 'optimized' && !isGGUF) {
    const pplOk = !pipelineResult?.optimized_ppl || !pipelineResult?.baseline_ppl ||
      ((pipelineResult.optimized_ppl.perplexity - pipelineResult.baseline_ppl.perplexity) / pipelineResult.baseline_ppl.perplexity * 100) <= 15;

    if (pplOk) {
      suggestions.push({
        icon: <Download size={18} />,
        title: t('dashboard.suggest.readyExport.title'),
        description: t('dashboard.suggest.readyExport.desc'),
        action: t('dashboard.suggest.export.action'),
        path: '/export',
        variant: 'success',
      });
    } else {
      suggestions.push({
        icon: <AlertTriangle size={18} />,
        title: t('dashboard.suggest.qualityWarn.title'),
        description: t('dashboard.suggest.qualityWarn.desc'),
        action: t('dashboard.suggest.qualityWarn.action'),
        path: '/pipeline',
        variant: 'warning',
      });
    }
  }

  // Always show chat suggestion for non-GGUF if no other strong suggestion
  if (suggestions.length < 2) {
    suggestions.push({
      icon: <MessageCircle size={18} />,
      title: t('dashboard.suggest.chat.title'),
      description: t('dashboard.suggest.chat.desc'),
      action: t('dashboard.suggest.chat.action'),
      path: '/chat',
      variant: 'info',
    });
  }

  // Determine which action to highlight based on stage
  const highlightPath = stage === 'just_loaded'
    ? '/architecture'
    : stage === 'profiled'
      ? '/pipeline'
      : stage === 'optimized'
        ? '/export'
        : undefined;

  const recommendedPaths: RecommendedPath[] = [
    {
      id: 'understand',
      icon: <Eye size={18} />,
      title: t('recommendedPaths.understand.title'),
      description: t('recommendedPaths.understand.desc'),
      steps: [
        { label: t('recommendedPaths.step.architecture') },
        { label: t('recommendedPaths.step.weights') },
        { label: t('recommendedPaths.step.quality') },
      ],
      actionLabel: t('recommendedPaths.start'),
      path: '/architecture',
      badge: t('recommendedPaths.badge.currentModel'),
    },
    {
      id: 'optimize',
      icon: <Workflow size={18} />,
      title: t('recommendedPaths.optimize.title'),
      description: t('recommendedPaths.optimize.desc'),
      steps: [
        { label: t('recommendedPaths.step.profile') },
        { label: t('recommendedPaths.step.pipeline') },
        { label: t('recommendedPaths.step.benchmark') },
      ],
      actionLabel: isGGUF ? t('recommendedPaths.unavailableForGguf') : t('recommendedPaths.start'),
      path: '/activation',
      disabled: isGGUF,
    },
    {
      id: 'ship',
      icon: <Download size={18} />,
      title: t('recommendedPaths.ship.title'),
      description: t('recommendedPaths.ship.desc'),
      steps: [
        { label: t('recommendedPaths.step.benchmark') },
        { label: t('recommendedPaths.step.export') },
        { label: t('recommendedPaths.step.verify') },
      ],
      actionLabel: isGGUF ? t('recommendedPaths.unavailableForGguf') : t('recommendedPaths.start'),
      path: '/benchmark-dashboard',
      badge: t('recommendedPaths.badge.app'),
      disabled: isGGUF,
    },
  ];

  return (
    <div className="mx-auto max-w-5xl space-y-8">
      {/* Hero: Model identity */}
      <div className="text-center">
        <p className="mb-1 text-xs font-medium uppercase tracking-widest text-gray-400 dark:text-stone-500">
          {categoryLabel} {qBits ? `· ${qBits}-bit` : ''} {isGGUF ? '· GGUF' : ''}
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-gray-900 dark:text-stone-100">
          {model.model_name}
        </h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-stone-500">{model.model_dir}</p>
      </div>

      {/* Progress Timeline */}
      {!isGGUF && (
        <div className="flex justify-center">
          <ProgressTimeline steps={timelineSteps} />
        </div>
      )}

      {/* AI Brief — model-as-interpreter for the dashboard */}
      {!isGGUF && (
        <div className="rounded-2xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
                {t('dashboard.briefLabel')}
              </span>
              {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
            </div>
            {chat.text && !chat.streaming && (
              <button
                onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildDashboardBrief(model, stageLabel, locale)); }}
                className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
                title={t('dashboard.briefRefresh')}
              >
                <RotateCcw size={12} />
              </button>
            )}
          </div>
          <div className="text-sm text-gray-700 dark:text-stone-300">
            {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
            {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('dashboard.briefPending')}</span>}
            {chat.streaming && chat.text && <span className="inline-block w-1 h-3.5 ml-0.5 bg-indigo-500 animate-pulse rounded-sm" />}
          </div>
        </div>
      )}

      {/* Optimization Summary (when optimized) */}
      {stage === 'optimized' && <OptimizationSummaryCard />}

      {/* Metrics row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard
          icon={<Hash size={18} />}
          label={t('dashboard.metric.params')}
          value={formatParamCount(model.total_params)}
          sub={`${model.tensor_count} tensors`}
        />
        <MetricCard
          icon={<HardDrive size={18} />}
          label={t('dashboard.metric.size')}
          value={`${sizeGB} GB`}
          sub={formatSize(model.total_size_bytes)}
        />
        <MetricCard
          icon={<Layers size={18} />}
          label={t('dashboard.metric.layers')}
          value={`${model.num_layers}`}
          sub={`hidden ${model.hidden_size}`}
        />
        <MetricCard
          icon={<Cpu size={18} />}
          label={t('dashboard.metric.heads')}
          value={`${model.num_attention_heads}`}
          sub={model.num_kv_heads !== model.num_attention_heads ? `KV: ${model.num_kv_heads}` : 'MHA'}
        />
      </div>

      <RecommendedPaths
        description={t('recommendedPaths.dashboardDesc')}
        paths={recommendedPaths}
      />

      {/* Quick Actions */}
      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-gray-400 dark:text-stone-500">
          {t('dashboard.actions')}
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <ActionCard
            icon={<Boxes size={20} />}
            title={t('nav.architecture')}
            description={t('dashboard.action.architecture')}
            path="/architecture"
            available={true}
            highlight={highlightPath === '/architecture'}
          />
          <ActionCard
            icon={<Scale size={20} />}
            title={t('nav.weights')}
            description={t('dashboard.action.weights')}
            path="/weights"
            available={true}
          />
          <ActionCard
            icon={<MessageCircle size={20} />}
            title={t('nav.chat')}
            description={t('dashboard.action.chat')}
            path="/chat"
            available={!isGGUF}
            badge={isGGUF ? 'GGUF' : undefined}
          />
          <ActionCard
            icon={<Workflow size={20} />}
            title={t('nav.pipeline')}
            description={t('dashboard.action.pipeline')}
            path="/pipeline"
            available={!isGGUF}
            highlight={highlightPath === '/pipeline'}
          />
          <ActionCard
            icon={<BarChart3 size={20} />}
            title={t('nav.benchmarkDashboard')}
            description={t('dashboard.action.benchmark')}
            path="/benchmark-dashboard"
            available={true}
          />
          <ActionCard
            icon={<Download size={20} />}
            title={t('nav.export')}
            description={t('dashboard.action.export')}
            path="/export"
            available={!isGGUF}
            highlight={highlightPath === '/export'}
          />
        </div>
      </div>

      {/* Smart Suggestions */}
      {suggestions.length > 0 && (
        <div>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-widest text-gray-400 dark:text-stone-500">
            {t('dashboard.suggestions')}
          </h2>
          <div className="space-y-2">
            {suggestions.map((s, i) => (
              <SuggestionCard key={i} {...s} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
