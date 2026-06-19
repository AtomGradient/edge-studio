// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Export — wizard-style export flow.
 *
 * Step 1: Choose format (card-based)
 * Step 2: Configure + Export (renders existing sub-component)
 *
 * Each sub-component handles its own config → export → result flow.
 */

import { useState, useEffect, useMemo, useRef } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { EmptyState } from '@/components/common/EmptyState';
import { formatParamCount, cn } from '@/lib/utils';
import { useT, useLocaleStore } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import {
  Package, FileCode, FileText, Smartphone,
  ArrowLeft, Star, CheckCircle, ChevronRight,
  Hash, HardDrive, Sparkles, RotateCcw, Loader2,
} from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt, deriveModelFacts } from '@/lib/chatPrompts';
import { GGUFExportMode } from './export/GGUFExportMode';
// CoreML export removed — edge-engine uses MLX Metal, not CoreML
import { ReportExportMode } from './export/ReportExportMode';
import { EdgeRuntimeExportMode } from './export/EdgeRuntimeExportMode';
import { ScaffoldAppExportMode } from './export/ScaffoldAppExportMode';

type ExportFormat = 'gguf' | 'report' | 'edge-runtime' | 'scaffold-app';

interface FormatOption {
  id: ExportFormat;
  icon: React.ReactNode;
  title: string;
  titleZh: string;
  description: string;
  descriptionZh: string;
  recommended?: boolean;
  badge?: string;
}

const FORMAT_OPTIONS: FormatOption[] = [
  {
    id: 'scaffold-app',
    icon: <Smartphone size={24} />,
    title: 'Edge iOS App',
    titleZh: 'Edge iOS App',
    description: 'Complete Xcode project with EdgeKit SDK. Build & run on device immediately.',
    descriptionZh: '完整 Xcode 项目，含 EdgeKit SDK。解压即可在真机运行。',
    recommended: true,
    badge: 'Recommended',
  },
  {
    id: 'edge-runtime',
    icon: <FileCode size={24} />,
    title: 'EdgeKit Swift',
    titleZh: 'EdgeKit Swift',
    description: 'Generate Swift package code using EdgeKit for direct SDK integration.',
    descriptionZh: '生成使用 EdgeKit 的 Swift Package 代码，可直接集成到项目中。',
  },
  {
    id: 'gguf',
    icon: <Package size={24} />,
    title: 'GGUF',
    titleZh: 'GGUF',
    description: 'Quantized binary for llama.cpp, Ollama, and other runtimes.',
    descriptionZh: '量化二进制格式，适用于 llama.cpp、Ollama 等运行时。',
  },
  {
    id: 'report',
    icon: <FileText size={24} />,
    title: 'HTML Report',
    titleZh: 'HTML 报告',
    description: 'Standalone analysis report with all available metrics.',
    descriptionZh: '独立 HTML 分析报告，包含所有可用指标。',
  },
];

// Step indicator
function StepIndicator({ step }: { step: 1 | 2 }) {
  const steps = [
    { num: 1, label: 'Choose Format' },
    { num: 2, label: 'Configure & Export' },
  ];

  return (
    <div className="flex items-center gap-2 mb-8">
      {steps.map((s, i) => (
        <div key={s.num} className="flex items-center gap-2">
          <div className={cn(
            'flex h-7 w-7 items-center justify-center rounded-full text-xs font-semibold transition-colors',
            s.num < step
              ? 'bg-green-100 text-green-700 dark:bg-green-500/20 dark:text-green-400'
              : s.num === step
                ? 'bg-indigo-100 text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-400'
                : 'bg-gray-100 text-gray-400 dark:bg-stone-800 dark:text-stone-500',
          )}>
            {s.num < step ? <CheckCircle size={14} /> : s.num}
          </div>
          <span className={cn(
            'text-sm font-medium',
            s.num === step ? 'text-gray-900 dark:text-stone-100' : 'text-gray-400 dark:text-stone-500',
          )}>
            {s.label}
          </span>
          {i < steps.length - 1 && (
            <ChevronRight size={14} className="mx-1 text-gray-300 dark:text-stone-600" />
          )}
        </div>
      ))}
    </div>
  );
}

export default function Export() {
  const model = useModelStore((s) => s.currentModel);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const [selectedFormat, setSelectedFormat] = useState<ExportFormat | null>(null);
  const briefFiredForRef = useRef<string | null>(null);

  // Stage-aware brief: model recommends which format suits itself
  const buildExportBrief = useMemo(() => {
    if (!model) return '';
    const f = deriveModelFacts(model);
    if (locale === 'zh') {
      return `用 2-3 句话推荐用户为你 (${f.name}, ${formatParamCount(f.totalParams)}, ${f.quantBits}-bit, ${f.category}) 选择哪种导出格式 (Scaffold iOS App / EdgeKit Swift / GGUF / HTML Report). 重点说明:基于你的类别和量化等级, 哪种最适合 iOS 端侧部署, 哪种适合开发集成。不要列项, 写成自然一段话。`;
    }
    return `In 2-3 sentences, recommend which export format the user should pick for you (${f.name}, ${formatParamCount(f.totalParams)}, ${f.quantBits}-bit, ${f.category}): Scaffold iOS App / EdgeKit Swift / GGUF / HTML Report. Focus on which best fits iOS on-device deployment given your category + quantization, vs. which suits dev integration. No bullets, one paragraph.`;
  }, [model, locale]);

  const chat = useModelChat({
    modelId: model?.model_id ?? null,
    systemPrompt: model ? buildModelSelfSystemPrompt(model, locale) : '',
    maxTokens: 350,
    temperature: 0.6,
  });

  useEffect(() => {
    briefFiredForRef.current = null;
    chat.reset();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Auto-fire brief once per model (only on Step 1)
  useEffect(() => {
    if (!model || selectedFormat) return;
    if (briefFiredForRef.current === model.model_id) return;
    if (chat.streaming) return;
    briefFiredForRef.current = model.model_id;
    const id = window.setTimeout(() => chat.send(buildExportBrief), 350);
    return () => window.clearTimeout(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, selectedFormat, buildExportBrief]);

  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  const sizeGB = (model.total_size_bytes / (1024 ** 3)).toFixed(1);
  const step: 1 | 2 = selectedFormat ? 2 : 1;

  return (
    <div className="mx-auto max-w-4xl">
      <StepIndicator step={step} />

      {/* Step 1: Choose format */}
      {!selectedFormat && (
        <>
          {/* Model summary bar */}
          <div className="mb-6 flex items-center gap-4 rounded-2xl border border-gray-200 bg-white px-5 py-3 dark:border-stone-700 dark:bg-stone-900">
            <div className="flex items-center gap-2">
              <Hash size={14} className="text-gray-400 dark:text-stone-500" />
              <span className="text-sm font-medium text-gray-900 dark:text-stone-100">{model.model_name}</span>
            </div>
            <div className="h-4 w-px bg-gray-200 dark:bg-stone-700" />
            <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-stone-400">
              <HardDrive size={12} />
              {sizeGB} GB
            </div>
            <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-stone-400">
              {formatParamCount(model.total_params)} params
            </div>
          </div>

          {/* AI Brief — model recommends a format for itself */}
          <div className="mb-4 rounded-2xl border bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
                <span className="text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
                  {t('export.briefLabel')}
                </span>
                {chat.status && <span className="text-[10px] text-gray-400 dark:text-stone-500">{chat.status}</span>}
              </div>
              {chat.text && !chat.streaming && (
                <button
                  onClick={() => { briefFiredForRef.current = null; chat.reset(); chat.send(buildExportBrief); }}
                  className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
                  title={t('export.briefRefresh')}
                >
                  <RotateCcw size={12} />
                </button>
              )}
            </div>
            <div className="text-sm text-gray-700 dark:text-stone-300">
              {chat.streaming && !chat.text && <Loader2 size={14} className="animate-spin inline mr-2" />}
              {chat.text ? <MarkdownContent content={chat.text} /> : <span className="text-gray-400 dark:text-stone-500">{t('export.briefPending')}</span>}
              {chat.streaming && chat.text && <span className="inline-block w-1 h-3.5 ml-0.5 bg-indigo-500 animate-pulse rounded-sm" />}
            </div>
          </div>

          {/* Format cards */}
          <div className="grid gap-3">
            {FORMAT_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setSelectedFormat(opt.id)}
                className={cn(
                  'group relative flex items-center gap-5 rounded-2xl border p-5 text-left transition-all duration-200',
                  opt.recommended
                    ? 'border-indigo-200 bg-indigo-50/30 hover:border-indigo-300 hover:shadow-md dark:border-indigo-500/30 dark:bg-indigo-950/20 dark:hover:border-indigo-500/50'
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md dark:border-stone-700 dark:bg-stone-900 dark:hover:border-stone-600',
                )}
              >
                <div className={cn(
                  'flex h-12 w-12 shrink-0 items-center justify-center rounded-xl transition-colors',
                  opt.recommended
                    ? 'bg-indigo-100 text-indigo-600 group-hover:bg-indigo-600 group-hover:text-white dark:bg-indigo-500/20 dark:text-indigo-400 dark:group-hover:bg-indigo-500 dark:group-hover:text-white'
                    : 'bg-gray-100 text-gray-500 group-hover:bg-gray-900 group-hover:text-white dark:bg-stone-800 dark:text-stone-400 dark:group-hover:bg-stone-100 dark:group-hover:text-stone-900',
                )}>
                  {opt.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-stone-100">
                      {t(`export.format.${opt.id}.title`) !== `export.format.${opt.id}.title` ? t(`export.format.${opt.id}.title`) : opt.title}
                    </h3>
                    {opt.recommended && (
                      <span className="flex items-center gap-0.5 rounded-full bg-indigo-100 px-2 py-0.5 text-[10px] font-medium text-indigo-700 dark:bg-indigo-500/20 dark:text-indigo-400">
                        <Star size={10} /> {opt.badge}
                      </span>
                    )}
                  </div>
                  <p className="mt-0.5 text-xs text-gray-500 leading-relaxed dark:text-stone-400">
                    {opt.description}
                  </p>
                </div>
                <ChevronRight size={16} className="shrink-0 text-gray-300 transition-transform group-hover:translate-x-0.5 group-hover:text-gray-500 dark:text-stone-600 dark:group-hover:text-stone-400" />
              </button>
            ))}
          </div>
        </>
      )}

      {/* Step 2: Configure & Export */}
      {selectedFormat && (
        <>
          {/* Back button */}
          <button
            type="button"
            onClick={() => setSelectedFormat(null)}
            className="mb-6 flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 transition-colors dark:text-stone-400 dark:hover:text-stone-200"
          >
            <ArrowLeft size={14} />
            {t('export.backToFormats') !== 'export.backToFormats' ? t('export.backToFormats') : 'Choose a different format'}
          </button>

          {/* Selected format header */}
          <div className="mb-6 flex items-center gap-3">
            {(() => {
              const opt = FORMAT_OPTIONS.find(o => o.id === selectedFormat);
              if (!opt) return null;
              return (
                <>
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-100 text-indigo-600 dark:bg-indigo-500/20 dark:text-indigo-400">
                    {opt.icon}
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-stone-100">{opt.title}</h2>
                    <p className="text-xs text-gray-500 dark:text-stone-400">{model.model_name} · {sizeGB} GB</p>
                  </div>
                </>
              );
            })()}
          </div>

          {/* Export mode component */}
          {selectedFormat === 'gguf' && <GGUFExportMode model={model} />}
          {selectedFormat === 'report' && <ReportExportMode model={model} />}
          {selectedFormat === 'edge-runtime' && <EdgeRuntimeExportMode model={model} />}
          {selectedFormat === 'scaffold-app' && <ScaffoldAppExportMode model={model} />}
        </>
      )}
    </div>
  );
}
