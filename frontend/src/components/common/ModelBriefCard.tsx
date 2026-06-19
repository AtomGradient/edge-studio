// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import type { ReactNode } from 'react';
import { Loader2, RotateCcw, Sparkles } from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { cn } from '@/lib/utils';

interface BriefPrompt {
  label: string;
  prompt: string;
}

interface ModelBriefCardProps {
  label: ReactNode;
  status?: ReactNode;
  text: string;
  streaming: boolean;
  emptyText: ReactNode;
  streamingText?: ReactNode;
  refreshTitle?: string;
  prompts?: BriefPrompt[];
  actions?: ReactNode;
  className?: string;
  onRefresh?: () => void;
  onPrompt?: (prompt: string) => void;
}

export function ModelBriefCard({
  label,
  status,
  text,
  streaming,
  emptyText,
  streamingText,
  refreshTitle,
  prompts = [],
  actions,
  className,
  onRefresh,
  onPrompt,
}: ModelBriefCardProps) {
  return (
    <div className={cn(
      'rounded-xl border p-4 transition-all',
      'bg-gradient-to-br from-indigo-50 to-purple-50 border-indigo-100',
      'dark:from-indigo-500/5 dark:to-purple-500/5 dark:border-indigo-500/20',
      className,
    )}>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex min-w-0 items-center gap-2">
          <Sparkles size={14} className="shrink-0 text-indigo-500 dark:text-indigo-400" />
          <span className="truncate text-[10px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
            {label}
          </span>
          {status && (
            <span className="min-w-0 max-w-[180px] truncate text-[10px] text-gray-400 dark:text-stone-500">
              {status}
            </span>
          )}
        </div>
        {((onRefresh && !streaming) || actions) && (
          <div className="flex shrink-0 items-center gap-1">
            {onRefresh && !streaming && (
              <button
                type="button"
                onClick={onRefresh}
                className="rounded p-1 text-indigo-500 hover:bg-indigo-100 disabled:opacity-40 dark:hover:bg-indigo-500/10"
                title={refreshTitle}
                aria-label={refreshTitle}
              >
                <RotateCcw size={12} />
              </button>
            )}
            {actions}
          </div>
        )}
      </div>
      <div className="text-sm text-gray-700 dark:text-stone-300">
        {streaming && !text && <Loader2 size={14} className="mr-2 inline animate-spin" />}
        {text ? (
          <MarkdownContent content={text} />
        ) : (
          <span className="text-gray-400 dark:text-stone-500">
            {streaming ? (streamingText || emptyText) : emptyText}
          </span>
        )}
        {streaming && text && (
          <span className="ml-0.5 inline-block h-3.5 w-1 animate-pulse rounded-sm bg-indigo-500" />
        )}
      </div>
      {prompts.length > 0 && onPrompt && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {prompts.map((prompt) => (
            <button
              key={prompt.label}
              type="button"
              onClick={() => onPrompt(prompt.prompt)}
              disabled={streaming}
              className="rounded-full border border-indigo-200 bg-white/60 px-2.5 py-0.5 text-[11px] text-indigo-700 hover:bg-white disabled:opacity-50 dark:border-indigo-500/30 dark:bg-stone-900/50 dark:text-indigo-300 dark:hover:bg-stone-900"
            >
              {prompt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
