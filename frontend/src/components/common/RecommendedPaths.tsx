// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import type { ReactNode } from 'react';
import { ArrowRight, CheckCircle2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { useT } from '@/i18n';

export interface RecommendedPathStep {
  label: ReactNode;
}

export interface RecommendedPath {
  id: string;
  icon: ReactNode;
  title: ReactNode;
  description: ReactNode;
  steps: RecommendedPathStep[];
  actionLabel: ReactNode;
  path?: string;
  disabled?: boolean;
  badge?: ReactNode;
  onAction?: () => void;
}

interface RecommendedPathsProps {
  title?: ReactNode;
  description?: ReactNode;
  paths: RecommendedPath[];
  className?: string;
}

export function RecommendedPaths({
  title,
  description,
  paths,
  className,
}: RecommendedPathsProps) {
  const navigate = useNavigate();
  const t = useT();

  if (paths.length === 0) return null;

  return (
    <section className={cn('space-y-3', className)}>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-widest text-gray-400 dark:text-stone-500">
            {title ?? t('recommendedPaths.title')}
          </h2>
          {description && (
            <p className="mt-1 max-w-2xl text-sm leading-relaxed text-gray-500 dark:text-stone-400">
              {description}
            </p>
          )}
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        {paths.map((path) => {
          const canRun = !path.disabled;
          const handleAction = () => {
            if (!canRun) return;
            if (path.onAction) {
              path.onAction();
              return;
            }
            if (path.path) navigate(path.path);
          };

          return (
            <article
              key={path.id}
              className={cn(
                'flex min-h-[220px] flex-col rounded-lg border bg-white p-4 transition-all dark:bg-stone-900',
                canRun
                  ? 'border-gray-200 hover:border-gray-300 hover:shadow-md dark:border-stone-700 dark:hover:border-stone-600'
                  : 'border-gray-100 opacity-70 dark:border-stone-800',
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-700 dark:bg-stone-800 dark:text-stone-300">
                  {path.icon}
                </div>
                {path.badge && (
                  <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-medium text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300">
                    {path.badge}
                  </span>
                )}
              </div>

              <div className="mt-3 min-w-0">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-stone-100">
                  {path.title}
                </h3>
                <p className="mt-1 text-xs leading-relaxed text-gray-500 dark:text-stone-400">
                  {path.description}
                </p>
              </div>

              <ol className="mt-4 flex-1 space-y-2">
                {path.steps.map((step, index) => (
                  <li key={index} className="flex items-center gap-2 text-xs text-gray-600 dark:text-stone-300">
                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-gray-100 text-[10px] font-semibold text-gray-500 dark:bg-stone-800 dark:text-stone-400">
                      {index + 1}
                    </span>
                    <span className="min-w-0 truncate">{step.label}</span>
                  </li>
                ))}
              </ol>

              <button
                type="button"
                onClick={handleAction}
                disabled={!canRun}
                className={cn(
                  'mt-4 inline-flex min-h-9 items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition-colors',
                  canRun
                    ? 'bg-gray-950 text-white hover:bg-black dark:bg-stone-100 dark:text-stone-950 dark:hover:bg-white'
                    : 'cursor-not-allowed bg-gray-100 text-gray-400 dark:bg-stone-800 dark:text-stone-600',
                )}
              >
                {path.actionLabel}
                {canRun ? <ArrowRight size={13} /> : <CheckCircle2 size={13} />}
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}
