// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * NextStepBanner — contextual "what to do next" banner at page bottom.
 * Creates workflow continuity between pages.
 */

import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useT } from '@/i18n';

interface NextStep {
  label: string;
  description: string;
  path: string;
  icon?: React.ReactNode;
}

interface NextStepBannerProps {
  steps: NextStep[];
  className?: string;
}

export function NextStepBanner({ steps, className }: NextStepBannerProps) {
  const navigate = useNavigate();
  const t = useT();

  if (steps.length === 0) return null;

  return (
    <div className={cn('mt-8 rounded-2xl border border-gray-200 bg-gray-50/50 p-5 dark:border-stone-700 dark:bg-stone-900/50', className)}>
      <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-stone-500">
        {t('nextSteps.title')}
      </p>
      <div className={cn('grid gap-2', steps.length > 1 ? 'sm:grid-cols-2' : '')}>
        {steps.map((step) => (
          <button
            key={step.path}
            type="button"
            onClick={() => navigate(step.path)}
            className="group flex items-center gap-3 rounded-xl bg-white px-4 py-3 text-left transition-all duration-200 hover:shadow-md dark:bg-stone-800 dark:hover:bg-stone-700"
          >
            {step.icon && (
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gray-100 text-gray-500 transition-colors group-hover:bg-gray-900 group-hover:text-white dark:bg-stone-700 dark:text-stone-400 dark:group-hover:bg-stone-100 dark:group-hover:text-stone-900">
                {step.icon}
              </div>
            )}
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-stone-100">{step.label}</p>
              <p className="text-xs text-gray-500 dark:text-stone-400">{step.description}</p>
            </div>
            <ArrowRight size={14} className="shrink-0 text-gray-300 transition-all group-hover:translate-x-0.5 group-hover:text-gray-600 dark:text-stone-600 dark:group-hover:text-stone-300" />
          </button>
        ))}
      </div>
    </div>
  );
}
