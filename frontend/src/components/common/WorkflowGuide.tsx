// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

type WorkflowGuideBadgeTone = 'neutral' | 'indigo';

const BADGE_TONE_CLASS: Record<WorkflowGuideBadgeTone, string> = {
  neutral: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300',
  indigo: 'bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300',
};

export interface WorkflowGuideBadge {
  label: ReactNode;
  tone?: WorkflowGuideBadgeTone;
}

export interface WorkflowGuideStep {
  icon: ReactNode;
  title: ReactNode;
  description: ReactNode;
}

interface WorkflowGuideProps {
  title: ReactNode;
  description: ReactNode;
  badges?: WorkflowGuideBadge[];
  steps: WorkflowGuideStep[];
  footerItems?: ReactNode[];
  className?: string;
  children?: ReactNode;
}

export function WorkflowGuide({
  title,
  description,
  badges = [],
  steps,
  footerItems = [],
  className,
  children,
}: WorkflowGuideProps) {
  if (steps.length === 0) return null;

  return (
    <section className={cn('min-w-0 rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900', className)}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <h2 className="break-words text-sm font-semibold text-gray-900 dark:text-gray-100">{title}</h2>
          <p className="mt-1 break-words text-xs leading-relaxed text-gray-500 dark:text-gray-400">{description}</p>
        </div>
        {badges.length > 0 && (
          <div className="flex shrink-0 flex-wrap gap-2 text-xs">
            {badges.map((badge, index) => (
              <span
                key={index}
                className={cn(
                  'rounded-full px-2.5 py-1',
                  BADGE_TONE_CLASS[badge.tone ?? 'neutral'],
                )}
              >
                {badge.label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-3">
        {steps.map((step, index) => (
          <div
            key={index}
            className={cn(
              'flex gap-3',
              index < steps.length - 1 && 'md:border-r md:border-gray-100 dark:md:border-gray-800',
              index === 0 && 'md:pr-4',
              index > 0 && index < steps.length - 1 && 'md:px-4',
              index === steps.length - 1 && 'md:pl-4',
            )}
          >
            <div className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center text-indigo-500">
              {step.icon}
            </div>
            <div className="min-w-0">
              <p className="break-words text-xs font-semibold text-gray-800 dark:text-gray-100">{step.title}</p>
              <p className="mt-1 break-words text-xs leading-relaxed text-gray-500 dark:text-gray-400">{step.description}</p>
            </div>
          </div>
        ))}
      </div>

      {footerItems.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400">
          {footerItems.map((item, index) => (
            <span key={index}>{item}</span>
          ))}
        </div>
      )}

      {children && <div className="mt-4">{children}</div>}
    </section>
  );
}
