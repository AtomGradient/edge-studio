// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * StepIndicator — horizontal step progress bar for wizard flow.
 * Anthropic-style warm design with stone/terracotta palette.
 */

import { Check } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Step {
  label: string;
  icon?: React.ReactNode;
}

interface StepIndicatorProps {
  steps: Step[];
  currentStep: number;
  onStepClick?: (step: number) => void;
}

export function StepIndicator({ steps, currentStep, onStepClick }: StepIndicatorProps) {
  return (
    <nav className="flex items-center justify-center gap-1" aria-label="Wizard progress">
      {steps.map((step, i) => {
        const isCompleted = i < currentStep;
        const isCurrent = i === currentStep;
        const isClickable = onStepClick && i <= currentStep;

        return (
          <div key={i} className="flex items-center">
            {/* Step circle + label */}
            <button
              type="button"
              onClick={() => isClickable && onStepClick?.(i)}
              disabled={!isClickable}
              className={cn(
                'flex items-center gap-2 rounded-full px-3 py-1.5 text-sm transition-all duration-200',
                isCompleted && 'bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400',
                isCurrent && 'bg-stone-100 text-stone-900 font-medium dark:bg-stone-800 dark:text-stone-100',
                !isCompleted && !isCurrent && 'text-stone-400 dark:text-stone-600',
                isClickable && 'cursor-pointer hover:bg-stone-100 dark:hover:bg-stone-800',
                !isClickable && 'cursor-default',
              )}
            >
              {/* Circle */}
              <span
                className={cn(
                  'flex h-6 w-6 items-center justify-center rounded-full text-xs font-medium transition-all duration-200',
                  isCompleted && 'bg-green-500 text-white',
                  isCurrent && 'bg-stone-800 text-white dark:bg-stone-200 dark:text-stone-900',
                  !isCompleted && !isCurrent && 'bg-stone-200 text-stone-500 dark:bg-stone-700 dark:text-stone-500',
                )}
              >
                {isCompleted ? <Check size={14} strokeWidth={2.5} /> : i + 1}
              </span>

              {/* Label (hidden on mobile for non-current steps) */}
              <span className={cn(
                'hidden sm:inline',
                isCurrent && 'inline',
              )}>
                {step.label}
              </span>
            </button>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div
                className={cn(
                  'mx-1 h-px w-6 sm:w-10 transition-colors duration-200',
                  i < currentStep ? 'bg-green-400' : 'bg-stone-200 dark:bg-stone-700',
                )}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}
