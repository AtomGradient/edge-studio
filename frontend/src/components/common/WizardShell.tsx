// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * WizardShell — container for wizard steps.
 * Provides step indicator, content area, navigation buttons, and help popover.
 */

import { useState } from 'react';
import { ChevronLeft, ChevronRight, HelpCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { StepIndicator, type Step } from './StepIndicator';
import { useT } from '@/i18n';

interface WizardShellProps {
  steps: Step[];
  currentStep: number;
  onStepClick?: (step: number) => void;
  onBack?: () => void;
  onNext?: () => void;
  nextLabel?: string;
  nextDisabled?: boolean;
  hideNav?: boolean;
  /** i18n key for per-step help text. If provided, shows a ? button. */
  helpKey?: string;
  children: React.ReactNode;
}

export function WizardShell({
  steps,
  currentStep,
  onStepClick,
  onBack,
  onNext,
  nextLabel,
  nextDisabled,
  hideNav,
  helpKey,
  children,
}: WizardShellProps) {
  const t = useT();
  const [showHelp, setShowHelp] = useState(false);

  return (
    <div className="flex min-h-screen flex-col">
      {/* Step indicator bar */}
      <div className="sticky top-0 z-10 border-b border-stone-200 bg-white/80 px-4 py-3 backdrop-blur-sm dark:border-stone-800 dark:bg-stone-950/80">
        <div className="flex items-center gap-2">
          <div className="flex-1">
            <StepIndicator
              steps={steps}
              currentStep={currentStep}
              onStepClick={onStepClick}
            />
          </div>
          {helpKey && (
            <button
              type="button"
              onClick={() => setShowHelp((v) => !v)}
              className={cn(
                'flex h-7 w-7 shrink-0 items-center justify-center rounded-full transition-colors',
                showHelp
                  ? 'bg-stone-200 text-stone-700 dark:bg-stone-700 dark:text-stone-200'
                  : 'text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300',
              )}
              aria-label="Help"
            >
              {showHelp ? <X size={14} /> : <HelpCircle size={14} />}
            </button>
          )}
        </div>

        {/* Help popover */}
        {showHelp && helpKey && (
          <div className="mt-2 rounded-lg border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-600 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300">
            {t(helpKey)}
          </div>
        )}
      </div>

      {/* Content area */}
      <div className="flex flex-1 flex-col items-center px-4 py-8 sm:px-8">
        <div className="w-full max-w-3xl">
          {children}
        </div>
      </div>

      {/* Navigation buttons */}
      {!hideNav && (
        <div className="sticky bottom-0 border-t border-stone-200 bg-white/80 px-4 py-4 backdrop-blur-sm dark:border-stone-800 dark:bg-stone-950/80">
          <div className="mx-auto flex max-w-3xl items-center justify-between">
            <button
              type="button"
              onClick={onBack}
              disabled={!onBack}
              className={cn(
                'flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200',
                !onBack
                  ? 'text-stone-300 dark:text-stone-700'
                  : 'text-stone-600 hover:bg-stone-100 dark:text-stone-400 dark:hover:bg-stone-800',
              )}
            >
              <ChevronLeft size={16} />
              {t('wizard.back')}
            </button>

            {onNext && (
              <button
                type="button"
                onClick={onNext}
                disabled={nextDisabled}
                className={cn(
                  'flex items-center gap-1.5 rounded-lg px-5 py-2.5 text-sm font-medium transition-all duration-200',
                  nextDisabled
                    ? 'bg-stone-100 text-stone-400 dark:bg-stone-800 dark:text-stone-600'
                    : 'bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200',
                )}
              >
                {nextLabel || t('wizard.next')}
                <ChevronRight size={16} />
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
