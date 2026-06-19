// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 0: Welcome page — hero screen with CTA to start wizard.
 */

import { useNavigate } from 'react-router-dom';
import { Sparkles, Microscope, ArrowRight } from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';
import { useWizardStore } from '@/stores/wizardStore';
import { useT } from '@/i18n';

export default function SimpleWelcome() {
  const t = useT();
  const navigate = useNavigate();
  const setUserMode = useUIStore((s) => s.setUserMode);
  const setCurrentStep = useWizardStore((s) => s.setCurrentStep);

  const handleStart = () => {
    setCurrentStep(1);
    navigate('/simple/device');
  };

  const handleExpertMode = () => {
    setUserMode('advanced');
    navigate('/');
  };

  return (
    <div className="flex min-h-[calc(100vh-49px)] flex-col items-center justify-center px-4">
      <div className="mx-auto max-w-lg text-center">
        {/* Hero */}
        <div className="mb-8 inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-100 to-orange-100 dark:from-amber-900/30 dark:to-orange-900/30">
          <Sparkles size={32} className="text-amber-600 dark:text-amber-400" />
        </div>

        <h1 className="mb-3 text-4xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.welcome.title')}
        </h1>
        <p className="mb-10 text-lg text-stone-500 dark:text-stone-400">
          {t('simple.welcome.subtitle')}
        </p>

        {/* CTA */}
        <button
          type="button"
          onClick={handleStart}
          className="group mb-4 inline-flex items-center gap-2 rounded-xl bg-stone-900 px-8 py-3.5 text-base font-medium text-white transition-all duration-200 hover:bg-stone-800 hover:shadow-lg dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
        >
          {t('simple.welcome.start')}
          <ArrowRight size={18} className="transition-transform group-hover:translate-x-0.5" />
        </button>

        <div className="text-sm text-stone-400 dark:text-stone-500">
          {t('simple.welcome.or')}{' '}
          <button
            type="button"
            onClick={handleExpertMode}
            className="inline-flex items-center gap-1 text-stone-600 underline decoration-stone-300 underline-offset-2 transition-colors hover:text-stone-900 dark:text-stone-400 dark:decoration-stone-700 dark:hover:text-stone-200"
          >
            <Microscope size={14} />
            {t('simple.welcome.expertMode')}
          </button>
        </div>
      </div>
    </div>
  );
}
