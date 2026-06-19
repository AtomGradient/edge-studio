// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 4: Complete — AI ready celebration + Phase 2 export guidance.
 * Shows success state, export CTA, and upgrade/change options.
 */

import { useNavigate } from 'react-router-dom';
import { PartyPopper, Smartphone, ArrowUpCircle, RefreshCw, Microscope } from 'lucide-react';
import { useSimpleStore } from '@/stores/simpleStore';
import { useUIStore } from '@/stores/uiStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS_V2 } from './wizardStepsV2';

export default function CompletePage() {
  const t = useT();
  const navigate = useNavigate();
  const { focus, tier, resetExport } = useSimpleStore();
  const setUserMode = useUIStore((s) => s.setUserMode);

  const handleStartExport = () => {
    resetExport();
    navigate('/simple/export/device');
  };

  const handleUpgradeTier = () => {
    navigate('/simple/tier');
  };

  const handleChangeFocus = () => {
    navigate('/simple/focus');
  };

  const handleExpertMode = () => {
    setUserMode('advanced');
    navigate('/');
  };

  const focusLabel = t(`simple.v2.focus.${focus}`) || focus;
  const tierLabel = tier.charAt(0).toUpperCase() + tier.slice(1);

  return (
    <WizardShell
      steps={WIZARD_STEPS_V2(t)}
      currentStep={4}
      onBack={() => navigate('/simple/setup')}
      helpKey="simple.v2.help.complete"
    >
      <div className="flex flex-col items-center gap-8 py-4 animate-[fadeInUp_0.5s_ease-out]">
        {/* Celebration */}
        <div className="text-center">
          <div className="mb-4 inline-flex h-16 w-16 animate-bounce items-center justify-center rounded-2xl bg-gradient-to-br from-green-100 to-emerald-100 dark:from-green-900/30 dark:to-emerald-900/30 [animation-iteration-count:3]">
            <PartyPopper size={32} className="text-green-600 dark:text-green-400" />
          </div>
          <h1 className="mb-2 text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
            {t('simple.v2.complete.title')}
          </h1>
          <p className="text-stone-500 dark:text-stone-400">
            {t('simple.v2.complete.subtitle')}
          </p>
        </div>

        {/* Export CTA card */}
        <div className="w-full max-w-md overflow-hidden rounded-2xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
          <div className="flex flex-col items-center gap-4 p-6 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-blue-50 dark:bg-blue-900/20">
              <Smartphone size={24} className="text-blue-500 dark:text-blue-400" />
            </div>
            <div>
              <h2 className="mb-1 text-lg font-semibold text-stone-900 dark:text-stone-100">
                {t('simple.v2.complete.exportPrompt')}
              </h2>
              <p className="text-sm text-stone-500 dark:text-stone-400">
                {t('simple.v2.complete.exportDesc')}
              </p>
            </div>
            <button
              type="button"
              onClick={handleStartExport}
              className="w-full rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white transition-all hover:bg-stone-800 hover:shadow-lg dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
            >
              {t('simple.v2.complete.startExport')} →
            </button>
          </div>
        </div>

        {/* Upgrade / Change options */}
        <div className="flex flex-wrap justify-center gap-3">
          <button
            type="button"
            onClick={handleUpgradeTier}
            className="flex items-center gap-2 rounded-xl border border-stone-200 bg-white px-5 py-3 text-sm font-medium text-stone-700 transition-all hover:border-stone-400 hover:shadow-sm dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300 dark:hover:border-stone-500"
          >
            <ArrowUpCircle size={16} className="text-stone-400" />
            <div className="text-left">
              <p>{t('simple.v2.complete.upgradeTier')}</p>
              <p className="text-xs text-stone-400">
                {t('simple.v2.complete.currentTier', { tier: tierLabel })}
              </p>
            </div>
          </button>

          <button
            type="button"
            onClick={handleChangeFocus}
            className="flex items-center gap-2 rounded-xl border border-stone-200 bg-white px-5 py-3 text-sm font-medium text-stone-700 transition-all hover:border-stone-400 hover:shadow-sm dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300 dark:hover:border-stone-500"
          >
            <RefreshCw size={16} className="text-stone-400" />
            <div className="text-left">
              <p>{t('simple.v2.complete.changeFocus')}</p>
              <p className="text-xs text-stone-400">
                {t('simple.v2.complete.currentFocus', { focus: focusLabel })}
              </p>
            </div>
          </button>
        </div>

        {/* Expert mode */}
        <button
          type="button"
          onClick={handleExpertMode}
          className="inline-flex items-center gap-1.5 text-sm text-stone-500 underline decoration-stone-300 underline-offset-2 transition-colors hover:text-stone-700 dark:text-stone-400 dark:decoration-stone-700 dark:hover:text-stone-200"
        >
          <Microscope size={14} />
          {t('simple.v2.complete.switchExpert')}
        </button>
      </div>
    </WizardShell>
  );
}
