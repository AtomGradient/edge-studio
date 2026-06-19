// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Phase 2 Step 1: Choose target iOS device for export.
 * Two large cards (iPhone / iPad) — single click to select and advance.
 */

import { useNavigate } from 'react-router-dom';
import { Smartphone, Tablet, Lightbulb } from 'lucide-react';
import { useSimpleStore } from '@/stores/simpleStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { EXPORT_STEPS } from './wizardStepsExport';
import { cn } from '@/lib/utils';

interface DeviceOption {
  key: string;
  icon: React.ReactNode;
  titleKey: string;
  descKey: string;
  gradient: string;
  iconColor: string;
}

const DEVICE_OPTIONS: DeviceOption[] = [
  {
    key: 'iphone',
    icon: <Smartphone size={32} />,
    titleKey: 'simple.v2.exportDevice.iphone',
    descKey: 'simple.v2.exportDevice.iphoneDesc',
    gradient: 'from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20',
    iconColor: 'text-blue-500 dark:text-blue-400',
  },
  {
    key: 'ipad',
    icon: <Tablet size={32} />,
    titleKey: 'simple.v2.exportDevice.ipad',
    descKey: 'simple.v2.exportDevice.ipadDesc',
    gradient: 'from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20',
    iconColor: 'text-violet-500 dark:text-violet-400',
  },
];

export default function ExportDevicePage() {
  const t = useT();
  const navigate = useNavigate();
  const { targetDevice, setTargetDevice } = useSimpleStore();

  const handleSelect = (key: string) => {
    setTargetDevice(key);
    navigate('/simple/export/generate');
  };

  return (
    <WizardShell
      steps={EXPORT_STEPS(t)}
      currentStep={0}
      onBack={() => navigate('/simple/done')}
      helpKey="simple.v2.help.exportDevice"
    >
      <div className="flex flex-col items-center gap-8 py-4">
        {/* Header */}
        <div className="text-center">
          <h1 className="mb-2 text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
            {t('simple.v2.exportDevice.title')}
          </h1>
          <p className="text-stone-500 dark:text-stone-400">
            {t('simple.v2.exportDevice.subtitle')}
          </p>
        </div>

        {/* Device cards */}
        <div className="grid w-full max-w-lg gap-4 sm:grid-cols-2">
          {DEVICE_OPTIONS.map((opt) => (
            <button
              key={opt.key}
              type="button"
              onClick={() => handleSelect(opt.key)}
              className={cn(
                'group flex flex-col items-center gap-4 rounded-2xl border-2 p-8 text-center transition-all duration-200',
                'hover:shadow-lg hover:-translate-y-0.5',
                targetDevice === opt.key
                  ? 'border-stone-900 bg-stone-50 dark:border-stone-100 dark:bg-stone-800/50'
                  : 'border-stone-200 bg-white hover:border-stone-400 dark:border-stone-800 dark:bg-stone-900 dark:hover:border-stone-600',
              )}
            >
              <div className={cn(
                'flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br',
                opt.gradient,
              )}>
                <span className={opt.iconColor}>{opt.icon}</span>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-stone-900 dark:text-stone-100">
                  {t(opt.titleKey)}
                </h3>
                <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
                  {t(opt.descKey)}
                </p>
              </div>
            </button>
          ))}
        </div>

        {/* Hint */}
        <div className="flex items-center gap-2 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:bg-amber-900/20 dark:text-amber-300">
          <Lightbulb size={16} className="shrink-0" />
          <span>{t('simple.v2.exportDevice.unsure')}</span>
        </div>
      </div>
    </WizardShell>
  );
}
