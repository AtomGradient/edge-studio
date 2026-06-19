// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 1: Device Assessment — auto-detect hardware and show capabilities.
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, RefreshCw, Monitor } from 'lucide-react';
import { useWizardStore } from '@/stores/wizardStore';
import { DeviceCard } from '@/components/common/DeviceCard';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS } from './wizardSteps';
import axios from 'axios';

function getErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

export default function DeviceAssessment() {
  const t = useT();
  const navigate = useNavigate();
  const { systemInfo, setSystemInfo, setCurrentStep, setTargetDevice } = useWizardStore();
  const [loading, setLoading] = useState(!systemInfo);
  const [error, setError] = useState('');

  const fetchSystemInfo = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await axios.get('/api/system-info');
      setSystemInfo(res.data);
      if (res.data.matched_device) {
        setTargetDevice(res.data.matched_device);
      }
    } catch (err: unknown) {
      setError(getErrorMessage(err, t('common.error')));
    } finally {
      setLoading(false);
    }
  };

  // Always re-detect on page visit — system info may differ across machines
  useEffect(() => {
    fetchSystemInfo();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleNext = () => {
    setCurrentStep(2);
    navigate('/simple/pick-model');
  };

  return (
    <WizardShell
      steps={WIZARD_STEPS(t)}
      currentStep={1}
      onBack={() => { setCurrentStep(0); navigate('/simple'); }}
      onNext={handleNext}
      nextDisabled={!systemInfo}
      onStepClick={(s) => { setCurrentStep(s); }}
    >
      <div className="text-center">
        <div className="mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-stone-100 dark:bg-stone-800">
          <Monitor size={24} className="text-stone-600 dark:text-stone-400" />
        </div>
        <h2 className="mb-1 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.device.title')}
        </h2>
        <p className="mb-8 text-stone-500 dark:text-stone-400">
          {t('simple.device.subtitle')}
        </p>
      </div>

      {loading && (
        <div className="flex flex-col items-center gap-3 py-12 text-stone-400">
          <Loader2 size={28} className="animate-spin" />
          <p className="text-sm">{t('simple.device.detecting')}</p>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
          <button
            type="button"
            onClick={fetchSystemInfo}
            className="ml-2 inline-flex items-center gap-1 text-red-600 underline hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
          >
            <RefreshCw size={12} />
            {t('simple.device.retry')}
          </button>
        </div>
      )}

      {systemInfo && !loading && (
        <div className="space-y-6">
          <DeviceCard info={systemInfo} />

          {/* Summary message */}
          <div className="rounded-xl bg-green-50 p-4 text-center text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300">
            {systemInfo.max_model_size_gb
              ? t('simple.device.canRun').replace('{size}', String(systemInfo.max_model_size_gb))
              : t('simple.device.detected')
            }
          </div>
        </div>
      )}
    </WizardShell>
  );
}
