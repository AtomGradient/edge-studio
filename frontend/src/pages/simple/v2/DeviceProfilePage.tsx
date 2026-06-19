// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 0: Device AI Profile — auto-detect hardware, show AI rating.
 * Zero user interaction needed — detects on mount, shows results.
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, RefreshCw, Monitor, Microscope, Cpu, MemoryStick, Star } from 'lucide-react';
import { useSimpleStore } from '@/stores/simpleStore';
import { useUIStore } from '@/stores/uiStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS_V2 } from './wizardStepsV2';
import axios from 'axios';

const RATING_COLORS: Record<string, string> = {
  air: 'from-sky-100 to-blue-100 dark:from-sky-900/30 dark:to-blue-900/30',
  standard: 'from-stone-100 to-stone-200 dark:from-stone-800/50 dark:to-stone-700/50',
  pro: 'from-violet-100 to-purple-100 dark:from-violet-900/30 dark:to-purple-900/30',
  max: 'from-amber-100 to-orange-100 dark:from-amber-900/30 dark:to-orange-900/30',
  ultra: 'from-rose-100 to-pink-100 dark:from-rose-900/30 dark:to-pink-900/30',
};

const RATING_TEXT_COLORS: Record<string, string> = {
  air: 'text-sky-600 dark:text-sky-400',
  standard: 'text-stone-600 dark:text-stone-400',
  pro: 'text-violet-600 dark:text-violet-400',
  max: 'text-amber-600 dark:text-amber-400',
  ultra: 'text-rose-600 dark:text-rose-400',
};

function getErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

export default function DeviceProfilePage() {
  const t = useT();
  const navigate = useNavigate();
  const { deviceProfile, setDeviceProfile } = useSimpleStore();
  const setUserMode = useUIStore((s) => s.setUserMode);
  const [loading, setLoading] = useState(!deviceProfile);
  const [error, setError] = useState('');

  const fetchProfile = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await axios.post('/api/simple/device-profile');
      setDeviceProfile(res.data);
    } catch (err: unknown) {
      setError(getErrorMessage(err, t('common.error')));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleConfigure = () => {
    navigate('/simple/focus');
  };

  const handleExpertMode = () => {
    setUserMode('advanced');
    navigate('/');
  };

  const rating = deviceProfile?.ai_rating || 'standard';
  const stars = deviceProfile?.ai_rating_stars || 0;

  return (
    <WizardShell
      steps={WIZARD_STEPS_V2(t)}
      currentStep={0}
      onNext={handleConfigure}
      helpKey="simple.v2.help.deviceProfile"
      nextDisabled={!deviceProfile}
      nextLabel={t('simple.v2.deviceProfile.configure')}
    >
      <div className="text-center">
        {/* Hero icon */}
        <div className="mb-6 inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-100 to-orange-100 dark:from-amber-900/30 dark:to-orange-900/30">
          <Monitor size={32} className="text-amber-600 dark:text-amber-400" />
        </div>

        <h1 className="mb-2 text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.v2.deviceProfile.title')}
        </h1>
        <p className="mb-8 text-stone-500 dark:text-stone-400">
          {t('simple.v2.deviceProfile.subtitle')}
        </p>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center gap-3 py-16 text-stone-400">
          <Loader2 size={28} className="animate-spin" />
          <p className="text-sm">{t('simple.v2.deviceProfile.detecting')}</p>
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
          <button
            type="button"
            onClick={fetchProfile}
            className="ml-2 inline-flex items-center gap-1 text-red-600 underline hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
          >
            <RefreshCw size={12} />
            {t('simple.device.retry')}
          </button>
        </div>
      )}

      {/* Device profile card */}
      {deviceProfile && !loading && (
        <div className="space-y-6">
          {/* Main card */}
          <div className="overflow-hidden rounded-2xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
            {/* Device name + chip */}
            <div className="border-b border-stone-100 px-6 py-5 text-center dark:border-stone-800">
              <p className="text-lg font-medium text-stone-900 dark:text-stone-100">
                {deviceProfile.chip}
              </p>
              <p className="text-sm text-stone-500 dark:text-stone-400">
                {deviceProfile.ram_gb} GB
              </p>
            </div>

            {/* AI Rating */}
            <div className="flex flex-col items-center gap-3 px-6 py-6">
              <div className={`inline-flex items-center gap-2 rounded-full bg-gradient-to-r px-5 py-2 ${RATING_COLORS[rating]}`}>
                <span className={`text-sm font-semibold ${RATING_TEXT_COLORS[rating]}`}>
                  AI {t(deviceProfile.ai_rating_label)}
                </span>
              </div>

              {/* Stars */}
              <div className="flex gap-1">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Star
                    key={i}
                    size={20}
                    className={i < stars
                      ? `fill-amber-400 text-amber-400`
                      : 'text-stone-200 dark:text-stone-700'
                    }
                  />
                ))}
              </div>
            </div>

            {/* Stats grid */}
            <div className="grid grid-cols-2 divide-x divide-stone-100 border-t border-stone-100 sm:grid-cols-4 dark:divide-stone-800 dark:border-stone-800">
              <div className="px-3 py-4 text-center">
                <MemoryStick size={16} className="mx-auto mb-1.5 text-stone-400" />
                <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.v2.deviceProfile.ram')}</p>
                <p className="text-sm font-semibold text-stone-900 dark:text-stone-100">{deviceProfile.ram_gb} GB</p>
              </div>
              <div className="px-3 py-4 text-center">
                <Cpu size={16} className="mx-auto mb-1.5 text-stone-400" />
                <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.v2.deviceProfile.gpu')}</p>
                <p className="text-sm font-semibold text-stone-900 dark:text-stone-100">
                  {deviceProfile.gpu_cores > 0 ? t('simple.v2.setup.gpuCores', { count: String(deviceProfile.gpu_cores) }) : '—'}
                </p>
              </div>
              <div className="px-3 py-4 text-center">
                <Monitor size={16} className="mx-auto mb-1.5 text-stone-400" />
                <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.v2.deviceProfile.maxModel')}</p>
                <p className="text-sm font-semibold text-stone-900 dark:text-stone-100">
                  ~{deviceProfile.max_model_size_gb} GB
                </p>
              </div>
              <div className="px-3 py-4 text-center">
                <Star size={16} className="mx-auto mb-1.5 text-stone-400" />
                <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.v2.deviceProfile.recommendedTier')}</p>
                <p className="text-sm font-semibold capitalize text-stone-900 dark:text-stone-100">
                  {deviceProfile.recommended_tier}
                </p>
              </div>
            </div>
          </div>

          {/* Expert mode link */}
          <div className="text-center text-sm text-stone-400 dark:text-stone-500">
            <button
              type="button"
              onClick={handleExpertMode}
              className="inline-flex items-center gap-1 text-stone-600 underline decoration-stone-300 underline-offset-2 transition-colors hover:text-stone-900 dark:text-stone-400 dark:decoration-stone-700 dark:hover:text-stone-200"
            >
              <Microscope size={14} />
              {t('simple.v2.deviceProfile.switchExpert')}
            </button>
          </div>
        </div>
      )}
    </WizardShell>
  );
}
