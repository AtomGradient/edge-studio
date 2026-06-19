// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 5: Simple Export — one-click iOS App ZIP export.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Smartphone, Loader2, Check, Download, RotateCcw, PartyPopper, Shield, Zap, WifiOff, Battery } from 'lucide-react';
import { useWizardStore } from '@/stores/wizardStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS } from './wizardSteps';
import { pollTask } from '../../api/polling';
import { exportScaffoldZip } from '../../api/endpoints';
import axios from 'axios';

type ScaffoldExportPollResult = string | { zip_path?: string; download_url?: string };

function getErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

export default function SimpleExport() {
  const t = useT();
  const navigate = useNavigate();
  const { loadedModelId, selectedModel, setCurrentStep, reset } = useWizardStore();

  const [phase, setPhase] = useState<'ready' | 'exporting' | 'done'>('ready');
  const [progress, setProgress] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState('');
  const [error, setError] = useState('');

  const handleExport = async () => {
    if (!loadedModelId) return;
    setError('');
    setPhase('exporting');
    setProgress(0);

    try {
      const appName = selectedModel?.name?.replace(/\s+/g, '') || 'MyAIApp';
      const { task_id: taskId } = await exportScaffoldZip(
        loadedModelId, appName, '', 'auto', true,
      );

      const poll = await pollTask<ScaffoldExportPollResult>(taskId, {
        onProgress: (percent) => setProgress(percent),
      });

      if (poll.success) {
        // Backend returns zip_path — construct download URL
        const zipPath = typeof poll.result === 'string'
          ? poll.result
          : poll.result?.zip_path || poll.result?.download_url || '';
        if (zipPath) {
          setDownloadUrl(`/api/model/export/scaffold-zip/download?path=${encodeURIComponent(zipPath)}`);
        }
        setPhase('done');
      } else {
        throw new Error(poll.error || 'Export failed');
      }
    } catch (err: unknown) {
      setError(getErrorMessage(err, 'Export failed'));
      setPhase('ready');
    }
  };

  const handleStartOver = () => {
    reset();
    navigate('/simple');
  };

  return (
    <WizardShell
      steps={WIZARD_STEPS(t)}
      currentStep={5}
      onBack={() => { setCurrentStep(4); navigate('/simple/test'); }}
      hideNav={phase === 'done'}
      onStepClick={(s) => { setCurrentStep(s); }}
    >
      <div className="text-center">
        <div className="mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-stone-100 dark:bg-stone-800">
          <Smartphone size={24} className="text-stone-600 dark:text-stone-400" />
        </div>
        <h2 className="mb-1 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.export.title')}
        </h2>
        <p className="mb-8 text-stone-500 dark:text-stone-400">
          {t('simple.export.subtitle')}
        </p>
      </div>

      {/* Ready state */}
      {phase === 'ready' && (
        <div className="space-y-4 text-center">
          {/* What you get */}
          <div className="rounded-xl bg-stone-50 p-6 dark:bg-stone-900">
            <h3 className="mb-3 text-base font-medium text-stone-900 dark:text-stone-100">
              {t('simple.export.whatYouGet')}
            </h3>
            <ul className="space-y-2 text-left text-sm text-stone-600 dark:text-stone-400">
              <li className="flex items-start gap-2.5"><Check size={16} className="mt-0.5 shrink-0 text-green-500" />{t('simple.export.feature1')}</li>
              <li className="flex items-start gap-2.5"><Check size={16} className="mt-0.5 shrink-0 text-green-500" />{t('simple.export.feature2')}</li>
              <li className="flex items-start gap-2.5"><Check size={16} className="mt-0.5 shrink-0 text-green-500" />{t('simple.export.feature3')}</li>
            </ul>
          </div>

          {/* EdgeRuntime advantages */}
          <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-5 dark:border-amber-900/50 dark:bg-amber-900/10">
            <h3 className="mb-3 text-sm font-semibold text-amber-800 dark:text-amber-300">
              {t('simple.export.poweredBy')}
            </h3>
            <div className="grid grid-cols-2 gap-3 text-left">
              <div className="flex items-start gap-2">
                <WifiOff size={15} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-xs font-medium text-stone-800 dark:text-stone-200">{t('simple.export.rt.offline')}</p>
                  <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.export.rt.offlineDesc')}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Shield size={15} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-xs font-medium text-stone-800 dark:text-stone-200">{t('simple.export.rt.privacy')}</p>
                  <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.export.rt.privacyDesc')}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Zap size={15} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-xs font-medium text-stone-800 dark:text-stone-200">{t('simple.export.rt.speed')}</p>
                  <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.export.rt.speedDesc')}</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Battery size={15} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-xs font-medium text-stone-800 dark:text-stone-200">{t('simple.export.rt.cost')}</p>
                  <p className="text-xs text-stone-500 dark:text-stone-400">{t('simple.export.rt.costDesc')}</p>
                </div>
              </div>
            </div>
          </div>

          <button
            type="button"
            onClick={handleExport}
            disabled={!loadedModelId}
            className="inline-flex items-center gap-2 rounded-xl bg-stone-900 px-8 py-3 text-sm font-medium text-white transition-all duration-200 hover:bg-stone-800 hover:shadow-lg dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            <Smartphone size={18} />
            {t('simple.export.generateApp')}
          </button>
        </div>
      )}

      {/* Exporting */}
      {phase === 'exporting' && (
        <div className="space-y-4 text-center">
          <Loader2 size={28} className="mx-auto animate-spin text-stone-400" />
          <p className="text-sm text-stone-600 dark:text-stone-400">
            {t('simple.export.generating')}
          </p>
          <div className="mx-auto h-2 max-w-xs overflow-hidden rounded-full bg-stone-100 dark:bg-stone-800">
            <div
              className="h-full rounded-full bg-amber-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Done! */}
      {phase === 'done' && (
        <div className="space-y-6 text-center">
          <div className="mx-auto inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-green-50 dark:bg-green-900/20">
            <PartyPopper size={32} className="text-green-600 dark:text-green-400" />
          </div>

          <div>
            <h3 className="mb-1 text-xl font-semibold text-stone-900 dark:text-stone-100">
              {t('simple.export.success')}
            </h3>
            <p className="text-sm text-stone-500 dark:text-stone-400">
              {t('simple.export.successDesc')}
            </p>
          </div>

          {downloadUrl && (
            <a
              href={downloadUrl}
              download
              className="inline-flex items-center gap-2 rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white transition-all duration-200 hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
            >
              <Download size={18} />
              {t('simple.export.download')}
            </a>
          )}

          <div>
            <button
              type="button"
              onClick={handleStartOver}
              className="inline-flex items-center gap-1.5 text-sm text-stone-500 transition-colors hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-300"
            >
              <RotateCcw size={14} />
              {t('simple.export.startOver')}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-center text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}
    </WizardShell>
  );
}
