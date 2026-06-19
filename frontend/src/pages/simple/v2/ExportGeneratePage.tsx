// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Phase 2 Step 2: Smart adaptation + export + post-export guidance.
 *
 * Flow:
 *  1. Check compatibility (export-check API)
 *  2. Show result: fits / too large / no fit
 *  3. User edits app name + clicks "Generate"
 *  4. Poll scaffold-zip export task
 *  5. Done → download ZIP + 1-2-3 step guide
 */

import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Loader2, CheckCircle2, AlertTriangle, XCircle,
  Download, Package, Sparkles,
  ArrowDownCircle, RefreshCw,
} from 'lucide-react';
import { useSimpleStore, type ExportCheck } from '@/stores/simpleStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { EXPORT_STEPS } from './wizardStepsExport';
import { cn } from '@/lib/utils';
import axios from 'axios';
import { friendlyError } from '@/lib/friendlyError';

function getAxiosDetail(error: unknown): string | undefined {
  if (!axios.isAxiosError<{ detail?: unknown }>(error)) return undefined;
  const detail = error.response?.data?.detail;
  return typeof detail === 'string' ? detail : undefined;
}

export default function ExportGeneratePage() {
  const t = useT();
  const navigate = useNavigate();
  const {
    targetDevice, focus, tier, loadedModelId, setupInfo,
    exportCheck, setExportCheck,
    exportPhase, setExportPhase,
    exportTaskId, setExportTaskId,
    appName, setAppName,
    downloadUrl, setDownloadUrl,
  } = useSimpleStore();

  const [error, setError] = useState('');
  const [progress, setProgress] = useState(0);
  const cancelledRef = useRef(false);

  // ── Guard: need a loaded model ──────────────────────────────────
  const hasModel = !!loadedModelId;

  // ── Compatibility check ─────────────────────────────────────────
  async function checkCompat() {
    setExportPhase('checking');
    setError('');
    try {
      const res = await axios.post('/api/simple/export-check', {
        target_device: targetDevice,
        focus,
        current_model_id: loadedModelId,
        current_model_size_gb: setupInfo?.size_gb || 0,
      });
      setExportCheck(res.data as ExportCheck);
      setExportPhase('idle');
    } catch (err: unknown) {
      setError(friendlyError(getAxiosDetail(err), t, 'simple.v2.exportGenerate.error'));
      setExportPhase('idle');
    }
  }

  async function pollExportTask(taskId: string) {
    while (!cancelledRef.current) {
      await new Promise((r) => setTimeout(r, 1000));
      if (cancelledRef.current) return;

      try {
        const res = await axios.get(`/api/task/${taskId}`);
        const task = res.data;
        if (cancelledRef.current) return;

        if (task.progress !== undefined) setProgress(Math.round(task.progress * 100));

        if (task.status === 'complete') {
          const result = task.result;
          if (result?.zip_path) {
            setDownloadUrl(`/api/model/export/scaffold-zip/download?path=${encodeURIComponent(result.zip_path)}`);
          }
          setExportTaskId('');
          setExportPhase('done');
          return;
        } else if (task.status === 'error' || task.status === 'cancelled') {
          setError(friendlyError(task.error, t, 'simple.v2.exportGenerate.error'));
          setExportPhase('idle');
          setExportTaskId('');
          return;
        }
      } catch {
        // Transient — keep polling
      }
    }
  }

  // ── On mount: run export-check ──────────────────────────────────
  useEffect(() => {
    if (!hasModel || !targetDevice) return;
    // Don't re-check if already done or exporting
    if (exportPhase === 'exporting' || exportPhase === 'done') return;

    // eslint-disable-next-line react-hooks/set-state-in-effect -- run persisted export check on page entry
    checkCompat();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Resume polling if exportTaskId persisted ────────────────────
  useEffect(() => {
    if (!exportTaskId || exportPhase === 'done') return;
    /* eslint-disable react-hooks/set-state-in-effect -- resume persisted export task on page entry */
    setExportPhase('exporting');
    pollExportTask(exportTaskId);
    /* eslint-enable react-hooks/set-state-in-effect */
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Set default app name ────────────────────────────────────────
  useEffect(() => {
    if (!appName) setAppName(t('simple.v2.exportGenerate.appNameDefault'));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Warn before closing tab during export ───────────────────────
  useEffect(() => {
    if (exportPhase !== 'exporting') return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [exportPhase]);

  // ── Export ───────────────────────────────────────────────────────
  const handleExport = async () => {
    if (!loadedModelId) return;
    setExportPhase('exporting');
    setError('');
    setProgress(0);
    cancelledRef.current = false;

    try {
      const encodedId = encodeURIComponent(loadedModelId);
      const res = await axios.post(`/api/model/${encodedId}/export/scaffold-zip`, {
        app_name: appName || t('simple.v2.exportGenerate.appNameDefault'),
        system_prompt: 'You are a helpful assistant.',
        model_tier: tier,
        enable_h2o: true,
      });
      const taskId = res.data.task_id;
      setExportTaskId(taskId);
      pollExportTask(taskId);
    } catch (err: unknown) {
      setError(friendlyError(getAxiosDetail(err), t, 'simple.v2.exportGenerate.error'));
      setExportPhase('idle');
    }
  };

  // ── No model guard UI ───────────────────────────────────────────
  if (!hasModel) {
    return (
      <WizardShell
        steps={EXPORT_STEPS(t)}
        currentStep={1}
        onBack={() => navigate('/simple/export/device')}
        helpKey="simple.v2.help.exportGenerate"
        >
        <div className="flex flex-col items-center gap-6 py-12 text-center">
          <XCircle size={48} className="text-stone-300 dark:text-stone-600" />
          <h2 className="text-xl font-semibold text-stone-900 dark:text-stone-100">
            {t('simple.v2.exportGenerate.noModel')}
          </h2>
          <p className="text-stone-500 dark:text-stone-400">
            {t('simple.v2.exportGenerate.noModelDesc')}
          </p>
          <button
            type="button"
            onClick={() => navigate('/simple/setup')}
            className="rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            {t('simple.v2.exportGenerate.goSetup')}
          </button>
        </div>
      </WizardShell>
    );
  }

  // ── Render ──────────────────────────────────────────────────────
  return (
    <WizardShell
      steps={EXPORT_STEPS(t)}
      currentStep={1}
      onBack={() => navigate('/simple/export/device')}
      helpKey="simple.v2.help.exportGenerate"
    >
      <div className="flex flex-col items-center gap-8 py-4">
        {/* Header */}
        <div className="text-center">
          <h1 className="mb-2 text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
            {t('simple.v2.exportGenerate.title')}
          </h1>
          <p className="text-stone-500 dark:text-stone-400">
            {t('simple.v2.exportGenerate.subtitle')}
          </p>
        </div>

        {/* === Phase: Checking compatibility === */}
        {exportPhase === 'checking' && (
          <div className="flex flex-col items-center gap-4 py-8">
            <Loader2 size={32} className="animate-spin text-stone-400" />
            <p className="text-stone-500 dark:text-stone-400">
              {t('simple.v2.exportGenerate.checking')}
            </p>
          </div>
        )}

        {/* === Phase: Idle (show check result + export form) === */}
        {exportPhase === 'idle' && exportCheck && (
          <div className="flex w-full max-w-md flex-col gap-6">
            {/* Compatibility result card */}
            <CompatCard check={exportCheck} t={t} />

            {/* Error */}
            {error && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
                {error}
              </div>
            )}

            {/* Action based on suggestion */}
            {exportCheck.suggestion === 'direct' && (
              <ExportForm
                appName={appName}
                setAppName={setAppName}
                onExport={handleExport}
                t={t}
              />
            )}

            {exportCheck.suggestion === 'downgrade' && (
              <div className="flex flex-col gap-3">
                <button
                  type="button"
                  onClick={() => navigate('/simple/tier')}
                  className="flex items-center justify-center gap-2 rounded-xl bg-amber-500 px-6 py-3 text-sm font-medium text-white hover:bg-amber-600"
                >
                  <ArrowDownCircle size={16} />
                  {t('simple.v2.exportGenerate.downgrade', { tier: exportCheck.suggested_tier })}
                </button>
                <button
                  type="button"
                  onClick={() => navigate('/simple/export/device')}
                  className="text-sm text-stone-500 underline underline-offset-2 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-200"
                >
                  {t('simple.v2.exportGenerate.changeDevice')}
                </button>
              </div>
            )}

            {exportCheck.suggestion === 'change_focus' && (
              <div className="flex flex-col gap-3">
                <button
                  type="button"
                  onClick={() => navigate('/simple/focus')}
                  className="flex items-center justify-center gap-2 rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
                >
                  <RefreshCw size={16} />
                  {t('simple.v2.exportGenerate.changeFocus')}
                </button>
                <button
                  type="button"
                  onClick={() => navigate('/simple/export/device')}
                  className="text-sm text-stone-500 underline underline-offset-2 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-200"
                >
                  {t('simple.v2.exportGenerate.changeDevice')}
                </button>
              </div>
            )}
          </div>
        )}

        {/* === Phase: Idle with error but no check result === */}
        {exportPhase === 'idle' && !exportCheck && error && (
          <div className="flex flex-col items-center gap-4">
            <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
              {error}
            </div>
            <button
              type="button"
              onClick={checkCompat}
              className="rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
            >
              {t('simple.v2.exportGenerate.retry')}
            </button>
          </div>
        )}

        {/* === Phase: Exporting === */}
        {exportPhase === 'exporting' && (
          <div className="flex w-full max-w-md flex-col items-center gap-6 py-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20">
              <Package size={28} className="animate-pulse text-blue-500 dark:text-blue-400" />
            </div>
            <p className="font-medium text-stone-900 dark:text-stone-100">
              {t('simple.v2.exportGenerate.packagingApp')}
            </p>

            {/* Progress bar */}
            <div className="w-full">
              <div className="mb-2 flex justify-between text-xs text-stone-500">
                <span>{t('simple.v2.exportGenerate.generating')}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-stone-200 dark:bg-stone-800">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* === Phase: Done === */}
        {exportPhase === 'done' && (
          <div className="flex w-full max-w-md flex-col gap-6 animate-[fadeInUp_0.5s_ease-out]">
            {/* Success celebration */}
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="flex h-16 w-16 animate-bounce items-center justify-center rounded-2xl bg-gradient-to-br from-green-100 to-emerald-100 [animation-iteration-count:3] dark:from-green-900/30 dark:to-emerald-900/30">
                <Sparkles size={28} className="text-green-600 dark:text-green-400" />
              </div>
              <h2 className="text-2xl font-semibold text-stone-900 dark:text-stone-100">
                {t('simple.v2.exportGenerate.done')}
              </h2>
            </div>

            {/* Download card */}
            <div className="overflow-hidden rounded-2xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
              <div className="flex flex-col gap-5 p-6">
                {/* App name (read-only display) */}
                <div className="text-center">
                  <p className="text-sm text-stone-500 dark:text-stone-400">
                    {t('simple.v2.exportGenerate.appName')}
                  </p>
                  <p className="mt-1 text-lg font-semibold text-stone-900 dark:text-stone-100">
                    {appName}
                  </p>
                </div>

                {/* Download button */}
                {downloadUrl && (
                  <a
                    href={downloadUrl}
                    download
                    className="flex items-center justify-center gap-2 rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
                  >
                    <Download size={16} />
                    {t('simple.v2.exportGenerate.downloadZip')}
                  </a>
                )}

                {/* Next steps 1-2-3 */}
                <div>
                  <p className="mb-3 text-sm font-medium text-stone-700 dark:text-stone-300">
                    {t('simple.v2.exportGenerate.nextSteps')}
                  </p>
                  <ol className="space-y-2">
                    {['step1', 'step2', 'step3'].map((key, i) => (
                      <li key={key} className="flex items-start gap-3 text-sm text-stone-600 dark:text-stone-400">
                        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-stone-100 text-xs font-semibold text-stone-600 dark:bg-stone-800 dark:text-stone-300">
                          {i + 1}
                        </span>
                        <span className="pt-0.5">{t(`simple.v2.exportGenerate.${key}`)}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            </div>

            {/* Footer: powered by + features */}
            <div className="text-center text-xs text-stone-400 dark:text-stone-500">
              <p className="font-medium">{t('simple.v2.exportGenerate.poweredBy')}</p>
              <p className="mt-1">{t('simple.v2.exportGenerate.features')}</p>
            </div>

            {/* Export another */}
            <button
              type="button"
              onClick={() => {
                setExportPhase('idle');
                setDownloadUrl('');
                setExportCheck(null);
                navigate('/simple/export/device');
              }}
              className="text-sm text-stone-500 underline underline-offset-2 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-200"
            >
              {t('simple.v2.exportGenerate.exportAnother')}
            </button>
          </div>
        )}
      </div>
    </WizardShell>
  );
}

// ── Sub-components ──────────────────────────────────────────────────

/** Compatibility result indicator card */
function CompatCard({ check, t }: { check: ExportCheck; t: (k: string, vars?: Record<string, string>) => string }) {
  if (check.suggestion === 'direct') {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-green-200 bg-green-50 p-4 dark:border-green-800 dark:bg-green-900/20">
        <CheckCircle2 size={20} className="mt-0.5 shrink-0 text-green-600 dark:text-green-400" />
        <div>
          <p className="font-medium text-green-700 dark:text-green-300">
            {t('simple.v2.exportGenerate.fits')}
          </p>
          <p className="mt-0.5 text-sm text-green-600/80 dark:text-green-400/70">
            {t('simple.v2.exportGenerate.fitsDesc')}
          </p>
        </div>
      </div>
    );
  }

  if (check.suggestion === 'downgrade') {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-900/20">
        <AlertTriangle size={20} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
        <div>
          <p className="font-medium text-amber-700 dark:text-amber-300">
            {t('simple.v2.exportGenerate.tooLarge', { tier: check.suggested_tier })}
          </p>
          {check.download_size_gb > 0 && (
            <p className="mt-0.5 text-sm text-amber-600/80 dark:text-amber-400/70">
              {t('simple.v2.exportGenerate.tooLargeDesc', { size: check.download_size_gb.toFixed(1) })}
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 dark:border-red-800 dark:bg-red-900/20">
      <XCircle size={20} className="mt-0.5 shrink-0 text-red-600 dark:text-red-400" />
      <div>
        <p className="font-medium text-red-700 dark:text-red-300">
          {t('simple.v2.exportGenerate.noFit')}
        </p>
        <p className="mt-0.5 text-sm text-red-600/80 dark:text-red-400/70">
          {t('simple.v2.exportGenerate.noFitDesc')}
        </p>
      </div>
    </div>
  );
}

/** App name + Generate button form */
function ExportForm({
  appName,
  setAppName,
  onExport,
  t,
}: {
  appName: string;
  setAppName: (name: string) => void;
  onExport: () => void;
  t: (k: string) => string;
}) {
  return (
    <div className="flex flex-col gap-4">
      {/* App name input */}
      <div>
        <label
          htmlFor="app-name"
          className="mb-1.5 block text-sm font-medium text-stone-700 dark:text-stone-300"
        >
          {t('simple.v2.exportGenerate.appName')}
        </label>
        <input
          id="app-name"
          type="text"
          value={appName}
          onChange={(e) => setAppName(e.target.value)}
          placeholder={t('simple.v2.exportGenerate.appNameDefault')}
          className="w-full rounded-xl border border-stone-300 bg-white px-4 py-3 text-sm text-stone-900 outline-none transition-colors focus:border-stone-500 focus:ring-2 focus:ring-stone-200 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-100 dark:focus:border-stone-500 dark:focus:ring-stone-700"
        />
      </div>

      {/* Generate button */}
      <button
        type="button"
        onClick={onExport}
        disabled={!appName.trim()}
        className={cn(
          'flex items-center justify-center gap-2 rounded-xl px-6 py-3.5 text-sm font-medium transition-all',
          appName.trim()
            ? 'bg-stone-900 text-white hover:bg-stone-800 hover:shadow-lg dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200'
            : 'bg-stone-200 text-stone-400 dark:bg-stone-800 dark:text-stone-600',
        )}
      >
        <Package size={16} />
        {t('simple.v2.exportGenerate.generate')}
      </button>
    </div>
  );
}
