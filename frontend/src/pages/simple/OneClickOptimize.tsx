// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 3: One-Click Optimize — show optimization plan and execute.
 */

import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Zap, Download, Check, AlertTriangle, X, Terminal as TerminalIcon, Trash2 } from 'lucide-react';
import { useWizardStore } from '@/stores/wizardStore';
import { useModelStore } from '@/stores/modelStore';
import { WizardShell } from '@/components/common/WizardShell';
import { Terminal } from '@/components/common/Terminal';
import { useT } from '@/i18n';
import { WIZARD_STEPS } from './wizardSteps';
import { cn } from '@/lib/utils';
import axios from 'axios';
import { runTerminalCommand, closeTerminal, checkLocalPath, deleteLocalModel } from '@/api/endpoints';

function getErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data?.detail || error.message || fallback;
  }
  return error instanceof Error ? error.message : fallback;
}

export default function OneClickOptimize() {
  const t = useT();
  const navigate = useNavigate();
  const {
    selectedModel, targetDevice,
    optimizationRec, setOptimizationRec,
    optimizationDone, setOptimizationDone,
    setLoadedModelDir,
    loadedModelId, setLoadedModelId,
    downloadTaskId, setDownloadTaskId,
    setCurrentStep,
  } = useWizardStore();
  const setCurrentModel = useModelStore((s) => s.setCurrentModel);

  const [phase, setPhase] = useState<'plan' | 'downloading' | 'loading' | 'ready'>('plan');
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const [error, setError] = useState('');
  const taskIdRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);

  // Terminal mode state (default ON for better UX)
  const [useTerminal, setUseTerminal] = useState(true);
  const [terminalSessionId, setTerminalSessionId] = useState<string | null>(null);
  const [startingTerminal, setStartingTerminal] = useState(false);

  // Incomplete download state
  const [incompleteDownload, setIncompleteDownload] = useState<{ path: string; size_bytes: number } | null>(null);
  const [clearingCache, setClearingCache] = useState(false);

  // Fetch optimization recommendation
  useEffect(() => {
    if (!selectedModel || optimizationRec) return;

    const fetchRec = async () => {
      try {
        const res = await axios.post('/api/recommend/optimization', {
          model_size_gb: selectedModel.estimated_size_gb,
          device_name: targetDevice,
          current_bits: 4,  // most recommended models are 4-bit
        });
        setOptimizationRec(res.data);
      } catch {
        // If recommendation fails, still allow proceeding
      }
    };
    fetchRec();
  }, [selectedModel, targetDevice]); // eslint-disable-line react-hooks/exhaustive-deps

  const [actionInProgress, setActionInProgress] = useState(false);

  const skipLocalCheckRef = useRef(false);

  const handleDownloadAndLoad = async () => {
    if (!selectedModel || actionInProgress) return;
    setError('');
    setActionInProgress(true);
    cancelledRef.current = false;

    const repoId = selectedModel.download_hint;
    const localName = repoId.replace('/', '_');
    const localDir = `~/mlx-community/${localName}`;

    // Check local state: complete / incomplete / missing
    if (skipLocalCheckRef.current) {
      skipLocalCheckRef.current = false;
    } else try {
      const check = await checkLocalPath(localDir);
      if (check.exists && check.complete) {
        // Complete download — load directly
        setPhase('loading');
        try {
          const loadRes = await axios.post('/api/model/load', { model_dir: check.path || localDir });
          setCurrentModel(loadRes.data);
          setLoadedModelId(loadRes.data.model_id);
          setLoadedModelDir(check.path || localDir);
          setPhase('ready');
          setOptimizationDone(true);
          return;
        } catch {
          setPhase('plan');
        }
      } else if (check.exists && !check.complete) {
        // Incomplete download — show cleanup option
        setIncompleteDownload({ path: check.path || localDir, size_bytes: check.size_bytes });
        setActionInProgress(false);
        return;
      }
    } catch {
      // Check failed — proceed with download
    }

    // Probe HuggingFace reachability, auto-switch mirror for China mainland
    let mirror: 'official' | 'hf-mirror' | 'modelscope' = 'official';
    try {
      const probe = await axios.get('/api/hf/probe');
      if (!probe.data.reachable && probe.data.suggestion) {
        mirror = probe.data.suggestion;
      }
    } catch {
      mirror = 'hf-mirror'; // Network error → assume blocked
    }
    if (useTerminal) {
      // Terminal mode: run download script in PTY
      // Pass cmd as array (not bash -c string) to prevent shell injection
      setStartingTerminal(true);
      try {
        const repoId = selectedModel.download_hint;
        const localName = repoId.replace('/', '_');
        const localDir = `~/mlx-community/${localName}`;

        // China mainland (hf-mirror/modelscope): msd.sh (ModelScope, faster in China)
        // Official: hfd.sh (direct HuggingFace)
        const useChinaSource = mirror === 'hf-mirror' || mirror === 'modelscope';
        let cmd: string[];
        if (useChinaSource) {
          cmd = ['bash', 'scripts/msd.sh', repoId, '--local-dir', localDir];
        } else {
          cmd = ['bash', 'scripts/hfd.sh', repoId, '--local-dir', localDir];
        }

        const { session_id } = await runTerminalCommand(cmd);
        setTerminalSessionId(session_id);
      } catch {
        setError(t('simple.v1.terminalFailed'));
      } finally {
        setStartingTerminal(false);
        setActionInProgress(false);
      }
      return;
    }

    // Background mode: use task system
    setPhase('downloading');
    setProgress(0);
    setProgressMsg('');

    try {
      // Start HF download with auto-detected mirror
      const dlRes = await axios.post('/api/hf/download', {
        repo_id: selectedModel.download_hint,
        mirror,
      });
      const taskId = dlRes.data.task_id;
      taskIdRef.current = taskId;
      setDownloadTaskId(taskId);

      // Poll task progress
      let done = false;
      let downloadedDir = '';
      while (!done) {
        await new Promise((r) => setTimeout(r, 1000));
        if (cancelledRef.current) return;
        try {
          const statusRes = await axios.get(`/api/task/${taskId}`);
          const task = statusRes.data;
          if (cancelledRef.current) return;
          if (task.progress !== undefined) {
            setProgress(Math.round(task.progress * 100));
          }
          if (task.message) {
            setProgressMsg(task.message);
          }
          if (task.status === 'complete') {
            done = true;
            const r = task.result;
            downloadedDir = r?.path || r?.model_dir || (typeof r === 'string' ? r : '');
            setLoadedModelDir(downloadedDir);
          } else if (task.status === 'error' || task.status === 'cancelled') {
            throw new Error(task.error || 'Download failed');
          }
        } catch (pollErr: unknown) {
          if (cancelledRef.current) return;
          if (pollErr instanceof Error && pollErr.message && pollErr.message !== 'Download failed') {
            continue;
          }
          throw pollErr;
        }
      }

      if (cancelledRef.current) return;

      // Load the model
      setPhase('loading');
      setProgress(0);
      const loadRes = await axios.post('/api/model/load', { model_dir: downloadedDir || selectedModel.download_hint });
      if (cancelledRef.current) return;
      setCurrentModel(loadRes.data);
      setLoadedModelId(loadRes.data.model_id);

      setPhase('ready');
      setOptimizationDone(true);
      setDownloadTaskId('');
    } catch (err: unknown) {
      if (cancelledRef.current) return;
      setError(getErrorMessage(err, 'Failed'));
      setPhase('plan');
      setDownloadTaskId('');
    } finally {
      setActionInProgress(false);
    }
  };

  const handleClearCache = async () => {
    if (!incompleteDownload) return;
    setClearingCache(true);
    try {
      await deleteLocalModel(incompleteDownload.path);
      setIncompleteDownload(null);
      // Now trigger download
      handleDownloadAndLoad();
    } catch {
      setError(t('simple.v1.clearCacheFailed'));
    } finally {
      setClearingCache(false);
    }
  };

  const handleResumeDownload = () => {
    setIncompleteDownload(null);
    skipLocalCheckRef.current = true;
    handleDownloadAndLoad();
  };

  const handleCancel = async () => {
    cancelledRef.current = true;
    const taskId = taskIdRef.current;
    if (taskId) {
      try {
        await axios.delete(`/api/task/${taskId}`);
      } catch {
        // Best effort cancel
      }
    }
    taskIdRef.current = null;
    setDownloadTaskId('');
    setPhase('plan');
    setProgress(0);
    setProgressMsg('');
    setError('');
  };

  const handleTerminalExit = async (code: number) => {
    // Close terminal first
    if (terminalSessionId) {
      closeTerminal(terminalSessionId).catch(() => {});
      setTerminalSessionId(null);
    }

    if (code === 0 && selectedModel) {
      // Download succeeded, load the model
      const localName = selectedModel.download_hint.replace('/', '_');
      // Use ~ path - backend will expand it
      const downloadedDir = `~/mlx-community/${localName}`;
      setLoadedModelDir(downloadedDir);

      setPhase('loading');
      try {
        const loadRes = await axios.post('/api/model/load', { model_dir: downloadedDir });
        setCurrentModel(loadRes.data);
        setLoadedModelId(loadRes.data.model_id);
        setPhase('ready');
        setOptimizationDone(true);
      } catch (err: unknown) {
        setError(getErrorMessage(err, 'Failed to load model'));
        setPhase('plan');
      }
    } else if (code !== 0) {
      setError(t('simple.v1.downloadExitCode', { code: String(code) }));
    }
  };

  const handleTerminalClose = () => {
    if (terminalSessionId) {
      closeTerminal(terminalSessionId).catch(() => {});
    }
    setTerminalSessionId(null);
  };

  // If model already loaded, skip download
  useEffect(() => {
    if (loadedModelId && !optimizationDone) {
      setOptimizationDone(true);
      setPhase('ready');
    }
  }, [loadedModelId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Resume polling if there's an active download task (user navigated away and back)
  useEffect(() => {
    if (!downloadTaskId || optimizationDone || loadedModelId) return;
    let cancelled = false;
    taskIdRef.current = downloadTaskId;
    setPhase('downloading');

    const poll = async () => {
      while (!cancelled) {
        await new Promise((r) => setTimeout(r, 1000));
        if (cancelled) return;
        try {
          const res = await axios.get(`/api/task/${downloadTaskId}`);
          const task = res.data;
          if (cancelled) return;
          if (task.progress !== undefined) setProgress(Math.round(task.progress * 100));
          if (task.message) setProgressMsg(task.message);
          if (task.status === 'complete') {
            const r = task.result;
            const dir = r?.path || r?.model_dir || (typeof r === 'string' ? r : '');
            setLoadedModelDir(dir);
            setDownloadTaskId('');
            // Load the model
            setPhase('loading');
            try {
              const loadRes = await axios.post('/api/model/load', { model_dir: dir || selectedModel?.download_hint });
              setCurrentModel(loadRes.data);
              setLoadedModelId(loadRes.data.model_id);
              setPhase('ready');
              setOptimizationDone(true);
            } catch (e: unknown) {
              setError(getErrorMessage(e, 'Failed to load model'));
              setPhase('plan');
            }
            return;
          } else if (task.status === 'error' || task.status === 'cancelled') {
            setError(task.error || 'Download failed');
            setPhase('plan');
            setDownloadTaskId('');
            return;
          }
        } catch {
          // Task might have been cleaned up
          setPhase('plan');
          setDownloadTaskId('');
          return;
        }
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleNext = () => {
    setCurrentStep(4);
    navigate('/simple/test');
  };

  const RISK_STYLES = {
    low: { icon: <Check size={16} />, color: 'text-green-600 dark:text-green-400', bg: 'bg-green-50 dark:bg-green-900/20' },
    medium: { icon: <AlertTriangle size={16} />, color: 'text-amber-600 dark:text-amber-400', bg: 'bg-amber-50 dark:bg-amber-900/20' },
    high: { icon: <AlertTriangle size={16} />, color: 'text-red-600 dark:text-red-400', bg: 'bg-red-50 dark:bg-red-900/20' },
  };

  return (
    <WizardShell
      steps={WIZARD_STEPS(t)}
      currentStep={3}
      onBack={() => { setCurrentStep(2); navigate('/simple/pick-model'); }}
      onNext={handleNext}
      nextDisabled={!optimizationDone}
      onStepClick={(s) => { setCurrentStep(s); }}
    >
      <div className="text-center">
        <div className="mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-stone-100 dark:bg-stone-800">
          <Zap size={24} className="text-stone-600 dark:text-stone-400" />
        </div>
        <h2 className="mb-1 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.optimize.title')}
        </h2>
        <p className="mb-8 text-stone-500 dark:text-stone-400">
          {selectedModel?.name || 'Model'}
        </p>
      </div>

      {/* Optimization plan */}
      {optimizationRec && (
        <div className="mb-6 space-y-3">
          <div className={cn(
            'rounded-xl p-4',
            RISK_STYLES[optimizationRec.risk_level as keyof typeof RISK_STYLES]?.bg || RISK_STYLES.low.bg,
          )}>
            <div className="mb-2 flex items-center gap-2">
              {RISK_STYLES[optimizationRec.risk_level as keyof typeof RISK_STYLES]?.icon}
              <span className={cn(
                'text-sm font-medium',
                RISK_STYLES[optimizationRec.risk_level as keyof typeof RISK_STYLES]?.color,
              )}>
                {optimizationRec.strategy_name}
              </span>
            </div>
            <p className="text-sm text-stone-600 dark:text-stone-400">
              {optimizationRec.description}
            </p>
          </div>

          {optimizationRec.steps.length > 0 && (
            <div className="space-y-2">
              {optimizationRec.steps.map((step, i) => (
                <div key={i} className="flex items-start gap-3 rounded-lg bg-stone-50 p-3 dark:bg-stone-900">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-stone-200 text-xs font-medium text-stone-700 dark:bg-stone-700 dark:text-stone-300">
                    {i + 1}
                  </span>
                  <span className="text-sm text-stone-700 dark:text-stone-300">{step}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Incomplete download warning */}
      {incompleteDownload && (
        <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/50 dark:bg-amber-900/10">
          <div className="mb-2 flex items-center gap-2 text-amber-800 dark:text-amber-300">
            <AlertTriangle size={16} />
            <span className="text-sm font-medium">{t('simple.v1.incompleteDownload')}</span>
          </div>
          <p className="mb-3 text-xs text-amber-700 dark:text-amber-400">
            {t('simple.v1.incompleteDesc', { size: (incompleteDownload.size_bytes / 1024 ** 3).toFixed(2) })}
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleResumeDownload}
              className="rounded-lg bg-amber-600 px-4 py-2 text-xs font-medium text-white hover:bg-amber-700"
            >
              {t('simple.v1.resumeDownload')}
            </button>
            <button
              type="button"
              onClick={handleClearCache}
              disabled={clearingCache}
              className="flex items-center gap-1.5 rounded-lg border border-amber-300 px-4 py-2 text-xs font-medium text-amber-700 hover:bg-amber-100 disabled:opacity-50 dark:border-amber-700 dark:text-amber-400 dark:hover:bg-amber-900/20"
            >
              <Trash2 size={12} />
              {clearingCache ? t('simple.v1.clearing') : t('simple.v1.clearRedownload')}
            </button>
          </div>
        </div>
      )}

      {/* Action / Progress */}
      {phase === 'plan' && !terminalSessionId && !incompleteDownload && (
        <div className="flex flex-col items-center gap-3">
          <button
            type="button"
            onClick={handleDownloadAndLoad}
            disabled={actionInProgress || startingTerminal}
            className="flex items-center gap-2 rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white transition-all duration-200 hover:bg-stone-800 disabled:opacity-60 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            {startingTerminal ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Download size={18} />
            )}
            {actionInProgress
              ? (startingTerminal ? t('simple.v1.starting') : t('simple.v1.checking'))
              : t('simple.optimize.downloadAndOptimize')}
          </button>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={useTerminal}
              onChange={(e) => setUseTerminal(e.target.checked)}
              className="h-3.5 w-3.5 rounded border-stone-300 text-stone-600 focus:ring-stone-500 dark:border-stone-600"
            />
            <TerminalIcon size={12} className="text-stone-500 dark:text-stone-400" />
            <span className="text-xs text-stone-500 dark:text-stone-400">{t('simple.v1.terminalMode')}</span>
          </label>
        </div>
      )}

      {/* Terminal mode: embedded terminal */}
      {terminalSessionId && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-stone-600 dark:text-stone-400">
              <TerminalIcon size={16} />
              <span className="text-sm font-medium">{t('simple.v1.downloading', { name: selectedModel?.name || 'model' })}</span>
            </div>
            <button
              type="button"
              onClick={handleTerminalClose}
              className="flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs font-medium text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-700 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-300"
            >
              <X size={14} />
              {t('simple.v1.cancel')}
            </button>
          </div>
          <div className="h-80 overflow-hidden rounded-xl border border-stone-200 dark:border-stone-700">
            <Terminal
              sessionId={terminalSessionId}
              onExit={handleTerminalExit}
              className="h-full"
            />
          </div>
        </div>
      )}

      {/* Background mode: progress bar */}
      {!terminalSessionId && (phase === 'downloading' || phase === 'loading') && (
        <div className="space-y-3">
          <div className="flex items-center justify-center gap-2 text-stone-600 dark:text-stone-400">
            <Loader2 size={18} className="animate-spin" />
            <span className="text-sm">
              {phase === 'downloading'
                ? t('simple.optimize.downloading')
                : t('simple.optimize.loading')
              }
            </span>
          </div>
          <div className="mx-auto h-2 max-w-xs overflow-hidden rounded-full bg-stone-100 dark:bg-stone-800">
            <div
              className="h-full rounded-full bg-amber-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          {progress > 0 && (
            <p className="text-center text-xs text-stone-400">
              {progress}%{progressMsg ? ` — ${progressMsg}` : ''}
            </p>
          )}
          <button
            type="button"
            onClick={handleCancel}
            className="mx-auto flex items-center gap-1.5 rounded-lg px-4 py-2 text-xs font-medium text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-700 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-300"
          >
            <X size={14} />
            {t('progress.cancel')}
          </button>
        </div>
      )}

      {phase === 'ready' && (
        <div className="rounded-xl bg-green-50 p-4 text-center dark:bg-green-900/20">
          <Check size={24} className="mx-auto mb-2 text-green-600 dark:text-green-400" />
          <p className="text-sm font-medium text-green-800 dark:text-green-300">
            {t('simple.optimize.ready')}
          </p>
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
