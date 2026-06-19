// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { useDistillInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { EmptyState } from '@/components/common/EmptyState';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { startDistillation } from '@/api/endpoints';
import { useModelStore } from '@/stores/modelStore';
import { useToastStore } from '@/stores/toastStore';
import { useT, useLocaleStore } from '@/i18n';
import { FolderOpen, GraduationCap, Loader2, Crown, BookOpen, Database, Shield, Sparkles, Send, Square, X as XIcon } from 'lucide-react';
import { FileBrowser } from '@/components/model/FileBrowser';
import type { DistillResult } from '@/api/types';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveDistillationCapabilities,
  assessDistillationConfig,
  buildDistillationContextSnippet,
  buildDistillationAutoBrief,
  getDistillationSuggestedPrompts,
} from '@/lib/distillationInsights';

type PickTarget = 'teacher' | 'student' | 'dataset' | null;

export default function DistillPage() {
  const t = useT();
  const addToast = useToastStore((s) => s.addToast);
  const model = useModelStore((s) => s.currentModel);
  const distillInsights = useDistillInsights(t, model);

  // Form state
  const [teacherDir, setTeacherDir] = useState('');
  const [studentDir, setStudentDir] = useState('');
  const [datasetPath, setDatasetPath] = useState('');
  const [mode] = useState<'offline' | 'taid'>('offline');
  const [numEpochs, setNumEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(4);
  const [learningRate, setLearningRate] = useState(1e-4);
  const [temperature, setTemperature] = useState(2.0);
  const [alpha, setAlpha] = useState(0.5);
  const [maxSamples, setMaxSamples] = useState(0);

  // UI state
  const [pickTarget, setPickTarget] = useState<PickTarget>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<DistillResult | null>(null);

  const canStart = teacherDir && studentDir && datasetPath;

  // ── §9.1 multi-component capability + risk + AI brief ─────────────────
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const distillCaps = useMemo(
    () => deriveDistillationCapabilities(
      teacherDir, studentDir, datasetPath, model, mode, numEpochs, batchSize,
      learningRate, temperature, alpha, maxSamples, result,
    ),
    [teacherDir, studentDir, datasetPath, model, mode, numEpochs, batchSize, learningRate, temperature, alpha, maxSamples, result],
  );
  const distillRisk = useMemo(() => assessDistillationConfig(distillCaps), [distillCaps]);

  const distillSystemPrompt = useMemo(() => {
    if (!model) return '';
    return (
      buildModelSelfSystemPrompt(model, locale) +
      '\n\n' +
      buildDistillationContextSnippet(distillCaps, locale)
    );
  }, [model, distillCaps, locale]);

  const briefChat = useModelChat({
    modelId: model?.model_id || null,
    systemPrompt: distillSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const distillPrompts = useMemo(
    () => getDistillationSuggestedPrompts(distillCaps, locale),
    [distillCaps, locale],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!model) return;
    if (briefChat.streaming) return;
    const sig = `${model.model_id}:${distillCaps.allReady}:${distillCaps.adapterStatus}:${distillCaps.lossBucket}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildDistillationAutoBrief(distillCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, distillCaps.allReady, distillCaps.adapterStatus, distillCaps.lossBucket, locale]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const DISTILL_RISK_BANNER_CLASS: Record<typeof distillRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const handleStart = async () => {
    try {
      const resp = await startDistillation({
        teacher_dir: teacherDir,
        student_dir: studentDir,
        dataset_path: datasetPath,
        mode,
        num_epochs: numEpochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        temperature,
        alpha,
        max_samples: maxSamples,
      });
      setTaskId(resp.task_id);
    } catch {
      addToast('Failed to start distillation', 'error');
    }
  };

  const handleComplete = (raw: unknown) => {
    const r = raw as DistillResult;
    setResult(r);
    if (r?.success) {
      addToast(`Distillation complete: ${r.output_dir}`, 'success');
    }
  };

  const handlePick = (path: string) => {
    if (pickTarget === 'teacher') setTeacherDir(path);
    else if (pickTarget === 'student') setStudentDir(path);
    else if (pickTarget === 'dataset') setDatasetPath(path);
    setPickTarget(null);
  };

  return (
    <div>
      <PageHeader
        title={t('page.distill')}
        description={t('distill.desc')}
      />

      <InsightPanel insights={distillInsights} />

      {/* 4-card identity strip — Teacher / Student / Dataset / Sovereignty (§9.1 + §9.2) */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Crown size={16} />}
          label={t('distill.cardTeacher')}
          value={distillCaps.teacherName || t('distill.empty')}
          hint={distillCaps.hasTeacher ? distillCaps.teacherDir : t('distill.teacherHint')}
          tone={distillCaps.hasTeacher ? 'indigo' : 'neutral'}
        />
        <IdentityCard
          icon={<BookOpen size={16} />}
          label={t('distill.cardStudent')}
          value={distillCaps.studentName || t('distill.empty')}
          hint={distillCaps.hasStudent ? distillCaps.studentDir : t('distill.studentHint')}
          tone={distillCaps.teacherSameAsStudent ? 'red' : (distillCaps.hasStudent ? 'emerald' : 'neutral')}
        />
        <IdentityCard
          icon={<Database size={16} />}
          label={t('distill.cardDataset')}
          value={distillCaps.datasetName || t('distill.empty')}
          hint={distillCaps.hasDataset
            ? distillCaps.datasetPath
            : t('distill.datasetHint')}
          tone={distillCaps.hasDataset ? 'emerald' : 'amber'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('distill.cardSovereignty')}
          value={t('distill.zeroCloud')}
          hint={t('distill.sovereigntyHint')}
          tone="emerald"
        />
      </div>

      {distillRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${DISTILL_RISK_BANNER_CLASS[distillRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {distillRisk.level === 'danger' ? t('distill.riskDanger') : t('distill.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? distillRisk.reasonZh : distillRisk.reason}
        </div>
      )}

      {model && (
        <ModelBriefCard
          className="mb-6"
          label={t('distill.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('distill.briefEmpty')}
          streamingText={t('distill.briefThinking')}
          refreshTitle={t('distill.briefRefire')}
          prompts={distillPrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildDistillationAutoBrief(distillCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Left: Config */}
        <div className="space-y-4">
          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="mb-4 text-sm font-semibold text-gray-700 uppercase tracking-wide">Models</h3>

            {/* Teacher */}
            <label className="block text-sm font-medium text-gray-700 mb-1">Teacher Model (large)</label>
            <div className="flex gap-2 mb-3">
              <input
                value={teacherDir}
                onChange={(e) => setTeacherDir(e.target.value)}
                placeholder="/path/to/teacher-model"
                className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              />
              <button
                onClick={() => setPickTarget('teacher')}
                className="rounded-lg border border-gray-300 px-3 py-2 text-gray-500 hover:bg-gray-50"
              >
                <FolderOpen size={16} />
              </button>
            </div>

            {/* Student */}
            <label className="block text-sm font-medium text-gray-700 mb-1">Student Model (small)</label>
            <div className="flex gap-2 mb-3">
              <input
                value={studentDir}
                onChange={(e) => setStudentDir(e.target.value)}
                placeholder="/path/to/student-model"
                className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              />
              <button
                onClick={() => setPickTarget('student')}
                className="rounded-lg border border-gray-300 px-3 py-2 text-gray-500 hover:bg-gray-50"
              >
                <FolderOpen size={16} />
              </button>
            </div>

            {/* Dataset */}
            <label className="block text-sm font-medium text-gray-700 mb-1">Dataset (.jsonl / .parquet)</label>
            <div className="flex gap-2">
              <input
                value={datasetPath}
                onChange={(e) => setDatasetPath(e.target.value)}
                placeholder="/path/to/dataset.jsonl"
                className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none"
              />
              <button
                onClick={() => setPickTarget('dataset')}
                className="rounded-lg border border-gray-300 px-3 py-2 text-gray-500 hover:bg-gray-50"
              >
                <FolderOpen size={16} />
              </button>
            </div>
          </div>

          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="mb-4 text-sm font-semibold text-gray-700 uppercase tracking-wide">Parameters</h3>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Epochs</label>
                <input type="number" value={numEpochs} onChange={(e) => setNumEpochs(+e.target.value)}
                  min={1} max={100} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Batch Size</label>
                <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)}
                  min={1} max={64} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Learning Rate</label>
                <input type="number" value={learningRate} onChange={(e) => setLearningRate(+e.target.value)}
                  step={0.0001} min={0} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Temperature</label>
                <input type="number" value={temperature} onChange={(e) => setTemperature(+e.target.value)}
                  step={0.1} min={0.1} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Alpha (KL weight)</label>
                <input type="number" value={alpha} onChange={(e) => setAlpha(+e.target.value)}
                  step={0.1} min={0} max={1} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Max Samples (0=all)</label>
                <input type="number" value={maxSamples} onChange={(e) => setMaxSamples(+e.target.value)}
                  min={0} className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm" />
              </div>
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={!canStart || !!taskId}
            className="w-full flex items-center justify-center gap-2 rounded-lg bg-indigo-500 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
          >
            {taskId ? <Loader2 size={16} className="animate-spin" /> : <GraduationCap size={16} />}
            {taskId ? 'Running...' : 'Start Distillation'}
          </button>
        </div>

        {/* Right: Result */}
        <div>
          {result ? (
            <div className="rounded-xl border border-gray-200 bg-white p-5">
              <h3 className="mb-3 text-sm font-semibold text-gray-700 uppercase tracking-wide">Result</h3>
              {result.success ? (
                <div className="space-y-2 text-sm">
                  <p><span className="text-gray-500">Teacher:</span> {result.teacher_name}</p>
                  <p><span className="text-gray-500">Student:</span> {result.student_name}</p>
                  <p><span className="text-gray-500">Output:</span> <code className="text-xs bg-gray-100 px-1 rounded">{result.output_dir}</code></p>
                  <p><span className="text-gray-500">Epochs:</span> {result.num_epochs}</p>
                  <p><span className="text-gray-500">Steps:</span> {result.total_steps}</p>
                  <p><span className="text-gray-500">Final Loss:</span> {result.final_loss}</p>
                  <p><span className="text-gray-500">KL Loss:</span> {result.final_kl_loss}</p>
                  <p><span className="text-gray-500">CE Loss:</span> {result.final_ce_loss}</p>
                  <p><span className="text-gray-500">Duration:</span> {result.duration_seconds}s</p>
                  <p><span className="text-gray-500">Samples:</span> {result.dataset_samples}</p>

                  {result.warning && (
                    <div className="mt-3 rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-xs text-yellow-800">
                      {result.warning}
                    </div>
                  )}

                  {result.loss_history.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-xs font-semibold text-gray-600 mb-2">Loss History</h4>
                      <div className="max-h-48 overflow-y-auto">
                        <table className="w-full text-xs">
                          <thead className="text-gray-500">
                            <tr>
                              <th className="text-left py-1">Epoch</th>
                              <th className="text-left py-1">Step</th>
                              <th className="text-right py-1">Loss</th>
                              <th className="text-right py-1">KL</th>
                              <th className="text-right py-1">CE</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.loss_history.map((h, i) => (
                              <tr key={i} className="border-t border-gray-100">
                                <td className="py-1">{h.epoch}</td>
                                <td className="py-1">{h.step}</td>
                                <td className="text-right py-1">{h.loss}</td>
                                <td className="text-right py-1">{h.kl_loss}</td>
                                <td className="text-right py-1">{h.ce_loss}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm text-red-600">{result.error}</p>
              )}
            </div>
          ) : (
            <EmptyState
              icon={<GraduationCap size={40} />}
              title="Knowledge Distillation"
              description="Transfer knowledge from a large teacher model to a smaller student model. Select teacher, student, and dataset to begin."
            />
          )}
        </div>
      </div>

      {/* File picker modal */}
      {pickTarget && (
        <FileBrowser
          onSelect={handlePick}
          onCancel={() => setPickTarget(null)}
        />
      )}

      {/* Progress overlay */}
      <ProgressOverlay
        taskId={taskId}
        title="Knowledge Distillation"
        onComplete={handleComplete}
        onError={(err) => addToast(err, 'error')}
        onClose={() => setTaskId(null)}
      />

      {/* Ask Model FAB — student speaks as itself about distillation */}
      {model && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('distill.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{model.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <GraduationCap size={13} className="text-indigo-500" />
                  {t('distill.askDrawerTitle')}
                  <span className="text-[10px] font-normal text-stone-400">[{model.model_name}]</span>
                </div>
                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-200"
                >
                  <XIcon size={14} />
                </button>
              </div>
              <div className="max-h-[50vh] min-h-[8rem] overflow-y-auto px-3 py-3 text-sm leading-relaxed text-stone-700 dark:text-stone-200">
                {briefChat.text ? (
                  <div className="whitespace-pre-wrap">{briefChat.text}</div>
                ) : (
                  <p className="text-xs text-stone-400">{t('distill.askDrawerHint')}</p>
                )}
                {briefChat.streaming && <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-indigo-400" />}
              </div>
              <div className="border-t border-stone-200 p-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={drawerInput}
                    onChange={(e) => setDrawerInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSendBriefDrawer(); } }}
                    placeholder={t('distill.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('distill.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('distill.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {distillPrompts.slice(0, 4).map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => { briefChat.reset(); briefChat.send(p.prompt); }}
                      disabled={briefChat.streaming}
                      className="rounded-full border border-stone-200 bg-stone-50 px-2 py-0.5 text-[10px] font-medium text-stone-600 hover:bg-stone-100 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300 dark:hover:bg-stone-700"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
