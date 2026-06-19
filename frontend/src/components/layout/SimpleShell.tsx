// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * SimpleShell — layout for simple mode wizard pages.
 * No sidebar, clean minimal layout with top step indicator + AI Brief
 * shared by all 7 wizard pages (playbook §9.1 + §9.2 + §9.7).
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import {
  Microscope, Sun, Moon, Compass, Cpu, Star, Shield, Sparkles, Send, Square,
  X as XIcon, ChevronUp, ChevronDown,
} from 'lucide-react';
import { useUIStore } from '@/stores/uiStore';
import { useLocaleStore } from '@/i18n/localeStore';
import { useSimpleStore } from '@/stores/simpleStore';
import { useModelStore } from '@/stores/modelStore';
import { useT } from '@/i18n';
import { useModelChat } from '@/hooks/useModelChat';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveSimpleCapabilities,
  assessWizardProgress,
  buildSimpleWizardContextSnippet,
  buildSimpleWizardAutoBrief,
  getSimpleWizardSuggestedPrompts,
  focusLabel,
  tierLabel,
  stepLabel,
  type SimpleStoreSnapshot,
} from '@/lib/simpleWizardInsights';

export function SimpleShell() {
  const t = useT();
  const location = useLocation();
  const darkMode = useUIStore((s) => s.darkMode);
  const toggleDarkMode = useUIStore((s) => s.toggleDarkMode);
  const setUserMode = useUIStore((s) => s.setUserMode);
  const { locale, setLocale } = useLocaleStore();
  const lang = locale as 'en' | 'zh';

  // ── Simple-mode store snapshot (selectors keep re-renders cheap) ─────
  const deviceProfile = useSimpleStore((s) => s.deviceProfile);
  const focus = useSimpleStore((s) => s.focus);
  const tier = useSimpleStore((s) => s.tier);
  const packages = useSimpleStore((s) => s.packages);
  const setupInfo = useSimpleStore((s) => s.setupInfo);
  const setupPhase = useSimpleStore((s) => s.setupPhase);
  const loadedModelId = useSimpleStore((s) => s.loadedModelId);
  const loadedModelDir = useSimpleStore((s) => s.loadedModelDir);
  const chatTested = useSimpleStore((s) => s.chatTested);
  const ttsModelId = useSimpleStore((s) => s.ttsModelId);
  const duplexLlmModelId = useSimpleStore((s) => s.duplexLlmModelId);
  const duplexAsrModelId = useSimpleStore((s) => s.duplexAsrModelId);
  const duplexDownloadStep = useSimpleStore((s) => s.duplexDownloadStep);
  const targetDevice = useSimpleStore((s) => s.targetDevice);
  const exportCheck = useSimpleStore((s) => s.exportCheck);
  const exportTaskId = useSimpleStore((s) => s.exportTaskId);
  const exportPhase = useSimpleStore((s) => s.exportPhase);
  const appName = useSimpleStore((s) => s.appName);
  const downloadUrl = useSimpleStore((s) => s.downloadUrl);

  const storeSnapshot: SimpleStoreSnapshot = useMemo(() => ({
    deviceProfile,
    focus,
    tier,
    packages,
    setupInfo,
    setupPhase,
    loadedModelId,
    loadedModelDir,
    chatTested,
    ttsModelId,
    duplexLlmModelId,
    duplexAsrModelId,
    duplexDownloadStep,
    targetDevice,
    exportCheck,
    exportTaskId,
    exportPhase,
    appName,
    downloadUrl,
  }), [deviceProfile, focus, tier, packages, setupInfo, setupPhase, loadedModelId, loadedModelDir, chatTested, ttsModelId, duplexLlmModelId, duplexAsrModelId, duplexDownloadStep, targetDevice, exportCheck, exportTaskId, exportPhase, appName, downloadUrl]);

  // ── Brain LLM (currentModel from /chat or /dashboard load) ────────────
  const brainModel = useModelStore((s) => s.currentModel);

  // ── Capabilities + risk + chat ────────────────────────────────────────
  const wizardCaps = useMemo(
    () => deriveSimpleCapabilities(storeSnapshot, location.pathname, brainModel),
    [storeSnapshot, location.pathname, brainModel],
  );
  const wizardRisk = useMemo(() => assessWizardProgress(wizardCaps), [wizardCaps]);

  const wizardSystemPrompt = useMemo(() => {
    if (!brainModel) return '';
    return (
      buildModelSelfSystemPrompt(brainModel, lang) +
      '\n\n' +
      buildSimpleWizardContextSnippet(wizardCaps, storeSnapshot, lang)
    );
  }, [brainModel, wizardCaps, storeSnapshot, lang]);

  const briefChat = useModelChat({
    modelId: brainModel?.model_id || null,
    systemPrompt: wizardSystemPrompt,
    maxTokens: 500,
    temperature: 0.6,
  });

  const wizardPrompts = useMemo(
    () => getSimpleWizardSuggestedPrompts(wizardCaps, storeSnapshot, lang),
    [wizardCaps, storeSnapshot, lang],
  );

  // ── UI state: collapsible brief + Ask FAB ─────────────────────────────
  const [briefCollapsed, setBriefCollapsed] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);
  const lastBriefSigRef = useRef<string | null>(null);

  // Step-aware fire-once: refire when step / phase / locale changes
  useEffect(() => {
    if (!brainModel) return;
    if (briefChat.streaming) return;
    const sig = `${brainModel.model_id}:${wizardCaps.step.key}:${wizardCaps.hasDevice}:${wizardCaps.hasFocus}:${wizardCaps.hasTier}:${storeSnapshot.setupPhase}:${storeSnapshot.exportPhase}:${lang}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => {
      if (lastBriefSigRef.current !== sig) {
        briefChat.reset();
        lastBriefSigRef.current = sig;
      }
      briefChat.send(buildSimpleWizardAutoBrief(wizardCaps, lang));
    }, 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brainModel?.model_id, wizardCaps.step.key, wizardCaps.hasDevice, wizardCaps.hasFocus, wizardCaps.hasTier, storeSnapshot.setupPhase, storeSnapshot.exportPhase, lang]);

  const handleSendBriefDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  const RISK_BANNER_CLASS: Record<typeof wizardRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const ratingLabel = deviceProfile?.ai_rating || '—';
  const stars = deviceProfile?.ai_rating_stars ?? 0;

  return (
    <div className="min-h-screen bg-stone-50 dark:bg-stone-950">
      {/* Minimal top bar */}
      <header className="flex items-center justify-between border-b border-stone-200 bg-white px-4 py-2 dark:border-stone-800 dark:bg-stone-950">
        <div className="flex items-center gap-2">
          <span className="text-base font-semibold text-stone-900 dark:text-stone-100">
            Edge Studio
          </span>
          <span className="rounded-full bg-stone-100 px-2 py-0.5 text-[10px] font-medium text-stone-500 dark:bg-stone-900 dark:text-stone-400">
            {t('mode.beginner')}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Switch to Expert */}
          <button
            type="button"
            onClick={() => { setUserMode('advanced'); window.location.href = '/'; }}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-700 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-300"
            title={t('mode.switchToExpert')}
          >
            <Microscope size={14} />
            {t('mode.expert')}
          </button>

          {/* Language toggle */}
          <button
            type="button"
            onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}
            className="rounded-lg px-2 py-1.5 text-xs font-medium text-stone-500 transition-colors hover:bg-stone-100 dark:text-stone-400 dark:hover:bg-stone-800"
          >
            {locale === 'en' ? '中' : 'EN'}
          </button>

          {/* Dark mode toggle */}
          <button
            type="button"
            onClick={toggleDarkMode}
            className="rounded-lg p-1.5 text-stone-500 transition-colors hover:bg-stone-100 dark:text-stone-400 dark:hover:bg-stone-800"
          >
            {darkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>

      {/* Sticky wizard insight strip — visible on every Simple-mode page */}
      <div className="border-b border-stone-200 bg-white px-4 py-3 dark:border-stone-800 dark:bg-stone-950">
        <div className="mx-auto max-w-5xl space-y-2">
          {/* 4-card identity strip — wizard progress + device + picks + sovereignty */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <IdentityCard
              icon={<Compass size={16} />}
              label={t('simpleWizard.cardStep')}
              value={`${wizardCaps.step.index + 1} / ${wizardCaps.totalSteps}`}
              hint={stepLabel(wizardCaps.step.key, lang)}
              tone="indigo"
            />
            <IdentityCard
              icon={<Cpu size={16} />}
              label={t('simpleWizard.cardDevice')}
              value={wizardCaps.hasDevice ? `${deviceProfile?.chip ?? '—'}` : t('simpleWizard.detecting')}
              hint={wizardCaps.hasDevice
                ? `${deviceProfile?.ram_gb ?? '?'} GB · ${ratingLabel}`
                : t('simpleWizard.deviceHint')}
              tone={wizardCaps.hasDevice ? (stars >= 4 ? 'emerald' : stars >= 2 ? 'indigo' : 'amber') : 'neutral'}
            />
            <IdentityCard
              icon={<Star size={16} />}
              label={t('simpleWizard.cardPicks')}
              value={wizardCaps.hasFocus
                ? `${focusLabel(focus, lang)}${wizardCaps.hasTier ? ` · ${tierLabel(tier, lang)}` : ''}`
                : t('simpleWizard.notPicked')}
              hint={wizardCaps.hasModel
                ? `${t('simpleWizard.loaded')} · ${wizardCaps.loadedName}`
                : (wizardCaps.hasFocus ? t('simpleWizard.tierHint') : t('simpleWizard.focusHint'))}
              tone={wizardCaps.hasModel ? 'emerald' : (wizardCaps.hasFocus && wizardCaps.hasTier ? 'indigo' : 'amber')}
            />
            <IdentityCard
              icon={<Shield size={16} />}
              label={t('simpleWizard.cardSovereignty')}
              value={t('simpleWizard.allLocal')}
              hint={t('simpleWizard.sovereigntyHint')}
              tone="emerald"
            />
          </div>

          {/* Risk / progress banner — shows only when caution / danger */}
          {wizardRisk.level !== 'safe' && (
            <div className={`rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[wizardRisk.level]}`}>
              <span className="font-semibold uppercase tracking-wider">
                {wizardRisk.level === 'danger' ? t('simpleWizard.riskDanger') : t('simpleWizard.riskCaution')}
              </span>
              {' '}— {lang === 'zh' ? wizardRisk.reasonZh : wizardRisk.reason}
            </div>
          )}

          {/* AI Brief — collapsible. Shows only when brain model is loaded. */}
          {brainModel && !briefCollapsed && (
            <ModelBriefCard
              label={t('simpleWizard.briefTitle')}
              status={brainModel.model_name}
              text={briefChat.text}
              streaming={briefChat.streaming}
              emptyText={t('simpleWizard.briefEmpty')}
              streamingText={t('simpleWizard.briefThinking')}
              refreshTitle={t('simpleWizard.briefRefire')}
              prompts={wizardPrompts.slice(0, 4)}
              onRefresh={() => {
                briefFiredForRef.current = null;
                briefChat.reset();
                briefChat.send(buildSimpleWizardAutoBrief(wizardCaps, lang));
              }}
              onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
              actions={(
                <button
                  type="button"
                  onClick={() => setBriefCollapsed(true)}
                  className="rounded p-1 text-indigo-500 hover:bg-indigo-100 dark:hover:bg-indigo-500/10"
                  title={t('simpleWizard.briefCollapse')}
                  aria-label={t('simpleWizard.briefCollapse')}
                >
                  <ChevronUp size={12} />
                </button>
              )}
            />
          )}
          {/* Collapsed brief — show a 1-line summary with expand chevron */}
          {brainModel && briefCollapsed && (
            <button
              type="button"
              onClick={() => setBriefCollapsed(false)}
              className="flex w-full items-center justify-between rounded-lg border border-indigo-100 bg-indigo-50/50 px-3 py-1.5 text-[11px] text-indigo-600 hover:bg-indigo-50 dark:border-indigo-900/40 dark:bg-indigo-950/30 dark:text-indigo-300 dark:hover:bg-indigo-950/50"
            >
              <span className="flex items-center gap-1.5">
                <Sparkles size={12} />
                {t('simpleWizard.briefExpand')}
              </span>
              <ChevronDown size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Page content */}
      <Outlet />

      {/* Ask Model FAB — brain narrates current step. */}
      {brainModel && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('simpleWizard.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brainModel.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Sparkles size={13} className="text-indigo-500" />
                  <div className="min-w-0">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="shrink-0">{t('simpleWizard.askDrawerTitle')}</span>
                      <span className="truncate text-[10px] font-normal text-stone-400 dark:text-stone-500">[{brainModel.model_name}]</span>
                    </div>
                    <p className="truncate text-[10px] font-normal text-stone-400 dark:text-stone-500">
                      {t('simpleWizard.scopeHint')}
                    </p>
                  </div>
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
                  <p className="text-xs text-stone-400">{t('simpleWizard.askDrawerHint')}</p>
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
                    placeholder={t('simpleWizard.askDrawerPlaceholder')}
                    disabled={briefChat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {briefChat.streaming ? (
                    <button
                      type="button"
                      onClick={() => briefChat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('simpleWizard.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendBriefDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('simpleWizard.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {wizardPrompts.slice(0, 4).map((p) => (
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
