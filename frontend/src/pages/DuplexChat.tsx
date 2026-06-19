// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * DuplexChat — Expert-mode voice duplex page.
 *
 * User manually selects LLM/VLM + ASR + TTS from loaded models,
 * then uses the shared DuplexPanel for voice conversation.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Mic, FolderOpen, Loader2, ImagePlus, X, Brain, Ear, Volume2, Shield, Sparkles, Send, Square } from 'lucide-react';
import { useT, useLocaleStore } from '@/i18n';
import { listLoadedModels, loadModel } from '@/api/endpoints';
import type { ModelInfo } from '@/api/types';
import DuplexPanel from '@/pages/simple/v2/DuplexPanel';
import { FileBrowser } from '@/components/model/FileBrowser';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useModelChat } from '@/hooks/useModelChat';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveDuplexCapabilities,
  assessDuplexConfig,
  buildDuplexContextSnippet,
  buildDuplexAutoBrief,
  getDuplexSuggestedPrompts,
} from '@/lib/duplexInsights';
import { formatParamCount, formatSize } from '@/lib/utils';

export default function DuplexChat() {
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const [loadedModels, setLoadedModels] = useState<ModelInfo[]>([]);
  const [llmModelId, setLlmModelId] = useState('');
  const [asrModelId, setAsrModelId] = useState('');
  const [ttsModelId, setTtsModelId] = useState('');

  // Ask Model FAB drawer state
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // TTS voice/instruct
  const [speakers, setSpeakers] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [instruct, setInstruct] = useState('');

  // Load-from-browser state
  const [loadingSlot, setLoadingSlot] = useState<'llm' | 'asr' | 'tts' | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // VLM image upload
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageThumb, setImageThumb] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Refresh loaded models and auto-select if exactly one per category
  const refreshModels = useCallback(async () => {
    try {
      const models = await listLoadedModels();
      setLoadedModels(models);
      const llms = models.filter((m) => m.model_category === 'llm' || m.model_category === 'vlm');
      const asrs = models.filter((m) => m.model_category === 'stt');
      const ttss = models.filter((m) => m.model_category === 'tts');
      if (llms.length === 1) setLlmModelId(llms[0].model_id);
      if (asrs.length === 1) setAsrModelId(asrs[0].model_id);
      if (ttss.length === 1) setTtsModelId(ttss[0].model_id);
      return models;
    } catch {
      return [];
    }
  }, []);

  // Fetch loaded models on mount
  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  // Handle model load from FileBrowser
  const handleBrowseLoad = useCallback(async (path: string) => {
    setLoadingSlot(null);
    setLoadingModel(true);
    setLoadError(null);
    try {
      const info = await loadModel(path);
      // Refresh list and auto-select based on actual category
      await refreshModels();
      // Also explicitly select the just-loaded model into its slot
      const cat = info.model_category;
      if (cat === 'llm' || cat === 'vlm') setLlmModelId(info.model_id);
      else if (cat === 'stt') setAsrModelId(info.model_id);
      else if (cat === 'tts') setTtsModelId(info.model_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load model';
      setLoadError(msg);
    } finally {
      setLoadingModel(false);
    }
  }, [refreshModels]);

  // Image upload handler for VLM
  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      setImageB64(dataUrl.split(',')[1]);
      setImageThumb(dataUrl);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }, []);

  const clearImage = useCallback(() => {
    setImageB64(null);
    setImageThumb(null);
  }, []);

  // Check if current LLM selection is a VLM
  const selectedLlmModel = loadedModels.find((m) => m.model_id === llmModelId);
  const isVLM = selectedLlmModel?.has_vision ?? false;

  // Fetch TTS voices when TTS model changes
  useEffect(() => {
    if (!ttsModelId) {
      setSpeakers([]);
      setSelectedVoice('');
      return;
    }
    fetch(`/api/chat/${encodeURIComponent(ttsModelId)}/tts-voices`)
      .then((r) => r.json())
      .then((d) => {
        const voices: string[] = d.voices || [];
        setSpeakers(voices);
        if (voices.length > 0) {
          setSelectedVoice(voices[0]);
        } else {
          setSelectedVoice('');
        }
      })
      .catch(() => {
        setSpeakers([]);
        setSelectedVoice('');
      });
  }, [ttsModelId]);

  const llmModels = loadedModels.filter((m) => m.model_category === 'llm' || m.model_category === 'vlm');
  const asrModels = loadedModels.filter((m) => m.model_category === 'stt');
  const ttsModels = loadedModels.filter((m) => m.model_category === 'tts');

  const allReady = !!llmModelId && !!asrModelId && !!ttsModelId;

  // 3-stack capabilities + config risk (north-star §1 on-device sovereignty + §3 common infrastructure)
  const caps = useMemo(
    () => deriveDuplexCapabilities(loadedModels, llmModelId, asrModelId, ttsModelId),
    [loadedModels, llmModelId, asrModelId, ttsModelId],
  );
  const configRisk = useMemo(
    () => assessDuplexConfig(caps, speakers.length),
    [caps, speakers.length],
  );

  // Model-self system prompt for the AI Brief / Ask drawer (LLM is the brain).
  const systemPrompt = useMemo(() => {
    if (!caps.llm) return '';
    return (
      buildModelSelfSystemPrompt(caps.llm, locale) +
      '\n\n' +
      buildDuplexContextSnippet(caps, speakers.length, locale)
    );
  }, [caps, speakers.length, locale]);

  const chat = useModelChat({
    modelId: caps.llm?.model_id || null,
    systemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const suggestedPrompts = useMemo(
    () => getDuplexSuggestedPrompts(caps, speakers.length, locale),
    [caps, speakers.length, locale],
  );

  // Auto-fire brief once per (trio composition + speaker availability) change.
  // playbook §7.5 fire-once + §8.4 stage-aware (trio status as stage).
  useEffect(() => {
    if (!caps.llm) return;
    if (chat.streaming) return;
    const key = `${caps.llm.model_id}:${caps.asr?.model_id || 'none'}:${caps.tts?.model_id || 'none'}:${speakers.length > 0 ? 'voiced' : 'instruct'}:${locale}`;
    if (briefFiredForRef.current === key) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => {
      chat.send(buildDuplexAutoBrief(caps, speakers.length, locale));
    }, 400);
    return () => window.clearTimeout(id);
    // chat & speakers.length captured via caps; intentionally not deps to avoid re-fire on stream tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caps.llm?.model_id, caps.asr?.model_id, caps.tts?.model_id, speakers.length > 0, locale]);

  const handleSendDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || chat.streaming) return;
    chat.send(q);
    setDrawerInput('');
  }, [chat, drawerInput]);

  const RISK_BANNER_CLASS: Record<typeof configRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const selectClass =
    'w-full rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm outline-none transition-colors focus:border-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:focus:border-stone-500';

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-6">
      {/* Header */}
      <div>
        <h1 className="flex items-center gap-2 text-xl font-semibold text-stone-900 dark:text-stone-100">
          <Mic size={22} />
          {t('duplex.title')}
        </h1>
        <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
          {t('duplex.selectModels')}
        </p>
      </div>

      {/* 4-card identity strip — trio + sovereignty (playbook §7.2 / §8.4 stage-aware) */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Brain size={16} />}
          label={t('duplex.cardBrain')}
          value={caps.llm ? caps.llm.model_name : t('duplex.empty')}
          hint={caps.llm ? `${formatParamCount(caps.llm.total_params ?? 0)} · ${formatSize(caps.llm.total_size_bytes ?? 0)}${caps.isVlm ? ' · VLM' : ''}` : t('duplex.notSelected')}
          tone={caps.hasLlm ? (caps.isVlm ? 'indigo' : 'emerald') : 'amber'}
        />
        <IdentityCard
          icon={<Ear size={16} />}
          label={t('duplex.cardEars')}
          value={caps.asr ? caps.asr.model_name : t('duplex.empty')}
          hint={caps.asr ? `${formatParamCount(caps.asr.total_params ?? 0)} · ${formatSize(caps.asr.total_size_bytes ?? 0)}` : t('duplex.notSelected')}
          tone={caps.hasAsr ? 'emerald' : 'amber'}
        />
        <IdentityCard
          icon={<Volume2 size={16} />}
          label={t('duplex.cardVoice')}
          value={caps.tts ? caps.tts.model_name : t('duplex.empty')}
          hint={caps.tts
            ? `${formatParamCount(caps.tts.total_params ?? 0)}${speakers.length > 0 ? ` · ${speakers.length} ${t('duplex.speakers')}` : ` · ${t('duplex.zeroShot')}`}`
            : t('duplex.notSelected')}
          tone={caps.hasTts ? 'emerald' : 'amber'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('duplex.cardSovereignty')}
          value={caps.allReady ? t('duplex.allLocal') : `${[caps.hasLlm, caps.hasAsr, caps.hasTts].filter(Boolean).length}/3`}
          hint={caps.allReady
            ? `${formatSize(caps.totalMemBytes)} · 0 ${t('duplex.cloudCalls')}`
            : t('duplex.partial')}
          tone={caps.allReady ? 'emerald' : 'neutral'}
        />
      </div>

      {/* Config risk banner — surface caution/danger inline (playbook §8.1 risk pattern) */}
      {configRisk.level !== 'safe' && (
        <div className={`rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[configRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {configRisk.level === 'danger' ? t('duplex.riskDanger') : t('duplex.riskCaution')}
          </span>{' '}
          — {locale === 'zh' ? configRisk.reasonZh : configRisk.reason}
        </div>
      )}

      {/* Loading indicator */}
      {loadingModel && (
        <div className="flex items-center gap-2 text-sm text-stone-500">
          <Loader2 size={16} className="animate-spin" />
          {t('common.loading')}
        </div>
      )}

      {/* Load error */}
      {loadError && (
        <p className="text-sm text-red-500">{loadError}</p>
      )}

      {/* Model selectors */}
      <div className="grid gap-4 sm:grid-cols-3">
        {/* LLM/VLM */}
        <div>
          <label className="mb-1 block text-xs font-medium text-stone-600 dark:text-stone-400">
            {t('duplex.llmLabel')}
          </label>
          <div className="flex gap-1.5">
            {llmModels.length === 0 ? (
              <p className="flex-1 py-2 text-xs text-stone-400">{t('duplex.noModels', { category: 'LLM/VLM' })}</p>
            ) : (
              <select className={`${selectClass} flex-1`} value={llmModelId} onChange={(e) => setLlmModelId(e.target.value)}>
                <option value="">{t('duplex.selectModel')}</option>
                {llmModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>{m.model_name}</option>
                ))}
              </select>
            )}
            <button
              type="button"
              onClick={() => setLoadingSlot('llm')}
              disabled={loadingModel}
              title={t('duplex.loadModel')}
              className="shrink-0 rounded-lg border border-stone-200 bg-white p-2 text-stone-500 transition-colors hover:bg-stone-50 hover:text-stone-700 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-200"
            >
              <FolderOpen size={16} />
            </button>
          </div>
        </div>

        {/* ASR */}
        <div>
          <label className="mb-1 block text-xs font-medium text-stone-600 dark:text-stone-400">
            {t('duplex.asrLabel')}
          </label>
          <div className="flex gap-1.5">
            {asrModels.length === 0 ? (
              <p className="flex-1 py-2 text-xs text-stone-400">{t('duplex.noModels', { category: 'ASR' })}</p>
            ) : (
              <select className={`${selectClass} flex-1`} value={asrModelId} onChange={(e) => setAsrModelId(e.target.value)}>
                <option value="">{t('duplex.selectModel')}</option>
                {asrModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>{m.model_name}</option>
                ))}
              </select>
            )}
            <button
              type="button"
              onClick={() => setLoadingSlot('asr')}
              disabled={loadingModel}
              title={t('duplex.loadModel')}
              className="shrink-0 rounded-lg border border-stone-200 bg-white p-2 text-stone-500 transition-colors hover:bg-stone-50 hover:text-stone-700 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-200"
            >
              <FolderOpen size={16} />
            </button>
          </div>
        </div>

        {/* TTS */}
        <div>
          <label className="mb-1 block text-xs font-medium text-stone-600 dark:text-stone-400">
            {t('duplex.ttsLabel')}
          </label>
          <div className="flex gap-1.5">
            {ttsModels.length === 0 ? (
              <p className="flex-1 py-2 text-xs text-stone-400">{t('duplex.noModels', { category: 'TTS' })}</p>
            ) : (
              <select className={`${selectClass} flex-1`} value={ttsModelId} onChange={(e) => setTtsModelId(e.target.value)}>
                <option value="">{t('duplex.selectModel')}</option>
                {ttsModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>{m.model_name}</option>
                ))}
              </select>
            )}
            <button
              type="button"
              onClick={() => setLoadingSlot('tts')}
              disabled={loadingModel}
              title={t('duplex.loadModel')}
              className="shrink-0 rounded-lg border border-stone-200 bg-white p-2 text-stone-500 transition-colors hover:bg-stone-50 hover:text-stone-700 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-400 dark:hover:bg-stone-800 dark:hover:text-stone-200"
            >
              <FolderOpen size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Speaker selector / instruct input */}
      {ttsModelId && (
        <div className="space-y-3">
          {speakers.length > 0 ? (
            <div>
              <label className="mb-1 block text-xs font-medium text-stone-600 dark:text-stone-400">
                {t('duplex.speakerLabel')}
              </label>
              <div className="flex flex-wrap gap-2">
                {speakers.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSelectedVoice(s)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                      selectedVoice === s
                        ? 'bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
                        : 'bg-stone-100 text-stone-600 hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-400 dark:hover:bg-stone-700'
                    }`}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div>
              <label className="mb-1 block text-xs font-medium text-stone-600 dark:text-stone-400">
                {t('duplex.instructLabel')}
              </label>
              <input
                type="text"
                value={instruct}
                onChange={(e) => setInstruct(e.target.value)}
                placeholder={t('duplex.instructPlaceholder')}
                className={selectClass}
              />
            </div>
          )}
        </div>
      )}

      {/* VLM image upload */}
      {isVLM && allReady && (
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-3 py-2 text-xs font-medium text-stone-600 transition-colors hover:bg-stone-50 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-400 dark:hover:bg-stone-800"
          >
            <ImagePlus size={14} />
            {t('simple.v1.uploadImage')}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleImageSelect}
            className="hidden"
          />
          {imageThumb && (
            <div className="relative">
              <img src={imageThumb} alt="" className="h-12 w-12 rounded-lg border border-stone-200 object-cover dark:border-stone-700" />
              <button
                type="button"
                onClick={clearImage}
                className="absolute -right-1.5 -top-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-stone-700 text-white hover:bg-stone-900 dark:bg-stone-400 dark:text-stone-900"
              >
                <X size={10} />
              </button>
            </div>
          )}
        </div>
      )}

      {/* AI Brief — model speaks AS the brain about the trio (playbook §1-D model as interpreter) */}
      {caps.llm && (
        <ModelBriefCard
          label={t('duplex.briefTitle')}
          text={chat.text}
          streaming={chat.streaming}
          emptyText={t('duplex.briefEmpty')}
          streamingText={t('duplex.briefThinking')}
          refreshTitle={t('duplex.briefRefire')}
          prompts={suggestedPrompts}
          onRefresh={() => {
            briefFiredForRef.current = null;
            chat.reset();
            chat.send(buildDuplexAutoBrief(caps, speakers.length, locale));
          }}
          onPrompt={(prompt) => { chat.reset(); chat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {/* Ready indicator + DuplexPanel */}
      {allReady ? (
        <div className="space-y-4">
          <p className="text-center text-sm font-medium text-green-600 dark:text-green-400">
            {t('duplex.ready')}
          </p>
          <DuplexPanel
            asrModelId={asrModelId}
            llmModelId={llmModelId}
            ttsModelId={ttsModelId}
            voice={selectedVoice || undefined}
            instruct={instruct || undefined}
            imageB64={imageB64}
            onImageConsumed={clearImage}
          />
        </div>
      ) : (
        <p className="py-8 text-center text-sm text-stone-400">
          {t('duplex.selectModels')}
        </p>
      )}

      {/* FileBrowser modal */}
      {loadingSlot && (
        <FileBrowser
          onSelect={handleBrowseLoad}
          onCancel={() => setLoadingSlot(null)}
        />
      )}

      {/* Ask Model FAB + drawer (playbook §1-D — always-on conversation with the brain) */}
      {caps.llm && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('duplex.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{caps.llm.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Sparkles size={13} className="text-indigo-500" />
                  {t('duplex.askDrawerTitle')}
                  <span className="text-[10px] font-normal text-stone-400">[{caps.llm.model_name}]</span>
                </div>
                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-200"
                >
                  <X size={14} />
                </button>
              </div>
              <div className="max-h-[50vh] min-h-[8rem] overflow-y-auto px-3 py-3 text-sm leading-relaxed text-stone-700 dark:text-stone-200">
                {chat.text ? (
                  <div className="whitespace-pre-wrap">{chat.text}</div>
                ) : (
                  <p className="text-xs text-stone-400">{t('duplex.askDrawerHint')}</p>
                )}
                {chat.streaming && <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-indigo-400" />}
              </div>
              <div className="border-t border-stone-200 p-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={drawerInput}
                    onChange={(e) => setDrawerInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSendDrawer(); } }}
                    placeholder={t('duplex.askDrawerPlaceholder')}
                    disabled={chat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {chat.streaming ? (
                    <button
                      type="button"
                      onClick={() => chat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('duplex.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('duplex.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {suggestedPrompts.slice(0, 4).map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => { chat.reset(); chat.send(p.prompt); }}
                      disabled={chat.streaming}
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
