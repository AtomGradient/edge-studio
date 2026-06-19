// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Chat — multi-modal chat with the loaded model (LLM / VLM / STT / TTS).
 *
 * Optimization layers (page-optimization-playbook §1):
 *  A. Information archaeology: surface model identity, capabilities, runtime cost.
 *  B. Information design: 4-card identity strip + token-usage bar + per-message TTFT.
 *  C. Visualization: suggested prompts grid (per-category) + STT segment timeline.
 *  D. Model-as-interpreter: auto-fire a 2-3 sentence "introduce yourself" brief
 *     on first model load + always inject a model-self system prompt so every
 *     reply is grounded in the model's actual config.
 */
import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useModelStore } from '@/stores/modelStore';
import { useLocaleStore, useT } from '@/i18n';
import { EmptyState } from '@/components/common/EmptyState';
import { ConversationSessionMenu } from '@/components/common/ConversationSessionMenu';
import {
  Send, Square, Loader2, Bot, User, Image as ImageIcon, X, Volume2, Pause,
  Mic, MicOff, Upload, Settings2, ChevronRight, Sparkles, Wand2,
  Layers, Cpu, HardDrive, Brain, Zap,
} from 'lucide-react';
import { cn, formatParamCount, formatSize } from '@/lib/utils';
import { createReconnectingWebSocket, buildChatWsUrl, type ReconnectingWebSocketHandle } from '@/api/websocket';
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  replaceConversationMessages,
  updateConversation,
  type ConversationMessage,
} from '@/api/conversations';
import { stripThinking } from '@/lib/textUtils';
import MarkdownContent from '@/components/MarkdownContent';
import { IdentityCard } from '@/components/common/IdentityCard';
import {
  buildModelSelfSystemPrompt, buildAutoBriefPrompt, buildExplainSelfPrompt,
  getSuggestedPrompts, getTTSSuggestedTexts, deriveModelFacts, estimateTokens,
} from '@/lib/chatPrompts';

// ───── Types ────────────────────────────────────────────────────────────────

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking?: string;
  imageThumb?: string;
  tokensPerSec?: number;
  totalTokens?: number;
  totalTime?: number;
  prefillTime?: number;
  streaming?: boolean;
  audioB64?: string;
  audioDuration?: number;
  sampleRate?: number;
  segments?: Array<{ start: number; end: number; text: string }>;
  detectedLanguage?: string;
  isBrief?: boolean;   // auto-generated model self-intro
}

interface StreamEvent {
  type: 'token' | 'status' | 'complete' | 'error' | 'cancelled' | 'audio_chunk';
  token?: string;
  token_id?: number;
  message?: string;
  full_text?: string;
  total_tokens?: number;
  tokens_per_sec?: number;
  total_time?: number;
  prefill_time?: number;
  audio_b64?: string;
  sample_rate?: number;
  duration?: number;
  language?: string;
  segments?: Array<{ start: number; end: number; text: string }>;
}

interface RecommendedParams {
  max_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty?: number;
}

// ───── Audio helper ─────────────────────────────────────────────────────────

function AudioPlayer({ audioB64, duration }: { audioB64: string; duration?: number }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) audio.pause();
    else audio.play();
    setPlaying(!playing);
  }, [playing]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onEnded = () => setPlaying(false);
    audio.addEventListener('ended', onEnded);
    return () => audio.removeEventListener('ended', onEnded);
  }, []);

  const src = `data:audio/wav;base64,${audioB64}`;
  return (
    <div className="flex items-center gap-2">
      <audio ref={audioRef} src={src} />
      <button
        onClick={togglePlay}
        className="flex items-center gap-1.5 rounded-lg bg-indigo-50 px-3 py-1.5 text-sm text-indigo-600 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-400 dark:hover:bg-indigo-500/20"
      >
        {playing ? <Pause size={14} /> : <Volume2 size={14} />}
        {playing ? 'Pause' : 'Play'}
      </button>
      {duration != null && (
        <span className="text-xs text-gray-400 dark:text-stone-500">{duration.toFixed(1)}s</span>
      )}
    </div>
  );
}

// ───── Main page ────────────────────────────────────────────────────────────

const CATEGORY_COLORS: Record<string, string> = {
  llm: 'bg-blue-100 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400',
  vlm: 'bg-purple-100 text-purple-700 dark:bg-purple-500/10 dark:text-purple-400',
  tts: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400',
  stt: 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400',
};

const CHAT_SURFACE = 'chat';

export default function Chat() {
  const model = useModelStore((s) => s.currentModel);
  const queryClient = useQueryClient();
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const isTTS = model?.model_category === 'tts';
  const isSTT = model?.model_category === 'stt';
  const isLLMlike = !!model && (model.model_category === 'llm' || model.model_category === 'vlm');

  // ── State (ALL hooks before any early return — playbook §2.1) ──
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState('');

  const [paramsOpen, setParamsOpen] = useState(false);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [enableThinking, setEnableThinking] = useState(false);
  const [enableDSR, setEnableDSR] = useState(false);
  const [dsrBudget, setDsrBudget] = useState<string>('');
  const [autoTruncate, setAutoTruncate] = useState(true);
  const [systemPromptOverride, setSystemPromptOverride] = useState<string | null>(null);

  const [ttsVoices, setTtsVoices] = useState<string[]>([]);
  const [ttsInstructMode, setTtsInstructMode] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<string>('');
  const [ttsInstruct, setTtsInstruct] = useState<string>('');

  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageThumb, setImageThumb] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioFileInputRef = useRef<HTMLInputElement>(null);

  // Refs
  const wsHandleRef = useRef<ReconnectingWebSocketHandle | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const streamingMsgIdRef = useRef<string | null>(null);
  const briefFiredForRef = useRef<string | null>(null);
  const streamingRef = useRef(false);

  // Derived facts (safe even if model is null — guarded inside)
  const facts = useMemo(() => (model ? deriveModelFacts(model) : null), [model]);
  const conversationsQ = useQuery({
    queryKey: ['conversations', CHAT_SURFACE],
    queryFn: () => listConversations({ surface: CHAT_SURFACE, limit: 50 }),
    refetchInterval: 5000,
  });

  // System prompt — model-as-self (live config). Editable via params panel.
  const defaultSystemPrompt = useMemo(
    () => (model && isLLMlike ? buildModelSelfSystemPrompt(model, locale) : ''),
    [model, locale, isLLMlike],
  );
  const effectiveSystemPrompt = systemPromptOverride ?? defaultSystemPrompt;

  // Estimate context usage from history
  const ctxUsageTokens = useMemo(() => {
    if (!facts) return 0;
    const histText = messages.map((m) => m.content).join('\n');
    return estimateTokens(effectiveSystemPrompt) + estimateTokens(histText);
  }, [messages, effectiveSystemPrompt, facts]);
  const ctxFillPct = facts && facts.maxCtx > 0 ? (ctxUsageTokens / facts.maxCtx) * 100 : 0;

  // Suggested prompts (per category + locale)
  const suggestedPrompts = useMemo(() => {
    if (!model) return [];
    if (isTTS) return getTTSSuggestedTexts(locale);
    if (isSTT) return [];
    return getSuggestedPrompts(model, locale);
  }, [model, locale, isTTS, isSTT]);

  // Latest assistant message (for live perf badge in identity strip)
  const lastAssistant = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === 'assistant' && !m.streaming && (m.tokensPerSec != null || m.totalTime != null)) return m;
    }
    return null;
  }, [messages]);
  const conversationItems = conversationsQ.data?.items ?? [];

  // ── Effects (still all before early return) ──
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  // Reset everything when model switches
  useEffect(() => {
    setMessages([]);
    setCurrentSessionId(null);
    setSystemPromptOverride(null);
    setImageB64(null);
    setImageThumb(null);
    setInput('');
    setStatus('');
    briefFiredForRef.current = null;
    wsHandleRef.current?.close();
    wsHandleRef.current = null;
  }, [model?.model_id]);

  // Pull recommended generation params from backend (model-aware).
  // Intentionally key only on model_id (a new model load triggers a fresh fetch);
  // we don't want every model-object identity change to refetch.
  useEffect(() => {
    if (!model || !isLLMlike) return;
    const mid = model.model_id;
    fetch(`/api/chat/${mid}/params`)
      .then((r) => r.json())
      .then((p: RecommendedParams) => {
        if (typeof p.max_tokens === 'number') setMaxTokens(p.max_tokens);
        if (typeof p.temperature === 'number') setTemperature(p.temperature);
        if (typeof p.top_k === 'number') setTopK(p.top_k);
        if (typeof p.top_p === 'number') setTopP(p.top_p);
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id, isLLMlike]);

  // TTS voice list + instruct_mode flag (Qwen3-TTS-CustomVoice)
  useEffect(() => {
    if (!model) return;
    if (model.model_category !== 'tts') {
      setTtsVoices([]);
      setTtsInstructMode(false);
      return;
    }
    fetch(`/api/chat/${model.model_id}/tts-voices`)
      .then((r) => r.json())
      .then((d) => {
        const voices: string[] = d.voices || [];
        setTtsVoices(voices);
        setTtsInstructMode(!!d.instruct_mode);
        if (voices.length && !selectedVoice) setSelectedVoice(voices[0]);
      })
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model?.model_id]);

  // Cleanup WS on unmount
  useEffect(() => {
    return () => { wsHandleRef.current?.close(); };
  }, []);

  // ── WebSocket helpers ──
  const connectWebSocket = useCallback((modelId: string): ReconnectingWebSocketHandle => {
    if (wsHandleRef.current && wsHandleRef.current.readyState() === WebSocket.OPEN) {
      return wsHandleRef.current;
    }
    wsHandleRef.current?.close();
    const handle = createReconnectingWebSocket(buildChatWsUrl(modelId), {
      onClose: () => {
        if (wsHandleRef.current === handle) wsHandleRef.current = null;
      },
      onReconnect: (attempt) => console.log(`[Chat] WS reconnecting (attempt ${attempt})...`),
    });
    wsHandleRef.current = handle;
    return handle;
  }, []);

  const persistConversation = useCallback((
    sessionId: string,
    nextMessages: ChatMessage[],
    status: 'active' | 'complete' | 'error' | 'cancelled' = 'active',
  ) => {
    void replaceConversationMessages(sessionId, serializeChatMessages(nextMessages))
      .then(() => updateConversation(sessionId, { status }))
      .then(() => queryClient.invalidateQueries({ queryKey: ['conversations', CHAT_SURFACE] }))
      .catch((err) => console.warn('[Chat] failed to persist conversation', err));
  }, [queryClient]);

  const ensureConversationSession = useCallback(async (firstText: string): Promise<string | null> => {
    if (!model) return null;
    if (currentSessionId) return currentSessionId;
    try {
      const session = await createConversation({
        surface: CHAT_SURFACE,
        title: titleFromText(firstText),
        model_id: model.model_id,
        source: 'edgestudio_web',
        status: 'active',
        metadata: {
          model_name: model.model_name,
          model_category: model.model_category,
        },
      });
      setCurrentSessionId(session.session_id);
      void queryClient.invalidateQueries({ queryKey: ['conversations', CHAT_SURFACE] });
      return session.session_id;
    } catch (err) {
      console.warn('[Chat] failed to create conversation session', err);
      return null;
    }
  }, [currentSessionId, model, queryClient]);

  const loadConversationSession = useCallback(async (sessionId: string) => {
    if (!sessionId || streamingRef.current) return;
    try {
      const session = await getConversation(sessionId);
      setCurrentSessionId(session.session_id);
      setMessages((session.messages ?? []).map(chatMessageFromStoredMessage));
      setStatus('');
      briefFiredForRef.current = model?.model_id ?? null;
      wsHandleRef.current?.close();
      wsHandleRef.current = null;
    } catch (err) {
      console.warn('[Chat] failed to load conversation session', err);
    }
  }, [model?.model_id]);

  // Build the history payload sent to the backend.
  // Auto-truncate keeps the most recent N pairs once the estimated context
  // crosses 80% of the model's max_position_embeddings (cheap guard against OOM).
  const buildHistoryPayload = useCallback((extraSystem?: string) => {
    const sys = extraSystem ?? effectiveSystemPrompt;
    let history: { role: string; content: string }[] = messages
      .filter((m) => !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }));
    if (autoTruncate && facts && facts.maxCtx > 0) {
      const cap = Math.floor(facts.maxCtx * 0.6);
      let rough = estimateTokens(sys);
      const kept: { role: string; content: string }[] = [];
      // walk backwards, keep newest first
      for (let i = history.length - 1; i >= 0; i--) {
        const t = estimateTokens(history[i].content);
        if (rough + t > cap && kept.length >= 2) break;
        kept.unshift(history[i]);
        rough += t;
      }
      history = kept;
    }
    if (sys) history = [{ role: 'system', content: sys }, ...history];
    return history;
  }, [messages, effectiveSystemPrompt, autoTruncate, facts]);

  // ── handleSend (LLM / VLM / TTS) ──
  const sendMessage = useCallback((opts: {
    userText: string;
    isBrief?: boolean;
    image?: string | null;
    imageThumbForUI?: string | null;
  }) => {
    if (!model) return;
    if (streamingRef.current) return;
    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: opts.userText,
      imageThumb: opts.imageThumbForUI ?? undefined,
    };
    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      streaming: true,
      isBrief: !!opts.isBrief,
    };
    streamingMsgIdRef.current = assistantId;
    streamingRef.current = true;

    if (opts.isBrief) {
      // brief replaces the empty state; we don't push a visible user msg
      setMessages([assistantMsg]);
    } else {
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
    }
    setStreaming(true);
    setStatus('');

    const history = buildHistoryPayload();

    void (async () => {
      const sessionId = opts.isBrief ? null : await ensureConversationSession(opts.userText);
      const handle = connectWebSocket(model.model_id);
    let audioChunkCount = 0;
    let audioCtx: AudioContext | null = null;
    let nextPlayTime = 0;
    if (isTTS) {
      audioCtx = new AudioContext({ sampleRate: 24000 });
      nextPlayTime = audioCtx.currentTime;
    }

    handle.setOnMessage((event) => {
      const data: StreamEvent = JSON.parse(event.data);
      if (data.type === 'status') {
        setStatus(data.message || '');
      } else if (data.type === 'token') {
        setStatus('');
        setMessages((prev) => prev.map((m) =>
          m.id === assistantId ? { ...m, content: m.content + (data.token || '') } : m,
        ));
      } else if (data.type === 'audio_chunk') {
        audioChunkCount++;
        setStatus(`Generating audio... (${audioChunkCount} chunks)`);
        if (audioCtx && data.audio_b64) {
          const binary = atob(data.audio_b64);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
          audioCtx.decodeAudioData(bytes.buffer.slice(0)).then((buffer) => {
            const source = audioCtx!.createBufferSource();
            source.buffer = buffer;
            source.connect(audioCtx!.destination);
            const now = audioCtx!.currentTime;
            const startAt = Math.max(nextPlayTime, now);
            source.start(startAt);
            nextPlayTime = startAt + buffer.duration;
          }).catch(() => {});
        }
      } else if (data.type === 'complete') {
        const finalText = data.full_text || '';
        setMessages((prev) => {
          const next = prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: finalText || m.content,
                  streaming: false,
                  tokensPerSec: data.tokens_per_sec,
                  totalTokens: data.total_tokens,
                  totalTime: data.total_time,
                  prefillTime: data.prefill_time,
                  audioB64: data.audio_b64,
                  audioDuration: data.duration,
                  sampleRate: data.sample_rate,
                  segments: data.segments,
                  detectedLanguage: data.language,
                }
              : m,
          );
          if (sessionId) persistConversation(sessionId, next, 'complete');
          return next;
        });
        setStreaming(false);
        streamingRef.current = false;
        setStatus('');
        streamingMsgIdRef.current = null;
      } else if (data.type === 'error') {
        setMessages((prev) => {
          const next = prev.map((m) =>
            m.id === assistantId ? { ...m, content: data.message || 'Error', streaming: false } : m,
          );
          if (sessionId) persistConversation(sessionId, next, 'error');
          return next;
        });
        setStreaming(false);
        streamingRef.current = false;
        setStatus('');
        streamingMsgIdRef.current = null;
      } else if (data.type === 'cancelled') {
        setMessages((prev) => {
          const next = prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m,
          );
          if (sessionId) persistConversation(sessionId, next, 'cancelled');
          return next;
        });
        setStreaming(false);
        streamingRef.current = false;
        setStatus('');
        streamingMsgIdRef.current = null;
      }
    });

    const payload: Record<string, unknown> = {
      prompt: opts.userText,
      history,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      top_p: topP,
    };
    if (model.supports_thinking) payload.enable_thinking = enableThinking;
    if (opts.image) payload.image_b64 = opts.image;
    if (isTTS) {
      if (ttsInstructMode && ttsInstruct) payload.instruct = ttsInstruct;
      else if (selectedVoice) payload.voice = selectedVoice;
    }
    if (enableDSR) {
      payload.enable_dsr = true;
      if (dsrBudget) payload.dsr_budget = parseInt(dsrBudget, 10);
    }
    handle.send(JSON.stringify(payload));
    })();
  }, [
    model, connectWebSocket, buildHistoryPayload, maxTokens, temperature, topK, topP,
    enableThinking, isTTS, ttsInstructMode, ttsInstruct, selectedVoice, enableDSR, dsrBudget,
    ensureConversationSession, persistConversation,
  ]);

  // Auto-fire AI brief once per loaded LLM/VLM model (model-as-interpreter, playbook §1-D)
  useEffect(() => {
    if (!model || !isLLMlike) return;
    if (briefFiredForRef.current === model.model_id) return;
    if (messages.length > 0) return;
    if (streamingRef.current) return;
    briefFiredForRef.current = model.model_id;
    // Slight delay so first paint happens before WS spin-up
    const id = window.setTimeout(() => {
      sendMessage({ userText: buildAutoBriefPrompt(model, locale), isBrief: true });
    }, 300);
    return () => window.clearTimeout(id);
  }, [model, isLLMlike, locale, messages.length, sendMessage]);

  const handleSendUserInput = useCallback(() => {
    if (!model || !input.trim() || streaming) return;
    sendMessage({
      userText: input.trim(),
      image: imageB64,
      imageThumbForUI: imageThumb,
    });
    setInput('');
    setImageB64(null);
    setImageThumb(null);
  }, [model, input, streaming, imageB64, imageThumb, sendMessage]);

  const handleSuggested = useCallback((p: string) => {
    setInput(p);
    inputRef.current?.focus();
  }, []);

  const handleExplainSelf = useCallback(() => {
    if (!model || !isLLMlike || streaming) return;
    sendMessage({ userText: buildExplainSelfPrompt(model, locale) });
  }, [model, isLLMlike, streaming, locale, sendMessage]);

  const handleCancel = useCallback(() => {
    wsHandleRef.current?.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  const handleClear = useCallback(() => {
    const sessionId = currentSessionId;
    setMessages([]);
    setCurrentSessionId(null);
    briefFiredForRef.current = null;
    wsHandleRef.current?.close();
    wsHandleRef.current = null;
    if (sessionId) {
      void deleteConversation(sessionId)
        .then(() => queryClient.invalidateQueries({ queryKey: ['conversations', CHAT_SURFACE] }))
        .catch((err) => console.warn('[Chat] failed to delete conversation', err));
    }
  }, [currentSessionId, queryClient]);

  const handleDeleteConversationSession = useCallback((sessionId: string) => {
    if (sessionId === currentSessionId) {
      setMessages([]);
      setCurrentSessionId(null);
      briefFiredForRef.current = null;
      wsHandleRef.current?.close();
      wsHandleRef.current = null;
    }
    void deleteConversation(sessionId)
      .then(() => queryClient.invalidateQueries({ queryKey: ['conversations', CHAT_SURFACE] }))
      .catch((err) => console.warn('[Chat] failed to delete conversation', err));
  }, [currentSessionId, queryClient]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendUserInput();
    }
  };

  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      const b64 = dataUrl.split(',')[1];
      setImageB64(b64);
      setImageThumb(dataUrl);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }, []);

  // ── STT-only path (separate WebSocket payload schema) ──
  const sendSTTAudio = useCallback((audioB64Local: string | null, fileName: string, fileId?: string) => {
    if (!model || streamingRef.current) return;
    const userMsg: ChatMessage = { id: `user-${Date.now()}`, role: 'user', content: `[Audio: ${fileName}]` };
    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', streaming: true };
    streamingMsgIdRef.current = assistantId;
    streamingRef.current = true;
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);
    setStatus('');

    const handle = connectWebSocket(model.model_id);
    handle.setOnMessage((event) => {
      const data: StreamEvent = JSON.parse(event.data);
      if (data.type === 'status') setStatus(data.message || '');
      else if (data.type === 'token') {
        setStatus('');
        setMessages((prev) => prev.map((m) =>
          m.id === assistantId ? { ...m, content: m.content + (data.token || '') } : m,
        ));
      } else if (data.type === 'complete') {
        setMessages((prev) => prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content: data.full_text || m.content || '(No speech detected)',
                streaming: false,
                tokensPerSec: data.tokens_per_sec,
                totalTokens: data.total_tokens,
                totalTime: data.total_time,
                prefillTime: data.prefill_time,
                segments: data.segments,
                detectedLanguage: data.language,
              }
            : m,
        ));
        setStreaming(false); streamingRef.current = false; setStatus(''); streamingMsgIdRef.current = null;
      } else if (data.type === 'error') {
        setMessages((prev) => prev.map((m) =>
          m.id === assistantId ? { ...m, content: data.message || 'Error', streaming: false } : m,
        ));
        setStreaming(false); streamingRef.current = false; setStatus(''); streamingMsgIdRef.current = null;
      } else if (data.type === 'cancelled') {
        setMessages((prev) => prev.map((m) => m.id === assistantId ? { ...m, streaming: false } : m));
        setStreaming(false); streamingRef.current = false; setStatus(''); streamingMsgIdRef.current = null;
      }
    });
    handle.send(JSON.stringify({
      audio_b64: audioB64Local || undefined,
      file_id: fileId || undefined,
      file_name: fileName,
      language: input.trim() || undefined,
    }));
  }, [model, input, connectWebSocket]);

  const handleAudioFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !model || streaming) return;
    e.target.value = '';
    const formData = new FormData();
    formData.append('file', file);
    try {
      setStatus('Uploading audio...');
      const resp = await fetch('/api/chat/upload-audio', { method: 'POST', body: formData });
      const { file_id, file_name } = await resp.json();
      setStatus('');
      sendSTTAudio(null, file_name || file.name, file_id);
    } catch (err) {
      setStatus('');
      console.error('Audio upload failed:', err);
    }
  }, [model, streaming, sendSTTAudio]);

  const toggleRecording = useCallback(async () => {
    if (recording) {
      mediaRecorderRef.current?.stop();
      setRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      audioChunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onload = (ev) => {
          const dataUrl = ev.target?.result as string;
          const b64 = dataUrl.split(',')[1];
          sendSTTAudio(b64, 'recording.webm');
        };
        reader.readAsDataURL(blob);
      };
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error('Microphone access denied:', err);
    }
  }, [recording, sendSTTAudio]);

  // ── Early return AFTER all hooks (playbook §2.1) ──
  if (!model) {
    return <EmptyState title={t('common.noModel')} description={t('common.noModelDesc')} />;
  }

  // ── Render-time helpers ──
  const ctxFillTone = ctxFillPct >= 95 ? 'red' : ctxFillPct >= 80 ? 'amber' : 'emerald';
  const ctxFillColor = {
    red: 'bg-red-500',
    amber: 'bg-amber-500',
    emerald: 'bg-emerald-500',
  }[ctxFillTone];

  return (
    <div className="-m-6 flex h-[calc(100vh-var(--header-height,49px))]">
      {/* Main chat area */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Identity header strip */}
        <div className="border-b border-gray-100 dark:border-stone-800 bg-white dark:bg-stone-950">
          <div className="flex items-center justify-between gap-3 px-6 py-2">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-sm font-semibold text-gray-800 dark:text-stone-200 truncate" title={model.model_dir}>
                {model.model_name}
              </span>
              <span className={cn('rounded-full px-2 py-0.5 text-[10px] font-medium uppercase shrink-0', CATEGORY_COLORS[model.model_category] || CATEGORY_COLORS.llm)}>
                {model.model_category}
              </span>
              {model.supports_thinking && (
                <span className="rounded-full px-2 py-0.5 text-[10px] font-medium bg-violet-100 text-violet-700 dark:bg-violet-500/10 dark:text-violet-400 shrink-0">{t('chat.tagThinking')}</span>
              )}
              {model.has_vision && (
                <span className="rounded-full px-2 py-0.5 text-[10px] font-medium bg-purple-100 text-purple-700 dark:bg-purple-500/10 dark:text-purple-400 shrink-0">{t('chat.tagVision')}</span>
              )}
              {model.has_moe && (
                <span className="rounded-full px-2 py-0.5 text-[10px] font-medium bg-orange-100 text-orange-700 dark:bg-orange-500/10 dark:text-orange-400 shrink-0">MoE</span>
              )}
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <ConversationSessionMenu
                sessions={conversationItems}
                currentSessionId={currentSessionId}
                disabled={streaming}
                labels={{
                  title: t('chat.savedSessions'),
                  newSession: t('chat.newSession'),
                  untitled: t('training.common.untitled'),
                  deleteSession: t('chat.deleteSession'),
                  deleteSessionConfirm: t('chat.deleteSessionConfirm'),
                }}
                onNewSession={() => {
                  setCurrentSessionId(null);
                  setMessages([]);
                  briefFiredForRef.current = null;
                }}
                onSelectSession={(sessionId) => { void loadConversationSession(sessionId); }}
                onDeleteSession={handleDeleteConversationSession}
                formatTime={formatSessionTime}
              />
              {messages.length > 0 && (
                <button onClick={handleClear} className="rounded-lg px-2.5 py-1 text-xs text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300">
                  {t('chat.clear')}
                </button>
              )}
              {isTTS && ttsInstructMode && (
                <input
                  type="text"
                  value={ttsInstruct}
                  onChange={(e) => setTtsInstruct(e.target.value)}
                  placeholder={t('chat.ttsInstructPlaceholder')}
                  className="rounded-lg border border-gray-200 px-2 py-1 text-xs w-56 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300"
                  title={t('chat.ttsInstructHint')}
                />
              )}
              {isTTS && !ttsInstructMode && ttsVoices.length > 0 && (
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  className="rounded-lg border border-gray-200 px-2 py-1 text-xs dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300"
                >
                  {ttsVoices.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              )}
              {!isTTS && !isSTT && (
                <button
                  onClick={() => setParamsOpen(!paramsOpen)}
                  className={cn(
                    'flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs transition-colors',
                    paramsOpen
                      ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-500/10 dark:text-indigo-400'
                      : 'text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300',
                  )}
                  title={t('chat.params')}
                >
                  <Settings2 size={13} />
                  {t('chat.params')}
                </button>
              )}
            </div>
          </div>
          {/* 4 micro-card identity strip */}
          {facts && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 px-6 pb-2">
              <IdentityCard
                icon={<HardDrive size={14} />}
                label={t('chat.idScale')}
                value={`${formatParamCount(facts.totalParams)} · ${facts.quantBits || '?'}-bit`}
                hint={`${formatSize(facts.totalSizeBytes)} on disk`}
              />
              <IdentityCard
                icon={<Layers size={14} />}
                label={t('chat.idArch')}
                value={`${facts.numLayers}L · h${facts.hiddenSize}${facts.gqaRatio > 1 ? ` · GQA ${facts.gqaRatio}:1` : ''}`}
                hint={`${facts.numHeads} Q heads / ${facts.numKVHeads} KV heads`}
              />
              <IdentityCard
                icon={<Cpu size={14} />}
                label={t('chat.idContext')}
                value={facts.maxCtx ? `${(facts.maxCtx / 1024).toFixed(0)}K max` : '—'}
                hint={`KV ${(facts.kvPerTokenBytes / 1024).toFixed(1)} KB/tok · @4K = ${formatSize(facts.kvAt4kBytes)}`}
              />
              <IdentityCard
                icon={<Zap size={14} />}
                label={t('chat.idPerf')}
                value={lastAssistant && lastAssistant.tokensPerSec != null
                  ? `${lastAssistant.tokensPerSec.toFixed(1)} tok/s${lastAssistant.prefillTime && lastAssistant.prefillTime > 0 ? ` · TTFT ${lastAssistant.prefillTime.toFixed(2)}s` : ''}`
                  : '—'}
                hint="Last assistant message"
                tone={lastAssistant ? 'emerald' : 'neutral'}
              />
            </div>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center gap-4 max-w-2xl mx-auto">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-100 to-purple-100 text-indigo-600 dark:from-indigo-500/20 dark:to-purple-500/20 dark:text-indigo-400">
                <Bot size={28} />
              </div>
              <p className="text-sm text-gray-500 dark:text-stone-400 text-center">
                {isSTT ? t('chat.sttEmpty')
                  : isTTS ? t('chat.ttsEmpty')
                  : t('chat.empty')}
              </p>
              {suggestedPrompts.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full mt-2">
                  {suggestedPrompts.map((sp) => (
                    <button
                      key={sp.label}
                      onClick={() => handleSuggested(sp.prompt)}
                      className="text-left rounded-xl border border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/40 dark:border-stone-700 dark:hover:border-indigo-500/40 dark:hover:bg-indigo-500/5 px-3 py-2.5 transition-colors"
                    >
                      <div className="text-xs font-medium text-gray-700 dark:text-stone-300">{sp.label}</div>
                      <div className="text-[11px] text-gray-400 dark:text-stone-500 mt-0.5 line-clamp-2">{sp.prompt}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={cn('mb-4 flex gap-3', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
              {msg.role === 'assistant' && (
                <div className={cn(
                  'flex h-7 w-7 shrink-0 items-center justify-center rounded-full',
                  msg.isBrief
                    ? 'bg-gradient-to-br from-indigo-100 to-purple-100 text-indigo-600 dark:from-indigo-500/20 dark:to-purple-500/20 dark:text-indigo-400'
                    : 'bg-indigo-100 text-indigo-600 dark:bg-indigo-500/10 dark:text-indigo-400',
                )}>
                  {msg.isBrief ? <Sparkles size={16} /> : <Bot size={16} />}
                </div>
              )}
              <div className={cn(
                'max-w-[75%] rounded-2xl px-4 py-2.5',
                msg.role === 'user'
                  ? 'bg-indigo-500 text-white'
                  : msg.isBrief
                    ? 'bg-gradient-to-br from-indigo-50 to-purple-50 text-gray-800 border border-indigo-100 dark:from-indigo-500/5 dark:to-purple-500/5 dark:text-stone-200 dark:border-indigo-500/20'
                    : 'bg-gray-100 text-gray-800 dark:bg-stone-800 dark:text-stone-200',
              )}>
                {msg.isBrief && (
                  <div className="flex items-center gap-1.5 mb-1.5 text-[10px] uppercase tracking-wider font-semibold text-indigo-600 dark:text-indigo-400">
                    <Sparkles size={10} />
                    {t('chat.aiBriefLabel')}
                  </div>
                )}
                {msg.imageThumb && (
                  <div className="mb-2">
                    <img src={msg.imageThumb} alt="uploaded" className="max-h-32 rounded-lg object-contain cursor-pointer" onClick={() => window.open(msg.imageThumb, '_blank')} />
                  </div>
                )}
                {msg.audioB64 && (
                  <div className="mb-2"><AudioPlayer audioB64={msg.audioB64} duration={msg.audioDuration} /></div>
                )}
                <div className="text-sm leading-relaxed">
                  {msg.audioB64 ? '' : msg.role === 'assistant' ? (
                    <MarkdownContent content={enableThinking ? msg.content : stripThinking(msg.content)} />
                  ) : (
                    <span className="whitespace-pre-wrap">{enableThinking ? msg.content : stripThinking(msg.content)}</span>
                  )}
                  {!msg.audioB64 && msg.streaming && (
                    <span className="inline-block w-1.5 h-4 ml-0.5 bg-indigo-500 animate-pulse rounded-sm" />
                  )}
                </div>
                {/* STT segments */}
                {msg.segments && msg.segments.length > 0 && (
                  <details className="mt-2 text-[11px]">
                    <summary className="cursor-pointer text-gray-500 dark:text-stone-400 hover:text-gray-700 dark:hover:text-stone-200">
                      {t('chat.segmentsCount', { n: msg.segments.length })}{msg.detectedLanguage ? ` · ${msg.detectedLanguage}` : ''}
                    </summary>
                    <div className="mt-1.5 space-y-0.5 max-h-48 overflow-y-auto pr-1">
                      {msg.segments.map((s, i) => (
                        <div key={i} className="flex gap-2">
                          <span className="font-mono text-[10px] text-gray-400 dark:text-stone-500 shrink-0 tabular-nums">
                            {s.start.toFixed(1)}-{s.end.toFixed(1)}s
                          </span>
                          <span className="text-gray-600 dark:text-stone-300">{s.text}</span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                {/* Per-message stats */}
                {!msg.streaming && msg.role === 'assistant' && (msg.tokensPerSec != null || msg.totalTime != null || (msg.prefillTime != null && msg.prefillTime > 0)) && (
                  <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 border-t border-gray-200/50 pt-1.5 text-[10px] text-gray-400 dark:border-stone-700 dark:text-stone-500">
                    {msg.totalTokens != null && msg.totalTokens > 0 && <span>{msg.totalTokens} {t('chat.statsTokens')}</span>}
                    {msg.tokensPerSec != null && msg.tokensPerSec > 0 && <span>{msg.tokensPerSec.toFixed(1)} tok/s</span>}
                    {msg.prefillTime != null && msg.prefillTime > 0 && <span>TTFT {msg.prefillTime.toFixed(2)}s</span>}
                    {msg.totalTime != null && msg.totalTime > 0 && <span>{msg.totalTime.toFixed(1)}s {t('chat.statsTotal')}</span>}
                  </div>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gray-200 text-gray-600 dark:bg-stone-700 dark:text-stone-300">
                  <User size={16} />
                </div>
              )}
            </div>
          ))}

          {status && (
            <div className="mb-4 flex items-center gap-2 text-sm text-gray-400 dark:text-stone-500">
              <Loader2 size={14} className="animate-spin" />
              {status}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="border-t border-gray-100 bg-white px-6 py-3 dark:border-stone-800 dark:bg-stone-950">
          {/* Token usage bar (LLM/VLM only, when model has known max ctx) */}
          {isLLMlike && facts && facts.maxCtx > 0 && messages.length > 0 && (
            <div className="mb-2 flex items-center gap-2 text-[10px]">
              <span className="text-gray-400 dark:text-stone-500 shrink-0 tabular-nums">
                ~{ctxUsageTokens.toLocaleString()} / {facts.maxCtx.toLocaleString()} {t('chat.ctxTokens')}
              </span>
              <div className="flex-1 h-1 bg-gray-100 dark:bg-stone-800 rounded-full overflow-hidden">
                <div className={cn('h-full transition-all', ctxFillColor)} style={{ width: `${Math.min(ctxFillPct, 100)}%` }} />
              </div>
              <span className={cn('shrink-0 tabular-nums font-medium',
                ctxFillTone === 'red' ? 'text-red-500' : ctxFillTone === 'amber' ? 'text-amber-500' : 'text-gray-400 dark:text-stone-500',
              )}>
                {ctxFillPct.toFixed(1)}%
              </span>
            </div>
          )}

          {/* Image preview */}
          {imageThumb && (
            <div className="mb-2 flex items-center gap-2">
              <div className="relative">
                <img src={imageThumb} alt="to send" className="h-16 rounded-lg object-contain border border-gray-200 dark:border-stone-700" />
                <button onClick={() => { setImageB64(null); setImageThumb(null); }} className="absolute -right-1.5 -top-1.5 rounded-full bg-gray-500 p-0.5 text-white hover:bg-gray-700">
                  <X size={10} />
                </button>
              </div>
            </div>
          )}

          <div className="flex items-end gap-2">
            <div className="flex shrink-0 gap-1 pb-1">
              {model.has_vision && !isSTT && (
                <>
                  <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleImageSelect} className="hidden" />
                  <button onClick={() => fileInputRef.current?.click()} disabled={streaming} className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:opacity-50 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300" title={t('chat.uploadImage')}>
                    <ImageIcon size={18} />
                  </button>
                </>
              )}
              {isLLMlike && (
                <button
                  onClick={handleExplainSelf}
                  disabled={streaming}
                  className="rounded-lg p-2 text-gray-400 hover:bg-indigo-50 hover:text-indigo-600 disabled:opacity-50 dark:text-stone-500 dark:hover:bg-indigo-500/10 dark:hover:text-indigo-400"
                  title={t('chat.explainSelf')}
                >
                  <Wand2 size={18} />
                </button>
              )}
              {isSTT && (
                <>
                  <input ref={audioFileInputRef} type="file" accept="audio/*" onChange={handleAudioFileSelect} className="hidden" />
                  <button onClick={() => audioFileInputRef.current?.click()} disabled={streaming} className="rounded-lg p-2 text-gray-400 hover:bg-gray-100 disabled:opacity-50 dark:text-stone-500 dark:hover:bg-stone-800" title={t('chat.uploadAudio')}>
                    <Upload size={18} />
                  </button>
                  <button onClick={toggleRecording} disabled={streaming && !recording} className={cn(
                    'rounded-lg p-2 disabled:opacity-50',
                    recording ? 'text-red-500 bg-red-50 dark:bg-red-500/10' : 'text-gray-400 hover:bg-gray-100 dark:text-stone-500 dark:hover:bg-stone-800',
                  )} title={recording ? t('chat.stopRec') : t('chat.startRec')}>
                    {recording ? <MicOff size={18} /> : <Mic size={18} />}
                  </button>
                </>
              )}
            </div>

            <div className="relative flex-1">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={isSTT ? t('chat.sttPlaceholder') : isTTS ? t('chat.ttsPlaceholder') : t('chat.placeholder')}
                rows={1}
                className="w-full resize-none rounded-2xl border border-gray-200 bg-gray-50 py-3 pl-4 pr-12 text-sm focus:border-indigo-400 focus:bg-white focus:outline-none dark:border-stone-700 dark:bg-stone-800 dark:text-stone-100 dark:focus:border-indigo-500 dark:focus:bg-stone-900"
                disabled={streaming}
              />
              {!isSTT && (
                <button
                  onClick={streaming ? handleCancel : handleSendUserInput}
                  disabled={!streaming && !input.trim()}
                  className={cn(
                    'absolute right-2 top-1/2 -translate-y-1/2 rounded-xl p-2 transition-colors',
                    streaming
                      ? 'text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10'
                      : 'bg-indigo-500 text-white hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 dark:disabled:bg-stone-700 dark:disabled:text-stone-500',
                  )}
                  title={streaming ? t('common.cancel') : t('chat.send')}
                >
                  {streaming ? <Square size={16} /> : <Send size={16} />}
                </button>
              )}
              {isSTT && streaming && (
                <button onClick={handleCancel} className="absolute right-2 top-1/2 -translate-y-1/2 rounded-xl p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10" title={t('common.cancel')}>
                  <Square size={16} />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Parameters side panel */}
      {paramsOpen && !isTTS && !isSTT && (
        <div className="w-72 shrink-0 border-l border-gray-100 bg-gray-50/50 p-4 overflow-y-auto dark:border-stone-800 dark:bg-stone-900/50">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-400 dark:text-stone-500">{t('chat.params')}</h3>
            <button onClick={() => setParamsOpen(false)} className="rounded p-0.5 text-gray-400 hover:text-gray-600 dark:text-stone-500 dark:hover:text-stone-300">
              <ChevronRight size={14} />
            </button>
          </div>

          <div className="space-y-4">
            {/* System prompt */}
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label className="flex items-center gap-1 text-xs text-gray-500 dark:text-stone-400">
                  <Brain size={11} />
                  {t('chat.systemPrompt')}
                </label>
                {systemPromptOverride !== null && (
                  <button
                    onClick={() => setSystemPromptOverride(null)}
                    className="text-[10px] text-indigo-500 hover:text-indigo-700 dark:text-indigo-400"
                  >
                    {t('chat.resetSystemPrompt')}
                  </button>
                )}
              </div>
              <textarea
                value={effectiveSystemPrompt}
                onChange={(e) => setSystemPromptOverride(e.target.value)}
                rows={6}
                className="w-full text-[11px] font-mono resize-y rounded-lg border border-gray-200 bg-white px-2 py-1.5 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200"
              />
              <p className="mt-1 text-[10px] text-gray-400 dark:text-stone-500">{t('chat.systemPromptHint')}</p>
            </div>

            <div>
              <label className="mb-1.5 block text-xs text-gray-500 dark:text-stone-400">{t('inference.maxTokens')}</label>
              <input type="number" min={1} max={32768} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200" />
            </div>

            <div>
              <label className="mb-1.5 flex items-baseline justify-between text-xs text-gray-500 dark:text-stone-400">
                <span>{t('inference.temperature')}</span>
                <span className="font-mono">{temperature.toFixed(2)}</span>
              </label>
              <input type="range" min={0} max={2} step={0.05} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="w-full accent-indigo-500" />
            </div>

            <div>
              <label className="mb-1.5 block text-xs text-gray-500 dark:text-stone-400">Top-K</label>
              <input type="number" min={0} max={200} value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200" />
            </div>

            <div>
              <label className="mb-1.5 flex items-baseline justify-between text-xs text-gray-500 dark:text-stone-400">
                <span>Top-P</span>
                <span className="font-mono">{topP.toFixed(2)}</span>
              </label>
              <input type="range" min={0} max={1} step={0.05} value={topP} onChange={(e) => setTopP(Number(e.target.value))} className="w-full accent-indigo-500" />
            </div>

            {model.supports_thinking && (
              <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-stone-400">
                <input type="checkbox" checked={enableThinking} onChange={(e) => setEnableThinking(e.target.checked)} className="rounded accent-indigo-500" />
                {t('inference.enableThinking')}
              </label>
            )}

            <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-stone-400">
              <input type="checkbox" checked={autoTruncate} onChange={(e) => setAutoTruncate(e.target.checked)} className="rounded accent-emerald-500" />
              {t('chat.autoTruncate')}
            </label>

            <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-stone-400">
              <input type="checkbox" checked={enableDSR} onChange={(e) => setEnableDSR(e.target.checked)} className="rounded accent-teal-500" />
              DSR Intelligent Cache
            </label>
            {enableDSR && (
              <div>
                <label className="mb-1 block text-xs text-gray-500 dark:text-stone-400">DSR Budget (tokens)</label>
                <input type="number" min={256} max={32768} value={dsrBudget} onChange={(e) => setDsrBudget(e.target.value)} placeholder="Auto" className="w-full rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200" />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function serializeChatMessages(messages: ChatMessage[]) {
  return messages
    .filter((message) => !message.streaming)
    .map((message, index) => ({
      id: message.id,
      sequence: index,
      role: message.role,
      content: message.content,
      metadata: {
        thinking: message.thinking,
        imageThumb: message.imageThumb,
        tokensPerSec: message.tokensPerSec,
        totalTokens: message.totalTokens,
        totalTime: message.totalTime,
        prefillTime: message.prefillTime,
        audioB64: message.audioB64,
        audioDuration: message.audioDuration,
        sampleRate: message.sampleRate,
        segments: message.segments,
        detectedLanguage: message.detectedLanguage,
        isBrief: message.isBrief,
      },
    }));
}

function chatMessageFromStoredMessage(message: ConversationMessage): ChatMessage {
  const metadata = message.metadata ?? {};
  return {
    id: message.message_id,
    role: message.role === 'assistant' ? 'assistant' : 'user',
    content: message.content,
    thinking: typeof metadata.thinking === 'string' ? metadata.thinking : undefined,
    imageThumb: typeof metadata.imageThumb === 'string' ? metadata.imageThumb : undefined,
    tokensPerSec: typeof metadata.tokensPerSec === 'number' ? metadata.tokensPerSec : undefined,
    totalTokens: typeof metadata.totalTokens === 'number' ? metadata.totalTokens : undefined,
    totalTime: typeof metadata.totalTime === 'number' ? metadata.totalTime : undefined,
    prefillTime: typeof metadata.prefillTime === 'number' ? metadata.prefillTime : undefined,
    audioB64: typeof metadata.audioB64 === 'string' ? metadata.audioB64 : undefined,
    audioDuration: typeof metadata.audioDuration === 'number' ? metadata.audioDuration : undefined,
    sampleRate: typeof metadata.sampleRate === 'number' ? metadata.sampleRate : undefined,
    segments: Array.isArray(metadata.segments) ? metadata.segments as Array<{ start: number; end: number; text: string }> : undefined,
    detectedLanguage: typeof metadata.detectedLanguage === 'string' ? metadata.detectedLanguage : undefined,
    isBrief: metadata.isBrief === true,
  };
}

function titleFromText(text: string): string {
  const title = text.replace(/\s+/g, ' ').trim();
  if (!title) return 'Untitled';
  return title.length > 48 ? `${title.slice(0, 48)}...` : title;
}

function formatSessionTime(seconds?: number | null): string {
  if (!seconds) return '-';
  return new Date(seconds * 1000).toLocaleTimeString();
}
