// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 4: Test Chat — multi-modal chat to validate model works.
 *
 * Supports LLM, VLM (image upload), TTS (voice playback), STT (audio upload / recording).
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { stripThinking } from '@/lib/textUtils';
import MarkdownContent from '@/components/MarkdownContent';
import {
  Send, Loader2, MessageCircle, Trash2, Image, X,
  Volume2, Pause, Mic, MicOff, Upload, Square,
} from 'lucide-react';
import { useWizardStore } from '@/stores/wizardStore';
import { useModelStore } from '@/stores/modelStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS } from './wizardSteps';
import { cn } from '@/lib/utils';
import { createReconnectingWebSocket, buildChatWsUrl, type ReconnectingWebSocketHandle } from '@/api/websocket';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  imageThumb?: string;
  audioB64?: string;
  audioDuration?: number;
  tokensPerSec?: number;
  totalTokens?: number;
  totalTime?: number;
  streaming?: boolean;
}

interface StreamEvent {
  type: 'token' | 'status' | 'complete' | 'error' | 'cancelled' | 'audio_chunk';
  token?: string;
  message?: string;
  full_text?: string;
  total_tokens?: number;
  tokens_per_sec?: number;
  total_time?: number;
  audio_b64?: string;
  audio_chunk?: string;
  sample_rate?: number;
  duration?: number;
}

// ---------------------------------------------------------------------------
// AudioPlayer (reused for TTS playback)
// ---------------------------------------------------------------------------

function AudioPlayer({ audioB64, duration }: { audioB64: string; duration?: number }) {
  const t = useT();
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) audio.pause(); else audio.play();
    setPlaying(!playing);
  }, [playing]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onEnded = () => setPlaying(false);
    audio.addEventListener('ended', onEnded);
    return () => audio.removeEventListener('ended', onEnded);
  }, []);

  return (
    <div className="flex items-center gap-2">
      <audio ref={audioRef} src={`data:audio/wav;base64,${audioB64}`} />
      <button
        onClick={togglePlay}
        className="flex items-center gap-1.5 rounded-lg bg-stone-200 px-3 py-1.5 text-sm text-stone-700 hover:bg-stone-300 dark:bg-stone-700 dark:text-stone-300 dark:hover:bg-stone-600"
      >
        {playing ? <Pause size={14} /> : <Volume2 size={14} />}
        {playing ? t('simple.v1.pause') : t('simple.v1.play')}
      </button>
      {duration != null && (
        <span className="text-xs text-stone-400 dark:text-stone-500">{duration.toFixed(1)}s</span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// TestChat
// ---------------------------------------------------------------------------

export default function TestChat() {
  const t = useT();
  const navigate = useNavigate();
  const { loadedModelId, setChatTested, setCurrentStep } = useWizardStore();
  const model = useModelStore((s) => s.currentModel);

  // Detect model type
  const isTTS = model?.model_category === 'tts';
  const isSTT = model?.model_category === 'stt';
  const hasVision = model?.has_vision === true;

  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsHandleRef = useRef<ReconnectingWebSocketHandle | null>(null);
  const streamingMsgIdRef = useRef<string | null>(null);

  // TTS state
  const [ttsVoices, setTtsVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [loadingVoices, setLoadingVoices] = useState(false);

  // VLM image state
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageThumb, setImageThumb] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // STT audio state
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioFileInputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, status]);

  // Cleanup WS on unmount
  useEffect(() => {
    return () => { wsHandleRef.current?.close(); };
  }, []);

  // Fetch TTS voices (retry until model is loaded and voices are available)
  useEffect(() => {
    if (!model || model.model_category !== 'tts') return;
    setLoadingVoices(true);
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const fetchVoices = () => {
      fetch(`/api/chat/${model.model_id}/tts-voices`)
        .then(r => r.json())
        .then(d => {
          if (cancelled) return;
          const voices: string[] = d.voices || [];
          if (voices.length > 0) {
            setTtsVoices(voices);
            if (!selectedVoice) setSelectedVoice(voices[0]);
            setLoadingVoices(false);
          } else {
            // Model still loading — retry
            timer = setTimeout(fetchVoices, 2000);
          }
        })
        .catch(() => {
          if (!cancelled) timer = setTimeout(fetchVoices, 2000);
        });
    };

    fetchVoices();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [model?.model_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // WebSocket helpers
  // -------------------------------------------------------------------------

  const connectWebSocket = useCallback((modelId: string): ReconnectingWebSocketHandle => {
    if (wsHandleRef.current && wsHandleRef.current.readyState() === WebSocket.OPEN) {
      return wsHandleRef.current;
    }
    wsHandleRef.current?.close();

    const handle = createReconnectingWebSocket(buildChatWsUrl(modelId), {
      onClose: () => {
        if (wsHandleRef.current === handle) wsHandleRef.current = null;
      },
    });
    wsHandleRef.current = handle;
    return handle;
  }, []);

  /** Shared stream event handler — updates messages for a given assistantId. */
  const bindStreamHandler = useCallback((handle: ReconnectingWebSocketHandle, assistantId: string, audioCtx: AudioContext | null) => {
    let nextPlayTime = audioCtx ? audioCtx.currentTime : 0;
    let audioChunkCount = 0;
    let rawText = '';  // accumulate raw tokens for stripThinking

    handle.setOnMessage((event) => {
      try {
        const data: StreamEvent = JSON.parse(event.data);

        if (data.type === 'status') {
          setStatus(data.message || '');
        } else if (data.type === 'token') {
          setStatus('');
          rawText += data.token || '';
          const display = stripThinking(rawText);
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: display } : m,
          ));
        } else if (data.type === 'audio_chunk') {
          audioChunkCount++;
          setStatus(t('simple.v1.generatingAudio', { count: String(audioChunkCount) }));
          if (audioCtx && data.audio_chunk) {
            const binary = atob(data.audio_chunk);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            audioCtx.decodeAudioData(bytes.buffer.slice(0)).then(buffer => {
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
          const finalText = stripThinking(data.full_text || rawText);
          setMessages(prev => prev.map(m =>
            m.id === assistantId
              ? {
                  ...m,
                  content: finalText || m.content,
                  streaming: false,
                  tokensPerSec: data.tokens_per_sec,
                  totalTokens: data.total_tokens,
                  totalTime: data.total_time,
                  audioB64: data.audio_b64,
                  audioDuration: data.duration,
                }
              : m,
          ));
          setStreaming(false);
          setStatus('');
          streamingMsgIdRef.current = null;
          setChatTested(true);
        } else if (data.type === 'error') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: data.message || t('simple.error.unknown'), streaming: false } : m,
          ));
          setStreaming(false);
          setStatus('');
          streamingMsgIdRef.current = null;
        } else if (data.type === 'cancelled') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, streaming: false } : m,
          ));
          setStreaming(false);
          setStatus('');
          streamingMsgIdRef.current = null;
        }
      } catch {
        // ignore parse errors
      }
    });
  }, [setChatTested]);

  // -------------------------------------------------------------------------
  // LLM / VLM / TTS send
  // -------------------------------------------------------------------------

  const handleSend = useCallback(() => {
    const modelId = loadedModelId || model?.model_id;
    if (!modelId || !input.trim() || streaming) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input.trim(),
      imageThumb: imageThumb ?? undefined,
    };
    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', streaming: true };
    const currentImageB64 = imageB64;

    streamingMsgIdRef.current = assistantId;
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput('');
    setImageB64(null);
    setImageThumb(null);
    setStreaming(true);
    setStatus('');

    const handle = connectWebSocket(modelId);

    // TTS: real-time audio playback
    let audioCtx: AudioContext | null = null;
    if (isTTS) {
      audioCtx = new AudioContext({ sampleRate: 24000 });
    }

    bindStreamHandler(handle, assistantId, audioCtx);

    const history = messages.filter(m => !m.streaming).map(m => ({ role: m.role, content: m.content }));

    handle.send(JSON.stringify({
      prompt: userMsg.content,
      history,
      max_tokens: 512,
      temperature: 0.7,
      enable_thinking: false,
      image_b64: currentImageB64 ?? undefined,
      voice: isTTS && selectedVoice ? selectedVoice : undefined,
    }));
  }, [model, loadedModelId, input, streaming, messages, imageB64, imageThumb, isTTS, selectedVoice, connectWebSocket, bindStreamHandler]);

  // -------------------------------------------------------------------------
  // VLM image upload
  // -------------------------------------------------------------------------

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

  // -------------------------------------------------------------------------
  // STT: audio upload / recording
  // -------------------------------------------------------------------------

  const sendSTTAudio = useCallback((audioB64Data: string | null, fileName: string, fileId?: string) => {
    const modelId = loadedModelId || model?.model_id;
    if (!modelId || streaming) return;

    const userMsg: ChatMessage = { id: `user-${Date.now()}`, role: 'user', content: t('simple.v1.audioLabel', { name: fileName }) };
    const assistantId = `assistant-${Date.now()}`;
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', streaming: true };

    streamingMsgIdRef.current = assistantId;
    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setStreaming(true);
    setStatus('');

    const handle = connectWebSocket(modelId);
    bindStreamHandler(handle, assistantId, null);

    handle.send(JSON.stringify({
      audio_b64: audioB64Data || undefined,
      file_id: fileId || undefined,
      file_name: fileName,
      language: input.trim() || undefined,
    }));
  }, [model, loadedModelId, streaming, input, connectWebSocket, bindStreamHandler]);

  const handleAudioFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || streaming) return;
    e.target.value = '';

    const formData = new FormData();
    formData.append('file', file);

    try {
      setStatus(t('simple.v1.uploadingAudio'));
      const resp = await fetch('/api/chat/upload-audio', { method: 'POST', body: formData });
      const { file_id, file_name } = await resp.json();
      setStatus('');
      sendSTTAudio(null, file_name || file.name, file_id);
    } catch {
      setStatus('');
    }
  }, [streaming, sendSTTAudio]);

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

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        const reader = new FileReader();
        reader.onload = (ev) => {
          const dataUrl = ev.target?.result as string;
          sendSTTAudio(dataUrl.split(',')[1], 'recording.webm');
        };
        reader.readAsDataURL(blob);
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setRecording(true);
    } catch {
      // Mic access denied
    }
  }, [recording, streaming, sendSTTAudio]);

  // -------------------------------------------------------------------------
  // Misc handlers
  // -------------------------------------------------------------------------

  const handleCancel = useCallback(() => {
    wsHandleRef.current?.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  const handleClear = useCallback(() => {
    setMessages([]);
    wsHandleRef.current?.close();
    wsHandleRef.current = null;
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNext = () => {
    setCurrentStep(5);
    navigate('/simple/export');
  };

  // -------------------------------------------------------------------------
  // UI helpers
  // -------------------------------------------------------------------------

  const categoryLabel: Record<string, string> = {
    llm: 'LLM', vlm: 'VLM', tts: 'TTS', stt: 'ASR',
  };
  const categoryBadgeColor: Record<string, string> = {
    llm: 'bg-blue-100 text-blue-700 dark:bg-blue-500/15 dark:text-blue-400',
    vlm: 'bg-purple-100 text-purple-700 dark:bg-purple-500/15 dark:text-purple-400',
    tts: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-400',
    stt: 'bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-400',
  };

  const emptyHint = isSTT
    ? t('simple.v1.hintSTT')
    : isTTS
      ? t('simple.v1.hintTTS')
      : hasVision
        ? t('simple.v1.hintVLM')
        : t('simple.test.empty');

  const placeholder = isSTT
    ? t('simple.v1.placeholderSTT')
    : isTTS
      ? t('simple.v1.placeholderTTS')
      : t('simple.test.placeholder');

  const cat = model?.model_category || 'llm';

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <WizardShell
      steps={WIZARD_STEPS(t)}
      currentStep={4}
      onBack={() => { setCurrentStep(3); navigate('/simple/optimize'); }}
      onNext={handleNext}
      nextDisabled={false}
      nextLabel={t('simple.test.skipOrNext')}
      onStepClick={(s) => { setCurrentStep(s); }}
    >
      {/* Header */}
      <div className="text-center">
        <div className="mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-stone-100 dark:bg-stone-800">
          <MessageCircle size={24} className="text-stone-600 dark:text-stone-400" />
        </div>
        <h2 className="mb-1 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.test.title')}
        </h2>
        <p className="mb-2 text-stone-500 dark:text-stone-400">
          {t('simple.test.subtitle')}
        </p>
        {/* Model badge */}
        {model && (
          <div className="mb-4 flex items-center justify-center gap-2">
            <span className="text-sm text-stone-500 dark:text-stone-400">{model.model_name}</span>
            <span className={cn('rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase', categoryBadgeColor[cat] || categoryBadgeColor.llm)}>
              {categoryLabel[cat] || cat}
            </span>
          </div>
        )}
      </div>

      {/* TTS voice selector / loading */}
      {isTTS && loadingVoices && (
        <div className="mb-3 flex items-center justify-center gap-2 text-sm text-stone-400 dark:text-stone-500">
          <Loader2 size={14} className="animate-spin" />
          {t('simple.v1.loadingVoices')}
        </div>
      )}
      {isTTS && !loadingVoices && ttsVoices.length > 0 && (
        <div className="mb-3 flex items-center justify-center gap-2">
          <label className="text-xs text-stone-500 dark:text-stone-400">{t('simple.v1.voiceLabel')}</label>
          <select
            value={selectedVoice}
            onChange={(e) => setSelectedVoice(e.target.value)}
            className="rounded-lg border border-stone-200 px-2.5 py-1 text-xs text-stone-700 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300"
          >
            {ttsVoices.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      )}

      {/* Chat area */}
      <div className="overflow-hidden rounded-2xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
        {/* Messages */}
        <div className="h-80 overflow-y-auto p-4">
          {messages.length === 0 && (
            <p className="py-12 text-center text-sm text-stone-400 dark:text-stone-500">
              {emptyHint}
            </p>
          )}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={cn(
                'mb-3 max-w-[85%] rounded-2xl px-4 py-2.5 text-sm',
                msg.role === 'user'
                  ? 'ml-auto bg-stone-900 text-white dark:bg-stone-200 dark:text-stone-900'
                  : 'bg-stone-100 text-stone-800 dark:bg-stone-800 dark:text-stone-200',
              )}
            >
              {/* User image preview */}
              {msg.imageThumb && (
                <div className="mb-2">
                  <img src={msg.imageThumb} alt="uploaded" className="max-h-24 rounded-lg object-contain" />
                </div>
              )}
              {/* TTS audio player */}
              {msg.audioB64 && (
                <div className="mb-2">
                  <AudioPlayer audioB64={msg.audioB64} duration={msg.audioDuration} />
                </div>
              )}
              {/* Text content */}
              <div>
                {msg.audioB64 ? '' : msg.role === 'assistant' ? (
                  <MarkdownContent content={stripThinking(msg.content)} />
                ) : (
                  <span className="whitespace-pre-wrap">{stripThinking(msg.content)}</span>
                )}
                {!msg.audioB64 && msg.streaming && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 bg-stone-500 animate-pulse rounded-sm" />
                )}
              </div>
              {/* Metrics */}
              {!msg.streaming && msg.role === 'assistant' && msg.tokensPerSec != null && (
                <div className="mt-1.5 flex gap-3 border-t border-stone-200/50 pt-1 text-[10px] text-stone-400 dark:border-stone-700 dark:text-stone-500">
                  <span>{t('simple.v1.tokens', { count: String(msg.totalTokens) })}</span>
                  <span>{t('simple.v1.tokPerSec', { count: String(msg.tokensPerSec) })}</span>
                  <span>{msg.totalTime?.toFixed(1)}s</span>
                </div>
              )}
            </div>
          ))}
          {/* Status indicator */}
          {status && (
            <div className="mb-3 flex items-center gap-2 text-xs text-stone-400 dark:text-stone-500">
              <Loader2 size={12} className="animate-spin" />
              {status}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
        <div className="border-t border-stone-200 p-3 dark:border-stone-800">
          {/* Image preview */}
          {imageThumb && (
            <div className="mb-2 flex items-center gap-2">
              <div className="relative">
                <img src={imageThumb} alt="to send" className="h-14 rounded-lg object-contain border border-stone-200 dark:border-stone-700" />
                <button
                  onClick={() => { setImageB64(null); setImageThumb(null); }}
                  className="absolute -right-1.5 -top-1.5 rounded-full bg-stone-500 p-0.5 text-white hover:bg-stone-700"
                >
                  <X size={10} />
                </button>
              </div>
            </div>
          )}

          <div className="flex items-end gap-2">
            {/* Left action buttons */}
            <div className="flex shrink-0 gap-1 pb-0.5">
              {/* Clear */}
              {messages.length > 0 && (
                <button
                  type="button"
                  onClick={handleClear}
                  className="rounded-lg p-2 text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-300"
                  title={t('chat.clear')}
                >
                  <Trash2 size={18} />
                </button>
              )}
              {/* VLM: image upload */}
              {hasVision && !isSTT && (
                <>
                  <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={handleImageSelect} className="hidden" />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={streaming}
                    className="rounded-lg p-2 text-stone-400 hover:bg-stone-100 hover:text-stone-600 disabled:opacity-50 dark:hover:bg-stone-800 dark:hover:text-stone-300"
                    title={t('simple.v1.uploadImage')}
                  >
                    <Image size={18} />
                  </button>
                </>
              )}
              {/* STT: audio upload + mic */}
              {isSTT && (
                <>
                  <input ref={audioFileInputRef} type="file" accept="audio/*" onChange={handleAudioFileSelect} className="hidden" />
                  <button
                    onClick={() => audioFileInputRef.current?.click()}
                    disabled={streaming}
                    className="rounded-lg p-2 text-stone-400 hover:bg-stone-100 disabled:opacity-50 dark:hover:bg-stone-800"
                    title={t('simple.v1.uploadAudio')}
                  >
                    <Upload size={18} />
                  </button>
                  <button
                    onClick={toggleRecording}
                    disabled={streaming && !recording}
                    className={cn(
                      'rounded-lg p-2 disabled:opacity-50',
                      recording
                        ? 'text-red-500 bg-red-50 dark:bg-red-500/10'
                        : 'text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-800',
                    )}
                    title={recording ? t('simple.v1.stopRecording') : t('simple.v1.startRecording')}
                  >
                    {recording ? <MicOff size={18} /> : <Mic size={18} />}
                  </button>
                </>
              )}
            </div>

            {/* Text input */}
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={!isSTT ? handleKeyDown : undefined}
              placeholder={placeholder}
              rows={1}
              className="flex-1 resize-none rounded-xl border border-stone-200 bg-stone-50 px-4 py-2.5 text-sm text-stone-900 placeholder-stone-400 outline-none transition-colors focus:border-stone-400 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-100 dark:placeholder-stone-500 dark:focus:border-stone-600"
              disabled={streaming || (isTTS && loadingVoices)}
            />

            {/* Right action buttons */}
            {!isSTT && (
              <button
                type="button"
                onClick={streaming ? handleCancel : handleSend}
                disabled={!streaming && (!input.trim() || (isTTS && loadingVoices))}
                className={cn(
                  'shrink-0 rounded-xl p-2.5 transition-all duration-200',
                  streaming
                    ? 'text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10'
                    : input.trim()
                      ? 'bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200'
                      : 'bg-stone-100 text-stone-400 dark:bg-stone-800 dark:text-stone-600',
                )}
              >
                {streaming ? <Square size={18} /> : <Send size={18} />}
              </button>
            )}
            {isSTT && streaming && (
              <button
                type="button"
                onClick={handleCancel}
                className="shrink-0 rounded-xl p-2.5 text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10"
              >
                <Square size={18} />
              </button>
            )}
          </div>
        </div>
      </div>
    </WizardShell>
  );
}
