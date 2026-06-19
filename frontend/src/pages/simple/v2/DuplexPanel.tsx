// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * DuplexPanel — Voice duplex experience: speak → ASR → LLM → TTS → listen.
 *
 * Three models chained via independent WebSocket connections.
 * State machine with error recovery and cancel support.
 * Frontend ensures serial GPU access (idle-gating).
 */

import { useState, useRef, useCallback, useEffect, type FormEvent } from 'react';
import { Mic, MicOff, X, Loader2, Volume2, Send } from 'lucide-react';
import { useT } from '@/i18n';
import { cn } from '@/lib/utils';
import { stripThinking } from '@/lib/textUtils';
import MarkdownContent from '@/components/MarkdownContent';
import { buildChatWsUrl, createReconnectingWebSocket } from '@/api/websocket';

interface DuplexPanelProps {
  asrModelId: string;
  llmModelId: string;
  ttsModelId: string;
  voice?: string;
  instruct?: string;
  imageB64?: string | null;
  onImageConsumed?: () => void;
}

type DuplexState = 'idle' | 'recording' | 'transcribing' | 'thinking' | 'speaking' | 'error';

interface DuplexError {
  stage: 'asr' | 'llm' | 'tts';
  message: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  audioSrc?: string;  // TTS audio for assistant messages
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown error';
}

function isCancelledError(error: unknown): boolean {
  return error instanceof Error && error.message === 'cancelled';
}

export default function DuplexPanel({ asrModelId, llmModelId, ttsModelId, voice, instruct, imageB64, onImageConsumed }: DuplexPanelProps) {
  const t = useT();
  const [state, setState] = useState<DuplexState>('idle');
  const [error, setError] = useState<DuplexError | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [statusText, setStatusText] = useState('');
  const [textInput, setTextInput] = useState('');

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioRef = useRef<HTMLAudioElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cancelledRef = useRef(false);

  // Web Audio API for gapless TTS streaming playback
  const audioCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef(0);

  // WebSocket refs — ASR has its own, LLM+TTS share one (duplex interleaving)
  const wsAsrRef = useRef<ReturnType<typeof createReconnectingWebSocket> | null>(null);
  const wsLlmRef = useRef<ReturnType<typeof createReconnectingWebSocket> | null>(null);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup WebSockets on unmount
  useEffect(() => {
    return () => {
      wsAsrRef.current?.close();
      wsLlmRef.current?.close();
    };
  }, []);

  // Cleanup AudioContext on unmount
  useEffect(() => {
    return () => { audioCtxRef.current?.close(); };
  }, []);

  const getWs = useCallback((
    ref: React.MutableRefObject<ReturnType<typeof createReconnectingWebSocket> | null>,
    modelId: string,
  ) => {
    if (ref.current && ref.current.readyState() === WebSocket.OPEN) return ref.current;
    ref.current?.close();
    const handle = createReconnectingWebSocket(buildChatWsUrl(modelId));
    ref.current = handle;
    return handle;
  }, []);

  // === Chain step 1: ASR ===
  const runASR = useCallback((audioB64: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      if (cancelledRef.current) { reject(new Error('cancelled')); return; }

      const handle = getWs(wsAsrRef, asrModelId);
      let fullText = '';

      handle.setOnMessage((event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'token') {
            fullText += data.token || '';
          } else if (data.type === 'complete') {
            resolve(data.full_text || fullText || '');
          } else if (data.type === 'error') {
            reject(new Error(data.message || 'ASR error'));
          }
        } catch {
          // Ignore malformed websocket frames.
        }
      });

      handle.send(JSON.stringify({ audio_b64: audioB64, file_name: 'recording.webm' }));
    });
  }, [asrModelId, getWs]);

  // === Chain step 2: LLM + TTS (duplex interleaved) ===
  // Backend interleaves LLM token generation with TTS audio streaming.
  // Returns { fullText, audioB64 } — audio is already playing via audio_chunk events.
  const runLLM = useCallback((prompt: string, history: { role: string; content: string }[], image?: string | null): Promise<{ fullText: string; audioB64: string | null }> => {
    return new Promise((resolve, reject) => {
      if (cancelledRef.current) { reject(new Error('cancelled')); return; }

      // Create AudioContext for gapless TTS streaming (audio_chunk events)
      const audioCtx = new AudioContext({ sampleRate: 24000 });
      audioCtxRef.current = audioCtx;
      nextPlayTimeRef.current = audioCtx.currentTime;

      const handle = getWs(wsLlmRef, llmModelId);
      let fullText = '';
      let audioStarted = false;

      handle.setOnMessage((event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'token') {
            fullText += data.token || '';
            const display = stripThinking(fullText);
            setMessages((prev) => {
              const updated = [...prev];
              if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
                updated[updated.length - 1] = { ...updated[updated.length - 1], text: display };
              }
              return updated;
            });
          } else if (data.type === 'audio_chunk' && data.audio_b64) {
            // TTS audio chunk — decode and schedule for gapless playback
            if (!audioStarted) {
              audioStarted = true;
              setState('speaking');
              setStatusText(t('simple.v2.setup.duplexSpeaking'));
            }
            const binary = atob(data.audio_b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            audioCtx.decodeAudioData(bytes.buffer.slice(0)).then((buffer) => {
              const source = audioCtx.createBufferSource();
              source.buffer = buffer;
              source.connect(audioCtx.destination);
              const now = audioCtx.currentTime;
              const startAt = Math.max(nextPlayTimeRef.current, now);
              source.start(startAt);
              nextPlayTimeRef.current = startAt + buffer.duration;
            }).catch(() => {
              // Ignore individual audio chunk decode failures.
            });
          } else if (data.type === 'complete') {
            if (data.full_text) fullText = data.full_text;
            const display = stripThinking(fullText);
            setMessages((prev) => {
              const updated = [...prev];
              if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
                updated[updated.length - 1] = { ...updated[updated.length - 1], text: display };
              }
              return updated;
            });
            resolve({ fullText: display, audioB64: data.audio_b64 || null });
          } else if (data.type === 'error') {
            reject(new Error(data.message || 'LLM error'));
          }
        } catch {
          // Ignore malformed websocket frames.
        }
      });

      // Send prompt with tts_model_id to enable duplex interleaving.
      // Explicitly disable thinking — voice mode never needs it.
      handle.send(JSON.stringify({
        prompt,
        history,
        tts_model_id: ttsModelId,
        enable_thinking: false,
        ...(voice ? { voice } : {}),
        ...(instruct ? { instruct } : {}),
        ...(image ? { image_b64: image } : {}),
      }));
    });
  }, [llmModelId, ttsModelId, voice, instruct, getWs, t]);

  // === Full chain (duplex: ASR → LLM+TTS interleaved) ===
  const runChain = useCallback(async (audioB64: string) => {
    cancelledRef.current = false;

    try {
      // Step 1: ASR
      setState('transcribing');
      setStatusText(t('simple.v2.setup.duplexListening'));
      const transcribedText = await runASR(audioB64);
      if (cancelledRef.current) return;

      if (!transcribedText.trim()) {
        setState('idle');
        return;
      }

      // Add user message (ASR result)
      setMessages((prev) => [...prev, { role: 'user', text: transcribedText }]);

      // Step 2: LLM + TTS interleaved — audio starts playing during LLM generation
      setState('thinking');
      setStatusText(t('simple.v2.setup.duplexThinking'));

      const history = messages.map((m) => ({ role: m.role, content: m.text }));
      setMessages((prev) => [...prev, { role: 'assistant', text: '' }]);

      // Capture and consume image (one-shot)
      const currentImage = imageB64 || null;
      if (currentImage) onImageConsumed?.();

      const { audioB64: fullAudioB64 } = await runLLM(transcribedText, history, currentImage);
      if (cancelledRef.current) return;

      // Attach full audio to message for replay button
      if (fullAudioB64) {
        const audioSrc = `data:audio/wav;base64,${fullAudioB64}`;
        setMessages((prev) => {
          const updated = [...prev];
          if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1] = { ...updated[updated.length - 1], audioSrc };
          }
          return updated;
        });
      }

      // Wait for streaming playback to finish before returning to idle
      const ctx = audioCtxRef.current;
      const remaining = ctx ? nextPlayTimeRef.current - ctx.currentTime : 0;
      if (remaining > 0) {
        await new Promise((r) => setTimeout(r, remaining * 1000));
      }
      setState('idle');

    } catch (err: unknown) {
      if (cancelledRef.current || isCancelledError(err)) {
        setState('idle');
        return;
      }

      let stage: 'asr' | 'llm' | 'tts' = 'asr';
      if (state === 'thinking' || state === 'speaking') stage = 'llm';

      setError({ stage, message: getErrorMessage(err) });
      setState('error');
    }
  }, [messages, runASR, runLLM, state, t, imageB64, onImageConsumed]);

  const isProcessing = state === 'transcribing' || state === 'thinking' || state === 'speaking';

  // === Text input chain (skip ASR, go directly to LLM+TTS) ===
  const sendText = useCallback(async (text: string) => {
    if (!text.trim() || isProcessing) return;
    cancelledRef.current = false;
    setTextInput('');

    try {
      setMessages((prev) => [...prev, { role: 'user', text }]);

      setState('thinking');
      setStatusText(t('simple.v2.setup.duplexThinking'));

      const history = messages.map((m) => ({ role: m.role, content: m.text }));
      setMessages((prev) => [...prev, { role: 'assistant', text: '' }]);

      // Capture and consume image (one-shot)
      const currentImage = imageB64 || null;
      if (currentImage) onImageConsumed?.();

      const { audioB64: fullAudioB64 } = await runLLM(text, history, currentImage);
      if (cancelledRef.current) return;

      if (fullAudioB64) {
        const audioSrc = `data:audio/wav;base64,${fullAudioB64}`;
        setMessages((prev) => {
          const updated = [...prev];
          if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1] = { ...updated[updated.length - 1], audioSrc };
          }
          return updated;
        });
      }

      const ctx = audioCtxRef.current;
      const remaining = ctx ? nextPlayTimeRef.current - ctx.currentTime : 0;
      if (remaining > 0) {
        await new Promise((r) => setTimeout(r, remaining * 1000));
      }
      setState('idle');

    } catch (err: unknown) {
      if (cancelledRef.current || isCancelledError(err)) {
        setState('idle');
        return;
      }
      setError({ stage: 'llm', message: getErrorMessage(err) });
      setState('error');
    }
  }, [messages, runLLM, isProcessing, t, imageB64, onImageConsumed]);

  const handleTextSubmit = useCallback((e: FormEvent) => {
    e.preventDefault();
    sendText(textInput);
  }, [sendText, textInput]);

  // === Recording controls ===
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      const recorder = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) audioChunksRef.current.push(ev.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop());
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        const reader = new FileReader();
        reader.onload = (ev) => {
          const b64 = (ev.target?.result as string).split(',')[1];
          runChain(b64);
        };
        reader.readAsDataURL(blob);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setState('recording');
    } catch {
      setError({ stage: 'asr', message: t('simple.v2.setup.micDenied') });
      setState('error');
    }
  }, [runChain, t]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    // State transitions to 'transcribing' in runChain
  }, []);

  const handleCancel = useCallback(() => {
    cancelledRef.current = true;

    // Stop recording if active
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stream.getTracks().forEach((tr) => tr.stop());
      mediaRecorderRef.current.stop();
    }

    // Stop streaming audio playback
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    // Stop HTML audio replay
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // Send cancel to active WebSocket
    [wsAsrRef, wsLlmRef].forEach((ref) => {
      try {
        if (ref.current && ref.current.readyState() === WebSocket.OPEN) {
          ref.current.send(JSON.stringify({ type: 'cancel' }));
        }
      } catch {
        // Best-effort cancellation.
      }
    });

    setState('idle');
    setError(null);
  }, []);

  const handleMainButton = useCallback(() => {
    if (state === 'idle' || state === 'error') {
      setError(null);
      startRecording();
    } else if (state === 'recording') {
      stopRecording();
    } else {
      handleCancel();
    }
  }, [state, startRecording, stopRecording, handleCancel]);

  return (
    <div className="space-y-4">
      {/* Chat messages */}
      <div className="min-h-[200px] max-h-[50vh] space-y-3 overflow-y-auto rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
        {messages.length === 0 && (
          <p className="py-8 text-center text-sm text-stone-400">
            {t('simple.v2.setup.duplexEmptyHybrid')}
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i}>
            <div
              className={cn(
                'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm',
                msg.role === 'user'
                  ? 'ml-auto bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
                  : 'bg-stone-100 text-stone-800 dark:bg-stone-800 dark:text-stone-200',
              )}
            >
              {msg.role === 'assistant' ? (
                <MarkdownContent content={msg.text || (state === 'thinking' && i === messages.length - 1 ? '\u2588' : '')} />
              ) : (
                <p className="whitespace-pre-wrap">
                  {msg.text || (state === 'thinking' && i === messages.length - 1 ? '\u2588' : '')}
                </p>
              )}
            </div>
            {/* Audio playback button for assistant messages */}
            {msg.role === 'assistant' && msg.audioSrc && (
              <button
                type="button"
                onClick={() => {
                  if (audioRef.current) {
                    audioRef.current.src = msg.audioSrc!;
                    audioRef.current.play().catch(() => {});
                  }
                }}
                className="mt-1 inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              >
                <Volume2 size={12} />
                {t('simple.v2.setup.duplexReplay')}
              </button>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Status indicator */}
      {isProcessing && (
        <div className="flex items-center justify-center gap-2 text-sm text-stone-500">
          <Loader2 size={14} className="animate-spin" />
          <span>{statusText}</span>
        </div>
      )}

      {/* Error display */}
      {state === 'error' && error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-center text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
          {t('simple.v2.setup.duplexError', { stage: error.stage.toUpperCase() })}: {error.message}
        </div>
      )}

      {/* Input area: text input + mic button */}
      <form onSubmit={handleTextSubmit} className="flex items-end gap-2">
        <input
          type="text"
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          disabled={isProcessing || state === 'recording'}
          placeholder={t('simple.v2.setup.duplexTextPlaceholder')}
          className="min-w-0 flex-1 rounded-full border border-stone-200 bg-white px-4 py-3 text-sm outline-none transition-colors focus:border-stone-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-900 dark:focus:border-stone-500"
        />
        {textInput.trim() ? (
          <button
            type="submit"
            disabled={isProcessing}
            className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-stone-900 text-white transition-all hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            <Send size={18} />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleMainButton}
            className={cn(
              'flex h-12 w-12 shrink-0 items-center justify-center rounded-full transition-all',
              state === 'recording'
                ? 'animate-pulse bg-red-500 text-white'
                : isProcessing
                  ? 'bg-stone-400 text-white hover:bg-stone-500 dark:bg-stone-600 dark:hover:bg-stone-500'
                  : 'bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200',
            )}
          >
            {state === 'recording' ? (
              <MicOff size={18} />
            ) : isProcessing ? (
              <X size={18} />
            ) : (
              <Mic size={18} />
            )}
          </button>
        )}
      </form>

      {/* Hidden audio element for TTS playback */}
      <audio ref={audioRef} className="hidden" />
    </div>
  );
}
