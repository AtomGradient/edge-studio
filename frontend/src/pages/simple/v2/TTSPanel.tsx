// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * TTSPanel — Standalone text-to-speech experience.
 *
 * Text input + voice selector → audio playback.
 * Extracted from VoicePanel TTSTab as an independent component.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Volume2, Pause, Send, Loader2 } from 'lucide-react';
import { useT } from '@/i18n';
import { cn } from '@/lib/utils';
import { buildChatWsUrl, createReconnectingWebSocket } from '@/api/websocket';
import axios from 'axios';

interface TTSPanelProps {
  modelId: string;
}

export default function TTSPanel({ modelId }: TTSPanelProps) {
  const t = useT();
  const [text, setText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [loadingVoices, setLoadingVoices] = useState(true);
  const [instructMode, setInstructMode] = useState(false);
  const [instruct, setInstruct] = useState('');
  const [audioSrc, setAudioSrc] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState('');
  const audioRef = useRef<HTMLAudioElement>(null);
  const wsRef = useRef<ReturnType<typeof createReconnectingWebSocket> | null>(null);

  // Load voices on mount
  useEffect(() => {
    if (!modelId) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await axios.get(`/api/chat/${encodeURIComponent(modelId)}/tts-voices`);
        if (!cancelled) {
          const v = res.data.voices || [];
          setVoices(v);
          if (v.length > 0) setSelectedVoice(v[0]);
          // VoiceDesign models: no predefined voices, use instruct mode
          if (v.length === 0 && res.data.instruct_mode) {
            setInstructMode(true);
            setInstruct('A gentle female voice with clear pronunciation');
          }
        }
      } catch {
        // Keep the panel usable even if voice discovery fails.
      }
      if (!cancelled) setLoadingVoices(false);
    })();
    return () => { cancelled = true; };
  }, [modelId]);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  // Audio ended
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onEnded = () => setPlaying(false);
    audio.addEventListener('ended', onEnded);
    return () => audio.removeEventListener('ended', onEnded);
  }, [audioSrc]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) audio.pause(); else audio.play();
    setPlaying(!playing);
  };

  const handleGenerate = useCallback(() => {
    if (!text.trim() || !modelId || generating) return;
    setGenerating(true);
    setAudioSrc(null);
    setError('');

    const handle = (() => {
      if (wsRef.current && wsRef.current.readyState() === WebSocket.OPEN) return wsRef.current;
      wsRef.current?.close();
      const h = createReconnectingWebSocket(buildChatWsUrl(modelId));
      wsRef.current = h;
      return h;
    })();

    handle.setOnMessage((event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'complete' && data.audio_b64) {
          setAudioSrc(`data:audio/wav;base64,${data.audio_b64}`);
          setGenerating(false);
        } else if (data.type === 'error') {
          setError(data.message || t('simple.error.unknown'));
          setGenerating(false);
        }
      } catch {
        // Ignore malformed websocket frames.
      }
    });

    const payload: Record<string, unknown> = { prompt: text.trim() };
    if (instructMode && instruct.trim()) {
      payload.instruct = instruct.trim();
    } else if (selectedVoice) {
      payload.voice = selectedVoice;
    }
    handle.send(JSON.stringify(payload));
  }, [text, modelId, generating, selectedVoice, instructMode, instruct, t]);

  if (loadingVoices) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-stone-400">
        <Loader2 size={14} className="animate-spin" />
        {t('simple.v2.setup.ttsLoading')}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <p className="text-center text-sm text-stone-500 dark:text-stone-400">
        {t('simple.v2.setup.ttsHint')}
      </p>

      {/* Voice selector (predefined voices) */}
      {voices.length > 0 && (
        <div className="flex items-center justify-center gap-2">
          <label className="text-xs text-stone-500 dark:text-stone-400">
            {t('simple.v2.setup.ttsVoice')}
          </label>
          <select
            value={selectedVoice}
            onChange={(e) => setSelectedVoice(e.target.value)}
            className="rounded-lg border border-stone-200 px-2.5 py-1.5 text-xs text-stone-700 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300"
          >
            {voices.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      )}

      {/* Instruct mode: describe voice in natural language (VoiceDesign models) */}
      {instructMode && (
        <div className="space-y-1.5">
          <label className="block text-center text-xs text-stone-500 dark:text-stone-400">
            {t('simple.v2.setup.ttsInstruct')}
          </label>
          <input
            type="text"
            value={instruct}
            onChange={(e) => setInstruct(e.target.value)}
            placeholder={t('simple.v2.setup.ttsInstructPlaceholder')}
            disabled={generating}
            className="w-full rounded-xl border border-stone-200 bg-white py-2 pl-4 pr-4 text-sm placeholder-stone-400 outline-none focus:border-stone-400 focus:ring-1 focus:ring-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100 dark:placeholder-stone-500"
          />
        </div>
      )}

      {/* Text input + generate */}
      <div className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
          placeholder={t('simple.v2.setup.ttsPlaceholder')}
          disabled={generating}
          className="flex-1 rounded-xl border border-stone-200 bg-white py-2.5 pl-4 pr-4 text-sm placeholder-stone-400 outline-none focus:border-stone-400 focus:ring-1 focus:ring-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100 dark:placeholder-stone-500"
        />
        <button
          type="button"
          onClick={handleGenerate}
          disabled={!text.trim() || generating}
          className={cn(
            'flex h-10 items-center gap-1.5 rounded-xl px-4 text-sm font-medium transition-all',
            text.trim() && !generating
              ? 'bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900'
              : 'bg-stone-100 text-stone-400 dark:bg-stone-800 dark:text-stone-600',
          )}
        >
          {generating ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
          {generating ? t('simple.v2.setup.ttsGenerating') : t('simple.v2.setup.ttsGenerate')}
        </button>
      </div>

      {/* Audio player */}
      {audioSrc && (
        <div className="flex items-center justify-center gap-3 rounded-xl border border-stone-200 bg-stone-50 p-4 dark:border-stone-700 dark:bg-stone-900">
          <audio ref={audioRef} src={audioSrc} />
          <button
            type="button"
            onClick={togglePlay}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900"
          >
            {playing ? <Pause size={16} /> : <Volume2 size={16} />}
          </button>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}
