// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * ASRPanel — Standalone speech-to-text experience.
 *
 * Microphone recording + file upload → text transcription.
 * Extracted from VoicePanel ASRTab as an independent component.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { Mic, MicOff, Upload, Loader2 } from 'lucide-react';
import { useT } from '@/i18n';
import { cn } from '@/lib/utils';
import { buildChatWsUrl, createReconnectingWebSocket } from '@/api/websocket';
import { friendlyError } from '@/lib/friendlyError';
import axios from 'axios';

interface ASRPanelProps {
  modelId: string;
}

function getAxiosDetail(error: unknown): string | undefined {
  if (!axios.isAxiosError<{ detail?: unknown }>(error)) return undefined;
  const detail = error.response?.data?.detail;
  return typeof detail === 'string' ? detail : undefined;
}

export default function ASRPanel({ modelId }: ASRPanelProps) {
  const t = useT();
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioFileInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<ReturnType<typeof createReconnectingWebSocket> | null>(null);

  const connectWs = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState() === WebSocket.OPEN) return wsRef.current;
    wsRef.current?.close();
    const handle = createReconnectingWebSocket(buildChatWsUrl(modelId));
    wsRef.current = handle;
    return handle;
  }, [modelId]);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const handleTranscription = useCallback((handle: ReturnType<typeof createReconnectingWebSocket>) => {
    let fullText = '';
    setTranscribing(true);
    setResult('');
    setError('');

    handle.setOnMessage((event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          fullText += data.token || '';
          setResult(fullText);
        } else if (data.type === 'complete') {
          if (data.full_text) fullText = data.full_text;
          setResult(fullText || '(No speech detected)');
          setTranscribing(false);
        } else if (data.type === 'error') {
          setError(data.message || t('simple.error.unknown'));
          setTranscribing(false);
        }
      } catch {
        // Ignore malformed websocket frames.
      }
    });
  }, [t]);

  // Record → send base64
  const toggleRecording = useCallback(async () => {
    if (recording) {
      mediaRecorderRef.current?.stop();
      setRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4';
      const recorder = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];
      recorder.ondataavailable = (ev) => { if (ev.data.size > 0) audioChunksRef.current.push(ev.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop());
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        const reader = new FileReader();
        reader.onload = (ev) => {
          const b64 = (ev.target?.result as string).split(',')[1];
          const handle = connectWs();
          handleTranscription(handle);
          handle.send(JSON.stringify({ audio_b64: b64, file_name: 'recording.webm' }));
        };
        reader.readAsDataURL(blob);
      };
      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch {
      setError(t('simple.v2.setup.micDenied'));
    }
  }, [recording, connectWs, handleTranscription, t]);

  // File upload → POST /api/chat/upload-audio → send file_id
  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || transcribing) return;
    e.target.value = '';

    setUploading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const resp = await axios.post('/api/chat/upload-audio', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploading(false);
      const { file_id, file_name } = resp.data;
      const handle = connectWs();
      handleTranscription(handle);
      handle.send(JSON.stringify({ file_id, file_name: file_name || file.name }));
    } catch (err: unknown) {
      setUploading(false);
      setError(friendlyError(getAxiosDetail(err), t, 'simple.error.unknown'));
    }
  }, [transcribing, connectWs, handleTranscription, t]);

  return (
    <div className="flex flex-col items-center gap-5">
      <p className="text-sm text-stone-500 dark:text-stone-400">
        {t('simple.v2.setup.asrHint')}
      </p>

      {/* Record button */}
      <button
        type="button"
        onClick={toggleRecording}
        disabled={transcribing || uploading}
        className={cn(
          'flex h-16 w-16 items-center justify-center rounded-full transition-all',
          recording
            ? 'animate-pulse bg-red-500 text-white'
            : transcribing || uploading
              ? 'bg-stone-200 text-stone-400 dark:bg-stone-800 dark:text-stone-600'
              : 'bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200',
        )}
      >
        {recording ? <MicOff size={24} /> : <Mic size={24} />}
      </button>
      <p className="text-xs text-stone-400">
        {recording ? t('simple.v2.setup.recordStop') : t('simple.v2.setup.recordStart')}
      </p>

      {/* File upload */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-stone-400">{t('simple.v2.setup.asrOrRecord')}</span>
        <input
          ref={audioFileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => audioFileInputRef.current?.click()}
          disabled={transcribing || recording || uploading}
          className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-400 dark:hover:bg-stone-800"
        >
          <Upload size={12} />
          {uploading ? t('simple.v2.setup.asrUploading') : t('simple.v2.setup.asrUpload')}
        </button>
      </div>

      {/* Transcription status */}
      {transcribing && (
        <div className="flex items-center gap-2 text-sm text-stone-500">
          <Loader2 size={14} className="animate-spin" />
          <span>{t('simple.v2.setup.recording')}</span>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="w-full rounded-xl border border-stone-200 bg-white p-4 text-sm text-stone-800 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200">
          <p className="whitespace-pre-wrap">{result}</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700 dark:border-red-800 dark:bg-red-900/20 dark:text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}
