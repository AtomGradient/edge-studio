// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * useModelChat — lightweight hook for model inference on the Architecture page.
 * Mirrors Chat page: createReconnectingWebSocket + buildChatWsUrl + pendingSend queue.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { createReconnectingWebSocket, buildChatWsUrl, type ReconnectingWebSocketHandle } from '@/api/websocket';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface UseModelChatOpts {
  modelId: string | null;
  systemPrompt?: string;
  maxTokens?: number;
  temperature?: number;
}

interface UseModelChatReturn {
  text: string;
  streaming: boolean;
  status: string;
  send: (userMessage: string) => void;
  cancel: () => void;
  reset: () => void;
}

export function useModelChat({
  modelId,
  systemPrompt = '',
  maxTokens = 512,
  temperature = 0.7,
}: UseModelChatOpts): UseModelChatReturn {
  const [text, setText] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState('');
  const handleRef = useRef<ReconnectingWebSocketHandle | null>(null);
  const historyRef = useRef<ChatMessage[]>([]);
  const busyRef = useRef(false);
  const requestSeqRef = useRef(0);

  const closeActiveHandle = useCallback((sendCancel = false) => {
    requestSeqRef.current += 1;
    const handle = handleRef.current;
    if (!handle) {
      busyRef.current = false;
      return;
    }

    if (sendCancel) {
      handle.send(JSON.stringify({ type: 'cancel' }));
    }
    handle.close();
    handleRef.current = null;
    busyRef.current = false;
  }, []);

  useEffect(() => {
    let cancelled = false;
    closeActiveHandle(true);
    historyRef.current = [];
    queueMicrotask(() => {
      if (cancelled) return;
      setText('');
      setStreaming(false);
      setStatus('');
    });

    return () => {
      cancelled = true;
      closeActiveHandle(true);
    };
  }, [modelId, closeActiveHandle]);

  const send = useCallback((userMessage: string) => {
    if (!modelId) { setStatus('No model loaded'); return; }
    if (busyRef.current) return;

    closeActiveHandle(true);
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;

    const history: ChatMessage[] = [];
    if (systemPrompt) history.push({ role: 'system', content: systemPrompt });
    history.push(...historyRef.current);

    setText('');
    setStreaming(true);
    setStatus('Connecting...');
    busyRef.current = true;
    let accumulated = '';

    let handle: ReconnectingWebSocketHandle | null = null;
    const isCurrentRequest = () => requestSeqRef.current === requestSeq && handleRef.current === handle;

    handle = createReconnectingWebSocket(buildChatWsUrl(modelId), {
      maxRetries: 0,
      onOpen: () => {
        if (isCurrentRequest()) setStatus('Generating...');
      },
      onClose: (e) => {
        if (!isCurrentRequest()) return;
        if (busyRef.current) {
          setStreaming(false);
          busyRef.current = false;
          if (!accumulated) setStatus(`Closed (${e.code}) without response`);
        }
        handleRef.current = null;
      },
      onError: () => {
        if (!isCurrentRequest()) return;
        setStatus('Connection error');
        setText(accumulated || 'Connection error — check browser console');
        setStreaming(false);
        busyRef.current = false;
      },
    });
    handleRef.current = handle;

    handle.setOnMessage((event) => {
      if (!isCurrentRequest()) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          // Backend sends "token" field (not "text")
          const tok = data.token ?? data.text ?? '';
          if (tok) {
            accumulated += tok;
            setText(accumulated);
          }
        } else if (data.type === 'complete') {
          // Use full_text from complete event if present (more reliable than accumulated)
          const finalText = data.full_text || accumulated;
          if (finalText !== accumulated) {
            setText(finalText);
            accumulated = finalText;
          }
          historyRef.current.push({ role: 'user', content: userMessage });
          historyRef.current.push({ role: 'assistant', content: accumulated });
          setStreaming(false);
          setStatus('');
          busyRef.current = false;
          handle.close();
          if (handleRef.current === handle) handleRef.current = null;
        } else if (data.type === 'error') {
          setText(accumulated || `Error: ${data.message || 'Unknown error'}`);
          setStatus(`Error: ${data.message || 'unknown'}`);
          setStreaming(false);
          busyRef.current = false;
          handle.close();
          if (handleRef.current === handle) handleRef.current = null;
        } else if (data.type === 'status') {
          setStatus(data.message || 'Working...');
        }
      } catch (err) {
        console.error('[useModelChat] parse error', err, event.data);
      }
    });

    handle.send(JSON.stringify({
      prompt: userMessage,
      history: history.map(m => ({ role: m.role, content: m.content })),
      max_tokens: maxTokens,
      temperature,
      enable_thinking: false,
    }));
  }, [modelId, systemPrompt, maxTokens, temperature, closeActiveHandle]);

  const cancel = useCallback(() => {
    handleRef.current?.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  const reset = useCallback(() => {
    closeActiveHandle(true);
    setText('');
    setStreaming(false);
    setStatus('');
    busyRef.current = false;
    historyRef.current = [];
  }, [closeActiveHandle]);

  return { text, streaming, status, send, cancel, reset };
}
