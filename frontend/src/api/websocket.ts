// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import type { TaskEvent } from './types';

export function connectTaskWebSocket(
  taskId: string,
  onEvent: (event: TaskEvent) => void,
  onClose?: () => void,
): () => void {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/task/${taskId}`);

  ws.onmessage = (event) => {
    try {
      const data: TaskEvent = JSON.parse(event.data);
      if (data.type !== 'ping') {
        onEvent(data);
      }
    } catch {
      // ignore parse errors
    }
  };

  ws.onclose = () => onClose?.();
  ws.onerror = () => ws.close();

  // Return cleanup function
  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
}

// ---- Auto-reconnecting WebSocket for chat ----

export interface ReconnectingWebSocketOptions {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  onOpen?: (ws: WebSocket) => void;
  onMessage?: (event: MessageEvent) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onReconnect?: (attempt: number) => void;
}

export interface ReconnectingWebSocketHandle {
  /** Send data. If not yet open, queues the data and sends on open. Returns false if closed permanently. */
  send: (data: string) => boolean;
  /** Replace the onMessage handler (persists across reconnects) */
  setOnMessage: (handler: ((event: MessageEvent) => void) | null) => void;
  /** Manually close the connection. Does NOT trigger reconnect. */
  close: () => void;
  /** Current ready state of the underlying WebSocket */
  readyState: () => number;
}

/**
 * Create a WebSocket with auto-reconnect capability.
 * Retries up to maxRetries times with exponential backoff.
 *
 * - Normal close (code 1000) does NOT trigger reconnect.
 * - Abnormal close triggers reconnect with exponential backoff.
 * - close() on the handle manually disconnects without reconnect.
 * - send() queues data if the socket is still connecting (sends on open).
 * - setOnMessage() allows per-request handler swapping that survives reconnects.
 */
export function createReconnectingWebSocket(
  url: string,
  options?: ReconnectingWebSocketOptions,
): ReconnectingWebSocketHandle {
  const maxRetries = options?.maxRetries ?? 5;
  const baseDelay = options?.baseDelay ?? 1000;
  const maxDelay = options?.maxDelay ?? 16000;

  let ws: WebSocket | null = null;
  let retryCount = 0;
  let intentionalClose = false;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  // Mutable message handler — caller can swap via setOnMessage()
  let messageHandler: ((event: MessageEvent) => void) | null = options?.onMessage ?? null;
  // Pending data to send once the socket opens
  let pendingSend: string | null = null;

  function connect() {
    ws = new WebSocket(url);

    ws.onopen = () => {
      retryCount = 0;
      options?.onOpen?.(ws!);
      // Flush pending send
      if (pendingSend !== null && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pendingSend);
        pendingSend = null;
      }
    };

    ws.onmessage = (event) => {
      messageHandler?.(event);
    };

    ws.onerror = (event) => {
      options?.onError?.(event);
    };

    ws.onclose = (event) => {
      options?.onClose?.(event);

      // Do not reconnect if:
      // - close() was called intentionally
      // - server sent a clean close (code 1000)
      // - max retries exhausted
      if (intentionalClose || event.code === 1000 || retryCount >= maxRetries) {
        ws = null;
        return;
      }

      // Exponential backoff with jitter
      const delay = Math.min(baseDelay * Math.pow(2, retryCount), maxDelay);
      const jitter = delay * 0.2 * Math.random();
      retryCount++;

      options?.onReconnect?.(retryCount);

      retryTimer = setTimeout(() => {
        retryTimer = null;
        connect();
      }, delay + jitter);
    };
  }

  connect();

  return {
    send: (data: string) => {
      if (intentionalClose) return false;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(data);
        return true;
      }
      // Queue for when the socket opens (connecting or reconnecting)
      pendingSend = data;
      return true;
    },
    setOnMessage: (handler) => {
      messageHandler = handler;
    },
    close: () => {
      intentionalClose = true;
      pendingSend = null;
      if (retryTimer !== null) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      if (ws) {
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
        ws = null;
      }
    },
    readyState: () => ws?.readyState ?? WebSocket.CLOSED,
  };
}

/** Build the ws:// or wss:// chat URL for a given model */
export function buildChatWsUrl(modelId: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/chat/${modelId}`;
}

/** Build the ws:// or wss:// Neural Imprint chat URL for a given model */
export function buildNeuralImprintChatWsUrl(modelId: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/neural-imprint-chat/${modelId}`;
}
