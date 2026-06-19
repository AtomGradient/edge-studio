// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useRef, useState, useCallback } from 'react';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import '@xterm/xterm/css/xterm.css';

interface TerminalProps {
  sessionId: string;
  onExit?: (code: number) => void;
  onError?: (error: string) => void;
  className?: string;
}

export function Terminal({ sessionId, onExit, onError, className = '' }: TerminalProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<XTerm | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [connectError, setConnectError] = useState<string | null>(null);

  // Store callbacks in refs to avoid useEffect re-triggering on every render
  const onExitRef = useRef(onExit);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onExitRef.current = onExit;
    onErrorRef.current = onError;
  }, [onExit, onError]);

  useEffect(() => {
    if (!containerRef.current) return;

    const term = new XTerm({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: '#1e1e1e',
        foreground: '#d4d4d4',
        cursor: '#d4d4d4',
        selectionBackground: '#264f78',
        black: '#1e1e1e',
        red: '#f44747',
        green: '#6a9955',
        yellow: '#dcdcaa',
        blue: '#569cd6',
        magenta: '#c586c0',
        cyan: '#4ec9b0',
        white: '#d4d4d4',
        brightBlack: '#808080',
        brightRed: '#f44747',
        brightGreen: '#6a9955',
        brightYellow: '#dcdcaa',
        brightBlue: '#569cd6',
        brightMagenta: '#c586c0',
        brightCyan: '#4ec9b0',
        brightWhite: '#ffffff',
      },
      scrollback: 10000,
      allowProposedApi: true,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();

    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);
    term.open(containerRef.current);
    fitAddon.fit();

    termRef.current = term;
    fitAddonRef.current = fitAddon;

    // Connect through the current origin so Vite's /api proxy honors VLM_PORT.
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/terminal/ws/${sessionId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setConnectError(null);
      const dims = fitAddon.proposeDimensions();
      if (dims) {
        ws.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'output') {
          term.write(msg.data);
        } else if (msg.type === 'exit') {
          term.write('\r\n\x1b[90m[Process exited]\x1b[0m\r\n');
          onExitRef.current?.(msg.code ?? 0);
        } else if (msg.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }));
        }
      } catch {
        // Ignore parse errors
      }
    };

    ws.onerror = () => {
      setConnectError('Connection failed — check if backend is running');
      onErrorRef.current?.('WebSocket connection error');
    };

    ws.onclose = (event) => {
      setConnected(false);
      if (event.code !== 1000) {
        const msg = `Connection lost (code ${event.code})`;
        setConnectError(msg);
        onErrorRef.current?.(msg);
      }
    };

    // User input → PTY
    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input', data }));
      }
    });

    // Resize handling
    const handleResize = () => {
      fitAddon.fit();
      const dims = fitAddon.proposeDimensions();
      if (dims && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
      }
    };

    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(containerRef.current);

    return () => {
      resizeObserver.disconnect();
      ws.close();
      term.dispose();
    };
  }, [sessionId]); // Only re-create when sessionId changes

  return (
    <div className={`relative ${className}`}>
      <div
        ref={containerRef}
        className="h-full w-full overflow-hidden rounded-lg bg-[#1e1e1e]"
        style={{ padding: '8px' }}
      />
      {!connected && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-lg">
          <div className="text-center">
            <div className="text-sm text-gray-400">{connectError ? 'Connection Failed' : 'Connecting...'}</div>
            {connectError && (
              <div className="mt-1 text-xs text-red-400">{connectError}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


interface TerminalOverlayProps {
  sessionId: string;
  title?: string;
  onClose: () => void;
  onExit?: (code: number) => void;
}

export function TerminalOverlay({ sessionId, title = 'Terminal', onClose, onExit }: TerminalOverlayProps) {
  const [exited, setExited] = useState(false);

  const handleExit = useCallback((code: number) => {
    setExited(true);
    onExit?.(code);
  }, [onExit]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="flex h-[70vh] w-full max-w-4xl flex-col rounded-xl bg-gray-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-700 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <div className="h-3 w-3 rounded-full bg-red-500" />
              <div className="h-3 w-3 rounded-full bg-yellow-500" />
              <div className="h-3 w-3 rounded-full bg-green-500" />
            </div>
            <span className="text-sm font-medium text-gray-300">{title}</span>
          </div>
          <button
            onClick={onClose}
            className="rounded px-3 py-1 text-sm text-gray-400 hover:bg-gray-800 hover:text-gray-200"
          >
            {exited ? 'Close' : 'Close (will terminate)'}
          </button>
        </div>

        {/* Terminal */}
        <div className="flex-1 p-2">
          <Terminal
            sessionId={sessionId}
            onExit={handleExit}
            className="h-full"
          />
        </div>
      </div>
    </div>
  );
}
