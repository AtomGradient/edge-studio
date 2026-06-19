// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useRef } from 'react';
import { ChevronDown, Trash2 } from 'lucide-react';
import type { ConversationSession } from '@/api/conversations';
import { cn } from '@/lib/utils';

export interface ConversationSessionMenuLabels {
  title: string;
  newSession: string;
  untitled: string;
  deleteSession: string;
  deleteSessionConfirm: string;
}

export function ConversationSessionMenu({
  sessions,
  currentSessionId,
  disabled = false,
  labels,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  formatTime,
  className,
}: {
  sessions: ConversationSession[];
  currentSessionId: string | null;
  disabled?: boolean;
  labels: ConversationSessionMenuLabels;
  onNewSession: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  formatTime: (seconds?: number | null) => string;
  className?: string;
}) {
  const detailsRef = useRef<HTMLDetailsElement>(null);
  const current = sessions.find((session) => session.session_id === currentSessionId);
  const currentTitle = current?.title || labels.newSession;

  const close = () => {
    detailsRef.current?.removeAttribute('open');
  };

  const handleDelete = (session: ConversationSession) => {
    const title = session.title || labels.untitled;
    if (!window.confirm(labels.deleteSessionConfirm.replace('{title}', title))) {
      return;
    }
    onDeleteSession(session.session_id);
  };

  return (
    <details ref={detailsRef} className={cn('relative', className)}>
      <summary
        onClick={(event) => {
          if (disabled) event.preventDefault();
        }}
        className={cn(
          'flex max-w-64 cursor-pointer list-none items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-2.5 py-1 text-xs text-stone-600 outline-none hover:bg-stone-50 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300 dark:hover:bg-stone-800',
          disabled && 'cursor-not-allowed opacity-50',
        )}
        title={labels.title}
      >
        <span className="truncate">{currentTitle}</span>
        <ChevronDown size={13} className="shrink-0 text-stone-400" />
      </summary>
      <div className="absolute right-0 z-30 mt-2 w-80 overflow-hidden rounded-lg border border-stone-200 bg-white shadow-xl dark:border-stone-700 dark:bg-stone-950">
        <button
          type="button"
          onClick={() => {
            onNewSession();
            close();
          }}
          className="block w-full px-3 py-2 text-left text-xs font-medium text-stone-700 hover:bg-stone-50 dark:text-stone-200 dark:hover:bg-stone-900"
        >
          {labels.newSession}
        </button>
        <div className="max-h-72 overflow-y-auto border-t border-stone-100 dark:border-stone-800">
          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={cn(
                'flex items-center gap-2 px-2 py-1.5',
                session.session_id === currentSessionId && 'bg-indigo-50/70 dark:bg-indigo-950/30',
              )}
            >
              <button
                type="button"
                onClick={() => {
                  onSelectSession(session.session_id);
                  close();
                }}
                className="min-w-0 flex-1 rounded-md px-2 py-1 text-left hover:bg-stone-50 dark:hover:bg-stone-900"
              >
                <div className="truncate text-xs font-medium text-stone-800 dark:text-stone-100">
                  {session.title || labels.untitled}
                </div>
                <div className="mt-0.5 text-[11px] text-stone-400">{formatTime(session.updated_at)}</div>
              </button>
              <button
                type="button"
                onClick={() => handleDelete(session)}
                className="rounded-md p-1.5 text-stone-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/30 dark:hover:text-red-300"
                title={labels.deleteSession}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </details>
  );
}
