// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useMemo, useRef, useState } from 'react';
import type React from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  Clock,
  Loader2,
  RefreshCw,
  Router,
  Send,
  Smartphone,
  Trash2,
  Wrench,
  XCircle,
} from 'lucide-react';
import {
  deleteJointInferenceHistoryItem,
  getJointInferenceHistory,
  getJointInferenceHistoryItem,
  listDevices,
  streamJointInferenceContinue,
  type JointInferenceHistoryItem,
} from '@/api/mesh';
import { PageHeader } from '@/components/layout/PageHeader';
import MarkdownContent from '@/components/MarkdownContent';
import { useT } from '@/i18n';

const REFRESH_MS = 2000;

type ThreadMessage = {
  id: string;
  role: string;
  content: string;
  streaming?: boolean;
  totalTokens?: number;
  tokensPerSec?: number;
  diagnostics?: ThreadMessage[];
};

type ConversationMessage = {
  role: string;
  content: string;
};

type Translator = ReturnType<typeof useT>;

export default function JointInferenceHistory() {
  const { requestId } = useParams();
  if (requestId) {
    return <JointInferenceDetail requestId={requestId} />;
  }
  return <JointInferenceList />;
}

function JointInferenceList() {
  const t = useT();
  const queryClient = useQueryClient();
  const [peerFilter, setPeerFilter] = useState('');
  const devicesQ = useQuery({
    queryKey: ['mesh', 'devices', 'joint-inference-history'],
    queryFn: listDevices,
    refetchInterval: 5000,
  });
  const historyQ = useQuery({
    queryKey: ['mesh', 'joint-inference-history', peerFilter],
    queryFn: () =>
      getJointInferenceHistory({
        peer_id: peerFilter || undefined,
        limit: 50,
      }),
    refetchInterval: REFRESH_MS,
  });

  const peers = devicesQ.data?.peers ?? [];
  const items = historyQ.data?.items ?? [];
  const active = useMemo(
    () => items.filter((item) => !isTerminal(item.status)),
    [items],
  );
  const complete = useMemo(() => items.filter((item) => item.status === 'complete'), [items]);
  const failed = useMemo(
    () => items.filter((item) => item.status === 'error' || item.status === 'cancelled'),
    [items],
  );

  return (
    <div className="p-6">
      <PageHeader
        title={t('training.jointInference.title')}
        description={t('training.jointInference.description')}
        actions={
          <button
            type="button"
            onClick={() => historyQ.refetch()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200 dark:hover:bg-stone-800"
          >
            <RefreshCw size={14} className={historyQ.isFetching ? 'animate-spin' : undefined} />
            {t('common.refresh')}
          </button>
        }
      />

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <MetricCard icon={<Router size={16} />} label={t('training.jointInference.recentRequests')} value={items.length.toString()} />
        <MetricCard
          icon={<Clock size={16} />}
          label={t('training.jointInference.inFlight')}
          value={active.length.toString()}
          tone={active.length > 0 ? 'amber' : 'neutral'}
        />
        <MetricCard icon={<CheckCircle2 size={16} />} label={t('training.jointInference.completed')} value={complete.length.toString()} tone="emerald" />
      </div>

      <div className="mt-4 flex flex-col gap-2 rounded-lg border border-stone-200 bg-white px-3 py-3 dark:border-stone-800 dark:bg-stone-950 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2 text-sm text-stone-700 dark:text-stone-200">
          <BrainCircuit size={15} className="text-indigo-500" />
          {t('training.jointInference.hostAuditTrail')}
        </div>
        <label className="flex items-center gap-2 text-xs text-stone-500 dark:text-stone-400">
          {t('training.common.device')}
          <select
            value={peerFilter}
            onChange={(event) => setPeerFilter(event.target.value)}
            className="rounded-lg border border-stone-200 bg-white px-2 py-1 text-xs text-stone-700 outline-none focus:border-indigo-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
          >
            <option value="">{t('training.jointInference.allTrustedDevices')}</option>
            {peers.map((peer) => (
              <option key={peer.peer_id} value={peer.peer_id}>
                {peer.display_name} · {shortId(peer.peer_id)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {historyQ.isLoading ? (
        <div className="mt-4 rounded-lg border border-stone-200 bg-white p-8 text-center text-sm text-stone-500 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-400">
          {t('training.jointInference.loadingHistory')}
        </div>
      ) : items.length === 0 ? (
        <div className="mt-4 rounded-lg border border-stone-200 bg-white p-8 text-center dark:border-stone-800 dark:bg-stone-950">
          <Smartphone className="mx-auto text-stone-300 dark:text-stone-700" size={32} />
          <p className="mt-2 text-sm font-medium text-stone-700 dark:text-stone-200">{t('training.jointInference.emptyTitle')}</p>
          <p className="mt-1 text-xs text-stone-500 dark:text-stone-400">
            {t('training.jointInference.emptyDescription')}
          </p>
        </div>
      ) : (
        <div className="mt-4 space-y-2">
          {items.map((item) => (
            <HistoryRow
              key={item.request_id}
              item={item}
              t={t}
              onDelete={(requestId) => {
                void deleteJointInferenceHistoryItem(requestId)
                  .then(() => queryClient.invalidateQueries({ queryKey: ['mesh', 'joint-inference-history'] }))
                  .catch((err) => console.warn('[JointInferenceHistory] failed to delete history item', err));
              }}
            />
          ))}
        </div>
      )}

      {failed.length > 0 && (
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          <AlertTriangle size={14} className="mr-1 inline" />
          {t('training.jointInference.failedSummary', { count: failed.length })}
        </div>
      )}
    </div>
  );
}

function JointInferenceDetail({ requestId }: { requestId: string }) {
  const t = useT();
  const detailQ = useQuery({
    queryKey: ['mesh', 'joint-inference-history', requestId, 'detail'],
    queryFn: () => getJointInferenceHistoryItem(requestId),
  });
  const item = detailQ.data?.item;
  const [thread, setThread] = useState<ThreadMessage[]>([]);
  const [contextMessages, setContextMessages] = useState<ConversationMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!item) return;
    setThread(materializeThread(item));
    setContextMessages(materializeContextMessages(item));
    setStatus(item.status);
  }, [item]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [thread, status]);

  async function submitFollowUp(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || streaming) return;

    const baseMessages = contextMessages.map(({ role, content }) => ({ role, content }));
    const contextUserMessage: ConversationMessage = { role: 'user', content: text };
    const userMessage: ThreadMessage = { id: makeId(), role: 'user', content: text };
    const assistantId = makeId();
    const assistantMessage: ThreadMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      streaming: true,
    };

    setInput('');
    setStreaming(true);
    setStatus(t('training.jointInference.macGenerating'));
    setThread((current) => [...current, userMessage, assistantMessage]);

    try {
      await streamJointInferenceContinue(
        requestId,
        {
          messages: [...baseMessages, { role: 'user', content: text }],
          max_tokens: item?.max_tokens ?? 2048,
          temperature: item?.temperature ?? 0.2,
          enable_thinking: item?.enable_thinking ?? false,
          use_neural_imprint: item?.use_neural_imprint ?? false,
        },
        (streamEvent) => {
          if (streamEvent.type === 'status' || streamEvent.type === 'accepted') {
            setStatus(streamEvent.message || streamEvent.type);
          } else if (streamEvent.type === 'token' && streamEvent.token) {
            setThread((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + streamEvent.token }
                  : message,
              ),
            );
          } else if (streamEvent.type === 'complete') {
            const finalText = streamEvent.full_text || '';
            setStatus(t('training.jointInference.completeWithTokens', { count: streamEvent.total_tokens ?? 0 }));
            setThread((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: finalText || message.content,
                      streaming: false,
                      totalTokens: streamEvent.total_tokens,
                      tokensPerSec: streamEvent.tokens_per_sec,
                    }
                  : message,
              ),
            );
            setContextMessages((current) => [
              ...current,
              contextUserMessage,
              { role: 'assistant', content: finalText },
            ]);
          } else if (streamEvent.type === 'error' || streamEvent.type === 'cancelled') {
            setStatus(streamEvent.error || streamEvent.message || streamEvent.type);
            setThread((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? {
                      ...message,
                      content: streamEvent.error || streamEvent.message || streamEvent.type,
                      streaming: false,
                    }
                  : message,
              ),
            );
          }
        },
      );
      detailQ.refetch();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setStatus(message);
      setThread((current) =>
        current.map((item) =>
          item.id === assistantId
            ? { ...item, content: `${t('common.error')}: ${message}`, streaming: false }
            : item,
        ),
      );
    } finally {
      setStreaming(false);
    }
  }

  if (detailQ.isLoading) {
    return (
      <div className="p-6">
        <PageHeader title={t('training.jointInference.detailTitle')} description={t('training.jointInference.detailDescription')} />
        <div className="mt-4 rounded-lg border border-stone-200 bg-white p-8 text-center text-sm text-stone-500 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-400">
          {t('training.jointInference.loadingRequest')}
        </div>
      </div>
    );
  }

  if (!item) {
    return (
      <div className="p-6">
        <PageHeader title={t('training.jointInference.detailTitle')} description={t('training.jointInference.detailDescription')} />
        <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-8 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
          {t('training.jointInference.requestNotFound')}
        </div>
      </div>
    );
  }

  return (
    <div className="-m-6 flex h-[calc(100vh-var(--header-height,49px))] flex-col bg-white dark:bg-stone-950">
      <div className="shrink-0 border-b border-stone-100 bg-white px-6 py-4 dark:border-stone-800 dark:bg-stone-950">
        <PageHeader
          title={t('training.jointInference.detailTitle')}
          description={t('training.jointInference.detailDescription')}
          actions={
            <Link
              to="/joint-inference"
              className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200 dark:hover:bg-stone-800"
            >
              <ArrowLeft size={14} />
              {t('training.jointInference.back')}
            </Link>
          }
        />

        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-4">
          <MetricCard icon={<Router size={16} />} label={t('training.jointInference.route')} value={item.route_reason || '-'} />
          <MetricCard icon={<Clock size={16} />} label={t('training.jointInference.duration')} value={formatDuration(item.duration_seconds)} />
          <MetricCard icon={<BrainCircuit size={16} />} label={t('training.jointInference.tokens')} value={formatTokens(item)} tone="emerald" />
          <MetricCard icon={<Smartphone size={16} />} label={t('training.common.device')} value={shortId(item.peer_id)} />
        </div>

        <div className="mt-3 text-xs text-stone-500 dark:text-stone-400">
          {shortId(item.request_id)} · {formatTime(item.accepted_at)} · {status || item.status}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-6 py-4">
        {thread.map((message) => (
          <ChatBubble key={message.id} message={message} t={t} />
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={submitFollowUp} className="shrink-0 border-t border-stone-100 bg-white px-6 py-3 dark:border-stone-800 dark:bg-stone-950">
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if (input.trim() && !streaming) {
                  event.currentTarget.form?.requestSubmit();
                }
              }
            }}
            placeholder={t('training.jointInference.continuePlaceholder')}
            rows={1}
            disabled={streaming}
            className="max-h-40 min-h-[3rem] flex-1 resize-none rounded-2xl border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-900 outline-none focus:border-indigo-400 focus:bg-white dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100 dark:focus:border-indigo-500"
          />
          <button
            type="submit"
            disabled={!input.trim() || streaming}
            className="inline-flex h-10 items-center gap-1.5 rounded-xl bg-indigo-600 px-3 text-sm font-medium text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {streaming ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
            {t('globalAsk.send')}
          </button>
        </div>
      </form>
    </div>
  );
}

function HistoryRow({
  item,
  t,
  onDelete,
}: {
  item: JointInferenceHistoryItem;
  t: Translator;
  onDelete: (requestId: string) => void;
}) {
  const terminal = isTerminal(item.status);
  const ok = item.status === 'complete';
  return (
    <Link to={`/joint-inference/${item.request_id}`} className="block">
      <article className="rounded-lg border border-stone-200 bg-white p-4 transition hover:border-indigo-200 hover:bg-indigo-50/40 dark:border-stone-800 dark:bg-stone-950 dark:hover:border-indigo-900 dark:hover:bg-indigo-950/20">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={item.status} t={t} />
              <span className="font-mono text-xs text-stone-400">{shortId(item.request_id)}</span>
              <span className="text-xs text-stone-400">{formatTime(item.accepted_at)}</span>
            </div>
            <p className="mt-2 text-sm font-medium text-stone-900 dark:text-stone-100">
              {item.prompt_preview || t('training.jointInference.noPromptPreview')}
            </p>
            {item.output_preview && (
              <p className="mt-1 line-clamp-2 text-sm text-stone-600 dark:text-stone-300">
                {item.output_preview}
              </p>
            )}
            {item.error && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-300">{item.error}</p>
            )}
          </div>
          <dl className="grid shrink-0 grid-cols-2 gap-x-4 gap-y-1 text-xs sm:min-w-[18rem]">
            <dt className="text-stone-400">{t('training.jointInference.peer')}</dt>
            <dd className="truncate font-mono text-stone-700 dark:text-stone-200">{shortId(item.peer_id)}</dd>
            <dt className="text-stone-400">{t('training.rppResults.model')}</dt>
            <dd className="truncate text-stone-700 dark:text-stone-200">{item.model_id || '-'}</dd>
            <dt className="text-stone-400">{t('training.jointInference.duration')}</dt>
            <dd className="text-stone-700 dark:text-stone-200">{formatDuration(item.duration_seconds)}</dd>
            <dt className="text-stone-400">{t('training.jointInference.tokens')}</dt>
            <dd className="text-stone-700 dark:text-stone-200">
              {typeof item.total_tokens === 'number'
                ? `${item.total_tokens}${typeof item.tokens_per_sec === 'number' ? ` · ${item.tokens_per_sec.toFixed(1)}/s` : ''}`
                : terminal ? '-' : t('training.jointInference.streamedTokens', { count: item.token_events ?? 0 })}
            </dd>
            <dt className="text-stone-400">{t('training.jointInference.route')}</dt>
            <dd className="truncate text-stone-700 dark:text-stone-200">{item.route_reason || '-'}</dd>
            <dt className="text-stone-400">{t('nav.neuralImprint')}</dt>
            <dd className="text-stone-700 dark:text-stone-200">
              {item.use_neural_imprint ? t('training.common.yes') : t('training.common.no')}
            </dd>
            <dt className="text-stone-400">{t('training.jointInference.think')}</dt>
            <dd className="text-stone-700 dark:text-stone-200">{item.enable_thinking ? t('training.common.on') : t('training.common.off')}</dd>
          </dl>
          <button
            type="button"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              if (window.confirm(t('training.jointInference.deleteRequestConfirm'))) {
                onDelete(item.request_id);
              }
            }}
            className="rounded-lg p-2 text-stone-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/30 dark:hover:text-red-300"
            title={t('training.jointInference.deleteRequest')}
          >
            <Trash2 size={15} />
          </button>
        </div>
        {!ok && terminal && (
          <div className="mt-3 flex items-center gap-1.5 text-xs text-red-600 dark:text-red-300">
            <XCircle size={13} />
            {t('training.jointInference.terminalStatus', { status: item.status })}
          </div>
        )}
      </article>
    </Link>
  );
}

function ChatBubble({ message, t }: { message: ThreadMessage; t: Translator }) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const align = isUser ? 'justify-end' : 'justify-start';
  const columnAlign = isUser ? 'items-end' : 'items-start';
  const bubble = isUser
    ? 'bg-indigo-600 text-white'
    : isSystem
      ? 'border border-stone-200 bg-stone-50 text-stone-600 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-300'
      : 'border border-stone-200 bg-white text-stone-900 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-100';
  return (
    <div className={`flex ${align}`}>
      <div className={`flex max-w-[min(48rem,90%)] flex-col gap-2 ${columnAlign}`}>
        <div className={`w-fit max-w-full rounded-lg px-3 py-2 text-sm ${bubble}`}>
          <div className="mb-1 text-[10px] font-semibold uppercase opacity-60">{roleLabel(message.role, t)}</div>
          {isUser || isSystem ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <MarkdownContent content={message.content || (message.streaming ? ' ' : '')} />
          )}
          {message.streaming && (
            <span className="mt-1 inline-block h-3 w-1.5 animate-pulse rounded-sm bg-indigo-400" />
          )}
          {!message.streaming && message.totalTokens != null && (
            <div className="mt-2 text-[10px] opacity-60">
              {t('training.common.tokens', { count: message.totalTokens })}
              {message.tokensPerSec != null ? ` · ${message.tokensPerSec.toFixed(1)}/s` : ''}
            </div>
          )}
        </div>
        {message.diagnostics?.length ? (
          <DeveloperDiagnostics messages={message.diagnostics} t={t} />
        ) : null}
      </div>
    </div>
  );
}

function DeveloperDiagnostics({ messages, t }: { messages: ThreadMessage[]; t: Translator }) {
  return (
    <details className="w-full rounded-lg border border-stone-200 bg-stone-50/80 text-sm dark:border-stone-800 dark:bg-stone-900/60">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2 text-xs font-medium text-stone-600 dark:text-stone-300">
        <span className="inline-flex items-center gap-2">
          <Wrench size={14} />
          {t('training.jointInference.developerDiagnostics')}
        </span>
        <span className="rounded-full bg-stone-200 px-2 py-0.5 font-mono text-[10px] text-stone-600 dark:bg-stone-800 dark:text-stone-300">
          {messages.length}
        </span>
      </summary>
      <div className="border-t border-stone-200 px-3 py-3 dark:border-stone-800">
        <p className="mb-3 text-xs text-stone-500 dark:text-stone-400">
          {t('training.jointInference.developerDiagnosticsDescription')}
        </p>
        <div className="space-y-2">
          {messages.map((message) => (
            <div key={message.id} className="rounded-md border border-stone-200 bg-white p-2 dark:border-stone-800 dark:bg-stone-950">
              <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-stone-400">
                {diagnosticRoleLabel(message, t)}
              </div>
              <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-stone-700 dark:text-stone-200">
                {message.content}
              </pre>
            </div>
          ))}
        </div>
      </div>
    </details>
  );
}

function MetricCard({
  icon,
  label,
  value,
  tone = 'neutral',
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  tone?: 'neutral' | 'emerald' | 'amber';
}) {
  const toneClass = {
    neutral: 'bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-300',
    emerald: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300',
    amber: 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300',
  }[tone];
  return (
    <div className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
      <div className="flex items-center gap-2">
        <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${toneClass}`}>{icon}</span>
        <div className="min-w-0">
          <p className="text-xs text-stone-500 dark:text-stone-400">{label}</p>
          <p className="truncate text-lg font-semibold text-stone-900 dark:text-stone-100">{value}</p>
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status, t }: { status: string; t: Translator }) {
  const ok = status === 'complete';
  const bad = status === 'error' || status === 'cancelled';
  const cls = ok
    ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300'
    : bad
      ? 'bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300'
      : 'bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300';
  return (
    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ${cls}`}>
      {statusLabel(status, t)}
    </span>
  );
}

function materializeThread(item: JointInferenceHistoryItem): ThreadMessage[] {
  const messages: ThreadMessage[] = [];
  let pendingDiagnostics: ThreadMessage[] = [];

  for (const [index, message] of (item.messages ?? []).entries()) {
    const threadMessage: ThreadMessage = {
      id: `${item.request_id}-${index}`,
      role: message.role,
      content: message.content,
    };
    if (isDeveloperDiagnosticMessage(threadMessage)) {
      pendingDiagnostics = [...pendingDiagnostics, threadMessage];
      continue;
    }

    if (threadMessage.role === 'assistant' && pendingDiagnostics.length > 0) {
      threadMessage.diagnostics = pendingDiagnostics;
      pendingDiagnostics = [];
    }
    messages.push(threadMessage);
  }

  const shouldAppendFullText =
    Boolean(item.full_text) && (messages.length === 0 || messages[messages.length - 1].role !== 'assistant');

  if (pendingDiagnostics.length > 0 && !shouldAppendFullText) {
    const lastAssistant = [...messages].reverse().find((message) => message.role === 'assistant');
    if (lastAssistant) {
      lastAssistant.diagnostics = [...(lastAssistant.diagnostics ?? []), ...pendingDiagnostics];
      pendingDiagnostics = [];
    }
  }

  if (shouldAppendFullText) {
    messages.push({
      id: `${item.request_id}-assistant`,
      role: 'assistant',
      content: item.full_text ?? '',
      totalTokens: item.total_tokens ?? undefined,
      tokensPerSec: item.tokens_per_sec ?? undefined,
      diagnostics: pendingDiagnostics.length > 0 ? pendingDiagnostics : undefined,
    });
  }
  return messages;
}

function materializeContextMessages(item?: JointInferenceHistoryItem): ConversationMessage[] {
  return (item?.messages ?? [])
    .filter((message) => typeof message.content === 'string' && typeof message.role === 'string')
    .map((message) => ({ role: message.role, content: message.content }));
}

function isDeveloperDiagnosticMessage(message: { role: string; content: string }) {
  if (message.role === 'system' || message.role === 'tool') return true;
  return message.role === 'assistant' && looksLikeToolCall(message.content);
}

function looksLikeToolCall(content: string) {
  const text = content.trim();
  return text.includes('<tool_call') || text.includes('<function=') || text.startsWith('tool_call:');
}

function isTerminal(status: string) {
  return status === 'complete' || status === 'error' || status === 'cancelled';
}

function shortId(id?: string | null) {
  if (!id) return '-';
  return id.length > 18 ? `${id.slice(0, 14)}...` : id;
}

function formatTokens(item: JointInferenceHistoryItem) {
  if (typeof item.total_tokens !== 'number') return '-';
  return `${item.total_tokens}${typeof item.tokens_per_sec === 'number' ? ` · ${item.tokens_per_sec.toFixed(1)}/s` : ''}`;
}

function formatTime(seconds?: number | null) {
  if (!seconds) return '-';
  return new Date(seconds * 1000).toLocaleTimeString();
}

function formatDuration(seconds?: number | null) {
  if (typeof seconds !== 'number') return '-';
  return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
}

function makeId() {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function statusLabel(status: string, t: Translator): string {
  if (status === 'complete') return t('training.common.completeStatus');
  if (status === 'error') return t('common.error');
  if (status === 'cancelled') return t('common.cancelled');
  return status;
}

function roleLabel(role: string, t: Translator): string {
  if (role === 'user') return t('training.common.user');
  if (role === 'assistant') return t('training.common.assistant');
  if (role === 'system') return t('training.common.system');
  if (role === 'tool') return t('training.jointInference.toolRole');
  return role;
}

function diagnosticRoleLabel(message: ThreadMessage, t: Translator): string {
  if (message.role === 'tool') return t('training.jointInference.toolResult');
  if (message.role === 'assistant' && looksLikeToolCall(message.content)) {
    return t('training.jointInference.toolCall');
  }
  return roleLabel(message.role, t);
}
