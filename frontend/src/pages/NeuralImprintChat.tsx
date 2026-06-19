// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Bot,
  CheckCircle2,
  Loader2,
  MessageCircle,
  RefreshCw,
  Send,
  ShieldCheck,
  Square,
  User,
  XCircle,
} from 'lucide-react';
import {
  getNeuralImprintRuntimeStatus,
  listLoadedModels,
  restoreNeuralImprint,
  unloadNeuralImprint,
} from '@/api/endpoints';
import {
  generateNeuralImprint,
  getLatestPersonaSource,
  getNeuralImprintGenerationJob,
  listDevices,
  listNeuralImprintArtifacts,
  type NeuralImprintGenerationJob,
  type NeuralImprintArtifactSource,
  type PersonaSourceLatestResponse,
  type TrustedPeer,
} from '@/api/mesh';
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  replaceConversationMessages,
  updateConversation,
  type ConversationMessage,
} from '@/api/conversations';
import { buildNeuralImprintChatWsUrl, createReconnectingWebSocket, type ReconnectingWebSocketHandle } from '@/api/websocket';
import type { ModelInfo, NeuralImprintRuntimeStatusResponse } from '@/api/types';
import { ConversationSessionMenu } from '@/components/common/ConversationSessionMenu';
import MarkdownContent from '@/components/MarkdownContent';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT } from '@/i18n';
import { cn, formatSize } from '@/lib/utils';

interface PreviewMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  streaming?: boolean;
  totalTokens?: number;
  tokensPerSec?: number;
  totalTime?: number;
}

interface StreamEvent {
  type: 'token' | 'status' | 'complete' | 'error' | 'cancelled';
  token?: string;
  message?: string;
  full_text?: string;
  total_tokens?: number;
  tokens_per_sec?: number;
  total_time?: number;
}

const NEURAL_IMPRINT_CHAT_SURFACE = 'neural_imprint_chat';

export default function NeuralImprintChat() {
  const t = useT();
  const queryClient = useQueryClient();
  const [selectedModelId, setSelectedModelId] = useState('');
  const [selectedArtifactKey, setSelectedArtifactKey] = useState('');
  const [generationPeerId, setGenerationPeerId] = useState('');
  const [generationJobId, setGenerationJobId] = useState<string | null>(null);
  const [messages, setMessages] = useState<PreviewMessage[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [runtimeStatus, setRuntimeStatus] = useState<NeuralImprintRuntimeStatusResponse | null>(null);
  const [loadingPersona, setLoadingPersona] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [streamStatus, setStreamStatus] = useState('');
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<ReconnectingWebSocketHandle | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const modelsQ = useQuery({
    queryKey: ['neural-imprint-chat', 'loaded-models'],
    queryFn: listLoadedModels,
    refetchInterval: 5000,
  });
  const artifactsQ = useQuery({
    queryKey: ['neural-imprint-chat', 'artifacts'],
    queryFn: () => listNeuralImprintArtifacts(false),
  });
  const devicesQ = useQuery({
    queryKey: ['neural-imprint-chat', 'devices'],
    queryFn: listDevices,
    refetchInterval: 10000,
  });
  const statusQ = useQuery({
    queryKey: ['neural-imprint-chat', 'status', selectedModelId],
    queryFn: () => getNeuralImprintRuntimeStatus(selectedModelId || undefined),
    enabled: !!selectedModelId,
    refetchInterval: 3000,
  });
  const conversationsQ = useQuery({
    queryKey: ['conversations', NEURAL_IMPRINT_CHAT_SURFACE],
    queryFn: () => listConversations({ surface: NEURAL_IMPRINT_CHAT_SURFACE, limit: 50 }),
    refetchInterval: 5000,
  });

  const eligibleModels = useMemo(
    () => (modelsQ.data ?? []).filter((model) => model.model_category === 'llm' || model.model_category === 'vlm'),
    [modelsQ.data],
  );
  const selectedModel = useMemo(
    () => eligibleModels.find((model) => model.model_id === selectedModelId) ?? null,
    [eligibleModels, selectedModelId],
  );
  const artifacts = useMemo(
    () => (artifactsQ.data?.artifacts ?? []).filter((item) => item.valid && item.artifact_path),
    [artifactsQ.data?.artifacts],
  );
  const activePeers = useMemo(
    () => (devicesQ.data?.peers ?? []).filter((peer) => !peer.revoked),
    [devicesQ.data?.peers],
  );
  const matchingArtifacts = useMemo(() => {
    if (!selectedModel) return artifacts;
    const names = new Set([
      selectedModel.model_name,
      selectedModel.model_dir.split('/').pop() ?? '',
      selectedModel.model_id,
    ].filter(Boolean));
    const matches = artifacts.filter((item) => item.base_model_id && names.has(item.base_model_id));
    return matches.length > 0 ? matches : artifacts;
  }, [artifacts, selectedModel]);
  const selectedArtifact = useMemo(
    () => matchingArtifacts.find((item) => artifactKey(item) === selectedArtifactKey) ?? null,
    [matchingArtifacts, selectedArtifactKey],
  );
  const isActiveForSelectedModel = !!(
    selectedModel
    && runtimeStatus?.active
    && runtimeStatus.model_id === selectedModel.model_id
  );
  const conversationItems = conversationsQ.data?.items ?? [];

  const sourceQ = useQuery({
    queryKey: ['neural-imprint-chat', 'source-latest', generationPeerId],
    queryFn: () => getLatestPersonaSource(generationPeerId),
    enabled: !!generationPeerId,
    retry: false,
    refetchInterval: 10000,
  });
  const source = sourceQ.data ?? null;
  const sourceModelId = source?.receipt.base_model_id ?? null;
  const generationModelMatchesSource = !sourceModelId || modelsMatch(selectedModel, sourceModelId);
  const generateMut = useMutation({
    mutationFn: () => {
      if (!selectedModel || !generationPeerId || !source) {
        throw new Error('missing device source or host model');
      }
      return generateNeuralImprint({
        peer_id: generationPeerId,
        model_dir: selectedModel.model_dir,
        model_id: sourceModelId ?? selectedModel.model_id,
        validate_restore: false,
      });
    },
    onSuccess: (response) => {
      setGenerationJobId(response.job.job_id);
      void queryClient.invalidateQueries({ queryKey: ['neural-imprint-chat', 'artifacts'] });
    },
  });
  const generationJobQ = useQuery({
    queryKey: ['neural-imprint-chat', 'generation-job', generationJobId],
    queryFn: () => getNeuralImprintGenerationJob(generationJobId as string),
    enabled: !!generationJobId,
    retry: false,
    refetchInterval: (query) => {
      const status = query.state.data?.job.status;
      return status === 'succeeded' || status === 'failed' ? false : 2000;
    },
  });
  const generationJob = generationJobQ.data?.job ?? generateMut.data?.job ?? null;
  const generationRunning =
    generateMut.isPending ||
    generationJob?.status === 'queued' ||
    generationJob?.status === 'running';
  const canGenerateArtifact =
    !!selectedModel &&
    !!generationPeerId &&
    !!source?.receipt.profile_body_sha256 &&
    generationModelMatchesSource &&
    !generationRunning;
  const generationStatusText = neuralImprintGenerationStatus({
    activePeers,
    source,
    sourceLoading: sourceQ.isLoading,
    sourceError: sourceQ.isError,
    selectedModel,
    sourceModelId,
    modelMatchesSource: generationModelMatchesSource,
    job: generationJob,
    mutationError: generateMut.error,
    t,
  });

  useEffect(() => {
    if (!selectedModelId && eligibleModels.length > 0) {
      setSelectedModelId(eligibleModels[0].model_id);
    }
  }, [eligibleModels, selectedModelId]);

  useEffect(() => {
    if (!generationPeerId && activePeers[0]?.peer_id) {
      setGenerationPeerId(activePeers[0].peer_id);
    }
  }, [activePeers, generationPeerId]);

  useEffect(() => {
    if (matchingArtifacts.length === 0) {
      setSelectedArtifactKey('');
      return;
    }
    if (!selectedArtifactKey || !matchingArtifacts.some((item) => artifactKey(item) === selectedArtifactKey)) {
      setSelectedArtifactKey(artifactKey(matchingArtifacts[0]));
    }
  }, [matchingArtifacts, selectedArtifactKey]);

  useEffect(() => {
    if (statusQ.data) setRuntimeStatus(statusQ.data);
  }, [statusQ.data]);

  useEffect(() => {
    if (generationJob?.status !== 'succeeded') return;
    void queryClient.invalidateQueries({ queryKey: ['neural-imprint-chat', 'artifacts'] });
  }, [generationJob?.status, queryClient]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamStatus]);

  useEffect(() => {
    return () => wsRef.current?.close();
  }, []);

  const persistConversation = useCallback((
    sessionId: string,
    nextMessages: PreviewMessage[],
    status: 'active' | 'complete' | 'error' | 'cancelled' = 'active',
  ) => {
    void replaceConversationMessages(sessionId, serializePreviewMessages(nextMessages))
      .then(() => updateConversation(sessionId, { status }))
      .then(() => queryClient.invalidateQueries({ queryKey: ['conversations', NEURAL_IMPRINT_CHAT_SURFACE] }))
      .catch((err) => console.warn('[NeuralImprintChat] failed to persist conversation', err));
  }, [queryClient]);

  const ensureConversationSession = useCallback(async (firstText: string): Promise<string | null> => {
    if (!selectedModel) return null;
    if (currentSessionId) return currentSessionId;
    try {
      const session = await createConversation({
        surface: NEURAL_IMPRINT_CHAT_SURFACE,
        title: titleFromText(firstText, t),
        model_id: selectedModel.model_id,
        source: 'edgestudio_web',
        status: 'active',
        metadata: {
          neural_imprint_active: isActiveForSelectedModel,
          artifact_key: selectedArtifactKey || null,
          artifact_id: selectedArtifact?.artifact_id || null,
        },
      });
      setCurrentSessionId(session.session_id);
      void queryClient.invalidateQueries({ queryKey: ['conversations', NEURAL_IMPRINT_CHAT_SURFACE] });
      return session.session_id;
    } catch (err) {
      console.warn('[NeuralImprintChat] failed to create conversation session', err);
      return null;
    }
  }, [
    currentSessionId,
    isActiveForSelectedModel,
    queryClient,
    selectedArtifact,
    selectedArtifactKey,
    selectedModel,
    t,
  ]);

  const loadConversationSession = useCallback(async (sessionId: string) => {
    if (!sessionId || streaming) return;
    try {
      const session = await getConversation(sessionId);
      setCurrentSessionId(session.session_id);
      setMessages((session.messages ?? []).map(previewMessageFromStoredMessage));
      setStreamStatus('');
      wsRef.current?.close();
      wsRef.current = null;
    } catch (err) {
      console.warn('[NeuralImprintChat] failed to load conversation session', err);
    }
  }, [streaming]);

  const handleDeleteConversationSession = useCallback((sessionId: string) => {
    if (sessionId === currentSessionId) {
      setMessages([]);
      setCurrentSessionId(null);
      setStreamStatus('');
      wsRef.current?.close();
      wsRef.current = null;
    }
    void deleteConversation(sessionId)
      .then(() => queryClient.invalidateQueries({ queryKey: ['conversations', NEURAL_IMPRINT_CHAT_SURFACE] }))
      .catch((err) => console.warn('[NeuralImprintChat] failed to delete conversation', err));
  }, [currentSessionId, queryClient]);

  const connect = useCallback((modelId: string) => {
    if (wsRef.current && wsRef.current.readyState() === WebSocket.OPEN) return wsRef.current;
    wsRef.current?.close();
    const handle = createReconnectingWebSocket(buildNeuralImprintChatWsUrl(modelId), {
      onClose: () => {
        if (wsRef.current === handle) wsRef.current = null;
      },
    });
    wsRef.current = handle;
    return handle;
  }, []);

  const loadNeuralImprint = async () => {
    if (!selectedModel || !selectedArtifact) return;
    setLoadingPersona(true);
    setError(null);
    try {
      const status = await restoreNeuralImprint({
        model_id: selectedModel.model_id,
        artifact_id: selectedArtifact.artifact_id || undefined,
        artifact_path: selectedArtifact.artifact_id ? undefined : selectedArtifact.artifact_path,
        sidecar_path: selectedArtifact.artifact_id ? undefined : selectedArtifact.sidecar_path,
      });
      setRuntimeStatus(status);
      setMessages([]);
      setCurrentSessionId(null);
      wsRef.current?.close();
      wsRef.current = null;
    } catch (err) {
      setError(errorMessage(err, t('training.neuralImprintChat.errorLoad')));
    } finally {
      setLoadingPersona(false);
    }
  };

  const unloadActiveNeuralImprint = async () => {
    if (!selectedModel) return;
    setLoadingPersona(true);
    setError(null);
    try {
      const status = await unloadNeuralImprint(selectedModel.model_id);
      setRuntimeStatus(status);
      wsRef.current?.close();
      wsRef.current = null;
    } catch (err) {
      setError(errorMessage(err, t('training.neuralImprintChat.errorUnload')));
    } finally {
      setLoadingPersona(false);
    }
  };

  const sendMessage = useCallback(() => {
    if (!selectedModel || !input.trim() || streaming) return;
    const text = input.trim();
    const assistantId = `assistant-${Date.now()}`;
    const userMessage: PreviewMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
    };
    const assistantMessage: PreviewMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      streaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput('');
    setStreaming(true);
    setStreamStatus('');
    setError(null);

    const history = messages
      .filter((message) => !message.streaming)
      .map((message) => ({ role: message.role, content: message.content }));

    void (async () => {
    const sessionId = await ensureConversationSession(text);
    const handle = connect(selectedModel.model_id);
    handle.setOnMessage((event) => {
      const data = JSON.parse(event.data) as StreamEvent;
      if (data.type === 'status') {
        setStreamStatus(data.message || '');
      } else if (data.type === 'token') {
        setStreamStatus('');
        setMessages((prev) => prev.map((message) =>
          message.id === assistantId
            ? { ...message, content: message.content + (data.token || '') }
            : message,
        ));
      } else if (data.type === 'complete') {
        setMessages((prev) => {
          const next = prev.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: data.full_text || message.content,
                  streaming: false,
                  totalTokens: data.total_tokens,
                  tokensPerSec: data.tokens_per_sec,
                  totalTime: data.total_time,
                }
              : message,
          );
          if (sessionId) persistConversation(sessionId, next, 'complete');
          return next;
        });
        setStreaming(false);
        setStreamStatus('');
      } else if (data.type === 'error') {
        setMessages((prev) => {
          const next = prev.map((message) =>
            message.id === assistantId
              ? { ...message, content: data.message || t('common.error'), streaming: false }
              : message,
          );
          if (sessionId) persistConversation(sessionId, next, 'error');
          return next;
        });
        setStreaming(false);
        setStreamStatus('');
      } else if (data.type === 'cancelled') {
        setMessages((prev) => {
          const next = prev.map((message) =>
            message.id === assistantId ? { ...message, streaming: false } : message,
          );
          if (sessionId) persistConversation(sessionId, next, 'cancelled');
          return next;
        });
        setStreaming(false);
        setStreamStatus('');
      }
    });
    handle.send(JSON.stringify({
      prompt: text,
      history,
      max_tokens: 16384,
      temperature: 0.4,
      top_k: 50,
      top_p: 0.9,
      enable_thinking: false,
    }));
    })();
  }, [connect, ensureConversationSession, input, messages, persistConversation, selectedModel, streaming, t]);

  const cancelGeneration = () => {
    wsRef.current?.send(JSON.stringify({ type: 'cancel' }));
  };

  const onInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-7rem)] flex-col gap-4 pb-8">
      <PageHeader
        title={t('training.neuralImprintChat.title')}
        description={t('training.neuralImprintChat.description')}
        actions={(
          <button
            type="button"
            onClick={() => {
              void modelsQ.refetch();
              void artifactsQ.refetch();
              void statusQ.refetch();
            }}
            className="inline-flex items-center gap-2 rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm font-medium text-stone-700 hover:bg-stone-50 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-200 dark:hover:bg-stone-900"
          >
            <RefreshCw size={15} className={modelsQ.isFetching || artifactsQ.isFetching ? 'animate-spin' : ''} />
            {t('common.refresh')}
          </button>
        )}
      />

      <section className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(220px,0.8fr)_minmax(260px,1.2fr)_auto]">
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-stone-500">
              {t('training.neuralImprintChat.baseModel')}
            </span>
            <select
              value={selectedModelId}
              onChange={(event) => {
                setSelectedModelId(event.target.value);
                setMessages([]);
                wsRef.current?.close();
                wsRef.current = null;
              }}
              className="h-10 w-full rounded-lg border border-stone-200 bg-white px-3 text-sm text-stone-900 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
            >
              {eligibleModels.length === 0 ? (
                <option value="">{t('training.neuralImprintChat.noLoadedModels')}</option>
              ) : eligibleModels.map((model) => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_dir.split('/').pop()} · {model.hidden_size}h · {model.num_layers}L
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-stone-500">
              {t('training.neuralImprintChat.artifact')}
            </span>
            <select
              value={selectedArtifactKey}
              onChange={(event) => setSelectedArtifactKey(event.target.value)}
              className="h-10 w-full rounded-lg border border-stone-200 bg-white px-3 text-sm text-stone-900 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
            >
              {matchingArtifacts.length === 0 ? (
                <option value="">{t('training.neuralImprintChat.noArtifacts')}</option>
              ) : matchingArtifacts.map((artifact) => (
                <option key={artifactKey(artifact)} value={artifactKey(artifact)}>
                  {artifact.artifact_id ?? shortPath(artifact.artifact_path)} · {artifact.base_model_id ?? t('training.common.unknown')} · {artifact.prefix_token_count ?? '?'} {t('training.common.tokensUnit')}
                </option>
              ))}
            </select>
          </label>

          <div className="flex items-end gap-2">
            <button
              type="button"
              onClick={loadNeuralImprint}
              disabled={!selectedModel || !selectedArtifact || loadingPersona}
              className="inline-flex h-10 items-center gap-2 rounded-lg bg-stone-900 px-4 text-sm font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
            >
              {loadingPersona ? <Loader2 size={15} className="animate-spin" /> : <ShieldCheck size={15} />}
              {t('training.neuralImprintChat.load')}
            </button>
            <button
              type="button"
              onClick={unloadActiveNeuralImprint}
              disabled={!selectedModel || loadingPersona || !isActiveForSelectedModel}
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-stone-200 px-3 text-sm font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-200 dark:hover:bg-stone-900"
            >
              <XCircle size={15} />
              {t('training.neuralImprintChat.unload')}
            </button>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
          <StatusTile
            label={t('training.neuralImprintChat.runtime')}
            value={isActiveForSelectedModel ? t('training.neuralImprintChat.active') : t('training.neuralImprintChat.baseOnly')}
            active={isActiveForSelectedModel}
          />
          <StatusTile
            label={t('training.neuralImprintChat.prefix')}
            value={runtimeStatus?.prefix_token_count != null ? t('training.common.tokens', { count: runtimeStatus.prefix_token_count }) : t('training.common.none')}
          />
          <StatusTile
            label={t('training.neuralImprintChat.artifactStatus')}
            value={selectedArtifact ? `${selectedArtifact.total_bytes ? formatSize(selectedArtifact.total_bytes) : t('training.common.local')} · ${selectedArtifact.base_model_id ?? t('training.common.unknown')}` : t('training.common.none')}
          />
        </div>

        {matchingArtifacts.length === 0 && (
          <div className="mt-4 border-t border-stone-200 pt-4 dark:border-stone-800">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
              <div className="min-w-0">
                <div className="text-sm font-medium text-stone-900 dark:text-stone-100">
                  {t('training.neuralImprintChat.generateTitle')}
                </div>
                <p className="mt-1 max-w-3xl text-sm text-stone-500 dark:text-stone-400">
                  {t('training.neuralImprintChat.generateDescription')}
                </p>
                <div className={cn(
                  'mt-2 truncate text-xs',
                  generationJob?.status === 'succeeded'
                    ? 'text-emerald-700 dark:text-emerald-300'
                    : generationJob?.status === 'failed' || generateMut.isError
                      ? 'text-red-700 dark:text-red-300'
                      : 'text-stone-500 dark:text-stone-400',
                )}>
                  {generationStatusText}
                </div>
              </div>

              <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
                <label className="block min-w-[220px] text-xs font-medium text-stone-700 dark:text-stone-200">
                  {t('training.neuralImprintChat.device')}
                  <select
                    value={generationPeerId}
                    disabled={activePeers.length === 0 || generationRunning}
                    onChange={(event) => {
                      setGenerationPeerId(event.target.value);
                      setGenerationJobId(null);
                      generateMut.reset();
                    }}
                    className="mt-1 block h-9 w-full rounded-lg border border-stone-200 bg-white px-2 text-xs text-stone-900 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
                  >
                    {activePeers.length === 0 ? (
                      <option value="">{t('training.neuralImprintChat.noPairedDevices')}</option>
                    ) : activePeers.map((peer) => (
                      <option key={peer.peer_id} value={peer.peer_id}>
                        {peerLabel(peer)} · {shortPeer(peer.peer_id)}
                      </option>
                    ))}
                  </select>
                </label>

                <button
                  type="button"
                  onClick={() => generateMut.mutate()}
                  disabled={!canGenerateArtifact}
                  title={canGenerateArtifact ? t('training.neuralImprintChat.generateButtonTitle') : generationStatusText}
                  className="inline-flex h-9 items-center justify-center gap-2 rounded-lg bg-stone-900 px-3 text-xs font-medium text-white hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
                >
                  {generationRunning ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
                  {generationRunning ? t('training.neuralImprintChat.generating') : t('training.neuralImprintChat.generate')}
                </button>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/20 dark:text-red-300">
            {error}
          </div>
        )}
      </section>

      <section className="flex min-h-[480px] flex-1 flex-col rounded-lg border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-950">
        <div className="flex items-center justify-between border-b border-stone-200 px-4 py-3 dark:border-stone-800">
          <div className="flex items-center gap-2">
            <MessageCircle size={16} className="text-stone-500" />
            <span className="text-sm font-medium text-stone-900 dark:text-stone-100">{t('training.neuralImprintChat.previewSession')}</span>
          </div>
          <div className="flex items-center gap-2">
            <ConversationSessionMenu
              sessions={conversationItems}
              currentSessionId={currentSessionId}
              disabled={streaming}
              labels={{
                title: t('training.neuralImprintChat.savedSessions'),
                newSession: t('training.neuralImprintChat.newSession'),
                untitled: t('training.common.untitled'),
                deleteSession: t('training.neuralImprintChat.deleteSession'),
                deleteSessionConfirm: t('training.neuralImprintChat.deleteSessionConfirm'),
              }}
              onNewSession={() => {
                setCurrentSessionId(null);
                setMessages([]);
              }}
              onSelectSession={(sessionId) => { void loadConversationSession(sessionId); }}
              onDeleteSession={handleDeleteConversationSession}
              formatTime={formatSessionTime}
            />
            {messages.length > 0 && (
              <button
                type="button"
                onClick={() => {
                  const sessionId = currentSessionId;
                  setMessages([]);
                  setCurrentSessionId(null);
                  if (sessionId) {
                    void deleteConversation(sessionId)
                      .then(() => queryClient.invalidateQueries({ queryKey: ['conversations', NEURAL_IMPRINT_CHAT_SURFACE] }))
                      .catch((err) => console.warn('[NeuralImprintChat] failed to delete conversation', err));
                  }
                }}
                className="rounded-lg px-2 py-1 text-xs text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:text-stone-500 dark:hover:bg-stone-900 dark:hover:text-stone-300"
              >
                {t('training.neuralImprintChat.clear')}
              </button>
            )}
            {streamStatus && (
              <span className="rounded-md bg-stone-100 px-2 py-1 text-xs text-stone-600 dark:bg-stone-900 dark:text-stone-300">
                {streamStatus}
              </span>
            )}
          </div>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="flex h-full min-h-[320px] items-center justify-center">
              <div className="text-center text-sm text-stone-500 dark:text-stone-400">
                {isActiveForSelectedModel ? t('training.neuralImprintChat.emptyActive') : t('training.neuralImprintChat.emptyInactive')}
              </div>
            </div>
          ) : messages.map((message) => (
            <ChatBubble key={message.id} message={message} t={t} />
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-stone-200 p-4 dark:border-stone-800">
          <div className="flex items-end gap-3">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onInputKeyDown}
              disabled={!selectedModel || streaming}
              rows={2}
              placeholder={isActiveForSelectedModel ? t('training.neuralImprintChat.placeholderActive') : t('training.neuralImprintChat.placeholderInactive')}
              className="min-h-[48px] flex-1 resize-none rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:bg-stone-50 disabled:text-stone-400 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950 dark:disabled:bg-stone-900"
            />
            {streaming ? (
              <button
                type="button"
                onClick={cancelGeneration}
                className="flex h-10 w-10 items-center justify-center rounded-lg border border-stone-200 text-stone-700 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-200 dark:hover:bg-stone-900"
                title={t('globalAsk.stop')}
              >
                <Square size={16} />
              </button>
            ) : (
              <button
                type="button"
                onClick={sendMessage}
                disabled={!selectedModel || !input.trim()}
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
                title={t('globalAsk.send')}
              >
                <Send size={16} />
              </button>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function StatusTile({ label, value, active = false }: { label: string; value: string; active?: boolean }) {
  return (
    <div className="rounded-lg border border-stone-200 px-3 py-2 dark:border-stone-800">
      <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-stone-500">{label}</div>
      <div className={cn(
        'flex items-center gap-1.5 truncate text-sm font-medium',
        active ? 'text-emerald-700 dark:text-emerald-300' : 'text-stone-800 dark:text-stone-100',
      )}>
        {active ? <CheckCircle2 size={14} /> : null}
        <span className="truncate">{value}</span>
      </div>
    </div>
  );
}

function ChatBubble({ message, t }: { message: PreviewMessage; t: ReturnType<typeof useT> }) {
  const isUser = message.role === 'user';
  return (
    <div className={cn('flex gap-3', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900">
          <Bot size={14} />
        </div>
      )}
      <div className={cn(
        'max-w-[min(780px,85%)] rounded-lg px-3 py-2 text-sm leading-6',
        isUser
          ? 'bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
          : 'border border-stone-200 bg-stone-50 text-stone-900 dark:border-stone-800 dark:bg-stone-900/60 dark:text-stone-100',
      )}>
        {message.content ? (
          isUser ? <div className="whitespace-pre-wrap">{message.content}</div> : <MarkdownContent content={message.content} />
        ) : (
          <span className="inline-flex items-center gap-2 text-stone-500">
            <Loader2 size={14} className="animate-spin" />
            {t('training.neuralImprintChat.generating')}
          </span>
        )}
        {!isUser && !message.streaming && message.totalTokens != null && (
          <div className="mt-2 border-t border-stone-200 pt-1 text-[11px] text-stone-500 dark:border-stone-800">
            ↑ {t('training.common.tokens', { count: message.totalTokens })}
            {message.tokensPerSec != null ? ` · ${message.tokensPerSec} tok/s` : ''}
            {message.totalTime != null ? ` · ${message.totalTime}s` : ''}
          </div>
        )}
      </div>
      {isUser && (
        <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-stone-200 bg-white text-stone-700 dark:border-stone-800 dark:bg-stone-950 dark:text-stone-200">
          <User size={14} />
        </div>
      )}
    </div>
  );
}

function serializePreviewMessages(messages: PreviewMessage[]) {
  return messages
    .filter((message) => !message.streaming)
    .map((message, index) => ({
      id: message.id,
      sequence: index,
      role: message.role,
      content: message.content,
      metadata: {
        totalTokens: message.totalTokens,
        tokensPerSec: message.tokensPerSec,
        totalTime: message.totalTime,
      },
    }));
}

function previewMessageFromStoredMessage(message: ConversationMessage): PreviewMessage {
  const metadata = message.metadata ?? {};
  return {
    id: message.message_id,
    role: message.role === 'assistant' ? 'assistant' : 'user',
    content: message.content,
    totalTokens: typeof metadata.totalTokens === 'number' ? metadata.totalTokens : undefined,
    tokensPerSec: typeof metadata.tokensPerSec === 'number' ? metadata.tokensPerSec : undefined,
    totalTime: typeof metadata.totalTime === 'number' ? metadata.totalTime : undefined,
  };
}

function titleFromText(text: string, t: ReturnType<typeof useT>): string {
  const title = text.replace(/\s+/g, ' ').trim();
  if (!title) return t('training.common.untitled');
  return title.length > 48 ? `${title.slice(0, 48)}...` : title;
}

function formatSessionTime(seconds?: number | null): string {
  if (!seconds) return '-';
  return new Date(seconds * 1000).toLocaleTimeString();
}

function neuralImprintGenerationStatus({
  activePeers,
  source,
  sourceLoading,
  sourceError,
  selectedModel,
  sourceModelId,
  modelMatchesSource,
  job,
  mutationError,
  t,
}: {
  activePeers: TrustedPeer[];
  source: PersonaSourceLatestResponse | null;
  sourceLoading: boolean;
  sourceError: boolean;
  selectedModel: ModelInfo | null;
  sourceModelId: string | null;
  modelMatchesSource: boolean;
  job: NeuralImprintGenerationJob | null;
  mutationError: unknown;
  t: ReturnType<typeof useT>;
}): string {
  if (job?.status === 'succeeded' && job.result) {
    return t('training.neuralImprintChat.generateSucceeded', { tokens: job.result.prefix_token_count });
  }
  if (job?.status === 'failed') {
    return job.error?.message ?? t('training.neuralImprintChat.generateFailed');
  }
  if (job?.status === 'queued' || job?.status === 'running') {
    return t('training.neuralImprintChat.generateRunning');
  }
  if (mutationError) {
    return errorMessage(mutationError, t('training.neuralImprintChat.generateFailed'));
  }
  if (activePeers.length === 0) return t('training.neuralImprintChat.noPairedDevices');
  if (!selectedModel) return t('training.neuralImprintChat.loadModelFirst');
  if (sourceLoading && !source) return t('training.neuralImprintChat.checkingSource');
  if (sourceError || !source) return t('training.neuralImprintChat.noDeviceSource');
  if (!source.receipt.profile_body_sha256) return t('training.neuralImprintChat.noRppProfileSource');
  if (sourceModelId && !modelMatchesSource) {
    return t('training.neuralImprintChat.modelMismatch', {
      sourceModel: sourceModelId,
      hostModel: selectedModel ? friendlyModelName(selectedModel) : '-',
    });
  }
  return t('training.neuralImprintChat.readyToGenerate', {
    model: sourceModelId ?? friendlyModelName(selectedModel),
  });
}

function friendlyModelName(model: ModelInfo): string {
  const fromPath = model.model_dir.split('/').filter(Boolean).at(-1);
  const raw = fromPath || model.model_name || model.model_id;
  return raw
    .replace(/Qwen3_5ForConditionalCaus(?:e|alLM)?/i, 'Qwen3.5')
    .replace(/Qwen3_5/i, 'Qwen3.5')
    .replace(/ForConditionalGeneration/i, '')
    .replace(/ForCausalLM/i, '')
    .replace(/[-_]?mlx[-_]?/i, '')
    .replace(/[_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function modelsMatch(model: ModelInfo | null, modelId: string): boolean {
  if (!model || !modelId.trim()) return false;
  const needle = normalizeModelKey(modelId);
  return [
    model.model_id,
    model.model_name,
    model.model_dir.split('/').filter(Boolean).at(-1) ?? '',
    friendlyModelName(model),
  ].some((value) => normalizeModelKey(value) === needle);
}

function normalizeModelKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function peerLabel(peer: TrustedPeer | null): string {
  if (!peer) return '';
  return peer.display_name || shortPeer(peer.peer_id);
}

function shortPeer(id: string): string {
  return id.length > 16 ? `${id.slice(0, 14)}...` : id;
}

function artifactKey(item: NeuralImprintArtifactSource): string {
  return item.artifact_id || item.artifact_sha256 || item.artifact_path;
}

function shortPath(path: string): string {
  const parts = path.split('/').filter(Boolean);
  return parts.slice(-3).join('/');
}

function errorMessage(err: unknown, fallback: string): string {
  const response = (err as { response?: { data?: { detail?: unknown } } })?.response;
  const detail = response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === 'string') return message;
  }
  return err instanceof Error ? err.message : fallback;
}
