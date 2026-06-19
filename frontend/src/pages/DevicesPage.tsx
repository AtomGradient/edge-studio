// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * DevicesPage — EdgeMesh trusted device manager.
 *
 * This page intentionally stays at the mesh/artifact level. Device
 * personalization now flows through RPP/profile artifacts and Neural Imprint
 * artifacts.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Database,
  Laptop,
  Network,
  Plus,
  RefreshCw,
  Send,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Smartphone,
  Sparkles,
  Square,
  Trash2,
  UploadCloud,
  WifiOff,
  X as XIcon,
} from 'lucide-react';
import {
  approvePairRequest,
  deletePeer,
  generateNeuralImprint,
  getEventStats,
  getHaloCapsuleAutomationPreview,
  getHaloCapsulePlan,
  getLatestPersonaSource,
  getMeshStatus,
  getNeuralImprintGenerationJob,
  listDevices,
  listNeuralImprintArtifacts,
  pushHaloCapsule,
  revokePeer,
  runHaloCapsuleAutomation,
  type EventStats,
  type HaloCapsuleAutomationPreviewResponse,
  type HaloCapsuleAutomationRunResponse,
  type HaloCapsuleApplyStatusReceipt,
  type HaloCapsuleCoordinatorPlanResponse,
  type HaloCapsulePushResponse,
  type HaloCapsuleTransferAckReceipt,
  type MeshStatus,
  type NeuralImprintGenerationJob,
  type PersonaSourceLatestResponse,
  type TrustedPeer,
} from '@/api/mesh';
import { listLoadedModels } from '@/api/endpoints';
import type { ModelInfo } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { QRPairingModal } from '@/components/mesh/QRPairingModal';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveMeshCapabilities,
  assessMeshHealth,
  buildMeshContextSnippet,
  buildMeshAutoBrief,
  getMeshSuggestedPrompts,
} from '@/lib/meshInsights';

interface PairRequestEvent {
  type: 'pair_request';
  requester_peer_id: string;
  requester_display_name: string;
  requester_fingerprint: string;
  pin: string;
  nonce: string;
  ttl_seconds: number;
  from_ip: string;
}

const DEVICES_POLL_MS = 3000;
const STATUS_POLL_MS = 5000;
const STATS_POLL_MS = 10000;

export default function DevicesPage() {
  const [pairOpen, setPairOpen] = useState(false);
  const [revokeConfirm, setRevokeConfirm] = useState<TrustedPeer | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<TrustedPeer | null>(null);
  const [incomingRequest, setIncomingRequest] = useState<PairRequestEvent | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const wsRef = useRef<WebSocket | null>(null);
  const briefFiredForRef = useRef<string | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/api/mesh/events/stream`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      try {
        const event = JSON.parse(ev.data);
        if (event.type === 'keepalive') return;
        if (event.type === 'pair_request') {
          setIncomingRequest(event as PairRequestEvent);
        }
        if (
          event.type === 'peer_paired' ||
          event.type === 'peer_revoked' ||
          event.type === 'peer_deleted' ||
          event.type === 'peer_connected' ||
          event.type === 'peer_disconnected' ||
          event.type === 'pair_approved'
        ) {
          queryClient.invalidateQueries({ queryKey: ['mesh', 'devices'] });
          queryClient.invalidateQueries({ queryKey: ['mesh', 'status'] });
        }
        if (event.type === 'halo_capsule_apply_status') {
          queryClient.invalidateQueries({
            queryKey: ['mesh', 'halo-capsule', 'plan', event.peer_id],
          });
          queryClient.invalidateQueries({
            queryKey: ['mesh', 'halo-capsule', 'automation-preview'],
          });
        }
        if (event.type === 'halo_capsule_offer_ack') {
          queryClient.invalidateQueries({
            queryKey: ['mesh', 'halo-capsule', 'plan', event.peer_id],
          });
        }
      } catch {
        // Ignore malformed frames; periodic polling still covers the page.
      }
    };

    ws.onerror = () => { /* polling covers transport blips */ };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [queryClient]);

  const statusQ = useQuery({
    queryKey: ['mesh', 'status'],
    queryFn: getMeshStatus,
    refetchInterval: STATUS_POLL_MS,
  });

  const devicesQ = useQuery({
    queryKey: ['mesh', 'devices'],
    queryFn: listDevices,
    refetchInterval: DEVICES_POLL_MS,
  });

  const statsQ = useQuery({
    queryKey: ['mesh', 'events', 'stats'],
    queryFn: getEventStats,
    refetchInterval: STATS_POLL_MS,
  });

  const modelsQ = useQuery({
    queryKey: ['model', 'loaded'],
    queryFn: listLoadedModels,
    refetchInterval: 10_000,
  });

  const revokeMut = useMutation({
    mutationFn: (peerId: string) => revokePeer(peerId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mesh', 'devices'] });
      setRevokeConfirm(null);
    },
  });

  const deleteMut = useMutation({
    mutationFn: (peerId: string) => deletePeer(peerId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mesh', 'devices'] });
      setDeleteConfirm(null);
    },
  });

  const approveMut = useMutation({
    mutationFn: (nonce: string) => approvePairRequest(nonce),
    onSuccess: () => {
      setIncomingRequest(null);
      queryClient.invalidateQueries({ queryKey: ['mesh', 'devices'] });
    },
  });

  const local = devicesQ.data?.local;
  const peers = useMemo(() => devicesQ.data?.peers ?? [], [devicesQ.data?.peers]);
  const pending = devicesQ.data?.pending ?? [];
  const activeCount = useMemo(() => peers.filter((p) => !p.revoked).length, [peers]);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';

  const brainModel = useMemo(() => {
    const list = modelsQ.data ?? [];
    return (
      list.find((m) => m.model_category === 'llm' || m.model_category === 'vlm') ??
      list[0] ??
      null
    );
  }, [modelsQ.data]);

  const caps = useMemo(
    () =>
      deriveMeshCapabilities(
        local ?? null,
        peers,
        statusQ.data ?? null,
        !!brainModel,
      ),
    [local, peers, statusQ.data, brainModel],
  );
  const meshRisk = useMemo(() => assessMeshHealth(caps), [caps]);
  const systemPrompt = useMemo(() => {
    if (!brainModel) return '';
    return `${buildModelSelfSystemPrompt(brainModel, locale)}\n\n${buildMeshContextSnippet(caps, brainModel, locale)}`;
  }, [brainModel, caps, locale]);
  const chat = useModelChat({
    modelId: brainModel?.model_id || null,
    systemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });
  const suggestedPrompts = useMemo(
    () => getMeshSuggestedPrompts(caps, brainModel, locale),
    [caps, brainModel, locale],
  );

  useEffect(() => {
    if (!brainModel || chat.streaming) return;
    const key = `${brainModel.model_id}:${caps.meshHealth}:${caps.activeCount}:${caps.onlineCount}:${locale}`;
    if (briefFiredForRef.current === key) return;
    briefFiredForRef.current = key;
    const id = window.setTimeout(() => {
      chat.send(buildMeshAutoBrief(caps, locale));
    }, 400);
    return () => window.clearTimeout(id);
    // chat captured via closure; refire only on real composition changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brainModel?.model_id, caps.meshHealth, caps.activeCount, caps.onlineCount, locale]);

  const handleSendDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || chat.streaming) return;
    chat.send(q);
    setDrawerInput('');
  }, [chat, drawerInput]);

  const riskBannerClass: Record<typeof meshRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  return (
    <div className="p-6">
      <PageHeader
        title="Devices"
        description="Manage paired iPhones, iPads and other Macs that can talk mTLS with this Edge Studio host. Device-side learning now syncs through RPP and Neural Imprint artifacts."
        actions={
          <button
            onClick={() => setPairOpen(true)}
            className="flex items-center gap-1.5 rounded-lg bg-stone-900 px-3 py-2 text-sm font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            <Plus size={14} />
            Pair new device
          </button>
        }
      />

      <StatusBanner status={statusQ.data} loading={statusQ.isLoading} />

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Brain size={16} />}
          label={t('mesh.cardThisMac')}
          value={brainModel ? brainModel.model_name : t('mesh.noBrain')}
          hint={brainModel
            ? `${(local?.display_name ?? 'this Mac')} · ${t('mesh.brainHost')}`
            : t('mesh.brainHint')}
          tone={brainModel ? 'indigo' : 'amber'}
        />
        <IdentityCard
          icon={<Network size={16} />}
          label={t('mesh.cardActive')}
          value={`${caps.activeCount}${caps.totalCount !== caps.activeCount ? ` / ${caps.totalCount}` : ''}`}
          hint={caps.activeCount === 0
            ? t('mesh.noPeers')
            : `${caps.activeCount} ${t('mesh.paired')}${caps.revoked.length > 0 ? `, ${caps.revoked.length} ${t('mesh.revoked')}` : ''}`}
          tone={caps.activeCount === 0 ? 'amber' : 'emerald'}
        />
        <IdentityCard
          icon={<CheckCircle2 size={16} />}
          label={t('mesh.cardOnline')}
          value={`${caps.onlineCount}`}
          hint={caps.activeCount === 0
            ? t('mesh.notApplicable')
            : caps.onlineCount === 0
              ? `${caps.stale.length} ${t('mesh.stale')}, ${caps.neverSeen.length} ${t('mesh.neverSeen')}`
              : `${t('mesh.lastMin')} · ${caps.recent.length} ${t('mesh.recent')}`}
          tone={caps.onlineCount > 0 ? 'emerald' : caps.activeCount > 0 ? 'amber' : 'neutral'}
        />
        <IdentityCard
          icon={<Shield size={16} />}
          label={t('mesh.cardSovereignty')}
          value={caps.meshUp ? t('mesh.mtlsUp') : t('mesh.mtlsDown')}
          hint={caps.meshUp
            ? `Bonjour · mTLS · 0 ${t('mesh.cloudRelays')}`
            : t('mesh.transportDown')}
          tone={caps.meshHealth === 'green' ? 'emerald' : caps.meshHealth === 'yellow' ? 'amber' : 'red'}
        />
      </div>

      {meshRisk.level !== 'safe' && (
        <div className={`mt-3 rounded-lg border px-3 py-2 text-xs ${riskBannerClass[meshRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {meshRisk.level === 'danger' ? t('mesh.riskDanger') : t('mesh.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? meshRisk.reasonZh : meshRisk.reason}
        </div>
      )}

      {brainModel && (
        <ModelBriefCard
          className="mt-4"
          label={t('mesh.briefTitle')}
          text={chat.text}
          streaming={chat.streaming}
          emptyText={t('mesh.briefEmpty')}
          streamingText={t('mesh.briefThinking')}
          refreshTitle={t('mesh.briefRefire')}
          prompts={suggestedPrompts}
          onRefresh={() => {
            briefFiredForRef.current = null;
            chat.reset();
            chat.send(buildMeshAutoBrief(caps, locale));
          }}
          onPrompt={(prompt) => {
            chat.reset();
            chat.send(prompt);
            setDrawerOpen(true);
          }}
        />
      )}

      {peers.some((p) => !p.revoked) && (
        <>
          <HaloCapsuleAutomationPreviewCard />
          <NeuralImprintGenerationCard
            peers={peers}
            hostModels={modelsQ.data ?? []}
          />
          <HaloCapsulePushCard peers={peers} />
        </>
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <LocalIdentityCard local={local} />

        <div className="rounded-xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-950 lg:col-span-2">
          <div className="flex items-center justify-between border-b border-stone-100 px-5 py-3 dark:border-stone-800">
            <div>
              <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
                Trusted devices
              </h3>
              <p className="text-xs text-stone-500 dark:text-stone-400">
                {activeCount} active{peers.length !== activeCount && `, ${peers.length - activeCount} revoked`}
                {pending.length > 0 && `, ${pending.length} pairing pending`}
              </p>
            </div>
            <button
              onClick={() => {
                devicesQ.refetch();
                statusQ.refetch();
              }}
              className="rounded-lg p-1.5 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              title="Refresh"
            >
              <RefreshCw size={14} className={devicesQ.isFetching ? 'animate-spin' : undefined} />
            </button>
          </div>

          {peers.length === 0 ? (
            <EmptyPeers />
          ) : (
            <ul className="divide-y divide-stone-100 dark:divide-stone-800">
              {peers.map((p) => (
                <PeerRow
                  key={p.peer_id}
                  peer={p}
                  onRevoke={() => setRevokeConfirm(p)}
                  onDelete={() => setDeleteConfirm(p)}
                />
              ))}
            </ul>
          )}
        </div>
      </div>

      <EventStatsCard stats={statsQ.data} />

      {pairOpen && <QRPairingModal onClose={() => setPairOpen(false)} />}

      {revokeConfirm && (
        <RevokeConfirmDialog
          peer={revokeConfirm}
          onCancel={() => setRevokeConfirm(null)}
          onConfirm={() => revokeMut.mutate(revokeConfirm.peer_id)}
          busy={revokeMut.isPending}
        />
      )}

      {deleteConfirm && (
        <DeleteConfirmDialog
          peer={deleteConfirm}
          onCancel={() => setDeleteConfirm(null)}
          onConfirm={() => deleteMut.mutate(deleteConfirm.peer_id)}
          busy={deleteMut.isPending}
        />
      )}

      {incomingRequest && (
        <IncomingPairRequestDialog
          req={incomingRequest}
          onReject={() => setIncomingRequest(null)}
          onApprove={() => approveMut.mutate(incomingRequest.nonce)}
          busy={approveMut.isPending}
        />
      )}

      {brainModel && (
        <>
          {!drawerOpen && (
            <button
              type="button"
              onClick={() => setDrawerOpen(true)}
              className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-1.5 rounded-full bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-indigo-500/30 transition-all hover:bg-indigo-700 hover:shadow-indigo-500/50 dark:bg-indigo-500 dark:hover:bg-indigo-600"
            >
              <Sparkles size={14} />
              {t('mesh.askFab')}
              <span className="ml-1 max-w-[140px] truncate text-[10px] font-normal opacity-80">
                [{brainModel.model_name}]
              </span>
            </button>
          )}
          {drawerOpen && (
            <div className="fixed bottom-6 right-6 z-40 flex w-[min(420px,calc(100vw-3rem))] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
              <div className="flex items-center justify-between border-b border-stone-200 px-3 py-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-stone-700 dark:text-stone-200">
                  <Sparkles size={13} className="text-indigo-500" />
                  {t('mesh.askDrawerTitle')}
                  <span className="text-[10px] font-normal text-stone-400">[{brainModel.model_name}]</span>
                </div>
                <button
                  type="button"
                  onClick={() => setDrawerOpen(false)}
                  className="rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800 dark:hover:text-stone-200"
                >
                  <XIcon size={14} />
                </button>
              </div>
              <div className="max-h-[50vh] min-h-[8rem] overflow-y-auto px-3 py-3 text-sm leading-relaxed text-stone-700 dark:text-stone-200">
                {chat.text ? (
                  <div className="whitespace-pre-wrap">{chat.text}</div>
                ) : (
                  <p className="text-xs text-stone-400">{t('mesh.askDrawerHint')}</p>
                )}
                {chat.streaming && <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-indigo-400" />}
              </div>
              <div className="border-t border-stone-200 p-2 dark:border-stone-700">
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={drawerInput}
                    onChange={(e) => setDrawerInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault();
                        handleSendDrawer();
                      }
                    }}
                    placeholder={t('mesh.askDrawerPlaceholder')}
                    disabled={chat.streaming}
                    className="flex-1 rounded-lg border border-stone-200 bg-white px-2.5 py-1.5 text-xs outline-none focus:border-indigo-400 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-200 dark:focus:border-indigo-500"
                  />
                  {chat.streaming ? (
                    <button
                      type="button"
                      onClick={() => chat.cancel()}
                      className="inline-flex items-center gap-1 rounded-lg bg-red-500 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-red-600"
                    >
                      <Square size={12} /> {t('mesh.askDrawerStop')}
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handleSendDrawer}
                      disabled={!drawerInput.trim()}
                      className="inline-flex items-center gap-1 rounded-lg bg-indigo-600 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-indigo-700 disabled:opacity-40 dark:bg-indigo-500 dark:hover:bg-indigo-600"
                    >
                      <Send size={12} /> {t('mesh.askDrawerSend')}
                    </button>
                  )}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {suggestedPrompts.slice(0, 4).map((p) => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => {
                        chat.reset();
                        chat.send(p.prompt);
                      }}
                      disabled={chat.streaming}
                      className="rounded-full border border-stone-200 bg-stone-50 px-2 py-0.5 text-[10px] font-medium text-stone-600 hover:bg-stone-100 disabled:opacity-40 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300 dark:hover:bg-stone-700"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StatusBanner({
  status,
  loading,
}: {
  status?: MeshStatus;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-stone-200 bg-stone-50 px-4 py-2 text-sm text-stone-600 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-400">
        <RefreshCw size={14} className="animate-spin" />
        Checking mesh status...
      </div>
    );
  }
  if (!status) return null;

  const ok = status.transport_running && status.discovery_running;
  return (
    <div
      className={`flex items-center justify-between rounded-lg border px-4 py-2 text-sm ${
        ok
          ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/50 dark:text-emerald-300'
          : 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-300'
      }`}
    >
      <span className="flex items-center gap-2">
        {ok ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />}
        {ok
          ? 'EdgeMesh transport + Bonjour are running.'
          : `Mesh not fully up: transport=${status.transport_running}, bonjour=${status.discovery_running}.`}
      </span>
      <span className="font-mono text-xs">
        mTLS :{status.mesh_port} · HTTP :{status.http_port}
      </span>
    </div>
  );
}

function LocalIdentityCard({
  local,
}: {
  local: Awaited<ReturnType<typeof listDevices>>['local'] | undefined;
}) {
  if (!local) {
    return (
      <div className="rounded-xl border border-stone-200 bg-white p-5 dark:border-stone-800 dark:bg-stone-950">
        <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">This Mac</h3>
        <p className="mt-2 text-xs text-stone-500 dark:text-stone-400">Loading...</p>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-stone-200 bg-white p-5 dark:border-stone-800 dark:bg-stone-950">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-100 text-indigo-600 dark:bg-indigo-900/40 dark:text-indigo-300">
          <Laptop size={20} />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold text-stone-900 dark:text-stone-100">
            {local.display_name}
          </h3>
          <p className="truncate text-xs text-stone-500 dark:text-stone-400">
            {local.ipv4 ?? '-'} · peer {local.peer_id.slice(0, 14)}...
          </p>
        </div>
      </div>
      <dl className="mt-4 grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-xs">
        <dt className="text-stone-500 dark:text-stone-400">mTLS port</dt>
        <dd className="font-mono text-stone-900 dark:text-stone-100">{local.mesh_port}</dd>
        <dt className="text-stone-500 dark:text-stone-400">HTTP port</dt>
        <dd className="font-mono text-stone-900 dark:text-stone-100">{local.http_port}</dd>
        <dt className="text-stone-500 dark:text-stone-400">Fingerprint</dt>
        <dd className="break-all font-mono text-[10px] text-stone-700 dark:text-stone-300">
          {local.fingerprint}
        </dd>
      </dl>
    </div>
  );
}

function EmptyPeers() {
  return (
    <div className="px-5 py-10 text-center">
      <WifiOff className="mx-auto text-stone-300 dark:text-stone-700" size={32} />
      <p className="mt-2 text-sm text-stone-500 dark:text-stone-400">
        No paired devices yet
      </p>
      <p className="mt-1 text-xs text-stone-400 dark:text-stone-500">
        Click "Pair new device" to show a QR code for your iPhone or iPad.
      </p>
    </div>
  );
}

function PeerRow({
  peer,
  onRevoke,
  onDelete,
}: {
  peer: TrustedPeer;
  onRevoke: () => void;
  onDelete: () => void;
}) {
  const paired = new Date(peer.paired_at * 1000);
  const lastSeen = peer.last_seen_at ? new Date(peer.last_seen_at * 1000) : null;
  return (
    <li className="flex items-center gap-4 px-5 py-3">
      <div
        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${
          peer.revoked
            ? 'bg-red-50 text-red-500 dark:bg-red-950/40 dark:text-red-300'
            : peer.role === 'brain'
              ? 'bg-purple-50 text-purple-600 dark:bg-purple-950/40 dark:text-purple-300'
              : 'bg-sky-50 text-sky-600 dark:bg-sky-950/40 dark:text-sky-300'
        }`}
      >
        {peer.role === 'brain' ? <Laptop size={18} /> : <Smartphone size={18} />}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <p
            className={`truncate text-sm font-medium ${
              peer.revoked
                ? 'text-stone-400 line-through dark:text-stone-500'
                : 'text-stone-900 dark:text-stone-100'
            }`}
          >
            {peer.display_name}
          </p>
          <RoleBadge role={peer.role} />
          {peer.revoked && <RevokedBadge />}
        </div>
        <p className="mt-0.5 truncate text-xs text-stone-500 dark:text-stone-400">
          paired {paired.toLocaleString()} ·{' '}
          {lastSeen ? `last seen ${relative(lastSeen)}` : 'never connected'}
        </p>
      </div>

      <div className="flex shrink-0 items-center gap-1">
        {peer.revoked ? (
          <button
            onClick={onDelete}
            className="flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-2.5 py-1 text-xs font-medium text-red-700 hover:bg-red-100 dark:border-red-900 dark:bg-red-950/50 dark:text-red-300 dark:hover:bg-red-950"
            title="Permanently delete this device from the trust store"
          >
            <Trash2 size={12} />
            Delete
          </button>
        ) : (
          <button
            onClick={onRevoke}
            className="flex items-center gap-1.5 rounded-lg border border-stone-200 px-2.5 py-1 text-xs font-medium text-red-600 hover:border-red-200 hover:bg-red-50 dark:border-stone-700 dark:text-red-400 dark:hover:border-red-900 dark:hover:bg-red-950/50"
          >
            <Trash2 size={12} />
            Revoke
          </button>
        )}
      </div>
    </li>
  );
}

function RoleBadge({ role }: { role: string }) {
  const map: Record<string, string> = {
    brain: 'bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300',
    sensor: 'bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300',
    peer: 'bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300',
  };
  return (
    <span
      className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
        map[role] ?? map.peer
      }`}
    >
      {role}
    </span>
  );
}

function RevokedBadge() {
  return (
    <span className="flex items-center gap-1 rounded-full bg-red-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-red-700 dark:bg-red-950/50 dark:text-red-300">
      <ShieldAlert size={10} />
      Revoked
    </span>
  );
}

function EventStatsCard({ stats }: { stats?: EventStats }) {
  return (
    <div className="mt-6 rounded-xl border border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-950">
      <div className="flex items-center gap-2 border-b border-stone-100 px-5 py-3 dark:border-stone-800">
        <Database size={14} className="text-stone-500" />
        <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
          Event stream
        </h3>
      </div>
      {!stats || stats.total_events === 0 ? (
        <div className="px-5 py-8 text-center text-xs text-stone-400 dark:text-stone-500">
          No events ingested yet. Paired iOS devices can upload over mTLS once the user interacts with the app.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 p-5 md:grid-cols-3">
          <StatTile
            label="Total events"
            value={stats.total_events.toLocaleString()}
            hint={`${(stats.total_bytes / 1024).toFixed(1)} KB payload`}
          />
          <StatTile
            label="By type"
            value={Object.keys(stats.per_type).length.toString()}
            hint={Object.entries(stats.per_type)
              .slice(0, 3)
              .map(([k, v]) => `${k}: ${v}`)
              .join(' · ') || '-'}
          />
          <StatTile
            label="By source"
            value={Object.keys(stats.per_source_peer).length.toString()}
            hint={Object.entries(stats.per_source_peer)
              .slice(0, 3)
              .map(([k, v]) => `${shortPeer(k)}: ${v}`)
              .join(' · ') || '-'}
          />
        </div>
      )}
    </div>
  );
}

function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-stone-100 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-900">
      <p className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 dark:text-stone-500">
        {label}
      </p>
      <p className="mt-1 text-2xl font-semibold text-stone-900 dark:text-stone-100">
        {value}
      </p>
      {hint && (
        <p className="mt-1 truncate font-mono text-[11px] text-stone-500 dark:text-stone-400">
          {hint}
        </p>
      )}
    </div>
  );
}

function HaloCapsuleAutomationPreviewCard() {
  const queryClient = useQueryClient();
  const [lastRun, setLastRun] = useState<HaloCapsuleAutomationRunResponse | null>(null);
  const [runningPeerId, setRunningPeerId] = useState<string | null>(null);
  const previewQ = useQuery({
    queryKey: ['mesh', 'halo-capsule', 'automation-preview'],
    queryFn: getHaloCapsuleAutomationPreview,
    retry: false,
    refetchInterval: 5000,
  });
  const preview = previewQ.data ?? null;
  const runMut = useMutation({
    mutationFn: (peerId: string) =>
      runHaloCapsuleAutomation({
        dry_run: false,
        peer_ids: [peerId],
        max_pushes: 1,
      }),
    onMutate: (peerId) => {
      setRunningPeerId(peerId);
    },
    onSuccess: (result, peerId) => {
      setLastRun(result);
      queryClient.invalidateQueries({ queryKey: ['mesh', 'halo-capsule', 'automation-preview'] });
      queryClient.invalidateQueries({ queryKey: ['mesh', 'halo-capsule', 'plan', peerId] });
    },
    onSettled: () => {
      setRunningPeerId(null);
    },
  });

  return (
    <div className="mt-4 rounded-xl border border-indigo-200 bg-white p-4 dark:border-indigo-900 dark:bg-stone-950">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
            Halo automation preview
          </h3>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            Dry-run by default. Run once requires an explicit peer.
          </p>
        </div>
        <button
          type="button"
          onClick={() => previewQ.refetch()}
          disabled={previewQ.isFetching}
          className="inline-flex items-center gap-1 rounded-md border border-stone-200 px-2 py-1 text-[11px] text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900"
        >
          <RefreshCw className={`h-3 w-3 ${previewQ.isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {previewQ.isLoading && !preview && (
        <div className="rounded-lg border border-stone-200 bg-stone-50 px-3 py-2 text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-400">
          <span className="inline-flex items-center gap-1.5">
            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
            Building automation preview...
          </span>
        </div>
      )}

      {previewQ.isError && !preview && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/35 dark:text-red-300">
          preview unavailable · {errorMessage(previewQ.error)}
        </div>
      )}

      {preview && (
        <>
          <div className="grid gap-2 text-[11px] text-stone-500 dark:text-stone-400 sm:grid-cols-3">
            <PreviewMetric
              label="Candidates"
              value={`${preview.candidate_count}`}
              hint={`${preview.peer_count} active peer${preview.peer_count === 1 ? '' : 's'}`}
              tone={preview.candidate_count > 0 ? 'emerald' : 'neutral'}
            />
            <PreviewMetric
              label="Skipped"
              value={`${preview.skipped_revoked_count}`}
              hint="revoked peers"
              tone={preview.skipped_revoked_count > 0 ? 'amber' : 'neutral'}
            />
            <PreviewMetric
              label="Mode"
              value={preview.dry_run ? 'dry-run' : 'active'}
              hint={preview.schema_version.replace('edgestudio.', '')}
              tone="indigo"
            />
          </div>

          <div className="mt-3 divide-y divide-stone-100 overflow-hidden rounded-lg border border-stone-100 dark:divide-stone-800 dark:border-stone-800">
            {preview.entries.length === 0 ? (
              <div className="px-3 py-2 text-xs text-stone-500 dark:text-stone-400">
                No active peers to plan.
              </div>
            ) : (
              preview.entries.map((entry) => (
                <AutomationPreviewRow
                  key={entry.peer_id}
                  entry={entry}
                  running={runningPeerId === entry.peer_id}
                  onRun={() => runMut.mutate(entry.peer_id)}
                />
              ))
            )}
          </div>

          {lastRun && (
            <div className="mt-2 rounded-lg border border-stone-100 bg-stone-50 px-3 py-2 text-[11px] text-stone-600 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-300">
              Last run · {lastRun.dry_run ? 'dry-run' : 'execute'} · pushed {lastRun.pushed_count}/{lastRun.attempted_count}
              {lastRun.results[0]?.status && ` · ${lastRun.results[0].peer_id}: ${lastRun.results[0].status}`}
            </div>
          )}

          {runMut.isError && (
            <div className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-[11px] text-red-700 dark:border-red-900 dark:bg-red-950/35 dark:text-red-300">
              run failed · {errorMessage(runMut.error)}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function PreviewMetric({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint: string;
  tone: 'emerald' | 'amber' | 'indigo' | 'neutral';
}) {
  const toneClass: Record<typeof tone, string> = {
    emerald: 'text-emerald-700 dark:text-emerald-300',
    amber: 'text-amber-700 dark:text-amber-300',
    indigo: 'text-indigo-700 dark:text-indigo-300',
    neutral: 'text-stone-700 dark:text-stone-200',
  };
  return (
    <div className="rounded-lg border border-stone-100 bg-stone-50 p-3 dark:border-stone-800 dark:bg-stone-900">
      <div className="text-[10px] font-semibold uppercase tracking-widest text-stone-400 dark:text-stone-500">
        {label}
      </div>
      <div className={`mt-1 text-lg font-semibold ${toneClass[tone]}`}>{value}</div>
      <div className="mt-0.5 truncate font-mono text-[10px] text-stone-500 dark:text-stone-400">
        {hint}
      </div>
    </div>
  );
}

function AutomationPreviewRow({
  entry,
  running,
  onRun,
}: {
  entry: HaloCapsuleAutomationPreviewResponse['entries'][number];
  running: boolean;
  onRun: () => void;
}) {
  const tone =
    entry.would_push
      ? 'text-emerald-600 dark:text-emerald-300'
      : entry.action.kind === 'already_applied' ||
          entry.action.kind === 'neural_imprint_active_no_push_needed' ||
          entry.action.kind === 'persona_active_no_push_needed'
        ? 'text-sky-600 dark:text-sky-300'
        : 'text-amber-700 dark:text-amber-300';
  return (
    <div className="grid gap-2 px-3 py-2 text-xs sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)_auto_auto] sm:items-center">
      <div className="min-w-0">
        <div className="truncate font-medium text-stone-800 dark:text-stone-100">
          {entry.display_name || shortPeer(entry.peer_id)}
        </div>
        <div className="truncate font-mono text-[10px] text-stone-400">
          {shortPeer(entry.peer_id)} · {entry.connected ? 'connected' : 'offline'}
        </div>
      </div>
      <div className="min-w-0">
        <div className={`truncate font-medium ${tone}`}>{entry.action.label}</div>
        <div className="truncate text-[11px] text-stone-500 dark:text-stone-400">
          {entry.action.reasons.join(' · ') || entry.action.kind}
        </div>
      </div>
      <div className="justify-self-start rounded-full border border-stone-200 px-2 py-0.5 font-mono text-[10px] text-stone-500 dark:border-stone-700 dark:text-stone-400 sm:justify-self-end">
        {entry.would_push ? 'would_push' : entry.action.kind}
      </div>
      <button
        type="button"
        onClick={onRun}
        disabled={!entry.would_push || running}
        className="inline-flex h-7 items-center justify-center gap-1 rounded-md border border-indigo-200 px-2 text-[10px] font-medium text-indigo-700 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-indigo-900 dark:text-indigo-300 dark:hover:bg-indigo-950/40"
        title={entry.would_push ? 'Run once for this peer' : entry.action.reasons[0] || entry.action.label}
      >
        {running ? <RefreshCw className="h-3 w-3 animate-spin" /> : <UploadCloud className="h-3 w-3" />}
        {running ? 'Running' : 'Run once'}
      </button>
    </div>
  );
}

function NeuralImprintGenerationCard({
  peers,
  hostModels,
}: {
  peers: TrustedPeer[];
  hostModels: ModelInfo[];
}) {
  const queryClient = useQueryClient();
  const activePeers = useMemo(() => peers.filter((p) => !p.revoked), [peers]);
  const modelOptions = useMemo(
    () => hostModels.filter((m) => m.model_category === 'llm' || m.model_category === 'vlm'),
    [hostModels],
  );
  const [peerId, setPeerId] = useState(activePeers[0]?.peer_id ?? '');
  const [modelDir, setModelDir] = useState(modelOptions[0]?.model_dir ?? '');
  const [jobId, setJobId] = useState<string | null>(null);
  const invalidatedJobRef = useRef<string | null>(null);
  const lastAutoModelIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!peerId && activePeers[0]?.peer_id) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPeerId(activePeers[0].peer_id);
    }
  }, [activePeers, peerId]);

  const sourceQ = useQuery({
    queryKey: ['neural-imprint', 'source-latest', peerId],
    queryFn: () => getLatestPersonaSource(peerId),
    enabled: Boolean(peerId),
    retry: false,
    refetchInterval: 10_000,
  });
  const source = sourceQ.data ?? null;
  const sourceModelId = source?.receipt.base_model_id ?? null;
  const selectedModel = modelOptions.find((model) => model.model_dir === modelDir) ?? null;

  useEffect(() => {
    const currentIsValid = modelOptions.some((model) => model.model_dir === modelDir);
    if (sourceModelId && lastAutoModelIdRef.current !== sourceModelId) {
      lastAutoModelIdRef.current = sourceModelId;
      const matched = modelOptions.find((model) => modelsMatch(model, sourceModelId));
      if (matched) {
        // eslint-disable-next-line react-hooks/set-state-in-effect
        setModelDir(matched.model_dir);
        return;
      }
    }
    if (!currentIsValid) {
      setModelDir(modelOptions[0]?.model_dir ?? '');
    }
  }, [modelDir, modelOptions, sourceModelId]);

  useEffect(() => {
    if (!sourceModelId) {
      lastAutoModelIdRef.current = null;
    }
  }, [sourceModelId]);

  const generateMut = useMutation({
    mutationFn: () => {
      if (!peerId || !selectedModel) {
        throw new Error('missing peer or host model');
      }
      return generateNeuralImprint({
        peer_id: peerId,
        model_dir: selectedModel.model_dir,
        model_id: sourceModelId ?? selectedModel.model_id,
        validate_restore: false,
      });
    },
    onSuccess: (response) => {
      setJobId(response.job.job_id);
      invalidatedJobRef.current = null;
      queryClient.invalidateQueries({ queryKey: ['neural-imprint', 'artifacts'] });
    },
  });

  const jobQ = useQuery({
    queryKey: ['neural-imprint', 'generation-job', jobId],
    queryFn: () => getNeuralImprintGenerationJob(jobId as string),
    enabled: Boolean(jobId),
    retry: false,
    refetchInterval: (query) => {
      const status = query.state.data?.job.status;
      return status === 'succeeded' || status === 'failed' ? false : 2000;
    },
  });
  const job = jobQ.data?.job ?? generateMut.data?.job ?? null;
  const modelMatchesSource = !sourceModelId || modelsMatch(selectedModel, sourceModelId);

  useEffect(() => {
    if (!job || job.status !== 'succeeded' || invalidatedJobRef.current === job.job_id) {
      return;
    }
    invalidatedJobRef.current = job.job_id;
    queryClient.invalidateQueries({ queryKey: ['neural-imprint', 'artifacts'] });
    queryClient.invalidateQueries({ queryKey: ['mesh', 'halo-capsule', 'automation-preview'] });
    queryClient.invalidateQueries({ queryKey: ['mesh', 'halo-capsule', 'plan', peerId] });
  }, [job, peerId, queryClient]);

  const canGenerate =
    Boolean(peerId) &&
    Boolean(selectedModel) &&
    modelMatchesSource &&
    Boolean(source?.receipt.profile_body_sha256) &&
    !generateMut.isPending &&
    job?.status !== 'queued' &&
    job?.status !== 'running';
  const selectedPeer = activePeers.find((peer) => peer.peer_id === peerId) ?? null;

  return (
    <div className="mt-4 rounded-xl border border-violet-200 bg-white p-4 dark:border-violet-900 dark:bg-stone-950">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
            Neural Imprint generation
          </h3>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            Generate on this Mac from the latest device persona source. Push remains explicit.
          </p>
        </div>
        <button
          type="button"
          onClick={() => sourceQ.refetch()}
          disabled={sourceQ.isFetching || !peerId}
          className="inline-flex items-center gap-1 rounded-md border border-stone-200 px-2 py-1 text-[11px] text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900"
        >
          <RefreshCw className={`h-3 w-3 ${sourceQ.isFetching ? 'animate-spin' : ''}`} />
          Source
        </button>
      </div>

      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <label className="block flex-1 text-xs font-medium text-stone-700 dark:text-stone-200">
          Device
          <select
            value={peerId}
            onChange={(event) => {
              setPeerId(event.target.value);
              setJobId(null);
            }}
            className="mt-1 block w-full rounded-md border border-stone-200 bg-white px-2 py-1.5 text-xs dark:border-stone-700 dark:bg-stone-900"
          >
            {activePeers.map((peer) => (
              <option key={peer.peer_id} value={peer.peer_id}>
                {(peer.display_name || shortPeer(peer.peer_id))} · {shortPeer(peer.peer_id)}
              </option>
            ))}
          </select>
        </label>

        <label className="block flex-[2] text-xs font-medium text-stone-700 dark:text-stone-200">
          Host model
          <select
            value={modelDir}
            onChange={(event) => setModelDir(event.target.value)}
            className="mt-1 block w-full rounded-md border border-stone-200 bg-white px-2 py-1.5 text-xs dark:border-stone-700 dark:bg-stone-900"
          >
            {modelOptions.length === 0 && <option value="">No loaded LLM/VLM model</option>}
            {modelOptions.map((model) => (
              <option key={model.model_dir} value={model.model_dir}>
                {friendlyModelName(model)} · {model.hidden_size || '?'}h
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={() => generateMut.mutate()}
          disabled={!canGenerate}
          className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md bg-violet-600 px-3 text-xs font-medium text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-50"
          title={neuralImprintGenerateTitle(source, selectedModel, modelMatchesSource)}
        >
          {generateMut.isPending || job?.status === 'queued' || job?.status === 'running'
            ? <RefreshCw className="h-3.5 w-3.5 animate-spin" />
            : <Sparkles className="h-3.5 w-3.5" />}
          {job?.status === 'queued' || job?.status === 'running' ? 'Generating' : 'Generate'}
        </button>
      </div>

      <div className="mt-3 grid gap-2 text-[11px] text-stone-500 dark:text-stone-400 md:grid-cols-3">
        <div>
          <span className="font-semibold text-stone-700 dark:text-stone-200">Source</span>
          <div className="mt-0.5 truncate font-mono">
            {sourceQ.isLoading
              ? 'loading source...'
              : source
                ? `${source.receipt.source_kind} · ${shortHash(source.receipt.source_id)}`
                : sourceQ.isError
                  ? 'no persona source'
                  : 'none'}
          </div>
        </div>
        <div>
          <span className="font-semibold text-stone-700 dark:text-stone-200">Model id</span>
          <div className="mt-0.5 truncate font-mono">
            {sourceModelId ?? selectedModel?.model_id ?? 'none'}
          </div>
        </div>
        <GenerationJobSummary job={job} peerName={selectedPeer?.display_name ?? null} />
      </div>

      {(generateMut.isError || job?.status === 'failed') && (
        <p className="mt-2 text-[11px] text-red-600 dark:text-red-300">
          generation failed · {job?.error?.message ?? errorMessage(generateMut.error)}
        </p>
      )}
    </div>
  );
}

function GenerationJobSummary({
  job,
  peerName,
}: {
  job: NeuralImprintGenerationJob | null;
  peerName: string | null;
}) {
  const tone =
    job?.status === 'succeeded'
      ? 'text-emerald-600 dark:text-emerald-300'
      : job?.status === 'failed'
        ? 'text-red-600 dark:text-red-300'
        : job
          ? 'text-violet-600 dark:text-violet-300'
          : 'text-stone-500 dark:text-stone-400';
  const text = job
    ? job.status === 'succeeded' && job.result
      ? `${job.result.prefix_token_count} tokens · ${shortHash(job.result.artifact_sha256)}`
      : `${job.status} · ${shortHash(job.job_id)}`
    : peerName
      ? 'not generated in this session'
      : 'no device selected';
  return (
    <div>
      <span className="font-semibold text-stone-700 dark:text-stone-200">Generation</span>
      <div className={`mt-0.5 truncate font-mono ${tone}`}>
        {text}
      </div>
    </div>
  );
}

function HaloCapsulePushCard({ peers }: { peers: TrustedPeer[] }) {
  const queryClient = useQueryClient();
  const activePeers = useMemo(() => peers.filter((p) => !p.revoked), [peers]);
  const artifactsQ = useQuery({
    queryKey: ['neural-imprint', 'artifacts'],
    queryFn: () => listNeuralImprintArtifacts(false),
    refetchInterval: 10_000,
  });
  const artifacts = useMemo(
    () => (artifactsQ.data?.artifacts ?? []).filter((item) => item.valid && item.artifact_id),
    [artifactsQ.data?.artifacts],
  );
  const [peerId, setPeerId] = useState(activePeers[0]?.peer_id ?? '');
  const [artifactId, setArtifactId] = useState(artifacts[0]?.artifact_id ?? '');
  const [lastPush, setLastPush] = useState<HaloCapsulePushResponse | null>(null);

  useEffect(() => {
    if (!peerId && activePeers[0]?.peer_id) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setPeerId(activePeers[0].peer_id);
    }
  }, [activePeers, peerId]);

  useEffect(() => {
    if (!artifactId && artifacts[0]?.artifact_id) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setArtifactId(artifacts[0].artifact_id);
    }
  }, [artifactId, artifacts]);

  const planQ = useQuery({
    queryKey: ['mesh', 'halo-capsule', 'plan', peerId],
    queryFn: () => getHaloCapsulePlan(peerId),
    enabled: Boolean(peerId),
    retry: false,
    refetchInterval: 5000,
  });

  const pushMut = useMutation({
    mutationFn: () => {
      const request = planQ.data?.action.can_push
        ? planQ.data.action.push_request
        : { peer_id: peerId, artifact_id: artifactId };
      if (!request?.peer_id || !request.artifact_id) {
        throw new Error('missing peer or artifact');
      }
      return pushHaloCapsule(request);
    },
    onSuccess: (result) => {
      setLastPush(result);
      queryClient.invalidateQueries({ queryKey: ['mesh', 'halo-capsule', 'plan', peerId] });
    },
  });

  const selectedArtifact = artifacts.find((item) => item.artifact_id === artifactId) ?? null;
  const selectedPeer = activePeers.find((peer) => peer.peer_id === peerId) ?? null;
  const plan = planQ.data ?? null;
  const planPushArtifactId = plan?.action.push_request?.artifact_id ?? null;
  const planPushRequestReady = Boolean(
    plan?.action.push_request?.peer_id && plan?.action.push_request?.artifact_id,
  );
  const planMatchesSelection = !planPushArtifactId || planPushArtifactId === artifactId;
  const canPush =
    Boolean(peerId && artifactId) &&
    Boolean(plan?.action.can_push) &&
    planPushRequestReady &&
    planMatchesSelection &&
    !pushMut.isPending;

  return (
    <div className="mt-4 rounded-xl border border-sky-200 bg-white p-4 dark:border-sky-900 dark:bg-stone-950">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
            Neural Imprint capsule push
          </h3>
          <p className="text-xs text-stone-500 dark:text-stone-400">
            {artifactsQ.isLoading
              ? 'Scanning local artifact registry...'
              : artifacts.length > 0
                ? `${artifacts.length} artifact${artifacts.length === 1 ? '' : 's'} ready`
                : 'No valid Neural Imprint artifacts in the registry'}
          </p>
        </div>
        <button
          type="button"
          onClick={() => artifactsQ.refetch()}
          disabled={artifactsQ.isFetching}
          className="inline-flex items-center gap-1 rounded-md border border-stone-200 px-2 py-1 text-[11px] text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900"
        >
          <RefreshCw className={`h-3 w-3 ${artifactsQ.isFetching ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <HaloCapsulePlanSummary
        plan={plan}
        loading={planQ.isLoading || planQ.isFetching}
        error={planQ.error}
      />

      <div className="flex flex-col gap-3 md:flex-row md:items-end">
        <label className="block flex-1 text-xs font-medium text-stone-700 dark:text-stone-200">
          Device
          <select
            value={peerId}
            onChange={(e) => setPeerId(e.target.value)}
            className="mt-1 block w-full rounded-md border border-stone-200 bg-white px-2 py-1.5 text-xs dark:border-stone-700 dark:bg-stone-900"
          >
            {activePeers.map((peer) => (
              <option key={peer.peer_id} value={peer.peer_id}>
                {(peer.display_name || shortPeer(peer.peer_id))} · {shortPeer(peer.peer_id)}
              </option>
            ))}
          </select>
        </label>

        <label className="block flex-[2] text-xs font-medium text-stone-700 dark:text-stone-200">
          Artifact
          <select
            value={artifactId}
            onChange={(e) => setArtifactId(e.target.value)}
            className="mt-1 block w-full rounded-md border border-stone-200 bg-white px-2 py-1.5 text-xs dark:border-stone-700 dark:bg-stone-900"
          >
            {artifacts.length === 0 && <option value="">No valid artifacts</option>}
            {artifacts.map((artifact) => (
              <option key={artifact.artifact_id} value={artifact.artifact_id}>
                {artifact.base_model_id ?? 'unknown model'} · {artifact.prefix_token_count ?? '?'} tokens · {shortHash(artifact.artifact_sha256)}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={() => pushMut.mutate()}
          disabled={!canPush}
          className="inline-flex h-8 items-center justify-center gap-1.5 rounded-md bg-sky-600 px-3 text-xs font-medium text-white hover:bg-sky-700 disabled:opacity-50"
          title={pushButtonTitle(plan, planMatchesSelection)}
        >
          {pushMut.isPending ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <UploadCloud className="h-3.5 w-3.5" />}
          {pushMut.isPending ? 'Pushing...' : 'Push capsule'}
        </button>
      </div>

      <div className="mt-3 grid gap-2 text-[11px] text-stone-500 dark:text-stone-400 md:grid-cols-4">
        <div>
          <span className="font-semibold text-stone-700 dark:text-stone-200">Selected</span>
          <div className="mt-0.5 truncate font-mono">
            {selectedArtifact
              ? `${selectedArtifact.base_model_id ?? 'unknown'} · ${formatBytes(selectedArtifact.total_bytes)}`
              : 'none'}
          </div>
        </div>
        <div>
          <span className="font-semibold text-stone-700 dark:text-stone-200">Last push</span>
          <div className="mt-0.5 truncate font-mono">
            {lastPush
              ? `${shortHash(lastPush.transfer_id)} · ${lastPush.frame_count} frames`
              : 'not pushed in this session'}
          </div>
        </div>
        <ApplyStatusSummary
          status={plan?.last_apply_status ?? null}
          selectedPeer={selectedPeer}
        />
        <TransferAckSummary ack={plan?.last_transfer_ack ?? null} selectedPeer={selectedPeer} />
      </div>

      {pushMut.isError && (
        <p className="mt-2 text-[11px] text-red-600 dark:text-red-300">
          push failed · {errorMessage(pushMut.error)}
        </p>
      )}
    </div>
  );
}

function ApplyStatusSummary({
  status,
  selectedPeer,
}: {
  status?: HaloCapsuleApplyStatusReceipt | null;
  selectedPeer: TrustedPeer | null;
}) {
  const receipt = status;
  const tone =
    receipt?.status === 'applied'
      ? 'text-emerald-600 dark:text-emerald-300'
      : receipt?.status === 'failed'
        ? 'text-red-600 dark:text-red-300'
        : 'text-stone-500 dark:text-stone-400';
  const text = receipt
    ? `${receipt.status} · ${shortHash(receipt.capsule_id)}`
    : selectedPeer
      ? 'no apply status yet'
      : 'no device selected';
  return (
    <div>
      <span className="font-semibold text-stone-700 dark:text-stone-200">Device status</span>
      <div className={`mt-0.5 truncate font-mono ${tone}`}>
        {text}
      </div>
    </div>
  );
}

function TransferAckSummary({
  ack,
  selectedPeer,
}: {
  ack?: HaloCapsuleTransferAckReceipt | null;
  selectedPeer: TrustedPeer | null;
}) {
  const tone =
    ack?.accepted === false
      ? 'text-red-600 dark:text-red-300'
      : ack?.accepted === true
        ? 'text-emerald-600 dark:text-emerald-300'
        : 'text-stone-500 dark:text-stone-400';
  const text = ack
    ? `${ack.accepted ? 'accepted' : 'rejected'} · ${shortHash(ack.transfer_id)}`
    : selectedPeer
      ? 'no ACK yet'
      : 'no device selected';
  return (
    <div>
      <span className="font-semibold text-stone-700 dark:text-stone-200">Last ACK</span>
      <div className={`mt-0.5 truncate font-mono ${tone}`} title={ack?.reason ?? undefined}>
        {text}
      </div>
      {ack?.accepted === false && ack.reason && (
        <div className="mt-0.5 truncate text-[10px] text-red-600 dark:text-red-300" title={ack.reason}>
          {ack.reason}
        </div>
      )}
    </div>
  );
}

function HaloCapsulePlanSummary({
  plan,
  loading,
  error,
}: {
  plan: HaloCapsuleCoordinatorPlanResponse | null;
  loading: boolean;
  error: unknown;
}) {
  const action = plan?.action;
  const tone =
    action?.can_push
      ? 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/35 dark:text-emerald-300'
      : action?.kind === 'already_applied' ||
          action?.kind === 'neural_imprint_active_no_push_needed' ||
          action?.kind === 'persona_active_no_push_needed'
        ? 'border-sky-200 bg-sky-50 text-sky-800 dark:border-sky-900 dark:bg-sky-950/35 dark:text-sky-300'
        : 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/35 dark:text-amber-300';

  if (loading && !plan) {
    return (
      <div className="mb-3 rounded-lg border border-stone-200 bg-stone-50 px-3 py-2 text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-400">
        <span className="inline-flex items-center gap-1.5">
          <RefreshCw className="h-3.5 w-3.5 animate-spin" />
          Planning capsule action...
        </span>
      </div>
    );
  }

  if (error && !plan) {
    return (
      <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900 dark:bg-red-950/35 dark:text-red-300">
        plan unavailable · {errorMessage(error)}
      </div>
    );
  }

  if (!action) return null;

  return (
    <div className={`mb-3 rounded-lg border px-3 py-2 text-xs ${tone}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="inline-flex items-center gap-1.5 font-semibold">
          {action.can_push ? <UploadCloud className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
          {action.label}
        </span>
        <span className="font-mono text-[10px] opacity-80">
          {plan.selected_model_id ?? 'no model'} · {action.kind}
        </span>
      </div>
      {action.reasons.length > 0 && (
        <div className="mt-1 line-clamp-2 text-[11px] opacity-85">
          {action.reasons.join(' · ')}
        </div>
      )}
    </div>
  );
}


function RevokeConfirmDialog({
  peer,
  onCancel,
  onConfirm,
  busy,
}: {
  peer: TrustedPeer;
  onCancel: () => void;
  onConfirm: () => void;
  busy: boolean;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/50 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl dark:bg-stone-950"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 text-red-500" size={20} />
          <div>
            <h4 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
              Revoke {peer.display_name}?
            </h4>
            <p className="mt-1 text-xs text-stone-500 dark:text-stone-400">
              Future mTLS connections from this device will be rejected. The peer can re-pair later.
            </p>
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded-lg border border-stone-200 px-3 py-1.5 text-sm font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-800"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
          >
            {busy ? <RefreshCw size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
            Revoke
          </button>
        </div>
      </div>
    </div>
  );
}

function DeleteConfirmDialog({
  peer,
  onCancel,
  onConfirm,
  busy,
}: {
  peer: TrustedPeer;
  onCancel: () => void;
  onConfirm: () => void;
  busy: boolean;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/50 backdrop-blur-sm"
      onClick={onCancel}
    >
      <div
        className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl dark:bg-stone-950"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <Trash2 className="mt-0.5 text-red-500" size={20} />
          <div>
            <h4 className="text-sm font-semibold text-stone-900 dark:text-stone-100">
              Permanently delete {peer.display_name}?
            </h4>
            <p className="mt-1 text-xs text-stone-500 dark:text-stone-400">
              This removes the device from the trust store entirely. The peer must scan a new QR / PIN to pair again.
            </p>
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded-lg border border-stone-200 px-3 py-1.5 text-sm font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-800"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-700 disabled:opacity-50"
          >
            {busy ? <RefreshCw size={14} className="animate-spin" /> : <Trash2 size={14} />}
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

function IncomingPairRequestDialog({
  req,
  onReject,
  onApprove,
  busy,
}: {
  req: PairRequestEvent;
  onReject: () => void;
  onApprove: () => void;
  busy: boolean;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/60 backdrop-blur-sm"
      onClick={onReject}
    >
      <div
        className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl dark:bg-stone-950"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-100 text-blue-600 dark:bg-blue-950/40 dark:text-blue-300">
            <Smartphone size={20} />
          </div>
          <div className="min-w-0 flex-1">
            <h4 className="text-base font-semibold text-stone-900 dark:text-stone-100">
              {req.requester_display_name} wants to pair
            </h4>
            <p className="mt-1 truncate text-xs text-stone-500 dark:text-stone-400">
              from {req.from_ip} · peer {req.requester_peer_id.slice(0, 16)}...
            </p>
          </div>
        </div>

        <div className="mt-6 rounded-xl border border-blue-200 bg-blue-50 p-4 dark:border-blue-900 dark:bg-blue-950/30">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-blue-600 dark:text-blue-400">
            Confirm this PIN appears on the iPhone
          </p>
          <p className="mt-2 font-mono text-3xl font-bold tracking-[0.5em] text-blue-700 dark:text-blue-200">
            {req.pin}
          </p>
        </div>

        <p className="mt-4 text-xs text-stone-500 dark:text-stone-400">
          If the PIN matches exactly, tap Approve. Approving any device that
          shows a different PIN lets an attacker onto your mesh.
        </p>

        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onReject}
            className="rounded-lg border border-stone-200 px-3 py-1.5 text-sm font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-800"
          >
            Reject
          </button>
          <button
            onClick={onApprove}
            disabled={busy}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
          >
            {busy ? <RefreshCw size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
            Approve
          </button>
        </div>
      </div>
    </div>
  );
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

function neuralImprintGenerateTitle(
  source: PersonaSourceLatestResponse | null,
  selectedModel: ModelInfo | null,
  modelMatchesSource: boolean,
): string {
  if (!source) return 'Latest persona source is required';
  if (!source.receipt.profile_body_sha256) return 'RPP profile source is required';
  if (!selectedModel) return 'Load a host LLM/VLM model first';
  if (!modelMatchesSource) return 'Host model must match the persona source base model id';
  return 'Generate a local Neural Imprint artifact';
}

function relative(d: Date): string {
  const secs = Math.floor((Date.now() - d.getTime()) / 1000);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function shortPeer(id: string): string {
  return id.length > 16 ? `${id.slice(0, 14)}...` : id;
}

function shortHash(value?: string | null): string {
  if (!value) return 'unknown';
  return value.length > 14 ? `${value.slice(0, 12)}...` : value;
}

function formatBytes(value?: number | null): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'unknown size';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function pushButtonTitle(
  plan: HaloCapsuleCoordinatorPlanResponse | null,
  planMatchesSelection: boolean,
): string {
  if (!plan) return 'Waiting for coordinator plan';
  if (!planMatchesSelection) return 'Selected artifact differs from coordinator recommendation';
  if (!plan.action.can_push) return plan.action.reasons[0] || plan.action.label;
  return 'Push the coordinator-recommended Neural Imprint capsule';
}

function errorMessage(error: unknown): string {
  // Check Axios response first — error.message is generic "Request failed with status code N"
  if (typeof error === 'object' && error && 'response' in error) {
    const response = (error as { response?: { status?: number; data?: { detail?: unknown } } }).response;
    const detail = response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') {
      // Backend to_error() format: {code, message, retryable, details}
      if ('message' in detail) return String((detail as { message?: unknown }).message);
      if ('code' in detail) return String((detail as { code?: unknown }).code);
    }
    // Fallback to status code if detail wasn't useful
    if (response?.status) return `HTTP ${response.status}`;
  }
  if (error instanceof Error) return error.message;
  return 'request failed';
}
