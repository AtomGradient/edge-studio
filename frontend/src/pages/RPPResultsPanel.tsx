// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import axios from 'axios';
import { useQuery } from '@tanstack/react-query';
import {
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock3,
  FileJson,
  Hash,
  Layers,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  XCircle,
} from 'lucide-react';
import { inspectLatestRPPArtifacts } from '@/api/endpoints';
import { listDevices, type TrustedPeer } from '@/api/mesh';
import type { RPPDirectionSummary, RPPResultsInspectResponse } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT } from '@/i18n';
import { cn, formatSize } from '@/lib/utils';

type Translator = ReturnType<typeof useT>;

export default function RPPResultsPanel() {
  const t = useT();
  const [peerId, setPeerId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RPPResultsInspectResponse | null>(null);

  const devicesQ = useQuery({
    queryKey: ['mesh', 'devices', 'rpp-results'],
    queryFn: listDevices,
    refetchInterval: 5000,
  });
  const peers = useMemo(
    () => (devicesQ.data?.peers ?? []).filter((peer) => !peer.revoked),
    [devicesQ.data?.peers],
  );

  const categoryCountRows = useMemo(
    () => extractRows(result?.dataset_summary, 'top_categories_by_count'),
    [result?.dataset_summary],
  );
  const categoryAmountRows = useMemo(
    () => extractRows(result?.dataset_summary, 'top_categories_by_amount'),
    [result?.dataset_summary],
  );
  const weekdayRows = useMemo(
    () => extractRows(result?.dataset_summary, 'top_weekdays_by_count'),
    [result?.dataset_summary],
  );

  useEffect(() => {
    if (peerId || peers.length === 0) return;
    setPeerId(peers[0].peer_id);
  }, [peerId, peers]);

  const loadLatest = async () => {
    const id = peerId.trim();
    if (!id) {
      setError(t('training.rppResults.errorSelectDevice'));
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await inspectLatestRPPArtifacts(id);
      setResult(response);
    } catch (err) {
      setError(formatRPPError(err, t));
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5 pb-12">
      <PageHeader
        title={t('training.rppResults.title')}
        description={t('training.rppResults.description')}
        actions={(
          <button
            type="button"
            onClick={loadLatest}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
            {t('common.load')}
          </button>
        )}
      />

      <section className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
        <label className="block">
          <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-stone-500">
            {t('training.common.device')}
          </span>
          <select
            value={peerId}
            onChange={(event) => {
              setPeerId(event.target.value);
              setResult(null);
            }}
            autoComplete="off"
            className="w-full rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-sky-950"
          >
            {peers.length === 0 ? (
              <option value="">{t('training.common.noPairedDevices')}</option>
            ) : peers.map((peer) => (
              <option key={peer.peer_id} value={peer.peer_id}>
                {deviceLabel(peer, t)}
              </option>
            ))}
          </select>
        </label>
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-stone-500">
          <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-900">
            <ShieldCheck size={13} />
            {t('training.rppResults.rawFiltered')}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-900">
            <Layers size={13} />
            {t('training.rppResults.headerOnly')}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-900">
            <Brain size={13} />
            {devicesQ.isFetching ? t('training.common.refreshingDevices') : t('training.common.availableDevices', { count: peers.length })}
          </span>
        </div>
        {devicesQ.isError && (
          <p className="mt-2 text-xs text-amber-600 dark:text-amber-300">
            {t('training.common.deviceListUnavailable')}
          </p>
        )}
      </section>

      {error && <InlineIssue message={error} />}

      {!result ? (
        <EmptyPanel />
      ) : result.status !== 'found' ? (
        <MissingPanel peerId={result.peer_id} warnings={result.warnings} />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={<Brain size={16} />}
              label={t('training.rppResults.model')}
              value={result.summary.base_model_id ?? t('training.common.unknown')}
              hint={t('training.rppResults.layerDirectionsHint', {
                layer: result.summary.layer_id ?? t('training.common.unknown'),
                count: result.summary.direction_count ?? 0,
              })}
            />
            <MetricCard
              icon={<BarChart3 size={16} />}
              label={t('training.rppResults.dataset')}
              value={formatNumber(result.summary.n_transactions)}
              hint={t('training.rppResults.selectedDirectionsHint', { count: formatNumber(result.summary.k_selected) })}
            />
            <MetricCard
              icon={<Clock3 size={16} />}
              label={t('training.rppResults.runtime')}
              value={formatSeconds(result.summary.total_elapsed_seconds)}
              hint={formatTimestamp(result.received_at_ms)}
            />
            <MetricCard
              icon={<FileJson size={16} />}
              label={t('training.rppResults.artifacts')}
              value={String(result.artifacts.length)}
              hint={t('training.rppResults.bTensorsHint', { count: result.b_directions_header?.tensor_count ?? 0 })}
            />
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
            <section className="space-y-5">
              <InfoPanel title={t('training.rppResults.profileNarrative')} icon={<Sparkles size={16} />}>
                <div className="space-y-3">
                  {result.profile.name && (
                    <FieldRow label={t('training.rppResults.name')} value={result.profile.name} />
                  )}
                  {result.profile.summary && (
                    <FieldRow label={t('training.rppResults.summary')} value={result.profile.summary} />
                  )}
                  <p className="whitespace-pre-wrap rounded-lg bg-stone-50 p-3 text-sm leading-6 text-stone-700 dark:bg-stone-900 dark:text-stone-200">
                    {result.profile.narrative || t('training.rppResults.noProfileNarrative')}
                  </p>
                </div>
              </InfoPanel>

              <InfoPanel title={t('training.rppResults.datasetStatistics')} icon={<BarChart3 size={16} />}>
                <div className="space-y-4">
                  <StatsList title={t('training.rppResults.byCount')} rows={categoryCountRows} valueKey="count" />
                  <StatsList title={t('training.rppResults.byAmount')} rows={categoryAmountRows} valueKey="total_amount" money />
                  <StatsList title={t('training.rppResults.weekdays')} rows={weekdayRows} valueKey="count" />
                </div>
              </InfoPanel>

              <InfoPanel title={t('training.rppResults.runMetadata')} icon={<Hash size={16} />}>
                <FieldRow label={t('training.rppResults.peer')} value={result.peer_id} mono />
                <FieldRow label={t('training.rppResults.run')} value={result.rpp_run_id ?? t('training.common.unknown')} mono />
                <FieldRow label={t('training.rppResults.aVersion')} value={result.summary.a_version ?? t('training.common.unknown')} />
                <FieldRow label={t('training.rppResults.aHash')} value={result.summary.a_hash ?? t('training.common.unknown')} mono />
                <FieldRow label={t('training.rppResults.path')} value={result.storage_path ?? t('training.common.unknown')} mono />
              </InfoPanel>
            </section>

            <section className="space-y-5">
              <InfoPanel title={t('training.rppResults.bDirections')} icon={<CheckCircle2 size={16} />}>
                <DirectionTable directions={result.directions} t={t} />
              </InfoPanel>

              <InfoPanel title={t('training.rppResults.bDirectionsHeader')} icon={<Layers size={16} />}>
                <TensorHeader result={result} t={t} />
              </InfoPanel>

              <InfoPanel title={t('training.rppResults.artifacts')} icon={<FileJson size={16} />}>
                <ArtifactTable result={result} t={t} />
              </InfoPanel>

              {result.warnings.length > 0 && (
                <InfoPanel title={t('training.common.warnings')} icon={<AlertTriangle size={16} />}>
                  <div className="flex flex-wrap gap-2">
                    {result.warnings.map((warning) => (
                      <span
                        key={warning}
                        className="rounded-full bg-amber-100 px-2 py-1 text-xs font-medium text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
                      >
                        {warning}
                      </span>
                    ))}
                  </div>
                </InfoPanel>
              )}
            </section>
          </div>
        </>
      )}
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  hint,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="rounded-lg border border-stone-200 bg-white px-4 py-3 dark:border-stone-800 dark:bg-stone-950">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-sky-600 dark:text-sky-300">
        {icon}
        {label}
      </div>
      <div className="truncate text-xl font-semibold text-stone-900 dark:text-stone-100" title={value}>
        {value}
      </div>
      <div className="mt-1 truncate text-xs text-stone-500" title={hint}>{hint}</div>
    </div>
  );
}

function InfoPanel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-stone-900 dark:text-stone-100">
        <span className="text-stone-500">{icon}</span>
        {title}
      </div>
      {children}
    </section>
  );
}

function FieldRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[100px_minmax(0,1fr)] gap-3 border-b border-stone-100 py-2 text-sm last:border-0 dark:border-stone-800">
      <span className="text-stone-500">{label}</span>
      <span className={cn('truncate text-stone-900 dark:text-stone-100', mono && 'font-mono text-xs')} title={value}>
        {value}
      </span>
    </div>
  );
}

function DirectionTable({ directions, t }: { directions: RPPDirectionSummary[]; t: Translator }) {
  if (directions.length === 0) {
    return <p className="text-sm text-stone-500">{t('training.rppResults.noSelectedDirections')}</p>;
  }
  return (
    <div className="max-h-[520px] overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.direction')}</th>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.name')}</th>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.confidence')}</th>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.bootstrap')}</th>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.reason')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
          {directions.map((direction) => (
            <tr key={`${direction.direction_idx}:${direction.direction_id}`}>
              <td className="px-3 py-2">
                <div className="font-medium text-stone-900 dark:text-stone-100">#{direction.direction_idx}</div>
                <div className="font-mono text-[11px] text-stone-500">{direction.direction_id}</div>
              </td>
              <td className="max-w-[220px] px-3 py-2">
                <div className="truncate font-medium text-stone-800 dark:text-stone-100" title={direction.name}>
                  {direction.name}
                </div>
                <div className="mt-1 text-[11px] text-stone-500">
                  +{direction.top_positive_count ?? 0} / -{direction.top_negative_count ?? 0}
                </div>
              </td>
              <td className="px-3 py-2 tabular-nums text-stone-600 dark:text-stone-300">
                {formatPercent(direction.confidence)}
              </td>
              <td className="px-3 py-2">
                <span className={cn(
                  'inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium',
                  direction.bootstrap_pass === true && 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300',
                  direction.bootstrap_pass === false && 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300',
                  direction.bootstrap_pass == null && 'bg-stone-100 text-stone-600 dark:bg-stone-900 dark:text-stone-300',
                )}>
                  {direction.bootstrap_pass === true ? <CheckCircle2 size={12} /> : direction.bootstrap_pass === false ? <XCircle size={12} /> : null}
                  {direction.bootstrap_pass == null ? t('training.common.unknown') : direction.bootstrap_pass ? t('training.common.pass') : t('training.common.fail')}
                </span>
                <div className="mt-1 tabular-nums text-stone-500">
                  μ {formatDecimal(direction.mean_similarity)}
                </div>
              </td>
              <td className="max-w-[360px] px-3 py-2 text-stone-600 dark:text-stone-300">
                <div className="line-clamp-3" title={direction.reason ?? ''}>
                  {direction.reason ?? '—'}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatsList({
  title,
  rows,
  valueKey,
  money = false,
}: {
  title: string;
  rows: DatasetRow[];
  valueKey: string;
  money?: boolean;
}) {
  if (rows.length === 0) return null;
  const maxValue = Math.max(...rows.map((row) => numericValue(row[valueKey])), 1);
  return (
    <div>
      <div className="mb-2 text-xs font-medium uppercase tracking-wide text-stone-500">{title}</div>
      <div className="space-y-2">
        {rows.slice(0, 5).map((row) => {
          const value = numericValue(row[valueKey]);
          return (
            <div key={`${title}:${row.key}`} className="grid grid-cols-[96px_minmax(0,1fr)_72px] items-center gap-3 text-xs">
              <div className="truncate font-medium text-stone-700 dark:text-stone-200" title={row.key}>
                {row.key}
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-stone-100 dark:bg-stone-900">
                <div
                  className="h-full rounded-full bg-sky-500"
                  style={{ width: `${Math.max(4, Math.min(100, (value / maxValue) * 100))}%` }}
                />
              </div>
              <div className="text-right tabular-nums text-stone-500">
                {money ? formatMoney(value) : formatNumber(value)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TensorHeader({ result, t }: { result: RPPResultsInspectResponse; t: Translator }) {
  const header = result.b_directions_header;
  if (!header) {
    return <p className="text-sm text-stone-500">{t('training.rppResults.noBHeader')}</p>;
  }
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <MiniStat label={t('training.rppResults.file')} value={header.name ?? t('training.common.unknown')} />
        <MiniStat label={t('training.rppResults.header')} value={formatSize(header.header_size_bytes ?? 0)} />
        <MiniStat label={t('training.rppResults.tensors')} value={String(header.tensor_count ?? 0)} />
      </div>
      <div className="overflow-hidden rounded-lg border border-stone-200 dark:border-stone-800">
        <table className="w-full text-left text-xs">
          <thead className="bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
            <tr>
              <th className="px-3 py-2 font-medium">{t('training.rppResults.tensor')}</th>
              <th className="px-3 py-2 font-medium">{t('training.rppResults.dtype')}</th>
              <th className="px-3 py-2 font-medium">{t('training.rppResults.shape')}</th>
              <th className="px-3 py-2 text-right font-medium">{t('training.rppResults.bytes')}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
            {(header.tensors ?? []).map((tensor) => (
              <tr key={tensor.name}>
                <td className="max-w-[220px] truncate px-3 py-2 font-mono text-stone-700 dark:text-stone-200" title={tensor.name}>{tensor.name}</td>
                <td className="px-3 py-2 text-stone-500">{tensor.dtype ?? '—'}</td>
                <td className="px-3 py-2 font-mono text-stone-500">[{tensor.shape.join(', ')}]</td>
                <td className="px-3 py-2 text-right tabular-nums text-stone-500">{formatSize(tensor.byte_count ?? 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ArtifactTable({ result, t }: { result: RPPResultsInspectResponse; t: Translator }) {
  if (result.artifacts.length === 0) {
    return <p className="text-sm text-stone-500">{t('training.rppResults.noArtifactsRecorded')}</p>;
  }
  return (
    <div className="overflow-hidden rounded-lg border border-stone-200 dark:border-stone-800">
      <table className="w-full text-left text-xs">
        <thead className="bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.name')}</th>
            <th className="px-3 py-2 font-medium">{t('training.rppResults.role')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('training.rppResults.size')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
          {result.artifacts.map((artifact) => (
            <tr key={`${artifact.role}:${artifact.name}`}>
              <td className="max-w-[260px] truncate px-3 py-2 font-mono text-stone-700 dark:text-stone-200" title={artifact.name ?? ''}>
                {artifact.name ?? '—'}
              </td>
              <td className="px-3 py-2 text-stone-500">{artifact.role ?? '—'}</td>
              <td className="px-3 py-2 text-right tabular-nums text-stone-500">
                {artifact.size_bytes != null ? formatSize(artifact.size_bytes) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-stone-50 px-3 py-2 dark:bg-stone-900">
      <div className="text-[11px] uppercase tracking-wide text-stone-500">{label}</div>
      <div className="mt-1 truncate text-sm font-medium text-stone-900 dark:text-stone-100" title={value}>{value}</div>
    </div>
  );
}

function EmptyPanel() {
  const t = useT();
  return (
    <div className="rounded-lg border border-dashed border-stone-300 px-6 py-12 text-center dark:border-stone-700">
      <Brain className="mx-auto mb-3 text-stone-400" size={28} />
      <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">{t('training.rppResults.emptyTitle')}</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-stone-500">
        {t('training.rppResults.emptyDescription')}
      </p>
    </div>
  );
}

function MissingPanel({ peerId, warnings }: { peerId: string; warnings: string[] }) {
  const t = useT();
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-5 py-4 text-sm text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-200">
      <div className="font-medium">{t('training.rppResults.missingTitle')}</div>
      <div className="mt-1 text-xs">
        {t('training.rppResults.missingDescription', { peer: peerId })}
      </div>
      {warnings.length > 0 && <div className="mt-2 text-xs opacity-80">{warnings.join(', ')}</div>}
    </div>
  );
}

function InlineIssue({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
      <AlertTriangle size={16} className="mt-0.5 shrink-0" />
      <span>{message}</span>
    </div>
  );
}

type DatasetRow = Record<string, unknown> & { key: string };

function extractRows(data: Record<string, unknown> | undefined, key: string): DatasetRow[] {
  const raw = data?.[key];
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((item): item is Record<string, unknown> => item != null && typeof item === 'object' && !Array.isArray(item))
    .map((item) => ({
      ...item,
      key: String(item.key ?? item.name ?? item.category ?? 'Unknown'),
    }));
}

function numericValue(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}

function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat().format(value);
}

function formatMoney(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

function formatSeconds(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatTimestamp(value: number | null | undefined): string {
  if (value == null) return '—';
  return new Date(value).toLocaleString();
}

function formatPercent(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  const normalized = value > 1 ? value : value * 100;
  return `${normalized.toFixed(0)}%`;
}

function formatDecimal(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(3);
}

function deviceLabel(peer: TrustedPeer, t: Translator): string {
  const name = peer.display_name?.trim() || t('training.common.device');
  return `${name} · ${peer.peer_id.slice(0, 12)}${peer.last_seen_at ? '' : ` · ${t('training.common.notSeenYet')}`}`;
}

function formatRPPError(err: unknown, t: Translator): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') {
      if (detail.includes('invalid_path_component') || detail.includes('..')) {
        return t('training.rppResults.invalidPeerPath');
      }
      return detail;
    }
    const message = detail?.error?.message ?? detail?.message;
    if (typeof message === 'string') return message;
  }
  return err instanceof Error ? err.message : t('training.rppResults.errorInspect');
}
