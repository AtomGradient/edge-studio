// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  AlertTriangle,
  Braces,
  CheckCircle2,
  CircleHelp,
  FileJson,
  FileSearch,
  Fingerprint,
  Hash,
  Layers,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { parseNeuralImprintArtifact } from '@/api/endpoints';
import { listNeuralImprintArtifacts, type NeuralImprintArtifactSource } from '@/api/mesh';
import type {
  NeuralImprintCompatibility,
  NeuralImprintInspectResponse,
  NeuralImprintTensorInfo,
} from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { useT } from '@/i18n';
import { useModelStore } from '@/stores/modelStore';
import { cn, formatSize } from '@/lib/utils';

type Translator = ReturnType<typeof useT>;

export default function NeuralImprintInspector() {
  const t = useT();
  const currentModel = useModelStore((s) => s.currentModel);
  const [artifactPath, setArtifactPath] = useState('');
  const [sidecarPath, setSidecarPath] = useState('');
  const [selectedArtifactId, setSelectedArtifactId] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [artifact, setArtifact] = useState<NeuralImprintInspectResponse | null>(null);

  const artifactsQ = useQuery({
    queryKey: ['neural-imprint', 'artifacts', 'inspector'],
    queryFn: () => listNeuralImprintArtifacts(false),
  });

  const registryArtifacts = useMemo(
    () => (artifactsQ.data?.artifacts ?? []).filter((item) => item.valid && item.artifact_path),
    [artifactsQ.data?.artifacts],
  );

  const compatibility = artifact?.compatibility ?? null;
  const parsedProfileBody = useMemo(
    () => stringifyUnknown(artifact?.summary.profile_body),
    [artifact?.summary.profile_body],
  );
  const parsedToolSchema = useMemo(
    () => stringifyUnknown(artifact?.summary.tool_schema),
    [artifact?.summary.tool_schema],
  );

  const parseArtifact = async (paths?: { artifactPath: string; sidecarPath?: string }) => {
    const path = (paths?.artifactPath ?? artifactPath).trim();
    const sidecar = (paths?.sidecarPath ?? sidecarPath).trim();
    if (!path) {
      setError(t('training.neuralImprint.errorPathRequired'));
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await parseNeuralImprintArtifact({
        path,
        sidecar_path: sidecar || undefined,
        current_model_id: currentModel?.model_name,
      });
      setArtifact(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('training.neuralImprint.errorParse'));
    } finally {
      setLoading(false);
    }
  };

  const applyRegistryArtifact = (item: NeuralImprintArtifactSource, shouldParse = true) => {
    const nextArtifactPath = item.artifact_path ?? '';
    const nextSidecarPath = item.sidecar_path ?? '';
    setSelectedArtifactId(item.artifact_id ?? item.artifact_sha256 ?? nextArtifactPath);
    setArtifactPath(nextArtifactPath);
    setSidecarPath(nextSidecarPath);
    if (shouldParse && nextArtifactPath) {
      void parseArtifact({
        artifactPath: nextArtifactPath,
        sidecarPath: nextSidecarPath,
      });
    }
  };

  useEffect(() => {
    if (selectedArtifactId || artifactPath || registryArtifacts.length === 0) return;
    applyRegistryArtifact(registryArtifacts[0], true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [registryArtifacts, selectedArtifactId, artifactPath]);

  return (
    <div className="space-y-5 pb-12">
      <PageHeader
        title={t('training.neuralImprint.title')}
        description={t('training.neuralImprint.description')}
        actions={(
          <button
            type="button"
            onClick={() => { void parseArtifact(); }}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
          >
            <RefreshCw size={15} className={loading ? 'animate-spin' : ''} />
            {t('training.neuralImprint.parse')}
          </button>
        )}
      />

      <section className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between gap-3">
            <span className="text-xs font-medium uppercase tracking-wide text-stone-500">
              {t('training.neuralImprint.availableArtifacts')}
            </span>
            <button
              type="button"
              onClick={() => artifactsQ.refetch()}
              disabled={artifactsQ.isFetching}
              className="inline-flex items-center gap-1 rounded-md border border-stone-200 px-2 py-1 text-xs font-medium text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900"
            >
              <RefreshCw size={12} className={artifactsQ.isFetching ? 'animate-spin' : ''} />
              {t('common.refresh')}
            </button>
          </div>
          {registryArtifacts.length === 0 ? (
            <div className="rounded-lg border border-dashed border-stone-200 px-3 py-3 text-sm text-stone-500 dark:border-stone-800">
              {artifactsQ.isFetching
                ? t('training.neuralImprint.scanningRegistry')
                : t('training.neuralImprint.noArtifacts')}
            </div>
          ) : (
            <select
              value={selectedArtifactId}
              onChange={(event) => {
                const selected = registryArtifacts.find((item) => {
                  const id = item.artifact_id ?? item.artifact_sha256 ?? item.artifact_path;
                  return id === event.target.value;
                });
                if (selected) applyRegistryArtifact(selected, true);
              }}
              className="w-full rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
            >
              {registryArtifacts.map((item) => {
                const id = item.artifact_id ?? item.artifact_sha256 ?? item.artifact_path;
                return (
                  <option key={id} value={id}>
                    {item.artifact_id ?? shortPath(item.artifact_path)} · {item.base_model_id ?? t('training.common.unknownModel')} · {item.prefix_token_count ?? '?'} {t('training.common.tokensUnit')} · {formatRegistryTime(item.mtime ?? item.created_at, t)}
                  </option>
                );
              })}
            </select>
          )}
          {artifactsQ.isError && (
            <p className="mt-2 text-xs text-amber-600 dark:text-amber-300">
              {t('training.neuralImprint.registryUnavailable')}
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(220px,0.35fr)]">
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-stone-500">
              {t('training.neuralImprint.artifactPath')}
            </span>
            <input
              value={artifactPath}
              onChange={(event) => {
                setArtifactPath(event.target.value);
                setSelectedArtifactId('');
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter') parseArtifact();
              }}
              placeholder="/path/to/neural_imprint.safetensors"
              autoComplete="off"
              className="w-full rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium uppercase tracking-wide text-stone-500">
              {t('training.neuralImprint.sidecarPath')}
            </span>
            <input
              value={sidecarPath}
              onChange={(event) => {
                setSidecarPath(event.target.value);
                setSelectedArtifactId('');
              }}
              placeholder={t('training.neuralImprint.autoSidecarPlaceholder')}
              autoComplete="off"
              className="w-full rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
            />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-stone-500">
          <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-900">
            <ShieldCheck size={13} />
            {t('training.neuralImprint.headerOnly')}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-900">
            <FileSearch size={13} />
            {t('training.neuralImprint.noTensorLoad')}
          </span>
          {currentModel && (
            <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-1 dark:bg-stone-900">
              <Fingerprint size={13} />
              {t('training.neuralImprint.compareModel', { model: currentModel.model_name })}
            </span>
          )}
        </div>
      </section>

      {error && <InlineIssue message={error} />}

      {!artifact ? (
        <EmptyPanel />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={<FileJson size={16} />}
              label={t('training.neuralImprint.artifactMetric')}
              value={artifact.artifact_name}
              hint={artifact.artifact_size_bytes ? formatSize(artifact.artifact_size_bytes) : t('training.common.uploadedArtifact')}
            />
            <MetricCard
              icon={<Layers size={16} />}
              label={t('training.neuralImprint.prefixMetric')}
              value={artifact.summary.prefix_token_count?.toLocaleString() ?? t('training.common.unknown')}
              hint={t('training.neuralImprint.prefixHint', { tensors: artifact.tensor_count, size: formatSize(artifact.header_size_bytes) })}
            />
            <MetricCard
              icon={<Hash size={16} />}
              label={t('training.neuralImprint.hashesMetric')}
              value={String(artifact.summary.hashes.length)}
              hint={artifact.sidecar_found ? t('training.neuralImprint.headerPlusSidecar') : t('training.neuralImprint.headerOnlyMetric')}
            />
            <MetricCard
              icon={<StatusIcon status={compatibility?.status} />}
              label={t('training.neuralImprint.compatibilityMetric')}
              value={compatibility?.status ?? t('training.common.unknown')}
              hint={compatibility?.message ?? t('training.neuralImprint.noCompatibilityReport')}
              tone={compatibilityTone(compatibility)}
            />
          </div>

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
            <section className="space-y-5">
              <InfoPanel title={t('training.neuralImprint.metadata')} icon={<FileJson size={16} />}>
                <FieldRow label={t('training.neuralImprint.modelId')} value={artifact.summary.model_id ?? t('training.common.unknown')} />
                <FieldRow label={t('training.neuralImprint.created')} value={formatValue(artifact.summary.created_at, t)} />
                <FieldRow label={t('training.neuralImprint.sidecar')} value={artifact.sidecar_found ? artifact.sidecar_path ?? t('training.common.attachedUpload') : t('training.common.notFound')} />
                <FieldRow label={t('training.neuralImprint.path')} value={artifact.artifact_path ?? t('training.common.uploadedArtifact')} mono />
              </InfoPanel>

              <InfoPanel title={t('training.neuralImprint.compatibilityChecks')} icon={<ShieldCheck size={16} />}>
                <CompatibilityBlock compatibility={compatibility} t={t} />
              </InfoPanel>

              <InfoPanel title={t('training.neuralImprint.hashesMetric')} icon={<Hash size={16} />}>
                {artifact.summary.hashes.length === 0 ? (
                  <p className="text-sm text-stone-500">{t('training.neuralImprint.noHashFields')}</p>
                ) : (
                  <div className="overflow-hidden rounded-lg border border-stone-200 dark:border-stone-800">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
                        <tr>
                          <th className="px-3 py-2 font-medium">{t('training.neuralImprint.name')}</th>
                          <th className="px-3 py-2 font-medium">{t('training.neuralImprint.value')}</th>
                          <th className="px-3 py-2 font-medium">{t('training.neuralImprint.source')}</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
                        {artifact.summary.hashes.map((entry) => (
                          <tr key={`${entry.source}:${entry.name}`}>
                            <td className="px-3 py-2 font-medium text-stone-700 dark:text-stone-200">{entry.name}</td>
                            <td className="max-w-[260px] truncate px-3 py-2 font-mono text-stone-500" title={entry.value}>{entry.value}</td>
                            <td className="px-3 py-2 text-stone-500">{entry.source}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </InfoPanel>
            </section>

            <section className="space-y-5">
              <InfoPanel title={t('training.neuralImprint.profileBody')} icon={<Braces size={16} />}>
                <JsonBlock value={parsedProfileBody || t('training.neuralImprint.noProfileBody')} />
              </InfoPanel>

              <InfoPanel title={t('training.neuralImprint.toolSchema')} icon={<FileSearch size={16} />}>
                <JsonBlock value={parsedToolSchema || t('training.neuralImprint.noToolSchema')} />
              </InfoPanel>

              <InfoPanel title={t('training.neuralImprint.tensorHeader')} icon={<Layers size={16} />}>
                <TensorTable tensors={artifact.tensors} t={t} />
              </InfoPanel>
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
  tone = 'stone',
}: {
  icon: ReactNode;
  label: string;
  value: string;
  hint: string;
  tone?: 'stone' | 'green' | 'red' | 'amber';
}) {
  const toneClass = {
    stone: 'text-stone-600 dark:text-stone-300',
    green: 'text-emerald-600 dark:text-emerald-300',
    red: 'text-red-600 dark:text-red-300',
    amber: 'text-amber-600 dark:text-amber-300',
  }[tone];
  return (
    <div className="rounded-lg border border-stone-200 bg-white px-4 py-3 dark:border-stone-800 dark:bg-stone-950">
      <div className={cn('mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide', toneClass)}>
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
    <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-3 border-b border-stone-100 py-2 text-sm last:border-0 dark:border-stone-800">
      <span className="text-stone-500">{label}</span>
      <span className={cn('truncate text-stone-900 dark:text-stone-100', mono && 'font-mono text-xs')} title={value}>
        {value}
      </span>
    </div>
  );
}

function CompatibilityBlock({ compatibility, t }: { compatibility: NeuralImprintCompatibility | null; t: Translator }) {
  if (!compatibility) {
    return <p className="text-sm text-stone-500">{t('training.neuralImprint.noCompatibilityReport')}</p>;
  }
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 text-sm">
        <StatusIcon status={compatibility.status} />
        <div>
          <div className="font-medium capitalize text-stone-900 dark:text-stone-100">{compatibility.status}</div>
          <div className="text-stone-500">{compatibility.message}</div>
        </div>
      </div>
      {compatibility.checks.length > 0 && (
        <div className="space-y-2">
          {compatibility.checks.map((check) => (
            <div key={check.name} className="rounded-lg bg-stone-50 px-3 py-2 text-xs dark:bg-stone-900">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-medium text-stone-700 dark:text-stone-200">{check.name}</span>
                <span className={cn(
                  'rounded-full px-2 py-0.5',
                  check.matched === true && 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300',
                  check.matched === false && 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300',
                  check.matched == null && 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300',
                )}>
                  {check.matched === true ? t('training.common.match') : check.matched === false ? t('training.common.mismatch') : t('training.common.unknown')}
                </span>
              </div>
              <div className="truncate font-mono text-stone-500" title={check.expected ?? ''}>{t('training.common.expected')}: {check.expected ?? '—'}</div>
              <div className="truncate font-mono text-stone-500" title={check.actual ?? ''}>{t('training.common.actual')}: {check.actual ?? '—'}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TensorTable({ tensors, t }: { tensors: NeuralImprintTensorInfo[]; t: Translator }) {
  if (tensors.length === 0) {
    return <p className="text-sm text-stone-500">{t('training.neuralImprint.noTensorEntries')}</p>;
  }
  return (
    <div className="max-h-[360px] overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 font-medium">{t('training.neuralImprint.tensor')}</th>
            <th className="px-3 py-2 font-medium">{t('training.neuralImprint.dtype')}</th>
            <th className="px-3 py-2 font-medium">{t('training.neuralImprint.shape')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('training.neuralImprint.bytes')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
          {tensors.map((tensor) => (
            <tr key={tensor.name}>
              <td className="max-w-[280px] truncate px-3 py-2 font-mono text-stone-700 dark:text-stone-200" title={tensor.name}>
                {tensor.name}
              </td>
              <td className="px-3 py-2 text-stone-500">{tensor.dtype ?? '—'}</td>
              <td className="px-3 py-2 font-mono text-stone-500">[{tensor.shape.join(', ')}]</td>
              <td className="px-3 py-2 text-right tabular-nums text-stone-500">
                {tensor.byte_count != null ? formatSize(tensor.byte_count) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JsonBlock({ value }: { value: string }) {
  return (
    <pre className="max-h-[320px] overflow-auto whitespace-pre-wrap break-words rounded-lg bg-stone-950 p-3 text-xs leading-relaxed text-stone-100">
      {value}
    </pre>
  );
}

function EmptyPanel() {
  const t = useT();
  return (
    <div className="rounded-lg border border-dashed border-stone-300 px-6 py-12 text-center dark:border-stone-700">
      <FileSearch className="mx-auto mb-3 text-stone-400" size={28} />
      <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">{t('training.neuralImprint.emptyTitle')}</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-stone-500">
        {t('training.neuralImprint.emptyDescription')}
      </p>
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

function StatusIcon({ status }: { status?: string | null }) {
  if (status === 'compatible') return <CheckCircle2 size={16} className="text-emerald-500" />;
  if (status === 'incompatible') return <XCircle size={16} className="text-red-500" />;
  return <CircleHelp size={16} className="text-amber-500" />;
}

function compatibilityTone(compatibility: NeuralImprintCompatibility | null): 'stone' | 'green' | 'red' | 'amber' {
  if (compatibility?.status === 'compatible') return 'green';
  if (compatibility?.status === 'incompatible') return 'red';
  if (compatibility?.status === 'unknown') return 'amber';
  return 'stone';
}

function shortPath(path?: string | null): string {
  if (!path) return 'neural_imprint.safetensors';
  const parts = path.split('/');
  return parts.slice(-2).join('/');
}

function formatRegistryTime(value: string | number | null | undefined, t: Translator): string {
  if (value == null || value === '') return t('training.common.unknownTime');
  if (typeof value === 'string') return value;
  let unixSeconds = value;
  if (value < 1_000_000_000) {
    unixSeconds = value + 978_307_200; // Apple CFAbsoluteTime -> Unix seconds.
  }
  const ms = unixSeconds < 10_000_000_000 ? unixSeconds * 1000 : unixSeconds;
  return new Date(ms).toLocaleString();
}

function stringifyUnknown(value: unknown): string {
  if (value == null || value === '') return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function formatValue(value: unknown, t: Translator): string {
  if (value == null || value === '') return t('training.common.unknown');
  return String(value);
}
