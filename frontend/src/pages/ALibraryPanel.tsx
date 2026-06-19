// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useMemo, useState, type ChangeEvent, type ReactNode } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  FileJson,
  Layers,
  Pencil,
  Play,
  RefreshCw,
  ShieldCheck,
  Upload,
  X,
  XCircle,
} from 'lucide-react';
import {
  generateALibrary,
  inspectALibrary,
  listALibraryHistory,
  refineALibraryDomainDescription,
  selectALibrary,
  suggestALibraryDirections,
  validateALibraryYaml,
} from '@/api/endpoints';
import type {
  ALibraryArtifactInfo,
  ALibraryDirectionRepairContext,
  ALibraryDirectionSuggestion,
  ALibraryGenerateRequest,
  ALibraryHealthReport,
  ALibraryHistoryItem,
  ALibraryInspectResponse,
  ALibrarySelectionResponse,
  ALibraryValidateYamlResponse,
} from '@/api/types';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { PageHeader } from '@/components/layout/PageHeader';
import { useModelStore } from '@/stores/modelStore';
import { usePageAskStore } from '@/stores/pageAskStore';
import { useT } from '@/i18n';
import { cn, formatSize } from '@/lib/utils';

const DEFAULT_LIBRARY_PATH = '~/.edgestudio/a_libraries/qwen35_9b_layer11';
const KNOWN_LIBRARY_PATHS = [
  {
    labelKey: 'training.aLibrary.knownQwen9BLabel',
    path: '~/.edgestudio/a_libraries/qwen35_9b_layer11',
    hintKey: 'training.aLibrary.knownQwen9BHint',
  },
];

type Translator = ReturnType<typeof useT>;
type TabId = 'overview' | 'history' | 'directions';

type DirectionDraft = {
  name: string;
  description: string;
  domain: string;
  positive: string[];
  negative: string[];
};

export default function ALibraryPanel() {
  const t = useT();
  const currentModel = useModelStore((s) => s.currentModel);
  const setPageAskContext = usePageAskStore((s) => s.setPageAskContext);
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [path, setPath] = useState(DEFAULT_LIBRARY_PATH);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ALibraryInspectResponse | null>(null);
  const [history, setHistory] = useState<ALibraryHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selection, setSelection] = useState<ALibrarySelectionResponse | null>(null);
  const [selectionLoading, setSelectionLoading] = useState(false);
  const [generateTaskId, setGenerateTaskId] = useState<string | null>(null);
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [advancedSweep, setAdvancedSweep] = useState(true);
  const [advancedLayersText, setAdvancedLayersText] = useState('');
  const [advancedPooling, setAdvancedPooling] = useState<'last_real' | 'mean'>('last_real');
  const [directionMode, setDirectionMode] = useState<'builtin' | 'custom'>('builtin');
  const [customDirectionSetId, setCustomDirectionSetId] = useState('');
  const [customYamlName, setCustomYamlName] = useState<string | null>(null);
  const [customValidation, setCustomValidation] = useState<ALibraryValidateYamlResponse | null>(null);
  const [customYamlError, setCustomYamlError] = useState<string | null>(null);
  const [customYamlLoading, setCustomYamlLoading] = useState(false);
  const [drafts, setDrafts] = useState<DirectionDraft[]>(() => makeInitialDirectionDrafts());
  const [domainDescription, setDomainDescription] = useState('');
  const [suggestCount, setSuggestCount] = useState(10);
  const [suggestLoading, setSuggestLoading] = useState(false);
  const [suggestError, setSuggestError] = useState<string | null>(null);
  const [refineLoading, setRefineLoading] = useState(false);
  const [refineError, setRefineError] = useState<string | null>(null);
  const [editorOpen, setEditorOpen] = useState(false);

  const sortedReports = useMemo(
    () => [...(result?.health_reports ?? [])].sort((a, b) => (a.layer_idx ?? 0) - (b.layer_idx ?? 0)),
    [result?.health_reports],
  );
  const repairContext = useMemo(() => buildDirectionRepairContext(result), [result]);
  const layerOverridePreview = useMemo(
    () => parseLayerOverride(advancedLayersText, currentModel?.num_layers),
    [advancedLayersText, currentModel?.num_layers],
  );
  const advancedGenerationSummary = useMemo(
    () => formatAdvancedGenerationSummary(layerOverridePreview, advancedSweep, advancedPooling),
    [advancedPooling, advancedSweep, layerOverridePreview],
  );

  // ── data loaders ──
  const loadLibrary = async (nextPath?: string) => {
    const trimmed = (nextPath ?? path).trim();
    if (!trimmed) { setError(t('training.aLibrary.errorPathRequired')); return; }
    setLoading(true); setError(null);
    try { setResult(await inspectALibrary(trimmed)); }
    catch (err) { setError(err instanceof Error ? err.message : t('training.aLibrary.errorInspect')); }
    finally { setLoading(false); }
  };

  const refreshSelection = async () => {
    if (!currentModel) { setSelection(null); return; }
    setSelectionLoading(true);
    try { setSelection(await selectALibrary({ model_id: currentModel.model_id })); }
    catch (err) {
      setSelection({
        ok: false, schema_version: 'edgestudio.a_library_selection.v1', status: 'error',
        model: { model_name: currentModel.model_name, model_dir: currentModel.model_dir, model_family: null, hidden_size: currentModel.hidden_size, layer_count: currentModel.num_layers },
        selected: null, candidates: [], reasons: [err instanceof Error ? err.message : 'selection_failed'], recommended_action: 'load_or_generate_a_library',
      });
    } finally { setSelectionLoading(false); }
  };

  const refreshHistory = async () => {
    setHistoryLoading(true); setHistoryError(null);
    try { setHistory((await listALibraryHistory(80)).items); }
    catch (err) { setHistoryError(err instanceof Error ? err.message : t('training.aLibrary.errorLoadHistory')); }
    finally { setHistoryLoading(false); }
  };

  useEffect(() => { void refreshSelection(); }, [currentModel?.model_id]);
  useEffect(() => { void refreshHistory(); }, []);
  useEffect(() => {
    setPageAskContext('/a-library', buildALibraryAskContext(result, selection, history));
    return () => setPageAskContext('/a-library', null);
  }, [history, result, selection, setPageAskContext]);

  // ── generation ──
  const startGeneration = async () => {
    if (!currentModel) { setGenerateError(t('training.aLibrary.errorLoadModelFirst')); return; }
    if (directionMode === 'custom' && !customValidation?.stored_path) { setGenerateError(t('training.aLibrary.errorSelectValidDirectionSet')); return; }
    const layerOverride = parseLayerOverride(advancedLayersText, currentModel.num_layers);
    if (layerOverride.error) { setGenerateError(layerOverride.error); return; }
    if (!advancedSweep && layerOverride.layers.length === 0 && currentModel.num_layers <= 23) {
      setGenerateError(`Sweep off with no layer override uses backend layer 23, but this model has ${currentModel.num_layers} layers. Enter a valid layer override.`);
      return;
    }
    setGenerateError(null);
    const directionPayload = directionMode === 'custom' && customValidation?.stored_path
      ? { yaml_path: customValidation.stored_path, direction_set_id: customValidation.direction_set_id }
      : { direction_set_id: 'directions_a' };
    const generationPayload: ALibraryGenerateRequest = {
      model_path: currentModel.model_dir,
      sweep: advancedSweep,
      pooling: advancedPooling,
      ...directionPayload,
    };
    if (layerOverride.layers.length > 0) generationPayload.layers = layerOverride.layers;
    try {
      const response = await generateALibrary(generationPayload);
      setGenerateTaskId(response.task_id);
    } catch (err) { setGenerateError(err instanceof Error ? err.message : t('training.aLibrary.errorStartGeneration')); }
  };

  const handleGenerationComplete = (payload: unknown) => {
    const generated = payload as { output_dir?: string; inspection?: ALibraryInspectResponse } | null;
    if (generated?.output_dir) setPath(generated.output_dir);
    if (generated?.inspection) setResult(generated.inspection);
    else if (generated?.output_dir) void loadLibrary(generated.output_dir);
    void refreshSelection(); void refreshHistory();
    setActiveTab('overview');
  };

  // ── custom yaml upload ──
  const handleDirectionYamlFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const setId = customDirectionSetId.trim() || inferDirectionSetId(file.name);
    setCustomDirectionSetId(setId); setCustomYamlName(file.name);
    setCustomValidation(null); setCustomYamlError(null); setCustomYamlLoading(true);
    try {
      const validation = await validateALibraryYaml({ content: await file.text(), direction_set_id: setId, persist: true });
      setCustomValidation(validation);
      if (!validation.ok) setCustomYamlError(validation.errors.map((e) => String(e.code ?? 'invalid_yaml')).join(', '));
    } catch (err) { setCustomYamlError(err instanceof Error ? err.message : t('training.aLibrary.errorValidateDirectionSet')); }
    finally { setCustomYamlLoading(false); event.target.value = ''; }
  };

  const confirmCustomDirectionSet = () => {
    if (customValidation?.ok && customValidation.stored_path) { setDirectionMode('custom'); setCustomYamlError(null); }
  };

  // ── form editor validation ──
  const validateDraftDirectionSet = async (): Promise<boolean> => {
    const setId = customDirectionSetId.trim() || 'custom_direction_set';
    setCustomDirectionSetId(setId); setCustomYamlName(t('training.aLibrary.formEditorYamlName'));
    setCustomValidation(null); setCustomYamlError(null); setCustomYamlLoading(true);
    try {
      const validation = await validateALibraryYaml({ content: serializeDirectionDraftsToYaml(setId, drafts), direction_set_id: setId, persist: true });
      setCustomValidation(validation);
      if (!validation.ok) {
        setCustomYamlError(validation.errors.map((e) => String(e.code ?? 'invalid_yaml')).join(', '));
        return false;
      }
      return true;
    } catch (err) {
      setCustomYamlError(err instanceof Error ? err.message : t('training.aLibrary.errorValidateDirectionSet'));
      return false;
    } finally { setCustomYamlLoading(false); }
  };

  // ── model suggest ──
  const suggestDirections = async (nextRepairContext?: ALibraryDirectionRepairContext | null) => {
    if (!currentModel) { setSuggestError(t('training.aLibrary.errorLoadModelFirst')); return; }
    if (!domainDescription.trim()) { setSuggestError(t('training.aLibrary.errorDomainRequired')); return; }
    setSuggestError(null); setSuggestLoading(true);
    try {
      const response = await suggestALibraryDirections({
        domain_description: domainDescription.trim(),
        target_count: suggestCount,
        model_id: currentModel.model_id,
        repair_context: nextRepairContext ?? undefined,
      });
      setDrafts(response.directions.map(suggestionToDraft));
      if (!customDirectionSetId.trim()) {
        setCustomDirectionSetId(nextRepairContext?.prev_direction_set_id || inferDirectionSetId(domainDescription));
      }
    } catch (err) { setSuggestError(err instanceof Error ? err.message : t('training.aLibrary.errorSuggestDirections')); }
    finally { setSuggestLoading(false); }
  };

  const refineDomainDescription = async () => {
    if (!currentModel) { setRefineError(t('training.aLibrary.errorLoadModelFirst')); return; }
    if (!domainDescription.trim()) { setRefineError(t('training.aLibrary.errorDomainRequired')); return; }
    setRefineError(null); setRefineLoading(true);
    try {
      const response = await refineALibraryDomainDescription({ domain_description: domainDescription.trim(), model_id: currentModel.model_id });
      setDomainDescription(response.refined_description);
    } catch (err) { setRefineError(err instanceof Error ? err.message : t('training.aLibrary.errorRefineDomainDescription')); }
    finally { setRefineLoading(false); }
  };

  // ── active direction set label ──
  const activeDirectionLabel = directionMode === 'custom' && customValidation?.ok
    ? `${customValidation.direction_set_id} (${customValidation.coverage.direction_count} directions)`
    : 'directions_a (50 directions, built-in)';

  // ── render ──
  return (
    <div className="space-y-5 pb-12">
      <PageHeader
        title={t('training.aLibrary.title')}
        description={t('training.aLibrary.description')}
      />

      {/* ── Top action bar: model status + generate ── */}
      <section className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3 min-w-0">
            <Cpu size={18} className="shrink-0 text-stone-400" />
            {currentModel ? (
              <div className="min-w-0">
                <div className="truncate text-sm font-medium text-stone-900 dark:text-stone-100">{currentModel.model_name}</div>
                <div className="text-xs text-stone-500">hidden={currentModel.hidden_size} · layers={currentModel.num_layers}</div>
              </div>
            ) : (
              <span className="text-sm text-stone-500">{t('training.aLibrary.loadModelHint')}</span>
            )}
            {selection && (
              <span className={cn(
                'inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
                selection.ok ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300'
                  : 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300',
              )}>
                {selection.ok ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />}
                {selection.ok ? 'Match' : 'No match'}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="mr-2 max-w-[260px] truncate text-xs text-stone-400" title={activeDirectionLabel}>
              {activeDirectionLabel}
            </div>
            <button type="button" onClick={() => { void refreshSelection(); }} disabled={!currentModel || selectionLoading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900">
              <RefreshCw size={13} className={selectionLoading ? 'animate-spin' : ''} />
              Check
            </button>
            <button type="button" onClick={() => { void startGeneration(); }}
              disabled={!currentModel || !!generateTaskId || (directionMode === 'custom' && !customValidation?.stored_path)}
              className="inline-flex items-center gap-1.5 rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200">
              <Play size={13} />
              {t('training.aLibrary.generate')}
            </button>
          </div>
        </div>
        <div className="mt-3 border-t border-stone-100 pt-3 dark:border-stone-800">
          <button type="button" onClick={() => setAdvancedOpen((open) => !open)}
            className="flex w-full items-center justify-between gap-3 rounded-lg px-2 py-1.5 text-left text-xs text-stone-500 hover:bg-stone-50 dark:hover:bg-stone-900">
            <span className="inline-flex items-center gap-1.5 font-medium text-stone-700 dark:text-stone-200">
              <Layers size={13} />
              Advanced generation
            </span>
            <span className="truncate text-stone-400">
              {advancedGenerationSummary} · {advancedOpen ? 'Hide' : 'Show'}
            </span>
          </button>
          {advancedOpen && (
            <div className="mt-3 grid gap-3 md:grid-cols-[minmax(0,1fr)_minmax(180px,220px)_minmax(160px,190px)]">
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-stone-500">Layer override</span>
                <input value={advancedLayersText} onChange={(event) => setAdvancedLayersText(event.target.value)}
                  placeholder="11,19,23,27" autoComplete="off"
                  className={cn(
                    'w-full rounded-lg border bg-white px-3 py-2 font-mono text-sm text-stone-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-100 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-sky-950',
                    layerOverridePreview.error ? 'border-red-300 dark:border-red-800' : 'border-stone-200 dark:border-stone-700',
                  )} />
                <span className={cn('mt-1 block text-xs', layerOverridePreview.error ? 'text-red-600 dark:text-red-400' : 'text-stone-400')}>
                  {layerOverridePreview.error || 'Empty uses default sweep; empty with sweep off uses layer 23.'}
                </span>
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-stone-500">Pooling</span>
                <select value={advancedPooling} onChange={(event) => setAdvancedPooling(event.target.value as 'last_real' | 'mean')}
                  className="w-full rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-sky-950">
                  <option value="last_real">last_real</option>
                  <option value="mean">mean (experimental)</option>
                </select>
              </label>
              <label className="flex items-center justify-between gap-3 rounded-lg border border-stone-200 px-3 py-2 dark:border-stone-700">
                <span>
                  <span className="block text-xs font-medium text-stone-700 dark:text-stone-200">Sweep</span>
                  <span className="block text-xs text-stone-400">Use backend sweep if no override</span>
                </span>
                <input type="checkbox" checked={advancedSweep}
                  onChange={(event) => setAdvancedSweep(event.target.checked)}
                  className="h-4 w-4 rounded border-stone-300 text-stone-900 focus:ring-stone-400 dark:border-stone-600" />
              </label>
            </div>
          )}
        </div>
        {generateError && <div className="mt-3"><InlineIssue message={generateError} /></div>}
      </section>

      {/* ── Tabs ── */}
      <div className="flex gap-1 border-b border-stone-200 dark:border-stone-800">
        {(['overview', 'history', 'directions'] as TabId[]).map((tab) => (
          <button key={tab} type="button" onClick={() => setActiveTab(tab)}
            className={cn(
              'px-4 py-2 text-sm font-medium transition',
              activeTab === tab
                ? 'border-b-2 border-stone-900 text-stone-900 dark:border-stone-100 dark:text-stone-100'
                : 'text-stone-500 hover:text-stone-700 dark:hover:text-stone-300',
            )}>
            {tab === 'overview' ? 'Overview' : tab === 'history' ? `History${history.length ? ` (${history.length})` : ''}` : 'Direction Sets'}
          </button>
        ))}
      </div>

      {/* ── Tab: Overview ── */}
      {activeTab === 'overview' && (
        <>
          {/* Path input (compact) */}
          <div className="flex gap-2">
            <input value={path} onChange={(e) => setPath(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void loadLibrary(); }}
              placeholder={t('training.aLibrary.libraryDirectoryPlaceholder')} autoComplete="off"
              className="flex-1 rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-100 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-sky-950" />
            <button type="button" onClick={() => { void loadLibrary(); }} disabled={loading}
              className="inline-flex items-center gap-1.5 rounded-lg bg-stone-900 px-4 py-2 text-sm font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200">
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
              {t('common.load')}
            </button>
          </div>

          {/* Known locations (compact row) */}
          <div className="flex flex-wrap gap-2">
            {KNOWN_LIBRARY_PATHS.map((item) => (
              <button key={item.path} type="button"
                onClick={() => { setPath(item.path); void loadLibrary(item.path); }}
                className={cn(
                  'rounded-full border px-3 py-1 text-xs transition',
                  path === item.path
                    ? 'border-sky-300 bg-sky-50 text-sky-800 dark:border-sky-700 dark:bg-sky-950/40 dark:text-sky-200'
                    : 'border-stone-200 text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900',
                )}>
                {t(item.labelKey)}
              </button>
            ))}
          </div>

          {error && <InlineIssue message={error} />}

          {!result ? <EmptyPanel /> : (
            <>
              <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
                <MetricCard icon={<Database size={16} />} label={t('training.aLibrary.modelFamily')} value={result.summary.model_family ?? t('training.common.unknown')} hint={t('training.aLibrary.hiddenSizeHint', { hidden: result.summary.hidden_size ?? '?' })} />
                <MetricCard icon={<Layers size={16} />} label={t('training.aLibrary.targetLayer')} value={formatNullable(result.summary.target_layer)} hint={result.summary.selected_reason ?? t('training.aLibrary.noSelectionReason')} />
                <MetricCard icon={<StatusIcon status={result.summary.health_status} />} label={t('training.aLibrary.health')} value={result.summary.health_status ?? t('training.common.unknown')} hint={t('training.aLibrary.reportsHint', { count: result.summary.report_count ?? 0 })} />
                <MetricCard icon={<FileJson size={16} />} label={t('training.aLibrary.artifacts')} value={String(result.summary.artifact_count ?? 0)} hint={t('training.aLibrary.directionsHint', { count: result.summary.n_directions ?? '?' })} />
              </div>

              <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
                <section className="space-y-5">
                  <InfoPanel title={t('training.aLibrary.manifestReadiness')} icon={<ShieldCheck size={16} />}>
                    <div className="mb-3 flex items-center gap-2 text-sm">
                      <span className={cn('inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium',
                        result.manifest.ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300')}>
                        {result.manifest.ready ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
                        {result.manifest.ready ? t('training.common.ready') : t('training.common.notReady')}
                      </span>
                    </div>
                    <div className="space-y-1">
                      {result.manifest.checks.map((check) => (
                        <div key={check.name} className="grid grid-cols-[150px_minmax(0,1fr)_72px] items-center gap-3 border-b border-stone-100 py-2 text-xs last:border-0 dark:border-stone-800">
                          <span className="font-medium text-stone-700 dark:text-stone-200">{check.name}</span>
                          <span className="truncate font-mono text-stone-500" title={String(check.value ?? '')}>{String(check.value ?? '—')}</span>
                          <StatusPill passed={check.passed ?? check.present} t={t} />
                        </div>
                      ))}
                    </div>
                  </InfoPanel>
                  <InfoPanel title={t('training.aLibrary.librarySummary')} icon={<Database size={16} />}>
                    <FieldRow label={t('training.aLibrary.kind')} value={result.summary.library_kind ?? '?'} />
                    <FieldRow label={t('training.aLibrary.directionSet')} value={result.summary.direction_set_id ?? '?'} />
                    <FieldRow label={t('training.aLibrary.pooling')} value={result.summary.pooling ?? '?'} />
                  </InfoPanel>
                  {result.warnings.length > 0 && (
                    <InfoPanel title={t('training.common.warnings')} icon={<AlertTriangle size={16} />}>
                      <div className="flex flex-wrap gap-2">{result.warnings.map((w) => (
                        <span key={w} className="rounded-full bg-amber-100 px-2 py-1 text-xs font-medium text-amber-700 dark:bg-amber-500/10 dark:text-amber-300">{w}</span>
                      ))}</div>
                    </InfoPanel>
                  )}
                </section>
                <section className="space-y-5">
                  <InfoPanel title={t('training.aLibrary.layerHealthReports')} icon={<CheckCircle2 size={16} />}>
                    <HealthReportTable reports={sortedReports} t={t} />
                  </InfoPanel>
                  <InfoPanel title={t('training.aLibrary.artifacts')} icon={<FileJson size={16} />}>
                    <ArtifactTable artifacts={result.artifacts} t={t} />
                  </InfoPanel>
                  <InfoPanel title={t('training.aLibrary.sweepSummary')} icon={<Layers size={16} />}>
                    <SweepSummary result={result} t={t} />
                  </InfoPanel>
                </section>
              </div>
            </>
          )}
        </>
      )}

      {/* ── Tab: History ── */}
      {activeTab === 'history' && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs text-stone-500">{t('training.aLibrary.generatedHistoryHint')}</p>
            <button type="button" onClick={() => { void refreshHistory(); }} disabled={historyLoading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900">
              <RefreshCw size={13} className={historyLoading ? 'animate-spin' : ''} />
              {t('common.refresh')}
            </button>
          </div>
          {historyError && <InlineIssue message={historyError} />}
          {!historyError && history.length === 0 ? (
            <p className="rounded-lg border border-dashed border-stone-200 px-3 py-8 text-center text-sm text-stone-500 dark:border-stone-800">{t('training.aLibrary.noGeneratedHistory')}</p>
          ) : (
            <div className="max-h-[480px] overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
                  <tr>
                    <th className="px-3 py-2 font-medium">{t('training.aLibrary.model')}</th>
                    <th className="px-3 py-2 font-medium">{t('training.aLibrary.directionSet')}</th>
                    <th className="px-3 py-2 font-medium">{t('training.aLibrary.targetLayer')}</th>
                    <th className="px-3 py-2 font-medium">{t('training.aLibrary.health')}</th>
                    <th className="px-3 py-2 font-medium">{t('training.aLibrary.generatedAt')}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
                  {history.map((item) => (
                    <tr key={item.path} className="cursor-pointer hover:bg-stone-50 dark:hover:bg-stone-900/60"
                      onClick={() => { setPath(item.path); void loadLibrary(item.path); setActiveTab('overview'); }}>
                      <td className="px-3 py-2 font-medium text-stone-900 dark:text-stone-100">{item.model_name ?? '?'}</td>
                      <td className="px-3 py-2 font-mono text-stone-500">{item.direction_set_id ?? '—'}</td>
                      <td className="px-3 py-2 tabular-nums text-stone-500">{formatNullable(item.target_layer)}</td>
                      <td className="px-3 py-2"><StatusPill passed={item.health_status == null ? null : item.health_status === 'pass'} label={statusText(item.health_status, t)} t={t} /></td>
                      <td className="px-3 py-2 text-stone-500">{formatTimestamp(item.created_at_unix)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* ── Tab: Direction Sets ── */}
      {activeTab === 'directions' && (
        <section className="space-y-4">
          {/* Compact selector: builtin vs custom */}
          <div className="grid gap-3 sm:grid-cols-2">
            <button type="button" onClick={() => setDirectionMode('builtin')}
              className={cn('rounded-lg border px-4 py-3 text-left transition',
                directionMode === 'builtin'
                  ? 'border-sky-300 bg-sky-50 dark:border-sky-700 dark:bg-sky-950/40'
                  : 'border-stone-200 hover:bg-stone-50 dark:border-stone-800 dark:hover:bg-stone-900')}>
              <div className="text-sm font-semibold text-stone-900 dark:text-stone-100">{t('training.aLibrary.builtinDirectionSet')}</div>
              <div className="mt-1 text-xs text-stone-500">directions_a · 50 directions · 5 domains</div>
            </button>
            <div className={cn('rounded-lg border px-4 py-3 transition',
              directionMode === 'custom'
                ? 'border-sky-300 bg-sky-50 dark:border-sky-700 dark:bg-sky-950/40'
                : 'border-stone-200 dark:border-stone-800')}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm font-semibold text-stone-900 dark:text-stone-100">{t('training.aLibrary.customDirectionSet')}</div>
                  <div className="mt-1 text-xs text-stone-500">
                    {customValidation?.ok
                      ? `${customValidation.direction_set_id} · ${customValidation.coverage.direction_count} directions`
                      : customYamlName
                        ? `${customYamlName} — validating...`
                        : t('training.aLibrary.customDirectionSetHint')}
                  </div>
                </div>
                <div className="flex gap-2">
                  <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-stone-200 px-2.5 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900">
                    <Upload size={13} />
                    YAML
                    <input type="file" accept=".yaml,.yml" className="hidden" disabled={customYamlLoading}
                      onChange={(e) => { void handleDirectionYamlFile(e); }} />
                  </label>
                  <button type="button" onClick={() => setEditorOpen(true)}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 px-2.5 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-900">
                    <Pencil size={13} />
                    Edit
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Direction Set ID */}
          {directionMode === 'custom' && (
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-stone-500">Direction Set ID</span>
              <input value={customDirectionSetId} onChange={(e) => { setCustomDirectionSetId(e.target.value); setCustomValidation(null); }}
                placeholder="my_domain_v1"
                className="w-full max-w-xs rounded-lg border border-stone-200 bg-white px-3 py-2 font-mono text-sm text-stone-900 outline-none focus:border-sky-400 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100" />
            </label>
          )}

          {/* Validation result (compact) */}
          {customValidation && (
            <div className={cn('rounded-lg border px-4 py-3 text-xs',
              customValidation.ok ? 'border-emerald-200 bg-emerald-50 dark:border-emerald-900/60 dark:bg-emerald-950/30'
                : 'border-red-200 bg-red-50 dark:border-red-900/60 dark:bg-red-950/30')}>
              <div className="flex items-center gap-2">
                <StatusPill passed={customValidation.ok} t={t} />
                <span className="font-mono text-stone-600 dark:text-stone-300">{customValidation.direction_set_id}</span>
                <span className="text-stone-500">
                  {customValidation.coverage.direction_count} directions · {customValidation.coverage.sentence_count} sentences
                </span>
                {customValidation.coverage.domains && (
                  <span className="text-stone-400">{formatDomains(customValidation.coverage.domains)}</span>
                )}
              </div>
              {customValidation.ok && !customValidation.stored_path && (
                <button type="button" onClick={confirmCustomDirectionSet}
                  className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200">
                  <CheckCircle2 size={13} />
                  {t('training.aLibrary.useCustomDirectionSet')}
                </button>
              )}
              {customValidation.ok && customValidation.stored_path && directionMode !== 'custom' && (
                <button type="button" onClick={() => setDirectionMode('custom')}
                  className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200">
                  <CheckCircle2 size={13} />
                  Use this direction set
                </button>
              )}
            </div>
          )}
          {customYamlError && <InlineIssue message={customYamlError} />}
        </section>
      )}

      {/* ── Direction Editor Modal ── */}
      {editorOpen && (
        <div className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-black/50 p-4 pt-12">
          <div className="w-full max-w-4xl rounded-2xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-950">
            {/* Modal header */}
            <div className="flex items-center justify-between border-b border-stone-200 px-6 py-4 dark:border-stone-800">
              <div>
                <h2 className="text-lg font-semibold text-stone-900 dark:text-stone-100">Direction Set Editor</h2>
                <p className="mt-0.5 text-xs text-stone-500">{drafts.length} directions · min 10 · each needs 5 positive + 5 negative examples</p>
                <p className="mt-1 max-w-3xl text-xs text-stone-500">
                  {t('training.aLibrary.rppBasisEditorHint')}
                </p>
              </div>
              <button type="button" onClick={() => setEditorOpen(false)}
                className="rounded-lg p-2 text-stone-400 hover:bg-stone-100 dark:hover:bg-stone-900">
                <X size={18} />
              </button>
            </div>

            {/* Model-assisted suggestion bar */}
            <div className="border-b border-stone-200 bg-stone-50 px-6 py-3 dark:border-stone-800 dark:bg-stone-900/40">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
                <div className="flex-1">
                  <label className="mb-1 block text-xs font-medium text-stone-500">Describe your app domain for AI-assisted direction generation</label>
                  <textarea value={domainDescription} onChange={(e) => setDomainDescription(e.target.value)}
                    rows={3}
                    placeholder={t('training.aLibrary.domainDescriptionPlaceholder')}
                    className="w-full resize-y rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none focus:border-sky-400 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100" />
                </div>
                <div className="flex items-center gap-2">
                  <label className="inline-flex items-center gap-1.5 text-xs text-stone-500">
                    Count
                    <input type="number" min={10} max={20} value={suggestCount}
                      onChange={(e) => setSuggestCount(Math.max(10, Math.min(20, Number(e.target.value) || 10)))}
                      className="w-14 rounded border border-stone-200 bg-white px-2 py-1 text-center font-mono text-xs dark:border-stone-700 dark:bg-stone-950" />
                  </label>
                  <button type="button" onClick={() => { void refineDomainDescription(); }} disabled={!currentModel || refineLoading || suggestLoading}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 bg-white px-3 py-2 text-xs font-medium text-stone-700 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-200 dark:hover:bg-stone-900">
                    <Cpu size={13} className={refineLoading ? 'animate-spin' : ''} />
                    {t('training.aLibrary.refineDomainDescription')}
                  </button>
                  <button type="button" onClick={() => { void suggestDirections(); }} disabled={!currentModel || suggestLoading || refineLoading}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-stone-900 px-3 py-2 text-xs font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200">
                    <Cpu size={13} className={suggestLoading ? 'animate-spin' : ''} />
                    {t('training.aLibrary.suggestDirections')}
                  </button>
                </div>
              </div>
              <div className="mt-3 rounded-lg border border-stone-200 bg-white px-3 py-2 dark:border-stone-800 dark:bg-stone-950">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <div className="text-xs font-medium text-stone-700 dark:text-stone-200">
                      {t('training.aLibrary.repairDirectionsTitle')}
                    </div>
                    <div className="mt-0.5 truncate text-xs text-stone-500" title={formatRepairContextSummary(repairContext, t)}>
                      {formatRepairContextSummary(repairContext, t)}
                    </div>
                  </div>
                  <button type="button" onClick={() => { void suggestDirections(repairContext); }}
                    disabled={!currentModel || !repairContext || suggestLoading || refineLoading}
                    className="inline-flex shrink-0 items-center gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800 hover:bg-amber-100 disabled:opacity-50 dark:border-amber-900/70 dark:bg-amber-950/30 dark:text-amber-200 dark:hover:bg-amber-950/50">
                    <RefreshCw size={13} className={suggestLoading && repairContext ? 'animate-spin' : ''} />
                    {t('training.aLibrary.repairDirections')}
                  </button>
                </div>
              </div>
              {refineLoading && (
                <div className="mt-3 flex items-center gap-3 rounded-lg border border-sky-200 bg-sky-50 px-4 py-3 dark:border-sky-900/60 dark:bg-sky-950/30">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
                  <div>
                    <div className="text-sm font-medium text-sky-800 dark:text-sky-200">
                      {t('training.aLibrary.refiningDomainDescriptionWithModel', { model: currentModel?.model_name ?? 'model' })}
                    </div>
                    <div className="text-xs text-sky-600 dark:text-sky-400">
                      {t('training.aLibrary.refineDomainDescriptionLoadingHint')}
                    </div>
                  </div>
                </div>
              )}
              {suggestLoading && (
                <div className="mt-3 flex items-center gap-3 rounded-lg border border-sky-200 bg-sky-50 px-4 py-3 dark:border-sky-900/60 dark:bg-sky-950/30">
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
                  <div>
                    <div className="text-sm font-medium text-sky-800 dark:text-sky-200">
                      Using {currentModel?.model_name ?? 'model'} to generate directions...
                    </div>
                    <div className="text-xs text-sky-600 dark:text-sky-400">
                      This may take 30–60 seconds depending on model size. The model is analyzing your domain and drafting contrastive direction pairs.
                    </div>
                  </div>
                </div>
              )}
              {refineError && <div className="mt-2"><InlineIssue message={refineError} /></div>}
              {suggestError && <div className="mt-2"><InlineIssue message={suggestError} /></div>}
            </div>

            {/* Direction list */}
            <div className="max-h-[60vh] overflow-auto px-6 py-4">
              <div className="space-y-3">
                {drafts.map((draft, idx) => (
                  <details key={`${idx}:${draft.name}`} className="group rounded-lg border border-stone-200 dark:border-stone-800">
                    <summary className="flex cursor-pointer items-center gap-3 px-4 py-3 text-sm">
                      <span className="font-mono text-xs text-stone-400">{idx + 1}</span>
                      <span className="font-medium text-stone-900 dark:text-stone-100">{draft.name || `direction_${idx + 1}`}</span>
                      <span className="text-xs text-stone-500">{draft.description}</span>
                      <span className="ml-auto rounded-full bg-stone-100 px-2 py-0.5 text-[10px] text-stone-500 dark:bg-stone-800">{draft.domain}</span>
                      {drafts.length > 10 && (
                        <button type="button" onClick={(e) => { e.preventDefault(); setDrafts((d) => removeDraftDirection(d, idx)); }}
                          className="rounded p-1 text-stone-400 hover:bg-red-50 hover:text-red-500 dark:hover:bg-red-950/30">
                          <X size={14} />
                        </button>
                      )}
                    </summary>
                    <div className="border-t border-stone-100 px-4 py-3 dark:border-stone-800">
                      <div className="mb-3 grid gap-2 md:grid-cols-3">
                        <label className="block">
                          <span className="mb-1 block text-[11px] font-medium uppercase text-stone-500">Name</span>
                          <input value={draft.name} onChange={(e) => setDrafts((d) => updateDraft(d, idx, { name: e.target.value }))}
                            className="w-full rounded border border-stone-200 bg-white px-2 py-1.5 font-mono text-xs outline-none focus:border-sky-400 dark:border-stone-700 dark:bg-stone-950" />
                        </label>
                        <label className="block">
                          <span className="mb-1 block text-[11px] font-medium uppercase text-stone-500">Description</span>
                          <input value={draft.description} onChange={(e) => setDrafts((d) => updateDraft(d, idx, { description: e.target.value }))}
                            className="w-full rounded border border-stone-200 bg-white px-2 py-1.5 text-xs outline-none focus:border-sky-400 dark:border-stone-700 dark:bg-stone-950" />
                        </label>
                        <label className="block">
                          <span className="mb-1 block text-[11px] font-medium uppercase text-stone-500">Domain</span>
                          <input value={draft.domain} onChange={(e) => setDrafts((d) => updateDraft(d, idx, { domain: e.target.value }))}
                            className="w-full rounded border border-stone-200 bg-white px-2 py-1.5 font-mono text-xs outline-none focus:border-sky-400 dark:border-stone-700 dark:bg-stone-950" />
                        </label>
                      </div>
                      <div className="grid gap-3 lg:grid-cols-2">
                        <ExampleList label="Positive examples" values={draft.positive}
                          onChange={(ei, v) => setDrafts((d) => updateDraftExample(d, idx, 'positive', ei, v))} />
                        <ExampleList label="Negative examples" values={draft.negative}
                          onChange={(ei, v) => setDrafts((d) => updateDraftExample(d, idx, 'negative', ei, v))} />
                      </div>
                    </div>
                  </details>
                ))}
              </div>
            </div>

            {/* Modal footer */}
            <div className="flex items-center justify-between border-t border-stone-200 px-6 py-4 dark:border-stone-800">
              <button type="button" onClick={() => setDrafts((d) => addDraftDirection(d))}
                className="inline-flex items-center gap-1.5 rounded-lg border border-stone-200 px-3 py-2 text-xs font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300">
                <Layers size={13} /> Add direction
              </button>
              <div className="flex gap-2">
                <button type="button" onClick={() => setEditorOpen(false)}
                  className="rounded-lg border border-stone-200 px-4 py-2 text-xs font-medium text-stone-600 hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300">
                  Cancel
                </button>
                <button type="button" disabled={customYamlLoading}
                  onClick={async () => {
                    const ok = await validateDraftDirectionSet();
                    if (ok) setEditorOpen(false);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-stone-900 px-4 py-2 text-xs font-medium text-white hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200">
                  <CheckCircle2 size={13} /> Validate & Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      <ProgressOverlay taskId={generateTaskId} title={t('training.aLibrary.generating')}
        onComplete={handleGenerationComplete} onError={(m) => setGenerateError(m)} onClose={() => setGenerateTaskId(null)} />
    </div>
  );
}

// ── Shared components ──

function MetricCard({ icon, label, value, hint }: { icon: ReactNode; label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg border border-stone-200 bg-white px-4 py-3 dark:border-stone-800 dark:bg-stone-950">
      <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase text-sky-600 dark:text-sky-300">{icon}{label}</div>
      <div className="truncate text-xl font-semibold text-stone-900 dark:text-stone-100" title={value}>{value}</div>
      <div className="mt-1 truncate text-xs text-stone-500" title={hint}>{hint}</div>
    </div>
  );
}

function InfoPanel({ title, icon, children }: { title: string; icon: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-stone-900 dark:text-stone-100">
        <span className="text-stone-500">{icon}</span>{title}
      </div>
      {children}
    </section>
  );
}

function FieldRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="grid grid-cols-[118px_minmax(0,1fr)] gap-3 border-b border-stone-100 py-2 text-sm last:border-0 dark:border-stone-800">
      <span className="text-stone-500">{label}</span>
      <span className={cn('truncate text-stone-900 dark:text-stone-100', mono && 'font-mono text-xs')} title={value}>{value}</span>
    </div>
  );
}

function HealthReportTable({ reports, t }: { reports: ALibraryHealthReport[]; t: Translator }) {
  if (reports.length === 0) return <p className="text-sm text-stone-500">{t('training.aLibrary.noHealthReports')}</p>;
  return (
    <div className="max-h-[520px] overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.layer')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.cosine')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.signal')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.verdict')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
          {reports.map((r) => (
            <tr key={`${r.layer_idx}:${r.report_path}`}>
              <td className="px-3 py-2 font-medium text-stone-900 dark:text-stone-100">{formatNullable(r.layer_idx)}</td>
              <td className="px-3 py-2 tabular-nums text-stone-600 dark:text-stone-300">
                max {formatDecimal(r.max_abs_cos_sim)} · mean {formatDecimal(r.mean_abs_cos_sim)}
              </td>
              <td className="px-3 py-2 tabular-nums text-stone-600 dark:text-stone-300">
                min {formatDecimal(r.min_signal_strength)} · {formatNullable(r.n_pass)}/{formatNullable(r.n_total)}
              </td>
              <td className="px-3 py-2">
                <div className="space-y-1.5">
                  <StatusPill passed={r.verdict === 'pass'} label={statusText(r.verdict, t)} t={t} />
                  <div className="max-w-[280px] text-[11px] leading-snug text-stone-500 dark:text-stone-400">
                    {healthDiagnosis(r, t)}
                  </div>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactTable({ artifacts, t }: { artifacts: ALibraryArtifactInfo[]; t: Translator }) {
  if (artifacts.length === 0) return <p className="text-sm text-stone-500">{t('training.aLibrary.noArtifactsFound')}</p>;
  return (
    <div className="overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
      <table className="w-full text-left text-xs">
        <thead className="bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.file')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.layer')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.shape')}</th>
            <th className="px-3 py-2 text-right font-medium">{t('training.aLibrary.size')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
          {artifacts.map((a) => (
            <tr key={`${a.path}:${a.kind}`}>
              <td className="max-w-[280px] px-3 py-2"><div className="truncate font-mono text-stone-700 dark:text-stone-200" title={a.path}>{a.name}</div></td>
              <td className="px-3 py-2 tabular-nums text-stone-500">{formatNullable(a.layer_idx)}</td>
              <td className="px-3 py-2 font-mono text-stone-500">{formatNullable(a.direction_count)} x {formatNullable(a.hidden_size)}</td>
              <td className="px-3 py-2 text-right tabular-nums text-stone-500">{a.size_bytes != null ? formatSize(a.size_bytes) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SweepSummary({ result, t }: { result: ALibraryInspectResponse; t: Translator }) {
  const rows = result.sweep_summaries.flatMap((s) => s.per_layer);
  if (rows.length === 0) return <p className="text-sm text-stone-500">{t('training.aLibrary.noSweepSummary')}</p>;
  return (
    <div className="max-h-[360px] overflow-auto rounded-lg border border-stone-200 dark:border-stone-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-stone-50 text-stone-500 dark:bg-stone-900 dark:text-stone-400">
          <tr>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.layer')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.maxCos')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.medianSignal')}</th>
            <th className="px-3 py-2 font-medium">{t('training.aLibrary.verdict')}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 dark:divide-stone-800">
          {rows.map((r) => (
            <tr key={`sweep:${r.layer_idx}`}>
              <td className="px-3 py-2 font-medium text-stone-900 dark:text-stone-100">{formatNullable(r.layer_idx)}</td>
              <td className="px-3 py-2 tabular-nums text-stone-500">{formatDecimal(r.max_abs_cos_sim)}</td>
              <td className="px-3 py-2 tabular-nums text-stone-500">{formatDecimal(r.median_signal_strength)}</td>
              <td className="px-3 py-2"><StatusPill passed={r.verdict === 'pass'} label={statusText(r.verdict, t)} t={t} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusIcon({ status }: { status?: string | null }) {
  if (status === 'pass') return <CheckCircle2 size={16} />;
  if (status === 'fail') return <XCircle size={16} />;
  return <AlertTriangle size={16} />;
}

function StatusPill({ passed, label, t }: { passed?: boolean | null; label?: string; t: Translator }) {
  const unknown = passed == null;
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
      passed === true && 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300',
      passed === false && 'bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-300',
      unknown && 'bg-stone-100 text-stone-600 dark:bg-stone-900 dark:text-stone-300')}>
      {passed === true ? <CheckCircle2 size={12} /> : passed === false ? <XCircle size={12} /> : null}
      {label ?? (unknown ? t('training.common.unknown') : passed ? t('training.common.pass') : t('training.common.fail'))}
    </span>
  );
}

function EmptyPanel() {
  const t = useT();
  return (
    <div className="rounded-lg border border-dashed border-stone-300 px-6 py-12 text-center dark:border-stone-700">
      <Database className="mx-auto mb-3 text-stone-400" size={28} />
      <h3 className="text-sm font-semibold text-stone-900 dark:text-stone-100">{t('training.aLibrary.emptyTitle')}</h3>
      <p className="mx-auto mt-1 max-w-md text-sm text-stone-500">{t('training.aLibrary.emptyDescription')}</p>
    </div>
  );
}

function InlineIssue({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
      <AlertTriangle size={16} className="mt-0.5 shrink-0" /><span>{message}</span>
    </div>
  );
}

function ExampleList({ label, values, onChange }: { label: string; values: string[]; onChange: (index: number, value: string) => void }) {
  return (
    <div>
      <div className="mb-1 text-[11px] font-medium uppercase text-stone-500">{label}</div>
      <div className="space-y-1.5">
        {values.map((v, i) => (
          <textarea key={i} value={v} onChange={(e) => onChange(i, e.target.value)} rows={2}
            className="w-full resize-y rounded border border-stone-200 bg-white px-2 py-1.5 text-xs text-stone-900 outline-none focus:border-sky-400 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100" />
        ))}
      </div>
    </div>
  );
}

// ── Pure helpers ──

const EXAMPLES_PER_SIDE = 5;

type LayerOverrideParse = {
  layers: number[];
  error: string | null;
};

function parseLayerOverride(value: string, layerCount?: number | null): LayerOverrideParse {
  const trimmed = value.trim();
  if (!trimmed) return { layers: [], error: null };
  const tokens = trimmed.split(/[,\s]+/).filter(Boolean);
  const badToken = tokens.find((token) => !/^\d+$/.test(token));
  if (badToken) return { layers: [], error: `Invalid layer "${badToken}". Use comma or space separated integers.` };
  const layers = [...new Set(tokens.map((token) => Number(token)))].sort((a, b) => a - b);
  const unsafe = layers.find((layer) => !Number.isSafeInteger(layer));
  if (unsafe != null) return { layers: [], error: `Layer "${unsafe}" is not a safe integer.` };
  if (layerCount != null && layerCount > 0) {
    const outOfRange = layers.find((layer) => layer < 0 || layer >= layerCount);
    if (outOfRange != null) {
      return { layers: [], error: `Layer ${outOfRange} is outside model range 0-${layerCount - 1}.` };
    }
  }
  return { layers, error: null };
}

function formatAdvancedGenerationSummary(
  layerOverride: LayerOverrideParse,
  sweep: boolean,
  pooling: 'last_real' | 'mean',
): string {
  if (layerOverride.error) return 'settings need attention';
  const layerText = layerOverride.layers.length > 0
    ? `layers ${layerOverride.layers.join(',')}`
    : sweep ? 'default sweep' : 'layer 23';
  return `${layerText} · ${pooling}`;
}

function makeInitialDirectionDrafts(): DirectionDraft[] {
  return Array.from({ length: 10 }, (_, i) => makeBlankDirectionDraft(i));
}

function makeBlankDirectionDraft(index: number): DirectionDraft {
  const n = index + 1;
  return {
    name: `custom_direction_${n}`, description: `Custom direction ${n}`, domain: 'custom',
    positive: Array.from({ length: EXAMPLES_PER_SIDE }, (_, j) => `The user consistently shows positive custom behavior ${n}.${j + 1} in this app domain.`),
    negative: Array.from({ length: EXAMPLES_PER_SIDE }, (_, j) => `The user consistently shows contrasting custom behavior ${n}.${j + 1} in this app domain.`),
  };
}

function updateDraft(drafts: DirectionDraft[], index: number, patch: Partial<DirectionDraft>): DirectionDraft[] {
  return drafts.map((d, i) => (i === index ? { ...d, ...patch } : d));
}

function updateDraftExample(drafts: DirectionDraft[], index: number, side: 'positive' | 'negative', exampleIndex: number, value: string): DirectionDraft[] {
  return drafts.map((d, i) => {
    if (i !== index) return d;
    const examples = normalizeExamples(d[side]); examples[exampleIndex] = value;
    return { ...d, [side]: examples };
  });
}

function addDraftDirection(drafts: DirectionDraft[]): DirectionDraft[] { return [...drafts, makeBlankDirectionDraft(drafts.length)]; }
function removeDraftDirection(drafts: DirectionDraft[], index: number): DirectionDraft[] { return drafts.length <= 10 ? drafts : drafts.filter((_, i) => i !== index); }
function normalizeExamples(values: string[]): string[] { return [...values, ...Array.from({ length: EXAMPLES_PER_SIDE }, () => '')].slice(0, EXAMPLES_PER_SIDE); }

function suggestionToDraft(s: ALibraryDirectionSuggestion): DirectionDraft {
  return { name: s.name, description: s.description, domain: s.domain, positive: normalizeExamples(s.positive), negative: normalizeExamples(s.negative) };
}

function serializeDirectionDraftsToYaml(directionSetId: string, drafts: DirectionDraft[]): string {
  const directions = drafts.map((d, i) => ({
    name: inferDirectionSetId(d.name || `custom_direction_${i + 1}`),
    description: d.description.trim() || `Custom direction ${i + 1}`,
    domain: inferDirectionSetId(d.domain || 'custom'),
    positive: normalizeExamples(d.positive).map((v) => v.trim()),
    negative: normalizeExamples(d.negative).map((v) => v.trim()),
  }));
  return [
    'schema_version: edgestudio.a_library_direction_set_source.v1',
    `direction_set_id: ${yamlScalar(inferDirectionSetId(directionSetId))}`,
    'directions:',
    ...directions.flatMap((d) => [
      `  - name: ${yamlScalar(d.name)}`, `    description: ${yamlScalar(d.description)}`, `    domain: ${yamlScalar(d.domain)}`,
      '    positive:', ...d.positive.map((v) => `      - ${yamlScalar(v)}`),
      '    negative:', ...d.negative.map((v) => `      - ${yamlScalar(v)}`),
    ]),
    '',
  ].join('\n');
}

function yamlScalar(value: string): string { return JSON.stringify(value); }

function buildALibraryAskContext(result: ALibraryInspectResponse | null, selection: ALibrarySelectionResponse | null, history: ALibraryHistoryItem[]): string {
  const lines = [
    'A-library page context:', 'Explain A-library health reports in developer language.',
    'Glossary: orthogonality means direction vectors should not collapse into one another; signal strength means each direction has a measurable activation contrast; pass/fail is the validator verdict; target_layer is the layer chosen for RPP; sweep is the process of testing candidate layers.',
  ];
  if (selection) {
    lines.push(`Current model selection: status=${selection.status}, ok=${selection.ok}, reasons=${selection.reasons.join(', ') || 'none'}`);
    if (selection.selected) lines.push(`Selected library: ${selection.selected.library_id}, layer=${selection.selected.target_layer}, hidden=${selection.selected.hidden_size}`);
  }
  if (result) {
    lines.push(`Loaded library: ${result.library_path}`);
    lines.push(`Summary: family=${result.summary.model_family}, hidden=${result.summary.hidden_size}, target_layer=${result.summary.target_layer}, health=${result.summary.health_status}, direction_set=${result.summary.direction_set_id}, directions=${result.summary.n_directions}`);
    const reports = [...(result.health_reports ?? [])].sort((a, b) => (a.layer_idx ?? 0) - (b.layer_idx ?? 0)).slice(0, 12)
      .map((r) => `L${r.layer_idx}:${r.verdict}, max_cos=${formatDecimal(r.max_abs_cos_sim)}, min_signal=${formatDecimal(r.min_signal_strength)}, pass=${formatNullable(r.n_pass)}/${formatNullable(r.n_total)}`).join(' | ');
    if (reports) lines.push(`Layer health: ${reports}`);
    if (result.warnings.length > 0) lines.push(`Warnings: ${result.warnings.join(', ')}`);
  }
  if (history.length > 0) {
    lines.push(`Generated history count=${history.length}`);
    lines.push(`Recent: ${history.slice(0, 8).map((h) => `${h.model_name ?? '?'}:${h.direction_set_id ?? '?'}:L${h.target_layer ?? '?'}:${h.health_status ?? '?'}`).join(' | ')}`);
  }
  return lines.join('\n').slice(0, 6000);
}

function buildDirectionRepairContext(result: ALibraryInspectResponse | null): ALibraryDirectionRepairContext | null {
  if (!result) return null;
  const failedReports = (result.health_reports ?? []).filter((report) => report.verdict === 'fail');
  if (failedReports.length === 0 && result.summary.health_status !== 'fail') return null;
  const reports = failedReports.length > 0 ? failedReports : result.health_reports ?? [];
  if (reports.length === 0) return null;

  const targetLayer = result.summary.target_layer;
  const targetReport = reports.find((report) => report.layer_idx === targetLayer);
  const mostCollapsedReport = [...reports].sort(
    (a, b) => finiteNumber(b.max_abs_cos_sim, -Infinity) - finiteNumber(a.max_abs_cos_sim, -Infinity),
  )[0];
  const selectedReport = targetReport ?? mostCollapsedReport;
  const maxAbsCos = maxFinite(reports.map((report) => report.max_abs_cos_sim));
  const meanAbsCos = maxFinite(reports.map((report) => report.mean_abs_cos_sim));
  const signalPass = selectedReport.signal_pass
    ?? (reports.every((report) => report.signal_pass === true)
      ? true
      : reports.every((report) => report.signal_pass === false) ? false : null);

  const validationCodes = new Set<string>();
  validationCodes.add('health_verdict_fail');
  if (selectedReport.signal_pass === false) validationCodes.add('weak_signal');
  if (selectedReport.signal_pass === true && (selectedReport.max_pass === false || selectedReport.mean_pass === false)) {
    validationCodes.add('orthogonality_collapse');
  }
  for (const check of result.manifest.checks ?? []) {
    if (check.passed === false || check.present === false) validationCodes.add(`manifest_${check.name}`);
  }

  const reasonParts = [
    `health_status=${result.summary.health_status ?? 'unknown'}`,
    targetLayer != null ? `target_layer=${targetLayer}` : null,
    selectedReport.layer_idx != null ? `evidence_layer=${selectedReport.layer_idx}` : null,
    selectedReport.signal_pass === true && (selectedReport.max_pass === false || selectedReport.mean_pass === false)
      ? 'signal passed but orthogonality failed'
      : null,
    selectedReport.signal_pass === false ? 'signal failed' : null,
  ].filter(Boolean);

  return {
    worst_pairs: collectWorstPairs(reports),
    max_abs_cos: maxAbsCos,
    mean_abs_cos: meanAbsCos,
    signal_pass: signalPass,
    validation_error_codes: Array.from(validationCodes),
    prev_direction_set_id: result.summary.direction_set_id ?? null,
    reason: reasonParts.join('; '),
  };
}

function collectWorstPairs(reports: ALibraryHealthReport[]): string[][] {
  const pairs: string[][] = [];
  const seen = new Set<string>();
  const sortedReports = [...reports].sort(
    (a, b) => finiteNumber(b.max_abs_cos_sim, -Infinity) - finiteNumber(a.max_abs_cos_sim, -Infinity),
  );
  for (const report of sortedReports) {
    const pair = (report.worst_pair ?? []).slice(0, 2).map((item) => String(item).trim()).filter(Boolean);
    if (pair.length < 2) continue;
    const key = pair.join('\u0000');
    if (seen.has(key)) continue;
    seen.add(key);
    pairs.push(pair);
    if (pairs.length >= 6) break;
  }
  return pairs;
}

function maxFinite(values: Array<number | null | undefined>): number | null {
  const finite = values.filter((value): value is number => value != null && Number.isFinite(value));
  return finite.length > 0 ? Math.max(...finite) : null;
}

function finiteNumber(value: number | null | undefined, fallback: number): number {
  return value != null && Number.isFinite(value) ? value : fallback;
}

function formatRepairContextSummary(context: ALibraryDirectionRepairContext | null, t: Translator): string {
  if (!context) return t('training.aLibrary.repairDirectionsUnavailable');
  const pair = context.worst_pairs?.[0]?.slice(0, 2).join(' / ');
  const metrics = [
    context.max_abs_cos != null ? `max ${formatDecimal(context.max_abs_cos)}` : null,
    context.mean_abs_cos != null ? `mean ${formatDecimal(context.mean_abs_cos)}` : null,
  ].filter(Boolean).join(' · ');
  return t('training.aLibrary.repairDirectionsSummary', {
    pair: pair || '—',
    metrics: metrics || '—',
  });
}

function formatTimestamp(v: number | null | undefined): string { return v != null && Number.isFinite(v) ? new Date(v * 1000).toLocaleString() : '—'; }
function formatNullable(v: number | string | null | undefined): string { return v == null || v === '' ? '—' : String(v); }
function inferDirectionSetId(n: string): string { return n.replace(/\.(yaml|yml)$/i, '').replace(/[^A-Za-z0-9_.-]+/g, '_').replace(/^[._-]+|[._-]+$/g, '').slice(0, 80) || 'custom_direction_set'; }
function formatDomains(d: Record<string, number>): string { return Object.entries(d).map(([k, v]) => `${k}: ${v}`).join(' · ') || 'none'; }
function formatDecimal(v: number | null | undefined): string { return v != null && Number.isFinite(v) ? v.toFixed(3) : '—'; }
function formatWorstPair(v: string[] | null | undefined): string {
  return Array.isArray(v) && v.length >= 2 ? `${v[0]} / ${v[1]}` : '—';
}
function healthDiagnosis(report: ALibraryHealthReport, t: Translator): string {
  if (report.verdict === 'pass') return t('training.aLibrary.healthDiagnosisPass');
  if (report.signal_pass === true && (report.max_pass === false || report.mean_pass === false)) {
    return t('training.aLibrary.healthDiagnosisOrthogonalityFail', { pair: formatWorstPair(report.worst_pair) });
  }
  if (report.signal_pass === false) {
    return t('training.aLibrary.healthDiagnosisSignalFail');
  }
  return t('training.aLibrary.healthDiagnosisFail');
}
function statusText(v: string | null | undefined, t: Translator): string {
  if (v === 'pass') return t('training.common.pass');
  if (v === 'fail') return t('training.common.fail');
  if (v === 'ready') return t('training.common.ready');
  if (v === 'not_ready') return t('training.common.notReady');
  return v ?? t('training.common.unknown');
}
