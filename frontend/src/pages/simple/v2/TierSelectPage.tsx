// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 2: Tier selection — choose performance level.
 * Shows Standard/Pro/Max/Ultra package cards from backend.
 * Keeps direct model search visible alongside the recommended tier path.
 */

import { useCallback, useEffect, useMemo, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Loader2, Star, Zap, Clock, ChevronDown, ChevronUp, Search,
  X, HardDrive, Sparkles, AlertTriangle, Brain, Volume2, Mic,
  type LucideIcon,
} from 'lucide-react';
import { useSimpleStore, type Package, type PackageModel } from '@/stores/simpleStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS_V2 } from './wizardStepsV2';
import { cn } from '@/lib/utils';
import axios from 'axios';

const TIER_ICONS: Record<string, React.ReactNode> = {
  standard: <Zap size={18} />,
  pro: <Star size={18} />,
  max: <Star size={18} />,
  ultra: <Star size={18} />,
};

const TIER_GRADIENTS: Record<string, string> = {
  standard: 'from-sky-50 to-blue-50 dark:from-sky-900/10 dark:to-blue-900/10',
  pro: 'from-violet-50 to-purple-50 dark:from-violet-900/10 dark:to-purple-900/10',
  max: 'from-amber-50 to-orange-50 dark:from-amber-900/10 dark:to-orange-900/10',
  ultra: 'from-rose-50 to-pink-50 dark:from-rose-900/10 dark:to-pink-900/10',
};

// Map simple focus → recommend API use_case
const FOCUS_TO_USE_CASE: Record<string, string> = {
  chat: 'chat',
  coding: 'coding',
  vision: 'multimodal',
  asr: 'asr',
  tts: 'tts',
  voice_duplex: 'chat',  // browse models shows LLM candidates
};

interface BrowseModel {
  name: string;
  description: string;
  estimated_size_gb: number;
  fits_device: boolean;
  headroom_gb: number;
  quality_tier: string;
  download_hint: string;
  category?: string;
  family?: string;
  params_b?: number;
}

interface CatalogStatus {
  source: string;
  version: string;
  generated_at?: string | null;
  total_models: number;
  refreshing: boolean;
  stale: boolean;
  started?: boolean;
  runtime_only_download_hints?: string[];
}

interface EmbeddingStatus {
  ready: boolean;
  dependency_ready: boolean;
  downloading: boolean;
  task_id?: string | null;
}

const QUALITY_STYLES: Record<string, { labelKey: string; color: string }> = {
  high: { labelKey: 'simple.v2.tier.qualityHigh', color: 'text-amber-600 bg-amber-50 dark:text-amber-400 dark:bg-amber-900/30' },
  balanced: { labelKey: 'simple.v2.tier.qualityBalanced', color: 'text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-900/30' },
  entry: { labelKey: 'simple.v2.tier.qualityCompact', color: 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-900/30' },
};

export default function TierSelectPage() {
  const t = useT();
  const navigate = useNavigate();
  const {
    focus, setTier, setPackages, packages, setSetupInfo, setCustomModelId, deviceProfile,
    setLoadedModelId, setLoadedModelDir, setSetupPhase,
    ttsVariant, setTtsVariant,
  } = useSimpleStore();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [recommendedTier, setRecommendedTier] = useState('');
  const [expandedTier, setExpandedTier] = useState<string | null>(null);

  const [browserModels, setBrowserModels] = useState<BrowseModel[]>([]);
  const [browserLoading, setBrowserLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const searchQueryRef = useRef('');
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const browserContextSigRef = useRef('');
  const catalogRefreshStartedRef = useRef(false);
  const catalogSigRef = useRef('');
  const embeddingAutoStartedRef = useRef(false);
  const [catalogStatus, setCatalogStatus] = useState<CatalogStatus | null>(null);
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus | null>(null);
  const [embeddingNeedsRetry, setEmbeddingNeedsRetry] = useState(false);
  const [embeddingRetrySeq, setEmbeddingRetrySeq] = useState(0);
  const searchTtsVariant = focus === 'voice_duplex' ? (ttsVariant || 'customvoice') : '';
  const quickSearchChips = useMemo(() => {
    if (focus === 'asr') return ['Whisper', 'SenseVoice', 'Parakeet'];
    if (focus === 'tts') return ['Qwen3 TTS', 'Kokoro', 'VoiceDesign', 'CustomVoice'];
    if (focus === 'voice_duplex') {
      return ['Qwen3.6', 'Whisper', ttsVariant === 'voicedesign' ? 'VoiceDesign' : 'CustomVoice', 'TTS'];
    }
    if (focus === 'vision') return ['Qwen3.6 VL', 'Gemma4', 'MiniCPM'];
    if (focus === 'coding') return ['Qwen3.6', 'Gemma4', 'DeepSeek'];
    return ['Qwen3.6', 'Gemma4', 'Whisper', 'TTS'];
  }, [focus, ttsVariant]);

  const fetchPackages = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await axios.post('/api/simple/packages', {
        focus,
        ram_gb: deviceProfile?.ram_gb || 0,
        tts_variant: focus === 'voice_duplex' ? (ttsVariant || 'customvoice') : '',
      });
      setPackages(res.data.packages);
      setRecommendedTier(res.data.recommended_tier);
    } catch (err: unknown) {
      const detail = axios.isAxiosError(err) ? err.response?.data?.detail : '';
      setError(detail || t('common.error'));
    } finally {
      setLoading(false);
    }
  }, [deviceProfile?.ram_gb, focus, setPackages, t, ttsVariant]);

  // Set default TTS variant for duplex on mount
  useEffect(() => {
    if (focus === 'voice_duplex' && !ttsVariant) {
      setTtsVariant('customvoice');
    }
  }, [focus]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!focus) {
      navigate('/simple/focus');
      return;
    }
    fetchPackages();
  }, [focus, ttsVariant]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!focus || catalogRefreshStartedRef.current) return;
    catalogRefreshStartedRef.current = true;

    let cancelled = false;
    let timer: number | null = null;

    const handleStatus = (status: CatalogStatus) => {
      if (cancelled) return;
      setCatalogStatus(status);
      const sig = `${status.source}:${status.version}:${status.generated_at || ''}:${status.total_models}`;
      const changed = Boolean(catalogSigRef.current && catalogSigRef.current !== sig);
      catalogSigRef.current = sig;
      if (changed) {
        fetchPackages();
      }
      if (status.refreshing) {
        timer = window.setTimeout(() => poll(false), 3000);
      }
    };

    const poll = async (start: boolean) => {
      try {
        const res = start
          ? await axios.post('/api/recommend/catalog-refresh')
          : await axios.get('/api/recommend/catalog-status');
        handleStatus(res.data);
      } catch {
        if (!cancelled) {
          setCatalogStatus(null);
        }
      }
    };

    poll(true);
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [fetchPackages, focus]);

  const handleEmbeddingRetry = useCallback(() => {
    embeddingAutoStartedRef.current = false;
    setEmbeddingNeedsRetry(false);
    setEmbeddingRetrySeq((prev) => prev + 1);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;

    const checkEmbedding = async () => {
      try {
        const res = await axios.get('/api/recommend/embedding-status');
        if (cancelled) return;
        const status: EmbeddingStatus = res.data;
        setEmbeddingStatus(status);

        if (!status.ready && status.dependency_ready) {
          if (status.downloading) {
            setEmbeddingNeedsRetry(false);
            timer = window.setTimeout(checkEmbedding, 5000);
          } else if (!embeddingAutoStartedRef.current) {
            embeddingAutoStartedRef.current = true;
            setEmbeddingNeedsRetry(false);
            const started = await axios.post('/api/recommend/embedding-download').catch(() => null);
            if (!started) {
              setEmbeddingNeedsRetry(true);
              return;
            }
            timer = window.setTimeout(checkEmbedding, 5000);
          } else {
            setEmbeddingNeedsRetry(true);
          }
        } else {
          setEmbeddingNeedsRetry(false);
        }
      } catch {
        if (!cancelled) {
          timer = window.setTimeout(checkEmbedding, 8000);
        }
      }
    };

    checkEmbedding();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [embeddingRetrySeq]);

  const handleSelectTier = async (pkg: Package) => {
    if (!pkg.available) return;
    setTier(pkg.tier);
    setCustomModelId('');
    // Clear previously loaded model — user is choosing a new one
    setLoadedModelId('');
    setLoadedModelDir('');
    setSetupPhase('idle');

    if (focus === 'voice_duplex') {
      // Duplex uses Package data directly — no setup API call needed
      // SetupPage reads packages from store for model download hints
      navigate('/simple/setup');
      return;
    }

    // Single-model: resolve model info via setup endpoint
    try {
      const res = await axios.post('/api/simple/setup', { focus, tier: pkg.tier });
      setSetupInfo(res.data);
    } catch {
      // Will be handled on setup page
    }
    navigate('/simple/setup');
  };

  const handleSelectBrowseModel = async (model: BrowseModel) => {
    setTier('');
    setCustomModelId(model.download_hint);
    // Clear previously loaded model — user is choosing a new one
    setLoadedModelId('');
    setLoadedModelDir('');
    setSetupPhase('idle');

    try {
      const res = await axios.post('/api/simple/setup', {
        focus,
        custom_model_id: model.download_hint,
      });
      setSetupInfo(res.data);
    } catch {
      // Will be handled on setup page
    }
    navigate('/simple/setup');
  };

  // Fetch models for browser
  const fetchBrowseModels = useCallback(async (query?: string) => {
    setBrowserLoading(true);
    try {
      const trimmedQuery = query?.trim() || '';
      const commonPayload = {
        device_name: deviceProfile?.chip || '',
        max_results: 50,
        tts_variant: searchTtsVariant,
      };

      if (trimmedQuery) {
        // Intent search
        const res = await axios.post('/api/recommend/intent-search', {
          ...commonPayload,
          query: trimmedQuery,
        });
        setBrowserModels(res.data.results || []);
      } else {
        // Category browse
        const res = await axios.post('/api/recommend/models', {
          ...commonPayload,
          use_case: FOCUS_TO_USE_CASE[focus] || 'chat',
        });
        setBrowserModels(res.data || []);
      }
    } catch {
      setBrowserModels([]);
    } finally {
      setBrowserLoading(false);
    }
  }, [deviceProfile?.chip, focus, searchTtsVariant]);

  useEffect(() => {
    if (loading || error || packages.length === 0) return;
    const sig = `${focus}:${deviceProfile?.chip || ''}:${searchTtsVariant}:${packages.length}`;
    if (browserContextSigRef.current === sig) return;
    browserContextSigRef.current = sig;
    fetchBrowseModels(searchQueryRef.current);
  }, [deviceProfile?.chip, error, fetchBrowseModels, focus, loading, packages.length, searchTtsVariant]);

  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    };
  }, []);

  const handleQuickSearch = (query: string) => {
    searchQueryRef.current = query;
    setSearchQuery(query);
    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
    }
    fetchBrowseModels(query);
  };

  // Debounced search
  const handleSearchChange = (val: string) => {
    searchQueryRef.current = val;
    setSearchQuery(val);
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(() => fetchBrowseModels(val), 400);
  };

  const formatTimeHint = (hint: string): string => {
    if (!hint) return '';
    if (hint.includes(':')) {
      const [key, val] = hint.split(':');
      return t(key, { count: val });
    }
    return t(hint);
  };

  const formatModelSize = (size?: number): string => {
    if (!size || size <= 0) return t('simple.v2.tier.unknownSize');
    return `${Number(size.toFixed(1))} GB`;
  };

  const formatParamCount = (params?: number): string => {
    if (!params || params <= 0) return t('simple.v2.tier.unknownValue');
    if (params < 1) return `${Math.round(params * 1000)}M`;
    return `${Number(params.toFixed(1))}B`;
  };

  const formatQuant = (quant?: string): string => {
    const value = (quant || '').trim();
    if (!value) return t('simple.v2.tier.unknownValue');
    return value
      .replace(/(\d)\s*bit/i, '$1-bit')
      .replace(/^bf16$/i, 'BF16')
      .replace(/^fp16$/i, 'FP16')
      .replace(/^fp8$/i, 'FP8')
      .replace(/^mxfp4$/i, 'MXFP4');
  };

  const getSingleModelMeta = (model: PackageModel): string[] => {
    const meta = [formatModelSize(model.size_gb)];
    if (model.params_b > 0) {
      meta.push(`${formatParamCount(model.params_b)} ${t('simple.v2.tier.detail.paramsShort')}`);
    }
    if (model.quant?.trim()) {
      meta.push(formatQuant(model.quant));
    }
    return meta;
  };

  const renderSingleModelDetail = (model: PackageModel) => (
    <div className="px-1 py-2">
      <p className="break-words text-sm font-medium leading-5 text-stone-800 dark:text-stone-100">
        {model.display_name}
      </p>
      <p className="mt-1.5 flex flex-wrap items-center gap-x-1.5 gap-y-1 text-xs text-stone-500 dark:text-stone-400">
        {getSingleModelMeta(model).map((item, index) => (
          <span key={item} className="inline-flex items-center">
            {index > 0 && (
              <span className="mr-1.5 text-stone-300 dark:text-stone-600">·</span>
            )}
            {item}
          </span>
        ))}
      </p>
    </div>
  );

  const renderDuplexRow = (model: PackageModel, Icon: LucideIcon, label: string) => (
    <div className="flex items-center gap-2.5 py-1.5">
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white text-stone-500 shadow-sm ring-1 ring-stone-200 dark:bg-stone-950 dark:text-stone-300 dark:ring-stone-800">
        <Icon size={13} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs font-medium text-stone-700 dark:text-stone-200">
          {model.display_name}
        </span>
        <span className="mt-0.5 block text-[11px] text-stone-400 dark:text-stone-500">
          {label}
        </span>
      </span>
      <span className="shrink-0 text-xs tabular-nums text-stone-500 dark:text-stone-400">
        {formatModelSize(model.size_gb)}
      </span>
    </div>
  );

  const catalogFooter = catalogStatus
    ? catalogStatus.source === 'remote_cache'
      ? t('simple.v2.catalog.browserUpdated', {
        date: (catalogStatus.generated_at || catalogStatus.version || '').slice(0, 10),
        count: catalogStatus.total_models,
      })
      : t('simple.v2.catalog.browserOffline', { count: catalogStatus.total_models })
    : '';

  return (
    <WizardShell
      steps={WIZARD_STEPS_V2(t)}
      currentStep={2}
      onBack={() => navigate('/simple/focus')}
      helpKey="simple.v2.help.tier"
    >
      <div className="text-center">
        <h1 className="mb-2 text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.v2.tier.title')}
        </h1>
        <p className="mb-8 text-stone-500 dark:text-stone-400">
          {t('simple.v2.tier.subtitle')}
        </p>
        {catalogStatus && (catalogStatus.refreshing || catalogStatus.source === 'remote_cache') && (
          <div className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white px-3 py-1.5 text-xs text-stone-500 dark:border-stone-800 dark:bg-stone-900 dark:text-stone-400">
            {catalogStatus.refreshing ? (
              <>
                <Loader2 size={12} className="animate-spin" />
                {t('simple.v2.catalog.refreshing')}
              </>
            ) : (
              <>
                <Sparkles size={12} className="text-emerald-500" />
                {t('simple.v2.catalog.updated', { count: catalogStatus.total_models })}
              </>
            )}
          </div>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center gap-3 py-16 text-stone-400">
          <Loader2 size={28} className="animate-spin" />
        </div>
      )}

      {/* Error */}
      {error && !loading && (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {/* TTS variant selector — only for voice_duplex */}
      {focus === 'voice_duplex' && !loading && (
        <div className="mb-6">
          <p className="mb-2 text-center text-sm font-medium text-stone-600 dark:text-stone-400">
            {t('simple.v2.tier.ttsVariantTitle')}
          </p>
          <div className="mx-auto flex w-fit rounded-xl bg-stone-100 p-1 dark:bg-stone-800">
            {([
              { value: 'customvoice', label: t('simple.v2.tier.ttsVariantCustom'), desc: t('simple.v2.tier.ttsVariantCustomDesc') },
              { value: 'voicedesign', label: t('simple.v2.tier.ttsVariantDesign'), desc: t('simple.v2.tier.ttsVariantDesignDesc') },
            ] as const).map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setTtsVariant(opt.value)}
                className={cn(
                  'rounded-lg px-4 py-2 text-sm transition-all',
                  ttsVariant === opt.value
                    ? 'bg-white font-medium text-stone-900 shadow-sm dark:bg-stone-700 dark:text-stone-100'
                    : 'text-stone-500 hover:text-stone-700 dark:text-stone-400 dark:hover:text-stone-300',
                )}
              >
                <span>{opt.label}</span>
                <span className="ml-1 text-xs text-stone-400 dark:text-stone-500">{opt.desc}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Package cards */}
      {!loading && !error && packages.length > 0 && (
        <div className="space-y-6">
          <div className={cn(
            'grid gap-3',
            packages.length <= 3
              ? 'sm:grid-cols-3'
              : 'sm:grid-cols-2 lg:grid-cols-4',
          )}>
            {packages.map((pkg) => {
              const isRecommended = pkg.tier === recommendedTier;
              const isExpanded = expandedTier === pkg.tier;

              return (
                <div key={pkg.tier} className="flex flex-col">
                  <button
                    type="button"
                    onClick={() => handleSelectTier(pkg)}
                    disabled={!pkg.available}
                    className={cn(
                      'group relative flex flex-1 flex-col rounded-2xl border-2 p-5 text-left transition-all duration-200',
                      pkg.available
                        ? 'hover:shadow-lg hover:-translate-y-0.5'
                        : 'cursor-not-allowed opacity-50',
                      isRecommended
                        ? 'border-stone-900 dark:border-stone-100'
                        : 'border-stone-200 dark:border-stone-800',
                      pkg.available && !isRecommended && 'hover:border-stone-400 dark:hover:border-stone-600',
                    )}
                  >
                    {/* Recommended badge */}
                    {isRecommended && (
                      <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 rounded-full bg-stone-900 px-3 py-0.5 text-xs font-medium text-white dark:bg-stone-100 dark:text-stone-900">
                        {t('simple.v2.tier.recommended')}
                      </span>
                    )}

                    {/* Tier name */}
                    <div className="mb-3 flex items-center gap-2">
                      <div className={cn(
                        'flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br',
                        TIER_GRADIENTS[pkg.tier] || TIER_GRADIENTS.standard,
                      )}>
                        {TIER_ICONS[pkg.tier]}
                      </div>
                      <span className="text-lg font-semibold text-stone-900 dark:text-stone-100">
                        {pkg.tier_label}
                      </span>
                    </div>

                    {/* Tier subtitle */}
                    <p className="mb-3 text-sm font-medium text-stone-700 dark:text-stone-300">
                      {t(`simple.v2.tier.${pkg.tier}.name`)}
                    </p>

                    {/* Capabilities */}
                    <ul className="mb-4 flex-1 space-y-1.5">
                      {pkg.capabilities.map((cap, i) => (
                        <li key={i} className="text-xs text-stone-500 dark:text-stone-400">
                          {t(cap)}
                        </li>
                      ))}
                    </ul>

                    {/* Size + time */}
                    <div className="flex flex-wrap items-center gap-2 border-t border-stone-100 pt-3 dark:border-stone-800">
                      <span className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2 py-0.5 text-xs font-medium text-stone-600 dark:bg-stone-800 dark:text-stone-300">
                        <HardDrive size={11} />
                        {formatModelSize(pkg.download_size_gb)}
                      </span>
                      {pkg.available && pkg.setup_time_hint && (
                        <span className="inline-flex items-center gap-1 text-xs text-stone-400 dark:text-stone-500">
                          <Clock size={11} />
                          <span>{formatTimeHint(pkg.setup_time_hint)}</span>
                        </span>
                      )}
                      {!pkg.available && (
                        <p className="text-xs text-red-400">
                          {t('simple.v2.tier.unavailable')}
                        </p>
                      )}
                    </div>
                  </button>

                  {/* Detail toggle */}
                  {pkg.model && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setExpandedTier(isExpanded ? null : pkg.tier);
                      }}
                      className="mt-1 flex items-center justify-center gap-1 py-1 text-xs text-stone-400 transition-colors hover:text-stone-600 dark:hover:text-stone-300"
                    >
                      {t('simple.v2.tier.details')}
                      {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                    </button>
                  )}

                  {/* Expanded detail */}
                  {isExpanded && pkg.model && (
                    <div className="mt-2 rounded-xl bg-stone-50/80 px-3 py-2 text-xs shadow-inner shadow-stone-200/40 dark:bg-stone-900/60 dark:shadow-black/10">
                      {(pkg.secondary_model || pkg.tertiary_model) ? (
                        <>
                          {renderDuplexRow(pkg.model, Brain, t('simple.v2.tier.detail.roleBrain'))}
                          {pkg.secondary_model && (
                            renderDuplexRow(pkg.secondary_model, Volume2, t('simple.v2.tier.detail.roleVoice'))
                          )}
                          {pkg.tertiary_model && (
                            renderDuplexRow(pkg.tertiary_model, Mic, t('simple.v2.tier.detail.roleListening'))
                          )}
                          <div className="mt-2 flex items-center justify-between border-t border-stone-200/70 pt-2 text-xs dark:border-stone-700/70">
                            <span className="font-medium text-stone-600 dark:text-stone-300">
                              {t('simple.v2.tier.detail.totalDownload')}
                            </span>
                            <span className="tabular-nums text-stone-600 dark:text-stone-300">
                              {formatModelSize(pkg.download_size_gb)}
                            </span>
                          </div>
                        </>
                      ) : (
                        renderSingleModelDetail(pkg.model)
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Recommendation hint */}
          <p className="text-center text-sm text-stone-400 dark:text-stone-500">
            {t('simple.v2.tier.unsure', { tier: recommendedTier.charAt(0).toUpperCase() + recommendedTier.slice(1) })}
          </p>

          {/* Inline model browser */}
          <div className="space-y-4 rounded-2xl border border-stone-200 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-900/50">
            {/* Search input */}
            <div className="space-y-3">
              <div className="relative">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => handleSearchChange(e.target.value)}
                  placeholder={t('simple.v2.tier.browserSearch')}
                  className="w-full rounded-xl border border-stone-300 bg-white py-2.5 pl-10 pr-10 text-sm text-stone-900 outline-none transition-colors focus:border-stone-500 focus:ring-2 focus:ring-stone-200 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-100 dark:focus:border-stone-500 dark:focus:ring-stone-700"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => {
                      searchQueryRef.current = '';
                      setSearchQuery('');
                      fetchBrowseModels('');
                    }}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-stone-400 hover:text-stone-600"
                  >
                    <X size={14} />
                  </button>
                )}
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                <span className="text-xs text-stone-400 dark:text-stone-500">
                  {t('simple.v2.tier.quickSearch')}
                </span>
                {quickSearchChips.map((chip) => (
                  <button
                    key={chip}
                    type="button"
                    onClick={() => handleQuickSearch(chip)}
                    className="rounded-full border border-stone-200 bg-white px-2.5 py-1 text-xs font-medium text-stone-600 transition-colors hover:border-stone-400 hover:text-stone-900 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-300 dark:hover:border-stone-500 dark:hover:text-stone-100"
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
            {embeddingStatus && !embeddingStatus.ready && (
              <p className="flex items-center justify-center gap-1.5 text-xs text-amber-500 dark:text-amber-400">
                {embeddingNeedsRetry ? (
                  <>
                    {t('simple.v2.embedding.failed')}
                    <button
                      type="button"
                      onClick={handleEmbeddingRetry}
                      className="font-medium underline underline-offset-2"
                    >
                      {t('simple.v2.embedding.retry')}
                    </button>
                  </>
                ) : embeddingStatus.dependency_ready ? (
                  <>
                    <Loader2 size={11} className="animate-spin" />
                    {embeddingStatus.downloading
                      ? t('simple.v2.embedding.downloading')
                      : t('simple.v2.embedding.preparing')}
                  </>
                ) : (
                  t('simple.v2.embedding.missingDependency')
                )}
              </p>
            )}

            {/* Loading */}
            {browserLoading && (
              <div className="flex items-center justify-center gap-2 py-6 text-sm text-stone-400">
                <Loader2 size={16} className="animate-spin" />
                {t('simple.v2.tier.browserLoading')}
              </div>
            )}

            {/* Results */}
            {!browserLoading && browserModels.length === 0 && (
              <p className="py-6 text-center text-sm text-stone-400">
                {t('simple.v2.tier.browserEmpty')}
              </p>
            )}

            {!browserLoading && browserModels.length > 0 && (
              <div className="max-h-80 space-y-2 overflow-y-auto">
                {browserModels.map((m) => {
                  const q = QUALITY_STYLES[m.quality_tier] || QUALITY_STYLES.entry;
                  const isNew = catalogStatus?.runtime_only_download_hints?.includes(m.download_hint) ?? false;
                  return (
                    <button
                      key={m.download_hint}
                      type="button"
                      onClick={() => handleSelectBrowseModel(m)}
                      className="flex w-full items-center gap-3 rounded-xl border border-stone-200 bg-white p-3 text-left transition-all hover:border-stone-400 hover:shadow-sm dark:border-stone-700 dark:bg-stone-800 dark:hover:border-stone-500"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-stone-900 dark:text-stone-100">
                          {m.name}
                        </p>
                        <div className="mt-1 flex items-center gap-3 text-xs text-stone-500 dark:text-stone-400">
                          <span className="flex items-center gap-1">
                            <HardDrive size={11} />
                            {formatModelSize(m.estimated_size_gb)}
                          </span>
                          {m.fits_device ? (
                            <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                              <Sparkles size={11} />
                              {t('simple.v2.tier.browserFits')}
                            </span>
                          ) : (
                            <span
                              className="flex items-center gap-1 rounded-full bg-red-50 px-1.5 py-0.5 font-medium text-red-600 dark:bg-red-900/30 dark:text-red-300"
                              title={t('simple.v2.tier.browserTooLargeHint')}
                            >
                              <AlertTriangle size={11} />
                              {t('simple.v2.tier.browserTooLarge')}
                            </span>
                          )}
                          {m.family && <span>{m.family}</span>}
                        </div>
                      </div>
                      <span className={cn('shrink-0 rounded-full px-2 py-0.5 text-xs font-medium', q.color)}>
                        {t(q.labelKey)}
                      </span>
                      {isNew && (
                        <span className="shrink-0 rounded-full bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-300">
                          {t('simple.v2.catalog.new')}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}

            {catalogFooter && (
              <p className="border-t border-stone-200 pt-3 text-center text-xs text-stone-400 dark:border-stone-800 dark:text-stone-500">
                {catalogFooter}
              </p>
            )}
          </div>
        </div>
      )}
    </WizardShell>
  );
}
