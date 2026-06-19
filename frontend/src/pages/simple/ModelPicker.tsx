// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 2: Model Picker — intent-driven search + category filter.
 *
 * Users can either type a natural language query or click a scenario chip.
 * The backend handles semantic search when the embedding model is ready,
 * falling back to keyword matching otherwise.
 *
 * Results are paginated (12 per page) with "Show More" to load additional pages.
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Package, Search, ChevronDown } from 'lucide-react';
import { SkeletonCard } from '@/components/common/Skeleton';
import { useWizardStore, type RecommendedModel } from '@/stores/wizardStore';
import { ModelCard } from '@/components/common/ModelCard';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS } from './wizardSteps';
import axios from 'axios';

const SCENARIOS = [
  { key: 'chat', queryHint: 'chat assistant 聊天助手' },
  { key: 'coding', queryHint: 'code generation 代码生成' },
  { key: 'reasoning', queryHint: 'reasoning analysis 推理分析' },
  { key: 'translation', queryHint: 'translation multilingual 翻译' },
  { key: 'multimodal', queryHint: 'image understanding 图片理解' },
  { key: 'asr', queryHint: 'speech recognition 语音识别' },
  { key: 'tts', queryHint: 'text to speech 语音合成' },
];

const PAGE_SIZE = 12;

export default function ModelPicker() {
  const t = useT();
  const navigate = useNavigate();
  const {
    targetDevice, setTargetDevice, useCase, setUseCase,
    searchQuery, setSearchQuery,
    recommendedModels, setRecommendedModels,
    selectedModel, setSelectedModel,
    setCurrentStep,
  } = useWizardStore();

  const [loading, setLoading] = useState(false);
  const [localQuery, setLocalQuery] = useState(searchQuery);
  const [embeddingReady, setEmbeddingReady] = useState(true);
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Intent search (semantic or fallback) — fetch all results, paginate on frontend
  const fetchByIntent = useCallback(async (query: string) => {
    setLoading(true);
    setVisibleCount(PAGE_SIZE);
    try {
      const res = await axios.post('/api/recommend/intent-search', {
        query,
        device_name: targetDevice,
        max_results: 100,
      });
      const { results, detected_device } = res.data;
      setRecommendedModels(results);
      if (detected_device) {
        setTargetDevice(detected_device);
      }
    } catch {
      // keep existing recommendations
    } finally {
      setLoading(false);
    }
  }, [targetDevice, setRecommendedModels]);

  // Category-based search (existing recommend API)
  const fetchByCategory = useCallback(async (uc: string) => {
    setLoading(true);
    setVisibleCount(PAGE_SIZE);
    try {
      const res = await axios.post('/api/recommend/models', {
        device_name: targetDevice,
        use_case: uc,
        max_results: 100,
      });
      setRecommendedModels(res.data);
    } catch {
      // keep existing
    } finally {
      setLoading(false);
    }
  }, [targetDevice, setRecommendedModels]);

  // Initial load
  useEffect(() => {
    if (searchQuery) {
      fetchByIntent(searchQuery);
    } else {
      fetchByCategory(useCase);
    }

    // Check & download embedding model for semantic search
    let embCancelled = false;
    let embTimer: ReturnType<typeof setTimeout>;
    const checkEmbedding = () => {
      axios.get('/api/recommend/embedding-status').then(res => {
        if (embCancelled) return;
        if (res.data.ready) {
          setEmbeddingReady(true);
        } else {
          setEmbeddingReady(false);
          if (res.data.dependency_ready) {
            if (!res.data.downloading) {
              axios.post('/api/recommend/embedding-download').catch(() => {});
            }
            embTimer = setTimeout(checkEmbedding, 5000);
          }
        }
      }).catch(() => {
        if (!embCancelled) embTimer = setTimeout(checkEmbedding, 5000);
      });
    };
    checkEmbedding();
    return () => { embCancelled = true; clearTimeout(embTimer); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle search input with debounce
  const handleSearchInput = (value: string) => {
    setLocalQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (!value.trim()) {
      // Clear search, revert to category mode
      setSearchQuery('');
      fetchByCategory(useCase);
      return;
    }

    debounceRef.current = setTimeout(() => {
      setSearchQuery(value);
      fetchByIntent(value);
    }, 500);
  };

  // Handle search submit (Enter key)
  const handleSearchSubmit = () => {
    if (!localQuery.trim()) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    setSearchQuery(localQuery);
    fetchByIntent(localQuery);
  };

  // Handle scenario chip click
  const handleScenarioClick = (scenario: typeof SCENARIOS[number]) => {
    setUseCase(scenario.key);
    setLocalQuery('');
    setSearchQuery('');
    fetchByCategory(scenario.key);
  };

  const handleSelect = (model: RecommendedModel) => {
    setSelectedModel(model);
  };

  const handleNext = () => {
    setCurrentStep(3);
    navigate('/simple/optimize');
  };

  const visibleModels = recommendedModels.slice(0, visibleCount);
  const hasMore = recommendedModels.length > visibleCount;

  return (
    <WizardShell
      steps={WIZARD_STEPS(t)}
      currentStep={2}
      onBack={() => { setCurrentStep(1); navigate('/simple/device'); }}
      onNext={handleNext}
      nextDisabled={!selectedModel}
      onStepClick={(s) => { setCurrentStep(s); }}
    >
      <div className="text-center">
        <div className="mb-2 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-stone-100 dark:bg-stone-800">
          <Package size={24} className="text-stone-600 dark:text-stone-400" />
        </div>
        <h2 className="mb-1 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.model.title')}
        </h2>
        <p className="mb-4 text-stone-500 dark:text-stone-400">
          {t('simple.model.subtitle')}
        </p>
      </div>

      {/* Intent search input */}
      <div className="mb-4">
        <div className="relative">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
          <input
            type="text"
            value={localQuery}
            onChange={(e) => handleSearchInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSearchSubmit(); }}
            placeholder={t('simple.intent.placeholder')}
            className="w-full rounded-xl border border-stone-200 bg-white py-2.5 pl-10 pr-4 text-sm text-stone-900 placeholder-stone-400 outline-none transition-all focus:border-stone-400 focus:ring-1 focus:ring-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100 dark:placeholder-stone-500 dark:focus:border-stone-500 dark:focus:ring-stone-500"
          />
        </div>
        <p className="mt-1 text-center text-xs text-stone-400 dark:text-stone-500">
          {t('simple.intent.hint')}
        </p>
        {!embeddingReady && (
          <p className="mt-1.5 flex items-center justify-center gap-1.5 text-xs text-amber-500 dark:text-amber-400">
            <Loader2 size={11} className="animate-spin" />
            {t('simple.v1.downloadingSearch')}
          </p>
        )}
      </div>

      {/* Scenario chips */}
      <div className="mb-6 flex flex-wrap justify-center gap-2">
        {SCENARIOS.map((sc) => (
          <button
            key={sc.key}
            type="button"
            onClick={() => handleScenarioClick(sc)}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-all duration-200 ${
              useCase === sc.key && !searchQuery
                ? 'bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
                : 'bg-stone-100 text-stone-600 hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-400 dark:hover:bg-stone-700'
            }`}
          >
            {t(`simple.useCase.${sc.key}`)}
          </button>
        ))}
      </div>

      {/* Model cards with pagination */}
      {loading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : (
        <>
          <div className="grid gap-3 sm:grid-cols-2">
            {visibleModels.map((model) => (
              <ModelCard
                key={model.download_hint}
                model={model}
                isSelected={selectedModel?.download_hint === model.download_hint}
                onSelect={() => handleSelect(model)}
              />
            ))}
          </div>

          {hasMore && (
            <div className="mt-4 text-center">
              <button
                type="button"
                onClick={() => setVisibleCount(prev => prev + PAGE_SIZE)}
                className="inline-flex items-center gap-1.5 rounded-full bg-stone-100 px-5 py-2 text-sm font-medium text-stone-600 transition-colors hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-400 dark:hover:bg-stone-700"
              >
                <ChevronDown size={16} />
                {t('simple.v1.showMore', { count: String(recommendedModels.length - visibleCount) })}
              </button>
            </div>
          )}
        </>
      )}

      {recommendedModels.length === 0 && !loading && (
        <p className="py-8 text-center text-sm text-stone-400">
          {t('simple.model.noModels')}
        </p>
      )}
    </WizardShell>
  );
}
