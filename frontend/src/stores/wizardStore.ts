// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Wizard store — tracks simple mode wizard progress, selections, and state.
 * Persisted to localStorage so users can resume where they left off.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface SystemInfo {
  chip: string;
  ram_gb: number;
  gpu_cores: number;
  os: string;
  os_version: string;
  arch: string;
  matched_device: string | null;
  max_model_size_gb: number | null;
}

export interface RecommendedModel {
  name: string;
  description: string;
  estimated_size_gb: number;
  fits_device: boolean;
  headroom_gb: number;
  quality_tier: string;
  download_hint: string;
}

export interface OptimizationRec {
  strategy_name: string;
  description: string;
  estimated_final_size_gb: number;
  fits_device: boolean;
  steps: string[];
  risk_level: string;
  quality_impact: string;
}

interface WizardState {
  /** Current wizard step (0-5) */
  currentStep: number;
  setCurrentStep: (step: number) => void;

  /** Detected system info */
  systemInfo: SystemInfo | null;
  setSystemInfo: (info: SystemInfo) => void;

  /** Target device (user may override auto-detected) */
  targetDevice: string;
  setTargetDevice: (device: string) => void;

  /** Selected use case */
  useCase: string;
  setUseCase: (uc: string) => void;

  /** Intent search query */
  searchQuery: string;
  setSearchQuery: (q: string) => void;

  /** Recommended models list */
  recommendedModels: RecommendedModel[];
  setRecommendedModels: (models: RecommendedModel[]) => void;

  /** User-selected model from recommendations */
  selectedModel: RecommendedModel | null;
  setSelectedModel: (model: RecommendedModel | null) => void;

  /** Loaded model dir (after download or selection) */
  loadedModelDir: string;
  setLoadedModelDir: (dir: string) => void;

  /** Loaded model ID from backend */
  loadedModelId: string;
  setLoadedModelId: (id: string) => void;

  /** Optimization recommendation */
  optimizationRec: OptimizationRec | null;
  setOptimizationRec: (rec: OptimizationRec | null) => void;

  /** Whether optimization is completed */
  optimizationDone: boolean;
  setOptimizationDone: (done: boolean) => void;

  /** Active download task ID (survives page navigation) */
  downloadTaskId: string;
  setDownloadTaskId: (id: string) => void;

  /** Chat tested flag */
  chatTested: boolean;
  setChatTested: (tested: boolean) => void;

  /** Reset wizard to initial state */
  reset: () => void;
}

const initialState = {
  currentStep: 0,
  systemInfo: null,
  targetDevice: '',
  useCase: 'chat',
  searchQuery: '',
  recommendedModels: [],
  selectedModel: null,
  loadedModelDir: '',
  loadedModelId: '',
  optimizationRec: null,
  downloadTaskId: '',
  optimizationDone: false,
  chatTested: false,
};

export const useWizardStore = create<WizardState>()(
  persist(
    (set) => ({
      ...initialState,

      setCurrentStep: (step) => set({ currentStep: step }),
      setSystemInfo: (info) => set({ systemInfo: info }),
      setTargetDevice: (device) => set({ targetDevice: device }),
      setUseCase: (uc) => set({ useCase: uc }),
      setSearchQuery: (q) => set({ searchQuery: q }),
      setRecommendedModels: (models) => set({ recommendedModels: models }),
      setSelectedModel: (model) => set({ selectedModel: model }),
      setLoadedModelDir: (dir) => set({ loadedModelDir: dir }),
      setLoadedModelId: (id) => set({ loadedModelId: id }),
      setOptimizationRec: (rec) => set({ optimizationRec: rec }),
      setOptimizationDone: (done) => set({ optimizationDone: done }),
      setDownloadTaskId: (id) => set({ downloadTaskId: id }),
      setChatTested: (tested) => set({ chatTested: tested }),
      reset: () => set(initialState),
    }),
    {
      name: 'vlm-wizard',
      partialize: (state) => ({
        currentStep: state.currentStep,
        // systemInfo intentionally excluded — must re-detect per session/machine
        targetDevice: state.targetDevice,
        useCase: state.useCase,
        selectedModel: state.selectedModel,
        loadedModelDir: state.loadedModelDir,
        loadedModelId: state.loadedModelId,
        downloadTaskId: state.downloadTaskId,
        optimizationDone: state.optimizationDone,
        chatTested: state.chatTested,
      }),
    },
  ),
);
