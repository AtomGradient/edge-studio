// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ModelInfo, ProfileSummary } from '@/api/types';

interface ModelState {
  // Current primary model
  currentModel: ModelInfo | null;
  setCurrentModel: (model: ModelInfo | null) => void;

  // Profile
  profileSummary: ProfileSummary | null;
  setProfileSummary: (summary: ProfileSummary | null) => void;

  // Trace availability
  hasTrace: boolean;
  hasAttentionTrace: boolean;
  hasTimingTrace: boolean;
  setTraceState: (has: boolean, hasAttention?: boolean, hasTiming?: boolean) => void;

  // Comparison model
  comparisonModel: ModelInfo | null;
  setComparisonModel: (model: ModelInfo | null) => void;

  // Reset all state
  reset: () => void;
}

export const useModelStore = create<ModelState>()(
  persist(
    (set) => ({
      currentModel: null,
      setCurrentModel: (model) => set({
        currentModel: model,
        profileSummary: null,
        hasTrace: false,
        hasAttentionTrace: false,
        hasTimingTrace: false,
      }),

      profileSummary: null,
      setProfileSummary: (summary) => set({ profileSummary: summary }),

      hasTrace: false,
      hasAttentionTrace: false,
      hasTimingTrace: false,
      setTraceState: (has, hasAttention = false, hasTiming = false) => set({
        hasTrace: has,
        hasAttentionTrace: hasAttention,
        hasTimingTrace: hasTiming,
      }),

      comparisonModel: null,
      setComparisonModel: (model) => set({ comparisonModel: model }),

      reset: () => set({
        currentModel: null,
        profileSummary: null,
        hasTrace: false,
        hasAttentionTrace: false,
        hasTimingTrace: false,
        comparisonModel: null,
      }),
    }),
    {
      name: 'vlm-model',
      partialize: (state) => ({
        currentModel: state.currentModel,
        comparisonModel: state.comparisonModel,
        hasTrace: state.hasTrace,
        hasAttentionTrace: state.hasAttentionTrace,
        hasTimingTrace: state.hasTimingTrace,
      }),
    },
  ),
);
