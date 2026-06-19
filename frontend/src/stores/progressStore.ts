// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * progressStore — tracks model workflow progress for Dashboard intelligence.
 *
 * Fetches session summary + pipeline result from backend to determine
 * what stage the user is at for the current model.
 */

import { create } from 'zustand';
import { getSession, getPipelineResult } from '@/api/endpoints';
import type { SessionSummary } from '@/api/endpoints';
import type { PipelineRunResult } from '@/api/types';

export type WorkflowStage =
  | 'just_loaded'    // model loaded, nothing done yet
  | 'profiled'       // activation profile exists
  | 'optimized'      // pipeline has been run
  | 'exported';      // (future) export completed

interface ProgressState {
  // Session flags from backend
  session: SessionSummary | null;

  // Pipeline result (if any)
  pipelineResult: PipelineRunResult | null;

  // Computed stage
  stage: WorkflowStage;

  // Loading state
  loading: boolean;

  // Fetch progress for a model
  fetchProgress: (modelId: string) => Promise<void>;

  // Clear state (on model unload)
  clear: () => void;

  // Manual update after pipeline completes in current session
  setPipelineResult: (result: PipelineRunResult) => void;
}

function computeStage(session: SessionSummary | null, pipelineResult: PipelineRunResult | null): WorkflowStage {
  if (pipelineResult?.success) return 'optimized';
  if (session?.has_pipeline) return 'optimized';
  if (session?.has_profile) return 'profiled';
  return 'just_loaded';
}

export const useProgressStore = create<ProgressState>()((set, get) => ({
  session: null,
  pipelineResult: null,
  stage: 'just_loaded',
  loading: false,

  fetchProgress: async (modelId: string) => {
    set({ loading: true });

    try {
      // Fetch session summary (always available)
      const session = await getSession(modelId);

      // Try to fetch pipeline result (may 404)
      let pipelineResult: PipelineRunResult | null = null;
      if (session.has_pipeline) {
        try {
          pipelineResult = await getPipelineResult(modelId);
        } catch {
          // 404 — no pipeline result stored
        }
      }

      const stage = computeStage(session, pipelineResult);
      set({ session, pipelineResult, stage, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  clear: () => set({
    session: null,
    pipelineResult: null,
    stage: 'just_loaded',
    loading: false,
  }),

  setPipelineResult: (result: PipelineRunResult) => {
    const session = get().session;
    const stage = computeStage(session, result);
    set({ pipelineResult: result, stage });
  },
}));
