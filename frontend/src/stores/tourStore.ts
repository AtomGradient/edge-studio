// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface TourState {
  /** Whether the tour has been completed at least once */
  tourCompleted: boolean;
  /** Whether the tour is currently active */
  tourActive: boolean;
  /** Current step index */
  currentStep: number;
  /** Start the tour */
  startTour: () => void;
  /** Go to next step */
  nextStep: () => void;
  /** Go to previous step */
  prevStep: () => void;
  /** End the tour */
  endTour: () => void;
  /** Jump to a specific step */
  goToStep: (step: number) => void;
}

export const useTourStore = create<TourState>()(
  persist(
    (set) => ({
      tourCompleted: false,
      tourActive: false,
      currentStep: 0,
      startTour: () => set({ tourActive: true, currentStep: 0 }),
      nextStep: () => set((s) => ({ currentStep: s.currentStep + 1 })),
      prevStep: () => set((s) => ({ currentStep: Math.max(0, s.currentStep - 1) })),
      endTour: () => set({ tourActive: false, tourCompleted: true, currentStep: 0 }),
      goToStep: (step) => set({ currentStep: step }),
    }),
    { name: 'vlm-tour', partialize: (s) => ({ tourCompleted: s.tourCompleted }) },
  ),
);
