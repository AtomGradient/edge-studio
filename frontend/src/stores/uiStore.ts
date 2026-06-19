// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UIState {
  sidebarOpen: boolean;
  toggleSidebar: () => void;

  fileBrowserOpen: boolean;
  setFileBrowserOpen: (open: boolean) => void;

  hfPickerOpen: boolean;
  setHfPickerOpen: (open: boolean) => void;

  prefer3D: boolean;
  togglePrefer3D: () => void;

  darkMode: boolean;
  toggleDarkMode: () => void;

  userMode: 'beginner' | 'simple' | 'advanced';
  setUserMode: (mode: 'beginner' | 'simple' | 'advanced') => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

      fileBrowserOpen: false,
      setFileBrowserOpen: (open) => set({ fileBrowserOpen: open }),

      hfPickerOpen: false,
      setHfPickerOpen: (open) => set({ hfPickerOpen: open }),

      prefer3D: false,
      togglePrefer3D: () => set((s) => ({ prefer3D: !s.prefer3D })),

      darkMode: false,
      toggleDarkMode: () =>
        set((s) => {
          const next = !s.darkMode;
          document.documentElement.classList.toggle('dark', next);
          return { darkMode: next };
        }),

      userMode: 'beginner',
      setUserMode: (mode) => set({ userMode: mode }),
    }),
    {
      name: 'vlm-ui',
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        darkMode: state.darkMode,
        prefer3D: state.prefer3D,
        userMode: state.userMode,
      }),
      onRehydrateStorage: () => (state) => {
        if (state?.darkMode) {
          document.documentElement.classList.add('dark');
        }
        // Migrate old 'expert' → 'advanced'
        if ((state?.userMode as string) === 'expert') {
          state!.userMode = 'advanced';
        }
      },
    },
  ),
);
