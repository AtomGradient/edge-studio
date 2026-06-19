// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { create } from 'zustand';

interface PageAskState {
  contexts: Record<string, string>;
  setPageAskContext: (path: string, context: string | null) => void;
}

export const usePageAskStore = create<PageAskState>((set) => ({
  contexts: {},
  setPageAskContext: (path, context) => set((state) => {
    const next = { ...state.contexts };
    if (!context) {
      delete next[path];
    } else {
      next[path] = context;
    }
    return { contexts: next };
  }),
}));
