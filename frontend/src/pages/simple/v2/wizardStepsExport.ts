// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Export wizard step definitions (Phase 2).
 * Separate from Phase 1 steps — export has its own 2-step flow.
 */

import type { Step } from '@/components/common/StepIndicator';

type TFn = (key: string) => string;

export const EXPORT_STEPS = (t: TFn): Step[] => [
  { label: t('simple.v2.export.steps.device') },
  { label: t('simple.v2.export.steps.generate') },
];
