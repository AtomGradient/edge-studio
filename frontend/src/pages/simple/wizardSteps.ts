// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Shared wizard step definitions.
 */

import type { Step } from '@/components/common/StepIndicator';

type TFn = (key: string) => string;

export const WIZARD_STEPS = (t: TFn): Step[] => [
  { label: t('simple.steps.welcome') },
  { label: t('simple.steps.device') },
  { label: t('simple.steps.model') },
  { label: t('simple.steps.optimize') },
  { label: t('simple.steps.test') },
  { label: t('simple.steps.export') },
];
