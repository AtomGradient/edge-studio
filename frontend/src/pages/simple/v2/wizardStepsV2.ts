// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Simple v2 wizard step definitions.
 */

import type { Step } from '@/components/common/StepIndicator';

type TFn = (key: string) => string;

export const WIZARD_STEPS_V2 = (t: TFn): Step[] => [
  { label: t('simple.v2.steps.profile') },
  { label: t('simple.v2.steps.focus') },
  { label: t('simple.v2.steps.tier') },
  { label: t('simple.v2.steps.setup') },
  { label: t('simple.v2.steps.done') },
];
