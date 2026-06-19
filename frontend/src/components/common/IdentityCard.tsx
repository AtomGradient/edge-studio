// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * IdentityCard — compact "icon + label + value + hint" tile used in the
 * 4-card identity strip pattern (page-optimization-playbook §1-B).
 *
 * Used by /chat, /weights, /kv-cache, /inference (extracted from 4 in-file
 * duplicates 2026-05-04). Keep visually consistent — adding tones / sizes
 * should be additive, not breaking.
 */
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

export type IdentityTone = 'neutral' | 'amber' | 'emerald' | 'red' | 'indigo';

const TONE_CLASS: Record<IdentityTone, string> = {
  neutral: 'text-gray-700 dark:text-stone-300',
  amber: 'text-amber-600 dark:text-amber-400',
  emerald: 'text-emerald-600 dark:text-emerald-400',
  red: 'text-red-600 dark:text-red-400',
  indigo: 'text-indigo-600 dark:text-indigo-400',
};

interface Props {
  icon: ReactNode;
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: IdentityTone;
  className?: string;
}

export function IdentityCard({ icon, label, value, hint, tone = 'neutral', className }: Props) {
  const toneClass = TONE_CLASS[tone];
  return (
    <div className={cn(
      'flex items-center gap-2.5 px-3 py-2 rounded-lg bg-gray-50 dark:bg-stone-900/60 min-w-0',
      className,
    )}>
      <div className={cn('shrink-0', toneClass)}>{icon}</div>
      <div className="min-w-0">
        <div className="text-[10px] uppercase tracking-wider text-gray-400 dark:text-stone-500 leading-none">{label}</div>
        <div className={cn('text-xs font-semibold leading-tight truncate', toneClass)} title={hint}>{value}</div>
      </div>
    </div>
  );
}
