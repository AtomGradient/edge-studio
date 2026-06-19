// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import type { ReactNode } from 'react';
import { Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';

interface AskModelFabProps {
  label: ReactNode;
  modelName?: string;
  onClick: () => void;
  icon?: ReactNode;
  className?: string;
}

export function AskModelFab({
  label,
  modelName,
  onClick,
  icon,
  className,
}: AskModelFabProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'fixed bottom-6 right-6 z-40 inline-flex min-h-12 max-w-[min(360px,calc(100vw-2rem))] items-center gap-2 rounded-full border px-3.5 py-2.5 text-sm font-semibold transition-all',
        'border-gray-950 bg-gray-950 text-white shadow-[0_18px_48px_rgba(0,0,0,0.35)] ring-2 ring-white/90 hover:-translate-y-0.5 hover:bg-black hover:shadow-[0_22px_56px_rgba(0,0,0,0.45)]',
        'dark:border-white dark:bg-white dark:text-gray-950 dark:ring-gray-950/80 dark:hover:bg-gray-100',
        className,
      )}
      aria-label={typeof label === 'string' && modelName ? `${label} ${modelName}` : undefined}
    >
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/15 text-white dark:bg-gray-950/10 dark:text-gray-950">
        {icon ?? <Sparkles size={15} />}
      </span>
      <span className="whitespace-nowrap">{label}</span>
      {modelName && (
        <span className="min-w-0 max-w-[150px] truncate rounded-full bg-white/10 px-2 py-1 text-[10px] font-medium text-white/85 dark:bg-gray-950/10 dark:text-gray-700">
          {modelName}
        </span>
      )}
    </button>
  );
}
