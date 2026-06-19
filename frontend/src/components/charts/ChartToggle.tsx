// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { cn } from '@/lib/utils';

interface ChartToggleProps {
  mode: string;
  options: { value: string; label: string }[];
  onChange: (mode: string) => void;
  className?: string;
}

export function ChartToggle({ mode, options, onChange, className }: ChartToggleProps) {
  return (
    <div className={cn('inline-flex rounded-lg border border-gray-200 bg-gray-50 p-0.5', className)}>
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={cn(
            'rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
            mode === opt.value
              ? 'bg-white text-indigo-700 shadow-sm'
              : 'text-gray-500 hover:text-gray-700',
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
