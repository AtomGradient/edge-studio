// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { cn } from '@/lib/utils';

export interface MetricCardData {
  label: string;
  value: string | number;
  subtitle?: string;
  color?: string;
}

interface MetricCardsProps {
  metrics: MetricCardData[];
  className?: string;
}

export function MetricCards({ metrics, className }: MetricCardsProps) {
  return (
    <div className={cn('grid gap-4', className)} style={{
      gridTemplateColumns: `repeat(${Math.min(metrics.length, 4)}, minmax(0, 1fr))`,
    }}>
      {metrics.map((m, i) => (
        <div key={i} className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-medium uppercase tracking-wider text-gray-500">{m.label}</p>
          <p className="mt-1 text-2xl font-bold" style={{ color: m.color }}>
            {m.value}
          </p>
          {m.subtitle && (
            <p className="mt-0.5 text-xs text-gray-400">{m.subtitle}</p>
          )}
        </div>
      ))}
    </div>
  );
}
