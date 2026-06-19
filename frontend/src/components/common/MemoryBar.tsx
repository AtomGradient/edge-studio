// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * MemoryBar — shows model memory usage vs device capacity.
 *
 * Displayed at the bottom of the sidebar.
 * Fetches system info once and updates when model changes.
 */

import { useEffect, useState } from 'react';
import { getSystemInfo } from '@/api/endpoints';
import { useModelStore } from '@/stores/modelStore';
import type { SystemInfo } from '@/api/types';
import { cn } from '@/lib/utils';

interface MemoryBarProps {
  collapsed?: boolean;
}

export function MemoryBar({ collapsed }: MemoryBarProps) {
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);
  const model = useModelStore((s) => s.currentModel);

  useEffect(() => {
    getSystemInfo().then(setSysInfo).catch(() => {});
  }, []);

  if (!sysInfo || !model) return null;

  const modelGB = model.total_size_bytes / (1024 ** 3);
  const totalGB = sysInfo.total_memory_gb;
  const usagePct = totalGB > 0 ? Math.min((modelGB / totalGB) * 100, 100) : 0;
  const isHigh = usagePct > 80;
  const isWarn = usagePct > 60;

  if (collapsed) {
    return (
      <div className="flex flex-col items-center gap-1 border-t border-stone-200 py-3 dark:border-stone-800">
        <div
          className={cn(
            'h-2 w-6 rounded-full',
            isHigh ? 'bg-red-500' : isWarn ? 'bg-amber-500' : 'bg-green-500',
          )}
          title={`${modelGB.toFixed(1)} / ${totalGB} GB (${usagePct.toFixed(0)}%)`}
        />
        <span className="text-[9px] text-stone-400 dark:text-stone-500">{modelGB.toFixed(1)}G</span>
      </div>
    );
  }

  return (
    <div className="border-t border-stone-200 px-4 py-3 dark:border-stone-800">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="truncate text-xs font-medium text-stone-700 dark:text-stone-300">
          {model.model_name}
        </span>
        <span className="ml-2 shrink-0 text-xs text-stone-500 dark:text-stone-400">
          {modelGB.toFixed(1)} GB
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1.5 w-full rounded-full bg-stone-100 dark:bg-stone-800 overflow-hidden">
        <div
          className={cn(
            'h-full rounded-full transition-all duration-500',
            isHigh ? 'bg-red-500' : isWarn ? 'bg-amber-500' : 'bg-green-500',
          )}
          style={{ width: `${usagePct}%` }}
        />
      </div>

      <div className="mt-1 flex items-baseline justify-between">
        <span className={cn(
          'text-[10px] font-medium',
          isHigh ? 'text-red-600 dark:text-red-400' : 'text-stone-400 dark:text-stone-500',
        )}>
          {usagePct.toFixed(0)}% of {totalGB} GB
        </span>
        <span className="text-[10px] text-stone-400 dark:text-stone-500">
          {sysInfo.device_name}
        </span>
      </div>
    </div>
  );
}
