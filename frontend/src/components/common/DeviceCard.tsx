// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * DeviceCard — displays detected device hardware info.
 * Shows chip, memory, GPU cores, and max model capacity.
 */

import { Cpu, MemoryStick, Monitor, Gauge } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { SystemInfo } from '@/stores/wizardStore';

interface DeviceCardProps {
  info: SystemInfo;
  className?: string;
}

export function DeviceCard({ info, className }: DeviceCardProps) {
  const usedPercent = info.max_model_size_gb
    ? Math.min(100, (info.max_model_size_gb / info.ram_gb) * 100)
    : 60;

  return (
    <div className={cn(
      'rounded-2xl border border-stone-200 bg-white p-6 dark:border-stone-800 dark:bg-stone-900',
      className,
    )}>
      {/* Chip name - hero */}
      <div className="mb-6 text-center">
        <div className="mb-1 text-sm font-medium text-stone-500 dark:text-stone-400">
          {info.os} {info.os_version}
        </div>
        <h3 className="text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {info.chip}
        </h3>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-4">
        <StatItem
          icon={<MemoryStick size={18} />}
          label="Memory"
          value={`${info.ram_gb} GB`}
        />
        <StatItem
          icon={<Cpu size={18} />}
          label="GPU Cores"
          value={info.gpu_cores > 0 ? `${info.gpu_cores}` : '--'}
        />
        <StatItem
          icon={<Monitor size={18} />}
          label="Architecture"
          value={info.arch}
        />
        <StatItem
          icon={<Gauge size={18} />}
          label="Max Model"
          value={info.max_model_size_gb ? `${info.max_model_size_gb} GB` : '--'}
        />
      </div>

      {/* Capacity bar */}
      {info.max_model_size_gb && (
        <div className="mt-6">
          <div className="mb-1.5 flex items-center justify-between text-xs text-stone-500 dark:text-stone-400">
            <span>Model capacity</span>
            <span>{info.max_model_size_gb} GB available</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-stone-100 dark:bg-stone-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-amber-400 to-green-400 transition-all duration-500"
              style={{ width: `${usedPercent}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function StatItem({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl bg-stone-50 px-4 py-3 dark:bg-stone-800/50">
      <div className="text-stone-400 dark:text-stone-500">{icon}</div>
      <div>
        <div className="text-xs text-stone-500 dark:text-stone-400">{label}</div>
        <div className="text-sm font-medium text-stone-900 dark:text-stone-100">{value}</div>
      </div>
    </div>
  );
}
