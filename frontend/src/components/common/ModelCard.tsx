// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * ModelCard — displays a recommended model with fit indicator and action button.
 */

import { Download, Check, HardDrive, Sparkles, ExternalLink } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useT } from '@/i18n';
import type { RecommendedModel } from '@/stores/wizardStore';

interface ModelCardProps {
  model: RecommendedModel;
  isSelected?: boolean;
  isDownloading?: boolean;
  onSelect?: () => void;
  className?: string;
}

const TIER_STYLES = {
  high: { labelKey: 'simple.v1.qualityHigh', color: 'text-amber-600 bg-amber-50 dark:text-amber-400 dark:bg-amber-900/30' },
  medium: { labelKey: 'simple.v1.qualityBalanced', color: 'text-blue-600 bg-blue-50 dark:text-blue-400 dark:bg-blue-900/30' },
  balanced: { labelKey: 'simple.v1.qualityCompact', color: 'text-green-600 bg-green-50 dark:text-green-400 dark:bg-green-900/30' },
} as const;

export function ModelCard({ model, isSelected, isDownloading, onSelect, className }: ModelCardProps) {
  const t = useT();
  const tier = TIER_STYLES[model.quality_tier as keyof typeof TIER_STYLES] || TIER_STYLES.balanced;

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={isDownloading}
      className={cn(
        'group relative w-full rounded-2xl border p-5 text-left transition-all duration-200',
        isSelected
          ? 'border-stone-900 bg-stone-50 ring-1 ring-stone-900 dark:border-stone-100 dark:bg-stone-800/50 dark:ring-stone-100'
          : 'border-stone-200 bg-white hover:border-stone-300 hover:shadow-md dark:border-stone-800 dark:bg-stone-900 dark:hover:border-stone-700',
        !model.fits_device && 'opacity-60',
        className,
      )}
    >
      {/* Selected indicator */}
      {isSelected && (
        <div className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900">
          <Check size={14} strokeWidth={3} />
        </div>
      )}

      {/* Header row */}
      <div className="mb-3 flex items-start justify-between">
        <div>
          <h4 className="text-base font-semibold text-stone-900 dark:text-stone-100">
            {model.name}
          </h4>
          <p className="mt-0.5 text-sm text-stone-500 dark:text-stone-400">
            {model.description}
          </p>
        </div>
        <span className={cn('ml-3 shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium', tier.color)}>
          {t(tier.labelKey)}
        </span>
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-4 text-xs text-stone-500 dark:text-stone-400">
        <span className="flex items-center gap-1">
          <HardDrive size={13} />
          {model.estimated_size_gb} GB
        </span>
        {model.fits_device ? (
          <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
            <Sparkles size={13} />
            {t('simple.v1.modelFits', { headroom: String(model.headroom_gb) })}
          </span>
        ) : (
          <span className="text-red-500 dark:text-red-400">
            {t('simple.v1.modelTooLarge')}
          </span>
        )}
      </div>

      {/* Download hint */}
      <div className="mt-3 flex items-center gap-1.5 text-xs text-stone-400 dark:text-stone-500">
        <Download size={12} />
        <span className="truncate">{model.download_hint}</span>
        <a
          href={`https://huggingface.co/${model.download_hint}`}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="ml-auto shrink-0 rounded p-0.5 text-stone-300 transition-colors hover:text-stone-600 dark:text-stone-600 dark:hover:text-stone-300"
          title={t('simple.v1.viewOnHF')}
        >
          <ExternalLink size={12} />
        </a>
      </div>
    </button>
  );
}
