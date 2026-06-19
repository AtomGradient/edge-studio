// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * InsightPanel — turns raw data into human-readable insights.
 *
 * Placed at the top of analysis pages, below PageHeader.
 * Each insight has: what you see → what it means → what to do.
 */

import { useNavigate } from 'react-router-dom';
import { ArrowRight, Info, AlertTriangle, CheckCircle, Lightbulb } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Insight {
  /** One-line summary of the finding */
  title: string;
  /** What it means in plain language */
  description: string;
  /** Severity affects icon and color */
  severity: 'info' | 'good' | 'warning' | 'tip';
  /** Optional action: where to go */
  action?: {
    label: string;
    path: string;
  };
}

interface InsightPanelProps {
  insights: Insight[];
  className?: string;
}

const SEVERITY_CONFIG = {
  info: {
    icon: Info,
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-100 dark:border-blue-800/40',
    iconColor: 'text-blue-500 dark:text-blue-400',
    titleColor: 'text-blue-900 dark:text-blue-200',
    descColor: 'text-blue-700 dark:text-blue-300',
    btnBg: 'bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-800/40 dark:text-blue-300 dark:hover:bg-blue-800/60',
  },
  good: {
    icon: CheckCircle,
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-green-100 dark:border-green-800/40',
    iconColor: 'text-green-500 dark:text-green-400',
    titleColor: 'text-green-900 dark:text-green-200',
    descColor: 'text-green-700 dark:text-green-300',
    btnBg: 'bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-800/40 dark:text-green-300 dark:hover:bg-green-800/60',
  },
  warning: {
    icon: AlertTriangle,
    bg: 'bg-amber-50 dark:bg-amber-900/20',
    border: 'border-amber-100 dark:border-amber-800/40',
    iconColor: 'text-amber-500 dark:text-amber-400',
    titleColor: 'text-amber-900 dark:text-amber-200',
    descColor: 'text-amber-700 dark:text-amber-300',
    btnBg: 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-800/40 dark:text-amber-300 dark:hover:bg-amber-800/60',
  },
  tip: {
    icon: Lightbulb,
    bg: 'bg-gray-50 dark:bg-gray-800/50',
    border: 'border-gray-200 dark:border-gray-700',
    iconColor: 'text-gray-500 dark:text-gray-400',
    titleColor: 'text-gray-900 dark:text-gray-200',
    descColor: 'text-gray-600 dark:text-gray-400',
    btnBg: 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600',
  },
};

export function InsightPanel({ insights, className }: InsightPanelProps) {
  const navigate = useNavigate();

  if (insights.length === 0) return null;

  return (
    <div className={cn('mb-6 space-y-2', className)}>
      {insights.map((insight, i) => {
        const config = SEVERITY_CONFIG[insight.severity];
        const Icon = config.icon;

        return (
          <div
            key={i}
            className={cn(
              'flex items-start gap-3 rounded-xl border px-4 py-3',
              config.bg, config.border,
            )}
          >
            <Icon size={18} className={cn('mt-0.5 shrink-0', config.iconColor)} />
            <div className="min-w-0 flex-1">
              <p className={cn('text-sm font-medium', config.titleColor)}>
                {insight.title}
              </p>
              <p className={cn('mt-0.5 text-xs leading-relaxed', config.descColor)}>
                {insight.description}
              </p>
            </div>
            {insight.action && (
              <button
                type="button"
                onClick={() => navigate(insight.action!.path)}
                className={cn(
                  'mt-0.5 flex shrink-0 items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition-colors',
                  config.btnBg,
                )}
              >
                {insight.action.label}
                <ArrowRight size={12} />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
