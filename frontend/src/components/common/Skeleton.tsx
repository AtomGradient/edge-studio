// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Skeleton loading placeholder — replaces spinner for better perceived performance.
 * Usage:
 *   <Skeleton className="h-6 w-48" />           — single line
 *   <SkeletonCard />                             — card placeholder
 *   <SkeletonList count={5} />                   — list placeholder
 *   <SkeletonPage />                             — full page placeholder (replaces PageLoader spinner)
 */

import { cn } from '@/lib/utils';

// ---- Base ----

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        'animate-pulse rounded-md bg-gray-200 dark:bg-stone-700',
        className,
      )}
    />
  );
}

// ---- Card ----

interface SkeletonCardProps {
  className?: string;
}

export function SkeletonCard({ className }: SkeletonCardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl border border-gray-200 bg-white p-5 dark:border-stone-700 dark:bg-stone-900',
        className,
      )}
    >
      {/* Title row */}
      <div className="mb-3 flex items-start justify-between">
        <div className="flex-1">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="mt-2 h-4 w-64" />
        </div>
        <Skeleton className="ml-3 h-5 w-16 shrink-0 rounded-full" />
      </div>

      {/* Body lines */}
      <Skeleton className="h-3.5 w-full" />
      <Skeleton className="mt-2 h-3.5 w-3/4" />

      {/* Footer */}
      <div className="mt-4 flex items-center gap-3">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-32" />
      </div>
    </div>
  );
}

// ---- List ----

interface SkeletonListProps {
  count?: number;
  className?: string;
}

export function SkeletonList({ count = 4, className }: SkeletonListProps) {
  return (
    <div className={cn('space-y-3', className)}>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="flex items-center gap-3 rounded-xl border border-gray-200 bg-white px-4 py-3 dark:border-stone-700 dark:bg-stone-900"
        >
          <Skeleton className="h-8 w-8 shrink-0 rounded-lg" />
          <div className="flex-1">
            <Skeleton className="h-4 w-36" />
            <Skeleton className="mt-1.5 h-3 w-56" />
          </div>
          <Skeleton className="h-3 w-16 shrink-0" />
        </div>
      ))}
    </div>
  );
}

// ---- Page (replaces PageLoader spinner) ----

interface SkeletonPageProps {
  className?: string;
}

export function SkeletonPage({ className }: SkeletonPageProps) {
  return (
    <div className={cn('mx-auto max-w-5xl space-y-6 py-8', className)}>
      {/* Page title area */}
      <div className="text-center">
        <Skeleton className="mx-auto h-3 w-32" />
        <Skeleton className="mx-auto mt-3 h-7 w-64" />
        <Skeleton className="mx-auto mt-2 h-4 w-80" />
      </div>

      {/* Metric cards row */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-3 rounded-2xl border border-gray-200 bg-white px-5 py-4 dark:border-stone-700 dark:bg-stone-900"
          >
            <Skeleton className="h-10 w-10 shrink-0 rounded-xl" />
            <div className="flex-1">
              <Skeleton className="h-3 w-12" />
              <Skeleton className="mt-1.5 h-5 w-20" />
            </div>
          </div>
        ))}
      </div>

      {/* Action cards grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div
            key={i}
            className="rounded-2xl border border-gray-200 bg-white p-5 dark:border-stone-700 dark:bg-stone-900"
          >
            <Skeleton className="h-10 w-10 rounded-xl" />
            <Skeleton className="mt-3 h-4 w-28" />
            <Skeleton className="mt-1.5 h-3 w-full" />
          </div>
        ))}
      </div>
    </div>
  );
}
