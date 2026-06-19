// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Theme system for pro mode — CSS variable-driven theming.
 *
 * Provides a structured approach to theming with semantic color tokens,
 * density controls, and dark mode support via CSS custom properties.
 */

export const themeTokens = {
  light: {
    // Surfaces
    '--surface-primary': '#ffffff',
    '--surface-secondary': '#f9fafb',
    '--surface-tertiary': '#f3f4f6',
    '--surface-elevated': '#ffffff',
    '--surface-overlay': 'rgba(0, 0, 0, 0.5)',

    // Borders
    '--border-primary': '#e5e7eb',
    '--border-secondary': '#f3f4f6',
    '--border-focus': '#6366f1',

    // Text
    '--text-primary': '#111827',
    '--text-secondary': '#6b7280',
    '--text-tertiary': '#9ca3af',
    '--text-inverse': '#ffffff',

    // Accent
    '--accent-primary': '#6366f1',
    '--accent-primary-hover': '#4f46e5',
    '--accent-subtle': 'rgba(99, 102, 241, 0.1)',

    // Semantic
    '--semantic-success': '#22c55e',
    '--semantic-success-subtle': 'rgba(34, 197, 94, 0.1)',
    '--semantic-warning': '#f59e0b',
    '--semantic-warning-subtle': 'rgba(245, 158, 11, 0.1)',
    '--semantic-danger': '#ef4444',
    '--semantic-danger-subtle': 'rgba(239, 68, 68, 0.1)',
    '--semantic-info': '#3b82f6',
    '--semantic-info-subtle': 'rgba(59, 130, 246, 0.1)',
  },

  dark: {
    '--surface-primary': '#111827',
    '--surface-secondary': '#1f2937',
    '--surface-tertiary': '#374151',
    '--surface-elevated': '#1f2937',
    '--surface-overlay': 'rgba(0, 0, 0, 0.7)',

    '--border-primary': '#374151',
    '--border-secondary': '#1f2937',
    '--border-focus': '#818cf8',

    '--text-primary': '#f9fafb',
    '--text-secondary': '#9ca3af',
    '--text-tertiary': '#6b7280',
    '--text-inverse': '#111827',

    '--accent-primary': '#818cf8',
    '--accent-primary-hover': '#6366f1',
    '--accent-subtle': 'rgba(129, 140, 248, 0.15)',

    '--semantic-success': '#4ade80',
    '--semantic-success-subtle': 'rgba(74, 222, 128, 0.15)',
    '--semantic-warning': '#fbbf24',
    '--semantic-warning-subtle': 'rgba(251, 191, 36, 0.15)',
    '--semantic-danger': '#f87171',
    '--semantic-danger-subtle': 'rgba(248, 113, 113, 0.15)',
    '--semantic-info': '#60a5fa',
    '--semantic-info-subtle': 'rgba(96, 165, 250, 0.15)',
  },
} as const;

/** Density presets for pro mode UI */
export const density = {
  compact: {
    cellPadding: '0.25rem 0.5rem',
    rowHeight: '2rem',
    fontSize: '0.75rem',
    gap: '0.25rem',
  },
  normal: {
    cellPadding: '0.5rem 0.75rem',
    rowHeight: '2.5rem',
    fontSize: '0.8125rem',
    gap: '0.5rem',
  },
  comfortable: {
    cellPadding: '0.75rem 1rem',
    rowHeight: '3rem',
    fontSize: '0.875rem',
    gap: '0.75rem',
  },
} as const;
