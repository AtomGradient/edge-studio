// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Design tokens for simple mode — Anthropic-inspired warm aesthetic.
 *
 * Warm stone palette, terracotta/amber accents, generous spacing,
 * soft shadows, Inter font family.
 */

export const colors = {
  // Warm neutral palette (stone)
  stone: {
    50: '#fafaf9',
    100: '#f5f5f4',
    200: '#e7e5e4',
    300: '#d6d3d1',
    400: '#a8a29e',
    500: '#78716c',
    600: '#57534e',
    700: '#44403c',
    800: '#292524',
    900: '#1c1917',
    950: '#0c0a09',
  },

  // Warm accent — terracotta
  accent: {
    50: '#fdf4f0',
    100: '#fbe8de',
    200: '#f7cebc',
    300: '#f0ab8d',
    400: '#e8845b',
    500: '#e06836',  // primary accent
    600: '#c9502a',
    700: '#a63e22',
    800: '#863420',
    900: '#6e2d1f',
  },

  // Warm secondary — amber
  amber: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
  },

  // Success — warm green
  success: {
    50: '#f0fdf4',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
  },

  // Info — warm blue
  info: {
    50: '#eff6ff',
    400: '#60a5fa',
    500: '#3b82f6',
    600: '#2563eb',
  },
} as const;

export const spacing = {
  wizard: {
    /** Padding around wizard content area */
    contentPadding: '2rem',
    /** Gap between wizard sections */
    sectionGap: '2.5rem',
    /** Max width for wizard content */
    maxWidth: '48rem',
    /** Card padding */
    cardPadding: '1.5rem',
  },
} as const;

export const shadows = {
  /** Soft card shadow */
  card: '0 1px 3px 0 rgb(0 0 0 / 0.04), 0 1px 2px -1px rgb(0 0 0 / 0.04)',
  /** Elevated card (hover) */
  cardHover: '0 4px 6px -1px rgb(0 0 0 / 0.06), 0 2px 4px -2px rgb(0 0 0 / 0.06)',
  /** Soft glow for focus states */
  focus: '0 0 0 3px rgb(224 104 54 / 0.2)',
} as const;

export const typography = {
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
  heading: {
    hero: 'text-4xl font-semibold tracking-tight',
    h1: 'text-2xl font-semibold tracking-tight',
    h2: 'text-xl font-medium',
    h3: 'text-lg font-medium',
    subtitle: 'text-base text-stone-500',
  },
} as const;

export const animation = {
  /** Standard transition for interactive elements */
  base: 'transition-all duration-200 ease-out',
  /** Slow transition for large state changes */
  slow: 'transition-all duration-500 ease-out',
} as const;
