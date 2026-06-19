// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

export const COLORS = {
  primary: '#6366f1',
  primaryLight: '#818cf8',
  primaryDark: '#4f46e5',
  success: '#22c55e',
  warning: '#f59e0b',
  danger: '#ef4444',
  info: '#3b82f6',
  muted: '#6b7280',

  // Chart colors
  attn: '#3b82f6',
  mlp: '#8b5cf6',
  alive: '#22c55e',
  dead: '#ef4444',

  // Attention patterns
  sink: '#f97316',
  local: '#3b82f6',
  global: '#22c55e',
  sparse: '#9ca3af',
} as const;

export const PATTERN_NAMES = {
  SINK: 'Sink',
  LOCAL: 'Local',
  GLOBAL: 'Global',
  SPARSE: 'Sparse',
} as const;

export const PRIORITY_COLORS = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#22c55e',
} as const;

export const RISK_COLORS = {
  high: '#ef4444',
  medium: '#f59e0b',
  low: '#22c55e',
} as const;
