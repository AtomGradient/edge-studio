// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Shared text processing utilities for chat output.
 *
 * Used by both expert Chat and simple DuplexPanel to ensure
 * consistent handling of model output (thinking blocks, etc.).
 */

const THINK_BLOCK_RE = /<think>[\s\S]*?<\/think>\s*/g;

/** Strip `<think>...</think>` blocks from model output. */
export function stripThinking(text: string): string {
  // Remove complete blocks
  let cleaned = text.replace(THINK_BLOCK_RE, '');
  // Remove unclosed <think> block at the end (model still thinking)
  const idx = cleaned.indexOf('<think>');
  if (idx !== -1) cleaned = cleaned.slice(0, idx);
  return cleaned.trimStart();
}
