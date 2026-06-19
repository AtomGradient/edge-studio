// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Map raw backend error messages to friendly i18n keys.
 * Falls back to the raw message if no pattern matches.
 */

type TFn = (key: string) => string;

const ERROR_PATTERNS: [RegExp, string][] = [
  [/connection refused|ECONNREFUSED|Network Error/i, 'simple.error.connectionRefused'],
  [/out of memory|OOM|memory/i, 'simple.error.outOfMemory'],
  [/not found|does not exist|no such file/i, 'simple.error.modelNotFound'],
  [/disk.*full|no space left|ENOSPC/i, 'simple.error.diskFull'],
  [/timeout|timed out/i, 'simple.error.timeout'],
  [/download.*fail|aria2c.*error/i, 'simple.error.downloadFailed'],
  [/load.*fail|parse.*error|invalid.*model/i, 'simple.error.loadFailed'],
  [/export.*fail/i, 'simple.error.exportFailed'],
];

export function friendlyError(raw: string | undefined, t: TFn, fallbackKey: string): string {
  if (!raw) return t(fallbackKey);

  for (const [pattern, key] of ERROR_PATTERNS) {
    if (pattern.test(raw)) return t(key);
  }

  // If the raw message is already an i18n key (starts with "simple."), translate it
  if (raw.startsWith('simple.')) return t(raw);

  // Return raw message as last resort — at least the user sees something
  return raw;
}
