// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import axios from 'axios';

interface PollOptions {
  intervalMs?: number;
  maxRetries?: number;
  backoff?: boolean;
  onProgress?: (percent: number, message: string) => void;
}

interface PollResult<T = unknown> {
  success: boolean;
  result?: T;
  error?: string;
}

/**
 * Poll a background task until completion.
 * Uses exponential backoff by default.
 */
export async function pollTask<T = unknown>(
  taskId: string,
  options: PollOptions = {},
): Promise<PollResult<T>> {
  const {
    intervalMs = 1000,
    maxRetries = 600,
    backoff = true,
    onProgress,
  } = options;

  let delay = intervalMs;
  for (let i = 0; i < maxRetries; i++) {
    await new Promise((r) => setTimeout(r, delay));
    try {
      const { data } = await axios.get(`/api/task/${taskId}`);
      if (data.progress !== undefined && onProgress) {
        onProgress(Math.round(data.progress * 100), data.message || '');
      }
      if (data.status === 'complete') {
        // Fetch full result
        const resultRes = await axios.get(`/api/task/${taskId}/result`);
        return { success: true, result: resultRes.data.result as T };
      }
      if (data.status === 'error' || data.status === 'cancelled') {
        return { success: false, error: data.error || data.message || 'Task failed' };
      }
    } catch (err: unknown) {
      // Network error — keep retrying
      if (i === maxRetries - 1) {
        const message = err instanceof Error ? err.message : 'Polling failed';
        return { success: false, error: message };
      }
    }
    if (backoff && delay < 5000) {
      delay = Math.min(delay * 1.2, 5000);
    }
  }
  return { success: false, error: 'Task timed out' };
}
