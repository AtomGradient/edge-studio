// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useState } from 'react';
import { connectTaskWebSocket } from '@/api/websocket';
import { getTaskStatus, getTaskResult, cancelTask } from '@/api/endpoints';
import { AlertCircle, Ban } from 'lucide-react';
import type { TaskEvent } from '@/api/types';

interface ProgressOverlayProps {
  taskId: string | null;
  title?: string;
  onComplete?: (result: unknown) => void;
  onError?: (error: string) => void;
  onClose?: () => void;
}

type OverlayStatus = 'running' | 'complete' | 'error' | 'cancelled';

export function ProgressOverlay({ taskId, title, onComplete, onError, onClose }: ProgressOverlayProps) {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('Starting...');
  const [status, setStatus] = useState<OverlayStatus>('running');
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    /* eslint-disable react-hooks/set-state-in-effect -- reset overlay state when task id changes */
    setProgress(0);
    setMessage('Starting...');
    setStatus('running');
    setError(null);
    setCancelling(false);
    /* eslint-enable react-hooks/set-state-in-effect */

    // Try WebSocket first, fall back to polling
    const cleanup = connectTaskWebSocket(
      taskId,
      (event: TaskEvent) => {
        if (event.type === 'progress') {
          setProgress(event.percent ?? 0);
          setMessage(event.message ?? '');
        } else if (event.type === 'complete') {
          setStatus('complete');
          setProgress(1);
          setMessage('Complete');
          getTaskResult(taskId).then(
            (data) => onComplete?.(data.result),
            () => onComplete?.(null),
          );
        } else if (event.type === 'cancelled') {
          setStatus('cancelled');
          setMessage(event.message ?? 'Operation cancelled');
        } else if (event.type === 'error') {
          setStatus('error');
          setError(event.message ?? 'Unknown error');
          onError?.(event.message ?? 'Unknown error');
        }
      },
      () => {
        // WebSocket closed — check status via polling
        if (status === 'running') {
          getTaskStatus(taskId).then((data) => {
            if (data.status === 'complete') {
              setStatus('complete');
              getTaskResult(taskId).then(
                (d) => onComplete?.(d.result),
                () => onComplete?.(null),
              );
            } else if (data.status === 'cancelled') {
              setStatus('cancelled');
            } else if (data.status === 'error') {
              setStatus('error');
              setError(data.error ?? 'Unknown error');
            }
          });
        }
      },
    );

    return cleanup;
  }, [taskId]);

  const handleCancel = async () => {
    if (!taskId || cancelling) return;
    setCancelling(true);
    try {
      await cancelTask(taskId);
    } catch {
      // Cancellation event will arrive via WebSocket; ignore HTTP errors
      setCancelling(false);
    }
  };

  if (!taskId) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-xl bg-white p-6 shadow-2xl">
        <h3 className="mb-4 text-lg font-semibold text-gray-900">{title ?? 'Processing...'}</h3>

        {status === 'running' && (
          <>
            <div className="mb-2 h-2 w-full overflow-hidden rounded-full bg-gray-200">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all duration-300"
                style={{ width: `${Math.max(progress * 100, 2)}%` }}
              />
            </div>
            <p className="mb-4 text-sm text-gray-500">{message}</p>
            <button
              onClick={handleCancel}
              disabled={cancelling}
              className="w-full rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 disabled:opacity-50"
            >
              {cancelling ? 'Cancelling...' : 'Cancel'}
            </button>
          </>
        )}

        {status === 'complete' && (
          <div className="space-y-3">
            <p className="text-sm text-green-600 font-medium">Completed successfully</p>
            <button
              onClick={onClose}
              className="w-full rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-600"
            >
              Close
            </button>
          </div>
        )}

        {status === 'cancelled' && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Ban size={18} className="text-gray-500" />
              <p className="text-sm text-gray-600 font-medium">Operation cancelled</p>
            </div>
            <button
              onClick={onClose}
              className="w-full rounded-lg bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-300"
            >
              Close
            </button>
          </div>
        )}

        {status === 'error' && (
          <div className="space-y-3">
            <div className="flex items-start gap-2">
              <AlertCircle size={18} className="mt-0.5 shrink-0 text-red-500" />
              <p className="text-sm text-red-600">{error}</p>
            </div>
            <button
              onClick={onClose}
              className="w-full rounded-lg bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-300"
            >
              Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
