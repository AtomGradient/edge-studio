// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import type { ModelInfo, ExportResult } from '@/api/types';
import { exportCoreML } from '@/api/endpoints';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { ExportResultCard } from './ExportResultCard';

const COMPUTE_UNITS = [
  { value: 'ALL', label: 'ALL (CPU + GPU + Neural Engine)' },
  { value: 'CPU_AND_GPU', label: 'CPU + GPU only' },
  { value: 'CPU_AND_NE', label: 'CPU + Neural Engine only' },
];

export function CoreMLExportMode({ model }: { model: ModelInfo }) {
  const [computeUnits, setComputeUnits] = useState('ALL');
  const [maxSeqLen, setMaxSeqLen] = useState(512);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<ExportResult | null>(null);

  const handleExport = async () => {
    try {
      const { task_id } = await exportCoreML(model.model_id, computeUnits, maxSeqLen);
      setTaskId(task_id);
    } catch { /* error handled by ProgressOverlay */ }
  };

  return (
    <>
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 space-y-3">
        <div className="max-w-md">
          <label className="mb-1 block text-xs text-gray-500">Compute units</label>
          <select
            value={computeUnits}
            onChange={(e) => setComputeUnits(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
          >
            {COMPUTE_UNITS.map((cu) => (
              <option key={cu.value} value={cu.value}>
                {cu.label}
              </option>
            ))}
          </select>
        </div>
        <div className="max-w-[200px]">
          <label className="mb-1 block text-xs text-gray-500">Max sequence length</label>
          <input
            type="number"
            min={32}
            max={4096}
            step={32}
            value={maxSeqLen}
            onChange={(e) => setMaxSeqLen(Number(e.target.value))}
            className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
          />
        </div>
        <button
          onClick={handleExport}
          disabled={!!taskId}
          className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          Export to CoreML
        </button>
      </div>

      {taskId && (
        <ProgressOverlay
          taskId={taskId}
          title="Exporting to CoreML"
          onComplete={(r) => {
            setResult(r as ExportResult);
            setTaskId(null);
          }}
          onError={() => setTaskId(null)}
          onClose={() => setTaskId(null)}
        />
      )}

      {result && (
        <ExportResultCard
          result={result}
          format="CoreML"
          originalSizeBytes={model.total_size_bytes}
          modelDir={model.model_dir}
        />
      )}
    </>
  );
}
