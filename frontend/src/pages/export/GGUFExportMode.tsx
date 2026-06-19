// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import type { ModelInfo, ExportResult } from '@/api/types';
import { exportGGUF } from '@/api/endpoints';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { ExportResultCard } from './ExportResultCard';

const GGUF_QUANT_TYPES = [
  { value: 'f16', label: 'f16 (lossless, largest)' },
  { value: 'q8_0', label: 'q8_0 (near-lossless)' },
  { value: 'q6_k', label: 'q6_k (excellent quality)' },
  { value: 'q5_k_m', label: 'q5_k_m (good quality)' },
  { value: 'q4_k_m', label: 'q4_k_m (balanced, recommended)' },
  { value: 'q4_0', label: 'q4_0 (fastest)' },
  { value: 'q3_k_m', label: 'q3_k_m (small)' },
  { value: 'q2_k', label: 'q2_k (smallest)' },
];

export function GGUFExportMode({ model }: { model: ModelInfo }) {
  const [quantType, setQuantType] = useState('q4_k_m');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<ExportResult | null>(null);

  const handleExport = async () => {
    try {
      const { task_id } = await exportGGUF(model.model_id, quantType);
      setTaskId(task_id);
    } catch { /* error handled by ProgressOverlay */ }
  };

  return (
    <>
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 space-y-3">
        <div className="max-w-md">
          <label className="mb-1 block text-xs text-gray-500">Quantization type</label>
          <select
            value={quantType}
            onChange={(e) => setQuantType(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
          >
            {GGUF_QUANT_TYPES.map((qt) => (
              <option key={qt.value} value={qt.value}>
                {qt.label}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={handleExport}
          disabled={!!taskId}
          className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          Export to GGUF
        </button>
      </div>

      {taskId && (
        <ProgressOverlay
          taskId={taskId}
          title="Exporting to GGUF"
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
          format="GGUF"
          quantType={quantType}
          originalSizeBytes={model.total_size_bytes}
          modelDir={result.output_path}
        />
      )}
    </>
  );
}
