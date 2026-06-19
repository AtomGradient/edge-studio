// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import type { ExportResult, BenchmarkRunResult } from '@/api/types';
import { MetricCards } from '@/components/data/MetricCards';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { formatSize } from '@/lib/utils';
import { runBenchmark } from '@/api/endpoints';

export function ExportResultCard({
  result,
  format,
  quantType,
  originalSizeBytes,
  modelDir,
}: {
  result: ExportResult;
  format: string;
  quantType?: string;
  originalSizeBytes?: number;
  modelDir?: string;
}) {
  const [benchTaskId, setBenchTaskId] = useState<string | null>(null);
  const [benchResult, setBenchResult] = useState<BenchmarkRunResult | null>(null);

  const reduction = originalSizeBytes && result.output_size_bytes
    ? ((1 - result.output_size_bytes / originalSizeBytes) * 100)
    : null;

  const handleBenchmark = async () => {
    if (!modelDir) return;
    try {
      const { task_id } = await runBenchmark(modelDir);
      setBenchTaskId(task_id);
    } catch { /* handled by ProgressOverlay */ }
  };

  const handleBenchComplete = (raw: unknown) => {
    setBenchTaskId(null);
    if (raw && typeof raw === 'object') {
      setBenchResult(raw as BenchmarkRunResult);
    }
  };

  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold text-gray-700">{format} Export Result</h3>
      {result.success ? (
        <>
          <MetricCards
            metrics={[
              ...(originalSizeBytes ? [{ label: 'Original Size', value: formatSize(originalSizeBytes) }] : []),
              { label: 'Output Size', value: formatSize(result.output_size_bytes) },
              ...(reduction !== null ? [{ label: 'Reduction', value: `${reduction.toFixed(1)}%` }] : []),
              ...(quantType ? [{ label: 'Quantization', value: quantType }] : []),
              { label: 'Duration', value: `${result.duration_seconds.toFixed(1)}s` },
            ]}
            className="mb-3"
          />
          {reduction !== null && reduction > 0 && (
            <p className="mb-2 text-xs font-medium text-green-600">
              Size reduced by {reduction.toFixed(1)}% ({formatSize(originalSizeBytes! - result.output_size_bytes)} saved)
            </p>
          )}
          <p className="text-xs text-gray-500">
            Output: <span className="font-mono">{result.output_path}</span>
          </p>

          {/* Benchmark section */}
          {modelDir && (
            <div className="mt-4 border-t border-gray-100 pt-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-gray-700">Benchmark Source Model</p>
                  <p className="text-xs text-gray-400">
                    Measures disk, memory, tok/s, and PPL of the source MLX model
                  </p>
                </div>
                <button
                  onClick={handleBenchmark}
                  disabled={!!benchTaskId}
                  className="rounded-lg bg-emerald-500 px-4 py-1.5 text-xs font-medium text-white hover:bg-emerald-600 disabled:opacity-50"
                >
                  Run Benchmark
                </button>
              </div>

              {benchResult?.baseline && (() => {
                const b = benchResult.baseline;
                const speed = b.tokens_per_second;
                const ppl = b.perplexity;
                const mem = b.memory_after_load_mb;

                const speedColor = speed >= 60 ? '#16a34a' : speed >= 30 ? '#ca8a04' : '#dc2626';
                const speedLabel = speed >= 60 ? 'Fast' : speed >= 30 ? 'OK' : 'Slow';

                const pplColor = ppl <= 15 ? '#16a34a' : ppl <= 25 ? '#ca8a04' : '#dc2626';
                const pplLabel = ppl <= 15 ? 'Good' : ppl <= 25 ? 'Acceptable' : 'Poor';

                const memColor = mem <= 2000 ? '#16a34a' : mem <= 4000 ? '#ca8a04' : '#dc2626';
                const memLabel = mem <= 2000 ? 'Mobile-ready' : mem <= 4000 ? 'Mid-range device' : 'Desktop only';

                return (
                  <div className="mt-3 space-y-2">
                    <MetricCards
                      metrics={[
                        { label: 'Disk', value: `${b.disk_size_mb.toFixed(0)} MB` },
                        { label: 'Memory', value: `${mem.toFixed(0)} MB`, color: memColor, subtitle: memLabel },
                        { label: 'Speed', value: `${speed.toFixed(1)} tok/s`, color: speedColor, subtitle: speedLabel },
                        { label: 'PPL', value: ppl.toFixed(2), color: pplColor, subtitle: pplLabel },
                      ]}
                    />
                    <p className={`text-xs font-medium ${
                      speed >= 30 && ppl <= 25 ? 'text-green-600' : 'text-amber-600'
                    }`}>
                      {speed >= 60 && ppl <= 15
                        ? 'Excellent — ready for production deployment'
                        : speed >= 30 && ppl <= 25
                          ? 'Acceptable — suitable for on-device use'
                          : 'Consider optimization — speed or quality may be insufficient'}
                    </p>
                  </div>
                );
              })()}

              {benchTaskId && (
                <ProgressOverlay
                  taskId={benchTaskId}
                  title="Running Benchmark"
                  onComplete={handleBenchComplete}
                  onError={() => setBenchTaskId(null)}
                  onClose={() => setBenchTaskId(null)}
                />
              )}
            </div>
          )}
        </>
      ) : (
        <p className="text-sm text-red-500">{result.error_message || 'Export failed'}</p>
      )}
    </div>
  );
}
