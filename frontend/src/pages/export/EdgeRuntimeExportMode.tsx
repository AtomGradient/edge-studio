// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import JSZip from 'jszip';
import type { ModelInfo, EdgeRuntimeExportResult } from '@/api/types';
import { generateEdgeRuntime } from '@/api/endpoints';
import { useToastStore } from '@/stores/toastStore';

export function EdgeRuntimeExportMode({ model }: { model: ModelInfo }) {
  const [optimizedDir, setOptimizedDir] = useState('');
  const [result, setResult] = useState<EdgeRuntimeExportResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [codeTab, setCodeTab] = useState<'package' | 'main' | 'readme'>('package');
  const addToast = useToastStore((s) => s.addToast);

  const handleExport = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await generateEdgeRuntime(model.model_id, optimizedDir || undefined);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      addToast('Copied to clipboard', 'success');
    });
  };

  const downloadZip = async () => {
    if (!result) return;
    const zip = new JSZip();
    zip.file('Package.swift', result.package_swift);
    zip.file('Sources/EdgeKitDemo/main.swift', result.main_swift);
    zip.file('README.md', result.readme);
    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${result.model_name}-EdgeKit.zip`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 space-y-4">
        <p className="text-sm text-gray-600">
          Generate a Swift CLI project using <strong>EdgeKit</strong> to run this model locally.
          Verify tok/s performance and Edge Studio optimization effects, then use the SDK to build your App.
        </p>

        <div className="max-w-xl">
          <label className="mb-1 block text-xs text-gray-500">
            Optimized model path (optional — leave blank to use current model)
          </label>
          <input
            type="text"
            value={optimizedDir}
            onChange={(e) => setOptimizedDir(e.target.value)}
            placeholder={model.model_dir}
            className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm font-mono"
          />
        </div>

        <button
          onClick={handleExport}
          disabled={loading}
          className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          {loading ? 'Generating...' : 'Generate EdgeKit Project'}
        </button>

        {error && <p className="text-sm text-red-500">{error}</p>}
      </div>

      {result && (
        <div className="mt-6 rounded-xl border border-indigo-200 bg-white p-4">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-gray-700">
                EdgeKit Project — {result.model_name}
              </h3>
              {result.is_optimized && (
                <p className="mt-0.5 text-xs text-green-600">
                  Optimizations: {result.optimization_summary}
                </p>
              )}
            </div>
            <button
              onClick={downloadZip}
              className="rounded-lg bg-indigo-500 px-4 py-1.5 text-xs font-medium text-white hover:bg-indigo-600"
            >
              Download All Files
            </button>
          </div>

          {/* Run command box */}
          <div className="mb-4 rounded-lg bg-gray-900 px-4 py-3">
            <p className="mb-1 text-xs text-gray-400">Run command:</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-sm text-green-400">$ {result.run_command}</code>
              <button
                onClick={() => copyToClipboard(result.run_command)}
                className="rounded px-2 py-0.5 text-xs text-gray-400 hover:bg-gray-700 hover:text-gray-200"
              >
                Copy
              </button>
            </div>
          </div>

          {/* Code file tabs */}
          <div className="mb-2 flex gap-1">
            {(['package', 'main', 'readme'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setCodeTab(tab)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  codeTab === tab
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab === 'package' ? 'Package.swift' : tab === 'main' ? 'main.swift' : 'README.md'}
              </button>
            ))}
          </div>

          <div className="relative">
            <button
              onClick={() => copyToClipboard(
                codeTab === 'package'
                  ? result.package_swift
                  : codeTab === 'main'
                    ? result.main_swift
                    : result.readme
              )}
              className="absolute right-2 top-2 z-10 rounded bg-gray-700 px-2 py-0.5 text-xs text-gray-300 hover:bg-gray-600"
            >
              Copy
            </button>
            <pre className="max-h-[500px] overflow-auto rounded-lg bg-gray-900 p-4 text-xs text-gray-100">
              {codeTab === 'package'
                ? result.package_swift
                : codeTab === 'main'
                  ? result.main_swift
                  : result.readme}
            </pre>
          </div>
        </div>
      )}
    </>
  );
}
