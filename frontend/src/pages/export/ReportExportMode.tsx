// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import type { ModelInfo } from '@/api/types';
import { getTraceResult, getCachedQuality } from '@/api/endpoints';
import { generateHTMLReport, buildReportSections, downloadReport } from '@/lib/reportGenerator';
import { useToastStore } from '@/stores/toastStore';

export function ReportExportMode({ model }: { model: ModelInfo }) {
  const [generating, setGenerating] = useState(false);
  const addToast = useToastStore((s) => s.addToast);

  const handleReport = async () => {
    setGenerating(true);
    try {
      let trace: Record<string, unknown> | null = null;
      let quality: Record<string, unknown> | null = null;

      try { trace = await getTraceResult(model.model_id) as unknown as Record<string, unknown>; } catch { /* no trace */ }
      try { quality = await getCachedQuality(model.model_id); } catch { /* no quality */ }

      const sections = buildReportSections(model, trace, quality);

      if (sections.length === 0) {
        addToast('No analysis data available. Run some analyses first (Inference Trace, Quality Validator, etc.)', 'warning');
        return;
      }

      const html = generateHTMLReport({
        model,
        sections,
        generatedAt: new Date().toLocaleString(),
      });

      const filename = `${model.model_name.replace(/[^a-zA-Z0-9]/g, '_')}_report.html`;
      downloadReport(html, filename);
      addToast(`Report downloaded: ${filename}`, 'success');
    } catch {
      addToast('Failed to generate report.', 'error');
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 space-y-3">
      <p className="text-sm text-gray-600">
        Generate a standalone HTML report containing all available analysis results
        (inference trace, quality metrics, etc.). The report can be opened in any browser
        and printed to PDF.
      </p>
      <ul className="text-xs text-gray-500 space-y-1">
        <li>- Model information and architecture details</li>
        <li>- Inference trace results (if available)</li>
        <li>- Quality validation metrics (if available)</li>
      </ul>
      <button
        onClick={handleReport}
        disabled={generating}
        className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
      >
        {generating ? 'Generating...' : 'Generate HTML Report'}
      </button>
    </div>
  );
}
