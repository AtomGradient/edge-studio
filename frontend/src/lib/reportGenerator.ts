// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Generate a standalone HTML analysis report from collected model data.
 * Opens the report in a new browser tab / triggers download.
 */

import type { ModelInfo } from '@/api/types';

interface ReportSection {
  title: string;
  html: string;
}

interface ReportData {
  model: ModelInfo;
  sections: ReportSection[];
  generatedAt: string;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatNum(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function buildModelInfoSection(model: ModelInfo): string {
  const rows = [
    ['Model Name', escapeHtml(model.model_name)],
    ['Model Type', escapeHtml(model.model_type)],
    ['Parameters', formatNum(model.total_params)],
    ['Size on Disk', formatBytes(model.total_size_bytes)],
    ['Layers', String(model.num_layers)],
    ['Hidden Size', String(model.hidden_size)],
    ['Attention Heads', String(model.num_attention_heads)],
    ['KV Heads', String(model.num_kv_heads)],
    ['Tensors', String(model.tensor_count)],
  ];
  if (model.quantization) {
    rows.push(['Quantization', `${model.quantization.bits}bit (group_size=${model.quantization.group_size})`]);
  }
  if (model.has_moe) {
    rows.push(['MOE', 'Yes']);
  }
  return `<table class="info-table">${rows.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join('')}</table>`;
}

function buildTraceSection(trace: Record<string, unknown>): string {
  const lines: string[] = [];
  const prompt = trace.prompt as string || '';
  const genText = trace.generated_text as string || '';
  const totalTime = trace.total_time_seconds as number;
  const steps = trace.steps as Array<Record<string, unknown>> || [];

  lines.push(`<p><strong>Prompt:</strong> ${escapeHtml(prompt)}</p>`);
  lines.push(`<p><strong>Generated Text:</strong></p><div class="code-block">${escapeHtml(genText)}</div>`);
  if (totalTime) lines.push(`<p><strong>Total Time:</strong> ${totalTime.toFixed(2)}s</p>`);
  lines.push(`<p><strong>Steps:</strong> ${steps.length} tokens generated</p>`);

  return lines.join('\n');
}

function buildQualitySection(quality: Record<string, unknown>): string {
  const lines: string[] = [];

  const ppl = quality.ppl as Record<string, unknown> | undefined;
  if (ppl) {
    lines.push('<h4>Perplexity</h4>');
    lines.push(`<p>PPL: <strong>${(ppl.perplexity as number)?.toFixed(2) ?? 'N/A'}</strong></p>`);
    lines.push(`<p>Tokens: ${ppl.num_tokens ?? 'N/A'}, Duration: ${(ppl.duration_seconds as number)?.toFixed(2) ?? '?'}s</p>`);
  }

  const report = quality.report as Record<string, unknown> | undefined;
  if (report) {
    lines.push('<h4>Quality Report</h4>');
    const avgPpl = report.avg_perplexity as number;
    if (avgPpl) lines.push(`<p>Average PPL: <strong>${avgPpl.toFixed(2)}</strong></p>`);
    const samples = report.generation_samples as Array<Record<string, unknown>> || [];
    if (samples.length > 0) {
      lines.push('<table class="data-table"><thead><tr><th>Prompt</th><th>Output</th><th>tok/s</th></tr></thead><tbody>');
      for (const s of samples.slice(0, 10)) {
        lines.push(`<tr><td>${escapeHtml(String(s.prompt || ''))}</td><td class="code-cell">${escapeHtml(String(s.generated_text || ''))}</td><td>${(s.tokens_per_second as number)?.toFixed(1) ?? ''}</td></tr>`);
      }
      lines.push('</tbody></table>');
    }
  }

  const gen = quality.generation as Array<Record<string, unknown>> | undefined;
  if (gen && gen.length > 0 && !report) {
    lines.push('<h4>Generation Samples</h4>');
    lines.push('<table class="data-table"><thead><tr><th>Prompt</th><th>Output</th><th>tok/s</th></tr></thead><tbody>');
    for (const s of gen.slice(0, 10)) {
      lines.push(`<tr><td>${escapeHtml(String(s.prompt || ''))}</td><td class="code-cell">${escapeHtml(String(s.generated_text || ''))}</td><td>${(s.tokens_per_second as number)?.toFixed(1) ?? ''}</td></tr>`);
    }
    lines.push('</tbody></table>');
  }

  return lines.length > 0 ? lines.join('\n') : '<p class="muted">No quality data available.</p>';
}

const REPORT_CSS = `
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; color: #1f2937; background: #fff; }
h1 { color: #4f46e5; border-bottom: 2px solid #e5e7eb; padding-bottom: 0.5rem; }
h2 { color: #374151; margin-top: 2rem; }
h3 { color: #6366f1; }
h4 { color: #4b5563; margin-top: 1rem; }
.meta { color: #9ca3af; font-size: 0.875rem; margin-bottom: 2rem; }
.info-table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
.info-table th { text-align: left; padding: 0.5rem 1rem; background: #f9fafb; border: 1px solid #e5e7eb; width: 200px; font-weight: 600; }
.info-table td { padding: 0.5rem 1rem; border: 1px solid #e5e7eb; }
.data-table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.875rem; }
.data-table th { background: #f3f4f6; padding: 0.5rem; border: 1px solid #e5e7eb; text-align: left; }
.data-table td { padding: 0.5rem; border: 1px solid #e5e7eb; }
.code-cell { font-family: 'SF Mono', Monaco, monospace; font-size: 0.8rem; max-width: 400px; word-break: break-word; }
.code-block { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.5rem; padding: 1rem; font-family: monospace; font-size: 0.875rem; white-space: pre-wrap; word-break: break-word; }
.muted { color: #9ca3af; font-style: italic; }
.section { margin-bottom: 2rem; page-break-inside: avoid; }
@media print { body { max-width: 100%; padding: 1rem; } h1 { font-size: 1.5rem; } }
`;

export function generateHTMLReport(data: ReportData): string {
  const sectionsHtml = data.sections
    .map((s) => `<div class="section"><h2>${escapeHtml(s.title)}</h2>${s.html}</div>`)
    .join('\n');

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Analysis Report — ${escapeHtml(data.model.model_name)}</title>
<style>${REPORT_CSS}</style>
</head>
<body>
<h1>Model Analysis Report</h1>
<p class="meta">Generated: ${data.generatedAt} | Model: ${escapeHtml(data.model.model_name)}</p>
<div class="section">
<h2>Model Information</h2>
${buildModelInfoSection(data.model)}
</div>
${sectionsHtml}
<hr>
<p class="meta" style="text-align:center">Generated by Edge Studio</p>
</body>
</html>`;
}

export function buildReportSections(
  model: ModelInfo,
  trace: Record<string, unknown> | null,
  quality: Record<string, unknown> | null,
): ReportSection[] {
  const sections: ReportSection[] = [];

  if (trace) {
    sections.push({ title: 'Inference Trace', html: buildTraceSection(trace) });
  }

  if (quality && Object.keys(quality).length > 0) {
    sections.push({ title: 'Quality Validation', html: buildQualitySection(quality) });
  }

  return sections;
}

export function downloadReport(html: string, filename: string) {
  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
