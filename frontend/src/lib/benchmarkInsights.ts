// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * benchmarkInsights — derived stats + chat helpers for the /benchmark-dashboard
 * page (history / batch-runner view, complement to /quality and /comparison).
 *
 * Consumes an array of BenchmarkResult records (each with TPS / TTFT / PPL /
 * memory / size for one model run), derives:
 *  - per-metric extremes + averages (best TPS, lowest TTFT, lowest PPL, etc.)
 *  - bucket classification (slow/ok/fast/blazing for TPS; healthy/concerning
 *    /broken for PPL) so the AI Brief can speak in human terms (§8.3 bucket)
 *  - per-run delta-vs-best (each run's TPS as % of session peak)
 *  - assess outliers (one model wildly slower than others = caution)
 *  - composes a session-aware system snippet so the loaded LLM (this Mac's
 *    brain) can speak as itself about the benchmark cohort
 *
 * Pure functions; no fetching. Uses BenchmarkItem + BenchmarkResult shapes
 * from the page's local types (mirrored here to avoid coupling).
 *
 * Sovereignty (§9.2 mandatory): benchmark telemetry never leaves this Mac —
 * no cloud reporting, no analytics. Identity card forces emerald.
 */
import type { ModelInfo } from '@/api/types';

type Locale = 'en' | 'zh';

/** Mirror of BenchmarkResult.result inside the dashboard page. */
export interface BenchmarkRecord {
  label: string;
  model_dir: string;
  success: boolean;
  result?: {
    disk_size_bytes: number;
    memory_peak_mb: number;
    tokens_per_second: number;
    time_to_first_token_ms: number;
    perplexity: number;
  };
  error?: string;
}

export type TPSBucket = 'unknown' | 'slow' | 'ok' | 'fast' | 'blazing';
export type TTFTBucket = 'unknown' | 'snappy' | 'ok' | 'sluggish' | 'painful';
export type PPLBucket = 'unknown' | 'excellent' | 'good' | 'concerning' | 'broken';

const TPS_OK = 10;
const TPS_FAST = 30;
const TPS_BLAZING = 60;

const TTFT_SNAPPY_MS = 200;
const TTFT_OK_MS = 800;
const TTFT_SLUGGISH_MS = 2000;

const PPL_EXCELLENT = 5;
const PPL_GOOD = 15;
const PPL_CONCERNING = 30;

export function tpsToBucket(tps: number): TPSBucket {
  if (!Number.isFinite(tps) || tps <= 0) return 'unknown';
  if (tps < TPS_OK) return 'slow';
  if (tps < TPS_FAST) return 'ok';
  if (tps < TPS_BLAZING) return 'fast';
  return 'blazing';
}

export function ttftToBucket(ttftMs: number): TTFTBucket {
  if (!Number.isFinite(ttftMs) || ttftMs <= 0) return 'unknown';
  if (ttftMs < TTFT_SNAPPY_MS) return 'snappy';
  if (ttftMs < TTFT_OK_MS) return 'ok';
  if (ttftMs < TTFT_SLUGGISH_MS) return 'sluggish';
  return 'painful';
}

export function pplToBucket(ppl: number): PPLBucket {
  if (!Number.isFinite(ppl) || ppl <= 0) return 'unknown';
  if (ppl < PPL_EXCELLENT) return 'excellent';
  if (ppl < PPL_GOOD) return 'good';
  if (ppl < PPL_CONCERNING) return 'concerning';
  return 'broken';
}

export interface BenchmarkSummary {
  totalRuns: number;
  successCount: number;
  errorCount: number;
  /** Distinct model labels (case-insensitive). */
  modelCount: number;
  /** Best (max) TPS observed across successful runs. 0 if no runs. */
  bestTPS: number;
  bestTPSLabel: string;
  /** Best (min) TTFT in ms. 0 if no runs. */
  bestTTFT: number;
  bestTTFTLabel: string;
  /** Best (min) perplexity. 0 if no runs. */
  bestPPL: number;
  bestPPLLabel: string;
  /** Mean TPS across successful runs. */
  meanTPS: number;
  /** Mean PPL across successful runs. */
  meanPPL: number;
  /** Total disk size (bytes) across all run models — sum, not unique. */
  totalDiskBytes: number;
  bestTPSBucket: TPSBucket;
  bestTTFTBucket: TTFTBucket;
  bestPPLBucket: PPLBucket;
  /** Outliers: labels of runs whose TPS is < 50% of bestTPS. */
  slowOutliers: string[];
  /** Labels with broken PPL bucket (>= 30). */
  brokenPPL: string[];
}

export function summarizeBenchmark(results: BenchmarkRecord[]): BenchmarkSummary {
  const successes = results.filter((r) => r.success && r.result);

  const tpsList = successes
    .map((r) => ({ label: r.label, v: r.result!.tokens_per_second }))
    .filter((x) => Number.isFinite(x.v) && x.v > 0);
  const ttftList = successes
    .map((r) => ({ label: r.label, v: r.result!.time_to_first_token_ms }))
    .filter((x) => Number.isFinite(x.v) && x.v > 0);
  const pplList = successes
    .map((r) => ({ label: r.label, v: r.result!.perplexity }))
    .filter((x) => Number.isFinite(x.v) && x.v > 0);

  const bestTps = tpsList.reduce((acc, c) => (c.v > acc.v ? c : acc), { label: '', v: 0 });
  const bestTtft = ttftList.reduce((acc, c) => (acc.v === 0 || c.v < acc.v ? c : acc), { label: '', v: 0 });
  const bestPpl = pplList.reduce((acc, c) => (acc.v === 0 || c.v < acc.v ? c : acc), { label: '', v: 0 });

  const meanTps = tpsList.length ? tpsList.reduce((s, c) => s + c.v, 0) / tpsList.length : 0;
  const meanPpl = pplList.length ? pplList.reduce((s, c) => s + c.v, 0) / pplList.length : 0;

  const totalDiskBytes = successes.reduce((s, r) => s + (r.result?.disk_size_bytes ?? 0), 0);

  const distinctLabels = new Set(results.map((r) => r.label.trim().toLowerCase()).filter(Boolean));

  // Outlier detection: a successful run whose TPS < 50% of the cohort peak.
  const peakTps = bestTps.v || 0;
  const slowOutliers = peakTps > 0
    ? successes
        .filter((r) => (r.result?.tokens_per_second ?? 0) > 0 && (r.result!.tokens_per_second / peakTps) < 0.5)
        .map((r) => r.label)
    : [];

  const brokenPPL = successes
    .filter((r) => pplToBucket(r.result?.perplexity ?? 0) === 'broken')
    .map((r) => r.label);

  return {
    totalRuns: results.length,
    successCount: successes.length,
    errorCount: results.length - successes.length,
    modelCount: distinctLabels.size,
    bestTPS: bestTps.v,
    bestTPSLabel: bestTps.label,
    bestTTFT: bestTtft.v,
    bestTTFTLabel: bestTtft.label,
    bestPPL: bestPpl.v,
    bestPPLLabel: bestPpl.label,
    meanTPS: meanTps,
    meanPPL: meanPpl,
    totalDiskBytes,
    bestTPSBucket: tpsToBucket(bestTps.v),
    bestTTFTBucket: ttftToBucket(bestTtft.v),
    bestPPLBucket: pplToBucket(bestPpl.v),
    slowOutliers,
    brokenPPL,
  };
}

export type BenchmarkRiskLevel = 'safe' | 'caution' | 'danger';
export interface BenchmarkRisk {
  level: BenchmarkRiskLevel;
  reason: string;
  reasonZh: string;
}

interface BenchmarkNarrationOptions {
  fixture?: boolean;
}

/**
 * Per-(cohort) risk assessment.
 *  - danger:  any run shows broken PPL (>= 30) — model fundamentally broken
 *  - caution: error count > 0 — one or more models failed to load/benchmark
 *  - caution: only 1 model in cohort — no comparison possible
 *  - caution: slow outliers — one model is < 50% the peak speed
 *  - caution: no runs yet (empty state)
 *  - safe:    >= 2 models, all successful, no broken PPL, no slow outliers
 */
export function assessBenchmarkCohort(
  summary: BenchmarkSummary,
): BenchmarkRisk {
  if (summary.totalRuns === 0) {
    return {
      level: 'caution',
      reason: 'No benchmark runs yet — add local model paths, then run the batch.',
      reasonZh: '还没有 benchmark 数据 — 先添加本地模型路径, 再运行批量测试.',
    };
  }
  if (summary.brokenPPL.length > 0) {
    return {
      level: 'danger',
      reason: `${summary.brokenPPL.length} model(s) show broken perplexity (>= ${PPL_CONCERNING}): ${summary.brokenPPL.slice(0, 2).join(', ')}${summary.brokenPPL.length > 2 ? '…' : ''}. Likely over-quantized or weights corrupted.`,
      reasonZh: `${summary.brokenPPL.length} 个模型 PPL 失控 (>= ${PPL_CONCERNING}): ${summary.brokenPPL.slice(0, 2).join(', ')}${summary.brokenPPL.length > 2 ? '…' : ''}. 多半是过量化或权重坏了.`,
    };
  }
  if (summary.errorCount > 0) {
    return {
      level: 'caution',
      reason: `${summary.errorCount} model(s) failed to benchmark — check the error column for details.`,
      reasonZh: `${summary.errorCount} 个模型 benchmark 失败 — 看 error 列了解原因.`,
    };
  }
  if (summary.modelCount === 1 && summary.totalRuns >= 1) {
    return {
      level: 'caution',
      reason: `Only 1 model in this cohort — add a baseline (or a peer) to enable comparison.`,
      reasonZh: `当前 cohort 只有 1 个模型 — 加一个 baseline (或对照) 才能对比.`,
    };
  }
  if (summary.slowOutliers.length > 0) {
    return {
      level: 'caution',
      reason: `Slow outliers (< 50% peak TPS): ${summary.slowOutliers.slice(0, 2).join(', ')}${summary.slowOutliers.length > 2 ? '…' : ''}. Quantization mismatch, KV-cache size, or thermal throttling likely.`,
      reasonZh: `慢速异常值 (< 50% 峰值 TPS): ${summary.slowOutliers.slice(0, 2).join(', ')}${summary.slowOutliers.length > 2 ? '…' : ''}. 量化不一致 / KV cache 太大 / 热降频 都可能.`,
    };
  }
  return {
    level: 'safe',
    reason: `${summary.successCount} runs · peak ${summary.bestTPS.toFixed(1)} tok/s · best PPL ${summary.bestPPL.toFixed(2)} · all healthy.`,
    reasonZh: `${summary.successCount} 个 run · 峰值 ${summary.bestTPS.toFixed(1)} tok/s · 最佳 PPL ${summary.bestPPL.toFixed(2)} · 全部健康.`,
  };
}

export function bucketLabel(b: TPSBucket | TTFTBucket | PPLBucket, locale: Locale): string {
  if (locale === 'zh') {
    return ({
      unknown: '未知',
      slow: '慢', ok: '一般', fast: '快', blazing: '极快',
      snappy: '迅捷', sluggish: '迟钝', painful: '痛苦',
      excellent: '优秀', good: '良好', concerning: '欠佳', broken: '损坏',
    } as const)[b];
  }
  return b;
}

export function buildBenchmarkContextSnippet(
  results: BenchmarkRecord[],
  summary: BenchmarkSummary,
  brain: ModelInfo | null,
  locale: Locale,
  options: BenchmarkNarrationOptions = {},
): string {
  const fixture = options.fixture === true;
  const lines: string[] = [
    locale === 'zh'
      ? fixture ? `## 你所在的 Benchmark 样例 Cohort` : `## 你所在的 Benchmark Cohort`
      : fixture ? `## YOUR SAMPLE BENCHMARK COHORT` : `## YOUR BENCHMARK COHORT`,
    locale === 'zh'
      ? `你 (${brain?.model_name ?? '本机加载的 LLM'}) 是 brain. 此页正在${fixture ? '展示虚构 onboarding 样例' : '浏览历史 benchmark'}, ${summary.totalRuns} 个 run, ${summary.successCount} 成功 / ${summary.errorCount} 失败.`
      : `You (${brain?.model_name ?? 'the loaded LLM'}) are the brain. This page is ${fixture ? 'showing fictional onboarding sample data' : 'browsing benchmark history'} with ${summary.totalRuns} runs (${summary.successCount} ok, ${summary.errorCount} failed).`,
    fixture
      ? locale === 'zh'
        ? `- Fixture boundary: 这些数字是 EdgeStudio 内置练手数据, 只用于教用户读 TPS / TTFT / PPL, 不是本机实测证据, 不能用于发布或宣称性能.`
        : `- Fixture boundary: these numbers are built-in EdgeStudio onboarding samples only. They teach TPS / TTFT / PPL reading and are not real local benchmark evidence or publishable performance claims.`
      : '',
    `- Distinct models: ${summary.modelCount}`,
    `- Best TPS: ${summary.bestTPS.toFixed(2)} tok/s (${summary.bestTPSLabel || '—'}, bucket: ${summary.bestTPSBucket})`,
    `- Best TTFT: ${summary.bestTTFT.toFixed(0)} ms (${summary.bestTTFTLabel || '—'}, bucket: ${summary.bestTTFTBucket})`,
    `- Best PPL: ${summary.bestPPL.toFixed(2)} (${summary.bestPPLLabel || '—'}, bucket: ${summary.bestPPLBucket})`,
    `- Mean TPS: ${summary.meanTPS.toFixed(2)} tok/s · Mean PPL: ${summary.meanPPL.toFixed(2)}`,
    summary.slowOutliers.length > 0
      ? `- Slow outliers (< 50% peak): ${summary.slowOutliers.join(', ')}`
      : `- Slow outliers: none`,
    summary.brokenPPL.length > 0
      ? `- Broken-PPL models: ${summary.brokenPPL.join(', ')}`
      : ``,
    ``,
    locale === 'zh'
      ? `### 当前 cohort 详情:`
      : `### Cohort detail:`,
  ];
  for (const r of results.slice(0, 10)) {
    if (r.success && r.result) {
      lines.push(`- ${r.label}: ${r.result.tokens_per_second.toFixed(1)} tok/s · ${r.result.time_to_first_token_ms.toFixed(0)} ms TTFT · PPL ${r.result.perplexity.toFixed(2)}`);
    } else {
      lines.push(`- ${r.label}: FAILED — ${r.error ?? '(unknown error)'}`);
    }
  }
  if (results.length > 10) lines.push(`- … and ${results.length - 10} more`);
  lines.push(
    ``,
    fixture
      ? locale === 'zh'
        ? `### Fixture 边界: 这批样例没有云上报, 但也没有真实测量. 回答时必须提醒用户点击真实 Run Benchmark 后才会得到本机证据.`
        : `### Fixture boundary: this sample did not report to cloud, but it was not measured either. Remind users they need a real Run Benchmark action for local evidence.`
      : locale === 'zh'
        ? `### 北极星 §1 主权: 这些数字全部本地测出, 0 次云上报 (no telemetry). 当用户问 "为什么不发到云分析" 时, 强调这一点.`
        : `### North-star §1 sovereignty: all numbers measured locally, zero cloud reporting. When user asks "why not report to cloud", emphasise this.`,
  );
  return lines.filter(Boolean).join('\n');
}

export function buildBenchmarkAutoBrief(
  summary: BenchmarkSummary,
  locale: Locale,
  options: BenchmarkNarrationOptions = {},
): string {
  if (options.fixture && summary.totalRuns > 0) {
    if (locale === 'zh') {
      return `当前 cohort 是 EdgeStudio 内置的虚构练手 benchmark 样例, 不是本机实测证据. 用 2-3 句话作为 brain 教用户读这组数字: TPS 越高越快, TTFT 越低越快出首 token, PPL 越低质量越稳; 最后提醒点击真实 Run Benchmark 才能得到可发布的本机证据. 第一人称, 不宣称性能提升.`;
    }
    return `The current cohort is EdgeStudio's fictional onboarding benchmark sample, not real local evidence. In 2-3 sentences as the brain, teach the user how to read it: higher TPS is faster generation, lower TTFT is faster first token, lower PPL is steadier quality; end by reminding them to run a real benchmark for publishable local evidence. First person, no performance improvement claims.`;
  }
  if (locale === 'zh') {
    if (summary.totalRuns === 0) {
      return `用户还没跑任何 benchmark. 用 2-3 句话作为本机 brain 介绍这页能干什么 (批量测多个模型, 看 TPS/TTFT/PPL 对比), 邀请用户加 1-2 个模型. 第一人称, 不列项.`;
    }
    if (summary.brokenPPL.length > 0) {
      return `${summary.brokenPPL.length} 个模型 PPL 失控 (>= ${PPL_CONCERNING}). 用 2-3 句话作为 brain 直接告诉用户哪个坏了 + 最可能的原因 (过量化 / 权重坏 / chat template 不对). 第一人称, 引用具体数字.`;
    }
    if (summary.slowOutliers.length > 0) {
      return `cohort 里有慢速异常 (${summary.slowOutliers.join(', ')} < 50% 峰值 ${summary.bestTPS.toFixed(1)} tok/s). 用 2-3 句话作为 brain 直接给个诊断: 这种 split 一般是什么原因 (KV cache 巨 / 量化不一致 / 不同精度). 第一人称.`;
    }
    return `cohort 健康 — ${summary.successCount} run, 峰值 ${summary.bestTPS.toFixed(1)} tok/s (${summary.bestTPSLabel}), 最佳 PPL ${summary.bestPPL.toFixed(2)}. 用 2-3 句话作为 brain 总结这批 run 的关键洞察 (谁最值得保留 / 哪个性价比最高), 第一人称, 引用真实数字.`;
  }
  if (summary.totalRuns === 0) {
    return `User has not run any benchmarks yet. In 2-3 sentences, as the brain LLM on this Mac, introduce what this page does (batch-bench multiple models, compare TPS/TTFT/PPL), and invite them to add 1-2 models. First person, no bullets.`;
  }
  if (summary.brokenPPL.length > 0) {
    return `${summary.brokenPPL.length} model(s) show broken PPL (>= ${PPL_CONCERNING}). In 2-3 sentences, as brain, tell the user which is broken + the most likely cause (over-quantization / corrupted weights / wrong chat template). First person, cite numbers.`;
  }
  if (summary.slowOutliers.length > 0) {
    return `Cohort has slow outliers (${summary.slowOutliers.join(', ')} below 50% of peak ${summary.bestTPS.toFixed(1)} tok/s). In 2-3 sentences, as brain, give a quick diagnosis (huge KV cache / quant mismatch / different precision). First person.`;
  }
  return `Cohort is healthy — ${summary.successCount} runs, peak ${summary.bestTPS.toFixed(1)} tok/s (${summary.bestTPSLabel}), best PPL ${summary.bestPPL.toFixed(2)}. In 2-3 sentences, as brain, summarise the key insight from this cohort (which to keep / best value-for-size). First person, cite real numbers.`;
}

export function getBenchmarkSuggestedPrompts(
  summary: BenchmarkSummary,
  brain: ModelInfo | null,
  locale: Locale,
  options: BenchmarkNarrationOptions = {},
): { label: string; prompt: string }[] {
  const brainName = brain?.model_name || (locale === 'zh' ? '我' : 'me');

  if (options.fixture && summary.totalRuns > 0) {
    if (locale === 'zh') {
      return [
        { label: '读样例', prompt: `这是虚构 benchmark 样例. 作为 ${brainName}, 用第一人称解释这 3 行里 TPS / TTFT / PPL / size 分别怎么看, 哪个样例最适合移动端, 并提醒这不是实测证据.` },
        { label: '换成真测', prompt: `告诉用户如何把这份样例换成真实 benchmark: 应该添加 baseline 和 optimized 两个本地路径, 点击 Run Benchmark, 跑完后先看哪三个数字. 简短具体.` },
        { label: '速度质量取舍', prompt: `用这份样例讲速度和质量的取舍: 为什么 Q4 通常更快更小, 但 PPL 可能上升; 怎样判断上升是否可接受. 明确说这是练手数据.` },
        { label: 'CSV 能不能用', prompt: `解释样例 CSV 只能用于熟悉格式, 不能写进报告或 leaderboard; 真实结果才可以作为本机证据. 语气直接.` },
      ];
    }
    return [
      { label: 'Read sample', prompt: `This is a fictional benchmark sample. As ${brainName}, explain how to read TPS / TTFT / PPL / size across these 3 rows, which sample looks best for mobile, and remind me this is not measured evidence.` },
      { label: 'Switch to real run', prompt: `Tell the user exactly how to replace this sample with a real benchmark: add baseline and optimized local paths, click Run Benchmark, then inspect which three numbers first. Keep it concrete.` },
      { label: 'Speed vs quality', prompt: `Use this sample to explain speed/quality trade-offs: why Q4 is often faster and smaller but PPL can rise; how to judge whether the rise is acceptable. Say clearly this is practice data.` },
      { label: 'Can I use CSV?', prompt: `Explain that sample CSV is only for learning the format and cannot go into a report or leaderboard; real runs become local evidence. Be direct.` },
    ];
  }

  if (locale === 'zh') {
    if (summary.totalRuns === 0) {
      return [
        { label: '🎯 怎么开始', prompt: `用户还没跑 benchmark. 给一个具体的入门路径: 第一次该测什么 (单模型还是 baseline + optimized 对比) / 测多少 token 合适 / 看哪几个数字最关键. 第一人称.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句话作为 ${brainName} 解释: benchmark 数据为什么不上报云 (隐私 / 设备身份 / 公司机密 model 名字), 比"匿名遥测"还要彻底.` },
        { label: '📊 怎么解读 PPL', prompt: `给用户一个 perplexity 速查表: <5 / 5-15 / 15-30 / >30 各代表什么质量, 多大的 PPL 上升算"灾难性"还能接受.` },
        { label: '⚡ 速度 vs 质量', prompt: `用一段话讲 TPS / TTFT / PPL 三个数字之间的 trade-off — 为什么不能光看 TPS, 何时应该牺牲速度换 PPL.` },
      ];
    }
    if (summary.brokenPPL.length > 0) {
      return [
        { label: '🚨 哪个坏了', prompt: `cohort 里 ${summary.brokenPPL.join(', ')} PPL >= ${PPL_CONCERNING}. 作为 brain 直接告诉用户每个坏在哪 + 最可能的 root cause + 怎么 verify (重测? 换 chat template? 换 quant?).` },
        { label: '🔍 诊断流程', prompt: `给用户一个具体的 PPL 失控诊断 checklist (5-7 步): chat template 对吗? quant bits 多少? 是否有 retry overhead? 是否 prompt 截断了? 我作为 brain 自己也帮你判断.` },
        { label: '💡 怎么修', prompt: `${summary.brokenPPL[0]} 这个最坏的, 假设是过量化导致, 给一个具体的修复方案 (回 8-bit / 重训 / 换更小 base). 第一人称, 我会陪你一起调.` },
        { label: '📦 哪个能保留', prompt: `${summary.successCount} 个 run 里有 ${summary.brokenPPL.length} 个 broken. 剩下的 ${summary.successCount - summary.brokenPPL.length} 个里, 哪个最值得保留 (size / TPS / PPL 综合), 哪个该淘汰?` },
      ];
    }
    return [
      { label: '🏆 最值得保留的', prompt: `${summary.successCount} 个 run 里, 综合 TPS / TTFT / PPL / size 看, 哪个性价比最高? 给一个具体的 ranking (top 3) + 理由. 作为 ${brainName} 直接说.` },
      { label: '🐌 慢速诊断', prompt: summary.slowOutliers.length > 0
        ? `${summary.slowOutliers.join(', ')} 比 cohort peak (${summary.bestTPS.toFixed(1)} tok/s) 慢一半以上. 作为 brain 给个具体诊断 (KV cache 大? quant 不一致? thermal?), 给最可能的根因.`
        : `cohort TPS 分布 ${summary.meanTPS.toFixed(1)} 平均 / ${summary.bestTPS.toFixed(1)} 峰值. 没有 outlier 但还能继续优化吗? 哪个最有 headroom?` },
      { label: '🌍 端侧主权', prompt: `${summary.totalRuns} 个 benchmark 数字全部本地测出, 0 次云上报. 用 2-3 句话作为 brain 解释: 这与 "提交到 HuggingFace leaderboard" / "上传到 W&B" 各有什么 trade-off, 隐私 vs 共享.` },
      { label: '⏱️ TTFT 分析', prompt: `cohort 最佳 TTFT ${summary.bestTTFT.toFixed(0)} ms (${summary.bestTTFTLabel}, bucket: ${summary.bestTTFTBucket}). 作为 brain 解释: TTFT 主要受什么影响 (prompt 长度 / KV cache prefill / RAM 带宽), 用户能怎么再降.` },
    ];
  }
  // English
  if (summary.totalRuns === 0) {
    return [
      { label: '🎯 How to start', prompt: `User hasn't benchmarked yet. Give a concrete onboarding path: first benchmark single model or baseline+optimized? How many tokens to test? Which numbers matter most? First person.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences as ${brainName}, explain why benchmark data doesn't ping any cloud (privacy / device fingerprint / company-confidential model names), stricter than "anonymous telemetry".` },
      { label: '📊 PPL primer', prompt: `Give a perplexity cheat-sheet: <5 / 5-15 / 15-30 / >30 each meaning what quality level, what PPL bump counts as "catastrophic" vs "tolerable".` },
      { label: '⚡ Speed vs quality', prompt: `In one paragraph, walk through the TPS / TTFT / PPL trade-off — why TPS alone is misleading and when to sacrifice speed for PPL.` },
    ];
  }
  if (summary.brokenPPL.length > 0) {
    return [
      { label: '🚨 Which is broken', prompt: `Cohort has ${summary.brokenPPL.join(', ')} with PPL >= ${PPL_CONCERNING}. As brain, tell the user exactly where each broke + most likely root cause + how to verify (retest? chat template? quant?).` },
      { label: '🔍 Diagnostic flow', prompt: `Give a concrete 5-7 step PPL-broken diagnostic checklist (chat template right? quant bits? retry overhead? prompt truncated?). I'll help judge each.` },
      { label: '💡 How to fix', prompt: `Assume ${summary.brokenPPL[0]} (the worst) failed from over-quant. Give a concrete fix plan (back to 8-bit / retrain / smaller base). First person, I'll walk it with you.` },
      { label: '📦 Which to keep', prompt: `Of ${summary.successCount} runs, ${summary.brokenPPL.length} broken. Of the remaining ${summary.successCount - summary.brokenPPL.length}, which deserves keeping (size / TPS / PPL combined), which to drop?` },
    ];
  }
  return [
    { label: '🏆 Best value', prompt: `Across ${summary.successCount} runs, weighted by TPS / TTFT / PPL / size, which has the best value? Give a concrete top-3 ranking + reasons. Direct as ${brainName}.` },
    { label: '🐌 Slow diagnosis', prompt: summary.slowOutliers.length > 0
      ? `${summary.slowOutliers.join(', ')} are below 50% of peak (${summary.bestTPS.toFixed(1)} tok/s). As brain, give a specific diagnosis (huge KV cache? quant mismatch? thermal?), most-likely root cause.`
      : `Cohort distribution: mean ${summary.meanTPS.toFixed(1)} tok/s, peak ${summary.bestTPS.toFixed(1)} tok/s. No outliers — but can it be optimised further? Which model has the most headroom?` },
    { label: '🌍 Edge sovereignty', prompt: `${summary.totalRuns} numbers all measured locally, zero cloud reporting. In 2-3 sentences as brain, explain the trade-off vs "submit to HuggingFace leaderboard" / "upload to W&B" — privacy vs sharing.` },
    { label: '⏱️ TTFT analysis', prompt: `Cohort best TTFT ${summary.bestTTFT.toFixed(0)} ms (${summary.bestTTFTLabel}, bucket: ${summary.bestTTFTBucket}). As brain, explain what dominates TTFT (prompt length / prefill / RAM bandwidth) and how the user could push it lower.` },
  ];
}
