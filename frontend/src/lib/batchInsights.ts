// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * batchInsights — derived N-item batch optimization state + chat helpers
 * for the /batch page (BatchOperations.tsx).
 *
 * Batch differs from /merge: each item is independent (own pipeline run,
 * own output), not fused. The page builds an N-item queue (each: model_dir
 * + label + bits + operation), then runs them serially via
 * `/api/model/load` + `/api/model/{id}/pipeline/run` + task polling.
 *
 * §9.1 multi-component pattern continues here as a "queue + cohort":
 *  - per-item config validity (path + op + bits sane?)
 *  - operation histogram (quant / neuron-prune / vocab-prune mix?)
 *  - duplicate path detection (same model added twice → wasted work)
 *  - bits=2 / bits=3 quant flagged as danger (mlx_lm will accept the call
 *    but PPL collapses; agent confirmed no backend guardrail at
 *    native_ops.py:465-530, so we surface from the UI)
 *
 * Result cohort (after each item finishes, frontend pushes BatchResult):
 *  - success / failed counts; partial-failure handling
 *  - total saved bytes / avg savings %
 *  - slowest + fastest item (run-time outliers)
 *  - best / worst savings (compression outliers)
 *  - per-item savings bucket (trim / modest / strong / extreme)
 *  - per-item duration bucket
 *
 * Pure functions; no fetching. The page passes already-fetched inputs in.
 *
 * §10.3 naming: capability uses `runPhase` not `status` to avoid contract-test
 *   grep collision (`status === 'X'` triggers task_status string check).
 *
 * Sovereignty (§9.2 mandatory): every operation here is a pure local weight
 * transform; the queue runner posts to localhost only. Surfaced as a
 * Sovereignty card in the IdentityCard strip.
 */
import type { ModelInfo } from '@/api/types';

type Locale = 'en' | 'zh';

export type BatchOperation = 'quantization' | 'neuron_pruning' | 'vocab_pruning';

export interface BatchItem {
  model_dir: string;
  label: string;
  bits: number;
  operation: BatchOperation | string;
}

export interface BatchResult {
  label: string;
  success: boolean;
  output_dir?: string;
  original_size?: number;
  result_size?: number;
  duration_seconds?: number;
  error?: string;
}

/** Cohort run lifecycle. Avoid `status` per §10.3. */
export type BatchRunPhase = 'idle' | 'running' | 'partial' | 'complete' | 'allFailed';

/** Per-item savings classification. */
export type SavingsBucket = 'none' | 'trim' | 'modest' | 'strong' | 'extreme';

/** Per-item duration classification. */
export type DurationBucket = 'fast' | 'ok' | 'slow' | 'painful';

export interface BatchCapabilities {
  // ── Inputs ─────────────────────────────────────────────────────────────
  items: BatchItem[];
  results: BatchResult[];
  running: boolean;
  brain: ModelInfo | null;

  // ── Queue derived ──────────────────────────────────────────────────────
  itemCount: number;
  hasItems: boolean;
  /** Distinct operations present in the queue. */
  opMix: BatchOperation[];
  /** Counts by operation. */
  opHistogram: Record<string, number>;
  /** Min / max bits in any quantization items (0 if none). */
  bitsRange: { min: number; max: number; values: number[] };
  /** Path-tail of each item, deduplicated. */
  uniquePathCount: number;
  /** Items whose path appears more than once in the queue. */
  duplicateLabels: string[];
  /** Items with quantization bits ≤ 3 (PPL collapse risk). */
  dangerousQuant: BatchItem[];
  /** Items with vocab+neuron mixed in same queue (caution). */
  mixedOpsPresent: boolean;

  // ── Cohort derived (after results stream in) ──────────────────────────
  runPhase: BatchRunPhase;
  successCount: number;
  failedCount: number;
  successRate: number; // 0..1, of completed items
  completedCount: number; // success + failed
  pendingCount: number; // items not yet attempted
  totalOriginalBytes: number;
  totalResultBytes: number;
  totalSavedBytes: number;
  avgSavingsPct: number; // 0..100
  avgDurationSec: number;
  slowestItem: BatchResult | null;
  fastestItem: BatchResult | null;
  bestSavingsItem: BatchResult | null; // largest savings %
  worstSavingsItem: BatchResult | null; // smallest savings % (still success)
  /** Per-item savings bucket aligned with results array. */
  savingsBuckets: SavingsBucket[];
  /** Per-item duration bucket. */
  durationBuckets: DurationBucket[];
}

function tail(p: string): string {
  if (!p) return '';
  return p.replace(/\/$/, '').split('/').pop() || p;
}

/** Savings % → bucket. Aligned with playbook §10.6 bucket convention. */
export function savingsToBucket(pct: number | null | undefined): SavingsBucket {
  if (pct == null || !Number.isFinite(pct) || pct <= 0) return 'none';
  if (pct < 20) return 'trim';
  if (pct < 50) return 'modest';
  if (pct < 75) return 'strong';
  return 'extreme';
}

/** Duration seconds → bucket. */
export function durationToBucket(sec: number | null | undefined): DurationBucket {
  if (sec == null || !Number.isFinite(sec)) return 'ok';
  if (sec < 60) return 'fast';
  if (sec < 300) return 'ok';
  if (sec < 900) return 'slow';
  return 'painful';
}

export function deriveBatchCapabilities(
  items: BatchItem[],
  results: BatchResult[],
  running: boolean,
  brain: ModelInfo | null,
): BatchCapabilities {
  const itemCount = items.length;

  const opHistogram: Record<string, number> = {};
  const bitsValues: number[] = [];
  const dangerousQuant: BatchItem[] = [];
  for (const it of items) {
    opHistogram[it.operation] = (opHistogram[it.operation] || 0) + 1;
    if (it.operation === 'quantization') {
      bitsValues.push(it.bits);
      if (it.bits <= 3) dangerousQuant.push(it);
    }
  }
  const opMix = Object.keys(opHistogram) as BatchOperation[];

  const pathCounts = new Map<string, number>();
  for (const it of items) {
    pathCounts.set(it.model_dir, (pathCounts.get(it.model_dir) || 0) + 1);
  }
  const uniquePathCount = pathCounts.size;
  const duplicateLabels = items
    .filter((it) => (pathCounts.get(it.model_dir) || 0) > 1)
    .map((it) => it.label || tail(it.model_dir));

  const successCount = results.filter((r) => r.success).length;
  const failedCount = results.filter((r) => !r.success).length;
  const completedCount = successCount + failedCount;
  const pendingCount = Math.max(0, itemCount - completedCount);
  const successRate = completedCount > 0 ? successCount / completedCount : 0;

  let runPhase: BatchRunPhase = 'idle';
  if (running) runPhase = 'running';
  else if (completedCount === 0) runPhase = 'idle';
  else if (failedCount === 0 && completedCount === itemCount) runPhase = 'complete';
  else if (successCount === 0 && completedCount > 0) runPhase = 'allFailed';
  else runPhase = 'partial';

  let totalOriginalBytes = 0;
  let totalResultBytes = 0;
  let durationSum = 0;
  let durationN = 0;
  let savingsSum = 0;
  let savingsN = 0;
  let slowestItem: BatchResult | null = null;
  let fastestItem: BatchResult | null = null;
  let bestSavingsItem: BatchResult | null = null;
  let worstSavingsItem: BatchResult | null = null;
  let bestSavingsPct = -Infinity;
  let worstSavingsPct = Infinity;

  for (const r of results) {
    if (r.success) {
      if (r.original_size != null && r.result_size != null && r.original_size > 0) {
        totalOriginalBytes += r.original_size;
        totalResultBytes += r.result_size;
        const pct = ((r.original_size - r.result_size) / r.original_size) * 100;
        savingsSum += pct;
        savingsN += 1;
        if (pct > bestSavingsPct) {
          bestSavingsPct = pct;
          bestSavingsItem = r;
        }
        if (pct < worstSavingsPct) {
          worstSavingsPct = pct;
          worstSavingsItem = r;
        }
      }
      if (r.duration_seconds != null && r.duration_seconds > 0) {
        durationSum += r.duration_seconds;
        durationN += 1;
        if (slowestItem == null || (r.duration_seconds > (slowestItem.duration_seconds ?? 0))) {
          slowestItem = r;
        }
        if (fastestItem == null || (r.duration_seconds < (fastestItem.duration_seconds ?? Infinity))) {
          fastestItem = r;
        }
      }
    }
  }

  const avgDurationSec = durationN > 0 ? durationSum / durationN : 0;
  const avgSavingsPct = savingsN > 0 ? savingsSum / savingsN : 0;
  const totalSavedBytes = totalOriginalBytes - totalResultBytes;

  const savingsBuckets: SavingsBucket[] = results.map((r) => {
    if (!r.success || r.original_size == null || r.result_size == null || r.original_size === 0) {
      return 'none';
    }
    const pct = ((r.original_size - r.result_size) / r.original_size) * 100;
    return savingsToBucket(pct);
  });
  const durationBuckets: DurationBucket[] = results.map((r) => durationToBucket(r.duration_seconds));

  return {
    items,
    results,
    running,
    brain,
    itemCount,
    hasItems: itemCount > 0,
    opMix,
    opHistogram,
    bitsRange: bitsValues.length === 0
      ? { min: 0, max: 0, values: [] }
      : { min: Math.min(...bitsValues), max: Math.max(...bitsValues), values: bitsValues },
    uniquePathCount,
    duplicateLabels,
    dangerousQuant,
    mixedOpsPresent: opMix.length > 1,
    runPhase,
    successCount,
    failedCount,
    successRate,
    completedCount,
    pendingCount,
    totalOriginalBytes,
    totalResultBytes,
    totalSavedBytes,
    avgSavingsPct,
    avgDurationSec,
    slowestItem,
    fastestItem,
    bestSavingsItem,
    worstSavingsItem,
    savingsBuckets,
    durationBuckets,
  };
}

export type BatchRiskLevel = 'safe' | 'caution' | 'danger';
export interface BatchRisk {
  level: BatchRiskLevel;
  reason: string;
  reasonZh: string;
}

interface BatchNarrationOptions {
  fixture?: boolean;
}

/**
 * Risk hierarchy (most dangerous first):
 *  - danger: any item is quantization with bits ≤ 2 (PPL collapse, no backend guardrail)
 *  - danger: any item is quantization with bits == 3 AND queue is huge (>3) — fan-out of bad quant
 *  - caution: bits == 3 single — borderline, usable for some bf16 → 3bit experiments
 *  - caution: duplicate model paths in queue (wasted serial work)
 *  - caution: > 5 items (long wall-clock run, no parallelism)
 *  - caution: mixed operations (user might want to split into separate batches)
 *  - caution: vocab_pruning before quantization in same queue (vocab prune should
 *    typically run on bf16 base, not after quantization)
 *  - safe: clean queue
 */
export function assessBatchConfig(caps: BatchCapabilities): BatchRisk {
  if (caps.itemCount === 0) {
    return {
      level: 'safe',
      reason: 'Empty queue — add models to begin.',
      reasonZh: '队列为空 — 添加模型开始.',
    };
  }
  const bits2 = caps.dangerousQuant.filter((it) => it.bits <= 2);
  if (bits2.length > 0) {
    const labels = bits2.map((it) => it.label).join(', ');
    return {
      level: 'danger',
      reason: `${bits2.length} item(s) request bits=${bits2[0].bits} quantization (${labels}). Below 4-bit, PPL typically collapses (10× degradation). Backend has no guardrail. Bump to 4-bit unless you've validated the architecture.`,
      reasonZh: `${bits2.length} 个 item 配了 bits=${bits2[0].bits} 量化 (${labels}). 低于 4-bit, PPL 通常会崩 (10× 退化). 后端不做拦截. 除非你已实测过, 否则改 4-bit.`,
    };
  }
  const bits3 = caps.dangerousQuant.filter((it) => it.bits === 3);
  if (bits3.length > 0 && caps.itemCount > 3) {
    return {
      level: 'danger',
      reason: `${bits3.length} item(s) at 3-bit in a ${caps.itemCount}-item queue. 3-bit is borderline; fanning out across many models multiplies the failure surface. Validate one model at 3-bit first.`,
      reasonZh: `${bits3.length} 个 item 用 3-bit, 队列 ${caps.itemCount} 个. 3-bit 边缘可用; 多模型批量扩大了失败面. 先单模型 3-bit 验证再批量.`,
    };
  }
  if (bits3.length > 0) {
    return {
      level: 'caution',
      reason: `${bits3.length} item(s) at 3-bit. Borderline — usable for bf16 base experiments, but verify PPL stays in range.`,
      reasonZh: `${bits3.length} 个 item 用 3-bit. 边缘可用 — 适合 bf16 base 试验, 但要 verify PPL 在合理范围.`,
    };
  }
  if (caps.duplicateLabels.length > 0) {
    return {
      level: 'caution',
      reason: `Duplicate model paths in queue: ${caps.duplicateLabels.slice(0, 3).join(', ')}${caps.duplicateLabels.length > 3 ? '…' : ''}. Each duplicate runs the full pipeline serially — wasted time unless you're A/B-ing different params.`,
      reasonZh: `队列里有重复路径: ${caps.duplicateLabels.slice(0, 3).join(', ')}${caps.duplicateLabels.length > 3 ? '…' : ''}. 每个重复都会跑一遍完整流程 — 除非你在 A/B 测不同参数, 否则浪费时间.`,
    };
  }
  // Detect: vocab_pruning sits ahead of quantization for the SAME model — only
  // surfaces when both ops appear and ≥1 path appears with both ops in queue.
  const pathOps = new Map<string, Set<string>>();
  for (const it of caps.items) {
    if (!pathOps.has(it.model_dir)) pathOps.set(it.model_dir, new Set());
    pathOps.get(it.model_dir)!.add(it.operation);
  }
  let vocabAfterQuant = false;
  for (const ops of pathOps.values()) {
    if (ops.has('vocab_pruning') && ops.has('quantization')) {
      vocabAfterQuant = true;
      break;
    }
  }
  if (vocabAfterQuant) {
    return {
      level: 'caution',
      reason: 'Same model has both vocab_pruning and quantization in this batch. Vocab pruning typically runs on bf16 base, before quantization — confirm intended order.',
      reasonZh: '同一模型在批次中既有 vocab_pruning 又有 quantization. 词表裁剪一般跑在 bf16 base 上, 在量化之前 — 确认顺序.',
    };
  }
  if (caps.itemCount > 5) {
    return {
      level: 'caution',
      reason: `${caps.itemCount} items will run serially (no parallelism). Expect 5-30 min per item — total wall-clock could exceed 1 hour. Consider splitting into smaller batches if you need to babysit.`,
      reasonZh: `${caps.itemCount} 个 item 串行跑 (没有并行). 单个 5-30 分钟 — 总耗时可能超过 1 小时. 如果需要看护, 拆成更小批次.`,
    };
  }
  if (caps.mixedOpsPresent) {
    return {
      level: 'caution',
      reason: `Queue mixes ${caps.opMix.length} operations (${caps.opMix.join(', ')}). That's fine, but per-op tuning advice differs — consider grouping.`,
      reasonZh: `队列混合了 ${caps.opMix.length} 种操作 (${caps.opMix.join(', ')}). 可以这么跑, 但每种操作的调参建议不同 — 可以按操作分组.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.itemCount} item(s), ${caps.opMix.length === 1 ? caps.opMix[0] : `${caps.opMix.length} ops`}, ready to run.`,
    reasonZh: `${caps.itemCount} 个 item, ${caps.opMix.length === 1 ? caps.opMix[0] : `${caps.opMix.length} 种操作`}, 可以开始.`,
  };
}

export function operationLabel(op: string, locale: Locale): string {
  const map: Record<string, [string, string]> = {
    quantization: ['Quantization', '量化'],
    neuron_pruning: ['Neuron pruning', '神经元剪枝'],
    vocab_pruning: ['Vocab pruning', '词表裁剪'],
  };
  const pair = map[op];
  if (!pair) return op;
  return locale === 'zh' ? pair[1] : pair[0];
}

export function buildBatchContextSnippet(
  caps: BatchCapabilities,
  locale: Locale,
  options: BatchNarrationOptions = {},
): string {
  const fixture = options.fixture === true;
  const lines: string[] = [
    locale === 'zh'
      ? fixture ? `## 你所在的批量优化样例队列` : `## 你所在的批量优化 (Batch) 队列`
      : fixture ? `## YOUR SAMPLE BATCH OPTIMIZATION QUEUE` : `## YOUR BATCH OPTIMIZATION QUEUE`,
    locale === 'zh'
      ? `这条流程${fixture ? '展示虚构 onboarding 样例' : `把 ${caps.itemCount} 个独立模型按队列串行跑 (每个 load → pipeline → 卸载)`}. 你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 是 brain, 用第一人称解释每一项的意图 + 风险 + 结果.`
      : `This flow ${fixture ? 'shows fictional onboarding sample data' : `runs ${caps.itemCount} independent models serially (each: load → pipeline → unload)`}. You (${caps.brain?.model_name ?? 'the loaded LLM'}) are the brain; explain intent + risks + results in first person.`,
    fixture
      ? locale === 'zh'
        ? `- Fixture boundary: 这些队列项和结果是 EdgeStudio 内置练手数据, 不是本机真实优化证据, 不能用于报告、发布或性能宣称.`
        : `- Fixture boundary: these queue items and results are built-in EdgeStudio onboarding samples only. They are not real local optimization evidence and cannot be used for reports, releases, or performance claims.`
      : '',
    `- Queue size: ${caps.itemCount} item(s)`,
    `- Run phase: ${caps.runPhase}`,
    `- Operation mix: ${caps.opMix.map((op) => `${op}×${caps.opHistogram[op]}`).join(', ') || '(none)'}`,
    caps.bitsRange.values.length > 0
      ? `- Quantization bits range: ${caps.bitsRange.min}..${caps.bitsRange.max} (values: [${caps.bitsRange.values.join(', ')}])`
      : '',
    caps.duplicateLabels.length > 0
      ? `- Duplicates: ${caps.duplicateLabels.join(', ')} (same path repeated)`
      : '',
    caps.dangerousQuant.length > 0
      ? `- ⚠ Dangerous quant items (bits ≤ 3): ${caps.dangerousQuant.map((it) => `${it.label}@${it.bits}bit`).join(', ')}`
      : '',
    ``,
  ];

  if (caps.completedCount > 0) {
    lines.push(
      locale === 'zh' ? `### 当前结果集 (cite when explaining):` : `### Current results cohort (cite when explaining):`,
      `- Completed: ${caps.completedCount}/${caps.itemCount} (${caps.successCount} ok, ${caps.failedCount} failed) — success rate ${(caps.successRate * 100).toFixed(0)}%`,
    );
    if (caps.successCount > 0) {
      lines.push(
        `- Total saved: ${(caps.totalSavedBytes / 1e9).toFixed(2)} GB (avg ${caps.avgSavingsPct.toFixed(1)}% per item)`,
        `- Avg duration: ${caps.avgDurationSec.toFixed(1)}s · slowest ${caps.slowestItem?.label ?? '?'} (${caps.slowestItem?.duration_seconds?.toFixed(1) ?? '?'}s) · fastest ${caps.fastestItem?.label ?? '?'} (${caps.fastestItem?.duration_seconds?.toFixed(1) ?? '?'}s)`,
        caps.bestSavingsItem
          ? `- Best savings: ${caps.bestSavingsItem.label} (${(((caps.bestSavingsItem.original_size ?? 0) - (caps.bestSavingsItem.result_size ?? 0)) / Math.max(1, caps.bestSavingsItem.original_size ?? 1) * 100).toFixed(1)}%)`
          : '',
        caps.worstSavingsItem && caps.worstSavingsItem !== caps.bestSavingsItem
          ? `- Worst savings: ${caps.worstSavingsItem.label} (${(((caps.worstSavingsItem.original_size ?? 0) - (caps.worstSavingsItem.result_size ?? 0)) / Math.max(1, caps.worstSavingsItem.original_size ?? 1) * 100).toFixed(1)}%)`
          : '',
      );
    }
    if (caps.failedCount > 0) {
      const failed = caps.results.filter((r) => !r.success).slice(0, 3);
      lines.push(
        `- Failed items: ${failed.map((r) => `${r.label} (${r.error || 'unknown error'})`).join('; ')}`,
      );
    }
    lines.push(``);
  }

  lines.push(
    locale === 'zh' ? `### 操作语义 (cite when explaining):` : `### Operation semantics (cite when explaining):`,
    locale === 'zh'
      ? `- quantization: 把 fp16/bf16 权重压到 N-bit (典型 4-bit). 收益 ~75%, ≤3-bit 一般 PPL 崩.`
      : `- quantization: compress fp16/bf16 weights to N-bit (typical 4-bit). ~75% size win, ≤3-bit usually breaks PPL.`,
    locale === 'zh'
      ? `- neuron_pruning: 按 activation 重要性删 FFN 通道. 需要先跑 profile, 收益 10-30% 但可能伤性能.`
      : `- neuron_pruning: drop FFN channels by activation importance. Needs prior profile; 10-30% size, may hurt perf.`,
    locale === 'zh'
      ? `- vocab_pruning: 删未在语料中出现的 token. 收益主要来自 embedding + lm_head, 跑在 bf16 base 上最稳.`
      : `- vocab_pruning: drop tokens unused in corpus. Savings mostly from embeddings + lm_head; safest on bf16 base.`,
    ``,
    locale === 'zh'
      ? fixture
        ? `### Fixture 边界: 样例没有云调用, 但也没有真实写出权重. 回答时必须提醒用户添加真实本地路径并点击全部执行后才会得到本机证据.`
        : `### 北极星 §1 主权: 整条队列在本机内存里串行跑, 0 次云调用; 每个产出权重也在本机.`
      : fixture
        ? `### Fixture boundary: the sample makes no cloud calls, but it did not produce real artifacts either. Remind users to add real local paths and run the queue for local evidence.`
        : `### North-star §1 sovereignty: queue runs serially in local memory, zero cloud calls; each artifact stays local.`,
  );
  return lines.filter(Boolean).join('\n');
}

export function buildBatchAutoBrief(
  caps: BatchCapabilities,
  locale: Locale,
  options: BatchNarrationOptions = {},
): string {
  if (options.fixture && caps.completedCount > 0) {
    if (locale === 'zh') {
      return `当前队列是 EdgeStudio 内置的虚构批量优化练手样例, 不是本机真实产物. 用 2-3 句话作为 brain 教用户读这组结果: Original/Result 看体积变化, Savings 看收益, Time 看单项耗时; 最后提醒添加真实本地路径并点击全部执行后才会得到可用于发布的证据. 第一人称, 不宣称真实性能提升.`;
    }
    return `The current queue is EdgeStudio's fictional batch optimization sample, not real local artifacts. In 2-3 sentences as the brain, teach the user how to read it: Original/Result for size change, Savings for compression gain, Time for per-item duration; end by reminding them to add real local paths and run the queue for publishable evidence. First person, no real improvement claims.`;
  }
  if (locale === 'zh') {
    if (caps.itemCount === 0) {
      return `批量队列还是空的. 用 2-3 句作为 brain 解释批量优化适合什么场景 (一次跑 N 个模型变体省切换/同一族多 size 量化扫描), 邀请用户加模型. 第一人称.`;
    }
    if (caps.runPhase === 'running') {
      return `正在跑批量 (${caps.completedCount}/${caps.itemCount} 完成, 当前阶段 ${caps.runPhase}). 用 2-3 句作为 brain 简述当前进度 + 已知结果信号 (${caps.successCount} 成功 / ${caps.failedCount} 失败). 第一人称.`;
    }
    if (caps.runPhase === 'complete') {
      return `批量全部完成! ${caps.successCount}/${caps.itemCount} 成功, 总省 ${(caps.totalSavedBytes / 1e9).toFixed(2)} GB, 平均 ${caps.avgSavingsPct.toFixed(1)}%. 用 2-3 句作为 brain 总结这批的结果 (亮点+异常), 推荐下一步. 第一人称, 引用具体数字.`;
    }
    if (caps.runPhase === 'partial') {
      return `批量部分完成 (${caps.successCount} 成功, ${caps.failedCount} 失败). 用 2-3 句作为 brain 解读: 失败的可能根因 (架构不支持 / quant 配置错 / 路径不对), 推荐失败 item 怎么排查. 第一人称.`;
    }
    if (caps.runPhase === 'allFailed') {
      return `批量全部失败 (${caps.failedCount} 个). 用 2-3 句作为 brain 直说: 通常这意味着系统级问题 (mlx 版本 / 路径权限 / 内存不够). 给 1 个最可能根因 + 排查命令. 第一人称.`;
    }
    // idle
    return `批量队列已配 ${caps.itemCount} 个 item (${caps.opMix.join(', ')}). 用 2-3 句作为 brain 评估这套队列: 预期总耗时 / 风险点 / 推荐. 第一人称.`;
  }
  // English
  if (caps.itemCount === 0) {
    return `Batch queue is empty. In 2-3 sentences as brain, explain when batch helps (sweep N variants of one family / N quant bit-widths in one go) and invite the user to add models. First person.`;
  }
  if (caps.runPhase === 'running') {
    return `Running batch (${caps.completedCount}/${caps.itemCount} done, phase ${caps.runPhase}). In 2-3 sentences as brain, summarize progress and what the early results say (${caps.successCount} ok / ${caps.failedCount} failed). First person.`;
  }
  if (caps.runPhase === 'complete') {
    return `Batch complete! ${caps.successCount}/${caps.itemCount} ok, total saved ${(caps.totalSavedBytes / 1e9).toFixed(2)} GB, avg ${caps.avgSavingsPct.toFixed(1)}%. In 2-3 sentences as brain, summarize highlights and outliers, recommend next step. First person, cite numbers.`;
  }
  if (caps.runPhase === 'partial') {
    return `Partial completion (${caps.successCount} ok, ${caps.failedCount} failed). In 2-3 sentences as brain, interpret: likely root causes for the failures (arch unsupported / quant misconfig / bad path), and how to triage. First person.`;
  }
  if (caps.runPhase === 'allFailed') {
    return `All ${caps.failedCount} items failed. In 2-3 sentences as brain, be blunt: this usually means a system-level issue (mlx version / path perms / OOM). Give the single most likely root cause + a triage command. First person.`;
  }
  return `Batch queue holds ${caps.itemCount} item(s) (${caps.opMix.join(', ')}). In 2-3 sentences as brain, assess the queue: expected total time / risks / recommendations. First person.`;
}

export function getBatchSuggestedPrompts(
  caps: BatchCapabilities,
  locale: Locale,
  options: BatchNarrationOptions = {},
): { label: string; prompt: string }[] {
  if (options.fixture && caps.completedCount > 0) {
    if (locale === 'zh') {
      return [
        { label: '读样例', prompt: `这是虚构 batch 样例. 用第一人称解释这几行的 Original / Result / Savings / Time 怎么看, 哪个样例最值得保留, 并提醒这不是实测证据.` },
        { label: '换成真跑', prompt: `告诉用户如何把样例换成真实批量优化: 添加本地模型路径, 确认 operation 和 bits, 点击全部执行, 跑完后先看哪几列. 简短具体.` },
        { label: '节省是否合理', prompt: `用这份练手数据讲 4-bit 量化一般为什么能省约 70% 体积, 哪些结果算异常. 明确说这是样例数据.` },
        { label: '产物能不能发布', prompt: `解释样例没有真实 output_dir, 不能发布; 真实批量运行结束后才会有本机产物目录. 语气直接.` },
      ];
    }
    return [
      { label: 'Read sample', prompt: `This is a fictional batch sample. In first person, explain how to read Original / Result / Savings / Time, which sample looks worth keeping, and remind me this is not measured evidence.` },
      { label: 'Switch to real run', prompt: `Tell the user how to replace the sample with a real batch: add local model paths, confirm operation and bits, click Run All, then inspect which columns first. Keep it concrete.` },
      { label: 'Are savings sane?', prompt: `Use this practice data to explain why 4-bit quantization often saves around 70% size and what result would look suspicious. Say clearly it is sample data.` },
      { label: 'Can I ship this?', prompt: `Explain that the sample has no real output_dir and cannot be released; a real batch run must finish before local artifact directories exist. Be direct.` },
    ];
  }

  if (locale === 'zh') {
    if (caps.itemCount === 0) {
      return [
        { label: '🎯 批量优化适合什么', prompt: `用 2-3 句作为 brain 解释批量优化的甜点场景 (例如一次扫 4-bit/8-bit/3-bit 看哪个 PPL/size 最甜, 或者同族多个 size 一次出端侧版).` },
        { label: '⚖️ 一次跑几个合理', prompt: `M-series Mac (192GB) 上, 单次批量队列建议放几个模型? 串行跑没并行, 给我一个"过夜 vs 半小时"的两档建议.` },
        { label: '🚧 批量常见坑', prompt: `批量优化最容易踩的 3 个坑 (跨架构混跑 / 路径不存在 / 中间失败一个全队列卡住), 各给一个排查动作.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 这个批量队列全程在本机内存串行运算, 0 次云调用 — 与"上传到 SaaS 平台跑量化"的对比.` },
      ];
    }
    if (caps.runPhase === 'complete') {
      return [
        { label: '📊 这批怎么样', prompt: `批量完成 ${caps.successCount}/${caps.itemCount}, 总省 ${(caps.totalSavedBytes / 1e9).toFixed(2)} GB. 用 2-3 句作为 brain 评估: 哪个最值, 哪个不值, 整体平均 ${caps.avgSavingsPct.toFixed(1)}% 是否在预期.` },
        { label: '🔍 异常检测', prompt: `这批结果里 ${caps.bestSavingsItem ? `最大省 ${caps.bestSavingsItem.label}` : ''}${caps.worstSavingsItem && caps.worstSavingsItem !== caps.bestSavingsItem ? `, 最少省 ${caps.worstSavingsItem.label}` : ''}. 这种差距正常吗 (是架构差异 还是配置差异 还是 bug)?` },
        { label: '⏱️ 耗时分析', prompt: `${caps.slowestItem ? `最慢 ${caps.slowestItem.label} ${caps.slowestItem.duration_seconds?.toFixed(1)}s, ` : ''}${caps.fastestItem ? `最快 ${caps.fastestItem.label} ${caps.fastestItem.duration_seconds?.toFixed(1)}s.` : ''} 这个差距合理吗, 给一个 profiling 推荐.` },
        { label: '📦 接下来', prompt: `这批结果都跑出来了. 用 2-3 句话告诉用户接下来怎么做: 哪些可以直接 export 到 iPhone, 哪些需要 benchmark 验证, 哪些建议丢弃.` },
      ];
    }
    if (caps.runPhase === 'partial' || caps.runPhase === 'allFailed') {
      const failed = caps.results.filter((r) => !r.success).slice(0, 3);
      return [
        { label: '🔥 失败根因', prompt: `这批 ${caps.failedCount} 个失败 (${failed.map((r) => `${r.label}: ${r.error || 'unknown'}`).join('; ')}). 用 2-3 句作为 brain 推断最可能根因 + 第一步怎么排查.` },
        { label: '🛠️ 怎么补救', prompt: `失败的 item 是不是配置错 (bits 不对 / 路径不存在 / 操作不适用) 还是系统问题 (内存不够 / mlx 版本不对)? 给一个分流决策树.` },
        { label: '⚖️ 是否值得 retry', prompt: `失败的几个 item 调一下参数 (例如 bits=4 → 8, 或者改成另一个 operation), 是否还有救? 用 2-3 句给具体建议.` },
        { label: '🌍 端侧主权', prompt: `即使部分失败, 整批仍然全程在本机. 用 2-3 句强调: 这种 fail-locally 比"云端跑挂了等支持"快得多.` },
      ];
    }
    if (caps.runPhase === 'running') {
      return [
        { label: '⏱️ 还要多久', prompt: `已完成 ${caps.completedCount}/${caps.itemCount}, 当前还在跑. 用 2-3 句作为 brain 估剩余时间 (基于已完成的平均 ${caps.avgDurationSec.toFixed(1)}s) + 是否要继续等.` },
        { label: '🎯 已完成的分析', prompt: `已完成 ${caps.completedCount} 个里 ${caps.successCount} 成功. 用 2-3 句给一个 early read: 这批趋势是 healthy 还是要警惕 (基于现有数据).` },
        { label: '⚠️ 中途要不要停', prompt: `如果发现某个 item 卡住或失败, 是否应该停掉整批? 给一个判断准则.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 当前正在本机串行跑 ${caps.itemCount} 个模型, 0 网络调用.` },
      ];
    }
    // idle
    return [
      { label: '⚙️ 这套配置怎么样', prompt: `${caps.itemCount} 个 item: ${caps.opMix.join(', ')}. ${caps.bitsRange.values.length > 0 ? `quant bits ${caps.bitsRange.min}-${caps.bitsRange.max}.` : ''} 用 2-3 句作为 brain 评估这套队列, 重点说风险点.` },
      { label: '⏱️ 估时和内存', prompt: `这 ${caps.itemCount} 个串行跑, 在 M2 Ultra 192GB 上预估总耗时, 内存峰值大概多少 (不会同时载入, 但每个单独需要满载内存).` },
      { label: '🎯 顺序优化', prompt: `当前队列顺序是 ${caps.items.map((it) => it.label).slice(0, 5).join(' → ')}${caps.itemCount > 5 ? '…' : ''}. 这个顺序合理吗 (例如先小后大 / 先稳后险)?` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句强调: 这个批量队列全程在本机内存串行运算, 0 次云调用.` },
    ];
  }
  // English
  if (caps.itemCount === 0) {
    return [
      { label: '🎯 When does batch help', prompt: `In 2-3 sentences as brain, explain the sweet spot for batch (sweep 4-bit/8-bit/3-bit on one model to see PPL/size tradeoff, or one family across many sizes for edge deployment).` },
      { label: '⚖️ How many per batch', prompt: `On an M-series Mac (192GB), how many models in one queue? It runs serially with no parallelism — give an "overnight vs 30 min" two-tier recommendation.` },
      { label: '🚧 Batch pitfalls', prompt: `The 3 most common batch pitfalls (mixed-architecture queue / bad paths / one mid-queue failure stalling everything), with one triage move each.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this entire queue runs locally serially with zero cloud — vs uploading models to a SaaS quantization service.` },
    ];
  }
  if (caps.runPhase === 'complete') {
    return [
      { label: '📊 How did this batch go', prompt: `Batch complete ${caps.successCount}/${caps.itemCount}, total saved ${(caps.totalSavedBytes / 1e9).toFixed(2)} GB. In 2-3 sentences as brain, assess: which is the best deal, which is dud, is the avg ${caps.avgSavingsPct.toFixed(1)}% in the expected range.` },
      { label: '🔍 Outlier check', prompt: `${caps.bestSavingsItem ? `Best savings ${caps.bestSavingsItem.label}` : ''}${caps.worstSavingsItem && caps.worstSavingsItem !== caps.bestSavingsItem ? `, worst ${caps.worstSavingsItem.label}` : ''}. Is the spread normal (architectural difference vs config diff vs bug)?` },
      { label: '⏱️ Duration analysis', prompt: `${caps.slowestItem ? `Slowest ${caps.slowestItem.label} at ${caps.slowestItem.duration_seconds?.toFixed(1)}s, ` : ''}${caps.fastestItem ? `fastest ${caps.fastestItem.label} at ${caps.fastestItem.duration_seconds?.toFixed(1)}s.` : ''} Is this spread reasonable, give a profiling recommendation.` },
      { label: '📦 Next steps', prompt: `Results are out. In 2-3 sentences as brain, tell the user what to do: which to export to iPhone now, which to benchmark first, which to discard.` },
    ];
  }
  if (caps.runPhase === 'partial' || caps.runPhase === 'allFailed') {
    const failed = caps.results.filter((r) => !r.success).slice(0, 3);
    return [
      { label: '🔥 Failure root cause', prompt: `${caps.failedCount} failed in this batch (${failed.map((r) => `${r.label}: ${r.error || 'unknown'}`).join('; ')}). In 2-3 sentences as brain, infer the most likely root cause + the first triage step.` },
      { label: '🛠️ How to recover', prompt: `Are the failures config issues (wrong bits / bad path / op not applicable) or system issues (OOM / mlx version)? Give a triage decision tree.` },
      { label: '⚖️ Worth retrying', prompt: `For the failed items, are they salvageable by tweaking params (bits=4 → 8, or switching operation)? In 2-3 sentences give concrete advice.` },
      { label: '🌍 Edge sovereignty', prompt: `Even with partial failure, the whole batch stayed local. In 2-3 sentences, emphasise: this fail-locally beats "cloud job died, wait for support".` },
    ];
  }
  if (caps.runPhase === 'running') {
    return [
      { label: '⏱️ How much longer', prompt: `${caps.completedCount}/${caps.itemCount} done, currently running. In 2-3 sentences as brain, estimate remaining time (based on avg ${caps.avgDurationSec.toFixed(1)}s of completed items) and whether to keep waiting.` },
      { label: '🎯 Early read', prompt: `${caps.successCount} of ${caps.completedCount} done items succeeded. In 2-3 sentences, give an early read: is the trend healthy or worrying (based on data so far).` },
      { label: '⚠️ Stop or continue', prompt: `If one item gets stuck or fails mid-run, should the whole batch be stopped? Give a decision rule.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this is ${caps.itemCount} models running serially on this Mac, zero network calls.` },
    ];
  }
  // idle
  return [
    { label: '⚙️ Is this config sane', prompt: `${caps.itemCount} items: ${caps.opMix.join(', ')}. ${caps.bitsRange.values.length > 0 ? `Quant bits ${caps.bitsRange.min}-${caps.bitsRange.max}.` : ''} In 2-3 sentences as brain, assess this queue, focus on risks.` },
    { label: '⏱️ Time + memory', prompt: `For ${caps.itemCount} items running serially on M2 Ultra 192GB, estimate total wall-clock and peak memory (not all loaded at once, but each loads fully).` },
    { label: '🎯 Order optimization', prompt: `Current order: ${caps.items.map((it) => it.label).slice(0, 5).join(' → ')}${caps.itemCount > 5 ? '…' : ''}. Is this order good (small-first / safe-first / risky-last)?` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this entire batch queue runs serially on this Mac with zero cloud calls.` },
  ];
}
