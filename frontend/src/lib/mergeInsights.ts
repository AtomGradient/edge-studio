// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * mergeInsights — derived N-slot state + chat helpers for the /merge page.
 *
 * Model merging takes N source models + a strategy (linear / slerp / ties /
 * task_arithmetic) and produces a single merged model. Unlike the previous
 * 3-slot pages (duplex / mesh / training / distill), this is a VARIABLE-N
 * multi-component capability — N can be 2..many.
 *
 * §9.1 multi-component pattern continues here, just with an array slot:
 *  - per-slot validity (path filled?)
 *  - per-strategy constraint check (slerp = exactly 2; task_arithmetic
 *    requires base; ties + linear take any N >= 2)
 *  - weights array sanity (must equal N or empty for "uniform")
 *  - weights sum-to-1 sanity (warn if user typed weights but they don't sum)
 *
 * Pure functions; no fetching. Page passes already-fetched inputs in.
 *
 * Sovereignty (§9.2 mandatory): merging is a pure local weight-space
 * operation; no inference of any input model is shipped anywhere.
 */
import type { ModelInfo } from '@/api/types';
import type { MergeResult } from '@/api/types';

type Locale = 'en' | 'zh';

export type Strategy = 'linear' | 'slerp' | 'ties' | 'task_arithmetic';

export interface MergeCapabilities {
  /** Source model paths (length === modelDirs.length, even if some are empty). */
  modelDirs: string[];
  /** Path of the base for task_arithmetic (only used when strategy = task_arithmetic). */
  baseModelDir: string;
  /** Strategy chosen. */
  strategy: Strategy;
  /** Comma-separated weights string the user typed. */
  weightsStr: string;
  /** Parsed weights array (NaNs filtered out). */
  weights: number[];
  /** TIES density param. */
  density: number;
  /** Brain LLM that narrates as itself. */
  brain: ModelInfo | null;
  /** Result of the last merge run. */
  result: MergeResult | null;

  // ── Derived ────────────────────────────────────────────────────────────
  /** Names extracted from path tails (length = modelDirs.length). */
  sourceNames: string[];
  baseName: string;
  /** Number of valid (non-empty) source paths. */
  validCount: number;
  /** True if strategy-specific min count met. */
  meetsCount: boolean;
  /** True if (strategy != task_arithmetic) OR baseModelDir filled. */
  hasBase: boolean;
  /** True if weights provided AND len(weights) == validCount. */
  weightsAligned: boolean;
  /** Sum of provided weights. */
  weightsSum: number;
  /** Whether the configuration is ready to start. */
  canStart: boolean;
  /** Brief status of the last run. */
  status: 'none' | 'running' | 'success' | 'failed';
}

function tail(p: string): string {
  if (!p) return '';
  return p.replace(/\/$/, '').split('/').pop() || p;
}

function strategyMinCount(s: Strategy): number {
  return s === 'slerp' ? 2 : 2;  // all need at least 2; slerp also caps at 2
}

function strategyMaxCount(s: Strategy): number {
  return s === 'slerp' ? 2 : 999;
}

export function deriveMergeCapabilities(
  modelDirs: string[],
  baseModelDir: string,
  strategy: Strategy,
  weightsStr: string,
  density: number,
  brain: ModelInfo | null,
  result: MergeResult | null,
): MergeCapabilities {
  const valid = modelDirs.filter((d) => d.trim());
  const validCount = valid.length;

  const min = strategyMinCount(strategy);
  const max = strategyMaxCount(strategy);
  const meetsCount = validCount >= min && validCount <= max;
  const hasBase = strategy !== 'task_arithmetic' || !!baseModelDir.trim();

  const weights = weightsStr.trim()
    ? weightsStr.split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n))
    : [];
  const weightsAligned = weights.length === 0 || weights.length === validCount;
  const weightsSum = weights.reduce((s, w) => s + w, 0);

  let status: MergeCapabilities['status'];
  if (!result) status = 'none';
  else if (result.success) status = 'success';
  else if (result.error) status = 'failed';
  else status = 'running';

  const canStart = meetsCount && hasBase;

  return {
    modelDirs,
    baseModelDir,
    strategy,
    weightsStr,
    weights,
    density,
    brain,
    result,
    sourceNames: modelDirs.map(tail),
    baseName: tail(baseModelDir),
    validCount,
    meetsCount,
    hasBase,
    weightsAligned,
    weightsSum,
    canStart,
    status,
  };
}

export type MergeRiskLevel = 'safe' | 'caution' | 'danger';
export interface MergeRisk {
  level: MergeRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Per-(config) risk:
 *  - danger:  slerp with N != 2 (strategy hard constraint)
 *  - danger:  task_arithmetic without base
 *  - caution: < 2 valid models
 *  - caution: weights provided but length != N
 *  - caution: weights provided but sum far from 1.0 (linear typically wants ~1)
 *  - caution: ties density extreme (< 0.1 or > 0.9)
 *  - caution: too many models (>5) — diminishing returns + memory
 *  - safe:    all checks pass
 */
export function assessMergeConfig(caps: MergeCapabilities): MergeRisk {
  if (caps.strategy === 'slerp' && caps.validCount !== 2) {
    return {
      level: 'danger',
      reason: `SLERP requires exactly 2 models, you have ${caps.validCount}.`,
      reasonZh: `SLERP 必须正好 2 个模型, 当前 ${caps.validCount} 个.`,
    };
  }
  if (caps.strategy === 'task_arithmetic' && !caps.hasBase) {
    return {
      level: 'danger',
      reason: 'task_arithmetic needs a base model — pick one above.',
      reasonZh: 'task_arithmetic 需要 base 模型 — 上面选一个.',
    };
  }
  if (caps.validCount < 2) {
    return {
      level: 'caution',
      reason: `Need at least 2 source models, have ${caps.validCount}.`,
      reasonZh: `至少需要 2 个源模型, 当前 ${caps.validCount} 个.`,
    };
  }
  if (caps.weights.length > 0 && !caps.weightsAligned) {
    return {
      level: 'caution',
      reason: `Weights count (${caps.weights.length}) doesn't match model count (${caps.validCount}). Either supply ${caps.validCount} weights or leave blank for uniform.`,
      reasonZh: `权重个数 (${caps.weights.length}) 与模型数 (${caps.validCount}) 不一致. 给 ${caps.validCount} 个权重, 或留空使用均匀权重.`,
    };
  }
  if (caps.strategy === 'linear' && caps.weights.length > 0 && Math.abs(caps.weightsSum - 1.0) > 0.05) {
    return {
      level: 'caution',
      reason: `Linear weights sum to ${caps.weightsSum.toFixed(3)}, not 1.0. Output magnitude will scale unexpectedly.`,
      reasonZh: `Linear 权重总和 ${caps.weightsSum.toFixed(3)} 不为 1.0. 输出权重会缩放, 可能引起异常.`,
    };
  }
  if (caps.strategy === 'ties' && (caps.density < 0.1 || caps.density > 0.9)) {
    return {
      level: 'caution',
      reason: `TIES density ${caps.density} is extreme (< 0.1 = nearly empty merge; > 0.9 = nearly all kept). Sweet spot 0.3-0.7.`,
      reasonZh: `TIES density ${caps.density} 偏极端 (< 0.1 = 几乎空合并; > 0.9 = 几乎全保留). 甜点区 0.3-0.7.`,
    };
  }
  if (caps.validCount > 5) {
    return {
      level: 'caution',
      reason: `${caps.validCount} models is a lot — diminishing returns and high memory load.`,
      reasonZh: `${caps.validCount} 个模型偏多 — 收益递减且内存压力大.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.strategy} · ${caps.validCount} models${caps.weights.length > 0 ? ` · weights sum ${caps.weightsSum.toFixed(2)}` : ' · uniform weights'} · ready.`,
    reasonZh: `${caps.strategy} · ${caps.validCount} 个模型${caps.weights.length > 0 ? ` · 权重和 ${caps.weightsSum.toFixed(2)}` : ' · 均匀权重'} · 可以开始.`,
  };
}

export function strategyLabel(s: Strategy, locale: Locale): string {
  if (locale === 'zh') {
    return ({
      linear: '加权平均',
      slerp: '球面插值',
      ties: 'TIES (稀疏+签名)',
      task_arithmetic: 'Task 算术',
    } as Record<Strategy, string>)[s];
  }
  return ({
    linear: 'Linear (weighted avg)',
    slerp: 'SLERP (2 models)',
    ties: 'TIES (sparse + sign)',
    task_arithmetic: 'Task arithmetic',
  } as Record<Strategy, string>)[s];
}

export function buildMergeContextSnippet(
  caps: MergeCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的模型合并 (Merge) 流程` : `## YOUR MODEL MERGE FLOW`,
    locale === 'zh'
      ? `这条流程把 ${caps.validCount} 个源模型在权重空间里融合成 1 个新模型. 你 (${caps.brain?.model_name ?? '已加载的 LLM'}) 是 brain, 用第一人称解释合并算法 + 风险.`
      : `This flow fuses ${caps.validCount} source models into 1 in weight space. You (${caps.brain?.model_name ?? 'the loaded LLM'}) are the brain; explain the merge algorithm + risks in first person.`,
    `- Strategy: ${caps.strategy.toUpperCase()} (${strategyLabel(caps.strategy, locale)})`,
    `- Source models (${caps.validCount}):`,
    ...caps.modelDirs.filter((d) => d.trim()).map((d) => `  - ${tail(d)} (${d})`),
    caps.strategy === 'task_arithmetic'
      ? `- Base model: ${caps.baseName || '(none)'}`
      : '',
    caps.weights.length > 0
      ? `- Weights: [${caps.weights.join(', ')}] · sum=${caps.weightsSum.toFixed(3)}`
      : `- Weights: uniform (${(1 / Math.max(1, caps.validCount)).toFixed(3)} each)`,
    caps.strategy === 'ties' ? `- TIES density: ${caps.density}` : '',
    `- Status: ${caps.status}${caps.result?.success ? ` · output ${caps.result.output_dir}` : ''}`,
    ``,
    locale === 'zh'
      ? `### 合并算法语义 (cite when explaining):`
      : `### Algorithm semantics (cite when explaining):`,
    locale === 'zh'
      ? `- linear = w1·M1 + w2·M2 + ... 直接加权平均权重 tensor (最简单, 易出 NaN 如果模型差异大).`
      : `- linear = w1·M1 + w2·M2 + ... weighted parameter average (simplest, can NaN if models diverge).`,
    locale === 'zh'
      ? `- slerp = 在 2 个模型间走 hypersphere 大圆弧 (保留范数, 比 linear 更稳).`
      : `- slerp = great-circle arc on the hypersphere between 2 models (preserves norm, more stable than linear).`,
    locale === 'zh'
      ? `- ties = 稀疏化每个 task vector + sign consensus, 减少冲突 (适合 N>=3).`
      : `- ties = sparsify each task vector + sign consensus, reduces conflict (best for N>=3).`,
    locale === 'zh'
      ? `- task_arithmetic = base + Σ(Mi - base) 直接加 task vector (需要 base 与各 Mi 同源).`
      : `- task_arithmetic = base + Σ(Mi - base), summing task vectors (requires base with shared origin).`,
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: 合并是纯本地权重运算, 无任何模型推理流出.`
      : `### North-star §1 sovereignty: merge is pure local weight-space arithmetic; no inference traffic leaves.`,
  ];
  return lines.filter(Boolean).join('\n');
}

export function buildMergeAutoBrief(
  caps: MergeCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (caps.validCount < 2) {
      return `用户还没填够源模型 (当前 ${caps.validCount}, 需要 >= 2). 用 2-3 句作为 brain 解释 merge 是干什么的 (在权重空间融合多个模型, 不是 ensemble), 邀请用户补足. 第一人称.`;
    }
    if (caps.status === 'success' && caps.result) {
      return `合并完成! 输出 ${caps.result.output_dir}. 用 2-3 句作为 brain 评估这次合并 (${caps.strategy} 策略合 ${caps.validCount} 个模型), 推荐用户接下来 benchmark 看效果. 第一人称, 引用具体数字.`;
    }
    return `用户配了 ${caps.strategy.toUpperCase()} 合 ${caps.validCount} 个模型. 用 2-3 句作为 brain 评估配置: 这套参数会出什么样的合并模型 (倾向哪个源 / 风险点 / 推荐). 第一人称.`;
  }
  if (caps.validCount < 2) {
    return `User has only ${caps.validCount} source models (need >= 2). In 2-3 sentences as brain, explain merge is weight-space fusion (not ensemble) and invite filling more slots. First person.`;
  }
  if (caps.status === 'success' && caps.result) {
    return `Merge done! Output at ${caps.result.output_dir}. In 2-3 sentences as brain, assess (${caps.strategy} on ${caps.validCount} models), recommend benchmarking next. First person, cite numbers.`;
  }
  return `User configured ${caps.strategy.toUpperCase()} on ${caps.validCount} models. In 2-3 sentences as brain, assess (what kind of merged model this produces / which source dominates / risks / recommendation). First person.`;
}

export function getMergeSuggestedPrompts(
  caps: MergeCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.validCount < 2) {
      return [
        { label: '🎯 merge 是干啥', prompt: `用 2-3 句话作为 brain 解释 model merging 是什么 (权重空间融合 ≠ ensemble inference), 什么场景用 (合并 fine-tuned 变体 / 多任务模型 fork / 缓解灾难性遗忘).` },
        { label: '⚖️ 4 种策略选哪个', prompt: `linear / slerp / ties / task_arithmetic 各自的甜点场景? 给一个具体决策树 (2 个模型 → slerp; >2 个 → ties; 有 base + 多个 task fork → task_arithmetic; 其他 → linear).` },
        { label: '🚧 常见坑', prompt: `用 2-3 句话讲 merge 最容易踩的 3 个坑 (vocab 不一致 / quant 不一致 / family 不同导致 layout 不匹配), 以及怎么 verify 模型可合并.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句话强调: 合并是纯权重运算, 不会让任何源模型的"推理痕迹"留下来 (与"上传到云端 mergekit"的对比).` },
      ];
    }
    if (caps.status === 'success' && caps.result) {
      return [
        { label: '📊 合得怎么样', prompt: `合并完成. 用 2-3 句作为 brain 评估这次 ${caps.strategy.toUpperCase()} 合 ${caps.validCount} 个模型: 出来的合并模型偏向哪个源, 哪些任务可能受益, 哪些可能退化.` },
        { label: '🚦 验证清单', prompt: `给一个 5 步验证清单, 让用户怎么知道这次合并质量好不好 (smoke test prompts / PPL benchmark / 各源原任务回归测试).` },
        { label: '⚖️ 与原模型对比', prompt: `如果让我对比合并模型 vs 原 ${caps.validCount} 个源模型, 在哪些维度上合并模型必然胜出 (容量没增加但学到 union), 哪些必然不如单一源 (没有 specialist depth).` },
        { label: '📦 接下来', prompt: `合并模型在 ${caps.result.output_dir}. 用 2-3 句话告诉用户接下来怎么用 (benchmark → 量化 → 推到 iPhone), 哪一步最关键.` },
      ];
    }
    return [
      { label: '⚙️ 这套配置怎么样', prompt: `${caps.strategy.toUpperCase()} 合 ${caps.validCount} 个模型${caps.weights.length > 0 ? `, 权重 [${caps.weights.join(', ')}]` : ' 均匀权重'}. 用 2-3 句作为 brain 评估这套配置 (各源贡献 / 是否 dominant / 风险).` },
      { label: '🎯 weights 怎么调', prompt: `${caps.weights.length === 0 ? '当前均匀权重. 推荐什么时候应该用非均匀权重 (例如想让某个模型主导).' : `当前权重 [${caps.weights.join(', ')}], 总和 ${caps.weightsSum.toFixed(3)}. 这个分布合理吗 (是否有模型权重过低, 几乎被忽略)?`} 给具体调整建议.` },
      { label: '⏱️ 估时和内存', prompt: `估计合并这 ${caps.validCount} 个模型在 M-series Mac 上要多久, 内存峰值大概多少 (合并需要同时载入所有源).` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句话强调: 这次合并 ${caps.validCount} 个模型, 全程权重在本机内存里运算, 0 次云调用. 与 mergekit 云服务的对比.` },
    ];
  }
  // English
  if (caps.validCount < 2) {
    return [
      { label: '🎯 What is merging', prompt: `In 2-3 sentences as brain, explain model merging (weight-space fusion ≠ ensemble inference), and good use cases (combining fine-tuned variants / multi-task model forks / mitigating forgetting).` },
      { label: '⚖️ Which strategy', prompt: `linear / slerp / ties / task_arithmetic — sweet spot for each? Give a concrete decision tree (2 models → slerp; >2 → ties; base + task forks → task_arithmetic; else → linear).` },
      { label: '🚧 Common pitfalls', prompt: `In 2-3 sentences, list the top 3 merge pitfalls (vocab mismatch / quant mismatch / family-incompatible layouts) and how to verify mergeability.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: merge is pure weight arithmetic, no inference trace of any source leaves. Contrast to "cloud mergekit".` },
    ];
  }
  if (caps.status === 'success' && caps.result) {
    return [
      { label: '📊 How well did it merge', prompt: `Merge complete. In 2-3 sentences as brain, assess ${caps.strategy.toUpperCase()} on ${caps.validCount} models: which source dominates, which tasks benefit, which might regress.` },
      { label: '🚦 Verification checklist', prompt: `Give a 5-step checklist for the user to know if this merge is good (smoke prompts / PPL bench / per-source task regression).` },
      { label: '⚖️ vs sources', prompt: `Comparing merged vs the ${caps.validCount} sources, on which dimensions does merged necessarily win (no capacity growth but union of learned signals), on which does it lose vs a single specialist.` },
      { label: '📦 Next steps', prompt: `Merged model at ${caps.result.output_dir}. In 2-3 sentences, tell user what's next (benchmark → quantize → push to iPhone) and which step is most critical.` },
    ];
  }
  return [
    { label: '⚙️ Is this config sane', prompt: `${caps.strategy.toUpperCase()} on ${caps.validCount} models${caps.weights.length > 0 ? `, weights [${caps.weights.join(', ')}]` : ', uniform weights'}. In 2-3 sentences as brain, assess (per-source contribution / dominance / risk).` },
    { label: '🎯 How to set weights', prompt: `${caps.weights.length === 0 ? 'Currently uniform weights. When should the user pick non-uniform (e.g. wanting one model to dominate)?' : `Weights [${caps.weights.join(', ')}], sum ${caps.weightsSum.toFixed(3)}. Is this distribution sensible (any model weighted too low, nearly ignored)?`} Give concrete tuning suggestions.` },
    { label: '⏱️ Time + memory', prompt: `Estimate merge time on M-series Mac for ${caps.validCount} models, and peak memory (merge requires loading all sources simultaneously).` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: this merge of ${caps.validCount} models is pure local weight-space arithmetic, zero cloud. Contrast to mergekit cloud services.` },
  ];
}
