// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * pipelineInsights — risk assessment + chat helpers for /pipeline page.
 *
 * The advisor returns "applicable" suggestions but the pipeline page lets
 * users build any stack. Without warnings, a user with a 4-bit model can
 * add another `quantization` step (→ 2-bit catastrophe) or `layer_pruning`
 * on a hybrid model (→ delete a full-attention layer = broken model).
 *
 * This file:
 *  - per-(operation, model) risk assessment (safe / caution / danger)
 *  - step-order validation (e.g. vocab_pruning should come before quantization)
 *  - model-aware chat brief recommending the right stack
 */
import type { ModelInfo } from '@/api/types';
import { formatParamCount, formatSize } from '@/lib/utils';
import { deriveModelFacts } from '@/lib/chatPrompts';

type Locale = 'en' | 'zh';

export type Operation =
  | 'neuron_pruning'
  | 'layer_pruning'
  | 'quantization'
  | 'vocab_pruning'
  | 'embedding_quantization';

export type Risk = 'safe' | 'caution' | 'danger';

export interface OperationRisk {
  level: Risk;
  reason: string;
  reasonZh: string;
}

/** Per-operation × model state risk assessment. */
export function assessOperationRisk(
  op: string,
  params: Record<string, unknown>,
  model: ModelInfo,
  hasProfile: boolean,
): OperationRisk {
  const f = deriveModelFacts(model);
  const isQuantized = f.quantBits > 0 && f.quantBits < 16;

  switch (op) {
    case 'quantization': {
      const requestedBits = typeof params.bits === 'number' ? (params.bits as number) : 4;
      if (isQuantized && requestedBits < f.quantBits) {
        return {
          level: 'danger',
          reason: `Already ${f.quantBits}-bit; requantizing to ${requestedBits}-bit will severely degrade quality (re-pack from packed weights). Consider unloading + loading the unquantized base.`,
          reasonZh: `已 ${f.quantBits}-bit, 再压到 ${requestedBits}-bit 会严重劣化质量 (基于已量化 weights 二次量化). 建议先加载未量化的 base 模型再量化.`,
        };
      }
      if (isQuantized && requestedBits === f.quantBits) {
        return {
          level: 'caution',
          reason: `Already ${f.quantBits}-bit; re-running with same bits is a no-op or worse.`,
          reasonZh: `已 ${f.quantBits}-bit, 同位再跑无意义 (或更糟).`,
        };
      }
      if (requestedBits <= 2) {
        return {
          level: 'danger',
          reason: `${requestedBits}-bit quantization typically destroys quality except in research workflows. Default safe range: 4-8 bits.`,
          reasonZh: `${requestedBits}-bit 量化通常会摧毁质量 (除非研究用途). 安全范围: 4-8 bits.`,
        };
      }
      return {
        level: 'safe',
        reason: `${requestedBits}-bit quantization is in the safe zone for an unquantized base model.`,
        reasonZh: `${requestedBits}-bit 量化对未量化 base 模型是安全的.`,
      };
    }

    case 'embedding_quantization': {
      if (isQuantized) {
        return {
          level: 'caution',
          reason: `Embeddings are likely already part of your ${f.quantBits}-bit pack. Adds little but may double-quantize.`,
          reasonZh: `embedding 大概率已在 ${f.quantBits}-bit pack 内. 重复量化收益小风险大.`,
        };
      }
      return { level: 'safe', reason: 'Quantizing embeddings reclaims a chunky tensor.', reasonZh: '量化 embedding 可以回收大体积 tensor.' };
    }

    case 'layer_pruning': {
      const layers = Array.isArray(params.layers_to_remove) ? (params.layers_to_remove as number[]) : [];
      const cfgLayerTypes = ((model.config as Record<string, unknown>)?.text_config as Record<string, unknown>)?.layer_types
        ?? (model.config as Record<string, unknown>)?.layer_types;
      const isHybrid = Array.isArray(cfgLayerTypes) && new Set(cfgLayerTypes as string[]).size > 1;
      if (layers.length === 0) {
        return {
          level: 'caution',
          reason: 'No layers selected — provide indices (irreversible operation).',
          reasonZh: '未选择层 — 必须提供 layer indices (不可逆操作).',
        };
      }
      if (isHybrid) {
        return {
          level: 'danger',
          reason: `Hybrid attention model — layers have different roles (full vs linear). Removing a full-attn layer can break the model. Cross-check layer_types in /architecture first.`,
          reasonZh: `Hybrid attention 模型 — 不同层职责不同 (full vs linear). 删 full-attn 层可能让模型崩. 先去 /architecture 看 layer_types.`,
        };
      }
      if (layers.length > Math.floor(f.numLayers * 0.25)) {
        return {
          level: 'danger',
          reason: `Removing ${layers.length}/${f.numLayers} layers (>25%) typically yields broken outputs. Stay below 15% for stability.`,
          reasonZh: `删 ${layers.length}/${f.numLayers} 层 (>25%) 通常输出会崩. 稳定性建议 <15%.`,
        };
      }
      if (layers.length > Math.floor(f.numLayers * 0.10)) {
        return {
          level: 'caution',
          reason: `Removing ${layers.length}/${f.numLayers} layers — measure PPL carefully.`,
          reasonZh: `删 ${layers.length}/${f.numLayers} 层 — 注意 PPL 测量.`,
        };
      }
      return { level: 'safe', reason: `Conservative layer prune (${layers.length}/${f.numLayers}).`, reasonZh: `保守的 layer 剪枝 (${layers.length}/${f.numLayers}).` };
    }

    case 'neuron_pruning': {
      if (!hasProfile) {
        return {
          level: 'danger',
          reason: 'Neuron pruning requires an activation profile to identify dead neurons. Run /activation first; otherwise this prunes random neurons.',
          reasonZh: '神经元剪枝必须有 activation profile 才能识别死神经元. 先去 /activation 跑 profile, 否则剪到随机神经元.',
        };
      }
      const reduction = typeof params.max_reduction === 'number' ? (params.max_reduction as number) : 0.5;
      if (reduction > 0.5) {
        return {
          level: 'caution',
          reason: `${(reduction * 100).toFixed(0)}% reduction is aggressive — may degrade quality on diverse tasks.`,
          reasonZh: `${(reduction * 100).toFixed(0)}% 剪枝幅度激进 — 在多样任务上可能劣化.`,
        };
      }
      return { level: 'safe', reason: 'Profile-guided neuron prune is the safest aggressive optimization.', reasonZh: 'Profile 引导的神经元剪枝是最安全的激进优化.' };
    }

    case 'vocab_pruning': {
      if (f.vocabSize > 100000) {
        return {
          level: 'safe',
          reason: `Vocab is ${f.vocabSize.toLocaleString()} (large). Trimming unused tokens recovers significant memory with negligible quality cost.`,
          reasonZh: `vocab ${f.vocabSize.toLocaleString()} (较大). 裁未用 token 能省可观内存, 质量损失可忽略.`,
        };
      }
      return { level: 'safe', reason: 'Vocab pruning is generally safe.', reasonZh: 'vocab 剪枝通常安全.' };
    }

    default:
      return { level: 'caution', reason: 'Unknown operation.', reasonZh: '未知操作.' };
  }
}

/** Step-order check: returns warnings (empty if all good). */
export function validateStepOrder(steps: { operation: string }[]): string[] {
  const warnings: string[] = [];
  const order = steps.map((s) => s.operation);
  const idxQuant = order.indexOf('quantization');
  const idxVocab = order.indexOf('vocab_pruning');
  const idxLayer = order.indexOf('layer_pruning');
  const idxNeuron = order.indexOf('neuron_pruning');

  if (idxVocab >= 0 && idxQuant >= 0 && idxVocab > idxQuant) {
    warnings.push('vocab_pruning should run BEFORE quantization (smaller embed → smaller quant pack).');
  }
  if (idxLayer >= 0 && idxNeuron >= 0 && idxLayer > idxNeuron) {
    warnings.push('layer_pruning should run BEFORE neuron_pruning (otherwise neuron stats become stale).');
  }
  if (idxQuant >= 0 && idxNeuron >= 0 && idxNeuron > idxQuant) {
    warnings.push('neuron_pruning should run BEFORE quantization (pruning operates on full-precision weights).');
  }
  // Same operation twice is a smell
  const counts: Record<string, number> = {};
  for (const op of order) counts[op] = (counts[op] ?? 0) + 1;
  for (const [op, n] of Object.entries(counts)) {
    if (n > 1) warnings.push(`${op} appears ${n}× in pipeline — likely an error.`);
  }
  return warnings;
}

export interface PipelineSummary {
  totalRiskScore: number;          // 0 = all safe, higher = more dangerous
  hasDanger: boolean;
  hasCaution: boolean;
  orderWarnings: string[];
  perStepRisk: Array<{ index: number; op: string; risk: OperationRisk }>;
}

const RISK_SCORE: Record<Risk, number> = { safe: 0, caution: 1, danger: 3 };

export function summarizePipeline(
  steps: Array<{ operation: string; params: Record<string, unknown> }>,
  model: ModelInfo,
  hasProfile: boolean,
): PipelineSummary {
  const perStepRisk = steps.map((s, i) => ({
    index: i,
    op: s.operation,
    risk: assessOperationRisk(s.operation, s.params, model, hasProfile),
  }));
  const totalRiskScore = perStepRisk.reduce((sum, r) => sum + RISK_SCORE[r.risk.level], 0);
  return {
    totalRiskScore,
    hasDanger: perStepRisk.some((r) => r.risk.level === 'danger'),
    hasCaution: perStepRisk.some((r) => r.risk.level === 'caution'),
    orderWarnings: validateStepOrder(steps),
    perStepRisk,
  };
}

export function buildPipelineContextSnippet(
  model: ModelInfo,
  steps: Array<{ operation: string; params: Record<string, unknown> }>,
  summary: PipelineSummary | null,
  hasProfile: boolean,
): string {
  const f = deriveModelFacts(model);
  const lines: string[] = [
    `## YOUR PIPELINE STATE (Edge Studio)`,
    `- Current you: ${model.model_name}, ${formatParamCount(f.totalParams)} params, ${formatSize(f.totalSizeBytes)}, ${f.quantBits}-bit, ${f.numLayers} layers, vocab ${f.vocabSize.toLocaleString()}`,
    `- Activation profile available: ${hasProfile ? 'YES (unlocks neuron_pruning)' : 'NO (neuron_pruning will prune random neurons)'}`,
    ``,
    `### Available operations (5 total):`,
    `- vocab_pruning — trims unused tokens, low risk on large vocabs (${f.vocabSize > 100000 ? 'YOUR vocab is large = good fit' : 'your vocab is modest'})`,
    `- quantization — ${f.quantBits > 0 ? `YOU ARE ALREADY ${f.quantBits}-bit, re-quantizing will degrade quality` : 'safe at 4-8 bits'}`,
    `- embedding_quantization — ${f.quantBits > 0 ? 'likely already done as part of pack' : 'reclaims chunky embed tensor'}`,
    `- layer_pruning — IRREVERSIBLE; needs careful index choice; hybrid models (different layer types) are extra dangerous`,
    `- neuron_pruning — needs activation profile; aggressive >50% reduction degrades diverse tasks`,
  ];
  if (steps.length > 0 && summary) {
    lines.push(``, `### User has built ${steps.length} step(s):`);
    summary.perStepRisk.forEach((p) => {
      lines.push(`- step ${p.index + 1}: \`${p.op}\` — ${p.risk.level.toUpperCase()}: ${p.risk.reason}`);
    });
    if (summary.orderWarnings.length > 0) {
      lines.push(``, `### Order warnings:`);
      summary.orderWarnings.forEach((w) => lines.push(`- ${w}`));
    }
  } else {
    lines.push(``, `### User has no steps yet — recommend a pipeline based on YOUR specific state.`);
  }
  return lines.join('\n');
}

export function buildPipelineAutoBrief(
  model: ModelInfo,
  steps: Array<{ operation: string; params: Record<string, unknown> }>,
  summary: PipelineSummary | null,
  hasProfile: boolean,
  locale: Locale,
): string {
  void model; void hasProfile;
  const stepCount = steps.length;
  if (locale === 'zh') {
    if (stepCount === 0) {
      return `用户还没添加任何 step. 用 2-3 句话基于你的当前状态 (是否已量化, 是否有 profile, vocab 多大) 推荐 1-2 个最适合你的 stage 顺序. 不要列项, 写成自然一段话, 结尾邀请用户点 Add Step 或 Import from Advisor.`;
    }
    if (summary?.hasDanger) {
      return `检测到 ${summary.perStepRisk.filter(p => p.risk.level === 'danger').length} 个 danger step. 用 2-3 句话直白告诉用户哪个 step 危险 + 为什么 + 推荐怎么改. 用第一人称.`;
    }
    return `用户已配置 ${stepCount} 个 step. 用 2-3 句话评估这套组合是否合理 (顺序 + 累计风险), 给用户一个 GO / WAIT / ABORT 建议. 用第一人称.`;
  }
  if (stepCount === 0) {
    return `User has no steps yet. In 2-3 sentences, based on your current state (already quantized? have profile? large vocab?), recommend 1-2 stages that fit you best, in the right order. End by inviting them to click Add Step or Import from Advisor. No bullets.`;
  }
  if (summary?.hasDanger) {
    return `${summary.perStepRisk.filter(p => p.risk.level === 'danger').length} dangerous step(s) detected. In 2-3 sentences, tell the user plainly which step is risky + why + how to fix. Speak in first person.`;
  }
  return `User has configured ${stepCount} step(s). In 2-3 sentences, assess whether this combination is sensible (order + cumulative risk), and give a GO / WAIT / ABORT recommendation. First person.`;
}

export function getPipelineSuggestedPrompts(
  model: ModelInfo,
  steps: Array<{ operation: string; params: Record<string, unknown> }>,
  summary: PipelineSummary | null,
  hasProfile: boolean,
  locale: Locale,
): { label: string; prompt: string }[] {
  const f = deriveModelFacts(model);
  void summary;
  if (locale === 'zh') {
    return [
      { label: '🎯 推荐顺序', prompt: `基于我现在的状态 (${f.quantBits}-bit 量化, ${f.numLayers} 层, vocab ${f.vocabSize.toLocaleString()}, ${hasProfile ? '有 profile' : '没 profile'}), 推荐一个 2-3 step 的具体 pipeline 顺序 + 每个 step 的参数. 给我一个能直接照抄的方案。` },
      { label: '⚠️ 风险评估', prompt: steps.length > 0
        ? `评估我刚配的 ${steps.length} 个 step 的累计风险, 哪个最有可能让 PPL 飙升? 给出一个改进版。`
        : `pipeline 里 5 个 operation, 哪个对我现在的状态最危险? 给出 1 个我应该避开的 operation + 1 个我必须先做的前置 (如 profile)。` },
      { label: '📐 vocab 收益', prompt: f.vocabSize > 100000
        ? `我 vocab 有 ${f.vocabSize.toLocaleString()} 个 token, vocab_pruning 大概能省多少 MB? 对我的多语言能力有什么影响?`
        : `vocab_pruning 适合我吗 (${f.vocabSize.toLocaleString()} vocab)? 收益和风险各是什么?` },
      { label: '🪞 我做不到的优化', prompt: `诚实告诉用户: 在这个 pipeline 里有哪些 stage 因为我已经的状态 (${f.quantBits}-bit, hybrid?) 已经不适合我. 第一人称承认。` },
    ];
  }
  return [
    { label: '🎯 Recommended order', prompt: `Based on my current state (${f.quantBits}-bit, ${f.numLayers} layers, vocab ${f.vocabSize.toLocaleString()}, ${hasProfile ? 'profile available' : 'no profile'}), recommend a concrete 2-3 step pipeline + parameters. Give me something I can copy directly.` },
    { label: '⚠️ Risk review', prompt: steps.length > 0
      ? `Review the ${steps.length} steps I just configured for cumulative risk. Which one is most likely to spike my PPL? Give an improved version.`
      : `Of the 5 operations available, which is most dangerous for my current state? Give one to avoid + one prerequisite I must run first (e.g. profile).` },
    { label: '📐 Vocab payoff', prompt: f.vocabSize > 100000
      ? `My vocab is ${f.vocabSize.toLocaleString()} tokens — roughly how much MB can vocab_pruning save? What's the impact on my multilingual ability?`
      : `Is vocab_pruning worth it for me (${f.vocabSize.toLocaleString()} vocab)? Trade off the savings vs the risk.` },
    { label: '🪞 What I can\'t do', prompt: `Honestly tell the user which stages in this pipeline are no longer appropriate for me given my current state (${f.quantBits}-bit, hybrid?). Admit limitations in first person.` },
  ];
}

/** Display chip color for a risk level. */
export const RISK_CHIP_CLASS: Record<Risk, string> = {
  safe: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-300',
  caution: 'bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-300',
  danger: 'bg-red-100 text-red-700 dark:bg-red-500/20 dark:text-red-300',
};

export const RISK_BORDER_CLASS: Record<Risk, string> = {
  safe: 'border-l-emerald-500',
  caution: 'border-l-amber-500',
  danger: 'border-l-red-500',
};
