// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * distillationInsights — derived 3-component state + chat helpers for the
 * /distill page (knowledge distillation: teacher → student via dataset).
 *
 * THREE components must combine for distillation to work (§9.1 multi-component
 * pattern, identical shape to duplex/mesh/personal-training):
 *   1. teacher — large pre-trained LLM (the "knowledge source")
 *   2. student — small target LLM (the "learner")
 *   3. dataset — distillation prompts (where teacher logits guide student)
 *
 * This file:
 *  - derives 3-slot capabilities + missing list + algorithm (offline / TAID)
 *  - assesses pairing sanity (teacher must be larger than student / category
 *    must match LLM-vs-VLM / vocab compat heuristic)
 *  - composes per-page system snippet so the loaded LLM can speak as the
 *    student about what it'll absorb from the teacher
 *  - generates 4 step-aware suggested prompts (configuring vs running vs done)
 *
 * Pure functions; no fetching. Page passes already-fetched inputs in.
 *
 * Sovereignty (§9.2 mandatory): distillation runs entirely on this Mac;
 * neither the teacher's logits nor the dataset ever leave the machine.
 */
import type { ModelInfo } from '@/api/types';
import type { DistillResult } from '@/api/types';

type Locale = 'en' | 'zh';

export type DistillMode = 'offline' | 'taid';

export type LossBucket = 'unknown' | 'broken' | 'concerning' | 'good' | 'excellent';
export type AdapterStatus = 'none' | 'training' | 'success' | 'failed';

export interface DistillationCapabilities {
  /** Path inputs from the form. */
  teacherDir: string;
  studentDir: string;
  datasetPath: string;
  /** Pretty names extracted from path tails (or model_name when loaded). */
  teacherName: string;
  studentName: string;
  datasetName: string;
  /** Loaded LLM that will narrate as itself (preferably the student). */
  brain: ModelInfo | null;
  /** Algorithm + hyperparams. */
  mode: DistillMode;
  numEpochs: number;
  batchSize: number;
  learningRate: number;
  temperature: number;
  alpha: number;
  maxSamples: number;
  /** Last completed result (null = no run yet). */
  result: DistillResult | null;

  // ── Derived booleans ────────────────────────────────────────────────────
  hasTeacher: boolean;
  hasStudent: boolean;
  hasDataset: boolean;
  allReady: boolean;
  hasBrain: boolean;
  /** Slot names that are still empty. */
  missing: Array<'teacher' | 'student' | 'dataset'>;

  // ── Derived from result ─────────────────────────────────────────────────
  adapterStatus: AdapterStatus;
  lossBucket: LossBucket;
  finalLoss: number;          // 0 if unknown
  durationS: number;          // 0 if unknown

  // ── Sanity heuristics ───────────────────────────────────────────────────
  /** Teacher vs student same-path warning (means user picked the same model). */
  teacherSameAsStudent: boolean;
}

function tail(p: string): string {
  if (!p) return '';
  return p.replace(/\/$/, '').split('/').pop() || p;
}

function lossToBucket(loss: number): LossBucket {
  if (!Number.isFinite(loss) || loss <= 0) return 'unknown';
  if (loss > 5) return 'broken';
  if (loss > 2) return 'concerning';
  if (loss > 0.8) return 'good';
  return 'excellent';
}

export function deriveDistillationCapabilities(
  teacherDir: string,
  studentDir: string,
  datasetPath: string,
  brain: ModelInfo | null,
  mode: DistillMode,
  numEpochs: number,
  batchSize: number,
  learningRate: number,
  temperature: number,
  alpha: number,
  maxSamples: number,
  result: DistillResult | null,
): DistillationCapabilities {
  const hasTeacher = !!teacherDir.trim();
  const hasStudent = !!studentDir.trim();
  const hasDataset = !!datasetPath.trim();
  const allReady = hasTeacher && hasStudent && hasDataset;
  const hasBrain = !!brain;

  const missing: DistillationCapabilities['missing'] = [];
  if (!hasTeacher) missing.push('teacher');
  if (!hasStudent) missing.push('student');
  if (!hasDataset) missing.push('dataset');

  const teacherSameAsStudent = !!teacherDir && teacherDir === studentDir;

  let adapterStatus: AdapterStatus;
  if (!result) adapterStatus = 'none';
  else if (result.success) adapterStatus = 'success';
  else if (result.error) adapterStatus = 'failed';
  else adapterStatus = 'training';

  const finalLoss = result?.final_loss ?? 0;
  const lossBucket = result?.success ? lossToBucket(finalLoss) : 'unknown';
  const durationS = result?.duration_seconds ?? 0;

  return {
    teacherDir,
    studentDir,
    datasetPath,
    teacherName: tail(teacherDir),
    studentName: brain?.model_name || tail(studentDir),
    datasetName: tail(datasetPath),
    brain,
    mode,
    numEpochs,
    batchSize,
    learningRate,
    temperature,
    alpha,
    maxSamples,
    result,
    hasTeacher,
    hasStudent,
    hasDataset,
    allReady,
    hasBrain,
    missing,
    adapterStatus,
    lossBucket,
    finalLoss,
    durationS,
    teacherSameAsStudent,
  };
}

export type DistillRiskLevel = 'safe' | 'caution' | 'danger';
export interface DistillRisk {
  level: DistillRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Per-(config) sanity risk.
 *  - danger:  teacher path == student path (degenerate distillation)
 *  - caution: any slot empty
 *  - caution: alpha 0 or 1 (degenerate weighting — pure CE or pure KL)
 *  - caution: temperature < 1 (sharpens too much, kills knowledge transfer)
 *  - caution: teacher 4-bit + student 4-bit (logit noise compounds)
 *  - safe:    all 3 slots filled, alpha in (0,1), T >= 1
 */
export function assessDistillationConfig(
  caps: DistillationCapabilities,
): DistillRisk {
  if (caps.teacherSameAsStudent) {
    return {
      level: 'danger',
      reason: 'Teacher and student point to the SAME path — distillation degenerates to no-op. Pick a smaller student.',
      reasonZh: 'Teacher 和 Student 指向同一路径 — 蒸馏退化成自己学自己 (no-op). 换一个更小的 student.',
    };
  }
  if (!caps.allReady) {
    return {
      level: 'caution',
      reason: `Missing: ${caps.missing.join(' + ')}. All three are required to start.`,
      reasonZh: `还缺: ${caps.missing.join(' + ')}. 三个槽位都填了才能开始.`,
    };
  }
  if (caps.alpha <= 0 || caps.alpha >= 1) {
    return {
      level: 'caution',
      reason: `alpha=${caps.alpha} is degenerate (0 = pure cross-entropy ignoring teacher; 1 = pure KL ignoring labels). Try 0.3-0.7.`,
      reasonZh: `alpha=${caps.alpha} 是退化值 (0 = 纯 CE 忽略 teacher; 1 = 纯 KL 忽略标签). 试 0.3-0.7.`,
    };
  }
  if (caps.temperature < 1) {
    return {
      level: 'caution',
      reason: `temperature=${caps.temperature} is < 1 — sharpens teacher logits too much, kills the soft-label signal that makes KD work. Try 2-4.`,
      reasonZh: `temperature=${caps.temperature} < 1 — 让 teacher logit 过于尖锐, 软标签信号丢失. 试 2-4.`,
    };
  }
  if (caps.numEpochs > 10) {
    return {
      level: 'caution',
      reason: `${caps.numEpochs} epochs is very high for KD — usually 3-5 is enough; risk of overfitting to teacher quirks.`,
      reasonZh: `${caps.numEpochs} epoch 对 KD 来说很高 — 通常 3-5 epoch 就够; 容易学到 teacher 的怪癖.`,
    };
  }
  return {
    level: 'safe',
    reason: `${caps.mode.toUpperCase()} · ${caps.numEpochs} ep · α=${caps.alpha} T=${caps.temperature} · ready to start.`,
    reasonZh: `${caps.mode.toUpperCase()} · ${caps.numEpochs} epoch · α=${caps.alpha} T=${caps.temperature} · 可以开始.`,
  };
}

export function buildDistillationContextSnippet(
  caps: DistillationCapabilities,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的蒸馏 (Distillation) 流程` : `## YOUR DISTILLATION FLOW`,
    locale === 'zh'
      ? `你 (${caps.studentName || '已加载的 student'}) 是 student. 一个更大的 ${caps.teacherName || 'teacher'} 模型会用 ${caps.datasetName || 'dataset'} 来教你 — 通过它的 soft logits 让你学到比硬标签更丰富的信号.`
      : `You (${caps.studentName || 'the loaded student'}) are the STUDENT. A larger ${caps.teacherName || 'teacher'} model will teach you using ${caps.datasetName || 'dataset'} — its soft logits transfer more knowledge than hard labels alone.`,
    `- Teacher path: ${caps.teacherDir || '(none)'}`,
    `- Student path: ${caps.studentDir || '(none)'}`,
    `- Dataset path: ${caps.datasetPath || '(none)'}`,
    `- Mode: ${caps.mode.toUpperCase()} ${caps.mode === 'taid' ? '(TAID — adaptive temperature dynamic)' : '(offline KD — fixed T+α)'}`,
    `- Hyperparams: ${caps.numEpochs} epochs · batch ${caps.batchSize} · lr ${caps.learningRate} · T=${caps.temperature} · α=${caps.alpha}${caps.maxSamples > 0 ? ` · max_samples=${caps.maxSamples}` : ''}`,
    `- Status: ${caps.adapterStatus}${caps.result?.success ? ` · final_loss=${caps.finalLoss.toFixed(4)} (${caps.lossBucket}) · ${caps.durationS.toFixed(1)}s` : ''}`,
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权: teacher 的 logits + dataset + student 权重 全部本地, 0 次云调用.`
      : `### North-star §1 sovereignty: teacher logits + dataset + student weights all local, zero cloud calls.`,
    ``,
    locale === 'zh'
      ? `### 当用户问 "我能学到什么" / "为什么用 KD" / "怎么调 T 和 α":`
      : `### When user asks "what will I learn" / "why use KD" / "how to tune T and α":`,
    locale === 'zh'
      ? `- 用第一人称 (我作为 student) 引用上面具体数字.`
      : `- First person (as the student); cite numbers above.`,
  ];
  return lines.join('\n');
}

export function buildDistillationAutoBrief(
  caps: DistillationCapabilities,
  locale: Locale,
): string {
  if (locale === 'zh') {
    if (!caps.allReady) {
      return `还缺组件: ${caps.missing.join(' + ')}. 用 2-3 句话作为 student (${caps.studentName || '我'}) 解释三个角色 (teacher / student / dataset) 各自做什么 + 还需要补哪个. 第一人称, 不列项.`;
    }
    if (caps.adapterStatus === 'success' && caps.result) {
      return `蒸馏完成! Final loss ${caps.finalLoss.toFixed(4)} (${caps.lossBucket}), ${caps.durationS.toFixed(1)}s. 用 2-3 句话作为已经学完 teacher 知识的 student 评估 (loss 表示什么 / 我学到了多少 / 推荐用户测试). 第一人称引用 bucket.`;
    }
    if (caps.adapterStatus === 'training') {
      return `蒸馏进行中. 用 2-3 句话作为 student 解释当前在做什么 (forward teacher → soft logits → forward student → KL+CE loss → backprop), 第一人称.`;
    }
    return `蒸馏配置完成: ${caps.mode.toUpperCase()} α=${caps.alpha} T=${caps.temperature}. 用 2-3 句话作为 student 评估这套参数合不合理 (alpha/T 选得对吗 / 适合 ${caps.numEpochs} epoch 吗), 第一人称.`;
  }
  if (!caps.allReady) {
    return `Missing slots: ${caps.missing.join(' + ')}. In 2-3 sentences, as the student (${caps.studentName || 'me'}), explain the three roles (teacher / student / dataset) and which is missing. First person, no bullets.`;
  }
  if (caps.adapterStatus === 'success' && caps.result) {
    return `Distillation done. Final loss ${caps.finalLoss.toFixed(4)} (${caps.lossBucket}), ${caps.durationS.toFixed(1)}s. In 2-3 sentences, as the student that just absorbed teacher knowledge, assess (what does loss mean / how much did I learn / invite test). First person, cite bucket.`;
  }
  if (caps.adapterStatus === 'training') {
    return `Distillation in progress. In 2-3 sentences, as the student, explain what's happening (forward teacher → soft logits → forward student → KL+CE loss → backprop). First person.`;
  }
  return `Config ready: ${caps.mode.toUpperCase()} α=${caps.alpha} T=${caps.temperature}. In 2-3 sentences, as the student, assess if these hyperparams fit (good alpha/T for ${caps.numEpochs} epochs?). First person.`;
}

export function getDistillationSuggestedPrompts(
  caps: DistillationCapabilities,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (!caps.allReady) {
      return [
        { label: '🎯 我能学到什么', prompt: `用 2-3 句话作为 ${caps.studentName || 'student'}, 解释通过蒸馏从 ${caps.teacherName || 'teacher'} 我能学到 vs 学不到的东西 (soft 信号传递, 但参数差距过大学不全).` },
        { label: '⚖️ T 和 α 怎么调', prompt: `用 2-3 句话讲 KD 关键超参 T (temperature) 和 α (KL/CE 权重) 的物理含义 + 一个常用甜点配置 (T=2-4, α=0.5-0.7).` },
        { label: '📋 dataset 要什么样', prompt: `用 2-3 句话告诉用户: 蒸馏 dataset 应该长什么样 (prompts 列表 / 还是 prompt + completion 对 / 还是纯文本), 给 1 个 JSONL 例子.` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句话作为 student 强调: 整个蒸馏过程 teacher 权重 + 我的权重 + dataset + 训练日志全部本地, 0 次云调用. 这与 "调 OpenAI fine-tune API" 的主权差异.` },
      ];
    }
    if (caps.adapterStatus === 'success' && caps.result) {
      return [
        { label: '📊 我学得怎么样', prompt: `Final loss ${caps.finalLoss.toFixed(4)} (bucket: ${caps.lossBucket}). 用 2-3 句话作为蒸馏完的 student 评估: 我学到了什么程度 / 哪些场景比 base student 强 / 哪些可能差了.` },
        { label: '🚦 该不该继续训', prompt: `loss=${caps.finalLoss.toFixed(4)}, ${caps.numEpochs} epoch, ${caps.durationS.toFixed(1)}s. 给一个 STOP / TUNE / RETRAIN 建议: 这个 adapter 直接用还是再调 hyperparam?` },
        { label: '🆚 我 vs base', prompt: `蒸馏完的我 vs 没蒸馏的原 student. 给一个具体对比维度 (推理速度 / 短文本质量 / 长文本质量 / 复杂推理). 第一人称.` },
        { label: '📦 怎么导出', prompt: `adapter 路径 ${caps.result.output_dir}. 用 2-3 句话告诉用户怎么把这个蒸馏过的 student 用起来 (直接 load / merge 进 base / 推到 iPhone via mTLS).` },
      ];
    }
    return [
      { label: '⚙️ 这套参数合理吗', prompt: `${caps.mode.toUpperCase()} 模式, α=${caps.alpha} T=${caps.temperature}, ${caps.numEpochs} epoch, lr=${caps.learningRate}. 作为 ${caps.studentName || 'student'} 评估这套配置 (会不会 overfit / underfit / soft 信号够强吗).` },
      { label: '⏱️ 估计要多久', prompt: `dataset ${caps.datasetName}, ${caps.numEpochs} epoch, batch ${caps.batchSize}. 估计训练时间 (M-series Mac), 内存峰值大概多少.` },
      { label: '🔬 offline vs TAID', prompt: `用户选了 ${caps.mode.toUpperCase()}. 作为 student 解释: offline 是固定 T+α, TAID 是动态调温, 哪个更适合我目前的状况.` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句话强调: 整个 KD 过程 teacher logits + dataset + my weights 全部本地, 0 次云调用. 这与云端蒸馏服务的差异.` },
    ];
  }
  // English
  if (!caps.allReady) {
    return [
      { label: '🎯 What I will learn', prompt: `In 2-3 sentences as ${caps.studentName || 'student'}, explain what I can absorb from ${caps.teacherName || 'teacher'} via distillation (soft-signal transfer) vs what I can't (large parameter gap).` },
      { label: '⚖️ How to tune T and α', prompt: `In 2-3 sentences, walk through KD's key hyperparams T (temperature) and α (KL/CE weighting) — physical meaning + a sweet-spot recipe (T=2-4, α=0.5-0.7).` },
      { label: '📋 What dataset format', prompt: `In 2-3 sentences, tell the user the right dataset shape (prompt list / prompt+completion pairs / raw text), with one JSONL example.` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences as student, emphasise: teacher weights + my weights + dataset + training logs all local, zero cloud. Sovereignty contrast to "OpenAI fine-tune API".` },
    ];
  }
  if (caps.adapterStatus === 'success' && caps.result) {
    return [
      { label: '📊 How well did I learn', prompt: `Final loss ${caps.finalLoss.toFixed(4)} (bucket: ${caps.lossBucket}). In 2-3 sentences, as the distilled student, assess: how well did I learn / where do I now beat the base student / where might I be worse.` },
      { label: '🚦 STOP / TUNE / RETRAIN', prompt: `loss=${caps.finalLoss.toFixed(4)}, ${caps.numEpochs} epochs, ${caps.durationS.toFixed(1)}s. Give a verdict: ship as-is, tune hyperparams, or retrain?` },
      { label: '🆚 Me vs base', prompt: `Distilled me vs un-distilled student. Give a concrete comparison dimension (inference speed / short-form quality / long-form / complex reasoning). First person.` },
      { label: '📦 How to export', prompt: `Adapter at ${caps.result.output_dir}. In 2-3 sentences, tell the user how to use the distilled student (direct load / merge into base / push to iPhone via mTLS).` },
    ];
  }
  return [
    { label: '⚙️ Are these params right', prompt: `${caps.mode.toUpperCase()} mode, α=${caps.alpha} T=${caps.temperature}, ${caps.numEpochs} epochs, lr=${caps.learningRate}. As ${caps.studentName || 'student'}, assess (overfit risk / underfit / soft-signal strength).` },
    { label: '⏱️ How long', prompt: `Dataset ${caps.datasetName}, ${caps.numEpochs} epochs, batch ${caps.batchSize}. Estimate training time (M-series Mac) and peak memory.` },
    { label: '🔬 offline vs TAID', prompt: `User picked ${caps.mode.toUpperCase()}. As student, explain: offline = fixed T+α, TAID = adaptive temperature; which fits my situation better.` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: full KD pipeline (teacher logits + dataset + my weights) is local, zero cloud. Contrast to cloud distillation services.` },
  ];
}
