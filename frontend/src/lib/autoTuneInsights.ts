// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * autoTuneInsights — derived helpers for /auto-tune (parameter sweep over a
 * single loaded model to find best inference hyperparams).
 *
 * Different from /auto-optimizer (which sweeps quant+prune):
 *  - /auto-tune = sweep inference-time knobs (kv strategy / batch / etc.)
 *  - benchmarks each config N runs, returns the fastest stable one
 *
 * §9.1 single-component (one model, many configs) variant. Sovereignty
 * (§9.2) — sweep is local, no cloud telemetry.
 */
import type { ModelInfo, AutoTuneResult, TuneCandidate } from '@/api/types';

type Locale = 'en' | 'zh';

export type TPSBucket = 'unknown' | 'slow' | 'ok' | 'fast' | 'blazing';

export interface AutoTuneCapabilities {
  brain: ModelInfo | null;
  maxTokens: number;
  numRuns: number;
  forceRerun: boolean;
  result: AutoTuneResult | null;

  // Derived
  runPhase: 'noResult' | 'hasResult';
  bestCandidate: TuneCandidate | null;
  bestTPS: number;
  bestTPSBucket: TPSBucket;
  candidateCount: number;
  variability: number;       // (max - min) / mean across all candidates' TPS
  runsBucket: 'sparse' | 'minimal' | 'good' | 'thorough';
  isCached: boolean;
}

function tpsBucket(tps: number): TPSBucket {
  if (!Number.isFinite(tps) || tps <= 0) return 'unknown';
  if (tps < 10) return 'slow';
  if (tps < 30) return 'ok';
  if (tps < 60) return 'fast';
  return 'blazing';
}

function runsToBucket(n: number): AutoTuneCapabilities['runsBucket'] {
  if (n <= 1) return 'sparse';
  if (n <= 2) return 'minimal';
  if (n <= 5) return 'good';
  return 'thorough';
}

export function deriveAutoTuneCapabilities(
  brain: ModelInfo | null,
  maxTokens: number,
  numRuns: number,
  forceRerun: boolean,
  result: AutoTuneResult | null,
): AutoTuneCapabilities {
  const candidates = result?.candidates ?? [];
  const tpsValues = candidates.map((c) => c.tokens_per_second).filter((v) => Number.isFinite(v) && v > 0);
  const bestTPS = result?.best?.tokens_per_second ?? 0;
  const meanTPS = tpsValues.length ? tpsValues.reduce((s, v) => s + v, 0) / tpsValues.length : 0;
  const minTPS = tpsValues.length ? Math.min(...tpsValues) : 0;
  const maxTPS = tpsValues.length ? Math.max(...tpsValues) : 0;
  const variability = meanTPS > 0 ? (maxTPS - minTPS) / meanTPS : 0;

  return {
    brain,
    maxTokens,
    numRuns,
    forceRerun,
    result,
    runPhase: result ? 'hasResult' : 'noResult',
    bestCandidate: result?.best ?? null,
    bestTPS,
    bestTPSBucket: tpsBucket(bestTPS),
    candidateCount: candidates.length,
    variability,
    runsBucket: runsToBucket(numRuns),
    isCached: !!result?.cached,
  };
}

export type AutoTuneRiskLevel = 'safe' | 'caution' | 'danger';
export interface AutoTuneRisk {
  level: AutoTuneRiskLevel;
  reason: string;
  reasonZh: string;
}

export function assessAutoTuneConfig(caps: AutoTuneCapabilities): AutoTuneRisk {
  if (caps.numRuns < 2) {
    return {
      level: 'caution',
      reason: `numRuns=${caps.numRuns} — single run can't measure variability. Use 3+ for stable results.`,
      reasonZh: `numRuns=${caps.numRuns} — 单次跑不出方差. 用 3 次以上稳定.`,
    };
  }
  if (caps.maxTokens < 30) {
    return {
      level: 'caution',
      reason: `maxTokens=${caps.maxTokens} — too short to capture sustained-decode TPS, prefill dominates measurement.`,
      reasonZh: `maxTokens=${caps.maxTokens} 太短 — 测的是 prefill 主导, 看不到稳态 decode TPS. 至少 50.`,
    };
  }
  if (caps.runPhase === 'hasResult' && caps.variability > 0.3) {
    return {
      level: 'caution',
      reason: `Run-to-run variability ${(caps.variability * 100).toFixed(0)}% is high — thermal throttling or cache effects. Re-run with cooler thermal state.`,
      reasonZh: `运行间差异 ${(caps.variability * 100).toFixed(0)}% 偏大 — 可能热降频或 cache 抖动. 等设备冷却后重跑.`,
    };
  }
  if (caps.runPhase === 'hasResult' && caps.bestTPSBucket === 'slow') {
    return {
      level: 'caution',
      reason: `Best TPS ${caps.bestTPS.toFixed(1)} tok/s is slow — model may be too large for this device, or KV cache exhausted.`,
      reasonZh: `最佳 TPS ${caps.bestTPS.toFixed(1)} tok/s 偏慢 — 模型可能对设备过大, 或 KV cache 已满.`,
    };
  }
  return {
    level: 'safe',
    reason: caps.runPhase === 'hasResult'
      ? `Best ${caps.bestTPS.toFixed(1)} tok/s · ${caps.candidateCount} configs tried${caps.isCached ? ' (cached)' : ''}.`
      : `${caps.numRuns} runs × ${caps.maxTokens} tokens · ready to sweep.`,
    reasonZh: caps.runPhase === 'hasResult'
      ? `最佳 ${caps.bestTPS.toFixed(1)} tok/s · 试了 ${caps.candidateCount} 套${caps.isCached ? ' (来自缓存)' : ''}.`
      : `${caps.numRuns} 次 × ${caps.maxTokens} token · 可以扫描.`,
  };
}

export function buildAutoTuneContextSnippet(caps: AutoTuneCapabilities, locale: Locale): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的 Auto-Tune 参数 sweep` : `## YOUR AUTO-TUNE SWEEP`,
    locale === 'zh'
      ? `用户在自动找你 (${caps.brain?.model_name ?? '已加载的模型'}) 在本机上的最佳推理参数. 第一人称解读结果.`
      : `User is auto-finding your (${caps.brain?.model_name ?? 'loaded model'}) best inference params on this machine. Speak in first person.`,
    `- maxTokens=${caps.maxTokens} · numRuns=${caps.numRuns} (${caps.runsBucket}) · forceRerun=${caps.forceRerun}`,
    `- Phase: ${caps.runPhase}${caps.isCached ? ' (cached)' : ''}`,
  ];
  if (caps.runPhase === 'hasResult' && caps.bestCandidate) {
    lines.push(`- Best config: ${caps.bestCandidate.label ?? '(unnamed)'} · ${caps.bestTPS.toFixed(2)} tok/s (${caps.bestTPSBucket})`);
    lines.push(`- Tried ${caps.candidateCount} configs · run-to-run variability ${(caps.variability * 100).toFixed(0)}%`);
  }
  lines.push('');
  lines.push(locale === 'zh'
    ? `### 北极星 §1 主权: sweep 全本地, 不上报"哪些参数对你最快"到任何云.`
    : `### North-star §1 sovereignty: sweep is local, never reports "which params are fastest for you" to any cloud.`);
  return lines.join('\n');
}

export function buildAutoTuneAutoBrief(caps: AutoTuneCapabilities, locale: Locale): string {
  if (locale === 'zh') {
    if (caps.runPhase === 'noResult') {
      return `用户在 Auto-Tune 配 sweep (${caps.numRuns} 次 × ${caps.maxTokens} token). 用 2-3 句作为 brain 评估这套配置 (${caps.runsBucket}) 是否合理, 提示用户跑完得到啥. 第一人称.`;
    }
    return `Sweep 完成! 最佳 ${caps.bestTPS.toFixed(2)} tok/s (${caps.bestTPSBucket}), 试了 ${caps.candidateCount} 套, 抖动 ${(caps.variability * 100).toFixed(0)}%${caps.isCached ? ' (来自缓存)' : ''}. 用 2-3 句作为 brain 解读结果 + 推荐用户用最佳配置. 第一人称.`;
  }
  if (caps.runPhase === 'noResult') {
    return `User configuring sweep (${caps.numRuns} runs × ${caps.maxTokens} tokens). In 2-3 sentences as brain, assess if this is sane (${caps.runsBucket}), tell what they'll get after sweep. First person.`;
  }
  return `Sweep done! Best ${caps.bestTPS.toFixed(2)} tok/s (${caps.bestTPSBucket}) across ${caps.candidateCount} configs, variability ${(caps.variability * 100).toFixed(0)}%${caps.isCached ? ' (cached)' : ''}. In 2-3 sentences as brain, narrate result + recommend they ship the best config. First person.`;
}

export function getAutoTuneSuggestedPrompts(caps: AutoTuneCapabilities, locale: Locale): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.runPhase === 'noResult') {
      return [
        { label: '🎯 应该跑几次', prompt: `用户准备 sweep ${caps.numRuns} 次. 用 2-3 句作为 brain 解释: 为什么 1 次不够 / 3 次够不够稳 / 5 次以上有什么收益.` },
        { label: '⚖️ maxTokens 设多少', prompt: `当前 maxTokens=${caps.maxTokens}. 用 2-3 句解释: 太短 (< 30) 测的是 prefill, 太长 (> 200) 浪费时间, 推荐用什么值.` },
        { label: '⚡ 缓存什么时候用', prompt: `forceRerun ${caps.forceRerun ? '已开' : '未开'}. 用 1-2 句作为 brain 解释: 上次 sweep 结果会被 cache, 什么时候应该 force_rerun (设备热状态变了 / 装了新依赖).` },
        { label: '🌍 端侧主权', prompt: `用 2-3 句强调: sweep 数据 (我在你机器上的具体 TPS) 不上报云, 这与"submit your benchmark to leaderboard"的 trade-off.` },
      ];
    }
    return [
      { label: '🏆 用最佳配置', prompt: `最佳 ${caps.bestTPS.toFixed(2)} tok/s. 用 2-3 句作为 brain 推荐: 我在这台 Mac 上, 这个配置 (${caps.bestCandidate?.label ?? ''}) 应该作为默认还是仅这次用? 第一人称.` },
      { label: '📊 抖动诊断', prompt: caps.variability > 0.15
        ? `抖动 ${(caps.variability * 100).toFixed(0)}% 偏大. 用 2-3 句作为 brain 给诊断 (热降频 / cache 抖动 / 后台进程), 推荐怎么稳.`
        : `抖动 ${(caps.variability * 100).toFixed(0)}% 偏小. 用 1-2 句作为 brain 解释这意味着什么 (设备稳定 + 结果可信).` },
      { label: '🆚 vs 其他设备', prompt: `${caps.bestTPS.toFixed(1)} tok/s 在 M-series 上算什么档次? 与 iPhone Air / iPad M2 / M1 Max MacBook 对比, 这套配置在它们上能达到什么 TPS?` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句话强调: sweep 数据全本地, 不分享"在你这台 Mac 上的最佳 TPS = X" 到任何云端.` },
    ];
  }
  if (caps.runPhase === 'noResult') {
    return [
      { label: '🎯 How many runs', prompt: `User picked ${caps.numRuns} runs. In 2-3 sentences as brain, explain: why 1 isn't enough, is 3 enough for stability, what does 5+ buy.` },
      { label: '⚖️ Max tokens setting', prompt: `Currently maxTokens=${caps.maxTokens}. In 2-3 sentences: too short (< 30) measures prefill, too long (> 200) wastes time. Recommend a value.` },
      { label: '⚡ When to force re-run', prompt: `forceRerun ${caps.forceRerun ? 'on' : 'off'}. In 1-2 sentences as brain, explain: previous sweep cached, when to force_rerun (thermal changed / new deps).` },
      { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: sweep data (my exact TPS on your machine) doesn't go cloud — trade-off vs "submit benchmark to leaderboard".` },
    ];
  }
  return [
    { label: '🏆 Use best config', prompt: `Best ${caps.bestTPS.toFixed(2)} tok/s. In 2-3 sentences as brain, recommend: should this config (${caps.bestCandidate?.label ?? ''}) be default for this Mac, or one-off? First person.` },
    { label: '📊 Variability diagnosis', prompt: caps.variability > 0.15
      ? `Variability ${(caps.variability * 100).toFixed(0)}% is high. In 2-3 sentences as brain, diagnose (thermal / cache / bg processes), recommend stabilising.`
      : `Variability ${(caps.variability * 100).toFixed(0)}% is low. In 1-2 sentences as brain, explain what this means (stable + trustworthy).` },
    { label: '🆚 vs other devices', prompt: `${caps.bestTPS.toFixed(1)} tok/s on M-series — what tier is that? Compare iPhone Air / iPad M2 / M1 Max MacBook — what TPS would this config get on each?` },
    { label: '🌍 Edge sovereignty', prompt: `In 2-3 sentences, emphasise: sweep data is fully local, the "best TPS for you on this Mac = X" data point never leaves.` },
  ];
}
