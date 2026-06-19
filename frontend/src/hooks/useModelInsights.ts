// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * useModelInsights — derive actionable insights from model metadata.
 */

import { useMemo } from 'react';
import type { ModelInfo } from '@/api/types';
import type { Insight } from '@/components/common/InsightPanel';

// ---- Helpers ----

function sizeGB(bytes: number): string {
  return (bytes / (1024 ** 3)).toFixed(1);
}

function paramsB(n: number): string {
  return (n / 1e9).toFixed(1);
}

type T = (key: string) => string;

// ---- Architecture ----

export function useArchitectureInsights(t: T, model: ModelInfo | null, hasProfile: boolean): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];
    const pb = model.total_params / 1e9;

    if (pb < 3) {
      insights.push({ title: t('insight.arch.compact.title').replace('{params}', paramsB(model.total_params)), description: t('insight.arch.compact.desc'), severity: 'good' });
    } else if (pb < 14) {
      insights.push({ title: t('insight.arch.midsize.title').replace('{params}', paramsB(model.total_params)), description: t('insight.arch.midsize.desc'), severity: 'info' });
    } else {
      insights.push({ title: t('insight.arch.large.title').replace('{params}', paramsB(model.total_params)), description: t('insight.arch.large.desc'), severity: 'warning', action: { label: t('insight.action.optimize'), path: '/pipeline' } });
    }

    const qBits = model.quantization?.bits;
    if (qBits && qBits <= 4) {
      insights.push({ title: t('insight.arch.quantized.title').replace('{bits}', String(qBits)), description: t('insight.arch.quantized.desc').replace('{size}', sizeGB(model.total_size_bytes)), severity: 'good' });
    } else if (qBits && qBits > 4) {
      insights.push({ title: t('insight.arch.highbit.title').replace('{bits}', String(qBits)), description: t('insight.arch.highbit.desc'), severity: 'tip', action: { label: t('insight.action.quantize'), path: '/pipeline' } });
    } else if (!model.is_gguf) {
      insights.push({ title: t('insight.arch.fullprec.title'), description: t('insight.arch.fullprec.desc').replace('{size}', sizeGB(model.total_size_bytes)), severity: 'warning', action: { label: t('insight.action.quantize'), path: '/pipeline' } });
    }

    if (model.num_kv_heads > 0 && model.num_kv_heads < model.num_attention_heads) {
      insights.push({ title: t('insight.arch.gqa.title').replace('{ratio}', String(model.num_attention_heads / model.num_kv_heads)), description: t('insight.arch.gqa.desc'), severity: 'good' });
    }

    if (!hasProfile && !model.is_gguf) {
      insights.push({ title: t('insight.arch.noProfile.title'), description: t('insight.arch.noProfile.desc'), severity: 'tip', action: { label: t('insight.action.generate'), path: '/activation' } });
    }

    if (model.has_moe) {
      insights.push({ title: t('insight.arch.moe.title'), description: t('insight.arch.moe.desc'), severity: 'info', action: { label: t('insight.action.moeAnalysis'), path: '/moe' } });
    }

    return insights;
  }, [t, model, hasProfile]);
}

// ---- Weights ----

export interface WeightStats {
  totalTensors: number;
  quantizedCount: number;
  totalParams: number;
  totalSizeBytes: number;
  dtypeBreakdown?: Array<{ dtype: string; count: number; params: number; size: number }>;
}

export function useWeightInsights(t: T, model: ModelInfo | null, stats: WeightStats | null): Insight[] {
  return useMemo(() => {
    if (!model || !stats) return [];
    const insights: Insight[] = [];

    if (stats.quantizedCount > 0 && stats.totalTensors > 0) {
      const pct = (stats.quantizedCount / stats.totalTensors * 100).toFixed(0);
      if (Number(pct) > 80) {
        insights.push({ title: t('insight.weight.mostQuantized.title').replace('{pct}', pct), description: t('insight.weight.mostQuantized.desc'), severity: 'good' });
      } else if (Number(pct) > 30) {
        insights.push({ title: t('insight.weight.partialQuantized.title').replace('{pct}', pct), description: t('insight.weight.partialQuantized.desc'), severity: 'tip', action: { label: t('insight.action.quantize'), path: '/pipeline' } });
      }
    } else if (stats.quantizedCount === 0 && !model.is_gguf) {
      insights.push({ title: t('insight.weight.noQuantized.title'), description: t('insight.weight.noQuantized.desc'), severity: 'warning', action: { label: t('insight.action.quantize'), path: '/pipeline' } });
    }

    const vocabSize = (model.config?.vocab_size as number) || 0;
    if (vocabSize > 100000) {
      insights.push({ title: t('insight.weight.largeVocab.title').replace('{size}', (vocabSize / 1000).toFixed(0)), description: t('insight.weight.largeVocab.desc'), severity: 'tip', action: { label: t('insight.action.vocabPrune'), path: '/pipeline' } });
    }

    const sg = stats.totalSizeBytes / (1024 ** 3);
    if (sg > 10) {
      insights.push({ title: t('insight.weight.tooLarge.title').replace('{size}', sg.toFixed(1)), description: t('insight.weight.tooLarge.desc'), severity: 'warning', action: { label: t('insight.action.deviceFit'), path: '/kv-cache' } });
    } else if (sg <= 4) {
      insights.push({ title: t('insight.weight.fitsMobile.title').replace('{size}', sg.toFixed(1)), description: t('insight.weight.fitsMobile.desc'), severity: 'good', action: { label: t('insight.action.export'), path: '/export' } });
    }

    return insights;
  }, [t, model, stats]);
}

// ---- Activation Heatmap ----

export function useActivationInsights(t: T, model: ModelInfo | null, hasProfile: boolean, deadRatio?: number): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (!hasProfile) {
      insights.push({ title: t('insight.activation.needProfile.title'), description: t('insight.activation.needProfile.desc'), severity: 'tip', action: { label: t('insight.action.runInference'), path: '/inference' } });
      return insights;
    }

    if (deadRatio !== undefined) {
      if (deadRatio > 0.3) {
        insights.push({ title: t('insight.activation.highDead.title').replace('{pct}', (deadRatio * 100).toFixed(0)), description: t('insight.activation.highDead.desc'), severity: 'warning', action: { label: t('insight.action.prune'), path: '/pruning' } });
      } else if (deadRatio > 0.1) {
        insights.push({ title: t('insight.activation.medDead.title').replace('{pct}', (deadRatio * 100).toFixed(0)), description: t('insight.activation.medDead.desc'), severity: 'tip', action: { label: t('insight.action.simulate'), path: '/pruning' } });
      } else {
        insights.push({ title: t('insight.activation.lowDead.title').replace('{pct}', (deadRatio * 100).toFixed(1)), description: t('insight.activation.lowDead.desc'), severity: 'good' });
      }
    }

    insights.push({ title: t('insight.activation.guide.title'), description: t('insight.activation.guide.desc'), severity: 'info' });

    return insights;
  }, [t, model, hasProfile, deadRatio]);
}

// ---- Pruning Simulator ----

export function usePruningInsights(t: T, model: ModelInfo | null, hasProfile: boolean): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (!hasProfile) {
      insights.push({ title: t('insight.pruning.needProfile.title'), description: t('insight.pruning.needProfile.desc'), severity: 'warning', action: { label: t('insight.action.loadProfile'), path: '/activation' } });
      return insights;
    }

    insights.push({ title: t('insight.pruning.howItWorks.title'), description: t('insight.pruning.howItWorks.desc'), severity: 'info' });
    insights.push({ title: t('insight.pruning.protectLayers.title'), description: t('insight.pruning.protectLayers.desc'), severity: 'tip' });
    insights.push({ title: t('insight.pruning.validate.title'), description: t('insight.pruning.validate.desc'), severity: 'tip', action: { label: t('insight.action.qualityCheck'), path: '/quality' } });

    return insights;
  }, [t, model, hasProfile]);
}

// ---- Inference Tracer ----

export function useInferenceInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (model.is_gguf) {
      insights.push({ title: t('insight.inference.gguf.title'), description: t('insight.inference.gguf.desc'), severity: 'warning' });
      return insights;
    }

    insights.push({ title: t('insight.inference.whatYouSee.title'), description: t('insight.inference.whatYouSee.desc'), severity: 'info' });

    if (model.supports_thinking) {
      insights.push({ title: t('insight.inference.thinking.title'), description: t('insight.inference.thinking.desc'), severity: 'tip' });
    }

    return insights;
  }, [t, model]);
}

// ---- Chat ----

export function useChatInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    const sg = model.total_size_bytes / (1024 ** 3);
    if (sg > 8) {
      insights.push({ title: t('insight.chat.largeModel.title'), description: t('insight.chat.largeModel.desc').replace('{size}', sg.toFixed(1)), severity: 'info', action: { label: t('insight.action.autoTune'), path: '/auto-tune' } });
    }

    if (model.has_vision) {
      insights.push({ title: t('insight.chat.vision.title'), description: t('insight.chat.vision.desc'), severity: 'tip' });
    }

    if (model.model_category === 'tts') {
      insights.push({ title: t('insight.chat.tts.title'), description: t('insight.chat.tts.desc'), severity: 'info' });
    } else if (model.model_category === 'stt') {
      insights.push({ title: t('insight.chat.stt.title'), description: t('insight.chat.stt.desc'), severity: 'info' });
    }

    return insights;
  }, [t, model]);
}

// ---- Attention Patterns ----

export function useAttentionInsights(t: T, model: ModelInfo | null, hasAttentionTrace: boolean): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (!hasAttentionTrace) {
      insights.push({ title: t('insight.attention.needTrace.title'), description: t('insight.attention.needTrace.desc'), severity: 'tip', action: { label: t('insight.action.inference'), path: '/inference' } });
      return insights;
    }

    insights.push({ title: t('insight.attention.patterns.title'), description: t('insight.attention.patterns.desc'), severity: 'info' });
    insights.push({ title: t('insight.attention.optHint.title'), description: t('insight.attention.optHint.desc'), severity: 'tip' });

    return insights;
  }, [t, model, hasAttentionTrace]);
}

// ---- Quality Validator ----

export function useQualityInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    insights.push({ title: t('insight.quality.whatPpl.title'), description: t('insight.quality.whatPpl.desc'), severity: 'info' });

    const qBits = model.quantization?.bits;
    if (qBits && qBits <= 3) {
      insights.push({ title: t('insight.quality.lowBit.title'), description: t('insight.quality.lowBit.desc').replace('{bits}', String(qBits)), severity: 'warning' });
    }

    insights.push({ title: t('insight.quality.quickVsFull.title'), description: t('insight.quality.quickVsFull.desc'), severity: 'tip' });

    return insights;
  }, [t, model]);
}

// ---- KV Cache ----

export function useKVCacheInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (model.num_kv_heads > 0 && model.num_kv_heads < model.num_attention_heads) {
      insights.push({ title: t('insight.kvCache.gqa.title'), description: t('insight.kvCache.gqa.desc').replace('{kvHeads}', String(model.num_kv_heads)).replace('{attnHeads}', String(model.num_attention_heads)), severity: 'good' });
    } else {
      insights.push({ title: t('insight.kvCache.mha.title'), description: t('insight.kvCache.mha.desc'), severity: 'info' });
    }

    insights.push({ title: t('insight.kvCache.chart.title'), description: t('insight.kvCache.chart.desc'), severity: 'info' });

    return insights;
  }, [t, model]);
}

// ---- Optimization Advisor ----

export function useOptAdvisorInsights(t: T, model: ModelInfo | null, hasProfile: boolean): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (!hasProfile) {
      insights.push({ title: t('insight.optAdvisor.needProfile.title'), description: t('insight.optAdvisor.needProfile.desc'), severity: 'tip', action: { label: t('insight.action.generateProfile'), path: '/activation' } });
    }

    insights.push({ title: t('insight.optAdvisor.priority.title'), description: t('insight.optAdvisor.priority.desc'), severity: 'info' });

    return insights;
  }, [t, model, hasProfile]);
}

// ---- Auto Optimizer ----

export function useAutoOptInsights(t: T, model: ModelInfo | null, hasProfile: boolean): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    if (!hasProfile) {
      insights.push({ title: t('insight.autoOpt.needProfile.title'), description: t('insight.autoOpt.needProfile.desc'), severity: 'warning', action: { label: t('insight.action.generateProfile'), path: '/activation' } });
      return insights;
    }

    insights.push({ title: t('insight.autoOpt.howItWorks.title'), description: t('insight.autoOpt.howItWorks.desc'), severity: 'info' });
    insights.push({ title: t('insight.autoOpt.targetDevice.title'), description: t('insight.autoOpt.targetDevice.desc'), severity: 'tip' });

    return insights;
  }, [t, model, hasProfile]);
}

// ---- Pipeline ----

export function usePipelineInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    insights.push({ title: t('insight.pipeline.order.title'), description: t('insight.pipeline.order.desc'), severity: 'tip' });

    if (model.is_gguf) {
      insights.push({ title: t('insight.pipeline.gguf.title'), description: t('insight.pipeline.gguf.desc'), severity: 'warning' });
    }

    return insights;
  }, [t, model]);
}

// ---- MOE ----

export function useMOEInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    return [
      { title: t('insight.moe.utilization.title'), description: t('insight.moe.utilization.desc'), severity: 'info' as const },
      { title: t('insight.moe.coldExperts.title'), description: t('insight.moe.coldExperts.desc'), severity: 'tip' as const, action: { label: t('insight.action.pipeline'), path: '/pipeline' } },
    ];
  }, [t, model]);
}

// ---- Model Comparison ----

export function useComparisonInsights(t: T): Insight[] {
  return useMemo(() => [
    { title: t('insight.comparison.beforeAfter.title'), description: t('insight.comparison.beforeAfter.desc'), severity: 'info' as const },
    { title: t('insight.comparison.latency.title'), description: t('insight.comparison.latency.desc'), severity: 'tip' as const, action: { label: t('insight.action.inference'), path: '/inference' } },
  ], [t]);
}

// ---- Export ----

export function useExportInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];
    const sg = model.total_size_bytes / (1024 ** 3);

    if (sg > 4) {
      insights.push({ title: t('insight.export.checkDevice.title').replace('{size}', sg.toFixed(1)), description: t('insight.export.checkDevice.desc'), severity: 'warning' });
    }

    if (model.model_category === 'vlm') {
      insights.push({ title: t('insight.export.vision.title'), description: t('insight.export.vision.desc'), severity: 'info' });
    }

    return insights;
  }, [t, model]);
}

// ---- Distillation ----

export function useDistillInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    return [
      { title: t('insight.distill.what.title'), description: t('insight.distill.what.desc'), severity: 'info' as const },
      { title: t('insight.distill.bestPractice.title'), description: t('insight.distill.bestPractice.desc'), severity: 'tip' as const },
      { title: t('insight.distill.alternatives.title'), description: t('insight.distill.alternatives.desc'), severity: 'tip' as const, action: { label: t('insight.action.pipeline'), path: '/pipeline' } },
    ];
  }, [t, model]);
}

// ---- Merge ----

export function useMergeInsights(t: T): Insight[] {
  return useMemo(() => [
    { title: t('insight.merge.when.title'), description: t('insight.merge.when.desc'), severity: 'info' as const },
    { title: t('insight.merge.strategy.title'), description: t('insight.merge.strategy.desc'), severity: 'tip' as const },
    { title: t('insight.merge.validate.title'), description: t('insight.merge.validate.desc'), severity: 'tip' as const, action: { label: t('insight.action.qualityCheck'), path: '/quality' } },
  ], [t]);
}

// ---- Auto-Tune ----

export function useAutoTuneInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    const insights: Insight[] = [];

    insights.push({ title: t('insight.autoTune.what.title'), description: t('insight.autoTune.what.desc'), severity: 'info' });

    const sg = model.total_size_bytes / (1024 ** 3);
    if (sg > 8) {
      insights.push({ title: t('insight.autoTune.largeModel.title'), description: t('insight.autoTune.largeModel.desc'), severity: 'info' });
    }

    insights.push({ title: t('insight.autoTune.cached.title'), description: t('insight.autoTune.cached.desc'), severity: 'good' });

    return insights;
  }, [t, model]);
}

// ---- Mixed Precision ----

export function useMixedPrecisionInsights(t: T, model: ModelInfo | null): Insight[] {
  return useMemo(() => {
    if (!model) return [];
    return [
      { title: t('insight.mixedPrec.selective.title'), description: t('insight.mixedPrec.selective.desc'), severity: 'info' as const },
      { title: t('insight.mixedPrec.recommended.title'), description: t('insight.mixedPrec.recommended.desc'), severity: 'tip' as const },
    ];
  }, [t, model]);
}

// ---- Benchmark Dashboard ----

export function useBenchmarkInsights(t: T): Insight[] {
  return useMemo(() => [
    { title: t('insight.benchmark.compare.title'), description: t('insight.benchmark.compare.desc'), severity: 'info' as const },
    { title: t('insight.benchmark.export.title'), description: t('insight.benchmark.export.desc'), severity: 'tip' as const },
  ], [t]);
}

// ---- Batch Operations ----

export function useBatchInsights(t: T): Insight[] {
  return useMemo(() => [
    { title: t('insight.batch.workflow.title'), description: t('insight.batch.workflow.desc'), severity: 'info' as const },
    { title: t('insight.batch.saved.title'), description: t('insight.batch.saved.desc'), severity: 'good' as const },
  ], [t]);
}
