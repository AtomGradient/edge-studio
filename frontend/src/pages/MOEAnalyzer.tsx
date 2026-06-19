// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import Plot from 'react-plotly.js';
import { Network, Sparkles, Send, X, Cpu, Flame, Snowflake, Shield } from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { analyzeMOE } from '@/api/endpoints';
import type { MOEAnalysisResponse } from '@/api/types';
import { PageHeader } from '@/components/layout/PageHeader';
import { InsightPanel } from '@/components/common/InsightPanel';
import { IdentityCard } from '@/components/common/IdentityCard';
import { ModelBriefCard } from '@/components/common/ModelBriefCard';
import { useMOEInsights } from '@/hooks/useModelInsights';
import { useModelChat } from '@/hooks/useModelChat';
import { useT, useLocaleStore } from '@/i18n';
import { MetricCards } from '@/components/data/MetricCards';
import { EmptyState } from '@/components/common/EmptyState';
import MarkdownContent from '@/components/MarkdownContent';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import {
  deriveMoeCapabilities,
  assessMoe,
  buildMoeContextSnippet,
  buildMoeAutoBrief,
  getMoeSuggestedPrompts,
  shapeLabel,
  type LayerPattern,
} from '@/lib/moeInsights';

const PATTERN_COLOR: Record<LayerPattern, string> = {
  balanced: '#22c55e',
  mild_skew: '#6366f1',
  monopolized: '#ef4444',
  hot_cold_split: '#f59e0b',
  sparse_sample: '#94a3b8',
};

export default function MOEAnalyzer() {
  const model = useModelStore((s) => s.currentModel);
  const hasTrace = useModelStore((s) => s.hasTrace);
  const t = useT();
  const locale = useLocaleStore((s) => s.locale) as 'en' | 'zh';
  const moeInsights = useMOEInsights(t, model);

  const [result, setResult] = useState<MOEAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── Hooks BEFORE any early return (§10.2) ────────────────────────────
  const brain = model;
  const moeCaps = useMemo(
    () => deriveMoeCapabilities(result, loading, hasTrace, brain),
    [result, loading, hasTrace, brain],
  );
  const moeRisk = useMemo(() => assessMoe(moeCaps), [moeCaps]);

  const moeSystemPrompt = useMemo(() => {
    if (!brain) return '';
    return buildModelSelfSystemPrompt(brain, locale) + '\n\n' + buildMoeContextSnippet(moeCaps, locale);
  }, [brain, moeCaps, locale]);

  const briefChat = useModelChat({
    modelId: brain?.model_id ?? null,
    systemPrompt: moeSystemPrompt,
    maxTokens: 700,
    temperature: 0.6,
  });

  const moePrompts = useMemo(
    () => getMoeSuggestedPrompts(moeCaps, locale),
    [moeCaps, locale],
  );

  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerInput, setDrawerInput] = useState('');
  const briefFiredForRef = useRef<string | null>(null);

  // ── AI Brief auto-fire on phase / shape / risk change ────────────────
  useEffect(() => {
    if (!brain) return;
    if (briefChat.streaming) return;
    const sig = `${brain.model_id}:${moeCaps.runPhase}:${moeCaps.shape}:${moeCaps.coldRatio.toFixed(2)}:${moeRisk.level}:${locale}`;
    if (briefFiredForRef.current === sig) return;
    briefFiredForRef.current = sig;
    const id = window.setTimeout(() => briefChat.send(buildMoeAutoBrief(moeCaps, locale)), 400);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id, moeCaps.runPhase, moeCaps.shape, moeCaps.coldRatio, moeRisk.level, locale]);

  // ── Auto-analyze when trace + MoE present and we don't have result yet ─
  const handleAnalyze = useCallback(async () => {
    if (!brain) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeMOE(brain.model_id);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }, [brain]);

  useEffect(() => {
    if (moeCaps.runPhase === 'idle' && !result && !loading) {
      void handleAnalyze();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moeCaps.runPhase]);

  const handleSendDrawer = useCallback(() => {
    const q = drawerInput.trim();
    if (!q || briefChat.streaming) return;
    briefChat.send(q);
    setDrawerInput('');
  }, [briefChat, drawerInput]);

  // Reset state when model switches
  useEffect(() => {
    setResult(null);
    setError(null);
    briefFiredForRef.current = null;
    setDrawerOpen(false);
    briefChat.reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [brain?.model_id]);

  if (!model) {
    return <EmptyState title="No Model" description="Load a model to analyze MOE experts" />;
  }

  // §11.1 — DO NOT early-return on notMoe/noTrace; render inline so brain
  // can narrate gating state.

  const RISK_BANNER_CLASS: Record<typeof moeRisk.level, string> = {
    safe: 'border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300',
    caution: 'border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300',
    danger: 'border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300',
  };

  const showResult = moeCaps.hasResult && !loading;

  return (
    <div>
      <PageHeader
        title="MOE Analyzer"
        description="Expert utilization analysis for Mixture-of-Experts models"
      />

      <InsightPanel insights={moeInsights} />

      {/* §9.1 4-card identity strip — Scale / Routing / Cold / Diversity */}
      <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <IdentityCard
          icon={<Cpu size={16} />}
          label={t('moe.cardScale')}
          value={
            moeCaps.runPhase === 'notMoe'
              ? (locale === 'zh' ? '非 MoE' : 'not MoE')
              : `${moeCaps.numExperts}×top-${moeCaps.topK}`
          }
          hint={
            moeCaps.runPhase === 'notMoe'
              ? (locale === 'zh' ? '当前模型是 dense FFN' : 'current model is dense FFN')
              : `${moeCaps.numLayers} ${locale === 'zh' ? '层 · 共' : 'layers · '}${moeCaps.totalSlots.toLocaleString()} ${locale === 'zh' ? 'expert slots' : 'expert slots'}`
          }
          tone={moeCaps.runPhase === 'notMoe' ? 'neutral' : 'indigo'}
        />
        <IdentityCard
          icon={<Network size={16} />}
          label={t('moe.cardRouting')}
          value={
            moeCaps.runPhase === 'noTrace'
              ? (locale === 'zh' ? '未 trace' : 'no trace')
              : showResult
                ? moeCaps.meanLoadBalance.toFixed(3)
                : moeCaps.runPhase === 'analyzing'
                  ? '…'
                  : '—'
          }
          hint={
            moeCaps.runPhase === 'noTrace'
              ? t('moe.routingNeedTrace')
              : showResult
                ? `${moeCaps.totalTokens} tokens · ${shapeLabel(moeCaps.shape, locale)}`
                : t('moe.routingHint')
          }
          tone={
            moeCaps.runPhase === 'noTrace' ? 'red'
            : !showResult ? 'neutral'
            : moeCaps.meanLoadBalance >= 0.7 ? 'emerald'
            : moeCaps.meanLoadBalance >= 0.4 ? 'amber' : 'red'
          }
        />
        <IdentityCard
          icon={<Snowflake size={16} />}
          label={t('moe.cardCold')}
          value={
            showResult
              ? `${moeCaps.coldExpertsGlobal.length}/${moeCaps.numExperts}`
              : '—'
          }
          hint={
            showResult
              ? moeCaps.coldExpertsGlobal.length === 0
                ? (locale === 'zh' ? '所有 expert 都被路由' : 'every expert hit')
                : `${(moeCaps.coldRatio * 100).toFixed(1)}% ${locale === 'zh' ? '全局零路由' : 'with zero routings'}`
              : t('moe.coldHint')
          }
          tone={
            !showResult ? 'neutral'
            : moeCaps.coldRatio === 0 ? 'emerald'
            : moeCaps.coldRatio < 0.1 ? 'indigo'
            : moeCaps.coldRatio < 0.3 ? 'amber' : 'red'
          }
        />
        <IdentityCard
          icon={<Flame size={16} />}
          label={t('moe.cardDiversity')}
          value={
            showResult
              ? `${moeCaps.layerDominanceDiversity.toFixed(2)} bits`
              : '—'
          }
          hint={
            showResult
              ? moeCaps.alwaysHotExperts.length > 0
                ? `${moeCaps.alwaysHotExperts.length} ${locale === 'zh' ? 'always-hot · top-1 全局 ' : 'always-hot · global top-1 '}E${moeCaps.topExpertId ?? '?'} (${(moeCaps.topExpertShare * 100).toFixed(1)}%)`
                : (locale === 'zh' ? '不同层有不同主导专家' : 'different experts dominate different layers')
              : t('moe.diversityHint')
          }
          tone={
            !showResult ? 'neutral'
            : moeCaps.alwaysHotExperts.length > 5 ? 'amber'
            : moeCaps.layerDominanceDiversity > 3 ? 'emerald' : 'indigo'
          }
        />
      </div>

      {moeRisk.level !== 'safe' && (
        <div className={`mb-4 rounded-lg border px-3 py-2 text-xs ${RISK_BANNER_CLASS[moeRisk.level]}`}>
          <span className="font-semibold uppercase tracking-wider">
            {moeRisk.level === 'danger' ? t('moe.riskDanger') : t('moe.riskCaution')}
          </span>
          {' '}— {locale === 'zh' ? moeRisk.reasonZh : moeRisk.reason}
        </div>
      )}

      {/* §10 AI Brief — auto-fired, brain narrates per phase */}
      {brain && (
        <ModelBriefCard
          className="mb-6"
          label={t('moe.briefTitle')}
          text={briefChat.text}
          streaming={briefChat.streaming}
          emptyText={t('moe.briefEmpty')}
          streamingText={t('moe.briefThinking')}
          refreshTitle={t('moe.briefRefire')}
          prompts={moePrompts}
          onRefresh={() => { briefFiredForRef.current = null; briefChat.reset(); briefChat.send(buildMoeAutoBrief(moeCaps, locale)); }}
          onPrompt={(prompt) => { briefChat.reset(); briefChat.send(prompt); setDrawerOpen(true); }}
        />
      )}

      {/* Inline gating EmptyStates — render only when no data is meaningful */}
      {moeCaps.runPhase === 'notMoe' && (
        <EmptyState
          title={t('moe.notMoeTitle')}
          description={t('moe.notMoeDesc')}
        />
      )}
      {moeCaps.runPhase === 'noTrace' && (
        <EmptyState
          title={t('moe.noTraceTitle')}
          description={t('moe.noTraceDesc')}
        />
      )}

      {/* Re-analyze button surfaces only when we have a trace */}
      {moeCaps.hasMoeArch && hasTrace && (
        <div className="mb-6 flex items-center gap-3">
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
          >
            {loading ? t('moe.analyzing') : showResult ? t('moe.reanalyze') : t('moe.analyze')}
          </button>
          {error && <p className="text-sm text-red-500">{error}</p>}
          {showResult && (
            <p className="text-xs text-stone-500 dark:text-stone-400">
              {locale === 'zh'
                ? `${moeCaps.totalTokens} tokens · ${moeCaps.numLayers}×${moeCaps.numExperts} 网格 · top-${moeCaps.topK}`
                : `${moeCaps.totalTokens} tokens · ${moeCaps.numLayers}×${moeCaps.numExperts} grid · top-${moeCaps.topK}`}
            </p>
          )}
        </div>
      )}

      {showResult && result && (
        <>
          {/* Summary metrics */}
          <MetricCards
            metrics={[
              { label: 'Experts', value: result.num_experts },
              { label: 'Top-K', value: result.top_k },
              { label: 'Avg Load Balance', value: result.avg_load_balance.toFixed(3) },
              {
                label: 'Cold Experts',
                value: result.cold_expert_count,
                color: result.cold_expert_count > 0 ? '#ef4444' : undefined,
              },
            ]}
            className="mb-6"
          />

          {/* Utilization Heatmap */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-200">
              {t('moe.heatmapTitle')}
            </h3>
            <p className="mb-3 text-xs text-gray-400 dark:text-stone-500">
              {t('moe.heatmapHint')}
            </p>
            <UtilizationHeatmap data={result} />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Load Balance per Layer — color encodes pattern */}
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-200">
                {t('moe.loadBalanceTitle')}
              </h3>
              <p className="mb-2 text-[11px] text-gray-400 dark:text-stone-500">
                {t('moe.loadBalanceHint')}
              </p>
              <LoadBalanceBar data={result} patterns={moeCaps.layerPatterns} locale={locale} />
            </div>

            {/* Global Expert Token Counts */}
            <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
              <h3 className="mb-2 text-sm font-semibold text-gray-700 dark:text-stone-200">
                {t('moe.globalTitle')}
              </h3>
              <p className="mb-2 text-[11px] text-gray-400 dark:text-stone-500">
                {locale === 'zh'
                  ? `Hot ${moeCaps.hotExperts.length} · Warm ${moeCaps.warmExperts.length} · Tepid ${moeCaps.tepidExperts.length} · Cold ${moeCaps.coldExpertsGlobal.length}`
                  : `Hot ${moeCaps.hotExperts.length} · Warm ${moeCaps.warmExperts.length} · Tepid ${moeCaps.tepidExperts.length} · Cold ${moeCaps.coldExpertsGlobal.length}`}
              </p>
              <GlobalTokenBar data={result} caps={moeCaps} />
            </div>
          </div>

          { }
          {moeCaps.alwaysHotExperts.length > 0 && (
            <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 p-4 dark:border-amber-900/40 dark:bg-amber-950/30">
              <h3 className="mb-1 text-sm font-semibold text-amber-700 dark:text-amber-300">
                {t('moe.alwaysHotTitle')} ({moeCaps.alwaysHotExperts.length})
              </h3>
              <p className="mb-2 text-xs text-amber-600 dark:text-amber-400">
                {locale === 'zh'
                  ? `这些 expert 在 ≥${Math.ceil(moeCaps.numLayers * 0.25)} 个层都是 top-1 — 是 dense promote 候选 (热门到值得直接换成 dense FFN).`
                  : `These experts are top-1 in ≥${Math.ceil(moeCaps.numLayers * 0.25)} layers — candidates to promote into dense FFN (hot enough to skip routing overhead).`}
              </p>
              <div className="flex flex-wrap gap-2">
                {moeCaps.alwaysHotExperts.map((e) => (
                  <span
                    key={e}
                    className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-200"
                  >
                    E{e}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Cold Expert Alert */}
          {result.cold_experts.length > 0 && (
            <div className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 dark:border-red-900/40 dark:bg-red-950/30">
              <h3 className="mb-2 text-sm font-semibold text-red-700 dark:text-red-300">
                {t('moe.coldAlertTitle')} ({result.cold_experts.length})
              </h3>
              <p className="mb-3 text-xs text-red-500 dark:text-red-400">
                {t('moe.coldAlertHint')}
              </p>
              <div className="flex flex-wrap gap-2">
                {result.cold_experts.slice(0, 200).map((ce, i) => (
                  <span
                    key={i}
                    className="rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-700 dark:bg-red-900/40 dark:text-red-200"
                  >
                    L{ce.layer} E{ce.expert}
                  </span>
                ))}
                {result.cold_experts.length > 200 && (
                  <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-700 dark:bg-red-900/40 dark:text-red-200">
                    +{result.cold_experts.length - 200} more
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Layer Detail Table — pattern-tagged per row */}
          <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
            <h3 className="mb-3 text-sm font-semibold text-gray-700 dark:text-stone-200">
              {t('moe.layerDetailTitle')}
            </h3>
            <div className="max-h-[400px] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-50 dark:bg-stone-800">
                  <tr className="text-left text-gray-500 dark:text-stone-400">
                    <th className="px-2 py-1.5 font-medium">{t('moe.colLayer')}</th>
                    <th className="px-2 py-1.5 font-medium">{t('moe.colPattern')}</th>
                    <th className="px-2 py-1.5 font-medium text-right">{t('moe.colBalance')}</th>
                    <th className="px-2 py-1.5 font-medium text-right">{t('moe.colEntropy')}</th>
                    <th className="px-2 py-1.5 font-medium text-right">{t('moe.colCold')}</th>
                    <th className="px-2 py-1.5 font-medium">{t('moe.colCounts')}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.layer_stats.map((ls, i) => {
                    const pattern = moeCaps.layerPatterns[i] ?? 'mild_skew';
                    const entropyNorm = moeCaps.layerEntropies[i] ?? 0;
                    const max = Math.max(...ls.expert_counts, 1);
                    return (
                      <tr
                        key={ls.layer_idx}
                        className="border-t border-gray-50 hover:bg-gray-50 dark:border-stone-800 dark:hover:bg-stone-800/50"
                      >
                        <td className="px-2 py-1.5 font-mono text-gray-700 dark:text-stone-300">L{ls.layer_idx}</td>
                        <td className="px-2 py-1.5">
                          <span
                            className="rounded px-1.5 py-0.5 text-[10px] font-medium uppercase"
                            style={{
                              backgroundColor: `${PATTERN_COLOR[pattern]}20`,
                              color: PATTERN_COLOR[pattern],
                            }}
                          >
                            {pattern.replace('_', ' ')}
                          </span>
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono text-gray-700 dark:text-stone-300">
                          <span
                            className={
                              ls.load_balance < 0.5
                                ? 'text-red-600 font-medium dark:text-red-400'
                                : ls.load_balance < 0.7
                                  ? 'text-amber-600 dark:text-amber-400'
                                  : ''
                            }
                          >
                            {ls.load_balance.toFixed(3)}
                          </span>
                        </td>
                        <td className="px-2 py-1.5 text-right font-mono text-gray-700 dark:text-stone-300">
                          {entropyNorm.toFixed(2)}
                        </td>
                        <td className="px-2 py-1.5 text-right">
                          {ls.cold_experts.length > 0 ? (
                            <span className="font-medium text-red-600 dark:text-red-400">
                              {ls.cold_experts.length}
                            </span>
                          ) : (
                            <span className="text-gray-400 dark:text-stone-600">0</span>
                          )}
                        </td>
                        <td className="px-2 py-1.5">
                          <div className="flex gap-0.5">
                            {ls.expert_counts.map((c, ei) => (
                              <div
                                key={ei}
                                className="h-4 flex-1 rounded-sm"
                                style={{
                                  backgroundColor: c === 0 ? '#fecaca' : '#6366f1',
                                  opacity: c === 0 ? 1 : Math.max(0.2, c / max),
                                }}
                                title={`E${ei}: ${c} tokens`}
                              />
                            ))}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Ask Model FAB drawer — must be inside main fn scope (§10.5) */}
      {!drawerOpen && brain && (
        <button
          onClick={() => setDrawerOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-500 px-4 py-3 text-sm font-medium text-white shadow-lg hover:bg-indigo-600"
          title={t('moe.askModel')}
        >
          <Sparkles size={16} /> {t('moe.askModel')}
        </button>
      )}
      {drawerOpen && (
        <div className="fixed bottom-0 right-0 z-40 flex h-[70vh] w-full max-w-md flex-col rounded-tl-2xl border-l border-t border-gray-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-950">
          <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3 dark:border-stone-800">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-indigo-500 dark:text-indigo-400" />
              <span className="text-sm font-semibold text-gray-700 dark:text-stone-300">{t('moe.askModel')}</span>
            </div>
            <button onClick={() => setDrawerOpen(false)} className="rounded p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-stone-800">
              <X size={14} />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 text-sm">
            {briefChat.text ? <MarkdownContent content={briefChat.text} /> : (
              <p className="text-xs text-gray-400 dark:text-stone-500">{t('moe.askEmpty')}</p>
            )}
          </div>
          {moePrompts.length > 0 && (
            <div className="flex flex-wrap gap-1.5 border-t border-gray-100 px-3 py-2 dark:border-stone-800">
              {moePrompts.slice(0, 3).map((p) => (
                <button
                  key={p.label}
                  onClick={() => briefChat.send(p.prompt)}
                  disabled={briefChat.streaming}
                  className="rounded-full border border-indigo-200 px-2 py-0.5 text-[10px] text-indigo-700 hover:bg-indigo-50 dark:border-indigo-500/30 dark:text-indigo-300 dark:hover:bg-indigo-500/10 disabled:opacity-50"
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
          <form
            onSubmit={(e) => { e.preventDefault(); handleSendDrawer(); }}
            className="flex gap-2 border-t border-gray-100 p-3 dark:border-stone-800"
          >
            <input
              value={drawerInput}
              onChange={(e) => setDrawerInput(e.target.value)}
              placeholder={t('moe.askPlaceholder')}
              disabled={briefChat.streaming}
              className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs focus:border-indigo-400 focus:outline-none dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200"
            />
            <button type="submit" disabled={!drawerInput.trim() || briefChat.streaming} className="rounded-lg bg-indigo-500 px-3 py-1.5 text-white hover:bg-indigo-600 disabled:opacity-50">
              <Send size={12} />
            </button>
          </form>
        </div>
      )}

      {/* Sovereignty footer — single mention to keep visual quiet */}
      <div className="mt-4 flex items-center justify-end gap-1.5 text-[10px] text-stone-400 dark:text-stone-600">
        <Shield size={11} />
        {t('moe.sovereignty')}
      </div>
    </div>
  );
}

// ── Charts ────────────────────────────────────────────────────────────────

function UtilizationHeatmap({ data }: { data: MOEAnalysisResponse }) {
  const numLayers = data.layer_stats.length;
  const numExperts = data.num_experts;

  return (
    <Plot
      data={[
        {
          z: data.utilization_matrix,
          x: Array.from({ length: numExperts }, (_, i) => `E${i}`),
          y: data.layer_stats.map((ls) => `L${ls.layer_idx}`),
          type: 'heatmap',
          colorscale: 'YlOrRd',
          hovertemplate: 'L%{y} E%{x}<br>Tokens: %{z}<extra></extra>',
        },
      ]}
      layout={{
        xaxis: { title: { text: 'Expert' } },
        yaxis: { title: { text: 'Layer' }, autorange: 'reversed' },
        height: Math.max(300, numLayers * 20 + 80),
        margin: { t: 10, l: 60, r: 20, b: 50 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function LoadBalanceBar({
  data,
  patterns,
  locale,
}: {
  data: MOEAnalysisResponse;
  patterns: LayerPattern[];
  locale: 'en' | 'zh';
}) {
  const labels = data.layer_stats.map((ls) => `L${ls.layer_idx}`);
  const values = data.layer_stats.map((ls) => ls.load_balance);
  const colors = patterns.map((p) => PATTERN_COLOR[p]);
  const customdata = patterns.map((p) => p.replace('_', ' '));
  const yMax = Math.max(...values, 1.1) * 1.05;

  return (
    <Plot
      data={[
        {
          x: labels,
          y: values,
          type: 'bar',
          marker: { color: colors },
          customdata,
          hovertemplate: '%{x}: %{y:.3f}<br>%{customdata}<extra></extra>',
        },
      ]}
      layout={{
        yaxis: {
          title: { text: locale === 'zh' ? '负载均衡' : 'Load Balance' },
          range: [0, yMax],
        },
        height: 280,
        margin: { t: 10, l: 50, r: 20, b: 40 },
        shapes: [
          {
            type: 'line',
            x0: -0.5,
            x1: labels.length - 0.5,
            y0: 1.0,
            y1: 1.0,
            line: { dash: 'dash', color: '#9ca3af', width: 1 },
          },
        ],
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}

function GlobalTokenBar({
  data,
  caps,
}: {
  data: MOEAnalysisResponse;
  caps: ReturnType<typeof deriveMoeCapabilities>;
}) {
  const labels = data.global_token_counts.map((_, i) => `E${i}`);
  const hot = new Set(caps.hotExperts);
  const warm = new Set(caps.warmExperts);
  const cold = new Set(caps.coldExpertsGlobal);
  const colors = data.global_token_counts.map((c, i) => {
    if (c === 0 || cold.has(i)) return '#ef4444';
    if (hot.has(i)) return '#f59e0b';
    if (warm.has(i)) return '#6366f1';
    return '#94a3b8';
  });

  return (
    <Plot
      data={[
        {
          x: labels,
          y: data.global_token_counts,
          type: 'bar',
          marker: { color: colors },
          hovertemplate: '%{x}: %{y} tokens<extra></extra>',
        },
      ]}
      layout={{
        yaxis: { title: { text: 'Token Count' } },
        height: 280,
        margin: { t: 10, l: 60, r: 20, b: 40 },
      }}
      config={{ responsive: true }}
      style={{ width: '100%' }}
    />
  );
}
