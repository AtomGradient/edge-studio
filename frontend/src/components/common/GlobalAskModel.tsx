// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { RotateCcw, Send, Sparkles, Square, X } from 'lucide-react';
import MarkdownContent from '@/components/MarkdownContent';
import { AskModelFab } from '@/components/common/AskModelFab';
import { useModelChat } from '@/hooks/useModelChat';
import { buildModelSelfSystemPrompt } from '@/lib/chatPrompts';
import { cn } from '@/lib/utils';
import { useLocaleStore, useT } from '@/i18n';
import { useModelStore } from '@/stores/modelStore';
import { usePageAskStore } from '@/stores/pageAskStore';

const ROUTES_WITH_PAGE_ASK = new Set([
  '/architecture',
  '/weights',
  '/activation',
  '/pruning',
  '/inference',
  '/attention',
  '/quality',
  '/kv-cache',
  '/optimization',
  '/auto-optimizer',
  '/pipeline',
  '/moe',
  '/comparison',
  '/distill',
  '/merge',
  '/auto-tune',
  '/mixed-precision',
  '/benchmark-dashboard',
  '/batch',
  '/devices',
  '/duplex',
  '/chat',
  '/neural-imprint-chat',
]);

export function GlobalAskModel() {
  const t = useT();
  const { locale } = useLocaleStore();
  const location = useLocation();
  const model = useModelStore((s) => s.currentModel);
  const pageDataContext = usePageAskStore((s) => s.contexts[location.pathname]);
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState('');

  const isCoveredByPage = ROUTES_WITH_PAGE_ASK.has(location.pathname);
  const canAsk = !!model && (model.model_category === 'llm' || model.model_category === 'vlm');
  const pageContext = useMemo(
    () => makePageContext(location.pathname, t),
    [location.pathname, t],
  );
  const pageGlossary = useMemo(
    () => makePageGlossary(location.pathname),
    [location.pathname],
  );
  const systemPrompt = useMemo(() => {
    if (!model) return '';
    return [
      buildModelSelfSystemPrompt(model, locale),
      '',
      '## CURRENT EDGE STUDIO PAGE',
      `- Route: ${location.pathname}`,
      `- Page: ${pageContext}`,
      ...(pageGlossary ? [`- Feature description: ${pageGlossary}`] : []),
      ...(pageDataContext ? ['', '## CURRENT PAGE DATA SUMMARY', pageDataContext] : []),
      '- Use this page context only as UI context. Do not infer hidden data not shown to you.',
      '- If the user asks about page data you cannot see, explain what the page is for and what they should inspect.',
    ].join('\n');
  }, [locale, location.pathname, model, pageContext, pageGlossary, pageDataContext]);

  const chat = useModelChat({
    modelId: canAsk ? model.model_id : null,
    systemPrompt,
    maxTokens: 768,
    temperature: 0.4,
  });

  if (!canAsk || isCoveredByPage) return null;

  const send = () => {
    const trimmed = input.trim();
    if (!trimmed || chat.streaming) return;
    chat.send(trimmed);
    setInput('');
    setOpen(true);
  };

  return (
    <>
      {!open && (
        <AskModelFab
          label={t('globalAsk.fab')}
          modelName={model.model_name}
          onClick={() => setOpen(true)}
          icon={<Sparkles size={15} />}
        />
      )}

      {open && (
        <div className="fixed bottom-6 right-6 z-50 flex w-[min(440px,calc(100vw-2rem))] max-h-[72vh] flex-col rounded-xl border border-stone-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-950">
          <div className="flex items-center justify-between border-b border-stone-200 px-4 py-3 dark:border-stone-800">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-sm font-semibold text-stone-900 dark:text-stone-100">
                <Sparkles size={15} />
                {t('globalAsk.title')}
              </div>
              <div className="mt-0.5 truncate text-[11px] text-stone-500">
                {pageContext} · {model.model_name}
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={chat.reset}
                className="rounded-md p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800"
                title={t('globalAsk.newConversation')}
              >
                <RotateCcw size={14} />
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800"
                title={t('common.close')}
              >
                <X size={14} />
              </button>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
            {!chat.text && !chat.streaming ? (
              <div className="rounded-lg border border-dashed border-stone-200 px-3 py-4 text-sm text-stone-500 dark:border-stone-800">
                {t('globalAsk.empty')}
              </div>
            ) : (
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <MarkdownContent content={chat.text} />
              </div>
            )}
            {chat.status && (
              <div className="mt-3 text-xs text-stone-500">
                {chat.status}
              </div>
            )}
          </div>

          <form
            className="border-t border-stone-200 p-3 dark:border-stone-800"
            onSubmit={(event) => {
              event.preventDefault();
              send();
            }}
          >
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                rows={2}
                placeholder={t('globalAsk.placeholder')}
                disabled={chat.streaming}
                className="min-h-[44px] flex-1 resize-none rounded-lg border border-stone-200 bg-white px-3 py-2 text-sm text-stone-900 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:opacity-60 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-indigo-950"
              />
              <button
                type="button"
                onClick={chat.streaming ? chat.cancel : send}
                disabled={!chat.streaming && !input.trim()}
                className={cn(
                  'flex h-10 w-10 items-center justify-center rounded-lg text-white disabled:opacity-50',
                  chat.streaming ? 'bg-red-600 hover:bg-red-700' : 'bg-stone-900 hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200',
                )}
                title={chat.streaming ? t('globalAsk.stop') : t('globalAsk.send')}
              >
                {chat.streaming ? <Square size={16} /> : <Send size={16} />}
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}

function makePageContext(pathname: string, t: (key: string) => string): string {
  const labels: Record<string, string> = {
    '/devices': t('nav.devices'),
    '/neural-imprint': t('nav.neuralImprint'),
    '/neural-imprint-chat': t('nav.neuralImprintChat'),
    '/rpp-results': t('nav.rppResults'),
    '/a-library': t('nav.aLibrary'),
    '/joint-inference': t('nav.jointInference'),
    '/dashboard': t('nav.dashboard'),
    '/export': t('nav.export'),
  };
  const matched = matchRoute(pathname, labels);
  return matched ?? (pathname.replace(/^\//, '') || t('nav.dashboard'));
}

function makePageGlossary(pathname: string): string {
  const glossary: Record<string, string> = {
    '/neural-imprint':
      'Neural Imprint is Edge Studio personalization based on a persisted full KV-cache prefill saved as a safetensors artifact. It represents the model-conditioned profile, tool protocol, and preferences for a specific user. Restoring it at inference time gives the model that personalized state without replaying long context. This page inspects artifact metadata, hashes, token counts, compatibility, and lifecycle state.',
    '/rpp-results':
      'In Edge Studio, RPP means on-device user profile extraction through Imprint Distillation; it does not mean requests-per-second. RPP processes real user records through the model, captures hidden states, then extracts stable user-specific direction vectors with bootstrap stability checks. This page shows direction labels, evidence records, profile narrative, and stability metrics.',
    '/a-library':
      'A-library is a model-specific library of precomputed activation direction vectors used by RPP / Imprint Distillation. Orthogonality means directions should not be too correlated; max_abs_cos_sim should stay below 0.4 and mean_abs_cos_sim below 0.15. Signal strength measures whether each direction separates positive and negative examples; values above 1.0 are usable, and higher median signal is better. A pass layer is suitable for RPP user-profile extraction; a fail layer has weak or correlated directions and should not be selected. target_layer is the hidden-state layer used by RPP. sweep means scanning multiple candidate layers to pick the best validated layer.',
    '/joint-inference':
      'Joint Inference lets an edge device route selected inference requests to a more capable Mac host running Edge Studio. The host can run the base model, optionally with a matching Neural Imprint, then stream the result back while preserving an edge-app style experience. This page shows request history, route status, latency, terminal state, token usage, and Neural Imprint participation.',
    '/dashboard':
      'The Edge Studio dashboard summarizes the loaded model, optimization readiness, device and runtime state, recent activity, and shortcuts into training, export, and analysis workflows.',
    '/export':
      'Export packages model configuration and Edge ecosystem capabilities into a developer-facing iOS app scaffold. It helps produce a buildable app template with selected model settings, sample data, Neural Imprint or A-library assets when applicable, and integration guidance.',
    '/devices':
      'Devices is the EdgeMesh device management page. It shows local-network peers discovered through EdgeMesh, trust and pairing status, device capabilities, learning snapshots, and whether a device can participate in Neural Imprint sync or Joint Inference.',
    '/simple':
      'The simplified workflow guides developers through device assessment, model selection, optimization, and export with fewer advanced controls. It is a guided path over the same Edge Studio capabilities.',
  };
  return matchRoute(pathname, glossary) ?? '';
}

function matchRoute(pathname: string, values: Record<string, string>): string | undefined {
  if (values[pathname]) return values[pathname];

  const matchedKey = Object.keys(values)
    .filter((key) => pathname === key || pathname.startsWith(`${key}/`))
    .sort((a, b) => b.length - a.length)[0];

  return matchedKey ? values[matchedKey] : undefined;
}
