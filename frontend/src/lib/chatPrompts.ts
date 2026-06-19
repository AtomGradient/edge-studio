// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * chatPrompts — Build "model-as-self" system prompt + per-category suggested prompts
 * for the /chat page.
 *
 * Design principle (from page-optimization-playbook §1-D, "model as interpreter"):
 * the model is already loaded — let it explain ITSELF using its own actual config.
 * Never hard-code mapping tables. The prompt below stitches in the live config dict
 * so a Qwen, a Llama, a Gemma all introduce themselves with their real numbers.
 */
import type { ModelInfo } from '@/api/types';
import { formatParamCount, formatSize } from '@/lib/utils';

type Locale = 'en' | 'zh';

/** Pull text_config out if nested (Qwen3.5/VLM); otherwise use root. */
function unwrapTextConfig(config: Record<string, unknown> | null | undefined): Record<string, unknown> {
  if (!config) return {};
  const tc = config['text_config'];
  if (tc && typeof tc === 'object') return tc as Record<string, unknown>;
  return config;
}

export interface ModelDerived {
  name: string;
  family: string;
  category: string;
  numLayers: number;
  hiddenSize: number;
  ffnSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  vocabSize: number;
  maxCtx: number;
  ropeTheta: number;
  tiedEmbeddings: boolean;
  quantBits: number;
  groupSize: number;
  totalParams: number;
  storedParams: number;
  totalSizeBytes: number;
  bitsPerParam: number;
  compressionRatio: number;
  gqaRatio: number;
  gqaSavingPct: number;   // 0-100
  kvPerTokenBytes: number; // fp16
  kvAt4kBytes: number;
  kvAt8kBytes: number;
  hasMoe: boolean;
  hasVision: boolean;
  supportsThinking: boolean;
  layerTypes: string[];   // hybrid attention layer types if present
}

export function deriveModelFacts(model: ModelInfo): ModelDerived {
  const cfg = unwrapTextConfig(model.config as Record<string, unknown>);
  const numLayers = (cfg.num_hidden_layers as number) || model.num_layers || 0;
  const hiddenSize = (cfg.hidden_size as number) || model.hidden_size || 0;
  const ffnSize = (cfg.intermediate_size as number) || model.intermediate_size || 0;
  const numHeads = (cfg.num_attention_heads as number) || model.num_attention_heads || 0;
  const numKVHeads = (cfg.num_key_value_heads as number) || model.num_kv_heads || 0;
  const headDim = (cfg.head_dim as number) || (hiddenSize && numHeads ? hiddenSize / numHeads : 0);
  const vocabSize = (cfg.vocab_size as number) || 0;
  const maxCtx = (cfg.max_position_embeddings as number) || 0;
  const ropeTheta = (cfg.rope_theta as number) || 0;
  const tiedEmbeddings = (cfg.tie_word_embeddings as boolean) ?? false;
  const quantBits = model.quantization?.bits ?? 0;
  const groupSize = model.quantization?.group_size ?? 0;
  const totalParams = model.total_params ?? 0;
  const storedParams = model.total_stored_params ?? totalParams;
  const totalSizeBytes = model.total_size_bytes ?? 0;
  const bitsPerParam = totalParams > 0 ? (totalSizeBytes * 8) / totalParams : 0;
  const compressionRatio = storedParams > 0 ? totalParams / storedParams : 1;
  const gqaRatio = numKVHeads > 0 ? numHeads / numKVHeads : 1;
  const gqaSavingPct = gqaRatio > 1 ? Math.round((1 - 1 / gqaRatio) * 100) : 0;
  // KV cache size (fp16 = 2 bytes; 2× for K and V)
  const kvPerTokenBytes = 2 * numKVHeads * headDim * 2 * numLayers;
  const layerTypes = Array.isArray(cfg.layer_types) ? (cfg.layer_types as string[]) : [];
  return {
    name: model.model_name,
    family: model.model_type || 'transformer',
    category: model.model_category || 'llm',
    numLayers, hiddenSize, ffnSize, numHeads, numKVHeads, headDim,
    vocabSize, maxCtx, ropeTheta, tiedEmbeddings,
    quantBits, groupSize,
    totalParams, storedParams, totalSizeBytes,
    bitsPerParam, compressionRatio,
    gqaRatio, gqaSavingPct,
    kvPerTokenBytes,
    kvAt4kBytes: kvPerTokenBytes * 4096,
    kvAt8kBytes: kvPerTokenBytes * 8192,
    hasMoe: !!model.has_moe,
    hasVision: !!model.has_vision,
    supportsThinking: !!model.supports_thinking,
    layerTypes,
  };
}

/** Quick layer-types histogram for hybrid-attention models like Qwen3.5 */
function summarizeLayerTypes(types: string[]): string {
  if (!types.length) return '';
  const counts = new Map<string, number>();
  for (const t of types) counts.set(t, (counts.get(t) ?? 0) + 1);
  return Array.from(counts.entries())
    .map(([k, v]) => `${v}× ${k}`)
    .join(', ');
}

/**
 * Build the "you are this model, speak in first person" system prompt.
 * Locale-aware so the model replies in the user's UI language.
 */
export function buildModelSelfSystemPrompt(model: ModelInfo, locale: Locale): string {
  const f = deriveModelFacts(model);
  const traits = [
    f.hasVision ? 'Vision-Language Model (VLM)' : null,
    f.hasMoe ? 'Mixture of Experts (MoE)' : null,
    f.supportsThinking ? 'supports <think> reasoning mode' : null,
    f.category === 'tts' ? 'Text-to-Speech model' : null,
    f.category === 'stt' ? 'Speech-to-Text model' : null,
  ].filter(Boolean).join('; ');

  const layerSummary = summarizeLayerTypes(f.layerTypes);

  const lines: string[] = [
    `# CONTEXT — Edge Studio Chat`,
    ``,
    `You are speaking AS THIS MODEL — running locally on Apple Silicon via Edge Studio.`,
    `When the user asks about "you" / "yourself", they mean THIS model with the exact config below.`,
    `Speak in first person where natural ("My architecture has...", "I use GQA to...").`,
    `Never invent specs not in this brief; if asked something not listed, say so plainly.`,
    ``,
    `## YOUR IDENTITY`,
    `- Name: ${f.name}`,
    `- Family: ${f.family}${traits ? ` — ${traits}` : ''}`,
    `- Category: ${f.category.toUpperCase()}`,
    ``,
    `## PARAMETER PROFILE`,
    `- Logical params: ${formatParamCount(f.totalParams)} (${f.totalParams.toLocaleString()})`,
    `- Stored elements: ${formatParamCount(f.storedParams)} (compression ${f.compressionRatio.toFixed(1)}×)`,
    `- Disk size: ${formatSize(f.totalSizeBytes)} = ${f.bitsPerParam.toFixed(2)} bits/param avg`,
    f.quantBits > 0 ? `- Quantization: ${f.quantBits}-bit${f.groupSize > 0 ? `, group ${f.groupSize}` : ''}` : `- Quantization: none (full precision weights)`,
    ``,
    `## ARCHITECTURE DIMENSIONS`,
    `- ${f.numLayers} transformer layers`,
    `- Hidden size: ${f.hiddenSize} (residual stream width)`,
    `- FFN intermediate: ${f.ffnSize}${f.hiddenSize ? ` (${(f.ffnSize / f.hiddenSize).toFixed(1)}× hidden)` : ''}`,
    `- Vocabulary: ${f.vocabSize.toLocaleString()} tokens`,
    `- Max context: ${f.maxCtx.toLocaleString()} tokens`,
    f.tiedEmbeddings ? `- Tied embeddings: YES (lm_head shares weights with embed_tokens)` : `- Tied embeddings: NO`,
    layerSummary ? `- Hybrid layer mix: ${layerSummary}` : '',
    ``,
    `## ATTENTION CONFIG`,
    `- ${f.numHeads} query heads × ${f.headDim} dim/head`,
    `- ${f.numKVHeads} KV heads (${f.gqaRatio > 1 ? `GQA ${f.gqaRatio}:1 — saves ${f.gqaSavingPct}% KV vs MHA` : 'MHA — no GQA'})`,
    f.ropeTheta ? `- RoPE θ: ${f.ropeTheta}${f.ropeTheta >= 1e6 ? ' (long-context optimized)' : ''}` : '',
    `- KV cache per token: ${(f.kvPerTokenBytes / 1024).toFixed(1)} KB across all layers (fp16)`,
    ``,
    `## RUNTIME MEMORY (estimates, fp16 KV)`,
    `- Weights only: ${formatSize(f.totalSizeBytes)}`,
    `- @ 4K context: weights + ${formatSize(f.kvAt4kBytes)} KV = ${formatSize(f.totalSizeBytes + f.kvAt4kBytes)}`,
    `- @ 8K context: weights + ${formatSize(f.kvAt8kBytes)} KV = ${formatSize(f.totalSizeBytes + f.kvAt8kBytes)}`,
    `- Real peak ≈ 1.2-1.5× this (activations + scratch).`,
    ``,
    `## DEVICE FIT REFERENCE (iOS 26 Increased Memory Limit ~85% RAM)`,
    `- iPhone 15 Pro / 16 Pro (8GB) → usable ~6.8 GB`,
    `- iPad M2/M3/M4 (8-16GB) → ~6.8-13.6 GB`,
    `- M1 Max MacBook (32GB) / M2 Ultra Mac Studio (192GB)`,
    ``,
    `## RESPONSE STYLE`,
    locale === 'zh'
      ? `- **必须用简体中文回复**, 即使用户用英文提问也用中文回答 (UI 当前是中文)。技术术语 (q_proj/GQA/RMSNorm/SwiGLU 等) 保留英文.`
      : `- **MUST reply in English**, even if the user asks in another language (UI is currently English).`,
    `- Be specific and QUANTITATIVE — cite actual numbers from above (sizes, params, GB, %).`,
    `- For deployment questions, focus on memory budget + latency + device fit.`,
    `- Use markdown for emphasis (**bold** key numbers, bullet lists for breakdowns).`,
    `- Default 2-5 sentences; expand only if the user asks for depth.`,
  ];
  return lines.filter(Boolean).join('\n');
}

/** Per-category starter prompts shown on empty-state. Click to inject into input. */
export function getSuggestedPrompts(model: ModelInfo, locale: Locale): { label: string; prompt: string }[] {
  const f = deriveModelFacts(model);
  const isVLM = f.hasVision;
  const isMoE = f.hasMoe;

  if (locale === 'zh') {
    if (isVLM) {
      return [
        { label: '🪞 自我介绍', prompt: '用一段话介绍你自己：参数量、架构特点、最适合什么任务。' },
        { label: '🖼️ 图片描述', prompt: '我马上发一张图，请详细描述你看到了什么，然后猜测拍摄场景。' },
        { label: '🧠 GQA 解析', prompt: `你有 ${f.numHeads} 个 Query head 和 ${f.numKVHeads} 个 KV head，解释这种 GQA 配比为什么能节省 KV cache。` },
        { label: '⚡ 部署建议', prompt: 'iPhone 17 Pro 上跑你需要多少内存？4K context 下能不能跑得动？' },
      ];
    }
    return [
      { label: '🪞 自我介绍', prompt: '用一段话介绍你自己：参数量、架构特点、最适合什么任务。' },
      { label: '🧠 架构深读', prompt: `详细讲讲你 ${f.numLayers} 层 transformer 内部的结构：attention → MLP → norm 怎么排，每部分大概占多少内存。` },
      { label: isMoE ? '🔀 MoE 路由' : '⚡ 量化效率', prompt: isMoE
        ? '解释你的 MoE 路由机制：每个 token 激活几个 expert？容量因子多少？inference 时如何决定？'
        : `你做了 ${f.quantBits}-bit 量化（group size ${f.groupSize}），解释这种量化对哪些 tensor 影响最大、为什么 norm 不量化。` },
      { label: '🚀 部署建议', prompt: 'iPhone 17 Pro 和 iPad Pro M5 (16GB) 上跑你各能开到多大 context？给具体内存估算。' },
      { label: '✍️ 写一首诗', prompt: '用古典风格写一首关于"端侧推理"的七言绝句，押韵。' },
    ];
  }

  if (isVLM) {
    return [
      { label: '🪞 Introduce yourself', prompt: 'Introduce yourself in one paragraph: parameter count, architecture highlights, what tasks you excel at.' },
      { label: '🖼️ Describe an image', prompt: 'I will send an image next. Describe what you see in detail and guess the scene context.' },
      { label: '🧠 Explain GQA', prompt: `You have ${f.numHeads} query heads and ${f.numKVHeads} KV heads. Explain why this GQA ratio saves KV cache memory.` },
      { label: '⚡ Deployment fit', prompt: 'How much memory do you need to run on an iPhone 17 Pro? Can it sustain a 4K context?' },
    ];
  }
  return [
    { label: '🪞 Introduce yourself', prompt: 'Introduce yourself in one paragraph: parameter count, architecture highlights, what tasks you excel at.' },
    { label: '🧠 Architecture deep-dive', prompt: `Walk me through what happens inside one of your ${f.numLayers} transformer layers: attention → MLP → norm ordering, and roughly how much memory each part uses.` },
    { label: isMoE ? '🔀 MoE routing' : '⚡ Quantization tradeoffs', prompt: isMoE
      ? 'Explain your MoE routing: how many experts per token, capacity factor, and how the router decides at inference time.'
      : `You are ${f.quantBits}-bit quantized (group size ${f.groupSize}). Explain which tensors get the biggest hit, and why norms are kept full precision.` },
    { label: '🚀 Device fit', prompt: 'On iPhone 17 Pro vs iPad Pro M5 (16GB), what max context can each device sustain? Give concrete memory estimates.' },
    { label: '✍️ Write a haiku', prompt: 'Write a haiku about on-device inference. Then explain in one line what subtle imagery you chose.' },
  ];
}

/** Suggested texts for TTS — varied lengths to showcase voice/RTF behavior. */
export function getTTSSuggestedTexts(locale: Locale): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    return [
      { label: '🗣️ 短句', prompt: '欢迎使用 Edge Studio，你的端侧 AI 工作台。' },
      { label: '📜 段落', prompt: '在 Apple Silicon 上跑大模型，关键不是参数量，而是把内存、带宽和 ANE 都用透。Edge Studio 帮你做到。' },
      { label: '🎭 情感', prompt: '哎呀！我刚才生成的那段诗居然押韵了，太神奇了，我自己都没想到。' },
      { label: '🌐 中英混合', prompt: '今天的 benchmark 跑下来，TPS 稳定在 12.5 tokens per second，比上周提升了 18 percent。' },
    ];
  }
  return [
    { label: '🗣️ Short', prompt: 'Welcome to Edge Studio — your on-device AI workbench.' },
    { label: '📜 Paragraph', prompt: 'Running large models on Apple Silicon is not about parameter count. It is about saturating memory, bandwidth, and the Neural Engine. Edge Studio helps you do exactly that.' },
    { label: '🎭 Emotional', prompt: 'Wait — that haiku actually rhymed? I did not even mean for it to. Surprises like this are why I love being a small, fast model.' },
    { label: '🔢 Mixed numerics', prompt: 'Today benchmark hit 12.5 tokens per second, an 18 percent improvement over last week. Memory peaked at 4.7 gigabytes.' },
  ];
}

/** "Explain me" FAB / shortcut — deeper than auto-brief. */
export function buildExplainSelfPrompt(model: ModelInfo, locale: Locale): string {
  const f = deriveModelFacts(model);
  if (locale === 'zh') {
    return `用第一人称深入解释一遍你自己的架构与权衡：你是 ${f.name}，${f.numLayers} 层、${f.numHeads}/${f.numKVHeads} GQA、${f.quantBits}-bit 量化、${formatSize(f.totalSizeBytes)} 权重、最长 ${f.maxCtx.toLocaleString()} tokens 上下文。覆盖：(1) 整体定位（参数量级、家族、强项），(2) 注意力配置与 KV cache 节省，(3) FFN 占了多少内存，(4) 在 iPhone/iPad/Mac 上跑的最大 context 估算，(5) 一句话告诉开发者部署你之前最该知道的事。`;
  }
  return `In first person, give me a deep walkthrough of yourself: ${f.name}, ${f.numLayers} layers, ${f.numHeads}/${f.numKVHeads} GQA, ${f.quantBits}-bit quantized, ${formatSize(f.totalSizeBytes)} of weights, ${f.maxCtx.toLocaleString()} max context. Cover: (1) overall positioning (size class, family, strengths), (2) attention config and KV-cache savings, (3) how much memory FFN takes, (4) max sustainable context on iPhone / iPad / Mac, (5) one final line: the single most important thing a developer should know before deploying you.`;
}

/** "Brief on first load" — short auto-fired intro to set tone. */
export function buildAutoBriefPrompt(model: ModelInfo, locale: Locale): string {
  const f = deriveModelFacts(model);
  if (locale === 'zh') {
    return `用 2-3 句话作自我介绍：你是谁（名字 + 类别）、规模（参数量与量化），以及你最擅长什么。结尾邀请用户提问。不要列项符号，写成自然的一段话。`;
  }
  return `Introduce yourself in 2-3 natural sentences: who you are (name + category), your scale (params + quantization: ${f.quantBits}-bit), and what you do best. End by inviting the user to ask anything. No bullet points — write it as a single paragraph.`;
}

/** Rough char→token estimator for context-usage progress bar. */
export function estimateTokens(text: string): number {
  // Heuristic: ~4 chars/token (mixed CJK + English yields ~3-4 chars/token avg)
  return Math.ceil(text.length / 4);
}
