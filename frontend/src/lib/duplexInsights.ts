// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * duplexInsights — derived 3-stack state + chat helpers for the /duplex page.
 *
 * Voice Duplex needs THREE models loaded simultaneously (LLM/VLM brain + ASR
 * ears + TTS voice). The page already has a state machine for the audio loop
 * (DuplexPanel) but no observability layer that lets the user understand
 * which slots are filled, what each model is, and why "all-local" is a
 * north-star property (data never leaves the device).
 *
 * This file:
 *  - derives 3-stack capabilities (which slots filled, who is who, total mem)
 *  - assesses config risk (missing slots / wrong-category / VLM no image)
 *  - composes the model-aware system snippet so the LLM speaks AS itself
 *    while also knowing it's part of a trio
 *  - generates 4 trio-aware suggested prompts
 *  - emits an auto brief that highlights 3-stack readiness + sovereignty
 *
 * No new backend; everything is computed from /api/model/loaded data already
 * being fetched on mount. Purely additive — does not touch DuplexPanel state.
 */
import type { ModelInfo } from '@/api/types';
import { formatParamCount, formatSize } from '@/lib/utils';

type Locale = 'en' | 'zh';

export interface DuplexCapabilities {
  llm: ModelInfo | null;
  asr: ModelInfo | null;
  tts: ModelInfo | null;
  hasLlm: boolean;
  hasAsr: boolean;
  hasTts: boolean;
  allReady: boolean;
  isVlm: boolean;
  totalMemBytes: number;
  totalParams: number;
  trio: 'complete' | 'partial' | 'empty';
  /** Slots not yet filled (subset of 'llm' | 'asr' | 'tts'). */
  missing: Array<'llm' | 'asr' | 'tts'>;
  /** Counts of each model category in the loaded pool (helps explain availability). */
  poolCounts: { llm: number; asr: number; tts: number; vlm: number };
}

/** Derive 3-stack capabilities from the loaded-models pool + per-slot ids. */
export function deriveDuplexCapabilities(
  loadedModels: ModelInfo[],
  llmId: string,
  asrId: string,
  ttsId: string,
): DuplexCapabilities {
  const llm = loadedModels.find((m) => m.model_id === llmId) || null;
  const asr = loadedModels.find((m) => m.model_id === asrId) || null;
  const tts = loadedModels.find((m) => m.model_id === ttsId) || null;
  const hasLlm = !!llm;
  const hasAsr = !!asr;
  const hasTts = !!tts;
  const filledCount = [hasLlm, hasAsr, hasTts].filter(Boolean).length;
  const trio: DuplexCapabilities['trio'] =
    filledCount === 3 ? 'complete' : filledCount === 0 ? 'empty' : 'partial';

  const missing: DuplexCapabilities['missing'] = [];
  if (!hasLlm) missing.push('llm');
  if (!hasAsr) missing.push('asr');
  if (!hasTts) missing.push('tts');

  const totalMemBytes =
    (llm?.total_size_bytes ?? 0) +
    (asr?.total_size_bytes ?? 0) +
    (tts?.total_size_bytes ?? 0);
  const totalParams =
    (llm?.total_params ?? 0) +
    (asr?.total_params ?? 0) +
    (tts?.total_params ?? 0);

  const poolCounts = {
    llm: loadedModels.filter((m) => m.model_category === 'llm').length,
    vlm: loadedModels.filter((m) => m.model_category === 'vlm').length,
    asr: loadedModels.filter((m) => m.model_category === 'stt').length,
    tts: loadedModels.filter((m) => m.model_category === 'tts').length,
  };

  return {
    llm,
    asr,
    tts,
    hasLlm,
    hasAsr,
    hasTts,
    allReady: filledCount === 3,
    isVlm: !!llm?.has_vision,
    totalMemBytes,
    totalParams,
    trio,
    missing,
    poolCounts,
  };
}

export type ConfigRiskLevel = 'safe' | 'caution' | 'danger';
export interface ConfigRisk {
  level: ConfigRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Per-(config, pool) risk assessment.
 *  - danger: a slot has a model of the WRONG category (e.g. ASR in the LLM slot)
 *  - caution: missing slots / TTS without speaker (only zero-shot voice description)
 *  - safe: trio complete, categories match, TTS has at least one speaker
 */
export function assessDuplexConfig(
  caps: DuplexCapabilities,
  speakerCount: number,
): ConfigRisk {
  // Wrong category checks (the dropdown filters by category but defensive — a
  // user could load a TTS model and the loadedModels list might contain it
  // miscategorised; we check the actual category of the selected model).
  const llmOk = caps.hasLlm && (caps.llm!.model_category === 'llm' || caps.llm!.model_category === 'vlm');
  const asrOk = caps.hasAsr && caps.asr!.model_category === 'stt';
  const ttsOk = caps.hasTts && caps.tts!.model_category === 'tts';

  if (caps.hasLlm && !llmOk) {
    return {
      level: 'danger',
      reason: `LLM slot has a "${caps.llm!.model_category}" model — must be llm or vlm.`,
      reasonZh: `LLM 槽位选了 "${caps.llm!.model_category}" 类别 — 必须是 llm 或 vlm。`,
    };
  }
  if (caps.hasAsr && !asrOk) {
    return {
      level: 'danger',
      reason: `ASR slot has a "${caps.asr!.model_category}" model — must be stt.`,
      reasonZh: `ASR 槽位选了 "${caps.asr!.model_category}" 类别 — 必须是 stt。`,
    };
  }
  if (caps.hasTts && !ttsOk) {
    return {
      level: 'danger',
      reason: `TTS slot has a "${caps.tts!.model_category}" model — must be tts.`,
      reasonZh: `TTS 槽位选了 "${caps.tts!.model_category}" 类别 — 必须是 tts。`,
    };
  }

  if (!caps.allReady) {
    const missingLabel = caps.missing.map((s) => s.toUpperCase()).join(' + ');
    return {
      level: 'caution',
      reason: `Trio incomplete — still need: ${missingLabel}.`,
      reasonZh: `三件套未齐 — 还缺: ${missingLabel}。`,
    };
  }

  if (caps.hasTts && speakerCount === 0) {
    return {
      level: 'caution',
      reason: `TTS has no preset speakers — voice will be sampled from a free-text description.`,
      reasonZh: `TTS 未提供预设音色 — 将通过自由文本描述生成声音。`,
    };
  }

  return {
    level: 'safe',
    reason: 'Trio complete, all categories match, voice ready.',
    reasonZh: '三件套就绪，类别正确，可以开始对话。',
  };
}

/** Build the per-page context snippet appended to chatPrompts.buildModelSelfSystemPrompt. */
export function buildDuplexContextSnippet(
  caps: DuplexCapabilities,
  speakerCount: number,
  locale: Locale,
): string {
  const llmName = caps.llm?.model_name ?? '(none)';
  const llmParams = caps.llm ? formatParamCount(caps.llm.total_params ?? 0) : '—';
  const asrName = caps.asr?.model_name ?? '(none)';
  const asrParams = caps.asr ? formatParamCount(caps.asr.total_params ?? 0) : '—';
  const ttsName = caps.tts?.model_name ?? '(none)';
  const ttsParams = caps.tts ? formatParamCount(caps.tts.total_params ?? 0) : '—';
  const totalMem = formatSize(caps.totalMemBytes);
  const totalParams = formatParamCount(caps.totalParams);
  const isVlm = caps.isVlm;

  const lines: string[] = [
    locale === 'zh' ? `## 你所在的 Voice Duplex 三件套` : `## YOUR VOICE DUPLEX TRIO`,
    locale === 'zh'
      ? `你 (${llmName}, ${llmParams}) 是 brain. 用户对你说话, 你用文字回答, TTS 把你的回答读出来.`
      : `You (${llmName}, ${llmParams}) are the brain. User speaks → ASR transcribes → you generate text → TTS speaks it back.`,
    `- LLM ${isVlm ? '(VLM)' : ''}: ${llmName}, ${llmParams}`,
    `- ASR (ears): ${asrName}, ${asrParams}`,
    `- TTS (voice): ${ttsName}, ${ttsParams}${speakerCount > 0 ? ` — ${speakerCount} preset speakers` : ' — zero-shot voice via instruct text'}`,
    locale === 'zh'
      ? `- 三件套总内存: ${totalMem} (${totalParams} 参数), 全部本地, 0 次云调用 (北极星 §1).`
      : `- Trio total memory: ${totalMem} (${totalParams} params); fully local, zero cloud calls (north-star §1).`,
    `- Trio status: ${caps.trio.toUpperCase()}${caps.missing.length > 0 ? ` (missing: ${caps.missing.map((s) => s.toUpperCase()).join(', ')})` : ''}`,
    isVlm
      ? (locale === 'zh'
          ? `- 你是 VLM, 用户也可以上传图片让你看 (image_b64 会一起送进来).`
          : `- You are a VLM — the user can attach an image (image_b64 piped through alongside speech).`)
      : '',
    locale === 'zh'
      ? `- 当用户问 "你是谁" / "你怎么工作", 用第一人称解释三件套链路, 引用上面具体名字 + 参数.`
      : `- When the user asks "who are you" / "how do you work", explain the trio in first person, cite the names + param counts above.`,
  ].filter(Boolean);
  return lines.join('\n');
}

/** Auto-fired brief — short, trio-aware, sovereignty-flavored. */
export function buildDuplexAutoBrief(
  caps: DuplexCapabilities,
  speakerCount: number,
  locale: Locale,
): string {
  void speakerCount;
  if (locale === 'zh') {
    if (!caps.allReady) {
      const missing = caps.missing.map((s) => s.toUpperCase()).join(' + ');
      return `用户还没把三件套配齐 (缺: ${missing}). 用 2-3 句话作为已加载的 ${caps.llm?.model_name || 'LLM'}, 解释 Voice Duplex 需要哪三个角色 + 你目前能做到哪一段, 邀请用户补齐. 第一人称, 不列项.`;
    }
    return `三件套已齐 (你=${caps.llm!.model_name} ${formatParamCount(caps.llm!.total_params ?? 0)}, ASR=${caps.asr!.model_name}, TTS=${caps.tts!.model_name}). 用 2-3 句话作为 brain 自介绍这条链路 + 强调"全部本地, 0 次云调用". 第一人称, 引用具体参数, 不列项.`;
  }
  if (!caps.allReady) {
    const missing = caps.missing.map((s) => s.toUpperCase()).join(' + ');
    return `User has not assembled the trio yet (missing: ${missing}). In 2-3 sentences, speaking as the loaded ${caps.llm?.model_name || 'LLM'}, explain the three roles Voice Duplex needs and which slot you currently fill, invite them to fill the rest. First person, no bullets.`;
  }
  return `Trio ready (you=${caps.llm!.model_name} ${formatParamCount(caps.llm!.total_params ?? 0)}, ASR=${caps.asr!.model_name}, TTS=${caps.tts!.model_name}). In 2-3 sentences, as the brain, introduce the chain + emphasise "all local, zero cloud calls". First person, cite actual params, no bullets.`;
}

/** 4 trio-aware suggested prompts — each must reference real numbers / names. */
export function getDuplexSuggestedPrompts(
  caps: DuplexCapabilities,
  speakerCount: number,
  locale: Locale,
): { label: string; prompt: string }[] {
  const llmName = caps.llm?.model_name || 'me';
  const asrName = caps.asr?.model_name || 'an ASR';
  const ttsName = caps.tts?.model_name || 'a TTS';
  const totalMem = formatSize(caps.totalMemBytes);
  const totalParams = formatParamCount(caps.totalParams);

  if (locale === 'zh') {
    if (!caps.allReady) {
      const missing = caps.missing.map((s) => s.toUpperCase()).join(' + ');
      return [
        { label: '🧩 我目前能做什么', prompt: `三件套还缺 ${missing}. 作为已加载的 ${llmName}, 用 2-3 句话告诉用户: 在缺这些角色的情况下, 我们能做的 / 不能做的 各是什么? 不要列项.` },
        { label: '🎯 配什么模型最搭我', prompt: `我是 ${llmName} (${formatParamCount(caps.llm?.total_params ?? 0)} 参数). 推荐适合搭配我的 ASR 和 TTS 各 1 个 (考虑总内存预算 + 端侧部署), 给出具体模型尺寸建议.` },
        { label: '🌍 端侧主权', prompt: `这一切都在本地跑 — 我, ASR, TTS 三个模型都不调云. 用 2-3 句话向用户说明: 为什么这条 "数据物理上不离开设备" 比云端 voice assistant 更重要 (隐私/延迟/离线).` },
        { label: '⚙️ 三件套链路', prompt: `画一遍从用户按下麦克风到听见回答的完整链路 (8-10 个步骤): 录音→VAD→ASR→LLM→TTS→播放. 每步标注谁负责, 数据怎么流转, 在哪里可能出延迟.` },
      ];
    }
    return [
      { label: '🎙️ 自我介绍', prompt: `作为 brain, 用第一人称介绍这套 Voice Duplex 三件套 (我=${llmName}, ASR=${asrName}, TTS=${ttsName}). 三件套总共 ${totalMem} / ${totalParams} 参数. 强调全部本地. 2-3 句话, 不列项.` },
      { label: '🔊 我的声音', prompt: speakerCount > 0
        ? `TTS (${ttsName}) 给我提供了 ${speakerCount} 个预设音色. 用一段话告诉用户: 这些音色一般怎么挑 (中文/英文/男声/女声/语速等), 端侧零样本 voice cloning 与预设音色各有什么权衡.`
        : `TTS (${ttsName}) 没有预设音色, 走自由文本描述生成 (instruct). 用 2-3 句话教用户怎么写好 instruct (举 1-2 个具体例子, 比如 "温柔女声, 中等语速"), 它对最终音色的影响有多大.` },
      { label: '📊 设备适配', prompt: `三件套总共 ${totalMem} 内存 / ${totalParams} 参数. 在 iPhone 15 Pro (8GB, 可用 ~6.8 GB) / iPad M2 (8GB) / M1 Max MacBook (32GB) 三档设备上, 这套配置流不流畅? 哪台是甜点?` },
      { label: '⚡ 链路延迟', prompt: `从用户按完麦克风到听见我的第一个字 (TTFT-audio), 估算理论下界: ASR 转写 + 我 prefill + TTS 第一段合成. 给个数字范围 (端侧 ms 级估算).` },
    ];
  }
  if (!caps.allReady) {
    const missing = caps.missing.map((s) => s.toUpperCase()).join(' + ');
    return [
      { label: '🧩 What I can do now', prompt: `Trio is missing ${missing}. Speaking as the loaded ${llmName}, in 2-3 sentences tell the user what we CAN and CANNOT do without those roles. No bullets.` },
      { label: '🎯 Best companions for me', prompt: `I am ${llmName} (${formatParamCount(caps.llm?.total_params ?? 0)} params). Recommend one ASR and one TTS that pair well with me, considering total memory budget for edge deployment. Give concrete size suggestions.` },
      { label: '🌍 Edge sovereignty', prompt: `Everything runs locally — me, the ASR, the TTS. Zero cloud calls. In 2-3 sentences, explain to the user why "data never physically leaves the device" matters more than a cloud voice assistant (privacy / latency / offline).` },
      { label: '⚙️ Pipeline walkthrough', prompt: `Walk through the full chain from "user presses mic" to "user hears my reply" in 8-10 steps: record → VAD → ASR → LLM → TTS → playback. Note who owns each step, where data flows, and where latency hides.` },
    ];
  }
  return [
    { label: '🎙️ Introduce yourself', prompt: `As the brain, introduce this Voice Duplex trio in first person (me=${llmName}, ASR=${asrName}, TTS=${ttsName}). Total trio = ${totalMem} / ${totalParams}. Emphasise all-local. 2-3 sentences, no bullets.` },
    { label: '🔊 My voice', prompt: speakerCount > 0
      ? `My TTS (${ttsName}) ships ${speakerCount} preset speakers. In one paragraph, tell the user how to choose between them (Chinese / English / male / female / pacing), and what trade-offs there are vs zero-shot cloning at the edge.`
      : `My TTS (${ttsName}) has no preset speakers — it uses a free-text instruct prompt. In 2-3 sentences, teach the user how to write a good instruct (1-2 concrete examples like "warm female voice, moderate pace") and how much it affects the final timbre.` },
    { label: '📊 Device fit', prompt: `Trio is ${totalMem} memory / ${totalParams} params total. Across iPhone 15 Pro (8GB → ~6.8 GB usable), iPad M2 (8GB), M1 Max MacBook (32GB), is this configuration smooth? Which is the sweet spot?` },
    { label: '⚡ Pipeline latency', prompt: `From "user finishes speaking" to "user hears my first audio chunk" (TTFT-audio), estimate the theoretical lower bound: ASR transcript + my prefill + TTS first segment synth. Give a numeric range for on-device ms.` },
  ];
}
