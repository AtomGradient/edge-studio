// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * simpleWizardInsights — derived wizard state + chat helpers for the
 * /simple/* routes (beginner mode, "2-clicks + auto-download" UX).
 *
 * Shared by the 6+ Simple-mode pages (DeviceProfile / FocusSelect / TierSelect
 * / Setup / Complete / ExportDevice / ExportGenerate). Lives in SimpleShell
 * so all wizard pages get the same Identity Strip + AI Brief + Ask FAB
 * without per-page duplication (§9.1 multi-component capability — here the
 * "components" are wizard steps + selections + setup/export phases).
 *
 * Pure functions; no fetching. Page passes already-fetched simpleStore
 * snapshot in.
 *
 * Sovereignty (§9.2 mandatory): Simple mode is the easiest entry point
 * users see. The whole app's "0 cloud" promise must be visible from
 * the very first page — emerald sovereignty card on every wizard step.
 */
import type { ModelInfo } from '@/api/types';
import type {
  DeviceProfile,
  Package,
  SetupInfo,
  ExportCheck,
} from '@/stores/simpleStore';

type Locale = 'en' | 'zh';

// ── Step routing ────────────────────────────────────────────────────────

export type WizardStepKey =
  | 'profile'        // /simple
  | 'focus'          // /simple/focus
  | 'tier'           // /simple/tier
  | 'setup'          // /simple/setup
  | 'done'           // /simple/done
  | 'exportDevice'   // /simple/export/device
  | 'exportGenerate';// /simple/export/generate

export interface WizardStepDef {
  key: WizardStepKey;
  route: string;
  /** Phase in the simple-mode lifecycle. */
  phase: 'phase1' | 'phase2';
  /** 0-based index for progress bar. */
  index: number;
}

export const WIZARD_STEP_LIST: WizardStepDef[] = [
  { key: 'profile',        route: '/simple',                  phase: 'phase1', index: 0 },
  { key: 'focus',          route: '/simple/focus',            phase: 'phase1', index: 1 },
  { key: 'tier',           route: '/simple/tier',             phase: 'phase1', index: 2 },
  { key: 'setup',          route: '/simple/setup',            phase: 'phase1', index: 3 },
  { key: 'done',           route: '/simple/done',             phase: 'phase1', index: 4 },
  { key: 'exportDevice',   route: '/simple/export/device',    phase: 'phase2', index: 5 },
  { key: 'exportGenerate', route: '/simple/export/generate',  phase: 'phase2', index: 6 },
];

export const TOTAL_STEPS = WIZARD_STEP_LIST.length;

export function routeToStep(pathname: string): WizardStepDef {
  // Match the most specific (longest) prefix first.
  const sorted = [...WIZARD_STEP_LIST].sort((a, b) => b.route.length - a.route.length);
  for (const s of sorted) {
    if (pathname === s.route || pathname.startsWith(s.route + '/')) return s;
  }
  // Default to first step.
  return WIZARD_STEP_LIST[0];
}

// ── Capability shape ────────────────────────────────────────────────────

export interface SimpleStoreSnapshot {
  deviceProfile: DeviceProfile | null;
  focus: string;
  tier: string;
  packages: Package[];
  setupInfo: SetupInfo | null;
  setupPhase: 'idle' | 'downloading' | 'loading' | 'ready';
  loadedModelId: string;
  loadedModelDir: string;
  chatTested: boolean;
  ttsModelId: string;
  duplexLlmModelId: string;
  duplexAsrModelId: string;
  duplexDownloadStep: number;
  targetDevice: string;
  exportCheck: ExportCheck | null;
  exportTaskId: string;
  exportPhase: 'idle' | 'checking' | 'adapting' | 'exporting' | 'done';
  appName: string;
  downloadUrl: string;
}

export interface WizardCapabilities {
  /** Current step (derived from current URL pathname). */
  step: WizardStepDef;
  /** Total steps. */
  totalSteps: number;
  /** 0..1 progress (current step index / total). */
  progress: number;

  // ── Phase-1 derived ───────────────────────────────────────────────────
  focus: string;
  tier: string;
  hasDevice: boolean;
  hasFocus: boolean;
  hasTier: boolean;
  isDownloading: boolean;
  isReady: boolean;            // setupPhase === 'ready'
  hasModel: boolean;           // loadedModelId truthy
  isDuplex: boolean;           // focus === 'voice_duplex'

  // ── Phase-2 derived ───────────────────────────────────────────────────
  hasTargetDevice: boolean;
  exportFits: boolean;         // exportCheck.fits
  exportCheck: ExportCheck | null;
  isExporting: boolean;
  exportDone: boolean;

  // ── Combined ──────────────────────────────────────────────────────────
  /** Best brain to address user as. */
  brain: ModelInfo | null;
  /** Loaded model name (preference: brain.model_name → setupInfo → empty). */
  loadedName: string;
  /** Star count from device profile. */
  deviceStars: number;
}

export function deriveSimpleCapabilities(
  store: SimpleStoreSnapshot,
  pathname: string,
  brain: ModelInfo | null,
): WizardCapabilities {
  const step = routeToStep(pathname);
  const hasDevice = !!store.deviceProfile;
  const hasFocus = !!store.focus;
  const hasTier = !!store.tier;
  const isDownloading = store.setupPhase === 'downloading' || store.setupPhase === 'loading';
  const isReady = store.setupPhase === 'ready';
  const hasModel = !!store.loadedModelId || !!store.duplexLlmModelId;
  const isDuplex = store.focus === 'voice_duplex';

  const hasTargetDevice = !!store.targetDevice;
  const exportFits = !!store.exportCheck?.fits;
  const isExporting = store.exportPhase === 'checking' || store.exportPhase === 'adapting' || store.exportPhase === 'exporting';
  const exportDone = store.exportPhase === 'done';

  const loadedName =
    brain?.model_name ||
    store.setupInfo?.model_display_name ||
    '';
  const deviceStars = store.deviceProfile?.ai_rating_stars ?? 0;

  return {
    step,
    totalSteps: TOTAL_STEPS,
    progress: TOTAL_STEPS > 0 ? (step.index + 1) / TOTAL_STEPS : 0,
    focus: store.focus,
    tier: store.tier,
    hasDevice,
    hasFocus,
    hasTier,
    isDownloading,
    isReady,
    hasModel,
    isDuplex,
    hasTargetDevice,
    exportFits,
    exportCheck: store.exportCheck,
    isExporting,
    exportDone,
    brain,
    loadedName,
    deviceStars,
  };
}

export type WizardRiskLevel = 'safe' | 'caution' | 'danger';
export interface WizardRisk {
  level: WizardRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Per-(wizard state) gentle guidance.
 *  - caution: missing device profile (rare, fetched on mount but could fail)
 *  - caution: at focus/tier step but the previous selection is still empty
 *  - danger:  export check failed (model doesn't fit target device)
 *  - safe:    on-track for current step
 */
export function assessWizardProgress(caps: WizardCapabilities): WizardRisk {
  if (caps.step.key === 'profile' && !caps.hasDevice) {
    return {
      level: 'caution',
      reason: 'Device profile not yet detected — please wait or click refresh.',
      reasonZh: '设备信息还没探测到 — 稍等或点刷新.',
    };
  }
  if (caps.step.key === 'focus' && !caps.hasDevice) {
    return {
      level: 'caution',
      reason: 'Skipped device profile — go back so we can size the model right.',
      reasonZh: '跳过了设备探测 — 回上一步, 让我们能算出适合的模型.',
    };
  }
  if (caps.step.key === 'tier' && !caps.hasFocus) {
    return {
      level: 'caution',
      reason: 'Pick a focus first — chat / coding / vision / asr / tts / voice duplex.',
      reasonZh: '先选一个 focus — 聊天 / 编码 / 视觉 / 语音识别 / 语音合成 / 语音对话.',
    };
  }
  if (caps.step.key === 'setup' && !caps.hasTier) {
    return {
      level: 'caution',
      reason: 'Pick a tier (air / standard / pro / max / ultra) before downloading.',
      reasonZh: '先选 tier (air / standard / pro / max / ultra) 再下载.',
    };
  }
  if (caps.step.key === 'exportGenerate' && caps.hasTargetDevice && !caps.exportFits && caps.exportCheck) {
    return {
      level: 'danger',
      reason: 'Selected model does not fit target device — pick a smaller tier or a smaller focus.',
      reasonZh: '所选模型不适合目标设备 — 选小一档的 tier 或换 focus.',
    };
  }
  return {
    level: 'safe',
    reason: `On track — step ${caps.step.index + 1} of ${caps.totalSteps}.`,
    reasonZh: `进度正常 — 第 ${caps.step.index + 1} 步, 共 ${caps.totalSteps} 步.`,
  };
}

// ── Pretty labels ───────────────────────────────────────────────────────

export function focusLabel(focus: string, locale: Locale): string {
  if (locale === 'zh') {
    return ({
      chat: '聊天',
      coding: '编码',
      vision: '视觉',
      asr: '语音识别',
      tts: '语音合成',
      voice_duplex: '语音对话',
    } as Record<string, string>)[focus] || focus || '未选';
  }
  return focus || 'not chosen';
}

export function tierLabel(tier: string, locale: Locale): string {
  if (!tier) return locale === 'zh' ? '未选' : 'not chosen';
  return tier;
}

export function stepLabel(key: WizardStepKey, locale: Locale): string {
  if (locale === 'zh') {
    return ({
      profile: '设备探测',
      focus: '选用途',
      tier: '选规格',
      setup: '下载安装',
      done: '完成',
      exportDevice: '选导出设备',
      exportGenerate: '生成 App',
    } as Record<WizardStepKey, string>)[key];
  }
  return ({
    profile: 'Detect device',
    focus: 'Pick focus',
    tier: 'Pick tier',
    setup: 'Set up',
    done: 'Done',
    exportDevice: 'Pick target',
    exportGenerate: 'Generate app',
  } as Record<WizardStepKey, string>)[key];
}

// ── Chat helpers ────────────────────────────────────────────────────────

export function buildSimpleWizardContextSnippet(
  caps: WizardCapabilities,
  store: SimpleStoreSnapshot,
  locale: Locale,
): string {
  const lines: string[] = [
    locale === 'zh' ? `## 你所在的 Simple Mode 向导` : `## YOUR SIMPLE MODE WIZARD`,
    locale === 'zh'
      ? `用户进了 Simple Mode (小白模式) — 用最少操作把 LLM 装到设备上 + 导出 App. 你 (${caps.loadedName || '本机加载的 LLM'}) 是 brain, 用最简单的话陪伴用户走完 7 步.`
      : `User entered Simple Mode (beginner) — minimum clicks to install an LLM and export an app. You (${caps.loadedName || 'the loaded LLM'}) are the brain; speak in simple terms while walking the 7 steps.`,
    `- Current step: ${caps.step.index + 1} / ${caps.totalSteps} — ${stepLabel(caps.step.key, locale)} (route: ${caps.step.route})`,
    `- Phase: ${caps.step.phase}`,
    `- Device: ${caps.hasDevice ? `${store.deviceProfile?.chip} · ${store.deviceProfile?.ram_gb}GB · rating ${store.deviceProfile?.ai_rating} (${caps.deviceStars}/5 stars)` : '(not detected)'}`,
    `- Focus: ${focusLabel(store.focus, locale)}`,
    `- Tier: ${tierLabel(store.tier, locale)}`,
    `- Setup phase: ${store.setupPhase}${store.setupInfo ? ` · ${store.setupInfo.model_display_name} (${store.setupInfo.size_gb.toFixed(1)} GB)` : ''}`,
    `- Loaded model: ${caps.hasModel ? caps.loadedName : '(none yet)'}`,
    `- Duplex mode: ${caps.isDuplex ? 'YES (LLM + ASR + TTS)' : 'NO (single model)'}`,
    `- Target device: ${caps.hasTargetDevice ? store.targetDevice : '(not picked)'}`,
    `- Export phase: ${store.exportPhase}${store.appName ? ` · app="${store.appName}"` : ''}`,
    ``,
    locale === 'zh'
      ? `### 北极星 §1 主权 (向导从头到尾必须明示):`
      : `### North-star §1 sovereignty (must be visible throughout the wizard):`,
    locale === 'zh'
      ? `- 用户输入、上传文件、模型推理和导出产物不离开本机. 设备探测也只读取本机硬件.`
      : `- User prompts, uploaded files, inference, and export artifacts do not leave the Mac. Device detection only reads local hardware.`,
    locale === 'zh'
      ? `- 刷新公开模型目录和下载权重会访问模型仓库; 下载后推理是 100% 本地, 0 次云推理调用.`
      : `- Public catalog refresh and weight download can contact model hubs; after download, inference is 100% local with zero cloud inference calls.`,
    ``,
    locale === 'zh'
      ? `### 当用户问 "我现在该做什么":`
      : `### When user asks "what should I do now":`,
    locale === 'zh'
      ? `- 用最简单的话讲 1-2 句, 引用上面具体数字 (设备 / focus / tier / phase).`
      : `- Speak in 1-2 simple sentences, cite the specifics above (device / focus / tier / phase).`,
  ];
  return lines.join('\n');
}

export function buildSimpleWizardAutoBrief(
  caps: WizardCapabilities,
  locale: Locale,
): string {
  const stepName = stepLabel(caps.step.key, locale);
  if (locale === 'zh') {
    if (caps.step.key === 'profile') {
      if (!caps.hasDevice) return `你正在 Simple Mode 第 1 步: ${stepName}. 用 1-2 句话告诉用户: 我正在自动探测你的 Mac (CPU / RAM / GPU 核数), 探测完会给你一个 1-5 星的 AI 评级. 第一人称, 简单亲切.`;
      return `第 1 步完成 (${caps.deviceStars}/5 星). 用 1-2 句话用最简单的话夸一下设备 + 邀请用户进下一步选 focus.`;
    }
    if (caps.step.key === 'focus') return `第 2 步: 选 focus (聊天/编码/视觉/ASR/TTS/语音对话). 用 1-2 句作为 brain 帮新手区分: 哪种最适合"日常聊天" vs "代码助手" vs "看图片". 简单不要列项.`;
    if (caps.step.key === 'tier') return `第 3 步: 选 tier. 用户已选 focus=${caps.hasFocus ? focusLabel(caps.focus, 'zh') : '(还没选)'}. 用 1-2 句解释 tier 是"模型大小档位" — air 最小最快, ultra 最强最大. 推荐基于设备评级 (${caps.deviceStars}/5 星) 用哪一档.`;
    if (caps.step.key === 'setup') return `第 4 步: 下载安装. 用 1-2 句作为 brain 告诉用户接下来会发生什么 (从 mlx-community/HuggingFace 拉公开权重 → 加载到内存 → ready), 明确: 下载会联网, 但用户输入和之后的推理不离开本机.`;
    if (caps.step.key === 'done') return `第 5 步: 完成! 模型已就绪 (${caps.loadedName}). 用 1-2 句话欢迎用户开始用 + 提示下一步可以测试聊天 or 直接导出 App.`;
    if (caps.step.key === 'exportDevice') return `第 6 步: 选导出目标设备. 用 1-2 句话简单说: 这是要把 ${caps.loadedName || '我'} 打包成一个能在 iPhone/iPad 上跑的 App, 你选目标设备我会自动适配模型大小.`;
    if (caps.step.key === 'exportGenerate') return `第 7 步: 生成 App. 用 1-2 句解释正在做的事 (检查 fit / 适配 → 生成 Xcode 工程 ZIP), app 完全本地, 用户拿到 ZIP 自己打开 Xcode 编译就完事了.`;
    return `Simple Mode 第 ${caps.step.index + 1} 步: ${stepName}. 用 1-2 句话以新手友好语气解释这步在做什么.`;
  }
  // English
  if (caps.step.key === 'profile') {
    if (!caps.hasDevice) return `Simple Mode step 1: ${stepName}. In 1-2 sentences, tell the user: I'm auto-detecting your Mac (CPU / RAM / GPU cores) and will give a 1-5 star AI rating. First person, simple + friendly.`;
    return `Step 1 done (${caps.deviceStars}/5 stars). In 1-2 simple sentences, comment on the device + invite them into step 2 to pick focus.`;
  }
  if (caps.step.key === 'focus') return `Step 2: pick focus (chat / coding / vision / ASR / TTS / voice duplex). In 1-2 sentences, help a newcomer distinguish "daily chat" vs "code helper" vs "image understanding". Simple, no bullets.`;
  if (caps.step.key === 'tier') return `Step 3: pick tier. User already picked focus=${caps.hasFocus ? focusLabel(caps.focus, 'en') : '(not picked)'}. In 1-2 sentences, explain tier = "model size tier" — air is smallest+fastest, ultra is biggest+strongest. Recommend a tier based on device rating (${caps.deviceStars}/5 stars).`;
  if (caps.step.key === 'setup') return `Step 4: download + install. In 1-2 sentences as brain, tell the user what happens (pull public weights from mlx-community/HuggingFace → load to memory → ready), and be precise: download uses the network, but user prompts and later inference stay on the Mac.`;
  if (caps.step.key === 'done') return `Step 5: done! Model ready (${caps.loadedName}). In 1-2 sentences, welcome them to start chatting + hint at next step (test chat or export to app).`;
  if (caps.step.key === 'exportDevice') return `Step 6: pick target device. In 1-2 simple sentences, say: this packages ${caps.loadedName || 'me'} into an iPhone/iPad app, and picking a target lets me auto-fit the model size.`;
  if (caps.step.key === 'exportGenerate') return `Step 7: generate app. In 1-2 sentences, explain what's happening (fit check / adapt → generate Xcode project ZIP); the app is fully local, user opens the ZIP in Xcode and compiles.`;
  return `Simple Mode step ${caps.step.index + 1}: ${stepName}. In 1-2 sentences, explain this step in beginner-friendly tone.`;
}

export function getSimpleWizardSuggestedPrompts(
  caps: WizardCapabilities,
  store: SimpleStoreSnapshot,
  locale: Locale,
): { label: string; prompt: string }[] {
  if (locale === 'zh') {
    if (caps.step.key === 'profile') {
      return [
        { label: '🤔 评级是什么意思', prompt: `用户拿到了 ${caps.deviceStars || 0}/5 星评级. 用 2-3 句简单话作为 brain 解释: 这个星级是怎么算的 (RAM / GPU / 芯片代际), 不同星级各能跑多大模型.` },
        { label: '🎯 我能跑什么', prompt: `${store.deviceProfile?.chip || '我'} (${store.deviceProfile?.ram_gb || 0}GB RAM) 大概能跑哪几档模型 (air / standard / pro / max / ultra)? 给具体例子, 每档配 1 个推荐模型.` },
        { label: '🌍 端侧 vs 云端', prompt: `用 2-3 句话解释: 在本机跑 LLM 比 ChatGPT API 哪些情况下更值得 (隐私 / 离线 / 成本), 哪些情况下不值得.` },
        { label: '⚡ 快速开始', prompt: `给一个 30 秒上手的具体路径: 我现在该按什么顺序点 (focus → tier → 下载 → 测试 → 导出), 每步大概要等多久.` },
      ];
    }
    if (caps.step.key === 'focus') {
      return [
        { label: '🎯 我该选哪个', prompt: `用户在选 focus. 设备 ${caps.deviceStars}/5 星. 用 2-3 句话推荐: 新手第一次玩, 应该选 "聊天" 还是 "编码"? 直接给答案.` },
        { label: '👁️ 视觉是啥', prompt: `用 1-2 句话解释 vision focus 是干嘛的 — 就是 LLM 能看图说话 (VLM), 适合什么场景 (拍照问问题 / 截图理解).` },
        { label: '🎙️ 语音对话', prompt: `voice duplex 是 LLM + 语音识别 + 语音合成 三个模型一起跑. 用 1-2 句话告诉新手: 这 RAM 占用大概是单 LLM 的 ${caps.deviceStars >= 3 ? '1.5-2 倍' : '吃力'}, 适合 ${caps.deviceStars >= 4 ? '可以试' : '高配 Mac'}.` },
        { label: '🌍 主权', prompt: `选了 focus 之后, 用户的所有输入 (文字 / 图片 / 语音) 物理上不离开本机吗? 用 2-3 句话作为 brain 给确认 + 解释为什么是绝对的.` },
      ];
    }
    if (caps.step.key === 'tier') {
      return [
        { label: '📊 tier 是啥', prompt: `用 2-3 句话作为 brain 解释 tier (air / standard / pro / max / ultra) 实际对应的模型尺寸 + 用法差异 (大致参数量 / 智能水平 / RAM 占用).` },
        { label: '🎯 推荐我用哪档', prompt: `用户设备 ${caps.deviceStars}/5 星, focus=${focusLabel(store.focus, 'zh')}. 直接推荐一档 + 给理由.` },
        { label: '⚡ 大 vs 小 trade-off', prompt: `用 2-3 句讲: tier 越大越聪明 但 RAM/速度代价是什么? 新手第一次玩应该激进还是保守?` },
        { label: '📈 之后能升级吗', prompt: `如果今天选了 air, 明天发现想升 pro, 流程怎么走 (是不是要重头来 / 还是直接换 tier 重下载)?` },
      ];
    }
    if (caps.step.key === 'setup') {
      return [
        { label: '⏱️ 还要多久', prompt: `${store.setupInfo?.model_display_name || '模型'} (${store.setupInfo?.size_gb.toFixed(1) || '?'} GB). 用 1-2 句话估个下载 + 加载时间 (家用 100M 网 / 千兆网各多久).` },
        { label: '📦 下载到哪', prompt: `用 1-2 句话告诉用户模型文件下载到哪个目录 (~/Documents/mlx-community/), 大约占多少磁盘.` },
        { label: '🌍 私密性', prompt: `下载是从 mlx-community/HuggingFace 拉的. 用 2-3 句话解释: 拉权重不算"上云", 之后所有推理 0 次云调用. HuggingFace 的隐私边界是什么.` },
        { label: '⚠️ 失败了怎么办', prompt: `下载常见失败原因 (网慢 / 镜像不通 / 磁盘不够). 用 2-3 句话给排错清单.` },
      ];
    }
    if (caps.step.key === 'done') {
      return [
        { label: '🎉 接下来玩啥', prompt: `${caps.loadedName} 已就绪. 给新手 3 个最该先试的玩法 (聊天 / 测试一个具体能力 / 导出 App), 每个一句话.` },
        { label: '📦 怎么导出 App', prompt: `用 2-3 句话解释: 把 ${caps.loadedName} 打包成 iPhone App 的步骤 (选目标设备 → 适配 → 下载 ZIP → Xcode 编译).` },
        { label: '🚀 高级模式', prompt: `什么时候该切到 Expert / 高级模式 (有哪些 Simple Mode 看不到的功能 — pipeline 优化 / 多模型对比 / EdgeMesh artifact 管理).` },
        { label: '🌍 私有 AI 飞轮', prompt: `用 2-3 句话作为 brain 描绘一下 EdgeStudio 的完整愿景 (北极星): 模型在你设备上 → 学你的画像 → 在多个 Apple 设备间组 mesh → 越用越懂你.` },
      ];
    }
    if (caps.step.key === 'exportDevice') {
      return [
        { label: '📱 选什么设备', prompt: `用户准备导出 ${caps.loadedName}. 给一个目标设备选择指南 (iPhone 15 Pro / iPad M2 / Mac mini), 每个的 RAM 限制 + 能跑多大模型.` },
        { label: '⚖️ 适配是啥', prompt: `用 2-3 句话解释: 选完目标设备, 系统会做什么 "适配" (是不是会自动量化 / 换更小 base / 还是直接 fail).` },
        { label: '🎁 ZIP 里有啥', prompt: `导出的 ZIP 里包含什么 (Xcode 工程 / 模型权重 / 推理 SDK), 用户解压后该怎么操作.` },
        { label: '🌍 上架要不要审核', prompt: `用 2-3 句话作为 brain 解释: 这种端侧 LLM App 上 App Store 是不是要特殊审核 (是 — 因为模型是用户隐私敏感场景).` },
      ];
    }
    if (caps.step.key === 'exportGenerate') {
      return [
        { label: '⚙️ 现在在做啥', prompt: `导出阶段 = ${store.exportPhase}. 用 1-2 句话作为 brain 告诉用户当前正在执行什么具体动作.` },
        { label: '✅ 成功标志', prompt: `用 1-2 句话告诉用户成功的样子 (downloadUrl 出现 + 一个 ZIP 文件大小约 X MB).` },
        { label: '⚠️ 失败排错', prompt: `如果导出失败, 给一个 3-step 排错清单 (检查目标设备 RAM 够不够 / 模型是否完整 / 磁盘空间).` },
        { label: '🚀 ZIP 后流程', prompt: `用户拿到 ZIP 之后的下一步: 解压 → 在 Xcode 打开 → 改 bundle id → 真机 verify → 上架. 给一个 30 秒走完的 cheatsheet.` },
      ];
    }
    return [
      { label: '🎯 现在该做什么', prompt: `用 1-2 句话作为 brain 直接告诉用户当前这步要做什么.` },
      { label: '🌍 端侧主权', prompt: `用 2-3 句话强调: Simple Mode 整个流程没有任何步骤让用户数据上云.` },
    ];
  }
  // English
  if (caps.step.key === 'profile') {
    return [
      { label: '🤔 What does the rating mean', prompt: `User got ${caps.deviceStars || 0}/5 stars. In 2-3 simple sentences as brain, explain how the rating is computed (RAM / GPU / chip generation) and what each rating tier can run.` },
      { label: '🎯 What I can run', prompt: `${store.deviceProfile?.chip || 'me'} (${store.deviceProfile?.ram_gb || 0}GB RAM) — what tiers (air / standard / pro / max / ultra) can run smoothly here? Give concrete examples + 1 recommended model per tier.` },
      { label: '🌍 Edge vs cloud', prompt: `In 2-3 sentences, explain when running an LLM locally beats ChatGPT API (privacy / offline / cost) and when it doesn't.` },
      { label: '⚡ Quick start', prompt: `Give a 30-second onboarding path: in what order do I click (focus → tier → download → test → export), how long does each step take?` },
    ];
  }
  if (caps.step.key === 'focus') {
    return [
      { label: '🎯 Which to pick', prompt: `User is picking focus. Device ${caps.deviceStars}/5. In 2-3 sentences, recommend "chat" vs "coding" for a first-time user. Direct answer.` },
      { label: '👁️ What is vision', prompt: `In 1-2 sentences, explain vision focus = LLM that can see images (VLM) and what it's good for (photo Q&A / screenshot understanding).` },
      { label: '🎙️ Voice duplex', prompt: `Voice duplex runs LLM + ASR + TTS together. In 1-2 sentences, tell a beginner: RAM is ~${caps.deviceStars >= 3 ? '1.5-2× a single LLM' : 'a stretch'}, suitable for ${caps.deviceStars >= 4 ? 'try it' : 'high-end Macs only'}.` },
      { label: '🌍 Sovereignty', prompt: `After picking focus, will any user input (text / image / voice) physically leave the Mac? In 2-3 sentences as brain, confirm + explain why this is absolute.` },
    ];
  }
  if (caps.step.key === 'tier') {
    return [
      { label: '📊 What is a tier', prompt: `In 2-3 sentences as brain, explain what tier (air / standard / pro / max / ultra) really means (param count / smarts / RAM cost).` },
      { label: '🎯 Recommend a tier', prompt: `Device ${caps.deviceStars}/5, focus=${focusLabel(store.focus, 'en')}. Recommend exactly one tier + reason.` },
      { label: '⚡ Big vs small trade-off', prompt: `In 2-3 sentences, explain bigger tier = smarter but what's the RAM/speed cost? Should a first-time user be aggressive or conservative?` },
      { label: '📈 Can I upgrade later', prompt: `If I pick air today and want pro tomorrow, what's the path (start over? swap tier and re-download?).` },
    ];
  }
  if (caps.step.key === 'setup') {
    return [
      { label: '⏱️ How long', prompt: `${store.setupInfo?.model_display_name || 'model'} (${store.setupInfo?.size_gb.toFixed(1) || '?'} GB). In 1-2 sentences, estimate download + load time on home 100Mbps vs gigabit.` },
      { label: '📦 Where it lands', prompt: `In 1-2 sentences, tell user where the model files go (~/Documents/mlx-community/) and rough disk usage.` },
      { label: '🌍 Privacy', prompt: `Download pulls from mlx-community/HuggingFace. In 2-3 sentences, explain: pulling weights ≠ "to the cloud"; after that all inference is 0 cloud. What's HuggingFace's privacy boundary.` },
      { label: '⚠️ If it fails', prompt: `Common download failures (slow net / mirror blocked / no disk). In 2-3 sentences, give a troubleshooting checklist.` },
    ];
  }
  if (caps.step.key === 'done') {
    return [
      { label: '🎉 What now', prompt: `${caps.loadedName} is ready. Give 3 first things to try (chat / test a specific capability / export to app), one sentence each.` },
      { label: '📦 How to export', prompt: `In 2-3 sentences, explain packaging ${caps.loadedName} into an iPhone App (pick target → adapt → download ZIP → Xcode build).` },
      { label: '🚀 Expert mode', prompt: `When should I switch to Expert / Advanced mode (what features are hidden in Simple — pipeline optimization / multi-model comparison / EdgeMesh artifact management).` },
      { label: '🌍 Private AI flywheel', prompt: `In 2-3 sentences as brain, paint the full EdgeStudio vision (north-star): model on your device → learns your persona → meshes across Apple devices → gets you over time.` },
    ];
  }
  if (caps.step.key === 'exportDevice') {
    return [
      { label: '📱 Which device', prompt: `User exporting ${caps.loadedName}. Give a target device picking guide (iPhone 15 Pro / iPad M2 / Mac mini), each with RAM limits + max model size.` },
      { label: '⚖️ What is adapting', prompt: `In 2-3 sentences, explain: after picking target, what does the system "adapt" (auto-quantize / smaller base / fail).` },
      { label: '🎁 What is in the ZIP', prompt: `What does the export ZIP contain (Xcode project / model weights / inference SDK), what should the user do after unzipping.` },
      { label: '🌍 App Store review', prompt: `In 2-3 sentences as brain, explain: do edge-LLM apps need special App Store review (yes — privacy-sensitive territory).` },
    ];
  }
  if (caps.step.key === 'exportGenerate') {
    return [
      { label: '⚙️ What is happening', prompt: `Export phase = ${store.exportPhase}. In 1-2 sentences as brain, tell user what specific thing is running right now.` },
      { label: '✅ Success looks like', prompt: `In 1-2 sentences, tell user what success looks like (downloadUrl appears + a ZIP about X MB).` },
      { label: '⚠️ If it fails', prompt: `If export fails, give a 3-step troubleshooting checklist (target RAM / model integrity / disk).` },
      { label: '🚀 Post-ZIP', prompt: `After downloading the ZIP: unzip → open in Xcode → change bundle id → device verify → publish. 30-second cheatsheet.` },
    ];
  }
  return [
    { label: '🎯 What do I do', prompt: `In 1-2 sentences as brain, tell the user what this step is asking them to do.` },
    { label: '🌍 Sovereignty', prompt: `In 2-3 sentences, emphasise: no step in Simple Mode lets user data leave the Mac.` },
  ];
}
