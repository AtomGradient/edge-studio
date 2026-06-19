// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * meshInsights — derived topology + chat helpers for the /devices (EdgeMesh) page.
 *
 * EdgeMesh is the cross-device feature: Bonjour
 * discovery + mTLS handshake forms a private inference cluster across a
 * user's Apple devices. The page already does heavy operational lifting
 * (pairing dialogs, training banners, peer rows). What was missing is an
 * "observability layer" that lets the user understand the topology AT A
 * GLANCE and lets the loaded LLM speak as the brain of this mesh.
 *
 * Three-tier role model from the Edge Studio north-star design:
 *   - Mac Studio / Mac        = brain (orchestrator, runs heavy LLMs)
 *   - MacBook                  = hands (compute / training fallback)
 *   - iPhone / iPad            = sensor (input capture, lightweight inference)
 *
 * This file:
 *  - derives mesh capabilities (active / online / stale / never-seen peers,
 *    role histogram, mesh-health colour, this-Mac role)
 *  - assesses mesh-health risk (Bonjour down / no peers / all stale / etc)
 *  - composes a "model speaks as brain in the mesh" system prompt that
 *    extends chatPrompts.buildModelSelfSystemPrompt (the model still knows
 *    its own params, *plus* the mesh state around it)
 *  - generates 4 trio-aware suggested prompts grounded in actual peer data
 *  - emits an auto brief tied to mesh composition + locale
 *
 * Pure functions; no fetching. Page passes already-fetched data in.
 */
import type { ModelInfo } from '@/api/types';
import type { LocalIdentity, MeshStatus, TrustedPeer } from '@/api/mesh';

type Locale = 'en' | 'zh';

/** Recency thresholds (seconds). last_seen_at is Unix seconds. */
const ONLINE_WINDOW_S = 60;        // < 60s ago = online
const STALE_WINDOW_S = 5 * 60;     // 60s..5min = recent
// > 5 min and not null = stale; null = never seen (paired but never connected)

export interface MeshCapabilities {
  local: LocalIdentity | null;
  status: MeshStatus | null;
  /** All peers, includes revoked. */
  peers: TrustedPeer[];
  /** Non-revoked peers. */
  active: TrustedPeer[];
  /** Active peers seen within last 60s. */
  online: TrustedPeer[];
  /** Active peers seen between 60s..5min ago. */
  recent: TrustedPeer[];
  /** Active peers seen >5min ago (likely backgrounded / off-network). */
  stale: TrustedPeer[];
  /** Active peers never observed (paired but never came back). */
  neverSeen: TrustedPeer[];
  /** Revoked peers (still in store, kept for audit). */
  revoked: TrustedPeer[];
  /** Histogram of peer.role values across active peers. */
  byRole: Record<string, TrustedPeer[]>;
  totalCount: number;
  activeCount: number;
  onlineCount: number;
  /** Effective role of THIS Mac in the mesh. 'brain' = LLM loaded + transport
   *  running. 'idle' = no model loaded yet. */
  localRole: 'brain' | 'idle';
  /** True when discovery + transport both running. */
  meshUp: boolean;
  /** Coarse health colour: green = healthy, yellow = caution, red = blocked. */
  meshHealth: 'green' | 'yellow' | 'red';
}

/**
 * Derive mesh capabilities from already-fetched data. Pure function — feeds
 * the IdentityCard tones, the risk banner, and the chat prompts.
 */
export function deriveMeshCapabilities(
  local: LocalIdentity | null,
  peers: TrustedPeer[],
  status: MeshStatus | null,
  hasModelLoaded: boolean,
): MeshCapabilities {
  const nowS = Date.now() / 1000;
  const active = peers.filter((p) => !p.revoked);
  const revoked = peers.filter((p) => p.revoked);

  const online: TrustedPeer[] = [];
  const recent: TrustedPeer[] = [];
  const stale: TrustedPeer[] = [];
  const neverSeen: TrustedPeer[] = [];

  for (const p of active) {
    if (p.last_seen_at == null) {
      neverSeen.push(p);
      continue;
    }
    const age = nowS - p.last_seen_at;
    if (age < ONLINE_WINDOW_S) online.push(p);
    else if (age < STALE_WINDOW_S) recent.push(p);
    else stale.push(p);
  }

  const byRole: Record<string, TrustedPeer[]> = {};
  for (const p of active) {
    const k = (p.role || 'peer').toLowerCase();
    (byRole[k] ||= []).push(p);
  }

  const meshUp = !!status?.transport_running && !!status?.discovery_running;
  const localRole: MeshCapabilities['localRole'] = hasModelLoaded ? 'brain' : 'idle';

  // Coarse health:
  //   red    = mesh transport/discovery down → can't accept any peer
  //   yellow = mesh up but no peers / all stale or never-seen
  //   green  = mesh up AND at least one online peer (real connectivity)
  let meshHealth: MeshCapabilities['meshHealth'];
  if (!meshUp) meshHealth = 'red';
  else if (online.length === 0) meshHealth = 'yellow';
  else meshHealth = 'green';

  return {
    local,
    status,
    peers,
    active,
    online,
    recent,
    stale,
    neverSeen,
    revoked,
    byRole,
    totalCount: peers.length,
    activeCount: active.length,
    onlineCount: online.length,
    localRole,
    meshUp,
    meshHealth,
  };
}

export type MeshRiskLevel = 'safe' | 'caution' | 'danger';
export interface MeshRisk {
  level: MeshRiskLevel;
  reason: string;
  reasonZh: string;
}

/**
 * Per-(mesh state) risk assessment.
 *  - danger:  mesh transport / discovery is not running (cannot accept any peer)
 *  - caution: mesh up but no peers paired yet, OR all active peers stale/never-seen
 *  - caution: revoked peers > active peers (audit smell)
 *  - safe:    mesh up + at least one online peer
 */
export function assessMeshHealth(caps: MeshCapabilities): MeshRisk {
  if (!caps.status) {
    return {
      level: 'caution',
      reason: 'Mesh status not yet available — backend may still be starting.',
      reasonZh: 'Mesh 状态未拉到 — 后端可能还在启动中。',
    };
  }
  if (!caps.status.transport_running) {
    return {
      level: 'danger',
      reason: 'Mesh mTLS transport is NOT running — peers cannot connect.',
      reasonZh: 'Mesh mTLS 通道未启动 — peer 无法接入。',
    };
  }
  if (!caps.status.discovery_running) {
    return {
      level: 'danger',
      reason: 'Bonjour discovery is NOT running — new peers cannot find this Mac.',
      reasonZh: 'Bonjour 发现服务未启动 — 新 peer 无法找到本机。',
    };
  }
  if (caps.activeCount === 0) {
    return {
      level: 'caution',
      reason: 'No paired peers yet. Tap "Pair new device" to bring an iPhone/iPad onto the mesh.',
      reasonZh: '还没有已配对设备. 点 "Pair new device" 把 iPhone/iPad 拉进 mesh.',
    };
  }
  if (caps.onlineCount === 0 && caps.activeCount > 0) {
    return {
      level: 'caution',
      reason: `${caps.activeCount} paired peer(s) but none online — devices may be backgrounded or off network.`,
      reasonZh: `已配对 ${caps.activeCount} 个 peer 但全部离线 — 设备可能后台/不在同一网络.`,
    };
  }
  if (caps.revoked.length > caps.activeCount && caps.revoked.length >= 3) {
    return {
      level: 'caution',
      reason: `More revoked peers (${caps.revoked.length}) than active (${caps.activeCount}) — consider deleting old entries.`,
      reasonZh: `已撤销 ${caps.revoked.length} 个超过 active ${caps.activeCount} 个 — 可考虑清理旧记录.`,
    };
  }
  return {
    level: 'safe',
    reason: `Mesh up · ${caps.onlineCount} online of ${caps.activeCount} paired.`,
    reasonZh: `Mesh 正常 · ${caps.activeCount} 个已配对中 ${caps.onlineCount} 个在线.`,
  };
}

/** Pretty role label used by IdentityCard hint + brief. */
export function roleLabel(role: 'brain' | 'idle' | string, locale: Locale): string {
  const l = role.toLowerCase();
  if (locale === 'zh') {
    if (l === 'brain') return '大脑 (host)';
    if (l === 'idle') return '空闲 (未加载模型)';
    if (l === 'sensor') return '感官';
    if (l === 'hands') return '四肢';
    if (l === 'peer') return '同伴';
    return role;
  }
  if (l === 'brain') return 'Brain (host)';
  if (l === 'idle') return 'Idle (no model)';
  if (l === 'sensor') return 'Sensor';
  if (l === 'hands') return 'Hands';
  if (l === 'peer') return 'Peer';
  return role;
}

/** Build the per-page snippet appended to chatPrompts.buildModelSelfSystemPrompt. */
export function buildMeshContextSnippet(
  caps: MeshCapabilities,
  model: ModelInfo | null,
  locale: Locale,
): string {
  const localName = caps.local?.display_name ?? '(this Mac)';
  const modelName = model?.model_name ?? '(no model loaded)';
  const peerLines: string[] = [];
  if (caps.active.length === 0) {
    peerLines.push(locale === 'zh' ? '- 还没有 paired peer.' : '- No paired peers yet.');
  } else {
    for (const p of caps.active.slice(0, 8)) {
      const last =
        p.last_seen_at == null
          ? (locale === 'zh' ? '从未上线' : 'never seen')
          : `last seen ${Math.round((Date.now() / 1000 - p.last_seen_at))}s ago`;
      peerLines.push(`- ${p.display_name} (role=${p.role}) — ${last}`);
    }
    if (caps.active.length > 8) peerLines.push(`- … and ${caps.active.length - 8} more`);
  }

  const lines: string[] = [
    locale === 'zh' ? `## 你所在的 EdgeMesh 拓扑` : `## YOUR EDGEMESH TOPOLOGY`,
    locale === 'zh'
      ? `你 (${modelName}) 跑在 "${localName}" — 这台 Mac 是 mesh 的 brain (host).`
      : `You (${modelName}) are running on "${localName}" — this Mac is the brain (host) of the mesh.`,
    `- This Mac role: ${caps.localRole.toUpperCase()} (${roleLabel(caps.localRole, locale)})`,
    `- Mesh transport: ${caps.status?.transport_running ? 'UP' : 'DOWN'} on port ${caps.status?.mesh_port ?? '?'}`,
    `- Bonjour discovery: ${caps.status?.discovery_running ? 'UP' : 'DOWN'}`,
    `- Paired peers (active / total): ${caps.activeCount} / ${caps.totalCount}`,
    `- Online peers (last 60s): ${caps.onlineCount}`,
    `- Stale peers (>5 min): ${caps.stale.length}`,
    `- Never-seen peers (paired, never connected): ${caps.neverSeen.length}`,
    `- Revoked peers: ${caps.revoked.length}`,
    `- Role histogram: ${Object.entries(caps.byRole).map(([k, v]) => `${v.length}× ${k}`).join(', ') || '(empty)'}`,
    ``,
    locale === 'zh' ? `### Peer 详情:` : `### Peer details:`,
    ...peerLines,
    ``,
    locale === 'zh'
      ? `### 三层架构 (北极星 §3):`
      : `### Three-tier architecture (north-star §3):`,
    locale === 'zh'
      ? `- iPhone / iPad → sensor (输入捕获 + 轻量推理)`
      : `- iPhone / iPad → sensor (input capture + lightweight inference)`,
    locale === 'zh'
      ? `- MacBook → hands (artifact 导出 + host 模型协助 + mesh 同步)`
      : `- MacBook → hands (artifact export + host-model assist + mesh sync)`,
    locale === 'zh'
      ? `- Mac Studio / Mac → brain (大模型 host, 你就在这里)`
      : `- Mac Studio / Mac → brain (heavy LLM host — that's where you are)`,
    ``,
    locale === 'zh'
      ? `### 关键事实 (用户问拓扑/角色/路由时引用):`
      : `### Key facts (cite when user asks about topology / roles / routing):`,
    locale === 'zh'
      ? `- 所有 peer 通过 mTLS + Bonjour 通信, 0 次云调用 (北极星 §1).`
      : `- All peers talk over mTLS + Bonjour, zero cloud relays (north-star §1).`,
    locale === 'zh'
      ? `- 数据物理上不离开 mesh, 即使在 peer 之间也是端到端加密.`
      : `- Data never physically leaves the mesh; even peer-to-peer traffic is end-to-end encrypted.`,
  ];
  return lines.join('\n');
}

/** Auto-fired brief — short, mesh-aware, sovereignty-flavored. */
export function buildMeshAutoBrief(caps: MeshCapabilities, locale: Locale): string {
  if (locale === 'zh') {
    if (!caps.meshUp) {
      return `mesh transport/discovery 未启动. 用 2-3 句话作为这台 Mac 的 brain (or 空闲), 解释当前哪个组件没起来 + 用户需要做什么. 第一人称, 不列项.`;
    }
    if (caps.activeCount === 0) {
      return `Mesh 已就绪但还没有 peer. 用 2-3 句话作为 brain 介绍三层架构 (iPhone=sensor, MacBook=hands, 你=brain), 邀请用户配对第一台设备. 第一人称, 不列项.`;
    }
    if (caps.onlineCount === 0) {
      return `已配对 ${caps.activeCount} 个 peer 但全部离线. 用 2-3 句话作为 brain 给出诊断 (检查同 WiFi / app 在前台 / 防火墙) + 1 个最可能的原因. 第一人称.`;
    }
    return `Mesh 运行中 — ${caps.activeCount} 个 peer 中 ${caps.onlineCount} 个在线 (角色: ${Object.entries(caps.byRole).map(([k, v]) => `${v.length}× ${k}`).join(', ') || '混合'}). 用 2-3 句话作为 brain 总结拓扑健康度 + 推荐用户接下来可以做什么 (同步 Neural Imprint artifact / 路由任务). 第一人称, 引用真实数字.`;
  }
  if (!caps.meshUp) {
    return `Mesh transport/discovery is not running. In 2-3 sentences, speaking as this Mac's brain (or idle), explain which component is down and what the user should do. First person, no bullets.`;
  }
  if (caps.activeCount === 0) {
    return `Mesh is up but no peers yet. In 2-3 sentences, as the brain, introduce the three-tier architecture (iPhone=sensor, MacBook=hands, you=brain) and invite the user to pair their first device. First person, no bullets.`;
  }
  if (caps.onlineCount === 0) {
    return `${caps.activeCount} paired peer(s) but none online. In 2-3 sentences, as the brain, give a quick diagnosis (same WiFi? app in foreground? firewall?) and the most likely root cause. First person.`;
  }
  return `Mesh is healthy — ${caps.onlineCount} of ${caps.activeCount} peers online (roles: ${Object.entries(caps.byRole).map(([k, v]) => `${v.length}× ${k}`).join(', ') || 'mixed'}). In 2-3 sentences, as the brain, summarise topology health and recommend a next step (sync a Neural Imprint artifact / route a task). First person, cite real numbers.`;
}

/** 4 mesh-aware suggested prompts. Each must reference real numbers / names. */
export function getMeshSuggestedPrompts(
  caps: MeshCapabilities,
  model: ModelInfo | null,
  locale: Locale,
): { label: string; prompt: string }[] {
  const peerNames = caps.active.slice(0, 3).map((p) => p.display_name).join(', ') || (locale === 'zh' ? '(无)' : '(none)');
  const modelName = model?.model_name || (locale === 'zh' ? '我' : 'me');

  if (locale === 'zh') {
    if (caps.activeCount === 0) {
      return [
        { label: '🧠 我的角色', prompt: `作为这台 Mac 上加载的 ${modelName}, 用第一人称解释三层架构 (iPhone=sensor, MacBook=hands, Mac Studio=brain), 我是 brain. 2-3 句话不列项.` },
        { label: '📡 怎么开始', prompt: `用户还没配对任何设备. 给一个具体的入门路径: 第一台应该配 iPhone 还是 iPad? 为什么? 配完能立刻看到什么变化?` },
        { label: '🔐 mTLS 主权', prompt: `作为 brain, 用 2-3 句话解释为什么 mesh 用 mTLS + Bonjour 而不是云端中继, 这条 "数据物理上不离开 mesh" 对用户意味着什么 (隐私 / 延迟 / 离线).` },
        { label: '🎯 飞轮第一步', prompt: `mesh 配齐后会怎么联动端侧自学习 (北极星 T11)? 用 2-3 句话讲一下: 用户在 iPhone 上的交互怎么生成 RPP/Neural Imprint artifact, 又怎么通过 EdgeStudio 审计和同步.` },
      ];
    }
    return [
      { label: '🧠 我的角色', prompt: `作为这台 Mac 上加载的 ${modelName}, 用第一人称解释自己是 mesh 的 brain. 我看到 ${caps.activeCount} 个 paired peer (${peerNames}), ${caps.onlineCount} 个在线. 介绍三层架构 + 我在其中扮演什么. 2-3 句话.` },
      { label: '📡 拓扑现状', prompt: `当前 ${caps.activeCount} 个 paired peer, ${caps.onlineCount} 在线, ${caps.stale.length} stale, ${caps.neverSeen.length} 从未上线. 作为 brain 给一个快速健康度诊断 + 最该先关注的 peer.` },
      { label: '🌐 联邦推理', prompt: `${caps.onlineCount} 个 peer 在线. 作为 brain, 给一个具体的任务路由策略: 哪种 prompt 留在我本地跑 (我是 ${modelName}), 哪种应该 offload 到 peer 上? 2-3 句话, 给具体例子.` },
      { label: '🔐 mTLS 主权', prompt: `作为 brain, 用 2-3 句话解释 mesh 上 ${caps.activeCount} 个 peer 之间的通信 — 为什么 0 次云调用比"端到端加密的云端 IM" 还要彻底 (mTLS 证书钉扎 / Bonjour 局域 / 物理边界).` },
    ];
  }
  if (caps.activeCount === 0) {
    return [
      { label: '🧠 My role', prompt: `As ${modelName} loaded on this Mac, explain in first person the three-tier architecture (iPhone=sensor, MacBook=hands, Mac Studio=brain) — I'm the brain. 2-3 sentences, no bullets.` },
      { label: '📡 How to start', prompt: `User has no paired devices yet. Give a concrete onboarding path: pair iPhone first or iPad? Why? What will they see change immediately after pairing?` },
      { label: '🔐 mTLS sovereignty', prompt: `As the brain, in 2-3 sentences explain why the mesh uses mTLS + Bonjour instead of a cloud relay, and what "data never physically leaves the mesh" means for the user (privacy / latency / offline).` },
      { label: '🎯 First flywheel step', prompt: `Once the mesh is populated, how does it connect to on-device self-learning (north-star T11)? In 2-3 sentences: how does iPhone interaction produce RPP/Neural Imprint artifacts, and how does EdgeStudio audit and sync them?` },
    ];
  }
  return [
    { label: '🧠 My role', prompt: `As ${modelName} loaded on this Mac, explain in first person that I'm the brain of this mesh. I see ${caps.activeCount} paired peer(s) (${peerNames}), ${caps.onlineCount} online. Cover the three-tier architecture + my place in it. 2-3 sentences.` },
    { label: '📡 Topology check', prompt: `Currently ${caps.activeCount} paired, ${caps.onlineCount} online, ${caps.stale.length} stale, ${caps.neverSeen.length} never seen. As brain, give a quick health diagnosis + the one peer I'd ask the user to look at first.` },
    { label: '🌐 Federated routing', prompt: `${caps.onlineCount} peers online. As the brain, propose a concrete task-routing policy: what prompts stay local on me (${modelName}) and what should I offload to a peer? 2-3 sentences with examples.` },
    { label: '🔐 mTLS sovereignty', prompt: `As brain, in 2-3 sentences explain how communication among the ${caps.activeCount} peers stays private — why "zero cloud calls" is stricter than "end-to-end encrypted cloud IM" (mTLS pinning / Bonjour LAN / physical boundary).` },
  ];
}
