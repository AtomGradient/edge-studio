// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 3: Setup — download model(s) + load + experience.
 *
 * Download modes:
 * - Terminal (default): PTY-based download via msd.sh/hfd.sh with real-time output
 * - Background: task-based polling with progress bar (fallback)
 *
 * Supports:
 * - Single-model focuses (chat/coding/vision/asr/tts): download one model
 * - Voice Duplex: downloads three models sequentially (ASR → TTS → LLM)
 */

import { useEffect, useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Loader2, Check, X, WifiOff, Shield, Zap, Battery, Smartphone,
  Send, ImagePlus,
} from 'lucide-react';
import { useSimpleStore } from '@/stores/simpleStore';
import { useModelStore } from '@/stores/modelStore';
import { stripThinking } from '@/lib/textUtils';
import MarkdownContent from '@/components/MarkdownContent';
import { WizardShell } from '@/components/common/WizardShell';
import { Terminal } from '@/components/common/Terminal';
import { useT } from '@/i18n';
import { WIZARD_STEPS_V2 } from './wizardStepsV2';
import { cn } from '@/lib/utils';
import axios from 'axios';
import { buildChatWsUrl, createReconnectingWebSocket } from '@/api/websocket';
import { friendlyError } from '@/lib/friendlyError';
import { runTerminalCommand, closeTerminal } from '@/api/endpoints';
import ASRPanel from './ASRPanel';
import TTSPanel from './TTSPanel';
import DuplexPanel from './DuplexPanel';

// Tips shown during download
const TIPS = [
  { icon: WifiOff, titleKey: 'simple.v2.setup.tip.offline.title', descKey: 'simple.v2.setup.tip.offline.desc' },
  { icon: Shield, titleKey: 'simple.v2.setup.tip.privacy.title', descKey: 'simple.v2.setup.tip.privacy.desc' },
  { icon: Zap, titleKey: 'simple.v2.setup.tip.metal.title', descKey: 'simple.v2.setup.tip.metal.desc' },
  { icon: Battery, titleKey: 'simple.v2.setup.tip.cost.title', descKey: 'simple.v2.setup.tip.cost.desc' },
  { icon: Smartphone, titleKey: 'simple.v2.setup.tip.app.title', descKey: 'simple.v2.setup.tip.app.desc' },
];

// Quick prompts per focus
const QUICK_PROMPTS: Record<string, string[]> = {
  chat: [
    'simple.v2.setup.quickPrompt.chat1',
    'simple.v2.setup.quickPrompt.chat2',
    'simple.v2.setup.quickPrompt.chat3',
    'simple.v2.setup.quickPrompt.chat4',
  ],
  coding: [
    'simple.v2.setup.quickPrompt.coding1',
    'simple.v2.setup.quickPrompt.coding2',
    'simple.v2.setup.quickPrompt.coding3',
  ],
  vision: [
    'simple.v2.setup.quickPrompt.vision1',
    'simple.v2.setup.quickPrompt.vision2',
  ],
};

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

interface TerminalResumeState {
  sessionId: string;
  mirror: string;
  mode: 'single' | 'duplex';
  queue: { hint: string; overall: number }[];
  step: number;
}

interface LoadedModelEntry {
  model_id?: string;
}

const TERMINAL_RESUME_KEY = 'edge-simple-v2-terminal-resume';

function readTerminalResume(): TerminalResumeState | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(TERMINAL_RESUME_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as TerminalResumeState;
    if (!parsed.sessionId || !parsed.mode || !Array.isArray(parsed.queue)) return null;
    return parsed;
  } catch {
    return null;
  }
}

function writeTerminalResume(state: TerminalResumeState) {
  if (typeof window === 'undefined') return;
  window.sessionStorage.setItem(TERMINAL_RESUME_KEY, JSON.stringify(state));
}

function clearTerminalResume() {
  if (typeof window === 'undefined') return;
  window.sessionStorage.removeItem(TERMINAL_RESUME_KEY);
}

// ---------------------------------------------------------------------------
// Mirror probe
// ---------------------------------------------------------------------------

async function probeMirror(): Promise<'official' | 'hf-mirror' | 'modelscope'> {
  try {
    const probe = await axios.get('/api/hf/probe');
    if (!probe.data.reachable && probe.data.suggestion) return probe.data.suggestion;
    return 'official';
  } catch {
    return 'hf-mirror';
  }
}

// ---------------------------------------------------------------------------
// Terminal download helper — runs msd.sh or hfd.sh in PTY
// ---------------------------------------------------------------------------

async function startTerminalDownload(
  repoId: string,
  mirror: string,
): Promise<string> {
  const localDir = `~/mlx-community/${repoId.replace('/', '_')}`;
  const useChinaSource = mirror === 'hf-mirror' || mirror === 'modelscope';

  // China mainland: msd.sh (ModelScope, more stable)
  // Official: hfd.sh (direct HuggingFace with aria2c)
  const cmd = useChinaSource
    ? ['bash', 'scripts/msd.sh', repoId, '--local-dir', localDir]
    : ['bash', 'scripts/hfd.sh', repoId, '--local-dir', localDir];

  const { session_id } = await runTerminalCommand(cmd);
  return session_id;
}

function getAxiosDetail(error: unknown): string | undefined {
  if (!axios.isAxiosError<{ detail?: unknown }>(error)) return undefined;
  const detail = error.response?.data?.detail;
  return typeof detail === 'string' ? detail : undefined;
}

// ---------------------------------------------------------------------------
// SetupPage
// ---------------------------------------------------------------------------

export default function SetupPage() {
  const t = useT();
  const navigate = useNavigate();
  const {
    focus, tier, customModelId, packages, setupInfo, setSetupInfo,
    setupPhase, setSetupPhase,
    loadedModelId, setLoadedModelId,
    loadedModelDir, setLoadedModelDir,
    setChatTested,
    // Duplex
    ttsModelId, setTtsModelId, setTtsModelDir,
    duplexLlmModelId, setDuplexLlmModelId, setDuplexLlmModelDir,
    duplexAsrModelId, setDuplexAsrModelId, setDuplexAsrModelDir,
    duplexDownloadStep, setDuplexDownloadStep,
    ttsVariant, duplexVoice, setDuplexVoice, duplexInstruct, setDuplexInstruct,
  } = useSimpleStore();
  const setCurrentModel = useModelStore((s) => s.setCurrentModel);

  const [error, setError] = useState('');
  const [tipIndex, setTipIndex] = useState(0);
  const [tipVisible, setTipVisible] = useState(true);

  // Terminal download state (default ON)
  const [terminalSessionId, setTerminalSessionId] = useState<string | null>(null);
  const [terminalMirror, setTerminalMirror] = useState<string>('');
  const [preparing, setPreparing] = useState(false); // show spinner during mirror probe

  // Guard: prevent double-starting terminal (async gap between probe and PTY creation)
  const downloadStartedRef = useRef(false);
  const terminalResumeRef = useRef<TerminalResumeState | null>(null);

  // For duplex terminal: track which model hint is being downloaded
  // Each entry: { hint: download_hint, overall: 1=ASR/2=TTS/3=LLM }
  const duplexTerminalQueueRef = useRef<{ hint: string; overall: number }[]>([]);
  const duplexTerminalStepRef = useRef(0);

  // Chat state (for text-based focuses)
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const wsRef = useRef<ReturnType<typeof createReconnectingWebSocket> | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cancelledRef = useRef(false);

  // VLM image state
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [imageThumb, setImageThumb] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // TTS voice/instruct state for duplex
  const [speakerList, setSpeakerList] = useState<string[]>([]);
  const [speakersLoading, setSpeakersLoading] = useState(false);

  const isDuplex = focus === 'voice_duplex';

  // Fetch speaker list for CustomVoice duplex when TTS model is loaded
  useEffect(() => {
    if (!isDuplex || !ttsModelId || ttsVariant !== 'customvoice') return;
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- refresh speaker list when active TTS model changes
    setSpeakersLoading(true);
    axios.get(`/api/chat/${ttsModelId}/tts-voices`)
      .then((res) => {
        if (cancelled) return;
        const voices: string[] = res.data?.voices || [];
        setSpeakerList(voices);
        if (voices.length > 0 && !duplexVoice) {
          setDuplexVoice(voices[0]);
        }
      })
      .catch(() => { if (!cancelled) setSpeakerList([]); })
      .finally(() => { if (!cancelled) setSpeakersLoading(false); });
    return () => { cancelled = true; };
  }, [ttsModelId, ttsVariant, isDuplex]); // eslint-disable-line react-hooks/exhaustive-deps

  // Set default instruct for VoiceDesign duplex
  useEffect(() => {
    if (isDuplex && ttsVariant === 'voicedesign' && !duplexInstruct) {
      setDuplexInstruct('A natural, friendly voice with moderate pace');
    }
  }, [ttsVariant, isDuplex]); // eslint-disable-line react-hooks/exhaustive-deps

  // Rotate tips every 5s during download with fade transition
  useEffect(() => {
    if (setupPhase !== 'downloading') return;
    const interval = setInterval(() => {
      setTipVisible(false);
      setTimeout(() => {
        setTipIndex((prev) => (prev + 1) % TIPS.length);
        setTipVisible(true);
      }, 300);
    }, 5000);
    return () => clearInterval(interval);
  }, [setupPhase]);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ---------------------------------------------------------------------------
  // Terminal download handlers
  // ---------------------------------------------------------------------------

  const rememberTerminalSession = useCallback((
    sessionId: string,
    mirror: string,
    mode: 'single' | 'duplex',
    queue: { hint: string; overall: number }[],
    step: number,
  ) => {
    const resume = { sessionId, mirror, mode, queue, step };
    terminalResumeRef.current = resume;
    writeTerminalResume(resume);
  }, []);

  const restoreTerminalSession = useCallback(() => {
    const resume = readTerminalResume();
    if (!resume) return false;
    terminalResumeRef.current = resume;
    downloadStartedRef.current = true;
    cancelledRef.current = false;
    setError('');
    setPreparing(false);
    setTerminalMirror(resume.mirror);
    setTerminalSessionId(resume.sessionId);
    setSetupPhase('downloading');
    if (resume.mode === 'duplex') {
      duplexTerminalQueueRef.current = resume.queue;
      duplexTerminalStepRef.current = resume.step;
      const current = resume.queue[resume.step - 1];
      if (current) setDuplexDownloadStep(current.overall);
    }
    return true;
  }, [setSetupPhase, setDuplexDownloadStep]);

  const handleTerminalError = useCallback(() => {
    clearTerminalResume();
    terminalResumeRef.current = null;
    downloadStartedRef.current = false;
    setTerminalSessionId(null);
    setPreparing(false);
    setError(t('simple.v2.setup.terminalReattachFailed'));
    setSetupPhase('idle');
  }, [setSetupPhase, t]);

  const handleTerminalExit = useCallback(async (code: number) => {
    // Clean up terminal session
    if (terminalSessionId) {
      closeTerminal(terminalSessionId).catch(() => {});
    }
    setTerminalSessionId(null);
    setPreparing(false);
    clearTerminalResume();

    if (cancelledRef.current) {
      cancelledRef.current = false;
      downloadStartedRef.current = false;
      setError(t('common.cancelled'));
      setSetupPhase('idle');
      return;
    }

    if (code !== 0) {
      downloadStartedRef.current = false;
      setError(t('simple.v1.downloadExitCode', { code: String(code) }));
      setSetupPhase('idle');
      return;
    }

    if (isDuplex) {
      // Duplex: load current model, then advance to next
      const resume = terminalResumeRef.current;
      const queue = duplexTerminalQueueRef.current.length > 0
        ? duplexTerminalQueueRef.current
        : (resume?.mode === 'duplex' ? resume.queue : []);
      const step = duplexTerminalStepRef.current || (resume?.mode === 'duplex' ? resume.step : 1);
      const entry = queue[step - 1]; // current completed entry
      if (!entry) {
        downloadStartedRef.current = false;
        setError(t('simple.v1.terminalFailed'));
        setSetupPhase('idle');
        return;
      }
      const localDir = `~/mlx-community/${entry.hint.replace('/', '_')}`;

      try {
        const loadRes = await axios.post('/api/model/load', { model_dir: localDir });
        const modelId = loadRes.data.model_id;

        // Store result based on overall step (1=ASR, 2=TTS, 3=LLM)
        if (entry.overall === 1) {
          setDuplexAsrModelId(modelId);
          setDuplexAsrModelDir(localDir);
        } else if (entry.overall === 2) {
          setTtsModelId(modelId);
          setTtsModelDir(localDir);
        } else {
          setDuplexLlmModelId(modelId);
          setDuplexLlmModelDir(localDir);
        }

        // More models to download?
        if (step < queue.length) {
          duplexTerminalStepRef.current = step + 1;
          const nextEntry = queue[step]; // 0-indexed, step is already 1-based
          setDuplexDownloadStep(nextEntry.overall);
          const mirror = terminalMirror || await probeMirror();
          setTerminalMirror(mirror);
          const sessionId = await startTerminalDownload(nextEntry.hint, mirror);
          rememberTerminalSession(sessionId, mirror, 'duplex', queue, step + 1);
          setTerminalSessionId(sessionId);
        } else {
          // All done
          downloadStartedRef.current = false;
          terminalResumeRef.current = null;
          setSetupPhase('ready');
        }
      } catch (err: unknown) {
        downloadStartedRef.current = false;
        setError(friendlyError(getAxiosDetail(err), t, 'simple.error.loadFailed'));
        setSetupPhase('idle');
      }
    } else {
      // Single model: load it
      const resume = terminalResumeRef.current;
      const hint = setupInfo?.download_hint || resume?.queue[0]?.hint || '';
      if (!hint) {
        downloadStartedRef.current = false;
        setError(t('simple.v1.terminalFailed'));
        setSetupPhase('idle');
        return;
      }
      const localDir = `~/mlx-community/${hint.replace('/', '_')}`;

      setSetupPhase('loading');
      try {
        const loadRes = await axios.post('/api/model/load', { model_dir: localDir });
        setCurrentModel(loadRes.data);
        setLoadedModelId(loadRes.data.model_id);
        setLoadedModelDir(localDir);
        downloadStartedRef.current = false;
        terminalResumeRef.current = null;
        setSetupPhase('ready');
      } catch (err: unknown) {
        downloadStartedRef.current = false;
        setError(friendlyError(getAxiosDetail(err), t, 'simple.error.loadFailed'));
        setSetupPhase('idle');
      }
    }
  }, [terminalSessionId, isDuplex, setupInfo, terminalMirror, t,
      setSetupPhase, setCurrentModel, setLoadedModelId, setLoadedModelDir,
      setDuplexAsrModelId, setDuplexAsrModelDir, setTtsModelId, setTtsModelDir,
      setDuplexLlmModelId, setDuplexLlmModelDir, setDuplexDownloadStep,
      rememberTerminalSession]);

  const handleTerminalClose = useCallback(() => {
    cancelledRef.current = true;
    clearTerminalResume();
    terminalResumeRef.current = null;
    if (terminalSessionId) {
      closeTerminal(terminalSessionId).catch(() => {});
    }
    setTerminalSessionId(null);
    setSetupPhase('idle');
    setPreparing(false);
    setError(t('common.cancelled'));
    downloadStartedRef.current = false;
  }, [terminalSessionId, setSetupPhase, t]);

  // Start terminal download for single-model
  const startTerminalSingleDownload = useCallback(async () => {
    if (!setupInfo || downloadStartedRef.current) return;
    downloadStartedRef.current = true;
    cancelledRef.current = false;
    setError('');
    setSetupPhase('downloading');
    setPreparing(true);

    try {
      const mirror = await probeMirror();
      setTerminalMirror(mirror);
      setPreparing(false);
      const sessionId = await startTerminalDownload(setupInfo.download_hint, mirror);
      rememberTerminalSession(sessionId, mirror, 'single', [{ hint: setupInfo.download_hint, overall: 1 }], 1);
      setTerminalSessionId(sessionId);
    } catch {
      setError(t('simple.v1.terminalFailed'));
      setSetupPhase('idle');
      setPreparing(false);
      downloadStartedRef.current = false;
    }
  }, [setupInfo, setSetupPhase, t]);

  // Start terminal download for duplex (three models sequentially)
  const startTerminalDuplexDownload = useCallback(async () => {
    if (downloadStartedRef.current) return;
    const selectedPkg = packages.find((p) => p.tier === tier);
    if (!selectedPkg?.model || !selectedPkg.secondary_model || !selectedPkg.tertiary_model) {
      setError('Missing model info in package');
      return;
    }

    downloadStartedRef.current = true;
    cancelledRef.current = false;
    setError('');
    setSetupPhase('downloading');
    setPreparing(true);

    // Build download queue: ASR(1) → TTS(2) → LLM(3)
    // Skip already-loaded models. Each entry carries its overall step number.
    const queue: { hint: string; overall: number }[] = [];
    const models = [
      { hint: selectedPkg.tertiary_model.download_hint, check: duplexAsrModelId, overall: 1,
        setId: setDuplexAsrModelId, setDir: setDuplexAsrModelDir },
      { hint: selectedPkg.secondary_model.download_hint, check: ttsModelId, overall: 2,
        setId: setTtsModelId, setDir: setTtsModelDir },
      { hint: selectedPkg.model.download_hint, check: duplexLlmModelId, overall: 3,
        setId: setDuplexLlmModelId, setDir: setDuplexLlmModelDir },
    ];

    for (const { hint, check, overall, setId, setDir } of models) {
      if (!check) {
        // Try to load from disk first (already downloaded but not loaded)
        const localDir = `~/mlx-community/${hint.replace('/', '_')}`;
        try {
          const loadRes = await axios.post('/api/model/load', { model_dir: localDir });
          setId(loadRes.data.model_id);
          setDir(localDir);
          continue; // Already on disk, skip download
        } catch {
          queue.push({ hint, overall }); // Need to download
        }
      }
    }

    if (queue.length === 0) {
      setSetupPhase('ready');
      return;
    }

    duplexTerminalQueueRef.current = queue;
    duplexTerminalStepRef.current = 1;
    setDuplexDownloadStep(queue[0].overall);

    try {
      const mirror = await probeMirror();
      setTerminalMirror(mirror);
      setPreparing(false);
      const sessionId = await startTerminalDownload(queue[0].hint, mirror);
      rememberTerminalSession(sessionId, mirror, 'duplex', queue, 1);
      setTerminalSessionId(sessionId);
    } catch {
      setError(t('simple.v1.terminalFailed'));
      setSetupPhase('idle');
      setPreparing(false);
      downloadStartedRef.current = false;
    }
  }, [packages, tier, duplexAsrModelId, ttsModelId, duplexLlmModelId,
      setSetupPhase, setDuplexDownloadStep, setDuplexAsrModelId, setDuplexAsrModelDir,
      setTtsModelId, setTtsModelDir, setDuplexLlmModelId, setDuplexLlmModelDir, t]);

  async function loadModel(modelDir: string) {
    setSetupPhase('loading');
    try {
      const res = await axios.post('/api/model/load', { model_dir: modelDir });
      setCurrentModel(res.data);
      setLoadedModelId(res.data.model_id);
      setLoadedModelDir(modelDir);
      clearTerminalResume();
      terminalResumeRef.current = null;
      setSetupPhase('ready');
    } catch (err: unknown) {
      setError(friendlyError(getAxiosDetail(err), t, 'simple.error.loadFailed'));
      setSetupPhase('idle');
    }
  }

  // ---------------------------------------------------------------------------
  // Model validation on mount
  // ---------------------------------------------------------------------------

  useEffect(() => {
    let cancelled = false;

    async function validate() {
      if (setupPhase === 'downloading') {
        if (restoreTerminalSession()) return;
        setError(t('simple.v2.setup.terminalReattachFailed'));
        setSetupPhase('idle');
        return;
      }

      if (isDuplex) {
        // Duplex: check if all three models are loaded
        if (duplexLlmModelId && ttsModelId && duplexAsrModelId) {
          try {
            const res = await axios.get('/api/model/loaded');
            const loadedIds = new Set((res.data as LoadedModelEntry[]).map((m) => m.model_id));
            if (loadedIds.has(duplexLlmModelId) && loadedIds.has(ttsModelId) && loadedIds.has(duplexAsrModelId)) {
              setSetupPhase('ready');
              return;
            }
          } catch {
            // Treat failed validation as not loaded and continue setup.
          }
        }
        if (!cancelled) startTerminalDuplexDownload();
        return;
      }

      // Single-model validation
      if (loadedModelId) {
        try {
          const res = await axios.get('/api/model/loaded');
          const loadedIds = new Set((res.data as LoadedModelEntry[]).map((m) => m.model_id));
          if (loadedIds.has(loadedModelId)) {
            setSetupPhase('ready');
            return;
          }
        } catch {
          // Treat failed validation as not loaded and fall back to local path.
        }

        if (loadedModelDir) {
          if (!cancelled) await loadModel(loadedModelDir);
          return;
        }

        if (!cancelled) {
          setLoadedModelId('');
          setLoadedModelDir('');
          setSetupPhase('idle');
        }
      }

      // No persisted model — resolve setup info if needed
      if (!setupInfo && !cancelled) {
        if (focus && (tier || customModelId)) {
          try {
            const payload: Record<string, string> = { focus };
            if (customModelId) {
              payload.custom_model_id = customModelId;
            } else {
              payload.tier = tier;
            }
            const res = await axios.post('/api/simple/setup', payload);
            if (!cancelled) setSetupInfo(res.data);
          } catch {
            if (!cancelled) navigate('/simple/tier');
          }
        } else if (!focus) {
          if (!cancelled) navigate('/simple/focus');
        } else {
          if (!cancelled) navigate('/simple/tier');
        }
      }
    }

    validate();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Start terminal download when setupInfo arrives and no model loaded yet
  useEffect(() => {
    if (isDuplex || !setupInfo || loadedModelId || setupPhase !== 'idle') return;

    if (setupInfo.already_downloaded && setupInfo.local_dir) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- load already-downloaded model when setup info arrives
      loadModel(setupInfo.local_dir);
    } else {
      startTerminalSingleDownload();
    }
  }, [setupInfo]); // eslint-disable-line react-hooks/exhaustive-deps

  // Warn before closing tab during download
  useEffect(() => {
    if (setupPhase !== 'downloading') return;
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = '';
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [setupPhase]);

  // Cleanup WS on unmount
  useEffect(() => {
    return () => { wsRef.current?.close(); };
  }, []);

  // Connect or reuse existing WebSocket
  const connectWs = useCallback((modelId: string) => {
    if (wsRef.current && wsRef.current.readyState() === WebSocket.OPEN) {
      return wsRef.current;
    }
    wsRef.current?.close();
    const handle = createReconnectingWebSocket(buildChatWsUrl(modelId));
    wsRef.current = handle;
    return handle;
  }, []);

  // VLM: image selection handler
  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target?.result as string;
      setImageB64(dataUrl.split(',')[1]);
      setImageThumb(dataUrl);
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }, []);

  // Chat: send text message (with optional image for VLM)
  const sendMessage = useCallback((text: string) => {
    if (!text.trim() || !loadedModelId || streaming) return;

    const userMsg: ChatMessage = { role: 'user', text: text.trim() };
    setMessages((prev) => [...prev, userMsg, { role: 'assistant', text: '' }]);
    setInput('');
    setStreaming(true);
    setChatTested(true);

    const history = messages.map((m) => ({
      role: m.role,
      content: m.text,
    }));

    const handle = connectWs(loadedModelId);

    let assistantText = '';
    handle.setOnMessage((event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'token') {
          assistantText += data.token || '';
          const display = stripThinking(assistantText);
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'assistant', text: display };
            return updated;
          });
        } else if (data.type === 'complete') {
          if (data.full_text) assistantText = data.full_text;
          const display = stripThinking(assistantText);
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'assistant', text: display };
            return updated;
          });
          setStreaming(false);
        } else if (data.type === 'error') {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: 'assistant', text: data.message || t('simple.error.unknown') };
            return updated;
          });
          setStreaming(false);
        }
      } catch {
        // Ignore malformed websocket frames.
      }
    });

    // Include image for VLM
    const payload: Record<string, unknown> = { prompt: text.trim(), history, enable_thinking: false };
    if (focus === 'vision' && imageB64) {
      payload.image_b64 = imageB64;
      setImageB64(null);
      setImageThumb(null);
    }
    handle.send(JSON.stringify(payload));
  }, [loadedModelId, messages, streaming, setChatTested, connectWs, focus, imageB64, t]);

  const handleNext = useCallback(() => {
    // Close WebSocket cleanly before navigating
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setTimeout(() => navigate('/simple/done'), 100);
  }, [navigate]);

  const currentTip = TIPS[tipIndex];
  const TipIcon = currentTip.icon;
  const prompts = QUICK_PROMPTS[focus] || [];
  const isReady = setupPhase === 'ready';

  // Duplex download step label
  const duplexStepLabel = isDuplex && duplexDownloadStep > 0
    ? t('simple.v2.setup.duplexStep', {
        name: duplexDownloadStep === 1 ? 'ASR' : duplexDownloadStep === 2 ? 'TTS' : 'LLM',
        step: String(duplexDownloadStep),
      })
    : '';

  // Display name for header
  const displayName = isDuplex
    ? packages.find((p) => p.tier === tier)?.model?.display_name || ''
    : setupInfo?.model_display_name || '';

  return (
    <WizardShell
      steps={WIZARD_STEPS_V2(t)}
      currentStep={3}
      onBack={() => navigate('/simple/tier')}
      helpKey="simple.v2.help.setup"
      onNext={isReady ? handleNext : undefined}
      nextLabel={t('simple.v2.setup.satisfied')}
      nextDisabled={!isReady}
    >
      {/* Download phase */}
      {(setupPhase === 'downloading' || setupPhase === 'loading' || setupPhase === 'idle') && !isReady && (
        <div className="space-y-6">
          <div className="text-center">
            <h1 className="mb-2 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
              {t('simple.v2.setup.title')}
            </h1>
            <p className="text-stone-500 dark:text-stone-400">
              {displayName}
            </p>
          </div>

          {/* Duplex step indicators */}
          {isDuplex && duplexDownloadStep > 0 && setupPhase === 'downloading' && (
            <div className="flex items-center justify-center gap-4 text-xs">
              {['ASR', 'TTS', 'LLM'].map((label, i) => {
                const step = i + 1;
                const done = duplexDownloadStep > step || (duplexDownloadStep === step && !terminalSessionId && setupPhase !== 'downloading');
                const active = duplexDownloadStep === step && setupPhase === 'downloading';
                return (
                  <div key={label} className={cn(
                    'flex items-center gap-1',
                    done ? 'text-green-600 dark:text-green-400'
                      : active ? 'text-stone-900 font-medium dark:text-stone-100'
                        : 'text-stone-400',
                  )}>
                    {done ? <Check size={12} /> : active ? <Loader2 size={12} className="animate-spin" /> : <span className="w-3" />}
                    {label}
                  </div>
                );
              })}
            </div>
          )}

          {/* Preparing: probing mirror, creating terminal session */}
          {preparing && setupPhase === 'downloading' && !terminalSessionId && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-stone-500 dark:text-stone-400">
              <Loader2 size={16} className="animate-spin" />
              <span>{t('simple.v2.setup.preparing')}</span>
            </div>
          )}

          {/* Terminal download — inline embedded terminal */}
          {terminalSessionId && setupPhase === 'downloading' && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-stone-500 dark:text-stone-400">
                  {duplexStepLabel || t('simple.v2.setup.downloading', { name: displayName })}
                </p>
                <button
                  type="button"
                  onClick={handleTerminalClose}
                  className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-stone-500 transition-colors hover:bg-stone-100 hover:text-stone-700 dark:text-stone-400 dark:hover:bg-stone-800"
                >
                  <X size={14} />
                  {t('simple.v2.setup.cancel')}
                </button>
              </div>
              <Terminal
                sessionId={terminalSessionId}
                onExit={handleTerminalExit}
                onError={handleTerminalError}
                className="h-[400px]"
              />
            </div>
          )}

          {/* Reconnecting to a terminal session after page refresh */}
          {setupPhase === 'downloading' && !preparing && !terminalSessionId && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-stone-500 dark:text-stone-400">
              <Loader2 size={16} className="animate-spin" />
              <span>{t('simple.v2.setup.reconnecting')}</span>
            </div>
          )}

          {/* Loading spinner (after terminal download completes) */}
          {setupPhase === 'loading' && (
            <div className="space-y-3">
              <div className="flex items-center justify-center gap-2 text-sm text-stone-500 dark:text-stone-400">
                <Loader2 size={14} className="animate-spin" />
                <span>{t('simple.v2.setup.loading')}</span>
              </div>
            </div>
          )}

          {/* Retry on idle with error */}
          {setupPhase === 'idle' && error && (
            <div className="space-y-3">
              <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-center text-sm text-red-700 dark:border-red-900 dark:bg-red-900/20 dark:text-red-400">
                {error}
              </div>
              <div className="text-center">
                <button
                  type="button"
                  onClick={isDuplex ? startTerminalDuplexDownload : startTerminalSingleDownload}
                  className="rounded-xl bg-stone-900 px-6 py-3 text-sm font-medium text-white transition-all hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900"
                >
                  {t('simple.error.retry')}
                </button>
              </div>
            </div>
          )}

          {/* Idle: waiting for setupInfo (no error) */}
          {setupPhase === 'idle' && !error && !terminalSessionId && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-stone-400">
              <Loader2 size={14} className="animate-spin" />
            </div>
          )}

          {/* Tips carousel with fade transition (show during terminal download) */}
          {setupPhase === 'downloading' && terminalSessionId && (
            <div
              className={cn(
                'mx-auto max-w-sm rounded-xl bg-stone-50 p-4 dark:bg-stone-900',
                'transition-opacity duration-300',
                tipVisible ? 'opacity-100' : 'opacity-0',
              )}
            >
              <div className="flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-stone-200/60 dark:bg-stone-800">
                  <TipIcon size={18} className="text-stone-500 dark:text-stone-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-stone-700 dark:text-stone-300">
                    {t(currentTip.titleKey)}
                  </p>
                  <p className="mt-0.5 text-xs text-stone-500 dark:text-stone-400">
                    {t(currentTip.descKey)}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Experience phase */}
      {isReady && (
        <div className="space-y-4">
          <div className="text-center">
            <div className="mb-3 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-green-50 dark:bg-green-900/20">
              <Check size={24} className="text-green-600 dark:text-green-400" />
            </div>
            <h1 className="mb-1 text-2xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
              {t('simple.v2.setup.ready')}
            </h1>
            <p className="text-sm text-stone-500 dark:text-stone-400">
              {displayName}
            </p>
          </div>

          {/* === Focus-specific experience panels === */}

          {focus === 'asr' ? (
            <ASRPanel modelId={loadedModelId} />
          ) : focus === 'tts' ? (
            <TTSPanel modelId={loadedModelId} />
          ) : focus === 'voice_duplex' ? (
            <>
              {/* Voice config for duplex */}
              {ttsVariant === 'customvoice' && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-stone-700 dark:text-stone-300">
                    {t('simple.v2.setup.speakerLabel')}
                  </p>
                  {speakersLoading ? (
                    <div className="flex items-center gap-2 text-xs text-stone-400">
                      <Loader2 size={12} className="animate-spin" />
                      {t('simple.v2.setup.ttsLoading')}
                    </div>
                  ) : speakerList.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {speakerList.map((spk) => (
                        <button
                          key={spk}
                          type="button"
                          onClick={() => setDuplexVoice(spk)}
                          className={cn(
                            'rounded-full px-3 py-1.5 text-xs transition-all',
                            duplexVoice === spk
                              ? 'bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
                              : 'bg-stone-100 text-stone-600 hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-400 dark:hover:bg-stone-700',
                          )}
                        >
                          {spk}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-stone-400">{t('simple.v2.setup.ttsNotAvailable')}</p>
                  )}
                </div>
              )}
              {ttsVariant === 'voicedesign' && (
                <div className="space-y-2">
                  <p className="text-sm font-medium text-stone-700 dark:text-stone-300">
                    {t('simple.v2.setup.ttsInstruct')}
                  </p>
                  <textarea
                    value={duplexInstruct}
                    onChange={(e) => setDuplexInstruct(e.target.value)}
                    placeholder={t('simple.v2.setup.ttsInstructPlaceholder')}
                    rows={3}
                    className="w-full rounded-xl border border-stone-200 bg-white p-3 text-sm text-stone-800 placeholder-stone-400 outline-none transition-colors focus:border-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-200 dark:placeholder-stone-500"
                  />
                </div>
              )}
              <DuplexPanel
                asrModelId={duplexAsrModelId}
                llmModelId={duplexLlmModelId}
                ttsModelId={ttsModelId}
                voice={ttsVariant === 'customvoice' ? duplexVoice : undefined}
                instruct={ttsVariant === 'voicedesign' ? duplexInstruct : undefined}
              />
            </>
          ) : (
            /* Chat/Coding/Vision: text-based experience */
            <>
              {/* Quick prompts */}
              {messages.length === 0 && prompts.length > 0 && (
                <div className="flex flex-wrap justify-center gap-2">
                  {prompts.map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => sendMessage(t(key))}
                      className="rounded-full bg-stone-100 px-4 py-1.5 text-sm text-stone-600 transition-colors hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-400 dark:hover:bg-stone-700"
                    >
                      {t(key)}
                    </button>
                  ))}
                </div>
              )}

              {/* Chat messages */}
              <div className="min-h-[200px] max-h-[50vh] space-y-3 overflow-y-auto rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-950">
                {messages.length === 0 && (
                  <p className="py-8 text-center text-sm text-stone-400">
                    {t('simple.test.empty')}
                  </p>
                )}
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={cn(
                      'max-w-[80%] rounded-2xl px-4 py-2.5 text-sm',
                      msg.role === 'user'
                        ? 'ml-auto bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900'
                        : 'bg-stone-100 text-stone-800 dark:bg-stone-800 dark:text-stone-200',
                    )}
                  >
                    {msg.role === 'assistant' ? (
                      <MarkdownContent content={msg.text || (streaming && i === messages.length - 1 ? '\u2588' : '')} />
                    ) : (
                      <p className="whitespace-pre-wrap">{msg.text || (streaming && i === messages.length - 1 ? '\u2588' : '')}</p>
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>

              {/* VLM image preview */}
              {focus === 'vision' && imageThumb && (
                <div className="flex items-center gap-2">
                  <div className="relative">
                    <img src={imageThumb} alt="" className="h-16 rounded-lg border border-stone-200 object-contain dark:border-stone-700" />
                    <button
                      type="button"
                      onClick={() => { setImageB64(null); setImageThumb(null); }}
                      className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-stone-500 text-white hover:bg-stone-700"
                    >
                      <X size={10} />
                    </button>
                  </div>
                </div>
              )}

              {/* Text input */}
              <div className="flex gap-2">
                {/* VLM: hidden file input + image button */}
                {focus === 'vision' && (
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/jpeg,image/png,image/webp,image/gif"
                      onChange={handleImageSelect}
                      className="hidden"
                    />
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={streaming}
                      title={t('simple.v2.setup.imageUpload')}
                      className={cn(
                        'flex h-10 w-10 shrink-0 items-center justify-center rounded-xl transition-all',
                        streaming
                          ? 'bg-stone-100 text-stone-400 dark:bg-stone-800 dark:text-stone-600'
                          : 'border border-stone-200 text-stone-500 hover:bg-stone-100 dark:border-stone-700 dark:text-stone-400 dark:hover:bg-stone-800',
                      )}
                    >
                      <ImagePlus size={16} />
                    </button>
                  </>
                )}

                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage(input)}
                  placeholder={t('simple.test.placeholder')}
                  disabled={streaming}
                  className="flex-1 rounded-xl border border-stone-200 bg-white py-2.5 pl-4 pr-4 text-sm placeholder-stone-400 outline-none transition-all focus:border-stone-400 focus:ring-1 focus:ring-stone-400 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-100 dark:placeholder-stone-500"
                />
                <button
                  type="button"
                  onClick={() => sendMessage(input)}
                  disabled={(!input.trim() && !(focus === 'vision' && imageB64)) || streaming}
                  className={cn(
                    'flex h-10 w-10 items-center justify-center rounded-xl transition-all',
                    (input.trim() || (focus === 'vision' && imageB64)) && !streaming
                      ? 'bg-stone-900 text-white hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900'
                      : 'bg-stone-100 text-stone-400 dark:bg-stone-800 dark:text-stone-600',
                  )}
                >
                  <Send size={16} />
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </WizardShell>
  );
}
