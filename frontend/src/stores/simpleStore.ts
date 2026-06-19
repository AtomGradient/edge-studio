// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Simple mode store — tracks the "2 clicks + auto download" wizard state.
 * Replaces wizardStore.ts for the new Simple mode UX.
 * Persisted to localStorage so users can resume where they left off.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ---------------------------------------------------------------------------
// Types (match backend schemas/simple.py)
// ---------------------------------------------------------------------------

export interface DeviceProfile {
  chip: string;
  ram_gb: number;
  gpu_cores: number;
  max_model_size_gb: number;
  ai_rating: string;        // "air" | "standard" | "pro" | "max" | "ultra"
  ai_rating_label: string;  // i18n key
  ai_rating_stars: number;  // 1-5
  available_tiers: string[];
  recommended_tier: string;
}

export interface PackageModel {
  catalog_id: string;
  display_name: string;
  family: string;
  category: string;
  params_b: number;
  size_gb: number;
  quant: string;
  download_hint: string;
}

export interface Package {
  tier: string;
  tier_label: string;
  available: boolean;
  unavailable_reason: string;
  download_size_gb: number;
  setup_time_hint: string;
  capabilities: string[];
  model: PackageModel | null;
  secondary_model: PackageModel | null;   // TTS in duplex mode
  tertiary_model: PackageModel | null;    // ASR in duplex mode
}

export interface SetupInfo {
  model_display_name: string;
  download_hint: string;
  size_gb: number;
  already_downloaded: boolean;
  local_dir: string;
}

export interface ExportCheck {
  fits: boolean;
  suggestion: string;  // "direct" | "downgrade" | "change_focus"
  suggested_tier: string;
  reason: string;
  needs_download: boolean;
  download_size_gb: number;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

type SetupPhase = 'idle' | 'downloading' | 'loading' | 'ready';
type ExportPhase = 'idle' | 'checking' | 'adapting' | 'exporting' | 'done';

interface SimpleState {
  // --- Phase 1: "AI your device" ---

  /** Detected device profile */
  deviceProfile: DeviceProfile | null;
  setDeviceProfile: (profile: DeviceProfile) => void;

  /** Selected focus: chat | coding | vision | asr | tts | voice_duplex */
  focus: string;
  setFocus: (focus: string) => void;

  /** Selected tier or empty for custom model */
  tier: string;
  setTier: (tier: string) => void;

  /** Custom model catalog ID (when user picks from "browse all") */
  customModelId: string;
  setCustomModelId: (id: string) => void;

  /** Available packages for current focus */
  packages: Package[];
  setPackages: (packages: Package[]) => void;

  /** Resolved model info for download/load (primary model) */
  setupInfo: SetupInfo | null;
  setSetupInfo: (info: SetupInfo | null) => void;

  /** Active download task ID */
  downloadTaskId: string;
  setDownloadTaskId: (id: string) => void;

  /** Setup progress phase */
  setupPhase: SetupPhase;
  setSetupPhase: (phase: SetupPhase) => void;

  /** Loaded primary model info (chat/coding/vision/asr/tts single-model focuses) */
  loadedModelId: string;
  setLoadedModelId: (id: string) => void;

  loadedModelDir: string;
  setLoadedModelDir: (dir: string) => void;

  /** Whether user has tested the chat */
  chatTested: boolean;
  setChatTested: (tested: boolean) => void;

  /** TTS model (used by standalone TTS focus AND voice_duplex) */
  ttsModelId: string;
  setTtsModelId: (id: string) => void;

  ttsModelDir: string;
  setTtsModelDir: (dir: string) => void;

  /** Duplex-specific: LLM brain model */
  duplexLlmModelId: string;
  setDuplexLlmModelId: (id: string) => void;

  duplexLlmModelDir: string;
  setDuplexLlmModelDir: (dir: string) => void;

  /** Duplex-specific: ASR model */
  duplexAsrModelId: string;
  setDuplexAsrModelId: (id: string) => void;

  duplexAsrModelDir: string;
  setDuplexAsrModelDir: (dir: string) => void;

  /** Duplex download step: 0=not started, 1=ASR, 2=TTS, 3=LLM */
  duplexDownloadStep: number;
  setDuplexDownloadStep: (step: number) => void;

  /** TTS variant for duplex: "customvoice" | "voicedesign" | "" */
  ttsVariant: string;
  setTtsVariant: (v: string) => void;

  /** CustomVoice: selected speaker name */
  duplexVoice: string;
  setDuplexVoice: (v: string) => void;

  /** VoiceDesign: instruct prompt */
  duplexInstruct: string;
  setDuplexInstruct: (v: string) => void;

  // --- Phase 2: "Make it an App" ---

  /** Target device for export */
  targetDevice: string;
  setTargetDevice: (device: string) => void;

  /** Export compatibility check result */
  exportCheck: ExportCheck | null;
  setExportCheck: (check: ExportCheck | null) => void;

  /** Export task ID */
  exportTaskId: string;
  setExportTaskId: (id: string) => void;

  /** Export progress phase */
  exportPhase: ExportPhase;
  setExportPhase: (phase: ExportPhase) => void;

  /** User-editable App name */
  appName: string;
  setAppName: (name: string) => void;

  /** Download URL for exported ZIP */
  downloadUrl: string;
  setDownloadUrl: (url: string) => void;

  // --- Actions ---

  /** Reset all state */
  reset: () => void;

  /** Reset only export state (Phase 2) */
  resetExport: () => void;
}

const initialPhase1 = {
  deviceProfile: null as DeviceProfile | null,
  focus: '',
  tier: '',
  customModelId: '',
  packages: [] as Package[],
  setupInfo: null as SetupInfo | null,
  downloadTaskId: '',
  setupPhase: 'idle' as SetupPhase,
  loadedModelId: '',
  loadedModelDir: '',
  chatTested: false,
  ttsModelId: '',
  ttsModelDir: '',
  duplexLlmModelId: '',
  duplexLlmModelDir: '',
  duplexAsrModelId: '',
  duplexAsrModelDir: '',
  duplexDownloadStep: 0,
  ttsVariant: '',
  duplexVoice: '',
  duplexInstruct: '',
};

const initialPhase2 = {
  targetDevice: '',
  exportCheck: null as ExportCheck | null,
  exportTaskId: '',
  exportPhase: 'idle' as ExportPhase,
  appName: '',
  downloadUrl: '',
};

const initialState = { ...initialPhase1, ...initialPhase2 };

export const useSimpleStore = create<SimpleState>()(
  persist(
    (set) => ({
      ...initialState,

      // Phase 1 setters
      setDeviceProfile: (profile) => set({ deviceProfile: profile }),
      setFocus: (focus) => set({
        focus,
        // Clear ALL stale state when switching focus — including loaded models,
        // because the old model is for a different category (e.g. LLM vs ASR)
        tier: '',
        customModelId: '',
        packages: [],
        setupInfo: null,
        setupPhase: 'idle',
        loadedModelId: '',
        loadedModelDir: '',
        chatTested: false,
        ttsModelId: '',
        ttsModelDir: '',
        duplexLlmModelId: '',
        duplexLlmModelDir: '',
        duplexAsrModelId: '',
        duplexAsrModelDir: '',
        duplexDownloadStep: 0,
        ttsVariant: '',
        duplexVoice: '',
        duplexInstruct: '',
      }),
      setTier: (tier) => set({ tier }),
      setCustomModelId: (id) => set({ customModelId: id }),
      setPackages: (packages) => set({ packages }),
      setSetupInfo: (info) => set({ setupInfo: info }),
      setDownloadTaskId: (id) => set({ downloadTaskId: id }),
      setSetupPhase: (phase) => set({ setupPhase: phase }),
      setLoadedModelId: (id) => set({ loadedModelId: id }),
      setLoadedModelDir: (dir) => set({ loadedModelDir: dir }),
      setChatTested: (tested) => set({ chatTested: tested }),
      setTtsModelId: (id) => set({ ttsModelId: id }),
      setTtsModelDir: (dir) => set({ ttsModelDir: dir }),
      setDuplexLlmModelId: (id) => set({ duplexLlmModelId: id }),
      setDuplexLlmModelDir: (dir) => set({ duplexLlmModelDir: dir }),
      setDuplexAsrModelId: (id) => set({ duplexAsrModelId: id }),
      setDuplexAsrModelDir: (dir) => set({ duplexAsrModelDir: dir }),
      setDuplexDownloadStep: (step) => set({ duplexDownloadStep: step }),
      setTtsVariant: (v) => set({ ttsVariant: v }),
      setDuplexVoice: (v) => set({ duplexVoice: v }),
      setDuplexInstruct: (v) => set({ duplexInstruct: v }),

      // Phase 2 setters
      setTargetDevice: (device) => set({ targetDevice: device }),
      setExportCheck: (check) => set({ exportCheck: check }),
      setExportTaskId: (id) => set({ exportTaskId: id }),
      setExportPhase: (phase) => set({ exportPhase: phase }),
      setAppName: (name) => set({ appName: name }),
      setDownloadUrl: (url) => set({ downloadUrl: url }),

      // Actions
      reset: () => set(initialState),
      resetExport: () => set(initialPhase2),
    }),
    {
      name: 'edge-simple',
      version: 3,
      partialize: (state) => ({
        // Persist selections and progress — NOT deviceProfile (re-detect per session)
        focus: state.focus,
        tier: state.tier,
        customModelId: state.customModelId,
        setupInfo: state.setupInfo,
        downloadTaskId: state.downloadTaskId,
        setupPhase: state.setupPhase,
        loadedModelId: state.loadedModelId,
        loadedModelDir: state.loadedModelDir,
        chatTested: state.chatTested,
        ttsModelId: state.ttsModelId,
        ttsModelDir: state.ttsModelDir,
        duplexLlmModelId: state.duplexLlmModelId,
        duplexLlmModelDir: state.duplexLlmModelDir,
        duplexAsrModelId: state.duplexAsrModelId,
        duplexAsrModelDir: state.duplexAsrModelDir,
        ttsVariant: state.ttsVariant,
        duplexVoice: state.duplexVoice,
        duplexInstruct: state.duplexInstruct,
        targetDevice: state.targetDevice,
        exportPhase: state.exportPhase,
        appName: state.appName,
      }),
      migrate: (persisted: unknown, version: number) => {
        const state = persisted && typeof persisted === 'object'
          ? persisted as Partial<SimpleState>
          : {};
        if (version < 2) {
          // v1 → v2: "voice" focus renamed to "asr"
          if (state.focus === 'voice') {
            state.focus = 'asr';
          }
          // Initialize new duplex fields
          state.duplexLlmModelId = '';
          state.duplexLlmModelDir = '';
          state.duplexAsrModelId = '';
          state.duplexAsrModelDir = '';
        }
        if (version < 3) {
          // v2 → v3: TTS variant fields for duplex
          state.ttsVariant = state.ttsVariant || '';
          state.duplexVoice = state.duplexVoice || '';
          state.duplexInstruct = state.duplexInstruct || '';
        }
        return state;
      },
    },
  ),
);

// One-time migration: remove legacy localStorage key from v1 wizard
if (typeof window !== 'undefined' && localStorage.getItem('vlm-wizard')) {
  localStorage.removeItem('vlm-wizard');
}
