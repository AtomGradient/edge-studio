// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import {
  FolderOpen, Upload, GitCompare, Loader2, X,
  LayoutDashboard, Boxes, Scale, Activity, Scissors, Play, Eye, CheckCircle, Database,
  Lightbulb, Sparkles, Network, Download, ArrowRightLeft, Workflow,
  PanelLeftClose, MessageCircle, Mic, GraduationCap, GitMerge, Gauge,
  Layers, BarChart3, Zap, Smartphone, Fingerprint,
  BrainCircuit,
} from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { useToastStore } from '@/stores/toastStore';
import { useUIStore } from '@/stores/uiStore';
import { loadModel, unloadModel } from '@/api/endpoints';
import { ModelInfoCard } from '@/components/model/ModelInfoCard';
import { FileBrowser } from '@/components/model/FileBrowser';
import { HFModelPicker } from '@/components/model/HFModelPicker';
import { ProfileLoader } from '@/components/model/ProfileLoader';
import { NavSection } from './NavSection';
import type { NavItem } from './NavSection';
import { MemoryBar } from '@/components/common/MemoryBar';
import { cn } from '@/lib/utils';
import { useT } from '@/i18n';

export function Sidebar() {
  const {
    currentModel, setCurrentModel,
    profileSummary,
    hasTrace, hasAttentionTrace,
    comparisonModel, setComparisonModel,
  } = useModelStore();
  const { sidebarOpen, toggleSidebar, fileBrowserOpen, setFileBrowserOpen, hfPickerOpen, setHfPickerOpen } = useUIStore();
  const addToast = useToastStore((s) => s.addToast);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profileLoaderOpen, setProfileLoaderOpen] = useState(false);
  const [compareBrowserOpen, setCompareBrowserOpen] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareHintDismissed, setCompareHintDismissed] = useState(
    () => localStorage.getItem('edge_compare_hint_dismissed') === '1',
  );

  const showCompareHint = !!currentModel && !comparisonModel && !compareHintDismissed;

  const handleLoadModel = async (path: string) => {
    setFileBrowserOpen(false);
    setLoading(true);
    setError(null);
    try {
      const info = await loadModel(path);
      setCurrentModel(info);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load model';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleUnload = async () => {
    if (!currentModel) return;
    setLoading(true);
    try {
      await unloadModel(currentModel.model_id);
      setCurrentModel(null);
      if (comparisonModel?.model_id === currentModel.model_id) {
        setComparisonModel(null);
      }
    } catch {
      addToast('Failed to unload model from backend.', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveComparison = async () => {
    if (!comparisonModel) return;
    setCompareLoading(true);
    try {
      if (comparisonModel.model_id !== currentModel?.model_id) {
        await unloadModel(comparisonModel.model_id);
      }
      setComparisonModel(null);
    } catch {
      addToast('Failed to unload comparison model from backend.', 'error');
    } finally {
      setCompareLoading(false);
    }
  };

  const handleLoadComparison = async (path: string) => {
    setCompareBrowserOpen(false);
    setCompareLoading(true);
    try {
      const info = await loadModel(path);
      setComparisonModel(info);
    } catch {
      addToast('Failed to load comparison model.', 'error');
    } finally {
      setCompareLoading(false);
    }
  };

  const t = useT();
  const userMode = useUIStore((s) => s.userMode);
  const isSimple = userMode === 'simple';
  const SIMPLE_PATHS = new Set(['/', '/dashboard', '/chat', '/pipeline', '/export']);
  const hasModel = !!currentModel;
  const isGGUF = currentModel?.is_gguf ?? false;
  const hasProfile = !!profileSummary;
  const hasMOE = currentModel?.has_moe ?? false;
  const hasComparison = !!comparisonModel;

  const hideIfSimple = (path: string) => isSimple && !SIMPLE_PATHS.has(path);

  const overviewItems: NavItem[] = [
    { path: '/dashboard', label: t('nav.dashboard'), icon: LayoutDashboard, available: hasModel },
  ];

  const analysisItems: NavItem[] = [
    { path: '/architecture', label: t('nav.architecture'), icon: Boxes, available: hasModel, hidden: hideIfSimple('/architecture') },
    { path: '/weights', label: t('nav.weights'), icon: Scale, available: hasModel, hidden: hideIfSimple('/weights') },
    { path: '/activation', label: t('nav.activations'), icon: Activity, available: hasProfile && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : 'Load activation profile first', hidden: hideIfSimple('/activation') },
    { path: '/pruning', label: t('nav.pruning'), icon: Scissors, available: hasProfile && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : 'Load activation profile first', hidden: hideIfSimple('/pruning') },
    { path: '/inference', label: t('nav.inference'), icon: Play, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : undefined, hidden: hideIfSimple('/inference') },
    { path: '/chat', label: t('nav.chat'), icon: MessageCircle, available: hasModel },
    { path: '/duplex', label: t('nav.duplex'), icon: Mic, available: true, hidden: hideIfSimple('/duplex'), comingSoon: true },
    { path: '/attention', label: t('nav.attention'), icon: Eye, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : !hasAttentionTrace ? 'No attention trace yet — open page to see how to capture' : undefined, hidden: hideIfSimple('/attention') },
    { path: '/quality', label: t('nav.quality'), icon: CheckCircle, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : undefined, hidden: hideIfSimple('/quality') },
    { path: '/kv-cache', label: t('nav.kvCache'), icon: Database, available: hasModel, hidden: hideIfSimple('/kv-cache') },
  ];

  const optimizeItems: NavItem[] = [
    { path: '/optimization', label: t('nav.optimizer'), icon: Lightbulb, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : undefined, hidden: hideIfSimple('/optimization') },
    { path: '/pipeline', label: t('nav.pipeline'), icon: Workflow, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : undefined },
    { path: '/auto-optimizer', label: t('nav.autoOptimizer'), icon: Sparkles, available: hasProfile && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : 'Load activation profile first', hidden: hideIfSimple('/auto-optimizer') },
    { path: '/distill', label: t('nav.distill'), icon: GraduationCap, available: true },
    { path: '/merge', label: t('nav.merge'), icon: GitMerge, available: true },
    { path: '/auto-tune', label: t('nav.autoTune'), icon: Gauge, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : undefined },
    { path: '/moe', label: t('nav.moe'), icon: Network, available: hasModel && !isGGUF, hidden: hideIfSimple('/moe'), tooltip: isGGUF ? 'Not available for GGUF' : !hasMOE ? 'Loaded model is dense — open page to see MoE explanation' : !hasTrace ? 'No trace yet — open page to see how to capture routing' : undefined },
    { path: '/mixed-precision', label: t('nav.mixedPrecision'), icon: Layers, available: hasModel && !isGGUF, hidden: hideIfSimple('/mixed-precision'), tooltip: isGGUF ? 'Not available for GGUF' : undefined },
  ];

  const batchItems: NavItem[] = [
    { path: '/benchmark-dashboard', label: t('nav.benchmarkDashboard'), icon: BarChart3, available: true },
    { path: '/batch', label: t('nav.batch'), icon: Zap, available: true },
  ];

  const trainingItems: NavItem[] = [
    { path: '/devices', label: t('nav.devices'), icon: Smartphone, available: true, hidden: hideIfSimple('/devices') },
    { path: '/joint-inference', label: t('nav.jointInference'), icon: BrainCircuit, available: true, hidden: hideIfSimple('/joint-inference') },
    { path: '/neural-imprint', label: t('nav.neuralImprint'), icon: Fingerprint, available: true, hidden: hideIfSimple('/neural-imprint') },
    { path: '/neural-imprint-chat', label: t('nav.neuralImprintChat'), icon: MessageCircle, available: true, hidden: hideIfSimple('/neural-imprint-chat') },
    { path: '/rpp-results', label: t('nav.rppResults'), icon: BarChart3, available: true, hidden: hideIfSimple('/rpp-results') },
    { path: '/a-library', label: t('nav.aLibrary'), icon: Database, available: true, hidden: hideIfSimple('/a-library') },
  ];

  const deployItems: NavItem[] = [
    { path: '/export', label: t('nav.export'), icon: Download, available: hasModel && !isGGUF, tooltip: isGGUF ? 'Not available for GGUF' : undefined },
    { path: '/comparison', label: t('nav.comparison'), icon: ArrowRightLeft, available: hasComparison, tooltip: 'Load a second model first', hidden: hideIfSimple('/comparison') },
  ];

  const collapsed = !sidebarOpen;

  return (
    <>
      <aside className={cn(
        'fixed left-0 top-0 z-30 flex h-full flex-col border-r border-stone-200 bg-white transition-all duration-200 overflow-hidden dark:border-stone-800 dark:bg-stone-950',
        sidebarOpen ? 'w-[var(--sidebar-width)]' : 'w-[var(--sidebar-collapsed-width)]',
      )}>
        {/* Logo / collapsed toggle */}
        <div className={cn(
          'flex items-center border-b border-stone-200 py-4 dark:border-stone-800',
          collapsed ? 'justify-center px-2' : 'gap-2 px-4',
        )}>
          {collapsed ? (
            <button
              onClick={toggleSidebar}
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-stone-900 text-white text-sm font-bold hover:bg-stone-800 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
              title="Expand sidebar (Cmd+B)"
            >
              E
            </button>
          ) : (
            <>
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-stone-900 text-white text-sm font-bold dark:bg-stone-100 dark:text-stone-900">
                E
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="truncate text-sm font-semibold text-stone-900 dark:text-stone-100">Edge Studio</h1>
                <p className="text-[10px] text-stone-400 dark:text-stone-500">{t('app.subtitle')}</p>
              </div>
              <button
                onClick={toggleSidebar}
                className="shrink-0 rounded-lg p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
                title="Collapse sidebar (Cmd+B)"
              >
                <PanelLeftClose size={16} />
              </button>
            </>
          )}
        </div>

        {/* Model controls — expanded only */}
        {!collapsed && (
          <div className="border-b border-stone-200 px-4 py-4 dark:border-stone-800">
            <div className="flex gap-2">
              <button
                onClick={() => setFileBrowserOpen(true)}
                disabled={loading}
                className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-stone-900 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-stone-800 disabled:opacity-50 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-200"
              >
                {loading ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <FolderOpen size={16} />
                )}
                {loading ? t('sidebar.loading') : t('sidebar.openModel')}
              </button>
              <button
                onClick={() => setHfPickerOpen(true)}
                className="flex items-center justify-center gap-1.5 rounded-lg border border-stone-200 px-3 py-2 text-sm font-medium text-stone-700 transition-colors hover:bg-stone-50 dark:border-stone-700 dark:text-stone-300 dark:hover:bg-stone-800"
                title="Model Hub — browse local & download from HuggingFace"
              >
                <Download size={16} />
                {t('sidebar.hub')}
              </button>
            </div>

            {error && (
              <p className="mt-2 text-xs text-red-500">{error}</p>
            )}

            {currentModel && (
              <div className="mt-3">
                <ModelInfoCard model={currentModel} onUnload={handleUnload} />
              </div>
            )}
          </div>
        )}

        {/* Collapsed: open model icon button */}
        {collapsed && (
          <div className="flex justify-center border-b border-stone-200 py-3 dark:border-stone-800">
            <button
              onClick={() => setFileBrowserOpen(true)}
              disabled={loading}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-stone-500 hover:bg-stone-100 disabled:opacity-50 dark:text-stone-400 dark:hover:bg-stone-800"
              title="Open Model..."
            >
              {loading ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <FolderOpen size={18} />
              )}
            </button>
          </div>
        )}

        {/* Navigation */}
        <div className={cn(
          'flex-1 overflow-y-auto py-4',
          collapsed ? 'px-1' : 'px-2',
        )}>
          <NavSection title="" items={overviewItems} collapsed={collapsed} />
          <NavSection title={t('nav.analysis')} items={analysisItems} collapsed={collapsed} />
          <NavSection title={t('nav.optimize')} items={optimizeItems} collapsed={collapsed} />
          {!isSimple && <NavSection title={t('nav.batchGroup')} items={batchItems} collapsed={collapsed} />}
          <NavSection title={t('nav.training')} items={trainingItems} collapsed={collapsed} />
          <NavSection title={t('nav.deploy')} items={deployItems} collapsed={collapsed} />
        </div>

        {/* Memory bar */}
        {hasModel && <MemoryBar collapsed={collapsed} />}

        {/* Bottom actions — expanded only, hidden in simple mode */}
        {!collapsed && hasModel && !isSimple && (
          <div className="border-t border-stone-200 px-4 py-3 space-y-2 dark:border-stone-800">
            <button
              onClick={() => setProfileLoaderOpen(true)}
              className="flex w-full items-center gap-2 rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 transition-colors hover:bg-stone-50 dark:border-stone-700 dark:text-stone-400 dark:hover:bg-stone-800"
            >
              <Upload size={14} />
              {profileSummary ? 'Change Profile...' : 'Load Profile...'}
            </button>
            <button
              onClick={() => {
                if (showCompareHint) {
                  setCompareHintDismissed(true);
                  localStorage.setItem('edge_compare_hint_dismissed', '1');
                }
                setCompareBrowserOpen(true);
              }}
              disabled={compareLoading}
              className="relative flex w-full items-center gap-2 rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 transition-colors hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-400 dark:hover:bg-stone-800"
            >
              <GitCompare size={14} />
              {compareLoading ? 'Loading...' : comparisonModel ? 'Change Comparison...' : 'Compare with...'}
              {showCompareHint && (
                <span className="absolute -right-1 -top-1 flex h-3 w-3">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-indigo-400 opacity-75" />
                  <span className="relative inline-flex h-3 w-3 rounded-full bg-indigo-500" />
                </span>
              )}
            </button>
            {comparisonModel && (
              <div className="mt-1 flex items-center justify-between rounded-lg bg-stone-50 px-2 py-1.5 dark:bg-stone-800">
                <span className="truncate text-xs text-stone-600 dark:text-stone-400">{comparisonModel.model_name}</span>
                <button
                  onClick={handleRemoveComparison}
                  className="ml-1 text-stone-400 hover:text-stone-600 dark:text-stone-500 dark:hover:text-stone-300"
                  title="Remove comparison model"
                >
                  <X size={12} />
                </button>
              </div>
            )}
          </div>
        )}

        {/* Collapsed bottom: icon buttons for profile/compare — hidden in simple mode */}
        {collapsed && hasModel && !isSimple && (
          <div className="flex flex-col items-center gap-2 border-t border-stone-200 py-3 dark:border-stone-800">
            <button
              onClick={() => setProfileLoaderOpen(true)}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-stone-500 hover:bg-stone-100 dark:text-stone-400 dark:hover:bg-stone-800"
              title={profileSummary ? 'Change Profile...' : 'Load Profile...'}
            >
              <Upload size={16} />
            </button>
            <button
              onClick={() => setCompareBrowserOpen(true)}
              disabled={compareLoading}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-stone-500 hover:bg-stone-100 disabled:opacity-50 dark:text-stone-400 dark:hover:bg-stone-800"
              title={comparisonModel ? 'Change Comparison...' : 'Compare with...'}
            >
              <GitCompare size={16} />
            </button>
          </div>
        )}
      </aside>

      {/* File browser modal */}
      {fileBrowserOpen && (
        <FileBrowser
          onSelect={handleLoadModel}
          onCancel={() => setFileBrowserOpen(false)}
        />
      )}

      {/* Profile loader modal */}
      {profileLoaderOpen && (
        <ProfileLoader onClose={() => setProfileLoaderOpen(false)} />
      )}

      {/* HuggingFace model picker */}
      {hfPickerOpen && (
        <HFModelPicker onClose={() => setHfPickerOpen(false)} />
      )}

      {/* Comparison model browser */}
      {compareBrowserOpen && (
        <FileBrowser
          onSelect={handleLoadComparison}
          onCancel={() => setCompareBrowserOpen(false)}
        />
      )}
    </>
  );
}
