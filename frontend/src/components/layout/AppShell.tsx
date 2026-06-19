// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { CommandPalette } from '@/components/common/CommandPalette';
import { GlobalAskModel } from '@/components/common/GlobalAskModel';
import { useUIStore } from '@/stores/uiStore';
import { useModelStore } from '@/stores/modelStore';
import { Moon, Sun, Globe, GraduationCap, Compass, Zap, Microscope, Search } from 'lucide-react';
import { useTourStore } from '@/stores/tourStore';
import { cn } from '@/lib/utils';
import { useState, useEffect } from 'react';
import { useLocaleStore, useT } from '@/i18n';

const PAGE_TITLE_KEYS: Record<string, string> = {
  '/dashboard': 'nav.dashboard',
  '/architecture': 'nav.architecture',
  '/weights': 'nav.weights',
  '/activation': 'nav.activations',
  '/pruning': 'nav.pruning',
  '/inference': 'nav.inference',
  '/chat': 'nav.chat',
  '/attention': 'nav.attention',
  '/quality': 'nav.quality',
  '/kv-cache': 'nav.kvCache',
  '/optimization': 'nav.optimizer',
  '/auto-optimizer': 'nav.autoOptimizer',
  '/pipeline': 'nav.pipeline',
  '/moe': 'nav.moe',
  '/comparison': 'nav.comparison',
  '/export': 'nav.export',
  '/mixed-precision': 'nav.mixedPrecision',
  '/benchmark-dashboard': 'nav.benchmarkDashboard',
  '/batch': 'nav.batch',
  '/devices': 'nav.devices',
  '/joint-inference': 'nav.jointInference',
  '/neural-imprint': 'nav.neuralImprint',
  '/neural-imprint-chat': 'nav.neuralImprintChat',
  '/rpp-results': 'nav.rppResults',
  '/a-library': 'nav.aLibrary',
};

export function AppShell() {
  const { sidebarOpen, toggleSidebar, darkMode, toggleDarkMode, userMode, setUserMode } = useUIStore();
  const navigate = useNavigate();
  const modelDir = useModelStore((s) => s.currentModel?.model_dir);
  const modelDisplayName = modelDir ? modelDir.split('/').pop() ?? '' : '';
  const { locale, setLocale } = useLocaleStore();
  const startTour = useTourStore((s) => s.startTour);
  const t = useT();
  const location = useLocation();
  const titleKey = PAGE_TITLE_KEYS[location.pathname];
  const pageTitle = titleKey ? t(titleKey) : '';
  const [cmdPaletteOpen, setCmdPaletteOpen] = useState(false);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
      if (meta && e.key === 'o') {
        e.preventDefault();
        useUIStore.getState().setFileBrowserOpen(true);
      }
      if (meta && e.key === 'k') {
        e.preventDefault();
        setCmdPaletteOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [toggleSidebar]);

  return (
    <div className={cn('flex min-h-screen', darkMode ? 'dark bg-stone-950' : 'bg-stone-50')}>
      <Sidebar />

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-25 bg-black/30 md:hidden"
          onClick={toggleSidebar}
        />
      )}

      {/* Main content */}
      <main className={cn(
        'flex-1 transition-all duration-200',
        sidebarOpen
          ? 'ml-[var(--sidebar-width)]'
          : 'ml-[var(--sidebar-collapsed-width)]',
      )}>
        {/* Top bar */}
        <header className="sticky top-0 z-20 flex items-center justify-between border-b border-stone-200 bg-white/80 px-6 py-2.5 backdrop-blur-sm dark:border-stone-800 dark:bg-stone-950/80">
          <div className="flex items-center gap-3">
            {modelDisplayName && (
              <div className="flex items-center gap-2 text-sm">
                <span className="font-semibold text-stone-900 dark:text-stone-100">{modelDisplayName}</span>
                {pageTitle && (
                  <>
                    <span className="text-stone-300 dark:text-stone-600">/</span>
                    <span className="text-stone-500 dark:text-stone-400">{pageTitle}</span>
                  </>
                )}
              </div>
            )}
            {!modelDisplayName && pageTitle && (
              <span className="text-sm font-medium text-stone-500 dark:text-stone-400">{pageTitle}</span>
            )}
          </div>
          <div className="flex items-center gap-0.5">
            <button
              onClick={() => setCmdPaletteOpen(true)}
              className="flex items-center gap-1.5 rounded-lg border border-stone-200 px-2.5 py-1 text-xs text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-600 dark:border-stone-700 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              title="Command Palette (Cmd+K)"
            >
              <Search size={12} />
              <kbd className="text-[10px] font-medium">⌘K</kbd>
            </button>
            <button
              onClick={startTour}
              className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              title={t('tour.startTour')}
            >
              <GraduationCap size={14} />
            </button>
            {/* Three-tier mode selector */}
            <div className="flex items-center rounded-lg border border-stone-200 dark:border-stone-700">
              {([
                { mode: 'beginner' as const, icon: Compass, label: t('mode.beginner') },
                { mode: 'simple' as const, icon: Zap, label: t('mode.simple') },
                { mode: 'advanced' as const, icon: Microscope, label: t('mode.advanced') },
              ]).map(({ mode, icon: Icon, label }) => (
                <button
                  key={mode}
                  onClick={() => {
                    setUserMode(mode);
                    if (mode === 'beginner') navigate('/simple');
                  }}
                  className={cn(
                    'flex items-center gap-1 px-2.5 py-1 text-xs font-medium transition-colors first:rounded-l-md last:rounded-r-md',
                    userMode === mode
                      ? 'bg-stone-100 text-stone-800 dark:bg-stone-800 dark:text-stone-200'
                      : 'text-stone-400 hover:text-stone-600 dark:text-stone-500 dark:hover:text-stone-300',
                  )}
                >
                  <Icon size={13} />
                  {label}
                </button>
              ))}
            </div>
            <button
              onClick={() => setLocale(locale === 'en' ? 'zh' : 'en')}
              className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              title={t('settings.language')}
            >
              <Globe size={14} />
              {locale === 'en' ? 'EN' : '中'}
            </button>
            <button
              onClick={toggleDarkMode}
              className="rounded-lg p-1.5 text-stone-400 transition-colors hover:bg-stone-100 hover:text-stone-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
              title="Toggle dark mode"
            >
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>

        {/* Page content */}
        <div className="p-6">
          <Outlet />
        </div>
        <GlobalAskModel />
      </main>
      <CommandPalette open={cmdPaletteOpen} onClose={() => setCmdPaletteOpen(false)} />
    </div>
  );
}
