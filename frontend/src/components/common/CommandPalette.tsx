// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * CommandPalette — Cmd+K to search and navigate to any page or action.
 *
 * Features:
 * - Fuzzy search across all pages
 * - Model actions (load, unload, generate profile)
 * - Settings toggles (dark mode, language)
 * - Keyboard navigation (Arrow up/down, Enter, Escape)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search, LayoutDashboard, Boxes, Scale, Activity, Scissors, Play,
  MessageCircle, Eye, CheckCircle, Database, Lightbulb, Sparkles,
  Network, Download, ArrowRightLeft, Workflow, GraduationCap,
  GitMerge, Gauge, Layers, BarChart3, Zap, FolderOpen,
  Moon, Sun, Globe, Upload,
} from 'lucide-react';
import { useModelStore } from '@/stores/modelStore';
import { useUIStore } from '@/stores/uiStore';
import { useLocaleStore } from '@/i18n';
import { cn } from '@/lib/utils';

interface CommandItem {
  id: string;
  label: string;
  labelZh?: string;
  icon: React.ReactNode;
  category: 'page' | 'action' | 'setting';
  action: () => void;
  available?: boolean;
  keywords?: string;
}

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

function fuzzyMatch(query: string, text: string): boolean {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  if (t.includes(q)) return true;
  // Simple fuzzy: all query chars appear in order
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) qi++;
  }
  return qi === q.length;
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const locale = useLocaleStore((s) => s.locale);

  const model = useModelStore((s) => s.currentModel);
  const hasModel = !!model;
  const isGGUF = model?.is_gguf ?? false;
  const profileSummary = useModelStore((s) => s.profileSummary);

  const nav = useCallback((path: string) => {
    navigate(path);
    onClose();
  }, [navigate, onClose]);

  const items: CommandItem[] = [
    // Pages
    { id: 'dashboard', label: 'Dashboard', labelZh: '仪表盘', icon: <LayoutDashboard size={16} />, category: 'page', action: () => nav('/dashboard'), available: hasModel, keywords: 'home overview' },
    { id: 'architecture', label: 'Architecture', labelZh: '架构浏览', icon: <Boxes size={16} />, category: 'page', action: () => nav('/architecture'), available: hasModel, keywords: 'layers structure tree' },
    { id: 'weights', label: 'Weights', labelZh: '权重分析', icon: <Scale size={16} />, category: 'page', action: () => nav('/weights'), available: hasModel, keywords: 'tensor dtype quantization' },
    { id: 'activation', label: 'Activations', labelZh: '激活热图', icon: <Activity size={16} />, category: 'page', action: () => nav('/activation'), available: hasModel && !isGGUF, keywords: 'heatmap profile neuron' },
    { id: 'pruning', label: 'Pruning Sim', labelZh: '剪枝模拟', icon: <Scissors size={16} />, category: 'page', action: () => nav('/pruning'), available: !!profileSummary && !isGGUF, keywords: 'prune trim dead' },
    { id: 'inference', label: 'Inference', labelZh: '推理追踪', icon: <Play size={16} />, category: 'page', action: () => nav('/inference'), available: hasModel && !isGGUF, keywords: 'trace generate tokens' },
    { id: 'chat', label: 'Chat', labelZh: '聊天', icon: <MessageCircle size={16} />, category: 'page', action: () => nav('/chat'), available: hasModel, keywords: 'conversation talk test' },
    { id: 'attention', label: 'Attention', labelZh: '注意力', icon: <Eye size={16} />, category: 'page', action: () => nav('/attention'), available: hasModel && !isGGUF, keywords: 'heads patterns sink local' },
    { id: 'quality', label: 'Quality', labelZh: '质量验证', icon: <CheckCircle size={16} />, category: 'page', action: () => nav('/quality'), available: hasModel && !isGGUF, keywords: 'perplexity ppl validation' },
    { id: 'kv-cache', label: 'KV Cache', labelZh: 'KV 缓存', icon: <Database size={16} />, category: 'page', action: () => nav('/kv-cache'), available: hasModel, keywords: 'memory context length' },
    { id: 'optimizer', label: 'Optimizer', labelZh: '优化建议', icon: <Lightbulb size={16} />, category: 'page', action: () => nav('/optimization'), available: hasModel && !isGGUF, keywords: 'suggestions advice' },
    { id: 'pipeline', label: 'Pipeline', labelZh: '流水线', icon: <Workflow size={16} />, category: 'page', action: () => nav('/pipeline'), available: hasModel && !isGGUF, keywords: 'optimize quantize prune batch' },
    { id: 'auto-optimizer', label: 'Auto Optimizer', labelZh: '自动优化', icon: <Sparkles size={16} />, category: 'page', action: () => nav('/auto-optimizer'), available: !!profileSummary && !isGGUF, keywords: 'search automatic best' },
    { id: 'mixed-precision', label: 'Mixed Precision', labelZh: '混合精度', icon: <Layers size={16} />, category: 'page', action: () => nav('/mixed-precision'), available: hasModel && !isGGUF, keywords: 'per-layer bits quantization' },
    { id: 'moe', label: 'MOE', labelZh: 'MOE 分析', icon: <Network size={16} />, category: 'page', action: () => nav('/moe'), available: hasModel && (model?.has_moe ?? false), keywords: 'mixture experts routing' },
    { id: 'distill', label: 'Distillation', labelZh: '知识蒸馏', icon: <GraduationCap size={16} />, category: 'page', action: () => nav('/distill'), available: true, keywords: 'teacher student knowledge transfer' },
    { id: 'merge', label: 'Merge', labelZh: '模型合并', icon: <GitMerge size={16} />, category: 'page', action: () => nav('/merge'), available: true, keywords: 'slerp ties linear combine' },
    { id: 'auto-tune', label: 'Auto-Tune', labelZh: '自动调优', icon: <Gauge size={16} />, category: 'page', action: () => nav('/auto-tune'), available: hasModel && !isGGUF, keywords: 'benchmark performance speed' },
    { id: 'benchmark', label: 'Benchmark', labelZh: 'Benchmark 看板', icon: <BarChart3 size={16} />, category: 'page', action: () => nav('/benchmark-dashboard'), available: true, keywords: 'compare performance speed' },
    { id: 'batch', label: 'Batch Ops', labelZh: '批量操作', icon: <Zap size={16} />, category: 'page', action: () => nav('/batch'), available: true, keywords: 'bulk multiple models' },
    { id: 'comparison', label: 'Comparison', labelZh: '对比', icon: <ArrowRightLeft size={16} />, category: 'page', action: () => nav('/comparison'), available: hasModel, keywords: 'diff side-by-side' },
    { id: 'export', label: 'Export', labelZh: '导出', icon: <Download size={16} />, category: 'page', action: () => nav('/export'), available: hasModel && !isGGUF, keywords: 'gguf coreml swift ios app report' },

    // Actions
    { id: 'open-model', label: 'Open Model...', labelZh: '打开模型...', icon: <FolderOpen size={16} />, category: 'action', action: () => { useUIStore.getState().setFileBrowserOpen(true); onClose(); }, keywords: 'load browse file' },
    { id: 'load-profile', label: 'Load Profile...', labelZh: '加载 Profile...', icon: <Upload size={16} />, category: 'action', action: () => { /* trigger profile loader from sidebar state */ onClose(); }, available: hasModel && !isGGUF, keywords: 'activation profile' },

    // Settings
    { id: 'toggle-dark', label: 'Toggle Dark Mode', labelZh: '切换深色模式', icon: useUIStore.getState().darkMode ? <Sun size={16} /> : <Moon size={16} />, category: 'setting', action: () => { useUIStore.getState().toggleDarkMode(); onClose(); }, keywords: 'theme light night' },
    { id: 'toggle-lang', label: 'Switch Language', labelZh: '切换语言', icon: <Globe size={16} />, category: 'setting', action: () => { const store = useLocaleStore.getState(); store.setLocale(store.locale === 'en' ? 'zh' : 'en'); onClose(); }, keywords: 'english chinese 中文 英文' },
  ];

  // Filter by query
  const filtered = query.trim()
    ? items.filter((item) => {
        const searchText = `${item.label} ${item.labelZh || ''} ${item.keywords || ''}`;
        return fuzzyMatch(query, searchText);
      })
    : items;

  // Clamp selection
  useEffect(() => {
    setSelectedIdx(0);
  }, [query]);

  // Focus input on open
  useEffect(() => {
    if (open) {
      setQuery('');
      setSelectedIdx(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  // Scroll selected into view
  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const selected = list.children[selectedIdx] as HTMLElement;
    if (selected) {
      selected.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIdx]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIdx((i) => Math.min(i + 1, filtered.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const item = filtered[selectedIdx];
      if (item) item.action();
    } else if (e.key === 'Escape') {
      onClose();
    }
  }, [filtered, selectedIdx, onClose]);

  if (!open) return null;

  const categoryLabels: Record<string, string> = {
    page: locale === 'zh' ? '页面' : 'Pages',
    action: locale === 'zh' ? '操作' : 'Actions',
    setting: locale === 'zh' ? '设置' : 'Settings',
  };

  // Group by category
  const grouped: { category: string; items: (CommandItem & { globalIdx: number })[] }[] = [];
  let globalIdx = 0;
  const categories = ['page', 'action', 'setting'] as const;
  for (const cat of categories) {
    const catItems = filtered.filter((i) => i.category === cat);
    if (catItems.length > 0) {
      grouped.push({
        category: cat,
        items: catItems.map((item) => ({ ...item, globalIdx: globalIdx++ })),
      });
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="fixed inset-x-0 top-[15%] z-50 mx-auto w-full max-w-lg animate-in fade-in slide-in-from-top-4 duration-200">
        <div className="mx-4 overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl dark:border-stone-700 dark:bg-stone-900">
          {/* Search input */}
          <div className="flex items-center gap-3 border-b border-gray-100 px-4 py-3 dark:border-stone-800">
            <Search size={18} className="shrink-0 text-gray-400 dark:text-stone-500" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={locale === 'zh' ? '搜索页面或操作...' : 'Search pages or actions...'}
              className="flex-1 bg-transparent text-sm text-gray-900 placeholder-gray-400 outline-none dark:text-stone-100 dark:placeholder-stone-500"
              autoComplete="off"
              spellCheck={false}
            />
            <kbd className="hidden sm:inline-flex items-center rounded border border-gray-200 px-1.5 py-0.5 text-[10px] font-medium text-gray-400 dark:border-stone-700 dark:text-stone-500">
              ESC
            </kbd>
          </div>

          {/* Results */}
          <div ref={listRef} className="max-h-80 overflow-y-auto p-2">
            {filtered.length === 0 && (
              <p className="px-3 py-6 text-center text-sm text-gray-400 dark:text-stone-500">
                {locale === 'zh' ? '无匹配结果' : 'No results found'}
              </p>
            )}

            {grouped.map((group) => (
              <div key={group.category}>
                <p className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest text-gray-400 dark:text-stone-500">
                  {categoryLabels[group.category]}
                </p>
                {group.items.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => item.action()}
                    onMouseEnter={() => setSelectedIdx(item.globalIdx)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors',
                      item.globalIdx === selectedIdx
                        ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-400'
                        : 'text-gray-700 hover:bg-gray-50 dark:text-stone-300 dark:hover:bg-stone-800',
                      item.available === false && 'opacity-40',
                    )}
                  >
                    <span className={cn(
                      'shrink-0',
                      item.globalIdx === selectedIdx ? 'text-indigo-500 dark:text-indigo-400' : 'text-gray-400 dark:text-stone-500',
                    )}>
                      {item.icon}
                    </span>
                    <span className="flex-1 truncate">
                      {locale === 'zh' ? (item.labelZh || item.label) : item.label}
                    </span>
                    {item.available === false && (
                      <span className="text-[10px] text-gray-400 dark:text-stone-600">
                        {locale === 'zh' ? '不可用' : 'N/A'}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            ))}
          </div>

          {/* Footer hint */}
          <div className="flex items-center gap-4 border-t border-gray-100 px-4 py-2 text-[10px] text-gray-400 dark:border-stone-800 dark:text-stone-500">
            <span><kbd className="font-medium">↑↓</kbd> {locale === 'zh' ? '导航' : 'navigate'}</span>
            <span><kbd className="font-medium">↵</kbd> {locale === 'zh' ? '选择' : 'select'}</span>
            <span><kbd className="font-medium">esc</kbd> {locale === 'zh' ? '关闭' : 'close'}</span>
          </div>
        </div>
      </div>
    </>
  );
}
