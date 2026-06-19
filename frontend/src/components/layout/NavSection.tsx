// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { NavLink, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import type { LucideIcon } from 'lucide-react';
import { useT } from '@/i18n';

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  available: boolean;
  hidden?: boolean;
  tooltip?: string;
  /** Feature depends on unreleased component — show badge, disable click */
  comingSoon?: boolean;
}

interface NavSectionProps {
  title: string;
  items: NavItem[];
  collapsed?: boolean;
}

export function NavSection({ title, items, collapsed }: NavSectionProps) {
  const location = useLocation();
  const t = useT();
  const visibleItems = items.filter(i => !i.hidden);
  if (visibleItems.length === 0) return null;

  const hasActiveChild = visibleItems.some(i => location.pathname === i.path);

  return (
    <div className="mb-3">
      {/* Section title — only when expanded and title is non-empty */}
      {!collapsed && title && (
        <div className="mb-1 flex items-center gap-2 px-3">
          {hasActiveChild && (
            <span className="h-1 w-1 rounded-full bg-gray-900 dark:bg-gray-100" />
          )}
          <p className={cn(
            'text-[10px] font-semibold uppercase tracking-widest',
            hasActiveChild
              ? 'text-gray-700 dark:text-gray-300'
              : 'text-gray-400 dark:text-gray-500',
          )}>
            {title}
          </p>
        </div>
      )}

      {/* Collapsed divider */}
      {collapsed && title && (
        <div className="mb-1.5 mx-auto h-px w-6 bg-gray-200 dark:bg-gray-700" />
      )}

      <nav className="space-y-0.5">
        {visibleItems.map((item) => {
          const Icon = item.icon;
          const disabled = !item.available || item.comingSoon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => cn(
                'flex items-center rounded-lg transition-all duration-150',
                collapsed ? 'justify-center px-0 py-2' : 'gap-2.5 px-3 py-1.5 text-sm',
                disabled
                  ? 'cursor-not-allowed text-gray-300 dark:text-gray-600'
                  : isActive
                    ? 'bg-gray-100 font-medium text-gray-900 dark:bg-gray-800 dark:text-gray-100'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800/50 dark:hover:text-gray-200',
              )}
              onClick={(e) => disabled && e.preventDefault()}
              title={collapsed
                ? (item.comingSoon ? `${item.label} — ${t('nav.comingSoon')}` : item.label)
                : (item.comingSoon ? t('nav.comingSoon') : (disabled ? item.tooltip : undefined))
              }
            >
              {collapsed ? (
                <Icon size={18} />
              ) : (
                <>
                  <Icon size={15} className="shrink-0 opacity-60" />
                  <span className="flex-1 truncate">{item.label}</span>
                  {item.comingSoon && (
                    <span className="ml-auto shrink-0 rounded bg-amber-100 px-1.5 py-0.5 text-[9px] font-semibold uppercase leading-none text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
                      {t('nav.soon')}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          );
        })}
      </nav>
    </div>
  );
}
