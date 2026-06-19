// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState } from 'react';
import { HelpCircle, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface InfoTipProps {
  /** Short title for the concept */
  title: string;
  /** Detailed explanation text */
  content: string;
  /** Optional className */
  className?: string;
}

/**
 * Educational info tooltip — click to reveal a concept explanation panel.
 */
export function InfoTip({ title, content, className }: InfoTipProps) {
  const [open, setOpen] = useState(false);

  return (
    <span className={cn('relative inline-flex', className)}>
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center justify-center rounded-full text-gray-400 hover:text-indigo-500 transition-colors"
        title={title}
      >
        <HelpCircle size={14} />
      </button>

      {open && (
        <>
          {/* Backdrop */}
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          {/* Popup */}
          <div className="absolute left-6 top-0 z-50 w-72 rounded-lg border border-gray-200 bg-white p-3 shadow-lg">
            <div className="mb-1 flex items-center justify-between">
              <span className="text-xs font-semibold text-indigo-600">{title}</span>
              <button
                onClick={() => setOpen(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X size={12} />
              </button>
            </div>
            <p className="text-xs leading-relaxed text-gray-600">{content}</p>
          </div>
        </>
      )}
    </span>
  );
}
