// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * PageHeader — enhanced page header with consistent pro styling.
 *
 * Every page gets the same visual treatment:
 * - Large bold title
 * - Subtle description
 * - Optional action buttons (right-aligned)
 * - Bottom border for separation
 */

interface PageHeaderProps {
  title: React.ReactNode;
  description?: string;
  actions?: React.ReactNode;
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="mb-8 border-b border-gray-100 pb-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight text-gray-900">{title}</h2>
          {description && (
            <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-gray-500">{description}</p>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </div>
    </div>
  );
}
