// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useModelStore } from '@/stores/modelStore';
import { useUIStore } from '@/stores/uiStore';
import { EmptyState } from '@/components/common/EmptyState';
import { FolderOpen } from 'lucide-react';
import { useT } from '@/i18n';

export default function Welcome() {
  const model = useModelStore((s) => s.currentModel);
  const setFileBrowserOpen = useUIStore((s) => s.setFileBrowserOpen);
  const setHfPickerOpen = useUIStore((s) => s.setHfPickerOpen);
  const t = useT();

  if (model) {
    return null; // Will redirect in router
  }

  return (
    <div className="mx-auto max-w-5xl">
      <EmptyState
        icon={<FolderOpen size={48} />}
        title={t('welcome.title')}
        description={t('welcome.desc')}
        action={{
          label: t('welcome.openModel'),
          onClick: () => setFileBrowserOpen(true),
        }}
        secondaryAction={{
          label: t('welcome.modelHub'),
          onClick: () => setHfPickerOpen(true),
        }}
        className="py-12"
      />
    </div>
  );
}
