// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import type { ModelInfo } from '@/api/types';
import { formatParamCount, formatSize } from '@/lib/utils';
import { Badge } from '@/components/common/Badge';
import { X } from 'lucide-react';

interface ModelInfoCardProps {
  model: ModelInfo;
  onUnload?: () => void;
}

export function ModelInfoCard({ model, onUnload }: ModelInfoCardProps) {
  const bitsPerParam = model.quantization?.bits
    ? `${model.quantization.bits}-bit`
    : (model.total_params !== model.total_stored_params ? 'quantized' : 'full precision');

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-stone-700 dark:bg-stone-900">
      <div className="flex items-start justify-between">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold text-gray-900 dark:text-stone-100">{model.model_name}</h3>
          <p className="mt-0.5 text-xs text-gray-500 dark:text-stone-400">{model.model_type}</p>
        </div>
        {onUnload && (
          <button
            onClick={onUnload}
            className="ml-2 rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 dark:text-stone-500 dark:hover:bg-stone-800 dark:hover:text-stone-300"
            title="Unload model"
          >
            <X size={14} />
          </button>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        <Badge variant={model.model_category === 'vlm' ? 'warning' : model.model_category === 'tts' ? 'success' : model.model_category === 'stt' ? 'warning' : 'info'}>
          {model.model_category?.toUpperCase() || 'LLM'}
        </Badge>
        <Badge variant="info">{formatSize(model.total_size_bytes)}</Badge>
        <Badge variant="default">{formatParamCount(model.total_params)}</Badge>
        <Badge variant={model.quantization ? 'warning' : 'default'}>{bitsPerParam}</Badge>
        <Badge variant="default">{model.num_layers}L</Badge>
      </div>

      <p className="mt-2 truncate text-xs text-gray-400 dark:text-stone-500" title={model.model_dir}>
        {model.model_dir}
      </p>
    </div>
  );
}
