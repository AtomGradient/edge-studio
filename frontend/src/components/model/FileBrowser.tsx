// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect } from 'react';
import { Folder, File, ArrowUp, Check, Loader2, Box } from 'lucide-react';
import { browsePath, getHome } from '@/api/endpoints';
import type { BrowseResponse, FileEntry } from '@/api/types';
import { formatSize, cn } from '@/lib/utils';

interface FileBrowserProps {
  onSelect: (path: string) => void;
  onCancel: () => void;
  /** 'model' = model dirs + .gguf (default), 'file' = select files matching allowedExtensions */
  mode?: 'model' | 'file';
  /** File extensions to allow in 'file' mode, e.g. ['.json', '.csv'] */
  allowedExtensions?: string[];
  /** Custom title */
  title?: string;
  /** Custom description */
  description?: string;
}

function isGgufFile(name: string) {
  return name.toLowerCase().endsWith('.gguf');
}

function matchesExtensions(name: string, exts: string[]) {
  const lower = name.toLowerCase();
  return exts.some(ext => lower.endsWith(ext));
}

export function FileBrowser({ onSelect, onCancel, mode = 'model', allowedExtensions, title, description }: FileBrowserProps) {
  const [data, setData] = useState<BrowseResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pathInput, setPathInput] = useState('');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const isFileMode = mode === 'file';
  const exts = allowedExtensions ?? [];

  const loadDir = async (path?: string) => {
    setLoading(true);
    setError(null);
    setSelectedFile(null);
    try {
      const result = await browsePath(path);
      setData(result);
      setPathInput(result.current_path);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to browse';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    getHome().then(({ path }) => loadDir(path));
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onCancel]);

  const handleEntry = (entry: FileEntry) => {
    if (entry.is_dir) {
      loadDir(entry.path);
    } else if (isFileMode && exts.length > 0 && matchesExtensions(entry.name, exts)) {
      setSelectedFile(entry.path);
    } else if (!isFileMode && isGgufFile(entry.name)) {
      setSelectedFile(entry.path);
    }
  };

  const handlePathSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    loadDir(pathInput);
  };

  const isValidSafetensors = data?.has_config_json && data?.has_safetensors;
  const canLoad = isFileMode
    ? !!selectedFile
    : isValidSafetensors || !!selectedFile || (data?.has_gguf && !selectedFile);

  // Count gguf files in current directory
  const ggufFiles = data?.entries.filter(e => !e.is_dir && isGgufFile(e.name)) ?? [];
  const multipleGguf = ggufFiles.length > 1;

  const handleLoad = () => {
    if (selectedFile) {
      onSelect(selectedFile);
    } else if (data) {
      onSelect(data.current_path);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="flex h-[80vh] w-full max-w-2xl flex-col rounded-xl bg-white shadow-2xl">
        {/* Header */}
        <div className="border-b px-6 py-4">
          <h2 className="text-lg font-semibold text-gray-900">
            {title ?? (isFileMode ? 'Select File' : 'Open Model')}
          </h2>
          <p className="text-sm text-gray-500">
            {description ?? (isFileMode && exts.length > 0
              ? `Select a ${exts.join(' / ')} file`
              : 'Select a model directory or a .gguf file')}
          </p>
        </div>

        {/* Path input */}
        <form onSubmit={handlePathSubmit} className="border-b px-6 py-3">
          <div className="flex gap-2">
            <input
              value={pathInput}
              onChange={(e) => setPathInput(e.target.value)}
              className="flex-1 rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
              placeholder="/path/to/model"
            />
            <button
              type="submit"
              className="rounded-lg bg-gray-100 px-3 py-1.5 text-sm font-medium hover:bg-gray-200"
            >
              Go
            </button>
          </div>
        </form>

        {/* Selection indicator */}
        {!isFileMode && isValidSafetensors && !selectedFile && (
          <div className="mx-6 mt-3 flex items-center gap-2 rounded-lg bg-green-50 px-3 py-2 text-sm text-green-700">
            <Check size={16} />
            <span>Valid model directory (config.json + .safetensors)</span>
          </div>
        )}
        {selectedFile && (
          <div className="mx-6 mt-3 flex items-center gap-2 rounded-lg bg-indigo-50 px-3 py-2 text-sm text-indigo-700">
            {isFileMode ? <File size={16} /> : <Box size={16} />}
            <span className="flex-1 truncate">
              {isFileMode ? 'Selected: ' : 'GGUF selected: '}{selectedFile.split('/').pop()}
            </span>
            <button
              onClick={() => setSelectedFile(null)}
              className="text-xs text-indigo-400 hover:text-indigo-600"
            >
              Clear
            </button>
          </div>
        )}
        {!isFileMode && !selectedFile && !isValidSafetensors && multipleGguf && (
          <div className="mx-6 mt-3 flex items-center gap-2 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
            <Box size={16} />
            <span>Multiple .gguf files found — click one to select it</span>
          </div>
        )}

        {/* File list */}
        <div className="flex-1 overflow-y-auto px-6 py-3">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="animate-spin text-gray-400" size={24} />
            </div>
          )}

          {error && (
            <p className="py-4 text-center text-sm text-red-500">{error}</p>
          )}

          {!loading && !error && data && (
            <div className="space-y-0.5">
              {/* Parent dir */}
              {data.parent_path && (
                <button
                  onClick={() => loadDir(data.parent_path!)}
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm hover:bg-gray-50"
                >
                  <ArrowUp size={16} className="text-gray-400" />
                  <span className="text-gray-500">..</span>
                </button>
              )}

              {data.entries.map((entry) => {
                const isGguf = !entry.is_dir && isGgufFile(entry.name);
                const isAllowedFile = !entry.is_dir && isFileMode && exts.length > 0 && matchesExtensions(entry.name, exts);
                const isClickable = entry.is_dir || (isFileMode ? isAllowedFile : isGguf);
                const isSelected = !entry.is_dir && selectedFile === entry.path;

                return (
                  <button
                    key={entry.path}
                    onClick={() => handleEntry(entry)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm',
                      isSelected
                        ? 'bg-indigo-50 ring-1 ring-indigo-300'
                        : 'hover:bg-gray-50',
                      !isClickable && 'cursor-default opacity-50',
                    )}
                    disabled={!isClickable}
                  >
                    {entry.is_dir ? (
                      <Folder size={16} className="text-indigo-400" />
                    ) : isAllowedFile || isGguf ? (
                      <File size={16} className={isSelected ? 'text-indigo-600' : isAllowedFile ? 'text-emerald-500' : 'text-orange-400'} />
                    ) : (
                      <File size={16} className="text-gray-400" />
                    )}
                    <span className={cn('flex-1 truncate', isSelected && 'font-medium text-indigo-700')}>
                      {entry.name}
                    </span>
                    {entry.size != null && !entry.is_dir && (
                      <span className="text-xs text-gray-400">{formatSize(entry.size)}</span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 border-t px-6 py-4">
          <button
            onClick={onCancel}
            className="rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100"
          >
            Cancel
          </button>
          <button
            onClick={handleLoad}
            disabled={!canLoad}
            className={cn(
              'rounded-lg px-4 py-2 text-sm font-medium text-white',
              canLoad
                ? 'bg-indigo-500 hover:bg-indigo-600'
                : 'cursor-not-allowed bg-gray-300',
            )}
          >
            {isFileMode ? 'Select' : selectedFile ? 'Load GGUF' : 'Load Model'}
          </button>
        </div>
      </div>
    </div>
  );
}
