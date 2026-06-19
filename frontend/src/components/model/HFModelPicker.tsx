// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect } from 'react';
import { Search, Download, Loader2, FolderOpen, X, Terminal as TerminalIcon, Trash2, HardDrive } from 'lucide-react';
import { searchHFModels, listLocalModels, deleteLocalModel, downloadHFModel, loadModel, probeHFNetwork, runTerminalCommand, closeTerminal } from '@/api/endpoints';
import type { HFModel, LocalModel } from '@/api/endpoints';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { TerminalOverlay } from '@/components/common/Terminal';
import { useModelStore } from '@/stores/modelStore';
import { useToastStore } from '@/stores/toastStore';
import { useT } from '@/i18n';

interface HFModelPickerProps {
  onClose: () => void;
}

type Tab = 'local' | 'search';

type MirrorSource = 'official' | 'hf-mirror' | 'modelscope';

export function HFModelPicker({ onClose }: HFModelPickerProps) {
  const setCurrentModel = useModelStore((s) => s.setCurrentModel);
  const addToast = useToastStore((s) => s.addToast);
  const t = useT();

  const [tab, setTab] = useState<Tab>('local');
  const [query, setQuery] = useState('mlx-community');
  const [searchResults, setSearchResults] = useState<HFModel[]>([]);
  const [localModels, setLocalModels] = useState<LocalModel[]>([]);
  const [searching, setSearching] = useState(false);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [downloadTaskId, setDownloadTaskId] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [mirror, setMirror] = useState<MirrorSource>('official');
  const [useTerminal, setUseTerminal] = useState(true);
  const [terminalSessionId, setTerminalSessionId] = useState<string | null>(null);
  const [terminalRepoId, setTerminalRepoId] = useState<string | null>(null);

  // Load local models on mount + probe HF network
  useEffect(() => {
    setLoadingLocal(true);
    listLocalModels()
      .then((data) => setLocalModels(data.models))
      .catch(() => console.warn('[HFModelPicker] Failed to load local models'))
      .finally(() => setLoadingLocal(false));

    // Probe HF reachability to auto-select mirror
    probeHFNetwork()
      .then((data) => {
        if (!data.reachable && data.suggestion === 'hf-mirror') {
          setMirror('hf-mirror');
          addToast(t('hub.mirrorAutoSwitched'), 'info');
        }
      })
      .catch(() => {
        // Probe failed, keep default
      });
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setSearching(true);
    try {
      const data = await searchHFModels(query.trim());
      setSearchResults(data.models);
    } catch {
      addToast('Search failed. Check your internet connection.', 'error');
    } finally {
      setSearching(false);
    }
  };

  const handleDownload = async (repoId: string) => {
    if (useTerminal) {
      // Terminal mode: run download script in PTY
      // Pass cmd as array (not bash -c string) to prevent shell injection
      try {
        const localName = repoId.replace('/', '_');
        const localDir = `~/mlx-community/${localName}`;

        const useChinaSource = mirror === 'hf-mirror' || mirror === 'modelscope';
        let cmd: string[];
        if (useChinaSource) {
          cmd = ['bash', 'scripts/msd.sh', repoId, '--local-dir', localDir];
        } else {
          cmd = ['bash', 'scripts/hfd.sh', repoId, '--local-dir', localDir];
        }

        const { session_id } = await runTerminalCommand(cmd);
        setTerminalSessionId(session_id);
        setTerminalRepoId(repoId);
      } catch {
        addToast('Failed to start terminal download.', 'error');
      }
    } else {
      // Background mode: use task system
      try {
        const { task_id } = await downloadHFModel(repoId, undefined, mirror);
        setDownloadTaskId(task_id);
      } catch {
        addToast('Failed to start download.', 'error');
      }
    }
  };

  const handleDownloadComplete = async (result: unknown) => {
    setDownloadTaskId(null);
    const r = result as { path: string; repo_id: string } | null;
    if (!r?.path) {
      addToast('Download completed but path not returned.', 'warning');
      return;
    }
    addToast(`Downloaded to ${r.path}`, 'success');
    // Auto-load the downloaded model
    try {
      setLoadingModel(r.repo_id);
      const info = await loadModel(r.path);
      setCurrentModel(info);
      addToast(`Model "${info.model_name}" loaded successfully!`, 'success');
      onClose();
    } catch {
      addToast('Model downloaded but failed to load. You can load it manually via the file browser.', 'warning');
    } finally {
      setLoadingModel(null);
    }
  };

  const handleLoadLocal = async (path: string) => {
    try {
      setLoadingModel(path);
      const info = await loadModel(path);
      setCurrentModel(info);
      addToast(`Model "${info.model_name}" loaded!`, 'success');
      onClose();
    } catch {
      addToast('Failed to load model.', 'error');
    } finally {
      setLoadingModel(null);
    }
  };

  const [deletingPath, setDeletingPath] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<LocalModel | null>(null);

  const formatSize = (bytes?: number) => {
    if (!bytes) return '';
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  };

  const handleDelete = async (model: LocalModel) => {
    setDeletingPath(model.path);
    try {
      const result = await deleteLocalModel(model.path);
      addToast(`Deleted ${model.name} (freed ${formatSize(result.freed_bytes)})`, 'success');
      setLocalModels(prev => prev.filter(m => m.path !== model.path));
    } catch {
      addToast('Failed to delete model.', 'error');
    } finally {
      setDeletingPath(null);
      setConfirmDelete(null);
    }
  };

  const handleTerminalExit = async (code: number) => {
    // Close terminal first
    if (terminalSessionId) {
      closeTerminal(terminalSessionId).catch(() => {});
      setTerminalSessionId(null);
      setTerminalRepoId(null);
    }

    if (code === 0 && terminalRepoId) {
      // Download succeeded, try to load the model
      const localName = terminalRepoId.replace('/', '_');
      const localDir = `~/mlx-community/${localName}`;
      addToast(`Download complete: ${localDir}`, 'success');
      // Refresh local models list
      try {
        const data = await listLocalModels();
        setLocalModels(data.models);
      } catch {
        // Ignore
      }
    } else if (code !== 0) {
      addToast(`Download exited with code ${code}`, 'warning');
    }
  };

  const handleTerminalClose = () => {
    if (terminalSessionId) {
      closeTerminal(terminalSessionId).catch(() => {});
    }
    setTerminalSessionId(null);
    setTerminalRepoId(null);
  };

  return (
    <>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
        <div className="flex h-[80vh] w-full max-w-2xl flex-col rounded-xl bg-white shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b px-6 py-4">
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Model Hub</h2>
              <p className="text-sm text-gray-500">Load local models or download from HuggingFace</p>
            </div>
            <button onClick={onClose} className="rounded-lg p-1 hover:bg-gray-100">
              <X size={20} className="text-gray-400" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b px-6">
            <button
              onClick={() => setTab('local')}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                tab === 'local'
                  ? 'border-indigo-500 text-indigo-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <FolderOpen size={14} className="inline mr-1.5 -mt-0.5" />
              Local Models
            </button>
            <button
              onClick={() => setTab('search')}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
                tab === 'search'
                  ? 'border-indigo-500 text-indigo-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <Download size={14} className="inline mr-1.5 -mt-0.5" />
              HuggingFace Hub
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto px-6 py-4">
            {tab === 'local' && (
              <>
                {loadingLocal ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="animate-spin text-gray-400" size={24} />
                  </div>
                ) : localModels.length === 0 ? (
                  <p className="py-8 text-center text-sm text-gray-400">
                    No local models found. Download one from HuggingFace Hub or use the file browser.
                  </p>
                ) : (
                  <div className="space-y-1">
                    {/* Total disk usage */}
                    <div className="mb-3 flex items-center gap-2 text-xs text-gray-500">
                      <HardDrive size={12} />
                      <span>
                        {localModels.length} model{localModels.length !== 1 ? 's' : ''} · {formatSize(localModels.reduce((sum, m) => sum + (m.size_bytes || 0), 0))}
                      </span>
                    </div>
                    {localModels.map((m) => (
                      <div
                        key={m.path}
                        className="flex items-center justify-between rounded-lg px-4 py-3 hover:bg-gray-50"
                      >
                        <button
                          onClick={() => handleLoadLocal(m.path)}
                          disabled={loadingModel === m.path || deletingPath === m.path}
                          className="min-w-0 flex-1 text-left disabled:opacity-50"
                        >
                          <p className="text-sm font-medium text-gray-900 truncate">{m.name}</p>
                          <div className="flex gap-2 text-xs text-gray-400">
                            <span className="truncate">{m.path}</span>
                            {m.size_bytes != null && m.size_bytes > 0 && (
                              <span className="shrink-0">{formatSize(m.size_bytes)}</span>
                            )}
                          </div>
                        </button>
                        <div className="ml-3 flex items-center gap-1.5 shrink-0">
                          {loadingModel === m.path ? (
                            <Loader2 size={16} className="animate-spin text-gray-400" />
                          ) : deletingPath === m.path ? (
                            <Loader2 size={16} className="animate-spin text-red-400" />
                          ) : (
                            <>
                              <span className="text-xs text-indigo-500 font-medium cursor-pointer">Load</span>
                              <button
                                onClick={(e) => { e.stopPropagation(); setConfirmDelete(m); }}
                                className="rounded p-1 text-gray-300 hover:text-red-500 hover:bg-red-50 transition-colors"
                                title="Delete model"
                              >
                                <Trash2 size={14} />
                              </button>
                            </>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}

            {tab === 'search' && (
              <>
                <form
                  onSubmit={(e) => { e.preventDefault(); handleSearch(); }}
                  className="mb-3 flex gap-2"
                >
                  <div className="relative flex-1">
                    <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder={t('hub.searchPlaceholder')}
                      className="w-full rounded-lg border border-gray-300 py-2 pl-8 pr-3 text-sm focus:border-indigo-500 focus:outline-none"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={searching}
                    className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
                  >
                    {searching ? <Loader2 size={14} className="animate-spin" /> : 'Search'}
                  </button>
                </form>

                {/* Mirror source selector + Terminal mode toggle */}
                <div className="mb-4 flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-gray-500">{t('hub.mirrorSource')}:</label>
                    <select
                      value={mirror}
                      onChange={(e) => setMirror(e.target.value as MirrorSource)}
                      className="rounded-md border border-gray-300 px-2 py-1 text-xs text-gray-700 focus:border-indigo-500 focus:outline-none"
                      title={t('hub.mirrorTooltip')}
                    >
                      <option value="official">{t('hub.mirrorOfficial')}</option>
                      <option value="hf-mirror">{t('hub.mirrorHFMirror')}</option>
                      <option value="modelscope">{t('hub.mirrorModelScope')}</option>
                    </select>
                  </div>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useTerminal}
                      onChange={(e) => setUseTerminal(e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-gray-300 text-indigo-500 focus:ring-indigo-500"
                    />
                    <TerminalIcon size={12} className="text-gray-500" />
                    <span className="text-xs text-gray-500">Terminal</span>
                  </label>
                </div>

                {searchResults.length === 0 && !searching && (
                  <p className="py-8 text-center text-sm text-gray-400">
                    Search HuggingFace Hub for MLX-compatible models
                  </p>
                )}

                <div className="space-y-1">
                  {searchResults.map((m) => (
                    <div
                      key={m.id}
                      className="flex items-center justify-between rounded-lg px-4 py-3 hover:bg-gray-50"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-gray-900 truncate">{m.id}</p>
                        <div className="flex gap-3 text-xs text-gray-400">
                          {m.downloads != null && <span>Downloads: {m.downloads.toLocaleString()}</span>}
                          {m.likes != null && <span>Likes: {m.likes}</span>}
                          {m.pipeline_tag && <span>{m.pipeline_tag}</span>}
                        </div>
                      </div>
                      <button
                        onClick={() => handleDownload(m.id)}
                        disabled={!!downloadTaskId || loadingModel === m.id}
                        className="ml-3 rounded-lg border border-indigo-200 px-3 py-1.5 text-xs font-medium text-indigo-600 hover:bg-indigo-50 disabled:opacity-50"
                      >
                        <Download size={12} className="inline mr-1 -mt-0.5" />
                        Download
                      </button>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {downloadTaskId && (
        <ProgressOverlay
          taskId={downloadTaskId}
          title="Downloading Model"
          onComplete={handleDownloadComplete}
          onError={() => setDownloadTaskId(null)}
          onClose={() => setDownloadTaskId(null)}
        />
      )}

      {terminalSessionId && (
        <TerminalOverlay
          sessionId={terminalSessionId}
          title={`Downloading ${terminalRepoId || 'model'}`}
          onClose={handleTerminalClose}
          onExit={handleTerminalExit}
        />
      )}

      {/* Delete confirmation dialog */}
      {confirmDelete && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50">
          <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow-2xl">
            <h3 className="text-base font-semibold text-gray-900 mb-2">Delete Model?</h3>
            <p className="text-sm text-gray-600 mb-1">
              {confirmDelete.name}
            </p>
            <p className="text-xs text-gray-400 mb-1 truncate">{confirmDelete.path}</p>
            {confirmDelete.size_bytes != null && confirmDelete.size_bytes > 0 && (
              <p className="text-xs text-gray-500 mb-4">
                This will free {formatSize(confirmDelete.size_bytes)} of disk space.
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setConfirmDelete(null)}
                className="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(confirmDelete)}
                disabled={deletingPath === confirmDelete.path}
                className="rounded-lg bg-red-500 px-4 py-2 text-sm font-medium text-white hover:bg-red-600 disabled:opacity-50"
              >
                {deletingPath === confirmDelete.path ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
