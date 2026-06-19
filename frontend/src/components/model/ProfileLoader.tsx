// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useState, useEffect } from 'react';
import { FileText, Loader2, Zap, X } from 'lucide-react';
import { listProfiles, loadProfile, generateProfile } from '@/api/endpoints';
import { useModelStore } from '@/stores/modelStore';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import type { ProfileSummary } from '@/api/types';
import { cn } from '@/lib/utils';

interface ProfileLoaderProps {
  onClose: () => void;
}

export function ProfileLoader({ onClose }: ProfileLoaderProps) {
  const model = useModelStore((s) => s.currentModel);
  const setProfileSummary = useModelStore((s) => s.setProfileSummary);
  const [profiles, setProfiles] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generation
  const [numRuns, setNumRuns] = useState(5);
  const [generatingTaskId, setGeneratingTaskId] = useState<string | null>(null);

  useEffect(() => {
    if (!model) return;
    setLoading(true);
    listProfiles(model.model_id)
      .then((data) => setProfiles(data.profiles))
      .catch(() => setProfiles([]))
      .finally(() => setLoading(false));
  }, [model?.model_id]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  const handleLoadProfile = async (path: string) => {
    if (!model) return;
    setLoadingProfile(true);
    setError(null);
    try {
      const summary = await loadProfile(model.model_id, path);
      setProfileSummary(summary);
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load profile';
      setError(msg);
    } finally {
      setLoadingProfile(false);
    }
  };

  const handleGenerate = async () => {
    if (!model) return;
    setError(null);
    try {
      const { task_id } = await generateProfile(model.model_id, numRuns);
      setGeneratingTaskId(task_id);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to start generation';
      setError(msg);
    }
  };

  const handleGenerateComplete = (result: unknown) => {
    // The result contains profile summary data
    if (result && typeof result === 'object') {
      const r = result as ProfileSummary;
      setProfileSummary(r);
    }
    setGeneratingTaskId(null);
    onClose();
  };

  if (generatingTaskId) {
    return (
      <ProgressOverlay
        taskId={generatingTaskId}
        title="Generating Activation Profile"
        onComplete={handleGenerateComplete}
        onError={(err) => {
          setError(err);
          setGeneratingTaskId(null);
        }}
        onClose={() => setGeneratingTaskId(null)}
      />
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg rounded-xl bg-white shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-6 py-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Load Activation Profile</h2>
            <p className="text-sm text-gray-500">Select an existing profile or generate a new one</p>
          </div>
          <button onClick={onClose} className="rounded-lg p-1 hover:bg-gray-100">
            <X size={18} className="text-gray-400" />
          </button>
        </div>

        {/* Profile list */}
        <div className="px-6 py-4">
          {loading && (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="animate-spin text-gray-400" size={24} />
            </div>
          )}

          {!loading && profiles.length > 0 && (
            <div className="space-y-1">
              <p className="mb-2 text-xs font-medium text-gray-500 uppercase">Auto-detected profiles</p>
              {profiles.map((path) => {
                const fileName = path.split('/').pop() || path;
                return (
                  <button
                    key={path}
                    onClick={() => handleLoadProfile(path)}
                    disabled={loadingProfile}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm hover:bg-gray-50',
                      loadingProfile && 'opacity-50',
                    )}
                  >
                    <FileText size={16} className="text-indigo-400" />
                    <div className="flex-1 min-w-0">
                      <p className="truncate font-medium text-gray-700">{fileName}</p>
                      <p className="truncate text-xs text-gray-400">{path}</p>
                    </div>
                    {loadingProfile && <Loader2 size={14} className="animate-spin text-gray-400" />}
                  </button>
                );
              })}
            </div>
          )}

          {!loading && profiles.length === 0 && (
            <p className="py-4 text-center text-sm text-gray-400">
              No profile files found in model directory
            </p>
          )}

          {error && (
            <p className="mt-2 text-center text-sm text-red-500">{error}</p>
          )}
        </div>

        {/* Generate section */}
        <div className="border-t px-6 py-4">
          <p className="mb-3 text-xs font-medium text-gray-500 uppercase">Generate new profile</p>
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <label className="mb-1 block text-xs text-gray-500">Profiling runs</label>
              <input
                type="number"
                min={1}
                max={50}
                value={numRuns}
                onChange={(e) => setNumRuns(Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
              />
            </div>
            <button
              onClick={handleGenerate}
              className="flex items-center gap-2 rounded-lg bg-amber-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-amber-600"
            >
              <Zap size={14} />
              Generate
            </button>
          </div>
          <p className="mt-2 text-xs text-gray-400">
            Requires running inference ({numRuns} forward passes). May take 2-5 minutes.
          </p>
        </div>

        {/* Footer */}
        <div className="flex justify-end border-t px-6 py-3">
          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-medium text-gray-600 hover:bg-gray-100"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
