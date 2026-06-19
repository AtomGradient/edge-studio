// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useMemo, useState } from 'react';
import type { ALibrarySelectionCandidate, ModelInfo, ScaffoldZipExportResult } from '@/api/types';
import { exportScaffoldZip, downloadScaffoldZip, selectALibrary } from '@/api/endpoints';
import { ProgressOverlay } from '@/components/common/ProgressOverlay';
import { MetricCards } from '@/components/data/MetricCards';
import { formatSize } from '@/lib/utils';
import { useT } from '@/i18n';
import { WifiOff, Shield, Zap, Battery } from 'lucide-react';

const DEFAULT_SCAFFOLD_DIRECTION_SET_ID = 'finance_consumer';

export function ScaffoldAppExportMode({ model }: { model: ModelInfo }) {
  const t = useT();
  const [appName, setAppName] = useState('MyApp');
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
  const modelTier = '';
  const enableDSR = true;
  const dsrBudget = '';
  const [taskId, setTaskId] = useState<string | null>(null);
  const [result, setResult] = useState<ScaffoldZipExportResult | null>(null);
  const [aLibraryCandidates, setALibraryCandidates] = useState<ALibrarySelectionCandidate[]>([]);
  const [selectedDirectionSetId, setSelectedDirectionSetId] = useState('directions_a');
  const [aLibraryError, setALibraryError] = useState('');

  useEffect(() => {
    let cancelled = false;
    /* eslint-disable react-hooks/set-state-in-effect -- clear stale A-library selection while loading a new model */
    setALibraryError('');
    setALibraryCandidates([]);
    /* eslint-enable react-hooks/set-state-in-effect */
    selectALibrary({ model_id: model.model_id })
      .then((selection) => {
        if (cancelled) return;
        const compatibleSorted = selection.candidates
          .filter((candidate) => candidate.match_reasons?.includes('matched') && candidate.direction_set_id)
          .sort((a, b) => {
            const aID = a.direction_set_id ?? '';
            const bID = b.direction_set_id ?? '';
            if (aID === bID) return 0;
            if (aID === DEFAULT_SCAFFOLD_DIRECTION_SET_ID) return -1;
            if (bID === DEFAULT_SCAFFOLD_DIRECTION_SET_ID) return 1;
            if (aID === 'directions_a') return 1;
            if (bID === 'directions_a') return -1;
            return aID.localeCompare(bID);
          });
        const compatible = Array.from(
          compatibleSorted.reduce((items, candidate) => {
            const directionSetId = candidate.direction_set_id;
            if (directionSetId && !items.has(directionSetId)) {
              items.set(directionSetId, candidate);
            }
            return items;
          }, new Map<string, ALibrarySelectionCandidate>()).values()
        );
        setALibraryCandidates(compatible);
        const defaultCandidate = compatible.find((candidate) => candidate.direction_set_id === DEFAULT_SCAFFOLD_DIRECTION_SET_ID)
          ?? compatible.find((candidate) => candidate.direction_set_id === 'directions_a')
          ?? compatible[0]
          ?? null;
        setSelectedDirectionSetId(defaultCandidate?.direction_set_id ?? 'directions_a');
        if (compatible.length === 0) {
          setALibraryError((selection.reasons ?? []).join(', ') || 'No compatible A-library found for this model.');
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setALibraryError(error instanceof Error ? error.message : 'Failed to load A-library candidates.');
        }
      });
    return () => { cancelled = true; };
  }, [model.model_id]);

  const selectedALibrary = useMemo(
    () => aLibraryCandidates.find((candidate) => candidate.direction_set_id === selectedDirectionSetId) ?? null,
    [aLibraryCandidates, selectedDirectionSetId]
  );

  const handleExport = async () => {
    try {
      const budget = dsrBudget ? parseInt(dsrBudget, 10) : null;
      const { task_id } = await exportScaffoldZip(
        model.model_id, appName, systemPrompt, modelTier, enableDSR, budget, selectedDirectionSetId
      );
      setTaskId(task_id);
    } catch { /* error handled by ProgressOverlay */ }
  };

  return (
    <>
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-4 space-y-4">
        <p className="text-sm text-gray-600">
          Export a complete iOS App scaffold as a ZIP. Contains the App project +
          <strong> EdgeKit</strong> SDK. Unzip, open in Xcode, build & run.
        </p>
        <p className="text-xs text-amber-600">
          Note: Model files are NOT included in the ZIP (too large). They are referenced
          in the Xcode project with ODR resource tags — ensure the model directory exists on this machine.
        </p>

        {/* EdgeRuntime advantages */}
        <div className="rounded-lg border border-amber-200 bg-amber-50/50 p-4 dark:border-amber-900/50 dark:bg-amber-900/10">
          <h4 className="mb-2 text-xs font-semibold text-amber-800 dark:text-amber-300">
            {t('simple.export.poweredBy')}
          </h4>
          <div className="grid grid-cols-2 gap-2.5">
            {([
              { icon: WifiOff, title: t('simple.export.rt.offline'), desc: t('simple.export.rt.offlineDesc') },
              { icon: Shield, title: t('simple.export.rt.privacy'), desc: t('simple.export.rt.privacyDesc') },
              { icon: Zap, title: t('simple.export.rt.speed'), desc: t('simple.export.rt.speedDesc') },
              { icon: Battery, title: t('simple.export.rt.cost'), desc: t('simple.export.rt.costDesc') },
            ] as const).map(({ icon: Icon, title, desc }) => (
              <div key={title} className="flex items-start gap-2">
                <Icon size={14} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <p className="text-xs font-medium text-gray-800 dark:text-stone-200">{title}</p>
                  <p className="text-[11px] text-gray-500 dark:text-stone-400">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 max-w-xl">
          <div>
            <label className="mb-1 block text-xs text-gray-500">App Name</label>
            <input
              type="text"
              value={appName}
              onChange={(e) => setAppName(e.target.value)}
              placeholder="MyApp"
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-gray-500">System Prompt</label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={3}
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-gray-500">RPP A-library</label>
            <select
              value={selectedDirectionSetId}
              onChange={(e) => setSelectedDirectionSetId(e.target.value)}
              disabled={aLibraryCandidates.length === 0}
              className="w-full rounded-lg border border-gray-300 px-3 py-1.5 text-sm"
            >
              {aLibraryCandidates.map((candidate) => (
                <option key={candidate.library_id ?? candidate.direction_set_id} value={candidate.direction_set_id ?? ''}>
                  {candidate.direction_set_id} · L{candidate.target_layer} · {candidate.n_directions ?? '?'} directions
                </option>
              ))}
            </select>
            {selectedALibrary && (
              <p className="mt-1 text-[11px] text-gray-500">
                {selectedALibrary.library_id}
              </p>
            )}
            {aLibraryError && (
              <p className="mt-1 text-xs text-red-500">
                {aLibraryError}. Generate a matching A-library in Training → A-library first.
              </p>
            )}
          </div>
        </div>

        <button
          onClick={handleExport}
          disabled={!!taskId || !appName.trim() || aLibraryCandidates.length === 0}
          className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600 disabled:opacity-50"
        >
          Export Scaffold App ZIP
        </button>
      </div>

      {taskId && (
        <ProgressOverlay
          taskId={taskId}
          title="Generating Scaffold App ZIP"
          onComplete={(r) => {
            setResult(r as ScaffoldZipExportResult);
            setTaskId(null);
          }}
          onError={() => setTaskId(null)}
          onClose={() => setTaskId(null)}
        />
      )}

      {result && (
        <div className="mt-6 rounded-xl border border-gray-200 bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-gray-700">Scaffold App Export Result</h3>
          {result.success ? (
            <div className="space-y-3">
              <MetricCards
                metrics={[
                  { label: 'App Name', value: result.app_name },
                  { label: 'Model', value: result.model_name },
                  { label: 'Tier', value: result.model_tier },
                  { label: 'A-library', value: result.direction_set_id || selectedDirectionSetId },
                  { label: 'ZIP Size', value: formatSize(result.zip_size_bytes) },
                ]}
              />
              <div className="rounded-lg bg-gray-50 p-3 text-xs text-gray-600 space-y-1">
                <p>Model path: <span className="font-mono">{result.model_dir}</span></p>
                <p>ODR tag: <span className="font-mono">model-{result.model_tier}</span></p>
                <p className="text-amber-600">Model is referenced in .xcodeproj via ODR resource tag — ensure the path above exists on this machine.</p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => downloadScaffoldZip(result.zip_path)}
                  className="rounded-lg bg-indigo-500 px-6 py-2 text-sm font-medium text-white hover:bg-indigo-600"
                >
                  Download ZIP
                </button>
              </div>
              <div className="rounded-lg bg-gray-900 px-4 py-3 text-xs text-gray-300 space-y-1">
                <p className="text-gray-400">After download:</p>
                <p><span className="text-green-400">1.</span> Unzip the archive</p>
                <p><span className="text-green-400">2.</span> Open <span className="font-mono text-green-400">{result.app_name}/{result.app_name}.xcodeproj</span> in Xcode</p>
                <p><span className="text-green-400">3.</span> Select your Development Team (Signing & Capabilities)</p>
                <p><span className="text-green-400">4.</span> Build & Run on a physical iOS device</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-red-500">{result.error || 'Export failed'}</p>
          )}
        </div>
      )}
    </>
  );
}
