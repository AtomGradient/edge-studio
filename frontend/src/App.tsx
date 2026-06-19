// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppShell } from '@/components/layout/AppShell';
import { SimpleShell } from '@/components/layout/SimpleShell';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import { ToastContainer } from '@/components/common/Toast';
import { GuidedTour } from '@/components/common/GuidedTour';
import { useModelStore } from '@/stores/modelStore';
import { useUIStore } from '@/stores/uiStore';
import { lazy, Suspense, useEffect, type ReactNode } from 'react';
import { listLoadedModels, loadModel } from '@/api/endpoints';
import { SkeletonPage } from '@/components/common/Skeleton';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

const Welcome = lazy(() => import('@/pages/Welcome'));
const ArchitectureBrowser = lazy(() => import('@/pages/ArchitectureBrowser'));
const WeightAnalysis = lazy(() => import('@/pages/WeightAnalysis'));
const ActivationHeatmap = lazy(() => import('@/pages/ActivationHeatmap'));
const PruningSimulator = lazy(() => import('@/pages/PruningSimulator'));
const InferenceTracer = lazy(() => import('@/pages/InferenceTracer'));
const Chat = lazy(() => import('@/pages/Chat'));
const DuplexChat = lazy(() => import('@/pages/DuplexChat'));
const AttentionPatterns = lazy(() => import('@/pages/AttentionPatterns'));
const QualityValidator = lazy(() => import('@/pages/QualityValidator'));
const KVCacheAnalysis = lazy(() => import('@/pages/KVCacheAnalysis'));
const NeuralImprintInspector = lazy(() => import('@/pages/NeuralImprintInspector'));
const NeuralImprintChat = lazy(() => import('@/pages/NeuralImprintChat'));
const RPPResultsPanel = lazy(() => import('@/pages/RPPResultsPanel'));
const ALibraryPanel = lazy(() => import('@/pages/ALibraryPanel'));
const OptimizationAdvisor = lazy(() => import('@/pages/OptimizationAdvisor'));
const AutoOptimizer = lazy(() => import('@/pages/AutoOptimizer'));
const MOEAnalyzer = lazy(() => import('@/pages/MOEAnalyzer'));
const ModelComparison = lazy(() => import('@/pages/ModelComparison'));
const OptimizationPipeline = lazy(() => import('@/pages/OptimizationPipeline'));
const Export = lazy(() => import('@/pages/Export'));
const DistillPage = lazy(() => import('@/pages/DistillPage'));
const MergePage = lazy(() => import('@/pages/MergePage'));
const AutoTunePage = lazy(() => import('@/pages/AutoTunePage'));
const DevicesPage = lazy(() => import('@/pages/DevicesPage'));
const JointInferenceHistory = lazy(() => import('@/pages/JointInferenceHistory'));

// Pro mode pages
const ProDashboard = lazy(() => import('@/pages/pro/ProDashboard'));
const MixedPrecisionPanel = lazy(() => import('@/pages/pro/MixedPrecisionPanel'));
const BenchmarkDashboard = lazy(() => import('@/pages/pro/BenchmarkDashboard'));
const BatchOperations = lazy(() => import('@/pages/pro/BatchOperations'));

// Beginner wizard pages (v1 — legacy, kept for backward compat)
const SimpleWelcome = lazy(() => import('@/pages/simple/SimpleWelcome'));
const DeviceAssessment = lazy(() => import('@/pages/simple/DeviceAssessment'));
const ModelPicker = lazy(() => import('@/pages/simple/ModelPicker'));
const OneClickOptimize = lazy(() => import('@/pages/simple/OneClickOptimize'));
const TestChat = lazy(() => import('@/pages/simple/TestChat'));
const SimpleExport = lazy(() => import('@/pages/simple/SimpleExport'));

// Simple mode v2 — "2 clicks + auto download"
const DeviceProfilePage = lazy(() => import('@/pages/simple/v2/DeviceProfilePage'));
const FocusSelectPage = lazy(() => import('@/pages/simple/v2/FocusSelectPage'));
const TierSelectPage = lazy(() => import('@/pages/simple/v2/TierSelectPage'));
const SetupPage = lazy(() => import('@/pages/simple/v2/SetupPage'));
const CompletePage = lazy(() => import('@/pages/simple/v2/CompletePage'));
const ExportDevicePage = lazy(() => import('@/pages/simple/v2/ExportDevicePage'));
const ExportGeneratePage = lazy(() => import('@/pages/simple/v2/ExportGeneratePage'));

function PageLoader() {
  return <SkeletonPage />;
}

function SafePage({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary>
      <Suspense fallback={<PageLoader />}>
        {children}
      </Suspense>
    </ErrorBoundary>
  );
}

function RootRedirect() {
  const model = useModelStore((s) => s.currentModel);
  const userMode = useUIStore((s) => s.userMode);
  if (userMode === 'beginner') return <Navigate to="/simple" replace />;
  if (model) return <Navigate to="/dashboard" replace />;
  return (
    <Suspense fallback={<PageLoader />}>
      <Welcome />
    </Suspense>
  );
}

/**
 * On mount, revalidate persisted models against the backend.
 * If the backend doesn't have them loaded, try to re-load from model_dir.
 * If re-load fails, clear the store entry.
 */
function useRevalidateModels() {
  const currentModel = useModelStore((s) => s.currentModel);
  const comparisonModel = useModelStore((s) => s.comparisonModel);
  const setCurrentModel = useModelStore((s) => s.setCurrentModel);
  const setComparisonModel = useModelStore((s) => s.setComparisonModel);

  useEffect(() => {
    if (!currentModel && !comparisonModel) return;

    let cancelled = false;

    async function revalidate() {
      try {
        const loaded = await listLoadedModels();
        const loadedIds = new Set(loaded.map((m) => m.model_id));

        // Revalidate current model
        if (currentModel && !loadedIds.has(currentModel.model_id)) {
          try {
            const reloaded = await loadModel(currentModel.model_dir);
            if (!cancelled) setCurrentModel(reloaded);
          } catch {
            if (!cancelled) setCurrentModel(null);
          }
        }

        // Revalidate comparison model
        if (comparisonModel && !loadedIds.has(comparisonModel.model_id)) {
          try {
            const reloaded = await loadModel(comparisonModel.model_dir);
            if (!cancelled) setComparisonModel(reloaded);
          } catch {
            if (!cancelled) setComparisonModel(null);
          }
        }
      } catch {
        // Backend unreachable — keep persisted state, will retry on next interaction
      }
    }

    revalidate();
    return () => { cancelled = true; };
    // Only run once on mount — persisted values are stable after hydration
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

export default function App() {
  useRevalidateModels();

  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <BrowserRouter>
          <Routes>
            {/* Simple mode v2 — "2 clicks + auto download" */}
            <Route element={<SimpleShell />}>
              <Route path="/simple" element={<SafePage><DeviceProfilePage /></SafePage>} />
              <Route path="/simple/focus" element={<SafePage><FocusSelectPage /></SafePage>} />
              <Route path="/simple/tier" element={<SafePage><TierSelectPage /></SafePage>} />
              <Route path="/simple/setup" element={<SafePage><SetupPage /></SafePage>} />
              <Route path="/simple/done" element={<SafePage><CompletePage /></SafePage>} />
              <Route path="/simple/export/device" element={<SafePage><ExportDevicePage /></SafePage>} />
              <Route path="/simple/export/generate" element={<SafePage><ExportGeneratePage /></SafePage>} />
            </Route>

            {/* Legacy Simple wizard (v1) — kept for backward compat */}
            <Route element={<SimpleShell />}>
              <Route path="/simple/v1" element={<SafePage><SimpleWelcome /></SafePage>} />
              <Route path="/simple/v1/device" element={<SafePage><DeviceAssessment /></SafePage>} />
              <Route path="/simple/v1/pick-model" element={<SafePage><ModelPicker /></SafePage>} />
              <Route path="/simple/v1/optimize" element={<SafePage><OneClickOptimize /></SafePage>} />
              <Route path="/simple/v1/test" element={<SafePage><TestChat /></SafePage>} />
              <Route path="/simple/v1/export" element={<SafePage><SimpleExport /></SafePage>} />
            </Route>

            {/* Expert modes (simple + advanced) — AppShell with sidebar */}
            <Route element={<AppShell />}>
              <Route path="/" element={<RootRedirect />} />
              <Route path="/dashboard" element={<SafePage><ProDashboard /></SafePage>} />
              <Route path="/architecture" element={<SafePage><ArchitectureBrowser /></SafePage>} />
              <Route path="/weights" element={<SafePage><WeightAnalysis /></SafePage>} />
              <Route path="/activation" element={<SafePage><ActivationHeatmap /></SafePage>} />
              <Route path="/pruning" element={<SafePage><PruningSimulator /></SafePage>} />
              <Route path="/inference" element={<SafePage><InferenceTracer /></SafePage>} />
              <Route path="/chat" element={<SafePage><Chat /></SafePage>} />
              <Route path="/duplex" element={<SafePage><DuplexChat /></SafePage>} />
              <Route path="/attention" element={<SafePage><AttentionPatterns /></SafePage>} />
              <Route path="/quality" element={<SafePage><QualityValidator /></SafePage>} />
              <Route path="/kv-cache" element={<SafePage><KVCacheAnalysis /></SafePage>} />
              <Route path="/neural-imprint" element={<SafePage><NeuralImprintInspector /></SafePage>} />
              <Route path="/neural-imprint-chat" element={<SafePage><NeuralImprintChat /></SafePage>} />
              <Route path="/rpp-results" element={<SafePage><RPPResultsPanel /></SafePage>} />
              <Route path="/a-library" element={<SafePage><ALibraryPanel /></SafePage>} />
              <Route path="/optimization" element={<SafePage><OptimizationAdvisor /></SafePage>} />
              <Route path="/auto-optimizer" element={<SafePage><AutoOptimizer /></SafePage>} />
              <Route path="/pipeline" element={<SafePage><OptimizationPipeline /></SafePage>} />
              <Route path="/moe" element={<SafePage><MOEAnalyzer /></SafePage>} />
              <Route path="/comparison" element={<SafePage><ModelComparison /></SafePage>} />
              <Route path="/export" element={<SafePage><Export /></SafePage>} />
              <Route path="/distill" element={<SafePage><DistillPage /></SafePage>} />
              <Route path="/merge" element={<SafePage><MergePage /></SafePage>} />
              <Route path="/auto-tune" element={<SafePage><AutoTunePage /></SafePage>} />
              <Route path="/mixed-precision" element={<SafePage><MixedPrecisionPanel /></SafePage>} />
              <Route path="/benchmark-dashboard" element={<SafePage><BenchmarkDashboard /></SafePage>} />
              <Route path="/batch" element={<SafePage><BatchOperations /></SafePage>} />
              <Route path="/devices" element={<SafePage><DevicesPage /></SafePage>} />
              <Route path="/joint-inference" element={<SafePage><JointInferenceHistory /></SafePage>} />
              <Route path="/joint-inference/:requestId" element={<SafePage><JointInferenceHistory /></SafePage>} />
            </Route>
          </Routes>
          <GuidedTour />
        </BrowserRouter>
        <ToastContainer />
      </ErrorBoundary>
    </QueryClientProvider>
  );
}
