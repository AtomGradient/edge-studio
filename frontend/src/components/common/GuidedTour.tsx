// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useEffect, useCallback } from 'react';
import { useTourStore } from '@/stores/tourStore';
import { useT } from '@/i18n';
import { X, ChevronLeft, ChevronRight, GraduationCap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface TourStep {
  titleKey: string;
  contentKey: string;
  /** CSS selector to highlight (optional) */
  target?: string;
  /** Route to navigate to for this step */
  route?: string;
}

const TOUR_STEPS: TourStep[] = [
  {
    titleKey: 'tour.step1.title',
    contentKey: 'tour.step1.content',
  },
  {
    titleKey: 'tour.step2.title',
    contentKey: 'tour.step2.content',
    route: '/',
  },
  {
    titleKey: 'tour.step3.title',
    contentKey: 'tour.step3.content',
    route: '/architecture',
  },
  {
    titleKey: 'tour.step4.title',
    contentKey: 'tour.step4.content',
    route: '/weights',
  },
  {
    titleKey: 'tour.step5.title',
    contentKey: 'tour.step5.content',
    route: '/inference',
  },
  {
    titleKey: 'tour.step6.title',
    contentKey: 'tour.step6.content',
    route: '/chat',
  },
  {
    titleKey: 'tour.step7.title',
    contentKey: 'tour.step7.content',
    route: '/optimization',
  },
  {
    titleKey: 'tour.step8.title',
    contentKey: 'tour.step8.content',
    route: '/export',
  },
];

export function GuidedTour() {
  const { tourActive, currentStep, nextStep, prevStep, endTour } = useTourStore();
  const t = useT();
  const navigate = useNavigate();

  const step = TOUR_STEPS[currentStep];
  const isLast = currentStep === TOUR_STEPS.length - 1;
  const isFirst = currentStep === 0;

  // Navigate to the step's route when step changes
  useEffect(() => {
    if (tourActive && step?.route) {
      navigate(step.route);
    }
  }, [tourActive, currentStep, step?.route, navigate]);

  // Keyboard navigation
  useEffect(() => {
    if (!tourActive) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') endTour();
      if (e.key === 'ArrowRight' && !isLast) nextStep();
      if (e.key === 'ArrowLeft' && !isFirst) prevStep();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [tourActive, isLast, isFirst, nextStep, prevStep, endTour]);

  const handleNext = useCallback(() => {
    if (isLast) {
      endTour();
    } else {
      nextStep();
    }
  }, [isLast, endTour, nextStep]);

  if (!tourActive || !step) return null;

  return (
    <>
      {/* Dark overlay */}
      <div className="fixed inset-0 z-50 bg-black/40" onClick={endTour} />

      {/* Tour card */}
      <div className="fixed left-1/2 top-1/2 z-50 w-[420px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-2xl bg-white p-6 shadow-2xl">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-100 text-indigo-600">
              <GraduationCap size={18} />
            </div>
            <span className="text-xs font-medium text-gray-400">
              {currentStep + 1} / {TOUR_STEPS.length}
            </span>
          </div>
          <button
            onClick={endTour}
            className="rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
          >
            <X size={16} />
          </button>
        </div>

        {/* Content */}
        <h3 className="mb-2 text-lg font-semibold text-gray-900">
          {t(step.titleKey)}
        </h3>
        <p className="mb-6 text-sm leading-relaxed text-gray-600">
          {t(step.contentKey)}
        </p>

        {/* Progress bar */}
        <div className="mb-4 h-1 rounded-full bg-gray-100">
          <div
            className="h-1 rounded-full bg-indigo-500 transition-all duration-300"
            style={{ width: `${((currentStep + 1) / TOUR_STEPS.length) * 100}%` }}
          />
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={prevStep}
            disabled={isFirst}
            className="flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm text-gray-500 hover:bg-gray-100 disabled:opacity-30"
          >
            <ChevronLeft size={16} />
            {t('tour.prev')}
          </button>
          <button
            onClick={endTour}
            className="text-xs text-gray-400 hover:text-gray-600"
          >
            {t('tour.skip')}
          </button>
          <button
            onClick={handleNext}
            className="flex items-center gap-1 rounded-lg bg-indigo-500 px-4 py-1.5 text-sm font-medium text-white hover:bg-indigo-600"
          >
            {isLast ? t('tour.finish') : t('tour.next')}
            {!isLast && <ChevronRight size={16} />}
          </button>
        </div>
      </div>
    </>
  );
}
