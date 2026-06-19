// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * Step 1: Focus selection — "What do you want AI to do?"
 * 6 cards: Chat, Coding, Vision, ASR, TTS, Voice Duplex.
 * Single click to select and auto-advance.
 */

import { useNavigate } from 'react-router-dom';
import { MessageSquare, Code2, Image, Mic, Volume2, Headphones } from 'lucide-react';
import { useSimpleStore } from '@/stores/simpleStore';
import { WizardShell } from '@/components/common/WizardShell';
import { useT } from '@/i18n';
import { WIZARD_STEPS_V2 } from './wizardStepsV2';
import { cn } from '@/lib/utils';

interface FocusOption {
  key: string;
  icon: React.ReactNode;
  titleKey: string;
  descKey: string;
  gradient: string;
  iconColor: string;
}

const FOCUS_OPTIONS: FocusOption[] = [
  {
    key: 'chat',
    icon: <MessageSquare size={28} />,
    titleKey: 'simple.v2.focus.chat',
    descKey: 'simple.v2.focus.chatDesc',
    gradient: 'from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20',
    iconColor: 'text-blue-500 dark:text-blue-400',
  },
  {
    key: 'coding',
    icon: <Code2 size={28} />,
    titleKey: 'simple.v2.focus.coding',
    descKey: 'simple.v2.focus.codingDesc',
    gradient: 'from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20',
    iconColor: 'text-emerald-500 dark:text-emerald-400',
  },
  {
    key: 'vision',
    icon: <Image size={28} />,
    titleKey: 'simple.v2.focus.vision',
    descKey: 'simple.v2.focus.visionDesc',
    gradient: 'from-purple-50 to-fuchsia-50 dark:from-purple-900/20 dark:to-fuchsia-900/20',
    iconColor: 'text-purple-500 dark:text-purple-400',
  },
  {
    key: 'asr',
    icon: <Mic size={28} />,
    titleKey: 'simple.v2.focus.asr',
    descKey: 'simple.v2.focus.asrDesc',
    gradient: 'from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20',
    iconColor: 'text-amber-500 dark:text-amber-400',
  },
  {
    key: 'tts',
    icon: <Volume2 size={28} />,
    titleKey: 'simple.v2.focus.tts',
    descKey: 'simple.v2.focus.ttsDesc',
    gradient: 'from-cyan-50 to-sky-50 dark:from-cyan-900/20 dark:to-sky-900/20',
    iconColor: 'text-cyan-500 dark:text-cyan-400',
  },
  {
    key: 'voice_duplex',
    icon: <Headphones size={28} />,
    titleKey: 'simple.v2.focus.voiceDuplex',
    descKey: 'simple.v2.focus.voiceDuplexDesc',
    gradient: 'from-rose-50 to-pink-50 dark:from-rose-900/20 dark:to-pink-900/20',
    iconColor: 'text-rose-500 dark:text-rose-400',
  },
];

export default function FocusSelectPage() {
  const t = useT();
  const navigate = useNavigate();
  const { focus, setFocus } = useSimpleStore();

  const handleSelect = (key: string) => {
    setFocus(key);
    navigate('/simple/tier');
  };

  return (
    <WizardShell
      steps={WIZARD_STEPS_V2(t)}
      currentStep={1}
      onBack={() => navigate('/simple')}
      helpKey="simple.v2.help.focus"
    >
      <div className="text-center">
        <h1 className="mb-2 text-3xl font-semibold tracking-tight text-stone-900 dark:text-stone-100">
          {t('simple.v2.focus.title')}
        </h1>
        <p className="mb-10 text-stone-500 dark:text-stone-400">
          {t('simple.v2.focus.subtitle')}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {FOCUS_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            type="button"
            onClick={() => handleSelect(opt.key)}
            className={cn(
              'group flex flex-col items-start gap-3 rounded-2xl border-2 p-6 text-left transition-all duration-200',
              'hover:shadow-lg hover:-translate-y-0.5',
              focus === opt.key
                ? 'border-stone-900 bg-stone-50 dark:border-stone-100 dark:bg-stone-800/50'
                : 'border-stone-200 bg-white hover:border-stone-400 dark:border-stone-800 dark:bg-stone-900 dark:hover:border-stone-600',
            )}
          >
            <div className={cn(
              'flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br',
              opt.gradient,
            )}>
              <span className={opt.iconColor}>{opt.icon}</span>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-stone-900 dark:text-stone-100">
                {t(opt.titleKey)}
              </h3>
              <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
                {t(opt.descKey)}
              </p>
            </div>
          </button>
        ))}
      </div>
    </WizardShell>
  );
}
