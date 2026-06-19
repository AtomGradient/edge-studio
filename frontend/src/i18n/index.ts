// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

import { useLocaleStore, type Locale } from './localeStore';
import en from './translations/en.json';
import zh from './translations/zh.json';

const translations: Record<Locale, Record<string, string>> = { en, zh };

/**
 * Get a translated string by key.
 * Falls back to English, then to the key itself.
 */
export function t(key: string): string {
  const locale = useLocaleStore.getState().locale;
  return translations[locale]?.[key] ?? translations.en[key] ?? key;
}

/**
 * React hook that returns a `t()` function bound to the current locale.
 * Re-renders when locale changes.
 */
export function useT() {
  const locale = useLocaleStore((s) => s.locale);
  return (key: string, params?: Record<string, string | number>): string => {
    let str = translations[locale]?.[key] ?? translations.en[key] ?? key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        str = str.replaceAll(`{${k}}`, String(v));
      }
    }
    return str;
  };
}

export { useLocaleStore, type Locale } from './localeStore';
