// Copyright © 2026 AtomGradient
// 版权所有 © 2026 质子梯度（北京）科技有限公司

/**
 * QR pairing modal.
 *
 * Click "Pair new device" on DevicesPage → this modal opens and immediately
 * requests a fresh pairing session from `POST /api/mesh/pair/qr`. It shows:
 *   - A QR encoding the full `QRPairingPayload` as JSON (Swift side decodes via
 *     `QRPairingPayload.decode(jsonString:)`)
 *   - A large monospace PIN as fallback if the user types it on iOS
 *   - A live countdown; when it hits 0 we auto-refresh to keep the UI useful
 *     without forcing the user to reopen the modal
 */

import { useEffect, useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { QRCodeSVG } from 'qrcode.react';
import { CheckCircle2, Loader2, RefreshCw, X } from 'lucide-react';
import { createPairing, listDevices, type QRPairingResponse } from '@/api/mesh';

interface Props {
  onClose: () => void;
}

export function QRPairingModal({ onClose }: Props) {
  const [session, setSession] = useState<QRPairingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Math.floor(Date.now() / 1000));
  const [consumed, setConsumed] = useState(false);
  /**
   * We can only claim "nonce consumed" if we first saw it **enter** pending[],
   * then watched it disappear. Otherwise we race the backend on first open:
   *   T=0   createPairing() returns → setSession(new)
   *   T<10ms  devicesQ returns cached `pending=[]` from previous session
   *   → false positive: "nonce not in pending, therefore consumed"
   * Track the positive observation explicitly.
   */
  const [nonceSeenInPending, setNonceSeenInPending] = useState(false);
  const queryClient = useQueryClient();

  // Tick every second for the countdown
  useEffect(() => {
    const id = setInterval(() => setNow(Math.floor(Date.now() / 1000)), 1000);
    return () => clearInterval(id);
  }, []);

  async function request() {
    setLoading(true);
    setError(null);
    // New session → reset the "have we observed our nonce in pending[] yet" guard.
    // Also invalidate the shared devices cache so the next poll fetches fresh data
    // (avoids racing against a stale pending[] that predates this new nonce).
    setNonceSeenInPending(false);
    setConsumed(false);
    queryClient.removeQueries({ queryKey: ['mesh', 'devices', 'qr-modal-watch'] });
    try {
      const resp = await createPairing();
      setSession(resp);
      setNow(Math.floor(Date.now() / 1000));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to create pairing session');
    } finally {
      setLoading(false);
    }
  }

  // Initial request on mount
  useEffect(() => {
    request();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const remaining = useMemo(() => {
    if (!session) return 0;
    return Math.max(0, session.payload.expiresAt - now);
  }, [session, now]);

  // Poll devices while showing a live session — detect when our nonce got consumed
  // (i.e. the peer successfully exchanged PIN / scanned QR and completed mTLS pairing).
  const devicesQ = useQuery({
    queryKey: ['mesh', 'devices', 'qr-modal-watch'],
    queryFn: listDevices,
    refetchInterval: 1500,
    enabled: !!session && !consumed,
  });

  // When our nonce disappears from `pending[]` while TTL hasn't elapsed → someone paired.
  // Two-phase detection to avoid race-condition false positives:
  //   Phase 1 — wait until we observe our own nonce in `pending[]` (proves backend saw the session)
  //   Phase 2 — then watch for it to disappear → "consumed"
  // Without phase 1, the cached listDevices response from a previous session would
  // trigger an immediate false "consumed" on modal re-open.
  useEffect(() => {
    if (!session || consumed) return;
    if (!devicesQ.data) return;
    if (remaining <= 0) return;      // gone because expired, not because consumed
    const myNonce = session.payload.nonce;
    const stillPending = devicesQ.data.pending.some((p) => p.nonce === myNonce);

    if (stillPending) {
      if (!nonceSeenInPending) setNonceSeenInPending(true);
      return;
    }
    // Not in pending[] — only claim consumption if we actually saw it there first.
    if (nonceSeenInPending) {
      setConsumed(true);
      queryClient.invalidateQueries({ queryKey: ['mesh', 'devices'] });
      setTimeout(() => onClose(), 800);
    }
  }, [devicesQ.data, session, remaining, consumed, nonceSeenInPending, onClose, queryClient]);

  // Auto-refresh when expired (but NOT when we've already been consumed — we're closing)
  useEffect(() => {
    if (session && remaining === 0 && !loading && !consumed) {
      request();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [remaining, consumed]);

  // Swift Codable expects sortedKeys + withoutEscapingSlashes.
  // A plain JSON.stringify matches Swift's decoder regardless of key order,
  // since decoding is key-lookup based. We only need to make sure we emit
  // a single-line JSON string (no extra whitespace) to keep the QR small.
  const qrContent = session ? JSON.stringify(session.payload) : '';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl dark:bg-stone-950"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded-lg p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-600 dark:hover:bg-stone-800"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        <h3 className="text-xl font-semibold text-stone-900 dark:text-stone-100">
          Pair a new device
        </h3>
        <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
          In your iPhone / iPad companion app, open <span className="font-medium">Settings → EdgeMesh → Add Device</span>.
          Scan this QR, or type the PIN code below.
        </p>

        <div className="mt-6 flex flex-col items-center gap-4">
          {consumed ? (
            <div className="flex h-64 w-64 flex-col items-center justify-center gap-3 rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300">
              <CheckCircle2 size={48} />
              <p className="text-sm font-medium">Device paired successfully</p>
              <p className="text-xs text-emerald-600 dark:text-emerald-400">Closing…</p>
            </div>
          ) : loading || !session ? (
            <div className="flex h-64 w-64 items-center justify-center rounded-lg border border-dashed border-stone-200 dark:border-stone-800">
              <Loader2 className="animate-spin text-stone-400" size={32} />
            </div>
          ) : (
            <div className="rounded-xl border border-stone-200 bg-white p-4 dark:border-stone-800 dark:bg-stone-900">
              <QRCodeSVG
                value={qrContent}
                size={240}
                level="M"
                includeMargin={false}
              />
            </div>
          )}

          {session && !consumed && (
            <>
              <div className="text-center">
                <p className="text-xs font-medium uppercase tracking-widest text-stone-400">
                  PIN code
                </p>
                <p className="mt-1 font-mono text-4xl font-bold tracking-[0.5em] text-stone-900 dark:text-stone-100">
                  {session.pin}
                </p>
              </div>

              <Countdown seconds={remaining} total={session.ttl_seconds} />
            </>
          )}

          {error && (
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          )}

          {!consumed && (
            <button
              onClick={request}
              disabled={loading}
              className="flex items-center gap-2 rounded-lg border border-stone-200 px-3 py-1.5 text-xs font-medium text-stone-600 hover:bg-stone-50 disabled:opacity-50 dark:border-stone-700 dark:text-stone-400 dark:hover:bg-stone-800"
            >
              <RefreshCw size={14} className={loading ? 'animate-spin' : undefined} />
              Generate a new code
            </button>
          )}
        </div>

        {session && (
          <details className="mt-4 rounded-lg border border-stone-100 bg-stone-50 p-3 text-xs dark:border-stone-800 dark:bg-stone-900">
            <summary className="cursor-pointer font-medium text-stone-600 dark:text-stone-400">
              Technical details
            </summary>
            <dl className="mt-2 grid grid-cols-[auto,1fr] gap-x-3 gap-y-1 text-stone-600 dark:text-stone-400">
              <dt>Peer ID</dt>
              <dd className="font-mono break-all">{session.payload.peerId}</dd>
              <dt>Fingerprint</dt>
              <dd className="font-mono break-all">{session.payload.certFingerprint}</dd>
              <dt>Endpoint</dt>
              <dd className="font-mono">
                {session.payload.endpoint.ipv4 ?? session.payload.endpoint.serviceName}
                :{session.payload.endpoint.port}
              </dd>
              <dt>Nonce</dt>
              <dd className="font-mono break-all">{session.payload.nonce}</dd>
            </dl>
          </details>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Countdown({ seconds, total }: { seconds: number; total: number }) {
  const pct = total > 0 ? (seconds / total) * 100 : 0;
  const tone = seconds <= 10 ? 'bg-red-500' : seconds <= 30 ? 'bg-amber-500' : 'bg-emerald-500';
  return (
    <div className="w-full max-w-xs">
      <div className="flex items-center justify-between text-xs text-stone-500 dark:text-stone-400">
        <span>Valid for</span>
        <span className="font-mono">{seconds}s</span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-stone-100 dark:bg-stone-800">
        <div
          className={`h-full transition-all duration-1000 ${tone}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
