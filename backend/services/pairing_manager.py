# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import base64
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


DEFAULT_EXPIRY_SECONDS = 60                 # QR display 1-minute timeout
PIN_LOCKOUT_WINDOW_SECONDS = 10
PIN_LOCKOUT_MAX_FAILURES = 3


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PairingPayload:

    version: int
    peer_id: str
    display_name: str
    role: str                        # "brain" | "sensor" | "peer"
    service_type: str
    service_name: str
    ipv4: Optional[str]
    port: int
    cert_fingerprint: str            # 64-char hex
    nonce: str                       # base64url(16 random bytes)
    expires_at: int                  # Unix seconds

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "peerId": self.peer_id,
            "displayName": self.display_name,
            "role": self.role,
            "endpoint": {
                "serviceType": self.service_type,
                "serviceName": self.service_name,
                "ipv4": self.ipv4,
                "port": self.port,
            },
            "certFingerprint": self.cert_fingerprint,
            "nonce": self.nonce,
            "expiresAt": self.expires_at,
        }

    @property
    def pin(self) -> str:
        raw = _b64url_decode(self.nonce)
        if len(raw) < 4:
            return "000000"
        return _base32_rfc4648(raw[:4])[:6].upper()

    def is_expired(self, now: Optional[float] = None) -> bool:
        now_s = now if now is not None else time.time()
        return now_s >= self.expires_at


@dataclass
class PendingSession:
    payload: PairingPayload
    created_at: float
    # Human approval gate — Mac-initiated sessions (/pair/qr) are pre-approved
    # (the user already clicked "Pair new device"); iOS-initiated tap-to-pair
    # sessions (/pair/request) start approved=False until the Mac user
    # visually confirms the PIN and clicks Approve.
    approved: bool = True

    @property
    def expires_at(self) -> float:
        return float(self.payload.expires_at)


@dataclass
class _IpFailureWindow:
    failures: list[float] = field(default_factory=list)

    def prune(self, now: float) -> None:
        cutoff = now - PIN_LOCKOUT_WINDOW_SECONDS
        self.failures = [t for t in self.failures if t > cutoff]

    def record(self, now: float) -> None:
        self.failures.append(now)

    def is_locked(self, now: float) -> bool:
        self.prune(now)
        return len(self.failures) >= PIN_LOCKOUT_MAX_FAILURES


# ---------------------------------------------------------------------------
# Helpers — base64url / base32 (consistent with Swift QRCodec)
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode(s + pad)


_BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


def _base32_rfc4648(data: bytes) -> str:
    result = []
    buffer = 0
    bits = 0
    for byte in data:
        buffer = (buffer << 8) | byte
        bits += 8
        while bits >= 5:
            idx = (buffer >> (bits - 5)) & 0x1F
            result.append(_BASE32_ALPHABET[idx])
            bits -= 5
    if bits > 0:
        idx = (buffer << (5 - bits)) & 0x1F
        result.append(_BASE32_ALPHABET[idx])
    return "".join(result)


# ---------------------------------------------------------------------------
# PairingManager
# ---------------------------------------------------------------------------


class PairingManager:

    def __init__(self) -> None:
        self._sessions: dict[str, PendingSession] = {}      # nonce -> session
        self._pin_index: dict[str, str] = {}                # pin -> nonce
        self._ip_failures: dict[str, _IpFailureWindow] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    def create_session(
        self,
        *,
        peer_id: str,
        display_name: str,
        role: str,
        service_type: str,
        service_name: str,
        ipv4: Optional[str],
        port: int,
        cert_fingerprint: str,
        ttl_seconds: int = DEFAULT_EXPIRY_SECONDS,
        approved: bool = True,
    ) -> PairingPayload:
        nonce_bytes = secrets.token_bytes(16)
        nonce = _b64url_encode(nonce_bytes)
        now = time.time()
        expires_at = int(now) + ttl_seconds

        payload = PairingPayload(
            version=1,
            peer_id=peer_id,
            display_name=display_name,
            role=role,
            service_type=service_type,
            service_name=service_name,
            ipv4=ipv4,
            port=port,
            cert_fingerprint=cert_fingerprint.lower(),
            nonce=nonce,
            expires_at=expires_at,
        )

        with self._lock:
            self._prune_expired_locked(now)
            self._sessions[nonce] = PendingSession(
                payload=payload, created_at=now, approved=approved,
            )
            self._pin_index[payload.pin] = nonce

        logger.info(
            "Pairing session created peer=%s pin=%s expires_in=%ds approved=%s",
            peer_id, payload.pin, ttl_seconds, approved,
        )
        return payload

    def approve_session(self, nonce: str) -> bool:
        """Mark a pending session as approved by the Mac user.

        Returns True if the session exists and transitioned to approved,
        False if nonce unknown / expired.
        """
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            session = self._sessions.get(nonce)
            if session is None or session.payload.is_expired(now):
                return False
            session.approved = True
            return True

    def get_session(self, nonce: str) -> Optional[PendingSession]:
        """Inspect a pending session (for status polling)."""
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            return self._sessions.get(nonce)

    # ------------------------------------------------------------------
    # Lookup / consume
    # ------------------------------------------------------------------

    def lookup_by_pin(self, pin: str, client_ip: str) -> PairingPayload:
        pin = (pin or "").strip().upper()
        if len(pin) != 6 or not all(c in _BASE32_ALPHABET for c in pin):
            raise ValueError(f"invalid PIN format: {pin!r}")

        now = time.time()
        with self._lock:
            win = self._ip_failures.get(client_ip) or _IpFailureWindow()
            if win.is_locked(now):
                raise PermissionError(
                    f"too many PIN attempts from {client_ip}, lockout active"
                )
            self._ip_failures[client_ip] = win

            self._prune_expired_locked(now)
            nonce = self._pin_index.get(pin)
            if nonce is None:
                win.record(now)
                raise KeyError(f"PIN {pin} not found or expired")
            session = self._sessions.get(nonce)
            if session is None or session.payload.is_expired(now):
                win.record(now)
                raise KeyError(f"PIN {pin} session gone")
            # Block unapproved iOS-initiated sessions at PIN exchange too —
            # iOS must wait for the Mac user to click Approve on the web UI.
            # Don't penalize with a failure counter (PIN is correct), just
            # report not-yet-approved.
            if not session.approved:
                raise KeyError(f"PIN {pin} session awaiting Mac approval")

            return session.payload

    def consume_nonce(self, nonce: str) -> Optional[PairingPayload]:
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            session = self._sessions.pop(nonce, None)
            if session is None:
                return None
            # Also clean up from pin index
            self._pin_index.pop(session.payload.pin, None)
            if session.payload.is_expired(now):
                return None
            if not session.approved:
                return None
            return session.payload

    # ------------------------------------------------------------------
    # Introspection (for tests / status pages)
    # ------------------------------------------------------------------

    def list_pending(self) -> list[PairingPayload]:
        now = time.time()
        with self._lock:
            self._prune_expired_locked(now)
            return [s.payload for s in self._sessions.values()]

    def reset(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._pin_index.clear()
            self._ip_failures.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_expired_locked(self, now: float) -> None:
        stale_nonces = [
            n for n, s in self._sessions.items() if s.payload.is_expired(now)
        ]
        for n in stale_nonces:
            payload = self._sessions.pop(n).payload
            self._pin_index.pop(payload.pin, None)
        if stale_nonces:
            logger.info("Pruned %d expired pairing sessions", len(stale_nonces))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_default_manager: Optional[PairingManager] = None
_default_manager_lock = threading.Lock()


def get_default_manager() -> PairingManager:
    global _default_manager
    with _default_manager_lock:
        if _default_manager is None:
            _default_manager = PairingManager()
        return _default_manager
