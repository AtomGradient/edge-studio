# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import json
import logging
import errno
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from OpenSSL import SSL, crypto

from .certificate_manager import Identity, fingerprint_of, load_or_create
from .mesh_events import get_default_bus
from .pairing_manager import PairingManager, get_default_manager
from .trust_store import TrustStore, TrustedPeer, get_default_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame codec (wire-compatible with Swift FrameCodec)
# ---------------------------------------------------------------------------


MAX_FRAME_BYTES = 64 * 1024 * 1024


class FrameTooLargeError(Exception):
    pass


def encode_frame(payload: bytes) -> bytes:
    if len(payload) > MAX_FRAME_BYTES:
        raise FrameTooLargeError(f"frame {len(payload)} > {MAX_FRAME_BYTES}")
    return struct.pack(">I", len(payload)) + payload


def read_frame(sock: SSL.Connection, timeout_s: float = 30.0) -> Optional[bytes]:
    """Blocking read of one length-prefixed frame from an SSL.Connection.

    Returns None on clean EOF. Raises on framing / protocol errors.
    """
    header = _read_exact(sock, 4, timeout_s)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME_BYTES:
        raise FrameTooLargeError(f"advertised frame {length} > {MAX_FRAME_BYTES}")
    if length == 0:
        return b""
    payload = _read_exact(sock, length, timeout_s)
    if payload is None:
        raise ConnectionError("peer closed mid-frame")
    return payload


def _read_exact(sock: SSL.Connection, n: int, timeout_s: float) -> Optional[bytes]:
    buf = bytearray()
    deadline = time.time() + timeout_s
    while len(buf) < n:
        remaining = n - len(buf)
        try:
            chunk = sock.recv(remaining)
        except SSL.WantReadError:
            if time.time() > deadline:
                raise TimeoutError("recv deadline")
            time.sleep(0.01)
            continue
        except SSL.SysCallError as exc:
            if _is_retryable_ssl_syscall_error(exc):
                if time.time() > deadline:
                    raise TimeoutError("recv deadline")
                time.sleep(0.01)
                continue
            raise
        except SSL.ZeroReturnError:
            return None
        if not chunk:
            return None if not buf else None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Protocol messages (mirrors Swift WireMessage enum)
# ---------------------------------------------------------------------------


@dataclass
class PairHelloMessage:
    peer_id: str
    display_name: str
    cert_fingerprint: str
    nonce: str


@dataclass
class PairAckMessage:
    peer_id: str
    display_name: str
    cert_fingerprint: str


def _wire_envelope(op: str, payload: dict) -> bytes:
    return json.dumps({"op": op, "payload": payload}).encode("utf-8")


def _decode_envelope(data: bytes) -> tuple[str, dict]:
    obj = json.loads(data.decode("utf-8"))
    return str(obj["op"]), dict(obj["payload"])


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


OpHandler = Callable[[dict, "PeerContext"], Optional[dict]]


@dataclass
class PeerContext:
    address: tuple[str, int]
    cert_fingerprint: str
    cert_der: bytes
    trust_store: TrustStore
    pairing_manager: PairingManager
    identity: Identity
    trusted_peer: Optional[TrustedPeer] = None   # None → still pairing, filled by pair_hello after handshake


def _is_retryable_ssl_syscall_error(exc: SSL.SysCallError) -> bool:
    """pyOpenSSL can surface nonblocking EAGAIN/EINTR as SysCallError."""

    for arg in getattr(exc, "args", ()):
        if isinstance(arg, int) and arg in {errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR}:
            return True
        if isinstance(arg, str) and arg.upper() in {"EAGAIN", "EWOULDBLOCK", "EINTR"}:
            return True
    return False


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class MeshTransportServer:

    def __init__(
        self,
        identity: Identity,
        trust_store: TrustStore,
        pairing_manager: PairingManager,
        host: str = "0.0.0.0",
        port: int = 18843,
    ) -> None:
        self.identity = identity
        self.trust_store = trust_store
        self.pairing_manager = pairing_manager
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._ctx: Optional[SSL.Context] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._running = False
        self._handlers: dict[str, OpHandler] = {}
        self._register_builtin_handlers()
        # Active connection registry — keyed by peer_id. Populated after successful
        # pair_hello / post-handshake trust lookup. Enables revoke/delete APIs to
        # proactively cut the mTLS socket instead of waiting for iOS to poll AND
        # enables server-initiated push (adapter_offer, training_available, ...).
        #
        # Entry: (ssl_conn, send_lock). send_lock serializes sendall() between the
        # op_loop (reply path) and external callers (send_to_peer push path) —
        # without it, interleaved partial sends corrupt frame boundaries.
        self._active_conns: dict[str, tuple[SSL.Connection, threading.Lock]] = {}
        self._active_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def register_handler(self, op: str, handler: OpHandler) -> None:
        self._handlers[op] = handler

    def _register_builtin_handlers(self) -> None:
        self.register_handler("pair_hello", self._handle_pair_hello)
        self.register_handler("ping", self._handle_ping)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._ctx = self._build_ssl_context()
        self._sock = self._bind_listen()
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="mesh-transport-accept",
            daemon=True,
        )
        self._accept_thread.start()
        logger.info(
            "MeshTransport listening on %s:%d fingerprint=%s peer_id=%s",
            self.host, self.port, self.identity.fingerprint, self.identity.peer_id,
        )

    def stop(self) -> None:
        self._running = False
        try:
            if self._sock:
                self._sock.close()
        except Exception:  # noqa: BLE001
            pass
        self._sock = None
        logger.info("MeshTransport stopped")

    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # SSL context
    # ------------------------------------------------------------------

    def _build_ssl_context(self) -> SSL.Context:
        ctx = SSL.Context(SSL.TLS_SERVER_METHOD)
        # TLS 1.3 only — match Swift min=max=.TLSv13
        ctx.set_min_proto_version(SSL.TLS1_3_VERSION)
        ctx.set_max_proto_version(SSL.TLS1_3_VERSION)

        # Load our identity
        cert_obj = crypto.load_certificate(crypto.FILETYPE_PEM, self.identity.cert_pem)
        key_obj = crypto.load_privatekey(crypto.FILETYPE_PEM, self.identity.key_pem)
        ctx.use_certificate(cert_obj)
        ctx.use_privatekey(key_obj)
        ctx.check_privatekey()

        # Request client cert, but **do not** verify against CA at TLS layer.
        # Fingerprint pinning happens post-handshake in the accept loop.
        # Without a verify callback returning True, pyOpenSSL's default rejects self-signed client certs.
        ctx.set_verify(
            SSL.VERIFY_PEER | SSL.VERIFY_CLIENT_ONCE,
            lambda _conn, _cert, _errno, _depth, _ok: True,
        )
        # Because every iOS device has its own self-signed cert, none of them chain to a CA we know.
        # The verify callback above always returns True; we re-check the fingerprint afterwards.
        return ctx

    def _bind_listen(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.listen(8)
        return sock

    # ------------------------------------------------------------------
    # Accept / per-connection loop
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._sock is not None
        assert self._ctx is not None
        while self._running:
            try:
                raw, addr = self._sock.accept()
            except OSError:
                if self._running:
                    logger.exception("accept failed")
                return
            t = threading.Thread(
                target=self._handle_connection,
                args=(raw, addr),
                name=f"mesh-conn-{addr[0]}:{addr[1]}",
                daemon=True,
            )
            t.start()

    def _handle_connection(self, raw: socket.socket, addr: tuple[str, int]) -> None:
        raw.settimeout(30.0)
        try:
            ssl_conn = SSL.Connection(self._ctx, raw)
            ssl_conn.set_accept_state()
            self._do_handshake(ssl_conn)

            peer_der = self._peer_cert_der(ssl_conn)
            if peer_der is None:
                logger.warning("mesh: peer %s presented no cert, closing", addr)
                return
            peer_fp = fingerprint_of(peer_der)

            ctx = PeerContext(
                address=addr,
                cert_fingerprint=peer_fp,
                cert_der=peer_der,
                trust_store=self.trust_store,
                pairing_manager=self.pairing_manager,
                identity=self.identity,
            )

            # If already trusted, preload the record
            existing = self.trust_store.lookup_by_fingerprint(peer_fp)
            if existing is not None:
                if existing.revoked:
                    logger.warning(
                        "mesh: revoked peer %s fp=%s allowed for restricted ops",
                        existing.peer_id, peer_fp,
                    )
                    ctx.trusted_peer = existing
                else:
                    ctx.trusted_peer = existing
                    self.trust_store.touch_last_seen(existing.peer_id)
                    # Register for revoke-time push disconnect
                    self._register_active(existing.peer_id, ssl_conn)

            try:
                self._op_loop(ssl_conn, ctx)
            finally:
                if ctx.trusted_peer is not None:
                    self._unregister_active(ctx.trusted_peer.peer_id, ssl_conn)

        except SSL.Error as exc:
            logger.warning("mesh: TLS error from %s: %s", addr, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("mesh: connection from %s failed: %s", addr, exc)
        finally:
            try:
                ssl_conn.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                raw.close()
            except Exception:  # noqa: BLE001
                pass

    def _do_handshake(self, ssl_conn: SSL.Connection) -> None:
        deadline = time.time() + 10.0
        while True:
            try:
                ssl_conn.do_handshake()
                return
            except SSL.WantReadError:
                if time.time() > deadline:
                    raise TimeoutError("handshake deadline")
                time.sleep(0.01)
            except SSL.WantWriteError:
                if time.time() > deadline:
                    raise TimeoutError("handshake deadline")
                time.sleep(0.01)

    def _peer_cert_der(self, ssl_conn: SSL.Connection) -> Optional[bytes]:
        cert = ssl_conn.get_peer_certificate()
        if cert is None:
            return None
        return crypto.dump_certificate(crypto.FILETYPE_ASN1, cert)

    # ------------------------------------------------------------------
    # Op loop
    # ------------------------------------------------------------------

    def _op_loop(self, ssl_conn: SSL.Connection, ctx: PeerContext) -> None:
        while True:
            # If a trusted peer was deleted, close ASAP. Revoked peers stay
            # connected only long enough to send the restricted
            # `peer_trust_deleted` control op; all other ops are rejected below.
            if ctx.trusted_peer is not None:
                current = self.trust_store.lookup(ctx.trusted_peer.peer_id)
                if current is None:
                    logger.info(
                        "mesh: peer %s deleted mid-session, closing",
                        ctx.trusted_peer.peer_id,
                    )
                    return

            try:
                frame = read_frame(ssl_conn)
            except TimeoutError:
                continue
            if frame is None:
                return
            try:
                op, payload = _decode_envelope(frame)
            except Exception as exc:  # noqa: BLE001
                logger.warning("mesh: bad frame from %s: %s", ctx.address, exc)
                self._send_error(ssl_conn, "bad_frame", str(exc))
                return

            handler = self._handlers.get(op)
            if handler is None:
                logger.info("mesh: unknown op %s from %s", op, ctx.address)
                self._send_error(ssl_conn, "unknown_op", op)
                continue

            if ctx.trusted_peer is not None:
                current = self.trust_store.lookup(ctx.trusted_peer.peer_id)
                if current is None:
                    logger.info(
                        "mesh: peer %s deleted before op=%s, closing",
                        ctx.trusted_peer.peer_id,
                        op,
                    )
                    return
                ctx.trusted_peer = current
                if current.revoked and op != "peer_trust_deleted":
                    logger.info(
                        "mesh: rejected op=%s from revoked peer %s",
                        op,
                        current.peer_id,
                    )
                    self._send_error(ssl_conn, "peer_revoked", "peer revoked")
                    return

            # Enhanced observability: entry op routing log (debug level to avoid ping spam, info level is noisy)
            payload_size = len(frame) if isinstance(frame, (bytes, bytearray)) else 0
            peer_label = (
                ctx.trusted_peer.peer_id if ctx.trusted_peer else f"untrusted@{ctx.address}"
            )
            logger.debug("mesh: op=%s peer=%s frame_bytes=%d", op, peer_label, payload_size)

            was_trusted_before = ctx.trusted_peer is not None
            try:
                reply = handler(payload, ctx)
            except Exception as exc:  # noqa: BLE001
                logger.exception("mesh: handler %s failed: %s", op, exc)
                self._send_error(ssl_conn, "handler_failed", str(exc))
                continue

            # Detect trust-promotion (e.g. pair_hello flipped ctx.trusted_peer from None→Some);
            # register this connection for push-disconnect.
            if not was_trusted_before and ctx.trusted_peer is not None:
                self._register_active(ctx.trusted_peer.peer_id, ssl_conn)

            if reply is not None:
                reply_op, reply_payload = reply["op"], reply["payload"]
                send_lock = self._send_lock_for(ctx.trusted_peer.peer_id if ctx.trusted_peer else None)
                frame = encode_frame(_wire_envelope(reply_op, reply_payload))
                if send_lock is not None:
                    with send_lock:
                        ssl_conn.sendall(frame)
                else:
                    ssl_conn.sendall(frame)

    def _send_error(self, ssl_conn: SSL.Connection, code: str, message: str) -> None:
        try:
            ssl_conn.sendall(encode_frame(_wire_envelope("error", {
                "code": code, "message": message,
            })))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Active connection registry (push-disconnect support)
    # ------------------------------------------------------------------

    def _register_active(self, peer_id: str, ssl_conn: Optional[SSL.Connection]) -> None:
        if ssl_conn is None:
            return
        with self._active_lock:
            was_new = peer_id not in self._active_conns
            # Replace any stale reference — keeping the latest connection wins.
            self._active_conns[peer_id] = (ssl_conn, threading.Lock())
        if was_new:
            get_default_bus().broadcast({"type": "peer_connected", "peer_id": peer_id})

    def _unregister_active(self, peer_id: str, ssl_conn: SSL.Connection) -> None:
        removed = False
        with self._active_lock:
            # Only remove if it's still the same connection (defend against races
            # where peer reconnects before the old worker thread notices the close).
            entry = self._active_conns.get(peer_id)
            if entry is not None and entry[0] is ssl_conn:
                self._active_conns.pop(peer_id, None)
                removed = True
        if removed:
            get_default_bus().broadcast({"type": "peer_disconnected", "peer_id": peer_id})

    def _send_lock_for(self, peer_id: Optional[str]) -> Optional[threading.Lock]:
        if peer_id is None:
            return None
        with self._active_lock:
            entry = self._active_conns.get(peer_id)
            return entry[1] if entry else None

    def send_frame_to_peer(
        self,
        peer_id: str,
        frame_payload: bytes,
        *,
        op_label: str = "raw_frame",
    ) -> bool:
        """Server-initiated raw frame push to a connected peer.

        Thread-safe: serialized with op_loop's reply writes via per-connection
        send_lock to preserve frame boundaries.

        Handles pyOpenSSL's non-blocking I/O by looping on WantWriteError /
        WantReadError with a short sleep — otherwise bulk chunk streaming (D1
        adapter distribution sends ~400 frames back-to-back) trips on transient
        buffer pressure and spuriously returns False.

        Returns True if the frame was handed to the socket; False if peer is not
        currently connected or send failed after retries.
        """
        if not isinstance(frame_payload, (bytes, bytearray)):
            raise TypeError("frame_payload must be bytes")
        with self._active_lock:
            entry = self._active_conns.get(peer_id)
        if entry is None:
            return False
        ssl_conn, send_lock = entry
        frame = encode_frame(bytes(frame_payload))
        deadline = time.time() + 30.0  # hard cap per-frame
        try:
            with send_lock:
                while True:
                    try:
                        ssl_conn.sendall(frame)
                        return True
                    except (SSL.WantWriteError, SSL.WantReadError):
                        if time.time() > deadline:
                            raise TimeoutError(
                                f"sendall per-frame deadline for peer {peer_id}"
                            )
                        time.sleep(0.005)
                    except SSL.SysCallError as exc:
                        if _is_retryable_ssl_syscall_error(exc):
                            if time.time() > deadline:
                                raise TimeoutError(
                                    f"sendall per-frame deadline for peer {peer_id}"
                                )
                            time.sleep(0.005)
                            continue
                        raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "send_frame_to_peer peer=%s op=%s failed: %s",
                peer_id,
                op_label,
                exc,
            )
            self._unregister_active(peer_id, ssl_conn)
            # Dump to a file so the operator can inspect without terminal access.
            try:
                import os, traceback as _tb
                with open("/tmp/edge_send_errors.txt", "a") as f:
                    f.write(
                        f"[{time.time():.0f}] peer={peer_id} op={op_label} "
                        f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}\n"
                    )
                    os.fsync(f.fileno())
            except Exception:  # noqa: BLE001
                pass
            return False

    def send_to_peer(self, peer_id: str, op: str, payload: dict) -> bool:
        """Server-initiated JSON-envelope push to a connected peer."""
        return self.send_frame_to_peer(
            peer_id,
            _wire_envelope(op, payload),
            op_label=op,
        )

    def disconnect_peer(
        self,
        peer_id: str,
        reason: str = "revoked",
        *,
        code: str = "peer_revoked",
    ) -> bool:
        """Force-close any active mTLS session for a given peer.

        Called by revoke / delete HTTP endpoints so the iPhone knows immediately
        (instead of waiting for its next reconcile poll). Returns True if a
        connection was actually closed.
        """
        with self._active_lock:
            entry = self._active_conns.pop(peer_id, None)
        if entry is None:
            return False
        conn, send_lock = entry
        try:
            with send_lock:
                conn.sendall(encode_frame(_wire_envelope("error", {
                    "code": code, "message": reason,
                })))
        except Exception:  # noqa: BLE001
            pass
        try:
            conn.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            conn.get_socket().close()
        except Exception:  # noqa: BLE001
            pass
        logger.info("mesh: force-disconnected peer %s (reason=%s)", peer_id, reason)
        return True

    def active_peer_ids(self) -> list[str]:
        """Snapshot of currently-connected peer_ids (for /api/mesh/status + tests)."""
        with self._active_lock:
            return list(self._active_conns.keys())

    def is_peer_connected(self, peer_id: str) -> bool:
        with self._active_lock:
            return peer_id in self._active_conns

    # ------------------------------------------------------------------
    # Built-in handlers
    # ------------------------------------------------------------------

    def _handle_pair_hello(self, payload: dict, ctx: PeerContext) -> dict:
        peer_id = str(payload.get("peerId", "")).strip()
        display_name = str(payload.get("displayName", "")).strip()
        claimed_fp = str(payload.get("certFingerprint", "")).lower().strip()
        nonce = str(payload.get("nonce", "")).strip()

        if not (peer_id and display_name and claimed_fp and nonce):
            raise ValueError("pair_hello missing required fields")

        # TLS layer real peer fingerprint must match the claim
        if claimed_fp != ctx.cert_fingerprint:
            raise ValueError(
                f"pair_hello fingerprint mismatch: claim={claimed_fp} tls={ctx.cert_fingerprint}"
            )

        # Redeem nonce (one-time use)
        pending = ctx.pairing_manager.consume_nonce(nonce)
        if pending is None:
            raise ValueError("pair_hello nonce unknown or expired")

        # pending records the Mac-issued payload (Mac's own fingerprint);
        # The client here is iOS, role should be opposite of QR generation (Mac=brain, iOS=sensor).
        # P0 decision: fall back to client's self-declared role, default sensor.
        peer_role = "sensor"

        now_ms = int(time.time() * 1000)
        new_peer = TrustedPeer(
            peer_id=peer_id,
            display_name=display_name,
            fingerprint=ctx.cert_fingerprint,
            role=peer_role,
            paired_at_ms=now_ms,
            last_seen_at_ms=now_ms,
            revoked=False,
            cert_der=ctx.cert_der,
        )
        ctx.trust_store.upsert(new_peer)
        ctx.trusted_peer = new_peer
        # Registration into active_conns happens in `_op_loop` post-dispatch
        # (has access to ssl_conn and detects None→Some trust transition).

        get_default_bus().broadcast({
            "type": "peer_paired",
            "peer_id": peer_id,
            "display_name": display_name,
            "fingerprint": ctx.cert_fingerprint,
        })

        logger.info(
            "Pair success peer=%s fingerprint=%s from %s",
            peer_id, ctx.cert_fingerprint, ctx.address,
        )

        ack = {
            "peerId": ctx.identity.peer_id,
            "displayName": ctx.identity.display_name,
            "certFingerprint": ctx.identity.fingerprint,
        }
        return {"op": "pair_ack", "payload": ack}

    def _handle_ping(self, payload: dict, ctx: PeerContext) -> dict:
        # Keepalive doubles as presence signal — refresh last_seen on every ping.
        # Stage 3 P1.1: legacy client (PingMessage has only timestamp) → touch_last_seen legacy path;
        # new client (PingMessage.stats != None) → update_peer_stats writes last_seen + stats in one call.
        if ctx.trusted_peer is not None:
            peer_id = ctx.trusted_peer.peer_id
            stats = payload.get("stats") if isinstance(payload, dict) else None
            try:
                if isinstance(stats, dict):
                    self.trust_store.update_peer_stats(peer_id, stats)
                    logger.info(
                        "[ping] peer=%s stats appId=%s eventStore=%s factsCls=%s factsRaw=%s",
                        peer_id,
                        stats.get("appId"),
                        stats.get("eventStoreTotal"),
                        stats.get("factsClassified"),
                        stats.get("factsRawUnclassified"),
                    )
                else:
                    self.trust_store.touch_last_seen(peer_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ping last_seen/stats failed: %s", exc)
        return {"op": "ping", "payload": {"timestamp": int(time.time() * 1000)}}


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------


_default_server: Optional[MeshTransportServer] = None
_default_server_lock = threading.Lock()


def get_default_server() -> MeshTransportServer:
    global _default_server
    with _default_server_lock:
        if _default_server is None:
            _default_server = MeshTransportServer(
                identity=load_or_create(),
                trust_store=get_default_store(),
                pairing_manager=get_default_manager(),
            )
        return _default_server
