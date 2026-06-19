# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import logging
import socket
import threading
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


SERVICE_TYPE = "_edgemesh._tcp.local."


def _local_ipv4_addresses() -> list[str]:
    addrs: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and ip != "127.0.0.1" and ip not in addrs:
                addrs.append(ip)
    except Exception:  # noqa: BLE001
        pass
    if not addrs:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            addrs.append(s.getsockname()[0])
            s.close()
        except Exception:  # noqa: BLE001
            pass
    return addrs


def _primary_ipv4(addrs: list[str]) -> str:
    for ip in addrs:
        if ip.startswith(("192.", "10.", "172.")):
            return ip
    return addrs[0] if addrs else ""


def _mdns_local_hostname() -> str:
    import re
    import subprocess

    try:
        r = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            capture_output=True, text=True, timeout=2,
        )
        cand = r.stdout.strip()
        if cand and re.match(r"^[A-Za-z0-9][A-Za-z0-9-]*$", cand):
            return cand
    except Exception:  # noqa: BLE001
        pass
    return socket.gethostname().split(".")[0] or "edgestudio"


class MeshDiscoveryBroadcaster:

    def __init__(
        self,
        *,
        peer_id: str,
        display_name: str,
        cert_fingerprint: str,
        http_port: int = 18842,
        mesh_port: int = 18843,
        studio_version: str = "2.0.0",
    ) -> None:
        self.peer_id = peer_id
        self.display_name = display_name
        self.cert_fingerprint = cert_fingerprint
        self.http_port = http_port
        self.mesh_port = mesh_port
        self.studio_version = studio_version

        self._zc = None
        self._service_info = None
        self._running = False
        self._lock = threading.Lock()
        self._instance_id = uuid.uuid4().hex[:8]
        self._service_name = ""

    def start(self) -> None:
        from zeroconf import IPVersion, ServiceInfo, Zeroconf

        with self._lock:
            if self._running:
                return
            self._zc = Zeroconf(ip_version=IPVersion.V4Only)

            hostname = _mdns_local_hostname()
            instance = f"EdgeStudio-{hostname}-{self._instance_id}"
            self._service_name = f"{instance}.{SERVICE_TYPE}"

            ipv4s = _local_ipv4_addresses()
            primary = _primary_ipv4(ipv4s)

            address_bytes = [socket.inet_aton(ip) for ip in ipv4s]

            server_host = f"edgestudio-{self._instance_id}.local."

            txt = {
                b"role": b"brain",
                b"tls_version": b"1.3",
                b"proto_ver": b"1",
                b"peer_id": self.peer_id.encode("utf-8"),
                b"fingerprint": self.cert_fingerprint.lower().encode("utf-8"),
                b"ipv4": primary.encode("utf-8"),
                b"http_port": str(self.http_port).encode("utf-8"),
                b"mesh_port": str(self.mesh_port).encode("utf-8"),
                b"display_name": self.display_name.encode("utf-8"),
                b"studio_version": self.studio_version.encode("utf-8"),
                b"platform": b"macOS",
            }

            self._service_info = ServiceInfo(
                type_=SERVICE_TYPE,
                name=self._service_name,
                addresses=address_bytes,
                port=int(self.mesh_port),       # Bonjour main port = mTLS
                properties=txt,
                server=server_host,
            )
            self._zc.register_service(self._service_info)
            self._running = True
            logger.info(
                "MeshDiscovery registered %s on mesh=%d http=%d ipv4=%s fingerprint=%s",
                self._service_name, self.mesh_port, self.http_port,
                primary, self.cert_fingerprint,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            try:
                if self._zc is not None and self._service_info is not None:
                    self._zc.unregister_service(self._service_info)
            except Exception:  # noqa: BLE001
                pass
            try:
                if self._zc is not None:
                    self._zc.close()
            except Exception:  # noqa: BLE001
                pass
            self._zc = None
            self._service_info = None
            self._running = False
            logger.info("MeshDiscovery stopped")

    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "service_type": SERVICE_TYPE,
            "service_name": self._service_name,
            "peer_id": self.peer_id,
            "fingerprint": self.cert_fingerprint,
            "http_port": self.http_port,
            "mesh_port": self.mesh_port,
            "ipv4": _primary_ipv4(_local_ipv4_addresses()),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_default_broadcaster: Optional[MeshDiscoveryBroadcaster] = None
_default_broadcaster_lock = threading.Lock()


def get_default_broadcaster(
    *,
    peer_id: str,
    display_name: str,
    cert_fingerprint: str,
    http_port: int = 18842,
    mesh_port: int = 18843,
    studio_version: str = "2.0.0",
) -> MeshDiscoveryBroadcaster:
    global _default_broadcaster
    with _default_broadcaster_lock:
        if _default_broadcaster is None:
            _default_broadcaster = MeshDiscoveryBroadcaster(
                peer_id=peer_id,
                display_name=display_name,
                cert_fingerprint=cert_fingerprint,
                http_port=http_port,
                mesh_port=mesh_port,
                studio_version=studio_version,
            )
        return _default_broadcaster


def get_primary_ipv4() -> str:
    return _primary_ipv4(_local_ipv4_addresses())
