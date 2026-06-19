# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .app_dirs import data_path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _app_support_dir() -> Path:
    """Canonical mTLS identity directory — macOS FileVault protects user data."""
    base = data_path("certs")
    base.mkdir(parents=True, exist_ok=True)
    return base


def default_cert_path() -> Path:
    return _app_support_dir() / "identity.crt.pem"


def default_key_path() -> Path:
    return _app_support_dir() / "identity.key.pem"


def default_meta_path() -> Path:
    return _app_support_dir() / "identity.meta.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Identity:

    peer_id: str
    display_name: str
    cert_pem: bytes       # X.509 PEM
    cert_der: bytes       # DER — used to compute fingerprint / package into payloads verifiable by other devices
    key_pem: bytes        # EC private key PEM (PKCS#8)
    fingerprint: str      # lowercase hex SHA-256(cert_der), aligned with Swift side

    @property
    def cert_path(self) -> Path:
        return default_cert_path()

    @property
    def key_path(self) -> Path:
        return default_key_path()


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def fingerprint_of(cert_der: bytes) -> str:
    return hashlib.sha256(cert_der).hexdigest().lower()


# ---------------------------------------------------------------------------
# Identity load / create
# ---------------------------------------------------------------------------


def _default_display_name() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["scutil", "--get", "ComputerName"],
            capture_output=True, text=True, timeout=2,
        )
        name = r.stdout.strip()
        if name:
            return name
    except Exception:  # noqa: BLE001
        pass
    return os.uname().nodename.split(".")[0] or "EdgeStudio Mac"


def _stable_peer_id() -> tuple[str, str]:
    meta_path = default_meta_path()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            pid = str(meta.get("peer_id") or "").strip()
            if pid:
                return pid, str(meta.get("display_name") or _default_display_name())
        except Exception as exc:  # noqa: BLE001
            logger.warning("identity.meta.json corrupt, regenerating: %s", exc)

    peer_id = f"mac-{uuid.uuid4().hex}"
    display_name = _default_display_name()
    meta_path.write_text(json.dumps({
        "peer_id": peer_id,
        "display_name": display_name,
    }, indent=2), encoding="utf-8")
    os.chmod(meta_path, 0o600)
    return peer_id, display_name


def load_or_create(
    peer_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Identity:
    if peer_id is None or display_name is None:
        pid_auto, name_auto = _stable_peer_id()
        peer_id = peer_id or pid_auto
        display_name = display_name or name_auto

    cert_path = default_cert_path()
    key_path = default_key_path()
    if cert_path.exists() and key_path.exists():
        try:
            return _load_from_disk(
                peer_id=peer_id,
                display_name=display_name,
                cert_path=cert_path,
                key_path=key_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("existing identity unreadable, regenerating: %s", exc)

    return _generate_and_persist(peer_id=peer_id, display_name=display_name)


def _load_from_disk(
    *,
    peer_id: str,
    display_name: str,
    cert_path: Path,
    key_path: Path,
) -> Identity:
    cert_pem = cert_path.read_bytes()
    key_pem = key_path.read_bytes()
    cert = x509.load_pem_x509_certificate(cert_pem)
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    return Identity(
        peer_id=peer_id,
        display_name=display_name,
        cert_pem=cert_pem,
        cert_der=cert_der,
        key_pem=key_pem,
        fingerprint=fingerprint_of(cert_der),
    )


def _generate_and_persist(peer_id: str, display_name: str) -> Identity:
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

    now = __import__("datetime").datetime.utcnow()
    not_before = now - __import__("datetime").timedelta(minutes=5)           # clock skew
    not_after = now + __import__("datetime").timedelta(days=365 * 10)        # 10 years

    cn = f"EdgeStudio-{peer_id}"
    san = f"edgestudio-{peer_id}.local"

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "EdgeStudio"),
    ])

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(not_before)
        .not_valid_after(not_after)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(san)]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=True,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.ObjectIdentifier("1.3.6.1.5.5.7.3.1"),  # serverAuth
                x509.ObjectIdentifier("1.3.6.1.5.5.7.3.2"),  # clientAuth
            ]),
            critical=False,
        )
    )
    cert = builder.sign(private_key=private_key, algorithm=hashes.SHA256())

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    cert_der = cert.public_bytes(serialization.Encoding.DER)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    cert_path = default_cert_path()
    key_path = default_key_path()
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    os.chmod(cert_path, 0o600)
    os.chmod(key_path, 0o600)

    fp = fingerprint_of(cert_der)
    logger.info(
        "EdgeMesh identity created peer_id=%s fingerprint=%s cn=%s",
        peer_id, fp, cn,
    )

    return Identity(
        peer_id=peer_id,
        display_name=display_name,
        cert_pem=cert_pem,
        cert_der=cert_der,
        key_pem=key_pem,
        fingerprint=fp,
    )


def reset_identity() -> None:
    for p in (
        default_cert_path(),
        default_key_path(),
        default_meta_path(),
    ):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
