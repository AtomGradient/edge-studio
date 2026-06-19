# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Server configuration."""

from __future__ import annotations

import os


HOST = os.getenv("VLM_HOST", "127.0.0.1")
PORT = int(os.getenv("VLM_PORT", "18842"))

ALLOWED_ORIGINS = os.getenv(
    "VLM_CORS_ORIGINS",
    f"http://localhost:5173,http://localhost:{PORT},http://127.0.0.1:{PORT}",
).split(",")
# Regex: also allow any local/private-network IP on the Vite dev port.
# Edge Studio is a local dev tool — no reason to restrict LAN origins.
ALLOWED_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$"

# Filesystem browse whitelist — restrict browsable directories.
# Default: user home only. Set VLM_BROWSE_ROOTS to add more (comma-separated).
_home = os.path.expanduser("~")
BROWSE_ROOTS: list[str] = [
    os.path.realpath(p)
    for p in os.getenv("VLM_BROWSE_ROOTS", _home).split(",")
    if p.strip()
]
