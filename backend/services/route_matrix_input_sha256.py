# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Input hash helper for route-matrix controlled-live allowlists.

The contract intentionally performs no Unicode normalization. It hashes the
raw UTF-8 bytes of the user-visible input text, matching Swift's
`Data(text.utf8)` path in EdgeRuntime.
"""

from __future__ import annotations

import hashlib


def route_matrix_input_sha256(text: str) -> str:
    """Return the lowercase SHA-256 hex digest of `text` encoded as raw UTF-8."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
