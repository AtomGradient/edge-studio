# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Shared exceptions for EdgeStudio core overlays."""

from __future__ import annotations


class EdgeStudioCoreError(RuntimeError):
    """Base class for edgestudio_core runtime errors."""


class UnsupportedRuntimeMutation(EdgeStudioCoreError):
    """Raised when a public runtime does not expose the requested mutation API."""
