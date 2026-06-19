# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Shared model catalog filters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def matches_tts_variant(model: Mapping[str, Any], tts_variant: str | None = None) -> bool:
    """Return whether a catalog model matches the requested TTS variant.

    The filter only applies to TTS models. Other categories stay visible so a
    duplex user can still search for an LLM by name while a TTS variant is set.
    """
    variant = (tts_variant or "").strip().lower()
    if not variant or model.get("category") != "tts":
        return True

    haystack = " ".join(
        str(model.get(key, ""))
        for key in ("id", "name", "family", "download_hint")
    ).lower()
    return variant in haystack
