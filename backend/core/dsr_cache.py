# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""DSR KV-cache integration boundary for the bundled edgestudio_core runtime."""

from __future__ import annotations

from typing import Any


class DSRCacheUnavailable(RuntimeError):
    """Raised when DSR was requested but no compatible cache backend exists."""


def build_dsr_config(dsr_budget: int | None, *, sink_size: int = 4) -> dict[str, int] | None:
    """Build the standard DSR cache budget split used by EdgeStudio."""
    if not dsr_budget:
        return None
    usable = max(dsr_budget - sink_size, 0)
    return {
        "max_size": dsr_budget,
        "heavy_budget": usable // 2,
        "recent_budget": usable - usable // 2,
        "sink_size": sink_size,
    }


def make_prompt_cache(model: Any, dsr_config: dict[str, int] | None = None) -> list[Any]:
    """Create a prompt cache, delegating DSR to edgestudio_core when enabled."""
    from mlx_lm.models.cache import make_prompt_cache as mlx_make_prompt_cache

    if dsr_config is None:
        return mlx_make_prompt_cache(model)

    try:
        from edgestudio_core.cache import make_dsr_prompt_cache
    except ImportError as exc:
        raise DSRCacheUnavailable(
            "DSR cache requires the bundled edgestudio_core runtime. "
            "Reinstall edgestudio or disable DSR for this run."
        ) from exc

    return make_dsr_prompt_cache(model, dsr_config=dsr_config)
