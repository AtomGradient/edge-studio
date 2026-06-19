# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Auto-tune schemas."""

from __future__ import annotations

from pydantic import BaseModel


class AutoTuneRequest(BaseModel):
    model_dir: str
    device_name: str = ""
    max_tokens: int = 50
    num_runs: int = 3
    search_temperatures: list[float] = [0.0, 0.7]
    search_kv_cache_sizes: list[int] = [512, 1024, 2048, 4096]
    force_rerun: bool = False
