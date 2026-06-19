# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model merge schemas."""

from __future__ import annotations

from pydantic import BaseModel


class MergeRequest(BaseModel):
    model_dirs: list[str]  # 2+ model directories to merge
    strategy: str = "linear"  # "linear" | "slerp" | "ties" | "task_arithmetic"
    weights: list[float] = []  # Per-model weights (empty = equal)
    base_model_dir: str = ""  # Base model for task_arithmetic
    density: float = 0.5  # For TIES: sparsification density
    output_dir: str = ""  # empty = auto
