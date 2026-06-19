# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Knowledge distillation schemas."""

from __future__ import annotations

from pydantic import BaseModel


class DistillRequest(BaseModel):
    teacher_dir: str
    student_dir: str
    dataset_path: str  # jsonl or parquet
    mode: str = "offline"  # "offline" | "taid"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    temperature: float = 2.0
    alpha: float = 0.5  # KL loss weight (vs CE loss)
    max_samples: int = 0  # 0 = use all
    output_dir: str = ""  # empty = auto


class DistillStatusResponse(BaseModel):
    epoch: int
    total_epochs: int
    step: int
    total_steps: int
    loss: float
    kl_loss: float
    ce_loss: float
    learning_rate: float
    tokens_per_second: float
