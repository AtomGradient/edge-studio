# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Device profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.analysis import DeviceProfileSchema

router = APIRouter(prefix="/api", tags=["devices"])


@router.get("/devices", response_model=list[DeviceProfileSchema])
def list_devices() -> list[DeviceProfileSchema]:
    from backend.core.device_profiles import all_devices

    return [
        DeviceProfileSchema(
            name=d.name,
            category=d.category,
            ram_gb=d.ram_gb,
            available_ram_gb=d.available_ram_gb,
            neural_engine_tops=d.neural_engine_tops,
            gpu_cores=d.gpu_cores,
            chip=d.chip,
            max_model_size_gb=d.max_model_size_gb,
        )
        for d in all_devices()
    ]
