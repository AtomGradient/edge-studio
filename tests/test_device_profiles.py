# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Regression checks for device defaults used by API and frontend flows."""

from __future__ import annotations

import ast
from pathlib import Path

from backend.core.device_profiles import get_device
from backend.schemas.analysis import KVReportRequest


ROOT = Path(__file__).resolve().parents[1]


def test_kv_report_schema_defaults_resolve_to_known_profiles() -> None:
    request = KVReportRequest(model_id="test-model")

    assert request.devices
    for name in request.devices:
        assert get_device(name) is not None, name


def test_frontend_kv_cache_default_devices_resolve_to_known_profiles() -> None:
    source = (ROOT / "frontend/src/pages/KVCacheAnalysis.tsx").read_text(encoding="utf-8")
    marker = "const DEFAULT_DEVICES = "
    start = source.index(marker) + len(marker)
    end = source.index("];", start) + 1
    default_devices = ast.literal_eval(source[start:end])

    assert default_devices == [
        "iPhone 17 Pro",
        "iPad Pro M5 (16GB)",
        "MacBook Air M5 (16GB)",
        "MacBook Pro M5 Max (48GB)",
        "Mac Studio M3 Ultra (256GB)",
    ]
    for name in default_devices:
        assert get_device(name) is not None, name
