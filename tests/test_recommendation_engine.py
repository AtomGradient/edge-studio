# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for device-aware model recommendation logic."""

from __future__ import annotations

from backend.core.device_profiles import DeviceProfile
from backend.core import recommendation_engine as rec


def _device(max_available_gb: float = 4.0) -> DeviceProfile:
    return DeviceProfile(
        name="Test Device",
        category="mac",
        ram_gb=max_available_gb + 2,
        available_ram_gb=max_available_gb,
        neural_engine_tops=0,
        gpu_cores=8,
        chip="M-Test",
    )


def test_recommend_models_filters_mlx_category_and_ranks_fitting_models(monkeypatch) -> None:
    monkeypatch.setattr(
        rec,
        "_load_catalog",
        lambda: [
            {
                "name": "Tiny Chat",
                "category": "llm",
                "mlx": True,
                "size_gb": 1.0,
                "strengths": ["chat"],
                "quality_tier": "balanced",
                "download_hint": "mlx/tiny",
                "params_b": 1,
            },
            {
                "name": "Premium Chat",
                "category": "llm",
                "mlx": True,
                "size_gb": 3.0,
                "strengths": ["chat"],
                "quality_tier": "premium",
                "download_hint": "mlx/premium",
                "params_b": 7,
            },
            {
                "name": "Torch Only",
                "category": "llm",
                "mlx": False,
                "size_gb": 1.0,
                "strengths": ["chat"],
                "quality_tier": "premium",
                "download_hint": "torch/skip",
            },
            {
                "name": "Voice Model",
                "category": "tts",
                "mlx": True,
                "size_gb": 0.5,
                "strengths": ["tts"],
                "quality_tier": "premium",
                "download_hint": "mlx/voice",
            },
        ],
    )

    results = rec.recommend_models(_device(max_available_gb=4.0), use_case="chat", max_results=5)

    assert [item.name for item in results] == ["Premium Chat", "Tiny Chat"]
    assert all(item.category == "llm" for item in results)
    assert all(item.fits_device for item in results)


def test_recommend_models_unknown_use_case_defaults_to_llm_and_category_filter_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        rec,
        "_load_catalog",
        lambda: [
            {
                "name": "Fallback LLM",
                "category": "llm",
                "mlx": True,
                "size_gb": 1.0,
                "strengths": [],
                "quality_tier": "entry",
                "download_hint": "mlx/fallback",
            },
            {
                "name": "ASR Small",
                "category": "asr",
                "mlx": True,
                "size_gb": 0.4,
                "strengths": ["asr"],
                "quality_tier": "high",
                "download_hint": "mlx/asr",
            },
        ],
    )

    unknown = rec.recommend_models(_device(), use_case="unknown")
    filtered = rec.recommend_models(_device(), use_case="unknown", category_filter="asr")

    assert [item.name for item in unknown] == ["Fallback LLM"]
    assert [item.name for item in filtered] == ["ASR Small"]


def test_recommend_optimization_handles_fit_and_already_quantized_large_model() -> None:
    fitting = rec.recommend_optimization(1.0, _device(max_available_gb=4.0), current_bits=4)
    large_quantized = rec.recommend_optimization(10.0, _device(max_available_gb=1.0), current_bits=4)

    assert fitting.strategy_name == "No optimization needed"
    assert fitting.fits_device is True
    assert large_quantized.strategy_name == "Multi-step optimization"
    assert large_quantized.fits_device is False
    assert any("Prune" in step for step in large_quantized.steps)
    assert any("Remove" in step for step in large_quantized.steps)
    assert large_quantized.risk_level == "high"
