# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for intent search lookup, semantic scoring, and tag fallback."""

from __future__ import annotations

import numpy as np

from backend.core import intent_search
from backend.core.device_profiles import DeviceProfile
from backend.core.recommendation_engine import ModelRecommendation
from backend.core import recommendation_engine


def _device() -> DeviceProfile:
    return DeviceProfile(
        name="Test Mac",
        category="mac",
        ram_gb=16,
        available_ram_gb=8,
        neural_engine_tops=0,
        gpu_cores=10,
        chip="M-Test",
    )


def test_model_lookup_search_matches_local_catalog_without_remote(monkeypatch) -> None:
    monkeypatch.setattr(
        recommendation_engine,
        "_load_catalog",
        lambda: [
            {
                "name": "Qwen3.6-35B-A3B-8bit",
                "id": "qwen36",
                "family": "qwen",
                "download_hint": "mlx-community/Qwen3.6-35B-A3B-8bit",
                "category": "llm",
                "mlx": True,
                "size_gb": 6.0,
                "quality_tier": "premium",
                "params_b": 35,
            }
        ],
    )
    monkeypatch.setattr(intent_search, "_resolve_search_device", lambda _query, _device_name: (_device(), None))

    result = intent_search.model_lookup_search("try Qwen3.6 on this Mac", max_results=3)

    assert result is not None
    assert result["results"][0]["name"] == "Qwen3.6-35B-A3B-8bit"
    assert result["results"][0]["fits_device"] is True


def test_intent_search_semantic_path_scores_and_sorts_without_real_embeddings(monkeypatch) -> None:
    catalog = [
        {
            "name": "Chat Small",
            "description": "small chat",
            "category": "llm",
            "mlx": True,
            "size_gb": 1.0,
            "quality_tier": "entry",
            "download_hint": "mlx/chat-small",
            "params_b": 1,
            "strengths": ["chat"],
        },
        {
            "name": "Vision Pro",
            "description": "image understanding",
            "category": "vlm",
            "mlx": True,
            "size_gb": 3.0,
            "quality_tier": "premium",
            "download_hint": "mlx/vision-pro",
            "params_b": 7,
            "strengths": ["multimodal"],
        },
    ]

    class FakeEncoder:
        def encode(self, texts, **_kwargs):
            if isinstance(texts, list) and len(texts) == 1:
                return np.array([[0.0, 1.0]], dtype=np.float32)
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    monkeypatch.setattr(intent_search, "is_embedding_ready", lambda: {"ready": True, "model_dir": "/fake/embed"})
    monkeypatch.setattr(intent_search, "_get_encoder", lambda _model_dir: FakeEncoder())
    monkeypatch.setattr(intent_search, "compute_catalog_vectors", lambda _catalog, _model_dir: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    monkeypatch.setattr(recommendation_engine, "_load_catalog", lambda: catalog)
    monkeypatch.setattr(intent_search, "_detect_device_from_query", lambda _query: None)
    monkeypatch.setattr(intent_search, "_resolve_search_device", lambda _query, _device_name: (_device(), None))
    monkeypatch.setattr("backend.core.device_profiles.get_device", lambda _name: _device())

    result = intent_search.intent_search("image understanding model", device_name="Test Mac", max_results=2)

    assert result["results"][0]["name"] == "Vision Pro"
    assert result["results"][0]["semantic_score"] == 1.0
    assert result["results"][1]["name"] == "Chat Small"


def test_tag_based_fallback_expands_voice_and_deduplicates(monkeypatch) -> None:
    calls: list[str] = []

    def fake_recommend_models(_device, *, use_case: str, max_results: int, tts_variant: str = ""):
        calls.append(use_case)
        if use_case == "asr":
            return [
                ModelRecommendation(
                    name="Shared Voice",
                    description="",
                    estimated_size_gb=1.0,
                    fits_device=True,
                    headroom_gb=1.0,
                    quality_tier="high",
                    download_hint="mlx/shared",
                    category="asr",
                )
            ]
        return [
            ModelRecommendation(
                name="Shared Voice",
                description="",
                estimated_size_gb=1.0,
                fits_device=True,
                headroom_gb=1.0,
                quality_tier="high",
                download_hint="mlx/shared",
                category="tts",
            ),
            ModelRecommendation(
                name="TTS Only",
                description="",
                estimated_size_gb=0.5,
                fits_device=True,
                headroom_gb=1.5,
                quality_tier="balanced",
                download_hint="mlx/tts",
                category="tts",
            ),
        ]

    monkeypatch.setattr(intent_search, "model_lookup_search", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(intent_search, "_resolve_search_device", lambda _query, _device_name: (_device(), None))
    monkeypatch.setattr(recommendation_engine, "recommend_models", fake_recommend_models)

    results = intent_search.tag_based_fallback("voice", max_results=5)

    assert calls == ["asr", "tts"]
    assert [item["download_hint"] for item in results] == ["mlx/shared", "mlx/tts"]
