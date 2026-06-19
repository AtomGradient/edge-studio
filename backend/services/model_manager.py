# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Model lifecycle management — load, cache, unload models and associated data."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadedModel:
    model_id: str
    model_dir: str
    architecture: Any  # ModelArchitecture
    weight_index: Any  # WeightIndex
    pruning_traces: list[Any]  # list[PruningTrace]
    config: dict[str, Any] = field(default_factory=dict)
    category: str = "llm"  # "llm" | "vlm" | "tts"


class ModelManager:
    """Thread-safe manager for loaded models, profiles, and traces."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, LoadedModel] = {}
        self._profiles: dict[str, Any] = {}  # model_id → ActivationProfile
        self._traces: dict[str, Any] = {}  # model_id → trace data dict
        self._quality: dict[str, dict[str, Any]] = {}  # model_id → {ppl, report, generation}
        self._pipelines: dict[str, Any] = {}  # model_id → pipeline result
        self._loading: dict[str, threading.Event] = {}

    @staticmethod
    def _make_id(model_dir: str) -> str:
        return hashlib.sha256(model_dir.encode()).hexdigest()[:12]

    @staticmethod
    def _canonical_model_dir(model_dir: str) -> str:
        return os.path.abspath(os.path.expanduser(model_dir))

    # ---- model lifecycle ----

    def load_model(self, model_dir: str) -> LoadedModel:
        from backend.core.model_registry import load_model

        model_dir = self._canonical_model_dir(model_dir)
        model_id = self._make_id(model_dir)

        while True:
            with self._lock:
                if model_id in self._models:
                    return self._models[model_id]
                loading = self._loading.get(model_id)
                if loading is None:
                    loading = threading.Event()
                    self._loading[model_id] = loading
                    break
            loading.wait()

        try:
            architecture, weight_index, pruning_traces, category = load_model(model_dir)

            loaded = LoadedModel(
                model_id=model_id,
                model_dir=model_dir,
                architecture=architecture,
                weight_index=weight_index,
                pruning_traces=pruning_traces,
                config=architecture.config,
                category=category.value,
            )

            with self._lock:
                existing = self._models.get(model_id)
                if existing is not None:
                    return existing
                self._models[model_id] = loaded

            return loaded
        finally:
            with self._lock:
                done = self._loading.pop(model_id, None)
                if done is not None:
                    done.set()

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            loaded = self._models.pop(model_id, None)
            self._profiles.pop(model_id, None)
            self._traces.pop(model_id, None)
            self._quality.pop(model_id, None)
            self._pipelines.pop(model_id, None)
        if loaded is not None:
            from backend.api.chat_loaders import clear_model_cache

            clear_model_cache(loaded.model_dir)

    def get_model(self, model_id: str) -> LoadedModel | None:
        with self._lock:
            return self._models.get(model_id)

    def list_models(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._models.values())

    # ---- activation profile ----

    def store_profile(self, model_id: str, profile: Any) -> None:
        with self._lock:
            self._profiles[model_id] = profile

    def get_profile(self, model_id: str) -> Any | None:
        with self._lock:
            return self._profiles.get(model_id)

    # ---- inference trace ----

    def store_trace(self, model_id: str, trace: Any) -> None:
        with self._lock:
            self._traces[model_id] = trace

    def get_trace(self, model_id: str) -> Any | None:
        with self._lock:
            return self._traces.get(model_id)

    # ---- quality results (ppl / generation / report) ----

    def store_quality(self, model_id: str, key: str, result: Any) -> None:
        with self._lock:
            self._quality.setdefault(model_id, {})[key] = result

    def get_quality(self, model_id: str, key: str) -> Any | None:
        with self._lock:
            return self._quality.get(model_id, {}).get(key)

    def get_all_quality(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._quality.get(model_id, {}))

    # ---- pipeline results ----

    def store_pipeline(self, model_id: str, result: Any) -> None:
        with self._lock:
            self._pipelines[model_id] = result

    def get_pipeline(self, model_id: str) -> Any | None:
        with self._lock:
            return self._pipelines.get(model_id)

    # ---- session summary (what cached data exists for a model) ----

    def get_session_summary(self, model_id: str) -> dict[str, bool]:
        with self._lock:
            return {
                "has_trace": model_id in self._traces,
                "has_profile": model_id in self._profiles,
                "has_ppl": "ppl" in self._quality.get(model_id, {}),
                "has_report": "report" in self._quality.get(model_id, {}),
                "has_generation": "generation" in self._quality.get(model_id, {}),
                "has_pipeline": model_id in self._pipelines,
            }


# Singleton instance
manager = ModelManager()
