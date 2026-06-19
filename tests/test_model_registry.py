# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for model registry discovery and load dispatch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.core.architecture import ArchNode, ModelArchitecture
from backend.core.weight_loader import WeightIndex
from backend.core import model_registry


def test_discover_local_models_requires_config_and_weights(tmp_path: Path) -> None:
    ready = tmp_path / "ready-model"
    ready.mkdir()
    (ready / "config.json").write_text("{}", encoding="utf-8")
    (ready / "model.safetensors").write_bytes(b"fake")

    nested = tmp_path / "org" / "nested-model"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")
    (nested / "model.gguf").write_bytes(b"fake")

    missing_weights = tmp_path / "config-only"
    missing_weights.mkdir()
    (missing_weights / "config.json").write_text("{}", encoding="utf-8")

    discovered = model_registry.discover_local_models([str(tmp_path)])

    assert discovered == {
        "[Local] nested-model": str(nested),
        "[Local] ready-model": str(ready),
    }


def test_detect_model_type_uses_exact_registry_key_only() -> None:
    assert model_registry.detect_model_type({"model_type": "qwen3"}) == "qwen3"
    assert model_registry.detect_model_type({"model_type": "qwen3_5"}) == "generic"
    assert model_registry.detect_model_type({"model_type": "unknown"}) == "generic"


def test_load_model_dispatches_to_registered_parser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    weight_index = WeightIndex(model_dir=str(tmp_path))
    architecture = ModelArchitecture(
        model_type="qwen3",
        model_name="FakeQwen",
        model_dir=str(tmp_path),
        root=ArchNode(name="FakeQwen", node_type="model"),
        config={"model_type": "qwen3"},
    )
    parser_calls: list[tuple[str, dict, WeightIndex]] = []

    class FakeParser:
        def __init__(self, model_dir: str, config: dict, weights: WeightIndex) -> None:
            parser_calls.append((model_dir, config, weights))

        def parse(self) -> ModelArchitecture:
            return architecture

    monkeypatch.setattr(model_registry, "find_gguf_file", lambda _path: None)
    monkeypatch.setattr(model_registry, "load_config", lambda _path: {"model_type": "qwen3"})
    monkeypatch.setattr(model_registry, "load_weight_index", lambda _path: weight_index)
    monkeypatch.setattr(model_registry, "detect_pruning", lambda _config: ["trace"])
    monkeypatch.setattr(model_registry, "detect_model_category", lambda _config: SimpleNamespace(value="llm"))
    monkeypatch.setitem(model_registry.PARSER_REGISTRY, "qwen3", FakeParser)

    loaded_arch, loaded_weights, pruning, category = model_registry.load_model(str(tmp_path))

    assert loaded_arch is architecture
    assert loaded_weights is weight_index
    assert pruning == ["trace"]
    assert category.value == "llm"
    assert parser_calls == [(str(tmp_path), {"model_type": "qwen3"}, weight_index)]
