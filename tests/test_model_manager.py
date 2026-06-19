# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Unit tests for model manager lifecycle behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from backend.core.architecture import ArchNode, ModelArchitecture
from backend.core.weight_loader import WeightIndex
from backend.services.model_manager import ModelManager


def test_model_manager_caches_loads_and_unload_clears_associated_state(monkeypatch, tmp_path: Path) -> None:
    manager = ModelManager()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    architecture = ModelArchitecture(
        model_type="fake",
        model_name="Fake",
        model_dir=str(model_dir),
        root=ArchNode(name="Fake", node_type="model"),
        config={"model_type": "fake"},
    )
    weights = WeightIndex(model_dir=str(model_dir))
    load_calls: list[str] = []
    cleared: list[str] = []

    def fake_load_model(path: str):
        load_calls.append(path)
        return architecture, weights, ["trace"], SimpleNamespace(value="llm")

    monkeypatch.setattr("backend.core.model_registry.load_model", fake_load_model)
    monkeypatch.setitem(
        sys.modules,
        "backend.api.chat_loaders",
        SimpleNamespace(clear_model_cache=lambda path: cleared.append(path)),
    )

    first = manager.load_model(str(model_dir))
    second = manager.load_model(str(model_dir))

    assert first is second
    assert load_calls == [str(model_dir)]
    assert first.config == {"model_type": "fake"}
    assert first.category == "llm"

    manager.store_profile(first.model_id, "profile")
    manager.store_trace(first.model_id, "trace")
    manager.store_quality(first.model_id, "ppl", {"value": 12.3})
    manager.store_pipeline(first.model_id, {"ok": True})
    assert manager.get_session_summary(first.model_id) == {
        "has_trace": True,
        "has_profile": True,
        "has_ppl": True,
        "has_report": False,
        "has_generation": False,
        "has_pipeline": True,
    }

    manager.unload_model(first.model_id)

    assert cleared == [str(model_dir)]
    assert manager.get_model(first.model_id) is None
    assert manager.get_profile(first.model_id) is None
    assert manager.get_trace(first.model_id) is None
    assert manager.get_all_quality(first.model_id) == {}
    assert manager.get_pipeline(first.model_id) is None


def test_model_manager_canonicalizes_model_dir_for_stable_ids(tmp_path: Path) -> None:
    manager = ModelManager()
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert manager._make_id(str(model_dir)) == manager._make_id(str(model_dir))
    assert manager._canonical_model_dir(str(model_dir / ".." / "model")) == str(model_dir)
