# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""FastAPI smoke tests for endpoints that should work without a loaded model."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app, raise_server_exceptions=False)


def test_health_and_public_endpoints_are_reachable() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    checks = [
        ("/api/devices", list),
        ("/api/fs/home", dict),
        ("/api/model/loaded", list),
        ("/api/recommend/use-cases", dict),
        ("/api/recommend/catalog-status", dict),
    ]
    for path, expected_type in checks:
        response = client.get(path)
        assert response.status_code == 200, path
        assert isinstance(response.json(), expected_type), path


def test_recommend_models_accepts_current_default_device() -> None:
    response = client.post(
        "/api/recommend/models",
        json={
            "device_name": "MacBook Air M5 (16GB)",
            "use_case": "chat",
            "max_results": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert 1 <= len(payload) <= 3
    assert {"name", "estimated_size_gb", "fits_device"} <= set(payload[0])


def test_api_unknown_route_returns_json_404_not_spa_index() -> None:
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
