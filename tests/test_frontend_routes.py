# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Static checks for frontend route wiring."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_TSX = ROOT / "frontend/src/App.tsx"


def test_lazy_page_imports_resolve_to_existing_files() -> None:
    source = APP_TSX.read_text(encoding="utf-8")
    imports = sorted(set(re.findall(r"import\('@/(pages/[^']+)'\)", source)))

    assert imports
    missing = []
    for import_path in imports:
        candidate = ROOT / "frontend/src" / f"{import_path}.tsx"
        index_candidate = ROOT / "frontend/src" / import_path / "index.tsx"
        if not candidate.exists() and not index_candidate.exists():
            missing.append(import_path)

    assert not missing


def test_primary_routes_are_registered() -> None:
    source = APP_TSX.read_text(encoding="utf-8")
    route_paths = set(re.findall(r'path="([^"]+)"', source))

    expected = {
        "/",
        "/simple",
        "/simple/setup",
        "/dashboard",
        "/architecture",
        "/weights",
        "/chat",
        "/duplex",
        "/quality",
        "/kv-cache",
        "/neural-imprint",
        "/a-library",
        "/optimization",
        "/auto-optimizer",
        "/export",
        "/devices",
        "/joint-inference",
    }
    assert expected <= route_paths
