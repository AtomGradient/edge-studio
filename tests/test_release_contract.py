# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Release contract checks for the pip package metadata."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _dependency_names() -> set[str]:
    project = _project_metadata()["project"]
    names = set()
    for dependency in project["dependencies"]:
        name = re.split(r"\[|<|>|=|~|!", dependency, maxsplit=1)[0]
        names.add(name.lower())
    return names


def test_console_scripts_are_declared_for_pip_install() -> None:
    scripts = _project_metadata()["project"]["scripts"]

    assert scripts == {"edge": "backend.cli.main:main"}


def test_distribution_name_matches_public_install_command() -> None:
    project = _project_metadata()["project"]

    assert project["name"] == "edge-studio"


def test_runtime_imports_have_explicit_dependencies() -> None:
    dependencies = _dependency_names()

    assert "pyyaml" in dependencies
    assert "platformdirs" in dependencies


def test_known_removed_runtime_dependencies_stay_removed() -> None:
    dependencies = _dependency_names()

    assert "librosa" not in dependencies
    assert "websockets" not in dependencies
