# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Tests for generated EdgeKit demo package metadata."""

from backend.core import edge_runtime_codegen


def test_edge_runtime_demo_defaults_to_public_edgekit_package() -> None:
    package_swift = edge_runtime_codegen._make_package_swift("")

    assert (
        '.package(url: "https://github.com/AtomGradient/edge-kit.git", '
        'exact: "1.0.0-rc98")'
    ) in package_swift
    assert "/Users/alex" not in package_swift
    assert ".package(path:" not in package_swift


def test_edge_runtime_demo_allows_explicit_local_edgekit_path() -> None:
    package_swift = edge_runtime_codegen._make_package_swift("../edge-kit")

    assert '.package(path: "../edge-kit")' in package_swift
