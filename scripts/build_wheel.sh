#!/usr/bin/env bash
# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIST="$ROOT_DIR/frontend/dist"
PACKAGED_FRONTEND="$ROOT_DIR/backend/resources/frontend/dist"
BUILD_DIR="$ROOT_DIR/build"
DIST_DIR="$ROOT_DIR/dist"
EGG_INFO_DIRS=("$ROOT_DIR/edgestudio.egg-info" "$ROOT_DIR/edge_studio.egg-info")
PYTHON_BIN="${PYTHON:-python3}"

cd "$ROOT_DIR"

rm -rf "$BUILD_DIR" "$DIST_DIR" "${EGG_INFO_DIRS[@]}"

npm --prefix frontend ci
npm --prefix frontend run build

rm -rf "$PACKAGED_FRONTEND"
mkdir -p "$(dirname "$PACKAGED_FRONTEND")"
cp -R "$FRONTEND_DIST" "$PACKAGED_FRONTEND"

"$PYTHON_BIN" -m build --wheel
