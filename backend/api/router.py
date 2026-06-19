# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Central router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from . import (
    a_library,
    activation,
    architecture,
    attention,
    auto_optimizer,
    auto_tune,
    benchmark_api,
    chat,
    comparison,
    conversations,
    devices,
    distill,
    export,
    filesystem,
    huggingface,
    inference,
    intent,
    kv_cache,
    merge,
    mesh,
    model,
    moe,
    optimization,
    neural_imprint,
    personal,
    pipeline,
    pruning,
    quality,
    recommend,
    simple,
    system_info,
    terminal,
    weights,
    ws,
)

api_router = APIRouter()

# REST routes
api_router.include_router(filesystem.router)
api_router.include_router(model.router)
api_router.include_router(architecture.router)
api_router.include_router(weights.router)
api_router.include_router(activation.router)
api_router.include_router(pruning.router)
api_router.include_router(inference.router)
api_router.include_router(quality.router)
api_router.include_router(attention.router)
api_router.include_router(optimization.router)
api_router.include_router(pipeline.router)
api_router.include_router(auto_optimizer.router)
api_router.include_router(kv_cache.router)
api_router.include_router(moe.router)
api_router.include_router(comparison.router)
api_router.include_router(devices.router)
api_router.include_router(system_info.router)
api_router.include_router(recommend.router)
api_router.include_router(intent.router)
api_router.include_router(export.router)
api_router.include_router(huggingface.router)
api_router.include_router(benchmark_api.router)
api_router.include_router(distill.router)
api_router.include_router(merge.router)
api_router.include_router(auto_tune.router)
api_router.include_router(simple.router)
api_router.include_router(a_library.router)
api_router.include_router(neural_imprint.router)
api_router.include_router(personal.router)
api_router.include_router(conversations.router)
api_router.include_router(mesh.router)

# WebSocket
api_router.include_router(ws.router)
api_router.include_router(chat.router)
api_router.include_router(terminal.router)
api_router.include_router(personal.ws_router)
