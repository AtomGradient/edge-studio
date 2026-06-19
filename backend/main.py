# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""FastAPI application entry point."""

from __future__ import annotations

import errno
import logging
import os
import threading
from contextlib import asynccontextmanager

# ── Disable tqdm's TMonitor daemon thread BEFORE any library imports it ──
# tqdm creates a daemon monitoring thread that can cause GIL crashes when
# combined with MLX/torch Metal operations. mlx-audio imports tqdm during
# model loading, so we must disable the monitor before that happens.
try:
    import tqdm
    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.router import api_router
from backend.config import ALLOWED_ORIGINS
from backend.resources.paths import frontend_dist_candidates
from backend.services.error_mapper import map_error

# Lesson learned: app logger defaults to WARNING, INFO messages are suppressed.
# uvicorn --log-level info only controls its own logs, not app loggers. Add basicConfig
# so INFO from event_upload / classify_request / pair all reach stdout.
# When LOG_LEVEL=debug, switch to DEBUG (includes mesh op routing internal debug logs).
_log_level = os.environ.get("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def _configure_mlx_metal_limits() -> None:
    """Apply optional public-MLX Metal limits before MLX initialises."""
    try:
        from edgestudio_core.metal_limits import configure_preimport
    except ImportError:
        return

    max_ops = os.environ.get("EDGESTUDIO_MLX_MAX_OPS_PER_BUFFER")
    max_mb = os.environ.get("EDGESTUDIO_MLX_MAX_MB_PER_BUFFER")
    if not max_ops and not max_mb:
        return
    configure_preimport(
        max_ops_per_buffer=int(max_ops) if max_ops else None,
        max_mb_per_buffer=int(max_mb) if max_mb else None,
    )
    logger.info(
        "Configured MLX Metal limits: max_ops=%s max_mb=%s",
        os.environ.get("MLX_MAX_OPS_PER_BUFFER"),
        os.environ.get("MLX_MAX_MB_PER_BUFFER"),
    )


_configure_mlx_metal_limits()


def _auto_download_embedding():
    """Download embedding model in background at startup."""
    if os.environ.get("EDGE_STUDIO_AUTO_DOWNLOAD_EMBEDDING", "0") != "1":
        logger.info(
            "Embedding auto-download disabled; set EDGE_STUDIO_AUTO_DOWNLOAD_EMBEDDING=1 to enable"
        )
        return
    try:
        from backend.core.intent_search import is_embedding_ready, download_embedding_model, detect_region
        status = is_embedding_ready()
        if status["ready"]:
            return
        region = detect_region()
        logger.info("Auto-downloading embedding model (region=%s)...", region)
        download_embedding_model(region=region)
        logger.info("Embedding model ready")
    except Exception as e:
        logger.info("Embedding auto-download deferred: %s", e)


def _auto_start_edgemesh():
    host = "127.0.0.1"
    port = 18842
    server = None
    try:
        from backend.config import HOST, PORT
        from backend.services import event_ingest
        from backend.services.certificate_manager import load_or_create
        from backend.services.mesh_discovery import get_default_broadcaster
        from backend.services.mesh_transport import get_default_server

        host = HOST
        port = PORT
        identity = load_or_create()
        server = get_default_server()
        # Register event_upload handler before accept loop starts — any connection
        # that arrives mid-race still gets the handler table fully populated.
        event_ingest.register(server)
        from backend.services import rpp_artifact_ingest
        rpp_artifact_ingest.register(server)
        from backend.services import persona_source_ingest
        persona_source_ingest.register(server)
        from backend.services import persona_rpp_input_ingest
        persona_rpp_input_ingest.register(server)
        from backend.services import device_snapshot_ingest
        device_snapshot_ingest.register(server)
        from backend.services import halo_capsule_apply_status_ingest
        halo_capsule_apply_status_ingest.register(server)
        from backend.services import halo_capsule_transfer_ack_ingest
        halo_capsule_transfer_ack_ingest.register(server)
        from backend.services import peer_trust_ingest
        peer_trust_ingest.register(server)
        # classify_request — iPhone daemon LLM inference offloaded to Mac
        # (~67s/item on iPhone Air → ~1-3s/item on M2 Ultra). iPhone uses this path only when mesh is reachable,
        # falls back to local LLMManager when unreachable. ClassifyService lazy-loads, model loaded on first request.
        from backend.services import classify_ingest
        classify_ingest.register(server)
        from backend.services import joint_inference_ingest
        joint_inference_ingest.register(server)
        server.start()
        broadcaster = get_default_broadcaster(
            peer_id=identity.peer_id,
            display_name=identity.display_name,
            cert_fingerprint=identity.fingerprint,
            http_port=PORT,
            mesh_port=server.port,
        )
        broadcaster.start()
        logger.info(
            "EdgeMesh auto-started: mTLS :%d / HTTP :%d / fingerprint=%s",
            server.port, PORT, identity.fingerprint,
        )
    except Exception as e:  # noqa: BLE001
        if isinstance(e, OSError) and e.errno == errno.EADDRINUSE:
            ui_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
            logger.warning(
                "EdgeMesh is already running on port %d; Studio UI is still available at http://%s:%d",
                getattr(server, "port", 18843) if server is not None else 18843,
                ui_host,
                port,
            )
            return
        logger.warning("EdgeMesh auto-start failed: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: auto-download embedding model silently in background
    threading.Thread(target=_auto_download_embedding, daemon=True).start()
    # Startup: EdgeMesh mTLS + Bonjour (P0)
    threading.Thread(target=_auto_start_edgemesh, daemon=True).start()
    try:
        yield
    finally:
        # Shutdown: stop EdgeMesh
        try:
            from backend.services.mesh_discovery import _default_broadcaster
            if _default_broadcaster is not None:
                _default_broadcaster.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            from backend.services.mesh_transport import _default_server
            if _default_server is not None:
                _default_server.stop()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(
    title="Edge Studio",
    version="2.0.0",
    description="API backend for Edge Studio — optimize, benchmark, and deploy LLMs on Apple Silicon edge devices",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Serve bundled frontend SPA (pip-installed mode) ──────────────────────
# When the frontend dist directory exists (built via `npm run build` or
# included in the pip package), serve it at "/" with a catch-all fallback
# to index.html for client-side routing. In dev mode the Vite dev server
# handles the frontend, so this block is a no-op.
_frontend_dist_candidates = frontend_dist_candidates()
_frontend_dist = next(
    (candidate for candidate in _frontend_dist_candidates if candidate.is_dir()),
    _frontend_dist_candidates[0],
)
if _frontend_dist.is_dir():
    from starlette.responses import FileResponse
    from starlette.routing import Match
    from starlette.staticfiles import StaticFiles

    # Serve /assets/*, /favicon.svg etc. directly
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="static-assets")

    @app.get("/{full_path:path}")
    async def _spa_fallback(full_path: str, request: Request):
        """SPA fallback — return index.html for non-API routes."""
        if full_path.startswith(("api/", "ws/")):
            scope = {"type": "http", "path": f"/{full_path}", "method": request.method}
            for route in api_router.routes:
                match, _ = route.matches(scope)
                if match == Match.PARTIAL:
                    raise HTTPException(status_code=405, detail="Method Not Allowed")
                if match == Match.FULL:
                    raise HTTPException(status_code=404, detail="Not Found")
            raise HTTPException(status_code=404, detail="Not Found")
        target = _frontend_dist / full_path
        if target.is_file() and ".." not in full_path:
            return FileResponse(target)
        return FileResponse(_frontend_dist / "index.html")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    user_msg, debug_detail = map_error(exc)
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, debug_detail, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": user_msg})


def main():
    """Run the local Studio server used by ``edge studio``."""
    import uvicorn

    from backend.config import HOST, PORT

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    import uvicorn

    from backend.config import HOST, PORT

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
