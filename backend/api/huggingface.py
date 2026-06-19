# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""HuggingFace Hub model search and download endpoints."""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.resources.paths import script_path
from backend.schemas.common import CreateTaskResponse
from backend.services.task_manager import task_manager

router = APIRouter(prefix="/api/hf", tags=["huggingface"])
logger = logging.getLogger(__name__)

# Default download location
_DEFAULT_DOWNLOAD_DIR = str(Path.home() / "mlx-community")

# Valid mirror sources
MirrorSource = Literal["official", "hf-mirror", "modelscope"]


@router.get("/search", response_model=dict[str, Any])
def search_models(
    query: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=50),
):
    """Search HuggingFace Hub for models (prioritizes mlx-community)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise HTTPException(500, "huggingface_hub not installed. Run: pip install huggingface_hub")

    api = HfApi()
    results = []

    try:
        models = api.list_models(
            search=query,
            sort="downloads",
            limit=limit,
        )
        for m in models:
            results.append({
                "id": m.id,
                "author": m.id.split("/")[0] if "/" in m.id else None,
                "downloads": m.downloads,
                "likes": m.likes,
                "tags": m.tags[:10] if m.tags else [],
                "pipeline_tag": m.pipeline_tag,
                "last_modified": m.last_modified.isoformat() if m.last_modified else None,
            })
    except Exception as exc:
        logger.error("HF search failed: %s", exc)
        raise HTTPException(502, f"HuggingFace search failed: {exc}")

    return {"models": results}


def _dir_total_size(path: str) -> int:
    """Total size of all files in a directory tree (bytes)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


@router.get("/local", response_model=dict[str, Any])
def list_local_models() -> dict[str, Any]:
    """List locally discovered models (HF cache + common locations)."""
    try:
        from backend.core.model_registry import discover_local_models
        models = discover_local_models()
        result = []
        for name, path in models.items():
            size_bytes = _dir_total_size(path)
            result.append({
                "name": name,
                "path": path,
                "size_bytes": size_bytes,
            })
        return {"models": result}
    except (ImportError, OSError) as exc:
        logger.warning("Local model discovery failed: %s", exc)
        return {"models": []}


class DeleteLocalModelRequest(BaseModel):
    path: str


@router.delete("/local", response_model=dict[str, Any])
def delete_local_model(req: DeleteLocalModelRequest) -> dict[str, Any]:
    """Delete a locally downloaded model directory.

    Only allows deleting from known safe directories (mlx-community, HF cache).
    """
    target = os.path.realpath(os.path.expanduser(req.path))

    # Safety: only allow deleting from known model locations
    safe_roots = [
        os.path.realpath(os.path.expanduser("~/mlx-community")),
        os.path.realpath(os.path.expanduser("~/Documents/mlx-community")),
        os.path.realpath(os.path.expanduser("~/.cache/huggingface/hub")),
    ]
    is_safe = any(target.startswith(root + os.sep) for root in safe_roots)
    if not is_safe:
        raise HTTPException(403, f"Cannot delete outside safe model directories")

    if not os.path.isdir(target):
        raise HTTPException(404, "Directory not found")

    size_bytes = _dir_total_size(target)
    shutil.rmtree(target)
    logger.info("Deleted local model: %s (%s)", target, _format_size(size_bytes))
    return {"status": "deleted", "path": target, "freed_bytes": size_bytes}


@router.get("/check-path", response_model=dict[str, Any])
def check_local_path(path: str = Query(..., description="Path to check")) -> dict[str, Any]:
    """Check if a model directory exists and whether it is complete.

    Returns: exists, complete (config.json + weight files), size_bytes.
    """
    import glob as _glob
    expanded = os.path.realpath(os.path.expanduser(path))
    if not os.path.isdir(expanded):
        return {"exists": False, "complete": False, "size_bytes": 0}

    has_config = os.path.exists(os.path.join(expanded, "config.json"))
    has_weights = bool(
        _glob.glob(os.path.join(expanded, "*.safetensors"))
        or _glob.glob(os.path.join(expanded, "*.gguf"))
        or _glob.glob(os.path.join(expanded, "*.npz"))
    )
    size_bytes = _dir_total_size(expanded)

    return {
        "exists": True,
        "complete": has_config and has_weights,
        "has_config": has_config,
        "has_weights": has_weights,
        "size_bytes": size_bytes,
        "path": expanded,
    }


@router.get("/probe", response_model=dict[str, Any])
def probe_hf_network() -> dict[str, Any]:
    """Probe HuggingFace reachability. Returns latency and suggestion if unreachable."""
    import httpx

    try:
        start = time.monotonic()
        resp = httpx.head("https://huggingface.co", timeout=3.0, follow_redirects=True)
        latency_ms = round((time.monotonic() - start) * 1000)
        reachable = resp.status_code < 500
        result: dict = {"reachable": reachable, "latency_ms": latency_ms}
        if not reachable:
            result["suggestion"] = "hf-mirror"
        return result
    except Exception:
        return {"reachable": False, "latency_ms": -1, "suggestion": "hf-mirror"}


class DownloadRequest(BaseModel):
    repo_id: str
    download_dir: str | None = None
    mirror: MirrorSource = "official"


# File patterns for snapshot_download fallback (skip large bin files, etc.)
_ALLOW_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "*.tiktoken", "*.model"]

# Download scripts
_HFD_PATH = script_path("hfd.sh")  # aria2c multi-threaded (HuggingFace / hf-mirror)
_MSD_PATH = script_path("msd.sh")  # ModelScope CLI (more reliable in mainland China)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size_bytes(path: str) -> int:
    """Total bytes of model files under *path* (excluding temp/metadata)."""
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            # Skip .hfd metadata directory
            if ".hfd" in dirs:
                dirs.remove(".hfd")
            for f in files:
                # Skip aria2 control files and hidden files
                if f.endswith(".aria2") or f.startswith("."):
                    continue
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _get_repo_total_size(repo_id: str, use_mirror: bool) -> int:
    """Query HF API for total repo size (bytes). Counts ALL files (same as hfd.sh)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        if use_mirror:
            api.endpoint = "https://hf-mirror.com"
        total = 0
        for item in api.list_repo_tree(repo_id, recursive=True):
            if hasattr(item, "size") and item.size:
                total += item.size
        return total
    except Exception as exc:
        logger.warning("Could not query repo size for %s: %s", repo_id, exc)
        return 0


def _format_speed(bps: float) -> str:
    if bps < 1024:
        return f"{bps:.0f} B/s"
    elif bps < 1024 ** 2:
        return f"{bps / 1024:.1f} KB/s"
    else:
        return f"{bps / 1024 ** 2:.1f} MB/s"


def _format_size(b: int | float) -> str:
    if b < 1024 ** 2:
        return f"{b / 1024:.0f} KB"
    elif b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f} MB"
    else:
        return f"{b / 1024 ** 3:.2f} GB"


def _verify_model_files(local_dir: str) -> bool:
    """Check that local_dir contains actual model files, not just metadata."""
    if not os.path.isdir(local_dir):
        return False
    for f in os.listdir(local_dir):
        if f.endswith((".safetensors", ".json")) and not f.startswith("."):
            return True
    # Check one level down (some repos have subdirectories)
    for sub in os.listdir(local_dir):
        subdir = os.path.join(local_dir, sub)
        if os.path.isdir(subdir) and not sub.startswith("."):
            for f in os.listdir(subdir):
                if f.endswith((".safetensors", ".json")):
                    return True
    return False


def _run_hfd(
    repo_id: str,
    local_dir: str,
    use_mirror: bool,
    cancel_flag: threading.Event,
    stall_timeout: int = 120,  # seconds without progress before giving up
) -> None:
    """Run hfd.sh — same as terminal `./hfd.sh repo_id`.

    Raises RuntimeError on failure/stall, TaskCancelledError on cancel.
    """
    cmd = ["bash", str(_HFD_PATH), repo_id, "--local-dir", local_dir]

    run_env = os.environ.copy()
    # Strip proxy vars that break aria2c
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"):
        run_env.pop(k, None)
    if use_mirror:
        run_env["HF_ENDPOINT"] = "https://hf-mirror.com"

    logger.info("Running: %s (mirror=%s)", " ".join(cmd), use_mirror)
    proc = subprocess.Popen(cmd, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    last_size = 0
    last_progress_time = time.monotonic()

    try:
        while proc.poll() is None:
            if cancel_flag.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                from backend.services.error_mapper import TaskCancelledError
                raise TaskCancelledError("Download cancelled")

            # Check for stall
            current_size = _dir_size_bytes(local_dir) if os.path.isdir(local_dir) else 0
            now = time.monotonic()
            if current_size > last_size:
                last_size = current_size
                last_progress_time = now
            elif current_size > 0 and (now - last_progress_time) > stall_timeout:
                # Stalled for too long, kill and fallback
                logger.warning("hfd.sh stalled for %ds, terminating", stall_timeout)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError(f"Download stalled for {stall_timeout}s")

            time.sleep(2)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise

    if proc.returncode != 0:
        output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        raise RuntimeError(f"hfd.sh failed (exit {proc.returncode}): {output[-500:]}")


def _run_msd(
    repo_id: str,
    local_dir: str,
    cancel_flag: threading.Event,
    stall_timeout: int = 120,  # seconds without progress before giving up
) -> None:
    """Run msd.sh — ModelScope downloader (more stable for China mainland).

    Raises RuntimeError on failure/stall, TaskCancelledError on cancel.
    """
    cmd = ["bash", str(_MSD_PATH), repo_id, "--local-dir", local_dir]

    run_env = os.environ.copy()
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    last_size = 0
    last_progress_time = time.monotonic()

    try:
        while proc.poll() is None:
            if cancel_flag.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                from backend.services.error_mapper import TaskCancelledError
                raise TaskCancelledError("Download cancelled")

            # Check for stall
            current_size = _dir_size_bytes(local_dir) if os.path.isdir(local_dir) else 0
            now = time.monotonic()
            if current_size > last_size:
                last_size = current_size
                last_progress_time = now
            elif current_size > 0 and (now - last_progress_time) > stall_timeout:
                # Stalled for too long, kill and fallback
                logger.warning("msd.sh stalled for %ds, terminating", stall_timeout)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError(f"Download stalled for {stall_timeout}s")

            time.sleep(2)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise

    if proc.returncode != 0:
        output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        raise RuntimeError(f"msd.sh failed (exit {proc.returncode}): {output[-500:]}")


# ---------------------------------------------------------------------------
# Download endpoint
# ---------------------------------------------------------------------------

def _get_modelscope_total_size(repo_id: str) -> int:
    """Query ModelScope API for total repo size (bytes)."""
    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        # ModelScope API returns file list with sizes
        files = api.get_model_files(repo_id)
        total = sum(f.get("Size", 0) for f in files if isinstance(f, dict))
        return total
    except Exception as exc:
        logger.warning("Could not query ModelScope size for %s: %s", repo_id, exc)
        return 0


@router.post("/download", response_model=CreateTaskResponse)
def download_model(req: DownloadRequest):
    repo_id = req.repo_id
    mirror = req.mirror
    # Mainland China: prefer msd.sh for both hf-mirror and ModelScope
    use_china_source = mirror in ("hf-mirror", "modelscope")

    # Check tool availability
    hfd_available = shutil.which("aria2c") is not None and _HFD_PATH.exists()
    msd_available = _MSD_PATH.exists()

    if use_china_source and not msd_available and not hfd_available:
        raise HTTPException(500, "Neither msd.sh nor aria2c available.")
    else:
        if not hfd_available:
            try:
                from huggingface_hub import HfApi  # noqa: F401
            except ImportError:
                raise HTTPException(500, "Install aria2c (brew install aria2) or huggingface_hub.")

    task_id = task_manager.create_task()
    target_dir = req.download_dir or _DEFAULT_DOWNLOAD_DIR

    def _run(progress_callback=None):
        if progress_callback:
            progress_callback("Querying repo info...", 0.01)

        os.makedirs(target_dir, exist_ok=True)
        local_name = repo_id.replace("/", "_")
        local_dir = os.path.join(target_dir, local_name)

        # Query total size — use matching API
        if mirror == 'modelscope':
            total_bytes = _get_modelscope_total_size(repo_id)
            source_label = "via ModelScope"
        elif mirror == 'hf-mirror':
            total_bytes = _get_repo_total_size(repo_id, use_mirror=True)
            source_label = "via HF Mirror"
        else:
            total_bytes = _get_repo_total_size(repo_id, use_mirror=False)
            source_label = "via HuggingFace"

        if progress_callback:
            size_hint = f" ({_format_size(total_bytes)})" if total_bytes else ""
            progress_callback(f"Downloading {repo_id}{size_hint}...", 0.02)

        # --- Progress monitor (scans directory every 2s) ---
        download_done = threading.Event()
        cancel_flag = threading.Event()
        prev = [0, time.monotonic()]  # [bytes, timestamp]
        last_progress_time = [time.monotonic()]  # track last time we saw progress
        current_source_label = [source_label]  # mutable for fallback update

        def _monitor():
            while not download_done.wait(2.0):
                if not os.path.isdir(local_dir):
                    continue
                current = _dir_size_bytes(local_dir)
                now = time.monotonic()
                dt = now - prev[1]
                speed = (current - prev[0]) / dt if dt > 0.5 else 0

                # Track stall duration
                if current > prev[0]:
                    last_progress_time[0] = now
                stall_duration = now - last_progress_time[0]

                prev[0], prev[1] = current, now

                # Build message
                if total_bytes > 0:
                    pct = 0.02 + min(current / total_bytes, 1.0) * 0.93
                    msg = f"{_format_size(current)} / {_format_size(total_bytes)}"
                else:
                    pct = min(0.02 + current / (4 * 1024 ** 3), 0.90)
                    msg = f"{_format_size(current)} downloaded"

                if speed > 0:
                    msg += f" · {_format_speed(speed)}"
                elif stall_duration > 10:
                    # Stalled for more than 10 seconds
                    msg += f" · stalled {int(stall_duration)}s"
                elif current > 0 and total_bytes > 0 and current / total_bytes > 0.9:
                    msg += " · finalizing"

                msg += f" · {current_source_label[0]}"

                if progress_callback:
                    try:
                        progress_callback(msg, pct)
                    except Exception:
                        cancel_flag.set()
                        return

        monitor = threading.Thread(target=_monitor, daemon=True)
        monitor.start()

        try:
            if use_china_source:
                # Mainland China: msd.sh preferred, fallback to hfd.sh + hf-mirror
                if msd_available:
                    try:
                        _run_msd(repo_id, local_dir, cancel_flag)
                        if not _verify_model_files(local_dir):
                            raise RuntimeError("msd.sh completed but no model files found")
                        result_path = local_dir
                    except Exception as e:
                        if "cancelled" in str(e).lower():
                            raise
                        # Fallback to HF mirror
                        logger.warning("msd.sh failed: %s — falling back to hf-mirror", e)
                        current_source_label[0] = "via HF Mirror (fallback)"
                        if hfd_available:
                            _run_hfd(repo_id, local_dir, use_mirror=True, cancel_flag=cancel_flag)
                            result_path = local_dir
                        else:
                            from huggingface_hub import snapshot_download
                            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                            try:
                                result_path = snapshot_download(
                                    repo_id=repo_id, local_dir=local_dir,
                                    allow_patterns=_ALLOW_PATTERNS,
                                )
                            finally:
                                os.environ.pop("HF_ENDPOINT", None)
                elif hfd_available:
                    # msd.sh unavailable, use hfd.sh + hf-mirror directly
                    current_source_label[0] = "via HF Mirror"
                    _run_hfd(repo_id, local_dir, use_mirror=True, cancel_flag=cancel_flag)
                    result_path = local_dir
                else:
                    from huggingface_hub import snapshot_download
                    current_source_label[0] = "via HF Mirror"
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                    try:
                        result_path = snapshot_download(
                            repo_id=repo_id, local_dir=local_dir,
                            allow_patterns=_ALLOW_PATTERNS,
                        )
                    finally:
                        os.environ.pop("HF_ENDPOINT", None)
            elif hfd_available:
                # Outside mainland China: hfd.sh connects to HuggingFace directly
                try:
                    _run_hfd(repo_id, local_dir, use_mirror=False, cancel_flag=cancel_flag)
                    if not _verify_model_files(local_dir):
                        raise RuntimeError("hfd.sh completed but no model files found")
                    result_path = local_dir
                except Exception as e:
                    if "cancelled" in str(e).lower():
                        raise
                    logger.warning("hfd.sh failed: %s — falling back to snapshot_download", e)
                    current_source_label[0] = "via HuggingFace (fallback)"
                    from huggingface_hub import snapshot_download
                    result_path = snapshot_download(
                        repo_id=repo_id, local_dir=local_dir,
                        allow_patterns=_ALLOW_PATTERNS,
                    )
            else:
                # Outside mainland China: direct snapshot_download
                from huggingface_hub import snapshot_download
                result_path = snapshot_download(
                    repo_id=repo_id, local_dir=local_dir,
                    allow_patterns=_ALLOW_PATTERNS,
                )
        finally:
            download_done.set()
            monitor.join(timeout=5)

        if progress_callback:
            progress_callback("Download complete", 1.0)

        return {"path": result_path, "repo_id": repo_id}

    task_manager.run_in_thread(task_id, _run)
    return CreateTaskResponse(task_id=task_id)
