# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Explicit model fetch command for the Edge developer CLI."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

from backend.cli.fingerprints import dir_size_bytes, directory_manifest_hash, model_dir_integrity, pretty_json
from backend.cli.models import CatalogResolution, resolve_model_reference, where_model
from backend.resources.paths import script_path
from backend.services.app_dirs import data_path


FETCH_SCHEMA_VERSION = "edge.models.fetch.receipt.v1"
SourceName = Literal["modelscope", "huggingface", "hf-mirror"]
SourceOption = Literal["auto", "modelscope", "huggingface", "hf-mirror"]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


CommandRunner = Callable[[Sequence[str], Mapping[str, str], float | None], CommandResult]
ProbeRunner = Callable[[SourceName, float], dict[str, object]]


@dataclass(frozen=True)
class FetchOptions:
    source: SourceOption = "auto"
    download_dir: Path | None = None
    receipt_path: Path | None = None
    dry_run: bool = False
    no_probe: bool = False
    force: bool = False
    clean: bool = False
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    status: str
    exit_code: int
    receipt: dict[str, object]
    receipt_path: Path | None

    def to_json(self) -> str:
        return json.dumps(self.receipt, ensure_ascii=False, indent=2)


def fetch_model(
    model_ref: str,
    *,
    options: FetchOptions | None = None,
    env: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
    prober: ProbeRunner | None = None,
) -> FetchResult:
    opts = options or FetchOptions()
    values = env or os.environ
    command_runner = runner or _run_command
    probe_runner = prober or _probe_source
    started = _now_utc()
    start_monotonic = time.monotonic()

    resolution = resolve_model_reference(model_ref)
    if resolution.status == "unknown":
        receipt = _base_receipt(
            model_ref=model_ref,
            resolution=resolution,
            source_order=[],
            probe_summary=[],
            status="unknown_model",
            started_at=started,
        )
        receipt["ok"] = False
        receipt["error"] = {
            "code": "unknown_model",
            "message": "Model is not in the bundled catalog; pass an explicit repository id such as org/name.",
        }
        return _finish_result(receipt, opts, start_monotonic, write_receipt=not opts.dry_run)

    repo_id = _repo_id_for_resolution(resolution)
    download_root = _download_root(opts.download_dir, values)
    local_dir = download_root / _local_dir_name(repo_id)
    cleaned_path: str | None = None
    if opts.clean and not opts.dry_run and local_dir.exists():
        shutil.rmtree(local_dir)
        cleaned_path = str(local_dir)
    before = where_model(model_ref, env=values)

    if before.status == "ok" and not opts.force and not opts.clean and not opts.dry_run and before.local_matches:
        local = before.local_matches[0]
        local_path = Path(local.path)
        receipt = _base_receipt(
            model_ref=model_ref,
            resolution=resolution,
            source_order=[],
            probe_summary=[],
            status="skipped_existing",
            started_at=started,
        )
        receipt.update({
            "ok": True,
            "repo_id": repo_id,
            "download_dir": str(download_root),
            "path": str(local_path),
            "force": opts.force,
            "clean": opts.clean,
            "dry_run": opts.dry_run,
            "selected_source": None,
            "size_bytes": local.size_bytes,
            "resumable": True,
            "attempted_sources": [],
            **directory_manifest_hash(local_path),
        })
        return _finish_result(receipt, opts, start_monotonic, write_receipt=True)

    source_order, probe_summary = _select_sources(
        requested=opts.source,
        no_probe=opts.no_probe,
        env=values,
        prober=probe_runner,
    )
    receipt = _base_receipt(
        model_ref=model_ref,
        resolution=resolution,
        source_order=source_order,
        probe_summary=probe_summary,
        status="planned" if opts.dry_run else "started",
        started_at=started,
    )
    receipt.update({
        "repo_id": repo_id,
        "download_dir": str(download_root),
        "path": str(local_dir),
        "force": opts.force,
        "clean": opts.clean,
        "cleaned_path": cleaned_path,
        "dry_run": opts.dry_run,
    })

    if opts.dry_run:
        receipt["ok"] = True
        receipt["status"] = "dry_run"
        return _finish_result(receipt, opts, start_monotonic, write_receipt=False)

    download_root.mkdir(parents=True, exist_ok=True)
    _progress(f"[models:fetch] source plan: {' -> '.join(source_order)}")
    _progress(f"[models:fetch] target: {local_dir}")
    attempts: list[dict[str, object]] = []
    selected_source: str | None = None
    cleaned_after_failed_sources: list[str] = []
    for source in source_order:
        attempt_started = time.monotonic()
        args, run_env = _download_command(source, repo_id, local_dir, values)
        _progress(f"[models:fetch] starting source={source} repo={repo_id}")
        command_result = command_runner(args, run_env, opts.timeout_seconds)
        elapsed = round(time.monotonic() - attempt_started, 3)
        attempt = {
            "source": source,
            "returncode": command_result.returncode,
            "timed_out": command_result.timed_out,
            "elapsed_seconds": elapsed,
            "stdout_tail": _safe_tail(command_result.stdout),
            "stderr_tail": _safe_tail(command_result.stderr),
        }
        attempts.append(attempt)
        if command_result.returncode == 0:
            selected_source = source
            _progress(f"[models:fetch] source={source} finished in {elapsed}s")
            break
        _progress(f"[models:fetch] source={source} failed returncode={command_result.returncode}; trying next source")
        if local_dir.exists():
            shutil.rmtree(local_dir)
            cleaned_after_failed_sources.append(source)
            _progress(f"[models:fetch] cleaned partial files from failed source={source}")

    receipt["attempted_sources"] = attempts
    receipt["selected_source"] = selected_source
    receipt["cleaned_after_failed_sources"] = cleaned_after_failed_sources
    if selected_source is None:
        receipt["ok"] = False
        receipt["status"] = "download_failed"
        receipt["error"] = {
            "code": "download_failed",
            "message": "All configured download sources failed.",
        }
        receipt["retry_command"] = _retry_command(model_ref, opts)
        return _finish_result(receipt, opts, start_monotonic, write_receipt=True)

    manifest = directory_manifest_hash(local_dir)
    expected_size_bytes = _expected_size_bytes(resolution)
    integrity = model_dir_integrity(local_dir, expected_size_bytes=expected_size_bytes)
    if not integrity.complete:
        receipt.update({
            "ok": False,
            "status": "download_incomplete",
            "path": str(local_dir),
            "size_bytes": dir_size_bytes(local_dir),
            "expected_size_bytes": expected_size_bytes,
            "integrity_issues": list(integrity.issues),
            "resumable": True,
            "retry_command": _retry_command(model_ref, opts),
            **manifest,
        })
        receipt["error"] = {
            "code": "model_integrity_failed",
            "message": "Downloaded model did not pass local integrity checks.",
        }
        return _finish_result(receipt, opts, start_monotonic, write_receipt=True)

    receipt.update({
        "ok": True,
        "status": "downloaded",
        "path": str(local_dir),
        "size_bytes": dir_size_bytes(local_dir),
        "expected_size_bytes": expected_size_bytes,
        "integrity_issues": [],
        "resumable": True,
        **manifest,
    })
    return _finish_result(receipt, opts, start_monotonic, write_receipt=True)


def format_fetch_result(result: FetchResult) -> str:
    receipt = result.receipt
    lines = [
        f"Edge models fetch ({receipt['schema_version']})",
        f"status: {receipt['status']}",
        f"model: {receipt.get('model_ref')}",
    ]
    repo_id = receipt.get("repo_id")
    if repo_id:
        lines.append(f"repo: {repo_id}")
    selected = receipt.get("selected_source")
    if selected:
        lines.append(f"source: {selected}")
    elif receipt.get("source_order"):
        lines.append(f"source plan: {' -> '.join(str(s) for s in receipt['source_order'])}")
    if receipt.get("path"):
        lines.append(f"path: {receipt['path']}")
    if receipt.get("receipt_path"):
        lines.append(f"receipt: {receipt['receipt_path']}")
    if receipt.get("status") == "download_failed":
        lines.append("error: all configured download sources failed")
    if receipt.get("status") == "download_incomplete":
        lines.append("error: downloaded model did not pass integrity checks")
    if receipt.get("integrity_issues"):
        lines.append(f"issues: {', '.join(str(issue) for issue in receipt['integrity_issues'][:5])}")
    if receipt.get("retry_command"):
        lines.append(f"retry: {receipt['retry_command']}")
    return "\n".join(lines)


def _finish_result(
    receipt: dict[str, object],
    opts: FetchOptions,
    start_monotonic: float,
    *,
    write_receipt: bool,
) -> FetchResult:
    receipt["elapsed_seconds"] = round(time.monotonic() - start_monotonic, 3)
    path: Path | None = None
    if write_receipt:
        path = opts.receipt_path or _default_receipt_path(str(receipt.get("model_ref") or "model"))
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt["receipt_path"] = str(path)
        path.write_text(pretty_json(receipt), encoding="utf-8")
    else:
        receipt["receipt_path"] = str(opts.receipt_path.expanduser()) if opts.receipt_path else None
    ok = bool(receipt.get("ok"))
    status = str(receipt.get("status") or "unknown")
    exit_code = 0 if ok or status == "skipped_existing" else 1
    return FetchResult(ok=ok, status=status, exit_code=exit_code, receipt=receipt, receipt_path=path)


def _base_receipt(
    *,
    model_ref: str,
    resolution: CatalogResolution,
    source_order: Sequence[SourceName],
    probe_summary: Sequence[dict[str, object]],
    status: str,
    started_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": FETCH_SCHEMA_VERSION,
        "ok": False,
        "status": status,
        "model_ref": model_ref,
        "resolution": resolution.as_dict(),
        "source_order": list(source_order),
        "network_probe": list(probe_summary),
        "raw_text_in_receipt": False,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
    }


def _repo_id_for_resolution(resolution: CatalogResolution) -> str:
    repo_id = resolution.download_hint or resolution.input
    return repo_id.strip()


def _select_sources(
    *,
    requested: SourceOption,
    no_probe: bool,
    env: Mapping[str, str],
    prober: ProbeRunner,
) -> tuple[list[SourceName], list[dict[str, object]]]:
    if requested != "auto":
        return [requested], []

    probe_summary: list[dict[str, object]] = []
    if not no_probe:
        for source in ("huggingface", "hf-mirror", "modelscope"):
            probe_summary.append(prober(source, 3.0))

    if _env_prefers_china(env) or _hf_probe_failed_or_slow(probe_summary):
        return ["modelscope", "hf-mirror", "huggingface"], probe_summary
    return ["huggingface", "hf-mirror", "modelscope"], probe_summary


def _hf_probe_failed_or_slow(probes: Sequence[dict[str, object]]) -> bool:
    for probe in probes:
        if probe.get("source") != "huggingface":
            continue
        reachable = probe.get("reachable") is True
        latency = probe.get("latency_ms")
        return (not reachable) or (isinstance(latency, (int, float)) and latency >= 2500)
    return False


def _env_prefers_china(env: Mapping[str, str]) -> bool:
    raw = " ".join(
        env.get(name, "")
        for name in (
            "EDGESTUDIO_REGION",
            "EDGESTUDIO_DOWNLOAD_REGION",
            "EDGESTUDIO_MODEL_SOURCE_REGION",
            "EDGE_REGION",
        )
    ).lower()
    return any(token in raw for token in ("china", "cn", "mainland", "zh"))


def _probe_source(source: SourceName, timeout: float) -> dict[str, object]:
    urls = {
        "huggingface": "https://huggingface.co",
        "hf-mirror": "https://hf-mirror.com",
        "modelscope": "https://www.modelscope.cn",
    }
    start = time.monotonic()
    request = urllib.request.Request(urls[source], method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "source": source,
                "reachable": response.status < 500,
                "latency_ms": round((time.monotonic() - start) * 1000),
                "status_code": response.status,
            }
    except urllib.error.HTTPError as exc:
        return {
            "source": source,
            "reachable": exc.code < 500,
            "latency_ms": round((time.monotonic() - start) * 1000),
            "status_code": exc.code,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "source": source,
            "reachable": False,
            "latency_ms": -1,
            "error": _safe_tail(str(exc), limit=300),
        }


def _download_command(
    source: SourceName,
    repo_id: str,
    local_dir: Path,
    env: Mapping[str, str],
) -> tuple[list[str], dict[str, str]]:
    run_env = dict(os.environ)
    run_env.update({key: value for key, value in env.items() if isinstance(key, str) and isinstance(value, str)})
    if source == "modelscope":
        return ["bash", str(script_path("msd.sh")), repo_id, "--local-dir", str(local_dir)], run_env
    if source == "hf-mirror":
        run_env["HF_ENDPOINT"] = "https://hf-mirror.com"
    return ["bash", str(script_path("hfd.sh")), repo_id, "--local-dir", str(local_dir)], run_env


def _run_command(args: Sequence[str], env: Mapping[str, str], timeout: float | None) -> CommandResult:
    output = ""
    start = time.monotonic()
    last_progress = start
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(args),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            returncode = process.wait(timeout=timeout if timeout and timeout > 0 else None)
            return CommandResult(returncode, output, "")

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if timeout and timeout > 0 and time.monotonic() - start > timeout:
                    process.kill()
                    process.wait()
                    return CommandResult(124, output, "command timed out", timed_out=True)

                events = selector.select(timeout=0.5)
                if events:
                    for key, _mask in events:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            break
                        text = chunk.decode("utf-8", errors="replace")
                        output = _append_capped(output, text)
                        _progress(text.rstrip("\n"))
                        last_progress = time.monotonic()
                elif process.poll() is None and time.monotonic() - last_progress >= 15:
                    elapsed = int(time.monotonic() - start)
                    _progress(f"[models:fetch] still downloading... elapsed={elapsed}s")
                    last_progress = time.monotonic()

                if process.poll() is not None:
                    if not selector.get_map():
                        break
                    continue

            return CommandResult(process.returncode or 0, output, "")
        finally:
            selector.close()
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def _download_root(download_dir: Path | None, env: Mapping[str, str]) -> Path:
    if download_dir is not None:
        return download_dir.expanduser()
    override = env.get("EDGESTUDIO_MODEL_DOWNLOAD_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    explicit_roots = env.get("EDGESTUDIO_MODEL_ROOTS", "").strip()
    if explicit_roots:
        separators = "," if "," in explicit_roots else os.pathsep
        for part in explicit_roots.split(separators):
            if part.strip():
                return Path(part).expanduser()
    return Path.home() / "Documents" / "mlx-community"


def _local_dir_name(repo_id: str) -> str:
    name = repo_id.strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "model"


def _default_receipt_path(model_ref: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_ref.strip()) or "model"
    stamp = _now_utc().strftime("%Y%m%dT%H%M%SZ")
    return data_path("receipts", "models", f"{stamp}_{safe}.json")


def _safe_tail(value: str, *, limit: int = 1200) -> str:
    if not value:
        return ""
    text = value[-limit:]
    text = re.sub(
        r"(?i)(authorization|bearer|hf_token|ms_token|token)(\s*[:=]\s*)([^\s]+)",
        r"\1\2***",
        text,
    )
    text = re.sub(r"hf_[A-Za-z0-9_-]{16,}", "hf_***", text)
    return text


def _append_capped(current: str, text: str, *, limit: int = 20000) -> str:
    combined = current + text
    return combined[-limit:] if len(combined) > limit else combined


def _expected_size_bytes(resolution: CatalogResolution) -> int | None:
    if resolution.size_gb is None or resolution.size_gb <= 0:
        return None
    return int(resolution.size_gb * 1_000_000_000)


def _retry_command(model_ref: str, opts: FetchOptions) -> str:
    parts = ["edge", "models", "fetch", model_ref, "--source", opts.source, "--retry"]
    if opts.download_dir is not None:
        parts.extend(["--download-dir", str(opts.download_dir)])
    return " ".join(_shell_token(part) for part in parts)


def _shell_token(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:@%+=,-]+", text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _progress(message: str) -> None:
    if not message:
        return
    print(message, file=sys.stderr, flush=True)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
