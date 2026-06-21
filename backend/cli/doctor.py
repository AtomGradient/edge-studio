# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Read-only environment checks for the ``edge doctor`` command."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from backend.cli.model_paths import detect_model_dirs, model_cache_roots


SCHEMA_VERSION = "edge.doctor.report.v1"

Status = str


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: Status
    summary: str
    details: dict[str, object]
    remediation: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class DoctorReport:
    schema_version: str
    overall_status: Status
    checks: list[CheckResult]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "overall_status": self.overall_status,
            "checks": [check.as_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], float], CommandResult]
HealthGetter = Callable[[str, float], tuple[int, str]]


def run_doctor(
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    python_version: tuple[int, int, int] | None = None,
    python_executable: str | None = None,
    command_runner: CommandRunner | None = None,
    health_getter: HealthGetter | None = None,
) -> DoctorReport:
    """Run the B1 doctor checks without mutating local state."""

    root = project_root or Path(__file__).resolve().parents[2]
    env = environ or os.environ
    runner = command_runner or _run_command
    get_health = health_getter or _get_backend_health
    version = python_version or sys.version_info[:3]
    executable = python_executable or sys.executable

    checks = [
        _check_python_version(version, executable),
        _check_virtualenv(env),
        _check_python_packages(),
        _check_node_and_npm(runner),
        _check_xcode_and_swift(runner),
        _check_preview_repos(root),
        _check_model_cache_roots(env),
        _check_backend_health(env, get_health),
    ]
    return DoctorReport(
        schema_version=SCHEMA_VERSION,
        overall_status=_overall_status(checks),
        checks=checks,
    )


def format_human(report: DoctorReport) -> str:
    lines = [
        f"Edge doctor ({report.schema_version})",
        f"overall: {report.overall_status}",
    ]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.id}: {check.summary}")
        if check.remediation:
            lines.append(f"  remediation: {check.remediation}")
    return "\n".join(lines)


def _overall_status(checks: Sequence[CheckResult]) -> Status:
    statuses = {check.status for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _check_python_version(version: tuple[int, int, int], executable: str) -> CheckResult:
    label = ".".join(str(part) for part in version)
    if version < (3, 11, 0):
        return CheckResult(
            id="python.version",
            status="fail",
            summary=f"Python {label} is below the required 3.11",
            details={"version": label, "executable": executable},
            remediation="Use a Python 3.11+ environment before running Edge Developer Preview commands.",
        )
    return CheckResult(
        id="python.version",
        status="ok",
        summary=f"Python {label}",
        details={"version": label, "executable": executable},
    )


def _check_virtualenv(env: Mapping[str, str]) -> CheckResult:
    virtual_env = env.get("VIRTUAL_ENV", "")
    in_venv = bool(virtual_env) or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv:
        return CheckResult(
            id="python.virtualenv",
            status="warn",
            summary="No active virtual environment detected",
            details={"virtual_env": None, "sys_prefix": sys.prefix},
            remediation="Use the EdgeStudio Python 3.11 environment or another isolated venv.",
        )
    return CheckResult(
        id="python.virtualenv",
        status="ok",
        summary=f"Virtual environment active: {virtual_env or sys.prefix}",
        details={"virtual_env": virtual_env or None, "sys_prefix": sys.prefix},
    )


def _check_python_packages() -> CheckResult:
    packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "httpx",
        "edge-studio",
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "mlx-audio",
    ]
    installed: dict[str, str] = {}
    missing: list[str] = []
    for package in packages:
        try:
            installed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)

    if missing:
        return CheckResult(
            id="python.packages",
            status="warn",
            summary=f"Missing optional/runtime packages: {', '.join(missing)}",
            details={"installed": installed, "missing": missing},
            remediation="Install EdgeStudio dependencies in the active Python environment.",
        )
    return CheckResult(
        id="python.packages",
        status="ok",
        summary="Required Python package metadata is available",
        details={"installed": installed, "missing": []},
    )


def _check_node_and_npm(runner: CommandRunner) -> CheckResult:
    node = runner(["node", "--version"], 5)
    npm = runner(["npm", "--version"], 5)
    node_version = _first_line(node.stdout)
    npm_version = _first_line(npm.stdout)
    node_major = _parse_major_version(node_version)

    details = {
        "node_returncode": node.returncode,
        "node_version": node_version,
        "npm_returncode": npm.returncode,
        "npm_version": npm_version,
    }
    if node.returncode != 0 or npm.returncode != 0:
        return CheckResult(
            id="node.toolchain",
            status="fail",
            summary="Node.js and npm are required for developer docs tooling",
            details=details,
            remediation="Install Node.js 20+ and npm, then rerun edge doctor.",
        )
    if node_major is None or node_major < 20:
        return CheckResult(
            id="node.toolchain",
            status="fail",
            summary=f"Node.js {node_version or 'unknown'} is below the required 20.x",
            details=details,
            remediation="Upgrade to Node.js 20+.",
        )
    return CheckResult(
        id="node.toolchain",
        status="ok",
        summary=f"Node.js {node_version}; npm {npm_version}",
        details=details,
    )


def _check_xcode_and_swift(runner: CommandRunner) -> CheckResult:
    xcode = runner(["xcodebuild", "-version"], 5)
    swift = runner(["swift", "--version"], 5)
    xcode_version = _first_line(xcode.stdout)
    swift_version = _first_line(swift.stdout)
    details = {
        "xcodebuild_returncode": xcode.returncode,
        "xcodebuild_version": xcode_version,
        "swift_returncode": swift.returncode,
        "swift_version": swift_version,
    }
    if xcode.returncode != 0 or swift.returncode != 0:
        return CheckResult(
            id="apple.toolchain",
            status="fail",
            summary="Xcode command line tools and Swift are required",
            details=details,
            remediation="Install Xcode command line tools and verify `xcodebuild -version` and `swift --version`.",
        )
    return CheckResult(
        id="apple.toolchain",
        status="ok",
        summary=f"{xcode_version}; {swift_version}",
        details=details,
    )


def _check_preview_repos(project_root: Path) -> CheckResult:
    expected = {
        "edge-kit": ["Package.swift"],
        "edge-halo": ["Package.swift"],
        "edge-engine": ["Package.swift"],
        "edge-scaffold": ["project.yml", "EdgeScaffold.xcodeproj"],
    }
    found: dict[str, str] = {}
    missing: dict[str, list[str]] = {}
    for repo, candidates in expected.items():
        repo_root = project_root / repo
        match = next((candidate for candidate in candidates if (repo_root / candidate).exists()), None)
        if match:
            found[repo] = str(repo_root / match)
        else:
            missing[repo] = [str(repo_root / candidate) for candidate in candidates]

    if missing:
        return CheckResult(
            id="preview.repos",
            status="warn",
            summary=f"Missing preview repo manifests: {', '.join(sorted(missing))}",
            details={"found": found, "missing": missing},
            remediation="Some preview repositories may require AtomGradient internal preview or SSH access.",
        )
    return CheckResult(
        id="preview.repos",
        status="ok",
        summary="Local preview repositories are present",
        details={"found": found, "missing": {}},
    )


def _check_model_cache_roots(env: Mapping[str, str]) -> CheckResult:
    roots = model_cache_roots(env)
    existing: list[str] = []
    detected_models: list[str] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        existing.append(str(root))
        detected_models.extend(detect_model_dirs(root))

    if not existing:
        return CheckResult(
            id="model.cache",
            status="warn",
            summary="No model cache roots found",
            details={"roots": [str(root) for root in roots], "existing_roots": [], "detected_models": []},
            remediation="Fetch or place models before running demo commands; B2 will add `edge models fetch`.",
        )
    summary = f"{len(existing)} model cache root(s), {len(detected_models)} shallow model candidate(s)"
    return CheckResult(
        id="model.cache",
        status="ok",
        summary=summary,
        details={
            "roots": [str(root) for root in roots],
            "existing_roots": existing,
            "detected_models": detected_models[:20],
            "truncated": len(detected_models) > 20,
        },
    )


def _check_backend_health(env: Mapping[str, str], health_getter: HealthGetter) -> CheckResult:
    host = env.get("VLM_HOST", "127.0.0.1")
    port = env.get("VLM_PORT", "18842")
    url = f"http://{host}:{port}/api/health"
    try:
        status_code, body = health_getter(url, 1.5)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            id="backend.health",
            status="warn",
            summary="EdgeStudio backend is not running or not reachable",
            details={"url": url, "error": str(exc)},
            remediation="Start the backend when you need API-backed demo commands; `edge doctor` itself does not start it.",
        )
    if status_code != 200:
        return CheckResult(
            id="backend.health",
            status="warn",
            summary=f"EdgeStudio backend returned HTTP {status_code}",
            details={"url": url, "status_code": status_code, "body": body[:300]},
            remediation="Check the backend process before running API-backed demo commands.",
        )
    return CheckResult(
        id="backend.health",
        status="ok",
        summary="EdgeStudio backend is reachable",
        details={"url": url, "status_code": status_code, "body": body[:300]},
    )


def _run_command(args: Sequence[str], timeout: float) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(completed.returncode, completed.stdout.strip(), completed.stderr.strip())
    except FileNotFoundError as exc:
        return CommandResult(127, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return CommandResult(124, stdout.strip(), stderr.strip() or "command timed out")


def _get_backend_health(url: str, timeout: float) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(4096).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(4096).decode("utf-8", errors="replace")


def _first_line(value: str) -> str:
    return value.strip().splitlines()[0].strip() if value.strip() else ""


def _parse_major_version(value: str) -> int | None:
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None
