# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""System information endpoint — device name, memory, chip, device profile matching."""

from __future__ import annotations

import platform
import subprocess

from typing import Any

from fastapi import APIRouter

from backend.core.device_profiles import DEVICE_PROFILES, DeviceProfile, make_profile_from_ram

router = APIRouter(prefix="/api", tags=["system"])


def _get_mac_info() -> dict:
    """Get macOS-specific system info via sysctl/system_profiler."""
    info: dict = {
        "device_name": platform.node(),
        "chip": "Unknown",
        "total_memory_gb": 0,
        "available_memory_gb": 0,
        "gpu_cores": 0,
        "os": "Darwin",
        "os_version": platform.mac_ver()[0] or platform.version(),
        "arch": platform.machine(),
        "matched_device": None,
        "max_model_size_gb": None,
    }

    # Total physical memory via sysctl
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
        ).strip()
        total_bytes = int(raw)
        info["total_memory_gb"] = round(total_bytes / (1024**3), 1)
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    # Available memory via vm_stat
    try:
        raw = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        page_size = 16384  # default on Apple Silicon
        free_pages = 0
        inactive_pages = 0
        for line in raw.splitlines():
            if "page size of" in line:
                page_size = int(line.split()[-2])
            elif "Pages free" in line:
                free_pages = int(line.split()[-1].rstrip("."))
            elif "Pages inactive" in line:
                inactive_pages = int(line.split()[-1].rstrip("."))
        available_bytes = (free_pages + inactive_pages) * page_size
        info["available_memory_gb"] = round(available_bytes / (1024**3), 1)
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    # Chip name
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5,
        ).strip()
        info["chip"] = chip
    except (subprocess.SubprocessError, OSError, ValueError):
        # Apple Silicon doesn't have brand_string — use system_profiler
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "hw.optional.arm64"], text=True, timeout=5,
            ).strip()
            if chip == "1":
                sp = subprocess.check_output(
                    ["system_profiler", "SPHardwareDataType"],
                    text=True, timeout=10,
                )
                for line in sp.splitlines():
                    if "Chip" in line and ":" in line:
                        info["chip"] = line.split(":", 1)[1].strip()
                        break
        except (subprocess.SubprocessError, OSError, ValueError):
            pass

    # Friendly device name from system_profiler
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"],
            text=True, timeout=10,
        )
        for line in sp.splitlines():
            if "Model Name" in line and ":" in line:
                info["device_name"] = line.split(":", 1)[1].strip()
                break
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    # GPU core count
    try:
        import json as _json
        result = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            text=True, timeout=10,
        )
        data = _json.loads(result)
        displays = data.get("SPDisplaysDataType", [])
        for d in displays:
            cores = d.get("sppci_cores", "")
            if cores:
                try:
                    info["gpu_cores"] = int(str(cores).strip())
                except ValueError:
                    pass
    except (subprocess.SubprocessError, OSError, ValueError):
        pass

    # Match to closest known device profile, fallback to RAM-based estimate
    matched = _match_device_profile(info["chip"], info["total_memory_gb"])
    if matched:
        info["matched_device"] = matched.name
        info["max_model_size_gb"] = round(matched.max_model_size_gb, 1)
    else:
        # No exact profile match — use actual RAM to estimate
        synth = make_profile_from_ram(info["chip"], info["total_memory_gb"])
        info["matched_device"] = synth.name
        info["max_model_size_gb"] = round(synth.max_model_size_gb, 1)

    # Alias for wizard compatibility
    info["ram_gb"] = info["total_memory_gb"]

    return info


def _match_device_profile(chip: str, ram_gb: float) -> DeviceProfile | None:
    """Find the closest matching Mac device profile for current hardware.

    Prefers longer chip name matches (M4 Pro > M4) to avoid
    'M4' matching 'Apple M4 Pro' when 'M4 Pro' exists.
    """
    chip_lower = chip.lower()
    best = None
    best_chip_len = 0
    best_diff = float("inf")

    for profile in DEVICE_PROFILES.values():
        if profile.category != "mac":
            continue
        pchip = profile.chip.lower()
        if pchip in chip_lower:
            chip_len = len(pchip)
            diff = abs(profile.ram_gb - ram_gb)
            # Prefer longer chip name match, then closest RAM
            if chip_len > best_chip_len or (chip_len == best_chip_len and diff < best_diff):
                best = profile
                best_chip_len = chip_len
                best_diff = diff

    return best


@router.get("/system-info", response_model=dict[str, Any])
def get_system_info() -> dict[str, Any]:
    """Return current device info: name, chip, memory, GPU cores, matched profile."""
    if platform.system() == "Darwin":
        return _get_mac_info()
    # Fallback for non-macOS
    return {
        "device_name": platform.node(),
        "chip": platform.processor() or "Unknown",
        "total_memory_gb": 0,
        "available_memory_gb": 0,
        "ram_gb": 0,
        "gpu_cores": 0,
        "os": platform.system(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "matched_device": None,
        "max_model_size_gb": None,
    }
