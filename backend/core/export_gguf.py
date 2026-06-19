# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""GGUF export — convert HuggingFace models to llama.cpp GGUF format.

Uses llama.cpp's native convert_hf_to_gguf.py for correct architecture
metadata (head_dim, attention params, etc.), then llama-quantize for
non-f16 quantization.

Two-step flow:
1. convert_hf_to_gguf.py → f16 GGUF (correct arch, tokenizer, head_dim)
2. If quantization != "f16", llama-quantize → final quantized GGUF
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


# Locations to search for convert_hf_to_gguf.py
_CONVERTER_SEARCH_PATHS = [
    "~/Documents/mlx-community/llama.cpp/convert_hf_to_gguf.py",
    "~/llama.cpp/convert_hf_to_gguf.py",
    "/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py",
]


@dataclass
class GGUFExportResult:
    """Result of GGUF export."""
    success: bool
    output_path: str
    output_size_bytes: int
    duration_seconds: float
    error_message: str = ""
    quantization_type: str = ""


def _find_converter() -> str | None:
    """Find convert_hf_to_gguf.py from llama.cpp."""
    for p in _CONVERTER_SEARCH_PATHS:
        expanded = Path(p).expanduser()
        if expanded.is_file():
            return str(expanded)
    return None


def export_to_gguf(
    model_dir: str,
    output_path: str | None = None,
    quantization: str = "f16",
    progress_callback: Callable[[str, float], None] | None = None,
) -> GGUFExportResult:
    """Export an HF model to GGUF format using llama.cpp native converter.

    Two-step flow:
    1. convert_hf_to_gguf.py → f16 GGUF (always)
    2. If quantization != "f16", llama-quantize → final quantized GGUF

    Args:
        model_dir: path to HuggingFace model directory
        output_path: output .gguf file path (auto-generated if None)
        quantization: GGUF quantization type (f16, q4_k_m, q8_0, etc.)
        progress_callback: (message, progress_fraction) callback
    """
    t0 = time.time()
    quantization = quantization.lower()

    model_path = Path(model_dir)
    if not model_path.exists():
        return GGUFExportResult(
            success=False, output_path="", output_size_bytes=0,
            duration_seconds=0, error_message=f"Model directory not found: {model_dir}",
        )

    # Determine output paths
    model_name = model_path.name
    if output_path is None:
        output_path = str(model_path.parent / f"{model_name}-{quantization}.gguf")

    # Intermediate f16 path (only used when quantization != f16)
    needs_quantize = quantization != "f16"
    if needs_quantize:
        f16_path = str(model_path.parent / f"{model_name}-f16-tmp.gguf")
    else:
        f16_path = output_path

    # Pre-check: find converter
    converter = _find_converter()
    if not converter:
        return GGUFExportResult(
            success=False, output_path=output_path, output_size_bytes=0,
            duration_seconds=time.time() - t0,
            error_message=(
                "convert_hf_to_gguf.py not found. "
                "Please clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp ~/Documents/mlx-community/llama.cpp"
            ),
            quantization_type=quantization,
        )

    # Pre-check: if we need llama-quantize, verify it exists
    if needs_quantize and not shutil.which("llama-quantize"):
        return GGUFExportResult(
            success=False, output_path=output_path, output_size_bytes=0,
            duration_seconds=time.time() - t0,
            error_message=(
                f"Quantization {quantization} requires llama-quantize. "
                "Install: brew install llama.cpp"
            ),
            quantization_type=quantization,
        )

    # Check config exists and model is not MLX-quantized
    config_path = model_path / "config.json"
    if not config_path.exists():
        return GGUFExportResult(
            success=False, output_path=output_path, output_size_bytes=0,
            duration_seconds=time.time() - t0,
            error_message="config.json not found in model directory",
        )

    with open(config_path) as f:
        config = json.load(f)

    if config.get("quantization"):
        return GGUFExportResult(
            success=False, output_path=output_path, output_size_bytes=0,
            duration_seconds=time.time() - t0,
            error_message=(
                "Cannot convert quantized MLX models to GGUF. "
                "Please use the original (non-quantized) model for GGUF export."
            ),
        )

    if progress_callback:
        progress_callback("Converting to f16 GGUF (llama.cpp)...", 0.2)

    try:
        # Step 1: convert_hf_to_gguf.py → f16 GGUF
        result = subprocess.run(
            [sys.executable, converter, str(model_path),
             "--outtype", "f16", "--outfile", f16_path],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "PYTHONPATH": str(Path(converter).parent)},
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Extract the last meaningful line for error display
            err_lines = [l for l in stderr.split("\n") if l.strip()]
            err_msg = err_lines[-1] if err_lines else "Unknown error"
            return GGUFExportResult(
                success=False, output_path=output_path, output_size_bytes=0,
                duration_seconds=time.time() - t0,
                error_message=f"convert_hf_to_gguf.py failed: {err_msg}",
                quantization_type=quantization,
            )

        if not Path(f16_path).exists():
            return GGUFExportResult(
                success=False, output_path=output_path, output_size_bytes=0,
                duration_seconds=time.time() - t0,
                error_message="f16 GGUF file was not created",
            )

        # Step 2: If non-f16 quantization requested, run llama-quantize
        if needs_quantize:
            if progress_callback:
                progress_callback(f"Quantizing to {quantization}...", 0.7)

            result = subprocess.run(
                ["llama-quantize", f16_path, output_path, quantization],
                capture_output=True, text=True, timeout=1800,
            )

            # Clean up intermediate f16 file
            try:
                Path(f16_path).unlink()
            except OSError:
                pass

            if result.returncode != 0:
                return GGUFExportResult(
                    success=False, output_path=output_path, output_size_bytes=0,
                    duration_seconds=time.time() - t0,
                    error_message=f"llama-quantize failed: {result.stderr.strip()}",
                    quantization_type=quantization,
                )

        if progress_callback:
            progress_callback("Export complete!", 1.0)

        output_file = Path(output_path)
        if output_file.exists():
            return GGUFExportResult(
                success=True, output_path=output_path,
                output_size_bytes=output_file.stat().st_size,
                duration_seconds=time.time() - t0,
                quantization_type=quantization,
            )
        else:
            return GGUFExportResult(
                success=False, output_path=output_path, output_size_bytes=0,
                duration_seconds=time.time() - t0,
                error_message="Output file was not created",
            )

    except subprocess.TimeoutExpired:
        # Clean up on timeout
        if needs_quantize:
            try:
                Path(f16_path).unlink(missing_ok=True)
            except OSError:
                pass
        return GGUFExportResult(
            success=False, output_path=output_path, output_size_bytes=0,
            duration_seconds=time.time() - t0,
            error_message="Export timed out", quantization_type=quantization,
        )
    except Exception as e:
        # Clean up intermediate file on error
        if needs_quantize:
            try:
                Path(f16_path).unlink(missing_ok=True)
            except OSError:
                pass
        return GGUFExportResult(
            success=False, output_path=output_path, output_size_bytes=0,
            duration_seconds=time.time() - t0,
            error_message=str(e), quantization_type=quantization,
        )


# Available GGUF quantization types
# f16 is produced by convert_hf_to_gguf.py.
# Others use llama-quantize (from llama.cpp) as a post-processing step.
GGUF_QUANTIZATION_TYPES = [
    ("f16", "Float16 — direct conversion"),
    ("q8_0", "8-bit quantization"),
    ("q6_k", "6-bit quantization"),
    ("q5_k_m", "5-bit quantization"),
    ("q4_k_m", "4-bit quantization (recommended)"),
    ("q4_0", "4-bit quantization (basic)"),
    ("q3_k_m", "3-bit quantization"),
    ("q2_k", "2-bit quantization"),
]
