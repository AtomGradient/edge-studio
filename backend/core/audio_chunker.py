# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Smart audio chunking — split long audio at silence boundaries for STT."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Analysis sample rate — low enough for memory efficiency, high enough for energy accuracy
_ANALYSIS_SR = 8000
_WINDOW_MS = 100
_WINDOW_SAMPLES = _ANALYSIS_SR * _WINDOW_MS // 1000  # 800 samples per window


@dataclass
class AudioChunk:
    """A segment of audio with time boundaries."""
    index: int
    start_time: float   # seconds
    end_time: float     # seconds
    duration: float     # seconds


def _get_duration(audio_path: str) -> float:
    """Get audio duration using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (ImportError, OSError, ValueError, RuntimeError):
        # Fallback to soundfile
        import soundfile as sf
        return sf.info(audio_path).duration


def needs_chunking(audio_path: str, max_duration: float = 600.0) -> bool:
    """Check if audio file exceeds max_duration and needs chunking.

    Default 600s (10 min) — SenseVoice handles up to ~5min easily,
    but OOMs on very long audio (>60min). 10min gives safe margin.
    """
    try:
        return _get_duration(audio_path) > max_duration
    except (ImportError, OSError, ValueError, RuntimeError):
        return False


def _stream_rms_energy(audio_path: str) -> tuple[list[float], float]:
    """Compute RMS energy via ffmpeg streaming — constant ~3MB memory.

    Streams audio as 8kHz mono PCM, computes RMS in 100ms windows.
    Never loads the full file into memory.

    Returns:
        (rms_values, total_duration_seconds)
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-i", audio_path,
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(_ANALYSIS_SR), "-ac", "1", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    rms_values: list[float] = []
    buf = b""
    total_samples = 0
    bytes_per_window = _WINDOW_SAMPLES * 4  # float32

    try:
        while True:
            # Read ~10 seconds at a time: 8000 * 10 * 4 = 320KB
            chunk = proc.stdout.read(_ANALYSIS_SR * 10 * 4)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= bytes_per_window:
                window_bytes = buf[:bytes_per_window]
                buf = buf[bytes_per_window:]
                samples = np.frombuffer(window_bytes, dtype=np.float32)
                rms_values.append(float(np.sqrt(np.mean(samples ** 2))))
                total_samples += len(samples)
    finally:
        proc.stdout.close()
        proc.wait()

    total_duration = total_samples / _ANALYSIS_SR
    return rms_values, total_duration


def chunk_audio(
    audio_path: str,
    target_duration: float = 120.0,
    max_duration: float = 600.0,
    min_duration: float = 10.0,
) -> list[AudioChunk] | None:
    """Split audio at silence boundaries for models without built-in chunking.

    Uses ffmpeg streaming for energy analysis — constant ~3MB memory
    regardless of file size. Never loads the full audio into memory.

    Algorithm:
    1. Stream audio at 8kHz mono via ffmpeg, compute RMS in 100ms windows
    2. Smooth energy, find quiet regions (< 30% of median, >= 200ms)
    3. Greedily split at quietest points near target boundaries

    Args:
        audio_path: Path to audio file.
        target_duration: Ideal chunk length in seconds (default 120s).
        max_duration: Below this, no split needed (default 600s).
        min_duration: Minimum chunk length (default 10s).

    Returns:
        List of AudioChunk, or None if no splitting needed.
    """
    total_duration = _get_duration(audio_path)
    if total_duration <= max_duration:
        return None

    # Stream energy analysis — ~3MB peak memory
    rms_raw, _ = _stream_rms_energy(audio_path)
    n_windows = len(rms_raw)
    if n_windows == 0:
        return None

    rms = np.array(rms_raw, dtype=np.float32)

    # Smooth with 500ms moving average (5 windows at 100ms each)
    smooth_len = 5
    kernel = np.ones(smooth_len, dtype=np.float32) / smooth_len
    rms_smooth = np.convolve(rms, kernel, mode="same")

    # Find quiet regions: below 30% of median, >= 200ms (2 windows)
    median_energy = float(np.median(rms_smooth))
    quiet_threshold = median_energy * 0.3

    quiet_midpoints: list[tuple[float, float]] = []  # (time_seconds, energy)
    region_start = None
    for i in range(n_windows):
        if rms_smooth[i] < quiet_threshold:
            if region_start is None:
                region_start = i
        else:
            if region_start is not None and i - region_start >= 2:
                mid = (region_start + i) // 2
                quiet_midpoints.append((mid * _WINDOW_MS / 1000, float(rms_smooth[mid])))
            region_start = None

    # Greedy splitting
    chunks: list[AudioChunk] = []
    chunk_start = 0.0
    idx = 0

    for target_end in np.arange(target_duration, total_duration, target_duration):
        search_start = max(chunk_start + min_duration, target_end - 15)
        search_end = min(target_end + 15, total_duration)

        candidates = [
            (t, e) for t, e in quiet_midpoints
            if search_start <= t <= search_end
        ]

        if candidates:
            best_time = min(candidates, key=lambda x: x[1])[0]
        else:
            w_start = max(0, int(search_start * 1000 / _WINDOW_MS))
            w_end = min(n_windows, int(search_end * 1000 / _WINDOW_MS))
            if w_start < w_end:
                best_window = w_start + int(np.argmin(rms_smooth[w_start:w_end]))
                best_time = best_window * _WINDOW_MS / 1000
            else:
                best_time = target_end

        best_time = min(best_time, total_duration)
        if best_time > chunk_start:
            chunks.append(AudioChunk(idx, chunk_start, best_time, best_time - chunk_start))
            idx += 1
        chunk_start = best_time

    # Last chunk
    if chunk_start < total_duration - 1:
        chunks.append(AudioChunk(idx, chunk_start, total_duration, total_duration - chunk_start))

    logger.info(
        "Audio chunked: %s → %d chunks (%.0fs total, target=%.0fs, peak ~3MB)",
        os.path.basename(audio_path), len(chunks), total_duration, target_duration,
    )
    return chunks
