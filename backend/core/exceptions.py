# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Domain exception hierarchy for Edge Studio.

All domain-specific exceptions inherit from EdgeStudioError so callers can
catch broad or narrow as needed.  The error_mapper translates these to
user-friendly messages for the frontend.
"""

from __future__ import annotations


class EdgeStudioError(Exception):
    """Base exception for all Edge Studio domain errors."""


# ── Model lifecycle ─────────────────────────────────────────────────────

class ModelNotLoadedError(EdgeStudioError):
    """Requested model is not currently loaded."""


class ModelLoadError(EdgeStudioError):
    """Failed to load a model (corrupt files, missing config, OOM, etc.)."""


# ── Export ──────────────────────────────────────────────────────────────

class ExportError(EdgeStudioError):
    """Base for all export-related failures."""


class GGUFConverterNotFoundError(ExportError):
    """llama.cpp convert_hf_to_gguf.py not found."""


# ── Inference / generation ──────────────────────────────────────────────

class InferenceError(EdgeStudioError):
    """Failure during model inference or token generation."""


class ChatTemplateError(InferenceError):
    """Failed to apply chat template to messages."""


# ── Optimization ────────────────────────────────────────────────────────

class PruningError(EdgeStudioError):
    """Failure during pruning simulation or execution."""


class QuantizationError(EdgeStudioError):
    """Failure during quantization."""


# ── Task management ─────────────────────────────────────────────────────

class TaskCancelledError(EdgeStudioError):
    """Raised when a running task is cancelled by the user."""
