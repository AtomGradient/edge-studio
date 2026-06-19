# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Map exceptions to user-friendly error messages."""

from __future__ import annotations

import errno
import logging

logger = logging.getLogger(__name__)

# Re-export TaskCancelledError from canonical location for backward compat
from backend.core.exceptions import (  # noqa: F401
    TaskCancelledError,
    EdgeStudioError,
    ModelNotLoadedError,
    ModelLoadError,
    ExportError,
    GGUFConverterNotFoundError,
    InferenceError,
    ChatTemplateError,
    PruningError,
    QuantizationError,
)


_ERRNO_MESSAGES: dict[int, str] = {
    errno.ENOENT: "File or directory not found. Please check the path.",
    errno.EACCES: "Permission denied. Check file permissions.",
    errno.ENOSPC: "No disk space left. Free up space and try again.",
    errno.EISDIR: "Expected a file but found a directory.",
}


def map_error(exc: Exception) -> tuple[str, str]:
    """Return ``(user_message, debug_detail)`` for *exc*.

    *user_message* is safe for display in the UI.
    *debug_detail* contains the full repr for server-side logging.
    """
    debug = repr(exc)

    if isinstance(exc, TaskCancelledError):
        return "Operation cancelled by user.", debug

    if isinstance(exc, ModelNotLoadedError):
        return "Model not loaded. Please load a model first.", debug

    if isinstance(exc, ModelLoadError):
        return f"Failed to load model: {exc}", debug

    if isinstance(exc, GGUFConverterNotFoundError):
        return "GGUF converter not found. Install llama.cpp or set LLAMA_CPP_CONVERTER_PATH.", debug

    if isinstance(exc, ChatTemplateError):
        return "Failed to apply chat template. The model may use an unsupported format.", debug

    if isinstance(exc, (PruningError, QuantizationError)):
        return f"Optimization failed: {exc}", debug

    if isinstance(exc, InferenceError):
        return f"Inference error: {exc}", debug

    if isinstance(exc, ExportError):
        return f"Export failed: {exc}", debug

    if isinstance(exc, EdgeStudioError):
        return str(exc), debug

    # --- Built-in exceptions ---

    if isinstance(exc, FileNotFoundError):
        return "Model directory not found. Please check the path.", debug

    if isinstance(exc, MemoryError):
        return "Not enough memory. Try a smaller model or close other apps.", debug

    if isinstance(exc, KeyError):
        return f"Unsupported model config format (missing key: {exc}).", debug

    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _ERRNO_MESSAGES:
        return _ERRNO_MESSAGES[exc.errno], debug

    if isinstance(exc, ValueError):
        return str(exc), debug

    return "An unexpected error occurred. Check backend logs for details.", debug
