# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Correction-triggered Neural Imprint regeneration coordinator.

This coordinator deliberately consumes only canonical correction context and
canonical Persona/RPP input. It does not parse app business payloads and it does
not push artifacts to devices automatically.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .correction_compiler import compile_correction_context
from .correction_ledger import (
    CorrectionLedgerError,
    build_correction_consumer_context,
)
from .host_rpp_processor import (
    HostRPPProcessorError,
    load_tool_schema_export_from_model_dir,
    process_canonical_rpp_input_to_persona_source,
)
from .neural_imprint_generation import (
    NeuralImprintGenerationError,
    enqueue_neural_imprint_generation,
)
from .persona_source_store import PersonaSourceStoreError

CORRECTION_REGEN_RECEIPT_SCHEMA_VERSION = "edgestudio.correction_regen_receipt.v1"


@dataclass
class CorrectionRegenError(ValueError):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": False,
            "details": self.details or {},
        }


def regenerate_neural_imprint_from_corrections(
    *,
    peer_id: str,
    model_dir: str | Path,
    model_id: str | None = None,
    base_model_id: str | None = None,
    validate_restore: bool = False,
    include_statuses: Sequence[str] | None = None,
    tool_schema_export: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    input_root: Path | None = None,
    source_root: Path | None = None,
    ledger_root: Path | None = None,
    created_at: float | None = None,
) -> dict[str, Any]:
    """Create a corrected persona source and queue Neural Imprint generation."""

    clean_peer_id = _required_text(peer_id, "peer_id")
    resolved_model_dir = Path(str(model_dir)).expanduser().resolve()
    context = _correction_context(
        peer_id=clean_peer_id,
        include_statuses=include_statuses,
        root=ledger_root,
    )
    flags = _object(context.get("flags"))
    compiled = compile_correction_context(context)
    overlay_text = _text(compiled.get("overlay_text"))
    fingerprints = _list_of_text(compiled.get("included_correction_fingerprints"))

    if not flags.get("requires_neural_imprint_regen") or not overlay_text:
        status = (
            "skipped_no_compiled_overlay"
            if flags.get("requires_neural_imprint_regen")
            else "skipped_no_regen_required"
        )
        return {
            "ok": True,
            "schema_version": CORRECTION_REGEN_RECEIPT_SCHEMA_VERSION,
            "status": status,
            "peer_id": clean_peer_id,
            "correction_context": _context_summary(context),
            "compiled_correction_overlay": _compiled_summary(compiled),
            "correction_overlay_sha256": None,
            "persona_source_receipt": None,
            "generation_job": None,
            "audit": _audit(triggers_neural_imprint_regen=False),
        }

    try:
        resolved_tool_schema_export = (
            tool_schema_export
            if tool_schema_export is not None
            else load_tool_schema_export_from_model_dir(resolved_model_dir)
        )
        source_receipt = process_canonical_rpp_input_to_persona_source(
            peer_id=clean_peer_id,
            tool_schema_export=resolved_tool_schema_export,
            base_model_id=base_model_id or model_id,
            profile_body_suffix=overlay_text,
            lineage_extra={
                "correction_context_schema_version": str(context.get("schema_version") or ""),
                "correction_compiler_schema_version": str(compiled.get("schema_version") or ""),
                "correction_fingerprints": fingerprints,
                "correction_overlay_sha256": _sha256_text(overlay_text),
                "correction_counts": dict(_object(compiled.get("counts"))),
            },
            input_root=input_root,
            source_root=source_root,
            created_at=created_at,
        )
        job = enqueue_neural_imprint_generation(
            peer_id=clean_peer_id,
            model_dir=resolved_model_dir,
            model_id=model_id,
            validate_restore=validate_restore,
        )
    except (
        CorrectionLedgerError,
        HostRPPProcessorError,
        NeuralImprintGenerationError,
        PersonaSourceStoreError,
    ) as exc:
        raise _wrap_error(exc) from exc

    return {
        "ok": True,
        "schema_version": CORRECTION_REGEN_RECEIPT_SCHEMA_VERSION,
        "status": "queued",
        "peer_id": clean_peer_id,
        "correction_context": _context_summary(context),
        "compiled_correction_overlay": _compiled_summary(compiled),
        "correction_overlay_sha256": _sha256_text(overlay_text),
        "persona_source_receipt": source_receipt,
        "generation_job": job,
        "audit": _audit(triggers_neural_imprint_regen=True),
    }


def build_correction_overlay_text(context: Mapping[str, Any]) -> str:
    """Render the compiled correction overlay for compatibility callers."""

    return _text(compile_correction_context(context).get("overlay_text"))


def _correction_context(
    *,
    peer_id: str,
    include_statuses: Sequence[str] | None,
    root: Path | None,
) -> dict[str, Any]:
    try:
        return build_correction_consumer_context(
            peer_id=peer_id,
            include_statuses=list(include_statuses) if include_statuses else None,
            root=root,
        )
    except CorrectionLedgerError as exc:
        raise _wrap_error(exc) from exc


def _compiled_summary(compiled: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(compiled.get("schema_version") or ""),
        "status": str(compiled.get("status") or ""),
        "policy": dict(_object(compiled.get("policy"))),
        "counts": dict(_object(compiled.get("counts"))),
        "included_correction_fingerprints": _list_of_text(
            compiled.get("included_correction_fingerprints")
        ),
        "overlay_sha256": compiled.get("overlay_sha256"),
        "audit": dict(_object(compiled.get("audit"))),
    }


def _context_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": str(context.get("schema_version") or ""),
        "status": str(context.get("status") or ""),
        "peer_id": str(context.get("peer_id") or ""),
        "counts": dict(_object(context.get("counts"))),
        "flags": dict(_object(context.get("flags"))),
        "rpp_context": dict(_object(context.get("rpp_context"))),
    }


def _audit(*, triggers_neural_imprint_regen: bool) -> dict[str, Any]:
    return {
        "writes_persona_source": bool(triggers_neural_imprint_regen),
        "triggers_neural_imprint_regen": bool(triggers_neural_imprint_regen),
        "triggers_capsule_push": False,
        "automatic_push": False,
        "consumes_canonical_correction_context": True,
        "consumes_canonical_rpp_input": bool(triggers_neural_imprint_regen),
    }


def _wrap_error(exc: Exception) -> CorrectionRegenError:
    code = getattr(exc, "code", "correction_regen_failed")
    message = getattr(exc, "message", str(exc))
    details = getattr(exc, "details", None)
    return CorrectionRegenError(str(code), str(message), details if isinstance(details, dict) else {})


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _list_of_objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _list_of_text(value: Any) -> list[str]:
    return [_text(item) for item in value if _text(item)] if isinstance(value, list) else []


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise CorrectionRegenError(
            "missing_required_field",
            f"{field} must be a non-empty string",
            {"field": field},
        )
    return text


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()
