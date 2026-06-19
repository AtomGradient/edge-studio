# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Local correction ledger for personalization feedback.

The ledger records user-visible corrections as auditable local data. It does
not mutate facts, trigger RPP, regenerate Neural Imprint, or change routing
behavior. Downstream jobs can consume accepted ledger entries later.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORRECTION_LEDGER_ENTRY_SCHEMA_VERSION = "edgestudio.correction_ledger_entry.v0"
CORRECTION_LEDGER_RECEIPT_SCHEMA_VERSION = "edgestudio.correction_ledger_receipt.v0"
CORRECTION_LEDGER_INDEX_SCHEMA_VERSION = "edgestudio.correction_ledger_index.v0"
CORRECTION_CONSUMER_CONTEXT_SCHEMA_VERSION = (
    "edgestudio.correction_consumer_context.v0"
)

ALLOWED_CORRECTION_TYPES = {
    "eval_feedback",
    "fact_correction",
    "profile_correction",
}
ALLOWED_STATUSES = {
    "recorded",
    "applied",
    "superseded",
    "rejected",
}
ALLOWED_EFFECT_KEYS = {
    "affects_eval",
    "affects_fact_table",
    "affects_profile_overlay",
    "requires_rpp_rerun",
    "requires_neural_imprint_regen",
}
MAX_ENTRIES_RETURNED = 200


@dataclass
class CorrectionLedgerError(ValueError):
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


def default_correction_ledger_root() -> Path:
    configured = os.environ.get("EDGE_CORRECTION_LEDGER_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "EdgeStudio"
        / "correction_ledger"
    )


def record_correction_entry(
    payload: dict[str, Any],
    *,
    root: Path | None = None,
    received_at_ms: int | None = None,
) -> dict[str, Any]:
    """Validate, normalize, and persist one correction ledger entry."""

    entry = normalize_correction_entry(payload, received_at_ms=received_at_ms)
    base = (root or default_correction_ledger_root()).expanduser().resolve()
    peer_dir = base / _path_component(entry["peer_id"], "peer_id")
    peer_dir.mkdir(parents=True, exist_ok=True)

    entry_path = peer_dir / f"{entry['correction_id']}.json"
    is_new = not entry_path.exists()
    entry_path.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "schema_version": CORRECTION_LEDGER_RECEIPT_SCHEMA_VERSION,
        "status": "stored" if is_new else "updated",
        "peer_id": entry["peer_id"],
        "correction_id": entry["correction_id"],
        "correction_type": entry["correction_type"],
        "entry": entry,
        "storage": {
            "path": str(entry_path),
            "is_new": is_new,
        },
        "audit": {
            "writes_runtime_artifacts": False,
            "triggers_rpp": False,
            "triggers_neural_imprint_regen": False,
            "method": "record_correction_entry",
        },
    }


def normalize_correction_entry(
    payload: dict[str, Any],
    *,
    received_at_ms: int | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CorrectionLedgerError(
            "invalid_input",
            "payload must be a JSON object",
            {"type": type(payload).__name__},
        )
    schema_version = str(payload.get("schema_version") or CORRECTION_LEDGER_ENTRY_SCHEMA_VERSION)
    if schema_version != CORRECTION_LEDGER_ENTRY_SCHEMA_VERSION:
        raise CorrectionLedgerError(
            "unsupported_schema_version",
            f"unsupported schema_version: {schema_version}",
            {"expected": CORRECTION_LEDGER_ENTRY_SCHEMA_VERSION},
        )

    correction_type = _required_text(payload.get("correction_type"), "correction_type")
    if correction_type not in ALLOWED_CORRECTION_TYPES:
        raise CorrectionLedgerError(
            "unsupported_correction_type",
            f"unsupported correction_type: {correction_type}",
            {"allowed": sorted(ALLOWED_CORRECTION_TYPES)},
        )
    peer_id = _required_text(payload.get("peer_id"), "peer_id")
    app_id = _text(payload.get("app_id"))
    source = _object(payload.get("source"))
    target = _object(payload.get("target"))
    correction = _object(payload.get("correction"))
    status = _text(payload.get("status")) or "recorded"
    if status not in ALLOWED_STATUSES:
        raise CorrectionLedgerError(
            "unsupported_status",
            f"unsupported status: {status}",
            {"allowed": sorted(ALLOWED_STATUSES)},
        )
    _validate_by_type(correction_type=correction_type, target=target, correction=correction)

    effective_received_at_ms = int(received_at_ms if received_at_ms is not None else time.time() * 1000)
    effects = _effects(correction_type, payload.get("effects"))
    fingerprint_material = {
        "peer_id": peer_id,
        "app_id": app_id,
        "correction_type": correction_type,
        "source": source,
        "target": target,
        "correction": correction,
    }
    fingerprint = _fingerprint(fingerprint_material)
    correction_id = _text(payload.get("correction_id")) or f"corr-{fingerprint[:32]}"

    return {
        "schema_version": CORRECTION_LEDGER_ENTRY_SCHEMA_VERSION,
        "correction_id": correction_id,
        "correction_fingerprint": fingerprint,
        "peer_id": peer_id,
        "app_id": app_id,
        "correction_type": correction_type,
        "status": status,
        "received_at_ms": effective_received_at_ms,
        "source": source,
        "target": target,
        "correction": correction,
        "effects": effects,
        "privacy": {
            "contains_raw_user_text": _contains_raw_user_text(source, target, correction),
            "local_only": True,
            "do_not_include_in_ai_mailbox": True,
        },
        "audit": {
            "writes_runtime_artifacts": False,
            "triggers_rpp": False,
            "triggers_neural_imprint_regen": False,
            "consumer_must_apply_effects_explicitly": True,
        },
    }


def list_correction_entries(
    *,
    peer_id: str | None = None,
    correction_type: str | None = None,
    status: str | None = None,
    limit: int = 100,
    root: Path | None = None,
) -> dict[str, Any]:
    """Return recent correction ledger entries, newest first."""

    if correction_type and correction_type not in ALLOWED_CORRECTION_TYPES:
        raise CorrectionLedgerError(
            "unsupported_correction_type",
            f"unsupported correction_type: {correction_type}",
            {"allowed": sorted(ALLOWED_CORRECTION_TYPES)},
        )
    if status and status not in ALLOWED_STATUSES:
        raise CorrectionLedgerError(
            "unsupported_status",
            f"unsupported status: {status}",
            {"allowed": sorted(ALLOWED_STATUSES)},
        )

    base = (root or default_correction_ledger_root()).expanduser().resolve()
    max_items = max(1, min(int(limit), MAX_ENTRIES_RETURNED))
    entries: list[dict[str, Any]] = []
    if not base.exists():
        return _index_response(
            root=base,
            entries=[],
            peer_id=peer_id,
            correction_type=correction_type,
            status=status,
            limit=max_items,
        )

    peer_dirs = [base / _path_component(peer_id, "peer_id")] if peer_id else [
        path for path in base.iterdir() if path.is_dir()
    ]
    for peer_dir in peer_dirs:
        if not peer_dir.exists() or not peer_dir.is_dir():
            continue
        for path in peer_dir.glob("*.json"):
            entry = _read_entry(path)
            if entry is None:
                continue
            if correction_type and entry.get("correction_type") != correction_type:
                continue
            if status and entry.get("status") != status:
                continue
            entries.append(entry)

    entries.sort(key=lambda item: int(item.get("received_at_ms") or 0), reverse=True)
    return _index_response(
        root=base,
        entries=entries[:max_items],
        peer_id=peer_id,
        correction_type=correction_type,
        status=status,
        limit=max_items,
    )


def load_correction_entry(
    correction_id: str,
    *,
    peer_id: str,
    root: Path | None = None,
) -> dict[str, Any]:
    base = (root or default_correction_ledger_root()).expanduser().resolve()
    entry_path = (
        base
        / _path_component(peer_id, "peer_id")
        / f"{_path_component(correction_id, 'correction_id')}.json"
    )
    entry = _read_entry(entry_path)
    if entry is None:
        return {
            "ok": True,
            "schema_version": CORRECTION_LEDGER_RECEIPT_SCHEMA_VERSION,
            "status": "missing",
            "peer_id": peer_id,
            "correction_id": correction_id,
            "entry": None,
            "storage": {"path": str(entry_path)},
        }
    return {
        "ok": True,
        "schema_version": CORRECTION_LEDGER_RECEIPT_SCHEMA_VERSION,
        "status": "found",
        "peer_id": peer_id,
        "correction_id": correction_id,
        "entry": entry,
        "storage": {"path": str(entry_path)},
    }


def build_correction_consumer_context(
    *,
    peer_id: str,
    include_statuses: list[str] | tuple[str, ...] | set[str] | None = None,
    limit: int = MAX_ENTRIES_RETURNED,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compile ledger entries into an explicit downstream-consumer context.

    This is the read side for C'2/C'3. It does not apply corrections to a DB,
    run RPP, or regenerate Neural Imprint. Consumers must opt in and use the
    returned fact/profile overlays deliberately.
    """

    requested_statuses = _include_statuses(include_statuses)
    indexed = list_correction_entries(
        peer_id=peer_id,
        limit=min(max(1, int(limit)), MAX_ENTRIES_RETURNED),
        root=root,
    )
    all_entries = indexed.get("entries")
    if not isinstance(all_entries, list):
        all_entries = []

    eligible_entries = [
        entry
        for entry in all_entries
        if _text(entry.get("status")) in requested_statuses
    ]
    eligible_entries.sort(
        key=lambda item: (
            int(item.get("received_at_ms") or 0),
            _text(item.get("correction_id")),
        )
    )

    fact_corrections = [
        _fact_correction_effect(entry)
        for entry in eligible_entries
        if entry.get("correction_type") == "fact_correction"
        and _object(entry.get("effects")).get("affects_fact_table") is True
    ]
    profile_overlays = [
        _profile_overlay_effect(entry)
        for entry in eligible_entries
        if entry.get("correction_type") == "profile_correction"
        and _object(entry.get("effects")).get("affects_profile_overlay") is True
    ]
    eval_feedback_count = sum(
        1 for entry in eligible_entries if entry.get("correction_type") == "eval_feedback"
    )
    requires_rpp_rerun = any(
        _object(entry.get("effects")).get("requires_rpp_rerun") is True
        for entry in eligible_entries
    )
    requires_neural_imprint_regen = any(
        _object(entry.get("effects")).get("requires_neural_imprint_regen") is True
        for entry in eligible_entries
    )

    return {
        "ok": True,
        "schema_version": CORRECTION_CONSUMER_CONTEXT_SCHEMA_VERSION,
        "status": "found" if eligible_entries else "empty",
        "peer_id": peer_id,
        "include_statuses": sorted(requested_statuses),
        "counts": {
            "total_entries_scanned": len(all_entries),
            "eligible_entries": len(eligible_entries),
            "fact_corrections": len(fact_corrections),
            "profile_overlays": len(profile_overlays),
            "eval_feedback": eval_feedback_count,
        },
        "flags": {
            "has_fact_corrections": bool(fact_corrections),
            "has_profile_overlays": bool(profile_overlays),
            "requires_rpp_rerun": requires_rpp_rerun,
            "requires_neural_imprint_regen": requires_neural_imprint_regen,
        },
        "fact_corrections": fact_corrections,
        "profile_overlays": profile_overlays,
        "rpp_context": {
            "fact_correction_fingerprints": [
                item["correction_fingerprint"] for item in fact_corrections
            ],
            "profile_overlay_fingerprints": [
                item["correction_fingerprint"] for item in profile_overlays
            ],
            "corrected_fact_count": len(fact_corrections),
            "profile_overlay_count": len(profile_overlays),
        },
        "privacy": {
            "local_only": True,
            "do_not_include_in_ai_mailbox": True,
            "may_contain_user_data": bool(fact_corrections or profile_overlays),
        },
        "audit": {
            "writes_runtime_artifacts": False,
            "triggers_rpp": False,
            "triggers_neural_imprint_regen": False,
            "consumer_must_apply_effects_explicitly": True,
            "source": "correction_ledger",
        },
    }


def _index_response(
    *,
    root: Path,
    entries: list[dict[str, Any]],
    peer_id: str | None,
    correction_type: str | None,
    status: str | None,
    limit: int,
) -> dict[str, Any]:
    return {
        "ok": True,
        "schema_version": CORRECTION_LEDGER_INDEX_SCHEMA_VERSION,
        "status": "found",
        "root": str(root),
        "peer_id": peer_id,
        "correction_type": correction_type,
        "entry_status": status,
        "limit": limit,
        "count": len(entries),
        "entries": entries,
        "audit": {
            "writes_runtime_artifacts": False,
            "triggers_rpp": False,
            "triggers_neural_imprint_regen": False,
        },
    }


def _validate_by_type(
    *,
    correction_type: str,
    target: dict[str, Any],
    correction: dict[str, Any],
) -> None:
    if correction_type == "eval_feedback":
        rating = _text(correction.get("rating"))
        if rating not in {"positive", "negative", "neutral"}:
            raise CorrectionLedgerError(
                "invalid_eval_feedback",
                "eval_feedback.correction.rating must be positive, negative, or neutral",
                {},
            )
        return
    if correction_type == "fact_correction":
        if not _text(target.get("fact_id")):
            raise CorrectionLedgerError(
                "invalid_fact_correction",
                "fact_correction.target.fact_id is required",
                {},
            )
        if not correction:
            raise CorrectionLedgerError(
                "invalid_fact_correction",
                "fact_correction.correction must include corrected structured fields",
                {},
            )
        return
    if correction_type == "profile_correction":
        if not (_text(target.get("direction_id")) or _text(target.get("profile_field"))):
            raise CorrectionLedgerError(
                "invalid_profile_correction",
                "profile_correction.target.direction_id or target.profile_field is required",
                {},
            )
        if not correction:
            raise CorrectionLedgerError(
                "invalid_profile_correction",
                "profile_correction.correction must include structured correction fields",
                {},
            )


def _effects(correction_type: str, value: Any) -> dict[str, bool]:
    raw = _object(value)
    defaults = {
        "affects_eval": correction_type == "eval_feedback",
        "affects_fact_table": correction_type == "fact_correction",
        "affects_profile_overlay": correction_type == "profile_correction",
        "requires_rpp_rerun": correction_type in {"fact_correction", "profile_correction"},
        "requires_neural_imprint_regen": correction_type in {"fact_correction", "profile_correction"},
    }
    for key, item in raw.items():
        if key in ALLOWED_EFFECT_KEYS and isinstance(item, bool):
            defaults[key] = item
    return defaults


def _contains_raw_user_text(*objects: dict[str, Any]) -> bool:
    text_keys = {
        "text",
        "note",
        "comment",
        "prompt",
        "user_input",
        "source_input_text",
        "correction_text",
        "natural_language_correction",
    }
    for obj in objects:
        for key, value in obj.items():
            if key in text_keys and isinstance(value, str) and value.strip():
                return True
    return False


def _include_statuses(value: list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if value is None:
        return {"recorded", "applied"}
    statuses = {_text(item) for item in value if _text(item)}
    unsupported = sorted(status for status in statuses if status not in ALLOWED_STATUSES)
    if unsupported:
        raise CorrectionLedgerError(
            "unsupported_status",
            f"unsupported include_status: {unsupported[0]}",
            {"allowed": sorted(ALLOWED_STATUSES), "unsupported": unsupported},
        )
    return statuses or {"recorded", "applied"}


def _fact_correction_effect(entry: dict[str, Any]) -> dict[str, Any]:
    target = _object(entry.get("target"))
    correction = _object(entry.get("correction"))
    return {
        "correction_id": _text(entry.get("correction_id")),
        "correction_fingerprint": _text(entry.get("correction_fingerprint")),
        "received_at_ms": int(entry.get("received_at_ms") or 0),
        "status": _text(entry.get("status")),
        "target": target,
        "correction": correction,
        "normalized_fields": _normalized_fact_fields(target, correction),
        "requires_rpp_rerun": _object(entry.get("effects")).get("requires_rpp_rerun")
        is True,
        "requires_neural_imprint_regen": _object(entry.get("effects")).get(
            "requires_neural_imprint_regen"
        )
        is True,
    }


def _profile_overlay_effect(entry: dict[str, Any]) -> dict[str, Any]:
    correction = _object(entry.get("correction"))
    overlay = _object(correction.get("profile_overlay")) or correction
    return {
        "correction_id": _text(entry.get("correction_id")),
        "correction_fingerprint": _text(entry.get("correction_fingerprint")),
        "received_at_ms": int(entry.get("received_at_ms") or 0),
        "status": _text(entry.get("status")),
        "target": _object(entry.get("target")),
        "overlay": overlay,
        "correction": correction,
        "requires_rpp_rerun": _object(entry.get("effects")).get("requires_rpp_rerun")
        is True,
        "requires_neural_imprint_regen": _object(entry.get("effects")).get(
            "requires_neural_imprint_regen"
        )
        is True,
    }


def _normalized_fact_fields(
    target: dict[str, Any],
    correction: dict[str, Any],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    target_field = _text(target.get("field"))
    if target_field and "new_value" in correction:
        fields[target_field] = correction.get("new_value")
    for key in ("fields", "corrected_fields", "updates"):
        nested = correction.get(key)
        if isinstance(nested, dict):
            fields.update(nested)
    return fields


def _read_entry(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_component(value: Any, field: str) -> str:
    text = _required_text(value, field)
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise CorrectionLedgerError(
            "invalid_path_component",
            f"{field} must not contain path separators",
            {"value": text},
        )
    return text


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise CorrectionLedgerError(
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


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
