# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Validation for restore/full-cache harness receipts.

The receipt is evidence metadata only. It must not carry user text, trigger
runtime restore, or convert a receipt-only smoke into a real-device claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


RESTORE_FULL_CACHE_HARNESS_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.restore_full_cache.harness_receipt.v1"
)

ALLOWED_MODES = {"receipt_only", "real_device_restore"}
ALLOWED_STATUSES = {"passed", "failed", "blocked", "not_run"}
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "run_id",
    "mode",
    "status",
    "surface",
    "device",
    "model",
    "artifact",
    "evidence",
    "privacy",
    "claims",
)
HASH_RE = re.compile(r"^(sha256:)?[0-9a-f]{64}$", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
RAW_TEXT_KEYS = {
    "answer",
    "base_answer",
    "correction_text",
    "generated_text",
    "messages",
    "prompt",
    "question",
    "raw_text",
    "response",
    "sample_text",
    "transcript",
    "user_text",
}
REAL_DEVICE_REQUIRED_ANCHOR_EVENTS = {
    "neural_imprint_restore_configured",
    "cmlx_neural_imprint_restore",
    "first_generation_after_restore",
}
RESTORE_PASSED_OUTCOMES = {"captured_and_restored", "restored"}
RESTORE_FAILED_OUTCOMES = {"halo_capsule_restore_failed", "restore_failed"}
RESTORE_STATUS_FILE_NAMES = (
    "neural_imprint_restore_status.json",
)


@dataclass(frozen=True)
class RestoreFullCacheReceiptIssue:
    level: str
    code: str
    message: str
    field: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True)
class RestoreFullCacheReceiptValidation:
    ok: bool
    errors: list[RestoreFullCacheReceiptIssue]
    warnings: list[RestoreFullCacheReceiptIssue]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }


def restore_full_cache_harness_receipt_schema() -> dict[str, Any]:
    """Return the minimal schema shape expected from receipt producers."""

    return {
        "schema_version": RESTORE_FULL_CACHE_HARNESS_RECEIPT_SCHEMA_VERSION,
        "modes": sorted(ALLOWED_MODES),
        "statuses": sorted(ALLOWED_STATUSES),
        "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
        "real_device_required_anchor_events": sorted(REAL_DEVICE_REQUIRED_ANCHOR_EVENTS),
        "claim_gate": {
            "receipt_only_must_not_claim_real_runtime_restore": True,
            "real_device_restore_requires_eval_log_anchors": True,
            "restore_full_cache_shipped_must_remain_false": True,
            "model_quality_improved_must_remain_false": True,
        },
    }


def build_restore_full_cache_harness_receipt_from_files(
    *,
    run_id: str,
    surface_kind: str,
    app_id: str,
    restore_status_path: Path,
    mode: str = "receipt_only",
    eval_log_path: Path | None = None,
    device_name: str | None = None,
    device_os_build: str | None = None,
    device_fingerprint: str | None = None,
    commands: Iterable[str] = (),
    claim_real_runtime_restore: bool = False,
    no_visible_prompt_fallback: bool = False,
    no_keyword_rule_fallback: bool = False,
    no_prompt_stuffing: bool = False,
    normal_app_runtime_path: bool = False,
) -> dict[str, Any]:
    """Build a harness receipt from pulled app Documents artifacts.

    This is a collector only: it reads restore status / metadata / optional
    eval.log files and emits evidence metadata. It does not launch an app,
    execute restore, or infer product readiness.
    """

    status_path = restore_status_path.expanduser()
    status = _read_json_object(status_path)
    status_value = _status_from_restore_outcome(_text(status.get("outcome")))
    metadata_path = _restore_metadata_path(status_path, status, required=status_value == "passed")
    metadata = _read_json_object(metadata_path) if metadata_path is not None else {}
    eval_path = eval_log_path.expanduser() if eval_log_path is not None else None
    eval_log_text = eval_path.read_text(encoding="utf-8") if eval_path is not None else None
    eval_log_anchors = _eval_log_anchors(eval_log_text) if eval_log_text is not None else []
    device_payload = _device_payload(
        name=device_name,
        os_build=device_os_build,
        fingerprint=device_fingerprint,
    )
    privacy_payload = {
        "raw_text_included": False,
        "external_network_used": False,
        "no_visible_prompt_fallback": bool(no_visible_prompt_fallback),
        "no_keyword_rule_fallback": bool(no_keyword_rule_fallback),
        "no_prompt_stuffing": bool(no_prompt_stuffing),
        "normal_app_runtime_path": bool(normal_app_runtime_path),
    }
    real_runtime_restore_validated = _real_runtime_restore_claim_allowed(
        requested=claim_real_runtime_restore,
        mode=mode,
        status=status_value,
        device=device_payload,
        eval_log_path=eval_path,
        eval_log_anchors=eval_log_anchors,
        privacy=privacy_payload,
    )

    receipt = {
        "schema_version": RESTORE_FULL_CACHE_HARNESS_RECEIPT_SCHEMA_VERSION,
        "run_id": _required_non_empty_text(run_id, "run_id"),
        "mode": _required_non_empty_text(mode, "mode"),
        "status": status_value,
        "surface": {
            "kind": _required_non_empty_text(surface_kind, "surface_kind"),
            "app_id": _required_non_empty_text(app_id, "app_id"),
        },
        "device": device_payload,
        "model": {
            "model_id": _model_id(status, metadata),
            "model_fingerprint": _model_fingerprint(status, metadata),
        },
        "artifact": {
            "artifact_sha256": _optional_sha256(status.get("artifact_sha256")),
            "metadata_sha256": _sha256_file(metadata_path) if metadata_path is not None else None,
            "prefix_token_count": _prefix_token_count(
                status,
                metadata,
                required=status_value == "passed",
            ),
            "cache_backend": _text(status.get("cache_backend")),
            "cache_backend_version": _text(status.get("cache_backend_version")),
            "restore_outcome": _text(status.get("outcome")),
        },
        "evidence": {
            "commands": [str(command) for command in commands],
            "restore_status_path": str(status_path),
            "restore_status_sha256": _sha256_file(status_path),
            "metadata_path": str(metadata_path) if metadata_path is not None else None,
            "eval_log_path": str(eval_path) if eval_path is not None else None,
            "eval_log_sha256": _sha256_file(eval_path) if eval_path is not None else None,
            "eval_log_anchors": eval_log_anchors,
        },
        "privacy": privacy_payload,
        "claims": {
            "real_runtime_restore_validated": real_runtime_restore_validated,
            "restore_full_cache_shipped": False,
            "model_quality_improved": False,
        },
    }
    return receipt


def validate_restore_full_cache_harness_receipt(
    receipt: Mapping[str, Any],
) -> RestoreFullCacheReceiptValidation:
    """Validate a restore/full-cache harness receipt without executing restore."""

    errors: list[RestoreFullCacheReceiptIssue] = []
    warnings: list[RestoreFullCacheReceiptIssue] = []

    if not isinstance(receipt, Mapping):
        return RestoreFullCacheReceiptValidation(
            ok=False,
            errors=[
                RestoreFullCacheReceiptIssue(
                    "error",
                    "invalid_receipt",
                    "receipt root must be an object",
                )
            ],
            warnings=[],
        )

    schema = receipt.get("schema_version")
    if schema != RESTORE_FULL_CACHE_HARNESS_RECEIPT_SCHEMA_VERSION:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "schema_version_mismatch",
                f"schema_version must be {RESTORE_FULL_CACHE_HARNESS_RECEIPT_SCHEMA_VERSION}",
                "schema_version",
            )
        )

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in receipt:
            errors.append(
                RestoreFullCacheReceiptIssue(
                    "error",
                    "missing_required_field",
                    f"missing required field: {field}",
                    field,
                )
            )

    mode = _text(receipt.get("mode"))
    status = _text(receipt.get("status"))
    if mode not in ALLOWED_MODES:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "invalid_mode",
                f"mode must be one of {sorted(ALLOWED_MODES)}",
                "mode",
            )
        )
    if status not in ALLOWED_STATUSES:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "invalid_status",
                f"status must be one of {sorted(ALLOWED_STATUSES)}",
                "status",
            )
        )

    surface = _mapping(receipt.get("surface"))
    device = _mapping(receipt.get("device"))
    model = _mapping(receipt.get("model"))
    artifact = _mapping(receipt.get("artifact"))
    evidence = _mapping(receipt.get("evidence"))
    privacy = _mapping(receipt.get("privacy"))
    claims = _mapping(receipt.get("claims"))

    _require_text(surface, "kind", "surface.kind", errors)
    _require_text(surface, "app_id", "surface.app_id", errors)
    _require_text(model, "model_id", "model.model_id", errors)
    _require_hash(model, "model_fingerprint", "model.model_fingerprint", errors)
    artifact_required = status == "passed"
    if artifact_required:
        _require_hash(artifact, "artifact_sha256", "artifact.artifact_sha256", errors)
        _require_hash(artifact, "metadata_sha256", "artifact.metadata_sha256", errors)
        _require_non_negative_int(
            artifact,
            "prefix_token_count",
            "artifact.prefix_token_count",
            errors,
        )
    else:
        _optional_hash(artifact, "artifact_sha256", "artifact.artifact_sha256", errors)
        _optional_hash(artifact, "metadata_sha256", "artifact.metadata_sha256", errors)
        _optional_non_negative_int(
            artifact,
            "prefix_token_count",
            "artifact.prefix_token_count",
            errors,
        )

    if privacy.get("raw_text_included") is not False:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "raw_text_included",
                "privacy.raw_text_included must be false",
                "privacy.raw_text_included",
            )
        )
    if privacy.get("external_network_used") is not False:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "external_network_used",
                "privacy.external_network_used must be false",
                "privacy.external_network_used",
            )
        )

    for field in _raw_text_fields(receipt):
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "raw_text_field_present",
                "receipt contains a raw-text field",
                field,
            )
        )
    for field, url in _external_urls(receipt):
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "external_url_present",
                f"receipt contains a non-localhost URL: {url}",
                field,
            )
        )

    if claims.get("restore_full_cache_shipped") is not False:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "restore_full_cache_shipped_claim",
                "restore_full_cache_shipped must remain false in harness receipts",
                "claims.restore_full_cache_shipped",
            )
        )
    if claims.get("model_quality_improved") is not False:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "model_quality_improved_claim",
                "model_quality_improved must remain false in restore receipts",
                "claims.model_quality_improved",
            )
        )

    real_claim = claims.get("real_runtime_restore_validated") is True
    if mode == "receipt_only" and real_claim:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "receipt_only_real_runtime_claim",
                "receipt_only mode cannot claim real runtime restore validation",
                "claims.real_runtime_restore_validated",
            )
        )
    if mode == "real_device_restore" and status == "passed":
        _validate_real_device_evidence(device, evidence, privacy, claims, errors, warnings)
    elif real_claim:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "real_runtime_claim_without_passed_device_run",
                "real_runtime_restore_validated requires mode=real_device_restore and status=passed",
                "claims.real_runtime_restore_validated",
            )
        )

    return RestoreFullCacheReceiptValidation(
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )


def find_restore_status_file(documents_dir: Path) -> Path:
    """Return the first known restore status file in a pulled Documents dir."""

    root = documents_dir.expanduser()
    for name in RESTORE_STATUS_FILE_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"missing restore status file under {root}; expected one of {RESTORE_STATUS_FILE_NAMES}"
    )


def _validate_real_device_evidence(
    device: Mapping[str, Any],
    evidence: Mapping[str, Any],
    privacy: Mapping[str, Any],
    claims: Mapping[str, Any],
    errors: list[RestoreFullCacheReceiptIssue],
    warnings: list[RestoreFullCacheReceiptIssue],
) -> None:
    _require_text(device, "name", "device.name", errors)
    _require_text(device, "os_build", "device.os_build", errors)
    _require_text(device, "device_fingerprint", "device.device_fingerprint", errors)
    _require_text(evidence, "eval_log_path", "evidence.eval_log_path", errors)
    _require_hash(evidence, "eval_log_sha256", "evidence.eval_log_sha256", errors)

    anchors = evidence.get("eval_log_anchors")
    if not isinstance(anchors, list) or not anchors:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "missing_eval_log_anchors",
                "real-device restore requires eval_log_anchors",
                "evidence.eval_log_anchors",
            )
        )
        observed_events: set[str] = set()
    else:
        observed_events = set()
        for index, anchor in enumerate(anchors):
            anchor_map = _mapping(anchor)
            event = _text(anchor_map.get("event"))
            if not event:
                errors.append(
                    RestoreFullCacheReceiptIssue(
                        "error",
                        "invalid_eval_log_anchor",
                        "eval_log anchor requires an event",
                        f"evidence.eval_log_anchors[{index}].event",
                    )
                )
                continue
            observed_events.add(event)
            _require_non_negative_int(
                anchor_map,
                "line",
                f"evidence.eval_log_anchors[{index}].line",
                errors,
            )
            _require_hash(
                anchor_map,
                "line_sha256",
                f"evidence.eval_log_anchors[{index}].line_sha256",
                errors,
            )
    missing = sorted(REAL_DEVICE_REQUIRED_ANCHOR_EVENTS - observed_events)
    if missing:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "missing_required_eval_log_anchor_events",
                f"missing required eval.log anchor events: {missing}",
                "evidence.eval_log_anchors",
            )
        )

    required_safety_flags = (
        "no_visible_prompt_fallback",
        "no_keyword_rule_fallback",
        "no_prompt_stuffing",
        "normal_app_runtime_path",
    )
    for key in required_safety_flags:
        if privacy.get(key) is not True:
            errors.append(
                RestoreFullCacheReceiptIssue(
                    "error",
                    "missing_real_device_safety_flag",
                    f"privacy.{key} must be true for real-device restore",
                    f"privacy.{key}",
                )
            )
    if claims.get("real_runtime_restore_validated") is not True:
        warnings.append(
            RestoreFullCacheReceiptIssue(
                "warning",
                "passed_real_device_without_validation_claim",
                "real-device run passed but does not claim validation",
                "claims.real_runtime_restore_validated",
            )
        )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing JSON file: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _resolve_status_reference(status_path: Path, reference: str) -> Path:
    raw = Path(reference).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([
        status_path.parent / raw,
        status_path.parent.parent / raw,
    ])
    parts = raw.parts
    if parts and parts[0] == "Documents":
        stripped = Path(*parts[1:])
        candidates.extend([
            status_path.parent / stripped,
            status_path.parent.parent / stripped,
        ])
    candidates.append(status_path.parent / raw.name)

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.is_file():
            return normalized
    raise FileNotFoundError(
        f"could not resolve restore status reference {reference!r} from {status_path}"
    )


def _restore_metadata_path(
    status_path: Path,
    status: Mapping[str, Any],
    *,
    required: bool,
) -> Path | None:
    reference = _text(status.get("metadata"))
    if not reference:
        if required:
            raise ValueError("restore status missing required text field: metadata")
        return None
    return _resolve_status_reference(status_path, reference)


def _required_non_empty_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _status_from_restore_outcome(outcome: str) -> str:
    if outcome in RESTORE_PASSED_OUTCOMES:
        return "passed"
    if outcome in RESTORE_FAILED_OUTCOMES or "failed" in outcome:
        return "failed"
    return "blocked"


def _device_payload(
    *,
    name: str | None,
    os_build: str | None,
    fingerprint: str | None,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    if _text(name):
        payload["name"] = _text(name)
    if _text(os_build):
        payload["os_build"] = _text(os_build)
    if _text(fingerprint):
        payload["device_fingerprint"] = _text(fingerprint)
    return payload


def _real_runtime_restore_claim_allowed(
    *,
    requested: bool,
    mode: str,
    status: str,
    device: Mapping[str, Any],
    eval_log_path: Path | None,
    eval_log_anchors: Iterable[Mapping[str, Any]],
    privacy: Mapping[str, Any],
) -> bool:
    if not requested or mode != "real_device_restore" or status != "passed":
        return False
    if eval_log_path is None:
        return False
    if not all(_text(device.get(key)) for key in ("name", "os_build", "device_fingerprint")):
        return False
    observed_events = {
        _text(anchor.get("event"))
        for anchor in eval_log_anchors
        if isinstance(anchor, Mapping)
    }
    if not REAL_DEVICE_REQUIRED_ANCHOR_EVENTS.issubset(observed_events):
        return False
    return all(
        privacy.get(key) is True
        for key in (
            "no_visible_prompt_fallback",
            "no_keyword_rule_fallback",
            "no_prompt_stuffing",
            "normal_app_runtime_path",
        )
    )


def _model_id(status: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    metadata_model = metadata.get("model") if isinstance(metadata.get("model"), Mapping) else {}
    return (
        _text(status.get("model_id"))
        or _text(status.get("scaffold_model_id"))
        or _text(metadata_model.get("id"))
    )


def _model_fingerprint(status: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    metadata_model = metadata.get("model") if isinstance(metadata.get("model"), Mapping) else {}
    if metadata_model:
        return _sha256_json(metadata_model)
    return _sha256_json({"model_id": _model_id(status, metadata)})


def _prefix_token_count(
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    required: bool,
) -> int | None:
    value = _prefix_token_count_value(status, metadata)
    if value is None and required:
        raise ValueError("restore status missing prefix_token_count")
    return value


def _prefix_token_count_value(status: Mapping[str, Any], metadata: Mapping[str, Any]) -> int | None:
    value = status.get("prefix_token_count")
    if isinstance(value, int) and value >= 0:
        return value
    prefix = metadata.get("prefix") if isinstance(metadata.get("prefix"), Mapping) else {}
    value = prefix.get("token_count")
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _eval_log_anchors(eval_log_text: str) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    observed: set[str] = set()
    for line_number, line in enumerate(eval_log_text.splitlines(), start=1):
        for event in sorted(REAL_DEVICE_REQUIRED_ANCHOR_EVENTS):
            if event in observed or event not in line:
                continue
            anchors.append({
                "event": event,
                "line": line_number,
                "line_sha256": _sha256_text(line),
            })
            observed.add(event)
    return anchors


def _normalize_sha256(value: str) -> str:
    text = value.strip()
    if text.startswith("sha256:"):
        return text
    return f"sha256:{text}"


def _optional_sha256(value: Any) -> str | None:
    text = _text(value)
    return _normalize_sha256(text) if text else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(payload)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _require_text(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[RestoreFullCacheReceiptIssue],
) -> None:
    if not _text(mapping.get(key)):
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "missing_text_field",
                f"{field} must be a non-empty string",
                field,
            )
        )


def _require_hash(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[RestoreFullCacheReceiptIssue],
) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not HASH_RE.match(value.strip()):
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "invalid_sha256",
                f"{field} must be a sha256 hash",
                field,
            )
        )


def _optional_hash(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[RestoreFullCacheReceiptIssue],
) -> None:
    value = mapping.get(key)
    if value is None:
        return
    _require_hash(mapping, key, field, errors)


def _require_non_negative_int(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[RestoreFullCacheReceiptIssue],
) -> None:
    value = mapping.get(key)
    if not isinstance(value, int) or value < 0:
        errors.append(
            RestoreFullCacheReceiptIssue(
                "error",
                "invalid_non_negative_int",
                f"{field} must be a non-negative integer",
                field,
            )
        )


def _optional_non_negative_int(
    mapping: Mapping[str, Any],
    key: str,
    field: str,
    errors: list[RestoreFullCacheReceiptIssue],
) -> None:
    value = mapping.get(key)
    if value is None:
        return
    _require_non_negative_int(mapping, key, field, errors)


def _raw_text_fields(payload: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            field = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in RAW_TEXT_KEYS:
                yield field
            yield from _raw_text_fields(value, field)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            field = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _raw_text_fields(value, field)


def _external_urls(payload: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            yield from _external_urls(value, field)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            field = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _external_urls(value, field)
    elif isinstance(payload, str):
        for match in URL_RE.finditer(payload):
            url = match.group(0)
            parsed = urlparse(url)
            if parsed.scheme in {"http", "https"} and not _is_localhost(parsed.hostname):
                yield prefix or "<root>", url


def _is_localhost(hostname: str | None) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1"}
