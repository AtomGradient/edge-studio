# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Demo receipt inspection and local-only validation for the Edge CLI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from backend.cli.fingerprints import pretty_json
from backend.services.app_dirs import data_path


DEMO_RECEIPT_SCHEMA_VERSION = "edge.demo.receipt.v1"
DEMO_RECEIPT_INSPECT_SCHEMA_VERSION = "edge.demo.receipt.inspect.v1"
DEMO_LOCAL_ONLY_SCHEMA_VERSION = "edge.demo.local_only.report.v1"

REQUIRED_FIELDS = (
    "schema_version",
    "run_id",
    "model_path",
    "model_sha256",
    "sample_id",
    "sample_sha256",
    "artifact_id",
    "artifact_sha256",
    "metadata_sha256",
    "raw_text_included",
    "network_used_during_demo",
    "status",
)
HASH_FIELDS = (
    "model_sha256",
    "sample_sha256",
    "artifact_sha256",
    "metadata_sha256",
    "base_answer_sha256",
    "personalized_answer_sha256",
)
RAW_TEXT_KEYS = {
    "answer",
    "base_answer",
    "personalized_answer",
    "before_answer",
    "after_answer",
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
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationIssue:
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
        if self.field:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True)
class DemoReceiptValidation:
    ok: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }


@dataclass(frozen=True)
class DemoReceiptResult:
    ok: bool
    exit_code: int
    receipt_path: Path | None
    receipt: dict[str, Any] | None
    validation: DemoReceiptValidation
    error: str | None = None

    def as_dict(self, *, schema_version: str) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": schema_version,
            "ok": self.ok,
            "receipt_path": str(self.receipt_path) if self.receipt_path else None,
            "validation": self.validation.as_dict(),
        }
        if self.receipt is not None:
            payload["receipt"] = self.receipt
        if self.error:
            payload["error"] = self.error
        return payload

    def to_json(self, *, schema_version: str) -> str:
        return json.dumps(self.as_dict(schema_version=schema_version), ensure_ascii=False, indent=2)


def default_demo_receipt_path(run_id: str) -> Path:
    return data_path("demo_runs", run_id, "receipt.json")


def resolve_demo_receipt_path(*, run_id: str | None = None, path: Path | None = None) -> Path:
    if path is not None:
        return path.expanduser()
    if run_id:
        return default_demo_receipt_path(run_id)
    raise ValueError("run_id or path is required")


def inspect_demo_receipt(*, run_id: str | None = None, path: Path | None = None) -> DemoReceiptResult:
    receipt_path = resolve_demo_receipt_path(run_id=run_id, path=path)
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        validation = DemoReceiptValidation(
            ok=False,
            errors=[ValidationIssue("error", "receipt_not_found", "Receipt file does not exist.", "receipt_path")],
            warnings=[],
        )
        return DemoReceiptResult(False, 1, receipt_path, None, validation, error="receipt_not_found")
    except json.JSONDecodeError as exc:
        validation = DemoReceiptValidation(
            ok=False,
            errors=[ValidationIssue("error", "invalid_json", f"Receipt is not valid JSON: {exc.msg}.", "receipt_path")],
            warnings=[],
        )
        return DemoReceiptResult(False, 1, receipt_path, None, validation, error="invalid_json")
    if not isinstance(payload, dict):
        validation = DemoReceiptValidation(
            ok=False,
            errors=[ValidationIssue("error", "invalid_receipt", "Receipt root must be a JSON object.")],
            warnings=[],
        )
        return DemoReceiptResult(False, 1, receipt_path, None, validation, error="invalid_receipt")
    validation = validate_demo_receipt(payload)
    return DemoReceiptResult(validation.ok, 0 if validation.ok else 1, receipt_path, payload, validation)


def validate_demo_receipt(receipt: Mapping[str, Any]) -> DemoReceiptValidation:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    schema = receipt.get("schema_version")
    if schema != DEMO_RECEIPT_SCHEMA_VERSION:
        errors.append(
            ValidationIssue(
                "error",
                "schema_version_mismatch",
                f"Expected schema_version {DEMO_RECEIPT_SCHEMA_VERSION}.",
                "schema_version",
            )
        )

    for field in REQUIRED_FIELDS:
        if field not in receipt:
            errors.append(ValidationIssue("error", "missing_required_field", f"Missing required field: {field}.", field))

    status = receipt.get("status")
    if not isinstance(status, str) or not status.strip():
        errors.append(ValidationIssue("error", "invalid_status", "status must be a non-empty string.", "status"))

    for field in HASH_FIELDS:
        value = receipt.get(field)
        if value is not None and not _is_sha256(value):
            errors.append(ValidationIssue("error", "invalid_sha256", f"{field} must be sha256:<64 hex>.", field))

    include_text_allowed = _include_text_allowed(receipt)
    raw_text_included = receipt.get("raw_text_included")
    if raw_text_included is not False:
        if include_text_allowed:
            warnings.append(
                ValidationIssue(
                    "warning",
                    "raw_text_explicitly_included",
                    "Receipt marks raw text as included under an explicit include-text marker.",
                    "raw_text_included",
                )
            )
        else:
            errors.append(
                ValidationIssue(
                    "error",
                    "raw_text_included",
                    "raw_text_included must be false unless an explicit include-text marker is present.",
                    "raw_text_included",
                )
            )

    if not include_text_allowed:
        for field in _raw_text_fields(receipt):
            errors.append(
                ValidationIssue(
                    "error",
                    "raw_text_field_present",
                    "Receipt contains a raw text field without an explicit include-text marker.",
                    field,
                )
            )

    external_urls = list(_external_urls(receipt))
    for field, url in external_urls:
        errors.append(
            ValidationIssue(
                "error",
                "external_url_present",
                f"Receipt contains a non-localhost URL: {url}.",
                field,
            )
        )

    network_used = receipt.get("network_used_during_demo")
    if network_used is not False:
        if network_used is True and not external_urls:
            warnings.append(
                ValidationIssue(
                    "warning",
                    "network_used_during_demo",
                    "network_used_during_demo is true; verify this run did not send user data externally.",
                    "network_used_during_demo",
                )
            )
        else:
            errors.append(
                ValidationIssue(
                    "error",
                    "invalid_network_flag",
                    "network_used_during_demo must be false or a boolean true warning state.",
                    "network_used_during_demo",
                )
            )

    for field, value in _path_like_values(receipt):
        if "://" in value and not value.startswith("file://"):
            parsed = urlparse(value)
            if parsed.scheme in {"http", "https"} and _is_localhost(parsed.hostname):
                continue
            errors.append(
                ValidationIssue(
                    "error",
                    "non_local_path",
                    f"Path-like field must refer to a local path, not {value}.",
                    field,
                )
            )

    return DemoReceiptValidation(ok=not errors, errors=errors, warnings=warnings)


def write_demo_receipt(receipt: Mapping[str, Any], *, run_id: str | None = None, path: Path | None = None) -> Path:
    output_path = resolve_demo_receipt_path(run_id=run_id or str(receipt.get("run_id") or ""), path=path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pretty_json(dict(receipt)), encoding="utf-8")
    return output_path


def demo_receipt_schema() -> dict[str, object]:
    return {
        "schema_version": DEMO_RECEIPT_SCHEMA_VERSION,
        "required_fields": list(REQUIRED_FIELDS),
        "hash_fields": list(HASH_FIELDS),
        "raw_text_default": "raw_text_included must be false unless include_text/include_text_acknowledged is true",
        "network_default": "network_used_during_demo should be false; true is a warning when no external URL is present",
    }


def format_demo_receipt(result: DemoReceiptResult) -> str:
    lines = [
        f"Edge demo receipt ({DEMO_RECEIPT_INSPECT_SCHEMA_VERSION})",
        f"status: {'ok' if result.ok else 'fail'}",
    ]
    if result.receipt_path:
        lines.append(f"receipt: {result.receipt_path}")
    if result.receipt:
        lines.append(f"run: {result.receipt.get('run_id')}")
        lines.append(f"receipt schema: {result.receipt.get('schema_version')}")
    _append_validation_lines(lines, result.validation)
    return "\n".join(lines)


def format_local_only(result: DemoReceiptResult) -> str:
    lines = [
        f"Edge demo local-only ({DEMO_LOCAL_ONLY_SCHEMA_VERSION})",
        f"overall: {'ok' if result.ok else 'fail'}",
    ]
    if result.receipt_path:
        lines.append(f"receipt: {result.receipt_path}")
    if result.receipt:
        lines.append(f"run: {result.receipt.get('run_id')}")
        lines.append(f"demo status: {result.receipt.get('status')}")
    _append_validation_lines(lines, result.validation)
    return "\n".join(lines)


def format_demo_schema() -> str:
    schema = demo_receipt_schema()
    return "\n".join(
        [
            f"Edge demo receipt schema ({schema['schema_version']})",
            "required fields:",
            *[f"- {field}" for field in schema["required_fields"]],
        ]
    )


def _append_validation_lines(lines: list[str], validation: DemoReceiptValidation) -> None:
    lines.append(f"validation: {'ok' if validation.ok else 'fail'}")
    for issue in validation.errors:
        suffix = f" ({issue.field})" if issue.field else ""
        lines.append(f"error: {issue.code}{suffix}: {issue.message}")
    for issue in validation.warnings:
        suffix = f" ({issue.field})" if issue.field else ""
        lines.append(f"warning: {issue.code}{suffix}: {issue.message}")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_RE.match(value.strip()) is not None


def _include_text_allowed(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("include_text") is True or receipt.get("include_text_acknowledged") is True


def _raw_text_fields(value: Any, *, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in RAW_TEXT_KEYS and child not in (None, "", [], {}):
                yield field
            yield from _raw_text_fields(child, prefix=field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _raw_text_fields(child, prefix=f"{prefix}[{index}]")


def _external_urls(value: Any, *, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            yield from _external_urls(child, prefix=field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _external_urls(child, prefix=f"{prefix}[{index}]")
    elif isinstance(value, str):
        for match in URL_RE.finditer(value):
            url = match.group(0)
            parsed = urlparse(url)
            if not _is_localhost(parsed.hostname):
                yield prefix, url


def _path_like_values(value: Any, *, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, str) and (field.endswith("_path") or field.endswith(".path") or field == "path"):
                yield field, child
            else:
                yield from _path_like_values(child, prefix=field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _path_like_values(child, prefix=f"{prefix}[{index}]")


def _is_localhost(hostname: str | None) -> bool:
    if hostname is None:
        return False
    return hostname.lower() in {"localhost", "127.0.0.1", "::1"}
