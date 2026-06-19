# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Host-side validation for iOS Halo Capsule accept-policy snapshots."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


HALO_POLICY_RECEIPT_SCHEMA_VERSION = "edgestudio.halo_capsule_policy_receipt.v1"
HALO_SUPPORTED_POLICY_SCHEMA_VERSIONS = {
    "edgestudio.scaffold_halo_capsule_accept_policy.v1",
    "edgestudio.preview_halo_capsule_accept_policy.v1",
}
HALO_SUPPORTED_MESSAGE_SCHEMA_VERSION = "edgestudio.halo_capsule_mesh_message.v1"
HALO_SUPPORTED_MESSAGE_KIND = "halo_capsule_offer"


class HaloCapsulePolicyValidationError(ValueError):
    pass


class HaloCapsulePolicySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    base_model_id: str
    model_display_name: str
    current_runtime_version: str
    supported_message_schema_version: str
    supported_message_kind: str
    tool_schema_sha256: str
    registered_tool_count: int = Field(ge=0)
    default_enable_thinking: bool

    @field_validator(
        "schema_version",
        "base_model_id",
        "model_display_name",
        "current_runtime_version",
        "supported_message_schema_version",
        "supported_message_kind",
        "tool_schema_sha256",
    )
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("must be a non-empty string")
        return text

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: str) -> str:
        if value not in HALO_SUPPORTED_POLICY_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported Halo policy schema_version: {value}")
        return value

    @field_validator("supported_message_schema_version")
    @classmethod
    def _supported_message_schema(cls, value: str) -> str:
        if value != HALO_SUPPORTED_MESSAGE_SCHEMA_VERSION:
            raise ValueError(f"unsupported Halo message schema: {value}")
        return value

    @field_validator("supported_message_kind")
    @classmethod
    def _supported_message_kind(cls, value: str) -> str:
        if value != HALO_SUPPORTED_MESSAGE_KIND:
            raise ValueError(f"unsupported Halo message kind: {value}")
        return value

    @field_validator("tool_schema_sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("tool_schema_sha256 must be a lowercase sha256 hex string")
        return value


def validate_halo_capsule_policy(
    payload: dict[str, Any],
    *,
    expected_base_model_id: str | None = None,
    expected_tool_schema_sha256: str | None = None,
    min_runtime_version: str | None = None,
) -> dict[str, Any]:
    """Validate an iOS Device Report Halo policy JSON snapshot.

    This deliberately checks only host/device contract metadata. It does not
    accept a capsule, route over mesh, or restore KV cache.
    """
    try:
        policy = HaloCapsulePolicySnapshot.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise HaloCapsulePolicyValidationError(str(exc)) from exc

    warnings: list[str] = []
    if expected_base_model_id is not None and policy.base_model_id != expected_base_model_id:
        raise HaloCapsulePolicyValidationError(
            f"base_model_id mismatch: expected {expected_base_model_id}, got {policy.base_model_id}"
        )
    if (
        expected_tool_schema_sha256 is not None
        and policy.tool_schema_sha256 != expected_tool_schema_sha256
    ):
        raise HaloCapsulePolicyValidationError(
            "tool_schema_sha256 mismatch: "
            f"expected {expected_tool_schema_sha256}, got {policy.tool_schema_sha256}"
        )
    if min_runtime_version is not None and _compare_version(
        policy.current_runtime_version,
        min_runtime_version,
    ) < 0:
        raise HaloCapsulePolicyValidationError(
            "runtime version unsupported: "
            f"minimum {min_runtime_version}, got {policy.current_runtime_version}"
        )
    if policy.registered_tool_count == 0:
        warnings.append("registered_tool_count_zero")

    return {
        "schema_version": HALO_POLICY_RECEIPT_SCHEMA_VERSION,
        "status": "accepted",
        "policy_schema_version": policy.schema_version,
        "base_model_id": policy.base_model_id,
        "current_runtime_version": policy.current_runtime_version,
        "supported_message_schema_version": policy.supported_message_schema_version,
        "supported_message_kind": policy.supported_message_kind,
        "tool_schema_sha256": policy.tool_schema_sha256,
        "registered_tool_count": policy.registered_tool_count,
        "default_enable_thinking": policy.default_enable_thinking,
        "warnings": warnings,
    }


def _compare_version(lhs: str, rhs: str) -> int:
    left = _parse_version(lhs)
    right = _parse_version(rhs)
    width = max(len(left[0]), len(right[0]))
    left_core = left[0] + [0] * (width - len(left[0]))
    right_core = right[0] + [0] * (width - len(right[0]))
    if left_core < right_core:
        return -1
    if left_core > right_core:
        return 1
    return _compare_prerelease(left[1], right[1])


def _parse_version(value: str) -> tuple[list[int], list[str] | None]:
    core_text, _, prerelease_text = value.partition("-")
    core = []
    for part in core_text.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        core.append(int(digits) if digits else 0)
    prerelease = prerelease_text.split(".") if prerelease_text else None
    return core, prerelease


def _compare_prerelease(lhs: list[str] | None, rhs: list[str] | None) -> int:
    if lhs is None and rhs is None:
        return 0
    if lhs is None:
        return 1
    if rhs is None:
        return -1
    width = max(len(lhs), len(rhs))
    for index in range(width):
        if index >= len(lhs):
            return -1
        if index >= len(rhs):
            return 1
        left = _prerelease_token(lhs[index])
        right = _prerelease_token(rhs[index])
        if left < right:
            return -1
        if left > right:
            return 1
    return 0


def _prerelease_token(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    prefix = "".join(ch for ch in value if not ch.isdigit())
    suffix = "".join(ch for ch in value if ch.isdigit())
    if suffix:
        return (1, f"{prefix}{int(suffix):08d}")
    return (1, value)
