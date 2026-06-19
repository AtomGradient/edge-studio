# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Dry-run Host Model generation for route-matrix plan gaps."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.services.host_model_assistant import HostModelGenerate
from backend.services.route_action_seed_generator import generate_route_action_seed_candidates


ROUTE_MATRIX_PLAN_GAP_GENERATION_SCHEMA_VERSION = (
    "edgestudio.route_matrix_plan_gap_generation.v0"
)


def generate_route_matrix_plan_gap_seed_candidates(
    *,
    plan_gap_request: dict[str, Any],
    peer_id: str | None = None,
    provider: str | None = None,
    host_model_id: str | None = None,
    host_model_generate: HostModelGenerate | None = None,
) -> dict[str, Any]:
    generated_at = _utc_now()
    try:
        result = _plan_gap_result(plan_gap_request)
        seed_request = _seed_request(result)
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="route matrix plan gap generation request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    golden_cases = seed_request.get("golden_cases")
    if not isinstance(golden_cases, list) or not golden_cases:
        return {
            "ok": True,
            "schema_version": ROUTE_MATRIX_PLAN_GAP_GENERATION_SCHEMA_VERSION,
            "status": "no_plan_gap_seed_cases",
            "result": {
                "run_id": result.get("run_id"),
                "app_id": result.get("app_id"),
                "route_action_seed_candidates": None,
            },
            "error": None,
            "audit": _audit(
                generated_at=generated_at,
                seed_case_count=0,
                provider=provider,
                host_model_id=host_model_id,
            ),
        }

    receipt = generate_route_action_seed_candidates(
        app_id=_required_text(seed_request.get("app_id"), "seed_request.app_id"),
        tool_registry=_list_of_dicts(seed_request.get("tool_registry")),
        golden_cases=golden_cases,
        target_seed_count=int(seed_request.get("target_seed_count") or len(golden_cases)),
        seed_run_id=_text(seed_request.get("seed_run_id")) or None,
        peer_id=peer_id,
        provider=provider,
        host_model_id=host_model_id,
        host_model_generate=host_model_generate,
    )
    status = _status_from_seed_receipt(receipt)
    return {
        "ok": receipt.get("ok") is True,
        "schema_version": ROUTE_MATRIX_PLAN_GAP_GENERATION_SCHEMA_VERSION,
        "status": status,
        "result": {
            "run_id": result.get("run_id"),
            "app_id": result.get("app_id"),
            "plan_seed_case_count": len(golden_cases),
            "route_action_seed_candidates": receipt,
        },
        "error": None if receipt.get("ok") is True else _receipt_error(receipt),
        "audit": _audit(
            generated_at=generated_at,
            seed_case_count=len(golden_cases),
            provider=provider,
            host_model_id=host_model_id,
        ),
    }


def _status_from_seed_receipt(receipt: dict[str, Any]) -> str:
    if receipt.get("ok") is not True:
        return "plan_gap_seed_generation_failed"
    seed_status = _text(receipt.get("status"))
    if seed_status == "seed_candidates_ready":
        return "plan_gap_seed_candidates_ready"
    if seed_status == "stub_pending_host_model":
        return "plan_gap_seed_generation_pending_host_model"
    return f"plan_gap_seed_generation_{seed_status or 'unknown'}"


def _receipt_error(receipt: dict[str, Any]) -> Any:
    error = receipt.get("error")
    if error is not None:
        return error
    route_action_response = receipt.get("route_action_response")
    if isinstance(route_action_response, dict):
        return route_action_response.get("error")
    return None


def _plan_gap_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("plan_gap_request must be an object")
    if value.get("ok") is not True:
        raise ValueError("plan_gap_request.ok must be true")
    result = value.get("result")
    if not isinstance(result, dict):
        raise ValueError("plan_gap_request.result must be an object")
    return result


def _seed_request(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("route_action_seed_candidates_request")
    if not isinstance(value, dict):
        raise ValueError("plan_gap_request.result.route_action_seed_candidates_request missing")
    return value


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("tool_registry must be a list")
    return [dict(item) for item in value if isinstance(item, dict)]


def _audit(
    *,
    generated_at: str,
    seed_case_count: int,
    provider: str | None,
    host_model_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": "edgestudio.route_matrix_plan_gap_generation_audit.v0",
        "method": "generate_route_matrix_plan_gap_seed_candidates",
        "generated_at": generated_at,
        "training_side_only": True,
        "writes_events": False,
        "writes_runtime_artifacts": False,
        "seed_case_count": seed_case_count,
        "provider": provider,
        "host_model_id": host_model_id,
    }


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_PLAN_GAP_GENERATION_SCHEMA_VERSION,
        "status": "invalid_input",
        "result": None,
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
            "details": details,
        },
        "audit": _audit(
            generated_at=generated_at,
            seed_case_count=0,
            provider=None,
            host_model_id=None,
        ),
    }


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
