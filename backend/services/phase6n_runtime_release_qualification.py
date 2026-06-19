# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Build Phase 6N runtime release qualification receipts.

This service is read-only over existing Phase 6 receipt files. It does not
launch apps, pull device snapshots, deploy artifacts, train models, or enable
live routing.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json


PHASE6N_RUNTIME_RELEASE_QUALIFICATION_SCHEMA_VERSION = (
    "edgestudio.phase6n_runtime_release_qualification.v0"
)

_FORBIDDEN_FRESH_DEVICE_KEYS = {
    "current_device_snapshot",
    "device_snapshot",
    "device_snapshot_path",
    "devicectl_output_path",
    "fresh_device_counts",
}


def build_phase6n_runtime_release_qualification(
    request: dict[str, Any],
) -> dict[str, Any]:
    generated_at = _utc_now()
    if not isinstance(request, dict):
        return _error(
            code="invalid_input",
            message="request must be an object",
            details={"received_type": type(request).__name__},
            generated_at=generated_at,
        )

    try:
        _reject_fresh_device_inputs(request)
        run_id = _required_text(request.get("run_id"), "run_id")
        phase6l = _required_object(request.get("phase6l_receipt"), "phase6l_receipt")
        phase6m = _required_object(request.get("phase6m_receipt"), "phase6m_receipt")
        shadow_review = _required_object(request.get("shadow_review"), "shadow_review")
        recovery = _required_object(request.get("recovery_receipt"), "recovery_receipt")
        attribution_correction = _optional_object(
            request.get("attribution_correction")
        )
    except (TypeError, ValueError) as exc:
        return _error(
            code="invalid_input",
            message="phase6n runtime release qualification request is invalid",
            details={"reason": str(exc)},
            generated_at=generated_at,
        )

    source_paths = _source_paths(request.get("source_paths"))
    gates = {
        "training_artifact_eligibility": _training_artifact_gate(phase6l, phase6m),
        "runtime_shadow_wiring": _runtime_shadow_wiring_gate(
            phase6l,
            phase6m,
            shadow_review,
        ),
        "stage_separated_mutation": _stage_separated_mutation_gate(
            phase6m,
            app_contract=_optional_object(request.get("app_contract")),
            attribution_correction=attribution_correction,
        ),
        "candidate_executability": _candidate_executability_gate(shadow_review),
        "live_policy_and_approval": _live_policy_gate(
            live_policy=_optional_object(request.get("live_policy")),
            user_approval=_optional_object(request.get("user_approval")),
        ),
        "newdailyn_recovery_context": _recovery_context_gate(recovery),
    }
    summary = _final_summary(gates)
    return {
        "ok": True,
        "schema_version": PHASE6N_RUNTIME_RELEASE_QUALIFICATION_SCHEMA_VERSION,
        "status": summary["status"],
        "runtime_release_ready": summary["runtime_release_ready"],
        "deployment_approved": summary["deployment_approved"],
        "broad_live_routing_enabled": summary["broad_live_routing_enabled"],
        "production_router_improved": summary["production_router_improved"],
        "ready_for_live_routing": summary["ready_for_live_routing"],
        "gates": gates,
        "evidence_refs": _evidence_refs(
            source_paths=source_paths,
            phase6l=phase6l,
            phase6m=phase6m,
            shadow_review=shadow_review,
            recovery=recovery,
            attribution_correction=attribution_correction,
        ),
        "residual_risks": summary["residual_risks"],
        "non_goals_enforced": {
            "fresh_device_snapshot": False,
            "devicectl": False,
            "app_launch": False,
            "install_or_uninstall": False,
            "retrain": False,
            "new_heldout_split": False,
            "controlled_live_routing": False,
            "broad_live_routing": False,
        },
        "audit": {
            "schema_version": "edgestudio.phase6n_runtime_release_qualification_audit.v0",
            "method": "build_phase6n_runtime_release_qualification",
            "requested_run_id": run_id,
            "generated_at": generated_at,
            "read_only_inputs": True,
            "source_paths": source_paths,
        },
        "error": None,
    }


def build_phase6n_runtime_release_qualification_from_files(
    *,
    run_id: str,
    phase6l_receipt_path: Path,
    phase6m_receipt_path: Path,
    shadow_review_path: Path,
    recovery_receipt_path: Path,
    attribution_correction_path: Path | None = None,
    app_contract_path: Path | None = None,
    live_policy_path: Path | None = None,
    user_approval_path: Path | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "run_id": run_id,
        "phase6l_receipt": _read_json(phase6l_receipt_path),
        "phase6m_receipt": _read_json(phase6m_receipt_path),
        "shadow_review": _read_json(shadow_review_path),
        "recovery_receipt": _read_json(recovery_receipt_path),
        "source_paths": {
            "phase6l_receipt": str(phase6l_receipt_path),
            "phase6m_receipt": str(phase6m_receipt_path),
            "shadow_review": str(shadow_review_path),
            "recovery_receipt": str(recovery_receipt_path),
        },
    }
    if attribution_correction_path is not None:
        request["attribution_correction"] = _read_json(attribution_correction_path)
        request["source_paths"]["attribution_correction"] = str(
            attribution_correction_path
        )
    if app_contract_path is not None:
        request["app_contract"] = _read_json(app_contract_path)
        request["source_paths"]["app_contract"] = str(app_contract_path)
    if live_policy_path is not None:
        request["live_policy"] = _read_json(live_policy_path)
        request["source_paths"]["live_policy"] = str(live_policy_path)
    if user_approval_path is not None:
        request["user_approval"] = _read_json(user_approval_path)
        request["source_paths"]["user_approval"] = str(user_approval_path)
    return build_phase6n_runtime_release_qualification(request)


def write_phase6n_runtime_release_qualification_receipts(
    *,
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "phase6n_runtime_release_qualification_receipt.json"
    md_path = output_dir / "phase6n_runtime_release_qualification_receipt.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown_receipt(result, json_path=json_path), encoding="utf-8")
    return {
        "json": str(json_path),
        "md": str(md_path),
    }


def default_phase6n_output_dir(
    *,
    phase6m_receipt: dict[str, Any],
    generated_at: str | None = None,
) -> Path:
    artifact = _object(phase6m_receipt.get("artifact"))
    artifact_path = _text(artifact.get("path"))
    if artifact_path:
        root = Path(artifact_path)
    else:
        root = Path.cwd()
    stamp = (generated_at or _utc_now()).replace("+00:00", "Z")
    stamp = stamp.replace("-", "").replace(":", "")
    return root / f"phase6n_runtime_release_qualification_{stamp}"


def _training_artifact_gate(
    phase6l: dict[str, Any],
    phase6m: dict[str, Any],
) -> dict[str, Any]:
    artifact_l = _object(phase6l.get("artifact"))
    artifact_m = _object(phase6m.get("artifact"))
    files_l = _artifact_files(artifact_l)
    files_m = _artifact_files(artifact_m)
    common_files = sorted(set(files_l) & set(files_m))
    hash_mismatches = [
        name for name in common_files if files_l.get(name) != files_m.get(name)
    ]
    training_run_id = _text(
        artifact_m.get("training_run_id") or artifact_l.get("training_run_id")
    )
    path_l = _text(artifact_l.get("path"))
    path_m = _text(artifact_m.get("path"))
    path_match = bool(path_l and path_m and path_l == path_m)
    passed = bool(
        training_run_id
        and path_match
        and common_files
        and not hash_mismatches
    )
    return {
        "status": "passed" if passed else "blocked_artifact_mismatch_or_missing",
        "passed": passed,
        "training_artifact_eligible": passed,
        "training_run_id": training_run_id,
        "artifact_path": path_m or path_l,
        "common_file_count": len(common_files),
        "hash_mismatches": hash_mismatches,
        "note": (
            "training artifact eligibility does not imply runtime release readiness"
        ),
    }


def _runtime_shadow_wiring_gate(
    phase6l: dict[str, Any],
    phase6m: dict[str, Any],
    shadow_review: dict[str, Any],
) -> dict[str, Any]:
    phase6l_status = _text(phase6l.get("status"))
    phase6m_status = _text(phase6m.get("status"))
    interpretation = _object(phase6m.get("interpretation"))
    device_run = _object(phase6m.get("device_run"))
    shadow_summary = _shadow_summary(shadow_review)
    shadow_audit = _object(shadow_review.get("audit")).get("input_summary")
    shadow_audit = _object(shadow_audit)
    prompt_count = _int(device_run.get("prompt_count"))
    scored_count = _int(device_run.get("scored_count"))
    runtime_validation_count = _int(device_run.get("runtime_validation_count"))
    cli_device_agree_count = _int(
        _object(phase6m.get("shadow_review")).get("cli_device_agree_count")
    )
    if cli_device_agree_count <= 0:
        cli_device_agree_count = _cli_device_agree_count(shadow_review)
    case_count = _int(shadow_summary.get("case_count")) or _int(
        shadow_audit.get("case_count")
    )
    clean_stability = _object(phase6m.get("stability")).get(
        "new_crash_or_jetsam_after_postinstall_run"
    ) is False
    passed = bool(
        phase6l_status == "passed_wiring_smoke"
        and interpretation.get("device_shadow_wiring_passed") is True
        and device_run.get("done") is True
        and prompt_count > 0
        and scored_count == prompt_count
        and runtime_validation_count == prompt_count
        and case_count > 0
        and cli_device_agree_count == case_count
        and clean_stability
    )
    return {
        "status": "passed" if passed else "blocked_runtime_shadow_wiring",
        "passed": passed,
        "runtime_shadow_wiring_passed": passed,
        "phase6l_status": phase6l_status,
        "phase6m_status": phase6m_status,
        "prompt_count": prompt_count,
        "scored_not_applied_count": scored_count,
        "runtime_validation_count": runtime_validation_count,
        "cli_device_agree_count": cli_device_agree_count,
        "case_count": case_count,
        "new_crash_or_jetsam": not clean_stability,
        "note": (
            "runtime shadow wiring is SDK consistency evidence, not model "
            "quality evidence"
        ),
    }


def _stage_separated_mutation_gate(
    phase6m: dict[str, Any],
    *,
    app_contract: dict[str, Any] | None,
    attribution_correction: dict[str, Any] | None,
) -> dict[str, Any]:
    audit = _object(phase6m.get("data_mutation_audit"))
    tracked_tables, source = _tracked_tables(audit=audit, app_contract=app_contract)
    correction = _valid_attribution_correction(attribution_correction)
    shadow_delta = _count_map(
        correction.get("shadow_run_delta")
        if correction
        else audit.get("shadow_run_delta")
    )
    install_startup_delta = _count_map(
        correction.get("router_or_install_mutation_delta")
        if correction
        else audit.get("release_install_startup_delta")
    )
    shadow_immutable = bool(
        tracked_tables
        and all(_int(shadow_delta.get(table)) == 0 for table in tracked_tables)
    )
    if correction:
        install_startup_mutation = any(
            _int(install_startup_delta.get(table)) != 0 for table in tracked_tables
        )
    else:
        install_startup_mutation = (
            audit.get("release_install_startup_mutation_detected") is True
            or any(
                _int(install_startup_delta.get(table)) != 0
                for table in tracked_tables
            )
        )
    status = "passed_shadow_run_immutable"
    if not shadow_immutable:
        status = "blocked_shadow_run_mutation"
    elif install_startup_mutation:
        status = "shadow_immutable_install_startup_mutation_detected"
    return {
        "status": status,
        "passed": shadow_immutable,
        "release_blocked": install_startup_mutation,
        "shadow_run_business_data_immutable": shadow_immutable,
        "shadow_run_business_data_mutated": not shadow_immutable,
        "install_or_startup_mutation_detected": install_startup_mutation,
        "post_install_pre_startup_counts": None,
        "install_startup_combined_delta": install_startup_delta,
        "shadow_run_delta": shadow_delta,
        "tracked_tables": tracked_tables,
        "tracked_tables_source": source,
        "attribution_correction_applied": bool(correction),
        "attribution_correction_status": (
            _text(correction.get("status")) if correction else ""
        ),
        "corrected_attribution": (
            _object(correction.get("corrected_attribution")) if correction else {}
        ),
        "note": (
            "install/startup deltas are separate from shadow-run immutability"
        ),
    }


def _candidate_executability_gate(shadow_review: dict[str, Any]) -> dict[str, Any]:
    summary = _shadow_summary(shadow_review)
    routing_candidates = _int(summary.get("routing_candidate_count"))
    executable_candidates = _int(summary.get("executable_candidate_count"))
    ready = summary.get("ready_for_live_routing") is True and executable_candidates > 0
    status = "passed" if ready else "not_live_ready_no_executable_candidates"
    return {
        "status": status,
        "passed": ready,
        "ready_for_live_routing": ready,
        "routing_candidate_count": routing_candidates,
        "executable_candidate_count": executable_candidates,
        "intent_only_candidate_count": _int(summary.get("intent_only_candidate_count")),
        "ready_for_live_routing_reason": _text(
            summary.get("ready_for_live_routing_reason")
        ),
        "routing_candidate_lanes": _text_list(summary.get("routing_candidate_lanes")),
        "evidence_gap_cases": _text_list(
            summary.get("routing_candidate_evidence_gap_cases")
        ),
    }


def _live_policy_gate(
    *,
    live_policy: dict[str, Any] | None,
    user_approval: dict[str, Any] | None,
) -> dict[str, Any]:
    approval = _object(user_approval)
    approved = approval.get("approved") is True
    if not live_policy:
        return {
            "status": "live_policy_missing_disabled",
            "passed": False,
            "live_routing_enabled": False,
            "ready_for_live_routing": False,
            "deployment_approved": False,
            "user_approval_recorded": approved,
            "reason": "missing_live_policy",
        }
    summary = _object(_object(live_policy.get("result")).get("summary"))
    live_enabled = summary.get("live_routing_enabled") is True
    ready = summary.get("ready_for_live_routing") is True
    passed = bool(live_enabled and ready and approved)
    return {
        "status": "passed" if passed else "live_policy_not_approved_or_not_ready",
        "passed": passed,
        "live_routing_enabled": live_enabled,
        "ready_for_live_routing": ready,
        "deployment_approved": approved,
        "user_approval_recorded": approved,
        "reason": _text(summary.get("ready_for_live_routing_reason")),
    }


def _recovery_context_gate(recovery: dict[str, Any]) -> dict[str, Any]:
    verification = _object(recovery.get("verification"))
    ack = _object(recovery.get("stalin_acknowledgement"))
    return {
        "status": _text(recovery.get("status")) or "missing",
        "passed": recovery.get("status") == "verified_canonical_json_restore_complete",
        "dependency_context_only": True,
        "router_quality_evidence": False,
        "rollback_window_accepted": ack.get("rollback_window_accepted") is True,
        "launch_approved": ack.get("launch_approved") is True,
        "sqlite_counts": _count_map(verification.get("sqlite_counts")),
    }


def _final_summary(gates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    artifact = gates["training_artifact_eligibility"]
    wiring = gates["runtime_shadow_wiring"]
    mutation = gates["stage_separated_mutation"]
    candidates = gates["candidate_executability"]
    live_policy = gates["live_policy_and_approval"]
    if artifact.get("passed") is not True:
        blockers.append("training_artifact_not_eligible")
    if wiring.get("passed") is not True:
        blockers.append("runtime_shadow_wiring_not_passed")
    if mutation.get("passed") is not True:
        blockers.append("shadow_run_business_data_mutation")
    if mutation.get("release_blocked") is True:
        blockers.append("install_startup_mutation_requires_user_acceptance")
    if candidates.get("ready_for_live_routing") is not True:
        blockers.append("no_executable_live_candidates")
    if live_policy.get("passed") is not True:
        blockers.append("live_policy_missing_or_not_approved")
    release_ready = not blockers
    status = "runtime_release_ready" if release_ready else (
        "not_release_ready_stage_separated"
    )
    risks = list(blockers)
    risks.append("newdailyn_recovery_context_is_not_router_evidence")
    return {
        "status": status,
        "runtime_release_ready": release_ready,
        "deployment_approved": live_policy.get("deployment_approved") is True
        and release_ready,
        "broad_live_routing_enabled": False,
        "production_router_improved": False,
        "ready_for_live_routing": candidates.get("ready_for_live_routing") is True,
        "residual_risks": _dedupe_text(risks),
    }


def _evidence_refs(
    *,
    source_paths: dict[str, str],
    phase6l: dict[str, Any],
    phase6m: dict[str, Any],
    shadow_review: dict[str, Any],
    recovery: dict[str, Any],
    attribution_correction: dict[str, Any] | None,
) -> dict[str, Any]:
    valid_attribution_correction = _valid_attribution_correction(
        attribution_correction
    )
    return {
        "source_paths": source_paths,
        "source_sha256": {
            key: _sha256_file(Path(path))
            for key, path in source_paths.items()
            if path and Path(path).is_file()
        },
        "phase6l": {
            "status": _text(phase6l.get("status")),
            "run_id": _text(phase6l.get("run_id")),
        },
        "phase6m": {
            "status": _text(phase6m.get("status")),
            "device_run_id": _text(_object(phase6m.get("device_run")).get("run_id")),
        },
        "shadow_review": {
            "ok": shadow_review.get("ok") is True,
            "run_id": _text(_object(shadow_review.get("result")).get("run_id")),
        },
        "recovery": {
            "status": _text(recovery.get("status")),
            "router_quality_evidence": False,
        },
        "attribution_correction": {
            "applied": bool(valid_attribution_correction),
            "status": _text(_object(attribution_correction).get("status")),
            "schema_version": _text(
                _object(attribution_correction).get("schema_version")
            ),
        },
    }


def _markdown_receipt(result: dict[str, Any], *, json_path: Path) -> str:
    gates = _object(result.get("gates"))
    mutation = _object(gates.get("stage_separated_mutation"))
    wiring = _object(gates.get("runtime_shadow_wiring"))
    candidates = _object(gates.get("candidate_executability"))
    return "\n".join([
        "# Phase 6N Runtime Release Qualification Receipt",
        "",
        f"Status: `{_text(result.get('status'))}`",
        "",
        "## Summary",
        "",
        f"- Runtime release ready: `{str(result.get('runtime_release_ready')).lower()}`",
        f"- Ready for live routing: `{str(result.get('ready_for_live_routing')).lower()}`",
        f"- Deployment approved: `{str(result.get('deployment_approved')).lower()}`",
        f"- Broad live routing enabled: `{str(result.get('broad_live_routing_enabled')).lower()}`",
        f"- Production router improved: `{str(result.get('production_router_improved')).lower()}`",
        "",
        "## Key Gates",
        "",
        f"- Runtime shadow wiring: `{_text(wiring.get('status'))}`",
        f"- Stage-separated mutation: `{_text(mutation.get('status'))}`",
        f"- Shadow-run business data immutable: `{str(mutation.get('shadow_run_business_data_immutable')).lower()}`",
        f"- Install/startup mutation detected: `{str(mutation.get('install_or_startup_mutation_detected')).lower()}`",
        f"- Routing candidates: `{_int(candidates.get('routing_candidate_count'))}`",
        f"- Executable candidates: `{_int(candidates.get('executable_candidate_count'))}`",
        "",
        "## Residual Risks",
        "",
        *[f"- `{item}`" for item in result.get("residual_risks") or []],
        "",
        "## Evidence",
        "",
        f"- Receipt JSON: `{json_path}`",
        "",
        "This receipt is read-only over existing evidence. It is not deployment "
        "approval, not a live-routing enablement, and not a production router "
        "improvement claim.",
        "",
    ])


def _tracked_tables(
    *,
    audit: dict[str, Any],
    app_contract: dict[str, Any] | None,
) -> tuple[list[str], str]:
    if app_contract:
        tables = _text_list(app_contract.get("tracked_tables"))
        if tables:
            return tables, "app_contract.tracked_tables"
    keys: set[str] = set()
    for name in (
        "shadow_run_delta",
        "release_install_startup_delta",
        "postinstall_baseline_counts",
        "after_shadow_run_counts",
    ):
        keys.update(_count_map(audit.get(name)).keys())
    return sorted(keys), "phase6m_data_mutation_audit_keys"


def _valid_attribution_correction(value: dict[str, Any] | None) -> dict[str, Any]:
    if not value:
        return {}
    if value.get("do_not_rewrite_original_receipt") is not True:
        return {}
    if _text(value.get("schema_version")) != (
        "edgestudio.phase6m_data_mutation_attribution_correction.v0"
    ):
        return {}
    return value


def _artifact_files(artifact: dict[str, Any]) -> dict[str, str]:
    raw_files = _object(artifact.get("files"))
    out: dict[str, str] = {}
    for name, value in raw_files.items():
        if isinstance(value, Mapping):
            sha = _text(value.get("sha256"))
        else:
            sha = _text(value)
        if name and sha:
            out[str(name)] = sha
    return out


def _shadow_summary(shadow_review: dict[str, Any]) -> dict[str, Any]:
    return _object(_object(shadow_review.get("result")).get("summary"))


def _cli_device_agree_count(shadow_review: dict[str, Any]) -> int:
    cases = _object(shadow_review.get("result")).get("cases")
    if not isinstance(cases, list):
        return 0
    return sum(
        1
        for row in cases
        if isinstance(row, dict)
        and _object(row.get("comparison")).get("cli_device_agree") is True
    )


def _source_paths(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if str(v).strip()}


def _reject_fresh_device_inputs(request: dict[str, Any]) -> None:
    present = sorted(key for key in _FORBIDDEN_FRESH_DEVICE_KEYS if key in request)
    if present:
        raise ValueError(
            "Phase 6N-A must not consume fresh device snapshots: "
            + ", ".join(present)
        )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(
    *,
    code: str,
    message: str,
    details: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "schema_version": PHASE6N_RUNTIME_RELEASE_QUALIFICATION_SCHEMA_VERSION,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
        "audit": {
            "schema_version": "edgestudio.phase6n_runtime_release_qualification_audit.v0",
            "method": "build_phase6n_runtime_release_qualification",
            "generated_at": generated_at,
            "read_only_inputs": True,
        },
    }


def _required_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _optional_object(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("optional object value must be an object")
    return value


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({
        text for item in value if (text := _text(item))
    })


def _count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _int(raw) for key, raw in value.items()}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
