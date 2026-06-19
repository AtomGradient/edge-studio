# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persona / RPP API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from backend.schemas.personal import (
    CorrectionLedgerEntryRequest,
    HardFactLeakageReviewRequest,
    ProfileNamingArtifactFromLatestRPPRequest,
    ProfileNamingArtifactRequest,
    ProfileNamingRequest,
    RPPArtifactsUploadRequest,
    RouteActionLearnerDatasetRequest,
    RouteActionSeedCandidatesRequest,
    RouteActionTrainingEventsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/personal", tags=["personal"])
ws_router = APIRouter(tags=["personal"])


@router.post("/rpp/artifacts")
def upload_rpp_artifacts(req: RPPArtifactsUploadRequest) -> dict:
    """Persist device-originated RPP artifacts on the local EdgeStudio host.

    v0 is intentionally a receipt-only backflow surface: it records a concrete
    RPP run and any uploaded artifact bytes/metadata, but does not start training or generate product decisions.
    """
    from backend.services.rpp_artifact_store import (
        RPPArtifactUploadError,
        store_rpp_artifact_upload,
    )

    try:
        return store_rpp_artifact_upload(req.model_dump())
    except RPPArtifactUploadError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.rpp_artifact_receipt.v0",
                "error": exc.to_error(),
            },
        ) from exc


@router.get("/rpp/artifacts/latest")
def latest_rpp_artifacts(peer_id: str = Query(...)) -> dict:
    """Return a raw-free summary of the latest stored RPP artifact run."""

    from backend.services.rpp_artifact_store import latest_rpp_artifact_run_for_peer

    latest = latest_rpp_artifact_run_for_peer(peer_id)
    if latest is None:
        return {
            "ok": True,
            "schema_version": "edgestudio.rpp_artifact_latest.v0",
            "status": "missing",
            "peer_id": peer_id,
            "rpp_run_id": None,
            "received_at_ms": None,
            "base_model_id": None,
            "layer_id": None,
            "a_version": None,
            "a_hash": None,
            "dataset_summary": {},
            "artifacts": [],
        }
    receipt = latest.receipt if isinstance(latest.receipt, dict) else {}
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    return {
        "ok": True,
        "schema_version": "edgestudio.rpp_artifact_latest.v0",
        "status": "found",
        "peer_id": latest.peer_id,
        "rpp_run_id": latest.rpp_run_id,
        "received_at_ms": latest.received_at_ms,
        "base_model_id": receipt.get("base_model_id"),
        "layer_id": receipt.get("layer_id"),
        "a_version": receipt.get("a_version"),
        "a_hash": receipt.get("a_hash"),
        "dataset_summary": receipt.get("dataset_summary") or {},
        "artifacts": [
            {
                "name": artifact.get("name"),
                "role": artifact.get("role"),
                "stored": artifact.get("stored") is True,
                "size_bytes": artifact.get("size_bytes"),
                "sha256": artifact.get("sha256"),
            }
            for artifact in artifacts
            if isinstance(artifact, dict)
        ],
    }


@router.get("/rpp/artifacts/latest/inspect")
def inspect_latest_rpp_artifacts(peer_id: str = Query(...)) -> dict:
    """Return a UI-ready local inspection of the latest stored RPP run."""

    from backend.services.rpp_artifact_inspector import (
        RPPArtifactInspectionError,
        inspect_latest_rpp_artifact_run_for_peer,
    )

    try:
        return inspect_latest_rpp_artifact_run_for_peer(peer_id)
    except RPPArtifactInspectionError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.rpp_artifact_inspection.v0",
                "status": "error",
                "error": exc.to_error(),
            },
        ) from exc


@router.post("/corrections")
def record_personal_correction(req: CorrectionLedgerEntryRequest) -> dict:
    """Record one local correction ledger entry without triggering regeneration."""

    from backend.services.correction_ledger import (
        CorrectionLedgerError,
        record_correction_entry,
    )

    try:
        return record_correction_entry(req.model_dump())
    except CorrectionLedgerError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.correction_ledger_receipt.v0",
                "status": "error",
                "error": exc.to_error(),
            },
        ) from exc


@router.get("/corrections")
def list_personal_corrections(
    peer_id: str | None = Query(None),
    correction_type: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=200),
) -> dict:
    """List recent local correction ledger entries."""

    from backend.services.correction_ledger import (
        CorrectionLedgerError,
        list_correction_entries,
    )

    try:
        return list_correction_entries(
            peer_id=peer_id,
            correction_type=correction_type,
            status=status,
            limit=limit,
        )
    except CorrectionLedgerError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.correction_ledger_index.v0",
                "status": "error",
                "error": exc.to_error(),
            },
        ) from exc


@router.get("/corrections/context")
def personal_correction_context(
    peer_id: str = Query(...),
    include_status: list[str] | None = Query(None),
    limit: int = Query(200, ge=1, le=200),
) -> dict:
    """Compile correction ledger entries for downstream RPP/profile consumers."""

    from backend.services.correction_ledger import (
        CorrectionLedgerError,
        build_correction_consumer_context,
    )

    try:
        return build_correction_consumer_context(
            peer_id=peer_id,
            include_statuses=include_status,
            limit=limit,
        )
    except CorrectionLedgerError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.correction_consumer_context.v0",
                "status": "error",
                "error": exc.to_error(),
            },
        ) from exc


@router.get("/corrections/{peer_id}/{correction_id}")
def load_personal_correction(peer_id: str, correction_id: str) -> dict:
    """Load one local correction ledger entry by peer and id."""

    from backend.services.correction_ledger import (
        CorrectionLedgerError,
        load_correction_entry,
    )

    try:
        return load_correction_entry(correction_id, peer_id=peer_id)
    except CorrectionLedgerError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": "edgestudio.correction_ledger_receipt.v0",
                "status": "error",
                "error": exc.to_error(),
            },
        ) from exc


@router.post("/profile_naming/generate")
def generate_profile_naming_for_rpp(
    req: ProfileNamingRequest,
) -> dict:
    """Generate host-model profile labels/narratives for one RPP output."""

    from backend.services.host_model_assistant import (
        HOST_MODEL_PROVIDER,
        generate_profile_naming,
    )

    return generate_profile_naming(
        req.rpp_output,
        forbidden_entities=req.forbidden_entities,
        host_model_id=req.host_model_id,
        provider=req.provider or HOST_MODEL_PROVIDER,
    )


@router.post("/profile_naming/artifact")
def generate_profile_naming_artifact(
    req: ProfileNamingArtifactRequest,
) -> dict:
    """Generate profile naming and persist it as a same-run RPP artifact."""

    from backend.services.profile_naming_artifacts import (
        generate_and_store_profile_naming_artifact,
    )

    return generate_and_store_profile_naming_artifact(
        rpp_output=req.rpp_output,
        peer_id=req.peer_id,
        forbidden_entities=req.forbidden_entities,
        host_model_id=req.host_model_id,
        provider=req.provider,
    )


@router.post("/profile_naming/artifact/from_latest_rpp")
def generate_profile_naming_artifact_from_latest_rpp(
    req: ProfileNamingArtifactFromLatestRPPRequest,
) -> dict:
    """Generate profile naming from the latest stored RPP run for one peer."""

    from backend.services.profile_naming_artifacts import (
        generate_and_store_profile_naming_artifact_from_latest_rpp,
    )

    return generate_and_store_profile_naming_artifact_from_latest_rpp(
        peer_id=req.peer_id,
        forbidden_entities=req.forbidden_entities,
        host_model_id=req.host_model_id,
        provider=req.provider,
    )


@router.post("/route_action/training_events")
def generate_route_action_training_events(
    req: RouteActionTrainingEventsRequest,
) -> dict:
    """Generate route/action supervision and persist it as training events."""

    from backend.services.route_action_training_events import (
        ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
        generate_and_store_route_action_training_events,
    )

    try:
        return generate_and_store_route_action_training_events(
            rpp_output=req.rpp_output,
            eval_cases=req.eval_cases,
            peer_id=req.peer_id,
            dry_run=req.dry_run,
            host_model_id=req.host_model_id,
            provider=req.provider,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": ROUTE_ACTION_TRAINING_EVENTS_SCHEMA_VERSION,
                "error": {
                    "code": "invalid_route_action_training_events_request",
                    "message": str(exc),
                    "retryable": False,
                    "details": {},
                },
            },
        ) from exc


@router.post("/route_action/learner_dataset")
def build_route_action_learner_dataset_endpoint(
    req: RouteActionLearnerDatasetRequest,
) -> dict:
    """Build a gated route/action learner feed from stored events."""

    from backend.services.route_action_training_events import (
        ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
        build_route_action_learner_dataset,
    )

    try:
        return build_route_action_learner_dataset(
            peer_id=req.peer_id,
            rpp_run_id=req.rpp_run_id,
            limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "ok": False,
                "schema_version": ROUTE_ACTION_LEARNER_DATASET_SCHEMA_VERSION,
                "error": {
                    "code": "invalid_route_action_learner_dataset_request",
                    "message": str(exc),
                    "retryable": False,
                    "details": {},
                },
            },
        ) from exc


@router.post("/route_action/seed_candidates")
def generate_route_action_seed_candidates_endpoint(
    req: RouteActionSeedCandidatesRequest,
) -> dict:
    """Generate dry-run Host Model route/action seed candidates."""

    from backend.services.route_action_seed_generator import (
        generate_route_action_seed_candidates,
    )

    return generate_route_action_seed_candidates(
        app_id=req.app_id,
        tool_registry=req.tool_registry,
        golden_cases=req.golden_cases,
        rpp_output=req.rpp_output,
        target_seed_count=req.target_seed_count,
        seed_run_id=req.seed_run_id,
        peer_id=req.peer_id,
        host_model_id=req.host_model_id,
        provider=req.provider,
    )


@router.post("/hard_fact_leakage/review")
def review_hard_fact_leakage_samples(
    req: HardFactLeakageReviewRequest,
) -> dict:
    """Review candidate samples for hard-fact leakage."""

    from backend.services.host_model_assistant import (
        HOST_MODEL_PROVIDER,
        review_hard_fact_leakage,
    )

    return review_hard_fact_leakage(
        req.samples,
        req.forbidden_entities,
        host_model_id=req.host_model_id,
        provider=req.provider or HOST_MODEL_PROVIDER,
    )
