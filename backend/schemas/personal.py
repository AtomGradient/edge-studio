# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Persona / RPP API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RPPArtifactUploadItem(BaseModel):
    """One device-originated RPP artifact or artifact metadata record."""

    name: str
    role: str
    size_bytes: int | None = None
    sha256: str | None = None
    content_base64: str | None = None


class RPPArtifactsUploadRequest(BaseModel):
    """Device → EdgeStudio RPP artifact backflow envelope."""

    schema_version: str = "edgestudio.rpp_artifact_upload.v0"
    peer_id: str
    rpp_run_id: str
    base_model_id: str = ""
    layer_id: int | None = None
    a_version: str = ""
    a_hash: str = ""
    dataset_summary: dict = Field(default_factory=dict)
    rpp_last_run: dict | None = None
    artifacts: list[RPPArtifactUploadItem] = Field(default_factory=list)


class ProfileNamingRequest(BaseModel):
    """Generate host-model labels/narratives for one RPP output."""

    schema_version: str = "edgestudio.profile_naming_request.v0"
    rpp_output: dict = Field(default_factory=dict)
    forbidden_entities: list[str] = Field(default_factory=list)
    host_model_id: str | None = None
    provider: str | None = None


class ProfileNamingArtifactRequest(BaseModel):
    """Generate profile naming and store it as an RPP B-naming artifact."""

    schema_version: str = "edgestudio.profile_naming_artifact_request.v0"
    peer_id: str
    rpp_output: dict = Field(default_factory=dict)
    forbidden_entities: list[str] = Field(default_factory=list)
    host_model_id: str | None = None
    provider: str | None = None


class ProfileNamingArtifactFromLatestRPPRequest(BaseModel):
    """Generate B-naming from the latest stored RPP run for one peer."""

    schema_version: str = (
        "edgestudio.profile_naming_artifact_from_latest_rpp_request.v0"
    )
    peer_id: str
    forbidden_entities: list[str] = Field(default_factory=list)
    host_model_id: str | None = None
    provider: str | None = None


class RouteActionTrainingEventsRequest(BaseModel):
    """Generate host-model route/action pairs and optionally store events."""

    schema_version: str = "edgestudio.route_action_training_events_request.v0"
    peer_id: str
    rpp_output: dict = Field(default_factory=dict)
    eval_cases: list[dict] = Field(default_factory=list)
    dry_run: bool = False
    host_model_id: str | None = None
    provider: str | None = None


class RouteActionLearnerDatasetRequest(BaseModel):
    """Build gated route/action learner feed from stored route/action events."""

    schema_version: str = "edgestudio.route_action_learner_dataset_request.v0"
    peer_id: str
    rpp_run_id: str | None = None
    limit: int = 1000


class RouteActionSeedCandidatesRequest(BaseModel):
    """Generate dry-run route/action seed candidates from app tool contracts."""

    schema_version: str = "edgestudio.route_action_seed_candidates_request.v0"
    app_id: str
    tool_registry: list[dict] = Field(default_factory=list)
    golden_cases: list[dict] = Field(default_factory=list)
    rpp_output: dict = Field(default_factory=dict)
    target_seed_count: int = 24
    seed_run_id: str | None = None
    peer_id: str | None = None
    host_model_id: str | None = None
    provider: str | None = None


class HardFactLeakageReviewRequest(BaseModel):
    """Review generated samples for hard-fact leakage with the host model."""

    schema_version: str = "edgestudio.hard_fact_leakage_review_request.v0"
    samples: list[dict] = Field(default_factory=list)
    forbidden_entities: list[str] = Field(default_factory=list)
    host_model_id: str | None = None
    provider: str | None = None


class CorrectionLedgerEntryRequest(BaseModel):
    """Record one local correction ledger entry for later personalization work."""

    schema_version: str = "edgestudio.correction_ledger_entry.v0"
    peer_id: str
    app_id: str = ""
    correction_id: str | None = None
    correction_type: str
    status: str = "recorded"
    source: dict = Field(default_factory=dict)
    target: dict = Field(default_factory=dict)
    correction: dict = Field(default_factory=dict)
    effects: dict = Field(default_factory=dict)
