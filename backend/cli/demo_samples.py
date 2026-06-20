# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Synthetic demo sample resolvers for Edge CLI dry-run planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from backend.cli.fingerprints import canonical_json_bytes, sha256_prefixed
from backend.services.persona_rpp_input_contract import (
    INPUT_SCHEMA_VERSION,
    records_sha256,
)


SAMPLE_SCHEMA_VERSION = "edge.demo.sample.v1"


@dataclass(frozen=True)
class DemoSample:
    sample_id: str
    profile_body: str
    tool_schema_export: dict[str, Any]

    @property
    def sample_sha256(self) -> str:
        payload = {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "profile_body_sha256": self.profile_body_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
        }
        return sha256_prefixed(canonical_json_bytes(payload))

    @property
    def profile_body_sha256(self) -> str:
        return sha256_prefixed(self.profile_body.encode("utf-8"))

    @property
    def tool_schema_sha256(self) -> str:
        return sha256_prefixed(canonical_json_bytes(self.tool_schema_export))

    def as_plan_summary(self) -> dict[str, object]:
        return {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "sample_sha256": self.sample_sha256,
            "profile_body_sha256": self.profile_body_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "source_kind": "host_rpp_profile",
            "raw_text_included": False,
        }


@dataclass(frozen=True)
class LearnDemoSample:
    sample_id: str
    peer_id: str
    app_id: str
    base_model_id: str
    question: str
    records: list[dict[str, Any]]
    corrections: list[dict[str, Any]]
    tool_schema_export: dict[str, Any]

    @property
    def rpp_input_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INPUT_SCHEMA_VERSION,
            "peer_id": self.peer_id,
            "app_id": self.app_id,
            "base_model_id": self.base_model_id,
            "source_kind": "app_facts",
            "created_at": 1779707079.0,
            "records": self.records,
            "records_sha256": records_sha256(self.records),
        }

    @property
    def sample_sha256(self) -> str:
        payload = {
            "schema_version": "edge.demo.learn.sample.v1",
            "sample_id": self.sample_id,
            "rpp_input_sha256": self.rpp_input_sha256,
            "correction_pack_sha256": self.correction_pack_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "question_sha256": self.question_sha256,
        }
        return sha256_prefixed(canonical_json_bytes(payload))

    @property
    def rpp_input_sha256(self) -> str:
        return sha256_prefixed(canonical_json_bytes(self.rpp_input_payload))

    @property
    def correction_pack_sha256(self) -> str:
        return sha256_prefixed(canonical_json_bytes(self.corrections))

    @property
    def tool_schema_sha256(self) -> str:
        return sha256_prefixed(canonical_json_bytes(self.tool_schema_export))

    @property
    def question_sha256(self) -> str:
        return sha256_prefixed(self.question.encode("utf-8"))

    def as_plan_summary(self) -> dict[str, object]:
        return {
            "schema_version": "edge.demo.learn.sample.v1",
            "sample_id": self.sample_id,
            "sample_sha256": self.sample_sha256,
            "rpp_input_sha256": self.rpp_input_sha256,
            "correction_pack_sha256": self.correction_pack_sha256,
            "tool_schema_sha256": self.tool_schema_sha256,
            "question_sha256": self.question_sha256,
            "source_kind": "synthetic_correction_fixture",
            "record_count": len(self.records),
            "correction_count": len(self.corrections),
            "correction_types": sorted(
                {str(item.get("correction_type") or "") for item in self.corrections}
            ),
            "raw_text_included": False,
        }

    def as_text_preview(self) -> dict[str, object]:
        return {
            "question": self.question,
            "records": self.records,
            "corrections": self.corrections,
        }


def resolve_demo_sample(sample_id: str) -> DemoSample:
    normalized = sample_id.strip().replace("-", "_")
    if normalized == "synthetic_profile_v1":
        return _synthetic_profile_v1()
    raise ValueError(f"unknown demo sample: {sample_id}")


def resolve_learn_demo_sample(sample_id: str) -> LearnDemoSample:
    normalized = sample_id.strip().replace("-", "_")
    if normalized == "finance_conservative_cashflow_v1":
        return _finance_conservative_cashflow_v1()
    if normalized == "synthetic_profile_correction_v1":
        return _synthetic_profile_correction_v1()
    raise ValueError(f"unknown learn demo sample: {sample_id}")


def list_demo_samples() -> list[dict[str, object]]:
    sample = _synthetic_profile_v1()
    return [sample.as_plan_summary()]


def _synthetic_profile_v1() -> DemoSample:
    tool_schema_export = {
        "schema_version": "edgestudio.tool_schema_export.v1",
        "tools": [
            {
                "name": "sample_profile_facts_lookup",
                "description": "Read-only lookup for synthetic profile facts.",
                "permissions": ["read_facts"],
                "intentTags": ["exact_fact"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Synthetic profile topic to inspect.",
                        }
                    },
                },
            },
            {
                "name": "sample_profile_summary",
                "description": "Read-only aggregate summary for the synthetic profile sample.",
                "permissions": ["read_facts"],
                "intentTags": ["aggregate_fact"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "description": "Synthetic profile scope to summarize.",
                        }
                    },
                },
            },
        ],
    }
    profile_body = json.dumps(
        {
            "sample": "synthetic_profile_v1",
            "traits": [
                "prefers concise technical answers",
                "asks for local-only receipts before trusting a workflow",
                "wants explicit fail-closed behavior when prerequisites are missing",
            ],
            "boundaries": [
                "synthetic sample only",
                "no real user data",
                "no domain-specific business logic",
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return DemoSample("synthetic_profile_v1", profile_body, tool_schema_export)


def _synthetic_profile_correction_v1() -> LearnDemoSample:
    tool_schema_export = {
        "schema_version": "edgestudio.tool_schema_export.v1",
        "tools": [
            {
                "name": "sample_profile_facts_lookup",
                "description": "Read-only lookup for synthetic profile facts.",
                "permissions": ["read_facts"],
                "intentTags": ["exact_fact"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Synthetic profile topic to inspect.",
                        }
                    },
                },
            },
            {
                "name": "sample_profile_summary",
                "description": "Read-only aggregate summary for the synthetic profile sample.",
                "permissions": ["read_facts"],
                "intentTags": ["aggregate_fact"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "description": "Synthetic profile scope to summarize.",
                        }
                    },
                },
            },
        ],
    }
    records = [
        {
            "record_id": "profile-001",
            "kind": "preference",
            "text": "The synthetic user prefers concise technical answers.",
            "tags": ["synthetic", "style"],
        },
        {
            "record_id": "profile-002",
            "kind": "trust_boundary",
            "text": "The synthetic user asks for local-only receipts before trusting a workflow.",
            "tags": ["synthetic", "privacy"],
        },
    ]
    corrections = [
        {
            "peer_id": "edge-demo-learn-peer",
            "app_id": "com.atomgradient.edge.demo.synthetic",
            "correction_type": "profile_correction",
            "target": {"profile_field": "answer_style"},
            "correction": {
                "profile_overlay": {
                    "style": "short direct summaries",
                    "boundary": "avoid quality claims without evidence",
                }
            },
            "status": "recorded",
        }
    ]
    return LearnDemoSample(
        sample_id="synthetic_profile_correction_v1",
        peer_id="edge-demo-learn-peer",
        app_id="com.atomgradient.edge.demo.synthetic",
        base_model_id="qwen3.5-9b-4bit",
        question="How should this assistant respond to technical workflow questions?",
        records=records,
        corrections=corrections,
        tool_schema_export=tool_schema_export,
    )


def _finance_conservative_cashflow_v1() -> LearnDemoSample:
    tool_schema_export = {
        "schema_version": "edgestudio.tool_schema_export.v1",
        "tools": [
            {
                "name": "sample_finance_facts_lookup",
                "description": "Read-only lookup for synthetic finance preference and cash-flow facts.",
                "permissions": ["read_facts"],
                "intentTags": ["exact_fact", "finance"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Synthetic finance topic to inspect, such as risk_boundary, cashflow, or trust_boundary.",
                        }
                    },
                },
            },
            {
                "name": "sample_finance_cashflow_summary",
                "description": "Read-only aggregate summary for the synthetic finance sample.",
                "permissions": ["read_facts"],
                "intentTags": ["aggregate_fact", "finance"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "description": "Synthetic finance scope to summarize.",
                        }
                    },
                },
            },
        ],
    }
    records = [
        {
            "record_id": "finance-001",
            "kind": "explicit_preference",
            "text": "The synthetic user avoids high-risk recommendations and prefers stable, cash-flow-aware guidance.",
            "tags": ["synthetic", "finance", "risk_boundary"],
        },
        {
            "record_id": "finance-002",
            "kind": "cashflow_context",
            "text": "The synthetic user's rent and fixed subscriptions are already covered; they have $800 left after bills this month.",
            "tags": ["synthetic", "finance", "cashflow"],
        },
        {
            "record_id": "finance-003",
            "kind": "trust_boundary",
            "text": "The synthetic user wants cash-flow impact explained before any recommendation and does not want unsupported return claims.",
            "tags": ["synthetic", "finance", "trust_boundary"],
        },
    ]
    corrections = [
        {
            "peer_id": "edge-demo-learn-peer",
            "app_id": "com.atomgradient.edge.demo.finance",
            "correction_type": "profile_correction",
            "target": {"profile_field": "financial_guidance_style"},
            "correction": {
                "profile_overlay": {
                    "risk_style": "avoid high-risk or leveraged recommendations",
                    "priority": "cash-flow stability first",
                    "answer_style": "explain conservative options before any upside discussion",
                    "boundary": "no financial return claims without user-provided facts",
                }
            },
            "status": "recorded",
        }
    ]
    return LearnDemoSample(
        sample_id="finance_conservative_cashflow_v1",
        peer_id="edge-demo-learn-peer",
        app_id="com.atomgradient.edge.demo.finance",
        base_model_id="qwen3.5-9b-4bit",
        question="I have $800 left after bills this month. What should I do with it?",
        records=records,
        corrections=corrections,
        tool_schema_export=tool_schema_export,
    )
