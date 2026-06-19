# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Store host-model profile naming output as RPP B-naming artifacts.

The host model owns the profile labels and narratives. This module only
serializes a successful `generate_profile_naming(...)` envelope into the local
RPP artifact store so the later adapter/RPP atomic bundle can carry the same
run's B directions and B naming together.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.services.rpp_artifact_store import store_rpp_artifact_upload


PROFILE_NAMING_ARTIFACT_SCHEMA_VERSION = "edgestudio.profile_naming_artifact.v0"
PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION = (
    "edgestudio.profile_naming_artifact_receipt.v0"
)


def build_profile_naming_artifact(
    profile_naming_response: dict[str, Any],
    *,
    layer_id: int | None = None,
) -> dict[str, Any]:
    """Build one `rpp_b_naming` artifact from a successful host-model envelope."""

    result = _profile_naming_result(profile_naming_response)
    rpp_run_id = _required_str(result.get("rpp_run_id"), "result.rpp_run_id")
    directions = result.get("directions")
    if not isinstance(directions, list):
        raise ValueError("result.directions must be a list")

    normalized_layer_id = _optional_positive_int(layer_id)
    if normalized_layer_id is None:
        normalized_layer_id = 23

    content = []
    for index, raw_direction in enumerate(directions):
        if not isinstance(raw_direction, dict):
            raise ValueError(f"result.directions[{index}] must be an object")
        name = _required_str(
            raw_direction.get("name"),
            f"result.directions[{index}].name",
        )
        reason = _required_str(
            raw_direction.get("reason"),
            f"result.directions[{index}].reason",
        )
        direction_idx = _optional_positive_int(raw_direction.get("direction_idx"))
        if direction_idx is None:
            raise ValueError(
                f"result.directions[{index}].direction_idx must be a positive int"
            )
        direction_id = _required_str(
            raw_direction.get("direction_id"),
            f"result.directions[{index}].direction_id",
        )
        content.append(
            {
                "schema_version": PROFILE_NAMING_ARTIFACT_SCHEMA_VERSION,
                "rpp_run_id": rpp_run_id,
                "profile_name": result.get("profile_name") or "",
                "profile_summary": result.get("profile_summary") or "",
                "direction_idx": direction_idx,
                "direction_id": direction_id,
                "name": name,
                "reason": reason,
                "raw_response": f"Name: {name}\nReason: {reason}",
                "confidence": raw_direction.get("confidence"),
                "evidence_refs": raw_direction.get("evidence_refs")
                if isinstance(raw_direction.get("evidence_refs"), list)
                else [],
                "source": raw_direction.get("source")
                if isinstance(raw_direction.get("source"), dict)
                else {},
            }
        )

    data = json.dumps(
        content,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    return {
        "name": f"B_naming_layer_{normalized_layer_id}.json",
        "role": "rpp_b_naming",
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_base64": base64.b64encode(data).decode("ascii"),
        "content": content,
    }


def store_profile_naming_artifact(
    profile_naming_response: dict[str, Any],
    *,
    peer_id: str,
    layer_id: int | None = None,
    rpp_metadata: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Store a successful profile naming envelope as a same-run RPP artifact."""

    metadata = rpp_metadata if isinstance(rpp_metadata, dict) else {}
    result = _profile_naming_result(profile_naming_response)
    rpp_run_id = _required_str(result.get("rpp_run_id"), "result.rpp_run_id")
    artifact = build_profile_naming_artifact(
        profile_naming_response,
        layer_id=layer_id or _metadata_layer_id(metadata),
    )
    upload_payload = {
        "peer_id": peer_id,
        "rpp_run_id": rpp_run_id,
        "base_model_id": metadata.get("base_model_id") or metadata.get("base_model"),
        "layer_id": layer_id or _metadata_layer_id(metadata),
        "a_version": metadata.get("a_version"),
        "a_hash": metadata.get("a_hash"),
        "dataset_summary": metadata.get("dataset_summary") or {},
        "artifacts": [
            {
                key: value
                for key, value in artifact.items()
                if key != "content"
            }
        ],
    }
    storage = store_rpp_artifact_upload(upload_payload, root=root)
    return {
        "ok": True,
        "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
        "status": "stored",
        "peer_id": peer_id,
        "rpp_run_id": rpp_run_id,
        "profile_naming_response": profile_naming_response,
        "artifact": {
            "name": artifact["name"],
            "role": artifact["role"],
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact["sha256"],
        },
        "storage": storage,
    }


def generate_and_store_profile_naming_artifact(
    *,
    rpp_output: dict[str, Any],
    peer_id: str,
    forbidden_entities: list[str] | None = None,
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: Any | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Generate profile naming with the host model, then store B-naming JSON."""

    from backend.services.host_model_assistant import (
        HOST_MODEL_PROVIDER,
        generate_profile_naming,
    )

    profile_naming_response = generate_profile_naming(
        rpp_output,
        forbidden_entities=forbidden_entities,
        host_model_id=host_model_id,
        provider=provider or HOST_MODEL_PROVIDER,
        host_model_generate=host_model_generate,
    )
    if profile_naming_response.get("ok") is not True:
        return {
            "ok": False,
            "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "status": "profile_naming_generation_failed",
            "peer_id": peer_id,
            "profile_naming_response": profile_naming_response,
            "storage": None,
        }

    try:
        return store_profile_naming_artifact(
            profile_naming_response,
            peer_id=peer_id,
            layer_id=_metadata_layer_id(rpp_output),
            rpp_metadata=rpp_output,
            root=root,
        )
    except ValueError as exc:
        return {
            "ok": False,
            "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "status": "profile_naming_artifact_failed",
            "peer_id": peer_id,
            "profile_naming_response": profile_naming_response,
            "storage": None,
            "error": {
                "code": "invalid_profile_naming_artifact",
                "message": str(exc),
                "retryable": False,
                "details": {},
            },
        }


def generate_and_store_profile_naming_artifact_from_latest_rpp(
    *,
    peer_id: str,
    forbidden_entities: list[str] | None = None,
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: Any | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Generate B-naming from the latest stored RPP artifact run for a peer."""

    from backend.services.rpp_artifact_store import latest_rpp_artifact_run_for_peer

    latest = latest_rpp_artifact_run_for_peer(peer_id, root=root)
    if latest is None:
        return {
            "ok": False,
            "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "status": "missing_rpp_artifact_run",
            "peer_id": peer_id,
            "profile_naming_response": None,
            "storage": None,
            "error": {
                "code": "missing_rpp_artifact_run",
                "message": "No stored RPP artifact run is available for this peer.",
                "retryable": False,
                "details": {"peer_id": peer_id},
            },
        }

    rpp_last_run_path = latest.path / "rpp_last_run.json"
    try:
        rpp_output = json.loads(rpp_last_run_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "status": "missing_rpp_last_run",
            "peer_id": peer_id,
            "rpp_run_id": latest.rpp_run_id,
            "profile_naming_response": None,
            "storage": None,
            "error": {
                "code": "missing_rpp_last_run",
                "message": "Stored RPP run has no readable rpp_last_run.json.",
                "retryable": False,
                "details": {
                    "peer_id": peer_id,
                    "rpp_run_id": latest.rpp_run_id,
                    "path": str(rpp_last_run_path),
                    "reason": str(exc),
                },
            },
        }
    if not isinstance(rpp_output, dict):
        return {
            "ok": False,
            "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "status": "invalid_rpp_last_run",
            "peer_id": peer_id,
            "rpp_run_id": latest.rpp_run_id,
            "profile_naming_response": None,
            "storage": None,
            "error": {
                "code": "invalid_rpp_last_run",
                "message": "Stored rpp_last_run.json must be a JSON object.",
                "retryable": False,
                "details": {"peer_id": peer_id, "rpp_run_id": latest.rpp_run_id},
            },
        }

    embedded_run_id = str(rpp_output.get("rpp_run_id") or "").strip()
    if embedded_run_id and embedded_run_id != latest.rpp_run_id:
        return {
            "ok": False,
            "schema_version": PROFILE_NAMING_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            "status": "rpp_run_id_mismatch",
            "peer_id": peer_id,
            "rpp_run_id": latest.rpp_run_id,
            "profile_naming_response": None,
            "storage": None,
            "error": {
                "code": "rpp_run_id_mismatch",
                "message": "Stored rpp_last_run.json does not match receipt rpp_run_id.",
                "retryable": False,
                "details": {
                    "expected": latest.rpp_run_id,
                    "actual": embedded_run_id,
                },
            },
        }
    rpp_output.setdefault("rpp_run_id", latest.rpp_run_id)
    _enrich_rpp_output_from_receipt(rpp_output, latest.receipt)

    receipt = generate_and_store_profile_naming_artifact(
        rpp_output=rpp_output,
        peer_id=peer_id,
        forbidden_entities=forbidden_entities,
        host_model_id=host_model_id,
        provider=provider,
        host_model_generate=host_model_generate,
        root=root,
    )
    source = receipt.get("source")
    if not isinstance(source, dict):
        source = {}
        receipt["source"] = source
    source.update(
        {
            "kind": "latest_rpp_artifact_run",
            "rpp_run_id": latest.rpp_run_id,
            "storage_path": str(latest.path),
        }
    )
    return receipt


def _enrich_rpp_output_from_receipt(
    rpp_output: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    for source_key, target_key in (
        ("base_model_id", "base_model_id"),
        ("layer_id", "layer_id"),
        ("a_version", "a_version"),
        ("a_hash", "a_hash"),
        ("dataset_summary", "dataset_summary"),
    ):
        value = receipt.get(source_key)
        if value not in (None, "", {}) and rpp_output.get(target_key) in (
            None,
            "",
            {},
        ):
            rpp_output[target_key] = value


def _profile_naming_result(response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("profile_naming_response must be an object")
    if response.get("ok") is not True:
        raise ValueError("profile_naming_response.ok must be true")
    result = response.get("result")
    if not isinstance(result, dict):
        raise ValueError("profile_naming_response.result must be an object")
    return result


def _metadata_layer_id(metadata: dict[str, Any]) -> int | None:
    for key in ("layer_id", "layer_idx", "target_layer"):
        found = _optional_positive_int(metadata.get(key))
        if found is not None:
            return found
    return None


def _required_str(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit() and int(stripped) > 0:
            return int(stripped)
    return None
