# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Store and derive lifecycle state from device learning snapshots.

This module is deliberately read-side only. It records what a paired device
reports and derives operator guidance; it does not trigger model pushes,
learning jobs, or regeneration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .app_dirs import data_path


SNAPSHOT_SCHEMA_VERSION = "edgestudio.device_learning_snapshot.v1"
RECEIPT_SCHEMA_VERSION = "edgestudio.device_learning_snapshot_receipt.v1"
LIFECYCLE_SCHEMA_VERSION = "edgestudio.device_learning_lifecycle.v1"
DEFAULT_DATA_READY_THRESHOLD = 10


@dataclass
class DeviceLearningSnapshotError(ValueError):
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


def default_device_snapshot_root() -> Path:
    configured = os.environ.get("EDGE_DEVICE_SNAPSHOT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return data_path("db", "snapshots")


def store_device_learning_snapshot(
    peer_id: str,
    snapshot: dict[str, Any],
    *,
    source: str = "mesh",
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate and persist one latest device learning snapshot."""

    clean_peer_id = _required_id(peer_id, "peer_id")
    if not isinstance(snapshot, dict):
        raise DeviceLearningSnapshotError(
            "invalid_snapshot",
            "snapshot must be a JSON object",
            {"type": type(snapshot).__name__},
        )

    schema_version = str(snapshot.get("schema_version") or "").strip()
    if schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise DeviceLearningSnapshotError(
            "unsupported_schema_version",
            f"unsupported snapshot schema_version: {schema_version}",
            {"expected": SNAPSHOT_SCHEMA_VERSION},
        )

    identity = snapshot.get("identity") if isinstance(snapshot.get("identity"), dict) else {}
    supplied_peer_id = str(identity.get("peer_id") or "").strip()
    if supplied_peer_id and supplied_peer_id != clean_peer_id:
        raise DeviceLearningSnapshotError(
            "peer_id_mismatch",
            "snapshot.identity.peer_id does not match trusted peer",
            {"expected": clean_peer_id, "actual": supplied_peer_id},
        )

    source_clean = _safe_source(source)
    payload_bytes = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    snapshot_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    received_at_ms = int(time.time() * 1000)
    lifecycle = derive_device_lifecycle(snapshot)
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "peer_id": clean_peer_id,
        "source": source_clean,
        "received_at": received_at_ms / 1000.0,
        "snapshot_sha256": snapshot_sha256,
        "lifecycle": lifecycle,
    }
    record = {
        "receipt": receipt,
        "snapshot": snapshot,
    }

    peer_dir = (root or default_device_snapshot_root()).expanduser().resolve() / _path_component(clean_peer_id)
    history_dir = peer_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    latest_path = peer_dir / "latest.json"
    history_path = history_dir / f"{received_at_ms}-{snapshot_sha256[:12]}.json"

    _write_json(latest_path, record)
    _write_json(history_path, record)
    return receipt


def latest_device_learning_snapshot(
    peer_id: str,
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    clean_peer_id = _required_id(peer_id, "peer_id")
    path = (
        (root or default_device_snapshot_root()).expanduser().resolve()
        / _path_component(clean_peer_id)
        / "latest.json"
    )
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeviceLearningSnapshotError(
            "snapshot_store_corrupt",
            f"failed to read latest snapshot for peer {clean_peer_id}",
            {"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("receipt"), dict):
        raise DeviceLearningSnapshotError(
            "snapshot_store_corrupt",
            f"latest snapshot record for peer {clean_peer_id} is invalid",
            {"path": str(path)},
        )
    return data


def derive_device_lifecycle(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Derive an S0-S5 lifecycle phase from a snapshot without side effects."""

    model = _dict(snapshot.get("model"))
    data = _dict(snapshot.get("data"))
    learning = _dict(snapshot.get("learning"))
    corrections = _dict(snapshot.get("corrections"))

    installed_models = model.get("installed_models")
    model_present = bool(
        model.get("selected_model_id")
        or model.get("loaded_model_id")
        or (isinstance(installed_models, list) and installed_models)
    )
    load_state = str(model.get("load_state") or "").strip().lower()
    model_loaded = bool(model.get("loaded_model_id")) or load_state in {"loaded", "ready"}

    facts_classified = _optional_int(data.get("facts_classified")) or 0
    event_total = _optional_int(data.get("event_store_total")) or 0
    readiness = str(data.get("readiness") or "").strip().lower()
    data_ready = readiness in {"enough", "ready", "sufficient"} or max(facts_classified, event_total) >= DEFAULT_DATA_READY_THRESHOLD

    tools_status = _artifact_status(_dict(learning.get("tools_only")))
    rpp_status = _artifact_status(_dict(learning.get("rpp")))
    neural_imprint_status = _artifact_status(_learning_neural_imprint(learning))
    active_artifact_kind = str(learning.get("active_artifact_kind") or "").strip().lower()
    needs_regen = corrections.get("needs_regen") is True or neural_imprint_status in {"stale", "incompatible"}

    if not model_present:
        return _lifecycle(
            "S0",
            "bare_install",
            False,
            ["install_model", "install_a_library"],
            ["no selected, loaded, or installed model was reported"],
        )

    if not model_loaded:
        return _lifecycle(
            "S1",
            "model_present_not_loaded",
            False,
            ["load_model", "fix_model_runtime"],
            [f"model present but load_state={load_state or 'unknown'}"],
        )

    if not data_ready:
        return _lifecycle(
            "S1",
            "model_ready_no_data",
            False,
            ["wait_for_data", "import_data"],
            [f"data below readiness threshold {DEFAULT_DATA_READY_THRESHOLD}"],
        )

    if needs_regen:
        return _lifecycle(
            "S5",
            "stale_or_corrections",
            neural_imprint_status == "active",
            ["consume_corrections", "regenerate_neural_imprint"],
            ["corrections or artifact drift require regeneration"],
        )

    if active_artifact_kind in {"tools_only", "tools_only_kv"} or (
        tools_status == "active" and rpp_status != "active"
    ):
        return _lifecycle(
            "S3",
            "tools_only_ready",
            False,
            ["run_rpp_self_learning", "generate_neural_imprint"],
            ["tools-only KV is active but Neural Imprint is not active"],
        )

    if neural_imprint_status == "active" or (
        neural_imprint_status == "present_inactive" and rpp_status == "active"
    ):
        return _lifecycle(
            "S4",
            "persona_ready",
            neural_imprint_status == "active",
            ["normal_chat", "inspect_profile", "run_eval"],
            ["Neural Imprint is active or ready to activate"],
        )

    return _lifecycle(
        "S2",
        "data_ready_not_learned",
        False,
        ["run_tool_self_learning", "run_rpp_self_learning"],
        ["model and data are ready, but no learning artifact is active"],
    )


def _lifecycle(
    phase: str,
    phase_label: str,
    ready_for_persona_chat: bool,
    recommended_actions: list[str],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "phase": phase,
        "phase_label": phase_label,
        "ready_for_persona_chat": ready_for_persona_chat,
        "recommended_actions": recommended_actions,
        "reasons": reasons,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _artifact_status(value: dict[str, Any]) -> str:
    return str(value.get("status") or "unknown").strip().lower()


def _learning_neural_imprint(learning: dict[str, Any]) -> dict[str, Any]:
    return _dict(learning.get("neural_imprint"))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _required_id(value: Any, field: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise DeviceLearningSnapshotError(
            "missing_required_id",
            f"{field} is required",
            {"field": field},
        )
    return clean


def _safe_source(value: Any) -> str:
    clean = str(value or "mesh").strip().lower()
    return clean if clean in {"mesh", "api", "test"} else "api"


def _path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not safe:
        raise DeviceLearningSnapshotError(
            "invalid_path_component",
            "peer_id cannot be converted to a safe path component",
            {"peer_id": value},
        )
    return safe


def _write_json(path: Path, data: dict[str, Any]) -> None:
    encoded = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)
