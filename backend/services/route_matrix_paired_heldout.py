# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Training-side fixed heldout builder for route-matrix paired evals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROUTE_MATRIX_PAIRED_HELDOUT_SCHEMA_VERSION = (
    "edgestudio.route_matrix_paired_heldout.v0"
)


def build_route_matrix_paired_heldout(
    *,
    dataset_path: Path,
    output_dir: Path,
    run_id: str,
    evidence_scope: str,
    heldout_sample_target: int = 30,
    seed: str = "route_matrix_paired_heldout_v0",
) -> dict[str, Any]:
    """Split a learner dataset into fixed train/heldout JSONL files.

    The split is cluster-level and deterministic. It is meant to create a stable
    paired-eval anchor for v4/v5/... comparisons, not to replace the trainer's
    internal release gates.
    """

    normalized_run_id = _required_text(run_id, "run_id")
    normalized_scope = _required_text(evidence_scope, "evidence_scope")
    target = max(1, int(heldout_sample_target))
    rows = _read_jsonl(Path(dataset_path))
    if not rows:
        raise ValueError("dataset must contain at least one sample")

    clusters = _clusters(rows)
    selected_cluster_keys, warnings = _select_heldout_clusters(
        clusters=clusters,
        target=target,
        seed=str(seed),
        evidence_scope=normalized_scope,
    )
    heldout_rows = [
        row
        for cluster in clusters
        if cluster["cluster_key"] in selected_cluster_keys
        for row in cluster["rows"]
    ]
    train_rows = [
        row
        for cluster in clusters
        if cluster["cluster_key"] not in selected_cluster_keys
        for row in cluster["rows"]
    ]
    if not heldout_rows:
        raise ValueError("heldout split is empty")
    if not train_rows:
        raise ValueError("train split is empty")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "route_action_policy_dataset_train.jsonl"
    heldout_path = output / "route_action_policy_dataset_heldout.jsonl"
    manifest_path = output / "route_matrix_paired_heldout_manifest.json"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(heldout_path, heldout_rows)

    selected_clusters = [
        _cluster_manifest(cluster)
        for cluster in clusters
        if cluster["cluster_key"] in selected_cluster_keys
    ]
    manifest = {
        "ok": True,
        "schema_version": ROUTE_MATRIX_PAIRED_HELDOUT_SCHEMA_VERSION,
        "status": "written",
        "run_id": normalized_run_id,
        "generated_at": _utc_now(),
        "training_side_only": True,
        "writes_runtime_artifacts": False,
        "evidence_scope": normalized_scope,
        "scope_boundaries": {
            "single_app_evidence_is_not_cross_app_generalization": (
                normalized_scope != "cross_app"
            ),
        },
        "seed": str(seed),
        "heldout_sample_target": target,
        "input_path": str(Path(dataset_path)),
        "output_dir": str(output),
        "train_dataset_path": str(train_path),
        "heldout_dataset_path": str(heldout_path),
        "manifest_path": str(manifest_path),
        "sample_count": len(rows),
        "train_sample_count": len(train_rows),
        "heldout_sample_count": len(heldout_rows),
        "cluster_count": len(clusters),
        "heldout_cluster_count": len(selected_clusters),
        "train_intent_counts": _intent_counts(train_rows),
        "heldout_intent_counts": _intent_counts(heldout_rows),
        "warnings": warnings,
        "heldout_clusters": selected_clusters,
        "heldout_sample_ids": sorted(_sample_id(row) for row in heldout_rows),
        "train_sha256": _sha256_path(train_path),
        "heldout_sha256": _sha256_path(heldout_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _select_heldout_clusters(
    *,
    clusters: list[dict[str, Any]],
    target: int,
    seed: str,
    evidence_scope: str,
) -> tuple[set[str], list[str]]:
    warnings: list[str] = []
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        by_intent[cluster["route_intent"]].append(cluster)
    for intent in by_intent:
        by_intent[intent].sort(
            key=lambda cluster: _stable_key(seed, evidence_scope, cluster["cluster_key"])
        )

    selected: set[str] = set()
    remaining_by_intent = {
        intent: len(items)
        for intent, items in by_intent.items()
    }
    heldout_count = 0
    progress = True
    while heldout_count < target and progress:
        progress = False
        for intent in sorted(by_intent):
            for cluster in by_intent[intent]:
                if cluster["cluster_key"] in selected:
                    continue
                if remaining_by_intent[intent] <= 1:
                    continue
                selected.add(cluster["cluster_key"])
                remaining_by_intent[intent] -= 1
                heldout_count += len(cluster["rows"])
                progress = True
                break
            if heldout_count >= target:
                break
    if heldout_count < target:
        warnings.append("heldout_target_not_reached_without_emptying_intent_train_clusters")
    return selected, warnings


def _clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        intent = _intent(row)
        grouped[f"{_case_id(row)}|intent:{intent}"].append(row)
    clusters = []
    for key, cluster_rows in grouped.items():
        intents = sorted({_intent(row) for row in cluster_rows})
        clusters.append(
            {
                "cluster_key": key,
                "route_intent": intents[0] if intents else "<missing>",
                "rows": sorted(cluster_rows, key=lambda row: _sample_id(row)),
            }
        )
    return sorted(clusters, key=lambda cluster: cluster["cluster_key"])


def _cluster_manifest(cluster: dict[str, Any]) -> dict[str, Any]:
    rows = cluster["rows"]
    return {
        "cluster_key": cluster["cluster_key"],
        "route_intent": cluster["route_intent"],
        "sample_count": len(rows),
        "sample_ids": sorted(_sample_id(row) for row in rows),
        "case_ids": sorted({_case_id(row) for row in rows}),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _intent_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(_intent(row) for row in rows).items()))


def _intent(row: dict[str, Any]) -> str:
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    return str(target.get("route_intent") or "<missing>").strip() or "<missing>"


def _case_id(row: dict[str, Any]) -> str:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    case_id = str(source.get("case_id") or "").strip()
    if case_id:
        return f"case:{case_id}"
    return f"sample:{_sample_id(row)}"


def _sample_id(row: dict[str, Any]) -> str:
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    text = ""
    input_value = row.get("input") if isinstance(row.get("input"), dict) else {}
    if isinstance(input_value.get("text"), str):
        text = input_value["text"]
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_key(seed: str, evidence_scope: str, cluster_key: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "seed": seed,
                "evidence_scope": evidence_scope,
                "cluster_key": cluster_key,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
