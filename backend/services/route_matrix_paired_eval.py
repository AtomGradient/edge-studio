# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Training-side same-heldout evaluator for route-matrix retrains."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from backend.services.route_matrix_paired_heldout import (
    ROUTE_MATRIX_PAIRED_HELDOUT_SCHEMA_VERSION,
)
from backend.services.route_router_training import (
    RouteRouterEmbeddingProvider,
    RouteRouterTrainConfig,
    _fit_intent_calibration,
    _fit_intent_head,
    _intent_split_metrics,
    _sample_prompt,
    _sample_target,
    build_route_router_training_splits,
    load_route_action_policy_dataset_jsonl,
)


ROUTE_MATRIX_PAIRED_EVAL_SCHEMA_VERSION = "edgestudio.route_matrix_paired_eval.v0"


def build_route_matrix_paired_eval(
    *,
    heldout_manifest_path: Path,
    baseline_dataset_path: Path,
    candidate_dataset_path: Path,
    run_id: str,
    baseline_run_id: str,
    candidate_run_id: str,
    config: RouteRouterTrainConfig,
    embedding_provider: RouteRouterEmbeddingProvider,
    real_correction_count: int = 0,
    min_significant_paired_corrections: int = 30,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    """Evaluate baseline and candidate train sets on the same fixed heldout."""

    generated_at_ms = int(time.time() * 1000)
    manifest = _load_manifest(Path(heldout_manifest_path))
    heldout_status = _verify_manifest(manifest)
    if heldout_status is not None:
        return _blocked_receipt(
            status=heldout_status["status"],
            error=heldout_status,
            run_id=run_id,
            generated_at_ms=generated_at_ms,
            manifest=manifest,
        )

    heldout_path = Path(manifest["heldout_dataset_path"])
    heldout_rows = load_route_action_policy_dataset_jsonl(heldout_path)
    baseline_rows = load_route_action_policy_dataset_jsonl(Path(baseline_dataset_path))
    candidate_rows = load_route_action_policy_dataset_jsonl(Path(candidate_dataset_path))
    overlap = _train_heldout_overlap(
        heldout_rows=heldout_rows,
        train_rows_by_name={
            "baseline": baseline_rows,
            "candidate": candidate_rows,
        },
    )
    if overlap["overlap_count"] > 0:
        return _blocked_receipt(
            status="train_heldout_overlap_blocked",
            error=overlap,
            run_id=run_id,
            generated_at_ms=generated_at_ms,
            manifest=manifest,
        )

    baseline = _train_and_eval(
        rows=baseline_rows,
        heldout_rows=heldout_rows,
        run_id=baseline_run_id,
        config=config,
        embedding_provider=embedding_provider,
    )
    if baseline["ok"] is not True:
        return _blocked_receipt(
            status="baseline_training_blocked",
            error=baseline,
            run_id=run_id,
            generated_at_ms=generated_at_ms,
            manifest=manifest,
        )
    candidate = _train_and_eval(
        rows=candidate_rows,
        heldout_rows=heldout_rows,
        run_id=candidate_run_id,
        config=config,
        embedding_provider=embedding_provider,
    )
    if candidate["ok"] is not True:
        return _blocked_receipt(
            status="candidate_training_blocked",
            error=candidate,
            run_id=run_id,
            generated_at_ms=generated_at_ms,
            manifest=manifest,
        )

    paired = _paired_metrics(
        baseline=baseline["fixed_heldout"],
        candidate=candidate["fixed_heldout"],
        seed=f"{run_id}:{manifest['heldout_sha256']}",
        bootstrap_iterations=bootstrap_iterations,
    )
    correction_count = max(0, int(real_correction_count))
    min_corrections = max(1, int(min_significant_paired_corrections))
    evidence_status = _evidence_status(
        real_correction_count=correction_count,
        min_significant_paired_corrections=min_corrections,
        paired=paired,
    )
    non_release_summary = _paired_non_release_summary_fields(
        evidence_status=evidence_status,
        manifest=manifest,
        runtime_release_blockers=["paired_eval_is_training_side_metrics_only"],
        correction_count=correction_count,
        min_corrections=min_corrections,
    )
    return {
        "ok": True,
        "schema_version": ROUTE_MATRIX_PAIRED_EVAL_SCHEMA_VERSION,
        "status": "evaluated",
        "run_id": run_id,
        "generated_at_ms": generated_at_ms,
        "result": {
            "summary": {
                "evidence_status": evidence_status,
                "evidence_scope": manifest["evidence_scope"],
                "scope_boundaries": manifest.get("scope_boundaries", {}),
                "heldout_sample_count": len(heldout_rows),
                "real_correction_count": correction_count,
                "min_significant_paired_corrections": min_corrections,
                "claim_evidence_volume_sufficient": evidence_status
                in {
                    "candidate_improved_on_paired_heldout",
                    "candidate_regressed_on_paired_heldout",
                    "paired_delta_not_significant",
                },
                "claim_evidence_volume_reason": (
                    "requires_min_significant_paired_corrections_and_paired_ci"
                ),
                "ready_for_live_routing": False,
                "ready_for_live_routing_reason": (
                    "paired_eval_is_training_side_metrics_only"
                ),
                **non_release_summary,
            },
            "heldout": {
                "manifest_path": str(Path(heldout_manifest_path)),
                "heldout_dataset_path": str(heldout_path),
                "heldout_sha256": manifest["heldout_sha256"],
                "manifest_run_id": manifest["run_id"],
                "heldout_intent_counts": manifest.get("heldout_intent_counts", {}),
            },
            "baseline": baseline,
            "candidate": candidate,
            "paired": paired,
        },
        "error": None,
        "audit": {
            "method": "build_route_matrix_paired_eval",
            "training_side_only": True,
            "writes_events": False,
            "writes_runtime_artifacts": False,
            "writes_training_sample_tags": False,
            "verified_heldout_sha256": True,
        },
    }


def _train_and_eval(
    *,
    rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    run_id: str,
    config: RouteRouterTrainConfig,
    embedding_provider: RouteRouterEmbeddingProvider,
) -> dict[str, Any]:
    splits = build_route_router_training_splits(
        rows,
        training_run_id=run_id,
        min_calibration_samples=config.min_calibration_set_size,
    )
    if splits["leakage_gate"]["ok"] is not True:
        return {
            "ok": False,
            "status": "split_leakage_blocked",
            "leakage_gate": splits["leakage_gate"],
        }
    train_rows = splits["splits"]["train"]
    calibration_rows = splits["splits"]["calibration"]
    if not train_rows:
        return {"ok": False, "status": "insufficient_train_samples"}
    if not calibration_rows:
        return {"ok": False, "status": "calibration_missing"}
    train_embeddings = _embedding_matrix(
        embedding_provider=embedding_provider,
        rows=train_rows,
        config=config,
        field="paired_eval_train_embeddings",
    )
    train_labels = [_sample_target(row)["route_intent"] for row in train_rows]
    weights, bias = _fit_intent_head(
        train_embeddings,
        train_labels,
        config=config,
    )
    calibration = _fit_intent_calibration(
        embedding_provider=embedding_provider,
        rows=calibration_rows,
        weights=weights,
        bias=bias,
        config=config,
    )
    fixed_heldout = _intent_split_metrics(
        embedding_provider,
        heldout_rows,
        weights,
        bias,
        config.hidden_size,
        list(config.intent_vocab),
        float(calibration["intent_temperature"]),
        "paired_heldout",
    )
    return {
        "ok": True,
        "status": "evaluated",
        "run_id": run_id,
        "train_sample_count": len(rows),
        "internal_split_summary": splits["summary"],
        "calibration": calibration,
        "fixed_heldout": fixed_heldout,
    }


def _embedding_matrix(
    *,
    embedding_provider: RouteRouterEmbeddingProvider,
    rows: list[dict[str, Any]],
    config: RouteRouterTrainConfig,
    field: str,
) -> Any:
    from backend.services.route_router_training import _embedding_matrix as build_matrix

    return build_matrix(
        embedding_provider,
        [_sample_prompt(row) for row in rows],
        config.hidden_size,
        field,
    )


def _paired_metrics(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    seed: str,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    baseline_predictions = baseline.get("predictions") or []
    candidate_predictions = candidate.get("predictions") or []
    if len(baseline_predictions) != len(candidate_predictions):
        raise ValueError("baseline and candidate heldout predictions differ in length")
    rows = []
    baseline_correct: list[float] = []
    candidate_correct: list[float] = []
    for left, right in zip(baseline_predictions, candidate_predictions):
        if left.get("prompt") != right.get("prompt") or left.get("expected") != right.get("expected"):
            raise ValueError("baseline and candidate heldout predictions are not aligned")
        left_correct = left.get("predicted") == left.get("expected")
        right_correct = right.get("predicted") == right.get("expected")
        baseline_correct.append(1.0 if left_correct else 0.0)
        candidate_correct.append(1.0 if right_correct else 0.0)
        rows.append({
            "prompt": left.get("prompt"),
            "expected": left.get("expected"),
            "baseline_predicted": left.get("predicted"),
            "candidate_predicted": right.get("predicted"),
            "baseline_correct": left_correct,
            "candidate_correct": right_correct,
        })
    baseline_acc = _mean(baseline_correct)
    candidate_acc = _mean(candidate_correct)
    delta = candidate_acc - baseline_acc
    ci = _bootstrap_delta_ci(
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
        seed=seed,
        iterations=bootstrap_iterations,
    )
    return {
        "sample_count": len(rows),
        "baseline_intent_acc": baseline_acc,
        "candidate_intent_acc": candidate_acc,
        "intent_acc_delta": delta,
        "bootstrap_iterations": max(0, int(bootstrap_iterations)),
        "intent_acc_delta_ci95": ci,
        "predictions": rows,
    }


def _bootstrap_delta_ci(
    *,
    baseline_correct: list[float],
    candidate_correct: list[float],
    seed: str,
    iterations: int,
) -> dict[str, float | None]:
    n = len(baseline_correct)
    count = max(0, int(iterations))
    if n == 0 or count <= 0:
        return {"low": None, "high": None}
    rng = random.Random(int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16))
    deltas = []
    for _ in range(count):
        baseline_sum = 0.0
        candidate_sum = 0.0
        for _ in range(n):
            idx = rng.randrange(n)
            baseline_sum += baseline_correct[idx]
            candidate_sum += candidate_correct[idx]
        deltas.append((candidate_sum / n) - (baseline_sum / n))
    deltas.sort()
    return {
        "low": float(deltas[int(0.025 * (count - 1))]),
        "high": float(deltas[int(0.975 * (count - 1))]),
    }


def _evidence_status(
    *,
    real_correction_count: int,
    min_significant_paired_corrections: int,
    paired: dict[str, Any],
) -> str:
    if real_correction_count < min_significant_paired_corrections:
        return "insufficient_paired_corrections"
    ci = paired.get("intent_acc_delta_ci95") or {}
    low = ci.get("low")
    high = ci.get("high")
    delta = float(paired.get("intent_acc_delta") or 0.0)
    if low is None or high is None or float(low) <= 0.0 <= float(high):
        return "paired_delta_not_significant"
    if delta > 0:
        return "candidate_improved_on_paired_heldout"
    if delta < 0:
        return "candidate_regressed_on_paired_heldout"
    return "paired_delta_not_significant"


def _paired_non_release_summary_fields(
    *,
    evidence_status: str,
    manifest: dict[str, Any] | None,
    runtime_release_blockers: list[str],
    correction_count: int = 0,
    min_corrections: int = 1,
) -> dict[str, Any]:
    return {
        "training_side_evidence_candidate_ready": (
            evidence_status == "candidate_improved_on_paired_heldout"
        ),
        "training_side_evidence_candidate_reason": evidence_status,
        "runtime_release_ready": False,
        "runtime_release_blockers": list(runtime_release_blockers),
        "diagnostic_warnings": _paired_diagnostic_warnings(
            manifest=manifest,
            correction_count=correction_count,
            min_corrections=min_corrections,
        ),
        "production_router_improved": False,
        "scope_bounded_to": _paired_scope_bounded_to(manifest),
    }


def _paired_diagnostic_warnings(
    *,
    manifest: dict[str, Any] | None,
    correction_count: int,
    min_corrections: int,
) -> list[str]:
    warnings: set[str] = set()
    if correction_count < min_corrections:
        warnings.add("insufficient_paired_corrections")
    if isinstance(manifest, dict):
        boundaries = manifest.get("scope_boundaries")
        if isinstance(boundaries, dict):
            for name, active in boundaries.items():
                if active is True:
                    warnings.add(str(name))
        warnings.update(str(item) for item in manifest.get("warnings") or [])
    return sorted(warning for warning in warnings if warning)


def _paired_scope_bounded_to(manifest: dict[str, Any] | None) -> list[str]:
    if not isinstance(manifest, dict):
        return []
    scope = str(manifest.get("evidence_scope") or "").strip()
    if not scope:
        return []
    if scope.endswith("_only"):
        return [scope[:-5]]
    return [scope]


def _verify_manifest(manifest: dict[str, Any]) -> dict[str, Any] | None:
    if manifest.get("schema_version") != ROUTE_MATRIX_PAIRED_HELDOUT_SCHEMA_VERSION:
        return {
            "status": "unsupported_heldout_manifest",
            "schema_version": manifest.get("schema_version"),
        }
    heldout_path = Path(str(manifest.get("heldout_dataset_path") or ""))
    if not heldout_path.exists():
        return {
            "status": "heldout_dataset_missing",
            "heldout_dataset_path": str(heldout_path),
        }
    expected = str(manifest.get("heldout_sha256") or "").strip()
    actual = hashlib.sha256(heldout_path.read_bytes()).hexdigest()
    if expected != actual:
        return {
            "status": "heldout_sha256_mismatch",
            "expected": expected,
            "actual": actual,
            "heldout_dataset_path": str(heldout_path),
        }
    return None


def _train_heldout_overlap(
    *,
    heldout_rows: list[dict[str, Any]],
    train_rows_by_name: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    heldout_sample_ids = {_sample_id(row) for row in heldout_rows}
    heldout_cluster_keys = {_cluster_key(row) for row in heldout_rows}
    overlaps = []
    for name, rows in sorted(train_rows_by_name.items()):
        sample_hits = sorted(heldout_sample_ids & {_sample_id(row) for row in rows})
        cluster_hits = sorted(heldout_cluster_keys & {_cluster_key(row) for row in rows})
        if sample_hits or cluster_hits:
            overlaps.append({
                "dataset": name,
                "sample_id_overlaps": sample_hits,
                "cluster_key_overlaps": cluster_hits,
            })
    return {
        "status": "passed" if not overlaps else "train_heldout_overlap_blocked",
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }


def _blocked_receipt(
    *,
    status: str,
    error: dict[str, Any],
    run_id: str,
    generated_at_ms: int,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_scope = (
        manifest.get("evidence_scope") if isinstance(manifest, dict) else None
    )
    return {
        "ok": False,
        "schema_version": ROUTE_MATRIX_PAIRED_EVAL_SCHEMA_VERSION,
        "status": status,
        "run_id": run_id,
        "result": {
            "summary": {
                "evidence_status": status,
                "evidence_scope": evidence_scope,
                "ready_for_live_routing": False,
                **_paired_non_release_summary_fields(
                    evidence_status=status,
                    manifest=manifest,
                    runtime_release_blockers=[f"paired_eval_blocked:{status}"],
                ),
            }
        },
        "error": error,
        "audit": {
            "method": "build_route_matrix_paired_eval",
            "generated_at_ms": generated_at_ms,
            "training_side_only": True,
            "writes_events": False,
            "writes_runtime_artifacts": False,
            "writes_training_sample_tags": False,
        },
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("heldout manifest must be a JSON object")
    return value


def _cluster_key(row: dict[str, Any]) -> str:
    return f"{_case_id(row)}|intent:{_intent(row)}"


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
    return "sha256:" + hashlib.sha256(_sample_prompt(row).encode("utf-8")).hexdigest()


def _intent(row: dict[str, Any]) -> str:
    return str(_sample_target(row).get("route_intent") or "<missing>").strip()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
