# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""R2.1 host-side route-router training and eval helpers.

The runtime contract is emitted only when the gated splits, heldout metrics,
tool-retrieval metrics, and calibration checks all pass release criteria.
Training-side split/cache/report artifacts stay under adapter/_training/.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Protocol

from backend.services.route_router_artifacts import (
    DEFAULT_INTENT_VOCAB,
    ROUTE_ROUTER_CALIBRATION_SET_NAME,
    ROUTE_ROUTER_EVAL_LEAKAGE_REPORT_NAME,
    ROUTE_ROUTER_EVAL_REPORT_NAME,
    ROUTE_ROUTER_TRAINING_SPLITS_NAME,
    build_route_router_calibration,
    build_route_router_eval_report,
    canonical_json_bytes,
    write_route_router_contract_artifacts,
)

ROUTE_ROUTER_TRAINING_SPLITS_SCHEMA_VERSION = (
    "edgestudio.route_router_training_splits.v0"
)
ROUTE_ROUTER_SPLIT_LEAKAGE_GATE_SCHEMA_VERSION = (
    "edgestudio.route_router_split_leakage_gate.v0"
)
ROUTE_ROUTER_TRAINING_SKELETON_SCHEMA_VERSION = (
    "edgestudio.route_router_training_skeleton.v0"
)
ROUTE_ROUTER_MATRIX_TRAINING_SCHEMA_VERSION = (
    "edgestudio.route_router_matrix_training.v0"
)
ROUTE_ROUTER_GATE_SEMANTICS_SCHEMA_VERSION = (
    "edgestudio.route_router_gate_semantics.v0"
)
ROUTE_ROUTER_CENTROID_TRAINER_METHOD = "centroid_linear_probe_v0"
ROUTE_ROUTER_LOGISTIC_TRAINER_METHOD = "multinomial_logistic_regression_v0"
ROUTE_ROUTER_MATRIX_TRAINER_METHOD = ROUTE_ROUTER_CENTROID_TRAINER_METHOD
ROUTE_ROUTER_TRAINER_METHODS = frozenset({
    ROUTE_ROUTER_CENTROID_TRAINER_METHOD,
    ROUTE_ROUTER_LOGISTIC_TRAINER_METHOD,
})
ROUTE_ROUTER_CALIBRATION_TEMPERATURE_OBJECTIVES = frozenset({"ece", "nll"})


class RouteRouterEmbeddingProvider(Protocol):
    """Frozen encoder interface used by the host-side matrix trainer."""

    def encode_texts(self, texts: list[str]) -> list[list[float]] | Any:
        ...


class PrecomputedRouteRouterEmbeddingProvider:
    """Embedding provider backed by a JSONL text -> embedding cache."""

    def __init__(
        self,
        embeddings_by_text: dict[str, list[float]],
        *,
        hidden_size: int,
    ):
        self.embeddings_by_text = embeddings_by_text
        self.hidden_size = int(hidden_size)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        *,
        hidden_size: int,
    ) -> "PrecomputedRouteRouterEmbeddingProvider":
        rows: dict[str, list[float]] = {}
        for line_no, raw in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"embedding cache line {line_no} must be an object")
            text = _required_str(payload.get("text"), f"embedding cache line {line_no}.text")
            embedding = payload.get("embedding")
            if not isinstance(embedding, list):
                raise ValueError(
                    f"embedding cache line {line_no}.embedding must be a list"
                )
            rows[text] = [float(value) for value in embedding]
        return cls(rows, hidden_size=hidden_size)

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        missing = [
            text for text in texts
            if text not in self.embeddings_by_text
        ]
        if missing:
            raise ValueError(f"missing precomputed embeddings for texts: {missing}")
        vectors = [self.embeddings_by_text[text] for text in texts]
        bad = [
            (text, len(vector))
            for text, vector in zip(texts, vectors)
            if len(vector) != self.hidden_size
        ]
        if bad:
            raise ValueError(f"precomputed embedding shape mismatch: {bad}")
        return vectors


@dataclass(frozen=True)
class RouteRouterTrainConfig:
    training_run_id: str
    base_model_id: str
    tokenizer_sha256: str
    hidden_size: int
    encoder_layer_index: int = -1
    intent_vocab: tuple[str, ...] = tuple(DEFAULT_INTENT_VOCAB)
    min_prompt_intent_acc: float = 0.80
    min_variant_intent_acc: float = 0.80
    min_unseen_tool_recall: float = 0.50
    max_calibration_ece: float = 0.10
    min_calibration_set_size: int = 20
    require_calibration_intent_coverage: bool = True
    trainer_method: str = ROUTE_ROUTER_MATRIX_TRAINER_METHOD
    logistic_l2: float = 0.0
    logistic_learning_rate: float = 0.50
    logistic_max_iter: int = 1500
    calibration_temperature_objective: str = "ece"
    calibration_threshold_percentile: float = 5.0
    max_calibrated_intent_threshold: float = 0.98
    prompt_gate_inconclusive_below_sample_count: int = 0
    calibration_temperatures: tuple[float, ...] = (
        0.35,
        0.45,
        0.5,
        0.6,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        4.0,
    )


def load_route_action_policy_dataset_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"route_action_policy_dataset line {line_no} must be an object")
        _sample_prompt(row)
        _sample_target(row)
        rows.append(row)
    return rows


def build_route_router_training_splits(
    samples: list[dict[str, Any]],
    *,
    training_run_id: str,
    min_calibration_samples: int = 1,
) -> dict[str, Any]:
    """Build deterministic split artifacts from learner samples.

    Splits are prompt-disjoint except `variant_heldout`, where cluster overlap
    is intentional but exact prompt overlap is still blocked by the leakage gate.
    """

    normalized_samples = [_normalize_sample(sample) for sample in samples]
    clusters = _clusters(normalized_samples)
    ordered_cluster_keys = sorted(
        clusters,
        key=lambda key: (_short_hash(key), key),
    )
    assignments: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "prompt_heldout": [],
        "variant_heldout": [],
        "unseen_tool_heldout": [],
        "calibration": [],
    }
    warnings: list[str] = []
    reserved: set[str] = set()

    unseen_tool, unseen_cluster = _select_unseen_tool_cluster(
        clusters,
        ordered_cluster_keys,
        reserved,
    )
    if unseen_tool and unseen_cluster and len(ordered_cluster_keys) >= 3:
        assignments["unseen_tool_heldout"].extend(clusters[unseen_cluster])
        reserved.add(unseen_cluster)
    elif unseen_tool:
        warnings.append("insufficient_clusters_for_unseen_tool_split")

    prompt_cluster = _last_available_cluster(
        ordered_cluster_keys,
        reserved,
        clusters=clusters,
        all_cluster_keys=ordered_cluster_keys,
        preserve_train_label_coverage=True,
    )
    if prompt_cluster and len(ordered_cluster_keys) - len(reserved) >= 2:
        assignments["prompt_heldout"].extend(clusters[prompt_cluster])
        reserved.add(prompt_cluster)
    else:
        warnings.append("insufficient_clusters_for_prompt_heldout_split")

    target_calibration_samples = max(1, int(min_calibration_samples))
    target_calibration_intents = {
        label
        for rows in clusters.values()
        for label in _cluster_intent_labels(rows)
    }
    while len(assignments["calibration"]) < target_calibration_samples:
        covered_calibration_intents = {
            _sample_target(row)["route_intent"]
            for row in assignments["calibration"]
        }
        missing_calibration_intents = (
            target_calibration_intents - covered_calibration_intents
        )
        calibration_cluster = None
        if missing_calibration_intents:
            calibration_cluster = _last_available_cluster_matching_intents(
                ordered_cluster_keys,
                reserved,
                required_intents=missing_calibration_intents,
                clusters=clusters,
                all_cluster_keys=ordered_cluster_keys,
                preserve_train_label_coverage=True,
            )
        if not calibration_cluster:
            calibration_cluster = _last_available_cluster(
                ordered_cluster_keys,
                reserved,
                clusters=clusters,
                all_cluster_keys=ordered_cluster_keys,
                preserve_train_label_coverage=True,
                allow_coverage_fallback=False,
            )
        if (
            not calibration_cluster
            or len(ordered_cluster_keys) - len(reserved) < 2
        ):
            if assignments["calibration"]:
                warnings.append("insufficient_calibration_samples")
            else:
                warnings.append("insufficient_clusters_for_calibration_split")
            break
        assignments["calibration"].extend(clusters[calibration_cluster])
        reserved.add(calibration_cluster)

    for cluster_key in ordered_cluster_keys:
        if cluster_key in reserved:
            continue
        primary, variants = _split_primary_and_variants(clusters[cluster_key])
        assignments["train"].extend(primary)
        assignments["variant_heldout"].extend(variants)

    leakage_gate = evaluate_route_router_split_leakage(assignments)
    unseen_tools = [unseen_tool] if unseen_tool else []
    return {
        "schema_version": ROUTE_ROUTER_TRAINING_SPLITS_SCHEMA_VERSION,
        "training_run_id": _required_str(training_run_id, "training_run_id"),
        "status": "built" if leakage_gate["ok"] else "leakage_blocked",
        "split_policy": {
            "prompt_heldout": "cluster-disjoint deterministic holdout",
            "variant_heldout": "same source cluster, exact prompt disjoint",
            "unseen_tool_heldout": "cluster-disjoint synthetic open-tool holdout",
            "calibration": "cluster-disjoint calibration prompts",
            "min_calibration_samples": target_calibration_samples,
        },
        "unseen_tool": unseen_tool,
        "unseen_tools": unseen_tools,
        "splits": assignments,
        "summary": {
            "sample_count": len(normalized_samples),
            "cluster_count": len(clusters),
            "split_counts": {
                name: len(rows)
                for name, rows in assignments.items()
            },
            "warnings": warnings,
        },
        "leakage_gate": leakage_gate,
    }


def evaluate_route_router_split_leakage(
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    prompt_index: dict[str, dict[str, set[str]]] = {}
    cluster_index: dict[str, dict[str, set[str]]] = {}
    for split_name, rows in splits.items():
        for row in rows:
            prompt_key = _prompt_key(_sample_prompt(row))
            prompt_index.setdefault(prompt_key, {}).setdefault(split_name, set()).add(
                _sample_id(row)
            )
            cluster_index.setdefault(_cluster_key(row), {}).setdefault(split_name, set()).add(
                _sample_id(row)
            )

    hard_pairs = {
        frozenset(("train", "prompt_heldout")),
        frozenset(("train", "calibration")),
        frozenset(("train", "unseen_tool_heldout")),
        frozenset(("prompt_heldout", "calibration")),
        frozenset(("prompt_heldout", "unseen_tool_heldout")),
        frozenset(("calibration", "unseen_tool_heldout")),
        frozenset(("train", "variant_heldout")),
        frozenset(("variant_heldout", "prompt_heldout")),
        frozenset(("variant_heldout", "calibration")),
        frozenset(("variant_heldout", "unseen_tool_heldout")),
    }
    overlaps: list[dict[str, Any]] = []
    for prompt_key, by_split in sorted(prompt_index.items()):
        split_names = sorted(by_split)
        for left_idx, left in enumerate(split_names):
            for right in split_names[left_idx + 1:]:
                if frozenset((left, right)) not in hard_pairs:
                    continue
                overlaps.append({
                    "prompt_fingerprint": _fingerprint(prompt_key),
                    "left_split": left,
                    "right_split": right,
                    "left_sample_ids": sorted(by_split[left]),
                    "right_sample_ids": sorted(by_split[right]),
                })

    variant_cluster_overlaps = []
    for cluster_key, by_split in sorted(cluster_index.items()):
        if "train" in by_split and "variant_heldout" in by_split:
            variant_cluster_overlaps.append({
                "cluster_fingerprint": _fingerprint(cluster_key),
                "train_sample_ids": sorted(by_split["train"]),
                "variant_sample_ids": sorted(by_split["variant_heldout"]),
            })

    return {
        "ok": not overlaps,
        "schema_version": ROUTE_ROUTER_SPLIT_LEAKAGE_GATE_SCHEMA_VERSION,
        "status": "passed" if not overlaps else "split_leakage_blocked",
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "variant_cluster_overlap_count": len(variant_cluster_overlaps),
        "variant_cluster_overlaps": variant_cluster_overlaps,
        "split_prompt_counts": {
            split_name: len({
                _prompt_key(_sample_prompt(row))
                for row in rows
            })
            for split_name, rows in splits.items()
        },
    }


def write_route_router_training_skeleton_artifacts(
    *,
    dataset_path: Path,
    adapter_dir: Path,
    training_run_id: str,
    base_model_id: str,
    tokenizer_sha256: str,
    hidden_size: int,
) -> dict[str, Any]:
    """Write R2.1 Phase 2 split/eval skeleton artifacts.

    Skeleton artifacts are training-side only. Runtime matrix artifacts are
    emitted exclusively by `train_route_router_matrix_artifacts` after release
    gates pass.
    """

    samples = load_route_action_policy_dataset_jsonl(dataset_path)
    splits = build_route_router_training_splits(
        samples,
        training_run_id=training_run_id,
    )
    leakage_gate = splits["leakage_gate"]
    eval_report = build_route_router_eval_report(
        training_run_id=training_run_id,
        status="skeleton_ready" if leakage_gate["ok"] else "leakage_blocked",
        metrics=_skeleton_metrics(splits),
        leakage_gate=leakage_gate,
    )

    adapter_dir = Path(adapter_dir)
    training_dir = adapter_dir / "_training"
    training_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        ROUTE_ROUTER_TRAINING_SPLITS_NAME: canonical_json_bytes(splits),
        ROUTE_ROUTER_EVAL_REPORT_NAME: canonical_json_bytes(eval_report),
        ROUTE_ROUTER_EVAL_LEAKAGE_REPORT_NAME: canonical_json_bytes(leakage_gate),
        ROUTE_ROUTER_CALIBRATION_SET_NAME: _jsonl_bytes(splits["splits"]["calibration"]),
    }
    for name, data in files.items():
        (training_dir / name).write_bytes(data)

    return {
        "ok": leakage_gate["ok"],
        "schema_version": ROUTE_ROUTER_TRAINING_SKELETON_SCHEMA_VERSION,
        "status": "written" if leakage_gate["ok"] else "leakage_blocked",
        "training_run_id": training_run_id,
        "dataset_path": str(dataset_path),
        "training_dir": str(training_dir),
        "split_summary": splits["summary"],
        "leakage_gate": leakage_gate,
        "eval_report": eval_report,
        "runtime_artifact": None,
        "files": [
            {
                "name": name,
                "path": str(training_dir / name),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
    }


def train_route_router_matrix_artifacts(
    *,
    dataset_path: Path,
    adapter_dir: Path,
    config: RouteRouterTrainConfig,
    embedding_provider: RouteRouterEmbeddingProvider,
    tool_descriptors: dict[str, str] | None = None,
    tool_route_intents: dict[str, list[str]] | None = None,
    write_runtime_contract: bool = False,
) -> dict[str, Any]:
    """Train the R2.1 matrix router over frozen embeddings.

    The encoder is injected so this host-side trainer stays independent from a
    concrete MLX model loader. Runtime artifacts are written only after the
    split leakage gate passes and calibration rows exist.
    """

    samples = load_route_action_policy_dataset_jsonl(dataset_path)
    splits = build_route_router_training_splits(
        samples,
        training_run_id=config.training_run_id,
        min_calibration_samples=config.min_calibration_set_size,
    )
    leakage_gate = splits["leakage_gate"]
    adapter_dir = Path(adapter_dir)
    training_dir = adapter_dir / "_training"
    training_dir.mkdir(parents=True, exist_ok=True)

    status = "trained"
    warnings = list(splits["summary"].get("warnings", []))
    runtime_receipt = None
    weights = None
    bias = None
    calibration = build_route_router_calibration(calibration_set_size=0)
    metrics: dict[str, Any] = {
        "mode": "matrix_training",
        "trainer_method": config.trainer_method,
        "release_ready": False,
    }

    try:
        _validate_train_config(config)
        _validate_labels_in_vocab(splits, list(config.intent_vocab))
        if not leakage_gate["ok"]:
            status = "leakage_blocked"
        elif not splits["splits"]["train"]:
            status = "insufficient_train_samples"
        elif not splits["splits"]["calibration"]:
            status = "calibration_missing"
        else:
            train_rows = splits["splits"]["train"]
            train_embeddings = _embedding_matrix(
                embedding_provider,
                [_sample_prompt(row) for row in train_rows],
                config.hidden_size,
                "train_embeddings",
            )
            train_labels = [
                _sample_target(row)["route_intent"]
                for row in train_rows
            ]
            weights, bias = _fit_intent_head(
                train_embeddings,
                train_labels,
                config=config,
            )
            calibration = _fit_intent_calibration(
                embedding_provider=embedding_provider,
                rows=splits["splits"]["calibration"],
                weights=weights,
                bias=bias,
                config=config,
            )
            metrics = _matrix_eval_metrics(
                embedding_provider=embedding_provider,
                splits=splits,
                weights=weights,
                bias=bias,
                config=config,
                calibration=calibration,
                tool_descriptors=tool_descriptors or {},
                tool_route_intents=tool_route_intents or {},
            )
            status = (
                "trained_release_ready"
                if metrics.get("release_ready") is True
                else "trained_not_release_ready"
            )
            if write_runtime_contract and metrics.get("release_ready") is True:
                matrix_bytes = _intent_matrix_safetensors_bytes(weights, bias)
                runtime_receipt = write_route_router_contract_artifacts(
                    output_dir=adapter_dir,
                    training_run_id=config.training_run_id,
                    base_model_id=config.base_model_id,
                    tokenizer_sha256=config.tokenizer_sha256,
                    hidden_size=config.hidden_size,
                    layer_index=config.encoder_layer_index,
                    intent_matrix_bytes=matrix_bytes,
                    intent_vocab=list(config.intent_vocab),
                    calibration=calibration,
                )
            elif write_runtime_contract:
                warnings.append("runtime_contract_blocked_not_release_ready")
    except ValueError as exc:
        status = "training_blocked"
        warnings.append(str(exc))

    metrics_with_summary = _with_non_release_summary_fields(
        metrics,
        status=status,
    )
    eval_report = build_route_router_eval_report(
        training_run_id=config.training_run_id,
        status=status,
        metrics=metrics_with_summary,
        leakage_gate=leakage_gate,
    )
    files: dict[str, bytes] = {
        ROUTE_ROUTER_TRAINING_SPLITS_NAME: canonical_json_bytes(splits),
        ROUTE_ROUTER_EVAL_REPORT_NAME: canonical_json_bytes(eval_report),
        ROUTE_ROUTER_EVAL_LEAKAGE_REPORT_NAME: canonical_json_bytes(leakage_gate),
        ROUTE_ROUTER_CALIBRATION_SET_NAME: _jsonl_bytes(splits["splits"]["calibration"]),
    }
    for name, data in files.items():
        (training_dir / name).write_bytes(data)

    ok = status in {"trained_release_ready", "trained_not_release_ready"}
    return {
        "ok": ok,
        "schema_version": ROUTE_ROUTER_MATRIX_TRAINING_SCHEMA_VERSION,
        "status": status,
        "training_run_id": config.training_run_id,
        "dataset_path": str(dataset_path),
        "training_dir": str(training_dir),
        "trainer_method": config.trainer_method,
        "intent_vocab": list(config.intent_vocab),
        "hidden_size": config.hidden_size,
        "split_summary": splits["summary"],
        "leakage_gate": leakage_gate,
        "calibration": calibration,
        "eval_report": eval_report,
        "training_side_evidence_candidate_ready": (
            metrics_with_summary["training_side_evidence_candidate_ready"]
        ),
        "training_side_evidence_candidate_reason": (
            metrics_with_summary["training_side_evidence_candidate_reason"]
        ),
        "runtime_release_ready": metrics_with_summary["runtime_release_ready"],
        "runtime_release_blockers": metrics_with_summary["runtime_release_blockers"],
        "diagnostic_warnings": metrics_with_summary["diagnostic_warnings"],
        "production_router_improved": metrics_with_summary[
            "production_router_improved"
        ],
        "scope_bounded_to": metrics_with_summary["scope_bounded_to"],
        "runtime_artifact": runtime_receipt,
        "warnings": warnings,
        "files": [
            {
                "name": name,
                "path": str(training_dir / name),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        ],
        "matrix_shape": (
            list(weights.shape)
            if weights is not None
            else [config.hidden_size, len(config.intent_vocab)]
        ),
        "bias_shape": (
            list(bias.shape)
            if bias is not None
            else [len(config.intent_vocab)]
        ),
    }


def _skeleton_metrics(splits: dict[str, Any]) -> dict[str, Any]:
    counts = splits["summary"]["split_counts"]
    all_samples = [
        row
        for rows in splits["splits"].values()
        for row in rows
    ]
    return {
        "mode": "skeleton_only",
        "intent_acc": None,
        "tool_macro_f1": None,
        "calibration_ece": None,
        "sample_count": splits["summary"]["sample_count"],
        "cluster_count": splits["summary"]["cluster_count"],
        "train_count": counts.get("train", 0),
        "prompt_heldout_count": counts.get("prompt_heldout", 0),
        "variant_heldout_count": counts.get("variant_heldout", 0),
        "unseen_tool_heldout_count": counts.get("unseen_tool_heldout", 0),
        "calibration_count": counts.get("calibration", 0),
        "split_counts": counts,
        "intent_label_count": _intent_label_counts(all_samples),
        "tool_label_count": _tool_label_counts(all_samples),
    }


def _with_non_release_summary_fields(
    metrics: dict[str, Any],
    *,
    status: str,
    scope_bounded_to: list[str] | None = None,
) -> dict[str, Any]:
    release_ready = metrics.get("release_ready") is True
    blockers = (
        []
        if release_ready
        else _runtime_release_blockers(metrics, status=status)
    )
    scope = sorted({
        str(item).strip()
        for item in (scope_bounded_to or [])
        if str(item).strip()
    })
    if not scope:
        scope = ["scope_not_declared"]
    return {
        **metrics,
        "training_side_evidence_candidate_ready": False,
        "training_side_evidence_candidate_reason": (
            "paired_heldout_evidence_not_attached"
        ),
        "runtime_release_ready": release_ready,
        "runtime_release_blockers": blockers,
        "diagnostic_warnings": _matrix_diagnostic_warnings(metrics),
        "production_router_improved": False,
        "scope_bounded_to": scope,
    }


def _runtime_release_blockers(metrics: dict[str, Any], *, status: str) -> list[str]:
    gates = metrics.get("release_gates")
    blockers: list[str] = []
    if isinstance(gates, dict):
        blockers = [
            _runtime_release_blocker_name(str(name), gate)
            for name, gate in sorted(gates.items())
            if not (isinstance(gate, dict) and gate.get("passed") is True)
        ]
    if blockers:
        return blockers
    if status != "trained_release_ready":
        return [f"training_status:{status}"]
    return ["release_ready_false_without_failed_gate"]


def _runtime_release_blocker_name(name: str, gate: Any) -> str:
    if not isinstance(gate, dict):
        return name
    status = str(gate.get("status") or "").strip()
    if gate.get("inconclusive") is True and status:
        return f"{name}:{status}"
    return name


def _matrix_diagnostic_warnings(metrics: dict[str, Any]) -> list[str]:
    warnings: set[str] = set()
    coverage = metrics.get("coverage")
    if isinstance(coverage, dict):
        warnings.update(str(item) for item in coverage.get("warnings") or [])
    gates = metrics.get("release_gates")
    if isinstance(gates, dict):
        for name, gate in gates.items():
            if isinstance(gate, dict) and gate.get("inconclusive") is True:
                status = str(gate.get("status") or "inconclusive").strip()
                warnings.add(f"{name}:{status}")
    if metrics.get("calibration_ece") is None:
        warnings.add("calibration_ece_missing")
    return sorted(warning for warning in warnings if warning)


def _validate_train_config(config: RouteRouterTrainConfig) -> None:
    _required_str(config.training_run_id, "config.training_run_id")
    _required_str(config.base_model_id, "config.base_model_id")
    _required_str(config.tokenizer_sha256, "config.tokenizer_sha256")
    if int(config.hidden_size) <= 0:
        raise ValueError("config.hidden_size must be positive")
    if not config.intent_vocab:
        raise ValueError("config.intent_vocab must not be empty")
    if any(not str(label or "").strip() for label in config.intent_vocab):
        raise ValueError("config.intent_vocab labels must be non-empty")
    if config.trainer_method not in ROUTE_ROUTER_TRAINER_METHODS:
        raise ValueError(
            "config.trainer_method must be one of "
            f"{sorted(ROUTE_ROUTER_TRAINER_METHODS)}"
        )
    if float(config.logistic_l2) < 0:
        raise ValueError("config.logistic_l2 must be non-negative")
    if float(config.logistic_learning_rate) <= 0:
        raise ValueError("config.logistic_learning_rate must be positive")
    if int(config.logistic_max_iter) <= 0:
        raise ValueError("config.logistic_max_iter must be positive")
    if (
        config.calibration_temperature_objective
        not in ROUTE_ROUTER_CALIBRATION_TEMPERATURE_OBJECTIVES
    ):
        raise ValueError(
            "config.calibration_temperature_objective must be one of "
            f"{sorted(ROUTE_ROUTER_CALIBRATION_TEMPERATURE_OBJECTIVES)}"
        )
    for name, threshold in {
        "min_prompt_intent_acc": config.min_prompt_intent_acc,
        "min_variant_intent_acc": config.min_variant_intent_acc,
        "min_unseen_tool_recall": config.min_unseen_tool_recall,
        "max_calibration_ece": config.max_calibration_ece,
    }.items():
        if float(threshold) < 0 or float(threshold) > 1:
            raise ValueError(f"config.{name} must be in [0, 1]")
    if int(config.min_calibration_set_size) < 0:
        raise ValueError("config.min_calibration_set_size must be non-negative")
    if int(config.prompt_gate_inconclusive_below_sample_count) < 0:
        raise ValueError(
            "config.prompt_gate_inconclusive_below_sample_count must be non-negative"
        )
    if not config.calibration_temperatures:
        raise ValueError("config.calibration_temperatures must not be empty")
    if any(float(temp) <= 0 for temp in config.calibration_temperatures):
        raise ValueError("config.calibration_temperatures must be positive")


def _validate_labels_in_vocab(
    splits: dict[str, Any],
    intent_vocab: list[str],
) -> None:
    labels = {
        _sample_target(row)["route_intent"]
        for rows in splits["splits"].values()
        for row in rows
    }
    unknown = sorted(label for label in labels if label not in intent_vocab)
    if unknown:
        raise ValueError(f"unknown route intent labels: {unknown}")


def _embedding_matrix(
    embedding_provider: RouteRouterEmbeddingProvider,
    texts: list[str],
    hidden_size: int,
    field: str,
) -> Any:
    import numpy as np

    if not texts:
        return np.zeros((0, int(hidden_size)), dtype=np.float32)
    raw = embedding_provider.encode_texts(texts)
    matrix = np.asarray(raw, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{field} must be a rank-2 embedding matrix")
    expected_shape = (len(texts), int(hidden_size))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{field} shape must be {expected_shape}, got {tuple(matrix.shape)}"
        )
    return _l2_normalize(matrix)


def _fit_intent_head(
    embeddings: Any,
    labels: list[str],
    *,
    config: RouteRouterTrainConfig,
) -> tuple[Any, Any]:
    if config.trainer_method == ROUTE_ROUTER_LOGISTIC_TRAINER_METHOD:
        return _fit_multinomial_logistic_regression(
            embeddings,
            labels,
            list(config.intent_vocab),
            l2=config.logistic_l2,
            learning_rate=config.logistic_learning_rate,
            max_iter=config.logistic_max_iter,
        )
    return _fit_centroid_linear_probe(
        embeddings,
        labels,
        list(config.intent_vocab),
    )


def _fit_centroid_linear_probe(
    embeddings: Any,
    labels: list[str],
    intent_vocab: list[str],
) -> tuple[Any, Any]:
    import numpy as np

    hidden_size = int(embeddings.shape[1])
    weights = np.zeros((hidden_size, len(intent_vocab)), dtype=np.float32)
    bias = np.full((len(intent_vocab),), -20.0, dtype=np.float32)
    total = max(1, len(labels))
    for idx, label in enumerate(intent_vocab):
        label_rows = embeddings[
            np.asarray([item == label for item in labels], dtype=bool)
        ]
        if len(label_rows) == 0:
            continue
        centroid = _l2_normalize(label_rows.mean(axis=0, keepdims=True))[0]
        prior = max(1, len(label_rows)) / total
        weights[:, idx] = centroid
        bias[idx] = float(np.log(prior) - 0.5 * np.dot(centroid, centroid))
    return weights, bias


def _fit_multinomial_logistic_regression(
    embeddings: Any,
    labels: list[str],
    intent_vocab: list[str],
    *,
    l2: float,
    learning_rate: float,
    max_iter: int,
) -> tuple[Any, Any]:
    import numpy as np

    x = np.asarray(embeddings, dtype=np.float32)
    n_rows, hidden_size = x.shape
    n_classes = len(intent_vocab)
    y = np.asarray([intent_vocab.index(label) for label in labels], dtype=np.int64)
    weights = np.zeros((hidden_size, n_classes), dtype=np.float32)

    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    priors = np.maximum(counts, 1e-3)
    priors = priors / priors.sum()
    bias = np.log(priors).astype(np.float32)

    y_one_hot = np.zeros((n_rows, n_classes), dtype=np.float32)
    y_one_hot[np.arange(n_rows), y] = 1.0
    step = float(learning_rate)
    penalty = float(l2)
    for _ in range(int(max_iter)):
        probs = _softmax((x @ weights) + bias)
        error = (probs - y_one_hot) / max(1, n_rows)
        grad_w = (x.T @ error) + (penalty * weights)
        grad_b = error.sum(axis=0)
        weights -= step * grad_w.astype(np.float32)
        bias -= step * grad_b.astype(np.float32)
    return weights.astype(np.float32), bias.astype(np.float32)


def _fit_intent_calibration(
    *,
    embedding_provider: RouteRouterEmbeddingProvider,
    rows: list[dict[str, Any]],
    weights: Any,
    bias: Any,
    config: RouteRouterTrainConfig,
) -> dict[str, Any]:
    import numpy as np

    if not rows:
        return build_route_router_calibration(calibration_set_size=0)
    embeddings = _embedding_matrix(
        embedding_provider,
        [_sample_prompt(row) for row in rows],
        config.hidden_size,
        "calibration_embeddings",
    )
    labels = [
        list(config.intent_vocab).index(_sample_target(row)["route_intent"])
        for row in rows
    ]
    label_array = np.asarray(labels, dtype=np.int64)
    logits = _intent_logits(embeddings, weights, bias)
    best_temperature = _select_calibration_temperature(
        logits,
        label_array,
        config=config,
    )
    probs = _softmax(logits / best_temperature)
    thresholds = _calibrated_thresholds(
        probs,
        label_array,
        list(config.intent_vocab),
        percentile=config.calibration_threshold_percentile,
        max_threshold=config.max_calibrated_intent_threshold,
    )
    return build_route_router_calibration(
        intent_thresholds=thresholds,
        intent_temperature=best_temperature,
        calibration_set_size=len(rows),
        calibration_ece=_expected_calibration_error(probs, label_array),
    )


def _matrix_eval_metrics(
    *,
    embedding_provider: RouteRouterEmbeddingProvider,
    splits: dict[str, Any],
    weights: Any,
    bias: Any,
    config: RouteRouterTrainConfig,
    calibration: dict[str, Any],
    tool_descriptors: dict[str, str],
    tool_route_intents: dict[str, list[str]],
) -> dict[str, Any]:
    intent_vocab = list(config.intent_vocab)
    temperature = float(calibration["intent_temperature"])
    prompt_metrics = _intent_split_metrics(
        embedding_provider,
        splits["splits"]["prompt_heldout"],
        weights,
        bias,
        config.hidden_size,
        intent_vocab,
        temperature,
        "prompt_heldout",
    )
    variant_metrics = _intent_split_metrics(
        embedding_provider,
        splits["splits"]["variant_heldout"],
        weights,
        bias,
        config.hidden_size,
        intent_vocab,
        temperature,
        "variant_heldout",
    )
    calibration_metrics = _intent_split_metrics(
        embedding_provider,
        splits["splits"]["calibration"],
        weights,
        bias,
        config.hidden_size,
        intent_vocab,
        temperature,
        "calibration",
    )
    unseen_tool_metrics = _tool_retrieval_metrics(
        embedding_provider=embedding_provider,
        rows=splits["splits"]["unseen_tool_heldout"],
        hidden_size=config.hidden_size,
        tool_descriptors=tool_descriptors,
        tool_route_intents=tool_route_intents,
    )
    counts = splits["summary"]["split_counts"]
    coverage = _matrix_data_coverage(
        splits=splits,
        tool_descriptors=tool_descriptors,
    )
    unseen_tool_ok = (
        counts.get("unseen_tool_heldout", 0) > 0
        and unseen_tool_metrics["status"] == "evaluated"
        and unseen_tool_metrics["unseen_tool_recall"]
        >= config.min_unseen_tool_recall
    )
    calibration_set_size = int(calibration.get("calibration_set_size") or 0)
    calibration_ece = calibration.get("calibration_ece")
    calibration_ok = (
        calibration_set_size >= int(config.min_calibration_set_size)
        and calibration_ece is not None
        and float(calibration_ece) <= config.max_calibration_ece
    )
    calibration_intent_coverage_ok = (
        not config.require_calibration_intent_coverage
        or not coverage["missing_calibration_intents"]
    )
    prompt_sample_count = int(prompt_metrics["sample_count"])
    prompt_gate_passed = (
        prompt_metrics["intent_acc"] >= config.min_prompt_intent_acc
    )
    prompt_gate_inconclusive = (
        not prompt_gate_passed
        and int(config.prompt_gate_inconclusive_below_sample_count) > 0
        and 0 < prompt_sample_count
        < int(config.prompt_gate_inconclusive_below_sample_count)
    )
    release_ready = (
        splits["leakage_gate"]["ok"] is True
        and counts.get("train", 0) > 0
        and counts.get("calibration", 0) > 0
        and prompt_gate_passed
        and variant_metrics["intent_acc"] >= config.min_variant_intent_acc
        and unseen_tool_ok
        and coverage["ok"] is True
        and calibration_ok
        and calibration_intent_coverage_ok
    )
    return {
        "mode": "matrix_training",
        "trainer_method": config.trainer_method,
        "calibration_temperature_objective": (
            config.calibration_temperature_objective
        ),
        "release_ready": release_ready,
        "sample_count": splits["summary"]["sample_count"],
        "cluster_count": splits["summary"]["cluster_count"],
        "split_counts": counts,
        "intent_vocab": intent_vocab,
        "prompt_split": prompt_metrics,
        "variant_split": variant_metrics,
        "calibration_split": calibration_metrics,
        "unseen_tool_split": unseen_tool_metrics,
        "coverage": coverage,
        "gate_semantics": {
            "schema_version": ROUTE_ROUTER_GATE_SEMANTICS_SCHEMA_VERSION,
            "prompt_gate_inconclusive_below_sample_count": int(
                config.prompt_gate_inconclusive_below_sample_count
            ),
            "inconclusive_gates_do_not_count_as_passed": True,
        },
        "release_gates": {
            "prompt_intent_acc": {
                "actual": prompt_metrics["intent_acc"],
                "sample_count": prompt_sample_count,
                "threshold": config.min_prompt_intent_acc,
                "passed": prompt_gate_passed,
                "inconclusive": prompt_gate_inconclusive,
                "status": (
                    "passed"
                    if prompt_gate_passed
                    else (
                        "inconclusive_small_n_not_failed"
                        if prompt_gate_inconclusive
                        else "failed"
                    )
                ),
                "inconclusive_below_sample_count": int(
                    config.prompt_gate_inconclusive_below_sample_count
                ),
            },
            "variant_intent_acc": {
                "actual": variant_metrics["intent_acc"],
                "threshold": config.min_variant_intent_acc,
                "passed": variant_metrics["intent_acc"] >= config.min_variant_intent_acc,
            },
            "unseen_tool_recall": {
                "actual": unseen_tool_metrics.get("unseen_tool_recall"),
                "threshold": config.min_unseen_tool_recall,
                "passed": unseen_tool_ok,
            },
            "calibration_ece": {
                "actual": calibration_ece,
                "threshold": config.max_calibration_ece,
                "passed": (
                    calibration_ece is not None
                    and float(calibration_ece) <= config.max_calibration_ece
                ),
            },
            "calibration_set_size": {
                "actual": calibration_set_size,
                "threshold": config.min_calibration_set_size,
                "passed": calibration_set_size >= config.min_calibration_set_size,
            },
            "calibration_intent_coverage": {
                "missing_intents": coverage["missing_calibration_intents"],
                "required": config.require_calibration_intent_coverage,
                "passed": calibration_intent_coverage_ok,
            },
            "coverage": {
                "passed": coverage["ok"] is True,
            },
        },
        "calibration_ece": calibration.get("calibration_ece"),
    }


def _matrix_data_coverage(
    *,
    splits: dict[str, Any],
    tool_descriptors: dict[str, str],
) -> dict[str, Any]:
    all_rows = [
        row
        for rows in splits["splits"].values()
        for row in rows
    ]
    train_rows = splits["splits"]["train"]
    calibration_rows = splits["splits"]["calibration"]
    intent_cluster_counts: dict[str, int] = {}
    for rows in _clusters(all_rows).values():
        for label in _cluster_intent_labels(rows):
            intent_cluster_counts[label] = intent_cluster_counts.get(label, 0) + 1
    seen_intents = sorted(intent_cluster_counts)
    train_intents = _intent_label_counts(train_rows)
    calibration_intents = _intent_label_counts(calibration_rows)
    missing_train_intents = [
        label for label in seen_intents
        if train_intents.get(label, 0) <= 0
    ]
    missing_calibration_intents = [
        label for label in seen_intents
        if calibration_intents.get(label, 0) <= 0
    ]
    seen_tools = sorted({
        tool
        for row in all_rows
        for tool in _selected_tools(row)
    })
    missing_tool_descriptors = [
        tool for tool in seen_tools
        if tool not in tool_descriptors
    ]
    warnings = []
    if missing_train_intents:
        warnings.append("missing_train_intent_coverage")
    if missing_calibration_intents:
        warnings.append("missing_calibration_intent_coverage")
    if missing_tool_descriptors:
        warnings.append("missing_tool_descriptors")
    if len(seen_intents) < 3:
        warnings.append("low_intent_diversity")
    if splits["summary"]["cluster_count"] < 6:
        warnings.append("low_cluster_count")
    return {
        "ok": not missing_train_intents and not missing_tool_descriptors,
        "sample_count": len(all_rows),
        "cluster_count": splits["summary"]["cluster_count"],
        "seen_intents": seen_intents,
        "intent_cluster_counts": dict(sorted(intent_cluster_counts.items())),
        "train_intent_counts": train_intents,
        "calibration_intent_counts": calibration_intents,
        "seen_tools": seen_tools,
        "registered_tool_count": len(tool_descriptors),
        "missing_train_intents": missing_train_intents,
        "missing_calibration_intents": missing_calibration_intents,
        "missing_tool_descriptors": missing_tool_descriptors,
        "warnings": warnings,
    }


def _intent_split_metrics(
    embedding_provider: RouteRouterEmbeddingProvider,
    rows: list[dict[str, Any]],
    weights: Any,
    bias: Any,
    hidden_size: int,
    intent_vocab: list[str],
    temperature: float,
    split_name: str,
) -> dict[str, Any]:
    import numpy as np

    if not rows:
        return {
            "split": split_name,
            "sample_count": 0,
            "intent_acc": 0.0,
            "intent_per_class_f1": {},
        }
    embeddings = _embedding_matrix(
        embedding_provider,
        [_sample_prompt(row) for row in rows],
        hidden_size,
        f"{split_name}_embeddings",
    )
    labels = [
        intent_vocab.index(_sample_target(row)["route_intent"])
        for row in rows
    ]
    label_array = np.asarray(labels, dtype=np.int64)
    probs = _softmax(_intent_logits(embeddings, weights, bias) / temperature)
    predictions = probs.argmax(axis=1)
    return {
        "split": split_name,
        "sample_count": len(rows),
        "intent_acc": float((predictions == label_array).mean()),
        "intent_per_class_f1": _per_class_f1(
            label_array,
            predictions,
            intent_vocab,
        ),
        "mean_top_confidence": float(probs.max(axis=1).mean()),
        "predictions": [
            {
                "prompt": _sample_prompt(row),
                "expected": intent_vocab[int(expected)],
                "predicted": intent_vocab[int(predicted)],
                "top_confidence": float(confidence),
            }
            for row, expected, predicted, confidence in zip(
                rows,
                label_array,
                predictions,
                probs.max(axis=1),
            )
        ],
    }


def _tool_retrieval_metrics(
    *,
    embedding_provider: RouteRouterEmbeddingProvider,
    rows: list[dict[str, Any]],
    hidden_size: int,
    tool_descriptors: dict[str, str],
    tool_route_intents: dict[str, list[str]],
) -> dict[str, Any]:
    import numpy as np

    target_tools = sorted({
        tool
        for row in rows
        for tool in _selected_tools(row)
    })
    if not rows or not target_tools:
        return {
            "status": "not_applicable",
            "sample_count": len(rows),
            "unseen_tool_recall": None,
            "unseen_tool_precision": None,
        }
    missing = [tool for tool in target_tools if tool not in tool_descriptors]
    if missing:
        return {
            "status": "missing_tool_descriptors",
            "sample_count": len(rows),
            "missing_tools": missing,
            "unseen_tool_recall": None,
            "unseen_tool_precision": None,
        }
    candidate_tools = sorted(
        tool
        for tool, description in tool_descriptors.items()
        if str(tool or "").strip() and str(description or "").strip()
    )
    if not candidate_tools:
        return {
            "status": "missing_tool_descriptors",
            "sample_count": len(rows),
            "missing_tools": target_tools,
            "unseen_tool_recall": None,
            "unseen_tool_precision": None,
        }
    candidate_filter = _tool_candidate_filter(
        rows=rows,
        candidate_tools=candidate_tools,
        tool_route_intents=tool_route_intents,
    )
    if candidate_filter["kind"] == "route_intent":
        candidate_tools = list(candidate_filter["candidate_tool_names"])
    if not candidate_tools:
        return {
            "status": "no_route_intent_candidates",
            "sample_count": len(rows),
            "missing_tools": target_tools,
            "unseen_tool_recall": None,
            "unseen_tool_precision": None,
            "candidate_filter": candidate_filter,
        }
    prompt_embeddings = _embedding_matrix(
        embedding_provider,
        [_sample_prompt(row) for row in rows],
        hidden_size,
        "unseen_tool_prompt_embeddings",
    )
    tool_embeddings = _embedding_matrix(
        embedding_provider,
        [tool_descriptors[tool] for tool in candidate_tools],
        hidden_size,
        "unseen_tool_descriptor_embeddings",
    )
    scores = prompt_embeddings @ tool_embeddings.T
    ranked_idx = np.argsort(-scores, axis=1)
    predicted_idx = ranked_idx[:, 0]
    true_sets = [set(_selected_tools(row)) for row in rows]
    predicted_tools = [
        candidate_tools[int(idx)]
        for idx in predicted_idx
    ]
    true_positive = sum(
        1 for pred, expected in zip(predicted_tools, true_sets)
        if pred in expected
    )
    expected_count = sum(len(expected) for expected in true_sets)
    top_k = min(3, len(candidate_tools))
    true_positive_at_k = 0
    reciprocal_ranks: list[float] = []
    predictions = []
    for row, expected, row_scores, row_ranked_idx, pred in zip(
        rows,
        true_sets,
        scores,
        ranked_idx,
        predicted_tools,
    ):
        ranked_tools = [
            candidate_tools[int(idx)]
            for idx in row_ranked_idx
        ]
        top_tools = ranked_tools[:top_k]
        true_positive_at_k += len(set(top_tools) & expected)
        rank = next(
            (
                position + 1
                for position, tool in enumerate(ranked_tools)
                if tool in expected
            ),
            None,
        )
        reciprocal_ranks.append(float(1.0 / rank) if rank else 0.0)
        predictions.append({
            "prompt": _sample_prompt(row),
            "target_tools": sorted(expected),
            "predicted_tool": pred,
            "top_score": float(row_scores[int(row_ranked_idx[0])]),
            "top_tools": [
                {
                    "tool": candidate_tools[int(idx)],
                    "score": float(row_scores[int(idx)]),
                    "expected": candidate_tools[int(idx)] in expected,
                }
                for idx in row_ranked_idx[:top_k]
            ],
        })
    recall = float(true_positive / expected_count) if expected_count else 0.0
    return {
        "status": "evaluated",
        "sample_count": len(rows),
        "tool_count": len(candidate_tools),
        "target_tool_names": target_tools,
        "candidate_tool_names": candidate_tools,
        "candidate_filter": candidate_filter,
        "unseen_tool_recall": recall,
        "unseen_tool_recall_at_1": recall,
        "unseen_tool_recall_at_3": (
            float(true_positive_at_k / expected_count)
            if expected_count
            else 0.0
        ),
        "unseen_tool_mrr": (
            float(sum(reciprocal_ranks) / len(reciprocal_ranks))
            if reciprocal_ranks
            else 0.0
        ),
        "unseen_tool_precision": float(true_positive / max(1, len(predicted_tools))),
        "mean_top_score": float(np.max(scores, axis=1).mean()),
        "predictions": predictions,
    }


def _tool_candidate_filter(
    *,
    rows: list[dict[str, Any]],
    candidate_tools: list[str],
    tool_route_intents: dict[str, list[str]],
) -> dict[str, Any]:
    route_intents = sorted({
        str(_sample_target(row).get("route_intent") or "").strip()
        for row in rows
        if str(_sample_target(row).get("route_intent") or "").strip()
    })
    if not tool_route_intents or not route_intents:
        return {
            "kind": "none",
            "route_intents": route_intents,
            "candidate_tool_names": list(candidate_tools),
        }
    normalized = {
        str(tool): {
            str(label).strip()
            for label in labels
            if str(label or "").strip()
        }
        for tool, labels in tool_route_intents.items()
        if isinstance(labels, list)
    }
    missing = [
        tool for tool in candidate_tools
        if not normalized.get(tool)
    ]
    if missing:
        return {
            "kind": "route_intent_unapplied",
            "reason": "missing_tool_route_intents",
            "route_intents": route_intents,
            "candidate_tool_names": list(candidate_tools),
            "missing_tool_route_intents": missing,
        }
    route_intent_set = set(route_intents)
    filtered = [
        tool for tool in candidate_tools
        if normalized.get(tool, set()) & route_intent_set
    ]
    return {
        "kind": "route_intent",
        "route_intents": route_intents,
        "candidate_tool_names": filtered,
        "before_count": len(candidate_tools),
        "after_count": len(filtered),
    }


def _intent_matrix_safetensors_bytes(weights: Any, bias: Any) -> bytes:
    import numpy as np
    from safetensors.numpy import save_file

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "route_intent_matrix.safetensors"
        save_file(
            {
                "intent_weights": np.asarray(weights, dtype=np.float16),
                "intent_bias": np.asarray(bias, dtype=np.float16),
            },
            str(path),
            metadata={"format": "edgestudio.route_router.matrix_v0"},
        )
        return path.read_bytes()


def _intent_logits(embeddings: Any, weights: Any, bias: Any) -> Any:
    return embeddings @ weights + bias


def _softmax(logits: Any) -> Any:
    import numpy as np

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _select_calibration_temperature(
    logits: Any,
    labels: Any,
    *,
    config: RouteRouterTrainConfig,
) -> float:
    def score(temp: float) -> float:
        probs = _softmax(logits / temp)
        if config.calibration_temperature_objective == "nll":
            return _nll(probs, labels)
        return _expected_calibration_error(probs, labels)

    return min(
        (float(temp) for temp in config.calibration_temperatures),
        key=score,
    )


def _nll(probs: Any, labels: Any) -> float:
    import numpy as np

    selected = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-8, 1.0)).mean())


def _calibrated_thresholds(
    probs: Any,
    labels: Any,
    intent_vocab: list[str],
    *,
    percentile: float = 5.0,
    max_threshold: float = 0.98,
) -> dict[str, float]:
    import numpy as np

    default = build_route_router_calibration()["intent_thresholds"]
    predictions = probs.argmax(axis=1)
    top_probs = probs.max(axis=1)
    thresholds: dict[str, float] = {}
    pct = min(100.0, max(0.0, float(percentile)))
    cap = min(1.0, max(0.0, float(max_threshold)))
    for idx, label in enumerate(intent_vocab):
        correct = top_probs[(labels == idx) & (predictions == idx)]
        if len(correct) == 0:
            thresholds[label] = float(default.get(label, 0.6))
        else:
            # Use a low percentile instead of the most extreme calibration row,
            # then cap the value so tiny per-class calibration sets cannot
            # require near-perfect probabilities for every valid runtime route.
            threshold = max(float(default.get(label, 0.6)), float(np.percentile(correct, pct)))
            thresholds[label] = float(min(cap, threshold))
    return thresholds


def _expected_calibration_error(
    probs: Any,
    labels: Any,
    *,
    bins: int = 10,
) -> float:
    import numpy as np

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float32)
    ece = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        mask = (confidences > low) & (confidences <= high)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += float(mask.mean() * abs(bin_acc - bin_conf))
    return float(ece)


def _per_class_f1(
    labels: Any,
    predictions: Any,
    intent_vocab: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx, label in enumerate(intent_vocab):
        true_positive = int(((labels == idx) & (predictions == idx)).sum())
        false_positive = int(((labels != idx) & (predictions == idx)).sum())
        false_negative = int(((labels == idx) & (predictions != idx)).sum())
        denom = (2 * true_positive) + false_positive + false_negative
        out[label] = float((2 * true_positive) / denom) if denom else 0.0
    return out


def _l2_normalize(matrix: Any) -> Any:
    import numpy as np

    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-8)


def _clusters(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        clusters.setdefault(_cluster_key(sample), []).append(sample)
    for rows in clusters.values():
        rows.sort(key=lambda row: (_sample_id(row), _sample_prompt(row)))
    return clusters


def _select_unseen_tool_cluster(
    clusters: dict[str, list[dict[str, Any]]],
    ordered_cluster_keys: list[str],
    reserved: set[str] | None = None,
) -> tuple[str | None, str | None]:
    reserved = reserved or set()
    tool_to_clusters: dict[str, list[str]] = {}
    for cluster_key in ordered_cluster_keys:
        if cluster_key in reserved:
            continue
        tools = sorted({
            tool
            for row in clusters[cluster_key]
            for tool in _selected_tools(row)
        })
        for tool in tools:
            tool_to_clusters.setdefault(tool, []).append(cluster_key)
    if not tool_to_clusters:
        return None, None
    tool = sorted(tool_to_clusters)[-1]
    cluster = _last_available_cluster(
        tool_to_clusters[tool],
        reserved,
        clusters=clusters,
        all_cluster_keys=ordered_cluster_keys,
        preserve_train_label_coverage=True,
    )
    return tool, cluster or tool_to_clusters[tool][-1]


def _last_available_cluster(
    ordered_cluster_keys: list[str],
    reserved: set[str],
    *,
    clusters: dict[str, list[dict[str, Any]]] | None = None,
    all_cluster_keys: list[str] | None = None,
    preserve_train_label_coverage: bool = False,
    allow_coverage_fallback: bool = True,
) -> str | None:
    fallback = None
    for cluster_key in reversed(ordered_cluster_keys):
        if cluster_key in reserved:
            continue
        if fallback is None:
            fallback = cluster_key
        if (
            preserve_train_label_coverage
            and clusters is not None
            and all_cluster_keys is not None
            and not _preserves_train_label_coverage(
                clusters,
                all_cluster_keys,
                reserved,
                cluster_key,
            )
        ):
            continue
        return cluster_key
    if preserve_train_label_coverage and allow_coverage_fallback:
        return fallback
    return None


def _last_available_cluster_matching_intents(
    ordered_cluster_keys: list[str],
    reserved: set[str],
    *,
    required_intents: set[str],
    clusters: dict[str, list[dict[str, Any]]],
    all_cluster_keys: list[str],
    preserve_train_label_coverage: bool,
) -> str | None:
    for cluster_key in reversed(ordered_cluster_keys):
        if cluster_key in reserved:
            continue
        if not (_cluster_intent_labels(clusters[cluster_key]) & required_intents):
            continue
        if (
            preserve_train_label_coverage
            and not _preserves_train_label_coverage(
                clusters,
                all_cluster_keys,
                reserved,
                cluster_key,
            )
        ):
            continue
        return cluster_key
    return None


def _preserves_train_label_coverage(
    clusters: dict[str, list[dict[str, Any]]],
    all_cluster_keys: list[str],
    reserved: set[str],
    candidate: str,
) -> bool:
    candidate_labels = _cluster_intent_labels(clusters[candidate])
    remaining_labels = {
        label
        for cluster_key in all_cluster_keys
        if cluster_key not in reserved and cluster_key != candidate
        for label in _cluster_intent_labels(clusters[cluster_key])
    }
    return candidate_labels.issubset(remaining_labels)


def _cluster_intent_labels(rows: list[dict[str, Any]]) -> set[str]:
    return {
        _sample_target(row)["route_intent"]
        for row in rows
    }


def _split_primary_and_variants(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [
        row for row in rows
        if not _source(row).get("prompt_variant_of")
    ]
    variants = [
        row for row in rows
        if _source(row).get("prompt_variant_of")
    ]
    if not primary and rows:
        return [rows[0]], rows[1:]
    return primary, variants


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    text = _sample_prompt(sample)
    target = _sample_target(sample)
    normalized = json.loads(json.dumps(sample, ensure_ascii=False))
    normalized.setdefault("input", {})["text"] = text
    normalized["target"] = target
    normalized.setdefault("source", {})
    normalized.setdefault("sample_id", _fingerprint({
        "text": text,
        "target": target,
    }))
    return normalized


def _sample_prompt(sample: dict[str, Any]) -> str:
    input_payload = sample.get("input") if isinstance(sample.get("input"), dict) else {}
    return _required_str(input_payload.get("text"), "sample.input.text")


def _sample_target(sample: dict[str, Any]) -> dict[str, Any]:
    target = sample.get("target")
    if not isinstance(target, dict):
        raise ValueError("sample.target must be an object")
    route_intent = _required_str(target.get("route_intent"), "sample.target.route_intent")
    normalized = dict(target)
    normalized["route_intent"] = route_intent
    selected_tools = target.get("selected_tools")
    if selected_tools is None:
        normalized["selected_tools"] = []
    elif isinstance(selected_tools, list):
        normalized["selected_tools"] = [
            str(tool).strip()
            for tool in selected_tools
            if str(tool or "").strip()
        ]
    else:
        raise ValueError("sample.target.selected_tools must be a list")
    tool_call_plan = target.get("tool_call_plan")
    if tool_call_plan is None:
        normalized["tool_call_plan"] = []
    elif not isinstance(tool_call_plan, list):
        raise ValueError("sample.target.tool_call_plan must be a list")
    return normalized


def _source(sample: dict[str, Any]) -> dict[str, Any]:
    source = sample.get("source")
    return source if isinstance(source, dict) else {}


def _cluster_key(sample: dict[str, Any]) -> str:
    source = _source(sample)
    case_id = source.get("case_id")
    if isinstance(case_id, str) and case_id.strip():
        return f"case:{case_id.strip()}"
    prompt_variant_of = source.get("prompt_variant_of")
    if isinstance(prompt_variant_of, str) and prompt_variant_of.strip():
        return _prompt_key(prompt_variant_of)
    return _prompt_key(_sample_prompt(sample))


def _sample_id(sample: dict[str, Any]) -> str:
    raw = sample.get("sample_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _fingerprint({
        "prompt": _sample_prompt(sample),
        "target": _sample_target(sample),
    })


def _selected_tools(sample: dict[str, Any]) -> list[str]:
    return [
        str(tool).strip()
        for tool in _sample_target(sample).get("selected_tools", [])
        if str(tool or "").strip()
    ]


def _intent_label_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        label = _sample_target(sample)["route_intent"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _tool_label_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        for tool in _selected_tools(sample):
            counts[tool] = counts.get(tool, 0) + 1
    return dict(sorted(counts.items()))


def _prompt_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower()).rstrip("?!。！？.")


def _fingerprint(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        return b""
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def _required_str(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text
