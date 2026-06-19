# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""End-to-end host-side driver for route-matrix live feedback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from backend.services.host_model_assistant import HOST_MODEL_PROVIDER, HostModelGenerate
from backend.services.route_matrix_live_audit import (
    build_route_matrix_live_feedback_review_request_from_event_store,
)
from backend.services.route_matrix_live_feedback_baseline import (
    build_route_matrix_live_feedback_baseline_from_event_store,
)
from backend.services.route_matrix_live_feedback_import import (
    import_route_matrix_live_feedback_review,
)
from backend.services.route_matrix_live_feedback_review import (
    generate_route_matrix_live_feedback_review,
)
from backend.stores.event_store import EventStore


ROUTE_MATRIX_LIVE_FEEDBACK_PIPELINE_SCHEMA_VERSION = (
    "edgestudio.route_matrix_live_feedback_pipeline.v0"
)


def run_route_matrix_live_feedback_pipeline(
    *,
    run_id: str,
    app_id: str,
    tool_registry: list[dict[str, Any]],
    output_dir: Path,
    event_store: EventStore | None = None,
    event_store_path: Path | None = None,
    peer_id: str | None = None,
    host_model_id: str | None = None,
    provider: str | None = None,
    host_model_generate: HostModelGenerate | None = None,
    limit: int = 1000,
    max_cases: int = 50,
    import_limit: int = 1000,
    max_tokens: int | None = None,
    review_chunk_size: int | None = None,
    min_evaluable_n: int = 100,
    heldout_cutoff_ms: int | None = None,
    baseline_eval: dict[str, Any] | None = None,
    retrained_eval: dict[str, Any] | None = None,
    allow_fixture_corrections: bool = False,
    evidence_scope: str = "unspecified",
) -> dict[str, Any]:
    """Run the 3l-beta host-side feedback path over explicit user corrections.

    This driver is intentionally conservative. It never promotes raw live
    decision audits directly into training data; it only imports after the
    review request is populated, the Host Model emits reviewed pair payloads,
    and the existing route/action gates accept the imported rows.
    """

    generated_at_ms = int(time.time() * 1000)
    normalized_output_dir = Path(output_dir).expanduser()
    normalized_output_dir.mkdir(parents=True, exist_ok=True)

    review_request = build_route_matrix_live_feedback_review_request_from_event_store(
        run_id=run_id,
        app_id=app_id,
        event_store=event_store,
        event_store_path=event_store_path,
        peer_id=peer_id,
        limit=limit,
        max_cases=max_cases,
    )
    review_request_path = normalized_output_dir / "live_feedback_review_request.json"
    _write_json(review_request_path, review_request)

    request_summary = review_request["result"]["summary"]
    host_model_review: dict[str, Any] | None = None
    import_receipt: dict[str, Any] | None = None
    host_model_review_path: Path | None = None
    import_receipt_path: Path | None = None

    if request_summary["ready_for_host_model_review"] is True:
        host_model_review = generate_route_matrix_live_feedback_review(
            review_request=review_request,
            tool_registry=tool_registry,
            host_model_id=host_model_id,
            provider=provider or HOST_MODEL_PROVIDER,
            host_model_generate=host_model_generate,
            max_tokens=max_tokens,
            review_chunk_size=review_chunk_size,
        )
        host_model_review_path = normalized_output_dir / "live_feedback_host_model_review.json"
        _write_json(host_model_review_path, host_model_review)

        review_summary = (
            host_model_review.get("result", {}).get("summary")
            if isinstance(host_model_review.get("result"), dict)
            else {}
        )
        if (
            host_model_review.get("ok") is True
            and review_summary.get("ready_for_live_feedback_import") is True
        ):
            import_receipt = import_route_matrix_live_feedback_review(
                review_request=review_request,
                host_model_review_receipt=host_model_review,
                tool_registry=tool_registry,
                output_dir=normalized_output_dir / "learner_feedstock",
                peer_id=peer_id,
                event_store_path=normalized_output_dir
                / "live_feedback_import_events.sqlite",
                limit=import_limit,
                allow_fixture_corrections=allow_fixture_corrections,
            )
            import_receipt_path = normalized_output_dir / "live_feedback_import_receipt.json"
            _write_json(import_receipt_path, import_receipt)

    baseline = build_route_matrix_live_feedback_baseline_from_event_store(
        run_id=run_id,
        app_id=app_id,
        event_store=event_store,
        event_store_path=event_store_path,
        peer_id=peer_id,
        limit=limit,
        min_evaluable_n=min_evaluable_n,
        heldout_cutoff_ms=heldout_cutoff_ms,
        baseline_eval=baseline_eval,
        retrained_eval=retrained_eval,
        host_model_review_receipt=host_model_review,
        evidence_scope=evidence_scope,
    )
    baseline_path = normalized_output_dir / "live_feedback_baseline_receipt.json"
    _write_json(baseline_path, baseline)

    result = {
        "ok": _pipeline_ok(
            request_ready=request_summary["ready_for_host_model_review"] is True,
            host_model_review=host_model_review,
            import_receipt=import_receipt,
        ),
        "schema_version": ROUTE_MATRIX_LIVE_FEEDBACK_PIPELINE_SCHEMA_VERSION,
        "status": _pipeline_status(
            request_ready=request_summary["ready_for_host_model_review"] is True,
            host_model_review=host_model_review,
            import_receipt=import_receipt,
        ),
        "run_id": run_id,
        "app_id": app_id,
        "result": {
            "review_request": _receipt_summary(review_request),
            "host_model_review": _receipt_summary(host_model_review),
            "import": _receipt_summary(import_receipt),
            "baseline": _receipt_summary(baseline),
            "paths": {
                "output_dir": str(normalized_output_dir),
                "review_request": str(review_request_path),
                "host_model_review": (
                    str(host_model_review_path) if host_model_review_path else None
                ),
                "import_receipt": str(import_receipt_path) if import_receipt_path else None,
                "baseline": str(baseline_path),
            },
        },
        "error": _pipeline_error(host_model_review=host_model_review, import_receipt=import_receipt),
        "audit": {
            "method": "run_route_matrix_live_feedback_pipeline",
            "generated_at_ms": generated_at_ms,
            "training_side_only": True,
            "writes_runtime_artifacts": False,
            "writes_events": bool(
                import_receipt
                and import_receipt.get("audit", {}).get("writes_events") is True
            ),
            "writes_training_sample_tags": bool(
                import_receipt
                and import_receipt.get("audit", {}).get("writes_training_sample_tags")
                is True
            ),
            "host_model_called": host_model_review is not None,
            "allow_fixture_corrections": bool(allow_fixture_corrections),
        },
    }
    summary_path = normalized_output_dir / "live_feedback_pipeline_summary.json"
    result["result"]["paths"]["summary"] = str(summary_path)
    _write_json(summary_path, result)
    return result


def _pipeline_ok(
    *,
    request_ready: bool,
    host_model_review: dict[str, Any] | None,
    import_receipt: dict[str, Any] | None,
) -> bool:
    if not request_ready:
        return True
    if not host_model_review or host_model_review.get("ok") is not True:
        return False
    review_summary = (
        host_model_review.get("result", {}).get("summary")
        if isinstance(host_model_review.get("result"), dict)
        else {}
    )
    if review_summary.get("ready_for_live_feedback_import") is not True:
        return True
    return bool(import_receipt and import_receipt.get("ok") is True)


def _pipeline_status(
    *,
    request_ready: bool,
    host_model_review: dict[str, Any] | None,
    import_receipt: dict[str, Any] | None,
) -> str:
    if not request_ready:
        return "pending_real_user_corrections"
    if not host_model_review:
        return "host_model_review_not_started"
    if host_model_review.get("ok") is not True:
        return "host_model_review_failed"
    review_summary = (
        host_model_review.get("result", {}).get("summary")
        if isinstance(host_model_review.get("result"), dict)
        else {}
    )
    if review_summary.get("ready_for_live_feedback_import") is not True:
        return "host_model_review_not_importable"
    if not import_receipt:
        return "live_feedback_import_not_started"
    if import_receipt.get("ok") is not True:
        return "live_feedback_import_failed"
    if import_receipt.get("status") == "learner_dataset_written":
        return "learner_feedstock_written"
    return str(import_receipt.get("status") or "live_feedback_import_completed")


def _pipeline_error(
    *,
    host_model_review: dict[str, Any] | None,
    import_receipt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if host_model_review and host_model_review.get("ok") is not True:
        return {
            "stage": "host_model_review",
            "status": host_model_review.get("status"),
            "error": host_model_review.get("error"),
        }
    if import_receipt and import_receipt.get("ok") is not True:
        return {
            "stage": "live_feedback_import",
            "status": import_receipt.get("status"),
            "error": import_receipt.get("error"),
        }
    return None


def _receipt_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else None
    receipt = {
        "ok": value.get("ok"),
        "status": value.get("status"),
        "summary": summary,
        "error": value.get("error"),
    }
    if isinstance(value.get("import_summary"), dict):
        receipt["import_summary"] = value.get("import_summary")
    if isinstance(value.get("learner_dataset"), dict):
        receipt["learner_dataset"] = {
            "ok": value["learner_dataset"].get("ok"),
            "status": value["learner_dataset"].get("status"),
            "sample_count": value["learner_dataset"].get("sample_count"),
            "path": value["learner_dataset"].get("path"),
        }
    return receipt


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
