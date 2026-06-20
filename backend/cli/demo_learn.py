# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Learning demo command: B5a dry-run planning for correction-triggered regen."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.cli.demo_imprint import (
    _generate_answer,
    _restore_neural_imprint_runtime,
)
from backend.cli.demo_samples import LearnDemoSample, resolve_learn_demo_sample
from backend.cli.fingerprints import directory_manifest_hash, pretty_json, sha256_hex, sha256_prefixed, utc_now_iso
from backend.cli.model_fetch import FetchOptions, FetchResult, SourceOption, fetch_model
from backend.cli.models import ModelWhereReport, where_model
from backend.services.app_dirs import data_path
from backend.services.correction_ledger import record_correction_entry
from backend.services.correction_regen_coordinator import (
    CorrectionRegenError,
    regenerate_neural_imprint_from_corrections,
)
from backend.services.neural_imprint_generation import get_neural_imprint_generation_job
from backend.services.persona_rpp_input_contract import store_persona_rpp_input_contract


LEARN_PLAN_SCHEMA_VERSION = "edge.demo.learn.plan.v1"
LEARN_RUN_SCHEMA_VERSION = "edge.demo.learn.run.v1"
LEARN_RECEIPT_SCHEMA_VERSION = "edge.demo.learn.receipt.v1"
DEFAULT_LEARN_SAMPLE_ID = "finance_conservative_cashflow_v1"
DEFAULT_MODEL_REF = "qwen3.5-9b-4bit"
DEFAULT_MAX_TOKENS = 128
GENERATION_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class LearnRunOptions:
    sample_id: str = DEFAULT_LEARN_SAMPLE_ID
    model_ref: str = "auto"
    question: str = ""
    dry_run: bool = False
    include_text: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    prepare_model: bool = False
    model_source: SourceOption = "auto"
    download_dir: Path | None = None
    no_probe: bool = False
    force_fetch: bool = False
    fetch_timeout_seconds: float | None = None


@dataclass(frozen=True)
class LearnPlanResult:
    ok: bool
    exit_code: int
    plan: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.plan, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class LearnRunResult:
    ok: bool
    exit_code: int
    report: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2)


def plan_learn_run(
    *,
    options: LearnRunOptions,
    env: Mapping[str, str] | None = None,
) -> LearnPlanResult | LearnRunResult:
    if not options.dry_run:
        return _run_learn_demo(options=options, env=env)

    try:
        sample = resolve_learn_demo_sample(options.sample_id)
    except ValueError as exc:
        plan = _base_plan(options)
        plan.update(
            {
                "ok": False,
                "status": "unknown_sample",
                "error": {"code": "unknown_sample", "message": str(exc)},
            }
        )
        return LearnPlanResult(False, 1, plan)

    model_ref = DEFAULT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    model = _model_plan(model_ref, where, local_match)
    ok = local_match is not None
    status = "ready" if ok else "missing_model"
    if not ok and options.prepare_model:
        status = "model_prepare_required"
    run_id = _plan_run_id(options, sample=sample)
    question = options.question.strip() or sample.question

    plan = _base_plan(options, run_id=run_id)
    plan.update(
        {
            "ok": ok,
            "status": status,
            "sample": sample.as_plan_summary(),
            "model": model,
            "question_sha256": sha256_prefixed(question.encode("utf-8")),
            "raw_text_included": options.include_text,
            "network_used_during_plan": False,
            "planned_receipt_schema": LEARN_RECEIPT_SCHEMA_VERSION,
            "planned_receipt_path": str(default_learn_receipt_path(run_id)),
            "planned_state_root": str(default_learn_state_root(run_id)),
            "model_prepare": _planned_model_prepare(options, model_ref=model_ref, local_match=local_match),
            "planned_steps": [
                "prepare a compatible local model if --prepare-model is set and no complete local match exists",
                "store synthetic Persona/RPP input under isolated demo state",
                "record synthetic correction under isolated correction ledger",
                "run correction regen with explicit tool_schema_export support",
                "restore regenerated local Neural Imprint artifact",
                "compare before/after answer hashes",
                "write edge.demo.learn.receipt.v1 local-only receipt",
            ],
            "preflight": _preflight(local_match),
            "audit": {
                "dry_run_only": True,
                "writes_global_user_state": False,
                "writes_correction_ledger": False,
                "calls_regen_api": False,
                "loads_model": False,
                "network_used": False,
                "synthetic_fixture_only": True,
                "model_downloads_without_explicit_flag": False,
            },
        }
    )
    if options.include_text:
        plan["include_text_acknowledged"] = True
        plan["question"] = question
        plan["sample_text"] = sample.as_text_preview()
    if not ok and not options.prepare_model:
        plan["error"] = {
            "code": "missing_model",
            "message": "A local compatible model is required before running the learn demo.",
            "remediation": where.fetch_command or f"edge models fetch {model_ref}",
        }
    return LearnPlanResult(ok, 0 if ok else 1, plan)


def _run_learn_demo(
    *,
    options: LearnRunOptions,
    env: Mapping[str, str] | None = None,
) -> LearnRunResult:
    try:
        sample = resolve_learn_demo_sample(options.sample_id)
    except ValueError as exc:
        return _run_error("unknown_sample", str(exc), options)

    model_ref = DEFAULT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    if local_match is None or (options.prepare_model and options.force_fetch):
        if not options.prepare_model:
            return _run_error(
                "missing_model",
                "A local compatible model is required before running the learn demo.",
                options,
                remediation=where.fetch_command or f"edge models fetch {model_ref}",
            )
        _progress("fetch", f"preparing model={model_ref}")
        fetch_result = fetch_model(
            model_ref,
            options=FetchOptions(
                source=options.model_source,
                download_dir=options.download_dir,
                no_probe=options.no_probe,
                force=options.force_fetch,
                timeout_seconds=options.fetch_timeout_seconds,
            ),
            env=env,
        )
        model_prepare = _model_prepare_from_fetch(fetch_result)
        if not fetch_result.ok:
            return _run_error(
                "model_prepare_failed",
                "Could not prepare the local model required for the learn demo.",
                options,
                remediation=where.fetch_command or f"edge models fetch {model_ref}",
                model_prepare=model_prepare,
            )
        where = where_model(model_ref, env=env)
        local_match = _first_complete_match(where)
        if local_match is None:
            return _run_error(
                "model_prepare_no_local_match",
                "Model preparation completed but no complete compatible local model was discovered.",
                options,
                remediation=f"edge models where {model_ref}",
                model_prepare=model_prepare,
            )
    else:
        model_prepare = _model_prepare_not_needed(options, model_ref=model_ref, local_match=local_match)

    model_path = Path(local_match.path)
    question = options.question.strip() or sample.question
    run_id = _run_id(options, sample=sample)
    started = time.time()
    state_root = default_learn_state_root(run_id)
    roots = _isolated_roots(state_root)

    _progress("load", f"model={model_path.name}")
    _progress("before", "generating answer before correction regen")
    try:
        before_answer = _generate_answer(
            model_id=model_ref,
            model_path=model_path,
            prompt=question,
            max_tokens=max(1, options.max_tokens),
            use_neural_imprint=False,
        )
    except Exception as exc:
        return _run_error("before_generation_failed", f"Failed to generate before answer: {exc}", options)

    _progress("ledger", "writing synthetic correction into isolated demo state")
    try:
        store_persona_rpp_input_contract(
            sample.rpp_input_payload,
            root=roots["rpp_inputs"],
        )
        correction_receipts = [
            record_correction_entry(
                correction,
                root=roots["correction_ledger"],
            )
            for correction in sample.corrections
        ]
    except Exception as exc:
        return _run_error("isolated_state_write_failed", f"Failed to write isolated learn state: {exc}", options)

    _progress("regen", "queueing correction-triggered Neural Imprint regen")
    try:
        with _temporary_env(
            {
                "EDGE_PERSONA_SOURCE_ROOT": str(roots["persona_sources"]),
                "EDGE_NEURAL_IMPRINT_ARTIFACT_ROOTS": str(roots["neural_imprint_artifacts"]),
            }
        ):
            regen_receipt = regenerate_neural_imprint_from_corrections(
                peer_id=sample.peer_id,
                model_dir=model_path,
                model_id=model_path.name,
                base_model_id=model_ref,
                validate_restore=False,
                input_root=roots["rpp_inputs"],
                source_root=roots["persona_sources"],
                ledger_root=roots["correction_ledger"],
                tool_schema_export=sample.tool_schema_export,
            )
            job = _wait_for_generation_job(
                regen_receipt,
                timeout_seconds=GENERATION_TIMEOUT_SECONDS,
            )
    except (CorrectionRegenError, LearnDemoError) as exc:
        return _run_error(
            getattr(exc, "code", "correction_regen_failed"),
            getattr(exc, "message", str(exc)),
            options,
        )
    except Exception as exc:
        return _run_error("correction_regen_failed", f"Correction regen failed: {exc}", options)

    result = _job_result(job)
    artifact_path = Path(str(result.get("artifact_path") or ""))
    metadata_path = Path(str(result.get("metadata_path") or ""))

    _progress("restore", "restoring corrected Neural Imprint artifact")
    try:
        runtime_model_id = _restore_neural_imprint_runtime(
            model_path=model_path,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            artifact_id=f"learn-{run_id}",
        )
        after_answer = _generate_answer(
            model_id=runtime_model_id,
            model_path=model_path,
            prompt=question,
            max_tokens=max(1, options.max_tokens),
            use_neural_imprint=True,
        )
    except Exception as exc:
        return _run_error("artifact_restore_failed", f"Failed to restore corrected Neural Imprint artifact: {exc}", options)
    finally:
        try:
            from backend.services.neural_imprint_generation import _clear_mlx_cache

            _clear_mlx_cache()
        except Exception:
            pass

    try:
        model_manifest = directory_manifest_hash(model_path)
        artifact_sha256 = sha256_prefixed(artifact_path.read_bytes())
        metadata_sha256 = sha256_prefixed(metadata_path.read_bytes())
    except Exception as exc:
        return _run_error("fingerprint_failed", f"Failed to fingerprint learn demo outputs: {exc}", options)

    correction_fingerprints = [
        str(receipt.get("entry", {}).get("correction_fingerprint") or "")
        for receipt in correction_receipts
    ]
    included_fingerprints = _list_of_text(
        regen_receipt.get("compiled_correction_overlay", {}).get("included_correction_fingerprints")
    )
    receipt = {
        "schema_version": LEARN_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "model_ref": model_ref,
        "model_path": str(model_path),
        "model_sha256": model_manifest["sha256"],
        "model_sha256_scope": model_manifest.get("sha256_scope", "directory_manifest_v1"),
        "sample_id": sample.sample_id,
        "sample_sha256": sample.sample_sha256,
        "question_sha256": sha256_prefixed(question.encode("utf-8")),
        "rpp_input_sha256": sample.rpp_input_sha256,
        "correction_pack_sha256": sample.correction_pack_sha256,
        "correction_fingerprints": correction_fingerprints,
        "included_correction_fingerprints": included_fingerprints,
        "correction_overlay_sha256": _prefixed_optional(regen_receipt.get("correction_overlay_sha256")),
        "artifact_id": f"learn-{run_id}",
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "metadata_sha256": metadata_sha256,
        "model_prepare": model_prepare,
        "generation_job_id": str(job.get("job_id") or ""),
        "generation_status": str(job.get("status") or ""),
        "state_root": str(state_root),
        "before_answer_sha256": sha256_prefixed(before_answer["text"].encode("utf-8")),
        "before_answer_tokens": before_answer["token_count"],
        "after_answer_sha256": sha256_prefixed(after_answer["text"].encode("utf-8")),
        "after_answer_tokens": after_answer["token_count"],
        "answers_differ": before_answer["text"] != after_answer["text"],
        "raw_text_included": options.include_text,
        "network_used_during_demo": False,
        "network_used_during_model_prepare": _model_prepare_used_network(model_prepare),
        "status": "completed",
        "created_at": utc_now_iso(),
    }
    if options.include_text:
        receipt["include_text_acknowledged"] = True
        receipt["question"] = question
        receipt["before_answer"] = before_answer["text"]
        receipt["after_answer"] = after_answer["text"]
        receipt["sample_text"] = sample.as_text_preview()

    try:
        written_receipt_path = write_learn_receipt(receipt, run_id=run_id)
    except Exception as exc:
        return _run_error("receipt_write_failed", f"Failed to write learn receipt: {exc}", options)

    report: dict[str, Any] = {
        "schema_version": LEARN_RUN_SCHEMA_VERSION,
        "ok": True,
        "status": "completed",
        "run_id": run_id,
        "model": {
            "model_ref": model_ref,
            "path": str(model_path),
            "sha256": model_manifest["sha256"],
            "sha256_scope": model_manifest.get("sha256_scope", "directory_manifest_v1"),
        },
        "model_prepare": model_prepare,
        "sample": sample.as_plan_summary(),
        "question_sha256": receipt["question_sha256"],
        "state": {
            "root": str(state_root),
            "rpp_input_root": str(roots["rpp_inputs"]),
            "correction_ledger_root": str(roots["correction_ledger"]),
            "persona_source_root": str(roots["persona_sources"]),
            "artifact_root": str(roots["neural_imprint_artifacts"]),
            "writes_global_user_state": False,
        },
        "correction": {
            "correction_fingerprints": correction_fingerprints,
            "included_correction_fingerprints": included_fingerprints,
            "overlay_sha256": receipt["correction_overlay_sha256"],
        },
        "generation": {
            "job_id": receipt["generation_job_id"],
            "status": receipt["generation_status"],
            "artifact_path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "artifact_sha256": artifact_sha256,
            "metadata_sha256": metadata_sha256,
        },
        "comparison": {
            "before_answer_sha256": receipt["before_answer_sha256"],
            "before_answer_tokens": before_answer["token_count"],
            "after_answer_sha256": receipt["after_answer_sha256"],
            "after_answer_tokens": after_answer["token_count"],
            "answers_differ": before_answer["text"] != after_answer["text"],
        },
        "receipt_path": str(written_receipt_path),
        "raw_text_included": options.include_text,
        "network_used_during_demo": False,
        "network_used_during_model_prepare": _model_prepare_used_network(model_prepare),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    if options.include_text:
        report["question"] = question
        report["comparison"]["before_answer"] = before_answer["text"]
        report["comparison"]["after_answer"] = after_answer["text"]

    _progress("done", f"receipt={written_receipt_path}")
    return LearnRunResult(True, 0, report)


def default_learn_receipt_path(run_id: str) -> Path:
    return data_path("demo_runs", run_id, "learn_receipt.json")


def default_learn_state_root(run_id: str) -> Path:
    return data_path("demo_runs", run_id, "learn_state")


def write_learn_receipt(receipt: Mapping[str, Any], *, run_id: str | None = None, path: Path | None = None) -> Path:
    output_path = path or default_learn_receipt_path(run_id or str(receipt.get("run_id") or ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pretty_json(dict(receipt)), encoding="utf-8")
    return output_path


def format_learn_plan(result: LearnPlanResult) -> str:
    plan = result.plan
    lines = [
        f"Edge demo learn plan ({plan['schema_version']})",
        f"status: {plan['status']}",
        f"sample: {plan.get('sample', {}).get('sample_id', plan.get('sample_id'))}",
        f"model: {plan.get('model', {}).get('model_ref', plan.get('model_ref'))}",
        f"dry_run: {str(plan.get('dry_run')).lower()}",
    ]
    model = plan.get("model")
    if isinstance(model, dict):
        if model.get("path"):
            lines.append(f"model path: {model['path']}")
        if model.get("fetch_command"):
            lines.append(f"next: {model['fetch_command']}")
    model_prepare = plan.get("model_prepare")
    if isinstance(model_prepare, dict):
        lines.append(f"model_prepare: {model_prepare.get('status')}")
    if plan.get("question_sha256"):
        lines.append(f"question_sha256: {plan['question_sha256']}")
    if plan.get("planned_receipt_path"):
        lines.append(f"planned receipt: {plan['planned_receipt_path']}")
    if plan.get("planned_state_root"):
        lines.append(f"planned isolated state: {plan['planned_state_root']}")
    preflight = plan.get("preflight")
    if isinstance(preflight, dict):
        lines.append(f"real_run_status: {preflight.get('real_run_status')}")
    error = plan.get("error")
    if isinstance(error, dict):
        lines.append(f"error: {error.get('code')}: {error.get('message')}")
        if error.get("remediation"):
            lines.append(f"remediation: {error['remediation']}")
    return "\n".join(lines)


def format_learn_run(result: LearnRunResult) -> str:
    report = result.report
    if not result.ok:
        lines = [
            f"Edge demo learn ({report.get('schema_version')})",
            f"status: {report.get('status')}",
        ]
        model_prepare = report.get("model_prepare")
        if isinstance(model_prepare, dict):
            lines.append(f"model_prepare: {model_prepare.get('status')}")
        error = report.get("error")
        if isinstance(error, dict):
            lines.append(f"error: {error.get('code')}: {error.get('message')}")
            if error.get("remediation"):
                lines.append(f"remediation: {error['remediation']}")
        return "\n".join(lines)

    comp = report.get("comparison", {})
    lines = [
        f"Edge demo learn ({report.get('schema_version')})",
        f"status: {report.get('status')}",
        f"model: {report.get('model', {}).get('model_ref')}",
        f"model_prepare: {report.get('model_prepare', {}).get('status')}",
        f"sample: {report.get('sample', {}).get('sample_id')}",
        f"state: {report.get('state', {}).get('root')}",
        f"generation_job: {report.get('generation', {}).get('job_id')}",
        f"artifact: {report.get('generation', {}).get('artifact_path')}",
        f"metadata: {report.get('generation', {}).get('metadata_path')}",
        f"before_answer_sha256: {comp.get('before_answer_sha256')}",
        f"after_answer_sha256: {comp.get('after_answer_sha256')}",
        f"answers_differ: {comp.get('answers_differ')}",
        f"receipt: {report.get('receipt_path')}",
        f"next: edge demo chat --model {report.get('model', {}).get('model_ref')} --interactive --with-imprint \"{report.get('receipt_path')}\"",
        "raw_text_in_receipt: false" if not report.get("raw_text_included") else "raw_text_in_receipt: true",
    ]
    if report.get("raw_text_included"):
        lines.extend(["", "[Before]", str(comp.get("before_answer") or "")])
        lines.extend(["", "[After]", str(comp.get("after_answer") or "")])
    return "\n".join(lines)


def _base_plan(options: LearnRunOptions, *, run_id: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": LEARN_PLAN_SCHEMA_VERSION,
        "ok": False,
        "status": "planned",
        "run_id": run_id or _basic_plan_run_id(options),
        "sample_id": options.sample_id,
        "model_ref": options.model_ref,
        "dry_run": options.dry_run,
        "prepare_model": options.prepare_model,
    }


def _planned_model_prepare(
    options: LearnRunOptions,
    *,
    model_ref: str,
    local_match: Any,
) -> dict[str, Any]:
    if not options.prepare_model:
        return {
            "requested": False,
            "status": "not_requested",
            "network_used": False,
            "downloads_without_explicit_flag": False,
        }
    if local_match is not None and not options.force_fetch:
        return _model_prepare_not_needed(options, model_ref=model_ref, local_match=local_match)
    return {
        "requested": True,
        "status": "would_fetch_model",
        "model_ref": model_ref,
        "source": options.model_source,
        "download_dir": str(options.download_dir.expanduser()) if options.download_dir else None,
        "no_probe": options.no_probe,
        "force_fetch": options.force_fetch,
        "network_used": False,
        "downloads_without_explicit_flag": False,
    }


def _model_prepare_not_needed(
    options: LearnRunOptions,
    *,
    model_ref: str,
    local_match: Any,
) -> dict[str, Any]:
    path = str(getattr(local_match, "path", "") or "")
    return {
        "requested": options.prepare_model,
        "status": "skipped_existing",
        "model_ref": model_ref,
        "path": path,
        "network_used": False,
        "downloads_without_explicit_flag": False,
    }


def _model_prepare_from_fetch(result: FetchResult) -> dict[str, Any]:
    receipt = result.receipt
    attempts = receipt.get("attempted_sources")
    network_probe = receipt.get("network_probe")
    attempted_sources = [
        {
            "source": str(item.get("source") or ""),
            "returncode": item.get("returncode"),
            "timed_out": item.get("timed_out"),
        }
        for item in attempts
        if isinstance(item, dict)
    ] if isinstance(attempts, list) else []
    return {
        "requested": True,
        "ok": result.ok,
        "status": receipt.get("status"),
        "model_ref": receipt.get("model_ref"),
        "repo_id": receipt.get("repo_id"),
        "selected_source": receipt.get("selected_source"),
        "source_order": receipt.get("source_order") if isinstance(receipt.get("source_order"), list) else [],
        "attempted_sources": attempted_sources,
        "network_probe_count": len(network_probe) if isinstance(network_probe, list) else 0,
        "path": receipt.get("path"),
        "receipt_path": receipt.get("receipt_path"),
        "network_used": bool(attempted_sources or receipt.get("selected_source") or network_probe),
        "downloads_without_explicit_flag": False,
    }


def _model_prepare_used_network(model_prepare: Mapping[str, Any]) -> bool:
    return bool(model_prepare.get("network_used"))


def _preflight(local_match: Any) -> dict[str, Any]:
    tool_specs_found = False
    if local_match is not None:
        tool_specs_found = (Path(local_match.path) / "tool_specs.json").is_file()
    return {
        "isolated_state_roots_required": True,
        "writes_global_user_state": False,
        "model_dir_tool_specs_found": tool_specs_found,
        "b5b_tool_schema_strategy": "tool_schema_export_parameter",
        "real_run_status": "available",
        "real_run_blockers": [],
    }


@contextmanager
def _temporary_env(values: Mapping[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _isolated_roots(state_root: Path) -> dict[str, Path]:
    return {
        "rpp_inputs": state_root / "rpp_inputs",
        "correction_ledger": state_root / "correction_ledger",
        "persona_sources": state_root / "persona_sources",
        "neural_imprint_artifacts": state_root / "neural_imprint_artifacts",
    }


def _wait_for_generation_job(
    regen_receipt: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    job = regen_receipt.get("generation_job")
    if not isinstance(job, dict):
        raise LearnDemoError("generation_not_queued", "Correction regen did not queue a generation job.")
    job_id = str(job.get("job_id") or "")
    if not job_id:
        raise LearnDemoError("generation_job_missing", "Correction regen receipt did not include a job id.")
    deadline = time.time() + max(1.0, timeout_seconds)
    latest = job
    while time.time() < deadline:
        polled = get_neural_imprint_generation_job(job_id)
        if isinstance(polled, dict):
            latest = polled
        status = str(latest.get("status") or "")
        if status == "succeeded":
            return latest
        if status == "failed":
            error = latest.get("error")
            message = "Neural Imprint generation failed."
            if isinstance(error, dict) and error.get("message"):
                message = str(error["message"])
            raise LearnDemoError("generation_failed", message)
        time.sleep(0.1)
    raise LearnDemoError(
        "generation_timeout",
        f"Neural Imprint generation timed out after {int(timeout_seconds)}s; model may be too large for this device.",
    )


def _job_result(job: Mapping[str, Any]) -> dict[str, Any]:
    result = job.get("result")
    if not isinstance(result, dict):
        raise LearnDemoError("generation_result_missing", "Generation job completed without a result.")
    if not result.get("artifact_path") or not result.get("metadata_path"):
        raise LearnDemoError("generation_result_incomplete", "Generation result is missing artifact paths.")
    return result


def _first_complete_match(where: ModelWhereReport):
    for match in where.local_matches:
        if match.complete:
            return match
    return None


def _model_plan(model_ref: str, where: ModelWhereReport, local_match: Any) -> dict[str, object]:
    model: dict[str, object] = {
        "model_ref": model_ref,
        "resolution": where.resolution.as_dict(),
        "status": where.status,
        "fetch_command": where.fetch_command,
        "selection_reason": None,
    }
    if local_match is None:
        model["selection_reason"] = "no complete local compatible model found"
        return model
    path = Path(local_match.path)
    model.update(
        {
            "path": str(path),
            "size_bytes": local_match.size_bytes,
            "sha256": directory_manifest_hash(path)["sha256"],
            "sha256_scope": "directory_manifest_v1",
            "selection_reason": (
                "selected default qwen3.5-9b-4bit local match"
                if model_ref == DEFAULT_MODEL_REF
                else "selected requested local model match"
            ),
        }
    )
    return model


def _plan_run_id(options: LearnRunOptions, *, sample: LearnDemoSample) -> str:
    material = (
        options.sample_id
        + options.model_ref
        + (options.question.strip() or sample.question)
        + sample.sample_sha256
    )
    return f"edge-learn-plan-{sha256_hex(material.encode('utf-8'))[:12]}"


def _run_id(options: LearnRunOptions, *, sample: LearnDemoSample) -> str:
    material = options.sample_id + options.model_ref + (options.question.strip() or sample.question) + str(time.time())
    return f"edge-learn-{sha256_hex(material.encode('utf-8'))[:12]}"


def _basic_plan_run_id(options: LearnRunOptions) -> str:
    material = options.sample_id + options.model_ref + options.question + str(options.dry_run)
    return f"edge-learn-plan-{sha256_hex(material.encode('utf-8'))[:12]}"


def _run_error(
    code: str,
    message: str,
    options: LearnRunOptions,
    *,
    remediation: str | None = None,
    model_prepare: Mapping[str, Any] | None = None,
) -> LearnRunResult:
    error: dict[str, str] = {"code": code, "message": message}
    if remediation:
        error["remediation"] = remediation
    prepare = dict(model_prepare or {
        "requested": options.prepare_model,
        "status": "not_run",
        "network_used": False,
        "downloads_without_explicit_flag": False,
    })
    return LearnRunResult(
        ok=False,
        exit_code=1,
        report={
            "schema_version": LEARN_RUN_SCHEMA_VERSION,
            "ok": False,
            "status": code,
            "run_id": _basic_plan_run_id(options).replace("edge-learn-plan-", "edge-learn-error-"),
            "error": error,
            "model_prepare": prepare,
            "raw_text_included": False,
            "network_used_during_demo": False,
            "network_used_during_model_prepare": _model_prepare_used_network(prepare),
        },
    )


def _prefixed_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _list_of_text(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _progress(tag: str, message: str) -> None:
    print(f"[learn:{tag}] {message}", file=sys.stderr)




class LearnDemoError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
