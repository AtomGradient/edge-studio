# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Neural Imprint demo CLI: dry-run planning (B4a) and real orchestration (B4b).

Real orchestration loads a local model, captures a Neural Imprint artifact from
a synthetic sample profile, and compares base vs personalized answers.  All
computation is local-only; the receipt written at the end must pass the B6a
local-only validator.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.cli.demo_chat import _generate_streamed_answer
from backend.cli.demo_receipts import (
    DEMO_RECEIPT_SCHEMA_VERSION,
    default_demo_receipt_path,
    inspect_demo_receipt,
    write_demo_receipt,
)
from backend.cli.demo_samples import DemoSample, resolve_demo_sample
from backend.cli.fingerprints import (
    canonical_json_bytes,
    directory_manifest_hash,
    sha256_hex,
    sha256_prefixed,
    utc_now_iso,
)
from backend.cli.models import ModelWhereReport, where_model


IMPRINT_PLAN_SCHEMA_VERSION = "edge.demo.imprint.plan.v1"
IMPRINT_RUN_SCHEMA_VERSION = "edge.demo.imprint.run.v1"
IMPRINT_COMPARE_SCHEMA_VERSION = "edge.demo.imprint.compare.v1"
DEFAULT_SAMPLE_ID = "synthetic_profile_v1"
DEFAULT_MODEL_REF = "qwen3.5-9b-4bit"
DEFAULT_MAX_TOKENS = 220


@dataclass(frozen=True)
class ImprintPlanOptions:
    sample_id: str = DEFAULT_SAMPLE_ID
    model_ref: str = "auto"
    question: str = ""
    offline: bool = False
    dry_run: bool = True
    include_text: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class ImprintPlanResult:
    ok: bool
    exit_code: int
    plan: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.plan, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ImprintRunResult:
    ok: bool
    exit_code: int
    report: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ImprintCompareResult:
    ok: bool
    exit_code: int
    report: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# B4a: dry-run planning (unchanged from original)
# ---------------------------------------------------------------------------

def plan_imprint_run(
    *,
    options: ImprintPlanOptions,
    env: Mapping[str, str] | None = None,
) -> ImprintPlanResult:
    if not options.dry_run:
        return _run_imprint_demo(options=options, env=env)  # type: ignore[return-value]

    try:
        sample = resolve_demo_sample(options.sample_id)
    except ValueError as exc:
        plan = _base_plan(options)
        plan.update({
            "ok": False,
            "status": "unknown_sample",
            "error": {"code": "unknown_sample", "message": str(exc)},
        })
        return ImprintPlanResult(False, 1, plan)

    model_ref = DEFAULT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    model = _model_plan(model_ref, where, local_match)
    ok = local_match is not None
    status = "ready" if ok else "missing_model"

    plan = _base_plan(options)
    plan.update({
        "ok": ok,
        "status": status,
        "sample": sample.as_plan_summary(),
        "model": model,
        "question_sha256": sha256_prefixed(options.question.encode("utf-8")),
        "raw_text_included": False,
        "network_used_during_plan": False,
        "planned_receipt_schema": DEMO_RECEIPT_SCHEMA_VERSION,
        "planned_receipt_path": str(default_demo_receipt_path(str(plan["run_id"]))),
        "planned_steps": [
            "store synthetic persona source locally",
            "generate local Neural Imprint artifact",
            "restore local artifact under compatibility gates",
            "compare base and restored-artifact answers",
            "write edge.demo.receipt.v1 local-only receipt",
        ],
    })
    if not ok:
        plan["error"] = {
            "code": "missing_model",
            "message": "A local compatible model is required before running the Neural Imprint demo.",
            "remediation": where.fetch_command or f"edge models fetch {model_ref}",
        }
    return ImprintPlanResult(ok, 0 if ok else 1, plan)


# ---------------------------------------------------------------------------
# B4b: real orchestration
# ---------------------------------------------------------------------------

def _run_imprint_demo(
    *,
    options: ImprintPlanOptions,
    env: Mapping[str, str] | None = None,
) -> ImprintRunResult:
    """Execute the full Neural Imprint demo locally.

    Steps:
      1. Resolve sample and local model
      2. Load model + tokenizer
      3. Generate base answer (no Neural Imprint)
      4. Capture Neural Imprint artifact from synthetic profile
      5. Restore artifact and generate personalized answer
      6. Compare base vs personalized
      7. Write local-only receipt
    """
    try:
        sample = resolve_demo_sample(options.sample_id)
    except ValueError as exc:
        return _run_error("unknown_sample", str(exc), options)

    model_ref = DEFAULT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    if local_match is None:
        remediation = where.fetch_command or f"edge models fetch {model_ref}"
        return _run_error(
            "missing_model",
            "A local compatible model is required before running the Neural Imprint demo.",
            options,
            remediation=remediation,
        )

    model_path = Path(local_match.path)
    run_id = _run_id(options)

    # --- Late imports: heavy MLX dependencies only when actually running ---
    try:
        from mlx_lm.utils import load as mlx_load
    except ImportError:
        return _run_error(
            "mlx_not_installed",
            "mlx-lm is required for Neural Imprint demo. Install with: pip install mlx-lm",
            options,
        )

    try:
        from edgestudio_core.halo_capsule import full_cache
    except ImportError:
        return _run_error(
            "edgestudio_core_not_installed",
            "The bundled edgestudio_core runtime is required for Neural Imprint demo.",
            options,
        )

    try:
        from backend.services.neural_imprint_generation import (
            _build_system_prompt,
            _clear_mlx_cache,
            _contract_inputs,
            _fact_tool_schema_export,
            _forward_last_logits,
            _render_combined_prefix,
            _tools_list,
        )
    except ImportError as exc:
        return _run_error(
            "neural_imprint_generation_unavailable",
            f"Neural Imprint generation helpers unavailable: {exc}",
            options,
        )

    _progress("load", f"model={model_path.name}")
    started = time.time()
    try:
        model, tokenizer = mlx_load(str(model_path))
    except Exception as exc:
        return _run_error("model_load_failed", f"Failed to load model: {exc}", options)

    model_id = model_path.name
    question = options.question

    # Step 1: Base answer (no Neural Imprint)
    _progress("base", "generating base answer without Neural Imprint")
    try:
        base_answer = _generate_answer(
            model_id=model_ref,
            model_path=model_path,
            prompt=question,
            max_tokens=options.max_tokens,
            use_neural_imprint=False,
        )
    except Exception as exc:
        return _run_error("base_generation_failed", f"Failed to generate base answer: {exc}", options)

    # Step 2: Capture Neural Imprint artifact
    _progress("capture", "building local Neural Imprint artifact from synthetic profile")
    profile_body = sample.profile_body
    artifact_dir = default_demo_receipt_path(run_id).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "neural_imprint.safetensors"
    metadata_path = artifact_dir / "neural_imprint_metadata.json"
    cache: Any | None = None
    try:
        tool_schema_export = _fact_tool_schema_export(sample.tool_schema_export)
        tools = _tools_list(tool_schema_export)
        prefix_text, prefix_ids = _render_combined_prefix(
            tokenizer,
            tools=tools,
            profile_body=profile_body,
        )
        cache = full_cache.capture_full_cache(
            model,
            prefix_ids,
            forward_fn=_forward_last_logits,
        )

        metadata, source, model_info, tokenizer_info, prefix_info = _contract_inputs(
            model_dir=model_path,
            tokenizer=tokenizer,
            model_id=model_id,
            profile_body_sha256=sha256_hex(profile_body.encode("utf-8")),
            tool_schema_sha256=sha256_hex(
                canonical_json_bytes(tool_schema_export)
            ),
            system_prompt=_build_system_prompt(profile_body),
            rendered_prefix=prefix_text,
            prefix_token_ids=prefix_ids,
            created_at=utc_now_iso(),
        )

        _save_receipt = full_cache.save_full_cache(
            artifact=artifact_path,
            cache=cache,
            metadata=metadata,
            source=source,
            model_info=model_info,
            tokenizer_info=tokenizer_info,
            prefix_info=prefix_info,
            metadata_path=metadata_path,
        )
    except Exception as exc:
        return _run_error("artifact_capture_failed", f"Failed to capture Neural Imprint artifact: {exc}", options)
    finally:
        del cache
        _clear_mlx_cache()

    artifact_sha256 = sha256_prefixed(artifact_path.read_bytes()) if artifact_path.is_file() else ""
    metadata_sha256 = sha256_prefixed(metadata_path.read_bytes()) if metadata_path.is_file() else ""

    # Step 3: Restore artifact and generate personalized answer
    _progress("restore", "restoring local Neural Imprint artifact")
    try:
        runtime_model_id = _restore_neural_imprint_runtime(
            model_path=model_path,
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            artifact_id=f"demo-{run_id}",
        )

        _progress("personalized", "generating personalized answer with Neural Imprint active")
        persona_answer = _generate_answer(
            model_id=runtime_model_id,
            model_path=model_path,
            prompt=question,
            max_tokens=options.max_tokens,
            use_neural_imprint=True,
        )
    except Exception as exc:
        return _run_error("artifact_restore_failed", f"Failed to restore Neural Imprint artifact: {exc}", options)
    finally:
        _clear_mlx_cache()

    elapsed = time.time() - started

    # Step 4: Build comparison and receipt
    try:
        model_manifest = directory_manifest_hash(model_path)
    except Exception as exc:
        return _run_error("model_fingerprint_failed", f"Failed to fingerprint model directory: {exc}", options)
    include_text = options.include_text

    comparison = {
        "base_answer_sha256": sha256_prefixed(base_answer["text"].encode("utf-8")),
        "base_answer_tokens": base_answer["token_count"],
        "personalized_answer_sha256": sha256_prefixed(persona_answer["text"].encode("utf-8")),
        "personalized_answer_tokens": persona_answer["token_count"],
        "answers_differ": base_answer["text"] != persona_answer["text"],
    }

    receipt = {
        "schema_version": DEMO_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "model_path": str(model_path),
        "model_sha256": model_manifest["sha256"],
        "sample_id": sample.sample_id,
        "sample_sha256": sample.sample_sha256,
        "artifact_id": f"ni-{run_id}",
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "artifact_sha256": artifact_sha256,
        "metadata_sha256": metadata_sha256,
        "prefix_tokens": len(prefix_ids),
        **comparison,
        "raw_text_included": include_text,
        "network_used_during_demo": False,
        "status": "completed",
    }
    if include_text:
        receipt["include_text_acknowledged"] = True
        receipt["base_answer"] = base_answer["text"]
        receipt["personalized_answer"] = persona_answer["text"]

    try:
        receipt_path = write_demo_receipt(receipt, run_id=run_id)
    except Exception as exc:
        return _run_error("receipt_write_failed", f"Failed to write demo receipt: {exc}", options)

    report: dict[str, Any] = {
        "schema_version": IMPRINT_RUN_SCHEMA_VERSION,
        "ok": True,
        "status": "completed",
        "run_id": run_id,
        "model": {
            "model_ref": model_ref,
            "path": str(model_path),
            "id": model_id,
            "sha256": model_manifest["sha256"],
        },
        "sample": sample.as_plan_summary(),
        "artifact": {
            "path": str(artifact_path),
            "metadata_path": str(metadata_path),
            "artifact_sha256": artifact_sha256,
            "metadata_sha256": metadata_sha256,
            "prefix_tokens": len(prefix_ids),
        },
        "comparison": comparison,
        "receipt_path": str(receipt_path),
        "elapsed_seconds": round(elapsed, 2),
        "network_used_during_demo": False,
        "offline": options.offline,
    }

    if include_text:
        report["comparison"]["base_answer"] = base_answer["text"]
        report["comparison"]["personalized_answer"] = persona_answer["text"]

    _progress("done", f"receipt={receipt_path}")
    return ImprintRunResult(True, 0, report)


def compare_imprint_receipt(
    *,
    run_id: str | None = None,
    path: Path | None = None,
    include_text: bool = False,
) -> ImprintCompareResult:
    result = inspect_demo_receipt(run_id=run_id, path=path)
    if not result.ok or result.receipt is None:
        return _compare_error(
            "receipt_invalid",
            "Receipt could not be read or did not pass local-only validation.",
            receipt_path=result.receipt_path,
            validation=result.validation.as_dict(),
            detail=result.error,
        )

    receipt = result.receipt
    if receipt.get("schema_version") != DEMO_RECEIPT_SCHEMA_VERSION:
        return _compare_error(
            "schema_version_mismatch",
            f"Expected {DEMO_RECEIPT_SCHEMA_VERSION}.",
            receipt_path=result.receipt_path,
            validation=result.validation.as_dict(),
        )
    if receipt.get("status") != "completed":
        return _compare_error(
            "receipt_not_completed",
            "Only completed Neural Imprint demo receipts can be compared.",
            receipt_path=result.receipt_path,
            validation=result.validation.as_dict(),
        )

    required = (
        "base_answer_sha256",
        "base_answer_tokens",
        "personalized_answer_sha256",
        "personalized_answer_tokens",
        "answers_differ",
    )
    missing = [field for field in required if field not in receipt]
    if missing:
        return _compare_error(
            "comparison_missing",
            "Receipt does not contain B4 comparison fields.",
            receipt_path=result.receipt_path,
            validation=result.validation.as_dict(),
            detail=", ".join(missing),
        )
    invalid_types: list[str] = []
    if not isinstance(receipt.get("base_answer_tokens"), int):
        invalid_types.append("base_answer_tokens")
    if not isinstance(receipt.get("personalized_answer_tokens"), int):
        invalid_types.append("personalized_answer_tokens")
    if not isinstance(receipt.get("answers_differ"), bool):
        invalid_types.append("answers_differ")
    if invalid_types:
        return _compare_error(
            "comparison_invalid",
            "Receipt comparison fields have invalid types.",
            receipt_path=result.receipt_path,
            validation=result.validation.as_dict(),
            detail=", ".join(invalid_types),
        )

    comparison: dict[str, Any] = {
        "base_answer_sha256": receipt["base_answer_sha256"],
        "base_answer_tokens": receipt["base_answer_tokens"],
        "personalized_answer_sha256": receipt["personalized_answer_sha256"],
        "personalized_answer_tokens": receipt["personalized_answer_tokens"],
        "answers_differ": receipt["answers_differ"],
    }
    raw_text_displayed = False
    if include_text and receipt.get("raw_text_included") is True:
        if isinstance(receipt.get("base_answer"), str):
            comparison["base_answer"] = receipt["base_answer"]
            raw_text_displayed = True
        if isinstance(receipt.get("personalized_answer"), str):
            comparison["personalized_answer"] = receipt["personalized_answer"]
            raw_text_displayed = True

    report = {
        "schema_version": IMPRINT_COMPARE_SCHEMA_VERSION,
        "ok": True,
        "status": "completed",
        "run_id": receipt.get("run_id"),
        "receipt_path": str(result.receipt_path) if result.receipt_path else None,
        "model": {
            "path": receipt.get("model_path"),
            "sha256": receipt.get("model_sha256"),
        },
        "sample": {
            "sample_id": receipt.get("sample_id"),
            "sample_sha256": receipt.get("sample_sha256"),
        },
        "artifact": {
            "artifact_id": receipt.get("artifact_id"),
            "artifact_path": receipt.get("artifact_path"),
            "metadata_path": receipt.get("metadata_path"),
            "artifact_sha256": receipt.get("artifact_sha256"),
            "metadata_sha256": receipt.get("metadata_sha256"),
            "prefix_tokens": receipt.get("prefix_tokens"),
        },
        "comparison": comparison,
        "raw_text_included": receipt.get("raw_text_included") is True,
        "raw_text_displayed": raw_text_displayed,
        "network_used_during_compare": False,
        "validation": result.validation.as_dict(),
    }
    return ImprintCompareResult(True, 0, report)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_imprint_plan(result: ImprintPlanResult) -> str:
    plan = result.plan
    lines = [
        f"Edge demo imprint plan ({plan['schema_version']})",
        f"status: {plan['status']}",
        f"sample: {plan.get('sample', {}).get('sample_id', plan.get('sample_id'))}",
        f"model: {plan.get('model', {}).get('model_ref', plan.get('model_ref'))}",
        f"offline: {str(plan.get('offline')).lower()}",
    ]
    model = plan.get("model")
    if isinstance(model, dict):
        if model.get("path"):
            lines.append(f"model path: {model['path']}")
        if model.get("selection_reason"):
            lines.append(f"model selection: {model['selection_reason']}")
        if model.get("fetch_command"):
            lines.append(f"next: {model['fetch_command']}")
    if plan.get("question_sha256"):
        lines.append(f"question_sha256: {plan['question_sha256']}")
    if plan.get("planned_receipt_path"):
        lines.append(f"planned receipt: {plan['planned_receipt_path']}")
    error = plan.get("error")
    if isinstance(error, dict):
        lines.append(f"error: {error.get('code')}: {error.get('message')}")
        if error.get("remediation"):
            lines.append(f"remediation: {error['remediation']}")
    return "\n".join(lines)


def format_imprint_run(result: ImprintRunResult) -> str:
    r = result.report
    lines = [
        f"Edge Demo: Neural Imprint ({r['schema_version']})",
        f"status: {r['status']}",
    ]
    model = r.get("model", {})
    if model:
        lines.append(f"model: {model.get('id', model.get('model_ref'))}")
        if model.get("path"):
            lines.append(f"model path: {model['path']}")
    sample = r.get("sample", {})
    if sample:
        lines.append(f"sample: {sample.get('sample_id')}")
    artifact = r.get("artifact", {})
    if artifact:
        lines.append(f"artifact: {artifact.get('path')}")
        lines.append(f"artifact_sha256: {artifact.get('artifact_sha256')}")
        lines.append(f"prefix_tokens: {artifact.get('prefix_tokens')}")
    comp = r.get("comparison", {})
    if comp:
        lines.append("")
        lines.append("[Base answer]")
        if "base_answer" in comp:
            lines.append(comp["base_answer"])
        else:
            lines.append(f"  sha256: {comp.get('base_answer_sha256')}")
        lines.append(f"  tokens: {comp.get('base_answer_tokens')}")
        lines.append("")
        lines.append("[Personalized answer — behavior changed after restoring local Neural Imprint artifact]")
        if "personalized_answer" in comp:
            lines.append(comp["personalized_answer"])
        else:
            lines.append(f"  sha256: {comp.get('personalized_answer_sha256')}")
        lines.append(f"  tokens: {comp.get('personalized_answer_tokens')}")
        lines.append(f"  answers_differ: {comp.get('answers_differ')}")
    if r.get("receipt_path"):
        lines.append("")
        lines.append(f"receipt: {r['receipt_path']}")
    if r.get("elapsed_seconds"):
        lines.append(f"elapsed: {r['elapsed_seconds']}s")
    error = r.get("error")
    if isinstance(error, dict):
        lines.append(f"error: {error.get('code')}: {error.get('message')}")
        if error.get("remediation"):
            lines.append(f"remediation: {error['remediation']}")
    return "\n".join(lines)


def format_imprint_compare(result: ImprintCompareResult) -> str:
    r = result.report
    lines = [
        f"Edge demo imprint compare ({r.get('schema_version')})",
        f"status: {r.get('status')}",
    ]
    if r.get("receipt_path"):
        lines.append(f"receipt: {r['receipt_path']}")
    if r.get("run_id"):
        lines.append(f"run: {r['run_id']}")
    model = r.get("model")
    if isinstance(model, dict):
        if model.get("path"):
            lines.append(f"model path: {model['path']}")
        if model.get("sha256"):
            lines.append(f"model_sha256: {model['sha256']}")
    artifact = r.get("artifact")
    if isinstance(artifact, dict):
        if artifact.get("artifact_path"):
            lines.append(f"artifact: {artifact['artifact_path']}")
        if artifact.get("artifact_sha256"):
            lines.append(f"artifact_sha256: {artifact['artifact_sha256']}")
        if artifact.get("prefix_tokens") is not None:
            lines.append(f"prefix_tokens: {artifact['prefix_tokens']}")
    comp = r.get("comparison")
    if isinstance(comp, dict):
        lines.extend(
            [
                "",
                "[Base answer]",
                f"  sha256: {comp.get('base_answer_sha256')}",
                f"  tokens: {comp.get('base_answer_tokens')}",
                "",
                "[Personalized answer]",
                f"  sha256: {comp.get('personalized_answer_sha256')}",
                f"  tokens: {comp.get('personalized_answer_tokens')}",
                f"  answers_differ: {comp.get('answers_differ')}",
            ]
        )
        if r.get("raw_text_displayed"):
            lines.extend(["", "[Base text]", str(comp.get("base_answer") or "")])
            lines.extend(["", "[Personalized text]", str(comp.get("personalized_answer") or "")])
    error = r.get("error")
    if isinstance(error, dict):
        lines.append(f"error: {error.get('code')}: {error.get('message')}")
        if error.get("detail"):
            lines.append(f"detail: {error['detail']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _base_plan(options: ImprintPlanOptions) -> dict[str, Any]:
    run_id = f"edge-run-plan-{sha256_hex((options.sample_id + options.model_ref + options.question)[:512].encode())[:12]}"
    return {
        "schema_version": IMPRINT_PLAN_SCHEMA_VERSION,
        "ok": False,
        "status": "planned",
        "run_id": run_id,
        "sample_id": options.sample_id,
        "model_ref": options.model_ref,
        "offline": options.offline,
        "dry_run": options.dry_run,
    }


def _run_id(options: ImprintPlanOptions) -> str:
    fingerprint = sha256_hex(
        (options.sample_id + options.model_ref + options.question + str(time.time()))[:512].encode()
    )[:12]
    return f"edge-run-{fingerprint}"


def _run_error(
    code: str,
    message: str,
    options: ImprintPlanOptions,
    *,
    remediation: str | None = None,
) -> ImprintRunResult:
    error: dict[str, str] = {"code": code, "message": message}
    if remediation:
        error["remediation"] = remediation
    return ImprintRunResult(
        ok=False,
        exit_code=1,
        report={
            "schema_version": IMPRINT_RUN_SCHEMA_VERSION,
            "ok": False,
            "status": code,
            "run_id": _run_id(options),
            "error": error,
        },
    )


def _compare_error(
    code: str,
    message: str,
    *,
    receipt_path: Path | None,
    validation: Mapping[str, Any] | None = None,
    detail: str | None = None,
) -> ImprintCompareResult:
    error: dict[str, Any] = {"code": code, "message": message}
    if detail:
        error["detail"] = detail
    report: dict[str, Any] = {
        "schema_version": IMPRINT_COMPARE_SCHEMA_VERSION,
        "ok": False,
        "status": code,
        "receipt_path": str(receipt_path) if receipt_path else None,
        "error": error,
        "raw_text_displayed": False,
        "network_used_during_compare": False,
    }
    if validation is not None:
        report["validation"] = dict(validation)
    return ImprintCompareResult(False, 1, report)


def _first_complete_match(where: ModelWhereReport):
    for match in where.local_matches:
        if match.complete:
            return match
    return None


def _model_plan(model_ref: str, where: ModelWhereReport, local_match) -> dict[str, object]:
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
    model.update({
        "path": str(path),
        "size_bytes": local_match.size_bytes,
        "sha256": directory_manifest_hash(path)["sha256"],
        "sha256_scope": "directory_manifest_v1",
        "selection_reason": (
            "selected default qwen3.5-9b-4bit local match"
            if model_ref == DEFAULT_MODEL_REF
            else "selected requested local model match"
        ),
    })
    return model


def _progress(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", file=sys.stderr)


def _restore_neural_imprint_runtime(
    *,
    model_path: Path,
    artifact_path: Path,
    metadata_path: Path,
    artifact_id: str,
) -> str:
    from backend.services.model_manager import manager
    from backend.services.neural_imprint_runtime import restore_neural_imprint_for_model

    loaded = manager.load_model(str(model_path))
    status = restore_neural_imprint_for_model(
        model_id=loaded.model_id,
        artifact_path=artifact_path,
        sidecar_path=metadata_path,
        artifact_id=artifact_id,
    )
    if not status.active or not status.model_id:
        raise RuntimeError("Neural Imprint restore completed without an active runtime state.")
    return status.model_id


def _generate_answer(
    *,
    model_id: str,
    model_path: Path,
    prompt: str,
    max_tokens: int,
    use_neural_imprint: bool,
) -> dict[str, Any]:
    started = time.time()
    answer = _generate_streamed_answer(
        model_id=model_id,
        model_path=model_path,
        prompt=prompt,
        history=[],
        max_tokens=max(1, max_tokens),
        use_neural_imprint=use_neural_imprint,
    )
    text = str(answer.get("text") or "")
    return {
        "text": text,
        "text_sha256": sha256_hex(text.encode("utf-8")),
        "token_count": int(answer.get("token_count") or 0),
        "elapsed_seconds": round(float(answer.get("elapsed_seconds") or (time.time() - started)), 2),
    }
