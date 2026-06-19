# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Command entry point for the ``edge`` developer CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from backend.cli.demo_chat import DEFAULT_MAX_TOKENS, ChatRunOptions, format_demo_chat, run_demo_chat, run_demo_chat_interactive
from backend.cli.demo_receipts import (
    demo_receipt_schema,
    format_demo_receipt,
    format_demo_schema,
    format_local_only,
    inspect_demo_receipt,
)
from backend.cli.demo_reuse import DemoReuseOptions, format_demo_reuse, run_demo_reuse
from backend.cli.demo_imprint import (
    ImprintPlanOptions,
    ImprintRunResult,
    compare_imprint_receipt,
    format_imprint_compare,
    format_imprint_plan,
    format_imprint_run,
    plan_imprint_run,
)
from backend.cli.demo_learn import (
    LearnRunResult,
    LearnRunOptions,
    format_learn_plan,
    format_learn_run,
    plan_learn_run,
)
from backend.cli.doctor import format_human, run_doctor
from backend.cli.model_fetch import FetchOptions, fetch_model, format_fetch_result
from backend.cli.models import (
    doctor_exit_code,
    doctor_model,
    format_model_doctor,
    format_model_where,
    format_models_list,
    list_models,
    where_exit_code,
    where_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge", description="Edge Developer Preview CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    studio = subparsers.add_parser("studio", help="Launch the local Edge Studio UI and API server")
    studio.add_argument("--host", help="Override VLM_HOST for the local server")
    studio.add_argument("--port", type=int, help="Override VLM_PORT for the local server")

    doctor = subparsers.add_parser("doctor", help="Run read-only environment checks")
    doctor.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    models = subparsers.add_parser("models", help="Inspect local model readiness")
    model_subparsers = models.add_subparsers(dest="models_command", required=True)

    models_list = model_subparsers.add_parser("list", help="List locally discovered models")
    models_list.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    models_where = model_subparsers.add_parser("where", help="Resolve a model and print its local path")
    models_where.add_argument("model", help="Catalog id, model name, or repository id")
    models_where.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    models_doctor = model_subparsers.add_parser("doctor", help="Diagnose model readiness without downloading")
    models_doctor.add_argument("model", help="Catalog id, model name, or repository id")
    models_doctor.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    models_fetch = model_subparsers.add_parser("fetch", help="Explicitly download a model and write a receipt")
    models_fetch.add_argument("model", help="Catalog id, model name, or repository id")
    models_fetch.add_argument(
        "--source",
        choices=["auto", "modelscope", "huggingface", "hf-mirror"],
        default="auto",
        help="Download source selection strategy",
    )
    models_fetch.add_argument("--download-dir", help="Directory for downloaded models")
    models_fetch.add_argument("--receipt", help="Receipt path override")
    models_fetch.add_argument("--dry-run", action="store_true", help="Resolve and print the fetch plan without downloading")
    models_fetch.add_argument("--no-probe", action="store_true", help="Skip network probes during auto source planning")
    models_fetch.add_argument("--force", action="store_true", help="Run the downloader even if a local match already exists")
    models_fetch.add_argument("--timeout", type=float, default=0.0, help="Per-source subprocess timeout in seconds; 0 disables")
    models_fetch.add_argument("--json", action="store_true", help="Emit a machine-readable receipt/report")

    demo = subparsers.add_parser("demo", help="Inspect demo receipts and local-only proofs")
    demo_subparsers = demo.add_subparsers(dest="demo_command", required=True)

    demo_chat = demo_subparsers.add_parser("chat", help="Run a local base-model chat demo")
    demo_chat.add_argument("--model", default="auto", help="Catalog id, model name, repo id, or auto")
    demo_chat.add_argument("--prompt", default="", help="Prompt for one-shot local base-model chat")
    demo_chat.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum tokens to generate; defaults to model generation config or model-aware "
            f"settings when available, otherwise {DEFAULT_MAX_TOKENS}"
        ),
    )
    demo_chat.add_argument("--include-text", action="store_true", help="Include raw prompt/answer text in JSON and receipt")
    demo_chat.add_argument(
        "--with-imprint",
        help="Load a completed learn receipt or Neural Imprint artifact before chatting",
    )
    demo_chat.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start a multi-turn local chat session; use /exit or /quit to stop",
    )
    demo_chat.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    demo_receipt = demo_subparsers.add_parser("receipt", help="Inspect a demo receipt")
    demo_receipt_source = demo_receipt.add_mutually_exclusive_group()
    demo_receipt_source.add_argument("--run", help="Demo run id under the EdgeStudio data directory")
    demo_receipt_source.add_argument("--path", help="Path to receipt.json")
    demo_receipt.add_argument("--schema", action="store_true", help="Print the expected receipt schema")
    demo_receipt.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    demo_local_only = demo_subparsers.add_parser("local-only", help="Validate local-only demo receipt invariants")
    demo_local_only_source = demo_local_only.add_mutually_exclusive_group(required=True)
    demo_local_only_source.add_argument("--run", help="Demo run id under the EdgeStudio data directory")
    demo_local_only_source.add_argument("--path", help="Path to receipt.json")
    demo_local_only.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    demo_reuse = demo_subparsers.add_parser("reuse", help="Simulate cross-App Neural Imprint artifact reuse from a receipt")
    demo_reuse_source = demo_reuse.add_mutually_exclusive_group(required=True)
    demo_reuse_source.add_argument("--run", help="Demo run id under the EdgeStudio data directory")
    demo_reuse_source.add_argument("--path", help="Path to a completed edge.demo.receipt.v1 receipt")
    demo_reuse_source.add_argument("--artifact", help="Path to a local artifact with a co-located receipt.json")
    demo_reuse.add_argument("--apps", default="notes,finance", help="Comma-separated synthetic app ids")
    demo_reuse.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    demo_imprint = demo_subparsers.add_parser("imprint", help="Plan Neural Imprint demo runs")
    demo_imprint_subparsers = demo_imprint.add_subparsers(dest="imprint_command", required=True)
    demo_imprint_run = demo_imprint_subparsers.add_parser("run", help="Plan a Neural Imprint demo run")
    demo_imprint_run.add_argument("--sample", default="synthetic_profile_v1", help="Synthetic sample id")
    demo_imprint_run.add_argument("--model", default="auto", help="Catalog id, model name, repo id, or auto")
    demo_imprint_run.add_argument("--question", required=True, help="Question to hash into the dry-run plan")
    demo_imprint_run.add_argument("--offline", action="store_true", help="Fail closed if local prerequisites are missing")
    demo_imprint_run.add_argument("--dry-run", action="store_true", help="Plan only; skip real model loading and inference")
    demo_imprint_run.add_argument("--include-text", action="store_true", help="Include raw answer text in receipt and report")
    demo_imprint_run.add_argument("--max-tokens", type=int, default=220, help="Maximum tokens to generate per answer")
    demo_imprint_run.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    demo_imprint_compare = demo_imprint_subparsers.add_parser("compare", help="Inspect a completed Neural Imprint demo receipt")
    demo_imprint_compare_source = demo_imprint_compare.add_mutually_exclusive_group(required=True)
    demo_imprint_compare_source.add_argument("--run", help="Demo run id under the EdgeStudio data directory")
    demo_imprint_compare_source.add_argument("--path", help="Path to receipt.json")
    demo_imprint_compare.add_argument("--include-text", action="store_true", help="Display raw answers only when the receipt explicitly included them")
    demo_imprint_compare.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    demo_learn = demo_subparsers.add_parser("learn", help="Plan correction learning demo runs")
    demo_learn_subparsers = demo_learn.add_subparsers(dest="learn_command", required=True)
    demo_learn_run = demo_learn_subparsers.add_parser("run", help="Plan a correction learning demo run")
    demo_learn_run.add_argument("--sample", default="synthetic_profile_correction_v1", help="Synthetic learn sample id")
    demo_learn_run.add_argument("--model", default="auto", help="Catalog id, model name, repo id, or auto")
    demo_learn_run.add_argument("--question", default="", help="Question to hash into the learn plan")
    demo_learn_run.add_argument("--dry-run", action="store_true", help="Plan only; skip correction ledger writes and regen")
    demo_learn_run.add_argument("--include-text", action="store_true", help="Include raw synthetic fixture text in the plan")
    demo_learn_run.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate per answer")
    demo_learn_run.add_argument(
        "--prepare-model",
        action="store_true",
        help="Explicitly prepare the local model first if no complete local match is found",
    )
    demo_learn_run.add_argument(
        "--source",
        choices=["auto", "modelscope", "huggingface", "hf-mirror"],
        default="auto",
        help="Model download source used only with --prepare-model",
    )
    demo_learn_run.add_argument("--download-dir", help="Directory for model downloads used only with --prepare-model")
    demo_learn_run.add_argument("--no-probe", action="store_true", help="Skip network probes during --prepare-model source planning")
    demo_learn_run.add_argument("--force-fetch", action="store_true", help="Force model fetch when --prepare-model is set")
    demo_learn_run.add_argument(
        "--fetch-timeout",
        type=float,
        default=0.0,
        help="Per-source model fetch subprocess timeout in seconds; 0 disables",
    )
    demo_learn_run.add_argument("--json", action="store_true", help="Emit a machine-readable report")

    return parser


def run_studio_server(*, host: str | None = None, port: int | None = None) -> int:
    if host or port:
        import os

        if host:
            os.environ["VLM_HOST"] = host
        if port:
            os.environ["VLM_PORT"] = str(port)

    from backend.main import main as studio_main

    studio_main()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "studio":
        return run_studio_server(host=args.host, port=args.port)

    if args.command == "doctor":
        report = run_doctor()
        if args.json:
            print(report.to_json())
        else:
            print(format_human(report))
        return 1 if report.overall_status == "fail" else 0

    if args.command == "models":
        if args.models_command == "list":
            report = list_models()
            print(report.to_json() if args.json else format_models_list(report))
            return 0
        if args.models_command == "where":
            report = where_model(args.model)
            print(report.to_json() if args.json else format_model_where(report))
            return where_exit_code(report)
        if args.models_command == "doctor":
            report = doctor_model(args.model)
            print(report.to_json() if args.json else format_model_doctor(report))
            return doctor_exit_code(report)
        if args.models_command == "fetch":
            result = fetch_model(
                args.model,
                options=FetchOptions(
                    source=args.source,
                    download_dir=Path(args.download_dir).expanduser() if args.download_dir else None,
                    receipt_path=Path(args.receipt).expanduser() if args.receipt else None,
                    dry_run=args.dry_run,
                    no_probe=args.no_probe,
                    force=args.force,
                    timeout_seconds=args.timeout if args.timeout and args.timeout > 0 else None,
                ),
            )
            print(result.to_json() if args.json else format_fetch_result(result))
            return result.exit_code
        parser.error(f"unknown models command: {args.models_command}")
        return 2

    if args.command == "demo":
        if args.demo_command == "chat":
            if args.interactive and args.json:
                parser.error("edge demo chat --interactive writes an interactive transcript and does not support --json")
            if not args.interactive and not args.prompt:
                parser.error("edge demo chat requires --prompt unless --interactive is set")
            if args.interactive:
                result = run_demo_chat_interactive(
                    options=ChatRunOptions(
                        model_ref=args.model,
                        prompt="",
                        max_tokens=args.max_tokens,
                        include_text=args.include_text,
                        interactive=True,
                        with_imprint=Path(args.with_imprint).expanduser() if args.with_imprint else None,
                    )
                )
                return result.exit_code
            result = run_demo_chat(
                options=ChatRunOptions(
                    model_ref=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    include_text=args.include_text,
                    with_imprint=Path(args.with_imprint).expanduser() if args.with_imprint else None,
                )
            )
            print(result.to_json() if args.json else format_demo_chat(result))
            return result.exit_code
        if args.demo_command == "receipt":
            if args.schema:
                payload = demo_receipt_schema()
                print(json.dumps(payload, ensure_ascii=False, indent=2) if args.json else format_demo_schema())
                return 0
            if not args.run and not args.path:
                parser.error("edge demo receipt requires --run, --path, or --schema")
            result = inspect_demo_receipt(
                run_id=args.run,
                path=Path(args.path).expanduser() if args.path else None,
            )
            print(result.to_json(schema_version="edge.demo.receipt.inspect.v1") if args.json else format_demo_receipt(result))
            return result.exit_code
        if args.demo_command == "local-only":
            result = inspect_demo_receipt(
                run_id=args.run,
                path=Path(args.path).expanduser() if args.path else None,
            )
            print(result.to_json(schema_version="edge.demo.local_only.report.v1") if args.json else format_local_only(result))
            return result.exit_code
        if args.demo_command == "reuse":
            result = run_demo_reuse(
                options=DemoReuseOptions(
                    run_id=args.run,
                    receipt_path=Path(args.path).expanduser() if args.path else None,
                    artifact_path=Path(args.artifact).expanduser() if args.artifact else None,
                    apps=args.apps,
                )
            )
            print(result.to_json() if args.json else format_demo_reuse(result))
            return result.exit_code
        if args.demo_command == "imprint":
            if args.imprint_command == "run":
                result = plan_imprint_run(
                    options=ImprintPlanOptions(
                        sample_id=args.sample,
                        model_ref=args.model,
                        question=args.question,
                        offline=args.offline,
                        dry_run=args.dry_run,
                        include_text=getattr(args, "include_text", False),
                        max_tokens=getattr(args, "max_tokens", 220),
                    )
                )
                if isinstance(result, ImprintRunResult):
                    print(result.to_json() if args.json else format_imprint_run(result))
                else:
                    print(result.to_json() if args.json else format_imprint_plan(result))
                return result.exit_code
            if args.imprint_command == "compare":
                result = compare_imprint_receipt(
                    run_id=args.run,
                    path=Path(args.path).expanduser() if args.path else None,
                    include_text=getattr(args, "include_text", False),
                )
                print(result.to_json() if args.json else format_imprint_compare(result))
                return result.exit_code
            parser.error(f"unknown demo imprint command: {args.imprint_command}")
            return 2
        if args.demo_command == "learn":
            if args.learn_command == "run":
                result = plan_learn_run(
                    options=LearnRunOptions(
                        sample_id=args.sample,
                        model_ref=args.model,
                        question=args.question,
                        dry_run=args.dry_run,
                        include_text=getattr(args, "include_text", False),
                        max_tokens=getattr(args, "max_tokens", 128),
                        prepare_model=getattr(args, "prepare_model", False),
                        model_source=getattr(args, "source", "auto"),
                        download_dir=Path(args.download_dir).expanduser() if getattr(args, "download_dir", None) else None,
                        no_probe=getattr(args, "no_probe", False),
                        force_fetch=getattr(args, "force_fetch", False),
                        fetch_timeout_seconds=(
                            args.fetch_timeout if getattr(args, "fetch_timeout", 0.0) and args.fetch_timeout > 0 else None
                        ),
                    )
                )
                if isinstance(result, LearnRunResult):
                    print(result.to_json() if args.json else format_learn_run(result))
                else:
                    print(result.to_json() if args.json else format_learn_plan(result))
                return result.exit_code
            parser.error(f"unknown demo learn command: {args.learn_command}")
            return 2
        parser.error(f"unknown demo command: {args.demo_command}")
        return 2

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
