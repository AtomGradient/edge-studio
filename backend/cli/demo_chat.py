# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Local base-chat demo command for the Edge developer CLI."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.api.chat_llm import _generate_streaming, _has_vision_config
from backend.api.chat_loaders import _get_or_load_mlx_model, _get_or_load_vlm_model
from backend.api.chat_params import get_generation_params
from backend.api.chat_vlm import _generate_streaming_vlm
from backend.cli.fingerprints import directory_manifest_hash, pretty_json, sha256_hex, sha256_prefixed, utc_now_iso
from backend.cli.models import ModelWhereReport, where_model
from backend.services.app_dirs import data_path
from backend.services.error_mapper import map_error
from backend.services.mlx_runtime_gate import mlx_runtime_gate
from backend.services.mlx_worker import submit_mlx_task
from backend.services.model_manager import manager
from backend.services.neural_imprint_runtime import NeuralImprintRuntimeError, restore_neural_imprint_for_model


CHAT_RECEIPT_SCHEMA_VERSION = "edge.demo.chat.receipt.v1"
CHAT_RUN_SCHEMA_VERSION = "edge.demo.chat.run.v1"
LEARN_RECEIPT_SCHEMA_VERSION = "edge.demo.learn.receipt.v1"
GENERATION_RECEIPT_SCHEMA_VERSION = "edgestudio.neural_imprint_generation_receipt.v2"
NEURAL_IMPRINT_METADATA_NAME = "neural_imprint_metadata.json"
DEFAULT_MODEL_REF = "qwen3.5-9b-4bit"
DEFAULT_MAX_TOKENS = 2048
MAX_CONFIGURED_TOKENS = 4096
INCOMPLETE_GENERATION_MESSAGE = "Generation finished without a complete event."
INCOMPLETE_GENERATION_RETRY_MESSAGE = (
    "Model warm-up did not finish. Try the same message again; if it repeats, restart `edge demo chat`."
)


@dataclass(frozen=True)
class ChatRunOptions:
    model_ref: str = "auto"
    prompt: str = ""
    max_tokens: int | None = None
    include_text: bool = False
    interactive: bool = False
    with_imprint: Path | None = None


@dataclass(frozen=True)
class ChatRunResult:
    ok: bool
    exit_code: int
    report: dict[str, Any]
    answer_text: str | None = None

    def to_json(self) -> str:
        return json.dumps(self.report, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ChatInteractiveResult:
    ok: bool
    exit_code: int
    session_id: str
    turn_count: int
    receipt_paths: list[str]


@dataclass(frozen=True)
class ChatImprintReference:
    source_path: Path
    artifact_path: Path
    sidecar_path: Path
    artifact_id: str | None = None
    schema_version: str | None = None


@dataclass(frozen=True)
class ChatImprintState:
    model_id: str
    artifact_path: Path
    sidecar_path: Path
    artifact_id: str | None
    prefix_token_count: int | None


class ChatImprintError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def run_demo_chat(
    *,
    options: ChatRunOptions,
    env: Mapping[str, str] | None = None,
) -> ChatRunResult:
    model_ref = DEFAULT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    if local_match is None:
        remediation = where.fetch_command or f"edge models fetch {model_ref}"
        return _run_error(
            "missing_model",
            "A local compatible model is required before running the chat demo.",
            options,
            remediation=remediation,
        )

    model_path = Path(local_match.path)
    max_tokens = _resolve_max_tokens(options, model_path)
    run_id = _run_id(options)
    started = time.time()
    use_imprint = options.with_imprint is not None
    _progress("load", f"loading model={model_path.name} (first load can take 30-90s)")

    try:
        _run_mlx_sync(_load_chat_model, model_path, use_imprint)
    except Exception as exc:
        return _run_error("model_load_failed", f"Failed to load model: {exc}", options)

    imprint_state: ChatImprintState | None = None
    if options.with_imprint is not None:
        try:
            imprint_state = _restore_chat_imprint(model_path=model_path, imprint_path=options.with_imprint)
        except ChatImprintError as exc:
            return _run_error(exc.code, exc.message, options)
    _progress("ready", "model loaded")

    try:
        _progress("generate", f"max_tokens={max_tokens}")
        answer = _generate_streamed_answer_with_retry(
            model_id=imprint_state.model_id if imprint_state is not None else model_ref,
            model_path=model_path,
            prompt=options.prompt,
            history=[],
            max_tokens=max_tokens,
            use_neural_imprint=imprint_state is not None,
        )
    except Exception as exc:
        return _run_error("generation_failed", f"Failed to generate chat answer: {exc}", options)

    try:
        model_manifest = directory_manifest_hash(model_path)
    except Exception as exc:
        return _run_error("model_fingerprint_failed", f"Failed to fingerprint model directory: {exc}", options)

    receipt = _chat_receipt(
        run_id=run_id,
        model_ref=model_ref,
        model_path=model_path,
        model_manifest=model_manifest,
        prompt=options.prompt,
        answer_text=answer["text"],
        answer_tokens=int(answer["token_count"]),
        max_tokens=max_tokens,
        include_text=options.include_text,
        imprint_state=imprint_state,
    )

    try:
        receipt_path = write_chat_receipt(receipt, run_id=run_id)
    except Exception as exc:
        return _run_error("receipt_write_failed", f"Failed to write chat receipt: {exc}", options)

    report: dict[str, Any] = {
        "schema_version": CHAT_RUN_SCHEMA_VERSION,
        "ok": True,
        "status": "completed",
        "run_id": run_id,
        "model": {
            "model_ref": model_ref,
            "path": str(model_path),
            "sha256": model_manifest["sha256"],
            "sha256_scope": model_manifest.get("sha256_scope", "directory_manifest_v1"),
        },
        "prompt_sha256": receipt["prompt_sha256"],
        "answer_sha256": receipt["answer_sha256"],
        "answer_tokens": answer["token_count"],
        "receipt_path": str(receipt_path),
        "raw_text_included": options.include_text,
        "network_used_during_demo": False,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    if imprint_state is not None:
        report["neural_imprint"] = _imprint_report(imprint_state)
    if options.include_text:
        report["prompt"] = options.prompt
        report["answer"] = answer["text"]

    _progress("done", f"receipt={receipt_path}")
    return ChatRunResult(ok=True, exit_code=0, report=report, answer_text=answer["text"])


def run_demo_chat_interactive(
    *,
    options: ChatRunOptions,
    env: Mapping[str, str] | None = None,
    input_stream: io.TextIOBase | None = None,
    output_stream: io.TextIOBase | None = None,
) -> ChatInteractiveResult:
    model_ref = DEFAULT_MODEL_REF if options.model_ref == "auto" else options.model_ref
    where = where_model(model_ref, env=env)
    local_match = _first_complete_match(where)
    if local_match is None:
        remediation = where.fetch_command or f"edge models fetch {model_ref}"
        print(
            "A local compatible model is required before running interactive chat.",
            file=output_stream or sys.stdout,
        )
        print(f"Run: {remediation}", file=output_stream or sys.stdout)
        return ChatInteractiveResult(False, 1, _interactive_session_id(model_ref), 0, [])

    model_path = Path(local_match.path)
    max_tokens = _resolve_max_tokens(options, model_path)
    stdin = input_stream or sys.stdin
    stdout = output_stream or sys.stdout
    session_id = _interactive_session_id(model_ref)
    receipt_paths: list[str] = []

    use_imprint = options.with_imprint is not None
    _progress("load", f"loading model={model_path.name} (first load can take 30-90s)")
    try:
        _run_mlx_sync(_load_chat_model, model_path, use_imprint)
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=stdout)
        return ChatInteractiveResult(False, 1, session_id, 0, receipt_paths)

    imprint_state: ChatImprintState | None = None
    if options.with_imprint is not None:
        try:
            imprint_state = _restore_chat_imprint(model_path=model_path, imprint_path=options.with_imprint)
        except ChatImprintError as exc:
            print(f"Failed to restore Neural Imprint: {exc.code}: {exc.message}", file=stdout)
            return ChatInteractiveResult(False, 1, session_id, 0, receipt_paths)
    _progress("ready", "type a message, /exit to quit")

    try:
        model_manifest = directory_manifest_hash(model_path)
    except Exception as exc:
        print(f"Failed to fingerprint model directory: {exc}", file=stdout)
        return ChatInteractiveResult(False, 1, session_id, 0, receipt_paths)

    history: list[dict[str, str]] = []
    turn_index = 0

    while True:
        print("you> ", end="", flush=True, file=stdout)
        line = stdin.readline()
        if line == "":
            print("", file=stdout)
            break
        prompt = line.strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break

        try:
            _progress("generate", f"turn={turn_index + 1} max_tokens={max_tokens}")
            print("assistant> ", end="", flush=True, file=stdout)
            answer = _generate_streamed_answer_with_retry(
                model_id=imprint_state.model_id if imprint_state is not None else model_ref,
                model_path=model_path,
                prompt=prompt,
                history=history,
                max_tokens=max_tokens,
                output_stream=stdout,
                use_neural_imprint=imprint_state is not None,
            )
            print("", file=stdout)
        except Exception as exc:
            print(f"\nassistant> generation failed: {exc}", file=stdout)
            return ChatInteractiveResult(False, 1, session_id, turn_index, receipt_paths)

        answer_text = str(answer.get("text") or "")
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_text},
            ]
        )
        run_id = f"{session_id}-turn-{turn_index + 1:03d}"
        receipt = _chat_receipt(
            run_id=run_id,
            model_ref=model_ref,
            model_path=model_path,
            model_manifest=model_manifest,
            prompt=prompt,
            answer_text=answer_text,
            answer_tokens=int(answer.get("token_count") or 0),
            max_tokens=max_tokens,
            include_text=options.include_text,
            session_id=session_id,
            turn_index=turn_index + 1,
            history_turn_count=len(history) // 2,
            imprint_state=imprint_state,
        )
        try:
            receipt_path = write_chat_receipt(receipt, run_id=run_id)
        except Exception as exc:
            print(f"assistant> receipt write failed: {exc}", file=stdout)
            return ChatInteractiveResult(False, 1, session_id, turn_index + 1, receipt_paths)
        receipt_paths.append(str(receipt_path))
        _progress("receipt", f"turn={turn_index + 1} path={receipt_path}")
        turn_index += 1

    _progress("done", f"session={session_id} turns={turn_index}")
    return ChatInteractiveResult(True, 0, session_id, turn_index, receipt_paths)


def default_chat_receipt_path(run_id: str) -> Path:
    return data_path("demo_runs", run_id, "chat_receipt.json")


def write_chat_receipt(receipt: Mapping[str, Any], *, run_id: str | None = None, path: Path | None = None) -> Path:
    output_path = path or default_chat_receipt_path(run_id or str(receipt.get("run_id") or ""))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pretty_json(dict(receipt)), encoding="utf-8")
    return output_path


def _chat_receipt(
    *,
    run_id: str,
    model_ref: str,
    model_path: Path,
    model_manifest: Mapping[str, Any],
    prompt: str,
    answer_text: str,
    answer_tokens: int,
    max_tokens: int,
    include_text: bool,
    session_id: str | None = None,
    turn_index: int | None = None,
    history_turn_count: int | None = None,
    imprint_state: ChatImprintState | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": CHAT_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "model_ref": model_ref,
        "model_path": str(model_path),
        "model_sha256": model_manifest["sha256"],
        "model_sha256_scope": model_manifest.get("sha256_scope", "directory_manifest_v1"),
        "prompt_sha256": sha256_prefixed(prompt.encode("utf-8")),
        "answer_sha256": sha256_prefixed(answer_text.encode("utf-8")),
        "answer_tokens": answer_tokens,
        "max_tokens": max_tokens,
        "raw_text_included": include_text,
        "network_used_during_demo": False,
        "status": "completed",
        "created_at": utc_now_iso(),
    }
    if session_id is not None:
        receipt["session_id"] = session_id
    if turn_index is not None:
        receipt["turn_index"] = turn_index
    if history_turn_count is not None:
        receipt["history_turn_count"] = history_turn_count
    if imprint_state is not None:
        receipt["neural_imprint"] = _imprint_report(imprint_state)
    if include_text:
        receipt["include_text_acknowledged"] = True
        receipt["prompt"] = prompt
        receipt["answer"] = answer_text
    return receipt


def format_demo_chat(result: ChatRunResult) -> str:
    report = result.report
    if not result.ok:
        error = report.get("error", {})
        lines = [
            f"Edge demo chat ({report.get('schema_version')})",
            f"status: {report.get('status')}",
        ]
        if isinstance(error, dict):
            lines.append(f"error: {error.get('code')}: {error.get('message')}")
            if error.get("remediation"):
                lines.append(f"remediation: {error['remediation']}")
        return "\n".join(lines)

    lines = [
        f"Edge demo chat ({report.get('schema_version')})",
        f"status: {report.get('status')}",
        f"model: {report.get('model', {}).get('model_ref')}",
        f"answer_sha256: {report.get('answer_sha256')}",
        f"answer_tokens: {report.get('answer_tokens')}",
        f"receipt: {report.get('receipt_path')}",
        "raw_text_in_receipt: false" if not report.get("raw_text_included") else "raw_text_in_receipt: true",
    ]
    if result.answer_text is not None:
        lines.extend(["", result.answer_text])
    return "\n".join(lines)


def _run_error(
    code: str,
    message: str,
    options: ChatRunOptions,
    *,
    remediation: str | None = None,
) -> ChatRunResult:
    error: dict[str, str] = {"code": code, "message": message}
    if remediation:
        error["remediation"] = remediation
    return ChatRunResult(
        ok=False,
        exit_code=1,
        report={
            "schema_version": CHAT_RUN_SCHEMA_VERSION,
            "ok": False,
            "status": code,
            "run_id": _run_id(options),
            "error": error,
            "raw_text_included": False,
            "network_used_during_demo": False,
        },
    )


def _first_complete_match(where: ModelWhereReport):
    for match in where.local_matches:
        if match.complete:
            return match
    return None


def _progress(tag: str, message: str) -> None:
    print(f"[chat:{tag}] {message}", file=sys.stderr)


def _interactive_session_id(model_ref: str) -> str:
    fingerprint = sha256_hex((model_ref + str(time.time())).encode("utf-8"))[:12]
    return f"edge-chat-session-{fingerprint}"


def _run_mlx_sync(fn, *args: Any) -> Any:
    return submit_mlx_task(fn, *args).result()


def _load_chat_model(model_path: Path, force_llm: bool = False) -> None:
    with mlx_runtime_gate("cli.demo_chat.load"):
        if _is_vlm_model(model_path) and not force_llm:
            _get_or_load_vlm_model(str(model_path))
        else:
            _get_or_load_mlx_model(str(model_path))


def _is_vlm_model(model_path: Path) -> bool:
    return _has_vision_config(str(model_path))


def _restore_chat_imprint(*, model_path: Path, imprint_path: Path) -> ChatImprintState:
    reference = _resolve_imprint_reference(imprint_path)
    _progress("imprint", f"restoring artifact={reference.artifact_path}")
    try:
        loaded = manager.load_model(str(model_path))
    except Exception as exc:
        raise ChatImprintError(
            "model_register_failed",
            f"Failed to register the base model before Neural Imprint restore: {exc}",
        ) from exc
    try:
        status = restore_neural_imprint_for_model(
            model_id=loaded.model_id,
            artifact_path=reference.artifact_path,
            sidecar_path=reference.sidecar_path,
            artifact_id=reference.artifact_id,
        )
    except NeuralImprintRuntimeError as exc:
        raise ChatImprintError(exc.code, exc.message) from exc
    except Exception as exc:
        raise ChatImprintError(
            "neural_imprint_restore_failed",
            f"Failed to restore Neural Imprint artifact: {exc}",
        ) from exc
    if not status.active or not status.model_id:
        raise ChatImprintError(
            "neural_imprint_not_active",
            "Neural Imprint restore completed without an active runtime state.",
        )
    return ChatImprintState(
        model_id=status.model_id,
        artifact_path=reference.artifact_path,
        sidecar_path=reference.sidecar_path,
        artifact_id=status.artifact_id or reference.artifact_id,
        prefix_token_count=status.prefix_token_count,
    )


def _resolve_imprint_reference(path: Path) -> ChatImprintReference:
    source = path.expanduser().resolve()
    if not source.exists():
        raise ChatImprintError(
            "imprint_path_not_found",
            f"Neural Imprint path not found: {source}",
        )
    if source.is_dir():
        raise ChatImprintError(
            "imprint_path_is_directory",
            f"Neural Imprint path must be a receipt or artifact file: {source}",
        )

    if source.suffix.lower() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChatImprintError(
                "invalid_imprint_receipt",
                f"Neural Imprint receipt is not valid JSON: {source}",
            ) from exc
        if not isinstance(payload, dict):
            raise ChatImprintError(
                "invalid_imprint_receipt",
                "Neural Imprint receipt must be a JSON object.",
            )
        return _resolve_imprint_receipt(source, payload)

    artifact = source
    sidecar = _default_imprint_sidecar_path(artifact)
    _require_existing_file(artifact, "artifact_path")
    _require_existing_file(sidecar, "metadata_path")
    return ChatImprintReference(
        source_path=source,
        artifact_path=artifact,
        sidecar_path=sidecar,
        artifact_id=artifact.stem,
    )


def _resolve_imprint_receipt(source: Path, payload: Mapping[str, Any]) -> ChatImprintReference:
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version == LEARN_RECEIPT_SCHEMA_VERSION:
        artifact = _receipt_path(payload, "artifact_path", receipt_path=source)
        sidecar = _receipt_path(payload, "metadata_path", receipt_path=source, required=False)
        if sidecar is None:
            sidecar = _default_imprint_sidecar_path(artifact)
        artifact_id = str(payload.get("artifact_id") or "").strip() or None
    elif schema_version == GENERATION_RECEIPT_SCHEMA_VERSION:
        artifact = _receipt_path(payload, "artifact_path", receipt_path=source)
        sidecar = _receipt_path(payload, "metadata_path", receipt_path=source)
        artifact_id = str(payload.get("job_id") or "").strip() or artifact.stem
    else:
        raise ChatImprintError(
            "unsupported_imprint_receipt_schema",
            (
                "Neural Imprint receipt schema must be "
                f"{LEARN_RECEIPT_SCHEMA_VERSION} or {GENERATION_RECEIPT_SCHEMA_VERSION}; "
                f"got {schema_version or '<missing>'}."
            ),
        )
    _require_existing_file(artifact, "artifact_path")
    _require_existing_file(sidecar, "metadata_path")
    return ChatImprintReference(
        source_path=source,
        artifact_path=artifact,
        sidecar_path=sidecar,
        artifact_id=artifact_id,
        schema_version=schema_version,
    )


def _receipt_path(
    payload: Mapping[str, Any],
    key: str,
    *,
    receipt_path: Path,
    required: bool = True,
) -> Path | None:
    raw = str(payload.get(key) or "").strip()
    if not raw:
        if required:
            raise ChatImprintError(
                "invalid_imprint_receipt",
                f"Neural Imprint receipt is missing {key}: {receipt_path}",
            )
        return None
    return Path(raw).expanduser().resolve()


def _require_existing_file(path: Path, field: str) -> None:
    if not path.exists():
        raise ChatImprintError(
            f"imprint_{field}_not_found",
            f"Neural Imprint {field} not found: {path}",
        )
    if not path.is_file():
        raise ChatImprintError(
            f"imprint_{field}_not_file",
            f"Neural Imprint {field} must be a file: {path}",
        )


def _default_imprint_sidecar_path(artifact: Path) -> Path:
    return artifact.with_name(NEURAL_IMPRINT_METADATA_NAME)


def _imprint_report(state: ChatImprintState) -> dict[str, Any]:
    return {
        "active": True,
        "model_id": state.model_id,
        "artifact_id": state.artifact_id,
        "artifact_path": str(state.artifact_path),
        "metadata_path": str(state.sidecar_path),
        "prefix_token_count": state.prefix_token_count,
    }


def _generate_streamed_answer(
    *,
    model_id: str,
    model_path: Path,
    prompt: str,
    history: list[dict[str, str]],
    max_tokens: int,
    output_stream: io.TextIOBase | None = None,
    use_neural_imprint: bool = False,
) -> dict[str, Any]:
    return asyncio.run(
        _generate_streamed_answer_async(
            model_id=model_id,
            model_path=model_path,
            prompt=prompt,
            history=history,
            max_tokens=max_tokens,
            output_stream=output_stream,
            use_neural_imprint=use_neural_imprint,
        )
    )


def _generate_streamed_answer_with_retry(
    *,
    model_id: str,
    model_path: Path,
    prompt: str,
    history: list[dict[str, str]],
    max_tokens: int,
    output_stream: io.TextIOBase | None = None,
    use_neural_imprint: bool = False,
) -> dict[str, Any]:
    try:
        return _generate_streamed_answer(
            model_id=model_id,
            model_path=model_path,
            prompt=prompt,
            history=history,
            max_tokens=max_tokens,
            output_stream=output_stream,
            use_neural_imprint=use_neural_imprint,
        )
    except Exception as exc:
        if not _is_incomplete_generation_error(exc):
            raise
        _progress("retry", "generation ended before completion; retrying once")
        try:
            return _generate_streamed_answer(
                model_id=model_id,
                model_path=model_path,
                prompt=prompt,
                history=history,
                max_tokens=max_tokens,
                output_stream=output_stream,
                use_neural_imprint=use_neural_imprint,
            )
        except Exception as retry_exc:
            if _is_incomplete_generation_error(retry_exc):
                raise RuntimeError(INCOMPLETE_GENERATION_RETRY_MESSAGE) from retry_exc
            raise


def _is_incomplete_generation_error(exc: Exception) -> bool:
    return INCOMPLETE_GENERATION_MESSAGE in str(exc)


async def _generate_streamed_answer_async(
    *,
    model_id: str,
    model_path: Path,
    prompt: str,
    history: list[dict[str, str]],
    max_tokens: int,
    output_stream: io.TextIOBase | None,
    use_neural_imprint: bool,
) -> dict[str, Any]:
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cancel_event = threading.Event()
    model_params = get_generation_params(str(model_path))
    history_snapshot = [dict(turn) for turn in history]

    if _is_vlm_model(model_path) and not use_neural_imprint:
        gen_fn = _generate_streaming_vlm
        gen_args = (
            str(model_path),
            prompt,
            None,
            history_snapshot,
            max_tokens,
            model_params.temperature,
            event_queue,
            loop,
            cancel_event,
        )
        gen_kwargs = {"enable_thinking": False}
    else:
        gen_fn = _generate_streaming
        gen_args = (
            model_id,
            str(model_path),
            prompt,
            history_snapshot,
            max_tokens,
            model_params.temperature,
            model_params.top_k,
            model_params.top_p,
            False,
            event_queue,
            loop,
            cancel_event,
        )
        gen_kwargs = {"use_neural_imprint": use_neural_imprint}

    future = submit_mlx_task(
        _run_cli_generation_with_gate,
        gen_fn,
        gen_args,
        gen_kwargs,
        event_queue,
        loop,
    )
    chunks: list[str] = []

    try:
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if future.done():
                    future.result()
                    break
                continue

            event_type = event.get("type")
            if event_type == "token":
                token_text = str(event.get("token") or "")
                if token_text:
                    chunks.append(token_text)
                    if output_stream is not None:
                        print(token_text, end="", flush=True, file=output_stream)
            elif event_type == "complete":
                answer_text = str(event.get("full_text") or "".join(chunks)).strip()
                future.result(timeout=5.0)
                return {
                    "text": answer_text,
                    "token_count": int(event.get("total_tokens") or len(chunks)),
                    "elapsed_seconds": float(event.get("total_time") or 0.0),
                    "tokens_per_sec": event.get("tokens_per_sec"),
                }
            elif event_type == "error":
                raise RuntimeError(str(event.get("message") or "generation failed"))
            elif event_type == "cancelled":
                raise RuntimeError("Generation was cancelled.")

        raise RuntimeError(INCOMPLETE_GENERATION_MESSAGE)
    finally:
        if not future.done():
            cancel_event.set()
            try:
                future.result(timeout=5.0)
            except concurrent.futures.TimeoutError:
                pass


def _run_cli_generation_with_gate(
    gen_fn,
    gen_args: tuple[Any, ...],
    gen_kwargs: dict[str, Any],
    event_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
) -> None:
    with mlx_runtime_gate("cli.demo_chat.generate"):
        try:
            gen_fn(*gen_args, **gen_kwargs)
        except Exception as exc:
            user_msg, _ = map_error(exc)
            asyncio.run_coroutine_threadsafe(
                event_queue.put({"type": "error", "message": user_msg}),
                loop,
            )


def _resolve_max_tokens(options: ChatRunOptions, model_path: Path) -> int:
    if options.max_tokens is not None:
        return max(1, int(options.max_tokens))
    configured = _configured_max_tokens(model_path)
    if configured is not None:
        return configured
    if not (model_path / "config.json").exists():
        return DEFAULT_MAX_TOKENS
    try:
        return min(get_generation_params(str(model_path)).max_tokens, MAX_CONFIGURED_TOKENS)
    except Exception:
        return DEFAULT_MAX_TOKENS


def _configured_max_tokens(model_path: Path) -> int | None:
    for name in ("generation_config.json", "config.json"):
        path = model_path / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = _configured_max_tokens_from_mapping(data)
        if value is not None:
            return value
    return None


def _configured_max_tokens_from_mapping(data: Mapping[str, Any]) -> int | None:
    for key in ("max_new_tokens", "max_output_tokens", "max_generation_tokens"):
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return min(value, MAX_CONFIGURED_TOKENS)
    text_config = data.get("text_config")
    if isinstance(text_config, Mapping):
        return _configured_max_tokens_from_mapping(text_config)
    return None


def _run_id(options: ChatRunOptions) -> str:
    fingerprint = sha256_hex(
        (options.model_ref + options.prompt + str(time.time()))[:512].encode("utf-8")
    )[:12]
    return f"edge-chat-{fingerprint}"
