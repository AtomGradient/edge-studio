# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Local base-chat demo command for the Edge developer CLI."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.api.chat_llm import _apply_chat_template, _apply_neural_imprint_turn_template
from backend.api.chat_loaders import _get_or_load_mlx_model
from backend.cli.fingerprints import directory_manifest_hash, pretty_json
from backend.cli.models import ModelWhereReport, where_model
from backend.services.app_dirs import data_path


CHAT_RECEIPT_SCHEMA_VERSION = "edge.demo.chat.receipt.v1"
CHAT_RUN_SCHEMA_VERSION = "edge.demo.chat.run.v1"
DEFAULT_MODEL_REF = "qwen3.5-9b-4bit"
DEFAULT_MAX_TOKENS = 512
MAX_CONFIGURED_TOKENS = 4096


@dataclass(frozen=True)
class ChatRunOptions:
    model_ref: str = "auto"
    prompt: str = ""
    max_tokens: int | None = None
    include_text: bool = False
    interactive: bool = False


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
    _progress("load", f"loading model={model_path.name} (first load can take 30-90s)")

    try:
        model, tokenizer = _get_or_load_mlx_model(str(model_path))
    except Exception as exc:
        return _run_error("model_load_failed", f"Failed to load model: {exc}", options)
    _progress("ready", "model loaded")

    try:
        prompt_text = _apply_chat_template(tokenizer, options.prompt, [], False)
        prompt_ids = _encode(tokenizer, prompt_text)
        _progress("generate", f"max_tokens={max_tokens}")
        answer = _generate_answer(
            model=model,
            tokenizer=tokenizer,
            input_ids=prompt_ids,
            max_tokens=max_tokens,
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

    _progress("load", f"loading model={model_path.name} (first load can take 30-90s)")
    try:
        model, tokenizer = _get_or_load_mlx_model(str(model_path))
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=stdout)
        return ChatInteractiveResult(False, 1, session_id, 0, receipt_paths)
    _progress("ready", "type a message, /exit to quit")

    try:
        model_manifest = directory_manifest_hash(model_path)
    except Exception as exc:
        print(f"Failed to fingerprint model directory: {exc}", file=stdout)
        return ChatInteractiveResult(False, 1, session_id, 0, receipt_paths)

    cache = _make_prompt_cache(model)
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
            if turn_index == 0:
                prompt_text = _apply_chat_template(tokenizer, prompt, [], False)
            else:
                # The session cache already contains prior turns, so history stays empty here.
                # Re-tokenizing history would duplicate context in the same KV cache.
                prompt_text = _apply_neural_imprint_turn_template(tokenizer, prompt, [], False)
            prompt_ids = _encode(tokenizer, prompt_text)
            _progress("generate", f"turn={turn_index + 1} max_tokens={max_tokens}")
            answer = _generate_answer(
                model=model,
                tokenizer=tokenizer,
                input_ids=prompt_ids,
                max_tokens=max_tokens,
                cache=cache,
            )
        except Exception as exc:
            print(f"assistant> generation failed: {exc}", file=stdout)
            return ChatInteractiveResult(False, 1, session_id, turn_index, receipt_paths)

        answer_text = str(answer.get("text") or "")
        history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer_text},
            ]
        )
        print(f"assistant> {answer_text}", file=stdout)
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
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": CHAT_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "model_ref": model_ref,
        "model_path": str(model_path),
        "model_sha256": model_manifest["sha256"],
        "model_sha256_scope": model_manifest.get("sha256_scope", "directory_manifest_v1"),
        "prompt_sha256": _sha256_prefixed(prompt.encode("utf-8")),
        "answer_sha256": _sha256_prefixed(answer_text.encode("utf-8")),
        "answer_tokens": answer_tokens,
        "max_tokens": max_tokens,
        "raw_text_included": include_text,
        "network_used_during_demo": False,
        "status": "completed",
        "created_at": _utc_now_iso(),
    }
    if session_id is not None:
        receipt["session_id"] = session_id
    if turn_index is not None:
        receipt["turn_index"] = turn_index
    if history_turn_count is not None:
        receipt["history_turn_count"] = history_turn_count
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
    fingerprint = _sha256_hex((model_ref + str(time.time())).encode("utf-8"))[:12]
    return f"edge-chat-session-{fingerprint}"


def _resolve_max_tokens(options: ChatRunOptions, model_path: Path) -> int:
    if options.max_tokens is not None:
        return max(1, int(options.max_tokens))
    return _configured_max_tokens(model_path) or DEFAULT_MAX_TOKENS


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
    fingerprint = _sha256_hex(
        (options.model_ref + options.prompt + str(time.time()))[:512].encode("utf-8")
    )[:12]
    return f"edge-chat-{fingerprint}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_prefixed(data: bytes) -> str:
    return f"sha256:{_sha256_hex(data)}"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    else:
        tokens = tokenizer._tokenizer.encode(text)
    return list(tokens) if not isinstance(tokens, list) else tokens


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        if hasattr(tokenizer, "decode"):
            value = tokenizer.decode(list(token_ids))
        else:
            value = tokenizer._tokenizer.decode(list(token_ids))
        return value if isinstance(value, str) else str(value)
    except Exception:
        return "".join(str(t) for t in token_ids)


def _eos_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, int) and value >= 0:
        ids.add(int(value))
    for token in ("<|im_end|>", "<|endoftext|>"):
        tok = tokenizer._tokenizer if hasattr(tokenizer, "_tokenizer") else tokenizer
        try:
            if hasattr(tok, "token_to_id"):
                token_id = tok.token_to_id(token)
            elif hasattr(tok, "convert_tokens_to_ids"):
                token_id = tok.convert_tokens_to_ids(token)
            else:
                continue
            if isinstance(token_id, int) and token_id >= 0:
                ids.add(token_id)
        except Exception:
            continue
    return ids


def _make_prompt_cache(model: Any) -> Any:
    from backend.core.dsr_cache import make_prompt_cache

    return make_prompt_cache(model)


def _forward_last_logits(model: Any, token_ids: Sequence[int], cache: Any = None) -> Any:
    import mlx.core as mx

    arr = mx.array(list(token_ids), dtype=mx.int32)[None, :]
    out = model(arr, cache=cache) if cache is not None else model(arr)
    logits = out[0] if isinstance(out, tuple) else out
    last = logits[:, -1, :].astype(mx.float32)
    mx.eval(last)
    return last[0]


def _generate_answer(
    *,
    model: Any,
    tokenizer: Any,
    input_ids: Sequence[int],
    max_tokens: int,
    cache: Any = None,
) -> dict[str, Any]:
    import mlx.core as mx

    started = time.time()
    prompt_cache = cache if cache is not None else _make_prompt_cache(model)
    logits = _forward_last_logits(model, input_ids, cache=prompt_cache)
    generated: list[int] = []
    stops = _eos_token_ids(tokenizer)

    for _ in range(max_tokens):
        token = int(mx.argmax(logits, axis=-1).item())
        if token in stops:
            _forward_last_logits(model, [token], cache=prompt_cache)
            break
        generated.append(token)
        logits = _forward_last_logits(model, [token], cache=prompt_cache)

    text = _decode(tokenizer, generated).strip()
    return {
        "text": text,
        "token_count": len(generated),
        "elapsed_seconds": round(time.time() - started, 2),
    }
