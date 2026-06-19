# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Fail-closed Halo capsule automation scheduler tick shell.

This module does not start background work. It exposes a pure, testable tick
function that a later lifecycle wiring step may call explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


HALO_CAPSULE_AUTOMATION_SCHEDULER_TICK_SCHEMA_VERSION = (
    "edgestudio.halo_capsule_automation_scheduler_tick.v1"
)
MIN_INTERVAL_SECONDS = 600
MAX_PUSHES = 20
MAX_CHUNK_SIZE = 8 * 1024 * 1024


class NonBlockingLock(Protocol):
    def acquire(self, blocking: bool = True) -> bool: ...

    def release(self) -> None: ...


SchedulerRunner = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class HaloCapsuleAutomationSchedulerConfig:
    enabled: bool = False
    dry_run: bool = True
    interval_seconds: int = MIN_INTERVAL_SECONDS
    max_pushes: int = 1
    chunk_size: int = 1024 * 1024
    peer_ids: tuple[str, ...] = field(default_factory=tuple)
    allow_side_effects: bool = False


@dataclass
class HaloCapsuleAutomationSchedulerError(ValueError):
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


def run_scheduler_tick(
    config: HaloCapsuleAutomationSchedulerConfig,
    *,
    runner: SchedulerRunner,
    clock: Callable[[], float],
    lock: NonBlockingLock,
    last_run_state: dict[str, Any],
) -> dict[str, Any]:
    """Run one explicitly invoked scheduler tick.

    The tick is fail-closed by default and never starts its own loop, thread, or
    retry path. Runner dependencies remain injected by the caller.
    """

    now = float(clock())
    try:
        peer_ids = validate_scheduler_config(config)
    except HaloCapsuleAutomationSchedulerError as exc:
        return _tick_record(
            config,
            now=now,
            status="failed",
            reason=exc.code,
            error=exc.to_error(),
        )

    if not config.enabled:
        return _tick_record(config, now=now, status="skipped", reason="disabled")

    acquired = lock.acquire(blocking=False)
    if not acquired:
        return _tick_record(config, now=now, status="skipped", reason="lock_busy")

    try:
        last_started_at = _optional_float(last_run_state.get("last_started_at"))
        if (
            last_started_at is not None
            and now - last_started_at < config.interval_seconds
        ):
            return _tick_record(
                config,
                now=now,
                status="skipped",
                reason="interval_not_elapsed",
            )

        if not config.dry_run and not peer_ids:
            return _tick_record(
                config,
                now=now,
                status="failed",
                reason="missing_peer_ids",
                error={
                    "code": "missing_peer_ids",
                    "message": "peer_ids is required when dry_run=false",
                    "retryable": False,
                    "details": {},
                },
            )
        if not config.dry_run and not config.allow_side_effects:
            return _tick_record(
                config,
                now=now,
                status="failed",
                reason="side_effects_not_allowed",
                error={
                    "code": "side_effects_not_allowed",
                    "message": "allow_side_effects=true is required when dry_run=false",
                    "retryable": False,
                    "details": {},
                },
            )

        request_payload = {
            "source": "scheduler",
            "dry_run": config.dry_run,
            "peer_ids": peer_ids,
            "max_pushes": config.max_pushes,
            "chunk_size": config.chunk_size,
        }
        last_run_state["last_started_at"] = now
        try:
            response = runner(
                dry_run=config.dry_run,
                peer_ids=peer_ids,
                max_pushes=config.max_pushes,
                chunk_size=config.chunk_size,
                source="scheduler",
                request_payload=request_payload,
            )
        except Exception as exc:
            last_run_state["last_failed_at"] = now
            return _tick_record(
                config,
                now=now,
                status="failed",
                reason="runner_failed",
                runner_called=True,
                error={
                    "code": "runner_failed",
                    "message": str(exc),
                    "retryable": False,
                    "details": {},
                },
            )

        last_run_state["last_completed_at"] = now
        receipt = response.get("receipt") if isinstance(response, dict) else None
        receipt_run_id = (
            str(receipt.get("run_id"))
            if isinstance(receipt, dict) and receipt.get("run_id")
            else None
        )
        return _tick_record(
            config,
            now=now,
            status="completed",
            reason="runner_completed",
            runner_called=True,
            receipt_run_id=receipt_run_id,
        )
    finally:
        lock.release()


def validate_scheduler_config(
    config: HaloCapsuleAutomationSchedulerConfig,
) -> list[str]:
    peer_ids = _clean_peer_ids(config.peer_ids)
    if config.interval_seconds < MIN_INTERVAL_SECONDS:
        raise HaloCapsuleAutomationSchedulerError(
            "invalid_interval_seconds",
            f"interval_seconds must be >= {MIN_INTERVAL_SECONDS}",
            {"interval_seconds": config.interval_seconds},
        )
    if config.max_pushes < 1 or config.max_pushes > MAX_PUSHES:
        raise HaloCapsuleAutomationSchedulerError(
            "invalid_max_pushes",
            f"max_pushes must be between 1 and {MAX_PUSHES}",
            {"max_pushes": config.max_pushes},
        )
    if config.chunk_size < 1 or config.chunk_size > MAX_CHUNK_SIZE:
        raise HaloCapsuleAutomationSchedulerError(
            "invalid_chunk_size",
            f"chunk_size must be between 1 and {MAX_CHUNK_SIZE}",
            {"chunk_size": config.chunk_size},
        )
    return peer_ids


def _tick_record(
    config: HaloCapsuleAutomationSchedulerConfig,
    *,
    now: float,
    status: str,
    reason: str,
    runner_called: bool = False,
    receipt_run_id: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": HALO_CAPSULE_AUTOMATION_SCHEDULER_TICK_SCHEMA_VERSION,
        "status": status,
        "reason": reason,
        "source": "scheduler",
        "enabled": bool(config.enabled),
        "dry_run": bool(config.dry_run),
        "runner_called": bool(runner_called),
        "receipt_run_id": receipt_run_id,
        "peer_ids": _clean_peer_ids(config.peer_ids),
        "max_pushes": config.max_pushes,
        "interval_seconds": config.interval_seconds,
        "timestamp": now,
    }
    if error is not None:
        record["error"] = error
    return record


def _clean_peer_ids(values: tuple[str, ...]) -> list[str]:
    return [peer_id.strip() for peer_id in values if peer_id.strip()]


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
