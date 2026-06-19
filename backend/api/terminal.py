# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Web Terminal — PTY + WebSocket for real shell experience."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.resources.paths import script_path

router = APIRouter(prefix="/api/terminal", tags=["terminal"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TerminalSession:
    """A PTY session with master/slave fds."""

    def __init__(self, session_id: str, cols: int = 80, rows: int = 24):
        self.session_id = session_id
        self.cols = cols
        self.rows = rows
        self.master_fd: int | None = None
        self.slave_fd: int | None = None
        self.pid: int | None = None
        self.closed = False
        self._lock = threading.Lock()
        self._exit_code: int | None = None  # cached exit code

    def start(self, cmd: list[str] | None = None, cwd: str | None = None, env: dict | None = None):
        """Fork a PTY with the given command."""
        if cmd is None:
            cmd = [os.environ.get("SHELL", "/bin/bash")]

        # Create PTY
        self.master_fd, self.slave_fd = pty.openpty()

        # Set window size
        winsize = struct.pack("HHHH", self.rows, self.cols, 0, 0)
        fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, winsize)

        # Build environment
        child_env = os.environ.copy()
        child_env["TERM"] = "xterm-256color"
        child_env["COLORTERM"] = "truecolor"
        child_env["EDGESTUDIO_PYTHON"] = sys.executable
        python_bin_dir = str(Path(sys.executable).parent)
        child_env["PATH"] = f"{python_bin_dir}{os.pathsep}{child_env.get('PATH', '')}"
        # Strip proxy vars that break aria2c
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "all_proxy", "ALL_PROXY"):
            child_env.pop(k, None)
        if env:
            child_env.update(env)

        # Fork
        pid = os.fork()
        if pid == 0:
            # Child process
            os.close(self.master_fd)
            os.setsid()
            os.dup2(self.slave_fd, 0)
            os.dup2(self.slave_fd, 1)
            os.dup2(self.slave_fd, 2)
            if self.slave_fd > 2:
                os.close(self.slave_fd)
            if cwd:
                try:
                    os.chdir(cwd)
                except OSError:
                    pass
            os.execvpe(cmd[0], cmd, child_env)
        else:
            # Parent
            os.close(self.slave_fd)
            self.slave_fd = None
            self.pid = pid
            # Set master to non-blocking
            flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
            fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def write(self, data: bytes):
        """Write data to PTY (user input)."""
        if self.master_fd is not None and not self.closed:
            try:
                os.write(self.master_fd, data)
            except OSError:
                pass

    def read(self, timeout: float = 0.1) -> bytes | None:
        """Read available data from PTY."""
        if self.master_fd is None or self.closed:
            return None
        try:
            r, _, _ = select.select([self.master_fd], [], [], timeout)
            if r:
                return os.read(self.master_fd, 65536)
        except (OSError, ValueError):
            pass
        return None

    def resize(self, cols: int, rows: int):
        """Resize PTY window."""
        self.cols = cols
        self.rows = rows
        if self.master_fd is not None:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            try:
                fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
            except OSError:
                pass

    def is_alive(self) -> bool:
        """Check if child process is still running.

        Caches exit code on first detection so waitpid is not consumed twice.
        """
        if self._exit_code is not None:
            return False
        if self.pid is None:
            return False
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid != 0:
                # Process exited — cache the exit code
                if os.WIFEXITED(status):
                    self._exit_code = os.WEXITSTATUS(status)
                elif os.WIFSIGNALED(status):
                    self._exit_code = 128 + os.WTERMSIG(status)
                else:
                    self._exit_code = -1
                return False
            return True
        except ChildProcessError:
            self._exit_code = -1
            return False

    @property
    def exit_code(self) -> int:
        """Return cached exit code, or -1 if unknown."""
        return self._exit_code if self._exit_code is not None else -1

    def close(self):
        """Close PTY and kill child process."""
        with self._lock:
            if self.closed:
                return
            self.closed = True

        if self.pid is not None and self._exit_code is None:
            # Process might still be running — terminate it
            try:
                os.killpg(self.pid, signal.SIGTERM)
                for _ in range(10):
                    try:
                        pid, _ = os.waitpid(self.pid, os.WNOHANG)
                        if pid != 0:
                            break
                    except ChildProcessError:
                        break
                    time.sleep(0.1)
                else:
                    os.killpg(self.pid, signal.SIGKILL)
                    try:
                        os.waitpid(self.pid, 0)
                    except ChildProcessError:
                        pass
            except (OSError, ChildProcessError):
                pass

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None


# Global session store
_sessions: dict[str, TerminalSession] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> TerminalSession | None:
    with _sessions_lock:
        return _sessions.get(session_id)


def _create_session(cols: int = 80, rows: int = 24) -> str:
    session_id = str(uuid.uuid4())[:8]
    session = TerminalSession(session_id, cols, rows)
    with _sessions_lock:
        _sessions[session_id] = session
    return session_id


def _remove_session(session_id: str):
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session:
        session.close()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class CreateTerminalRequest(BaseModel):
    cols: int = 80
    rows: int = 24
    cmd: list[str] | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None


class CreateTerminalResponse(BaseModel):
    session_id: str


@router.post("/create", response_model=CreateTerminalResponse)
def create_terminal(req: CreateTerminalRequest):
    """Create a new terminal session."""
    session_id = _create_session(req.cols, req.rows)
    session = _get_session(session_id)
    if not session:
        raise HTTPException(500, "Failed to create session")

    try:
        session.start(cmd=req.cmd, cwd=req.cwd, env=req.env)
    except Exception as e:
        _remove_session(session_id)
        raise HTTPException(500, f"Failed to start PTY: {e}")

    logger.info("Created terminal session %s (cmd=%s)", session_id, req.cmd)
    return CreateTerminalResponse(session_id=session_id)


@router.delete("/{session_id}")
def close_terminal(session_id: str):
    """Close a terminal session."""
    session = _get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    _remove_session(session_id)
    return {"status": "closed"}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws/{session_id}")
async def terminal_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for terminal I/O.

    Protocol:
    - Client sends JSON: {"type": "input", "data": "..."} or {"type": "resize", "cols": N, "rows": N}
    - Server sends JSON: {"type": "output", "data": "..."} or {"type": "exit", "code": N}
    """
    session = _get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    logger.info("Terminal WebSocket connected: %s", session_id)

    # Reader task: PTY -> WebSocket
    async def read_pty():
        loop = asyncio.get_event_loop()
        try:
            while not session.closed:
                data = await loop.run_in_executor(None, session.read, 0.05)
                if data:
                    await websocket.send_json({"type": "output", "data": data.decode("utf-8", errors="replace")})
                elif not session.is_alive():
                    # Process exited — drain any remaining output first
                    while True:
                        leftover = await loop.run_in_executor(None, session.read, 0.01)
                        if not leftover:
                            break
                        await websocket.send_json({"type": "output", "data": leftover.decode("utf-8", errors="replace")})
                    await websocket.send_json({"type": "exit", "code": session.exit_code})
                    break
                else:
                    await asyncio.sleep(0.01)
        except Exception as e:
            logger.debug("PTY read error: %s", e)

    reader_task = asyncio.create_task(read_pty())

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            msg_type = msg.get("type")
            if msg_type == "input":
                data = msg.get("data", "")
                if isinstance(data, str):
                    session.write(data.encode("utf-8"))
            elif msg_type == "resize":
                cols = msg.get("cols", 80)
                rows = msg.get("rows", 24)
                session.resize(cols, rows)
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("Terminal WebSocket disconnected: %s", session_id)
    except Exception as e:
        logger.warning("Terminal WebSocket error: %s", e)
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass
        # Only clean up if the process has already exited.
        # If still running, keep session alive for potential reconnect.
        # Frontend is responsible for calling DELETE when done.
        if session and not session.is_alive():
            _remove_session(session_id)


# ---------------------------------------------------------------------------
# Convenience endpoint: run a command in a new terminal
# ---------------------------------------------------------------------------

# Default working directory for terminal commands (EdgeStudio project root)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
_INSTALL_ROOT = str(Path(sys.prefix))
_PACKAGED_SCRIPT_NAMES = {"hfd.sh", "msd.sh"}


def _default_terminal_cwd() -> str:
    """Return a cwd that works both from source checkout and installed wheel."""
    project_root = Path(_PROJECT_ROOT)
    if (project_root / "frontend").exists():
        return str(project_root)
    return _INSTALL_ROOT


def _resolve_packaged_arg(arg: str) -> str:
    """Resolve known EdgeStudio helper scripts when running from a wheel."""
    expanded = os.path.expanduser(arg)
    path = Path(expanded)
    if path.is_absolute() or len(path.parts) != 2:
        return expanded
    if path.parts[0] != "scripts" or path.parts[1] not in _PACKAGED_SCRIPT_NAMES:
        return expanded

    candidate = script_path(path.parts[1])
    if candidate.exists():
        return str(candidate)
    return expanded


class RunCommandRequest(BaseModel):
    cmd: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None
    cols: int = 120
    rows: int = 30


@router.post("/run", response_model=CreateTerminalResponse)
def run_command(req: RunCommandRequest):
    """Create a terminal session with a specific command.

    The command will run in a PTY, allowing full terminal emulation.
    If cwd is not specified, defaults to the EdgeStudio project root.
    Connect via WebSocket at /api/terminal/ws/{session_id} to interact.
    """
    session_id = _create_session(req.cols, req.rows)
    session = _get_session(session_id)
    if not session:
        raise HTTPException(500, "Failed to create session")

    try:
        # Default to project root if cwd not specified
        cwd = req.cwd or _default_terminal_cwd()
        # Expand ~ in cmd arguments (shell doesn't expand ~ in non-interactive exec)
        cmd = [_resolve_packaged_arg(arg) for arg in req.cmd]
        session.start(cmd=cmd, cwd=cwd, env=req.env)
    except Exception as e:
        _remove_session(session_id)
        raise HTTPException(500, f"Failed to start command: {e}")

    logger.info("Created terminal for command: %s (session=%s)", req.cmd, session_id)
    return CreateTerminalResponse(session_id=session_id)
