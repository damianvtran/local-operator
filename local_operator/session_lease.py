"""Atomic sole-writer leases for durable session transcripts.

The compatibility ``.session.pid`` marker is useful for old readers but cannot
be authoritative because replacing a text file is not acquisition.  This
module stays stdlib-only so resume discovery can consult it without importing
the engine or mobile stack.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LEASE_NAME = ".execution-lease"
MIRROR_NAME = ".session.pid"


class SessionLeaseHeldError(RuntimeError):
    """Raised when another live or unverifiable process owns a transcript."""

    def __init__(self, session_dir: Path, pid: int | None) -> None:
        owner = f"pid {pid}" if pid is not None else "an unverifiable process"
        super().__init__(
            f"session {session_dir.name} is already open in {owner}; "
            "attach to that session or wait for its owner to exit"
        )
        self.session_dir = session_dir
        self.pid = pid


def _pid_state(pid: int) -> Literal["live", "dead", "uncertain"]:
    """Probe only what the platform can prove; uncertainty never permits theft."""
    if pid <= 0:
        return "uncertain"
    if os.name == "nt":
        # OpenProcess is the Windows liveness primitive. Access denial means the
        # process may exist under another integrity level, so fail closed.
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if handle:
                kernel32.CloseHandle(handle)
                return "live"
            error = ctypes.get_last_error()
            return "dead" if error == 87 else "uncertain"  # ERROR_INVALID_PARAMETER
        except Exception:
            return "uncertain"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "live"
    except OSError:
        return "uncertain"
    return "live"


def _read_claim(path: Path) -> tuple[str | None, int | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        token = data.get("generation")
        pid = data.get("pid")
        return (str(token) if token else None, int(pid) if isinstance(pid, int) else None)
    except (OSError, ValueError, TypeError):
        return None, None


@dataclass(frozen=True)
class SessionLease:
    """One generation claim; release removes only this exact generation."""

    session_dir: Path
    generation: str
    pid: int

    def release(self) -> None:
        path = self.session_dir / LEASE_NAME
        token, _ = _read_claim(path)
        if token != self.generation:
            return
        try:
            path.unlink()
        except OSError:
            return
        # The mirror is never authority. Compare its pid so an old generation
        # cannot erase the compatibility signal written by a successor.
        mirror = self.session_dir / MIRROR_NAME
        try:
            if mirror.read_text(encoding="utf-8").strip() == str(self.pid):
                mirror.unlink()
        except OSError:
            pass


def acquire_session_lease(session_dir: Path, pid: int | None = None) -> SessionLease:
    """Atomically acquire sole-writer ownership, recovering proven-dead claims."""
    owner_pid = os.getpid() if pid is None else pid
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / LEASE_NAME
    mirror = session_dir / MIRROR_NAME
    # During mixed-version rollout an old writer has only the pid mirror. It is
    # still authoritative when live or uncertain; otherwise a new binary could
    # acquire a lease beside an old binary that knows nothing about leases.
    if not path.exists():
        try:
            legacy_pid = int(mirror.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            legacy_pid = None
        if legacy_pid is not None and _pid_state(legacy_pid) != "dead":
            raise SessionLeaseHeldError(session_dir, legacy_pid)
    generation = secrets.token_hex(16)
    payload = json.dumps(
        {"schema": 1, "session_id": session_dir.name, "generation": generation, "pid": owner_pid},
        separators=(",", ":"),
    ).encode()

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            _, existing_pid = _read_claim(path)
            if existing_pid is None or _pid_state(existing_pid) != "dead":
                raise SessionLeaseHeldError(session_dir, existing_pid)
            tombstone = session_dir / f"{LEASE_NAME}.stale.{secrets.token_hex(8)}"
            try:
                os.replace(path, tombstone)
            except OSError:
                # Another contender recovered it, or Windows would not permit
                # the rename. Re-read rather than granting ourselves ownership.
                continue
            try:
                tombstone.unlink()
            except OSError:
                pass
            continue
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            (session_dir / MIRROR_NAME).write_text(str(owner_pid), encoding="utf-8")
        except OSError:
            pass
        return SessionLease(session_dir, generation, owner_pid)
