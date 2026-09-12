"""Atomic sole-writer leases for durable session transcripts.

The compatibility ``.session.pid`` marker is useful for old readers but cannot
be authoritative because replacing a text file is not acquisition.  This
module stays stdlib-only so resume discovery can consult it without importing
the engine or mobile stack.

Liveness is asked through :func:`local_operator.procstate.is_zombie` rather
than signal 0 alone, because the claim this module arbitrates is only ever
taken over from a holder that is PROVEN dead, and a zombie is exactly the
holder that looks alive forever.  The probe stays a leaf module so this one
keeps its stdlib-only contract.
"""

from __future__ import annotations

import json
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from local_operator.procstate import is_zombie

LEASE_NAME = ".execution-lease"
MIRROR_NAME = ".session.pid"
RECOVERY_LOCK_NAME = ".execution-lease.recovery"


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


def _pid_state(pid: int, *, check_zombie: bool = True) -> Literal["live", "dead", "uncertain"]:
    """Probe only what the platform can prove; uncertainty never permits theft.

    **An exited-but-unreaped process is DEAD here, not live.** Signal 0
    succeeds against a zombie, so a probe built on it alone reports the corpse
    of a SIGKILLed runtime as a working owner — forever, because the pid is not
    reused while it lingers and nothing may reap it on the owner's behalf. Both
    readers of this verdict require a holder to be *proven dead* before they
    move its claim, so the wrong answer here is not a misleading log line: it
    is a transcript whose sole-writer claim can never be acquired. The operator
    sees a session that no interface will open (`live_runtime_pid` refuses with
    "already open in pid N", where N is the corpse) and no mechanism will
    recover. :func:`local_operator.procstate.is_zombie` documents the incident.

    **The zombie probe is not free, so the caller decides how often it is
    worth spending.** Signal 0 costs ~1 µs and settles two of the three
    answers (a live owner, a pid that is simply gone); the proof costs a `ps`
    fork (2.4-3.9 ms measured across runs on this host) and is therefore spent
    only after the cheap probe has already said "live". ``check_zombie=False``
    skips it,
    which is what the engage loop passes while it is polling every 10 ms
    against a claim it believes is mid-construction (`launch._lease_holder`).
    That choice is a LATENCY trade, never a safety one: the verdict it produces
    can be "live" for a corpse only where the caller has already decided to
    wait, and every acquisition still requires the proof before it may take a
    claim.
    """
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
        # Another account's process. It cannot be a zombie of ours, and an
        # unverifiable holder must never lose its claim: fail closed.
        return "live"
    except OSError:
        return "uncertain"
    if not check_zombie:
        # The caller is on a dense poll cadence and has already accepted that
        # it will wait; see the docstring. A zombie reads as live here.
        return "live"
    return "dead" if is_zombie(pid) else "live"


def _read_claim(path: Path) -> tuple[str | None, int | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        token = data.get("generation")
        pid = data.get("pid")
        return (str(token) if token else None, int(pid) if isinstance(pid, int) else None)
    except (OSError, ValueError, TypeError):
        return None, None


@contextmanager
def _stale_recovery_right(session_dir: Path) -> Iterator[bool]:
    """Try to serialize stale takeover with a crash-released kernel lock.

    A persistent side file avoids the same pathname replacement race as the
    lease itself. The kernel lock, not the file's lifetime, is authoritative:
    process death releases it automatically, so recovery cannot become
    immortal. Windows lock failures stay closed because access denial and a
    live contender are intentionally indistinguishable here.
    """
    path = session_dir / RECOVERY_LOCK_NAME
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    token = secrets.token_hex(16)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            try:
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                yield False
                return
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                yield False
                return
        # Metadata is diagnostic only, but records both process identity and a
        # per-attempt token so a surviving file is never mistaken for authority.
        os.ftruncate(fd, 0)
        os.write(
            fd,
            json.dumps({"pid": os.getpid(), "token": token}, separators=(",", ":")).encode(),
        )
        os.fsync(fd)
        yield True
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


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


def reap_proven_dead_session_claim(session_dir: Path, owner_pid: int) -> bool:
    """Remove only the exact dead owner's lease and compatibility mirror.

    The daemon calls this after discovery proves a record pid is gone. Recovery
    still revalidates under the same kernel lock used by acquisition, because a
    successor may claim the durable transcript between the scan and cleanup.
    Windows remains conservative through ``_pid_state`` and lock acquisition.
    """
    if _pid_state(owner_pid) != "dead":
        return False
    path = session_dir / LEASE_NAME
    with _stale_recovery_right(session_dir) as may_recover:
        if not may_recover:
            return False
        generation, current_pid = _read_claim(path)
        if generation is None or current_pid is None or current_pid != owner_pid:
            return False
        if _pid_state(current_pid) != "dead":
            return False
        # Re-read immediately before unlink so cleanup is generation-fenced even
        # if a future platform changes lock semantics around pathname replacement.
        if _read_claim(path) != (generation, current_pid):
            return False
        try:
            path.unlink()
        except OSError:
            return False
        mirror = session_dir / MIRROR_NAME
        try:
            if mirror.read_text(encoding="utf-8").strip() == str(owner_pid):
                mirror.unlink()
        except OSError:
            pass
        return True


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
            inspected_generation, inspected_pid = _read_claim(path)
            if inspected_pid is None or _pid_state(inspected_pid) != "dead":
                raise SessionLeaseHeldError(session_dir, inspected_pid)
            with _stale_recovery_right(session_dir) as may_recover:
                if not may_recover:
                    # A live recoverer is indistinguishable from ownership until
                    # it publishes its successor. Fail closed instead of racing it.
                    _, current_pid = _read_claim(path)
                    raise SessionLeaseHeldError(session_dir, current_pid)
                current_generation, current_pid = _read_claim(path)
                if (
                    current_generation != inspected_generation
                    or current_pid != inspected_pid
                    or current_pid is None
                    or _pid_state(current_pid) != "dead"
                ):
                    # The exact generation/process pair changed, became live, or
                    # cannot still be proven dead. Never move that successor.
                    raise SessionLeaseHeldError(session_dir, current_pid)
                tombstone = session_dir / f"{LEASE_NAME}.stale.{secrets.token_hex(8)}"
                try:
                    os.replace(path, tombstone)
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except OSError:
                    # Windows rename denial and any unexpected successor both
                    # stay closed; neither permits speculative ownership.
                    raise SessionLeaseHeldError(session_dir, current_pid) from None
                finally:
                    try:
                        tombstone.unlink()
                    except OSError:
                        pass
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            mirror.write_text(str(owner_pid), encoding="utf-8")
        except OSError:
            pass
        return SessionLease(session_dir, generation, owner_pid)
