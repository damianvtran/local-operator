"""Discovery records: how every lop session becomes findable.

One JSON file per live session at ``~/.local-operator/run/mobile/<pid>.json``,
mode 0600 under a 0700 directory — the file is the only place a session's
control key exists outside the process that owns it, so the permissions ARE
the authorization model: anything that can read the record is already the
owning account, and the daemon needs no credential of its own to adopt a
session.

Publication is staged-write + rename so a scanner never reads a torn record,
and every write rewrites the heartbeat, so "is this session alive" is two
checks with no coordination: pid liveness (a SIGKILLed session leaves its
record behind) and heartbeat freshness (a live pid whose owner wedged).

Stdlib-only and import-light: the runtime sits on the CLI startup path.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from local_operator.paths import config_dir
from local_operator.session.runtime.types import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_TIMEOUT_S,
    RUN_DIRNAME,
    SessionRecord,
)


def run_dir(root: Path | None = None) -> Path:
    """The record directory, created 0700 on first use. The daemon creates it
    at startup too, so the very first session on a fresh machine is caught."""
    path = (root or config_dir()) / RUN_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def publish(record: SessionRecord, root: Path | None = None) -> Path:
    """Write (or refresh) a session's record, staged so scanners see either
    the old file or the new one, never a half-written one."""
    directory = run_dir(root)
    record.heartbeat_at = time.time()
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{record.pid}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(record.to_json(), handle)
        os.chmod(tmp, 0o600)
        target = directory / f"{record.pid}.json"
        os.replace(tmp, target)
        return target
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def unpublish(pid: int, root: Path | None = None) -> None:
    """Remove a session's record on clean exit. Best-effort: an exit path
    must never raise over a missing file."""
    try:
        (run_dir(root) / f"{pid}.json").unlink()
    except OSError:
        pass


def pid_alive(pid: int, *, check_zombie: bool = False) -> bool:
    """Signal-0 liveness, the cheapest check that answers "is there a process
    with this pid" without disturbing it. EPERM means alive-but-not-ours,
    which for our purposes is alive.

    A ZOMBIE IS NOT ALIVE. `kill(pid, 0)` succeeds against a process that has
    exited but not yet been reaped, so a `kill -9`'d runtime kept reporting
    `live` — with `0B` RSS — until the heartbeat aged it out 45 s later, and
    `lop sessions`, the one place a user checks to understand the failure,
    actively misled them (round 3, U10). The window is real rather than
    theoretical: a runtime's parent is often the shell that launched it and
    has since exited, so nothing reaps the entry promptly.

    Deliberately NOT psutil: this module is stdlib-only by contract (it is on
    the CLI startup path), and `/proc` does not exist on macOS.

    **The zombie probe is opt-in via `check_zombie`**, because on macOS it
    costs a `ps` fork — measured at 3.9 ms, against ~1 µs for signal-0 — and
    `scan()` runs on every `lop` invocation. Paying that per live session on
    startup would trade a rare stale row for a routine slowdown. `scan` asks
    for it only where the answer changes what a user is told.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return not check_zombie or not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    """Whether this pid is an exited-but-unreaped process.

    Fails CLOSED (returns False, i.e. "treat as alive") on any doubt: calling
    a live session dead would reap a record out from under a working runtime,
    which is far worse than the stale row this exists to avoid.
    """
    try:
        proc_status = Path(f"/proc/{pid}/stat")
        if proc_status.exists():  # Linux: no subprocess needed
            # `comm` can contain spaces and parentheses; state is the field
            # after the LAST ')'.
            data = proc_status.read_text()
            return data.rpartition(")")[2].strip().startswith("Z")
        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["/bin/ps", "-o", "state=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except Exception:  # noqa: BLE001 — an unprobeable pid is treated as alive
        return False
    return result.stdout.strip().upper().startswith("Z")


def scan(root: Path | None = None) -> list[tuple[SessionRecord, str]]:
    """Read every record, classifying each as ``live`` / ``wedged`` / ``stale``.

    - ``stale``: pid is gone — the caller reaps the file.
    - ``wedged``: pid alive but heartbeat older than the timeout — the owner
      is stuck; the daemon shows it degraded and keeps the record.
    - ``live``: pid alive and heartbeating.

    Unparseable records are treated as stale and reaped: the only writers are
    this module's staged writes, so a torn file means an interrupted crash,
    not a format to preserve.
    """
    directory = run_dir(root)
    out: list[tuple[SessionRecord, str]] = []
    now = time.time()
    for path in sorted(directory.glob("*.json")):
        try:
            record = SessionRecord.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            try:
                path.unlink()
            except OSError:
                pass
            continue
        # The zombie probe costs a `ps` fork on macOS, so it is spent only on
        # records whose heartbeat has already gone quiet: a healthy runtime
        # beats every 15 s, so a gap means either a wedge or a process that
        # died without being reaped. That is exactly the case that used to
        # report `live` with 0B RSS for 45 s (round 3, U10), and it keeps the
        # common path (every session, every `lop` invocation) fork-free.
        quiet = now - record.heartbeat_at > HEARTBEAT_INTERVAL_S * 1.5
        if not pid_alive(record.pid, check_zombie=quiet):
            try:
                path.unlink()
            except OSError:
                pass
            out.append((record, "stale"))
        elif now - record.heartbeat_at > HEARTBEAT_TIMEOUT_S:
            out.append((record, "wedged"))
        else:
            out.append((record, "live"))
    return out


class RecordPublisher:
    """A runtime's side of the contract: publish on start, heartbeat on a
    timer, unpublish on exit. Held by the runtime; nothing here blocks."""

    def __init__(self, record: SessionRecord, root: Path | None = None) -> None:
        self.record = record
        self._root = root
        self.path = publish(record, root)

    def heartbeat(self, **updates: object) -> None:
        """Rewrite the record with fresh liveness plus any changed fields
        (model switch, conversation rename, new session id after /resume)."""
        for key, value in updates.items():
            if hasattr(self.record, key):
                setattr(self.record, key, value)
        publish(self.record, self._root)

    def close(self) -> None:
        unpublish(self.record.pid, self._root)
