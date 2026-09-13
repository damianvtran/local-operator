"""Discovery records: how every lop session, and every ``serve`` daemon,
becomes findable.

One JSON file per live process at ``<config root>/<dirname>/<pid>.json``, mode
0600 under a 0700 directory — for a session the file is the only place its
control key exists outside the process that owns it, so the permissions ARE
the authorization model: anything that can read the record is already the
owning account, and the daemon needs no credential of its own to adopt a
session.

Publication is staged-write + rename so a scanner never reads a torn record,
and every write rewrites the heartbeat, so "is this process alive" is two
checks with no coordination: pid liveness (a SIGKILLed process leaves its
record behind) and heartbeat freshness (a live pid whose owner wedged).

**Two namespaces share this one implementation.** A session publishes to
``RUN_DIRNAME`` and a ``serve`` daemon to ``SERVE_RUN_DIRNAME`` (see its
comment in :mod:`types` for why they are separate directories); everything
after the directory is identical, so the ``dirname`` parameter below is the
only difference between them. Deliberately NOT a second copy of the staged
write, the permissions or the ``live``/``wedged``/``stale`` rule: two copies
would be free to disagree about what "alive" means, and the reader that
disagreed would be the one nobody was looking at.

Stdlib-only and import-light: the runtime sits on the CLI startup path.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from local_operator.paths import config_dir
from local_operator.procstate import is_zombie
from local_operator.session.runtime.types import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_TIMEOUT_S,
    RUN_DIRNAME,
    DiscoveryRecord,
    SessionRecord,
)

#: Narrower than :class:`DiscoveryRecord` on purpose: the caller's own record
#: type comes back out of :func:`scan`, so a typed caller (the picker, the
#: attach client, ``lop sessions``) keeps its precise field access instead of
#: being handed the three members the shared path happens to read.
T = TypeVar("T", bound=DiscoveryRecord)


def run_dir(root: Path | None = None, dirname: str = RUN_DIRNAME) -> Path:
    """The record directory, created 0700 on first use. The daemon creates it
    at startup too, so the very first session on a fresh machine is caught."""
    path = (root or config_dir()) / dirname
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def record_path(pid: int, root: Path | None = None, dirname: str = RUN_DIRNAME) -> Path:
    """Where one process's record lives. Keyed by pid, like every reader here.

    The ``<pid>.json`` spelling was inline in three places (publish, unpublish,
    and every caller that wanted to NAME the file). It is named once now
    because a host that publishes a record may need to tell an external
    supervisor where to read it — ``exec --control`` prints exactly this path
    on stderr so the supervisor can pick the control key out of a file only
    the owning account can open, rather than being handed the key in a log.

    Creating the directory is :func:`run_dir`'s job and happens here too, so
    the returned path's parent always exists with the right mode.
    """
    return run_dir(root, dirname) / f"{pid}.json"


def publish(record: DiscoveryRecord, root: Path | None = None, dirname: str = RUN_DIRNAME) -> Path:
    """Write (or refresh) a process's record, staged so scanners see either
    the old file or the new one, never a half-written one."""
    directory = run_dir(root, dirname)
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


def unpublish(pid: int, root: Path | None = None, dirname: str = RUN_DIRNAME) -> None:
    """Remove a process's record on clean exit. Best-effort: an exit path
    must never raise over a missing file."""
    try:
        record_path(pid, root, dirname).unlink()
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

    The probe itself lives in :func:`local_operator.procstate.is_zombie`, which
    is the one implementation the lease and the resume path share: a holder
    that is a zombie must be reported dead by all three, or discovery reaps the
    record while the claim that keeps the session un-attachable survives it.
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
    return not check_zombie or not is_zombie(pid)


def scan(
    root: Path | None = None,
    dirname: str = RUN_DIRNAME,
    parse: Callable[[dict[str, Any]], T] = SessionRecord.from_json,
) -> list[tuple[T, str]]:
    """Read every record in one namespace, classifying each as ``live`` /
    ``wedged`` / ``stale``.

    - ``stale``: pid is gone — the caller reaps the file.
    - ``wedged``: pid alive but heartbeat older than the timeout — the owner
      is stuck; the daemon shows it degraded and keeps the record.
    - ``live``: pid alive and heartbeating.

    Unparseable records are treated as stale and reaped: the only writers are
    this module's staged writes, so a torn file means an interrupted crash,
    not a format to preserve.

    ``parse`` is what makes the rule above usable by both namespaces without a
    second copy of it: the classification reads only ``pid`` and
    ``heartbeat_at``, which every record type has, so the caller supplies the
    deserializer for its own type and gets its own type back. It defaults to
    the session record, which is what every existing caller means.
    """
    directory = run_dir(root, dirname)
    out: list[tuple[T, str]] = []
    now = time.time()
    for path in sorted(directory.glob("*.json")):
        try:
            record = parse(json.loads(path.read_text()))
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
    """A process's side of the contract: publish on start, heartbeat on a
    timer, unpublish on exit. Held by the runtime and by the ``serve`` daemon;
    nothing here blocks.

    Typed on the shared protocol rather than on ``SessionRecord`` because the
    daemon's record is not a session: this class touches the pid, the
    heartbeat and the payload, and nothing else. It stays non-generic — no
    caller reads a session-specific field off a publisher, and widening only
    the input is what keeps the two namespaces on one implementation.
    """

    def __init__(
        self,
        record: DiscoveryRecord,
        root: Path | None = None,
        dirname: str = RUN_DIRNAME,
    ) -> None:
        self.record = record
        self._dirname = dirname
        # RESOLVE THE DIRECTORY ONCE, HERE, and use that resolution for the
        # rest of this publisher's life. ``root=None`` means "whatever
        # ``config_dir()`` says now", and ``config_dir()`` deliberately reads
        # the environment on every call (see its docstring: tests re-point it
        # after import) — so leaving ``self._root`` as None made every later
        # ``heartbeat``/``close`` re-resolve it. A runtime that outlived its
        # own config dir then rewrote its record into whatever directory was
        # current at that moment, and deleted THAT file on close, leaving its
        # own record behind: records are keyed by pid alone, so the file it
        # clobbered belonged to a different session. Measured in this suite,
        # where the autouse fixture hands every test a fresh HOME: a runtime
        # whose shutdown landed after the NEXT test had started removed that
        # test's record, which is how two unrelated tests read "no session
        # matches <name>" and "the record still says started=False".
        #
        # Pin the CONFIG dir rather than the run dir so ``root`` keeps meaning
        # what every caller already passes.
        self._root = root if root is not None else config_dir()
        self.path = publish(record, self._root, self._dirname)

    def heartbeat(self, **updates: object) -> None:
        """Rewrite the record with fresh liveness plus any changed fields
        (model switch, conversation rename, new session id after /resume)."""
        for key, value in updates.items():
            if hasattr(self.record, key):
                setattr(self.record, key, value)
        publish(self.record, self._root, self._dirname)

    def close(self) -> None:
        unpublish(self.record.pid, self._root, self._dirname)
