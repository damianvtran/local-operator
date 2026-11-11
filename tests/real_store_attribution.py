"""Attribute a real-store shrink to an INDEPENDENT retention sweep, or fail closed.

WHY THIS EXISTS. The root conftest's real-store tripwire fails a run whenever an
entry of the operator's real ``~/.local-operator/sessions`` vanishes during it.
That is the right default — a test reaching the real store is the incident this
guard exists for — but the store is not only ever changed by tests. A live
``lop`` session on the same machine runs its own retention sweeper (startup
maintenance, ``delegated_max_age`` and friends) on its own clock, and a pytest
run that happens to overlap it was failed as though the suite had deleted those
sessions. That happened in test run ebc3f4ee0319: 24 stale delegated sessions
removed by an unrelated ``lop`` process (pid 34093) whose own cleanup log
recorded ``actor=startup policy=delegated_max_age``, and the suite was blamed.

WHAT IT DOES. Only the explanation is new; the protection is not weakened. A
missing entry is *excused* (reported as a warning, never silently dropped) only
when the product's OWN audit row for it, appended to ``.cleanup-log.jsonl``
during this run, proves an automatic retention policy removed it from a process
that is provably not part of this pytest run. Everything else still fails.

WHY ATTRIBUTION FAILS CLOSED. A false excuse is strictly worse than a false
alarm: a false alarm costs a re-run, a false excuse hides the exact data-loss
incident the tripwire is for. So every input that cannot be verified — a log
that was truncated, rotated or is malformed, an actor that is dead, a ``ps``
that cannot answer, a process whose start time cannot be ordered against this
run — is treated as "not explained". In particular a bare ``pid != os.getpid()``
proves nothing: a test can spawn a real ``lop`` that runs the very same startup
sweep, and that child (or a grandchild re-parented to launchd after its parent
exited, which defeats a parent-chain walk) writes an otherwise-genuine row.

THE INDEPENDENCE PROOF is process START ORDER, not ancestry: a process that
started strictly before this run's root pytest process cannot be its
descendant, and a process that is alive now and started before the row was
written is the process that held that pid when it wrote the row (pids are
unique among live processes), which is what rules out pid reuse. ``ps -o
lstart=`` is second-resolution, so an actor that started in the same second as
the root is NOT proved independent and fails. Ancestry is still used, but only
to verify that the recorded root really is this process or one of its ancestors
(an inherited marker that cannot be verified gives no root, hence no excuse).

The product's cleanup code and its audit schema are deliberately untouched;
this reads the rows ``cleanup.remove_session_dir`` already writes BEFORE its
``rmtree`` (``at``, ``session``, ``policy``, ``actor``, ``pid``, ``path``,
``guards``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

#: The product's cleanup log, inside the store it describes
#: (``local_operator.session.cleanup.CLEANUP_LOG_NAME``). Not imported: the root
#: conftest runs before anything else and must not pay for, or depend on, the
#: product import graph. ``tests/unit/test_real_store_attribution.py`` pins the
#: two names together so a rename fails a test instead of disabling excuses.
CLEANUP_LOG_NAME = ".cleanup-log.jsonl"

#: The ONLY actor that removes on its own clock. ``"cli"`` is an operator
#: command and ``"runtime"`` an explicit delete; neither is retention, and a
#: test can run either, so neither is ever excused.
AUTOMATIC_ACTOR = "startup"

#: Policies the automatic sweep removes under: ``run_cleanup``'s
#: ``remove_empty``/``max_inactive_days``/``max_sessions``/``max_total_bytes``
#: and ``delegated_retention.DELEGATED_POLICY``. ``explicit-delete``,
#: ``mesh-move``, ``mesh-remnant`` and ``removal-failed`` are absent on purpose:
#: those are deliberate acts or failures, and stay failures here. Pinned against
#: the product by the unit test named above.
AUTOMATIC_POLICIES = frozenset(
    {"delegated_max_age", "remove_empty", "max_inactive_days", "max_sessions", "max_total_bytes"}
)

#: Every removal row written by ``remove_session_dir`` names the gates that
#: cleared it. A row without both is not one the product's guarded path wrote.
REQUIRED_GUARDS = frozenset({"refusal", "holds"})

#: Environment marker the FIRST process to load the conftest (the controller,
#: before xdist spawns workers) sets to ``"<pid>:<lstart epoch>"`` so workers
#: compare against the controller's start, not their own later one. Verified
#: against the actual ancestry on every read, so a stale or forged value can
#: only remove the root (and with it every excuse), never widen it. Deliberately
#: NOT ``LOCAL_OPERATOR_``/``LOP_``-prefixed: several rigs strip those prefixes
#: from a child's environment, and a nested run that lost the marker would take
#: its own, LATER start as the root and so excuse more than the outer run would.
ROOT_ENV = "PYTEST_REAL_STORE_TRIPWIRE_ROOT"

#: Same bound, and the same reason, as ``team_bridge/connector/reap.py``'s
#: ``PROBE_TIMEOUT_S``: this runs inside a process (here: pytest's session
#: finish) that must not hang on an unresponsive process table.
PROBE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ProcessIdentity:
    """What ``ps`` answers for one live pid: its parent and start (epoch s)."""

    pid: int
    ppid: int
    start: int


Probe = Callable[[int], "ProcessIdentity | None"]


def read_process(pid: int) -> ProcessIdentity | None:
    """``ps``'s ppid and start time for ``pid``; ``None`` when it cannot say.

    ``ps`` because macOS has no ``/proc`` and psutil is not a dependency (the
    same reasoning as ``tools/group_reaper._owner_start_token``). Unlike that
    helper the start IS parsed, because this check must ORDER two starts; a
    token that does not parse yields ``None`` and so fails closed, which is the
    property the opaque-token rule there protects.
    """
    if pid <= 0:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell
            ["ps", "-o", "ppid=,lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PROBE_TIMEOUT_S,
            env={**os.environ, "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    parts = completed.stdout.strip().split(None, 1)
    if len(parts) != 2:
        return None
    try:
        ppid = int(parts[0])
        # lstart is local wall time under LC_ALL=C (``Tue Nov 10 18:31:38 2026``)
        # in the zone of the environment passed above, which is ours.
        fields = time.strptime(" ".join(parts[1].split()), "%a %b %d %H:%M:%S %Y")
        # A local time inside a DST fall-back hour names TWO instants an hour
        # apart; picking one could date a process an hour too early and so
        # "prove" it predates the run. Both readings are tried and an ambiguous
        # (or nonexistent) one is refused rather than guessed.
        readings = set()
        for dst in (0, 1):
            epoch = int(time.mktime((*fields[:8], dst)))
            if time.localtime(epoch)[:6] == fields[:6]:
                readings.add(epoch)
    except (ValueError, OverflowError):
        return None
    if len(readings) != 1:
        return None
    return ProcessIdentity(pid=pid, ppid=ppid, start=readings.pop())


def _ancestry(pid: int, probe: Probe) -> list[ProcessIdentity] | None:
    """``pid`` and every ancestor up to the top, or ``None`` if any link is unreadable."""
    chain: list[ProcessIdentity] = []
    seen: set[int] = set()
    current = pid
    while current > 0 and current not in seen:
        seen.add(current)
        identity = probe(current)
        if identity is None:
            return None
        chain.append(identity)
        current = identity.ppid
    return chain


def record_root(probe: Probe | None = None) -> None:
    """Make the first process to load the conftest this run's root.

    Called at the snapshot. Only the first loader pays for one ``ps``; later
    loaders (xdist workers, nested pytest runs, the private conftest copies
    some tests load) inherit the marker, so the OLDEST root wins, which is the
    conservative direction: an earlier root excuses fewer processes. Nothing is
    verified here; :func:`establish_root` verifies at the point of use, so the
    ancestry walk is only paid for by a run whose store actually shrank.
    """
    if os.environ.get(ROOT_ENV):
        return
    me = (probe or read_process)(os.getpid())
    if me is not None:
        os.environ[ROOT_ENV] = f"{me.pid}:{me.start}"


def establish_root(probe: Probe | None = None) -> ProcessIdentity | None:
    """The recorded root, verified to be this process or one of its ancestors."""
    chain = _ancestry(os.getpid(), probe or read_process)
    if not chain:
        return None
    raw = os.environ.get(ROOT_ENV, "")
    try:
        pid_text, start_text = raw.split(":", 1)
        recorded = (int(pid_text), int(start_text))
    except ValueError:
        return None
    for identity in chain:
        if (identity.pid, identity.start) == recorded:
            return identity
    return None


@dataclass(frozen=True)
class LogMark:
    """Where the cleanup log stood at the snapshot.

    ``identity`` is ``(st_dev, st_ino)`` or ``None`` when the log did not exist
    yet; ``offset`` ends on a newline so a row being appended at the snapshot is
    read whole afterwards; ``digest`` covers ``[0, offset)`` so a truncate-and-
    regrow (same inode, larger size) is still caught.
    """

    identity: tuple[int, int] | None
    offset: int
    digest: str


def _prefix(handle, size: int) -> tuple[int, str]:
    data = handle.read(size)
    if len(data) != size:
        raise OSError("cleanup log shrank while it was being read")
    offset = data.rfind(b"\n") + 1
    return offset, hashlib.sha256(data[:offset]).hexdigest()


def mark_log(store: Path) -> LogMark | None:
    """Snapshot the log; ``None`` when it exists but cannot be read (no excuses)."""
    path = store / CLEANUP_LOG_NAME
    try:
        with open(path, "rb") as handle:
            stat = os.fstat(handle.fileno())
            offset, digest = _prefix(handle, stat.st_size)
    except FileNotFoundError:
        return LogMark(identity=None, offset=0, digest=hashlib.sha256(b"").hexdigest())
    except OSError:
        return None
    return LogMark(identity=(stat.st_dev, stat.st_ino), offset=offset, digest=digest)


def appended_rows(store: Path, mark: LogMark) -> tuple[list[dict[str, Any]], str | None]:
    """Rows appended since ``mark``, or ``([], why)`` when the log is not trustworthy."""
    path = store / CLEANUP_LOG_NAME
    try:
        with open(path, "rb") as handle:
            stat = os.fstat(handle.fileno())
            if mark.identity is not None and (stat.st_dev, stat.st_ino) != mark.identity:
                return [], "cleanup log was replaced (rotated or recreated) during the run"
            if stat.st_size < mark.offset:
                return [], "cleanup log shrank (truncated) during the run"
            head = handle.read(mark.offset)
            if hashlib.sha256(head).hexdigest() != mark.digest:
                return [], "cleanup log's pre-run contents changed during the run"
            tail = handle.read(stat.st_size - mark.offset)
    except FileNotFoundError:
        return [], "cleanup log is missing"
    except OSError as exc:
        return [], f"cleanup log unreadable: {exc}"
    if tail and not tail.endswith(b"\n"):
        return [], "cleanup log ends in a partial row"
    rows: list[dict[str, Any]] = []
    for line in tail.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            return [], "cleanup log has a malformed row appended during the run"
        if not isinstance(row, dict):
            return [], "cleanup log has a non-object row appended during the run"
        rows.append(row)
    return rows, None


def _row_epoch(row: dict[str, Any]) -> int | None:
    try:
        return int(datetime.strptime(str(row["at"]), "%Y-%m-%dT%H:%M:%S%z").timestamp())
    except (KeyError, ValueError, OverflowError):
        return None


@dataclass
class Attribution:
    explained: dict[str, dict[str, Any]] = field(default_factory=dict)
    unexplained: dict[str, str] = field(default_factory=dict)


def attribute(
    missing: Iterable[str],
    *,
    store: Path,
    mark: LogMark | None,
    started_at: float | None,
    root: ProcessIdentity | None,
    now: float | None = None,
    probe: Probe | None = None,
) -> Attribution:
    """Split ``missing`` into excused (with their row) and unexplained (with why).

    ``store`` must be the already-resolved real store. Every condition below is
    necessary; failing any one leaves the entry unexplained.
    """
    # Late-bound so a test can substitute the module's probe; a default bound
    # at definition time would keep the real ``ps`` regardless.
    probe = probe or read_process
    result = Attribution()
    names = sorted(missing)
    blanket: str | None = None
    rows: list[dict[str, Any]] = []
    if mark is None:
        blanket = "cleanup log was unreadable at the snapshot"
    elif started_at is None:
        blanket = "no snapshot time recorded"
    elif root is None:
        blanket = "this run's root pytest process could not be verified"
    else:
        rows, blanket = appended_rows(store, mark)
    if blanket is not None:
        result.unexplained = {name: blanket for name in names}
        return result
    assert started_at is not None and root is not None
    window = (int(started_at), int(time.time() if now is None else now))
    own = {os.getpid(), root.pid}
    actors: dict[int, ProcessIdentity | None] = {}
    for name in names:
        mine = [row for row in rows if row.get("session") == name]
        if not mine:
            result.unexplained[name] = "no cleanup-log row appended during this run"
            continue
        why = None
        for row in mine:
            why = _disqualify(row, name, store, window, own, root, actors, probe)
            if why is not None:
                break
        if why is None:
            result.explained[name] = mine[-1]
        else:
            result.unexplained[name] = why
    return result


def _disqualify(
    row: dict[str, Any],
    name: str,
    store: Path,
    window: tuple[int, int],
    own: set[int],
    root: ProcessIdentity,
    actors: dict[int, ProcessIdentity | None],
    probe: Probe,
) -> str | None:
    """The first reason ``row`` does not prove an independent retention removal."""
    path = row.get("path")
    if not isinstance(path, str) or os.path.basename(path) != name:
        return f"row path {path!r} is not this entry"
    if Path(os.path.realpath(os.path.dirname(path))) != store:
        return f"row path {path!r} is not under {store}"
    if row.get("actor") != AUTOMATIC_ACTOR or row.get("policy") not in AUTOMATIC_POLICIES:
        return f"actor={row.get('actor')!r} policy={row.get('policy')!r} is not automatic retention"
    guards = row.get("guards")
    if not isinstance(guards, list) or not REQUIRED_GUARDS.issubset(guards):
        return f"row guards {guards!r} lack {sorted(REQUIRED_GUARDS)}"
    at = _row_epoch(row)
    if at is None or not window[0] <= at <= window[1]:
        return f"row time {row.get('at')!r} is outside this run"
    pid = row.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return f"row pid {pid!r} is not a process id"
    if pid in own:
        return f"pid {pid} is this pytest run"
    if pid not in actors:
        actors[pid] = probe(pid)
    actor = actors[pid]
    if actor is None:
        return f"actor pid {pid} is dead or unreadable, so its independence cannot be proved"
    # Strictly earlier, at ps's one-second resolution: an equal second is
    # ambiguous and therefore not a proof (see the module docstring).
    if not actor.start < root.start:
        return (
            f"actor pid {pid} started at {actor.start}, not before this run's root "
            f"pid {root.pid} ({root.start}); it may be a descendant or a reused pid"
        )
    return None
