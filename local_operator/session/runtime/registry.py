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
record behind) and heartbeat freshness (a live pid whose owner has stopped
reporting).

**Freshness is EVIDENCE, not a verdict.** The beat is authored by the
runtime's own event loop, so on the in-process kinds (``daemon``, ``exec``)
a long turn or a starved scheduler stalls it while the process is
demonstrably working — measured on this host at 105.8 s and 205.8 s against a
45 s timeout. A stale beat therefore says the owner is not answering, and
nothing more: not that it is dead, not that its workload stopped. The words
on every surface follow that rule (:func:`classify`).

**A dead record is evidence, not litter.** ``scan`` used to unlink a record as
soon as its pid was proven gone, which is precisely the file the attention
classifier reads to say what a dead run was — and ``scan`` IS the mobile
daemon's discovery loop, so a sweep that happened to run first destroyed the
answer before anyone asked the question. Stale records are now MOVED to
``reaped/`` (bounded by count and age, invisible to discovery); see
:data:`REAPED_DIRNAME`. The same module also carries the DURABLE STOP MARKER a
killer stages before an irreversible step (:data:`STOP_MARKER_NAME`), because
both exist to make one runtime death attributable from the artifacts alone.

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
from typing import Any, Callable, Literal, NamedTuple, TypeVar

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

#: Where a scan MOVES a record whose owner it has proven dead, instead of
#: deleting it.
#:
#: A SUBDIRECTORY of the run directory, and that placement is load-bearing
#: rather than tidy. Every discovery reader globs ``*.json`` in the run
#: directory itself (``scan`` here, and everything downstream of it), so a
#: sidecar is invisible to ``lop sessions``, the picker and the phone while
#: staying readable by the one reader that must see it: the attention
#: classifier, which is asked "why did this run die" *after* some sweep has
#: already reaped the record. Deleting on reap destroyed exactly that evidence
#: — ``registry.scan`` is the mobile daemon's discovery loop, so a run whose
#: record was reaped before the classifier read it landed on the no-evidence
#: arm and reported "the cause could not be determined" for a death that HAD a
#: determined cause (INCIDENT 2026-09-13, session ``5e109d459222``). A record
#: here is evidence only; it is never a source for discovery, so a sidecar
#: must not be added to any listing's glob.
REAPED_DIRNAME = "reaped"

#: Retention for the sidecar, by COUNT and by AGE. The evidence is worth one
#: look after the death (the classifier runs at the next engage), not
#: indefinite storage: without a bound this directory would grow one file per
#: session death forever, and a host that has been up for months is exactly
#: where someone would be reading it. 200 entries is past any plausible burst
#: (the incident was twelve) and a day is past any plausible reading delay.
REAPED_MAX_FILES = 200
REAPED_MAX_AGE_S = 24 * 60 * 60.0


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
    target = directory / f"{record.pid}.json"
    _staged_write(target, record.to_json(), prefix=f".{record.pid}.")
    return target


def _staged_write(target: Path, payload: Any, *, prefix: str) -> None:
    """Write ``payload`` as JSON to ``target`` staged, 0600.

    The one write shape every runtime artifact here uses — a temp file in the
    SAME directory (so the rename cannot cross a filesystem), ``json.dump``,
    chmod 0600, ``os.replace`` — so a reader of any of them sees either the
    old bytes or the new ones and never a torn file. Extracted when the stop
    marker became the second artifact to need it: two copies of this are free
    to disagree about the mode, and the mode is the authorization model for
    the record that carries a control key.

    The target's directory is NOT created here: a writer that invented a
    directory would let, say, a stop marker conjure a conversation directory
    that every session listing then shows as an empty session.

    NO FSYNC, AND THE GUARANTEE IS STATED AT THAT STRENGTH. There is no
    ``fsync`` between the write and the rename, so what is guaranteed is
    PROCESS-durability: the bytes are in the page cache and every reader sees
    the old file or the new one, which is the semantics all of these artifacts
    need (the process that dies is the ARTIFACT'S SUBJECT; the host does not).
    Host-durability would be a different contract, and it is not this shape: a
    power loss between the write and the rename is uncovered. Adding it here
    would put an fsync on every heartbeat of every live session, which the one
    artifact that would arguably want it (the stop marker) does not justify.
    A writer killed mid-call leaves a ``.<name>.*.tmp`` sibling: bounded, tiny,
    and never read as the artifact itself (readers name the file, not a
    pattern).
    """
    directory = target.parent
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
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


#: The DURABLE STOP MARKER: the evidence a process leaves in a conversation
#: directory immediately BEFORE it takes an irreversible step against that
#: conversation's runtime.
#:
#: WHY A MARKER AND NOT A LOG LINE. The one fact nobody could recover after the
#: 2026-09-13 kill wave was "who took this runtime down, and was it asked for".
#: At the SIGKILL rung the target is not executing and cannot record anything:
#: the ONLY process that can attest to a kill is the one issuing it, and the
#: only moment it can do so is before the signal. Everything downstream is
#: reconstruction from absence — a missing record, a silent socket — which is
#: why one event reached the operator as three incompatible verdicts
#: (``runtime-killed``, no cause at all, and ``owner-lost``).
#: ``runtime-stop.json`` sits in the conversation directory rather than the run
#: directory because it must outlive the record it describes: the record is
#: unpublished by a clean stop and reaped by a scan, while the transcript
#: directory survives both. Its contents are documented by
#: ``control._stop_marker_payload``, the one writer.
STOP_MARKER_NAME = "runtime-stop.json"


def stop_marker_path(conversation_dir: Path) -> Path:
    """Where one conversation's durable stop marker lives."""
    return conversation_dir / STOP_MARKER_NAME


def write_stop_marker(conversation_dir: Path, payload: dict[str, Any]) -> Path:
    """Stage-write the durable stop marker (0600, :func:`publish`'s shape).

    Raises only when the write itself fails — a stop must be able to report
    that it could not leave evidence — so the caller decides whether a missing
    marker aborts the step. Deliberately does NOT create the conversation
    directory: see :func:`_staged_write`, which is also where the strength of
    the "durable" in this name is stated (process-durable; no fsync).

    Deliberately NOT a permanent write either: ``control`` withdraws its own
    marker when the ladder refuses, so :func:`remove_stop_marker` exists and is
    the only supported way to take one back.
    """
    target = stop_marker_path(conversation_dir)
    _staged_write(target, payload, prefix=f".{STOP_MARKER_NAME}.")
    return target


def remove_stop_marker(conversation_dir: Path) -> None:
    """Take one conversation's stop marker back. Best-effort.

    The ONLY caller is the ladder's refusal path
    (``control._withdraw_staged_stop_marker``), and it checks the file's own
    run key and killer before calling: a marker attests to an act, so a stop
    that ends up acting on nothing must not leave one behind — the target is
    still alive and its later, involuntary death would read as the user's own
    stop. Never raises: a refusal must not fail over evidence cleanup.
    """
    try:
        stop_marker_path(conversation_dir).unlink()
    except OSError:
        pass


def read_stop_marker(conversation_dir: Path) -> dict[str, Any] | None:
    """The durable stop marker as a dict, or ``None`` when there is none.

    Tolerant by design: an unreadable or malformed marker means "no usable
    evidence", never an exception on a classification path that runs at
    session boot.
    """
    try:
        data = json.loads(stop_marker_path(conversation_dir).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


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
    startup would trade a rare stale row for a routine slowdown.
    :func:`classify` derives the policy (probe only where the answer changes
    what a user is told) and `scan` delegates to it, so the rule has one home
    for every reader.

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


class Liveness(NamedTuple):
    """One record's liveness verdict, plus the two facts behind it.

    The FIELDS are as load-bearing as the state: a caller that wants the age
    without re-deriving it (the sidebar tooltip, ``/info``'s row, the wake
    supervisor's sentence) reads it here rather than deciding for itself what
    a clock step or a future-dated stamp means.
    """

    state: Literal["live", "wedged", "stale"]
    pid_alive: bool
    heartbeat_age_s: float


def classify(
    record: DiscoveryRecord,
    *,
    now: float | None = None,
    check_zombie: bool | None = None,
) -> Liveness:
    """The single owner of the ``live`` / ``wedged`` / ``stale`` vocabulary.

    Two facts, no more: pid liveness and heartbeat freshness.

    - ``stale`` — the pid is gone. Nothing is left to talk to.
    - ``wedged`` — the pid exists and the owner has not written a beat inside
      ``HEARTBEAT_TIMEOUT_S``. **DEGRADED-RESPONSIVENESS EVIDENCE, and that is
      the whole of the claim.** The beat is authored by the runtime's own
      event loop, so for the in-process kinds a long turn, a starved
      scheduler or a loop parked in a blocking call produces exactly this
      reading while the process executes work — measured on this host at
      105.8 s and 205.8 s gaps against a 45 s timeout, with CPU time
      advancing. It is therefore NOT proof the process is dead, NOT proof its
      workload stopped, and NOT a diagnosis; every surface that renders it
      says the owner is not ANSWERING rather than that it is broken (see
      ``session.catalog.CatalogEntry.status`` and the wake supervisor's
      sentence).
    - ``live`` — the pid exists and the beat is fresh. The converse caveat
      applies: it says the owner reported recently, and nothing about whether
      its control socket is free this instant.

    The word ``wedged`` is kept as the STATE because ~15 call sites and the
    desktop catalogue's ``status.code`` branch on it, and a rename would
    change a wire value to restate a sentence; the honesty lives in the words
    a person reads, which is where it was missing.

    ``check_zombie=None`` derives the probe policy rather than making every
    caller restate it (a ``ps`` fork on macOS, so it is spent only where the
    answer changes what a user is told): probe when the heartbeat has already
    gone quiet, which is either a stopped reporter or a process that died
    without being reaped. Pass a bool to force it.
    """
    moment = time.time() if now is None else now
    # Clamped: a stamp dated in the future is clock skew, never evidence
    # against the process, so it can only make this register quieter.
    age = max(0.0, moment - record.heartbeat_at)
    zombie_probe = age > HEARTBEAT_INTERVAL_S * 1.5 if check_zombie is None else check_zombie
    alive = pid_alive(record.pid, check_zombie=zombie_probe)
    if not alive:
        state: Literal["live", "wedged", "stale"] = "stale"
    elif age > HEARTBEAT_TIMEOUT_S:
        state = "wedged"
    else:
        state = "live"
    return Liveness(state=state, pid_alive=alive, heartbeat_age_s=age)


def scan(
    root: Path | None = None,
    dirname: str = RUN_DIRNAME,
    parse: Callable[[dict[str, Any]], T] = SessionRecord.from_json,
) -> list[tuple[T, str]]:
    """Read every record in one namespace, classifying each as ``live`` /
    ``wedged`` / ``stale``.

    - ``stale``: pid is gone — the record is moved aside (:func:`_reap_dead_record`).
    - ``wedged``: pid alive but the owner has not reported for longer than
      the timeout. The daemon keeps the record and shows it degraded; the
      verdict's exact weight is :func:`classify`'s to state, and it is not a
      claim that the process is dead.
    - ``live``: pid alive and heartbeating.

    Unparseable records are deleted, not moved: a torn file has no pid to key
    a sidecar on and nothing an "why did this die" reader could use.

    It stays the one implementation of the state rule — the tuple shape is
    deliberate, because ~15 call sites read it positionally and most want
    only the state. A caller that also wants the AGE asks the same record's
    :func:`classify` rather than widening this return type.

    ``parse`` is what makes the rule above usable by both namespaces without a
    second copy of it: the classification reads only ``pid`` and
    ``heartbeat_at``, which every record type has, so the caller supplies the
    deserializer for its own type and gets its own type back. It defaults to
    the session record, which is what every existing caller means.

    A STALE RECORD IS EVIDENCE, SO IT IS REPLACED, NOT DESTROYED. This
    function is the daemon's discovery loop as well as `lop sessions`, so a
    caller that classified after its own sweep used to find the record already
    deleted and could only report "no cause determined" for a death whose
    cause was on disk a moment earlier. The move is a rename within the run
    directory, so it costs the same unlink it replaced; see
    :data:`REAPED_DIRNAME` for why the sidecar is invisible to discovery.
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
        # The zombie probe is the CLASSIFIER's policy now, not this
        # function's: it costs a `ps` fork on macOS, so it is spent only on
        # records whose heartbeat has already gone quiet — a healthy runtime
        # beats every 15 s, so a quiet stamp means either a stopped reporter or
        # a process that died without being reaped. That is exactly the case
        # that used to report `live` with 0B RSS for 45 s (round 3, U10), and
        # it keeps the common path (every session, every `lop` invocation)
        # fork-free. The reaping stays HERE, because it is this function's
        # contract with its callers rather than a fact about the record.
        verdict = classify(record, now=now)
        if not verdict.pid_alive:
            _reap_dead_record(directory, path, record.pid)
        out.append((record, verdict.state))
    return out


def reaped_dir(root: Path | None = None, dirname: str = RUN_DIRNAME) -> Path:
    """:func:`run_dir`'s sidecar, created 0700 on first use.

    For the WRITER path and for tests asserting retention. The reader
    (``attention._run_record_evidence``) intentionally does NOT call this: it
    opens the run directory it was pointed at and globs ``REAPED_DIRNAME``
    beneath it, because a READ must never create a directory on a machine whose
    only problem is that something died.
    """
    path = run_dir(root, dirname) / REAPED_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def _unlink_quietly(path: Path) -> None:
    """Best-effort delete, the shape every reaper here ends in."""
    try:
        path.unlink()
    except OSError:
        pass


def _reap_dead_record(directory: Path, path: Path, pid: int) -> None:
    """Move one proven-dead record into the sidecar instead of deleting it.

    Best-effort in the same direction :func:`scan` always was: a sidecar that
    cannot be written (a full disk, a read-only root) falls back to the old
    delete, because failing the whole classification to preserve evidence
    would break `lop sessions` on exactly the machine under stress. The
    fallback loses the evidence, so it is the second choice, not the default.
    """
    sidecar = directory / REAPED_DIRNAME
    try:
        sidecar.mkdir(parents=True, exist_ok=True)
        os.chmod(sidecar, 0o700)
        # Keyed by pid like every record here, and REPLACED rather than
        # uniquified: two deaths of the same pid are two different processes,
        # and the newer one (a recycled pid) is the one the classifier is
        # being asked about.
        os.replace(path, sidecar / f"{pid}.json")
    except OSError:
        _unlink_quietly(path)
        return
    _prune_reaped(sidecar)


def _prune_reaped(directory: Path) -> None:
    """Keep the sidecar bounded by AGE first, then by COUNT.

    Age is applied first so a burst cannot evict today's evidence in favour of
    yesterday's; count still bounds a machine that dies faster than a day,
    keeping the newest :data:`REAPED_MAX_FILES`. Best-effort throughout — this
    runs inside a discovery sweep, and a listing must never fail over
    housekeeping.
    """
    try:
        entries: list[tuple[float, Path]] = []
        for path in directory.glob("*.json"):
            try:
                entries.append((path.stat().st_mtime, path))
            except OSError:
                continue
    except OSError:
        return
    cutoff = time.time() - REAPED_MAX_AGE_S
    fresh: list[tuple[float, Path]] = []
    for mtime, path in entries:
        if mtime < cutoff:
            _unlink_quietly(path)
        else:
            fresh.append((mtime, path))
    fresh.sort()
    for _, path in fresh[: max(0, len(fresh) - REAPED_MAX_FILES)]:
        _unlink_quietly(path)


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
