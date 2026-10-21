"""The runtime's own account of what it was doing when it stopped.

WHY THIS MODULE EXISTS, in the operator's own incident. On 2026-09-15 three
fleet outages hit one machine in one evening; the third killed 36 session
runtimes at 19:41 in a single SIGKILL-class sweep, and every one of them
recorded the SAME anonymous verdict — "the runtime disappeared without exiting
cleanly while this turn was running". ``runtime.log`` was demonstrably writable
(the successor fleet's lines land in it minutes later, so the file was right
there) and holds not one line from 19:41: no SIGTERM exit, no idle exit, no
jetsam kill. Nothing on disk could separate a stop sweep from a torn install
from a crash, so every later decision — and every agent's own trust in its
state — was made blind.

This module is the missing evidence. It is deliberately an INSTRUMENT and
nothing else: nothing here changes when a runtime exits, whether it stays
resident, how it is signalled, or what a stop does. Lifetime and residency stay
exactly where they are (``process._should_exit`` and ``docs/design-idle-reap.md``);
what changes is that a death can now be NAMED instead of inferred.

FOUR ARTIFACTS, ONE STORY
-------------------------
* **The turn journal** (``registry.turn_journal_path``, in the conversation
  directory) — the runtime's own statement that turn N was in flight, written
  at turn start and closed at turn end. An OPEN row is the positive evidence the
  existing taxonomy demanded and could not obtain: ``attention._classify_orphaned_run``
  could only reach "the record's pid is dead" or nothing at all, because a
  stopped process leaves no statement about its own turn.
* **The boot record** (``run/host/<pid>.json``) — "I existed, on this build,
  under this parent", published before the control socket listens, so a
  spawn that died at load and a fleet that vanished are distinguishable. It is
  the missing link in the chain ``app -> backend -> runtime -> runtimes``: every
  process in that chain could be taken away by a restart, an install or a
  sweep, and none of them could report the loss afterwards.
* **The exit record** (``sessions/<id>/runtime-exit.json``, added 2026-10-21) —
  "I am leaving NOW, for this reason, and nobody asked me to". The boot record's
  withdrawal says the exit ordering RAN; it cannot say why, so an idle policy
  exit and a SIGKILL were the same ``stored`` row in every listing. This is the
  durable half: the ``_clean_exit`` reason and the sentence the exit path
  already composed, with the run key that keeps an older departure from
  explaining a newer death. Read by ``info.collect`` for a STORED row's STOPPED
  age/reason and by ``rescue`` so a policy departure is never re-engaged.
* **The run record** (``sessions/<id>/runtime-run.json``) — the exit record's
  counterpart, written at boot and removed by that run's own clean exit. It is
  what lets a listing tell "the record on disk is the last thing that happened"
  from "an earlier run exited, and the one after it was killed"; without it a
  row borrowed its predecessor's departure (review round 1, MAJOR-1).

Why a row and not a log line: the questions asked afterwards are asked from a
DIFFERENT process, on a machine whose log may have rotated, and the answer has
to be readable without replaying anything. A row keyed by pid and turn sequence
is the smallest artifact that answers "was this pid working, on what, when it
went away".

NOTHING HERE MAY END A SESSION OR RERUN A TURN. The successor that reads these
artifacts narrates what it found; the design's rule is "a continuation errand on
the next engage, never an automatic re-run", because replaying a turn that
already executed tool calls duplicates its side effects.

Stdlib-only apart from ``local_operator.update``, which is itself stdlib-only
and imported lazily: this module sits on the runtime's boot path.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.helpers import retention_label
from local_operator.paths import config_dir
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import HOST_RUN_DIRNAME

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.update import BuildStamp

logger = logging.getLogger(__name__)

#: What a boot record says it is. The namespace is shared with the future
#: supervised session host (design-session-survival §4), which will publish its
#: own records here, so a reader has to be able to tell a runtime's boot from a
#: host's: a host may spawn and observe, a runtime may only be spawned.
BOOT_RECORD_KIND = "runtime-boot"

#: The signal a runtime's own exit path can have recorded in a row it did not get
#: to close. A row carrying one is the ESCALATED SWEEP — the target was asked to
#: leave, was draining, and was killed before its turn ended — which is the one
#: shape of sweep that names itself, because the runtime wrote the signal down
#: before the second signal arrived.
#:
#: ONE WRITER-SIDE NORMALIZER, because the two writers spell it differently and
#: the reader keys on the token: ``amain``'s shutdown block passes the signal's
#: own name (``SIGTERM``), while the signal drain reaches ``_clean_exit`` with a
#: sentence (``"leaving after SIGTERM"``). Matching the raw spellings would make
#: the escalated-sweep rung hold only by WHICH writer happened to run last —
#: and a mislabel there hands the operator ``runtime-killed``, whose sentence
#: ends "and no stop was asked for", for a death where a signal was recorded.
_SIGNAL_TOKEN = re.compile(
    r"\bSIG(?:HUP|INT|QUIT|ILL|TRAP|ABRT|BUS|FPE|KILL|USR1|SEGV|USR2|PIPE|ALRM|"
    r"TERM|CHLD|CONT|STOP|TSTP|TTIN|TTOU|URG|XCPU|XFSZ|VTALRM|PROF|WINCH|IO|PWR|SYS)\b"
)


def signal_exit_token(text: str) -> str:
    """The signal name inside an exit cause, or ``""`` when there is none.

    The single spelling both writers funnel through and the single one the
    reader matches, so "who wrote last" cannot decide whether a sweep is named.
    Deliberately matched on the SIGNAL NAMES rather than on a bare ``SIG``
    prefix: a future cause spelled ``SIGnalled by a peer`` would otherwise be
    read as a termination signal.
    """
    match = _SIGNAL_TOKEN.search(str(text or ""))
    return match.group(0) if match else ""


def _now() -> float:
    return time.time()


def _install_root() -> str:
    """The install this process is running from, as a comparable path string.

    ``sys.prefix`` rather than ``__file__``'s directory: the prefix is what
    ``lop-update`` replaces and what a reader compares across processes
    (``0.55.9@abc1234`` in one venv means nothing about another), while the
    module path of a legacy ``lop mobile`` shim points at whatever tree happens
    to be on the import path.
    """
    try:
        return str(Path(sys.prefix).resolve())
    except OSError:  # pragma: no cover - a path that cannot resolve is not fatal
        return str(sys.prefix)


def _build_fields(build: Any | None) -> dict[str, str]:
    """Serialize a ``BuildStamp``-shaped object into the row's ``build`` block.

    Duck-typed rather than typed as ``BuildStamp`` so this module does not
    import ``update`` at module scope, and tolerant because the value reaches
    here from a boot path that must not fail over an instrument.
    """
    version = str(getattr(build, "version", "") or "")
    source_ref = str(getattr(build, "source_ref", "") or "")
    if not version and not source_ref:
        return {}
    return {"version": version, "source_ref": source_ref}


def _build_stamp(fields: Any) -> "BuildStamp | None":
    """Rebuild a comparable stamp from a serialized ``build`` block.

    ``None`` when nothing usable was recorded — a row written by an older build,
    or one whose install could not be read. Callers treat that as "no build
    evidence" rather than as a comparison against an empty stamp, because an
    empty stamp compares equal to an install nobody could read.
    """
    if not isinstance(fields, dict):
        return None
    version = str(fields.get("version") or "")
    source_ref = str(fields.get("source_ref") or "")
    if not version and not source_ref:
        return None
    from local_operator.update import BuildStamp

    return BuildStamp(version=version, source_ref=source_ref)


def _build_label(fields: Any) -> str:
    stamp = _build_stamp(fields)
    return "<unknown>" if stamp is None else stamp.label()


# ---------------------------------------------------------------------------
# The boot record
# ---------------------------------------------------------------------------


def _parse_spawn_chain(value: Any) -> list[dict[str, Any]] | None:
    """The stored spawn chain, validated; ``None`` on anything malformed.

    Delegates to ``macos_disclaim``'s parser — the writer of the format — so a
    reader cannot disagree with a writer about what a chain is.
    """
    from local_operator.macos_disclaim import parse_spawn_chain

    return parse_spawn_chain(value)


@dataclass
class BootRecord:
    """One process's statement that it existed, published before it listens.

    Satisfies ``types.DiscoveryRecord`` (``pid``, ``heartbeat_at``, ``to_json``)
    so it is written through ``registry.publish`` — the ONE staged write at 0600
    under a 0700 directory — rather than a second copy of it that would be free
    to disagree about the mode.

    Deliberately NOT a liveness record. A boot record whose process is gone is
    the artifact doing its job, so nothing may read this as "a runtime is live
    here" (that question is ``registry.scan``'s, over ``run/mobile``). It is
    removed on a CLEAN exit and left behind by every other kind, which is what
    makes "a record with no reader" the shape of an unaccounted death.
    """

    pid: int
    session_id: str
    kind: str = BOOT_RECORD_KIND
    parent_pid: int = 0
    build_version: str = ""
    build_ref: str = ""
    install_root: str = ""
    cwd: str = ""
    #: The lineage this process was spawned into (``macos_disclaim``), recorded
    #: at boot because it is what makes a later unattributed signal readable:
    #: "the app at the root of this chain was gone when the signal landed".
    #: ``None`` on processes spawned without a chain — absent stays absent.
    spawn_chain: list[dict[str, Any]] | None = None
    started_at: float = field(default_factory=_now)
    heartbeat_at: float = field(default_factory=_now)

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "kind": self.kind,
            "session_id": self.session_id,
            "parent_pid": self.parent_pid,
            "build_version": self.build_version,
            "build_ref": self.build_ref,
            "install_root": self.install_root,
            "cwd": self.cwd,
            "started_at": self.started_at,
            "heartbeat_at": self.heartbeat_at,
            # Additive and omitted when absent (same rule as the old-record
            # compatibility the readers already honor): an older build reading
            # this row never sees the key, and this build reading an older row
            # gets ``None`` from ``from_json``.
            **({"spawn_chain": self.spawn_chain} if self.spawn_chain else {}),
        }

    @classmethod
    def from_json(cls, data: Any) -> "BootRecord | None":
        if not isinstance(data, dict):
            return None
        try:
            pid = int(data["pid"])
        except (KeyError, TypeError, ValueError):
            return None
        return cls(
            pid=pid,
            session_id=str(data.get("session_id") or ""),
            kind=str(data.get("kind") or ""),
            parent_pid=int(data.get("parent_pid") or 0),
            build_version=str(data.get("build_version") or ""),
            build_ref=str(data.get("build_ref") or ""),
            install_root=str(data.get("install_root") or ""),
            cwd=str(data.get("cwd") or ""),
            spawn_chain=_parse_spawn_chain(data.get("spawn_chain")),
            started_at=float(data.get("started_at") or 0.0),
            heartbeat_at=float(data.get("heartbeat_at") or 0.0),
        )

    def build_stamp(self) -> "BuildStamp | None":
        return _build_stamp({"version": self.build_version, "source_ref": self.build_ref})

    def build_label(self) -> str:
        return _build_label({"version": self.build_version, "source_ref": self.build_ref})


#: The boot record THIS process published, held in memory so the durable second
#: opinion can prove the row it reads back is its own.
#:
#: The identity check is the whole reason this exists rather than re-reading the
#: file: a nonce is not available, macOS offers no cheap process start time
#: (``psutil`` is deliberately not a dependency), and the pid alone is a
#: RECYCLABLE name — so a process that never managed to publish could otherwise
#: pick up a long-dead stranger's build for its own pid, and
#: ``update.classify_import_failure`` would name ``install-mid-update`` for a
#: genuine packaging bug, the false positive its docstring forbids.
_own_record: "BootRecord | None" = None


def write_boot_record(
    session_id: str,
    build: "BuildStamp | None",
    *,
    root: Path | None = None,
    pid: int | None = None,
    parent_pid: int | None = None,
    cwd: str | None = None,
    spawn_chain: list[dict[str, Any]] | None = None,
) -> Path:
    """Publish this process's boot record. Returns where it landed.

    Raises only when the write itself fails; the caller (``process.amain``)
    swallows that, because a runtime whose boot record cannot be written must
    still run its turns.
    """
    global _own_record
    record = BootRecord(
        pid=os.getpid() if pid is None else pid,
        session_id=session_id,
        parent_pid=os.getppid() if parent_pid is None else parent_pid,
        build_version=str(getattr(build, "version", "") or ""),
        build_ref=str(getattr(build, "source_ref", "") or ""),
        install_root=_install_root(),
        cwd=cwd if cwd is not None else os.getcwd(),
        spawn_chain=spawn_chain,
    )
    path = registry.publish(record, root, HOST_RUN_DIRNAME)
    if record.pid == os.getpid() and root is None:
        # Only the AMBIENT record is "ours": a caller writing into an explicit
        # root (a test, a tool) is describing some other process's namespace,
        # and treating that as this process's own would let
        # ``recorded_boot_build`` vouch for a stranger's row.
        _own_record = record
    return path


def clear_boot_record(pid: int | None = None, root: Path | None = None) -> None:
    """Withdraw this process's boot record on a CLEAN exit. Best-effort.

    The asymmetry with a death is the point rather than tidiness: a record that
    survives its process says "this pid stopped without exiting cleanly", which
    is exactly the fact tonight's incident could not establish. Withdrawing on
    the clean paths is also what keeps the namespace bounded — a record is left
    behind per unaccounted death, not per runtime ever spawned.
    """
    registry.unpublish(os.getpid() if pid is None else pid, root, HOST_RUN_DIRNAME)


def read_boot_record(pid: int, root: Path | None = None) -> "BootRecord | None":
    """One pid's boot record, or ``None`` when there is none to read.

    Deliberately does not require the pid to be alive: the interesting case is a
    record whose process is gone, which is the evidence the successor reads.
    """
    try:
        path = registry.record_path(pid, root, HOST_RUN_DIRNAME)
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return BootRecord.from_json(data)


def recorded_boot_build(pid: int | None = None, root: Path | None = None) -> "BuildStamp | None":
    """The build THIS process recorded for itself, or ``None``.

    The durable second opinion for ``update.classify_import_failure``: a process
    whose install is torn badly enough that the live stamp cannot be read still
    has its own boot record to say what it loaded.

    ONLY THIS PROCESS'S OWN RECORD ANSWERS, and that is a correctness rule
    rather than caution: the pid is a recyclable name and ``run/host`` outlives
    the processes that wrote it, so "the record on disk with my pid" can be
    somebody else's — long dead, from another build. Handing that build to the
    import classifier would name an install race for a genuine packaging bug,
    which is the one direction its own docstring forbids. The on-disk row is
    compared against the one this process published (``_own_record``), so a
    process that never published one gets ``None`` — no evidence, the legacy
    path — rather than a stranger's.
    """
    target = os.getpid() if pid is None else pid
    own = _own_record
    if own is None or own.pid != target:
        return None
    record = read_boot_record(target, root)
    if record is None or record.started_at != own.started_at or record.kind != own.kind:
        return None
    return record.build_stamp()


def prune_boot_records(root: Path | None = None, *, now: float | None = None) -> int:
    """Drop boot records whose process is gone. Returns how many were removed.

    ``run/host`` is the one namespace nothing else reaps. ``registry.scan``
    reaps ``run/mobile`` and ``_prune_reaped`` bounds only the ``reaped/``
    sidecar, so without this the directory grows one file per UNCLEAN death
    forever — and a growing directory is what makes the pid-recycling case above
    reachable at all.

    A LIVE pid's record is never touched, whatever its age: that is the record
    of a runtime still running, and deleting it would erase the "this pid
    existed, on this build" evidence the namespace exists for. Retention reuses
    ``registry``'s own numbers rather than inventing a second policy for the
    same kind of artifact — evidence is worth one look soon after the death, not
    indefinite storage.

    Called from the boot path, which is the one moment a new writer joins this
    namespace. Cheap by construction: one directory listing and one signal-0
    probe per record, on a directory that this function keeps small.

    THE ZOMBIE PROBE IS DELIBERATELY NOT PAID HERE, and the reason is that it
    cannot be made cheap for THIS namespace the way ``registry.classify`` makes
    it cheap for session records. That rule — ``age > HEARTBEAT_INTERVAL_S *
    1.5`` — needs a heartbeat that keeps moving; ``RecordPublisher`` re-stamps a
    ``SessionRecord`` every 15 s, while a ``BootRecord`` is a boot-time snapshot
    by design (see its docstring) and nothing ever refreshes its heartbeat. So
    every record of a runtime that has been up longer than 22.5 s is "quiet", and
    deriving the probe from that age here would spend the zombie probe (see
    ``procstate.is_zombie`` for its measured cost) once per LIVE runtime at every boot — ~100-200 ms
    for a forty-session fleet, on the path this design measured at ~1.2 s.

    The consequence is bounded and in the safe direction: a record whose pid has
    exited but has not been reaped yet reads as alive through signal-0 and is
    kept one cycle longer. That costs a file, not an answer — a zombie's record
    is still a true statement about a pid that booted on a build, and the next
    boot after its parent reaps it prunes it.

    A RECORD THAT CANNOT BE PARSED IS AGED OUT LIKE A DEAD ONE, on the file's own
    mtime, and this is what keeps the namespace from wedging the prune (review
    round 2, MAJOR 1). This is the only reaper ``run/host`` has, and the reader
    that decides what a prune must keep reports its answer INCOMPLETE while any
    entry in the namespace is unreadable (``update._boot_records_under``) — so a
    single non-record ``*.json`` here used to stop generation reclamation on every
    later ``lop update``, permanently and silently. The previous shape could not
    clear it in either direction: a payload the parser could not use raised OUT of
    this function (a wrong-typed field is a ``TypeError``, and the probe that
    found this measured the exception escaping ``_bind_boot_instrumentation`` into
    the BOOT of every session — the pre-existing #1175 defect this closes), and a
    payload that parses to ``None`` (``{}``, a JSON array) was ``continue``d and
    never aged out. Both are ``None`` here now, and both carry their mtime as the
    age the policy below is applied to, which is exactly what
    ``registry._prune_reaped`` does for the sidecar that has no readable pid
    either: evidence is worth one look soon after it lands, not indefinite
    storage. A torn record younger than the retention window is KEPT and logged —
    the age is what makes ageing a decision rather than a deletion of fresh
    evidence, and the log line is what keeps the condition from being silent.

    AND THE TWO BOUNDS AGREE, so no entry can be IMMORTAL BY MTIME (review round 3,
    MINOR 1). The age rule alone is ``age > REAPED_MAX_AGE_S``, which a future-dated
    mtime makes NEGATIVE — never expired — while the count rule below keeps the
    NEWEST entries, which is exactly where a future-dated file sorts. Measured with
    the two together: 250 dead records were pruned and the future-dated one
    survived both. A negative age is therefore STALE. The producers are restore,
    ``os.utime`` and clock skew rather than anything written here, which is why the
    line for that shape says the stamp is in the future instead of promising a
    retention window that will not apply to it.

    THE WARNING NAMES THE EXCEPTION TYPE, so a bug inside ``BootRecord.from_json``
    does not read exactly like a torn file (review round 3, MINOR 5);
    ``update._entries_in_directory`` has named it that way all along.
    """
    directory = (root or config_dir()) / HOST_RUN_DIRNAME
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return 0
    if not paths:
        return 0
    moment = _now() if now is None else now
    dead: list[tuple[float, Path]] = []
    for path in paths:
        failure = ""
        try:
            record = BootRecord.from_json(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001 — see the docstring: an entry costs itself
            record = None
            # THE TYPE IS KEPT AS A NAME, not the exception: ``exc`` is unbound once
            # its handler ends, and the warning below is outside it. This is the
            # half of review round 3's MINOR 5 that makes a bug inside
            # ``BootRecord.from_json`` stop reading exactly like a torn file.
            failure = type(exc).__name__
        if record is None:
            why = failure or "not a record of this kind"
            try:
                landed = path.stat().st_mtime
            except OSError:  # pragma: no cover — vanished under us, nothing to age
                continue
            if landed > moment:
                # A STAMP IN THE FUTURE IS NOT A KEEP (review round 3, MINOR 1): the
                # age below is negative, so the window would never expire it and the
                # count path keeps the NEWEST entries — a future-dated file sorts last
                # and survives both bounds, immortal by mtime. ``os.utime``, a restore
                # and clock skew are the plausible producers, and the writers here
                # never make one; the line says which shape it is rather than
                # promising a retention window that will not apply to it.
                logger.warning(
                    "boot record %s cannot be read (%s) and is stamped in the future: "
                    "ageing it out now, because no mtime may make an entry immortal",
                    path,
                    why,
                )
            else:
                logger.warning(
                    "boot record %s cannot be read (%s): keeping it for %s, then ageing "
                    "it out like a dead runtime's record (an unreadable entry keeps "
                    "every generation until it goes)",
                    path,
                    why,
                    retention_label(registry.REAPED_MAX_AGE_S),
                )
            dead.append((landed, path))
            continue
        if registry.pid_alive(record.pid):
            continue
        dead.append((record.started_at or path.stat().st_mtime, path))
    # NEWEST-FIRST EVIDENCE, by AGE rather than by filename. ``dead`` arrives in
    # directory-glob order (pid-as-string), which has nothing to do with when a
    # runtime died, so walking it directly made the count path evict whatever
    # sorted first — deleting the freshest death and keeping a 23-hour-old one,
    # the exact inversion ``registry._prune_reaped`` exists to prevent ("a burst
    # cannot evict today's evidence in favour of yesterday's"). The numbers are
    # reused from that function; this is the rule.
    dead.sort()
    removed = 0
    for started_at, path in dead:
        # A NEGATIVE AGE IS STALE, NOT FRESH (review round 3, MINOR 1). It could not be:
        # the count path keeps the NEWEST ``REAPED_MAX_FILES``, so an entry stamped ten
        # years ahead sorts last and is never evicted either — measured: 250 dead records
        # pruned and the future-dated one surviving. Together the two bounds therefore
        # left one shape nothing could ever reclaim, which is the wedge this function
        # exists to close for every other shape.
        age = moment - started_at
        aged_out = age > registry.REAPED_MAX_AGE_S or age < 0
        over_count = len(dead) - removed > registry.REAPED_MAX_FILES
        if not (aged_out or over_count):
            continue
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# The exit record
# ---------------------------------------------------------------------------

#: One runtime's statement about WHY its own exit path was reached, in the
#: conversation directory.
#:
#: THE GAP THIS CLOSES is the same one the boot record's asymmetry leaves open.
#: A boot record that survives its process says "this pid stopped without running
#: its own exit ordering" — which is a true and useful sentence, and also the
#: sentence a SIGKILL crash and an ordinary idle exit both produce when the exit
#: is the runtime's own POLICY: ``process._clean_exit`` withdrew the boot record
#: and logged one INFO line, so nothing durable distinguished "left because it was
#: idle for 3 s" from "was killed". ``lop sessions --all`` then showed both as
#: ``stored``, and the operator's question — when did it stop, and was it asked to
#: — had no answer anywhere in the store.
#:
#: WHY THE CONVERSATION DIRECTORY and not ``run/host``: this is evidence about a
#: SESSION's runtime, it must outlive the pid-keyed namespaces (a scan reaps and
#: an exit unpublishes those), and every reader that asks the question already
#: has the session directory in hand. It is also the shape ``registry``'s stop
#: marker uses, for the same durability reason.
#:
#: A NEW BOOT DELIBERATELY DOES NOT DELETE IT. The record is HISTORY — the last
#: departure of this session's runtime — and the listing shows it only for a row
#: with no live runtime, so the next run overwrites it at ITS exit and no reader
#: ever sees a live row carrying a stale stop. A session that has never exited
#: cleanly has no file, which is what keeps "unknown" distinguishable from "it
#: stopped at time T".
EXIT_RECORD_NAME = "runtime-exit.json"

#: What an exit record says it is. Namespaced like the boot record: this
#: directory also holds the transcript, roster and lease, so a reader must be
#: able to tell one artifact from another by its own contents.
EXIT_RECORD_KIND = "runtime-exit"

#: Schema version. A reader that does not know a version must not guess at its
#: fields — ``ExitRecord.from_json`` refuses an unknown one rather than
#: half-reading a future row.
EXIT_RECORD_VERSION = 1

#: How far an exit record's ``started_at`` may differ from a run key read from
#: another artifact, in seconds. Same allowance (and the same reasoning — the key
#: is written by two processes from the same clock, so this only absorbs
#: rounding) as ``attention._RUN_KEY_TOLERANCE_S``, restated here rather than
#: imported because this module sits on the runtime's boot path and must stay
#: stdlib-only apart from ``update``.
EXIT_RUN_KEY_TOLERANCE_S = 1.0


def exit_record_path(conversation_dir: Path) -> Path:
    """Where one conversation's exit record lives.

    A RESOLVER: it names the file and never creates the directory, on the same
    rule ``registry.record_path`` states — readers that only want to ask whether
    a record exists must not bring a session directory into being.
    """
    return Path(conversation_dir) / EXIT_RECORD_NAME


@dataclass
class ExitRecord:
    """One runtime's durable account of reaching its own exit path.

    ``reason`` is the exact string ``process._clean_exit`` was given, and
    ``detail`` is the human sentence that path already computed — carried here
    rather than re-derived, because a reader re-deriving it from the reason would
    be inventing a second vocabulary for a fact the runtime already knew.

    ``requested`` is the answer to "did a person ask for this": True only when a
    durable stop marker covers THIS run (pid + start time). A policy exit is
    False, and so is a signal nobody staged — which is what makes the field
    worth carrying at all: after 2026-09-30 an unattributed SIGTERM and a
    ``lop stop`` were indistinguishable in the store.

    ``pressure`` is context, not a cause: free disk bytes at the store's root and
    the host's 1-minute load when the runtime left. Both come from cheap,
    fork-free reads, and either is ``None`` when it could not be answered —
    never 0, because a zero would read as "the disk was full".
    """

    session_id: str
    pid: int
    started_at: float
    at: float
    reason: str
    detail: str = ""
    requested: bool = False
    build: str = ""
    free_disk_bytes: int | None = None
    load_1m: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "v": EXIT_RECORD_VERSION,
            "kind": EXIT_RECORD_KIND,
            "session_id": self.session_id,
            "pid": self.pid,
            "started_at": self.started_at,
            "at": self.at,
            "reason": self.reason,
            "detail": self.detail,
            "requested": self.requested,
            "build": self.build,
            # ALWAYS PRESENT, with nulls inside, even when nothing was measured:
            # a consumer that reads ``pressure`` must not have to branch on the
            # key's existence, and the two readings are a pair.
            "pressure": {"free_disk_bytes": self.free_disk_bytes, "load_1m": self.load_1m},
        }

    @classmethod
    def from_json(cls, data: Any) -> "ExitRecord | None":
        """A stored record, or ``None`` for anything a reader cannot trust.

        Tolerant by contract: this is read on a LISTING path and by the rescue
        pass, and a torn or future row means "no evidence", never an exception.
        A record with no ``reason`` is refused rather than read as an empty
        reason — the reason is the one field the file exists for.

        NON-FINITE AND NON-POSITIVE STAMPS ARE REFUSED, not tolerated (review
        round 1, minor): the listing renders an age from ``at``, and ``inf``
        renders as a fabricated ``20747d`` while a zero renders a departure before
        the epoch. ``int('inf')``-typed payloads raise ``OverflowError``, which is
        caught here — a numeric field a reader cannot use is the same answer as a
        missing one, and neither may escape as an exception on a listing path.
        """
        if not isinstance(data, dict):
            return None
        try:
            version = int(data.get("v") or 0)
        except (TypeError, ValueError, OverflowError):
            return None
        if version != EXIT_RECORD_VERSION:
            return None
        try:
            pid = int(data["pid"])
            started_at = float(data.get("started_at") or 0.0)
            at = float(data.get("at") or 0.0)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(started_at) or not math.isfinite(at):
            return None
        # A RUN HAS A POSITIVE PID AND A REAL DEPARTURE TIME; ``started_at`` may be
        # 0.0 only where the writer had no record to read it from, which
        # ``covers_run`` treats as "no run key" rather than as 1970.
        if pid <= 0 or at <= 0 or started_at < 0:
            return None
        reason = str(data.get("reason") or "")
        if not reason:
            return None
        pressure = data.get("pressure")
        free = load = None
        if isinstance(pressure, dict):
            raw_free = pressure.get("free_disk_bytes")
            raw_load = pressure.get("load_1m")
            try:
                free = None if raw_free is None else int(raw_free)
            except (TypeError, ValueError):
                free = None
            try:
                load = None if raw_load is None else float(raw_load)
            except (TypeError, ValueError):
                load = None
        return cls(
            session_id=str(data.get("session_id") or ""),
            pid=pid,
            started_at=started_at,
            at=at,
            reason=reason,
            detail=str(data.get("detail") or ""),
            # ``is True`` rather than ``bool(...)``: a payload carrying the STRING
            # "false" would otherwise read as a requested stop, which is the one
            # direction this field must never err in — it would narrate a crash as
            # the user's own act (the wrong-verdict class QA round 1 found in the
            # marker path). Anything that is not a JSON true is "not requested".
            requested=data.get("requested") is True,
            build=str(data.get("build") or ""),
            free_disk_bytes=free,
            load_1m=load,
        )

    def covers_run(self, pid: int, started_at: float) -> bool:
        """Whether this record attests to the run ``(pid, started_at)``.

        The same run-key rule every other durable artifact applies, and for the
        same reason: a session can exit cleanly and be resumed, and the record of
        that PREVIOUS departure must not explain a later, unexplained death.

        THE PID-ONLY BRANCH IS DELIBERATE AND ITS BOUND IS STATED (review round
        1, nit). When either side has no start stamp — a writer whose record
        could not be read (``0.0``), or a reader classifying a boot record with
        no start — the pid match is all the evidence there is. It is accepted
        rather than refused because the only caller that can produce that shape
        is a reduced host, and refusing would delete an explanation on no
        evidence (the wrong-verdict direction this module forbids elsewhere).
        The pid is recyclable, so the bound is: on such a host a same-pid
        stranger could in principle satisfy this check; every path that HAS a
        start stamp — every real runtime — is compared within
        ``EXIT_RUN_KEY_TOLERANCE_S``.
        """
        if int(pid) != self.pid:
            return False
        if not started_at:
            return True
        if not self.started_at:
            return True
        return abs(self.started_at - float(started_at)) < EXIT_RUN_KEY_TOLERANCE_S


def _exit_pressure() -> tuple[int | None, float | None]:
    """``(free disk bytes at the config root, 1-minute load)``, or nulls.

    CHEAP AND FORK-FREE BY CONSTRUCTION, because this runs on the exit path the
    reaper's own tests time to a 40 ms spread. ``os.statvfs`` and
    ``os.getloadavg`` are single syscalls; neither spawns anything, and the
    ``statvfs`` is spelled ``getattr`` because Windows has no such call — an
    unsupported reading is ``None`` there, which is the same answer as a store
    whose root cannot be stat'ed.
    """
    free: int | None = None
    load: float | None = None
    try:
        statvfs = getattr(os, "statvfs", None)
        if statvfs is not None:
            stats = statvfs(str(config_dir()))
            free = int(stats.f_bavail) * int(stats.f_frsize)
    except (OSError, ValueError, AttributeError):
        free = None
    try:
        load = float(os.getloadavg()[0])
    except (OSError, AttributeError):
        load = None
    return free, load


def write_exit_record(
    session_id: str,
    *,
    conversation_dir: Path | None,
    pid: int | None = None,
    started_at: float = 0.0,
    reason: str = "",
    detail: str = "",
    requested: bool = False,
    build: str = "",
) -> Path | None:
    """Publish this run's exit record. Returns where it landed, or ``None``.

    NEVER RAISES: every caller is an exit path, and a runtime that cannot write
    its own account must still leave. Absence of the file is the same answer as
    "no record" to every reader, so a failed write degrades to today's behaviour
    rather than to a wedged exit.

    The write is ``registry._staged_write`` — one temp file in the same
    directory, 0600, ``os.replace`` — so a concurrent listing sees the old row or
    the new one and never a torn one. The directory is NOT created: a runtime
    that never had a session directory has nothing to report about, and
    inventing one would make an empty session appear in every listing.

    ``pid``/``started_at`` identify the RUN, not the session: they are what
    later lets a reader (and the rescue pass) prove this record describes the
    death it is being read for. ``started_at`` unknown is 0.0, which
    ``ExitRecord.covers_run`` treats as "no run key", never as a match for 1970.
    """
    if conversation_dir is None or not str(reason):
        return None
    directory = Path(conversation_dir)
    try:
        if not directory.is_dir():
            return None
        free, load = _exit_pressure()
        record = ExitRecord(
            session_id=str(session_id),
            pid=os.getpid() if pid is None else int(pid),
            started_at=float(started_at or 0.0),
            at=time.time(),
            reason=str(reason),
            detail=str(detail),
            requested=bool(requested),
            build=str(build),
            free_disk_bytes=free,
            load_1m=load,
        )
        target = exit_record_path(directory)
        registry._staged_write(target, record.to_json(), prefix=f".{EXIT_RECORD_NAME}.")
        return target
    except Exception:  # noqa: BLE001 — an exit path never fails over an instrument
        logger.debug("session runtime: could not write its exit record", exc_info=True)
        return None


def read_exit_record(conversation_dir: Path | None) -> "ExitRecord | None":
    """One conversation's last exit record, or ``None`` when there is none.

    ONE SMALL READ, and that bound is the contract the listing relies on: the
    stored half of ``lop sessions --all`` calls this once per row it actually
    shows, never over the whole store. A malformed file is ``None`` (see
    ``ExitRecord.from_json``), so an unreadable artifact degrades to "unknown"
    rather than to a failed listing.

    THE DIRECTORY IS THE AUTHORITY ON WHICH SESSION THIS IS (review round 1,
    nit): a record whose own ``session_id`` disagrees with the directory it sits
    in describes some other conversation — a hand-copied file, a torn write from
    a tool that renamed a directory — and its time must not be attached to this
    row. A mismatch is refused as no evidence.
    """
    if conversation_dir is None:
        return None
    directory = Path(conversation_dir)
    try:
        data = json.loads(exit_record_path(directory).read_text())
    except (OSError, ValueError):
        return None
    record = ExitRecord.from_json(data)
    if record is None:
        return None
    if record.session_id and record.session_id != directory.name:
        return None
    return record


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------

#: WHICH RUN THIS SESSION'S LAST DEPARTURE BELONGS TO — written at boot, removed
#: by that run's own clean exit.
#:
#: WHY A SECOND ARTIFACT, when the exit record already carries ``pid`` and
#: ``started_at``: a stored row has to be able to tell "the exit record on disk
#: IS the last thing that happened here" from "an earlier run exited cleanly and
#: the one after it was killed". The listing has no record of the dead run at all
#: — its record file was reaped — so without a statement about the LATEST run it
#: would borrow the previous departure's time and reason, which is the
#: wrong-history defect review round 1 measured: idle-exit run A, resume, SIGKILL
#: run B, and the row still reports A's ``idle-exit`` at A's time.
#:
#: The boot record's asymmetry is the same shape, one namespace over, but
#: ``run/host`` is pid-keyed and PRUNED — a later boot removes a dead pid's
#: record, so it cannot answer a question asked days later. This one lives in the
#: conversation directory, is overwritten by the next boot (so it always names
#: the latest run), and is removed only by that run's own clean exit.
#:
#: ABSENT IS "NO RUN TO COMPARE" and is what a session that exited cleanly looks
#: like; PRESENT with a key that differs from the exit record's is a death nobody
#: has an exit record for, and the listing reads it as unknown.
RUN_RECORD_NAME = "runtime-run.json"
RUN_RECORD_KIND = "runtime-run"
RUN_RECORD_VERSION = 1


def run_record_path(conversation_dir: Path) -> Path:
    """Where one conversation's run record lives. A resolver: never creates."""
    return Path(conversation_dir) / RUN_RECORD_NAME


@dataclass
class RunRecord:
    """The latest run of this session: pid, start, and when it was announced.

    Deliberately smaller than a boot record: it answers exactly one question
    ("which run is the newest here") and is compared, not narrated.
    """

    session_id: str
    pid: int
    started_at: float
    at: float = field(default_factory=_now)

    def to_json(self) -> dict[str, Any]:
        return {
            "v": RUN_RECORD_VERSION,
            "kind": RUN_RECORD_KIND,
            "session_id": self.session_id,
            "pid": self.pid,
            "started_at": self.started_at,
            "at": self.at,
        }

    @classmethod
    def from_json(cls, data: Any) -> "RunRecord | None":
        """A stored row, or ``None`` for anything that cannot identify a run."""
        if not isinstance(data, dict):
            return None
        try:
            if int(data.get("v") or 0) != RUN_RECORD_VERSION:
                return None
            pid = int(data["pid"])
            started_at = float(data.get("started_at") or 0.0)
            at = float(data.get("at") or 0.0)
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(started_at) or not math.isfinite(at):
            return None
        if pid <= 0 or started_at < 0:
            return None
        return cls(
            session_id=str(data.get("session_id") or ""),
            pid=pid,
            started_at=started_at,
            at=at,
        )

    def covers_run(self, pid: int, started_at: float) -> bool:
        """Whether this row names the run ``(pid, started_at)``.

        The pid-only branch is the same documented bound ``ExitRecord.covers_run``
        carries: a missing start stamp on either side leaves the pid as the only
        evidence, and refusing there would delete an explanation rather than
        protect one.
        """
        if int(pid) != self.pid:
            return False
        if not started_at or not self.started_at:
            return True
        return abs(self.started_at - float(started_at)) < EXIT_RUN_KEY_TOLERANCE_S


def write_run_record(
    session_id: str,
    *,
    conversation_dir: Path | None,
    pid: int | None = None,
    started_at: float = 0.0,
) -> Path | None:
    """Publish this run's run record at boot. Returns where it landed, or ``None``.

    NEVER RAISES, on the same rule as the boot record: a runtime that cannot
    write its own account must still run. The write is ``registry._staged_write``
    so a concurrent listing sees the old row or the new one, and the directory is
    not created (a runtime with no session directory has nothing to report).
    """
    if conversation_dir is None:
        return None
    directory = Path(conversation_dir)
    try:
        if not directory.is_dir():
            return None
        record = RunRecord(
            session_id=str(session_id),
            pid=os.getpid() if pid is None else int(pid),
            started_at=float(started_at or 0.0),
        )
        target = run_record_path(directory)
        registry._staged_write(target, record.to_json(), prefix=f".{RUN_RECORD_NAME}.")
        return target
    except Exception:  # noqa: BLE001 — boot must not fail over an instrument
        logger.debug("session runtime: could not write its run record", exc_info=True)
        return None


def read_run_record(conversation_dir: Path | None) -> "RunRecord | None":
    """The latest run this session announced, or ``None`` when none is on disk.

    Tolerant and one small read, exactly like ``read_exit_record``. The
    directory's name is the authority on the session, so a row naming another
    conversation is refused.
    """
    if conversation_dir is None:
        return None
    directory = Path(conversation_dir)
    try:
        data = json.loads(run_record_path(directory).read_text())
    except (OSError, ValueError):
        return None
    record = RunRecord.from_json(data)
    if record is None:
        return None
    if record.session_id and record.session_id != directory.name:
        return None
    return record


def clear_run_record(
    conversation_dir: Path | None, *, pid: int | None = None, started_at: float = 0.0
) -> None:
    """Remove THIS run's run record on a clean exit. Best-effort.

    ONLY ITS OWN, compared by run key: this is called before the dispose releases
    the lease, so a successor cannot normally exist yet, but a reduced host
    without a lease could race one — and deleting a successor's marker would make
    its eventual death read as a clean exit.

    Unlink rather than a write: an absent file is the state a clean exit means,
    and it leaves nothing for a later reader to misread as a live run.
    """
    if conversation_dir is None:
        return
    directory = Path(conversation_dir)
    try:
        record = read_run_record(directory)
        if record is None:
            return
        if not record.covers_run(os.getpid() if pid is None else int(pid), started_at):
            return
        run_record_path(directory).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# The turn journal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnJournalRow:
    """One runtime's statement about one turn, as read back from disk."""

    session_id: str
    pid: int
    parent_pid: int
    turn_seq: int
    command_id: str
    started_at: float
    ended_at: float | None
    open: bool
    end_cause: str
    exit_cause: str
    still_open_at_exit: bool
    last_boundary: str
    build: dict[str, Any]
    install_root: str
    updated_at: float

    @classmethod
    def from_json(cls, data: Any) -> "TurnJournalRow | None":
        """Parse a row, or ``None`` when the payload is not one.

        Tolerance is the contract, not a convenience: this runs at session boot
        on a file a killed process may have been midway through replacing, and a
        malformed row means "no evidence" — never an exception that would stop a
        session from opening.
        """
        if not isinstance(data, dict):
            return None
        try:
            pid = int(data["pid"])
            turn_seq = int(data.get("turn_seq") or 0)
            started_at = float(data.get("started_at") or 0.0)
        except (KeyError, TypeError, ValueError):
            return None
        ended_raw = data.get("ended_at")
        try:
            ended_at = None if ended_raw is None else float(ended_raw)
        except (TypeError, ValueError):
            ended_at = None
        # ``open`` is stored explicitly AND derived from ``ended_at``: a writer
        # that stored only one of them could be read two ways by two readers,
        # and this row's whole value is that one reading of it is the truth.
        is_open = bool(data.get("open", ended_at is None)) and ended_at is None
        return cls(
            session_id=str(data.get("session_id") or ""),
            pid=pid,
            parent_pid=int(data.get("parent_pid") or 0),
            turn_seq=turn_seq,
            command_id=str(data.get("command_id") or ""),
            started_at=started_at,
            ended_at=ended_at,
            open=is_open,
            end_cause=str(data.get("end_cause") or ""),
            exit_cause=str(data.get("exit_cause") or ""),
            still_open_at_exit=bool(data.get("still_open_at_exit") or False),
            last_boundary=str(data.get("last_boundary") or ""),
            build=dict(data.get("build") or {}),
            install_root=str(data.get("install_root") or ""),
            updated_at=float(data.get("updated_at") or 0.0),
        )

    def build_stamp(self) -> "BuildStamp | None":
        return _build_stamp(self.build)

    def build_label(self) -> str:
        return _build_label(self.build)

    def turn_label(self) -> str:
        return f"turn {self.turn_seq}" if self.turn_seq else "a turn"


@dataclass(frozen=True)
class TurnSnapshot:
    """What the runtime hands the journal at turn start.

    A value rather than four arguments so the writer's call sites cannot drift
    from each other about which turn the row is about, and so a test can build
    the same statement the runtime does.
    """

    command_id: str = ""


class TurnJournal:
    """The writer side: one per runtime process, owned by the handle it serves.

    THE FAILURE CONTRACT IS "NEVER RAISE". Every method swallows an OSError and
    logs it. The alternative — a turn that fails because an instrument could not
    write to disk — would put the evidence ahead of the work it is evidence
    about, and would do it on the boot path of every session on the machine.
    Degrading to "no evidence" is the same direction the whole taxonomy already
    fails in: ``CUT_OFF_UNKNOWN`` is an answer, and an instrument that removed
    turns is not.
    """

    def __init__(
        self,
        conversation_dir: Path,
        session_id: str,
        build: "BuildStamp | None" = None,
        *,
        parent_pid: int | None = None,
        install_root: str | None = None,
        pid: int | None = None,
    ) -> None:
        self.conversation_dir = conversation_dir
        self.session_id = session_id
        self.build = build
        self.pid = os.getpid() if pid is None else pid
        self.parent_pid = os.getppid() if parent_pid is None else parent_pid
        self.install_root = _install_root() if install_root is None else install_root
        self._row: dict[str, Any] | None = None
        #: The runtime's OWN turn sequence, not the harness's generation: the
        #: row is read by a process that has no access to this one's event
        #: stream, so the number has to mean "the Nth turn THIS pid ran".
        self._turn_seq = 0

    # -- writing ------------------------------------------------------------

    def _write(self, what: str) -> None:
        row = self._row
        if row is None:
            return
        row["updated_at"] = _now()
        try:
            registry.write_turn_journal(self.conversation_dir, row)
        except OSError:
            # Deliberately a warning, not a debug: a failure here is silent
            # evidence loss for the NEXT incident, and the log is the only place
            # it can surface.
            logger.warning(
                "turn journal: could not record the %s for %s",
                what,
                self.session_id,
                exc_info=True,
            )

    def open_turn(self, *, command_id: str = "") -> None:
        """Open the row for a turn that is STARTING.

        Written BEFORE the turn does any of its work, because the row's value is
        the window it covers: a row written lazily could miss exactly the turn
        that was killed.
        """
        self._turn_seq += 1
        self._row = {
            "session_id": self.session_id,
            "pid": self.pid,
            "parent_pid": self.parent_pid,
            "turn_seq": self._turn_seq,
            "command_id": command_id,
            "started_at": _now(),
            "ended_at": None,
            "open": True,
            "end_cause": "",
            "exit_cause": "",
            "still_open_at_exit": False,
            "last_boundary": "",
            "build": _build_fields(self.build),
            "install_root": self.install_root,
            "updated_at": _now(),
        }
        self._write("turn start")

    def close_turn(self, end_cause: str) -> None:
        """Close the open row. A no-op when no turn is open.

        Every terminal path of a turn reaches this (completed, aborted, errored,
        cut off), which is what makes "the row is still open" mean exactly one
        thing to the successor: the turn never ended.
        """
        row = self._row
        if row is None or not row.get("open"):
            return
        row["ended_at"] = _now()
        row["open"] = False
        row["end_cause"] = str(end_cause or "")
        self._write("turn end")

    def note_boundary(self, tool_name: str) -> None:
        """Record the last tool boundary a turn reached, while it is running.

        The point of a live update rather than a closing one: a turn that is
        killed never reaches its close, and "last completed boundary B" is what
        tells the successor's agent which of its remembered steps actually
        happened. Cheap enough to write per tool result — a tool call costs
        orders of magnitude more than this row does.
        """
        row = self._row
        if row is None or not row.get("open") or not tool_name:
            return
        if row.get("last_boundary") == tool_name:
            return
        row["last_boundary"] = str(tool_name)
        self._write("turn boundary")

    def note_exit(self, cause: str) -> None:
        """Record WHY this process is leaving, when it leaves with work open.

        Called by the runtime's own exit paths before they dispose. A row it
        writes here is the ESCALATED case: the runtime decided to leave, the
        turn had not finished, and (if this row is what a successor finds) the
        process was killed before it could finish. That is how a stop sweep that
        arrived as a SIGTERM names itself, where a SIGKILL-class sweep still
        leaves nothing — which is the honest limit of what a target can attest
        to about its own death.
        """
        row = self._row
        if row is None or not row.get("open"):
            return
        # ONE VOCABULARY, whatever the writer was handed (see
        # ``signal_exit_token``): the row is a machine artifact and
        # ``death_verdict`` keys on the token, so a sentence from the drain and a
        # bare name from the shutdown block must land identically.
        token = signal_exit_token(cause) or str(cause or "")
        # A SIGNAL IS STICKY. The two writers are two rungs of one exit, and
        # which of them runs last is an ordering accident: a later writer with
        # no signal to report must not unname a signal that was recorded, or an
        # escalated sweep would be reported as a crash whose sentence claims
        # no stop was asked for.
        if row.get("exit_cause") and signal_exit_token(str(row["exit_cause"])):
            row["still_open_at_exit"] = True
            self._write("exit")
            return
        row["exit_cause"] = token
        row["still_open_at_exit"] = True
        self._write("exit")


def journal_token(row: TurnJournalRow) -> str:
    """The dedupe token for one interrupted run.

    Derived from the run's identity rather than minted, so every boot of every
    successor computes the SAME token and the notice is narrated once
    (``Session._journal_cut_off_once`` dedupes by scanning the transcript for
    it). A random token would narrate the interruption on every open, which is
    the one way a notice about lost work can itself become noise.
    """
    return f"turn-journal:{row.session_id}:{row.pid}:{row.turn_seq}"


def row_detail(row: TurnJournalRow, *, lead: str = "") -> str:
    """The parenthetical riding with a death's reason sentence.

    Built here rather than at each call site so the live notice and the durable
    outcome cannot describe the same death differently. ONE parenthetical, never
    nested: ``incidents.render_cut_off_reason`` prints this inside a sentence
    that a one-line tooltip truncates, so ``lead`` is a comma-separated clause
    (``"SIGTERM received"``) rather than a second set of brackets — the first
    draft nested them and read as ``(SIGTERM received), (turn 4 in flight …)``.
    """
    parts = [f"{row.turn_label()} in flight on {row.build_label()}"]
    if row.last_boundary:
        parts.append(f"last boundary {row.last_boundary}")
    if row.install_root:
        # The field that separates a uv-tool install from a dev checkout's venv,
        # and the one an investigation reads first when two hosts disagree about
        # what a build was.
        parts.append(f"install {row.install_root}")
    parts.append(f"pid {row.pid}")
    if lead:
        parts.insert(0, lead)
    return f" ({', '.join(parts)})"


def open_row_after_death(conversation_dir: Path) -> TurnJournalRow | None:
    """The row that says a turn was in flight when its runtime went away.

    ``None`` unless all three hold: a row exists, it is OPEN, and the pid that
    wrote it is gone. The liveness test is what keeps this honest in both
    directions — a live runtime with a turn in flight has an open row too, and
    reporting that as a death would be the taxonomy's worst error in the
    opposite direction. ``check_zombie`` because a zombie is not alive: a
    ``kill -9``'d runtime whose parent has not reaped it still answers signal-0.
    """
    row = TurnJournalRow.from_json(registry.read_turn_journal(conversation_dir))
    if row is None or not row.open:
        return None
    if row.pid == os.getpid():
        # Our own row, i.e. a pid-recycled writer or an in-process reader. Never
        # a predecessor's death.
        return None
    if registry.pid_alive(row.pid, check_zombie=True):
        return None
    return row


def install_moved(row: TurnJournalRow) -> bool:
    """Did the install move away from this row's build WHILE its runtime was alive?

    The one comparison that separates an INSTALL-WINDOW TEAR from a plain death,
    and it is the same comparison ``update.classify_import_failure`` makes for a
    live process — with the bound that function does not need and this reader
    does.

    WHY THE BOUND IS LOAD-BEARING. A "differs" comparison alone answers the
    wrong question, because the reader runs LATER: a successor process reading a
    row hours after the death is running whatever install is on disk NOW, and on
    this host a release can land minutes after a sweep. The 19:41 fleet died
    while 0.55.10 was being published, so the unbounded comparison attributed
    that sweep to "a local-operator install was being replaced on disk while
    this turn was running" — a named-but-wrong cause on the artifact this PR
    exists to make trustworthy, and one that MASKS the sweep pattern it exists
    to reveal.

    THE BOUND IS THE ROW'S OWN LAST WRITE. ``updated_at``/``started_at`` are
    written by the runtime itself at turn start and at every tool boundary, so
    they are the last moment this runtime is KNOWN to have been alive; the
    install's own date comes from ``update.build_marker_age_s`` (the newest of
    ``.lop-source`` and the running dist-info). The tear is claimed only when
    the install had ALREADY moved no later than that instant, i.e. when the
    runtime is demonstrably alive after the tree under it was rewritten — which
    is exactly the ordering the docstring claims, for the provable subset.

    The conservative direction is deliberate: an install that moved AFTER the
    runtime's last known write leaves it unknowable whether the runtime was
    already dead, so the row falls through to ``runtime-killed`` (a crash) rather
    than claiming a tear it cannot prove. ``False`` on any missing half — an
    unreadable on-disk stamp, an absent install date, a row with no timestamp —
    for the same reason: naming a tear on no evidence would mislabel every death
    on a machine whose install metadata is absent (a dev checkout).

    The prefix is read through the SAME seam as the writer
    (``buildwatch.build_prefix``), because both sides of this comparison have to
    mean the same tree: the row's build is stamped through it at boot, so
    comparing against an unprefixed ``sys.prefix`` would label every death a
    tear on any host where the e2e seam is set (and would fail the e2e cell
    itself for a developer who has ``LOP_BUILD_PREFIX`` exported).
    """
    recorded = row.build_stamp()
    if recorded is None:
        return False
    try:
        from local_operator import buildwatch
        from local_operator.update import build_marker_age_s, installed_build

        prefix = buildwatch.build_prefix()
        if installed_build(prefix) == recorded:
            return False
        age = build_marker_age_s(prefix)
    except Exception:  # noqa: BLE001 — an unreadable stamp is not evidence
        return False
    if age is None:
        # No install date is no evidence: the comparison degrades to the
        # question this reader must not answer unbound.
        return False
    alive_until = row.updated_at or row.started_at
    if not alive_until:
        return False
    return (_now() - age) <= alive_until


#: The clause carried by every verdict for a turn whose runtime SURVIVED its own
#: stall bound before dying of something else — the third state, in the one place a
#: person reads it.
#:
#: WHY IT IS A LEAD RATHER THAN A CAUSE TOKEN. The death this rides with is whatever
#: killed the runtime AFTERWARDS (a signal, a recorded token, or nothing at all), and
#: naming the bound as its cause is the inversion this exists to prevent: a bound that
#: dumped and held is EVIDENCE the runtime was stalled, never evidence about what
#: ended it. It is the same shape as the other attribution leads on this function
#: (``SIGTERM received``, ``unattributed``) because it is the same kind of fact —
#: something established about the death that is not its name.
HELD_BOUND_LEAD = (
    "its own stall bound fired earlier, dumped every thread beside its log and did NOT "
    "end this runtime"
)

#: The clause for a fire the ARTIFACT DOES NOT ACCOUNT FOR: it fired and dumped every
#: thread, and nothing in the file says whether the runtime survived it.
#:
#: WHY THIS EXISTS RATHER THAN SILENCE (finding A, cell "absence of a post-fire marker is
#: not proof"). The old code turned that silence into the loudest available claim — "the
#: bound ended it". The opposite silence is no better: a reader looking at a successor's
#: row that was in fact preceded by a bound expiry would never learn to open the dump.
#: So the fire is named and the missing fact is named with it, which is what makes the
#: row honest on an artifact written between the dump-only change and the policy marker.
STALL_BOUND_LEAD = (
    "its own stall bound fired earlier and dumped every thread beside its log; the "
    "artifact does not say whether that fire ended this runtime"
)

#: The clause for a fire the artifact places INSIDE the asyncio runner's OWN teardown
#: (:func:`stall_watchdog.fired_in_runner_teardown` returned ``True``), i.e. the fire the
#: annotation exists for: the phase where task cancellation, async-generator shutdown and
#: the default executor's join run, and where the loop can sit for the whole
#: ``THREAD_JOIN_TIMEOUT`` while nothing about it is a turn that stopped reporting.
#:
#: A LEAD FOR THE SAME REASON AS ITS SIBLINGS: it says what the runtime was DOING at the
#: fire — the fact that sends a reader to the right part of the dump — and never what
#: ended it. It is written in the subjunctive about the join because that is all the
#: annotation's ordering proves (see :data:`stall_watchdog.TEARDOWN_MARKER`).
TEARDOWN_LEAD = (
    "its asyncio runner had begun its own teardown when its own stall bound fired "
    "(the executor join may follow)"
)

#: What :func:`_stall_bound_evidence` answers when the artifact cannot be read at all.
#:
#: THE EMPTY TOKEN, so it is never equal to one of
#: :mod:`stall_watchdog`'s outcomes: the unreadable path must not borrow the vocabulary
#: of an answer it did not make, and every reader below compares by equality.
_STALL_EVIDENCE_UNREADABLE = ""


def _stall_bound_evidence(
    row: TurnJournalRow,
) -> tuple[str | None, tuple[str, ...], str, bool]:
    """``(which leg fired, which tickers died, how the fire ended, teardown-ordered)``.

    ``(None, (), _STALL_EVIDENCE_UNREADABLE, False)`` when this turn left no dump of its
    own — the empty token, so that the unreadable path cannot borrow the vocabulary of
    an answer it did not make (see the constant).

    FOUR FACTS, ONE READ OF ONE FILE, deliberately: the narration consumes them
    together (agent review round 1, MINOR 3), because a bare
    SILENCE leg must not be reported as "the loop went silent" while the same
    artifact says the plane's own reporter was gone. Two separate readers would be
    two chances for the pair to disagree about which file they came from, which is
    the same class of error the fence below guards in the time direction.

    THE THIRD FACT REPLACED A BOOL, and the substitution is finding A of the
    2026-09-23 convergence round: ``held_fire``'s old answer was "the runtime survived
    it", which meant "the artifact carries a marker only a live Python thread could
    append" — and reading its ABSENCE as "so the bound ended this runtime" is false on
    every dump-only build. :func:`stall_watchdog.fire_outcome` answers from what the
    header STATES instead, and its third answer (:data:`stall_watchdog.FIRE_UNKNOWN`)
    is what keeps an artifact that proves nothing from becoming a death claim.

    THE FOURTH IS ORDERING-ONLY too, for the class of fires taken inside
    ``asyncio``'s own shutdown (:func:`stall_watchdog.fired_in_runner_teardown`, which
    returns ``None`` unless the teardown annotation stands above the fire).

    THE FILE IS SEARCHED FOR, NOT COMPUTED, and the reason is measured rather than
    defensive. ``dump_path`` resolves its directory from ``paths.log_dir()``, which
    honours ``LOCAL_OPERATOR_CONFIG_DIR`` PER PROCESS — and on this fleet (measured
    2026-09-22) the session runtimes are started with that variable set, to the
    DEFAULT config directory, while the ``lop serve`` daemon that classifies their
    deaths is not (``ps eww``: pid 92815 carries it, pid 1276 does not). A reader
    that asked for one path therefore opened a directory the dump had never been
    written into, and answered "this turn left no dump of its own" about a runtime
    that had dumped every thread: pid 96510 on 0.62.4, whose own cut-off card read
    ``unattributed`` while its fired dump sat in ``~/.local-operator/logs`` — the
    exact mis-narration this rung exists to end, reached by a path the rung's own
    author could not see because the test wrote and read in one store.
    ``stall_watchdog.dump_evidence`` searches every directory a writer on this machine
    can have used and answers with the file that FIRED, so the fence below is applied
    to the artifact this death actually left.

    THE DUMP IS THE EVIDENCE, AND BOTH ENDS OF ITS LIFE ARE READ OFF IT (review
    round 1, MAJOR-2). A file last written before this turn began belongs to a
    predecessor that happened to hold the same pid — the mtime is the LOWER bound on
    its life. The header's own arm epoch (``armed for <bound>s at <epoch>``,
    :func:`stall_watchdog.armed_at`) is the UPPER one: ``arm`` runs in the child's
    entry point BEFORE anything can write a journal row, so a dump whose arm is later
    than this row's turn was written by a LATER PROCESS that drew the same pid — the
    case the mtime alone cannot see, because a successor's dump is newer than an
    earlier life's row by construction. A dump whose header names no readable epoch is
    not evidence here: every artifact this fleet holds spells it (178 of 178, measured
    2026-09-22), and answering a death from a file whose life cannot be placed is the
    mis-narration this whole rung exists to end.

    WHAT THE FENCE STILL CANNOT DISTINGUISH, stated rather than implied:

    * **an erased artifact**, unchanged from before — ``arm`` opens the dump with
      ``"w"``, so the next runtime to draw a recycled pid truncates the evidence away
      and that death falls to ``runtime-killed``/``unattributed``. Closing it means a
      per-run filename, and the arm site has no session identity to key one on (see
      ``process._live_handle``'s comment for the same constraint from the other
      side);
    * **an mtime rewritten by something other than the fire** — a backup restore, a
      copy, a ``touch``. The epoch check narrows it (a restored file still carries its
      own arm epoch, so it is placed by that rather than by its mtime), but a file
      whose mtime was moved into this turn's window AND whose arm epoch falls inside
      it is indistinguishable from this turn's own evidence.

    THE ARM EPOCH USED TO BE FOR THE READER'S OWN EYES (agent review round 1, MINOR
    3, which named the cross-check and left it by hand) and it is now APPLIED: see the
    two-ends paragraph above, which is where the recycling case and the residual it
    leaves are stated. What that paragraph keeps is this one's rule — the fence never
    invents evidence to close a gap.

    NEVER RAISES: this runs while a session is opening, on a file a killed
    process may have been midway through, and an unreadable instrument must
    degrade to the rungs below rather than stop the reader.
    """
    try:
        from local_operator.session.runtime import stall_watchdog

        evidence = stall_watchdog.dump_evidence(row.pid)
        if evidence is None:
            return None, (), _STALL_EVIDENCE_UNREADABLE, False
        path, text = evidence
        if not stall_watchdog.dump_is_current(path, row.started_at):
            return None, (), _STALL_EVIDENCE_UNREADABLE, False
        armed = stall_watchdog.armed_at(text)
        if armed is None or armed > row.started_at:
            return None, (), _STALL_EVIDENCE_UNREADABLE, False
        # THE FOUR FACT-READERS ARE PINNED TO THE ARTIFACT THE FENCE PLACED. Handed no
        # directory they search the stores themselves, and a second search is a second
        # choice: the narration would then be describing one file while the fence above
        # described another, which is the disagreement the one-read paragraph above
        # exists to prevent — and it is reachable, because the two searches see a
        # different world once a write lands between them. The chosen store is the
        # whole answer to the alignment question, so it is passed rather than left
        # implicit.
        store = path.parent
        return (
            stall_watchdog.fired_leg(row.pid, store),
            stall_watchdog.tick_deaths(row.pid, store),
            stall_watchdog.fire_outcome(row.pid, store),
            stall_watchdog.fired_in_runner_teardown(row.pid, store) is True,
        )
    except Exception:  # noqa: BLE001 — an unreadable dump is not a dead session
        logger.debug("stall dump unreadable for pid %s", row.pid, exc_info=True)
        return None, (), _STALL_EVIDENCE_UNREADABLE, False


def _stall_bound_detail(leg: str, tick_deaths: tuple[str, ...] = ()) -> str:
    """The clause that tells the bound's two legs apart on every surface.

    The CLASS is deliberately one token for both (see
    ``incidents.STALL_BOUND_CAUSE``), so this is the only place a reader learns
    whether a runtime went silent or spun without advancing — and the two want
    different investigations. The peer's case of 2026-09-21 was the SILENCE leg
    over a thread parked in ``queue.get``: a wait, not a wedge, which is a design
    decision this bound tolerates right up to the point where nothing has
    reported for the whole bound.

    A SILENCE LEG OVER A DUMP THAT NAMES A DEAD TICKER IS A THIRD READING, and it
    is the one the operator actually needs (agent review round 1, MINOR 3). The
    leg is a PREDICATE — "no plane reported for the whole bound" — and it stays
    one, because a dead ticker does not fire anything; it is the REASON the plane
    had nothing left to report with, and the two send a reader to opposite
    places. Without this the automated verdict kept calling a runtime whose
    reporter had died a runtime whose loop had gone quiet, which is the exact
    false attribution the tick-death record exists to end — and it survived in
    the one surface that is not a human reading the file.
    """
    from local_operator.session.runtime.stall_watchdog import LEG_PROGRESS, WORKLOAD

    if leg == LEG_PROGRESS:
        return (
            "its own stall bound fired: the loops kept running while the work "
            "stopped advancing, and every thread's stack is in its dump"
        )
    if WORKLOAD in tick_deaths:
        return (
            "its own stall bound fired: its workload tick had already died (recorded in "
            "its dump), so the silence is the REPORTER's rather than the loop's, and every "
            "thread's stack is in its dump"
        )
    return (
        "its own stall bound fired: no plane reported for the whole bound, and "
        "every thread's stack is in its dump"
    )


def death_verdict(row: TurnJournalRow) -> tuple[str, str, str]:
    """``(kind, cause, reason)`` for a runtime that left an open row behind.

    THE PREFERENCE ORDER IS THE FIX, and every rung is a NAMED cause:

    0. **the runtime's own stall bound ended it** → ``runtime-stall-bound``. ONLY A
       LEGACY ARTIFACT MAY REACH THIS RUNG (finding A of the 2026-09-23 convergence
       round). The bound's timer has been dump-only since the build that made it so,
       and the header states which convention the file was written under: a
       :data:`stall_watchdog.DUMP_POLICY_MARKER` header proves the fire could not have
       ended anything, a LEGACY fatal-policy sentence proves it could, and an artifact
       that proves neither gets no such claim. Before this the rung read "fired with
       no post-fire held marker => the bound ended it", which was TRUE of the legacy
       builds — an idle fire there took the native exit, so nothing was left to write
       a marker — and FALSE of everything since: a fire held by the GIL cannot append
       a Python marker at all, and any later stop or crash is a separate cause with
       its own rung. The dump is still keyed to this row's pid and still written AT
       THE INSTANT OF THE FIRE — ``faulthandler`` reaches it from a C thread that runs
       no Python — which is why the rung sits above every rung below; what changed is
       that the CONVENTION has to be stated by the artifact rather than inferred from
       what it does not contain. See :func:`_stall_bound_evidence` for how a fired
       dump is told from a file a SIGKILL left and
       :data:`incidents.STALL_BOUND_CAUSE` for what its absence cost.

    1. **the row's own recorded exit cause, when it names a signal** →
       ``runtime-shutdown``. This is a stop sweep that reached its target: the
       runtime was asked to leave, wrote the signal down, and was killed before
       its turn ended. First-hand evidence, so it outranks everything below.
    2. **the row's own recorded exit cause, when it is a token this taxonomy
       knows AND IS NOT A VERDICT** → that token. The runtime wrote it on its
       way out (``Session.note_cut_off`` -> ``TurnJournal.note_exit``), which
       makes it first-hand in the same way rung 1 is, and it is the ONLY rung
       that can carry a bound: the bounded handover records
       ``runtime-overdue`` (:data:`types.BUILD_DRAIN_OVERDUE_CAUSE`) and the
       sentence that names its bound, where the inferences below can only speak
       about the install or about a death nobody recorded. It outranks rung 3
       because "the install moved since this build" is an inference from the
       DISK, while this is a statement about itself — and for a drained runtime
       the install has usually moved, so the inference used to win by default
       (QA round 1, Q-2).

       ``KILL_CAUSE`` IS EXCLUDED BY THE PREDICATE, NOT BY A LEAD (review
       round 4, MINOR 1). ``incidents.CUT_OFF_CAUSES`` holds the very token the
       arm below answers with, because that table is the taxonomy's vocabulary
       and rung 4 is one of its causes; matching on membership alone would let a
       row that recorded ``runtime-killed`` be answered by itself, rendering a
       harness-caused death with no actor — the thing this ordering exists to
       forbid. No writer records it today (``note_exit`` is reached only from
       ``process._clean_exit`` and the direct-dispose path, and neither passes
       it), so this is an invariant held against a future writer rather than a
       live defect; the alternative fix, passing ``KILL_UNATTRIBUTED`` as this
       rung's lead, is wrong because the exclusion must not be reachable either
       way. Rung 2 keeps no lead for the tokens it DOES answer:
       ``runtime-overdue`` already names its own mechanism and bound, and that
       sentence is pinned by ``test_a_recorded_bound_outranks_the_install_inference``.
    3. **an install that moved since the row's build** → ``install-mid-update``
       — the install-window tear.
    4. **nothing else** → ``runtime-killed``, now carried by positive evidence
       (this turn was in flight and never ended) instead of by the absence of a
       record.

    RUNG 4 IS EXPLICITLY UNATTRIBUTED, and the word is part of the answer rather
    than a decoration. Nothing that reads this arm has seen a marker — a marker
    covering this run is consumed by ``attention._classify_orphaned_run`` before
    the row is ever consulted — so "this turn died and no act was recorded" is
    exactly what the evidence supports, and saying so is what turns a fleet-wide
    event from "we cannot tell you why" into "none of these deaths had a recorded
    actor". ``incidents.KILL_UNATTRIBUTED`` is the same word the classifier's
    marker arms use when a marker names no actor, so a reader comparing a
    marked death with an unmarked one sees one vocabulary rather than two. It is
    also the arm a row that recorded ``KILL_CAUSE`` lands on, which is why rung 2
    refuses that token (review round 4, MINOR 1).

    ``CUT_OFF_UNKNOWN`` is unreachable from this function and that is the point:
    it is the taxonomy's statement that nothing on disk could say what happened,
    and an open row is something. The token is still the answer for a death that
    left no row at all, which is what "keep the legacy path working when the
    evidence is absent" means.
    """
    from local_operator.incidents import (
        CUT_OFF_CAUSES,
        KILL_CAUSE,
        KILL_UNATTRIBUTED,
        STALL_BOUND_CAUSE,
        render_cut_off_reason,
    )
    from local_operator.session.runtime.stall_watchdog import (
        FIRE_LEGACY_FATAL,
        FIRE_SURVIVED,
        FIRE_UNKNOWN,
    )

    leg, tick_deaths, outcome, teardown = _stall_bound_evidence(row)
    # ONLY A LEGACY ARTIFACT THAT STATES ITS OWN CONVENTION MAY REACH THIS RUNG, and the
    # predicate is an equality against the artifact's answer rather than the absence of a
    # marker (finding A of the 2026-09-23 convergence round). The old test — fired and
    # not held — read "no post-fire marker" as "the bound ended it", which is false on
    # every dump-only build and is how a still-live runtime came to be narrated as a
    # watchdog death.
    if leg is not None and outcome == FIRE_LEGACY_FATAL:
        return (
            "error",
            STALL_BOUND_CAUSE,
            render_cut_off_reason(STALL_BOUND_CAUSE, detail=_stall_bound_detail(leg, tick_deaths)),
        )
    # THE FIRE IS STILL ON THE RECORD — as a CLAUSE, never as the cause, and WHICH clause
    # is what the artifact proved: SURVIVED (a dump-only build, or a legacy held fire),
    # or nothing proved, in which case the fire is named AND the missing fact is named
    # with it (see :data:`STALL_BOUND_LEAD` — the reading that replaces the old
    # inference). The teardown clause is independent of the outcome: ordering is what it
    # proves, and it is added whatever the file says the build's policy was.
    clauses: list[str] = []
    if leg is not None:
        if outcome == FIRE_SURVIVED:
            clauses.append(HELD_BOUND_LEAD)
        elif outcome == FIRE_UNKNOWN:
            clauses.append(STALL_BOUND_LEAD)
        if teardown:
            clauses.append(TEARDOWN_LEAD)
    fire_clause = "; ".join(clauses)
    signal = signal_exit_token(row.exit_cause)
    if signal:
        return (
            "error",
            "runtime-shutdown",
            render_cut_off_reason(
                "runtime-shutdown",
                detail=row_detail(row, lead=f"{signal} received"),
                clause=fire_clause,
            ),
        )
    recorded = str(row.exit_cause or "")
    if recorded in CUT_OFF_CAUSES and recorded != KILL_CAUSE:
        # The runtime's own last word, and a token only if it is one: every other
        # caller of ``note_exit`` passes a sentence ("retiring for 0.59.9"), and a
        # sentence is not a rung of this taxonomy — it would render as itself, which
        # is how the bound used to be invisible to a successor. ``KILL_CAUSE`` is
        # held out of the match rather than given a lead: it is this function's own
        # verdict for a death with no recorded act, so a row carrying it belongs on
        # the unattributed arm below, which is the one that names its actor (review
        # round 4, MINOR 1).
        return (
            "error",
            recorded,
            render_cut_off_reason(recorded, detail=row_detail(row), clause=fire_clause),
        )
    if install_moved(row):
        # The lead rides here too: the install-window arm has the narrowest detail of
        # the four (a build pair), and a runtime that survived a bound before the
        # install moved under it is a fact the reader needs on THIS arm most of all —
        # a stall is why it was still alive to be caught by the tear.
        detail = f"{row.build_label()} → {_current_build_label()}"
        return (
            "error",
            "install-mid-update",
            render_cut_off_reason("install-mid-update", detail=detail, clause=fire_clause),
        )
    return (
        "error",
        KILL_CAUSE,
        render_cut_off_reason(
            KILL_CAUSE,
            detail=row_detail(row, lead=KILL_UNATTRIBUTED),
            # ...AND THE FIRE CLAUSE IS A CLAUSE HERE TOO (agent review round 2, MAJOR-2;
            # design review round 2, D3). This is the arm round 1 measured first — the
            # "died of nothing recorded" row of that table — and it kept the lead inside
            # the parenthetical, so a held-then-killed death and a no-dump death still
            # rendered byte-identically in the listing cell: exactly the collapse of
            # three states into two that the lead's placement exists to prevent. Since
            # finding A the clause is whichever one the artifact earned — survived,
            # unknown, teardown — and the argument is unchanged for all three: the
            # UNATTRIBUTED half stays in the DETAIL, where it belongs, because it
            # qualifies the death, while "the bound fired" changes what the row MEANS.
            clause=fire_clause,
        ),
    )


def _current_build_label() -> str:
    try:
        from local_operator import buildwatch
        from local_operator.update import installed_build

        return installed_build(buildwatch.build_prefix()).label()
    except Exception:  # noqa: BLE001 — the detail is a nicety, never a gate
        return "<unknown>"


def restored_interruption(conversation_dir: Path) -> tuple[str, str, str, str] | None:
    """``(kind, cause, reason, token)`` for a successor boot, or ``None``.

    The trust half of the same problem: a session resumed after a kill has to be
    able to tell an interruption from a completion IN ITS OWN STATE, rather than
    by asking a human — an agent that "remembers" finishing step three must be
    told that step four's state is unknown before it acts on that memory.

    The tuple is in the exact shape ``Session._restored_cut_off`` already
    consumes, so this reaches the model through the ONE narration path that
    exists (``Session._journal_cut_off_once``, deduped on the token) rather than
    through a second one that could double-announce the same death.

    NEVER A RE-RUN. Nothing here submits a prompt, replays a turn or queues work:
    the notice is a transcript row, and the design's rule is that resuming the
    turn is a decision for the agent and the person who read the notice.
    """
    row = open_row_after_death(conversation_dir)
    if row is None:
        return None
    kind, cause, reason = death_verdict(row)
    return kind, cause, reason, journal_token(row)
