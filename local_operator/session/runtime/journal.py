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

TWO ARTIFACTS, ONE STORY
------------------------
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
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
#: ends "and nothing recorded a stop", for a death where a signal was recorded.
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
    deriving the probe from that age here would fork ``ps`` (~2.4-4.6 ms, see
    ``procstate.is_zombie``) once per LIVE runtime at every boot — ~100-200 ms
    for a forty-session fleet, on the path this design measured at ~1.2 s.

    The consequence is bounded and in the safe direction: a record whose pid has
    exited but has not been reaped yet reads as alive through signal-0 and is
    kept one cycle longer. That costs a file, not an answer — a zombie's record
    is still a true statement about a pid that booted on a build, and the next
    boot after its parent reaps it prunes it.
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
        try:
            record = BootRecord.from_json(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
        if record is None:
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
        aged_out = (moment - started_at) > registry.REAPED_MAX_AGE_S
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
        # nothing recorded a stop.
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


def death_verdict(row: TurnJournalRow) -> tuple[str, str, str]:
    """``(kind, cause, reason)`` for a runtime that left an open row behind.

    THE PREFERENCE ORDER IS THE FIX, and every rung is a NAMED cause:

    1. **the row's own recorded exit cause, when it names a signal** →
       ``runtime-shutdown``. This is a stop sweep that reached its target: the
       runtime was asked to leave, wrote the signal down, and was killed before
       its turn ended. First-hand evidence, so it outranks everything below.
    2. **the row's own recorded exit cause, when it is a token this taxonomy
       knows** → that token. The runtime wrote it on its way out
       (``Session.note_cut_off`` -> ``TurnJournal.note_exit``), which makes it
       first-hand in the same way rung 1 is, and it is the ONLY rung that can
       carry a bound: the bounded handover records ``runtime-overdue``
       (:data:`types.BUILD_DRAIN_OVERDUE_CAUSE`) and the sentence that names its
       bound, where the inferences below can only speak about the install or about
       a death nobody recorded. It outranks rung 3 because "the install moved since
       this build" is an inference from the DISK, while this is a statement about
       itself — and for a drained runtime the install has usually moved, so the
       inference used to win by default (QA round 1, Q-2).
    3. **an install that moved since the row's build** → ``install-mid-update``
       — the install-window tear.
    4. **nothing else** → ``runtime-killed``, now carried by positive evidence
       (this turn was in flight and never ended) instead of by the absence of a
       record.

    ``CUT_OFF_UNKNOWN`` is unreachable from this function and that is the point:
    it is the taxonomy's statement that nothing on disk could say what happened,
    and an open row is something. The token is still the answer for a death that
    left no row at all, which is what "keep the legacy path working when the
    evidence is absent" means.
    """
    from local_operator.incidents import CUT_OFF_CAUSES, render_cut_off_reason

    signal = signal_exit_token(row.exit_cause)
    if signal:
        return (
            "error",
            "runtime-shutdown",
            render_cut_off_reason(
                "runtime-shutdown", detail=row_detail(row, lead=f"{signal} received")
            ),
        )
    recorded = str(row.exit_cause or "")
    if recorded in CUT_OFF_CAUSES:
        # The runtime's own last word, and a token only if it is one: every other
        # caller of ``note_exit`` passes a sentence ("retiring for 0.59.9"), and a
        # sentence is not a rung of this taxonomy — it would render as itself, which
        # is how the bound used to be invisible to a successor.
        return (
            "error",
            recorded,
            render_cut_off_reason(recorded, detail=row_detail(row)),
        )
    if install_moved(row):
        detail = f" ({row.build_label()} → {_current_build_label()})"
        return (
            "error",
            "install-mid-update",
            render_cut_off_reason("install-mid-update", detail=detail),
        )
    return (
        "error",
        "runtime-killed",
        render_cut_off_reason("runtime-killed", detail=row_detail(row)),
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
