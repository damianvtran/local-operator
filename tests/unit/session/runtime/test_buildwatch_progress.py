"""The drain's progress bound: bound the hold by the WORK MOVING, not by time.

``_drain_for`` is right that a build drain must not be bounded by a clock of
its own — it promises that the turn in flight FINISHES, and a bound on time
would cut a turn merely for being long. That promise is only as good as the
work's willingness to finish, and ``is_busy()`` counts two things that can hold
for hours: a gate parked on a user's answer and a lane parked behind a child
process. Measured on the reporting host (2026-09-18, pid 70950, session
8c13a003dc6d, 0.59.7): a stale-build drain latched at 20:32:29 and two hours
later the session still refused every prompt, reporting ``state=wedged`` with
three subagent lanes stalled behind a bash child that had been running 23
minutes. The runtime holds the transcript lease for the whole of that hold, so
no successor can be engaged: not a slow handover, no handover.

What is pinned here is the backstop that ends that state, and — at least as
importantly — what it must NOT do:

* it fires only after ``BUILD_DRAIN_PROGRESS_S`` with no movement at ANY of the
  layers ``_work_motion`` reads (turn footprint, subagent roster generation, job
  rows, spool), so a hold that is moving is never cut, however long it runs;
* the clock is reset by WORK, never by liveness: a viewer attached, a
  ``is_streaming`` flag held True or a reaper tick that keeps ticking is not
  movement, and the incident's runtime reported all three for the two hours it
  was stuck;
* the exit it takes is the SIGNAL drain's (announce with a phrase that names the
  bound, deny a parked gate, hand the drain's wakes over, ``_clean_exit``), so a
  successor reads a journal row that says a bound cut the turn.

Fakes in the style of ``test_process_build_bound.py``, with the four motion
signals on the session double so each can be moved on its own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import update as update_mod
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import _Drain, _drain_for, _reaper
from local_operator.session.runtime.types import (
    BUILD_DRAIN_PROGRESS_S,
    LEAVING_FOR_BUILD,
)
from local_operator.update import BuildStamp

BOUND = BUILD_DRAIN_PROGRESS_S

#: The pair the reporting host's install spanned, as ``test_process_build_bound``
#: uses: a runtime that booted on OLD while the install on disk is NEW.
OLD = BuildStamp(version="0.59.7", source_ref="7fe8b1005")
NEW = BuildStamp(version="0.59.8", source_ref="dec7933a6")

#: The clock's start instant. Arbitrary and far from zero, so a bug that mixed
#: the injected clock up with ``time.monotonic`` would show as a held drain
#: rather than as a coincidence.
T0 = 1000.0


class FakeEntry:
    def __init__(self, entry_id: str) -> None:
        self.id = entry_id
        self.ts = 1.0


class FakeTranscript:
    """The two reads ``_work_motion`` makes: the newest row per kind, and where
    the spool lives."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._latest: dict[str, FakeEntry] = {}

    def latest_entry(self, entry_type: str) -> FakeEntry | None:
        return self._latest.get(entry_type)

    def land(self, entry_type: str, entry_id: str) -> None:
        self._latest[entry_type] = FakeEntry(entry_id)


class FakeJob:
    """A job row, with the two fields that separate a working job from a live id.

    ``output_seq``/``latest_details`` are the real ``AsyncJob`` fields
    (``harness/jobs.py``): the first is the live-output offset ``append_output``
    counts up, the second carries the child relay's activity string. A double that
    omitted them would make every job look silent and quietly delete the signal
    this file exists to pin.
    """

    def __init__(self, job_id: str, status: str = "running") -> None:
        self.id = job_id
        self.status = status
        self.output_seq = 0
        self.latest_details: dict[str, Any] = {}


class FakeJobs:
    def __init__(self) -> None:
        self.rows: list[FakeJob] = []

    def list(self) -> list[FakeJob]:
        return list(self.rows)


class FakeSession:
    """The four motion signals, mute until a test moves one of them."""

    def __init__(self, directory: Path) -> None:
        self.transcript = FakeTranscript(directory)
        self.jobs = FakeJobs()
        self._subagent_roster_generation = 0
        #: Held True by the incident's own shape: the session reported
        #: ``is_streaming`` for two hours with its work stopped.
        self.is_streaming = True
        self.session_id = "0" * 12
        self.handed_wakes = 0

    async def hand_wakes_to_successor(self) -> int:
        self.handed_wakes += 1
        return 0


class FakeJournal:
    def __init__(self) -> None:
        self.exits: list[str] = []

    def note_exit(self, cause: str) -> None:
        self.exits.append(cause)


class FakeHandle:
    def __init__(self, session: FakeSession, *, busy: bool = True, viewers: int = 0) -> None:
        self._session = session
        self._busy = busy
        self._viewers = viewers
        self._turn_journal = FakeJournal()
        self.disposed = False
        self.denials = 0
        self.drains = 0
        self.releases = 0
        self.draining = False
        self.update_failed = ""
        self.drain_cause = ""
        self.retired = False
        self.retire_cause = ""
        self.retire_detail = ""

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return None

    def may_refresh(self) -> str:
        return "busy" if self._busy else ""

    def attach_clients(self) -> int:
        return self._viewers

    def begin_drain(self, cause: str, detail: str = "") -> bool:
        self.drains += 1
        self.draining = True
        self.drain_cause = cause
        return True

    def end_drain(self) -> bool:
        """The release the abandon arm calls (``serving.end_drain``).

        Modelled rather than stubbed away: the assertion the pinning test makes is that
        the latch comes OFF, and a fake that could not record it would let the arm pass
        while the production handle kept refusing admissions.
        """
        if not self.draining:
            return False
        self.draining = False
        self.releases += 1
        return True

    def note_update_failed(self, pair: str, bound: float = 0.0) -> None:
        self.update_failed = pair

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        if self.may_refresh():
            return False
        self.retired = True
        self.retire_cause = cause
        self.retire_detail = detail
        return True

    def _deny_pending_gates(self) -> None:
        self.denials += 1

    async def dispose(self) -> None:
        self.disposed = True


class FakeRuntime:
    def __init__(self, *, boot: BuildStamp | None = None) -> None:
        self._boot_build = boot
        self.closed = False
        self.retiring: list[tuple[str, str, bool, str]] = []
        self.failures: list[tuple[str, float]] = []

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.retiring.append((reason, to, draining, leaving))

    async def note_update_failed(self, pair: str, bound: float = 0.0) -> None:
        self.failures.append((pair, bound))

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def rig(tmp_path: Path) -> SimpleNamespace:
    """One latched build drain over a handle whose work never moves by itself."""
    session = FakeSession(tmp_path)
    (tmp_path / "inbox.jsonl").write_text("", encoding="utf-8")
    handle = FakeHandle(session)
    runtime = FakeRuntime(boot=OLD)
    drain = _Drain(
        detail="the runtime declined to hand over 3x (0.59.7 -> 0.59.8)",
        to=NEW.label(),
        reason="retiring for " + NEW.label(),
        stagger_until=0.0,
        boot=OLD,
    )
    return SimpleNamespace(
        session=session,
        handle=handle,
        runtime=runtime,
        drain=drain,
        stop=asyncio.Event(),
        dir=tmp_path,
    )


def _land_a_tool_boundary(rig: SimpleNamespace, n: int = 1) -> None:
    rig.session.transcript.land("message", f"row-{n}")


def _bump_the_roster(rig: SimpleNamespace) -> None:
    rig.session._subagent_roster_generation += 1


def _settle_a_job(rig: SimpleNamespace) -> None:
    rig.session.jobs.rows.append(FakeJob(f"job-{len(rig.session.jobs.rows)}"))


def _print_from_a_job(rig: SimpleNamespace, chunk: int = 64) -> None:
    """A job's live output growing: what ``append_output`` does as chunks arrive.

    The LAST row, because a test that settles a job first then prints is modelling
    the ordinary shape — a job that has started WORKING, not one that has just been
    admitted.
    """
    if not rig.session.jobs.rows:
        _settle_a_job(rig)
    rig.session.jobs.rows[-1].output_seq += chunk


def _report_from_a_lane(rig: SimpleNamespace, progress: str = "responding") -> None:
    """A lane's activity string moving: written by the child's own relay."""
    if not rig.session.jobs.rows:
        _settle_a_job(rig)
    rig.session.jobs.rows[-1].latest_details = {"progress": progress}


def _spool_a_message(rig: SimpleNamespace) -> None:
    with (rig.dir / "inbox.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"text": "hello"}\n')


async def _tick(rig: SimpleNamespace, at: float) -> bool:
    return await _drain_for(rig.drain, rig.handle, rig.runtime, rig.stop, now=at)


async def _wait_for(predicate: Any, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.fixture
def disk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Control the build stamp on disk, as ``test_process_build_bound`` does."""
    state: dict[str, Any] = {"build": NEW, "age": 999.0}
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: state["build"])
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: state["build"])
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: state["age"])
    for name in ("LOP_BUILD_SETTLE_S", "LOP_BUILD_STAGGER_S", "LOP_SESSION_GRACE_S"):
        monkeypatch.delenv(name, raising=False)
    return state


# -- the backstop -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_drain_whose_work_stops_moving_ABANDONS_the_move(rig) -> None:
    """THE PINNING TEST, and its subject is the operator's rule about build moves.

    A busy runtime whose work reports nothing for the bound used to be released BY
    FORCE: the parked gates denied, the wakes handed to a successor, the record
    re-published and the process gone with ``runtime-overdue`` as the cause — a turn
    the operator was still running, cut so that a build could land. That is the
    behaviour this cell now forbids, and every assertion below is one of its parts,
    inverted deliberately:

    * the tick does NOT retire (False), and nothing is disposed or stopped;
    * the gate is NOT denied and the wakes are NOT handed over: the turn keeps its
      tools and a successor is not coming;
    * the LATCH IS RELEASED, so a runtime that keeps serving can take work again —
      the property that makes "abandon" different from "hold forever";
    * the failure is PUBLISHED, which is what turns a stuck handover from a silent
      hold into something reportable (the operator's own requirement);
    * the phrase the record carries stays the ordinary build one, because a departure
      that did not happen may not wear a departure's words.

    The bound is still honoured, which is the half that must NOT change: one second
    short of it the drain is still holding.
    """
    # The rig builds the drain object directly, so the production latch is taken here
    # instead (``_commit_to_leaving`` does it): the abandon's subject is a LATCHED
    # drain, and without this the release would have nothing to release.
    rig.handle.begin_drain(rig.drain.cause, rig.drain.detail)
    assert await _tick(rig, T0) is False, "a fresh drain holds: its work may still be finishing"
    assert rig.runtime.retiring == []
    assert (
        await _tick(rig, T0 + BOUND - 1) is False
    ), "one second short of the bound is not the bound"
    assert not rig.stop.is_set() and not rig.handle.disposed

    assert (
        await _tick(rig, T0 + BOUND) is False
    ), "the bound retired a runtime whose turn was still in flight"
    assert rig.handle.releases == 1, "the drain latch was never released"
    assert rig.handle.draining is False, "the handle still believes it is leaving"
    assert not rig.stop.is_set(), "the process must keep serving, not stop"
    assert rig.handle.disposed is False, "a runtime that keeps its build is never disposed"
    assert rig.handle.denials == 0, "nothing is denied: the parked gate keeps its turn"
    assert rig.session.handed_wakes == 0, "the wakes belong to no successor here"
    assert rig.handle._turn_journal.exits == [], "no exit was journalled"
    assert rig.drain.progress is not None and rig.drain.progress.abandoned is True
    assert rig.runtime.failures, "the abandoned handover was never published"
    assert rig.runtime.failures[0][0] == NEW.label()
    assert rig.runtime.failures[0][1] == BOUND, (
        "the failure was published without the bound it ran out of, so no surface can "
        "say why the update did not happen"
    )
    assert rig.handle.update_failed == "", (
        "the handle remembered the pair, which is the WINDOW rung's memo for not burning "
        "its bound twice — here it would stop the retry the drain rung owes"
    )
    assert [
        leaving for _r, _t, _d, leaving in rig.runtime.retiring
    ] == [], "an abandoned handover announced a departure it did not take"


@pytest.mark.asyncio
async def test_a_landed_tool_boundary_resets_the_clock(rig) -> None:
    """A turn that is DOING something lands a transcript row per completed
    boundary — the same event the drain's own prompt queue subscribes to."""
    assert await _tick(rig, T0) is False
    _land_a_tool_boundary(rig, 2)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + BOUND - 1
    ), (
        "the movement was never observed, so the bound is being measured from the latch: "
        "this cell would pass with a drain clock that never advances"
    )
    assert (
        rig.drain.progress.abandoned is False
    ), "the bound expired before the movement was observed, so nothing was reset"
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the bound was never reached, so this cell says nothing about what resets it"


@pytest.mark.asyncio
async def test_a_lane_step_resets_the_clock(rig) -> None:
    """The roster generation is the one LANE-level signal that reaches the
    parent: a lane's every completed assistant message bumps it, while its
    parent's own transcript stays frozen for the whole lane."""
    assert await _tick(rig, T0) is False
    _bump_the_roster(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + BOUND - 1
    ), (
        "the movement was never observed, so the bound is being measured from the latch: "
        "this cell would pass with a drain clock that never advances"
    )
    assert (
        rig.drain.progress.abandoned is False
    ), "the bound expired before the movement was observed, so nothing was reset"
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the bound was never reached, so this cell says nothing about what resets it"


@pytest.mark.asyncio
async def test_a_settling_job_resets_the_clock(rig) -> None:
    """A job row changing — a bash job settling, a queued one admitted, a lane
    opening or closing. Read as rows rather than as a count, so one lane ending
    as another starts is visible."""
    assert await _tick(rig, T0) is False
    _settle_a_job(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + BOUND - 1
    ), (
        "the movement was never observed, so the bound is being measured from the latch: "
        "this cell would pass with a drain clock that never advances"
    )
    assert (
        rig.drain.progress.abandoned is False
    ), "the bound expired before the movement was observed, so nothing was reset"
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the bound was never reached, so this cell says nothing about what resets it"


@pytest.mark.asyncio
async def test_a_job_that_is_printing_resets_the_clock(rig) -> None:
    """A job whose child is WRITING OUTPUT is work, and it moves the clock (R1).

    ``append_output`` advances the job's live-output offset as chunks arrive from
    the pipe reader, so a background build or a mirrored bash child that keeps
    printing separates from a child that is alive at 0.1% CPU having said nothing
    for the whole bound. The first cut of this backstop could not tell those two
    apart and force-cut both, which is the finding this signal answers.
    """
    assert await _tick(rig, T0) is False
    _settle_a_job(rig)
    assert await _tick(rig, T0 + 1) is False, "the row is seen here; the clock starts at it"

    _print_from_a_job(rig)
    # Observed at the NEXT tick, and measured from THERE: a hold longer than the
    # bound survives on printing alone, one tick at a time.
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + BOUND - 1
    ), (
        "the movement was never observed, so the bound is being measured from the latch: "
        "this cell would pass with a drain clock that never advances"
    )
    assert (
        rig.drain.progress.abandoned is False
    ), "the bound expired before the movement was observed, so nothing was reset"
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the bound was never reached, so this cell says nothing about what resets it"


@pytest.mark.asyncio
async def test_a_lane_reporting_activity_resets_the_clock(rig) -> None:
    """A lane's activity string is written by the CHILD's own event stream.

    ``report_progress`` -> ``latest_details[\"progress\"]`` is coarse next to a step
    boundary, but it is not a tick: it changes when the lane's relay sees its model
    call or message move, so a lane parked inside a silent tool is distinguishable
    from one that is responding.
    """
    assert await _tick(rig, T0) is False
    _settle_a_job(rig)
    assert await _tick(rig, T0 + 1) is False, "the row is seen here; the clock starts at it"

    _report_from_a_lane(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + BOUND - 1
    ), (
        "the movement was never observed, so the bound is being measured from the latch: "
        "this cell would pass with a drain clock that never advances"
    )
    assert (
        rig.drain.progress.abandoned is False
    ), "the bound expired before the movement was observed, so nothing was reset"
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the bound was never reached, so this cell says nothing about what resets it"


@pytest.mark.asyncio
async def test_a_job_row_that_never_changes_cannot_hold_the_drain_open(rig) -> None:
    """The other half of the same signal, and the half that must not be silent.

    A row is not movement; only a CHANGE to one is. A child that has died leaves
    its row's status, output offset and activity frozen, so a drain held behind it
    still reaches the bound and is released — the failure was the drain waiting for
    a predicate that never clears, and no new signal may reintroduce it by
    reporting a constant as if it were a heartbeat.
    """
    assert await _tick(rig, T0) is False
    _settle_a_job(rig)
    _print_from_a_job(rig)
    _report_from_a_lane(rig)
    assert await _tick(rig, T0 + 1) is False, "the whole footprint is seen here"
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + 1
    ), (
        "the footprint change was never observed, so the bound is being measured from the "
        "latch: this cell would pass with a drain clock that never advances"
    )
    assert rig.drain.progress.abandoned is False

    assert await _tick(rig, T0 + 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "a frozen job row held the drain past its bound"
    assert rig.handle.denials == 0, "a frozen row is no reason to deny anyone's tool"


@pytest.mark.asyncio
async def test_every_probe_reads_a_field_the_real_classes_still_have() -> None:
    """R4's second half: the doubles must not be the ONLY source of these shapes.

    Every signal ``_work_motion`` reads is read through ``getattr(..., default)`` so
    that an unreadable probe cannot stop a drain — which means a RENAME on the real
    class freezes that probe for ever while every test in this file stays green: the
    doubles carry the old name, and the clock silently loses a signal. The pin is
    therefore against the REAL classes: model fields by name, methods and properties
    by ``hasattr``, and ``Session``'s own instance attributes by parsing its source
    for ``self.<name> =`` — a plain ``hasattr(Session, name)`` excludes exactly the
    names whose absence caused this class of bug, which is why
    ``tests/unit/session/test_remote_registries.py`` parses the source for the same
    reason.
    """
    import inspect
    import re

    from local_operator.harness.jobs import AsyncJob, AsyncJobManager
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    job_fields = set(getattr(AsyncJob, "__fields__", {}))
    assert {
        "id",
        "status",
        "output_seq",
        "latest_details",
    } <= job_fields, (
        f"_job_footprint reads fields AsyncJob no longer declares: {sorted(job_fields)}"
    )
    assert hasattr(AsyncJobManager, "list"), "the job-row probe would report no jobs at all"
    assert hasattr(Transcript, "latest_entry"), "the transcript probe would freeze"
    assert hasattr(Session, "transcript"), "the transcript probe would report nothing"
    assigned = set(
        re.findall(r"^\s+self\.([a-z_][a-z_0-9]*)\s*(?:[:=])", inspect.getsource(Session), re.M)
    )
    assert {
        "jobs",
        "_subagent_roster_generation",
    } <= assigned, "Session no longer assigns the two attributes the job and lane probes read"


@pytest.mark.asyncio
async def test_the_abandon_publishes_the_bound_and_brands_no_cut_off(rig) -> None:
    """What a successor and a reader can learn about a handover that did not happen.

    THIS CELL USED TO PIN THE EXIT'S PROVENANCE (a journal row carrying
    ``BUILD_DRAIN_OVERDUE_CAUSE`` and a cut-off note carrying the why-now re-read at
    the exit). Both are gone with the cut they described, and their absence is now
    the assertion: a runtime that keeps serving must not hand a successor a cut-off
    note for a turn nobody cut, and must not journal an exit it did not take. What
    it owes instead is the FAILURE — the pair and the bound it ran out of — because
    that is the fact a person acts on ("the update did not happen, and here is
    why"), and it is published under ``UPDATE_FAILED_CAUSE`` by the runtime itself.
    """
    rig.handle.begin_drain(rig.drain.cause, rig.drain.detail)
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND) is False
    assert rig.runtime.failures == [
        (NEW.label(), BOUND)
    ], "the abandoned handover was not published with the pair and the bound"
    assert rig.handle._turn_journal.exits == [], "a runtime that kept serving journalled an exit"
    assert (
        getattr(rig.handle, "_retiring_cause", "") == ""
    ), "a cut-off note was branded for a turn that was never cut"


@pytest.mark.asyncio
async def test_a_spooled_message_resets_the_clock(rig) -> None:
    """The one reset an OUTSIDE actor can drive: a peer message or a fired wake
    reaching the draining runtime and being preserved for its successor."""
    assert await _tick(rig, T0) is False
    _spool_a_message(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    # THE DISCRIMINATING ASSERTION (agent review round 1, MAJOR-1). A return value
    # cannot carry this claim any more: without a reset the drain ABANDONS at the
    # later tick just as surely as with one, so `is False` + `abandoned is True`
    # there is satisfied by a clock that never moved. What this cell is about is the
    # clock itself, so it asserts the clock: the movement was OBSERVED at this tick,
    # i.e. this is the instant the bound is now measured from. Delete the reset and
    # ``moved_at`` stays at the latch's ``T0`` and this fails.
    assert rig.drain.progress is not None and rig.drain.progress.moved_at == pytest.approx(
        T0 + BOUND - 1
    ), (
        "the movement was never observed, so the bound is being measured from the latch: "
        "this cell would pass with a drain clock that never advances"
    )
    assert (
        rig.drain.progress.abandoned is False
    ), "the bound expired before the movement was observed, so nothing was reset"
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the bound was never reached, so this cell says nothing about what resets it"


@pytest.mark.asyncio
async def test_a_lane_stepping_keeps_a_long_hold_alive(rig) -> None:
    """A hold that KEEPS MOVING is never cut, however long it runs.

    Measured here as three hours of holding — twelve times the bound — with one
    lane step every five minutes, which is the shape of a long subagent lane
    whose parent's own transcript never changes for the whole of it.
    """
    at = T0
    for step in range(36):
        at += 300.0
        _bump_the_roster(rig)
        assert await _tick(rig, at) is False, f"cut a moving hold at step {step}"
    assert rig.runtime.retiring == []
    assert not rig.stop.is_set() and not rig.handle.disposed
    assert rig.drain.progress is not None and rig.drain.progress.abandoned is False


@pytest.mark.asyncio
async def test_the_bound_is_measured_from_the_last_movement_not_from_the_latch(rig) -> None:
    """Movement at minute ten buys the hold a full bound from THEN, not from the
    latch: the clock is a staleness clock, not a deadline."""
    _land_a_tool_boundary(rig)
    assert await _tick(rig, T0 + 600) is False
    assert await _tick(rig, T0 + 600 + BOUND - 1) is False
    assert await _tick(rig, T0 + 600 + BOUND) is False
    assert rig.drain.progress is not None and rig.drain.progress.abandoned is True


@pytest.mark.asyncio
async def test_a_streaming_session_is_not_movement(rig) -> None:
    """``is_streaming`` is the flag the incident's runtime held True while its
    work was stopped, and ``is_busy`` reads it for the same reason. It is NOT a
    motion signal: a provider that stopped answering keeps the flag set, so a
    clock that read it would never expire on exactly the wedged session this
    bound exists for."""
    assert rig.session.is_streaming is True
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND) is False
    assert rig.drain.progress is not None and rig.drain.progress.abandoned is True


# -- what must NOT change ----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_drain_never_waits_for_a_viewer(rig, disk) -> None:
    """Design §4.6's one property a helpful-looking change would break: the exit
    happens at the first idle instant, with an interactive viewer attached."""
    rig.handle._viewers = 3
    rig.handle._busy = False
    assert await _tick(rig, T0) is True, "a viewer may not hold the exit"
    assert rig.handle.retired and rig.handle.disposed and rig.stop.is_set()
    assert rig.runtime.retiring == [], "the frame went out at the latch; the exit adds nothing"


@pytest.mark.asyncio
async def test_a_viewer_resets_no_clock(rig) -> None:
    """The other half, and the one the backstop could reintroduce: a viewer's
    presence is not movement, so a stalled hold with viewers attached is still
    cut at the bound."""
    rig.handle._viewers = 3
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert await _tick(rig, T0 + BOUND) is False
    assert rig.drain.progress is not None and rig.drain.progress.abandoned is True


@pytest.mark.asyncio
async def test_an_idle_instant_still_wins_over_the_backstop(rig, disk) -> None:
    """The two rungs are ordered, and the order is load-bearing both ways.

    The stagger is the only thing that may hold a stalled drain past its bound
    (sixteen runtimes that went stale together must not spawn sixteen successors
    together), and the idle check comes BEFORE the backstop — so a work that
    finished while the bound was running out is handed over by the quiet rung,
    which aborts nothing, rather than cut by the forced one.
    """
    rig.drain.stagger_until = T0 + BOUND + 5
    assert await _tick(rig, T0 + BOUND + 1) is False, "the stagger does not sit above the backstop"
    assert rig.runtime.retiring == [] and not rig.handle.disposed

    rig.handle._busy = False  # the turn ended while the backstop was waiting
    assert await _tick(rig, T0 + BOUND + 6) is True
    assert rig.handle.retired, "the quiet retire rung is the one that should have run"
    assert rig.runtime.retiring == [], "a clean exit does not re-announce"
    assert rig.handle.denials == 0, "nothing was cut, so nothing needed denying"


@pytest.mark.asyncio
async def test_the_abandon_is_taken_once_per_drain(rig) -> None:
    """A later tick (the signal drain's loop can ask in principle) is answered
    without a second release, a second publish or a second announcement.

    The guard is what keeps a runtime that cannot reach idle from reporting its
    failure on every ``REAP_CHECK_S`` tick — four times a minute, forever, on a
    surface a person reads. The retry belongs to the WATCH, which re-commits a
    fresh drain on its own terms, not to the rung being re-entered.
    """
    rig.handle.begin_drain(rig.drain.cause, rig.drain.detail)
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND) is False
    assert await _tick(rig, T0 + BOUND + 1) is False
    assert len(rig.runtime.failures) == 1, rig.runtime.failures
    assert rig.handle.releases == 1, "the latch was released more than once"
    assert rig.runtime.retiring == [], "the abandon announced a departure it did not take"


@pytest.mark.asyncio
async def test_the_clock_runs_on_monotonic_time_when_no_clock_is_injected(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The injected clock is for the tests above; production callers pass none,
    so this drives the real one over a shortened bound."""
    monkeypatch.setattr(child_mod, "BUILD_DRAIN_PROGRESS_S", 0.05)
    assert await _drain_for(rig.drain, rig.handle, rig.runtime, rig.stop) is False
    await asyncio.sleep(0.06)
    assert await _drain_for(rig.drain, rig.handle, rig.runtime, rig.stop) is False
    assert (
        rig.drain.progress is not None and rig.drain.progress.abandoned is True
    ), "the real clock never reached the bound, so this cell proves nothing about it"


# -- the ladder it lands on --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_reaper_abandons_a_stalled_drain_and_retries_it(
    disk, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole ladder, on the REAL reaper, with the operator's rule applied.

    A permanently busy runtime declines a newer build, latches the drain, and its
    work reports nothing for the bound. What must happen then is the change this
    file is about: the handover is ABANDONED — the latch released, the failure
    published, the process still serving — and the next check re-commits a fresh
    drain, so a build that cannot land while the work is stuck keeps being asked
    about instead of taking the turn with it.

    The two announcements are still the shape of the record: the FIRST is the
    latch's (``stale-build``, promising the turn finishes), and the assertion that
    there is no SECOND is what separates this from the cut it replaced.
    """
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(child_mod, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(child_mod, "BUILD_DRAIN_PROGRESS_S", 0.05)
    monkeypatch.setattr(child_mod.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "60")
    disk["build"] = NEW
    runtime = FakeRuntime(boot=OLD)
    handle = FakeHandle(FakeSession(tmp_path), busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, runtime, stop))

    assert await _wait_for(lambda: handle.drains == 1), "the drain latch never engaged"
    assert runtime.retiring == [("stale-build", NEW.label(), True, LEAVING_FOR_BUILD)]
    assert not stop.is_set() and not handle.disposed, "in-flight work is not aborted on the latch"

    assert await _wait_for(lambda: runtime.failures), (
        "the stalled drain was never abandoned: the failed handover is what says why "
        "this runtime is still serving the build it loaded"
    )
    # ORDERED AFTER THE PUBLISH ON PURPOSE. ``_abandon_move`` releases the latch
    # BEFORE it publishes, so waiting on the release and then reading ``failures``
    # raced the publish on a loaded worker (measured: green in this file alone, red
    # inside the runtime slice). Waiting on the publish proves both happened.
    assert (
        handle.releases == 1
    ), "the latch was never released, so this runtime cannot take work again"
    # THE COMMITMENT OUTLIVES THE ABANDON, and both halves are asserted because a
    # re-latch would break the first without failing the second: the drain object
    # stays (a second ``begin_drain`` re-runs ``Session.retire_wakes_to_inbox``,
    # which starts a fresh list of one-shot wakes and discards the ones this drain
    # already swallowed), and the departure still happens at the first idle instant.
    assert handle.drains == 1, "the handover was latched a second time"
    assert runtime.retiring == [
        ("stale-build", NEW.label(), True, LEAVING_FOR_BUILD)
    ], "the departure was announced a second time, so a viewer would go cold twice"
    handle._busy = False  # the work finally ends
    assert await _wait_for(lambda: handle.disposed), (
        "the work finished and the runtime never left: the abandonment turned a stalled "
        "handover into a permanent one"
    )
    assert stop.is_set(), "the clean idle exit is the one that ends this runtime"
    await task
