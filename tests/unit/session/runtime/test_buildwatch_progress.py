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
from local_operator.session.runtime.process import (
    _BUILD_OVERDUE_EXIT_REASON,
    _BUILD_OVERDUE_REASON,
    _Drain,
    _drain_detail_at_exit,
    _drain_for,
    _reaper,
)
from local_operator.session.runtime.types import (
    BUILD_DRAIN_OVERDUE_CAUSE,
    BUILD_DRAIN_PROGRESS_S,
    LEAVING_FOR_BUILD,
    LEAVING_FOR_BUILD_OVERDUE,
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
        self.drain_cause = cause
        return True

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

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        self.retiring.append((reason, to, draining, leaving))

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
async def test_a_drain_whose_work_stops_moving_is_cut_at_the_progress_bound(rig) -> None:
    """THE PINNING TEST. A busy runtime whose work reports nothing for the bound
    is released BY FORCE, and the release says why.

    Red before this change, in the incident's own shape: the drain held at
    every tick past the bound for as long as the test cared to tick, because
    the only term that could release it was ``_idle_for_refresh`` and a lane
    parked behind a bash child keeps ``is_busy()`` True forever.
    """
    assert await _tick(rig, T0) is False, "a fresh drain holds: its work may still be finishing"
    assert rig.runtime.retiring == []
    assert (
        await _tick(rig, T0 + BOUND - 1) is False
    ), "one second short of the bound is not the bound"
    assert not rig.stop.is_set() and not rig.handle.disposed

    assert await _tick(rig, T0 + BOUND) is True
    assert rig.runtime.retiring == [
        (_BUILD_OVERDUE_REASON, NEW.label(), True, LEAVING_FOR_BUILD_OVERDUE)
    ], "the frame's reason and the record's phrase must BOTH name the bound"
    assert rig.handle.disposed and rig.stop.is_set()
    assert rig.handle.denials == 1, "a gate parked on a user must not hold the exit"
    assert rig.session.handed_wakes == 1, "the drain's swallowed wakes belong to the successor"
    assert rig.handle._turn_journal.exits == [_BUILD_OVERDUE_EXIT_REASON]
    assert rig.drain.progress is not None and rig.drain.progress.overdue is True
    assert rig.handle.drains == 0, "the latch is not re-taken (see the rung's docstring)"
    assert rig.handle.drain_cause == "", "and the drain's own cause is left as the latch wrote it"


@pytest.mark.asyncio
async def test_a_landed_tool_boundary_resets_the_clock(rig) -> None:
    """A turn that is DOING something lands a transcript row per completed
    boundary — the same event the drain's own prompt queue subscribes to."""
    assert await _tick(rig, T0) is False
    _land_a_tool_boundary(rig, 2)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is True


@pytest.mark.asyncio
async def test_a_lane_step_resets_the_clock(rig) -> None:
    """The roster generation is the one LANE-level signal that reaches the
    parent: a lane's every completed assistant message bumps it, while its
    parent's own transcript stays frozen for the whole lane."""
    assert await _tick(rig, T0) is False
    _bump_the_roster(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is True


@pytest.mark.asyncio
async def test_a_settling_job_resets_the_clock(rig) -> None:
    """A job row changing — a bash job settling, a queued one admitted, a lane
    opening or closing. Read as rows rather than as a count, so one lane ending
    as another starts is visible."""
    assert await _tick(rig, T0) is False
    _settle_a_job(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is True


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
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is True


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
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is True


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

    assert await _tick(rig, T0 + 1 + BOUND) is True
    assert rig.handle.denials == 1


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
async def test_the_exit_records_a_renderable_cause_and_a_fresh_why_now(rig) -> None:
    """Q-2 and R2: what a successor can read about this departure afterwards.

    Two facts, both of which were wrong or absent in the first cut. The journal
    gets the TOKEN (``types.BUILD_DRAIN_OVERDUE_CAUSE``) rather than a sentence,
    because the row is all that outlives the process and a taxonomy can only
    render a rung it knows — that is what makes a successor able to say the session
    was handed over by a bound. And the cut-off note gets the why-now RE-READ at
    this instant: this rung is only ever reached after hours, so the latch's pair
    can name a build the install left long ago.
    """
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND) is True
    assert rig.handle._turn_journal.exits == [BUILD_DRAIN_OVERDUE_CAUSE]
    # The attribution the disposal's cut-off note is built from: the departure's own
    # token, and the pair re-read at the exit rather than the latch's stale one.
    assert rig.handle._retiring_cause == BUILD_DRAIN_OVERDUE_CAUSE
    assert rig.handle._retiring_detail == _drain_detail_at_exit(rig.drain)
    assert rig.handle._retiring_detail != rig.drain.detail


@pytest.mark.asyncio
async def test_a_spooled_message_resets_the_clock(rig) -> None:
    """The one reset an OUTSIDE actor can drive: a peer message or a fired wake
    reaching the draining runtime and being preserved for its successor."""
    assert await _tick(rig, T0) is False
    _spool_a_message(rig)
    assert await _tick(rig, T0 + BOUND - 1) is False
    assert rig.runtime.retiring == []
    assert await _tick(rig, T0 + BOUND - 1 + BOUND) is True


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
    assert rig.drain.progress is not None and rig.drain.progress.overdue is False


@pytest.mark.asyncio
async def test_the_bound_is_measured_from_the_last_movement_not_from_the_latch(rig) -> None:
    """Movement at minute ten buys the hold a full bound from THEN, not from the
    latch: the clock is a staleness clock, not a deadline."""
    _land_a_tool_boundary(rig)
    assert await _tick(rig, T0 + 600) is False
    assert await _tick(rig, T0 + 600 + BOUND - 1) is False
    assert await _tick(rig, T0 + 600 + BOUND) is True


@pytest.mark.asyncio
async def test_a_streaming_session_is_not_movement(rig) -> None:
    """``is_streaming`` is the flag the incident's runtime held True while its
    work was stopped, and ``is_busy`` reads it for the same reason. It is NOT a
    motion signal: a provider that stopped answering keeps the flag set, so a
    clock that read it would never expire on exactly the wedged session this
    bound exists for."""
    assert rig.session.is_streaming is True
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND) is True


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
    assert await _tick(rig, T0 + BOUND) is True


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
async def test_the_forced_exit_is_taken_once(rig) -> None:
    """A later tick (the signal drain's loop can ask in principle) is answered
    without a second announcement, a second denial or a second disposal."""
    assert await _tick(rig, T0) is False
    assert await _tick(rig, T0 + BOUND) is True
    assert await _tick(rig, T0 + BOUND + 1) is True
    assert len(rig.runtime.retiring) == 1
    assert rig.handle.denials == 1 and rig.session.handed_wakes == 1


@pytest.mark.asyncio
async def test_the_clock_runs_on_monotonic_time_when_no_clock_is_injected(
    rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The injected clock is for the tests above; production callers pass none,
    so this drives the real one over a shortened bound."""
    monkeypatch.setattr(child_mod, "BUILD_DRAIN_PROGRESS_S", 0.05)
    assert await _drain_for(rig.drain, rig.handle, rig.runtime, rig.stop) is False
    await asyncio.sleep(0.06)
    assert await _drain_for(rig.drain, rig.handle, rig.runtime, rig.stop) is True


# -- the ladder it lands on --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_reaper_releases_a_stalled_drain_and_journals_the_bound(
    disk, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole ladder, on the real reaper: a permanently busy runtime declines
    a newer build, latches the drain, and — with its work reporting nothing —
    is released by the backstop instead of holding forever.

    The two announcements are the point: the FIRST is the latch's
    (``stale-build``, promising the turn finishes), and a session that never got
    there used to publish only that one, for hours, while refusing every
    message.
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

    assert await _wait_for(lambda: handle.disposed), "the stalled drain was never released"
    assert runtime.retiring[-1] == (
        _BUILD_OVERDUE_REASON,
        NEW.label(),
        True,
        LEAVING_FOR_BUILD_OVERDUE,
    )
    assert stop.is_set() and handle.retired is False, "the cut rung is not the quiet retire rung"
    await task
