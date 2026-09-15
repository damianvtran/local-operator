"""The drain latch: leaving a build behind WITHOUT aborting what is running.

``begin_retire`` refuses while any work would be lost, and that refusal is the
defect the bound exists to bound: a session busy for hours never reaches an
idle instant, so a runtime whose install was replaced underneath it keeps
executing a tree that is gone. ``begin_drain`` (design §4 F1) drops the idle
gate and keeps every other guarantee — no new work is admitted, nothing in
flight is touched, and a message that arrives while the runtime is still
finishing is SPOOLED for the successor rather than dropped or run against the
build that is leaving.

Exercised the way ``test_cut_off_turns._LatchHost`` exercises the sibling
latch: the handle's real methods over a stub session and a stub predicate, so
the assertions land on the latch rather than on a runtime boot (the e2e stage
covers the boot).
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.wake import WakeSchedule
from local_operator.session.errors import RuntimeRetiring
from local_operator.session.runtime.inbox import (
    INBOX_NAME,
    SPOOL_RECEIPT_WAKE,
    peek_inbox,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import SIGNAL_DRAIN_CAUSE
from local_operator.session.session import Session


class FakeSession:
    """The slice of Session these latches touch."""

    def __init__(self, directory: Path, *, busy: bool = True) -> None:
        self.transcript = SimpleNamespace(directory=directory)
        self.busy = busy
        self.notes: list[tuple[str, str]] = []
        self.deliberate = 0
        self.peer_calls: list[tuple[str, str, bool]] = []

    # -- the two verdict writers -------------------------------------------
    def note_cut_off(self, cause: str, detail: str = "") -> None:
        self.notes.append((cause, detail))

    def note_deliberate_stop(self) -> None:
        self.deliberate += 1

    # -- peer delivery ------------------------------------------------------
    async def receive_peer_message(
        self, text: str, *, mode: str = "mailbox", wake: bool = False, sender: Any = None
    ) -> str:
        self.peer_calls.append((text, mode, wake))
        return "delivered"


class DrainHost:
    """``ServingSessionHandle``'s drain latch over a stub session.

    ``may_refresh`` is the only thing stubbed about the predicate: a runtime
    that reports work it would lose, which is the case the whole rung exists
    for.
    """

    begin_drain = ServingSessionHandle.begin_drain
    begin_retire = ServingSessionHandle.begin_retire
    # Bound like any other method: ``_retiring_refusal`` reads the handle's own
    # latched cause to choose its sentence, so the stub has to hand it ``self``
    # (design round 4, D10).
    _retiring_refusal = ServingSessionHandle._retiring_refusal
    _spool_for_successor = ServingSessionHandle._spool_for_successor
    receive_peer_message = ServingSessionHandle.receive_peer_message
    _note_deliberate_stop = ServingSessionHandle._note_deliberate_stop
    _check_loop_thread = lambda self: None  # noqa: E731 — the real one only asserts

    def __init__(self, session: FakeSession, *, busy: bool = True) -> None:
        self._session = session
        self._busy = busy
        self._retiring_cause = ""
        self._draining = False
        self._exit_committed = False
        self._disposing = False
        self._fold = SimpleNamespace(note_peer_message=lambda *_a, **_k: None)

    def may_refresh(self) -> str:
        return "busy" if self._busy else ""

    def _notify(self) -> None:
        pass


def _host(tmp_path: Path, *, busy: bool = True) -> tuple[DrainHost, FakeSession]:
    session = FakeSession(tmp_path / "sessions" / "s1")
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return DrainHost(session, busy=busy), session


# -- the residency predicate's warm-window term (QA round 1, Q-1) ----------------
#
# Only these tests use the REAL ``may_refresh``. Everything else in this file
# stubs the predicate wholesale, because there the LATCH is what is under test;
# Q-1 lives inside the predicate, where the wedge was.


class PredicateHost:
    """The production ``may_refresh`` over stubbed terms."""

    may_refresh = ServingSessionHandle.may_refresh

    def __init__(self, *, busy: bool = False, wake_due_ms: int | None = None) -> None:
        self._busy = busy
        self._wake_due_ms = wake_due_ms

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._wake_due_ms


def test_may_refresh_names_a_wake_inside_the_warm_window() -> None:
    """The term, through the shared home, on a handle whose accessor answers."""
    now_ms = int(time.time() * 1000)
    assert (
        PredicateHost(wake_due_ms=now_ms + 1_000).may_refresh() == "wake due within the warm window"
    )
    assert PredicateHost(wake_due_ms=now_ms + 3_600_000).may_refresh() == ""
    assert PredicateHost(busy=True, wake_due_ms=now_ms).may_refresh() == "busy", "busy wins"


def test_may_refresh_fails_open_when_the_warm_window_probe_breaks(monkeypatch) -> None:
    """Q-1: a predicate that cannot be evaluated must NEVER pin the runtime.

    The blocker QA found was this exact shape: the term was reached through a
    function-local import of ``...runtime.process``, which a runtime executing
    that module as ``__main__`` answers from DISK — so at the one moment the
    predicate mattered most (the loaded tree had been replaced) it raised
    ``ImportError``. ``process._idle_for_refresh`` reads a failing predicate as
    "not idle" by design, so the drain never saw an idle instant, never reached
    ``begin_retire``/``_clean_exit``, and the session was refused forever while
    holding the lease — no successor could boot. The helper now lives in the
    stdlib-only module both sides import, and the consult is guarded besides:
    a raising probe reads as "no wake", the same answer the helper gives its
    own broken accessor.
    """
    from local_operator.session.runtime import serving as serving_mod

    def explode(_handle: Any, **_kwargs: Any) -> bool:
        raise ImportError("cannot import name 'process' from 'local_operator.session.runtime'")

    monkeypatch.setattr(serving_mod, "_wake_within_window", explode)
    assert PredicateHost().may_refresh() == "", "a broken probe must not pin the runtime"


def test_the_warm_window_term_is_not_reached_through_the_runtime_module() -> None:
    """Q-1's pin on the construct, not just the behaviour.

    The behavioural pin is the ARMED e2e stage (``test_runtime_refresh_e2e``),
    which is the only place the loaded tree can actually be gone — the repo's
    own interpreter is an editable venv, where the probe is disarmed, which is
    why the wedge shipped. This one is cheap insurance that the exact construct
    that caused it cannot come back: an import of the runtime module from
    inside the predicate. Read off the SOURCE, because a call cannot tell the
    two homes apart while the tree is intact — both return the same bool.
    """
    import inspect

    from local_operator import buildwatch
    from local_operator.session.runtime import process, serving

    assert serving._wake_within_window is buildwatch.wake_within_window
    assert process._wake_within_window is buildwatch.wake_within_window
    source = inspect.getsource(ServingSessionHandle.may_refresh)
    assert "session.runtime.process import" not in source, "the wedge's construct is back"
    assert "runtime.process" not in inspect.getsource(buildwatch.wake_within_window).replace(
        "runtime", ""
    ), "the shared home must not reach back into the runtime"


# -- the latch ------------------------------------------------------------------


def test_begin_drain_latches_while_a_turn_is_running(tmp_path: Path) -> None:
    """The one difference from ``begin_retire``: the latch does not wait."""
    host, session = _host(tmp_path, busy=True)
    assert host.begin_retire("runtime-retired") is False, "the soft rung still refuses"
    assert host._retiring_cause == ""

    assert host.begin_drain("runtime-retired", " (0.54.33@7fe8b10 → 0.54.39@dec7933)") is True
    assert host._retiring_cause == "runtime-retired"
    assert host._draining is True
    assert host._exit_committed is False, "the exit has not been taken yet"
    # The refusal is a TYPED admission category carrying the shared sentence,
    # and that sentence deliberately does NOT name the internal cause token any
    # more — the token was the complaint (design round 1, D2). The cause is
    # still on the handle, which the assertion above pins.
    refusal = host._retiring_refusal()
    assert isinstance(refusal, RuntimeRetiring)
    assert "runtime-retired" not in str(refusal)
    assert session.notes == [], "no turn is being cut off, so no cut-off may be recorded"


def test_the_refusal_describes_the_departure_the_handle_latched(tmp_path: Path) -> None:
    """D10/MAJOR-2: one gate, two departures, and the sentence follows the cause.

    ``prompt`` refuses for the whole of ANY drain, and ``begin_drain`` latches
    from the SIGTERM arm too — so this accessor served a signalled runtime the
    build sentence, "the one it loaded is gone from disk", under a notice that
    correctly said the session had been signalled to stop. The cause the handle
    latched is the discriminator, and it survives to the exit rung because
    ``process._drain_for`` re-passes ``drain.cause`` to ``begin_retire``.

    BOTH HALVES ARE PINNED HERE, because the fix would be just as wrong the
    other way round: the build handover keeps the sentence the design rounds
    measured for it, and the signal arm is the one that had none of its own.
    """
    host, _session = _host(tmp_path, busy=True)

    assert host.begin_drain(SIGNAL_DRAIN_CAUSE, "SIGTERM: drained to the turn's end") is True
    signalled = host._retiring_refusal()
    assert isinstance(signalled, RuntimeRetiring)
    assert signalled.HEAD == RuntimeRetiring.HEAD_SIGNALLED, signalled.HEAD
    assert "newer build" not in str(signalled), str(signalled)
    assert "send it again" in str(signalled), "the one act left still has to be named"
    assert SIGNAL_DRAIN_CAUSE not in str(signalled), str(signalled)

    host, _session = _host(tmp_path, busy=True)
    assert host.begin_drain("runtime-retired", " (0.54.33@7fe8b10 → 0.54.39@dec7933)") is True
    build = host._retiring_refusal()
    assert isinstance(build, RuntimeRetiring)
    assert build.HEAD == RuntimeRetiring.HEAD, build.HEAD


@pytest.mark.parametrize(
    "token",
    ["", "retiring", "probably-fine", RuntimeRetiring.SIGNAL, RuntimeRetiring.BUILD],
)
def test_only_the_two_enumerated_triggers_cross_the_transport(token: str) -> None:
    """The decode validates the token, because the far side rebuilds SENTENCES.

    ``admission_error`` is the only entry point that turns a peer's frame into a
    local exception, and its contract is the module docstring's: an enumerated
    category and nothing else — which is why the count is an int and not a
    string. A trigger is therefore admitted only from the closed set, and
    anything else (including absent, which is what an older runtime sends) means
    "this raiser cannot name its departure": the default sentence.
    """
    from local_operator.session.errors import admission_error

    decoded = admission_error(RuntimeRetiring.code, None, token)
    assert isinstance(decoded, RuntimeRetiring)
    expected = (
        RuntimeRetiring.HEAD_SIGNALLED if token == RuntimeRetiring.SIGNAL else RuntimeRetiring.HEAD
    )
    assert decoded.HEAD == expected, (token, decoded.HEAD)


def test_begin_retire_still_records_the_cut_off_it_owes(tmp_path: Path) -> None:
    """The contrast that keeps the taxonomy honest: the EXIT rung notes it."""
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired", " (a → b)") is True
    assert host._exit_committed is True
    assert session.notes == [("runtime-retired", " (a → b)")]


def test_a_disposing_handle_refuses_the_latch(tmp_path: Path) -> None:
    """A second exit must not race the disposal that is already running."""
    host, _ = _host(tmp_path)
    host._disposing = True
    assert host.begin_drain("runtime-retired") is False
    assert host._draining is False


# -- admissions during the drain ------------------------------------------------


@pytest.mark.asyncio
async def test_a_peer_wake_during_the_drain_is_spooled_for_the_successor(
    tmp_path: Path,
) -> None:
    """invariant (iv): deferred, not dropped, and never run on the dying build."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired", "declined 3x") is True

    receipt = await host.receive_peer_message(
        "the build is moving", mode="steer", wake=True, sender={"name": "peer"}
    )
    assert receipt == SPOOL_RECEIPT_WAKE, receipt
    assert session.peer_calls == [], "a turn must not be started on a build that is leaving"

    directory = session.transcript.directory
    rows = peek_inbox(directory)
    assert len(rows) == 1
    assert rows[0].text == "the build is moving"
    assert rows[0].sender == {"name": "peer"}
    # The successor's boot drain (``process._drain_inbox_into``) reads exactly
    # this file, so the row is what the next engage delivers.
    assert (directory / INBOX_NAME).exists()


@pytest.mark.asyncio
async def test_a_committed_exit_refuses_a_wake_instead(tmp_path: Path) -> None:
    """Once the exit is being taken there is no successor window left, so the
    message keeps the vocabulary refusal the sender can act on."""
    host, session = _host(tmp_path, busy=False)
    assert host.begin_retire("runtime-retired") is True
    with pytest.raises(RuntimeError) as caught:
        await host.receive_peer_message("too late", mode="steer", wake=True)
    assert "send it again" in str(caught.value)
    assert peek_inbox(session.transcript.directory) == []


@pytest.mark.asyncio
async def test_a_quiet_note_is_still_delivered_during_the_drain(tmp_path: Path) -> None:
    """Unchanged behaviour, and deliberately so: a record-only note opens no
    turn, so refusing it would drop something the sender was told had landed."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired") is True
    assert await host.receive_peer_message("fyi") == "delivered"
    assert session.peer_calls == [("fyi", "mailbox", False)]
    assert peek_inbox(session.transcript.directory) == []


def test_a_user_stop_during_the_drain_is_still_the_users_own(tmp_path: Path) -> None:
    """A drain lasts as long as the work does, so a ``/stop`` landing inside it
    ends the turn by the operator's own hand — reporting it as housekeeping
    would misattribute the cancel. Only the committed EXIT suppresses it."""
    host, session = _host(tmp_path)
    assert host.begin_drain("runtime-retired") is True
    host._note_deliberate_stop()
    assert session.deliberate == 1

    host._exit_committed = True
    host._note_deliberate_stop()
    assert session.deliberate == 1, "the retirement owns the verdict from here"


# -- the wake divert (the session side of invariant iv) --------------------------


class WakeHost:
    """``Session``'s wake-handover methods over a stub transcript and scheduler."""

    retire_wakes_to_inbox = Session.retire_wakes_to_inbox
    _spool_wake_to_inbox = Session._spool_wake_to_inbox
    _queue_wake_rearm = Session._queue_wake_rearm
    hand_wakes_to_successor = Session.hand_wakes_to_successor

    #: Declared for the type checker, which cannot see an attribute a borrowed
    #: method assigns on the instance (``retire_wakes_to_inbox`` sets it).
    _wake_rearms: list[Any]

    def __init__(self, directory: Path | None, *, live: tuple[Any, ...] = ()) -> None:
        self._transcript = SimpleNamespace(directory=directory)
        self._wake_deliver_hook: Any = None
        self._wake = SimpleNamespace(schedules=live)
        self.persisted: list[list[Any]] = []

    def _missed_delivery_note(self, _due: Any) -> None:
        return None

    async def _persist_wake_schedules(self, schedules: list[Any]) -> None:
        self.persisted.append(list(schedules))


def _due(text: str = "check the deploy", *, final: bool = False) -> Any:
    return SimpleNamespace(
        schedule=SimpleNamespace(id="w1", every_ms=None, message=text),
        occurrence=1,
        planned_total=None,
        final=final,
    )


def _final_due(text: str = "check the deploy") -> Any:
    """A fire that RETIRED its schedule: a real ``WakeSchedule``, so the hook can
    copy it into the one-shot the successor owes."""
    return SimpleNamespace(
        schedule=WakeSchedule(id="w1", message=text, next_due_at=1),
        occurrence=1,
        planned_total=1,
        final=True,
    )


@pytest.mark.asyncio
async def test_a_wake_fired_during_the_drain_lands_in_the_inbox(tmp_path: Path) -> None:
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    assert host._wake_deliver_hook == host._spool_wake_to_inbox

    await host._spool_wake_to_inbox(_due())
    rows = peek_inbox(tmp_path)
    assert len(rows) == 1
    assert "check the deploy" in rows[0].text
    assert rows[0].mode == "mailbox", "the same mailbox vehicle, never a new one"
    assert rows[0].wake is True, "a fired wake asked for a turn; the successor owes it one"


@pytest.mark.asyncio
async def test_a_retiring_schedule_is_re_armed_instead_of_spooled(tmp_path: Path) -> None:
    """MINOR 3, the sharp half: a fire that RETIRES its schedule leaves nothing
    able to engage the session.

    No index row survives it and no errand is raised for a schedule that has
    already fired, so a spooled note would keep the reminder and never run the
    work until a human opened the conversation. Re-armed as a one-shot due now,
    the supervisor starts a runtime for it and the successor folds it as an
    overdue occurrence.
    """
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())

    assert peek_inbox(tmp_path) == [], "the re-arm carries it; a note would double it"
    assert len(host._wake_rearms) == 1
    rearmed = host._wake_rearms[0]
    assert (rearmed.id, rearmed.message) == ("w1", "check the deploy"), "same handle, cancellable"
    assert rearmed.every_ms is None and rearmed.limit is None and rearmed.until_at is None
    assert rearmed.next_due_at > 0, "due now, so the supervisor engages on its next pass"


@pytest.mark.asyncio
async def test_a_wake_that_cannot_be_re_armed_is_still_spooled(tmp_path: Path) -> None:
    """The outcome this path must never produce: an occurrence that exists
    neither as a schedule nor as a spooled reminder.

    The fake's schedule here cannot be copied, standing in for any re-arm
    failure (a shape the harness does not model, an OOM between the two writes);
    the hook falls back to the note, loudly.
    """
    host = WakeHost(tmp_path)
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_due(final=True))

    rows = peek_inbox(tmp_path)
    assert len(rows) == 1 and rows[0].wake is True
    assert host._wake_rearms == [], "nothing to hand over, and nothing claimed"


class PersistHost:
    """The real ``Session._persist_wake_schedules`` over a recording transcript.

    The one piece of the session that review round 2's MINOR 1 turns on: the
    queue a fire leaves behind has to reach disk through the persist the
    scheduler runs right after that fire, not through an exit that may never
    come.
    """

    _persist_wake_schedules = Session._persist_wake_schedules
    retire_wakes_to_inbox = Session.retire_wakes_to_inbox
    _spool_wake_to_inbox = Session._spool_wake_to_inbox
    _queue_wake_rearm = Session._queue_wake_rearm

    def __init__(self) -> None:
        self._wake_rearms: list[Any] = []
        self._wake_fired_since_persist = True  # the delivery that just happened
        self._session_id = "s1"
        self._cwd = "/tmp"
        self.appended: list[list[Any]] = []
        self.indexed: list[list[Any]] = []
        self.supervisors = 0
        self._transcript = SimpleNamespace(append_custom=self._record_append)

    async def _record_append(self, _kind: str, payload: dict[str, Any]) -> None:
        self.appended.append(payload["schedules"])

    def _write_wake_index_entry(self, schedules: list[Any], **_kwargs: Any) -> None:
        self.indexed.append(list(schedules))

    def _ensure_wake_supervisor(self) -> None:
        self.supervisors += 1

    def _missed_delivery_note(self, _due: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_a_final_fires_re_arm_is_durable_without_the_exit() -> None:
    """MINOR 1 (review round 2): durable at FIRE time, not at the exit.

    The queue used to be written only by ``hand_wakes_to_successor``, so from
    the fire to an exit that is deliberately unbounded the occurrence existed
    only in memory: the socket ``stop`` op, a SIGTERM, a dispose or a crash in
    that window lost it, and the pump had already persisted the retire, so
    nothing retried it. The scheduler persists immediately after the delivery
    and that persist IS this method, so the queue joins whatever list it is
    handed. The exit write stays as the retry.
    """
    host = PersistHost()
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())
    assert host._wake_rearms, "the fire queued a re-arm"

    # The pump's own persist, right after that delivery: the list it hands over
    # is the post-retire one, WITHOUT the re-arm in it.
    await host._persist_wake_schedules([])

    assert len(host.appended) == 1
    # The transcript payload is the DUMPED form — that is what becomes durable.
    written = host.appended[0]
    assert [row["id"] for row in written] == ["w1"], "the re-arm rode the fire's persist"
    assert written[0]["every_ms"] is None and written[0]["limit"] is None, "and as the one-shot"
    assert [schedule.id for schedule in host.indexed[0]] == [
        "w1"
    ], "the index too, so the supervisor can engage"
    assert host.supervisors == 1, "which is the install-on-demand hook the index write feeds"


@pytest.mark.asyncio
async def test_the_durable_re_arm_supersedes_a_live_copy_of_the_same_id(tmp_path: Path) -> None:
    """One row per id on disk: the re-armed one-shot replaces the live schedule."""
    host = PersistHost()
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())
    live = WakeSchedule(id="w1", message="check the deploy", next_due_at=9, every_ms=3_600_000)
    other = WakeSchedule(id="w2", message="hourly", next_due_at=9, every_ms=3_600_000)

    await host._persist_wake_schedules([live, other])

    written = host.appended[0]
    assert [row["id"] for row in written] == ["w2", "w1"]
    assert written[1]["every_ms"] is None, "the surviving w1 is the re-arm, not the live copy"


@pytest.mark.asyncio
async def test_the_handover_writes_live_and_re_armed_wakes_once(tmp_path: Path) -> None:
    """The successor is engaged by the INDEX, so the re-arm has to reach it
    through the one writer of schedule state — with the same id appearing once,
    the live copy of a re-armed schedule dropped in favour of the one-shot."""
    live = WakeSchedule(id="w2", message="hourly", next_due_at=2, every_ms=3_600_000)
    superseded = WakeSchedule(id="w1", message="check the deploy", next_due_at=1)
    host = WakeHost(tmp_path, live=(live, superseded))
    host.retire_wakes_to_inbox()
    await host._spool_wake_to_inbox(_final_due())

    assert await host.hand_wakes_to_successor() == 1
    assert len(host.persisted) == 1, "one write, transcript-then-index"
    written = host.persisted[0]
    assert [schedule.id for schedule in written] == ["w2", "w1"], "one row per id, re-arm last"
    assert written[1].every_ms is None, "and the surviving w1 is the one-shot"
    assert await host.hand_wakes_to_successor() == 0, "handed over exactly once"


@pytest.mark.asyncio
async def test_an_unspoolable_wake_is_loud_and_does_not_raise(tmp_path: Path) -> None:
    """A drain must not die on a spool write, and a lost wake must leave a trace."""
    host = WakeHost(None)
    await host._spool_wake_to_inbox(_due())


@pytest.mark.asyncio
async def test_the_spooled_row_reaches_the_next_boot_drain(tmp_path: Path) -> None:
    """The end of the chain: what the drain spools is what the successor's boot
    drain delivers, in write order."""
    from local_operator.session.runtime.inbox import (
        InboxLine,
        append_inbox,
        drain_inbox,
    )

    assert append_inbox(
        tmp_path,
        InboxLine(text="wake fired while draining", sender={}, mode="mailbox", written_at=1.0),
    )
    lines = drain_inbox(tmp_path)
    assert [line.text for line in lines] == ["wake fired while draining"]
    assert lines[0].to_json()["mode"] == "mailbox"
