"""The update window: an idle handover QUEUES admissions instead of refusing them.

The incident this file answers (2026-09-19, live fleet): a session whose runtime
was still on 0.59.9 was reported by its own TUI as "it will switch to the new
version when it is next idle", and the operator's next message came back

    This session is leaving; it will not start a new turn. Your message is back in
    the composer — send it again once the session is running again.

The only recovery was ``/stop`` + ``/resume``, and the message had to be retyped.
The runtime was IDLE — it was leaving precisely because it had no work — so the
refusal protected nothing: the successor would have run the message had anyone
held it for them.

Four invariants are pinned here, in the order the spec states them:

1. an admission arriving while the window is open is SPOOLED and answered with the
   queued receipt, from every channel entry point (owner prompt, steer, peer wake);
2. the window is BOUNDED — with no heartbeat it fails open: the lock is released,
   the handover aborts, the runtime stays on the build it is running, and the
   failure is published with the incident cause token;
3. no admission path can block past ``UPDATE_LOCK_S``: they never touch the lock;
4. the successor publishes the one-shot "update applied" fact when it boots on the
   new build.

Driven the way ``test_serving_drain.py`` drives the drain latch: the production
methods over stub collaborators, so the assertion lands on the admission rather
than on a runtime boot (the e2e stage covers the boot).
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator import buildwatch
from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.inbox import (
    SOURCE_USER,
    SPOOL_RECEIPT_PROMPT,
    SPOOL_RECEIPT_WAKE,
    InboxLine,
    append_inbox,
    peek_inbox,
    read_update_window,
    write_update_window,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import (
    UPDATE_FAILED_CAUSE,
    UPDATING,
    UPDATING_DONE,
    update_phrase,
)
from local_operator.update import BuildStamp
from tests.unit.session.runtime.test_serving_drain import PromptHost, PromptSession

OLD = BuildStamp(version="0.59.9", source_ref="")
NEW = BuildStamp(version="0.59.11", source_ref="ead71b673a9a")
#: Built through the production formatter rather than typed, so the fixture carries
#: exactly the shape ``BuildStamp.label()`` produces (short ref, no brackets) — a
#: hand-typed pair would pin a string no runtime can emit.
PAIR = buildwatch.update_pair_text(OLD, NEW)


class WindowHost(PromptHost):
    """``PromptHost`` plus the window state the production ``__init__`` owns.

    The state is set here rather than reached through the real constructor
    because this file drives the admission decision, not a handle boot — the
    same split ``PromptHost`` makes for the prompt queue.
    """

    begin_update = ServingSessionHandle.begin_update
    heartbeat_update = ServingSessionHandle.heartbeat_update
    end_update = ServingSessionHandle.end_update
    # The three readers of the window's state, bound rather than re-implemented for
    # the reason every other production method here is: the cells below assert what
    # the RUNTIME publishes, not what a stub kept.
    note_update_failed = ServingSessionHandle.note_update_failed
    updating = ServingSessionHandle.updating
    update_failed_pair = ServingSessionHandle.update_failed_pair
    update_lock_remaining = ServingSessionHandle.update_lock_remaining
    lock_held = property(lambda self: self._update_lock.held)
    # The marker's own two-hop read of the session directory, bound for the reason
    # the other production methods here are: ``begin_update`` writes the marker
    # through it, and a stubbed directory would pin nothing about which one it means.
    _session_directory = ServingSessionHandle._session_directory

    def __init__(self, session: PromptSession, *, busy: bool = True) -> None:
        super().__init__(session, busy=busy)
        self._updating = ""
        self._update_failed = ""
        #: The pair whose ONE retry has been spent (``ServingSessionHandle``), owned
        #: by the production ``begin_update`` this host binds.
        self._update_retried = ""
        self._update_lock = buildwatch.UpdateLock()
        self._applied_update = ""


def _window_host(tmp_path: Path, *, busy: bool = False) -> tuple[WindowHost, PromptSession]:
    session = PromptSession(tmp_path / "sessions" / "s1", busy=busy)
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return WindowHost(session, busy=busy), session


# -- 1. the admission is QUEUED, from every channel ---------------------------


@pytest.mark.asyncio
async def test_a_prompt_during_the_window_is_queued_not_refused(tmp_path: Path) -> None:
    """The incident itself: the owner's message must be held, not handed back.

    The window state is set DIRECTLY rather than through ``begin_update`` so this
    cell runs on a tree without the window too — there ``_updating`` is ignored
    and the admission refuses, which is the red this asserts against.
    """
    host, session = _window_host(tmp_path)
    host._updating = PAIR

    receipt = await host.prompt("now summarise the build staleness fix", command_id="p" * 8)

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    assert receipt != "prompt admitted"
    assert session.prompt_calls == [], "a turn was started on a runtime that is leaving"
    rows = peek_inbox(session.transcript.directory)
    assert len(rows) == 1, rows
    assert rows[0].source == SOURCE_USER, "the successor must run it as the owner's own"
    assert rows[0].wake is True, "a user prompt asks for a turn"


@pytest.mark.asyncio
async def test_a_steer_during_the_window_is_queued_not_lost(tmp_path: Path) -> None:
    """A steer has no turn to join: the runtime is idle by construction.

    Handing it to ``Session.steer`` would queue it against a turn that will never
    run in this process, and the process exits moments later — the exact
    silent-loss shape the spool exists to prevent.
    """
    host, session = _window_host(tmp_path)
    host._updating = PAIR

    receipt = await host.steer("actually, use the other provider", command_id="s" * 8)

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt
    assert session.steered == [], "a steer was queued against a turn nobody will run"
    assert len(peek_inbox(session.transcript.directory)) == 1


@pytest.mark.asyncio
async def test_a_peer_wake_during_the_window_is_spooled(tmp_path: Path) -> None:
    """The peer channel is the one that already had the better answer."""
    host, session = _window_host(tmp_path)
    host._updating = PAIR

    receipt = await host.receive_peer_message("build is green", mode="wake", wake=True, sender={})

    assert receipt == SPOOL_RECEIPT_WAKE, receipt
    assert session.peer_calls == []
    assert len(peek_inbox(session.transcript.directory)) == 1


# -- 2. the bound: fail open, keep the old build, say so -----------------------


@pytest.mark.asyncio
async def test_a_stalled_window_fails_open_and_keeps_the_old_build(monkeypatch) -> None:
    """``UPDATE_LOCK_S`` with no heartbeat ends the attempt, never the runtime.

    The announce is the window's longest await (it drains each viewer's writer),
    so a viewer that never lets go is exactly the stall the bound is for. The
    assertion is the whole of the fail-open contract: the handover is ABANDONED
    (no disposal, no exit), the lock is RELEASED, the runtime keeps the build it
    is running, and the failure is published with the incident token so an
    operator can report it.
    """
    monkeypatch.setattr(child_mod, "_build_stagger_seconds", lambda: 0.0)
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: NEW)
    monkeypatch.setattr(buildwatch, "update_lock_seconds", lambda: 0.05)
    monkeypatch.setattr(child_mod, "update_lock_seconds", lambda: 0.05, raising=False)

    handle = _WindowHandle()
    runtime = _WindowRuntime(announce_delay=0.5)
    stop = asyncio.Event()

    exited = await child_mod._refresh_for(NEW, handle, runtime, stop)

    assert exited is False, "a stalled window must not take the exit"
    assert handle.disposed is False, "the runtime stays on the build it is running"
    assert stop.is_set() is False
    assert handle.updating == "", "the window is closed on the failure path"
    assert handle.lock_held is False, "the admission lock is released"
    assert runtime.failures == [PAIR], "the failure is published"
    assert runtime.retiring == [], "the latency must not be reported as an ordinary handover"
    assert runtime.announced == [], "a stalled announce never completed, so nothing was said"


@pytest.mark.asyncio
async def test_a_window_that_finishes_in_time_hands_over(monkeypatch) -> None:
    """The green half: a healthy window is not touched by the bound."""
    monkeypatch.setattr(child_mod, "_build_stagger_seconds", lambda: 0.0)
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: NEW)

    handle = _WindowHandle()
    runtime = _WindowRuntime(announce_delay=0.0)
    stop = asyncio.Event()

    exited = await child_mod._refresh_for(NEW, handle, runtime, stop)

    assert exited is True
    assert stop.is_set() is True
    assert runtime.failures == []
    assert handle.updating == PAIR, "the window stays open until the exit takes it"
    assert runtime.announced == ["stale-build"], "the announce must have gone out"


# -- 3. never deadlock: no admission path blocks on the lock -------------------


@pytest.mark.asyncio
async def test_an_admission_resolves_while_the_window_is_stalled(tmp_path: Path) -> None:
    """The no-deadlock invariant, exercised on the stall itself.

    ``prompt`` must answer while the window that is holding the lock is still
    stalled — not after it. ``asyncio.wait_for`` is the instrument rather than a
    wall-clock assertion: the admission coroutine either finishes inside the wake
    of the stalled task or it raised, and there is no budget to calibrate.
    """
    host, _session = _window_host(tmp_path)
    host._updating = PAIR

    async def stalled_window() -> None:
        await asyncio.sleep(10.0)

    stalled = asyncio.ensure_future(stalled_window())
    try:
        for _ in range(20):
            await asyncio.sleep(0)
            if stalled.done():  # pragma: no cover - a 10 s sleep cannot have finished
                raise AssertionError("the fixture's window did not stall")
        receipt = await asyncio.wait_for(
            host.prompt("held, not dropped", command_id="p" * 8), timeout=1.0
        )
    finally:
        stalled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stalled

    assert receipt == SPOOL_RECEIPT_PROMPT, receipt


def test_the_admission_paths_never_touch_the_update_lock() -> None:
    """The invariant's construct, pinned rather than timed.

    No portable numeric bound exists for "this never blocks" (AGENTS.md
    "Prefer a structural invariant to a numeric one"), and the property is
    structural here: the three admission paths read the window STRING and nothing
    else, so there is no lock for them to wait on. A future revision that reached
    for the lock to "check" it would reintroduce the deadlock the spec forbids,
    and no timing test would catch it on an idle box.
    """
    for method in (
        ServingSessionHandle.prompt,
        ServingSessionHandle.steer,
        ServingSessionHandle.receive_peer_message,
    ):
        source = inspect.getsource(method)
        assert "update_lock" not in source.lower(), (
            f"{method.__name__} reaches for the update lock; an admission must read the "
            f"window string and never wait on the lock that holds it"
        )


# -- 4. the completion fact ----------------------------------------------------


def test_the_successor_publishes_the_applied_update(tmp_path: Path) -> None:
    """The successor's boot is where the queued messages run, so it is where the
    "it worked" fact belongs — one read, once, and the marker is consumed."""
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    write_update_window(directory, PAIR)
    handle = _BootHandle(directory)

    assert child_mod._consume_update_marker(handle) == PAIR

    assert handle.applied_update == PAIR
    assert read_update_window(directory) == "", "the fact is one-shot: the marker is consumed"


def test_a_boot_with_no_marker_publishes_nothing(tmp_path: Path) -> None:
    """The negative control: every ordinary boot must stay silent."""
    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    handle = _BootHandle(directory)

    assert child_mod._consume_update_marker(handle) == ""
    assert handle.applied_update == ""


def test_the_retiring_frame_carries_the_window_off_the_record() -> None:
    """The frame key is read off the RECORD, not taken as an argument.

    A parameter would be a second copy of a field the server already holds, and — the
    part that was measured — a caller whose ``announce_retiring`` predates the
    parameter takes a ``TypeError`` inside the ``except Exception`` that guards a
    viewer's writer, so the ENTIRE announcement is swallowed by the failure path meant
    for something else (``test_process_refresh``'s fake registrant lost its only frame
    exactly that way). The window is published on the record before the announce, so
    the record is the one place both ends can read it from.
    """
    import inspect

    from local_operator.session.runtime.server import RuntimeServer

    source = inspect.getsource(RuntimeServer._announce_retiring_on_loop)
    assert '"updating": self._record.updating,' in source, source


def test_the_record_seeds_the_applied_fact_from_the_handle() -> None:
    """The wire half: a session record must carry the fact the fleet surfaces read.

    Read off the source because constructing a ``RuntimeServer`` boot is the e2e
    stage's job, and the property here is that the ONE writer of the record
    consults the handle's boot fact at all.
    """
    from local_operator.session.runtime.server import RuntimeServer

    source = inspect.getsource(RuntimeServer.__init__)
    assert "applied_update" in source, (
        "the record is the only place a fleet surface can read the applied-update fact "
        "and it is built in RuntimeServer.__init__; nothing there reads the handle's"
    )


# -- the lock itself -----------------------------------------------------------


def test_the_lock_is_dead_after_the_bound_without_a_heartbeat() -> None:
    """The bound is on the HEARTBEAT, so a live window is never mistaken for one."""
    lock = buildwatch.UpdateLock()
    assert lock.acquire(PAIR, "stale-build") is True
    opened = lock.last_heartbeat()
    assert lock.expired(now=opened + buildwatch.UPDATE_LOCK_S - 0.01) is False
    assert lock.expired(now=opened + buildwatch.UPDATE_LOCK_S + 0.01) is True

    lock.heartbeat(now=opened + buildwatch.UPDATE_LOCK_S - 0.01)
    assert (
        lock.expired(now=opened + buildwatch.UPDATE_LOCK_S + 0.01) is False
    ), "a heartbeat inside the bound must move the deadline"
    assert (
        lock.heartbeat_interval() <= buildwatch.UPDATE_LOCK_S
    ), "a heartbeat interval at or above the bound can never keep a live lock alive"


def test_a_second_holder_cannot_take_the_lock() -> None:
    """One window at a time, and a dead one is recoverable rather than quarantined."""
    lock = buildwatch.UpdateLock()
    assert lock.acquire(PAIR) is True
    assert lock.acquire("other") is False
    lock.release()
    assert lock.acquire("other") is True


@pytest.mark.asyncio
async def test_a_failed_pair_gets_one_retry_and_then_stops(tmp_path: Path) -> None:
    """The anti-churn rule, with a retry rather than a dead flag.

    The first version of this took ``retry_failed=True`` from an "explicit operator
    refresh" that does not exist in the tree, so a failed window was never tried
    again and the pair stayed refused until the process EXITED — the stale session
    the incident was about, made permanent (agent review round 1, NIT 1). One retry
    answers that: a transient stall is survivable, and a move that fails the bound
    twice stops being retried, because a third attempt repeats the second.
    """
    host, _session = _window_host(tmp_path)
    assert host.begin_update(PAIR) is True
    host.end_update()
    host.note_update_failed(PAIR, buildwatch.UPDATE_LOCK_S)

    assert host.begin_update(PAIR) is True, "the first failure must not be final"
    host.end_update()
    host.note_update_failed(PAIR, buildwatch.UPDATE_LOCK_S)

    assert host.begin_update(PAIR) is False, "a second failure for one pair is final"
    assert (
        "retry_failed" not in inspect.signature(ServingSessionHandle.begin_update).parameters
    ), "the flag that had no caller must not come back as a parameter nothing passes"


# -- the two arms of a window that did NOT hand over (agent review round 1, MAJOR 1) ---
#
# ``_announce_and_latch`` returns False for two facts that need OPPOSITE answers, and
# the stop arm is the one that destroyed a message: the window's spooled row was
# drained into a runtime that was taking its exit, so the owner's text left
# ``inbox.jsonl`` for a queue ``dispose`` then rejected — while the receipt told them
# the successor would run it.


@pytest.mark.asyncio
async def test_the_stop_arm_leaves_the_spooled_message_for_the_next_boot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """MAJOR 1: a stop that lands mid-announce must not eat the queue.

    The rig is the incident's own shape: the window is open while the announce runs,
    so the owner's prompt arrives in it and is SPOOLED. Then a stop lands — the
    supervisor's SIGTERM, or the operator's ``/stop`` — and this runtime is going
    away. The row is the next boot's to deliver, which is exactly what the receipt
    promised; running it here runs it in a process that is about to disappear.
    """
    _patch_build(monkeypatch, bound=5.0)
    host, session = _handover_host(tmp_path)
    stop = asyncio.Event()
    drained: list[object] = []

    async def record_drain(handle: object) -> int:
        drained.append(handle)
        return 0

    monkeypatch.setattr(child_mod, "_drain_inbox_into", record_drain)
    runtime = _SpoolingRuntime(message="now summarise the build staleness fix", press_stop=stop)
    runtime.handle = host

    exited = await child_mod._refresh_for(NEW, host, runtime, stop)

    assert drained == [], (
        "the stop arm must NOT drain the spool back in: that is what destroyed the "
        "operator's message (agent review round 1, MAJOR 1)"
    )
    assert exited is False, "the stop's own path owns the exit, not this rung"
    assert stop.is_set() is True
    assert runtime.receipt == SPOOL_RECEIPT_PROMPT, runtime.receipt
    assert session.prompt_calls == [], (
        "the message was run on a runtime that is EXITING: ``dispose`` rejects exactly "
        "that admission, so it would exist in no transcript at all"
    )
    rows = peek_inbox(session.transcript.directory)
    assert len(rows) == 1, "the spool belongs to the next boot, which is the reader promised"
    assert rows[0].text == runtime.message, rows[0].text
    assert rows[0].source == SOURCE_USER
    assert host.updating == "", "the window still closes: it queues for nobody now"
    assert host.lock_held is False, "and the lock goes with it"
    assert runtime.failures == [], "a stop is not a failed update"


@pytest.mark.asyncio
async def test_the_keep_arm_runs_the_spooled_message_here(tmp_path: Path, monkeypatch: Any) -> None:
    """The other arm, and the reason the two had to be told apart.

    Work arrived during the announce, so the runtime KEEPS: the window closes and
    the message it was holding is the runtime's own again — the only writer that
    owes it a turn. It lands either as a prompt or, if the work that arrived is a
    turn still in flight, as a steer; both are the same guarantee and which one it
    is is the mid-turn rule ``_run_owner_prompt`` documents.
    """
    _patch_build(monkeypatch, bound=5.0)
    host, session = _handover_host(tmp_path)
    drained: list[object] = []

    async def record_drain(handle: object) -> int:
        drained.append(handle)
        return 0

    monkeypatch.setattr(child_mod, "_drain_inbox_into", record_drain)
    # ``busy`` flips DURING the announce, which is the race ``begin_retire`` exists to
    # close: the idle sample that admitted this rung is not true any more.
    runtime = _SpoolingRuntime(message="carry on with the other provider", then_busy=host)
    runtime.handle = host

    exited = await child_mod._refresh_for(NEW, host, runtime, asyncio.Event())

    assert exited is False
    assert host._busy is True, "premise: work arrived, so this runtime is keeping"
    assert drained == [host], (
        "the keep arm hands the spool back to the ONLY writer that owes it a turn — " "this runtime"
    )
    assert host.updating == ""
    # And the delivery itself is the drain's own contract, pinned below on a real
    # session directory rather than through a stub admission.
    assert session.prompt_calls == [], "premise: the spool is delivered by the drain"


# -- the spool on an unengaged session (agent review round 1, MAJOR 2) ----------------


@pytest.mark.asyncio
async def test_a_pristine_session_runs_the_message_the_window_queued(tmp_path: Path) -> None:
    """MAJOR 2: no durable history is not "nobody is here".

    ``_drain_inbox_into`` returned 0 without draining when the transcript had no real
    turn, which is right for a PEER row on a cold session and wrong for the owner's
    own: the row exists because the owner typed it into a runtime that was moving,
    and nothing else in the process will run it — the once-per-lifetime first-turn
    drain only fires if they type a SECOND message.
    """
    # No transcript file at all: the pristine case.
    entity = _DrainEntity(tmp_path)
    append_inbox(
        entity.directory,
        InboxLine(text="ship it", sender={}, wake=True, source=SOURCE_USER, command_id="c" * 8),
    )

    delivered = await child_mod._drain_inbox_into(entity)

    assert delivered == 1, "the gate returned 0 without draining, so nothing ran the row"
    assert entity.prompt_calls + entity.steered == [
        "ship it"
    ], f"the owner's message never ran: {entity.prompt_calls}, {entity.steered}"
    assert peek_inbox(entity.directory) == []


@pytest.mark.asyncio
async def test_a_pristine_session_still_defers_a_peer_row(tmp_path: Path) -> None:
    """The gate this change scoped rather than removed, pinned as it was.

    A peer row on a session whose owner has not started a conversation must not open
    its history: the boot drain runs before the socket listens, which is also before
    the owner's first turn. The row is put BACK — ``drain_inbox`` empties the file by
    contract, so a reader that discards what it will not deliver has destroyed it.
    """
    entity = _DrainEntity(tmp_path)
    append_inbox(
        entity.directory,
        InboxLine(text="build is green", sender={"session_id": "peer"}, wake=False),
    )

    delivered = await child_mod._drain_inbox_into(entity)

    assert delivered == 0, "a peer note must not drive a turn on a cold session"
    assert entity.peer_calls == []
    left = peek_inbox(entity.directory)
    assert [row.text for row in left] == ["build is green"], left


@pytest.mark.asyncio
async def test_a_row_that_cannot_be_delivered_is_kept_rather_than_destroyed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """MAJOR 2's sibling (agent review round 1, MINOR 5): the failure arm.

    Measured by the reviewer with a host that reached delivery and then raised: the
    row was consumed by the read and gone, with an ERROR line as its only trace —
    while the sender's receipt said the successor would run it. The file's contract
    is at-least-once and the durable command index makes a redelivery idempotent, so
    putting the row back is the correct failure behaviour rather than a hopeful one.
    """
    entity = _DrainEntity(tmp_path, fail_delivery=True)
    append_inbox(
        entity.directory,
        InboxLine(text="keep me", sender={}, wake=True, source=SOURCE_USER, command_id="d" * 8),
    )

    delivered = await child_mod._drain_inbox_into(entity)

    assert delivered == 0
    left = peek_inbox(entity.directory)
    assert [row.command_id for row in left] == [
        "d" * 8
    ], "an undelivered row must still exist somewhere: it is the only copy"


@pytest.mark.asyncio
async def test_the_abandon_arm_runs_the_spool_on_a_pristine_session(tmp_path: Path) -> None:
    """Both arms of the pristine gate, because MAJOR 2 measured both.

    The abandon arm is the one that runs on the OWNER's own machine without a
    successor: the bound expires, the runtime keeps its build, and the message it
    promised to hold must run HERE — on a session whose transcript has no turn yet,
    which is where the gate used to drop it.
    """
    entity = _DrainEntity(tmp_path)
    append_inbox(
        entity.directory,
        InboxLine(text="run it here", sender={}, wake=True, source=SOURCE_USER, command_id="e" * 8),
    )
    runtime = _SpoolingRuntime()

    await child_mod._abandon_update_window(entity, runtime, PAIR, 0.05)

    assert entity.prompt_calls + entity.steered == [
        "run it here"
    ], f"the fail-open arm must deliver what it promised: {entity.prompt_calls}"
    assert peek_inbox(entity.directory) == []
    assert runtime.failures == [PAIR], "and it is published as a failure"


# -- the exit leg is bounded too (agent review round 1, MINOR 1) ---------------------


@pytest.mark.asyncio
async def test_the_exit_leg_is_bounded_without_cancelling_the_exit(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The window must not outlive the bound while the process disposes.

    ``_clean_exit`` awaits the turn abort and every viewer's writer — the stage that
    measured at MINUTES in the incident — and the window used to stay open and beating
    across all of it, handing out receipts for a successor that had not been spawned
    yet. It is bounded now, and the exit is still never cancelled: a dispose cut in
    half is worse than a slow one, so the window's promise is what gets withdrawn.
    """
    _patch_build(monkeypatch, bound=0.05)
    host, _session = _handover_host(tmp_path, dispose_delay=0.4)
    runtime = _SpoolingRuntime()

    exited = await child_mod._refresh_for(NEW, host, runtime, asyncio.Event())

    assert exited is True, "a slow exit is still an exit"
    assert host.disposed is True, "the dispose ran to completion"
    assert runtime.retiring == ["stale-build"], "the handover was announced and latched"
    assert host.updating == "", "the window is retracted once its bound is spent"
    assert host.lock_held is False, "and the lock is released with it"


# -- the bound is the HEARTBEAT's (agent review round 1, MINOR 2) --------------------


@pytest.mark.asyncio
async def test_a_handover_that_keeps_beating_is_not_bounded(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A slow handover that keeps proving progress must NOT be failed.

    This is the cell that makes the pump load-bearing: with the beats arriving, the
    lock's deadline keeps moving, so an announce that runs four times the bound is a
    healthy handover rather than a stalled one. Deleting the pump (the reviewer's own
    mutation) makes this cell fail, which is the property the previous shape lacked —
    there, the timeout was a total duration and no beat decided anything.
    """
    _patch_build(monkeypatch, bound=0.05, heartbeat=0.005)
    host, _session = _handover_host(tmp_path)
    runtime = _SpoolingRuntime(announce_delay=0.2)

    exited = await child_mod._refresh_for(NEW, host, runtime, asyncio.Event())

    assert runtime.retiring == ["stale-build"], "the announce completed"
    assert exited is True, "a beating handover takes the exit however long it takes"
    assert runtime.failures == [], "no failure is published for a live window"


@pytest.mark.asyncio
async def test_the_window_expires_when_the_beats_stop(tmp_path: Path, monkeypatch: Any) -> None:
    """The same rig, with the pump removed: the bound is the beats, or it is nothing.

    Written as its own cell rather than as a comment about the one above, because the
    reviewer's measurement was exactly this: with ``_pump_update_heartbeat`` returning
    immediately, EVERY cell stayed green. Here the difference is the assertion.
    """
    _patch_build(monkeypatch, bound=0.05, heartbeat=0.005)
    monkeypatch.setattr(child_mod, "_pump_update_heartbeat", _dead_pump)
    host, _session = _handover_host(tmp_path)
    runtime = _SpoolingRuntime(announce_delay=0.2)

    exited = await child_mod._refresh_for(NEW, host, runtime, asyncio.Event())

    assert exited is False, "with no beats the window is DEAD at the bound"
    assert runtime.failures == [PAIR], "and the failure is published"
    assert host.lock_held is False


# -- the marker (agent review round 1, MINOR 4) --------------------------------------


def test_the_marker_survives_two_writers_in_one_directory(tmp_path: Path) -> None:
    """Two runtimes can serve one session directory: the temporary must not be shared.

    Measured by the reviewer at 2 threads x 3000 writes against the old
    ``path.with_suffix('.tmp')``: 59 reads of an absent-or-corrupt marker and 2374
    writes reporting failure, because both writers renamed ONE temporary. The silent
    half is the worse one — a writer whose ``os.replace`` installs the other's
    truncated file returns ``True``, and the successor then publishes no ``updated``
    fact at all. ``mkstemp`` gives each writer its own name and O_EXCL.
    """
    import threading

    directory = tmp_path / "sessions" / "s1"
    directory.mkdir(parents=True)
    failures: list[bool] = []
    corrupt: list[str] = []

    def hammer(pair: str) -> None:
        for _ in range(300):
            if not write_update_window(directory, pair):
                failures.append(False)
            read = read_update_window(directory)
            if read not in (PAIR, "0.59.10 → 0.59.11@ead71b6"):
                corrupt.append(read)

    threads = [
        threading.Thread(target=hammer, args=(PAIR,)),
        threading.Thread(target=hammer, args=("0.59.10 → 0.59.11@ead71b6",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == [], f"{len(failures)} writes could not install their marker"
    assert corrupt == [], f"{len(corrupt)} reads saw an absent or malformed marker"


@pytest.mark.asyncio
async def test_a_window_cannot_be_opened_without_a_pair(tmp_path: Path) -> None:
    """``""`` is the sentinel AND the gate, so a window must never hold it.

    Agent review round 1 (NIT 2): a pair-less window would be invisible on every
    surface and would queue nothing, while the sender still got a receipt for it — an
    unreachable state today, and one the sentinel makes reachable tomorrow.
    """
    host, _session = _handover_host(tmp_path)

    assert host.begin_update("") is False
    assert host.updating == ""
    assert host.lock_held is False


def test_opening_a_window_clears_the_previous_failure() -> None:
    """Agent review round 1 (NIT 4): a record must not describe an abandoned move forever.

    Nothing else clears the field — the successor's record never had it, and the arm
    that writes it is the one that keeps serving — so a retry that succeeds would leave
    the fleet saying "update failed" about a session that had moved on.
    """
    from local_operator.session.runtime.server import RuntimeServer

    record = SimpleNamespace(updating="", update_failed=PAIR)
    stub = SimpleNamespace(_record=record, _updating="", _republish=lambda: None, _failed_retries=0)

    RuntimeServer.note_updating(cast("RuntimeServer", stub), PAIR)

    assert record.updating == PAIR
    assert record.update_failed == "", "a new window supersedes the last failure"


# -- the record field ----------------------------------------------------------


def test_the_record_carries_the_window_and_the_failure() -> None:
    """Additive fields, and the pair the surfaces render.

    ``PROTOCOL_VERSION`` deliberately does not move: an older reader drops unknown
    keys (``SessionRecord.from_json``), which is the contract every live-state
    field on this record already keeps.
    """
    from local_operator.session.runtime.types import PROTOCOL_VERSION, SessionRecord

    def _record() -> SessionRecord:
        """A fresh record from LITERALS: the dataclass is typed, and a
        ``dict[str, int | str]`` splatted into it reads to the type checker as
        thirty errors rather than as the one field this cell is about."""
        return SessionRecord(
            pid=1,
            kind="exec",
            session_id="s",
            conversation_name="c",
            cwd="/tmp",
            model_label="m",
            control_port=1,
            control_key="k",
        )

    record = _record()
    assert record.updating == "" and record.update_failed == "" and record.updated == ""

    record.updating = PAIR
    record.update_failed = PAIR
    record.updated = PAIR
    round_tripped = SessionRecord.from_json(record.to_json())
    assert round_tripped.updating == PAIR
    assert round_tripped.update_failed == PAIR, (
        "the failure has to survive the wire: it is published on the record a runtime "
        "that STAYED owns, and a fleet surface reads it after the fact"
    )
    assert round_tripped.updated == PAIR
    assert (
        PROTOCOL_VERSION == _record().protocol
    ), "an additive field must not spend the one number that gates frames"


def test_the_phrase_vocabulary_renders_the_pair_once() -> None:
    """One formatter, so the surfaces cannot disagree about the same window."""
    assert PAIR == "0.59.9 → 0.59.11@ead71b6", PAIR
    assert UPDATING in update_phrase(UPDATING, PAIR)
    assert PAIR in update_phrase(UPDATING, PAIR)
    assert PAIR in update_phrase(UPDATING_DONE, PAIR)
    # A runtime that could not read a stamp still says something true.
    assert "build on disk" in update_phrase(UPDATING, "")
    assert UPDATE_FAILED_CAUSE == "runtime-update-failed"


# -- the process-level rig -----------------------------------------------------


class _DrainEntity:
    """The drain's own collaborators over a REAL session directory.

    Deliberately not ``WindowHost``: the cells that use this are about the GATE (which
    rows a drain may deliver on a session nobody has engaged yet) and about a row's
    fate when delivery FAILS — and the production ``prompt`` needs a whole runtime
    behind it (it names the conversation, then pumps a queue against a session that
    can ``subscribe``), which those two questions never touch. The directory, the
    inbox and the drain are all production; only where the message lands is recorded.
    """

    def __init__(self, root: Path, *, fail_delivery: bool = False) -> None:
        self._session = SimpleNamespace(
            transcript=SimpleNamespace(directory=root / "sessions" / "s1")
        )
        self.directory = self._session.transcript.directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fail_delivery = fail_delivery
        self.prompt_calls: list[str] = []
        self.steered: list[str] = []
        self.peer_calls: list[str] = []

    def has_admitted_command(self, command_id: str) -> bool:
        return False

    async def prompt(self, text: str, images: Any = None, command_id: str | None = None) -> None:
        if self.fail_delivery:
            raise RuntimeError("the session refused the row")
        self.prompt_calls.append(text)

    async def steer(self, text: str, images: Any = None, **kwargs: Any) -> None:
        self.steered.append(text)

    async def receive_peer_message(self, text: str, **kwargs: Any) -> str:
        self.peer_calls.append(text)
        return "delivered"


class _HandoverHost(WindowHost):
    """``WindowHost`` plus the rung's other collaborators: the gate, the latch, the exit.

    The production methods over a stub session, the way ``test_serving_drain`` builds
    its hosts, so the assertions land on the rung's ORDERING rather than on a boot.
    """

    begin_retire = ServingSessionHandle.begin_retire

    def __init__(self, session: PromptSession, *, busy: bool = False, dispose_delay: float = 0.0):
        super().__init__(session, busy=busy)
        self._draining = False
        self._disposing = False
        self._exit_committed = False
        self._retiring_cause = ""
        self._retiring_detail = ""
        self.dispose_delay = dispose_delay
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1
        await asyncio.sleep(self.dispose_delay)
        self.disposed = True


class _SpoolingRuntime:
    """A handover whose announce takes a message: the incident's own shape.

    The window is OPEN while the announce runs — that is what it was opened for — so
    a prompt that arrives during it goes through the production admission and comes
    back with the spool receipt. What happens to that row afterwards is the fact the
    two arms disagree about, and this rig is the same one for both.
    """

    _boot_build = OLD

    def __init__(
        self,
        *,
        announce_delay: float = 0.0,
        message: str = "",
        press_stop: "asyncio.Event | None" = None,
        then_busy: Any = None,
    ) -> None:
        self.announce_delay = announce_delay
        self.message = message
        self.press_stop = press_stop
        self.then_busy = then_busy
        #: The handle whose admission the announce uses. Named on the runtime rather
        #: than passed in because the rung's own seam gives ``announce_retiring`` no
        #: handle — the frame's pair rides the RECORD — which is the shape modelled.
        self.handle: Any = None
        self.receipt = ""
        self.retiring: list[str] = []
        self.failures: list[str] = []

    async def announce_retiring(
        self,
        reason: str,
        *,
        to: str = "",
        draining: bool = False,
        leaving: str = "",
        updating: str = "",
    ) -> None:
        await asyncio.sleep(self.announce_delay)
        self.retiring.append(reason)
        if self.message and self.receipt == "":
            # A message arriving DURING the announce: the window is open, so this is
            # the production admission answering with the spool receipt.
            self.receipt = await self.handle.prompt(self.message, command_id="a" * 8)
        if self.press_stop is not None:
            # The stop lands between the spool and the latch — the ordering the two
            # arms of ``_announce_and_latch`` are decided by.
            self.press_stop.set()
        if self.then_busy is not None:
            self.then_busy._busy = True

    async def note_update_failed(self, pair: str, bound: float) -> None:
        """The rung's publication seam, recorded rather than written to a journal."""
        self.failures.append(pair)


def _handover_host(
    tmp_path: Path, *, busy: bool = False, dispose_delay: float = 0.0
) -> "tuple[_HandoverHost, PromptSession]":
    session = PromptSession(tmp_path / "sessions" / "s1", busy=busy)
    session.transcript.directory.mkdir(parents=True, exist_ok=True)
    return _HandoverHost(session, busy=busy, dispose_delay=dispose_delay), session


async def _dead_pump(handle: object, *, interval: float) -> None:
    """``_pump_update_heartbeat`` with the beats removed — the reviewer's own mutation."""
    return


def _patch_build(monkeypatch: Any, *, bound: float, heartbeat: float = 1.0) -> None:
    """Pin the rung's build decision and shorten its bound for one cell."""
    monkeypatch.setattr(child_mod, "_build_stagger_seconds", lambda: 0.0)
    monkeypatch.setattr(child_mod, "_build_changed", lambda _boot: NEW)
    monkeypatch.setattr(buildwatch, "update_lock_seconds", lambda: bound)
    monkeypatch.setattr(buildwatch, "update_lock_heartbeat_seconds", lambda: heartbeat)


class _WindowHandle:
    """The window API the refresh rung drives, over no session at all.

    Deliberately NOT the production handle: these cells are about the rung's
    ORDERING (open before the announce, close on either outcome), which a boot
    would bury in a provider.
    """

    def __init__(self, *, dispose_delay: float = 0.0) -> None:
        self.updating = ""
        self.lock = buildwatch.UpdateLock()
        self.disposed = False
        self.spool_drains = 0
        #: How long the dispose takes. The exit leg's bound is the cell that needs
        #: this to be a number rather than a speed (agent review round 1, MINOR 1).
        self.dispose_delay = dispose_delay

    def may_refresh(self) -> str:
        return ""

    def begin_update(self, pair: str, holder: str = "") -> bool:
        if self.lock.acquire(pair, holder):
            self.updating = pair
            return True
        return False

    def heartbeat_update(self) -> None:
        self.lock.heartbeat()

    def end_update(self) -> bool:
        self.updating = ""
        return self.lock.release()

    @property
    def lock_held(self) -> bool:
        return self.lock.held

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        return True

    async def dispose(self) -> None:
        await asyncio.sleep(self.dispose_delay)
        self.disposed = True


class _WindowRuntime:
    """The two things the rung talks to: the announce and the failure record."""

    _boot_build = OLD

    def __init__(self, *, announce_delay: float) -> None:
        self.announce_delay = announce_delay
        self.retiring: list[tuple[str, str, bool, str]] = []
        #: Every announce that completed, in order — so a cell can assert the frame
        #: went out at all (the window's pair rides the RECORD, not this call).
        self.announced: list[str] = []
        self.failures: list[str] = []

    async def announce_retiring(
        self,
        reason: str,
        *,
        to: str = "",
        draining: bool = False,
        leaving: str = "",
        updating: str = "",
    ) -> None:
        await asyncio.sleep(self.announce_delay)
        self.retiring.append((reason, to, draining, leaving))
        self.announced.append(reason)

    async def note_update_failed(self, pair: str, bound: float) -> None:
        self.failures.append(pair)


class _BootHandle:
    """The boot half: a session directory, and the fact the server will seed."""

    def __init__(self, directory: Path) -> None:
        self._session = SimpleNamespace(transcript=SimpleNamespace(directory=directory))
        self.applied_update = ""
