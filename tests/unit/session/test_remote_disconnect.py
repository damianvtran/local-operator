"""Deferred abort on transient viewer disconnect (design §5.2 tests 16–19).

A dropped socket is not an abort: the runtime is usually still running the
turn. These pin that ``_on_disconnected`` no longer synthesises an
``AgentEndEvent``, and that recovery's verdict is the only writer of one.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

import local_operator.session.attached as remote_module
from local_operator.harness.types import AgentEndEvent, AgentStartEvent
from local_operator.session.attached import AttachedSession
from local_operator.session.frontend_state import FrontendSessionState


def _facade(tmp_path, monkeypatch, *, can_go_cold: bool = False) -> AttachedSession:
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._can_go_cold = can_go_cold
    remote._streaming = True
    remote._generation = 7
    remote._ready_for_events = True
    return remote


async def _cancel_recovery(remote: AttachedSession) -> None:
    if remote._recovery_task is not None:
        remote._recovery_task.cancel()
        try:
            await remote._recovery_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown only
            pass


@pytest.mark.asyncio
async def test_mid_turn_socket_close_does_not_synthesise_an_end(tmp_path, monkeypatch) -> None:
    """Test 16a: mid-turn socket close → no AgentEndEvent; ``_streaming`` stays True."""
    remote = _facade(tmp_path, monkeypatch)
    received: list[Any] = []
    remote.subscribe(received.append)

    remote._on_disconnected("send timeout")
    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert ends == [], f"a dropped socket synthesised {ends}"
    assert remote.is_streaming is True
    assert remote._suspect_generation == 7
    assert remote._recovering is True
    await _cancel_recovery(remote)


@pytest.mark.asyncio
async def test_rebind_to_the_same_live_generation_synthesises_nothing(
    tmp_path, monkeypatch
) -> None:
    """Test 16b: re-bind with streaming True and generation == suspect → no end.

    Deviation from the design's "seed AgentStartEvent": re-seeding start
    would run ``_handle_agent_start``, which clears ``_started_tools``. A
    later real ``tool_end`` would then miss the live card and
    ``_finalize_turn`` would paint ⊘ interrupted — the bug this PR exists
    to stop. Generation is already applied from the snapshot; the ledger
    is left untouched.
    """
    remote = _facade(tmp_path, monkeypatch)
    received: list[Any] = []
    remote.subscribe(received.append)
    remote._on_disconnected("send timeout")
    await _cancel_recovery(remote)

    remote._install_frontend(
        FrontendSessionState(
            session_id="s1",
            epoch="e1",
            streaming=True,
            generation=7,
            live_events=[
                {
                    "type": "tool_execution_start",
                    "tool_call_id": "t1",
                    "tool_name": "hang",
                    "args": {},
                }
            ],
        )
    )
    remote._settle_suspect_turn()
    remote._finish_sync()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    starts = [event for event in received if isinstance(event, AgentStartEvent)]
    assert ends == []
    assert starts == [], f"same-live-turn rebind re-seeded starts {starts}"
    assert remote.is_streaming is True
    assert remote._suspect_generation is None
    # The gap's own rows DO come through — only the synthetic start is
    # suppressed (review round 1, MAJOR-1; the sibling test below pins the
    # positive case). ``_is_duplicate`` keeps rows the ledger already holds
    # from painting twice.
    types = [getattr(event, "type", None) for event in received]
    assert "agent_start" not in types


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, aborted, error",
    [
        ("completed", False, None),
        ("aborted", True, None),
        ("error", False, "turn failed"),
        ("", True, None),
    ],
    ids=["completed", "aborted", "error", "legacy-empty"],
)
async def test_rebind_after_the_turn_ended_synthesises_one_matching_end(
    tmp_path, monkeypatch, outcome: str, aborted: bool, error: str | None
) -> None:
    """Test 17: re-bind with streaming False → exactly one end, aborted per outcome."""
    remote = _facade(tmp_path, monkeypatch)
    received: list[Any] = []
    remote.subscribe(received.append)
    remote._on_disconnected("send timeout")
    await _cancel_recovery(remote)

    remote._install_frontend(
        FrontendSessionState(
            session_id="s1",
            epoch="e1",
            streaming=False,
            generation=7,
            last_turn_outcome=outcome,  # type: ignore[arg-type]
        )
    )
    remote._settle_suspect_turn()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1, f"expected one end, got {ends}"
    assert ends[0].aborted is aborted
    assert ends[0].error == error
    assert ends[0].generation == 0
    assert remote.is_streaming is False
    assert remote._suspect_generation is None


@pytest.mark.asyncio
async def test_rebind_to_a_newer_generation_ends_the_suspect_not_the_successor(
    tmp_path, monkeypatch
) -> None:
    """Test 17 (generation moved): synthesise one end, keep the successor live."""
    remote = _facade(tmp_path, monkeypatch)
    received: list[Any] = []
    remote.subscribe(received.append)
    remote._on_disconnected("send timeout")
    await _cancel_recovery(remote)

    remote._install_frontend(
        FrontendSessionState(
            session_id="s1",
            epoch="e1",
            streaming=True,
            generation=8,
            last_turn_outcome="completed",
        )
    )
    remote._settle_suspect_turn()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1 and ends[0].aborted is False
    assert remote.is_streaming is True
    assert remote._generation == 8


@pytest.mark.asyncio
async def test_recovery_going_cold_emits_one_cut_off_end(tmp_path, monkeypatch) -> None:
    """Test 18: recovery goes cold → one CUT-OFF end (exactly once).

    The exactly-once property is what this test owns and it is unchanged. The
    KIND changed deliberately: a confirmed owner death is the case the cut-off
    taxonomy exists for, and it used to be published as a bare abort — the same
    shape a user's own ``/stop`` produces.
    """
    remote = _facade(tmp_path, monkeypatch, can_go_cold=True)
    received: list[Any] = []
    remote.subscribe(received.append)
    remote._on_disconnected("owner exited")
    await _cancel_recovery(remote)
    remote._go_cold()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1
    assert ends[0].aborted is False
    assert ends[0].cut_off_cause == "owner-lost"
    assert remote.is_streaming is False


@pytest.mark.asyncio
async def test_abort_while_detached_does_not_spawn_a_task(tmp_path, monkeypatch) -> None:
    """Test 19: ``abort()`` while detached → no task, no 'never retrieved'."""
    remote = _facade(tmp_path, monkeypatch)
    remote._client = None
    # ``connected`` False on a leftover client is the other half of the guard.
    remote.abort("interrupted")
    await asyncio.sleep(0)
    # No unretrieved exception: the done-callback (or the early return) ate it.
    assert remote._client is None


@pytest.mark.asyncio
async def test_rebind_after_an_idle_runtime_posts_turn_ended(tmp_path, monkeypatch) -> None:
    """PR C deferred: TUI kept painting working after the runtime went idle.

    Disconnect mid-turn, then rebind to a snapshot with ``streaming=False``
    and ``last_turn_outcome="completed"``. The synthesised end must reach
    EventController so the band can leave ``working``/``thinking``. An
    unstamped end (generation=0) is what the controller accepts for the
    open turn; a missing end is the stuck-indicator bug.
    """
    from local_operator.tui.events import EventController, TurnEnded, TurnStarted
    from tests.unit.tui.test_events import FakeApp

    remote = _facade(tmp_path, monkeypatch)
    app = FakeApp()
    controller = EventController(remote, app)  # type: ignore[arg-type]
    app.controller = controller
    controller.subscribe()
    remote._on_wire_event(AgentStartEvent(generation=7).model_dump(mode="json"))
    assert sum(isinstance(m, TurnStarted) for m in app.posted) == 1

    remote._on_disconnected("send timeout")
    await _cancel_recovery(remote)
    assert not any(isinstance(m, TurnEnded) for m in app.posted)

    remote._install_frontend(
        FrontendSessionState(
            session_id="s1",
            epoch="e1",
            streaming=False,
            generation=7,
            last_turn_outcome="completed",
        )
    )
    remote._settle_suspect_turn()

    ends = [m for m in app.posted if isinstance(m, TurnEnded)]
    assert len(ends) == 1, f"band never learned the turn ended: {app.posted}"
    assert ends[0].aborted is False
    assert remote.is_streaming is False


@pytest.mark.asyncio
async def test_the_real_recovery_loop_bounds_the_turn_on_a_terminal_viewer(
    tmp_path, monkeypatch
) -> None:
    """Review round 1, BLOCKER-1: a real death must not spin forever on the TUI.

    The 8 s cold deadline only fired for ``_can_go_cold`` (desktop). The
    ordinary TUI attach viewer — ``tui/app.py`` calls ``connect()`` without
    ``surface``, so ``surface == "terminal"`` and ``_can_go_cold`` is False —
    could therefore never reach a verdict once the abort moved to recovery:
    a takeover that keeps failing retries forever by design, so nothing was
    left to end the turn.

    THIS TEST LETS THE REAL LOOP RUN. Every other verdict test here hand-calls
    ``_settle_suspect_turn`` / ``_go_cold``, and that test shape is what let
    the blocker through — ``_facade`` defaults ``can_go_cold=False`` and no
    test ever drove ``_recover_runtime`` itself on that surface.

    ``COLD_FALLBACK_S`` is monkeypatched so the bound is exercised in
    milliseconds; the assertion is on the VERDICT, never on elapsed time.
    """
    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", 0.05)
    takeover_attempts: list[int] = []

    async def failing_takeover() -> Any:
        takeover_attempts.append(1)
        raise RuntimeError("the lease is held by another follower")

    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=failing_takeover,
    )
    # The surface the operator actually uses: AttachedSession.connect defaults
    # to "terminal", so this is what a real TUI viewer looks like.
    assert remote._can_go_cold is False
    remote._streaming = True
    remote._generation = 7
    remote._ready_for_events = True
    received: list[Any] = []
    remote.subscribe(received.append)
    went_cold: list[str] = []
    remote.set_went_cold_callback(lambda: went_cold.append("cold"))

    remote._on_disconnected("owner exited")
    assert remote._recovery_task is not None
    # Wait on the PUBLICATION (the synthesised end), never on the clock.
    for _ in range(400):
        if any(isinstance(event, AgentEndEvent) for event in received):
            break
        await asyncio.sleep(0.01)
    await _cancel_recovery(remote)

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1, (
        "a dead runtime on the terminal surface left the turn spinning: "
        f"ends={ends} attempts={len(takeover_attempts)}"
    )
    # The VERDICT, which is now a named cut-off rather than the bare abort a
    # user's Esc produces: this arm is the one the watched TUI session takes
    # (``_can_go_cold`` is False for every viewer built through ``connect()``),
    # and reporting a runtime death as the user's own cancel was the reported
    # bug (QA round 1, Q-1 / UX U1). The deliberately-not-taken desktop exit
    # below is what this test has always been about.
    assert ends[0].aborted is False
    assert ends[0].cut_off_cause == "owner-lost"
    assert "cut off" in str(ends[0].error)
    assert remote.is_streaming is False
    assert remote._suspect_generation is None
    # The legacy contract is preserved: the loop keeps CHASING a successor
    # instead of taking the desktop-only cold exit, which would have stopped
    # recovery and fired the went-cold callback.
    assert went_cold == [], "the terminal surface took the desktop cold exit"
    assert takeover_attempts, "the loop stopped retrying after ending the turn"


@pytest.mark.asyncio
async def test_the_terminal_bound_does_not_fire_when_no_turn_was_live(
    tmp_path, monkeypatch
) -> None:
    """The bound ends a SUSPECT turn only: an idle viewer synthesises nothing."""
    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", 0.05)

    async def failing_takeover() -> Any:
        raise RuntimeError("no successor yet")

    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *args: (None, None))
    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=failing_takeover,
    )
    remote._ready_for_events = True
    remote._streaming = False  # nothing was running when the socket dropped
    received: list[Any] = []
    remote.subscribe(received.append)

    remote._on_disconnected("owner exited")
    for _ in range(30):
        await asyncio.sleep(0.01)
    await _cancel_recovery(remote)

    assert [event for event in received if isinstance(event, AgentEndEvent)] == []
    assert remote._suspect_generation is None


@pytest.mark.asyncio
async def test_a_same_live_turn_rebind_still_seeds_what_the_gap_produced(
    tmp_path, monkeypatch
) -> None:
    """Review round 1, MAJOR-1: only the synthetic start is suppressed.

    ``live_events`` is the turn AS IT IS NOW, so a tool that started while
    the socket was down is in the snapshot and must paint. Dropping the whole
    seed meant its later real ``tool_end`` arrived orphaned and was discarded
    unrendered at ``agent_end`` — the user permanently lost that card.

    The ⊘ interrupted trap the deviation identified is still covered: the
    start event must NOT be re-seeded, because ``_handle_agent_start`` clears
    ``_started_tools``. Both halves are asserted here.
    """
    remote = _facade(tmp_path, monkeypatch)
    received: list[Any] = []
    remote.subscribe(received.append)
    remote._on_disconnected("send timeout")
    await _cancel_recovery(remote)

    remote._install_frontend(
        FrontendSessionState(
            session_id="s1",
            epoch="e1",
            streaming=True,
            generation=7,
            live_events=[
                {
                    "type": "tool_execution_start",
                    "tool_call_id": "gap-tool",
                    "tool_name": "hang",
                    "args": {},
                }
            ],
        )
    )
    remote._settle_suspect_turn()
    remote._finish_sync()

    types = [getattr(event, "type", None) for event in received]
    assert (
        "tool_execution_start" in types
    ), f"the gap's tool was dropped instead of painted: {types}"
    assert (
        "agent_start" not in types
    ), f"re-seeding the start clears _started_tools and orphans the card: {types}"
    assert [event for event in received if isinstance(event, AgentEndEvent)] == []
    assert remote.is_streaming is True


@pytest.mark.asyncio
async def test_a_recall_with_no_client_declines_instead_of_claiming_success(
    tmp_path, monkeypatch
) -> None:
    """``True`` is a promise the app commits to irreversibly.

    Round-1 review MAJOR-1. ``recall_steering`` skipped issuing the op when
    ``_client`` was None but still answered True — and True is what makes the
    TUI put the text in the composer and remove the steer's rows. So the
    message stayed queued on the owner, the composer held a copy, and no
    rejection was ever reported because there was no request to fail: the
    silent double-send this seam exists to remove, through another door.

    ``_client`` is None after ``dispose`` and for the whole window between a
    dropped socket and a reattach, which is the disconnect-mid-recall case
    this file is about.
    """
    from types import SimpleNamespace

    from local_operator.harness.types import Message

    remote = _facade(tmp_path, monkeypatch)
    message = Message.user("use 0.75 for the direct API")
    # The queue this follower BELIEVES the owner holds — the same replicated
    # frontend state the real recall reads (`test_remote_refresh.py` stubs the
    # store the same way).
    remote._frontend_store = SimpleNamespace(  # type: ignore[assignment]
        state=SimpleNamespace(queued_steering=[{"id": message.id, "text": message.text}])
    )
    refusals: list[str] = []
    remote.set_recall_resolution(refusals.append)
    assert remote._client is None, "the premise: this viewer has no socket"

    assert remote.recall_steering(message) is False, "no client means no recall"
    assert remote._recall_task is None, "and no op was issued"
    await _cancel_recovery(remote)


# --- the live-but-silent owner: recovery must reach a verdict ---------------
#
# An owner that is LIVE (a record is found on every pass) but SILENT (the
# socket accepts, the canonical sync never lands) is the shape that had NO
# bounded exit: `COLD_FALLBACK_S` returns only for `_can_go_cold`, and the
# takeover the legacy arm is written to reach lives in the no-record `else`
# branch a discoverable record never takes. Reported from the field as a
# session whose model could not be switched by any retry or `/resume`.


def _live_record() -> Any:
    """A record that always reads live, protocol 5, frontend-capable."""
    import time as _time

    from local_operator.session.frontend_state import FRONTEND_CAPABILITY
    from local_operator.session.runtime.types import SessionRecord

    return SessionRecord(
        pid=99999,
        kind="tui",
        session_id="s1",
        conversation_name="probe",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
        protocol=5,
        capabilities=[FRONTEND_CAPABILITY],
        heartbeat_at=_time.time(),
    )


def _silent_owner_facade(tmp_path, monkeypatch) -> tuple[AttachedSession, list[float]]:
    """A terminal viewer whose owner accepts the dial and then never speaks.

    Bounds are compressed so the exits are exercised in milliseconds; every
    assertion is on the VERDICT, never on elapsed time.
    """
    import time as _time

    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", 0.05)
    monkeypatch.setattr(remote_module, "FRONTEND_SYNC_BLOCKED_S", 0.05)
    monkeypatch.setattr(remote_module, "RECOVERY_GIVE_UP_S", 0.3)
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *a: (_live_record(), None))

    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    # The surface the operator actually uses: connect() defaults to "terminal".
    assert remote._can_go_cold is False
    remote._streaming = True
    remote._generation = 7
    remote._ready_for_events = True

    dials: list[float] = []

    async def silent_dial(record: Any) -> Any:
        dials.append(_time.monotonic())
        # The REAL `_dial` stamps the identity on entry, before the sync it
        # will never get; without this the `_runtime_pid` assertion below is
        # vacuous, because nothing ever set the pid it checks is cleared.
        remote._runtime_pid = record.pid
        return asyncio.get_running_loop().create_future()  # never resolves

    monkeypatch.setattr(remote, "_dial", silent_dial)
    return remote, dials


async def _await_verdict(remote: AttachedSession) -> None:
    """Wait on the STATE, never on the clock."""
    for _ in range(600):
        if not remote._recovering:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_live_but_silent_owner_does_not_latch_the_viewer_forever(
    tmp_path, monkeypatch
) -> None:
    """THE regression guard: recovery must not latch `_recovering` forever.

    Fails on the pre-fix tree, where the loop has no reachable return while a
    record keeps being found. `_recovering` is not a cosmetic flag: it is what
    refuses `/model`, `/goal`, `/rename`, `/effort`, `/compact`, `/fork` and
    `/credential` at every request/response seam.
    """
    remote, dials = _silent_owner_facade(tmp_path, monkeypatch)
    remote._on_disconnected("send timeout")
    assert remote._recovering is True, "the premise: recovery is running"
    await _await_verdict(remote)

    # SAMPLED BEFORE TEARDOWN. Cancelling the recovery task runs the loop's
    # `finally`, which clears `_recovering` and makes a broken tree look green
    # — this test passed against the defect for exactly that reason once.
    latched = remote._recovering
    await _cancel_recovery(remote)

    assert latched is False, (
        "the viewer latched in recovery against a live-but-silent owner: "
        f"dials={len(dials)} — every /model, /fork, /compact is refused forever"
    )


@pytest.mark.asyncio
async def test_the_give_up_exit_leaves_a_viewer_that_can_bind_again(tmp_path, monkeypatch) -> None:
    """`_can_go_cold` must be flipped BEFORE `_go_cold`, or the fix is worse.

    `_ensure_bound` returns immediately when `_can_go_cold` is False, so
    clearing `_recovering` without the flip yields a viewer that is cold,
    permanently unbindable, and silently no-ops — worse than the honest
    refusal it replaces. Nothing else covers that variant.
    """
    remote, _ = _silent_owner_facade(tmp_path, monkeypatch)
    remote._on_disconnected("send timeout")
    await _await_verdict(remote)
    can_go_cold = remote._can_go_cold
    runtime_pid = remote.runtime_pid
    await _cancel_recovery(remote)

    assert can_go_cold is True, "the viewer can never bind again"
    # `runtime_pid` promises None while cold. Today `_discard_rejected_client`
    # already clears it on every failure path, so this pins the INVARIANT at
    # the exit rather than a defect: a stale live pid would let
    # `take_unannounced_cleanup` claim another runtime's notice (F3).
    assert runtime_pid is None, f"a stale owner pid outlived the unbind: {runtime_pid}"


@pytest.mark.asyncio
async def test_the_refusal_is_no_longer_driven_by_the_recovery_flag(tmp_path, monkeypatch) -> None:
    """The PERMANENT refusal lifts: `_recovering` no longer gates the seams.

    A cold viewer still refuses a routed slash on `client is None` — that is
    the ordinary, self-correcting cold state every viewer reaches after
    `/stop`, and it is not what this fix is about. What changed is that the
    refusal is no longer held open by a flag nothing can clear: the facade is
    rebindable, so the next action repairs it. Asserted on the CAUSE (the
    flag, and the dial the repair reaches) rather than on the string, because
    both causes raise the same sentence.
    """
    remote, dials = _silent_owner_facade(tmp_path, monkeypatch)
    # `_bind_under_lock` imports both of these function-locally, so the
    # module-level patches the fixture makes do not reach it. The engage is a
    # no-op because the record below already names a live runtime — which is
    # exactly what `engage_runtime` short-circuits on in production.
    import local_operator.mobile.attach_client as attach_client
    import local_operator.session.runtime.launch as launch

    monkeypatch.setattr(attach_client, "find_runtime_record", lambda *a: (_live_record(), None))
    monkeypatch.setattr(launch, "engage_runtime", lambda *a, **k: asyncio.sleep(0))

    remote._on_disconnected("send timeout")
    await _await_verdict(remote)

    # Issued BEFORE teardown, for the same reason `latched` is sampled early.
    recovering = remote._recovering
    dials_before = len(dials)
    # The repair path the user actually takes: any action rebinds. It must
    # reach the DIAL — a viewer left with `_can_go_cold` False returns from
    # `_ensure_bound` immediately and silently no-ops forever instead.
    try:
        await asyncio.wait_for(remote._ensure_bound(), timeout=1.0)
    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass  # the owner is still silent; that it TRIED is the assertion
    dials_after = len(dials)
    await _cancel_recovery(remote)

    assert recovering is False, "the recovery flag still gates every seam"
    assert dials_after > dials_before, (
        "the repair path never reached a dial: this viewer is permanently "
        f"unbindable (dials {dials_before} -> {dials_after})"
    )


@pytest.mark.asyncio
async def test_the_parked_prompt_is_released_by_the_bound(tmp_path, monkeypatch) -> None:
    """The silent half of the defect: `prompt` waits on `_runtime_ready` forever.

    `_on_disconnected` clears `_runtime_ready` and the only setters are
    `_go_cold`, the takeover branch and the deliberate-stop arms — none of
    which fired on this path. The user saw no message and no spinner
    resolution, which is worse than the refusal `/model` at least printed.
    """
    remote, _ = _silent_owner_facade(tmp_path, monkeypatch)
    released = asyncio.Event()

    async def parks_on_owner_ready() -> None:
        await remote._runtime_ready.wait()
        released.set()

    waiter = asyncio.create_task(parks_on_owner_ready())
    await asyncio.sleep(0)
    remote._on_disconnected("send timeout")
    assert released.is_set() is False, "the premise: the prompt path is parked"

    await _await_verdict(remote)
    for _ in range(100):
        if released.is_set():
            break
        await asyncio.sleep(0.01)
    was_released = released.is_set()
    waiter.cancel()
    await _cancel_recovery(remote)

    assert was_released is True, "the prompt path is still parked on _runtime_ready"


@pytest.mark.asyncio
async def test_a_genuinely_dead_owner_still_takes_the_legacy_takeover_path(
    tmp_path, monkeypatch
) -> None:
    """Guard against over-reach: the chase contract survives for a DEAD owner.

    The give-up exit is scoped to a record having been SEEN (`record_seen` in
    `_recover_runtime`). With no record the loop must still reach
    `_takeover_factory` and keep chasing, exactly as
    ``test_the_real_recovery_loop_bounds_the_turn_on_a_terminal_viewer`` pins.

    WATCHES PAST ITS OWN BOUND, and that is the whole guard. Sampling at an
    attempt COUNT is what made the first version of this test unable to fail:
    three takeover attempts arrive at ~0.28 s against a monkeypatched 0.3 s
    `RECOVERY_GIVE_UP_S`, so it stopped watching ~20 ms before the deadline it
    exists to prove is never reached, and it stayed green against a tree that
    took the exit (QA round 1, Q849-2). The wait below is therefore expressed
    as a multiple of the bound rather than as a number of attempts, and the
    assertions are read after it has comfortably elapsed.
    """
    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", 0.05)
    monkeypatch.setattr(remote_module, "RECOVERY_GIVE_UP_S", 0.3)
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *a: (None, None))
    attempts: list[int] = []

    async def failing_takeover() -> Any:
        attempts.append(1)
        raise RuntimeError("the lease is held by another follower")

    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=failing_takeover,
    )
    remote._streaming = True
    remote._generation = 7
    remote._ready_for_events = True
    went_cold: list[str] = []
    remote.set_went_cold_callback(lambda: went_cold.append("cold"))

    remote._on_disconnected("owner exited")
    # Watch WELL PAST the bound, never to an attempt count. Three attempts land
    # before the deadline, so breaking on them samples a loop that has not yet
    # had the chance to take the exit — the blindness Q849-2 measured. The wait
    # is a multiple of the monkeypatched bound, so compressing the bound
    # compresses the test with it and no wall-clock literal is asserted on.
    deadline = time.monotonic() + remote_module.RECOVERY_GIVE_UP_S * 3
    while time.monotonic() < deadline:
        if not remote._recovering:
            break  # the loop returned: the exit was taken, which is the failure
        await asyncio.sleep(0.01)
    still_chasing = remote._recovering
    cold_calls = list(went_cold)
    attempt_count = len(attempts)
    await _cancel_recovery(remote)

    assert attempt_count >= 3, f"the loop stopped chasing a successor: {attempt_count}"
    assert still_chasing is True, "the no-record branch took the give-up exit"
    assert cold_calls == [], "a dead owner took the give-up cold exit"


@pytest.mark.asyncio
async def test_an_unusable_record_is_not_a_sighting_for_the_give_up_exit(
    tmp_path, monkeypatch
) -> None:
    """A record this viewer cannot ATTACH to belongs to the dead-owner contract.

    `record_seen` is stamped on the same condition that selects the reattach
    arm — protocol >= 5 with `FRONTEND_CAPABILITY` — rather than on `record is
    not None`. A record failing either check falls through to the takeover
    `else`, so counting it as a sighting would arm the give-up exit for a chase
    that never had a reattach to give up on: the loop would go cold instead of
    chasing a successor, which is the over-reach Q849-1 measured wearing a
    different hat.
    """
    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", 0.05)
    monkeypatch.setattr(remote_module, "RECOVERY_GIVE_UP_S", 0.3)

    stale = _live_record()
    stale.protocol = 4  # a pre-frontend owner: discoverable, not attachable
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *a: (stale, None))
    attempts: list[int] = []

    async def failing_takeover() -> Any:
        attempts.append(1)
        raise RuntimeError("the lease is held by another follower")

    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=failing_takeover,
    )
    remote._streaming = True
    remote._generation = 7
    remote._ready_for_events = True
    went_cold: list[str] = []
    remote.set_went_cold_callback(lambda: went_cold.append("cold"))

    remote._on_disconnected("owner exited")
    deadline = time.monotonic() + remote_module.RECOVERY_GIVE_UP_S * 3
    while time.monotonic() < deadline:
        if not remote._recovering:
            break
        await asyncio.sleep(0.01)
    still_chasing = remote._recovering
    cold_calls = list(went_cold)
    attempt_count = len(attempts)
    await _cancel_recovery(remote)

    assert attempt_count >= 3, f"the loop stopped chasing a successor: {attempt_count}"
    assert still_chasing is True, "an unattachable record armed the give-up exit"
    assert cold_calls == [], "an unattachable record took the give-up cold exit"


@pytest.mark.asyncio
async def test_the_dial_failure_backoff_reaches_the_recovery_cap(tmp_path, monkeypatch) -> None:
    """The dial-failure arm paces at `_RECOVERY_DIAL_CAP_S`, not the old 0.5.

    That arm `continue`s, skipping the sleep at the bottom of the loop, so its
    own sleep is the only pacing on the path — and against `ATTACH_MAX_CLIENTS`
    with LRU eviction its rate is a real cascade risk. Asserted structurally on
    the delay sequence handed to `asyncio.sleep`, never as a rate measured
    against wall time.
    """
    monkeypatch.setattr(remote_module, "COLD_FALLBACK_S", 60.0)
    monkeypatch.setattr(remote_module, "RECOVERY_GIVE_UP_S", 60.0)
    monkeypatch.setattr(remote_module, "find_runtime_record", lambda *a: (_live_record(), None))

    remote = AttachedSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._ready_for_events = True

    async def refusing_dial(record: Any) -> Any:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(remote, "_dial", refusing_dial)

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def recording_sleep(delay: float, *args: Any, **kwargs: Any) -> Any:
        delays.append(delay)
        return await real_sleep(0, *args, **kwargs)  # run the loop hot

    monkeypatch.setattr(remote_module.asyncio, "sleep", recording_sleep)

    remote._on_disconnected("send timeout")
    for _ in range(400):
        if delays and max(delays) >= remote_module._RECOVERY_DIAL_CAP_S:
            break
        await real_sleep(0.005)
    observed = list(delays)
    await _cancel_recovery(remote)

    assert observed, "the dial-failure arm never paced at all"
    assert max(observed) == pytest.approx(remote_module._RECOVERY_DIAL_CAP_S), (
        f"the backoff ceiling is {max(observed)}, not _RECOVERY_DIAL_CAP_S "
        f"({remote_module._RECOVERY_DIAL_CAP_S}) — sequence={observed[:12]}"
    )
    # BOTH VALUES, not the ordering. `>` alone stays green if someone raises
    # `_BIND_RETRY_DELAY_CAP_S` to 1.9 — which is the re-unification the two
    # constants' comments explicitly forbid, and is the change this assertion
    # exists to catch (review round 1, R5). Pinning the pair asserts what those
    # comments promise: 2.0 s of pacing on the storm-capable loop, 0.5 s on the
    # attempt-bounded one a caller waits through.
    assert remote_module._RECOVERY_DIAL_CAP_S == 2.0, (
        "the recovery dial ceiling moved; see _RECOVERY_DIAL_CAP_S for why it "
        "is 2.0 and must not be re-unified with the bind cap"
    )
    assert remote_module._BIND_RETRY_DELAY_CAP_S == 0.5, (
        "the bind retry ceiling moved; a longer cap here only adds latency to "
        "a bind a caller is waiting on"
    )
