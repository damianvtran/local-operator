"""Owner-death recovery semantics for RemoteSession."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import local_operator.session.remote as remote_module
from local_operator.session.remote import RemoteSession


@pytest.mark.asyncio
async def test_owner_death_takes_over_silently_and_retains_submitted_input(
    tmp_path, monkeypatch
) -> None:
    """A prompt submitted during rotation reaches the lease-winning Session."""

    class LocalWinner:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def prompt(self, text, images=None):  # noqa: ANN001
            self.prompts.append(text)

        async def dispose(self) -> None:
            pass

    winner = LocalWinner()

    async def takeover():
        return winner

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=takeover,
    )
    adopted: list[Any] = []

    async def adopt(local):  # noqa: ANN001
        adopted.append(local)
        # Mirrors OperatorApp._adopt_takeover_session: disposal happens from
        # inside the recovery task and must not self-cancel that task.
        await remote.dispose()

    remote.set_takeover_callback(adopt)
    remote._owner_ready.set()
    remote._on_disconnected("owner exited")
    submitted = asyncio.create_task(remote.prompt("continue after death"))
    await asyncio.wait_for(submitted, timeout=2)
    assert adopted == [winner]
    assert winner.prompts == ["continue after death"]


@pytest.mark.asyncio
async def test_takeover_swaps_subscription_and_updates_flow_from_the_winner(
    tmp_path, monkeypatch
) -> None:
    """M3/n4: after adoption the dead remote store is silent and the local
    winner's store is live, with no additive cost from the swap."""
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        FrontendStateStore,
    )

    class LocalWinner:
        def __init__(self) -> None:
            self._store = FrontendStateStore(
                FrontendSessionState(
                    session_id="s1",
                    epoch="winner",
                    cumulative_parent_cost=12.34,
                )
            )

        @property
        def frontend_state(self):  # noqa: ANN202
            return self._store.state

        def subscribe_frontend(self, handler):  # noqa: ANN001, ANN202
            return self._store.subscribe(handler)

        async def dispose(self) -> None:
            pass

    winner = LocalWinner()

    async def takeover():
        return winner

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(config_dir=tmp_path, session_id="s1", takeover_factory=takeover)
    remote._install_frontend(
        FrontendSessionState(session_id="s1", epoch="dead-owner", cumulative_parent_cost=12.34)
    )

    adopted: list[Any] = []
    received: list[Any] = []

    async def adopt(local):  # noqa: ANN001
        # Mirrors _adopt_takeover_session: unsubscribe the dead remote store,
        # subscribe the winner, and apply its snapshot as replacement state.
        adopted.append(local)
        subscription = local.subscribe_frontend(received.append)
        # Checkpoint reconciliation is replacement, never addition (no 24.68).
        assert subscription.sync.snapshot.cumulative_parent_cost == 12.34
        await remote.dispose()

    remote.set_takeover_callback(adopt)
    remote._owner_ready.set()
    remote._on_disconnected("owner exited")
    for _ in range(200):
        if adopted:
            break
        await asyncio.sleep(0.01)
    assert adopted == [winner]

    # Updates published after adoption arrive from the WINNER's store.
    winner._store.mutate(cumulative_parent_cost=12.5)
    assert [u.changes.get("cumulative_parent_cost") for u in received] == [12.5]


@pytest.mark.asyncio
async def test_deliberate_stop_never_takes_over(tmp_path, monkeypatch) -> None:
    """A follower that stopped the session it watched stays a viewer of the
    cold session — no lease win, no republished record, no SIGTERM bait.

    The disconnect after the follower's own ``request_stop`` is the op
    landing, not owner death (U2-4): ``_deliberate_stop`` is set before the
    op is sent, and ``_recover_owner`` must return immediately instead of
    calling the takeover factory (which would republish a live record for a
    session the user just ended).
    """
    takeover_calls: list[bool] = []

    async def takeover():
        takeover_calls.append(True)
        raise AssertionError("takeover must not run after a deliberate stop")

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=takeover,
    )
    remote._deliberate_stop = True  # as request_stop sets it before the op
    remote._on_disconnected("owner exited")
    if remote._recovery_task is not None:
        await remote._recovery_task
    assert takeover_calls == []
    assert remote._takeover_target is None


@pytest.mark.asyncio
async def test_wire_stop_by_another_process_never_takes_over(tmp_path, monkeypatch) -> None:
    """A shell ``lop stop`` of the owner while this follower watched: the
    transcript's ``stopped_at`` marker plus no live owner is the deliberate
    shape, and the follower stays a viewer (U2-4's wire variant)."""
    from local_operator.wakes import store as wake_store

    wake_store.write_entry(
        tmp_path,
        "s1",
        cwd="/tmp",
        schedules=[{"id": "w1", "message": "x", "every_ms": 60000, "next_due_at": 1}],
        preserve={"stopped_at": 1},
    )
    takeover_calls: list[bool] = []

    async def takeover():
        takeover_calls.append(True)
        raise AssertionError("takeover must not run after a deliberate stop")

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=takeover,
    )
    # The follower did NOT stop it itself; the marker is the wire evidence.
    remote._on_disconnected("owner exited")
    if remote._recovery_task is not None:
        await remote._recovery_task
    assert takeover_calls == []
    assert remote._takeover_target is None


@pytest.mark.asyncio
async def test_owner_death_without_stopped_marker_still_takes_over(tmp_path, monkeypatch) -> None:
    """A dead owner with NO ``stopped_at`` marker is recovered exactly as
    before: the deliberate-stop check must not swallow real owner death."""
    from local_operator.wakes import store as wake_store

    # The entry exists but was never stopped (no stopped_at).
    wake_store.write_entry(
        tmp_path,
        "s1",
        cwd="/tmp",
        schedules=[{"id": "w1", "message": "x", "every_ms": 60000, "next_due_at": 1}],
    )

    class Winner:
        async def dispose(self) -> None:
            pass

    winner = Winner()
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=winner),
    )
    # The callback is what ENDS the recovery loop: without it the loop
    # disposes each winner and retries forever (the real app always installs
    # one at adoption), so the sibling tests above install it too.
    adopted: list[Any] = []

    async def adopt(local):  # noqa: ANN001
        adopted.append(local)
        await remote.dispose()

    remote.set_takeover_callback(adopt)
    remote._on_disconnected("owner exited")
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)
    assert adopted == [winner]
    assert remote._takeover_target is winner


@pytest.mark.asyncio
async def test_wire_stopping_frame_prevents_takeover_without_any_wake_entry(
    tmp_path, monkeypatch
) -> None:
    """The owner's ``stopping`` announcement is what makes the wire case work
    for a session with NO wakes.

    The on-disk ``stopped_at`` marker only exists when a session HAS
    schedules (``write_entry``'s empty-schedules contract removes the file),
    so a wakeless session stopped by another process leaves nothing to read.
    The owner therefore announces the stop before closing, and the client
    surfaces it as ``STOPPED_REASON`` — the only evidence available here.
    """
    from local_operator.mobile.attach_client import STOPPED_REASON

    assert not (tmp_path / "wakes").exists()  # no marker to fall back on

    async def takeover():
        raise AssertionError("takeover must not run after an announced stop")

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=takeover,
    )
    remote._on_disconnected(STOPPED_REASON)
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)
    assert remote._deliberate_stop is True
    assert remote._takeover_target is None
    # The viewer stays usable: a prompt resolves against the stopped notice
    # instead of blocking forever on an owner that will never return.
    assert remote._owner_ready.is_set()


@pytest.mark.asyncio
async def test_a_failed_stop_does_not_disable_later_owner_death_recovery(
    tmp_path, monkeypatch
) -> None:
    """A follower whose ``/stop`` FAILED must still take over a real death.

    Round-3 MAJOR-1: the flag was set before the op and never cleared, so
    both reachable failures — no client attached, and an old owner answering
    unknown-op — told the user the stop had failed and then silently
    disabled recovery for the rest of the session. Only an ACCEPTED stop may
    suppress the takeover.
    """

    class OldOwner:
        """An owner too old to know the op: answers with an error."""

        connected = True

        async def request_stop(self) -> str:
            raise RuntimeError("this owner cannot stop itself gracefully")

    class Winner:
        async def dispose(self) -> None:
            pass

    winner = Winner()
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=winner),
    )
    remote._client = OldOwner()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await remote.request_stop()
    assert remote._deliberate_stop is False

    # ... and the owner is genuinely killed later: recovery must still run.
    adopted: list[Any] = []

    async def adopt(local):  # noqa: ANN001
        adopted.append(local)
        await remote.dispose()

    remote.set_takeover_callback(adopt)
    remote._client = None
    remote._on_disconnected("owner exited")
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)
    assert adopted == [winner]


@pytest.mark.asyncio
async def test_stop_with_no_client_attached_does_not_latch(tmp_path, monkeypatch) -> None:
    """The simpler failure shape: nothing attached, so nothing was stopped."""
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._client = None
    with pytest.raises(ConnectionError):
        await remote.request_stop()
    assert remote._deliberate_stop is False


@pytest.mark.asyncio
async def test_a_stopped_viewer_is_told_so_and_never_says_reconnecting(
    tmp_path, monkeypatch
) -> None:
    """The viewer's side of a deliberate stop is legible, not just safe.

    Round-3 D3-1/Q3-2/U3-1: the viewer painted nothing at the moment the
    stop landed (indistinguishable from an idle agent) and then answered
    every later message with the owner-death wording — promising a
    reconnection that was never coming, on the one screen where the kill
    switch has to be readable.
    """
    from local_operator.mobile.attach_client import STOPPED_REASON

    told: list[str] = []
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote.set_stopped_callback(lambda: told.append("stopped"))
    remote._on_disconnected(STOPPED_REASON)
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)

    assert told == ["stopped"]  # the app was told exactly once
    with pytest.raises(ConnectionError, match="stopped"):
        await remote.prompt("a message after the stop")
    # Fired once even if both recognition points run.
    remote._notify_stopped()
    assert told == ["stopped"]


@pytest.mark.asyncio
async def test_a_dropped_connection_still_says_reconnecting(tmp_path, monkeypatch) -> None:
    """The converse: a real owner loss keeps the recovery wording.

    The two states are opposites and must not collapse into one sentence —
    this is what stops the D3-1 fix from lying in the other direction.
    """
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    assert remote._unavailable_reason() == "session owner is reconnecting"


@pytest.mark.asyncio
@pytest.mark.parametrize("mid_turn", [False, True])
async def test_a_deliberate_stop_ends_the_turn_in_both_states(
    tmp_path, monkeypatch, mid_turn: bool
) -> None:
    """A stop ends the turn whether or not one was in flight.

    Round-4 MAJOR-3/D4-1: the deliberate-stop branch returned ABOVE the block
    that ends the turn, and every other writer of ``_streaming`` is fed by the
    owner whose socket just closed — so a viewer stopped MID-TURN reported
    streaming forever. The spinner never stopped, and the next message routed
    into the steer branch, was dropped, and was receipted as "sends when this
    step finishes" for a step that had ended. Both states are parameterised
    because only the idle one was covered, which is why this survived.
    """
    from local_operator.mobile.attach_client import STOPPED_REASON

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._streaming = mid_turn
    remote._on_disconnected(STOPPED_REASON)
    if remote._recovery_task is not None:
        await asyncio.wait_for(remote._recovery_task, timeout=5)

    assert remote.is_streaming is False
    # The honest refusal lives on the prompt path; ending the turn is what
    # lets a message reach it instead of the silent steer queue.
    with pytest.raises(ConnectionError, match="stopped"):
        await remote.prompt("the message typed after the stop")


@pytest.mark.asyncio
async def test_going_cold_ends_an_in_flight_turn_directly(tmp_path, monkeypatch) -> None:
    """#642 (UX U11 from #619): a viewer going cold mid-turn gets a terminal state.

    ``_go_cold`` cleared ``_streaming`` without emitting an ``AgentEndEvent``,
    so the app — which keeps its working line, band and title open until an
    end reaches it — held a spinner forever with the toast suppressed too.

    The end must be delivered DIRECTLY, not buffered: a failed successor
    ``_dial`` leaves ``_ready_for_events`` False, and a buffered end would be
    drained by the NEXT bind's ``_finish_sync`` behind that bind's seeded
    ``AgentStartEvent``, where the controller (which drops only ``gen <
    current``) would let it tear down the fresh turn. So the assertion is on
    both sides: exactly one aborted end reached the handler, and nothing was
    left in the buffer for a later runtime to inherit.
    """
    from local_operator.harness.types import AgentEndEvent

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._can_go_cold = True
    remote._streaming = True
    remote._ready_for_events = False
    received: list[Any] = []
    remote.subscribe(received.append)

    remote._go_cold()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1 and ends[0].aborted is True and ends[0].error is None
    assert remote._buffered_events == []
    assert remote.is_streaming is False
    assert remote.is_cold


@pytest.mark.asyncio
async def test_a_disconnect_followed_by_going_cold_ends_the_turn_once(
    tmp_path, monkeypatch
) -> None:
    """The common path: ``_on_disconnected`` already ended the turn, so the
    go-cold end is a no-op rather than a second "interrupted" notice.

    ``_end_turn_locally`` returns on ``not self._streaming``, which is what
    makes the two callers safe to stack — pinned here so a later rewrite of
    either does not turn one owner loss into two aborted ends on the app.
    """
    from local_operator.harness.types import AgentEndEvent

    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._can_go_cold = True
    remote._streaming = True
    remote._ready_for_events = True
    received: list[Any] = []
    remote.subscribe(received.append)

    remote._on_disconnected("owner exited")
    if remote._recovery_task is not None:
        remote._recovery_task.cancel()
        try:
            await remote._recovery_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — teardown only
            pass
    remote._go_cold()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1, f"one owner loss produced {len(ends)} aborted ends"
    assert remote._buffered_events == []
    assert remote.is_streaming is False
