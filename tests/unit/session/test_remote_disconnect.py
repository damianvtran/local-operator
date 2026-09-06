"""Deferred abort on transient viewer disconnect (design §5.2 tests 16–19).

A dropped socket is not an abort: the runtime is usually still running the
turn. These pin that ``_on_disconnected`` no longer synthesises an
``AgentEndEvent``, and that recovery's verdict is the only writer of one.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import local_operator.session.remote as remote_module
from local_operator.harness.types import AgentEndEvent, AgentStartEvent
from local_operator.session.frontend_state import FrontendSessionState
from local_operator.session.remote import RemoteSession


def _facade(tmp_path, monkeypatch, *, can_go_cold: bool = False) -> RemoteSession:
    monkeypatch.setattr(remote_module, "find_owner_record", lambda *args: (None, None))
    remote = RemoteSession(
        config_dir=tmp_path,
        session_id="s1",
        takeover_factory=lambda: asyncio.sleep(0, result=None),
    )
    remote._can_go_cold = can_go_cold
    remote._streaming = True
    remote._generation = 7
    remote._ready_for_events = True
    return remote


async def _cancel_recovery(remote: RemoteSession) -> None:
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
    types = [getattr(event, "type", None) for event in received]
    assert "tool_execution_start" not in types


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
async def test_recovery_going_cold_emits_one_aborted_end(tmp_path, monkeypatch) -> None:
    """Test 18: recovery goes cold → one aborted end (unchanged)."""
    remote = _facade(tmp_path, monkeypatch, can_go_cold=True)
    received: list[Any] = []
    remote.subscribe(received.append)
    remote._on_disconnected("owner exited")
    await _cancel_recovery(remote)
    remote._go_cold()

    ends = [event for event in received if isinstance(event, AgentEndEvent)]
    assert len(ends) == 1 and ends[0].aborted is True and ends[0].error is None
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
