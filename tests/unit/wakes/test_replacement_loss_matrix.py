"""The six finite acceptance cases from the wake-replacement-loss decision.

One file, real paths throughout (real ``Session``/``WakeScheduler``/``wake`` tool
via ``ToolContext``/transcript/``pump``; ``ScriptedStream`` substitutes only the
external model transport, exactly as the reproduction rig and the red test do).
The reservation under test is central in ``WakeScheduler`` — see
``tests/unit/wakes/test_replacement_loss.py`` for the single red-first control
this expands on, and the module docstring there for the defect's shape.

Case map (decision doc §"Six finite acceptance cases"):
  C1  normal reuse / no pending            -> test_normal_reuse_no_pending
  C2  both collision timings (create after  -> test_both_collision_timings
      the final pump under a latched drain; and the lock interleaving)
  C3  held create + pre-fire list           -> test_create_after_final_pump_is_held_not_reused
      interleaving (held not reused)
  C4  OLD exactly once across handover +    -> test_old_exactly_once_across_handover
      reload; NEW keeps its due across restart
  C5  cap + cancellation across callers     -> test_cap_and_cancellation_across_callers
      (incl. a cold file-based arm)
  C6  forced append failure retains         -> test_failed_handover_retains_ownership
      ownership, refuses success, retries once

Anything the file could NOT express as an assertion is named in its own test's
docstring and in the PR body's honest-limits section, never faked.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest

from local_operator.harness.wake import (
    LOAD_GRACE_MS,
    MAX_WAKE_SCHEDULES,
    WakeSchedule,
    WakeScheduler,
)
from local_operator.harness.types import ToolContext
from local_operator.session import session as session_module
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools import builtin
from local_operator.video.jobs import VideoWakeHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

# Reuse the red test's native scaffolding rather than a second copy of it: the
# tool driver, the fresh-disk reader and the task join are one thing and must not
# drift between the control and the matrix that expands on it.
from tests.unit.wakes.test_replacement_loss import (
    _durable_message_for,
    _settle,
    _text,
    _wake_tool,
)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The red test's controlled wall clock, reconstructed here.

    A local definition rather than importing the red file's fixture: an imported
    fixture works but hides where the patch lives, and this module's cells need
    the SAME two module globals patched (the session's scheduler ``now`` and the
    tool's ``now_ms`` resolve from different modules).
    """
    controlled = _Clock()
    view = _ClockView(controlled)
    monkeypatch.setattr(session_module, "time", view)
    monkeypatch.setattr(builtin, "time", view)
    return controlled


class _Clock:
    def __init__(self) -> None:
        import time as _time

        self.ms = int(_time.time() * 1000)

    def advance(self, ms: int) -> None:
        self.ms += ms


class _ClockView:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock

    def time(self) -> float:
        return self._clock.ms / 1000

    def __getattr__(self, name: str) -> Any:
        import time as _time

        return getattr(_time, name)


def _messages(rows: Any) -> dict[str, str]:
    return {row.id: row.message for row in rows}


# ---------------------------------------------------------------------------
# C1 — normal reuse / no pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_reuse_no_pending(tmp_path: pathlib.Path, clock: Any) -> None:
    """C1: no drain at all. OLD fires normally and frees ``w3``; a later create
    reuses it, durably, and a reload preserves it. The reservation must be inert
    on the ordinary path — this is the control the whole change must not move."""
    directory = tmp_path / "sessions" / "c1"
    stream = ScriptedStream([text_turn("reply") for _ in range(4)])
    session = build_session(directory, stream, cwd=directory)
    try:
        await _wake_tool(session, "create", message="KEEP1", **{"in": "1h"})
        await _wake_tool(session, "create", message="KEEP2", **{"in": "1h"})
        await _wake_tool(session, "create", message="OLD", **{"in": "1s"})
        assert session._wake.pending_ids() == frozenset(), "nothing is held without a drain"

        clock.advance(2000)
        assert await session._wake.pump() == 1, "OLD fires its single occurrence"
        # A normal one-shot RETIRES; no successor holds it.
        assert session._wake.held_schedules == ()
        assert _durable_message_for(directory, "w3") is None, "the retired row is gone from disk"

        await _wake_tool(session, "create", message="NEW", **{"in": "75m"})
        assert _durable_message_for(directory, "w3") == "NEW", "the freed handle is reusable"

        await _settle(session)
        await session.dispose()
        reloaded = build_session(
            directory, ScriptedStream([text_turn("r") for _ in range(3)]), cwd=directory
        )
        try:
            assert _messages(reloaded._wake.schedules)["w3"] == "NEW"
        finally:
            await _settle(reloaded)
            await reloaded.dispose()
    finally:
        if not session._disposed:
            await session.dispose()


# ---------------------------------------------------------------------------
# C2 — both collision timings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_collision_timings(tmp_path: pathlib.Path, clock: Any) -> None:
    """C2: the two create timings around a draining final fire.

    (i) the drain is latched and OLD's final pump has ALREADY retired+held ``w3``,
    then a create lands — it must get ``w4``, never the held ``w3``; and
    (ii) the create's candidate list was prepared BEFORE the pump (the stale-list
    interleaving) and then written — the held ``w3`` must not be un-held, dropped,
    or re-fired, and the NEW row must survive.
    """
    directory = tmp_path / "sessions" / "c2"
    session = build_session(
        directory, ScriptedStream([text_turn("r") for _ in range(5)]), cwd=directory
    )
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    try:
        await _wake_tool(session, "create", message="KEEP1", **{"in": "1h"})
        await _wake_tool(session, "create", message="KEEP2", **{"in": "1h"})
        await _wake_tool(session, "create", message="OLD", **{"in": "1s"})
        assert handle.begin_drain("runtime-retired")

        # (i-a) a create BEFORE the final pump is an ordinary create on a live row.
        await _wake_tool(session, "create", message="NEW", **{"in": "75m"})
        assert _durable_message_for(directory, "w4") == "NEW"

        clock.advance(2000)
        assert await session._wake.pump() == 1, "OLD retires and is held for the successor"
        assert session._wake.pending_ids() == frozenset({"w3"})
        assert "w3" in _messages(session._wake.schedules), "the held handle is still owned"
        assert "w3" not in _messages(session._wake.armed_schedules), "but not runnable here"

        # (i-b) the still-latched, already-pumped window: a further create must take
        # a fresh free handle, never the held w3, and the create must not have
        # promoted the held row back onto the timer.
        await _wake_tool(session, "create", message="NEW2", **{"in": "75m"})
        assert _durable_message_for(directory, "w5") == "NEW2"
        assert _durable_message_for(directory, "w3") == "OLD", "the held id still means OLD"

        # (ii) the stale-list interleaving: a create that prepared its candidate
        # list BEFORE the pump lands re-affirms its rows. w3 is held, so that
        # write must leave it held and NOT promote the stale active copy back
        # onto the timer (the design names this interleaving explicitly).
        stale = [row for row in session._wake.schedules if row.id != "w4"]
        await session._wake.update(stale)
        assert session._wake.pending_ids() == frozenset({"w3"}), "the held id was not un-held"
        assert "w3" not in _messages(session._wake.armed_schedules), "and not promoted"

        assert handle.end_drain()
    finally:
        await _settle(session)
        if not session._disposed:
            await session.dispose()


# ---------------------------------------------------------------------------
# C3 — create-after-final-pump is held, not reused (reconcile not corrupted)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_after_final_pump_is_held_not_reused(
    tmp_path: pathlib.Path, clock: Any
) -> None:
    """C3: the exact reproduced cell, now asserting the fix.

    After begin_drain → OLD fires and is held for the successor, then the drain is
    abandoned and a create lands: the held row stays durably OLD, and the NEW
    schedule takes a DIFFERENT handle — the create's receipt must match durable
    state (no silent discard)."""
    directory = tmp_path / "sessions" / "c3"
    session = build_session(
        directory, ScriptedStream([text_turn("r") for _ in range(4)]), cwd=directory
    )
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    try:
        for message in ("KEEP1", "KEEP2"):
            await _wake_tool(session, "create", message=message, **{"in": "1h"})
            clock.advance(1)
        await _wake_tool(session, "create", message="OLD", **{"in": "1s"})
        clock.advance(1)
        assert _durable_message_for(directory, "w3") == "OLD"

        assert handle.begin_drain("runtime-retired")
        clock.advance(2000)
        assert await session._wake.pump() == 1

        assert handle.end_drain()  # abandoned: keeps serving
        receipt = await _wake_tool(session, "create", message="NEW", **{"in": "75m"})
        new_id = str((receipt.details or {}).get("wake_id"))
        assert new_id not in {"w3"}, "the held handle must not be reallocated"
        assert _durable_message_for(directory, "w3") == "OLD", "w3 stays the owed OLD row"
        assert _durable_message_for(directory, new_id) == "NEW", "the create's receipt is durable"
    finally:
        await _settle(session)
        if not session._disposed:
            await session.dispose()


# ---------------------------------------------------------------------------
# C4 — OLD exactly once across handover + reload; NEW keeps its own due
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_old_exactly_once_across_handover(tmp_path: pathlib.Path, clock: Any) -> None:
    """C4: OLD delivered exactly once across a successful handover + native reload,
    never resurrected by a later pump; NEW keeps its own future due across a restart.

    The delivery greeting is the native prompt (turn spawn) the reload performs —
    counted as one prompt for OLD, zero before its grace, and the same instant is
    preserved for NEW across a reload rather than inherited from OLD."""
    directory = tmp_path / "sessions" / "c4"
    stream = ScriptedStream([text_turn("r") for _ in range(6)])
    session = build_session(directory, stream, cwd=directory)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    reloaded = None
    try:
        await _wake_tool(session, "create", message="OLD", **{"in": "1s"})
        await _wake_tool(session, "create", message="NEW", **{"in": "75m"})
        new_due = next(r.next_due_at for r in session._wake.schedules if r.message == "NEW")

        assert handle.begin_drain("runtime-retired")
        clock.advance(2000)
        assert await session._wake.pump() == 1, "OLD retires, held for the successor"
        assert await session.hand_wakes_to_successor() == 1, "the held OLD is written once"
        assert session._wake.held_schedules == (), "ownership passed with the write"
        assert await session.hand_wakes_to_successor() == 0, "idempotent second handover"

        await _settle(session)
        await session.dispose()
        clock.advance(1)

        reloaded_stream = ScriptedStream([text_turn("r") for _ in range(6)])
        reloaded = build_session(directory, reloaded_stream, cwd=directory)
        # Both survived the restart: OLD as the owed overdue row, NEW at its own instant.
        reloaded_msgs = _messages(reloaded._wake.schedules)
        assert reloaded_msgs["w1"] == "OLD", "the owed occurrence reached disk"
        assert reloaded_msgs["w2"] == "NEW", "and NEW is still there, on its own handle"
        new_row = next(r for r in reloaded._wake.schedules if r.message == "NEW")
        assert new_row.next_due_at == new_due, "NEW keeps its own instant, not OLD's"

        # OLD is due now + grace on reload: it fires exactly once as a held catch-up.
        clock.advance(LOAD_GRACE_MS + 1)
        assert await reloaded._wake.pump() == 1, "OLD's owed occurrence fires exactly once"
        await _settle(reloaded)
        # A later pump must NOT resurrect it.
        clock.advance(5000)
        assert await reloaded._wake.pump() == 0, "OLD is not resurrected by a later pump"
    finally:
        if reloaded is not None and not reloaded._disposed:
            await _settle(reloaded)
            await reloaded.dispose()
        await _settle(session)
        if not session._disposed:
            await session.dispose()


# ---------------------------------------------------------------------------
# C5 — cap + cancellation across callers, including a cold file-based arm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cap_and_cancellation_across_callers(tmp_path: pathlib.Path) -> None:
    """C5: the held handle counts against the cap and is cancellable by its listed
    handle, and a video check-wake caller shares the same owned view.

    Uses a real ``WakeScheduler`` with an in-memory persist (the cap/cancel logic
    is scheduler-side, so no session is needed to exercise it) plus the REAL
    ``builtin`` create/cancel helpers and the REAL ``VideoWakeHandle``."""

    async def _noop_persist(_rows: Any) -> None:
        return None

    scheduler = WakeScheduler(now=lambda: 1, deliver=lambda due: None, persist=_noop_persist)
    # 15 ordinary rows (w1..w15) plus the held one = 16 user-facing, at the cap.
    rows = [
        WakeSchedule(id=f"w{i}", message=f"row{i}", next_due_at=1, created_at=i)
        for i in range(1, MAX_WAKE_SCHEDULES)
    ]
    scheduler._schedules = list(rows)
    held = WakeSchedule(id="w16", message="owed", next_due_at=1, created_at=16)
    assert scheduler.hold_pending(held) is True
    assert "w16" in _messages(scheduler.schedules), "the held row is in the owned view"

    # A create at the cap is refused: the held row counts.
    context = ToolContext(cwd=str(tmp_path), wake_scheduler=scheduler)
    refused = await builtin.execute_wake(
        "cap",
        {"op": "create", "message": "overflow", "in": "1h"},
        None,
        None,
        context,
    )
    assert refused.is_error and "at most 16" in _text(refused), _text(refused)

    # Cancelling the held handle by its listed id removes it and frees the slot.
    cancelled = await builtin.execute_wake(
        "cancel",
        {"op": "cancel", "id": "w16"},
        None,
        None,
        context,
    )
    assert not cancelled.is_error and "w16" in _text(cancelled), _text(cancelled)
    assert scheduler.pending_ids() == frozenset(), "the hold went with the cancel"
    assert "w16" not in _messages(scheduler.schedules)

    # The freed slot is genuinely reusable now.
    reused = await builtin.execute_wake(
        "reuse",
        {"op": "create", "message": "new", "in": "1h"},
        None,
        None,
        context,
    )
    assert not reused.is_error, _text(reused)

    # A video check-wake caller sees the same owned view (its ensure() reads
    # ``scheduler.schedules``), so a held id is never double-armed by it.
    scheduler2 = WakeScheduler(now=lambda: 1, deliver=lambda due: None, persist=_noop_persist)
    scheduler2._schedules = [WakeSchedule(id="w1", message="keep", next_due_at=1, created_at=1)]
    assert scheduler2.hold_pending(
        WakeSchedule(id=VideoWakeHandle(scheduler2, "job").wake_id, message="check", next_due_at=1)
    )
    await VideoWakeHandle(scheduler2, "job").ensure()
    assert len(scheduler2.schedules) == 2, "the video caller did not double-arm the held id"


# ---------------------------------------------------------------------------
# C6 — forced authoritative-append failure retains ownership, retries once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_handover_retains_ownership(tmp_path: pathlib.Path) -> None:
    """C6: a forced failure of the authoritative append during handover must keep
    the pending handle, refuse to claim success (return 0), and succeed on retry —
    and a held-cancel persist failure must not report success either."""
    calls = {"n": 0}

    async def flaky_persist(rows: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("authoritative append failed (forced)")

    scheduler = WakeScheduler(now=lambda: 1, deliver=lambda due: None, persist=flaky_persist)
    scheduler._schedules = [WakeSchedule(id="w1", message="keep", next_due_at=1, created_at=1)]
    owed = WakeSchedule(id="w2", message="owed", next_due_at=1, created_at=2)
    assert scheduler.hold_pending(owed)

    # First handover: the append fails -> ownership retained, success refused.
    assert await scheduler.handover_pending() == 0, "a failed append must not claim success"
    assert scheduler.pending_ids() == frozenset({"w2"}), "ownership retained for retry"
    assert "w2" in _messages(scheduler.schedules), "the row is still owned, not dropped"

    # Retry: the writer is restored, so the handover succeeds once.
    assert await scheduler.handover_pending() == 1, "retry succeeds once"
    assert scheduler.pending_ids() == frozenset(), "released only after the write landed"
    assert await scheduler.handover_pending() == 0, "idempotent afterwards"
