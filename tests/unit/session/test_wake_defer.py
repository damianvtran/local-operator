"""Wake-defer tests: a wake that fires right after an operator turn must not
append its own turn below the operator's exchange.

Pins the brief's 7 cells. W is patched small (0.3 s) and D is patched so the
cap case is reachable. Tests read `local_operator.session.session.WAKE_OPERATOR_QUIET_S`
/ `WAKE_PARK_CAP_S` so the constants' patch point is part of the contract.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import StreamEndEvent, StreamTextDelta
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.session.session import Session
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _due(wake_id: str = "w1", message: str = "wake up") -> DueWake:
    schedule = WakeSchedule(id=wake_id, message=message, next_due_at=0, created_at=0)
    return DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)


def _wake_rows(session: Session):
    """Persisted wake_prompt custom rows, in transcript order."""
    rows = []
    for e in session._transcript.entries():
        payload = getattr(e, "payload", None) or {}
        if isinstance(payload, dict) and payload.get("custom_type") == "wake_prompt":
            rows.append(payload)
    return rows


@pytest.mark.asyncio
async def test_wake_right_after_operator_turn_is_parked_not_delivered(tmp_path, monkeypatch):
    """Cell (c): prompt(q) then deliver wake -> only 1 request, no wake row
    before W; after W the wake IS delivered (idle guarantee)."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 0.3)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("q")
    assert len(stream.requests) == 1

    await session._deliver_wake(_due())
    # Parked: no second turn, no wake row, still only one request.
    await asyncio.sleep(0.05)
    assert len(stream.requests) == 1, f"parked wake spawned a turn early: {len(stream.requests)}"
    assert _wake_rows(session) == [], f"wake row persisted before W: {_wake_rows(session)}"

    # After W the deadline fires and the wake is delivered as its own turn.
    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    assert _wake_rows(session), "wake row never persisted after W"
    await session.dispose()


@pytest.mark.asyncio
async def test_parked_wake_folds_into_next_operator_prompt(tmp_path, monkeypatch):
    """Cell fold: prompt(q1), wake, prompt(q2) within W -> exactly 2 requests;
    wake text in request 2 BEFORE q2's user message; no third request after W."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 5.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="c"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("q1")
    await session._deliver_wake(_due(message="check the build"))
    await session.prompt("q2")

    assert len(stream.requests) == 2, f"expected 2 requests, got {len(stream.requests)}"
    texts = [m.text for m in stream.requests[1].messages]
    wake_idx = next((i for i, t in enumerate(texts) if "check the build" in t), None)
    q2_idx = next((i for i, t in enumerate(texts) if t == "q2"), None)
    assert wake_idx is not None, f"wake text not in request 2: {texts}"
    assert q2_idx is not None, f"q2 not in request 2: {texts}"
    assert wake_idx < q2_idx, "wake must precede q2 in the folded request"

    # The deadline must not spawn a third turn: the parked wake was consumed.
    await asyncio.sleep(0.1)
    assert len(stream.requests) == 2, f"deadline spawned extra turn: {len(stream.requests)}"
    await session.dispose()


@pytest.mark.asyncio
async def test_wake_during_operator_turn_open_is_parked(tmp_path, monkeypatch):
    """Cell (b): block _flush_shell_records during prompt(); lock held but
    _is_streaming still False; deliver wake; release; wake row not appended
    directly after the operator row."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 5.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    entered = asyncio.Event()
    release = asyncio.Event()
    original_flush = session._flush_shell_records

    async def blocked_flush() -> None:
        entered.set()
        await release.wait()
        return await original_flush()

    session._flush_shell_records = blocked_flush  # type: ignore[method-assign]
    prompt_task = asyncio.ensure_future(session.prompt("q"))
    await wait_for(lambda: entered.is_set(), timeout=3.0)

    # The brief's shape: lock held, streaming not yet set.
    assert session._turn_lock.locked()
    assert not session._is_streaming

    await session._deliver_wake(_due(message="during open"))
    release.set()
    await prompt_task

    # The wake was parked and FOLDED into the operator's own turn (one
    # request, wake row in the SAME turn's context ahead of "q"), not
    # appended as a separate turn below the operator row. The fold note is
    # the marker: it only rides folded wake rows.
    await asyncio.sleep(0.05)
    assert len(stream.requests) == 1, f"wake spawned its own turn: {len(stream.requests)}"
    rows = _wake_rows(session)
    assert all(
        "folded into your current answer" in r["details"]["text"] for r in rows
    ), f"wake row is not a folded row: {rows}"
    await session.dispose()


@pytest.mark.asyncio
async def test_idle_wake_delivered_immediately_and_wake_turn_does_not_open_window(
    tmp_path, monkeypatch
):
    """Cell idle guarantee: wake with no operator turn -> delivered immediately.
    A wake-initiated turn followed by another wake is NOT parked (the second
    wake opens its own turn, not a park)."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 5.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    await session._deliver_wake(_due("w1", "first"))
    await wait_for(lambda: len(stream.requests) == 1, timeout=3.0)

    await session._deliver_wake(_due("w2", "second"))
    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    await session.dispose()


@pytest.mark.asyncio
async def test_park_cap_delivers_even_when_operator_stays_active(tmp_path, monkeypatch):
    """Cell cap: operator turn stays open past patched D -> wake delivered at D,
    not starved."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 10.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 0.3)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("q")
    await session._deliver_wake(_due())
    await asyncio.sleep(0.05)
    assert len(stream.requests) == 1

    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    await session.dispose()


@pytest.mark.asyncio
async def test_two_fires_of_one_id_while_parked_deliver_once(tmp_path, monkeypatch):
    """Cell dedup: two fires of one id while parked -> one delivery."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 0.3)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="c"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("q")
    await session._deliver_wake(_due("w1", "first fire"))
    await session._deliver_wake(_due("w1", "second fire"))

    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    await asyncio.sleep(0.15)
    assert len(stream.requests) == 2, f"expected 1 wake delivery, got {len(stream.requests)}"
    assert len(_wake_rows(session)) == 1, f"expected 1 wake row, got {len(_wake_rows(session))}"
    await session.dispose()


@pytest.mark.asyncio
async def test_dispose_with_parked_wake_leaves_no_pending_task(tmp_path, monkeypatch):
    """Cell dispose: dispose with a parked wake -> no pending task."""
    import local_operator.session.session as sess_mod

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 5.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)

    await session.prompt("q")
    await session._deliver_wake(_due())
    await session.dispose()

    await asyncio.sleep(0.05)
    assert len(stream.requests) == 1, "parked wake fired after dispose"
