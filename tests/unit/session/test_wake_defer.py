"""Wake-defer tests: a wake that fires right after an operator turn must not
append its own turn below the operator's exchange.

Pins the brief's 7 cells. W is patched small (0.3 s) and D is patched so the
cap case is reachable. Tests read `local_operator.session.session.WAKE_OPERATOR_QUIET_S`
/ `WAKE_PARK_CAP_S` so the constants' patch point is part of the contract.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import (
    StreamEndEvent,
    StreamTextDelta,
    WakeDeliveredEvent,
)
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
    # Event-driven, not a sleep race: the deadline fires at last_end + W,
    # deadlines are cancelled on dispose, so count requests from the loop
    # itself — no request may appear before the deadline fires.
    await asyncio.sleep(0.05)
    assert session._parked_wakes, "expected the wake to park"
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
    assert rows, "expected a folded wake row, got none"
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
    """Cell cap: an operator turn that stays open past patched D delivers the
    parked wake at D, not starved (the deadline's re-arm branch).

    The operator turn is held open by blocking ``_flush_shell_records`` in
    ``prompt()``: the wake parks at open, the deadline wakes with the turn
    still open and re-arms, and delivery lands at ``first_park + D``.
    """
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
    assert session._operator_turn_open

    await session._deliver_wake(_due())
    assert session._parked_wakes, "expected the wake to park"
    # The deadline re-armed while the turn stayed open (no delivery yet).
    await asyncio.sleep(0.4)
    assert len(stream.requests) == 0, "deadline delivered before D"
    release.set()
    await prompt_task

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
    """Cell dispose: dispose with a parked wake -> no pending task, and the
    wake is durable (held-delivery precedent), not lost."""
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
    assert session._parked_wakes, "expected the wake to park"
    await session.dispose()

    await asyncio.sleep(0.05)
    assert len(stream.requests) == 1, "parked wake fired after dispose"
    assert _wake_rows(session), "parked wake was lost on dispose (no durable row)"


@pytest.mark.asyncio
async def test_parked_wake_emits_no_receipt_until_delivery(tmp_path, monkeypatch):
    """MAJOR-1: a parked wake paints no ``WakeDeliveredEvent`` (and no TUI wake
    line) until it actually delivers; delivery emits exactly one."""
    import local_operator.session.session as sess_mod
    from local_operator.harness.types import WakeDeliveredEvent

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 0.3)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)
    events: list[object] = []
    session.subscribe(events.append)

    await session.prompt("q")
    events.clear()
    await session._deliver_wake(_due())
    await asyncio.sleep(0.05)
    assert not [
        e for e in events if isinstance(e, WakeDeliveredEvent)
    ], "parked wake emitted a receipt before delivery"

    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    receipts = [e for e in events if isinstance(e, WakeDeliveredEvent)]
    assert len(receipts) == 1, f"expected 1 receipt at delivery, got {len(receipts)}"
    await session.dispose()


@pytest.mark.asyncio
async def test_fold_emits_one_receipt_before_the_turn_runs(tmp_path, monkeypatch):
    """M-1 / Q5: the fold emits exactly one receipt per parked wake, before
    the turn's first agent event. The parked id fired twice → one receipt,
    one persisted row, none while merely parked."""
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
    events: list[object] = []
    session.subscribe(events.append)

    await session.prompt("q1")
    events.clear()
    await session._deliver_wake(_due("w1", "first fire"))
    await session._deliver_wake(_due("w1", "second fire"))
    assert session._parked_wakes
    assert not [
        e for e in events if isinstance(e, WakeDeliveredEvent)
    ], "receipt while merely parked"

    await session.prompt("q2")
    receipts = [e for e in events if isinstance(e, WakeDeliveredEvent)]
    assert len(receipts) == 1, f"fold emitted {len(receipts)} receipts, expected 1"
    assert receipts[0].wake_id == "w1" and receipts[0].occurrence == 1
    # Receipt before the turn's first agent event.
    first_agent = next(
        (i for i, e in enumerate(events) if type(e).__name__ == "AgentStartEvent"),
        None,
    )
    receipt_at = events.index(receipts[0])
    assert (
        first_agent is None or receipt_at < first_agent
    ), f"receipt at {receipt_at} not before AgentStart at {first_agent}"
    assert (
        len(_wake_rows(session)) == 1
    ), f"expected 1 persisted wake row, got {len(_wake_rows(session))}"
    await session.dispose()


@pytest.mark.asyncio
async def test_harness_injected_prompt_does_not_open_window(tmp_path, monkeypatch):
    """MAJOR-2: a ``prompt(..., harness_injected=True)`` row must not open the
    quiet window (neither ``_operator_turn_open`` nor the end stamp)."""
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

    await session.prompt("goal loop text that is not chrome", harness_injected=True)
    assert session._last_operator_turn_end is None, "harness-injected prompt stamped the window"
    assert not session._operator_active()

    await session._deliver_wake(_due())
    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    assert not session._parked_wakes, "idle wake after injected prompt parked"
    await session.dispose()


@pytest.mark.asyncio
async def test_wake_during_harness_injected_turn_is_not_parked(tmp_path, monkeypatch):
    """m-1: ``_operator_turn_open = not harness_injected`` — a wake delivered
    while a harness-injected prompt is in flight (lock held, not streaming)
    must NOT park. That gate is the only thing deciding here (no last-end
    stamp yet), so reverting it to True fails this test."""
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
    prompt_task = asyncio.ensure_future(session.prompt("goal loop text", harness_injected=True))
    await wait_for(lambda: entered.is_set(), timeout=3.0)
    assert session._turn_lock.locked()
    assert not session._is_streaming
    assert not session._operator_turn_open, "harness-injected prompt opened the operator turn"

    await session._deliver_wake(_due())
    assert not session._parked_wakes, "wake parked during harness-injected turn"
    release.set()
    await prompt_task
    await session.dispose()


@pytest.mark.asyncio
async def test_retire_spools_parked_wake_to_inbox(tmp_path, monkeypatch):
    """m-2: begin retire with a parked wake → exactly one inbox row, nothing
    parked, no deadline task. The successor drains the file at boot."""
    import local_operator.session.session as sess_mod
    from local_operator.session.runtime.inbox import drain_inbox

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
    await session._deliver_wake(_due("w1", "remind me at 3pm"))
    assert session._parked_wakes

    session.retire_wakes_to_inbox()
    assert not session._parked_wakes, "retire left wakes parked"
    assert session._parked_wake_task is None, "deadline task survived retire"
    # The spool is a background write: wait for it to land, then assert rows.
    directory = session._transcript.directory
    assert directory is not None
    rows = []
    for _ in range(100):
        rows = drain_inbox(directory) if directory else []
        if rows:
            break
        await asyncio.sleep(0.05)
    assert len(rows) == 1, f"expected 1 inbox row, got {len(rows)}"
    assert rows[0].wake and rows[0].wake_id == "w1"
    assert "remind me at 3pm" in rows[0].text
    await session.dispose()


@pytest.mark.asyncio
async def test_steer_into_wake_turn_opens_window_for_next_wake(tmp_path, monkeypatch):
    """MAJOR-3: an operator steer admitted into a wake-initiated turn opens
    the window; the next wake parks."""
    import local_operator.session.session as sess_mod
    from local_operator.harness.types import (
        AgentTool,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 5.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def blocking_execute(tool_call_id, args, signal, on_update, context):
        tool_started.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="block",
            content=[TextContent(text="finished")],
        )

    tool = AgentTool(
        name="block",
        parameters={"type": "object", "properties": {}},
        interruptible=True,
        execute=blocking_execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[tool])

    await session._deliver_wake(_due("w1", "first"))
    await wait_for(lambda: tool_started.is_set(), timeout=3.0)
    session.steer("operator: actually do X")
    release_tool.set()
    await wait_for(lambda: not session._is_streaming, timeout=3.0)
    assert (
        session._last_operator_turn_end is not None
    ), "steer admitted into wake turn did not open the window"

    await session._deliver_wake(_due("w2", "second"))
    assert session._parked_wakes, "second wake did not park after steered turn"
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wake_id", ["aida-cadence", "aida-extra-3", "librarian-cadence", "librarian-extra-2"]
)
async def test_cadence_family_wakes_never_park(tmp_path, monkeypatch, wake_id):
    """MINOR-2: cadence-family rows from BOTH engines open their own turn."""
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
    await session._deliver_wake(_due(wake_id, "check-in"))
    await wait_for(lambda: len(stream.requests) == 2, timeout=3.0)
    assert not session._parked_wakes, f"cadence wake {wake_id} parked"
    await session.dispose()


@pytest.mark.asyncio
async def test_operator_wake_on_duty_session_parks_normally(tmp_path, monkeypatch):
    """MINOR-2: an operator-armed wake on a duty session still parks — the
    gate is the cadence-family id, not the duty flag."""
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
    session._aida_duty = True
    await session.prompt("q")
    await session._deliver_wake(_due("w7", "remind me"))
    await asyncio.sleep(0.05)
    assert session._parked_wakes, "operator wake on duty session did not park"
    assert len(stream.requests) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_parked_wake_is_busy_and_warm(tmp_path, monkeypatch):
    """MAJOR-4(b): the runtime idle-exit predicates see a parked wake."""
    import local_operator.session.session as sess_mod
    from local_operator.session.runtime.serving import ServingSessionHandle

    monkeypatch.setattr(sess_mod, "WAKE_OPERATOR_QUIET_S", 5.0)
    monkeypatch.setattr(sess_mod, "WAKE_PARK_CAP_S", 5.0)

    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="a"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="b"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    await session.prompt("q")
    await session._deliver_wake(_due())
    assert session._parked_wakes
    assert handle.is_busy() is True, "parked wake does not pin residency"
    due_at = handle.next_wake_due_at()
    assert due_at is not None, "parked wake is invisible to the warmth signal"
    # The delivery instant is first_park + D (the cap binds when W >> D here:
    # W=5 s, D=5 s, parked immediately after the turn ends, so the quiet
    # target is last_end + 5 s ≈ first_park + 5 s and the cap is the same).
    first_park = session._parked_wakes_first_park
    assert first_park is not None
    expected = int((first_park + 5.0) * 1000)
    assert abs(due_at - expected) < 5_000, f"warmth instant {due_at} != first_park + D ({expected})"
    await session.dispose()
