"""Hidden wake deliveries must paint nothing on ANY TUI surface.

A patience fire is an internal timer: "no wake line, no card, no badge, no
timer notification" (design §8.2.2). These pin the three TUI surfaces the
design names — the replay fold, the resumed-frame snap, and the band's wake
panel — each with a VISIBLE-wake control beside it, so the tests fail in both
directions: a leak paints something, and an over-broad filter would hide a
wake the user actually scheduled.

The classifier itself is ``harness.rows.is_hidden_wake_delivery``; what these
pin is that each surface ASKS it, because a shared helper nobody calls is the
drift this file exists to catch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.harness.types import CustomMessage
from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
from local_operator.tui.app import OperatorApp, _resume_tail_start
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text
from tests.unit.tui.test_wake_ui import _paint, _schedule


def hidden_wake(text: str = "(patience) Still no reply from the operator") -> CustomMessage:
    return CustomMessage(
        custom_type=WAKE_PROMPT_MESSAGE_TYPE,
        attribution="user",
        details={
            "text": text,
            "wake_id": "patience-1",
            "occurrence": 1,
            "kind": "patience",
            "hidden": True,
            "attempt": 1,
            "episode_id": "patience-1",
        },
    )


def visible_wake(text: str = "check the build") -> CustomMessage:
    return CustomMessage(
        custom_type=WAKE_PROMPT_MESSAGE_TYPE,
        attribution="user",
        details={"text": f"(alarm) Scheduled wake w1 — {text}", "wake_id": "w1", "occurrence": 3},
    )


@pytest.mark.asyncio
async def test_replay_paints_no_row_for_a_hidden_delivery_but_keeps_the_visible_one() -> None:
    session = FakeSession()
    session._history = [
        simple_user("morning"),
        hidden_wake(),
        visible_wake(),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)

    # The visible wake still paints its receipt (the control).
    assert "check the build" in shown
    # The hidden one paints NOTHING — no receipt, and not as a user bubble.
    assert "patience" not in shown
    assert "Still no reply" not in shown


@pytest.mark.asyncio
async def test_replay_does_not_register_a_live_receipt_for_a_hidden_fire() -> None:
    """The sibling rule: a hidden fire must not poison the dedupe key either.

    The fold skips a replayed receipt whose ``(wake_id, occurrence)`` the live
    turn already painted. A hidden row that registered a key would look, to a
    later replay of a VISIBLE receipt with the same key, like something already
    on screen — silently dropping the visible line. Registration is where that
    key is set, so the assertion is on the app's own bookkeeping.
    """
    session = FakeSession()
    fire = hidden_wake()
    session._history = [fire]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()

    assert ("patience-1", 1) not in app._live_wake_receipts


def simple_user(text: str):
    from local_operator.harness.types import Message

    return Message.user(text)


def test_the_resumed_frame_walk_steps_over_a_hidden_delivery() -> None:
    """The fold anchor: the backward walk must not spend its first slot on a
    row that paints nothing (the same rule the injected-render case pinned).

    Hand-built history with the hidden row placed EXACTLY where the naive cut
    lands, so the walk has to step over it to reach a turn the reader can see.
    """
    from local_operator.harness.types import Message

    history: list[object] = []
    for i in range(4):
        history.append(Message.user(f"turn {i}"))
        history.append(
            SimpleNamespace(role="assistant", id=f"a{i}", text="ok", tool_calls=None, content=[])
        )
    # 8 rows, so bound=4's naive cut is index 4. Inserting the hidden row at 5
    # pushes the naive cut to 5 — exactly where the walk starts — and the
    # nearest visible boundary below it is "turn 2"'s user row at index 4.
    history.insert(5, hidden_wake())
    start = _resume_tail_start(history, 4)
    assert start == 4
    anchored = history[start]
    assert getattr(anchored, "role", None) == "user"
    assert getattr(anchored, "custom_type", None) is None


@pytest.mark.asyncio
async def test_the_wake_panel_lists_scheduled_rows_and_never_patience_ones() -> None:
    visible, text = await _paint([_schedule("w1", "check the build")])
    assert visible is True and "check the build" in text

    # A patience row beside it is invisible; the panel is unchanged.
    visible, text = await _paint(
        [
            _schedule("w1", "check the build"),
            _schedule(
                "patience-1",
                "internal",
                kind="patience",
                hidden=True,
                episode_id="patience-1",
                attempt=1,
                armed_at=1,
            ),
        ]
    )
    assert visible is True and "check the build" in text and "patience" not in text

    # Patience only: the panel collapses to its hidden state entirely — a
    # session with one internal timer has no wakes a user can see or manage.
    visible, text = await _paint(
        [
            _schedule(
                "patience-1",
                "internal",
                kind="patience",
                hidden=True,
                episode_id="patience-1",
                attempt=1,
                armed_at=1,
            )
        ]
    )
    assert visible is False


def tool_pair(call_id: str, name: str):
    """One assistant call row + its result row, the shape the journal holds."""
    from local_operator.harness.types import Message, TextContent, ToolCall, ToolResult

    return [
        Message.assistant("", tool_calls=[ToolCall(id=call_id, name=name, arguments={})]),
        Message.tool_result(
            ToolResult(
                tool_call_id=call_id,
                tool_name=name,
                content=[TextContent(text=f"{name} output")],
            )
        ),
    ]


@pytest.mark.asyncio
async def test_replay_paints_no_ledger_row_for_a_hidden_tool_call() -> None:
    """The ``patience`` arm is an internal timer — it never renders (U2).

    The tool's own result copy promises the model the wait "is invisible to
    the user", and R30 promises no timer artifact; the ledger row
    (``▸ patience arm 90s ✓``) contradicted both on live AND replay (UX round
    1, U2). The ordinary call beside it is the control: an over-broad filter
    would take that row too.
    """
    from local_operator.tui.widgets.tool_card import ToolCard

    session = FakeSession()
    session._history = [simple_user("ask"), *tool_pair("p1", "patience"), *tool_pair("r1", "read")]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)
        cards = [card for card in app.query(ToolCard)]

    assert "read" in shown, "the visible control must keep its ledger row"
    assert "patience" not in shown
    assert [card.tool_call_id for card in cards] == ["r1"]


@pytest.mark.asyncio
async def test_the_live_seams_mount_no_card_for_a_hidden_tool_call() -> None:
    """BOTH live frames are gated, not just one: they race, either first (U2).

    ``tool_call_compose`` announces the row before the call exists and
    ``tool_execution_start`` would mount one for a call with none; suppressing
    only one seam leaves the other free to paint the row a moment later.
    """
    from local_operator.harness.types import (
        ToolCallComposeEvent,
        ToolExecutionStartEvent,
    )
    from local_operator.tui.events import ToolComposing, ToolStarted
    from local_operator.tui.widgets.tool_card import ToolCard

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.on_tool_composing(
            ToolComposing(event=ToolCallComposeEvent(tool_call_id="p1", tool_name="patience"))
        )
        app.on_tool_started(
            ToolStarted(event=ToolExecutionStartEvent(tool_call_id="p1", tool_name="patience"))
        )
        await pilot.pause()
        assert list(app.query(ToolCard)) == []

        # The control: an ordinary tool frames normally.
        app.on_tool_composing(
            ToolComposing(event=ToolCallComposeEvent(tool_call_id="r1", tool_name="read"))
        )
        await pilot.pause()
        assert [card.tool_call_id for card in app.query(ToolCard)] == ["r1"]


def quiet_pair(call_id: str = "q1"):
    """The pair a quiet turn persists: the call row, then its marked result.

    Product-shaped (``docs/design/quiet-turns.md`` §5, S1): a text-free
    assistant row carrying the one ``no_reply`` call, and its tool result with
    S0a's marker in ``details`` — the same two rows the e2e finds on disk.
    """
    from local_operator.harness.types import (
        QUIET_TURN_KEY,
        Message,
        TextContent,
        ToolCall,
        ToolResult,
    )

    return [
        Message.assistant("", tool_calls=[ToolCall(id=call_id, name="no_reply", arguments={})]),
        Message.tool_result(
            ToolResult(
                tool_call_id=call_id,
                tool_name="no_reply",
                content=[TextContent(text="Quiet.")],
                details={QUIET_TURN_KEY: True},
            )
        ),
    ]


@pytest.mark.asyncio
async def test_replay_paints_no_ledger_row_for_the_quiet_call() -> None:
    """The quiet call is the turn's door, not work (design §5, S1).

    The replay is where a hidden row comes back on every reopen, and the
    ``no_reply`` sentinel is the case the operator reported as filler — it
    must never paint as a ledger row. The ordinary call beside it is the
    control: an over-broad filter would take that row too.
    """
    from local_operator.tui.widgets.tool_card import ToolCard

    session = FakeSession()
    session._history = [simple_user("ask"), *quiet_pair("q1"), *tool_pair("r1", "read")]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._project_settled_rows(list(session._history))
        await pilot.pause()
        shown = _transcript_text(app)
        cards = [card for card in app.query(ToolCard)]

    assert "read" in shown, "the visible control must keep its ledger row"
    assert "no_reply" not in shown
    assert [card.tool_call_id for card in cards] == ["r1"]


@pytest.mark.asyncio
async def test_the_live_seams_mount_no_card_for_the_quiet_call() -> None:
    """BOTH live frames are gated, not just one: they race, either first (S1).

    ``tool_call_compose`` announces the row before the call exists and
    ``tool_execution_start`` would mount one for a call with none — suppressing
    only one seam leaves the other free to paint the ``no_reply`` row a moment
    later, exactly as for a hidden tool.
    """
    from local_operator.harness.types import (
        ToolCallComposeEvent,
        ToolExecutionStartEvent,
    )
    from local_operator.tui.events import ToolComposing, ToolStarted
    from local_operator.tui.widgets.tool_card import ToolCard

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.on_tool_composing(
            ToolComposing(event=ToolCallComposeEvent(tool_call_id="q1", tool_name="no_reply"))
        )
        app.on_tool_started(
            ToolStarted(event=ToolExecutionStartEvent(tool_call_id="q1", tool_name="no_reply"))
        )
        await pilot.pause()
        assert list(app.query(ToolCard)) == []

        # The control: an ordinary tool frames normally.
        app.on_tool_composing(
            ToolComposing(event=ToolCallComposeEvent(tool_call_id="r1", tool_name="read"))
        )
        await pilot.pause()
        assert [card.tool_call_id for card in app.query(ToolCard)] == ["r1"]


@pytest.mark.asyncio
async def test_a_quiet_result_settles_nothing_and_retires_a_stray_card() -> None:
    """The marker-read half of the pair's contract (design §5, S1).

    The composing and started gates are the first doors; the END frame reads
    the marker itself, so a card a seam older than the gates could have
    mounted is DROPPED rather than settled — the same settle-side shape the
    ask gate's divert arm keeps beside it. The frame is consumed with nothing
    painted either way.
    """
    from local_operator.harness.types import (
        QUIET_TURN_KEY,
        TextContent,
        ToolExecutionEndEvent,
        ToolResult,
    )
    from local_operator.tui.events import ToolEnded
    from local_operator.tui.widgets.tool_card import ToolCard

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Simulate the mixed-build residual the drop half exists for: a card
        # for the call that this process's seams never mounted.
        card = ToolCard("q1", "no_reply", {})
        app._append_block(card)
        await pilot.pause()
        assert app._painted_tool_card("q1") is card

        app.on_tool_ended(
            ToolEnded(
                event=ToolExecutionEndEvent(
                    tool_call_id="q1",
                    tool_name="no_reply",
                    result=ToolResult(
                        tool_call_id="q1",
                        tool_name="no_reply",
                        content=[TextContent(text="Quiet.")],
                        details={QUIET_TURN_KEY: True},
                    ),
                )
            )
        )
        await pilot.pause()
        assert app._painted_tool_card("q1") is None
        assert list(app.query(ToolCard)) == []
