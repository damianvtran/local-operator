"""The ask gate's TUI seams: settle-only ask rows, and the never-run exemption.

Under the queued engine an ``ask`` call is SETTLE-ONLY (design
docs/design/ask-gate.md §3 row 2): no row while it composes or runs, one row at
settle — the receipt for a raise, nothing for a divert. The suppression has one
documented EXEMPTION (a design-review contract line): a compose frame carrying
``not_run_reason`` is a verdict, not a dictation — no gate ran, a divert is
impossible, and a call that died before asking the user must stay visible
exactly as with the gate off.

These drive the REAL handlers through the pilot, with opposite-case controls
so a test cannot pass by suppressing too much — the discipline
``test_cross_session_visibility.py`` records for the same class of seam.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import (
    Message,
    TextContent,
    ToolCall,
    ToolCallComposeEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import ToolComposing, ToolEnded, ToolStarted
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import TranscriptView

from .test_app_pilot import FakeSession, _factory

ASK_ARGS = {
    "questions": [
        {
            "id": "q0",
            "question": "Which database?",
            "options": [{"label": "staging"}, {"label": "prod"}],
        }
    ]
}

MARKER = {"ask_gate": {"hidden": True, "verdict": "clear", "reason": "plainly best"}}


def _queued(session: FakeSession) -> None:
    """Make the fake answer as a live queued engine does (the gate's mode read)."""
    session.ask_queue = lambda: object()  # type: ignore[attr-defined]


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("session did not finish booting")


def _blocks(app: OperatorApp) -> list[Any]:
    return app.query_one(TranscriptView).blocks()


def _ask_cards(app: OperatorApp) -> list[ToolCard]:
    return [b for b in _blocks(app) if isinstance(b, ToolCard) and b.tool_name == "ask"]


def _ask_result(text: str, *, marker: bool = False) -> ToolResult:
    return ToolResult(
        tool_call_id="call-ask",
        tool_name="ask",
        content=[TextContent(text=text)],
        details=dict(MARKER) if marker else {},
    )


# --- composing / started suppression + the not-run exemption -----------------


@pytest.mark.asyncio
async def test_composing_under_the_queued_engine_mounts_nothing() -> None:
    session = FakeSession()
    _queued(session)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask", intent="asking")
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert _ask_cards(app) == []
        assert "call-ask" not in app._composing_cards
        assert "call-ask" not in app._tool_cards


@pytest.mark.asyncio
async def test_composing_on_the_blocking_arm_still_mounts() -> None:
    """The control: without the queued engine the dictation row is today's."""
    session = FakeSession()  # no ``ask_queue`` -> the mode read is False
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask", intent="asking")
            )
        )
        await pilot.pause()
        assert "call-ask" in app._composing_cards
        assert len(_ask_cards(app)) == 1


@pytest.mark.asyncio
async def test_the_never_run_ending_still_paints_under_the_queued_engine() -> None:
    """The CONTRACT LINE: a not-run compose frame is a verdict row, not a dictation.

    No gate ran and a divert is impossible, so the suppression must not swallow
    it: the row mounts at the frame and settles with the harness's reason —
    parity with every other tool and with the engine off.
    """
    session = FakeSession()
    _queued(session)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    intent="asking",
                    not_run_reason="the turn ended before the call could run",
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        cards = _ask_cards(app)
        assert len(cards) == 1, "a never-run ask must stay visible"
        # Settled, not running: `mark_not_run` is the settle.
        assert cards[0]._state != "running"
        assert "call-ask" not in app._composing_cards


@pytest.mark.asyncio
async def test_started_under_the_queued_engine_stashes_and_mounts_nothing() -> None:
    session = FakeSession()
    _queued(session)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert _ask_cards(app) == []
        assert "call-ask" not in app._tool_cards
        # The identity is stashed for the settle-mount (the END frame has none).
        assert app._ask_gate_settled_calls["call-ask"][0] == ASK_ARGS


# --- settle: the raise mounts, the divert drops ------------------------------


@pytest.mark.asyncio
async def test_ended_mounts_the_settled_receipt_for_a_raise() -> None:
    session = FakeSession()
    _queued(session)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask"))
        )
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    args=ASK_ARGS,
                    started_at_epoch=1000.0,
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert _ask_cards(app) == []

        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    result=_ask_result("Ask a-1 queued (1 question(s))."),
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        cards = _ask_cards(app)
        assert len(cards) == 1, "a raise's receipt row appears at settle"
        assert cards[0]._state == "success"
        # The row's identity came from the START frame's stash: the summary is
        # the same one a replay of the call's arguments paints.
        assert cards[0]._args == ASK_ARGS
        assert "call-ask" not in app._ask_gate_settled_calls


@pytest.mark.asyncio
async def test_ended_drops_on_the_divert_marker() -> None:
    """A divert settles NOTHING — no row ever appears."""
    session = FakeSession()
    _queued(session)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask"))
        )
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
            )
        )
        for _ in range(10):
            await pilot.pause()

        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    result=_ask_result(
                        "[Ask clearance] No question was put to the user.", marker=True
                    ),
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert _ask_cards(app) == []
        assert "call-ask" not in app._tool_cards
        assert "call-ask" not in app._ask_gate_settled_calls


@pytest.mark.asyncio
async def test_a_diverted_ask_does_not_linger_in_the_working_line() -> None:
    """MINOR-1(c): the working line and every registry drop a diverted ask.

    TWO LEGS, because only the second can fail on the thing this cell exists
    to pin (round-2 reviewer NIT: the first version passed with
    ``_refresh_working_activity`` no-op'd, so it proved no refresh ran).

    LEG 1 — the queued engine: compose and start are suppressed, so no card
    ever exists to name the ask; the START frame's stash (the one registry a
    settle-only ask ever touches) must be empty after the marker, and the
    mid-sequence assert proves the stash was really populated, so the final
    emptiness is a DROP rather than a never-wrote.

    LEG 2 — the MIXED BUILD (no ``ask_queue``: a surface that cannot read the
    owner's mode): compose and start MOUNT a card, so the working line
    genuinely names the ask, and the marker branch's
    ``_refresh_working_activity`` is the only thing that can take it back off.
    With that call no-op'd ``working.activity`` keeps the flash and this leg
    fails, which is the discrimination the first version lacked.
    """
    session = FakeSession()
    _queued(session)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._start_working_block()
        app.post_message(
            ToolComposing(
                ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask", intent="asking")
            )
        )
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    args=ASK_ARGS,
                    started_at_epoch=1_000.0,
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        assert "call-ask" in app._ask_gate_settled_calls, "the start frame stashed the identity"

        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    result=_ask_result("[Ask clearance] x", marker=True),
                )
            )
        )
        for _ in range(10):
            await pilot.pause()

        assert "call-ask" not in app._ask_gate_settled_calls
        assert "call-ask" not in app._tool_cards and "call-ask" not in app._composing_cards
        label = app._current_activity()[0]
        assert "ask" not in label.lower(), "the working line re-derived off the ask"
        working = app._working_block
        if working is not None:
            assert working.activity == label, "the refresh pushed the re-derived label"

        # LEG 2 — the DISCRIMINATING half (round-2 reviewer NIT): the same
        # divert on a surface that cannot read the owner's mode (no
        # ``ask_queue``). Compose and start mount a card here, so the working
        # line genuinely names the ask, and the marker branch's refresh is the
        # only thing that can take it back off. Under a no-op'd
        # ``_refresh_working_activity`` the working block keeps the flash and
        # the asserts below fail; leg 1 alone could not notice.
        mixed = FakeSession()
        app2 = OperatorApp(lambda: _factory(mixed))
        async with app2.run_test(size=(100, 30)) as pilot2:
            await _boot(pilot2, app2)
            app2._start_working_block()
            app2.post_message(
                ToolComposing(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask"))
            )
            app2.post_message(
                ToolStarted(
                    ToolExecutionStartEvent(
                        tool_call_id="call-ask",
                        tool_name="ask",
                        args=ASK_ARGS,
                        started_at_epoch=1_000.0,
                    )
                )
            )
            for _ in range(10):
                await pilot2.pause()
            before = app2._current_activity()[0]
            assert "ask" in before.lower(), "the mixed-build flash names the ask"
            working2 = app2._working_block
            assert working2 is not None
            assert working2.activity == before, "the mount's refresh pushed the flash"

            app2.post_message(
                ToolEnded(
                    ToolExecutionEndEvent(
                        tool_call_id="call-ask",
                        tool_name="ask",
                        result=_ask_result("[Ask clearance] x", marker=True),
                    )
                )
            )
            for _ in range(10):
                await pilot2.pause()
            after = app2._current_activity()[0]
            assert "ask" not in after.lower(), "the marker branch re-derived the label"
            assert working2.activity == after, (
                "the marker refresh PUSHED the re-derived label; a no-op refresh keeps the" " flash"
            )


@pytest.mark.asyncio
async def test_ended_on_the_blocking_arm_is_todays_settle() -> None:
    """The control: without the queued engine an ask settles like every tool."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app.post_message(
            ToolComposing(ToolCallComposeEvent(tool_call_id="call-ask", tool_name="ask"))
        )
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="call-ask", tool_name="ask", args=ASK_ARGS)
            )
        )
        await pilot.pause()
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="call-ask",
                    tool_name="ask",
                    result=_ask_result("answered: staging"),
                )
            )
        )
        for _ in range(10):
            await pilot.pause()
        cards = _ask_cards(app)
        assert len(cards) == 1
        assert cards[0]._state == "success"


# --- the paint seams (direct, no pilot) --------------------------------------


def _still_interrupted_card() -> ToolCard:
    card = ToolCard("call-ask", "ask")
    card.restore(state="interrupted")
    return card


def test_mark_pending_skips_a_settle_only_ask() -> None:
    def make_session(queue: Any) -> SimpleNamespace:
        return SimpleNamespace(
            pending_display_tool_ids=lambda: set(),
            executing_display_tool_ids=lambda: {"call-ask"},
            ask_queue=(lambda: queue),
        )

    card = _still_interrupted_card()
    OperatorApp._mark_pending_tool_rows([card], session=make_session(object()))
    assert card._state == "interrupted", "a gated ask row must not be repainted live"

    control = _still_interrupted_card()
    blocking = SimpleNamespace(
        pending_display_tool_ids=lambda: set(),
        executing_display_tool_ids=lambda: {"call-ask"},
    )
    OperatorApp._mark_pending_tool_rows([control], session=blocking)
    assert control._state == "running", "the control: without the engine it goes live"


def test_paint_skipped_live_skips_a_settle_only_ask() -> None:
    call = SimpleNamespace(id="call-ask", name="ask", arguments={})
    view = SimpleNamespace(blocks=lambda: [])

    collect: list[Any] = []
    painted = OperatorApp._paint_skipped_live_tool_rows(
        view,
        {},
        [call],
        collect=collect,
        session=SimpleNamespace(ask_queue=lambda: object()),
    )
    assert painted == [] and collect == []

    control: list[Any] = []
    painted = OperatorApp._paint_skipped_live_tool_rows(
        view, {}, [call], collect=control, session=SimpleNamespace()
    )
    assert painted == ["call-ask"] and len(control) == 1


# --- the replay fold ---------------------------------------------------------


def _ask_tail(*, marker: bool) -> list[Any]:
    return [
        Message.user("deploy it"),
        Message(
            role="assistant",
            content=[TextContent(text="")],
            tool_calls=[ToolCall(id="call-ask", name="ask", arguments=ASK_ARGS)],
        ),
        Message(
            role="tool",
            content=[TextContent(text="[Ask clearance] No question was put to the user.")],
            tool_call_id="call-ask",
            tool_name="ask",
            provider_payload={"details": dict(MARKER)} if marker else None,
        ),
    ]


@pytest.mark.asyncio
async def test_replay_skips_a_diverted_ask_pair_and_paints_a_normal_one() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        app._project_settled_rows(_ask_tail(marker=True))
        await pilot.pause()
        assert _ask_cards(app) == [], "a diverted ask paints no row on replay"

        app._project_settled_rows(_ask_tail(marker=False))
        await pilot.pause()
        assert len(_ask_cards(app)) == 1, "the visible control still paints its row"


@pytest.mark.asyncio
async def test_settle_painted_card_drops_on_the_marker() -> None:
    """A card mounted before the mode was knowable is dropped, not settled."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        card = ToolCard("call-ask", "ask")
        app._append_block(card)
        await pilot.pause()
        assert len(_ask_cards(app)) == 1

        marked = SimpleNamespace(
            text="[Ask clearance] x",
            provider_payload={"details": dict(MARKER)},
            is_error=False,
        )
        app._settle_painted_tool_card(card, marked)
        for _ in range(10):
            await pilot.pause()
        assert _ask_cards(app) == [], "the marker drop removes the row"
