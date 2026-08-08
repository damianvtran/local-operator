"""Mid-turn steering, Esc-to-stop, and the in-TUI tool-approval prompt.

These three exist because of one shared defect class: the TUI owned the terminal
but not the interactions that needed it.

- Typing during a turn called ``prompt()``, which the session REJECTS while a
  turn holds its lock, so the text was thrown away behind an error notice.
- Esc was bound to nothing.
- Approvals fell through to the factory's stdin gate, which cannot be answered
  while Textual holds the terminal in raw mode — the turn parked on ``input()``
  forever, which is what the user saw as a frozen agent.

The assertions are about the observable contract (what the session is told, what
the frame says, whether the awaited future settles), never about how the app
routes messages internally.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from local_operator.harness.types import ToolExecutionStartEvent
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import AssistantMessageStart, ToolStarted, TurnStarted
from local_operator.tui.widgets.approval import ApprovalBlock
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import TranscriptView

from .test_app_pilot import FakeSession, _factory


class SteerableSession(FakeSession):
    """A fake whose streaming state the test drives, recording steer calls."""

    def __init__(self) -> None:
        super().__init__()
        self.steers: list[str] = []
        self.streaming = False
        self.approval_handler: Any | None = None

    @property
    def is_streaming(self) -> bool:
        return self.streaming

    def steer(self, text: str) -> None:
        self.steers.append(text)

    def set_approval_handler(self, handler: object | None) -> None:
        self.approval_handler = handler


def _approval_gate(session: SteerableSession) -> Callable[[str, str], Awaitable[bool]]:
    """The handler the app installed, narrowed to non-optional.

    Asserting here rather than in each test keeps "the app installed a gate" a
    single failure with one message, instead of an AttributeError per call site.
    """
    handler = session.approval_handler
    assert handler is not None, "the app never installed its approval handler"
    return cast("Callable[[str, str], Awaitable[bool]]", handler)


def rows(app: OperatorApp) -> list[str]:
    """The painted frame as plain text, one entry per row."""
    return [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]


async def _submit(pilot: Any, app: OperatorApp, text: str) -> None:
    app.query_one(Editor).text = text
    await pilot.press("enter")
    await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_mid_turn_submit_steers_instead_of_prompting() -> None:
    """A turn is already running: the text rides the steering queue.

    The session's own contract is that ``prompt()`` raises while streaming, so
    this is not a preference — re-prompting is the bug, and the receipt tells the
    user their words were kept rather than dropped.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        await _submit(pilot, app, "first task")
        assert session.prompts == ["first task"]

        session.streaming = True
        await _submit(pilot, app, "actually use pygame-ce")

        assert session.steers == ["actually use pygame-ce"]
        assert session.prompts == ["first task"]  # no second prompt attempted
        painted = rows(app)
        assert any("actually use pygame-ce" in row for row in painted)
        assert any("queued" in row for row in painted)


@pytest.mark.asyncio
async def test_escape_stops_a_running_turn() -> None:
    """Esc aborts while streaming, and does nothing at all when idle.

    "Does nothing when idle" is the load-bearing half: Esc must never clear the
    composer, because that would discard typed text on the key people press to
    cancel.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)

        editor = app.query_one(Editor)
        editor.text = "kept"
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert session.aborts == []
        assert editor.text == "kept"
        # Focus MUST stay in the composer. TextArea binds Escape to `blur`, which
        # made the first press move focus out of the input while looking like it
        # did nothing — every keystroke after it went nowhere.
        assert app.screen.focused is editor

        session.streaming = True
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert session.aborts == ["interrupted"]
        assert app.screen.focused is editor


@pytest.mark.asyncio
async def test_empty_assistant_message_mounts_no_block() -> None:
    """A tool-use turn carries no prose, so it must not spend rows on a block.

    Every Anthropic tool turn opens a message and goes straight to the calls;
    mounting the block eagerly cost two rows (the empty block plus the blank row
    the spacing rule opens above a new kind), which read as a hole between the
    working line and the tool ledger.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        await _submit(pilot, app, "make a game")
        app.post_message(TurnStarted())
        app.post_message(AssistantMessageStart())
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="c0", tool_name="bash", args={"command": "ls -la"}
                )
            )
        )
        await pilot.pause(0.3)

        assert not app.query(AssistantBlock)
        painted = rows(app)
        working = next(index for index, row in enumerate(painted) if "working" in row)
        first_tool = next(index for index, row in enumerate(painted) if "bash" in row)
        assert first_tool - working - 1 == 1  # exactly one blank row of air


@pytest.mark.asyncio
async def test_approval_prompt_resolves_from_a_keystroke() -> None:
    """The awaited future settles from the UI, which is the anti-freeze contract."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        assert session.approval_handler is not None

        ask = _approval_gate(session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf /tmp/x"))
        await pilot.pause(0.3)
        block = app.query_one(ApprovalBlock)
        assert app.screen.focused is block  # else the keys go to the composer
        assert any("allow all" in row for row in rows(app))

        await pilot.press("y")
        assert await asyncio.wait_for(pending, 2) is True
        await pilot.pause(0.1)
        assert isinstance(app.screen.focused, Editor)  # focus handed back
        assert any("allowed" in row for row in rows(app))  # decision kept


@pytest.mark.asyncio
async def test_escape_denies_a_pending_approval_without_aborting() -> None:
    """Esc on a question means "don't do it", not "kill the turn"."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        session.streaming = True
        ask = _approval_gate(session)
        pending = asyncio.ensure_future(ask("write", "write: /etc/hosts"))
        await pilot.pause(0.3)

        await pilot.press("escape")
        assert await asyncio.wait_for(pending, 2) is False
        assert session.aborts == []  # the turn continues; only the tool was refused
        assert any("denied" in row for row in rows(app))


@pytest.mark.asyncio
async def test_allow_all_latches_for_the_session() -> None:
    """``a`` answers every later ask without a second prompt."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        ask = _approval_gate(session)
        first = asyncio.ensure_future(ask("bash", "run: make"))
        await pilot.pause(0.3)
        await pilot.press("a")
        assert await asyncio.wait_for(first, 2) is True
        await pilot.pause(0.1)

        # No prompt is mounted for the second ask at all.
        before = len(app.query(ApprovalBlock))
        second = asyncio.ensure_future(ask("write", "write: out.txt"))
        assert await asyncio.wait_for(second, 2) is True
        await pilot.pause(0.1)
        assert len(app.query(ApprovalBlock)) == before


@pytest.mark.asyncio
async def test_interrupt_denies_a_parked_approval() -> None:
    """Ctrl+C must not leave the engine awaiting a future nobody will answer.

    A turn parked in the approval callback cannot observe the abort signal until
    the callback returns, so the prompt has to be settled first or the abort is
    silently ineffective — the same hang, reached a different way.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        session.streaming = True
        ask = _approval_gate(session)
        pending = asyncio.ensure_future(ask("bash", "run: sleep 99"))
        await pilot.pause(0.3)

        app.action_interrupt()
        assert await asyncio.wait_for(pending, 2) is False
        assert session.aborts == ["interrupted"]


@pytest.mark.asyncio
async def test_clearing_the_transcript_settles_a_pending_approval() -> None:
    """The prompt's widget is about to be removed; the awaiting turn is denied."""
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        ask = _approval_gate(session)
        pending = asyncio.ensure_future(ask("bash", "run: make"))
        await pilot.pause(0.3)

        app.query_one(TranscriptView).clear_blocks()
        await pilot.pause(0.1)
        assert await asyncio.wait_for(pending, 2) is False


@pytest.mark.asyncio
async def test_escape_closes_an_open_picker_before_it_stops_anything() -> None:
    """Esc precedence: the editor's open list wins, the turn-stop is the fallback.

    Binding Esc with ``priority=True`` broke this — Textual matches priority
    bindings BEFORE dispatching the key to the focused widget, so the picker's
    own Esc handler (and its ``event.stop()``) never ran and the command/model
    lists could not be dismissed at all. The binding must bubble instead, which
    is what these two halves pin.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        session.streaming = True  # a stop IS available, so precedence is observable

        editor = app.query_one(Editor)
        editor.text = "/"
        await pilot.pause(0.2)
        assert editor.picker.is_open()

        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not editor.picker.is_open()  # the list closed…
        assert session.aborts == []  # …and the turn was NOT stopped

        # The very next Esc stops the turn — no dead press in between, and focus
        # never leaves the composer.
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert session.aborts == ["interrupted"]
        assert app.screen.focused is editor
