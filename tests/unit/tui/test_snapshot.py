"""SVG snapshot regression for the composed TUI screens.

Covers the three canonical frames: fresh boot, a populated transcript
(user echo, streamed markdown, three tool cards with one error), and the
narrow 80-column terminal. Shimmer is pinned OFF so frames are
deterministic (D26's static fallback stays legible in stills).

Regenerate after intentional visual changes with::

    env -u NO_COLOR TERM=xterm-256color \\
        .venv/bin/python -m pytest tests/unit/tui/test_snapshot.py \\
        --snapshot-update
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import pytest

#: Golden-SVG comparison is opt-in because Textual's SVG output is not
#: byte-stable across interpreters, OSes, or container images (it flapped
#: between ubuntu CI legs, docker bookworm, and macOS). The portable visual
#: signal lives in the text-based TUI assertions (minimalism/spacing/tool_card)
#: plus live screenshots; these SVG stills are a local design aid, run with::
#:
#:     LO_RUN_SNAPSHOTS=1 env -u NO_COLOR TERM=xterm-256color \\
#:         .venv/bin/python -m pytest tests/unit/tui/test_snapshot.py
#:
#: and regenerated with the same env plus ``--snapshot-update``.
pytestmark = pytest.mark.skipif(
    not os.environ.get("LO_RUN_SNAPSHOTS"),
    reason="SVG goldens are environment-bound; opt in with LO_RUN_SNAPSHOTS=1",
)

from local_operator.harness.types import (  # noqa: E402
    AgentEndEvent,
    AgentStartEvent,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
    TurnEndEvent,
    TurnStartEvent,
    Usage,
)
from local_operator.session.naming import ConversationName  # noqa: E402
from local_operator.session.protocol import CompactionOutcome  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.welcome import WelcomeView  # noqa: E402

MARKDOWN = (
    "Here is the **plan** for `parser.py`:\n"
    "\n"
    "- tokenize the input\n"
    "- build the AST\n"
    "\n"
    "```python\n"
    "def parse(src: str) -> AST:\n"
    "    return AST(lex(src))\n"
    "```\n"
    "\n"
    "Running the checks now."
)


class FakeSession:
    """Records prompts/aborts; satisfies SessionProtocol."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.aborts: list[str] = []
        self.disposed = False
        self._handlers: list[Any] = []
        self.asides: list[list[Any]] = []
        self.adopted: list[list[Any]] = []

    @property
    def session_id(self) -> str:
        return "sess"

    @property
    def agent_id(self) -> str:
        return "agent"

    @property
    def is_streaming(self) -> bool:
        return False

    @property
    def model_label(self) -> str:
        return "test/model"

    @property
    def model(self) -> Any:
        return None

    @property
    def effective_model(self) -> Any:
        # The fake never falls back, so selection and effective agree.
        return self.model

    @property
    def effective_model_label(self) -> str:
        return self.model_label

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def seed_history(self, messages: list[Any]) -> None:
        pass

    async def prompt(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str, images: Sequence[ImageContent] | None = None) -> None:
        pass

    def set_approval_handler(self, handler: object | None) -> None:
        # The TUI installs its own approval gate on boot (the stdin gate
        # deadlocks under a full-screen app); fakes only need to accept it.
        self.approval_handler = handler

    def set_ask_handler(self, handler: object | None) -> None:
        # The TUI installs the `ask` tool's picker surface on boot, and that
        # install is what makes the tool exist; fakes only need to accept it.
        self.ask_handler = handler

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def running_subagents(self) -> int:
        """No subagents in this fake; the protocol requires the method."""
        return 0

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    @property
    def conversation_name(self) -> str:
        return self.conversation_name_state.text

    @property
    def conversation_name_state(self) -> ConversationName:
        # The real holder, created on first read: `user_set` precedence (a
        # human rename outranks every generated title, forever) is behaviour
        # the TUI reads before it spends a re-title call, so a fake that
        # reimplemented it as a bare string would hide a regression in it.
        state = getattr(self, "_name_state", None)
        if state is None:
            state = self._name_state = ConversationName()
        return state

    def set_conversation_name(self, text: str, *, user_set: bool = True) -> str:
        return self.conversation_name_state.set(text, user_set=user_set)

    async def complete_once(self, system: str, prompt: str) -> str:
        return ""

    async def dispose(self) -> None:
        self.disposed = True

    def history(self) -> list[Any]:
        return getattr(self, "_history", [])

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)

    async def complete_aside(
        self,
        turns: list[Any],
        *,
        on_delta: Callable[[str], None] | None = None,
        on_usage: Callable[[Any], None] | None = None,
    ) -> str:
        # Recorded, not answered: the aside's no-trace contract is proven
        # against the real Session in tests/unit/session/test_aside.py. Here
        # the only thing that must hold is that the app can call it.
        self.asides.append(list(turns))
        return ""

    async def adopt_aside(self, messages: list[Any]) -> None:
        self.adopted.append(list(messages))

    async def compact_now(self) -> CompactionOutcome:
        # No history to compact: this fake never carries a conversation, which
        # is the state a real session answers with the same refusal.
        return CompactionOutcome(
            ran=False, reason="nothing_to_compact", detail="nothing to compact"
        )


async def _factory(session: FakeSession) -> FakeSession:
    return session


def _tool_end(
    call_id: str,
    name: str,
    text: str,
    is_error: bool = False,
    details: dict[str, Any] | None = None,
) -> ToolExecutionEndEvent:
    from local_operator.harness.types import TextContent

    return ToolExecutionEndEvent(
        tool_call_id=call_id,
        tool_name=name,
        result=ToolResult(
            tool_call_id=call_id,
            tool_name=name,
            content=[TextContent(type="text", text=text)],
            is_error=is_error,
            details=details,
        ),
    )


def _populate(session: FakeSession) -> None:
    """Drive one full turn: streamed markdown + four tool cards (one error).

    The cards deliberately cover the full row vocabulary — a plain success,
    a failure, a result with nothing to expand, and a write carrying diff
    counters — so the frame regression actually sees every segment the row
    can grow.
    """
    session.emit(AgentStartEvent())
    session.emit(TurnStartEvent())
    session.emit(MessageStartEvent(message=Message.assistant("")))
    session.emit(MessageUpdateEvent(message=Message.assistant(MARKDOWN), delta=MARKDOWN))
    session.emit(MessageEndEvent(message=Message.assistant(MARKDOWN)))
    session.emit(
        ToolExecutionStartEvent(
            tool_call_id="t1", tool_name="bash", args={"command": "pytest tests/unit -q"}
        )
    )
    session.emit(_tool_end("t1", "bash", "66 passed"))
    session.emit(
        ToolExecutionStartEvent(tool_call_id="t2", tool_name="grep", args={"pattern": "parse"})
    )
    session.emit(_tool_end("t2", "grep", "permission denied while reading the file", is_error=True))
    session.emit(
        ToolExecutionStartEvent(tool_call_id="t3", tool_name="read", args={"path": "src/parser.py"})
    )
    session.emit(_tool_end("t3", "read", "ok"))
    session.emit(
        ToolExecutionStartEvent(
            tool_call_id="t4",
            tool_name="write",
            args={"path": "src/parser.py", "content": "def parse(src):\n    ...\n"},
        )
    )
    session.emit(
        _tool_end(
            "t4",
            "write",
            "Overwrote src/parser.py (412 chars).",
            details={
                "path": "src/parser.py",
                "added": 12,
                "removed": 3,
                "diff": [
                    "--- ",
                    "+++ ",
                    "@@ -1,6 +1,8 @@",
                    " def parse(src):",
                    "     tokenize(src)",
                    "+    build_ast(tokens)",
                    "+    check(ast)",
                    "     return ast",
                    "-    # old single-line comment",
                ],
            },
        )
    )
    # Engine-faithful: the loop always emits turn_end carrying the assistant
    # message and its tool results; the status line reads usage off it, so a
    # bare event freezes frames the live app can never show.
    session.emit(
        TurnEndEvent(
            message=Message.assistant(
                MARKDOWN,
                usage=Usage(context_tokens=12400, input_tokens=12000, output_tokens=800),
            ),
            tool_results=[_tool_end("t1", "bash", "66 passed").result],
        )
    )
    session.emit(
        AgentEndEvent(
            messages=[
                Message.assistant(
                    "done", usage=Usage(context_tokens=12400, input_tokens=12000, output_tokens=800)
                )
            ]
        )
    )


def _make_app() -> tuple[OperatorApp, FakeSession]:
    session = FakeSession()
    return OperatorApp(lambda: _factory(session)), session


def _freeze_cursor(pilot) -> None:  # type: ignore[no-untyped-def]
    """Stop the editor's cursor blinking.

    The blink is a timer, so whether the frame catches the cursor ON or OFF
    is a race — and a single inverted cell is enough to fail an SVG compare.
    """
    pilot.app.query_one(Editor).cursor_blink = False


async def _settle(pilot, ticks: int = 24) -> None:  # type: ignore[no-untyped-def]
    """Pause until the boot frame stops changing, not for a fixed count.

    ``WelcomeView`` polls every 250 ms until the session reports a model label,
    then repaints once and stops its own timer (``WelcomeView._sync_timer``).
    A fixed ``pilot.pause()`` count races that last tick: both outcomes render
    the SAME characters, so the frame LOOKS identical, but the repaint splits
    the row into different Rich segments — and ``export_svg`` derives its
    element-id prefix from ``adler32`` over the segment reprs, so a byte compare
    fails on an id that has nothing to do with what the user would see.

    That is the residual "snapshots are flaky" tail after the cursor blink was
    pinned: roughly 1 run in 6. Waiting for the timer to retire makes the
    captured frame the settled one every time, and asserts a real property —
    the boot frame reaches a steady state — instead of hiding the race behind a
    normalised comparison.
    """
    welcome = pilot.app.query_one(WelcomeView)
    for _ in range(ticks):
        await pilot.pause()
        if welcome._timer is None:
            break
    # One more so the retiring tick's repaint is composited before capture.
    await pilot.pause()


async def _boot_only(pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    _freeze_cursor(pilot)
    await _settle(pilot)


async def _populate_and_submit(pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    _freeze_cursor(pilot)
    await _settle(pilot)
    pilot.app.query_one(Editor).focus()
    await pilot.pause()
    for key in "parse the source":
        await pilot.press(key if key != " " else "space")
    await pilot.press("enter")
    await pilot.pause()
    _populate(pilot.app._session)
    await pilot.pause()
    await pilot.pause()


def test_boot_snapshot(snap_compare) -> None:  # type: ignore[no-untyped-def]
    app, _session = _make_app()
    assert snap_compare(app, terminal_size=(100, 28), run_before=_boot_only)


def test_populated_snapshot(snap_compare) -> None:  # type: ignore[no-untyped-def]
    app, _session = _make_app()
    assert snap_compare(app, terminal_size=(100, 28), run_before=_populate_and_submit)


def test_narrow_80_snapshot(snap_compare) -> None:  # type: ignore[no-untyped-def]
    app, _session = _make_app()
    assert snap_compare(app, terminal_size=(80, 24), run_before=_populate_and_submit)


def test_write_diff_expanded_snapshot(snap_compare) -> None:  # type: ignore[no-untyped-def]
    """The expanded write card shows the rendered diff, colourised by hunk.

    Distinct from the collapsed populated frame: the write card is expanded so
    the designer and the regression both see the +/-, @@ and context lines in
    their tint ramp. The band (todo/subagent) stays empty here — this frame is
    about the diff, not the band.
    """

    async def expand_then_capture(pilot) -> None:  # type: ignore[no-untyped-def]
        await _populate_and_submit(pilot)
        # Find the write card (the 4th, tool t4) and expand it.
        from local_operator.tui.widgets.tool_card import ToolCard

        cards = list(pilot.app.query(ToolCard))
        write_card = next((c for c in cards if c.tool_name == "write"), None)
        assert write_card is not None, "populated frame must include a write card"
        write_card.toggle_expanded()
        await pilot.pause()
        await pilot.pause()

    app, _session = _make_app()
    assert snap_compare(app, terminal_size=(100, 28), run_before=expand_then_capture)
