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

from typing import Any


from local_operator.harness.types import (  # noqa: E402
    AgentEndEvent,
    AgentStartEvent,
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
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402

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
    def model(self):
        return None

    def set_model(self, model):
        pass

    @property
    def goal(self) -> str:
        return getattr(self, "_goal", "")

    def set_goal(self, text: str) -> str:
        self._goal = (text or "").strip()
        return self._goal

    async def prompt(self, text: str, attachments: list[Any] | None = None) -> None:
        self.prompts.append(text)

    def steer(self, text: str) -> None:
        pass

    def abort(self, reason: str = "interrupted") -> None:
        self.aborts.append(reason)

    def subscribe(self, handler: Any) -> Any:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def dispose(self) -> None:
        self.disposed = True

    def emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)


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
            details={"path": "src/parser.py", "added": 12, "removed": 3},
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
    That is the whole of the "snapshots are flaky" reputation these tests
    had; pinning it here makes the frames byte-stable across runs.
    """
    pilot.app.query_one(Editor).cursor_blink = False


async def _boot_only(pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    _freeze_cursor(pilot)
    await pilot.pause()


async def _populate_and_submit(pilot) -> None:  # type: ignore[no-untyped-def]
    await pilot.pause()
    _freeze_cursor(pilot)
    await pilot.pause()
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
