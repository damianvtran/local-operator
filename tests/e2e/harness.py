"""Building blocks shared by the end-to-end tests: real app, real session.

The rule this module exists to hold is that as little as possible is faked.
The tests here are the answer to "the whole unit suite was green while the app
was completely frozen", so anything replaced by a double is a place the stage
cannot see. What IS replaced, and why:

* **The provider stream.** A scripted ``stream_fn`` replaces the network, not
  the harness: the real :class:`~local_operator.session.session.Session` still
  runs its real agent loop over the scripted events, executes real tools and
  writes a real transcript. This is what makes the stage deterministic enough
  to run on every PR including forks, which is where the regression under test
  has to be caught. See ``tests/e2e/test_tui_e2e.py``'s module docstring for
  the full split.

* **OAuth endpoint discovery.** Two HTTP round trips to a server that does not
  exist. Replaced by the SDK's own documented fallback shape so the refresh
  path proceeds to the lock — which is the part under test — without a network.

Everything else is production code: ``OperatorApp``, ``Session``,
``Transcript``, ``McpManager``, the real tool implementations, and the real
cross-process OAuth refresh lock.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterable, Sequence
from pathlib import Path
from typing import Any

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

#: A model spec no provider is ever asked about. ``provider="test"`` keeps the
#: pricing and discovery paths on their unknown-model branches instead of
#: reaching the model registry over the network.
TEST_MODEL = ModelSpec(provider="test", model_id="e2e-model", context_window=100_000)

#: How long the pilot may wait for the app to adopt its session before the
#: test calls it a failure. Generous relative to the real cost (adoption is a
#: handful of loop turns) because a slow CI runner must not flake; short enough
#: that a genuinely stuck boot is reported by the assertion rather than by the
#: watchdog, which gives a clearer message.
ADOPT_TIMEOUT_S = 20.0


class ScriptedStream:
    """Replays one canned event list per model call; records the requests.

    Mirrors ``tests/unit/session/test_session.py``'s stream double deliberately
    — the session's contract with a provider is one place, and a second shape
    for it here would be a second thing to keep in step.
    """

    def __init__(self, turns: Sequence[Sequence[StreamEvent]]) -> None:
        self.turns = [list(turn) for turn in turns]
        self.requests: list[ChatRequest] = []

    def __call__(
        self, request: ChatRequest, signal: AbortSignal | None = None
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        # A call past the end of the script is a test bug (the loop re-entered
        # when the author expected it to stop), and an IndexError names it
        # exactly. Answering with a bare stop would hide the extra turn.
        turn = self.turns[len(self.requests) - 1]

        async def gen() -> AsyncIterator[StreamEvent]:
            for event in turn:
                yield event

        return gen()


def tool_call_turn(
    *,
    text: str,
    tool_name: str,
    tool_call_id: str,
    arguments: dict[str, Any],
) -> list[StreamEvent]:
    """One model turn that says something and then asks for one tool call."""
    return [
        StreamTextDelta(delta=text),
        StreamToolCallDelta(
            index=0,
            id=tool_call_id,
            name=tool_name,
            argument_delta=json.dumps(arguments),
        ),
        StreamEndEvent(stop_reason="toolUse"),
    ]


def text_turn(text: str) -> list[StreamEvent]:
    """One model turn that just answers and stops."""
    return [StreamTextDelta(delta=text), StreamEndEvent(stop_reason="stop")]


def build_session(
    directory: Path,
    stream: Any,
    *,
    tools: Iterable[Any] = (),
    cwd: Path | None = None,
) -> Session:
    """A REAL session over a real transcript directory.

    ``yolo=True`` because these tests drive turns, not the approval prompt:
    with the gate armed every tool call would park waiting for a keypress the
    test is not sending, and the approval surface has its own dedicated unit
    coverage (``tests/unit/tui/test_approvals_ux.py``).
    """
    return Session(
        model=TEST_MODEL,
        stream_fn=stream,
        tools=list(tools),
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
        yolo=True,
        cwd=str(cwd) if cwd is not None else None,
    )


async def seed_transcript(directory: Path, messages: Sequence[Message]) -> Transcript:
    """Write ``messages`` to a session directory as a prior conversation.

    This is what makes ``/resume`` a real resume rather than a screen swap:
    the resumed session is built over a directory that already holds a
    transcript on disk, exactly as a session closed yesterday would.
    """
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    for message in messages:
        await transcript.append_message(message)
    transcript.flush()
    return transcript


def user_message(text: str) -> Message:
    return Message(role="user", content=[TextContent(text=text)])


def assistant_message(text: str) -> Message:
    return Message(role="assistant", content=[TextContent(text=text)], stop_reason="stop")


async def wait_for_adoption(app: Any, pilot: Any, timeout_s: float = ADOPT_TIMEOUT_S) -> None:
    """Pump the pilot until the app has adopted a session, or fail saying so.

    Polling rather than awaiting an event because adoption happens inside a
    Textual worker the test does not own a handle to; this is the same shape
    ``tests/unit/tui/test_resumed_conversation_name.py`` uses.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if app._session is not None:
            return
        await pilot.pause()
    raise AssertionError(
        f"the app never adopted a session within {timeout_s:g}s; "
        f"transcript on screen: {transcript_text(app)!r}"
    )


def transcript_text(app: Any) -> str:
    """Every transcript block flattened to the plain text a user would read.

    Reads what was actually COMPOSED rather than what the session holds, which
    is the distinction the resume test needs: a session can carry history that
    never reached the screen, and that is precisely the bug shape here.
    """
    from local_operator.tui.widgets.transcript import TranscriptView
    from tests.unit.tui.test_app_pilot import _renderable_plain

    try:
        view = app.query_one(TranscriptView)
    except Exception:  # noqa: BLE001 — a screen with no transcript reads as empty
        return ""
    return "\n".join(_renderable_plain(getattr(block, "renderable", "")) for block in view.blocks())


async def drain(pilot: Any, cycles: int = 30) -> None:
    """Let the app settle: mount, layout, paint, and any queued workers."""
    for _ in range(cycles):
        await pilot.pause()


class LoopLiveness:
    """Counts how many times the event loop came back to us, and how late.

    The distinction this class exists to draw: a test that merely waits for a
    frame to appear passes on a loop that painted once and then died. What has
    to be asserted after ``/resume`` is that the loop is STILL SCHEDULING —
    that it keeps handing control back at roughly the cadence asked for. So
    this records both the number of resumptions and the worst gap between them.

    Note that on a fully deadlocked process this object never gets to report
    anything at all: the freeze under test parks the loop thread inside a
    syscall, so no counter advances and nothing raises. That is not a gap in
    this class — it is exactly why :mod:`tests.e2e.watchdog` exists and why the
    liveness assertion must run INSIDE a ``bounded`` block.
    """

    def __init__(self) -> None:
        self.resumptions = 0
        self.worst_gap_s = 0.0

    async def observe(self, pilot: Any, seconds: float, interval_s: float = 0.02) -> None:
        """Pump the pilot for ``seconds``, recording cadence as it goes."""
        deadline = time.monotonic() + seconds
        last = time.monotonic()
        while time.monotonic() < deadline:
            await pilot.pause(interval_s)
            now = time.monotonic()
            self.worst_gap_s = max(self.worst_gap_s, now - last)
            last = now
            self.resumptions += 1

    def assert_alive(self, *, minimum: int, ceiling_s: float, context: str) -> None:
        """The loop kept scheduling, and never went quiet for ``ceiling_s``."""
        assert self.resumptions >= minimum, (
            f"the event loop scheduled only {self.resumptions} times after {context} "
            f"(expected at least {minimum}): the loop stopped servicing work"
        )
        assert self.worst_gap_s < ceiling_s, (
            f"the event loop went quiet for {self.worst_gap_s:.2f}s after {context} "
            f"(ceiling {ceiling_s:g}s): the app froze rather than kept painting"
        )


async def dispose_quietly(*sessions: Any) -> None:
    """Tear sessions down without letting teardown noise mask a real failure.

    Bounded, because a session whose MCP manager is wedged is exactly what
    these tests construct on purpose, and an unbounded dispose in a teardown
    path would convert an informative assertion failure into a hang.
    """
    for session in sessions:
        if session is None:
            continue
        try:
            await asyncio.wait_for(session.dispose(), timeout=10.0)
        except BaseException:  # noqa: BLE001 — teardown must never mask the real error
            pass
