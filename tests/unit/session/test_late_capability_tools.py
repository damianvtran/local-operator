"""Tools gated on a capability the session gets AFTER it is constructed.

``Session.__init__`` ends with a capability merge, because ``create_tools`` is
createIf-driven and the factory's ToolContext carries none of the fields only a
session owns. That merge runs ONCE, which is exactly long enough to miss the
``ask`` hook: every real host installs it later (the TUI resolves its session in
a worker and calls ``set_ask_handler`` in ``_adopt_session``), so ``ask`` was
built against ``ask_user=None``, returned ``None``, and was advertised to
nobody — while the system prompt told the model to use it. The per-turn
``_build_tool_context`` could not save it: that context decides what a tool RUNS
against, never whether the tool reached the provider's tools array.

So these assert on the array the provider actually receives, not on
``session._tools``, and the last one is derived rather than enumerated: it fails
when a NEW session-gated builder is added to the registry without joining the
merge set, which is the shape of this bug rather than one instance of it.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AskQuestion,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
)
from local_operator.session.session import SESSION_CAPABILITY_TOOLS, Session
from local_operator.session.transcript import Transcript
from local_operator.tools.registry import create_tools

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class RecordingStream:
    """Answers every turn with one line, keeping the requests it was given."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    def advertised(self) -> list[str]:
        return [tool.name for tool in self.requests[-1].tools]


def make_session(tmp_path, stream: RecordingStream, **kwargs: Any) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        # A host that owns a terminal, which is the only kind that can answer a
        # question: ``build_ask_tool`` requires both the hook and this flag.
        has_ui=True,
        **kwargs,
    )


async def answer_nothing(questions: list[AskQuestion]) -> dict[str, list[str]] | None:
    """A host hook that exists: the tool gates on its presence, not its answer."""
    return None


@pytest.mark.asyncio
async def test_an_ask_handler_installed_after_construction_is_advertised(tmp_path) -> None:
    """The blocker: `ask` reached no model in any real host.

    The handler arrives after ``__init__``, so a merge that only runs there
    leaves the tool out of every request for the life of the session.
    """
    stream = RecordingStream()
    session = make_session(tmp_path, stream)

    await session.prompt("before the front end wires itself up")
    assert "ask" not in stream.advertised()

    session.set_ask_handler(answer_nothing)
    await session.prompt("and now?")

    assert "ask" in stream.advertised()
    # The rescue is additive: nothing the session already advertised moved out.
    assert {"task", "wait", "jobs", "wake", "hub"} <= set(stream.advertised())
    await session.dispose()


@pytest.mark.asyncio
async def test_a_session_with_no_ask_handler_never_advertises_it(tmp_path) -> None:
    """The other half of the gate, and the reason it is a gate: a server, exec
    mode or a subagent has nobody at a keyboard, so a question it could only
    block on must not be offered."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream)

    await session.prompt("hello")

    assert "ask" not in stream.advertised()
    await session.dispose()


@pytest.mark.asyncio
async def test_uninstalling_the_handler_takes_ask_back_off_the_inventory(tmp_path) -> None:
    """A host that hands the terminal back can no longer answer, and a tool
    advertised with no hook behind it can only fail when the model calls it."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session.set_ask_handler(answer_nothing)
    await session.prompt("with a picker")
    assert "ask" in stream.advertised()

    session.set_ask_handler(None)
    await session.prompt("without one")

    assert "ask" not in stream.advertised()
    await session.dispose()


@pytest.mark.asyncio
async def test_installing_ask_does_not_resurrect_a_pruned_capability(tmp_path) -> None:
    """``_build_child_session`` prunes ``wake`` from a subagent's inventory (a
    child ends after one prompt, so a wake armed there is silently lost). The
    late merge re-runs builders, so it is scoped to the one capability that
    arrived — otherwise installing a question surface would quietly hand a
    pruned tool back."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session.refresh_tools([tool for tool in session._tools if tool.name != "wake"])

    session.set_ask_handler(answer_nothing)
    await session.prompt("go")

    advertised = stream.advertised()
    assert "ask" in advertised
    assert "wake" not in advertised
    await session.dispose()


@pytest.mark.asyncio
async def test_every_session_gated_tool_is_in_the_merge_set(tmp_path) -> None:
    """The drift guard. A builder gated on a field only a session can fill is
    invisible unless the merge names it, and the failure is silent: the tool
    simply never appears in any request. Derived from the two contexts rather
    than listed, so a NEW session-gated tool fails here instead of shipping
    unadvertised."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session.set_ask_handler(answer_nothing)

    # What ``session_factory`` builds the inventory from: no session fields.
    factory_side = {
        tool.name for tool in create_tools(ToolContext(cwd=".", session_id="s", has_ui=True))
    }
    session_side = {tool.name for tool in create_tools(session._build_tool_context())}

    assert session_side - factory_side == set(SESSION_CAPABILITY_TOOLS)
    await session.dispose()


def test_the_merge_set_names_only_real_tools() -> None:
    """A typo in the tuple is silent: ``create_tools`` skips unknown names, so
    the tool it was meant to rescue stays missing."""
    from local_operator.tools.registry import TOOL_BUILDERS

    assert set(SESSION_CAPABILITY_TOOLS) <= set(TOOL_BUILDERS)
