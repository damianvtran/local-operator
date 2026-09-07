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

Asserting on the array alone was not enough, which is why the second half of
this file exists. Reaching the array only makes a tool CALLABLE; the model also
has to be told it exists, and it is told by the system prompt's
"## Available tools" inventory. Those two came from different objects: the
prompt provider closed over the list the factory built, while ``Session``
copied it and every later change REBOUND the copy. Nothing mutated the shared
list, so the prompt described the factory's inventory for the life of the
session and all six capability tools were advertised to a model never told they
existed (0 of 880 rendered inventories across 1056 local transcripts named
``ask``). The tests below therefore bind the prompt TO the array rather than
checking either alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AskQuestion,
    ChatRequest,
    CustomMessage,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
)
from local_operator.prompts_api import TOOL_INVENTORY_HEADING, build_system_blocks
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

    def described(self) -> list[str]:
        """The tool names the model is EFFECTIVELY told about in the last request.

        Not simply the system blocks. Once the prefix is frozen those blocks
        are re-sent verbatim and later inventories arrive as ``[session-state]``
        records at the history tail, where each supplied section REPLACES the
        earlier snapshot — so the newest inventory the request carries, in
        either place, is what the model reads. Reading only the blocks would
        report a stale answer for exactly the sessions this bug affected;
        reading a session attribute would miss the wire entirely.
        """
        request = self.requests[-1]
        sources = list(request.system_blocks or [])
        # Rendered history, so a state record is a plain ``Message`` by the time
        # it reaches here: its custom type is gone and its text is exactly the
        # bytes the provider receives.
        sources += [
            message.text
            for message in request.messages
            if (message.text or "").startswith("[session-state]")
        ]
        latest = [text for text in sources if TOOL_INVENTORY_HEADING in text]
        if not latest:
            return []
        section = latest[-1].split(TOOL_INVENTORY_HEADING, 1)[1]
        names: list[str] = []
        for line in section.splitlines():
            if line.startswith("## "):  # the next section of a multi-part delta
                break
            if line.startswith("- "):
                names.append(line[2:].split(":", 1)[0])
        return names


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


def make_prompt_session(tmp_path, stream: RecordingStream) -> Session:
    """A session whose prompt is REAL, wired exactly as the shipped hosts wire it.

    ``make_session`` above returns a fixed one-block prompt, which is fine for
    array assertions and useless here. Both production providers
    (``session_factory._make_system_blocks_provider`` and
    ``subagent``'s) close over the tool list they were built with and hand the
    SAME list to ``Session`` — so this closure reproduces the shipped wiring
    including the bug it used to carry. ``append_only_state`` marks it as
    opting into the persisted-prefix protocol, as the real ones do.
    """
    tools: list[Any] = []

    def provider(model_label: str = "") -> list[str]:
        return build_system_blocks(tools, "", "", "2026-09-07")

    setattr(provider, "append_only_state", True)
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools,
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=provider,
        has_ui=True,
    )


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


@pytest.mark.asyncio
async def test_a_late_installed_ask_is_described_in_the_prompt_not_just_advertised(
    tmp_path,
) -> None:
    """The blocker, and the half the array tests could not see.

    ``ask`` reached the tools array as soon as the handler was installed, so
    every assertion in the first half of this file passed while the prompt went
    on describing the factory's empty inventory. The model was offered a schema
    it was never told about, and the field evidence is unambiguous: 104 ask
    calls in 1056 local transcripts, 0 of 880 rendered inventories naming it.
    """
    stream = RecordingStream()
    session = make_prompt_session(tmp_path, stream)
    session.set_ask_handler(answer_nothing)

    await session.prompt("go")

    assert "ask" in stream.described()
    # The other five capabilities merge in ``__init__`` rather than later, and
    # were equally undescribed: the prompt was frozen, not merely late.
    assert set(SESSION_CAPABILITY_TOOLS) <= set(stream.described())
    await session.dispose()


@pytest.mark.asyncio
async def test_no_advertised_tool_is_ever_missing_from_the_prompt_inventory(tmp_path) -> None:
    """The invariant, derived rather than enumerated.

    This is the guard that matters: it does not name ``ask``, so it fails for
    ANY tool that reaches the provider without reaching the prompt — the next
    session-gated capability, an MCP tool spliced in mid-session, a role's
    rebuilt inventory. One instance of this bug was fixed before; asserting on
    the array alone let the same class ship again.
    """
    stream = RecordingStream()
    session = make_prompt_session(tmp_path, stream)
    session.set_ask_handler(answer_nothing)

    await session.prompt("go")

    described = set(stream.described())
    advertised = {tool.name for tool in stream.requests[-1].tools if not tool.hidden}
    assert advertised - described == set()
    # And the converse: nothing is promised that the request cannot serve.
    assert described - {tool.name for tool in stream.requests[-1].tools} == set()
    await session.dispose()


@pytest.mark.asyncio
async def test_uninstalling_the_handler_takes_ask_out_of_the_prompt_too(tmp_path) -> None:
    """A host that hands the terminal back stops offering ``ask`` in the array;
    the prompt has to stop naming it in the same turn, or the model is told to
    use a tool the request no longer carries."""
    stream = RecordingStream()
    session = make_prompt_session(tmp_path, stream)
    session.set_ask_handler(answer_nothing)
    await session.prompt("with a picker")
    assert "ask" in stream.described()

    session.set_ask_handler(None)
    await session.prompt("without one")

    assert "ask" not in stream.described()
    assert "ask" not in stream.advertised()
    await session.dispose()


@pytest.mark.asyncio
async def test_a_changed_inventory_rides_a_state_delta_and_keeps_block_zero(tmp_path) -> None:
    """The cache contract: fixing the inventory must not re-anchor the prefix.

    Block 0 is the byte-stable cached head and blocks 1+ ride ``[session-state]``
    deltas (``Session._prepare_system_blocks``). The inventory is block 1, so a
    capability arriving mid-session is journalled the way a changed goal or
    environment is. Re-rendering it into the head instead would invalidate the
    whole provider prefix on the first turn after the front end wires itself
    up — i.e. on essentially every real session.
    """
    stream = RecordingStream()
    session = make_prompt_session(tmp_path, stream)
    await session.prompt("before the front end wires itself up")
    first_blocks = list(stream.requests[0].system_blocks or [])
    assert "ask" not in stream.described()

    session.set_ask_handler(answer_nothing)
    await session.prompt("and now?")

    # The head is untouched, and the persisted prefix is re-sent verbatim.
    assert (stream.requests[-1].system_blocks or [])[:1] == first_blocks[:1]
    assert list(stream.requests[-1].system_blocks or []) == first_blocks
    # The change reached the model as a journalled state record instead.
    updates = [
        message
        for message in session._context.messages
        if isinstance(message, CustomMessage) and message.custom_type == "session_state"
    ]
    assert len(updates) == 1
    # Names only: the inventory deliberately carries no descriptions (schemas
    # ride the tools array), so match the whole line rather than a prefix.
    assert "\n- ask\n" in str(updates[-1].details["text"])
    await session.dispose()
