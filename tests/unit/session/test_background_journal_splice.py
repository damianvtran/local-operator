"""A background journal append must never land INSIDE an open tool batch.

The failure these tests pin down was observed live (session ``5187e748833c``):
a user pressed ``/model`` while the agent was running a tool batch, and every
turn from then on — including a bare "Continue" — died on the same Anthropic
400::

    messages.2: `tool_use` ids were found without `tool_result` blocks
    immediately after: toolu_019T5KXtGAzsqT7jbDbRyecz, toolu_01KW58PLNHHQv75fBBvqz53h

The PERSISTED transcript of that session is well formed — every ``tool_use``
is followed by its ``tool_result``, and replaying it through
``_default_convert_to_llm`` + ``AnthropicClient._build_body`` produces a legal
body. So this is not transcript corruption. The malformed thing was the LIVE
``_context.messages`` list, and it was malformed because of an ordering the
transcript cannot show:

1. ``AgentLoop._run`` appends the assistant message the moment the model turn
   ends and appends the tool results only once the WHOLE batch returns
   (``_append_results``). For the entire duration of a tool batch the live list
   therefore ends in an assistant message whose ``tool_calls`` have no answers.
2. ``Session.set_model`` ends by firing ``journal_model_switch`` through
   ``_spawn_background`` — "Background because ``set_model`` is sync and runs
   on the UI loop".
3. ``journal_model_switch`` (and ``journal_incident``) did a bare
   ``self._context.messages.append(message)`` with NO turn-boundary guard, and
   ``_default_convert_to_llm`` renders both custom types as ``role="user"``.

The fix is two layers, and this file tests both:

- **Layer 1** parks the LIVE append (never the transcript write) whenever a
  turn owns the message list, and drains the FIFO at the turn-safe boundaries.
  That closes the source.
- **Layer 2** repairs at request assembly (``_pair_spliced_tool_results`` in
  ``_render_history``), which is the only thing that can help a session already
  bricked in memory or a future writer that bypasses Layer 1.

The determinism here is not a sleep. The fake tool calls ``set_model`` and then
drains the session's own background task set to completion before returning its
result, so the interleaving under test is forced rather than raced.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.providers.clients import (
    AnthropicClient,
    _messages_to_openai_responses,
)
from local_operator.session.session import (
    SESSION_MODEL_SWITCH_MESSAGE_TYPE,
    CustomMessage,
    Session,
    _pair_spliced_tool_results,
)

from .test_session import MODEL, ScriptedStream, make_session

#: What the user switches TO mid-batch. A different provider AND model id, so
#: ``set_model`` takes the genuine-switch path rather than the same-pair knob
#: early return that never journals. Mirrors the live incident.
SWITCHED = ModelSpec(provider="anthropic", model_id="claude-opus-5", context_window=200_000)

#: The spec the assertions build a wire body against. Only the pair matters —
#: ``_build_body`` reads ``model_id`` and the token ceiling, nothing routing.
WIRE_MODEL = SWITCHED


async def _drain_background(session: Session) -> None:
    """Run every task ``_spawn_background`` has outstanding to completion.

    This is what makes the repro deterministic: the bug needs the background
    journal append to land while the tool batch is still open, and the honest
    way to force that ordering is to drive the spawned task at the chosen
    point rather than sleeping and hoping the scheduler cooperates. Looped
    because a drained task can spawn another (the selection write and the
    switch notice are two separate spawns), and bounded so a self-respawning
    task fails the test instead of hanging it.
    """
    for _ in range(10):
        pending = [task for task in list(session._background_tasks) if not task.done()]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
    raise AssertionError("background tasks never settled")


def _switching_tool(holder: dict[str, Session]) -> AgentTool:
    """A tool that switches the model while its own batch is still open.

    Stands in for the user pressing ``/model`` mid-turn: the TUI's key handler
    calls ``set_model`` on the same loop, at a moment when the assistant
    message is already in the live context and no tool result is.
    """

    async def execute(tool_call_id, args, signal, on_update, context):
        session = holder["session"]
        session.set_model(SWITCHED, explicit=True)
        # The switch's journal is fire-and-forget; force it to land HERE,
        # before this result is appended, which is the window under test.
        await _drain_background(session)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    return AgentTool(name="echo", parameters={"type": "object"}, execute=execute)


def _tool_then_text() -> ScriptedStream:
    """Two calls: a two-call tool batch, then a closing text turn.

    Two tool calls rather than one because the live 400 named two ids, and a
    batch of two proves the splice lands ahead of the whole group rather than
    interleaving with it.
    """
    return ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="toolu_A", name="echo", argument_delta="{}"),
                StreamToolCallDelta(index=1, id="toolu_B", name="echo", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="again"), StreamEndEvent(stop_reason="stop")],
        ]
    )


def _wire_messages(request: ChatRequest) -> list[dict[str, Any]]:
    """The body Anthropic actually receives for ``request``."""
    return AnthropicClient()._build_body(
        ChatRequest(
            model=WIRE_MODEL,
            system_blocks=list(request.system_blocks),
            messages=list(request.messages),
            tools=[],
        )
    )["messages"]


def _dangling_tool_use(messages: list[dict[str, Any]]) -> list[str]:
    """``tool_use`` ids Anthropic will reject as unanswered.

    "Immediately after" is stricter than "somewhere in the next message", and
    the difference is the whole bug. ``AnthropicClient._build_body`` COALESCES
    tool results onto a preceding user message, so the splice does not survive
    as a separate user turn on the wire — it becomes a text block sitting AHEAD
    of the tool_result blocks inside one user message. That still 400s.

    Both halves were verified against the live API (claude-sonnet-4-5,
    2026-08-25), which is why the rule is written as a leading-run check
    rather than a set membership:

        [tool_result, tool_result]           -> HTTP 200
        [tool_result, tool_result, text]     -> HTTP 200
        [text, tool_result, tool_result]     -> HTTP 400, names both ids
        text as its own user message between -> HTTP 400, names both ids

    So only the tool_results in the next message's LEADING run count as
    answers; anything the splice pushes in front of them orphans the whole
    batch.
    """
    dangling: list[str] = []
    for index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            continue
        ids = [block["id"] for block in content if block.get("type") == "tool_use"]
        if not ids:
            continue
        following = messages[index + 1] if index + 1 < len(messages) else None
        answered: set[str] = set()
        if following is not None and isinstance(following.get("content"), list):
            for block in following["content"]:
                if block.get("type") != "tool_result":
                    break  # the leading run ends at the first non-result block
                answered.add(block.get("tool_use_id"))
        dangling.extend(id_ for id_ in ids if id_ not in answered)
    return dangling


def _responses_dangling(items: list[dict[str, Any]]) -> list[str]:
    """``function_call`` ids the OpenAI Responses wire leaves unanswered.

    The same positional rule as :func:`_dangling_tool_use`, expressed for the
    Responses item stream: a run of ``function_call`` items must be followed
    immediately by their ``function_call_output`` items. A bare ``role:user``
    item spliced between them is the shape ``clients.py`` emits for a journal
    notice, and it is rejected for the same reason Anthropic rejects the
    coalesced form.
    """
    dangling: list[str] = []
    index = 0
    while index < len(items):
        if items[index].get("type") != "function_call":
            index += 1
            continue
        calls: list[str] = []
        while index < len(items) and items[index].get("type") == "function_call":
            calls.append(items[index].get("call_id", ""))
            index += 1
        answered: set[str] = set()
        while index < len(items) and items[index].get("type") == "function_call_output":
            answered.add(items[index].get("call_id", ""))
            index += 1
        dangling.extend(call for call in calls if call not in answered)
    return dangling


def _roles(session: Session) -> list[str]:
    """The live context's shape, customs named by their type."""
    return [
        getattr(message, "role", None) or f"custom:{getattr(message, 'custom_type', '?')}"
        for message in session._context.messages
    ]


def _switch_notice(text: str = "switched to anthropic/claude-opus-5") -> CustomMessage:
    return CustomMessage(
        custom_type=SESSION_MODEL_SWITCH_MESSAGE_TYPE,
        attribution="system",
        details={"text": text, "new_label": "anthropic/claude-opus-5", "transient": False},
    )


def _batch(*call_ids: str) -> list[Message]:
    """``assistant(tool_calls) + its answers`` — a legal, closed tool batch."""
    assistant = Message.assistant("")
    assistant.tool_calls = [ToolCall(id=call_id, name="echo", arguments={}) for call_id in call_ids]
    results = [
        Message(
            role="tool",
            content=[TextContent(text="ok")],
            tool_call_id=call_id,
            tool_name="echo",
        )
        for call_id in call_ids
    ]
    return [assistant, *results]


# --- Layer 1: the live context never takes the splice -------------------------


@pytest.mark.asyncio
async def test_a_mid_batch_model_switch_does_not_splice_into_the_tool_batch(tmp_path):
    """The switch notice lands AFTER the batch, not between its halves.

    Asserted on the request the loop ACTUALLY issued for the turn's second
    provider call, not on a list reconstructed by the test: the corruption has
    to be absent on the wire to explain why the 400 is gone.
    """
    holder: dict[str, Session] = {}
    stream = _tool_then_text()
    session = make_session(tmp_path, stream, tools=[_switching_tool(holder)])
    holder["session"] = session
    try:
        await session.prompt("go")
    finally:
        await session.dispose()

    # The live list: the notice waited for the batch to close instead of
    # splicing between the assistant that asked for the tools and its results.
    assert _roles(session) == [
        "user",
        "assistant",
        "tool",
        "tool",
        "custom:session_model_switch",
        "assistant",
    ], f"unexpected live shape: {_roles(session)}"

    # And through the real renderer + the real Anthropic body builder.
    assert len(stream.requests) >= 2, "the turn never made its post-batch call"
    wire = _wire_messages(stream.requests[1])
    shape = [
        (message["role"], [block.get("type") for block in message["content"]]) for message in wire
    ]
    assert _dangling_tool_use(wire) == [], (
        "the switch notice orphaned the tool batch on the wire: every tool_use "
        f"must be answered by the leading tool_result run of the next message. wire was {shape}"
    )
    # The notice must still REACH the model — parking is a delay, not a drop.
    assert any(
        "claude-opus-5" in str(block.get("text", ""))
        for message in wire
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "text"
    ), "the parked switch notice never reached the model"


@pytest.mark.asyncio
async def test_a_later_turn_rides_a_clean_prefix(tmp_path):
    """The session stays usable — this is the "Continue" that used to 400."""
    holder: dict[str, Session] = {}
    stream = _tool_then_text()
    session = make_session(tmp_path, stream, tools=[_switching_tool(holder)])
    holder["session"] = session
    try:
        await session.prompt("go")
        await session.prompt("Continue")
    finally:
        await session.dispose()

    assert len(stream.requests) >= 3, "the follow-up turn never reached the provider"
    wire = _wire_messages(stream.requests[2])
    assert _dangling_tool_use(wire) == [], (
        "the follow-up turn re-sent an unrepaired prefix, so the session could "
        "never recover on its own"
    )


@pytest.mark.asyncio
async def test_journal_incident_is_parked_the_same_way(tmp_path):
    """The sibling door: ``journal_incident`` shares the guard.

    ``_on_mcp_incident`` fires it from a background task and the failover path
    calls it on every provider error, so an incident landing mid-batch would
    corrupt the context exactly as the model switch did. Driven through
    ``_on_mcp_incident`` so the real trigger is covered, not just the method.
    """
    holder: dict[str, Session] = {}
    stream = _tool_then_text()

    async def execute(tool_call_id, args, signal, on_update, context):
        session = holder["session"]
        # Only the FIRST call of the batch trips the breaker. Both calls
        # journalling would prove nothing extra and would make the expected
        # shape depend on batch size rather than on the guard.
        if tool_call_id == "toolu_A":
            session._on_mcp_incident("files", "circuit breaker opened")
            await _drain_background(session)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    tool = AgentTool(name="echo", parameters={"type": "object"}, execute=execute)
    session = make_session(tmp_path, stream, tools=[tool], model=MODEL)
    holder["session"] = session
    try:
        await session.prompt("go")
    finally:
        await session.dispose()

    assert _roles(session) == [
        "user",
        "assistant",
        "tool",
        "tool",
        "custom:session_incident",
        "assistant",
    ], f"unexpected live shape: {_roles(session)}"
    assert len(stream.requests) >= 2
    wire = _wire_messages(stream.requests[1])
    assert (
        _dangling_tool_use(wire) == []
    ), "journal_incident spliced its notice into the open batch, orphaning it"


@pytest.mark.asyncio
async def test_the_direct_await_failover_path_is_guarded_too(tmp_path):
    """``_on_route_settled`` awaits ``journal_model_switch`` ON the turn loop.

    It is not a ``_spawn_background`` caller, so a guard placed at the call
    sites would have missed it entirely. The guard lives INSIDE the journal
    methods precisely so this path is covered for free: a provider failing
    over mid-batch is the single most likely way to hit this window in
    production, because the failover fires exactly when a request is in flight.
    """
    from local_operator.providers.failover import FallbackTarget

    holder: dict[str, Session] = {}
    stream = _tool_then_text()

    async def execute(tool_call_id, args, signal, on_update, context):
        session = holder["session"]
        # Directly awaited, mid-batch, exactly as the failover driver does it.
        await session._on_route_settled(
            FallbackTarget(selector="anthropic/claude-opus-5"), "provider error: HTTP 529"
        )
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="echo", content=[TextContent(text="ok")]
        )

    tool = AgentTool(name="echo", parameters={"type": "object"}, execute=execute)
    session = make_session(tmp_path, stream, tools=[tool], model=MODEL)
    holder["session"] = session
    try:
        await session.prompt("go")
    finally:
        await session.dispose()

    roles = _roles(session)
    assert "custom:session_model_switch" in roles, "the failover notice never reached the context"
    assert roles.index("custom:session_model_switch") > roles.index("tool"), (
        "the failover notice spliced into the open batch instead of waiting for "
        f"it to close: {roles}"
    )
    wire = _wire_messages(stream.requests[1])
    assert _dangling_tool_use(wire) == [], "the direct-await failover path orphaned the batch"


# --- Layer 1: a parked notice is never lost -----------------------------------


@pytest.mark.asyncio
async def test_a_notice_parked_by_an_aborted_turn_is_delivered_at_the_next_prompt(tmp_path):
    """Parking is a delay, never a drop.

    A turn that ends without reaching a steering drain (aborted mid-batch)
    still has to hand its parked notice to the next turn — otherwise the model
    is never told it was switched, which is the very thing the journal exists
    to do.
    """
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="one"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="two"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)
    try:
        # Park by hand with the lock held: this is the state an aborted turn
        # leaves behind, without needing to race a real abort.
        await session._turn_lock.acquire()
        try:
            await session.journal_model_switch("anthropic/claude-opus-5")
            assert len(session._pending_context_journal) == 1, "the notice was not parked"
            assert not any(
                isinstance(message, CustomMessage) for message in session._context.messages
            ), "the notice reached the live context while the lock was held"
        finally:
            session._turn_lock.release()

        await session.prompt("next")
        assert session._pending_context_journal == [], "the parked notice was never drained"
        assert "custom:session_model_switch" in _roles(session)
        # It has to be in the request this prompt built, not merely in the list
        # afterwards: a notice delivered after the turn it was meant for is the
        # same as a lost one.
        sent = stream.requests[0].messages
        assert any(
            "claude-opus-5" in message.text for message in sent
        ), "the drained notice did not reach the request"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_notice_parked_at_dispose_still_reached_the_transcript(tmp_path):
    """The transcript write happens at PARK time, so ``--resume`` is safe even
    if the process dies before any drain boundary runs."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    await session._turn_lock.acquire()
    try:
        await session.journal_model_switch("anthropic/claude-opus-5")
        assert len(session._pending_context_journal) == 1
    finally:
        session._turn_lock.release()
    await session.dispose()

    # Drained into the live list at dispose...
    assert session._pending_context_journal == []
    assert "custom:session_model_switch" in _roles(session)
    # ...and on disk regardless, because the write preceded the park. Asserted
    # through the renderer: `build_llm_history` returns AgentMessage, and it is
    # the RENDERED form that a resumed session would actually send.
    replayed = session._render_history(list(session._transcript.build_llm_history()))
    assert any(
        "claude-opus-5" in message.text for message in replayed
    ), "the parked notice never reached the transcript"


# --- Layer 2: the pure repair -------------------------------------------------


def test_pair_spliced_tool_results_moves_an_interloper_after_the_results():
    assistant, result_a, result_b = _batch("toolu_A", "toolu_B")
    notice = Message.user("switched to anthropic/claude-opus-5")
    spliced = [Message.user("go"), assistant, notice, result_a, result_b]

    repaired = _pair_spliced_tool_results(spliced)

    # Identity, not equality: the repair must MOVE the caller's own objects, not
    # rebuild them. Compaction memoizes token estimates per message object (see
    # `Message`), so a rendered copy would silently invalidate that cache.
    assert [message.id for message in repaired] == [
        spliced[0].id,
        assistant.id,
        result_a.id,
        result_b.id,
        notice.id,
    ]
    assert repaired[-1] is notice


def test_pair_spliced_tool_results_keeps_multiple_interlopers_in_order():
    assistant, result_a, result_b = _batch("toolu_A", "toolu_B")
    first = Message.user("switched to opus")
    second = Message.user("provider error: HTTP 529")
    spliced = [assistant, first, result_a, second, result_b]

    repaired = _pair_spliced_tool_results(spliced)

    assert [message.id for message in repaired] == [
        assistant.id,
        result_a.id,
        result_b.id,
        first.id,
        second.id,
    ], "interlopers must move after the results and keep their relative order"


def test_pair_spliced_tool_results_returns_a_legal_list_unchanged():
    assistant, result_a, result_b = _batch("toolu_A", "toolu_B")
    legal = [Message.user("go"), assistant, result_a, result_b, Message.assistant("done")]

    repaired = _pair_spliced_tool_results(legal)

    assert repaired == legal
    # A trailing message AFTER a closed batch is legal (verified live: the
    # [tool_result, tool_result, text] shape returns 200), so nothing moves.
    assert [message.id for message in repaired] == [message.id for message in legal]


def test_pair_spliced_tool_results_leaves_an_unanswered_tail_alone():
    """An open batch is ``_wire_legal_snapshot``'s job, not this one.

    Synthesizing placeholders here would be actively wrong: the architect
    tested it against the live API and it draws a different 400
    (``unexpected tool_use_id found in tool_result blocks``) as soon as the
    genuine results arrive behind the placeholders.
    """
    assistant, result_a, _ = _batch("toolu_A", "toolu_B")
    # toolu_B never answered; a notice sits behind the partial batch.
    partial = [assistant, result_a, Message.user("switched to opus")]

    repaired = _pair_spliced_tool_results(partial)

    assert [message.id for message in repaired] == [message.id for message in partial]


def test_pair_spliced_tool_results_is_a_noop_without_tool_calls():
    plain = [Message.user("go"), Message.assistant("hi")]
    assert _pair_spliced_tool_results(plain) is plain, "the early exit should return the input"


def test_pair_spliced_tool_results_warns_when_it_fires(caplog):
    """Silence would hide a Layer 1 regression: after the guard this is dead
    code, so the one time it runs it has to say so."""
    assistant, result_a, result_b = _batch("toolu_A", "toolu_B")
    spliced = [assistant, Message.user("switched"), result_a, result_b]

    with caplog.at_level("WARNING"):
        _pair_spliced_tool_results(spliced)

    assert any(
        "spliced into an open tool batch" in record.message for record in caplog.records
    ), "the repair fired without logging"


# --- Layer 2: end-to-end self-heal and both wires -----------------------------


@pytest.mark.asyncio
async def test_a_session_bricked_in_memory_heals_itself_on_the_next_turn(tmp_path):
    """Layer 1 cannot help a context that is ALREADY spliced.

    This is the live incident's actual recovery story: a session corrupted
    before the fix shipped, or by a future writer that bypasses the guard,
    must still be able to send a legal request.
    """
    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    assistant, result_a, result_b = _batch("toolu_A", "toolu_B")
    # Hand it the exact bricked shape, notice included.
    session._context.messages.extend([assistant, _switch_notice(), result_a, result_b])
    try:
        await session.prompt("Continue")
    finally:
        await session.dispose()

    wire = _wire_messages(stream.requests[0])
    shape = [
        (message["role"], [block.get("type") for block in message["content"]]) for message in wire
    ]
    assert _dangling_tool_use(wire) == [], f"the bricked session did not heal: {shape}"


@pytest.mark.asyncio
async def test_both_wires_accept_the_repaired_history(tmp_path):
    """Anthropic coalesces, Responses emits a bare item — same rejection.

    Patching ``AnthropicClient`` alone would have left the OpenAI Responses
    path broken in exactly the same way, which is why the repair lives at the
    render and not in a client.
    """
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    assistant, result_a, result_b = _batch("toolu_A", "toolu_B")
    session._context.messages.extend([assistant, _switch_notice(), result_a, result_b])
    try:
        rendered = session._render_history(list(session._context.messages))
    finally:
        await session.dispose()

    anthropic_body = AnthropicClient()._build_body(
        ChatRequest(model=WIRE_MODEL, system_blocks=[], messages=rendered, tools=[])
    )["messages"]
    assert _dangling_tool_use(anthropic_body) == [], "Anthropic wire still orphans the batch"

    responses_items = _messages_to_openai_responses(rendered)
    assert _responses_dangling(responses_items) == [], (
        "OpenAI Responses wire still orphans the batch: a bare role:user item "
        "sits between function_call and function_call_output"
    )


@pytest.mark.asyncio
async def test_resume_replays_a_clean_history_in_write_order(tmp_path):
    """The transcript's row order is WRITE order, not live-append order.

    ``journal_model_switch`` awaits its transcript write BEFORE the live
    append, while the assistant message is written later at the batch
    boundary — so a clean-looking transcript proves nothing about the live
    list, and the reverse has to be checked separately. This is that check:
    what ``--resume`` rebuilds must be legal too.
    """
    holder: dict[str, Session] = {}
    stream = _tool_then_text()
    session = make_session(tmp_path, stream, tools=[_switching_tool(holder)])
    holder["session"] = session
    try:
        await session.prompt("go")
    finally:
        await session.dispose()

    replayed = list(session._transcript.build_llm_history())
    body = AnthropicClient()._build_body(
        ChatRequest(
            model=WIRE_MODEL,
            system_blocks=[],
            messages=session._render_history(replayed),
            tools=[],
        )
    )["messages"]
    assert _dangling_tool_use(body) == [], "a resumed session replays an illegal body"
