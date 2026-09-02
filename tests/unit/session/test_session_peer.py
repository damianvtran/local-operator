"""`Session.receive_peer_message` — the receive half of `lop send`.

These guard the correctness traps the design's §2.4/§7 call out and the review
will check: record-only persists exactly one transcript row without running a
turn, wake drives a turn without a double append, the peer message reaches the
model through the allow-list, and a busy steer routes through the steer queue.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import (
    CustomMessage,
    Message,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.session.transcript import Transcript
from tests.unit.session.test_session import ScriptedStream, make_session, wait_for


def _peer_rows(session) -> list[Any]:
    return [
        e
        for e in session._transcript.entries()
        if e.type == "message"
        and e.payload.get("kind") == "custom"
        and e.payload.get("custom_type") == PEER_MESSAGE_MESSAGE_TYPE
    ]


@pytest.mark.asyncio
async def test_record_only_persists_once_and_stays_idle(tmp_path):
    """Mailbox, no wake, idle: one durable row appears, no turn runs, and the
    message is queued in live context for the next turn."""
    stream = ScriptedStream([[StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    detail = await session.receive_peer_message(
        "mailbox note",
        mode="mailbox",
        wake=False,
        sender={"pid": 42, "conversation_name": "peer"},
    )

    assert "mailbox" in detail
    # No turn ran: the model was never called.
    assert stream.requests == []
    # Exactly ONE transcript row for the delivery (the double-append trap).
    rows = _peer_rows(session)
    assert len(rows) == 1
    # And it is in live context so the NEXT turn will read it.
    assert any(
        getattr(m, "custom_type", None) == PEER_MESSAGE_MESSAGE_TYPE
        for m in session._context.messages
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_wake_drives_a_turn_without_double_append(tmp_path):
    """Mailbox + wake while idle: a turn runs and the row is persisted exactly
    once (the pipeline appends it; the method must not append separately)."""
    stream = ScriptedStream([[StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    detail = await session.receive_peer_message(
        "wake and act", mode="mailbox", wake=True, sender={"pid": 7}
    )
    await wait_for(lambda: bool(stream.requests))  # the spawned turn ran

    assert "woke" in detail
    # The wrapped envelope reached the model as a user turn.
    delivered = stream.requests[0].messages
    assert any("wake and act" in m.text for m in delivered)
    # Persisted exactly once — no double write.
    assert len(_peer_rows(session)) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_peer_message_reaches_the_model_via_allow_list(tmp_path):
    """The persisted peer row must render into build_llm_history as a user
    message, guarding the session.py allow-list edit. Without the allow-list
    entry the human sees the row but the model never does."""
    from local_operator.session.session import _default_convert_to_llm

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    await session.receive_peer_message("visible to model?", mode="mailbox", wake=False)

    # build_llm_history() replays raw messages; the allow-list lives in the
    # transcript→LLM converter, which is what a turn actually feeds the model.
    replayed = Transcript(session._transcript.directory).build_llm_history()
    converted = _default_convert_to_llm(replayed)
    user_texts = [m.text for m in converted if isinstance(m, Message) and m.role == "user"]
    assert any("visible to model?" in t for t in user_texts)
    await session.dispose()


@pytest.mark.asyncio
async def test_record_only_while_busy_parks_to_the_turn_boundary(tmp_path):
    """C1 regression: record-only delivery against a BUSY session (mid tool
    batch) must NOT splice the user-attributed peer message between the open
    assistant tool_calls and their tool_results. A bare
    ``_context.messages.append`` produced the illegal
    ``assistant(tool_use) -> user -> tool_result`` order every provider rejects
    (and tripped ``_pair_spliced_tool_results``, same class as PR #302). The
    fix routes the live append through ``_append_or_park_journal``, which parks
    it to the next turn-safe boundary while writing the transcript now.
    """
    import asyncio

    from local_operator.harness.types import (
        AgentTool,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def blocking_execute(tool_call_id, args, signal, on_update, context):
        tool_started.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="block", content=[TextContent(text="done")]
        )

    tool = AgentTool(
        name="block",
        parameters={"type": "object", "properties": {}},
        execute=blocking_execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
            # Third script feeds the NEXT turn that must read the parked peer
            # message as model input.
            [StreamTextDelta(delta="next"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[tool])

    prompt_task = asyncio.ensure_future(session.prompt("long task"))
    await wait_for(lambda: tool_started.is_set())  # tool batch is OPEN: busy, splicing window

    detail = await session.receive_peer_message(
        "busy note", mode="mailbox", wake=False, sender={"pid": 42, "conversation_name": "peer"}
    )
    assert "mailbox" in detail

    # While the tool batch is open, the peer message must NOT be spliced into
    # the live list: it is parked for the turn boundary.
    assert any(
        getattr(m, "custom_type", None) == PEER_MESSAGE_MESSAGE_TYPE
        for m in session._pending_context_journal
    )
    assert not any(
        getattr(m, "custom_type", None) == PEER_MESSAGE_MESSAGE_TYPE
        for m in session._context.messages
    )
    # But the transcript row is written NOW (durably), exactly once.
    assert len(_peer_rows(session)) == 1

    release_tool.set()
    await prompt_task

    # After the turn boundary the parked message drained into live context, in a
    # legal position (after the tool batch closed — never between tool_use and
    # tool_result).
    assert any(
        getattr(m, "custom_type", None) == PEER_MESSAGE_MESSAGE_TYPE
        for m in session._context.messages
    )
    # Still exactly ONE durable row (the parked append did not re-write).
    assert len(_peer_rows(session)) == 1

    # The NEXT turn reads the parked peer message as model input: prompt() holds
    # the turn lock, so the parked journal drains at its boundary and the next
    # request's context includes "busy note".
    next_prompt = asyncio.ensure_future(session.prompt("what now"))
    await wait_for(lambda: len(stream.requests) >= 3)
    next_turn_msgs = stream.requests[2].messages
    assert any("busy note" in (m.text or "") for m in next_turn_msgs)
    await next_prompt
    await session.dispose()


@pytest.mark.asyncio
async def test_steer_while_busy_routes_through_the_steer_queue(tmp_path):
    """Steer mode against a busy session injects mid-turn through steer(),
    which persists its own row — the method must not also append."""
    import asyncio

    from local_operator.harness.types import (
        AgentTool,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def blocking_execute(tool_call_id, args, signal, on_update, context):
        tool_started.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="block", content=[TextContent(text="done")]
        )

    tool = AgentTool(
        name="block",
        parameters={"type": "object", "properties": {}},
        execute=blocking_execute,
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, tools=[tool])
    events: list[Any] = []
    session.subscribe(events.append)

    prompt_task = asyncio.ensure_future(session.prompt("long task"))
    await wait_for(lambda: tool_started.is_set())

    detail = await session.receive_peer_message("redirect now", mode="steer", sender={"pid": 3})
    assert "steer" in detail
    # It went onto the steering queue, not a direct transcript append — and
    # the queued item is the peer row ITSELF, not a plain user Message minted
    # from its body. A plain Message is what produced the double paint: the
    # drain announced it as a user MessageStartEvent under the PeerMessageBlock
    # the receipt had already painted, and persisted a bare user row that lost
    # the sender and the provenance envelope.
    queued = session.queued_steering()
    assert len(queued) == 1
    assert isinstance(queued[0], CustomMessage)
    assert queued[0].custom_type == PEER_MESSAGE_MESSAGE_TYPE
    assert queued[0].details["body"] == "redirect now"
    # Nothing persisted yet: the drain owns the write.
    assert _peer_rows(session) == []
    # An explicit `now=True` send is a real steer, not a courtesy wake, so it
    # must count as urgent (it may interrupt a running tool like a typed one).
    assert session._has_urgent_steering()

    release_tool.set()
    await prompt_task

    # Persisted exactly once, as the peer CustomMessage with its provenance —
    # not as a plain user row.
    rows = _peer_rows(session)
    assert len(rows) == 1
    assert rows[0].payload["details"]["body"] == "redirect now"
    assert rows[0].payload["details"]["sender"]["pid"] == 3
    plain_user_rows = [
        e
        for e in session._transcript.entries()
        if e.type == "message"
        and e.payload.get("kind") != "custom"
        and e.payload.get("role") == "user"
        and "redirect now" in str(e.payload.get("content"))
    ]
    assert plain_user_rows == []

    # The model saw the ENVELOPE (sender provenance), not the bare body.
    follow_up = [m.text for m in stream.requests[1].messages]
    assert any("<peer-session-message" in t and "redirect now" in t for t in follow_up)

    # The front end got exactly one receipt for it — the peer delivery — and
    # NO user-role MessageStartEvent (that is the second, spurious UserBlock).
    receipts = [e for e in events if getattr(e, "type", None) == "peer_message_delivered"]
    assert len(receipts) == 1
    assert receipts[0].message_id == queued[0].id
    user_starts = [
        e
        for e in events
        if getattr(e, "type", None) == "message_start"
        and getattr(e.message, "role", None) == "user"
        and "redirect now" in getattr(e.message, "text", "")
    ]
    assert user_starts == []
    # The steer drain still reported that it delivered the message.
    assert any(getattr(e, "type", None) == "steering_delivered" for e in events)
    await session.dispose()


@pytest.mark.asyncio
async def test_every_delivery_path_marks_a_peer_arrival(tmp_path):
    """The wake half of the mailbox fix.

    `wait` parks on this signal, so a delivery path that forgets to mark it is
    a message the waiting session does not see until its budget expires. The
    count is monotonic and the event is never cleared by the producer —
    re-arming is the consumer's job (see PeerArrivalProtocol).
    """
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream)
    peer = session._peer_arrival

    assert peer.count() == 0
    assert not peer.event().is_set()

    # Record-only (idle, no wake): the path a blocking `wait` has to observe.
    await session.receive_peer_message("mailbox note", mode="mailbox", wake=False)
    assert peer.count() == 1
    assert peer.event().is_set(), "a parked wait must be woken"

    # Wake-an-idle-session path.
    await session.receive_peer_message("wake note", mode="mailbox", wake=True)
    assert peer.count() == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_the_mailbox_wake_appends_nothing_extra_to_context(tmp_path):
    """The splice-hazard guard.

    Waking a `wait` must set an event and NOTHING else. The message reaches
    context through the unchanged park -> flush path at the loop's post-batch
    boundary; a context append here would be the C1-class splice bug the
    record-only branch documents at length.
    """
    stream = ScriptedStream([[StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)

    before = len(session._context.messages)
    await session.receive_peer_message("mailbox note", mode="mailbox", wake=False)

    rows = _peer_rows(session)
    assert len(rows) == 1, "exactly one durable row, as before the wake was added"
    # Idle appends straight through: exactly ONE message, not a doubled splice.
    assert len(session._context.messages) == before + 1
    assert session._peer_arrival.count() == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_a_pid_only_sender_is_named_from_the_local_registry(tmp_path, monkeypatch):
    """The receive side resolves the sender against the registry rather than
    trusting its self-report.

    A `lop send` whose ancestry lookup found no session record arrives as
    `{"pid": N}`, which painted `peer message from (pid 1)` — no name, no model,
    nothing to follow in a busy transcript. The registry is same-account and
    local, so it is the authoritative answer to "who is pid N", and the
    enrichment has to land where BOTH the card and the model-visible provenance
    envelope read it.
    """
    from local_operator.mobile import peer_send as peer_send_mod
    from local_operator.session.runtime import registry

    class _Rec:
        pid = 4321
        session_id = "peer-session-id"
        conversation_name = "release cutter"
        model_label = "anthropic/claude-opus-5"
        cwd = "/tmp/release"

    monkeypatch.setattr(peer_send_mod.registry, "scan", lambda root=None: [(_Rec(), "live")])
    assert registry is not None  # the module the enrichment reads

    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    await session.receive_peer_message(
        "the gates are green",
        mode="mailbox",
        wake=False,
        sender={"pid": 4321},
    )

    rows = _peer_rows(session)
    assert len(rows) == 1
    details = rows[0].payload["details"]

    # The indicator (TUI/phone) reads details["sender"].
    sender = details["sender"]
    assert sender["conversation_name"] == "release cutter"
    assert sender["model_label"] == "anthropic/claude-opus-5"
    assert sender["session_id"] == "peer-session-id"

    # The MODEL reads details["text"] — the provenance envelope — so the
    # enrichment must reach it too, not just the card.
    assert "release cutter" in details["text"]
    assert "anthropic/claude-opus-5" in details["text"]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_sender_with_no_record_still_delivers(tmp_path, monkeypatch):
    """Enrichment is a nicety: an unknown pid must not break delivery."""
    from local_operator.mobile import peer_send as peer_send_mod

    monkeypatch.setattr(peer_send_mod.registry, "scan", lambda root=None: [])
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream)
    detail = await session.receive_peer_message(
        "still lands", mode="mailbox", wake=False, sender={"pid": 999999}
    )
    assert "mailbox" in detail
    rows = _peer_rows(session)
    assert rows[0].payload["details"]["sender"] == {"pid": 999999}
    await session.dispose()
