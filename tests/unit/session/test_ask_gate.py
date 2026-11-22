"""Queued ask admission and the standalone no-write clearance transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from local_operator.asks import policy
from local_operator.harness.types import (
    Message,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.session.errors import AsideUnanswered

from .test_aside import _BARE_TOOL_CALL, RecordingStream, _tool, make_session


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["clear", "resolve", "raise", "error"])
async def test_explicit_ask_queue_owns_admission(tmp_path, monkeypatch, verdict: str) -> None:
    from local_operator.asks import store
    from local_operator.session.runtime.serving import ServingSessionHandle
    from local_operator.tools.builtin import execute_ask

    fork_calls = []

    async def stream(request, signal):
        fork_calls.append(request)
        assert request.purpose == "clearance"
        assert any("raise the card and I'll answer it" in m.text for m in request.messages)
        if verdict == "error":
            raise RuntimeError("clearance unavailable")
        yield StreamTextDelta(delta=f"VERDICT: {verdict}\nREASON: Choose the blue label.")

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    monkeypatch.setattr(policy, "ASK_GATE", True)
    session = make_session(tmp_path, stream, session_id="da010e010001")
    user_turn = Message.user("For the label color, raise the card and I'll answer it.")
    await session._transcript.append_message(user_turn)
    session._context.messages.append(user_turn)
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(tmp_path), install_ask_surface=True
    )
    desktop = []
    phone = []
    desktop_sub = session._frontend_state_store.subscribe(desktop.append)
    unsub_phone = handle.subscribe(
        lambda: phone.append(list(handle.session_projection_seed.asks or []))
    )
    args = {
        "questions": [
            {
                "id": "color",
                "question": "What color should the label be?",
                "recommended": 0,
                "options": [{"label": "Blue"}, {"label": "Green"}],
            }
        ]
    }
    try:
        context = session._build_tool_context()
        result = await execute_ask("first", args, None, None, context)
        queue = session.ask_queue()
        assert queue is not None
        records = queue.records()
        first_events = store.read_events(queue.session_dir)
        first_desktop, first_phone = list(desktop), list(phone)
        retry = await execute_ask("retry", args, None, None, context)
        receipt = result.model_dump_json().lower()
        assert len(records) == 1, (
            f"first_events={len(first_events)} first_hidden={(result.details or {}).get('ask_gate')} "
            f"retry_status={(retry.details or {}).get('status')} fork_calls={len(fork_calls)}"
        )
        record = records[0]
        assert record["status"] == store.STATUS_OPEN
        assert [o["label"] for o in record["questions"][0]["options"]] == ["Blue", "Green"]
        assert not result.is_error
        assert result.details is not None
        assert result.details["ask_id"] == record["ask_id"]
        for text in (receipt, retry.model_dump_json().lower()):
            assert all(
                fragment not in text
                for fragment in ("ask_gate", "proceed with", "not asked", "blue", "green")
            )
        assert any(record["ask_id"] in update.model_dump_json() for update in first_desktop)
        assert any(row.ask_id == record["ask_id"] for rows in first_phone for row in rows)
        assert [event["kind"] for event in first_events] == [store.EVENT_QUEUED]
        assert fork_calls == []

        assert retry.is_error
        assert "already carries that question text" in retry.text
        assert [r["ask_id"] for r in queue.records()] == [record["ask_id"]]
        assert [event["kind"] for event in store.read_events(queue.session_dir)] == [
            store.EVENT_QUEUED
        ]
    finally:
        desktop_sub.unsubscribe()
        unsub_phone()


@pytest.mark.asyncio
async def test_complete_clearance_leaves_no_trace(tmp_path) -> None:
    """The §2.1 contract: reads everything, writes nothing, on every surface."""
    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session._context.messages.extend([Message.user("port it"), Message.assistant("done.")])
    events: list[object] = []
    session.subscribe(events.append)
    before = list(session._context.messages)

    answer = await session.complete_clearance([Message.user("<ask-clearance>x</ask-clearance>")])

    assert answer == "answer."
    assert all(a is b for a, b in zip(session._context.messages, before))
    assert session._transcript.entries() == []
    assert events == []
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_request_shape(tmp_path) -> None:
    """purpose / tools / tool_choice / warm-prefix, exactly as the design tables."""
    tools = [_tool("bash"), _tool("read")]
    stream = RecordingStream()
    session = make_session(tmp_path, stream, tools=tools)
    session._context.messages.append(Message.user("port it"))

    await session.complete_clearance([Message.user("gate message")])

    request = stream.requests[-1]
    assert request.purpose == "clearance"
    assert request.replayable is True
    # ``isolated`` stays absent/False — load-bearing: isolation would strip the
    # session's cache key and put the gate on a cold namespace (design §2.1).
    assert request.isolated is False
    # ``None`` HERE is correct and half the lineage story: the built request is
    # unprefixed by construction, and the session's stream fn stamps
    # ``prompt_cache_key = _cache_lineage_id`` one layer down — the cell below
    # drives that layer on the real ``SessionStreamFn`` so the pair is pinned.
    assert request.prompt_cache_key is None
    assert request.tool_choice == "none"
    # Tools mirror the array the turn publishes — the front of the cached
    # prefix (never []). That is the PUBLISHABLE inventory, not ``context.tools``:
    # the latter keeps the deferred schemas (``no_reply`` among them) so
    # resolution can call them first try; the wire array withholds them.
    assert request.tools == session._publishable(session._tools)
    assert [m.text for m in request.messages] == ["port it", "gate message"]
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_rides_the_sessions_cache_lineage(tmp_path, monkeypatch) -> None:
    """The lineage reaches the wire, driven through the REAL stream fn (§4).

    The shape test above stops at the built request, whose ``prompt_cache_key``
    is ``None`` — so on its own it cannot see a regression that keeps the
    request shape and loses the lineage. The stamp lives in
    ``SessionStreamFn.__call__`` (``prompt_cache_key = _cache_lineage_id``,
    SKIPPED for ``isolated`` requests), so this drives a real stream fn with
    the provider boundary patched out and reads the stamped request off that
    boundary. A fork's ``cache_lineage_id`` is its PARENT's id; this session
    stands in with a synthetic one.
    """
    from local_operator.model.configure import create_stream_fn
    from local_operator.providers import failover as failover_module

    class _Auth:  # only the resolved ids are inspected; nothing is streamed
        pass

    captured: list[Any] = []

    async def _fake_stream_with_failover(request: Any, *args: Any, **kwargs: Any):
        captured.append(request)
        yield StreamTextDelta(delta="VERDICT: clear\nREASON: lineage")

    # Patched at the SOURCE module: ``__call__`` imports the name at call time
    # (``from local_operator.providers.failover import stream_with_failover``),
    # so the failover module's attribute is the one it reads.
    monkeypatch.setattr(failover_module, "stream_with_failover", _fake_stream_with_failover)

    stream = create_stream_fn(
        _Auth(),  # type: ignore[arg-type]
        settings={},
        session_id="s-lineage",
        cache_lineage_id="s-parent-lineage",
    )
    session = make_session(tmp_path, stream)
    answer = await session.complete_clearance([Message.user("gate message")])

    assert answer == "VERDICT: clear\nREASON: lineage"
    assert captured, "the clearance request must reach the provider boundary"
    request = captured[-1]
    assert request.prompt_cache_key == "s-parent-lineage"
    # And the wire-visible shape survived the real pipeline: no isolation was
    # re-introduced by a layer below the session.
    assert request.isolated is False and request.purpose == "clearance"
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_prefix_is_byte_identical_to_the_context(tmp_path) -> None:
    """The cache READ this design exists to preserve (§2.1, §4 item 6's unit
    half): the request's message prefix IS the session's rendered context —
    byte-for-byte under the canonical JSON dump — with exactly one message
    appended after it.

    ``expected`` is built through the two functions the TURN's own seam uses
    (``harness/loop.py``: ``_render_history`` + the replay bound), so a
    transformation that leaked into only one of the two paths (an extra
    summary, a changed bound) fails here even though both still \"work\". The
    LIVE byte-identity read is the measure-script clearance arm (§4 item 6),
    which QA drives against a real warm session.
    """
    from local_operator.harness.replay_bound import bound_replay_payloads

    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session._context.messages.extend([Message.user("port it"), Message.assistant("done.")])

    before = list(session._context.messages)
    expected = bound_replay_payloads(session._render_history(before))

    await session.complete_clearance([Message.user("gate message")])

    request = stream.requests[-1]
    assert len(request.messages) == len(expected) + 1, "exactly one message is appended"

    def _bytes(messages: list[Any]) -> bytes:
        return json.dumps([m.model_dump(mode="json") for m in messages]).encode()

    assert _bytes(request.messages[:-1]) == _bytes(expected)
    assert request.messages[-1].text == "gate message"
    # Nothing was mutated to build the fork: the context is untouched by
    # identity, the same contract the no-trace cell pins.
    assert all(a is b for a, b in zip(session._context.messages, before))
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_pairs_a_dangling_tool_call(tmp_path) -> None:
    """Mid-batch the live list is not wire-legal; the fork rides ``_wire_legal_snapshot``."""
    from local_operator.harness.types import ToolCall

    stream = RecordingStream()
    session = make_session(tmp_path, stream)
    session._context.messages.extend(
        [
            Message.user("run it"),
            Message(
                role="assistant",
                content=[TextContent(text="running")],
                tool_calls=[ToolCall(id="call_1", name="bash", arguments={})],
            ),
        ]
    )

    await session.complete_clearance([Message.user("gate message")])

    messages = stream.requests[-1].messages
    answered = {m.tool_call_id for m in messages if m.role == "tool"}
    for message in messages:
        for call in message.tool_calls:
            assert call.id in answered, "every tool call must be answered on the wire"
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_retries_a_bare_tool_call_without_tools(tmp_path) -> None:
    """The aside's one bounded retry, inherited: rejected call handed back, paired."""
    tools = [_tool("bash"), _tool("read")]
    stream = RecordingStream(
        scripted=[
            _BARE_TOOL_CALL,
            [
                StreamTextDelta(delta="VERDICT: clear\nREASON: fine."),
                StreamEndEvent(stop_reason="stop"),
            ],
        ]
    )
    session = make_session(tmp_path, stream, tools=tools)

    answer = await session.complete_clearance([Message.user("gate message")])

    assert answer == "VERDICT: clear\nREASON: fine."
    assert len(stream.requests) == 2
    first, retry = stream.requests
    # Same shape as the cell above: the wire array is the publishable
    # inventory (deferred schemas withheld), not ``context.tools``.
    assert first.tools == session._publishable(session._tools)
    assert retry.tools == []
    call, refusal = retry.messages[-2:]
    assert call.role == "assistant" and [(c.id, c.name) for c in call.tool_calls] == [
        ("call_1", "read")
    ]
    assert refusal.role == "tool" and refusal.tool_call_id == "call_1"
    assert "clearance check" in refusal.text
    assert "VERDICT: clear|resolve|raise" in refusal.text
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_retry_is_bounded_to_one(tmp_path) -> None:
    stream = RecordingStream(scripted=[_BARE_TOOL_CALL])
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    with pytest.raises(AsideUnanswered):
        await session.complete_clearance([Message.user("gate message")])

    assert len(stream.requests) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_complete_clearance_retry_that_answers_nothing_raises(tmp_path) -> None:
    stream = RecordingStream(scripted=[_BARE_TOOL_CALL, [StreamEndEvent(stop_reason="refusal")]])
    session = make_session(tmp_path, stream, tools=[_tool("read")])

    with pytest.raises(AsideUnanswered):
        await session.complete_clearance([Message.user("gate message")])

    assert len(stream.requests) == 2
    await session.dispose()
