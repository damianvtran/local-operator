"""Structural guards for cheaper session boundaries, not machine-speed bets."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, cast

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    CustomMessage,
    Message,
    MessageEndEvent,
    MessageUpdateEvent,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session
from local_operator.session.subscriptions import PresentationSubscription
from local_operator.session.transcript import Transcript
from tests.unit.session.test_session import MODEL, ScriptedStream, make_session


def state_provider(state):
    def blocks(model_label=""):
        return ["standing instructions", "tools", f"date {state['date']}", state["goal"]]

    setattr(blocks, "append_only_state", True)
    return blocks


@pytest.mark.asyncio
async def test_live_state_changes_preserve_prefix_and_replay_at_same_request(tmp_path):
    state = {"date": "2026-09-04", "goal": "read only"}
    stream = ScriptedStream(
        [
            [StreamTextDelta(delta="first"), StreamEndEvent(stop_reason="stop")],
            [StreamTextDelta(delta="second"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = make_session(tmp_path, stream, system_blocks_provider=state_provider(state))
    await session.prompt("inspect")
    first_blocks = list(stream.requests[0].system_blocks)
    first_history = [(m.id, m.text) for m in stream.requests[0].messages]
    state.update(date="2026-09-05", goal="only inspect files under src")
    await session.prompt("continue")
    assert stream.requests[1].system_blocks == first_blocks
    assert [
        (m.id, m.text) for m in stream.requests[1].messages[: len(first_history)]
    ] == first_history
    assert "only inspect files under src" in stream.requests[1].messages[-1].text
    assert "2026-09-05" in stream.requests[1].messages[-1].text
    updates = [
        m
        for m in session._context.messages
        if isinstance(m, CustomMessage) and m.custom_type == "session_state"
    ]
    assert len(updates) == 1
    assert session._transcript.has_entry(updates[0].id)
    # A read-only helper sees the same cached head and never changes the journal.
    entries_before = session._transcript.entries()
    blocks, history = await session._read_only_prompt([Message.user("aside")])
    assert blocks == first_blocks
    assert history[-1].text == "aside"
    assert session._transcript.entries() == entries_before
    await session.dispose()

    resumed_stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    resumed = Session(
        model=MODEL,
        stream_fn=resumed_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=state_provider(state),
    )
    await resumed.prompt("resume")
    assert resumed_stream.requests[0].system_blocks == first_blocks
    assert (
        len(
            [
                m
                for m in resumed._context.messages
                if isinstance(m, CustomMessage) and m.custom_type == "session_state"
            ]
        )
        == 1
    )
    await resumed.dispose()


@pytest.mark.asyncio
async def test_failed_state_journal_cannot_issue_request_with_old_goal(tmp_path, monkeypatch):
    state = {"date": "today", "goal": "original"}
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, system_blocks_provider=state_provider(state))
    await session.prompt("first")
    state["goal"] = "stop all writes"
    original = session._transcript.append_message

    async def append(message, **kwargs):
        if isinstance(message, CustomMessage) and message.custom_type == "session_state":
            raise OSError("disk full")
        return await original(message, **kwargs)

    monkeypatch.setattr(session._transcript, "append_message", append)
    events = []
    session.subscribe(events.append)
    await session.prompt("second")
    assert len(stream.requests) == 1
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and "disk full" in (ends[-1].error or "")
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_changed_standing_instructions_replace_the_durable_prefix(tmp_path, resume):
    """Standing authority outranks prefix reuse, in a live session and on resume."""
    state = {"instructions": "OLD production allowed", "goal": "inspect old target"}

    def provider():
        return [state["instructions"], "tools", "environment", state["goal"]]

    setattr(provider, "append_only_state", True)
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]] * 4)
    session = make_session(tmp_path, stream, system_blocks_provider=provider)
    await session.prompt("start")
    state["goal"] = "intermediate target"
    await session.prompt("continue")
    old_prefix = session._transcript.latest_custom_entry("system_prefix")
    assert old_prefix is not None
    if resume:
        await session.dispose()
        stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]] * 2)
        session = make_session(tmp_path, stream, system_blocks_provider=provider)

    state.update(instructions="NEW never touch production", goal="inspect new target")
    before_helper = session._transcript.entries()
    helper_blocks, helper_history = await session._read_only_prompt([Message.user("aside")])
    assert helper_blocks[0] == state["instructions"]
    assert "inspect new target" in helper_history[-2].text
    assert session._transcript.entries() == before_helper
    await session.prompt("work under the new rule")
    request = stream.requests[-1]
    assert request.system_blocks[0] == state["instructions"]
    assert all("OLD production allowed" not in block for block in request.system_blocks)
    assert "inspect new target" in request.messages[-1].text
    new_prefix = session._transcript.latest_custom_entry("system_prefix")
    assert new_prefix is not None and new_prefix.id != old_prefix.id
    assert new_prefix.payload["details"]["blocks"][0] == state["instructions"]
    await session.prompt("same rule")
    assert session._transcript.latest_custom_entry("system_prefix") is new_prefix
    await session.dispose()

    reopened = make_session(tmp_path, ScriptedStream([]), system_blocks_provider=provider)
    assert await reopened._prepare_system_blocks() == request.system_blocks
    restored_prefix = reopened._transcript.latest_custom_entry("system_prefix")
    assert restored_prefix is not None and restored_prefix.id == new_prefix.id
    await reopened.dispose()


@pytest.mark.asyncio
async def test_failed_standing_instruction_epoch_stops_before_provider(tmp_path, monkeypatch):
    state = {"head": "OLD permission"}

    def provider():
        return [state["head"], "tools", "environment", "goal"]

    setattr(provider, "append_only_state", True)
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, system_blocks_provider=provider)
    await session.prompt("start")
    old_prefix = session._transcript.latest_custom_entry("system_prefix")
    original = session._transcript.append_custom

    async def append(custom_type, data, **kwargs):
        if custom_type == "system_prefix":
            raise OSError("disk full")
        return await original(custom_type, data, **kwargs)

    monkeypatch.setattr(session._transcript, "append_custom", append)
    state["head"] = "NEW revoke permission"
    with pytest.raises(OSError, match="disk full"):
        await session.prompt("continue")
    assert len(stream.requests) == 1
    assert session._transcript.latest_custom_entry("system_prefix") is old_prefix
    assert session._frozen_system_blocks is not None
    assert session._frozen_system_blocks[0] == "OLD permission"
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("goal", ["new goal", ""])
async def test_helpers_reanchor_compacted_state_without_journaling(tmp_path, resume, goal):
    state = {"date": "today", "goal": "old goal"}
    provider = state_provider(state)
    session = make_session(tmp_path, ScriptedStream([]), system_blocks_provider=provider)
    await session._prepare_system_blocks()
    state["goal"] = goal
    await session._prepare_system_blocks()
    kept = Message.user("retained task")
    await session._transcript.append_message(kept)
    await session._transcript.append_compaction("summary without host state", kept.id, 100)
    session._context.messages = session._transcript.build_llm_history()
    if resume:
        await session.dispose()
        session = make_session(tmp_path, ScriptedStream([]), system_blocks_provider=provider)
    before = session._transcript.entries()
    blocks, messages = await session._read_only_prompt([Message.user("aside")])
    assert blocks[3] == "old goal", "the historical prefix remains cacheable"
    assert "## Knowledge and session state\n" + (goal or "(empty)") in messages[-2].text
    assert session._transcript.entries() == before
    # A transient helper cannot consume the real turn's publication obligation.
    await session._prepare_system_blocks()
    assert len(session._transcript.entries()) == len(before) + 1
    current_state = session._context.messages[-1]
    assert isinstance(current_state, CustomMessage)
    assert current_state.details["blocks"]["3"] == goal
    await session.dispose()


@pytest.mark.asyncio
async def test_asides_are_durable_when_drain_returns(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]))
    message = CustomMessage(custom_type="hub_message", details={"text": "constraint"})
    session.queue_aside(lambda: message)
    drained = await session._drain_asides()
    assert drained == [message]
    assert session._transcript.has_entry(message.id)
    reopened = Transcript(session._transcript.directory)
    assert any(row.id == message.id for row in reopened.entries())
    await session.dispose()


@pytest.mark.asyncio
async def test_slow_presentation_observer_does_not_block_and_keeps_text_order():
    entered = asyncio.Event()
    release = asyncio.Event()
    events = []

    async def slow(event):
        if isinstance(event, AgentStartEvent):
            entered.set()
            await release.wait()
        events.append(event)

    observer = PresentationSubscription(slow)
    observer.enqueue(AgentStartEvent())
    await asyncio.wait_for(entered.wait(), 10)
    message = Message.assistant("complete")
    for delta in ("com", "ple", "te"):
        observer.enqueue(MessageUpdateEvent(message=message, delta=delta))
    observer.enqueue(MessageEndEvent(message=message))
    observer.enqueue(AgentEndEvent())
    # The producer queued the whole lifecycle while the observer is suspended.
    assert not events
    release.set()
    await asyncio.wait_for(observer.flush(), 10)
    assert [e.type for e in events] == ["agent_start", "message_update", "message_end", "agent_end"]
    assert events[1].delta == "complete"
    await observer.aclose()


@pytest.mark.asyncio
async def test_overloaded_presentation_explicitly_disconnects_in_bounded_space():
    entered = asyncio.Event()
    release = asyncio.Event()
    overflows = []

    async def slow(event):
        entered.set()
        await release.wait()

    observer = PresentationSubscription(
        slow, max_pending=2, on_overflow=lambda: overflows.append(True)
    )
    observer.enqueue(AgentStartEvent())
    await asyncio.wait_for(entered.wait(), 10)
    for _ in range(3):
        observer.enqueue(MessageEndEvent(message=Message.assistant("x")))
    assert observer.closed
    assert overflows == [True]
    assert not observer._queue
    await observer.aclose()


@pytest.mark.asyncio
async def test_child_owns_stream_and_inherits_guidance_without_parent_history(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from local_operator.harness.subagent import _build_child_session

    class OwnedStream(ScriptedStream):
        def __init__(self):
            super().__init__([])
            self.children = []
            self.closed = False

        def fork(self, session_id):
            child = OwnedStream()
            self.children.append((session_id, child))
            return child

        async def close(self):
            self.closed = True

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    provider = state_provider({"date": "today", "goal": "task"})
    setattr(provider, "repo_guidance", "Repository rule: edits need a regression check.")
    setattr(
        provider, "knowledge_hooks", SimpleNamespace(frozen_block="Read skill://repository-testing")
    )
    stream = OwnedStream()
    parent = make_session(tmp_path, stream, system_blocks_provider=provider)
    parent._context.messages.append(Message.user("parent-only conversation fact"))
    child = await _build_child_session(
        label="review",
        prompt="inspect the change",
        parent_session=parent,
        model_spec=None,
        job_id="review-job",
    )
    try:
        assert stream.children == [(child.session_id, child._stream_fn)]
        assert child._stream_fn is not stream
        blocks = child._system_blocks_provider()
        if inspect.isawaitable(blocks):
            blocks = await blocks
        assert "Repository rule:" in blocks[0]
        assert "skill://repository-testing" in blocks[3]
        assert not any(
            "parent-only" in m.text for m in child._context.messages if isinstance(m, Message)
        )
    finally:
        await child.dispose()
        await parent.dispose()
    assert stream.children[0][1].closed
    assert not stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason, expected", [("stop", 0), ("length", 1)])
async def test_compaction_continues_only_a_length_interrupted_answer(
    tmp_path, stop_reason, expected
):
    from types import SimpleNamespace

    from local_operator.session.session import CompactionOutcome

    session = make_session(tmp_path, ScriptedStream([]))
    terminal = Message.assistant("answer", stop_reason=stop_reason)
    session._held_end = AgentEndEvent(messages=[terminal])
    plan = SimpleNamespace(
        settings=SimpleNamespace(auto_continue=True),
        compaction_api=SimpleNamespace(
            resolve_threshold_tokens=lambda *_: 100000, RECOVERY_BAND=0.8
        ),
    )
    outcome = CompactionOutcome(
        ran=True, reason="compacted", tokens_before=90000, tokens_after=1000
    )
    session._after_compaction_pass(cast(Any, plan), outcome)
    assert len(session._continuation_queue) == expected
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_writer", [False, True])
async def test_hub_answer_waits_for_its_durable_receipt(tmp_path, monkeypatch, cancel_writer):
    """An answered ask is observable only after its child receipt commits."""
    from tests.unit.harness.test_comms import wire

    comms, _jobs, child, parent = wire()
    transcript = Transcript(tmp_path / "hub-child")
    setattr(child, "_transcript", transcript)
    background = []

    def spawn(coro):
        task = asyncio.create_task(coro)
        background.append(task)
        return task

    setattr(parent, "_spawn_background", spawn)
    queued = asyncio.Event()
    original_queue = child.queue_aside

    def queue(thunk):
        original_queue(thunk)
        queued.set()

    monkeypatch.setattr(child, "queue_aside", queue)
    entered = asyncio.Event()
    release = asyncio.Event()
    original_append = transcript.append_custom

    async def append(kind, details):
        if details.get("direction") == "to_parent":
            entered.set()
            await release.wait()
        return await original_append(kind, details)

    monkeypatch.setattr(transcript, "append_custom", append)
    question = asyncio.create_task(comms.ask("job-1", "status?", 10000))
    try:
        await asyncio.wait_for(queued.wait(), 10)
        child.materialize()
        comms.reply_to_parent("job-1", "ready")
        await asyncio.wait_for(entered.wait(), 10)
        assert not question.done(), "reply acknowledged before durable receipt"
        comms.detach("job-1")
        if cancel_writer:
            background[-1].cancel()
            await asyncio.sleep(0)
        assert not question.done(), "detach discarded a reply whose receipt is committing"
        release.set()
        reply = await asyncio.wait_for(question, 10)
        assert reply.text == "ready"
        assert any(
            entry.payload.get("details", {}).get("body") == "ready"
            for entry in Transcript(transcript.directory).entries()
        )
    finally:
        release.set()
        await asyncio.gather(*background, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "cancel_before_start", "write_failed"])
async def test_hub_reply_publication_failure_releases_its_waiter(tmp_path, monkeypatch, failure):
    from tests.unit.harness.test_comms import wire

    comms, _jobs, child, parent = wire()
    transcript = Transcript(tmp_path / "hub-child")
    setattr(child, "_transcript", transcript)
    tasks = []

    def spawn(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    setattr(parent, "_spawn_background", spawn)
    queued = asyncio.Event()
    original_queue = child.queue_aside

    def queue(thunk):
        original_queue(thunk)
        queued.set()

    monkeypatch.setattr(child, "queue_aside", queue)
    question = asyncio.create_task(comms.ask("job-1", "status?", 10000))
    await asyncio.wait_for(queued.wait(), 10)
    child.materialize()

    def refuse(coro):
        coro.close()
        return None

    def cancel(coro):
        task = spawn(coro)
        task.cancel()
        task.add_done_callback(lambda _: coro.close())
        return task

    original_append = transcript.append_custom

    async def append(kind, details):
        if details.get("direction") == "to_parent":
            raise OSError("reply disk full")
        return await original_append(kind, details)

    if failure == "write_failed":
        monkeypatch.setattr(transcript, "append_custom", append)
    else:
        setattr(parent, "_spawn_background", refuse if failure == "refused" else cancel)
    try:
        comms.reply_to_parent("job-1", "ready")
        comms.detach("job-1")
        reply = await asyncio.wait_for(question, 10)
        assert reply.error and (
            "saved" in reply.error or "cancelled" in reply.error or "disk full" in reply.error
        )
        assert not reply.timed_out
        record = comms._record("job-1")
        assert record is not None and record.reply_in_flight is None
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_durable_reply_cannot_complete_a_replacement_ask(tmp_path, monkeypatch):
    from tests.unit.harness.test_comms import wire

    comms, _jobs, child, parent = wire()
    transcript = Transcript(tmp_path / "hub-child")
    setattr(child, "_transcript", transcript)
    tasks = []

    def spawn(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    setattr(parent, "_spawn_background", spawn)
    queued = asyncio.Event()
    original_queue = child.queue_aside

    def queue(thunk):
        original_queue(thunk)
        queued.set()

    monkeypatch.setattr(child, "queue_aside", queue)
    entered, release = asyncio.Event(), asyncio.Event()
    original_append = transcript.append_custom

    async def append(kind, details):
        if details.get("body") == "old answer":
            entered.set()
            await release.wait()
        return await original_append(kind, details)

    monkeypatch.setattr(transcript, "append_custom", append)
    first = asyncio.create_task(comms.ask("job-1", "first?", 10000))
    await asyncio.wait_for(queued.wait(), 10)
    child.materialize()
    comms.reply_to_parent("job-1", "old answer")
    await asyncio.wait_for(entered.wait(), 10)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    queued.clear()
    second = asyncio.create_task(comms.ask("job-1", "second?", 10000))
    try:
        await asyncio.wait_for(queued.wait(), 10)
        child.materialize()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert not second.done()
        comms.reply_to_parent("job-1", "new answer")
        reply = await asyncio.wait_for(second, 10)
        assert reply.text == "new answer"
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
