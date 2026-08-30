"""Streaming semantics of ``AgentEventBridge``.

This class had no test coverage at all, and that gap let a harness-side
optimization silently break the websocket/SSE streaming contract: when the
harness stopped rebuilding ``message.content`` on every delta (it was quadratic
in response length), the bridge — which re-read that field per delta — began
broadcasting an empty string for the whole turn and only filled the text in at
message_end. A green suite said nothing, because nothing exercised this path.

These tests pin the CONTRACT stated in the class docstring: one long-lived
record per assistant message, re-broadcast on every delta carrying the text
accumulated SO FAR. They deliberately assert on the accumulation rather than on
which object the text is read from, so they hold whichever side owns the
buffer.
"""

from __future__ import annotations

import queue
from typing import Any

from local_operator.harness.types import (
    Content,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
)
from local_operator.server.utils.operator import AgentEventBridge


def _bridge() -> tuple[AgentEventBridge, "queue.Queue[Any]"]:
    q: queue.Queue[Any] = queue.Queue()
    return AgentEventBridge(status_queue=q, job_id="job-1"), q


def _streamed(message_id: str = "m1") -> Message:
    """An assistant message shaped the way the harness emits it while streaming.

    Content stays EMPTY for the life of the stream — the harness assembles it
    once the turn ends — so a bridge that re-reads ``message.content`` here sees
    nothing. That is exactly the regression these tests guard.
    """
    return Message(id=message_id, role="assistant")


def test_the_record_carries_the_text_accumulated_so_far_on_every_delta() -> None:
    """The streaming contract: text grows delta by delta, not all at the end."""
    bridge, _ = _bridge()
    message = _streamed()

    bridge.handle(MessageStartEvent(message=message))
    seen: list[str] = []
    for delta in ("Hello", " there", " world"):
        bridge.handle(MessageUpdateEvent(message=message, delta=delta))
        record = bridge._streams[message.id]
        seen.append(record.message or "")

    assert seen == ["Hello", "Hello there", "Hello there world"], (
        "the record must carry the accumulated text on every delta; "
        "an empty or lagging value collapses streaming into an end-of-turn dump"
    )


def test_the_final_text_wins_at_message_end() -> None:
    """message_end carries the harness's authoritative text, and it is adopted."""
    bridge, _ = _bridge()
    message = _streamed()

    bridge.handle(MessageStartEvent(message=message))
    bridge.handle(MessageUpdateEvent(message=message, delta="partial"))

    # The harness assembles the real content only now.
    message.content = [TextContent(text="partial and then some")]
    bridge.handle(MessageEndEvent(message=message))

    record = bridge._streams[message.id]
    assert record.message == "partial and then some"
    assert record.is_complete is True
    assert bridge.final_response == "partial and then some"


def test_a_tool_call_only_turn_keeps_the_text_it_streamed() -> None:
    """An end event with empty content must not wipe what was streamed.

    A turn that answers with text and then calls a tool ends with the text
    already assembled, but a turn whose message carries no text blocks at all
    ends with empty content. Adopting the end event's text unconditionally would
    throw away everything accumulated in that second case — silently, since the
    deltas were already broadcast. The fallback in the bridge exists for this,
    and without this test a mutation removing it keeps the suite green.
    """
    empty_contents: list[list[Content]] = [[], [TextContent(text="")]]
    for empty in empty_contents:
        bridge, _ = _bridge()
        message = _streamed()

        bridge.handle(MessageStartEvent(message=message))
        bridge.handle(MessageUpdateEvent(message=message, delta="streamed text"))

        message.content = empty
        bridge.handle(MessageEndEvent(message=message))

        assert (
            bridge._streams[message.id].message == "streamed text"
        ), f"content={empty!r} at message_end wiped the accumulated text"


def test_a_message_that_never_streamed_still_reports_its_text() -> None:
    """Not every assistant message arrives as deltas.

    A start/end pair with content and no updates in between must still project
    the text, so the bridge cannot rely on having seen deltas.
    """
    bridge, _ = _bridge()
    message = Message(id="m2", role="assistant", content=[TextContent(text="one shot")])

    bridge.handle(MessageStartEvent(message=message))
    bridge.handle(MessageEndEvent(message=message))

    assert bridge._streams["m2"].message == "one shot"
    assert bridge.final_response == "one shot"


def test_the_sse_delta_frame_carries_a_running_snapshot() -> None:
    """``message.delta`` promises the increment AND a snapshot inclusive of it.

    ``sse.py`` documents that a late or lossy consumer can repaint from any
    single frame. The snapshot used to ride along implicitly in
    ``message.content``; it is published explicitly now that the harness
    assembles the text once per turn instead of once per delta.
    """
    bridge, q = _bridge()
    message = _streamed()

    bridge.handle(MessageStartEvent(message=message))
    for delta in ("Hello", " there", " world"):
        bridge.handle(MessageUpdateEvent(message=message, delta=delta))

    frames = []
    while not q.empty():
        kind, _id, payload = q.get()
        if kind == "agent_event" and payload.get("type") == "message_update":
            frames.append((payload.get("delta"), payload.get("snapshot")))

    assert frames == [
        ("Hello", "Hello"),
        (" there", "Hello there"),
        (" world", "Hello there world"),
    ], "each delta frame must carry the text through and including that delta"


def test_two_messages_accumulate_independently() -> None:
    """One record per message id; a second message must not inherit the first's
    text, which a single shared buffer would cause."""
    bridge, _ = _bridge()
    first, second = _streamed("a"), _streamed("b")

    bridge.handle(MessageStartEvent(message=first))
    bridge.handle(MessageUpdateEvent(message=first, delta="alpha"))
    bridge.handle(MessageStartEvent(message=second))
    bridge.handle(MessageUpdateEvent(message=second, delta="beta"))
    bridge.handle(MessageUpdateEvent(message=first, delta="-more"))

    assert bridge._streams["a"].message == "alpha-more"
    assert bridge._streams["b"].message == "beta"
