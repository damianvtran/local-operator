"""The wire bound for an ``agent_end`` conversation frame.

``AgentEndEvent.messages`` is the turn's whole conversation, and every encoder
that turns an event into viewer bytes used to ship it unbounded — which is not
a storage-hygiene complaint on the path that matters. ``lop exec --json``
writes one NDJSON line per event, and an external supervisor reads that stream
with a ``bufio.Scanner`` bounded at 4 MiB, returning ``scanner.Err()`` as a RUN
FAILURE: five pasted screenshots measure a 4,667,340-byte line, so the run
dies. Measured on a 104-message / 50-tool-row conversation the frame is 531,082
bytes, 88.7% of it tool rows whose bytes already crossed during the turn as
``tool_execution_end`` events.

These tests assert on the LINE each real encoder emits, not on the helper in
isolation, because "the next transport bypasses the bound" is the failure mode
worth pinning: three encoders exist (NDJSON, SSE, the runtime socket), and one
of them not calling the bound is invisible from the helper's own tests.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import queue
from contextlib import redirect_stdout
from typing import Any, cast

from local_operator.harness import wire
from local_operator.harness.types import (
    AgentEndEvent,
    AgentMessage,
    ImageContent,
    Message,
    NoticeEvent,
    TextContent,
    ToolCall,
    ToolResult,
    Usage,
)
from local_operator.headless_print import PrintRenderer, strip_provider_payload
from local_operator.mobile.tui_handle import TuiSessionHandle
from local_operator.server.utils.operator import AgentEventBridge
from local_operator.server.utils.sse import envelope, frame
from local_operator.session.protocol import SessionProtocol
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    Transcript,
    read_transcript_page,
)

SESSION_ID = "sess-0123456789abcdef"
JOB_ID = "job-1"

#: The supervisor's own ceiling: ``scanner.Buffer(64*1024, 4*1024*1024)`` in
#: Minerva's ``adapters/lopcli/adapter.go``. Crossing it fails the whole run.
SUPERVISOR_LINE_MAX_BYTES = 4 * 1024 * 1024


def _lorem(n: int, seed: int = 0) -> str:
    unit = (
        "the quick brown fox jumps over the lazy dog 0123456789 "
        "def foo(bar): return bar.baz  # src/main.py\n"
    )
    return (unit * (n // len(unit) + 1))[:n]


def _conversation(rows: int = 50) -> list[AgentMessage]:
    """A 104-message turn when ``rows`` is 50, matching the design's fixture."""
    msgs: list[AgentMessage] = [Message(role="user", content=[TextContent(text=_lorem(1_100))])]
    oversized = [72_000, 66_000, 51_000, 24_000, 21_000]
    sizes = (oversized + [7_000] * 5 + [1_500] * max(0, rows - len(oversized) - 5))[:rows]
    for i, chars in enumerate(sizes):
        call = ToolCall(
            id=f"call_{i:03d}",
            name=["Read", "Grep", "Bash", "Glob"][i % 4],
            arguments={"path": f"src/mod_{i:03d}.py"},
        )
        msgs.append(
            Message(
                role="assistant",
                content=[TextContent(text=_lorem(300))],
                tool_calls=[call],
                usage=Usage(input_tokens=18_000 + i, output_tokens=180 + i, context_tokens=19_000),
            )
        )
        msgs.append(
            Message.tool_result(
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    content=[TextContent(text=_lorem(chars, seed=i))],
                    details={"path": f"src/mod_{i:03d}.py"},
                )
            )
        )
    msgs.append(Message(role="assistant", content=[TextContent(text=_lorem(700))]))
    return msgs


def _conversation_event(rows: int = 50) -> AgentEndEvent:
    return AgentEndEvent(messages=_conversation(rows), generation=831, context_tokens=214_000)


def _small_event() -> AgentEndEvent:
    return AgentEndEvent(
        messages=[
            Message(role="user", content=[TextContent(text="hello")]),
            Message(
                role="assistant",
                content=[TextContent(text="hi")],
                tool_calls=[ToolCall(id="call_0", name="Read", arguments={"path": "a.py"})],
                usage=Usage(input_tokens=10, output_tokens=1),
            ),
            Message.tool_result(
                ToolResult(
                    tool_call_id="call_0",
                    tool_name="Read",
                    content=[TextContent(text="ok")],
                )
            ),
        ],
        generation=1,
    )


class _StubSession:
    """The minimum a renderer or a serving handle touches: an id and subscribe."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.handler: Any = None

    def subscribe(self, handler: Any) -> Any:
        self.handler = handler
        return lambda: None


def _ndjson_line(event: AgentEndEvent, *, session_id: str | None = SESSION_ID) -> bytes:
    """The NDJSON line ``lop exec --json`` writes, through the real renderer."""
    renderer = PrintRenderer(json_mode=True)
    # ``cast`` rather than a real session: the renderer touches only
    # ``subscribe`` and ``session_id``, and building a Session here would test
    # the session rather than the encoder (the same shape
    # ``tests/unit/tui/test_herdr_reporter.py`` uses for its fakes).
    renderer.attach(cast(SessionProtocol, _StubSession(session_id or "")))
    out = io.StringIO()
    with redirect_stdout(out):
        renderer.handle(event)
    return out.getvalue().encode("utf-8")


def _sse_frame(event: AgentEndEvent) -> bytes:
    """The SSE frame the broker publishes, through the real bridge."""
    captured: queue.Queue[Any] = queue.Queue()
    bridge = AgentEventBridge(status_queue=captured, job_id=JOB_ID, session_id=SESSION_ID)
    bridge.handle(event)
    payload = None
    while not captured.empty():
        kind, _job, body = captured.get()
        if kind == "agent_event":
            payload = body
    assert payload is not None, "the bridge published no agent_event"
    body = {key: value for key, value in payload.items() if key != "type"}
    body["job_id"] = JOB_ID
    return frame("agent.end", envelope("agent.end", body)).encode("utf-8")


def _socket_payload(event: AgentEndEvent) -> dict[str, Any]:
    """The payload the runtime relay frames, through the real subscribe handler."""
    relayed: list[dict[str, Any]] = []
    handle = ServingSessionHandle.__new__(ServingSessionHandle)
    stub = _StubSession(SESSION_ID)
    handle._session = stub  # type: ignore[assignment]
    ServingSessionHandle.subscribe_events(handle, relayed.append)
    stub.handler(event)
    return relayed[0]


def _tui_socket_payload(event: AgentEndEvent) -> dict[str, Any]:
    """The same relay payload from the OTHER handle capability implementation.

    ``RuntimeServer`` relays whichever handle the host supplied, and the TUI's
    handle serialises events itself (``mobile/tui_handle.py``). A bound that
    only covered the serving handle would be bypassed by any TUI-owned session.
    """
    relayed: list[dict[str, Any]] = []
    handle = TuiSessionHandle.__new__(TuiSessionHandle)
    stub = _StubSession(SESSION_ID)
    handle._session = lambda: stub  # type: ignore[method-assign]
    TuiSessionHandle.subscribe_events(handle, relayed.append)
    stub.handler(event)
    return relayed[0]


def _size_of(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))


def test_a_small_turn_is_returned_by_identity() -> None:
    """The common path must not get more expensive: no copy, no allocation."""
    payload = _small_event().model_dump(mode="json")
    assert wire.bound_agent_end_for_wire(payload, session_id=SESSION_ID) is payload


def test_a_frame_already_under_budget_is_the_dump_it_always_was() -> None:
    """A small turn's NDJSON line is byte-identical to the unbounded shape."""
    event = _small_event()
    line = _ndjson_line(event)
    dump = strip_provider_payload(event.model_dump(mode="json"))
    expected = json.dumps({**dump, "session_id": SESSION_ID}, ensure_ascii=False) + "\n"
    assert line.decode() == expected


def test_every_encoder_bounds_the_104_message_conversation() -> None:
    """The three transports, each measured on the line/payload it emits."""
    event = _conversation_event()
    budget = wire.AGENT_END_FRAME_BUDGET_BYTES

    assert len(_ndjson_line(event)) <= budget
    sse = _sse_frame(event)
    # The SSE frame is the payload plus the transport's own envelope and framing.
    assert _size_of(_socket_payload(event)) <= budget
    assert _size_of(_tui_socket_payload(event)) <= budget
    assert len(sse) <= budget + 1024, f"SSE frame {len(sse)} bytes"


def test_every_usage_receipt_and_all_conversation_text_survives() -> None:
    """Receipts are the billing record and prose is the conversation: neither
    may be elided, because cost reconciliation and the frozen ``/v1/chat``
    projection read them straight off this frame."""
    event = _conversation_event()
    line = json.loads(_ndjson_line(event))
    source = event.model_dump(mode="json")["messages"]

    assert [m.get("usage") for m in line["messages"] if m.get("usage")] == [
        m.get("usage") for m in source if m.get("usage")
    ]
    for role in ("assistant", "user"):
        assert [
            b.get("text") for m in line["messages"] if m.get("role") == role for b in m["content"]
        ] == [b.get("text") for m in source if m.get("role") == role for b in m["content"]]
    assert [m["id"] for m in line["messages"]] == [m["id"] for m in source]


def test_tool_rows_past_the_preview_cap_carry_an_honest_marker() -> None:
    """A row that loses its content in whole says so, in its own block."""
    event = _conversation_event(rows=wire.AGENT_END_PREVIEW_ROWS_MAX + 20)
    line = json.loads(_ndjson_line(event))
    rows = [m for m in line["messages"] if m.get("role") == "tool"]

    assert len(rows) == wire.AGENT_END_PREVIEW_ROWS_MAX + 20
    assert line["elided_tool_rows"] == 20
    assert line["elided_bytes"] > 0
    elided = [
        m
        for m in rows
        if len(m["content"]) == 1 and "[tool output elided" in m["content"][0]["text"]
    ]
    assert len(elided) == 20
    marker = elided[0]["content"][0]["text"]
    assert f"of session {SESSION_ID}" in marker
    assert f"transcript entry {elided[0]['id']}" in marker
    # The marker is where the reader is sent, so it must not promise the frame.
    assert "http" not in marker and "/Users/" not in marker


def test_a_marker_without_a_session_still_names_what_it_can() -> None:
    """An encoder that has no session id must degrade honestly, not lie.

    The transcript is per session, so without one the reference cannot be
    complete: the marker keeps the entry id and drops the session clause rather
    than naming a session it never saw.
    """
    event = _conversation_event(rows=wire.AGENT_END_PREVIEW_ROWS_MAX + 5)
    payload = wire.bound_agent_end_for_wire(event.model_dump(mode="json"))
    rows = [m for m in payload["messages"] if m.get("role") == "tool"]
    marker = next(
        m["content"][0]["text"]
        for m in rows
        if len(m["content"]) == 1 and "[tool output elided" in m["content"][0]["text"]
    )
    assert "transcript entry " in marker and "of session" not in marker
    assert payload["elided_tool_rows"] == 5


def test_the_marker_s_resolution_is_proved_not_asserted(tmp_path) -> None:
    """The elided row's text is really retrievable at the reference it names.

    The marker claims "full text: transcript entry <id> of session <sid>". This
    writes the SAME message to a real transcript through the real writer and
    reads it back through the real pager with the id the marker named — the only
    form of this claim that is worth anything.
    """
    session_dir = tmp_path / SESSION_ID
    transcript = Transcript(session_dir)
    event = _conversation_event(rows=wire.AGENT_END_PREVIEW_ROWS_MAX + 20)
    # The first tool row is the oldest, so it is one of the elided ones.
    elided_message = next(m for m in event.messages if isinstance(m, Message) and m.role == "tool")
    full_text = elided_message.text
    assert len(full_text) > 10_000
    asyncio.run(transcript.append_message(elided_message))

    line = json.loads(_ndjson_line(event))
    marker = next(
        m["content"][0]["text"]
        for m in line["messages"]
        if m.get("id") == elided_message.id and m.get("role") == "tool"
    )
    assert f"transcript entry {elided_message.id}" in marker
    assert f"of session {SESSION_ID}" in marker

    # ``through_id`` is the inclusive cursor: the page whose NEWEST row is the
    # one the marker names. Every viewer's pager takes the same argument, so
    # this is the read a reader actually performs on the marker.
    page = read_transcript_page(session_dir, through_id=elided_message.id, limit=5)
    assert elided_message.id in [entry.id for entry in page.entries]
    recovered = next(entry for entry in page.entries if entry.id == elided_message.id).payload
    assert "".join(b["text"] for b in recovered["content"]) == full_text
    assert (session_dir / TRANSCRIPT_FILENAME).exists()


def test_the_bound_never_mutates_the_payload_it_was_handed() -> None:
    """The loop's message objects are the durable record and the ledger input."""
    payload = _conversation_event().model_dump(mode="json")
    before = copy.deepcopy(payload)
    wire.bound_agent_end_for_wire(payload, session_id=SESSION_ID)
    assert payload == before


def test_the_bound_fails_open() -> None:
    """A size bound must never turn a normal event into a failed run."""

    def boom(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("simulated measurement failure")

    original = wire._frame_bytes
    wire._frame_bytes = boom  # type: ignore[assignment]
    try:
        payload = _conversation_event().model_dump(mode="json")
        assert wire.bound_agent_end_for_wire(payload, session_id=SESSION_ID) is payload
    finally:
        wire._frame_bytes = original  # type: ignore[assignment]


def test_only_agent_end_is_touched() -> None:
    """Every other frame keeps the byte-identical path it had."""
    notice = NoticeEvent(text=_lorem(1_000)).model_dump(mode="json")
    assert wire.bound_agent_end_for_wire(notice, session_id=SESSION_ID) is notice


def test_the_supervisor_breakpoint_is_no_longer_crossed() -> None:
    """The incident: pasted screenshots used to fail the whole external run.

    Five 700 KB images inline measure a 4,667,340-byte NDJSON line against the
    supervisor's 4 MiB ``bufio.Scanner`` buffer, and ``scanner.Err()`` is
    returned as a run failure. The bound is not image-specific — these are user
    blocks, and stage 4a is what reaches them.
    """
    content: list[Any] = [TextContent(text="look at these")]
    for i in range(5):
        content.append(ImageContent(data="A" * 700_000))
    event = AgentEndEvent(messages=[Message(role="user", content=content)], generation=1)

    line = _ndjson_line(event)
    assert len(line) < SUPERVISOR_LINE_MAX_BYTES
    assert len(_sse_frame(event)) < SUPERVISOR_LINE_MAX_BYTES
    assert wire._frame_bytes(_socket_payload(event)) < SUPERVISOR_LINE_MAX_BYTES
    assert wire._frame_bytes(_tui_socket_payload(event)) < SUPERVISOR_LINE_MAX_BYTES
