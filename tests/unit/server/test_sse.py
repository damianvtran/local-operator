"""Tests for the SSE transport: broker semantics, framing, and route behaviour.

Each test defends a property a client depends on, and most of them exist
because the property was violated during development:

* a subscriber set that could not hold a subscriber (unhashable dataclass);
* a finished channel that never closed, so a late listener hung forever;
* an early close that would have swallowed a resume backlog.

Live end-to-end coverage (a real agent turn, resume across a disconnect, and
websocket/SSE record parity) lives in ``docs/VERIFICATION.md``; these are the
deterministic invariants.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from local_operator.server.routes.sse import (
    job_channel,
    message_channel,
    resolve_cursor,
    snapshot_from,
)
from local_operator.server.utils.event_broker import EventBroker
from local_operator.server.utils.sse import (
    EventName,
    comment,
    envelope,
    frame,
    gap_payload,
    heartbeat,
    open_payload,
)
from local_operator.server.utils.sse_publisher import legacy_record_frame, publish_record
from local_operator.server.utils.websocket_manager import WebsocketConnectionType
from local_operator.types import (
    CodeExecutionResult,
    ConversationRole,
    ExecutionType,
    ProcessResponseStatus,
)


def _record(**overrides) -> CodeExecutionResult:
    payload = {
        "message": "hello",
        "role": ConversationRole.ASSISTANT,
        "status": ProcessResponseStatus.IN_PROGRESS,
        "execution_type": ExecutionType.RESPONSE,
        "is_streamable": True,
    }
    payload.update(overrides)
    return CodeExecutionResult(**payload)


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def test_frame_emits_id_retry_event_and_single_data_line() -> None:
    """The frame layout is the contract; a stray blank line would split it."""
    text = frame("record.update", {"type": "record.update", "seq": 7}, event_id=7, retry_ms=1000)
    lines = text.split("\n")
    assert lines[0] == "id: 7"
    assert lines[1] == "retry: 1000"
    assert lines[2] == "event: record.update"
    assert lines[3].startswith("data: ")
    # Exactly one blank-line terminator, and nothing after it.
    assert text.endswith("\n\n")
    assert len([line for line in lines if line == ""]) == 2


def test_frame_never_lets_a_newline_split_the_event() -> None:
    """A newline inside a value must not end the frame early.

    ``json.dumps`` escapes newlines, so the payload stays on one ``data:``
    line - this test pins that, because a multi-line body would be read as two
    frames and the second would be garbage.
    """
    text = frame("message.delta", {"delta": "line one\nline two\n\nline three"})
    data_lines = [line for line in text.split("\n") if line.startswith("data: ")]
    assert len(data_lines) == 1
    restored = json.loads(data_lines[0][6:])
    assert restored["delta"] == "line one\nline two\n\nline three"


def test_frame_degrades_unserialisable_values_instead_of_dropping_the_event() -> None:
    """An exotic field must not cost the whole frame."""

    class Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    text = frame("notice", {"thing": Opaque()})
    body = json.loads([line for line in text.split("\n") if line.startswith("data: ")][0][6:])
    assert body["thing"] == "<opaque>"


def test_comment_and_heartbeat_are_valid_sse_comments() -> None:
    assert comment("connected") == ": connected\n\n"
    assert heartbeat() == ": heartbeat\n\n"


def test_envelope_mirrors_the_event_name_as_the_inner_discriminator() -> None:
    """Both discriminators must agree, or a client switching on either breaks."""
    body = envelope(EventName.MESSAGE_DELTA, {"delta": "x"}, seq=3, channel="job:1")
    assert body["type"] == EventName.MESSAGE_DELTA
    assert body["seq"] == 3
    assert body["channel"] == "job:1"
    assert body["delta"] == "x"


def test_open_payload_reports_snapshot_size_and_resume_state() -> None:
    body = open_payload("job:1", last_seq=9, resumed=True, snapshot=[{"id": "a"}, {"id": "b"}])
    assert body["last_seq"] == 9
    assert body["resumed"] is True
    assert body["snapshot_count"] == 2
    assert body["transport"] == "sse"


def test_gap_payload_always_demands_reconciliation() -> None:
    """A gap the client can ignore is a gap that becomes a rendering bug."""
    assert gap_payload("overflow", dropped=4)["reconcile"] is True
    assert gap_payload("evicted", expected_seq=12)["expected_seq"] == 12


# ---------------------------------------------------------------------------
# cursor resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [
        (None, None, None),
        ("5", None, 5),
        (None, 5, 5),
        # The furthest-forward cursor wins: on a browser auto-reconnect the
        # header is current while the URL may still hold the cursor the stream
        # was opened with, and replaying from the stale one duplicates events.
        ("9", 3, 9),
        ("3", 9, 9),
        # A malformed header is ignored rather than fatal - it arrives from the
        # browser without application involvement.
        ("not-a-number", 4, 4),
        ("not-a-number", None, None),
    ],
)
def test_resolve_cursor(header, query, expected) -> None:
    assert resolve_cursor(header, query) == expected


# ---------------------------------------------------------------------------
# broker
# ---------------------------------------------------------------------------


def test_subscribers_are_identity_keyed_not_value_keyed() -> None:
    """Two listeners with identical counters must stay two listeners.

    Regression: ``_Subscriber`` was a plain dataclass, which is unhashable and
    could not enter the subscriber set at all - every stream 500'd on attach.
    """
    broker = EventBroker()
    first = broker.subscribe("c")
    second = broker.subscribe("c")
    assert broker.subscriber_count("c") == 2
    first.close()
    assert broker.subscriber_count("c") == 1
    second.close()


def test_publish_assigns_contiguous_sequences_per_channel() -> None:
    """Each channel owns its sequence space, so resuming one is unaffected by
    traffic on the other."""
    broker = EventBroker()
    a1 = broker.publish("a", EventName.NOTICE, {})
    b1 = broker.publish("b", EventName.NOTICE, {})
    a2 = broker.publish("a", EventName.NOTICE, {})
    assert (a1.seq, a2.seq) == (1, 2)
    assert b1.seq == 1


@pytest.mark.asyncio
async def test_subscriber_receives_live_events_in_order() -> None:
    broker = EventBroker()
    sub = broker.subscribe("c")
    for i in range(3):
        broker.publish("c", EventName.NOTICE, {"i": i})
    seen = [await sub.get(timeout=1) for _ in range(3)]
    assert [event.data["i"] for event in seen if event] == [0, 1, 2]
    sub.close()


@pytest.mark.asyncio
async def test_resume_replays_only_events_after_the_cursor() -> None:
    broker = EventBroker()
    for i in range(5):
        broker.publish("c", EventName.NOTICE, {"i": i})
    sub = broker.subscribe("c", after_seq=3)
    replayed = [await sub.get(timeout=1) for _ in range(2)]
    assert [event.seq for event in replayed if event] == [4, 5]
    assert sub.resumed_with_gap is False
    sub.close()


@pytest.mark.asyncio
async def test_resume_from_an_evicted_cursor_reports_a_gap() -> None:
    """A client away longer than the buffer must be told, not quietly reset."""
    broker = EventBroker(replay_buffer=3)
    for i in range(10):
        broker.publish("c", EventName.NOTICE, {"i": i})
    sub = broker.subscribe("c", after_seq=1)
    assert sub.resumed_with_gap is True
    sub.close()


def test_replay_buffer_is_bounded() -> None:
    broker = EventBroker(replay_buffer=4)
    for i in range(50):
        broker.publish("c", EventName.NOTICE, {"i": i})
    retained = broker.retained("c")
    assert len(retained) == 4
    assert [event.data["i"] for event in retained] == [46, 47, 48, 49]
    # The sequence keeps counting even though the buffer rolled.
    assert broker.last_sequence("c") == 50


@pytest.mark.asyncio
async def test_slow_subscriber_is_dropped_and_counted_not_grown() -> None:
    """Overflow must be reported so the client reconciles."""
    broker = EventBroker(subscriber_queue=3)
    sub = broker.subscribe("c")
    for i in range(10):
        broker.publish("c", EventName.NOTICE, {"i": i})
    assert sub.dropped == 7
    assert sub.take_dropped() == 7
    # Reported once, then cleared.
    assert sub.dropped == 0
    sub.close()


@pytest.mark.asyncio
async def test_get_returns_none_on_timeout_without_closing() -> None:
    """A timeout is the heartbeat tick, not a shutdown."""
    broker = EventBroker()
    sub = broker.subscribe("c")
    assert await sub.get(timeout=0.01) is None
    assert sub.is_closed is False
    sub.close()


@pytest.mark.asyncio
async def test_close_wakes_a_parked_reader_and_marks_closed() -> None:
    """Shutdown must not leave a response parked until a proxy times it out."""
    broker = EventBroker()
    sub = broker.subscribe("c")

    async def reader():
        return await sub.get(timeout=5)

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.01)
    broker.close()
    assert await task is None
    assert sub.is_closed is True


def test_idle_channels_are_evicted_but_subscribed_ones_are_never_cut() -> None:
    broker = EventBroker(channel_ttl_s=0.0)
    broker.publish("idle", EventName.NOTICE, {})
    live = broker.subscribe("live")
    broker.publish("live", EventName.NOTICE, {})
    # Opening a channel triggers the eviction pass.
    broker.publish("trigger", EventName.NOTICE, {})
    assert "idle" not in broker.channel_names()
    assert "live" in broker.channel_names()
    live.close()


def test_channel_ceiling_sheds_the_least_recently_active() -> None:
    broker = EventBroker(max_channels=5, channel_ttl_s=1e9)
    for i in range(20):
        broker.publish(f"c{i}", EventName.NOTICE, {})
    assert len(broker.channel_names()) <= 6  # ceiling plus the one just opened


def test_publish_with_lets_the_body_carry_its_own_sequence() -> None:
    """The envelope echoes ``seq`` so a client persisting bodies can resume."""
    broker = EventBroker()
    event = broker.publish_with(
        "c", EventName.NOTICE, lambda seq: envelope(EventName.NOTICE, {}, seq=seq, channel="c")
    )
    assert event.seq == event.data["seq"] == 1


# ---------------------------------------------------------------------------
# publisher: legacy compatibility and terminal semantics
# ---------------------------------------------------------------------------


def test_legacy_record_frame_matches_the_websocket_wire_shape() -> None:
    """SSE must be a transport swap, so the record keys cannot drift.

    ``WebSocketManager.broadcast()`` dumps the record then injects
    ``message_id`` and ``connection_type``; this reproduces both.
    """
    record = _record()
    sse_frame = legacy_record_frame(record, record.id)
    expected = record.model_dump()
    expected["message_id"] = record.id
    expected["connection_type"] = WebsocketConnectionType.MESSAGE.value
    assert sse_frame == expected


def test_publish_record_fans_out_to_both_the_record_and_job_channels() -> None:
    broker = EventBroker()
    record = _record()
    publish_record(broker, "job-1", record)
    assert broker.last_sequence(message_channel(record.id)) >= 1
    assert broker.last_sequence(job_channel("job-1")) == 1


def test_completed_record_ends_its_own_channel_but_not_the_job_channel() -> None:
    """Regression: a finished record channel used to stay open forever, so a
    late listener hung until a proxy killed it."""
    broker = EventBroker()
    record = _record(status=ProcessResponseStatus.SUCCESS, is_complete=True)
    publish_record(broker, "job-1", record)

    assert broker.is_terminal(message_channel(record.id)) is True
    assert broker.is_terminal(job_channel("job-1")) is False
    # The last retained frame on the record channel is the terminal one, which
    # is what lets the route close immediately on a late attach.
    assert broker.retained(message_channel(record.id))[-1].name == EventName.TERMINAL


def test_publish_record_names_completion_distinctly() -> None:
    broker = EventBroker()
    publish_record(broker, "job-1", _record())
    publish_record(broker, "job-1", _record(is_complete=True))
    names = [event.name for event in broker.retained(job_channel("job-1"))]
    assert names == [EventName.RECORD_UPDATE, EventName.RECORD_COMPLETE]


def test_publisher_never_raises_when_the_broker_is_absent() -> None:
    """The pump must survive a server built without streaming."""
    publish_record(None, "job-1", _record())


def test_snapshot_folds_cumulative_records_to_current_state() -> None:
    """Attach cost must not scale with turn length.

    A long message produces hundreds of cumulative frames; the snapshot is one
    entry per record, newest wins, in first-appearance order.
    """
    broker = EventBroker()
    first = _record(message="a")
    for text in ("a", "ab", "abc"):
        publish_record(broker, "job-1", _record(id=first.id, message=text))
    other = _record(message="tool")
    publish_record(broker, "job-1", other)

    snapshot = snapshot_from(broker, job_channel("job-1"))
    assert [entry["id"] for entry in snapshot] == [first.id, other.id]
    assert snapshot[0]["message"] == "abc"


def test_snapshot_ignores_non_record_events() -> None:
    broker = EventBroker()
    broker.publish("job:x", EventName.MESSAGE_DELTA, {"delta": "hi"})
    assert snapshot_from(broker, "job:x") == []


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capabilities_advertises_sse_first_and_names_its_events(test_app_client) -> None:
    """A client negotiates from this; a 404 here means fall back to websockets."""
    response = await test_app_client.get("/v1/sse/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["preferred"] == "sse"
    assert body["transports"] == ["sse", "websocket"]
    assert body["websocket"]["deprecated"] is True
    # The event vocabulary is data, so a newer client can detect an older
    # backend instead of waiting forever for a frame it will never receive.
    for required in (
        EventName.OPEN,
        EventName.RECORD_UPDATE,
        EventName.RECORD_COMPLETE,
        EventName.MESSAGE_DELTA,
        EventName.TOOL_START,
        EventName.TOOL_END,
        EventName.TERMINAL,
        EventName.GAP,
    ):
        assert required in body["sse"]["events"]
    assert body["sse"]["resume"]["last_event_id"] is True


@pytest.mark.asyncio
async def test_stream_opens_with_a_snapshot_and_closes_on_terminal(test_app_client) -> None:
    """The whole late-attach story in one request: state, then a clean close."""
    broker = test_app_client._transport.app.state.event_broker  # type: ignore[attr-defined]
    record = _record(status=ProcessResponseStatus.SUCCESS, is_complete=True, message="done")
    publish_record(broker, "job-late", record)

    response = await test_app_client.get(f"/v1/sse/messages/{record.id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    # Proxies buffer text/event-stream without this, which turns a live stream
    # into one burst at the end.
    assert response.headers["x-accel-buffering"] == "no"

    text = response.text
    assert ": connected" in text
    assert f"event: {EventName.OPEN}" in text
    assert f"event: {EventName.TERMINAL}" in text
    assert "retry: 1000" in text

    open_body = json.loads(text.split(f"event: {EventName.OPEN}\ndata: ")[1].split("\n")[0])
    assert open_body["snapshot_count"] == 1
    assert open_body["snapshot"][0]["message"] == "done"
    assert open_body["resumed"] is False


@pytest.mark.asyncio
async def test_stream_reports_a_gap_when_the_cursor_was_evicted(test_app_client) -> None:
    broker = test_app_client._transport.app.state.event_broker  # type: ignore[attr-defined]
    channel = job_channel("job-gap")
    for _ in range(400):  # overruns the 256-event replay buffer
        broker.publish(channel, EventName.NOTICE, {"type": EventName.NOTICE})
    broker.publish(channel, EventName.TERMINAL, {"type": EventName.TERMINAL}, terminal=True)

    response = await test_app_client.get("/v1/sse/jobs/job-gap", headers={"Last-Event-ID": "1"})
    assert response.status_code == 200
    assert f"event: {EventName.GAP}" in response.text
    gap_line = response.text.split(f"event: {EventName.GAP}\ndata: ")[1].split("\n")[0]
    body = json.loads(gap_line)
    assert body["reason"] == "evicted"
    assert body["reconcile"] is True


@pytest.mark.asyncio
async def test_resume_backlog_is_not_swallowed_by_the_terminal_shortcut(
    test_app_client,
) -> None:
    """Regression: the early close for a finished channel skipped replay.

    A client resuming a completed turn must still receive what it missed
    *before* the stream closes.
    """
    broker = test_app_client._transport.app.state.event_broker  # type: ignore[attr-defined]
    channel = job_channel("job-resume")
    broker.publish(channel, EventName.NOTICE, {"type": EventName.NOTICE, "n": 1})
    broker.publish(channel, EventName.NOTICE, {"type": EventName.NOTICE, "n": 2})
    broker.publish(channel, EventName.TERMINAL, {"type": EventName.TERMINAL}, terminal=True)

    response = await test_app_client.get("/v1/sse/jobs/job-resume", headers={"Last-Event-ID": "1"})
    text = response.text
    # Event 2 was missed and must be replayed; event 1 was already seen.
    assert '"n":2' in text
    assert '"n":1' not in text
    assert f"event: {EventName.TERMINAL}" in text
