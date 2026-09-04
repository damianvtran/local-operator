"""Streaming a plain harness session into the SSE broker.

``publish_agent_event`` had exactly one caller - the ``lop serve`` job pump - so
a host that drives ``Session.prompt`` directly (headless ``exec``, an external
supervisor) produced no SSE events at all while the entire transport sat
unused. These tests pin the seam that closes that gap, and the two properties
that make it worth having rather than a second projection to keep in sync:

* the wire shape is the one the pump already produces, because the same
  ``AgentEventBridge`` produces it;
* ``provider.start`` reaches a subscriber with its provider-native
  ``response_id`` intact, which is the whole reason an external runtime attaches
  at all - it is the token it commits as no-replay proof.
"""

from __future__ import annotations

from typing import Any, Callable, List

import pytest

from local_operator.harness.types import (
    AgentEvent,
    Message,
    MessageStartEvent,
    MessageUpdateEvent,
    ProviderTurnStartEvent,
    SteeringDeliveredEvent,
)
from local_operator.jobs import JobStatus
from local_operator.server.utils.event_broker import (
    EventBroker,
    job_channel,
    message_channel,
)
from local_operator.server.utils.sse import EventName
from local_operator.server.utils.sse_bridge import BrokerStatusSink, attach_broker
from local_operator.server.utils.sse_publisher import publish_job_status


class _FakeSession:
    """Just the ``subscribe`` half of ``SessionProtocol``.

    The bridge only ever calls ``subscribe`` and the callable it returns, so a
    stub keeps these tests on the seam under test instead of on a real session's
    provider, transcript and tool wiring.
    """

    def __init__(self) -> None:
        self.handlers: List[Callable[[AgentEvent], Any]] = []
        self.unsubscribed = 0

    def subscribe(self, handler: Callable[[AgentEvent], Any]) -> Callable[[], None]:
        self.handlers.append(handler)

        def _unsubscribe() -> None:
            self.unsubscribed += 1
            self.handlers.remove(handler)

        return _unsubscribe

    def emit(self, event: AgentEvent) -> None:
        for handler in list(self.handlers):
            handler(event)


def _names(broker: EventBroker, channel: str) -> List[str]:
    return [event.name for event in broker.retained(channel)]


# ---------------------------------------------------------------------------
# the sink
# ---------------------------------------------------------------------------


def test_sink_publishes_records_and_agent_events_to_the_job_channel() -> None:
    """The sink stands in for the parent-side pump, tuple vocabulary included."""
    broker = EventBroker()
    sink = BrokerStatusSink(broker, "job-1")

    sink.put(("agent_event", "job-1", {"type": "turn_start"}))
    sink.put(("agent_event", "job-1", {"type": "notice", "level": "info"}))

    assert _names(broker, job_channel("job-1")) == [EventName.TURN_START, EventName.NOTICE]


def test_sink_drops_the_legacy_execution_placeholder() -> None:
    """``execution_update`` is job-ledger bookkeeping, not a stream frame.

    The pump routes it to the JobManager and never to the broker; emitting one
    here would put a record on the wire that the server path does not.
    """
    broker = EventBroker()
    BrokerStatusSink(broker, "job-1").put(("execution_update", "job-1", object()))
    assert broker.retained(job_channel("job-1")) == []


def test_sink_ignores_malformed_traffic_instead_of_raising() -> None:
    """``put`` runs on the turn's own task, so it must never abort the turn."""
    broker = EventBroker()
    sink = BrokerStatusSink(broker, "job-1")
    sink.put("not-a-tuple")
    sink.put(("agent_event", "job-1"))
    # A payload that is not a mapping cannot be published, and must not escape.
    sink.put(("agent_event", "job-1", None))
    assert broker.retained(job_channel("job-1")) == []


def test_sink_never_raises_when_publishing_fails() -> None:
    """Same failure posture as the publisher, one step earlier in the chain."""

    class _Exploding(EventBroker):
        def publish_with(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("broker down")

    BrokerStatusSink(_Exploding(), "job-1").put(("agent_event", "job-1", {"type": "turn_start"}))


# ---------------------------------------------------------------------------
# attach_broker
# ---------------------------------------------------------------------------


def test_attach_broker_streams_session_events_without_a_server() -> None:
    """The gap this closes: a session with no job pump now reaches the wire."""
    broker = EventBroker()
    session = _FakeSession()
    unsubscribe = attach_broker(session, broker, "job-1")

    message = Message(id="m1", role="assistant")
    session.emit(MessageStartEvent(message=message))
    session.emit(MessageUpdateEvent(message=message, delta="hi"))

    names = _names(broker, job_channel("job-1"))
    # Both taxonomies, from one subscription: the richer engine event AND the
    # legacy record frame the existing UI renders.
    assert EventName.MESSAGE_DELTA in names
    assert EventName.RECORD_UPDATE in names

    unsubscribe()
    assert session.unsubscribed == 1
    assert session.handlers == []


def test_attach_broker_reuses_the_record_projection_verbatim() -> None:
    """A delta frame keeps its running ``snapshot`` and the legacy record body.

    This is why the bridge reuses ``AgentEventBridge`` rather than projecting
    again: the snapshot guarantee and the legacy-compatible record keys are
    behaviour a second implementation would quietly lose.
    """
    broker = EventBroker()
    session = _FakeSession()
    attach_broker(session, broker, "job-1")

    message = Message(id="m1", role="assistant")
    session.emit(MessageStartEvent(message=message))
    session.emit(MessageUpdateEvent(message=message, delta="Hello"))
    session.emit(MessageUpdateEvent(message=message, delta=" world"))

    deltas = [
        event.data
        for event in broker.retained(job_channel("job-1"))
        if event.name == EventName.MESSAGE_DELTA
    ]
    assert [d["delta"] for d in deltas] == ["Hello", " world"]
    assert [d["snapshot"] for d in deltas] == ["Hello", "Hello world"]

    records = [
        event.data["record"]
        for event in broker.retained(job_channel("job-1"))
        if event.name == EventName.RECORD_UPDATE
    ]
    assert records[-1]["message"] == "Hello world"
    # The legacy injections the WebSocket transport made are still present.
    assert records[-1]["message_id"] == "m1"
    assert records[-1]["connection_type"]

    # And the record-keyed channel gets the same fan-out the pump produces.
    assert EventName.MESSAGE_DELTA in _names(broker, message_channel("m1"))


def test_attach_broker_without_a_broker_is_a_no_op() -> None:
    """A host wires this unconditionally; configuration decides if it streams."""
    session = _FakeSession()
    unsubscribe = attach_broker(session, None, "job-1")
    assert session.handlers == []
    unsubscribe()


def test_terminal_framing_stays_the_hosts_decision() -> None:
    """Only the host knows the outcome, so it publishes the terminal frame.

    Without one the client keepalives until it gives up - the exact hang the
    transport exists to remove.
    """
    broker = EventBroker()
    session = _FakeSession()
    attach_broker(session, broker, "job-1")
    session.emit(MessageStartEvent(message=Message(id="m1", role="assistant")))
    assert not broker.is_terminal(job_channel("job-1"))

    publish_job_status(broker, "job-1", JobStatus.COMPLETED)
    assert broker.is_terminal(job_channel("job-1"))
    assert _names(broker, job_channel("job-1"))[-1] == EventName.TERMINAL


# ---------------------------------------------------------------------------
# the two new event types
# ---------------------------------------------------------------------------


def test_provider_start_carries_the_native_response_id_to_the_wire() -> None:
    """The point of the whole exercise.

    An external supervisor commits ``response_id`` as no-replay proof, so it has
    to survive the projection, the sink and the envelope - not just be emitted
    by the harness.
    """
    broker = EventBroker()
    session = _FakeSession()
    attach_broker(session, broker, "job-1")

    session.emit(
        ProviderTurnStartEvent(
            response_id="msg_01ABC",
            provider="anthropic",
            model_id="claude-opus-5",
        )
    )

    events = [
        event for event in broker.retained(job_channel("job-1")) if event.name == "provider.start"
    ]
    assert len(events) == 1
    data = events[0].data
    assert data["response_id"] == "msg_01ABC"
    assert data["provider"] == "anthropic"
    assert data["model_id"] == "claude-opus-5"
    # The dual discriminator the taxonomy promises: event name and inner type
    # always agree, so a client may switch on either.
    assert data["type"] == EventName.PROVIDER_START


def test_provider_start_without_a_native_id_stays_indeterminate() -> None:
    """Google exposes no per-response id; the frame must say so rather than
    invent one, so a consumer requiring proof can fail closed."""
    broker = EventBroker()
    session = _FakeSession()
    attach_broker(session, broker, "job-1")

    session.emit(ProviderTurnStartEvent(response_id=None, provider="google"))

    data = broker.retained(job_channel("job-1"))[0].data
    assert data["response_id"] is None


def test_steering_delivered_reaches_the_wire_with_its_count() -> None:
    """The transition a UI reconciles its "queued" row against.

    ``count`` is load-bearing: three lines sent while a tool ran are delivered
    at one boundary, and a receipt per message would claim three deliveries.
    """
    broker = EventBroker()
    session = _FakeSession()
    attach_broker(session, broker, "job-1")

    session.emit(SteeringDeliveredEvent(count=3))

    events = broker.retained(job_channel("job-1"))
    assert [e.name for e in events] == [EventName.STEERING_DELIVERED]
    assert events[0].data["count"] == 3


@pytest.mark.parametrize(
    "raw_type, expected",
    [
        ("provider_turn_start", EventName.PROVIDER_START),
        ("steering_delivered", EventName.STEERING_DELIVERED),
    ],
)
def test_new_event_types_are_no_longer_dropped(raw_type: str, expected: str) -> None:
    """An unmapped type is DISCARDED by ``publish_agent_event``, silently.

    That is the correct default - a consumer cannot render an event it was never
    told about - which is exactly why adding an event type means adding it here.
    """
    from local_operator.server.utils.sse_publisher import publish_agent_event

    broker = EventBroker()
    publish_agent_event(broker, "job-1", {"type": raw_type})
    assert _names(broker, job_channel("job-1")) == [expected]
