"""The runtime event relay and reconnect seed publish the same legacy prose."""

from types import SimpleNamespace

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
)
from local_operator.session.frontend_state import FrontendStateStore
from local_operator.session.runtime.inline_ui_egress import InlineUiEvents
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.test_frontend_state import _state


@pytest.mark.parametrize("marked", [True, False])
def test_runtime_live_and_reconnect_egress(marked: bool) -> None:
    relay = object.__new__(ServingSessionHandle)
    listeners = []
    relay._session = SimpleNamespace(subscribe=lambda callback: listeners.append(callback))
    frames = []
    relay.subscribe_events(frames.append)
    seed = FrontendStateStore(_state())
    msg = Message(role="assistant")
    msg._inline_ui_source_version = 1 if marked else None

    def emit(event):
        listeners[0](event)
        seed._fold_live_event(event)

    emit(MessageStartEvent(message=msg))
    raw = 'Before.\n\n```lop-ui\n{"type":"bar","title":"Internal","data":[["A",1]]}\n```\nAfter.'
    for char in raw:
        emit(MessageUpdateEvent(message=msg, delta=char))
        if marked:
            assert "lop-ui" not in frames[-1]["delta"]
            assert "Internal" not in str(seed.live_events_copy())
    prose = "Before.\n\n\nAfter." if marked else raw
    assert "".join(frame.get("delta", "") for frame in frames) == prose
    last_seed = next(
        frame for frame in seed.live_events_copy() if frame["type"] == "message_update"
    )
    assert last_seed["delta"] == (prose if marked else raw[-1:])

    mid = raw.index("lop-ui") + 2
    msg.content = [TextContent(text=raw[:mid]), TextContent(text=raw[mid:])]
    emit(MessageEndEvent(message=msg))
    assert "".join(block["text"] for block in frames[-1]["message"]["content"]) == prose
    emit(AgentEndEvent(messages=[msg]))
    assert "".join(block["text"] for block in frames[-1]["messages"][0]["content"]) == prose
    assert msg.text == raw


def test_sidecar_is_cumulative_validated_and_kept_out_of_message_payload() -> None:
    marked = Message(role="assistant")
    marked._inline_ui_source_version = 1
    projector = InlineUiEvents(sidecars=True)
    lead = "Before 📈\n\n"
    fence = '```lop-ui\n{"type":"bar","title":"Usage","data":[["A",1]],"extra":true}\n```'
    for char in lead + fence:
        assert "inline_ui" not in projector.dump(MessageUpdateEvent(message=marked, delta=char))
    complete = projector.dump(MessageUpdateEvent(message=marked, delta="\n"))
    widgets = [
        {"offset": len(lead) + 1, "payload": {"type": "bar", "title": "Usage", "data": [["A", 1]]}}
    ]
    message_sidecar = {"version": 1, "text": lead + "\n", "widgets": widgets}
    expected = {"version": 1, "messages": {marked.id: message_sidecar}}
    assert complete["inline_ui"] == expected
    after = projector.dump(MessageUpdateEvent(message=marked, delta="After."))
    message_sidecar["text"] = lead + "\nAfter."
    assert after["inline_ui"] == expected
    assert "inline_ui" not in after["message"]
    assert after["delta"] == "After."
    marked.content = [TextContent(text=lead + fence + "\nAfter.")]
    assert projector.dump(MessageEndEvent(message=marked))["inline_ui"] == expected


def test_authoritative_final_retracts_a_provisional_widget() -> None:
    message = Message(role="assistant")
    message._inline_ui_source_version = 1
    projector = InlineUiEvents(sidecars=True)
    event = projector.dump(
        MessageUpdateEvent(
            message=message,
            delta='```lop-ui\n{"type":"bar","title":"Provisional","data":[["A",1]]}\n```\n',
        )
    )
    assert event["inline_ui"]["messages"][message.id]["widgets"]
    message.content = [TextContent(text='```lop-ui\n{"type":"bar",')]
    final = projector.dump(MessageEndEvent(message=message))
    assert final["inline_ui"] == {"version": 1, "messages": {message.id: None}}
    assert '"type":"bar"' not in str(final)
