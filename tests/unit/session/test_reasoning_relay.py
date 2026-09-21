"""A relayed reasoning frame decodes into the event its consumer expects.

An attach follower (the daemon, the TUI's remote viewer, the phone) never sees
the harness's event objects: it reads the wire dict the owner relayed and
rehydrates it through ``deserialize_event``. An unregistered type is deliberately
TOLERANT — it falls back to the base ``AgentEvent``, which keeps the fields but
declares none of them — so the failure a missing registry entry produces is not
an error but a silent loss of shape: the TUI dispatches on ``event.type``, and a
reasoning frame without its declared fields is a stream that stops being rendered
the moment it leaves the owner's process.

The reconnect side is the other half: a frame is replayed from the broker's ring
after a drop, so what a follower decodes is exactly what the owner dumped.
"""

from __future__ import annotations

import json

from local_operator.harness.types import AgentEvent, ReasoningDeltaEvent
from local_operator.session.attached import deserialize_event


def test_a_relayed_reasoning_frame_decodes_to_its_declared_type() -> None:
    event = deserialize_event({"type": "reasoning_delta", "message_id": "m1", "delta": "weighing"})
    assert isinstance(event, ReasoningDeltaEvent)
    assert event.message_id == "m1"
    assert event.delta == "weighing"


def test_the_decode_is_the_inverse_of_the_dump_a_reconnect_replays() -> None:
    """Dump -> JSON -> decode must round-trip: that is the reconnect path.

    Asserted through ``json`` rather than dict-in/dict-out because the relay
    serializes, and a field that survives one but not the other (a non-JSON type)
    would work against a fake frame and fail in production.
    """
    original = ReasoningDeltaEvent(message_id="m1", delta="weighing the options")
    wire = json.loads(json.dumps(original.model_dump(mode="json")))
    decoded = deserialize_event(wire)
    assert isinstance(decoded, ReasoningDeltaEvent)
    assert decoded.message_id == original.message_id
    assert decoded.delta == original.delta


def test_an_unregistered_type_still_degrades_instead_of_raising() -> None:
    """The tolerance that makes a newer owner safe for an older follower.

    Pinned beside the reasoning case because the two are the SAME rule seen from
    both sides: adding a member to the registry must not narrow what an unknown
    type does, or a version skew becomes a killed stream.
    """
    event = deserialize_event({"type": "some_future_event", "anything": 1})
    assert type(event) is AgentEvent
    assert event.type == "some_future_event"
