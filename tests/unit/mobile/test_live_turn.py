"""The v4 live-turn seed covers exactly the mid-turn join gap."""

from local_operator.harness.types import (
    AgentEndEvent,
    AgentStartEvent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)
from local_operator.mobile.live_turn import LiveTurnTracker


def _wire(event):  # noqa: ANN001, ANN202
    return event.model_dump(mode="json")


def test_seed_tracks_accumulated_message_and_open_tools() -> None:
    tracker = LiveTurnTracker()
    message = Message.assistant("hello", id="m1")
    tracker.fold(_wire(AgentStartEvent(generation=7)))
    tracker.fold(_wire(MessageStartEvent(message=Message.assistant("", id="m1"))))
    tracker.fold(_wire(MessageUpdateEvent(message=message, delta="hello")))
    tracker.fold(
        _wire(
            ToolExecutionStartEvent(
                tool_call_id="t1", tool_name="bash", args={"command": "echo hi"}
            )
        )
    )

    seed = tracker.seed()
    assert seed.streaming is True
    assert seed.generation == 7
    assert seed.assistant_open is True
    assert seed.assistant_text == "hello"
    assert seed.assistant_message_id == "m1"
    assert [event["type"] for event in seed.open_tools] == ["tool_execution_start"]

    tracker.fold(
        _wire(
            ToolExecutionEndEvent(
                tool_call_id="t1",
                tool_name="bash",
                result=ToolResult(tool_call_id="t1", tool_name="bash"),
            )
        )
    )
    tracker.fold(_wire(MessageEndEvent(message=message)))
    tracker.fold(_wire(AgentEndEvent(generation=7)))
    assert tracker.seed().streaming is False
    assert tracker.seed().open_tools == []


def test_seed_join_at_each_event_index_matches_continuous_open_state() -> None:
    """Property-style regression: every join point reconstructs live state.

    The continuously folded tracker is the owner-side authority. Replaying the
    same prefix into a fresh tracker (what a join seed represents) must produce
    the identical bounded state at every event index.
    """
    message = Message.assistant("part", id="m2")
    stream = [
        AgentStartEvent(generation=3),
        MessageStartEvent(message=Message.assistant("", id="m2")),
        MessageUpdateEvent(message=message, delta="part"),
        ToolExecutionStartEvent(tool_call_id="t2", tool_name="read", args={"path": "x"}),
        ToolExecutionEndEvent(
            tool_call_id="t2",
            tool_name="read",
            result=ToolResult(tool_call_id="t2", tool_name="read"),
        ),
        MessageEndEvent(message=message),
        AgentEndEvent(generation=3),
    ]
    continuous = LiveTurnTracker()
    prefix: list[dict[str, object]] = []
    for event in stream:
        data = _wire(event)
        prefix.append(data)
        continuous.fold(data)
        joined = LiveTurnTracker()
        for prior in prefix:
            joined.fold(prior)
        assert joined.seed().to_json() == continuous.seed().to_json()
