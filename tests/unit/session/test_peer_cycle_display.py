"""Finite display attestations: never a model-visible nested tool result."""
import pytest
from local_operator.harness.peer_cycle_display import DisplayCycleAccumulator, record_id
from local_operator.harness.types import (
    Message, MessageEndEvent, PeerMessageDeliveredEvent, ToolCall,
    ToolExecutionStartEvent, ToolExecutionEndEvent, ToolResult, TextContent,
)


def exercise(*, child_name="no_reply", failed=False, unknown_child=False,
             peer_custom_type="", interrupt_custom_type=None):
    fold = DisplayCycleAccumulator("display-test", "epoch-test")
    fold.observe(PeerMessageDeliveredEvent(message_id="peer-test", body="private incoming", sender={},
                                          custom_type=peer_custom_type))
    if interrupt_custom_type is not None:
        fold.observe(PeerMessageDeliveredEvent(message_id="bridge-test", body="private boundary", sender={},
                                              custom_type=interrupt_custom_type))
    parent = Message(id="parent-test", role="assistant", tool_calls=[
        ToolCall(id="send-root", name="send", arguments={"message": "reply"}),
        ToolCall(id="quiet-root", name="eval", arguments={"code": "not parsed"}),
    ])
    fold.observe(MessageEndEvent(message=parent))
    for name, call, args in [("send", "send-root", {"message": "reply"}),
                             ("eval", "quiet-root", {"code": "not parsed"})]:
        fold.observe(ToolExecutionStartEvent(tool_name=name, tool_call_id=call, args=args))
        if name == "send":
            fold.observe(ToolExecutionEndEvent(tool_name=name, tool_call_id=call,
                result=ToolResult(tool_name=name, tool_call_id=call, content=[TextContent(text="sent")]), is_error=False))
    if unknown_child:
        fold.observe(ToolExecutionStartEvent(tool_name="bash", tool_call_id="quiet-root:" + "b" * 32, args={}))
    # The real eval worker namespaces uuid4.hex child IDs below its known root.
    # A human placeholder suffix exercises namespace refusal, not positive seal.
    call = "quiet-root:" + "a" * 32
    fold.observe(ToolExecutionStartEvent(tool_name=child_name, tool_call_id=call, args={}))
    fold.observe(ToolExecutionEndEvent(tool_name=child_name, tool_call_id=call,
        result=ToolResult(tool_name=child_name, tool_call_id=call, details={"__quiet_turn": True}, is_error=failed), is_error=failed))
    # Child quiet does NOT terminate the outer eval or publish premature proof.
    assert fold.records == []
    fold.observe(ToolExecutionEndEvent(tool_name="eval", tool_call_id="quiet-root",
        result=ToolResult(tool_name="eval", tool_call_id="quiet-root", details={"kernel_generation": "opaque"}), is_error=False))
    return fold


@pytest.mark.parametrize("custom_type", ["", "peer_message"])
def test_legacy_and_explicit_peer_provenance_still_publish_one_record(custom_type):
    fold = exercise(peer_custom_type=custom_type)
    assert len(fold.records) == 1
    assert fold.records[0]["member_ids"] == ["peer-test:0", "tool:send-root:0", "tool:quiet-root:0"]
    assert fold.records[0]["logical_actions"] == 1


@pytest.mark.parametrize("position", ["opener", "pending"])
@pytest.mark.parametrize("custom_type", ["team_bridge_message", "future_custom_kind"])
def test_nonpeer_typed_delivery_never_opens_or_extends_display_publication(position, custom_type):
    from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE

    # The event class is shared by transport, not an ownership grant: the real
    # bridge discriminator must invalidate an earlier peer before tools execute.
    kind = TEAM_BRIDGE_MESSAGE_TYPE if custom_type == "team_bridge_message" else custom_type
    if position == "opener":
        fold = exercise(peer_custom_type=kind)
    else:
        fold = exercise(interrupt_custom_type=kind)
    assert fold.peer_id is None
    assert fold.records == []
    assert fold.roots == {} and fold.quiet is None


def test_mixed_seals_on_actual_quiet_root_end_without_later_final():
    fold = exercise()
    assert len(fold.records) == 1
    record = fold.records[0]
    assert record["member_ids"] == ["peer-test:0", "tool:send-root:0", "tool:quiet-root:0"]
    assert record["child_rows"] == []
    assert record["logical_actions"] == 1
    assert record["id"] == record_id("display-test", "parent-test", "quiet-root")
    assert "private incoming" not in str(record)
    assert "not parsed" not in str(record)


def test_actual_typed_parent_media_is_a_boundary_not_an_empty_text_carrier():
    from local_operator.harness.types import ImageContent
    fold = DisplayCycleAccumulator("display-test", "epoch-test")
    fold.observe(PeerMessageDeliveredEvent(message_id="peer-test", body="incoming", sender={}))
    parent = Message(id="parent-test", role="assistant", content=[ImageContent(data="eA==")],
                     tool_calls=[ToolCall(id="root", name="eval", arguments={"code": "opaque"})])
    fold.observe(MessageEndEvent(message=parent))
    assert fold.peer_id is None and not fold.roots and not fold.records


def test_empty_tool_parent_remains_eligible_but_first_public_token_refuses():
    for text in ("", "first public token"):
        fold = DisplayCycleAccumulator("display-test", "epoch-test")
        fold.observe(PeerMessageDeliveredEvent(message_id="peer-test", body="incoming", sender={}))
        parent = Message(id="parent-test", role="assistant", content=[TextContent(text=text)],
                         tool_calls=[ToolCall(id="root", name="eval", arguments={"code": "opaque"})])
        fold.observe(MessageEndEvent(message=parent))
        assert (fold.peer_id == "peer-test" and bool(fold.roots)) is (not text)


def test_new_actual_reasoning_does_not_reopen_or_erase_sealed_facts():
    from local_operator.harness.types import ReasoningDeltaEvent
    fold = exercise()
    sealed = list(fold.records)
    fold.observe(ReasoningDeltaEvent(message_id="new-parent", delta="new thinking"))
    assert fold.peer_id is None and fold.records == sealed


@pytest.mark.parametrize("case", ["reasoning", "provider", "compose", "never-sent-compose",
                                      "unknown", "steer", "notice", "fake-reasoning", "fake-update"])
def test_only_known_lifecycle_scalars_preserve_pending_peer(case):
    from local_operator.harness.types import (
        AgentEvent, ProviderTurnStartEvent, ReasoningDeltaEvent, ToolCallComposeEvent,
    )
    fold = DisplayCycleAccumulator("display-test", "epoch-test")
    fold.observe(PeerMessageDeliveredEvent(message_id="peer-test", body="incoming", sender={}))
    original_members = list(fold.members)
    if case == "reasoning":
        event = ReasoningDeltaEvent(message_id="parent-test", delta="synthetic private reasoning")
    elif case == "provider":
        event = ProviderTurnStartEvent(response_id="synthetic-response")
    elif case in {"compose", "never-sent-compose"}:
        event = ToolCallComposeEvent(tool_call_id="root", tool_name="eval",
            not_run_reason="not executed" if case == "never-sent-compose" else None)
    else:
        event = AgentEvent(type={"unknown": "future_event", "steer": "steering_delivered",
                                "notice": "notice", "fake-reasoning": "reasoning_delta",
                                "fake-update": "tool_execution_update"}[case])
    fold.observe(event)
    allowed = case in {"reasoning", "provider", "compose"}
    assert (fold.peer_id == "peer-test") is allowed
    assert not fold.roots and fold.quiet is None and not fold.records
    assert fold.members == (original_members if allowed else [])
    # A scalar is not an opener even when its event family is known.
    unopened = DisplayCycleAccumulator("display-test", "epoch-test")
    unopened.observe(event)
    assert unopened.peer_id is None and not unopened.members and not unopened.records


@pytest.mark.parametrize("case", ["text", "eval-display", "unknown-call", "wrong-name",
                                      "media", "attention", "unknown-details", "late-end"])
def test_tool_update_is_benign_only_for_current_permitted_call_and_safe_payload(case):
    from local_operator.harness.types import AgentToolUpdate, ImageContent, ToolExecutionUpdateEvent
    fold = DisplayCycleAccumulator("display-test", "epoch-test")
    fold.observe(PeerMessageDeliveredEvent(message_id="peer-test", body="incoming", sender={}))
    fold.observe(MessageEndEvent(message=Message(id="parent-test", role="assistant", tool_calls=[
        ToolCall(id="root", name="eval", arguments={"code": "opaque"})])))
    fold.observe(ToolExecutionStartEvent(tool_call_id="root", tool_name="eval", args={"code": "opaque"}))
    if case == "late-end":
        fold.observe(ToolExecutionEndEvent(tool_call_id="root", tool_name="eval",
            result=ToolResult(tool_call_id="root", tool_name="eval", content=[TextContent(text="done")])))
    details = {"tool_name": "eval", "display": True} if case == "eval-display" else (
        {"requires_attention": True} if case == "attention" else (
        {"unrecognized_progress": True} if case == "unknown-details" else None))
    update = AgentToolUpdate(content=[ImageContent(data="eA==")] if case == "media" else [TextContent(text="progress")], details=details)
    fold.observe(ToolExecutionUpdateEvent(tool_call_id="foreign" if case == "unknown-call" else "root",
        tool_name="send" if case == "wrong-name" else "eval", partial_result=update))
    allowed = case in {"text", "eval-display"}
    assert (fold.peer_id == "peer-test") is allowed
    assert fold.quiet is None and not fold.records
    if allowed:
        assert fold.calls["root"]["result"] is None
        assert [member["source_id"] for member in fold.members] == ["peer-test:0", "tool:root:0"]


def test_unknown_or_failed_child_cannot_attest_quiet():
    assert exercise(unknown_child=True).records == []
    assert exercise(failed=True).records == []
    assert exercise(child_name="other").records == []
