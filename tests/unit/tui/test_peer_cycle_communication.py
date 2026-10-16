"""The send name is overloaded: only its typed communication arm can close a cycle.

SOURCE-authored after the first real-producer failure; no execution claimed.
The live row adapter and raw replay must both use the same original arguments,
not success text, target aliases, or a model-switch receipt named ``send``.
"""
from __future__ import annotations

import pytest

from local_operator.harness.peer_cycles import is_peer_communication_call
from local_operator.harness.types import Message, ToolResult
from local_operator.tui.closed_peer_cycles import PeerCycleReplay
from tests.unit.tui.test_closed_peer_cycles import IDENTITY, raw_cycle


@pytest.mark.parametrize("name,arguments,expected", [
    ("send", {"target": "other", "message": "ack"}, True),
    ("send", {"message": "", "model": None}, True),
    ("send", {"model": "synthetic/model"}, False),
    ("send", {"message": "ack", "model": "synthetic/model"}, False),
    ("send", {"message": None}, False),
    ("send", {"message": 17}, False),
    ("send", {}, False),
    ("send", None, False),
    ("send", "message='ack'", False),
    ("read", {"message": "ack"}, False),
])
def test_original_arguments_select_communication_not_model_mutation(name, arguments, expected):
    assert is_peer_communication_call(name, arguments) is expected


@pytest.mark.parametrize("arguments", [
    {"model": "synthetic/model"},
    {"message": "ack", "model": "synthetic/model"},
    {}, {"message": None}, {"message": 17},
])
def test_successful_send_named_model_or_nonmessage_call_never_certifies_replay(arguments):
    messages = raw_cycle()
    declaration = messages[1]
    assert isinstance(declaration, Message)
    declaration.tool_calls[0].arguments = arguments
    replay = PeerCycleReplay(IDENTITY)
    closed = [close for message in messages if (close := replay.step(message)) is not None]
    assert len(closed) == 0


@pytest.mark.parametrize("fault", ["refused", "unknown-success", "never-sent"])
def test_communication_arguments_without_known_success_are_not_enough(fault):
    messages = raw_cycle()
    if fault == "refused":
        messages[2] = Message.tool_result(ToolResult(tool_call_id="send", tool_name="send", is_error=True))
    elif fault == "unknown-success":
        del messages[2]
    else:
        messages[2] = Message.tool_result(ToolResult(tool_call_id="send", tool_name="send", details={"never_sent": True}))
    replay = PeerCycleReplay(IDENTITY)
    assert not any(replay.step(message) is not None for message in messages)


@pytest.mark.parametrize("arguments,expected_kind", [
    ({"message": "ack"}, "send"),
    ({"model": "synthetic/model"}, "boundary"),
    ({"message": "ack", "model": "synthetic/model"}, "boundary"),
    ({}, "boundary"),
])
def test_live_card_adapter_uses_original_arguments_even_when_output_says_delivered(arguments, expected_kind):
    from local_operator.tui.widgets.tool_card import ToolCard
    from local_operator.tui.widgets.transcript import peer_cycle_rows_of_blocks

    card = ToolCard("send", "send", arguments)
    card.work_view_identity = IDENTITY
    card.mark_done("delivered", result=ToolResult(tool_call_id="send", tool_name="send"))
    records = peer_cycle_rows_of_blocks([card])
    assert len(records) == 1
    assert records[0].kind == expected_kind
