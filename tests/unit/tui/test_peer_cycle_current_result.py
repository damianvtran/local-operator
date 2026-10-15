"""Current typed results, not historical or manually painted success, certify COMM.

SOURCE-authored; execution and source-discriminating controls remain pending.
No Widget-valued equality assertions: failures must stay primitive/readable.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.harness.peer_cycles import current_peer_communication_succeeded
from local_operator.harness.types import FAULT_KEY, OUTPUT_LIMIT_KEY, Message, ToolResult
from local_operator.tui.closed_peer_cycles import PeerCycleReplay, begin_peer_cycle, close_peer_cycle
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import TranscriptView, peer_cycle_rows_of_blocks
from tests.unit.tui.test_closed_peer_cycles import IDENTITY, quiet, raw_cycle, rows


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("details,expected", [
    ({}, True),
    ({"delivery": {"state": "delivered", "message_id": "synthetic-id"}}, True),
    ({"duplicate_suppressed": True, "original_receipt": {"details": {}}}, False),
    ({"not_run": True}, False), ({"never_sent": True}, False),
    ({"partial_result": True}, False), ({FAULT_KEY: "refused"}, False),
    ({OUTPUT_LIMIT_KEY: True}, False),
])
def test_live_and_provider_payload_details_have_identical_current_result_admission(durable, details, expected):
    result = ToolResult(tool_call_id="send", tool_name="send", details=details)
    result = Message.tool_result(result) if durable else result
    if durable:
        assert result.provider_payload["details"] == details
    assert current_peer_communication_succeeded(result, "send") is expected


def test_missing_untyped_failed_wrong_name_and_foreign_call_are_not_current_success():
    assert current_peer_communication_succeeded(None, "send") is None
    assert current_peer_communication_succeeded(
        SimpleNamespace(tool_name="send", tool_call_id="send", is_error=False, details={}), "send") is None
    assert current_peer_communication_succeeded(
        ToolResult(tool_call_id="send", tool_name="send", is_error=True), "send") is False
    assert current_peer_communication_succeeded(
        ToolResult(tool_call_id="send", tool_name="read"), "send") is False
    assert current_peer_communication_succeeded(
        ToolResult(tool_call_id="other", tool_name="send"), "send") is False


def test_raw_duplicate_provider_receipt_cannot_certify_a_new_communication():
    messages = raw_cycle()
    messages[2] = Message.tool_result(ToolResult(tool_call_id="send", tool_name="send",
        details={"duplicate_suppressed": True, "original_receipt": {"text": "old success", "details": {}}}))
    replay = PeerCycleReplay(IDENTITY)
    assert not any(replay.step(message) is not None for message in messages)


def _card():
    card = ToolCard("send", "send", {"message": "ack"})
    card.work_view_identity = IDENTITY
    return card


def test_manual_success_is_unknown_until_a_current_typed_end_is_supplied():
    card = _card()
    assert card.peer_cycle_communication_succeeded is None
    card.mark_done("delivered", {"delivery": {"state": "delivered"}})
    assert card.state == "success"
    assert card.peer_cycle_communication_succeeded is None
    assert peer_cycle_rows_of_blocks([card])[0].successful is False
    card.mark_done("delivered", result=ToolResult(tool_call_id="send", tool_name="send"))
    assert card.peer_cycle_communication_succeeded is True
    assert peer_cycle_rows_of_blocks([card])[0].successful is True


@pytest.mark.parametrize("reset", ["begin", "restore-running", "restore-success-untyped", "done-untyped", "failed"])
def test_reuse_or_unknown_completion_clears_previous_positive_provenance_before_close(reset):
    card = _card()
    card.mark_done("delivered", result=ToolResult(tool_call_id="send", tool_name="send"))
    before = [rows()[0], peer_cycle_rows_of_blocks([card])[0]]
    start = begin_peer_cycle(before, IDENTITY, "quiet")
    assert start is not None
    if reset == "begin":
        card.begin_running("send", {"message": "ack"}, intent=None)
    elif reset == "restore-running":
        card.restore(state="running")
    elif reset == "restore-success-untyped":
        card.restore(state="success", result_text="delivered")
    elif reset == "done-untyped":
        card.mark_done("delivered")
    else:
        card.mark_failed("refused")
    assert card.peer_cycle_communication_succeeded is None
    after = [rows()[0], peer_cycle_rows_of_blocks([card])[0]]
    assert close_peer_cycle(start, after, IDENTITY, "quiet", "no_reply", quiet()) is None


def test_changed_provenance_notifies_existing_view_even_when_success_word_is_unchanged(monkeypatch):
    card = _card()
    card.mark_done("delivered", result=ToolResult(tool_call_id="send", tool_name="send"))
    view = TranscriptView()
    observed = []
    # Source-publication unit proof, NOT styled/mounted geometry evidence.
    monkeypatch.setattr(ToolCard, "parent", property(lambda self: view))
    monkeypatch.setattr(view, "restate_quiet_member", lambda block:
        observed.append((block is card, block.state, block.peer_cycle_communication_succeeded)))
    card.mark_done("delivered", result=ToolResult(tool_call_id="send", tool_name="send",
                                               details={"duplicate_suppressed": True}))
    assert card.state == "success"
    assert observed == [(True, "success", False)]
    assert peer_cycle_rows_of_blocks([card])[0].successful is False
