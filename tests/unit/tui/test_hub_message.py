"""Drive the real styled OperatorApp projection and existing group tracker."""

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    HubMessageBlock,
    QuietGroupBlock,
    TranscriptView,
)

from .test_app_pilot import FakeSession, _factory
from .test_peer_message import _settle_for_session


def _message() -> CustomMessage:
    return CustomMessage(
        id="hub-synthetic-1",
        custom_type="hub_message",
        attribution="user",
        details={
            "direction": "to_child",
            "body": "  first paragraph\n\nsecond paragraph  ",
            "text": "model envelope",
            "expects_reply": True,
        },
    )


@pytest.mark.asyncio
async def test_hub_materialized_history_projects_once_on_live_and_replay() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        message = _message()
        app._project_settled_rows([message])
        await pilot.pause()
        app._project_settled_rows([message])
        await pilot.pause()
        blocks = app.query_one(TranscriptView).blocks()
        receipts = [block for block in blocks if isinstance(block, HubMessageBlock)]
        assert len(receipts) == 1
        assert receipts[0].navigation_anchor_id == message.id
        assert receipts[0].text() == message.details["body"]
        assert "Reply requested" in receipts[0]._summary()[0]
        assert "to_child" in receipts[0]._header()


def _marked_steer() -> Message:
    """A hub steer as ``comms.steer`` persists it: the model envelope as the
    row text, the typed facts in the namespaced provider marker."""
    message = Message.user(
        "<parent-message>\nThis changes your instructions.\n\n"
        "  first paragraph\n\nsecond paragraph  \n</parent-message>",
        id="hub-marked-steer-1",
    )
    message.provider_payload = {
        "hub_receipt": {
            "direction": "to_child",
            "expects_reply": False,
            "steer": True,
            "body": "  first paragraph\n\nsecond paragraph  ",
        }
    }
    return message


@pytest.mark.asyncio
async def test_marked_user_role_hub_row_projects_as_receipt_not_the_users_words() -> None:
    """The marker wins before the human-row classification: the receipt paints
    the authored body (never the XML envelope), and a same-id restate cannot
    double it — the live/reconnect contract the custom path already keeps."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        message = _marked_steer()
        app._project_settled_rows([message])
        await pilot.pause()
        app._project_settled_rows([message])
        await pilot.pause()
        blocks = app.query_one(TranscriptView).blocks()
        receipts = [block for block in blocks if isinstance(block, HubMessageBlock)]
        assert len(receipts) == 1
        assert receipts[0].navigation_anchor_id == message.id
        assert receipts[0].text() == message.provider_payload["hub_receipt"]["body"]
        assert "to_child" in receipts[0]._header()
        rendered = " ".join(block.text() for block in blocks)
        assert "<parent-message>" not in rendered


@pytest.mark.asyncio
async def test_third_tool_after_hub_forms_existing_mixed_group_and_keeps_expansion() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        for index in range(2):
            app._append_block(ToolCard(f"tool-{index}", "read", {"path": "synthetic.txt"}))
        app._project_settled_rows([_message()])
        await pilot.pause()
        view = app.query_one(TranscriptView)
        assert not any(isinstance(block, QuietGroupBlock) for block in view.blocks())
        app._append_block(ToolCard("tool-2", "read", {"path": "synthetic.txt"}))
        await pilot.pause()
        bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
        assert len(bars) == 1
        bar = bars[0]
        assert bar.group.count == 1
        assert bar.group.actions == 3
        assert "3 actions" in bar._summary_text()
        assert "reply requested" in bar._summary_text()
        bar.toggle_expanded()
        await pilot.pause()
        assert all(block.display for block in bar._members)
        app._project_settled_rows([_message()])
        await pilot.pause()
        assert bar._expanded
        assert len([block for block in view.blocks() if isinstance(block, HubMessageBlock)]) == 1
        richer = _message()
        richer.details["label"] = "manager-corrected"
        richer.details["expects_reply"] = False
        app._project_settled_rows([richer])
        await pilot.pause()
        assert bar._expanded
        assert bar.group.actions == 3
        assert bar.group.reply_requested == 0
        assert len([block for block in view.blocks() if isinstance(block, HubMessageBlock)]) == 1
        assert (
            "manager-corrected"
            in next(
                block for block in view.blocks() if isinstance(block, HubMessageBlock)
            )._header()
        )
