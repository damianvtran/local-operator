"""Drive the real styled OperatorApp projection and existing group tracker."""

import pytest

from local_operator.harness.types import CustomMessage, Message
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.image_block import ImageBlock
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    HubMessageBlock,
    QuietGroupBlock,
    TranscriptView,
    UserBlock,
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
        marker_body = (message.provider_payload or {})["hub_receipt"]["body"]
        assert receipts[0].text() == marker_body
        assert "to_child" in receipts[0]._header()
        parts: list[str] = []
        for block in blocks:
            text_method = getattr(block, "text", None)
            if callable(text_method):
                parts.append(str(text_method()))
        rendered = " ".join(parts)
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


@pytest.mark.asyncio
async def test_same_id_richer_replay_replaces_the_human_seat_in_place() -> None:
    """R3-2 (review round 3): a row painted as the operator's words on an
    earlier pass and upgraded by the marker on a later one keeps ONE row —
    the receipt takes the human seat at its own position instead of mounting
    a second block beside it."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        plain = Message.user("model envelope", id="upgrade-seat-1")
        app._project_settled_rows([plain])
        await pilot.pause()
        view = app.query_one(TranscriptView)
        seats = [index for index, block in enumerate(view.blocks()) if isinstance(block, UserBlock)]
        assert len(seats) == 1

        marked = Message.user("model envelope", id="upgrade-seat-1")
        marked.provider_payload = {
            "hub_receipt": {
                "direction": "to_child",
                "expects_reply": False,
                "steer": True,
                "body": "Focus on retries",
            }
        }
        app._project_settled_rows([marked])
        await pilot.pause()
        blocks = view.blocks()
        receipts = [block for block in blocks if isinstance(block, HubMessageBlock)]
        assert len(receipts) == 1
        assert not [block for block in blocks if isinstance(block, UserBlock)]
        assert receipts[0].navigation_anchor_id == "upgrade-seat-1"
        assert blocks.index(receipts[0]) == seats[0]
        assert "model envelope" not in " ".join(
            str(getattr(block, "text", "") or "") for block in blocks
        )


@pytest.mark.asyncio
async def test_marked_row_adopts_a_picture_that_arrives_on_a_later_same_id_pass() -> None:
    """Direct acceptance: a marked row painted WITHOUT a picture, then
    replayed same-ID WITH authoritative ImageContent, mounts that first
    picture (adoption by anchor count) — and one more replay re-carrying it
    adds no second copy."""
    from local_operator.harness.types import ImageContent

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        message = _marked_steer()
        app._project_settled_rows([message])
        await pilot.pause()
        view = app.query_one(TranscriptView)
        assert not [block for block in view.blocks() if isinstance(block, ImageBlock)]

        picture = Message.user(message.text, id=message.id)
        picture.provider_payload = dict(message.provider_payload)
        picture.content.append(ImageContent(data="QUJD", mime_type="image/png"))
        app._project_settled_rows([picture])
        await pilot.pause()
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        assert len(images) == 1
        assert images[0].navigation_anchor_id == message.id

        app._project_settled_rows([picture])
        await pilot.pause()
        assert len([block for block in view.blocks() if isinstance(block, ImageBlock)]) == 1


@pytest.mark.asyncio
async def test_a_stamped_but_invalid_marker_keeps_the_row_human() -> None:
    """R3-1 acceptance: an invalid marker with wrapper-looking text renders
    the operator's own row — never a receipt, never a body guess."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        message = Message.user(
            "<parent-message>\nFocus on retries\n</parent-message>", id="stamped-invalid"
        )
        message.provider_payload = {
            "hub_receipt": {"direction": "sideways", "body": "Focus on retries"}
        }
        app._project_settled_rows([message])
        await pilot.pause()
        blocks = app.query_one(TranscriptView).blocks()
        assert not [block for block in blocks if isinstance(block, HubMessageBlock)]
        assert len([block for block in blocks if isinstance(block, UserBlock)]) == 1


@pytest.mark.asyncio
async def test_marked_image_row_keeps_its_picture_and_never_doubles() -> None:
    """R3-3 (review round 3): the marker reclassifies the ROW, never its
    attachments — a marked row carrying an image keeps its ImageBlock beside
    the receipt, and a replay of the same row appends no second copy."""
    from local_operator.harness.types import ImageContent

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        message = _marked_steer()
        message.content.append(ImageContent(data="AAAA", mime_type="image/png"))
        app._project_settled_rows([message])
        await pilot.pause()
        app._project_settled_rows([message])
        await pilot.pause()
        blocks = app.query_one(TranscriptView).blocks()
        assert len([block for block in blocks if isinstance(block, HubMessageBlock)]) == 1
        assert len([block for block in blocks if isinstance(block, ImageBlock)]) == 1
