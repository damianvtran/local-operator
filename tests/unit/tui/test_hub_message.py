"""Drive the real styled OperatorApp projection and existing group tracker."""

import base64
import struct
import zlib
from typing import Any

import pytest

from local_operator.harness.types import CustomMessage, ImageContent, Message
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.image_block import ImageBlock
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import (
    HubMessageBlock,
    QuietGroupBlock,
    QuietGroupTracker,
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
        picture.provider_payload = dict(message.provider_payload or {})
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
async def test_progressive_pictures_own_distinct_source_parts() -> None:
    """R3-3 / Q-R5-1 direct acceptance (review round 4): the receipt body owns
    logical part 0, picture one part 1 and picture two part 2 — across
    progressive adoption passes — a replay re-carrying both adds nothing, and
    labels re-read from the FULL source sequence so progressive adoption
    converges on the full-initial labels (not only the same count)."""
    from local_operator.harness.types import ImageContent

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        base = _marked_steer()

        def with_pictures(count: int) -> Message:
            message = Message.user(base.text, id=base.id)
            message.provider_payload = dict(base.provider_payload or {})
            for _ in range(count):
                message.content.append(ImageContent(data="QUJD", mime_type="image/png"))
            return message

        view = app.query_one(TranscriptView)
        app._project_settled_rows([with_pictures(0)])
        await pilot.pause()
        app._project_settled_rows([with_pictures(1)])
        await pilot.pause()
        app._project_settled_rows([with_pictures(2)])
        await pilot.pause()
        receipt = next(block for block in view.blocks() if isinstance(block, HubMessageBlock))
        assert (receipt.navigation_anchor_id, receipt.navigation_anchor_part) == (base.id, 0)
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        assert [(b.navigation_anchor_id, b.navigation_anchor_part) for b in images] == [
            (base.id, 1),
            (base.id, 2),
        ]
        assert [b._label for b in images] == ["#1", "#2"]

        app._project_settled_rows([with_pictures(2)])
        await pilot.pause()
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        assert [(b.navigation_anchor_id, b.navigation_anchor_part) for b in images] == [
            (base.id, 1),
            (base.id, 2),
        ]
        assert [b._label for b in images] == ["#1", "#2"]


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


@pytest.mark.asyncio
async def test_ordered_same_id_repair_groups_like_the_first_paint_control() -> None:
    """R4-1 / Q-R5-2 direct acceptance: the same final rows reached by a late
    in-place upgrade group EXACTLY like the control painted with the valid hub
    row from the start — one bar, same members in source order, same counts —
    and a following user boundary stays visible with later appends joining a
    new stretch, never the closed bar."""

    def tool(label: str) -> ToolCard:
        card = ToolCard(label, "read", {"path": "synthetic.txt"})
        card.mark_done()
        return card

    async def painted(
        marked_first: bool, trailing_user: bool = False
    ) -> tuple[int, int, list[tuple[str, str]], int, int]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)
            steer = _marked_steer()
            app._append_block(tool("tool-a"))
            if marked_first:
                app._project_settled_rows([steer])
            else:
                plain = Message.user(steer.text, id=steer.id)
                app._project_settled_rows([plain])
            app._append_block(tool("tool-b"))
            app._append_block(tool("tool-c"))
            if trailing_user:
                app._append_block(UserBlock("audit the ingest path", fold_width=0))
            await pilot.pause()
            if not marked_first:
                app._project_settled_rows([steer])
                await pilot.pause()
            view = app.query_one(TranscriptView)
            seats = [
                block
                for block in view.blocks()
                if block.navigation_anchor_id == steer.id and not isinstance(block, QuietGroupBlock)
            ]
            assert seats
            assert all(block.work_view_identity == view.work_identity for block in seats)
            bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
            assert len(bars) == 1
            bar = bars[0]
            members = [
                (str(type(member).__name__), member.navigation_anchor_id) for member in bar._members
            ]
            visible_users = [
                block for block in view.blocks() if isinstance(block, UserBlock) and block.display
            ]
            actions_before = bar.group.actions
            # A later real append joins the actual tail.
            app._append_block(tool("tool-d"))
            await pilot.pause()
            after = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
            actions_after = after[0].group.actions if after else 0
            return (
                bar.group.count,
                actions_before,
                members,
                len(visible_users),
                actions_after,
            )

    control = await painted(marked_first=True)
    repaired = await painted(marked_first=False)
    assert control[:4] == repaired[:4]
    assert control[0] == 1
    assert control[1] == 3
    # The next real append extends the tail in BOTH histories, identically.
    assert control[4] == repaired[4] == 4

    closed = await painted(marked_first=False, trailing_user=True)
    # The user boundary stays visible and the closed bar keeps its arithmetic
    # when a later append lands outside it.
    assert closed[2] and all(name != "UserBlock" for name, _ in closed[2])
    assert closed[3] == 1
    assert closed[4] == 3


def _marked_receipt(receipt_id: str, body: str) -> Message:
    """A hub steer as ``comms.steer`` persists it, under a chosen source id."""
    message = Message.user(
        f"<parent-message>\nsteer preamble\n\n{body}\n</parent-message>", id=receipt_id
    )
    message.provider_payload = {
        "hub_receipt": {
            "direction": "to_child",
            "expects_reply": False,
            "steer": True,
            "body": body,
        }
    }
    return message


def _plain_envelope(receipt_id: str, body: str) -> Message:
    """The same row before its provenance arrives: envelope text, no marker."""
    return Message.user(
        f"<parent-message>\nsteer preamble\n\n{body}\n</parent-message>", id=receipt_id
    )


def _source_order(view: TranscriptView) -> list[tuple[str, str, object]]:
    """The ACTUAL source rows in order (bars are view objects, never rows)."""
    return [
        (
            type(block).__name__,
            str(getattr(block, "navigation_anchor_id", "")),
            getattr(block, "navigation_anchor_part", None),
        )
        for block in view.blocks()
        if not isinstance(block, QuietGroupBlock)
    ]


@pytest.mark.asyncio
async def test_ordered_repair_with_bars_on_both_sides_converges_with_the_control() -> None:
    """R4-1 round-5 acceptance, first half: two existing bars with an unmarked
    row between them must converge on the fresh control's ONE group when that
    row's provenance arrives in place — same members in source order, same
    counts, no stale bars, five distinct leaves, and the reader's open intent
    survives."""

    def rows(marked_bridge: bool) -> list[Message]:
        bridge = (
            _marked_receipt("qa_bridge", "bridging instructions")
            if marked_bridge
            else _plain_envelope("qa_bridge", "bridging instructions")
        )
        return [
            _marked_receipt("qa_p1", "one"),
            _marked_receipt("qa_p2", "two"),
            bridge,
            _marked_receipt("qa_p3", "three"),
            _marked_receipt("qa_p4", "four"),
        ]

    async def painted(repaired: bool) -> dict[str, Any]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)
            view = app.query_one(TranscriptView)
            if repaired:
                # The pre-provenance history, the first bar opened by hand,
                # then ONLY the same-ID repair (no peer prefix replay).
                for message in rows(marked_bridge=False):
                    app._project_settled_rows([message])
                await pilot.pause()
                bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
                assert [bar.group.count for bar in bars] == [2, 2]
                bars[0].toggle_expanded()
                await pilot.pause()
                app._project_settled_rows([rows(marked_bridge=True)[2]])
                await pilot.pause()
            else:
                for message in rows(marked_bridge=True):
                    app._project_settled_rows([message])
                await pilot.pause()
            bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
            report = {
                "counts": [bar.group.count for bar in bars],
                "order": _source_order(view),
                "expanded": [bar._expanded for bar in bars],
                "hubs": len([b for b in view.blocks() if isinstance(b, HubMessageBlock)]),
            }
            app._append_block(ToolCard("tool-a", "read", {"path": "synthetic.txt"}))
            await pilot.pause()
            report["actions"] = [bar.group.actions for bar in bars]
            app._project_settled_rows([_marked_receipt("qa_p7", "seven")])
            await pilot.pause()
            report["counts_after"] = [bar.group.count for bar in bars]
            return report

    control = await painted(repaired=False)
    repaired = await painted(repaired=True)
    assert control["counts"] == [5], control["counts"]
    assert repaired["counts"] == [5], repaired["counts"]
    assert control["order"] == repaired["order"]
    assert [row[1] for row in repaired["order"] if row[0] == "HubMessageBlock"] == [
        "qa_p1",
        "qa_p2",
        "qa_bridge",
        "qa_p3",
        "qa_p4",
    ]
    assert control["hubs"] == repaired["hubs"] == 5
    assert control["actions"] == repaired["actions"] == [1]
    assert control["counts_after"] == repaired["counts_after"] == [6]
    assert repaired["expanded"] == [True]


@pytest.mark.asyncio
async def test_ordered_repair_keeps_the_untouched_live_tail_connected() -> None:
    """R4-1 round-5 acceptance, second half: replanning an earlier closed
    stretch leaves an UNTOUCHED live tail bar connected — the control's
    action1 parity — without crossing the real human boundary and without
    unfreezing the earlier closed group."""

    def tool(label: str) -> ToolCard:
        card = ToolCard(label, "read", {"path": "synthetic.txt"})
        card.mark_done()
        return card

    async def painted(repaired: bool) -> dict[str, Any]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)
            view = app.query_one(TranscriptView)
            bridge = (
                _plain_envelope("qa_bridge", "bridging instructions")
                if repaired
                else _marked_receipt("qa_bridge", "bridging instructions")
            )
            app._append_block(tool("tool-0"))
            app._project_settled_rows([bridge])
            app._append_block(tool("tool-1"))
            app._append_block(tool("tool-2"))
            app._project_settled_rows([Message.user("audit the ingest path", id="qa_u")])
            app._project_settled_rows([_marked_receipt("qa_p5", "five")])
            app._project_settled_rows([_marked_receipt("qa_p6", "six")])
            await pilot.pause()
            if repaired:
                app._project_settled_rows([_marked_receipt("qa_bridge", "bridging instructions")])
                await pilot.pause()
            bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
            before = [(bar.group.actions, bar.group.count, bar.group.open) for bar in bars]
            tail = bars[-1]
            app._append_block(tool("tool-3"))
            await pilot.pause()
            after = [(bar.group.actions, bar.group.count, bar.group.open) for bar in bars]
            tail_actions = tail.group.actions
            user_visible = [
                block for block in view.blocks() if isinstance(block, UserBlock) and block.display
            ]
            return {
                "before": before,
                "after": after,
                "tail_actions": tail_actions,
                "tail_is_same_object": tail is bars[-1],
                "users": len(user_visible),
            }

    control = await painted(repaired=False)
    repaired = await painted(repaired=True)
    assert control["before"] == repaired["before"], (control["before"], repaired["before"])
    assert repaired["before"][-1][2] is True, "the untouched tail bar stays open"
    assert control["after"] == repaired["after"]
    assert repaired["tail_actions"] == 1 or repaired["after"][-1][0] == 1
    assert repaired["tail_is_same_object"] is True
    assert control["users"] == repaired["users"] == 1


@pytest.mark.asyncio
async def test_late_picture_lands_at_its_original_source_seat() -> None:
    """R3-3 placement acceptance: a late same-ID picture lands in the source's
    own segment — never past a later human turn — so the progressive history's
    final source order equals the initial-full one, with a following user and
    final row on hand."""
    from local_operator.harness.types import ImageContent

    def picture_row(with_image: bool) -> Message:
        message = _marked_receipt("qa_h", "look at this")
        if with_image:
            message.content.append(ImageContent(data="QUJD", mime_type="image/png"))
        return message

    async def painted(initial_full: bool) -> list[tuple[str, str, object]]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)
            view = app.query_one(TranscriptView)
            # Identical FINAL source rows in both histories; only the pass
            # structure differs (the progressive half paints h before its
            # picture exists).
            app._project_settled_rows([picture_row(initial_full)])
            await pilot.pause()
            app._project_settled_rows([Message.user("audit the ingest path", id="qa_u")])
            await pilot.pause()
            if not initial_full:
                app._project_settled_rows([picture_row(True)])
                await pilot.pause()
            app._project_settled_rows([Message.assistant("done")])
            await pilot.pause()
            # Assistant rows carry a generated id, so the equivalence is
            # asserted over the SOURCE-OWNED rows; the final row is asserted
            # separately.
            rows = _source_order(view)
            assert rows[-1][0] == "AssistantBlock"
            return [row for row in rows if row[0] != "AssistantBlock"]

    control = await painted(initial_full=True)
    repaired = await painted(initial_full=False)
    assert control == repaired, (control, repaired)
    assert [(row[0], row[1]) for row in control[:3]] == [
        ("HubMessageBlock", "qa_h"),
        ("ImageBlock", "qa_h"),
        ("UserBlock", "qa_u"),
    ]


@pytest.mark.asyncio
async def test_omitted_pictures_keep_their_citation_labels() -> None:
    """Q-R6-3 acceptance: a metadata patch that OMITS the pictures is not an
    instruction to forget them — both widgets, their source parts and their
    citation labels stay exactly as the authoritative source left them."""
    from local_operator.harness.types import ImageContent

    def picture_row(count: int) -> Message:
        message = _marked_receipt("qa_late", "look at this")
        for _ in range(count):
            message.content.append(ImageContent(data="QUJD", mime_type="image/png"))
        return message

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        view = app.query_one(TranscriptView)
        for count in (0, 1, 2, 2):
            app._project_settled_rows([picture_row(count)])
            await pilot.pause()
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        assert [(b.navigation_anchor_part, b._label) for b in images] == [(1, "#1"), (2, "#2")]

        # The omission: same id, pictures absent from the row's content.
        app._project_settled_rows([_marked_receipt("qa_late", "look at this")])
        await pilot.pause()
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        assert len(images) == 2
        assert [(b.navigation_anchor_part, b._label) for b in images] == [(1, "#1"), (2, "#2")]
        assert len([b for b in view.blocks() if isinstance(b, HubMessageBlock)]) == 1


def _healthy_png() -> str:
    """A real 2x2 RGB PNG as ``ImageContent`` carries it, base64-encoded.

    Built here rather than imported: the acceptance cases below exist to prove
    that a picture with genuinely decodable bytes is hidden with the receipt
    that owns it, so a stand-in that decodes would weaken the case (review
    round 6, R6-1).
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(row) + bytes(row) for row in ((62, 122, 214), (214, 69, 65)))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


@pytest.mark.asyncio
async def test_seated_picture_joins_its_owner_group_membership() -> None:
    """R6-1 / Q-R7-1 direct acceptance: a picture seated at its source part is a
    MEMBER of its owner's bar, so a collapsed bar hides it and reopening shows
    it again — for the full-initial row and for a late same-ID adoption alike,
    with a real user boundary, a final row and an untouched tail bar on hand."""

    def receipt(receipt_id: str, *, picture: bool = False) -> Message:
        message = _marked_receipt(receipt_id, "picture body")
        if picture:
            message.content.append(ImageContent(data=_healthy_png(), mime_type="image/png"))
        return message

    async def painted(late: bool) -> dict[str, Any]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)
            view = app.query_one(TranscriptView)
            app._project_settled_rows([receipt("qa_h1", picture=not late)])
            await pilot.pause()
            app._project_settled_rows([receipt("qa_h2")])
            await pilot.pause()
            if late:
                app._project_settled_rows([receipt("qa_h1", picture=True)])
                await pilot.pause()
            source_parts = [
                block
                for block in view.blocks()
                if block.navigation_anchor_id == "qa_h1" and not isinstance(block, QuietGroupBlock)
            ]
            assert len(source_parts) == 2
            assert all(block.work_view_identity == view.work_identity for block in source_parts)
            bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
            assert len(bars) == 1
            bar = bars[0]
            picture = next(block for block in view.blocks() if isinstance(block, ImageBlock))
            states: dict[str, Any] = {
                "member": picture in bar._members,
                "collapsed": picture.display,
                "count": bar.group.count,
                "order": _source_order(view),
            }
            bar.toggle_expanded()
            await pilot.pause()
            states["open"] = picture.display
            bar.toggle_expanded()
            await pilot.pause()
            states["reclosed"] = picture.display
            # A real human boundary, a final row and an untouched tail pair:
            # no group may form beyond the source bounds.
            app._project_settled_rows([Message.user("audit the ingest path", id="qa_u")])
            await pilot.pause()
            app._project_settled_rows([receipt("qa_t1"), receipt("qa_t2")])
            await pilot.pause()
            app._project_settled_rows([Message.assistant("done")])
            await pilot.pause()
            states["bars"] = [
                block.group.count for block in view.blocks() if isinstance(block, QuietGroupBlock)
            ]
            states["open_users"] = len(
                [block for block in view.blocks() if isinstance(block, UserBlock) and block.display]
            )
            return states

    for late in (False, True):
        states = await painted(late)
        assert states["member"] is True, states
        assert states["collapsed"] is False, states
        assert states["open"] is True and states["reclosed"] is False, states
        assert states["count"] == 2, states
        assert states["bars"] == [2, 2], states
        assert states["open_users"] == 1, states


@pytest.mark.asyncio
async def test_contiguous_owned_pictures_all_join_their_owner_run() -> None:
    """R6-1 / Q-R8-1 plural acceptance: every picture under ONE receipt joins
    that receipt's bar (parts 1..n continuing the body's part 0), the full-
    initial and the late-adopted histories reach the SAME fold shape, and the
    visibility of each picture follows the bar for both."""

    def receipt_pics(receipt_id: str, count: int) -> Message:
        message = _marked_receipt(receipt_id, "picture body")
        for _ in range(count):
            message.content.append(ImageContent(data=_healthy_png(), mime_type="image/png"))
        return message

    def snapshot(view: TranscriptView) -> dict[str, Any]:
        bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
        bar = bars[0]
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        return {
            "bars": [block.group.count for block in bars],
            "row_ids": list(bar.group.row_ids),
            "parts": [block.navigation_anchor_part for block in images],
            "labels": [block._label for block in images],
            "members": [block in bar._members for block in images],
        }

    async def painted(initial: int, later: list[int]) -> dict[str, Any]:
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await _settle_for_session(pilot, app)
            view = app.query_one(TranscriptView)
            app._project_settled_rows([receipt_pics("qa_h1", initial)])
            await pilot.pause()
            app._project_settled_rows([receipt_pics("qa_h2", 0)])
            await pilot.pause()
            for count in later:
                app._project_settled_rows([receipt_pics("qa_h1", count)])
                await pilot.pause()
            states = snapshot(view)
            bar = next(block for block in view.blocks() if isinstance(block, QuietGroupBlock))
            images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
            bar.toggle_expanded()
            await pilot.pause()
            states["open"] = [block.display for block in images]
            bar.toggle_expanded()
            await pilot.pause()
            states["reclosed"] = [block.display for block in images]
            states["collapsed"] = None
            images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
            # Re-read the collapsed visibility last, after the toggles.
            states["collapsed"] = [block.display for block in images]
            return states

    initial = await painted(3, [])
    late = await painted(1, [2, 3, 3, 0])
    assert (
        initial["row_ids"]
        == late["row_ids"]
        == [
            "qa_h1:0",
            "qa_h1:1",
            "qa_h1:2",
            "qa_h1:3",
            "qa_h2:0",
        ]
    )
    assert initial["parts"] == late["parts"] == [1, 2, 3]
    assert initial["labels"] == late["labels"] == ["#1", "#2", "#3"]
    assert initial["members"] == late["members"] == [True, True, True]
    assert initial["bars"] == late["bars"] == [2]
    for states in (initial, late):
        assert states["collapsed"] == [False, False, False], states
        assert states["open"] == [True, True, True], states
        assert states["reclosed"] == [False, False, False], states


@pytest.mark.asyncio
async def test_a_human_prefix_picture_stays_visible_and_the_receipt_run_folds() -> None:
    """R8-1 / Q-R9-A1 real-app acceptance: a human row carrying its OWN
    picture is a boundary — the picture stays visible and never seeds the
    receipt chain — and the two-receipt run behind it still folds; only the
    receipt's own picture hides with its bar."""
    human = Message.user("audit the ingest path", id="qa_human")
    human.content.append(ImageContent(data=_healthy_png(), mime_type="image/png"))
    receipt = _marked_receipt("qa_h1", "picture body")
    receipt.content.append(ImageContent(data=_healthy_png(), mime_type="image/png"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        view = app.query_one(TranscriptView)
        app._project_settled_rows([human])
        await pilot.pause()
        app._project_settled_rows([receipt])
        await pilot.pause()
        app._project_settled_rows([_marked_receipt("qa_h2", "picture body")])
        await pilot.pause()

        bars = [block for block in view.blocks() if isinstance(block, QuietGroupBlock)]
        assert len(bars) == 1 and bars[0].group.count == 2
        assert list(bars[0].group.row_ids) == ["qa_h1:0", "qa_h1:1", "qa_h2:0"]
        images = [block for block in view.blocks() if isinstance(block, ImageBlock)]
        human_pic = next(block for block in images if block.navigation_anchor_id == "qa_human")
        receipt_pic = next(block for block in images if block.navigation_anchor_id == "qa_h1")
        assert human_pic.display is True, "the human's own picture stays visible"
        assert human_pic not in bars[0]._members
        assert receipt_pic.display is False and receipt_pic in bars[0]._members

        bars[0].toggle_expanded()
        await pilot.pause()
        assert receipt_pic.display is True and human_pic.display is True
        bars[0].toggle_expanded()
        await pilot.pause()
        assert receipt_pic.display is False and human_pic.display is True


@pytest.mark.asyncio
async def test_the_page_fold_refuses_a_fragment_continuing_an_owned_picture_chain() -> None:
    """R8-2 / Q-R10-1 actual page caller: the loaded prefix is a receipt with
    TWO pictures; the page that follows continues the same source run, so the
    page fold must not carve a page-local bar away from it. The control shows
    a REAL top boundary — a human with its own picture — still closing."""
    from local_operator.tui.session_presentation import PreparedReplay

    def receipt_pics(receipt_id: str, count: int) -> Message:
        message = _marked_receipt(receipt_id, "picture body")
        for _ in range(count):
            message.content.append(ImageContent(data=_healthy_png(), mime_type="image/png"))
        return message

    def page() -> list[Any]:
        replay = PreparedReplay()
        replay.prepare(
            [
                _marked_receipt("qa_h2", "picture body"),
                _marked_receipt("qa_h3", "picture body"),
            ],
            fold_width=80,
        )
        return list(replay.blocks)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        view = app.query_one(TranscriptView)
        app._project_settled_rows([receipt_pics("qa_h0", 2)])
        await pilot.pause()
        folded = view.fold_quiet_group_page(page(), len(view.blocks()))
        assert not [
            block for block in folded if isinstance(block, QuietGroupBlock)
        ], "the page continues the on-hand run and must not fold locally"

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        view = app.query_one(TranscriptView)
        human = Message.user("audit the ingest path", id="qa_human")
        human.content.append(ImageContent(data=_healthy_png(), mime_type="image/png"))
        app._project_settled_rows([human])
        await pilot.pause()
        folded = view.fold_quiet_group_page(page(), len(view.blocks()))
        assert (
            len([block for block in folded if isinstance(block, QuietGroupBlock)]) == 1
        ), "a human boundary still closes the page"


@pytest.mark.asyncio
async def test_the_span_head_rule_reads_the_family_above_an_owned_picture() -> None:
    """R8-2 / Q-R10-1 helper audit: the live tracker's span-head rule reads
    the row above the span with the same (family, owner) pair, so an owned
    picture above a span whose chain says interior makes it a fragment
    (refused), while a real boundary keeps folding."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        view = app.query_one(TranscriptView)
        human = UserBlock("audit the ingest path", fold_width=0)
        human.navigation_anchor_id = "qa_human"
        human.navigation_anchor_part = 0
        human_pic = ImageBlock(None, "image/png")
        human_pic.navigation_anchor_id = "qa_human"
        human_pic.navigation_anchor_part = 1
        h0 = _hub_block("qa_h0", 0)
        i1 = _hub_image("qa_h0", 1)
        i2 = _hub_image("qa_h0", 2)
        h1 = _hub_block("qa_h1", 0)
        # The helpers read the view's block list in order; these are the real
        # widgets, staged the way the projection would have appended them.
        view._blocks.extend([human, human_pic, h0, i1, i2, h1])
        tracker = QuietGroupTracker(view)
        # The span starting at h1 has i2 directly above it: an owned picture
        # whose chain derives interior — not a boundary, so refuse.
        assert tracker._span_head_bounded(5) is False
        # The span starting at human_pic index 1 has the human above it: a
        # real boundary, so fold.
        assert tracker._span_head_bounded(1) is True
        view._blocks.clear()


def _hub_block(owner: str, part: int) -> HubMessageBlock:
    block = HubMessageBlock(
        "picture body",
        {"direction": "to_child", "body": "picture body", "steer": True},
    )
    block.navigation_anchor_id = owner
    block.navigation_anchor_part = part
    return block


def _hub_image(owner: str, part: int) -> ImageBlock:
    block = ImageBlock(None, "image/png")
    block.navigation_anchor_id = owner
    block.navigation_anchor_part = part
    return block
