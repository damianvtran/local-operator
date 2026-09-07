"""TUI rendering of inbound peer messages (`lop send`).

Two delivery paths, mirroring the wake receipt: a LIVE delivery paints a
``PeerMessageBlock`` the instant the event lands, and a resumed conversation
REPLAYS a persisted ``peer_message`` custom row as the same block — without
double-painting one already shown live this session.

The block itself is a ledger CARD (``ExpandableActionBlock``), not the
rule-and-header block it started as. Cross-session traffic became routine
enough that a receipt printing its whole body cost 15 rows for one realistic
release-window announcement, and two of those pushed the conversation off
screen. ``TestPeerCard`` onward pins the card's contract: one row closed, the
sender named on it, full identity and full body on expand — and the sender
fields, which cross the wire from another process, still hardened exactly as
they were.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.cells import cell_len

from local_operator.harness.types import PeerMessageDeliveredEvent
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.tui.app import OperatorApp
from local_operator.tui.glyphs import tool_icon
from local_operator.tui.widgets.tool_card import (
    COLLAPSE_HINT,
    EXPAND_HINT,
    OUTPUT_INDENT,
)
from local_operator.tui.widgets.transcript import (
    ExpandableActionBlock,
    NoticeBlock,
    PeerMessageBlock,
    TranscriptView,
    wrap_cells,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The shape the operator reported: a multi-paragraph release announcement.
#: Long enough that the old block's row cost is what these tests are about.
LONG_BODY = (
    "I am claiming the current release window as owner (pid 48213).\n\n"
    "Window contents so far: #744, #751, and #747 once its QA round lands. If your PR "
    "merged since v0.50.3 and is not on that list, send me the PR number, the merge "
    "SHA and your Release: line and I will fold it in.\n\n"
    "I am arguing patch for the whole window: none of the three clears the "
    "step-function bar on its own."
)
LONG_SENDER = {
    "pid": 48213,
    "conversation_name": "lo-release-window",
    "model_label": "anthropic/claude-opus-5",
}


def _peer_blocks(app) -> list[PeerMessageBlock]:
    return [b for b in app.query_one(TranscriptView).blocks() if isinstance(b, PeerMessageBlock)]


async def _settle_for_session(pilot, app) -> None:
    """Wait until boot has resolved the session and subscribed its event bridge.

    The TUI responsiveness work makes boot deliberately more asynchronous. A
    live event emitted before ``app._session`` is set has no session handler to
    receive it, so no amount of post-emit settling can recover that lost event.
    Tests that inject host events must first establish the same precondition the
    real registrant has: the session is live and addressable.
    """
    for _ in range(200):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("session did not finish booting")


async def _settle_for_peer_block(pilot, app, *, want: int = 1) -> None:
    """Pump the event loop until ``want`` peer blocks are mounted.

    A single ``pilot.pause()`` after an emit is racy here: delivery is a
    two-hop path (session.emit -> PeerMessageDeliveredEvent -> a Textual
    message -> block mount), and one frame is not reliably enough for both
    hops to complete before the assert (measured >40% flake in isolation,
    C2). Poll for the block to appear instead, bounded so a genuine failure
    still terminates — the same settle-loop discipline the app-pilot tests
    use for their own two-hop posts. The bound is generous (200, matching
    ``test_app_pilot``) because a ``pause()`` only advances one frame and
    under whole-suite CPU contention the two hops can need far more than a
    small fixed count; the loop exits the moment the block mounts, so the
    extra headroom costs nothing on a fast machine.
    """
    for _ in range(200):
        await pilot.pause()
        if len(_peer_blocks(app)) >= want:
            return


@pytest.mark.asyncio
async def test_live_peer_delivery_paints_a_cross_session_block() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        session.emit(
            PeerMessageDeliveredEvent(
                body="gates are green",
                sender={
                    "pid": 4242,
                    "conversation_name": "peer-send design",
                    "model_label": "anthropic/claude-opus-4",
                },
                message_id="peer-1",
            )
        )
        await _settle_for_peer_block(pilot, app)
        blocks = _peer_blocks(app)
        assert len(blocks) == 1
        # The block reads as inbound cross-session: the sender label names who
        # reached in, and the body is present.
        header = blocks[0]._header()
        assert "peer-send design" in header
        assert "pid 4242" in header
        assert blocks[0].text() == "gates are green"
        # The live receipt id was recorded so a replay won't double-paint it.
        assert "peer-1" in app._live_peer_receipts


@pytest.mark.asyncio
async def test_resume_replays_peer_message_without_double_paint() -> None:
    """A persisted peer row replays as a block; but one already painted live
    this session (its id in _live_peer_receipts) is skipped on replay."""
    session = FakeSession()
    session._history = [
        SimpleNamespace(
            role=None,
            custom_type=PEER_MESSAGE_MESSAGE_TYPE,
            id="peer-1",
            text="",
            tool_calls=None,
            content=[],
            details={
                "body": "replayed note",
                "sender": {"pid": 9, "conversation_name": "other"},
            },
        )
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        await _settle_for_peer_block(pilot, app)
        # Boot replayed the history: exactly one peer block from the persisted row.
        assert len(_peer_blocks(app)) == 1
        assert _peer_blocks(app)[0].text() == "replayed note"


@pytest.mark.asyncio
async def test_busy_steer_paints_one_peer_block_and_no_user_block(tmp_path) -> None:
    """A `send now=True` landing mid-turn paints ONE PeerMessageBlock, and
    nothing else for the same message.

    Regression: the busy-steer path used to queue a plain user Message built
    from the peer body, so the steering drain announced it with a user
    ``MessageStartEvent`` — which the app, finding no echo it registered,
    painted as a second copy in a ``UserBlock`` right under the
    ``PeerMessageBlock`` the live receipt had already put up. A REAL Session
    in the REAL app is the only host that exercises both hops (the receipt
    and the drain's announcement), so the FakeSession is no use here.
    """
    import asyncio

    from local_operator.harness.types import (
        AgentTool,
        StreamEndEvent,
        StreamTextDelta,
        StreamToolCallDelta,
        TextContent,
        ToolResult,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.tui.widgets.transcript import UserBlock
    from tests.unit.session.test_session import MODEL, ScriptedStream

    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    async def blocking_execute(tool_call_id, args, signal, on_update, context):
        tool_started.set()
        await release_tool.wait()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name="block", content=[TextContent(text="done")]
        )

    tool = AgentTool(
        name="block", parameters={"type": "object", "properties": {}}, execute=blocking_execute
    )
    stream = ScriptedStream(
        [
            [
                StreamToolCallDelta(index=0, id="c1", name="block", argument_delta="{}"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="ack"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    session = Session(
        model=MODEL,
        stream_fn=stream,
        tools=[tool],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
    )

    async def factory():
        return session

    app = OperatorApp(factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        # The gate would otherwise park the scripted tool call on a prompt the
        # test never answers (see AGENTS.md on the capture scripts).
        app._set_approve_all(True)
        prompt_task = asyncio.ensure_future(session.prompt("long task"))
        for _ in range(200):
            await pilot.pause()
            if tool_started.is_set():
                break
        assert tool_started.is_set(), "the scripted tool never started"

        await session.receive_peer_message(
            "redirect now", mode="steer", sender={"pid": 3, "conversation_name": "peer"}
        )
        await _settle_for_peer_block(pilot, app)
        release_tool.set()
        await prompt_task
        # Let the drain's events cross into the app before counting; the
        # spurious UserBlock (if any) is painted from that announcement.
        for _ in range(20):
            await pilot.pause()

        peers = _peer_blocks(app)
        assert len(peers) == 1
        assert peers[0].text() == "redirect now"
        user_bodies = [
            b.text() for b in app.query_one(TranscriptView).blocks() if isinstance(b, UserBlock)
        ]
        # Only the prompt that opened the turn is a user row.
        assert user_bodies == ["long task"], user_bodies
    await session.dispose()


@pytest.mark.asyncio
async def test_live_receipt_suppresses_its_replay() -> None:
    session = FakeSession()
    session._history = [
        SimpleNamespace(
            role=None,
            custom_type=PEER_MESSAGE_MESSAGE_TYPE,
            id="peer-1",
            text="",
            tool_calls=None,
            content=[],
            details={"body": "dup note", "sender": {}},
        )
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _settle_for_session(pilot, app)
        # Pretend this delivery was already painted live before the replay.
        app._live_peer_receipts.add("peer-1")
        app._render_resumed_history(session)
        await _settle_for_peer_block(pilot, app)
        # The replay path skipped the already-live id, so no NEW duplicate was
        # mounted for it beyond the one from boot.
        bodies = [b.text() for b in _peer_blocks(app)]
        assert bodies.count("dup note") == 1


class TestPeerCard:
    """The receipt is a ledger card, not a rule-and-header block.

    Why it changed: the old shape painted a full-height ``↔`` gutter, a
    wrapped sender header and the WHOLE body at body weight — 15 rows at 100
    columns for one realistic release-window announcement, and two of them
    pushed the conversation off screen. These pin the card's contract: one row
    closed, the sender named on it, full identity and full body on expand.
    """

    def test_the_collapsed_row_is_exactly_one_row(self) -> None:
        """The whole point of the change. A multi-paragraph note costs ONE row
        until the reader asks for it."""
        block = PeerMessageBlock(LONG_BODY, LONG_SENDER)
        assert block._row_count == 1
        assert len(block._build_row(100).plain.splitlines()) == 1
        assert block.spans_multiple_rows() is False

    def test_the_collapsed_row_names_the_sender_before_the_snippet(self) -> None:
        """Identity leads: the row builder truncates the composed summary from
        the RIGHT as one string, so free text is what sheds. `_send_summary`
        in tool_card.py learned this the expensive way — with the identity
        after the free text, rows that differ in who sent them paint
        byte-identical at ordinary widths."""
        row = PeerMessageBlock(LONG_BODY, LONG_SENDER)._build_row(100).plain
        assert "peer" in row  # the name column
        assert '"lo-release-window"' in row
        assert row.index('"lo-release-window"') < row.index("I am claiming")

    def test_the_snippet_sheds_before_the_sender_at_narrow_widths(self) -> None:
        """Monotonic degradation, in the order that keeps the row useful: the
        message preview gives way, the address does not."""
        block = PeerMessageBlock(LONG_BODY, LONG_SENDER)
        wide = block._build_row(120).plain
        narrow = block._build_row(56).plain
        assert "lo-release-window" in narrow
        assert len(narrow.splitlines()) == 1
        assert cell_len(narrow) < cell_len(wide)

    def test_the_snippet_carries_no_cap_of_its_own(self) -> None:
        """A second arbitrary bound only left the line empty at wide widths
        while protecting nothing — the row's own truncate_cells is the budget
        (the same finding `_send_summary` records)."""
        block = PeerMessageBlock(LONG_BODY, LONG_SENDER)
        rows = [block._build_row(width).plain for width in (80, 100, 140, 200)]
        lengths = [cell_len(row.rstrip()) for row in rows]
        assert lengths == sorted(lengths), lengths
        # At 200 cells the row is genuinely using the room it was given.
        assert lengths[-1] > lengths[0] + 40

    def test_a_multiline_body_becomes_a_one_line_snippet(self) -> None:
        """An authored newline inside the snippet would be measured into a
        word's width and then printed literally mid-row."""
        row = PeerMessageBlock("first line\n\nsecond line", {"pid": 3})._build_row(100).plain
        assert "\n" not in row
        assert "first line second line" in row

    def test_expanding_reveals_the_full_sender_identity_and_the_full_body(self) -> None:
        """The operator's ask: 'details of the recipient should be shown on
        expand'. The identity line carries what the old always-on header did —
        name, pid and model — and the body is complete, not a preview."""
        block = PeerMessageBlock(LONG_BODY, LONG_SENDER)
        assert block.can_expand() is True
        assert block.toggle_expanded() is True
        rendered = block._build_content(100).plain
        assert 'peer message from "lo-release-window"' in rendered
        assert "pid 48213" in rendered
        assert "anthropic/claude-opus-5" in rendered
        # Every paragraph of the body, not just the snippet's first sentence.
        for paragraph in LONG_BODY.split("\n\n"):
            first_words = " ".join(paragraph.split()[:5])
            assert first_words in " ".join(rendered.split()), first_words
        assert block.spans_multiple_rows() is True

    def test_activate_toggles_like_the_rest_of_the_ledger(self) -> None:
        block = PeerMessageBlock("note", {"pid": 3, "conversation_name": "peer"})
        assert block.expanded is False
        assert block.activate() is True
        assert block.expanded is True
        assert block.has_class(PeerMessageBlock.EXPANDED_CLASS)
        block.activate()
        assert block.expanded is False
        assert not block.has_class(PeerMessageBlock.EXPANDED_CLASS)

    def test_the_hint_appears_only_when_pointed_at_or_focused(self) -> None:
        """Same affordance contract as the tool and wake rows."""
        block = PeerMessageBlock("note", {"pid": 3, "conversation_name": "peer"})
        assert EXPAND_HINT not in block._build_row(100).plain
        block._set_hovered(True)
        assert EXPAND_HINT in block._build_row(100).plain
        block._set_hovered(False)
        block._set_focused(True)
        assert EXPAND_HINT in block._build_row(100).plain
        block.toggle_expanded()
        assert COLLAPSE_HINT in block._build_row(100).plain

    def test_retheme_is_the_shared_finalized_re_entry_point(self) -> None:
        """A theme switch repaints settled ledger rows through the ONE hook on
        the base, not a per-subclass override."""
        assert PeerMessageBlock.retheme is ExpandableActionBlock.retheme

    def test_a_height_only_resize_rebuilds_nothing(self) -> None:
        """WakeBlock's `_built_width` guard: an expansion raises a Resize back
        into on_resize, and rebuilding an identical row was a third of all
        builds on a measured replay."""
        block = PeerMessageBlock(LONG_BODY, LONG_SENDER)
        width = block._built_width
        builds = {"n": 0}
        original = PeerMessageBlock._refresh_row
        block._refresh_row = (  # type: ignore[assignment]
            lambda: (builds.__setitem__("n", builds["n"] + 1), original(block))[1]
        )
        block.on_resize(SimpleNamespace(size=SimpleNamespace(width=width)))
        assert builds["n"] == 0
        block.on_resize(SimpleNamespace(size=SimpleNamespace(width=width - 20)))
        assert builds["n"] == 1


class TestPeerCardDegradation:
    """A receipt that cannot name its sender must still say something a reader
    can act on. This is the principle the old header ladder pinned, carried
    onto the collapsed row where it now matters most: at one row there is no
    second line to recover on."""

    def test_the_row_falls_back_to_cwd_then_session_id_not_a_bare_pid(self) -> None:
        by_cwd = PeerMessageBlock("body", {"pid": 1, "cwd": "/Users/x/minerva-core"})
        assert "minerva-core" in by_cwd._build_row(100).plain

        by_id = PeerMessageBlock("body", {"pid": 1, "session_id": "01JQ9ZK4W7X2M8N3PVQ6TYRB5H"})
        row = by_id._build_row(100).plain
        assert "01JQ9ZK4" in row
        # Only a short prefix: a full ULID is 26 cells of entropy that would
        # push the message preview off the row.
        assert "01JQ9ZK4W7X2M8N3PVQ6TYRB5H" not in row

        # A real name still wins over both fallbacks, and is QUOTED — the
        # fallbacks are the app guessing and must not look like a chosen title.
        named = PeerMessageBlock(
            "body",
            {"pid": 1, "cwd": "/Users/x/minerva-core", "conversation_name": "release cutter"},
        )
        assert '"release cutter"' in named._build_row(100).plain
        assert '"minerva-core' not in by_cwd._build_row(100).plain

    def test_a_sender_with_nothing_but_a_pid_still_identifies_itself(self) -> None:
        """The last resort. A pid is a poor address but it is an address; a row
        that named nothing at all would be unactionable."""
        assert "pid 1" in PeerMessageBlock("body", {"pid": 1})._build_row(100).plain

    def test_a_sender_with_no_fields_at_all_still_says_what_it_is(self) -> None:
        row = PeerMessageBlock("body", {})._build_row(100).plain
        assert "peer" in row
        assert "another session" in row


class TestPeerCardHardening:
    """The sender fields cross the wire from another process, so they are the
    least trusted strings this widget paints. The card changed shape; the
    hardening did not."""

    def test_a_control_character_in_a_sender_name_cannot_break_the_row(self) -> None:
        """A newline split the label into rows the block never counted — its
        height is PINNED to that count, so the extra row painted outside the
        reserved space — and an escape sequence would re-ink the transcript
        from inside a label."""
        block = PeerMessageBlock(
            "body",
            {"pid": 7, "conversation_name": "line1\nline2\nline3", "model_label": "m\x1b[31m"},
        )
        row = block._build_row(100).plain
        assert "\n" not in row
        assert "\x1b" not in row
        assert block._row_count == 1
        # The identity line in the EXPANSION is bounded the same way.
        assert "\x1b" not in block._header(100)
        assert "\n" not in block._header(100)

    def test_a_giant_sender_name_cannot_own_the_viewport(self) -> None:
        """Sanitization fixed the SHAPE of an advisory field; this bounds its
        SIZE. Uncapped, a 50,000-character name wrapped to a block hundreds of
        rows tall that pushed the whole conversation off screen.

        The card makes this stronger than the old block could: collapsed, a
        hostile name cannot cost more than the one row every receipt costs."""
        block = PeerMessageBlock("body", {"pid": 7, "conversation_name": "x" * 50_000})
        assert block._row_count == 1
        assert len(block._build_row(100).plain.splitlines()) == 1
        assert len(block._header(100)) < 200

        # Expanded, the identity line is bounded too: a handful of rows, not
        # hundreds, and the card's own count matches what it painted.
        block.toggle_expanded()
        rendered = block._build_content(100).plain
        assert len(rendered.splitlines()) == block._row_count
        assert block._row_count < 10, block._row_count

        # Every advisory field is bounded, not just the name.
        wide = PeerMessageBlock(
            "body",
            {
                "pid": 7,
                "conversation_name": "y" * 9000,
                "model_label": "m" * 9000,
                "cwd": "/" + "d" * 9000,
                "session_id": "s" * 9000,
            },
        )
        assert len(wide._header(100)) < 400
        assert wide._row_count == 1

    def test_a_bidi_override_cannot_scramble_the_pid(self) -> None:
        """Unicode format characters (category Cf) reorder the glyphs AROUND
        them. An unterminated U+202E visibly scrambled the pid — the one field
        a reader uses to address the peer back — so the label misreported the
        address. C0/C1 stripping did not touch these."""
        import unicodedata

        payloads = {
            "rtl-override": "evil\u202ename",
            "rtl-embedding": "evil\u202bname",
            "zwsp": "a\u200bb",
            "zwj": "a\u200db",
            "bom": "\ufeffname",
            "lro": "\u202dname",
            "isolate": "a\u2066b",
        }
        for label, name in payloads.items():
            block = PeerMessageBlock("body", {"pid": 48213, "conversation_name": name})
            row = block._build_row(100).plain
            header = block._header(100)
            assert not [c for c in row if unicodedata.category(c) == "Cf"], label
            assert not [c for c in header if unicodedata.category(c) == "Cf"], label
            # The addressing field survives intact and in order.
            assert "(pid 48213)" in header, label

    def test_a_wider_pane_never_makes_the_identity_line_taller(self) -> None:
        """Wrapping must be monotonic in the right direction. The model label
        used to attach at a fixed column threshold calibrated on a 14-cell
        name; for real 22-57 cell names, crossing it re-attached a ~23-cell
        label that cost more than the columns just gained, so dragging a pane
        from 70 to 80 columns made the card TALLER."""
        names = [
            "release cutter",  # 14
            "minerva-user-dashboard",  # 22
            "01JQ9ZK4W7X2M8N3PVQ6TYRB5H",  # 26 (a session id)
            "minerva-user-dashboard-release-cutter",  # 37
            "minerva-user-dashboard-release-cutter-standby",  # 45
            "minerva-user-dashboard-release-cutter-standby-secondary",  # 57
        ]
        widths = (60, 70, 80, 100, 120)
        for name in names:
            rows = [
                len(
                    wrap_cells(
                        PeerMessageBlock(
                            "body",
                            {
                                "pid": 48213,
                                "conversation_name": name,
                                "model_label": "anthropic/claude-opus-5",
                            },
                        )._header(width),
                        width,
                    )
                )
                for width in widths
            ]
            assert all(
                rows[i] >= rows[i + 1] for i in range(len(rows) - 1)
            ), f"{len(name)}-cell name wraps non-monotonically across {widths}: {rows}"


@pytest.mark.asyncio
async def test_dragging_over_a_peer_message_copies_no_app_chrome() -> None:
    """The card is a ledger row for selection too: the summary and the sender
    identity are the app talking, and only the peer's body reaches the
    clipboard — text the user can paste into what they were quoting into.

    The old block had the same contract over a wrapped header; the card moves
    the boundary to `_chrome_rows`, set by the same build that painted the
    frame so a resize cannot make the two disagree.
    """
    from textual.selection import Selection as ScreenSelection

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(62, 14)) as pilot:
        await _settle_for_session(pilot, app)
        block = PeerMessageBlock("gates are green\n\non the second line", LONG_SENDER)
        app._append_block(block)
        await pilot.pause()
        block.toggle_expanded()
        await pilot.pause()

        assert block.copy_gutter(0) == 2  # the icon field, like ToolCard
        assert block.copy_gutter(1) == OUTPUT_INDENT
        assert block._chrome_rows > 1, "the identity line is chrome too"

        copied = block.get_selection(ScreenSelection(None, None))
        assert copied is not None
        text = copied[0]
        assert "gates are green" in text
        assert "on the second line" in text
        # No fragment of the app's own label may ride along.
        assert "peer message from" not in text
        assert "lo-release-window" not in text
        assert "claude-opus" not in text
        assert "pid 48213" not in text
        assert tool_icon("peer") not in text


@pytest.mark.asyncio
async def test_the_card_shares_the_ledger_spine_with_the_tool_rows() -> None:
    """A receipt sitting in a run of actions must be a column, not a second
    layout: same SPACING_KIND, same name column, same air above and below.

    This is why the block is an ExpandableActionBlock rather than a bespoke
    shape — the old `SPACING_KIND = "peer"` gave it its own spacing class and
    its own left edge, so it interrupted the ledger it landed in.
    """
    from local_operator.tui.widgets.tool_card import ToolCard

    assert PeerMessageBlock.SPACING_KIND == ToolCard.SPACING_KIND
    assert PeerMessageBlock.LEDGER_ROW is True
    assert PeerMessageBlock.SPACING_AIRY is True

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await _settle_for_session(pilot, app)
        card = ToolCard("t1", "read", {"path": "a.py"})
        app._append_block(card)
        peer = PeerMessageBlock("note", LONG_SENDER)
        app._append_block(peer)
        await pilot.pause()
        await pilot.pause()

        # Same name column, so the two summaries start at the same cell.
        assert peer._name_col(98) == card._name_col(98)
        assert peer.size.height == 1
        # One blank row between them: the AIRY rule, not a bespoke lead.
        assert peer.region.y - card.region.y == 2


@pytest.mark.asyncio
async def test_real_pointer_hover_and_click_use_the_ledger_contract() -> None:
    """Exercise the actual Textual event path under the production sheet: the
    terminal receives a hand pointer, the real hover event reveals the hint,
    and a click grows and collapses the card without losing the row of air
    below it."""
    from tests.unit.tui.conftest import StyledTranscriptApp

    app = StyledTranscriptApp()
    async with app.run_test(size=(100, 24)) as pilot:
        view = app.query_one(TranscriptView)
        peer = PeerMessageBlock(LONG_BODY, LONG_SENDER)
        below = NoticeBlock("trying another account", "warning")
        view.append_block(peer)
        view.append_block(below)
        await pilot.pause()
        await pilot.pause()

        assert peer.size.height == 1
        assert below.region.y - peer.region.y == 2

        landed = await pilot.hover(peer)
        assert landed, "hover missed the peer card"
        await pilot.pause()
        assert app.screen._pointer_shape == "pointer"

        await pilot.click(peer)
        await pilot.pause()
        assert peer.expanded is True
        assert peer.size.height > 1
        # The card grew by exactly what it painted — a height that disagrees
        # with the frame laps the block below.
        assert peer.size.height == peer._row_count

        await pilot.click(peer)
        await pilot.pause()
        assert peer.expanded is False
        assert peer.size.height == 1
        assert below.region.y - peer.region.y == 2


@pytest.mark.asyncio
async def test_a_narrow_pane_keeps_the_one_row_guarantee() -> None:
    """The guarantee has to hold where it is hardest: a long name and a long
    body at a width that cannot hold either."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(48, 20)) as pilot:
        await _settle_for_session(pilot, app)
        block = PeerMessageBlock(
            LONG_BODY,
            {
                "pid": 48213,
                "conversation_name": "minerva-user-dashboard-release-cutter-standby",
                "model_label": "anthropic/claude-opus-5",
            },
        )
        app._append_block(block)
        await pilot.pause()
        await pilot.pause()
        assert block.size.height == 1
        assert block._row_count == 1
