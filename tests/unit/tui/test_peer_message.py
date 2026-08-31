"""TUI rendering of inbound peer messages (`lop send`).

Two paths, mirroring the wake receipt: a LIVE delivery paints a
``PeerMessageBlock`` the instant the event lands, and a resumed conversation
REPLAYS a persisted ``peer_message`` custom row as the same block — without
double-painting one already shown live this session.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from local_operator.harness.types import PeerMessageDeliveredEvent
from local_operator.session.peer import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import PeerMessageBlock, TranscriptView
from tests.unit.tui.test_app_pilot import FakeSession, _factory


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


def test_the_header_falls_back_to_cwd_then_session_id_not_a_bare_pid() -> None:
    """A row reading `peer message from (pid 1)` names nothing a reader can act
    on. The receive side resolves the name from the registry first; when even
    that finds nothing, the cwd basename and then a short session id are still
    better than a kernel-assigned number."""
    by_cwd = PeerMessageBlock("body", {"pid": 1, "cwd": "/Users/x/minerva-core"})
    assert "minerva-core" in by_cwd._header()

    by_id = PeerMessageBlock("body", {"pid": 1, "session_id": "01JQ9ZK4W7X2M8N3PVQ6TYRB5H"})
    header = by_id._header()
    assert "01JQ9ZK4" in header
    # Only a short prefix: a full ULID is 26 cells of entropy that would push
    # the pid and model out of the header.
    assert "01JQ9ZK4W7X2M8N3PVQ6TYRB5H" not in header

    # A real name still wins over both fallbacks.
    named = PeerMessageBlock(
        "body", {"pid": 1, "cwd": "/Users/x/minerva-core", "conversation_name": "release cutter"}
    )
    assert "release cutter" in named._header()

    # Nothing at all: the old bare-pid shape is still the last resort.
    assert "pid 1" in PeerMessageBlock("body", {"pid": 1})._header()


@pytest.mark.asyncio
async def test_dragging_over_a_peer_message_copies_no_app_chrome() -> None:
    """Every row the header WRAPPED to is chrome, not just the first.

    The header is one paragraph, but an ordinary sender name wraps it at
    half-screen widths. Marking only row 0 meant a drag over the message copied
    the tail of the app's own label into the user's clipboard — text they never
    wrote, landing in whatever they were quoting into.
    """
    from textual.selection import Selection as ScreenSelection

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(62, 14)) as pilot:
        await _settle_for_session(pilot, app)
        block = PeerMessageBlock(
            "gates are green",
            {
                "pid": 48213,
                "conversation_name": "minerva-user-dashboard-release-cutter",
                "model_label": "anthropic/claude-opus-5",
            },
        )
        app._append_block(block)
        await pilot.pause()

        # The case only bites when the header actually wraps.
        assert block._header_rows > 1, "widen the case: the header did not wrap"

        copied = block.get_selection(ScreenSelection(None, None))
        assert copied is not None
        text = copied[0]
        assert "gates are green" in text
        # No fragment of the app's label may ride along.
        assert "peer message from" not in text
        assert "minerva-user-dashboard" not in text
        assert "claude-opus" not in text
        assert "pid 48213" not in text


@pytest.mark.asyncio
async def test_a_control_character_in_a_sender_name_cannot_break_the_row() -> None:
    """The sender name crosses the wire from another process, so it is the
    least trusted string this widget paints.

    A newline split the header into rows the block never counted — its height is
    PINNED to that count, so the extra row paints outside the reserved space —
    and an escape sequence would re-ink the transcript from inside a label.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 14)) as pilot:
        await _settle_for_session(pilot, app)
        block = PeerMessageBlock(
            "body",
            {"pid": 7, "conversation_name": "line1\nline2\nline3", "model_label": "m\x1b[31m"},
        )
        app._append_block(block)
        await pilot.pause()

        header = block._header()
        assert "\n" not in header
        assert "\x1b" not in header
        # The pinned height matches what the block actually painted.
        assert block.styles.height is not None
        assert block.styles.height.value == block._header_rows + 1
