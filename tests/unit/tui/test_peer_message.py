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
        await pilot.pause()
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
        await pilot.pause()
        # Pretend this delivery was already painted live before the replay.
        app._live_peer_receipts.add("peer-1")
        app._render_resumed_history(session)
        await _settle_for_peer_block(pilot, app)
        # The replay path skipped the already-live id, so no NEW duplicate was
        # mounted for it beyond the one from boot.
        bodies = [b.text() for b in _peer_blocks(app)]
        assert bodies.count("dup note") == 1
