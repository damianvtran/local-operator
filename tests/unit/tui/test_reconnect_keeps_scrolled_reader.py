"""A reconnect's gap replay must not move a reader who scrolled back.

The defect: when an attached TUI's owner socket drops and comes back,
``AttachedSession`` recovery replays the durable rows it missed as a
``HistoryDeltaEvent(reset=False)``; the app projects them through
``_project_settled_rows``, and ``project_settled_rows`` used to call
``transcript.follow_tail()`` whenever it appended anything. ``follow_tail``
re-acquires the tail anchor unconditionally, so a reader parked in history was
yanked to the newest message on every reconnect.

The rule the fix encodes, and each case below pins one side of it:

* a reader who scrolled away keeps their place (``scrolled_back``);
* a reader at the tail still follows the new rows down (``control_tail``);
* a reconnect that missed nothing changes nothing (``no_new_rows``).

All three drive the REAL reconnect — the production ``RuntimeServer``
registrant and ``AttachedSession`` recovery loop, the harness
``test_reconnect_parity.py`` established — because the defect lives in the
hand-off between the session's recovery and the app's settled-row projection;
posting a synthetic ``HistoryRowsSettled`` would skip the half that decides
whether a gap exists at all.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import TAIL_TOLERANCE_ROWS, UserBlock
from tests.unit.session.runtime.test_server import FakeHandle
from tests.unit.session.test_remote import _never_take_over, _wait_record
from tests.unit.tui.waiting import settle

#: Where the scrolled-back reader parks: well above the tail, so "kept their
#: place" and "followed to the tail" cannot be confused within the tolerance.
_PARKED_Y = 3

#: Filler rows painted before the disconnect, enough to give the transcript a
#: scroll range several viewports tall at the test's 120x24 size.
_FILLER_ROWS = 40


def _gap_rows() -> list[Message]:
    """The rows that become durable while the owner socket is down."""
    return [
        Message.user("durable while disconnected"),
        Message(role="assistant", content=[TextContent(text="answer while disconnected")]),
    ]


def _geometry(view: Any) -> tuple[float, float, bool]:
    return (float(view.scroll_y), float(view.max_scroll_y), bool(view.is_following_tail))


async def _reconnect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scroll_back: bool,
    gap: list[Message],
) -> dict[str, Any]:
    """Boot an attached app, optionally park the reader, drop and restore the owner.

    Returns the geometry before the drop and after recovery settled, plus how
    many ``HistoryRowsSettled`` the app projected (zero for an empty gap).
    """
    # Per-test config root: the registrant record, the transcript and the
    # session pid all resolve under it, never under the operator's config.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    transcript = Transcript(tmp_path / "sessions" / "s1")
    await transcript.append_message(Message.user("visible before disconnect"))

    # Count projections of the reconnect gap so each wait names its event
    # rather than a number of frames.
    settled: list[bool] = []
    real_settled = OperatorApp.on_history_rows_settled

    def on_history_rows_settled(self: OperatorApp, message: Any) -> None:
        try:
            real_settled(self, message)
        finally:
            settled.append(message.reset)

    monkeypatch.setattr(OperatorApp, "on_history_rows_settled", on_history_rows_settled)

    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote: AttachedSession | None = None
    try:
        record = await _wait_record(tmp_path)
        remote = await AttachedSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never_take_over
        )
        attached = remote

        async def factory() -> AttachedSession:
            return attached

        app = OperatorApp(factory)
        async with app.run_test(size=(120, 24)) as pilot:
            await settle(
                pilot,
                lambda: app._session is attached and len(app._transcript_view().blocks()) >= 1,
                what="boot adopted the attached session and painted its row",
            )
            view = app._transcript_view()
            for i in range(_FILLER_ROWS):
                app._append_block(UserBlock(f"question number {i:02d} about the codebase"))
            await settle(
                pilot,
                lambda: view.max_scroll_y > _PARKED_Y + TAIL_TOLERANCE_ROWS
                and view.scroll_y == view.max_scroll_y
                and view.is_following_tail,
                what="filler rows laid out with the reader on the tail",
            )

            if scroll_back:
                # The real user-scroll path: report the gesture (releases the
                # anchor), then move the viewport the way the wheel would.
                view.note_user_scroll()
                view.scroll_to(y=_PARKED_Y, animate=False, immediate=True)
                await settle(
                    pilot,
                    lambda: view.scroll_y == _PARKED_Y and not view.is_following_tail,
                    what="reader parked in history with the tail anchor released",
                )
            before = _geometry(view)

            # The real owner-socket drop: closing the registrant is what a
            # dying runtime does, and the pid file keeps the session "alive"
            # so the follower recovers instead of taking over.
            registrant.close()
            (tmp_path / "sessions" / "s1" / ".session.pid").write_text(str(os.getpid()))
            await settle(
                pilot,
                lambda: attached._recovering,
                what="the owner socket drop started the follower's recovery",
            )
            for row in gap:
                await transcript.append_message(row)

            replacement = RuntimeServer(handle, kind="tui")
            replacement.start()
            try:
                expected = 1 + len(gap)
                await settle(
                    pilot,
                    lambda: not attached._recovering and len(attached.history()) == expected,
                    what="recovery re-attached and caught up on the durable gap",
                )
                if gap:
                    # Wait for the projection AND the relayout it causes: the
                    # yank (when it happens) lands with the size update, so a
                    # wait on the message alone could read the pre-yank frame.
                    await settle(
                        pilot,
                        lambda: bool(settled) and view.max_scroll_y > before[1],
                        what="the gap's rows were projected and laid out",
                    )
                    if not scroll_back:
                        await settle(
                            pilot,
                            lambda: view.scroll_y == view.max_scroll_y,
                            what="the tail reader landed on the new tail",
                        )
                after = _geometry(view)
                blocks = len(view.blocks())
            finally:
                replacement.close()
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()
    return {"before": before, "after": after, "settled": settled, "blocks": blocks}


@pytest.mark.asyncio
async def test_a_scrolled_back_reader_keeps_their_place_when_a_reconnect_paints_gap_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect itself: before the fix this reader landed on max_scroll_y."""
    out = await _reconnect(tmp_path, monkeypatch, scroll_back=True, gap=_gap_rows())
    y0, max0, following0 = out["before"]
    y1, max1, following1 = out["after"]
    assert (y0, following0) == (_PARKED_Y, False)
    # The gap really was painted below the reader, through the reset=False path.
    assert out["settled"] == [False]
    assert max1 > max0
    assert y1 == y0, f"reader moved from {y0} to {y1} (max {max1}): yanked by the gap replay"
    assert y1 < max1 - TAIL_TOLERANCE_ROWS
    assert following1 is False


@pytest.mark.asyncio
async def test_a_reader_at_the_tail_follows_the_reconnect_gap_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the rule: the fix must not strand a tail reader above new rows."""
    out = await _reconnect(tmp_path, monkeypatch, scroll_back=False, gap=_gap_rows())
    y0, max0, following0 = out["before"]
    y1, max1, following1 = out["after"]
    assert (y0 == max0, following0) == (True, True)
    assert out["settled"] == [False]
    assert max1 > max0
    assert y1 == max1
    assert following1 is True


@pytest.mark.asyncio
async def test_a_reconnect_that_missed_nothing_leaves_the_reader_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty gap posts no settled rows at all, so nothing may move.

    Pins that the guard is not the only thing keeping a reader in place: the
    empty-gap early return in the session's durable-suffix replay must keep
    the app from projecting (or following) anything.
    """
    out = await _reconnect(tmp_path, monkeypatch, scroll_back=True, gap=[])
    assert out["settled"] == []
    assert out["after"] == out["before"]
    assert out["blocks"] == 1 + _FILLER_ROWS
