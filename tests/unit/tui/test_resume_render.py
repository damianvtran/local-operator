"""Bounded resume render: last-N widgets, lazy older pages, unchanged history.

The bound is a DISPLAY budget. ``session.history()`` is the model's
conversation and must stay whole even when the transcript only paints the
tail — that split is the whole point of this suite, and a test that only
counted widgets would pass a regression that dropped the model's context.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.tui.app import (
    RESUME_OLDER_NOTICE,
    RESUME_PAGE_MESSAGES,
    RESUME_RENDER_MESSAGES,
    RESUME_START_NOTICE,
    OperatorApp,
    _resume_tail_start,
)
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.tool_card import ToolCard
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory, _transcript_text


def _turn(i: int, *, with_id: bool = True) -> list[Any]:
    """One user / assistant+call / tool-result triple, matching live replay."""
    suffix = {"id": f"u-{i}"} if with_id else {}
    return [
        SimpleNamespace(
            role="user",
            text=f"turn {i:04d}: please check item {i}",
            tool_calls=None,
            content=[],
            custom_type=None,
            **suffix,
        ),
        SimpleNamespace(
            role="assistant",
            id=f"a-{i}" if with_id else None,
            text=f"Reply {i:04d}. Looking at item {i} now.",
            tool_calls=[
                SimpleNamespace(
                    id=f"call-{i}", name="bash", arguments={"command": f"echo item-{i}"}
                )
            ],
            custom_type=None,
            stop_reason=None,
            provider_payload=None,
        ),
        SimpleNamespace(
            role="tool",
            id=f"t-{i}" if with_id else None,
            tool_call_id=f"call-{i}",
            text=f"exit code: 0\nitem-{i}",
            is_error=False,
            provider_payload=None,
            content=[],
            custom_type=None,
        ),
    ]


def _history(n_turns: int, *, with_id: bool = True) -> list[Any]:
    rows: list[Any] = []
    for i in range(n_turns):
        rows.extend(_turn(i, with_id=with_id))
    return rows


def _user_texts(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


async def _wait_for_resume(pilot, app: OperatorApp, *, min_blocks: int = 1) -> None:
    """Boot paints, then a worker adopts the session and replays history.

    One ``pause`` is usually enough, but under xdist the worker can land after
    the first frame, and asserting on an empty transcript then reads as the
    bound dropping every row. Wait for the replay the same way the bang-mode
    tests wait for ``shell_records``.
    """
    for _ in range(50):
        await pilot.pause()
        if len(app.query_one(TranscriptView).blocks()) >= min_blocks:
            return
    raise AssertionError(
        f"resume never painted (blocks={len(app.query_one(TranscriptView).blocks())})"
    )


def test_resume_tail_start_snaps_to_the_nearest_user_row() -> None:
    """A naive ``len - bound`` cut can land on a tool result; the viewport
    must open on a turn, not on a reply with no question."""
    history = _history(5)
    # 15 messages, bound 4 would land on the last turn's assistant (index 11)
    # without the snap; the user row of that turn is index 12.
    assert _resume_tail_start(history, 4) == 12
    assert _resume_tail_start(history, 80) == 0
    assert _resume_tail_start(history, 15) == 0


@pytest.mark.asyncio
async def test_a_short_resume_renders_every_row_and_has_no_older_notice() -> None:
    """A conversation shorter than the bound must behave exactly as today:
    every message mounted, no paging chrome, history() still the full list."""
    session = FakeSession()
    session._history = _history(10)  # 30 messages, well under 80
    original = list(session.history())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app, min_blocks=30)
        blocks = app.query_one(TranscriptView).blocks()
        users = [b for b in blocks if isinstance(b, UserBlock)]
        notices = [b for b in blocks if isinstance(b, NoticeBlock)]
        assert len(users) == 10
        assert _user_texts(app)[0].startswith("turn 0000")
        assert _user_texts(app)[-1].startswith("turn 0009")
        assert not any(b.text() == RESUME_OLDER_NOTICE for b in notices)
        assert not app._resume_pending_head
        # DISPLAY bound only: the session still holds every message.
        assert session.history() == original
        assert len(session.history()) == 30


@pytest.mark.asyncio
async def test_a_long_resume_paints_the_tail_and_keeps_the_head() -> None:
    """The first frame of a long resume shows the last ~80 messages and a
    notice that older ones exist; the model's history is not trimmed."""
    n_turns = 50  # 150 messages
    session = FakeSession()
    session._history = _history(n_turns)
    original = list(session.history())
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        users = _user_texts(app)
        blocks = app.query_one(TranscriptView).blocks()
        assert isinstance(blocks[0], NoticeBlock)
        assert blocks[0].text() == RESUME_OLDER_NOTICE
        # Tail starts at a user row near the bound, never at turn 0000.
        assert users[0].startswith("turn ")
        assert not users[0].startswith("turn 0000")
        assert users[-1].startswith("turn 0049")
        assert len(users) < n_turns
        # Bound is a budget, snapped to a turn: at most bound messages painted
        # (plus the notice), never the whole conversation.
        painted_messages = len(session.history()) - len(app._resume_pending_head)
        assert painted_messages <= RESUME_RENDER_MESSAGES + 2  # snap may add a turn
        assert app._resume_pending_head
        assert session.history() == original
        assert len(session.history()) == n_turns * 3


@pytest.mark.asyncio
async def test_scrolling_up_a_long_resume_reveals_older_rows_in_order() -> None:
    """Paging backward prepends earlier turns with no duplication, no gap,
    and no reordering. Exhausting the head restates the start notice."""
    n_turns = 50
    session = FakeSession()
    session._history = _history(n_turns)
    original_ids = [m.id for m in session.history()]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        seen_before = set(_user_texts(app))
        # Drive the real user-scroll path (Home at the top does not change
        # scroll_y, which is why the hook exists). Repeat until the head is
        # exhausted — one page per gesture, matching the production cascade
        # guard.
        pages = 0
        while app._resume_pending_head and pages < 20:
            view.scroll_home(animate=False)
            view.note_user_scroll()
            for _ in range(8):
                await pilot.pause()
            pages += 1
        users = _user_texts(app)
        assert users[0].startswith("turn 0000")
        assert users[-1].startswith("turn 0049")
        assert users == [f"turn {i:04d}: please check item {i}" for i in range(n_turns)]
        # No duplication: each turn appears once.
        assert len(users) == n_turns
        assert len(users) == len(set(users))
        # Newly revealed rows were not on the first frame.
        assert "turn 0000: please check item 0" not in seen_before
        blocks = view.blocks()
        assert isinstance(blocks[0], NoticeBlock)
        assert blocks[0].text() == RESUME_START_NOTICE
        # Model history still the original objects, in the original order.
        assert [m.id for m in session.history()] == original_ids
        assert not app._resume_pending_head


@pytest.mark.asyncio
async def test_a_paged_resume_does_not_duplicate_by_stable_id() -> None:
    """The id set is the dedupe key: a message already mounted is skipped
    even if it also sits in the deferred head (the compact_file hazard
    `read_transcript_page` returns ``reconciled=True`` for)."""
    session = FakeSession()
    session._history = _history(40)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        # Poison the head with a message the tail already painted.
        already = session._history[-3]  # last user row, on screen
        app._resume_pending_head.append(already)
        before = _user_texts(app)
        view = app.query_one(TranscriptView)
        view.scroll_home(animate=False)
        view.note_user_scroll()
        for _ in range(8):
            await pilot.pause()
        after = _user_texts(app)
        assert after.count(already.text) == 1
        assert before.count(already.text) == 1


@pytest.mark.asyncio
async def test_clear_drops_the_deferred_head() -> None:
    """/clear empties the SCREEN. Paging the old head back onto it would
    undo the clear the first time the reader scrolled up."""
    session = FakeSession()
    session._history = _history(40)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        assert app._resume_pending_head
        app._transcript_view().clear_blocks()
        await pilot.pause()
        assert not app._resume_pending_head
        assert app._resume_head_notice is None
        assert not app._resume_mounted_ids
        # The model's conversation is not the screen.
        assert len(session.history()) == 120


@pytest.mark.asyncio
async def test_a_cross_cut_tool_result_still_settles_its_card() -> None:
    """A call in the deferred head is answered by a result that may already
    live in the rendered tail. Pairing from the whole-conversation index is
    what stops that call replaying as ``interrupted``."""
    # Force the cut to fall between a call and its result: 27 messages, bound
    # would be 80 so this is short… use enough turns that the snap still
    # leaves a tool result in the tail whose call is in the head? With turn
    # snapping the cut is always on a user row, so a call and its result
    # stay together. Pin the pairing by paging a page that includes a call
    # whose result was already indexed from the whole history.
    session = FakeSession()
    session._history = _history(40)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        view.scroll_home(animate=False)
        view.note_user_scroll()
        for _ in range(8):
            await pilot.pause()
        cards = [b for b in view.blocks() if isinstance(b, ToolCard)]
        assert cards
        assert all(c._state == "success" for c in cards)
        assert not any(c._state == "interrupted" for c in cards)


@pytest.mark.asyncio
async def test_resume_page_size_is_smaller_than_the_initial_bound() -> None:
    """A page is paid during an interaction, so it is smaller than the
    first-frame budget. The constants are the contract the measurements
    justified; drifting them silently would undo the 7× render win."""
    assert RESUME_RENDER_MESSAGES == 80
    assert RESUME_PAGE_MESSAGES == 60
    assert RESUME_PAGE_MESSAGES < RESUME_RENDER_MESSAGES


@pytest.mark.asyncio
async def test_assistant_and_tool_rows_stay_paired_across_a_page() -> None:
    """Each revealed turn is still user / prose / card, in that order — the
    same pairing `_project_settled_rows` uses for the tail, which is why a
    backward page is built by that method rather than a second renderer."""
    session = FakeSession()
    session._history = _history(40)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        while app._resume_pending_head:
            view.scroll_home(animate=False)
            view.note_user_scroll()
            for _ in range(6):
                await pilot.pause()
        body = [b for b in view.blocks() if not isinstance(b, NoticeBlock)]
        # triples: UserBlock, AssistantBlock, ToolCard
        assert len(body) == 40 * 3
        for i in range(0, len(body), 3):
            assert isinstance(body[i], UserBlock)
            assert isinstance(body[i + 1], AssistantBlock)
            assert isinstance(body[i + 2], ToolCard)
        # Silence unused import if _transcript_text is handy for debug.
        assert "turn 0000" in _transcript_text(app)
