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
from textual.events import Key, MouseScrollUp

from local_operator.tui.app import (
    RESUME_OLDER_NOTICE,
    RESUME_PAGE_MESSAGES,
    RESUME_PAGE_TRIGGER_ROWS,
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


async def _press_and_settle(pilot, view: TranscriptView, key: str) -> None:
    """Press a real key and let its scroll animation fully settle.

    ``pilot.press`` waits for the animator between keys, which hides exactly
    the mid-animation behaviour these tests exist to pin; posting the event
    directly and draining frames keeps the gesture's own timeline visible.
    The pause loop must outlive the animation (Textual animates a Home over
    ~1 s at speed 50) AND the settle callback a page mount schedules.
    """
    event = Key(key, None)
    event.set_sender(pilot.app)
    pilot.app.post_message(event)
    for _ in range(160):
        await pilot.pause()
        if not pilot.app.animator.is_being_animated(view, "scroll_y"):
            # One extra beat for the settle/anchor callbacks the mount queued.
            for _ in range(4):
                await pilot.pause()
            return


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
async def test_an_animated_page_up_mounts_exactly_one_page() -> None:
    """One animated PageUp mounts ONE page — the page-per-gesture contract.

    Textual ANIMATES pageup/home, so the offset crosses the trigger row many
    times inside one gesture. The re-entry guard must hold for the whole
    gesture, not one synchronous callback: the version this test was written
    against let each animation frame mount another page (three pages for one
    PageUp, and the ENTIRE deferred head for one Home on a 600-message
    session), which is the unbounded render cost the display bound exists to
    remove, paid mid-interaction. Drives the REAL key path — focus the
    transcript, post the key, let the animation run — never
    ``scroll_home(animate=False)``.
    """
    session = FakeSession()
    session._history = _history(200)  # 600 messages; ~120 turns deferred
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        before_pending = len(app._resume_pending_head)
        before_blocks = len(view.blocks())
        # Place the viewport ABOVE the trigger row but off the top, the
        # position a reader is in when they page up toward the history.
        view.scroll_to(y=RESUME_PAGE_TRIGGER_ROWS + 8, animate=False)
        await pilot.pause()
        view.focus()
        await pilot.pause()
        await _press_and_settle(pilot, view, "pageup")
        assert len(app._resume_pending_head) == before_pending - RESUME_PAGE_MESSAGES
        assert len(view.blocks()) > before_blocks
        assert view.scroll_offset.y <= RESUME_PAGE_TRIGGER_ROWS


@pytest.mark.asyncio
async def test_an_animated_home_mounts_one_page_and_lands_at_the_top() -> None:
    """One animated Home mounts ONE page and lands at the TOP of it.

    The cascade this guards against was worst for Home: the whole remaining
    conversation mounted in one keypress and the viewport landed
    mid-conversation (y=146 of 270 on a 150-message session), because the Home
    animation and the mount's anchor restore fought for the offset. The
    gesture must be fast, mount one page, and leave the reader at the start
    of what is rendered — not in the middle, and not needing a second press.
    """
    session = FakeSession()
    session._history = _history(50)  # 150 messages; ~70 deferred
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        before_pending = len(app._resume_pending_head)
        assert before_pending
        view.focus()
        await pilot.pause()
        await _press_and_settle(pilot, view, "home")
        # Exactly one page: the head shrank by one page, not to zero.
        assert len(app._resume_pending_head) == before_pending - RESUME_PAGE_MESSAGES
        # The reader landed AT THE TOP of what is rendered, not mid-page.
        assert view.scroll_offset.y <= RESUME_PAGE_TRIGGER_ROWS
        first_users = [t for t in _user_texts(app)][:1]
        assert first_users, "a page mounted"
        # A second Home mounts the NEXT page (the gesture re-arms), and still
        # lands at the top — the reader walks back in pages, never a cascade.
        # The SECOND press may exhaust the head (a 150-message session holds
        # only ~72 deferred, so two 60-message pages reach the start); what
        # is pinned is that the walk is bounded — never more than one page
        # per press — and never a cascade to zero from a single gesture.
        await _press_and_settle(pilot, view, "home")
        assert (
            before_pending - 2 * RESUME_PAGE_MESSAGES
            <= len(app._resume_pending_head)
            < (before_pending - RESUME_PAGE_MESSAGES)
        )
        assert view.scroll_offset.y <= RESUME_PAGE_TRIGGER_ROWS


@pytest.mark.asyncio
async def test_the_composer_pages_the_transcript_by_keyboard() -> None:
    """``ctrl+home`` from the composer reaches the transcript (UX1, U2).

    Default focus is the composer and every plain scroll key is spoken for by
    the Editor, so without a chord the "scroll up to load" affordance was
    mouse-only. ``ctrl+home`` must mount one page without moving focus, and
    ``ctrl+end`` must return the reader to the tail.
    """
    session = FakeSession()
    session._history = _history(50)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        before_pending = len(app._resume_pending_head)
        assert before_pending
        # The composer holds focus, as it does at rest after a resume.
        composer_focused = app.focused
        assert composer_focused is not view
        await _press_and_settle(pilot, view, "ctrl+home")
        assert len(app._resume_pending_head) == before_pending - RESUME_PAGE_MESSAGES
        assert view.scroll_offset.y <= RESUME_PAGE_TRIGGER_ROWS
        # Focus never left the composer — the reader can type immediately.
        assert app.focused is composer_focused
        await _press_and_settle(pilot, view, "ctrl+end")
        assert view.scroll_offset.y >= view.max_scroll_y - 1


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
async def test_arriving_at_the_top_mounts_one_page_then_stops() -> None:
    """Arriving at the top loads ONE page — and nothing more until the reader
    actually travels away from the top rows and comes back.

    The pre-fix trigger was LEVEL-triggered: after a page prepended, the
    anchor restore parked the viewport back inside the trigger rows, the gate
    released while the reader was still at the top, and the next watch firing
    — the settle frames of the mount itself, or a wheel notch clamped against
    the top — mounted another page, and another. To the reader that is "I
    scroll to the top and it loads chunks one after another without me
    scrolling up again". The trigger must be an EDGE, and the property that
    makes it one is that NO page mounts without the reader having travelled
    out of the trigger zone since the previous one.

    Driven with real wheel events as a CONTINUED drag that does not stop at
    the first mount (review round 1, M1: stopping there ends the gesture
    exactly where a cascade would begin, so that shape passed pre-fix). The
    drag runs in the rhythm a trackpad delivers — several notches per frame —
    and then holds the wheel against the clamped top, where the offset cannot
    move at all. On this view the violation is deterministic: a 40-row
    viewport against a ~120-row page means the reader reaches the clamped top
    long before the head is anywhere near exhausted, so pre-fix every mount
    after the first happened with no travel at all (8 of 9 on a 600-message
    session, 3/3 runs). Every mount is audited against the offset's peak
    since the previous one.
    """
    session = FakeSession()
    session._history = _history(200)  # 600 messages; many pages available
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        assert len(app._resume_pending_head) > 2 * RESUME_PAGE_MESSAGES
        # Travel audit: the highest offset seen since the previous mount is
        # how far the reader actually went. A mount with no intervening travel
        # is the level-trigger cascade, whatever gesture was in flight.
        mounts = {"n": 0, "no_travel": 0, "first": True, "peak": 0.0}
        real_mount = OperatorApp._mount_older_resume_page

        def auditing_mount(self: OperatorApp) -> None:
            if not mounts["first"] and mounts["peak"] <= RESUME_PAGE_TRIGGER_ROWS:
                mounts["no_travel"] += 1
            mounts["first"] = False
            mounts["n"] += 1
            mounts["peak"] = 0.0
            real_mount(self)

        OperatorApp._mount_older_resume_page = auditing_mount  # type: ignore[method-assign]

        def notch() -> None:
            view.post_message(
                MouseScrollUp(
                    widget=view,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                    x=10,
                    y=10,
                    delta_x=0,
                    delta_y=-2,
                )
            )

        # A continued drag: bursts per frame, straight through every mount it
        # causes. No early stop — stopping at the first mount ends the
        # gesture exactly where the cascade would begin.
        for _ in range(25):
            for _ in range(5):
                notch()
            mounts["peak"] = max(mounts["peak"], view.scroll_offset.y)
            await pilot.pause()
        # ...then HOLD the wheel against the clamped top: notches keep
        # arriving while the offset is pinned and cannot move. A clamped
        # notch is not travel and must not earn a page.
        for _ in range(30):
            for _ in range(5):
                notch()
            await pilot.pause()
        for _ in range(60):
            await pilot.pause()
        OperatorApp._mount_older_resume_page = real_mount  # type: ignore[method-assign]

        # The contract: a long drag may mount several pages (each genuine
        # re-arrival at the top earns one), but NEVER without travel away
        # from the zone since the previous page.
        assert mounts["no_travel"] == 0, (
            f"{mounts['no_travel']} of {mounts['n']} pages mounted with no "
            "travel away from the trigger zone — the level-trigger cascade"
        )
        assert mounts["n"] >= 1, "the drag reached the top and mounted a page"
        # And once the gesture is over, parked wherever it left them, no
        # further page mounts however long the view idles.
        settled_pending = len(app._resume_pending_head)
        settled_blocks = len(view.blocks())
        for _ in range(120):
            await pilot.pause()
        assert len(app._resume_pending_head) == settled_pending
        assert len(view.blocks()) == settled_blocks
        # A deliberate discrete act at the top — the Home key — still mounts
        # exactly one more page (the pinned per-press contract).
        if app._resume_pending_head:
            view.focus()
            await pilot.pause()
            before = len(app._resume_pending_head)
            await _press_and_settle(pilot, view, "home")
            assert len(app._resume_pending_head) < before
            assert view.scroll_offset.y <= RESUME_PAGE_TRIGGER_ROWS
            for _ in range(60):
                await pilot.pause()
            assert len(app._resume_pending_head) == before - RESUME_PAGE_MESSAGES


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
async def test_a_page_splitting_a_turn_still_pairs_call_with_result() -> None:
    """A page can split a turn whose body exceeds one page: the call and its
    result may land in different pages, and the whole-conversation results
    index is what pairs them — a page-local index would replay the call as
    ``interrupted`` (its result is not in the page) and drop the orphaned
    result's card.

    Constructs the real hazard rather than asserting around it: one turn of
    100 messages (a user row, then a batched run of assistant+call / result
    pairs) is larger than ``RESUME_PAGE_MESSAGES``, so the page cut falls
    INSIDE the turn and the first page mounts a call whose result is still
    deferred.
    """
    # One oversized turn: user row + 99 assistant-with-call / result pairs.
    big_turn: list[Any] = [
        SimpleNamespace(
            role="user",
            text="turn 0000: run the whole batch",
            tool_calls=None,
            content=[],
            custom_type=None,
            id="u-0",
        )
    ]
    for i in range(99):
        big_turn.extend(_turn(i, with_id=False)[1:])  # the assistant+call and result rows only
    session = FakeSession()
    session._history = big_turn + _history(30)[3:]  # the oversized turn first
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        # The oversized turn's head is deferred by the initial bound…
        assert app._resume_pending_head
        # …and paging once splits it: the cut lands inside the turn.
        view.scroll_home(animate=False)
        view.note_user_scroll()
        for _ in range(8):
            await pilot.pause()
        cards = [b for b in view.blocks() if isinstance(b, ToolCard)]
        assert cards
        assert all(c._state == "success" for c in cards)
        assert not any(c._state == "interrupted" for c in cards)
        assert app._resume_pending_head  # the turn still spans the cut


def test_resume_page_size_is_smaller_than_the_initial_bound() -> None:
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
