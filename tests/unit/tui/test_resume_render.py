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

import local_operator.tui.app as app_module
from local_operator.tui.app import (
    RESUME_OLDER_NOTICE,
    RESUME_PAGE_MESSAGES,
    RESUME_PAGE_TRIGGER_ROWS,
    RESUME_RENDER_MESSAGES,
    RESUME_START_NOTICE,
    OperatorApp,
    _resume_tail_start,
)
from local_operator.tui.session_presentation import OlderHistoryNotice
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


def test_resume_tail_start_snaps_back_to_the_nearest_user_row() -> None:
    """A naive ``len - bound`` cut can land on a tool result; the viewport
    must open on a turn, not on a reply with no question — and the snap goes
    BACKWARD, so the frame is never shorter than the budget.

    This test previously asserted 12 (a FORWARD snap). That direction is the
    defect: the nearest boundary after the cut can be arbitrarily far down the
    conversation, so the "budget" silently became a ceiling and the frame
    rendered fewer messages than asked for. See
    ``test_resume_tail_start_never_renders_fewer_than_the_bound`` for the
    pathological shape that produced 2 rendered messages out of 1007.
    """
    history = _history(5)
    # 15 messages, bound 4: the naive cut is index 11, turn 3's tool result.
    # Backward lands on turn 3's user row at index 9 — a turn boundary, and a
    # frame of 6 messages, which is >= the bound rather than < it.
    assert _resume_tail_start(history, 4) == 9
    assert _resume_tail_start(history, 80) == 0
    assert _resume_tail_start(history, 15) == 0


def test_resume_tail_start_never_renders_fewer_than_the_bound() -> None:
    """The budget is a FLOOR on what gets painted, on every history shape.

    The shape that broke it is the normal one for this harness: ONE prompt,
    a long run of assistant/tool rows, then a short recent follow-up whose
    user row sits inside the last ``bound`` messages. A forward snap jumps to
    that late row and paints only the handful of messages after it — measured
    at **2 of 1007** on the first shape below.

    Rendering that little is not merely a short frame. The content then fits
    inside the viewport, so there is no scrollbar and no offset to travel, and
    the page-back trigger (which only fires on a scroll INTO the top rows) can
    never be reached — the deferred head is unreachable forever while the top
    row promises the reader it is one scroll up.
    """
    bound = RESUME_RENDER_MESSAGES

    def shape(turns: int, steps: int, followups: int = 0) -> list[Any]:
        rows: list[Any] = []
        for i in range(turns):
            rows.append(SimpleNamespace(role="user", custom_type=None, id=f"u-{i}"))
            for k in range(steps):
                rows.append(SimpleNamespace(role="assistant", custom_type=None, id=f"a-{i}-{k}"))
        for f in range(followups):
            rows.append(SimpleNamespace(role="user", custom_type=None, id=f"fu-{f}"))
            rows.append(SimpleNamespace(role="assistant", custom_type=None, id=f"fa-{f}"))
        return rows

    cases = {
        "long turns then a follow-up": shape(5, 200, followups=1),
        "one very long turn": shape(1, 400),
        "one long turn then a follow-up": shape(1, 400, followups=1),
        "ordinary turns": shape(30, 39),
        "many tiny turns": shape(200, 1),
    }
    for name, history in cases.items():
        start = _resume_tail_start(history, bound)
        rendered = len(history) - start
        assert rendered >= bound, f"{name}: rendered {rendered} of {len(history)}, bound {bound}"
        # And still bounded: the backward walk is capped at one more budget, so
        # the render cost the bound exists to control stays controlled.
        assert rendered <= 2 * bound, f"{name}: rendered {rendered}, over the 2x ceiling"


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


def _agentic_history(steps: int, followups: int = 0) -> list[Any]:
    """ONE prompt, then ``steps`` assistant/tool pairs, then short follow-ups.

    The shape the operator reported and the shape this harness produces most:
    a single instruction followed by a long run of tool work. A follow-up
    ("thanks, now do X") puts a user row inside the last ``bound`` messages,
    which is what the old forward snap collapsed onto.
    """
    rows: list[Any] = [
        SimpleNamespace(
            role="user",
            id="u-0",
            text="turn 0000: please check every item",
            tool_calls=None,
            content=[],
            custom_type=None,
        )
    ]
    for k in range(steps):
        rows.append(
            SimpleNamespace(
                role="assistant",
                id=f"a-0-{k}",
                text=f"Step {k:03d}: inspecting the next candidate row.",
                tool_calls=[
                    SimpleNamespace(
                        id=f"call-0-{k}", name="bash", arguments={"command": f"echo {k}"}
                    )
                ],
                custom_type=None,
                stop_reason=None,
                provider_payload=None,
            )
        )
        rows.append(
            SimpleNamespace(
                role="tool",
                id=f"t-0-{k}",
                tool_call_id=f"call-0-{k}",
                text=f"exit code: 0\nitem-{k}",
                is_error=False,
                provider_payload=None,
                content=[],
                custom_type=None,
            )
        )
    for f in range(followups):
        rows.extend(_turn(900 + f))
    return rows


@pytest.mark.asyncio
async def test_a_resumed_long_single_turn_transcript_is_scrollable() -> None:
    """The first frame of a resume must be scrollable when history remains.

    This is the operator's report: a resumed conversation whose transcript had
    NO scrollbar and could not be scrolled up, while its top row read "older
    messages above — scroll up to load". Two separate causes, both asserted
    here because either one alone reproduces it:

    * the tail cut snapped FORWARD onto the late follow-up prompt and painted
      4 blocks of a 404-message history (see the ``_resume_tail_start`` tests);
    * a message-count budget is a proxy for HEIGHT, so even a correct 80-message
      frame can fit inside a tall viewport.

    The consequence is the same and is what makes it a defect rather than a
    short frame: ``_check_resume_page`` only fires on a scroll INTO the trigger
    zone, so a transcript that cannot scroll can never mount its deferred head.
    The notice then promises history the reader physically cannot reach.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        # Let the geometry-driven fill settle. It runs after the initial mount
        # and may mount pages of its own, each with its own settle pass.
        for _ in range(40):
            await pilot.pause()

        viewport = view.container_size.height or view.size.height
        assert viewport, "the transcript has no viewport to measure against"
        # THE assertion: content taller than the viewport, so there is an
        # offset to travel and a scrollbar to travel it with.
        assert view.virtual_size.height > viewport, (
            f"resumed transcript is not scrollable: virtual={view.virtual_size.height} "
            f"viewport={viewport} blocks={len(view.blocks())}"
        )
        assert view.max_scroll_y > 0
        assert view.show_vertical_scrollbar

        # And the promise is actionable: scrolling to the top really does mount
        # the deferred head, which is the whole point of being scrollable.
        assert app._resume_pending_head
        before = len(app._resume_pending_head)
        view.scroll_home(animate=False)
        view.note_user_scroll()
        for _ in range(10):
            await pilot.pause()
        assert len(app._resume_pending_head) < before


@pytest.mark.asyncio
async def test_the_fill_does_not_arm_or_consume_the_page_back_latch() -> None:
    """The scrollability fill is not a gesture and must not spend one.

    It runs outside the page-back latch by construction. If it armed
    ``_resume_in_zone`` the reader's first scroll would collect a free extra
    page; if it consumed one, that scroll's legitimate page would be swallowed.
    Either way the hard-won "ONE page per user gesture" contract breaks, so it
    is pinned rather than trusted.

    PINNED AT 120x170, NOT 120x40, and that is the whole value of this test.
    At 40 rows the frame is already scrollable, the fill returns at its
    scrollable-exit having mounted NOTHING, and the assertion below reads back
    a value no code touched — it passed while the latch was in fact being
    consumed on every taller frame. A guard that cannot go red is worse than
    no guard, so the size is chosen to be one where the fill really mounts.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    app = OperatorApp(lambda: _factory(session))
    # Counted by instrumenting the mount, NOT by sampling a "pending before"
    # after `_wait_for_resume`: the fill starts as soon as the initial mount
    # settles, so by the time the first frame paints it may already have run
    # and a before/after comparison silently reads zero.
    fill_mounts = 0
    async with app.run_test(size=(120, 170)) as pilot:
        original_mount = app._mount_older_resume_page

        def counting_mount(*args: Any, **kwargs: Any) -> None:
            nonlocal fill_mounts
            fill_mounts += 1
            original_mount(*args, **kwargs)

        app._mount_older_resume_page = counting_mount  # type: ignore[assignment]
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        for _ in range(60):
            await pilot.pause()
        app._mount_older_resume_page = original_mount  # type: ignore[assignment]

        # PRECONDITION, asserted rather than assumed: the fill actually did
        # the thing whose side effects this test is about.
        assert fill_mounts, (
            "the fill mounted no pages at this size, so the latch assertion "
            "below would pin a value nothing touched"
        )
        # The latch is still armed: the fill is not a gesture, so the reader's
        # first arrival at the top is still owed a page.
        assert app._resume_in_zone is True
        assert app._resume_paging is False

        # And that first gesture mounts exactly ONE page, not the cascade.
        before = len(app._resume_pending_head)
        assert before
        await _press_and_settle(pilot, view, "ctrl+home")
        assert len(app._resume_pending_head) == before - RESUME_PAGE_MESSAGES


@pytest.mark.asyncio
async def test_a_wheel_reader_at_the_top_can_still_page_after_the_fill() -> None:
    """The latch guarantee, from the READER's side rather than the flag's.

    The flag assertion above says the fill left the latch armed. This says what
    that is FOR, through the only input that cannot re-arm it: a wheel notch
    arriving while the viewport is already clamped at the top moves nothing, so
    it is not evidence the reader left and came back, so it never re-arms. If
    the fill spends the latch, such a reader is stuck forever — measured before
    the fix as 60 settled notches at y=0 mounting exactly nothing, while the
    first row of that transcript told them to scroll up.

    A wheel gesture rather than a keypress on purpose: `pageup`, `home` and
    `ctrl+home` are discrete acts and re-arm unconditionally, so they cannot
    observe this defect at all.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 170)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        for _ in range(60):
            await pilot.pause()
        before = len(app._resume_pending_head)
        assert before, "no deferred head to page"

        # Real wheel events through the widget's own input surface, which is
        # what routes through `note_user_scroll` and the latch. Enough notches
        # to travel the whole frame and sit clamped at the top afterwards.
        for _ in range(40):
            view.post_message(
                MouseScrollUp(
                    widget=view,
                    x=5,
                    y=5,
                    delta_x=0,
                    delta_y=-1,
                    button=0,
                    shift=False,
                    meta=False,
                    ctrl=False,
                )
            )
            for _ in range(14):
                await pilot.pause()

        assert len(app._resume_pending_head) < before, (
            "a wheel reader scrolling to the top of a filled resume earned no "
            f"page: pending {before} -> {len(app._resume_pending_head)} at "
            f"y={view.scroll_y}"
        )


@pytest.mark.asyncio
async def test_mounting_an_older_page_never_moves_the_rows_under_the_reader() -> None:
    """The insert invariant, asserted on every PAINTED frame of the mount.

    Mounting rows ABOVE the viewport changes the scroll EXTENT and the reader's
    absolute offset by the same amount, so the content under their eyes must
    not move. The measurable form is the ANCHOR GAP — the distance from the top
    of the block the reader is on to the top of the viewport — which is
    invariant under a correct insert and jumps by the inserted extent on any
    frame where only one of the two has moved.

    Sampled from the compositor refresh, which is the moment the terminal is
    actually written; sampling after a ``pause`` instead drains the whole
    callback queue and coalesces the entire settle into one observation, which
    reports every intermediate paint as though it never happened. Before this
    was fixed the gap excursed to 120 rows on a 31-row viewport across 5
    painted frames — the reader saw the transcript lurch down ~4 screens and
    snap back.
    """
    session = FakeSession()
    session._history = _agentic_history(200)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _wait_for_resume(pilot, app)
        view = app.query_one(TranscriptView)
        for _ in range(40):
            await pilot.pause()
        assert app._resume_pending_head

        # Park the reader inside the trigger zone. `note_user_scroll` first:
        # a bare scroll leaves the tail anchor following, and the mount would
        # then legitimately drag the viewport to the end — which measures tail
        # following, not the insert.
        view.note_user_scroll()
        view.scroll_to(y=6, animate=False)
        for _ in range(8):
            await pilot.pause()

        anchor = next((b for b in view.blocks() if b.virtual_region.bottom > view.scroll_y), None)
        assert anchor is not None
        baseline_gap = anchor.virtual_region.y - view.scroll_y

        gaps: list[float] = []
        screen = app.screen
        original_refresh = screen._compositor_refresh

        def compositor_refresh() -> None:
            gaps.append(anchor.virtual_region.y - view.scroll_y)
            original_refresh()

        screen._compositor_refresh = compositor_refresh  # type: ignore[method-assign]
        try:
            app._mount_older_resume_page()
            for _ in range(16):
                await pilot.pause()
        finally:
            screen._compositor_refresh = original_refresh  # type: ignore[method-assign]

        assert gaps, "the mount painted no frames to measure"
        displaced = [g for g in gaps if abs(g - baseline_gap) > 0.5]
        assert not displaced, (
            f"the reader's rows moved on {len(displaced)} painted frame(s); "
            f"max excursion {max(abs(g - baseline_gap) for g in gaps)} rows "
            f"(baseline gap {baseline_gap})"
        )
        # And it settled where it started, so the correction is a hold rather
        # than a drift that happened to end in the right place.
        assert abs((anchor.virtual_region.y - view.scroll_y) - baseline_gap) <= 0.5


@pytest.mark.asyncio
async def test_an_unreachable_head_never_says_scroll_up_to_load() -> None:
    """The head notice is an instruction; it must only appear when it works.

    "older messages above — scroll up to load" is what the operator's frame
    showed while the transcript could not be scrolled at all. Even with the
    fill in place a frame can legitimately end up unscrollable — a very tall
    terminal, a head of rows that measure to almost nothing, or the fill's own
    iteration cap — and in that state the instruction is still unfollowable.

    REACHED THROUGH GEOMETRY, not by monkeypatching the cap to 0. The earlier
    version did that, which routed past the stand-down return that was blocking
    this branch in production: the test was green while the notice fired in 0
    of 35 real frames. 600 rows is where three genuinely-mounted fill pages
    still fit inside the viewport, so this reaches the state the same way a
    reader on a large display at a small font does.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 600)) as pilot:
        await _wait_for_resume(pilot, app)
        for _ in range(80):
            await pilot.pause()
        view = app.query_one(TranscriptView)
        viewport = view.container_size.height or view.size.height
        # Precondition: this really is the unreachable state, reached by
        # geometry alone.
        assert view.virtual_size.height <= viewport
        assert app._resume_pending_head

        notices = [b.text() for b in view.blocks() if isinstance(b, NoticeBlock)]
        assert notices, "the head notice vanished"
        assert (
            app_module.RESUME_OLDER_NOTICE not in notices
        ), "the transcript cannot be scrolled, but still tells the reader to scroll up"
        # And it does not claim the conversation starts here either — more
        # history genuinely exists.
        assert app_module.RESUME_START_NOTICE not in notices
        assert app_module.RESUME_UNREACHABLE_NOTICE in notices

        # The state is a dead end for SCROLLING by construction, so the notice
        # itself has to be the way out: it is a control, and activating it
        # loads the page no gesture could reach.
        notice = app._resume_head_notice
        assert isinstance(notice, OlderHistoryNotice)
        assert notice.focusable
        before = len(app._resume_pending_head)
        notice.action_older()
        for _ in range(60):
            await pilot.pause()
        assert len(app._resume_pending_head) < before, (
            "activating the head notice in the unreachable state loaded " "nothing"
        )


@pytest.mark.asyncio
async def test_the_head_notice_stops_saying_unreachable_once_it_is_reachable() -> None:
    """The notice must restate BOTH ways, not latch on its darkest state.

    ``_reconcile_head_notice`` only ever demoted: there was no path from
    "unreachable" back to "scroll up to load", so after the reader activated
    the notice — with a scrollbar now on screen and rows of travel available —
    the top row still told them it could not be scrolled to. That is precisely
    the stale-copy failure `NoticeBlock.restate` exists to prevent, one state
    over from the one this file fixes.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 600)) as pilot:
        await _wait_for_resume(pilot, app)
        for _ in range(80):
            await pilot.pause()
        view = app.query_one(TranscriptView)

        def notices() -> list[str]:
            return [b.text() for b in view.blocks() if isinstance(b, NoticeBlock)]

        assert app_module.RESUME_UNREACHABLE_NOTICE in notices()

        # Obey the notice. The frame becomes scrollable, so the copy must go
        # back to the promise it can now keep.
        app._resume_head_notice.action_older()  # type: ignore[union-attr]
        for _ in range(80):
            await pilot.pause()
        viewport = view.container_size.height or view.size.height
        assert view.virtual_size.height > viewport, "expected the frame to become scrollable"
        assert app_module.RESUME_UNREACHABLE_NOTICE not in notices(), (
            "the frame is scrollable again but the notice still tells the "
            "reader it cannot be reached by scrolling"
        )
        assert app_module.RESUME_OLDER_NOTICE in notices()


@pytest.mark.asyncio
async def test_the_fill_makes_a_tall_terminal_scrollable() -> None:
    """A message budget is a proxy for HEIGHT, and on a tall terminal a bad one.

    80 short messages are ~161 rows; a 170-row terminal swallows them whole, so
    the count-bounded frame is not scrollable even though the tail cut is
    perfectly correct. This is the layer that has to catch it, and it is
    separate from the ``_resume_tail_start`` fix — the cut is right here and
    the frame is still wrong without the fill.
    """
    session = FakeSession()
    session._history = _agentic_history(300)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 170)) as pilot:
        await _wait_for_resume(pilot, app)
        for _ in range(60):
            await pilot.pause()
        view = app.query_one(TranscriptView)
        viewport = view.container_size.height or view.size.height
        assert view.virtual_size.height > viewport, (
            f"tall terminal not scrollable: virtual={view.virtual_size.height} "
            f"viewport={viewport} blocks={len(view.blocks())}"
        )
        # Bounded: the fill stops as soon as it is scrollable, so the render
        # cost the budget exists to control is still controlled.
        assert (
            len(view.blocks())
            <= (RESUME_RENDER_MESSAGES + app_module.RESUME_FILL_MAX_PAGES * RESUME_PAGE_MESSAGES)
            * 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [40, 170, 200, 250, 300])
async def test_a_resumed_conversation_opens_on_its_newest_message(rows: int) -> None:
    """A resume must land on the TAIL, at every viewport height.

    The bounded-resume feature exists to make the HEAD cheap; it must never
    move the tail. This is parametrized across the band where the fill mounts
    because the regression it guards lived only there: the fill's insert ran
    while the frame was still unscrollable (`scroll_y` and `max_scroll_y` both
    0), the held insert anchor clamped the target to row 0, and the offset
    change released tail-following so nothing recovered it. The reader opened a
    resumed conversation ~120 messages BEHIND its newest message, on a session
    they opened to see the latest state — a worse outcome than the missing
    scrollbar this file repairs.

    40 rows is included as the control: the fill mounts nothing there, so this
    height passed throughout and is what made the defect look absent.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, rows)) as pilot:
        await _wait_for_resume(pilot, app)
        for _ in range(80):
            await pilot.pause()
        view = app.query_one(TranscriptView)
        assert view.is_near_bottom(), (
            f"resume opened away from the newest message at 120x{rows}: "
            f"scroll_y={view.scroll_y} max_scroll_y={view.max_scroll_y} "
            f"blocks={len(view.blocks())}"
        )
        # Following, not merely parked at the end by luck: the next thing the
        # conversation appends must carry the viewport with it.
        assert view.is_following_tail


@pytest.mark.asyncio
async def test_the_fill_pages_a_remote_session_through_its_fetch_worker() -> None:
    """The remote/``history_before_token`` branch, which nothing reached before.

    This is the riskiest path in the fill: it is the only one that sets the
    paging gate by hand, spawns a worker, and re-enters itself from that
    worker's completion across a network round trip. Every other test in this
    file uses a local ``FakeSession`` with no token, so the branch had zero
    coverage while carrying the generation guard and the latch restore.

    The fetch itself is stubbed — the point is the fill's control flow around
    it (does it terminate, does it re-enter, does it leave the gate closed),
    not ``RemoteSession``'s wire format.
    """
    session = FakeSession()
    session._history = _agentic_history(200, followups=1)
    # A token means "the conversation continues on the server", which is the
    # condition the remote branch keys on.
    session.history_before_token = "cursor-0"
    app = OperatorApp(lambda: _factory(session))
    fetches = 0

    async def fake_fetch(source: Any) -> None:
        # Stands in for the round trip: prepend a page's worth of history and
        # release the gate, exactly as the real fetch does on success.
        nonlocal fetches
        fetches += 1
        app._resume_pending_head = _agentic_history(30) + app._resume_pending_head
        if fetches >= 2:
            session.history_before_token = None
        app._resume_paging = False

    async with app.run_test(size=(120, 600)) as pilot:
        await _wait_for_resume(pilot, app)
        for _ in range(60):
            await pilot.pause()
        app._fetch_older_display_page = fake_fetch  # type: ignore[assignment]
        # Drain the local head so the fill must take the REMOTE branch.
        app._resume_pending_head = []
        app._fill_resume_until_scrollable()
        for _ in range(120):
            await pilot.pause()

    assert fetches, "the fill never took the remote branch"
    # Terminates rather than spinning: the cap still bounds a branch whose
    # geometry never improves.
    assert fetches <= app_module.RESUME_FILL_MAX_PAGES
    # And it leaves the gate open, so a real gesture is not locked out.
    assert app._resume_paging is False
    # The latch is untouched by the fetch's own mount, same as the local path.
    assert app._resume_in_zone is True


@pytest.mark.asyncio
async def test_the_fill_can_mount_more_than_one_page() -> None:
    """``RESUME_FILL_MAX_PAGES`` must be reachable, not decorative.

    The fill chained its re-measure with ``call_after_refresh`` while the
    paging gate is released from the settle callback ``insert_blocks``
    schedules — which runs strictly later. So attempt 1 always found the gate
    held, took the stand-down return, and nothing rescheduled it: the effective
    cap was ONE page regardless of the constant, and on a viewport needing two
    the resume stayed unscrollable with its head unreachable.

    600 rows needs more than one page here, so a single-page fill cannot
    satisfy the assertion.
    """
    session = FakeSession()
    session._history = _agentic_history(300)
    app = OperatorApp(lambda: _factory(session))
    fill_mounts = 0
    async with app.run_test(size=(120, 600)) as pilot:
        original_mount = app._mount_older_resume_page

        def counting_mount(*args: Any, **kwargs: Any) -> None:
            nonlocal fill_mounts
            fill_mounts += 1
            original_mount(*args, **kwargs)

        # Instrumented before the resume paints, because the fill runs off the
        # initial mount's settle and a before/after count misses it entirely.
        app._mount_older_resume_page = counting_mount  # type: ignore[assignment]
        await _wait_for_resume(pilot, app)
        for _ in range(120):
            await pilot.pause()
        app._mount_older_resume_page = original_mount  # type: ignore[assignment]

    assert fill_mounts > 1, (
        f"the fill mounted {fill_mounts} page(s): the cap of "
        f"{app_module.RESUME_FILL_MAX_PAGES} is unreachable, so it cannot "
        "iterate on a viewport that needs more than one"
    )
