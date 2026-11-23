"""Transcript scrolling from the composer, and the jump-to-latest chip.

The keys are app-level bindings, but their hardest half is that the composer is
a ``TextArea``: `alt+<letter>` is PRINTABLE, so the key would insert a letter
before any binding ran. ``Editor._on_key`` claims it from the live binding map,
which is what these tests exercise — posting a ``Key`` to the editor is the
faithful shape (`pilot.press("alt+j")` builds a key with no character, which is
not what a terminal delivers; see the architect's probe).

Every assertion is structural: offsets in rows, widget display flags, editor
text and selection. Nothing here asserts a duration.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len
from textual import events
from textual.widgets import Static

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.subagent_view import SubagentTranscriptView
from local_operator.tui.widgets.transcript import (
    JUMP_CHIP_GLYPH,
    JUMP_CHIP_PAD_CELLS,
    JUMP_CHIP_RIGHT_MARGIN,
    JumpToLatestChip,
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_band_panels import FakeSession as BandFakeSession
from tests.unit.tui.test_band_panels import _async_factory, _fake_jobs, _Job

DRAFT = "half a thought"


@pytest.fixture(autouse=True)
def _own_analytics_writer(tmp_path: Path) -> Iterator[None]:
    """Stop the process-wide analytics writer this module's send path starts.

    A real ``_submit_prompt`` lazily starts ``lo-analytics-writer`` (a daemon
    thread), and the suite's thread-leak reporter pins it on whichever test
    started it — here, ``test_sending_while_scrolled_up_snaps_to_the_bottom``.
    Same shape as ``tests/unit/aida/test_cos_prompt_name.py``: give the module
    its own isolated store and close the writer at teardown (QA round 1, Q1).
    """
    from local_operator.analytics.recorder import reset_recorder_for_test
    from local_operator.analytics.store import AnalyticsStore

    recorder = reset_recorder_for_test(AnalyticsStore(tmp_path / "analytics-test.db"))
    yield
    recorder.close()


async def _settle(pilot: Any, times: int = 4) -> None:
    for _ in range(times):
        await pilot.pause()


async def _boot(pilot: Any, app: OperatorApp, *, rows: int = 40) -> tuple[TranscriptView, Any]:
    """A scrollable transcript and a focused composer holding a draft."""
    await pilot.pause()
    view = app._transcript_view()
    for index in range(rows):
        app._append_block(UserBlock(f"row {index}"))
    await _settle(pilot, 3)
    editor = app._editor()
    editor.focus()
    editor.load_text(DRAFT)
    await _settle(pilot, 2)
    assert view.max_scroll_y > 0, "the fixture never grew a scrollable extent"
    return view, editor


async def _rest_at(pilot: Any, view: TranscriptView, y: int) -> None:
    """Simulate a READER resting at ``y`` (a gesture, not a programmatic move)."""
    view.note_user_scroll(upward=y < view.scroll_y)
    view.scroll_to(y=y, animate=False, immediate=True)
    await _settle(pilot, 2)


def _key(key: str, character: str | None = None) -> events.Key:
    return events.Key(key, character)


async def _press_editor(pilot: Any, editor: Any, key: str, character: str | None) -> None:
    editor.post_message(_key(key, character))
    await _settle(pilot, 3)


@pytest.mark.asyncio
async def test_one_press_steps_two_rows_and_leaves_the_draft_and_caret_alone() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        selection = editor.selection

        await _press_editor(pilot, editor, "alt+k", "k")
        assert view.scroll_y == 28, "alt+k is a TWO-row step"

        await _press_editor(pilot, editor, "alt+j", "j")
        assert view.scroll_y == 30, "alt+j steps the same two rows back"

        # The chord is a scroll, never an edit: TextArea would otherwise have
        # inserted `k` / `j` at the caret.
        assert editor.text == DRAFT
        assert editor.selection == selection
        assert editor.has_focus


@pytest.mark.asyncio
async def test_the_fallbacks_and_half_page_top_and_bottom_all_move_the_reader() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        half = max(1, view.scrollable_content_region.height // 2)

        await _press_editor(pilot, editor, "ctrl+right_square_bracket", None)
        assert view.scroll_y == 28, "ctrl+] is the non-Option fallback"

        await _press_editor(pilot, editor, "ctrl+backslash", None)
        assert view.scroll_y == 30, "ctrl+\\ is the non-Option fallback for down"

        await _press_editor(pilot, editor, "alt+u", "u")
        assert view.scroll_y == max(0, 30 - half), "alt+u is a half page"

        await _press_editor(pilot, editor, "alt+d", "d")
        assert view.scroll_y == 30, "alt+d is the same half page back"

        await _press_editor(pilot, editor, "alt+g", "g")
        await pilot.wait_for_scheduled_animations()
        assert view.scroll_y == 0, "alt+g is the transcript top"

        await _press_editor(pilot, editor, "alt+shift+g", "G")
        await pilot.wait_for_scheduled_animations()
        assert view.scroll_y == view.max_scroll_y, "alt+shift+g returns to the tail"
        assert editor.text == DRAFT


@pytest.mark.asyncio
async def test_an_upward_press_from_the_tail_actually_moves() -> None:
    """Two rows is inside ``TAIL_TOLERANCE_ROWS``; the press must still land.

    Measured without the tolerance rule: the deferred resync re-acquired the
    anchor and the next layout pass pulled the frame home, so the first press
    from the bottom did nothing at all.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        assert view.scroll_y == view.max_scroll_y

        await _press_editor(pilot, editor, "alt+k", "k")
        assert view.scroll_y < view.max_scroll_y - 2
        assert not view.is_following_tail


@pytest.mark.asyncio
async def test_new_output_never_yanks_a_scrolled_up_reader_and_raises_the_chip() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        chip = view._jump_chip
        assert chip is not None and not chip.display

        app._append_block(UserBlock("the next reply"))
        await _settle(pilot, 4)

        assert view.scroll_y == 30, "growth below must not move the reader"
        assert view._unseen_below is True
        assert chip.display is True
        assert chip.region.height == 1


@pytest.mark.asyncio
async def test_returning_to_the_bottom_by_any_means_hides_the_chip() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        app._append_block(UserBlock("the next reply"))
        await _settle(pilot)
        chip = view._jump_chip
        assert chip is not None and chip.display

        # A wheel/drag landing: the reader gestures to the end, not ctrl+end.
        view.note_user_scroll(upward=False)
        view.scroll_to(y=view.max_scroll_y, animate=False, immediate=True)
        await _settle(pilot)

        assert view.is_following_tail
        assert view._unseen_below is False
        assert chip.display is False


@pytest.mark.asyncio
async def test_clicking_the_chip_is_an_end_request() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        app._append_block(UserBlock("the next reply"))
        await _settle(pilot)
        chip = view._jump_chip
        assert chip is not None and chip.display

        await pilot.click(chip)
        await _settle(pilot, 6)

        assert view.scroll_y == view.max_scroll_y
        assert view.is_following_tail
        assert chip.display is False
        assert editor.text == DRAFT


@pytest.mark.asyncio
async def test_the_chip_shifts_no_layout_and_only_its_own_box_takes_clicks() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        app._append_block(UserBlock("the next reply"))
        await _settle(pilot)
        chip = view._jump_chip
        assert chip is not None and chip.display

        shown = (view.max_scroll_y, view.virtual_size, view.scroll_y)
        chip.display = False
        await _settle(pilot)
        hidden = (view.max_scroll_y, view.virtual_size, view.scroll_y)
        assert shown == hidden, "showing the chip must not grow the scroll extent"
        chip.display = True
        await _settle(pilot)

        # Content-sized PLATE: the label plus one cell of padding either side
        # (and the right margin), so the last row of conversation text beside
        # it stays clickable and the label never abuts covered text.
        assert chip.outer_size.width == cell_len(chip._label) + 2 * JUMP_CHIP_PAD_CELLS, (
            chip.outer_size.width,
            chip._label,
        )
        region = view.scrollable_content_region
        assert chip.region.right == region.right - JUMP_CHIP_RIGHT_MARGIN
        assert chip.outer_size.width < region.width


def test_the_label_sheds_exactly_when_its_plate_no_longer_fits() -> None:
    """The rungs are the labels' widths, not constants (design round 1, D4).

    The full label is 16 cells in `en` but 22 in `es`/`fr`, so a fixed
    24-cell rung shed it early and a longer translation would shed the wrong
    way. Swept over 18..40 cells: the widest label whose PLATE (label plus
    both padding cells) still leaves the one-cell right margin is the one
    painted, and nothing ever asks for more room than the region has.
    """
    chip = JumpToLatestChip()
    full = f"{JUMP_CHIP_GLYPH} Jump to latest"
    short = f"{JUMP_CHIP_GLYPH} Latest"
    overhead = 2 * JUMP_CHIP_PAD_CELLS + JUMP_CHIP_RIGHT_MARGIN
    for width in range(18, 41):
        label = chip.label_for_width(width)
        assert cell_len(label) + overhead <= width, (width, label)
        expected = full if cell_len(full) + overhead <= width else short
        assert label == expected, (width, label, expected)
    # The labelling still bottoms out at the glyph when even the short label
    # cannot fit, and the tooltip keeps the sentence (the accessible words).
    assert chip.label_for_width(cell_len(short)) == JUMP_CHIP_GLYPH
    chip.apply_label(cell_len(short))
    assert chip.tooltip == f"{JUMP_CHIP_GLYPH} Jump to latest"


@pytest.mark.asyncio
async def test_sending_while_scrolled_up_snaps_to_the_bottom() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 12)
        assert not view.is_following_tail

        app._submit_prompt("a new prompt")
        await _settle(pilot, 6)

        assert view.scroll_y == view.max_scroll_y
        assert view.is_following_tail
        assert view._unseen_below is False


@pytest.mark.asyncio
async def test_a_remap_moves_the_claim_with_the_binding() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)

        app._apply_keymap({"keymap.transcript_down": "alt+i"}, announce=False)
        await _settle(pilot, 2)

        await _press_editor(pilot, editor, "alt+i", "i")
        assert view.scroll_y == 32, "the remapped key runs the action"

        # The old key is no longer a chord: TextArea inserts the letter, which
        # is the proof the claim follows the LIVE binding map.
        await _press_editor(pilot, editor, "alt+j", "j")
        assert view.scroll_y == 32, "the retired key must not scroll"
        assert editor.text == "j" + DRAFT


@pytest.mark.asyncio
async def test_a_focused_control_keeps_a_key_it_owns() -> None:
    """A remap onto a key the focused widget owns loses there, by design.

    App bindings are non-priority, so the composer's own ``down`` cursor move
    wins when the transcript action is remapped onto it; with the modal session
    picker focused, the same key moves the picker and the transcript never
    moves behind it.
    """
    from local_operator.paths import config_dir
    from tests.e2e.harness import assistant_message, seed_transcript, user_message

    # The picker refuses to open on an empty store (it says so instead), so
    # the fixture seeds ONE real session under the isolated config dir.
    await seed_transcript(
        config_dir() / "sessions" / "synthetic-scroll-picker",
        [user_message("yesterday"), assistant_message("still here")],
    )

    app = OperatorApp(
        lambda: _factory(FakeSession()),
        resume_factory=lambda _identifier: _factory(FakeSession()),
    )
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)
        app._apply_keymap({"keymap.transcript_down": "down"}, announce=False)
        await _settle(pilot, 2)

        before = view.scroll_y
        editor.post_message(_key("down", None))
        await _settle(pilot)
        assert view.scroll_y == before, "the composer's cursor key was stolen"
        assert editor.text == DRAFT, "the key was not inserted as text either"

        app.action_keymap_resume()
        await _settle(pilot, 10)
        from local_operator.tui.widgets.session_picker import SessionPickerScreen

        picker = app.screen
        assert isinstance(picker, SessionPickerScreen)
        moves: list[int] = []
        original = picker.action_move

        def record(delta: int) -> None:
            moves.append(delta)
            original(delta)

        picker.action_move = record  # type: ignore[method-assign]
        await pilot.press("down")
        await _settle(pilot)
        assert moves == [1], "the picker lost its own key"
        assert view.scroll_y == before, "a modal picker must keep the key from the app"


@pytest.mark.asyncio
async def test_focus_on_the_transcript_or_sidebar_still_scrolls_the_reader() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 30)

        view.focus()
        await _settle(pilot)
        assert app.focused is view
        view.post_message(_key("alt+k", "k"))
        await _settle(pilot, 4)
        assert view.scroll_y == 28, "typing at the transcript still scrolls it"
        assert app.focused is editor, "the keystroke hands focus back to the composer"

        sidebar = app._session_sidebar
        await _rest_at(pilot, view, 30)
        sidebar.focus()
        await _settle(pilot)
        # A printable key the list does not bind is re-posted to the composer,
        # whose claim runs the transcript action.
        sidebar.post_message(_key("alt+k", "k"))
        await _settle(pilot, 4)
        assert view.scroll_y == 28, "typing at the sessions list still scrolls it"


@pytest.mark.asyncio
async def test_key_repeat_at_the_top_coalesces_to_one_paging_check() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        view, editor = await _boot(pilot, app)
        await _rest_at(pilot, view, 0)

        checks: list[int] = []
        original = app._check_resume_page

        def spy(*args: Any, **kwargs: Any) -> None:
            checks.append(1)
            return original(*args, **kwargs)

        app._check_resume_page = spy  # type: ignore[method-assign]
        try:
            # One burst, one drain: the app's one-boolean debt coalesces the
            # repeat into a single deferred check.
            for _ in range(30):
                editor.post_message(_key("alt+k", "k"))
            await _settle(pilot, 6)
        finally:
            app._check_resume_page = original  # type: ignore[method-assign]

        assert len(checks) <= 1, f"key repeat requested {len(checks)} pages"
        assert view.scroll_y == 0, "the top is a clamp, not an overshoot"


@pytest.mark.asyncio
async def test_the_subagent_page_scrolls_its_own_transcript() -> None:
    session = BandFakeSession()
    session.jobs = _fake_jobs(_Job("sub-1", "audit the ingest path"))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._append_block(UserBlock("audit the ingest path"))
        app._refresh_band()
        await _settle(pilot)
        app._open_subagent_view("sub-1")
        await _settle(pilot, 6)

        page = app._subagent_view
        assert page is not None
        body = page.transcript
        assert isinstance(body, SubagentTranscriptView)
        assert body.SHOWS_JUMP_CHIP is False
        assert body._jump_chip is None, "the page body owns no chip"
        for index in range(60):
            body.append_block(UserBlock(f"child row {index}"))
        await _settle(pilot, 4)
        assert body.max_scroll_y > 0

        main = app._transcript_view()
        main_y = main.scroll_y
        body.note_user_scroll(upward=True)
        body.scroll_to(y=body.max_scroll_y - 10, animate=False, immediate=True)
        await _settle(pilot, 2)
        before = body.scroll_y

        # The page owns the keyboard: the app binding is dispatched by the
        # screen, so the pilot's key name is the faithful route here (there is
        # no composer TextArea to insert a character, which is the only reason
        # the other tests post a Key with a character).
        await pilot.press("alt+k")
        await _settle(pilot, 4)

        assert body.scroll_y == before - 2, "the visible page's transcript moved"
        assert main.scroll_y == main_y, "the hidden parent transcript did not"


@pytest.mark.asyncio
async def test_ctrl_home_still_retires_a_wedged_parent_lease_with_a_page_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ctrl+home``'s lease recovery survives an open subagent page (F3).

    The round-1 cut skipped ``_break_abandoned_paging_lease`` whenever a
    subagent page was open, which removed the parent transcript's documented
    keyboard recovery for a wedged fetch. The helper is a no-op when nothing
    is leased, so the call must be unconditional; this pins it while the page
    (not the parent) is the scroll target.
    """
    session = BandFakeSession()
    session.jobs = _fake_jobs(_Job("sub-1", "audit the ingest path"))
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        app._append_block(UserBlock("audit the ingest path"))
        app._refresh_band()
        await _settle(pilot)
        app._open_subagent_view("sub-1")
        await _settle(pilot, 6)
        page = app._subagent_view
        assert page is not None

        calls: list[Any] = []
        monkeypatch.setattr(
            app, "_break_abandoned_paging_lease", lambda source: calls.append(source) or False
        )
        await pilot.press("ctrl+home")
        await _settle(pilot, 2)

        assert calls == [app._interaction], "the parent's lease retirement is not skipped"


def test_the_chip_is_a_transcript_child_not_a_screen_chip() -> None:
    """Structural guard: the design put the chip INSIDE the view, and moving it
    to the screen would change the scroll extent and leave it visible over a
    page that is not the conversation."""
    assert issubclass(JumpToLatestChip, Static)
    assert TranscriptView.SHOWS_JUMP_CHIP is True
    assert SubagentTranscriptView.SHOWS_JUMP_CHIP is False
