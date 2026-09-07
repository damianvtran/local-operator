"""The `/copy` picker's mouse gestures, preview scrolling and scroll cues.

Separate from `test_copy_picker.py` (layout and navigation) and
`test_copy_picker_qa.py` (the target grammar and the clipboard payload)
because this file asks a different question: does the surface behave when it
is DRIVEN rather than merely drawn. Everything here goes through the real app
and the real stylesheet — the lightweight hosts elsewhere in this suite
declare no `CSS_PATH`, so a card sized by percentage rules is not sized at all
under one, and a mouse coordinate against an unsized card means nothing.

Mouse events are posted as messages rather than driven through
``pilot.click``. That is not a shortcut: the pilot pauses between clicks, so
it cannot produce the case that matters most here — two events queued with no
event loop turn between them, which is what a fast double-click on a busy
machine delivers and what used to crash the app.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual import events
from textual.binding import Binding
from textual.geometry import Size

from local_operator.tui.copy_targets import CopyTarget, build_copy_targets
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.copy_picker import (
    HEADER_ROWS,
    MIN_GUTTER_TRACK_ROWS,
    PREVIEW_HINT,
    PREVIEW_WRAP_BUDGET,
    PREVIEW_WRAP_STRIDE,
    TOO_SMALL_NOTICE,
    TOO_SMALL_NOTICE_SHORT,
    CopyPickerScreen,
)

CODE_ANSWER = "Here it is.\n\n```python\ndef f():\n    return 1\n```\n\n> and a quote"


def _answer(text: str, truncated: bool = False) -> AssistantBlock:
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    if truncated:
        block.mark_truncated()
    return block


def _targets(*texts: str) -> list[CopyTarget]:
    return build_copy_targets([_answer(text) for text in texts])


def _real_app():
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


async def _open(app, targets: list[CopyTarget], pilot, callback=None) -> CopyPickerScreen:
    screen = CopyPickerScreen(targets)
    if callback is None:
        app.push_screen(screen)
    else:
        app.push_screen(screen, callback)
    await pilot.pause()
    await pilot.pause()
    return screen


def _card_row(screen: CopyPickerScreen, row: int) -> tuple[int, int]:
    """Screen coordinates of card row ``row``, a few cells in from its left."""
    region = screen._body.region
    return region.x + 4, region.y + row


def _tree_row(screen: CopyPickerScreen, offset: int) -> tuple[int, int]:
    """Screen coordinates of the ``offset``-th DRAWN tree row."""
    return _card_row(screen, HEADER_ROWS + offset)


def _preview_row(screen: CopyPickerScreen, offset: int = 1) -> tuple[int, int]:
    """Screen coordinates inside the preview pane (``offset`` past its header)."""
    tree_rows, _ = screen._split_rows()
    return _card_row(screen, HEADER_ROWS + tree_rows + 1 + offset)


def _click(screen: CopyPickerScreen, x: int, y: int, button: int = 1, chain: int = 1):
    return events.Click(
        widget=screen._body,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
        chain=chain,
    )


def _move(screen: CopyPickerScreen, x: int, y: int) -> events.MouseMove:
    return events.MouseMove(
        widget=screen._body,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
    )


def _wheel(screen: CopyPickerScreen, x: int, y: int, down: bool):
    kind = events.MouseScrollDown if down else events.MouseScrollUp
    return kind(
        widget=screen._body,
        x=0,
        y=0,
        delta_x=0,
        delta_y=1 if down else -1,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=x,
        screen_y=y,
    )


# --- the wheel ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_wheel_over_the_tree_moves_the_cursor_by_the_app_sensitivity() -> None:
    """Stepped by the LIVE `App.scroll_sensitivity_y`, not a hardcoded 1.

    It is a per-instance attribute set in `App.__init__` (2.0 here), so a
    constant would silently desynchronise the moment anything changed it and
    one gesture would then travel at two speeds depending on where the pointer
    sat — the position-dependence `settings_view` records the measurement for.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(20)]), pilot)
        step = max(1, int(app.scroll_sensitivity_y))
        assert step > 1, "this test is vacuous if the app steps by one"

        screen.post_message(_wheel(screen, *_tree_row(screen, 0), down=True))
        await pilot.pause()
        assert screen._selected == step

        app.scroll_sensitivity_y = 5.0
        screen.post_message(_wheel(screen, *_tree_row(screen, 0), down=True))
        await pilot.pause()
        assert screen._selected == step + 5, "the step is read live, not cached"


@pytest.mark.asyncio
async def test_the_wheel_over_the_tree_clamps_rather_than_wrapping() -> None:
    """This screen's documented rule for every gesture: a scroll that came
    round to the other end of the list would read as the picker resetting
    itself."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(8)]), pilot)
        for _ in range(20):
            screen.post_message(_wheel(screen, *_tree_row(screen, 0), down=False))
        await pilot.pause()
        assert screen._selected == 0

        for _ in range(40):
            screen.post_message(_wheel(screen, *_tree_row(screen, 0), down=True))
        await pilot.pause()
        assert screen._selected == len(screen.visible_rows) - 1


@pytest.mark.asyncio
async def test_the_wheel_over_the_preview_scrolls_it_and_leaves_the_cursor_alone() -> None:
    """The two panes are ONE `Static`, so the event's widget is identical over
    both and cannot discriminate them — the routing is by screen row against
    the card's own layout. A wheel over text the user is reading must not
    change what is selected under them."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Most-recent-FIRST: `build_copy_targets` walks the transcript in
        # reverse, so the long answer has to be seeded last to sit at row 0.
        body = "\n".join(f"line {index}" for index in range(120))
        screen = await _open(app, _targets("second answer", body), pilot)
        before = screen._selected

        screen.post_message(_wheel(screen, *_preview_row(screen), down=True))
        await pilot.pause()
        assert screen._preview_offset == max(1, int(app.scroll_sensitivity_y))
        assert screen._selected == before, "the cursor must not move with the preview"

        screen.post_message(_wheel(screen, *_tree_row(screen, 0), down=True))
        await pilot.pause()
        assert screen._selected != before, "the tree still takes its own notches"


@pytest.mark.asyncio
async def test_the_preview_offset_clamps_at_both_ends() -> None:
    """Clamped, never wrapped — and the DOWN ceiling is the offset whose pane
    ends on the last source line, not the last source line itself.

    This test previously asserted `== 39` on a 40-line document, i.e. the
    document scrolled until only its final line remained above a pane of blank
    rows. Review round 1 (MAJOR-2, and U5 independently) rejected that
    ceiling: it pushed up to 33 rows of already-read text off the top of a
    document the user was still reading, where `less` and this card's own tree
    both stop with the last line at the bottom. The assertion is rewritten to
    the corrected contract rather than relaxed — the last line must be ON
    SCREEN at the ceiling, and nothing may remain below it.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(40))
        screen = await _open(app, _targets(body), pilot)

        for _ in range(200):
            screen.post_message(_wheel(screen, *_preview_row(screen), down=True))
        await pilot.pause()
        ceiling = screen._preview_offset
        assert (
            0 < ceiling < 39
        ), f"stops with the document's end at the foot, not past it: {ceiling}"
        lines = screen.render_lines_for_test()
        assert any("line 39" in line for line in lines), "the last line is on screen"
        assert not any("more lines" in line for line in lines), "and nothing is below it"

        for _ in range(200):
            screen.post_message(_wheel(screen, *_preview_row(screen), down=False))
        await pilot.pause()
        assert screen._preview_offset == 0


# --- clicks ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_single_click_selects_and_previews_without_copying() -> None:
    """The deliberate divergence from `session_picker`, whose click acts
    immediately. That shape is right there because its list has no preview: the
    row text is all the information that exists. Here the preview IS the
    picker's reason to exist, so a click that copied would remove the
    confirmation step from the mouse path only — handing the mouse user a less
    careful version of the feature than the keyboard user gets, and the mistake
    it enables is silent (the clipboard is overwritten and the modal is gone).
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("first", "second", "third"), pilot, chosen.append)

        screen.post_message(_click(screen, *_tree_row(screen, 2)))
        await pilot.pause()
        assert screen._selected == 2
        assert chosen == [], "a single click must not copy"
        assert screen.is_active, "a single click must not dismiss"
        preview = screen.render_lines_for_test()
        assert any(line.startswith("Preview · ") for line in preview)


@pytest.mark.asyncio
async def test_a_double_click_copies_the_row_it_landed_on() -> None:
    """`Click.chain` is Textual's own double-click count (0.5 s threshold), so
    the contract needs no timing invented here."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("first", "second", "third"), pilot, chosen.append)

        screen.post_message(_click(screen, *_tree_row(screen, 1), chain=2))
        await pilot.pause()
        await pilot.pause()
        assert len(chosen) == 1
        assert chosen[0] is not None and chosen[0].label == "second"


@pytest.mark.asyncio
@pytest.mark.parametrize("button", [2, 3])
async def test_the_other_buttons_are_ignored_before_any_state_changes(button: int) -> None:
    """A right-click asking for a context menu is measured to arrive here, and
    a middle-click paste with it. Neither may move the cursor on its way to
    being ignored, which is why the button is tested first."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("first", "second", "third"), pilot, chosen.append)

        screen.post_message(_click(screen, *_tree_row(screen, 2), button=button))
        screen.post_message(_click(screen, *_tree_row(screen, 2), button=button, chain=2))
        await pilot.pause()
        assert screen._selected == 0
        assert chosen == []
        assert screen.is_active


@pytest.mark.asyncio
async def test_the_chrome_and_the_backdrop_are_inert() -> None:
    """`session_picker`'s three guards, and its docstring records why they are
    mandatory rather than defensive: its first cut without them resolved a
    click on the footer to session #12 and the dimmed backdrop to row 0. Both
    coordinates were re-measured as still delivered here.

    The backdrop deliberately does NOT dismiss. The app's other two modals
    have no mouse handling at all, so backdrop-dismissal would exist on
    exactly one of three and a user who learned it here would be stuck on
    `/resume`.

    The tree is deliberately LONGER than the pane. On a short tree the third
    guard ("the index must exist") catches the footer on its own, so the
    second one looks redundant — it is only load-bearing when rows exist below
    the drawn page, which is precisely session_picker's documented case of a
    footer click resolving to session #12. Measured here at 100x30: without
    the drawn-page cap, the footer's offset is 29 against a 40-row list and
    selects row 29.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(
            app, _targets(*[f"answer {i}" for i in range(40)]), pilot, chosen.append
        )
        screen._move_to(3)
        await pilot.pause()

        region = screen._body.region
        tree_rows, preview_rows = screen._split_rows()
        assert len(screen.visible_rows) > tree_rows + preview_rows + 2, (
            "the fixture must have rows below the drawn page, or the drawn-page "
            "guard is not exercised"
        )
        elsewhere = [
            ("title", (region.x + 4, region.y)),
            ("rule under title", (region.x + 4, region.y + 1)),
            ("rule under tree", (region.x + 4, region.y + HEADER_ROWS + tree_rows)),
            ("preview header", (region.x + 4, region.y + HEADER_ROWS + tree_rows + 1)),
            ("preview body", _preview_row(screen)),
            ("footer", (region.x + 4, region.y + HEADER_ROWS + tree_rows + preview_rows + 2)),
            ("card padding", (region.x - 1, region.y + 3)),
            ("backdrop left", (0, region.y + 3)),
            ("backdrop above", (region.x + 4, 0)),
        ]
        for name, (x, y) in elsewhere:
            screen.post_message(_click(screen, x, y))
            screen.post_message(_click(screen, x, y, chain=2))
            await pilot.pause()
            assert screen._selected == 3, f"{name} moved the cursor"
            assert chosen == [], f"{name} copied"
            assert screen.is_active, f"{name} dismissed the modal"


@pytest.mark.asyncio
async def test_a_short_tree_leaves_no_clickable_blank_below_its_rows() -> None:
    """A short tree's pane is capped at the rows that EXIST, so the space under
    them belongs to the preview rather than to a blank tail of the tree — and
    the region below the last row must select nothing whatever the cause.

    Asserted as an invariant of the layout rather than of one coordinate: the
    drawn page and the row list are the same length here, which is what makes
    `_index_at`'s second guard non-binding on THIS shape (the third catches
    the overshoot). The long-tree case, where the second guard is the only
    thing standing between a footer click and row 29, is covered by
    `test_the_chrome_and_the_backdrop_are_inert`.
    """
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("only answer"), pilot)
        tree_rows, _ = screen._split_rows()
        rows = len(screen.visible_rows)
        assert tree_rows == rows, "the pane is capped at the rows that exist"

        # Every row from the end of the tree to the bottom of the card.
        for offset in range(rows, screen._row_budget()):
            screen.post_message(_click(screen, *_tree_row(screen, offset), chain=2))
        await pilot.pause()
        assert screen.is_active
        assert screen._selected == 0


# --- hover -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hover_highlights_a_row_without_looking_like_the_selection() -> None:
    """Hover and selection share the ground and the CARET discriminates them.
    If hover painted like selection the user would see two selected rows and
    could not tell which one Enter takes."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("first", "second", "third"), pilot)

        screen.post_message(_move(screen, *_tree_row(screen, 2)))
        await pilot.pause()
        assert screen._hovered == 2
        assert screen._selected == 0, "hovering must not select"

        lines = screen.render_lines_for_test()
        carets = [line for line in lines if line.startswith("❯")]
        assert len(carets) == 1, "exactly one row may carry the cursor mark"
        assert screen.styles.pointer == "pointer"


@pytest.mark.asyncio
async def test_the_pointer_is_a_hand_only_over_a_tree_row() -> None:
    """Including NOT over the preview: it is scrollable, not clickable, and a
    hand there would promise a click it does not keep."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(60))
        screen = await _open(app, _targets("second", body), pilot)

        screen.post_message(_move(screen, *_tree_row(screen, 0)))
        await pilot.pause()
        assert screen.styles.pointer == "pointer"

        screen.post_message(_move(screen, *_preview_row(screen)))
        await pilot.pause()
        assert screen.styles.pointer == "default"
        assert screen._hovered is None

        screen.post_message(_move(screen, *_tree_row(screen, 0)))
        await pilot.pause()
        screen.post_message(events.Leave(screen._body))
        await pilot.pause()
        assert screen._hovered is None
        assert screen.styles.pointer == "default"


# --- the dismiss race --------------------------------------------------------


@pytest.mark.asyncio
async def test_two_queued_clicks_dismiss_once_instead_of_crashing() -> None:
    """The regression this guard exists for. Two `Click` events with no event
    loop turn between them — a fast double-click on a busy machine — reached
    `dismiss` after the screen had already popped, and Textual raised
    `ScreenStackError: Can't pop screen`. Reproduced on the unguarded code
    before the guard was written.

    Posted rather than driven through `pilot.click(times=2)`, which pauses
    between clicks and therefore CANNOT reproduce it — a test written that way
    would pass against the crashing code.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("first", "second"), pilot, chosen.append)

        x, y = _tree_row(screen, 0)
        screen.post_message(_click(screen, x, y, chain=2))
        screen.post_message(_click(screen, x, y, chain=3))
        await pilot.pause()
        await pilot.pause()
        assert len(chosen) == 1, "one gesture, one clipboard write"


@pytest.mark.asyncio
async def test_the_keyboard_path_cannot_crash_by_dismissing_twice() -> None:
    """The same race exists WITHOUT a mouse: two `action_choose()` calls, and
    choose-then-cancel, both raised `ScreenStackError` on the unguarded code,
    so a user hammering Enter could crash the app. Guarding only the click
    handler would have left that in place, which is why the guard lives in
    `_dismiss_result` where both paths pass through."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("first", "second"), pilot, chosen.append)
        screen.action_choose()
        screen.action_choose()
        screen.action_cancel()
        await pilot.pause()
        await pilot.pause()
        assert len(chosen) == 1
        assert chosen[0] is not None


@pytest.mark.asyncio
async def test_the_pointer_shape_is_released_before_the_screen_leaves() -> None:
    """The modal goes without another mouse move, so OSC 22 has to be restored
    while this screen still owns the pointer. Without it a user who clicks to
    copy is left with a hand cursor over their transcript.

    Observed AT DISMISS TIME, in the dismiss callback, rather than after the
    screen has left the stack. Round 1 (MINOR-1) showed the after-the-fact
    assertion was vacuous: `styles.pointer` reads `"default"` once the screen
    is off the stack whether or not the code resets it, so deleting the line
    under test left all 127 tests green. The callback runs while the screen is
    still the one that owns the pointer, which is the only moment at which the
    reset is observable — and is exactly the moment the terminal is being told.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        at_dismiss: list[str] = []
        screen = await _open(
            app,
            _targets("first", "second"),
            pilot,
            lambda _result: at_dismiss.append(str(screen.styles.pointer)),
        )
        screen.post_message(_move(screen, *_tree_row(screen, 0)))
        await pilot.pause()
        assert screen.styles.pointer == "pointer"

        screen.post_message(_click(screen, *_tree_row(screen, 0), chain=2))
        await pilot.pause()
        await pilot.pause()
        assert at_dismiss == ["default"], (
            "the hand cursor was still set as the screen left, so it would "
            f"outlive the modal: {at_dismiss}"
        )


# --- preview scrolling -------------------------------------------------------


@pytest.mark.asyncio
async def test_shift_arrows_scroll_the_preview_for_keyboard_parity() -> None:
    """Every mouse capability needs a key, or the mouse becomes the only way to
    do something — on a terminal UI that is a regression, not a feature."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(120))
        screen = await _open(app, _targets(body), pilot)

        await pilot.press("shift+down", "shift+down", "shift+down")
        await pilot.pause()
        assert screen._preview_offset == 3
        assert screen._selected == 0

        await pilot.press("shift+up")
        await pilot.pause()
        assert screen._preview_offset == 2

        await pilot.press("shift+pagedown")
        await pilot.pause()
        assert screen._preview_offset > 2, "a page moves by more than a line"


@pytest.mark.asyncio
async def test_left_right_and_tab_stay_unbound() -> None:
    """Both plausible uses are wrong here. Focus-swapping would give a screen
    with no focus model a MODE, in which `up`/`down` mean two different things
    depending on state nothing in the frame shows; collapse/expand is a
    different feature. `shift+↑↓` scrolls the preview without a mode, which is
    why it is the right shape."""
    keys = {
        binding.key if isinstance(binding, Binding) else binding[0]
        for binding in CopyPickerScreen.BINDINGS
    }
    assert keys, "bindings must not be empty"
    for key in ("left", "right", "tab", "space", "j", "k", "slash"):
        assert key not in keys, f"{key} is bound"
    # The preview keys ARE bound, so the assertions above are about restraint
    # rather than about an empty table.
    assert {"shift+up", "shift+down"} <= keys
    # `ctrl+c` stays the global interrupt: a modal claiming it would change
    # what stopping a turn means depending on whether an overlay is open.
    assert "ctrl+c" not in keys


@pytest.mark.asyncio
async def test_the_preview_offset_resets_on_every_cursor_move() -> None:
    """The preview is a DIFFERENT DOCUMENT once the selection changes.
    Carrying an offset across targets would open the next preview part-way
    down a document the user never scrolled."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # Seeded last so the long answer is row 0 (most-recent-first).
        long_body = "\n".join(f"line {index}" for index in range(120))
        screen = await _open(app, _targets("second answer", long_body), pilot)

        await pilot.press("shift+down", "shift+down", "shift+down")
        await pilot.pause()
        assert screen._preview_offset == 3

        await pilot.press("down")
        await pilot.pause()
        assert screen._preview_offset == 0

        await pilot.press("up")
        await pilot.pause()
        assert screen._preview_offset == 0


@pytest.mark.asyncio
async def test_the_remainder_counts_down_as_the_preview_scrolls() -> None:
    """A LIVE remainder, which is what makes it self-checking without ordinals:
    scroll, and the number goes down. It is counted in SOURCE lines — the unit
    the header beside it reports — because a marker counting wrapped rows
    contradicted the header in the same pane."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        total = 120
        # Each source line WRAPS to several rows, which is what makes this test
        # able to tell the two units apart at all. With short lines the counts
        # coincide and a row-counting implementation passes — measured: a
        # 79-line answer of 43-cell lines produces exactly 79 rows in an
        # 88-cell pane, so the units are indistinguishable there.
        body = "\n".join(f"line {index} " + "padding " * 20 for index in range(total))
        screen = await _open(app, _targets(body), pilot)

        target = screen.selected_target()
        assert target is not None
        _, rows_per_source = screen._wrap_preview(target, max(1, screen._card_width() - 2), 0)
        assert max(rows_per_source) > 1, "the fixture must wrap or this test is vacuous"

        def _claim() -> int:
            marker = next(line for line in screen.render_lines_for_test() if "more lines" in line)
            return int(marker.strip().split()[1])

        first = _claim()
        # Every source line WHOLLY on screen is one the marker must not count.
        # A line only partly drawn still counts as remaining — the user cannot
        # read it, and claiming otherwise is the same over-count in miniature.
        drawn = screen.render_lines_for_test()
        rows_on_screen = sum(1 for line in drawn if line.startswith(("line ", "padding")))
        consumed = 0
        whole_lines = 0
        for rows in rows_per_source:
            if consumed + rows > rows_on_screen:
                break
            consumed += rows
            whole_lines += 1
        assert first == total - whole_lines

        # And it is genuinely in the other unit: a row-counting implementation
        # would quote a bigger number here, which is the contradiction the
        # header made visible in the same pane.
        assert first < len(rows_per_source) * max(rows_per_source) - rows_on_screen

        await pilot.press("shift+down", "shift+down", "shift+down", "shift+down", "shift+down")
        await pilot.pause()
        assert _claim() == first - 5


@pytest.mark.asyncio
async def test_the_remainder_vanishes_at_the_end_of_the_document() -> None:
    """Its ABSENCE has to mean "this is everything" — `todo_panel`'s rule. A
    marker that stayed at `… 0 more lines`, or that kept a stale count at the
    bottom, would make the presence of the marker meaningless."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(40))
        screen = await _open(app, _targets(body), pilot)
        assert any("more lines" in line for line in screen.render_lines_for_test())

        for _ in range(60):
            await pilot.press("shift+down")
        await pilot.pause()
        lines = screen.render_lines_for_test()
        assert not any("more lines" in line for line in lines)
        assert any("line 39" in line for line in lines), "the last line must be reachable"


@pytest.mark.asyncio
async def test_the_header_says_where_a_scrolled_preview_starts() -> None:
    """The symmetric cue to the foot marker, and free: the header row already
    exists and carries only the total. Without it a scrolled preview looks
    like a document that simply starts at "line 40"."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(120))
        screen = await _open(app, _targets(body), pilot)
        header = next(line for line in screen.render_lines_for_test() if line.startswith("Preview"))
        assert "from line" not in header, "an unscrolled preview claims no position"

        await pilot.press("shift+down", "shift+down")
        await pilot.pause()
        header = next(line for line in screen.render_lines_for_test() if line.startswith("Preview"))
        assert "from line 3" in header


@pytest.mark.asyncio
async def test_the_wrap_budget_is_a_window_and_not_a_dead_end() -> None:
    """`PREVIEW_WRAP_BUDGET` was a PREFIX cap: it wrapped the first 200 source
    lines and no more. That was invisible while only ~15 rows were ever shown,
    and became a visible dead end the moment the preview could scroll —
    scrolling a 600-line answer stopped at source line 200 with a remainder
    that would not go down, which reads as a broken surface.

    Deliberately walks PAST the budget in one jump and then reads the frame:
    the lines beyond it must be drawn and the remainder must still be honest.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        total = 600
        body = "\n".join(f"line {index}" for index in range(total))
        screen = await _open(app, _targets(body), pilot)

        beyond = PREVIEW_WRAP_BUDGET + 50
        screen._scroll_preview_to(beyond)
        await pilot.pause()
        lines = screen.render_lines_for_test()
        assert any(f"line {beyond}" in line for line in lines), "past the budget, still drawn"

        marker = next(line for line in lines if "more lines" in line)
        shown = sum(1 for line in lines if line.startswith("line "))
        assert int(marker.strip().split()[1]) == total - beyond - shown

        # And the far end is reachable rather than floored at the budget. The
        # ceiling is the offset whose pane ENDS on the last line (MAJOR-2/U5),
        # so the property that matters is that the last line is drawn — not
        # that the offset equals `total - 1`, which was the old ceiling that
        # scrolled the document off its own pane.
        screen._scroll_preview_to(total)
        await pilot.pause()
        assert screen._preview_offset > PREVIEW_WRAP_BUDGET, "far past the budget"
        lines = screen.render_lines_for_test()
        assert any(f"line {total - 1}" in line for line in lines)
        assert not any("more lines" in line for line in lines), "nothing left below"


# --- the scroll cues ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tree_cues_appear_only_in_the_direction_that_overflows() -> None:
    """`↓ N more` and `↑ N` ride the rules that already exist, so they cost no
    row. BOTH directions are cued, unlike `todo_panel`'s single remainder,
    because `_window_start` centres the cursor: this list is very often clipped
    at both ends at once, and a down-only cue would say the list continues
    below while silently hiding the rows above."""
    app = _real_app()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(30)]), pilot)

        def _cues() -> tuple[bool, bool]:
            # Keyed on the ARROW, not on the word "more": both cues carry
            # `more` since D12 made the up cue a remainder rather than the
            # bare `↑ 6`, which reads as an ordinal.
            lines = screen.render_lines_for_test()
            return (
                any("↑" in line and "─" in line for line in lines),
                any("↓" in line and "─" in line for line in lines),
            )

        assert _cues() == (False, True), "at the top: only the down cue"

        await pilot.press("end")
        await pilot.pause()
        assert _cues() == (True, False), "at the bottom: only the up cue"

        await pilot.press("home")
        for _ in range(12):
            await pilot.press("down")
        await pilot.pause()
        assert _cues() == (True, True), "windowed from the middle: both at once"


@pytest.mark.asyncio
async def test_a_tree_that_fits_advertises_no_paging() -> None:
    """The cue's absence has to mean "this is everything", so a short tree must
    not carry one."""
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("only answer"), pilot)
        lines = screen.render_lines_for_test()
        assert not any("more" in line and "─" in line for line in lines)
        assert not any("↑ " in line and "─" in line for line in lines)


@pytest.mark.asyncio
async def test_the_cue_slot_does_not_resize_as_its_digits_change() -> None:
    """Sized on the WIDEST rendering, not the current value. Sizing it on the
    current text makes the rule change length as the digit count does, so a
    user who is only scrolling watches the divider twitch under them for no
    reason they can see (todo_panel's U1)."""
    app = _real_app()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(30)]), pilot)
        widths = set()
        for _ in range(len(screen.visible_rows)):
            rules = [line for line in screen.render_lines_for_test() if "─" in line]
            widths.update(len(line) for line in rules)
            await pilot.press("down")
            await pilot.pause()
        assert len(widths) == 1, f"the rules changed width as the cue did: {widths}"


@pytest.mark.asyncio
async def test_the_gutter_thumb_tracks_the_window_and_is_shed_when_narrow() -> None:
    """The only CONTINUOUS signal here — the remainder says how much is left,
    the thumb says where you are and moves as you go.

    It is a SHED, not a decoration: it widens every tree row, so drawing it
    unconditionally would raise the drawability floor and newly HIDE the card
    on terminals that draw one today."""
    app = _real_app()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(30)]), pilot)
        assert screen._gutter_drawn(screen._card_width())

        top = screen._thumb_span(screen._tree_rows, screen._window_start(screen._tree_rows))
        await pilot.press("end")
        await pilot.pause()
        bottom = screen._thumb_span(screen._tree_rows, screen._window_start(screen._tree_rows))
        assert top.start < bottom.start, "the thumb moves down with the window"
        assert len(top) == len(bottom) >= 1, "the span is proportional and never empty"

    # A tree that FITS carries no track. A gutter drawn on a list that cannot
    # scroll is a scrollbar for a document that fits, and it would break the
    # same rule the remainder cues keep: the absence of the signal has to mean
    # "this is everything".
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("only answer"), pilot)
        assert not screen._gutter_drawn(screen._card_width())
        assert not any("│" in line or "█" in line for line in screen.render_lines_for_test())

    # A card wide enough to draw but not to spare the gutter's two cells. The
    # hint is what sets the floor (it never truncates), so the shed case needs
    # rows carrying real hints rather than bare labels.
    blocky = "An answer.\n\n" + "\n".join(f"```py\nx{i}\n```\n" for i in range(4)) + "\n> quote\n"
    app = _real_app()
    async with app.run_test(size=(44, 24)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(blocky), pilot)
        assert screen.is_drawable, "shedding the gutter, not hiding the card"
        assert not screen._gutter_drawn(screen._card_width(), rows=1)
        assert "esc quit" in "\n".join(screen.render_lines_for_test())


# --- resize ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_resize_clamps_the_preview_offset_and_drops_the_wrap_cache() -> None:
    """Both are width-dependent. A stale offset would show the preview at a
    position the document no longer has, and old-width cache entries would sit
    there for the life of the screen."""
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(120))
        screen = await _open(app, _targets(body), pilot)
        screen._scroll_preview_to(100)
        await pilot.pause()
        assert screen._wrap_cache, "the preview was wrapped at least once"

        app.post_message(events.Resize(screen.size, screen.size))
        await pilot.resize_terminal(70, 20)
        await pilot.pause()
        await pilot.pause()

        assert screen._preview_offset <= max(0, 120 - 1)
        keys = {key[1] for key in screen._wrap_cache}
        assert len(keys) <= 1, f"entries from more than one width survived: {keys}"
        # And the frame is still coherent at the new size.
        assert screen.render_lines_for_test()


@pytest.mark.asyncio
async def test_nothing_here_makes_the_screen_scrollable() -> None:
    """A scrollbar on this app is always a bug and silently costs two cells of
    width, reflowing the transcript behind the overlay. Everything the cues,
    the gutter and the preview scroll add is drawn INSIDE the one existing
    `Static`; the preview "scroll" is an offset into a `Text`, not a
    container."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(300))
        screen = await _open(app, _targets(body, *[f"answer {i}" for i in range(30)]), pilot)
        for keys in (("down",), ("shift+down",), ("end",), ("home",)):
            await pilot.press(*keys)
            await pilot.pause()
            assert screen.virtual_size.height <= screen.size.height, keys
            assert not screen.show_vertical_scrollbar, keys


# --- honesty in the frame ----------------------------------------------------


@pytest.mark.asyncio
async def test_an_empty_block_says_so_rather_than_refusing_silently() -> None:
    """Enter REFUSES an empty block, and that refusal is right — but refusing
    silently reads as a broken key, and the reasoning behind the refusal is
    itself that a deliberate action which does nothing and says nothing is the
    failure to avoid.

    Marking the row is the better fix than a message: the refusal becomes
    PREDICTED by the frame rather than explained after the fact, and a row
    visibly marked `empty` does not invite the double-click either. The
    refusal itself is unchanged.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("Here:\n\n```py\n```\n"), pilot, chosen.append)
        row = next(line for line in screen.render_lines_for_test() if "Block 1" in line)
        assert "empty" in row and "0 lines" not in row

        screen._move_to(1)
        await pilot.press("enter")
        await pilot.pause()
        assert chosen == [], "the refusal still stands"
        assert screen.is_active


@pytest.mark.asyncio
async def test_a_terminal_too_small_for_the_card_says_so() -> None:
    """From the user's chair, `/copy` on a small terminal was "I typed a
    command and my screen went dim and blank".

    The card's two-sided contract is untouched and is asserted here too: the
    notice is a SIBLING of the card, so `render_lines_for_test` still reports
    `[]`, `Copy to clipboard` is still absent, and `esc` still works.
    """
    app = _real_app()
    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets(CODE_ANSWER), pilot, chosen.append)
        assert not screen.is_drawable
        assert screen.render_lines_for_test() == []

        strips = app.screen._compositor.render_strips()
        frame = "\n".join("".join(segment.text for segment in strip) for strip in strips)
        assert TOO_SMALL_NOTICE in frame
        assert "Copy to clipboard" not in frame

        await pilot.press("escape")
        await pilot.pause()
        assert chosen == [None]


@pytest.mark.asyncio
async def test_the_wrap_cache_serves_the_same_rows_it_would_have_computed() -> None:
    """The memo's key is `(target.id, width, window_start)` and the claim is
    that it is TOTAL: `CopyTarget` is frozen, the tree is a snapshot that
    cannot change while the screen is open, and those three are the only other
    inputs. So a hit must be byte-identical to a recomputation — asserted
    rather than assumed, because the cost of being wrong is a stale preview
    beside an honest header.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(80))
        screen = await _open(app, _targets(body), pilot)
        target = screen.selected_target()
        assert target is not None

        cached_rows, cached_map = screen._wrap_preview(target, 88, 0)
        screen._wrap_cache.clear()
        fresh_rows, fresh_map = screen._wrap_preview(target, 88, 0)
        assert [row.plain for row in cached_rows] == [row.plain for row in fresh_rows]
        assert cached_map == fresh_map


# --- review round 1 regressions ----------------------------------------------


@pytest.mark.asyncio
async def test_a_double_click_copies_the_row_the_first_click_selected() -> None:
    """The BLOCKER from round 1 (U1/Q1), and the one that put wrong bytes on a
    real clipboard: aimed at `Quote 1`, received `Answer 10 body text.`

    `_window_start` centres the cursor, so click 1 recentres the tree UNDER A
    STATIONARY POINTER and click 2 resolves the same screen coordinate against
    a different window. It covered ~41% of the pane on any overflowing tree —
    the lower half, which is exactly where a user clicks after scanning a list
    top to bottom.

    Driven through Textual's own `pilot.click(times=2)` rather than posted
    messages, because the pilot is what produces a real chained click, and
    aimed BELOW `tree_rows / 2` on an overflowing tree: a version of this test
    that clicks row 0 passes against the broken code.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bodies = [f"Answer {i} body text.\n\n> Quote of answer {i}." for i in range(16, 0, -1)]
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets(*bodies), pilot, chosen.append)

        tree_rows, _ = screen._split_rows()
        assert len(screen.visible_rows) > tree_rows, "the tree must overflow to recentre"
        pane_row = tree_rows - 3
        assert pane_row > tree_rows // 2, "aimed in the band that used to copy the wrong row"

        start = screen._window_start(tree_rows)
        aimed = screen.visible_rows[start + pane_row].target
        x, y = _tree_row(screen, pane_row)

        await pilot.click(offset=(x, y), times=2)
        await pilot.pause()
        await pilot.pause()

        assert len(chosen) == 1
        assert chosen[0] is not None
        assert (
            chosen[0].id == aimed.id
        ), f"copied {chosen[0].label!r}, user aimed at {aimed.label!r}"


@pytest.mark.asyncio
async def test_a_wheel_off_the_card_does_nothing_at_all() -> None:
    """Round 1, U3/Q2. `_pointer_row_of` returns `None` off the body and the
    notch used to fall through to `action_move`, which reset the preview — so
    a pointer ONE COLUMN outside the card moved the cursor and silently
    discarded the reading position. At 100x30 that zone is three cells left of
    the preview text being read, and trackpad drift of three cells is routine.

    Off the card is now inert, exactly as it already was for the click.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(200))
        # Seeded LAST so the long answer is row 0 (most-recent-first), which is
        # what makes a preview offset reachable at all.
        screen = await _open(app, _targets("second answer", body), pilot)

        screen._scroll_preview_to(60)
        await pilot.pause()
        assert screen._preview_offset == 60

        region = screen._body.region
        row = region.y + screen._preview_top_row + 1
        outside = [
            (region.x - 1, row),
            (region.x + region.width, row),
            (region.x + 4, region.y + region.height),
        ]
        for x, y in outside:
            screen.post_message(_wheel(screen, x, y, down=True))
            await pilot.pause()
            assert screen._preview_offset == 60, f"the page was lost at {(x, y)}"
            assert screen._selected == 0, f"the cursor moved at {(x, y)}"


@pytest.mark.asyncio
async def test_a_move_that_changes_nothing_keeps_the_reading_position() -> None:
    """Round 1, U2 — the one KEYBOARD-ONLY regression against main. The offset
    was reset BEFORE the clamp, so `up`/`home`/`pageup` at the top row (a total
    visual no-op) threw away the reading position with no way back: 90
    `shift+down` to recover. On a single long answer the tree is ONE row, so
    every arrow press was such a no-op.

    A move that DOES change the selection still resets, because the preview is
    then a different document — that contract is unchanged and is pinned by
    `test_the_preview_offset_resets_on_every_cursor_move`.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(200))
        screen = await _open(app, _targets(body), pilot)
        assert len(screen.visible_rows) == 1, "the single-node case U2 bites hardest on"

        screen._scroll_preview_to(60)
        await pilot.pause()
        assert screen._preview_offset == 60

        for key in ("up", "home", "pageup", "down", "end", "pagedown"):
            await pilot.press(key)
            await pilot.pause()
            assert screen._selected == 0, f"{key} moved the cursor on a one-row tree"
            assert screen._preview_offset == 60, f"{key} moved nothing but lost the page"


@pytest.mark.asyncio
async def test_the_footer_advertises_the_scroll_whenever_the_pane_overflows() -> None:
    """Round 1, MAJOR-1: the footer's gate and the pane's marker were computed
    in DIFFERENT UNITS — source lines against wrapped rows. On ordinary prose
    (one long source line per paragraph, the normal shape of an answer) the
    pane overflowed with FEWER source lines than rows, so the frame drew
    `… N more lines` while the footer — the only place `shift+↑↓` is ever
    named — declined to say the preview scrolled, in one screenshot.

    The fixture is deliberately one whose two units DISAGREE: 11 source lines
    wrapping to 17 rows in a 16-row pane. A fixture of short lines passes
    against the broken predicate.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        prose = "\n\n".join(
            "This is a long paragraph of ordinary assistant prose written as one "
            "single source line, which therefore wraps across several rows of the "
            f"preview pane when it is painted at the card's width. Paragraph {i}."
            for i in range(6)
        )
        screen = await _open(app, _targets(prose), pilot)

        assert screen._preview_source_lines() <= screen._preview_rows, (
            "the fixture must have FEWER source lines than rows, or the old "
            "source-line predicate would have been right by accident"
        )
        lines = screen.render_lines_for_test()
        assert any("more lines" in line for line in lines), "the pane overflows"
        assert (
            PREVIEW_HINT in lines[-1]
        ), f"the pane says it overflows and the footer disagrees: {lines[-1]!r}"


@pytest.mark.asyncio
async def test_a_preview_that_fits_does_not_scroll_at_all() -> None:
    """Round 1, MAJOR-2/U5. The clamp ceilinged at `source_lines - 1`
    regardless of whether the document already fitted, so a 3-line preview in a
    16-row pane scrolled its own content off the top and left blank rows — the
    "a scroll gesture that teleports reads as the list resetting itself"
    failure AGENTS.md warns about. `less` stops with the last line at the
    bottom; so does the tree's own `down`; so does this now.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("alpha\nbeta\ngamma"), pilot)
        assert screen._preview_source_lines() < screen._preview_rows, "it fits"

        before = screen.render_lines_for_test()
        for _ in range(6):
            await pilot.press("shift+down")
        await pilot.pause()
        for _ in range(6):
            screen.post_message(_wheel(screen, *_preview_row(screen), down=True))
        await pilot.pause()

        assert screen._preview_offset == 0, "a document that fits has nowhere to go"
        assert screen.render_lines_for_test() == before
        assert PREVIEW_HINT not in before[-1], "and the footer does not advertise it"


@pytest.mark.asyncio
async def test_a_long_preview_stops_with_its_last_line_at_the_foot() -> None:
    """The other end of the same clamp: scrolling a document that DOES overflow
    still reaches its last line, and stops there rather than sliding the text
    up into blank rows (round 1, U5 measured 33 rows of read text pushed off
    screen at 200x50)."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(121))
        screen = await _open(app, _targets(body), pilot)

        for _ in range(300):
            await pilot.press("shift+down")
        await pilot.pause()

        lines = screen.render_lines_for_test()
        assert any("line 120" in line for line in lines), "the last line is on screen"
        assert not any("more lines" in line for line in lines), "and nothing remains below"

        # The pane must still be FULL. This is the assertion the finding turns
        # on: the old ceiling reached the same last line but with fifteen of
        # sixteen rows blank beneath it, having pushed everything already read
        # off the top. `less` leaves the screen full; so does this.
        top = screen._preview_top_row
        pane = lines[top + 1 : top + 1 + screen._preview_rows]
        blank = sum(1 for line in pane if not line.strip())
        assert blank <= 1, (
            f"{blank} of {len(pane)} preview rows are empty — the document was "
            f"scrolled off its own pane"
        )
        assert sum(1 for line in pane if line.strip()) >= screen._preview_rows - 1


@pytest.mark.asyncio
async def test_the_hover_highlight_follows_the_window_under_a_resting_pointer() -> None:
    """Round 1, U4/Q3. `on_mouse_move` was the only thing that updated
    `_hovered`, but the wheel and the click move rows under a stationary
    pointer and a real terminal sends NO `MouseMove` while only the wheel
    turns — so the highlight was left on a row the pointer was not over, and
    enough notches scrolled it off the pane entirely while the hand remained.

    Cosmetic in isolation (the click resolves by coordinate, not by the
    highlight), but the hover highlight is the ENTIRE affordance for the mouse
    here by design, and a stale one confirmed the wrong aim in the exact frame
    where U1 was still catchable.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bodies = [f"Answer {i} body text.\n\n> Quote of answer {i}." for i in range(16, 0, -1)]
        screen = await _open(app, _targets(*bodies), pilot)

        pane_row = 3
        x, y = _tree_row(screen, pane_row)
        screen.post_message(_move(screen, x, y))
        await pilot.pause()

        def _row_under_pointer() -> int:
            tree_rows, _ = screen._split_rows()
            return screen._window_start(tree_rows) + pane_row

        assert screen._hovered == _row_under_pointer()

        for notch in range(8):
            screen.post_message(_wheel(screen, x, y, down=True))
            await pilot.pause()
            assert (
                screen._hovered == _row_under_pointer()
            ), f"after notch {notch + 1} the highlight is on a row the pointer is not over"

        # And the same under a click that recentres the tree.
        screen.post_message(_click(screen, x, y))
        await pilot.pause()
        assert screen._hovered == _row_under_pointer()


@pytest.mark.asyncio
async def test_the_thumb_reaches_each_end_exactly_when_that_cue_goes() -> None:
    """Round 1 D11, and round 2 MAJOR-1/D16 — the SAME invariant at both ends.

    D11: `_thumb_span` anchored on the window's fraction of the LIST and then
    clamped, so the clamp won one window position early — the thumb sat at the
    bottom of the track while the rule still read `↓ 1 more`, two distinct
    positions painted an identical gutter, and the user's final arrow press
    moved the one CONTINUOUS signal not at all.

    Round 1 fixed that with a floored proportion and asserted only the bottom
    half. **That is why this test had to change shape rather than gain a
    fixture.** The floor moved the contradiction to the TOP — `top == 0` for a
    RANGE of early windows, so at the very 19-row shape D11 was filed against,
    `start=0` and `start=1` painted an identical gutter while the cue changed
    from `↓ 7 more` to `↑ 1 more ↓ 6 more`. Enumerated, floor scored 0 bottom
    contradictions and 427,381 top ones. The round-1 guard could not see any
    of it: it asserted `at_bottom == nothing_below` and nothing else, and all
    four of its fixtures VIOLATE the top invariant on the code they pass
    against. A guard that structurally cannot observe the defect it is named
    for is worse than no guard, so both directions are pinned here now:

        thumb flush with the bottom of the track ⟺ nothing below
        thumb flush with the top of the track    ⟺ nothing above

    **The row counts are chosen because they DISCRIMINATE.** The round-1
    formula is accidentally correct at many shapes — at 32 rows in a 12-row
    pane it has no violation at all — so a single convenient fixture passes
    against broken code. 19 is the shape D11 and D16 were both reported
    against; 15, 26 and 40 are the other published failures, 40 breaking at
    three consecutive positions. Each of the four also fails the TOP assertion
    on the round-1 floor (at `start=1`; 26 at 1–2 and 40 at 1–3), so the same
    fixtures discriminate in the new direction — verified by reverting.
    """
    for total_rows in (15, 19, 26, 40):
        app = _real_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            bodies = [f"answer {i}" for i in range(total_rows)]
            screen = await _open(app, _targets(*bodies), pilot)
            tree_rows = screen._tree_rows
            total = len(screen.visible_rows)
            max_start = total - tree_rows
            assert max_start > 0, f"{total_rows} must overflow the pane"

            for start in range(max_start + 1):
                span = screen._thumb_span(tree_rows, start)
                at_bottom = span.stop == tree_rows
                nothing_below = start == max_start
                assert at_bottom == nothing_below, (
                    f"{total} rows in {tree_rows}: start={start}/{max_start} thumb "
                    f"{span.start}..{span.stop}, {total - start - tree_rows} rows still below"
                )
                at_top = span.start == 0
                nothing_above = start == 0
                assert at_top == nothing_above, (
                    f"{total} rows in {tree_rows}: start={start}/{max_start} thumb "
                    f"{span.start}..{span.stop}, {start} rows still above"
                )
            # And it actually travels rather than sitting still.
            first = screen._thumb_span(tree_rows, 0)
            last = screen._thumb_span(tree_rows, max_start)
            assert first.start < last.start, f"the thumb never moved at {total} rows"


@pytest.mark.asyncio
async def test_a_track_too_short_to_hold_a_proportion_is_shed() -> None:
    """Round 1, D14. At 100x14 the split gives the tree ONE row with eighteen
    hidden, and a one-row track can only paint a thumb that FILLS it — the
    visual language for "everything fits" — beside a `↓ 18 more` cue saying
    the opposite. Forced by geometry rather than by D11's anchor, so fixing
    D11 does not fix it. The cue does the work alone instead."""
    app = _real_app()
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(19)]), pilot)
        assert screen._tree_rows < 3, "the geometry D14 was reported against"
        assert not screen._gutter_drawn(screen._card_width())

        lines = screen.render_lines_for_test()
        assert not any("█" in line or "│" in line for line in lines), "no contradictory thumb"
        assert any("more" in line and "─" in line for line in lines), "the cue still says so"


@pytest.mark.asyncio
async def test_both_tree_cues_read_as_remainders_and_hold_their_column() -> None:
    """Round 1, D12 and U9, which are one shape between them.

    D12: `↑ 6` is a bare number where `↓ 7 more` is a remainder, and a bare N
    reads as an ordinal ("row 6") — the exact hazard that ruled out an `N of M`
    indicator on this card. `↑ N more` fits the slot already reserved.

    U9: the cue moved COLUMN as its digit count changed (measured stepping at
    9→10 on a 39-row tree), which is precisely what brief item C1 asked to
    avoid. The count is now right-aligned in a field sized on the largest value
    it can reach, so the digits line up and nothing twitches under a user who
    is only scrolling.
    """
    app = _real_app()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(39)]), pilot)

        columns: set[int] = set()
        seen_up = False
        for index in range(len(screen.visible_rows)):
            screen._move_to(index)
            await pilot.pause()
            for line in screen.render_lines_for_test():
                if "↑" in line and "─" in line:
                    seen_up = True
                    assert "more" in line, f"the up cue reads as an ordinal: {line.strip()[-12:]!r}"
                    columns.add(line.index("↑"))
                if "↓" in line and "─" in line:
                    columns.add(line.index("↓"))

        assert seen_up, "the fixture must scroll far enough to raise the up cue"
        assert len(columns) == 1, f"the cue changed column as its digits did: {columns}"


@pytest.mark.asyncio
async def test_enter_on_an_empty_block_is_audible_rather_than_silent() -> None:
    """Round 1, U6. Marking the row shipped and does real work — the refusal is
    PREDICTED by the frame — but the audible half did not, so a user who had
    not read the hint column pressed Enter and got nothing at all: no notice,
    no bell, no frame change, which is an application that appears to have
    stopped responding to the key. That was the original complaint.

    `bell` rather than a toast: a toast would demand a dismissal for a keypress
    the user can simply repeat elsewhere, and this screen raises no notices of
    its own.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("Here:\n\n```py\n```\n"), pilot, chosen.append)

        rings: list[int] = []
        app.bell = lambda: rings.append(1)  # type: ignore[method-assign]

        empty = next(
            index for index, node in enumerate(screen.visible_rows) if not node.target.content
        )
        screen._move_to(empty)
        await pilot.press("enter")
        await pilot.pause()

        assert rings == [1], "the refusal is audible"
        assert chosen == [], "and it is still a refusal"
        assert screen.is_active

        # A row with content copies, and does NOT ring.
        screen._move_to(0)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert rings == [1], "a successful copy is not an error"
        assert len(chosen) == 1


@pytest.mark.asyncio
async def test_the_too_small_notice_keeps_esc_when_it_cannot_keep_the_rest() -> None:
    """Round 1, U8 and NIT-1. The notice is pinned to ONE row in the
    stylesheet, so below its own width it wrapped and the pin clipped it: `esc`
    — the ACTIONABLE half — sheds first, leaving `terminal too small for /copy
    ·`, a dangling separator that reads as a rendering fault. The footer
    already solves this by keeping `esc quit` last; the notice now does the
    same with two forms.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        chosen: list[CopyTarget | None] = []
        screen = await _open(app, _targets("first", "second"), pilot, chosen.append)

        for width in (40, 30, 20, 16):
            await pilot.resize_terminal(width, 8)
            await pilot.pause()
            await pilot.pause()
            assert not screen.is_drawable, f"{width} columns must be too small"
            text = str(screen._too_small.content)
            assert "esc" in text, f"the way out was shed at {width} columns: {text!r}"
            assert cell_len(text) <= width, f"the notice wraps at {width}: {text!r}"
            assert not text.rstrip().endswith("·"), f"dangling separator at {width}"

        await pilot.press("escape")
        await pilot.pause()
        assert chosen == [None], "and esc really does work throughout"


@pytest.mark.asyncio
async def test_a_resize_keeps_the_reading_position_instead_of_dropping_it() -> None:
    """Round 1, U7. `on_resize` clamped the offset and then called `_move_to`,
    which reset it to 0 regardless, so the clamp could not be observed and
    someone widening their terminal to read a long block more comfortably was
    thrown back to line 1. Falls out of the U2 fix; asserted so it stays out.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(201))
        screen = await _open(app, _targets(body), pilot)

        screen._scroll_preview_to(40)
        await pilot.pause()
        assert screen._preview_offset == 40

        for size in ((80, 24), (140, 40), (100, 30)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            await pilot.pause()
            assert screen._preview_offset > 0, f"the position was dropped resizing to {size}"
            assert screen._preview_offset <= 40


@pytest.mark.asyncio
async def test_the_thumb_holds_both_ends_across_every_shape_it_can_draw() -> None:
    """Round 2, MAJOR-1/D16 — the same invariant as the test above, ENUMERATED.

    The test above drives the real app, so it can afford four shapes; this one
    calls `_thumb_span` directly and sweeps the whole space both reviewers
    scored, which is what turns "0 violations" from a claim into a measurement.
    Both are kept: the four-shape version proves the invariant holds on a card
    that is actually laid out and painted, this one proves no shape escapes it.

    Restricted to `rows >= MIN_GUTTER_TRACK_ROWS`, because below that the
    gutter is SHED rather than drawn (round 1, D14) and an undrawn thumb
    cannot contradict a cue.

    The five residuals are `travel == 1` with `max_start > 1` — a one-cell
    travel has two expressible positions for three or more window states, so
    by pigeonhole one must share a cell with an extreme. They are asserted
    EXACTLY rather than tolerated: the set is pinned by shape, so a formula
    that regressed a sixth position would fail here even though the count is
    non-zero, and the forfeited end is asserted to be the top.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("a", "b"), pilot)

        top_bad: list[tuple[int, int, int]] = []
        bottom_bad: list[tuple[int, int, int]] = []
        states = 0
        for rows in range(MIN_GUTTER_TRACK_ROWS, 60):
            for total in range(rows + 1, 400):
                screen._flat = [None] * total  # type: ignore[assignment]
                max_start = total - rows
                for start in range(max_start + 1):
                    span = screen._thumb_span(rows, start)
                    states += 1
                    if (span.stop == rows) != (start == max_start):
                        bottom_bad.append((rows, total, start))
                    if (span.start == 0) != (start == 0):
                        top_bad.append((rows, total, start))

        assert states > 3_800_000, f"the sweep must be a sweep: {states} states"
        assert bottom_bad == [], f"the thumb bottoms out away from the list's end: {bottom_bad[:5]}"
        assert top_bad == [
            (3, 5, 1),
            (3, 6, 1),
            (3, 6, 2),
            (4, 6, 1),
            (5, 7, 1),
        ], f"unexpected top-end residual: {top_bad[:8]}"


@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 44)])
@pytest.mark.asyncio
async def test_a_fully_scrolled_preview_ends_on_a_full_pane_at_every_length(size) -> None:
    """Round 2, BLOCKER-1/U11 — MAJOR-2's clamp did not survive the wrap budget.

    `_preview_tail_offset` walked only the LAST wrap window. `_wrap_window_start`
    snaps to `PREVIEW_WRAP_STRIDE`, so a document of `100k + 1 … 100k + rows`
    lines has a final window holding fewer lines than the pane draws: the walk
    ran out of MAP before it ran out of PANE and fell through to `window_start`,
    which for that band IS the `source_lines - 1` ceiling round 1 removed. A
    fully scrolled 205-line answer ended on five lines of text above eleven
    blank rows — the exact frame MAJOR-2 was filed against, restored by its own
    fix, on this surface's headline case (one long answer).

    **Enumerated across lengths AND sizes because both reviewers found it that
    way and neither found it by spot-checking.** The band is `preview_rows`
    wide, so it widens with the terminal — measured on the pre-fix code at
    17/60 sampled lengths at 80x24, 30/60 at 100x30 and 43/60 at 140x44 — and
    a fixture at one size and one length is exactly the guard that missed it.
    The round-1 tests used 40- and 201-line documents and asserted `0 < offset
    <= 40`, never that the end frame was FULL, which is why this asserts the
    painted pane rather than the offset.

    The lengths deliberately straddle two stride boundaries and include the
    lengths either side, so a fix that merely special-cased one boundary fails.
    """
    lengths = (
        [PREVIEW_WRAP_BUDGET - 1, PREVIEW_WRAP_BUDGET]
        + list(range(PREVIEW_WRAP_BUDGET + 1, PREVIEW_WRAP_BUDGET + 32))
        + [3 * PREVIEW_WRAP_STRIDE + 1, 3 * PREVIEW_WRAP_STRIDE + 6, 4 * PREVIEW_WRAP_STRIDE + 1]
    )
    app = _real_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        drained = []
        for length in lengths:
            body = "\n".join(f"line {index}" for index in range(length))
            screen = await _open(app, _targets(body), pilot)

            # Past any reachable end, so this is the CEILING and not a stop.
            screen._scroll_preview_to(10**6)
            await pilot.pause()

            lines = screen.render_lines_for_test()
            top = screen._preview_top_row
            pane = lines[top + 1 : top + 1 + screen._preview_rows]
            blank = sum(1 for line in pane if not line.strip())
            if blank > 1:
                drained.append((length, screen._preview_offset, blank, len(pane)))
            assert any(
                f"line {length - 1}" in line for line in pane
            ), f"{length} lines: the last line is not on screen"
            app.pop_screen()
            await pilot.pause()

        assert not drained, "a fully scrolled preview drained its own pane at " + ", ".join(
            f"{n} lines (offset {o}, {b}/{p} rows blank)" for n, o, b, p in drained
        )


@pytest.mark.asyncio
async def test_a_wheel_or_a_key_between_the_clicks_refuses_rather_than_copying() -> None:
    """Round 2, U10. The U1 anchor binds the copy to the row the FIRST click
    resolved, which is right while nothing moves in between. But Textual breaks
    a click chain on only two things — a changed screen offset and the clock —
    and a wheel notch changes neither, nor does a keypress. So the second click
    still arrived as `chain=2`, the anchor fired, and the CLIPBOARD got the
    pre-wheel row while the caret and preview had moved on: measured 5/5
    disagreements with a wheel between the clicks, 5/5 with a key, 0/5 plain.

    That is the same silent failure U1 was filed for — the clipboard receives
    something other than what the frame promised — through a narrower but
    entirely natural gesture: click a row, realise it is not the one, nudge the
    wheel, click again.

    **The refusal is the fix, not a re-hit-test.** Re-resolving the row under
    the pointer was measured first: because a wheel over the tree moves the
    cursor and `_window_start` recentres it, the pointer's row equals the
    previewed row in only 1 of 9 pane positions, so that fallback copies a
    third row that is neither aimed at nor shown. A disturbed chain therefore
    selects and previews without copying, exactly as a chain broken by the
    clock already does.

    Asserted on the CLIPBOARD PAYLOAD, not on the anchor attribute: the defect
    was what the user got, and an attribute assertion would pass against a
    fix that cleared the anchor and copied the wrong row anyway.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        bodies = [f"Answer {index} unique body." for index in range(20, 0, -1)]

        async def _drive(interlude: str, pane_row: int) -> tuple[CopyTarget, CopyTarget | None]:
            got: list[CopyTarget | None] = []
            screen = await _open(app, _targets(*bodies), pilot, got.append)
            x, y = _tree_row(screen, pane_row)

            screen.post_message(_click(screen, x, y, chain=1))
            await pilot.pause()
            if interlude == "wheel":
                screen.post_message(_wheel(screen, x, y, down=True))
                await pilot.pause()
            elif interlude == "key":
                await pilot.press("down")
                await pilot.pause()

            framed = screen.selected_target()
            assert framed is not None, "the frame must be previewing something to disagree with"
            screen.post_message(_click(screen, x, y, chain=2))
            await pilot.pause()
            await pilot.pause()
            copied = got[0] if got else None
            if screen.is_attached:
                app.pop_screen()
                await pilot.pause()
            return framed, copied

        # The plain double-click still copies, and copies what it aimed at.
        # Without this the "fix" could be `never copy`, which passes the rest.
        for pane_row in (0, 4, 8):
            framed, copied = await _drive("none", pane_row)
            assert copied is not None, f"a plain double-click at row {pane_row} did not copy"
            assert copied.content == framed.content, "the plain double-click copied a stale row"

        for interlude in ("wheel", "key"):
            for pane_row in (0, 2, 4, 6, 8):
                framed, copied = await _drive(interlude, pane_row)
                if copied is None:
                    continue  # refused, which is the fix's chosen behaviour
                got, shown = copied.content or "", framed.content or ""
                assert got == shown, (
                    f"a {interlude} between the clicks copied {got.splitlines()[0]!r} "
                    f"while the frame showed {shown.splitlines()[0]!r}"
                )


@pytest.mark.asyncio
async def test_the_too_small_notice_never_picks_a_form_wider_than_its_box() -> None:
    """Round 2, D17. `_repaint` chose the notice's form against
    `self._screen_size()[0]`, but the notice is laid out in the SCREEN'S
    CONTENT BOX, which `Screen { padding: 1 }` makes two cells narrower. Those
    agree once laid out; `_screen_size`'s two fallbacks do not — it answers
    `self.app.size` before layout resolves and a hardcoded (80, 24) on
    exception, both reporting more room than the notice has. At 34 and 35
    columns that selects the 34-cell long form for a 32- or 33-cell box, and
    it clips back to `terminal too small for /copy ·` — the dangling separator
    U8 removed.

    The design round kept a captured frame of that rendering but could not
    reproduce it across 22 repeats, so this asserts the PROPERTY rather than
    the race: whatever form is chosen must fit the box it is laid out in, at
    every width, and an unresolved measurement must fall back to the short
    form because it fits everywhere the long one does.
    """
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("body"), pilot)

        # Down to 17 columns, where the SHORT form still fits its box. Below
        # that neither form does and the notice truncates whatever it is given
        # — the 13–16 column residue UX round 2 recorded as a follow-up under
        # U8, which is about the short form's own width and not about D17's
        # choice between the two.
        for columns in (60, 40, 37, 36, 35, 34, 33, 32, 30, 24, 20, 18, 17):
            await pilot.resize_terminal(columns, 10)
            await pilot.pause()
            await pilot.pause()
            assert not screen.is_drawable, f"{columns} cols must be too small for the card"
            notice = screen._too_small
            painted = str(notice.content)
            box = notice.size.width
            assert box, f"{columns} cols: the notice has no resolved width"
            assert cell_len(painted) <= box, (
                f"{columns} cols: the notice is {cell_len(painted)} cells in a {box}-cell box "
                f"and will clip: {painted!r}"
            )
            assert not painted.rstrip().endswith("·"), f"dangling separator at {columns}"
            assert "esc" in painted, f"the way out was shed at {columns} columns"
            # And the widest form that FITS is the one chosen — the fix must
            # not degrade to "always short", which would pass everything above.
            if cell_len(TOO_SMALL_NOTICE) <= box:
                assert (
                    painted == TOO_SMALL_NOTICE
                ), f"{columns} cols: short form in a box that fits the long one"

        # THE DISCRIMINATING FIXTURE, and the reason this test exists at all.
        # On the laid-out path every source agrees (`app.size` 34 → `self.size`
        # 32 → the notice's box 32, measured at every width above), so no
        # resize can tell the old code from the new one: a test that only
        # swept widths would be exactly the guard that cannot see its own
        # defect. The discrepancy is `_screen_size`'s FALLBACK — with
        # `self.size` unresolved it answers `self.app.size`, the terminal,
        # which is two cells wider than the box `Screen { padding: 1 }` leaves
        # the notice.
        #
        # 34x8, not 34x10, and that is load-bearing: the same fallback feeds
        # `is_drawable`, and at height 10 the patched screen reports itself
        # DRAWABLE, so `_repaint` skips the notice branch and the assertion
        # reads a stale string that passes against either code. Height 8 is
        # below `MIN_CARD_INNER_ROWS` however the width is measured, so the
        # notice branch really runs. Verified: the pre-fix expression paints
        # the 34-cell long form into the 32-cell box here.
        await pilot.resize_terminal(34, 8)
        await pilot.pause()
        await pilot.pause()
        assert screen._too_small.size.width == 32, "the fixture's premise: a 32-cell box at 34 cols"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(type(screen), "size", property(lambda self: Size(0, 0)))
            assert screen._screen_size()[0] == 34, "the fallback reports the terminal, not the box"
            assert not screen.is_drawable, "the notice branch must actually run"
            screen._repaint()
            painted = str(screen._too_small.content)
        assert cell_len(painted) <= 32, (
            f"the unresolved-layout path chose a {cell_len(painted)}-cell form for a "
            f"32-cell box, which clips to a dangling separator: {painted!r}"
        )
        assert painted == TOO_SMALL_NOTICE_SHORT

        # And with NOTHING resolved it must still prefer the short form: it
        # fits everywhere the long one does, so an uncertain measurement must
        # not select the one that can clip.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(type(screen._too_small), "size", property(lambda self: Size(0, 0)))
            patch.setattr(type(screen), "size", property(lambda self: Size(0, 0)))
            screen._repaint()
        assert (
            str(screen._too_small.content) == TOO_SMALL_NOTICE_SHORT
        ), "with no resolved width the notice chose the form that can clip"
