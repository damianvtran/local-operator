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
from textual import events
from textual.binding import Binding

from local_operator.tui.copy_targets import CopyTarget, build_copy_targets
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.copy_picker import (
    HEADER_ROWS,
    PREVIEW_WRAP_BUDGET,
    TOO_SMALL_NOTICE,
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
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(40))
        screen = await _open(app, _targets(body), pilot)

        for _ in range(200):
            screen.post_message(_wheel(screen, *_preview_row(screen), down=True))
        await pilot.pause()
        assert screen._preview_offset == 39, "stops on the last source line"

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
    copy is left with a hand cursor over their transcript."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("first", "second"), pilot)
        screen.post_message(_move(screen, *_tree_row(screen, 0)))
        await pilot.pause()
        assert screen.styles.pointer == "pointer"

        screen.post_message(_click(screen, *_tree_row(screen, 0), chain=2))
        await pilot.pause()
        await pilot.pause()
        assert screen.styles.pointer == "default"


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

        # And the far end is reachable rather than floored at the budget.
        screen._scroll_preview_to(total)
        await pilot.pause()
        assert screen._preview_offset == total - 1
        assert any(f"line {total - 1}" in line for line in screen.render_lines_for_test())


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
            lines = screen.render_lines_for_test()
            return (
                any("↑ " in line and "─" in line for line in lines),
                any("more" in line and "─" in line for line in lines),
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
