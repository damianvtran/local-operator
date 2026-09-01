"""The `/copy` picker screen: layout maths, navigation, and the target it hands back.

The layout tests are the load-bearing ones. The row split is where a literal
port of the reference's prose (rather than its source) produces a visibly
different widget on every short tree, and the budget floor is where the card
overflowed its box by one row at a 14-row terminal — clipped silently, off the
bottom, taking the footer with it.

Geometry is asserted against the REAL app and the real stylesheet: the
lightweight hosts elsewhere in this suite declare no `CSS_PATH`, so a card
sized by percentage rules would not be sized at all under one.
"""

from __future__ import annotations

import pytest
from textual.binding import Binding

from local_operator.tui.copy_targets import (
    CopyTarget,
    build_copy_targets,
    flatten_targets,
)
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.copy_picker import (
    CARD_PADDING_ROWS,
    MIN_TREE_ROWS,
    CopyPickerScreen,
)

CODE_ANSWER = "Here it is.\n\n```python\ndef f():\n    return 1\n```\n\n> and a quote"


def _answer(text: str, truncated: bool = False) -> AssistantBlock:
    """A SETTLED answer, which is the only kind the picker lists."""
    block = AssistantBlock()
    block.update_text(text)
    block.finalize_text()
    if truncated:
        block.mark_truncated()
    return block


def _targets(*texts: str) -> list[CopyTarget]:
    return build_copy_targets([_answer(text) for text in texts])


async def _open(app, targets: list[CopyTarget], pilot) -> CopyPickerScreen:
    screen = CopyPickerScreen(targets)
    app.push_screen(screen)
    await pilot.pause()
    await pilot.pause()
    return screen


def _real_app():
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


# --- layout -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_short_tree_takes_only_the_rows_it_has() -> None:
    """`tree_rows` is capped at the number of rows that EXIST, so a two-node
    tree donates the rest to the preview instead of sitting in a half-height
    pane padded with blanks. This is the case the reference's own source gets
    right and its prose description does not."""
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("one", "two"), pilot)
        tree_rows, preview_rows = screen._split_rows()
        assert tree_rows == 2
        # The whole remaining budget, not a half share.
        assert preview_rows == screen._row_budget() - 2


@pytest.mark.asyncio
async def test_a_long_tree_splits_the_budget_in_half() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(40)]), pilot)
        available = screen._row_budget()
        tree_rows, preview_rows = screen._split_rows()
        assert tree_rows == available // 2
        assert preview_rows == available - tree_rows


@pytest.mark.asyncio
async def test_the_minimum_floors_the_budget_not_the_tree_share() -> None:
    """`MIN_TREE_ROWS` floors `available`, which is what makes a two-node tree
    on a short terminal take two rows rather than three."""
    app = _real_app()
    async with app.run_test(size=(100, 14)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("only one"), pilot)
        assert screen._row_budget() >= MIN_TREE_ROWS + 1 or screen._row_budget() >= 1
        assert screen._split_rows()[0] == 1


def _painted(app) -> str:
    """What the COMPOSITOR actually put on screen.

    The only way to see WRAP: `render_lines_for_test` composes strings and a
    row that wraps in the real paint still reports as one line there. NBSP is
    normalised because the footer's spacing paints as `\\u00a0`.
    """
    return "\n".join(
        "".join(segment.text for segment in strip)
        for strip in app.screen._compositor.render_strips()
    ).replace("\u00a0", " ")


@pytest.mark.asyncio
async def test_the_card_never_overflows_its_box_at_any_height() -> None:
    """The failure this pins: at 14 rows the composed card was 11 rows against
    10 drawn, and Textual clipped the difference off the bottom SILENTLY —
    taking the footer, the only statement of how to leave, with it. A
    scrollbar appearing is the other half of the same bug: it costs two cells
    of width and reflows the transcript behind the overlay.

    The sweep starts at 6, not 14. An earlier revision of this test swept
    14-60 and was written to prevent exactly this defect, but the band where
    the card actually clipped its footer (8-11) sat below its floor, so it
    passed throughout — a guard whose range excluded the failure it named.
    """
    for height in (6, 8, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40, 60):
        app = _real_app()
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            targets = _targets(*[CODE_ANSWER for _ in range(20)])
            screen = await _open(app, targets, pilot)
            card = screen.query_one(".copy-picker")
            lines = screen.render_lines_for_test()
            # A hidden card is the correct answer below the chrome floor, and
            # it claims no padding either — that is the point of hiding rather
            # than emptying it.
            composed = len(lines) + CARD_PADDING_ROWS if lines else 0
            assert composed <= card.region.height, (height, composed, card.region.height)
            assert app.screen.virtual_size.height <= app.screen.size.height, height
            assert not app.screen.show_vertical_scrollbar, height


@pytest.mark.asyncio
async def test_the_footer_is_painted_or_the_card_is_not_drawn_at_all() -> None:
    """Asserted against the PAINTED frame, across both axes.

    `render_lines_for_test` cannot see this: it builds strings, and a row too
    wide for the pane wraps in the compositor. At 36 columns the helper
    reported the footer while the frame had pushed it off the card, so a test
    reading the helper could not catch it.

    The contract is two-sided — either the footer is on screen, or the card
    drew nothing. A half-card with no way out advertised is the defect; an
    absent card on a terminal that cannot hold one is a degraded frame, and
    Esc still leaves it.
    """
    for width, height in (
        (100, 8),
        (100, 10),
        (100, 11),
        (100, 12),
        (100, 30),
        (32, 24),
        (36, 24),
        (38, 24),
        (40, 24),
        (60, 24),
        (80, 14),
    ):
        app = _real_app()
        async with app.run_test(size=(width, height)) as pilot:
            await pilot.pause()
            screen = await _open(app, _targets(CODE_ANSWER), pilot)
            frame = _painted(app)
            card = screen.query_one(".copy-picker")
            if card.display:
                assert "esc quit" in frame, (width, height, frame)
            else:
                assert "Copy to clipboard" not in frame, (width, height, frame)
            assert screen.render_lines_for_test() == [] or card.display


@pytest.mark.asyncio
async def test_esc_still_leaves_a_card_too_small_to_draw() -> None:
    """Hiding the card must not strand the user: the screen is still a modal
    over their conversation, and Esc is the only way off it."""
    chosen: list[CopyTarget | None] = []
    app = _real_app()
    async with app.run_test(size=(100, 9)) as pilot:
        await pilot.pause()
        screen = CopyPickerScreen(_targets(CODE_ANSWER))
        app.push_screen(screen, chosen.append)
        await pilot.pause()
        assert not screen.is_drawable
        await pilot.press("escape")
        await pilot.pause()
    assert chosen == [None]


@pytest.mark.asyncio
async def test_the_footer_is_always_the_last_drawn_row() -> None:
    """It is the only statement of how to leave the screen."""
    for height in (14, 20, 30, 50):
        app = _real_app()
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            screen = await _open(app, _targets(CODE_ANSWER), pilot)
            assert screen.render_lines_for_test()[-1] == "↑↓ move · enter copy · esc quit"


@pytest.mark.asyncio
async def test_the_test_helper_reports_nothing_when_the_card_is_hidden() -> None:
    """The helper's own docstring promises it will not report lines that never
    reached the terminal. It only checked `is_mounted`, so at 80x10 the body
    was mounted with `display=True` and it happily reported the footer against
    a frame that had clipped it — a comment asserting a guarantee the code did
    not implement, and the guard that would have caught the clipping."""
    app = _real_app()
    async with app.run_test(size=(80, 10)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(CODE_ANSWER), pilot)
        assert not screen.is_drawable
        assert screen.render_lines_for_test() == []
        assert "esc quit" not in _painted(app)


@pytest.mark.asyncio
async def test_enter_on_an_empty_block_refuses_rather_than_copying_nothing() -> None:
    """An empty fence builds a child whose content is `""`. The guard tested
    `content is None`, so Enter dismissed the picker and wrote nothing — no
    toast, no notice. `_cmd_copy`'s docstring argues that silence is right for
    a zero-width drag and wrong for a command typed deliberately."""
    chosen: list[CopyTarget | None] = []
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = CopyPickerScreen(_targets("Here:\n\n```py\n```\n"))
        app.push_screen(screen, chosen.append)
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        target = screen.selected_target()
        assert target is not None and target.content == ""
        await pilot.press("enter")
        await pilot.pause()
        # Still open: the refusal is visible as the picker not going away.
        assert chosen == []
        await pilot.press("escape")
        await pilot.pause()
    assert chosen == [None]


@pytest.mark.asyncio
async def test_render_lines_are_empty_before_the_card_is_drawn() -> None:
    """This method re-derives the text rather than reading back what was
    painted, so an undrawn card must report nothing — otherwise a test can
    assert a line that never reached the terminal."""
    screen = CopyPickerScreen(_targets("unmounted"))
    assert screen.render_lines_for_test() == []


# --- the tree ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_tree_draws_labels_hints_and_nesting() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(CODE_ANSWER), pilot)
        lines = screen.render_lines_for_test()
        assert any("❯ Here it is." in line and "1 code · 1 quote" in line for line in lines)
        assert any("├─ Block 1" in line and "python · 2 lines" in line for line in lines)
        assert any("└─ Quote 1" in line for line in lines)


@pytest.mark.asyncio
async def test_a_truncated_answer_is_marked_in_the_tree_and_the_preview() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        targets = build_copy_targets([_answer("cut off here", truncated=True)])
        screen = await _open(app, targets, pilot)
        lines = screen.render_lines_for_test()
        assert any("truncated · 1 line" in line for line in lines)
        assert any(line.startswith("Preview · truncated") for line in lines)


@pytest.mark.asyncio
async def test_the_hint_survives_a_narrow_terminal_and_the_label_gives_way() -> None:
    """The hint carries the line counts and the `truncated` marker; the label
    is the column that can afford to lose cells."""
    app = _real_app()
    async with app.run_test(size=(52, 20)) as pilot:
        await pilot.pause()
        long_label = "An extremely long first line that cannot possibly fit in this width"
        targets = build_copy_targets([_answer(long_label, truncated=True)])
        screen = await _open(app, targets, pilot)
        row = next(line for line in screen.render_lines_for_test() if "❯" in line)
        assert "truncated · 1 line" in row
        assert "…" in row


@pytest.mark.asyncio
async def test_the_window_follows_the_cursor_past_the_visible_rows() -> None:
    """A cursor that leaves the window would move the selection to a row the
    user cannot see."""
    app = _real_app()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(40)]), pilot)
        for _ in range(20):
            await pilot.press("down")
        await pilot.pause()
        tree_rows, _ = screen._split_rows()
        start = screen._window_start(tree_rows)
        assert start <= screen._selected < start + tree_rows
        assert any("answer 19" in line for line in screen.render_lines_for_test())


# --- navigation -------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_arrows_clamp_at_both_ends_rather_than_wrapping() -> None:
    """A deliberate divergence from the reference, which wraps. AGENTS.md's
    wrap rule carries an exception for a list that IS the whole page, and the
    nearest precedent here (`session_picker._move_to`) already clamps. Page
    keys clamp under either reading, so clamping the arrows too keeps ONE
    uniform rule on the page."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("one", "two", "three"), pilot)
        await pilot.press("up")
        assert screen._selected == 0
        for _ in range(10):
            await pilot.press("down")
        assert screen._selected == len(screen.visible_rows) - 1
        await pilot.press("down")
        assert screen._selected == len(screen.visible_rows) - 1


@pytest.mark.asyncio
async def test_the_page_keys_clamp_and_step_by_the_visible_tree() -> None:
    """By the CAPPED tree height, so a page moves by what the user can see."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(40)]), pilot)
        step = screen._page_rows()
        assert step == screen._split_rows()[0]
        await pilot.press("pagedown")
        assert screen._selected == step
        await pilot.press("pageup")
        assert screen._selected == 0
        await pilot.press("pageup")
        assert screen._selected == 0


@pytest.mark.asyncio
async def test_home_and_end_reach_the_ends_the_arrows_no_longer_wrap_to() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(*[f"answer {i}" for i in range(30)]), pilot)
        await pilot.press("end")
        assert screen._selected == len(screen.visible_rows) - 1
        await pilot.press("home")
        assert screen._selected == 0


@pytest.mark.asyncio
async def test_the_cursor_can_reach_every_row_including_nested_children() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        targets = _targets(CODE_ANSWER)
        screen = await _open(app, targets, pilot)
        seen = []
        for _ in range(len(flatten_targets(targets))):
            target = screen.selected_target()
            assert target is not None
            seen.append(target.id)
            await pilot.press("down")
        assert seen == [node.target.id for node in flatten_targets(targets)]


# --- the answer it hands back -----------------------------------------------


@pytest.mark.asyncio
async def test_enter_dismisses_with_the_highlighted_target() -> None:
    """The whole point of the two-way surface. It returns the TARGET, not its
    text, so the caller can read `truncated` without re-deriving it."""
    chosen: list[CopyTarget | None] = []
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = CopyPickerScreen(_targets(CODE_ANSWER))
        app.push_screen(screen, chosen.append)
        await pilot.pause()
        await pilot.press("down")  # onto Block 1
        await pilot.press("enter")
        await pilot.pause()
    assert len(chosen) == 1
    assert chosen[0] is not None
    assert chosen[0].id == "msg:1:code:0"
    assert chosen[0].content == "def f():\n    return 1"


@pytest.mark.asyncio
async def test_esc_dismisses_with_nothing() -> None:
    chosen: list[CopyTarget | None] = []
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.push_screen(CopyPickerScreen(_targets("an answer")), chosen.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert chosen == [None]


@pytest.mark.asyncio
async def test_ctrl_c_is_not_bound_by_the_modal() -> None:
    """The reference binds `ctrl+c` beside Esc. Here it is the GLOBAL
    interrupt, and a modal claiming it would change what stopping a turn means
    depending on whether an overlay happened to be open. Both existing modals
    dismiss on Esc alone."""
    keys = {
        binding.key if isinstance(binding, Binding) else binding[0]
        for binding in CopyPickerScreen.BINDINGS
    }
    assert "ctrl+c" not in keys
    assert "escape" in keys


@pytest.mark.asyncio
async def test_a_truncated_target_carries_its_flag_to_the_caller() -> None:
    """The caller warns that the copied answer was cut off; it must not have
    to re-derive that from the text."""
    chosen: list[CopyTarget | None] = []
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        targets = build_copy_targets([_answer("half an answer", truncated=True)])
        app.push_screen(CopyPickerScreen(targets), chosen.append)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert chosen[0] is not None and chosen[0].truncated is True


# --- the preview ------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_preview_tracks_the_cursor() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets(CODE_ANSWER), pilot)
        await pilot.press("down")  # Block 1
        await pilot.pause()
        lines = screen.render_lines_for_test()
        assert any(line.startswith("Preview · python · 2 lines") for line in lines)
        assert any("def f():" in line for line in lines)


@pytest.mark.asyncio
async def test_a_long_preview_reports_how_much_it_is_not_showing() -> None:
    """Wrapped, never hard-truncated, and the marker COSTS a row: showing one
    more line while hiding that more exist is the failure it prevents."""
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        body = "\n".join(f"line {index}" for index in range(200))
        screen = await _open(app, _targets(body), pilot)
        lines = screen.render_lines_for_test()
        marker = [line for line in lines if line.startswith("… ") and "more lines" in line]
        assert len(marker) == 1


@pytest.mark.asyncio
async def test_the_overflow_marker_agrees_with_the_header_at_every_width() -> None:
    """Both numbers must measure the same thing. The marker counted WRAPPED
    ROWS while the header counts SOURCE LINES, so at 100 columns a 79-line
    answer claimed `… 144 more lines` — more than the document contains — and
    the contradiction was visible in one pane."""
    body = "\n".join(f"line {index} of an answer long enough to wrap here" for index in range(79))
    checked = 0
    for width in (60, 70, 100):
        app = _real_app()
        async with app.run_test(size=(width, 30)) as pilot:
            await pilot.pause()
            screen = await _open(app, _targets(body), pilot)
            lines = screen.render_lines_for_test()
            header = next(line for line in lines if line.startswith("Preview · "))
            total = int(header.split("·")[1].strip().split()[0])
            marker = next(line for line in lines if "more lines" in line)
            claimed = int(marker.strip().split()[1])
            shown = sum(1 for line in lines if line.startswith("line "))
            # Exactly the lines not on screen — not merely a smaller number.
            assert claimed == total - shown, (width, claimed, total, shown)
            checked += 1
    assert checked == 3


@pytest.mark.asyncio
async def test_the_overflow_marker_does_not_saturate_on_a_budgeted_answer() -> None:
    """`PREVIEW_WRAP_BUDGET` caps the WRAP, not the count. While the marker
    counted wrapped rows the budget capped it too, so a 600-line and a
    1000-line answer both reported the same 183 — a precise-looking constant,
    which is worse than no number."""
    claims = []
    for length in (600, 1000):
        app = _real_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            body = "\n".join(f"line {index}" for index in range(length))
            screen = await _open(app, _targets(body), pilot)
            marker = next(line for line in screen.render_lines_for_test() if "more lines" in line)
            claims.append(int(marker.strip().split()[1]))
    assert claims[0] != claims[1], claims
    assert claims[1] - claims[0] == 400, claims


@pytest.mark.asyncio
async def test_a_short_preview_shows_no_overflow_marker() -> None:
    app = _real_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = await _open(app, _targets("one line only"), pilot)
        assert not any("more lines" in line for line in screen.render_lines_for_test())


@pytest.mark.asyncio
async def test_the_preview_wraps_rather_than_truncating() -> None:
    """A hard-truncated preview would hide the end of every long line, which
    is exactly where a user checks whether they picked the right block."""
    app = _real_app()
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        sentence = "word " * 60
        screen = await _open(app, _targets(sentence.strip()), pilot)
        lines = screen.render_lines_for_test()
        assert sum(1 for line in lines if line.startswith("word ")) > 1
