"""The attachment marker as a painted object, and clicking one to select it.

``[Image #1, 1568x200]`` used to render in the prose colour on the prose ground
— indistinguishable from something the user typed, in a field where everything
else IS something the user typed. These assert what the terminal was SENT,
because "does it stand out" is a question about cells and colours and not about
any attribute the widget carries.

Every paste goes to the APP, not to the widget: ``App.on_event`` forwards a
non-forwarded ``Paste`` to the focused widget, so posting straight to the widget
delivers it twice. ``test_paste_images.py`` pins that with a control test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PIL import Image
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets.text_area import Selection

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.editor import Editor
from tests.unit.tui.conftest import TCSS_PATH


class ChipHost(App[None]):
    """A composer under the REAL sheet and the real brand variables.

    The chip's colours are resolved by the stylesheet from ``$lo-*``, so a host
    with a convenient inline style would assert this file's opinion rather than
    the shipped one.
    """

    CSS_PATH = TCSS_PATH

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables

    def compose(self) -> ComposeResult:
        yield Editor()


@pytest.fixture
def theme_name(request: pytest.FixtureRequest) -> Iterator[str]:
    """Activate a ramp for the whole app, including its CSS variables.

    Set BEFORE the app is constructed: ``get_css_variables`` runs once at
    stylesheet load, so a switch afterwards leaves the sheet on the old ramp.
    """
    name = getattr(request, "param", "dark")
    original = theme_mod.current_theme()
    theme_mod.set_theme(name)
    try:
        yield name
    finally:
        theme_mod.set_theme(original)


def _png(path, width: int = 1568, height: int = 200) -> str:
    Image.new("RGB", (width, height), (30, 30, 40)).save(path)
    return str(path)


async def _paste(app: App[None], pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    await pilot.pause()
    await pilot.pause()


def cells(editor: Editor, y: int) -> list[tuple[str, str | None, str | None]]:
    """``(character, fg hex, bg hex)`` for every CELL of rendered row ``y``.

    One entry per cell rather than per segment: the questions here are all of
    the form "what colour is column 8", and a segment list makes the reader do
    the arithmetic that the assertion is supposed to be checking.
    """
    out: list[tuple[str, str | None, str | None]] = []
    for segment in editor.render_line(y):
        style = segment.style
        fg = style.color.get_truecolor().hex.lower() if style and style.color else None
        bg = style.bgcolor.get_truecolor().hex.lower() if style and style.bgcolor else None
        out.extend((character, fg, bg) for character in segment.text)
    return out


def grounds(editor: Editor, y: int, start: int, end: int) -> set[str | None]:
    """The distinct backgrounds under cells ``[start, end)`` of row ``y``."""
    return {bg for _, _, bg in cells(editor, y)[start:end]}


def at(editor: Editor, column: int) -> tuple[int, int]:
    """The widget-relative mouse offset that lands on document ``column``.

    ``TextArea`` ships ``padding: 0 1``, so the first character of the buffer
    is drawn one cell in from the widget's own left edge. Hard-coding the click
    offset as the column made every assertion here off by one, in the direction
    that looks exactly like a real span bug.
    """
    return column + editor.gutter.left, 0


def _relative_luminance(hex_colour: str) -> float:
    channels = []
    for offset in (0, 2, 4):
        value = int(hex_colour.lstrip("#")[offset : offset + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(one: str, other: str) -> float:
    first, second = _relative_luminance(one), _relative_luminance(other)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


# -- the chip -----------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("theme_name", ["dark", "light"], indirect=True)
async def test_a_marker_is_painted_unlike_the_prose_beside_it(tmp_path, theme_name) -> None:
    """The reported defect, in one assertion: the marker's cells and the cells
    of the words on either side of it were the same two colours."""
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("now")
        await pilot.pause()
        assert editor.text == "look at [Image #1, 1568x200] now"

        row = cells(editor, 0)
        prose = row[2]  # the "o" of "look"
        for column in range(8, 28):
            assert row[column][1] != prose[1], f"column {column} kept the prose ink"
            assert row[column][2] != prose[2], f"column {column} kept the prose ground"
        # Named, not merely "different": the chip is `signal` (the ramp's
        # file/reference hue) on `tint-attach`, and a silent drift to some
        # other pair would still satisfy an inequality.
        assert {(fg, bg) for _, fg, bg in row[8:28]} == {
            (theme_mod.semantic_color("signal"), theme_mod.semantic_color("tint-attach"))
        }
        # The prose is untouched either side of it.
        assert grounds(editor, 0, 0, 8) == {theme_mod.semantic_color("surface")}
        assert grounds(editor, 0, 28, 32) == {theme_mod.semantic_color("surface")}


@pytest.mark.asyncio
async def test_two_markers_are_chips_and_the_words_between_them_are_not(tmp_path) -> None:
    """Two attachments in one draft read as two objects, not as one long
    highlighted run — the words between them keep the field's ground."""
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("and ")
        await _paste(app, pilot, _png(tmp_path / "b.png", 640, 480))
        await pilot.pause()
        assert editor.text == "[Image #1, 1568x200] and [Image #2, 640x480] "

        chip = theme_mod.semantic_color("tint-attach")
        assert grounds(editor, 0, 0, 20) == {chip}
        assert grounds(editor, 0, 20, 25) == {theme_mod.semantic_color("surface")}
        assert grounds(editor, 0, 25, 44) == {chip}


@pytest.mark.asyncio
async def test_a_marker_that_soft_wraps_is_painted_on_both_rows(tmp_path) -> None:
    """The composer soft-wraps, and it wraps INSIDE a marker: the break lands
    on the space after the comma. A row-is-a-line assumption paints the head
    and abandons the tail, which is worse than painting nothing — half a chip
    reads as a rendering fault.
    """
    app = ChipHost()
    async with app.run_test(size=(30, 8)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at this one ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        await pilot.pause()
        assert editor.text == "look at this one [Image #1, 1568x200] "
        # The marker opens at column 17 and the field is 30 wide, so it cannot
        # fit on the row that starts it.
        assert editor.wrapped_document.get_offsets(0), "the line did not wrap at all"

        chip = theme_mod.semantic_color("tint-attach")
        head = [column for column, (_, _, bg) in enumerate(cells(editor, 0)) if bg == chip]
        tail = [column for column, (_, _, bg) in enumerate(cells(editor, 1)) if bg == chip]
        assert head, "the head of the marker was left unpainted on its own row"
        assert tail, "the tail of the marker was left unpainted after the wrap"
        # Contiguous runs, and between them they cover the whole token.
        assert head == list(range(head[0], head[-1] + 1))
        assert tail == list(range(tail[0], tail[-1] + 1))
        painted = "".join(
            [character for character, _, bg in cells(editor, 0) if bg == chip]
            + [character for character, _, bg in cells(editor, 1) if bg == chip]
        )
        assert painted.replace(" ", "") == "[Image#1,1568x200]"


@pytest.mark.asyncio
async def test_the_caret_inside_a_chip_is_still_visible(tmp_path) -> None:
    """The chip is opaque, so painting it over every cell swallowed the caret:
    parking the cursor inside the marker left twenty flat cells and no answer
    to "where does the next keystroke land". The caret's cell is carved out.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.selection = Selection((0, 6), (0, 6))
        await pilot.pause()

        row = cells(editor, 0)
        caret_ground = theme_mod.semantic_color("fg")
        assert row[6][2] == caret_ground, "the chip painted over the caret"
        assert grounds(editor, 0, 0, 6) == {theme_mod.semantic_color("tint-attach")}
        assert grounds(editor, 0, 7, 20) == {theme_mod.semantic_color("tint-attach")}


# -- clicking one -------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("column", [8, 9, 17, 27])
async def test_a_click_inside_a_marker_selects_exactly_the_marker(tmp_path, column) -> None:
    """Anywhere on the token, including its first and last cell, and the
    selection is the token — not a caret, and not a cell more."""
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        await pilot.pause()
        assert editor.text == "look at [Image #1, 1568x200] "

        await pilot.click(Editor, offset=at(editor, column))
        await pilot.pause()
        assert editor.selection == Selection((0, 8), (0, 28))


@pytest.mark.asyncio
@pytest.mark.parametrize("column", [7, 28])
async def test_a_click_beside_a_marker_places_a_caret_and_nothing_else(tmp_path, column) -> None:
    """The cells one before the opening bracket and one after the closing one
    belong to the neighbouring characters. Widening by one in either direction
    would make it impossible to put the caret against the token.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        await pilot.pause()

        await pilot.click(Editor, offset=at(editor, column))
        await pilot.pause()
        assert editor.selection == Selection((0, column), (0, column))


@pytest.mark.asyncio
@pytest.mark.parametrize("theme_name", ["dark", "light"], indirect=True)
async def test_a_selected_marker_stands_further_out_than_a_resting_one(
    tmp_path, theme_name
) -> None:
    """ "Brighten it to show it is selected" is a claim about DIRECTION, so this
    checks the direction: the selected chip's ground pulls further away from the
    field than the resting chip's does. Asserting a hex would pass just as
    happily with the two tokens swapped.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        await pilot.pause()
        resting = grounds(editor, 0, 8, 28)

        await pilot.click(Editor, offset=at(editor, 12))
        await pilot.pause()
        selected = grounds(editor, 0, 8, 28)

        assert len(resting) == len(selected) == 1
        resting_ground, selected_ground = resting.pop(), selected.pop()
        assert resting_ground is not None and selected_ground is not None
        field = theme_mod.semantic_color("surface")
        assert contrast(selected_ground, field) > contrast(resting_ground, field)
        # And it is not merely the ordinary text selection: prose selected by a
        # drag gets `edge`, which the chip must out-read rather than borrow.
        assert selected_ground != theme_mod.semantic_color("edge")


@pytest.mark.asyncio
async def test_clicking_a_marker_then_backspace_removes_exactly_it(tmp_path) -> None:
    """The whole point of selecting it. ``_delete_marker`` stands aside for a
    real selection, so the click makes the selection and the ordinary delete
    path removes the span — and the attachment goes with it, or the next
    submit would carry an image the text no longer names.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("now")
        await pilot.pause()
        assert len(editor.referenced_images()) == 1

        await pilot.click(Editor, offset=at(editor, 12))
        await pilot.pause()
        await pilot.press("backspace")
        await pilot.pause()
        assert editor.text == "look at  now"
        assert editor.referenced_images() == []


@pytest.mark.asyncio
async def test_a_drag_that_starts_inside_a_marker_stays_a_drag(tmp_path) -> None:
    """A press is not yet a click. Collapsing the range on mouse-up would make
    it impossible to select from inside a marker outwards, which is the gesture
    for "take this token and the words after it".
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("now")
        await pilot.pause()

        await pilot.mouse_down(Editor, offset=at(editor, 12))
        await pilot.hover(Editor, offset=at(editor, 31))
        await pilot.mouse_up(Editor, offset=at(editor, 31))
        await pilot.pause()
        assert editor.selection == Selection((0, 12), (0, 31))

        # And the paint follows the truth: the covered half of the chip is the
        # selected ground, the uncovered half is still resting.
        assert grounds(editor, 0, 8, 12) == {theme_mod.semantic_color("tint-attach")}
        assert grounds(editor, 0, 12, 28) == {theme_mod.semantic_color("tint-attach-hi")}


@pytest.mark.asyncio
async def test_a_hand_typed_duplicate_of_a_live_number_is_not_chipped(tmp_path) -> None:
    """Design round 17, D4. The chip and the send must agree.

    `referenced_images` sends one image however many times its number is
    written, so a second `[Image #1]` typed by hand is a reference to something
    already accounted for - and it carried its own dimensions, which the user
    can write as anything. Two chips claiming two attachments, one image sent.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = ChipHost()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.text = f"{editor.text} and again [Image #1, 999x999]"
        await pilot.pause()

        assert len(editor.referenced_images()) == 1
        # By SPAN, not by exact start: the chip omits the caret's own cell, so
        # a marker under the caret is painted in two runs with a hole.
        gutter = editor.gutter_width
        painted = {
            column
            for y in range(3)
            for start, end, _ in editor._marker_cells(y)
            for column in range(start - gutter, end - gutter)
        }
        first = editor.text.index("[Image #1")
        duplicate = editor.text.rindex("[Image #1")
        live = set(range(first, editor.text.index("]") + 1))
        stale = set(range(duplicate, len(editor.text)))
        assert painted & live, "the live marker lost its chip"
        assert not painted & stale, "the hand-typed duplicate was chipped"


@pytest.mark.asyncio
async def test_the_receipt_row_is_inked_as_chrome_not_as_the_user_s_prose() -> None:
    """Design round 17, D5. The receipt is excluded from a copy, so it must not
    look like the sentence above it - the drag lit three rows while the toast
    said two, and nothing in the frame explained which row was furniture."""
    from local_operator.tui import theme as theme_mod
    from local_operator.tui.widgets.transcript import UserBlock

    app = ChipHost()
    async with app.run_test(size=(100, 30)) as pilot:
        block = UserBlock("look at this", attachments=1)
        await app.mount(block)
        await pilot.pause()

        rendered = block._build()
        assert isinstance(rendered, Text)
        assert block._receipt_row is not None
        receipt = rendered.split("\n")[block._receipt_row]
        assert "image attached" in receipt.plain
        muted = theme_mod.semantic_color("muted")
        prose = theme_mod.semantic_color(UserBlock.TEXT_TOKEN)
        assert muted != prose, "the fixture cannot tell the two inks apart"
        inks = {
            span.style.color.name
            for span in receipt.spans
            if isinstance(span.style, Style) and span.style.color is not None
        }
        assert muted in inks, f"the receipt wears prose ink, not the receipt ink: {inks}"
        assert prose not in inks, f"the receipt still carries the prompt's ink: {inks}"


# -- the selection band, as RENDERED (design round 1, D1) ---------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("theme_name", ["dark", "light"], indirect=True)
async def test_selected_composer_text_is_legible_on_its_band(theme_name) -> None:
    """A selection must highlight the text, not erase it.

    Design round 1, D1. `Editor .text-area--selection` set only a background,
    so the selected run fell back to Textual's built-in `#e0e0e0` ink. On the
    light ramp that was `#e0e0e0` on `#e5e0d5` — 1.003:1, no contrast at all —
    and a triple-click made the whole draft read as deleted, which is the exact
    "my draft disappeared" conclusion the multi-click gesture exists to remove.

    Asserted from the CELLS the widget actually painted rather than from the
    ramp, because the defect was in what the rule left UNSAID: the tokens were
    fine and the rendered pair was not. Both ramps, since the dark one was
    already correct and must stay that way.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("summarise the ingest path please")
        await pilot.pause()
        await pilot.pause()

        editor.selection = Selection((0, 0), (0, 9))
        await pilot.pause()

        band = theme_mod.semantic_color("edge")
        ink = theme_mod.semantic_color("fg")
        selected = {(fg, bg) for _, fg, bg in cells(editor, 0)[:9]}
        assert selected == {
            (ink, band)
        }, f"the selected run is not prose ink on the selection band: {selected}"
        # Read from the ramp rather than from the cells: the assertion above has
        # already established they are the same pair, and `cells` types its
        # colours as optional because an unstyled segment carries none.
        ratio = contrast(ink, band)
        assert ratio >= 4.5, f"{theme_name}: selected text reads {ratio:.2f}:1 ({ink} on {band})"


# -- the marker under the CLICK CHAIN (agent review round 1, R1-1) ------------
#
# Every test above drives mouse-down/up or the keyboard, and all 98 of them
# passed while a double-click inside a chip selected the bare word `Image` —
# the same structural blind spot the composer-copy change diagnosed for its own
# drag helper, one file over. `pilot.click(times=...)` is what builds the chain
# a real terminal sends, so these are the fixtures that can see the defect at
# all.
async def _multi_click(pilot, editor: Editor, column: int, times: int) -> None:
    """A double/triple click at a document column, through the CLICK path."""
    await pilot.click(editor, offset=at(editor, column), times=times)
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_double_click_inside_a_chip_takes_the_whole_marker(tmp_path) -> None:
    """The marker is ATOMIC under the click chain, not just under backspace.

    Regression for agent review round 1, R1-1. A double-click inside
    `[Image #1, 1568x200]` ran a word span over the raw line and selected
    `Image` alone; typing over that selection left `[x #1, 1568x200]` in the
    draft, which no longer matches `IMAGE_MARKER`. `resolve_markers` then drops
    it, so the attachment is SILENTLY not sent and what remains is neither a
    chip nor prose — data loss reachable by double-clicking a chip.

    Chain 1 already did this correctly via `_on_mouse_up`; the new handler has
    to ask `_marker_span` the same question rather than inventing a second
    answer.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("here")
        await pilot.pause()
        assert editor.text == "look at [Image #1, 1568x200] here"

        # Column 10 is inside the word `Image`, which is what a user aiming at
        # the chip's label hits.
        await _multi_click(pilot, editor, editor.text.index("Image") + 2, times=2)

        assert editor.selected_text == "[Image #1, 1568x200]"


@pytest.mark.asyncio
async def test_typing_over_a_double_clicked_chip_removes_it_whole(tmp_path) -> None:
    """The user-visible half of R1-1: the edit after the gesture.

    Type-over is the commonest edit following a double-click, and it is where
    the fragment became data loss: the corrupted token stopped matching, the
    image left the prompt, and nothing said so. The whole marker goes, and the
    attachment goes with it — the contract `_delete_marker` already keeps for
    backspace.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("here")
        await pilot.pause()

        await _multi_click(pilot, editor, editor.text.index("Image") + 2, times=2)
        await pilot.press("x")
        await pilot.pause()

        assert editor.text == "look at x here"
        # No hanging fragment: nothing that still looks like a marker survives.
        assert "#1" not in editor.text
        assert editor.attachments() == {}


@pytest.mark.asyncio
async def test_a_triple_click_over_a_chip_keeps_the_marker_intact(tmp_path) -> None:
    """Chain 3 takes the line, which CONTAINS the marker entire.

    Pinned beside the chain-2 case because the two reach the selection by
    different branches, and a line selection that happened to split the token
    would be the same silent drop by another route.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("here")
        await pilot.pause()

        await _multi_click(pilot, editor, editor.text.index("Image") + 2, times=3)

        assert editor.selected_text == "look at [Image #1, 1568x200] here"


@pytest.mark.asyncio
async def test_a_double_click_beside_a_chip_still_selects_the_word(tmp_path) -> None:
    """The marker gate must not swallow ordinary prose on the same row.

    The guard is scoped to the marker's own cells: a word elsewhere on a line
    that happens to contain a chip is still a word. Without this, "consult
    `_marker_span` first" could be implemented as "any line with a marker is
    atomic", which would break every draft that has an attachment in it.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("here")
        await pilot.pause()

        await _multi_click(pilot, editor, editor.text.index("here") + 1, times=2)

        assert editor.selected_text == "here"


@pytest.mark.asyncio
async def test_a_double_click_on_a_fragment_that_only_looks_like_a_marker(tmp_path) -> None:
    """Prose that matches the grammar is NOT atomic — the chip rule, unchanged.

    `_marker_span` protects what `cite()` chipped and nothing else: a
    hand-typed second citation is the user's own text, editable one character
    at a time (design round 18, D7). Asking `_marker_span` from the click
    handler inherits that distinction rather than restating it, and this pins
    that the gesture and the paint keep agreeing.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        await pilot.pause()
        # A second, hand-typed citation of the same number: prose, not a chip.
        editor.text = f"{editor.text}and again [Image #1, 999x999]"
        await pilot.pause()

        duplicate = editor.text.rindex("[Image #1")
        await _multi_click(pilot, editor, duplicate + 2, times=2)

        assert editor.selected_text == "Image"


@pytest.mark.asyncio
async def test_a_double_click_past_a_marker_line_keeps_the_attachment(tmp_path) -> None:
    """Clicking in the EMPTY SPACE right of a marker must not eat its bracket.

    Regression for agent review round 2, R2-1 — R1-1's exact failure mode,
    reintroduced by R1-1's own fix. The composer is wider than its text, so a
    click to the right of the line resolved to a column PAST the last
    character. `_marker_span` answered `None` there while `_word_span` clamped
    straight back into the line, and because `]` and the trailing space are
    both non-word characters `_word_pattern` merged them into one span: the
    gesture selected `'] '`, taking the closing bracket of an ATOMIC token.

    Typing over that corrupts the marker, so it stops matching `IMAGE_MARKER`,
    `resolve_markers` drops it, and THE IMAGE IS SILENTLY NOT SENT. Reachable
    from a bare paste with no editing at all, which is why this is pinned on the
    unedited draft rather than a constructed one.

    Pins the attachment surviving type-over, which is the user-visible harm;
    the span itself is pinned below.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        await pilot.pause()
        assert editor.text == "look at [Image #1, 1568x200] "
        assert len(editor.attachments()) == 1

        # Well past the end of the text, in the composer's own empty width.
        await _multi_click(pilot, editor, len(editor.text) + 6, times=2)
        await pilot.press("x")
        await pilot.pause()

        assert editor.attachments(), "the attachment was silently dropped"
        assert "[Image #1, 1568x200]" in editor.text, "the marker was corrupted"


@pytest.mark.asyncio
async def test_no_click_column_ever_splits_a_marker(tmp_path) -> None:
    """Sweep EVERY column, including past the line end, for a straddling span.

    The R2-1 defect lived in the region between `len(line) - 1` and the
    composer's right edge, which every earlier fixture missed because they all
    click INSIDE the marker. Sampling named columns is what let the suite stay
    green while the defect was live, so this asserts the invariant across the
    whole width instead: a span either takes the marker WHOLE or does not touch
    it, and never takes part of it.

    Both edges are swept deliberately. The closing edge is the reported case
    (`'] '`); the opening edge produced `' ['` by the same mechanism, one
    character the user typed plus one the app wrote.
    """
    app = ChipHost()
    async with app.run_test(size=(80, 6)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("look at ")
        await _paste(app, pilot, _png(tmp_path / "a.png"))
        editor.insert("here")
        await pilot.pause()

        line = editor.document[0]
        marker_start = line.index("[Image #1")
        marker_end = line.index("]", marker_start) + 1

        for column in range(len(line) + 8):
            clamped = min(column, max(len(line) - 1, 0))
            span = editor._marker_span(0, clamped, before=False)
            if span is None:
                span = editor._word_span(line, clamped, editor._marker_edges(0))
            start, end = span
            overlaps = max(start, marker_start) < min(end, marker_end)
            whole = start <= marker_start and end >= marker_end
            assert not overlaps or whole, (
                f"column {column} selected {line[start:end]!r}, which takes part "
                "of the marker without taking all of it"
            )
