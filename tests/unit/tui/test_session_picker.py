"""The `/resume` picker: names, filtering, cursor, and the id it hands back.

The picker replaced a block of `<hex id>  3h ago` rows printed into the
transcript. What that surface could not do — be navigated, be searched, and
answer with a choice — is what these tests pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual import events
from textual.app import App, ComposeResult

from local_operator.resume import (
    NAME_MAX_CHARS,
    NAME_SCAN_CHARS,
    SessionRow,
    recent_session_rows,
    session_name,
)
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.session_picker import (
    CARD_MAX_HEIGHT_FRACTION,
    CARD_PADDING_ROWS,
    NAME_MIN_CELLS,
    PAGE_ROWS_MAX,
    SessionPickerScreen,
    filter_rows,
    plan_columns,
    render_rows,
)

NOW = 1_000_000.0


def _row(session_id: str, name: str, age_s: float = 60.0) -> SessionRow:
    return SessionRow(id=session_id, mtime=NOW - age_s, name=name)


def _write_transcript(root: Path, session_id: str, entries: list[dict[str, object]]) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return directory


def _message(role: str, text: str, **payload: object) -> dict[str, object]:
    return {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {"kind": "message", "role": role, "content": [{"text": text}], **payload},
    }


# --- naming -----------------------------------------------------------------


def test_a_session_is_named_by_its_opening_user_message(tmp_path: Path) -> None:
    """The only per-session title on disk is what the user typed first."""
    directory = _write_transcript(
        tmp_path,
        "abc123",
        [_message("user", "Make an asteroids game"), _message("assistant", "sure")],
    )
    assert session_name(directory) == "Make an asteroids game"


def test_a_tool_result_never_names_the_conversation(tmp_path: Path) -> None:
    """``tool`` is also a four-character role and its content is command
    output; matching the role loosely would name sessions after directory
    listings."""
    directory = _write_transcript(
        tmp_path,
        "abc123",
        [
            _message("tool", "total 48\ndrwxr-xr-x  12 damian", tool_call_id="c1", tool_name="ls"),
            _message("user", "why is the build failing"),
        ],
    )
    assert session_name(directory) == "why is the build failing"


def test_a_multi_line_prompt_becomes_one_scannable_line(tmp_path: Path) -> None:
    """A prompt is usually several lines; the picker has one row per session."""
    directory = _write_transcript(
        tmp_path, "abc123", [_message("user", "fix   the parser\n\nit crashes on   empty input")]
    )
    assert session_name(directory) == "fix the parser it crashes on empty input"


def test_a_long_prompt_is_ellipsised_within_the_budget(tmp_path: Path) -> None:
    directory = _write_transcript(tmp_path, "abc123", [_message("user", "word " * 100)])
    name = session_name(directory)
    assert len(name) <= NAME_MAX_CHARS
    assert name.endswith("…")


def test_an_unreadable_or_empty_transcript_yields_no_name(tmp_path: Path) -> None:
    """A nameless row beats taking the picker down: this runs over every
    session directory, including ones a live session is still writing."""
    empty = _write_transcript(tmp_path, "empty", [])
    assert session_name(empty) == ""
    # A half-written final line is normal for a running session.
    partial = tmp_path / "sessions" / "partial"
    partial.mkdir(parents=True)
    (partial / "transcript.jsonl").write_text('{"id": "e1", "type": "mess', encoding="utf-8")
    assert session_name(partial) == ""
    assert session_name(tmp_path / "sessions" / "missing") == ""


def _message_with_image(text: str, data_chars: int) -> dict[str, object]:
    """A user message carrying a pasted image, text block first — the order
    ``Message.user(text, images)`` produces and the writer preserves."""
    return {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "user",
            "content": [
                {"text": text},
                {"data": "A" * data_chars, "mime_type": "image/png"},
            ],
        },
    }


def test_a_session_opening_with_a_screenshot_is_still_named(tmp_path: Path) -> None:
    """One pasted image puts the first line past the scan window, and the
    fragment used to be dropped — which left every session that begins with a
    screenshot reading `(unnamed session)` in the picker for the rest of its
    life. Measured on two real sessions whose first lines were 115,289 and
    733,034 characters.

    The window still bounds the read; what changed is that a fragment is mined
    for the opener instead of discarded, which is safe because the text block
    precedes the image data on the line.
    """
    directory = _write_transcript(
        tmp_path,
        "shot01",
        [_message_with_image("why does the resume picker forget my sessions", NAME_SCAN_CHARS * 2)],
    )
    with (directory / "transcript.jsonl").open(encoding="utf-8") as handle:
        assert len(handle.readline()) > NAME_SCAN_CHARS, "this case needs an oversized line"
    assert session_name(directory) == "why does the resume picker forget my sessions"


def test_a_fragment_is_never_named_after_the_image_it_carries(tmp_path: Path) -> None:
    """A name taken from base64 would be worse than no name. When the text block
    does NOT come first, the scan declines rather than reaching past the data."""
    directory = tmp_path / "sessions" / "shot02"
    directory.mkdir(parents=True)
    payload = {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "user",
            "content": [
                {"data": "A" * (NAME_SCAN_CHARS * 2), "mime_type": "image/png"},
                {"text": "this text is past the image"},
            ],
        },
    }
    (directory / "transcript.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert session_name(directory) == ""


def test_a_fragment_whose_text_is_cut_off_yields_no_name(tmp_path: Path) -> None:
    """A title cut mid-word reads like a bug. The value must close inside the
    window to be used at all, so an opener longer than the window is declined
    rather than truncated to whatever the read happened to reach."""
    directory = tmp_path / "sessions" / "shot03"
    directory.mkdir(parents=True)
    line = '{"id":"e1","ts":0,"type":"message","payload":{"kind":"message","role":"user"'
    line += ',"content":[{"text":"' + "word " * (NAME_SCAN_CHARS // 2)
    (directory / "transcript.jsonl").write_text(line, encoding="utf-8")
    assert session_name(directory) == ""


def test_a_fragment_from_a_non_user_opener_is_declined(tmp_path: Path) -> None:
    """The strict path matches ``role`` exactly so a tool result cannot name a
    session; the fragment path has to hold the same line."""
    directory = tmp_path / "sessions" / "shot04"
    directory.mkdir(parents=True)
    payload = {
        "id": "e1",
        "ts": 0,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "tool",
            "tool_call_id": "c1",
            "content": [{"text": "total 48"}, {"data": "A" * (NAME_SCAN_CHARS * 2)}],
        },
    }
    (directory / "transcript.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert session_name(directory) == ""


def test_rows_are_newest_first_and_carry_their_name(tmp_path: Path) -> None:
    older = _write_transcript(tmp_path, "older1", [_message("user", "the older one")])
    newer = _write_transcript(tmp_path, "newer1", [_message("user", "the newer one")])
    import os

    os.utime(older / "transcript.jsonl", (1_000, 1_000))
    os.utime(newer / "transcript.jsonl", (2_000, 2_000))
    rows = recent_session_rows(tmp_path)
    assert [row.id for row in rows] == ["newer1", "older1"]
    assert [row.name for row in rows] == ["the newer one", "the older one"]


# --- filtering --------------------------------------------------------------


def test_filtering_matches_name_or_id_and_preserves_order() -> None:
    """Order must not change under a filter: a row that moved under the cursor
    while the query grew would resume the wrong conversation."""
    rows = [_row("aaa1", "asteroids game"), _row("bbb2", "parser crash"), _row("ccc3", "asteroid")]
    assert [r.id for r in filter_rows(rows, "aster")] == ["aaa1", "ccc3"]
    assert [r.id for r in filter_rows(rows, "BBB")] == ["bbb2"]
    assert [r.id for r in filter_rows(rows, "")] == ["aaa1", "bbb2", "ccc3"]
    assert filter_rows(rows, "nothing here") == []


# --- rendering --------------------------------------------------------------


def test_a_row_shows_the_name_the_age_and_the_id() -> None:
    lines = [line.plain for line in render_rows([_row("abc123def456", "ship it")], 0, 74, NOW)]
    assert "ship it" in lines[0]
    assert "abc123def456" in lines[0]
    assert "1m ago" in lines[0]


def test_the_cursor_marks_exactly_one_row() -> None:
    rows = [_row("a1", "one"), _row("b2", "two"), _row("c3", "three")]
    lines = [line.plain for line in render_rows(rows, 1, 74, NOW)]
    assert [line.startswith("❯") for line in lines] == [False, True, False]


def test_an_unnamed_session_says_so_rather_than_rendering_a_blank() -> None:
    """An empty cell reads as a rendering fault; the row is still pickable."""
    line = render_rows([_row("abc123", "")], 0, 74, NOW)[0].plain
    assert "(unnamed session)" in line
    assert "abc123" in line


# --- the screen -------------------------------------------------------------


class _PickerHost(App[None]):
    """A host whose only job is to own the modal under test."""

    def __init__(self, rows: list[SessionRow]) -> None:
        super().__init__()
        self._rows = rows
        self.chosen: list[str | None] = []

    def compose(self) -> ComposeResult:
        return iter(())

    async def open_picker(self) -> SessionPickerScreen:
        screen = SessionPickerScreen(self._rows, NOW)
        self.push_screen(screen, self.chosen.append)
        return screen


async def _picker(rows: list[SessionRow], size: tuple[int, int] = (100, 30)):
    app = _PickerHost(rows)
    return app, size


@pytest.mark.asyncio
async def test_enter_answers_with_the_highlighted_session_id() -> None:
    """The whole point of the two-way surface: it hands a choice back."""
    rows = [_row("first1", "one"), _row("second", "two"), _row("third3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["second"]


@pytest.mark.asyncio
async def test_escape_answers_nothing_and_resumes_no_session() -> None:
    app = _PickerHost([_row("first1", "one")])
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.chosen == [None]


@pytest.mark.asyncio
async def test_typing_filters_the_list_and_enter_takes_the_match() -> None:
    """The ids are unmemorable and the names are not — with a hundred
    sessions, typing is how the right one is found."""
    rows = [_row("aaa111", "asteroids game"), _row("bbb222", "parser crash")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "parser":
            await pilot.press(char)
        await pilot.pause()
        assert screen.filter_query == "parser"
        assert [row.id for row in screen.visible_rows] == ["bbb222"]
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["bbb222"]


@pytest.mark.asyncio
async def test_backspace_widens_the_filter_again() -> None:
    rows = [_row("aaa111", "asteroids"), _row("bbb222", "parser")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "past":
            await pilot.press(char)
        await pilot.pause()
        assert screen.visible_rows == []
        for _ in range(2):
            await pilot.press("backspace")
        await pilot.pause()
        assert screen.filter_query == "pa"
        assert [row.id for row in screen.visible_rows] == ["bbb222"]


@pytest.mark.asyncio
async def test_the_cursor_clamps_instead_of_wrapping() -> None:
    """A Down at the bottom that returned to the top reads as the list having
    reset itself."""
    rows = [_row("a1", "one"), _row("b2", "two")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for _ in range(5):
            await pilot.press("down")
        await pilot.pause()
        assert screen.selected_index == 1
        for _ in range(5):
            await pilot.press("up")
        await pilot.pause()
        assert screen.selected_index == 0


@pytest.mark.asyncio
async def test_a_filter_that_empties_the_list_says_so_and_answers_nothing() -> None:
    """Enter on no match must not resume an arbitrary session."""
    app = _PickerHost([_row("aaa111", "asteroids")])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for char in "zzz":
            await pilot.press(char)
        await pilot.pause()
        assert "no session matches that filter" in "\n".join(screen.render_lines_for_test())
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == [None]


@pytest.mark.asyncio
async def test_a_long_list_pages_and_reports_its_position() -> None:
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(PAGE_ROWS_MAX * 3)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 40)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("end")
        await pilot.pause()
        assert screen.selected_index == len(rows) - 1
        text = "\n".join(screen.render_lines_for_test())
        # The last row is on screen, and the position is stated.
        assert f"session {len(rows) - 1}" in text
        assert f"of {len(rows)}" in text


@pytest.mark.asyncio
async def test_the_picker_names_every_session_it_offers() -> None:
    """The regression the whole change exists for: the list used to be ids."""
    rows = [_row("abc123def456", "review the usage endpoint")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
    assert "review the usage endpoint" in text
    assert "type to filter" in text  # the keys are advertised on the card


def _wheel(widget, *, down: bool):
    """A real ``MouseScrollDown``/``Up`` aimed at ``widget``.

    Posted rather than calling the handler directly: the wiring under test is
    Textual's ``on_mouse_scroll_*`` dispatch, and a direct call would pass
    even if the method were named something Textual never looks for.
    """
    kind = events.MouseScrollDown if down else events.MouseScrollUp
    return kind(
        widget=widget,
        x=1,
        y=1,
        delta_x=0,
        delta_y=1 if down else -1,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
    )


@pytest.mark.asyncio
async def test_the_mouse_wheel_moves_the_cursor_and_clamps() -> None:
    """A wheel notch moves a row. Clamped, never wrapping: a scroll gesture
    that teleported to the other end would read as the list resetting."""
    rows = [_row(f"id{index:02d}", f"session {index}") for index in range(5)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for _ in range(2):
            screen.post_message(_wheel(screen, down=True))
        await pilot.pause()
        assert screen.selected_index == 2
        for _ in range(9):
            screen.post_message(_wheel(screen, down=True))
        await pilot.pause()
        assert screen.selected_index == len(rows) - 1  # clamped at the bottom
        for _ in range(20):
            screen.post_message(_wheel(screen, down=False))
        await pilot.pause()
        assert screen.selected_index == 0  # clamped at the top, not wrapped


class _FakeScroll:
    """The one thing the handlers use from a scroll event."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_the_wheel_handlers_stop_the_event_so_the_transcript_stays_put() -> None:
    """The card floats over the conversation; an un-stopped wheel scrolls
    both surfaces for one gesture."""
    screen = SessionPickerScreen([_row("a1", "one")], NOW)
    down, up = _FakeScroll(), _FakeScroll()
    screen.on_mouse_scroll_down(down)
    screen.on_mouse_scroll_up(up)
    assert down.stopped and up.stopped


# --- measured geometry (the card reads the terminal) ------------------------


def test_a_narrow_card_drops_the_id_rather_than_cutting_it() -> None:
    """A cut hex id still LOOKS like a valid id, and it is the one field a
    user copies into `/resume <id>`. Columns are dropped, never truncated —
    the id first, then the age, and the name last (a prefix of a sentence is
    still recognisable)."""
    rows = [_row("abc123def456", "port the CLI to typer")]
    ages = ["3h ago"]
    wide_name, wide_age, wide_id = plan_columns(rows, 74, ages)
    assert wide_id == 12 and wide_age == 6

    # Too narrow for the id: it goes, the age stays, nothing is cut.
    _, age_col, id_col = plan_columns(rows, 34, ages)
    assert id_col == 0 and age_col == 6

    # Narrower still: the age goes too, and the name keeps its floor.
    name_col, age_col, id_col = plan_columns(rows, 20, ages)
    assert (age_col, id_col) == (0, 0)
    assert name_col >= NAME_MIN_CELLS


@pytest.mark.asyncio
async def test_the_card_never_renders_wider_than_the_terminal() -> None:
    """The first cut was a fixed 78 cells that a 70-column terminal simply
    clipped, amputating the id column mid-token."""
    rows = [_row(f"id{index:010d}", f"session number {index}") for index in range(20)]
    for width in (60, 70, 80, 100, 120, 190):
        app = _PickerHost(rows)
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            for line in screen.render_lines_for_test():
                assert cell_len(line) <= width, (width, line)


@pytest.mark.asyncio
async def test_a_short_terminal_loses_list_rows_not_the_way_out() -> None:
    """Chrome is reserved first. The footer is the only place the card says
    `esc cancel`, so a clip that ate it left no stated way out."""
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    for height in (16, 18, 22, 30, 48):
        app = _PickerHost(rows)
        async with app.run_test(size=(100, height)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            lines = screen.render_lines_for_test()
            assert "esc cancel" in lines[-1], (height, lines[-1])
            # And the whole card fits the share of the screen it may occupy.
            budget = int(height * CARD_MAX_HEIGHT_FRACTION) - CARD_PADDING_ROWS
            assert len(lines) <= budget, (height, len(lines), budget)


@pytest.mark.asyncio
async def test_the_cursor_is_always_on_a_row_the_card_actually_draws() -> None:
    """A fixed page size let the cursor sit on a row that was never rendered,
    and Enter then resumed a session the user could not see."""
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 16)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        for _ in range(20):
            await pilot.press("down")
        await pilot.pause()
        drawn = "\n".join(screen.render_lines_for_test())
        chosen = screen.visible_rows[screen.selected_index]
        assert chosen.name in drawn, (screen.selected_index, drawn)


# --- selection legibility ---------------------------------------------------


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_the_caret_is_muted_like_both_sibling_pickers(theme_name: str) -> None:
    """The caret is `muted`, never the violet meta ink.

    It was `label` — the ramp's violet for tips and skill labels — held in a
    local named `accent`, which put the one cool mark on a warm card and said
    "meta" where the frame meant "position". Asserted per ramp and against the
    TOKEN rather than a hex, because the defect was a token choice: violet is
    `#b48cd6` on dark and `#7c5a9e` on paper, and pinning one hex would let the
    other ramp keep the bug.
    """
    original = theme_mod.current_theme()
    theme_mod.set_theme(theme_name)
    try:
        rows = [_row("aaa111", "a named session"), _row("bbb222", "")]
        for selected in (0, 1):  # named and unnamed: one caret, one ink
            span = render_rows(rows, selected, 74, NOW)[selected].spans[0]
            # Rich types ``Span.style`` as ``str | Style``; a span this file
            # built carries the object, and parsing narrows it either way.
            style = span.style if isinstance(span.style, Style) else Style.parse(span.style)
            colour = style.color
            assert colour is not None and colour.triplet is not None
            assert colour.triplet.hex == theme_mod.semantic_color("muted")
            assert colour.triplet.hex != theme_mod.semantic_color("label")
            assert colour.triplet.hex != theme_mod.semantic_color("accent")
    finally:
        theme_mod.set_theme(original)


def test_selecting_an_unnamed_row_brightens_it_like_any_other() -> None:
    """Pinning the placeholder to the dim floor made a SELECTED unnamed row
    darker than every unselected named row, inverting the highlight."""
    rows = [_row("aaa1", ""), _row("bbb2", "a named session")]

    def name_colour(lines, index: int) -> str:
        colour = lines[index].spans[1].style.color
        assert colour is not None and colour.triplet is not None
        return colour.triplet.hex

    at_rest = render_rows(rows, 1, 74, NOW)  # the NAMED row is selected
    selected = render_rows(rows, 0, 74, NOW)  # the UNNAMED row is selected
    assert name_colour(selected, 0) != name_colour(at_rest, 0)
    # And selecting it does not make it dimmer than an unselected named row.
    assert name_colour(selected, 0) == theme_mod.semantic_color("muted")


def test_every_row_style_clears_the_dim_step() -> None:
    """`faint` is 1.49:1 against this card's raised ground — the ramp is
    calibrated against the app background, and an overlay lifts the ground
    without lifting the text."""
    faint = theme_mod.semantic_color("faint")
    lines = render_rows([_row("abc123def456", "a session")], 0, 74, NOW)
    for span in lines[0].spans:
        style = span.style
        if isinstance(style, str):  # rich allows a named style; ours are objects
            continue
        colour = style.color
        if colour is not None and colour.triplet is not None:
            assert colour.triplet.hex != faint, span


# --- mouse ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clicking_a_row_resumes_it() -> None:
    """The card invited the mouse in with the wheel; a list you can scroll
    with the mouse and cannot click is a half-built affordance."""
    rows = [_row("first1", "one"), _row("second", "two"), _row("third3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        body = screen.query_one("#session-picker-body")
        # Row 1 (the second session) sits under the header and the rule.
        await pilot.click(body, offset=(4, 3))
        await pilot.pause()
    assert app.chosen == ["second"]


@pytest.mark.asyncio
async def test_narrowing_the_filter_selects_the_first_match() -> None:
    """Clamping the old index landed the cursor on the LAST match, so Enter
    took the least related row still standing."""
    rows = [
        _row("aaa111", "alpha session"),
        _row("bbb222", "beta session"),
        _row("ccc333", "gamma session"),
    ]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("end")  # cursor on the LAST row
        await pilot.pause()
        assert screen.selected_index == 2
        for char in "session":  # matches all three
            await pilot.press(char)
        await pilot.pause()
        assert screen.selected_index == 0
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["aaa111"]


@pytest.mark.asyncio
async def test_the_card_ends_on_quiet_ground_then_its_meta_in_both_states() -> None:
    """The position and the key hints are the same kind of row — statements
    ABOUT the list, not entries in it — so they travel together at the bottom
    with one quiet row above the pair. Same grammar as the usage card; the two
    differ only by whether the position row exists. Emitting an EMPTY counter
    row left two blank rows and pushed the keys off the block."""
    many = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    few = [_row("only01", "the only session")]

    app = _PickerHost(many)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        scrolled = screen.render_lines_for_test()
    assert scrolled[-1].startswith("↑↓")  # keys last
    assert scrolled[-2].startswith("showing ")  # position above them
    assert scrolled[-3] == ""  # one quiet row above the pair
    assert scrolled[-4] != ""  # and the report right above that

    app = _PickerHost(few)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        fits = screen.render_lines_for_test()
    assert fits[-1].startswith("↑↓")
    assert fits[-2] == ""  # no position row, and NOT a second blank
    assert fits[-3] != ""


def test_a_single_entry_with_no_trailing_newline_still_names_the_session(
    tmp_path: Path,
) -> None:
    """The window's last line is dropped only when the read was TRUNCATED. A
    complete final line simply has no newline after it, and dropping that lost
    the name of any session whose transcript is one entry."""
    directory = tmp_path / "sessions" / "onlyone"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps(_message("user", "the only line, no newline")), encoding="utf-8"
    )
    assert session_name(directory) == "the only line, no newline"


def test_a_truncated_first_line_is_not_parsed_as_a_name(tmp_path: Path) -> None:
    """The other half of the same rule: a line the cap cut in half is a
    fragment, not a message, and must not be mined for a name."""
    directory = tmp_path / "sessions" / "huge"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps(_message("user", "x" * (NAME_SCAN_CHARS * 2))) + "\n",
        encoding="utf-8",
    )
    assert session_name(directory) == ""


# --- painted-frame checks (the real app, so the stylesheet actually applies) --
#
# The round-1 D2 test asserted `render_lines_for_test()` — the widget's OWN
# arithmetic — inside `_PickerHost`, which declares no CSS_PATH. It therefore
# agreed with the bug: the card computed rows the container then clipped. These
# mount the real `OperatorApp` and measure what was drawn.


async def _real_picker(rows, size):
    """Push the picker onto the REAL app so `local_operator.tcss` applies."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    return app, size


@pytest.mark.asyncio
async def test_the_footer_survives_at_every_height_on_the_real_stylesheet() -> None:
    """`max-height: 80%` resolves against the screen's CONTENT box, which
    `Screen { padding: 1 }` insets by two rows. Measuring the terminal instead
    over-counted the room, so Textual clipped the overflow off the bottom —
    silently, taking the footer (the only statement of `esc`) with it."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    for height in (14, 16, 18, 20, 23, 30, 48):
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, height)) as pilot:
            await pilot.pause()
            screen = SessionPickerScreen(rows, NOW)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.pause()
            card = screen.query_one(".session-picker")
            drawn = card.region.height
            composed = len(screen.render_lines_for_test()) + CARD_PADDING_ROWS
            assert composed <= drawn, (height, composed, drawn)


@pytest.mark.asyncio
async def test_clicking_the_chrome_or_the_backdrop_resumes_nothing() -> None:
    """A false positive here disposes the live session and reboots onto another
    one. The first cut resolved a footer click to session #12, the blank spacer
    to #10, and the dimmed backdrop beside the card to row 0."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(40)]
    chosen: list[str | None] = []
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen(rows, NOW)
        app.push_screen(screen, chosen.append)
        await pilot.pause()
        await pilot.pause()
        body = screen.query_one("#session-picker-body")
        region = body.region
        lines = screen.render_lines_for_test()
        footer_row = len(lines) - 1
        spacer_row = next(i for i, line in enumerate(lines) if line == "")

        class _At:
            def __init__(self, x: int, y: int) -> None:
                self.screen_x = x
                self.screen_y = y

        # Header, rule, spacer, counter and footer are all chrome.
        for row in (0, 1, spacer_row, footer_row):
            assert screen._index_at(_At(region.x + 4, region.y + row)) is None, row
        # The backdrop to the left of the card, on a row that IS a list row.
        assert screen._index_at(_At(max(0, region.x - 12), region.y + 3)) is None
        # And below the card entirely.
        assert screen._index_at(_At(region.x + 4, region.y + region.height + 2)) is None
        assert chosen == []


@pytest.mark.asyncio
async def test_the_card_never_outgrows_a_narrow_terminal() -> None:
    """The minimum width is a PREFERENCE; the terminal is not. Applying the
    floor unconditionally returned a 30-cell content box inside 4 cells of
    padding on a 30-column screen — a 38-wide card, rule and header cut."""
    rows = [_row("abc123def456", "a session")]
    for width in (24, 30, 34, 40, 50):
        app = _PickerHost(rows)
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            for line in screen.render_lines_for_test():
                assert cell_len(line) <= width, (width, cell_len(line), line)


@pytest.mark.asyncio
async def test_typing_a_long_filter_does_not_grow_the_card() -> None:
    """The body is `width: auto`, so an unbounded filter echo made the card
    grow — and shift, since it is centred — with every character typed, moving
    the list under the eye of the person searching it."""
    rows = [_row("abc123def456", "a session")]
    app = _PickerHost(rows)
    async with app.run_test(size=(80, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        before = max(cell_len(line) for line in screen.render_lines_for_test())
        for char in "a" * 60:
            await pilot.press(char)
        await pilot.pause()
        after = max(cell_len(line) for line in screen.render_lines_for_test())
    assert after == before, (before, after)


@pytest.mark.asyncio
async def test_a_narrow_painted_header_keeps_the_active_filter() -> None:
    """At 50 columns the old header spent the row on its static title and
    clipped ``filter asteroid`` — the only receipt that typing reached it."""
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    rows = [_row(f"id{index:04d}", f"asteroid session {index}") for index in range(40)]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(50, 20)) as pilot:
        await pilot.pause()
        screen = SessionPickerScreen(rows, NOW)
        app.push_screen(screen)
        await pilot.pause()
        for char in "asteroid":
            await pilot.press(char)
        await pilot.pause()
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())

    assert screen.filter_query == "asteroid"
    assert "filter asteroid" in painted, painted


def test_a_right_click_never_resumes_a_session() -> None:
    """The action behind a click disposes the live session and reboots; a
    context-menu click must not reach it."""
    screen = SessionPickerScreen([_row("a1", "one")], NOW)
    resumed: list[str] = []
    screen.dismiss = lambda value=None: resumed.append(value)  # type: ignore[assignment]

    class _RightClick:
        button = 3
        screen_x = 0
        screen_y = 0

    screen.on_click(_RightClick())
    assert resumed == []
