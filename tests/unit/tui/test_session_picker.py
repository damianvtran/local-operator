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
    ORIGIN_SUBAGENT,
    SessionRow,
    mark_session_origin,
    recent_session_rows,
    session_name,
)
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.session_picker import (
    _MARKER_LEGEND,
    _SOFT_TIER_MIN_HITS,
    BODY_MATCH_MARKER,
    CARD_MAX_HEIGHT_FRACTION,
    CARD_PADDING_ROWS,
    GUTTER_CELLS,
    NAME_MIN_CELLS,
    PAGE_ROWS_MAX,
    PICKER_MIN_WIDTH,
    SessionPickerScreen,
    _footer_hints,
    filter_rows,
    matched_in_body,
    plan_columns,
    rank_rows,
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


def test_the_picker_offers_only_sessions_the_user_started(tmp_path: Path) -> None:
    """A subagent's child session is not a conversation the user can recognise.

    Children land in the same ``sessions/`` tree with the same shape, so the
    picker named them by their opening message — which for a delegated run is
    the role preamble the parent wrote. On one machine 40 of 50 offered rows
    were ``[role: reviewer] You are an INDEPENDENT reviewer…`` and the user's
    own sessions were paged off the bottom.
    """
    _write_transcript(tmp_path, "mine", [_message("user", "fix the resume picker")])
    child = _write_transcript(
        tmp_path, "child", [_message("user", "[role: reviewer] You are an INDEPENDENT reviewer")]
    )
    mark_session_origin(child, ORIGIN_SUBAGENT, label="reviewer")

    rows = recent_session_rows(tmp_path)
    assert [row.id for row in rows] == ["mine"]
    # Hidden from the listing, still on disk: the transcript is what makes a
    # stopped child resumable by id and readable after the fact.
    assert (child / "transcript.jsonl").is_file()


def test_a_session_with_no_marker_is_still_the_user_s(tmp_path: Path) -> None:
    """Every conversation that predates the marker must keep appearing.

    The filter reads absence as "the user's" precisely so an upgrade does not
    empty the picker of real work.
    """
    _write_transcript(tmp_path, "before", [_message("user", "an older conversation")])
    assert [row.id for row in recent_session_rows(tmp_path)] == ["before"]


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

    def __init__(self, rows: list[SessionRow], digests: dict[str, str] | None = None) -> None:
        super().__init__()
        self._rows = rows
        self._digests = digests
        self.chosen: list[str | None] = []

    def compose(self) -> ComposeResult:
        return iter(())

    async def open_picker(self) -> SessionPickerScreen:
        screen = SessionPickerScreen(self._rows, NOW, self._digests)
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


def test_the_empty_card_says_whose_sessions_are_missing() -> None:
    """ "no previous sessions to resume" was false when only children exist.

    Delegated runs share the sessions directory and are deliberately unlisted,
    and retention evicts an older parent before its newer children — so a
    machine can reach a state with resumable directories on disk and nothing
    the picker will offer. The old sentence stated a fact about the disk that
    was untrue, and named no way forward.
    """
    from local_operator.tui.widgets.session_picker import RESUME_EMPTY_NOTICE

    screen = SessionPickerScreen([], NOW)
    card = "\n".join(screen.render_lines_for_test())
    assert RESUME_EMPTY_NOTICE in card
    assert "subagent runs are not listed" in card


def test_one_session_is_not_announced_as_1_sessions() -> None:
    """Filtering makes a one-row list the common case rather than the rare
    one, so the header's plural now shows up routinely."""
    single = "\n".join(
        SessionPickerScreen([_row("aabbcc", "the only one")], NOW).render_lines_for_test()
    )
    assert "1 session" in single
    assert "1 sessions" not in single

    plural = "\n".join(
        SessionPickerScreen(
            [_row("aabbcc", "one"), _row("ddeeff", "two")], NOW
        ).render_lines_for_test()
    )
    assert "2 sessions" in plural


@pytest.mark.asyncio
async def test_the_empty_card_never_renders_wider_than_the_terminal() -> None:
    """The empty body is the one line with no truncation behind it.

    Every other row is bounded by the width the card MEASURES; the notice was
    a constant, so it satisfied the 74-cell ceiling and still overflowed the
    real card on any narrow terminal — at 60 columns it was cut to
    "…subagent runs are", losing the clause that explains why the list is
    empty, which is the whole reason the wording changed.

    Asserted against the TERMINAL width like the populated-rows guard above,
    never against ``PICKER_MAX_WIDTH``: the ceiling is 74 while an 80-column
    screen gives the card 70, so a constant-based assertion passes on a string
    that overflows.
    """
    for width in (60, 70, 80, 100, 120):
        app = _PickerHost([])
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            lines = screen.render_lines_for_test()
            for line in lines:
                assert cell_len(line) <= width, (width, line)
            # The explanation survives the narrow case rather than being the
            # first thing dropped: it is what makes the empty state honest.
            body = " ".join(line.strip() for line in lines)
            assert "subagent runs are not listed" in body, (width, body)


# --- searching the conversation body ----------------------------------------
# The reported failure: a session was findable only by the words in its NAME,
# so a conversation whose title did not happen to contain what the user
# remembered was unreachable. These pin the widened filter and the marker that
# keeps its results explicable.


def test_a_row_matches_on_its_conversation_body(tmp_path: Path) -> None:
    rows = [_row("aaa1", "vague title"), _row("bbb2", "another")]
    assert [r.id for r in filter_rows(rows, "retention", {"aaa1"})] == ["aaa1"]


def test_a_body_match_never_reorders_the_list() -> None:
    """Same invariant as every other filter here: a row that moved under the
    cursor while the query grew would resume the wrong conversation."""
    rows = [_row("aaa1", "one"), _row("bbb2", "two"), _row("ccc3", "three")]
    assert [r.id for r in filter_rows(rows, "topic", {"ccc3", "aaa1"})] == ["aaa1", "ccc3"]


def test_a_caller_without_an_index_keeps_the_old_behaviour() -> None:
    """Hosts with no index — tests, embedders — must not lose the filter."""
    rows = [_row("aaa1", "asteroids game"), _row("bbb2", "parser crash")]
    assert [r.id for r in filter_rows(rows, "aster")] == ["aaa1"]
    assert filter_rows(rows, "retention") == []


def test_only_a_body_match_is_marked() -> None:
    """A row whose visible name contains the query needs no explanation; one
    that does not would otherwise read as an arbitrary result."""
    assert matched_in_body(_row("aaa1", "vague"), "retention", {"aaa1"}) is True
    assert matched_in_body(_row("bbb2", "retention sweep"), "retention", {"bbb2"}) is False
    assert matched_in_body(_row("ccc3", "vague"), "retention", set()) is False
    assert matched_in_body(_row("aaa1", "vague"), "", {"aaa1"}) is False


def test_a_marked_row_still_occupies_exactly_one_row_width() -> None:
    """The marker is reserved as a column, so it cannot push the row past the
    card and silently eat the age and id columns."""
    row = _row("abc123def456", "a name long enough to need the whole budget here")
    plain = render_rows([row], 0, 74, NOW)[0].plain
    marked = render_rows([row], 0, 74, NOW, None, {"abc123def456"})[0].plain
    assert cell_len(marked) == cell_len(plain)
    assert BODY_MATCH_MARKER.strip() in marked
    assert BODY_MATCH_MARKER.strip() not in plain
    assert "abc123def456" in marked


def test_the_marker_never_pushes_the_name_below_its_floor() -> None:
    """The marker must not jump the queue in which the id and the age give up
    their cells before the name gives up any.

    Subtracting it from the name AFTER the budget was already spent down to
    the floor rendered marked names at 14 cells at several reachable widths.
    """
    rows = [_row("abc123def456", "a long conversation title"), _row("bbb222ccc333", "another")]
    for width in range(PICKER_MIN_WIDTH, 140):
        name_col, _, _ = plan_columns(rows, width, ["1m ago", "1h ago"], True)
        assert name_col >= NAME_MIN_CELLS, width


def test_every_row_starts_its_name_at_the_same_column() -> None:
    """Reserving the marker only on matched rows ragged the left edge of the
    one field the user reads down the list."""
    rows = [_row("abc123def456", "matched inside"), _row("bbb222ccc333", "not matched")]
    lines = [line.plain for line in render_rows(rows, 0, 74, NOW, None, {"abc123def456"})]
    # The marker column is reserved on BOTH rows, so each name begins at the
    # same offset: the marked row spends it on the mark, the other on blanks.
    assert lines[0].index("matched inside") == lines[1].index("not matched")


def test_a_marked_list_fills_the_card_at_every_reachable_width() -> None:
    rows = [_row("abc123def456", "a long conversation title"), _row("bbb222ccc333", "another")]
    for width in range(PICKER_MIN_WIDTH, 140):
        for line in render_rows(rows, 0, width, NOW, None, {"abc123def456"}):
            assert cell_len(line.plain) == width, width


@pytest.mark.asyncio
async def test_typing_finds_a_session_by_its_conversation_not_its_name() -> None:
    """End to end through the real screen: the query appears nowhere in the
    row's name, and the picker still hands that session back."""
    rows = [_row("aaa111", "a forgettable opening line"), _row("bbb222", "something else")]
    app = _PickerHost(rows, {"aaa111": "we fixed the retention sweep eviction"})
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        for char in "retention":
            await pilot.press(char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == ["aaa111"]


def test_the_marker_column_does_not_depend_on_what_is_scrolled_into_view() -> None:
    """A page is not the result set.

    Deciding the reservation from the rows currently ON SCREEN made it vanish
    when the one marked row scrolled off, so every name jumped two cells
    sideways on a single arrow press and truncation changed for rows that had
    not changed. Needs more rows than a page holds -- the earlier fixtures
    could not scroll, which is why this was missed.
    """
    rows = [_row(f"id{index:09d}", f"session number {index}") for index in range(6)]
    marked = {"id000000000"}  # only the first row matched inside its body
    with_marked = render_rows(rows[0:3], 0, 74, NOW, None, marked)
    without_marked = render_rows(rows[3:6], 0, 74, NOW, None, marked)
    assert with_marked[0].plain.index("session number 0") == (
        without_marked[0].plain.index("session number 3")
    )


def test_an_unfiltered_list_reserves_no_marker_column() -> None:
    """The reservation costs nothing when nothing matched, so an ordinary
    open of the picker renders exactly as it did before the marker existed.

    Asserted against the marked rendering rather than against another call
    with the same arguments: comparing a no-match render to a no-match render
    passes however the reservation behaves, which is a test that cannot fail.
    """
    rows = [_row("abc123def456", "one"), _row("bbb222ccc333", "two")]
    unmarked = render_rows(rows, 0, 74, NOW, None, set())[0].plain
    marked = render_rows(rows, 0, 74, NOW, None, {"abc123def456"})[0].plain
    # The name starts flush against the cursor gutter when nothing matched,
    # and exactly the marker's width later when something did.
    assert unmarked.index("one") == GUTTER_CELLS
    assert marked.index("one") == GUTTER_CELLS + cell_len(BODY_MATCH_MARKER)


# --- relevance ranking ------------------------------------------------------
# Ordering is a property of the QUERY, applied by rank_rows only when a query is
# active, so the "no reorder under the cursor" invariant holds: the only event
# that reorders (a query change) is the same one that re-homes the cursor.


def test_ranking_orders_name_above_body_above_soft() -> None:
    """A query hit in the visible name outranks one in the body, which outranks
    a soft-only match — the tiered rule the design specifies."""
    name_hit = _row("aaa1", "classifier tuning", age_s=300)  # matches by name
    body_hit = _row("bbb2", "unrelated title", age_s=200)  # exact body match
    soft_hit = _row("ccc3", "another title", age_s=100)  # soft-only match
    rows = [name_hit, body_hit, soft_hit]
    # body_matches carries only the exact-body id; the soft id is in `rows`
    # (already filtered) but not in body_matches, so it takes the soft tier.
    ranked = rank_rows(rows, "classifier", {"bbb2"})
    assert [r.id for r in ranked] == ["aaa1", "bbb2", "ccc3"]


def test_ranking_breaks_ties_by_recency_within_a_tier() -> None:
    """Two rows in the same tier keep newest-first order (stable tie-break)."""
    older = _row("aaa1", "classifier one", age_s=500)
    newer = _row("bbb2", "classifier two", age_s=100)
    # Passed newest-first, as the picker builds them; both match by name.
    ranked = rank_rows([newer, older], "classifier", set())
    assert [r.id for r in ranked] == ["bbb2", "aaa1"]


def test_an_empty_query_keeps_recency_order_unchanged() -> None:
    """No query means no ordering: the list stays exactly as it arrived."""
    rows = [_row("aaa1", "one"), _row("bbb2", "two"), _row("ccc3", "three")]
    assert rank_rows(rows, "") == rows
    assert rank_rows(rows, "   ") == rows


def test_ranking_is_stable_for_a_fixed_query() -> None:
    """A fixed query must produce byte-for-byte the same order every call, so a
    repaint or resize never moves a row under the cursor."""
    rows = [_row("aaa1", "classifier"), _row("bbb2", "unrelated"), _row("ccc3", "classify")]
    first = rank_rows(rows, "class", {"bbb2"})
    second = rank_rows(rows, "class", {"bbb2"})
    assert [r.id for r in first] == [r.id for r in second]


@pytest.mark.asyncio
async def test_the_cursor_re_homes_to_the_top_match_on_a_query_change() -> None:
    """set_query pins the cursor to rank 0, so it tracks the best match rather
    than sitting on a row ranking then slides away from."""
    rows = [_row("aaa1", "one"), _row("bbb2", "two"), _row("ccc3", "three")]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen._move_to(2)  # move the cursor off the top
        assert screen.selected_index == 2
        screen.set_query("three")
        await pilot.pause()
        # Cursor is re-homed to index 0 (the top match), not clamped to the
        # last surviving row.
        assert screen.selected_index == 0
        assert screen.selected_id() == "ccc3"


@pytest.mark.asyncio
async def test_visible_rows_is_stable_across_repaints_for_a_fixed_query() -> None:
    """The invariant end to end: repeated repaints under a FIXED query never
    reorder the visible rows."""
    rows = [_row("aaa1", "classifier"), _row("bbb2", "unrelated"), _row("ccc3", "classify")]
    digests = {"bbb2": "a body mentioning classifier deep inside"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("classifier")
        await pilot.pause()
        first = [r.id for r in screen.visible_rows]
        screen._repaint()
        screen._repaint()
        assert [r.id for r in screen.visible_rows] == first


@pytest.mark.asyncio
async def test_a_row_matched_only_by_a_soft_query_is_shown_and_marked() -> None:
    """A soft match (typo) surfaces the row via the body path and carries the
    body-match marker, since it did not match the visible name."""
    rows = [_row("aaa1", "vague title"), _row("bbb2", "another")]
    digests = {"aaa1": "improve adm classifier throughput"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("classifer")  # typo, soft match only
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == ["aaa1"]
        assert screen.body_matched_ids == {"aaa1"}


@pytest.mark.asyncio
async def test_a_row_matched_only_by_a_past_name_is_shown_and_marked() -> None:
    """A session found by a name it was renamed AWAY from surfaces via the body
    path (the digest folds past names in) and carries the marker."""
    rows = [_row("aaa1", "Current Title"), _row("bbb2", "another")]
    # The digest carries a PAST name the visible row name no longer shows.
    digests = {"aaa1": "Old Abandoned Name the session body text"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("abandoned")
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == ["aaa1"]
        assert screen.body_matched_ids == {"aaa1"}


def test_footer_legend_appears_only_when_a_row_is_marked() -> None:
    """D2: the ``"`` marker is meaningless without a legend, but advertising it
    when nothing is marked would explain a glyph the user cannot see. So the
    legend is present exactly when the footer has room AND a marked row exists,
    and absent otherwise."""
    # A card wide enough to hold the legend beside the essential keys.
    with_legend = _footer_hints(74, has_marked=True)
    assert _MARKER_LEGEND in with_legend
    # Same width, nothing marked: no legend, and the full key row is intact.
    without = _footer_hints(74, has_marked=False)
    assert _MARKER_LEGEND not in without
    assert ("pgup/pgdn", "page") in without


def test_footer_legend_drops_before_the_movement_and_action_keys() -> None:
    """The legend teaches; it must never crowd out the keys that OPERATE the
    card. Under width pressure it sheds after the two disposable hints but
    before movement/resume/cancel, and it never survives as a bare unlabelled
    glyph (which would be the very artifact-looking mark D2 flagged)."""
    # Wide: legend shown, and it displaced only a disposable hint (pgup/pgdn).
    wide = _footer_hints(74, has_marked=True)
    assert _MARKER_LEGEND in wide
    assert ("pgup/pgdn", "page") not in wide
    # Narrow: the essential keys survive and the legend is gone entirely — not
    # reduced to a lone glyph.
    narrow = _footer_hints(40, has_marked=True)
    assert _MARKER_LEGEND not in narrow
    assert (_MARKER_LEGEND[0], "") not in narrow
    assert ("enter", "resume") in narrow
    assert ("esc", "cancel") in narrow


@pytest.mark.asyncio
async def test_the_soft_tier_is_skipped_once_the_exact_tiers_fill_a_page() -> None:
    """The soft tier's first call tokenises every digest and builds a vocabulary
    over them — 324 ms and 95 MB resident over the 2681-digest store the picker
    now reaches uncapped, against ~5 ms for the exact tier.

    Since the picker went uncapped that build sat on the first character typed.
    It is deferred behind the cheap tiers: a query that already returns a
    screenful has nothing for soft matching to rescue, and the extra recall
    would land below the fold.
    """
    rows = [_row(f"s{i:03d}", f"classifier run {i}") for i in range(_SOFT_TIER_MIN_HITS + 5)]
    digests = {row.id: "body text" for row in rows}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        calls: list[str] = []
        real = screen._soft_index.search

        def counting(digests_arg, query):
            calls.append(query)
            return real(digests_arg, query)

        screen._soft_index.search = counting  # type: ignore[method-assign]
        screen.set_query("classifier")
        await pilot.pause()

        assert len(screen.visible_rows) == len(rows)
        assert calls == [], "the soft tier ran for a query the exact tiers already answered"


@pytest.mark.asyncio
async def test_the_soft_tier_still_runs_when_the_exact_tiers_come_up_short() -> None:
    """The other half of the gate: a typo the exact tiers cannot answer must
    still reach the soft tier, or deferring it would silently cost recall."""
    rows = [_row("aaa1", "vague title"), _row("bbb2", "another")]
    digests = {"aaa1": "improve adm classifier throughput"}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("classifer")  # typo: no exact hit anywhere
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == ["aaa1"]
        assert screen.body_matched_ids == {"aaa1"}


@pytest.mark.asyncio
async def test_the_header_tally_reports_the_stores_true_total() -> None:
    """Once the cap is gone ``_all`` IS the store's user-session total, so the
    counter stops reporting a number that is not the total and never said so.

    The filtered ``showing a-b of N`` counter keeps reporting the FILTERED
    count — the two answer different questions and must not converge.
    """
    rows = [_row(f"s{i:04d}", f"session {i}") for i in range(2700)]
    app = _PickerHost(rows)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert "2700 sessions" in "\n".join(screen.render_lines_for_test())

        screen.set_query("session 1")
        await pilot.pause()
        lines = "\n".join(screen.render_lines_for_test())
        shown = len(screen.visible_rows)
        assert f"of {len(rows)}" in lines  # header tally: the store total
        assert f"of {shown}" in lines  # position counter: the filtered count


@pytest.mark.asyncio
async def test_a_large_row_set_does_not_move_a_row_under_the_cursor() -> None:
    """R1 at the scale the uncapped picker now reaches.

    A fixed query must order rows identically across repaints AND a resize, and
    the row under the cursor must keep its identity — a row moving out from
    under the cursor is how a user resumes the wrong session.
    """
    rows = [_row(f"s{i:04d}", f"session {i} work") for i in range(2700)]
    digests = {row.id: f"body {row.id}" for row in rows}
    app = _PickerHost(rows, digests)
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        screen.set_query("session 12")
        await pilot.pause()
        first = [r.id for r in screen.visible_rows]
        screen.action_move(1)
        screen.action_move(1)
        selected = screen.selected_id()

        screen._repaint()
        screen._repaint()
        assert [r.id for r in screen.visible_rows] == first
        assert screen.selected_id() == selected

        # A resize repaints at a different width; ordering is a pure function of
        # the query, so it must survive that too.
        await pilot.resize_terminal(60, 30)
        await pilot.pause()
        assert [r.id for r in screen.visible_rows] == first
        assert screen.selected_id() == selected
