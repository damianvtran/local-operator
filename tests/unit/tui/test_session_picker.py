"""The `/resume` picker: names, filtering, cursor, and the id it hands back.

The picker replaced a block of `<hex id>  3h ago` rows printed into the
transcript. What that surface could not do — be navigated, be searched, and
answer with a choice — is what these tests pin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual import events
from textual.app import App, ComposeResult

from local_operator.resume import (
    NAME_MAX_CHARS,
    SessionRow,
    recent_session_rows,
    session_name,
)
from local_operator.tui.widgets.session_picker import (
    PAGE_ROWS,
    SessionPickerScreen,
    filter_rows,
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
        assert "nothing matches" in "\n".join(screen.render_lines_for_test())
        await pilot.press("enter")
        await pilot.pause()
    assert app.chosen == [None]


@pytest.mark.asyncio
async def test_a_long_list_pages_and_reports_its_position() -> None:
    rows = [_row(f"id{index:04d}", f"session {index}") for index in range(PAGE_ROWS * 3)]
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
