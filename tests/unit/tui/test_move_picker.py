"""The `/move` picker screen: navigation, the two input modes, and geometry.

Geometry is asserted against the REAL app and the real stylesheet: the
lightweight hosts elsewhere in this suite declare no `CSS_PATH`, so a card
sized by percentage rules would not be sized at all under one — the same
reason `test_copy_picker` gives.

The mode tests are the load-bearing ones. This card has one input and two jobs
(filter the suggestions, complete a path), and the whole design rests on the
split being predictable from what was typed rather than from a key the user has
to know.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.tui.app import OperatorApp
from local_operator.tui.move_targets import MoveTarget
from local_operator.tui.widgets.move_picker import (
    NO_MATCH_NOTICE,
    NO_PATH_NOTICE,
    PAGE_ROWS_MAX,
    MovePickerScreen,
    _truncate_head,
    render_rows,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _target(path: str, kind: str = "recent", detail: str = "") -> MoveTarget:
    return MoveTarget(path=path, label=path, kind=kind, detail=detail)


def _targets(count: int = 4) -> list[MoveTarget]:
    rows = [_target("/here", "current", "current")]
    rows += [_target(f"/dir{index}", "recent", "session") for index in range(count - 1)]
    return rows


# -- pure rendering ----------------------------------------------------------


def test_a_long_path_is_truncated_from_the_LEFT() -> None:
    """The tail distinguishes two siblings; the head is what every row shares,
    so a tail cut renders two different directories as the same string."""
    cut = _truncate_head("/Users/damian/workspace/repos/lo-move-cmd", 20)
    assert cut.startswith("…")
    assert cut.endswith("lo-move-cmd")


def test_a_short_path_is_left_alone() -> None:
    assert _truncate_head("/tmp", 20) == "/tmp"


def test_the_note_is_dropped_before_the_path_is_cut() -> None:
    """The path is the thing being chosen; the note is only why it was
    offered, so on a narrow card the reason goes first."""
    wide = render_rows([_target("/a/very/long/directory/name", detail="session")], 0, 60)
    narrow = render_rows([_target("/a/very/long/directory/name", detail="session")], 0, 24)
    assert "session" in wide[0].plain
    assert "session" not in narrow[0].plain
    assert "name" in narrow[0].plain


def test_every_row_reserves_the_cursor_column() -> None:
    """So selecting a row cannot shift its text sideways."""
    rows = render_rows(_targets(3), 1, 60)
    assert rows[0].plain.startswith("  ")
    assert rows[1].plain.startswith("❯ ")


# -- navigation --------------------------------------------------------------


def test_movement_clamps_at_both_ends() -> None:
    """`session_picker._move_to` clamps and AGENTS.md's exception covers a full
    surface like this one: a Down at the bottom that returned to the top reads
    as the list having reset itself."""
    screen = MovePickerScreen(_targets(4))
    screen._move_to(99)
    assert screen.selected_index == 3
    screen.action_move(1)
    assert screen.selected_index == 3
    screen.action_move(-99)
    assert screen.selected_index == 0
    screen.action_move(-1)
    assert screen.selected_index == 0


def test_a_new_query_homes_the_cursor_on_the_first_match() -> None:
    """Not the nearest surviving row: clamping lands the cursor on the LAST
    match, so Enter would take the least related row still standing."""
    screen = MovePickerScreen(_targets(6))
    screen._move_to(4)
    screen.set_query("dir")
    assert screen.selected_index == 0
    assert screen.selected_path() == screen.visible_rows[0].path


def test_selecting_from_an_empty_result_set_answers_nothing() -> None:
    screen = MovePickerScreen(_targets(3))
    screen.set_query("nothing matches this")
    assert screen.visible_rows == []
    assert screen.selected_path() is None


def test_printable_keys_type_into_the_query() -> None:
    screen = MovePickerScreen(_targets(3))

    class _Key:
        character = "d"
        stopped = False

        def stop(self) -> None:
            self.stopped = True

        def prevent_default(self) -> None:
            pass

    screen.on_key(_Key())
    assert screen.filter_query == "d"


def test_backspace_edits_the_query_and_stops_at_empty() -> None:
    screen = MovePickerScreen(_targets(3))
    screen.set_query("ab")
    screen.action_backspace()
    assert screen.filter_query == "a"
    screen.action_backspace()
    screen.action_backspace()
    assert screen.filter_query == ""


# -- the two input modes -----------------------------------------------------


def test_a_plain_word_FILTERS_the_suggestions() -> None:
    screen = MovePickerScreen(_targets(4))
    screen.set_query("dir1")
    assert screen.is_path_query is False
    assert [row.path for row in screen.visible_rows] == ["/dir1"]


def test_a_path_COMPLETES_against_the_filesystem(tmp_path: Path) -> None:
    """The half of the picker that reaches a directory the suggestions never
    guessed — without it the card is abandoned the first time someone wants
    one it did not offer."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    asked: list[str] = []

    def complete(query: str) -> list[MoveTarget]:
        asked.append(query)
        return [_target(str(tmp_path / "alpha"), "typed")]

    screen = MovePickerScreen(_targets(3), complete=complete)
    screen.set_query(f"{tmp_path}/al")
    assert screen.is_path_query is True
    # Reading the rows is what runs the tier: the result is cached against the
    # query, so the filesystem is touched once per keystroke and not per paint.
    assert [row.path for row in screen.visible_rows] == [str(tmp_path / "alpha")]
    assert asked == [f"{tmp_path}/al"]
    # Cached: a repaint must not re-list the directory.
    assert [row.path for row in screen.visible_rows] == [str(tmp_path / "alpha")]
    assert asked == [f"{tmp_path}/al"]


def test_a_failing_completer_yields_an_empty_list_not_an_error() -> None:
    def boom(_query: str) -> list[MoveTarget]:
        raise OSError("gone")

    screen = MovePickerScreen(_targets(3), complete=boom)
    screen.set_query("/nowhere/x")
    assert screen.visible_rows == []


def test_without_a_completer_a_path_falls_back_to_FILTERING() -> None:
    """A smaller feature rather than an error, so the widget stays testable
    and usable by an embedder that has no session to resolve against. The
    fallback filters rather than answering empty: a host with no completer
    should still be able to reach the rows it WAS given, and an empty list
    would make the card look broken instead of merely less capable."""
    screen = MovePickerScreen(_targets(3))
    screen.set_query("/dir1")
    assert [row.path for row in screen.visible_rows] == ["/dir1"]


def test_tab_completes_the_highlighted_row_and_descends(tmp_path: Path) -> None:
    """The trailing separator is what makes a second tab list INSIDE the
    directory rather than re-matching it among its siblings."""
    screen = MovePickerScreen(_targets(3), complete=lambda _q: [])
    screen.action_complete()
    assert screen.filter_query == "/here/"
    assert screen.is_path_query is True


def test_tab_on_an_empty_list_does_nothing() -> None:
    screen = MovePickerScreen(_targets(3), complete=lambda _q: [])
    screen.set_query("/no/such")
    screen.action_complete()
    assert screen.filter_query == "/no/such"


# -- the card's own text -----------------------------------------------------


def test_the_header_names_which_mode_is_in_force() -> None:
    """An empty list means different things in the two modes, so a user who
    cannot tell which they are in cannot read the answer."""
    screen = MovePickerScreen(_targets(3), complete=lambda _q: [])
    screen.set_query("dir")
    assert "filter dir" in "\n".join(screen.render_lines_for_test())
    screen.set_query("/dir")
    assert "path /dir" in "\n".join(screen.render_lines_for_test())


def test_the_two_empty_states_say_different_things() -> None:
    screen = MovePickerScreen(_targets(3), complete=lambda _q: [])
    screen.set_query("zzz")
    assert NO_MATCH_NOTICE in "\n".join(screen.render_lines_for_test())
    screen.set_query("/zzz")
    assert NO_PATH_NOTICE in "\n".join(screen.render_lines_for_test())


def test_the_footer_always_says_how_to_leave() -> None:
    """It is the only statement of how to get out, so it is never shed."""
    for width in (100, 60, 40, 26):
        screen = MovePickerScreen(_targets(3))
        lines = screen._card_text().split("\n")
        assert "esc" in lines[-1].plain


def test_the_position_row_appears_only_when_the_list_scrolls() -> None:
    """Printing an empty line in its place leaves two blank rows and pushes
    the keys away from the block they belong to."""
    short = MovePickerScreen(_targets(3))
    assert "showing" not in "\n".join(short.render_lines_for_test())
    long = MovePickerScreen(_targets(PAGE_ROWS_MAX + 5))
    assert "showing" in "\n".join(long.render_lines_for_test())


# -- geometry, against the real app and stylesheet ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(100, 30), (150, 40), (80, 24), (60, 16)])
async def test_the_card_never_makes_the_screen_scrollable(size) -> None:
    """A tall overlay that pushes virtual height past the screen's own size
    silently costs two cells of width and reflows the transcript behind it —
    a defect this repo has hit before."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app.push_screen(MovePickerScreen(_targets(PAGE_ROWS_MAX + 10)))
        for _ in range(4):
            await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MovePickerScreen)
        assert screen.virtual_size.height <= screen.size.height
        assert screen.show_vertical_scrollbar is False


@pytest.mark.asyncio
async def test_the_cursor_can_only_sit_on_a_row_the_card_drew() -> None:
    """A fixed page let the cursor sit on a row the card never rendered —
    Enter then moved somewhere the user could not see."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 16)) as pilot:
        await pilot.pause()
        screen = MovePickerScreen(_targets(30))
        app.push_screen(screen)
        for _ in range(4):
            await pilot.pause()
        drawn = [line for line in screen.render_lines_for_test() if line.startswith(("❯ ", "  /"))]
        assert len(drawn) <= screen._page_rows()
        screen.action_jump(1)
        for _ in range(2):
            await pilot.pause()
        painted = "\n".join(screen.render_lines_for_test())
        assert screen.selected_path() is not None
        assert str(screen.selected_path()) in painted
