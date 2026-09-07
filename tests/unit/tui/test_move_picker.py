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
    _footer_hints,
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


# -- the completion dead end (design D1 / UX U1) -----------------------------


def _leaf_tree(tmp_path: Path) -> Path:
    """A directory that exists, is readable, and has no SUBdirectories."""
    leaf = tmp_path / "scripts"
    leaf.mkdir()
    (leaf / "a_file.py").write_text("files are not subdirectories")
    return leaf


def _walking_screen(tmp_path: Path) -> MovePickerScreen:
    from local_operator.tui.move_targets import (
        complete_path,
        resolve_self,
        suggest_targets,
    )

    targets = suggest_targets(tmp_path, config_dir=tmp_path / "cfg", home=tmp_path)
    return MovePickerScreen(
        targets,
        current=str(tmp_path),
        complete=lambda q: complete_path(q, cwd=tmp_path, home=tmp_path),
        self_target=lambda q: resolve_self(q, cwd=tmp_path, home=tmp_path),
    )


def test_tab_onto_a_childless_directory_leaves_it_selectable(tmp_path: Path) -> None:
    """`tab` appends a separator and completing INSIDE a directory with no
    subdirectories returns nothing — so the directory the user had just walked
    to rendered as "no directory matches that path", and `enter` there
    dismissed with `None`, which the caller reads as Esc: picker closed, no
    move, no notice. Found independently by design (D1) and UX (U1), and it
    made the picker strictly weaker than the typed path it fronts.
    """
    leaf = _leaf_tree(tmp_path)
    screen = _walking_screen(tmp_path)
    index = next(i for i, r in enumerate(screen.visible_rows) if r.path == str(leaf))
    screen._move_to(index)
    screen.action_complete()

    assert screen.visible_rows, "the directory the user walked to vanished from the card"
    assert screen.selected_path() == str(leaf), "enter would not move to that directory"
    painted = "\n".join(screen.render_lines_for_test())
    assert NO_PATH_NOTICE not in painted, "a real directory reported as no match"
    assert "enter" in painted.splitlines()[-1], "the footer dropped `enter move`"


def test_a_path_that_does_not_resolve_still_reports_no_match(tmp_path: Path) -> None:
    """The fix must not make the empty state unreachable: a path that genuinely
    does not exist is still a no-match, and keeps that sentence."""
    _leaf_tree(tmp_path)
    screen = _walking_screen(tmp_path)
    screen.set_query("~/nope/")
    assert screen.visible_rows == []
    assert NO_PATH_NOTICE in "\n".join(screen.render_lines_for_test())


def test_tab_completes_into_the_tidy_label_not_a_raw_absolute_path(tmp_path: Path) -> None:
    """The row reads `~/scripts`; inserting the absolute path made the header
    jump to a long string that did not resemble the row just chosen (D3)."""
    leaf = _leaf_tree(tmp_path)
    screen = _walking_screen(tmp_path)
    index = next(i for i, r in enumerate(screen.visible_rows) if r.path == str(leaf))
    screen._move_to(index)
    screen.action_complete()
    assert screen.filter_query == "~/scripts/"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_cols", "expect_tab"),
    [
        # Measured against the REAL app, not computed: the card is TWELVE
        # cells narrower than the terminal — `_screen_size()` is already inset
        # by `Screen { padding: 1 }` (2), then `_card_width` subtracts
        # PICKER_WIDTH_MARGIN (6) and its own padding (4). So the 73 cells that
        # carry both hints need an 85-column TERMINAL. 84 and 85 are pinned as
        # a PAIR so the boundary is guarded rather than straddled: an earlier
        # version pinned 84 and 86 and named 86 as "first width with it", which
        # is off by one (review MAJOR-4.3).
        (80, False),  # the standard terminal — documented to give `tab` up
        (84, False),  # last width without it
        (85, True),  # THE BOUNDARY: 73 card cells, measured end to end
        (86, True),
        (100, True),
    ],
)
async def test_the_footer_boundary_stated_in_TERMINAL_columns(
    tmp_path: Path, terminal_cols: int, expect_tab: bool
) -> None:
    """Where `tab complete` actually appears, measured end to end.

    `type to filter or path` and `tab complete` are the two hints nobody
    infers. D2: `type` was shed first, so the opening frame never carried it.
    D7: fixing that by shedding `tab` instead swapped the navigator's own
    gesture off the frame — a worse trade, since `pgup/pgdn` is the one hint a
    `showing N-M of T` counter already implies.

    D10 is why this test drives the app rather than calling `_footer_hints`
    with a number. The previous version passed `80` — which is 80 CARD cells,
    a 100-column terminal — while its docstring claimed that was "the width
    the picker actually opens at". Literally true and thoroughly misleading:
    it never asked about an 80-column terminal, where `tab` is still absent.
    The shed ORDER is correct and is not changed; what was wrong was the unit
    the guard and its comment were written in.

    So the boundary is pinned in the unit a user has: terminal columns.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(terminal_cols, 30)) as pilot:
        await pilot.pause()
        screen = _walking_screen(tmp_path)
        app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()
        footer = screen.render_lines_for_test()[-1]

    assert ("tab" in footer) is expect_tab, (
        f"at a {terminal_cols}-column terminal (card {screen._card_width()}) the "
        f"footer reads {footer!r}"
    )
    # The mode hint is never the one given up: it outlives `tab` at every width.
    assert "type" in footer, f"the mode hint was shed at {terminal_cols} columns"


def test_the_footer_sheds_paging_first_and_tab_only_when_truly_cramped() -> None:
    """The whole shed sequence, pinned at the measured boundaries.

    The previous test pins the opening width; this one pins the ORDER, so a
    future reorder cannot pass by happening to fit at 80. The numbers are
    measured from `_shed_to_width`, not chosen: the full row is 90 cells,
    dropping `pgup/pgdn` leaves 73, and dropping `tab` as well leaves 58 — so
    73 is the widest card that must still carry everything but paging, and 72
    is the first that has to give up `tab` (design D7).

    `↑↓`, `enter` and `esc` are never shed while any label survives: between
    them they are how the card is moved through, used and left.
    """
    assert [k for k, _ in _footer_hints(90)] == [
        "↑↓",
        "pgup/pgdn",
        "type",
        "tab",
        "enter",
        "esc",
    ], "the full row must fit at its own measured width"
    # The card at every terminal the picker realistically opens at.
    for width in (89, 80, 73):
        assert [k for k, _ in _footer_hints(width)] == [
            "↑↓",
            "type",
            "tab",
            "enter",
            "esc",
        ], f"paging must be the only thing shed at width={width}"
    # The cramped card: `tab` goes, and only here. The header already names
    # the mode the instant you type, while nothing else announces `tab`.
    for width in (72, 68, 58):
        assert [k for k, _ in _footer_hints(width)] == [
            "↑↓",
            "type",
            "enter",
            "esc",
        ], f"`tab` should be the second and last disclosure shed, at width={width}"
    # Narrower still: the mode hint goes, then the labels — never the three
    # keys that make the card usable at all.
    assert [k for k, _ in _footer_hints(40)] == ["↑↓", "enter", "esc"]


def test_ctrl_w_backs_out_one_path_segment(tmp_path: Path) -> None:
    """Backing out of a completed path cost 15 backspaces — the only editing
    key bound — which is why walking in felt one-way (U1)."""
    screen = _walking_screen(tmp_path)
    screen.set_query("~/a/b/c/")
    screen.action_kill_segment()
    assert screen.filter_query == "~/a/b/"
    screen.action_kill_segment()
    assert screen.filter_query == "~/a/"
    screen.action_kill_query()
    assert screen.filter_query == ""


def test_the_current_marker_survives_a_long_path(tmp_path: Path) -> None:
    """`current` answers "where am I?", not "why is this offered", so unlike
    every other note it is not dropped to buy label cells — it disappeared on
    exactly the deep paths this machine is full of (D4)."""
    deep = "~/workspace/repos/minerva-data-agent-service-integration-v2"
    row = MoveTarget(path=deep, label=deep, kind="current", detail="current")
    painted = render_rows([row], 0, 68)[0].plain
    assert "current" in painted
