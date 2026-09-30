"""The projects detail page (S6d parity P2): rows, ruler, keys, clicks.

The pure text builders are tested directly (they take composed payload rows
and a style resolver); the page and state machine are driven through the real
``OperatorApp`` with isolated HOME/config, the same harness the P1 suites use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len

from local_operator.projects import (
    MilestoneEdit,
    ProjectEdit,
    ProjectMilestone,
    ProjectRegistry,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.projects_render import (
    detail_meta_line,
    detail_milestone_row_text,
    detail_progress_line,
    detail_ruler,
    detail_section_heading,
    detail_session_row_text,
    detail_todo_lines,
    format_short_date,
)
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.transcript import GAP_CLASS
from tests.unit.tui.test_projects_view import (
    SESSION_ID,
    _boot,
    _factory,
    _grouped_registry,
    _open,
    _ProjectSession,
    _registry,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _scratch_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _rich_registry(tmp_path: Path) -> ProjectRegistry:
    """One fully-populated core project plus one untitled sibling in personal."""
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name="parity-spec",
            title="TUI parity spec",
            team="core",
            owner="damian",
            tags=["tui", "spec"],
            estimate=5,
            start_date="2026-09-20",
            target_date="2026-10-04",
            description="TUI parity spec.\n\n## Goal\nOne reviewed spec.",
            milestones=[
                ProjectMilestone(
                    name="groundwork", target_date="2026-09-20", completed_at="2026-09-20"
                ),
                ProjectMilestone(name="spec shipped", target_date="2026-10-01"),
                ProjectMilestone(name="impl dispatched", target_date="2026-09-25"),
            ],
        ),
        sessions=[SESSION_ID],
    )
    registry.create_project(ProjectEdit(name="board-entry", title="Board entry", team="personal"))
    return registry


def _row(**project: Any) -> dict[str, Any]:
    """A minimal composed view row, shaped like ``build_project_view``'s."""
    base = {"id": "p1", "name": "alpha", "status": "active", "milestones": []}
    base.update(project)
    return {"project": base, "progress_stale": False, "sessions": []}


# ---------------------------------------------------------------------------
# Pure text builders
# ---------------------------------------------------------------------------


async def test_format_short_date_says_dates_the_way_a_person_does() -> None:
    assert format_short_date("2026-10-04", today="2026-09-29") == "4 Oct"
    assert format_short_date("2027-01-02", today="2026-09-29") == "2 Jan 2027"
    assert format_short_date(None) is None
    assert format_short_date("") is None
    # Unparseable values pass through untouched (a store row, never a crash).
    assert format_short_date("sometime") == "sometime"


async def test_detail_meta_sheds_whole_clauses_and_states_nothing_set() -> None:
    view = _row(
        owner="damian",
        team="core",
        start_date="2026-09-20",
        target_date="2026-10-04",
        estimate=5,
        tags=["tui", "spec"],
    )
    full = detail_meta_line(view, width=200).plain
    assert (
        full
        == "owner: damian · team: core · target 4 Oct · start 20 Sep · est 5pt · tags tui · spec"
    )
    # Narrow: clauses drop from the TAIL, whole, until the line fits — and the
    # DEADLINE outlives the start date (design review round 1, D2).
    narrow = detail_meta_line(view, width=30).plain
    assert narrow.startswith("owner: damian")
    assert "tags" not in narrow and "est 5pt" not in narrow
    assert len(narrow) <= 30
    mid = detail_meta_line(view, width=50).plain
    assert "target 4 Oct" in mid and "start 20 Sep" not in mid
    assert detail_meta_line(_row(), width=80).plain == "no estimate or dates"
    # Nothing set but a name/status: still the shipped sentence, not a blank.
    assert detail_meta_line(_row(team=None), width=80).plain == "no estimate or dates"


async def test_detail_progress_line_attributes_and_degrades() -> None:
    import time as _time

    now = _time.time()
    view = _row(
        progress="one line", progress_updated_at=now - 7200, progress_reported_by=SESSION_ID
    )
    view["project"]["updated_at"] = now
    line = detail_progress_line(view, width=100).plain
    assert line.startswith("progress reported 2h ago by session ab12cd34ef56 · updated ")
    # The update stamp sheds first; the attribution next; the age is the point.
    mid = detail_progress_line(view, width=60).plain
    assert "by session ab12cd34ef56" in mid and "updated" not in mid
    narrow = detail_progress_line(view, width=40).plain
    assert narrow == "progress reported 2h ago"
    operator = _row(progress="x", progress_updated_at=now - 60, progress_reported_by="operator")
    assert "by the operator" in detail_progress_line(operator, width=100).plain
    assert detail_progress_line(_row(), width=100).plain == "no progress recorded"


async def test_detail_section_heading_is_two_aligned_lines() -> None:
    head, underline = detail_section_heading("milestones", "1/3")
    assert head == "milestones (1/3)"
    assert underline == "─" * len(head)
    assert detail_section_heading("todos")[0] == "todos"


async def test_detail_milestone_rows_carry_the_timeline_glyph_trio() -> None:
    # The row builders read COMPOSED rows (dicts), like every renderer.
    done = detail_milestone_row_text(
        {"name": "groundwork", "target_date": "2026-09-20", "completed_at": "2026-09-20"},
        selected=False,
    ).plain
    assert done == "  ◆ groundwork · 2026-09-20 · completed"
    overdue = detail_milestone_row_text(
        {"name": "impl dispatched", "target_date": "2026-09-25"}, selected=True
    ).plain
    assert overdue.startswith("▸ ! impl dispatched")
    assert overdue.endswith("· overdue")
    upcoming = detail_milestone_row_text({"name": "spec shipped"}, selected=False).plain
    assert upcoming == "  ◇ spec shipped · upcoming"


async def test_detail_session_rows_mark_the_own_session_and_omit_nulls() -> None:
    row = {
        "session_id": SESSION_ID,
        "exists": True,
        "title": "projects review",
        "runtime": {"state": "stopped"},
        "subagents": {"running": 2, "settled": 3},
        "todos": {"open": 1, "total": 4},
    }
    line = detail_session_row_text(row, selected=True, own_session=SESSION_ID).plain
    assert line == '▸◆ab12cd34ef56 [stopped] · "projects review" · 2 running/3 settled · todos 1/4'
    bare = detail_session_row_text(
        {"session_id": "aa01bb02cc03", "exists": False, "runtime": {"state": "stopped"}},
        selected=False,
        own_session=SESSION_ID,
    ).plain
    assert bare == "  aa01bb02cc03 [missing]"
    # Unknown counts are omitted entirely — never zeroed.
    assert "running" not in bare and "todos" not in bare


async def test_detail_todo_lines_name_the_own_session_and_its_next_items() -> None:
    view = _row()
    view["sessions"] = [
        {
            "session_id": SESSION_ID,
            "todos": {"open": 11, "total": 15, "next": ["answer review round 1", "fold frames"]},
        },
        {"session_id": "aa01bb02cc03", "todos": {"open": 2, "total": 4}},
        {"session_id": "aa01bb02cc04", "todos": None},
    ]
    lines = [line.plain for line in detail_todo_lines(view, own_session=SESSION_ID)]
    assert lines == [
        'own session: 11 open / 15 — next: "answer review round 1", "fold frames"',
        "aa01bb02cc03: 2 open / 4",
    ]
    none_yet = [line.plain for line in detail_todo_lines(_row(), own_session=SESSION_ID)]
    assert none_yet == ["todo snapshots: none yet"]


async def test_detail_ruler_names_the_section_at_the_viewport_top() -> None:
    sections = [(0, "overview", None), (4, "description", None), (8, "milestones", "1/3")]
    top = detail_ruler(sections, 0, width=60)
    assert top is not None and top.plain.startswith("── overview ")
    middle = detail_ruler(sections, 5, width=60)
    assert middle is not None and middle.plain.startswith("── description ")
    # The count sheds before the name when narrow.
    at_milestones = detail_ruler(sections, 8, width=80)
    assert at_milestones is not None and at_milestones.plain.startswith("── milestones (1/3) ")
    narrow = detail_ruler(sections, 8, width=17)
    assert narrow is not None and narrow.plain.startswith("── milestones ")
    assert detail_ruler([], 0, width=60) is None


# ---------------------------------------------------------------------------
# The state machine, driven through the real app
# ---------------------------------------------------------------------------


async def test_d_opens_the_detail_and_esc_pops_one_level(tmp_path: Path) -> None:
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        view._body.scroll_to(y=0, animate=False)
        await pilot.pause()
        await pilot.press("d")
        cursor_before = view.cursor
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"
        assert not view._body.display and view._detail_page.display
        rows = view.rendered_rows()
        assert (
            rows[0]
            == "projects · TUI parity spec (parity-spec) [● active] · updated " + rows[0][-5:]
        )
        assert rows[1].startswith("── overview ")
        assert any("milestones (1/3)" in row for row in rows)
        assert "◆ ab12cd34ef56 [missing]" in " ".join(rows)  # own session, missing dir
        assert any(row.startswith("no progress recorded") for row in rows)
        # esc pops back to the SAME canvas with the SAME cursor.
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"
        assert view._body.display and not view._detail_page.display
        assert view.cursor == cursor_before  # popped one level, nothing moved
        assert app._projects_view is view  # the mode itself was not left


async def test_the_detail_cursor_walks_selectables_and_the_hint_names_the_verb(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        page = view._detail_page
        assert page.selectable_count == 4  # three milestones + the session link
        # The label NAMES the row it would toggle (UX round 1, U3).
        assert page.selected_index == 0 and page.selected_action_label() == "toggle groundwork"
        assert view._open_hint._label == " toggle groundwork"  # the context slot
        for _ in range(3):
            await pilot.press("down")
        await pilot.pause()
        assert page.selected_index == 3 and page.selected_action_label() == "open"
        assert view._open_hint._label == " open"
        for _ in range(4):
            await pilot.press("down")
        await pilot.pause()
        assert page.selected_index == 3  # clamped at the end
        for _ in range(4):
            await pilot.press("up")
        await pilot.pause()
        assert page.selected_index == 0  # clamped at the start


async def test_enter_toggles_a_milestone_through_the_store(tmp_path: Path) -> None:
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("enter")  # groundwork, completed -> uncheck
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"  # the page stays; only the row changes
        project = registry.get_project_by_name("parity-spec")
        assert project is not None and project.milestones[0].completed_at is None
        painted = view._detail_page.painted_rows()
        assert any("! groundwork" in row and "overdue" in row for row in painted)
        await pilot.press("enter")  # and back on again
        await pilot.pause()
        await pilot.pause()
        project = registry.get_project_by_name("parity-spec")
        assert project is not None and project.milestones[0].completed_at is not None
        painted = view._detail_page.painted_rows()
        assert any("◆ groundwork" in row and "completed" in row for row in painted)


async def test_enter_on_a_dead_session_row_keeps_the_page_and_says_so(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        for _ in range(3):
            await pilot.press("down")  # onto the session row
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # The ladder found nothing live: the page KEEPS the reader and the
        # honest sentence lands on ITS OWN footer (UX round 1, U4 — it used
        # to close the mode and post into the transcript it was hiding).
        assert app._projects_view is view
        assert view._mode == "detail"
        assert "no live session to open for 'parity-spec'" in view.rendered_rows()[-1]


async def test_refresh_keeps_the_detail_open_and_its_row_cursor(tmp_path: Path) -> None:
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert view._detail_page.selected_index == 1
        # A store change between refreshes must show up — and the reader must
        # stay exactly where they were reading (same mode, same row cursor).
        refreshed = registry.get_project_by_name("parity-spec")
        assert refreshed is not None
        registry.set_milestone(
            refreshed.id,
            MilestoneEdit(name="spec shipped", completed=True),
        )
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"
        assert view._detail_page.selected_index == 1
        assert "milestones (2/3)" in " ".join(view.rendered_rows())


async def test_detail_section_jumps_move_the_row_cursor_to_the_neighbour(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        # From the first milestone row, one section down lands on the session
        # link (todos has no selectables); one more clamps there.
        await pilot.press("shift+down")
        await pilot.pause()
        assert view._detail_page.selected_action_label() == "open"
        await pilot.press("shift+down")
        await pilot.pause()
        assert view._detail_page.selected_action_label() == "open"
        await pilot.press("shift+up")
        await pilot.pause()
        assert view._detail_page.selected_action_label() == "toggle groundwork"


def _dom_top_section(page: Any) -> str | None:
    """The section the viewport actually shows: the last heading at/above the
    container's content top, read from live widget regions — the DOM truth a
    ruler assertion compares against."""
    base = page.content_region.y
    best: str | None = None
    for child in page.children:
        section = getattr(child, "section", None)
        if section is None:
            continue
        if child.region.y <= base:
            best = section[0]
        else:
            break
    return best


async def test_the_detail_ruler_tracks_scrolling_with_a_tall_page(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name="tall",
            title="Tall project",
            description="One reviewed spec.",
            milestones=[
                ProjectMilestone(name=f"step {index:02d}", target_date="2026-10-01")
                for index in range(14)
            ],
        ),
        sessions=[SESSION_ID],
    )
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "tall")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        # The ruler must equal the DOM truth: the section whose heading is the
        # last one at or above the viewport top. Checked against the live
        # widget regions, not against a literal, so the invariant survives
        # fixture edits — design review round 1, D1 (the old math read a
        # section the viewport did not show).
        page = view._detail_page
        assert _dom_top_section(page) == "overview"
        assert view.rendered_rows()[1].startswith("── overview ")
        assert page.max_scroll_y > 0
        # A FORCED repaint must not change the reading: the entry frames used
        # to pass by paint ORDER (the reveal scroll never repainted the rule),
        # not by computation (D1).
        view._paint_rule()
        await pilot.pause()
        assert view.rendered_rows()[1].startswith("── overview ")
        await pilot.press("end")  # to the content's bottom; cursor stays put
        await pilot.pause()
        await pilot.pause()
        assert _dom_top_section(page) == "milestones"
        assert view.rendered_rows()[1].startswith("── milestones (0/14) ")
        view._paint_rule()
        await pilot.pause()
        assert view.rendered_rows()[1].startswith("── milestones (0/14) ")


async def test_detail_heading_gap_rides_the_sanctioned_class(
    tmp_path: Path,
) -> None:
    """Review r2, R2-1: the sheet declares no vertical margin for this page.

    The heading gap is the sheet's single sanctioned spacing class
    (``.gap-above``), applied by the page; the FIRST row deliberately does not
    carry it, so the heading that opens the page keeps its place (and with it
    every section anchor).
    """
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        # DOM-truth rows read through the same duck-typed lens as
        # `_dom_top_section` above; pyright holds the guard for `Widget`.
        headings: list[Any] = [row for row in page.children if getattr(row, "section", None)]
        assert [row.section[0] for row in headings] == [
            "overview",
            "description",
            "milestones",
            "todos",
            "sessions",
        ]
        assert not headings[0].has_class(GAP_CLASS)
        assert headings[0].styles.margin.top == 0
        assert all(row.has_class(GAP_CLASS) for row in headings[1:])
        assert all(row.styles.margin.top == 1 for row in headings[1:])


async def test_a_named_show_leaves_the_detail_and_the_page_keeps_its_own_project(
    tmp_path: Path,
) -> None:
    """QA round 1, Q1 — the split page and the crosswrite it enabled.

    Reading one project's detail, `/project show <other>` must land on the
    CANVAS of the named project (no chrome/body split), and the row verb's
    write target is the PAGE's provenance: moving the canvas cursor under an
    open page must not move where `↵` writes.
    """
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    parity_row = registry.get_project_by_name("parity-spec")
    board_row = registry.get_project_by_name("board-entry")
    assert parity_row is not None and board_row is not None
    parity_id = str(parity_row.id)
    board_id = str(board_row.id)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"
        page = view._detail_page
        assert page.project_id == parity_id
        # The retarget, through the real command: the page closes and the
        # canvas lands on the named project — no split page.
        app._run_slash_command("/project show board-entry")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "canvas"
        assert not page.display
        assert view.current_project_id() == board_id
        # The belt: with the page OPEN again, the row verb targets the PAGE's
        # project even when the canvas cursor points at another project's row
        # (the exact state the bug exploited; caught crosswrite pre-fix).
        await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        board_index = next(
            index
            for index, row in enumerate(view._views)
            if str((row.get("project") or {}).get("id")) == board_id
        )
        view._cursor = board_index
        view._detail_row_action("milestone", {"name": "step 01", "completed_at": None})
        await pilot.pause()
        await pilot.pause()
        parity = registry.get_project_by_name("parity-spec")
        board = registry.get_project_by_name("board-entry")
        assert parity is not None and board is not None
        assert any(m.name == "step 01" for m in parity.milestones)
        assert not any(m.name == "step 01" for m in board.milestones)


async def test_a_refused_toggle_says_so_on_the_page_and_names_the_project(
    tmp_path: Path,
) -> None:
    """UX round 1, U1: the "no" lands where the reader pressed.

    The page is a re-read-only reader, so a project deleted underneath it
    keeps its rows; `↵` must then say so IN PLACE — naming the project, never
    the store's raw id — instead of leaving every painted row identical while
    the sentence hides in the transcript this mode covers.
    """
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    project = registry.get_project_by_name("parity-spec")
    assert project is not None
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        registry.delete_project(project.id)  # another session; the page stays up
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"  # the page stays; the sentence is ON it
        footer = view.rendered_rows()[-1]
        assert "no longer in the store" in footer
        assert "parity-spec" in footer
        assert project.id not in "".join(view.rendered_rows())


async def test_a_dead_session_link_keeps_the_page_and_says_so(tmp_path: Path) -> None:
    """UX round 1, U4: the row says `open`, the outcome must not cost the page.

    A dead link now answers ON the page (the shipped sentence) instead of
    closing the mode and losing the reader's place.
    """
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        for _ in range(3):  # three milestones down to the session row
            await pilot.press("down")
            await pilot.pause()
        assert view._detail_page.selected_action_label() == "open"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"  # the page KEEPS the reader
        assert view._detail_page.display
        assert "no live session to open for 'parity-spec'" in view.rendered_rows()[-1]


async def test_a_vanished_project_pops_with_a_sentence(tmp_path: Path) -> None:
    """UX round 1, U5: the pop says WHY instead of just happening."""
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    project = registry.get_project_by_name("parity-spec")
    assert project is not None
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        registry.delete_project(project.id)
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "canvas"
        assert not view._detail_page.display
        footer = view.rendered_rows()[-1]
        assert "no longer in the store" in footer
        assert "parity-spec" in footer


async def test_painted_rows_include_the_markdown_description(tmp_path: Path) -> None:
    """UX round 1, U6: the readback sees the headline feature.

    The description row holds a rich ``Markdown`` renderable, so the page's
    own readback returned ``""`` for it — a regression in the headline
    feature would have looked like a passing test.
    """
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        rows = view._detail_page.painted_rows()
        assert any("## Goal" in row for row in rows)
        assert any("TUI parity spec." in row for row in rows)
        # The empty case still reads its honest sentence through the plain
        # ``Text`` path (``readback()`` answers None there).
        await _open(pilot, app, "board-entry")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert any("no description yet" in row for row in view._detail_page.painted_rows())


async def test_the_toggle_hint_names_the_row_it_will_toggle(tmp_path: Path) -> None:
    """UX round 1, U3: `↵ toggle groundwork`, not a bare `toggle`.

    The cursor can sit off-screen (pgdn scrolls without moving it); the named
    hint is what tells the reader what the press will change.
    """
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert view._detail_page.selected_action_label() == "toggle groundwork"
        assert view._detail_page.selected_action_verb() == "toggle"
        painted = [
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        ]
        assert any("toggle groundwork" in text for text in painted)


async def test_the_detail_page_re_resolves_styles_on_every_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 1, U2: the resolver is built AT SHOW TIME, not at construction.

    A theme switch under an open page must repaint in the new palette on the
    next re-show — the sibling surfaces' rule, which the constructor-only
    capture defeated. Pinned at the seam: the page's bound resolver is a NEW
    object after a recompose, and each one was built during this run.
    """
    from local_operator.tui.widgets import projects_view as pv

    resolvers: list[object] = []
    real = pv._style_resolver

    def recording():
        resolver = real()
        resolvers.append(resolver)
        return resolver

    monkeypatch.setattr(pv, "_style_resolver", recording)
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        before = view._detail_page._style_for
        assert before in resolvers
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        after = view._detail_page._style_for
        assert after is not before
        assert after in resolvers


async def test_a_long_refusal_notice_fits_a_narrow_footer(tmp_path: Path) -> None:
    """UX round 1, U8: the refusal sentence must obey a 50x18 footer.

    Measured pre-fix: 72 cells into a 46-cell box, chopped mid-token with no
    ellipsis (Rich's ``overflow="ellipsis"`` is inert on a Static). The
    sentence now leads with its cause and the fit is done against the
    measured box — cell-accurate, ellipsis included.
    """
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    project = registry.get_project_by_name("parity-spec")
    assert project is not None
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(50, 18)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        registry.delete_project(project.id)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        footer = view.rendered_rows()[-1]
        box = view._detail.size.width
        assert box > 0
        assert cell_len(footer) <= box
        assert footer.endswith("…")
        assert "no longer in the store" in footer


async def test_the_canvas_ladder_advertises_d_detail_and_keeps_the_60_snapshot(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _grouped_registry(tmp_path)
    for size, detail_visible in (((150, 40), True), ((100, 30), True), ((60, 20), False)):
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await _boot(pilot, app)
            view = await _open(pilot, app, "board-entry")
            assert view._detail_hint.display is detail_visible, size
            # The 60-column snapshot is the shipped row, unchanged: neither
            # the new key nor the detail hint pushes the view triplet around.
            if size == (60, 20):
                assert not view._open_hint.display
                assert not view._refresh_hint.display
    # The detail ladder itself: all P2 hints at 100; `pgup/pgdn` still shown
    # at 60 (measured rung widths: 60-full / 47 without refresh / 29 without
    # page / 24 without the esc label / 14 move only / esc alone).
    session2 = _ProjectSession()
    session2.project_registry = _rich_registry(tmp_path / "second")
    app = OperatorApp(lambda: _factory(session2))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        assert view._move_hint.display and view._page_hint.display
        assert view._refresh_hint.display and view._open_hint.display


async def test_d_opens_with_no_actions_and_enter_is_inert(tmp_path: Path) -> None:
    """A project with no milestones and no sessions still opens; `↵` does nothing."""
    session = _ProjectSession()
    session.project_registry = _registry(tmp_path, "alpha")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "alpha")
        await pilot.press("d")
        await pilot.pause()
        assert view._mode == "detail"
        assert view._detail_page.selectable_count == 0
        assert not view._open_hint._actionable
        await pilot.press("enter")  # nothing selectable: no activation, no crash
        await pilot.pause()
        assert view._mode == "detail"
        await pilot.press("escape")
        await pilot.pause()
        assert view._mode == "canvas"


async def test_a_deleted_project_pops_the_detail_back_to_the_canvas(
    tmp_path: Path,
) -> None:
    """`r` with the shown project gone: the page re-finds by id and pops home."""
    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        assert view._mode == "detail"
        doomed = registry.get_project_by_name("parity-spec")
        assert doomed is not None
        registry.delete_project(doomed.id)
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "canvas"
        assert view._body.display
