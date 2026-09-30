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


async def test_the_detail_page_splits_the_session_count_and_carries_the_refresh(
    tmp_path: Path,
) -> None:
    """The page's own clauses on one working link plus one filing (schema 2):
    the heading counts the split in the canvases' words, the filed row says
    ``[filed]``, and the pinned footer carries the live refresh assertion."""
    import json
    import time as _time

    session = _ProjectSession()
    registry = _rich_registry(tmp_path)
    project = registry.get_project_by_name("parity-spec")
    assert project is not None
    registry.link_session(project.id, "439818272d84", role="coordination")
    path = tmp_path / "projects" / f"{project.id}.json"
    payload = json.loads(path.read_text())
    payload["progress"] = "one line"
    payload["progress_updated_at"] = _time.time() - 5 * 3600
    payload["progress_reported_by"] = SESSION_ID
    payload["progress_refreshed_at"] = _time.time() - 3600
    payload["progress_refreshed_by"] = "439818272d84"
    path.write_text(json.dumps(payload))
    session.project_registry = ProjectRegistry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(160, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"
        joined = "\n".join(view.rendered_rows())
        assert "sessions (1 working · 1 filed)" in joined
        assert "439818272d84 [filed]" in joined
        assert "refreshed 60m ago by session 439818272d84 — no new content since" in joined


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
        # The ruler names the section the viewport is actually showing: the
        # feed's rows moved where the entry reveal settles, so the expectation
        # comes from the DOM truth rather than a literal (design D1's rule).
        assert rows[1].startswith(f"── {_dom_top_section(view._detail_page)} ")
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
    """The section the viewport actually shows, read from live widget regions.

    The last heading at/above the container's content top — the DOM truth a
    ruler assertion compares against. A heading's ``.gap-above`` blank row
    counts as part of its section (design review round 1, D6): when the top row
    is that blank the reader is looking at the heading below it, not at the
    section that scrolled away above.
    """
    base = page.content_region.y
    best: str | None = None
    for child in page.children:
        section = getattr(child, "section", None)
        if section is None:
            continue
        start = child.region.y - (1 if "gap-above" in child.classes else 0)
        if start <= base:
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
        assert view.rendered_rows()[1].startswith(f"── {_dom_top_section(page)} ")
        assert page.max_scroll_y > 0
        # A FORCED repaint must not change the reading: the entry frames used
        # to pass by paint ORDER (the reveal scroll never repainted the rule),
        # not by computation (D1).
        view._paint_rule()
        await pilot.pause()
        assert view.rendered_rows()[1].startswith(f"── {_dom_top_section(page)} ")
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
            "updates",
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


def _feed_registry(tmp_path: Path) -> ProjectRegistry:
    """One project with a two-entry feed, an image attachment and a long entry.

    The long entry overflows :data:`UPDATE_BODY_LINES` so the clamp marker and
    its `↵` toggle have something real to act on; the attachment is COPIED into
    the store, so its row carries a resolvable path.
    """
    registry = ProjectRegistry(tmp_path)
    project = registry.create_project(
        ProjectEdit(
            name="parity-spec",
            title="TUI parity spec",
            description="The spec, in one paragraph.",
            milestones=[ProjectMilestone(name="groundwork", target_date="2026-09-20")],
        )
    )
    shot = tmp_path / "board-60x20.png"
    shot.write_bytes(b"x" * 83904)
    registry.update_project(
        project.id,
        ProjectEdit(progress="first report"),
        reporter="operator",
        attachments=[shot],
    )
    long_body = "a much longer second report\n\n## Heading\n" + "\n".join(
        f"line {n} of a long entry" for n in range(1, 12)
    )
    registry.update_project(
        project.id,
        ProjectEdit(progress=long_body),
        reporter="ab12cd34ef56",
    )
    return registry


async def test_the_updates_feed_renders_newest_first_with_day_and_attachments(
    tmp_path: Path,
) -> None:
    """Spec §7.3/§7.4: day group, newest entry first, markdown body, files."""
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        rows = view._detail_page.painted_rows()
        assert any(row.startswith("updates (2)") for row in rows)
        assert any(row.startswith("── today ──") for row in rows)
        stamps = [row for row in rows if "session ab12cd34ef56" in row]
        assert stamps, rows
        # Newest first: the long (second) entry's body precedes the first's.
        joined = "\n".join(rows)
        assert joined.index("a much longer second report") < joined.index("first report")
        # The clamped tail names the key that opens it.
        assert any("more lines — ↵ expand" in row for row in rows)
        # The attachment affordance carries kind, name and size; the path
        # rides its own row under it and is visible without selecting.
        attachment = next(row for row in rows if row.startswith("[img]"))
        assert "board-60x20.png" in attachment and "KB" in attachment
        # The path is the COPY's, under the store's own name for it — that is
        # the file a reader can actually open (the original may be long gone).
        path_row = next(row for row in rows if row.strip().startswith("→"))
        assert "attachments" in path_row and path_row.strip().endswith(".png")
        # The feed is a SECTION: the ruler can name it.
        ruler = view.rendered_rows()[1]
        assert ruler.startswith("── overview ")


async def test_the_updates_section_says_so_when_there_are_no_entries(tmp_path: Path) -> None:
    """Empty is an honest sentence, never a zeroed placeholder (spec §7.3)."""
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        rows = view._detail_page.painted_rows()
        assert any(row.startswith("updates") and "(" not in row for row in rows)
        assert any("no updates recorded yet" in row for row in rows)


async def test_enter_on_an_update_stamp_opens_and_closes_its_clamped_body(
    tmp_path: Path,
) -> None:
    """Spec §7.3: `↵` toggles the tail; the cursor stays on the same row."""
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert page.selected_action_label() == "expand"
        assert any("more lines — ↵ expand" in row for row in page.painted_rows())
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert page.selected_action_label() == "collapse"
        rows = page.painted_rows()
        assert any("more lines — ↵ collapse" in row for row in rows)
        assert any("line 11 of a long entry" in row for row in rows)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert page.selected_action_label() == "expand"
        assert any("more lines — ↵ expand" in row for row in page.painted_rows())


async def test_enter_on_an_attachment_asks_the_host_to_open_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §7.4: the row's verb is ``open``, the host opens, the page says so.

    The opener itself is stubbed: the real one is a process spawn handing a
    file to a GUI, which a headless run must never do. What this pins is the
    wiring — the row's verb, the path that reaches the host, and the sentence
    that comes back — with the spawn replaced at the module seam the handler
    reads.
    """
    import local_operator.tui.attachments as attachments_mod

    opened: list[str] = []

    async def _fake_open(path: str) -> bool:
        opened.append(path)
        return True

    monkeypatch.setattr(attachments_mod, "opener_argv", lambda path: ["true", path])
    monkeypatch.setattr(attachments_mod, "open_path_quietly", _fake_open)
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        while page.selected_action_label() != "open":
            await pilot.press("down")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert opened, "the attachment row never reached the opener"
        # The store keeps the copy under its own name; the ORIGINAL name is
        # what the row and the receipt show.
        assert "attachments" in opened[-1]
        assert Path(opened[-1]).is_file()
        assert "opened board-60x20.png" in view.rendered_rows()[-1]
        assert view._mode == "detail"  # the page keeps the reader


async def test_a_gone_attachment_is_flagged_and_offers_no_verb(tmp_path: Path) -> None:
    """Spec §7.4 + UX round 1, U3: the row stops offering what it cannot do."""
    registry = _feed_registry(tmp_path)
    project = registry.get_project_by_name("parity-spec")
    assert project is not None
    Path(project.updates[0].attachments[0].path).unlink()
    session = _ProjectSession()
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert any("[missing on disk]" in row for row in page.painted_rows())
        # The row is still a cursor stop, but it advertises nothing: the reader
        # learns the row is dead from the marker, not from spending a press.
        attachment_rows = [row for row in page._selectables if "open" in (row.action_label() or "")]
        assert attachment_rows == []
        assert page.selected_action_label() != "open"


async def test_a_copy_that_vanishes_after_composition_answers_honestly(
    tmp_path: Path,
) -> None:
    """The handler still guards at the boundary: a press against a gone file."""
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        while page.selected_action_label() != "open":
            await pilot.press("down")
            await pilot.pause()
        # Delete it AFTER composition, so the page still believes it is there.
        project = session.project_registry.get_project_by_name("parity-spec")
        assert project is not None
        Path(project.updates[0].attachments[0].path).unlink()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert view._mode == "detail"  # the page keeps the reader
        assert "missing on disk" in view.rendered_rows()[-1]


async def test_a_row_without_the_updates_field_still_loads(tmp_path: Path) -> None:
    """Back-compat: a row stored before the feed existed renders its sentence."""
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "board-entry")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        rows = view._detail_page.painted_rows()
        assert any("no updates recorded yet" in row for row in rows)
        assert any(row.startswith("milestones") for row in rows)


async def test_a_long_attachment_path_is_fitted_to_its_row(tmp_path: Path) -> None:
    """A path too long for the box is cut HERE, with an ellipsis (spec §7.4).

    Textual wraps a ``Static``'s text whatever its ``no_wrap`` says, so a long
    path painted as a bare ``→`` with the rest hard-split onto following rows —
    measured in the 60- and 100-column frames. The row fits itself to the box
    instead, so the reader always gets as much of the handle as the width
    allows.
    """
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        from local_operator.tui.widgets.projects_detail import DetailAttachmentPathRow

        path_row = next(c for c in page.children if isinstance(c, DetailAttachmentPathRow))
        # `content` is a union on the Static; the file's own idiom is getattr.
        text = str(getattr(path_row.content, "plain", ""))
        assert text.strip().startswith("→ ")
        # Middle-ellipsized: the FILE's tail survives, which is what tells one
        # attachment row from another (design review round 1, D2).
        assert "…" in text and text.strip().endswith(".png")
        assert path_row.region.height == 1  # fitted: nothing wrapped
        assert cell_len(text) <= path_row.region.width


async def test_the_selected_feed_row_is_on_screen_when_the_page_opens(tmp_path: Path) -> None:
    """The entry reveal must survive a box too short to hold the page (60x24).

    Textual treats a row on the viewport's bottom EDGE as visible while the
    painted rows stop a cell earlier, so the selected entry sat one row below
    the box and the feed looked empty until a key was pressed — measured, and
    the reason the reveal clamps explicitly.
    """
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(60, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        row = page._selectables[page.selected_index]
        # `region.y` IS the painted screen row (measured: the same value the
        # frame shows), so containment is a direct comparison — the earlier
        # `content_region.y + region.y - scroll_offset.y` reconstruction
        # described a different quantity than the one it claimed (agent review
        # round 1, MINOR-4).
        assert page.region.y <= row.region.y <= page.region.y + page.region.height - 1


async def test_a_one_line_entry_offers_no_verb_and_toggles_nothing(tmp_path: Path) -> None:
    """Design review round 1, D1: no hint for a key that cannot act.

    The first thing a reader saw on open was `↵ expand` on an entry with nothing
    to expand: the press flipped the row's label while `painted_rows()` stayed
    byte-identical.
    """
    registry = ProjectRegistry(tmp_path)
    project = registry.create_project(ProjectEdit(name="shorty", title="Shorty"))
    registry.update_project(project.id, ProjectEdit(progress="one line only"), reporter="operator")
    session = _ProjectSession()
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "shorty")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert page.selected_action_label() is None
        assert "expand" not in view.rendered_rows()[-1]
        before = page.painted_rows()
        await pilot.press("enter")
        await pilot.pause()
        assert page.painted_rows() == before
        assert page.selected_action_label() is None
        assert page._expanded == set()


async def test_expanding_an_entry_does_not_leak_into_another_project(tmp_path: Path) -> None:
    """Agent review round 1, MINOR-1: expansion is per-project view state."""
    registry = ProjectRegistry(tmp_path)
    long_body = "\n".join(f"line {n}" for n in range(1, 12))
    for name in ("alpha", "beta"):
        project = registry.create_project(ProjectEdit(name=name, title=name.title()))
        registry.update_project(project.id, ProjectEdit(progress=long_body), reporter="operator")
    session = _ProjectSession()
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "alpha")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert page.selected_action_label() == "expand"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert page.selected_action_label() == "collapse"
        assert any("line 11" in row for row in page.painted_rows())
        await pilot.press("escape")
        await pilot.pause()
        view = await _open(pilot, app, "beta")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        # beta's own entry: same ordinal, different project — not expanded.
        assert page.selected_action_label() == "expand"
        assert not any("line 11" in row for row in page.painted_rows())


async def test_a_verbless_row_offers_no_open_hint_at_all(tmp_path: Path) -> None:
    """UX review round 2, U6: no verb means no `↵` rung, not a dimmed one.

    The ladder's `" open"` fallback was unreachable until design D1 and UX U3
    made verb-less rows real; on a one-line entry it printed a dimmed `↵ open`
    for a key that cannot act there.
    """
    import re

    registry = ProjectRegistry(tmp_path)
    project = registry.create_project(ProjectEdit(name="shorty", title="Shorty"))
    registry.update_project(project.id, ProjectEdit(progress="one line only"), reporter="operator")
    long_body = "\n".join(f"line {n}" for n in range(1, 12))
    other = registry.create_project(ProjectEdit(name="longie", title="Longie"))
    registry.update_project(other.id, ProjectEdit(progress=long_body), reporter="operator")

    def painted(view: Any) -> str:
        text = " ".join(
            hint.rendered()
            for hint in view._hints.children
            if isinstance(hint, HintButton) and hint.display
        )
        return re.sub(r"\s+", " ", text)

    session = _ProjectSession()
    session.project_registry = registry
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "shorty")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert page.selected_action_label() is None
        hints = painted(view)
        assert "↵" not in hints, hints
        assert " move " in hints  # the ladder itself still offers what works
        await pilot.press("escape")
        await pilot.pause()
        view = await _open(pilot, app, "longie")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        assert view._detail_page.selected_action_label() == "expand"
        assert "↵ expand" in painted(view)


async def test_clicking_a_feed_row_selects_and_a_second_click_acts(tmp_path: Path) -> None:
    """UX review round 1, U4: the new affordances are not keyboard-only.

    One `esc` away the canvas selects on the first click and acts on the
    second; the page's rows looked the same and did nothing at all.
    """
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        stamps = [row for row in page._selectables if "expand" in (row.action_label() or "")]
        assert stamps, page.painted_rows()
        target = stamps[0]
        # The first click moves the cursor and reveals, like the canvas.
        await pilot.click(target, offset=(3, 0))
        await pilot.pause()
        await pilot.pause()
        assert page._selectables[page.selected_index] is target
        assert page.selected_action_label() == "expand"
        # A second click on the SAME row activates it (both clicks in one
        # chain, the canvas's `event.chain == 2`).
        await pilot.click(target, offset=(3, 0), times=2)
        await pilot.pause()
        await pilot.pause()
        assert page.selected_action_label() == "collapse"
        assert any("line 11" in row for row in page.painted_rows())


async def test_the_hint_row_re_syncs_the_verb_after_a_toggle(tmp_path: Path) -> None:
    """UX review round 1, U2: the verb is the hint row's own input.

    The footer kept offering `↵ expand` on the row the reader had just opened
    and only caught up on the next cursor move — the press's only visible
    effect being that the offered key no longer matched what it would do.
    """
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 40)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()

        def hints() -> str:
            import re

            text = " ".join(
                hint.rendered()
                for hint in view._hints.children
                if isinstance(hint, HintButton) and hint.display
            )
            return re.sub(r"\s+", " ", text)

        assert "↵ expand" in hints()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        # No cursor move in between: the toggle itself must re-arm the hint.
        assert "↵ collapse" in hints()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert "↵ expand" in hints()


async def test_the_cursor_stays_on_screen_through_every_feed_stop(tmp_path: Path) -> None:
    """UX round 1, U1: `↓` through the feed never leaves the row off-screen.

    The earlier reveal clamped the scroll itself, comparing the row's
    ``region.y`` (a SCREEN row) with ``scroll_offset.y`` (a virtual one): the
    cursor went off-screen at 4/14 stops at 100x30 and 8/14 at 60x24 while the
    footer kept advertising a verb. Both sizes are walked here.
    """
    for size in ((100, 30), (60, 24)):
        # A store per size: the fixture names its project, and one tmp_path
        # cannot hold two of them.
        store = tmp_path / f"{size[0]}x{size[1]}"
        store.mkdir()
        session = _ProjectSession()
        session.project_registry = _feed_registry(store)
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await _boot(pilot, app)
            view = await _open(pilot, app, "parity-spec")
            await pilot.press("d")
            await pilot.pause()
            await pilot.pause()
            page = view._detail_page
            stops = page.selectable_count
            assert stops > 1
            for _ in range(stops - 1):
                await pilot.press("down")
                await pilot.pause()
                row = page._selectables[page.selected_index]
                assert (
                    page.region.y <= row.region.y <= page.region.y + page.region.height - 1
                ), f"cursor off-screen at {size}, stop {page.selected_index} of {stops}"


async def test_the_cursor_is_on_screen_when_a_tall_page_opens(tmp_path: Path) -> None:
    """The pin the design round asked for in place of the old literal (D6).

    A tall page reveals its first selectable row on open — the standing
    "reveal-then-act" rule. Measured on the page box, because the literal it
    replaces ("opens at overview") no longer holds once a feed sits above the
    milestones.
    """
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert page.max_scroll_y > 0, "the fixture must be taller than the box"
        row = page._selectables[page.selected_index]
        assert page.region.y <= row.region.y <= page.region.y + page.region.height - 1


async def test_the_ruler_matches_the_dom_truth_at_every_scroll_offset(tmp_path: Path) -> None:
    """Design review round 1, D6: a heading's blank row belongs to ITS section.

    The defect was one specific offset — the one the page OPENS at, where the
    viewport's top row is the updates heading's ``.gap-above`` blank and the
    ruler still named the description that had scrolled away above it. Rather
    than pin that single state, this sweeps every offset the page can hold and
    asserts the reading equals the DOM truth at each: with the gap ignored by
    the anchor math, the boundary offset disagrees.
    """
    session = _ProjectSession()
    session.project_registry = _feed_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        view = await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        page = view._detail_page
        assert page.max_scroll_y > 0, "the fixture must be taller than the box"
        for offset in range(int(page.max_scroll_y) + 1):
            page.scroll_to(y=offset, animate=False)
            await pilot.pause()
            truth = _dom_top_section(page)
            ruler = view.rendered_rows()[1]
            if truth is None:
                continue
            assert ruler.startswith(f"── {truth} "), (offset, ruler, truth)


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
