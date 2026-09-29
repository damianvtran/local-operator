"""The projects detail page (S6d parity P2): rows, ruler, keys, clicks.

The pure text builders are tested directly (they take composed payload rows
and a style resolver); the page and state machine are driven through the real
``OperatorApp`` with isolated HOME/config, the same harness the P1 suites use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.projects import MilestoneEdit, ProjectEdit, ProjectMilestone, ProjectRegistry
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
from tests.unit.tui.test_projects_view import (
    SESSION_ID,
    _boot,
    _factory,
    _grouped_registry,
    _notices,
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
        == "owner: damian · team: core · start 20 Sep · target 4 Oct · est 5pt · tags tui · spec"
    )
    # Narrow: clauses drop from the TAIL, whole, until the line fits.
    narrow = detail_meta_line(view, width=30).plain
    assert narrow.startswith("owner: damian")
    assert "tags" not in narrow and "est 5pt" not in narrow
    assert len(narrow) <= 30
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
        assert page.selected_index == 0 and page.selected_action_label() == "toggle"
        assert view._open_hint._label == " toggle"  # the context hint, in the ↵ slot
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


async def test_enter_on_a_session_row_takes_the_shipped_conversation_ladder(
    tmp_path: Path,
) -> None:
    session = _ProjectSession()
    session.project_registry = _rich_registry(tmp_path)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(pilot, app)
        await _open(pilot, app, "parity-spec")
        await pilot.press("d")
        await pilot.pause()
        for _ in range(3):
            await pilot.press("down")  # onto the session row
        await pilot.press("enter")
        await pilot.pause()
        # The ladder found nothing live: the mode closes and the honest
        # sentence lands in the transcript it was hiding.
        assert app._projects_view is None
        assert any("parity-spec" in notice for notice in _notices(app))


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
        assert view._detail_page.selected_action_label() == "toggle"


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
        assert view.rendered_rows()[1].startswith("── overview ")
        assert view._detail_page.max_scroll_y > 0
        await pilot.press("end")  # to the content's bottom; cursor stays put
        await pilot.pause()
        await pilot.pause()
        assert view.rendered_rows()[1].startswith("── sessions (1) ")


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
