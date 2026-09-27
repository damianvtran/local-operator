"""The three project canvases: geometry and content, asserted as plain strings.

``projects_render`` exists so these assertions can be written at all — the
widget is a scroll container around what it returns, so the grid arithmetic,
the caps and the truncation rules are checked here rather than guessed from a
frame. The one thing a screenshot can still prove (that the widget pins its
canvas to these numbers) is asserted in ``test_projects_view.py``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from local_operator.tui.projects_render import (
    BOARD_CARDS_MAX,
    BOARD_COLUMN_WIDTH,
    PROJECTS_MAX,
    TIMELINE_NAME_WIDTH,
    aggregate_footer,
    auto_timeline_tier,
    detail_footer,
    render_project_board,
    render_project_list,
    render_project_timeline,
    timeline_axis_cells,
)

NOW = 1_760_000_000.0
DAY = 86400.0


def _view(name: str = "alpha", sessions: list[dict[str, Any]] | None = None, **over: Any) -> dict:
    project = {
        "id": f"id-{name}",
        "name": name,
        "status": "active",
        "description": "",
        "progress": "",
        "progress_updated_at": None,
        "progress_reported_by": "",
        "tags": [],
        "sessions": [],
        "created_at": 0.0,
        "updated_at": 0.0,
        "start_date": None,
        "target_date": None,
        "completed_at": None,
        "estimate": None,
        "estimate_unit": "points",
        "milestones": [],
        "schema": 1,
    }
    project.update(over)
    return {"project": project, "progress_stale": False, "sessions": sessions or []}


def _session(session_id: str = "ab12cd34ef56", state: str = "stopped", **over: Any) -> dict:
    row = {
        "session_id": session_id,
        "exists": True,
        "title": "t",
        "created_at": None,
        "archived": False,
        "runtime": {"state": state, "busy": None, "heartbeat_age_s": None, "pid": None},
        "subagents": None,
        "todos": None,
    }
    row.update(over)
    return row


# -- the list view -----------------------------------------------------------


def test_list_row_carries_every_field_the_row_owns() -> None:
    view = _view(
        estimate=13.0,
        target_date="2026-10-15",
        milestones=[
            {"name": "beta cut", "target_date": "2026-10-01", "completed_at": "2026-09-30"},
            {"name": "ga", "target_date": "2026-11-01", "completed_at": None},
        ],
        description="Payments migration",
        sessions=[_session(state="live")],
        progress="dashboard cutover done",
        progress_updated_at=NOW - 7200,
    )
    result = render_project_list([view], cursor=0, now=NOW)
    assert result.height == 1
    assert result.text.plain == (
        "▸ alpha [active] · est 13pt · →2026-10-15 · M 1/2 · 1 session (1 live) "
        '· progress 2h ago · "Payments migration"'
    )


def test_list_row_marks_the_cursor_and_stale_progress() -> None:
    fresh = _view("alpha")
    stale = _view("beta", progress="old", progress_updated_at=NOW - 10 * DAY)
    # The staleness flag is COMPUTED BY THE COMPOSITION (one threshold, one
    # place — `build_project_view`); the renderer reads it rather than
    # re-deriving it, so the fixture states it the way the composer would.
    stale["progress_stale"] = True
    result = render_project_list([fresh, stale], cursor=1, now=NOW)
    lines = result.text.plain.splitlines()
    assert lines[0].startswith("  ")  # no marker on the unselected row
    assert lines[1].startswith("▸ ")
    assert "no progress" in lines[0]
    assert "progress 10d ago (stale)" in lines[1]


def test_list_row_empty_store_names_the_shared_sentence() -> None:
    from local_operator.slash_commands import project_empty_text

    result = render_project_list([])
    assert result.text.plain == project_empty_text()
    assert result.width == len(project_empty_text())
    assert result.height == 1


def test_list_row_caps_a_long_description_at_the_tool_budget() -> None:
    view = _view(description="x" * 400)
    result = render_project_list([view], cursor=0, now=NOW)
    assert len(result.text.plain) == 160
    assert result.text.plain.endswith("…")


def test_list_truncates_past_the_render_cap_naming_the_count() -> None:
    views = [_view(f"p{i:03d}") for i in range(PROJECTS_MAX + 3)]
    result = render_project_list(views, cursor=0, now=NOW)
    assert result.height == PROJECTS_MAX + 1
    assert result.text.plain.splitlines()[-1].strip() == "… +3 more not shown"


# -- the board view ----------------------------------------------------------


def test_board_columns_fixed_order_and_archived_only_when_non_empty() -> None:
    views = [
        _view("a", status="active"),
        _view("p", status="paused"),
        _view("d", status="done"),
    ]
    result = render_project_board(views, now=NOW)
    header = result.text.plain.splitlines()[0]
    assert "active · 1" in header
    assert "paused · 1" in header
    assert "done · 1" in header
    assert "archived" not in header

    with_archived = render_project_board([*views, _view("z", status="archived")], now=NOW)
    assert "archived · 1" in with_archived.text.plain.splitlines()[0]


def test_board_card_is_three_lines_with_absent_chips_omitted() -> None:
    view = _view(
        "solo",
        estimate=9.0,
        target_date="2026-10-15",
        milestones=[{"name": "m", "target_date": None, "completed_at": None}],
        sessions=[_session(state="live"), _session("99ffaa11bb22", state="live")],
        progress="done-ish",
        progress_updated_at=NOW - 300,
    )
    result = render_project_board([view], now=NOW)
    lines = result.text.plain.splitlines()
    # header, blank, name, facts, freshness
    assert lines[0].startswith("active · 1")
    assert lines[2] == "solo"
    assert lines[3] == "est 9pt · M 0/1 · →2026-10-15"
    assert lines[4] == "reported 5m ago · 2 live"


def test_board_omits_absent_chips_rather_than_saying_none() -> None:
    result = render_project_board([_view("bare")], now=NOW)
    lines = result.text.plain.splitlines()
    assert lines[3] == "no estimate or dates"
    assert lines[4] == "no progress"


def test_board_caps_cards_per_column_with_a_more_row() -> None:
    views = [_view(f"p{i}") for i in range(BOARD_CARDS_MAX + 4)]
    result = render_project_board(views, now=NOW)
    assert "… +4 more" in result.text.plain
    # header + 12 * (blank + 3) + blank + more row
    assert result.height == 1 + BOARD_CARDS_MAX * 4 + 2


def test_board_truncates_every_cell_to_the_fixed_column_width() -> None:
    view = _view("n" * 80, target_date="2026-10-15", description="d")
    result = render_project_board([view], now=NOW)
    # Each column owns BOARD_COLUMN_WIDTH cells (content + its joint); with the
    # three fixed status columns that bounds every line, whatever it carries.
    for line in result.text.plain.splitlines():
        assert len(line) <= 3 * BOARD_COLUMN_WIDTH, repr(line)
    assert "…" in result.text.plain
    assert "n" * 80 not in result.text.plain


def test_board_empty_store_uses_the_shared_sentence() -> None:
    from local_operator.slash_commands import project_empty_text

    result = render_project_board([])
    assert result.text.plain == project_empty_text()
    assert result.height == 1


# -- the timeline view -------------------------------------------------------


def test_timeline_axis_cells_per_tier() -> None:
    start = date(2026, 1, 5)
    end = date(2026, 3, 5)
    # Quarter granularity: Q1 → Q1 is one cell; months: Jan→Mar is three.
    assert timeline_axis_cells(start, end, "quarter") == 1
    assert timeline_axis_cells(start, end, "month") == 3
    # Weeks: the cells are week-STARTS, so four whole weeks apart is five cells
    # (the week containing the 5th through the week containing the 5th of the
    # next month, inclusive).
    assert timeline_axis_cells(start, end, "week") == 9


def test_timeline_draws_bar_milestones_and_today_marker() -> None:
    view = _view(
        "ship",
        start_date="2026-01-05",
        target_date="2026-06-05",
        milestones=[
            {"name": "cut", "target_date": "2026-01-20", "completed_at": "2026-01-19"},
            {"name": "late", "target_date": "2026-02-28", "completed_at": None},
        ],
    )
    result = render_project_timeline([view], tier="month", today=date(2026, 3, 1), now=NOW)
    body = result.text.plain.splitlines()[1]
    assert "ship" in body
    # The bar runs Jan → Jun; the completed and overdue milestones sit over the
    # months they name, and the today marker over March — so the trailing
    # months are plain bar and everything is visible.
    assert "███" in body
    assert "◆" in body  # completed milestone
    assert "!" in body  # overdue milestone
    assert "┊" in body  # today column
    assert result.text.plain.splitlines()[0].startswith(" " * TIMELINE_NAME_WIDTH)


def test_timeline_completed_project_bars_to_completed_at() -> None:
    view = _view(
        "done-project",
        status="done",
        start_date="2026-01-05",
        completed_at="2026-02-01",
        target_date="2026-06-01",
    )
    result = render_project_timeline([view], tier="month", today=date(2026, 3, 1), now=NOW)
    body = result.text.plain.splitlines()[1]
    # Jan + Feb bars, and NOT the target's June cell.
    assert body.count("█") == 2


def test_timeline_lists_undated_projects_in_its_own_section() -> None:
    result = render_project_timeline(
        [_view("ship", start_date="2026-01-05", target_date="2026-03-05"), _view("someday")],
        tier="month",
        today=date(2026, 3, 1),
        now=NOW,
    )
    assert "no dates (1): someday" in result.text.plain


def test_timeline_with_no_dated_projects_is_only_the_section() -> None:
    result = render_project_timeline([_view("someday")], tier="month", today=date(2026, 3, 1))
    assert result.text.plain == "no dates (1): someday"


def test_timeline_truncates_names_to_the_pinned_name_column() -> None:
    view = _view("n" * 60, start_date="2026-01-05", target_date="2026-02-05")
    result = render_project_timeline([view], tier="month", today=date(2026, 3, 1), now=NOW)
    body = result.text.plain.splitlines()[1]
    assert body[: TIMELINE_NAME_WIDTH - 1] == "n" * (TIMELINE_NAME_WIDTH - 1)
    assert body[TIMELINE_NAME_WIDTH - 1] == "…"
    assert body[TIMELINE_NAME_WIDTH] == " "  # the joint before the axis


def test_auto_tier_coarsens_until_the_span_fits() -> None:
    short = [_view("a", start_date="2026-01-05", target_date="2026-03-05")]
    assert auto_timeline_tier(short, today=date(2026, 3, 1)) == "week"

    # A decade fits at MONTH granularity (122 cells + name column < 366), so
    # "coarsen to fit" must not fire early:
    decade = [_view("a", start_date="2016-01-05", target_date="2026-03-05")]
    assert auto_timeline_tier(decade, today=date(2026, 3, 1)) == "month"

    # ~36 years does not fit at month granularity; quarter is the coarsest tier.
    long = [_view("a", start_date="1990-01-05", target_date="2026-03-05")]
    assert auto_timeline_tier(long, today=date(2026, 3, 1)) == "quarter"


def test_timeline_empty_store_uses_the_shared_sentence() -> None:
    from local_operator.slash_commands import project_empty_text

    result = render_project_timeline([])
    assert result.text.plain == project_empty_text()


# -- the pinned footer -------------------------------------------------------


def test_detail_footer_names_progress_reporter_and_staleness() -> None:
    view = _view(
        progress="cutover done",
        progress_updated_at=NOW - 7200,
        progress_reported_by="4e92693767fa",
    )
    view["progress_stale"] = True
    text = detail_footer(view, now=NOW).plain
    assert "alpha [active]" in text
    assert "progress reported 2h ago by 4e92693767fa · stale: cutover done" in text
    assert "no linked sessions" in text


def test_detail_footer_omits_unknown_counts_instead_of_zeroing_them() -> None:
    view = _view(
        sessions=[
            _session(
                "ab12cd34ef56",
                state="live",
                subagents={"running": 2, "settled": 1, "names": ["a"]},
                todos={"open": 3, "total": 5},
            ),
            _session("99ffaa11bb22", state="stopped"),
        ]
    )
    text = detail_footer(view, now=NOW).plain
    assert "ab12cd34ef56 [live] 2 running/1 settled · todos 3/5" in text
    assert "99ffaa11bb22 [stopped]" in text
    # The second session's unknown counts are simply absent.
    assert "99ffaa11bb22 [stopped] 0" not in text


def test_detail_footer_milestones_carry_derived_status() -> None:
    view = _view(
        milestones=[
            {"name": "cut", "target_date": "2026-01-01", "completed_at": "2026-01-02"},
            {"name": "late", "target_date": "2020-01-01", "completed_at": None},
            {"name": "next", "target_date": "2099-01-01", "completed_at": None},
        ]
    )
    text = detail_footer(view, now=NOW).plain
    assert "M 1/3" in text
    assert "cut [completed]" in text
    assert "late [overdue]" in text
    assert "next [upcoming]" in text


def test_aggregate_footer_counts_projects_and_live_sessions() -> None:
    views = [
        _view("a", sessions=[_session(state="live")]),
        _view("p", status="paused"),
        _view("d", status="done", sessions=[_session(state="stopped")]),
    ]
    text = aggregate_footer(views, tier="month").plain
    assert "3 projects" in text
    assert "1 active" in text and "1 paused" in text and "1 done" in text
    assert "1 live session" in text
    assert "zoom: month" in text
