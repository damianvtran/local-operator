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

from rich.style import Style

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


def _view(
    name: str = "alpha", sessions: list[dict[str, Any]] | None = None, **over: Any
) -> dict[str, Any]:
    project = {
        "id": f"id-{name}",
        "name": name,
        "title": None,
        "owner": None,
        "team": None,
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


def _session(
    session_id: str = "ab12cd34ef56", state: str = "stopped", **over: Any
) -> dict[str, Any]:
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


def test_list_row_shows_the_title_first_with_the_key_secondary() -> None:
    titled = _view("payments-migration", title="Q4 Payments Migration")
    plain = render_project_list([titled], cursor=0, now=NOW).text.plain
    assert plain.startswith("▸ Q4 Payments Migration (payments-migration) [active]")

    # No title → the key is the label (every surface's fallback).
    fallback = render_project_list([_view("plain-key")], cursor=0, now=NOW).text.plain
    assert fallback.startswith("▸ plain-key [active]")


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
    # header, blank, name, facts, freshness. The name line opens with the
    # two-cell marker column (S3b): `▸` the selection, `◆` the session's own,
    # two spaces otherwise — the card grid does not move.
    assert lines[0].startswith("active · 1")
    assert lines[2] == "  solo"
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


def test_board_adds_the_key_only_when_the_column_holds_title_and_key() -> None:
    short = _view("k", title="Short")
    long = _view("payments-migration", title="Q4 Payments Migration")
    result = render_project_board([short, long], now=NOW)
    plain = result.text.plain
    assert "Short (k)" in plain  # both fit in 31 cells
    assert "Q4 Payments Migration" in plain
    assert "(payments-migration)" not in plain  # too wide — the title wins alone


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


def test_timeline_adds_the_key_only_when_the_name_column_holds_both() -> None:
    short = _view("k", title="Tiny", start_date="2026-01-05", target_date="2026-03-05")
    long = _view(
        "payments-migration",
        title="Q4 Payments Migration",
        start_date="2026-01-05",
        target_date="2026-03-05",
    )
    result = render_project_timeline([short, long], tier="month", today=date(2026, 2, 1), now=NOW)
    plain = result.text.plain
    assert "Tiny (k)" in plain  # 8 cells <= TIMELINE_NAME_WIDTH
    assert "Q4 Payments Migration" in plain
    assert "(payments-migration)" not in plain  # too wide — the title wins alone


def test_timeline_lists_undated_projects_in_its_own_section() -> None:
    result = render_project_timeline(
        [_view("ship", start_date="2026-01-05", target_date="2026-03-05"), _view("someday")],
        tier="month",
        today=date(2026, 3, 1),
        now=NOW,
    )
    assert "no dates (1): someday" in result.text.plain


def test_timeline_undated_section_shows_the_display_name() -> None:
    result = render_project_timeline(
        [_view("someday", title="Someday Maybe")],
        tier="month",
        today=date(2026, 3, 1),
        now=NOW,
    )
    assert "no dates (1): Someday Maybe" in result.text.plain


def test_timeline_with_no_dated_projects_is_only_the_section() -> None:
    result = render_project_timeline([_view("someday")], tier="month", today=date(2026, 3, 1))
    assert result.text.plain == "no dates (1): someday"


def test_timeline_truncates_names_to_the_pinned_name_column() -> None:
    view = _view("n" * 60, start_date="2026-01-05", target_date="2026-02-05")
    result = render_project_timeline([view], tier="month", today=date(2026, 3, 1), now=NOW)
    body = result.text.plain.splitlines()[1]
    # The name FIELD is unchanged in width; its first two cells are the marker
    # column (S3b), so the name truncates two cells earlier and still pads.
    assert body[:2] == "  "
    assert body[2:TIMELINE_NAME_WIDTH] == "n" * (TIMELINE_NAME_WIDTH - 3) + "…"
    assert body[TIMELINE_NAME_WIDTH] == " "
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


def test_detail_footer_shows_title_first_with_the_key() -> None:
    view = _view("payments-migration", title="Q4 Payments Migration")
    text = detail_footer(view, now=NOW).plain
    assert text.startswith("Q4 Payments Migration (payments-migration) [active]")


def test_detail_footer_trades_the_key_away_before_the_title() -> None:
    view = _view("payments-migration", title="Q4 Payments Migration")
    # Wide: both. Too tight for both but enough for title + chip: the key
    # sheds first (`title wins when only one fits`), the chip survives whole.
    wide = detail_footer(view, now=NOW, width=200).plain
    assert "Q4 Payments Migration (payments-migration) [active]" in wide
    tight = detail_footer(view, now=NOW, width=30).plain
    assert "(payments-migration)" not in tight
    assert tight.startswith("Q4 Payments Migration [active]")


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


# -- the axis, the footer and the shared rules (remediation round 1) ----------


def test_axis_labels_never_touch_and_never_clip() -> None:
    """QA Q3 / review finding 10: a label needs a blank cell and the whole label.

    The guard used to allow a label to start on the very next cell after its
    neighbour (``MayAugN``) and to clip the last label at the canvas edge.
    """
    views = [_view("alpha", start_date="2026-01-01", target_date="2026-12-31")]
    result = render_project_timeline(views, tier="month")
    axis = result.text.plain.split("\n")[0][TIMELINE_NAME_WIDTH + 1 :]
    assert "Jan " in axis and "May " in axis and "Sep " in axis
    # No label starts on the cell right after another's end (and none is
    # truncated): every present label is followed by a blank cell.
    assert "Apr" not in axis and "Aug" not in axis  # would have touched their neighbours


def test_axis_marks_the_first_label_of_each_new_year() -> None:
    """Design D6: an 18-month span otherwise reads ``Jan … Jan``."""
    month_views = [_view("alpha", start_date="2026-05-01", target_date="2027-11-30")]
    axis = render_project_timeline(month_views, tier="month").text.plain.split("\n")[0]
    assert "'27" in axis
    # Quarter labels are six cells wide once cued, so the span gives them room.
    quarter_views = [_view("alpha", start_date="2026-01-01", target_date="2028-12-31")]
    quarter = render_project_timeline(quarter_views, tier="quarter").text.plain.split("\n")[0]
    assert "'27" in quarter


def _long_footer_view() -> dict[str, Any]:
    return _view(
        "annotator-overhaul",
        progress="prototype landed; wiring the beta cut-over and the review gate",
        progress_updated_at=NOW - 3,
        progress_reported_by="fa11bacc0001",
        milestones=[
            {"name": "prototype", "target_date": "2026-09-01", "completed_at": "2026-09-02"},
            {"name": "beta", "target_date": "2026-12-01", "completed_at": None},
        ],
        sessions=[_session("fa11bacc0001", "live", todos={"open": 1, "total": 2})],
    )


def test_detail_footer_sheds_whole_clauses_to_fit() -> None:
    """UX U1: the footer fits its width by dropping clauses, never mid-word."""
    view = _long_footer_view()
    full = detail_footer(view, now=NOW)
    assert "prototype" in full.plain and "sessions:" in full.plain
    assert "\u2026" not in full.plain  # no "more exists" marker on the full rung
    # 148 cells: identity + milestones + rollup fit; the progress body sheds.
    fitted = detail_footer(view, now=NOW, width=148)
    assert "sessions: fa11bacc0001 [live]" in fitted.plain
    assert "prototype [completed]" in fitted.plain
    assert "prototype landed" not in fitted.plain
    assert fitted.plain.endswith("\u2026")
    # 98 cells: milestones survive, the rollup sheds (its counts are on the row).
    narrow = detail_footer(view, now=NOW, width=98)
    assert "prototype [completed]" in narrow.plain
    assert "sessions:" not in narrow.plain
    assert not narrow.plain.endswith(" ")
    # Every rung is a whole-clause prefix: no word is cut mid-way.
    for fitted_text in (fitted, narrow):
        for clause in fitted_text.plain.split("  \u00b7  "):
            assert clause == clause.strip()
        assert "prototy " not in fitted_text.plain


def test_detail_footer_truncates_explicitly_when_even_the_identity_overflows() -> None:
    """A name longer than the page is ellipsized, never silently cut.

    The STATUS CHIP is the whole token that survives when it can (U5): the
    name gives way first, and only a chip that cannot fit at all is ellipsized.
    """
    from rich.cells import cell_len

    view = _view("x" * 120)
    narrow = detail_footer(view, now=NOW, width=20)
    assert "\u2026 [active]" in narrow.plain
    assert cell_len(narrow.plain) <= 20
    tiny = detail_footer(view, now=NOW, width=6)
    assert tiny.plain.endswith("\u2026")
    assert cell_len(tiny.plain) <= 6


def test_detail_footer_omits_unknown_todo_counts() -> None:
    """UX U2: ``todos`` with null counts is omitted, not printed as ``None/None``."""
    view = _view("tiny", sessions=[_session("fa11bacc0001", "live", todos={})])
    text = detail_footer(view, now=NOW, width=400)
    assert "todos" not in text.plain
    known = _view(
        "tiny", sessions=[_session("fa11bacc0001", "live", todos={"open": 3, "total": 5})]
    )
    assert "todos 3/5" in detail_footer(known, now=NOW, width=400).plain


def test_detail_footer_marks_a_missing_session() -> None:
    """F5 / GUIDE: a linked session whose directory is gone renders ``missing``."""
    view = _view("stale-link", sessions=[_session("deadbeef0000", exists=False, todos=None)])
    text = detail_footer(view, now=NOW, width=400).plain
    assert "deadbeef0000 [missing]" in text
    assert "[stopped]" not in text


def test_footer_overdue_milestone_uses_the_late_token() -> None:
    """D3: one derived state, one colour — the timeline's ``!`` red, not amber."""
    seen: list[str] = []

    def resolve(key: str) -> Any:
        seen.append(key)
        return Style()

    view = _view(
        "late",
        milestones=[{"name": "beta", "target_date": "2025-06-01", "completed_at": None}],
    )
    detail_footer(view, now=NOW, style_for=resolve, width=400)
    assert "milestone_late" in seen
    assert "stale" not in seen


def test_status_chips_use_the_per_status_styles() -> None:
    """D4: the colour channel carries the scan — paused/done are not accent."""
    seen: list[str] = []

    def resolve(key: str) -> Any:
        seen.append(key)
        return Style()

    result = render_project_list(
        [_view("alpha"), _view("beta", status="paused"), _view("gamma", status="done")],
        cursor=0,
        style_for=resolve,
    )
    assert result.text.plain.count("[active]") == 1
    assert "status_active" in seen and "status_paused" in seen and "status_done" in seen


def test_truncate_row_caps_cells_not_characters() -> None:
    """F4: the cap bounds what the reader SEES — a CJK row wraps no more."""
    from rich.cells import cell_len

    from local_operator.projects import PROJECT_ROW_CAP, truncate_row

    row = "漢" * 120
    capped = truncate_row(row)
    assert capped.endswith("\u2026")
    assert cell_len(capped) <= PROJECT_ROW_CAP
    # The point of the fix: the CAP is in cells, so the row is 79 glyphs, not
    # 159 characters — a 160-character CJK row wrapped to three visual lines.
    assert len(capped) == 80


# -- the round-2 footer deltas -------------------------------------------------


def test_footer_does_not_paint_a_seam_for_an_empty_clause() -> None:
    """UX round 2, U4: a milestone-less project showed a doubled separator.

    The empty milestones clause still cost its seam (``·  ·``), and the five
    cells that phantom slot measured flipped the shed ladder on an 82-86-cell
    box, hiding a rollup that fits.
    """
    from rich.cells import cell_len

    view = _view(
        "annotator-overhaul",
        progress="prototype landed",
        progress_updated_at=NOW - 3,
        sessions=[_session("fa11bacc0001", "live")],
    )
    text = detail_footer(view, now=NOW).plain
    assert "·  ·" not in text
    # The phantom slot cannot flip the ladder: four cells narrower than the
    # composed row, the rollup still fits (it is shed only after the body).
    fitted = detail_footer(view, now=NOW, width=cell_len(text) - 4).plain
    assert "sessions: fa11bacc0001 [live]" in fitted
    assert fitted.endswith("…")


def test_footer_sheds_the_progress_body_before_the_session_rollup() -> None:
    """Agent review round 2: the 192-235 band kept the body and dropped the rollup."""
    from rich.cells import cell_len

    view = _long_footer_view()
    full = detail_footer(view, now=NOW).plain
    fitted = detail_footer(view, now=NOW, width=cell_len(full) - 10).plain
    assert "sessions: fa11bacc0001 [live]" in fitted
    assert "prototype landed" not in fitted
    assert "prototype [completed]" in fitted


def test_footer_keeps_the_status_chip_whole_when_the_name_overflows() -> None:
    """UX round 2, U5: a blind truncate left `… [p…` at 60 columns."""
    from rich.cells import cell_len

    view = _view("long-horizon-annotator-overhaul-with-many-many-words", status="paused")
    text = detail_footer(view, now=NOW, width=56).plain
    assert "[paused]" in text
    assert "[p…" not in text
    assert cell_len(text) <= 56
    assert text.endswith("[paused]")  # the chip is the survivor, the name gives way


def test_detail_footer_sheds_the_key_before_the_milestone_rollup() -> None:
    """D1: the key has another home; the rollup does not — so the key goes first."""
    view = _view(
        "payments-migration",
        title="Q4 Payments Migration",
        milestones=[
            {"name": "prototype", "target_date": None, "completed_at": "2026-01-01"},
            {"name": "cutover", "target_date": "2026-10-01", "completed_at": None},
        ],
    )
    # 96 cells is the 100x30 footer box (the D1 measurement): the rollup stays,
    # the key sheds.
    tight = detail_footer(view, width=96).plain
    assert "Q4 Payments Migration [active]" in tight
    assert "M 1/2" in tight and "[completed]" in tight
    assert "(payments-migration)" not in tight
    # Wide enough for identity-with-key plus the rollup, the key returns.
    wide = detail_footer(view, width=140).plain
    assert "Q4 Payments Migration (payments-migration) [active]" in wide
    assert "M 1/2" in wide


def test_detail_footer_marks_shed_content_down_to_the_bare_identity() -> None:
    """D4: the marker now reaches the keyless rung; only the chip tight band stays bare."""
    view = _view("payments-migration", title="Q4 Payments Migration")
    marked = detail_footer(view, width=40).plain
    assert marked == "Q4 Payments Migration [active] …"
    # Below the bare identity the chip already spends the width (the tight
    # rebuild) — the marker-less band is only 27-32 cells now (was 27-49).
    bare = detail_footer(view, width=31).plain
    assert bare == "Q4 Payments Migration [active]"


# -- S3b: the marker column and the reveal's geometry -------------------------


def test_every_canvas_marks_the_associated_set_and_the_selection() -> None:
    """`◆` = the calling session's own; `▸` = the selection; cursor wins."""
    dated_a = _view("alpha", start_date="2026-05-01", target_date="2026-09-01")
    middle = _view("beta", start_date="2026-06-01", target_date="2026-10-01")
    dated_c = _view("gamma", start_date="2026-07-01", target_date="2026-11-01")
    views = [dated_a, middle, dated_c]
    mine = frozenset({str(dated_a["project"]["id"]), str(dated_c["project"]["id"])})
    listed = render_project_list(views, cursor=1, associated=mine).text.plain.splitlines()
    assert listed[0].startswith("◆ alpha")
    assert listed[1].startswith("▸ beta")  # the selection owns the column
    assert listed[2].startswith("◆ gamma")
    board = render_project_board(views, cursor=1, associated=mine).text.plain
    assert "◆ alpha" in board and "◆ gamma" in board and "▸ beta" in board
    timeline = render_project_timeline(views, cursor=0, associated=mine).text.plain
    # cursor=0 is alpha AND alpha is associated: it carries BOTH markers
    # (design round 1, D3 — the cursor must not hide the diamond), and gamma
    # keeps its own.
    assert "▸◆alpha" in timeline
    assert "◆ gamma" in timeline
    assert "  beta" in timeline


def test_position_helpers_name_cells_the_canvases_actually_paint() -> None:
    """The reveal's geometry is the painters' own, cross-checked against them."""
    from local_operator.tui.projects_render import board_position, timeline_position

    dated = _view("alpha", start_date="2026-05-01", target_date="2026-09-01")
    undated = _view("beta")
    views = [dated, undated]
    # Board: `active` column, first card's name line, then the second card's.
    assert board_position(views, 0) == (0, 2)
    assert board_position(views, 1) == (0, 6)
    # Timeline: alpha under the axis; beta on the no-dates tail line, after the
    # blank the renderer inserts.
    assert timeline_position(views, 0) == (0, 1)
    assert timeline_position(views, 1) == (0, 3)
    timeline_line = timeline_position(views, 0)
    tail_line = timeline_position(views, 1)
    board_line = board_position(views, 0)
    assert timeline_line is not None and tail_line is not None and board_line is not None
    lines = render_project_timeline(views).text.plain.splitlines()
    assert "alpha" in lines[timeline_line[1]]
    assert "beta" in lines[tail_line[1]]
    assert "alpha" in render_project_board(views).text.plain.splitlines()[board_line[1]]


def test_position_helpers_answer_none_for_rows_nothing_paints() -> None:
    """Past the caps the answer is `None` — the page clamps instead of aiming."""
    from local_operator.tui.projects_render import board_position, timeline_position

    many = [_view(f"p{i:03d}") for i in range(PROJECTS_MAX + 2)]
    assert board_position(many, PROJECTS_MAX + 1) is None
    assert timeline_position(many, PROJECTS_MAX + 1) is None
    assert board_position([], 0) is None
    assert timeline_position([], 0) is None


# -- round 1 remediation: both markers, the tail, the cap ---------------------


def test_a_selected_associated_row_carries_both_markers() -> None:
    """D3: the cursor must not hide the `◆` — the count matches the diamonds."""
    view = _view("alpha", start_date="2026-05-01", target_date="2026-09-01")
    mine = frozenset({str(view["project"]["id"])})
    listed = render_project_list([view], cursor=0, associated=mine).text.plain
    assert listed.startswith("▸◆alpha")
    board = render_project_board([view], cursor=0, associated=mine).text.plain
    assert "▸◆alpha" in board
    timeline = render_project_timeline(
        [view], cursor=0, associated=mine, today=date(2026, 3, 1)
    ).text.plain
    assert "▸◆alpha" in timeline


def test_timeline_tail_marks_the_selection_and_the_set() -> None:
    """Agent review r1 F1: an undated project is markable on the tail line."""
    dated = _view("ship", start_date="2026-01-05", target_date="2026-01-30")
    other = _view("someday")
    mine = frozenset({str(other["project"]["id"])})
    result = render_project_timeline(
        [dated, other], cursor=1, associated=mine, today=date(2026, 3, 1)
    )
    assert result.text.plain.splitlines()[-1] == "no dates (1): ▸◆someday"
    # Unmarked names stay bare — the tail has no fixed column to align.
    plain = render_project_timeline([dated, other], today=date(2026, 3, 1))
    assert plain.text.plain.splitlines()[-1] == "no dates (1): someday"


def test_past_the_board_cap_the_selected_card_is_still_painted() -> None:
    """UX r1 U3: no silent clamp — the `↵` object stays visible, the count honest."""
    from local_operator.tui.projects_render import board_position

    views = [_view(f"p{index:02d}") for index in range(BOARD_CARDS_MAX + 3)]
    selected_index = BOARD_CARDS_MAX + 1
    mine = frozenset(
        {
            str(views[selected_index]["project"]["id"]),
            str(views[selected_index + 1]["project"]["id"]),
        }
    )
    result = render_project_board(views, cursor=selected_index, associated=mine)
    canvas = result.text.plain
    # 15 cards, cap 12: the selected one is painted, the note counts the rest
    # (including how many of them are the session's own).
    assert "… +2 more (1 yours)" in canvas
    assert f"▸◆p{selected_index:02d}" in canvas
    position = board_position(views, selected_index)
    assert position is not None
    lines = canvas.splitlines()
    assert f"p{selected_index:02d}" in lines[position[1]]


def test_the_cap_note_disappears_when_nothing_is_left_hidden() -> None:
    """R2-1: the selected card alone past the cap must not print `… +0 more`."""
    from local_operator.tui.projects_render import board_position

    views = [_view(f"p{index:02d}") for index in range(BOARD_CARDS_MAX + 1)]
    selected_index = BOARD_CARDS_MAX  # the only card past the cap
    result = render_project_board(views, cursor=selected_index)
    canvas = result.text.plain
    assert "+0 more" not in canvas
    assert f"▸ p{selected_index:02d}" in canvas
    # The position helper mirrors the note-less layout: 2 + 4 * V.
    position = board_position(views, selected_index)
    assert position is not None
    assert position[1] == 2 + 4 * BOARD_CARDS_MAX
    assert f"p{selected_index:02d}" in canvas.splitlines()[position[1]]


def test_board_key_budget_counts_the_marker_column() -> None:
    """D5: the two marker cells share the name cell, so `label (key)` has 29."""
    from local_operator.tui.projects_render import cell_len

    kept = _view("keeptest", title="A" * 18)
    shed = _view("shedtest", title="B" * 19)
    assert cell_len("A" * 18 + " (keeptest)") == 29
    assert cell_len("B" * 19 + " (shedtest)") == 30
    plain = render_project_board([kept, shed], now=NOW).text.plain
    assert "A" * 18 + " (keeptest)" in plain  # exactly the budget: key kept
    assert " (shedtest)" not in plain  # one past it: the title wins alone
    assert "B" * 19 in plain
