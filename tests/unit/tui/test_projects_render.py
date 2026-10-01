"""The three project canvases: geometry and content, asserted as plain strings.

``projects_render`` exists so these assertions can be written at all — the
widget is a scroll container around what it returns, so the grid arithmetic,
the caps and the truncation rules are checked here rather than guessed from a
frame. The one thing a screenshot can still prove (that the widget pins its
canvas to these numbers) is asserted in ``test_projects_view.py``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pytest
from rich.cells import cell_len
from rich.style import Style

from local_operator import projects as projects_mod
from local_operator.tui import projects_render
from local_operator.tui.projects_render import (
    BOARD_CARDS_MAX,
    BOARD_COLUMN_WIDTH,
    PROJECTS_MAX,
    TIMELINE_NAME_WIDTH,
    aggregate_footer,
    auto_timeline_tier,
    board_position,
    detail_footer,
    detail_milestone_row_text,
    list_position,
    project_at,
    render_project_board,
    render_project_list,
    render_project_timeline,
    section_at,
    section_header_at,
    section_ruler,
    sections_of,
    timeline_axis_cells,
    timeline_position,
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
        "▸ alpha [● active] · est 13pt · →2026-10-15 · M 1/2 · 1 session (1 live) "
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
    assert plain.startswith("▸ Q4 Payments Migration (payments-migration) [● active]")

    # No title → the key is the label (every surface's fallback).
    fallback = render_project_list([_view("plain-key")], cursor=0, now=NOW).text.plain
    assert fallback.startswith("▸ plain-key [● active]")


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
    # D4: the bucket column is LABELLED for what it holds — planning/qa/
    # validation ride it, so "active · 1" over one true active was two things
    # called active on one screen.
    assert "in flight · 1" in header
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
    assert lines[0].startswith("in flight · 1")
    assert lines[2] == "  solo [● active]"
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
    # The chip spends 10 of the 29 content cells, so `Short (k)` fits beside it
    # and the long title is ellipsized to make room for the chip (D1: the chip
    # is never shed).
    assert "Short (k) [● active]" in plain
    assert "Q4 Payments Migra… [● active]" in plain
    assert "(payments-migration)" not in plain  # too wide — the key never joins


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


def test_the_footer_rows_and_tool_share_one_local_day_on_an_evening_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent review round 5: ONE basis, or the footer contradicts the rest.

    Evening probe, west of Greenwich: the local day is the 29th while UTC has
    rolled to the 30th. ``projects_render.today_iso`` was the last UTC reader,
    so the footer called a 2026-09-30 milestone ``[overdue]`` while the page
    rows and the tool receipt called it ``upcoming``. The fake clock keeps the
    two days distinct whatever the wall clock says, so a revert to UTC fails
    here.
    """

    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return datetime(2026, 9, 29, 20, 55)
            return datetime(2026, 9, 30, 0, 55, tzinfo=timezone.utc)

    monkeypatch.setattr(projects_mod, "datetime", FakeDatetime)

    milestone = {"name": "spec shipped", "target_date": "2026-09-30", "completed_at": None}
    footer = detail_footer(_view(milestones=[milestone])).plain
    assert "[upcoming]" in footer

    row = detail_milestone_row_text(milestone, selected=False).plain
    assert "upcoming" in row

    assert projects_mod.milestone_state(None, "2026-09-30") == "upcoming"

    # One basis, pinned against a revert: both readers must say the LOCAL 29th,
    # never UTC's 30th.
    assert projects_render.today_iso() == "2026-09-29"
    assert projects_mod._today_iso() == "2026-09-29"


def test_update_day_label_says_today_yesterday_and_a_dated_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The day-group header: relative for the reader's own days, dated beyond."""
    import datetime as dt

    from local_operator import projects as projects_mod
    from local_operator.tui.projects_render import update_day_label

    class FakeDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                dt.datetime(2026, 9, 30, 9, 0)
                if tz is None
                else dt.datetime(2026, 9, 30, 13, 0, tzinfo=dt.timezone.utc)
            )

    monkeypatch.setattr(projects_mod, "datetime", FakeDatetime)
    # Midday UTC, so the LOCAL day is the same one in any westward zone: the
    # 02:00Z stamps of an evening-written entry belong to the day before.
    assert update_day_label("2026-09-30T12:00:00Z") == "today"
    assert update_day_label("2026-09-29T12:00:00Z") == "yesterday"
    assert update_day_label("2026-09-25T12:00:00Z") == "25 Sep"
    # Another year carries the year, or "25 Sep" reads as the next one.
    assert update_day_label("2025-09-25T12:00:00Z") == "25 Sep 2025"
    assert update_day_label("not a date") == "not a date"


def test_update_stamp_names_the_reporter_on_the_footers_vocabulary() -> None:
    """Spec §7.3: ``HH:MM · {reporter}``, one vocabulary with the footer."""
    from local_operator.tui.projects_render import (
        update_day_label,
        update_reporter_text,
        update_stamp_text,
    )

    assert update_reporter_text("ab12cd34ef56") == "session ab12cd34ef56"
    assert update_reporter_text("operator") == "the operator"
    assert update_reporter_text("scout") == "scout"
    assert update_reporter_text("") == ""
    assert update_day_label("bad") == "bad"
    entry = {"at": "2026-09-27T17:52:00Z", "by": "ab12cd34ef56"}
    compact = update_stamp_text(entry).plain
    assert compact.endswith("· session ab12cd34ef56")
    # The clock is the READER's, so the expected hour comes from the same
    # conversion rather than a hard-coded string that only passes in one zone.
    from local_operator.tui.projects_render import update_local_moment

    moment = update_local_moment("2026-09-27T17:52:00Z")
    assert moment is not None
    assert compact.startswith(moment.strftime("%H:%M"))
    assert "2026-09-27" in update_stamp_text(entry, selected=True).plain
    # No reporter at all: the stamp is the time alone.
    assert " · " not in update_stamp_text({"at": "2026-09-27T17:52:00Z"}).plain


def test_attachment_rows_carry_kind_size_path_and_the_missing_marker() -> None:
    """Spec §7.4: the honest fallback — kind, name, size, path, or the marker."""
    from local_operator.tui.projects_render import (
        attachment_path_text,
        attachment_row_text,
    )

    image = {"kind": "image", "name": "board-60x20.png", "bytes": 83904}
    assert attachment_row_text(image).plain == "[img] board-60x20.png · 81.9 KB"
    data = {"kind": "data", "name": "notes.md", "bytes": 12288}
    assert attachment_row_text(data).plain == "[file] notes.md · 12.0 KB"
    # An unknown kind reads the store's own default word rather than a blank.
    assert attachment_row_text({"name": "x"}).plain.startswith("[file] x")
    # A size the store never recorded is omitted, never printed as "0 B".
    assert "·" not in attachment_row_text({"name": "x"}).plain
    path = attachment_path_text({"path": "/tmp/a/b.png"}).plain
    assert path.strip() == "→ /tmp/a/b.png"
    assert (
        "[missing on disk]" in attachment_path_text({"path": "/tmp/a/b.png", "missing": True}).plain
    )
    assert "(no path recorded)" in attachment_path_text({}).plain


def test_the_clamp_marker_names_the_key_that_actually_toggles_it() -> None:
    """A hinted key must do what it says — nowhere more than on this marker."""
    from local_operator.tui.projects_render import (
        UPDATE_BODY_LINES,
        older_updates_text,
        update_body_is_clamped,
        update_body_lines,
        update_more_lines_text,
    )

    assert update_more_lines_text(4, expanded=False).plain == "[4 more lines — ↵ expand]"
    assert update_more_lines_text(4, expanded=True).plain == "[4 more lines — ↵ collapse]"
    # Inflected: `1 more line` (agent review round 1, NIT-1).
    assert update_more_lines_text(1, expanded=False).plain == "[1 more line — ↵ expand]"
    assert older_updates_text(7).plain == "… 7 older updates"
    body = "\n".join(f"line {n}" for n in range(1, UPDATE_BODY_LINES + 3))
    assert len(update_body_lines(body)) == UPDATE_BODY_LINES + 2
    assert update_body_lines("a\r\nb") == ["a", "b"]
    # The predicate behind the row's verb (design D1): fits == no tail.
    assert update_body_is_clamped("a\nb") is False
    assert update_body_is_clamped("\n".join(str(n) for n in range(UPDATE_BODY_LINES))) is False
    assert update_body_is_clamped(body) is True


def test_the_marker_wears_muted_not_dim() -> None:
    """Design review round 1, D4: the feed's one affordance must read as one.

    `dim` is the quiet layer (paths, day headers) and sits at 3.77:1 on the
    light ground; `muted` is 8.62 dark / 7.18 light.
    """
    from rich.style import Style

    from local_operator.tui.projects_render import update_more_lines_text

    seen: list[str] = []

    def resolver(key: str) -> Style:
        seen.append(key)
        return Style()

    update_more_lines_text(2, expanded=False, style_for=resolver)
    assert seen == ["muted"], seen


def test_the_path_row_abbreviates_home_and_keeps_the_file_tail() -> None:
    """Design review round 1, D2: the tail is what tells one row from another.

    At 100 columns the stored path is 126 cells against a 95-cell box, so a
    right-truncation cut it inside the unique hex and every attachment row read
    ``…/attachments/<cut>…``.
    """
    from pathlib import Path

    from local_operator.tui.projects_render import attachment_path_text

    home = str(Path.home())
    attachment = {"path": f"{home}/.local-operator/projects/attachments/6870f1/eb904c99aa.png"}
    unmeasured = attachment_path_text(attachment).plain
    assert unmeasured.startswith("    → ~/.local-operator/")
    fitted = attachment_path_text(attachment, width=60).plain
    assert "…" in fitted
    assert fitted.endswith(".png"), fitted
    assert cell_len(fitted) <= 60
    # The missing marker is reserved BEFORE the path is cut, so the caveat
    # survives at the width the page has.
    with_marker = attachment_path_text({**attachment, "missing": True}, width=60).plain
    assert with_marker.endswith("[missing on disk]"), with_marker
    assert cell_len(with_marker) <= 60
    # A path that already fits is untouched.
    assert "…" not in attachment_path_text({"path": "~/a/b.png"}, width=95).plain


def test_detail_footer_names_progress_reporter_and_staleness() -> None:
    view = _view(
        progress="cutover done",
        progress_updated_at=NOW - 7200,
        progress_reported_by="4e92693767fa",
    )
    view["progress_stale"] = True
    text = detail_footer(view, now=NOW).plain
    assert "alpha [● active]" in text
    assert "progress reported 2h ago by 4e92693767fa · stale: cutover done" in text
    assert "no linked sessions" in text


def test_detail_footer_shows_title_first_with_the_key() -> None:
    view = _view("payments-migration", title="Q4 Payments Migration")
    text = detail_footer(view, now=NOW).plain
    assert text.startswith("Q4 Payments Migration (payments-migration) [● active]")


def test_detail_footer_trades_the_key_away_before_the_title() -> None:
    view = _view("payments-migration", title="Q4 Payments Migration")
    # Wide: both. Too tight for both but enough for title + chip: the key
    # sheds first (`title wins when only one fits`), the chip survives whole.
    wide = detail_footer(view, now=NOW, width=200).plain
    assert "Q4 Payments Migration (payments-migration) [● active]" in wide
    tight = detail_footer(view, now=NOW, width=34).plain
    assert "(payments-migration)" not in tight
    assert tight.startswith("Q4 Payments Migration [● active]")


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
    assert "\u2026 [● active]" in narrow.plain
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
    assert result.text.plain.count("[● active]") == 1
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
    assert "[‖ paused]" in text
    assert "[‖…" not in text
    assert cell_len(text) <= 56
    assert text.endswith("[‖ paused]")  # the chip is the survivor, the name gives way


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
    assert "Q4 Payments Migration [● active]" in tight
    assert "M 1/2" in tight and "[completed]" in tight
    assert "(payments-migration)" not in tight
    # Wide enough for identity-with-key plus the rollup, the key returns.
    wide = detail_footer(view, width=140).plain
    assert "Q4 Payments Migration (payments-migration) [● active]" in wide
    assert "M 1/2" in wide


def test_detail_footer_marks_shed_content_down_to_the_bare_identity() -> None:
    """D4: the marker now reaches the keyless rung; only the chip tight band stays bare."""
    from rich.cells import cell_len

    view = _view("payments-migration", title="Q4 Payments Migration")
    marked = detail_footer(view, width=40).plain
    assert marked == "Q4 Payments Migration [● active] …"
    # Below the bare identity the chip already spends the width (the tight
    # rebuild): 31 cells hold a 10-cell chip, a space and a truncated name.
    bare = detail_footer(view, width=31).plain
    assert bare.endswith("[● active]")
    assert "…" in bare
    assert cell_len(bare) == 31


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


def test_board_key_budget_shares_the_line_with_the_chip() -> None:
    """D1/D5: the chip is never shed, so `label (key)` budgets the rest."""
    from local_operator.tui.projects_render import cell_len, status_chip_text

    chip = status_chip_text("active")
    # 31 cells = 2 marker cells + label + key + separator space + chip.
    budget = 31 - 2 - cell_len(chip) - 1
    assert budget == 18
    kept = _view("keeptest", title="A" * (budget - len(" (keeptest)")))
    shed = _view("shedtest", title="B" * (budget - len(" (shedtest)") + 1))
    plain = render_project_board([kept, shed], now=NOW).text.plain
    # Exactly at the budget: the key stays. One cell past it: the key sheds
    # and the title keeps its whole self beside the chip.
    assert f"{'A' * (budget - len(' (keeptest)'))} (keeptest) [● active]" in plain
    assert " (shedtest)" not in plain
    assert f"{'B' * (budget - len(' (shedtest)') + 1)} [● active]" in plain


def test_board_buckets_lifecycle_statuses_into_the_in_flight_column() -> None:
    views = [
        _view("qa-run", status="qa"),
        _view("plan-new", status="planning"),
        _view("val-ship", status="validation"),
    ]
    board = render_project_board(views, now=NOW).text.plain
    header = board.splitlines()[0]
    assert "in flight · 3" in header  # they ride the in-flight column...
    assert "qa ·" not in header and "planning ·" not in header  # ...not columns
    # D1: every card carries its OWN exact chip (glyph + word) — the column
    # tells the coarse story, the chip stays exact.
    assert "[◐ qa]" in board and "[○ planning]" in board and "[◉ validation]" in board


def test_board_falls_back_to_the_leading_column_for_unknown_statuses() -> None:
    board = render_project_board([_view("mystery", status="shipped")], now=NOW).text.plain
    assert "mystery" in board  # a newer build's status must not vanish
    assert "[? shipped]" in board  # ...and it says it does not know the word


def test_lifecycle_chips_paint_in_the_footer() -> None:
    assert "[◐ qa]" in detail_footer(_view("qa-run", status="qa"), now=NOW, width=120).plain
    assert (
        "[◉ validation]"
        in detail_footer(_view("val-ship", status="validation"), now=NOW, width=120).plain
    )


def test_aggregate_footer_counts_every_true_status() -> None:
    from local_operator.tui.projects_render import aggregate_footer

    views = [
        _view("q", status="qa"),
        _view("p", status="planning"),
        _view("a", status="archived"),
        _view("x", status="active"),
    ]
    text = aggregate_footer(views).plain
    assert "1 planning" in text and "1 active" in text and "1 qa" in text
    assert "1 archived" in text
    assert "paused" not in text


# -- round 3: the glyph family, the ladder and the per-card chips --------------


def test_every_status_glyph_is_one_cell_and_its_chip_is_the_one_composition() -> None:
    from rich.cells import cell_len

    from local_operator.tui.projects_render import STATUS_GLYPHS, status_chip_text

    for status, glyph in STATUS_GLYPHS.items():
        assert cell_len(glyph) == 1, (status, glyph)
        assert status_chip_text(status) == f"[{glyph} {status}]"
    # An unknown status keeps a chip: the '?' glyph says the word is not ours.
    assert status_chip_text("shipped") == "[? shipped]"
    # The chip is its own whole token — the screen keeps it intact.
    for status in [*STATUS_GLYPHS, "shipped"]:
        chip = status_chip_text(status)
        assert cell_len(chip) == len(chip)


def test_the_board_card_carries_every_statuses_exact_chip() -> None:
    from local_operator.tui.projects_render import STATUS_GLYPHS, status_chip_text

    views = [_view(f"card-{status}", status=status) for status in STATUS_GLYPHS]
    board = render_project_board(views, now=NOW).text.plain
    for status in STATUS_GLYPHS:
        assert status_chip_text(status) in board, status


def test_the_counts_line_sheds_whole_segments_at_60_100_150() -> None:
    from rich.cells import cell_len

    from local_operator.tui.projects_render import aggregate_footer

    views = [
        _view("pl", status="planning"),
        _view("ac", status="active", sessions=[_session(state="live")]),
        _view("qa", status="qa"),
        _view("va", status="validation"),
        _view("pa", status="paused"),
        _view("do", status="done"),
        _view("ar", status="archived"),
    ]
    full = aggregate_footer(views, style_for=lambda key: Style()).plain
    assert full == (
        "7 projects · 1 planning · 1 active · 1 qa · 1 validation · 1 paused "
        "· 1 done · 1 archived · 1 live session"
    )
    # 150: everything fits.
    assert aggregate_footer(views, style_for=lambda key: Style(), width=150).plain == full
    # 100: the live clause shortens to the card idiom ("1 live"); nothing is
    # cut mid-word (the old bug's "· 0" tail).
    at_100 = aggregate_footer(views, style_for=lambda key: Style(), width=100).plain
    assert at_100.endswith("1 live")
    assert cell_len(at_100) <= 100
    # 60: the counts themselves ellipsize at a SEGMENT boundary with an
    # explicit marker — a cut line says more exists.
    at_60 = aggregate_footer(views, style_for=lambda key: Style(), width=60).plain
    assert at_60 == "7 projects · 1 planning · 1 active · 1 qa · 1 validation · …"
    assert cell_len(at_60) <= 60
    # Below that the segment list is a STRICT PREFIX — one that cannot fit
    # ends the line; a shorter later segment must never be skipped in (the
    # "1 qa · 1 paused · …" shape that silently dropped validation).
    at_56 = aggregate_footer(views, style_for=lambda key: Style(), width=56).plain
    assert at_56 == "7 projects · 1 planning · 1 active · 1 qa · …"
    assert "paused" not in at_56
    # With a tier the zoom clause sheds before anything else does.
    tiered = aggregate_footer(views, tier="month", style_for=lambda key: Style(), width=150).plain
    assert tiered.endswith("· zoom: month")
    tiered_100 = aggregate_footer(
        views, tier="month", style_for=lambda key: Style(), width=100
    ).plain
    assert "zoom" not in tiered_100 and tiered_100.endswith("1 live")


# -- S6d: team-grouped sections (the parity slice) ----------------------------


def _grouped_store() -> list[dict[str, Any]]:
    """The design's own store shape: two teams, one teamless, dated rows both ways."""
    return [
        _view(
            "parity-spec",
            title="TUI parity spec",
            team="core",
            status="active",
            estimate=5.0,
            target_date="2026-10-04",
            progress="Spec reviewed",
            progress_updated_at=NOW - 7200,
        ),
        _view("board-entry", title="Board entry", team="core", target_date="2026-09-28"),
        _view("desktop-tab", title="Desktop tab", team="core", status="qa"),
        _view("classifier", team="core", status="done"),
        _view("mobile-sheet", title="Mobile sheet", team="personal", status="done"),
        _view("references", status="paused"),
    ]


def test_sections_order_teams_case_insensitively_with_no_team_last() -> None:
    views = [
        _view("z", team="Zeta"),
        _view("a", team="alpha"),
        _view("n"),
        _view("b", team="beta"),
    ]
    assert sections_of(views) == [
        ("alpha", [1]),
        ("beta", [3]),
        ("Zeta", [0]),
        ("no team", [2]),
    ]
    # No team anywhere: no sections at all — the canvases render flat, which
    # is the byte-identical path the shipped tests above already pin.
    assert sections_of([_view("a"), _view("b")]) is None


def test_sections_cover_only_the_painted_rows() -> None:
    views = [_view(f"p{index}", team="core") for index in range(PROJECTS_MAX + 2)]
    assert sections_of(views) == [("core", list(range(PROJECTS_MAX)))]


def test_teamless_canvases_report_no_sections() -> None:
    flat = [_view("alpha"), _view("beta")]
    assert render_project_list(flat, cursor=0).sections == ()
    assert render_project_board(flat, cursor=0).sections == ()
    assert render_project_timeline(flat, cursor=0).sections == ()


def test_grouped_list_paints_headers_around_the_store_order() -> None:
    views = _grouped_store()
    result = render_project_list(views, cursor=1, now=NOW)
    lines = result.text.plain.splitlines()
    assert lines[0].startswith("── core · 4 ")
    assert lines[5].startswith("── personal · 1 ")
    assert lines[7].startswith("── no team · 1 ")
    # Headers fill the canvas's own width (the widest row's edge).
    assert cell_len(lines[0]) == result.width
    # Sections keep the store's order (grouping is not a second sort) and the
    # count is the section's painted rows.
    assert "parity-spec" in lines[1] and "classifier" in lines[4]
    assert "mobile-sheet" in lines[6] and "references" in lines[8]
    assert result.sections == ((0, 5, "core"), (5, 7, "personal"), (7, 9, "no team"))
    # The cursor's canvas row is no longer its index: the reveal and the
    # clicks both read this table.
    assert [list_position(views, index) for index in range(6)] == [1, 2, 3, 4, 6, 8]


def test_grouped_board_stacks_a_block_per_section_with_nonempty_columns() -> None:
    views = [
        *_grouped_store(),
        _view("old-thing", title="Old thing", team="core", status="archived"),
    ]
    result = render_project_board(views, cursor=0, now=NOW)
    lines = result.text.plain.splitlines()
    assert lines[0].startswith("── core · 5 ")
    assert "in flight · 3" in lines[1] and "done · 1" in lines[1]
    assert "archived · 1" in lines[1]
    personal = next(
        index for index, line in enumerate(lines) if line.startswith("── personal · 1 ")
    )
    # Personal paints ONLY its done column: a column carrying no cards in a
    # section is not painted at all (no `paused · 0` under a populated block).
    assert lines[personal + 1].startswith("done · 1")
    assert "in flight" not in lines[personal + 1]
    assert "paused · 0" not in result.text.plain and "archived · 0" not in result.text.plain
    no_team = next(index for index, line in enumerate(lines) if line.startswith("── no team · 1 "))
    assert lines[no_team + 1].startswith("paused · 1")
    assert [label for _start, _end, label in result.sections] == ["core", "personal", "no team"]


def test_grouped_board_positions_land_on_the_painted_cards() -> None:
    views = _grouped_store()
    result = render_project_board(views, cursor=0, now=NOW)
    lines = result.text.plain.splitlines()
    names = [
        "TUI parity spec",
        "Board entry",
        "Desktop tab",
        "classifier",
        "Mobile sheet",
        "references",
    ]
    for index, name in enumerate(names):
        position = board_position(views, index)
        assert position is not None, index
        x, y = position
        assert lines[y][x:].lstrip("▸ ◆ ").startswith(name), (index, lines[y])
    # The second column's card sits a full pitch right of the first's, on the
    # same card line.
    first = board_position(views, 0)
    second = board_position(views, 3)
    assert first is not None and second is not None
    assert second == (BOARD_COLUMN_WIDTH, first[1])


def test_grouped_board_position_paints_the_past_cap_selected_card() -> None:
    rows = [_view(f"c{index}", team="core") for index in range(BOARD_CARDS_MAX + 1)]
    views = [*rows, _view("other", team="zeta")]
    selected = BOARD_CARDS_MAX  # the (BOARD_CARDS_MAX + 1)-th card: past the cap
    result = render_project_board(views, cursor=selected, now=NOW)
    lines = result.text.plain.splitlines()
    position = board_position(views, selected)
    assert position is not None
    x, y = position
    assert lines[y][x:].lstrip("▸ ◆ ").startswith("c12")
    # The extra card pushes the LATER section down. With the cursor on the
    # past-cap card, core paints its header, 12 card rows and the extra card;
    # with the cursor in the later section, the same card is only COUNTED by
    # the overflow note, two lines shorter.
    core_height = 1 + (1 + 4 * BOARD_CARDS_MAX + 2)  # section header + block
    zeta_name = core_height + 3  # zeta's header + its column header + the blank
    late = board_position(views, BOARD_CARDS_MAX + 1)
    assert late is not None and late[1] == zeta_name
    late_view = render_project_board(views, cursor=BOARD_CARDS_MAX + 1, now=NOW)
    late_lines = late_view.text.plain.splitlines()
    assert "other" in late_lines[late[1]]
    # Without the selection the same card is only COUNTED by the overflow
    # note, and the later section sits two lines higher.
    assert any("… +1 more" in line for line in late_lines)


def test_grouped_timeline_interrupts_dated_rows_and_splits_the_tail() -> None:
    views = _grouped_store()
    result = render_project_timeline(views, tier="month", cursor=0, now=NOW)
    lines = result.text.plain.splitlines()
    # ONE count meaning across the screen (U2): the header carries the
    # section's painted total (2 dated + 2 undated), and the ruler agrees;
    # the tail line's parenthetical counts the names it lists.
    assert lines[1].startswith("── core · 4 ")
    ruler = section_ruler(views, "timeline", top_row=1, cursor=0, width=80)
    assert ruler is not None and ruler.plain.startswith("── core · 4 ")
    # The name column shows the title first (the key joins only when the
    # 24-cell column holds it — D2's tight-fit rule).
    assert "TUI parity spec" in lines[2] and "Board entry" in lines[3]
    assert lines[4] == ""
    assert lines[5].startswith("no dates (core, 2): ")
    assert "Desktop tab" in lines[5] and "classifier" in lines[5]
    assert lines[6].startswith("no dates (personal, 1): ") and "Mobile sheet" in lines[6]
    assert lines[7].startswith("no dates (no team, 1): ") and "references" in lines[7]
    assert result.sections == (
        (1, 4, "core"),
        (5, 6, "core"),
        (6, 7, "personal"),
        (7, 8, "no team"),
    )


def test_grouped_timeline_positions_land_on_the_painted_rows() -> None:
    views = _grouped_store()
    result = render_project_timeline(views, tier="month", cursor=0, now=NOW)
    lines = result.text.plain.splitlines()
    names = [
        "TUI parity spec",
        "Board entry",
        "Desktop tab",
        "classifier",
        "Mobile sheet",
        "references",
    ]
    for index, name in enumerate(names):
        position = timeline_position(views, index)
        assert position is not None, index
        _x, y = position
        assert name in lines[y], (index, lines[y])


def test_project_at_maps_cells_back_to_the_painted_projects() -> None:
    views = _grouped_store()
    # List rows address their whole row; headers are not rows.
    assert project_at(views, "list", 3, 1) == 0
    assert project_at(views, "list", 3, 0) is None
    assert project_at(views, "list", 3, 6) == 4
    # Board cards their three content lines; the blank above a card, the
    # gutter between columns and the header row are not cards.
    card = board_position(views, 3)
    assert card is not None
    x, y = card
    assert project_at(views, "board", x + 2, y) == 3
    assert project_at(views, "board", x + 2, y - 1) is None
    assert project_at(views, "board", BOARD_COLUMN_WIDTH - 1, y) is None
    # The timeline's dated rows address the row; its tail lines pick the NAME
    # the click landed on (one line lists several projects).
    result = render_project_timeline(views, tier="month", cursor=0, now=NOW)
    lines = result.text.plain.splitlines()
    assert project_at(views, "timeline", 0, 0) is None  # the axis row
    assert project_at(views, "timeline", 0, 2) == 0
    tail_position = timeline_position(views, 3)
    assert tail_position is not None
    tail_y = tail_position[1]
    assert project_at(views, "timeline", lines[tail_y].index("classifier") + 1, tail_y) == 3
    assert project_at(views, "timeline", 0, tail_y) is None  # the prefix is not a name


def test_section_lookups_cover_headers_and_bands() -> None:
    views = _grouped_store()
    assert section_header_at(views, "list", 0, 0) == ("core", 0)
    assert section_header_at(views, "list", 0, 1) is None
    assert section_at(views, "list", 0, 3) == "core"
    assert section_at(views, "list", 0, 5) == "personal"
    assert section_at(views, "list", 0, 9) is None
    assert section_at(views, "board", 0, 0) == "core"
    assert section_header_at(views, "timeline", 0, 1) == ("core", 0)
    # Ungrouped canvases have no sections to look up.
    flat = [_view("alpha")]
    assert section_at(flat, "list", 0, 0) is None
    assert section_header_at(flat, "board", 0, 0) is None


def test_a_team_named_no_team_cannot_alias_the_sentinel() -> None:
    """R1-2/U1: the sentinel and a real `no team` stay distinguishable."""
    views = [
        _view("alpha", team="core"),
        _view("binary", team="no team"),
        _view("charlie", team="no team"),
        _view("delta"),  # teamless → the sentinel bucket
    ]
    assert sections_of(views) == [
        ("core", [0]),
        ("no team", [1, 2]),
        ("no team (unset)", [3]),
    ]
    lines = render_project_list(views, cursor=0).text.plain.splitlines()
    assert lines[0].startswith("── core · 1 ")
    assert lines[2].startswith("── no team · 2 ")
    assert lines[5].startswith("── no team (unset) · 1 ")
    # Counts stay per-group (the old bug printed the sentinel's 1 over the
    # real team's two rows) and the sel clause still fires between them.
    ruler = section_ruler(views, "list", top_row=5, cursor=0, width=60)
    assert ruler is not None
    assert ruler.plain.startswith("── no team (unset) · 1 · sel core ")
    # Header clicks carry their own band's target: the lower header can no
    # longer jump into the upper section (nor the upper into the lower).
    assert section_header_at(views, "list", 0, 2) == ("no team", 1)
    assert section_header_at(views, "list", 0, 5) == ("no team (unset)", 3)
    # Tails name each membership with its own label (no cursor: no markers).
    tail_lines = [
        line
        for line in render_project_timeline(
            views, tier="month", cursor=None, now=NOW
        ).text.plain.splitlines()
        if line.startswith("no dates")
    ]
    assert tail_lines == [
        "no dates (core, 1): alpha",
        "no dates (no team, 2): binary, charlie",
        "no dates (no team (unset), 1): delta",
    ]


def test_the_ruler_survives_a_board_past_the_project_cap() -> None:
    """R1-1: >PROJECTS_MAX boards — the overflow row shifts the bands, and
    the ruler must read the bands the painter drew."""
    views = [_view(f"core-{index:03d}", team="core", status="active") for index in range(100)]
    views += [_view(f"zeta-{index:03d}", team="zeta", status="active") for index in range(100)]
    views.append(_view("orphan", team="omega", status="active"))
    painter = render_project_board(views, cursor=None, now=NOW)
    assert [label for _start, _end, label in painter.sections] == ["core", "zeta"]
    core_end = painter.sections[0][1]  # zeta's header row
    zeta_end = painter.sections[1][1]
    last_core = section_ruler(views, "board", top_row=core_end - 1, cursor=None, width=80)
    first_zeta = section_ruler(views, "board", top_row=core_end, cursor=None, width=80)
    last_zeta = section_ruler(views, "board", top_row=zeta_end - 1, cursor=None, width=80)
    assert last_core is not None and last_core.plain.startswith("── core · 100 ")
    assert first_zeta is not None and first_zeta.plain.startswith("── zeta · 100 ")
    assert last_zeta is not None and last_zeta.plain.startswith("── zeta · 100 ")
    # Single-section variant: the section's own last rows must still name it
    # (pre-fix the ruler returned None — the plain rule — there).
    solo = [
        _view(f"solo-{index:03d}", team="core", status="active")
        for index in range(PROJECTS_MAX + 1)
    ]
    solo_painter = render_project_board(solo, cursor=None, now=NOW)
    solo_last = section_ruler(
        solo, "board", top_row=solo_painter.sections[0][1] - 1, cursor=None, width=80
    )
    assert solo_last is not None and solo_last.plain.startswith("── core · 200 ")


def test_timeline_chart_hits_target_their_own_block() -> None:
    """U3: a mixed team's chart header targets the chart, not the tail line."""
    views = [
        _view("aaa-notes", team="core"),  # undated, first in store order
        _view("zzz-ship", team="core", target_date="2026-10-04"),
    ]
    lines = render_project_timeline(
        views, tier="month", cursor=None, now=NOW
    ).text.plain.splitlines()
    # rows: 0 axis, 1 header, 2 dated row, 3 blank, 4 tail line
    assert lines[1].startswith("── core · 2 ")
    assert "zzz-ship" in lines[2]
    assert lines[4].startswith("no dates (core, 1): ") and "aaa-notes" in lines[4]
    assert section_header_at(views, "timeline", 0, 1) == ("core", 1)  # the dated row
    assert section_header_at(views, "timeline", 0, 4) == ("core", 0)  # the tail line


def test_section_ruler_names_the_viewport_section_and_sheds_in_order() -> None:
    views = _grouped_store()
    full = section_ruler(views, "list", top_row=0, cursor=0, width=40)
    assert full is not None
    assert full.plain == "── core · 4 " + "─" * (40 - cell_len("── core · 4 "))
    # A cursor in a DIFFERENT section appends the dim `· sel {team}` clause.
    sel = section_ruler(views, "list", top_row=5, cursor=0, width=40)
    assert sel is not None
    assert sel.plain == "── personal · 1 · sel core " + "─" * (
        40 - cell_len("── personal · 1 · sel core ")
    )
    # Same section: no clause.
    same = section_ruler(views, "list", top_row=5, cursor=4, width=40)
    assert same is not None and "sel" not in same.plain
    # The shed order: the sel clause first, then the count, then the label's
    # own fit (truncated with an ellipsis, never vanished).
    no_sel = section_ruler(views, "list", top_row=5, cursor=0, width=26)
    assert no_sel is not None and no_sel.plain == "── personal · 1 " + "─" * (
        26 - cell_len("── personal · 1 ")
    )
    # The count still fits at 20 — it only sheds when it cannot.
    counted = section_ruler(views, "list", top_row=5, cursor=0, width=20)
    assert counted is not None and counted.plain == "── personal · 1 " + "─" * (
        20 - cell_len("── personal · 1 ")
    )
    no_count = section_ruler(views, "list", top_row=5, cursor=0, width=14)
    assert no_count is not None and no_count.plain == "── personal " + "─" * (
        14 - cell_len("── personal ")
    )
    tiny = section_ruler(views, "list", top_row=5, cursor=0, width=10)
    assert tiny is not None and tiny.plain == "── perso… "
    # Off any band (and ungrouped) the caller paints the shipped plain rule.
    assert section_ruler(views, "list", top_row=99, cursor=0, width=40) is None
    assert section_ruler([_view("alpha")], "list", top_row=0, cursor=0, width=40) is None
