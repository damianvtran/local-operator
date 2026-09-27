"""Paint project data onto character canvases — PURE, no Textual.

Companion of :mod:`local_operator.tui.org_render`, and the same split: the
widget (:mod:`local_operator.tui.widgets.projects_view`) is a scroll container
around what the three renderers here return, so the grid arithmetic, the bar
math and the truncation rules are unit-tested against plain strings rather than
guessed from a screenshot.

THREE VIEWS, ONE SHAPE
======================

Every renderer takes the SAME input — the list of composed project views
(``build_project_view``'s payload, one dict per project, in registry order) —
and answers a :class:`RenderResult` whose ``width``/``height`` are the exact
canvas the widget must pin its ``Static`` to (the org-chart mechanics: the
container's virtual size equals the canvas, so scrollbars appear exactly when
the content overflows).

- ``render_project_list`` — one line per project, plus a cursor marker the
  widget's list view moves (clamped).
- ``render_project_board`` — status columns painted side by side, cards of
  three lines, the design's fixed ~32-cell columns.
- ``render_project_timeline`` — the "gantt in a terminal" (§V2.B.4): one row
  per project, one column per time unit (week / month / quarter), bars of
  ``█``, milestone diamonds, a today marker. Time-scaled cells, no pixel
  graphics, no sub-cell precision — exactly what was promised.

HONEST DEGRADATION IS THE DEFAULT
=================================

A missing field reads as a sentence, never as a zero: no progress renders "no
progress", an unknown subagent/todo count (``null`` in the payload) renders as
absent rather than ``0``, and a project with no dates lands in the timeline's
trailing no-dates section instead of on a row pretending to have a schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.projects import PROJECT_ROW_CAP
from local_operator.projects import age_text as derived_age_text
from local_operator.projects import milestone_state as derived_milestone_state

#: Cap on projects a single canvas renders. Past it the canvas names the
#: overflow in one truncation row; the cap exists so one runaway store cannot
#: make a keystroke path paint unbounded text (the settings list's discipline).
PROJECTS_MAX = 200

#: One list row stays scannable, the same budget the project tool's own listing
#: uses — THE value lives in the store module (``projects.PROJECT_ROW_CAP``) and
#: every surface imports it, so no two surfaces can truncate the same row at two
#: widths (agent review round 1, F4).
LIST_ROW_CAP = PROJECT_ROW_CAP

#: Board columns in their fixed order; ``archived`` joins only when non-empty
#: (the desktop board's rule, kept identical so the two boards agree).
BOARD_COLUMNS: tuple[str, ...] = ("active", "paused", "done")
BOARD_EXTRA_COLUMN = "archived"

#: Card column width in cells, header included. "Fixed ~32 cells" per the
#: design: columns must be comparable side by side, so the width cannot be
#: content-derived per column.
BOARD_COLUMN_WIDTH = 32

#: Cards per column before the ``… +N more`` row. Twelve cards of 4 rows is 48
#: rows — already beyond any sane terminal — so the cap bounds the CANVAS, not
#: the reading experience.
BOARD_CARDS_MAX = 12

#: Timeline tiers: cells per time unit, finest first. Zoom = level of detail
#: (the org-chart rule applied to time), so ``+`` moves toward ``week`` and
#: ``-`` toward ``quarter``.
TIMELINE_TIERS: tuple[str, ...] = ("week", "month", "quarter")

#: The timeline's auto-fit budget, total canvas width in cells (name column
#: included). The widget picks the finest tier that fits; if even ``quarter``
#: does not, the canvas scrolls horizontally — bounded, not truncated.
TIMELINE_MAX_COLUMNS = 366

#: Width of the pinned name column, cells. Milestone glyphs and bars need the
#: rest of the row, so names are truncated here rather than allowed to push the
#: axis off screen.
TIMELINE_NAME_WIDTH = 24

StyleFor = Callable[[str], Style]


@dataclass(frozen=True)
class RenderResult:
    """The painted canvas plus the geometry behind it (org_render's shape)."""

    text: Text
    width: int
    height: int


def _styles(style_for: StyleFor | None) -> StyleFor:
    """A resolver for the keys this module paints, defaulting to plain ink."""
    if style_for is not None:
        return style_for
    return lambda key: Style()


# ---------------------------------------------------------------------------
# Derived display fields (shared by the renderers and the widget's footer)
# ---------------------------------------------------------------------------


def _row(view: dict[str, Any]) -> dict[str, Any]:
    project = view.get("project")
    return project if isinstance(project, dict) else {}


def age_text(updated_at: float | None, *, now: float | None = None) -> str | None:
    """``3s``/``5m``/``2h``/``1d`` age of a timestamp, or ``None``.

    Thin alias for :func:`local_operator.projects.age_text` — THE age
    arithmetic, one copy (the 90 s / 90 min / 48 h cut points live there, and
    ``reported_age`` layers the progress guard on top for model callers), so
    the list, the board, the detail footer, the terminal listing and the
    project tool cannot disagree about how old a progress line is.
    """
    return derived_age_text(updated_at, now=now)


def progress_age_text(view: dict[str, Any], *, now: float | None = None) -> str | None:
    """The progress age for one composed view, or ``None`` when none recorded."""
    project = _row(view)
    if not project.get("progress") or project.get("progress_updated_at") is None:
        return None
    return age_text(project.get("progress_updated_at"), now=now)


def live_count(view: dict[str, Any]) -> int:
    """Linked sessions whose runtime record classifies ``live``."""
    rows = view.get("sessions")
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        runtime = row.get("runtime") if isinstance(row, dict) else None
        if isinstance(runtime, dict) and runtime.get("state") == "live":
            total += 1
    return total


def session_count(view: dict[str, Any]) -> int:
    rows = view.get("sessions")
    return len(rows) if isinstance(rows, list) else 0


def estimate_text(project: dict[str, Any]) -> str | None:
    """``13pt`` / ``4d`` — or ``None`` when no estimate is set."""
    estimate = project.get("estimate")
    if estimate is None:
        return None
    unit = project.get("estimate_unit") or "points"
    suffix = "pt" if unit == "points" else "d"
    return f"est {estimate:g}{suffix}"


def milestone_counts(project: dict[str, Any]) -> str:
    milestones = project.get("milestones") or []
    done = sum(1 for m in milestones if isinstance(m, dict) and m.get("completed_at"))
    return f"M {done}/{len(milestones)}"


def _milestone_style(style_for: StyleFor, state: str) -> Style:
    """Colour for a derived milestone state — ONE treatment, every surface.

    ``overdue`` is the danger tone everywhere: the footer used to paint the
    same late milestone warning-amber while the timeline painted it red, so
    "red = late" did not transfer between two views of one page (design
    round 1, D3) — both now read this mapping, and so does the board.
    """
    if state == "completed":
        return style_for("milestone_done")
    if state == "overdue":
        return style_for("milestone_late")
    # `milestone_due` (muted), the same token the timeline's `◇` uses — the
    # footer and the timeline agree about every derived milestone state.
    return style_for("milestone_due")


def _status_style(style_for: StyleFor, status: str) -> Style:
    """Colour for a project's status chip — ONE mapping (design round 1, D4).

    ``active`` keeps the accent every other "running" chip on the page uses,
    ``paused`` takes the warning tone, and ``done``/``archived`` recede — so
    the colour channel carries the scan the page exists for instead of three
    identical green badges.
    """
    return style_for(f"status_{status or 'active'}")


def _sessions_text(view: dict[str, Any]) -> str:
    total = session_count(view)
    live = live_count(view)
    return f"{total} session{'' if total == 1 else 's'} ({live} live)"


def _timeline_cell_index(day: date, start: date, tier: str) -> int:
    """Cell index of ``day`` from the axis origin, aligned to unit starts."""
    if tier == "week":
        origin = start - timedelta(days=start.weekday())
        return ((day - origin).days) // 7
    if tier == "month":
        origin = start.replace(day=1)
        return (day.year - origin.year) * 12 + (day.month - origin.month)
    origin_month = ((start.month - 1) // 3) * 3 + 1
    origin = start.replace(month=origin_month, day=1)
    return ((day.year - origin.year) * 12 + (day.month - origin.month)) // 3


def _timeline_unit_start(day: date, tier: str) -> date:
    if tier == "week":
        return day - timedelta(days=day.weekday())
    if tier == "month":
        return day.replace(day=1)
    return day.replace(month=((day.month - 1) // 3) * 3 + 1, day=1)


def _timeline_unit_label(day: date, tier: str) -> str:
    if tier == "quarter":
        return f"Q{(day.month - 1) // 3 + 1}"
    return day.strftime("%b")


def timeline_axis_cells(span_start: date, span_end: date, tier: str) -> int:
    """Cells from the unit containing ``span_start`` through ``span_end``'s unit."""
    return _timeline_cell_index(span_end, span_start, tier) + 1


def timeline_span(
    views: list[dict[str, Any]], *, today: date | None = None
) -> tuple[date, date] | None:
    """The axis range over the dated projects, or ``None`` when none dated.

    ``today`` is always included so the today marker exists on every timeline —
    a schedule whose entire point is "where you are in time" must show now.
    """
    days: list[date] = [today or date.today()]
    for view in views:
        project = _row(view)
        for key in ("start_date", "target_date", "completed_at"):
            value = project.get(key)
            if not value:
                continue
            try:
                days.append(date.fromisoformat(str(value)))
            except ValueError:
                continue
        for milestone in project.get("milestones") or []:
            if isinstance(milestone, dict) and milestone.get("target_date"):
                try:
                    days.append(date.fromisoformat(str(milestone["target_date"])))
                except ValueError:
                    continue
    if len(days) == 1:
        # Nothing was dated: a timeline of today alone says nothing, so the
        # caller renders the no-dates section instead (see render_project_timeline).
        return None
    return min(days), max(days)


def auto_timeline_tier(
    views: list[dict[str, Any]],
    *,
    today: date | None = None,
    max_columns: int = TIMELINE_MAX_COLUMNS,
) -> str:
    """The finest tier whose canvas fits the auto-fit budget, else ``quarter``.

    "Auto tier zoom-out to fit": start from the finest (``week``) and coarsen
    until the whole canvas — name column included — fits ``max_columns``. If
    even ``quarter`` overflows, it returns ``quarter``: the canvas then scrolls
    horizontally rather than dropping to a tier below the vocabulary.
    """
    span = timeline_span(views, today=today)
    if span is None:
        return TIMELINE_TIERS[1]
    start, end = span
    for tier in TIMELINE_TIERS:
        width = TIMELINE_NAME_WIDTH + 1 + timeline_axis_cells(start, end, tier)
        if width <= max_columns:
            return tier
    return TIMELINE_TIERS[-1]


# ---------------------------------------------------------------------------
# The list view
# ---------------------------------------------------------------------------


def _list_row(
    view: dict[str, Any], *, selected: bool, now: float | None, style_for: StyleFor
) -> Text:
    project = _row(view)
    row = Text(no_wrap=True)
    row.append("▸ " if selected else "  ", style=style_for("cursor") if selected else Style())
    row.append(str(project.get("name") or "(unnamed)"), style=style_for("name"))
    status = str(project.get("status") or "active")
    row.append(f" [{status}]", style=_status_style(style_for, status))
    extras: list[tuple[str, str]] = []
    estimate = estimate_text(project)
    if estimate:
        extras.append((estimate, "dim"))
    if project.get("target_date"):
        extras.append((f"→{project['target_date']}", "dim"))
    if project.get("milestones"):
        extras.append((milestone_counts(project), "dim"))
    extras.append((_sessions_text(view), "dim" if live_count(view) == 0 else "live"))
    age = progress_age_text(view, now=now)
    if age is None:
        extras.append(("no progress", "dim"))
    else:
        stale = " (stale)" if view.get("progress_stale") else ""
        extras.append((f"progress {age} ago{stale}", "stale" if stale else "dim"))
    for text, key in extras:
        row.append(" · ", style=style_for("dim"))
        row.append(text, style=style_for(key))
    summary = str(project.get("description") or "").strip()
    if summary:
        row.append(" · ", style=style_for("dim"))
        tail = f'"{summary}"'
        if cell_len(row.plain) + cell_len(tail) > LIST_ROW_CAP:
            # The shared row budget (``projects.PROJECT_ROW_CAP``) in CELLS: the
            # DERIVED rules — age, cap, milestone status — come from the store
            # module so the operator's and the model's surfaces cannot disagree
            # about them; the rows themselves legitimately differ by surface
            # (this one adds the cursor marker and the live count, the tool
            # words its session field differently).
            tail = tail[: max(1, LIST_ROW_CAP - cell_len(row.plain) - 1)].rstrip() + "…"
        row.append(tail, style=style_for("dim"))
    return row


def render_project_list(
    views: list[dict[str, Any]],
    *,
    cursor: int | None = None,
    now: float | None = None,
    style_for: StyleFor | None = None,
) -> RenderResult:
    """One line per project; the selected row carries the cursor marker."""
    resolver = _styles(style_for)
    lines: list[Text] = []
    rendered = views[:PROJECTS_MAX]
    for index, view in enumerate(rendered):
        lines.append(_list_row(view, selected=index == cursor, now=now, style_for=resolver))
    if len(views) > PROJECTS_MAX:
        lines.append(
            Text(
                f"  … +{len(views) - PROJECTS_MAX} more not shown",
                style=resolver("dim"),
            )
        )
    if not lines:
        from local_operator.slash_commands import project_empty_text

        lines.append(Text(project_empty_text(), style=resolver("dim")))
    width = max(cell_len(line.plain) for line in lines)
    text = Text("\n").join(lines)
    return RenderResult(text=text, width=width, height=len(lines))


# ---------------------------------------------------------------------------
# The board view
# ---------------------------------------------------------------------------


def _card_lines(view: dict[str, Any], *, now: float | None, style_for: StyleFor) -> list[Text]:
    """One board card: name / facts / freshness — three lines."""
    project = _row(view)
    name = Text(no_wrap=True, style=style_for("name"))
    name.append(str(project.get("name") or "(unnamed)"))
    facts = Text(no_wrap=True)
    bits: list[str] = []
    estimate = estimate_text(project)
    if estimate:
        bits.append(estimate)
    if project.get("milestones"):
        bits.append(milestone_counts(project))
    if project.get("target_date"):
        bits.append(f"→{project['target_date']}")
    # An absent chip is OMITTED, not painted as "none": a column of "no
    # estimate" cells is noise, and the card's third line already carries the
    # honest absence ("no progress"). A card with nothing at all says so once.
    facts.append(" · ".join(bits) if bits else "no estimate or dates", style=style_for("dim"))
    fresh = Text(no_wrap=True)
    age = progress_age_text(view, now=now)
    if age is None:
        fresh.append("no progress", style=style_for("dim"))
    else:
        stale = " (stale)" if view.get("progress_stale") else ""
        stale_style = style_for("stale") if stale else style_for("dim")
        fresh.append(f"reported {age} ago{stale}", style=stale_style)
    live = live_count(view)
    if live:
        fresh.append(" · ", style=style_for("dim"))
        fresh.append(f"{live} live", style=style_for("live"))
    return [name, facts, fresh]


def _columns_of(views: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    by_status: dict[str, list[dict[str, Any]]] = {name: [] for name in BOARD_COLUMNS}
    for view in views:
        status = str(_row(view).get("status") or "active")
        by_status.setdefault(status, []).append(view)
    columns = [(name, by_status.get(name, [])) for name in BOARD_COLUMNS]
    archived = by_status.get(BOARD_EXTRA_COLUMN) or []
    if archived:
        columns.append((BOARD_EXTRA_COLUMN, archived))
    return columns


def render_project_board(
    views: list[dict[str, Any]],
    *,
    now: float | None = None,
    style_for: StyleFor | None = None,
) -> RenderResult:
    """Status columns painted side by side; cards of three lines, fixed width."""
    resolver = _styles(style_for)
    rendered = views[:PROJECTS_MAX]
    columns = _columns_of(rendered)
    blocks: list[list[Text]] = []
    for status, rows in columns:
        block: list[Text] = []
        header = Text(no_wrap=True)
        header.append(status, style=resolver("status"))
        header.append(f" · {len(rows)}", style=resolver("dim"))
        block.append(header)
        for view in rows[:BOARD_CARDS_MAX]:
            block.append(Text(""))
            block.extend(_card_lines(view, now=now, style_for=resolver))
        if len(rows) > BOARD_CARDS_MAX:
            block.append(Text(""))
            block.append(Text(f"… +{len(rows) - BOARD_CARDS_MAX} more", style=resolver("dim")))
        blocks.append(block)
    if len(views) > PROJECTS_MAX:
        blocks[0].append(Text(""))
        blocks[0].append(
            Text(f"… +{len(views) - PROJECTS_MAX} more not shown", style=resolver("dim"))
        )
    if not rendered:
        from local_operator.slash_commands import project_empty_text

        empty = Text(project_empty_text(), style=resolver("dim"))
        return RenderResult(text=empty, width=cell_len(empty.plain), height=1)
    height = max((len(block) for block in blocks), default=1)
    lines: list[Text] = []
    for index in range(height):
        line = Text(no_wrap=True)
        for column_index, block in enumerate(blocks):
            cell = block[index].copy() if index < len(block) else Text("")
            cell.truncate(BOARD_COLUMN_WIDTH - 1, overflow="ellipsis", pad=True)
            line.append_text(cell)
            if column_index < len(blocks) - 1:
                line.append(" ")
        # Trailing whitespace is not content: a canvas line that ends in 30
        # painted spaces both wastes width and makes every board frame differ
        # from the next by cells nobody can see. ``right_crop`` keeps the
        # remaining cells' spans intact, which is why the trim is not a rebuild.
        trailing = len(line.plain) - len(line.plain.rstrip())
        if trailing:
            line.right_crop(trailing)
        lines.append(line)
    width = max(cell_len(line.plain) for line in lines)
    return RenderResult(text=Text("\n").join(lines), width=width, height=len(lines))


# ---------------------------------------------------------------------------
# The timeline view
# ---------------------------------------------------------------------------


def _timeline_row(
    view: dict[str, Any],
    *,
    start: date,
    axis_cells: int,
    tier: str,
    today: date,
    style_for: StyleFor,
) -> Text:
    project = _row(view)
    row = Text(no_wrap=True)
    name = Text(str(project.get("name") or "(unnamed)"), style=style_for("name"))
    name.truncate(TIMELINE_NAME_WIDTH, overflow="ellipsis", pad=True)
    row.append_text(name)
    row.append(" ", style=style_for("dim"))

    cells: list[tuple[str, str]] = [(" ", "dim")] * axis_cells

    def parse(value: Any) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None

    start_day = parse(project.get("start_date"))
    target_day = parse(project.get("target_date"))
    completed_day = parse(project.get("completed_at"))
    done_status = str(project.get("status")) == "done"
    end_day = completed_day if (done_status and completed_day) else target_day
    if start_day is not None or end_day is not None:
        begin = start_day or end_day
        finish = end_day or start_day
        if begin is not None and finish is not None:
            first = _timeline_cell_index(begin, start, tier)
            last = _timeline_cell_index(finish, start, tier)
            if last < first:
                first, last = last, first
            for cell in range(max(0, first), min(axis_cells - 1, last) + 1):
                cells[cell] = ("█", "bar")
    for milestone in project.get("milestones") or []:
        if not isinstance(milestone, dict):
            continue
        day = parse(milestone.get("target_date"))
        if day is None:
            continue
        cell = _timeline_cell_index(day, start, tier)
        if not 0 <= cell < axis_cells:
            continue
        if milestone.get("completed_at"):
            cells[cell] = ("◆", "milestone_done")
        elif day < today:
            cells[cell] = ("!", "milestone_late")
        else:
            cells[cell] = ("◇", "milestone_due")
    today_cell = _timeline_cell_index(today, start, tier)
    if 0 <= today_cell < axis_cells:
        cells[today_cell] = ("┊", "today")
    for glyph, key in cells:
        row.append(glyph, style=style_for(key))
    return row


def _timeline_axis(
    start: date, end: date, *, axis_cells: int, tier: str, today: date, style_for: StyleFor
) -> Text:
    """The label row: a unit label at each unit start that is a month (or quarter).

    A label is placed only when the WHOLE label fits inside the canvas with a
    blank cell between it and its neighbour — the guard used to allow a label
    to start on the very next cell, so a full-year month axis read
    ``JanAprJulOct`` and the last label was clipped to a single character at
    the edge (agent review round 1 finding 10 / QA Q3). The first label of
    each new year carries a two-digit year cue (design round 1, D6): an
    18-month span otherwise reads ``Jan … Jan`` with nothing to tell them
    apart.
    """
    row = Text(no_wrap=True)
    row.append(" " * TIMELINE_NAME_WIDTH)
    row.append(" ", style=style_for("dim"))
    labels: list[tuple[int, str, int]] = [(0, _timeline_unit_label(start, tier), start.year)]
    cursor = _timeline_unit_start(start, tier)
    while cursor <= end:
        if tier == "week":
            cursor = cursor + timedelta(days=7)
        elif tier == "month":
            cursor = (cursor.replace(day=1) + timedelta(days=32)).replace(day=1)
        else:
            cursor = (cursor.replace(day=1) + timedelta(days=100)).replace(day=1)
        if cursor > end:
            break
        if tier == "quarter":
            cell = _timeline_cell_index(cursor, start, tier)
            labels.append((cell, _timeline_unit_label(cursor, tier), cursor.year))
        elif tier == "week":
            # One label per month the axis crosses: a label per week would be
            # unreadable at one cell per week, and a month is the coarsest fact
            # the axis says for itself. The month turns over inside this week
            # exactly when the previous week sat in a different month.
            if cursor.month != (cursor - timedelta(days=7)).month:
                cell = _timeline_cell_index(cursor.replace(day=1), start, tier)
                labels.append((cell, cursor.strftime("%b"), cursor.year))
        else:
            cell = _timeline_cell_index(cursor, start, tier)
            labels.append((cell, cursor.strftime("%b"), cursor.year))
    cells = [" "] * axis_cells
    placed = -1
    emitted_year: int | None = None
    text = Text(no_wrap=True)
    text.append(" " * TIMELINE_NAME_WIDTH)
    text.append(" ", style=style_for("dim"))
    for cell, label, year in labels:
        if emitted_year is not None and year != emitted_year:
            label = f"{label} '{year % 100:02d}"
        if cell + len(label) > axis_cells:
            continue
        if placed >= 0 and cell <= placed + 1:
            continue
        for offset, char in enumerate(label):
            cells[cell + offset] = char
        placed = cell + len(label) - 1
        emitted_year = year
    today_cell = _timeline_cell_index(today, start, tier)
    if 0 <= today_cell < axis_cells and cells[today_cell] == " ":
        # The marker runs through every DATA row; on the axis it yields to a
        # label rather than eating a month's last character (`Jan` → `Ja┊`),
        # because orientation is the axis row's entire job.
        cells[today_cell] = "┊"
    text.append("".join(cells), style=style_for("dim"))
    return text


def render_project_timeline(
    views: list[dict[str, Any]],
    *,
    tier: str = "month",
    today: date | None = None,
    now: float | None = None,
    style_for: StyleFor | None = None,
) -> RenderResult:
    """The time-scaled cell chart: bars, milestone glyphs, a today marker.

    Projects with no dates are NOT silently dropped and not drawn on a
    fabricated schedule: they land in the trailing ``no dates`` section, which
    is the honest degradation the design asks for.
    """
    resolver = _styles(style_for)
    moment = today or date.today()
    rendered = views[:PROJECTS_MAX]
    if tier not in TIMELINE_TIERS:
        tier = "month"
    span = timeline_span(rendered, today=moment)
    dated: list[dict[str, Any]] = []
    undated: list[dict[str, Any]] = []
    for view in rendered:
        project = _row(view)
        if any(project.get(key) for key in ("start_date", "target_date", "completed_at")) or any(
            isinstance(m, dict) and m.get("target_date") for m in project.get("milestones") or []
        ):
            dated.append(view)
        else:
            undated.append(view)

    lines: list[Text] = []
    if span is not None and dated:
        start, end = span
        axis_cells = timeline_axis_cells(start, end, tier)
        lines.append(
            _timeline_axis(
                start, end, axis_cells=axis_cells, tier=tier, today=moment, style_for=resolver
            )
        )
        for view in dated:
            lines.append(
                _timeline_row(
                    view,
                    start=start,
                    axis_cells=axis_cells,
                    tier=tier,
                    today=moment,
                    style_for=resolver,
                )
            )
    if undated:
        if lines:
            lines.append(Text(""))
        names = ", ".join(str(_row(view).get("name") or "(unnamed)") for view in undated)
        lines.append(
            Text(
                f"no dates ({len(undated)}): {names}",
                style=resolver("dim"),
            )
        )
    if not lines:
        from local_operator.slash_commands import project_empty_text

        lines.append(Text(project_empty_text(), style=resolver("dim")))
    width = max(cell_len(line.plain) for line in lines)
    return RenderResult(text=Text("\n").join(lines), width=width, height=len(lines))


# ---------------------------------------------------------------------------
# The pinned footer content (chrome, not canvas)
# ---------------------------------------------------------------------------


def today_iso(now: float | None = None) -> str:
    """Today as ISO, from the same clock ``age_text`` reads.

    ``now=None`` uses the UTC date — the same basis ``projects._utc_today``
    stamps milestone completion from — so "overdue" and "done today" cannot
    disagree about which day it is.
    """
    if now is None:
        import datetime as _datetime

        return _datetime.datetime.now(tz=_datetime.timezone.utc).date().isoformat()
    import datetime as _datetime

    return _datetime.datetime.fromtimestamp(now, tz=_datetime.timezone.utc).date().isoformat()


def milestone_state(milestone: dict[str, Any], *, today: str | None = None) -> str:
    """``completed | overdue | upcoming`` — the store's derived-status rule.

    Delegates to :func:`local_operator.projects.milestone_state`, which the
    model side (:func:`projects.milestone_status`) also calls: one rule for
    both shapes, so a footer's colour and the tool's reported status cannot
    disagree about one milestone. Tolerant of malformed dates for the reason
    stated there — a canvas frame must not crash on a bad row.
    """
    return derived_milestone_state(
        milestone.get("completed_at"), milestone.get("target_date"), today=today
    )


def detail_footer(
    view: dict[str, Any],
    *,
    now: float | None = None,
    style_for: StyleFor | None = None,
    width: int | None = None,
) -> Text:
    """``progress + milestones + session rollup`` for the highlighted project.

    One bounded line for the pinned footer under the list: the canvas shows
    every project; this says what the highlighted one is DOING — its progress
    with its reporter and age, its milestones with derived status, and each
    linked session's runtime state beside what it is carrying (subagent and
    todo counts when the store knows them). ``null`` counts are OMITTED, never
    rendered as zeroes.

    ``width`` fits the line to the space the page actually has: clauses are
    shed WHOLE, tail-first among equals, in a fixed preference order, and the
    row never clips mid-word — the Rich ``ellipsis`` this Text declares is
    inert under the widget's default fold, so the fitting is done here
    (UX round 1, U1). The order keeps the footer's stated purpose longest:
    the identity always, the milestones and the session rollup before the
    (long, and duplicated by the canvas row) progress body. When even the
    identity does not fit it is ellipsized explicitly, never cut silently.
    """
    resolver = _styles(style_for)
    project = _row(view)

    def chip(status: str) -> Style:
        # D4: one mapping, every surface. `active` keeps the accent every other
        # "running" chip uses; `paused` takes the warning tone; `done` and
        # `archived` recede — the colour channel then carries the scan the page
        # exists for instead of three identical green badges.
        return resolver(f"status_{status or 'active'}")

    clauses: dict[str, Text] = {}
    identity = Text(no_wrap=True)
    status = str(project.get("status") or "active")
    identity.append(str(project.get("name") or "(unnamed)"), style=resolver("name"))
    identity.append(f" [{status}]", style=chip(status))
    clauses["identity"] = identity

    progress = Text(no_wrap=True)
    age = progress_age_text(view, now=now)
    if age is None:
        progress.append("progress: none recorded", style=resolver("dim"))
    else:
        stale = " · stale" if view.get("progress_stale") else ""
        by = str(project.get("progress_reported_by") or "")
        reporter = f" by {by}" if by else ""
        progress.append(f"progress reported {age} ago{reporter}{stale}: ", style=resolver("dim"))
        progress.append(str(project.get("progress") or ""))
    clauses["progress"] = progress

    milestones = Text(no_wrap=True)
    if project.get("milestones"):
        milestones.append(milestone_counts(project), style=resolver("dim"))
        today = today_iso(now)
        for milestone in project["milestones"][:3]:
            if not isinstance(milestone, dict):
                continue
            state = milestone_state(milestone, today=today)
            milestones.append(
                f" · {milestone.get('name')} [{state}]",
                style=_milestone_style(resolver, state),
            )
    clauses["milestones"] = milestones

    sessions = Text(no_wrap=True)
    rows_value = view.get("sessions")
    rows: list[Any] = rows_value if isinstance(rows_value, list) else []
    if not rows:
        sessions.append("no linked sessions", style=resolver("dim"))
    else:
        sessions.append("sessions: ", style=resolver("dim"))
        bits: list[str] = []
        for row in rows[:4]:
            if not isinstance(row, dict):
                continue
            session_row: dict[str, Any] = row
            session_id = session_row.get("session_id")
            if session_row.get("exists") is False:
                # The link is stale — say so rather than "stopped", which
                # would be a wrong statement about a session that is gone
                # (agent review round 1, F5; GUIDE.md's `missing` promise).
                bits.append(f"{session_id} [missing]")
                continue
            runtime_value = session_row.get("runtime")
            runtime: dict[str, Any] = runtime_value if isinstance(runtime_value, dict) else {}
            state = str(runtime.get("state") or "stopped")
            busy = ", busy" if runtime.get("busy") else ""
            bit = f"{session_id} [{state}{busy}]"
            subagents_value = session_row.get("subagents")
            subagents: dict[str, Any] = subagents_value if isinstance(subagents_value, dict) else {}
            if subagents.get("running") is not None and subagents.get("settled") is not None:
                running = int(subagents["running"])
                settled = int(subagents["settled"])
                bit += f" {running} running/{settled} settled"
            todos_value = session_row.get("todos")
            todos: dict[str, Any] = todos_value if isinstance(todos_value, dict) else {}
            # BOTH values must exist: a snapshot-less session carries the key
            # with null counts, and `todos None/None` is user-visible junk
            # (UX round 1, U2 — the receipt path omits it; so does this now).
            if todos.get("open") is not None and todos.get("total") is not None:
                bit += f" · todos {todos['open']}/{todos['total']}"
            bits.append(bit)
        sessions.append(" · ".join(bits))
        if len(rows) > 4:
            sessions.append(f" · +{len(rows) - 4} more", style=resolver("dim"))
    clauses["sessions"] = sessions

    order = ("identity", "progress", "milestones", "sessions")
    # Preference ladder, longest first. Every rung keeps the identity; the
    # FULL rung is preferred whenever it fits, and each rung below sheds the
    # clause whose information survives elsewhere first (the progress age is
    # also on the canvas row; the rollup counts are too), so a narrow terminal
    # loses the least.
    rungs = (
        order,
        order[:3],
        ("identity", "milestones", "sessions"),
        ("identity", "milestones"),
        ("identity",),
    )

    def compose(keys: tuple[str, ...]) -> Text:
        row_text = Text(no_wrap=True)
        for index, key in enumerate(keys):
            if index:
                row_text.append("  ·  ", style=resolver("dim"))
            row_text.append_text(clauses[key])
        if len(keys) < len(order):
            # Say that more exists; a footer that stopped at a clause boundary
            # with no marker reads as if it were the whole story.
            row_text.append(" …", style=resolver("dim"))
        return row_text

    if width is None:
        return compose(order)
    fitted = compose(order)
    for keys in rungs:
        candidate = compose(keys)
        if cell_len(candidate.plain) <= width:
            fitted = candidate
            break
    if cell_len(fitted.plain) > width:
        fitted.truncate(width, overflow="ellipsis")
    return fitted


def aggregate_footer(
    views: list[dict[str, Any]],
    *,
    tier: str | None = None,
    style_for: StyleFor | None = None,
) -> Text:
    """Counts for the board/timeline footer: what the page is summarising."""
    resolver = _styles(style_for)
    live = sum(live_count(view) for view in views)
    counts: dict[str, int] = {}
    for view in views:
        status = str(_row(view).get("status") or "active")
        counts[status] = counts.get(status, 0) + 1
    text = Text(no_wrap=True, overflow="ellipsis")
    order = [*BOARD_COLUMNS, BOARD_EXTRA_COLUMN]
    parts = [f"{counts[status]} {status}" for status in order if counts.get(status)]
    text.append(f"{len(views)} project{'' if len(views) == 1 else 's'}", style=resolver("dim"))
    if parts:
        text.append(" · " + " · ".join(parts), style=resolver("dim"))
    text.append(f" · {live} live session{'' if live == 1 else 's'}", style=resolver("dim"))
    if tier:
        text.append(f" · zoom: {tier}", style=resolver("dim"))
    return text
