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
from local_operator.projects import truncate_row

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


def _marker_cells(*, selected: bool, associated: bool) -> tuple[tuple[str, str], ...]:
    """The two-cell marker column: ``▸`` the selection, ``◆`` the session's set.

    BOTH stay visible when a row is both: the reader opened this page to find
    their own projects, and the answer must not vanish because the cursor
    happens to sit on one of them (design round 1, D3 — the title count and
    the painted diamonds disagreed). The column is always exactly two cells
    wide, so no canvas moves.
    """
    if selected and associated:
        return (("▸", "cursor"), ("◆", "session"))
    if selected:
        return (("▸", "cursor"), (" ", "dim"))
    if associated:
        return (("◆", "session"), (" ", "dim"))
    return ((" ", "dim"), (" ", "dim"))


def _list_row(
    view: dict[str, Any],
    *,
    selected: bool,
    associated: bool,
    now: float | None,
    style_for: StyleFor,
) -> Text:
    project = _row(view)
    row = Text(no_wrap=True)
    for glyph, key in _marker_cells(selected=selected, associated=associated):
        row.append(glyph, style=style_for(key) if key != "dim" else Style())
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
    associated: frozenset[str] | None = None,
    now: float | None = None,
    style_for: StyleFor | None = None,
) -> RenderResult:
    """One line per project; the selected row carries the cursor marker.

    ``cursor`` is the page's ONE selection (an index into ``views``), and
    ``associated`` names the project ids the calling session is linked to,
    whose rows carry the `◆` marker (S3b) — the same selection and the same
    marker set the board and the timeline read.
    """
    resolver = _styles(style_for)
    mine = associated or frozenset()
    lines: list[Text] = []
    rendered = views[:PROJECTS_MAX]
    for index, view in enumerate(rendered):
        lines.append(
            _list_row(
                view,
                selected=index == cursor,
                associated=str(_row(view).get("id") or "") in mine,
                now=now,
                style_for=resolver,
            )
        )
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


def _card_lines(
    view: dict[str, Any],
    *,
    selected: bool,
    associated: bool,
    now: float | None,
    style_for: StyleFor,
) -> list[Text]:
    """One board card: name / facts / freshness — three lines.

    The name line carries the same two-cell marker column the list rows do
    (``▸`` the selection, ``◆`` the calling session's own set), so which card
    ``↵`` would open is legible in every view without shifting the card grid.
    """
    project = _row(view)
    name = Text(no_wrap=True, style=style_for("name"))
    for glyph, key in _marker_cells(selected=selected, associated=associated):
        name.append(glyph, style=style_for(key) if key != "dim" else style_for("dim"))
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


def _painted_cards(
    rows: list[dict[str, Any]], selected_id: str | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """``(visible, hidden, extra)`` for one board column.

    ``extra`` is the SELECTED card when it would fall past ``BOARD_CARDS_MAX``:
    the selection is what `↵` acts on, and a column that clamps the reader onto
    a card it never paints is a silent clamp (UX round 1, U3). The overflow
    note counts only what is still hidden, so its number stays honest.
    """
    visible = list(rows[:BOARD_CARDS_MAX])
    hidden = list(rows[BOARD_CARDS_MAX:])
    extra = None
    if selected_id is not None:
        for index, view in enumerate(hidden):
            if str(_row(view).get("id") or "") == selected_id:
                extra = hidden.pop(index)
                break
    return visible, hidden, extra


def render_project_board(
    views: list[dict[str, Any]],
    *,
    cursor: int | None = None,
    associated: frozenset[str] | None = None,
    now: float | None = None,
    style_for: StyleFor | None = None,
) -> RenderResult:
    """Status columns painted side by side; cards of three lines, fixed width.

    ``cursor`` selects the ONE card the page's selection points at (an index
    into ``views``; the cards are grouped by status, so the index is resolved
    to an id) and ``associated`` marks the calling session's own set — both
    painted in the card's marker column (S3b), never shifting the grid.
    """
    resolver = _styles(style_for)
    mine = associated or frozenset()
    rendered = views[:PROJECTS_MAX]
    selected_id: str | None = None
    if cursor is not None and 0 <= cursor < len(rendered):
        selected_id = str(_row(rendered[cursor]).get("id") or "")
    columns = _columns_of(rendered)
    blocks: list[list[Text]] = []
    for status, rows in columns:
        block: list[Text] = []
        header = Text(no_wrap=True)
        header.append(status, style=resolver("status"))
        header.append(f" · {len(rows)}", style=resolver("dim"))
        block.append(header)
        visible, hidden, extra = _painted_cards(rows, selected_id)
        for view in visible:
            row_id = str(_row(view).get("id") or "")
            block.append(Text(""))
            block.extend(
                _card_lines(
                    view,
                    selected=selected_id is not None and row_id == selected_id,
                    associated=row_id in mine,
                    now=now,
                    style_for=resolver,
                )
            )
        if hidden or extra is not None:
            # The note names the hidden remainder — and, when the session's own
            # set is in it, how many of those are theirs, so a marker the
            # column cannot paint is at least COUNTED where the reader looks
            # for it (UX round 1, U3).
            note = f"… +{len(hidden)} more"
            yours = sum(1 for view in hidden if str(_row(view).get("id") or "") in mine)
            if yours:
                note += f" ({yours} yours)"
            block.append(Text(""))
            block.append(Text(note, style=resolver("dim")))
        if extra is not None:
            # The selected card is painted even past the cap: `↵` acts on it,
            # and an action must have a visible object (UX round 1, U2/U3).
            extra_id = str(_row(extra).get("id") or "")
            block.append(Text(""))
            block.extend(
                _card_lines(
                    extra,
                    selected=True,
                    associated=extra_id in mine,
                    now=now,
                    style_for=resolver,
                )
            )
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
    selected: bool = False,
    associated: bool = False,
    style_for: StyleFor,
) -> Text:
    project = _row(view)
    row = Text(no_wrap=True)
    # The marker column rides INSIDE the fixed name field — two cells off the
    # name, never a wider row — so the timeline's grid arithmetic (and every
    # frame measured against it) does not move (S3b).
    for glyph, key in _marker_cells(selected=selected, associated=associated):
        row.append(glyph, style=style_for(key) if key != "dim" else style_for("dim"))
    name = Text(str(project.get("name") or "(unnamed)"), style=style_for("name"))
    name.truncate(TIMELINE_NAME_WIDTH - 2, overflow="ellipsis", pad=True)
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


def _split_dated(
    views: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(dated, undated)`` — the renderer's own split, shared with the reveal.

    A project is DATED when it carries a date of its own or a milestone target:
    otherwise it lands in the trailing ``no dates`` tail rather than on a
    fabricated schedule. The timeline canvas and :func:`timeline_position`
    read this ONE split, so a reveal cannot land on a row the chart did not
    paint.
    """
    dated: list[dict[str, Any]] = []
    undated: list[dict[str, Any]] = []
    for view in views:
        project = _row(view)
        if any(project.get(key) for key in ("start_date", "target_date", "completed_at")) or any(
            isinstance(m, dict) and m.get("target_date") for m in project.get("milestones") or []
        ):
            dated.append(view)
        else:
            undated.append(view)
    return dated, undated


def render_project_timeline(
    views: list[dict[str, Any]],
    *,
    tier: str = "month",
    cursor: int | None = None,
    associated: frozenset[str] | None = None,
    today: date | None = None,
    now: float | None = None,
    style_for: StyleFor | None = None,
) -> RenderResult:
    """The time-scaled cell chart: bars, milestone glyphs, a today marker.

    Projects with no dates are NOT silently dropped and not drawn on a
    fabricated schedule: they land in the trailing ``no dates`` section, which
    is the honest degradation the design asks for. ``cursor`` marks the page's
    selection in the name column and ``associated`` the calling session's own
    rows (S3b), both inside the fixed name field.
    """
    resolver = _styles(style_for)
    moment = today or date.today()
    rendered = views[:PROJECTS_MAX]
    if tier not in TIMELINE_TIERS:
        tier = "month"
    mine = associated or frozenset()
    selected_id: str | None = None
    if cursor is not None and 0 <= cursor < len(rendered):
        selected_id = str(_row(rendered[cursor]).get("id") or "")
    span = timeline_span(rendered, today=moment)
    dated, undated = _split_dated(rendered)

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
                    selected=selected_id is not None
                    and str(_row(view).get("id") or "") == selected_id,
                    associated=str(_row(view).get("id") or "") in mine,
                    style_for=resolver,
                )
            )
    if undated:
        if lines:
            lines.append(Text(""))
        # The tail is ONE line, and the markers live ON it: an undated project
        # is still selectable and still part of the session's set, so both
        # glyphs are painted beside its name — otherwise `↵` would aim at a
        # row with no visible selection at all (agent review round 1, F1).
        tail = Text()
        tail.append(f"no dates ({len(undated)}): ", style=resolver("dim"))
        for index, view in enumerate(undated):
            if index:
                tail.append(", ", style=resolver("dim"))
            row_id = str(_row(view).get("id") or "")
            marked = (selected_id is not None and row_id == selected_id) or row_id in mine
            if marked:
                # The tail is free-form (no fixed name column to align), so a
                # marked name carries its cells and an unmarked one stays bare.
                for glyph, key in _marker_cells(
                    selected=selected_id is not None and row_id == selected_id,
                    associated=row_id in mine,
                ):
                    tail.append(glyph, style=resolver(key) if key != "dim" else resolver("dim"))
            tail.append(
                str(_row(view).get("name") or "(unnamed)"),
                style=resolver("name") if marked else resolver("dim"),
            )
        lines.append(tail)
    if not lines:
        from local_operator.slash_commands import project_empty_text

        lines.append(Text(project_empty_text(), style=resolver("dim")))
    width = max(cell_len(line.plain) for line in lines)
    return RenderResult(text=Text("\n").join(lines), width=width, height=len(lines))


def board_position(views: list[dict[str, Any]], cursor: int) -> tuple[int, int] | None:
    """The board-canvas ``(x, y)`` of ``views[cursor]``'s card, or ``None``.

    ``None`` means the card is NOT painted — past ``PROJECTS_MAX`` or its
    column's ``BOARD_CARDS_MAX`` — which is what keeps the page's selection off
    a cell no reader can see (design round 1, D2's rule, applied to the board).
    The point is the card's NAME line, where the marker column lives, so a
    reveal that lands there shows the selection itself and not just its card.
    """
    rendered = views[:PROJECTS_MAX]
    if not (0 <= cursor < len(rendered)):
        return None
    target = str(_row(rendered[cursor]).get("id") or "")
    x = 0
    for _status, rows in _columns_of(rendered):
        visible, _hidden, extra = _painted_cards(rows, target)
        for index, view in enumerate(visible):
            if str(_row(view).get("id") or "") == target:
                # Each card is a blank line and three content lines under the
                # column header: its name line sits at 2 + index * 4.
                return (x * BOARD_COLUMN_WIDTH, 2 + index * 4)
        if extra is not None and str(_row(extra).get("id") or "") == target:
            # Past the cap the selected card is painted after the overflow
            # note: 4 lines per visible card, then the note's blank + line,
            # then this card's blank — its NAME line lands at 4 + 4 * V.
            return (x * BOARD_COLUMN_WIDTH, 4 + 4 * len(visible))
        x += 1
    return None


def timeline_position(views: list[dict[str, Any]], cursor: int) -> tuple[int, int] | None:
    """The timeline-canvas ``(x, y)`` of ``views[cursor]``'s row, or ``None``.

    Dated projects are one row each under the axis; the undated tail is ONE
    line naming them all, so an undated selection answers with THAT line's
    ``y`` — its name is on it. ``None`` means the index is past
    ``PROJECTS_MAX`` or its project has no painted line at all.
    """
    rendered = views[:PROJECTS_MAX]
    if not (0 <= cursor < len(rendered)):
        return None
    target = str(_row(rendered[cursor]).get("id") or "")
    dated, undated = _split_dated(rendered)
    span = timeline_span(rendered)
    y = 0
    if span is not None and dated:
        y = 1  # the axis line; the rows follow it one per dated project
        for view in dated:
            if str(_row(view).get("id") or "") == target:
                return (0, y)
            y += 1
    if undated:
        if y:
            y += 1  # the blank line between the chart and the tail
        if any(str(_row(view).get("id") or "") == target for view in undated):
            return (0, y)
    return None


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
    name_text = str(project.get("name") or "(unnamed)")
    status = str(project.get("status") or "active")
    identity.append(name_text, style=resolver("name"))
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

    # An EMPTY clause is not a clause: a milestone-less project must not paint
    # a doubled seam (`·  ·`) where its milestone list would be, and the five
    # cells a phantom slot costs flip the shed ladder on an 82-86-cell box,
    # hiding a rollup that fits (UX round 2, U4 — the pre-remediation builder
    # guarded this and the guard was lost in the ladder rewrite).
    order = tuple(
        key
        for key in ("identity", "progress", "milestones", "sessions")
        if clauses[key].plain.strip()
    )
    # Preference ladder, longest first, by PROGRESSIVE shedding in the order
    # the docstring states: the progress body goes first (its age is also on
    # the canvas row), then the session rollup, then the milestones — so the
    # 192-235-cell band keeps the rollup instead of the body (agent review
    # round 2: the rollup used to shed before the body, against this ladder's
    # own statement). Every rung keeps the identity.
    rungs: list[tuple[str, ...]] = [order]
    remaining = order
    for shed in ("progress", "sessions", "milestones"):
        candidate = tuple(key for key in remaining if key != shed)
        if candidate != remaining:
            rungs.append(candidate)
        remaining = candidate

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
        # A blind truncate can cut INSIDE the status chip (`… [p…` — UX round
        # 2, U5). The chip is a whole token, so when it fits alone the NAME
        # gives way first and the chip survives intact; only when even the
        # chip cannot fit does the line ellipsize mid-token.
        chip_text = f"[{status}]"
        if cell_len(chip_text) + 2 <= width:
            name_cap = max(width - cell_len(chip_text) - 1, 1)
            fitted = Text(truncate_row(name_text, cap=name_cap) + " " + chip_text)
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
