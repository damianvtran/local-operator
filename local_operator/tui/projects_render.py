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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence, cast

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.projects import PROJECT_ROW_CAP, PROJECT_STATUSES
from local_operator.projects import age_text as derived_age_text
from local_operator.projects import display_name, file_size_text, is_session_id
from local_operator.projects import milestone_state as derived_milestone_state
from local_operator.projects import refreshed_age_text, refreshed_note, truncate_row

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

#: The lifecycle statuses the three columns do not name, mapped to the column
#: they ride in. Kept as data (not inline in the renderer) because the desktop
#: board reads the same mapping off the payload's status; a card's own chip
#: stays the exact word either way.
BOARD_STATUS_BUCKET: dict[str, str] = {
    "planning": "active",
    "qa": "active",
    "validation": "active",
}

#: Column header LABELS: the bucket column says what it HOLDS (design review
#: round 1, D4) — planning/qa/validation cards ride the leading column, so
#: "active · 4" over one true active was two things called active on one
#: screen. "in flight · 4" is true of every card under it.
BOARD_COLUMN_LABELS: dict[str, str] = {"active": "in flight"}

#: Section label for the bucket of projects that name no team — said out loud
#: (the page's honest-tail convention: `no dates`, `no progress`), and the name
#: the ruler and the section jumps address that bucket by.
NO_TEAM_LABEL = "no team"

#: The sentinel's label when a real team is literally named like it: two
#: sections sharing one label would paint identically, and every reader (the
#: ruler's count, the `· sel` clause, a header click) would be left guessing
#: which `no team` is which. Only the collision pays for the qualifier, and it
#: still says exactly what the bucket is — the team field, unset (R1-2/U1).
NO_TEAM_UNSET_LABEL = "no team (unset)"

#: One glyph per status — the SHAPE channel, chosen because the ink channel
#: cannot separate seven statuses out of the five semantic tokens that fit
#: statuses at all (planning/active/qa/validation/paused/done/archived vs
#: accent/success/warning/muted/dim). The repo already ships a shape-only
#: contract for exactly this reason: the tool-status family separates
#: `✓`/`◐`/`✗`/`⊘` in a colourless frame. `✓` is that family's completion
#: glyph and `◐` its in-progress shape; `○`(not started)/`●`(running)/
#: `◉`(under watch)/`‖`(held)/`▤`(filed) extend the same geometric family.
#: An unknown status (a row from a newer build) takes `?` — visibly unknown,
#: still a chip.
STATUS_GLYPHS: dict[str, str] = {
    "planning": "○",
    "active": "●",
    "qa": "◐",
    "validation": "◉",
    "paused": "‖",
    "done": "✓",
    "archived": "▤",
}


def status_chip_text(status: str) -> str:
    """The ONE chip rendering every surface composes: ``[◐ qa]``.

    Glyph, word and ink together: the glyph separates by shape (colourless
    frames included), the word names the status exactly, and the ink is
    :func:`_status_style`'s one mapping. Used by the list rows, the board
    cards, the detail footer and the chip-preserving truncation — a chip can
    never render two ways on one page.
    """
    return f"[{STATUS_GLYPHS.get(status, '?')} {status}]"


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
    #: The team-grouped section bands this canvas paints — ``(start_row,
    #: end_row, label)`` in paint order with ``end_row`` EXCLUSIVE (S6d
    #: parity). One description of where each section sits, read by the
    #: section ruler, the jumps and the header clicks; empty when the canvas
    #: is ungrouped, which is exactly when it paints as shipped.
    sections: tuple[tuple[int, int, str], ...] = ()


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
    """Linked WORKING sessions whose runtime record classifies ``live``.

    A coordination row ("filed by") carries no ``runtime`` key at all — by
    construction, see ``projects.build_project_view`` — so a filing can never
    satisfy this count and the working set is the only thing measured.
    """
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
    """The WORK set: composed rows whose ``role`` is not ``coordination``."""
    rows = view.get("sessions")
    if not isinstance(rows, list):
        return 0
    return sum(
        1 for row in rows if not (isinstance(row, dict) and row.get("role") == "coordination")
    )


def filed_count(view: dict[str, Any]) -> int:
    """The FILING set: composed rows tagged ``role="coordination"``."""
    rows = view.get("sessions")
    if not isinstance(rows, list):
        return 0
    return sum(1 for row in rows if isinstance(row, dict) and row.get("role") == "coordination")


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
    identical green badges. Where the ink channel runs out the SHAPE channel
    carries the separation instead: chips render through
    :func:`status_chip_text`, whose per-status glyphs keep every pair apart in
    a colourless frame (the measured ink table and its acceptances live in
    ``projects_view._style_resolver``).
    """
    return style_for(f"status_{status or 'active'}")


def _sessions_text(view: dict[str, Any]) -> str:
    """``2 working (1 live)`` — work-only counts; a filing named separately.

    "working" is the same word the tool, the slash listing and the ``@project``
    block use, so no surface can make a filed row read as a worker.
    """
    total = session_count(view)
    live = live_count(view)
    text = f"{total} working ({live} live)"
    filed = filed_count(view)
    if filed:
        text += f" · {filed} filed"
    return text


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


def _team_of(view: dict[str, Any]) -> str | None:
    """The project's team as a trimmed name, or ``None`` when unknown."""
    text = str(_row(view).get("team") or "").strip()
    return text or None


def sections_of(views: list[dict[str, Any]]) -> list[tuple[str, list[int]]] | None:
    """``(label, project indexes)`` sections in paint order, or ``None``.

    Grouping is automatic (S6d parity, design §6): a canvas groups by ``team``
    when at least one PAINTED project carries one, and is byte-identical to
    the shipped flat canvases otherwise. Teams sort case-insensitively A→Z and
    the ``no team`` bucket is LAST (the page's honest-tail convention); within
    a section the store's own order is kept, so grouping never becomes a
    second sort. Indexes address ``views`` — the same list every surface
    already uses.
    """
    rendered = views[:PROJECTS_MAX]
    buckets: dict[str | None, list[int]] = {}
    for index, view in enumerate(rendered):
        buckets.setdefault(_team_of(view), []).append(index)
    teams = sorted((team for team in buckets if team is not None), key=str.casefold)
    if not teams:
        return None
    sentinel = NO_TEAM_LABEL
    if any(team.casefold() == NO_TEAM_LABEL.casefold() for team in teams):
        # Display disambiguation only; every behavioural reader keys on a
        # band's position and its group, never on label equality (R1-2/U1).
        sentinel = NO_TEAM_UNSET_LABEL
    order: list[str | None] = [*teams, None]
    return [
        (team if team is not None else sentinel, buckets[team])
        for team in order
        if buckets.get(team)
    ]


def _section_header_text(label: str, count: int, width: int, style_for: StyleFor) -> Text:
    """One painted section header row: ``── core · 4 ──…`` (design §6).

    Dim dashes, the name in the canvases' name ink, the count dim, filled to
    the canvas (list) or the section block (board/timeline) so the band reads
    as a divider. Headers are NOT selectable — no marker column — and are
    addressed only by the section jumps and a header click.
    """
    header = Text(no_wrap=True)
    header.append("── ", style=style_for("dim"))
    header.append(label, style=style_for("name"))
    header.append(f" · {count} ", style=style_for("dim"))
    pad = width - cell_len(header.plain)
    if pad > 0:
        header.append("─" * pad, style=style_for("dim"))
    return header


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
    row.append(display_name(project) or "(unnamed)", style=style_for("name"))
    if project.get("title"):
        # Title-first, key secondary (muted): the key is how the project is
        # ADDRESSED (slash verbs, ``@project:<name>``), so a titled row keeps
        # it visible beside the label the reader knows.
        row.append(f" ({project.get('name') or ''})", style=style_for("dim"))
    status = str(project.get("status") or "active")
    row.append(" " + status_chip_text(status), style=_status_style(style_for, status))
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
        text = f"progress {age} ago{stale}"
        refreshed = refreshed_age_text(_row(view), now=now)
        if refreshed is not None:
            # The refreshed TOKEN beside the age (list contract): the stale
            # badge keeps telling the content clock's truth, and the token
            # says the record was checked.
            text += f" · refreshed {refreshed} ago"
        extras.append((text, "stale" if stale else "dim"))
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
    marker set the board and the timeline read. With teams present the rows
    group under painted section headers (S6d parity); without them this is
    byte-identical to the shipped flat canvas.
    """
    resolver = _styles(style_for)
    mine = associated or frozenset()
    rendered = views[:PROJECTS_MAX]
    rows = [
        _list_row(
            view,
            selected=index == cursor,
            associated=str(_row(view).get("id") or "") in mine,
            now=now,
            style_for=resolver,
        )
        for index, view in enumerate(rendered)
    ]
    lines: list[Text] = []
    bands: list[tuple[int, int, str]] = []
    groups = sections_of(rendered)
    if groups is None:
        lines.extend(rows)
    else:
        # A header row per team, filled to the canvas's own width (the widest
        # row), then that section's rows in the store's order.
        fill_width = max((cell_len(row.plain) for row in rows), default=0)
        for label, indexes in groups:
            start = len(lines)
            lines.append(_section_header_text(label, len(indexes), fill_width, resolver))
            lines.extend(rows[index] for index in indexes)
            bands.append((start, len(lines), label))
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
    return RenderResult(text=text, width=width, height=len(lines), sections=tuple(bands))


def list_position(views: list[dict[str, Any]], cursor: int) -> int | None:
    """Canvas row of ``views[cursor]``'s list row, or ``None`` (past the cap).

    The grouped list interleaves header rows, so a cursor's canvas row is no
    longer its index; this mirrors the painter's walk exactly (the reveal and
    the hit tests read it, and tests cross-check it against the painted
    lines).
    """
    rendered = views[:PROJECTS_MAX]
    if not (0 <= cursor < len(rendered)):
        return None
    groups = sections_of(rendered)
    if groups is None:
        return cursor
    y = 0
    for _label, indexes in groups:
        y += 1  # the header row
        for index in indexes:
            if index == cursor:
                return y
            y += 1
    return None


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
    label = display_name(project) or "(unnamed)"
    status = str(project.get("status") or "active")
    chip = status_chip_text(status)
    chip_w = cell_len(chip)
    key = str(project.get("name") or "")
    key_part = ""
    # After the two-cell marker column, a card line has BOARD_COLUMN_WIDTH - 3
    # cells for `label` + `key` + the separator space + the chip.
    content_room = BOARD_COLUMN_WIDTH - 3
    if project.get("title") and key:
        candidate = f" ({key})"
        # The key joins the title only when the whole line fits: the chip is
        # the exact status the card exists to state (design review round 1,
        # D1) and is never shed; the key is recoverable in the footer/detail,
        # so it sheds first; a label that still cannot fit truncates with an
        # ellipsis — never a mid-token clip at the cell edge (design review
        # round 2, D5).
        if cell_len(label) + cell_len(candidate) + 1 + chip_w <= content_room:
            key_part = candidate
    room = content_room - chip_w - 1 - cell_len(key_part)
    if cell_len(label) > room:
        label = truncate_row(label, cap=room)
    name.append(label)
    if key_part:
        name.append(key_part, style=style_for("dim"))
    name.append(" " + chip, style=_status_style(style_for, status))
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
        # The refreshed assertion does NOT ride this line: a card is the
        # narrowest surface (the three-column board truncates at ~31 cells) and
        # the sentence clipped mid-word there ("· refr…", measured in the
        # capture frames). §4.3's contract puts the board's copy in the
        # tooltip; on the TUI board the two facts live on the list row and the
        # detail page instead.
        fresh.append(f"reported {age} ago{stale}", style=stale_style)
    live = live_count(view)
    if live:
        fresh.append(" · ", style=style_for("dim"))
        fresh.append(f"{live} live", style=style_for("live"))
    return [name, facts, fresh]


def _columns_of(
    views: list[dict[str, Any]], *, skip_empty: bool = False
) -> list[tuple[str, list[dict[str, Any]]]]:
    """The board's columns: the fixed three, plus ``archived`` when non-empty.

    ``skip_empty`` drops columns carrying no cards — the grouped board's
    per-section rule (a stacked block paints only the columns that carry
    cards; an empty ``paused · 0`` header under a populated ``core · 4`` header
    is double-counted emptiness, design §6). The ungrouped board passes it
    False and keeps the shipped fixed-three shape byte-identically.
    """
    by_status: dict[str, list[dict[str, Any]]] = {name: [] for name in BOARD_COLUMNS}
    for view in views:
        status = str(_row(view).get("status") or "active")
        # Lifecycle statuses beyond the board's three columns are BUCKETED into
        # the column they belong to, not given columns of their own: planning,
        # qa and validation are all "in the machine", and every card carries
        # its own exact ``[◐ qa]``-style chip (glyph + word + ink), so the
        # column ("in flight") tells the coarse story while the chip stays
        # exact. An entirely unknown status (a row from a newer build) rides
        # the leading column rather than vanishing — the board's one job is
        # that every project is visible somewhere.
        bucket = BOARD_STATUS_BUCKET.get(status, status)
        if bucket not in (*BOARD_COLUMNS, BOARD_EXTRA_COLUMN):
            bucket = BOARD_COLUMNS[0]
        by_status.setdefault(bucket, []).append(view)
    columns = [(name, by_status.get(name, [])) for name in BOARD_COLUMNS]
    archived = by_status.get(BOARD_EXTRA_COLUMN) or []
    if archived:
        columns.append((BOARD_EXTRA_COLUMN, archived))
    if skip_empty:
        columns = [column for column in columns if column[1]]
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


def _board_blocks(
    columns: list[tuple[str, list[dict[str, Any]]]],
    *,
    selected_id: str | None,
    mine: frozenset[str],
    now: float | None,
    resolver: StyleFor,
) -> list[list[Text]]:
    """One block of lines per column: header, cards, the overflow note.

    Shared by the flat board and the grouped one (S6d parity) so a column
    paints identically in both; ``selected_id``/``mine`` are the page's one
    selection and marker set.
    """
    blocks: list[list[Text]] = []
    for status, rows in columns:
        block: list[Text] = []
        label = BOARD_COLUMN_LABELS.get(status, status)
        header = Text(no_wrap=True)
        header.append(label, style=resolver("status"))
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
            # for it (UX round 1, U3). When nothing is hidden (the selected
            # card was the only one past the cap) there is no remainder to
            # name: a `… +0 more` reads as leftovers (review round 2, R2-1).
            if hidden:
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
    return blocks


def _stack_board_blocks(blocks: list[list[Text]]) -> list[Text]:
    """Stack column blocks side by side, cell-padded and trailing-trimmed."""
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
    return lines


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
    painted in the card's marker column (S3b), never shifting the grid. With
    teams present each team paints as a stacked column-block under its own
    header row, holding only the columns that carry cards (S6d parity, design
    §6); with none the flat board is byte-identical to the shipped one.
    """
    resolver = _styles(style_for)
    mine = associated or frozenset()
    rendered = views[:PROJECTS_MAX]
    selected_id: str | None = None
    if cursor is not None and 0 <= cursor < len(rendered):
        selected_id = str(_row(rendered[cursor]).get("id") or "")
    if not rendered:
        from local_operator.slash_commands import project_empty_text

        empty = Text(project_empty_text(), style=resolver("dim"))
        return RenderResult(text=empty, width=cell_len(empty.plain), height=1)
    groups = sections_of(rendered)
    if groups is None:
        blocks = _board_blocks(
            _columns_of(rendered),
            selected_id=selected_id,
            mine=mine,
            now=now,
            resolver=resolver,
        )
        if len(views) > PROJECTS_MAX:
            blocks[0].append(Text(""))
            blocks[0].append(
                Text(f"… +{len(views) - PROJECTS_MAX} more not shown", style=resolver("dim"))
            )
        lines = _stack_board_blocks(blocks)
        width = max(cell_len(line.plain) for line in lines)
        return RenderResult(text=Text("\n").join(lines), width=width, height=len(lines))
    overflow = (
        f"… +{len(views) - PROJECTS_MAX} more not shown" if len(views) > PROJECTS_MAX else None
    )
    lines: list[Text] = []
    bands: list[tuple[int, int, str]] = []
    for label, indexes in groups:
        section_views = [rendered[index] for index in indexes]
        blocks = _board_blocks(
            _columns_of(section_views, skip_empty=True),
            selected_id=selected_id,
            mine=mine,
            now=now,
            resolver=resolver,
        )
        if overflow is not None:
            # The PROJECTS_MAX overflow row rides the FIRST painted column of
            # the first section — the shipped placement, kept per the design.
            blocks[0].append(Text(""))
            blocks[0].append(Text(overflow, style=resolver("dim")))
            overflow = None
        body = _stack_board_blocks(blocks)
        fill_width = max((cell_len(line.plain) for line in body), default=0)
        start = len(lines)
        lines.append(_section_header_text(label, len(indexes), fill_width, resolver))
        lines.extend(body)
        bands.append((start, len(lines), label))
    width = max(cell_len(line.plain) for line in lines)
    return RenderResult(
        text=Text("\n").join(lines), width=width, height=len(lines), sections=tuple(bands)
    )


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
    label = display_name(project) or "(unnamed)"
    key = str(project.get("name") or "")
    name = Text(label, style=style_for("name"))
    if project.get("title") and key and cell_len(f"{label} ({key})") <= TIMELINE_NAME_WIDTH - 2:
        # Same rule as the board card: both only when the fixed name column
        # holds them (here minus the two marker cells); the title wins a tight
        # fit.
        name.append(f" ({key})", style=style_for("dim"))
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


def _tail_entries(
    undated: list[dict[str, Any]],
    *,
    selected_id: str | None,
    mine: frozenset[str],
    resolver: StyleFor,
    offset: int = 0,
) -> tuple[Text, list[tuple[int, int, str]]]:
    """One tail line's items: the text, and each name's ``(x0, x1, id)`` range.

    The tail is free-form (no fixed name column to align), so a marked name
    carries its marker cells and an unmarked one stays bare — the shipped
    rule. The cell RANGES (markers included, ``x1`` exclusive) are what let a
    click select the item it landed on: the grouped tail paints one line per
    team, and one line lists several projects. ``offset`` shifts the recorded
    ranges onto the painted line's canvas coordinates — the prefix
    (``no dates (…)``) is not part of the entries text, but clicks see it.
    """
    line = Text()
    ranges: list[tuple[int, int, str]] = []
    for index, view in enumerate(undated):
        if index:
            line.append(", ", style=resolver("dim"))
        row_id = str(_row(view).get("id") or "")
        start = cell_len(line.plain)
        marked = (selected_id is not None and row_id == selected_id) or row_id in mine
        if marked:
            # The tail is free-form (no fixed name column to align), so a
            # marked name carries its cells and an unmarked one stays bare.
            for glyph, key in _marker_cells(
                selected=selected_id is not None and row_id == selected_id,
                associated=row_id in mine,
            ):
                line.append(glyph, style=resolver(key) if key != "dim" else resolver("dim"))
        line.append(
            display_name(_row(view)) or "(unnamed)",
            style=resolver("name") if marked else resolver("dim"),
        )
        ranges.append((offset + start, offset + cell_len(line.plain), row_id))
    return line, ranges


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
    rows (S3b), both inside the fixed name field. With teams present the dated
    rows group under painted section headers and the tail prints one line per
    team (S6d parity, design §6); with none the chart is byte-identical to the
    shipped one.
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
    groups = sections_of(rendered)

    lines: list[Text] = []
    bands: list[tuple[int, int, str]] = []
    axis_cells = 0
    axis_start: date | None = None
    if span is not None and dated:
        axis_start, end = span
        axis_cells = timeline_axis_cells(axis_start, end, tier)
        lines.append(
            _timeline_axis(
                axis_start, end, axis_cells=axis_cells, tier=tier, today=moment, style_for=resolver
            )
        )

    def paint_dated(view: dict[str, Any]) -> Text:
        row_id = str(_row(view).get("id") or "")
        return _timeline_row(
            view,
            start=cast(date, axis_start),
            axis_cells=axis_cells,
            tier=tier,
            today=moment,
            selected=selected_id is not None and row_id == selected_id,
            associated=row_id in mine,
            style_for=resolver,
        )

    if groups is None:
        for view in dated:
            lines.append(paint_dated(view))
    else:
        # Grouped (S6d parity, design §6): section header rows interrupt the
        # dated rows, filled to the section block's width, and the `no dates`
        # tail prints one line PER TEAM present among the undated so the team
        # structure survives the tail.
        dated_ids = {str(_row(view).get("id") or "") for view in dated}
        for label, indexes in groups:
            section_dated = [
                rendered[index]
                for index in indexes
                if str(_row(rendered[index]).get("id") or "") in dated_ids
            ]
            if not section_dated:
                continue
            rows = [paint_dated(view) for view in section_dated]
            fill_width = max(cell_len(row.plain) for row in rows)
            start = len(lines)
            # The count is the SECTION's painted projects (dated + its tail) —
            # the one number every surface shows for a team; the tail line's
            # own parenthetical counts the names it lists (UX round 1, U2).
            lines.append(_section_header_text(label, len(indexes), fill_width, resolver))
            lines.extend(rows)
            bands.append((start, len(lines), label))

    if undated:
        if lines:
            lines.append(Text(""))
        if groups is None:
            # The tail is ONE line, and the markers live ON it: an undated
            # project is still selectable and still part of the session's set,
            # so both glyphs are painted beside its name — otherwise `↵` would
            # aim at a row with no visible selection at all (agent review
            # round 1, F1).
            tail = Text()
            tail.append(f"no dates ({len(undated)}): ", style=resolver("dim"))
            entries, _ranges = _tail_entries(
                undated, selected_id=selected_id, mine=mine, resolver=resolver
            )
            tail.append_text(entries)
            lines.append(tail)
        else:
            undated_ids = {str(_row(view).get("id") or "") for view in undated}
            for label, indexes in groups:
                section_undated = [
                    rendered[index]
                    for index in indexes
                    if str(_row(rendered[index]).get("id") or "") in undated_ids
                ]
                if not section_undated:
                    continue
                start = len(lines)
                tail = Text()
                tail.append(f"no dates ({label}, {len(section_undated)}): ", style=resolver("dim"))
                entries, _ranges = _tail_entries(
                    section_undated, selected_id=selected_id, mine=mine, resolver=resolver
                )
                tail.append_text(entries)
                lines.append(tail)
                bands.append((start, len(lines), label))
    if not lines:
        from local_operator.slash_commands import project_empty_text

        lines.append(Text(project_empty_text(), style=resolver("dim")))
    width = max(cell_len(line.plain) for line in lines)
    return RenderResult(
        text=Text("\n").join(lines), width=width, height=len(lines), sections=tuple(bands)
    )


def _board_card_position(
    columns: list[tuple[str, list[dict[str, Any]]]],
    target: str,
    *,
    y_base: int,
) -> tuple[int, int] | None:
    """``(x, y)`` of ``target``'s card name line within one column block.

    One section's arithmetic, shared by the flat and the grouped reveals so a
    section shift cannot drift the two apart; ``y_base`` is where this block's
    first header line sits (0 for the flat board, the section header + 1 when
    grouped).
    """
    x = 0
    for _status, rows in columns:
        visible, hidden, extra = _painted_cards(rows, target)
        for index, view in enumerate(visible):
            if str(_row(view).get("id") or "") == target:
                # Each card is a blank line and three content lines under the
                # column header: its name line sits at 2 + index * 4.
                return (x * BOARD_COLUMN_WIDTH, y_base + 2 + index * 4)
        if extra is not None and str(_row(extra).get("id") or "") == target:
            # Past the cap the selected card is painted after the overflow
            # note (or straight after the last card, when nothing is hidden):
            # 4 lines per visible card, then the note's blank + line when one
            # is painted, then this card's blank — its NAME line lands at
            # 4 + 4 * V with a note and 2 + 4 * V without one.
            y = (4 if hidden else 2) + 4 * len(visible)
            return (x * BOARD_COLUMN_WIDTH, y_base + y)
        x += 1
    return None


def _board_block_height(
    columns: list[tuple[str, list[dict[str, Any]]]],
    *,
    selected_id: str | None,
    extra_note: bool,
) -> int:
    """Lines one stacked column-block paints — the painter's own arithmetic.

    ``1 + 4V + 2(when cards are hidden) + 4(when a past-cap selected card is
    painted)`` per column (the card/note arithmetic above), maxed across the
    columns; ``extra_note`` adds the PROJECTS_MAX overflow row's blank + line
    to the FIRST column, where the painter places it.
    """
    heights: list[int] = []
    for _status, rows in columns:
        visible, hidden, extra = _painted_cards(rows, selected_id)
        heights.append(
            1 + 4 * len(visible) + (2 if hidden else 0) + (4 if extra is not None else 0)
        )
    if extra_note and heights:
        heights[0] += 2
    return max(heights, default=1)


def board_position(views: list[dict[str, Any]], cursor: int) -> tuple[int, int] | None:
    """The board-canvas ``(x, y)`` of ``views[cursor]``'s card, or ``None``.

    ``None`` means the card is NOT painted — past ``PROJECTS_MAX`` or its
    column's ``BOARD_CARDS_MAX`` — which is what keeps the page's selection off
    a cell no reader can see (design round 1, D2's rule, applied to the board).
    The point is the card's NAME line, where the marker column lives, so a
    reveal that lands there shows the selection itself and not just its card.
    Grouped canvases add each section's header row and the height of the
    sections above (``_board_block_height``), so the reveal lands on the card
    the painter drew.
    """
    rendered = views[:PROJECTS_MAX]
    if not (0 <= cursor < len(rendered)):
        return None
    target = str(_row(rendered[cursor]).get("id") or "")
    groups = sections_of(rendered)
    if groups is None:
        return _board_card_position(_columns_of(rendered), target, y_base=0)
    y_base = 0
    for index, (_label, indexes) in enumerate(groups):
        section_views = [rendered[position] for position in indexes]
        columns = _columns_of(section_views, skip_empty=True)
        found = _board_card_position(columns, target, y_base=y_base + 1)
        if found is not None:
            return found
        y_base += 1 + _board_block_height(
            columns,
            selected_id=target,
            extra_note=index == 0 and len(views) > PROJECTS_MAX,
        )
    return None


def timeline_position(views: list[dict[str, Any]], cursor: int) -> tuple[int, int] | None:
    """The timeline-canvas ``(x, y)`` of ``views[cursor]``'s row, or ``None``.

    Dated projects are one row each under the axis; the undated tail is ONE
    line naming them all, so an undated selection answers with THAT line's
    ``y`` — its name is on it. Grouped canvases add each section's header row
    and the section's rows above (and, for the tail, one line per team).
    ``None`` means the index is past ``PROJECTS_MAX`` or its project has no
    painted line at all.
    """
    rendered = views[:PROJECTS_MAX]
    if not (0 <= cursor < len(rendered)):
        return None
    target = str(_row(rendered[cursor]).get("id") or "")
    dated, undated = _split_dated(rendered)
    span = timeline_span(rendered)
    groups = sections_of(rendered)
    if groups is None:
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
    dated_ids = {str(_row(view).get("id") or "") for view in dated}
    undated_ids = {str(_row(view).get("id") or "") for view in undated}
    y = 0
    if span is not None and dated:
        y = 1
        for _label, indexes in groups:
            section_dated = [
                index
                for index in indexes
                if str(_row(rendered[index]).get("id") or "") in dated_ids
            ]
            if not section_dated:
                continue
            y += 1  # the section header row
            for index in section_dated:
                if str(_row(rendered[index]).get("id") or "") == target:
                    return (0, y)
                y += 1
    if undated:
        if y:
            y += 1
        for _label, indexes in groups:
            section_undated = [
                index
                for index in indexes
                if str(_row(rendered[index]).get("id") or "") in undated_ids
            ]
            if not section_undated:
                continue
            if any(
                str(_row(rendered[index]).get("id") or "") == target for index in section_undated
            ):
                return (0, y)
            y += 1
    return None


# ---------------------------------------------------------------------------
# Canvas hit tests and the section ruler (S6d parity)
# ---------------------------------------------------------------------------
# These are the INVERSE of the painters above and read the same primitives
# (`sections_of`, `_columns_of`, `_painted_cards`, `_split_dated`, the card
# height arithmetic), so what answers is what was painted; tests cross-check
# every position against the painted lines.


class _Band(NamedTuple):
    """One painted section band, as the painters lay it out.

    ``end`` is EXCLUSIVE. ``target`` is the project index a click on this
    band's header selects — the first row of the block the header actually
    heads (a mixed team's timeline chart header lands on its first DATED row,
    not on an undated member's tail line; UX round 1, U3). ``group`` is the
    index into :func:`sections_of`'s groups, so the ruler compares the
    cursor's section with the band's by IDENTITY — a real team named
    `no team` must never alias the sentinel bucket (R1-2/U1).
    """

    start: int
    end: int
    label: str
    target: int
    group: int


def _bands_for(
    views: list[dict[str, Any]],
    view_type: str,
    *,
    cursor: int | None = None,
) -> tuple[_Band, ...]:
    """The section bands ``render_*`` paints, recomputed for hit tests.

    Built from the FULL ``views``, not a pre-capped list: the board's overflow
    row exists when ``len(views) > PROJECTS_MAX`` and it shifts every row
    after section 0 down by two, so a capped list here made the ruler drift on
    >200-project boards (review round 1, R1-1). ``cursor`` matters on the
    board (a past-cap SELECTED card is painted past the cap, which shifts
    later sections) — the same term the painters read. Empty on an ungrouped
    canvas.
    """
    rendered = views[:PROJECTS_MAX]
    groups = sections_of(rendered)
    if groups is None:
        return ()
    selected_id: str | None = None
    if cursor is not None and 0 <= cursor < len(rendered):
        selected_id = str(_row(rendered[cursor]).get("id") or "")
    if view_type == "list":
        bands: list[_Band] = []
        y = 0
        for group_index, (label, indexes) in enumerate(groups):
            start = y
            y += 1 + len(indexes)
            bands.append(_Band(start, y, label, indexes[0], group_index))
        return tuple(bands)
    if view_type == "board":
        bands = []
        y = 0
        for group_index, (label, indexes) in enumerate(groups):
            section_views = [rendered[position] for position in indexes]
            columns = _columns_of(section_views, skip_empty=True)
            start = y
            y += 1 + _board_block_height(
                columns,
                selected_id=selected_id,
                extra_note=group_index == 0 and len(views) > PROJECTS_MAX,
            )
            bands.append(_Band(start, y, label, indexes[0], group_index))
        return tuple(bands)
    if view_type == "timeline":
        dated, undated = _split_dated(rendered)
        dated_ids = {str(_row(view).get("id") or "") for view in dated}
        undated_ids = {str(_row(view).get("id") or "") for view in undated}
        span = timeline_span(rendered)
        bands = []
        y = 1 if (span is not None and dated) else 0
        for group_index, (label, indexes) in enumerate(groups):
            section_dated = [
                index
                for index in indexes
                if str(_row(rendered[index]).get("id") or "") in dated_ids
            ]
            if not section_dated:
                continue
            start = y
            y += 1 + len(section_dated)
            bands.append(_Band(start, y, label, section_dated[0], group_index))
        if undated:
            if y:
                y += 1  # the blank line between the chart and the tail
            for group_index, (label, indexes) in enumerate(groups):
                section_undated = [
                    index
                    for index in indexes
                    if str(_row(rendered[index]).get("id") or "") in undated_ids
                ]
                if not section_undated:
                    continue
                bands.append(_Band(y, y + 1, label, section_undated[0], group_index))
                y += 1
        return tuple(bands)
    return ()


def _board_card_at(
    views_list: list[dict[str, Any]],
    columns: list[tuple[str, list[dict[str, Any]]]],
    x: int,
    y: int,
    *,
    y_base: int,
    selected_id: str | None,
) -> int | None:
    """Index (into ``views_list``) of the card at ``(x, y)`` in one block."""
    if x < 0 or x % BOARD_COLUMN_WIDTH >= BOARD_COLUMN_WIDTH - 1:
        return None  # the one-cell gutter between columns is not a card
    column_index = x // BOARD_COLUMN_WIDTH
    if column_index >= len(columns):
        return None
    _status, rows = columns[column_index]
    visible, hidden, extra = _painted_cards(rows, selected_id)
    index_by_id = {
        str(_row(view).get("id") or ""): position for position, view in enumerate(views_list)
    }
    for position, view in enumerate(visible):
        name_y = y_base + 2 + position * 4
        if name_y <= y < name_y + 3:
            return index_by_id.get(str(_row(view).get("id") or ""))
    if extra is not None:
        name_y = y_base + (4 if hidden else 2) + 4 * len(visible)
        if name_y <= y < name_y + 3:
            return index_by_id.get(str(_row(extra).get("id") or ""))
    return None


def project_at(
    views: list[dict[str, Any]],
    view_type: str,
    x: int,
    y: int,
    *,
    cursor: int | None = None,
    associated: frozenset[str] | None = None,
) -> int | None:
    """Index into ``views`` of the painted project at canvas cell ``(x, y)``.

    THE click map's one lookup: list rows address their whole row (x carries
    nothing a row does not), board cards their three content lines (the blank
    above a card and the `… +N more` notes are not cards), the timeline its
    rows and its per-team tail lines — where x picks the NAME the click landed
    on, because one tail line lists several projects. ``None`` for header
    rows, blanks, fillers and the truncation row; ``cursor``/``associated`` are
    the page's selection and marker set, so a past-cap selected card (which IS
    painted) answers too.
    """
    rendered = views[:PROJECTS_MAX]
    if y < 0:
        return None
    mine = associated or frozenset()
    if view_type == "list":
        groups = sections_of(rendered)
        if groups is None:
            return y if 0 <= y < len(rendered) else None
        row = 0
        for _label, indexes in groups:
            row += 1  # the section header
            for index in indexes:
                if row == y:
                    return index
                row += 1
        return None
    if view_type == "board":
        selected_id: str | None = None
        if cursor is not None and 0 <= cursor < len(rendered):
            selected_id = str(_row(rendered[cursor]).get("id") or "")
        groups = sections_of(rendered)
        if groups is None:
            return _board_card_at(
                rendered, _columns_of(rendered), x, y, y_base=0, selected_id=selected_id
            )
        y_base = 0
        for index, (_label, indexes) in enumerate(groups):
            section_views = [rendered[position] for position in indexes]
            columns = _columns_of(section_views, skip_empty=True)
            found = _board_card_at(
                section_views, columns, x, y, y_base=y_base + 1, selected_id=selected_id
            )
            if found is not None:
                return indexes[found]
            y_base += 1 + _board_block_height(
                columns,
                selected_id=selected_id,
                extra_note=index == 0 and len(views) > PROJECTS_MAX,
            )
        return None
    if view_type == "timeline":
        selected_id = None
        if cursor is not None and 0 <= cursor < len(rendered):
            selected_id = str(_row(rendered[cursor]).get("id") or "")
        index_by_id = {
            str(_row(view).get("id") or ""): position for position, view in enumerate(rendered)
        }
        groups = sections_of(rendered)
        dated, undated = _split_dated(rendered)
        date_ids = {str(_row(view).get("id") or "") for view in dated}
        undated_ids = {str(_row(view).get("id") or "") for view in undated}
        span = timeline_span(rendered)
        row = 0
        if span is not None and dated:
            row = 1
            if groups is None:
                for view in dated:
                    if row == y:
                        return index_by_id.get(str(_row(view).get("id") or ""))
                    row += 1
            else:
                for _label, indexes in groups:
                    section_dated = [
                        index
                        for index in indexes
                        if str(_row(rendered[index]).get("id") or "") in date_ids
                    ]
                    if not section_dated:
                        continue
                    row += 1  # the section header row
                    for index in section_dated:
                        if row == y:
                            return index
                        row += 1
        if undated:
            if row:
                row += 1  # the blank line between the chart and the tail
            if groups is None:
                if row == y:
                    prefix = f"no dates ({len(undated)}): "
                    _line, ranges = _tail_entries(
                        undated,
                        selected_id=selected_id,
                        mine=mine,
                        resolver=_styles(None),
                        offset=cell_len(prefix),
                    )
                    for x0, x1, row_id in ranges:
                        if x0 <= x < x1:
                            return index_by_id.get(row_id)
                return None
            for _label, indexes in groups:
                section_undated = [
                    index
                    for index in indexes
                    if str(_row(rendered[index]).get("id") or "") in undated_ids
                ]
                if not section_undated:
                    continue
                if row == y:
                    prefix = f"no dates ({_label}, {len(section_undated)}): "
                    _line, ranges = _tail_entries(
                        [rendered[index] for index in section_undated],
                        selected_id=selected_id,
                        mine=mine,
                        resolver=_styles(None),
                        offset=cell_len(prefix),
                    )
                    for x0, x1, row_id in ranges:
                        if x0 <= x < x1:
                            return index_by_id.get(row_id)
                    return None
                row += 1
        return None
    return None


def section_at(
    views: list[dict[str, Any]],
    view_type: str,
    x: int,
    y: int,
    *,
    cursor: int | None = None,
) -> str | None:
    """The label of the section band containing canvas ``(x, y)``, or ``None``.

    Bands are full-width — a row belongs to a section regardless of x — and
    run from the section's header row through its last painted row (or tail
    line), the span the ruler tracks and the jumps address.
    """
    for band in _bands_for(views, view_type, cursor=cursor):
        if band.start <= y < band.end:
            return band.label
    return None


def section_header_at(
    views: list[dict[str, Any]],
    view_type: str,
    x: int,
    y: int,
    *,
    cursor: int | None = None,
) -> tuple[str, int] | None:
    """``(label, target)`` when ``(x, y)`` is a section's header row.

    ``target`` is the project index a click selects: the first row of the
    block the header actually HEADS (a mixed team's timeline chart header
    lands on its first DATED row, not on an undated member's tail line — UX
    round 1, U3). A timeline tail line answers with its own first name, so
    both bands of a split section are reachable by their own row. Headers are
    the click targets that jump to a section (`section_at` would also match a
    project row, which clicks handle as a selection).
    """
    for band in _bands_for(views, view_type, cursor=cursor):
        if band.start == y:
            return (band.label, band.target)
    return None


def section_ruler(
    views: list[dict[str, Any]],
    view_type: str,
    *,
    top_row: int,
    cursor: int | None,
    width: int,
    style_for: StyleFor | None = None,
) -> Text | None:
    """The rule row as a section ruler: ``── core · 4 ─────`` or ``None``.

    ``None`` means "no section at the viewport top" — an ungrouped canvas, the
    axis row, a blank separator, the truncation row — and the caller paints
    the shipped plain rule (design D2: the sticky counterpart is this ZERO-row
    ruler, never faked motion). The count is the SECTION's painted projects —
    the one number every surface shows for a team (UX round 1, U2; the
    timeline tail's own parenthetical counts the names it lists). When the
    cursor sits in a DIFFERENT section than the one at the viewport top, a dim
    ``· sel {team}`` clause says so; the comparison is by group IDENTITY, so a
    real team named like the sentinel can never alias it (R1-2/U1). Shed
    order (design §6): the sel clause first, then the count, then — only
    below the label's own fit — the label truncated with an ellipsis; the team
    name never simply vanishes.
    """
    resolver = _styles(style_for)
    rendered = views[:PROJECTS_MAX]
    groups = sections_of(rendered)
    if groups is None:
        return None
    hit: _Band | None = None
    for band in _bands_for(views, view_type, cursor=cursor):
        if band.start <= top_row < band.end:
            hit = band
            break
    if hit is None:
        return None
    label_at = hit.label
    count_at = len(groups[hit.group][1])
    cursor_label: str | None = None
    if cursor is not None and 0 <= cursor < len(rendered):
        for group_index, (label, indexes) in enumerate(groups):
            # IDENTITY, not the label: the cursor's OWN group is the one whose
            # membership names it, whatever two sections happen to be called.
            if cursor in indexes:
                if group_index != hit.group:
                    cursor_label = label
                break

    return _ruler_line(
        label_at, count=str(count_at), sel=cursor_label, width=width, resolver=resolver
    )


def _ruler_line(
    label: str,
    *,
    count: str | None,
    sel: str | None,
    width: int,
    resolver: Callable[[str], Style],
    count_paren: bool = False,
) -> Text:
    """One section-ruler line, shed to fit — the ladder BOTH rulers share.

    Candidates in shed order: count + `sel` clause → count → bare label →
    truncated label; the name never simply vanishes and the fill always lands
    on the row's own width (design §6). The canvas form's count is the dot
    clause (` · 4 `); the detail page's is the parenthesised one the spec's
    sketch shows (`── updates (3) ──`), toggled by ``count_paren``. The canvas
    form's `· sel` clause is UX round 1's U3 remainder.
    """

    def fill(line: Text) -> Text:
        pad = width - cell_len(line.plain)
        if pad > 0:
            line.append("─" * pad, style=resolver("dim"))
        return line

    def build(*, with_sel: bool, with_count: bool) -> Text:
        line = Text(no_wrap=True)
        line.append("── ", style=resolver("dim"))
        line.append(label, style=resolver("name"))
        tail = ""
        if with_count and count is not None:
            tail += f" ({count}) " if count_paren else f" · {count} "
        if with_sel and sel is not None:
            tail += f"· sel {sel} "
        if not tail:
            # With the count (and sel) shed, keep a space so the label does
            # not run into the fill (`── core───` reads as one word).
            tail = " "
        line.append(tail, style=resolver("dim"))
        return fill(line)

    candidates: list[Text] = []
    if sel is not None:
        candidates.append(build(with_sel=True, with_count=True))
    candidates.append(build(with_sel=False, with_count=True))
    candidates.append(build(with_sel=False, with_count=False))
    for candidate in candidates:
        if cell_len(candidate.plain) <= width:
            return candidate
    room = width - cell_len("── ") - 1
    if room <= 0:
        return fill(Text(no_wrap=True))
    line = Text(no_wrap=True)
    line.append("── ", style=resolver("dim"))
    line.append(truncate_row(label, cap=room), style=resolver("name"))
    line.append(" ", style=resolver("dim"))
    return fill(line)


def detail_ruler(
    sections: Sequence[tuple[int, str, str | None]],
    top_row: int,
    *,
    width: int,
    style_for: StyleFor | None = None,
) -> Text | None:
    """The detail page's rule row: the in-page section at the viewport top.

    ``sections`` is ``(start_row, label, count)`` in paint order — the ruler
    names the LAST section that starts at or above ``top_row`` (design §6:
    "on the detail page the same row names the current in-page section"),
    shedding its count before the name through the canvas ruler's own ladder.
    ``None`` (no sections) leaves the caller painting the plain rule.
    """
    resolver = _styles(style_for)
    hit: tuple[int, str, str | None] | None = None
    for start, label, count in sections:
        if start <= top_row:
            hit = (start, label, count)
        else:
            break
    if hit is None:
        return None
    return _ruler_line(
        hit[1], count=hit[2], sel=None, width=width, resolver=resolver, count_paren=True
    )


# ---------------------------------------------------------------------------
# The pinned footer content (chrome, not canvas)
# ---------------------------------------------------------------------------


def today_iso(now: float | None = None) -> str:
    """Today as ISO, on the SAME basis the store stamps ``completed_at`` from.

    Delegates to :func:`local_operator.projects._local_today` — the operator's
    LOCAL day — so the footer's ``[overdue]``/``[completed]`` and the page
    rows, the tool receipt and the stored stamp cannot disagree about which
    day it is (agent review round 5: this was the last UTC reader, and an
    evening probe had the footer call a milestone overdue while the rows and
    the tool called it upcoming). ``now`` is a POSIX timestamp, kept for the
    freshness callers; it is converted on the same local basis.
    """
    import datetime as _datetime

    from local_operator.projects import _local_today

    if now is None:
        return _local_today().isoformat()
    return _datetime.datetime.fromtimestamp(now).date().isoformat()


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
    (UX round 1, U1). The order keeps the footer's stated purpose longest, the
    identity always surviving: the progress body goes first (its age is also
    on the canvas row), then the session rollup, then the identity's ``(key)``
    span — painted on the canvas row and recoverable in the detail/receipts,
    while the milestone rollup has no other home (design review round 1, D1)
    — and the milestones last. When even the bare identity does not fit it is
    ellipsized explicitly, never cut silently.
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
    # Title-first, key secondary. The key span is its own SHEDDABLE piece: it
    # is also painted on the canvas row and recoverable in the detail and
    # receipts, while the milestone rollup has no other home — so the ladder
    # trades the key away before the rollup (design review round 1, D1).
    name_text = display_name(project) or "(unnamed)"
    status = str(project.get("status") or "active")
    key = str(project.get("name") or "")
    keyed_identity = bool(project.get("title") and key)
    identity.append(name_text, style=resolver("name"))
    if keyed_identity:
        identity.append(f" ({key})", style=resolver("dim"))
    identity.append(" " + status_chip_text(status), style=chip(status))
    clauses["identity"] = identity
    if keyed_identity:
        keyless_identity = Text(no_wrap=True)
        keyless_identity.append(name_text, style=resolver("name"))
        keyless_identity.append(" " + status_chip_text(status), style=chip(status))
    else:
        keyless_identity = identity

    progress = Text(no_wrap=True)
    age = progress_age_text(view, now=now)
    if age is None:
        progress.append("progress: none recorded", style=resolver("dim"))
    else:
        stale = " · stale" if view.get("progress_stale") else ""
        by = str(project.get("progress_reported_by") or "")
        reporter = f" by {by}" if by else ""
        # The refresh assertion, when live, rides the same clause (detail
        # contract): "reported 5h ago by X · stale · refreshed 1h ago by Y —
        # no new content since 2026-09-29".
        note = refreshed_note(project, now=now)
        checked = f" · {note}" if note is not None else ""
        progress.append(
            f"progress reported {age} ago{reporter}{stale}{checked}: ", style=resolver("dim")
        )
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
        # Work links and filings are distinct clauses: the working list is
        # what the runtime words describe; a filing says `filed by <id>` and
        # carries no runtime claim of its own.
        working_rows = [
            row for row in rows if not (isinstance(row, dict) and row.get("role") == "coordination")
        ]
        filed_rows = [
            row for row in rows if isinstance(row, dict) and row.get("role") == "coordination"
        ]
        if working_rows:
            sessions.append("working: ", style=resolver("dim"))
            bits: list[str] = []
            for row in working_rows[:4]:
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
                subagents: dict[str, Any] = (
                    subagents_value if isinstance(subagents_value, dict) else {}
                )
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
            if len(working_rows) > 4:
                sessions.append(f" · +{len(working_rows) - 4} more", style=resolver("dim"))
        if filed_rows:
            if working_rows:
                sessions.append(" · ", style=resolver("dim"))
            filed_ids = ", ".join(str(row.get("session_id") or "") for row in filed_rows[:3])
            more = f" +{len(filed_rows) - 3} more" if len(filed_rows) > 3 else ""
            sessions.append(f"filed by {filed_ids}{more}", style=resolver("dim"))
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
    # the canvas row), then the session rollup, then the identity's KEY span
    # (also on the canvas row, recoverable in the detail/receipts — design
    # review round 1, D1), then the milestones — so the 192-235-cell band
    # keeps the rollup instead of the body (agent review round 2), and a
    # titled identity at a 96-cell box keeps the rollup instead of shedding
    # the whole milestone clause (D1). Every rung keeps the (bare) identity.
    no_progress = tuple(key for key in order if key != "progress")
    no_sessions = tuple(key for key in no_progress if key != "sessions")
    no_milestones = tuple(key for key in no_sessions if key != "milestones")
    rungs: list[tuple[tuple[str, ...], bool]] = [
        (order, True),
        (no_progress, True),
        (no_sessions, True),
        (no_sessions, False),
        (no_milestones, True),
        (no_milestones, False),
    ]

    def compose(keys: tuple[str, ...], with_key: bool = True) -> Text:
        row_text = Text(no_wrap=True)
        for index, key in enumerate(keys):
            if index:
                row_text.append("  ·  ", style=resolver("dim"))
            if key == "identity" and not with_key:
                row_text.append_text(keyless_identity)
            else:
                row_text.append_text(clauses[key])
        if keys != order or (keyed_identity and not with_key):
            # Say that more exists; a footer that stopped at a clause (or key)
            # boundary with no marker reads as if it were the whole story.
            row_text.append(" …", style=resolver("dim"))
        return row_text

    if width is None:
        return compose(order)
    fitted = compose(order)
    for keys, with_key in rungs:
        candidate = compose(keys, with_key=with_key)
        if cell_len(candidate.plain) <= width:
            fitted = candidate
            break
    if cell_len(fitted.plain) > width:
        # A blind truncate can cut INSIDE the status chip (`… [p…` — UX round
        # 2, U5). The chip is a whole token, so when it fits alone the NAME
        # gives way first and the chip survives intact; only when even the
        # chip cannot fit does the line ellipsize mid-token.
        chip_text = status_chip_text(status)
        if cell_len(chip_text) + 2 <= width:
            name_cap = max(width - cell_len(chip_text) - 1, 1)
            fitted = Text(truncate_row(name_text, cap=name_cap) + " " + chip_text)
        if cell_len(fitted.plain) > width:
            fitted.truncate(width, overflow="ellipsis")
    return fitted


# ---------------------------------------------------------------------------
# The detail page (S6d parity P2): pure text for its rows, heading and footer
# ---------------------------------------------------------------------------


def format_short_date(value: str | None, *, today: str | None = None) -> str | None:
    """``2026-10-04`` → ``4 Oct`` (the year appended only across a boundary).

    The detail's prose contexts state a date the way a person says it; a date
    in another year carries the year, because "4 Oct" alone would read as the
    next one. Unparseable values come back untouched — a store row is data,
    never a crash (the read-side rule :func:`milestone_state` states).
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        day = date.fromisoformat(text[:10])
    except ValueError:
        return text
    label = f"{day.day} {day.strftime('%b')}"
    basis = str(today or today_iso())[:10]
    try:
        if day.year != date.fromisoformat(basis).year:
            label += f" {day.year}"
    except ValueError:
        pass
    return label


def detail_section_heading(name: str, count: str | None = None) -> tuple[str, str]:
    """The detail page's two-line section heading: text + its dash underline.

    ``milestones (1/3)`` / ``sessions (3)``; the underline is exactly as wide
    as the heading, so it reads as an underline rather than as a rule.
    """
    head = f"{name} ({count})" if count else name
    return head, "─" * cell_len(head)


def detail_meta_line(
    view: dict[str, Any], *, width: int, style_for: StyleFor | None = None
) -> Text:
    """The overview sentence, fitted by shedding WHOLE clauses.

    Absent fields drop out; nothing set reads the shipped sentence ``no
    estimate or dates``. Shed order drops the sentence's TAIL first (tags,
    est, completed, start, target, team), so the deadline survives the start
    date and "which stream, and is it current" survives longest (design
    review round 1, D2).
    """
    resolver = _styles(style_for)
    project = _row(view)
    clauses: list[tuple[str, str]] = []
    if view.get("progress_stale"):
        clauses.append(("[stale]", "stale"))
    owner = str(project.get("owner") or "")
    if owner:
        clauses.append((f"owner: {owner}", "dim"))
    team = str(project.get("team") or "")
    if team:
        clauses.append((f"team: {team}", "dim"))
    # The TARGET leads its start: the deadline is the "is this on track" fact,
    # and whole-clause shedding pops from the tail — putting start last means
    # the deadline outlives it at narrow widths (design review round 1, D2;
    # the canvas rows lead with `→{target}` for the same reason).
    target = format_short_date(project.get("target_date"))
    if target:
        clauses.append((f"target {target}", "dim"))
    start = format_short_date(project.get("start_date"))
    if start:
        clauses.append((f"start {start}", "dim"))
    completed = format_short_date(project.get("completed_at"))
    if completed:
        clauses.append((f"completed {completed}", "dim"))
    estimate = estimate_text(project)
    if estimate:
        clauses.append((estimate, "dim"))
    tags = [str(tag) for tag in project.get("tags") or [] if str(tag)]
    if tags:
        clauses.append((f"tags {' · '.join(tags)}", "dim"))

    def compose(kept: list[tuple[str, str]]) -> Text:
        line = Text(no_wrap=True)
        for index, (text, key) in enumerate(kept):
            if index:
                line.append(" · ", style=resolver("dim"))
            line.append(text, style=resolver(key))
        return line

    if not clauses:
        return Text("no estimate or dates", style=resolver("dim"), no_wrap=True)
    kept = list(clauses)
    while True:
        line = compose(kept)
        if cell_len(line.plain) <= width or len(kept) == 1:
            return line
        kept.pop()


def detail_progress_line(
    view: dict[str, Any], *, width: int, style_for: StyleFor | None = None
) -> Text:
    """The detail page's pinned footer: freshness attributed, or recorded absent.

    ``progress reported 2h ago by session ab12cd34ef56 · refreshed 1h ago by
    session ab12cd34ef56 — no new content since 2026-09-29 · updated 17:52``
    (``by the operator`` when the store says so). The refresh clause rides while
    the assertion is newer than the content (§4.3's detail contract). Fitted by
    shedding the update stamp first, then the refresh clause, then the
    attribution — the age is the point of the line. No record: ``no progress
    recorded``.
    """
    resolver = _styles(style_for)
    project = _row(view)
    age = progress_age_text(view)
    if age is None:
        return Text("no progress recorded", style=resolver("dim"), no_wrap=True)
    reporter = str(project.get("progress_reported_by") or "")
    attribution = ""
    if reporter and reporter != "operator":
        attribution = f" by session {reporter}"
    elif reporter == "operator":
        attribution = " by the operator"
    stamp = ""
    updated = project.get("updated_at")
    if isinstance(updated, (int, float)) and updated:
        import time as _time

        stamp = f" · updated {_time.strftime('%H:%M', _time.localtime(float(updated)))}"
    note = refreshed_note(project)
    refreshed = f" · {note}" if note is not None else ""
    base = f"progress reported {age} ago"
    ladder = (
        base + attribution + refreshed + stamp,
        base + attribution + refreshed,
        base + attribution + stamp,
        base + attribution,
        base,
    )
    for text in ladder:
        if cell_len(text) <= width or text == base:
            return Text(text, style=resolver("dim"), no_wrap=True)
    return Text(base, style=resolver("dim"), no_wrap=True)


#: Detail session-row state → style key. ``wedged``/``stale`` read the warning
#: tone the canvases already use for the same runtime states; ``missing`` is
#: the receipt's word for a gone directory, styled like the other caveats.
_SESSION_STATE_STYLES: dict[str, str] = {
    "live": "live",
    "wedged": "stale",
    "stale": "stale",
    "stopped": "dim",
    "missing": "stale",
    # A coordination row ("filed by") has no runtime state to word: the
    # receipt's word for it is `filed`, never `stopped` — nothing exists that
    # could be misread as a liveness claim (schema 2's role split).
    "filed": "dim",
}


def detail_session_state(row: dict[str, Any]) -> str:
    """One session row's state word — ``missing`` from ``exists``, like the jump.

    The composed row carries ``stopped`` for a record-less link; the receipt's
    word for a directory that is gone is ``missing``, and the two surfaces
    must not disagree (the rule ``action_jump``'s message states). A
    coordination row reads ``filed``: it is provenance, not a runtime state,
    and it must never be worded like one.
    """
    if row.get("role") == "coordination":
        return "filed"
    if row.get("exists") is False:
        return "missing"
    runtime_value = row.get("runtime")
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    return str(runtime.get("state") or "stopped")


def detail_milestone_row_text(
    row: dict[str, Any], *, selected: bool, style_for: StyleFor | None = None
) -> Text:
    """``{▸} ◆ groundwork · 2026-09-20 · completed`` — glyph trio from the timeline."""
    resolver = _styles(style_for)
    line = Text(no_wrap=True)
    for glyph, key in _marker_cells(selected=selected, associated=False):
        line.append(glyph, style=resolver(key) if key != "dim" else Style())
    state = milestone_state(row)
    glyph = {"completed": "◆", "overdue": "!", "upcoming": "◇"}.get(state, "◇")
    line.append(glyph, style=_milestone_style(resolver, state))
    line.append(f" {str(row.get('name') or '')}", style=resolver("name"))
    target = str(row.get("target_date") or "")
    if target:
        line.append(f" · {target}", style=resolver("dim"))
    line.append(f" · {state}", style=resolver("dim"))
    return line


def detail_session_row_text(
    row: dict[str, Any],
    *,
    selected: bool,
    own_session: str | None,
    style_for: StyleFor | None = None,
) -> Text:
    """``{▸}◆ ab12cd34ef56 [stopped] · "title" · 2 running/1 settled · todos 1/4``.

    ``◆`` marks the calling process's OWN session (the canvas marker column's
    convention); the counts appear only when the store knows them (``null``
    counts are omitted, never zeroed).
    """
    resolver = _styles(style_for)
    line = Text(no_wrap=True)
    session_id = str(row.get("session_id") or "")
    own = bool(own_session) and session_id == own_session
    for glyph, key in _marker_cells(selected=selected, associated=own):
        line.append(glyph, style=resolver(key) if key != "dim" else Style())
    line.append(session_id, style=resolver("name"))
    state = detail_session_state(row)
    line.append(f" [{state}]", style=resolver(_SESSION_STATE_STYLES.get(state, "dim")))
    title = str(row.get("title") or "")
    if title:
        line.append(f' · "{title}"', style=resolver("dim"))
    subagents = row.get("subagents")
    if isinstance(subagents, dict):
        running = subagents.get("running")
        settled = subagents.get("settled")
        if isinstance(running, int) and isinstance(settled, int):
            line.append(f" · {running} running/{settled} settled", style=resolver("dim"))
    todos = row.get("todos")
    if isinstance(todos, dict):
        open_count = todos.get("open")
        total = todos.get("total")
        if isinstance(open_count, int) and isinstance(total, int):
            line.append(f" · todos {open_count}/{total}", style=resolver("dim"))
    return line


def detail_todo_lines(
    view: dict[str, Any], *, own_session: str | None, style_for: StyleFor | None = None
) -> list[Text]:
    """The ``todos`` section: one line per session that ever snapshotted, or the
    single honest sentence when none did (``null`` counts are omitted, never
    zeroed). The calling session's line is labelled ``own session`` and carries
    its next items when the live overlay knows them."""
    resolver = _styles(style_for)
    lines: list[Text] = []
    for row in view.get("sessions") or []:
        if not isinstance(row, dict):
            continue
        todos = row.get("todos")
        if not isinstance(todos, dict):
            continue
        open_count = todos.get("open")
        total = todos.get("total")
        if not isinstance(open_count, int) or not isinstance(total, int):
            continue
        session_id = str(row.get("session_id") or "")
        own = bool(own_session) and session_id == own_session
        line = Text(no_wrap=True)
        line.append("own session" if own else session_id, style=resolver("name"))
        line.append(": ", style=resolver("dim"))
        line.append(f"{open_count} open / {total}", style=resolver("dim"))
        next_items = todos.get("next")
        if own and isinstance(next_items, list) and next_items:
            joined = ", ".join(f'"{str(item)}"' for item in next_items[:3])
            line.append(f" — next: {joined}", style=resolver("dim"))
        lines.append(line)
    if not lines:
        lines.append(Text("todo snapshots: none yet", style=resolver("dim"), no_wrap=True))
    return lines


#: Entries the detail page renders before one trailing ``N older updates`` row
#: (spec §7.3): the store keeps them all (cap ``UPDATES_MAX``); the page stays
#: bounded so a long history cannot make one frame unbounded work.
UPDATES_PER_PAGE = 100

#: Body lines an UNEXPANDED update entry shows before its ``N more lines``
#: marker; `↵` on the entry's stamp toggles the rest in (spec §7.3).
UPDATE_BODY_LINES = 6


def update_reporter_text(by: str) -> str:
    """The stamp's reporter clause, on the footer's own vocabulary.

    ``operator`` reads ``the operator`` and a session id reads ``session <id>``
    — the SAME two shapes :func:`detail_progress_line` paints, so one reporter
    is named one way everywhere. Anything else is an agent label and stands as
    written. ``""`` is unknown and renders nothing, so the stamp is the time
    alone (spec §7.3).
    """
    reporter = str(by or "").strip()
    if not reporter:
        return ""
    if reporter == "operator":
        return "the operator"
    if is_session_id(reporter):
        return f"session {reporter}"
    return reporter


def update_local_moment(at: str) -> datetime | None:
    """An entry's ``at`` as a LOCAL moment, or ``None`` when unparseable.

    The store writes ISO-8601 UTC; every surface that SHOWS a time renders it
    in the reader's own clock (the spec's ``local time``). A row whose stamp
    will not parse returns ``None`` so the stamp falls back to the raw text
    rather than guessing a day it cannot know.
    """
    text = str(at or "").strip()
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone()


def update_day_label(at: str, *, now: float | None = None) -> str:
    """The day-group header text: ``today`` / ``yesterday`` / ``25 Sep`` (+year).

    The two relative words are compared on the SAME local basis the stamp is
    rendered on and the store dates from (:func:`local_operator.projects._local_today`),
    so a group labelled ``today`` is the day the reader is having. A date in
    another year carries the year, because ``25 Sep`` alone would read as the
    next one (the rule :func:`format_short_date` states).
    """
    from local_operator.projects import _local_today

    moment = update_local_moment(at)
    if moment is None:
        return str(at or "").strip() or "unknown date"
    day = moment.date()
    today = _local_today() if now is None else datetime.fromtimestamp(now).date()
    if day == today:
        return "today"
    if day == today - timedelta(days=1):
        return "yesterday"
    label = f"{day.day} {day.strftime('%b')}"
    if day.year != today.year:
        label += f" {day.year}"
    return label


def update_stamp_text(
    entry: dict[str, Any], *, selected: bool = False, style_for: StyleFor | None = None
) -> Text:
    """``17:52 · session ab12cd34ef56`` — the entry's stamp line.

    The SELECTED entry states the full local date (``2026-09-27 17:52``), the
    compact form otherwise: within a day group the date is the group header's
    job, and repeating it on every row would spend the width the reporter
    needs (spec §7.3).
    """
    resolver = _styles(style_for)
    line = Text(no_wrap=True)
    at = str(entry.get("at") or "")
    moment = update_local_moment(at)
    if moment is None:
        stamp = at.strip()
    elif selected:
        stamp = moment.strftime("%Y-%m-%d %H:%M")
    else:
        stamp = moment.strftime("%H:%M")
    if stamp:
        line.append(stamp, style=resolver("name"))
    reporter = update_reporter_text(str(entry.get("by") or ""))
    if reporter:
        line.append(" · " if stamp else "", style=resolver("dim"))
        line.append(reporter, style=resolver("dim"))
    return line


def update_body_lines(text: str) -> list[str]:
    """An entry's markdown body as its own source lines (the clamp unit).

    The body renders through the transcript's rich-Markdown path, and the
    clamp counts LINES OF THE SOURCE: rendering to count wrapped lines would
    cost a Console per entry per repaint, and the spec's ``6 body lines`` is
    the entry's own shape rather than the viewport's.
    """
    return [line.rstrip() for line in str(text or "").replace("\r\n", "\n").split("\n")]


def update_body_is_clamped(text: str) -> bool:
    """Does this entry's body overflow the clamp — i.e. is there a tail to open?

    THE predicate behind the row's verb (design review round 1, D1): a stamp
    whose body fits the clamp is not expandable, so it must advertise no verb
    and hold no ``_expanded`` state — a hint for a key that does nothing is
    exactly what the page's own rule forbids.
    """
    return len(update_body_lines(text)) > UPDATE_BODY_LINES


def update_more_lines_text(
    count: int, *, expanded: bool, style_for: StyleFor | None = None
) -> Text:
    """The clamp marker: ``[4 more lines — ↵ expand]`` (``↵ collapse`` when open).

    The key is advertised only because it is bound: `↵` on the entry's stamp
    row toggles this entry, the same slot every other row verb uses, so the
    marker names a key that does what it says (the page's own rule). Two
    corrections from design review round 1: the count is INFLECTED (``1 more
    line``), and the ink is ``muted`` rather than ``dim`` (D4) — this is the
    feed's only in-content affordance and it was sharing the quiet ink of
    paths and day headers, at 4.55:1 dark and 3.77:1 light.
    """
    resolver = _styles(style_for)
    verb = "collapse" if expanded else "expand"
    noun = "line" if count == 1 else "lines"
    return Text(f"[{count} more {noun} — ↵ {verb}]", style=resolver("muted"), no_wrap=True)


def attachment_row_text(
    attachment: dict[str, Any],
    *,
    selected: bool = False,
    style_for: StyleFor | None = None,
    width: int | None = None,
    previewable: bool = False,
) -> Text:
    """``[img] board-60x20.png · 82 KB · space`` — one attachment affordance.

    A terminal cannot render the image, and this row does not pretend: it
    carries the KIND, the name and the size, and the path lives on its own
    line under it (:func:`attachment_path_text`). The kind word is the
    store's own classification, so the row cannot disagree with what was
    copied in.

    ``previewable`` appends the ``· space`` affordance to the SELECTED row
    whose copy can actually be shown (design review round 1, D1). The ladder
    cannot afford that key at the width most readers have — it costs 17 cells
    against a 96-cell footer — so without it the capability is invisible to
    anyone who has not read the spec, and the only advertised verb on that row
    (``↵ open``) hands the file to the OS opener.

    ``width`` is the MEASURED box the row renders into, and it is what keeps
    the affordance from costing a WRAP: a ``Static`` wraps what it is given
    whatever its ``no_wrap`` says (see :func:`attachment_path_text`), so the
    name is middle-ellipsized to make room for the size and the affordance —
    the file's own tail, which names it, is the end that survives.
    """
    resolver = _styles(style_for)
    kind = str(attachment.get("kind") or "data")
    name = str(attachment.get("name") or "(unnamed)")
    size = attachment.get("bytes")
    marker = "[img]" if kind == "image" else "[file]"
    tail = f" · {file_size_text(size)}" if isinstance(size, int) and size >= 0 else ""
    affordance = " · space" if previewable and selected else ""
    if width is not None:
        budget = max(width - cell_len(f"{marker} {tail}{affordance}"), 1)
        if cell_len(name) > budget:
            name = _middle_ellipsize(name, budget)
    line = Text(no_wrap=True)
    line.append(marker, style=resolver("dim"))
    line.append(f" {name}", style=resolver("name" if selected else "dim"))
    if tail:
        line.append(tail, style=resolver("dim"))
    if affordance:
        line.append(affordance, style=resolver("dim"))
    return line


def _home_relative(path: str) -> str:
    """``/Users/me/x`` → ``~/x`` — the head every row shares is the noise.

    Design review round 1, D2: at 100 columns the stored path is 126 cells (114
    with ``~``) against a 95-cell box, so the ellipsis landed in the unique hex
    tail and every row read ``…/attachments/<cut>…``.
    """
    if not path.startswith("/"):
        return path
    try:
        home = str(Path.home()).rstrip("/")
    except Exception:  # noqa: BLE001 — no home is a cosmetic loss, not a failure
        return path
    if home and path.startswith(home + "/"):
        return "~" + path[len(home) :]
    return path


def _middle_ellipsize(text: str, budget: int) -> str:
    """``head…tail`` within ``budget`` cells, keeping BOTH ends (design D2).

    The tail carries the information — ``…/eb904c…png`` names the file and its
    extension — while the head is the same on every row of the feed, so a plain
    right-truncation keeps exactly the part that distinguishes nothing.
    """
    if cell_len(text) <= budget:
        return text
    if budget <= 1:
        return "…"
    # The tail gets the larger share: it is the part that names the file.
    tail_budget = max(budget * 2 // 3, 1)
    head_budget = max(budget - tail_budget - 1, 0)
    head = ""
    used = 0
    for char in text:
        size = cell_len(char)
        if used + size > head_budget:
            break
        head += char
        used += size
    tail = ""
    used = 0
    for char in reversed(text):
        size = cell_len(char)
        if used + size > tail_budget:
            break
        tail = char + tail
        used += size
    return f"{head}…{tail}"


def attachment_path_text(
    attachment: dict[str, Any], *, width: int | None = None, style_for: StyleFor | None = None
) -> Text:
    """``→ <path>`` — always shown, with ``[missing on disk]`` when it is gone.

    The path is the one thing a reader can act on themselves (copy it, open
    it elsewhere), so it is never hidden behind a selection; the missing
    marker rides the same line, and `↵` answers with the honest sentence
    rather than a silent no-op.

    ``width`` is the MEASURED box the row renders into, and it is what makes
    the line readable: Textual WRAPS a ``Static``'s text whatever its
    ``no_wrap`` says — measured, a 113-cell path in a 95-cell box painted as
    a bare ``→`` with the rest hard-split onto the following rows. Given the
    width the path is abbreviated (``~`` for the home prefix) and
    MIDDLE-ellipsized so the file's own tail survives, with the missing
    marker's cells reserved first so the caveat never falls off the end.
    """
    resolver = _styles(style_for)
    path = _home_relative(str(attachment.get("path") or "(no path recorded)"))
    marker = "  [missing on disk]" if attachment.get("missing") else ""
    prefix = "    → "
    line = Text(no_wrap=True)
    if width is None:
        line.append(f"{prefix}{path}", style=resolver("dim"))
        if marker:
            line.append(marker, style=resolver("stale"))
        return line
    reserved = cell_len(marker) if cell_len(marker) < width else 0
    budget = max(width - reserved - cell_len(prefix), 1)
    line.append(f"{prefix}{_middle_ellipsize(path, budget)}", style=resolver("dim"))
    if reserved:
        line.append(marker, style=resolver("stale"))
    return line


def older_updates_text(count: int, *, style_for: StyleFor | None = None) -> Text:
    """``… N older updates`` — the trailing row for the render cap (spec §7.3)."""
    resolver = _styles(style_for)
    return Text(f"… {count} older updates", style=resolver("dim"), no_wrap=True)


def aggregate_footer(
    views: list[dict[str, Any]],
    *,
    tier: str | None = None,
    style_for: StyleFor | None = None,
    width: int | None = None,
) -> Text:
    """Counts for the board/timeline footer: what the page is summarising.

    ``width`` is the measured box the line renders into. Without it a line
    wider than its box was cut by the container, mid-clause and without a
    marker (design review round 1, D5: at 100×30 an "· 0" tail read as a
    count, and at 60 the live-session clause silently vanished). The ladder
    SHEDS whole segments, most-expendable first — zoom (the header and the
    hints carry it), then the live-session count (each card still says "N
    live"), then the counts themselves, ellipsized at a SEGMENT boundary
    with an explicit ``…`` so a cut line always says more exists and never
    cuts inside a clause. ``None`` keeps the full line for callers with no
    box to fit.
    """
    resolver = _styles(style_for)
    live = sum(live_count(view) for view in views)
    counts: dict[str, int] = {}
    for view in views:
        status = str(_row(view).get("status") or "active")
        counts[status] = counts.get(status, 0) + 1
    text = Text(no_wrap=True, overflow="ellipsis")
    head = f"{len(views)} project{'' if len(views) == 1 else 's'}"
    # The counts are per TRUE status, in lifecycle order: the board buckets
    # planning/qa/validation into one column, but the summary must not merge
    # them — "1 active" over a store that also holds two qa rows would be a lie
    # of omission (the same reason archived has always had its own count).
    order = [*PROJECT_STATUSES]
    parts = [f"{counts[status]} {status}" for status in order if counts.get(status)]
    joined = " · ".join(parts)
    zoom = f"zoom: {tier}" if tier else None
    rungs = [
        " · ".join(
            filter(None, [head, joined, f"{live} live session{'' if live == 1 else 's'}", zoom])
        ),
        " · ".join(filter(None, [head, joined, f"{live} live", zoom])),
        " · ".join(filter(None, [head, joined, f"{live} live"])),
        " · ".join(filter(None, [head, joined])),
    ]
    chosen: str | None = None
    for rung in rungs:
        if width is None or cell_len(rung) <= width:
            chosen = rung
            break
    if chosen is None:
        # Nothing above fits: keep the head plus as many count segments as
        # the width allows, then an explicit ``…``. A rung that fits is never
        # ellipsized; an ellipsized rung never pretends to be complete. (The
        # loop above binds the first rung whenever ``width`` is None, so
        # reaching here means a width exists — pinned for the type checker.)
        assert width is not None
        kept: list[str] = []
        for part in parts:
            candidate = " · ".join([head, *kept, part])
            if cell_len(candidate + " · …") <= width:
                kept.append(part)
            else:
                # STRICT prefix: the first segment that cannot fit ends the
                # list. Skipping it and picking a shorter later one ("1 qa ·
                # 1 paused · …", with validation silently missing) claims a
                # completeness the line does not have.
                break
        chosen = (
            " · ".join([head, *kept]) + " · …" if kept else truncate_row(head + " …", cap=width)
        )
    text.append(chosen, style=resolver("dim"))
    return text
