"""Painting for the ``/notifications`` page: rows, headers, detail, as rich Text.

PURE FUNCTIONS, NO WIDGETS. The page widget owns state and keys; everything
that turns a row dict into cells lives here, so the column arithmetic (which is
what keeps the cursor exact and the right-hand columns aligned) is testable at
any width without an app (design §8.2 column spec).

INK. Rows on ``bg`` use ``fg``/``muted``; source and time use ``muted``, never
``dim`` — ``ask_queue.py`` records ``dim`` failing contrast on raised grounds and
the selection ground is one (design §8.2). The selected row's meta goes one rung
brighter (``fg``) for the schedule page's measured reason (``_meta_ink``).
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text

from local_operator.i18n import keys as i18n_keys
from local_operator.notifications.items import first_question as inbox_first_question
from local_operator.resume import format_age
from local_operator.tui import notifications_model as model
from local_operator.tui import theme as theme_mod
from local_operator.tui.i18n_status import display_tr
from local_operator.tui.widgets.tool_card import truncate_cells

_t = display_tr(
    "tui.notifications",
    # The literal /commands ride as islands (see model.SHELL_COMMANDS):
    # per-string isolation alone reorders them in Urdu.
    technical_literals=model.SHELL_COMMANDS,
    technical_commands=model.SHELL_COMMAND_TEMPLATES,
)
_k = i18n_keys.tui_notifications

#: Chrome, not copy: glyphs no locale translates (the schedule page's rule).
CURSOR = "▸"
UNREAD = "•"
READ_ASK = "·"
SELECTED = "◼"
ASK_GLYPH = "?"
RULE_GLYPH = "─"
EDGE_GLYPH = "┃"
#: The priority ladder in block glyphs: the desktop's bars at terminal scale.
BARS = {"high": "▁▃▅", "normal": "▁▃ ", "low": "▁  "}
#: Right-hand columns (design §8.2): source ellipsised at 12 cells, time 5.
SOURCE_CELLS = 12
TIME_CELLS = 5
#: Cursor, space, unread, space, kind (3), space — before the title: 8 cells.
#: (It read 7 once, and every row overran its width by one cell, which the
#: exact-width test caught at 40, 76 and 120 columns.)
LEAD_CELLS = 8


def ink(token: str, *, ground: Style | None = None, bold: bool = False) -> Style:
    try:
        color = theme_mod.semantic_color(token)
    except KeyError:  # pragma: no cover - the ramp carries every token used
        color = None
    return Style(color=color, bgcolor=ground.bgcolor if ground else None, bold=bold)


def selection_ground() -> Style:
    """The app's one selection ground (sidebar, pickers, ToolCard)."""
    try:
        return Style(bgcolor=theme_mod.semantic_color("tint-select"))
    except KeyError:  # pragma: no cover
        return Style()


def compact_age(at_ms: int, now_ms: int | None = None) -> str:
    """``4m`` / ``3h`` / ``2d`` — the sidebar's age idiom, suffix dropped."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    seconds = max(0.0, (now_ms - int(at_ms or 0)) / 1000)
    if seconds < 60:
        # A sentinel, never painted as-is: callers render ``just_now`` copy.
        return "now"
    return format_age(seconds).replace(" ago", "")


def compact_until(at_ms: int, now_ms: int | None = None) -> str:
    """``38m`` / ``2h`` until a deadline: :func:`compact_age` read forwards."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return compact_age(now_ms, int(at_ms or 0))


def row_time(at_ms: int, now_ms: int | None = None) -> str:
    """Age today; ``HH:MM`` for older rows (the yesterday group in §8.2)."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    if now_ms - int(at_ms or 0) < 12 * 3600 * 1000:
        age = compact_age(at_ms, now_ms)
        return _t(_k.row_now()) if age == "now" else age
    return datetime.fromtimestamp(int(at_ms) / 1000).strftime("%H:%M")


def group_label(kind: str, value: str) -> str:
    """The section header label for one group (see ``model.group``)."""
    if kind == "waiting":
        return _t(_k.group_waiting())
    if kind == "conversation":
        return value
    return {
        "today": _t(_k.group_today()),
        "yesterday": _t(_k.group_yesterday()),
        "week": _t(_k.group_week()),
        "older": _t(_k.group_older()),
    }.get(value, value)


def section_header(label: str, count: int, width: int) -> Text:
    """``── label · n ──…``, the projects page's divider (not selectable)."""
    header = Text(no_wrap=True)
    header.append(" ── ", style=ink("dim"))
    header.append(label, style=ink("muted", bold=True))
    header.append(f" · {count} ", style=ink("dim"))
    pad = width - cell_len(header.plain)
    if pad > 0:
        header.append(RULE_GLYPH * pad, style=ink("dim"))
    return header


def _kind_cells(row: dict[str, Any]) -> tuple[str, str]:
    """The 3-cell kind/priority column and its ink."""
    if row.get("source") == "ask":
        return (f"{ASK_GLYPH}  ", "signal")
    return (BARS.get(str(row.get("priority") or "normal"), BARS["normal"]), "muted")


def _seen_clause(row: dict[str, Any]) -> str:
    via = row.get("read_via")
    if via == "watching":
        return _t(_k.seen_live())
    if via == "in_view":
        return _t(_k.seen_in_chat())
    return ""


def _pad_cells(text: str, cells: int) -> str:
    """Right-align ``text`` in ``cells`` terminal cells (never truncating).

    The complement of ``truncate_cells``: a cell budget needs cell-aware
    padding, because a wide character occupies two columns and a combining
    mark none.
    """
    return " " * max(0, cells - cell_len(text)) + text


def row_line(
    row: dict[str, Any],
    width: int,
    *,
    cursor: bool,
    selected: bool,
    now_ms: int | None = None,
) -> Text:
    """One list row at ``width`` cells, exactly (design §8.2 columns)."""
    ground = selection_ground() if (cursor or selected) else None
    line = Text(no_wrap=True, overflow="crop")
    line.append(CURSOR if cursor else " ", style=ink("accent", ground=ground))
    line.append(" ", style=ink("fg", ground=ground))
    if selected:
        line.append(SELECTED, style=ink("accent", ground=ground))
    elif not row.get("read"):
        line.append(UNREAD, style=ink("accent", ground=ground))
    elif row.get("source") == "ask":
        line.append(READ_ASK, style=ink("muted", ground=ground))
    else:
        line.append(" ", style=ink("fg", ground=ground))
    line.append(" ", style=ink("fg", ground=ground))
    kind, kind_ink = _kind_cells(row)
    line.append(kind, style=ink(kind_ink, ground=ground))
    line.append(" ", style=ink("fg", ground=ground))
    # Right side first, so the title takes exactly what is left.
    source = truncate_cells(model.source_name(row), SOURCE_CELLS)
    # CELL budgets, CELL padding (zh-CN audit, and reachable in English): both
    # fields are sized in terminal cells, so `str.rjust` — which counts
    # characters — overran the width it had just reserved for any non-ASCII
    # stamp or conversation name. Measured at a 78-cell list in zh-CN: 80 cells
    # for a just-now row with an ASCII title, 86 with a Chinese one; and 82 in
    # ENGLISH with a Chinese title, because the source field was the other half.
    stamp = _pad_cells(row_time(int(row.get("updated_at") or 0), now_ms), TIME_CELLS)
    chips: list[tuple[str, str]] = []
    if row.get("priority") == "high" and row.get("source") != "ask":
        chips.append((_t(_k.chip_high()), "warning"))
    repeat = int(row.get("repeat_count") or 1)
    if repeat > 1:
        chips.append((f"×{repeat}", "muted"))
    seen = _seen_clause(row) if row.get("read") else ""
    if seen:
        chips.append((seen, "dim" if not ground else "muted"))
    right_cells = 1 + SOURCE_CELLS + 1 + TIME_CELLS
    chip_cells = sum(cell_len(text) + 1 for text, _ in chips)
    title_cells = max(4, width - LEAD_CELLS - right_cells - chip_cells)
    if title_cells == 4 and chips:
        # Too narrow for chips and a readable title: the title wins.
        chips, chip_cells = [], 0
        title_cells = max(4, width - LEAD_CELLS - right_cells)
    title = truncate_cells(str(row.get("title") or ""), title_cells)
    title_ink = "fg" if (not row.get("read") or cursor) else "muted"
    line.append(title, style=ink(title_ink, ground=ground, bold=cursor))
    line.append(" " * (title_cells - cell_len(title)), style=ink("fg", ground=ground))
    for text, token in chips:
        line.append(" ", style=ink("fg", ground=ground))
        line.append(text, style=ink(token, ground=ground))
    line.append(" ", style=ink("fg", ground=ground))
    line.append(
        _pad_cells(source, SOURCE_CELLS), style=ink("fg" if cursor else "muted", ground=ground)
    )
    line.append(" ", style=ink("fg", ground=ground))
    line.append(stamp, style=ink("fg" if cursor else "muted", ground=ground))
    pad = width - cell_len(line.plain)
    if pad > 0:
        line.append(" " * pad, style=ink("fg", ground=ground))
    return line


def header_line(
    snapshot: model.Snapshot,
    filters: model.Filters,
    width: int,
    *,
    selected: int = 0,
    selection_hint: str = "",
) -> Text:
    """The page's first row: title and counts, or the selection summary.

    The right edge names the active filters (``unread │ time``) and drops whole
    segments, never clipped ones, when the row is narrow (design §8.2).
    """
    text = Text(no_wrap=True, overflow="crop")
    text.append(" ")
    if selected:
        text.append(_t(_k.selected(count=str(selected))), style=ink("accent", bold=True))
        if selection_hint:
            text.append(" · " + selection_hint, style=ink("muted"))
        return text
    text.append(_t(_k.title()), style=ink("fg", bold=True))
    # T6: an ERRORED snapshot has no counts to substantiate. The page states its
    # own rule elsewhere ("absence is not emptiness: paint nothing rather than a
    # 0"), and "0 unread · 0 waiting on you" above "database disk image is
    # malformed" is that rule contradicted one row up.
    if not snapshot.error:
        unread = snapshot.unread_count + snapshot.waiting_unread_count
        text.append(
            " · " + _t(_k.header_counts(unread=unread, waiting=snapshot.waiting_count)),
            style=ink("muted"),
        )
    if not snapshot.complete:
        partial = _t(_k.header_partial(count=str(len(snapshot.items))))
        text.append(" · " + partial, style=ink("muted"))
    segments = [
        _t(_k.filter_unread()) if filters.read == "unread" else _t(_k.filter_all()),
    ]
    if filters.kind == "asks":
        segments.append(_t(_k.kind_asks()))
    elif filters.kind == "updates":
        segments.append(_t(_k.kind_updates()))
    if filters.priority == "high":
        segments.append(_t(_k.priority_high()))
    elif filters.priority == "normal_up":
        segments.append(_t(_k.priority_normal_up()))
    segments.append(
        _t(_k.group_time()) if filters.grouping == "time" else _t(_k.group_conversation())
    )
    while segments:
        edge = " │ ".join(segments) + " "
        room = width - cell_len(text.plain) - 2
        if cell_len(edge) <= room:
            text.append(" " * (width - cell_len(text.plain) - cell_len(edge)))
            text.append(edge, style=ink("muted"))
            break
        # Drop the widest non-read segment first; the read filter goes last.
        if len(segments) == 1:
            break
        segments.pop(max(range(1, len(segments)), key=lambda i: cell_len(segments[i])))
    return text


def empty_text(snapshot: model.Snapshot, filters: model.Filters) -> Text:
    """Empty, caught-up and filtered states (design §13 copy)."""
    text = Text()
    if snapshot.error:
        text.append(" " + _t(_k.error_load(error=snapshot.error)), style=ink("warning"))
        return text
    filtered = filters.kind != "all" or filters.priority != "any" or bool(filters.session_id)
    if filtered:
        text.append(" " + _t(_k.empty_filtered_title()), style=ink("fg", bold=True))
        return text
    # "Caught up" claims the operator WORKED THROUGH something; a store whose
    # high-water mark is still 0 has never held an item or a read mark, so the
    # honest sentence there is "nothing yet" (measured on the first empty
    # capture, which said "All caught up" over a store nothing had touched).
    if filters.read == "unread" and snapshot.high_water > 0:
        text.append(" " + _t(_k.empty_caught_up_title()), style=ink("fg", bold=True))
        text.append("\n\n ")
        text.append(_t(_k.empty_caught_up_body()), style=ink("muted"))
        return text
    text.append(" " + _t(_k.empty_none_title()), style=ink("fg", bold=True))
    text.append("\n\n ")
    text.append(_t(_k.empty_none_body()), style=ink("muted"))
    return text


def list_text(
    groups: Sequence[model.Group],
    width: int,
    *,
    cursor_id: str,
    selected_ids: set[str],
    now_ms: int | None = None,
) -> tuple[Text, dict[str, int]]:
    """The grouped list and each row's line offset (for scroll-to-cursor)."""
    text = Text(no_wrap=True, overflow="crop")
    offsets: dict[str, int] = {}
    line = 0
    for index, grp in enumerate(groups):
        if index:
            text.append("\n")
        text.append_text(section_header(grp.label, len(grp.rows), width))
        line += 1
        for row in grp.rows:
            text.append("\n")
            offsets[str(row["id"])] = line
            text.append_text(
                row_line(
                    row,
                    width,
                    cursor=str(row["id"]) == cursor_id,
                    selected=str(row["id"]) in selected_ids,
                    now_ms=now_ms,
                )
            )
            line += 1
    return text, offsets


def detail_text(
    payload: dict[str, Any] | None,
    row: dict[str, Any],
    width: int,
    *,
    position: tuple[int, int],
    now_ms: int | None = None,
    show_back: bool = True,
) -> Text:
    """One item's detail (design §8.2 "Detail at 80 columns").

    ``payload`` is ``notifications.items.detail``'s answer for an update, or
    None for an ask (whose own row carries its question) or a detail not read
    yet. Sections appear only when they have something to say.
    """
    width = max(20, width)
    item = (payload or {}).get("item") or row
    text = Text(no_wrap=False)
    is_ask = item.get("source") == "ask"
    kind_word = _t(_k.ask_kind()) if is_ask else _t(_k.update_kind())
    priority = str(item.get("priority") or "normal")
    right: list[str] = []
    if not is_ask:
        # BARS, and the LOCALIZED chip only for `high` — never the raw data
        # token (vi audit). `… + " " + priority` printed the English `normal`
        # inside every locale's header; the list row's own column is three
        # cell-bars with no word (`_kind_cells`), so the header follows it, and
        # the frozen namespace needs no key for the default level.
        cells = BARS.get(priority, BARS["normal"]).rstrip()
        if priority == "high":
            cells = cells + " " + _t(_k.chip_high())
        right.append(cells)
    right.append(kind_word)
    right.append(_t(_k.position(index=str(position[0]), total=str(position[1]))))
    top = Text(no_wrap=True, overflow="crop")
    top.append(" ")
    if show_back:
        top.append(_t(_k.back()), style=ink("muted"))
    tail = " · ".join(right) + " "
    gap = width - cell_len(top.plain) - cell_len(tail)
    top.append(" " * max(1, gap))
    top.append(tail, style=ink("warning" if priority == "high" and not is_ask else "muted"))
    text.append_text(top)
    text.append("\n ")
    text.append(str(item.get("title") or ""), style=ink("fg", bold=True))
    text.append("\n ")
    meta = [model.source_name(item) or str((item.get("session") or {}).get("id") or "")]
    # "now", not "now ago": under a minute the age IS the sentence.
    age = compact_age(int(item.get("updated_at") or 0), now_ms)
    meta.append(_t(_k.just_now()) if age == "now" else _t(_k.ago(age=age)))
    history = list((payload or {}).get("history") or [])
    if int(item.get("repeat_count") or 1) > 1 and history:
        first = min(int(h.get("at") or 0) for h in history)
        meta.append(
            _t(
                _k.updates_since(
                    count=int(item.get("repeat_count") or 1),
                    time=datetime.fromtimestamp(first / 1000).strftime("%H:%M"),
                )
            )
        )
    seen = _seen_clause(item) if item.get("read") else ""
    if seen:
        meta.append(seen)
    text.append(" · ".join(meta), style=ink("muted"))
    text.append("\n ")
    text.append(RULE_GLYPH * (width - 2), style=ink("dim"))
    text.append("\n")
    body = str(item.get("body_md") or "") if not is_ask else str(item.get("excerpt") or "")
    if body.strip():
        for paragraph in body.strip().splitlines():
            text.append(" " + paragraph + "\n", style=ink("fg"))
    else:
        text.append(" " + _t(_k.no_body()) + "\n", style=ink("muted"))
    links = item.get("links") or []
    if links:
        text.append("\n")
        text.append_text(section_header(_t(_k.section_links()), len(links), width))
        for link in links:
            text.append("\n  ")
            text.append(str(link.get("label") or ""), style=ink("fg"))
            text.append("  " + str(link.get("url") or ""), style=ink("muted"))
        text.append("\n")
    asks = [a for a in (payload or {}).get("asks") or [] if a.get("status", "open") == "open"]
    if asks:
        text.append("\n")
        text.append_text(section_header(_t(_k.section_waiting_here()), len(asks), width))
        for ask in asks:
            text.append("\n  ")
            text.append(ASK_GLYPH + " ", style=ink("signal"))
            # The detail's ``asks`` are the ask store's own records, so the
            # question is read the way ``items`` reads it for a reel row — one
            # reader, never a second guess at the record's shape.
            question = inbox_first_question(ask)
            deadline = int(ask.get("expires_at") or ask.get("deadline_at") or 0)
            tail_text = ""
            if deadline:
                tail_text = _t(_k.ask_expires(time=compact_until(deadline, now_ms)))
            # Indent (2) + glyph (2) + the two-space seams (4) + the ``a ↵``
            # hint (3): the question takes exactly what is left, so the hint
            # never wraps onto its own row (measured at 140x32, where it did).
            room = max(8, width - 2 - 2 - 4 - 3 - cell_len(tail_text) - (2 if tail_text else 0))
            text.append(truncate_cells(question, room), style=ink("fg"))
            if tail_text:
                text.append("  " + tail_text, style=ink("muted"))
            text.append("  a ↵", style=ink("muted"))  # i18n: ignore key glyphs, chrome
        text.append("\n")
    projects = list((payload or {}).get("projects") or [])
    if projects:
        text.append("\n")
        text.append_text(section_header(_t(_k.section_projects()), len(projects), width))
        for project in projects:
            text.append("\n  ")
            text.append(str(project.get("title") or project.get("name") or ""), style=ink("fg"))
            status = str(project.get("status") or "")
            if status:
                text.append(" · " + status, style=ink("muted"))
        text.append("\n")
    earlier = [h for h in history if int(h.get("revision") or 0) < int(item.get("revision") or 0)]
    if earlier:
        text.append("\n")
        text.append_text(section_header(_t(_k.section_history()), len(earlier), width))
        for entry in sorted(earlier, key=lambda h: -int(h.get("revision") or 0)):
            text.append("\n  ")
            stamp = datetime.fromtimestamp(int(entry.get("at") or 0) / 1000).strftime("%H:%M")
            text.append(stamp + "  ", style=ink("muted"))
            text.append(
                truncate_cells(str(entry.get("excerpt") or entry.get("title") or ""), width - 10),
                style=ink("fg"),
            )
        text.append("\n")
    return text


def hint_row(labels: Sequence[tuple[str, str]], width: int) -> list[tuple[str, str]]:
    """Drop whole hints widest-first until the row fits; ``esc`` never drops.

    The ``aggregate_footer`` rule (design §8.4): a clipped hint names a key with
    half its meaning, so a hint is shown whole or not at all.
    """

    def cost(entries: Sequence[tuple[str, str]]) -> int:
        return sum(cell_len(f"{k} {v}") for k, v in entries) + 2 * max(0, len(entries) - 1) + 1

    entries = list(labels)
    while cost(entries) > width and len(entries) > 1:
        droppable = [i for i, (key, _) in enumerate(entries) if key != "esc"]
        if not droppable:
            break
        entries.pop(max(droppable, key=lambda i: cell_len(f"{entries[i][0]} {entries[i][1]}")))
    return entries


def hints_text(entries: Sequence[tuple[str, str]]) -> Text:
    text = Text(no_wrap=True, overflow="crop")
    text.append(" ")
    for index, (key, label) in enumerate(entries):
        if index:
            text.append("  ")
        text.append(key, style=ink("fg", bold=True))
        text.append(" " + label, style=ink("muted"))
    return text


#: Satisfies the "labels" callback ``model.group`` takes.
GroupLabel = Callable[[str, str], str]
