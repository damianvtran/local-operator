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
#: NOT a code-owned arrow. The design review asked for `←` in every locale, and
#: the Urdu audit had already decided the opposite for a reason a test pins
#: (`test_urdu_paints_the_back_control_with_the_forward_arrow_it_reads_by`): the
#: sentence is isolated RTL, so its `→` points the way an RTL reader goes BACK,
#: and U+2190 is not mirrored by bidi. The arrow therefore stays copy-owned.
EDGE_GLYPH = "┃"
#: The priority ladder in block glyphs: the desktop's bars at terminal scale.
#: THREE CELLS EVERY TIME (DT12). A lone ``▁`` plus two spaces read as an
#: underscore — an empty input caret beside the title — because the glyph has no
#: neighbour to compare with. All three levels now share the silhouette and
#: differ only in fill: 3 high, 2 mid, 3 flat low.
BARS = {"high": "▁▃▅", "normal": "▁▃▁", "low": "▁▁▁"}
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
    """Age today, ``HH:MM`` yesterday, a date beyond it (design §8.2).

    DT7: a bare clock time under ``Earlier this week`` or ``Older`` names no day
    at all, so the reader cannot tell Tuesday's row from today's. The bucket is
    derived from the age rather than passed in, because every caller has the
    same stamp and a second argument would be a second thing to get wrong.

    The date is NUMERIC (``10/23``), deliberately: a three-letter weekday or a
    CLDR short date would be English — or locale-dependent output from a process
    whose LC_TIME is not the reader's — inside copy that seven locales have
    already been audited for. Digits fit the existing five cells and read the
    same in every shipped locale.
    """
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    age_ms = now_ms - int(at_ms or 0)
    if age_ms < 12 * 3600 * 1000:
        age = compact_age(at_ms, now_ms)
        return _t(_k.row_now()) if age == "now" else age
    moment = datetime.fromtimestamp(int(at_ms) / 1000)
    if age_ms < 48 * 3600 * 1000:
        return moment.strftime("%H:%M")
    return f"{moment.month}/{moment.day}"


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
    # Counts carry information, unlike decorative rules; muted clears 4.5:1
    # in light mode where dim does not (design round 3 D7).
    header.append(f" · {count} ", style=ink("muted"))
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


def _append_folded_context(text: Text, item: dict[str, Any], width: int) -> None:
    """Same-topic updates folded into an ask (``asks/dedupe.py``), one block each.

    Under the existing "Earlier updates" divider, so the operator sees the text
    came from the agent's updates (design D4) with no new copy to translate. Each
    entry: its title as a bold line, then its body with paragraph breaks KEPT
    (a blank source line stays a blank row), one blank row between entries
    (design D2). The count includes any entries the fold's cap dropped
    (``context_dropped``, review m3), so a capped list never reads as complete.
    """
    ask = item.get("ask") or {}
    entries = [c for c in (ask.get("context") or []) if isinstance(c, dict)]
    if not entries:
        return
    total = len(entries) + int(ask.get("context_dropped") or 0)
    text.append("\n")
    text.append_text(section_header(_t(_k.section_history()), total, width))
    text.append("\n")
    for index, entry in enumerate(entries):
        if index:
            text.append("\n")
        title = str(entry.get("title") or "").strip()
        if title:
            for line in wrap_cells(title, width - 4):
                text.append("   " + line + "\n", style=ink("fg", bold=True))
        for paragraph in str(entry.get("body_md") or "").strip().splitlines():
            if not paragraph.strip():
                text.append("\n")
                continue
            for line in wrap_cells(paragraph, width - 4):
                text.append("   " + line + "\n", style=ink("fg"))


def wrap_cells(text: str, cells: int) -> list[str]:
    """Wrap ``text`` to ``cells`` terminal cells, never splitting a wide glyph.

    ``textwrap`` counts code points, so a CJK title or an Urdu line would overrun
    the box it was measured against. This walks the string and breaks on the last
    space that still fits, falling back to a hard break at a glyph boundary.
    """
    limit = max(8, cells)
    lines: list[str] = []
    rest = text
    while rest:
        if cell_len(rest) <= limit:
            lines.append(rest)
            break
        take = 0
        cut = 0
        for index in range(1, len(rest) + 1):
            width = cell_len(rest[:index])
            if width > limit:
                break
            take = index
            if rest[index - 1] == " ":
                cut = index
        if cut:
            lines.append(rest[: cut - 1])
            rest = rest[cut:]
            continue
        if take == 0:
            lines.append(rest)
            break
        lines.append(rest[:take])
        rest = rest[take:]
    return lines


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
        # DT6: ``dim`` on ``bg`` fails WCAG AA in 30 of the 54 shipped themes
        # (light 3.77:1, gruvbox-light 3.61:1). This module's own rule above
        # bans ``dim`` for source and time for exactly this reason, and a chip
        # that says WHY an item is already read is not less important than the
        # time it was read at. ``muted`` passes on ``bg`` in all 54.
        chips.append((seen, "muted"))
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
    # DT1: and not while the first read is still in flight. ``snapshot is None``
    # reaches this line as an EMPTY snapshot, whose counts are all zero, so the
    # loading frame claimed "0 unread · 0 waiting on you" at the one moment the
    # operator might decide whether to wait. Absence is not emptiness, and that
    # rule is the same one the error state already follows two lines down.
    if snapshot.loaded and not snapshot.error:
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
    focused: bool = False,
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
    # DT3: the pane carries the cursor glyph while IT owns the keys, so "the
    # arrows move this, not the list" is visible without a word of new copy.
    top.append(f"{CURSOR} " if focused else " ")
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
    if is_ask:
        # D3: the deadline lives where the PRODUCER puts it. `notifications/
        # items.py:ask_item` nests it at ``ask.deadline_at`` (from the ask row's
        # own ``expires_at``); the top-level read here was a shape the producer
        # never emits, so every real ask pane omitted its expiry while the same
        # ask named it under "Waiting on you". The two top-level keys stay as
        # fallbacks for rows a hand-built caller may pass.
        deadline = int(
            item.get("expires_at")
            or item.get("deadline_at")
            or (item.get("ask") or {}).get("deadline_at")
            or 0
        )
        if deadline:
            meta.append(_t(_k.ask_expires(time=compact_until(deadline, now_ms))))
    seen = _seen_clause(item) if item.get("read") else ""
    if seen:
        meta.append(seen)
    text.append(" · ".join(meta), style=ink("muted"))
    text.append("\n ")
    text.append(RULE_GLYPH * (width - 2), style=ink("dim"))
    text.append("\n")
    # DT4: an ask's detail repeated its own question as the body — the title IS
    # that question — and dropped the context the row already knows. The body is
    # the update's prose; an ask's context is its expiry (in the meta line) and
    # its related projects below.
    # An ask's markdown ``body_md`` (folded update text, built for the desktop
    # and phone renderers) is never drawn here: this pane is plain text and
    # would print its ``**`` literally (design D2 / QA Q2, #3094). The folded
    # entries are drawn from ``ask.context`` below instead.
    body = str(item.get("body_md") or "") if not is_ask else ""
    if body.strip():
        # DT9: a wrapped paragraph lost its inset on every continuation line,
        # because the widget folded the text with no hanging indent. Wrapping
        # here, cell-aware, keeps the shape on every line of every paragraph.
        for paragraph in body.strip().splitlines():
            for line in wrap_cells(paragraph, width - 4):
                text.append("   " + line + "\n", style=ink("fg"))
    if is_ask:
        _append_folded_context(text, item, width)
    elif not body.strip():
        # D7: "No details were attached." under an ask read like an ERROR — for
        # an ask the question IS the detail, so the line said the item was
        # broken when it was complete. An ask simply has no body row: its meta
        # line (with the expiry) sits straight above Related projects.
        text.append(" " + _t(_k.no_body()) + "\n", style=ink("muted"))
    links = item.get("links") or []
    if links:
        text.append("\n")
        text.append_text(section_header(_t(_k.section_links()), len(links), width))
        for link in links:
            # D6: the URL was appended raw, so a long one was folded by the
            # widget at column 0 — one cell LEFT of its own label's inset, which
            # read as a broken continuation. It is wrapped here, cell-aware, and
            # a URL that cannot share the label's line moves onto its own lines
            # at the block's inset (2), the body's own rule.
            label = str(link.get("label") or "")
            url = str(link.get("url") or "")
            text.append("\n  ")
            text.append(label, style=ink("fg"))
            if url and cell_len(label) + 2 + cell_len(url) <= width - 2:
                text.append("  " + url, style=ink("muted"))
            elif url:
                for line in wrap_cells(url, width - 4):
                    text.append("\n  " + line, style=ink("muted"))
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
    # An ask has no payload: its projects ride the row, which is the same list
    # the reel row was built from, so the pane and the row cannot disagree.
    projects = list((payload or {}).get("projects") or item.get("projects") or [])
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
            # DT8: the title is what CHANGED between revisions ("11 sign-ups, 9
            # blocked") and the excerpt is the first paragraph, which barely
            # moves — so five revisions painted five identical lines. The
            # excerpt stays as the fallback for a revision with no title.
            text.append(
                truncate_cells(str(entry.get("title") or entry.get("excerpt") or ""), width - 10),
                style=ink("fg"),
            )
        text.append("\n")
    if is_ask:
        # DT4: the ASK's own pane showed no gesture at all, so the one screen
        # where answering is the obvious next step never said how.
        text.append("\n ")
        # The ``a ↵`` prefix is the SAME chrome as the row's ask line above: a
        # key letter and a key glyph, not copy — the translatable word beside it
        # is `hint_answer`, which is why the escape is the file's existing one
        # rather than a new catalogue key duplicating one word in seven locales.
        text.append(
            "a ↵ " + _t(_k.hint_answer()),  # i18n: ignore key glyphs, chrome
            style=ink("muted"),
        )
        text.append("\n")
    return text


#: Which hint survives a narrow row, most valuable first (DT2).
#:
#: Widest-first was the only rule, so at 80 columns the volume gestures — the
#: ones requirement 6 exists for — were the first to go, leaving the operator to
#: already know them. The order below ranks by "how expensive is this to
#: rediscover": ``esc`` is the way out, then the row actions, then the bulk
#: gestures, then selection, and only then browsing/filters, which the detail
#: and the header both restate.
#:
#: ``z`` sits at 1, directly behind ``esc``, because D1 measured the opposite:
#: it was ranked 8, so at 80 columns the row dropped it FIRST, and an undo
#: receipt ("Dismissed 1 update.") then sat above a footer naming every key but
#: the one that reverses it — right after `E`, the most destructive gesture on
#: the page. It costs the row nothing when no undo is live: the label is only
#: emitted while the window is open.
#:
#: ``↵`` keeps rank 2, ABOVE the secondary row gestures (R4-2). The D1 re-rank that
#: put ``z`` first also pushed ``↵`` behind ``u`` and ``space``, and the 80-column
#: row then dropped the PRIMARY action — open — while keeping the enable/select
#: pair. The rule the order follows: the way out, then the undo of a destructive
#: act, then the primary action, then the row's own gestures, then volume and
#: selection, then the browse-only keys the header and the detail both restate.
_HINT_RANK = {
    "esc": 0,
    "z": 1,
    "↵": 2,
    "e": 3,
    "E": 4,
    "space": 5,
    "A": 6,
    "u": 7,
    "a": 8,
    "o": 9,
    "f": 10,
}


def hint_row(labels: Sequence[tuple[str, str]], width: int) -> list[tuple[str, str]]:
    """Drop whole hints until the row fits; ``esc`` never drops.

    The ``aggregate_footer`` rule (design §8.4): a clipped hint names a key with
    half its meaning, so a hint is shown whole or not at all. WHAT drops is the
    ranked order above, not raw width (DT2); width only breaks a tie.
    """

    def cost(entries: Sequence[tuple[str, str]]) -> int:
        return sum(cell_len(f"{k} {v}") for k, v in entries) + 2 * max(0, len(entries) - 1) + 1

    def worth(entry: tuple[str, str]) -> tuple[int, int]:
        return (_HINT_RANK.get(entry[0], 50), cell_len(f"{entry[0]} {entry[1]}"))

    entries = list(labels)
    while cost(entries) > width and len(entries) > 1:
        droppable = [i for i, (key, _) in enumerate(entries) if key != "esc"]
        if not droppable:
            break
        entries.pop(max(droppable, key=lambda i: worth(entries[i])))
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
