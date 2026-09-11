"""PROTOTYPE — THROWAWAY. Variant screens for the redesigned `/resume` picker.

Design and measurements: `~/workspace/PROPOSAL-resume-picker.md` §"The variants".

Three variants of the /resume picker, switchable via --variant and ctrl+t, against
the real read-only session store.

Deliberately NOT built on `SessionPickerScreen`: inheriting it drags in latched
columns, live-state refresh, a spinner and mouse hit-testing, and every variant
would then fight the exact 74-column / 10-row caps this prototype exists to
question. The three screens share no layout base class ON PURPOSE — a shared
Layout would converge them and defeat the comparison. Duplication here is the
feature.

This file answers a design question and is then deleted. It authorizes no
production change.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from local_operator.resume import SessionRow, format_age
from local_operator.session.search_index import search_digests, soft_search_digests
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.prototype_resume_data import PreviewData
from local_operator.tui.widgets.session_picker import filter_rows, matched_in_body, rank_rows
from local_operator.tui.widgets.tool_card import truncate_cells

#: Role gutters for the preview. Proposal §"Variant A".
GUTTER = {"user": "▸ you", "assistant": "▪ lop"}

#: Printed for every unknown metadata field. The column is RESERVED whether or
#: not the value exists: `render_rows` (session_picker.py:662) documents at
#: length why a column that appears and disappears — moving names sideways — is
#: worse than an empty one. ~19% of rows have no checkpoint.
UNKNOWN = "·"


def _ink() -> dict[str, str]:
    """Semantic colours, resolved once per paint."""
    return {
        name: theme_mod.semantic_color(name)
        for name in ("fg", "muted", "dim", "accent", "success", "warning", "faint")
    }


def _short_model(checkpoint: dict[str, Any]) -> str:
    """`anthropic/claude-opus-5` → `claude-opus-5`, or the unknown mark."""
    model = (checkpoint.get("effective_model") or {}).get("model_id") or ""
    return model.rsplit("/", 1)[-1] if model else UNKNOWN


def _short_cwd(checkpoint: dict[str, Any]) -> str:
    """`/Users/x/workspace` → `~/workspace`, or the unknown mark."""
    cwd = checkpoint.get("cwd") or ""
    if not cwd:
        return UNKNOWN
    home = str(Path.home())
    return "~" + cwd[len(home) :] if cwd.startswith(home) else cwd


def _highlight(text: str, query: str, base: str, hit: str) -> Text:
    """`text` with every case-insensitive run of `query` lifted to `hit`."""
    out = Text(text, style=base)
    needle = query.strip().lower()
    if not needle:
        return out
    hay = text.lower()
    start = 0
    while True:
        found = hay.find(needle, start)
        if found < 0:
            return out
        out.stylize(f"bold {hit}", found, found + len(needle))
        start = found + len(needle)


class _Filtering:
    """Query state shared by VALUE, not by inheritance of any layout.

    Only the filter mechanics live here — the thing all three variants are
    required to keep identical (`session_picker.py:1122`: printable keys type
    into the filter, every new affordance is a chord). Layout, row rendering
    and the preview are deliberately NOT here: the variants must stay free to
    throw out the layout entirely, which is the point of the prototype.
    """

    def _init_filter(self, rows: list[SessionRow], digests: dict[str, str]) -> None:
        self._all_rows = rows
        self._digests = digests
        self._query = ""
        #: Exact body hits get a grep context line; soft-only hits get `~` and
        #: no context. Kept as two sets so C can tell them apart.
        self._exact: set[str] = set()
        self._soft: set[str] = set()
        self._rows = list(rows)
        self._cursor = 0
        self._top = 0

    def _apply_query(self) -> None:
        """Recompute membership and order. Called ONLY when the query CHANGES.

        `filter_rows` is a pure membership filter that preserves order, so for
        a fixed query nothing ever moves under the cursor. `rank_rows` re-homes
        the cursor to 0 and is therefore applied here and nowhere else.
        """
        query = self._query.strip()
        if query:
            self._exact = search_digests(self._digests, query)
            self._soft = soft_search_digests(self._digests, query) - self._exact
            body = self._exact | self._soft
            self._rows = rank_rows(filter_rows(self._all_rows, query, body), query, body)
        else:
            self._exact, self._soft = set(), set()
            self._rows = list(self._all_rows)
        self._cursor = 0
        self._top = 0

    def _body_matched(self, row: SessionRow) -> bool:
        return matched_in_body(row, self._query, self._exact | self._soft)

    def _move(self, delta: int) -> None:
        if self._rows:
            self._cursor = max(0, min(len(self._rows) - 1, self._cursor + delta))
        self.refresh_view()  # type: ignore[attr-defined]

    def _scroll_into_window(self, drawn: int) -> None:
        """Keep the cursor inside the DRAWN window.

        Proposal Risk #1: `_page_rows` exists because a cursor once sat on an
        undrawn row and Enter resumed an invisible session. Every variant here
        uncaps the height, so the clamp has to be re-applied on every resize
        and every filter change, not just on cursor moves.
        """
        if drawn <= 0:
            self._top = 0
            return
        self._top = max(0, min(self._top, max(0, len(self._rows) - drawn)))
        if self._cursor < self._top:
            self._top = self._cursor
        elif self._cursor >= self._top + drawn:
            self._top = self._cursor - drawn + 1

    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Printable keys type into the filter; backspace deletes.

        Matches `session_picker.on_key` (:1122) — the filter accepts every
        character, so a binding per key would be a table that still missed one.
        """
        char = event.character
        if char is not None and char.isprintable() and len(char) == 1:
            event.stop()
            event.prevent_default()
            self._query += char
            self._apply_query()
            self.refresh_view()  # type: ignore[attr-defined]

    def action_backspace(self) -> None:
        if self._query:
            self._query = self._query[:-1]
            self._apply_query()
            self.refresh_view()  # type: ignore[attr-defined]

    def action_select(self) -> None:
        """`enter` PRINTS the chosen id and exits. It never resumes.

        The read-only guarantee made mechanical: this prototype has no route
        that could resume or mutate a session.
        """
        if self._rows:
            self.app.exit(self._rows[self._cursor].id)  # type: ignore[attr-defined]

    def action_cancel(self) -> None:
        self.app.exit(None)  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════
# Variant A — Telescope. Full-screen two-pane; the bet is content over labels.
# ══════════════════════════════════════════════════════════════════════════


class TelescopeScreen(_Filtering, Screen[None]):
    """Full-screen `Horizontal(results 2fr, preview 3fr)` with a live preview.

    Layout follows `settings_view.py:894`: the preview is `can_focus = False`
    so arrows never silently move focus off the list (UX round 3, U19).

    The row is DELIBERATELY today's row — cursor, name, age, id — so that A
    isolates the PANE as the single variable against B.
    """

    DEFAULT_CSS = """
    TelescopeScreen { layout: vertical; background: $surface; }
    TelescopeScreen #cols { height: 1fr; }
    TelescopeScreen #results { width: 2fr; padding: 0 1; }
    TelescopeScreen #preview { width: 3fr; padding: 0 1; border-left: solid $panel; }
    TelescopeScreen #filter { height: 1; padding: 0 1; background: $panel; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "close", show=False),
        Binding("enter", "select", "pick", show=False),
        Binding("up", "move(-1)", "up", show=False),
        Binding("down", "move(1)", "down", show=False),
        Binding("ctrl+p", "move(-1)", "up", show=False),
        Binding("ctrl+n", "move(1)", "down", show=False),
        Binding("backspace", "backspace", "delete", show=False),
        # `ctrl+e` is this codebase's established reveal chord (ask_picker.py:559,
        # keymap.py:213) — reusing it adds a shortcut, not a second vocabulary.
        Binding("ctrl+e", "toggle_verbose", "verbose", show=False),
        Binding("ctrl+u", "pane_scroll(-1)", "pane up", show=False),
        Binding("ctrl+d", "pane_scroll(1)", "pane down", show=False),
        Binding("ctrl+g", "pane_end", "newest", show=False),
    ]

    def __init__(
        self,
        rows: list[SessionRow],
        now: float,
        digests: dict[str, str],
        data: PreviewData,
    ) -> None:
        super().__init__()
        self._init_filter(rows, digests)
        self._now = now
        self._data = data
        #: Sticky for the LIFETIME of the open picker, not per row — a mode
        #: that resets as the cursor moves is a mode the user re-sets on every
        #: row (proposal §"The condensed / verbose toggle").
        self._verbose = False
        self._pane_top = 0
        self._drawn = 0

    def compose(self) -> ComposeResult:
        self._results = Static(id="results")
        self._preview = Static(id="preview")
        self._preview.can_focus = False
        self._filter = Static(id="filter")
        with Horizontal(id="cols"):
            yield self._results
            yield self._preview
        yield self._filter

    def on_mount(self) -> None:
        self.refresh_view()

    def on_resize(self, _event) -> None:  # type: ignore[no-untyped-def]
        self.refresh_view()

    def visible_rows(self) -> tuple[int, int]:
        return self._drawn, len(self._rows)

    def _budget(self) -> int:
        """Rows the list can draw: `int(height * 0.9)` with NO PAGE_ROWS_MAX.

        That 10-row ceiling is the single largest defect this prototype
        attacks — measured, production draws 10 rows at every terminal height.
        """
        height = self.app.size.height
        return max(1, int(height * 0.9) - 3)

    def action_move(self, delta: int) -> None:
        self._move(delta)

    def action_toggle_verbose(self) -> None:
        self._verbose = not self._verbose
        self._pane_top = 0
        self.refresh_view()

    def action_pane_scroll(self, direction: int) -> None:
        """Scroll the pane WITHOUT moving the list cursor."""
        self._pane_top = max(0, self._pane_top + direction * 5)
        self.refresh_view()

    def action_pane_end(self) -> None:
        self._pane_top = max(0, len(self._pane_lines()) - self._pane_height())
        self.refresh_view()

    def _pane_height(self) -> int:
        return max(1, int(self.app.size.height * 0.9) - 5)

    def _pane_lines(self) -> list[tuple[str, str]]:
        """`(role, line)` for the current row, oldest-first from the TOP.

        The top of the conversation is what the session is ABOUT — and the
        recon's "first message is a huge injected brief" is false of the rows
        a preview ever shows (6 of 139; proposal correction #3).
        """
        if not self._rows:
            return []
        sid = self._rows[self._cursor].id
        turns = self._data.verbose(sid) if self._verbose else self._data.condensed(sid)
        width = max(20, int(self.app.size.width * 0.6) - 6)
        out: list[tuple[str, str]] = []
        for role, text, _ts in turns:
            out.append(("gutter", GUTTER.get(role, f"▪ {role}")))
            for para in text.splitlines():
                if not para.strip():
                    continue
                # Hard character wrap, not cell-exact: a prototype pane only
                # has to be readable, and `textwrap` on CJK would need the
                # cell model `truncate_cells` carries.
                for start in range(0, len(para), width):
                    out.append((role, para[start : start + width]))
            out.append(("blank", ""))
        return out

    def refresh_view(self) -> None:
        ink = _ink()
        drawn = self._budget()
        self._scroll_into_window(drawn)
        page = self._rows[self._top : self._top + drawn]
        self._drawn = len(page)

        # The real pane width, not a fraction of the app: `2fr` of the split
        # is what the row actually gets, and guessing it wrapped every row onto
        # a second line at 80 cols.
        width = max(12, (self._results.size.width or int(self.app.size.width * 0.4)) - 2)
        # The id is the first thing shed when the pane is narrow. A wrapped row
        # breaks the one-row-per-session contract the cursor arithmetic assumes.
        show_id = width >= 52
        lines: list[Text] = []
        for offset, row in enumerate(page):
            selected = self._top + offset == self._cursor
            line = Text()
            line.append("❯ " if selected else "  ", style=ink["accent"])
            age = format_age(max(0.0, self._now - row.mtime))
            budget = width - len(age) - (16 if show_id else 4)
            name = truncate_cells(row.name or row.id, max(8, budget))
            line.append(name, style=f"bold {ink['fg']}" if selected else ink["fg"])
            line.append(f"  {age}", style=ink["muted"])
            if show_id:
                line.append(f"  {row.id}", style=ink["dim"])
            lines.append(line)
        self._results.update(Text("\n").join(lines) if lines else Text("no sessions"))

        self._preview.update(self._render_pane(ink))

        bar = Text()
        bar.append(" / ", style=ink["accent"])
        bar.append(self._query or "type to filter", style=ink["fg"] if self._query else ink["dim"])
        bar.append(
            f"   {self._drawn} drawn · {len(self._rows)} matching · {len(self._all_rows)} total",
            style=ink["muted"],
        )
        bar.append(
            f"   ctrl+e {'verbose' if self._verbose else 'condensed'} · ctrl+u/d scroll",
            style=ink["dim"],
        )
        self._filter.update(bar)

    def _render_pane(self, ink: dict[str, str]) -> Text:
        if not self._rows:
            return Text("no session", style=ink["dim"])
        row = self._rows[self._cursor]
        checkpoint = self._data.checkpoint(row.id)
        created = self._data.created_at(row.id)
        out = Text()
        out.append(f"{row.name or row.id}\n", style=f"bold {ink['fg']}")
        # `started X ago` beside `last worked Y ago`: the prototype reads real
        # mtimes, so the 19 mislabelled rows are IN it — the review has to be
        # able to see the discrepancy rather than absorb it (Risk #6).
        started = format_age(max(0.0, self._now - created)) if created else UNKNOWN
        worked = format_age(max(0.0, self._now - row.mtime))
        out.append(f"started {started} · last worked {worked}\n", style=ink["muted"])
        out.append(
            f"{_short_model(checkpoint)} · {_short_cwd(checkpoint)}\n", style=ink["dim"]
        )
        out.append("─" * max(10, int(self.app.size.width * 0.6) - 4) + "\n", style=ink["faint"])

        lines = self._pane_lines()
        height = self._pane_height()
        self._pane_top = max(0, min(self._pane_top, max(0, len(lines) - height)))
        if not lines:
            out.append("(no prose in this transcript)", style=ink["dim"])
            return out
        for kind, text in lines[self._pane_top : self._pane_top + height]:
            if kind == "gutter":
                style = ink["accent"] if text.endswith("you") else ink["success"]
                out.append(f"{text}\n", style=f"bold {style}")
            elif kind == "blank":
                out.append("\n")
            else:
                out.append(f"  {text}\n", style=ink["fg"])
        return out


# ══════════════════════════════════════════════════════════════════════════
# Variant B — Ledger. Centred CARD (deliberately not full-screen), no preview.
# ══════════════════════════════════════════════════════════════════════════


class LedgerScreen(_Filtering, Screen[None]):
    """Two lines per row, rich metadata, NO preview pane — and still a card.

    B is the CONTROL that tests whether "bigger" is what is actually needed,
    so it stays a centred card while A and C take the screen. The row CAP goes
    (no PAGE_ROWS_MAX) but the card does not become full-screen.

    Line 2 is reserved UNCONDITIONALLY with `·` for unknowns. ~19% of rows
    carry no checkpoint, and ragged metadata is the risk most likely to sink B
    on looks rather than on the idea.
    """

    DEFAULT_CSS = """
    LedgerScreen { align: center middle; background: $surface; }
    LedgerScreen #card {
        width: 100%; max-width: 86; max-height: 80%; border: round $panel;
        background: $boost; padding: 0 1;
    }
    /* The list takes what is left AFTER the footer is reserved. `height: auto`
       let the list eat the card and push the footer out of the border — the
       footer is the only statement of the filter and the row counts. */
    LedgerScreen #list { height: 1fr; }
    LedgerScreen #foot { height: 1; dock: bottom; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "close", show=False),
        Binding("enter", "select", "pick", show=False),
        Binding("up", "move(-1)", "up", show=False),
        Binding("down", "move(1)", "down", show=False),
        Binding("ctrl+p", "move(-1)", "up", show=False),
        Binding("ctrl+n", "move(1)", "down", show=False),
        Binding("backspace", "backspace", "delete", show=False),
        # No further chords. B's strongest claim is that it adds no modes.
    ]

    def __init__(
        self,
        rows: list[SessionRow],
        now: float,
        digests: dict[str, str],
        data: PreviewData,
    ) -> None:
        super().__init__()
        self._init_filter(rows, digests)
        self._now = now
        self._data = data
        self._drawn = 0
        #: Drawn rows that had NO checkpoint, so the run can report how ragged
        #: line 2 really got.
        self.missing_checkpoints = 0

    def compose(self) -> ComposeResult:
        self._list = Static(id="list")
        self._foot = Static(id="foot")
        with Vertical(id="card"):
            yield self._list
            yield self._foot

    def on_mount(self) -> None:
        self.refresh_view()

    def on_resize(self, _event) -> None:  # type: ignore[no-untyped-def]
        self.refresh_view()

    def visible_rows(self) -> tuple[int, int]:
        return self._drawn, len(self._rows)

    def _budget(self) -> int:
        """Card grows to `int(height * 0.8)`; TWO lines per row, no cap.

        Measured from the list's REAL height when it has one. A row whose
        line 2 is clipped by the card edge is precisely the ragged metadata
        this variant must not show — a half-drawn pair reads as a missing
        field rather than as a scroll boundary.
        """
        inner = self._list.size.height if self._list.size.height else int(
            self.app.size.height * 0.8
        ) - 4
        return max(1, inner // 2)

    def action_move(self, delta: int) -> None:
        self._move(delta)

    def refresh_view(self) -> None:
        ink = _ink()
        drawn = self._budget()
        self._scroll_into_window(drawn)
        page = self._rows[self._top : self._top + drawn]
        self._drawn = len(page)
        self.missing_checkpoints = 0

        # Real card width, so the two lines fit the terminal rather than a
        # constant that overflowed at 80 cols.
        card = max(30, (self._list.size.width or 84))
        out = Text()
        for offset, row in enumerate(page):
            selected = self._top + offset == self._cursor
            checkpoint = self._data.checkpoint(row.id)
            if not checkpoint:
                self.missing_checkpoints += 1
            out.append("❯ " if selected else "  ", style=ink["accent"])
            out.append(
                truncate_cells(row.name or row.id, max(12, card - 26)),
                style=f"bold {ink['fg']}" if selected else ink["fg"],
            )
            out.append(
                f"  last worked {format_age(max(0.0, self._now - row.mtime))}\n",
                style=ink["muted"],
            )
            # Reserved unconditionally — never collapsed, never re-ordered.
            created = self._data.created_at(row.id)
            started = f"started {format_age(max(0.0, self._now - created))}" if created else UNKNOWN
            count = self._data.msg_count(row.id)
            msgs = f"{count} msgs" if count else UNKNOWN
            outcome = checkpoint.get("last_turn_outcome") or UNKNOWN
            meta = " · ".join(
                (started, msgs, _short_model(checkpoint), _short_cwd(checkpoint), outcome)
            )
            out.append(f"    {truncate_cells(meta, max(10, card - 6))}\n", style=ink["dim"])
        self._list.update(out if page else Text("no sessions", style=ink["dim"]))

        bar = Text()
        bar.append("/ ", style=ink["accent"])
        bar.append(self._query or "type to filter", style=ink["fg"] if self._query else ink["dim"])
        bar.append(
            f"   {self._drawn} drawn · {len(self._rows)} matching · {len(self._all_rows)} total",
            style=ink["muted"],
        )
        self._foot.update(bar)


# ══════════════════════════════════════════════════════════════════════════
# Variant C — Grep. Full-width single column; the match IS the preview.
# ══════════════════════════════════════════════════════════════════════════


class GrepScreen(_Filtering, Screen[None]):
    """Full-width rows that EXPLAIN why they matched — ripgrep, not a file list.

    The picker already searches transcript bodies and marks hits with a quiet
    `” ` glyph (`BODY_MATCH_MARKER`, session_picker.py:180) and the user does
    not know any of it exists. C shows the match instead of marking it.

    With an EMPTY filter C degrades to today's list plus the height change.
    That is its sharpest trade-off and it is left visible on purpose — no
    content is invented for the empty-query case.
    """

    DEFAULT_CSS = """
    GrepScreen { layout: vertical; background: $surface; }
    GrepScreen #list { height: 1fr; padding: 0 1; }
    GrepScreen #filter { height: 1; padding: 0 1; background: $panel; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "close", show=False),
        Binding("enter", "select", "pick", show=False),
        Binding("up", "move(-1)", "up", show=False),
        Binding("down", "move(1)", "down", show=False),
        Binding("ctrl+p", "move(-1)", "up", show=False),
        Binding("ctrl+n", "move(1)", "down", show=False),
        Binding("backspace", "backspace", "delete", show=False),
    ]

    def __init__(
        self,
        rows: list[SessionRow],
        now: float,
        digests: dict[str, str],
        data: PreviewData,
    ) -> None:
        super().__init__()
        self._init_filter(rows, digests)
        self._now = now
        self._data = data
        self._drawn = 0

    def compose(self) -> ComposeResult:
        self._list = Static(id="list")
        self._filter = Static(id="filter")
        yield self._list
        yield self._filter

    def on_mount(self) -> None:
        self.refresh_view()

    def on_resize(self, _event) -> None:  # type: ignore[no-untyped-def]
        self.refresh_view()

    def visible_rows(self) -> tuple[int, int]:
        return self._drawn, len(self._rows)

    def _line_budget(self) -> int:
        """LINES, not rows — a row costs 1 or 2 depending on its context line."""
        return max(2, int(self.app.size.height * 0.9) - 2)

    def _row_lines(self, row: SessionRow) -> int:
        if not self._query.strip() or not self._body_matched(row):
            return 1
        return 2 if self._data.grep_context(row.id, self._query.strip()) else 1

    def action_move(self, delta: int) -> None:
        self._move(delta)

    def _fit(self) -> int:
        """How many rows from `_top` fit in the line budget."""
        budget = self._line_budget()
        used = 0
        count = 0
        for row in self._rows[self._top :]:
            cost = self._row_lines(row)
            if used + cost > budget:
                break
            used += cost
            count += 1
        return max(1, count)

    def refresh_view(self) -> None:
        ink = _ink()
        # Two passes: the window size depends on the rows in it, and the
        # cursor clamp depends on the window. One pass could leave the cursor
        # on an undrawn row, which is the exact defect `_page_rows` exists for.
        self._scroll_into_window(self._fit())
        drawn = self._fit()
        self._scroll_into_window(drawn)
        drawn = self._fit()
        page = self._rows[self._top : self._top + drawn]
        self._drawn = len(page)

        query = self._query.strip()
        width = max(30, self.app.size.width - 8)
        out = Text()
        for offset, row in enumerate(page):
            selected = self._top + offset == self._cursor
            out.append("❯ " if selected else "  ", style=ink["accent"])
            out.append(
                truncate_cells(row.name or row.id, max(10, width - 30)),
                style=f"bold {ink['fg']}" if selected else ink["fg"],
            )
            out.append(f"  {format_age(max(0.0, self._now - row.mtime))}", style=ink["muted"])
            context = None
            if query and self._body_matched(row):
                context = self._data.grep_context(row.id, query)
                if context is None:
                    # Soft/fuzzy hit: no literal substring to locate. Marked,
                    # not explained — whether soft hits DESERVE context is one
                    # of the things this prototype is for.
                    out.append("  ~", style=ink["warning"])
            out.append("\n")
            if context:
                out.append("    ")
                out.append(
                    _highlight(
                        truncate_cells(context, width - 4), query, ink["dim"], ink["warning"]
                    )
                )
                out.append("\n")
        self._list.update(out if page else Text("no sessions", style=ink["dim"]))

        bar = Text()
        bar.append(" / ", style=ink["accent"])
        bar.append(
            self._query or "type to search names AND bodies",
            style=ink["fg"] if self._query else ink["dim"],
        )
        bar.append(
            f"   {self._drawn} drawn · {len(self._rows)} matching · {len(self._all_rows)} total",
            style=ink["muted"],
        )
        if query:
            bar.append(f"   {len(self._exact)} body · {len(self._soft)} fuzzy", style=ink["dim"])
        self._filter.update(bar)


#: THE SEAM — frozen, byte-identical in both briefs. Slice 1 imports this.
#: `Screen[None]` follows this repo's convention (app.py:2347); the brief's
#: bare `Screen` is the same runtime type, and every screen exits via
#: `app.exit`, never a screen result.
VARIANTS: dict[str, tuple[str, type[Screen[None]]]] = {
    "A": ("Telescope", TelescopeScreen),
    "B": ("Ledger", LedgerScreen),
    "C": ("Grep", GrepScreen),
}


if __name__ == "__main__":
    print(f"{len(VARIANTS)} variants, store read-only, now={time.time():.0f}")
    for key, (label, screen) in VARIANTS.items():
        print(f"  {key}  {label:<10} {screen.__name__}")
