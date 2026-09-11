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

import re
import textwrap
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

#: Longest name the list pane must show uncut before a side-by-side split is
#: allowed: the p75 of the 141 real session names (median 33, p75 39, p90 43,
#: p95 46, max 64). Measured on this machine's store, not estimated.
NAME_P75 = 39

#: Cells the name column is capped at: the longest of the 141 real names (max
#: 64, p95 48, p99 53 — measured in CELLS, not chars). This cap is what makes
#: the breakpoint SOUND, and it is not cosmetic.
#:
#: Round 2 set the breakpoint from "when does side-by-side fit the p75 name",
#: which is the wrong question and produced D16 (BLOCKER): widening 139 -> 140
#: CUT the name field. Measured, the uncapped fields are
#: `stacked = W - 30` and `side-by-side = split * W - 30`, so stacked gains a
#: full cell per terminal column and side-by-side gains only `split`. Their gap
#: therefore DIVERGES without limit (measured at split 0.6: -64 cells at W=160,
#: -128 at W=320) and NO breakpoint can satisfy "side-by-side is never narrower
#: than stacked would be at the same width". Raising the breakpoint alone is
#: unsatisfiable, not merely expensive.
#:
#: A cap makes both layouts SATURATE at the same value, so beyond the width
#: where side-by-side reaches the cap the two are exactly equal and the
#: invariant holds for every larger width. 64 is chosen because 0 of 141 real
#: names exceed it: at the cap nothing truncates, so equal fields also means
#: equally zero truncation rather than equally bad.
#:
#: The two `List every tool name you have available, as a plain comma-separ…`
#: rows that still show an ellipsis at 160 and 180 are NOT truncated by this
#: cap: their stored titles are 64 cells INCLUDING a literal `…`, written that
#: way by whatever generated them. No width renders them in full, so round 2's
#: "indistinguishable pair" is a session-titling defect upstream of the picker,
#: not a layout one. Noted, out of scope for this prototype.
NAME_MAX = 64

#: Below this terminal width A stacks vertically. MEASURED, not derived on
#: paper: the value is the smallest width at which the rendered side-by-side
#: name field reaches NAME_MAX and so stops being narrower than the stacked
#: field at that same width. `scripts/prototype_resume_d16.py` sweeps both
#: layouts at every width and prints the crossover; re-run it after touching
#: LIST_FR/PREVIEW_FR or the fixed columns, because all three move this number.
#:
#: 159, not the 154 the same arithmetic predicts: the rendered pane comes out
#: 2-3 cells narrower than `split * cols` because the preview's `border-left`
#: and both panes' `padding: 0 1` are taken before the text width, and
#: Textual's `fr` resolution rounds. Deriving this on paper is exactly how
#: round 2 produced D16; the sweep is the source of truth.
STACK_BELOW_COLS = 159

#: The side-by-side split, in the LIST's favour (60/40). Round 2 had this
#: backwards at 2fr/3fr: the pane that truncates got the minority share while
#: the preview showed visible slack (measured at 140: an 83-cell pane drawing a
#: 76-cell rule beside a 56-cell list truncating 33% of its names).
LIST_FR = 3
PREVIEW_FR = 2

#: Stacked-layout row split. A fraction with clamps, not a fixed count: a fixed
#: preview height either starves the list on a 24-row terminal or wastes half a
#: 50-row one.
PREVIEW_MIN = 8  # 3 header lines + 1 rule + body — below this the pane shows
#: metadata and no conversation, which is not a preview
PREVIEW_MAX = 14  # past this the list starves for no gain; the pane scrolls anyway
LIST_MIN = 6  # fewer rows than this is a menu, not a list

#: Markdown emphasis strippers for the preview (D5). Measured over 58
#: previewable sessions: `code` in 74%, **bold** in 57%, ## heading in 34%,
#: - bullet in 43%, _em_ in 2%. Regex-and-move-on is correct here — this is a
#: throwaway preview, not a markdown parser, and it must not add a dependency.
_MD_HEADING = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_MD_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*(\S(?:[^*]*\S)?)\*\*|__(\S(?:[^_]*\S)?)__")
_MD_EM_STAR = re.compile(r"(?<!\*)\*(\S(?:[^*\n]*\S)?)\*(?!\*)")
#: `_` only when not flanked by word characters, so `snake_case` survives.
_MD_EM_UNDER = re.compile(r"(?<![\w_])_(\S(?:[^_\n]*\S)?)_(?![\w_])")


def _demark(text: str) -> str:
    """Strip markdown emphasis markers, KEEP the text they wrapped.

    Bullet markers (`- `, `* `) are deliberately kept: they are structure the
    reader wants, not emphasis. The `*em*` pattern requires a non-space after
    the opening star, so a `* item` bullet never matches it.
    """
    text = _MD_HEADING.sub("", text)
    text = _MD_CODE.sub(r"\1", text)
    text = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _MD_EM_STAR.sub(r"\1", text)
    return _MD_EM_UNDER.sub(r"\1", text)


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
    TelescopeScreen #results { width: 3fr; padding: 0 1; }
    TelescopeScreen #preview { width: 2fr; padding: 0 1; border-left: solid $panel; }
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
        #: Tri-state: None means "never applied", which forces the first
        #: application on mount regardless of which side of the breakpoint we
        #: start on.
        self._stacked: bool | None = None
        #: The preview height last applied in the stacked layout. Part of the
        #: restyle guard: see `_apply_layout`.
        self._applied_pane_rows: int | None = None

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

    #: Rows `#cols` never gets: 1 filter row, 1 prototype state bar, 2 screen
    #: chrome. Measured as a constant 4 at every geometry from 80x24 to 200x60.
    CHROME_ROWS = 4

    def _cols_height(self) -> int:
        """Rows available to `#cols`, derived from the app rather than measured.

        `self.query_one("#cols").size.height` is the obvious source and is the
        WRONG one here: it lags the paint. On mount it reads one row ahead of
        the settled layout, and immediately after a resize it reads the
        PREVIOUS geometry — either way the state bar disagreed with the rows on
        screen (`13 drawn` over 12 rendered at 80x24). `app.size` is always the
        current terminal, and the chrome above is fixed, so this agrees with
        what is painted on the first pass and after every resize.
        """
        return max(1, self.app.size.height - self.CHROME_ROWS)

    def _pane_rows(self) -> int:
        """Rows the preview gets in the STACKED layout, list keeps the rest."""
        height = self._cols_height()
        if height >= LIST_MIN + PREVIEW_MIN:
            return min(PREVIEW_MAX, max(PREVIEW_MIN, height // 3))
        # Too short for both. Chrome is reserved first, the list takes what is
        # left and scrolls — the proposal's own precedent.
        return max(4, height - LIST_MIN)

    def _apply_layout(self) -> bool:
        """Flip `#cols` between side-by-side and stacked. Returns `stacked`.

        Restyles ONLY on an actual mode change: `refresh_view` runs on every
        cursor move, and restyling there thrashes the layout engine. The widget
        tree is never rebuilt — remounting the panes would lose `_pane_top`.
        """
        stacked = self.app.size.width < STACK_BELOW_COLS
        # Guard on the mode AND the stacked preview height. Mode alone is not
        # enough: the preview height is a function of terminal HEIGHT, which
        # changes without crossing the width breakpoint, and a mode-only guard
        # leaves a stale height applied (measured: 160x45 -> 80x24 kept a
        # 13-row preview, giving a 7-row list where 12 is required). Neither
        # value moves on a cursor keypress, so this still restyles only on a
        # real geometry change rather than on every paint.
        pane_rows = self._pane_rows() if stacked else None
        if stacked == self._stacked and pane_rows == self._applied_pane_rows:
            return stacked
        self._stacked = stacked
        self._applied_pane_rows = pane_rows
        cols = self.query_one("#cols")
        panel = self.app.theme_variables.get("panel", "#444444")
        if stacked:
            cols.styles.layout = "vertical"
            self._results.styles.width = "100%"
            self._preview.styles.width = "100%"
            self._results.styles.height = "1fr"
            self._preview.styles.height = pane_rows
            # Clear the border we are not using: a `border-left` left on draws a
            # stray vertical rule down the full-width preview. The ('none', c)
            # TUPLE is the clear that always works — assigning None only drops
            # the inline rule and lets DEFAULT_CSS's border back in, and the
            # literal "none" reads the existing rule and so raises on an edge
            # that never had one.
            self._preview.styles.border_left = ("none", panel)
            self._preview.styles.border_top = ("solid", panel)
        else:
            cols.styles.layout = "horizontal"
            self._results.styles.width = f"{LIST_FR}fr"
            self._preview.styles.width = f"{PREVIEW_FR}fr"
            self._results.styles.height = "1fr"
            self._preview.styles.height = "1fr"
            self._preview.styles.border_top = ("none", panel)
            self._preview.styles.border_left = ("solid", panel)
        return stacked

    def _budget(self) -> int:
        """LINES the list may draw — NO PAGE_ROWS_MAX.

        That 10-row ceiling is the single largest defect this prototype
        attacks — measured, production draws 10 rows at every terminal height.
        Layout-aware: stacked, the list gets whatever the preview leaves.

        Stacked, the list gets `#cols` less the preview. Both terms come from
        `_cols_height`, which is derived from `app.size` and therefore never
        lags a resize the way a measured widget height does.
        """
        if self._stacked:
            return max(1, self._cols_height() - self._pane_rows())
        return max(1, int(self.app.size.height * 0.9) - 3)

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
        """Body lines the preview can show: its REAL height less the header."""
        # 4 = 3 header lines + 1 rule. Falls back to the side-by-side estimate
        # on the first paint, before layout has resolved a height.
        real = self._preview.size.height
        if real:
            return max(1, real - 4)
        return max(1, int(self.app.size.height * 0.9) - 5)

    def _pane_width(self) -> int:
        """The preview's REAL text width; the 0.6 fraction was side-by-side only."""
        real = self._preview.size.width
        if real:
            return max(20, real - 4)
        return max(20, int(self.app.size.width * 0.6) - 6)

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
        # D18: start at the session's first USER turn when it has one. The
        # preview exists for recognition ("which session is this?"), and the
        # user's own request states what the session is FOR; an assistant turn
        # states where it had got to. Measured on the real store: 103 of 141
        # transcripts open on a user turn, but 36 open on assistant narration
        # mid-thought (`Fixing the stale import first…`) — and those 36 include
        # the default cursor row, which is why no round-2 frame contained a
        # single `▸ you`. Leading assistant turns are DROPPED rather than
        # scrolled past: at 80x24 the pane shows one turn, so "somewhere below"
        # is the same as absent.
        first_user = next((i for i, (role, _t, _ts) in enumerate(turns) if role == "user"), None)
        if first_user is not None:
            turns = turns[first_user:]
        width = self._pane_width()
        out: list[tuple[str, str]] = []
        for role, text, _ts in turns:
            out.append(("gutter", GUTTER.get(role, f"▪ {role}")))
            for para in _demark(text).splitlines():
                if not para.strip():
                    continue
                # Word-boundary wrap, still character-counted rather than
                # cell-exact: a prototype pane only has to be readable, and a
                # cell-exact wrap on CJK would need the model `truncate_cells`
                # carries. `break_long_words` keeps a 200-char URL from
                # overflowing the pane.
                for line in textwrap.wrap(
                    para, width, break_long_words=True, break_on_hyphens=False
                ):
                    # Belt-and-braces against D11: a continuation line that
                    # kept a leading space turns `_render_pane`'s uniform
                    # 2-space indent into 3 on that row alone.
                    stripped = line.rstrip()
                    if stripped:
                        out.append((role, stripped))
            out.append(("blank", ""))
        return out

    def _usable(self) -> int:
        """The list pane's REAL text width, not a fraction of the app.

        `2fr` of the split is what the row actually gets, and guessing it
        wrapped every row onto a second line at 80 cols. The `or` fallback
        matters on the first paint, before layout resolves.
        """
        fraction = 1.0 if self._stacked else LIST_FR / (LIST_FR + PREVIEW_FR)
        return max(12, (self._results.size.width or int(self.app.size.width * fraction)) - 2)

    def _show_id(self, usable: int) -> bool:
        """Show the id only when it does not push the name below p75.

        The old `width >= 52` turned the id on at 142 cols and cost 53 points
        of uncut-name share — a wider window with a worse list. One rule for
        both layouts: side-by-side keeps the id off until ~176 cols, stacked
        has it on from 80, where the full width affords both.
        """
        return (usable - 2 - 2 - 8 - 2 - 12) >= NAME_P75

    def _name_w(self) -> int:
        """Cells the name column gets — the ONE definition, so D16 is measurable.

        Inlined in `refresh_view` before, which meant the breakpoint could only
        be argued on paper. The invariant the breakpoint has to satisfy is a
        comparison between two layouts at the SAME width, so the quantity being
        compared needs a name and a single source of truth.
        """
        usable = self._usable()
        mark_w = 2 if self._query.strip() else 0
        raw = usable - 2 - 2 - 8 - ((2 + 12) if self._show_id(usable) else 0) - mark_w
        # The cap is what satisfies D16's invariant: both layouts saturate at
        # NAME_MAX, so past the breakpoint the two fields are EQUAL rather than
        # diverging. It also closes D24 — the 40-cell void between a short name
        # and a right-aligned age at wide stacked geometries.
        return max(8, min(NAME_MAX, raw))

    def _context_w(self) -> int:
        """Cells the context line gets: the 4-cell indent off the pane width."""
        return max(10, self._usable() - 4)

    def _raw_context(self, row: SessionRow) -> str | None:
        """Whether an exact-hit context EXISTS for `row`, regardless of layout.

        Kept separate from `_context_for` so the `~` mark keeps one meaning in
        both layouts: matching `GrepScreen`, `~` says the row is a soft/fuzzy
        hit with no literal substring to show. A row whose context is merely
        suppressed by the stacked layout is NOT a soft hit, and marking it as
        one would make the glyph lie about the row.
        """
        query = self._query.strip()
        if not query or not self._body_matched(row):
            return None
        # D17: ask for a window the width we will actually DRAW at. The default
        # `width=150` is centred on the match, but A then truncated it to the
        # list pane (~55 cells side-by-side) from the LEFT end, which cut the
        # match off the right: measured, the query sat at index 73 of a
        # 152-char snippet rendered into 55 cells, so 0 of 9 context lines
        # contained the query. C never hit this only because its line is
        # near-terminal-width. Same `grep_context` path as C, correct window.
        return self._data.grep_context(row.id, query, width=self._context_w())

    def _context_for(self, row: SessionRow) -> str | None:
        """The context line to DRAW for `row`, or None when it draws none.

        Stacked, the context line is the CURSOR row's only: cells are scarce
        (12 list rows at 80x24) and a line on each would halve the list, while
        the preview directly beneath already shows that row's content.
        """
        if self._stacked and self._rows[self._cursor].id != row.id:
            return None
        return self._raw_context(row)

    def _row_lines(self, row: SessionRow) -> int:
        """A row costs 1 line, or 2 when it DRAWS a context line."""
        return 2 if self._context_for(row) else 1

    def _fit(self) -> int:
        """How many rows from `_top` fit in the LINE budget."""
        budget = self._budget()
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
        # First: the layout may have just flipped, and every measurement below
        # (pane widths, budgets, the preview's height) depends on which side of
        # the breakpoint we are on.
        self._apply_layout()
        ink = _ink()
        # Two passes: the window size depends on the rows in it, and the cursor
        # clamp depends on the window. One pass could leave the cursor on an
        # undrawn row, which is the exact defect `_page_rows` exists for.
        self._scroll_into_window(self._fit())
        drawn = self._fit()
        self._scroll_into_window(drawn)
        drawn = self._fit()
        page = self._rows[self._top : self._top + drawn]
        self._drawn = len(page)

        query = self._query.strip()
        usable = self._usable()
        show_id = self._show_id(usable)
        # Fixed fields, so the id starts at a CONSTANT column on every row:
        # "❯ " (2) | name (NAME_W, padded) | 2 | age (8, right) | 2 | id (12).
        # Deriving the name budget from len(age) — which varies 6..8 cells
        # across the real rows — is what made the id ragged.
        #
        # The trailing `~` gets a RESERVED 2-cell gutter whenever a query is
        # active. Appending it unreserved overflowed the pane (measured: a
        # 59-cell row + 3 = 62 in a 59-cell pane) and wrapped the mark onto its
        # own line, which breaks the one-row-per-session contract the cursor
        # arithmetic assumes.
        name_w = self._name_w()
        lines: list[Text] = []
        for offset, row in enumerate(page):
            selected = self._top + offset == self._cursor
            line = Text()
            line.append("❯ " if selected else "  ", style=ink["accent"])
            age = format_age(max(0.0, self._now - row.mtime))
            # `truncate_cells` is the repo's one cell-width model; pad with it
            # rather than mixing in `len()`.
            name = truncate_cells(row.name or row.id, name_w).ljust(name_w)
            line.append(name, style=f"bold {ink['fg']}" if selected else ink["fg"])
            line.append(f"  {age:>8}", style=ink["muted"])
            if show_id:
                line.append(f"  {row.id}", style=ink["dim"])
            context = self._context_for(row)
            if query:
                # Soft/fuzzy hit: no literal substring to locate. Soft hits are
                # the majority for a broad query, so without this mark most
                # rows would be unexplained and the user could not tell "no
                # context" from "not a body match". Keyed to `_raw_context`, so
                # a stacked row whose context is merely not drawn is not
                # mismarked as fuzzy. The gutter is reserved above, so the
                # glyph sits at a FIXED column rather than trailing text.
                soft = self._body_matched(row) and self._raw_context(row) is None
                line.append(f" {'~' if soft else ' '}", style=ink["warning"])
            lines.append(line)
            if context:
                marked = Text("    ")
                # `_demark` before highlighting (D19): the strip was applied to
                # the preview and not to this excerpt, so literal `**` and
                # backticks rode into the list pane. Stripping BEFORE the
                # highlight also keeps the match offsets honest.
                marked.append(
                    _highlight(
                        truncate_cells(_demark(context), self._context_w()),
                        query,
                        ink["dim"],
                        ink["warning"],
                    )
                )
                lines.append(marked)
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
        # Omit the line entirely when there is no checkpoint, rather than
        # printing a bare `· · ·` that reads as a loading state that never
        # resolved. Measured: 112/141 rows have a checkpoint and on ALL of them
        # both model and cwd are present, so the line is fully populated or
        # fully absent — there is no partial case to design for. The reserved-
        # column argument behind UNKNOWN is about LIST ROWS, where a vanishing
        # column shifts names sideways; a header line in a single-row pane has
        # no such alignment to protect.
        if checkpoint:
            out.append(f"{_short_model(checkpoint)} · {_short_cwd(checkpoint)}\n", style=ink["dim"])
        # The rule must span the REAL preview width or it runs off the edge in
        # the stacked layout.
        out.append("─" * max(10, self._pane_width()) + "\n", style=ink["faint"])

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
