"""The dock-band todo panel (item 8).

Renders the session's todo list in the shared ``#band`` above the input, so
progress survives compaction AND stays visible without scrolling the
transcript. The todo tool (``local_operator.tools.builtin.execute_todo``) is
the only writer, and it writes to ``builtin.TODO_STORE`` keyed by session id
when the host attaches no ``todos`` dict to the ToolContext — which this
session never does (``Session._build_tool_context`` leaves ``todos=None``).
Reading that module table therefore reads the LIVE state; the import is
deliberately local to the accessor so merely composing the panel cannot pull
the tool registry into a reduced host.

The tool's list is FLAT (``init``/``add``/``done``/``block``/``drop``/``view``;
items carry ``text`` and ``status``, plus a ``reason`` on a blocked item — no
phases). The row model still routes every item through one builder
(:meth:`TodoPanel._item_row`) so a phase header could drop in later without
restructuring the render — but the tool schema is not extended to get there.

The panel sits above the input and must stay SHORT: long lists collapse to a
dim ``… N more todos`` line rather than pushing the composer off screen, and
the row cap answers to the screen's height rather than to a constant (see
:data:`_DOCK_ROWS`). Rows are truncated against the width they are drawn at
for the same reason — nothing in the layout constrains them (see
:meth:`TodoPanel._row_cells`).

Zero height when empty: the panel starts ``display: none`` and only shows
once the store for this session has at least one item; going back to empty
hides it again.
"""

from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from textual.containers import Container
from textual.dom import NoScreen
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod

#: Rows shown before the list collapses to a ``… N more`` line, on any terminal
#: tall enough to afford them. The panel is chrome above the composer, not the
#: transcript: a twenty-item list rendered in full pushed the input down by a
#: screenful on the exact turn the user is trying to read it. Eight keeps the
#: longest plausible plan on screen and stays out of the way.
MAX_TODO_ROWS = 8

#: Screen rows the dock spends around this panel, subtracted from the screen
#: height to get the rows the panel may paint. ``MAX_TODO_ROWS`` used to be
#: ABSOLUTE, so at a 14-row terminal the panel asked for ten rows inside a
#: twelve-row screen and ``Screen { overflow: hidden }`` swallowed the
#: difference silently: the ``Todos · 0/12`` header and the first three rows
#: were clipped ABOVE the top edge, and the surviving ``… 4 more todos`` marker
#: counted only the tail the panel dropped itself while seven were unseen.
#:
#: Measured at 100x12/14/16/18/20 in a live session, the rows around the panel
#: are:
#:
#: * 2 — ``TranscriptView``'s own padding. It is ``height: 1fr`` and collapses
#:   to zero content rows, but the padding rows are irreducible.
#: * 5 — ``#input-shell``: the editor row, the status band, the row between
#:   them, and one row of padding on each side.
#: * 1 — this slot's own rhythm row (``.band-slot { padding: 0 0 1 0 }``).
#:
#: The band's top inset (``#band.has-slot { padding: 1 0 0 0 }``) is NOT in this
#: constant, because unlike every row above it that row is conditional: the app
#: drops it on screens too short to afford it (``MIN_BAND_INSET_SCREEN_ROWS``).
#: It is measured off the band itself instead — see
#: :meth:`TodoPanel._band_inset_rows` — so the budget and what is actually
#: painted cannot disagree. Counting it here unconditionally would starve the
#: shortest terminals of a row they are not being charged for.
#:
#: The header and the overflow marker are NOT in here: they are rows of the
#: panel, allocated out of what is left (see :meth:`TodoPanel._build`), because
#: the marker is the row the shortest terminals cannot afford.
#:
#: A SHARE of the screen (``command_picker``'s ``_SCREEN_HEIGHT_DIVISOR``) is
#: the wrong shape for this panel: everything around it competes for a fixed
#: number of rows rather than a proportion, so a divisor either overflows the
#: short terminals or starves the tall ones — a third of a 12-row screen is 4,
#: which still overflows, while a third of a 26-row screen is 8, which is the
#: cap anyway. Subtracting the dock resolves to exactly ``MAX_TODO_ROWS`` from a
#: 20-row terminal up, and gives back one row at a time below it.
_DOCK_ROWS = 8

#: Floor for the panel: the header plus one row, the least that is worth
#: painting. It binds below a 10-row SCREEN (a 12-row terminal), where the dock
#: exceeds the terminal on its own — ``#input-shell`` and the transcript's
#: padding are seven of those ten rows — so the panel is not the surface that
#: can fix it, and shrinking further would only trade one clipped row for a
#: panel that says nothing.
_MIN_BODY_ROWS = 2

#: Cells the dock's rail spends on every row: ``.band-body { padding: 0 1 }``,
#: one cell on each side of the body.
_BODY_RAIL_CELLS = 2

#: Checkbox mark per status, matching the todo tool's ``view`` op
#: (``local_operator.tools.builtin._TODO_MARKS``) mark for mark: the band and
#: the transcript receipt describe one list, so they must not spell it two ways.
#: Restated here rather than imported because reading the store is the only
#: reason this module ever touches the tools package, and it does that behind a
#: function-local import (see :func:`todo_items`).
STATUS_MARKS = {"pending": " ", "done": "x", "blocked": "~", "dropped": "-"}

#: Statuses needing no further work. ``blocked`` is absent deliberately — it is
#: open work waiting on a decision, and counting it would make a stalled list
#: read as a finished one.
RESOLVED_STATUSES = ("done", "dropped")


def todo_items(session_id: str) -> list[dict[str, str]]:
    """The live todo list for ``session_id`` (a copy; never the store itself).

    Reads the todo tool's own store — the one writer is the tool, so the table
    IS the state. Copies each item because the tool mutates its dicts in place
    (``done`` flips a status), and a panel holding the originals would repaint
    from a stale snapshot. A missing/empty id or a store the import cannot
    reach degrades to "no todos" — a panel must never take the app down.
    """
    if not session_id:
        return []
    try:
        from local_operator.tools.builtin import TODO_STORE
    except Exception:
        return []
    items = TODO_STORE.get(session_id)
    if not items:
        return []
    return [dict(item) for item in items]


class TodoPanel(Container):
    """The session's todo list, rendered in the dock band above the input.

    The panel is a transparent SLOT (its BOTTOM padding row is the rhythm row
    between it and whatever sits below — a margin would violate the sheet's
    one-margin rule), holding one filled body. Transparent means the dock's
    fill, not the screen's: ``#band`` carries ``$lo-surface``, so the slot's
    blank row is part of the dock rather than a strip of transcript showing
    through between the list and the composer. Visibility is the panel's own:
    ``display: none`` whenever the store is empty, so the band collapses to
    zero rows.
    """

    def __init__(self) -> None:
        super().__init__(id="todo-panel", classes="band-slot")
        self._body = Static(classes="band-body", id="todo-body")
        #: What is painted: the ``(text, status, reason)`` fingerprint per row
        #: AND the row budget they were rendered against (see :meth:`sync`), so
        #: the 1 Hz poll repaints only when something actually changed — an
        #: equality guard, same discipline as the assistant flush. Both terms,
        #: because the same list renders differently when its space changes.
        self._shown: tuple[tuple[tuple[str, str, str], ...], int] | None = None
        # Hidden until the first todo exists: an empty panel is not content.
        self.display = False

    def compose(self):  # type: ignore[override]
        yield self._body

    # -- sync -----------------------------------------------------------------
    def sync(self, session: Any) -> None:
        """Re-read the store and repaint only on change.

        Called on the todo tool's ``tool_execution_end`` (immediate) AND on the
        app's 1 Hz job poll (the belt to that event's suspenders: a ``done``
        op that settles while the card is still painting still lands here).
        Never raises: a status surface must not be able to take the app down.
        """
        try:
            session_id = getattr(session, "session_id", "") or ""
            items = todo_items(session_id)
            # The reason rides IN the fingerprint: re-blocking an item with a
            # new reason changes what the row says, and an equality guard blind
            # to it would leave the old wait on screen.
            fingerprint = tuple(
                (
                    str(item.get("text", "")),
                    str(item.get("status", "pending")),
                    str(item.get("reason", "")),
                )
                for item in items
            )
            # The BUDGET rides in the guard beside the list, because what the
            # panel paints is a function of both: the same items render a
            # different number of rows (and a different `… N more` count) when
            # the rows available change. A guard blind to it left the previous
            # paint standing after a resize — or after the band's own top inset
            # appeared, which takes a row from this budget — and a body one row
            # taller than its budget is clipped, silently, by
            # `Screen { overflow: hidden }`.
            budget = self._body_rows()
            state = (fingerprint, budget)
            if state == self._shown:
                return  # equality guard — identical list and budget = no work
            self._shown = state
            if not fingerprint:
                self.display = False
                return
            self.display = True
            self._body.update(self._build(fingerprint))
        except Exception:
            self.display = False

    # -- rendering -----------------------------------------------------------
    def _build(self, rows: tuple[tuple[str, str, str], ...]) -> RenderableType:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))

        # ``n/total`` on its own was a lie by arithmetic: one ``done`` beside
        # four ``dropped`` rendered ``Todos · 5/5``, which is what a FINISHED
        # plan looks like. Taking the four back out of the numerator would swap
        # that for the opposite lie — a fully triaged list reading as
        # permanently stalled — so the arithmetic stays and the words that make
        # it honest come back. The tool's own transcript receipt says "n/total
        # resolved"; the band had kept the sum and dropped the word. The
        # abandoned count is then stated BESIDE the fraction rather than hidden
        # inside it, so four abandoned items can never read as four finished
        # ones. ``blocked`` stays outside the numerator either way: it is work
        # waiting on an answer, and letting it read as complete is how a
        # stalled list looks finished.
        resolved = sum(1 for _text, status, _reason in rows if status in RESOLVED_STATUSES)
        dropped = sum(1 for _text, status, _reason in rows if status == "dropped")
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Todos", style=muted)
        header.append(" · ", style=dim)
        # ``muted``, not ``dim``: ``dim`` is 4.18:1 on the band's ground and
        # these are the numbers the panel exists to add — the same call the
        # subagent panel's numbers got one file over (``compose_row``).
        header.append(f"{resolved}/{len(rows)} resolved", style=muted)
        if dropped:
            header.append(" · ", style=dim)
            header.append(f"{dropped} dropped", style=muted)

        # Rows are allocated in priority order out of what the screen affords.
        # The header is never negotiable: it is the only line that states the
        # total, so it is not counted here. What is left goes to items, and the
        # overflow marker takes one of them only if there is one to take.
        room = max(1, self._body_rows() - 1)
        cap = min(room, MAX_TODO_ROWS)
        # ``room == 1`` keeps the item and drops the marker: on a screen that
        # short there is one row to spend, and an item is worth more than a count
        # the header's own denominator already implies. It is also what keeps the
        # band inside a 12-row terminal at all.
        marker = len(rows) > cap and room > 1
        if marker:
            cap = min(room - 1, MAX_TODO_ROWS)
            if len(rows) == cap + 1:
                # ``… 1 more todo`` costs exactly the row the item itself costs,
                # so it is never worth drawing. The panel's height is identical
                # either way, which is what makes going one over the cap safe at
                # the boundary.
                cap += 1
                marker = False
        visible = rows[:cap]

        lines = [header]
        # Every item routes through ONE row builder: the tool's list is flat
        # today, but a phase header later drops in beside this call rather
        # than restructuring the render.
        for text, status, reason in visible:
            lines.append(self._item_row(text, status, reason))
        if marker:
            overflow = Text(no_wrap=True, overflow="ellipsis")
            # Counts what the reader cannot see, not what this loop dropped.
            overflow.append(f"… {len(rows) - len(visible)} more todos", style=dim)
            lines.append(overflow)

        # The clip happens HERE, against a width, because nothing in the layout
        # supplies one (see :meth:`_row_cells`).
        cells = self._row_cells()
        if cells:
            for line in lines:
                if line.cell_len <= cells:
                    continue
                # Rich's own ``overflow="ellipsis"`` leaves the cut as "word …"
                # whenever it lands on a space, so the same row would truncate in
                # two typographic styles one column apart. The project's one
                # truncator rstrips first (``tool_card.truncate_cells``) and this
                # does the same, span-aware: crop a cell short, drop the trailing
                # space, then spend that cell on the marker — which is `dim`
                # because a truncation mark is chrome, not part of the sentence.
                line.truncate(cells - 1, overflow="crop")
                while line.plain.endswith(" "):
                    line.right_crop(1)
                line.append("…", style=dim)
        return Text("\n").join(lines)

    # -- budgets --------------------------------------------------------------
    def _row_cells(self) -> int:
        """Cells one row may occupy, or 0 for "nothing to clamp against".

        A ``Text``'s ``no_wrap``/``overflow="ellipsis"`` pair only fires when
        something lays it out against a width, and nothing here does: ``#band``
        is ``width: auto`` and ``.band-slot``/``.band-body`` are ``1fr``, so the
        band's width IS the widest row's natural width and every row is laid out
        at exactly the size it asked for. Measured at a 52-column terminal,
        ``#todo-body`` came out **129 cells inside a 50-cell screen**: the
        container cut the long rows flush against the edge with no marker, so a
        blocked reason stopped mid-word (``— blocked: wa``) and nothing said it
        continued. The new ``reason`` suffix is what makes rows long enough to
        reach that.

        Truncating against the SCREEN is what makes the ellipsis real, and it
        settles the width overflow at the same time: with the rows clamped, the
        auto-width band measures the screen instead of 129 cells.
        ``Screen.size`` already excludes the app's one-cell edge padding, so the
        only chrome left to subtract is the dock's own rail.
        """
        try:
            width = self.screen.size.width
        except NoScreen:
            return 0
        if width <= 0:
            return 0
        return max(1, width - _BODY_RAIL_CELLS)

    def predicted_rows(self) -> int:
        """Content rows this panel will paint, for a caller that cannot measure.

        The dock's inset check runs at the moment a panel appears, when the slot
        has not been arranged yet and measures zero (see ``app._slot_rows``).
        Answering from the painted body — falling back to the budget before the
        first paint — is what lets that check be right on the first frame rather
        than correcting itself a tick later, which the user sees as the dock
        jumping.

        Never raises and never returns less than one: a displayed panel is at
        least a row, and under-counting hands the transcript a row the dock is
        about to take.
        """
        try:
            content = str(self._body.content)
        except Exception:  # body not built yet
            content = ""
        if content:
            return max(1, len(content.split("\n")))
        return max(1, self._body_rows())

    def _body_rows(self) -> int:
        """Rows this paint may fill — header, items and any overflow marker.

        What the screen has left after the dock around the panel
        (:data:`_DOCK_ROWS`) and after whatever the band's other slot is already
        spending: the subagent panel shares this band, and a budget blind to it
        would put the same upward clip back the moment a subagent runs on a short
        terminal. Ceiling is a full list plus its two chrome rows; floor is
        :data:`_MIN_BODY_ROWS`.
        """
        ceiling = MAX_TODO_ROWS + 2
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            return ceiling
        if screen_height <= 0:
            return ceiling
        spare = screen_height - _DOCK_ROWS - self._band_inset_rows() - self._band_sibling_rows()
        return max(_MIN_BODY_ROWS, min(ceiling, spare))

    def _band_inset_rows(self) -> int:
        """Rows the BAND's own top inset is spending, as it is right now.

        Measured rather than assumed because the inset is conditional: the app
        drops ``has-slot`` on screens too short to afford the row (see
        ``OperatorApp._sync_band_inset``), and a budget that charged for it
        unconditionally would take a row off the panel that nothing was
        spending — on exactly the terminals with none to spare.

        Read from the band's CLASS rather than its resolved padding, and the
        difference is a visible reflow. The app toggles ``has-slot`` before the
        panels paint, but Textual resolves the padding that class carries in a
        LATER layout pass — so a padding read during this paint returns the
        PREVIOUS frame's value, the list is sized against a budget that is about
        to change, and the panel repaints one row shorter on the next tick. That
        is motion the user sees: the todo list appears, then visibly loses a row.
        The class is set synchronously and is therefore the honest answer to
        "how much room will this paint actually have".

        The row's SIZE is still the stylesheet's (``#band.has-slot`` declares
        one row); this only asks whether it is being spent. Degrades to 0 for
        any host whose band is missing, the same posture as
        :meth:`_band_sibling_rows`.
        """
        parent = self.parent
        if parent is None:
            return 0
        try:
            return 1 if parent.has_class("has-slot") else 0
        except Exception:  # not a band (reduced test hosts); charge nothing
            return 0

    def _band_sibling_rows(self) -> int:
        """Rows the band's OTHER visible slots occupy, outer size included
        because each slot owns a rhythm row below itself.

        Read off the last layout, which is the honest source: the repaint that
        calls this runs after the band settled, so a sibling's height is the one
        already on screen. Zero before the first layout — that only makes the
        very first paint as generous as it used to be, and the next one corrects
        it.
        """
        parent = self.parent
        if parent is None:
            return 0
        rows = 0
        for slot in parent.children:
            if slot is self or not slot.display:
                continue
            rows += slot.outer_size.height
        return rows

    def _item_row(self, text: str, status: str, reason: str = "") -> Text:
        """One ``- [ ]``/``- [x]``/``- [~]``/``- [-]`` row — the tool's own
        vocabulary (its ``view`` op renders exactly these marks), so the panel
        and the transcript receipt read identically.

        Two axes, because one was not enough. OPEN vs SETTLED is LUMINANCE: open
        work is ``muted``/``fg``, settled work is ``dim`` + strikethrough,
        because a settled item is a record, not an instruction — the same trade
        the tool ledger makes. WHICH open and WHICH settled state is a WORD,
        because inside a tier the rows were one dim character apart: ``- [x]``
        beside ``- [-]`` made a finished item and an abandoned one read as the
        same thing, and ``blocked`` sat in ``pending``'s ink with its reason —
        the part that says what it waits on — at 4.18:1, the least legible text
        in the band, on the one row the user has to act on.

        So a state that needs saying gets said, in one grammar: ``text —
        state``. ``— blocked``, reason and all, is what the tool's own receipt
        already writes (``builtin._todo_rows``), so the band drawing it whether
        or not a reason came with it closes a drift rather than opening one.
        ``— dropped`` is the band's own addition, and the one place it says a
        word the receipt does not: in the transcript that mark is read inside a
        printed list of four, while here a dim struck row's only differentiator
        was a single glyph. The tag is NOT struck, so it stays readable on a row
        that is crossed out.

        Ink: ``blocked`` takes ``fg`` (13.76:1), the loudest ink in the band,
        because it is the only row asking for something, and its reason steps up
        to ``muted`` (7.93:1) from ``dim``. No colour, deliberately — the dock
        band spends colour on failure and on nothing else (``subagent_panel``'s
        ink law), and a todo waiting on an answer has not failed.

        An unrecognised status falls back to the open rendering, never the
        settled one: a future status must not silently read as finished.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        row = Text(no_wrap=True, overflow="ellipsis")
        # One gutter, one ink: the mark column is the dock's rail — the same
        # column as the composer's chevron and the status band's leading glyph —
        # so it stays `dim` on every row and the state is said in the text.
        row.append(f"- [{STATUS_MARKS.get(status, ' ')}] ", style=dim)
        if status in RESOLVED_STATUSES:
            body = dim + Style(strike=True)
        elif status == "blocked":
            body = Style(color=theme_mod.semantic_color("fg"))
        else:
            body = muted
        # Model-controlled text reaches a real terminal: stripped like every
        # other untrusted string this app renders (same discipline as the
        # approval prompt and the tool cards).
        row.append(strip_control_sequences(text), style=body)
        if status == "blocked":
            row.append(" — blocked", style=muted)
            if reason:
                row.append(f": {strip_control_sequences(reason)}", style=muted)
        elif status == "dropped":
            row.append(" — dropped", style=dim)
        return row
