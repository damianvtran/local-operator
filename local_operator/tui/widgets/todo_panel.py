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

The tool's store is PHASED: ``list[{"name", "items":[{text,status[,reason]}]}]``
(``local_operator.tools.builtin.TODO_STORE``). A flat ``init`` maps to ONE
implicit phase named ``"Todos"``, which this panel renders HEADERLESS so an
existing single-phase list looks byte-identical to the pre-phases panel — the
back-compat guarantee the ``test_band_panels.py`` goldens guard (design §6.3).
Every item still routes through one builder (:meth:`TodoPanel._item_row`, the
single mark authority the transcript receipt mirrors); a phase header is drawn
beside that call in :meth:`_build`, and the item indent is added there too so
``_item_row`` never learns about phases.

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

import time
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

#: Sentinel name of the implicit phase a flat ``init`` maps to. A single phase
#: with this name renders HEADERLESS and byte-identical to the pre-phases panel
#: (design §6.3) — the back-compat contract the existing goldens assert.
_IMPLICIT_PHASE = "Todos"

#: Closed rows kept directly above a phase's open window in the collapsed view,
#: so a just-completed item stays visible as it settles rather than silently
#: vanishing (omp ``COLLAPSED_CLOSED_CONTEXT``, todo.ts:271). The lead is
#: additive — it never costs an open row.
_COLLAPSED_CLOSED_CONTEXT = 1

#: Open items previewed per phase in the collapsed view (omp ``activeTaskCap``,
#: interactive-mode.ts:2214). Beyond this the phase's affordance counts the rest.
_ACTIVE_TASK_CAP = 5

#: Phases shown after the active one in the collapsed view (omp
#: ``subsequentStageCap``, interactive-mode.ts:2213); the header's ``i/n`` count
#: implies any not shown.
_SUBSEQUENT_STAGE_CAP = 4

#: Seconds a phase must stay CONTINUOUSLY, FULLY settled before the panel hides
#: it from the VIEW (never the store — design §7.4). Matches omp's default
#: ``todoClearDelay``. The hide rides the app's existing 1 Hz poll (§7.5): no
#: dedicated timer, so no callback can fire against a torn-down panel.
TODO_PHASE_HIDE_DELAY_S = 60.0


def _as_phases(raw: list[Any]) -> list[dict[str, Any]]:
    """Coerce a stored owner-list to phases — the one shape every reader walks.

    A legacy/flat list (item dicts at the top level, ``{"text", ...}``) becomes
    one implicit ``"Todos"`` phase; an already-phased list (``{"name",
    "items"}``) passes through. The tool's own ``_as_phases`` (design §3.3) is
    canonical; this mirror exists so the panel tolerates either shape defensively
    — a panel must never take the app down over a store it did not expect.
    """
    if raw and isinstance(raw[0], dict) and "items" in raw[0]:
        return raw  # already phased
    return [{"name": _IMPLICIT_PHASE, "items": list(raw)}]  # legacy/flat


def todo_items(session_id: str) -> list[dict[str, Any]]:
    """The live todo list for ``session_id``, as phases (copies, never the store).

    Reads the todo tool's own store — the one writer is the tool, so the table
    IS the state — and returns a list of ``{\"name\", \"items\":[copied dicts]}``
    phases via :func:`_as_phases`, so a legacy/flat store still yields one
    implicit ``\"Todos\"`` phase (rendered headerless downstream). Copies each
    item because the tool mutates its dicts in place (``done`` flips a status),
    and a panel holding the originals would repaint from a stale snapshot. A
    missing/empty id or a store the import cannot reach degrades to "no todos" —
    a panel must never take the app down.
    """
    if not session_id:
        return []
    try:
        from local_operator.tools.builtin import TODO_STORE
    except Exception:
        return []
    raw = TODO_STORE.get(session_id)
    if not raw:
        return []
    return [
        {
            "name": str(phase.get("name", _IMPLICIT_PHASE)),
            "items": [dict(item) for item in phase.get("items", [])],
        }
        for phase in _as_phases(raw)
    ]


def _is_closed(item: dict[str, Any]) -> bool:
    """A todo needing no further work. ``blocked`` is NOT closed — it is open
    work waiting on a decision, so treating it as closed is exactly the omp
    ``#isTodoListSettled`` bug (interactive-mode.ts:2093) that scrubbed live
    phases. Only ``done``/``dropped`` (``RESOLVED_STATUSES``) close a row."""
    return item.get("status") in RESOLVED_STATUSES


def select_collapsed(items: list[dict[str, Any]], cap: int) -> tuple[list[dict[str, Any]], int]:
    """Return ``(rows_to_show, hidden_open_count)`` for one phase's collapsed
    preview — a direct port of omp ``selectCollapsedTodos``/``selectWithinCap``
    (todo.ts:332, :286), minus the subagent-match axis this panel has no notion
    of (local-operator's panel does not light pending rows by subagent).

    Open items fill the cap; the last ``_COLLAPSED_CLOSED_CONTEXT`` closed items
    are kept as an additive lead so a just-completed row stays visible as it
    settles rather than silently disappearing; a phase with no open work left
    selects over its own closed rows (the settled-phase case). ``done`` accepts
    any named item, so closed items are not necessarily a prefix — hence the
    lead is taken as the LAST closed items in order, not a slice off the front.
    """
    open_items = [i for i in items if not _is_closed(i)]
    if not open_items:
        # Settled phase: select over its own closed rows (omp selectWithinCap).
        shown = items[-cap:] if len(items) > cap else items
        return list(shown), 0
    lead = [i for i in items if _is_closed(i)][-_COLLAPSED_CLOSED_CONTEXT:]
    within = open_items[:cap]
    hidden = len(open_items) - len(within)
    return [*lead, *within], hidden


def _active_phase_index(phases: list[dict[str, Any]]) -> int:
    """Index of the EARLIEST phase still holding open work — the collapsed
    view's viewport anchor (omp ``formatSummary``'s ``currentIdx``, todo.ts:732).

    Open = any item not closed (pending or blocked). A phase can be worked
    "ahead" (a later phase completed while an earlier one still has open items),
    so the pointer is the earliest open phase, not the last-touched one. When
    every phase is settled the pointer falls to the last phase, so a fully-done
    plan still anchors somewhere real. Empty list → 0 (nothing to anchor)."""
    for idx, phase in enumerate(phases):
        if any(not _is_closed(item) for item in phase.get("items") or []):
            return idx
    return max(0, len(phases) - 1)


def _phase_settled(phase: dict[str, Any]) -> bool:
    """True only when EVERY item in the phase is closed (``done``/``dropped``).

    The auto-hide safety invariant (design §7.4, omp ``#isTodoListSettled``): a
    phase is hideable ONLY when it holds no pending AND no blocked item. A
    blocked item is open work waiting on an answer — hiding a phase that still
    holds one would drop the one row the user has to act on and reset the
    phase's progress counter. An empty phase is not settled (nothing to hide).
    """
    items = phase.get("items") or []
    return bool(items) and all(_is_closed(item) for item in items)


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
        #: What is painted, as an equality guard the 1 Hz poll repaints only on
        #: change (same discipline as the assistant flush). The state tuple is
        #: ``(fingerprint, budget, expanded, hidden_phase_names)`` (design §7.3):
        #: * fingerprint — ``(phase_name, text, status, reason)`` per item, so a
        #:   phase rename or an item moving phases repaints;
        #: * budget — the same list renders a different number of rows when its
        #:   space changes;
        #: * expanded — a ``ctrl+t`` toggle changes what is shown with no store
        #:   change, so it must be in the guard or the toggle would no-op;
        #: * hidden — a phase crossing the auto-hide threshold changes the view
        #:   with no store change, same reason.
        self._shown: (
            tuple[tuple[tuple[str, str, str, str], ...], int, bool, frozenset[str]] | None
        ) = None
        #: Collapse/expand flag flipped by ``ctrl+t`` (:meth:`toggle_expanded`).
        #: Collapsed applies the walking-viewport cap and hides settled phases;
        #: expanded shows every phase and item (design §7.6).
        self._expanded: bool = False
        #: phase-name -> ``time.monotonic()`` at which it FIRST became fully
        #: settled. Drives the view-only auto-hide (§7.4): a phase settled for
        #: ``TODO_PHASE_HIDE_DELAY_S`` is hidden; regaining open work clears its
        #: entry so the clock restarts. Tests seed this directly as the settable
        #: seam the design calls for. NEVER mutated into a store change.
        self._settled_since: dict[str, float] = {}
        # Hidden until the first todo exists: an empty panel is not content.
        self.display = False

    def toggle_expanded(self) -> None:
        """Flip the collapse/expand flag and force a repaint on the next sync.

        Bound to ``ctrl+t`` via ``OperatorApp.action_toggle_todos``. Clearing
        ``_shown`` is what forces the repaint: the store did not move, so the
        equality guard would otherwise return early and the toggle would appear
        to do nothing. ``_refresh_band`` (the caller) re-syncs immediately.
        """
        self._expanded = not self._expanded
        self._shown = None

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
            phases = todo_items(session_id)
            # The phase name and the reason both ride IN the fingerprint: a
            # phase rename, an item moving between phases, or a re-block with a
            # new reason all change what the panel says, and a guard blind to
            # them would leave a stale paint on screen.
            fingerprint = tuple(
                (
                    phase["name"],
                    str(item.get("text", "")),
                    str(item.get("status", "pending")),
                    str(item.get("reason", "")),
                )
                for phase in phases
                for item in phase["items"]
            )
            hidden = self._hidden_phase_names(phases)
            # The BUDGET, the EXPANDED flag and the HIDDEN set all ride in the
            # guard beside the list, because what the panel paints is a function
            # of all four (design §7.3): the same list renders a different
            # number of rows when the rows available change (budget); a ctrl+t
            # toggle and a phase crossing the auto-hide threshold both change
            # the view with NO store change, so a guard blind to them would let
            # the toggle or the hide silently no-op. A body one row taller than
            # its budget is clipped, silently, by `Screen { overflow: hidden }`.
            budget = self._body_rows()
            state = (fingerprint, budget, self._expanded, hidden)
            if state == self._shown:
                return  # equality guard — nothing that affects the paint moved
            self._shown = state
            if not fingerprint:
                self.display = False
                return
            self.display = True
            self._body.update(self._build(phases, hidden))
        except Exception:
            self.display = False

    def _hidden_phase_names(self, phases: list[dict[str, Any]]) -> frozenset[str]:
        """Phase names the VIEW hides right now (never the store — design §7.4).

        Auto-hide safety invariant (omp ``#isTodoListSettled``,
        interactive-mode.ts:2093): a phase is hidden only after it has stayed
        CONTINUOUSLY, FULLY settled (:func:`_phase_settled` — all items closed,
        NEVER while a pending or blocked item remains) for
        ``TODO_PHASE_HIDE_DELAY_S``. ``_settled_since`` records when a phase
        first settled; regaining open work clears its entry so the clock
        restarts. This rides the existing 1 Hz poll — no dedicated timer, so no
        callback can fire against a torn-down panel (§7.5).

        A single implicit ``\"Todos\"`` phase (the flat back-compat case) is
        NEVER hidden: hiding the only phase would blank a panel that behaves
        exactly as it does today, and today a settled flat list stays visible.
        So auto-hide only engages once the store is genuinely multi-phase.
        """
        if len(phases) <= 1:
            # Byte-identical back-compat path: no auto-hide, restart every clock
            # so a later multi-phase list starts its timers fresh.
            self._settled_since.clear()
            return frozenset()
        now = time.monotonic()
        live = {phase["name"] for phase in phases}
        # Drop timers for phases that no longer exist (e.g. a re-init), so a
        # stale name can never linger in the hidden set.
        for name in [n for n in self._settled_since if n not in live]:
            self._settled_since.pop(name, None)
        for phase in phases:
            if _phase_settled(phase):
                self._settled_since.setdefault(phase["name"], now)
            else:
                # Regained open work — clear the timer so it restarts on the
                # next settle. This is the guard against hiding a live phase.
                self._settled_since.pop(phase["name"], None)
        return frozenset(
            name
            for name, since in self._settled_since.items()
            if now - since >= TODO_PHASE_HIDE_DELAY_S
        )

    # -- rendering -----------------------------------------------------------
    def _build(self, phases: list[dict[str, Any]], hidden: frozenset[str]) -> RenderableType:
        """Paint the panel from phases.

        Two shapes, ONE clip pass. A single implicit ``\"Todos\"`` phase renders
        HEADERLESS and byte-identical to the pre-phases panel (design §6.3, the
        back-compat guarantee the goldens assert) — that path is
        :meth:`_build_flat`. A genuinely multi-phase store renders phase headers
        with indented items and the collapse/auto-hide machinery
        (:meth:`_build_phased`). Both return a list of render rows this method
        then clips to width uniformly.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        # The single-phase back-compat path is chosen on phase COUNT alone, not
        # the name: any lone phase renders headerless, because a header on the
        # only phase is redundant with the root ``Todos`` line and would break
        # the byte-identical guarantee the existing goldens depend on.
        if len(phases) == 1:
            lines = self._build_flat(phases[0]["items"])
        else:
            lines = self._build_phased(phases, hidden)

        # The clip happens HERE, against a width, because nothing in the layout
        # supplies one (see :meth:`_row_cells`). Applied uniformly to headers,
        # items and the affordance so no row can push the band past the screen.
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

    def _build_flat(self, items: list[dict[str, Any]]) -> list[Text]:
        """The single-phase (implicit ``\"Todos\"``) render — HEADERLESS and
        byte-identical to the pre-phases panel (design §6.3).

        This is the back-compat contract: an existing caller that never mentions
        phases must see the exact panel it saw before. The row-budget/marker
        arithmetic here is unchanged from the pre-phases ``_build`` — do not fold
        it into the phased path, because that path spends rows on phase headers
        and an affordance line this one must not grow.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        rows = [
            (
                str(item.get("text", "")),
                str(item.get("status", "pending")),
                str(item.get("reason", "")),
            )
            for item in items
        ]

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
        for text, status, reason in visible:
            lines.append(self._item_row(text, status, reason))
        if marker:
            overflow = Text(no_wrap=True, overflow="ellipsis")
            # Counts what the reader cannot see, not what this loop dropped.
            overflow.append(f"… {len(rows) - len(visible)} more todos", style=dim)
            lines.append(overflow)
        return lines

    def _build_phased(self, phases: list[dict[str, Any]], hidden: frozenset[str]) -> list[Text]:
        """The multi-phase render: a root progression line, phase headers with
        indented items, and one trailing affordance line (design §6.2, §7.6).

        CRITICAL (§6.4): every phase header and item is flattened into a single
        render-row list BEFORE the ``cap``/marker/clip arithmetic runs, so a
        header counts toward the budget exactly like an item and can never push
        the composer off screen under ``Screen { overflow: hidden }``. The
        affordance line's hidden count counts ITEMS the reader cannot see, never
        rows (headers are chrome, not todos).
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))

        # Expanded reveals every phase (including auto-hidden settled ones — the
        # user asked to see everything); collapsed omits hidden phases first,
        # then walks from the active phase (§7.6). "no auto-hide" in expanded is
        # exactly what lets ctrl+t reveal a phase the collapsed view hid.
        if self._expanded:
            considered = list(phases)
        else:
            considered = [p for p in phases if p["name"] not in hidden]

        active_idx = _active_phase_index(considered)
        total = len(considered)

        # Root progression line — always shown, never counted in the item cap,
        # mirroring the single-phase header's status as the one non-negotiable
        # line. omp's ``Todos · i/n`` (interactive-mode.ts:2280): muted name, dim
        # progression, because stage progression is context, not the count the
        # panel exists to add.
        root = Text(no_wrap=True, overflow="ellipsis")
        root.append("Todos", style=muted)
        if total:
            root.append(f" · {active_idx + 1}/{total}", style=dim)

        # Items hidden from THIS view that the affordance line must confess.
        # Split done/open so an affordance can never label an open item "done" —
        # the same "no lie by arithmetic" rule the flat header follows. Hidden
        # settled phases and auto-hidden phases contribute done items; a
        # per-phase cap or the subsequent-stage cap contributes open ones.
        hidden_done = 0
        hidden_open = 0

        # Phases the auto-hide dropped from the collapsed view (their items are
        # all closed by the hideable invariant, so they count as done).
        if not self._expanded:
            for phase in phases:
                if phase["name"] in hidden:
                    hidden_done += len(phase["items"])

        # In collapsed mode, walk from the active phase for a bounded number of
        # phases; anything before the active one or beyond the cap is hidden.
        if self._expanded:
            phase_slice = considered
        else:
            phase_slice = considered[active_idx : active_idx + 1 + _SUBSEQUENT_STAGE_CAP]
            for phase in (
                considered[:active_idx] + considered[active_idx + 1 + _SUBSEQUENT_STAGE_CAP :]
            ):
                for item in phase["items"]:
                    if _is_closed(item):
                        hidden_done += 1
                    else:
                        hidden_open += 1

        # Flatten headers + items into ONE render-row list, tagging each row so
        # the budget can count dropped ITEMS (not headers) truthfully (§6.4.2).
        # Entry: (Text, is_item, is_open).
        body: list[tuple[Text, bool, bool]] = []
        for phase in phase_slice:
            items = phase["items"]
            done = sum(1 for item in items if item.get("status") in RESOLVED_STATUSES)
            body.append((self._phase_header_row(phase["name"], done, len(items)), False, False))
            if self._expanded:
                shown_items = items
            else:
                shown_items, phase_hidden_open = select_collapsed(items, _ACTIVE_TASK_CAP)
                hidden_open += phase_hidden_open
            for item in shown_items:
                row = self._item_row(
                    str(item.get("text", "")),
                    str(item.get("status", "pending")),
                    str(item.get("reason", "")),
                )
                # Indent added HERE, not in ``_item_row``: ``_item_row`` stays the
                # single mark authority the transcript receipt mirrors, so the
                # gutter is the panel's own presentation and never leaks into the
                # marks (design §6.2).
                indented = Text("  ", no_wrap=True, overflow="ellipsis")
                indented.append_text(row)
                body.append((indented, True, not _is_closed(item)))

        # Row budget. Root line takes one; the affordance line takes one more
        # whenever the screen can afford it. Whatever body rows do not fit are
        # dropped and confessed in the affordance's count — the affordance line
        # subsumes the flat panel's ``… N more`` marker in this path.
        room = max(1, self._body_rows() - 1)  # after the root line
        show_affordance = room > 1
        body_cap = room - 1 if show_affordance else room
        if len(body) > body_cap:
            for _text, is_item, is_open in body[body_cap:]:
                if not is_item:
                    continue
                if is_open:
                    hidden_open += 1
                else:
                    hidden_done += 1
            body = body[:body_cap]

        lines = [root]
        lines.extend(text for text, _is_item, _is_open in body)
        if show_affordance:
            lines.append(self._affordance_row(hidden_done, hidden_open))
        return lines

    def _phase_header_row(self, name: str, done: int, total: int) -> Text:
        """A phase header ``PhaseName · done/total`` — muted name, dim progress,
        the same treatment the single-phase ``Todos`` header uses and omp's
        non-active phase header (interactive-mode.ts:2259). ``done`` counts
        CLOSED items (``done``/``dropped``, ``RESOLVED_STATUSES``), matching the
        auto-hide's "fully settled" notion so a hidden phase reads ``n/n``."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append(strip_control_sequences(name), style=muted)
        header.append(f" · {done}/{total}", style=dim)
        return header

    def _affordance_row(self, hidden_done: int, hidden_open: int) -> Text:
        """The trailing control line (design §7.6).

        Collapsed: names what is hidden and how to see it. The prefix is
        truthful about done vs open so a capped-open list never reads as
        finished — ``+N done`` only when every hidden item is closed (the common
        case, hidden settled phases), ``+N more`` when any open item is hidden.
        Expanded: ``ctrl+t to collapse``, plus a ``+N more`` prefix if the row
        budget still had to drop items (expanded is bounded by ``_body_rows()``
        too, §7.6). Dim throughout: chrome, not a todo.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        row = Text(no_wrap=True, overflow="ellipsis")
        total_hidden = hidden_done + hidden_open
        if total_hidden:
            if hidden_open:
                row.append(f"+{total_hidden} more · ", style=dim)
            else:
                row.append(f"+{hidden_done} done · ", style=dim)
        row.append("ctrl+t to collapse" if self._expanded else "ctrl+t to expand", style=dim)
        return row

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
        has not been arranged yet and measures zero (see ``app.slot_rows``).
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
        one row); this only asks whether it is being spent, which is true
        whenever any slot is docked. Degrades to 0 for any host whose band is
        missing, the same posture as :meth:`_band_sibling_rows`.
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

        Measured where the sibling has been laid out and PREDICTED where it has
        not, through the same ``app.slot_rows`` the band's inset check uses —
        one answer to "how tall is that slot", so the two cannot disagree about
        the same frame.

        A raw ``outer_size`` read was wrong here for the reason it was wrong
        there: a sibling that has just been un-hidden measures zero until
        Textual re-arranges, and this runs at exactly that moment (a subagent
        starting while a todo list is up). Reading zero makes this panel budget
        for a band that is about to be several rows taller, so it paints too
        many rows and the dock overflows the screen until the next poll — the
        both-panels-docked case, which the single-panel fix did not reach.

        Imported lazily: the app imports this module, so a module-level import
        would close the cycle.
        """
        parent = self.parent
        if parent is None:
            return 0
        from local_operator.tui.app import slot_rows

        return sum(slot_rows(slot) for slot in parent.children if slot is not self and slot.display)

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
