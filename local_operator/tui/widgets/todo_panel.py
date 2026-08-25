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
from textual.containers import Container, VerticalScroll
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

#: Transcript rows kept visible ABOVE an expanded todo list. Expand is a
#: deliberate user toggle (``ctrl+t``), so the panel is allowed to take rows
#: FROM the transcript — the transcript is ``1fr`` and scrolls, the dock does
#: not — but it must never take ALL of them: a few rows of conversation stay on
#: screen so the expanded list reads as sitting BELOW the transcript rather than
#: as having replaced it. This is the expanded mirror of the collapsed budget's
#: ``_DOCK_ROWS`` reservation, and the reason the composer (``#input-shell``,
#: five rows counted in ``_DOCK_ROWS``) is still safe: expanded spends the
#: screen MINUS the same dock chrome MINUS this floor, never the composer's own
#: rows, so ``Screen { overflow: hidden }`` can never clip the input away.
_EXPANDED_TRANSCRIPT_FLOOR = 3

#: Hard ceiling on the expanded body's own painted rows, above which the body
#: SCROLLS instead of growing further (``#todo-body`` gains ``overflow-y: auto``
#: in expanded mode — see :meth:`TodoPanel._body_rows`). A list longer than the
#: screen can show at once must stay reachable without pushing the composer off
#: the bottom; capping the paint and scrolling the remainder is how every todo
#: is reachable while the dock's height stays bounded. Generous enough that a
#: normal-height terminal shows a typical multi-phase plan in FULL and only a
#: genuinely oversized list (40 items on a short terminal) ever scrolls.
_MAX_EXPANDED_ROWS = 200

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


def todo_items(session_id: str, transcript_directory: str | None = None) -> list[dict[str, Any]]:
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
    if not raw and transcript_directory:
        # A resumed/swept child has no live process store. Its own transcript is
        # authoritative; never consult the root session as a convenience.
        try:
            from local_operator.session.transcript import Transcript

            details = Transcript(transcript_directory).latest_custom("todo_snapshot") or {}
            candidate = details.get("items") or []
            raw = candidate if isinstance(candidate, list) else []
        except Exception:
            raw = []
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


class TodoAffordance(Static):
    """The collapse/expand control row, as its OWN clickable widget.

    Split out of the single body Static (defect 2) for the same reason
    ``SubagentRow`` is its own widget: only a widget can carry a ``:hover``
    rule and an ``on_click`` scoped to JUST this row. Folded into the list body,
    a hover would light the whole list and a click anywhere in it would toggle —
    the affordance is a button, so only the button hovers and only the button
    clicks. The hover ground and the pointer are the stylesheet's
    (``#todo-affordance`` / ``:hover``), the SAME ``$lo-overlay`` step
    ``SubagentRow:hover`` and ``ToolCard:hover`` use, because pointing at a row
    should always look the same.
    """

    def __init__(self) -> None:
        # `.band-body` gives it the dock's fill and one-cell rail, so it lines up
        # under the list rows rather than sitting on bare ground one cell left —
        # the same fix `SubagentPanel`'s rows needed (design round 12, D1/D5).
        super().__init__(classes="band-body", id="todo-affordance")

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        # `event.stop()` FIRST, before the toggle, so the click that expands the
        # list does not ALSO reach the transcript behind the dock and scroll it
        # — the band's mouse-isolation rule (AGENTS.md "Overlays float;
        # event.stop()"): one gesture must move one thing. The toggle then goes
        # through the panel's single source of truth so the key path and the
        # click path can never diverge.
        event.stop()
        panel = self.parent
        if isinstance(panel, TodoPanel):
            panel.request_toggle()


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
        # The list rows live in a Static INSIDE a scroll region, so the expanded
        # view can overflow the panel's height budget and stay reachable by
        # scrolling rather than silently clipping under `Screen { overflow:
        # hidden }` (defect 1's overflow case). The Static no longer carries
        # `.band-body`: the fill and the dock rail move to the scroll wrapper so
        # the rail is spent ONCE, not doubled by a nested padded child.
        self._body = Static(id="todo-body")
        self._scroll = VerticalScroll(self._body, classes="band-body", id="todo-scroll")
        # The scroll region must NOT take keyboard focus (U3). It is a status
        # surface, not an input: `VerticalScroll` is focusable by default, so a
        # click in the list body moved focus off the composer and INTO the
        # scroll — the app looked focused while every keystroke vanished into a
        # widget that does nothing with them, and the next typed message was
        # silently swallowed. This is the exact class of bug the app already
        # guards against for `TranscriptView` (app.py `_editor().focus()`, "the
        # app looked correctly focused while every keystroke went to a widget
        # that does nothing with them"). Dropping `can_focus` keeps the composer
        # holding focus through a body click; mouse-wheel scroll over the panel
        # still works (wheel does not require focus), and the keyboard path is
        # the app's `scroll_todos_*` bindings (U2), not focus-then-arrow.
        self._scroll.can_focus = False
        # The collapse/expand control is its OWN widget (defect 2): a `:hover`
        # rule and an `on_click` can then target JUST this row, so a click in the
        # list never toggles and only the affordance lights on hover. It sits
        # OUTSIDE the scroll region so it stays pinned while the list scrolls.
        self._affordance = TodoAffordance()
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
            tuple[
                str,
                str | None,
                tuple[tuple[str, str, str, str], ...],
                int,
                bool,
                frozenset[str],
            ]
            | None
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
        #: Rows the panel actually painted last sync (scroll body capped to its
        #: budget PLUS the pinned affordance row). ``predicted_rows`` answers
        #: from this so the band budgets the panel's REAL height in each state —
        #: an expanded list that scrolls occupies its capped height, not its full
        #: line count, and the two must agree or the band reflows on the first
        #: frame (see :meth:`predicted_rows`).
        self._painted_rows: int = 0
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

    def request_toggle(self) -> None:
        """Toggle from a click on the affordance, repainting the band at once.

        The single source of truth for "the affordance was activated": both the
        ``ctrl+t`` key path (``action_toggle_todos``) and the click path
        (:meth:`TodoAffordance.on_click`) land here, so the two can never
        diverge. It does what the key action does — flip the flag, then settle
        the band inset and row budget in the SAME tick the flag changed
        (``_refresh_band``) so the list never paints one frame at the wrong
        height. Imported lazily and guarded because the panel is reachable in
        reduced hosts that have no app.
        """
        self.toggle_expanded()
        app = getattr(self, "app", None)
        refresh = getattr(app, "_refresh_band", None)
        if callable(refresh):
            refresh()

    def scroll_expanded(self, *, down: bool) -> bool:
        """Scroll the expanded overflow by one page from the KEYBOARD (U2).

        Returns ``True`` when it moved the region, ``False`` when there was
        nothing to scroll (collapsed, or expanded but fully on screen). The app
        binds ``ctrl+down``/``ctrl+up`` here and reports the ``False`` to the
        user, because a key that silently does nothing reads as broken.

        The scroll region itself is non-focusable (U3), so a focus-then-arrow
        gesture cannot reach the overflow \u2014 the todos the footer says ``ctrl+t``
        reveals would otherwise be mouse-only. This drives the SAME
        ``VerticalScroll`` the wheel drives, so the two paths can never diverge,
        and it is a no-op unless the region actually overflows (``max_scroll_y``
        > 0), so it never fights the collapsed panel for the key.
        """
        if not self._expanded:
            return False
        scroll = self._scroll
        if scroll.max_scroll_y <= 0:
            return False
        scroll.scroll_page_down() if down else scroll.scroll_page_up()
        return True

    def compose(self):  # type: ignore[override]
        # Two children now: the scrollable list body, and the affordance row
        # pinned beneath it. `sync` updates both in one pass so the equality
        # guard still governs a single repaint.
        yield self._scroll
        yield self._affordance

    # -- sync -----------------------------------------------------------------
    def sync(
        self,
        session: Any,
        *,
        session_id: str | None = None,
        transcript_directory: str | None = None,
    ) -> None:
        """Re-read the selected session's store and repaint only on change.

        Called on the todo tool's ``tool_execution_end`` (immediate) AND on the
        app's 1 Hz job poll (the belt to that event's suspenders: a ``done``
        op that settles while the card is still painting still lands here).
        Never raises: a status surface must not be able to take the app down.
        """
        try:
            frontend = getattr(session, "frontend_state", None)
            selected_id = (
                session_id if session_id is not None else getattr(session, "session_id", "")
            )
            if frontend is not None and selected_id == getattr(session, "session_id", ""):
                phases = [phase.model_dump(mode="json") for phase in frontend.todos]
            else:
                # Historical child pages read their own durable transcript;
                # the live owner/follower session reads canonical state.
                phases = (
                    todo_items(selected_id or "", transcript_directory)
                    if transcript_directory is not None
                    else todo_items(selected_id or "")
                )
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
            # Session identity belongs in the guard even when both lists are
            # empty or byte-identical: retargeting must never retain another
            # session's panel state or delayed phase-hide clocks.
            state = (
                selected_id,
                transcript_directory,
                fingerprint,
                budget,
                self._expanded,
                hidden,
            )
            if state == self._shown:
                return  # equality guard — nothing that affects the paint moved
            self._shown = state
            if not fingerprint:
                self.display = False
                return
            self.display = True
            body, affordance = self._build(phases, hidden)
            self._body.update(body)
            # The affordance is a SEPARATE widget now (defect 2), updated in the
            # same sync so the equality guard still governs one repaint. It is
            # shown only in the phased path — the flat back-compat panel has no
            # collapse/expand control, so its marker stays a body line. Reserve
            # one row for it, so the scroll region below is capped to the rest of
            # the budget: `height: auto` + this cap is what makes the body FIT
            # its budget in the common case and SCROLL (never clip) when an
            # expanded list is longer than the screen can show (defect 1's
            # overflow case). Under `Screen { overflow: hidden }` a body one row
            # over budget would be silently swallowed, so the cap is load-bearing.
            affordance_rows = 1 if affordance is not None else 0
            self._affordance.display = affordance is not None
            if affordance is not None:
                self._affordance.update(affordance)
            self._scroll.styles.max_height = max(1, budget - affordance_rows)
            # In expanded mode the body can outrun its budget and the scroll
            # region absorbs the overflow; the panel's occupied height is then
            # the capped scroll plus the pinned affordance, which is what
            # `predicted_rows` must report so the band budgets the real height.
            body_lines = len(str(body).split("\n")) if str(body) else 0
            self._painted_rows = min(body_lines, max(1, budget - affordance_rows)) + affordance_rows
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
    def _build(
        self, phases: list[dict[str, Any]], hidden: frozenset[str]
    ) -> tuple[RenderableType, Text | None]:
        """Paint the panel from phases, returning ``(body, affordance)``.

        Two shapes, ONE clip pass. A single implicit ``\"Todos\"`` phase renders
        HEADERLESS and byte-identical to the pre-phases panel (design §6.3, the
        back-compat guarantee the goldens assert) — that path is
        :meth:`_build_flat` and has NO affordance control (its overflow marker
        stays a body line). A genuinely multi-phase store renders phase headers
        with indented items and returns the collapse/expand affordance SEPARATELY
        (:meth:`_build_phased`) so it can be mounted as its own clickable widget
        (defect 2). Both list-of-rows results and the affordance are clipped to
        width here, uniformly, so no row can push the band past the screen.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        # The headerless back-compat path is the IMPLICIT single phase only, so
        # the panel and the ``view`` receipt make the identical choice (design
        # §5.2 — one list spelled one way). ``builtin._todo_view_text`` gates its
        # headerless output on exactly this predicate
        # (``len == 1 and name == _IMPLICIT_PHASE``); gating on COUNT alone here
        # dropped a lone EXPLICITLY-named phase's header in the dock while the
        # receipt kept it, so the two surfaces disagreed (U5). The byte-identical
        # flat store still routes here because a legacy flat ``init`` coerces to
        # exactly one implicit ``"Todos"`` phase.
        if len(phases) == 1 and phases[0]["name"] == _IMPLICIT_PHASE:
            lines, affordance = self._build_flat(phases[0]["items"])
        else:
            lines, affordance = self._build_phased(phases, hidden)

        # EXPANDED item-visibility floor (U6). The affordance is pinned as its
        # OWN row OUTSIDE the scroll region, so at the floored budget (h=12 both
        # shapes, h=13 phased) it consumed the single row the first item would
        # occupy: expand painted the header + ``ctrl+t to collapse`` and ZERO
        # todo rows where collapsed showed one — the exact "expand shows nothing"
        # defect the feature exists to kill, resurfacing at the extreme floor.
        # Guarantee an item survives before the affordance claims its row,
        # mirroring collapsed's ``_fit_body`` floor. Only touches the tightest
        # budgets; every h>=14 keeps the affordance untouched.
        if self._expanded and affordance is not None:
            lines, affordance = self._guarantee_expanded_item(lines, affordance)

        # The clip happens HERE, against a width, because nothing in the layout
        # supplies one (see :meth:`_row_cells`). Applied uniformly to headers,
        # items and the affordance so no row can push the band past the screen.
        cells = self._row_cells()
        if cells:
            for line in [*lines, *([affordance] if affordance is not None else [])]:
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
        return Text("\n").join(lines), affordance

    def _build_flat(self, items: list[dict[str, Any]]) -> tuple[list[Text], Text | None]:
        """The single-phase (implicit ``\"Todos\"``) render — HEADERLESS and
        byte-identical to the pre-phases panel (design §6.3) when COLLAPSED.

        Returns ``(body_lines, affordance)`` to mirror :meth:`_build_phased`.

        COLLAPSED is the back-compat contract: an existing caller that never
        mentions phases must see the exact panel it saw before — header, capped
        items, and the ``… N more todos`` marker AS A BODY LINE, with NO
        affordance widget (``affordance = None``). The row-budget/marker
        arithmetic here is unchanged from the pre-phases ``_build`` — do not fold
        it into the phased path, because that path spends rows on phase headers
        and an affordance line this one must not grow. The ``test_band_panels``
        goldens assert this body content byte-for-byte.

        EXPANDED (M1/U1): ``ctrl+t`` on a flat list — the DEFAULT shape a
        ``todo init items=[...]`` produces — was a no-op: this method ignored
        ``self._expanded`` and capped at ``MAX_TODO_ROWS`` in both states, so the
        headline ``expand reveals the whole list`` fix never reached the common
        non-phased case, and no clickable control was ever mounted. The expanded
        branch mirrors ``_build_phased``: paint EVERY item (no marker), let the
        scroll region (capped to the expanded budget in :meth:`sync`) absorb any
        overflow, and return the ``ctrl+t to collapse`` affordance so the flat
        path gets the same discoverable, clickable button and a keyboard/mouse
        way back. Nothing is hidden in this state, so the affordance confesses
        nothing (``0, 0``).
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
        # EXPANDED: paint the full list, no marker, and hand back the collapse
        # affordance. The header is still the one non-negotiable line; every
        # item follows it and the scroll region (:meth:`sync`) absorbs anything
        # taller than the expanded budget, so no todo is dropped or clipped.
        if self._expanded:
            lines = [header]
            for text, status, reason in rows:
                lines.append(self._item_row(text, status, reason))
            return lines, self._affordance_row(0, 0)

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
        # Collapsed flat stays byte-identical: the marker is a BODY line and no
        # affordance widget is mounted (design §6.3, the back-compat goldens).
        return lines, None

    def _build_phased(
        self, phases: list[dict[str, Any]], hidden: frozenset[str]
    ) -> tuple[list[Text], Text | None]:
        """The multi-phase render: a root progression line, phase headers with
        indented items, and the collapse/expand affordance (design §6.2, §7.6).

        Returns ``(body_lines, affordance)``: the body rows the scroll region
        paints, and the affordance Text (or ``None`` on a terminal too short to
        afford its row) mounted as a SEPARATE clickable widget below the scroll
        (defect 2). The affordance's hidden count counts ITEMS the reader cannot
        see, never rows (headers are chrome, not todos).

        CRITICAL (§6.4): in COLLAPSED mode every phase header and item is
        flattened into one render-row list BEFORE the ``cap`` arithmetic runs, so
        a header counts toward the budget exactly like an item and can never push
        the composer off screen under ``Screen { overflow: hidden }``. In
        EXPANDED mode the body is painted in FULL and the scroll region (capped to
        the budget in :meth:`sync`) absorbs any overflow — the deliberate
        ``ctrl+t`` toggle is allowed to take rows from the transcript, so a long
        list stays reachable by scrolling rather than being clipped or relabelled
        (defect 1).
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

        # Root progression line — always shown, never counted in the item cap,
        # mirroring the single-phase header's status as the one non-negotiable
        # line. omp's ``Todos · i/n`` (interactive-mode.ts:2280): muted name, dim
        # progression, because stage progression is context, not the count the
        # panel exists to add.
        #
        # The ``stage`` label is load-bearing (D1/U1). Without it, ``Todos · 2/3``
        # wears the EXACT ``done/total`` grammar of the phase headers one line
        # below (``Auth · 0/2``), in the same column and ink — so a reader parses
        # the root as a completion fraction and sees "finished" while work
        # remains, the one thing a progress panel must never say. The root's
        # number is a phase POINTER, not a count; the word breaks the collision.
        # It is spelled over the ABSOLUTE phase set (``phases``), not the collapsed
        # ``considered`` view, so an auto-hidden settled phase cannot make three
        # stages read as two — the stage total is a fact about the plan, not
        # about what currently fits on screen. Multi-phase always has >= 2
        # phases (a lone implicit phase routes to ``_build_flat``), so the stage
        # line is present in every state, including all-settled-and-hidden where
        # the count used to collapse to a bare ``Todos`` (U6).
        active_stage = _active_phase_index(phases)
        total_stages = len(phases)
        root = Text(no_wrap=True, overflow="ellipsis")
        root.append("Todos", style=muted)
        if total_stages:
            root.append(f" · stage {active_stage + 1}/{total_stages}", style=dim)

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

        # EXPANDED: paint every row. The panel is allowed to grow to a generous
        # share of the screen (``_body_rows`` in expanded mode), and the scroll
        # region — capped to that budget in :meth:`sync` — absorbs anything
        # longer, so no item is ever dropped or clipped (defect 1). The affordance
        # is always shown here (it is the ONLY signal ``ctrl+t`` collapses again),
        # and it confesses nothing because nothing is hidden: a body that outruns
        # the budget SCROLLS, it is not truncated. ``show_affordance`` gates only
        # the collapsed short-terminal case below.
        if self._expanded:
            lines = [root, *(text for text, _is_item, _is_open in body)]
            return lines, self._affordance_row(0, 0)

        # COLLAPSED row budget. Root line takes one; the affordance takes one
        # more whenever the screen can afford it. Whatever body rows do not fit
        # are dropped and confessed in the affordance's count — the affordance
        # subsumes the flat panel's ``… N more`` marker in this path.
        room = max(1, self._body_rows() - 1)  # after the root line
        show_affordance = room > 1
        body_cap = room - 1 if show_affordance else room
        # Guarantee at least one ITEM survives (U2). On the shortest terminals
        # ``body_cap`` is 1 and the walking viewport puts a phase header at
        # ``body[0]``, so a naive ``body[:cap]`` kept the header and painted zero
        # todos — the panel read as empty though six existed, strictly worse than
        # the flat list it replaces. ``_fit_body`` drops the header before the
        # item, matching the flat path's ``room == 1 keeps the item`` floor.
        body, dropped = self._fit_body(body, body_cap)
        for _text, is_item, is_open in dropped:
            if not is_item:
                continue
            if is_open:
                hidden_open += 1
            else:
                hidden_done += 1

        lines = [root]
        lines.extend(text for text, _is_item, _is_open in body)
        affordance = self._affordance_row(hidden_done, hidden_open) if show_affordance else None
        return lines, affordance

    def _fit_body(
        self, body: list[tuple[Text, bool, bool]], cap: int
    ) -> tuple[list[tuple[Text, bool, bool]], list[tuple[Text, bool, bool]]]:
        """Clip the flattened header+item rows to ``cap`` while GUARANTEEING at
        least one item survives when any exists (U2).

        Returns ``(kept, dropped)``; ``dropped`` is what the affordance count
        must confess. A naive ``body[:cap]`` regressed the short terminal: with
        ``cap == 1`` the walking viewport puts a phase HEADER at ``body[0]``, so
        the slice kept the header and painted zero todos — the panel read as
        empty though six existed, strictly worse than the flat list it replaces
        (which keeps the item at the same height). When the slice is all chrome
        but items exist below the fold, trade the lowest-priority kept header for
        the first dropped item, so a real todo is always visible. This is the
        phased mirror of ``_build_flat``'s ``room == 1 keeps the item`` floor:
        an item is worth more than a header whose count the root line implies.
        """
        if cap <= 0:
            return [], list(body)
        kept = body[:cap]
        dropped = body[cap:]
        if not any(is_item for _t, is_item, _o in kept) and any(
            is_item for _t, is_item, _o in dropped
        ):
            first_item = next(i for i, (_t, is_item, _o) in enumerate(dropped) if is_item)
            item = dropped[first_item]
            # Evict the trailing kept row (a header — the slice is all headers)
            # to make room, and hand it to ``dropped`` so the count stays whole.
            evicted = kept[-1]
            kept = [*kept[:-1], item]
            dropped = [evicted, *dropped[:first_item], *dropped[first_item + 1 :]]
        # Suppress a DANGLING TRAILING HEADER (D1). The walking-viewport slice
        # can admit the next phase's HEADER at ``body[cap-1]`` but run out of
        # budget before any of its items, so the phase renders as an empty group
        # (``Validation · 1/6`` with nothing beneath it, then ``+N more``) — a
        # header with no children is visual noise the original defect reports
        # were about. Drop any trailing kept row that is a header: its ``i/n``
        # count is already implied by the root stage line and the affordance's
        # ``+N more``, and the header's items (all in ``dropped``) are still
        # confessed by that count, so the hidden total stays honest. Never
        # strips past the last item — the floor above already guarantees one
        # survives, so this cannot empty the panel.
        while kept and not kept[-1][1]:  # kept[-1] is a header (is_item False)
            dropped = [kept[-1], *dropped]
            kept = kept[:-1]
        return kept, dropped

    def _guarantee_expanded_item(
        self, lines: list[Text], affordance: Text
    ) -> tuple[list[Text], Text | None]:
        """Ensure an ITEM row is on screen when expanded at the floored budget (U6).

        The affordance is pinned as its OWN row OUTSIDE the scroll region, so the
        scroll viewport gets ``budget - 1`` rows (:meth:`sync`). At the collapsed
        floor (h=12 both shapes, h=13 phased) that one row went to the affordance
        and the first item fell just below the fold: expand painted the header +
        ``ctrl+t to collapse`` and ZERO todo rows where collapsed showed one — the
        "expand shows nothing" defect returning at the extreme floor. Collapsed
        never hits this because its marker is a body line, not a pinned widget, so
        its scroll region is one row taller.

        The fix mirrors collapsed's ``_fit_body`` floor — an item is worth more
        than the redundant chrome — with two levers, cheapest first:

        1. DROP THE AFFORDANCE row. Its hint is implied (the header names the
           list, ``ctrl+t`` still collapses, the wheel/``ctrl+down`` still reach
           the rest), so handing its row back to the scroll is a pure win when it
           is what buries the item.
        2. At the very floor, leading CHROME (the root line plus a phase header)
           can still push the first item below a two-row viewport. Keep the root
           line — the one non-negotiable total — and drop the chrome line(s)
           between it and the first item until the item fits. The dropped phase
           header's count is implied by the root ``stage`` line, and the whole
           list is one ``ctrl+down`` away.

        Only ever touches the tightest budgets: every h>=14 has room for an item
        beneath the affordance, so the guard returns untouched and the affordance
        stays. Returns ``(lines, affordance-or-None)``.
        """
        budget = self._body_rows()

        def _first_item_idx(rows: list[Text]) -> int | None:
            for i, row in enumerate(rows):
                # Item rows carry the tool's ``- [`` mark; headers/root do not.
                if "- [" in row.plain:
                    return i
            return None

        first = _first_item_idx(lines)
        if first is None:
            return lines, affordance  # nothing to keep visible

        # Viewport = rows the scroll shows before any scroll, matching sync's
        # ``max_height = budget - affordance_rows``.
        if first < max(1, budget - 1):
            return lines, affordance  # item already visible with the affordance kept

        # Lever 1: give the affordance's row back to the scroll.
        if first < max(1, budget):
            return lines, None

        # Lever 2: trim leading chrome (never the root at index 0) until the
        # first item fits the affordance-free viewport.
        lines = list(lines)
        viewport = max(1, budget)
        while first >= viewport and first > 1:
            del lines[first - 1]
            first -= 1
        return lines, None

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
        Expanded: a bare ``ctrl+t to collapse`` — the expanded view hides
        NOTHING (the scroll region reaches an over-long list), so it is called
        with ``0, 0`` and never carries a hidden-count prefix. This corrects the
        original design's "expanded is bounded by ``_body_rows()`` too": expanded
        genuinely reveals every todo (defect 1), so there is no hidden remainder
        left to confess.

        The hidden-count prefix stays ``dim`` (chrome — a running total), but the
        ``ctrl+t`` hotkey token steps up to ``muted`` (D3/U3): fully ``dim`` it is
        4.18:1 on the band's ground, and this line is the ONLY signal the toggle
        exists, so a user who never squints at it never learns the panel expands.
        ``muted`` is 7.93:1 — the same call ``_item_row`` makes for the one row
        that asks for action (the blocked reason).
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        row = Text(no_wrap=True, overflow="ellipsis")
        total_hidden = hidden_done + hidden_open
        if total_hidden:
            if hidden_open:
                row.append(f"+{total_hidden} more · ", style=dim)
            else:
                row.append(f"+{hidden_done} done · ", style=dim)
        row.append("ctrl+t to collapse" if self._expanded else "ctrl+t to expand", style=muted)
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

        Answered from ``_painted_rows`` (the scroll body CAPPED to its budget
        plus the pinned affordance), NOT the raw body line count: an expanded
        list longer than its budget is SCROLLED, so it occupies the capped height,
        not its full line count. Reading the full count would over-report the
        panel's height and the band would budget for rows the scroll region never
        shows — the very mis-budget/reflow the state guard exists to avoid. Falls
        back to the budget before the first paint (``_painted_rows`` still zero).
        """
        if self._painted_rows > 0:
            return max(1, self._painted_rows)
        return max(1, self._body_rows())

    def _body_rows(self) -> int:
        """Rows this paint may fill — header, items and any overflow marker.

        Two ceilings, because the two states answer different questions.

        COLLAPSED (unchanged): the panel is chrome above the composer, kept SHORT
        so a long plan never pushes the input down. Ceiling is a full list plus
        its two chrome rows (``MAX_TODO_ROWS + 2``); the walking viewport hides
        the rest behind the affordance's count.

        EXPANDED (defect 1): the user asked, via ``ctrl+t``, to SEE the whole
        list, so the panel grows to a generous share of the screen and takes rows
        from the transcript — which is ``1fr`` and scrolls, so it can yield them,
        unlike the composer. The share is the screen MINUS the dock chrome
        (:data:`_DOCK_ROWS`, which already reserves the composer's five rows and
        the slot rhythm) MINUS a small transcript floor
        (:data:`_EXPANDED_TRANSCRIPT_FLOOR`, so a few rows of conversation stay
        on screen and the list reads as sitting below it) MINUS the band's own
        inset and any sibling slot. That budget is what a normal-height terminal
        (30-45 rows) needs to show a ~16-item multi-phase list in FULL; a list
        longer still SCROLLS within it (the scroll region in :meth:`sync`), never
        clips. Capped at :data:`_MAX_EXPANDED_ROWS` so the arithmetic cannot run
        away on an enormous virtual screen.

        THE EXPANDED-FLOOR RULE (U1): expand must never show FEWER todos than
        collapsed at the same height. Below ~24 rows the transcript floor drove
        the grown share BELOW the collapsed budget, so ``ctrl+t`` shrank the
        panel to a 1-row porthole into a list the collapsed view had shown three
        rows of — the exact ``expand is a no-op / makes it worse`` defect this
        PR exists to kill, returning on the common short terminal. So the
        expanded budget is floored at the collapsed budget: ``max(collapsed,
        grown)``. On a genuinely short screen the two are equal and the scroll
        region (capped to the same budget in :meth:`sync`) absorbs the rest, so
        expanded shows AT LEAST what collapsed showed and reaches the remainder
        by scrolling rather than by shrinking.

        Both states subtract what the band's other slot is already spending: the
        subagent panel shares this band, and a budget blind to it would put the
        upward clip back the moment a subagent runs on a short terminal. Floor is
        :data:`_MIN_BODY_ROWS` in both.
        """
        collapsed_ceiling = MAX_TODO_ROWS + 2
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            return _MAX_EXPANDED_ROWS if self._expanded else collapsed_ceiling
        if screen_height <= 0:
            return _MAX_EXPANDED_ROWS if self._expanded else collapsed_ceiling
        dock = _DOCK_ROWS + self._band_inset_rows() + self._band_sibling_rows()
        # The collapsed budget is the floor for BOTH states (see the
        # expanded-floor rule above): expanded may only ever GROW the panel.
        collapsed_budget = max(_MIN_BODY_ROWS, min(collapsed_ceiling, screen_height - dock))
        if self._expanded:
            grown = screen_height - dock - _EXPANDED_TRANSCRIPT_FLOOR
            grown = max(_MIN_BODY_ROWS, min(_MAX_EXPANDED_ROWS, grown))
            return max(collapsed_budget, grown)
        return collapsed_budget

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
