"""A viewport-sized session list, distinct from the conversation it navigates.

One widget renders only the visible row window: a growing session directory
must not grow the Textual DOM. Cursor, current session and requested session
are separate identities, so a pending attach cannot pretend to have succeeded.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Self

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.geometry import Region
from textual.message import Message
from textual.reactive import Reactive
from textual.strip import Strip
from textual.timer import Timer
from textual.widget import Widget

from local_operator.resume import (
    UNNAMED_DEVICE,
    UNTITLED_CONVERSATION,
    format_age,
    peer_reason_words,
)
from local_operator.session.preview import AGENT_OPENED_MARK, opener_role
from local_operator.tui import theme as theme_mod
from local_operator.tui.animation import BLURRED_SPINNER_INTERVAL_S, animation_focused
from local_operator.tui.composer_focus import composer_may_take_focus, focus_is_claimed
from local_operator.tui.session_catalog import CatalogEntry, rank_entries
from local_operator.tui.terminal_title import SPINNER_FRAMES
from local_operator.tui.widgets.ask_queue import ASK_MARKER
from local_operator.tui.widgets.session_picker import (
    AIDA_MARKER,
    COMPLETION_MARKERS,
    aida_row_id,
    row_state_mark,
)
from local_operator.tui.widgets.tool_card import truncate_cells

SIDEBAR_WIDTH = 30

#: The pin cell's two glyphs: the durable mark (accent ink, on a pinned row in
#: every state) and the outline a mouse user sees while the pointer rests on an
#: unpinned row. The cell exists to be FOUND — the issue this slice serves is
#: that pinning was discoverable only through documentation (issue #1357).
PIN_MARK = "★"
PIN_HOVER = "☆"

#: Cells the pin cell owns at the head of every entry row. Two, so the glyph
#: carries its own separator and the cell is a mouse target the width of every
#: other slot. A NEW column rather than a reuse of the cursor-prefix slot:
#: that slot's cell 0 is shared between the caret and the star — only one of
#: them can show — so a press there has always meant "open the row" and must
#: keep meaning it. See `on_mouse_down`.
PIN_CELL_WIDTH = 2

#: Cells every entry row spends before its title: the pin cell, the caret
#: cell, the locality mark's cell, and the state mark plus its trailing space.
#: `render`'s title arithmetic and the tests that pin the row geometry both
#: read this, so the two cannot drift apart.
_ROW_PREFIX_CELLS = 4 + PIN_CELL_WIDTH

#: The state mark's column — the cell `row_state_mark`'s urgency ladder owns,
#: and the one cell `_advance_spinner` patches in place on its fast path.
_ROW_MARK_COLUMN = 2 + PIN_CELL_WIDTH

#: The locality mark's column. A peer section's heading is indented to it so
#: the heading's `⇄` stacks with the rows it governs (see `_peer_heading_text`).
_LOCALITY_COLUMN = PIN_CELL_WIDTH + 1

#: The widest the list may grow on a roomy terminal, in the same units as
#: :data:`SIDEBAR_WIDTH` (content, before the gutter is added).
#:
#: At the base 30 the title column is about 20 cells once the caret, the state
#: mark and the age have taken theirs, and real conversation names are longer
#: than that: the reporter's own frame showed "Article-search-svc s…",
#: "Add Flavia's Adverse Med…", "Update Provider Onbo…" — eleven of twelve rows
#: ellipsized, several of them cut before the word that distinguishes them from
#: their neighbour. A list whose whole job is "recognise your own conversation"
#: cannot do it at that budget, and a wide terminal has the cells to spare.
#:
#: 44 gives about 34 title cells, which fits the great majority of generated
#: titles outright. It is a CAP rather than a target: growth is paid for out of
#: surplus only (see :data:`SIDEBAR_MAIN_COMFORT_WIDTH`), so a narrow terminal
#: is unaffected and nothing here can squeeze the conversation.
SIDEBAR_MAX_WIDTH = 44

#: The remedy clause a wedged row's tooltip carries, after the backend's own
#: state sentence. The STATE WORDS are not the client's to change —
#: ``session/catalog.py`` owns them (``WEDGED_STATUS``, and with it the heartbeat
#: age), and a second spelling here would be the drift the surface map forbids.
#: The remedy is different in kind: it names an affordance, and each client has
#: its own. This surface's user is at a terminal, where ``lop stop`` is the rung
#: that asks a runtime to stop and signals it only if it will not answer — the
#: patient option, which matters because a stale beat is usually a busy session
#: rather than a dead one, and this whole row exists because a WORKING session
#: can produce it. "If it stays silent" is a condition, not a forecast:
#: ``registry.classify`` is explicit that a stale beat establishes no diagnosis,
#: and this codebase has already removed one promise of recovery for exactly
#: that reason.
WEDGED_REMEDY = "lop stop if it stays silent"

#: The main lane's COMFORT width, which growth may not eat into.
#:
#: Distinct from :data:`SIDEBAR_MAIN_MIN_WIDTH`, and the distinction is what
#: makes widening safe. That constant is a HARD floor — below it the drawer
#: stops displacing the conversation and becomes an overlay. This one is the
#: point past which extra columns are surplus: the sidebar grows only into
#: terminal width beyond it, so the transcript keeps a comfortable measure at
#: every size and the list simply stays at its base width when there is nothing
#: spare. 80 is the conventional readable measure and the width the transcript's
#: own wrapping is tuned around.
#:
#: Measured against the TERMINAL, not against the conversation widget, and the
#: two differ by :data:`APP_SCREEN_INSET` (code review round 1, M1: the lane
#: was measured at 78 cells where a naive reading of this constant promises
#: 80). The inset is subtracted explicitly in :func:`sidebar_content_width` so
#: the guarantee this constant names is the one the layout actually delivers.
SIDEBAR_MAIN_COMFORT_WIDTH = 80

#: Cells the app's own chrome takes out of the terminal before any lane is
#: measured — the outer inset the stylesheet puts on the workspace (one cell
#: each side). Named rather than inlined because it is the difference between
#: "terminal width" and "width the lanes divide up", and a growth rule that
#: forgets it over-promises by exactly this much.
APP_SCREEN_INSET = 2

#: How long a requested row waits before its state mark becomes a spinner.
#: The tint itself is immediate — it is the acknowledgement that the click
#: landed. The spinner is only for a switch that is genuinely taking a while,
#: and in practice a warm switch completes well inside this window, so the
#: glyph should almost never be seen: a warm sample that shows it is a
#: regression, not a feature working.
REQUESTED_SPINNER_DELAY_S = 0.15

#: How long a `ctrl+o` jump stays armed waiting for the rows it asked for.
#:
#: The chord cannot land its own jump from a layer-OFF sidebar: `load_catalog`
#: filters the hidden population out at the load site, so the rows only exist
#: once the re-poll the chord posts comes back. The intent therefore has to
#: outlive the keystroke — but only just. Sized UNDER the app's 2 s catalog
#: tick (`app.py`, `_sidebar_timer`) so a routine poll can never be the one
#: that fires it: what lands the jump is the re-poll this chord triggered,
#: which is a local store scan and returns in milliseconds. Past this the
#: intent is dropped, because a layer that came back empty is a store with no
#: subagent runs in it, and a delegated run starting later must not yank the
#: cursor out from under whatever the user is doing by then.
PENDING_SUBAGENT_JUMP_S = 1.0

#: How long an ARRIVAL reveal stays armed waiting for the row it is for.
#:
#: `set_current` fires on every edge where the app takes a session as its own —
#: boot, `/new`, `/resume`, `/new remote <peer>`, a notification click, a remote
#: takeover — and the list can be up to one poll behind that edge. `/new remote`
#: is the shape that makes the lag real rather than theoretical: the session is
#: minted by the PEER inside the keystroke, so the row the app is now standing in
#: cannot be in a snapshot a poll took before it, and a reveal that only fires
#: when the row is already in `entries` would silently do nothing at exactly the
#: moment it matters most.
#:
#: Sized ABOVE the app's 2 s catalog tick (`app.py`, `_sidebar_timer`), which is
#: the opposite of the `ctrl+o` arm above. That arm lands on the re-poll its own
#: chord triggered (a local store scan, milliseconds); this one lands on the next
#: TICK, because the row is delivered by the poll rather than by any read this
#: gesture started — the peer catalogue is pre-warmed by the create
#: (`_adopt_created_remote_session`'s `ttl_s=0` read), so one tick carries it and
#: two and a half ticks mean a slow read cannot lose the arrival. Past the
#: deadline the intent is dropped, because a row that turns up long after the
#: user has moved on is not this arrival's business and must not yank the caret.
PENDING_ARRIVAL_REVEAL_S = 5.0

#: Blank cells between the list and the conversation it sits beside.
#:
#: The list's right-hand age column ("6m", "23h", "1d") ended one cell from the
#: transcript's first character, which read as two columns jammed together
#: rather than two regions — the user's report was that it "looks overcrowded".
#: Three cells is what actually separates them at a glance; two still reads as
#: tight against a full-bleed transcript.
#:
#: ADDED TO THE WIDTH rather than taken out of the content, because the content
#: has none to give: at 28 cells the title column is already 20 after the
#: cursor, state mark and age, and real titles ellipsize there today. Spending
#: the gutter from that budget would have paid for whitespace with the one
#: thing the list exists to show. The main lane can afford it — at 100 columns
#: it keeps 65, above the 60-column floor `SIDEBAR_MAIN_MIN_WIDTH` sets, and
#: below that threshold the drawer is an overlay that does not displace the
#: conversation at all.
#:
#: Whitespace, not a rule: the chrome is borderless, so the gap IS the
#: separator (see the stylesheet's own note on the app's outer inset).
SIDEBAR_GUTTER = 3

#: Below this the gutter is surrendered before the list narrows further. A
#: squeezed terminal needs the cells for a legible title far more than it needs
#: the separation the gutter buys.
SIDEBAR_MIN_CONTENT_WIDTH = 24
SIDEBAR_MAIN_MIN_WIDTH = 60


def sidebar_content_width(terminal_width: int) -> int:
    """How wide the list's CONTENT should be in a terminal this wide.

    One pure function so the width the app docks and the width any test or
    capture reasons about are the same number, derived the same way — the
    layout used to compute it inline, which is fine until a second caller needs
    to predict it.

    The rule, in one sentence: **start at the base width and spend only
    surplus.** Surplus is whatever the terminal has beyond the app's own inset,
    the base sidebar, its gutter and a comfortable main lane
    (:data:`SIDEBAR_MAIN_COMFORT_WIDTH`); the list takes it up to
    :data:`SIDEBAR_MAX_WIDTH` and never more. Consequences worth stating
    because they are the safety argument:

    * A terminal at or below the comfort threshold gets exactly today's
      layout — this cannot regress a narrow window, and the overlay behaviour
      at genuinely small sizes is untouched.
    * The conversation never drops below :data:`SIDEBAR_MAIN_COMFORT_WIDTH`
      *because of growth*: the sidebar is only ever handed columns the
      conversation did not need. Below the threshold the lane can of course be
      narrower — that is the terminal being small, not the list taking
      anything.
    * Growth is monotonic in terminal width, so dragging a window wider never
      makes the list narrower.

    :data:`APP_SCREEN_INSET` is subtracted because the lanes divide the
    terminal MINUS the app's outer inset, not the terminal. Omitting it made
    the promise wrong by two cells at every size where growth is active (code
    review round 1, M1) — a real measurement of 78 against a documented 80,
    which is the kind of quiet drift that makes a stated invariant untrustable
    even when nothing visibly breaks.
    """
    reserved = APP_SCREEN_INSET + SIDEBAR_WIDTH + SIDEBAR_GUTTER + SIDEBAR_MAIN_COMFORT_WIDTH
    surplus = terminal_width - reserved
    if surplus <= 0:
        return SIDEBAR_WIDTH
    return min(SIDEBAR_MAX_WIDTH, SIDEBAR_WIDTH + surplus)


#: The list's own spinner cadence, deliberately slower than
#: :data:`terminal_title.SPINNER_INTERVAL_S` that the band and the panels use.
#: The band animates ONE glyph; this list repaints every visible row per tick,
#: so the same nominal rate costs ~38x more terminal output for the same
#: information. At 0.15s the eight-frame cycle still reads as clearly alive
#: (1.2s) while writing ~1.9x less, and it does not visibly disagree with the
#: band because the rate a throttled surface ACHIEVES is well under its
#: nominal one either way. Local rather than a change to the shared constant:
#: the other surfaces read that one and are not paying this cost.
SIDEBAR_SPINNER_INTERVAL_S = 0.15


def _strip_dangling_separator(title: str) -> str:
    """Drop a ``·`` left stranded at the end of a truncated ``label · role``.

    ``truncate_cells`` cuts on a cell boundary, so a sub row whose role does
    not fit can land on the separator and render ``label ·…`` — which reads as
    a rendering fault rather than as an ellipsis. Only the separator is
    removed; the ellipsis stays, because the title genuinely is cut.
    """
    for suffix in ("·…", "· …"):
        if title.endswith(suffix):
            return title[: -len(suffix)].rstrip() + "…"
    return title


#: Section RANK to the name a TIER header row carries. One mapping so
#: `_display_rows` and `render` can never disagree about which sections exist.
#:
#: The PEER sections are absent from this table on purpose: their rank is shared
#: and their heading is per-device, so `_section_key` builds both and `render`
#: reads the heading back out of the row kind. Everything else here is unchanged,
#: which is what keeps a device with no peers painting byte-identically.
_SECTION_NAMES = {0: "pinned", 1: "active", 2: "previous", 4: "subagent"}

#: The rank a PEER's rows take: after `previous`, before `subagent`. Another
#: device's sessions are colder than anything you are doing here and warmer than
#: an expired agent run (``mesh-ui.md`` §1.3 decision 1). This is why
#: ``subagent`` moved from 3 to 4: the ranks are an ORDER, and inserting a tier
#: between two of them means renumbering the one below rather than sharing a
#: number with it — two tiers on one rank would sort through each other.
_SECTION_PEER_RANK = 3


def _peer_heading_text(label: str, reachable: bool) -> str:
    """The ONE spelling of a peer section's heading, for both of its sources.

    Split out because the heading is now DERIVED from two different things: a
    peer's rows (which carry the device's label and its reachability) and a
    peer that answered NOTHING (``SessionSidebar._silent_peer_tiers``, which
    carries the relay's own report). A second ``f" ⇄ …"`` beside this one is how
    the two would drift into two vocabularies for one state — and the reader has
    to be able to tell a live section from a silent one by that suffix alone.

    The indent aligns the heading's `⇄` with the rows' locality cell
    (`_LOCALITY_COLUMN`), the same property the round-1 fix installed when the
    mark was one cell in: the heading has no caret and no pin, and a heading
    whose glyph starts at a different x from the cells it governs reads as a
    separate column rather than a label for the block under it.
    """
    return f"{' ' * _LOCALITY_COLUMN}⇄ {label}" + ("" if reachable else " (unreachable)")


#: Prefix a peer section's heading carries in its row kind, so `render` can tell
#: a device's heading from the four tier names WITHOUT the heading having to be a
#: key in a table (`header:peer:⇄ damian-mbp`). The label itself is derived once,
#: by `_section_key`, and read back here.
_PEER_HEADER_PREFIX = "peer:"


class SessionSidebar(Widget, can_focus=True):
    #: A pointer press on this list moves the keyboard to it — a row, the pin
    #: cell and the panel's dead space alike — and the list wears the shipped
    #: focus ground. Three rules bound that: the two from the issue #1357
    #: decision (`### lopdev — design decision: click-to-focus`, recorded on
    #: the issue), plus the footer-chip exemption reconciled with #1841:
    #:
    #: * the press is refused while a hard claimant holds the keyboard — a live
    #:   approval or ask, the aside, a full-page mode or pushed screen, a
    #:   read-only composer. Those surfaces need the keys they hold, so a press
    #:   may act on the row but must not take them (G7); the guard lives in
    #:   :meth:`focus_on_click`, which Textual's click-to-focus walk consults
    #:   before any handler here runs;
    #: * in the narrow drawer placement a row press CLOSES the panel and the
    #:   keyboard goes back to the composer (G8/T1n), so a closed panel is never
    #:   left holding the keyboard — a hidden widget owning it is the "typed
    #:   input going nowhere" state arriving by another route;
    #: * the footer chip's cells are EXEMPT: the chip is a control with its own
    #:   action (#1841), so its press toggles the ⌥ layer and the keyboard
    #:   stays where it was — one cell off the chip is a normal press on the
    #:   panel and focuses the list.
    #:
    #: What makes the press safe is no longer "the list never gets the
    #: keyboard". The composer-focus change forbade the press for a real reason
    #: — the walk left the keyboard on a widget with no text input, and every
    #: typed key went nowhere (measured then at 120x40: four typed keys, a
    #: 0-cell frame delta, the draft untouched) — but by the time this was
    #: decided the list ALREADY had a keyboard mode (`f9`, `/sidebar focus`)
    #: and text typed into it was already swallowed (measured: `f9` then `A`
    #: left `editor.text` empty, with the keys still on the list). The safety
    #: property therefore moved to the widget: :meth:`on_key` and
    #: :meth:`on_paste` hand the first printable character (or paste) to the
    #: composer, and the composer takes the keyboard back — **however the
    #: list's keyboard mode was entered** (press, `f9`, `/sidebar focus`).
    #: One exception: while a hard claimant holds the keyboard the key stays
    #: with the claimant (the refusal in :meth:`on_key`) — reachable by `f9`
    #: alone, since a press is refused at the walk — and the key there is
    #: inert rather than typed. That corner is pre-existing and frozen by the
    #: decision, not part of this contract.
    #:
    #: ``can_focus`` stays ``True`` and the list keeps its arrows/``enter``/
    #: cursor tint unchanged; `escape`/`f9` still leave the mode. The one
    #: shipped behaviour this decision changes is called out on its PR: pressing
    #: the already-attached session's row keeps the keyboard on the list rather
    #: than returning it to the composer — every row press behaves alike — and
    #: typing-home makes the cost brief, because the first keystroke is typed
    #: rather than discarded.
    FOCUS_ON_CLICK = True

    def focus_on_click(self) -> bool:
        """Textual's click-to-focus walk for this widget, gated by the claims.

        ``Screen._forward_event`` consults this on every ``MouseDown`` before
        any handler here runs, and focuses the widget when it answers ``True``.
        The base answer is :data:`FOCUS_ON_CLICK` minus two exceptions:

        * a hard claimant keeps its keys: the ONE hard-claim predicate
          (``composer_focus.focus_is_claimed``) refuses the focus move for a
          live approval/ask, the aside, a pushed screen, a full-page mode or a
          read-only composer (issue #1357 decision, G7), so the press may act
          on the row while the keyboard does not move;
        * the footer chip is EXEMPT — the reconciliation with the chip slice
          (#1841). The chip is a control with its OWN action (toggle the ⌥
          layer), so a press on its cells must not also run the panel's
          behaviour; that slice's test pins that a chip press moves no
          keyboard. A press one cell OFF the chip is a normal press on the
          panel and does focus the list.

        The walk is handed no event coordinates, so the chip test reads
        ``App.mouse_position`` — the press position set by every route before
        forwarding: ``App.on_event`` for terminal input, the pilot's
        ``_post_mouse_events``, and the chip slice's own footer gesture. The
        cells come from :meth:`_chip_hit`, the chip's one hit zone, never a
        copy of it — so the exemption covers exactly what the chip's press
        covers, and nothing widens silently. The hit is resolved HERE, under
        the ladder as painted when the press lands, and stashed for
        ``on_mouse_down`` (``_press_chip_hit``): the walk's own focus move
        repaints the footer — the focused rungs are shorter, so the chip's
        cells move with them — and a handler that re-tested the cells
        afterwards would reinterpret the gesture against cells the user never
        saw.

        Soft on the other side, deliberately: this gate protects claims that
        need their keys, not a claim on who presses where. The list's own claim
        in ``_focus_is_claimed`` stays SOFT (design round D2), which is what
        lets the composer's own chrome take the keyboard back from it.
        """
        position = getattr(self.app, "mouse_position", None)
        if position is None:
            # No press position to resolve against (a stripped harness): the
            # chip cannot be claimed, so the stash is cleared and the focus
            # decision stands — the pre-reconciliation behaviour.
            self._press_chip_hit = False
            return super().focus_on_click() and not focus_is_claimed(self.app)
        self._press_chip_hit = self._chip_hit(
            position.x - self.region.x, position.y - self.region.y
        )
        if self._press_chip_hit:
            return False
        return super().focus_on_click() and not focus_is_claimed(self.app)

    #: Textual re-renders a widget on every pointer move to look for link
    #: spans (``Widget.watch_hover_style``, whose own comment notes it fires
    #: "even when there are no links"). That repaint is paid INLINE on the
    #: input thread, before the handler runs, and it fires for every widget
    #: under the pointer anywhere in the app — measured at 4.03ms and 9KB of
    #: terminal output per hover event here, against 0.90ms with it off. This
    #: list renders no links (its rows are `Text` spans with colour and bold
    #: only), so nothing is given up; if a link is ever added here, this must
    #: go back to True or its hover highlight will silently stop working.
    auto_links: Reactive[bool] = Reactive(False)
    BINDINGS = [
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
        Binding("pageup", "page(-1)", show=False),
        Binding("pagedown", "page(1)", show=False),
        Binding("home", "edge(False)", show=False),
        Binding("end", "edge(True)", show=False),
        Binding("enter", "select", show=False),
        Binding("escape", "leave", show=False),
        # Sidebar-scoped on purpose. `ctrl+a` and `ctrl+o` are BOTH in
        # `keymap.COMPOSER_KEYS` (keymap.py:212, :210) — `ctrl+a` is the
        # composer's line-start and `ctrl+o` expands a collapsed paste. They are
        # safe HERE because in F9 mode the sidebar owns the focus chain and the
        # composer is not in it. Promoted to app level, non-priority, they would
        # silently lose to TextArea and the chord would look broken. NEVER
        # promote them.
        Binding("ctrl+a", "toggle_subagents", show=False),
        Binding("ctrl+o", "jump_to_subagents", show=False),
    ]

    class Selected(Message):
        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    class Dismissed(Message):
        pass

    class SubagentLayerToggled(Message):
        """``ctrl+a`` — or a press on the footer chip — flipped the ⌥ layer.

        Carries the new value so the app can re-poll the catalog with the flag.
        The widget cannot load the catalog itself — that is worker-thread work
        the app owns.
        """

        def __init__(self, show_subagents: bool) -> None:
            super().__init__()
            self.show_subagents = show_subagents

    class PinToggled(Message):
        """A pointer press on a row's pin cell asked to toggle that row's pin.

        Carries the row id because the app cannot re-derive WHICH row was
        pressed after the fact — the cell belongs to a row a poll may have
        reordered. The widget writes nothing itself: `toggle_pin` is file I/O
        and belongs on the app's worker, the same split `Selected` keeps.
        """

        def __init__(self, session_id: str) -> None:
            super().__init__()
            self.session_id = session_id

    def __init__(self) -> None:
        super().__init__(id="session-sidebar")
        self.entries: tuple[CatalogEntry, ...] = ()
        #: Her id for `AIDA_MARKER`; refreshed with the entries (`set_entries`),
        #: ``None`` until a list has been built.
        self._aida_id: str | None = None
        self.current_id = ""
        self._requested_id = ""
        self._requested_at = 0.0
        self.cursor_id = ""
        self.error = ""
        self._catalog_loading = True
        self._offset = 0
        self._frame = 0
        self._painted_state: tuple[object, ...] | None = None
        self._painted_lines: dict[int, Strip] = {}
        self._painted_lines_valid = False
        self._spinner_paint = False
        self._timer: Timer | None = None
        self._spinner_rate = SIDEBAR_SPINNER_INTERVAL_S
        self._pressed_id: str | None = None
        #: Whether the press in flight started on the pin cell. Read together
        #: with `_pressed_id` (which still names the row, so `set_entries`'s
        #: gesture freeze covers both) — `on_click` branches on it to toggle
        #: the pin instead of opening the row.
        self._pressed_pin = False
        #: Whether the press in flight started on the footer chip instead. The
        #: chip names no row, so `_pressed_id` stays empty for it — `on_click`
        #: reads this alone to run the layer flip and nothing else.
        self._pressed_chip = False
        #: The chip hit resolved at PRESS time by `focus_on_click` and consumed
        #: by `on_mouse_down`. The walk runs before any handler, and its focus
        #: move repaints this footer (the focused rungs are shorter, so the
        #: chip's cells move with them); re-testing the cells in the handler
        #: would reinterpret a dead-space press against cells the user never
        #: saw — measured during the click-to-focus reconciliation: a press one
        #: cell left of the unfocused chip flipped the layer.
        #:
        #: Deliberately NOT cleared where `_pressed_chip` is (`on_mouse_up`,
        #: `on_click`, `set_open`): no route to a stale read was found — the
        #: walk rewrites it on every MouseDown this widget receives, before any
        #: handler can read it, and Textual synthesizes a Click only when down
        #: and up resolve to the SAME widget (which re-armed the stash for that
        #: widget's own press), while a release outside the region cancels
        #: `_pressed_chip`. The asymmetry is recorded so the next reader does
        #: not have to rebuild that analysis.
        self._press_chip_hit = False
        self._deferred: tuple[CatalogEntry, ...] | None = None
        #: Row under the pointer, by identity rather than by row index: a
        #: catalog refresh reorders rows beneath a stationary pointer, and a
        #: remembered index would light (and describe) whichever session slid
        #: into that slot. `_hover_y` is what re-resolves the identity when the
        #: order changes without the mouse moving.
        self._hover_id: str = ""
        self._hover_y: int | None = None
        #: When the pointer entered the current row. An in-row move restores
        #: the description only once Textual's own show delay has elapsed
        #: since then, so it never appears earlier than Textual would show it.
        self._hover_since: float | None = None
        #: Pinned ids, newest pin first, handed in by the app's poll. Display
        #: only: pins lift rows into their own SECTION in `_display_rows` and
        #: never touch `rank_entries`, which is the partition the mobile relay
        #: shares.
        self._pins: tuple[str, ...] = ()
        #: Peers that did NOT answer the last listing read, as ``(name, reason)``.
        #: Handed in by the app's poll from the relay's own answer
        #: (`peer_rows.unanswered_peers`), and painted as a HEADING-ONLY section —
        #: see `_silent_peer_rows` for why one line with no rows under it is the
        #: honest shape rather than a bug in the section machinery.
        self._silent_peers: tuple[tuple[str, str], ...] = ()
        #: Startup default from `tui.sidebar_show_subagents`; flipped by
        #: `ctrl+a` for THIS session only, never written back to config.
        self.show_subagents: bool = False
        #: ``(session_id, outstanding_ask_count)`` for the session whose asks this
        #: sidebar can actually count — the one the app is attached to. Empty
        #: string means "no mark". COUNTED OVER THE OUTSTANDING SET (open +
        #: timed-out-and-answerable), matching the wire's ``asks_open`` and the
        #: bar's ``present``: a queue of nothing but timed-out asks is still one
        #: the user can answer, so its row must keep its mark. See
        #: :meth:`set_asking` for why the count is
        #: live-only rather than read out of the catalogue.
        self._asking: tuple[str, int] = ("", 0)
        #: Hidden-population size for the footer chip, from `subagent_population`.
        self._subagent_total: int = 0
        #: Whether the pointer rests on that chip. The chip is a CONTROL (issue
        #: #1357 principle 5): the underline this lights is its affordance,
        #: shown on the same hover event that reveals the pin cell's `☆`, and
        #: the press that flips the layer fires from the same hit-test. It is
        #: painted as a STYLE on the chip's cells — never new text — so the
        #: count keeps its place in the ladder.
        self._chip_hover: bool = False
        #: The pointer's last position, kept so `set_entries` can re-derive
        #: `_chip_hover` under a RESTING pointer: the ladder can move the
        #: chip's cells when the count, the paging facts or focus change.
        self._hover_x: int | None = None
        #: Monotonic deadline for a `ctrl+o` jump armed before its rows
        #: existed, or 0.0 for none. See `_land_pending_jump`.
        self._pending_jump_until: float = 0.0
        #: Monotonic deadline for an ARRIVAL reveal armed before the list could
        #: carry the row, or 0.0 for none. See `set_current`'s `_reveal_current`
        #: and `_land_pending_reveal`.
        self._pending_reveal_until: float = 0.0
        self.display = False

    @property
    def requested_id(self) -> str:
        """The row a click asked for that has not yet become ``current_id``.

        Set synchronously at ``_sidebar_navigation_pending`` — before any
        await — so the tint is on the very next frame after the click. It is
        an acknowledgement and nothing more: it does not move ``current_id``,
        does not touch the transcript, enables no input and acks nothing.
        Cleared by ``pending("")`` on commit, failure or cancel.
        """
        return self._requested_id

    @requested_id.setter
    def requested_id(self, session_id: str) -> None:
        if session_id != self._requested_id:
            self._requested_at = time.monotonic() if session_id else 0.0
        self._requested_id = session_id
        self._sync_animation()

    def _requested_spinning(self) -> bool:
        return bool(self._requested_id) and (
            time.monotonic() - self._requested_at >= REQUESTED_SPINNER_DELAY_S
        )

    @property
    def page_size(self) -> int:
        """How many ENTRIES fit, so that chrome + entries + footer fills the height.

        Minus the footer, the title if one is drawn, and whatever section
        chrome the window carries: each occupies a line that cannot hold a
        session, so the count of ENTRIES that fit shrinks by exactly that.

        AND IT IS THE WINDOW'S OWN CHROME THAT DECIDES, not the whole list's
        (design round 1, D5b). ``_header_lines`` charges a header for every
        section the LIST carries, because the window it feeds is what decides
        which sections are painted — a circularity it breaks by over-reserving.
        The reservation is safe and it is not free: on a list whose headings are
        spread across more sections than one page can show, the page comes out
        several ROWS short and the sidebar paints blank beneath its last entry,
        which reads as "no more sessions". The peers frame measured exactly
        that — nine entries of twenty-four, six sections, twelve rows of nothing
        — and the fix is the second reading of the same question: with a
        candidate window in hand, the sections it really covers can be counted.

        ``painted(p) = p + chrome(p) + footer`` is monotone in ``p`` (chrome
        only ever grows as a window takes in more of the order), so the largest
        window that fits is found by bisection between ``_header_lines``'
        safe seed and the height itself. The seed stays the floor, which keeps
        this a pure IMPROVEMENT on the old number: a list whose headings all fit
        on one page resolves to the same value it always did.
        """
        return self._page_size_at(self._offset)

    def _page_size_at(self, offset: int) -> int:
        """`page_size` asked of a candidate offset, without moving there.

        A page move has to price the window it is about to land on — and the
        bottom-aligned window the wheel clamp settles on — BEFORE committing
        to either. The bisection is the same question at any offset; this is
        the one place it is asked, so the two callers cannot drift.
        """
        chrome = self._header_lines(offset)
        low = max(1, self.size.height - 1 - (0 if chrome else 1) - chrome)
        # No window can be taller than the sidebar, nor longer than the order
        # left to show: past either, the candidate is describing rows that do
        # not exist.
        high = min(len(self.entries) - offset, max(1, self.size.height - 1))
        # The seed can exceed the rows left to show — its chrome count is the
        # WHOLE list's — and near the tail that leaves `low > high`, so the loop
        # never runs. Returning the bare seed there would describe rows that do
        # not exist (`page_size` larger than the slice it windows, which the
        # page arithmetic then trusts). Clamp the starting best to `high`; while
        # `low <= high` this is `low` unchanged.
        best = min(low, high)
        while low <= high:
            middle = (low + high) // 2
            window = self.entries[offset : offset + middle]
            inner = self._chrome_for(window)
            if middle + inner + 1 + (1 if inner == 0 else 0) <= self.size.height:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return max(1, best)

    def _chrome_for(self, window: Sequence[CatalogEntry]) -> int:
        """Lines the section headings consume for THIS window, note included.

        ``_header_lines``' own calculation, asked of a window instead of the
        whole list — which is what makes the exact page size computable at all
        (:meth:`page_size`). The note counts here for the same reason it does
        there: it is as tall as a heading, and both sides have to charge it or
        the frame overruns its height.
        """
        if not window:
            return 0
        tiers = {self._section_key(entry) for entry in window}
        # The row-less peer sections are ALWAYS painted, whatever the window is
        # (design round 4, D27), so they are charged unconditionally — a heading
        # the page size did not reserve is an overrun, which is the one thing
        # this accounting exists to prevent.
        tiers |= self._silent_peer_tiers()
        chrome = len(tiers) * 2 + (len(tiers) - 1)
        # The `+N more pinned` note is chrome too, and the charge must follow
        # the PAINT: `_display_rows` emits the note whenever a pinned row is not
        # on the window's page — including a page that carries no pinned row at
        # all — so the predicate here is exactly its own (`_pinned_overflow`).
        # The old rank-0 guard charged only windows that DID carry a pinned row,
        # so a page past the pinned block painted one line more than
        # `page_size` reserved: the frame overran its height, and the line the
        # compositor cropped off the bottom was the footer.
        if self._pinned_overflow(window):
            chrome += 1
        return chrome

    def _header_lines(self, offset: int | None = None) -> int:
        """Lines the section headers will consume, over the WHOLE list.

        Computed from the whole ranked list rather than the visible slice
        because it feeds ``page_size``, which decides that slice: asking the
        slice would be circular. Over-counting by one on a boundary scroll
        costs a row of headroom, never a wrong or hidden entry.

        It is now ``page_size``'s safe SEED rather than its answer — see
        :meth:`_chrome_for`, which asks the same question of a candidate window
        so the page can grow back into rows this count over-reserved (design
        round 1, D5b).

        ``offset`` prices the note for a candidate landing rather than the
        live window (``_page_size_at``); the note's charge is the only part of
        this count that depends on where the window sits.
        """
        if not self.entries:
            return 0
        # The SECTION KEY, not the rank: every peer is its own section with its
        # own heading, so counting distinct ranks would charge one heading for
        # two peers and the frame would overrun its height by a line — the same
        # fault this accounting exists to prevent.
        tiers = {self._section_key(entry) for entry in self.entries}
        # The silent peers are sections too (design round 4, D27) and cost the
        # same chrome, so they are charged here as well — `_chrome_for` asks the
        # same question of a window and must get the same shape of answer.
        tiers |= self._silent_peer_tiers()
        # Per section: its heading, the blank beneath it, and a blank above
        # every heading after the first. The leading blank is load-bearing:
        # without it a heading sits flush against the previous group's last
        # ENTRY row (the section does not end in a blank — the blank belongs
        # under the heading), and the two groups read as one. See the comment
        # in `_display_rows` and the assertion it is defended by.
        #
        # ONE OVER-COUNT IS DELIBERATE: where two ROW-LESS sections meet,
        # `_append_section` declines the second break (design round 4, D28), so
        # this charges one line more than the paint emits. That is the direction
        # this number is allowed to be wrong in — it is `page_size`'s safe seed
        # and a spare line is headroom, where the other direction hides a row.
        chrome = len(tiers) * 2 + (len(tiers) - 1)
        # The `+N more pinned` note (`_display_rows`) is chrome too, and the
        # charge follows the same predicate the paint uses (`_chrome_for`): a
        # pinned row missing from the window. This count's own old rank-0
        # guard already reduced to that predicate — an overflow is only
        # possible when a pin exists, so the guard was implied by it — which
        # leaves the under-charge the note fix repaired in `_chrome_for`
        # alone, whose per-window guard could miss a page that no longer
        # carried any pinned row. Sized against a window computed WITHOUT the
        # note: adding the note can only shrink that window, which can only
        # push more pinned rows out, so this never claims a note that
        # `_display_rows` then declines to emit.
        base = max(1, self.size.height - 1 - chrome)
        where = self._offset if offset is None else offset
        window = self.entries[where : where + base]
        if self._pinned_overflow(window):
            chrome += 1
        return chrome

    def _pinned_overflow(self, window: Sequence[CatalogEntry]) -> int:
        """How many pinned rows exist that ``window`` does not contain.

        Pinned rows are NOT hoisted into the window — the scroll axis is
        pin-blind (`_unpinned_rank`), which keeps `page_size`, `action_move`
        and `_entry_at` on one geometry — so at a short height a pinned row's
        slot can sit below the page. Without this the `★ Pinned` heading
        silently claims to be the whole pinned set while showing part of it,
        and the `+N more pinned` note has nothing to be counted from.
        """
        shown = {entry.id for entry in window}
        return sum(1 for entry in self.entries if entry.id in self._pins and entry.id not in shown)

    @property
    def visible_entries(self) -> tuple[CatalogEntry, ...]:
        return self.entries[self._offset : self._offset + self.page_size]

    def _asking_mark(self, entry: CatalogEntry) -> tuple[str, str] | None:
        """``(glyph, ink)`` for a row the user OWES an answer on, else ``None``.

        The queued-ask sibling of ``row_state_mark``'s needs-you rung, and it
        sits directly below the GATE mark for a reason worth stating: a pending
        gate is a turn that cannot proceed, while a queued ask is one the agent
        has already walked past — both want the user, and the blocked one is
        the louder fact, so ``!`` keeps the cell when both are true.

        Above the spinner and the completion marks, because "you owe an answer"
        outranks "this session is working" and "this session finished" — the
        same urgency ordering ``row_state_mark`` uses, applied one rung lower.

        Ink is the BAR's, not the gate's (``accent``, not ``warning``), so the
        two marks cannot be read as one state in two colours; see
        ``widgets/ask_queue`` for the glyph's own provenance.
        """
        session_id, count = self._asking
        if not count or not session_id or entry.id != session_id:
            return None
        if entry.row.pending:
            return None
        return ASK_MARKER, "accent"

    def _special_mark(self, entry: CatalogEntry) -> tuple[str, str] | None:
        """``(glyph, ink)`` for a row whose mark is NOT its live state, else None.

        Only a SUBAGENT row qualifies: it carries no live state at all by
        construction — the hidden population is never polled for one — so
        ``⌥`` owning its mark column displaces nothing.

        A pinned row does NOT qualify, and this is the correction the design
        round forced. ``row_state_mark`` ranks by urgency because a person
        blocked on a gate is the most important thing a list can say, and the
        user pins the sessions they care about most: a ``★`` here made exactly
        those sessions the only ones that could not report being blocked,
        broken or finished. ``★`` is the DURABLE fact — it does not change
        while the user looks at it — and the state glyph is the volatile one,
        so the pin is what moves. It is painted in the pin cell by ``render``
        instead — its own leading column pair since issue #1357 slice 2a.

        One helper, consulted by BOTH ``render`` and ``_advance_spinner``, so
        the full frame and the 150 ms mark-cell patch cannot disagree about who
        owns the mark column. They did disagree in the first cut: a pinned BUSY row
        painted ``★`` on a full frame and then had a spinner frame patched over
        it on the very next tick, so the mark flickered between the two at
        7 Hz. With ``★`` out of that column the pinned row simply spins.
        """
        if entry.subagent:
            return "⌥", "muted"
        return None

    def _section_of(self, entry: CatalogEntry) -> int:
        """0 pinned, 1 active, 2 previous, 3 peer, 4 subagent.

        A THREE-way key over ``active``, not a widening of it:
        ``CatalogEntry.active`` is ``pending or unseen or live_state``, and the
        attention store keys on conversation identity so it answers for
        subagent ids too — 45% of them carry an unseen receipt. Ranking them by
        ``active`` alone would put agent runs above the user's own work.

        Pinned outranks subagent, which is what makes a pinned subagent row
        appear in ``★ Pinned`` even while the layer is off.

        A REMOTE row takes the peer rank here but its SECTION is per device —
        see :meth:`_section_key`, which is what the sort and the paint both use.

        This is the PAINT the pin lifts into: :meth:`_unpinned_rank` is the rank
        without it, and the scroll axis uses that one (see `_axis_key`).
        """
        if entry.id in self._pins:
            return 0
        return self._unpinned_rank(entry)

    def _unpinned_rank(self, entry: CatalogEntry) -> int:
        """Where the row files when nothing is pinned — the SCROLL AXIS' rank.

        Split out of `_section_of` so the two orders that read it cannot drift:
        the axis (`_axis_key`, and through it `_presentation_order`) places a
        row by the section it belongs to with NO pin, and the paint
        (`_section_key`) lets the pin lift it under `★ Pinned` when the window
        carries it. The pin is a display lift in both directions — it moves
        what a window shows, never where the row travels — which is what keeps
        pins out of view until the page reaches their slot (`_pinned_overflow`).
        """
        if entry.subagent:
            return 4
        if entry.row.is_remote and entry.row.owner_device:
            return _SECTION_PEER_RANK
        return 1 if entry.active else 2

    @staticmethod
    def _peer_heading(row: Any) -> str:
        """The heading a peer's section carries: `` ⇄ <label>``, plus its state.

        The label is the peer's NAME — a 32-hex device id is not something a user
        recognises their own laptop by — and the suffix is the design's two-word
        case (``mesh-ui.md`` §1.3): ``(unreachable)`` when the owning device did
        not answer the read that stamped these rows. Nothing claims staleness of
        the ROWS here; that is ``placement_stale``'s business and the tooltip's,
        because a heading that stacked both would be a sentence, not a label.

        THE LEADING INDENT IS THE LOCALITY COLUMN (design round 2, D18, moved
        by issue #1357 slice 2a). Rows paint the locality glyph in
        `_LOCALITY_COLUMN` (after the pin cell and the caret), and this heading
        used to paint it at the start of the line — so the one glyph that says
        "everything under this line is another device" started at a different x
        from every mark it governed and the column did not stack. A heading has
        no pin and no caret, so those columns are empty on it by definition, and
        indenting it costs no row its width: the tier headings above
        (``★ Pinned``, ``Active Sessions``) keep the star column and stay put.
        """
        return _peer_heading_text(row.owner_label or UNNAMED_DEVICE, bool(row.reachable))

    def _section_key(self, entry: CatalogEntry) -> tuple[int, str]:
        """``(rank, section heading)`` — the ONE key the PAINT's sort uses.

        The heading is the section's IDENTITY here, not a label looked up later:
        a peer's rows form their own contiguous section per DEVICE, so two peers
        are two sections rather than one rank sorted through each other. That is
        why the pair (and not :meth:`_section_of`) is what `_display_rows` and
        `render` both call — the two cannot disagree about a heading they each
        derive from one function.

        :meth:`_axis_key` is its pin-blind sibling — the key the WINDOW slides
        over (`_presentation_order`). Both read the heading through
        :meth:`_peer_heading`, so the axis and the paint cannot disagree about
        which rows are one peer's section either.
        """
        rank = self._section_of(entry)
        if rank == _SECTION_PEER_RANK:
            return rank, _PEER_HEADER_PREFIX + self._peer_heading(entry.row)
        return rank, _SECTION_NAMES[rank]

    def _axis_key(self, entry: CatalogEntry) -> tuple[int, str]:
        """`(rank, section heading)` for the SCROLL AXIS — the pin is not a section.

        See `_unpinned_rank` for why the axis is pin-blind: a pinned row that
        jumped to the head of the axis would be LIFTED INTO VIEW, and the
        `+N more pinned` note exists precisely because that deferral was made
        (`_pinned_overflow`).
        """
        rank = self._unpinned_rank(entry)
        if rank == _SECTION_PEER_RANK:
            return rank, _PEER_HEADER_PREFIX + self._peer_heading(entry.row)
        return rank, _SECTION_NAMES[rank]

    def _presentation_order(self, ranked: Sequence[CatalogEntry]) -> tuple[CatalogEntry, ...]:
        """The order the list presents — and therefore the order it SCROLLS.

        `rank_entries` has decided the ranking; this only regroups it so each
        section is one block in the order the frame paints them (active,
        previous, one block per peer device, subagents — the placement
        ``mesh-ui.md`` decision 1 specifies: "a peer's rows form one contiguous
        section each, placed after `previous` and before `subagent`"). Within a
        section the ranking's own order survives: the sort is stable.

        WHY THE WINDOW SLIDES OVER THIS ORDER (operator report, 2026-09-29:
        remote sessions "disappear when scrolling down ... among previous
        sessions"). The window used to slide over the RANK order — where a
        peer's rows sit interleaved with `previous` by birth — while the frame
        painted them regrouped into their own per-device sections. The two
        orders disagreed, so the frame's edges and the window's edges disagreed:
        a remote row left the window while it was drawn in the MIDDLE of the
        frame (the rows below it stayed put), and rows near it slid DOWN while
        the user scrolled down. With one order, a window's edges are the frame's
        edges: rows leave and enter at the top and bottom of the list, in the
        order they are shown.
        """
        return tuple(sorted(ranked, key=self._axis_key))

    @staticmethod
    def _draws_section_headers(rows: tuple[tuple[str, CatalogEntry | None], ...]) -> bool:
        """Whether these display rows carry section headings.

        The single rule both `render` and `_entry_at` consult, so the painted
        title line and the hit-test can never disagree about whether it is
        there.
        """
        return any(kind.startswith("header:") for kind, _entry in rows)

    def _display_rows(self) -> tuple[tuple[str, CatalogEntry | None], ...]:
        """The lines to paint: ``("header", None)`` or ``("entry", entry)``.

        Headers are RENDERED ROWS and never members of ``self.entries``. That
        is load-bearing: ``action_move`` indexes ``entries``, and
        ``_switch_session_from`` traverses it while the list is CLOSED, so a
        header in that tuple would let ``ctrl+shift+down`` "switch" to one and
        desync open-from-closed navigation. Keeping the split purely
        presentational is also what makes keyboard traversal cross a boundary
        as an ordinary step, with no stall and no skip.

        The boundary is ``_section_of``: pinned, then ``CatalogEntry.active``
        — i.e. the tier ``session_category`` assigns, 0 pending, 1
        unseen-complete, 2 error, 3 interrupted, 4 busy and 5 live are active;
        6 previous is not — then the hidden subagent population. The
        active/previous split is the same partition the mobile relay draws, so
        the two surfaces agree without a new field; pins and sub rows are
        DISPLAY-only lifts on top of it and never reach ``rank_entries``.
        Ordering carries one further key than the tier (an armed wake floats
        within PREVIOUS, where every row is cold), but it never crosses this
        boundary and never applies above it. An empty section contributes no
        header.
        """
        rows: list[tuple[str, CatalogEntry | None]] = []
        # Sections must be CONTIGUOUS, and a pinned row can come from anywhere
        # in the ranking. Group the window's rows by section key, then paint the
        # groups in key order; inside a section the order the rows arrived in
        # survives (the sort is stable).
        #
        # `self.entries` ALREADY ARRIVES in this order (`_presentation_order`),
        # so the grouping below is idempotent for everything but the pin: the
        # window is a slice of the presented list and the frame is that slice —
        # its edges are the frame's edges, which is what makes scrolling move
        # rows up and down the frame instead of re-sorting them under the
        # cursor (the operator report `_presentation_order` records).
        # `action_move`, `_cursor_index` and `_switch_session_from` all index
        # `entries`, so they walk the order the user sees, and `page_size`/
        # `action_move`/`_entry_at` stay on one geometry.
        #
        # The pin is the one lift left in here: a pinned row inside the window
        # paints under `★ Pinned` at the top. A pinned row the window does NOT
        # carry is not visible until the page reaches its slot — the pin never
        # reaches the axis — which is what the `+N more pinned` note is for.
        #
        # A SECTION WITHOUT ROWS IS STILL A SECTION (design round 4, D27). The
        # silent peers are put through the SAME key as the live ones inside this
        # one pass, so a peer's place in the list is a property of its NAME and
        # not of whether it answered: built at the end of the list instead, its
        # section sat below `⌥ Subagent Runs` while it was unreachable and JUMPED
        # above it the moment it recovered — the peer axis' rank (mesh-ui.md
        # decision 1: after `previous`, before `subagent`) is not conditional on
        # liveness. They also interleave with the live peer sections by heading,
        # so two devices do not re-order themselves when one of them comes back.
        sections: dict[tuple[int, str], list[CatalogEntry]] = {}
        for entry in sorted(self.visible_entries, key=self._section_key):
            sections.setdefault(self._section_key(entry), []).append(entry)
        for key in self._silent_peer_tiers():
            sections.setdefault(key, [])
        for key, entries in sorted(sections.items()):
            self._append_section(rows, key[1])
            rows.extend(("entry", entry) for entry in entries)
        # An honest heading: say how many pinned rows are not on this page
        # rather than under-reporting the set. A CHROME row, never an entry —
        # `entry=None` keeps it out of `self.entries` and makes `_entry_at`
        # return `None` for it, so it is not a click target.
        missing = self._pinned_overflow([entry for _kind, entry in rows if entry is not None])
        if missing:
            # Directly after the LAST PINNED ENTRY, and before anything else —
            # including the next section's header and its blank. Matching on
            # "the first non-pinned entry" instead skipped straight past that
            # header (headers carry `entry=None`), so the note landed under
            # `Active Sessions` and read as "+1 more active session", the
            # opposite of what it says. A note about a section has to sit
            # inside it.
            last_pinned = -1
            for index, (_kind, entry) in enumerate(rows):
                if entry is not None and self._section_of(entry) == 0:
                    last_pinned = index
            # NO PINNED ENTRY ON THIS PAGE ⇒ the note goes where the `★ Pinned`
            # heading would have been: the TOP of the window. The fallback used to
            # be ``len(rows)``, which put it under whatever section happened to be
            # last — under the peer rows' device, in the frame that filed this
            # (design round 1, D10), where "+2 more pinned — scroll" reads as a
            # claim about that device. That is the exact failure the comment above
            # says this rule exists to prevent, reached by a new route: the peer
            # sections are three chrome lines per device, so they are what can
            # push the whole pinned tier off a full-height frame.
            insert_at = last_pinned + 1 if last_pinned >= 0 else 0
            rows.insert(insert_at, ("note:pinned-overflow", None))
        return tuple(rows)

    def _append_section(self, rows: list[tuple[str, CatalogEntry | None]], heading: str) -> None:
        """Append one section's chrome: a break, the heading, and the gap beneath it.

        A blank ABOVE every heading but the first, and one BELOW every heading.
        The ask was "an active sessions header and then padding, previous
        sessions": the heading owns the space beneath it, so the gap reads as
        "this group starts here" rather than "the last one ended". The leading
        blank is still needed or the second heading collides with the row above
        it — with padding only underneath, `Previous Sessions` sat flush against
        the last active row and the two groups ran together. The first heading
        takes no leading blank: nothing sits above it to separate from.

        AND NEVER TWO BREAKS IN A ROW (design round 4, D28). A row-less section
        ends on the blank it just emitted, so the section after it may not emit
        another one: two unreachable peers were separated by TWO blank rows where
        every other boundary in the list is one, which is the frame a user with
        two lost devices actually gets. Every section goes through here — the
        live ones, the peer ones and the silent ones — so "one blank per
        boundary" is a property of the builder rather than of each call site.
        """
        if rows and rows[-1][0] != "blank":
            rows.append(("blank", None))
        rows.append((f"header:{heading}", None))
        rows.append(("blank", None))

    def _silent_peer_tiers(self) -> set[tuple[int, str]]:
        """The section KEY of every peer that did not answer.

        WHY A HEADING WITH NOTHING UNDER IT. ``mesh-ui.md`` §8.3 says a peer that
        does not answer contributes NO ROWS rather than stale ones — dropping the
        rows is right, and it is not what UX round 3's U16 filed. What it also
        did was drop the FACT: the section was built from rows, so a peer that
        stopped answering lost its heading too, and the sidebar then read as a
        complete list. "My peer has no sessions" and "my peer is gone" became
        one picture, and the six sessions the user was looking at a minute ago
        vanished and came back with nothing said either way.

        The heading keeps §8.3's promise (no row claims a state the peer did not
        send) and restores the one sentence that explains the frame. It is the
        SAME string a live section would carry with ``reachable`` false, because
        the state is the same state — see :func:`_peer_heading_text` — and it is
        built from the SAME key (`_section_key`) so it takes the peer rank's
        place in the list rather than the end of it (design round 4, D27).

        Chrome, never entries: these sections contribute no ``CatalogEntry``, so
        they stay out of ``self.entries`` — which is what `action_move`,
        `_cursor_index` and `_switch_session_from` index — and `_entry_at`
        already answers ``None`` for a row whose entry is ``None``.

        The relay's ``reason`` is deliberately NOT painted. It is the machine
        token ``connect_failed:ConnectionRefusedError`` that UX round 3 filed as
        U23 on the listing surface; the tooltip on a real row shows it, and a
        heading is not the place to grow a second, unlocalised spelling of it.

        Shared by the three places that must agree about these sections: the
        paint (`_display_rows`, which builds them) and the two chrome counts
        (`_header_lines`, `_chrome_for`, which reserve their height). A second
        derivation of the heading text is how a heading gets painted without
        being charged, and an uncharged line is an overrun.
        """
        return {
            (
                _SECTION_PEER_RANK,
                _PEER_HEADER_PREFIX + _peer_heading_text(name or UNNAMED_DEVICE, False),
            )
            for name, _reason in self._silent_peers
        }

    def set_asking(self, session_id: str, count: int) -> None:
        """Mark a row as holding OUTSTANDING ASKS the user has not answered.

        The sidebar's own state for the queued-ask feature (design §5.1):
        distinct from ``pending``, which means a GATE — a turn that cannot
        proceed until someone answers. A queued ask is the opposite kind of
        fact: the agent already moved on, and the mark says the user still owes
        it an answer rather than that anything is blocked. Two facts, two
        marks, which is why this does not simply feed ``pending``.

        ``count`` is the OUTSTANDING tally (open + timed-out-and-answerable, the
        same set the wire's ``asks_open`` publishes): a timed-out ask is still
        one the user can answer, so a queue of nothing but those must keep the
        mark rather than dropping it to absence.

        Only the CURRENT session is marked, and the count is painted rather
        than the sessions': a sidebar row's queue depth lives on that session's
        own runtime, and the list is a durable catalogue that a cooled session
        is read out of. Marking a row we have no live count for would be
        inventing one.

        Display only, and best-effort by contract — the caller passes "" for
        "no session" and 0 for "nothing outstanding".
        """
        pair = (str(session_id or ""), int(count or 0))
        if pair == self._asking:
            return
        self._asking = pair
        self.refresh()

    def set_entries(self, entries: Sequence[CatalogEntry]) -> None:
        # The order the list presents — and scrolls in — not the bare ranking:
        # see `_presentation_order` for why the window has to slide over the
        # sections the frame paints.
        ordered = self._presentation_order(rank_entries(entries))
        # A refresh between mouse-down and click must not change what was
        # pressed. Freeze the entire order until that gesture has completed.
        if self._pressed_id is not None:
            self._deferred = ordered
            return
        self.entries = ordered
        # Her id, read once per list build (see `session_picker.aida_row_id`)
        # so the row that names her can carry her mark and no other row's
        # geometry moves.
        self._aida_id = aida_row_id()
        self._catalog_loading = False
        self.error = ""
        # Before the re-adopt below, never after: a landed jump must be the
        # cursor that branch sees, or it would restore the pre-chord row over
        # it. See `_land_pending_jump`.
        self._land_pending_jump(ordered)
        self._land_pending_reveal(ordered)
        if not any(entry.id == self.cursor_id for entry in ordered):
            # `current_id` is adopted only when it is a row in THIS list. The
            # attached session is not necessarily a catalog row, and adopting
            # it blind left `cursor_id` naming nothing: `_cursor_index` falls
            # back to 0 so the caret RENDERS on row 0 while `cursor_id`
            # disagrees, and `action_select`'s membership guard then swallows
            # ENTER entirely. Reachable before the ⌥ layer by letting a row age
            # out of the page; `ctrl+a` turns it into a routine keystroke that
            # drops up to 40 rows at once.
            adopted = self.current_id if any(e.id == self.current_id for e in ordered) else ""
            self.cursor_id = adopted or (ordered[0].id if ordered else "")
        self._offset = min(self._offset, max(0, len(ordered) - self.page_size))
        # Re-resolve against the NEW order at the pointer's unchanged position:
        # a reorder under a resting pointer must relabel the row it is actually
        # over, not keep describing the session that moved away. Blanking it
        # instead left the affordance and description dark until the user
        # jiggled the mouse.
        self._set_hover(self._hover_y)
        # The footer chip re-resolves the same way: the ladder under a resting
        # pointer can move the chip's cells (the count, the paging facts, the
        # focus just changed). No repaint is forced here — the span moving is
        # itself what repaints (a count change refreshes, a focus change
        # refreshes, and the paint-state check below carries the flag) — but
        # the flag must tell the truth about the NEXT painted frame.
        self._set_chip_hover(self._hover_x, self._hover_y)
        self._sync_animation()
        # Polling must still adopt fresh summaries (including tooltip-only
        # status), but an unchanged frame must not invalidate Rich's content
        # and Textual's compositor every two seconds in every open terminal.
        if self._paint_state() != self._painted_state:
            self.refresh()

    def set_current(self, session_id: str) -> None:
        """Take ``session_id`` as the row the list calls CURRENT, and show it.

        THE EDGE THE LIST WAS MISSING (found by the lane that re-shot the sidebar
        evidence): the app arrives at a session — boot, `/new`, `/resume`,
        `/new remote <peer>`, a notification click, a remote takeover — and the
        list kept whatever scroll position and cursor it had. On a populated
        store that left the row the pane says the user is in BELOW THE FOLD with
        the caret on a row they are no longer in. Arriving at a session is a
        promise to show it, and the list already has one mechanism for that:
        ``_reveal``, the same call `action_move` and `action_select` make. This is
        its second caller, not a second mechanism.

        GATED ON THE ID ACTUALLY MOVING, which is load-bearing rather than an
        optimisation: the app re-publishes `current_id` on every catalog poll and
        on every sidebar navigation commit, and an unconditional reveal there
        would drag the caret back to the attached session every two seconds while
        the user is parked on another row deciding what to open — which is
        precisely what the arrows are for.

        The app calls this from `_adopt_session` (the one edge every arrival goes
        through) and from the two places that publish the current session beside
        it, so the rule has ONE home rather than one per caller.
        """
        if session_id == self.current_id:
            return
        self.current_id = session_id
        self._reveal_current()

    def set_pins(self, ids: Sequence[str]) -> None:
        """Replace the pinned-id set. Display-only; see `_display_rows`.

        Refreshes only on an actual change: this runs on every catalog poll,
        and an unconditional repaint here would re-ink the list every 2 s in
        every open terminal.
        """
        pins = tuple(ids)
        if pins == self._pins:
            return
        self._pins = pins
        self.refresh()

    def set_silent_peers(self, peers: Sequence[tuple[str, str]]) -> None:
        """Replace the peers that did not answer. ``(name, reason)`` per peer.

        Display-only, and refreshes only on an actual change for the same reason
        `set_pins` does: this runs on every catalog poll, and un-inking the list
        every two seconds in every open terminal is a cost the user pays without
        seeing anything move. See `_silent_peer_rows` for what it paints and why.
        """
        silent = tuple((str(name), str(reason)) for name, reason in peers)
        if silent == self._silent_peers:
            return
        self._silent_peers = silent
        self.refresh()

    def set_subagent_total(self, total: int) -> None:
        """How many hidden subagent runs the store holds — the footer chip."""
        if total == self._subagent_total:
            return
        self._subagent_total = total
        self.refresh()

    @property
    def hovered_id(self) -> str:
        """The row under the pointer, or "".

        Public because the app's pin chord resolves its target from hover
        first, and must not reach into `_hover_id`.
        """
        return self._hover_id

    def show_error(self, message: str) -> None:
        # A refresh error does not erase the last usable catalog.
        self.error = message
        self._catalog_loading = False
        self.refresh()

    def set_open(self, opened: bool) -> None:
        if not opened and (self._pressed_id is not None or self._pressed_chip):
            # A gesture interrupted by the close is cancelled, not left primed:
            # the pin cell is protected by its row id being cleared here, but
            # the chip has no row id, so its own flag is cleared too — the next
            # press anywhere (even dead space, whose press records nothing)
            # must not inherit the interrupted gesture and flip the layer on a
            # click that never touched the chip.
            self.release_mouse()
            self._pressed_id = None
            self._pressed_pin = False
            self._pressed_chip = False
            if self._deferred is not None:
                deferred, self._deferred = self._deferred, None
                self.set_entries(deferred)
        self.display = opened
        self._sync_animation()
        self.refresh()

    def on_mount(self) -> None:
        self._spinner_rate = self._spinner_interval()
        self._timer = self.set_interval(self._spinner_rate, self._advance_spinner, pause=True)
        self._sync_animation()

    def _spinner_interval(self) -> float:
        """Full cadence when the terminal is focused, reduced when it is not.

        Every other animated surface (the band, both subagent surfaces) already
        rates through :func:`animation_focused`; this list was the only one
        that did not, so a blurred window kept repainting 38 rows for nobody.
        """
        return SIDEBAR_SPINNER_INTERVAL_S if animation_focused() else BLURRED_SPINNER_INTERVAL_S

    def sync_animation_rate(self) -> None:
        """Re-rate the tick after a focus change, through the shared seam.

        A Textual timer's interval is fixed at creation, so the timer is
        replaced rather than adjusted — the same thing the subagent panel does
        for the same reason. Whether it should be RUNNING is left to
        ``_sync_animation``, which owns that question already.
        """
        if not self.is_mounted:
            return
        rate = self._spinner_interval()
        if rate == self._spinner_rate:
            return
        self._spinner_rate = rate
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(rate, self._advance_spinner, pause=True)
        self._sync_animation()

    def _sync_animation(self) -> None:
        if self._timer is None:
            return
        if self.display and (
            self._catalog_loading
            or bool(self._requested_id)
            or any(entry.row.live_state == "busy" for entry in self.visible_entries)
        ):
            self._timer.resume()
        else:
            self._timer.pause()

    def _age(self, entry: CatalogEntry) -> str:
        age = (
            format_age(max(0, time.time() - entry.row.mtime)).replace(" ago", "")
            if self.size.width >= 28
            else ""
        )
        return age if len(age) <= 4 else ""

    def _paint_state(self) -> tuple[object, ...]:
        # Keep the complete immutable entries, not a partial fingerprint of
        # their titles/status. New fields must never silently evade invalidation.
        # Age is time-derived, so equal catalog bytes alone are insufficient.
        return (
            self.entries,
            self.current_id,
            self.cursor_id,
            self._requested_id,
            self._requested_spinning(),
            self._offset,
            self.has_focus,
            self._hover_id,
            self.error,
            self._catalog_loading,
            self.size,
            tuple(self._age(entry) for entry in self.visible_entries),
            # Sectioning and the footer chip are paint inputs that do NOT live
            # in `entries`. Omit them and either the ctrl+a toggle does not
            # repaint, or `set_entries`' equality check passes on a frame whose
            # pins changed and the lift is invisible until something else
            # invalidates. `_chip_hover` rides the same rule: the chip's
            # underline is paint, and a re-derive under a resting pointer must
            # reach the frame.
            self._pins,
            self.show_subagents,
            self._subagent_total,
            self._chip_hover,
        )

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        # Any ordinary invalidation owns the next render: hover, theme, resize,
        # focus and new summaries must never reuse a spinner-only projection.
        # Textual may call refresh during Widget construction, before our init.
        if repaint or layout or recompose:
            self._spinner_paint = False
            self._painted_lines_valid = False
        return super().refresh(*regions, repaint=repaint, layout=layout, recompose=recompose)

    def render_line(self, y: int) -> Strip:
        # Widget's default render_line rebuilds the WHOLE Rich Text for even a
        # one-cell dirty region. Reuse public content strips only for a known
        # spinner-only paint; normal rendering and style application stay with
        # Textual, and no private render/compositor cache is modified.
        if self._spinner_paint and y in self._painted_lines:
            return self._painted_lines[y]
        line = super().render_line(y)
        self._painted_lines[y] = line
        return line

    def _advance_spinner(self) -> None:
        self._frame += 1
        if not self._painted_lines_valid or self._paint_state() != self._painted_state:
            # A pending catalog/focus/age change needs its whole frame; never
            # let a tick turn that invalidation into only a spinner-cell paint.
            self.refresh()
            return
        if not self.entries and self._catalog_loading:
            self.refresh(Region(0, 1, 1, 1))
            return
        rows = self._display_rows()
        title = 0 if self._draws_section_headers(rows) else 1
        regions = []
        replacements: dict[int, Strip] = {}
        for y, (_kind, entry) in enumerate(rows, title):
            if entry is None:
                continue
            requested = entry.id == self._requested_id and entry.id != self.current_id
            if entry.subagent:
                # A sub row's mark column is `⌥`, not a spinner's to patch
                # (`_special_mark`). A PINNED row is not skipped: its `★` has
                # its own cell now, so the mark column is free to animate and
                # a pinned busy row spins like any other.
                continue
            if (requested and self._requested_spinning()) or (
                entry.row.live_state == "busy"
                and not entry.row.pending
                and not entry.shows_completion_mark
            ):
                line = self._painted_lines.get(y)
                if line is None:
                    self.refresh()
                    return
                # The old cell carries its exact resolved foreground/background
                # (including cursor/hover selection). Only its glyph changes.
                cell = line.crop(_ROW_MARK_COLUMN, _ROW_MARK_COLUMN + 1)
                glyph = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
                mark = Strip([Segment(glyph, next(iter(cell)).style)], 1)
                replacements[y] = Strip.join(
                    [line.crop(0, _ROW_MARK_COLUMN), mark, line.crop(_ROW_MARK_COLUMN + 1)]
                )
                regions.append(Region(_ROW_MARK_COLUMN, y, 1, 1))
        # refresh() with no regions means the ENTIRE widget, not no work.
        # Only glyphs move at this cadence; titles, chrome and empty space do
        # not need recompositing. The existing timer rate is unchanged.
        if regions:
            self.refresh(*regions)
            self._painted_lines.update(replacements)
            self._painted_lines_valid = True
            self._spinner_paint = True

    def _cursor_index(self) -> int:
        return next((i for i, row in enumerate(self.entries) if row.id == self.cursor_id), 0)

    def _reveal(self) -> None:
        """Scroll so the caret's row is genuinely ON the page.

        THE PAGE SIZE DEPENDS ON THE OFFSET, so `index - page_size + 1` is a
        first guess and not an answer. `page_size` is the largest window that
        fits from the CURRENT offset, and a window that reaches into another
        section pays that section's heading, its blank, the leading break and the
        row itself — chrome the guess did not charge, because the old offset's
        window did not contain that section.

        The case that makes it matter, measured on this branch: a populated store
        with a peer's session below the fold. The guess lands the offset one row
        short of the peer row and the page still ends above it, so a reveal that
        "scrolled" leaves the row the user was told about off-screen — and no
        formula lands it, because the row is on the page only for offsets low
        enough that its whole section fits (`41 entries / page_size 24-25 / 28-row
        panel`: offset 16 ends at the local row; the peer row is drawn at 19,
        where the window is 22 entries plus the section's 5 chrome lines).

        So walk the offset up until the row is in the window. Terminates at
        `index` at the latest, where the window starts ON the row, and each step
        is a `page_size` bisection over at most a screen of entries. The walk
        finds the SMALLEST such offset at or after the guess, which is the
        smallest scroll that shows the row — the row ends up as low on the page
        as it can, the same place the old formula aimed for.

        It is not only the arrival that was short: measured on this branch's
        fixture (41 entries, 28-row panel), `action_move` down to the peer row
        left the caret off-page for that one step, and `action_edge(True)` — the
        `end` key — moved the offset to 17 with `page_size` 23 against a row at
        40. Both land it after the walk, which is why the correction belongs in
        this one mechanism rather than at the arrival's call site.
        """
        index = self._cursor_index()
        self._offset = max(min(self._offset, index), index - self.page_size + 1)
        while self._offset < index and index >= self._offset + self.page_size:
            self._offset += 1
        self._sync_animation()
        self.refresh()

    def _reveal_current(self) -> None:
        """Put the caret on the current row and scroll it onto the page.

        The row is not always here yet — that is what the arm below is for, and
        `_land_pending_reveal` is where it lands — so this either reveals NOW or
        arms. `current_id` is the target rather than an argument because that is
        the one value both halves of the contract read; an argument could be
        stale by the time a poll lands the row.
        """
        if not self.current_id:
            return
        if any(entry.id == self.current_id for entry in self.entries):
            # An immediate reveal SATISFIES any arm still outstanding for this
            # row, so it is dropped here rather than left to fire on a later poll
            # — which could only move a caret the user has since placed.
            self._pending_reveal_until = 0.0
            self.cursor_id = self.current_id
            self._reveal()
            return
        self._pending_reveal_until = time.monotonic() + PENDING_ARRIVAL_REVEAL_S

    def _land_pending_reveal(self, ordered: Sequence[CatalogEntry]) -> None:
        """Land an arrival reveal armed before the list carried its row.

        The same shape as `_land_pending_jump`, and for the same reason: a row
        the app names can fail to exist at the moment it names it. There the
        chord outran a catalog re-poll; here the ARRIVAL outruns the poll, because
        `/new remote <peer>` mints the session on the peer inside the keystroke
        and the sidebar learns about it on the next tick.

        Called from `set_entries` AFTER `_land_pending_jump` and BEFORE the
        membership-safe re-adopt, and that order is the deliberate one: the
        re-adopt below only fills a cursor that names nothing, so it cannot undo
        either landing, and the arrival lands SECOND because its contract is the
        stronger of the two — the app IS in that session now, where a `ctrl+o`
        jump is a convenience whose rows may have arrived in the same poll. A jump
        landing second would scroll to the ⌥ section and take the arrived row off
        the page again.
        """
        if not self._pending_reveal_until:
            return
        if time.monotonic() >= self._pending_reveal_until:
            # The window closed, and a row arriving later is not this arrival's
            # business — see `PENDING_ARRIVAL_REVEAL_S`.
            self._pending_reveal_until = 0.0
            return
        if not any(entry.id == self.current_id for entry in ordered):
            # Still waiting: this poll crossed the arrival, or carried a read
            # that predates it. Stay armed until the deadline.
            return
        self._pending_reveal_until = 0.0
        self.cursor_id = self.current_id
        # Safe before `set_entries`'s own offset clamp, which only lowers the
        # offset to `len - page_size` and so cannot push the row back out — the
        # same argument `_land_pending_jump` records for its own reveal.
        self._reveal()

    def action_move(self, delta: int) -> None:
        if self.entries:
            index = max(0, min(len(self.entries) - 1, self._cursor_index() + delta))
            self.cursor_id = self.entries[index].id
            self._reveal()

    def action_page(self, delta: int) -> None:
        """Move a page: the VIEWPORT travels and the cursor rides it.

        WHY NOT `action_move(delta * page_size)` — the old spelling — AND WHAT
        IT FIXED (operator report, 2026-09-29): paging by moving the CURSOR and
        revealing it is contiguous only while the page size is constant, and
        this list's page size varies with the sections a window carries (each
        peer device's section spends three chrome lines, and the pinned note
        appears and disappears). When the window shrank across the move, the
        reveal walk advanced the offset PAST rows the move never showed — a full
        pagedown sweep skipped four peer rows in one fixture and a pinned row in
        another — and a wheel-scrolled viewport, whose cursor the wheel
        deliberately leaves behind, teleported back to the cursor on the first
        page key, dropping every remote row it was showing.

        A page key moves the PAGE, then: the next window starts where this one
        ended — overlap is allowed, a gap never — and the cursor keeps its row
        in the frame where that row still exists, clamped into the new window
        when it shrank. A press whose page already carries the whole tail is a
        no-op, because the bottom is a destination rather than something to
        re-show smaller. Upward, the mirror: the window moves back a page and
        walks forward (never past its old start) until it reaches the row above
        the window it came from, so a traversal retraces rows instead of
        stepping over them.
        """
        if not self.entries or not delta:
            return
        for _ in range(abs(delta)):
            if not self._page_once(-1 if delta < 0 else 1):
                break
        self._sync_animation()
        self.refresh()

    def _page_once(self, direction: int) -> bool:
        """One page of travel; False when the viewport cannot move that way."""
        start = self._offset
        page = self.page_size
        cursor = self._cursor_index()
        if direction > 0:
            if start + page >= len(self.entries):
                # The whole tail is already on the page: the bottom is a
                # destination, and a further press must not re-show it smaller.
                return False
            # No clamp is needed here: the guard above proves `start + page`
            # is at most the last row's index.
            new_start = start + page
            # And the bottom is ALSO a destination for the press that crosses
            # into the tail (UX round 1, U1): when that landing is itself the
            # FINAL window — it reaches the last row — and the bottom-aligned
            # window (where the wheel's `len - page_size` clamp settles) would
            # show at least as many rows, settle there rather than on the
            # remainder below it. `bottom <= new_start` is that no-gap check:
            # the bottom window lies at or beyond the row where this page
            # ended, so the union stays contiguous and the press still moves.
            if new_start + self._page_size_at(new_start) >= len(self.entries):
                bottom = self._bottom_start()
                if start < bottom <= new_start:
                    new_start = bottom
        else:
            new_start = max(0, start - page)
        if new_start == start:
            return False
        self._offset = new_start
        new_page = self.page_size
        if direction < 0:
            # A window that shrank as it moved must still reach the row above
            # the one it came from, or the rows between the two windows were
            # never shown: walk it forward (never past `start - 1`) until it
            # does.
            while new_start < start - 1 and new_start + new_page - 1 < start - 1:
                new_start += 1
                self._offset = new_start
                new_page = self.page_size
        index = max(new_start, min(new_start + new_page - 1, cursor + (new_start - start)))
        self.cursor_id = self.entries[index].id
        return True

    def _bottom_start(self) -> int:
        """The offset of the window that carries the tail — the bottom-aligned
        landing the wheel's ``len - page_size`` clamp comes to rest on.

        The candidates are the offsets whose window shows every row left below
        it (``offset + page_size(offset) >= len``), and at a short height there
        can be several, because a window can grow again where a section's
        heading leaves it. The wheel stops on the first candidate it reaches
        from above, which is the EARLIEST one — the window showing the most
        rows while still carrying the tail. So that is what is walked for
        here: start at the deepest window (one row, always a candidate) and
        step up while the offset above still carries the tail. The walk is
        bounded by the panel height, not the list — only the last
        ``height`` offsets can reach "carry" at all.
        """
        entries = len(self.entries)
        offset = max(0, entries - 1)
        while offset > 0 and (offset - 1) + self._page_size_at(offset - 1) >= entries:
            offset -= 1
        return offset

    def action_edge(self, end: bool) -> None:
        if self.entries:
            self.cursor_id = self.entries[-1 if end else 0].id
            self._reveal()

    def action_select(self) -> None:
        if self.cursor_id and any(entry.id == self.cursor_id for entry in self.entries):
            self._reveal()
            self.post_message(self.Selected(self.cursor_id))

    def action_leave(self) -> None:
        self.post_message(self.Dismissed())

    def action_toggle_subagents(self) -> None:
        """Flip the ⌥ layer for THIS session. Never writes the setting.

        The ONE flip both routes share: the `ctrl+a` chord and a pointer press
        on the footer chip (`on_click`), which is the route that needs no F9.
        Neither writes the setting: a `write_setting` here would fan out
        through `ConfigWatcher` to every running `lop` process and flip
        another terminal's sidebar. The same rule `ctrl+g` follows for dock
        density.
        """
        self.show_subagents = not self.show_subagents
        self.refresh()
        # The widget cannot load the catalog itself: `load_catalog` is I/O and
        # belongs on the app's worker thread. Announce the flip and let the
        # app re-poll with it.
        self.post_message(self.SubagentLayerToggled(self.show_subagents))

    def action_jump_to_subagents(self) -> None:
        """Move the cursor to the first subagent row, revealing it.

        With the layer hidden this SHOWS it and then jumps, rather than doing
        nothing: asking to go somewhere is asking to see it, and a chord that
        no-ops with no feedback is indistinguishable from a chord that is
        broken.

        One press does both. From a layer-off sidebar the rows do not exist
        yet — `load_catalog` filters the hidden population at the load site,
        so they arrive only with the re-poll the message below triggers. The
        jump is therefore ARMED here and landed by `_land_pending_jump` from
        the `set_entries` that delivers them; it used to be attempted once
        against rows that could not be there, which left the reveal done and
        the jump undone and made a SECOND press the thing that landed it.
        """
        revealed = False
        if not self.show_subagents:
            self.show_subagents = True
            revealed = True
            self.refresh()
            self.post_message(self.SubagentLayerToggled(True))
        target = next((entry for entry in self.entries if entry.subagent), None)
        if target is not None:
            # The rows are already here, so the jump lands now and nothing is
            # left pending. This is also the no-op path: with the layer on and
            # the cursor already on the first subagent row, both assignments
            # are writes of the value already there.
            self._pending_jump_until = 0.0
            self.cursor_id = target.id
            self._reveal()
            return
        # Nothing to jump to yet. Arm the intent only when THIS call turned
        # the layer on, i.e. only when a re-poll is genuinely inbound. With
        # the layer already on, the rows the catalog holds are already here
        # and their absence means the store has no subagent runs — arming
        # there would hand the cursor to an unrelated poll that happens to be
        # the one a delegated run first appears in.
        self._pending_jump_until = time.monotonic() + PENDING_SUBAGENT_JUMP_S if revealed else 0.0

    def _land_pending_jump(self, ordered: Sequence[CatalogEntry]) -> None:
        """Land a `ctrl+o` jump armed before its rows existed.

        Called from `set_entries` against the new order and BEFORE the
        membership-safe re-adopt below it, which is what makes the two
        cooperate rather than fight: the chord leaves `cursor_id` naming a row
        the layer-off list no longer has, and the re-adopt exists precisely to
        replace such a cursor (QA D3). Landing first means `cursor_id` already
        names a row in `ordered` by the time that branch tests it, so it
        correctly does nothing; when the rows still have not arrived, the
        cursor is left exactly as it was and the re-adopt still runs in full.
        """
        if not self._pending_jump_until:
            return
        if time.monotonic() >= self._pending_jump_until:
            # The window closed. A layer that came back with no subagent rows
            # is a store with none in it, and a run starting minutes later is
            # not this keystroke's business.
            self._pending_jump_until = 0.0
            return
        target = next((entry for entry in ordered if entry.subagent), None)
        if target is None:
            # Still waiting: this poll crossed the chord, or carried the
            # layer-off population. Stay armed until the deadline.
            return
        self._pending_jump_until = 0.0
        self.cursor_id = target.id
        # Same reveal the immediate path does: the ⌥ section is the LAST one,
        # so on any real list the row it lands on is below the fold and a
        # cursor there without a scroll is a cursor the user cannot see. Safe
        # before `set_entries`'s own offset clamp, which only lowers the
        # offset to `len - page_size` and so cannot push the row back out.
        self._reveal()

    def _entry_at(self, y: int) -> CatalogEntry | None:
        # Against the DISPLAY rows, not the entries: a header and its blank
        # line occupy lines that hold no session, so indexing entries directly
        # would attribute a click to whatever sits that many rows further
        # down. `None` on a header is already what hover and click want — both
        # no-op on it.
        rows = self._display_rows()
        # Same rule `render` uses to decide whether a title line is drawn, so
        # the hit-test cannot drift a row away from the paint.
        index = y - (0 if self._draws_section_headers(rows) else 1)
        if not 0 <= index < len(rows):
            return None
        kind, entry = rows[index]
        return entry if kind == "entry" else None

    def _pin_cell_pressed(self, event: events.MouseEvent) -> bool:
        """Whether a pointer event landed on a row's pin cell.

        The column is resolved against the widget's LEFT PADDING because an
        event's x arrives relative to the widget's outer box (Textual forwards
        ``screen_x - region.x``), while the row's columns belong to the content
        box. Reading `styles.padding.left` rather than a constant keeps this
        true in all three placements: the gutter swaps sides with the dock and
        the overlay keeps the base left pad, so the resolved pad is the only
        thing the column arithmetic may depend on (see `_sync_sidebar_layout`).
        """
        column = event.x - self.styles.padding.left
        return 0 <= column < PIN_CELL_WIDTH

    def _chip_hit(self, x: int, y: int) -> bool:
        """Whether a widget-relative pointer position lands on the footer chip.

        The footer is the widget's LAST content line (`render` pads to it),
        and the chip's cells come back from the same fitted ladder the paint
        uses (`_footer_line`), so the press target can never drift from the
        thing the user sees — the same reason `_entry_at` reads render's own
        header rule. False whenever no chip is painted: the error/loading/
        opening states, and any width where the ladder kept a chip-less
        fallback.

        Like `_pin_cell_pressed`, the column is resolved against the widget's
        LEFT padding (events arrive relative to the outer box; the chip's
        columns belong to the content box), which is the only thing the column
        arithmetic may depend on — the gutter swaps sides with the dock and
        the overlay keeps the base left pad, so the resolved pad is what keeps
        this true in all three placements.

        The row test carries the same vertical invariant `_entry_at` already
        relies on: an event's y arrives relative to the outer box while
        `size.height` counts the content box, so the last content line IS
        `y == size.height - 1` only while `padding.top == 0` — true in every
        placement today (the base tcss and `_sync_sidebar_layout` both keep
        the top pad at 0); a top pad would make both resolve y first.
        """
        if y != self.size.height - 1:
            return False
        _text, span = self._footer_line(max(1, self.size.width))
        if span is None:
            return False
        column = x - self.styles.padding.left
        return span[0] <= column < span[1]

    def on_mouse_down(self, event: events.MouseDown) -> None:
        entry = self._entry_at(event.y)
        if entry is not None and event.button == 1:
            self._pressed_id = entry.id
            self._pressed_pin = self._pin_cell_pressed(event)
            self._pressed_chip = False
            self.capture_mouse()
        elif event.button == 1 and self._press_chip_hit:
            # A press on the footer chip — the hit as resolved by the walk at
            # PRESS time (`_press_chip_hit`; re-testing here would read the
            # focused ladder the press itself may have just painted). The
            # flip fires on the CLICK, like the pin cell's, so a press dragged
            # off the list is cancelled by `on_mouse_up` instead of toggling
            # on the way out; and no row id is recorded — the chip belongs to
            # no row, which is what keeps the click from opening, switching or
            # pinning anything.
            self._pressed_pin = False
            self._pressed_chip = True
            self.capture_mouse()
        event.stop()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self.release_mouse()
        if not self.region.contains(event.screen_x, event.screen_y):
            self._pressed_id = None
            self._pressed_pin = False
            self._pressed_chip = False
            if self._deferred is not None:
                deferred, self._deferred = self._deferred, None
                self.set_entries(deferred)
        event.stop()

    def on_click(self, event: events.Click) -> None:
        entry = (
            self._entry_at(event.y)
            if self.region.contains(event.screen_x, event.screen_y)
            else None
        )
        target = (self._pressed_id or (entry.id if entry else "")) if entry is not None else ""
        pin_press = self._pressed_pin
        chip_press = self._pressed_chip
        self._pressed_id = None
        self._pressed_pin = False
        self._pressed_chip = False
        if chip_press:
            # The chip's press runs the ONE flip `ctrl+a` runs, on the same
            # per-session discipline: never a config write, because a write
            # would fan out through the config watcher to every running `lop`
            # process and flip another terminal's sidebar (`action_toggle_
            # subagents` carries the rest). No `Selected` is posted and no
            # cursor moves, so a press meant for the count can never open,
            # switch or pin the row it was never aimed at (issue #1357
            # principle 5).
            self.action_toggle_subagents()
        elif target and pin_press:
            # A press on the pin cell toggles the pin and NOTHING else: the
            # cursor does not move and no `Selected` is posted, so a click
            # meant for the star can never open or switch the row under it
            # (issue #1357 slice 2a). The row is the one the press STARTED on
            # — the same freeze `set_entries` keeps for an ordinary press.
            self.post_message(self.PinToggled(target))
        elif target:
            self.cursor_id = target
            self.post_message(self.Selected(target))
        if self._deferred is not None:
            deferred, self._deferred = self._deferred, None
            self.set_entries(deferred)
        event.stop()

    @classmethod
    def _bound_keys(cls) -> frozenset[str]:
        """The keys this widget answers itself — never candidates for typing-home.

        Read at call time from ``_merged_bindings`` rather than derived in the
        class body, for the reason ``TranscriptView._bound_keys`` records:
        ``DOMNode.__init_subclass__`` assigns that map AFTER the body runs, so
        a body-time read would see the parent's map.
        """
        merged = cls._merged_bindings
        return frozenset() if merged is None else frozenset(merged.key_to_bindings)

    def on_key(self, event: events.Key) -> None:
        """TYPING-HOME: text pressed on the list is typed, not swallowed.

        Issue #1357 decision, T12/G1 (the `### lopdev — design decision:
        click-to-focus` comment on the issue): the list's keyboard mode —
        entered by a press, by `f9` or by `/sidebar focus` — is for this
        widget's OWN bindings (arrows, enter, escape, ctrl+a/ctrl+o). Any
        printable character it does not bind is handed to the composer, which
        takes the keyboard back — so "typed input going nowhere" cannot
        happen while the list owns the keys, with ONE exception: a hard
        claimant keeps its keys, and in that state (reachable by `f9`, never
        by a press — the walk refuses) a key is left to the claimant and is
        inert on the list. Pre-existing, frozen by the decision. Measured on
        main before this existed: `f9` then `A` left `editor.text == ""` with
        focus still on the list — the failure mode the old
        `FOCUS_ON_CLICK = False` rule was written to stop, reachable all
        along on the deliberate path (F3).

        A FRESH ``Key`` is posted to the editor rather than the original, and
        the editor is focused first, matching ``TranscriptView.on_key``: this
        event is already part-way through Textual's dispatch, and the
        composer's own ``_on_key`` must see a key that behaves exactly as if
        the composer had held focus all along (draft, caret, shell mode and
        the live-answer hold all live there — the decision forbids a parallel
        text path).

        Keys this widget binds itself are excluded via :meth:`_bound_keys`, so
        a future PRINTABLE binding (a filter field, say) is not shadowed by
        this handler. Today's chords are kept by ``event.is_printable``
        instead: it admits a printable character only, so every control key —
        ctrl+a and ctrl+o included — and every arrow stays with this widget's
        own bindings and can never be mistaken for text (G9).

        Refusal mirrors the guard the composer's own routes use: while the
        composer is read-only or a hard claimant holds the keyboard, the key
        is left UNSTOPPED so whatever owns it still receives it — a key is
        not ours to take (``composer_focus.composer_may_take_focus``).
        """
        if not self.has_focus:
            # Mirrors ``TranscriptView.on_key``: a key can reach a container by
            # bubbling from a focused child. This widget mounts none today —
            # the guard is what keeps a future focusable descendant (the
            # filter field the decision names) owning its own printable keys.
            return
        if event.key in self._bound_keys() or not event.is_printable:
            return
        from local_operator.tui.widgets.editor import Editor

        try:
            editor = self.app.query_one(Editor)
        except Exception:  # noqa: BLE001 — a harness that hosts a list and no composer
            return
        if not composer_may_take_focus(self.app, editor):
            return
        editor.focus()
        editor.post_message(events.Key(event.key, event.character))
        event.stop()
        event.prevent_default()

    def on_paste(self, event: events.Paste) -> None:
        """TYPING-HOME for a paste: it is delivered to the composer, never dropped.

        The same rule as :meth:`on_key` (issue #1357 decision, T12/G1). A
        bracketed paste arrives as ONE event rather than as keystrokes, and
        with the list owning the keyboard it must not vanish with nothing on
        the frame to say it did: the composer takes the keyboard and receives
        a fresh ``Paste``, so its own ``_on_paste`` (credential capture, image
        attachment, collapse) and ``TextArea``'s insert run exactly as if the
        composer had held focus. A paste has no bindings of its own, so the
        only refusal is the same claimed/read-only guard the key path uses.
        """
        from local_operator.tui.widgets.editor import Editor

        try:
            editor = self.app.query_one(Editor)
        except Exception:  # noqa: BLE001 — a harness that hosts a list and no composer
            return
        if not composer_may_take_focus(self.app, editor):
            return
        editor.focus()
        paste = events.Paste(event.text)
        # Stopped BEFORE it is posted, and that is load-bearing: ``Paste``
        # bubbles, the bubble reaches ``App.on_event``, and its Paste route
        # re-forwards anything not already marked forwarded to
        # ``self.focused`` — the editor we just focused — so the payload would
        # land twice (measured: "pasted textpasted text"). Pre-stopping keeps
        # the delivery to the ONE handler chain a focused composer runs
        # (``Editor._on_paste`` plus ``TextArea``'s insert).
        paste.stop()
        editor.post_message(paste)
        event.stop()

    def _describe(self, entry: CatalogEntry | None) -> str | None:
        """Hover text: who, in what state, and which session id.

        The state line is the BACKEND's sentence verbatim — see
        ``session/catalog.py``'s ``status`` for the wording's own argument — plus,
        for ``wedged`` only, one client-owned remedy clause (``WEDGED_REMEDY``,
        and its comment for why the two halves have different owners).

        Keyed on ``status_code == "wedged"`` rather than on the sentence, because
        that is the value the app and the catalogue share; a row that is ALSO
        holding a gate keeps its gate words and gets no remedy, which is right —
        the remedy names what to do about SILENCE, and a row asking a question is
        not silent.
        """
        if entry is None:
            return None
        status = entry.status
        if entry.status_code == "wedged":
            status = f"{status} · {WEDGED_REMEDY}"
        # WHERE IT LIVES, and only when there is something to say: this is the
        # peer's NAME readable after the user has scrolled past the heading (the
        # heading is one line of chrome, the tooltip travels with the row), plus
        # the reason when the link is down. Local rows gain no clause — "on this
        # device" on every row is the noise the mark's absence already avoids.
        location = ""
        if entry.row.is_remote or entry.row.owner_device:
            location = f"on {entry.row.owner_label or UNNAMED_DEVICE}"
            if not entry.row.reachable:
                # THE WORDS, NOT THE TOKEN (design round 1, D3; UX round 3, U23).
                # This tooltip used to print the relay's raw reason —
                # ``connect_failed:ConnectionRefusedError`` — on a line a user
                # reads, while the panel beside it said the same fact in words.
                # One gloss, shared, so the sibling surfaces read as one voice.
                location += f" — unreachable: {peer_reason_words(entry.row.unreachable_reason)}"
            if entry.row.placement_stale:
                location += " (last known state)"
        lines = [entry.row.name, status]
        if location:
            lines.append(location)
        # An AGENT-OPENED row names its opener here, the one TUI place with room
        # for it: the row itself is width-bound and otherwise identical to the
        # operator's own (PR #1436 design review round 1, D2). Same vocabulary
        # as the desktop flyout (#448) — `opened by <role>`, or the certain fact
        # `agent-opened` when the role could not be read. It follows the device
        # clause (where the row lives, then who opened it) and precedes the id,
        # which both features keep as the tooltip's last line.
        if entry.row.opened_by is not None:
            role = opener_role(entry.row.opened_by)
            lines.append(f"opened by {role}" if role else AGENT_OPENED_MARK)
        lines.append(entry.id)
        return "\n".join(lines)

    def _set_hover(self, y: int | None) -> bool:
        """Point the hover affordance and the tooltip at the row under `y`.

        Called both from pointer movement and from a catalog refresh, because a
        stationary pointer covers a DIFFERENT session once the ranking
        reorders — the row must re-resolve without waiting for a mouse event.

        Returns whether the hovered identity changed, so the caller can repaint
        exactly once instead of on every pointer move within a row.
        """
        self._hover_y = y
        entry = self._entry_at(y) if y is not None else None
        hover_id = entry.id if entry is not None else ""
        description = self._describe(entry)
        # Re-arm the widget's ``tooltip`` only when the row identity changes.
        # This is NOT what keeps it up under pointer movement (round 4, M1:
        # ``Screen._handle_mouse_move`` hides a showing tooltip BEFORE the
        # event reaches this widget, so nothing set here can prevent that —
        # see ``on_mouse_move`` for the restore). What this does fix is the
        # OTHER way the description vanished: the 2 s catalog poll called
        # ``set_entries``, which blanked ``tooltip`` under a perfectly still
        # pointer. Keeping the value stable across a refresh is what lets a
        # resting description survive the poll.
        changed = hover_id != self._hover_id
        if changed:
            self._hover_since = time.monotonic() if hover_id else None
        if changed or self.tooltip != description:
            self._hover_id = hover_id
            self.tooltip = description
        return changed

    def _set_chip_hover(self, x: int | None, y: int | None) -> bool:
        """Point the chip affordance at the pointer, reporting any change.

        A separate state from the row hover — the chip names no row — and
        repainted the same byte-scoped way: the caller refreshes the footer
        line alone, not the widget.
        """
        hovered = x is not None and y is not None and self._chip_hit(x, y)
        if hovered == self._chip_hover:
            return False
        self._chip_hover = hovered
        return True

    def _refresh_rows(self, *rows: int | None) -> None:
        """Repaint just these row lines, skipping any outside the widget.

        A row's ``y`` is a widget-local line, which is exactly what a
        one-line ``Region`` wants; anything off the edge (or ``None``, the
        pointer having left the list) is simply not painted.
        """
        width = max(1, self.size.width)
        for y in rows:
            if y is not None and 0 <= y < self.size.height:
                self.refresh(Region(0, y, width, 1))

    def on_mouse_move(self, event: events.MouseMove) -> None:
        # The row being LEFT, captured before `_set_hover` overwrites it.
        left = self._hover_y
        self._hover_x = event.x
        row_changed = self._set_hover(event.y)
        # The chip lights under the pointer the way the pin cell's `☆` does:
        # painted on the hover event, so it is seen before it is pressed — and
        # only its own line is repainted, not the list.
        if self._set_chip_hover(event.x, event.y):
            self._refresh_rows(self.size.height - 1)
        if row_changed:
            # Only the two rows whose ground changes, not all 38: a hover move
            # wrote 9,092 bytes of escape sequences to the terminal and now
            # writes 924. That output is CPU burned in the terminal emulator
            # competing with the pointer, which is why bytes are worth
            # scoping even though the widget's own repaint cost is unchanged.
            #
            # Rows are independent (see `render`: each line reads only its own
            # entry), so repainting two of them cannot leave a third stale.
            # This only pays stacked on `auto_links = False` — Textual's own
            # hover watcher would otherwise dirty the whole widget anyway.
            self._refresh_rows(left, self._hover_y)
            # A NEW row: Textual's delay timer was (re)armed by this very
            # move only if its tooltip was not showing; when it WAS showing
            # it hid it and armed nothing, so a resting pointer on the new
            # row would never get that row's description. Arm our own
            # one-shot at the same delay, so a row change behaves like a
            # first arrival rather than a dead end.
            if self._hover_id:
                self.set_timer(
                    float(self.app.TOOLTIP_DELAY),
                    self._show_tooltip_if_still_here,
                    name="sidebar-tooltip",
                )
            return
        # Same row, pointer merely moved within it. Textual has ALREADY hidden
        # the tooltip by the time this runs (``Screen._handle_mouse_move``
        # sets ``display = False`` on any move over the widget that owns a
        # showing tooltip, before forwarding the event) and will not re-show
        # it until a further move restarts its delay timer — so one cell of
        # jitter dropped the description and resting did not bring it back.
        # The user's ask is "stays while the mouse is hovered over the
        # session": restore it. Only when it was showing for THIS widget, so
        # a tooltip that was never up (first arrival, or a different widget's)
        # still goes through Textual's normal delay rather than popping.
        if self._hover_id and self._tooltip_due():
            self._show_tooltip_now()

    def _tooltip_due(self) -> bool:
        """Has the description been up (or is it owed) for the hovered row?

        Textual keeps no "was shown" bit a widget can read, and by the time
        this runs it has already hidden the tooltip — so the honest signal is
        time: the pointer entered this row at ``_hover_since`` and Textual's
        own ``TOOLTIP_DELAY`` has elapsed, which is exactly the condition
        under which its timer would have shown it. Using its constant means
        the restore never pops earlier than Textual itself would.
        """
        if self._hover_since is None:
            return False
        return time.monotonic() - self._hover_since >= float(self.app.TOOLTIP_DELAY)

    def _show_tooltip_if_still_here(self) -> None:
        # The one-shot may fire after the pointer moved on or left; only the
        # row it was armed for gets shown, and only if nothing hid it since.
        if self._hover_id and self._tooltip_due() and self.screen.app.mouse_over is self:
            self._show_tooltip_now()

    def _show_tooltip_now(self) -> None:
        """Re-show the description Textual just hid for an in-row move."""
        try:
            from textual.widgets import Tooltip

            tooltip = self.screen.get_child_by_type(Tooltip)
        except Exception:
            return
        if self.tooltip is None:
            return
        tooltip.display = True
        tooltip.absolute_offset = self.app.mouse_position
        tooltip.update(self.tooltip)

    def on_leave(self, event: events.Leave) -> None:
        # The pointer left the list: drop both the affordance and the
        # description rather than leaving a row lit under an absent cursor.
        if self._hover_id or self._hover_y is not None or self._chip_hover:
            self._hover_id = ""
            self._hover_y = None
            self._hover_x = None
            self._hover_since = None
            self._chip_hover = False
            self.tooltip = None
            self.refresh()

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        self._scroll(1)
        event.stop()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        self._scroll(-1)
        event.stop()

    def _scroll(self, direction: int) -> None:
        # Wheel moves the viewport, never the keyboard target. Match the live
        # app sensitivity rather than invent a second scrolling speed.
        step = max(1, int(self.app.scroll_sensitivity_y))
        self._offset = max(
            0, min(max(0, len(self.entries) - self.page_size), self._offset + direction * step)
        )
        self._sync_animation()
        self.refresh()

    def on_resize(self, event: events.Resize) -> None:
        self._offset = min(self._offset, max(0, len(self.entries) - self.page_size))
        self._sync_animation()

    def render(self) -> Text:
        self._spinner_paint = False
        self._painted_lines.clear()
        self._painted_lines_valid = True
        self._painted_state = self._paint_state()
        width = max(1, self.size.width)
        result = Text(no_wrap=True, overflow="crop")
        rows = self._display_rows()
        # The outer title only when nothing else heads the list. Once the
        # section headings are present it paints identically to them — same
        # `muted`, same weight, same indent, on the adjacent line — so the
        # frame opened with the word "Sessions" twice and no rendered cue
        # which was the panel and which the group. The relay this mirrors has
        # its two headings under the PRODUCT BRAND, not under a third
        # "Sessions" label. The quiet states still need it: with no sections
        # there is nothing to head, and loading/empty/error would open on bare
        # body copy — which an empty list gets for free, since it emits no
        # headers.
        first = True
        if not self._draws_section_headers(rows):
            result.append(
                truncate_cells("Sessions", width).ljust(width),
                style=theme_mod.semantic_color("muted"),
            )
            first = False
        for kind, entry in rows:
            if first:
                first = False
            else:
                result.append("\n")
            if kind == "blank":
                # Whitespace is the separator; the chrome stays borderless.
                result.append(" " * width)
                continue
            if kind.startswith("header:"):
                # A PEER heading carries its own label after the prefix, so a
                # section whose text depends on a device is not in the table at
                # all. The table below stays the TIER vocabulary — the four
                # names every device with no peers still paints — and the
                # fallback reads the label `_section_key` already derived, which
                # is why the two can never disagree about what a section is
                # called.
                if kind.startswith("header:" + _PEER_HEADER_PREFIX):
                    label = kind[len("header:" + _PEER_HEADER_PREFIX) :]
                else:
                    label = {
                        "header:pinned": "★ Pinned",
                        "header:active": "Active Sessions",
                        "header:previous": "Previous Sessions",
                        "header:subagent": "⌥ Subagent Runs",
                    }[kind]
                # Same treatment as the "Sessions" title above: `muted`, no
                # rule, no new palette entry. Mirrors the mobile relay's two
                # headings so the surfaces read the same.
                result.append(
                    truncate_cells(label, width).ljust(width),
                    style=theme_mod.semantic_color("muted"),
                )
                continue
            if kind == "note:pinned-overflow":
                # Keeps the `★ Pinned` heading honest when a pinned row ranks
                # below the page. Same muted treatment as a heading and no new
                # palette entry; it is chrome, so it is never a click target.
                missing = self._pinned_overflow([row for _k, row in rows if row is not None])
                # "— scroll" says what to DO about it; the count alone states a
                # fact and leaves the user to guess the remedy. The tail is
                # deliberately short of the designer's "— scroll to see": that
                # phrasing measures 30 cells against a 29-cell content width
                # and cropped to "scroll to s…" at N=1, and past N=9 the count
                # widens and crops it further. This form is 24 cells at its
                # worst (N=99) and never truncates.
                result.append(
                    truncate_cells(f"+{missing} more pinned — scroll", width).ljust(width),
                    style=theme_mod.semantic_color("muted"),
                )
                continue
            assert entry is not None
            current = entry.id == self.current_id
            cursor = self.has_focus and entry.id == self.cursor_id
            hovered = entry.id == self._hover_id and entry.id != ""
            # The row a click asked for, until it becomes current. It shares
            # the cursor's ground so the eye reads "this one is being opened"
            # where it just clicked, and stays distinct from current (bold on
            # `surface`) because it is NOT current yet — readiness is a
            # separate, later fact that only commit may assert.
            requested = entry.id == self._requested_id and entry.id != "" and not current
            # Three states have to stay separable, so they occupy three
            # different grounds: the keyboard cursor keeps `tint-select` (the
            # only tinted one), the attached session keeps `surface` plus bold,
            # and hover is `overlay` — the SAME ground `ToolCard:hover` and
            # `SubagentRow:hover` use, because pointing at a row should look
            # the same everywhere in this app. Hover yields to both of the
            # other two: it is transient pointer feedback, and must not
            # overpaint the identity of where you ARE or where the keyboard is.
            background = (
                ("tint-select-hi" if self.has_focus else "tint-select")
                if cursor or requested
                else "surface" if current else "overlay" if hovered else None
            )
            style = Style(
                color=theme_mod.semantic_color("fg"),
                bgcolor=theme_mod.semantic_color(background) if background else None,
                bold=current,
            )
            line = Text(style=style, no_wrap=True)
            # THE PIN CELL (issue #1357 slice 2a). The durable star leaves the
            # cursor-prefix slot for its own leading cell pair. Cell 0 of that
            # slot is shared between the caret and the star — only one of them
            # can show — so a pinned row that was also the cursor lost its star,
            # and for a pointer user the star's position was not a click target
            # at all: `★` and `›` lived in the same cell, and a press there has
            # always meant "open the row". With its own column the row carries
            # both facts at once (`★ ›`), the star never moves, and the cell is
            # a stable target whose press can only mean "toggle the pin" (see
            # `on_mouse_down` / `_pin_cell_pressed`).
            #
            # `☆` under the pointer is the unpinned row's affordance: an
            # unpinned row has no durable fact to paint, and the discovery
            # problem this slice closes is precisely that nothing on a resting
            # frame named the feature. It appears on the same hover event that
            # paints the row's `overlay` ground, so it is seen before it is
            # clicked; resting unpinned rows stay blank rather than carrying a
            # column of empty stars.
            if entry.id in self._pins:
                line.append(f"{PIN_MARK} ", style=theme_mod.semantic_color("accent"))
            elif hovered:
                line.append(f"{PIN_HOVER} ", style=theme_mod.semantic_color("muted"))
            else:
                line.append(" " * PIN_CELL_WIDTH)
            # The requested row gets a caret the cursor does not have. Sharing
            # BOTH `tint-select` and `›` made the two indistinguishable, and
            # they are reachable on opposite rows from real bindings (focus the
            # list, ctrl+shift+down, or press down before commit) — the frame
            # then could not say which row was opening (round 5, D6). `»` is
            # the doubled form of the same caret: one cell, same ink, same
            # column, so nothing reflows and the ramp is untouched. Requested
            # wins when a row is both, because "this is opening" is the fact
            # the user is waiting on.
            #
            # THE SLOT HOLDS TWO FACTS, NOT ONE (design round 1, D4). Taking both
            # cells for the caret was the defect: the one row the user is
            # deciding about was the one row that stopped saying it was remote,
            # and the peer heading that carries the fact independently is only
            # two lines above it, on a screen the user is scrolling. The slot's
            # first cell is the caret, its second is the locality mark; a local
            # row still paints two blanks, so the mark grows no row
            # (mesh-ui.md §1.3 decision 2, which this keeps).
            if requested or cursor:
                line.append("»" if requested else "›")
            else:
                line.append(" ")
            if entry.row.is_remote:
                # The LOCALITY mark: `⇄` occupies one cell of the slot the
                # caret run reserved, and it never displaces the caret. Muted,
                # like the other prefixes: it is a durable property, never a
                # state — it does not spin and never turns `danger`.
                line.append("⇄", style=theme_mod.semantic_color("muted"))
            else:
                # Nothing to say: this row is a session on THIS machine, which is
                # what the list has always shown. The blank keeps the title at
                # the same column a remote row puts it (mesh-ui.md §1.3
                # "local: no mark").
                line.append(" ")
            mark, ink = row_state_mark(entry.row, self._frame)
            special = self._special_mark(entry)
            asking = self._asking_mark(entry)
            if special is not None:
                # A subagent row has no live state to displace; see
                # `_special_mark`.
                mark, ink = special
            elif asking is not None:
                # An outstanding queued ask: the user owes an answer the agent
                # has already stopped waiting for. See `_asking_mark` for where
                # it sits in the urgency ladder.
                mark, ink = asking
            elif requested and self._requested_spinning():
                # Same ink a busy row's spinner uses (``row_state_mark``), so one
                # spinner means one thing everywhere in the list.
                # Only after the delay: a switch that is taking long enough to
                # notice gets a spinner ON THE ROW, where the eye is. The
                # footer "Opening…" stays as the textual counterpart.
                mark, ink = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)], "accent"
            elif entry.shows_completion_mark:
                # `shows_completion_mark`, not `unseen`, is the test: an unread
                # completion says what the session did LAST, and a busy or
                # wedged row is saying what it is doing NOW. The old condition
                # was `unseen and not pending`, which let a resumed session
                # paint its previous turn's mark over its own spinner —
                # `unseen` only clears on acknowledgement, and resuming does
                # not acknowledge. That predicate is shared with
                # `CatalogEntry.status` so this glyph and that tooltip cannot
                # disagree; see its docstring for the whole ordering argument.
                #
                # `interrupted` no longer borrows the error glyph. It is the
                # commonest of the three in practice — the operator's store
                # held 41 interrupted and zero errors — so the shared `✗` meant
                # the failure mark had, in practice, only ever been wrong.
                mark, ink = COMPLETION_MARKERS.get(
                    entry.completion_kind, COMPLETION_MARKERS["complete"]
                )
            line.append(f"{mark or ' '} ", style=theme_mod.semantic_color(ink))
            age = self._age(entry)
            title_width = max(1, width - _ROW_PREFIX_CELLS - (len(age) + 1 if age else 0))
            # A subagent row is identified by what it was delegated to do
            # ("label · role"), not by the session name the runtime generated
            # for it. `sub_title` already degrades to either half alone, and
            # to `row.name` when it has neither.
            name = entry.sub_title if entry.subagent else entry.row.name
            if self._aida_id is not None and entry.row.id == self._aida_id:
                # Her mark, prefixed before truncation so a narrow row eats the
                # title and never the mark (design round 1, D1).
                name = f"{AIDA_MARKER} {name}"
            title = truncate_cells(name or UNTITLED_CONVERSATION, title_width)
            if entry.subagent:
                # `sub_title` is "label · role", and the role is the half that
                # truncation eats first. When the cut lands on the separator
                # the row ends "label ·…" — a dangling separator reads as a
                # rendering fault rather than an ellipsis, so drop it. Which
                # HALF survives truncation is a separate question and belongs
                # to `sub_title` itself.
                title = _strip_dangling_separator(title)
            line.append(title)
            line.pad_right(max(0, width - line.cell_len - len(age)))
            line.append(age, style=theme_mod.semantic_color("dim"))
            result.append_text(line)
        if not self.entries:
            text = (
                "Could not load conversations"
                if self.error
                else (
                    f"{SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]} Loading conversations…"
                    if self._catalog_loading
                    else "No conversations yet"
                )
            )
            result.append("\n" + truncate_cells(text, width), style=theme_mod.semantic_color("dim"))
        while result.plain.count("\n") < self.size.height - 2:
            result.append("\n")
        footer, chip_span = self._footer_line(width)
        footer_style = theme_mod.semantic_color("warning" if self.error else "dim")
        if chip_span is not None and self._chip_hover:
            # The chip's affordance is a STYLE on its own cells, never new
            # text: the count stays on the frame in every state — a hover must
            # not steal it — and the ladder's arithmetic is untouched (`f9
            # focus · ctrl+b hide · ⌥1k+` still measures 29). Underline rather
            # than an ink step: the accent already means "a turn is live" and
            # the focus ground sits at near-iso-luminance, so neither ink is
            # the pointer's to spend. The resting path appends the SAME single
            # span it always did, so an un-hovered frame's bytes are unchanged.
            start, end = chip_span
            result.append("\n" + footer[:start], style=footer_style)
            result.append(
                footer[start:end], style=Style(color=footer_style) + Style(underline=True)
            )
        else:
            result.append("\n" + footer, style=footer_style)
        return result

    def _footer_line(self, width: int) -> tuple[str, tuple[int, int] | None]:
        """The footer as painted, plus the `⌥N` chip's span within it, if any.

        ONE implementation, shared by `render` (which paints the line) and
        `_chip_hit` (which resolves the chip's press target) — for the same
        reason `_entry_at` reads render's own header rule: a second copy of
        this ladder would drift from the painted bytes, and the press target
        would stop matching what the user sees. A `None` span means no chip is
        on the line — the error/loading/opening states, or a width where the
        ladder kept a chip-less fallback — so nothing is pressable.
        """
        # TWO rules, one helper, and they are the same rule applied to the
        # two ends of the list's keyboard mode. D4: the counter may never be
        # the reason the EXIT hint disappears — it used to REPLACE the hint
        # outright, so the focused frame that most needs to name a way out
        # (41 rows, cursor deep inside) read `1–32/41 · ctrl+b hide` and
        # `esc return` was gone. U1: the same counter left a full list
        # naming NOWHERE how to enter that mode — a whole-frame search for
        # `f9`/`focus` found nothing at 120x40 or 70x24, focused or not, so
        # the only route in (f9, or `/sidebar focus`) was advertised solely
        # by the copy the counter had just displaced.
        #
        # Longest form that fits, then the lead key WITH the position, then
        # today's unpaginated copy. The lead key is the exit when the list
        # has the keyboard and the way IN when it does not, so the counter
        # can never be the reason either one is missing. `ctrl+b hide` is
        # what gives way: the position is discoverable by scrolling (the
        # cursor row is painted) and the panel's own toggle is on the frame
        # that opens it.
        lead = "esc return" if self.has_focus else "f9 focus"
        base = lead if self.has_focus else f"{lead} · ctrl+b hide"
        # The chip is a PARTICIPANT in the ladder above, never an append after
        # it: appending past a candidate that has already been fit-tested voids
        # that test and the final `truncate_cells` then crops mid-word (the
        # blind append painted `1–10/21 · f9 focus · ⌥4…` at the 24-cell floor).
        # As a candidate it drops a fact WHOLE or not at all. It never yields,
        # because the position is recoverable by scrolling while the hidden
        # population is recoverable from nowhere else on the frame; `ctrl+b hide`
        # yields first and the position second (design round 4, §R4.3/§R4.4, on
        # top of main's D4/U1 above). The cap is `1k+` above 999 — one cell
        # cheaper than spelling the overflow out in three digits, and that cell
        # is binding at the 29-cell content width, where
        # `f9 focus · ctrl+b hide · ⌥1k+` is exactly 29 and survives while the
        # three-digit form makes 30 and drops `ctrl+b hide` whole.
        chip = ""
        if self._subagent_total > 0:
            chip = f"⌥{'1k+' if self._subagent_total > 999 else self._subagent_total}"
        tail = f" · {chip}" if chip else ""
        # THE FOCUSED LADDER (issue #1357 slice 2a). When the list holds the
        # keyboard its footer teaches the action set the keyboard now owns:
        # `f10 pin` from 27 cells of list width up — pinning was
        # documentation-only before this slice, and this is its one in-product
        # teacher (the pin cell's `☆` only appears under the pointer) — and
        # `ctrl+a ⌥`, the layer toggle, which has NO other in-product teacher:
        # `/help` deliberately excludes the two sidebar-scoped chords (they
        # would be lies outside f9 mode). Both are candidates, never appends,
        # and drop whole: ctrl+a needs the chip's rung, since the count it
        # flips is what it acts on (`⌥N`; nothing hidden, nothing said — same
        # rule the chip itself follows).
        #
        # THE MEASURED FLOORS, because "rides every rung" was an overstatement
        # (review round 1, MINOR 2). Reading down the widths, what yields is:
        # the paged four-fact form, then the position form, then the pin,
        # then the chip, never the lead (main's D4/U1). With a chip, the pin
        # rung (`esc return · f10 pin · ⌥1k+`) is 27 cells and renders at
        # >= 27; between 17 and 26 the chip outranks it (`esc return · ⌥1k+` —
        # a 30-column terminal lands exactly here, and NO pin is taught);
        # below 17 only the lead remains. `ctrl+a ⌥` needs the chip and 38
        # cells — `esc return · f10 pin · ctrl+a ⌥ · ⌥1k+` is 38 and is tried
        # AFTER the position form, so it renders exactly where that form does
        # not fit: unpaged, every width >= 38; paged, the band from 38 up to
        # one cell under `{position} · esc return · f10 pin · ⌥1k+`, which is
        # EMPTY for a position string of <= 8 cells (`1–18/21`: that form is
        # 38 and wins at 38) and OPENS for longer ones — a 152-entry list on
        # a deep page (`128–152/152`, its form 41) shows the layer rung at
        # content 38–40 and the position form returns from 41 (review round
        # 2 measured exactly this; round 1's "a paged list never shows it"
        # was false). Every width here moves with the position string AND
        # the chip — the paged four-fact form is 48/49/52 at 7/8/11 position
        # cells, and a two-digit chip shifts the whole band two cells down —
        # so the fit test, not this comment, is the authority (design round
        # 1, D1). `ctrl+o` is deliberately absent: no spelling of it fits a
        # real content width beside the chip and the pin, and a bare chord
        # would break this footer's key-and-what-it-does contract — recorded
        # on the PR rather than smuggled in as an append.
        if self.has_focus:
            pin = f"{lead} · f10 pin"
            layer = " · ctrl+a ⌥" if chip else ""
            candidates: list[str] = []
            if len(self.entries) > self.page_size:
                last = min(len(self.entries), self._offset + self.page_size)
                position = f"{self._offset + 1}–{last}/{len(self.entries)}"
                if chip:
                    candidates.append(f"{position} · {pin}{layer}{tail}")
                candidates.append(f"{position} · {pin}{tail}")
            if chip:
                candidates.append(f"{pin}{layer}{tail}")
            candidates += [f"{pin}{tail}", f"{lead}{tail}", lead]
        else:
            if len(self.entries) > self.page_size:
                last = min(len(self.entries), self._offset + self.page_size)
                position = f"{self._offset + 1}–{last}/{len(self.entries)}"
                candidates = [
                    f"{position} · {lead} · ctrl+b hide{tail}",
                    f"{position} · {lead}{tail}",
                ]
            else:
                candidates = [f"{base}{tail}"]
            if chip:
                candidates.append(f"{lead} · {chip}")
        hint = next(
            (
                candidate
                for candidate in candidates
                if truncate_cells(candidate, width) == candidate
            ),
            base,
        )
        footer = "Refresh failed" if self.error else "Opening…" if self.requested_id else hint
        painted = truncate_cells(footer, width)
        # The chip is the ladder's SUFFIX whenever it is painted — every
        # chip-carrying candidate ends with ` · ⌥N` — so its span is the last
        # `len(chip)` characters of the truncated line; `None` otherwise.
        # `len` is the cell count for every glyph this ladder can emit (`⌥`,
        # `·`, digits, ASCII) — the single-cell property the width comments
        # above already rest on.
        span: tuple[int, int] | None = None
        if chip and painted.endswith(chip):
            span = (len(painted) - len(chip), len(painted))
        return painted, span
