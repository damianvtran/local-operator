"""The full-page ``/settings`` view — every configurable value, in sections.

A MODE of the main screen, cloned from :class:`OrgChartView` and, through it,
:class:`SubagentView`: the page takes the transcript's region and leaves the
dock (band, status, composer) where it is, greyed, so it reads as the same app
looking somewhere else.

WHY A MODE AND NOT A MODAL
==========================

``analytics_panel`` is a ``ModalScreen`` and says why that is acceptable there:
that surface "never mutates anything". This one mutates on every Enter, and a
modal blacks out the dock — which is exactly where the status band shows a
change landing (a theme switch, a model default, an approval mode). A user who
cannot see the effect of the write they just made has to take the page's word
for it. A docked card was the other candidate and is too small: ``ask_picker``
caps its card at ``PROMPT_HEIGHT_SHARE = 0.7`` of the screen, and this content
is a fifty-row list with a second pane beside it.

WHAT THIS PAGE OWES THE USER THAT A LIST DOES NOT
=================================================

- **Scope.** Writes are immediate (see ``settings_io``), so "when does this
  take effect?" is a live question. The answer is a dim tag on the SECTION
  header — per row it would be fifty tags of noise, and scope is uniform within
  a section by construction.
- **Undo.** Immediate-write's one real cost. ``r`` resets the highlighted row
  to its shipped default, and a changed row is styled so the thing to reset is
  findable without remembering what shipped.
- **A way back.** The footer names ``esc``, and sheds clauses widest-first the
  way the org chart's does, so a 50-column terminal still says how to leave.

Identified by CLASS and not id, all the way down — the ``DuplicateIds``-on-
fast-reopen lesson ``org_chart_view`` records: ``remove()`` only POSTS a prune,
so a reopen inside that window would mount a second same-id widget and raise
out of a click handler.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Static

from local_operator import settings_io
from local_operator.settings_io import Kind, Section, Setting
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.subagent_view import HintButton
from local_operator.tui.widgets.tool_card import truncate_cells

#: The read-only pane's width in cells, MIRRORED from `.settings-view-pane` in
#: local_operator.tcss. Duplicated deliberately: the widths have to be known
#: before layout runs (see `_list_width`), and Textual offers no way to read a
#: rule's value. `test_settings_view_pane_width_matches_tcss` asserts the two
#: agree, which is the guard the analytics panel wanted between its
#: `max-width` and `_card_width`.
_PANE_WIDTH = 34

#: Kinds the inline text editor may open on. Written as an ALLOW-list, not a
#: deny-list, so a kind added to `settings_io.Kind` without an
#: `action_activate` branch is refused by default rather than silently handed
#: to a free-text editor. That silence is what made #440 destructive:
#: `Kind.CASCADE` fell through, the editor seeded itself with the mapping's
#: Python repr, and committing it overwrote the user's whole failover cascade
#: with a string. A missing branch should cost a key that says it cannot act,
#: never a config the user cannot get back.
#:
#: `Kind.LIST` is in here on purpose — `web_search.providers` is stored as a
#: list and genuinely edited as comma-separated text (see `settings_io.coerce`),
#: which is why this is keyed on the interaction rather than on the stored type.
_TEXT_EDITABLE_KINDS = frozenset({Kind.INT, Kind.FLOAT, Kind.TEXT, Kind.LIST})

#: One line of the read-only pane, as its styled segments. A LIST of segments
#: rather than one string plus one style, because a provider row carries two
#: inks on one line (the id in `muted`, its state in `faint`) and the pane's
#: fitting pass (`_fit_pane`) has to count lines before they are painted.
_PaneLine = list[tuple[str, "Style"]]

#: One footer hint: the button, its trailing label, and whether a separator
#: precedes it. A module-scope alias because a function-local assignment is not
#: a valid type expression, and the literal labels otherwise infer as distinct
#: `Literal` types that do not unify into the list `_paint_hints` builds.
_Hint = tuple["HintButton", str, bool]

#: Left indent of a setting row's label, so rows read as children of the
#: section header above them rather than as siblings of it.
_ROW_INDENT = 2
#: Column the value starts at. Fixed rather than measured from the longest
#: label: a value column that moved as the user paged between sections would
#: read as the page reflowing under them.
_VALUE_COLUMN = 34
#: Extra indent for an expanded enum's choices — one level in from the row that
#: owns them, which is what makes the expansion read as belonging to that row.
_CHOICE_INDENT = _ROW_INDENT + 4

#: Rows one wheel notch moves the VIEWPORT by. Matches Textual's own
#: `App.scroll_sensitivity_y` (2.0), because the body's container handles the
#: wheel directly over the list and this method handles it everywhere else on
#: the page — a different step here would make one gesture travel at two speeds
#: depending on where the pointer sat.
_WHEEL_ROWS = 2

#: Marker on the row the cursor is on. A glyph rather than a background sweep
#: so the row's own changed/default ink survives the highlight.
_CURSOR = "›"
#: Cells the cursor marker occupies on every row, cursor or not — the column is
#: reserved on unselected rows too, or the labels would shift sideways as the
#: cursor moved down the page.
_MARKER_WIDTH = 2

#: Cells a row label may occupy. Derived from the value column rather than
#: guessed, so the widest label still leaves exactly one space before its value
#: and the value column stays fixed: `Connectivity backoff cap (ms)` is 29 cells
#: and fits whole, where a hand-picked budget one cell shorter clipped it to
#: `(m…` — an opened parenthetical that never closes reads as a rendering fault
#: rather than as an abbreviation (design round 1, D4).
_LABEL_BUDGET = _VALUE_COLUMN - _ROW_INDENT - _MARKER_WIDTH - 1

#: Rows of the page's own chrome that sit outside the two columns: the title,
#: the rule, the columns' top padding, the two-row detail line, and the two-row
#: hint footer. The pane's height is derived from the VIEW's height through this
#: rather than read off `_pane_view.size`, for the reason `_pane_width` records
#: — `_repaint` runs from `on_mount`, before layout, where every child size is
#: still 0, and a pane budgeted against 0 rows paints a different first frame
#: from its settled one.
_PANE_CHROME_ROWS = 7

#: The right-hand pane's two read-only tabs. Read-only on purpose: teams and
#: agents are configured in files, and a page that showed an editable-looking
#: roster it could not write would be worse than one that admits the boundary.
_PANE_TEAMS = "teams"
_PANE_AGENTS = "agents"


class SettingsViewDismissed(Message):
    """The page's ``esc`` hint was clicked. The app owns leaving the mode.

    A DEDICATED message (not the org chart's or the subagent view's) so the
    app's Esc-chain routes the click to ``_close_settings_view`` — reusing
    another mode's message would hit a handler that does not own this widget.
    """


class SettingsChanged(Message):
    """A setting was written. Carries the key so the app can react live.

    The app applies the handful of settings that have an immediate effect on a
    RUNNING session (the theme, in particular) rather than the page doing it:
    the page knows how to store a value, the app knows what a value means to
    the surfaces already on screen, and putting both here would make this
    widget import half the app.
    """

    def __init__(self, key: str, value: Any) -> None:
        super().__init__()
        self.key = key
        self.value = value


class SettingsView(Vertical):
    """The page: title, a rule, the settings body, a side pane, and the footer.

    Class-identified (see the module docstring). ``can_focus`` so the movement
    and edit keys land here rather than on the composer the mode made inert.
    """

    can_focus = True

    # Movement WRAPS on the arrows (a discrete, deliberate press) and CLAMPS on
    # page and wheel — the convention `session_picker` and `model_picker` both
    # follow, and the reason is the same: a scroll gesture that teleports to the
    # other end of the list reads as the list resetting itself.
    #
    # ctrl+p/ctrl+n as well as the arrows because every printable key belongs to
    # the filter (cf. session_picker.py), so the readline pair is the only other
    # way to move a hand that is already typing.
    #
    # Bindings are show=False: the footer hints are the visible affordance, the
    # same split every other mode in this app uses.
    BINDINGS = [
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        Binding("ctrl+p", "move(-1)", "Up", show=False),
        Binding("ctrl+n", "move(1)", "Down", show=False),
        Binding("pageup", "section(-1)", "Previous section", show=False),
        Binding("pagedown", "section(1)", "Next section", show=False),
        Binding("home", "jump(0)", "First", show=False),
        Binding("end", "jump(1)", "Last", show=False),
        Binding("enter", "activate", "Change", show=False),
        Binding("space", "activate", "Change", show=False),
        # left/right switch the side pane's tab. They do NOT move within a row:
        # every editable row here is either a list to expand or a field to type
        # into, so a horizontal cursor would have nothing to travel along.
        Binding("left", "pane(-1)", "Previous pane", show=False),
        Binding("right", "pane(1)", "Next pane", show=False),
        Binding("r", "reset", "Reset to default", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(self, manager: Any) -> None:
        super().__init__(classes="settings-view")
        #: The ConfigManager every read and write goes through. Held rather
        #: than re-derived per row: a fresh manager per read would re-parse
        #: config.yml on every repaint.
        #:
        #: Holding it is only safe because ``settings_io`` reloads the manager
        #: before every write (``_reload_before_write``). This instance is
        #: long-lived and ``set_config_value`` dumps the manager's whole
        #: in-memory snapshot, so without that reload a single row toggle here
        #: would write back a stale copy of the ENTIRE file and revert whatever
        #: ``/theme`` or another session changed while the page sat open
        #: (review round 1, B1). Do not add a write path that bypasses
        #: ``settings_io``.
        self._manager = manager
        #: Rows are rebuilt from the registry on every repaint (cheap: it is a
        #: tuple walk) so a write is reflected without a second bookkeeping
        #: path that could disagree with the config.
        self._rows: list[_Row] = []
        self._selected = 0
        self._hovered: int | None = None
        #: Key of the row whose enum choices are expanded, or None. One at a
        #: time: two open dropdowns in a vertical list make the second one's
        #: choices read as the first one's.
        self._expanded: str | None = None
        #: The row being text-edited, its buffer, and the last rejection. The
        #: error is held on the WIDGET rather than flashed as a notice because
        #: it has to survive the repaint that follows the rejected Enter — an
        #: error that scrolled away with the notice would leave the editor open
        #: with no statement of what was wrong with what is still in it.
        self._editing: str | None = None
        self._buffer = ""
        self._error = ""
        #: An INFORMATIONAL message for the detail row, kept apart from
        #: `_error` so the two can be inked differently. Same clear-on-move
        #: lifetime, because it answers a keypress and stops being true as soon
        #: as the cursor is somewhere else — but painting it in the danger ink
        #: said "you did something wrong" about a press the footer advertises
        #: (UX round 2, U16).
        self._notice = ""
        #: Caret position INSIDE the buffer. The editor owns left/right/home/end
        #: while it is open (UX round 1, U2): unhandled they fell through to the
        #: page bindings and switched the read-only side pane mid-typing, and
        #: `home` discarded the edit and teleported the cursor to row 0. Fixing
        #: one character of a long endpoint otherwise costs retyping the tail.
        self._caret = 0
        #: What the editor was seeded with, so leaving a buffer the user never
        #: altered can skip the write entirely rather than rewriting the same
        #: value (see :meth:`_settle_edit`).
        self._edit_seed = ""
        #: Cascade editor state: which chain is open for hop editing, if any.
        self._chain: str | None = None
        #: Chain key whose deletion has been asked for and not yet confirmed.
        #: `d` on a CHAIN row destroys every hop in it and immediate-write has
        #: no undo, so it asks first — a magnitude above `d` on a single hop,
        #: which is one line and cheap to retype (UX round 1, U5).
        self._confirm_delete: str | None = None
        self._pane = _PANE_TEAMS
        #: Read-only pane content, injected by the app (it owns the registries).
        self._teams: list[tuple[str, str, str]] = []
        self._agents: list[tuple[str, str, str]] = []
        #: Provider login state, injected the same way and for the same reason.
        self._providers: list[tuple[str, str]] = []

        self._detail = Static(classes="settings-view-detail")
        self._title = Static(classes="settings-view-title")
        self._rule = Static(classes="settings-view-rule")
        self._list = Static(classes="settings-view-list")
        self._body = ScrollableContainer(self._list, classes="settings-view-body")
        # NOT focusable. With it in the focus chain, one `tab` moved focus from
        # the page to this container, which owns the scroll keys — so the arrows
        # then scrolled the VIEWPORT while the cursor stayed where it was, with
        # no focus ring or any other cue that focus had moved. `enter` still
        # bubbled to the page, so a user looking at rows 24-37 with the cursor
        # stranded on row 1 pressed `enter` on the row they could see and opened
        # an editor on a row off screen, then wrote a setting they never chose
        # (UX round 3, U19). Same remedy and same reasoning as `todo_panel.py`
        # and `subagent_panel.py`: "the app looked correctly focused while every
        # keystroke went to a widget that does nothing with them". Mouse-wheel
        # scrolling does not require focus, so the list still scrolls with the
        # wheel, and `_scroll_to_selection` drives this container programmatically
        # rather than through focus.
        self._body.can_focus = False
        self._pane_view = Static(classes="settings-view-pane")
        self._columns = Horizontal(self._body, self._pane_view, classes="settings-view-columns")
        # Footer hints, same vocabulary and same shedding ladder as the org
        # chart's, so the two modes read consistently.
        self._move_hint = HintButton("↑↓", self._focus_self)
        self._enter_hint = HintButton("enter", lambda: self.action_activate())
        self._reset_hint = HintButton("r", lambda: self.action_reset())
        self._pane_hint = HintButton("←→", lambda: self.action_pane(1))
        self._exit_hint = HintButton("esc", self._leave)
        self._hints = Horizontal(classes="settings-view-hints")
        self._title_text = Text()
        self._rule_text = Text()
        self._list_text = Text()
        self._detail_text = Text()
        self._pane_text = Text()

    # -- composition --------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        with self._columns:
            yield self._body
            yield self._pane_view
        yield self._detail
        with self._hints:
            yield self._move_hint
            yield self._enter_hint
            yield self._reset_hint
            yield self._pane_hint
            yield self._exit_hint

    def on_mount(self) -> None:
        # Focus lands here rather than at the app's open call: focus() on a
        # widget not yet in the focus chain is a silent no-op, which is the bug
        # the subagent view records — the advertised keys would go to the inert
        # composer. Repaint after focus so the first frame is the settled one.
        self._repaint()
        try:
            self.focus()
        except Exception:
            pass

    def on_resize(self) -> None:
        # The rule spans the page and the hints shed against a width only the
        # layout knows, so both are repainted on resize.
        self._repaint()

    # -- data ---------------------------------------------------------------
    def load(
        self,
        *,
        teams: Sequence[tuple[str, str, str]] = (),
        agents: Sequence[tuple[str, str, str]] = (),
        providers: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Point the page at the read-only content the app resolved for it.

        Teams, agents and provider state are RESOLVED BY THE APP (they read the
        session's registries and the credential store) and handed here already
        flattened to rows. The page never reaches into a registry itself, which
        is what keeps a repaint free of I/O — the same split ``OrgChartView``
        makes when the app resolves the org tree and the widget only paints it.
        """
        self._teams = list(teams)
        self._agents = list(agents)
        self._providers = list(providers)
        self._repaint()

    # -- rows ---------------------------------------------------------------
    def _build_rows(self) -> list["_Row"]:
        """The flat row list the cursor travels, rebuilt from the registry.

        Flat rather than a tree, because a cursor that had to know whether it
        was on a header, a setting, or a choice would need three movement rules;
        one list with a ``selectable`` flag needs one. Headers are rows so
        PgUp/PgDn has something to land on and so a section's scope tag has a
        row to live on.
        """
        rows: list[_Row] = []
        for section in settings_io.SECTIONS:
            rows.append(_Row(kind="header", section=section))
            for setting in settings_io.settings_for(section.name):
                rows.append(_Row(kind="setting", setting=setting))
                if setting.kind is Kind.CASCADE:
                    rows.extend(self._cascade_rows(setting))
                elif self._expanded == setting.key:
                    # `resolved_choices`: a registry-sourced enum (tui.theme)
                    # declares no static choices, so `choices` would expand to
                    # an empty group and read as the expansion having failed.
                    for choice in setting.resolved_choices:
                        rows.append(_Row(kind="choice", setting=setting, choice=choice))
        return rows

    def _cascade_rows(self, setting: Setting) -> list["_Row"]:
        """The failover cascade's two levels: a row per chain, then its hops.

        Hops are only listed for the OPEN chain. A cascade with four chains of
        five hops would otherwise put twenty rows in the middle of the settings
        list, burying every section below it — the same reason the enum choices
        are an expansion rather than always-on.
        """
        rows: list[_Row] = []
        chains = settings_io.read_chains(self._manager)
        if not chains:
            # A cascade the page cannot read says WHY it shows nothing, and
            # names the key that fixes it. Without this the only signal a #440
            # victim got was `no cascade configured` under a value column
            # showing their repr, and nothing on the page ever admitted the
            # stored value was broken (UX round 1, U2). Same predicate as the
            # value column and as `action_reset`, so the three cannot drift.
            text = (
                "malformed cascade — press r to clear it"
                if self._cascade_is_malformed(setting)
                else "no cascade configured"
            )
            rows.append(_Row(kind="empty", text=text))
        for key, hops in sorted(chains.items()):
            rows.append(_Row(kind="chain", chain=key, setting=setting))
            if self._chain == key:
                for index, hop in enumerate(hops):
                    rows.append(_Row(kind="hop", chain=key, hop=hop, hop_index=index))
                rows.append(_Row(kind="hop_add", chain=key))
        rows.append(_Row(kind="chain_add", setting=setting))
        return rows

    def _cascade_is_malformed(self, setting: Setting) -> bool:
        """Whether the cascade's STORED value is not a mapping at all.

        `read_chains` answers ``{}`` for both "unset" and "unreadable", so it
        cannot tell those apart — and they are different things to say to a
        user. The raw read can: an unset key falls back to the setting's ``{}``
        default, while the #440 wreckage reads back as the ``str`` that was
        written over it. THE one predicate for that state, shared by the row
        painter, the empty-state line and ``action_reset``, because three
        copies of it would be three chances for the frame to contradict itself
        again (UX round 1, U1/U2).
        """
        return not isinstance(settings_io.read_setting(self._manager, setting), Mapping)

    def _selectable(self) -> list[int]:
        return [index for index, row in enumerate(self._rows) if row.selectable]

    def _current(self) -> "_Row | None":
        if 0 <= self._selected < len(self._rows):
            return self._rows[self._selected]
        return None

    # -- movement -----------------------------------------------------------
    def action_move(self, delta: int) -> None:
        """Move the cursor by ``delta`` selectable rows, CLAMPED.

        This page is the documented EXCEPTION to the repo's wrap-vs-clamp rule
        (AGENTS.md, "Wrapping vs clamping"), which has arrow keys wrap because a
        press is deliberate. That reasoning holds for a short picker, where the
        whole list is on screen and coming round is a shortcut to a row the user
        can already see. It does not hold here: the settings are a long,
        sectioned, SCROLLED list (60-odd rows against a viewport of 14 at
        100x30), so the bottom is a destination travelled to rather than a place
        stumbled onto, and wrapping teleported the reader from the section they
        were working in to the top of the page with the viewport jumping with
        them. Reported against v0.43.0: the bottom is expected to hold.

        Clamping also makes the page agree with ITSELF. The wheel
        (:meth:`_scroll_rows`) and paging (:meth:`action_section`) already
        clamp, so `down` and a wheel notch on the same last row disagreed about
        what the end of the list means — worse than either rule applied
        uniformly.
        """
        if not self._selectable():
            return
        # The row is remembered by IDENTITY and the list is re-derived AFTER
        # the commit, never before it. `_leave_row` writes, and a write on an
        # add row inserts rows above the cursor, so stepping from an index
        # snapshotted beforehand lands on a row the user did not ask for —
        # committing a chain with `down` put the cursor back on `+ add a chain`,
        # which reads as the arrow key being dead (review round 2, U13).
        # `_close_chain` already resolves its target this way for the same
        # reason, and this is that pattern applied to movement.
        # An arrow after the wheel has scrolled the cursor off screen spends
        # itself bringing the cursor BACK into view rather than moving it, so
        # the row that moves is always one the user can see. Without it, `down`
        # from a viewport 40 rows away moves a cursor nobody is looking at and
        # then teleports the view to it, which is the bounce wearing a
        # different hat (see `_reorient`).
        if self._reorient():
            return
        anchor = self._current()
        identity = anchor.identity if anchor is not None else None
        if not self._leave_row():
            return
        indices = self._selectable()
        if not indices:
            return
        position = self._position_of(identity, indices)
        self._selected = indices[min(max(position + delta, 0), len(indices) - 1)]
        self._repaint()
        self._scroll_to_selection()

    def _position_of(self, identity: "tuple[str, str, int] | None", indices: list[int]) -> int:
        """Where the row with ``identity`` sits in the REBUILT selectable list.

        Falls back to the current index, then to the top, because a commit can
        make the anchor row stop existing entirely (an empty chain is dropped on
        write) and the movement still has to land somewhere real.

        The common fallback is the useful one: when the identity is gone but
        ``_selected`` is still selectable, the current index is returned and the
        move proceeds normally from where the user was. The top is the last
        resort, and under the clamp it is a genuine dead end \u2014 ``action_move(-1)``
        from position 0 does not move (review round 1, F4). That is accepted
        rather than papered over: both outcomes have already teleported the
        cursor away from the row the user was on, which is the real fault, and a
        wrap here would hide it by jumping to the bottom of a 60-row page.
        """
        if identity is not None:
            for position, index in enumerate(indices):
                if self._rows[index].identity == identity:
                    return position
        if self._selected in indices:
            return indices.index(self._selected)
        return 0

    def action_section(self, delta: int) -> None:
        """PgUp/PgDn jump to the next section header's first row, CLAMPED."""
        # The section the cursor is in is resolved by the header ABOVE it, so
        # it has to be captured before `_select_after` commits and rebuilds —
        # a commit that inserts chain rows moves every header index below it,
        # and paging then landed inside the chain just created rather than on
        # the next section (review round 2, U13). The header's own section is
        # the identity that survives the rebuild.
        headers = [index for index, row in enumerate(self._rows) if row.kind == "header"]
        if not headers:
            return
        current = max([index for index in headers if index <= self._selected] or [headers[0]])
        position = headers.index(current)
        # Clamped: paging past the last section should sit on the last section,
        # not wrap to the first. A page gesture is travel, and travel that
        # teleports across the whole document is how a reader loses their place.
        target = min(max(position + delta, 0), len(headers) - 1)
        # ...but the LAST section is not the last row, and paging has to be able
        # to reach the end the other gestures reach. Held `down` and `end` both
        # finish on the final row; `pagedown` settled on the last section's
        # first row and stopped with five rows still below it, so the page had
        # two different answers for where it ends (UX round 1, U3). Once paging
        # can advance no further by section it finishes the journey to the last
        # row. `pageup` needs no counterpart: the first section's first row IS
        # the first selectable row, so travelling up already terminates there.
        if delta > 0 and target == position:
            indices = self._selectable()
            if indices and self._selected != indices[-1]:
                if not self._leave_row():
                    return
                indices = self._selectable()
                if indices:
                    self._selected = indices[-1]
                    self._repaint()
                    self._scroll_to_selection()
            return
        wanted = self._rows[headers[target]].section
        if not self._leave_row():
            return
        rebuilt = [index for index, row in enumerate(self._rows) if row.kind == "header"]
        landing = next(
            (index for index in rebuilt if self._rows[index].section is wanted),
            rebuilt[min(target, len(rebuilt) - 1)] if rebuilt else None,
        )
        if landing is None:
            return
        self._select_after_index(landing)

    def action_jump(self, to_end: int) -> None:
        if not self._selectable():
            return
        if not self._leave_row():
            return
        # Re-derived after the commit for the reason `action_move` records: the
        # ends of the list move when a commit inserts or drops rows.
        indices = self._selectable()
        if not indices:
            return
        self._selected = indices[-1] if to_end else indices[0]
        self._repaint()
        self._scroll_to_selection()

    def _select_after(self, header_index: int) -> None:
        """Put the cursor on the first selectable row at or after a header."""
        if not self._leave_row():
            return
        self._select_after_index(header_index)

    def _select_after_index(self, header_index: int) -> None:
        """Land on the first selectable row at or after an ALREADY-VALID index.

        Split from :meth:`_select_after` so a caller that has re-resolved the
        header itself after a commit does not commit a second time.
        """
        for index in range(header_index, len(self._rows)):
            if self._rows[index].selectable:
                self._selected = index
                break
        self._repaint()
        self._scroll_to_selection()

    def action_pane(self, delta: int) -> None:
        """←/→ switch the read-only side pane between teams and agents.

        Inert when the pane is hidden (a narrow terminal): switching a tab
        nobody can see would be a keypress with no visible effect, which reads
        as the key being broken rather than as the pane being absent.
        """
        if not self._pane_fits():
            return
        # Disarms a pending delete for the reason `action_activate` records: a
        # key the footer advertises that is not `d` or `esc` cancels the ask.
        self._confirm_delete = None
        panes = [_PANE_TEAMS, _PANE_AGENTS]
        # CYCLES, and stays cycling while the list movement above clamps. This
        # is not the same gesture: two tabs, both labelled and both on screen,
        # are a closed cycle rather than a list with ends — there is no "bottom"
        # to travel to and nothing scrolls, so `→` on the last tab has no
        # meaning other than the first one. Clamping it would make the second
        # press of a two-tab toggle silently dead, which is the stuck key the
        # wrap convention exists to avoid. The clamp on `action_move` is about
        # losing your place in a long scrolled list; neither term applies here.
        position = (panes.index(self._pane) + delta) % len(panes)
        self._pane = panes[position]
        self._repaint()

    def _cursor_visible(self) -> bool:
        """Is the highlighted row inside the viewport right now?

        The wheel moves the VIEWPORT and leaves the cursor where it is (see
        :meth:`_scroll_rows`), so the two can legitimately disagree — which is
        the whole point of that model, and also its one hazard. This is the
        predicate the acting keys gate on so that hazard cannot become a write.
        """
        try:
            height = self._body.size.height
        except Exception:
            return True
        if height <= 0:
            return True
        offset = self._body.scroll_offset.y
        return offset <= self._selected < offset + height

    def _reorient(self) -> bool:
        """Bring an off-screen cursor back into view WITHOUT acting. True if it did.

        The safety interlock for the wheel-scrolls-the-viewport model. Letting
        the wheel leave the cursor behind is what makes a scrollbar honest, but
        it re-opens the exact defect ``_body.can_focus = False`` was set for
        (UX round 3, U19): a user looking at rows 24-37 with the cursor stranded
        on row 1 presses ``enter`` on the row they can SEE and writes a setting
        they never chose. There the viewport moved because focus had silently
        shifted; here it moves because the user asked it to — but the blind
        write at the end is identical, and this page writes config immediately
        with no undo beyond ``r``.

        So no key ACTS on a row that is not on screen. The first press spends
        itself putting the cursor back in view, the second one acts, and the
        row being acted on is always a row the user is looking at. Deliberately
        does NOT commit anything — it is a pure viewport move, so it stays
        correct regardless of when a value is taken to commit.
        """
        if self._cursor_visible():
            return False
        self._scroll_to_selection()
        self._repaint()
        return True

    def _scroll_to_selection(self) -> None:
        """Keep the cursor row inside the scrolled viewport.

        The body is a ScrollableContainer around ONE painted Static, so there
        is no child widget to call ``scroll_visible`` on — the offset is
        computed from the row index directly. Guarded because the container has
        no size until it is laid out, and a first movement before layout would
        divide by a zero height.

        This is the CURSOR's claim on the viewport, and it runs only from a key
        that moved the cursor. The wheel has its own claim and the two are kept
        apart on purpose — see :meth:`_scroll_rows`.
        """
        try:
            height = self._body.size.height
        except Exception:
            return
        if height <= 0:
            return
        offset = self._body.scroll_offset.y
        # Scroll far enough to show the row's own SECTION HEADER, not just the
        # row. Headers are unselectable, so travelling to the first row of the
        # page settled at `scroll_y=1` with the `Model` header one line off the
        # top edge: a highlighted row whose section title is missing and a
        # scrollbar thumb not quite at the start of its track. The bottom end
        # reads as arrival and the top did not, and the clamp is what made
        # users dwell there long enough to notice (UX round 1, U1). Only the
        # contiguous run of headers directly above the row is included, so this
        # reveals the row's own title and never scrolls past unrelated content.
        top = self._selected
        while top > 0 and self._rows[top - 1].kind == "header":
            top -= 1
        if top < offset:
            self._body.scroll_to(y=top, animate=False)
        elif self._selected >= offset + height:
            self._body.scroll_to(y=self._selected - height + 1, animate=False)

    # -- activation ---------------------------------------------------------
    def action_activate(self) -> None:
        """Enter/Space on the highlighted row: the row decides what that means.

        One key for every row kind, because "the obvious thing" is unambiguous
        per kind: a bool toggles, an enum expands (and a choice inside it
        commits), text opens an editor, a chain expands, an add row starts a
        new entry. A page with a different key per kind would be a page whose
        footer cannot state its own contract in one clause.
        """
        row = self._current()
        if row is None:
            return
        # `enter` NEVER writes a row that is off screen. The wheel can leave the
        # cursor behind by design, and this page writes config immediately, so
        # an activate aimed at "the row I can see" must not land on the row the
        # cursor happens to still be on — the U19 defect exactly. The press
        # reorients instead, and the next one acts on a visible row.
        if self._editing is None and self._reorient():
            return
        # Any action that is not `d` or `esc` DISARMS a pending delete. The ask
        # was cleared by a cursor move and by `esc`, but not by a key acting on
        # the same row: `enter` toggled the chain open underneath a question
        # about deleting that chain, and a later `d` — pressed to look inside
        # before deciding — still deleted it (UX round 2, U18). The rule the
        # ask's own wording implies is "answered by `d` or `esc`, cancelled by
        # anything else".
        self._confirm_delete = None
        if self._editing is not None:
            self._commit_edit()
            return
        if row.kind == "choice" and row.setting is not None and row.choice is not None:
            # The cursor goes back to the SETTING row, not to the index the
            # choice occupied. A choice row stops existing when the expansion
            # collapses — the same hazard `_close_chain` documents for hops —
            # so the index left behind names whatever the rebuild put there:
            # picking the 34th theme left the cursor 34 rows away on an
            # unrelated setting two sections down, with `r default` lit on it
            # (UX round 2, U17). Harmless-looking on a 2-choice enum, which is
            # why it went unnoticed until `tui.theme` grew 35 members.
            owner = row.setting.key
            self._write(row.setting, row.choice.value)
            self._expanded = None
            self._repaint()
            self._select_setting_row(owner)
            self._repaint()
            self._scroll_to_selection()
            return
        if row.kind == "chain" and row.chain is not None:
            self._chain = None if self._chain == row.chain else row.chain
            self._repaint()
            # Reveal the hops, for the reason the enum expansion does: a chain
            # whose `▾` says open with nothing visible under it reads as the
            # press having failed.
            self._scroll_to_expansion()
            return
        if row.kind in ("hop", "hop_add", "chain_add"):
            self._begin_edit(row)
            return
        setting = row.setting
        if setting is None:
            return
        if setting.kind is Kind.BOOL:
            # A bool has exactly two states, so Enter TOGGLES rather than
            # expanding a two-item list: an expansion that always shows the
            # same two rows is a click the user pays for every time.
            current = settings_io.read_setting(self._manager, setting)
            self._write(setting, not bool(current))
            return
        if setting.kind is Kind.ENUM:
            self._expanded = None if self._expanded == setting.key else setting.key
            self._repaint()
            # Reveal the CHOICES, not just the row that owns them. Scrolling
            # the selected row into view is not enough: expanding a row sitting
            # on the viewport's bottom edge put every choice below the fold, so
            # the frame showed a `▾` marker and nothing under it and the
            # expansion read as having failed to open.
            self._scroll_to_expansion()
            return
        if setting.kind is Kind.READONLY:
            self._error = "this setting is retired and cannot be changed"
            self._repaint()
            return
        if setting.kind is Kind.CASCADE:
            # A cascade has NO scalar to edit — it is a mapping of chains to
            # ordered hops, and its editor is the two levels of rows already
            # painted underneath this one. Without this branch the row fell
            # through to `_begin_edit`, which seeded a free-text editor with
            # `str(dict)`, and accepting that repr stored it as a STRING:
            # `read_chains` then returned `{}`, the whole cascade was gone, and
            # `r` could not bring it back because the stored value was no
            # longer a mapping. A silent, unrecoverable loss of the user's own
            # configuration from one press of the key the footer advertises
            # (#440).
            #
            # `enter` therefore travels INTO the group rather than doing
            # nothing: the footer offers `enter change` on this row, and a lit
            # hint whose key is inert is the same "nothing happens when I
            # click" complaint U5 records. The first selectable row below is
            # the first chain, or `+ add a chain` when no chain exists yet —
            # either way it is where an edit of this setting begins.
            self._enter_cascade()
            return
        self._begin_edit(row)

    def _enter_cascade(self) -> None:
        """Put the cursor on the first row of the cascade's own editor.

        Resolved by SCANNING FORWARD from the cascade setting row rather than
        by a fixed offset: `_cascade_rows` emits an unselectable
        ``no cascade configured`` line when the mapping is empty, and the hops
        of an already-open chain sit inside the group too. The first selectable
        row after the setting is the right target in every one of those shapes.
        """
        # This is a cursor MOVE, so it settles the row it leaves exactly as
        # `action_move` does. Setting `_selected` directly skipped `_leave_row`
        # and therefore skipped the line that clears `_notice`, so `r` then
        # `enter` — the natural sequence on this row, since `r`'s own answer
        # points at `d` on a chain row — carried `r resets one setting…` onto
        # the chain row the user had just travelled to, hiding that row's own
        # `d deletes it` contract. The notice answers a keypress on ONE row and
        # stops being true when the cursor leaves it (review round 1, B2; the
        # same "model changed, paint did not" class as the armed-delete bug
        # `action_reset` documents).
        if not self._leave_row():
            return
        for index in range(self._selected + 1, len(self._rows)):
            row = self._rows[index]
            # The group ends at the next row that belongs to something else —
            # a header, or another setting. Everything the cascade owns is one
            # of its own kinds.
            if row.kind not in ("empty", "chain", "hop", "hop_add", "chain_add"):
                break
            if row.selectable:
                self._selected = index
                self._repaint()
                self._scroll_to_selection()
                return

    def _scroll_to_expansion(self) -> None:
        """Bring the highlighted row AND the group it just opened into view.

        Scrolls only as far as it must: if the whole group already fits, the
        offset is left alone, because a list that jumped on every expansion
        would lose the reader's place for no gain.
        """
        try:
            height = self._body.size.height
        except Exception:
            return
        if height <= 0:
            return
        last = self._selected
        for index in range(self._selected + 1, len(self._rows)):
            # Choices belong to an enum row; hops (and the trailing "+ add a
            # hop") belong to a chain row. Both are the group the press just
            # opened, and both were off-screen for a row near the bottom edge.
            if self._rows[index].kind not in ("choice", "hop", "hop_add"):
                break
            last = index
        offset = self._body.scroll_offset.y
        if last >= offset + height:
            # Prefer keeping the OWNING row on screen when the group is taller
            # than the viewport: the choices are meaningless without the label
            # that says what is being chosen.
            target = min(self._selected, last - height + 1)
            self._body.scroll_to(y=max(target, 0), animate=False)

    def action_reset(self) -> None:
        """``r`` — put the highlighted row back to its shipped default.

        The mitigation for immediate-write having no undo. Deletes the stored
        key rather than writing the default over it, so a config file carries
        only what its owner actually chose (see ``settings_io.reset_setting``).
        """
        # Disarms a pending delete for the reason `action_activate` records.
        # Captured BEFORE the early returns below, because a disarm is a change
        # to what the screen is saying and every exit from here has to paint it.
        # It did not: `r` on an armed chain row cleared the flag and returned on
        # the `kind != "setting"` line, leaving the detail row still asking
        # `press d again to confirm` and the footer still offering `esc cancel`
        # for an ask that no longer existed — after which `esc` left the page
        # instead of cancelling and `d` re-armed instead of confirming. Safe in
        # the destructive direction, but the same "model changed, paint did not"
        # class as the hint clipping in D16 (UX round 4, follow-up).
        # Same interlock as `action_activate`, and for the same reason: `r`
        # writes the config, so it may not act on a row the user cannot see.
        if self._reorient():
            return
        disarmed = self._confirm_delete is not None
        self._confirm_delete = None
        row = self._current()
        if row is None or row.setting is None or row.kind != "setting":
            if disarmed:
                self._repaint()
            return
        cleared_notice = ""
        if row.setting.kind is Kind.CASCADE:
            stored = settings_io.read_setting(self._manager, row.setting)
            if isinstance(stored, Mapping):
                # A HEALTHY cascade has no shipped default to restore — the
                # chains are entirely the user's own — so `r` cannot mean here
                # what it means everywhere else. It SAYS so rather than
                # swallowing the press, because the footer advertises
                # `r default` on this row and a lit hint whose key does nothing
                # is the "nothing happens when I click" bug one step earlier
                # (UX round 1, U5).
                self._notice = "r resets one setting; delete a chain with d on its row"
                self._repaint()
                return
            # A cascade that is NOT a mapping is the wreckage of #440: every
            # user who pressed `enter` on this row before that fix has the
            # mapping's Python repr stored here as a STRING. For that value
            # "reset this setting" is both meaningful and exactly what is
            # wanted, so it falls through to `reset_setting` below, which
            # deletes the key and puts the row back to a clean empty cascade.
            # The early return above is right for the healthy case and wrong
            # for this one: it left the corrupt string in place while telling
            # the user to delete a chain with `d` on a row that does not exist
            # (`read_chains` returns `{}`, so no chain row is painted), which
            # is advice they cannot act on from the one key they would reach
            # for (UX round 1, U1). Their original hops are gone either way —
            # destroyed by the shipped bug — but the page stops lying about it.
            cleared_notice = "cleared a malformed cascade"
        if row.setting.kind is Kind.READONLY:
            # Same rule as the exit above: `_leave_row` makes an armed ask on a
            # read-only row unreachable today, but the guard is on the EXIT
            # rather than on the state, so a future arming path cannot
            # reintroduce a stale ask through this door.
            if disarmed:
                self._repaint()
            return
        setting = row.setting
        if not self._save(lambda: settings_io.reset_setting(self._manager, setting)):
            return
        self.post_message(
            SettingsChanged(setting.key, settings_io.read_setting(self._manager, setting))
        )
        # Set AFTER `_save`, which clears `_notice` on success. Empty for every
        # ordinary reset, so this adds no message where the value column
        # already shows the change landing.
        self._notice = cleared_notice
        self._repaint()

    def _save(self, action: Callable[[], None]) -> bool:
        """Run one write, holding any failure on the page. True when it landed.

        Every mutation funnels through here so the "reported, never raised"
        rule holds at all of them rather than at the ones someone remembered:
        a page that crashed on an unwritable config would take the whole TUI
        down at the exact moment the user is trying to fix their configuration.
        The cascade commits had no guard at all, which is how a broken
        config.yml under an open page could reach the app as a traceback
        (review round 2, B3).

        The unreadable-config case is separated from a schema rejection because
        the two ask the user for different things. A ``ValueError`` is "the
        value you typed is wrong", fixable in the editor; ``ConfigUnreadable``
        is "the file underneath is broken", fixable only outside the app — so
        it names the file and says nothing was written, rather than sitting in
        the editor's error slot implying the text is at fault.
        """
        try:
            action()
        except settings_io.ConfigUnreadableError as error:
            self._error = f"config.yml is unreadable, nothing was written — {error}"
            self._repaint()
            return False
        except ValueError as error:
            self._error = str(error)
            self._repaint()
            return False
        except TypeError as error:
            # A config that parses to a mapping but holds the wrong SHAPE
            # underneath — `values: not-a-mapping` is the reachable case — gets
            # this far, because the pre-parse deliberately checks only what
            # `_load_config` checks and must stay in step with it. Left to the
            # generic branch it reached the page as a raw Python message
            # ("'str' object does not support item assignment"), which names
            # nothing the user can act on (review round 3, n3). The bytes are
            # preserved either way; only the wording was unhelpful, so this
            # names the file and the remedy while keeping the original text for
            # anyone reporting it.
            self._error = (
                f"could not save — {self._config_path()} has an unexpected structure "
                f"and may need repairing ({error})"
            )
            self._repaint()
            return False
        except Exception as error:  # noqa: BLE001 — a read-only config dir, a full disk
            self._error = f"could not save: {error}"
            self._repaint()
            return False
        self._error = ""
        self._notice = ""
        return True

    def _write(self, setting: Setting, value: Any) -> None:
        """Store ``value`` and report it, or hold the reason it was refused."""
        if not self._save(lambda: settings_io.write_setting(self._manager, setting, value)):
            return
        self.post_message(SettingsChanged(setting.key, value))
        self._repaint()

    # -- text editing -------------------------------------------------------
    def _begin_edit(self, row: "_Row") -> None:
        """Open the inline editor on ``row``, seeded with its STORED value.

        Seeded from what is stored, never from what is DISPLAYED. ``_render_value``
        speaks display vocabulary — an unset value reads ``—`` and a ``None``
        reads ``auto`` — and neither was ever meant to be parsed back. Seeding
        the buffer from it made the placeholder a committable value: enter twice
        on the page's first unset row wrote ``hosting: '—'``, the provider the
        next launch boots on, and the row still read ``—`` afterwards so nothing
        on screen said a write had happened (UX round 1, U1 — the same silent-
        destructive shape as #369 itself). An unset row therefore opens EMPTY,
        which is also what ``empty_unsets`` means by "clear to unset".
        """
        if row.kind == "hop" and row.hop is not None:
            self._editing = f"hop:{row.chain}:{row.hop_index}"
            self._buffer = row.hop
        elif row.kind == "hop_add":
            self._editing = f"hopadd:{row.chain}"
            self._buffer = ""
        elif row.kind == "chain_add":
            self._editing = "chainadd"
            self._buffer = ""
        elif row.setting is not None:
            if row.setting.kind not in _TEXT_EDITABLE_KINDS:
                # Belt and braces for the #440 class of defect. Every kind that
                # is NOT edited as text has its own branch in
                # `action_activate`, so reaching here means a kind was added
                # without one — and the failure mode of that omission is not a
                # dead key but a DESTRUCTIVE one: the editor seeds itself with
                # `str(stored_value)` and commits whatever that renders as, in
                # `Kind.CASCADE`'s case a Python repr written over the user's
                # own mapping. Refusing to open is the safe half of that pair,
                # and it says so rather than swallowing the press.
                #
                # Guarded on the KIND and not on the value's shape because
                # `Kind.LIST` legitimately stores a list and legitimately edits
                # as comma-separated text; "is this a scalar" would have to
                # carve that out and would still miss a scalar-valued kind that
                # needs its own widget.
                self._error = "this setting is not edited as text"
                self._repaint()
                return
            self._editing = row.setting.key
            self._buffer = _edit_seed(settings_io.read_setting(self._manager, row.setting))
        else:
            return
        self._edit_seed = self._buffer
        self._caret = len(self._buffer)
        self._error = ""
        self._repaint()

    def _cancel_edit(self) -> None:
        self._editing = None
        self._buffer = ""
        self._edit_seed = ""
        self._caret = 0
        self._error = ""

    def _leave_row(self) -> bool:
        """Settle whatever the current row has open, and say whether to move.

        Movement used to call ``_cancel_edit`` unconditionally, so a plain
        ``down`` after typing a valid value threw it away silently — the footer
        promises ``enter saves · esc cancels`` and says nothing about arrows, so
        the user's model is "esc is how I lose this" (UX round 1, U3). The rule
        here is the one the page can state honestly: a VALID buffer commits on
        the way out, and an INVALID one holds the cursor where it is with the
        rejected text and the reason both still on screen — the same contract
        Enter already has, so there is only one rule to learn.

        Returns ``False`` when the caller must NOT move.
        """
        # A pending chain-delete confirmation is answered by `d` or by `esc`,
        # never by drifting off the row: an ask that survived the cursor moving
        # away would fire on a row the user is no longer looking at.
        self._confirm_delete = None
        # The notice answered a keypress on THIS row, so it stops being true
        # the moment the cursor leaves it — the same lifetime `_error` has.
        self._notice = ""
        if self._editing is None:
            self._error = ""
            return True
        if self._buffer == self._edit_seed:
            # Nothing was typed. Closing beats committing an identical value:
            # a write would stamp a defaulted key into the file just because the
            # cursor passed through the row.
            self._cancel_edit()
            return True
        self._commit_edit()
        return self._editing is None

    def _caret_left(self) -> None:
        self._caret = max(0, self._caret - 1)
        self._repaint()

    def _caret_right(self) -> None:
        self._caret = min(len(self._buffer), self._caret + 1)
        self._repaint()

    def _commit_edit(self) -> None:
        """Enter inside the editor: validate, then save or keep it open.

        A REJECTED value keeps the editor open with the text still in it. The
        alternative — close and report — throws away what the user typed at the
        exact moment they need to correct one character of it, and is the
        behaviour that makes a validating form feel hostile.
        """
        target = self._editing
        if target is None:
            return
        if target.startswith("hop:") or target.startswith("hopadd:"):
            self._commit_hop(target)
            return
        if target == "chainadd":
            self._commit_chain_add()
            return
        setting = settings_io.resolve_key(target)
        if setting is None:
            self._cancel_edit()
            return
        if self._buffer == self._edit_seed:
            # Nothing was typed, so nothing is saved — not even a rewrite of the
            # value already there. Enter on an untouched editor must leave the
            # file BYTE-identical: routing it through the normal save path had
            # an unset row's empty buffer call `reset_setting`, which rewrote
            # config.yml (and its `last_modified`) to delete a key that was
            # never present. The user pressed enter twice on a row and their
            # config file changed (UX round 1, U1).
            self._cancel_edit()
            self._repaint()
            return
        text = self._buffer.strip()
        if not text and setting.empty_unsets:
            # An empty field UNSETS rather than storing "": for `hosting` and
            # the subagent tiers, "" and absent mean the same thing to their
            # consumers, and storing the empty string leaves a key in the file
            # that reads as a deliberate choice of nothing.
            if not self._save(lambda: settings_io.reset_setting(self._manager, setting)):
                return
            self._cancel_edit()
            self.post_message(SettingsChanged(setting.key, setting.default))
            self._repaint()
            return
        try:
            value = settings_io.coerce(setting, text)
        except ValueError as error:
            self._error = str(error)
            self._repaint()
            return
        problem = settings_io.validate(setting, value)
        if problem is not None:
            self._error = problem
            self._repaint()
            return
        self._write(setting, value)
        if not self._error:
            self._cancel_edit()
            self._repaint()

    def _chain_edit(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """A working copy of the cascade and the BASE snapshot it was read from.

        Two copies rather than one because ``write_chains`` needs to know which
        chains this edit actually touched: it merges the caller's own change
        over what is on disk NOW, and without the base it can only replace the
        cascade wholesale — deleting a chain another session added and dropping
        the effort from a hop another session edited (review round 2, M2).
        The base is a separate deep copy so mutating the working copy cannot
        change what it is compared against.
        """
        read = settings_io.read_chains(self._manager)
        working = {key: list(hops) for key, hops in read.items()}
        base = {key: list(hops) for key, hops in read.items()}
        return working, base

    def _commit_hop(self, target: str) -> None:
        """Save an edited or newly-added cascade hop."""
        problem = settings_io.validate_hop(self._buffer)
        if problem is not None:
            self._error = problem
            self._repaint()
            return
        chains, base = self._chain_edit()
        _, chain_key, *rest = target.split(":")
        hops = chains.setdefault(chain_key, [])
        if target.startswith("hopadd:"):
            hops.append(self._buffer.strip())
        else:
            index = int(rest[0])
            if 0 <= index < len(hops):
                hops[index] = self._buffer.strip()
        if not self._save(lambda: settings_io.write_chains(self._manager, chains, base=base)):
            return
        self._cancel_edit()
        self.post_message(SettingsChanged("retry.fallbackChains", chains))
        self._repaint()

    def _commit_chain_add(self) -> None:
        """Create a new, empty cascade chain under the typed key.

        The chain is created with ONE hop rather than empty, because
        ``write_chains`` drops empty chains (the failover layer does too), so an
        empty new chain would vanish the moment it was saved and read as the
        add having failed. The first hop is what the user typed after the key.
        """
        text = self._buffer.strip()
        key, _, first_hop = text.partition(" ")
        if not key:
            self._error = "expected: <chain-key> <provider>/<model>"
            self._repaint()
            return
        problem = settings_io.validate_hop(first_hop)
        if problem is not None:
            self._error = f"chain needs a first hop — {problem}"
            self._repaint()
            return
        chains, base = self._chain_edit()
        chains[key] = [first_hop.strip()]
        if not self._save(lambda: settings_io.write_chains(self._manager, chains, base=base)):
            return
        self._chain = key
        self._cancel_edit()
        self.post_message(SettingsChanged("retry.fallbackChains", chains))
        self._repaint()

    def _config_path(self) -> str:
        """The config file's path, home-relative, for messages that name it.

        Shared by the title and the error slots so a user reading "this file has
        an unexpected structure" sees the same string the page already shows
        them at the top, rather than two spellings of one path.
        """
        return _home_relative(str(getattr(self._manager, "config_file", ""))) or "config.yml"

    def _select_setting_row(self, key: str) -> None:
        """Put the cursor on the SETTING row owning ``key``, if it is present.

        The counterpart of :meth:`_close_chain` for an enum expansion: choice
        rows stop existing when the expansion collapses, so the index they
        occupied names whatever the rebuild put there. Both exits from an
        expansion — picking a choice and abandoning with ``esc`` — route through
        here, because two copies of this search is how the abandon kept the
        drift after the pick was fixed (UX round 2 U17, round 3 U20).

        Call it AFTER the repaint that rebuilds ``_rows``: it resolves against
        the current row list, not the one the expansion was open in.
        """
        for index, row in enumerate(self._rows):
            if row.kind == "setting" and row.setting is not None and row.setting.key == key:
                self._selected = index
                return

    def _close_chain(self) -> None:
        """Close the open chain and put the cursor back on its own row.

        The cursor has to be MOVED, not merely left where it was: the hop rows
        it may be sitting on stop existing when the chain closes, so an index
        left alone would land on whatever the rebuild put in their place.
        """
        chain = self._chain
        self._chain = None
        self._repaint()
        if chain is None:
            return
        for index, row in enumerate(self._rows):
            if row.kind == "chain" and row.chain == chain:
                self._selected = index
                break
        self._repaint()
        self._scroll_to_selection()

    def _delete_hop(self) -> None:
        """Remove the highlighted hop, or the whole chain from its key row.

        A HOP deletes outright: it is one line, and retyping it is cheap. A
        CHAIN asks first. Deleting a chain destroys every hop in it, `r` cannot
        bring it back (there is no shipped default to restore a user's own
        failover configuration from), and `d` sits one row from `enter`, which
        on the same row means "open" — so a user exploring the two keys the
        detail line names could destroy a multi-hop failover config by pressing
        the second one, with nothing on screen saying it happened (UX round 1,
        U5). The confirmation lives in the detail line rather than in a modal
        because the page is a mode for the reason the module docstring gives.
        """
        row = self._current()
        if row is None:
            return
        if row.kind == "chain" and row.chain is not None and self._confirm_delete != row.chain:
            self._confirm_delete = row.chain
            self._repaint()
            return
        chains, base = self._chain_edit()
        if row.kind == "hop" and row.chain in chains:
            hops = chains[row.chain]
            if 0 <= row.hop_index < len(hops):
                del hops[row.hop_index]
        elif row.kind == "chain" and row.chain in chains:
            del chains[row.chain]
            self._chain = None
        else:
            self._confirm_delete = None
            return
        self._confirm_delete = None
        if not self._save(lambda: settings_io.write_chains(self._manager, chains, base=base)):
            return
        self.post_message(SettingsChanged("retry.fallbackChains", chains))
        self._repaint()

    def _move_hop(self, delta: int) -> None:
        """Reorder the highlighted hop within its chain.

        Order IS the setting here — a cascade is tried top to bottom — so
        reordering has to be a first-class action rather than "delete and
        retype in the right place".
        """
        row = self._current()
        if row is None or row.kind != "hop" or row.chain is None:
            return
        chains, base = self._chain_edit()
        hops = chains.get(row.chain, [])
        index = row.hop_index
        target = index + delta
        if not (0 <= index < len(hops)) or not (0 <= target < len(hops)):
            return
        hops[index], hops[target] = hops[target], hops[index]
        if not self._save(lambda: settings_io.write_chains(self._manager, chains, base=base)):
            return
        # Follow the moved hop with the cursor: a reorder that left the
        # highlight on the row number rather than on the row's content makes a
        # second press move a different hop than the user is looking at.
        self._selected += delta
        self.post_message(SettingsChanged("retry.fallbackChains", chains))
        self._repaint()

    # -- keys ---------------------------------------------------------------
    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Route keys the bindings do not own: the editor's, and the filter's.

        Handled here rather than as bindings because an OPEN EDITOR must own
        every printable key — a ``d`` typed into a model id cannot also be the
        delete-hop shortcut. The editor is checked first for exactly that
        reason, and the Esc LADDER (editor → chain → expansion → page, one press
        each) falls out of the same ordering.

        The editor owns its NAVIGATION keys for the same reason it owns the
        printable ones. Left/right/home/end were the keys a user reaches for
        mid-edit and the ones that leaked: unhandled they fell through to the
        page bindings, switched the read-only side pane while the user was
        typing, and — for ``home`` — discarded the edit and jumped to row 0
        (UX round 1, U2/U3).
        """
        key = event.key
        if self._editing is not None:
            if key == "escape":
                event.stop()
                event.prevent_default()
                self._cancel_edit()
                self._repaint()
                return
            if key == "enter":
                event.stop()
                event.prevent_default()
                self._commit_edit()
                return
            if key == "backspace":
                event.stop()
                event.prevent_default()
                # Deletes at the CARET, not off the tail: the caret exists so a
                # typo in the middle of a long endpoint is fixable in place.
                if self._caret > 0:
                    self._buffer = self._buffer[: self._caret - 1] + self._buffer[self._caret :]
                    self._caret -= 1
                self._repaint()
                return
            if key == "delete":
                event.stop()
                event.prevent_default()
                self._buffer = self._buffer[: self._caret] + self._buffer[self._caret + 1 :]
                self._repaint()
                return
            if key in ("left", "right", "home", "end"):
                event.stop()
                event.prevent_default()
                if key == "left":
                    self._caret_left()
                elif key == "right":
                    self._caret_right()
                else:
                    # HOME/END move within the BUFFER, not the page. `home`
                    # previously ran `action_jump(0)`, which cancelled the edit
                    # and left the user 25 rows from where they were.
                    self._caret = 0 if key == "home" else len(self._buffer)
                    self._repaint()
                return
            char = getattr(event, "character", None)
            if char and char.isprintable():
                event.stop()
                event.prevent_default()
                self._buffer = self._buffer[: self._caret] + char + self._buffer[self._caret :]
                self._caret += 1
                self._repaint()
            return
        if key == "escape" and self._confirm_delete is not None:
            # An unanswered "delete this chain?" is backed out of before any
            # other Esc meaning applies, so the key that cancels the ask is the
            # key the ask itself advertises.
            event.stop()
            event.prevent_default()
            self._confirm_delete = None
            self._repaint()
            return
        if key == "escape" and self._chain is not None:
            # The rung the ladder was missing (UX round 1, U4). The enum
            # expansion below DOES consume Esc, so a cascade — the only
            # multi-level editor on the page, and the one place a user is two
            # levels down — dumping them out of the whole page taught a rule and
            # then broke it. Closing leaves the cursor on the chain's own row.
            event.stop()
            event.prevent_default()
            self._close_chain()
            return
        if key == "escape" and self._expanded is not None:
            # Rung three of the ladder: Esc closes the EXPANSION before it
            # closes the page, so a user who opened a dropdown to look at it can
            # back out of it without losing the whole surface.
            #
            # The cursor returns to the SETTING row, exactly as `action_activate`
            # does when a choice is picked. U17 fixed the pick and left the
            # abandon, so backing out of `tui.theme` from its 34th member landed
            # the cursor 34 rows away with `r default` lit on an unrelated
            # setting two sections down (UX round 3, U20) — and backing out is
            # the MORE conservative gesture of the two, so it must not be the
            # one that moves you. `_close_chain` is the model this follows.
            event.stop()
            event.prevent_default()
            owner = self._expanded
            self._expanded = None
            self._repaint()
            self._select_setting_row(owner)
            self._repaint()
            self._scroll_to_selection()
            return
        row = self._current()
        if row is not None and row.kind in ("hop", "chain"):
            if key in ("d", "delete"):
                event.stop()
                event.prevent_default()
                # The interlock `action_activate` and `action_reset` carry, on
                # the page's DESTRUCTIVE key: `d` deletes a hop outright, so an
                # off-screen cursor must cost a reorientation press rather than
                # a chain the user never looked at.
                if self._reorient():
                    return
                self._delete_hop()
                return
            if key in ("shift+up", "shift+down") and row.kind == "hop":
                event.stop()
                event.prevent_default()
                if self._reorient():
                    return
                self._move_hop(-1 if key == "shift+up" else 1)
                return

    # -- mouse --------------------------------------------------------------
    # Every handler stops its event so one gesture does not also move the page
    # beneath, the discipline every mouse handler in this app follows.
    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Primary click SELECTS the row, and activates it if already selected.

        Select-then-activate rather than activate-on-first-click, cf.
        ``session_picker``: these rows write config, and a single stray click
        landing on "tool approval mode" should move the cursor there so the
        user can see what they are about to change, not change it.

        Button 1 only: a right-click asking for a context menu and a
        middle-click paste must not be able to write a setting.

        Deliberately does NOT scroll. A click names a row the user is already
        looking at, so moving the cursor there can never put it off screen —
        and a view that re-centred on every click would jump under the pointer
        for no gain. This is also why the click path needs no ``_reorient``
        guard: it cannot select an invisible row, and the activate-on-second-
        click branch below therefore always acts on a visible one.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        row = self._rows[index]
        if not row.selectable:
            return
        if index == self._selected:
            self.action_activate()
            return
        # WHICH row was clicked is resolved before the commit and WHERE it
        # ended up after it. A click carries an explicit target — the label the
        # user could see under the pointer — so it is the one gesture that must
        # not be re-derived from the y afterwards: `_leave_row` commits, a
        # commit on an add row inserts rows above the click, and both the stale
        # index and a re-read of the same y then name a row the user never
        # aimed at. Clicking `Theme` landed on a hop three rows away (review
        # round 2, U13). The identity travels with the row across the rebuild.
        wanted = row.identity
        if not self._leave_row():
            return
        settled = next(
            (index for index, candidate in enumerate(self._rows) if candidate.identity == wanted),
            None,
        )
        if settled is None or not self._rows[settled].selectable:
            return
        self._selected = settled
        self._repaint()

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
        # The hand pointer over an actionable row only; the headers and the
        # page's padding keep the default shape, so the pointer itself says
        # which rows do something.
        actionable = index is not None and self._rows[index].selectable
        self.styles.pointer = "pointer" if actionable else "default"

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
        self.styles.pointer = "default"

    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._scroll_rows(1)

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._scroll_rows(-1)

    def _scroll_rows(self, delta: int) -> None:
        """Wheel movement — the VIEWPORT moves and the cursor stays put.

        ONE SOURCE OF TRUTH FOR THE VIEWPORT
        ------------------------------------
        This method used to move the CURSOR by a row and then call
        ``_scroll_to_selection``, which re-derived the viewport from the cursor.
        The body is a ``ScrollableContainer``, so the container ALSO owns a
        scroll offset that the wheel and the scrollbar drive directly — two
        positions for one view, each able to overwrite the other. The user-
        visible result was a bounce, reported against v0.43.10 and reproduced at
        100x30 (viewport 14, virtual 60, ``max_scroll_y=46``):

            notch 23: scroll_y=46/46  selected=1   <- wheeled to the bottom
            notch 24: scroll_y= 2/46  selected=2   <- BOUNCE, back to the top

        The path is worth stating because it is NOT the obvious one. Textual
        handles the wheel on the container first (``_scroll_down_for_pointer``)
        and stops the event while it still has somewhere to go, so over the list
        this method never ran at all. It ran only once the container hit its
        limit and let the event bubble — at which point the cursor, still parked
        on row 1 from before the gesture, moved one row and dragged the viewport
        46 rows back with it. Wheeling to the bottom of a list is precisely how
        a user arrives at that state, which is why the bug reads as the page
        snapping back the moment you reach the end.

        It also made the gesture POSITION-DEPENDENT: the same notch scrolled the
        viewport over the list, and moved the cursor over the title, the detail
        row or the side pane, because only the body region consumed the event.
        One page, two scroll models, chosen by where the pointer happened to be.

        So the wheel is now the container's gesture everywhere: it moves the
        viewport, the cursor stays on the row the user put it on, and the cursor
        is allowed to go off screen. That is the model list UIs and editors use,
        it is the only model under which the scrollbar thumb tells the truth,
        and it is what makes "scroll to the bottom" stay at the bottom. The
        cursor keeps its own claim on the viewport for the KEYS that move it
        (:meth:`_scroll_to_selection`), and the two claims can no longer fight
        because only one of them is driven by any given gesture.

        Clamping is unchanged and now comes from the container, which cannot
        scroll past either end — so the wheel still never teleports to the other
        end of the list (AGENTS.md, "Wrapping vs clamping").

        No commit semantics are involved: this moves the view, never the cursor,
        so it neither writes a value nor settles an open editor.
        """
        try:
            # Accumulated from `scroll_target_y`, NOT from `scroll_relative`.
            # `scroll_relative` adds to the CURRENT offset, so several notches
            # arriving before the next refresh each compute from the same stale
            # position and collapse into one — 40 notches moved the viewport two
            # rows. `scroll_target_y` is where the container is already headed,
            # which is what Textual's own `_scroll_down_for_pointer` adds to, so
            # a fast flick travels the same distance here as it does over the
            # list. Textual clamps the target to `[0, max_scroll_y]`.
            #
            # `immediate=True` for the same reason: the default defers the move
            # to `call_after_refresh`, so `scroll_target_y` still reads the old
            # position for every notch that arrives before that refresh and the
            # accumulation above collapses anyway. Textual's own wheel handler
            # reaches `_scroll_to` synchronously; this is that path, spelled
            # through the public API.
            self._body.scroll_to(
                y=self._body.scroll_target_y + delta * _WHEEL_ROWS,
                animate=False,
                immediate=True,
            )
        except Exception:
            # The container has no size before layout; a wheel notch that
            # arrives that early has no viewport to move and is dropped rather
            # than falling back to moving the cursor, which is the behaviour
            # this method exists to stop.
            return

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """Row index under a mouse event, or ``None`` outside the list.

        Measured against the BODY's region and offset by its scroll, because
        the list is one painted Static: a click anywhere in it reports a y
        relative to the whole block, and the container may be scrolled. Guarded
        the way ``session_picker._index_at`` is — a false positive here writes
        a setting.
        """
        try:
            region = self._body.region
        except Exception:
            return None
        x, y = event.screen_x, event.screen_y
        if not region.contains(x, y):
            return None
        index = (y - region.y) + self._body.scroll_offset.y
        if 0 <= index < len(self._rows):
            return index
        return None

    # -- painting -----------------------------------------------------------
    def _repaint(self) -> None:
        self._rows = self._build_rows()
        indices = self._selectable()
        if indices and self._selected not in indices:
            # A rebuild can drop the row the cursor was on (an expansion
            # closing, a chain deleted). Land on the nearest surviving row
            # rather than resetting to the top, which would lose the user's
            # place for a change they made three sections down.
            self._selected = min(indices, key=lambda index: abs(index - self._selected))
        self._paint_list()
        self._paint_pane()
        self._paint_detail()
        self._paint_chrome()

    def _paint_list(self) -> None:
        width = self._list_width()
        text = Text(no_wrap=True, overflow="ellipsis")
        for index, row in enumerate(self._rows):
            if index:
                text.append("\n")
            line = self._row_text(row, index, width)
            # Clipped rather than allowed to wrap. A wrapped row breaks the
            # one-row-per-setting contract the cursor and the click handler
            # both depend on: `_index_at` maps a click's y to a row index, so a
            # row occupying two lines would make every click below it land on
            # the wrong setting.
            line.truncate(width, overflow="ellipsis")
            text.append_text(line)
        self._list_text = text
        self._list.update(text)
        # Pin the Static to the row count so the container's virtual size
        # equals the list and Textual scrolls the difference. `auto` collapses
        # it to the handed content with no room to scroll.
        self._list.styles.height = max(len(self._rows), 1)

    def _row_text(self, row: "_Row", index: int, width: int) -> Text:
        selected = index == self._selected
        hovered = index == self._hovered
        line = Text(no_wrap=True, overflow="ellipsis")
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        accent = Style(color=theme_mod.semantic_color("accent"))
        faint = Style(color=theme_mod.semantic_color("faint"))

        if row.kind == "header" and row.section is not None:
            # The scope tag rides the SECTION header, right-aligned and dim: it
            # answers "when does this take effect" once per group rather than
            # fifty times down the page. See the module docstring.
            head = Text(no_wrap=True)
            head.append(
                row.section.title, style=Style(color=theme_mod.semantic_color("fg"), bold=True)
            )
            # The tag sheds its PREFIX before it sheds the scope itself: on a
            # narrow body "takes effect: new sessions" does not fit beside the
            # title, and the half that carries the meaning is the scope. Dropped
            # entirely only when even the bare scope will not fit, which is the
            # one case where a clipped tag would be worse than none.
            full = f"takes effect: {row.section.scope.value}"
            short = row.section.scope.value
            # LEFT-aligned on one shared column, not right-aligned against each
            # title. Right-aligning made the tag's position depend on the
            # title's length, so `Model` and `Failover and retry` — two headers
            # of the same rank — put their tags two cells apart and the eye read
            # the difference as an accident rather than as a grid (design round
            # 1, D3). The column is the value column the settings rows already
            # align on, so the tag lands on structure the page already has.
            room = width - _VALUE_COLUMN
            tag = full if cell_len(full) <= room else (short if cell_len(short) <= room else "")
            if tag and cell_len(head.plain) < _VALUE_COLUMN:
                head.append(" " * (_VALUE_COLUMN - cell_len(head.plain)))
                head.append(tag, style=dim)
            return head

        if row.kind == "empty":
            line.append(" " * _CHOICE_INDENT)
            line.append(row.text, style=dim)
            return line

        marker = f"{_CURSOR} " if selected else "  "
        base = fg if selected else (muted if hovered else muted)

        if row.kind == "choice" and row.choice is not None and row.setting is not None:
            current = settings_io.read_setting(self._manager, row.setting)
            chosen = current == row.choice.value
            line.append(" " * (_CHOICE_INDENT - 2))
            line.append(marker, style=accent if selected else dim)
            # The chosen member is marked rather than merely coloured: a colour
            # difference alone does not survive a monochrome terminal, and
            # "which one am I on" is the question the expansion exists to answer.
            line.append("● " if chosen else "○ ", style=accent if chosen else dim)
            line.append(row.choice.label, style=base)
            if row.choice.description:
                line.append(f"  {row.choice.description}", style=faint)
            return line

        if row.kind == "chain" and row.chain is not None:
            hops = settings_io.read_chains(self._manager).get(row.chain, [])
            line.append(" " * (_ROW_INDENT))
            line.append(marker, style=accent if selected else dim)
            line.append("▾ " if self._chain == row.chain else "▸ ", style=dim)
            line.append(row.chain, style=base)
            line.append(f"  {len(hops)} hop{'' if len(hops) == 1 else 's'}", style=faint)
            return line

        if row.kind == "hop":
            editing = self._editing == f"hop:{row.chain}:{row.hop_index}"
            line.append(" " * _CHOICE_INDENT)
            line.append(marker, style=accent if selected else dim)
            line.append(f"{row.hop_index + 1}. ", style=dim)
            if editing:
                line.append_text(self._editor_text())
            else:
                line.append(row.hop, style=base)
                if selected:
                    line.append("  shift+↑↓ reorder · d delete", style=faint)
            return line

        if row.kind in ("hop_add", "chain_add"):
            target = f"hopadd:{row.chain}" if row.kind == "hop_add" else "chainadd"
            editing = self._editing == target
            indent = _CHOICE_INDENT if row.kind == "hop_add" else _ROW_INDENT
            line.append(" " * indent)
            line.append(marker, style=accent if selected else dim)
            if editing:
                line.append_text(self._editor_text())
            else:
                label = "+ add a hop" if row.kind == "hop_add" else "+ add a chain"
                line.append(label, style=base if selected else dim)
            return line

        setting = row.setting
        if setting is None:
            return line
        line.append(" " * _ROW_INDENT)
        line.append(marker, style=accent if selected else dim)
        label = truncate_cells(setting.label, _LABEL_BUDGET)
        line.append(label, style=base)
        pad = max(1, _VALUE_COLUMN - cell_len(line.plain))
        line.append(" " * pad)
        if self._editing == setting.key:
            line.append_text(self._editor_text())
            return line
        value = settings_io.read_setting(self._manager, setting)
        changed = value != setting.default
        # A CHANGED value is painted in the foreground ink and a defaulted one
        # dim, so the config's actual shape is visible at a glance — which is
        # the state a user needs before they can decide what to reset.
        value_style = fg if changed else dim
        if setting.kind is Kind.READONLY:
            value_style = dim
        if setting.kind is Kind.CASCADE and self._cascade_is_malformed(setting):
            # NOT `_render_value`, which would fall through to `str(value)` and
            # paint the corrupt Python repr as if it were the setting's value —
            # directly above a group line saying there is no cascade. The frame
            # then stated two contradictory things at once and a user could not
            # tell from it whether their chains existed (UX round 1, U1). `—`
            # is the page's existing glyph for "nothing to show here", and the
            # line below carries the explanation.
            line.append("—", style=dim)
            return line
        line.append(_render_value(value), style=value_style)
        if setting.kind is Kind.ENUM and self._expanded == setting.key:
            line.append(" ▾", style=dim)
        return line

    def _editor_text(self) -> Text:
        """The inline editor: the buffer, a caret, its contract, and any error.

        The error is rendered ON the editor row and the editor STAYS OPEN, so
        the rejected text and the reason it was rejected are on screen at the
        same time — the state a user needs to fix one character.
        """
        editor = Text(no_wrap=True)
        accent = Style(color=theme_mod.semantic_color("accent"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        fg = Style(color=theme_mod.semantic_color("fg"))
        # The caret is painted IN the buffer at its index rather than always at
        # the tail, so the frame shows where the next character will land —
        # without that, left/right moved a position nothing on screen reported
        # and the keys read as dead.
        caret = max(0, min(self._caret, len(self._buffer)))
        before, after, elided = self._editor_window(caret)
        if elided:
            # A leading ellipsis says the buffer continues to the LEFT, so a
            # window that has scrolled does not read as the value having been
            # truncated at the front.
            editor.append("…", style=faint)
        editor.append(before, style=fg)
        editor.append("▏", style=accent)
        editor.append(after, style=fg)
        # The CONTRACT rides the row; the ERROR does not. The detail line below
        # already carries the rejection in full width, and printing it twice
        # read as two separate problems — the row's copy also had to compete
        # with the value column for space it does not have.
        #
        # `↑↓` is named alongside `enter` because moving off the row SAVES, and
        # a contract enumerating exactly two exits with `esc cancels` beside
        # them implies every other key is one or the other. That is the
        # opposite of most editors, so it has to be taught rather than
        # discovered by losing a value to it (UX round 2, U14). The clause
        # sheds first when the row is narrow: it is the least load-bearing of
        # the three, and the other two are the keys a user needs to get out.
        contract = "  enter or ↑↓ saves · esc cancels · clear to unset"
        room = self._list_width() - _VALUE_COLUMN - cell_len(editor.plain)
        if cell_len(contract) > room:
            contract = "  enter or ↑↓ saves · esc cancels"
        editor.append(contract, style=faint)
        return editor

    def _editor_window(self, caret: int) -> tuple[str, str, bool]:
        """The slice of the buffer to paint around ``caret``.

        Returns ``(before_caret, after_caret, scrolled)``.

        The buffer is painted as a WINDOW rather than from index 0 because the
        caret is the entire feedback for four keys — left, right, home, end —
        and past roughly 26 cells at 100 columns it fell off the right edge
        with the last characters typed. Between there and dragging it back the
        keys moved a position nothing on screen reported, which is exactly the
        failure the caret was added to prevent (UX round 2, U15).

        Anchored on the caret rather than on the tail so the same window serves
        typing at the end and editing in the middle. The window is only as wide
        as the value column's remaining room, which is why it is derived from
        `_list_width` here rather than assumed.
        """
        # Room for the value, minus three cells that are not the buffer's to
        # spend: the caret glyph, the leading ellipsis a scrolled window shows,
        # and one for the row's OWN overflow ellipsis. That last one is the
        # subtle one — the contract text after the caret always overruns the
        # row, so Rich truncates, and a window sized to exactly the remaining
        # room had its final cell (the caret) replaced by that ellipsis. The
        # caret was invisible for the same reason it was before, one cell later.
        room = max(self._list_width() - _VALUE_COLUMN - 3, 8)
        if len(self._buffer) <= room:
            return (self._buffer[:caret], self._buffer[caret:], False)
        # Keep the caret inside the window with a little context behind it, so
        # a backspace at the window's left edge does not jump the whole view.
        start = max(0, caret - room + 4)
        start = min(start, max(0, len(self._buffer) - room))
        return (self._buffer[start:caret], self._buffer[caret:], start > 0)

    #: Leads the delete confirmation so an ASK is distinguishable from a
    #: REPORT. Both occupy the detail row in the same danger ink, and a
    #: question holding a destructive action that looks exactly like a
    #: validation error gets read as "something was already rejected" and
    #: dismissed — leaving the chain undeleted and the user believing it went
    #: through (design round 2, D7). A marker plus a bold question clause is
    #: the cheapest separation that stays inside the page's existing ink.
    _CONFIRM_MARKER = "▸ "

    def _confirm_parts(self) -> tuple[str, str]:
        """The pending deletion as ``(question, key contract)``, or two "".

        Split in two because the halves shed differently. The question names
        the chain AND what it costs — "Are you sure?" would make the user
        answer about a row they have to look away to identify — while the
        contract states how to answer and how to back out. At 80 columns the
        whole line ran past the frame with the ellipsis landing OUTSIDE it, so
        `to confirm · esc cancels` disappeared with no visible mark that
        anything had been cut, on the page's only destructive prompt (design
        round 2, D8). The CHAIN NAME is what gives way instead: clipping the
        least load-bearing segment is the rule ``_paint_pane`` already follows,
        and inverted here it says keep the keys and clip the name.
        """
        chain = self._confirm_delete
        if chain is None:
            return ("", "")
        hops = settings_io.read_chains(self._manager).get(chain, [])
        count = f"{len(hops)} hop{'' if len(hops) == 1 else 's'}"
        # The row this text is PAINTED into, not the settings list beside it.
        # `_list_width()` subtracts the pane, which the full-width detail row
        # does not lose — budgeting against it clipped the chain name with a
        # third of the row empty (design round 3, D12).
        width = self._detail_width()

        #: The key contract, longest first. It sheds to a terser form BEFORE
        #: the chain name is clipped below readability, because a name cut to
        #: three cells names nothing while `d confirms · esc cancels` still
        #: carries the whole answer. Both rungs name the confirming key AND the
        #: cancelling key: that pair is what must never be lost, since losing
        #: it is the defect (design round 2, D8).
        ladder = ("press d again to confirm · esc cancels", "d confirms · esc cancels")
        #: Cells the name may take before the line overruns the row. The
        #: two-space gap is the one `_paint_detail` paints between the question
        #: and the contract, so the budget matches the frame.
        for contract in ladder:
            fixed = cell_len(f"{self._CONFIRM_MARKER}delete chain  and its {count}?  {contract}")
            room = width - fixed
            if room >= cell_len(chain) or contract is ladder[-1]:
                return (
                    f"delete chain {truncate_cells(chain, max(room, 8))} and its {count}?",
                    contract,
                )
        raise AssertionError("the ladder's last rung always returns")

    def _confirm_text(self) -> str:
        """The pending ask as one plain string. Kept for the assertions."""
        question, contract = self._confirm_parts()
        return f"{question} · {contract}" if question else ""

    def _paint_detail(self) -> None:
        """The one-line explanation of the HIGHLIGHTED row.

        A dedicated row rather than a gloss appended to the row itself. Inline
        it had to either wrap — which breaks the one-row-per-setting contract
        `_index_at` maps clicks through — or be clipped mid-sentence, and half
        a help string is not help. Here it has the full width, and it changes
        as the cursor moves, so the page explains itself continuously instead
        of making the user go looking for documentation.

        The row is ALWAYS present, even when empty. A detail line that appeared
        and disappeared would move the footer and the whole body with it on
        every cursor move — the "rows are load-bearing" rule in AGENTS.md.
        """
        faint = Style(color=theme_mod.semantic_color("faint"))
        error = Style(color=theme_mod.semantic_color("danger"))
        text = Text(no_wrap=True, overflow="ellipsis")
        row = self._current()
        question, contract = self._confirm_parts()
        if question:
            # Above the error and above the help: an unanswered destructive
            # question is the only thing the user needs the row to say.
            #
            # Shaped so an ASK does not read as a REPORT (design round 2, D7).
            # The marker and the BOLD question clause are what a validation
            # error never has, and the key contract stays unbolded so the row
            # separates "what I am about to destroy" from "how to answer".
            text.append(self._CONFIRM_MARKER, style=error + Style(bold=True))
            text.append(question, style=error + Style(bold=True))
            text.append(f"  {contract}", style=error)
        elif self._notice:
            # An informational message is NOT an error. It used to route
            # through `self._error` and inherit the danger ink, so telling a
            # user what a key they pressed does here looked like a report that
            # they had done something wrong (UX round 2, U16).
            text.append(self._notice, style=faint)
        elif self._error:
            text.append(self._error, style=error)
        elif row is None:
            pass
        elif row.kind == "hop":
            text.append("shift+↑↓ reorder · d delete · enter edits this hop", style=faint)
        elif row.kind == "chain":
            text.append("enter opens the chain · d deletes it", style=faint)
        elif row.kind == "chain_add":
            text.append("enter, then type: <chain-key> <provider>/<model>", style=faint)
        elif row.kind == "hop_add":
            text.append("enter, then type: <provider>/<model>", style=faint)
        elif row.setting is not None:
            text.append(row.setting.help, style=faint)
            text.append(f"   {row.setting.key}", style=Style(color=theme_mod.semantic_color("dim")))
        self._detail_text = text
        self._detail.update(text)

    #: Narrowest page that still shows BOTH columns. Below it the pane is
    #: hidden and the settings list takes the whole body: the pane is context,
    #: the list is the subject, and two columns in 70 cells leaves neither one
    #: readable. Measured from the pane's own 34 cells plus a settings list
    #: wide enough for a label and its value (`_VALUE_COLUMN` + room).
    _TWO_COLUMN_MIN_WIDTH = 92

    def _pane_width(self) -> int:
        """Usable cells in the read-only pane, minus its left padding.

        Derived from :data:`_PANE_WIDTH` and NOT from ``self._pane_view.size``.
        That is the fix for a real defect: ``_repaint`` runs from ``on_mount``,
        which is BEFORE Textual has laid the children out, so every child size
        is still ``0`` at that moment. Truncating against a zero width clipped
        the pane to "si…"/"manager…" and wrapped the section headers, and
        because nothing repainted after layout the first frame stayed wrong.
        The view's OWN width is known in ``on_resize``, so every width here is
        computed from it.
        """
        return max(_PANE_WIDTH - 3, 12)

    def _pane_fits(self) -> bool:
        """Whether this terminal is wide enough to carry the second column."""
        try:
            return self.size.width >= self._TWO_COLUMN_MIN_WIDTH
        except Exception:
            return True

    def _detail_width(self) -> int:
        """Cells the detail ROW may occupy, which is not the list's width.

        The detail line is a full-width ``Static`` spanning the whole view
        (``compose`` yields it OUTSIDE ``_columns``), so it keeps its cells
        whether or not the read-only pane is displayed. Budgeting it with
        :meth:`_list_width` — which subtracts ``_PANE_WIDTH`` — clipped the
        delete confirmation to 60 cells inside a 96-cell row at 100 columns,
        truncating `openrouter-budget-fallback` to `openrou…` while the chain
        row directly above it showed the same name in full, with 29 cells of the
        row empty (design round 3, D12). The 80-column case hid it because the
        pane is hidden there and the two figures come within two cells.

        Two cells go to the view's own horizontal padding (`.settings-view`
        declares `padding: 1 1 1 1`), the same reservation `_list_width` makes
        for the scrollbar gutter — derived from the view's width for the reason
        :meth:`_pane_width` records: children are still ``0`` wide at
        ``on_mount``.
        """
        try:
            total = self.size.width
        except Exception:
            total = 80
        if total <= 0:
            total = 80
        return max(total - 2, 24)

    def _list_width(self) -> int:
        """Cells a settings row may occupy, computed from the VIEW's width.

        The view's width is the one dimension known before the children are
        laid out (see :meth:`_pane_width`), so the list's share is derived
        rather than read back off the body. Two cells go to the scrollbar
        unconditionally: with ``scrollbar-gutter: stable`` the column is
        reserved whether or not the thumb is painted, and a fifty-row registry
        is taller than any viewport here anyway — assuming them away is what let
        rows wrap by exactly two cells. The reservation and the painted bar now
        agree rather than merely happening to match (design round 1, D1).
        """
        try:
            total = self.size.width
        except Exception:
            total = 80
        if total <= 0:
            total = 80
        if self._pane_fits():
            total -= _PANE_WIDTH
        return max(total - 2, 24)

    def _paint_pane(self) -> None:
        """The read-only side pane: providers plus teams or agents.

        Read-only, and it says so. Teams and agents are configured in files
        this page has no business rewriting, but a user who has never run
        ``/team`` does not know the feature exists — so the page makes them
        DISCOVERABLE without pretending to own them.
        """
        # Hidden, not squeezed, below the two-column threshold. `display` is
        # what the layout reads, so the settings body reclaims the cells
        # instead of the pane sitting there at an unreadable width.
        self._pane_view.display = self._pane_fits()
        if not self._pane_fits():
            self._pane_text = Text()
            return
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        head = Style(color=theme_mod.semantic_color("fg"), bold=True)
        accent = Style(color=theme_mod.semantic_color("accent"))

        room = self._pane_width()
        # Assembled as a LIST of lines rather than appended straight into a
        # Text, so the whole block can be measured against the pane's height
        # before it is painted. Appending directly is what let the pane paint
        # eight lines into seven rows: nothing downstream could see the total
        # (review round 2, D6).
        # Each line is a LIST of styled segments, because a provider row is two
        # inks on one line (the id in `muted`, its state in `faint`) and folding
        # them into a single style loses the distinction the row is built on.
        header: _PaneLine = [("providers", head)]
        provider_lines: list[_PaneLine] = []
        if self._providers:
            for name, state in self._providers:
                # Truncated per SEGMENT so the id survives and the state gives
                # way: "anthropic" is what the row is about, and a clipped id
                # is unreadable where a clipped state is merely terse.
                width = max(room - cell_len(name) - 4, 3)
                provider_lines.append(
                    [
                        (f"  {truncate_cells(name, room - 2)}  ", muted),
                        (truncate_cells(state, width), faint),
                    ]
                )
        else:
            # `room - 2` because the two-space indent is part of the painted
            # line: budgeting the text against the full width and THEN adding
            # the prefix overran the pane by exactly those two cells, which is
            # the one line in the pane that got this wrong (design round 2, D10).
            provider_lines.append(
                [(f"  {truncate_cells('none logged in — /login <provider>', room - 2)}", dim)]
            )
        # The header and the provider rows stay SEPARATE all the way into
        # `_fit_pane`, which sheds them as a section. Flattening them into one
        # list is what let the shedding loop delete `rendered[1]` believing it
        # was the separator when it was the first provider (design round 3,
        # D11). `used` below is the flat count the roster budget needs: the
        # header, the provider rows and the blank that follows them.

        # The tab row names BOTH panes and marks the live one, so ←/→ has a
        # visible target rather than being a key you have to already know about.
        tabs = Text(no_wrap=True, overflow="ellipsis")
        for pane in (_PANE_TEAMS, _PANE_AGENTS):
            tabs.append(f"{pane}  ", style=accent if pane == self._pane else dim)
        tabs.append("(←→)", style=faint)

        rows = self._teams if self._pane == _PANE_TEAMS else self._agents
        # Grouped per ROSTER ENTRY rather than flattened into lines, for the
        # reason the header/provider split records one section up: `_fit_pane`
        # has to shed a roster entry as a unit and turn it into a count, and it
        # cannot do that to a flat list of lines it can no longer attribute to
        # an entry. Flattening here is what made the roster's spill line — the
        # LAST element of the flat list — the FIRST thing the shedding loop
        # popped, so the roster lost the line saying it existed before it lost
        # the rows that line was summarising (design round 4, D15).
        entries: list[list[_PaneLine]] = []
        hidden = 0
        # The empty state is a STATEMENT about absence, not a roster that can be
        # folded into a count: `… 1 more` over an empty registry would invent an
        # agent. So it sheds by dropping its lines, exactly as the provider
        # section's own empty state does, and never grows a count.
        foldable = bool(rows)
        if not rows:
            # An HONEST empty state: name the thing that is missing and the
            # command that creates it. "No teams" alone leaves a user unable to
            # tell an empty registry from a broken page.
            verb = "/team" if self._pane == _PANE_TEAMS else "/agent"
            entries.append(
                [[(f"  {truncate_cells(f'no {self._pane} configured', room - 2)}", dim)]]
            )
            entries.append(
                [[(f"  {truncate_cells(f'{verb} lists and attaches them', room - 2)}", faint)]]
            )
        else:
            # The ROSTER is what gives way, not the caption. `read-only` used to
            # be appended unconditionally and simply fell off the bottom once
            # three agents at three rows each overran the pane's height — and
            # that caption is the one word carrying the editable/not boundary,
            # beside rows that ARE editable in the identical row grammar, in
            # exactly the pane a user is most likely to want to edit ("can I
            # change this agent's effort here?"). It degraded silently with
            # roster size (design round 1, D2). A clipped roster with an intact
            # boundary statement is the right trade, and the `+N more` line
            # makes the clipping visible rather than silent.
            #
            # `used` counts the head block and the tab row, both of which are
            # already decided above.
            used = 1 + len(provider_lines) + 1
            shown, hidden = self._budget_pane_rows(rows, used=used + 1)
            for name, facts, summary in shown:
                entry: list[_PaneLine] = [[(f"  {truncate_cells(name, room - 2)}", muted)]]
                entry.append([(f"    {truncate_cells(facts, room - 4)}", faint)])
                if summary:
                    entry.append([(f"    {truncate_cells(summary, room - 4)}", faint)])
                entries.append(entry)

        self._pane_text = self._fit_pane(
            header,
            provider_lines,
            tabs,
            entries,
            hidden,
            dim,
            roster_foldable=foldable,
        )
        self._pane_view.update(self._pane_text)

    def _fit_pane(
        self,
        header: "_PaneLine",
        provider_lines: list["_PaneLine"],
        tabs: Text,
        entries: list[list["_PaneLine"]],
        roster_hidden: int,
        dim: Style,
        *,
        roster_foldable: bool,
    ) -> Text:
        """Assemble the pane so the ``read-only`` caption is always the last row.

        The caption is reserved OUTSIDE the competition rather than appended
        after it, because it is the one line carrying the editable/not boundary
        and it was the line that fell off. ``_budget_pane_rows`` gets the roster
        right whenever its height derivation holds, but its ``room <= 0`` early
        return handed back "show nothing, hide everything" while the caller went
        on to paint a spill line AND the caption anyway — eight lines into a
        seven-row pane, with `read-only` the row that vanished, which is D2's
        symptom one size band below where D2 was tested (review round 2, D6).

        So the fit is enforced HERE, on the assembled block, where the total is
        knowable. Shedding runs least-load-bearing first: the roster and its
        spill line, then the blank separator, then the provider detail rows.
        The tab row and the caption survive to the last two rows, because a pane
        that cannot say what it is or whether it is editable is worse than no
        pane at all.

        BOTH sections shed into a COUNT rather than vanishing, and that symmetry
        is the invariant rather than a per-section rule. The first version of
        this ladder gave the guarantee to providers and took it from the roster
        in the same commit: step 1 popped from the end of a FLAT list of roster
        lines, whose last element is the ``… N more`` spill line, so the roster
        lost its count before it lost the rows the count summarised — the exact
        inversion of step 3's provider logic, painting `teams  agents  (←→)` over
        nothing while three agents were configured (design round 4, D15). The
        roster now arrives as ENTRIES with a separate ``roster_hidden``, so the
        same "pop a unit, increment the count" move is available to it, and both
        sections collapse to a one-line count before either disappears.

        Provider rows shed into a COUNT rather than vanishing. The previous
        version deleted from index 1 — which is the first provider row, not the
        separator its comment named — so between 20 and 26 terminal rows it ate
        signed-in providers from the top, and at 20-24 rows it painted a bold
        `providers` header with nothing under it. That frame is indistinguishable
        from the honest empty state the page deliberately paints (`none logged
        in — /login <provider>`), so the pane stated the OPPOSITE of the truth
        about the user's own logins while satisfying the height invariant D6
        added (design round 3, D11). A `… N more` line is the vocabulary the
        roster already uses for the same trade, and the last provider row is
        given up to make space for it rather than the count displacing content
        silently. Below that the header and the count collapse onto ONE line
        (`providers  … 3 more`), and only when even that will not fit does the
        section go entirely. Saying nothing about providers is honest; a header
        standing over nothing is the one shape this pane must never paint,
        because it is indistinguishable from the empty state.
        """
        height = self._pane_height()
        if height <= 1:
            # One row to spend: it goes to the boundary statement, not to the
            # `providers` header. A pane that shows a roster-ish line without
            # saying it is read-only is the failure D2 and D6 are both about;
            # a pane showing only `read-only` is merely small.
            return Text("read-only", style=dim, no_wrap=True, overflow="ellipsis")
        # Two rows for the caption block (its blank separator and the word),
        # collapsing to one when the pane cannot afford the separator. The
        # separator is a SPACING row, so it sheds later than the roster but
        # earlier than any row carrying a fact — see step 5.
        separator = height >= 3

        def available() -> int:
            return height - (2 if separator else 1)

        roster = list(entries)
        spilled = roster_hidden
        roster_collapsed = False
        shown = list(provider_lines)
        hidden = 0
        gap = True
        # The empty state (`none logged in — /login <provider>`) is a STATEMENT,
        # not a row that can be folded into a count — "… 1 more" over an empty
        # registry would invent a provider. So it sheds with its header instead.
        foldable = bool(self._providers)

        collapsed = False

        def provider_block() -> list[_PaneLine]:
            """The provider section exactly as it would paint right now."""
            if collapsed:
                # Header and count on ONE line, so a section with configured
                # providers still states that it has them when it cannot afford
                # a row each. `providers  … 3 more` is two cells longer than the
                # header alone and says something true, where the header alone
                # says the opposite of the truth.
                return [[("providers", header[0][1]), (f"  … {hidden} more", dim)]]
            if not shown and not hidden:
                return []
            block = [header, *shown]
            if hidden:
                block.append([(f"  … {hidden} more", dim)])
            return block

        def roster_block() -> list[_PaneLine]:
            """The roster's own lines, BELOW the tab row, as they would paint.

            Excludes the tab row itself, which is painted from ``tabs`` and
            costs its line unconditionally. When the roster collapses, the count
            rides on that tab row instead of appearing here, so this returns
            nothing and the caller adds the count to the tab line.
            """
            if roster_collapsed:
                return []
            block: list[_PaneLine] = []
            for entry in roster:
                block.extend(entry)
            if spilled:
                # `+` dropped from the count. `+ add a hop` and `+ add a chain`
                # are BUTTONS in the same frame and the same ink, so leading a
                # statement of fact with the page's own "this row adds one"
                # glyph made a count read as an affordance (design round 2, D9).
                block.append([(f"  … {spilled} more", dim)])
            return block

        def painted() -> int:
            # The tab row always costs its one line: a pane that cannot say
            # which of the two rosters it is showing is worse than a short one.
            # The collapse rung does not save that line, it MOVES the count onto
            # it — `teams  agents  (←→)  … 3 more` is one line where the count
            # was a second one.
            return len(provider_block()) + (1 if gap else 0) + 1 + len(roster_block())

        # 1. Roster ENTRIES fold into a count, LAST first — the same priority
        #    step 3 applies to providers. The count is what survives, because a
        #    roster that says "… 3 more" is telling the truth about a registry
        #    the pane cannot afford to list, while a tab row over nothing is
        #    indistinguishable from the honest empty state this pane paints for
        #    a genuinely empty registry (design round 4, D15).
        while painted() > available() and roster and roster_foldable:
            roster.pop()
            spilled += 1
        # 2. An unfoldable roster is the empty STATEMENT, which has no count to
        #    fold into, so its lines simply go — the same trade step 6 makes for
        #    the provider empty state. The tab row still says which pane is live.
        while painted() > available() and roster and not roster_foldable:
            roster.pop()
        # 3. Down to one line for the whole roster: the tab row carries the count
        #    itself. `teams  agents  (←→)  … 3 more` is the roster's exact
        #    analogue of `providers  … 3 more`, and it is what keeps the pane
        #    from claiming an empty roster in order to fit its box.
        if painted() > available() and roster_foldable and spilled:
            roster_collapsed = True
        # 4. The blank separating the provider block from the tab row.
        if painted() > available():
            gap = False
        # 5. Provider rows fold into a count, LAST first, so the pane keeps
        #    saying how many it is withholding instead of silently dropping
        #    them (design round 3, D11).
        while painted() > available() and shown and foldable:
            shown.pop()
            hidden += 1
        # 6. Down to one line for the whole section: the header carries the
        #    count itself rather than standing over nothing. A `providers`
        #    header with no rows beneath it reads as "none configured", which is
        #    the opposite of the truth when providers ARE signed in — the one
        #    statement this pane must never make (design round 3, D11).
        if painted() > available() and foldable:
            collapsed = True
        # 7. The caption's own blank line. It is spacing, and spacing is worth
        #    less than the fact that providers exist: at a 3-row pane (a 20-row
        #    terminal) keeping it is what forced the section out entirely, so
        #    the pane spent a row on whitespace to say nothing about three
        #    signed-in providers. `read-only` still sits last, just without the
        #    gap above it.
        if painted() > available() and separator:
            separator = False
        # 8. No room even for one line: the provider section goes entirely.
        #    Saying NOTHING about providers is honest; announcing an empty one
        #    is not. The roster has no equivalent rung, because its tab row is
        #    reserved outside the competition with the caption — so the roster's
        #    last surviving form is always the collapsed count, never silence.
        if painted() > available():
            collapsed = False
            shown = []
            hidden = 0

        text = Text(no_wrap=True, overflow="ellipsis")
        for segments in provider_block():
            for chunk, style in segments:
                text.append(chunk, style=style)
            text.append("\n")
        if gap:
            text.append("\n")
        text.append_text(tabs)
        if roster_collapsed:
            # The count rides on the tab row itself. The tab row keeps its own
            # accent because it still names the live pane; the count follows in
            # `dim` so it reads as a fact rather than as a third tab.
            text.append(f"  … {spilled} more", style=dim)
        text.append("\n")
        for segments in roster_block():
            for chunk, style in segments:
                text.append(chunk, style=style)
            text.append("\n")
        if separator:
            text.append("\n")
        text.append("  read-only", style=dim)
        return text

    def _budget_pane_rows(
        self, rows: Sequence[tuple[str, str, str]], *, used: int
    ) -> tuple[list[tuple[str, str, str]], int]:
        """Split the roster into what fits above the caption, and the rest.

        Each entry costs up to three rows (name, facts, an optional summary),
        and the caption plus its leading blank line costs two that are reserved
        here rather than competed for. A `+N more` line is only worth its own
        row when something is actually hidden, so the budget is recomputed once
        the spill line is known to be needed — otherwise an exactly-fitting
        roster would shed an entry to make room for a line saying it had.
        """
        height = self._pane_height()
        # Two for the caption block; one held back for a possible `+N more`.
        room = height - used - 2
        if room <= 0:
            return ([], len(rows))

        def cost(entry: tuple[str, str, str]) -> int:
            return 3 if entry[2] else 2

        def take(limit: int) -> int:
            spent = 0
            count = 0
            for entry in rows:
                spent += cost(entry)
                if spent > limit:
                    break
                count += 1
            return count

        count = take(room)
        if count < len(rows):
            count = take(room - 1)
        return (list(rows[:count]), len(rows) - count)

    def _pane_height(self) -> int:
        """Rows the pane has to work with, derived from the VIEW's height.

        Derived rather than read off ``self._pane_view.size`` for the reason
        :meth:`_pane_width` records in full: ``_repaint`` runs from
        ``on_mount``, before Textual has laid the children out, so every child
        size is still ``0`` at that moment and a budget computed against zero
        paints a first frame that differs from the settled one.
        """
        try:
            height = self.size.height
        except Exception:
            height = 0
        if height <= 0:
            return 14
        # Floored at 1, not at 4. A floor of 4 claimed rows the pane does not
        # have on a very short terminal — at 16 rows the view is 7 tall and the
        # pane is laid out 1 row high, so a budget of 4 overpainted it by three
        # lines however carefully the caller then measured (review round 2, D6).
        # Measured against the laid-out `_pane_view.size.height` at every height
        # from 16 to 44: `view height - _PANE_CHROME_ROWS` matches it exactly
        # once the floor stops interfering.
        return max(height - _PANE_CHROME_ROWS, 1)

    def _paint_chrome(self) -> None:
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        head = Style(color=theme_mod.semantic_color("fg"), bold=True)
        title = Text(no_wrap=True, overflow="ellipsis")
        title.append("settings", style=head)
        title.append("  ·  ", style=muted)
        # The FILE is named on the title, because "where did that go?" is the
        # question an immediate write raises and the answer is not guessable.
        #
        # Truncated HERE rather than left to the Text's own ellipsis: a segment
        # that overflowed the line was dropped whole by the renderer rather
        # than cut, so a long config path (a scratch dir under /var/folders,
        # which is exactly what a capture and a test use) rendered the title as
        # "settings  ·" with no path at all.
        path = self._config_path()
        title.append(truncate_cells(path, max(self._title_room(), 12)), style=dim)
        self._title_text = title
        self._title.update(title)

        width = max(self.size.width - 2, 1)
        self._rule_text = Text("─" * width, style=dim)
        self._rule.update(self._rule_text)
        self._paint_hints()

    def _current_is_readonly(self) -> bool:
        """Is the highlighted row a retired setting that no key can act on?

        Only while nothing is in progress: an open editor or an armed delete
        owns the footer's wording (it teaches `enter save` / `esc cancel`), and
        neither state can be entered from a read-only row anyway.
        """
        if self._editing is not None or self._confirm_delete is not None:
            return False
        row = self._current()
        return row is not None and row.setting is not None and row.setting.kind is Kind.READONLY

    def _title_room(self) -> int:
        """Cells left for the config path after the ``settings ·`` lead."""
        try:
            return self.size.width - 16
        except Exception:
            return 48

    def _paint_hints(self) -> None:
        """Lay out the footer, shedding WHOLE hints until the row fits.

        The same ladder ``OrgChartView`` uses and for the same reason: a footer
        rendered as one over-wide string is clipped mid-word by the terminal
        ("…back to conversatio"). ``esc`` is never shed — it is the only way
        out — and each rung is measured before it is committed.
        """
        # Annotated: without it each label infers as a distinct `Literal`, and
        # the resulting union is not assignable to `rung`'s `str`/`bool` tuple.
        move: _Hint = (self._move_hint, " move", False)
        enter: _Hint = (self._enter_hint, " change", True)
        reset: _Hint = (self._reset_hint, " default", True)
        pane: _Hint = (self._pane_hint, " panes", True)

        # The footer states what the keys do RIGHT NOW, per state, because it
        # is the row users scan for exactly that. Two states rewrite it:
        #
        # While a delete confirmation is pending, `esc` cancels the ask rather
        # than leaving the page — the code was already correct (the ask consumes
        # the key before the page-exit rung) but the footer advertised the other
        # meaning while the detail row said `esc cancels`, so both were on
        # screen at once disagreeing about one key (design round 2, D7).
        #
        # While an editor is open, MOVING SAVES. The page committed on move but
        # taught only `enter saves · esc cancels`, and a two-exit contract with
        # `esc` beside it implies that anything else is neither — which is how a
        # user arrows away from a value expecting to abandon it and stores it
        # instead (UX round 2, U14). Naming it on the move hint puts the rule on
        # the key it applies to.
        if self._confirm_delete is not None:
            exit_label = "cancel"
        else:
            exit_label = "back to conversation"
        if self._editing is not None:
            move = (self._move_hint, " move · saves", False)
            enter = (self._enter_hint, " save", True)

        def rung(
            leads: list[tuple[HintButton, str, bool]], esc_label: str
        ) -> tuple[list[tuple[HintButton, str, bool]], str]:
            row = list(leads)
            row.append((self._exit_hint, esc_label, bool(row)))
            return (row, esc_label)

        # The pane hint is only offered when there IS a pane. A lit hint whose
        # key does nothing is the "nothing happens when I click" bug one step
        # earlier — the same rule `HintButton.set_actionable` states.
        leads: list[_Hint] = [move, enter, reset]
        # The same rule applied to the ROW: on a retired setting neither key
        # does anything — `enter` only reports that it cannot be changed and `r`
        # returns without resetting — so neither is offered. The last six rows
        # of the page are read-only, and the clamp turned the bottom from a
        # waypoint into a place users park, under a footer promising `enter
        # change · r default` on a row that honours neither (UX round 1, U2).
        # The detail line already says WHY the row is retired; the footer's job
        # is only to stop advertising keys that will not act.
        if self._current_is_readonly():
            leads = [move]
        if self._pane_fits():
            leads.append(pane)
        # The narrow rungs shed to a one-word `esc` label, except in the
        # confirm state where `cancel` IS the one word and shedding it back to
        # `back` would restore the very ambiguity D7 is about.
        narrow = "cancel" if self._confirm_delete is not None else "back"
        # The narrower rungs are DERIVED from what this row actually offers,
        # dropping one hint at a time from the right, rather than restating the
        # full ladder: a hardcoded `[move, enter, reset]` rung would put the
        # keys back on a read-only row as soon as the terminal got narrow
        # enough to shed the pane hint.
        shed = [rung(leads[:count], narrow) for count in range(len(leads) - 1, -1, -1)]
        rungs = [
            rung(leads, exit_label),
            rung(leads, narrow),
            *shed,
        ]
        width = max(self.size.width - 2, 1)
        chosen = rungs[-1]
        for plan, esc_label in rungs:
            if self._measure_hints(plan, esc_label) <= width:
                chosen = (plan, esc_label)
                break
        plan, esc_label = chosen
        visible = {hint for hint, _label, _lead in plan}
        for hint, label, lead in plan:
            hint.paint(esc_label if hint is self._exit_hint else label, lead=lead)
        for hint in (
            self._move_hint,
            self._enter_hint,
            self._reset_hint,
            self._pane_hint,
            self._exit_hint,
        ):
            hint.display = hint in visible

    def _measure_hints(self, plan: list[tuple[HintButton, str, bool]], esc_label: str) -> int:
        row = Text()
        for hint, label, lead in plan:
            row.append(hint.preview(esc_label if hint is self._exit_hint else label, lead=lead))
        return cell_len(row.plain)

    # -- test hooks ---------------------------------------------------------
    def render_lines_for_test(self) -> list[str]:
        """The page as plain strings — title, rule, rows. Assertable.

        The same hook ``UsagePanel`` and ``SessionPickerScreen`` expose, and for
        the same reason: a test that asserts on the strings a user reads cannot
        drift from the frame the way one asserting on widget internals can.
        """
        rows = [self._title_text.plain, self._rule_text.plain]
        rows.extend(self._list_text.plain.split("\n"))
        rows.append(self._detail_text.plain)
        return rows

    def detail_spans(self) -> list[tuple[str, Style]]:
        """The detail row as ``(text, style)`` pairs.

        The styled form is exposed because the row's three meanings — a help
        string, a validation ERROR and a destructive ASK — are distinguished by
        ink and weight, not by text. A test asserting only on the plain string
        cannot tell an ask from a report, which is the confusion design round 2
        (D7) found in the frame itself.
        """
        from rich.console import Console

        return [
            (segment.text, segment.style or Style())
            for segment in self._detail_text.render(Console())
        ]

    def rendered_pane(self) -> str:
        """The read-only pane as plain text. Assertable, and "" when hidden."""
        return self._pane_text.plain

    def rendered_hints(self) -> str:
        """The footer as one string, for the width-shedding assertions."""
        return "".join(
            hint.rendered()
            for hint in (
                self._move_hint,
                self._enter_hint,
                self._reset_hint,
                self._pane_hint,
                self._exit_hint,
            )
            if hint.display
        )

    @property
    def selected_key(self) -> str | None:
        """The highlighted row's setting key, or None on a non-setting row."""
        row = self._current()
        if row is None or row.setting is None:
            return None
        return row.setting.key

    @property
    def editing_key(self) -> str | None:
        """Which editor is open, or None."""
        return self._editing

    @property
    def error_text(self) -> str:
        """The inline validation error currently on screen ("" when none)."""
        return self._error

    @property
    def notice_text(self) -> str:
        """The informational detail-row message ("" when none).

        Separate from :attr:`error_text` because the two are inked differently
        and a test that could not tell them apart is how an informational
        message ended up in the danger colour (UX round 2, U16).
        """
        return self._notice

    # -- leaving ------------------------------------------------------------
    def _focus_self(self) -> None:
        """Focus the page (the move hint's click action).

        Routed through a None-returning helper because ``focus()`` returns
        ``self``, which the ``Callable[[], None]`` action type rejects — the
        same wrapper ``OrgChartView`` needs for its scroll hint.
        """
        try:
            self.focus()
        except Exception:
            pass

    def action_leave(self) -> None:
        self._leave()

    def _leave(self) -> None:
        self.post_message(SettingsViewDismissed())


class _Row:
    """One painted line: a header, a setting, a choice, or a cascade entry.

    A plain class rather than a dataclass because the union is loose by design
    — a header has a section and no setting, a hop has neither — and declaring
    every field Optional on a frozen dataclass buys nothing over this.
    """

    __slots__ = ("kind", "section", "setting", "choice", "chain", "hop", "hop_index", "text")

    def __init__(
        self,
        *,
        kind: str,
        section: Section | None = None,
        setting: Setting | None = None,
        choice: Any = None,
        chain: str | None = None,
        hop: str | None = None,
        hop_index: int = -1,
        text: str = "",
    ) -> None:
        self.kind = kind
        self.section = section
        self.setting = setting
        self.choice = choice
        self.chain = chain
        self.hop = hop or ""
        self.hop_index = hop_index
        self.text = text

    @property
    def selectable(self) -> bool:
        """Whether the cursor may land here.

        Headers and the empty-state line are not selectable: there is nothing
        to do on them, and a cursor that stopped on them would make every
        section boundary cost an extra keypress to cross.
        """
        return self.kind not in ("header", "empty")

    @property
    def identity(self) -> tuple[str, str, int]:
        """What this row IS, independent of where it currently sits.

        Movement needs this because a commit can rebuild the row list under the
        mover: committing a new chain inserts its chain row, its hops and an
        ``+ add a hop`` row ABOVE the cursor, so an index resolved before the
        commit points somewhere else after it, and the user who pressed ``down``
        lands back where they started (review round 2, U13). An identity
        survives the rebuild where an index does not.

        Deliberately NOT the ``hop`` text: a hop row's identity is its position
        in its chain, and the text is what an edit changes.
        """
        key = self.setting.key if self.setting is not None else ""
        choice = str(getattr(self.choice, "value", "")) if self.choice is not None else ""
        return (self.kind, f"{key}|{self.chain or ''}|{choice}", self.hop_index)


def _render_value(value: Any) -> str:
    """One vocabulary for every displayed value.

    Booleans read as ``on``/``off`` rather than ``True``/``False`` because that
    is the word the choices use and the word ``config edit`` accepts; ``None``
    reads as ``auto`` because that is what absence MEANS for the one tri-state
    here (``display.nerd_icons``), not as an empty cell that would look broken.
    """
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "—"
    if isinstance(value, Mapping):
        return f"{len(value)} chain{'' if len(value) == 1 else 's'}"
    text = str(value)
    return text if text else "—"


def _edit_seed(value: Any) -> str:
    """What an editor opens with for a STORED value — the other vocabulary.

    Deliberately NOT :func:`_render_value`. That function answers "what does
    this row read as", and its answers include ``—`` for absent and ``auto``
    for ``None``; both are glyphs for the ABSENCE of a value, and neither is
    something a user could have typed or that the coercers would accept back.
    Feeding them into an editable buffer is what let the placeholder be
    committed as a real value (UX round 1, U1), so the two vocabularies are
    kept apart here rather than sharing one function with a flag: a display
    string and an editable string are different questions about the same value.

    Booleans are absent on purpose — a BOOL row toggles and never opens an
    editor, so there is no ``on``/``off`` case to round-trip.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _home_relative(path: str) -> str:
    """``~/...`` for a path under the user's home; unchanged otherwise."""
    from pathlib import Path

    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except Exception:
        return path


#: Exported so the app can type its handler without importing the module twice.
SettingsCallback = Callable[[str, Any], None]
