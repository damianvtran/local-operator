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

#: Marker on the row the cursor is on. A glyph rather than a background sweep
#: so the row's own changed/default ink survives the highlight.
_CURSOR = "›"

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
        #: config.yml on every repaint, and two managers in one process is how
        #: a write lands in one instance's memory and not the other's (the bug
        #: ``ConfigManager.reload`` exists for).
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
        #: Cascade editor state: which chain is open for hop editing, if any.
        self._chain: str | None = None
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
                    for choice in setting.choices:
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
            rows.append(_Row(kind="empty", text="no cascade configured"))
        for key, hops in sorted(chains.items()):
            rows.append(_Row(kind="chain", chain=key, setting=setting))
            if self._chain == key:
                for index, hop in enumerate(hops):
                    rows.append(_Row(kind="hop", chain=key, hop=hop, hop_index=index))
                rows.append(_Row(kind="hop_add", chain=key))
        rows.append(_Row(kind="chain_add", setting=setting))
        return rows

    def _selectable(self) -> list[int]:
        return [index for index, row in enumerate(self._rows) if row.selectable]

    def _current(self) -> "_Row | None":
        if 0 <= self._selected < len(self._rows):
            return self._rows[self._selected]
        return None

    # -- movement -----------------------------------------------------------
    def action_move(self, delta: int) -> None:
        """Move the cursor by ``delta`` selectable rows, WRAPPING.

        Wrapping because an arrow key is a discrete, deliberate press: a user
        holding ``down`` at the bottom of a list expects to come round, and the
        alternative (silently stopping) reads as a stuck key. Page and wheel
        clamp instead — see :meth:`action_section` and the scroll handlers.
        """
        indices = self._selectable()
        if not indices:
            return
        if self._selected in indices:
            position = indices.index(self._selected)
        else:
            position = 0
        self._selected = indices[(position + delta) % len(indices)]
        self._cancel_edit()
        self._repaint()
        self._scroll_to_selection()

    def action_section(self, delta: int) -> None:
        """PgUp/PgDn jump to the next section header's first row, CLAMPED."""
        headers = [index for index, row in enumerate(self._rows) if row.kind == "header"]
        if not headers:
            return
        current = max([index for index in headers if index <= self._selected] or [headers[0]])
        position = headers.index(current)
        # Clamped: paging past the last section should sit on the last section,
        # not wrap to the first. A page gesture is travel, and travel that
        # teleports across the whole document is how a reader loses their place.
        target = min(max(position + delta, 0), len(headers) - 1)
        self._select_after(headers[target])

    def action_jump(self, to_end: int) -> None:
        indices = self._selectable()
        if not indices:
            return
        self._selected = indices[-1] if to_end else indices[0]
        self._cancel_edit()
        self._repaint()
        self._scroll_to_selection()

    def _select_after(self, header_index: int) -> None:
        """Put the cursor on the first selectable row at or after a header."""
        for index in range(header_index, len(self._rows)):
            if self._rows[index].selectable:
                self._selected = index
                break
        self._cancel_edit()
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
        panes = [_PANE_TEAMS, _PANE_AGENTS]
        position = (panes.index(self._pane) + delta) % len(panes)
        self._pane = panes[position]
        self._repaint()

    def _scroll_to_selection(self) -> None:
        """Keep the cursor row inside the scrolled viewport.

        The body is a ScrollableContainer around ONE painted Static, so there
        is no child widget to call ``scroll_visible`` on — the offset is
        computed from the row index directly. Guarded because the container has
        no size until it is laid out, and a first movement before layout would
        divide by a zero height.
        """
        try:
            height = self._body.size.height
        except Exception:
            return
        if height <= 0:
            return
        offset = self._body.scroll_offset.y
        if self._selected < offset:
            self._body.scroll_to(y=self._selected, animate=False)
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
        if self._editing is not None:
            self._commit_edit()
            return
        if row.kind == "choice" and row.setting is not None and row.choice is not None:
            self._write(row.setting, row.choice.value)
            self._expanded = None
            self._repaint()
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
        self._begin_edit(row)

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
        row = self._current()
        if row is None or row.setting is None or row.kind != "setting":
            return
        if row.setting.kind in (Kind.READONLY, Kind.CASCADE):
            return
        try:
            settings_io.reset_setting(self._manager, row.setting)
        except ValueError as error:
            self._error = str(error)
            self._repaint()
            return
        self._error = ""
        self.post_message(
            SettingsChanged(row.setting.key, settings_io.read_setting(self._manager, row.setting))
        )
        self._repaint()

    def _write(self, setting: Setting, value: Any) -> None:
        """Store ``value`` and report it, or hold the reason it was refused."""
        try:
            settings_io.write_setting(self._manager, setting, value)
        except ValueError as error:
            self._error = str(error)
            self._repaint()
            return
        except Exception as error:  # noqa: BLE001 — a read-only config dir, a full disk
            # Reported, never raised: a page that crashed on an unwritable
            # config would take the whole TUI down at the exact moment the user
            # is trying to fix their configuration.
            self._error = f"could not save: {error}"
            self._repaint()
            return
        self._error = ""
        self.post_message(SettingsChanged(setting.key, value))
        self._repaint()

    # -- text editing -------------------------------------------------------
    def _begin_edit(self, row: "_Row") -> None:
        """Open the inline editor on ``row``, seeded with its current value."""
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
            self._editing = row.setting.key
            self._buffer = _render_value(settings_io.read_setting(self._manager, row.setting))
        else:
            return
        self._error = ""
        self._repaint()

    def _cancel_edit(self) -> None:
        self._editing = None
        self._buffer = ""
        self._error = ""

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
        text = self._buffer.strip()
        if not text and setting.empty_unsets:
            # An empty field UNSETS rather than storing "": for `hosting` and
            # the subagent tiers, "" and absent mean the same thing to their
            # consumers, and storing the empty string leaves a key in the file
            # that reads as a deliberate choice of nothing.
            try:
                settings_io.reset_setting(self._manager, setting)
            except ValueError as error:
                self._error = str(error)
                self._repaint()
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

    def _commit_hop(self, target: str) -> None:
        """Save an edited or newly-added cascade hop."""
        problem = settings_io.validate_hop(self._buffer)
        if problem is not None:
            self._error = problem
            self._repaint()
            return
        chains: dict[str, list[str]] = {
            key: list(hops) for key, hops in settings_io.read_chains(self._manager).items()
        }
        _, chain_key, *rest = target.split(":")
        hops = chains.setdefault(chain_key, [])
        if target.startswith("hopadd:"):
            hops.append(self._buffer.strip())
        else:
            index = int(rest[0])
            if 0 <= index < len(hops):
                hops[index] = self._buffer.strip()
        settings_io.write_chains(self._manager, chains)
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
        chains: dict[str, list[str]] = {
            existing: list(hops)
            for existing, hops in settings_io.read_chains(self._manager).items()
        }
        chains[key] = [first_hop.strip()]
        settings_io.write_chains(self._manager, chains)
        self._chain = key
        self._cancel_edit()
        self.post_message(SettingsChanged("retry.fallbackChains", chains))
        self._repaint()

    def _delete_hop(self) -> None:
        """Remove the highlighted hop, or the whole chain from its key row."""
        row = self._current()
        if row is None:
            return
        chains: dict[str, list[str]] = {
            key: list(hops) for key, hops in settings_io.read_chains(self._manager).items()
        }
        if row.kind == "hop" and row.chain in chains:
            hops = chains[row.chain]
            if 0 <= row.hop_index < len(hops):
                del hops[row.hop_index]
        elif row.kind == "chain" and row.chain in chains:
            del chains[row.chain]
            self._chain = None
        else:
            return
        settings_io.write_chains(self._manager, chains)
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
        chains: dict[str, list[str]] = {
            key: list(hops) for key, hops in settings_io.read_chains(self._manager).items()
        }
        hops = chains.get(row.chain, [])
        index = row.hop_index
        target = index + delta
        if not (0 <= index < len(hops)) or not (0 <= target < len(hops)):
            return
        hops[index], hops[target] = hops[target], hops[index]
        settings_io.write_chains(self._manager, chains)
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
        reason, and the Esc LADDER (editor → expansion → page, one press each)
        falls out of the same ordering.
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
                self._buffer = self._buffer[:-1]
                self._repaint()
                return
            char = getattr(event, "character", None)
            if char and char.isprintable():
                event.stop()
                event.prevent_default()
                self._buffer += char
                self._repaint()
            return
        if key == "escape" and self._expanded is not None:
            # Rung two of the ladder: Esc closes the EXPANSION before it closes
            # the page, so a user who opened a dropdown to look at it can back
            # out of it without losing the whole surface.
            event.stop()
            event.prevent_default()
            self._expanded = None
            self._repaint()
            return
        row = self._current()
        if row is not None and row.kind in ("hop", "chain"):
            if key in ("d", "delete"):
                event.stop()
                event.prevent_default()
                self._delete_hop()
                return
            if key in ("shift+up", "shift+down") and row.kind == "hop":
                event.stop()
                event.prevent_default()
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
        self._selected = index
        self._cancel_edit()
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
        """Wheel movement — CLAMPED, unlike the wrapping arrows.

        The convention this repo states in AGENTS.md: arrows wrap because a
        press is deliberate, wheel and page clamp because a gesture that
        teleports to the other end of the list reads as the list resetting.
        """
        indices = self._selectable()
        if not indices:
            return
        position = indices.index(self._selected) if self._selected in indices else 0
        self._selected = indices[min(max(position + delta, 0), len(indices) - 1)]
        self._cancel_edit()
        self._repaint()
        self._scroll_to_selection()

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
            room = width - cell_len(head.plain) - 2
            tag = full if cell_len(full) <= room else (short if cell_len(short) <= room else "")
            if tag:
                head.append(" " * max(1, width - cell_len(head.plain) - cell_len(tag)))
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
        label = truncate_cells(setting.label, _VALUE_COLUMN - _ROW_INDENT - 4)
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
        editor.append(self._buffer, style=Style(color=theme_mod.semantic_color("fg")))
        editor.append("▏", style=accent)
        # The CONTRACT rides the row; the ERROR does not. The detail line below
        # already carries the rejection in full width, and printing it twice
        # read as two separate problems — the row's copy also had to compete
        # with the value column for space it does not have.
        editor.append("  enter saves · esc cancels · clear to unset", style=faint)
        return editor

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
        if self._error:
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

    def _list_width(self) -> int:
        """Cells a settings row may occupy, computed from the VIEW's width.

        The view's width is the one dimension known before the children are
        laid out (see :meth:`_pane_width`), so the list's share is derived
        rather than read back off the body. The scrollbar's two cells are
        always subtracted because a fifty-row registry is always taller than
        the viewport — assuming them away is what let rows wrap by exactly two
        cells.
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
        text = Text(no_wrap=True, overflow="ellipsis")

        room = self._pane_width()
        text.append("providers\n", style=head)
        if self._providers:
            for name, state in self._providers:
                # Truncated per SEGMENT so the id survives and the state gives
                # way: "anthropic" is what the row is about, and a clipped id
                # is unreadable where a clipped state is merely terse.
                text.append(f"  {truncate_cells(name, room)}", style=muted)
                text.append(
                    f"  {truncate_cells(state, max(room - cell_len(name) - 4, 3))}\n",
                    style=faint,
                )
        else:
            text.append(
                f"  {truncate_cells('none logged in — /login <provider>', room)}\n", style=dim
            )
        text.append("\n")

        # The tab row names BOTH panes and marks the live one, so ←/→ has a
        # visible target rather than being a key you have to already know about.
        for pane in (_PANE_TEAMS, _PANE_AGENTS):
            style = accent if pane == self._pane else dim
            text.append(f"{pane}  ", style=style)
        text.append("(←→)\n", style=faint)

        rows = self._teams if self._pane == _PANE_TEAMS else self._agents
        if not rows:
            # An HONEST empty state: name the thing that is missing and the
            # command that creates it. "No teams" alone leaves a user unable to
            # tell an empty registry from a broken page.
            verb = "/team" if self._pane == _PANE_TEAMS else "/agent"
            text.append(f"  {truncate_cells(f'no {self._pane} configured', room)}\n", style=dim)
            text.append(
                f"  {truncate_cells(f'{verb} lists and attaches them', room)}\n", style=faint
            )
        else:
            for name, facts, summary in rows:
                text.append(f"  {truncate_cells(name, room - 2)}\n", style=muted)
                text.append(f"    {truncate_cells(facts, room - 4)}\n", style=faint)
                if summary:
                    text.append(f"    {truncate_cells(summary, room - 4)}\n", style=faint)
        text.append("\n  read-only", style=dim)
        self._pane_text = text
        self._pane_view.update(text)

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
        path = _home_relative(str(getattr(self._manager, "config_file", "")))
        title.append(truncate_cells(path, max(self._title_room(), 12)), style=dim)
        self._title_text = title
        self._title.update(title)

        width = max(self.size.width - 2, 1)
        self._rule_text = Text("─" * width, style=dim)
        self._rule.update(self._rule_text)
        self._paint_hints()

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
        if self._pane_fits():
            leads.append(pane)
        rungs = [
            rung(leads, "back to conversation"),
            rung(leads, "back"),
            rung([move, enter, reset], "back"),
            rung([move, enter], "back"),
            rung([move], "back"),
            rung([], "back"),
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


def _home_relative(path: str) -> str:
    """``~/...`` for a path under the user's home; unchanged otherwise."""
    from pathlib import Path

    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except Exception:
        return path


#: Exported so the app can type its handler without importing the module twice.
SettingsCallback = Callable[[str, Any], None]
