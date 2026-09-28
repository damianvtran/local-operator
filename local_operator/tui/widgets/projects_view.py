"""The full-page ``/project`` view — every tracked workstream on one page.

A MODE of the main screen, cloned from :class:`SettingsView` and, through it,
:class:`OrgChartView`/:class:`SubagentView`: the page takes the transcript's
region and leaves the dock (band, status, composer) where it is, greyed, so it
reads as the same app looking somewhere else. It is deliberately NOT a
``textual.push_screen`` and not a floating card — the design's refinement
(§V2.B.2) fixes the mechanics as ``SettingsView``'s.

WHAT THIS PAGE OWES THE USER
============================

- **Three views, one canvas.** ``1``/``2``/``3`` jump between list, board and
  timeline; ``v`` cycles. The list keeps a real cursor (clamped, reveal-then-act
  — the full-page exception AGENTS.md documents), board and timeline are
  canvases scrolled by the arrows. ``r`` asks the app to recompose; the page
  itself does no I/O — the same split ``/settings`` makes with its resolved
  rows, so a repaint costs no registry or filesystem read.
- **Honest absence.** A field the store does not know renders as a sentence
  (``no progress``, ``no estimate or dates``), and subagent/todo counts the
  data marks ``null`` are omitted, never shown as zeroes.
- **A way back.** The footer names ``esc`` and sheds whole hints widest-first
  the way the org chart's does, so a 50-column terminal still says how to leave.

The canvas is sized in Python to what :mod:`local_operator.tui.projects_render`
returns (the org-chart mechanics): the ``ScrollableContainer``'s virtual size
equals the canvas, so scrollbars appear exactly when the content overflows.

Identified by CLASS and not id, all the way down — the ``DuplicateIds``-on-fast-
reopen lesson ``org_chart_view`` records: ``remove()`` only POSTS a prune, so a
reopen inside that window would mount a second same-id widget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.projects_render import (
    PROJECTS_MAX,
    TIMELINE_TIERS,
    RenderResult,
    aggregate_footer,
    auto_timeline_tier,
    board_position,
    detail_footer,
    render_project_board,
    render_project_list,
    render_project_timeline,
    timeline_position,
    timeline_span,
)
from local_operator.tui.widgets.subagent_view import READ_ONLY_NOTE, HintButton

#: The view vocabulary, in the order ``1``/``2``/``3`` address it and ``v``
#: cycles through it. One tuple so the bindings, the cycle and the title cannot
#: disagree about what the third view is.
VIEWS: tuple[str, ...] = ("list", "board", "timeline")


class ProjectsViewDismissed(Message):
    """The page's ``esc`` hint was clicked. The app owns leaving the mode.

    A dedicated message, for the reason ``OrgChartView`` states: reusing a
    sibling mode's message would hit that mode's handler, which does not own
    this widget.
    """


class ProjectsViewRefreshRequested(Message):
    """``r`` was pressed. The APP recomposes — the page never reads the store.

    The page is a pure viewer (see the module docstring): recomposition means
    a registry read, a runtime scan and per-session tail reads, all of which
    belong to the app so the widget can be exercised without a filesystem.
    """


def _style_resolver() -> Callable[[str], Style]:
    """A style-key → ``rich.Style`` resolver bound to the CURRENT theme.

    Rebuilt on each render so a theme switch while the page is open repaints in
    the new palette — the org chart's rule, same implementation shape. The keys
    are the ones ``projects_render`` paints.
    """

    def color(token: str) -> str:
        return theme_mod.semantic_color(token)

    styles = {
        "name": Style(color=color("fg"), bold=True),
        "status": Style(color=color("accent")),
        # D4: per-status chips — `active` keeps the accent, `paused` warns,
        # `done`/`archived` recede. One mapping, read by the list rows, the
        # footer chip and the board.
        "status_active": Style(color=color("accent")),
        "status_paused": Style(color=color("warning")),
        "status_done": Style(color=color("muted")),
        "status_archived": Style(color=color("dim")),
        "cursor": Style(color=color("accent"), bold=True),
        # The session's own projects carry `◆` in the row's leading column
        # (S3b): the accent without the cursor's bold, so `▸` still owns the
        # glyph where both apply — the marker column costs no width either way.
        "session": Style(color=color("accent")),
        "dim": Style(color=color("dim")),
        # A live session is the accent-of-success: the one fact the page exists
        # to surface ("what is actually running?").
        "live": Style(color=color("success")),
        # Stale progress is a warning tone — the whole point of showing an age
        # is that an old line must not read like a fresh one.
        "stale": Style(color=color("warning")),
        "bar": Style(color=color("muted")),
        "milestone_done": Style(color=color("success")),
        "milestone_late": Style(color=color("danger")),
        "milestone_due": Style(color=color("muted")),
        "today": Style(color=color("accent")),
    }

    def resolve(key: str) -> Style:
        return styles.get(key, Style())

    return resolve


class ProjectsViewJumpRequested(Message):
    """``↵`` on the selected project: open its conversation (S3b).

    Carries the project's name and its linked sessions (id + state, a stale
    link as ``missing``) so the host can choose a live one — or say honestly
    what exists — without re-reading the store for one keystroke. The page
    never touches session machinery its host owns; it asks.
    """

    def __init__(
        self,
        *,
        project_id: str,
        project_name: str,
        sessions: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__()
        self.project_id = project_id
        self.project_name = project_name
        self.sessions = sessions


class ProjectsView(Vertical):
    """The page: a title, a rule, the scrollable canvas, the detail footer, hints.

    Class-identified (see the module docstring). ``can_focus`` so the view
    keys, the list cursor and the canvas scrolling land here rather than on the
    composer the mode made inert.
    """

    can_focus = True

    # Arrows CLAMP — this is the full-page mode AGENTS.md names as the second
    # member of the clamp exception (`/settings` is the first): its list is
    # several times its viewport, so the bottom is a destination, not a place
    # to wrap away from. In EVERY view up/down move the CURSOR (clamped, with
    # the canvas following through the reveal — UX round 1, U2 made board and
    # timeline selectable, not just pannable); ←/→ pan, and the page keys page
    # the canvas. `←→` are NOT
    # view-switch keys — they belong to the canvas scroll, because a page that
    # scrolls horizontally must keep its pan axis (the settings PANE-cycle
    # convention does not transfer). Shift+arrows page horizontally and
    # PageUp/Down vertically, the org-chart scheme.
    BINDINGS = [
        Binding("1", "show_list", "List view", show=False),
        Binding("2", "show_board", "Board view", show=False),
        Binding("3", "show_timeline", "Timeline view", show=False),
        Binding("v", "cycle_view", "Next view", show=False),
        Binding("r", "refresh", "Refresh", show=False),
        # Zoom is TIME resolution on the timeline (the org-chart "zoom is level
        # of detail" rule); in the other views it is inert and the footer sheds
        # the hint rather than advertising a key that does nothing.
        Binding("plus,equals_sign,equal", "zoom_in", "Finer time", show=False),
        Binding("minus,underscore", "zoom_out", "Coarser time", show=False),
        Binding("up", "up", "Up", show=False),
        Binding("down", "down", "Down", show=False),
        Binding("left", "scroll_left", "Scroll left", show=False),
        Binding("right", "scroll_right", "Scroll right", show=False),
        # The page keys still PAGE the canvas in board/timeline: with ↑/↓ on the
        # selection, a reader who wants to read further without moving `↵`'s
        # target reaches for these.
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("shift+left", "page_left", "Page left", show=False),
        Binding("shift+right", "page_right", "Page right", show=False),
        # Home/End jump to the ends of the CURSOR in the list view and to the
        # canvas corners elsewhere (org chart's explicit scroll_to, because
        # Textual's scroll_end reaches the bottom-LEFT, not the right edge).
        Binding("home", "scroll_home", "To start", show=False),
        Binding("end", "scroll_end", "To end", show=False),
        # `↵` opens the SELECTED project's conversation (S3b) — the one action
        # that reaches outside the page, and the same key in every view.
        Binding("enter", "jump", "Open", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(classes="projects-view")
        #: The composed project views (``build_project_view`` payloads), handed
        #: in by the app. Held so view/zoom/cursor repaints never re-read.
        self._views: list[dict[str, Any]] = []
        #: Project ids the CALLING session is linked to (S3b): their rows/cards/
        #: rows get the `◆` marker and the title names the set. Empty on the
        #: all-projects entries (bare `board`/`timeline`, and the nameless
        #: `show` of a session with no links).
        self._associated: frozenset[str] = frozenset()
        #: Current view type, the list cursor (list view only; clamped), the
        #: timeline tier, and the one-clock "updated at" the title states.
        self._view: str = "list"
        self._cursor: int = 0
        self._tier: str = "month"
        #: Whether the tier above was chosen by hand (``+``/``-``) and, if so,
        #: the dated span it was chosen for. A manual zoom survives a
        #: recomposition while that span is unchanged; a changed span
        #: re-derives the auto tier (the rule the zoom comments state, now
        #: actually implemented — agent review round 1, F2).
        self._tier_manual: bool = False
        self._manual_span: tuple[str, str] | None = None
        self._updated_at: float | None = None
        #: Last render, kept for the geometry probes and rendered_rows().
        self._last: RenderResult | None = None
        self._title = Static(classes="projects-view-title")
        self._rule = Static(classes="projects-view-rule")
        # The canvas is a Static inside a BOTH-AXES scroll container: the
        # Static is sized to the painted canvas and the container clips and
        # scrolls it, so Textual's own scrollbars appear exactly when over.
        self._canvas = Static(classes="projects-view-canvas")
        self._body = ScrollableContainer(self._canvas, classes="projects-view-body")
        # The pinned footer: the highlighted project's detail in the list view,
        # aggregate counts elsewhere. ALWAYS one row (a footer that appeared
        # and disappeared would move the body on every view switch).
        self._detail = Static(classes="projects-view-detail")
        self._scroll_hint = HintButton("↔↕", self._focus_canvas)
        # `↵` opens the selected project's conversation (S3b). The rung is a
        # pure addition to the BOARD's ladder — ` · ↵  open` = 10 cells (key +
        # label + seam) — and sheds first, ahead of `r refresh`. On the
        # TIMELINE it defers `+/- zoom`: the with-open+zoom plan measures 101
        # cells, so zoom flips back at 107 terminal columns (absent at 106,
        # avail 100; present at 107, avail 101 — this harness) where it used to
        # return at ~97 (open outranks zoom in `all_leads` — the recorded
        # trade; review rounds 2 R2-2b and 3 R3-1).
        self._open_hint = HintButton("↵", lambda: self.action_jump())
        self._list_hint = HintButton("1", lambda: self.action_show_list())
        self._board_hint = HintButton("2", lambda: self.action_show_board())
        self._timeline_hint = HintButton("3", lambda: self.action_show_timeline())
        self._next_hint = HintButton("v", lambda: self.action_cycle_view())
        self._refresh_hint = HintButton("r", lambda: self.action_refresh())
        self._zoom_hint = HintButton("+/-", self._cycle_tier)
        self._exit_hint = HintButton("esc", self._leave)
        self._state_hint = HintButton(READ_ONLY_NOTE)
        self._hints = Horizontal(classes="projects-view-hints")

    # -- data ---------------------------------------------------------------
    def load(
        self,
        *,
        views: list[dict[str, Any]],
        highlight: str | None = None,
        updated_at: float | None = None,
        view: str | None = None,
        associated: frozenset[str] | None = None,
    ) -> None:
        """Point the page at a fresh composition and paint it.

        ``highlight`` names a project id to put the list cursor on (``/project
        show <name>`` lands here); left ``None`` the cursor is KEPT — a refresh
        must not move the reader off the row they were reading. ``view`` opens
        a named canvas directly (``/project board`` and ``/project timeline``,
        which do not force the list), and ``associated`` marks the calling
        session's own project ids with ``◆`` (the nameless ``/project show``;
        S3b) — the title says whose set it is, and a load CARRYING a set seeds
        the cursor onto it (an entry), while one that carries none (a refresh)
        never moves the reader.
        """
        self._views = list(views)
        if associated is not None:
            self._associated = frozenset(str(pid) for pid in associated)
            if self._associated:
                # The caller's own set seeds the SELECTION (UX round 1, U1):
                # the nameless entry's one action (`↵`) must aim at a MEMBER,
                # and with several links the cursor otherwise stayed on row 0
                # — a project the reader has no link to. A reader already
                # inside their set keeps their row. The seed rides the load
                # that SPECIFIES a set: a refresh passes none (it must not move
                # the reader — R2-3), and an entry naming no set has nothing
                # to seed from.
                if self.current_project_id() not in self._associated:
                    for index, view_row in enumerate(self._views):
                        project = view_row.get("project") if isinstance(view_row, dict) else None
                        if isinstance(project, dict) and str(project.get("id")) in self._associated:
                            self._cursor = index
                            break
        if updated_at is not None:
            self._updated_at = updated_at
        if view is not None and view in VIEWS:
            self._view = view
        if highlight is not None:
            for index, view_row in enumerate(self._views):
                project = view_row.get("project") if isinstance(view_row, dict) else None
                if isinstance(project, dict) and str(project.get("id")) == str(highlight):
                    self._cursor = index
                    break
            if view is None and self._view != "list":
                # The cursor only exists on the list canvas; a NAMED `show`
                # means being able to see it, so it lands on the list. A
                # requested `view` (the board/timeline entries) outranks that.
                self._view = "list"
        self._cursor = max(0, min(self._cursor, max(self._painted_count() - 1, 0)))
        if self._view == "timeline":
            # A recomposition can change the dated span (a new target date),
            # and the auto tier exists to fit it; an explicit zoom is kept
            # while the span it was chosen for still fits (see `_choose_tier`).
            self._tier = self._choose_tier()
        self._repaint()
        # Reveal the highlighted row, and again once the first layout lands:
        # the app seeds the data before mount, so the body may be zero-sized
        # here. Without this a `show` on a store longer than the viewport left
        # the cursor off-screen while the footer named it (QA Q2 / design D1).
        self._scroll_cursor_into_view()
        self.call_after_refresh(self._scroll_cursor_into_view)

    def focus_project(self, project_id: str) -> None:
        """Move the list cursor to a project (an already-open page, re-shown)."""
        for index, view in enumerate(self._views):
            project = view.get("project") if isinstance(view, dict) else None
            if isinstance(project, dict) and str(project.get("id")) == str(project_id):
                self._view = "list"
                self._cursor = max(0, min(index, max(self._painted_count() - 1, 0)))
                self._repaint()
                self._scroll_cursor_into_view()
                self.call_after_refresh(self._scroll_cursor_into_view)
                return

    def _painted_count(self) -> int:
        """Rows the list canvas actually paints — its own ``PROJECTS_MAX`` cap in.

        Past the cap the canvas stops painting rows and names the overflow in
        ONE truncation row, so a cursor beyond it would rest on a row the
        canvas never draws and the footer would name a project no reader can
        see anywhere (design round 1, D2). The cap is the renderer's own
        constant, imported rather than copied, so the clamp cannot drift from
        what is painted.
        """
        return max(0, min(len(self._views), PROJECTS_MAX))

    def _span_key(self) -> tuple[str, str] | None:
        """The dated span the timeline axis covers, as an ISO pair (or ``None``)."""
        span = timeline_span(self._views)
        if span is None:
            return None
        return (span[0].isoformat(), span[1].isoformat())

    def _choose_tier(self) -> str:
        """The tier to paint: the manual zoom while its span survives, else auto.

        `r` (and any recomposition) must not discard a zoom the reader chose
        when nothing about the span changed — the contradiction agent review
        round 1 (F2) measured: `-` to month, `r`, and the page was back on
        week. A span change re-derives the auto tier, which is the rule the
        zoom comment always claimed.
        """
        if self._tier_manual and self._manual_span == self._span_key():
            return self._tier
        self._tier_manual = False
        self._manual_span = None
        return auto_timeline_tier(self._views)

    @property
    def tracked(self) -> int:
        """How many projects the page was handed."""
        return len(self._views)

    # -- rendering ----------------------------------------------------------
    def _render_view(self) -> RenderResult:
        """The canvas for the CURRENT view type. Named ``_render_view`` and NOT
        ``_render``: ``Widget._render`` is Textual's own hook (it promotes
        ``render()``'s result to a Visual), and a same-named method returning
        our ``RenderResult`` made the widget hand Textual a plain dataclass as
        its visual — every frame raised ``'RenderResult' object has no
        attribute 'render_strips'``. The ``render``/``query``/``visible``
        shadowing rule AGENTS.md states extends to this one.
        """
        resolver = _style_resolver()
        if self._view == "board":
            return render_project_board(
                self._views,
                cursor=self._cursor if self._views else None,
                associated=self._associated,
                style_for=resolver,
            )
        if self._view == "timeline":
            return render_project_timeline(
                self._views,
                tier=self._tier,
                cursor=self._cursor if self._views else None,
                associated=self._associated,
                style_for=resolver,
            )
        return render_project_list(
            self._views,
            cursor=self._cursor if self._views else None,
            associated=self._associated,
            style_for=resolver,
        )

    def _repaint(self) -> None:
        result = self._render_view()
        self._last = result
        self._canvas.update(result.text)
        # Pin the Static to the painted canvas size so the ScrollableContainer's
        # virtual size equals the canvas (Textual scrolls the difference).
        self._canvas.styles.width = result.width
        self._canvas.styles.height = result.height
        self._paint_chrome()
        # Two deferred passes: the first paint runs before (or mid-) layout,
        # when the footer's own box may not carry its final width yet
        # (measured: a 96-cell box was painted as if it were 30, shedding a
        # clause the space holds), and the hint's actionability reads the final
        # scroll geometry. Both are idempotent and neither re-schedules.
        self.call_after_refresh(self._paint_chrome)
        self.call_after_refresh(self._sync_scroll_hint)

    def _paint_chrome(self) -> None:
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        # The nameless entry's set (S3b): whose projects the `◆` markers are,
        # and how many. It SHEDS FIRST when the title cannot hold the whole
        # line (design round 1, D2; round 2 extended the ladder to the
        # timeline's `zoom:` clause, D6): the title Static clips silently —
        # Rich's `overflow="ellipsis"` is inert on it — so the newest clauses
        # yield rather than letting `tracked`/`updated` be cut with no `…`.
        set_clause = f" · this session ({len(self._associated)})" if self._associated else None

        def build_title(*, with_set: bool, with_zoom: bool = True) -> Text:
            title = Text(no_wrap=True, overflow="ellipsis")
            title.append("projects", style=Style(color=theme_mod.semantic_color("fg"), bold=True))
            title.append(f" · {self._view}", style=muted)
            if with_set and set_clause:
                title.append(set_clause, style=muted)
            tracked = len(self._views)
            title.append(f" · {tracked} tracked", style=dim)
            if with_zoom and self._view == "timeline":
                # The tier is always stated so an auto-chosen axis explains
                # itself (the org chart's tier-title rule) — until the row
                # cannot hold it, which was the pre-existing 60-col clip D6.
                title.append(f" · zoom: {self._tier}", style=dim)
            if self._updated_at is not None:
                import time as _time

                stamp = _time.strftime("%H:%M", _time.localtime(self._updated_at))
                title.append(f" · updated {stamp}", style=dim)
            return title

        from rich.cells import cell_len

        title = build_title(with_set=True)
        available = self._title.content_size.width or self._title.size.width
        if available and cell_len(title.plain) > available:
            if set_clause is not None:
                title = build_title(with_set=False)
            if cell_len(title.plain) > available and self._view == "timeline":
                title = build_title(with_set=False, with_zoom=False)
        self._title.update(title)

        width = max(self.size.width - 2, 1)
        self._rule.update(Text("─" * width, style=dim))

        if self._view == "list" and self._views:
            index = max(0, min(self._cursor, max(self._painted_count() - 1, 0)))
            # The footer fits the DETAIL box's own measured width, not the
            # page's minus its padding: the two differ by the box's position in
            # the layout, and fitting to the smaller number skipped a rung the
            # space could hold (measured: at 100x30 the box is 96 cells, the
            # page arithmetic says 94, and a 97-cell rung was shed that fits).
            footer_width = self._detail.size.width or width
            self._detail.update(
                detail_footer(self._views[index], style_for=_style_resolver(), width=footer_width)
            )
        else:
            tier = self._tier if self._view == "timeline" else None
            self._detail.update(
                aggregate_footer(self._views, tier=tier, style_for=_style_resolver())
            )
        self._paint_hints()

    def _paint_hints(self) -> None:
        """Lay out the footer hints, shedding WHOLE hints until the row fits.

        The org-chart ladder, same rule: each rung is measured before it is
        committed and ``esc`` is never dropped because it is the only way out.
        What sheds first is the order's own statement: ``+/-`` (timeline only)
        and ``r`` go before the view triplet, and ``↔↕ scroll`` — a gesture a
        reader finds by trying an arrow — goes before ``3 timeline``/``v next``,
        so the newest view types stay advertised on a narrow terminal (UX
        round 1, U3).
        """

        def rung(
            leads: list[tuple[HintButton, str, bool]],
            esc_label: str,
            *,
            state: bool,
        ) -> tuple[list[tuple[HintButton, str, bool]], str]:
            row = list(leads)
            row.append((self._exit_hint, esc_label, bool(row)))
            if state:
                row.append((self._state_hint, "", True))
            return (row, esc_label)

        scroll = (self._scroll_hint, " scroll", False)
        list_hint = (self._list_hint, " list", True)
        board_hint = (self._board_hint, " board", True)
        timeline_hint = (self._timeline_hint, " timeline", True)
        refresh = (self._refresh_hint, " refresh", True)
        open_hint = (self._open_hint, " open", True)
        nxt = (self._next_hint, " next", True)
        # `+/-` is TIME zoom: it acts only on the timeline, and a hinted key
        # that changes nothing is worse than an absent one (the org chart's own
        # rule for its zoom hint) — so the button is advertised where it works
        # and dropped from every rung elsewhere.
        zoom: tuple[HintButton, str, bool] | None = (
            (self._zoom_hint, " zoom", True) if self._view == "timeline" else None
        )

        def leads_of(
            *leads: tuple[HintButton, str, bool] | None,
        ) -> list[tuple[HintButton, str, bool]]:
            # The seam (` · `) belongs to the ROW: re-derive it here so the
            # first hint in a rung never paints a leading separator. With
            # `↔↕ scroll` shed the row used to open with a dangling `·`
            # (UX round 2, U6), and measuring the same plan the row will be
            # painted from keeps the ladder's arithmetic honest.
            return [
                (hint, label, index > 0)
                for index, (hint, label, _lead) in enumerate(
                    lead for lead in leads if lead is not None
                )
            ]

        all_leads = leads_of(
            scroll, list_hint, board_hint, timeline_hint, nxt, refresh, open_hint, zoom
        )
        rungs: list[tuple[list[tuple[HintButton, str, bool]], str]] = [
            rung(all_leads, "back to conversation", state=True),
            rung(all_leads, "back to conversation", state=False),
            rung(all_leads, "back", state=False),
            rung(
                leads_of(scroll, list_hint, board_hint, timeline_hint, nxt, refresh, open_hint),
                "back",
                state=False,
            ),
            # `↵ open` sheds HERE, before `r refresh` in the RUNG ORDER. The
            # measured boundaries (terminal columns, this harness): `scroll`
            # returns at 72, `refresh` at 85, `open` at 95; at 60 all three are
            # absent and the row is `1 list · 2 board · 3 timeline · v next ·
            # esc back` (review round 2 R2-2a corrected the earlier claim that
            # 60 keeps the refresher).
            rung(
                leads_of(scroll, list_hint, board_hint, timeline_hint, nxt, refresh),
                "back",
                state=False,
            ),
            rung(
                leads_of(scroll, list_hint, board_hint, timeline_hint, nxt),
                "back",
                state=False,
            ),
            # `↔↕ scroll` sheds HERE, before any view key: a view the reader
            # cannot discover is worse than a gesture they will try anyway.
            rung(leads_of(list_hint, board_hint, timeline_hint, nxt), "back", state=False),
            # …and the esc LABEL sheds before a view key too (the key itself
            # never drops): at 60 columns `1/2/3 · v · esc` all fit as bare
            # keys, and the dock still spells the full sentence
            # (`Read-only · esc back`), so nothing is lost that a reader needs
            # to leave the page (UX round 1, U3).
            rung(leads_of(list_hint, board_hint, timeline_hint, nxt), "", state=False),
            rung(leads_of(list_hint, board_hint, timeline_hint), "", state=False),
            rung(leads_of(list_hint, board_hint), "", state=False),
            rung(leads_of(list_hint), "", state=False),
            rung(leads_of(), "", state=False),
        ]
        width = max(self.size.width - 2, 1)
        chosen = rungs[-1]
        for leads, esc_label in rungs:
            if self._measure_hints(leads, esc_label) <= width:
                chosen = (leads, esc_label)
                break
        plan, esc_label = chosen
        visible = {hint for hint, _label, _lead in plan}
        for hint, label, lead in plan:
            hint.paint(esc_label if hint is self._exit_hint else label, lead=lead)
        for hint in (
            self._scroll_hint,
            self._list_hint,
            self._board_hint,
            self._timeline_hint,
            self._next_hint,
            self._refresh_hint,
            self._open_hint,
            self._zoom_hint,
            self._exit_hint,
            self._state_hint,
        ):
            hint.display = hint in visible
        # Arm the scroll hint against the geometry just painted; the deferred
        # pass in `_repaint` re-arms it once the layout has settled.
        self._sync_scroll_hint()

    def _measure_hints(self, plan: list[tuple[HintButton, str, bool]], esc_label: str) -> int:
        """Cell width of a candidate hint row, measured before it is painted."""
        from rich.cells import cell_len

        row = Text()
        for hint, label, lead in plan:
            row.append(hint.preview(esc_label if hint is self._exit_hint else label, lead=lead))
        return cell_len(row.plain)

    # -- lifecycle ----------------------------------------------------------
    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        yield self._body
        yield self._detail
        with self._hints:
            yield self._scroll_hint
            yield self._list_hint
            yield self._board_hint
            yield self._timeline_hint
            yield self._next_hint
            yield self._refresh_hint
            yield self._open_hint
            yield self._zoom_hint
            yield self._exit_hint
            yield self._state_hint

    def on_mount(self) -> None:
        # Focus lands here rather than at the app's open call: focus() on a
        # widget not yet in the focus chain is a silent no-op (the subagent
        # view's recorded bug), and the advertised keys would go to the inert
        # composer. Repaint after focus so the first frame is the settled one.
        self._repaint()
        try:
            self.focus()
        except Exception:
            pass

    def on_resize(self) -> None:
        # The rule spans the page and the hints shed against a width only the
        # layout knows, so both repaint on resize. The canvas is
        # width-independent (it scrolls), so only the chrome moves.
        self._paint_chrome()
        self.call_after_refresh(self._sync_scroll_hint)

    def _sync_scroll_hint(self) -> None:
        """Arm ``↔↕ scroll`` only while the body has somewhere to scroll.

        ``HintButton.set_actionable`` exists for exactly this (its docstring
        calls the alternative "the reported 'nothing happens when I click' bug
        one step earlier"), and the sibling subagent page drives it the same
        way. Cheap and idempotent, so it rides the chrome paint plus one
        deferred pass for the post-layout geometry (agent review round 1,
        finding 8: the hint lit on hover on a canvas that could not scroll).
        """
        self._scroll_hint.set_actionable(self._body.max_scroll_x > 0 or self._body.max_scroll_y > 0)
        # `↵ open` needs an object: on an empty store there is nothing to open
        # and the hint stops offering itself (the same rule as the arrows).
        self._open_hint.set_actionable(bool(self._views))

    # -- geometry probes (for tests / visual validation) --------------------
    @property
    def canvas_size(self) -> tuple[int, int]:
        """The painted canvas (width, height) in cells — the body's virtual."""
        if self._last is None:
            return (0, 0)
        return (self._last.width, self._last.height)

    @property
    def last_result(self) -> RenderResult | None:
        return self._last

    @property
    def view_type(self) -> str:
        return self._view

    @property
    def tier(self) -> str:
        return self._tier

    @property
    def cursor(self) -> int:
        return self._cursor

    def rendered_rows(self) -> list[str]:
        """The page as plain strings — title, rule, canvas rows, footer. Assertable.

        The FOOTER rides last: its fit (U1) and its ``missing`` badge (F5) are
        read here by tests rather than guessed from a frame.
        """

        def plain(widget: Static) -> str:
            # ``Static.render()`` returns a Visual, NOT the renderable it was
            # updated with; the original content lives on ``content``. Read the
            # plain form from whichever candidate carries it, so this helper
            # works however Textual wires the two.
            for candidate in (getattr(widget, "content", None), widget.render()):
                text = getattr(candidate, "plain", None)
                if text is not None:
                    return str(text)
            return ""

        rows = [plain(self._title), plain(self._rule)]
        if self._last is not None:
            rows.extend(text.plain for text in self._last.text.split("\n"))
        rows.append(plain(self._detail))
        return rows

    # -- view switching -----------------------------------------------------
    def _set_view(self, view: str) -> None:
        if view not in VIEWS or view == self._view:
            return
        self._view = view
        if view == "timeline":
            # An auto tier per composition: the axis exists to fit the data it
            # was opened on, and a manual zoom survives until the span changes
            # (`_choose_tier`).
            self._tier = self._choose_tier()
        # The canvas is a different shape now; start the reader at its origin
        # rather than at a scroll offset computed for the previous canvas —
        # then reveal the SELECTION, so the canvas opens on the card/row `↵`
        # would act on (S3b; with the default cursor at row 0 both land on the
        # origin, as before). The reveal runs NOW and again once the new
        # layout lands: the limits read on the first call still belong to the
        # canvas being left, and nothing re-ran it before (QA round 1, Q1 — a
        # switch left the selection off-screen at every size).
        self._body.scroll_to(x=0, y=0, animate=False)
        self._repaint()
        self._scroll_cursor_into_view()
        self.call_after_refresh(self._scroll_cursor_into_view)

    def action_show_list(self) -> None:
        self._set_view("list")

    def action_show_board(self) -> None:
        self._set_view("board")

    def action_show_timeline(self) -> None:
        self._set_view("timeline")

    def action_cycle_view(self) -> None:
        self._set_view(VIEWS[(VIEWS.index(self._view) + 1) % len(VIEWS)])

    def action_jump(self) -> None:
        """``↵``: ask the host to open the selected project's conversation.

        The selection is the page's ONE cursor — the row the list paints `▸`
        on and the card/row the other canvases mark — so the key means the
        same thing in every view (S3b). The message carries each linked
        session with its state, and the HOST decides: a live one is switched
        to through the existing session machinery, and anything less is named
        honestly. The page does not guess, and never opens a session itself.
        """
        if not (0 <= self._cursor < len(self._views)):
            return
        view_row = self._views[self._cursor]
        project_value = view_row.get("project") if isinstance(view_row, dict) else None
        project = project_value if isinstance(project_value, dict) else {}
        sessions: list[tuple[str, str]] = []
        rows = view_row.get("sessions")
        for session_row in rows if isinstance(rows, list) else []:
            if not isinstance(session_row, dict):
                continue
            session_id = str(session_row.get("session_id") or "")
            if not session_id:
                continue
            if session_row.get("exists") is False:
                # The RECEIPT's word, deliberately: a link whose directory is
                # gone reads `missing` on every surface, so `↵` must not say
                # `[stopped]` about the same link (QA round 1, Q2 — the
                # composed row carries state `stopped` for a record-less link,
                # which made the old branch unreachable).
                sessions.append((session_id, "missing"))
                continue
            runtime_value = session_row.get("runtime")
            runtime = runtime_value if isinstance(runtime_value, dict) else {}
            sessions.append((session_id, str(runtime.get("state") or "stopped")))
        self.post_message(
            ProjectsViewJumpRequested(
                project_id=str(project.get("id") or ""),
                project_name=str(project.get("name") or "(unnamed)"),
                sessions=tuple(sessions),
            )
        )

    def action_refresh(self) -> None:
        self.post_message(ProjectsViewRefreshRequested())

    # -- zoom (timeline only) ----------------------------------------------
    def _set_tier(self, tier: str) -> None:
        if tier not in TIMELINE_TIERS or tier == self._tier:
            return
        self._tier = tier
        # A hand-chosen tier, remembered with the span it was chosen for: it
        # survives a recomposition that leaves the span alone, and yields to
        # the auto tier when the span changes (agent review round 1, F2).
        self._tier_manual = True
        self._manual_span = self._span_key()
        if self._view == "timeline":
            self._body.scroll_to(x=0, y=0, animate=False)
            self._repaint()

    def action_zoom_in(self) -> None:
        # "In" = MORE detail = a finer unit (the org chart's rule: zoom is
        # level of detail). Inert off the timeline: the footer does not offer
        # it there, and a key that changes nothing is worse than absent.
        if self._view != "timeline":
            return
        index = TIMELINE_TIERS.index(self._tier)
        self._set_tier(TIMELINE_TIERS[max(0, index - 1)])

    def action_zoom_out(self) -> None:
        if self._view != "timeline":
            return
        index = TIMELINE_TIERS.index(self._tier)
        self._set_tier(TIMELINE_TIERS[min(len(TIMELINE_TIERS) - 1, index + 1)])

    def _cycle_tier(self) -> None:
        # The +/- hint click steps finer and wraps, the way the org chart's
        # sole zoom hint does (a single click target cannot mean both).
        if self._view != "timeline":
            return
        index = TIMELINE_TIERS.index(self._tier)
        self._set_tier(TIMELINE_TIERS[(index + 1) % len(TIMELINE_TIERS)])

    # -- movement and scrolling (all CLAMP; no wrap on a canvas) ------------
    def action_up(self) -> None:
        # ↑/↓ move the ONE selection in every canvas (UX round 1, U2): a rung
        # that says `↵ open` needs an object the reader can choose, and before
        # this the board/timeline arrows only panned — `↵` stayed aimed at
        # whatever the list cursor happened to be, on a card often off-screen.
        # The canvas follows the selection through the reveal; ←/→ and the
        # page keys still pan.
        self._move(-1)

    def action_down(self) -> None:
        self._move(1)

    def current_project_id(self) -> str | None:
        """The selected project's id, or ``None`` (the host seeds from it)."""
        if not (0 <= self._cursor < len(self._views)):
            return None
        view_row = self._views[self._cursor]
        project = view_row.get("project") if isinstance(view_row, dict) else None
        if not isinstance(project, dict):
            return None
        return str(project.get("id") or "") or None

    def _move(self, delta: int) -> None:
        """Move the list cursor, CLAMPED, then reveal it (reveal-then-act)."""
        if not self._views:
            return
        position = max(0, min(self._cursor + delta, max(self._painted_count() - 1, 0)))
        if position == self._cursor:
            return
        self._cursor = position
        self._repaint()
        self._scroll_cursor_into_view()

    def _usable_height(self) -> int:
        """Rows the body can actually show — the h-scrollbar's row excluded.

        ``max_scroll_y`` is computed as ``virtual - (container - bar)``, so the
        reveal has to use the SAME arithmetic: on a canvas wider than the
        viewport the horizontal bar paints over the content region's last row,
        and a reveal that counted it parked the selected row underneath the bar
        — invisible to the keyboard (design round 1, D1).
        """
        return max(
            0,
            self._body.container_size.height - self._body.scrollbar_size_horizontal,
        )

    def _usable_width(self) -> int:
        """Columns the body can show — the v-scrollbar's column excluded.

        The arithmetic ``_usable_height`` states, on the other axis: a reveal
        that counts the vertical scrollbar's column can park a card under it.
        """
        return max(
            0,
            self._body.container_size.width - self._body.scrollbar_size_vertical,
        )

    def _position_for(self, index: int) -> tuple[int, int] | None:
        """Canvas ``(x, y)`` of ``views[index]`` in the CURRENT view, or None.

        The board and the timeline answer from their own renderers' position
        helpers — one source for the painter's geometry and the reveal's, so
        the two cannot disagree about where a card or row is; the list reveals
        by row index and answers ``None`` here.
        """
        if self._view == "board":
            return board_position(self._views, index)
        if self._view == "timeline":
            return timeline_position(self._views, index)
        return None

    def _scroll_cursor_into_view(self) -> None:
        """Keep the SELECTION inside the scrolled viewport, in every view.

        The body is a ScrollableContainer around ONE painted Static, so there
        is no child widget to call ``scroll_visible`` on — the offset is
        computed from the selection's own coordinates. The list reveals by row
        index; the board and the timeline read their renderers' position
        helpers and CLAMP the selection back to the last painted project
        first, so `↵` never asks about a card or row no reader can see (design
        round 1, D2's rule, applied to the other canvases; S3b). Guarded
        because the container has no size until it is laid out.
        """
        height = self._usable_height()
        width = self._usable_width()
        if height <= 0 or width <= 0:
            return
        offset_x = self._body.scroll_offset.x
        offset_y = self._body.scroll_offset.y
        if self._view == "list":
            x: int | None = None  # the list scrolls vertically only
            y = self._cursor
        else:
            position = self._position_for(self._cursor)
            if position is None:
                clamped = False
                for candidate in range(self._cursor - 1, -1, -1):
                    if self._position_for(candidate) is not None:
                        self._cursor = candidate
                        clamped = True
                        position = self._position_for(candidate)
                        break
                if clamped:
                    # The clamp MOVED the reader's selection: repaint, or the
                    # clamped card/row shows no `▸` until something else
                    # repaints (agent review round 1, MINOR 3).
                    self._repaint()
            if position is None:
                return
            x, y = position

        def window(offset: int, size: int, span: int) -> int:
            """The offset that brings ``span`` inside ``[offset, offset+size)``."""
            if span < offset:
                return span
            if span >= offset + size:
                return span - size + 1
            return offset

        target_y = max(0, min(window(offset_y, height, y), self._body.max_scroll_y))
        target_x = (
            offset_x
            if x is None
            else max(0, min(window(offset_x, width, x), self._body.max_scroll_x))
        )
        if (target_x, target_y) == (offset_x, offset_y):
            return
        self._body.scroll_to(x=target_x, y=target_y, animate=False)

    def action_scroll_left(self) -> None:
        self._body.scroll_left()

    def action_scroll_right(self) -> None:
        self._body.scroll_right()

    def action_page_up(self) -> None:
        if self._view == "list":
            self._move(-max(1, self._usable_height()))
            return
        self._body.scroll_page_up()

    def action_page_down(self) -> None:
        if self._view == "list":
            self._move(max(1, self._usable_height()))
            return
        self._body.scroll_page_down()

    def action_page_left(self) -> None:
        self._body.scroll_page_left()

    def action_page_right(self) -> None:
        self._body.scroll_page_right()

    def action_scroll_home(self) -> None:
        if self._view == "list":
            self._move(-self._cursor)
            return
        # Top-left corner, both axes pinned (Textual's scroll_home resets only
        # the Y axis unless x is passed).
        self._body.scroll_to(x=0, y=0, animate=False)

    def action_scroll_end(self) -> None:
        if self._view == "list":
            # The cursor stops at the last PAINTED row (see `_painted_count`).
            self._move(max(0, self._painted_count() - 1 - self._cursor))
            return
        # Bottom-RIGHT, the org chart's explicit maxima: ``scroll_end`` reaches
        # the bottom-LEFT on this Textual, and the wide axis is the one a
        # timeline overflows.
        self._body.scroll_to(
            x=self._body.max_scroll_x,
            y=self._body.max_scroll_y,
            animate=False,
        )

    def _focus_canvas(self) -> None:
        """Focus the view so the arrow/scroll keys land here (hint click)."""
        self.focus()

    # -- leaving ------------------------------------------------------------
    def action_leave(self) -> None:
        self._leave()

    def _leave(self) -> None:
        self.post_message(ProjectsViewDismissed())
