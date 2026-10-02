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
    detail_progress_line,
    detail_ruler,
    detail_session_state,
    display_name,
    list_position,
    project_at,
    render_project_board,
    render_project_list,
    render_project_timeline,
    section_header_at,
    section_ruler,
    sections_of,
    status_chip_text,
    timeline_position,
    timeline_span,
)
from local_operator.tui.widgets.projects_detail import ProjectDetailPage
from local_operator.tui.widgets.projects_form import FORM_FOOTER_HINT, ProjectsFormPage
from local_operator.tui.widgets.projects_send import (
    SendTarget,
    SendTargetCard,
    send_targets,
)
from local_operator.tui.widgets.projects_start import (
    PLAIN_DETAIL,
    PLAIN_LABEL,
    StartPickerCard,
    StartTarget,
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


class ProjectsViewFormSubmitted(Message):
    """The create form was submitted with values that passed local validation.

    The page never writes (module docstring): the app performs the write
    through the same ``registry.create_project`` core the ``project`` tool and
    the slash verbs use, then recomposes and hands the page fresh data. A
    refusal from the STORE (a taken name, the schema guard) comes back as an
    in-form line, never a toast over a form that has already closed (spec
    §7.7).
    """

    def __init__(self, *, edit: Any) -> None:
        super().__init__()
        self.edit = edit


class ProjectsViewSendRequested(Message):
    """Send this text to this target (spec §7.5.3).

    The page never delivers (module docstring): the app resolves the target and
    calls the same core ``send`` does, then reports the honest outcome back. A
    refusal the page can decide itself (an empty body) never leaves the page.
    """

    def __init__(self, *, target: SendTarget, text: str) -> None:
        super().__init__()
        self.target = target
        self.text = text


class ProjectsViewComposeChanged(Message):
    """Compose mode opened (``target``) or closed (``None``).

    The composer belongs to the APP — the dock is not the page's — so the page
    cannot enter compose by itself, and it must say when it leaves so the
    composer can go back to being read-only. ``target`` is the chosen row.
    """

    def __init__(self, *, target: SendTarget | None) -> None:
        super().__init__()
        self.target = target


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
        # D4: per-status chips — one mapping, read by the list rows, the board
        # cards and the footer chip. `active` keeps the accent every other
        # "running" chip uses; `paused` warns; `done`/`archived` recede. The
        # lifecycle additions and the measured constraints behind them
        # (design review round 1, D2/D3; ΔE00 across all 54 registered
        # themes):
        #   * `planning` recedes with `done` — the ends of the arc are quiet —
        #     and the two share `muted` (ΔE00 0.00 in every theme). That reuse
        #     is deliberate and is carried by SHAPE: the chips render through
        #     `projects_render.status_chip_text`, whose per-status glyphs
        #     (`○` vs `✓`) keep every pair apart in a colourless frame — the
        #     same shape-only contract the tool-status family (`✓`/`◐`/`✗`/`⊘`)
        #     already ships.
        #   * `qa` keeps the running accent (ΔE00 0.00 against `active`),
        #     the same statement one phase further along; `◐` vs `●`
        #     separates them.
        #   * `validation` deliberately does NOT take `success`: the theme's
        #     two greens measured ΔE00 2.23 in `light` (dark 5.07) — the
        #     accidental-duplicate class the sibling fix rejected at 2.22 —
        #     and the available re-inks that keep meaning are scarcer than the
        #     bar; the deployed-and-proving state takes the `fg` focus ink
        #     instead (worst pair 4.44, against the quiet `muted` pair; every
        #     other pair ≥9.06), with `◉` — the chip you watch.
        "status_active": Style(color=color("accent")),
        "status_paused": Style(color=color("warning")),
        "status_done": Style(color=color("muted")),
        "status_archived": Style(color=color("dim")),
        "status_planning": Style(color=color("muted")),
        "status_qa": Style(color=color("accent")),
        "status_validation": Style(color=color("fg")),
        # The quick-send card's `live` chip (P5a, design review round 2, D8):
        # the card's chips sit on `overlay`, a ground this ramp's accent was
        # never solved against — see the `chip-live` token for the measurement
        # and why only the light ramp needs the solve.
        "chip_live": Style(color=color("chip-live")),
        "cursor": Style(color=color("accent"), bold=True),
        # The selected-row band on the quick-send card (P5a, design review
        # round 1, D4): `tint-select` is the app's selection ground — the
        # sidebar, the pickers, `ToolCard:focus` — and the card's selected row
        # takes it verbatim, so "this is the row you are on" is said one way
        # everywhere. Ground only; the row's own spans carry the foreground.
        "row_selected": Style(bgcolor=color("tint-select")),
        # The cells a FILTER matched (design review round 1, D2). A typographic
        # signal rather than an ink: it inherits the run's own foreground, so it
        # cannot fall under the contrast floor on either theme (the accent
        # measures 3.49:1 on the card's light ground) and it survives a
        # colourless terminal, which is the whole point of marking a match.
        "match": Style(bold=True, underline=True),
        # The session's own projects carry `◆` in the row's leading column
        # (S3b): the accent without the cursor's bold, so `▸` still owns the
        # glyph where both apply — the marker column costs no width either way.
        "session": Style(color=color("accent")),
        "dim": Style(color=color("dim")),
        # The quiet ink a LABEL wears — the form's field names. It is the same
        # token `status_planning`/`status_done` take; it is named separately
        # because `resolve` answers an unknown key with PLAIN ink rather than an
        # error, and a form asking for a key the map does not carry would paint
        # its labels in the terminal's default colour without saying so.
        "muted": Style(color=color("muted")),
        # The ink a REFUSAL wears. The form paints its in-field errors with it:
        # guidance (`muted`) and a refusal must never read alike (design review
        # round 1, D4), and the token is the same `warning` the app's own
        # refusal receipts (`_notice_text`) take.
        "refusal": Style(color=color("warning")),
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


class ProjectsViewStartRequested(Message):
    """The start picker chose a row: boot a session for this project (P5b).

    ONE message for one outcome, and it names BOTH things the create needs:
    the project the session is auto-linked to (and whose snapshot the kickoff
    prompt quotes) and the target the row picked. The page cannot answer
    either on its own — the registries and the creation core are the app's —
    so it hands the whole question over rather than half of it.
    """

    def __init__(self, *, project_id: str, target: StartTarget) -> None:
        super().__init__()
        self.project_id = project_id
        self.target = target


class ProjectsViewMilestoneToggled(Message):
    """``↵`` on a milestone row: flip its completion through the store.

    The page never writes (module docstring): the app calls the same
    ``registry.set_milestone`` core the tool and the API use, recomposes,
    and hands the page fresh data — a refusal lands as a notice, never a
    silent no-op (spec §10.5).
    """

    def __init__(
        self, project_id: str, name: str, completed: bool, *, project_name: str = ""
    ) -> None:
        super().__init__()
        self.project_id = project_id
        self.name = name
        self.completed = completed
        self.project_name = project_name


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


class ProjectsViewAttachmentOpened(Message):
    """``↵`` on an attachment row: hand the copied file to the OS (spec §7.4).

    Carries the path and the display name so the host can open it and say what
    happened — the page never spawns a process, exactly as it never touches
    session machinery. A path that is gone is answered by the host with the
    honest sentence rather than a silent no-op.
    """

    def __init__(self, *, path: str, name: str, project_name: str) -> None:
        super().__init__()
        self.path = path
        self.name = name
        self.project_name = project_name


class ProjectsViewAttachmentCopied(Message):
    """`y` on an attachment row: put its stored path on the clipboard (§7.4).

    Carries the row's path and display name so the host can write the
    clipboard and say what it did. The PATH is the payload, not the file's
    contents: the row is an affordance for the copy under the project store,
    and that is what a reader pastes into a shell, an editor or a bug report.
    """

    def __init__(self, *, path: str, name: str, project_name: str) -> None:
        super().__init__()
        self.path = path
        self.name = name
        self.project_name = project_name


class ProjectsViewAttachmentPreviewRequested(Message):
    """`space` on an image attachment: read the file so the page can show it.

    The page is I/O-free, so it cannot read the copy itself; this asks the
    host for the bytes and the host answers through
    :meth:`ProjectsView.show_attachment_preview`. The message carries the
    display name so a read that FAILS can name the file in its sentence — a
    refusal is never a silent no-op (the opener's own rule).
    """

    def __init__(self, *, path: str, name: str, project_name: str) -> None:
        super().__init__()
        self.path = path
        self.name = name
        self.project_name = project_name


class ProjectsView(Vertical):
    """The page: a title, a rule, the scrollable canvas, the detail footer, hints.

    Class-identified (see the module docstring). ``can_focus`` so the view
    keys, the list cursor and the canvas scrolling land here rather than on the
    composer the mode made inert.
    """

    can_focus = True

    #: The actions that act on the CANVAS or on the project list. While the
    #: FORM owns the page they must not fire from a key that bubbled past the
    #: focused field: `↑`/`↓` are not an ``Input``'s own keys, so they reach
    #: this view, and the canvas must not move under a form being filled in.
    #: ``open_detail`` is in the set for the same reason (`d` is swallowed by a
    #: field anyway, but a click on the canvas is not the only way in) and
    #: ``esc`` is deliberately NOT — it is the form's own way out.
    _CANVAS_ACTIONS = frozenset(
        {
            "show_list",
            "show_board",
            "show_timeline",
            "cycle_view",
            "open_detail",
            "refresh",
            "zoom_in",
            "zoom_out",
            "up",
            "down",
            "scroll_left",
            "scroll_right",
            "page_up",
            "page_down",
            "page_left",
            "page_right",
            "scroll_home",
            "scroll_end",
            "jump",
            "next_section",
            "prev_section",
        }
    )

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
        # `d` opens the DETAIL page for the selection (S6d parity P2). The
        # recorded fallback of the spec's D1 — `↵` keeps its shipped meaning
        # (open the conversation), and the detail takes the free letter.
        Binding("d", "open_detail", "Detail", show=False),
        # `c` opens the CREATE form (P4) — the spec's own key for it, and one
        # of the letters the canvases left free.
        Binding("c", "create", "Create", show=False),
        # `m` messages a linked session (P5a): on a session row it sends
        # straight to it, anywhere else it opens the target picker (spec
        # §7.5.1). Free in the mode and in the app's focused chain.
        Binding("m", "message", "Message", show=False),
        # `s` starts a session for the selected/current project (P5b) — the
        # spec's own key for it (§3.2), free in the mode and in the app's
        # focused chain. It opens the team/agent picker; the boot itself is the
        # APP's (the registries and the creation core are not the page's).
        Binding("s", "start", "Start session", show=False),
        # `y` puts the selected attachment's STORED path on the clipboard (spec
        # §3.2/§7.4) — the one thing a reader can act on when the file is gone
        # or when the opener is what they want to avoid. Row-scoped like `m` on
        # a session row: it acts where it applies and is inert elsewhere.
        Binding("y", "copy_attachment", "Copy path", show=False),
        # `space` shows an image attachment's pixels inline and hides them
        # again (spec §7.4's deliberately staged half, using the image stack the
        # transcript already ships). Unbound in this mode and in the app's
        # focused chain; the row-scoped rule above applies.
        Binding("space", "preview_attachment", "Preview", show=False),
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
        # Section jumps (S6d parity): the selection moves to the next/previous
        # team section's first row — clamped at the ends and inert while the
        # canvas is ungrouped (one binding list; the actions no-op where they
        # do not apply, the zoom pattern).
        Binding("shift+down", "next_section", "Next section", show=False),
        Binding("shift+up", "prev_section", "Previous section", show=False),
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
        #: ``canvas`` | ``detail`` — where the page is (spec §10.1's state
        #: machine). The canvas keeps the shipped mechanics while the detail
        #: owns its own body; `esc` pops ONE level and only the canvas exits.
        self._mode: str = "canvas"
        #: The calling process's own session id, when the host knows it — the
        #: `◆` marker and `own session` label on the detail page. Sticky: a
        #: refresh load may omit it without clearing it.
        self._own_session: str | None = None
        #: The project the detail state is showing, by id — the anchor a
        #: recomposition re-finds (a rename can move its index).
        self._detail_project_id: str | None = None
        self._tier: str = "month"
        #: Whether the tier above was chosen by hand (``+``/``-``) and, if so,
        #: the dated span it was chosen for. A manual zoom survives a
        #: recomposition while that span is unchanged; a changed span
        #: re-derives the auto tier (the rule the zoom comments state, now
        #: actually implemented — agent review round 1, F2).
        self._tier_manual: bool = False
        self._manual_span: tuple[str, str] | None = None
        self._updated_at: float | None = None
        #: The footer's one-sentence notice (refusals, pops) — UX round 1.
        self._notice: str | None = None
        # Quick-send state (P5a): the open picker card and the target compose
        # is addressed to.
        self._send_card: SendTargetCard | None = None
        self._send_target: SendTarget | None = None
        # The manager row, when the host resolved one (P5a). Injected rather
        # than derived: only the app can read the registry, and a page that
        # guessed would paint a row nobody answers to.
        self._manager_target: SendTarget | None = None
        # Start-session state (P5b): the open picker card, and the ROWS it
        # offers — injected by the app from the same team/agent catalogues the
        # desktop pane's new-chat picker reads, because no registry is
        # reachable from here. The seed is the honest one for a host that never
        # injects: the plain row, which is the whole feature on an install with
        # no registries at all (spec D7).
        self._start_card: StartPickerCard | None = None
        #: Whether the reader dismissed the card while its create was running
        #: (agent review round 1, F5): one start's worth of state, consumed by
        #: the hand-off.
        self._start_cancelled = False
        self._start_rows: list[StartTarget] = [
            StartTarget(kind="plain", name="", label=PLAIN_LABEL)
        ]
        #: Last render, kept for the geometry probes and rendered_rows().
        self._last: RenderResult | None = None
        self._title = Static(classes="projects-view-title")
        self._rule = Static(classes="projects-view-rule")
        # The canvas is a Static inside a BOTH-AXES scroll container: the
        # Static is sized to the painted canvas and the container clips and
        # scrolls it, so Textual's own scrollbars appear exactly when over.
        self._canvas = Static(classes="projects-view-canvas")
        self._body = ScrollableContainer(self._canvas, classes="projects-view-body")
        self._detail_page = ProjectDetailPage(
            self._detail_row_action,
            _style_resolver(),
            on_nav=self._detail_nav,
            # A toggle changes the selected row's verb, which is the hint row's
            # own input (UX review round 1, U2).
            on_state_change=self._detail_state_changed,
        )
        self._detail_page.display = False
        # The FORM state (P4): the create page. `c` opens it, `esc` pops one
        # level, and its `ctrl+s` is the app's `resume` hotkey shadowed while
        # the form owns the page — a full-page form owns its keys, which is
        # why the footer advertises it (spec §7.7).
        self._form_page = ProjectsFormPage(
            on_submit=self._form_submitted,
            on_cancel=self.close_form,
            on_state_change=self._form_state_changed,
            style_for=_style_resolver(),
        )
        self._form_page.display = False
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
        self._detail_hint = HintButton("d", lambda: self.action_open_detail())
        self._move_hint = HintButton("↑↓", self._focus_detail_page)
        self._page_hint = HintButton("pgup/pgdn", lambda: self._detail_page.scroll_page_down())
        self._list_hint = HintButton("1", lambda: self.action_show_list())
        self._board_hint = HintButton("2", lambda: self.action_show_board())
        self._timeline_hint = HintButton("3", lambda: self.action_show_timeline())
        self._next_hint = HintButton("v", lambda: self.action_cycle_view())
        self._refresh_hint = HintButton("r", lambda: self.action_refresh())
        # The canvas's headline action: a reader who cannot see how to make a
        # project cannot use the page at all (UX round 1, U1).
        self._create_hint = HintButton("c", lambda: self.action_create())
        # `m` message (P5a). It MUST be in `_hint_buttons()` as well as in a
        # rung: that list is what the painter hides, and a button absent from
        # it stays visible at its DOM slot, ahead of the rung it belongs to.
        self._msg_hint = HintButton("m", lambda: self.action_message())
        # `s` start (P5b). In `_hint_buttons()` for the same reason `m` is:
        # that list is what the painter hides, and a button absent from it stays
        # visible at its DOM slot, ahead of the rung it belongs to.
        self._start_hint = HintButton("s", lambda: self.action_start())
        # `y` copy path / `space` preview (P6, spec §7.4). Row-scoped: the rungs
        # carry them only while an attachment row is selected, and they are in
        # `_hint_buttons()` so a state change HIDES them rather than leaving
        # them parked at their DOM slot ahead of the rung they belong to.
        self._copy_hint = HintButton("y", lambda: self.action_copy_attachment())
        self._preview_hint = HintButton("space", lambda: self.action_preview_attachment())
        self._zoom_hint = HintButton("+/-", self._cycle_tier)
        self._exit_hint = HintButton("esc", self._leave_or_pop)
        self._tab_hint = HintButton("tab", self._form_focus_next)
        self._save_hint = HintButton("ctrl+s", self._form_save)
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
        own_session: str | None = None,
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
        never moves the reader. ``own_session`` is THIS process's session id,
        used by the detail page's `◆`/`own session` labels; sticky, so a
        refresh load may omit it.
        """
        # A recompose answers whatever the last notice said; `_resync_detail`
        # below may set its own (the vanished-project pop, UX round 1, U5).
        self._notice = None
        self._views = list(views)
        if own_session is not None:
            self._own_session = own_session
        if view is not None and self._mode == "form":
            # The same rule the detail obeys: an EXPLICIT canvas request
            # outranks the open form. A plain recomposition (no `view`) leaves
            # the form — and the draft in it — exactly as it is.
            self._exit_form()
        if view is not None and self._mode == "detail":
            # An EXPLICIT canvas request outranks the open detail: the reader
            # asked for the board/timeline, not the page they were on.
            self._exit_detail()
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
        if self._mode == "detail":
            # Keep the detail open across a recomposition: re-show the SAME
            # project (by id — the store's order can shift with a rename) and
            # keep the row cursor; a deleted project pops back to the canvas.
            self._resync_detail()
        elif self._detail_page.display:
            self._exit_detail()
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
                # A retarget is a fresh intent: whatever the last notice said
                # belonged to the project being left (UX round 1).
                self._notice = None
                if self._mode == "detail":
                    # A named show is a CANVAS retarget, obeying the same rule
                    # `load` states for an explicit canvas request: the reader
                    # asked to be shown a project, not the page they were on.
                    # Leaving the page open split it — chrome on the named
                    # project, body on the old one — and `↵` then wrote through
                    # the cursor into a project its row never came from
                    # (QA round 1, Q1).
                    self._exit_detail()
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

    def set_manager_target(self, target: SendTarget | None) -> None:
        """Inject the manager row the picker should offer (P5a).

        ``None`` means this session has no manager — the row is then simply
        absent, which is the spec's rule (never a dead row).
        """
        self._manager_target = target

    def escape_surface(self) -> bool:
        """Consume ``esc`` for a send surface, if one is up (P5a).

        The composer owns the caret while composing, so the key reaches the
        APP's Esc binding rather than this view's; this is the door the app
        asks before it dismisses a page someone is still typing into.
        """
        if self._mode == "compose":
            self.end_compose()
            return True
        if self._mode == "send":
            self._close_send_picker()
            return True
        if self._mode == "start":
            self._close_start_picker()
            return True
        return False

    def show_notice(self, text: str) -> None:
        """One sentence in the footer until the state changes (UX round 1).

        The mode's OWN line for refusals and pops: while a page is up the
        transcript it hides is no surface at all, so a sentence posted there
        reached nobody (U1) — and the footer is the pinned one-row chrome, so
        nothing moves when the line appears. Cleared by the next ``load``, a
        fresh detail entry, or leaving the detail by hand.

        An EMPTY sentence is NO sentence: the footer shows the notice INSTEAD
        of the project detail while one exists, so ``show_notice("")`` — how
        the delivered receipt clears the in-flight line — must put the detail
        back rather than blank the row (caught in the pass-5 frame).
        """
        self._notice = text or None
        self._paint_chrome()

    def _notice_text(self, width: int) -> Text:
        """The footer's notice line: warning ink, fitted to the MEASURED box.

        Rich's ``overflow="ellipsis"`` is inert on a Static (the title's
        recorded lesson), so the cut is done HERE — cell-accurate, with an
        ellipsis — instead of letting the box clip mid-token (UX round 1,
        U8: at 50x18 the refusal sentence ran 72 cells into a 46-cell box).
        """
        from rich.cells import cell_len

        sentence = self._notice or ""
        if cell_len(sentence) > width:
            budget = max(width - 1, 1)  # one cell for the ellipsis
            kept: list[str] = []
            used = 0
            for char in sentence:
                size = cell_len(char)
                if used + size > budget:
                    break
                kept.append(char)
                used += size
            sentence = "".join(kept).rstrip() + "…"
        return Text(
            sentence,
            style=Style(color=theme_mod.semantic_color("warning")),
            no_wrap=True,
        )

    def _paint_chrome(self) -> None:
        if self._mode == "detail":
            self._paint_detail_chrome()
            return
        if self._mode == "form":
            self._paint_form_chrome()
            return
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
        self._paint_rule(width)

        # The footer fits the DETAIL box's own measured width, not the page's
        # minus its padding: the two differ by the box's position in the
        # layout, and fitting to the smaller number skipped a rung the space
        # could hold (measured: at 100x30 the box is 96 cells, the page
        # arithmetic says 94, and a 97-cell rung was shed that fits). The
        # board/timeline counts line takes the same measured width so its own
        # ladder sheds whole segments instead of the container cutting a
        # clause mid-token (design review round 1, D5).
        footer_width = self._detail.size.width or width
        if self._notice is not None:
            self._detail.update(self._notice_text(footer_width))
        elif self._view == "list" and self._views:
            index = max(0, min(self._cursor, max(self._painted_count() - 1, 0)))
            self._detail.update(
                detail_footer(self._views[index], style_for=_style_resolver(), width=footer_width)
            )
        else:
            tier = self._tier if self._view == "timeline" else None
            self._detail.update(
                aggregate_footer(
                    self._views,
                    tier=tier,
                    style_for=_style_resolver(),
                    width=footer_width,
                )
            )
        self._paint_hints()

    def _paint_form_chrome(self) -> None:
        """Chrome for the form state (spec §5.5): identity, rule, one footer line.

        The title states where the reader is rather than counting what exists —
        no tracked total, no `◆` clause: none of them is about this surface,
        and the key the reader is typing is the one fact the title cannot know
        until the field says it. The rule is the plain one (the form has no
        sections to be measured against) and the footer carries the notice when
        there is one, else the sentence that names the form's one rule.
        """
        from rich.cells import cell_len

        # The fields re-resolve their own ink from the LIVE resolver, so a theme
        # switch reaches the form's labels and `‹ value ›` rows mid-fill.
        self._form_page.restyle(_style_resolver())
        muted = Style(color=theme_mod.semantic_color("muted"))
        title = Text(no_wrap=True, overflow="ellipsis")
        title.append("projects", style=Style(color=theme_mod.semantic_color("fg"), bold=True))
        title.append(" · new project", style=muted)
        self._title.update(title)
        width = max(self.size.width - 2, 1)
        self._paint_rule(width)
        footer_width = self._detail.size.width or width
        if self._notice is not None:
            self._detail.update(self._notice_text(footer_width))
        else:
            sentence = FORM_FOOTER_HINT
            if cell_len(sentence) > footer_width:
                budget = max(footer_width - 1, 1)
                kept: list[str] = []
                used = 0
                for char in sentence:
                    size = cell_len(char)
                    if used + size > budget:
                        break
                    kept.append(char)
                    used += size
                sentence = "".join(kept).rstrip() + "…"
            self._detail.update(Text(sentence, style=muted, no_wrap=True))
        self._paint_hints()

    def _paint_detail_chrome(self) -> None:
        """Chrome for the detail state: identity title, in-page ruler, freshness.

        The title pins the identity (spec §5.4) and sheds the newest clauses
        first — the update stamp, then the `(key)` parenthetical — the canvas
        title's own ladder. The rule is the detail's section ruler (design §6)
        and the footer the freshness attribution, which must stay visible
        while the description scrolls under it.
        """
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        view_row = self._detail_view_row() or {}
        project = view_row.get("project") if isinstance(view_row.get("project"), dict) else {}
        project = project if isinstance(project, dict) else {}
        name = str(project.get("name") or "")

        def build_title(*, with_name: bool, with_updated: bool) -> Text:
            title = Text(no_wrap=True, overflow="ellipsis")
            label = display_name(project) or "(unnamed)"
            title.append("projects", style=Style(color=theme_mod.semantic_color("fg"), bold=True))
            title.append(" · ", style=muted)
            title.append(label, style=Style(color=theme_mod.semantic_color("fg"), bold=True))
            if with_name and name and name != label:
                title.append(f" ({name})", style=muted)
            status = str(project.get("status") or "active")
            title.append(
                f" {status_chip_text(status)}", style=_style_resolver()(f"status_{status}")
            )
            if with_updated and self._updated_at is not None:
                import time as _time

                stamp = _time.strftime("%H:%M", _time.localtime(self._updated_at))
                title.append(f" · updated {stamp}", style=dim)
            return title

        from rich.cells import cell_len

        title = build_title(with_name=True, with_updated=True)
        available = self._title.content_size.width or self._title.size.width
        if available and cell_len(title.plain) > available:
            title = build_title(with_name=True, with_updated=False)
            if cell_len(title.plain) > available:
                title = build_title(with_name=False, with_updated=False)
        self._title.update(title)

        width = max(self.size.width - 2, 1)
        self._paint_rule(width)

        footer_width = self._detail.size.width or width
        if self._notice is not None:
            self._detail.update(self._notice_text(footer_width))
        else:
            self._detail.update(
                detail_progress_line(view_row, width=footer_width, style_for=_style_resolver())
            )
        self._paint_hints()

    def _paint_rule(self, width: int | None = None) -> None:
        """The rule row: the shipped plain rule, or the section ruler.

        Grouped canvases turn the rule into the sticky counterpart (design
        D2): the zero-row ruler names the section at the VIEWPORT's top, and
        the dim `sel` clause names the cursor's section when the two differ.
        Ungrouped — and off any section (the axis row, a blank separator, the
        truncation row) — this is exactly the shipped dim rule.
        """
        if width is None:
            width = max(self.size.width - 2, 1)
        if self._mode == "form":
            # The form has no sections to measure against, so its rule is the
            # shipped plain one — the `ruler is None` branch below.
            ruler = None
        elif self._mode == "detail":
            ruler = detail_ruler(
                self._detail_page.section_anchors(),
                int(self._detail_page.scroll_offset.y),
                width=width,
                style_for=_style_resolver(),
            )
        else:
            ruler = section_ruler(
                self._views,
                self._view,
                top_row=int(self._body.scroll_offset.y),
                cursor=self._cursor if self._views else None,
                width=width,
                style_for=_style_resolver(),
            )
        if ruler is None:
            ruler = Text("─" * width, style=Style(color=theme_mod.semantic_color("dim")))
        self._rule.update(ruler)

    def _scroll_changed(self, *_args: Any) -> None:
        """The ruler tracks the viewport (design D2), not the cursor.

        Fired by the body's ``scroll_y`` watch for every scroll path; a plain
        rule never changes with scrolling, so it is skipped there.
        """
        if self._last is None or not self._last.sections:
            return
        self._paint_rule()

    def _detail_state_changed(self) -> None:
        """The detail page changed a row's own verb: re-arm the chrome.

        Without this the footer kept offering `↵ expand` on the row the reader
        had just opened (UX review round 1, U2). Deferred like every other
        chrome paint here — the toggled page settles first, so the hint is
        measured against the row it will act on.
        """
        self.call_after_refresh(self._paint_detail_chrome)

    def _detail_scroll_changed(self, *_args: Any) -> None:
        """The detail's ruler tracks ITS viewport (design §6).

        Same zero-row counterpart as the canvas rule, watching the detail
        body's own ``scroll_y``. The repaint is DEFERRED past the refresh: a
        scroll watch can fire before the children's regions move, and an
        anchor read at that instant is the stale offset's (design review
        round 1, D1, measured — the entry reveal's scroll was exactly this
        case). After the refresh the children sit at the settled offset and
        the anchors are content-relative again; the pass is idempotent like
        every deferred chrome paint in this file.
        """
        if self._mode != "detail":
            return
        self.call_after_refresh(self._paint_rule)

    def _paint_hints(self) -> None:
        """Lay out the footer hints, shedding WHOLE hints until the row fits.

        TWO ladders, one machinery: the canvases' and the detail page's, each
        built as candidate rungs and committed through the same measure-then-
        paint path. The org-chart rule throughout: each rung is measured
        before it is committed and ``esc`` is never dropped because it is the
        only way out.
        """
        rungs = self._canvas_hint_rungs()
        if self._mode == "detail":
            rungs = self._detail_hint_rungs()
        elif self._mode == "form":
            rungs = self._form_hint_rungs()
        elif self._mode == "compose":
            rungs = self._compose_hint_rungs()
        elif self._mode == "send":
            rungs = self._send_hint_rungs()
        elif self._mode == "start":
            rungs = self._start_hint_rungs()
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
        for hint in self._hint_buttons():
            hint.display = hint in visible
        # The hint BUTTONS keep one DOM order (the two ladders share them), and
        # a Horizontal lays children out by document order — not by paint
        # time — so the row read `r refresh · ↵ toggle · ↑↓ move` on the
        # detail page (measured, and exactly what a shared widget pool costs)
        # until the painter re-ordered it to the plan it just chose. Each
        # rung's visible order IS its priority order; the seam flags painted
        # above come from the plan, so they cannot disagree with it.
        if [child for child in self._hints.children if child.display] != [
            hint for hint, _label, _lead in plan
        ]:
            for index in range(len(plan) - 2, -1, -1):
                self._hints.move_child(plan[index][0], before=plan[index + 1][0])
        # Arm what was just painted against the geometry it would act on; the
        # deferred pass in `_repaint` re-arms once the layout has settled.
        if self._mode == "detail":
            self._sync_detail_hints()
        elif self._mode == "form":
            self._sync_form_hints()
        else:
            self._sync_scroll_hint()

    def _hint_buttons(self) -> tuple[HintButton, ...]:
        """Every button the painter owns — one list for the hide pass."""
        return (
            self._scroll_hint,
            self._list_hint,
            self._board_hint,
            self._timeline_hint,
            self._next_hint,
            self._detail_hint,
            self._refresh_hint,
            self._create_hint,
            self._msg_hint,
            self._start_hint,
            self._copy_hint,
            self._preview_hint,
            self._open_hint,
            self._move_hint,
            self._page_hint,
            self._zoom_hint,
            self._tab_hint,
            self._save_hint,
            self._exit_hint,
            self._state_hint,
        )

    def _form_hint_rungs(
        self,
    ) -> list[tuple[list[tuple[HintButton, str, bool]], str]]:
        """The form's ONE rung (spec §4): `tab next field · ctrl+s save · esc cancel`.

        One rung rather than a ladder, and that is a measurement rather than a
        wish: the plan is 41 cells, which fits at the 60-column floor, so there
        is nothing to shed. `esc` keeps its meaning in both states — it is what
        gets a reader out of the form and out of the discard confirm.
        """

        def rung(
            leads: list[tuple[HintButton, str, bool]], esc_label: str
        ) -> tuple[list[tuple[HintButton, str, bool]], str]:
            # `esc` is appended by the RUNG, exactly as the canvas ladders do it
            # — the label is supplied per mode, so the button itself never
            # carries one — and the state note is never painted here: the form
            # is the surface, not a read-only view of one.
            row = list(leads)
            row.append((self._exit_hint, esc_label, bool(row)))
            return (row, esc_label)

        tab = (self._tab_hint, " next field", False)
        save = (self._save_hint, " save", True)
        return [rung([tab, save], "cancel")]

    def _compose_hint_rungs(self) -> list[tuple[list[tuple[HintButton, str, bool]], str]]:
        """The compose mode's ONE rung: ``esc cancel``, and nothing else (U5).

        While the composer holds the caret, every other key the canvas ladder
        advertises — ``1 list``, ``v next``, ``c create``, ``m message``,
        ``d detail`` — TYPES into the message. A hinted key that does nothing
        is the defect the page's own hints rules name; a hinted key that types
        your sentence is worse (UX round 1, U5). ``esc`` is the one page key
        that still means something here, and it means cancel.
        """

        def rung(
            leads: list[tuple[HintButton, str, bool]], esc_label: str
        ) -> tuple[list[tuple[HintButton, str, bool]], str]:
            row = list(leads)
            row.append((self._exit_hint, esc_label, bool(row)))
            return (row, esc_label)

        return [rung([], "cancel")]

    def _send_hint_rungs(self) -> list[tuple[list[tuple[HintButton, str, bool]], str]]:
        """The target picker's rung: EMPTY — the card carries the whole grammar (D10).

        Two rounds of the same defect, in opposite directions. The page row must
        not advertise keys the card CONSUMES (R2-3): while the card is up it
        holds the keyboard, so `1 list`, `c create`, `m message`, `d detail` type
        into its filter. Nor may it repeat what the card already says: the legend
        sits inside the card one row above this row, so painting `esc close` here
        stated one instruction twice in two inks (design round 2, D10). An empty
        rung hides every button — the row is blank while the card owns the keys,
        and the canvas ladder returns when it closes.

        P5b's start picker takes the same rung for the same reasons; it is a
        separate mode (``start``) because its card is a separate surface, and
        the two must never both be up.
        """
        return [([], "")]

    def _start_hint_rungs(self) -> list[tuple[list[tuple[HintButton, str, bool]], str]]:
        """The start picker's rung: EMPTY, exactly the send card's (D10)."""
        return [([], "")]

    def _sync_form_hints(self) -> None:
        """Arm the form's hints against what they would act on just now.

        `ctrl+s save` and `tab next field` are disarmed while the discard
        confirm is up: the page ignores both there (the confirm owns the
        keyboard), and a lit key that does nothing is the defect
        ``HintButton.set_actionable`` exists to prevent.
        """
        confirming = self._form_page.confirming
        self._save_hint.set_actionable(not confirming)
        self._tab_hint.set_actionable(not confirming)

    def _canvas_hint_rungs(
        self,
    ) -> list[tuple[list[tuple[HintButton, str, bool]], str]]:
        """The canvases' ladder, in priority order (widest rung first).

        What sheds first is the order's own statement: ``+/-`` (timeline
        only), then ``↵ open`` and ``r refresh`` before the newest action,
        ``d detail`` (S6d parity P2 — the canvas's headline affordance
        outranks the legacy conversation key under width pressure; the
        designer round may re-rank, it is one rung swap); then ``↔↕ scroll``,
        then ``d`` itself, and only then the esc LABEL and the view triplet —
        a gesture a reader finds by trying an arrow goes before a view they
        cannot discover, and the newest view types stay advertised on a
        narrow terminal (UX round 1, U3).

        ``c create`` joined at the same rank as ``d detail`` (UX round 1, U1):
        a reader who cannot see how to make a project cannot use the page, so
        the two share the newest rungs and shed together. It was a BINDING
        with no hint at all before this round — the create key existed and
        nothing advertised it, which is the one failure this ladder exists to
        prevent.
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

        scroll = (self._scroll_hint, " scroll", False)
        list_hint = (self._list_hint, " list", True)
        board_hint = (self._board_hint, " board", True)
        timeline_hint = (self._timeline_hint, " timeline", True)
        refresh = (self._refresh_hint, " refresh", True)
        create_hint = (self._create_hint, " create", True)
        # `m message`: the one affordance a reader cannot guess. The invariant
        # this ladder enforces is measured, not aspirational (agent review
        # round 1, F3): `m message` is advertised on EVERY rung that still
        # carries `c create`. The rung at the scroll/refresh rank used to omit
        # it while an equal-width rung below carried it, so widening the
        # terminal made the hint VANISH (absent at 110-130 view columns where
        # `c create` still painted). `r refresh` sheds first; the pair falls
        # together.
        msg_hint = (self._msg_hint, " message", True)
        # `s start` (P5b) — the spec's own label for the key that sets a session
        # going (§3.3). It rides the ladder's WIDEST rung as a SEPARATE,
        # one-rung-wider variant of it rather than as an extra element inside it.
        #
        # THE ARITHMETIC, re-derived twice (agent review round 1, F3/F4; QA
        # round 2, Q3). The canvas budget is `size.width - 2`, and the page's
        # `size.width` is terminal − 4 — 144 at a 150-column terminal, 94 at 100
        # — so the withdrawn "149 against the 148 a 150-column terminal offers"
        # was measuring the wrong budget. Two things follow, both measured:
        #
        #   * `s start` cannot be spliced into a SHIPPED rung in place: adding
        #     its 11 cells pushed every budget in 103–111 past those rungs onto
        #     the next one, and `↔↕ scroll` — which the shipped row carried —
        #     vanished at terminals 109–117 (F4).
        #   * nor can a variant ride a rung that is not the WIDEST: the ladder
        #     takes the FIRST rung that fits, so the key-less shipped rung that
        #     follows each variant pushed the next variant out of reach and the
        #     key BLINKED — absent at 160/150/140/130/125/118…, present at 145
        #     and 120 (Q3). A hint that vanishes as the terminal grows is the
        #     same defect the `m message` comment below records.
        #
        # What is left is what the detail ladder already does: the key rides the
        # rung that has room for it, and every shipped row below is untouched.
        # `s` itself works at every width whether or not it is advertised.
        start_hint = (self._start_hint, " start", True)
        open_hint = (self._open_hint, " open", True)
        detail_hint = (self._detail_hint, " detail", True)
        nxt = (self._next_hint, " next", True)
        # `+/-` is TIME zoom: it acts only on the timeline, and a hinted key
        # that changes nothing is worse than an absent one (the org chart's own
        # rule for its zoom hint) — so the button is advertised where it works
        # and dropped from every rung elsewhere.
        zoom: tuple[HintButton, str, bool] | None = (
            (self._zoom_hint, " zoom", True) if self._view == "timeline" else None
        )

        def with_start(
            leads: list[tuple[HintButton, str, bool]],
        ) -> list[tuple[HintButton, str, bool]]:
            """The same row with `s start` spliced in right after `m message`."""
            spliced: list[tuple[HintButton, str, bool]] = []
            for lead in leads:
                spliced.append(lead)
                if lead[0] is self._msg_hint:
                    spliced.append(start_hint)
            if not any(hint is self._start_hint for hint, _label, _lead in spliced):
                spliced.append(start_hint)
            return leads_of(*spliced)

        rungs: list[tuple[list[tuple[HintButton, str, bool]], str]] = []

        #: The key may only ride a PREFIX of the ladder, and that is a measured
        #: property rather than a preference. The ladder takes the FIRST rung
        #: that fits, so a key-less rung sitting in front of a narrower
        #: key-carrying one makes the key VANISH as the terminal grows — the
        #: shape QA round 2 (Q3) measured on the previous revision: `s start`
        #: absent at 160/150/140/130/125/118…, present at 145 and 120. This flag
        #: closes for good as soon as a rung lands that the key cannot join.
        key_prefix = True

        def add(
            *hints: tuple[HintButton, str, bool] | None,
            esc_label: str,
            state: bool = False,
            variant: bool = False,
        ) -> None:
            """Append a shipped rung, and — while the key's prefix is open — its variant.

            The variant is the same row with `s start` spliced in after
            `m message`, so it can only ever ADD the key; it goes in front of its
            own shipped rung, which is where the first-fit ladder reaches it.
            Two gates keep the result honest, both measured rather than
            tabulated: the variant must be strictly NARROWER than the rung before
            it (a wider one would sit behind its predecessor forever — a dead
            rung), and `key_prefix` must still be open. In practice that leaves
            the key on the ladder's WIDEST rung — the one with room for it —
            which is also the shape the detail ladder takes, so the two agree.
            """
            nonlocal key_prefix
            base = rung(leads_of(*hints), esc_label, state=state)
            if variant and key_prefix:
                candidate = rung(with_start(leads_of(*hints)), esc_label, state=state)
                if not rungs or self._measure_hints(*candidate) < self._measure_hints(*rungs[-1]):
                    rungs.append(candidate)
            rungs.append(base)
            # A SHIPPED rung never carries the key, so the prefix ends here:
            # every rung below it is narrower, and a key there would vanish as
            # the terminal widens (QA round 2, Q3).
            key_prefix = False

        add(
            scroll,
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            refresh,
            create_hint,
            msg_hint,
            open_hint,
            detail_hint,
            zoom,
            esc_label="back to conversation",
            state=True,
            variant=True,
        )
        add(
            scroll,
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            refresh,
            create_hint,
            msg_hint,
            open_hint,
            detail_hint,
            zoom,
            esc_label="back to conversation",
        )
        add(
            scroll,
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            refresh,
            create_hint,
            msg_hint,
            open_hint,
            detail_hint,
            zoom,
            esc_label="back",
        )
        # The timeline's own row gives up `d detail` before it gives up `+/-`
        # zoom: zoom acts on THIS view and `d` acts on the page, and the rung
        # exists only where zoom does (the list and board rows keep `d detail`
        # at every width they have it today).
        if zoom is not None:
            add(
                scroll,
                list_hint,
                board_hint,
                timeline_hint,
                nxt,
                refresh,
                create_hint,
                msg_hint,
                open_hint,
                zoom,
                esc_label="back",
            )
        add(
            scroll,
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            refresh,
            create_hint,
            msg_hint,
            open_hint,
            detail_hint,
            esc_label="back",
        )
        # `↵ open` sheds first among the shipped hints (`scale`-free rows keep
        # `r refresh`): the 116-cell shipped rung, which the ladder chose at
        # 109–121 columns before this key existed.
        add(
            scroll,
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            refresh,
            create_hint,
            msg_hint,
            detail_hint,
            esc_label="back",
        )
        add(
            scroll,
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            create_hint,
            msg_hint,
            detail_hint,
            esc_label="back",
        )
        add(
            list_hint,
            board_hint,
            timeline_hint,
            nxt,
            create_hint,
            msg_hint,
            detail_hint,
            esc_label="back",
        )
        # Everything below sheds the page's older keys; the key's prefix is long
        # closed, and the flag means none of them can reopen it.
        add(list_hint, board_hint, timeline_hint, nxt, esc_label="back", variant=False)
        for tail in (
            (list_hint, board_hint, timeline_hint, nxt),
            (list_hint, board_hint, timeline_hint),
            (list_hint, board_hint),
            (list_hint,),
            (),
        ):
            add(*tail, esc_label="", variant=False)
        return rungs

    def _detail_hint_rungs(
        self,
    ) -> list[tuple[list[tuple[HintButton, str, bool]], str]]:
        """The detail page's ladder — P2's subset of the spec's §3.3 row.

        ``↑↓ move · pgup/pgdn page · ↵ <context> · r refresh · esc back``:
        only what WORKS on this slice (`m`/`s`/`e` land with P4–P6; a hinted
        key that does nothing is the failure the page's own rules name), and
        the `↵` label is the selected row's VERB — ``open`` on a session,
        ``toggle`` on a milestone — inside the shipped slot (spec §3.2/10.3).
        ``esc`` pops one level here, so its label is ``back`` at every rung.
        """

        def rung(
            leads: list[tuple[HintButton, str, bool]], esc_label: str
        ) -> tuple[list[tuple[HintButton, str, bool]], str]:
            row = list(leads)
            row.append((self._exit_hint, esc_label, bool(row)))
            return (row, esc_label)

        context = self._detail_page.selected_action_label()
        verb = self._detail_page.selected_action_verb()
        move = (self._move_hint, " move", False)
        page = (self._page_hint, " page", True)
        # The `↵` hint exists only while the selected row HAS a verb: its old
        # `" open"` fallback was unreachable until design D1 and UX U3 made
        # verb-less rows real (a one-line entry, a missing copy), and a dimmed
        # `↵ open` on a row that cannot act is the same wrong promise in a
        # quieter ink (UX review round 2, U6). No verb, no rung.
        named = context or verb
        open_hint = (self._open_hint, f" {named}", True)
        # The bare verb is a LOW rung: a long target name sheds before the key
        # does (UX round 1, U3 — the name informs, the key acts).
        open_bare = (self._open_hint, f" {verb or context}", True)
        refresh = (self._refresh_hint, " refresh", True)
        # `m message` (P5a): the detail page is where messaging lives, and it
        # is the first of the new keys to shed (spec §3.3), so it rides only
        # the two widest rungs of each branch.
        msg = (self._msg_hint, " message", True)
        # `s start` (P5b), on the canvas's own rule: the rungs below are the
        # SHIPPED rows and each one that can carry the new key is preceded by a
        # wider variant of itself. The ladder stops at the first rung that fits,
        # so the variant can only ADD `s` — where it does not fit, the shipped
        # row is chosen unchanged. Measured: the widest detail row on a row
        # reading `↵ toggle groundwork` costs 87 cells, and with `s start`
        # spliced in it costs 97 against the 94 a 100-column terminal offers, so
        # `s` is advertised from 101 columns up and every shipped row holds.
        # `s` still WORKS wherever it is not advertised.
        start = (self._start_hint, " start", True)
        # `y copy` / `space preview` (P6, spec §7.4): the two ROW-SCOPED keys,
        # advertised only while an attachment row has the cursor. They follow
        # the `s start` variant rule exactly — each rides a one-extra-wider
        # VARIANT of the shipped row rather than an extra element inside it, so
        # a variant can only ADD a key and no shipped label is ever displaced
        # at any budget (the F4 sweep). The chain is a strict prefix, so
        # widening the terminal only ever ADDS a key (QA Q3's monotonicity).
        # `y` is offered wherever a path exists — a gone copy still has a path
        # worth pasting — and `space` only where there are pixels to read.
        extras: list[tuple[HintButton, str, bool]] = []
        if self._detail_page.selected_can_preview():
            extras.append(
                (
                    self._preview_hint,
                    " hide" if self._detail_page.selected_preview_open() else " preview",
                    True,
                )
            )
        if self._detail_page.selected_can_copy():
            extras.append((self._copy_hint, " copy", True))
        head = [move, page] if named is None else [move, page, open_hint]
        # Shedding among the extras goes WIDEST LABEL FIRST, so a narrower
        # terminal keeps the cheaper key: the reader reaches `↵ open · y copy`
        # — the pair spec §3.3 names — before `space preview` (17 cells to
        # `y copy`'s 10), and a rung that carries both is strictly wider than
        # the one that follows (the variants are never shadowed).
        variants: list[list[tuple[HintButton, str, bool]]] = [extras]
        if len(extras) > 1:
            variants.append(extras[1:])
        if named is None:
            return [
                *(rung([*head, *chosen, msg, start, refresh], "back") for chosen in variants),
                rung([move, page, msg, start, refresh], "back"),
                rung([move, page, msg, refresh], "back"),
                rung([move, page, msg], "back"),
                rung([move, page], "back"),
                rung([move], ""),
                rung([], ""),
            ]
        return [
            *(rung([*head, *chosen, msg, start, refresh], "back") for chosen in variants),
            rung([move, page, open_hint, msg, start, refresh], "back"),
            rung([move, page, open_hint, msg, refresh], "back"),
            rung([move, page, open_hint, msg], "back"),
            rung([move, page, open_hint], "back"),
            rung([move, open_hint], "back"),
            rung([move, open_hint], ""),
            rung([move, open_bare], ""),
            rung([move], ""),
            rung([], ""),
        ]

    def _sync_detail_hints(self) -> None:
        """Arm the detail hints against what they would act on just now.

        ``↑↓ move`` needs a selectable row, ``pgup/pgdn`` a scrollable page,
        and the ``↵`` context action a selected row that HAS one — a heading
        or the description offers nothing, and the hint states the row's verb
        or stops offering itself (``HintButton.set_actionable``'s rule).
        """
        self._move_hint.set_actionable(self._detail_page.selectable_count > 0)
        self._page_hint.set_actionable(self._detail_page.max_scroll_y > 0)
        self._open_hint.set_actionable(self._detail_page.selected_action_label() is not None)
        # The two row-scoped keys are armed against the SELECTED row's own
        # facts, not against the page's: there is nothing to copy off a
        # description row, and a lit `y copy` there is the same wrong promise
        # `↵ open` was on a verb-less row (UX round 2, U6).
        self._copy_hint.set_actionable(self._detail_page.selected_can_copy())
        self._preview_hint.set_actionable(self._detail_page.selected_can_preview())

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
        yield self._detail_page
        yield self._form_page
        yield self._detail
        with self._hints:
            yield self._scroll_hint
            yield self._list_hint
            yield self._board_hint
            yield self._timeline_hint
            yield self._next_hint
            yield self._detail_hint
            yield self._refresh_hint
            yield self._create_hint
            yield self._msg_hint
            yield self._start_hint
            yield self._open_hint
            yield self._copy_hint
            yield self._preview_hint
            yield self._move_hint
            yield self._page_hint
            yield self._zoom_hint
            yield self._tab_hint
            yield self._save_hint
            yield self._exit_hint
            yield self._state_hint

    def on_mount(self) -> None:
        # Focus lands here rather than at the app's open call: focus() on a
        # widget not yet in the focus chain is a silent no-op (the subagent
        # view's recorded bug), and the advertised keys would go to the inert
        # composer. Repaint after focus so the first frame is the settled one.
        self._repaint()
        # The rule row is a section ruler on a grouped canvas and must track
        # the VIEWPORT (design D2) — nothing else repaints when a wheel
        # scrolls, and a ruler frozen on the section the reader scrolled away
        # from is worse than none. The subagent page's watch shape, same
        # reason.
        self.watch(self._body, "scroll_y", self._scroll_changed, init=False)
        # The detail body has its own viewport and its own ruler watch: the
        # rule must follow whichever body is on screen (design §6).
        self.watch(self._detail_page, "scroll_y", self._detail_scroll_changed, init=False)
        try:
            self.focus()
        except Exception:
            pass

    def on_resize(self) -> None:
        # The rule spans the page and the hints shed against a width only the
        # layout knows, so both repaint on resize. The canvas is
        # width-independent (it scrolls), so only the chrome moves. A floating
        # send card re-reads its anchor and budget from the same settled
        # layout, deferred like every geometry read here.
        self._paint_chrome()
        self.call_after_refresh(self._sync_scroll_hint)
        if self._send_card is not None:
            self.call_after_refresh(self._place_send_card)
        if self._start_card is not None:
            self.call_after_refresh(self._place_start_card)

    def _sync_scroll_hint(self) -> None:
        """Arm ``↔↕ scroll`` only while the body has somewhere to scroll.

        ``HintButton.set_actionable`` exists for exactly this (its docstring
        calls the alternative "the reported 'nothing happens when I click' bug
        one step earlier"), and the sibling subagent page drives it the same
        way. Cheap and idempotent, so it rides the chrome paint plus one
        deferred pass for the post-layout geometry (agent review round 1,
        finding 8: the hint lit on hover on a canvas that could not scroll).
        """
        if self._mode != "canvas":
            return
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
        if self._mode == "form":
            # The form's rows ARE its fields: a readback that skipped them
            # would leave this slice's headline surface unassertable — the
            # same reason the detail's Markdown readback exists (P3, U6).
            rows.extend(self._form_page.readback())
            rows.append(plain(self._detail))
            return rows
        if self._mode == "detail":
            # The detail state swaps the CANVAS rows for the page's own rows;
            # the title, rule and footer are the same boxes either way (S6d
            # P2), so the reads stay in one place for tests and evidence.
            rows.extend(self._detail_page.painted_rows())
            rows.append(plain(self._detail))
            return rows
        if self._last is not None:
            rows.extend(text.plain for text in self._last.text.split("\n"))
        rows.append(plain(self._detail))
        return rows

    # -- view switching -----------------------------------------------------
    def _set_view(self, view: str) -> None:
        if self._mode != "canvas":
            # The view triplet is a CANVAS control; on the detail page it is
            # inert rather than a hidden exit (spec §10.1: actions no-op where
            # they do not apply — the `action_zoom_in` pattern).
            return
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

    # -- sections (S6d parity) ---------------------------------------------
    def action_next_section(self) -> None:
        self._section_jump(1)

    def action_prev_section(self) -> None:
        self._section_jump(-1)

    def _section_jump(self, direction: int) -> None:
        """``shift+↓``/``shift+↑``: the neighbouring section's first row.

        Clamped at the ends (a canvas is several viewports tall; the bottom is
        a destination — the same clamp the arrows state) and inert while the
        canvas is ungrouped. On the DETAIL page the same keys move the row
        cursor to the neighbouring SECTION's first selectable row (spec §3.2).
        """
        if self._mode == "detail":
            self._detail_page.jump_to_section(direction)
            self._paint_chrome()
            return
        groups = sections_of(self._views)
        if groups is None or not self._views:
            return
        target: list[int] | None = None
        for position, (_label, indexes) in enumerate(groups):
            if self._cursor in indexes:
                neighbour = position + direction
                if 0 <= neighbour < len(groups):
                    target = groups[neighbour][1]
                break
        if not target:
            return
        self._select_index(min(target[0], max(self._painted_count() - 1, 0)))

    def _select_index(self, index: int) -> None:
        """Land the selection on a project (a click, a header, a section jump).

        The exact `_move` steps — clamp, repaint, reveal — so the mouse and
        the keyboard cannot diverge about what is selected.
        """
        if not self._views:
            return
        position = max(0, min(index, max(self._painted_count() - 1, 0)))
        if position == self._cursor:
            return
        self._cursor = position
        self._repaint()
        self._scroll_cursor_into_view()

    def action_jump(self) -> None:
        """``↵``: the selected row's own action — canvas: open the conversation;
        detail: activate the selected row (a session opens through the same
        shipped ladder; a milestone toggles) — spec §3.2/§10.3."""
        if self._mode == "detail":
            self._detail_page.activate()
            return
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
        # page keys still pan. On the detail page the arrows move the ROW
        # cursor over the page's selectable rows, clamped (spec §10.3).
        if self._mode == "detail":
            self._detail_nav(-1)
            return
        self._move(-1)

    def action_down(self) -> None:
        if self._mode == "detail":
            self._detail_nav(1)
            return
        self._move(1)

    def current_project_id(self) -> str | None:
        """The selected project's id, or ``None`` (the host seeds from it)."""
        row = self._current_project_row()
        return str(row.get("id") or "") or None if row is not None else None

    def current_project_name(self) -> str:
        """The selected project's NAME — the address the card names and the
        kickoff's ``@project:`` reference resolves against."""
        row = self._current_project_row()
        return str(row.get("name") or "") if row is not None else ""

    def _current_project_row(self) -> dict[str, Any] | None:
        """The selected row's project payload, or ``None`` (one lookup for both
        accessors above, so id and name can never come from different rows)."""
        if not (0 <= self._cursor < len(self._views)):
            return None
        view_row = self._views[self._cursor]
        project = view_row.get("project") if isinstance(view_row, dict) else None
        return project if isinstance(project, dict) else None

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
            y = list_position(self._views, self._cursor)
            if y is None:
                return
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
        if self._mode == "detail":
            return
        self._body.scroll_left()

    def action_scroll_right(self) -> None:
        if self._mode == "detail":
            return
        self._body.scroll_right()

    def action_page_up(self) -> None:
        if self._mode == "detail":
            self._detail_page.scroll_page_up()
            return
        if self._view == "list":
            self._move(-max(1, self._usable_height()))
            return
        self._body.scroll_page_up()

    def action_page_down(self) -> None:
        if self._mode == "detail":
            self._detail_page.scroll_page_down()
            return
        if self._view == "list":
            self._move(max(1, self._usable_height()))
            return
        self._body.scroll_page_down()

    def action_page_left(self) -> None:
        self._body.scroll_page_left()

    def action_page_right(self) -> None:
        self._body.scroll_page_right()

    def action_scroll_home(self) -> None:
        if self._mode == "detail":
            # The detail page scrolls one axis; home/end are the content's
            # edges and leave the row cursor where it is (the wheel's rule).
            self._detail_page.scroll_to(y=0, animate=False)
            return
        if self._view == "list":
            self._move(-self._cursor)
            return
        # Top-left corner, both axes pinned (Textual's scroll_home resets only
        # the Y axis unless x is passed).
        self._body.scroll_to(x=0, y=0, animate=False)

    def action_scroll_end(self) -> None:
        if self._mode == "detail":
            self._detail_page.scroll_to(y=self._detail_page.max_scroll_y, animate=False)
            return
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

    # -- mouse (S6d parity): rows, cards, header clicks --------------------
    def _canvas_cell(self, event: Any) -> tuple[int, int] | None:
        """Canvas ``(x, y)`` of a mouse event, or ``None`` outside the viewport.

        Resolved from the scroll viewport plus the scroll offset — never the
        painted Static's own ``region``, which layout recomputes and which
        lags a scroll by a frame (the analytics page's measured lesson).
        """
        viewport = self._body.scrollable_content_region
        if not viewport.contains(event.screen_x, event.screen_y):
            return None
        x = event.screen_x - viewport.x + int(self._body.scroll_offset.x)
        y = event.screen_y - viewport.y + int(self._body.scroll_offset.y)
        return (x, y)

    def on_click(self, event: Any) -> None:
        """The canvas click map (design §4): select + reveal; a second click
        activates the shipped action (``↵``); a header row jumps to its section.

        One gesture acts on one surface: the handler claims the event only
        when the click WAS on the canvas — the hint buttons and the page's
        own chrome keep their gestures.
        """
        if getattr(event, "button", 1) != 1:
            return
        if event.widget is not self._canvas:
            return
        cell = self._canvas_cell(event)
        if cell is None:
            return
        x, y = cell
        index = project_at(
            self._views,
            self._view,
            x,
            y,
            cursor=self._cursor,
            associated=self._associated,
        )
        if index is not None:
            event.stop()
            if getattr(event, "chain", 1) == 2:
                # The double-click IS `↵`: select first, then act — the
                # acting gesture moves the caret, so the hint and the action
                # cannot disagree about which row is current.
                self._select_index(index)
                self.action_jump()
                return
            self._select_index(index)
            return
        header = section_header_at(self._views, self._view, x, y, cursor=self._cursor)
        if header is not None:
            event.stop()
            self._jump_to_section(header[1])

    def _jump_to_section(self, target: int) -> None:
        """Header clicks land on the row the clicked band NAMES.

        The band carries its own target — the first row of the block the
        header heads, so a mixed team's timeline chart header lands on its
        first DATED row rather than an undated member's tail line (UX round 1,
        U3). Jumping by target (not by matching labels) is also what keeps
        two sections whose labels collide from crossing wires (R1-2/U1).
        """
        self._select_index(min(target, max(self._painted_count() - 1, 0)))

    # -- leaving ------------------------------------------------------------
    def action_leave(self) -> None:
        """``esc``: pop ONE level — form → canvas, detail → canvas keeps view and
        cursor — and only the canvas exits the mode (spec §1)."""
        if self._mode == "form":
            # The FORM owns its own cancel: a clean form closes, a dirty one
            # shows the inline discard confirm (spec §7.7). The view cannot
            # answer that question, so it asks the page.
            self._form_page.action_cancel_request()
            return
        if self._mode in ("compose", "send", "start"):
            # The send and start surfaces pop one level, exactly like the detail
            # (P5a/P5b): `esc cancel` on the band, `esc close` on either card.
            self.escape_surface()
            return
        if self._mode == "detail":
            # Leaving by hand drops any refusal/pop sentence with the page it
            # belonged to (UX round 1, U1/U5).
            self._notice = None
            self._exit_detail()
            return
        self._leave()

    def _leave_or_pop(self) -> None:
        """The `esc` HINT's action: the button must do what the key does."""
        if self._mode == "form":
            self._form_page.action_cancel_request()
            return
        if self._mode == "detail":
            # Leaving by hand drops any refusal/pop sentence with the page it
            # belonged to (UX round 1, U1/U5).
            self._notice = None
            self._exit_detail()
            return
        self._leave()

    # -- the form state (S6d parity P4) -------------------------------------
    def action_create(self) -> None:
        """``c``: open the create form (spec §3), from the canvas or the detail."""
        if self._mode in ("canvas", "detail"):
            self._enter_form()

    def _enter_form(self) -> None:
        """Show the create page, reset to a fresh set of fields.

        The reset is unconditional: `c` is a create, and a form that reopened
        holding the last abandoned draft would submit a stranger's leftovers
        (see ``ProjectsFormPage.reset``). The baseline a cancel returns to is
        taken from the reset state, so an untouched form is CLEAN — `esc` then
        closes immediately rather than asking about edits nobody made.
        """
        self._notice = None
        # The team field's hint says which names exist (spec §7.7): read on
        # entry from the registry `/team` uses, so it cannot drift from the
        # vocabulary the reader is actually allowed to write. The lookup lives
        # on the APP (it owns the team registry); a page mounted by a test host
        # that is not an ``OperatorApp`` simply has none, and the hint falls back
        # to its own honest sentence — the settings page's `getattr` rule.
        lookup = getattr(self.app, "_known_team_names", None)
        names: list[str] = []
        if callable(lookup):
            try:
                found = lookup()
                # Narrowed rather than assumed: ``getattr`` hands back an
                # untyped callable, and the hint must not depend on the host
                # answering with exactly a list.
                if isinstance(found, (list, tuple)):
                    names = [str(name) for name in found]
            except Exception:  # noqa: BLE001 — a hint must never fail a keypress
                names = []
        self._form_page.set_known_teams(names)
        #: Where a CANCEL returns to: a form opened from a project's own page
        #: goes back to that page (spec §1), while a SAVE always lands on a
        #: canvas — "cursor on the new project" only means something there.
        self._form_from = "detail" if self._mode == "detail" else "canvas"
        self._mode = "form"
        self._form_page.reset()
        self._form_page.display = True
        self._body.display = False
        self._detail_page.display = False
        # The canvas actions disarm against the new mode; the active binding
        # map is cached until this recomputes it (the app's recorded lesson).
        self.refresh_bindings()
        self._paint_chrome()
        self.call_after_refresh(self._paint_chrome)
        try:
            self._form_page.focus_first()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass
        self.call_after_refresh(self._form_page.arm_current)

    def _exit_form(self, *, back: str | None = None) -> None:
        """Leave the form for the state the reader came FROM (spec §1).

        A create is often started from a project's own page (`detail --c-->
        form`), and a cancel there puts the reader back on that page rather than
        on the canvas — the promise the entry made. A SAVE passes
        ``back="canvas"`` instead: the new project must be visible, and the
        spec's "cursor on the new project" is a canvas fact. When the
        remembered project has left the store, ``_resync_detail`` pops to the
        canvas and says why.
        """
        destination = back or self._form_from
        self._form_page.disarm_confirm()
        self._form_page.display = False
        self._mode = "canvas"
        self._body.display = True
        self._detail_page.display = False
        if destination == "detail" and self._detail_project_id:
            self._mode = "detail"
            self._body.display = False
            self._detail_page.display = True
            self._resync_detail()
        self.refresh_bindings()
        self._paint_chrome()
        self.call_after_refresh(self._paint_chrome)
        try:
            self.focus()
        except Exception:  # noqa: BLE001 — focus is a nicety
            pass

    def close_form(self) -> None:
        """Leave the form without writing — the page's own cancel route."""
        if self._mode == "form":
            self._exit_form()

    def form_created(self, project_id: str, name: str) -> None:
        """A create landed: leave the form, cursor on the new row, say so.

        The CANVAS the reader came from is kept (spec §7.7: "back to the view
        you came from") — ``focus_project`` would force the list canvas, which
        is a different promise — while the cursor moves to the new project so
        `↵` and `d` act on what was just made.
        """
        self._exit_form(back="canvas")
        for index, view_row in enumerate(self._views):
            project = view_row.get("project") if isinstance(view_row, dict) else None
            if isinstance(project, dict) and str(project.get("id") or "") == str(project_id):
                self._cursor = index
                break
        self._repaint()
        self._scroll_cursor_into_view()
        self.show_notice(f"created '{name}'")

    def show_form_refusal(self, text: str) -> None:
        """A STORE refusal, painted in the form (spec §7.7) — never a toast.

        The page stays open with the reader's values intact: the refusal is
        about one field, and losing the draft to read it would be the worse
        trade. The sentence is the store's own (``store_error_text``'s rule at
        the call site), so the form never invents a second wording for a rule
        the store owns.
        """
        self._form_page.show_refusal(text)
        self.call_after_refresh(self._paint_chrome)

    def _form_submitted(self, edit: Any) -> None:
        """The page's local validation passed — the APP performs the write."""
        self.post_message(ProjectsViewFormSubmitted(edit=edit))

    def _form_state_changed(self) -> None:
        """The form changed something the chrome states: re-arm it."""
        self.call_after_refresh(self._paint_chrome)

    def _form_focus_next(self) -> None:
        """The `tab` hint's action: the button must do what the key does."""
        self._form_page.action_focus_next_field()

    def _form_save(self) -> None:
        """The `ctrl+s` hint's action."""
        self._form_page.action_save()

    @property
    def wants_field_tab(self) -> bool:
        """True while the FORM MODE is up — the app's `shift+tab` asks.

        `shift+tab` is an app-wide PRIORITY binding (`cycle_effort`), so the
        focused field can never see the chord; the app asks this page instead of
        disarming every hotkey the way key capture does (see
        :meth:`form_focus_previous`).

        The claim is the WHOLE MESSAGE the mode is up, and that is the fix QA
        round 2 (Q-4) measured: narrowing it to "not confirming" made the app
        fall THROUGH the delegation for the chord, straight into
        `action_cycle_effort` — every `shift+tab` at the discard question moved
        a billable setting silently while a reader answered it. What must not
        happen is the journey to the next field, and that is the belt inside
        :meth:`form_focus_previous`, not the claim.

        ``@property`` is load-bearing rather than decorative: without it the app
        received the bound METHOD — always truthy — so the claim was never the
        one asked about (QA round 1, Q-2).
        """
        return self._mode == "form"

    def form_focus_previous(self) -> None:
        """``shift+tab`` — the app's priority binding delegates here.

        `shift+tab` is bound app-wide to ``cycle_effort`` with ``priority=True``
        (the settings page's recorded trap: an app priority binding is matched
        BEFORE the focused widget), so a form field can never see it. Rather
        than disarming every hotkey the way key CAPTURE does, the app asks this
        page first — the one key, delegated — which leaves ctrl+c and the rest
        of the app's bindings exactly where they were.
        """
        if self._mode == "form" and not self._form_page.confirming:
            self._form_page.focus_prev_field()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disarm the canvas actions while the FORM owns the page.

        Not a preference: `↑`/`↓` are not an ``Input``'s own keys, so they
        bubble past the focused field to THIS view, and the canvas must not
        move under a form the reader is filling in. ``esc`` is deliberately
        NOT in the set — it is the form's own way out — and neither is `c`,
        which is inert in form mode by its own guard.
        """
        if self._mode in ("form", "send", "compose", "start") and action in self._CANVAS_ACTIONS:
            return False
        return super().check_action(action, parameters)

    # -- the detail state (S6d parity P2) -----------------------------------
    def action_open_detail(self) -> None:
        """``d``: open the detail page for the selection (spec §3.2)."""
        if self._mode != "canvas":
            return
        self._enter_detail()

    # -- quick-send (S6d parity P5a) ---------------------------------------
    def action_message(self) -> None:
        """``m``: message a linked session, or ask which one (spec §7.5)."""
        if self._mode == "send":
            # `m` again re-targets rather than stacking a second card.
            self.close_send_picker()
            return
        if self._mode in ("form", "compose", "start"):
            return
        direct = self._detail_session_target()
        if direct is not None:
            self.begin_compose(direct)
            return
        self.open_send_picker()

    def _send_targets(self) -> list[SendTarget]:
        """The rows the picker offers, built ONCE for every caller.

        ``send_targets`` owns the order (manager first, live-first sessions,
        this session never a target), so a direct row send and the picker can
        never disagree about who is addressable.
        """
        row = self._views[self._cursor] if 0 <= self._cursor < len(self._views) else None
        view_row = row if isinstance(row, dict) else {}
        return send_targets(
            view_row,
            own_session=self._own_session,
            manager=self._manager_target,
        )

    def _detail_session_target(self) -> SendTarget | None:
        """The session row the cursor sits on, when the detail page is up.

        One keystroke from the row you are reading to a message to it; any
        other row asks for a target instead of guessing (spec §7.5.1).
        """
        if self._mode != "detail":
            return None
        payload = self._detail_page.selected_session()
        if not isinstance(payload, dict):
            return None
        session_id = str(payload.get("session_id") or "")
        if not session_id:
            return None
        for target in self._send_targets():
            if target.session_id == session_id:
                return target
        return None

    def open_send_picker(self) -> None:
        """Float the target card over the page and let it take the keys."""
        if self._send_card is not None:
            return
        rows = self._send_targets()
        card = SendTargetCard(rows, style_for=_style_resolver())
        self._send_card = card
        self._mode = "send"
        self.mount(card, before=self._title)
        # Placement is idempotent and cheap: once now (the regions it reads are
        # settled), once after the mount's first layout, and on every resize
        # through `on_resize` — one computation, recomputed, never accumulated.
        self._place_send_card()
        self.call_after_refresh(self._place_send_card)
        self.call_after_refresh(self._paint_chrome)

    def _place_send_card(self) -> None:
        """Float the send card over the canvas (design review round 1, D1/D2)."""
        self._place_card(self._send_card)

    def _place_start_card(self) -> None:
        """Float the start card over the same ground, by the same arithmetic.

        One placement rule for both cards on purpose: the two surfaces are the
        same family, the geometry is the part a reader notices when it drifts,
        and a second copy of this arithmetic is how the two would come to
        disagree about where a card sits (spec §5.6 draws them side by side).
        """
        self._place_card(self._start_card)

    def _place_card(self, card: Any | None) -> None:
        """Float ``card`` over the canvas (design review round 1, D1/D2).

        The anchor is the canvas's first painted row, and the row budget is
        everything from there to the page's own content bottom — so the card
        is always inside the page and never clipped (the failures D1 measured
        at 60x24 and 80x24). The card lives on the overlay layer, so moving it
        moves nothing else: no reflow, no scroll-region growth, no row taken
        from the canvas.
        """
        if card is None:
            return
        # THE GROUND IS THE SURFACE THAT IS ACTUALLY SHOWING, not the canvas
        # (UX review round 1, U1 — a BLOCKER). In detail mode the canvas body is
        # `display=False`, so its region is `[0,0,0,0]` and the card fell to the
        # `max(20, …)` floor: a 20-cell card whose rows wrapped onto three lines
        # over the detail prose, at every terminal width. `s` is documented as
        # working from BOTH entries, so one placement rule must read the ground
        # the reader is looking at.
        body = self._body if self._body.display else self._detail_page
        padding = body.styles.padding
        top = body.region.y + padding.top
        # The ground runs from the canvas's first row to the page's own content
        # bottom: while the card is up it may cover the footer strip and the
        # hint row below the canvas (its own legend then carries the grammar,
        # and the page keys are inert anyway with mode='send'/'start'), but it
        # can never reach the dock. Budgeting only the body clipped the list to
        # a single row at 60x24 and 80x24 — still inside the page, still
        # useless.
        ground = self.content_region.y + self.content_region.height - top
        card.set_available(ground)
        # D9 (design round 2): the card takes the width it can hold — the body's
        # whole content box. A 60-cell cap left 20 cells of page beside the card
        # at 80x24, level with the card's own last row, so the canvas fragment
        # `essions` read as part of its edge; a page row must never be visible
        # beside the card while the card owns those rows. The tcss carries no cap
        # either (a second one is how a width becomes invisible until a frame is
        # inspected).
        card.styles.width = max(20, body.region.width - padding.left - padding.right)
        # The OFFSET still hangs off the page's own content box: the card is a
        # child of ProjectsView (mounted before the title) and floats on the
        # overlay layer, so its position is relative to this widget, not to the
        # surface it covers. Only the WIDTH and the row budget come from the
        # displayed ground.
        content = self.content_region
        card.styles.offset = (
            body.region.x + padding.left - content.x,
            top - content.y,
        )

    # -- attachment row keys (P6, spec §7.4) --------------------------------
    def action_copy_attachment(self) -> None:
        """`y` — copy the selected attachment's stored path (spec §3.2/§7.4).

        Row-scoped: it acts only while an attachment row has the cursor, and is
        inert anywhere else (the ``action_zoom_in`` rule — a key that applies
        in one place no-ops everywhere it does not). A path is the payload on
        purpose: the copy under the store is what a reader pastes into a shell
        or a bug report, and it is the only handle they still have once the
        file itself is gone.

        The one refusal that is NOT a no-op is a pathless record: the reader
        pressed a key and the row cannot answer it, so the page says why — the
        path line already reads ``(no path recorded)`` and the sentence names
        the file that has none.
        """
        if self._mode != "detail":
            return
        attachment = self._detail_page.selected_attachment()
        if attachment is None:
            return
        name = str(attachment.get("name") or "(unnamed)")
        path = str(attachment.get("path") or "")
        if not path:
            self.show_notice(f"'{name}' has no stored path to copy")
            return
        self.post_message(
            ProjectsViewAttachmentCopied(
                path=path, name=name, project_name=self._detail_page.project_name
            )
        )

    def action_preview_attachment(self) -> None:
        """`space` — show or hide the selected image attachment (spec §7.4).

        The open/close half is pure VIEW state the page holds, so a re-press
        costs no read: only the FIRST press asks the host for bytes. A row
        that cannot be previewed answers with its reason rather than nothing,
        because the reader pressed the key expecting the picture — a data file
        has no pixels, and a copy that is gone has no bytes to read.
        """
        if self._mode != "detail":
            return
        attachment = self._detail_page.selected_attachment()
        if attachment is None:
            return
        path = str(attachment.get("path") or "")
        name = str(attachment.get("name") or "(unnamed)")
        if self._detail_page.selected_preview_open():
            self._detail_page.close_preview(path)
            return
        if not self._detail_page.selected_can_preview():
            if not path:
                self.show_notice(f"'{name}' has no stored path to preview")
            elif attachment.get("missing"):
                self.show_notice(
                    f"{name} is missing on disk — its stored copy was moved or deleted"
                )
            else:
                # A data file: the honest sentence names the verb that DOES
                # work for it instead of leaving the press unanswered.
                self.show_notice(f"{name} is not an image — ↵ opens it instead")
            return
        self.post_message(
            ProjectsViewAttachmentPreviewRequested(
                path=path, name=name, project_name=self._detail_page.project_name
            )
        )

    def show_attachment_preview(self, *, path: str, data_b64: str, mime_type: str) -> None:
        """Hand the page the bytes the app just read for ``path`` (spec §7.4).

        The app-side reader's answer, and the ONLY way a picture reaches the
        page: the file was read off the UI loop by the host, and this relays
        it without the page ever touching the filesystem.
        """
        self._detail_page.apply_preview(path, data_b64=data_b64, mime_type=mime_type)

    # -- start session (P5b, spec §7.6) -------------------------------------
    def set_start_rows(self, rows: list[StartTarget] | None) -> None:
        """Hand the page the picker's rows — the host's job, not the page's.

        The catalogues live behind the agent/team registries, which this widget
        has no business reaching; the app injects them (the same two
        projections the desktop pane's new-chat picker reads). ``None``
        restores the seeded plain row, which is what a host with no registry
        can honestly offer.
        """
        self._start_rows = (
            list(rows)
            if rows
            else [StartTarget(kind="plain", name="", label=PLAIN_LABEL, detail=PLAIN_DETAIL)]
        )

    def action_start(self) -> None:
        """`s` — start a session for the selected/current project (spec §3.2).

        Off the canvas the cursor's project is the one being worked on, and off
        the detail page it is the project the page is showing; both are
        ``current_project_id``, so one guard covers both.
        """
        if self._mode not in ("canvas", "detail"):
            return
        if self.current_project_id() is None:
            # Nothing selected (an empty store): there is no project to link a
            # session to and no snapshot to quote, so there is nothing to ask.
            return
        self.open_start_card()

    def open_start_card(self) -> None:
        """Float the start card over the page and let it take the keys."""
        if self._start_card is not None:
            return
        card = StartPickerCard(
            self._start_rows,
            project=self.current_project_name(),
            style_for=_style_resolver(),
        )
        self._start_card = card
        self._start_cancelled = False
        self._mode = "start"
        self.mount(card, before=self._title)
        self._place_start_card()
        self.call_after_refresh(self._place_start_card)
        self.call_after_refresh(self._paint_chrome)

    def close_start_picker(self) -> None:
        self._close_start_picker()

    def _close_start_picker(self) -> None:
        card = self._start_card
        if card is not None and card.pending:
            # THE READER CANCELLED A START THAT WAS ALREADY RUNNING (agent
            # review round 1, F5 / UX U: reproduced — `esc` on `starting
            # session …` closed the card and the reader was switched into the
            # new session anyway). The create itself is durable and keeps
            # going; what `esc` buys is that the app does NOT take them
            # anywhere, so the flag is consumed by the hand-off and the receipt
            # goes to the page instead. This matches the card's own documented
            # behaviour ("the create runs off the loop and its receipt lands on
            # the page") rather than contradicting it.
            self._start_cancelled = True
        self._start_card = None
        if card is not None:
            card.remove()
        if self._mode == "start":
            self._mode = "detail" if self._detail_page.display else "canvas"
        self._paint_chrome()

    @property
    def start_cancelled(self) -> bool:
        """Whether the reader dismissed the card while its create was running.

        A property rather than a one-shot consume: the app asks it TWICE — once
        before it sends the kickoff turn and once before it hands off, because
        the two are separate favours the reader may have cancelled and the flag
        must not be spent by the first answer. `open_start_card` resets it, so
        one start's cancellation can never silence the next one's hand-off.
        """
        return self._start_cancelled

    def start_pending(self) -> None:
        """The create is running off the loop (spec §7.6.2's pending state)."""
        if self._start_card is not None:
            self._start_card.set_pending()

    def start_refusal(self, sentence: str) -> None:
        """A refusal, said where the reader asked (in the card, not a toast).

        If the reader closed the card while the create was running the sentence
        goes to the page's own notice row instead: the outcome still has to
        land somewhere, and a receipt painted into a removed widget is a
        receipt nobody reads.
        """
        if self._start_card is not None:
            self._start_card.show_refusal(sentence)
            return
        self.show_notice(sentence)

    def on_start_picker_card_chosen(self, message: StartPickerCard.Chosen) -> None:
        """A row was picked: hand the whole question to the app (P5b).

        The page cannot boot a session — the registries, the creation core and
        the hand-off machinery are all the app's — so it re-posts one message
        naming what it knows: WHICH project and WHICH target.
        """
        message.stop()
        project_id = self.current_project_id()
        if project_id is None:
            return
        self.post_message(ProjectsViewStartRequested(project_id=project_id, target=message.target))

    def on_start_picker_card_closed(self, message: StartPickerCard.Closed) -> None:
        message.stop()
        self._close_start_picker()
        self._focus_canvas()

    def close_send_picker(self) -> None:
        self._close_send_picker()

    def _close_send_picker(self) -> None:
        card = self._send_card
        self._send_card = None
        if card is not None:
            card.remove()
        if self._mode == "send":
            self._mode = "detail" if self._detail_page.display else "canvas"
        self._paint_chrome()

    def begin_compose(self, target: SendTarget) -> None:
        """Hand the composer over, addressed to ``target`` (spec §7.5.2).

        The page does not own the composer, so this is a REQUEST; ``compose``
        mode is entered here because the page's own keys (and the ladder) have
        to reflect it immediately, and the app answers by giving the composer
        back and painting the recipient strip.
        """
        self._close_send_picker()
        self._send_target = target
        self._mode = "compose"
        self._notice = None
        self.post_message(ProjectsViewComposeChanged(target=target))
        self._paint_chrome()

    def end_compose(self) -> None:
        """``esc`` out of compose: no write, the target is dropped."""
        if self._mode != "compose":
            return
        self._send_target = None
        self._mode = "detail" if self._detail_page.display else "canvas"
        # Focus lands HERE, before the app answers the message: the composer is
        # about to go read-only, and `_set_composer_read_only` blurs a caret it
        # can no longer honour — with nothing taking the focus, `app.focused`
        # was None and the next `m` was a silent no-op until a Tab (QA round 1,
        # Q4). Taking it first also means the blur finds the editor already
        # unfocused, so it does not clear the focus a second time.
        try:
            self.focus()
        except Exception:  # noqa: BLE001 — an unmounted page has nothing to focus
            pass
        self.post_message(ProjectsViewComposeChanged(target=None))
        self._paint_chrome()

    @property
    def composing(self) -> bool:
        return self._mode == "compose"

    @property
    def compose_target(self) -> SendTarget | None:
        return self._send_target

    def submit_compose(self, text: str) -> bool:
        """The composer's submit while composing — true when the page took it.

        An empty body is refused HERE, in-surface: nothing is dialled and the
        draft rule is untouched. Anything else is the app's to deliver.
        """
        if self._mode != "compose" or self._send_target is None:
            return False
        body = text.strip()
        if not body:
            self.show_notice("nothing to send — type a message first")
            return True
        self.post_message(ProjectsViewSendRequested(target=self._send_target, text=body))
        return True

    def compose_receipt(self, sentence: str) -> None:
        """Report a send's outcome in the surface the reader is looking at.

        ONE branch (agent review F7 dropped the dead ``ok`` flag two identical
        calls used to take; agent review F4 moved the DELIVERED receipt to the
        composer's band, so what lands here is amber and refused outcomes).
        The page's notice line is right for both: the reader is still on the
        page, and a late receipt for a compose they already left still has to
        be said somewhere the page can see.
        """
        self.show_notice(sentence)

    def on_send_target_card_chosen(self, message: SendTargetCard.Chosen) -> None:
        message.stop()
        if self._send_card is not None and message.card is not self._send_card:
            return
        self.begin_compose(message.target)

    def on_send_target_card_closed(self, message: SendTargetCard.Closed) -> None:
        message.stop()
        if self._send_card is not None and message.card is not self._send_card:
            return
        self._close_send_picker()

    def _enter_detail(self) -> None:
        view_row = self._detail_view_row()
        if view_row is None:
            return
        project_value = view_row.get("project")
        project = project_value if isinstance(project_value, dict) else {}
        self._detail_project_id = str(project.get("id") or "") or None
        self._notice = None
        self._mode = "detail"
        self._detail_page.show(view_row, own_session=self._own_session, style_for=_style_resolver())
        self._body.display = False
        self._detail_page.display = True
        self._paint_chrome()
        # The footer's own box width is only final after layout, and the
        # hint arming reads the settled scroll geometry — the same deferred
        # pair every repaint schedules.
        self.call_after_refresh(self._paint_chrome)
        # And the row cursor's reveal, again: `show` revealed while the page
        # was still hidden (its rows had no regions), so at narrow widths the
        # selection landed a row below the fold (measured at 60x24).
        self.call_after_refresh(self._detail_page.reveal_selected)
        try:
            self._detail_page.focus()
        except Exception:
            pass  # focus is a nicety; the keys bubble to this view either way

    def _exit_detail(self) -> None:
        self._mode = "canvas"
        self._detail_page.display = False
        self._body.display = True
        self._paint_chrome()
        self.call_after_refresh(self._paint_chrome)
        try:
            self.focus()
        except Exception:
            pass

    def _resync_detail(self) -> None:
        """Keep the detail page open across a recomposition, by project ID.

        A refresh must not bounce the reader back to the canvas (they may be
        reading the description), and a rename may have moved the project's
        index — so the row is found by id, the cursor follows it, and only a
        project that is GONE pops back to the canvas.
        """
        detail_id = self._detail_project_id
        index: int | None = None
        for position, view_row in enumerate(self._views):
            project = view_row.get("project") if isinstance(view_row, dict) else None
            if isinstance(project, dict) and str(project.get("id") or "") == detail_id:
                index = position
                break
        if index is None:
            name = self._detail_page.project_name or "that project"
            # The pop says WHY — the reader pressed `r` on a page whose
            # project left the store underneath it (UX round 1, U5).
            self._notice = f"'{name}' is no longer in the store — back to the canvas."
            self._exit_detail()
            return
        self._cursor = index
        self._detail_page.show(
            self._views[index],
            own_session=self._own_session,
            selected=self._detail_page.selected_index,
            style_for=_style_resolver(),
        )

    def _focus_detail_page(self) -> None:
        """The `↑↓` hint's click target: focus the page (a void wrapper — the
        hint's action type is `() -> None`, and ``focus()`` returns the widget)."""
        self._detail_page.focus()

    def _detail_nav(self, delta: int) -> None:
        """The ONE row-cursor path: move the page, then repaint the chrome.

        The page's own up/down bindings, the view's arrow actions and any
        click-driven move all land here, so the `↵ <verb>` hint label and
        the ruler can never trail the cursor (the page binding consumes the
        arrows before an ancestor sees them — the measured reason the route
        exists).
        """
        self._detail_page.move(delta)
        self._paint_chrome()

    def _detail_view_row(self) -> dict[str, Any] | None:
        if 0 <= self._cursor < len(self._views) and isinstance(self._views[self._cursor], dict):
            return self._views[self._cursor]
        return None

    def _detail_row_action(self, kind: str, row: dict[str, Any]) -> None:
        """Relay one row activation to the host — the page never acts alone.

        Sessions reuse the shipped conversation ladder by posting the SAME
        message the canvas does, scoped to the one row; milestones post the
        toggle the app answers with the same store core the tool uses.

        The target is the PAGE's own provenance — the id and name its rows
        were built from in ``DetailPage.show`` — never the canvas cursor's
        current row: a retarget mid-read once split the page, and `↵` then
        wrote through the cursor into a project the row never came from
        (QA round 1, Q1). Bound to the build snapshot, that write is
        impossible by construction.
        """
        project_id = self._detail_page.project_id or ""
        if kind == "session":
            session_id = str(row.get("session_id") or "")
            if not session_id:
                return
            self.post_message(
                ProjectsViewJumpRequested(
                    project_id=project_id,
                    project_name=self._detail_page.project_name or "(unnamed)",
                    sessions=((session_id, detail_session_state(row)),),
                )
            )
            return
        if kind == "milestone":
            name = str(row.get("name") or "")
            if not name:
                return
            self.post_message(
                ProjectsViewMilestoneToggled(
                    project_id=project_id,
                    name=name,
                    completed=not bool(row.get("completed_at")),
                    project_name=self._detail_page.project_name,
                )
            )
            return
        if kind == "attachment":
            # The attachment's own row, not the page's provenance: the file is
            # the thing being opened, and the message carries it by path.
            path = str(row.get("path") or "")
            self.post_message(
                ProjectsViewAttachmentOpened(
                    path=path,
                    name=str(row.get("name") or "(unnamed)"),
                    project_name=self._detail_page.project_name,
                )
            )

    def _leave(self) -> None:
        self.post_message(ProjectsViewDismissed())
