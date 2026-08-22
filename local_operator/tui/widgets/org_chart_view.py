"""The full-page org-chart view — a team's org rendered as a tidy tree.

A MODE of the main screen, modelled on :class:`SubagentView`: it takes the
transcript's region and leaves the dock (band, status, composer) where it was,
greyed, so the page reads as the same app looking somewhere else rather than a
modal over it. The design weighed a pushed Screen (blacks out the dock) and a
floating card (cannot scroll a large canvas or hold focus for zoom keys) and
chose this for the same reasons the subagent redesign did.

WHAT IS DIFFERENT FROM THE SUBAGENT VIEW
========================================

- The body scrolls BOTH axes. An org chart is wide, so horizontal scroll is
  required — unlike the subagent view (vertical-only) and unlike the main
  screen (where a horizontal scrollbar is always a bug). The tcss allows
  ``overflow: auto`` on this body precisely because the wide canvas is
  legitimate here.
- Content is a single painted :class:`~rich.text.Text` from
  :func:`local_operator.tui.org_render.render_org`, not a live transcript. The
  widget re-renders on a zoom/expand change (cheap: the tree is already
  resolved; only box widths and which rows draw change) and never on scroll.
- Three ZOOM tiers on ``+``/``-`` (outline/standard/detailed), whole-canvas
  expand-all toggle, ``f`` fit-to-width (coarsest tier that fits), arrow/page
  scroll that CLAMPS (a canvas, not a list — a wrapping scroll reads as the
  canvas resetting itself), Esc to leave.

Identified by CLASS and not id, all the way down — the ``DuplicateIds``-on-
fast-reopen lesson the subagent view records: ``remove()`` only POSTS a prune,
so a reopen inside that window would mount a second same-id widget and raise
out of a click handler. An observability surface may not take the app down.
"""

from __future__ import annotations

from typing import Any, Callable

from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.widgets import Static

from local_operator.org_chart import OrgNode, resolve_org
from local_operator.tui import theme as theme_mod
from local_operator.tui.org_render import RenderResult, Tier, render_org
from local_operator.tui.widgets.subagent_view import READ_ONLY_NOTE, HintButton

#: Tier index → the word shown in the title. The title always states the tier
#: so "why is this collapsed?" is answered on screen (a small terminal
#: auto-selects the outline tier).
_TIER_NAME = {0: "outline", 1: "standard", 2: "detailed"}


class OrgChartViewDismissed(Message):
    """The page's ``esc`` hint was clicked. The app owns leaving the mode.

    A DEDICATED message (not the subagent view's) so the app's Esc-chain routes
    the click to :meth:`_close_org_chart_view` — reusing the subagent message
    would hit that mode's handler, which does not own this widget.
    """


def _style_resolver() -> Callable[[str], Style]:
    """A style-key → ``rich.Style`` resolver bound to the CURRENT theme.

    Rebuilt on each render so a theme switch while the chart is open repaints
    in the new palette. The keys are the ones ``org_render`` paints:
    ``team``/``manager``/``agent``/``ghost``/``marker``/``connector``.
    """

    def color(token: str) -> str:
        return theme_mod.semantic_color(token)

    styles = {
        # A team boundary is the structural spine — bright and bold.
        "team": Style(color=color("fg"), bold=True),
        # The manager reads as the accent of its team.
        "manager": Style(color=color("accent"), bold=True),
        # An ordinary agent leaf: legible but not competing with the spine.
        "agent": Style(color=color("muted")),
        # A missing ref is a dim ghost so the GAP is visible, not dropped.
        "ghost": Style(color=color("dim"), italic=True),
        # Cycle/depth markers are annotations, tinted like a warning so they
        # read as "the resolver stopped here" rather than as real structure.
        "marker": Style(color=color("warning")),
        # Connectors recede behind the boxes they join.
        "connector": Style(color=color("dim")),
        # Per-team boundary rules (D1) are GROUPING, not structure — painted
        # fainter than a connector (dim + 55% opacity) so they demarcate a
        # team's members at a glance without competing with the tree's spine.
        "rule": Style(color=color("dim"), dim=True),
    }

    def resolve(key: str) -> Style:
        return styles.get(key, Style())

    return resolve


class OrgChartView(Vertical):
    """The page: a title, a rule, the scrollable chart canvas, and the way out.

    Class-identified (see module docstring). ``can_focus`` so the zoom/scroll
    keys land here rather than on the composer the mode made inert.
    """

    can_focus = True

    # Arrow keys scroll the canvas by a line and CLAMP (no wrap) — this is a
    # canvas, not a list, so a wrapping arrow would teleport across the chart.
    # Zoom on +/- (and =, the unshifted +). Expand-all, fit, and esc round it
    # out. Bindings show=False: the footer hints are the visible affordance,
    # the same split the subagent view uses.
    BINDINGS = [
        Binding("plus,equals_sign,equal", "zoom_in", "Zoom in", show=False),
        Binding("minus,underscore", "zoom_out", "Zoom out", show=False),
        Binding("f", "fit", "Fit to width", show=False),
        Binding("e,space,enter", "toggle_expand", "Expand/collapse all", show=False),
        # `?` reveals a one-line glyph legend (U5) and the `(declared)` gloss
        # (U6) — the vocabulary (◆ ↩ ? ⋯ ·N ×N) is otherwise learnable only
        # from source. Toggled, not always-on, so it costs no row until asked.
        Binding("question_mark,question", "toggle_legend", "Legend", show=False),
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("left", "scroll_left", "Scroll left", show=False),
        Binding("right", "scroll_right", "Scroll right", show=False),
        # Vertical paging on PageUp/Down; HORIZONTAL paging on shift+arrows,
        # because the chart overflows the wide axis far more than the tall one
        # (U4) and single-cell arrows alone made the right edge ~58 presses
        # away. Shift+left/right move a viewport-width at a time.
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("shift+left", "page_left", "Page left", show=False),
        Binding("shift+right", "page_right", "Page right", show=False),
        # Home/End jump to the CORNERS (top-left / bottom-right). End reaching
        # the bottom-right is what gives a keyboard jump to the RIGHT edge —
        # the axis that actually overflows — which a y-only End could never do
        # on a chart whose max_scroll_y is 0 (U4).
        Binding("home", "scroll_home", "To start", show=False),
        Binding("end", "scroll_end", "To end", show=False),
        Binding("escape", "leave", "Back", show=False),
    ]

    def __init__(self, team_name: str) -> None:
        super().__init__(classes="org-chart-view")
        self._team_name = team_name
        #: The resolved org tree, held so a zoom/expand re-render never
        #: re-resolves (resolution reads the registry; a repaint must not).
        self._root: OrgNode | None = None
        #: Current zoom tier and whether the whole canvas is force-expanded.
        self._tier: Tier = 1
        self._expand_all = False
        #: Whether the one-shot §5.5 auto-fit has run. The body has no real
        #: width until it is laid out, so the fit that picks the opening tier
        #: has to wait for the first ``on_resize``; this guards it to fire once,
        #: so a user's later manual zoom is never overridden on a resize.
        self._auto_fitted = False
        #: Whether the glyph legend line (U5/U6) is showing. Off by default so
        #: it costs no vertical row until a user presses ``?``.
        self._legend_open = False
        #: Last render, kept for the geometry probes and rendered_rows().
        self._last: RenderResult | None = None
        self._title = Static(classes="org-chart-view-title")
        self._rule = Static(classes="org-chart-view-rule")
        #: The `?`-toggled glyph legend (U5/U6). A Static that is display:none
        #: until opened, so it reserves no row in the resting layout.
        self._legend = Static(classes="org-chart-view-legend")
        # A Static inside a both-axes scroll container: the Static is sized to
        # the painted canvas, the container clips and scrolls it. Virtual size
        # equals the canvas size, so Textual's own scrollbars appear when the
        # org exceeds the viewport.
        self._canvas = Static(classes="org-chart-view-canvas")
        self._body = ScrollableContainer(self._canvas, classes="org-chart-view-body")
        # Footer hints, same vocabulary as the subagent view so the two modes
        # read consistently. Each is its own widget so it can be hovered/clicked.
        # Scroll is the headline affordance on a chart that overflows both axes
        # (U1) — advertised first so a keyboard user learns arrows/Page scroll
        # and a mouse user is not left hunting the thin scrollbar. `↔↕` names
        # BOTH axes because the chart is wide, so horizontal scroll is the
        # travel that matters, not an afterthought.
        # Clicking the scroll hint just focuses the canvas so the arrow keys
        # land here; routed through a None-returning helper because ``focus()``
        # returns ``self``, which the ``Callable[[], None]`` action type rejects.
        self._scroll_hint = HintButton("↔↕", self._focus_canvas)
        self._zoom_hint = HintButton("+/-", self._cycle_zoom)
        self._fit_hint = HintButton("f", lambda: self.action_fit())
        self._expand_hint = HintButton("e", lambda: self.action_toggle_expand())
        self._legend_hint = HintButton("?", lambda: self.action_toggle_legend())
        self._exit_hint = HintButton("esc", self._leave)
        self._state_hint = HintButton(READ_ONLY_NOTE)
        self._hints = Horizontal(classes="org-chart-view-hints")
        self._title_text = Text()
        self._rule_text = Text()

    @property
    def team_name(self) -> str:
        """The team this page is charting."""
        return self._team_name

    def compose(self):  # type: ignore[override]
        yield self._title
        yield self._rule
        yield self._legend
        yield self._body
        with self._hints:
            yield self._scroll_hint
            yield self._zoom_hint
            yield self._fit_hint
            yield self._expand_hint
            yield self._legend_hint
            yield self._exit_hint
            yield self._state_hint

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
        # layout knows, so both are repainted on resize. The canvas itself is
        # width-independent (it scrolls), so only the chrome moves.
        self._paint_chrome()
        # §5.5 auto-fit: on the FIRST layout (when the body finally has a real
        # width — it is 0 until mounted, so this cannot run in on_mount), pick a
        # tier that fits rather than always opening at standard. Runs once so a
        # user who then zooms by hand is not overridden on the next resize. It
        # uses the U2-aware ``action_fit``, so a small terminal lands on the
        # coarsest tier that still shows members and only falls to outline when
        # nothing else fits — never silently collapsing the roster to a box.
        if not self._auto_fitted and self._root is not None and self._body.size.width > 0:
            self._auto_fitted = True
            self.action_fit()

    # -- data ---------------------------------------------------------------
    def show(self, root: OrgNode) -> None:
        """Point the page at a resolved org tree and paint it."""
        self._root = root
        self._repaint()

    def load(self, *, teams: Any, agents: Any = None) -> None:
        """Resolve THIS page's team from a registry and paint it.

        Kept beside :meth:`show` so a caller can either hand a pre-resolved
        tree (tests, a shot script) or ask the page to resolve its own name.
        """

        self._root = resolve_org(self._team_name, teams=teams, agents=agents)
        self._repaint()

    # -- rendering ----------------------------------------------------------
    def _repaint(self) -> None:
        if self._root is None:
            return
        result = render_org(
            self._root,
            tier=self._tier,
            expand_all=self._expand_all,
            style_for=_style_resolver(),
        )
        self._last = result
        self._canvas.update(result.text)
        # Pin the Static to the painted canvas size so the ScrollableContainer's
        # virtual size equals the canvas (Textual scrolls the difference). The
        # +1 columns of width the Static reserves nothing extra — the canvas
        # already includes its one-cell margins.
        self._canvas.styles.width = result.width
        self._canvas.styles.height = result.height
        self._paint_chrome()

    def _paint_chrome(self) -> None:
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        # `team <name> · org chart · zoom: <tier>` — the tier is always stated so
        # an auto-collapsed small terminal explains itself.
        title = Text(no_wrap=True, overflow="ellipsis")
        head = Style(color=theme_mod.semantic_color("fg"), bold=True)
        title.append(f"team {self._team_name}", style=head)
        title.append(" · org chart", style=muted)
        title.append(f" · zoom: {_TIER_NAME[self._tier]}", style=dim)
        if self._expand_all:
            title.append(" · expanded", style=dim)
        # A team-boundary tag: the chart renders the DECLARED org; the runtime
        # that makes a nested team executable is a follow-up, so the page says
        # so rather than implying a capability that is not live.
        if self._root is not None and any(c.kind == "team" for c in self._root.children):
            title.append(" · (declared)", style=dim)
        self._title_text = title
        self._title.update(title)
        width = max(self.size.width - 2, 1)
        self._rule_text = Text("─" * width, style=dim)
        self._rule.update(self._rule_text)
        # Legend line (U5/U6): the glyph vocabulary + the `(declared)` gloss,
        # shown only while toggled on. Painted here so a theme switch or resize
        # repaints it in the current palette. It is display:none when closed, so
        # the resting layout is unchanged.
        self._legend.display = self._legend_open
        if self._legend_open:
            # TWO rows (D7): the glyph vocabulary is 70 cells and the (declared)
            # gloss is 56 — on one line they total 129 and the gloss, the very
            # thing U6 added to explain the tag, was the part ellipsis-clipped at
            # ≤~128 cols. Splitting them keeps each row well under a standard
            # 100-col terminal, so nothing essential is ever truncated.
            legend = Text(no_wrap=True, overflow="ellipsis")
            legend.append("◆ manager", style=Style(color=theme_mod.semantic_color("accent")))
            for glyph, meaning in (
                ("? unresolved", "ghost"),
                ("↩ cycle", "marker"),
                ("⋯ depth-limit", "marker"),
                ("·N members", None),
                ("×N copies", None),
            ):
                legend.append("  ", style=dim)
                style = (
                    Style(color=theme_mod.semantic_color("warning")) if meaning == "marker" else dim
                )
                legend.append(glyph, style=style)
            # (declared) gloss (U6) on its OWN row so it always reads in full.
            legend.append("\n", style=dim)
            legend.append(
                "(declared) = shown org, not yet an executable delegation",
                style=dim,
            )
            self._legend.update(legend)
        self._paint_hints()

    def _paint_hints(self) -> None:
        """Lay out the footer hints, shedding WHOLE hints until the row fits.

        D3 — the row used to render as one over-wide string that the terminal
        clipped mid-word ("…back to conversatio") at 60 columns. Instead the
        row sheds whole hints widest-first: the affordances a reader needs most
        (scroll, zoom, esc) survive, and `esc` is never dropped because it is
        the only way out. Each rung is measured before it is committed, so the
        painted row is always one that fits.
        """

        # (visible hints with their labels, esc label) per rung, widest first.
        # `esc` and `scroll` are the survivors; `read-only`, then the secondary
        # controls, shed as the width tightens. Each rung is built from a list
        # of the LEADING hints plus esc (always present, so a page always says
        # how to leave) plus optionally the read-only note. Building each rung
        # explicitly — rather than slicing one list — is what guarantees esc is
        # never sliced off along with the tail (the bug a `full[:-2]` hid).
        scroll = (self._scroll_hint, " scroll", False)
        zoom = (self._zoom_hint, " zoom", True)
        fit = (self._fit_hint, " fit", True)
        expand = (self._expand_hint, " expand", True)
        keys = (self._legend_hint, " keys", True)

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

        rungs: list[tuple[list[tuple[HintButton, str, bool]], str]] = [
            rung([scroll, zoom, fit, expand, keys], "back to conversation", state=True),
            rung([scroll, zoom, fit, expand, keys], "back to conversation", state=False),
            rung([scroll, zoom, fit, expand, keys], "back", state=False),
            rung([scroll, zoom, fit, expand], "back", state=False),  # drop keys
            rung([scroll, zoom], "back", state=False),  # drop fit/expand
            rung([scroll], "back", state=False),  # scroll + esc
            rung([], "back", state=False),  # esc alone
        ]
        width = max(self.size.width - 2, 1)
        chosen = rungs[-1]
        for plan, esc_label in rungs:
            measured = self._measure_hints(plan, esc_label)
            if measured <= width:
                chosen = (plan, esc_label)
                break
        plan, esc_label = chosen
        visible = {hint for hint, _label, _lead in plan}
        for hint, label, lead in plan:
            hint.paint(esc_label if hint is self._exit_hint else label, lead=lead)
        # Hide the shed hints so the row is exactly what was measured.
        for hint in (
            self._scroll_hint,
            self._zoom_hint,
            self._fit_hint,
            self._expand_hint,
            self._legend_hint,
            self._exit_hint,
            self._state_hint,
        ):
            hint.display = hint in visible

    def _measure_hints(self, plan: list[tuple[HintButton, str, bool]], esc_label: str) -> int:
        """Cell width of a candidate hint row, measured before it is painted."""
        from rich.cells import cell_len

        row = Text()
        for hint, label, lead in plan:
            row.append(hint.preview(esc_label if hint is self._exit_hint else label, lead=lead))
        return cell_len(row.plain)

    def rendered_rows(self) -> list[str]:
        """The page as plain strings — title, rule, canvas rows. Assertable."""
        rows = [self._title_text.plain, self._rule_text.plain]
        if self._last is not None:
            rows.extend(self._last.text.plain.split("\n"))
        return rows

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

    # -- zoom / expand ------------------------------------------------------
    def _set_tier(self, tier: int) -> None:
        tier = max(0, min(2, tier))
        if tier == self._tier:
            return
        self._tier = tier  # type: ignore[assignment]
        self._repaint()

    def action_zoom_in(self) -> None:
        # "In" = MORE detail = higher tier index.
        self._set_tier(self._tier + 1)

    def action_zoom_out(self) -> None:
        self._set_tier(self._tier - 1)

    def _cycle_zoom(self) -> None:
        # The +/- hint click cycles through tiers (wrapping at the top): a
        # single click target cannot mean both directions, so it steps up and
        # wraps, the way a mouse-only user reaches every tier.
        self._set_tier(0 if self._tier >= 2 else self._tier + 1)

    def action_toggle_expand(self) -> None:
        self._expand_all = not self._expand_all
        self._repaint()

    def _focus_canvas(self) -> None:
        """Focus the view so the arrow/scroll keys land here (scroll-hint click)."""
        self.focus()

    def action_toggle_legend(self) -> None:
        # Show/hide the glyph legend line (U5/U6). Only the chrome changes, so a
        # repaint of the chrome is enough — the canvas is untouched.
        self._legend_open = not self._legend_open
        self._paint_chrome()

    def action_fit(self) -> None:
        """Fit to width WITHOUT hiding the roster — never auto-collapse to a box.

        Fit-to-WIDTH because the chart overflows the wide axis first. The naive
        "coarsest tier that fits" collapsed any wide flat team to outline, where
        members fold into a ``·N`` badge — so pressing "fit" to SEE the org gave
        strictly less than before the press (U2). The fix: ``f`` only ever
        chooses among the MEMBER-SHOWING tiers (standard, detailed):

        - Pick the coarsest of {standard, detailed} that fits (standard first —
          less to read when both fit).
        - If NEITHER fits (a genuinely wide org), land on STANDARD and let
          horizontal scroll carry the rest. Standard still draws every member
          box, so "fit" always leaves the roster visible; the reader scrolls to
          the ones off-screen rather than watching them vanish.

        Outline (tier 0) is never chosen by ``f`` — it is a deliberate ``-``
        zoom the user asks for, where the ``·N ?`` badge summarises the roster
        (member count + ghost flag) legibly. That split is what makes ``f``
        honest: it may zoom out one step, but never past where the people are.
        """

        if self._root is None:
            return
        viewport = max(self._body.size.width, 1)
        chosen: Tier = 1  # standard + scroll is the floor: members always shown
        for tier in (1, 2):
            result = render_org(
                self._root, tier=tier, expand_all=self._expand_all, style_for=_style_resolver()
            )
            if result.width <= viewport:
                chosen = tier  # type: ignore[assignment]
                break  # coarsest-first: standard wins when it fits
        self._set_tier(chosen)

    # -- scrolling (all CLAMP; no wrap on a canvas) -------------------------
    def action_scroll_up(self) -> None:
        self._body.scroll_up()

    def action_scroll_down(self) -> None:
        self._body.scroll_down()

    def action_scroll_left(self) -> None:
        self._body.scroll_left()

    def action_scroll_right(self) -> None:
        self._body.scroll_right()

    def action_page_up(self) -> None:
        self._body.scroll_page_up()

    def action_page_down(self) -> None:
        self._body.scroll_page_down()

    def action_page_left(self) -> None:
        # Horizontal paging: the wide axis, reached in a few keystrokes (U4).
        self._body.scroll_page_left()

    def action_page_right(self) -> None:
        self._body.scroll_page_right()

    def action_scroll_home(self) -> None:
        # Jump to the top-LEFT corner, so a wide-scrolled reader returns to the
        # root box in one key. Explicit x=0,y=0 rather than ``scroll_home``:
        # Textual's ``scroll_home`` only resets the Y axis unless ``x`` is
        # passed, so on a wide-scrolled chart it would leave the reader off to
        # the right. ``scroll_to`` pins BOTH axes to the origin.
        self._body.scroll_to(x=0, y=0, animate=False)

    def action_scroll_end(self) -> None:
        # Jump to the bottom-RIGHT corner — the wide axis is the one that
        # overflows (U4), so this is the keyboard jump to the right edge even
        # when there is no vertical travel. Explicit maxima via ``scroll_to``,
        # NOT ``scroll_end``: on this Textual (8.2.8) ``scroll_end`` passes x=0
        # (its "end" is the bottom-LEFT), so on a wide-only chart it was a
        # no-op — never the right edge. Pinning ``max_scroll_x``/``max_scroll_y``
        # reaches the true far corner; the container clamps each to its own
        # travel, so an axis with no overflow simply stays at 0.
        self._body.scroll_to(
            x=self._body.max_scroll_x,
            y=self._body.max_scroll_y,
            animate=False,
        )

    # -- leaving ------------------------------------------------------------
    def action_leave(self) -> None:
        self._leave()

    def _leave(self) -> None:
        self.post_message(OrgChartViewDismissed())
