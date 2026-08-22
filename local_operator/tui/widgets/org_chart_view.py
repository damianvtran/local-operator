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
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("left", "scroll_left", "Scroll left", show=False),
        Binding("right", "scroll_right", "Scroll right", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
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
        #: Last render, kept for the geometry probes and rendered_rows().
        self._last: RenderResult | None = None
        self._title = Static(classes="org-chart-view-title")
        self._rule = Static(classes="org-chart-view-rule")
        # A Static inside a both-axes scroll container: the Static is sized to
        # the painted canvas, the container clips and scrolls it. Virtual size
        # equals the canvas size, so Textual's own scrollbars appear when the
        # org exceeds the viewport.
        self._canvas = Static(classes="org-chart-view-canvas")
        self._body = ScrollableContainer(self._canvas, classes="org-chart-view-body")
        # Footer hints, same vocabulary as the subagent view so the two modes
        # read consistently. Each is its own widget so it can be hovered/clicked.
        self._zoom_hint = HintButton("+/-", self._cycle_zoom)
        self._fit_hint = HintButton("f", lambda: self.action_fit())
        self._expand_hint = HintButton("e", lambda: self.action_toggle_expand())
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
        yield self._body
        with self._hints:
            yield self._zoom_hint
            yield self._fit_hint
            yield self._expand_hint
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
        self._paint_hints()

    def _paint_hints(self) -> None:
        plan = [
            (self._zoom_hint, " zoom", False),
            (self._fit_hint, " fit", True),
            (self._expand_hint, " expand", True),
            (self._exit_hint, "back to conversation", True),
            (self._state_hint, "", True),
        ]
        for hint, label, lead in plan:
            hint.paint(label, lead=lead)

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

    def action_fit(self) -> None:
        """Pick the coarsest tier whose canvas fits the viewport width.

        Fit-to-WIDTH because the chart is wide and the horizontal axis is the
        one that overflows first. Tries outline→detailed and keeps the first
        that fits; if none fits (a huge org) it lands on outline and scroll
        absorbs the rest.
        """

        if self._root is None:
            return
        viewport = max(self._body.size.width, 1)
        chosen: Tier = 0
        for tier in (0, 1, 2):
            result = render_org(
                self._root, tier=tier, expand_all=self._expand_all, style_for=_style_resolver()
            )
            if result.width <= viewport:
                chosen = tier  # type: ignore[assignment]
            else:
                break
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

    def action_scroll_home(self) -> None:
        self._body.scroll_home()

    def action_scroll_end(self) -> None:
        self._body.scroll_end()

    # -- leaving ------------------------------------------------------------
    def action_leave(self) -> None:
        self._leave()

    def _leave(self) -> None:
        self.post_message(OrgChartViewDismissed())
