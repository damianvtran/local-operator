"""The ``/analytics /usage`` screen: aggregated token consumption, Esc to close.

This is the read side of the analytics feature. It queries the shared ledger
(:class:`local_operator.analytics.AnalyticsStore`) for a summed view across
every session on the machine and renders it as a scrollable, Esc-dismissable
full-screen overlay — the same ``ModalScreen`` shape as ``/resume``, so the one
surface the user did not build looks like the rest of the app.

What it shows, top to bottom:

- **Totals** — every token the providers billed, the exact thinking/generation
  split of output, and the cache hit rate. These are AUTHORITATIVE: they come
  straight off the provider usage numbers, not an estimate.
- **Where the input went** — the estimated breakdown of context tokens across
  the system prompt, custom instructions (agent/team profiles), tool inventory,
  tool schemas, environment, knowledge, conversation, and tool results. Marked
  as an estimate because the provider bills one input total and the split is
  apportioned by character length.
- **By provider** — the same totals grouped per provider/model source.
- **By session** — per-session context spend, named where a title is known.

The renderer here is a set of pure functions returning ``rich.text.Text`` so
the content can be asserted as plain strings in a test
(``render_lines_for_test``), exactly like ``usage_panel`` and
``session_picker`` do — a passing test is not evidence a TUI looks right, but
it is the right way to pin what the screen SAYS.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    COMPONENT_LABELS,
    UsageAggregate,
    short_session_label,
)
from local_operator.tui import theme as theme_mod


def format_tokens(n: int) -> str:
    """Compact token count: ``912`` / ``3.4k`` / ``1.2M`` / ``4.1B``.

    Analytics totals cross from a handful of tokens on a fresh install to
    billions on a long-lived machine, so the headline numbers are abbreviated
    the way ``/usage`` abbreviates its amounts — a raw ``1204331902`` is
    unreadable at a glance and the whole point of this screen is the glance.
    """
    n = int(n)
    if n < 1000:
        return str(n)
    # Compare the ROUNDED value to each ceiling, not the raw one: 999_950
    # rounds to "1000.0k" under a raw ``n < 1_000_000`` check (review A5), so a
    # number that rounds up to the next unit is promoted to that unit here.
    for divisor, suffix, ceiling in (
        (1000, "k", 1_000_000),
        (1_000_000, "M", 1_000_000_000),
    ):
        if n < ceiling and round(n / divisor, 1) < ceiling / divisor:
            return f"{n / divisor:.1f}{suffix}".replace(f".0{suffix}", suffix)
    return f"{n / 1_000_000_000:.1f}B".replace(".0B", "B")


def format_percent(fraction: float | None) -> str:
    """``73%`` / ``—`` for an unmeasurable rate.

    ``100%`` is reserved for a genuinely complete rate (review D4): a value that
    is merely close — 99.6% — floors to ``99%`` rather than rounding up to a
    flat ``100%`` that reads as mocked or broken next to the cache-read total it
    is derived from. Only an exact 1.0 prints ``100%``.
    """
    if fraction is None:
        return "—"
    pct = fraction * 100
    if 99 < pct < 100:
        return "99%"
    return f"{round(pct)}%"


def format_cost(aggregate: "UsageAggregate") -> str:
    """A dollar figure for one scope: ``$12.34`` / ``$1.2k`` / ``$0.0042`` / ``$—``.

    Reads three states off the aggregate, because "how much did this cost" has
    three honest answers and collapsing them lies:

    - **Nothing priceable** (``cost_is_known`` false — e.g. a local-model-only
      run): ``$—``, never ``$0.00``. Free and unknown are different facts, the
      same distinction the status band's ``$—`` makes.
    - **Partial** (``cost_is_partial``: some calls used unpriced models): a
      trailing ``+`` marks the figure as a LOWER BOUND (``$12.30+``) so it is
      never read as the complete bill.
    - **Complete**: the plain figure.

    Small sums keep more precision (``$0.0042``) because a fresh install's spend
    is fractions of a cent and rounding it to ``$0.00`` would read as free;
    large sums abbreviate (``$1.2k``) for the same glanceability as the tokens.
    """
    if not aggregate.cost_is_known:
        return "$—"
    usd = aggregate.cost_usd
    if usd >= 1000:
        body = f"${usd / 1000:.1f}k".replace(".0k", "k")
    elif usd >= 1:
        body = f"${usd:.2f}"
    elif usd >= 0.01:
        body = f"${usd:.3f}"
    else:
        # Sub-cent: show enough digits that a real spend is not rounded to $0.
        body = f"${usd:.4f}"
    return body + ("+" if aggregate.cost_is_partial else "")


def _append_cost(
    block: Text, aggregate: "UsageAggregate", cell: int, fg: Style, dim: Style
) -> None:
    """Append a right-aligned cost cell, with the lower-bound ``+`` in ``dim``.

    The ``+`` is a STATUS FLAG, not a digit (review D1): rendering it in the same
    full-strength weight as the number let it read as part of the figure, so a
    lower bound looked like a precise total. Painting it ``dim`` — and leaving
    ``$—`` legible but distinct — keeps the figure honest at a glance, and the
    footnote (``_cost_legend``) says what both marks mean.
    """
    text = format_cost(aggregate)
    pad = " " * max(0, cell - len(text))
    block.append(pad)
    if text.endswith("+"):
        block.append(text[:-1], style=fg)
        block.append("+", style=dim)
    else:
        block.append(text, style=fg)


def _needs_cost_legend(aggregate: "UsageAggregate") -> bool:
    """Whether any scope on screen shows a ``+`` (partial) or ``$—`` (unknown).

    The legend is drawn only when a mark actually appears — a fully-priced run
    needs no explaining, and a footnote for a symbol that is not on screen is
    noise.
    """
    scopes = [aggregate, *aggregate.by_provider.values(), *aggregate.by_session.values()]
    return any((s.cost_is_partial and s.cost_is_known) or not s.cost_is_known for s in scopes)


def proportion_bar(fraction: float, width: int) -> str:
    """A filled proportion bar of ``width`` cells for ``fraction`` in 0..1.

    A NONZERO fraction always fills at least one cell (review D3): without the
    floor, any component under ~2% rounded to an all-dots bar indistinguishable
    from a rounding-to-zero one, so a real 2% contributor read as empty. Only a
    genuine zero renders as no fill.
    """
    width = max(1, width)
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    if filled == 0 and fraction > 0:
        filled = 1
    return "█" * filled + "·" * (width - filled)


@dataclass(frozen=True)
class _Row:
    """One label/value/bar row in the breakdown, pre-measured for alignment."""

    label: str
    value: int
    fraction: float


def _component_rows(aggregate: UsageAggregate) -> list[_Row]:
    """The input-attribution rows, largest first, hiding empties.

    Ordered by size rather than by the fixed taxonomy order so the biggest
    consumer of context is the first thing read — the question this screen
    exists to answer is "where is it going", and the answer is whatever is at
    the top. Zero-token components are dropped: a fresh session has no tool
    results yet, and a row of zeros is noise in the one place that must be
    scannable.
    """
    total = sum(aggregate.components.get(k, 0) for k in COMPONENT_KEYS)
    rows: list[_Row] = []
    for key in COMPONENT_KEYS:
        value = int(aggregate.components.get(key, 0))
        if value <= 0:
            continue
        rows.append(
            _Row(
                label=COMPONENT_LABELS[key],
                value=value,
                fraction=(value / total) if total > 0 else 0.0,
            )
        )
    rows.sort(key=lambda r: r.value, reverse=True)
    return rows


def _semantic(name: str) -> Style:
    return Style(color=theme_mod.semantic_color(name))


#: The glyph that marks a section header. A single low-weight bullet in the
#: accent tint, set one column into the left margin so the eye finds the section
#: starts down the panel without the shouting of all-caps. The app nowhere else
#: uses all-caps headers — its list sections (``/agent``, ``/team``, ``/skills``)
#: are lowercase bold ``fg`` — so the analytics headers follow that voice and add
#: only this quiet marker to delineate the larger sections a scrolling report has.
_SECTION_MARK = "▌"


def _section_header(title: str, meta: str = "") -> Text:
    """A section header in the app's own voice: title-case, bold ``fg``, marked.

    NOT all-caps (review: the app uses that pattern nowhere else). The ``▌``
    accent bar in the left margin is the delineation — it gives a scrolling
    report a scannable left edge for its major sections the way a rule would,
    without a full-width line between every group. ``meta`` is a dim trailing
    note (a count, the estimate caveat) that qualifies the section without
    competing with its name.
    """
    fg_bold = Style(color=theme_mod.semantic_color("fg"), bold=True)
    accent = Style(color=theme_mod.semantic_color("accent"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    row = Text()
    row.append(_SECTION_MARK + " ", style=accent)
    row.append(title, style=fg_bold)
    if meta:
        row.append(f"   {meta}", style=dim)
    return row


def build_report(aggregate: UsageAggregate, width: int) -> list[Text]:
    """Render one aggregate as a list of ``Text`` lines for the screen body.

    Pure: takes the summed data and a width, returns lines. The screen wraps
    this in a scroll container and owns the chrome, so everything about WHAT is
    shown lives here where a test can read it back as plain strings.
    """
    width = max(40, width)
    fg = _semantic("fg")
    dim = _semantic("dim")
    accent = _semantic("accent")

    lines: list[Text] = []

    if aggregate.calls == 0:
        line = Text()
        line.append("No usage recorded yet.", style=fg)
        lines.append(line)
        hint = Text()
        # ``dim`` not ``faint``: this is the one line telling a first-time user
        # how the screen fills, so it must be legible, not decorative (D2).
        hint.append(
            "Analytics accrue as sessions make provider calls. Come back after a few turns.",
            style=dim,
        )
        lines.append(hint)
        return lines

    # -- headline totals -----------------------------------------------------
    calls_meta = f"{aggregate.calls} calls"
    if aggregate.ok_calls != aggregate.calls:
        # Failed-call count is real information, not chrome — keep it in the meta.
        calls_meta += f" ({aggregate.calls - aggregate.ok_calls} failed)"
    calls_meta += " · measured"
    lines.append(_section_header("Totals", calls_meta))

    def kv(name: str, value: str, note: str = "") -> Text:
        row = Text()
        row.append(f"  {name:<22}", style=dim)
        row.append(value, style=fg)
        if note:
            # ``dim`` not ``faint`` (D2): the note carries the actual cache and
            # thinking/generation breakdown — the substance a diagnostics reader
            # came for — so it must clear the contrast floor.
            row.append(f"  {note}", style=dim)
        return row

    lines.append(kv("Total billed", format_tokens(aggregate.total_tokens) + " tokens"))
    lines.append(
        kv(
            "Input",
            format_tokens(aggregate.input_tokens),
            f"+{format_tokens(aggregate.cache_read_tokens)} cache read, "
            f"{format_tokens(aggregate.cache_write_tokens)} cache write",
        )
    )
    lines.append(
        kv(
            "Output",
            format_tokens(aggregate.output_tokens),
            f"{format_tokens(aggregate.generation_tokens)} generation, "
            f"{format_tokens(aggregate.reasoning_tokens)} thinking",
        )
    )
    lines.append(
        kv(
            "Cache hit rate",
            format_percent(aggregate.cache_hit_rate),
            "of context served from cache",
        )
    )
    # Cost rides the TOTALS block because it is a headline figure, but it is an
    # ESTIMATE (published list price × billed tokens; it cannot see a plan,
    # discount, or free tier), so its note says so — the same measured-vs-modelled
    # honesty the WHERE-INPUT-WENT caveat carries. ``$—`` for a run with no
    # priceable model; a trailing ``+`` when some calls used an unpriced one.
    if aggregate.cost_is_known:
        # The trailing ``+`` on the figure already flags a partial (lower-bound)
        # sum, so the note stays short enough to fit a narrow frame; the caveat
        # it must always carry is that this is list price, not a billed invoice.
        cost_note = "≈ list price × tokens"
    else:
        cost_note = "no published price"
    # Built directly (not via ``kv``) so the lower-bound ``+`` is dimmed like the
    # table cells (review D1) — the figure reads as a number, the ``+`` as a flag.
    cost_row = Text()
    cost_row.append(f"  {'Est. cost':<22}", style=dim)
    _append_cost(cost_row, aggregate, len(format_cost(aggregate)), fg, dim)
    cost_row.append(f"  {cost_note}", style=dim)
    lines.append(cost_row)
    lines.append(Text())

    # -- input attribution (estimated) --------------------------------------
    # The estimate caveat rides the section meta (``dim``, not ``faint``): the
    # word "estimated" is why this section reads differently from Totals, and it
    # is ALSO carried at the data level — the ``≈`` mark and the ``~`` on every
    # percentage below — so the distinction survives the heading scrolling away.
    lines.append(_section_header("Where input went", "≈ estimated split of context tokens"))

    rows = _component_rows(aggregate)
    if not rows:
        empty = Text()
        empty.append("  no component data yet", style=dim)
        lines.append(empty)
    else:
        # One guaranteed space of gutter between the label and its bar (D5): the
        # longest label is exactly ``label_col`` wide, so without the trailing
        # gap its bar butts against the final glyph while every shorter row has
        # air. The ``+ 2`` reserves that gutter for every row uniformly.
        label_col = min(36, max(len(r.label) for r in rows) + 2)
        value_col = max(len(format_tokens(r.value)) for r in rows)
        # ``~NN%`` is one cell wider than ``NN%``; size the column for it.
        bar_width = max(8, min(24, width - label_col - value_col - 13))
        for row in rows:
            line = Text()
            line.append(f"  {row.label:<{label_col}}", style=fg)
            line.append(proportion_bar(row.fraction, bar_width), style=accent)
            line.append(f" {format_tokens(row.value):>{value_col}}", style=fg)
            # ``~`` marks the percentage as modelled, not measured (D1).
            line.append(f" ~{format_percent(row.fraction):>3}", style=dim)
            lines.append(line)
    lines.append(Text())

    # -- by provider / by session -------------------------------------------
    # Both tables share ONE ``name_col`` (review D2): computed across every row
    # of both groups so the tokens/cost/calls columns line up vertically down
    # the panel instead of starting at two different x-positions. On a wide
    # frame the shared column is allowed to grow (review D3) so the extra width
    # widens the content rather than leaving a dead right gutter.
    names = getattr(aggregate, "session_names", {}) or {}
    session_labelled = {
        short_session_label(sid, names.get(sid, "")): agg
        for sid, agg in aggregate.by_session.items()
    }
    all_names = [n for n in aggregate.by_provider] + [n for n in session_labelled]
    if all_names:
        # Grow the name column with the frame: a wide card gets a roomier column
        # (up to 48) so its width is used; a narrow one stays compact (30).
        name_cap = 30 if width < 96 else min(48, width - 40)
        name_col = min(name_cap, max((len(n) for n in all_names), default=0) + 1)
    else:
        name_col = 0

    if aggregate.by_provider:
        lines.append(_group_section("By provider", aggregate.by_provider, width, name_col))
        lines.append(Text())

    if session_labelled:
        lines.append(_group_section("By session", session_labelled, width, name_col))

    # Legend for the cost markers, drawn only when a ``+`` or ``$—`` is on
    # screen (review D1). ``dim`` so it reads as a footnote, not a row.
    if _needs_cost_legend(aggregate):
        lines.append(Text())
        legend = Text()
        legend.append("  + lower bound (some calls unpriced)   $— no published price", style=dim)
        lines.append(legend)

    return lines


#: Below this content width the per-group tables drop the cache column to keep
#: the cost column, which is the one this feature adds and the one a narrow
#: frame should not be the reason to lose. Cost + tokens + calls are the
#: irreducible trio; cache is the first to shed, exactly like the status band's
#: drop ladder.
_WIDE_TABLE_MIN = 72


def _group_section(
    title: str,
    groups: dict[str, "UsageAggregate"],
    width: int,
    name_col: int,
) -> Text:
    """A per-provider or per-session table as one multi-line ``Text`` block.

    Returned as a single ``Text`` with embedded newlines so the caller keeps a
    flat list of blocks; the screen splits on newlines only for the scroll
    measurement. Sorted by **cost** (falling back to tokens) descending — the
    biggest *spend* is what a cost-aware diagnostics reader looks for first, and
    where nothing is priced this is exactly the old token order.

    ``name_col`` is shared across both tables (review D2) so their columns line
    up. ``width`` decides the rest: a wide frame shows tokens · cost · calls ·
    cache; a narrow one drops the cache column (see ``_WIDE_TABLE_MIN``) so the
    cost column this feature adds always survives.
    """
    fg = _semantic("fg")
    dim = _semantic("dim")

    # Same marked, title-case header as every other section (no all-caps).
    block = _section_header(title)

    # Sort by cost when any of these groups is priced, else by tokens. Keyed on
    # the tuple so an unpriced group sorts by tokens as a tiebreak rather than
    # collapsing to a single $0 bucket.
    ordered = sorted(
        groups.items(),
        key=lambda kv: (kv[1].cost_micro, kv[1].total_tokens),
        reverse=True,
    )
    if not ordered:
        block.append("\n  (none)", style=dim)
        return block

    show_cache = width >= _WIDE_TABLE_MIN
    cost_col = max(len(format_cost(agg)) for _, agg in ordered)
    for name, agg in ordered:
        block.append("\n")
        block.append(f"  {name:<{name_col}}", style=fg)
        block.append(f"{format_tokens(agg.total_tokens):>8} tokens", style=fg)
        # Cost sits next to tokens as the other headline number, in full-strength
        # ``fg`` — it is the answer this feature exists to give, not a footnote.
        # The lower-bound ``+`` is dimmed by ``_append_cost`` (review D1).
        block.append("   ")
        _append_cost(block, agg, cost_col, fg, dim)
        block.append(f"   {agg.calls:>4} calls", style=dim)
        if show_cache:
            block.append(f"   {format_percent(agg.cache_hit_rate):>4} cache", style=dim)
    return block


class AnalyticsScreen(ModalScreen[None]):
    """Full-screen, scrollable, Esc-dismissable usage analytics.

    Pushed by ``/analytics /usage`` and dismissed with Esc (or ``q``), which
    restores the previous view exactly — the screen reads the ledger and shows
    it, it never mutates anything, so leaving it is a plain pop with no state to
    reconcile. Modelled on :class:`SessionPickerScreen`: a centred card over a
    dimmed transcript, one ``Static`` body inside a ``VerticalScroll`` so a long
    per-session table scrolls rather than clipping.
    """

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=False),
        Binding("q", "dismiss_screen", "Back", show=False),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(self, aggregate: UsageAggregate) -> None:
        super().__init__()
        self._aggregate = aggregate
        self._body: Static
        self._scroll: VerticalScroll

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            yield Static(self._title_text(), id="analytics-title")
            with VerticalScroll(id="analytics-scroll") as scroll:
                self._scroll = scroll
                self._body = Static(id="analytics-body")
                yield self._body
            self._hint = Static(self._hint_text(scrollable=False), id="analytics-hint")
            yield self._hint

    def on_mount(self) -> None:
        self._repaint()
        # After layout settles: ``max_scroll_y`` is only meaningful once the
        # body has been measured against the viewport.
        self.call_after_refresh(self._sync_hint)

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        self._repaint()
        self.call_after_refresh(self._sync_hint)

    def _card_width(self) -> int:
        # Track the CSS card (90% of the terminal, capped at 140 — see the
        # ``.analytics-panel`` rule) minus its 2-cell horizontal padding each
        # side, so the report's own column maths matches the width it is
        # actually painted into. The cap here mirrors the CSS cap; raising one
        # without the other either wastes the frame or overruns it.
        try:
            terminal = self.app.size.width
            card = min(140, int(terminal * 0.9))
            return max(40, card - 6)
        except Exception:  # noqa: BLE001 — before mount, a sane default
            return 88

    def _title_text(self) -> Text:
        # ``fg`` bold, matching the ``/usage`` panel's title (and the app's list
        # headers) rather than the violet ``label`` — one title voice across the
        # overlays. A ``─`` rule under it (second line) delineates the pinned
        # header from the scrolling body, the same device ``/usage`` uses.
        fg = Style(color=theme_mod.semantic_color("fg"), bold=True)
        faint = Style(color=theme_mod.semantic_color("faint"))
        title = Text()
        title.append("Usage analytics", style=fg)
        title.append("   all sessions", style=faint)
        title.append("\n")
        title.append("─" * max(1, self._card_width()), style=faint)
        return title

    def _hint_text(self, *, scrollable: bool) -> Text:
        # The scroll affordance is advertised only when there is something to
        # scroll (D5): the empty state and any report that fits told the user to
        # scroll a screen that could not, which reads as a dead control.
        faint = Style(color=theme_mod.semantic_color("faint"))
        hint = Text()
        hint.append("esc back", style=faint)
        if scrollable:
            hint.append(" · ↑↓ scroll", style=faint)
        return hint

    def _sync_hint(self) -> None:
        """Show the scroll hint only when the body overflows its viewport."""
        hint = getattr(self, "_hint", None)
        scroll = getattr(self, "_scroll", None)
        if hint is None or scroll is None or not hint.is_mounted:
            return
        try:
            scrollable = scroll.max_scroll_y > 0
        except Exception:  # noqa: BLE001 — before layout, assume not scrollable
            scrollable = False
        hint.update(self._hint_text(scrollable=scrollable))

    def _report_lines(self) -> list[Text]:
        return build_report(self._aggregate, self._card_width())

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        combined = Text()
        for i, line in enumerate(self._report_lines()):
            if i:
                combined.append("\n")
            combined.append_text(line)
        body.update(combined)

    def render_lines_for_test(self) -> list[str]:
        """The report as plain strings — what a user reads."""
        out: list[str] = []
        for line in self._report_lines():
            out.extend(line.plain.split("\n"))
        return out

    # -- actions -------------------------------------------------------------
    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    def action_scroll_up(self) -> None:
        self._scroll.scroll_up()

    def action_scroll_down(self) -> None:
        self._scroll.scroll_down()

    def action_page_up(self) -> None:
        self._scroll.scroll_page_up()

    def action_page_down(self) -> None:
        self._scroll.scroll_page_down()

    def action_scroll_home(self) -> None:
        self._scroll.scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll.scroll_end()
