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
from typing import Protocol

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
    UsagePeriod,
    short_session_label,
)
from local_operator.tui import theme as theme_mod


class _CostLike(Protocol):
    """The cost interface both a scope and a calendar bucket expose.

    ``format_cost``/``append_cost`` render the same ``$—``/``$X.XX``/``$X+``
    honesty for a :class:`UsageAggregate` (a provider/session scope) AND a
    :class:`UsagePeriod` (a day/month bucket). Rather than duplicate the money
    formatter or widen the annotation to a lie, this Protocol names the three
    members both dataclasses share, so the formatter is typed for exactly what
    it reads.
    """

    @property
    def cost_usd(self) -> float: ...

    @property
    def cost_is_known(self) -> bool: ...

    @property
    def cost_is_partial(self) -> bool: ...


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


def format_cost(aggregate: "_CostLike") -> str:
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


def append_cost(block: Text, aggregate: "_CostLike", cell: int, fg: Style, dim: Style) -> None:
    """Append a right-aligned cost cell, with the lower-bound ``+`` in ``dim``.

    Public (it was ``_append_cost``) because ``session_panel`` renders the same
    money cells: one screen showing ``$1.20+`` and its sibling showing a plain
    ``$1.20`` for the same partial sum would be two honesty vocabularies. An
    underscore name imported across modules is a private contract in all but
    spelling, so the name says what the visibility already is.

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


def scope_needs_cost_legend(scope: "_CostLike") -> bool:
    """Whether ONE scope renders a ``+`` (partial) or a ``$—`` (unknown).

    The predicate, not the loop: ``/session`` decides the same question over a
    different set of scopes (its per-model and per-purpose groups), and two
    copies of "what counts as a marked figure" would eventually disagree about
    when the footnote is owed.
    """
    return (scope.cost_is_partial and scope.cost_is_known) or not scope.cost_is_known


def _needs_cost_legend(aggregate: "UsageAggregate") -> bool:
    """Whether any scope on screen shows a ``+`` (partial) or ``$—`` (unknown).

    The legend is drawn only when a mark actually appears — a fully-priced run
    needs no explaining, and a footnote for a symbol that is not on screen is
    noise.
    """
    scopes = [aggregate, *aggregate.by_provider.values(), *aggregate.by_session.values()]
    return any(scope_needs_cost_legend(s) for s in scopes)


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


def semantic_style(name: str) -> Style:
    """A ``rich`` style for one semantic theme token.

    Public (it was ``_semantic``) so ``session_panel`` resolves its row colours
    through the same one-line helper rather than re-spelling
    ``Style(color=theme_mod.semantic_color(...))`` at every call site.
    """
    return Style(color=theme_mod.semantic_color(name))


#: The glyph that marks a section header. A single low-weight bullet in the
#: accent tint, set one column into the left margin so the eye finds the section
#: starts down the panel without the shouting of all-caps. The app nowhere else
#: uses all-caps headers — its list sections (``/agent``, ``/team``, ``/skills``)
#: are lowercase bold ``fg`` — so the analytics headers follow that voice and add
#: only this quiet marker to delineate the larger sections a scrolling report has.
_SECTION_MARK = "▌"


def section_header(title: str, meta: str = "") -> Text:
    """A section header in the app's own voice: title-case, bold ``fg``, marked.

    Public (it was ``_section_header``) for the same reason as
    :func:`append_cost`: ``/session`` draws the identical ``▌``-marked headers,
    and forking the glyph or the meta styling would give the two diagnostics
    screens two visual languages.

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


#: The two metrics the daily/monthly bar charts can plot. ``cost`` is the
#: default because "how much am I spending over time" is the stated purpose of
#: the historical view; ``t`` toggles to ``tokens`` for a run whose models are
#: mostly unpriced (where cost bars would all be floor marks). Held as a small
#: vocabulary so the toggle, the header, and the renderer cannot disagree on the
#: legal values.
METRIC_COST = "cost"
METRIC_TOKENS = "tokens"

#: The footnote explaining the two money marks. One string so ``/analytics`` and
#: ``/session`` cannot drift into two different explanations of the same glyph.
COST_LEGEND = "+ lower bound (some calls unpriced)   $— no published price"


def _month_name(mm: int) -> str:
    return (
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()[mm - 1] if 1 <= mm <= 12 else "?"
    )


def _period_label(period: str) -> str:
    """A compact, human bucket label: ``Aug 21`` for a day, ``Aug 2026`` for a month.

    Parses the stored ``YYYY-MM-DD`` / ``YYYY-MM`` key rather than reformatting
    a datetime, because the key is already the local calendar bucket the store
    chose (see the schema comment) and re-deriving it from a timestamp risks a
    tz round-trip disagreeing with what was recorded. An unparseable key falls
    back to itself so a malformed row is still legible, never a crash.
    """
    parts = period.split("-")
    try:
        if len(parts) == 3:
            return f"{_month_name(int(parts[1]))} {int(parts[2]):02d}"
        if len(parts) == 2:
            return f"{_month_name(int(parts[1]))} {parts[0]}"
    except (ValueError, IndexError):
        return period
    return period


def _metric_value(period: UsagePeriod, metric: str) -> float:
    """The number a bar is drawn proportional to, for the chosen metric.

    Cost is read in micro-USD (kept as a float for the fraction maths); tokens
    is the full billed total. Both are non-negative, so the fraction against the
    window max is always well defined.
    """
    if metric == METRIC_TOKENS:
        return float(period.total_tokens)
    return float(period.cost_micro)


def _format_metric_cell(period: UsagePeriod, metric: str) -> str:
    """The right-hand numeric label on a bar row for the chosen metric."""
    if metric == METRIC_TOKENS:
        return format_tokens(period.total_tokens)
    # ``format_cost`` is duck-typed on the cost_* / cost_usd interface that
    # UsagePeriod shares with UsageAggregate, so a period formats identically to
    # a scope — same ``$—``/``+`` honesty, no second money formatter.
    return format_cost(period)


def _metric_meta(metric: str, *, prefix: str = "") -> str:
    """The section-meta string that self-describes the active metric + toggle.

    ``cost · t → tokens`` / ``tokens · t → cost`` — so BOTH the daily and the
    monthly chart state what their bars plot and that ``t`` flips it (review
    U2/D5). Shared by both call sites so the two sibling charts speak with one
    voice. ``prefix`` prepends a functional descriptor (e.g. ``by calendar
    month``) ahead of the metric clause where the section wants one.
    """
    active = "cost" if metric == METRIC_COST else "tokens"
    other = "tokens" if metric == METRIC_COST else "cost"
    clause = f"{active} · t → {other}"
    return f"{prefix} · {clause}" if prefix else clause


def _series_chart(
    title: str,
    meta: str,
    periods: list[UsagePeriod],
    metric: str,
    width: int,
    *,
    empty_note: str,
) -> list[Text]:
    """A titled horizontal bar chart, one row per calendar bucket.

    Each bar's fill is that bucket's value as a fraction of the WINDOW MAX (the
    largest bucket in view), so the tallest bar is full and the rest read
    relative to it — the standard "which day/month was biggest" shape. Columns
    are measured across every row so the labels, bars, and numbers align into
    one table (the same discipline ``_group_section`` and ``/usage`` use).

    The ``≥`` floor mark is prepended to a cost cell that is a genuine lower
    bound — a bucket that mixed priced AND unpriced calls, so its dollar figure
    is real money that undercounts (``cost_is_floor and cost_is_known``, review
    D1). A FULLY-unpriced bucket has no dollar figure to bound: it renders a
    clean ``$—`` with NO ``≥``, because "≥ unknown" is a contradiction (you
    cannot lower-bound a value you do not have). This matters for Local
    Operator's common local-model-only run, whose default cost chart would
    otherwise be a wall of ``≥ $—``. Because the mark is the single lower-bound
    signal here, ``format_cost``'s trailing ``+`` — which means the same thing —
    is stripped from a marked cell (review D2), so a floored row reads
    ``≥ $0.700`` rather than the doubled ``≥ $0.700+``.

    Newest bucket LAST (the store returns oldest-first) to match the
    transcript's top-to-bottom reading order.

    Pure: returns ``Text`` lines so ``render_lines_for_test`` reads the chart
    back as plain strings, exactly like the rest of ``build_report``.
    """
    fg = semantic_style("fg")
    dim = semantic_style("dim")
    accent = semantic_style("accent")

    lines: list[Text] = [section_header(title, meta)]
    if not periods:
        empty = Text()
        empty.append(f"  {empty_note}", style=dim)
        lines.append(empty)
        return lines

    labels = [_period_label(p.period) for p in periods]
    values = [_metric_value(p, metric) for p in periods]
    # The ``≥`` marks a genuine lower bound: cost mode, some spend we could
    # price (``cost_is_known``), and some we could not (``cost_is_floor``). A
    # fully-unpriced bucket is ``cost_is_known == False`` → no mark, and its
    # cell is a plain ``$—`` (review D1).
    floored = [metric == METRIC_COST and p.cost_is_floor and p.cost_is_known for p in periods]
    # A marked cell drops the redundant trailing ``+`` (review D2): the ``≥``
    # already says "lower bound", so ``format_cost``'s ``+`` would say it twice.
    cells = [
        (_format_metric_cell(p, metric).rstrip("+") if is_floor else _format_metric_cell(p, metric))
        for p, is_floor in zip(periods, floored)
    ]
    max_value = max(values) if values else 0.0

    label_col = max((len(lbl) for lbl in labels), default=0)
    # The value cell reserves room for the floor mark (`≥ `) so a floored and an
    # unfloored row's numbers right-align in the same column.
    value_col = max((len(c) for c in cells), default=0) + 2
    bar_width = max(8, min(28, width - label_col - value_col - 6))

    for period, label, value, cell, is_floor in zip(periods, labels, values, cells, floored):
        fraction = (value / max_value) if max_value > 0 else 0.0
        row = Text()
        row.append(f"  {label:<{label_col}}  ", style=fg)
        row.append(proportion_bar(fraction, bar_width), style=accent)
        mark = "≥ " if is_floor else "  "
        # The mark is dim (a status flag, not a digit — review D1 on the ``+``),
        # the number full-strength ``fg``. Right-align the number within the
        # column after the mark so bars of different magnitudes still line up.
        row.append(" ")
        row.append(mark, style=dim)
        row.append(f"{cell:>{max(0, value_col - 2)}}", style=fg)
        lines.append(row)
    return lines


def build_report(
    aggregate: UsageAggregate,
    width: int,
    *,
    daily: list[UsagePeriod] | None = None,
    monthly: list[UsagePeriod] | None = None,
    window_totals: UsagePeriod | None = None,
    metric: str = METRIC_COST,
) -> list[Text]:
    """Render one aggregate as a list of ``Text`` lines for the screen body.

    Pure: takes the summed data and a width, returns lines. The screen wraps
    this in a scroll container and owns the chrome, so everything about WHAT is
    shown lives here where a test can read it back as plain strings.

    ``daily``/``monthly`` are the calendar rollup series
    (:meth:`AnalyticsStore.daily_series` / ``monthly_series``); when present
    they render as "Last N days" and "Monthly" bar charts between the headline
    totals and the input attribution — the historical arc the raw aggregate
    cannot show. ``metric`` (``cost`` or ``tokens``) selects what the bars plot;
    the screen's ``t`` key flips it. They default to ``None``/``cost`` so a
    caller with no rollups (or a pre-rollup test) gets exactly the original
    report.
    """
    width = max(40, width)
    fg = semantic_style("fg")
    dim = semantic_style("dim")
    accent = semantic_style("accent")

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
    lines.append(section_header("Totals", calls_meta))

    # Value cells share a gutter so notes line up even when compact figures
    # differ in width (``3M`` vs ``387k``). 11 cells matches the existing
    # ``3.7M tokens`` / ``$18.40`` slot; a longer value just grows the cell.
    _VALUE_CELL = 11

    def kv(name: str, value: str, note: str = "") -> Text:
        row = Text()
        row.append(f"  {name:<22}", style=dim)
        row.append(f"{value:<{_VALUE_CELL}}", style=fg)
        if note:
            # ``dim`` not ``faint`` (D2): the note carries the actual cache and
            # thinking/generation breakdown — the substance a diagnostics reader
            # came for — so it must clear the contrast floor.
            row.append(f"  {note}", style=dim)
        return row

    lines.append(
        kv(
            "Total billed",
            format_tokens(aggregate.total_tokens) + " tokens",
            f"{format_tokens(aggregate.context_tokens)} in · "
            f"{format_tokens(aggregate.output_tokens)} out",
        )
    )
    # NESTED input breakdown. The old flat "Input NNN" row read as "total input"
    # and, at a 97% cache-hit rate, showed a tiny number (only the fresh/uncached
    # slice) that looked like a bug. All three sub-values are AUTHORITATIVE
    # provider counts (not estimates), so the tree makes explicit that the small
    # "Fresh (uncached)" figure sits UNDER the full "Context read" total, with
    # cache reads/writes as its siblings. The wording is deliberate: "Fresh
    # (uncached)" is ALL uncached input (new user turns plus freshly-added tool
    # results/reads/system content not yet cached), NOT "user input" — labelling
    # it as user messages would be wrong. ``kv`` pads the name to 22 uniformly,
    # so the leading space + tree glyph on the sub-rows indents them while the
    # values stay column-aligned.
    #
    # Fresh is ``aggregate.fresh_tokens`` (context − cache_read − cache_write),
    # not ``input_tokens``: providers disagree on whether input already includes
    # cache, so binding Fresh to input would show the FULL context on OpenAI-
    # shaped usage. The three children partition the parent on every provider.
    # The Context-read note restates that composition so compact formatting
    # (387k + 3M + 113k all printing as 3.5M / 3M) cannot hide the sum.
    fresh = aggregate.fresh_tokens
    cache_read = aggregate.cache_read_tokens
    cache_write = aggregate.cache_write_tokens
    lines.append(
        kv(
            "Context read",
            format_tokens(aggregate.context_tokens),
            f"{format_tokens(fresh)} fresh · "
            f"{format_tokens(cache_read)} cached · "
            f"{format_tokens(cache_write)} written",
        )
    )
    lines.append(
        kv(
            " ├ Fresh (uncached)",
            format_tokens(fresh),
            "new input, billed at full rate",
        )
    )
    lines.append(
        kv(
            " ├ Cache read",
            format_tokens(cache_read),
            "input served from cache",
        )
    )
    lines.append(
        kv(
            " └ Cache write",
            format_tokens(cache_write),
            "new input written to cache",
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
    # ``append_cost`` right-aligns (table cells); here we pass the figure's own
    # width so it left-aligns with the token values, then pad out to
    # ``_VALUE_CELL`` so the cost note shares the gutter (D2).
    cost_text = format_cost(aggregate)
    cost_row = Text()
    cost_row.append(f"  {'Est. cost':<22}", style=dim)
    append_cost(cost_row, aggregate, len(cost_text), fg, dim)
    cost_row.append(" " * max(0, _VALUE_CELL - len(cost_text)))
    cost_row.append(f"  {cost_note}", style=dim)
    lines.append(cost_row)
    lines.append(Text())

    # -- historical time series (daily + monthly bars) ----------------------
    # Drawn only when the store handed the screen rollup rows. The metric label
    # in each section header states what the bars plot AND how to flip it, so a
    # reader who sees dollar bars knows tokens are one keypress away (and vice
    # versa on a run whose models are unpriced and whose cost bars are all
    # floors). Placed above the input attribution because "when did I spend"
    # precedes "what was the spend made of" in a diagnostics read.
    if daily is not None:
        # ``len(daily)`` is the number of DAYS WITH USAGE the store returned,
        # not a calendar window (``daily_series`` skips idle days), so the label
        # says exactly that rather than "Last N days" — which read as a calendar
        # span and misstated sparse usage (review D3). Singular "day" for one
        # (review D4). The empty case keeps a static title (no count to state).
        if daily:
            n = len(daily)
            day_title = f"{n} day{'s' if n != 1 else ''} with usage"
        else:
            day_title = "Days with usage"
        # The window's grand total rides the section meta ahead of the metric
        # clause (review C2 — this is where ``series_totals`` surfaces): the
        # daily bars show the per-day shape, and this states what they sum to
        # over the same window, in the active metric so the two agree. Falls
        # back to just the metric clause when no window total was supplied.
        if window_totals is not None and daily:
            if metric == METRIC_TOKENS:
                window_summary = f"{format_tokens(window_totals.total_tokens)} tokens"
            else:
                window_summary = format_cost(window_totals)
            daily_meta = _metric_meta(metric, prefix=window_summary)
        else:
            daily_meta = _metric_meta(metric)
        lines.extend(
            _series_chart(
                day_title,
                daily_meta,
                daily,
                metric,
                width,
                empty_note="no daily usage recorded yet",
            )
        )
        lines.append(Text())
    if monthly is not None:
        lines.extend(
            _series_chart(
                "Monthly",
                # Functional meta that ALSO self-describes the metric and toggle
                # (reviews D5 + U2): the sibling chart must state $ vs tokens too,
                # so a reader parked on Monthly is not left guessing.
                _metric_meta(metric, prefix="by calendar month"),
                monthly,
                metric,
                width,
                empty_note="no monthly usage recorded yet",
            )
        )
        lines.append(Text())

    # -- input attribution (estimated) --------------------------------------
    # The estimate caveat rides the section meta (``dim``, not ``faint``): the
    # word "estimated" is why this section reads differently from Totals, and it
    # is ALSO carried at the data level — the ``≈`` mark and the ``~`` on every
    # percentage below — so the distinction survives the heading scrolling away.
    lines.append(section_header("Where input went", "≈ estimated split of context tokens"))

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
        legend.append("  " + COST_LEGEND, style=dim)
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
    fg = semantic_style("fg")
    dim = semantic_style("dim")

    # Same marked, title-case header as every other section (no all-caps).
    block = section_header(title)

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
        # The lower-bound ``+`` is dimmed by ``append_cost`` (review D1).
        block.append("   ")
        append_cost(block, agg, cost_col, fg, dim)
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
        Binding("t", "toggle_metric", "Cost/tokens", show=False),
        Binding("up", "scroll_up", "Up", show=False),
        Binding("down", "scroll_down", "Down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Top", show=False),
        Binding("end", "scroll_end", "Bottom", show=False),
    ]

    def __init__(
        self,
        aggregate: UsageAggregate,
        *,
        daily: list[UsagePeriod] | None = None,
        monthly: list[UsagePeriod] | None = None,
        window_totals: UsagePeriod | None = None,
    ) -> None:
        super().__init__()
        self._aggregate = aggregate
        # Grand total over the daily chart's window (``series_totals``), shown in
        # that chart's meta so the bars and their sum describe the same span.
        self._window_totals = window_totals
        # The calendar rollup series the store handed us on open. Held so the
        # ``t`` toggle can re-render the SAME data with the other metric without
        # a second store read — the numbers do not change, only which of them
        # the bars plot. ``None`` (a caller that passed no rollups) hides the
        # chart sections entirely rather than drawing an empty frame.
        self._daily = daily
        self._monthly = monthly
        #: Which metric the bar charts plot; ``t`` flips it. Cost by default
        #: (the historical view's stated purpose).
        self._metric = METRIC_COST
        self._title: Static
        self._body: Static
        self._scroll: VerticalScroll

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            # Held so the metric toggle can repaint the pinned title in place
            # (the ``bars: cost``/``bars: tokens`` suffix — reviews U1/U4).
            self._title = Static(self._title_text(), id="analytics-title")
            yield self._title
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
        # ``no_wrap`` + crop so the rule (and the title's suffix) CROP to the
        # widget's real content box instead of wrapping. ``_card_width`` floors
        # at 40, but the painted title box is narrower on a sub-46-col terminal
        # (review MINOR): an unbounded ``─`` run would wrap to a second line and
        # eat into the fixed ``height: 3``. Cropping keeps the rule one line at
        # any width; the report rows already truncate per-row for the same reason.
        title = Text(no_wrap=True, overflow="crop")
        title.append("Usage analytics", style=fg)
        title.append("   all sessions", style=faint)
        # Carry the active chart metric in the PINNED title so ``t`` gives
        # on-screen feedback regardless of scroll position (reviews U1/U4): a
        # user parked past both charts still sees ``cost``↔``tokens`` flip up
        # here, so the advertised key never reads as a dead control. Only shown
        # when there are charts to toggle — an empty/rollup-less report has no
        # metric, so the suffix would be noise.
        if self._has_charts():
            active = "cost" if self._metric == METRIC_COST else "tokens"
            title.append(f"   bars: {active}", style=faint)
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
        # The metric toggle is advertised only when there is a chart to toggle;
        # an empty or rollup-less report has no bars, so the key is a no-op and
        # promising it would be a dead control (the same rule the scroll hint
        # follows). ``bool(series)`` is false for both ``None`` and ``[]``.
        if self._has_charts():
            hint.append(" · t cost/tokens", style=faint)
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

    def _has_charts(self) -> bool:
        """Whether any non-empty rollup series is loaded (a bar to toggle).

        ``[]`` and ``None`` both read as no charts: an empty-store report shows
        the "no usage yet" line, not bars, so the metric toggle has nothing to
        act on and must not be advertised.
        """
        return bool(self._daily or self._monthly)

    def _report_lines(self) -> list[Text]:
        return build_report(
            self._aggregate,
            self._card_width(),
            daily=self._daily,
            monthly=self._monthly,
            window_totals=self._window_totals,
            metric=self._metric,
        )

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

    def action_toggle_metric(self) -> None:
        """Flip the bar charts between cost and tokens, in place.

        Re-renders from the series already in hand (no store read): the toggle
        changes only which number the bars are proportional to, so the data is
        unchanged and only the body repaints. A no-op when no rollup series are
        loaded — there is nothing to plot, so the key does nothing rather than
        silently toggling invisible state.
        """
        if not self._has_charts():
            return
        self._metric = METRIC_TOKENS if self._metric == METRIC_COST else METRIC_COST
        self._repaint()
        # Repaint the PINNED title too, so its ``bars: cost``/``bars: tokens``
        # suffix reflects the flip even when the charts are scrolled off screen
        # (reviews U1/U4). The body repaint above does not touch the title Static.
        title = getattr(self, "_title", None)
        if title is not None and title.is_mounted:
            title.update(self._title_text())

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
