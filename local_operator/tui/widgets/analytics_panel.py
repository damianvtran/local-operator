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

The body is a :class:`~local_operator.tui.widgets.report_view.ReportView`, not
a ``Static``, and that is a fix rather than a preference. A ``Static`` renders
its whole content on every dirty repaint — ``Widget._render_content`` converts
the widget's OWN height to strips — and this body's height is the whole report
(846 lines / 85,446 cells on the operator's ledger against a 2,929-cell
viewport), so every pointer row crossing re-stripped the entire report to
change two of its lines. Measured on a copy of that ledger at 120x45 through
the real app, loop-thread CPU: **243 ms mean / 320 ms max per row crossed**,
174 ms per wheel line with the pointer resting on the table, against a 6.6 ms
mean / 13.6 ms max floor for the same pointer moves with no analytics screen
open. ``ReportView`` renders only the lines the compositor asks for, and a
hover change recomposes just the two rows whose tint changed — 8 ms mean per
row-to-row crossing, of which the body's own share is two strip conversions
(4 ms across twelve crossings in total). The widths a single-row recompose
needs come from :class:`ReportLayout`, never from the row list it is handed —
see :func:`_session_row_line` for why that is load-bearing rather than tidy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Mapping, NamedTuple, Protocol, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    COMPONENT_LABELS,
    SessionNode,
    UsageAggregate,
    UsagePeriod,
    build_session_forest,
    session_table_labels,
)
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.report_view import ReportView
from local_operator.tui.widgets.tool_card import truncate_cells


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


def _needs_cost_legend(
    aggregate: "UsageAggregate", forest: list["SessionNode"] | None = None
) -> bool:
    """Whether any scope on screen shows a ``+`` (partial) or ``$—`` (unknown).

    The legend is drawn only when a mark actually appears — a fully-priced run
    needs no explaining, and a footnote for a symbol that is not on screen is
    noise.

    INVARIANT: this must name every scope from which a ``$`` figure is drawn.
    ``by_session`` covers the nested table's rows too — a row renders its
    subtree TOTAL, which can carry a ``+`` no individual session shows, so the
    rolled-up scopes are added rather than relying on their parts.

    ``forest`` is the one ``build_report`` already built. Pass it: building a
    second identical forest measured 3.1 ms against 16.9 ms for the whole report
    (~18% of it, review F3), and two independent constructions of the structure
    the table's correctness rests on can disagree. Defaulted for the callers
    (tests, the desktop route) that have only an aggregate.
    """
    if forest is None:
        forest = build_session_forest(
            aggregate.by_session, getattr(aggregate, "session_parents", {}) or {}
        )
    scopes = [
        aggregate,
        *aggregate.by_provider.values(),
        *aggregate.by_session.values(),
        *(node.total for node in _iter_nodes(forest)),
    ]
    return any(scope_needs_cost_legend(s) for s in scopes)


def _iter_nodes(forest: list["SessionNode"]):
    """Every node in the forest, roots and nested children alike."""
    for node in forest:
        yield node
        yield from _iter_nodes(list(node.children))


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
    expanded: "Collection[str] | None" = None,
    cursor: str | None = None,
    hover: str | None = None,
    layout: ReportLayout | None = None,
    forest: list["SessionNode"] | None = None,
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

    ``expanded`` is the set of session IDs whose subagent rows are shown. The
    default — none — is what the user asked for: the operator's ledger renders
    2,240 session rows fully expanded, of which 1,645 are subagents nobody
    asked to see, and the cost is paid again on EVERY rebuild (a terminal resize
    emits a storm of them, which is the "freezing" in the report).

    ``cursor`` is the SESSION ID the keyboard is on — an id and not a row index,
    for the same reason the expansion set is: expanding a row inserts rows above
    every later index, so an index-keyed cursor would slide onto a different
    session on the very keypress that is supposed to leave it where it is.

    ``hover`` is the SESSION ID the POINTER is over, keyed by id for that same
    reason, and it is independent of ``cursor``: the two selection models are
    allowed to sit on different rows, and the report paints both marks at once.

    ``layout``, if given, receives which rows were painted and where, which the
    interactive screen needs to scroll the cursor into view; recomputing that
    outside this function would be a second, possibly disagreeing, notion of
    which rows are on screen.

    ``forest`` lets a caller that already has one pass it in rather than paying
    to rebuild it. It measured 34 ms on the operator's ledger — irrelevant for a
    one-shot report, and the dominant cost of a REPAINT once every arrow key
    triggers one (0.25 s per press, which does not read as a cursor). The screen
    holds the aggregate immutable for its lifetime, so the forest it caches
    cannot go stale; a caller that passes a forest built from a different
    aggregate would get a report describing neither, which is why this is not
    derived from a mutable field.
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
    # ROOTS carry their subtree's total and children hang beneath them, rather
    # than every session being listed flat with its own spend. This is the only
    # arrangement that both credits a parent with what its subagents spent AND
    # keeps this column summing to the headline total above it: rolling up while
    # still listing children at top level inflates the table by $8,077 (12.7%)
    # on the operator's ledger. See ``build_session_forest``.
    if forest is None:
        forest = build_session_forest(
            aggregate.by_session, getattr(aggregate, "session_parents", {}) or {}
        )
    # STRUCTURE FIRST, LABELS SECOND. The walk yields ``(session_id, depth,
    # subtree total)`` and nothing about how a row reads, because the label
    # budget below cannot be computed until the numbers that share the row are
    # known, and the numbers come from the forest. Composing labels inside the
    # walk (as the nesting change first did) forces them to be built against a
    # constant, which is exactly the defect the budgeting work removed.
    structure = _forest_rows(forest, expanded)

    # The label budget is decided BEFORE the labels are composed, from the frame
    # this report is being rendered into (design review D2). Composing against a
    # fixed constant and then padding to a different column is what left 34
    # cells of dead gutter at 140 columns while simultaneously overrunning the
    # 30-cell column at 71 — rows cut to fit a budget the frame was not
    # enforcing. One number now drives both: the labels are built to it and the
    # column is drawn at it.
    # What the name may spend is what the frame has LEFT after the columns that
    # are not negotiable (design review D8). The previous rule stepped 30 -> 48
    # the instant ``width`` reached 96 against a flat ``- 40`` allowance, but the
    # rest of a row measures 51-55 cells, so one extra cell of frame bought 18
    # cells of label and the widest row overran the content box across terminal
    # widths 114-120 — clipping ``% cache`` off every row, silently, because a
    # row ending in ``cach`` still looks like a row. Measuring the remainder
    # instead means no width can lose a column: the budget rises a cell at a
    # time as the frame does, and the clamp only ever narrows it further.
    #
    # Both tables share one ``name_col``, so the budget must clear the WIDER of
    # the two overheads — the provider table's cost column is sized
    # independently of the session table's and either may be the wider row.
    #
    # The session side is measured over the SUBTREE TOTALS the forest produced,
    # not over ``by_session``'s own aggregates. A root row paints what its whole
    # subtree spent, so its cost and calls figures are strictly larger than its
    # own — budgeting against the flat map would understate the very columns
    # ``_row_overhead`` exists to measure and re-open the D8/D11 clipping one
    # rollup later.
    overhead = max(
        _row_overhead(list(aggregate.by_provider.items()), width),
        _row_overhead([(row.session_id, row.aggregate) for row in structure], width),
    )
    name_cap = max(_MIN_NAME_COL, min(_MAX_NAME_COL, width - overhead))

    # The disclosure gutter exists only when SOMETHING can be expanded, and that
    # test is stable across expansion: roots are always visible, so "a visible
    # row has descendants" is the same question as "this forest has children at
    # all". It has to be stable, or opening a row would add a gutter to every
    # other row and slide the whole table sideways on one keypress.
    disclosure = any(row.expandable for row in structure)
    markers = {row.session_id: _disclosure_marker(row, present=disclosure) for row in structure}
    suffixes = {row.session_id: _row_suffix(row) for row in structure}

    # Keyed by SESSION ID, never by the rendered label. Two sessions can render
    # the same string (identical names, or names agreeing within the budget),
    # and a label-keyed dict does not merge those rows — it silently keeps the
    # last one and drops the rest, taking their tokens, calls and cost off the
    # screen entirely. On the operator's ledger that hid 46 sessions before
    # subagent naming existed and 355 after it, so the table is built as
    # (label, depth, aggregate) triples whose identity is the id.
    session_labels = _forest_labels(
        [(row.session_id, row.depth, row.aggregate) for row in structure],
        names,
        name_cap,
        # The marker and the ``+N`` count are printed inside the name column, so
        # they are charged to the label's budget for the same reason the nesting
        # prefix is: a label composed to the full budget and THEN given a prefix
        # is over the column, and the paint takes the tail back off with no
        # marker to say it did.
        reserved={
            row.session_id: cell_len(markers[row.session_id]) + cell_len(suffixes[row.session_id])
            for row in structure
        },
    )
    session_rows = [
        (
            _row_prefix(row.depth, markers[row.session_id])
            + session_labels[row.session_id]
            + suffixes[row.session_id],
            row.depth,
            row.aggregate,
        )
        for row in structure
    ]
    all_names = [n for n in aggregate.by_provider] + [label for label, _, _ in session_rows]
    if all_names:
        # Grow the name column with the frame: a wide card gets a roomier column
        # (up to 48) so its width is used; a narrow one stays compact (30).
        name_col = min(name_cap, max((cell_len(n) for n in all_names), default=0) + 1)
    else:
        name_col = 0

    if aggregate.by_provider:
        lines.append(_group_section("By provider", aggregate.by_provider, width, name_col))
        lines.append(Text())

    if session_rows:
        # Meta says the rollup happened, because a row whose figure exceeds its
        # own spend must say why — and it is also what tells a reader the
        # indented rows are already counted in the row above them.
        #
        # Keyed off ``disclosure`` (does the forest HAVE children) rather than
        # off the visible depths, because with the table collapsed there are no
        # indented rows on screen and the rollup has still happened. Reading it
        # off the paint would drop the one sentence that explains why a root's
        # figure exceeds its own spend, precisely in the default state.
        meta = "totals include subagents" if disclosure else ""
        if layout is not None:
            # The header is the block's first line; row ``i`` is one line below
            # it. Counted from the blocks composed SO FAR, because ``_repaint``
            # joins them with a single newline each — the same arithmetic the
            # body's line numbering uses, derived once here rather than guessed
            # by a caller counting sections it cannot see.
            layout.session_rows = structure
            layout.session_first_line = sum(block.plain.count("\n") + 1 for block in lines) + 1
        cursor_index = None
        if cursor is not None:
            cursor_index = next(
                (i for i, row in enumerate(structure) if row.session_id == cursor), None
            )
        # Resolved to an index the same way, and by SESSION ID for the same
        # reason: expanding a row inserts rows above every later index, so an
        # index carried across a rebuild would land on a different session. A
        # hovered id no longer on screen (its parent was collapsed) simply finds
        # nothing and paints no highlight, which is the honest answer.
        hover_index = None
        if hover is not None:
            hover_index = next(
                (i for i, row in enumerate(structure) if row.session_id == hover), None
            )
        lines.append(
            _session_section(
                session_rows,
                width,
                name_col,
                meta,
                cursor=cursor_index,
                hover=hover_index,
                suffixes=[suffixes[row.session_id] for row in structure],
                layout=layout,
            )
        )

    # Legend for the cost markers, drawn only when a ``+`` or ``$—`` is on
    # screen (review D1). ``dim`` so it reads as a footnote, not a row.
    if _needs_cost_legend(aggregate, forest):
        lines.append(Text())
        legend = Text()
        legend.append("  " + COST_LEGEND, style=dim)
        lines.append(legend)

    return lines


def _flatten_blocks(blocks: list[Text]) -> list[Text]:
    """One ``Text`` per BODY LINE, from the multi-line blocks the report builds.

    ``build_report`` composes section by section — a table is one ``Text``
    holding hundreds of lines — because that is the readable way to write it,
    and it keeps returning blocks on purpose: the plain-text tests index a
    section's own block, so flattening AT THE RENDERER would move a test's
    premise for no reader-visible gain. The body widget is where the shape has
    to change, because it addresses lines by index; the split therefore happens
    at that boundary, which is also where the blocks stop being readable.

    The line numbering ``ReportLayout.session_first_line`` publishes (and so
    every hit-test and every patch) is a count of THESE lines, which is why the
    two must agree about blank spacers: ``allow_blank=True`` is load-bearing,
    since a plain ``str.split`` drops the empty pieces and every index below a
    section break would shift by one. Splitting a ``Text`` is style-preserving,
    so a row keeps its spans.
    """
    flat: list[Text] = []
    for block in blocks:
        flat.extend(block.split("\n", allow_blank=True))
    return flat


#: Below this content width the per-group tables drop the cache column to keep
#: the cost column, which is the one this feature adds and the one a narrow
#: frame should not be the reason to lose. Cost + tokens + calls are the
#: irreducible trio; cache is the first to shed, exactly like the status band's
#: drop ladder.
_WIDE_TABLE_MIN = 72

#: Floor and ceiling on the session/provider name column. The floor keeps a
#: label readable on a narrow frame even when the numbers would rather have the
#: space (a 12-cell label is still a recognisable prefix); the ceiling is the
#: point past which more name stops buying legibility and just spreads the row.
_MIN_NAME_COL = 30
_MAX_NAME_COL = 48

#: Cells the body widget reserves for its own vertical scrollbar. The
#: ``#analytics-scroll`` rule sets ``scrollbar-gutter: stable``, so the column is
#: held whether or not the bar is drawn — a report composed against the full
#: card width therefore paints one cell wider than the box that receives it, and
#: the rightmost column is cut. The gutter belongs to the ``ReportView`` that
#: paints the report (body and viewport are one widget), which is why it is still
#: subtracted here and not measured off a child. Kept beside the width maths
#: that spends it.
_SCROLLBAR_GUTTER = 1


def _row_overhead(groups: "Sequence[tuple[str, UsageAggregate]]", width: int) -> int:
    """Cells one table row spends on everything that is NOT the name column.

    Measured from the same pieces ``_group_section`` paints, in the same order,
    rather than carried as a constant — that is the whole point. Design review
    D8: ``name_cap`` was ``min(48, width - 40)``, and ``- 40`` understated the
    real overhead by up to 15 cells. The budget therefore stepped 30 -> 48 the
    instant the content box reached 96 while the rest of the row still needed
    51-55, so the widest row jumped to 103 against a 96-cell box and terminal
    widths 114-120 silently clipped the ``% cache`` column off every row. The
    row still looked complete; it just ended in ``cach``.

    A constant cannot fix that, because the overhead is not constant: the cost
    column is sized to the widest figure actually present (``$4.20`` vs
    ``$3.4k+``), the calls column to the widest call count, and the cache column
    is dropped entirely below ``_WIDE_TABLE_MIN``. So it is computed from the
    groups being rendered, and the caller subtracts it from the content box. The
    arithmetic below mirrors the ``block.append`` sequence in
    :func:`_group_section` line for line; the two must be changed together.

    Design review D11 is the same error one column further along, and it is why
    every term here is now measured rather than assumed. The calls column was
    ``3 + 4 + len(" calls")`` — a 4-digit allowance chosen against a fixture
    rendering ``16 calls``. On the operator's real ledger ``anthropic`` has
    317,977 calls and eight sessions are past 9,999, so the pad ran two cells
    over its allowance and pushed ``% cache`` off the box for the 13 most
    expensive rows of both tables across terminals 104-123. The rule this
    function now holds to: **no column width is assumed from a literal where the
    data can size it**, because a constant that is right for the fixture is
    wrong for the ledger, three times running.
    """
    if not groups:
        return 0
    # ``  {name}`` indent, then ``{tokens:>NN} tokens``.
    overhead = 2 + _tokens_col(groups) + len(" tokens")
    # ``   `` gap + the right-aligned cost cell, sized to the widest figure in
    # this table exactly as ``_group_section`` sizes it.
    overhead += 3 + max(len(format_cost(agg)) for _, agg in groups)
    # ``   {calls:>NN} calls`` — likewise sized to the widest count present.
    overhead += 3 + _calls_col(groups) + len(" calls")
    # ``   {pct:>4} cache`` — only when the frame is wide enough to keep it.
    # This 4 is the one literal that stays, because it is not an allowance: it
    # is the exact maximum :func:`format_percent` can return. That function is
    # total over its domain and its widest output is ``100%`` (``—`` is 1 cell,
    # ``99%``/``73%`` are 3), so the pad can never be overrun by data the way
    # the calls and cost pads could. Widen it if that formatter ever grows.
    if width >= _WIDE_TABLE_MIN:
        overhead += 3 + 4 + len(" cache")
    return overhead


def _calls_col(groups: "Sequence[tuple[str, UsageAggregate]]") -> int:
    """Cells the ``calls`` column needs for the widest count in this table.

    Floored at 4 so the ordinary small-ledger layout is unchanged (design review
    D11 asked only that the column stop being *understated*, not that it shrink
    on a fresh install), and measured above that so a six-digit provider total
    widens the column instead of overrunning it.

    Shared by :func:`_row_overhead` and :func:`_group_section` so the budget and
    the paint agree by construction rather than by two literals happening to
    match — the divergence between those two is precisely what D8 and D11 were.
    """
    return max(4, max((len(f"{agg.calls}") for _, agg in groups), default=0))


def _tokens_col(groups: "Sequence[tuple[str, UsageAggregate]]") -> int:
    """Cells the ``tokens`` column needs, floored at the historical 8.

    Audited as part of D11 and included for the same reason: ``{…:>8}`` is a pad,
    not a truncation, so it is the same latent defect as the calls column even
    though no plausible ledger reaches it today. :func:`format_tokens` abbreviates
    to a unit suffix, so 51B tokens (the operator's real total) is 3 cells and 8
    is not exceeded until roughly 10**16 tokens. That makes this a guard rather
    than a fix: it is a no-op on every real dataset and cannot narrow the column,
    but it means no term in the row overhead is a bare constant that data can
    outgrow silently.
    """
    return max(8, max((len(format_tokens(agg.total_tokens)) for _, agg in groups), default=0))


#: One indent step per level of session nesting. Two cells: enough that a
#: sub-row is unmistakably subordinate, small enough that it cannot be what
#: pushes the table past ``_WIDE_TABLE_MIN`` and costs everyone the cache column.
_NEST_INDENT = 2

#: Disclosure glyphs for a row whose subagent rows are hidden / shown. ``▸``/``▾``
#: is the codebase's existing plain-fallback vocabulary (``glyphs.PLAIN_ICON_DEFAULT``
#: is ``▸``) and both measure ONE cell, which the gutter arithmetic below relies
#: on — a two-cell glyph would shift every numeric column by a cell, the exact
#: class of defect ``_row_overhead`` exists to prevent.
_DISCLOSURE_COLLAPSED = "▸"
_DISCLOSURE_EXPANDED = "▾"

#: Cells the disclosure gutter costs a row: the glyph plus one space. Paid by
#: EVERY row of the table once any row is expandable, so the names stay in one
#: column — a table where expandable and childless rows start at different
#: x-positions reads as ragged rather than as structured. It is not paid at all
#: on a ledger with no subagents (see ``_disclosure_marker``), which is what
#: keeps a flat table byte-identical to the one this screen has always shipped.
_DISCLOSURE_CELLS = 2

#: The row cursor. ``❯`` matches the command and session pickers (a caret rather
#: than a reversed row: an inverted block reads as a selection the user MADE,
#: not as the position they are on). It is painted into the two leading indent
#: cells every row already spends, so a cursor costs the label budget nothing.
_ROW_CURSOR = "❯"


class _PointerAt(NamedTuple):
    """A bare screen coordinate with the two attributes the hit-test reads.

    :meth:`AnalyticsScreen._row_at` takes an "event" but uses only
    ``screen_x``/``screen_y``. Re-resolving the hover after the viewport moved
    has a coordinate and NO event: synthesising a real ``events.MouseMove`` for
    it would mean inventing a widget, a button state and deltas that no hit-test
    reads, and posting it would re-enter the handler. This is the coordinate,
    and nothing else — the same shape ``copy_picker`` uses for the same case.
    """

    screen_x: int
    screen_y: int


@dataclass(frozen=True)
class SessionRow:
    """One rendered row of the By-session table, before it has a label.

    Identity is the SESSION ID all the way to the paint (never the rendered
    label — two sessions can render the same string, and a label-keyed structure
    silently drops all but one of them; that hid 355 sessions on the operator's
    ledger). The screen keys its expansion state off ``session_id`` for the same
    reason: a repaint at a different width recomposes every label, so anything
    remembered by label would be forgotten by the next resize.

    ``descendants`` is the size of the whole subtree under this row, not the
    number of direct children. It is what the row advertises as the cost of
    expanding, because the question the collapse creates is "how much is hidden
    here" — and on the real ledger the two figures are equal for all but a
    handful of rows (1,637 depth-1 nodes against 8 at depth 2).
    """

    session_id: str
    depth: int
    aggregate: "UsageAggregate"
    descendants: int
    expanded: bool

    @property
    def expandable(self) -> bool:
        return bool(self.descendants)


@dataclass
class ReportLayout:
    """What :func:`build_report` just painted, for a caller that must drive it.

    A receiver rather than a return value because ``build_report`` returns the
    rendered lines and three callers (two scripts and the plain-text tests) want
    only those. The interactive screen needs one thing more — WHICH row is on
    WHICH body line — so it can put a cursor on a row, scroll that row into
    view, and map a mouse click back to the session it landed on. Recomputing
    that outside the renderer would mean building the forest a second time
    (31 ms on the operator's ledger) and, worse, would be a second definition of
    which rows are visible that could disagree with the paint.
    """

    #: The visible rows, in paint order. Collapsed subtrees are simply absent.
    session_rows: list[SessionRow] = field(default_factory=list)
    #: Body line index of ``session_rows[0]``. The section header sits one line
    #: above it, and row ``i`` is at ``session_first_line + i`` because every row
    #: is composed ``no_wrap`` and truncated to the content box.
    session_first_line: int = 0
    #: The ``(label, depth, aggregate)`` triples the table painted, in the same
    #: order as ``session_rows``. Kept because a single-row REPAINT needs the
    #: inputs the full paint used, and re-deriving them means walking the forest
    #: again (31 ms on the operator's ledger) as a second definition of what is
    #: visible that could disagree with the paint.
    session_triples: list[tuple[str, int, "UsageAggregate"]] = field(default_factory=list)
    #: The table's column widths AS PAINTED, alongside the switches that decide
    #: which columns exist. A repaint of one row must take every width from
    #: here: these are maxima over the whole table, and a caller that recomposes
    #: a single row from that row's OWN figures gets a narrower column and moves
    #: the row sideways (see :func:`_session_row_line`).
    name_col: int = 0
    tokens_col: int = 0
    cost_col: int = 0
    calls_col: int = 0
    #: Whether the ``% cache`` column is present — the width switch, not a width.
    show_cache: bool = False
    #: The ``+N subagents`` tail each row advertises, parallel to
    #: ``session_triples``. Part of the row's content, so a repaint needs it too.
    suffixes: list[str] = field(default_factory=list)


def _descendant_count(node: "SessionNode") -> int:
    """Nodes beneath ``node``, at any depth. Bounded by the forest's own depth cap."""
    return sum(1 + _descendant_count(child) for child in node.children)


def _disclosure_marker(row: SessionRow, *, present: bool) -> str:
    """The disclosure cell a row carries, or ``""`` when the table has no gutter.

    ``present`` is false when NOTHING in the table can be expanded, and then no
    row pays for the gutter. That is the same rule the hint line and the metric
    toggle follow (design D5): a control that cannot act must not be advertised,
    and a permanently blank two-cell column advertising an absent affordance is
    the visual form of the same defect. It also means a ledger that never ran a
    subagent renders exactly the table it rendered before this feature existed.

    A childless row gets BLANK cells, never a glyph: it is not interactive, and
    a glyph on it would promise a press that does nothing. 492 of the operator's
    595 roots are childless, so this is the majority case, not an edge.
    """
    if not present:
        return ""
    if not row.expandable:
        return " " * _DISCLOSURE_CELLS
    return (_DISCLOSURE_EXPANDED if row.expanded else _DISCLOSURE_COLLAPSED) + " "


def _row_suffix(row: SessionRow) -> str:
    """The hidden-row count a row advertises after its name, or ``""``.

    This is the information the collapse REMOVES, handed back in one cell-cheap
    form: without it a user cannot tell an expensive leaf session from an
    expensive session with ninety subagents under it, which is precisely the
    distinction they opened the screen to make.

    Shown in BOTH states, not just when collapsed. A suffix that disappeared on
    expand would change the widest label in the table, ``name_col`` is sized to
    that, and every numeric column would shift sideways on a keypress — a reflow
    the reader sees as the table twitching. Its width is charged to the label
    budget by :func:`_forest_labels`, exactly as the prefix is.

    Spelled out ("+12 subagents") rather than a bare "+12", which costs about
    ten cells of a 48-cell name column and is worth it: every other number on
    this row is money, tokens or calls, so a bare ``+12`` beside them reads as a
    quantity of the same kind. "subagents" is the word the section meta already
    uses, so this borrows the table's own vocabulary rather than adding one.
    """
    if not row.expandable:
        return ""
    noun = "subagent" if row.descendants == 1 else "subagents"
    return f" +{row.descendants} {noun}"


def _row_prefix(depth: int, marker: str = "") -> str:
    """The indent-and-glyph a row at ``depth`` carries before its label.

    A child carries a ``└`` glyph, not just the indent and the dim style
    (design D4). 21% of ROOTS are themselves unnamed 12-hex ids, so an unnamed
    root and a child differ by two leading spaces and a colour — and colour is
    the cue that disappears under ``NO_COLOR``, a weak-dim theme, or low vision.
    The glyph makes the hierarchy survive without it. ``/session``'s Tool
    surface section already uses ``└`` for exactly this, so this is the
    codebase's existing vocabulary rather than a new one.

    ``marker`` is the disclosure cell (:func:`_disclosure_marker`) and it lands
    AFTER the nesting indent, immediately before the name, rather than in a
    fixed gutter at the far left. A disclosure glyph is a control attached to
    one row's label; parked in a shared left gutter it would sit at the same
    x-position for a root and its grandchild, so the column of glyphs would say
    nothing about which of them a press would act on.

    Split out from the walk because the prefix is width the LABEL cannot also
    spend: it is prepended after condensing, so :func:`_forest_labels` has to
    subtract exactly this many cells from a nested row's budget. One function
    now defines the prefix for both the measurement and the paint, so the two
    cannot drift — the same rule ``_calls_col`` follows for the calls column.
    """
    if not depth:
        return marker
    return " " * ((depth - 1) * _NEST_INDENT) + "└ " + marker


def _forest_rows(
    forest: list["SessionNode"],
    expanded: "Collection[str] | None" = None,
) -> list[SessionRow]:
    """Flatten the session forest to the rows that are currently VISIBLE.

    Depth-first so a child sits directly under its parent, and each row carries
    the TREE total (own plus descendants) — the same figure its label is sorted
    by. A child row's dollars are therefore already counted in its parent's, and
    the section meta says so; only the ROOT rows sum to the table total. That
    invariant is untouched by collapsing: a hidden child was never a row that
    summed, and its parent's figure already contains it.

    ``expanded`` is the set of session IDs whose children are shown; everything
    else stops the walk at its own row. ``None`` means none — COLLAPSED is the
    default, which is the whole point. The operator's ledger renders 2,240 rows
    fully expanded against 595 collapsed, and the difference is a 0.65 s first
    paint against 0.23 s plus a full rebuild on every one of the resize events a
    terminal drag emits in a storm.

    Yields the session ID rather than a rendered label, because a label cannot
    be composed until the column budget is known and the budget is computed from
    the aggregates this walk selects. Identity stays the id all the way to the
    paint, which is what stops two rows that read alike from collapsing into
    one; :func:`_forest_labels` turns these into display text.
    """
    open_ids = frozenset(expanded or ())
    rows: list[SessionRow] = []

    def walk(node: "SessionNode", depth: int) -> None:
        is_open = node.session_id in open_ids
        rows.append(
            SessionRow(
                session_id=node.session_id,
                depth=depth,
                aggregate=node.total,
                descendants=_descendant_count(node),
                expanded=is_open,
            )
        )
        # A closed subtree is not walked at all. Skipping the RECURSION rather
        # than filtering afterwards is what makes the collapsed report cheap:
        # nothing under a closed row is measured, labelled, or composed.
        if is_open:
            for child in node.children:
                walk(child, depth + 1)

    for root in forest:
        walk(root, 0)
    return rows


def _forest_labels(
    structure: "Sequence[tuple[str, int, UsageAggregate]]",
    names: Mapping[str, str],
    name_cap: int,
    reserved: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Budgeted, collision-free display labels for every row, keyed by id.

    Rows are disambiguated per DEPTH, and that is sufficient for the rendered
    labels to be globally unique: ``_row_prefix`` is strictly wider at each
    level, so two rows at different depths already differ in the prefix the
    reader sees, and two rows at the same depth are separated by
    ``session_table_labels`` in the ordinary way. Depth is keyed here by the
    budget it produces, which is the same partition (the prefix determines the
    budget and the budget determines the prefix) expressed as the number each
    group is actually composed against.

    A nested row's budget is reduced by its own prefix (:func:`_row_prefix`).
    Neither slice's arithmetic knew about the other's here: the nesting change
    prepends the glyph AFTER the label is composed, and the budgeting change
    sized labels to fill ``name_cap`` exactly — so a depth-1 label composed to
    the full budget and then given a 2-cell prefix is 2 cells over the column
    it is padded to, and ``_group_section``'s truncation cuts the tail back off.
    That is a mid-word cut with no ellipsis at the exact widths where the label
    already fits, which is the defect this slice exists to remove. Subtracting
    the prefix first means the composed label plus its prefix is what
    ``name_cap`` promised, and the ellipsis lands where the reader can see it.

    ``reserved`` extends that rule to everything else a row prints INSIDE the
    name column — the disclosure marker and the ``+N`` hidden-row count — keyed
    by session id because those two vary per ROW rather than per depth (an
    expandable root and a childless one sit at the same depth and spend
    different numbers of cells). Same reasoning as the prefix: whatever the row
    prints beside its label is width the label cannot also have, and charging it
    afterwards is what produced the unmarked mid-word cut this budgeting exists
    to remove. When it is absent, only the prefix is charged, which is exactly
    the behaviour every caller had before the disclosure column existed.

    Floored at ``_MIN_LABEL_CHARS`` by ``session_table_labels`` itself, so a
    pathologically deep tree on a narrow frame degrades to short labels rather
    than to empty ones.
    """
    labels: dict[str, str] = {}
    # Group by the budget each row actually gets, so every disambiguation group
    # is decided against the width its members will really be composed at.
    by_budget: dict[int, dict[str, str]] = {}
    for sid, depth, _ in structure:
        extra = cell_len(_row_prefix(depth))
        if reserved is not None:
            extra += reserved.get(sid, 0)
        by_budget.setdefault(name_cap - extra, {})[sid] = names.get(sid, "")
    for budget, group in by_budget.items():
        labels.update(session_table_labels(group, budget))
    return labels


def _session_section(
    rows: list[tuple[str, int, "UsageAggregate"]],
    width: int,
    name_col: int,
    meta: str = "",
    *,
    cursor: int | None = None,
    hover: int | None = None,
    suffixes: Sequence[str] | None = None,
    layout: "ReportLayout | None" = None,
) -> Text:
    """The per-session table, pre-ordered and pre-indented by the forest walk.

    Deliberately NOT ``_group_section``: that one sorts its own rows, which
    would scatter a subtree across the table and destroy the nesting the walk
    just built. Everything else — the column layout, the ``_WIDE_TABLE_MIN``
    cache shed, the dimmed lower-bound ``+`` — is identical, because the two
    tables sit one above the other and must read as the same table.

    "Identical" is load-bearing rather than aspirational: the columns here are
    sized through the SAME ``_tokens_col``/``_calls_col``/``truncate_cells``
    helpers ``_group_section`` and ``_row_overhead`` use. This function forked
    from ``_group_section`` before those existed, and a fork that keeps the
    literal ``:>8``/``:>4`` pads is the D8/D11 defect preserved in a second
    place — the budget would be measured through the helpers while the paint
    used constants, so the two would disagree by exactly the amount the ledger
    exceeds the fixture (``anthropic`` at 317,977 calls overruns ``:>4`` by two
    cells and pushes ``% cache`` off the box). Sizing runs over the rows THIS
    table paints, which carry subtree totals, so it matches what
    ``build_report`` budgeted against.

    ``cursor`` is the index in ``rows`` the keyboard is currently on, painted
    into the two leading indent cells every row already spends — so the cursor
    costs the label budget nothing and cannot shift a numeric column. ``None``
    (the default, and what every non-interactive caller passes) paints no
    cursor at all, which keeps the plain-text renderers and the scripts on the
    same output they had before the row cursor existed.

    ``hover`` is the index the POINTER is over, and it is deliberately a second
    and independent index from ``cursor``: the mouse and the keyboard are two
    selection models that must be able to sit on different rows without either
    dragging the other. It is painted as a background tint across the row's full
    cell width and nothing else — no inserted glyph, no changed indent — because
    this table's "no reflow on interaction" property is load-bearing (three
    review rounds of #873 pinned it), and a hover that reflowed the table under
    a moving pointer would move the very row the user is aiming at.

    The tint is ``tint-select``, the same ground the ``/copy`` picker's hover
    uses, so a highlighted row means one thing across the app. The caret is what
    DISCRIMINATES hover from the keyboard cursor: on the cursor row the ground
    is ``tint-select-hi`` (a step up, so "the pointer is on the row the keyboard
    is also on" is its own visible state) and the ``❯`` is present either way.
    Hover alone never paints a caret — two carets would be two claims about
    where ``enter`` acts.

    ``suffixes`` is the ``+N subagents`` tail already composed into each label,
    handed over separately ONLY so it can be painted ``dim`` (review D3). It is
    metadata about the row, not part of the session's name, and at full
    foreground it scanned as the end of the title — while ``calls``, ``% cache``
    and the section meta beside it are all dim. Passing the string rather than a
    length keeps the split honest under truncation: a suffix the budget cut is
    simply not found at the tail and the row paints in one style, as before.

    ``layout`` publishes everything a LATER single-row repaint needs — the row
    triples, the label budget, the four column widths and their switch. The
    screen owns a long-lived layout and asks for one row to be recomposed on
    every pointer crossing; without these figures the patch path would have to
    walk the forest and re-measure the columns a second time, which is both the
    31 ms it exists to avoid and a second definition of the table's geometry
    that could disagree with the paint. ``None`` (the default, and what the
    plain-text callers and the scripts pass) just skips publishing.
    """
    dim = semantic_style("dim")
    block = section_header("By session", meta)
    if not rows:
        block.append("\n  (none)", style=dim)
        return block

    # Every column is a maximum over the WHOLE table, which is exactly why it is
    # published rather than left in the loop: a single-row repaint recomposes
    # one row against the table's columns, and a row measured against ITSELF
    # gets a narrower column and shifts sideways (the width trap).
    target = layout if layout is not None else ReportLayout()
    pairs = [(label, agg) for label, _, agg in rows]
    target.session_triples = rows
    target.name_col = name_col
    target.tokens_col = _tokens_col(pairs)
    target.cost_col = max(len(format_cost(agg)) for _, agg in pairs)
    target.calls_col = _calls_col(pairs)
    target.show_cache = width >= _WIDE_TABLE_MIN
    target.suffixes = list(suffixes) if suffixes is not None else [""] * len(rows)
    for index, row in enumerate(rows):
        block.append("\n")
        block.append_text(_session_row_line(target, index, row, cursor=cursor, hover=hover))
    return block


def _session_row_line(
    layout: ReportLayout,
    index: int,
    row: tuple[str, int, "UsageAggregate"],
    *,
    cursor: int | None,
    hover: int | None,
) -> Text:
    """ONE row of the per-session table, composed from the LAYOUT's columns.

    One composer with two call sites: the full paint loops over the rows, and a
    hover/cursor change recomposes the one or two rows whose appearance changed
    instead of the whole table. A second composer for the patch path would be a
    second definition of what a row looks like, and the two would drift.

    **The argument it takes is the layout, not a row list, and that is the
    point.** ``tokens_col``/``cost_col``/``calls_col`` are maxima over every row
    in the table, so composing a row against its OWN figures silently changes
    the numeric columns: measured on the operator's ledger, a patch that
    re-entered the table composer with a one-row list produced rows shifted
    sideways in 24 of the 26 crossings, and 26 of 26 were byte-identical to a
    full recompose only once the widths were hoisted out of the loop and read
    from the layout. Passing ``ReportLayout`` rather than the widths keeps that
    trap out of reach of a future caller, which a bag of four integers would not.

    The two indices are the paint's current selection state, both resolved
    against the layout's row order: ``cursor`` is the row the keyboard is on,
    ``hover`` the row the pointer is on. A row is a pure function of the layout,
    its index and those two indices, which is what lets the patch path recompose
    a row and get exactly the cells the full paint would have produced.
    """
    fg = semantic_style("fg")
    dim = semantic_style("dim")
    accent = semantic_style("accent")
    label, depth, agg = row
    on_cursor = cursor is not None and index == cursor
    on_hover = hover is not None and index == hover
    line = Text()
    # A nested row is dimmed as well as indented: its dollars are already inside
    # the root above it, so it must not compete visually with the rows that
    # actually partition the total.
    style = fg if depth == 0 else dim
    if on_cursor:
        line.append(f"{_ROW_CURSOR} ", style=accent)
    else:
        line.append("  ")
    # TRUNCATE as well as pad, in CELLS, exactly as ``_group_section`` does: a
    # bare ``{label:<{name_col}}`` pushes every numeric column right by whatever
    # the label overran, and the cost column is the one thing this screen exists
    # to let you scan straight down. Labels arrive budgeted (prefix included), so
    # this is a backstop rather than the mechanism.
    clipped = truncate_cells(label, layout.name_col)
    pad = " " * max(0, layout.name_col - cell_len(clipped))
    suffix = layout.suffixes[index] if index < len(layout.suffixes) else ""
    # The cursor row keeps ONE style across the whole label: the caret's
    # highlight is what says "you are here", and breaking it in the middle would
    # read as two spans rather than one selected row.
    if suffix and not on_cursor and clipped.endswith(suffix):
        line.append(clipped[: len(clipped) - len(suffix)], style=style)
        line.append(suffix, style=dim)
        line.append(pad)
    else:
        line.append(clipped + pad, style=accent if on_cursor else style)
    line.append(f"{format_tokens(agg.total_tokens):>{layout.tokens_col}} tokens", style=style)
    line.append("   ")
    append_cost(line, agg, layout.cost_col, style, dim)
    line.append(f"   {agg.calls:>{layout.calls_col}} calls", style=dim)
    if layout.show_cache:
        line.append(f"   {format_percent(agg.cache_hit_rate):>4} cache", style=dim)
    if on_hover:
        # A BACKGROUND laid over the finished row, never a re-styling of it:
        # `Text.stylize` adds a span, so every foreground the row already chose
        # (dim suffix, dim calls, accent caret) survives underneath and only the
        # ground changes. Re-composing the row in a "hover style" would flatten
        # those distinctions exactly on the row the user is looking at hardest.
        #
        # Spanned to the row's own composed cells rather than padded out to the
        # box: the table is narrower than the card whenever the name column hits
        # its cap (100 cells inside a 134-cell box), and a tint run to the box
        # edge would highlight a band of empty margin that is not part of the
        # table. Every row composes to the same cell count, so the tint is a
        # clean rectangle down the table. The run starts at 0 because this is one
        # row's `Text`, not a slice of the multi-row block the full paint builds.
        ground = "tint-select-hi" if on_cursor else "tint-select"
        line.stylize(Style(bgcolor=theme_mod.semantic_color(ground)), 0, len(line.plain))
    return line


def _group_section(
    title: str,
    groups: "Mapping[str, UsageAggregate] | Sequence[tuple[str, UsageAggregate]]",
    width: int,
    name_col: int,
) -> Text:
    """A per-provider or per-session table as one multi-line ``Text`` block.

    Takes either a name->aggregate mapping (the provider table, whose keys are
    genuinely unique) or a SEQUENCE of ``(label, aggregate)`` pairs (the session
    table, whose labels are display text and may repeat). The pair form exists
    because a dict keyed by a rendered label silently drops every row after the
    first collision; rows are identified upstream by session id and arrive here
    already labelled.

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
    pairs = list(groups.items()) if isinstance(groups, Mapping) else list(groups)
    ordered = sorted(
        pairs,
        key=lambda kv: (kv[1].cost_micro, kv[1].total_tokens),
        reverse=True,
    )
    if not ordered:
        block.append("\n  (none)", style=dim)
        return block

    show_cache = width >= _WIDE_TABLE_MIN
    cost_col = max(len(format_cost(agg)) for _, agg in ordered)
    # Sized from the data through the SAME helpers ``_row_overhead`` budgets
    # with, so the space reserved and the space painted cannot drift apart.
    # These were literal ``:>8``/``:>4`` pads; a per-row pad also left the
    # column ragged once counts varied in width (``317977 calls`` beside
    # ``16 calls`` put the two ``calls`` labels at different offsets), which
    # defeats scanning the column straight down exactly as D3 described.
    tokens_col = _tokens_col(ordered)
    calls_col = _calls_col(ordered)
    for name, agg in ordered:
        block.append("\n")
        # TRUNCATE as well as pad (design review D3). ``{name:<{name_col}}``
        # alone is a pad and nothing else, so any name wider than the column
        # pushed tokens/cost/calls right by however much it overran and the
        # numeric columns stopped lining up — the cost column is the one thing
        # this screen exists to let you scan straight down. Provider names reach
        # here uncondensed and session labels are already budgeted to
        # ``name_col``, so this is a backstop for the former and a no-op for the
        # latter; ``truncate_cells`` measures in CELLS, matching the pad.
        block.append(f"  {truncate_cells(name, name_col):<{name_col}}", style=fg)
        block.append(f"{format_tokens(agg.total_tokens):>{tokens_col}} tokens", style=fg)
        # Cost sits next to tokens as the other headline number, in full-strength
        # ``fg`` — it is the answer this feature exists to give, not a footnote.
        # The lower-bound ``+`` is dimmed by ``append_cost`` (review D1).
        block.append("   ")
        append_cost(block, agg, cost_col, fg, dim)
        block.append(f"   {agg.calls:>{calls_col}} calls", style=dim)
        if show_cache:
            block.append(f"   {format_percent(agg.cache_hit_rate):>4} cache", style=dim)
    return block


class AnalyticsScreen(ModalScreen[None]):
    """Full-screen, scrollable, Esc-dismissable usage analytics.

    Pushed by ``/analytics /usage`` and dismissed with Esc (or ``q``), which
    restores the previous view exactly — the screen reads the ledger and shows
    it, it never mutates anything, so leaving it is a plain pop with no state to
    reconcile. Modelled on :class:`SessionPickerScreen`: a centred card over a
    dimmed transcript, and a scrolling body so a long per-session table scrolls
    rather than clipping.

    The body is a :class:`ReportView` — one widget that is BOTH the scroller and
    the renderer — rather than a ``Static`` inside a ``VerticalScroll``. That is
    not a preference: the two-widget shape re-renders the whole report on every
    dirty repaint, and this screen repaints on every pointer row crossing (see
    the module docstring for the measured cost). Keeping the body and the
    viewport in one widget is also what makes the hit-test honest: the row under
    a screen coordinate is the viewport's own geometry, not a second widget's
    region that lagged a scroll by a frame.
    """

    BINDINGS = [
        Binding("escape", "dismiss_screen", "Back", show=False),
        Binding("q", "dismiss_screen", "Back", show=False),
        Binding("t", "toggle_metric", "Cost/tokens", show=False),
        # ``priority=True`` on the row keys, and it is load-bearing rather than
        # defensive. Focus sits on the scroller (the ``ReportView`` body, whose
        # ``ScrollableContainer`` base handles the arrows itself whenever it can
        # still scroll) — so a plain screen binding is reached only at the ends
        # of the travel. That is why the
        # ``action_scroll_up``/``down`` this replaces were effectively dead on
        # any report tall enough to scroll, and why the cursor would otherwise
        # move only on a short one (measured: it worked at 110x40, where the
        # body fits, and never at 110x20, where it does not).
        #
        # Taking the arrows from the container is the deliberate half of the
        # "one gesture owns the viewport" split: KEYS move the cursor and scroll
        # it into view, while the WHEEL and the scrollbar still move the viewport
        # alone and leave the cursor where it was.
        Binding("up", "move_up", "Up", show=False, priority=True),
        Binding("down", "move_down", "Down", show=False, priority=True),
        Binding("enter", "toggle_row", "Expand/collapse", show=False, priority=True),
        Binding("space", "toggle_row", "Expand/collapse", show=False, priority=True),
        Binding("right", "expand_row", "Expand", show=False, priority=True),
        Binding("left", "collapse_row", "Collapse", show=False, priority=True),
        Binding("e", "toggle_all", "Expand/collapse all", show=False),
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
        #: Session IDs whose subagent rows are shown. Empty by default: the
        #: operator's ledger has 1,645 subagent rows against 595 roots, and
        #: painting them all cost a 0.65 s first paint plus a full rebuild on
        #: every resize event a terminal drag emits.
        #:
        #: Keyed by session ID and NEVER by the rendered label or the row index.
        #: A label is recomposed at every width (two sessions can render the same
        #: string; that silently dropped 355 sessions once), and an index moves
        #: the moment any row above it expands.
        self._expanded: set[str] = set()
        #: Session ID the row cursor is on, or ``None`` before the first paint
        #: has told us which rows exist. Same keying rule, same reasons.
        self._cursor: str | None = None
        #: Session ID the POINTER is over, or ``None``. A SECOND selection model
        #: beside ``_cursor`` and deliberately independent of it: the pointer can
        #: rest on one row while the keyboard cursor sits on another, and neither
        #: may drag the other around. Same id-keying rule as ``_cursor``.
        self._hover: str | None = None
        #: Last known pointer position, in SCREEN coordinates. Kept because the
        #: viewport moves under a resting pointer — a real terminal sends no
        #: ``MouseMove`` while only the wheel turns — so the highlight has to be
        #: re-resolved from a coordinate that has no event behind it. See
        #: ``_refresh_hover``; this is ``copy_picker``'s ``_pointer_at`` and it
        #: exists here for the identical reason, on a surface that scrolls much
        #: further.
        self._pointer_at: tuple[int, int] | None = None
        #: What the last paint put where — the rows it drew and the body line the
        #: first of them landed on. Written by ``build_report`` through the
        #: ``ReportLayout`` receiver so cursor movement and scroll-to-cursor read
        #: the SAME notion of visible rows the paint used, rather than a second
        #: one derived alongside it that could disagree.
        self._layout = ReportLayout()
        #: Content-box width the body was last composed at. ``on_resize`` fires a
        #: storm of events during a drag and each rebuild was 0.5-0.6 s on a real
        #: ledger — the freeze in the user's report. Almost all of those events
        #: leave ``_card_width()`` unchanged (the card is 90% of the terminal,
        #: capped, and integer-floored, so several terminal widths map to one box
        #: and the CAP makes every width past 156 identical), and a rebuild at an
        #: unchanged width cannot produce different output. So the width is the
        #: repaint's cache key.
        self._painted_width: int | None = None
        #: Whether a hover re-resolve is already queued for after the next
        #: refresh. ``scroll_y`` is an ANIMATED value — one ``pagedown`` fired 28
        #: notifications — so the re-resolve is coalesced to one per frame rather
        #: than run per notification at 12 ms a repaint. See ``_viewport_moved``.
        self._hover_refresh_pending = False
        #: Lazily built by ``_forest()``; see there for why it is cached.
        self._forest_cache: list["SessionNode"] | None = None
        self._title: Static
        #: The body IS the viewport — one ``ReportView``, aliased twice because
        #: the screen has always spoken of them separately (``_body`` where it
        #: paints, ``_scroll`` where it scrolls) and the geometry is now the
        #: same object's either way.
        self._body: ReportView
        self._scroll: ReportView

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            # Held so the metric toggle can repaint the pinned title in place
            # (the ``bars: cost``/``bars: tokens`` suffix — reviews U1/U4).
            self._title = Static(self._title_text(), id="analytics-title")
            yield self._title
            # One widget for the body AND the viewport, deliberately: ``_body``
            # and ``_scroll`` are the same object, so the row under a screen
            # coordinate is the scroller's own geometry (see ``_row_at``) and
            # none of this screen's state can disagree with itself about where a
            # row is. The id stays ``analytics-scroll`` because that rule is
            # what gives the body its height and scrollbar chrome; it must NOT
            # become ``analytics-body``, whose ``height: auto`` would collapse
            # the viewport to the report's height and take the scrolling away.
            view = ReportView(id="analytics-scroll")
            self._body = self._scroll = view
            yield view
            self._hint = Static(self._hint_text(scrollable=False), id="analytics-hint")
            yield self._hint

    def on_mount(self) -> None:
        self._repaint()
        # After layout settles: ``max_scroll_y`` is only meaningful once the
        # body has been measured against the viewport.
        self.call_after_refresh(self._sync_hint)
        self.call_after_refresh(self._place_initial_cursor)
        # THE WHEEL GOTCHA, covered at its source. A real terminal sends no
        # ``MouseMove`` while only the wheel turns, so the rows slide under a
        # resting pointer and the highlight stays painted on a row the pointer
        # is no longer over. Watching the container's own ``scroll_y`` catches
        # every way the viewport can move — wheel, scrollbar drag, page/home/end
        # keys, and ``_scroll_cursor_into_view`` — as one hook, instead of a
        # ``_refresh_hover()`` call bolted onto each of those paths that the
        # next new one would forget.
        #
        # A wheel notch measurably never reaches a screen-level handler: Textual
        # stops the event on the container while it can still scroll, so a
        # ``on_mouse_scroll_*`` override here would run only at the ends of the
        # travel (the trap AGENTS.md records for ``/settings``). The watcher sees
        # all of them.
        self.watch(self._scroll, "scroll_y", self._viewport_moved, init=False)

    def _place_initial_cursor(self) -> None:
        """Draw the caret from frame 1, without moving the viewport.

        The hint says ``enter expand`` on the opening frame, but with no cursor
        set the first ``enter`` only PLACED one — so the advertised key took two
        presses to do the advertised thing (review D2; the PR's own capture
        script pressed ``enter`` twice for exactly this reason). Placing it at
        mount makes every ``enter`` act.

        Placed WITHOUT scrolling: the opening frame is the top of the report and
        scrolling to row 0 of a table 57 lines down would throw the reader out of
        the totals block they opened the screen to read. This is the same rule
        U2 asks for — a cursor never teleports the viewport — so the two are one
        behaviour, not two. When the table is off screen the cursor is simply
        parked on row 0 unpainted; the first arrow press then line-scrolls
        (``_table_on_screen``) rather than jumping to it.
        """
        if self._cursor is not None:
            return
        rows = self._layout.session_rows
        if not rows:
            return
        index = self._visible_row_index()
        self._cursor = rows[index if index is not None else 0].session_id
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        # NOT an unconditional rebuild. Dragging a terminal edge emits a burst of
        # resize events, and rebuilding the whole report per event measured
        # 0.5-0.6 s each against a real ledger — the "freezing" in the report,
        # since the burst arrives faster than one rebuild finishes. ``_repaint``
        # is a no-op when ``_card_width()`` is unchanged, and the card is a
        # floored 90% of the terminal capped at 140, so many terminal widths map
        # to one content box and every width past 156 maps to the same one.
        #
        # The HINT is still re-synced every time: it depends on whether the body
        # overflows the viewport, which changes with HEIGHT, and height does not
        # enter ``_card_width`` at all.
        self._repaint(force=False)
        self.call_after_refresh(self._sync_hint)
        # A resize relays the rows under a pointer that did not move, so the
        # highlight has to be re-resolved for the same reason a scroll does.
        # After the refresh, because it reads the new geometry.
        self.call_after_refresh(self._refresh_hover)

    def _card_width(self) -> int:
        # Track the CSS card (90% of the terminal, capped at 140 — see the
        # ``.analytics-panel`` rule) minus its 2-cell horizontal padding each
        # side, so the report's own column maths matches the width it is
        # actually painted into. The cap here mirrors the CSS cap; raising one
        # without the other either wastes the frame or overruns it.
        #
        # ``_SCROLLBAR_GUTTER`` comes off the top because ``#analytics-scroll``
        # sets ``scrollbar-gutter: stable`` (see the stylesheet): the column is
        # reserved whether or not the bar is currently drawn, so the report is
        # ALWAYS painted one cell narrower than the card. Counting it here
        # rather than at the call sites keeps one number describing "cells the
        # report may paint into" — measured, not assumed: at a 114-column
        # terminal the card is 96 and ``scrollable_content_region`` is 95. It
        # was measured with a ``Static`` body inside a ``VerticalScroll`` and is
        # deliberately unchanged now the body and the viewport are one widget:
        # the reserved column is the same column either way, and the frame
        # comparison pins that the paint did not move.
        try:
            terminal = self.app.size.width
            card = min(140, int(terminal * 0.9))
            return max(40, card - 6 - _SCROLLBAR_GUTTER)
        except Exception:  # noqa: BLE001 — before mount, a sane default
            return 88 - _SCROLLBAR_GUTTER

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

    def _hint_width(self) -> int:
        """Cells the hint ``Static`` can actually paint into on this frame.

        The hint is ``height: 2`` with ``padding-top: 1`` (see
        ``#analytics-hint`` in the stylesheet), i.e. exactly ONE content line.
        Anything wider wraps to a second line that is never painted — which is
        why the width has to be measured rather than assumed.

        The pre-mount fallback to the card width is deliberately approximate and
        is not load-bearing (review R8). It is reached only from ``compose``,
        where the layout carries no rows yet, so the only candidates are
        ``esc back · t cost/tokens`` (24 cells) and ``esc back`` — both inside
        the 40-cell floor ``_card_width`` cannot go below — and ``_sync_hint``
        replaces the text after the first refresh, before the frame the reader
        sees. It only has to be sane, not exact.
        """
        hint = getattr(self, "_hint", None)
        if hint is not None and hint.is_mounted:
            width = hint.content_size.width
            if width > 0:
                return width
        return self._card_width()

    def _hint_text(self, *, scrollable: bool) -> Text:
        # The scroll affordance is advertised only when there is something to
        # scroll (D5): the empty state and any report that fits told the user to
        # scroll a screen that could not, which reads as a dead control.
        faint = Style(color=theme_mod.semantic_color("faint"))
        # ``↑↓`` is described as "row" whenever session rows exist, NOT only when
        # something is expandable (review R2). The cursor moves on every ledger
        # with rows — ``_move_cursor`` never consults expandability — so the old
        # ``↑↓ scroll`` on a subagent-free ledger described a control the screen
        # did not have. It now also genuinely scrolls above the table, but "row"
        # is the half a reader needs named: line-scrolling is the behaviour they
        # already expect from an arrow on a report.
        rows = bool(self._layout.session_rows)
        expandable = self._expandable_rows()
        # Ordered widest-first; the first one that FITS the real box is painted.
        # Degrading by dropping whole clauses (rather than letting the tail wrap
        # away unpainted) is the fix for U3: at 50 and 60 columns the cut used to
        # eat ``enter expand · e all`` and leave a dangling ``·``, so exactly
        # where the table is most cramped the expand keys stopped being
        # documented while the ``▸`` glyphs stayed on screen inviting the press.
        #
        # ``esc back`` is shed BEFORE the expand keys because ``esc`` is the one
        # key a reader tries unprompted on a modal; ``enter``/``e`` are the ones
        # they cannot guess. Both keep their names in every tier that mentions
        # them — an abbreviation that drops the key name documents nothing.
        if expandable:
            candidates = [
                "esc back · t cost/tokens · ↑↓ row · enter expand · e all",
                "esc back · ↑↓ row · enter expand · e all",
                # ``esc back`` goes BEFORE ``expand`` is abbreviated away — the
                # rule two lines up, which the ladder used to contradict at its
                # own third rung (review D5). A 33-cell box took
                # ``esc back · ↑↓ row · enter · e all``, leaving ``enter · e
                # all`` to be read as two unexplained keys, while the 29-cell
                # tier below keeps ``enter expand`` and fits every box in the
                # 44-50 column band (measured: 33-39 cells).
                "↑↓ row · enter expand · e all",
                "↑↓ row · enter · e all",
                "enter · e all",
            ]
            if not self._has_charts():
                candidates = candidates[1:]
        elif rows:
            candidates = [
                "esc back · t cost/tokens · ↑↓ row",
                "esc back · ↑↓ row",
                "↑↓ row",
            ]
            if not self._has_charts():
                candidates = candidates[1:]
        elif scrollable:
            # No rows at all (an empty or provider-only report): there is no
            # cursor to move, so the arrows only ever scroll and the hint says so.
            candidates = [
                "esc back · t cost/tokens · ↑↓ scroll",
                "esc back · ↑↓ scroll",
                "↑↓ scroll",
            ]
            if not self._has_charts():
                candidates = candidates[1:]
        else:
            candidates = ["esc back · t cost/tokens", "esc back"]
            if not self._has_charts():
                candidates = ["esc back"]

        width = self._hint_width()
        # The narrowest tier is the floor: below it there is nothing left to shed
        # and a clipped hint beats no hint.
        text = next((c for c in candidates if cell_len(c) <= width), candidates[-1])
        hint = Text(no_wrap=True, overflow="crop")
        hint.append(text, style=faint)
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
        # ``self._layout`` is the RECEIVER, refreshed by every render: the rows
        # this paint drew and where it drew them. It must be written by the same
        # call that composes the body or the cursor could point into a layout the
        # screen is no longer showing.
        return build_report(
            self._aggregate,
            self._card_width(),
            daily=self._daily,
            monthly=self._monthly,
            window_totals=self._window_totals,
            metric=self._metric,
            expanded=self._expanded,
            cursor=self._cursor,
            hover=self._hover,
            layout=self._layout,
            # Every arrow key repaints, and rebuilding the forest each time was
            # 34 ms of the ~0.25 s that made the cursor feel laggy rather than
            # instant. Safe to share because the aggregate is a snapshot read on
            # a worker thread before the screen was pushed and never mutates.
            forest=self._forest(),
        )

    def _expandable_rows(self) -> bool:
        """Whether the CURRENT paint has a row that can be expanded.

        Read off the last layout rather than recomputed from the aggregate: this
        gates a hint line, and a hint that describes a table other than the one
        on screen is the defect the D5 rule exists to prevent. Before the first
        paint the layout is empty and the answer is no, which is correct — no
        rows are on screen to expand.
        """
        return any(row.expandable for row in self._layout.session_rows)

    def _repaint(self, *, force: bool = True) -> None:
        """Recompose the WHOLE body, unless a width-driven repaint changes nothing.

        ``force=False`` is the resize path (see ``on_resize``): a rebuild is
        0.5-0.6 s on a real ledger and a terminal drag emits a burst of resize
        events, so the ones that leave the content box unchanged must not each
        pay for one.

        This is the coarse path, and it is the right one whenever the REPORT's
        SHAPE changes: a metric flip re-draws the charts, an expand inserts rows
        and so renumbers every line below it, a resize recomposes at a new width.
        The narrow path for a change of two rows' COLOUR is ``_patch_rows``.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        width = self._card_width()
        if not force and width == self._painted_width:
            return
        body.set_lines(_flatten_blocks(self._report_lines()), width)
        self._painted_width = width

    def _patch_rows(self, indices: Sequence[int | None]) -> None:
        """Recompose and repaint ONLY the listed session rows.

        The point of the whole virtualized body. A hover change alters the
        composed text of exactly two rows — the one the pointer left and the one
        it entered — and re-entering ``build_report`` for that costs a full
        recompose of all 846 lines (measured: 40 ms per crossing with the
        virtualized body alone, against 4.0 ms for two patched rows).

        The rows are composed by :func:`_session_row_line`, the SAME function
        the full paint uses, fed the layout the full paint published. Taking the
        widths from anywhere else is the trap that misaligns a patched row.

        Out-of-range indices are dropped rather than clamping: an index that no
        longer names a row means the paint it came from is stale, and painting a
        different row's content on that line would be worse than leaving it.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        layout = self._layout
        triples = layout.session_triples
        if not triples:
            # Nothing painted yet (or no table in this report): fall back so a
            # caller never has to know which shape it is in.
            self._repaint()
            return
        cursor = self._cursor_index()
        hover = self._hover_index()
        for index in sorted({i for i in indices if i is not None}):
            if not 0 <= index < len(triples):
                continue
            body.set_line(
                layout.session_first_line + index,
                _session_row_line(layout, index, triples[index], cursor=cursor, hover=hover),
            )

    def render_lines_for_test(self) -> list[str]:
        """The report as plain strings — what a user reads.

        Reads the renderer, not the widget: this is what the report SAYS, which
        is what the plain-text tests are about. ``_body.lines_for_test()`` is the
        report as PAINTED (the lines the strips are built from) and is what a
        test that asks about the shape of the body — one line per row, patched
        rows included — should read.
        """
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

    # -- the session-row cursor ----------------------------------------------
    #
    # ``↑↓`` used to scroll the container by a line. It now moves a row cursor
    # and scrolls that row into view, which is the arrangement the codebase
    # already settled on for every list that IS the page: the WHEEL and the
    # scrollbar move the viewport and leave the cursor alone, while keys that
    # move the CURSOR scroll it into view. Letting both drive the viewport from
    # one gesture is what made ``/settings`` snap back to the top on a wheel
    # notch at the bottom.
    #
    # ``pageup``/``pagedown``/``home``/``end`` deliberately stay pure VIEWPORT
    # moves. This screen is mostly NOT a list — the totals, two charts and the
    # input attribution sit above the table — so paging is how a reader travels
    # between sections, and rebinding it to the cursor would strand them in the
    # one section that has rows.
    #
    # Movement CLAMPS rather than wraps: this is a full-page surface several
    # times its viewport, the documented exception (``/settings``,
    # ``session_picker._move_to``). Wrapping from the last session back to the
    # totals would throw the reader out of the section they were working in.

    def _visible_ids(self) -> list[str]:
        return [row.session_id for row in self._layout.session_rows]

    def _cursor_index(self) -> int | None:
        """Where the cursor sits in the CURRENT paint, or ``None`` if nowhere.

        The cursor is a session id, so it can legitimately be off-screen: the row
        it names is inside a subtree that has since been collapsed. Callers treat
        that as "no cursor" and re-home, rather than guessing an index.
        """
        if self._cursor is None:
            return None
        return next(
            (
                i
                for i, row in enumerate(self._layout.session_rows)
                if row.session_id == self._cursor
            ),
            None,
        )

    def _hover_index(self) -> int | None:
        """Where the POINTER sits in the CURRENT paint, or ``None`` if nowhere.

        The mirror of ``_cursor_index``, and it exists for the same reason: the
        hover is stored as a session id (so it survives a rebuild), while a
        repaint needs the row it lands on. Resolved against the same layout the
        paint published, so the index a patch is composed for is the index the
        row is painted at.
        """
        if self._hover is None:
            return None
        return next(
            (i for i, row in enumerate(self._layout.session_rows) if row.session_id == self._hover),
            None,
        )

    def _table_on_screen(self) -> bool:
        """Whether any session row is inside the viewport right now.

        This is the hand-off test between the two jobs ``↑↓`` has to do on this
        screen. The report is mostly NOT a list: on the operator's ledger the
        session table starts ~57 body lines down, behind the totals block, both
        bar charts and the input attribution. A cursor addresses ONLY table
        rows, so while the reader is still up in that non-row content there is
        nothing for the cursor to move and the arrows must keep doing what they
        did before this feature existed — line-scroll the container.

        Measured before the fix (real ledger, 120x40): ``end`` then 400 ``up``
        presses left the viewport pinned at y=56, i.e. the top of the report was
        unreachable by arrow key while the hint said ``↑↓ row`` (reviews
        R1/R2/D1/U1/U2 — four streams, one root cause).
        """
        rows = self._layout.session_rows
        scroll = getattr(self, "_scroll", None)
        if not rows or scroll is None or not scroll.is_mounted:
            return False
        height = scroll.size.height
        if height <= 0:
            return False
        top = scroll.scroll_offset.y
        first = self._layout.session_first_line
        last = first + len(rows) - 1
        # Any overlap between [first, last] and the viewport [top, top+height).
        return first <= top + height - 1 and last >= top

    def _visible_row_index(self) -> int | None:
        """Index of the first session row already ON SCREEN, or ``None``.

        Used to place a cursor that has none instead of teleporting to row 0.
        Homing to row 0 is what made ``up`` the largest jump on the keyboard
        (review U2: after ``end`` at y=629, one ``up`` landed at y=56 — 570
        lines BACKWARD), because ``_scroll_cursor_into_view`` then drags the
        viewport to wherever row 0 happens to be. Landing on a row the reader
        can already see keeps them where they were, which is the property that
        makes a cursor feel like a cursor rather than a jump.
        """
        rows = self._layout.session_rows
        scroll = getattr(self, "_scroll", None)
        if not rows or scroll is None or not scroll.is_mounted:
            return None
        height = scroll.size.height
        if height <= 0:
            return None
        top = scroll.scroll_offset.y
        first = self._layout.session_first_line
        # The first row at or below the viewport's top edge, if it is still
        # inside the viewport.
        index = max(0, top - first)
        if index >= len(rows) or first + index > top + height - 1:
            return None
        return index

    def _cursor_on_screen(self) -> bool:
        """Whether the cursor's row is inside the viewport right now.

        The cursor is ALLOWED to drift off screen — the wheel and the scrollbar
        move the viewport without touching it, which is the split AGENTS.md
        requires. But a KEY that acts on the cursor must not then drag the
        viewport back to wherever it drifted to: that is the 570-line backward
        jump review U2 measured after ``end``. So an arrow press re-homes an
        off-screen cursor to a row the reader can see, rather than moving it
        from a position they can no longer verify.
        """
        index = self._cursor_index()
        scroll = getattr(self, "_scroll", None)
        if index is None or scroll is None or not scroll.is_mounted:
            return False
        height = scroll.size.height
        if height <= 0:
            return False
        top = scroll.scroll_offset.y
        line = self._layout.session_first_line + index
        return top <= line <= top + height - 1

    # -- mouse ----------------------------------------------------------------
    # Public handler names on purpose: Textual dispatches ``_on_<event>`` and
    # then ``on_<event>``, so overriding the PRIVATE name shadows the base
    # ``Widget``'s own ``mouse_hover`` bookkeeping — the thing that drives every
    # ``:hover`` rule — and silently latches it on. ``command_picker`` documents
    # the same trap at its own mouse block.

    def _row_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """Index of the session row under a mouse event, or ``None``.

        Resolved against the VIEWPORT plus the scroll offset. Note that after
        the body was virtualized the viewport and the body are the SAME widget,
        so this is no longer a choice between two regions — but the reason the
        arithmetic reads ``scrollable_content_region`` and ``scroll_offset``
        rather than ``region`` is worth keeping, because it was measured: a hit
        test based on the body widget's ``region`` was recomputed by layout and
        lagged a scroll by a frame, while ``scroll_y`` is the reactive whose
        change notifies ``_viewport_moved`` in the first place. That basis read
        a stale ``body.y`` of -9 against an already settled offset of 0
        (measured), resolving the pointer 16 rows away from the truth and
        leaving the highlight stuck on a row it had scrolled past. This basis is
        correct at the instant the watcher runs.

        Two guards, and the FIRST is mandatory rather than defensive:

        - the point must be inside the SCROLL VIEWPORT, not merely inside the
          widget's own box. It always was, and it still is now that the body IS
          the scroller: the content region is the box minus the reserved
          scrollbar gutter, so the gutter column resolves to no row, and a
          coordinate BELOW the card (the hint line, or the modal's backdrop,
          which bubbles events from well outside the panel) is outside the box
          entirely and is rejected by the same test. The regression this pins is
          older and worse — while the body was a separate ``Static`` it
          overhung the viewport by everything scrolled out of sight (measured:
          27 rows on a 40-row frame) and its region CONTAINED the hint line, so
          a guard written against the body region alone resolved clipped
          coordinates to real rows;
        - and the resulting index must be a row that EXISTS. The table is one
          section of a long report: everything above it (totals, both charts,
          the input attribution, the provider table) and the legend below it
          resolve to indices outside ``session_rows`` and must be inert.

        Rows are addressable by their whole width. Every row composes ``no_wrap``
        to the same cell count, so the row is a clean rectangle and the ``x``
        coordinate carries no information the ``y`` does not already have.
        """
        scroll = getattr(self, "_scroll", None)
        if scroll is None or not scroll.is_mounted:
            return None
        viewport = scroll.scrollable_content_region
        if not viewport.contains(event.screen_x, event.screen_y):
            return None
        line = event.screen_y - viewport.y + int(scroll.scroll_offset.y)
        index = line - self._layout.session_first_line
        return index if 0 <= index < len(self._layout.session_rows) else None

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        self._pointer_at = (event.screen_x, event.screen_y)
        self._apply_hover(self._row_at(event))

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        """Drop the highlight when the pointer leaves the rows.

        ``Leave`` bubbles from the body widget, and its ORDERING relative to
        the ``MouseMove`` that lands somewhere else was measured before relying
        on it: moving row -> hint delivers ``move`` and then ``leave``, and hint
        -> row delivers ``leave`` (of the hint) and then ``move``. In neither
        direction does a stale ``leave`` arrive after the move that set a new
        hover, so this cannot erase a highlight that was just correctly placed.
        """
        self._pointer_at = None
        self._apply_hover(None)

    def _apply_hover(self, index: int | None) -> None:
        """Move the highlight to visible row ``index`` and set the pointer shape.

        One place, because every path that resolves a hover — a real pointer
        event and the eventless re-resolve after the viewport moved — must agree
        about what a hover DOES, including the pointer shape. Splitting them is
        how the shape ends up correct on one path and latched on the other.

        The hand appears over EXPANDABLE rows only. A childless row still takes
        the highlight — it is genuinely the row under the pointer and saying so
        is honest — but its click does nothing, and a hand over it would promise
        a click it does not keep. That is the majority case, not an edge one:
        492 of the operator's 595 roots are childless.
        """
        rows = self._layout.session_rows
        row = rows[index] if index is not None else None
        session_id = row.session_id if row is not None else None
        self.styles.pointer = "pointer" if row is not None and row.expandable else "default"
        if session_id == self._hover:
            return
        # The row the pointer left and the row it entered — no more. Both are
        # recomposed from the layout's own columns, so the cells are the ones a
        # full paint would have produced; the rest of the report is untouched.
        #
        # This covers entering and leaving the table as well as crossing inside
        # it, and that is worth stating because the obvious reading — "if either
        # index is missing, fall back to the whole body" — costs a full 846-line
        # recompose on two ordinary gestures (measured on the real ledger: 71 ms
        # entering the table, 40 ms leaving it, against 2-4 ms for a patched
        # pair). Neither index being absent changes anything for the OTHER row:
        # a row's cells are a function of the layout, its index and the current
        # cursor/hover indices, so with no hover painted anywhere the row to
        # un-tint is exactly the one ``previous`` names.
        #
        # ``_patch_rows`` still falls back to ``_repaint`` when there is no
        # layout to patch (nothing painted yet), so this needs no guard here.
        previous = self._hover_index()
        self._hover = session_id
        self._patch_rows([previous, index])

    def _refresh_hover(self) -> None:
        """Re-resolve the highlight against the LAST KNOWN pointer position.

        The rows move under a resting pointer — the wheel scrolls the report,
        the scrollbar drags it, ``_scroll_cursor_into_view`` moves it for a key
        — and a real terminal sends NO ``MouseMove`` for any of that. Without
        this the highlight stays painted on a row the pointer is not over, and
        on a report this tall it scrolls off the viewport entirely while the
        hand cursor stays (the defect ``copy_picker._refresh_hover`` was written
        for, on a surface that scrolls much further).

        Takes a coordinate with no event behind it, which is what
        ``_PointerAt`` is for: synthesising a ``MouseMove`` would mean inventing
        a widget, a button state and deltas no hit-test reads, and POSTING one
        would re-enter the handler.
        """
        if self._pointer_at is None:
            return
        self._apply_hover(self._row_at(_PointerAt(*self._pointer_at)))

    def _viewport_moved(self) -> None:
        """Re-resolve the hover after the viewport moved, at most once a frame.

        ``scroll_y`` is watched rather than each mover being patched, but it is
        an ANIMATED value: one ``pagedown`` on a real ledger fired 28 separate
        notifications as the scroll eased to its target. A repaint per
        notification is 12 ms each measured against the operator's 2,779-session
        ledger — a third of a second of rebuilds for one keypress, which is the
        freeze this panel's width cache already exists to prevent.

        So the work is coalesced: the flag marks the viewport dirty and
        ``call_after_refresh`` resolves it once, after the whole burst has been
        applied and the geometry it reads is final.
        """
        if self._hover is None and self._pointer_at is None:
            # Nothing to re-resolve and nothing painted: the common case (no
            # pointer has ever been over this screen) costs one attribute read.
            return
        if self._hover_refresh_pending:
            return
        self._hover_refresh_pending = True
        self.call_after_refresh(self._resolve_pending_hover)

    def _resolve_pending_hover(self) -> None:
        self._hover_refresh_pending = False
        self._refresh_hover()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Click a row to expand or collapse it.

        **The WHOLE row is the target, not just the ``\u25b8`` glyph.** The glyph is
        two cells against a hundred-cell row, and a two-cell target is a
        precision the mouse was invited in to avoid — the sibling pickers
        (`command_picker`, `copy_picker`, `session_picker`, `model_picker`) all
        make the whole row clickable and a user who learned that here would find
        this one screen demanding aim. The usual argument for the narrow target
        is accidental toggles, and it does not apply: the rows carry no other
        click action to collide with, and the mistake is instantly visible and
        undone by clicking again. The cost of the wide target is one wrong
        expand; the cost of the narrow one is a control most users never find.

        **The click also moves the keyboard cursor to the row.** The two
        selection models may sit apart while the pointer merely HOVERS — that is
        what makes hover a preview — but an ACT is different: after clicking row
        7 open, an ``enter`` that collapsed row 2 instead (because the caret
        never left it) would be the two models visibly disagreeing about which
        row is "current". Moving the caret on the acting gesture is what
        `copy_picker` does for the same reason, and it keeps the hint honest:
        ``enter expand`` names the row the user just acted on.

        A click on a CHILDLESS row is a clean no-op — no cursor move, no
        viewport move, no repaint. Not merely "toggling nothing": moving the
        caret there would scroll the view for a gesture the user experienced as
        doing nothing, and 492 of 595 real roots are childless, so this is the
        majority click.

        ``event.stop()`` because this is a modal over the transcript and a
        bubbled click hands the gesture to a parent — one gesture must not move
        two surfaces. Button 1 only, and the button is tested BEFORE any state
        changes: a right-click asking for a context menu must not toggle a row
        on its way to being ignored.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._row_at(event)
        if index is None:
            return
        event.stop()
        row = self._layout.session_rows[index]
        if not row.expandable:
            return
        self._cursor = row.session_id
        self.action_toggle_row()

    def _move_cursor(self, delta: int) -> None:
        rows = self._layout.session_rows
        if not rows:
            return
        index = self._cursor_index() if self._cursor_on_screen() else None
        if index is None:
            # No cursor to move. Two cases, and conflating them is the defect
            # the round-2 remediation fixes:
            #
            # 1. The table is not on screen — the reader is up in the totals or
            #    the charts. There is no row to point at, so the arrow keeps its
            #    pre-feature meaning and line-scrolls the container. Handled by
            #    the callers (``action_move_up``/``action_move_down``), which
            #    check ``_table_on_screen`` before ever reaching here.
            # 2. The table IS on screen but no cursor is set (first press, or
            #    the cursor's row was collapsed away). Place it on a row the
            #    reader can already SEE rather than scrolling to row 0.
            target = self._visible_row_index() or 0
        else:
            target = max(0, min(len(rows) - 1, index + delta))
        # Both caret positions, taken BEFORE the id moves: a cursor that drifted
        # off screen still has a painted line inside the body's model, and a
        # wheel or a scroll back would show a stale caret on it if the patch
        # only covered the rows the viewport can see.
        previous = self._cursor_index()
        self._cursor = rows[target].session_id
        self._patch_rows([previous, target])
        self._scroll_cursor_into_view()

    def _scroll_cursor_into_view(self) -> None:
        """Put the cursor's body line inside the viewport, moving as little as possible.

        The cursor is ALLOWED to be off screen — the wheel and the scrollbar move
        the viewport without touching it — so a key that acts on the cursor has
        to reveal it, or it writes to a row the reader cannot see and the frame
        does not appear to change. Reveal-then-act, never an interlock that makes
        the first press a no-op.
        """
        index = self._cursor_index()
        if index is None:
            return
        scroll = getattr(self, "_scroll", None)
        if scroll is None or not scroll.is_mounted:
            return
        line = self._layout.session_first_line + index
        top = scroll.scroll_offset.y
        height = scroll.size.height
        if height <= 0:
            return
        # One row of slack at each edge so the cursor is never flush against the
        # frame, where a reader cannot tell whether another row follows. The
        # slack is dropped on a viewport too short to afford it (a 1-2 row body
        # on a very short terminal): asking for context that does not fit makes
        # the two branches fight and the cursor lands off screen, which is worse
        # than no slack at all.
        slack = 1 if height >= 3 else 0
        if line < top + slack:
            scroll.scroll_to(y=max(0, line - slack), animate=False)
        elif line > top + height - 1 - slack:
            scroll.scroll_to(y=max(0, line - height + 1 + slack), animate=False)

    def _scroll_by_line(self, delta: int) -> None:
        """Move the viewport one line, the way the container did before the cursor.

        ``animate=False`` deliberately: an animated single-line step makes an
        autorepeating arrow queue animations behind each other and the viewport
        lags the key by hundreds of milliseconds. Every other viewport move on
        this screen (``_scroll_cursor_into_view``) is unanimated for the same
        reason, so this keeps one feel across the surface.
        """
        scroll = getattr(self, "_scroll", None)
        if scroll is None or not scroll.is_mounted:
            return
        scroll.scroll_relative(y=delta, animate=False)

    def action_move_up(self) -> None:
        # The arrows serve the TABLE only once the table is on screen. Above it
        # the report is charts and totals with no rows to address, so the key
        # line-scrolls exactly as ``origin/main``'s ``Binding("up",
        # "scroll_up")`` did. Without this the cursor's re-home clamped the
        # viewport at the table's first line and the top of the report could not
        # be reached by arrow at all (R1/D1/U1), and ``up`` at the bottom jumped
        # 570 lines backward (U2).
        if not self._table_on_screen():
            self._scroll_by_line(-1)
            return
        # At the TOP of the table the arrow keeps travelling instead of
        # clamping. Clamping here is what produced the hard floor: ``end`` then
        # 400 ``up`` presses left the viewport stuck at y=56 forever, because
        # every press re-pinned the cursor to row 0 and dragged the viewport
        # back to it. Falling through to a line-scroll hands the reader back the
        # 57 lines of totals and charts above the table.
        if self._cursor_on_screen() and self._cursor_index() == 0:
            self._scroll_by_line(-1)
            return
        self._move_cursor(-1)

    def action_move_down(self) -> None:
        if not self._table_on_screen():
            self._scroll_by_line(1)
            return
        rows = self._layout.session_rows
        # Symmetric to ``action_move_up``: at the last row the arrow scrolls on
        # rather than dying, so the reader can reach the true bottom of the body
        # with the same key that got them there.
        if rows and self._cursor_on_screen() and self._cursor_index() == len(rows) - 1:
            self._scroll_by_line(1)
            return
        self._move_cursor(1)

    def _ancestor_chain(self) -> list[str]:
        """Session ids of the cursor row's ancestors, nearest parent first.

        Read off the CURRENT paint, before the collapse that is about to hide
        the cursor. The walk is depth-first, so the nearest row above the cursor
        at each shallower depth is exactly its ancestor at that depth — the same
        exact (not heuristic) scan ``action_collapse_row`` already uses to step
        out to a parent.
        """
        index = self._cursor_index()
        if index is None:
            return []
        rows = self._layout.session_rows
        chain: list[str] = []
        depth = rows[index].depth
        for candidate in range(index - 1, -1, -1):
            if rows[candidate].depth < depth:
                depth = rows[candidate].depth
                chain.append(rows[candidate].session_id)
                if depth == 0:
                    break
        return chain

    def _rehome_to_ancestor(self, chain: list[str]) -> None:
        """After a collapse hid the cursor, put it on the nearest VISIBLE ancestor.

        The root that swallowed the row is where the reader's attention already
        is — it is the row still on screen at the position they were reading.
        Leaving the cursor naming a hidden session instead (the pre-fix
        behaviour) made the NEXT arrow press re-home to table row 0 and teleport
        the viewport: observed at ~200 lines after collapse-all (review U2c).
        """
        if self._cursor_index() is not None:
            return
        visible = {row.session_id for row in self._layout.session_rows}
        for ancestor in chain:
            if ancestor in visible:
                self._cursor = ancestor
                return

    def _set_expanded(self, session_id: str, open_: bool) -> None:
        # Captured BEFORE the repaint: collapsing a subtree removes the cursor's
        # row from the layout, and the ancestry that says where to land is only
        # readable while that row is still painted.
        chain = [] if open_ else self._ancestor_chain()
        if open_:
            self._expanded.add(session_id)
        else:
            self._expanded.discard(session_id)
        self._repaint()
        if chain:
            self._rehome_to_ancestor(chain)
            self._repaint()
        self._scroll_cursor_into_view()
        # An expand/collapse is the one mover the ``scroll_y`` watch does not
        # see: it changes the body's height, and only moves the viewport when
        # the cursor reveal above happens to scroll. Either way rows slide
        # under a resting pointer — 30 children appearing below the cursor
        # pushes every row beneath it down by 30 — so the highlight is
        # re-resolved here too. Deferred rather than run in place: the new
        # geometry is only settled after the refresh that paints it, and
        # ``_viewport_moved`` already coalesces this with any watch callback
        # the reveal fired, so one repaint carries both.
        self.call_after_refresh(self._viewport_moved)
        # Expanding changes the body's height, so whether it overflows its
        # viewport — and therefore whether the scroll hint is honest — can change
        # with it. Deferred, because the new height is only known after the
        # refresh that paints it.
        self.call_after_refresh(self._sync_hint)

    def _cursor_row(self) -> SessionRow | None:
        index = self._cursor_index()
        return None if index is None else self._layout.session_rows[index]

    def action_toggle_row(self) -> None:
        """Enter/space: open a closed row, close an open one.

        With no cursor yet, the first press places it rather than toggling. The
        user's report says "hits enter to expand that row", and there is no
        "that row" until a cursor exists — toggling whatever happens to be first
        would act on a row they never pointed at.
        """
        row = self._cursor_row()
        if row is None:
            self._move_cursor(0)
            return
        if not row.expandable:
            return
        self._set_expanded(row.session_id, not row.expanded)

    def action_expand_row(self) -> None:
        row = self._cursor_row()
        if row is None:
            self._move_cursor(0)
            return
        if row.expandable and not row.expanded:
            self._set_expanded(row.session_id, True)

    def action_collapse_row(self) -> None:
        """Left: close this row, or step out to the parent of an already-closed one.

        The step-out is what makes ``left`` usable inside a deep subtree — the
        alternative is a key that does nothing on every row that is not itself an
        open parent, which reads as broken rather than as restrained.
        """
        row = self._cursor_row()
        if row is None:
            self._move_cursor(0)
            return
        if row.expanded:
            self._set_expanded(row.session_id, False)
            return
        if row.depth:
            index = self._cursor_index()
            rows = self._layout.session_rows
            # The parent is the nearest row ABOVE this one at a shallower depth;
            # the walk is depth-first, so scanning back is exact rather than a
            # heuristic.
            for candidate in range((index or 0) - 1, -1, -1):
                if rows[candidate].depth < row.depth:
                    self._cursor = rows[candidate].session_id
                    self._repaint()
                    self._scroll_cursor_into_view()
                    return

    def action_toggle_all(self) -> None:
        """``e``: open every expandable row; press it again to close them all.

        The test is "is EVERYTHING already open", not "is anything open". The
        latter reads better as a safety rule — it makes the key an unconditional
        way out of the expensive fully-expanded state — but it is wrong at the
        keyboard: a reader who has opened one row and presses ``e`` meaning
        "now show me all of it" gets everything HIDDEN instead, which is the
        opposite of the request. Observed on the operator's ledger while
        exercising the real screen.

        Surprise is worse than slowness when the slowness is what was asked for.
        Expanding all 2,240 rows measured 1.4 s and is one press to undo, so the
        expensive state is reachable deliberately and never sticky.
        """
        expandable = {node.session_id for node in _iter_nodes(self._forest()) if node.has_children}
        collapsing = bool(expandable) and expandable <= self._expanded
        # Read BEFORE the repaint, for two independent reasons.
        #
        # ``chain``: collapse-all hides every child row, so the cursor's
        # ancestry has to be read while it is still painted (same as
        # ``_set_expanded``).
        #
        # ``anchored``: whether to reveal the cursor afterwards at all. ``e`` is
        # a GLOBAL action — it acts on every row, not on the row under the
        # caret — so it must not move the viewport of a reader who is not
        # standing in the table. Reviews D6/U8: the cursor is placed at MOUNT
        # (D2), and the reveal below then dragged the opening frame down to
        # table row 0, silently replacing the totals block the screen exists to
        # show (+34 lines at 120x40, +43 at 120x30, +51 at 110x20, and pressing
        # ``e`` again did not bring it back). On base the same call was a no-op
        # only because no cursor existed yet, so the mount cursor turned a
        # dormant call into a teleport.
        #
        # Revealing is still right when the cursor IS on screen: the reader is
        # working in the table, expand-all moves their row hundreds of lines
        # down the body, and keeping it in frame is what preserves their place.
        # Same rule ``_move_cursor`` already uses — a cursor the reader cannot
        # see is not a place to return to.
        anchored = self._cursor_on_screen()
        chain = self._ancestor_chain() if collapsing else []
        if collapsing:
            self._expanded.clear()
        else:
            self._expanded = expandable
        self._repaint()
        if chain:
            self._rehome_to_ancestor(chain)
            self._repaint()
        if anchored:
            self._scroll_cursor_into_view()
        # Same re-resolve as ``_set_expanded``, for the case it exists to
        # close: ``e`` moves rows under a resting pointer even when the
        # viewport does not. Children appear below every expandable root (or
        # vanish on collapse), sliding the table under the pointer — but the
        # ``scroll_y`` watch only fires when the cursor reveal above happens
        # to scroll, and with the cursor row already visible the reveal is a
        # no-op. No notification, no re-resolve, and the highlight stays
        # painted on a row the pointer is not over: the wheel-gotcha defect
        # via a keypress (review R1). ``_viewport_moved`` coalesces this with
        # any watch callback the reveal did fire.
        self.call_after_refresh(self._viewport_moved)
        self.call_after_refresh(self._sync_hint)

    def _forest(self) -> list["SessionNode"]:
        """The full session forest, built once and cached.

        ``build_report`` builds its own each paint (31 ms on the operator's
        ledger), but ``expand all`` needs the ids of rows that are NOT currently
        painted, so it cannot read them off the layout. Cached because the
        aggregate is immutable for the life of the screen — it is a snapshot read
        on a worker thread before the screen was pushed.
        """
        if self._forest_cache is None:
            self._forest_cache = build_session_forest(
                self._aggregate.by_session,
                getattr(self._aggregate, "session_parents", {}) or {},
            )
        return self._forest_cache

    def action_page_up(self) -> None:
        self._scroll.scroll_page_up()

    def action_page_down(self) -> None:
        self._scroll.scroll_page_down()

    def action_scroll_home(self) -> None:
        self._scroll.scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll.scroll_end()
