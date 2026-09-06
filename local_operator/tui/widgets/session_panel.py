"""Read-only current-session diagnostics, drawn with the analytics chart vocabulary.

The report is a close/reopen snapshot, not a live bill. Runtime scalars are
captured before the worker reads SQLite; neither prompts nor tool payloads ever
enter this view.

Every proportional row on this screen is drawn by :func:`_render_rows` on top of
``analytics_panel``'s primitives (``proportion_bar``, ``section_header``,
``format_tokens``/``format_cost``/``format_percent``, ``append_cost``,
``METRIC_COST``/``METRIC_TOKENS``), imported rather than copied: two diagnostics
screens with two bar styles, two money formatters or two honesty vocabularies
would be a defect, not a variation.

**The denominators differ deliberately, and each is commented where it is
computed.** A bar means nothing without knowing what it is a fraction OF, and
this screen answers four different questions:

- *by model* / *by purpose* — share of the session total, because those rows
  PARTITION one quantity and the reader's question is "what fraction of this
  session was X".
- *context window* — share of a hard limit, an absolute gauge.
- *where input went* / *tool surface* — share of the component total, so the
  three tool components read with identical percentages in both sections.
- *last N requests* — share of the window max, because a tail does NOT partition
  the session and share-of-total would render twelve indistinguishable 4% bars.

What is deliberately NOT here: per-tool-name tokens or dollars (the ledger has no
such column and the provider bills one input total per call — see
:data:`_TOOL_KEYS`), and any calendar chart (the daily/monthly rollups are
machine-wide, not session-scoped, so drawing them under a "this session" heading
would attribute other sessions' spend to this one).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.analytics.model import (
    COMPONENT_KEYS,
    COMPONENT_LABELS,
    SessionReport,
    TimingSummary,
    UsageAggregate,
)
from local_operator.session.protocol import SessionProtocol
from local_operator.tui.widgets.analytics_panel import (
    COST_LEGEND,
    METRIC_COST,
    METRIC_TOKENS,
    append_cost,
    format_cost,
    format_percent,
    format_tokens,
    proportion_bar,
    scope_needs_cost_legend,
    section_header,
    semantic_style,
)

#: The ledger's tool-shaped component columns, biggest-first for the child rows
#: of the Tool surface section. These are the ONLY tool-shaped numbers recorded:
#: there is no per-tool-name token or cost column anywhere in ``calls``, and
#: there cannot be one without both a schema migration and a recording-side
#: change, because the provider bills one input total per call and never
#: attributes it to a tool. Anything labelled "cost per tool" here would be an
#: invented number, which is why the section carries a caveat instead.
_TOOL_KEYS = ("tool_results", "tool_schemas", "tool_inventory")

#: Responsive ladder, in the order things shed as the card narrows. Modelled on
#: ``analytics_panel._WIDE_TABLE_MIN`` — the value column never sheds, because a
#: bar without its number is a picture and a number without its bar is the
#: prose screen this replaces.
#:
#: - below :data:`_NOTE_MIN` the trailing note column goes (``N req``, the
#:   purpose tag, the timing range, the Totals qualifiers). Measured need: at 50
#:   columns a note pushes the row past the viewport and folds onto its own
#:   line, where it reads as a separate record.
#: - below :data:`_PCT_MIN` the percentage goes too, leaving label + bar + value.
_NOTE_MIN = 60
_PCT_MIN = 52

#: Card width the column arithmetic degrades to before mount, when the app size
#: is not yet knowable. Matches ``AnalyticsScreen._card_width``'s fallback.
_DEFAULT_CARD_WIDTH = 88

#: The narrowest frame the ladder in :data:`_NOTE_MIN` / :data:`_PCT_MIN` is
#: defined down to: a 50-column terminal's measured content box.
_MIN_CARD_WIDTH = 38

#: Bar floor. Below eight cells a bar cannot express a proportion, so the label
#: column yields (and its labels ellipsise) before the bar is allowed to shrink.
_BAR_MIN = 8
_BAR_MAX = 24

#: Value cell for the Totals ``kv`` rows, matching ``build_report``'s so the two
#: screens' headline blocks have the same rhythm.
_VALUE_CELL = 11


@dataclass(frozen=True)
class SessionDiagnostics:
    """Only scalar public state; never hold a mutable session across a disk read."""

    session_id: str
    name: str
    selected_model: str
    effective_model: str
    streaming: bool
    context_tokens: int | None = None
    context_window: int | None = None
    context_is_estimate: bool | None = None
    generation: int | None = None
    epoch: str | None = None

    @classmethod
    def capture(cls, session: SessionProtocol) -> SessionDiagnostics:
        # The canonical frontend snapshot is already mirrored on RemoteSession.
        # Reduced SDK facades need not have it; don't traverse private context or
        # tokenize history just to fill a missing diagnostic.
        state = getattr(session, "frontend_state", None)
        model = session.effective_model
        return cls(
            session_id=session.session_id,
            name=session.conversation_name,
            selected_model=session.model_label,
            effective_model=session.effective_model_label,
            streaming=session.is_streaming,
            context_tokens=getattr(state, "context_tokens", None),
            context_window=getattr(state, "context_window", None)
            or getattr(model, "context_window", None),
            context_is_estimate=getattr(state, "context_is_estimate", None),
            generation=getattr(state, "generation", None),
            epoch=getattr(state, "epoch", None),
        )


def _stamp(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "unknown"
    return datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _clock(ts_ms: int | None) -> str:
    """Wall-clock label for one request row: ``HH:MM:SS``.

    Only the time of day, not the date: the sequence chart's rows are a tail of
    one session, so the date is the same on every row and repeating it would
    spend the whole label column saying nothing.
    """
    if ts_ms is None:
        return "unknown"
    return datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%H:%M:%S")


def _milliseconds(value: float | None) -> str:
    return "unknown" if value is None else f"{value:,.0f} ms"


# -- the shared row engine ---------------------------------------------------


@dataclass(frozen=True)
class _BarRow:
    """One proportional row, pre-measured so a whole table shares its columns.

    ``note`` is a sequence of ``(text, semantic token)`` pairs rather than a
    plain string because the parts are not all the same kind of thing: a request
    count is ``dim`` status chrome, while ``N failed`` is a fact a reader must
    not scroll past and is drawn in ``warning``.

    ``bar=False`` suppresses the bar for a row that is not a peer of the others
    — the ``Unattributed`` residual, or an unpriced row in cost mode — so the
    reader is never shown a fill implying a share the row does not have.
    """

    label: str
    fraction: float
    value: str
    note: tuple[tuple[str, str], ...] = ()
    bar: bool = True
    #: The scope whose money this row's value renders, when the active metric is
    #: cost. Carried (rather than pre-formatted into ``value``) so the row is
    #: drawn by ``append_cost`` and inherits its dimmed lower-bound ``+``;
    #: ``None`` on a token row, which has no such flag.
    cost: UsageAggregate | None = None


@dataclass(frozen=True)
class _Columns:
    """The measured column widths one or more bar tables are drawn into."""

    label: int
    bar: int
    value: int
    show_pct: bool
    show_note: bool


def _measure_columns(rows: list[_BarRow], width: int, *, show_pct: bool = True) -> _Columns:
    """Measure label/bar/value columns across EVERY row before any is emitted.

    Callers pass the UNION of rows from every section that must line up, so one
    left edge, one bar column and one number column run down the whole screen.
    ``/analytics`` measures per table, which is why its component bars and its
    per-provider numbers start at two different x positions on one panel; that
    is the flaw this signature exists to make impossible here.

    The label cap is bounded by the FRAME, not only by a constant: at 40 cells
    of card width a 34-character label such as ``Custom instructions
    (agents/teams)`` cannot coexist with a bar, so the label column collapses
    (and :func:`_render_rows` ellipsises) rather than starving the bar, which
    below :data:`_BAR_MIN` cells stops expressing a proportion at all.

    The note column is measured too, and shed WHOLE rather than cropped: a note
    that runs off the right edge renders as ``14 req · 2 f``, which reads as a
    truncated number and is worse than no note at all. So the bar gives back
    width down to its floor to keep the notes, and if even that is not enough
    the whole note column drops — the ladder's rule that a qualifier sheds and
    the fact stays.
    """
    show_pct = show_pct and width >= _PCT_MIN
    pct_col = 5 if show_pct else 0
    value_col = max((len(r.value) for r in rows), default=0)
    natural = max((len(r.label) for r in rows), default=0) + 2
    note_col = max((sum(len(text) for text, _ in r.note) for r in rows), default=0)
    show_note = width >= _NOTE_MIN and note_col > 0
    # 5 = the 2-cell left indent, the value gutter, and the note gutter.
    fixed = value_col + pct_col + 5 + (note_col if show_note else 0)
    label_col = min(36, natural, max(_BAR_MIN, width - _BAR_MIN - fixed))
    if show_note and width - label_col - fixed < _BAR_MIN:
        # The bar is at its floor and the row still does not fit, so the note —
        # the least load-bearing column — goes and the label reclaims the space.
        show_note = False
        fixed -= note_col
        label_col = min(36, natural, max(_BAR_MIN, width - _BAR_MIN - fixed))
    bar_width = max(_BAR_MIN, min(_BAR_MAX, width - label_col - fixed))
    return _Columns(
        label=max(1, label_col),
        bar=bar_width,
        value=value_col,
        show_pct=show_pct,
        show_note=show_note,
    )


def _render_rows(
    rows: list[_BarRow], cols: _Columns, width: int, *, estimated: bool = False
) -> list[Text]:
    """Draw pre-measured rows: label, bar, value, percentage, note.

    ``estimated`` prefixes every percentage with ``~``, the mark ``/analytics``
    already uses for a modelled (character-apportioned) split rather than a
    measured one. Full-strength ``fg`` carries the digits; ``dim`` carries the
    status flags, because a flag is not a number.

    Rows are cropped to ``width`` at build time rather than declared
    ``no_wrap``: the screen flattens these lines into one container ``Text``
    with ``Text.append_text``, which copies characters and spans but NOT the
    per-line wrap policy, so a row long enough to fold would fold anyway and
    read as two records. The container keeps ``fold`` so the prose footnotes
    below still wrap, which is right for prose and wrong for a table.
    """
    fg = semantic_style("fg")
    dim = semantic_style("dim")
    accent = semantic_style("accent")
    out: list[Text] = []
    for row in rows:
        label = row.label
        if len(label) > cols.label - 1:
            label = label[: max(0, cols.label - 2)] + "…"
        line = Text()
        line.append(f"  {label:<{cols.label}}", style=fg)
        # A row with ``bar=False`` still consumes the bar column so its value
        # stays in the shared number column; it simply draws no fill.
        line.append(proportion_bar(row.fraction, cols.bar) if row.bar else " " * cols.bar, accent)
        line.append(" ")
        if row.cost is not None:
            # A money cell goes through ``append_cost`` so the lower-bound ``+``
            # is dimmed as the status flag it is. Rendering it at the same
            # strength as the digits makes a lower bound read as a precise
            # total — the exact defect that helper was written to fix, which a
            # second hand-rolled cost cell here would silently reintroduce.
            append_cost(line, row.cost, cols.value, fg, dim)
        else:
            line.append(f"{row.value:>{cols.value}}", style=fg)
        if cols.show_pct:
            # A bar-less row has no share to state, so it prints neither a
            # percentage nor the ``~`` estimate mark — marking a blank as
            # "modelled" would attach an honesty flag to nothing.
            mark = ("~" if estimated else " ") if row.bar else " "
            pct = format_percent(row.fraction) if row.bar else ""
            line.append(f" {mark}{pct:>3}", style=dim)
        if cols.show_note and row.note:
            line.append("  ")
            for text, token in row.note:
                line.append(text, style=semantic_style(token))
        line.truncate(width, overflow="crop")
        out.append(line)
    return out


def _metric_value(aggregate: UsageAggregate, metric: str) -> float:
    """The number a bar is drawn proportional to, for the active metric.

    Cost in micro-USD (an exact integer accumulator, floated only for the
    fraction), tokens as the full billed total — the same pair
    ``analytics_panel._metric_value`` reads off a calendar bucket.
    """
    if metric == METRIC_TOKENS:
        return float(aggregate.total_tokens)
    return float(aggregate.cost_micro)


def _metric_cell(aggregate: UsageAggregate, metric: str) -> str:
    """The right-hand numeric label for the active metric."""
    if metric == METRIC_TOKENS:
        return format_tokens(aggregate.total_tokens)
    return format_cost(aggregate)


def _metric_meta(metric: str, prefix: str) -> str:
    """``share of session tokens · t → cost`` — the section's own key.

    The toggle is disclosed in TWO places (here and in the hint line), the same
    redundancy ``/analytics`` uses, so a reader parked halfway down a scrolling
    report can still discover what ``t`` does and what the bars currently plot.
    """
    active = "tokens" if metric == METRIC_TOKENS else "cost"
    other = "cost" if metric == METRIC_TOKENS else "tokens"
    # Without a prefix this is the narrow form: the toggle disclosure alone.
    return f"{prefix} {active} · t → {other}" if prefix else f"t → {other}"


def _group_rows(
    groups: list[tuple[str, UsageAggregate]],
    metric: str,
    *,
    failures: dict[str, int] | None = None,
) -> tuple[list[_BarRow], bool]:
    """Rows for a partitioning table (by model, by purpose), biggest first.

    Returns the rows and whether the active metric has anything to be a fraction
    of. **Denominator: share of the session total for the active metric.** These
    groups partition one quantity, so the percentages sum to 100% and the row
    answers "what fraction of this session was the expensive model / was not a
    turn". A window-max denominator (the rule ``_series_chart`` uses, correct
    there because calendar buckets do NOT partition a total) would make the
    largest group always full-width and destroy exactly that reading.

    Sorted by the active metric with ``total_tokens`` as the tiebreak — the same
    ``(cost_micro, total_tokens)`` key ``_group_section`` sorts by, so an
    unpriced group falls back to token order rather than collapsing into one
    indistinguishable $0 bucket.
    """
    total = sum(_metric_value(agg, metric) for _, agg in groups)
    ordered = sorted(
        groups,
        key=lambda item: (_metric_value(item[1], metric), item[1].total_tokens),
        reverse=True,
    )
    rows: list[_BarRow] = []
    for name, agg in ordered:
        note: list[tuple[str, str]] = [(f"{agg.calls} req", "dim")]
        failed = (failures or {}).get(name, 0)
        if failed:
            # ``warning``, not ``dim``: a failed request is the one thing on this
            # screen a reader must not scroll past, and at the far right of the
            # widest column a dim tally reads as decoration.
            note.append((" · ", "dim"))
            note.append((f"{failed} failed", "warning"))
        rows.append(
            _BarRow(
                label=name,
                fraction=(_metric_value(agg, metric) / total) if total > 0 else 0.0,
                value=_metric_cell(agg, metric),
                note=tuple(note),
                bar=total > 0,
                cost=agg if metric == METRIC_COST else None,
            )
        )
    return rows, total > 0


def _failures(report: SessionReport) -> dict[str, int]:
    """Non-``ok`` request count per purpose, from the existing outcome cross-tab.

    ``by_purpose`` carries the consumption and ``by_purpose_outcome`` carries
    the outcomes; the failure tally is the second folded onto the first, not a
    third query.
    """
    # ``unknown`` is excluded, not folded into the failures: an older ledger has
    # no ``outcome`` column, so every row reads ``unknown`` and counting those
    # would paint a healthy legacy session as entirely failed — in ``warning``,
    # the one colour on this screen that must never cry wolf.
    return {
        purpose: sum(
            count
            for (name, outcome), count in report.by_purpose_outcome.items()
            if name == purpose and outcome not in ("ok", "unknown")
        )
        for purpose in report.by_purpose
    }


def _component_rows(aggregate: UsageAggregate) -> tuple[list[_BarRow], int]:
    """Nonzero input components, biggest first, plus the component total.

    **Denominator: share of the component total, not of ``context_tokens``.**
    The split is apportioned by largest-remainder to sum exactly to the recorded
    context, so share-of-components is the only denominator under which these
    percentages sum to 100%.

    Zero-value components are dropped: a fresh session has no tool results, and
    ``Environment ~0`` / ``Images (est.) ~0`` are rows that say nothing.
    """
    total = sum(aggregate.components.get(key, 0) for key in COMPONENT_KEYS)
    rows = [
        _BarRow(
            label=COMPONENT_LABELS.get(key, key),
            fraction=aggregate.components.get(key, 0) / total,
            value=format_tokens(aggregate.components.get(key, 0)),
        )
        for key in COMPONENT_KEYS
        if total > 0 and aggregate.components.get(key, 0) > 0
    ]
    rows.sort(key=lambda row: -row.fraction)
    return rows, total


def _tool_rows(aggregate: UsageAggregate, component_total: int) -> list[_BarRow]:
    """The tool machinery's share of input: a subtotal row plus its children.

    **Denominator: the component total — the SAME denominator as "Where input
    went".** A reader must be able to check the three child rows against the
    other section and find identical percentages; scaling these to the tool
    subtotal instead would print ``Tool results`` as 60% here and 23% there, on
    one screen, for the same tokens.

    The parent is a subtotal, not a peer, so it holds a fixed first position and
    is drawn in the same denominator, making "parent ≈ sum of children" visually
    checkable.
    """
    values = {key: aggregate.components.get(key, 0) for key in _TOOL_KEYS}
    subtotal = sum(values.values())
    if component_total <= 0 or subtotal <= 0:
        return []
    rows = [
        _BarRow(
            label="All tool context",
            fraction=subtotal / component_total,
            value=format_tokens(subtotal),
        )
    ]
    for key, value in sorted(values.items(), key=lambda item: -item[1]):
        if value > 0:
            rows.append(
                _BarRow(
                    label=" └ " + COMPONENT_LABELS.get(key, key),
                    fraction=value / component_total,
                    value=format_tokens(value),
                )
            )
    return rows


def _timing_rows(timings: dict[str, TimingSummary], width: int) -> list[Text]:
    """Three range bars on ONE shared millisecond scale.

    **Denominator: the maximum across all three rows.** Three independently
    scaled bars would draw preparation and total duration the same length, which
    is the opposite of the truth; a shared scale makes ``First output`` visibly
    a fraction of ``Duration``. The legend states that the scale is shared,
    because a shared axis the reader cannot see is a trap.

    Glyphs: ``·`` outside the observed range, ``─`` inside min–max, ``█`` at the
    mean, which wins the cell on a collision. This is a purpose-built track
    rather than ``proportion_bar`` because a range is not a proportion —
    reaching for the proportion primitive here would draw a fill from zero and
    claim a share that does not exist.

    Fixed order (Duration, First output, Prep): these are nested concepts, not
    competitors, and sorting them by size would scramble a stable mental model.
    """
    fg = semantic_style("fg")
    dim = semantic_style("dim")
    accent = semantic_style("accent")
    keys = (("duration_ms", "Duration"), ("ttft_ms", "First output"), ("preparation_ms", "Prep"))
    present = [timings.get(key) for key, _ in keys]
    scale = max(
        (t.max_ms for t in present if t is not None and t.samples and t.max_ms is not None),
        default=0.0,
    )
    label_col = max(len(label) for _, label in keys) + 2
    value_col = max(
        (len(_milliseconds(t.mean_ms)) for t in present if t is not None and t.samples), default=8
    )
    bar_width = max(10, min(_BAR_MAX, width - label_col - value_col - 6))
    lines: list[Text] = []
    for (key, label), timing in zip(keys, present):
        row = Text()
        row.append(f"  {label:<{label_col}}", style=fg)
        if timing is None or not timing.samples or scale <= 0:
            # No bar at all: a zero-width bar beside a real one implies a
            # measured zero, and an absent sample is not a fast one.
            row.append("unknown (0 samples)", style=dim)
            row.truncate(width, overflow="crop")
            lines.append(row)
            continue
        low = int(round((timing.min_ms or 0) / scale * bar_width))
        high = int(round((timing.max_ms or 0) / scale * bar_width))
        mean = min(int(round((timing.mean_ms or 0) / scale * bar_width)), bar_width - 1)
        track = "".join(
            "█" if i == mean else "─" if low <= i < max(high, low + 1) else "·"
            for i in range(bar_width)
        )
        row.append(track, style=accent)
        row.append(f" {_milliseconds(timing.mean_ms):>{value_col}}", style=fg)
        if width >= _NOTE_MIN:
            row.append(
                f"  {_milliseconds(timing.min_ms)}–{_milliseconds(timing.max_ms)}", style=dim
            )
        row.truncate(width, overflow="crop")
        lines.append(row)
    return lines


@dataclass
class _Body:
    """Accumulates the report's lines, keeping the append shapes in one place."""

    width: int
    lines: list[Text] = field(default_factory=list)

    def blank(self) -> None:
        self.lines.append(Text())

    def kv(self, name: str, value: str, note: str = "") -> None:
        """A Totals-style scalar row, matching ``build_report``'s ``kv`` exactly."""
        row = Text()
        row.append(f"  {name:<22}", style=semantic_style("dim"))
        row.append(f"{value:<{_VALUE_CELL}}", style=semantic_style("fg"))
        # Below _NOTE_MIN the qualifier is shed and the fact kept: at 50 columns
        # the note wraps onto its own unindented line and reads as a new record.
        if note and self.width >= _NOTE_MIN:
            row.append(f"  {note}", style=semantic_style("dim"))
        row.truncate(self.width, overflow="crop")
        self.lines.append(row)

    def note(self, text: str) -> None:
        """A dim footnote. Wrapped, not cropped — it is prose, not a table row."""
        row = Text(overflow="fold")
        row.append(f"  {text}", style=semantic_style("dim"))
        self.lines.append(row)

    def header(self, title: str, meta: str = "", short: str = "") -> None:
        """A section header, shedding its meta to ``short`` on a narrow frame.

        A header is one ``Text`` and its meta is not a column, so cropping it
        would silently lose the toggle disclosure — and left alone it wraps to
        an unindented second line that reads as a new record, the very fault
        the table rows shed their notes to avoid. So the qualifier goes and the
        operative part stays: ``share of session tokens · t → cost`` becomes
        ``t → cost``, which is what a reader needs in order to act.
        """
        if meta and self.width < _NOTE_MIN:
            meta = short
        self.lines.append(section_header(title, meta))

    def extend(self, lines: list[Text]) -> None:
        self.lines.extend(lines)

    def to_text(self) -> Text:
        # ``fold`` on the container so the prose footnotes wrap; every table row
        # was already cropped to width at build time, because ``append_text``
        # discards a line's own wrap policy (see ``_render_rows``).
        out = Text(style=semantic_style("fg"), overflow="fold")
        for index, line in enumerate(self.lines):
            if index:
                out.append("\n")
            out.append_text(line)
        return out


def build_session_report(
    report: SessionReport | None,
    runtime: SessionDiagnostics,
    width: int = _DEFAULT_CARD_WIDTH,
    *,
    metric: str = METRIC_TOKENS,
) -> Text:
    """Render one session's diagnostics as charts, at ``width`` content cells.

    ``metric`` (the screen's ``t`` toggle) selects what the by-model, by-purpose
    and request-sequence bars plot. Tokens is the default here — unlike
    ``/analytics``, whose stated purpose is historical spend, this screen is
    read mid-session to answer "where did my context go", which is a token
    question.
    """
    width = max(_MIN_CARD_WIDTH, width)
    body = _Body(width)
    fg = semantic_style("fg")
    dim = semantic_style("dim")

    # -- header: name, then the ID as a dim lookup key, not a headline --------
    head = Text()
    head.append(runtime.name or "Untitled session", style=fg)
    head.truncate(width, overflow="crop")
    body.lines.append(head)
    ident = Text()
    ident.append(f"  {runtime.session_id}", style=dim)
    ident.truncate(width, overflow="crop")
    body.lines.append(ident)

    if report is None:
        body.blank()
        body.header("Loading usage records")
        body.note("Reading the local ledger. Esc or q cancels.")
        body.note("No model request is made.")
        return body.to_text()

    aggregate = report.aggregate
    gauge = _gauge_row(runtime)
    body.blank()
    if not report.available:
        # No chart sections at all here, including the live gauge: when the read
        # failed we cannot say which of these numbers are trustworthy, and one
        # lone bar under a failure heading invites the reader to trust the rest.
        body.header("Ledger unavailable")
        body.note("Could not read local usage records. Close and reopen to try again.")
        body.blank()
    elif not aggregate.calls:
        body.header("No recorded requests")
        body.note("No retained usage records for this session yet.")
        body.blank()
        # The gauge is LIVE runtime data, so it is available even with an empty
        # ledger — a fresh session's one true visual.
        _draw_context_gauge(body, gauge, _measure_columns([gauge] if gauge else [], width), width)
    else:
        _draw_recorded_usage(body, report, gauge, width, metric)

    _draw_runtime_and_scope(body, report, runtime)
    return body.to_text()


def _gauge_row(runtime: SessionDiagnostics) -> _BarRow | None:
    """The live context gauge's single row, or ``None`` when unmeasured.

    **Denominator: the model's context window — a hard limit, not a share of
    anything else on screen.** That is why the gauge is its own section rather
    than a Totals row: "how full am I" is a different question from "where did
    my tokens go", and an absolute gauge inside a partitioning table would draw
    a bar incomparable with its neighbours'.

    ``None`` when the window is unmeasured, which drops the section entirely: a
    0% bar would claim an empty context, and an unmeasured window is not an
    empty one.
    """
    if runtime.context_tokens is None or not runtime.context_window:
        return None
    estimate_mark = "~" if runtime.context_is_estimate else ""
    return _BarRow(
        label="In context now",
        fraction=min(1.0, runtime.context_tokens / runtime.context_window),
        value=f"{estimate_mark}{format_tokens(runtime.context_tokens)}",
        note=((f"of {format_tokens(runtime.context_window)}", "dim"),),
    )


def _draw_context_gauge(body: _Body, row: _BarRow | None, cols: _Columns, width: int) -> None:
    """Draw the gauge, sourced from the RUNTIME snapshot rather than the ledger.

    The meta says so out loud: every other number on this screen is retained
    ledger data, and silently mixing a live scalar into it would be the
    dishonest move.
    """
    if row is None:
        return
    body.header("Context window", "live · not from the ledger", "live")
    body.extend(_render_rows([row], cols, width))
    body.blank()


def _shared_columns(
    report: SessionReport,
    aggregate: UsageAggregate,
    gauge: _BarRow | None,
    width: int,
    metric: str,
) -> _Columns:
    """ONE column set measured across every proportional section on the screen.

    The gauge, by model, by purpose, where input went and tool surface are
    measured TOGETHER. Sizing each table on its own rows is what gives
    ``/analytics`` two different bar columns down one panel — its component bars
    start at x≈46 while its per-provider numbers start at x≈35 — and measuring
    the union is what gives this screen one left edge, one bar column and one
    number column top to bottom.

    The request-sequence chart is deliberately excluded: its labels are a fixed
    8-cell clock, and forcing them into a column sized for model names would
    push twelve bars far right for no alignment gain.
    """
    rows: list[_BarRow] = [gauge] if gauge else []
    if report.by_model:
        rows.extend(
            _group_rows([(f"{p}/{m}", a) for (p, m), a in report.by_model.items()], metric)[0]
        )
    # Measured WITH the failure annotations, because they are what the draw
    # pass emits: measuring `14 req` and then drawing `14 req · 2 failed` sizes
    # the note column too small and crops it to `14 req · 2 f`.
    rows.extend(_group_rows(list(report.by_purpose.items()), metric, failures=_failures(report))[0])
    component_rows, component_total = _component_rows(aggregate)
    rows.extend(component_rows)
    rows.extend(_tool_rows(aggregate, component_total))
    return _measure_columns(rows, width)


def _draw_recorded_usage(
    body: _Body,
    report: SessionReport,
    gauge: _BarRow | None,
    width: int,
    metric: str,
) -> None:
    """Totals, then every proportional section, on one shared column set."""
    aggregate = report.aggregate
    cols = _shared_columns(report, aggregate, gauge, width, metric)

    # -- Totals: five unrelated scalars, so no bars ---------------------------
    # A bar implies a shared denominator, and these five do not have one.
    meta = f"{aggregate.calls} requests"
    if aggregate.ok_calls != aggregate.calls:
        meta += f" ({aggregate.calls - aggregate.ok_calls} failed)"
    body.header("Totals", meta + " · measured", meta)
    body.kv(
        "Total billed",
        format_tokens(aggregate.total_tokens) + " tokens",
        f"{format_tokens(aggregate.context_tokens)} in · "
        f"{format_tokens(aggregate.output_tokens)} out",
    )
    body.kv(
        "Context read",
        format_tokens(aggregate.context_tokens),
        f"{format_tokens(aggregate.fresh_tokens)} fresh · "
        f"{format_tokens(aggregate.cache_read_tokens)} cached",
    )
    body.kv(
        "Output",
        format_tokens(aggregate.output_tokens),
        f"{format_tokens(aggregate.generation_tokens)} generation · "
        f"{format_tokens(aggregate.reasoning_tokens)} thinking",
    )
    body.kv("Cache hit rate", format_percent(aggregate.cache_hit_rate), "of context from cache")
    body.kv(
        "Est. cost",
        format_cost(aggregate),
        "≈ list price × tokens" if aggregate.cost_is_known else "no published price",
    )
    # Suppressed when both are zero: on the healthy path "0 requests; 0 unknown"
    # is a row whose only content is the absence of a problem.
    if report.missing_usage_calls or report.unknown_usage_calls:
        body.note(
            f"{report.missing_usage_calls:,} requests missing usage · "
            f"{report.unknown_usage_calls:,} unknown"
        )
    body.blank()

    _draw_context_gauge(body, gauge, cols, width)
    _draw_by_model(body, report, width, metric, cols)
    _draw_by_purpose(body, report, width, metric, cols)
    _draw_input_split(body, aggregate, width, cols)
    _draw_tool_surface(body, aggregate, width, cols)
    _draw_request_sequence(body, report, width, metric)
    body.header("Timings", "observed wall time · includes retries", "wall time")
    body.extend(_timing_rows(report.timings, width))
    body.note("█ mean · ─ min–max range · shared ms scale")
    body.note(
        "Wall time includes retries and consumer backpressure, not provider "
        "compute or per-attempt timing. First output is the first text or "
        "tool-call delta."
    )
    body.blank()

    # The money footnote is drawn only when a mark is actually on screen, over
    # exactly the scopes this report renders — a footnote for a symbol nobody
    # can see is noise.
    scopes = [aggregate, *report.by_model.values(), *report.by_purpose.values()]
    if any(scope_needs_cost_legend(scope) for scope in scopes):
        body.note(COST_LEGEND)
        body.blank()


def _unpriced_note(body: _Body, metric: str, priced: bool) -> bool:
    """Say why the bars are missing in cost mode, instead of drawing floor marks.

    A fully unpriced scope in cost mode would render every row as a one-cell
    floor bar — a wall of identical marks that looks like data. The rows still
    show their ``$—``; the bar is what is withheld, plus a line naming the
    escape hatch.
    """
    if priced or metric != METRIC_COST:
        return False
    body.note("no published price for these models · t → tokens")
    return True


def _draw_by_model(
    body: _Body, report: SessionReport, width: int, metric: str, cols: _Columns
) -> None:
    if not report.by_model:
        return
    groups = [(f"{provider}/{model}", agg) for (provider, model), agg in report.by_model.items()]
    rows, priced = _group_rows(groups, metric)
    body.header("By model", _metric_meta(metric, "share of session"), _metric_meta(metric, ""))
    _unpriced_note(body, metric, priced)
    body.extend(_render_rows(rows, cols, width))
    body.blank()


def _draw_by_purpose(
    body: _Body, report: SessionReport, width: int, metric: str, cols: _Columns
) -> None:
    """Consumption per purpose — the "by use case" read, not a bare count.

    Labels are the RAW stored values (``turn``, ``compaction``, ``aside``,
    ``naming``, ``compaction_advisor``): they are what a reader greps the code
    for, and a friendly relabelling would create a second vocabulary for the
    same thing. A legend explains them instead, and only when there is something
    to explain.
    """
    if not report.by_purpose:
        return
    rows, priced = _group_rows(list(report.by_purpose.items()), metric, failures=_failures(report))
    body.header("By purpose", _metric_meta(metric, "share of session"), _metric_meta(metric, ""))
    # Only when there is a contrast to explain: the legend defines ``turn``
    # against the harness's own purposes, so it is noise when the rows are all
    # ``turn``, and meaningless on a legacy ledger whose single row is
    # ``unknown`` and where no ``turn`` row exists to be contrasted with.
    if "turn" in report.by_purpose and len(report.by_purpose) > 1:
        body.note("turn = your requests · others are the harness working on your behalf")
    _unpriced_note(body, metric, priced)
    body.extend(_render_rows(rows, cols, width))
    body.blank()


def _draw_input_split(body: _Body, aggregate: UsageAggregate, width: int, cols: _Columns) -> None:
    rows, total = _component_rows(aggregate)
    body.header("Where input went", "≈ estimated split of context tokens", "≈ estimated")
    if not rows:
        body.note("no component data yet")
        body.blank()
        return
    # The residual is what the recorded context exceeds the attributed
    # components by — a leftover, not a component. It gets no bar: giving it one
    # in the same denominator would double-count against the rows above it.
    unattributed = max(0, aggregate.context_tokens - total)
    if unattributed:
        rows = rows + [
            _BarRow(
                label="Unattributed (older records)",
                fraction=0.0,
                value=format_tokens(unattributed),
                bar=False,
            )
        ]
    body.extend(_render_rows(rows, cols, width, estimated=True))
    body.blank()


def _draw_tool_surface(body: _Body, aggregate: UsageAggregate, width: int, cols: _Columns) -> None:
    rows = _tool_rows(aggregate, _component_rows(aggregate)[1])
    body.header("Tool surface", "≈ estimated share of context tokens", "≈ estimated")
    if rows:
        body.extend(_render_rows(rows, cols, width, estimated=True))
    else:
        body.note("no tool context recorded yet")
    # NOT optional and NOT shortened. It is the only thing preventing this
    # section from being read as per-tool billing, which the ledger cannot
    # support: there is no per-tool-name token or cost column, and cost is
    # priced per call at record time, so splitting it across these three
    # components would invent a number the provider never billed.
    body.note(
        "Per-tool-name tokens and dollars are not recorded; this is the tool "
        "machinery's share of input, not the cost of any one tool call."
    )
    body.blank()


def _draw_request_sequence(body: _Body, report: SessionReport, width: int, metric: str) -> None:
    """The recent tail as one bar per request, oldest → newest.

    **Denominator: the window max (the largest request in view).** These rows do
    NOT partition the session — they are a tail of it — so a share-of-total
    denominator would draw twelve bars at 4% each and communicate nothing. The
    question here is "which requests were big", and window-max is the shape that
    answers it. This is the ``_series_chart`` rule, applied for the same reason.

    The x axis is the request ORDINAL, not time: rows are evenly spaced whatever
    the gap between them, so the meta says ``oldest → newest`` rather than "over
    time" and no gap is drawn or interpolated. A session with a four-hour idle
    stretch must not render a misleading flat run.

    In cost mode the bars plot ``duration_ms`` instead, and the meta says so:
    ``SessionRequest`` carries no cost field, and dividing the session total by
    the request count would fabricate a per-request price the ledger never held.
    """
    if not report.recent:
        return
    series = list(reversed(report.recent))  # the store returns newest-first
    tokens = metric == METRIC_TOKENS
    values = [
        float(r.context_tokens + r.output_tokens) if tokens else float(r.duration_ms or 0.0)
        for r in series
    ]
    top = max(values) if values else 0.0
    rows: list[_BarRow] = []
    for request, value in zip(series, values):
        note: tuple[tuple[str, str], ...] = ((request.purpose, "dim"),)
        if request.outcome != "ok":
            note = ((request.purpose, "dim"), (" · ", "dim"), (request.outcome, "warning"))
        rows.append(
            _BarRow(
                label=_clock(request.ts_ms),
                fraction=(value / top) if top > 0 else 0.0,
                value=format_tokens(int(value)) if tokens else _milliseconds(value),
                note=note,
            )
        )
    plural = "request" if len(series) == 1 else "requests"
    bars = "tokens" if tokens else "duration"
    body.header(
        f"Last {len(series)} {plural}",
        f"of {report.aggregate.calls} · oldest → newest · bars: {bars}",
        f"bars: {bars}",
    )
    # Its own column set: these labels are a fixed 8-cell clock, and forcing
    # them into the model-name column above would push every bar far right for
    # no alignment gain.
    body.extend(_render_rows(rows, _measure_columns(rows, width, show_pct=False), width))
    body.blank()


def _draw_runtime_and_scope(
    body: _Body, report: SessionReport | None, runtime: SessionDiagnostics
) -> None:
    body.header("Runtime", "live")
    body.kv("Selected", runtime.selected_model)
    body.kv("Effective", runtime.effective_model)
    body.kv("State", "streaming" if runtime.streaming else "idle")
    # Dropped when absent rather than printed as "unavailable": a row whose only
    # content is the word "unavailable" is exactly the noise this screen sheds.
    if runtime.generation is not None:
        body.kv("Turn generation", str(runtime.generation))
    if report is not None and report.first_ts_ms is not None:
        body.note("Period  " + _stamp(report.first_ts_ms) + " to " + _stamp(report.last_ts_ms))
    body.blank()

    body.header("Scope", "this session ID only")
    body.note(
        "Retained local ledger rows for this exact ID only. Child sessions and copied "
        "fork history are excluded; resuming the same ID includes its retained records."
    )
    body.note(
        "Older records may have been pruned. In-flight requests and pending recorder "
        "writes may not appear yet. Cost is provider-reported or a list-price estimate; "
        "input, output and tool dollars are not recorded separately. This command makes "
        "no model request."
    )


class SessionScreen(ModalScreen[None]):
    """A snapshot with analytics chrome and its bar vocabulary, session-scoped.

    Deliberately without the calendar charts: the daily/monthly rollups are
    machine-wide, so a "this session" heading over them would attribute other
    sessions' spend here.
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

    def __init__(self, report: SessionReport | None, runtime: SessionDiagnostics) -> None:
        super().__init__()
        self.report = report
        self.runtime = runtime
        self.presentation_cancelled = False
        #: Which metric the bars plot; ``t`` flips it. Tokens by default: this
        #: screen is read mid-session to answer "where did my context go", which
        #: is a token question, where ``/analytics`` answers a spend question.
        self._metric = METRIC_TOKENS

    def compose(self) -> ComposeResult:
        with Container(classes="analytics-panel"):
            self._title = Static(self._title_text(), id="session-report-title")
            yield self._title
            with VerticalScroll(id="session-report-scroll") as scroll:
                self._scroll = scroll
                self._body = Static(self._report_text(), id="session-report-body")
                yield self._body
            self._hint = Static(self._back_hint(), id="session-report-hint")
            yield self._hint

    def on_mount(self) -> None:
        # A fast disk read can finish while this screen is still mounting. The
        # stored result, not the first compose-time text, must win that race.
        self._repaint()
        self.call_after_refresh(self._dismiss_if_cancelled)

    def on_unmount(self) -> None:
        self.presentation_cancelled = True

    def on_screen_resume(self) -> None:
        self._dismiss_if_cancelled()

    def invalidate(self) -> None:
        """Retire this request without ever popping a newer modal above it."""
        self.presentation_cancelled = True
        self._dismiss_if_cancelled()

    def _dismiss_if_cancelled(self) -> None:
        # An owner switch may happen while another modal covers this one.
        # Dismiss only when resumed: Screen.dismiss() pops the CURRENT screen,
        # not necessarily the instance on which it was called.
        # A queued screen is registered before its mount finishes. Popping it
        # during that mount races Textual's parent teardown; on_mount schedules
        # another check after layout, without ever publishing the stale data.
        if self.presentation_cancelled and self.is_mounted and self.app.screen is self:
            self.dismiss(None)

    def set_report(self, report: SessionReport) -> None:
        """Publish the disk result only while this presentation is still owned."""
        if self.presentation_cancelled:
            return
        self.report = report
        self._repaint()

    def _card_width(self) -> int:
        """The content cells a report row may actually occupy.

        MEASURED off the mounted scroll container rather than recomputed from
        the CSS, because a formula and a stylesheet drift: ``AnalyticsScreen``
        derives ``card - 6`` from the ``.analytics-panel`` rule (90% of the
        terminal, capped at 140, ``padding: 1 2``) and lands one cell wide,
        because ``#*-scroll`` also reserves a stable scrollbar gutter. That one
        cell is why ``/analytics`` overflows horizontally at 50 columns, and
        building rows one cell too wide here folded the session ID onto a
        second line — the exact "one record reads as two" fault these charts
        exist to remove.

        Before mount there is nothing to measure, so the same geometry is
        derived arithmetically, gutter included.
        """
        scroll = getattr(self, "_scroll", None)
        try:
            if scroll is not None and scroll.is_mounted and scroll.size.width:
                # The gutter is reserved whether or not the thumb is showing,
                # so subtracting it unconditionally matches the painted box.
                return max(_MIN_CARD_WIDTH, scroll.size.width - 1)
            card = min(140, int(self.app.size.width * 0.9))
            return max(_MIN_CARD_WIDTH, card - 7)  # 4 padding + 2 border + 1 gutter
        except Exception:  # noqa: BLE001 — before mount, a sane default
            return _DEFAULT_CARD_WIDTH

    def _title_text(self) -> Text:
        # The rule is sized to the measured card rather than a hardcoded 140
        # cropped by the widget, matching AnalyticsScreen.
        return Text(
            "Session diagnostics\n" + "─" * max(1, self._card_width()),
            no_wrap=True,
            overflow="crop",
        )

    def _report_text(self) -> Text:
        return build_session_report(
            self.report, self.runtime, self._card_width(), metric=self._metric
        )

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is not None and body.is_mounted and not self.presentation_cancelled:
            body.update(self._report_text())
            title = getattr(self, "_title", None)
            if title is not None and title.is_mounted:
                title.update(self._title_text())
            self.call_after_refresh(self._sync_hint)

    def _has_charts(self) -> bool:
        """Whether any bar is on screen for ``t`` to act on.

        A loading, unavailable or empty report has no bars, so advertising the
        toggle would promise a dead control — the same rule the scroll hint
        follows.
        """
        report = self.report
        return bool(report and (report.by_model or report.by_purpose or report.recent))

    def _back_hint(self) -> str:
        hint = "esc / q cancel" if self.report is None else "esc / q back"
        if self._has_charts():
            hint += " · t cost/tokens"
        return hint

    def on_resize(self) -> None:
        # Column arithmetic is a function of the card width, so a resize is a
        # re-render, not just a hint refresh.
        self._repaint()
        self.call_after_refresh(self._sync_hint)

    def _sync_hint(self) -> None:
        # Like analytics, don't advertise a dead scroll control when an empty
        # report fits a tall terminal. Layout must settle before measuring it.
        scroll = getattr(self, "_scroll", None)
        hint = getattr(self, "_hint", None)
        if scroll is not None and hint is not None and hint.is_mounted:
            hint.update(self._back_hint() + (" · ↑↓ scroll" if scroll.max_scroll_y > 0 else ""))

    def action_dismiss_screen(self) -> None:
        self.invalidate()

    def action_toggle_metric(self) -> None:
        """Flip the bars between tokens and cost, repainting from the held report.

        No second store read: the numbers do not change, only which of them the
        bars are proportional to. A no-op when nothing is charted, so the key
        never silently toggles invisible state.
        """
        if not self._has_charts():
            return
        self._metric = METRIC_COST if self._metric == METRIC_TOKENS else METRIC_TOKENS
        self._repaint()

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
