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
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    return f"{n / 1_000_000_000:.1f}B".replace(".0B", "B")


def format_percent(fraction: float | None) -> str:
    """``73%`` / ``—`` for an unmeasurable rate."""
    if fraction is None:
        return "—"
    return f"{round(fraction * 100)}%"


def proportion_bar(fraction: float, width: int) -> str:
    """A filled proportion bar of ``width`` cells for ``fraction`` in 0..1."""
    width = max(1, width)
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
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


def build_report(aggregate: UsageAggregate, width: int) -> list[Text]:
    """Render one aggregate as a list of ``Text`` lines for the screen body.

    Pure: takes the summed data and a width, returns lines. The screen wraps
    this in a scroll container and owns the chrome, so everything about WHAT is
    shown lives here where a test can read it back as plain strings.
    """
    width = max(40, width)
    fg = _semantic("fg")
    label = _semantic("label")
    dim = _semantic("dim")
    faint = _semantic("faint")
    accent = _semantic("accent")

    lines: list[Text] = []

    if aggregate.calls == 0:
        line = Text()
        line.append("No usage recorded yet.", style=dim)
        lines.append(line)
        hint = Text()
        hint.append(
            "Analytics accrue as sessions make provider calls. " "Come back after a few turns.",
            style=faint,
        )
        lines.append(hint)
        return lines

    # -- headline totals -----------------------------------------------------
    section = Text()
    section.append("TOTALS", style=label + Style(bold=True))
    section.append(f"   {aggregate.calls} calls", style=dim)
    if aggregate.ok_calls != aggregate.calls:
        section.append(f" ({aggregate.calls - aggregate.ok_calls} failed)", style=faint)
    lines.append(section)

    def kv(name: str, value: str, note: str = "") -> Text:
        row = Text()
        row.append(f"  {name:<22}", style=dim)
        row.append(value, style=fg)
        if note:
            row.append(f"  {note}", style=faint)
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
    lines.append(Text())

    # -- input attribution (estimated) --------------------------------------
    heading = Text()
    heading.append("WHERE INPUT WENT", style=label + Style(bold=True))
    heading.append("   estimated split of context tokens", style=faint)
    lines.append(heading)

    rows = _component_rows(aggregate)
    if not rows:
        empty = Text()
        empty.append("  no component data yet", style=faint)
        lines.append(empty)
    else:
        label_col = min(34, max(len(r.label) for r in rows) + 1)
        value_col = max(len(format_tokens(r.value)) for r in rows)
        bar_width = max(8, min(24, width - label_col - value_col - 12))
        for row in rows:
            line = Text()
            line.append(f"  {row.label:<{label_col}}", style=fg)
            line.append(proportion_bar(row.fraction, bar_width), style=accent)
            line.append(f" {format_tokens(row.value):>{value_col}}", style=fg)
            line.append(f" {format_percent(row.fraction):>4}", style=dim)
            lines.append(line)
    lines.append(Text())

    # -- by provider ---------------------------------------------------------
    if aggregate.by_provider:
        lines.append(_group_section("BY PROVIDER", aggregate.by_provider))
        lines.append(Text())

    # -- by session ----------------------------------------------------------
    if aggregate.by_session:
        names = getattr(aggregate, "session_names", {}) or {}
        labelled = {
            short_session_label(sid, names.get(sid, "")): agg
            for sid, agg in aggregate.by_session.items()
        }
        lines.append(_group_section("BY SESSION", labelled))

    return lines


def _group_section(
    title: str,
    groups: dict[str, UsageAggregate],
) -> Text:
    """A per-provider or per-session table as one multi-line ``Text`` block.

    Returned as a single ``Text`` with embedded newlines so the caller keeps a
    flat list of blocks; the screen splits on newlines only for the scroll
    measurement. Sorted by total billed tokens, descending — the biggest
    spender is what a diagnostics reader looks for first.
    """
    fg = _semantic("fg")
    label = _semantic("label")
    dim = _semantic("dim")
    faint = _semantic("faint")

    block = Text()
    block.append(title, style=label + Style(bold=True))

    ordered = sorted(groups.items(), key=lambda kv: kv[1].total_tokens, reverse=True)
    if not ordered:
        block.append("\n  (none)", style=faint)
        return block

    name_col = min(30, max(len(name) for name, _ in ordered) + 1)
    for name, agg in ordered:
        block.append("\n")
        block.append(f"  {name:<{name_col}}", style=fg)
        block.append(f"{format_tokens(agg.total_tokens):>8} tokens", style=fg)
        cache = format_percent(agg.cache_hit_rate)
        block.append(f"   {agg.calls:>4} calls", style=dim)
        block.append(f"   {cache:>4} cache", style=faint)
    return block


class AnalyticsDismissed:
    """Sentinel result for the screen's dismiss (kept simple: no payload)."""


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
            yield Static(self._hint_text(), id="analytics-hint")

    def on_mount(self) -> None:
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        self._repaint()

    def _card_width(self) -> int:
        try:
            return max(40, min(96, self.app.size.width - 8))
        except Exception:  # noqa: BLE001 — before mount, a sane default
            return 80

    def _title_text(self) -> Text:
        label = Style(color=theme_mod.semantic_color("label"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        title = Text()
        title.append("Usage analytics", style=label + Style(bold=True))
        title.append("  ·  all sessions", style=faint)
        return title

    def _hint_text(self) -> Text:
        faint = Style(color=theme_mod.semantic_color("faint"))
        hint = Text()
        hint.append("esc back · ↑↓ scroll", style=faint)
        return hint

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
