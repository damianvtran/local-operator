"""The ``/usage`` popup — provider quota as a scrollable, focused overlay.

Why a popup rather than the transcript table it replaces: a quota report is
REFERENCE material, not conversation. Appended as a block it was pushed off
screen by the next turn, it could not be re-read without re-fetching, and a
long report (four Anthropic windows, an OpenRouter balance, a Kimi plan) simply
scrolled the work out of view to say something the user asked about for five
seconds. The overlay holds the numbers still, scrolls on its own, and leaves
without a trace.

The reference implementation for the CONTENT is ``omp usage``: a bar, a
percentage, and a reset countdown per window, grouped by provider and account.
Three things are done differently here, all of them because this surface is a
live panel rather than a dump into a terminal:

- **The binding window is named, not buried.** ``omp`` lists every window at
  equal weight and leaves the reader to scan for the one that is about to stop
  them. Each provider block here leads with its most-pressured window, so the
  answer to "can I keep working" is on the first row of the block.
- **Account-wide and per-model caps are visually separate.** A tier row
  (Anthropic's ``7 day (Fable)``) stops one model family; a shared row stops the
  account. ``omp`` renders both identically, which is how a 100% Fable cap reads
  as a dead account. Tier rows are indented and dimmed.
- **Chrome is pinned, content scrolls.** The title (with fetch age) and the key
  hints never move, so a user scrolling a long report never loses the way out.

Everything above the widget class is a pure function of ``(reports, width,
now)`` and is tested without a running app; the widget only holds the scroll
offset and the focus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import NoScreen
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells

#: Bar geometry. The bar is the only element here that can be made narrower
#: without losing information — a percentage is exact and a countdown is words —
#: so it is what absorbs a small terminal, down to the floor where a bar stops
#: being readable as a proportion at all and is dropped entirely.
BAR_MAX_CELLS = 24
BAR_MIN_CELLS = 8

#: Filled / empty cells, and the glyph for a window that reported no number.
#: Dots rather than an empty bar for the unknown case: an empty bar says "you
#: have used none of it", which is a claim, and none was made.
BAR_FILLED = "█"
BAR_EMPTY = "░"
BAR_UNKNOWN = "·"

#: Status marks. Filled for a window with a number, hollow for one without —
#: the same "reported / not reported" distinction ``omp`` draws with ``●``/``○``.
MARK_KNOWN = "●"
MARK_UNKNOWN = "○"

#: Indent for a per-model tier row, so it reads as subordinate to the
#: account-wide windows above it rather than as a peer that gates everything.
TIER_INDENT = "  "

#: Panel geometry. The width cap is a measure, not a fraction of the terminal:
#: a label, a bar and two numbers need about seventy cells and gain nothing from
#: two hundred. The width margin keeps the card off the screen's edge padding.
PANEL_MAX_WIDTH = 76
PANEL_MIN_WIDTH = 32
PANEL_WIDTH_MARGIN = 4

#: Ground the card keeps between itself and the two surfaces it floats between:
#: one row under the screen's top inset, one row above the docked input panel.
#: The card's fill is an elevation step, and a step flush against the input
#: panel's fill reads as two stacked bars rather than a card lifted off the page.
#:
#: Advisory rather than reserved — it comes off the rows the card may use, and
#: the one-row body floor overrides it on a terminal too short to grant both.
PANEL_HEIGHT_MARGIN = 2

#: Rows the pinned chrome costs: the title, the rule under it, the blank row
#: that separates the report from the footer, the ``N of M`` position, and the
#: hint row itself. The spacer is chrome rather than a body row because it must
#: survive scrolling — a footer that sits flush against the last meter reads as
#: one more data row.
#:
#: The position row is counted even though it is drawn only when the report
#: SCROLLS, because scrolling is exactly when the budget binds: a card sized
#: without it came out one row taller than the space it had been measured
#: against, and that row landed on the docked prompt.
CHROME_ROWS = 5

#: Inner padding, as CELLS across (``padding: 1 2`` → 2 left + 2 right) and
#: ROWS down (1 top + 1 bottom). Both are declared here as well as in the
#: stylesheet because the widget SIZES ITSELF: Textual's width/height are
#: border-box, so the panel must add the padding back when it pins its own
#: height and subtract it when it measures the content that has to fit. A
#: single-cell gutter is what made the card read as cramped against the text
#: it floats over — the overlay needs to feel lifted off the transcript, and
#: a row of quiet at the top and bottom is what does that.
PANEL_PADDING_CELLS = 4
PANEL_PADDING_ROWS = 2

#: Rows above the dock at which the card can still afford its vertical gutter.
#: Below it the gutter is dropped (the ``-squeezed`` class in the stylesheet),
#: because the alternative is a card that covers the prompt: the margin, chrome,
#: one body row and gutter are the smallest useful card, and the gutter is the
#: only decorative part. A terminal that short has no breathing room to protect.
SQUEEZE_ROWS = PANEL_HEIGHT_MARGIN + CHROME_ROWS + PANEL_PADDING_ROWS + 1

#: Keys the panel offers, in the order the hint row prints them. Data rather
#: than a literal string so the hints and the bindings cannot drift apart.
KEY_HINTS: tuple[tuple[str, str], ...] = (
    ("↑↓", "scroll"),
    ("r", "refresh"),
    ("esc", "close"),
)


class UsageDismissed(Message):
    """The panel closed itself (Esc/q). The app restores focus and hides it."""


class UsageRefreshRequested(Message):
    """``r`` was pressed. The app owns the fetch, so it owns the refresh."""


@dataclass(frozen=True)
class UsageStats:
    """The one-line tally the footer reports.

    Counted over WINDOWS rather than providers because a window is what runs
    out: one provider with an exhausted weekly cap and three healthy ones is a
    different situation from four providers each at 90%, and a provider count
    cannot tell them apart.
    """

    windows: int = 0
    exhausted: int = 0
    warning: int = 0
    unknown: int = 0

    def describe(self) -> str:
        """``6 windows · 1 exhausted · 1 near limit`` — worst state first."""
        if not self.windows:
            return ""
        parts = [f"{self.windows} window{'s' if self.windows != 1 else ''}"]
        if self.exhausted:
            parts.append(f"{self.exhausted} exhausted")
        if self.warning:
            parts.append(f"{self.warning} near limit")
        if self.unknown:
            parts.append(f"{self.unknown} not reported")
        return " · ".join(parts)


def collect_stats(
    reports,
) -> UsageStats:  # noqa: ANN001 — provider types stay off the TUI import graph
    """Tally every window across every report by its effective status."""
    windows = exhausted = warning = unknown = 0
    for report in reports:
        for limit in report.limits:
            windows += 1
            status = limit.effective_status()
            if status == "exhausted":
                exhausted += 1
            elif status == "warning":
                warning += 1
            elif status == "unknown":
                unknown += 1
    return UsageStats(windows, exhausted, warning, unknown)


def format_age(ms: float) -> str:
    """``just now`` / ``40s ago`` / ``3m ago`` — how stale the numbers are.

    Sub-minute is ``just now`` because the exact second a report landed is never
    the question; whether it predates the work being done is.
    """
    seconds = max(0.0, ms / 1000)
    if seconds < 60:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    return f"{hours}h ago" if hours < 24 else f"{hours // 24}d ago"


def format_countdown(ms: int | None) -> str:
    """``3h24m`` / ``2d11h`` / ``45m`` — time until a window rolls over.
    Two units at most, largest first, and the smaller one is dropped when it is
    zero. A countdown is read at a glance to answer "wait, or switch model", and
    ``3d11h47m`` answers that no better than ``3d11h`` while costing the width
    the reset column has to fit in.
    """
    if ms is None or ms <= 0:
        return ""
    minutes = int(ms // 60_000)
    if minutes < 60:
        return f"{max(1, minutes)}m"
    hours, rem_minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{rem_minutes}m" if rem_minutes else f"{hours}h"
    days, rem_hours = divmod(hours, 24)
    return f"{days}d{rem_hours}h" if rem_hours else f"{days}d"


#: The human prefix a reset countdown sits under. Held as a constant so the
#: width measurement (:func:`_measure_columns`) and the row renderer use the
#: same string and the reset column always fits its own content.
_RESET_PREFIX = "resets in "


def format_amount(amount) -> str:  # noqa: ANN001 — see collect_stats
    """The number for one row, or ``""`` when the amount carries none.

    ``used`` first, then ``remaining``, then ``limit``, then an explicit
    fraction. A remaining-only balance — what both account-balance fetchers
    report, since neither vendor gives a limit to derive spend from — must still
    print its number, or a row labelled "Balance" never says how much.

    Percent rows are integers: the endpoint quotes ``2.0`` and ``100.0``, and a
    trailing ``.0`` on every row costs two cells to say nothing. USD keeps its
    cents, because there money is the unit of measure rather than a proportion.
    """
    from local_operator.providers.usage import UNIT_LABELS

    unit = getattr(amount, "unit", "unknown")
    label = UNIT_LABELS.get(unit, unit)

    def with_unit(value: float) -> str:
        if unit == "usd":
            text = f"{value:.2f}"
        elif unit == "percent":
            text = f"{value:.0f}"
        else:
            text = f"{value:g}"
        if not label:
            return text
        # "%" is a suffix; every other label is a separate word.
        return f"{text}{label}" if label == "%" else f"{text} {label}"

    if amount.used is not None:
        # A used/limit pair in real units says more than the used half alone:
        # "$12.00" is meaningless without the cap it is drawn against, while a
        # percentage already carries its own denominator.
        if unit != "percent" and amount.limit:
            return f"{with_unit(amount.used)} / {with_unit(amount.limit)}"
        return with_unit(amount.used)
    if amount.remaining is not None:
        return f"{with_unit(amount.remaining)} left"
    if amount.limit is not None:
        return f"{with_unit(amount.limit)} limit"
    if amount.used_fraction is not None:
        return f"{amount.used_fraction * 100:.0f}% used"
    return ""


def usage_bar(fraction: float | None, width: int) -> str:
    """A proportion bar, or dots when the fraction is unmeasurable.

    A non-zero fraction always fills at least one cell. Rounding 1% of a
    24-cell bar to zero draws an empty bar for an account that HAS started
    spending, which is the one reading the bar exists to prevent.
    """
    if width <= 0:
        return ""
    if fraction is None:
        return BAR_UNKNOWN * width
    clamped = max(0.0, min(1.0, fraction))
    filled = round(clamped * width)
    if clamped > 0:
        filled = max(1, filled)
    if clamped < 1.0:
        filled = min(filled, width - 1)
    return BAR_FILLED * filled + BAR_EMPTY * (width - filled)


def _status_color(status: str) -> str:
    return {
        "ok": theme_mod.semantic_color("success"),
        "warning": theme_mod.semantic_color("warning"),
        "exhausted": theme_mod.semantic_color("danger"),
    }.get(status, theme_mod.semantic_color("dim"))


def binding_limit(report):  # noqa: ANN001, ANN201 — see collect_stats
    """The window closest to stopping this account, or None.

    ACCOUNT-WIDE windows win over per-model ones regardless of fill: a Fable cap
    at 100% stops Fable, while a shared window at 80% is the one throttling
    every request, and calling the tier row "binding" would tell a user to stop
    working when they only needed to change model.
    """
    best = None
    best_key: tuple[int, float] | None = None
    for limit in report.limits:
        fraction = limit.amount.fraction()
        if fraction is None:
            continue
        key = (1 if limit.shared else 0, fraction)
        if best_key is None or key > best_key:
            best, best_key = limit, key
    return best


@dataclass(frozen=True)
class _Columns:
    """The measured column widths one panel's rows share.

    Every column except the bar has a width its content dictates; the bar takes
    whatever is left, down to nothing. That ordering is the layout rule: a
    percentage is exact and a countdown is words, while a bar is a redundant
    picture of a number already on the row, so it is what a narrow terminal
    takes cells from.
    """

    label: int
    numbers: int
    reset: int
    bar: int


def _measure_columns(reports, width: int, now_ms: float) -> _Columns:  # noqa: ANN001
    """Column widths measured over EVERY row, so the whole panel is one table.

    Measured per provider, two blocks would each be internally aligned and
    misaligned with each other, which reads as two tables that happen to be
    stacked.
    """
    limits = [limit for report in reports for limit in report.limits]
    label_cells = [
        cell_len(limit.label) + (len(TIER_INDENT) if limit.tier else 0) for limit in limits
    ]
    numbers_cells = [cell_len(format_amount(limit.amount) or "not reported") for limit in limits]
    reset_cells = [cell_len(format_countdown(limit.resets_in_ms(now_ms))) for limit in limits]
    # The label truncates rather than pushing the numbers off the row: it is the
    # one column whose content the user can still identify from a prefix.
    label = min(max(label_cells, default=0), max(12, width // 3))
    numbers = max(numbers_cells, default=0)
    longest_reset = max(reset_cells, default=0)
    # `resets in ` is only paid for when some row actually has a countdown.
    reset = (len(_RESET_PREFIX) + longest_reset) if longest_reset else 0
    fixed = 2 + label + 2 + numbers + (2 + reset if reset else 0)
    bar = max(0, min(BAR_MAX_CELLS, width - fixed - 2))
    return _Columns(label=label, numbers=numbers, reset=reset, bar=bar)


def _limit_row(limit, columns: _Columns, now_ms: float) -> Text:  # noqa: ANN001
    """One window: mark, label, bar, number, reset countdown."""
    status = limit.effective_status()
    tint = Style(color=_status_color(status))
    dim = Style(color=theme_mod.semantic_color("dim"))
    faint = Style(color=theme_mod.semantic_color("faint"))
    # A tier row is subordinate, so it takes the dim ramp even at full: the
    # colour says "this window", the weight says "this window gates everything".
    label_style = dim if limit.tier else Style(color=theme_mod.semantic_color("muted"))

    fraction = limit.amount.fraction()
    row = Text()
    row.append(f"{MARK_KNOWN if fraction is not None else MARK_UNKNOWN} ", style=tint)

    label = limit.label
    if limit.tier:
        label = TIER_INDENT + label
    row.append(truncate_cells(label, columns.label).ljust(columns.label), style=label_style)

    if columns.bar:
        bar = usage_bar(fraction, columns.bar)
        row.append("  ")
        if fraction is None:
            row.append(bar, style=faint)
        else:
            filled = bar.count(BAR_FILLED)
            row.append(bar[:filled], style=tint)
            row.append(bar[filled:], style=faint)

    numbers = format_amount(limit.amount) or "not reported"
    row.append("  ")
    row.append(numbers.rjust(columns.numbers), style=dim)

    if columns.reset:
        countdown = format_countdown(limit.resets_in_ms(now_ms))
        # A window with no countdown still pads its cell, so the rows that do
        # have one stay in a column rather than starting wherever their numbers
        # happened to end.
        text = f"{_RESET_PREFIX}{countdown}" if countdown else ""
        row.append("  ")
        row.append(text.ljust(columns.reset), style=faint)
    return row


def _provider_header(report, now_ms: float) -> Text:  # noqa: ANN001
    """``anthropic  (me@example.com)`` plus the window that binds.

    The binding window is stated HERE, on the block's first row, because that is
    the question a usage view is opened to answer. ``omp`` leaves it to be found
    among equals.
    """
    heading = Style(color=theme_mod.semantic_color("label"))
    muted = Style(color=theme_mod.semantic_color("muted"))
    dim = Style(color=theme_mod.semantic_color("dim"))

    row = Text()
    row.append(report.provider, style=heading)
    if report.identity:
        row.append(f"  {report.identity}", style=muted)
    binding = binding_limit(report)
    if binding is not None:
        fraction = binding.amount.fraction() or 0.0
        tint = Style(color=_status_color(binding.effective_status()))
        row.append("  ·  ", style=dim)
        row.append(f"{binding.label} {fraction * 100:.0f}%", style=tint)
        countdown = format_countdown(binding.resets_in_ms(now_ms))
        if countdown:
            row.append(f", resets in {countdown}", style=dim)
    return row


@dataclass(frozen=True)
class UsageBody:
    """The scrolling half of the panel and its semantic block boundaries.

    ``cuts`` holds row counts at which a window may end. ``blocks`` holds
    ``(start, end)`` ranges for provider groups, allowing a short viewport to
    remove decorative air without separating meters from their provider.
    """

    lines: list[Text]
    cuts: frozenset[int]
    blocks: tuple[tuple[int, int], ...] = ()


def build_usage_body(reports, width: int, now_ms: float) -> UsageBody:  # noqa: ANN001
    """The scrolling half of the panel: one block per report.

    Column widths are measured over EVERY row rather than per provider, so the
    bars of two providers line up and the whole panel can be read as one table.

    Each block is heading → (notes) → blank → meters. The blank row is what
    makes the heading read as a heading: without it the identity line, the
    account note and the first meter run together as three equal rows, which is
    the "tight" the card was reported for. The blank is also why the returned
    :class:`UsageBody` carries its cut points: those three rows are one unit,
    and a window may not stop part way through them.
    """
    lines: list[Text] = []
    cuts: set[int] = set()
    blocks: list[tuple[int, int]] = []
    if not reports:
        return UsageBody(lines, frozenset({0}))

    columns = _measure_columns(reports, width, now_ms)

    dim = Style(color=theme_mod.semantic_color("dim"))
    for index, report in enumerate(reports):
        if index:
            lines.append(Text())
        block_start = len(lines)
        header = _provider_header(report, now_ms)
        # Provider/account identifiers are user data, not geometry. A long
        # identity must lose its tail rather than widen or clip the whole card.
        header.truncate(max(1, width), overflow="ellipsis")
        lines.append(header)
        if report.notes:
            note = Text(f"  {report.notes}", style=dim)
            note.truncate(max(1, width), overflow="ellipsis")
            lines.append(note)
        if not report.limits:
            lines.append(Text("  no windows reported", style=dim))
            cuts.add(len(lines))
            blocks.append((block_start, len(lines)))
            continue
        lines.append(Text())
        for limit in report.limits:
            lines.append(_limit_row(limit, columns, now_ms))
            cuts.add(len(lines))
        blocks.append((block_start, len(lines)))
    return UsageBody(lines, frozenset(cuts), tuple(blocks))


class UsagePanel(Static):
    """The overlay itself: pinned chrome, a scrolling body, and the keys.

    State is only what the widget can own — the reports it was handed, whether a
    fetch is in flight, the scroll offset. The fetch belongs to the app (it owns
    the provider controller), so ``r`` and Esc are reported as messages rather
    than acted on here. That is the same split the pickers use.
    """

    can_focus = True

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("r", "refresh", "Refresh"),
        ("up", "scroll_rows(-1)", "Up"),
        ("down", "scroll_rows(1)", "Down"),
        ("pageup", "scroll_page(-1)", "Page up"),
        ("pagedown", "scroll_page(1)", "Page down"),
        ("home", "scroll_home", "Top"),
        ("end", "scroll_end", "Bottom"),
    ]

    def __init__(self) -> None:
        super().__init__(id="usage-panel")
        self._reports: list[Any] = []
        self._offset = 0
        self._loading = False
        self._error = ""
        self._target = ""
        self._fetched_ms: float | None = None
        self._clock: float = 0.0
        # Screen size plus the live bottom-dock ceiling used for the last
        # paint. The dock can grow without resizing this overlay, so Textual
        # emits no resize event for the panel itself.
        self._layout_shown: tuple[int, int, int] | None = None
        self.display = False

    # -- state ---------------------------------------------------------------
    def start_fetch(self, target: str = "") -> None:
        """Show the panel in its loading state.

        Opened BEFORE the request rather than after it: the fetch crosses the
        network for every logged-in provider, and a command that does nothing
        visible for two seconds reads as a command that did nothing.
        """
        self._loading = True
        self._error = ""
        self._target = target
        self.display = True
        self._repaint()

    def show_reports(self, reports, *, now_ms: float | None = None) -> None:  # noqa: ANN001
        """Install a finished fetch and paint it."""
        self._reports = list(reports)
        self._loading = False
        self._error = ""
        self._fetched_ms = self._now() if now_ms is None else now_ms
        self._offset = 0
        self.display = True
        self._repaint()

    def show_error(self, message: str) -> None:
        """A failed fetch stays IN the panel, next to the key that retries it."""
        self._loading = False
        self._error = message
        self.display = True
        self._repaint()

    def close(self) -> None:
        """Hide the panel and forget the scroll position."""
        self._loading = False
        self._offset = 0
        self.display = False

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    @property
    def target(self) -> str:
        """The provider ``/usage`` was scoped to, or ``""`` for all of them.

        Held HERE rather than on the app because the panel is what a refresh is
        issued from: two copies of "which provider is on screen" is how ``r``
        ends up re-fetching a different provider than the one being read.
        """
        return self._target

    @property
    def reports(self) -> list[Any]:
        return list(self._reports)

    @property
    def view_offset(self) -> int:
        return self._offset

    def set_clock(self, now_ms: float) -> None:
        """Pin the clock (tests; the panel otherwise reads the wall clock)."""
        self._clock = now_ms

    def _now(self) -> float:
        if self._clock:
            return self._clock
        import time

        return time.time() * 1000

    # -- actions -------------------------------------------------------------
    def action_dismiss(self) -> None:
        self.close()
        self.post_message(UsageDismissed())

    def action_refresh(self) -> None:
        self.post_message(UsageRefreshRequested())

    def action_scroll_rows(self, delta: int) -> None:
        self._scroll_by(delta)

    def action_scroll_page(self, delta: int) -> None:
        # A page is what the card is SHOWING, not what it budgeted for: the
        # window stops at a block boundary, so paging by the budget would step
        # over the heading that the boundary held back.
        body = self._body()
        shown = self._window_end(body, self._body_budget()) - self._offset
        self._scroll_by(delta * max(1, shown))

    def action_scroll_home(self) -> None:
        self._offset = 0
        self._repaint()

    def action_scroll_end(self) -> None:
        self._offset = self._max_offset()
        self._repaint()

    # The wheel scrolls the report the same rows the arrow keys do. Stopped
    # here because the card floats over the transcript: left to bubble, one
    # gesture would scroll both the quota table and the conversation behind it.
    def on_mouse_scroll_down(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(1)

    def on_mouse_scroll_up(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(-1)

    def _scroll_by(self, delta: int) -> None:
        """Scroll, CLAMPED rather than wrapping.

        A list of quota windows has a top and a bottom that mean something —
        the first provider and the last — and a Down that silently returns to
        the top looks like the panel reset itself.
        """
        self._offset = max(0, min(self._max_offset(), self._offset + delta))
        self._repaint()

    # -- geometry ------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        try:
            size = self.screen.size
        except NoScreen:
            return 80, 24
        return max(20, size.width), max(8, size.height)

    def panel_width(self) -> int:
        width, _ = self._screen_size()
        return max(PANEL_MIN_WIDTH, min(PANEL_MAX_WIDTH, width - PANEL_WIDTH_MARGIN))

    def _content_width(self) -> int:
        return max(1, self.panel_width() - PANEL_PADDING_CELLS)

    def _rows_above_dock(self) -> int:
        """Rows the card may occupy: the ground ABOVE the docked input panel.

        NOT "the screen height less a constant". The prompt is DOCKED, so the
        layout engine reserves its rows before it offers the rest to anything
        else, and how many it takes is a function of the editor's line count,
        the subagent/todo band and the boot layout — five rows to ten across the
        sizes the tests sweep. Measured against a constant instead, the card ran
        over the prompt at every size once the report was long enough to scroll:
        ``❯  Message Local Ope  24 windows · ↑↓ scroll`` sharing one row (D19).

        The absolute and relative boxes are reconciled HERE and handed back as a
        COUNT, so no caller can mix them: ``region`` is absolute while
        ``screen.size`` is the content box that ``Screen { padding: 1 }`` insets,
        and subtracting one from the other is what made a centred card look a
        cell off-centre in an earlier round.

        The bound is read off whatever is docked rather than off an id: the
        invariant is "the card covers no docked surface", and a rule that names
        ``#input-dock`` would go quietly back to overlapping if the dock were
        ever renamed. Hosts with nothing docked (the widget-only test app) get
        the whole content box, which is the same answer by the same rule.
        """
        try:
            screen = self.screen
        except NoScreen:
            return self._screen_size()[1]
        content = screen.content_region
        ceiling = content.bottom
        for sibling in screen.children:
            if sibling.display and sibling.styles.dock == "bottom":
                ceiling = min(ceiling, sibling.region.y)
        return max(1, ceiling - content.y)

    def _fit(self) -> tuple[int, int, int]:
        """``(rows above the dock, gutter rows, body budget)`` — one measurement.

        The three travel together because they are one sum: the pinned height a
        scrolled card settles at is exactly ``budget + CHROME_ROWS + gutter``, and
        that has to come out no greater than the rows above the dock. Splitting
        the terms across methods that each re-measure is how the budget and the
        height came to disagree by the position row in the first place.
        """
        rows = self._rows_above_dock()
        gutter = PANEL_PADDING_ROWS if rows >= SQUEEZE_ROWS else 0
        return rows, gutter, max(1, rows - PANEL_HEIGHT_MARGIN - CHROME_ROWS - gutter)

    def sync_layout(self) -> None:
        """Repaint when the screen or live dock changed around an open card."""
        if not self.display or not self.is_mounted:
            return
        width, height = self._screen_size()
        fingerprint = (width, height, self._rows_above_dock())
        if fingerprint != self._layout_shown:
            self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """A terminal resize changes both the card width and body budget."""
        self._repaint()

    def _body_budget(self) -> int:
        """Rows the scrolling half may use, after the chrome and the dock."""
        return self._fit()[2]

    def _max_offset(self) -> int:
        body = self._body()
        budget = self._body_budget()
        raw = max(0, len(body.lines) - budget)
        # If the raw tail begins inside a provider block, start at its heading.
        # `_window_rows` then removes notes/blank air and keeps the meters that
        # fit, so End never shows anonymous numbers.
        for start, end in body.blocks:
            if start <= raw < end and end - start > budget:
                return start
        return raw

    # -- rendering -----------------------------------------------------------
    def _body(self) -> UsageBody:
        dim = Style(color=theme_mod.semantic_color("dim"))
        width = self._content_width()
        if self._error:
            danger = Style(color=theme_mod.semantic_color("danger"))
            # Error text commonly comes from a provider and is unbounded. The
            # popup is intentionally single-line-per-row: wrap would consume
            # its footer/close receipt, while no truncation clips both.
            return UsageBody(
                [Text(truncate_cells(self._error, width), style=danger)],
                frozenset({1}),
            )
        if self._loading:
            return UsageBody([Text("fetching…", style=dim)], frozenset({1}))
        if not self._reports:
            # Two failures look identical in an empty panel and only one is
            # actionable, so the message names both rather than the shorter one.
            target = f"{self._target} reports" if self._target else "no provider reports"
            message = f"{target} no usage — no quota endpoint, or no credential for one"
            return UsageBody(
                [Text(truncate_cells(message, width), style=dim)],
                frozenset({1}),
            )
        return build_usage_body(self._reports, width, self._now())

    def _title_row(self) -> Text:
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        row = Text()
        row.append("Usage", style=Style(color=theme_mod.semantic_color("fg")))
        if self._target:
            row.append(f"  {self._target}", style=muted)
        if self._fetched_ms is not None and not self._loading and not self._error:
            row.append(f"  {format_age(self._now() - self._fetched_ms)}", style=dim)
        return row

    def _hint_row(self, scrolled: bool) -> Text:
        """Footer facts that fit, preserving actions in operational order.

        ``esc`` is non-negotiable, refresh repairs stale/error states, and
        scrolling is merely a convenience. The old left-to-right append let
        ``↑↓ scroll`` consume a narrow footer before ``r refresh`` appeared.
        """
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        width = self._content_width()
        stats = collect_stats(self._reports).describe() if self._reports else ""
        segments: list[tuple[str, str, str]] = []
        if stats:
            segments.append(("stats", stats, ""))
        for key, what in KEY_HINTS:
            if key == "↑↓" and not scrolled:
                continue
            segments.append((key, key, what))

        def painted_width(items: list[tuple[str, str, str]]) -> int:
            return (
                sum(cell_len(key) + (1 + cell_len(what) if what else 0) for _, key, what in items)
                + max(0, len(items) - 1) * 3
            )

        # Drop the least important receipts until the canonical, fully-labelled
        # row fits. Stats go only after scrolling; close is never dropped.
        for disposable in ("↑↓", "stats"):
            if painted_width(segments) <= width:
                break
            segments = [segment for segment in segments if segment[0] != disposable]

        row = Text()
        for _, key, what in segments:
            if row.plain:
                row.append(" · ", style=faint)
            row.append(key, style=dim)
            if what:
                row.append(f" {what}", style=faint)
        row.truncate(max(1, width), overflow="ellipsis")
        return row

    def render_lines_for_test(self) -> list[str]:
        """The panel as plain strings, chrome included — what a user reads."""
        return [line.plain for line in self._compose_rows()]

    def _compose_rows(self) -> list[Text]:
        body = self._body()
        lines = body.lines
        budget = self._body_budget()
        self._offset = max(0, min(self._offset, self._max_offset()))
        scrolled = len(lines) > budget
        window, shown_end = self._window_rows(body, budget)
        # ``edge`` is tuned against the app ground; the raised overlay ground
        # needs one raised step to preserve the same intentionally quiet ratio.
        faint = Style(color=theme_mod.semantic_color("faint"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        rule = Text("─" * self._content_width(), style=faint)
        rows = [self._title_row(), rule, *window]
        # Quiet ground between the report and the card's bottom meta, in BOTH
        # states. Without it the tally and the key hints sit flush against the
        # last meter and read as one more data row rather than as chrome.
        #
        # The rows a block boundary gave back (see _window_end) are quiet HERE,
        # above the meta, rather than taken off the card. Off the card they would
        # shrink it, and the card is centred on its own height — it would walk up
        # and down the screen as a reader scrolled it. Between the position and
        # the keys they would punch a hole through the middle of two meta rows,
        # which reads as something failing to render; above the pair the same
        # rows read as the card's bottom margin. So the frame never moves and
        # only the interior does.
        rows.extend(Text() for _ in range(1 + (budget - len(window) if scrolled else 0)))
        if scrolled:
            # The position is part of the CHROME, not a row of the report: a
            # counter that scrolled away with the content would be useless
            # exactly when it is needed. It sits with the key hints because the
            # two are the same KIND of row — both are statements about the list
            # rather than entries in it — so they travel together at the bottom.
            position = Text()
            position.append("showing ", style=faint)
            position.append(str(shown_end), style=dim)
            position.append(" of ", style=faint)
            position.append(str(len(lines)), style=dim)
            rows.append(position)
        rows.append(self._hint_row(scrolled))
        return rows

    def _window_rows(self, body: UsageBody, budget: int) -> tuple[list[Text], int]:
        """Visible rows plus the source position represented by their tail.

        A provider block taller than the short viewport is compacted
        semantically: heading first, then the last meters that fit. Notes and
        the decorative blank yield before identity or quota values.
        """
        for start, end in body.blocks:
            if self._offset == start and end - start > budget:
                data = [
                    body.lines[index] for index in range(start + 1, end) if index + 1 in body.cuts
                ]
                if budget <= 1:
                    return [body.lines[start]], end
                return [body.lines[start], *data[-(budget - 1) :]], end
        end = self._window_end(body, budget)
        return body.lines[self._offset : end], end

    def _window_end(self, body: UsageBody, budget: int) -> int:
        """Where the window stops: the last row that COMPLETES a block.

        A budget that ran out mid-block left a provider heading — or a heading
        and the blank under it — as the last thing on the card, announcing a
        provider and then showing none of its numbers (80x24 landed exactly
        there). The cut moves back to the end of the last whole block instead.

        A budget too small to hold even one block has nothing to give back, so
        it keeps the raw cut: the head of a block reads better than a card with
        an empty body.
        """
        end = min(self._offset + budget, len(body.lines))
        pulled = end
        while pulled > self._offset and pulled not in body.cuts:
            pulled -= 1
        return pulled if pulled > self._offset else end

    def _recentre(self, width: int, height: int) -> None:
        """Centre the host in the ground ABOVE the docked input panel.

        The host is ``width: auto`` so it hugs this card, exactly as the toast's
        does: a widget owns its whole region and Textual blanks all of it, so a
        stretched host on the overlay layer erases the transcript either side of
        the panel. Centring therefore cannot be ``align: center middle`` in the
        stylesheet — that needs the stretched host — and is done here instead.

        Horizontally the symmetric screen inset cancels, which a painted-frame
        test pins. Vertically there is NO such cancellation: the input is docked
        at the bottom. Centring against the whole screen put the lower half of a
        tall card on top of the prompt even though ``_fit`` had correctly sized
        it for the rows above that dock (D19). The same available-row count must
        therefore size AND place the card.
        """
        parent = self.parent
        if parent is None or isinstance(parent, Screen):
            # A panel mounted straight on the screen must not move the screen.
            return
        screen_width, _ = self._screen_size()
        rows_above_dock = self._rows_above_dock()
        parent.styles.offset = (
            max(0, (screen_width - width) // 2),
            max(0, (rows_above_dock - height) // 2),
        )

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        rows = self._compose_rows()
        width = self.panel_width()
        self.styles.width = width
        # Pinned for the same reason the tool card's height is: `auto` measures
        # the content against a guessed width before layout and settles a row
        # too tall, and this widget is repainted on every keystroke that scrolls
        # it — a card that grew by a row per keypress would crawl down the
        # screen while being read.
        #
        # The padding rows are ADDED BACK here (and to the centring height):
        # Textual sizes border-box, so pinning the row count alone would give
        # the gutter the last two content rows and clip the hint row off the
        # bottom of the card. At extreme heights ``_fit`` deliberately drops
        # that gutter; the class changes the stylesheet and the arithmetic as
        # one decision rather than merely pretending the padding disappeared.
        _, gutter, _ = self._fit()
        self.set_class(gutter == 0, "-squeezed")
        outer_height = len(rows) + gutter
        self.styles.height = outer_height
        self._recentre(width, outer_height)
        screen_width, screen_height = self._screen_size()
        self._layout_shown = (screen_width, screen_height, self._rows_above_dock())
        out = Text()
        for index, row in enumerate(rows):
            if index:
                out.append("\n")
            out.append_text(row)
        self.update(out)

    def on_mount(self) -> None:
        if self.display:
            self._repaint()
