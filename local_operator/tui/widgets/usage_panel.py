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
#: two hundred. The margins keep the card off the screen's own edge padding.
PANEL_MAX_WIDTH = 76
PANEL_MIN_WIDTH = 32
PANEL_WIDTH_MARGIN = 4
PANEL_HEIGHT_MARGIN = 4

#: Rows the pinned chrome costs: the title, the rule under it, and the hint row.
CHROME_ROWS = 3

#: One cell of padding inside the card, matching the toast's.
PANEL_PADDING_CELLS = 2

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
    reset_cells = [
        cell_len(format_countdown(limit.resets_in_ms(now_ms))) for limit in limits
    ]
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


def build_usage_body(reports, width: int, now_ms: float) -> list[Text]:  # noqa: ANN001
    """The scrolling half of the panel: one block per report.

    Column widths are measured over EVERY row rather than per provider, so the
    bars of two providers line up and the whole panel can be read as one table.
    """
    lines: list[Text] = []
    if not reports:
        return lines

    columns = _measure_columns(reports, width, now_ms)

    dim = Style(color=theme_mod.semantic_color("dim"))
    for index, report in enumerate(reports):
        if index:
            lines.append(Text())
        lines.append(_provider_header(report, now_ms))
        if report.notes:
            lines.append(Text(f"  {report.notes}", style=dim))
        if not report.limits:
            lines.append(Text("  no windows reported", style=dim))
            continue
        for limit in report.limits:
            lines.append(_limit_row(limit, columns, now_ms))
    return lines


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
        self._scroll_by(delta * max(1, self._body_budget()))

    def action_scroll_home(self) -> None:
        self._offset = 0
        self._repaint()

    def action_scroll_end(self) -> None:
        self._offset = self._max_offset()
        self._repaint()

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

    def _body_budget(self) -> int:
        """Rows the scrolling half may use, after the chrome and the screen."""
        _, height = self._screen_size()
        return max(1, height - PANEL_HEIGHT_MARGIN - CHROME_ROWS)

    def _max_offset(self) -> int:
        return max(0, len(self._body_lines()) - self._body_budget())

    # -- rendering -----------------------------------------------------------
    def _body_lines(self) -> list[Text]:
        dim = Style(color=theme_mod.semantic_color("dim"))
        if self._error:
            return [Text(self._error, style=Style(color=theme_mod.semantic_color("danger")))]
        if self._loading:
            return [Text("fetching…", style=dim)]
        if not self._reports:
            # Two failures look identical in an empty panel and only one is
            # actionable, so the message names both rather than the shorter one.
            target = f"{self._target} reports" if self._target else "no provider reports"
            return [
                Text(
                    f"{target} no usage — no quota endpoint, or no credential for one",
                    style=dim,
                )
            ]
        return build_usage_body(self._reports, self._content_width(), self._now())

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
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        row = Text()
        stats = collect_stats(self._reports).describe() if self._reports else ""
        if stats:
            row.append(stats, style=dim)
            row.append("   ", style=faint)
        for index, (key, what) in enumerate(KEY_HINTS):
            # Scroll keys are only offered when there is something to scroll: a
            # hint for a key that does nothing teaches the user to distrust the
            # other two.
            if key == "↑↓" and not scrolled:
                continue
            if index:
                row.append(" · ", style=faint)
            row.append(key, style=dim)
            row.append(f" {what}", style=faint)
        return row

    def render_lines_for_test(self) -> list[str]:
        """The panel as plain strings, chrome included — what a user reads."""
        return [line.plain for line in self._compose_rows()]

    def _compose_rows(self) -> list[Text]:
        body = self._body_lines()
        budget = self._body_budget()
        self._offset = max(0, min(self._offset, max(0, len(body) - budget)))
        window = body[self._offset : self._offset + budget]
        scrolled = len(body) > budget
        rule = Text(
            "─" * self._content_width(), style=Style(color=theme_mod.semantic_color("edge"))
        )
        rows = [self._title_row(), rule, *window]
        if scrolled:
            # The position is part of the CHROME, not a row of the report: a
            # "12 of 30" that scrolled away with the content would be useless
            # exactly when it is needed.
            marker = Text(
                f"{self._offset + len(window)} of {len(body)}",
                style=Style(color=theme_mod.semantic_color("faint")),
            )
            rows.append(marker)
        rows.append(self._hint_row(scrolled))
        return rows

    def _recentre(self, width: int, height: int) -> None:
        """Pull the HOST over the panel and centre the pair on the screen.

        The host is ``width: auto`` so it hugs this card, exactly as the toast's
        does: a widget owns its whole region and Textual blanks all of it, so a
        stretched host on the overlay layer erases the transcript either side of
        the panel. Centring therefore cannot be `align: center middle` in the
        stylesheet — that needs the stretched host — and is done here instead,
        against the same measured screen the bars are sized from.
        """
        parent = self.parent
        if parent is None or isinstance(parent, Screen):
            # A panel mounted straight onto the screen would shift the whole app
            # sideways, which is louder than an off-centre popup.
            return
        screen_width, screen_height = self._screen_size()
        parent.styles.offset = (
            max(0, (screen_width - width) // 2),
            max(0, (screen_height - height) // 2),
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
        self.styles.height = len(rows)
        self._recentre(width, len(rows))
        out = Text()
        for index, row in enumerate(rows):
            if index:
                out.append("\n")
            out.append_text(row)
        self.update(out)

    def on_mount(self) -> None:
        if self.display:
            self._repaint()
