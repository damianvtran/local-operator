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
from textual.dom import NoScreen
from textual.message import Message
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import overlay
from local_operator.tui.widgets.tool_card import truncate_cells

#: Bar geometry. The bar is the only element here that can be made narrower
#: without losing information — a percentage is exact and a countdown is words —
#: so it is what absorbs a small terminal, down to the floor where a bar stops
#: being readable as a proportion at all and is dropped entirely.
BAR_MAX_CELLS = 40
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

#: The scrollbar occupies the body's rightmost column. Its gutter is reserved in
#: EVERY state (see :meth:`UsagePanel._body_content_width`), not only while the
#: bar is drawn: reserving it conditionally would slide the report's
#: right-aligned numbers one column left the instant the bar appeared, which is
#: exactly the reflow the transcript avoids with ``scrollbar-gutter: stable``.
SCROLLBAR_GUTTER_CELLS = 1

#: How many cells LEFT of the bar column can still count as a grab. The bar is a
#: single hand-painted column, and a 1-cell mouse target is easy to miss; a miss
#: lands on the selectable ``Static`` and arms a text selection whose drag reads
#: as the messy highlight the user reported. Widening the HIT test (not the
#: painted bar — see :meth:`UsagePanel._paint_scrollbar`) leftward closes that
#: miss. Unlike the transcript (``TranscriptScreen.SCROLLBAR_GRAB_PAD`` in
#: ``app.py``), whose pad eats the view's own EMPTY right padding, this panel
#: composes rows flush to ``_body_content_width`` and appends the bar there, so
#: the pad columns CAN hold real right-aligned data (a "10%", a "resets in 5h"
#: countdown, a truncated identity). The forgiveness is therefore CONTENT-AWARE:
#: :meth:`UsagePanel._scrollbar_hit` grabs in the pad band only when those cells
#: are blank in that row, so a near-miss over the common blank tail grabs while
#: a press on a visible character still selects. Raising this only widens the
#: blank-tail target; it never begins stealing content, because the blank-tail
#: gate holds regardless of the pad's width.
SCROLLBAR_GRAB_PAD = 2

#: Track vs thumb glyphs. The track is a hairline so the reserved column reads
#: as a place to aim a drag rather than a right-hand border (the same reason the
#: transcript's idle bar is ``edge``, not ``dim``); the thumb is a solid block a
#: pointer can grab. Colours are semantic and mirror the transcript scrollbar's
#: idle/active intent: ``edge`` track, ``muted`` idle thumb, ``accent`` while a
#: drag is in flight. (No hover tint — a hand-painted ``Static`` has no cheap
#: per-cell hover, and the gesture that matters, the grab, is the active state.)
SCROLLBAR_TRACK = "│"
SCROLLBAR_THUMB = "█"

#: Panel geometry. The width cap is a measure, not a fraction of the terminal.
#: Seventy cells was enough for one provider and one identity; four Anthropic
#: logins plus long notes (``extra usage disabled — out of credits``, ``usage
#: unavailable — last known 40m ago``) and ``7 day (Fable)`` labels need more
#: air on a laptop, but a 200-col sheet still reads as a wall. 104 is the
#: laptop measure. The width margin keeps the card off the screen's edge padding.
PANEL_MAX_WIDTH = 104
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

#: How far behind the title's stamp a report must fall before it labels itself
#: stale (see :func:`_account_status_note`). One minute because that is
#: :func:`format_age`'s own resolution — under it the title and the row both
#: render ``just now``, so a note would be contradicting a title that agrees
#: with it.
_STALE_BEHIND_MS = 60_000

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


def _limit_row(  # noqa: ANN001
    limit, columns: _Columns, now_ms: float, degraded: bool = False
) -> Text:
    """One window: mark, label, bar, number, reset countdown.

    ``degraded`` marks a row whose account is serving last-known numbers. The
    mark takes the ``dim`` ramp then, because a healthy-green dot on a
    two-hour-old meter is the block's highest-contrast element saying "fine"
    while the note under the heading says otherwise. The BAR keeps its quota
    tint — the fill still means what it measures; only the confidence in its
    freshness has changed.
    """
    status = limit.effective_status()
    tint = Style(color=_status_color(status))
    mark_style = Style(color=theme_mod.semantic_color("dim")) if degraded else tint
    dim = Style(color=theme_mod.semantic_color("dim"))
    faint = Style(color=theme_mod.semantic_color("faint"))
    # A tier row is subordinate, so it takes the dim ramp even at full: the
    # colour says "this window", the weight says "this window gates everything".
    label_style = dim if limit.tier else Style(color=theme_mod.semantic_color("muted"))

    fraction = limit.amount.fraction()
    row = Text()
    row.append(f"{MARK_KNOWN if fraction is not None else MARK_UNKNOWN} ", style=mark_style)

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


def _account_status_note(  # noqa: ANN001
    report, now_ms: float, header_ms: float | None = None
) -> str:
    """Per-account stale / unavailable copy, or empty when the row is live.

    Age lives in the panel title for the *set*, measured from the newest
    confirmation in it (``OperatorApp._usage_data_fetched_ms``). That makes this
    note load-bearing rather than decorative: it is what stops a block the title
    does not speak for from being read at the title's age.

    So the trigger is "the title does not describe this row", not merely "this
    row has a failure streak". ``header_ms`` is the stamp the title is showing;
    a report materially older than it labels itself whatever its counters say.
    Without that, a row could be hours stale with ``consecutive_failures == 0``
    and ``usage_unavailable`` false and render completely unmarked beside a
    20-second-old sibling — reachable on ordinary paths, since
    ``cached_usage_reports`` reads with ``include_expired=True`` and the
    lease-loser branch in ``_refresh_provider_usage`` returns the stale payload
    verbatim, neither of which touches a counter. ``header_ms`` is optional so
    the pure renderer stays callable without a title (tests, ``_measure_columns``).

    An exhausted 200 (100% weekly) is quota, not this path.
    """
    unavailable = bool(getattr(report, "usage_unavailable", False))
    failures = int(getattr(report, "consecutive_failures", 0) or 0)
    fetched_at = int(getattr(report, "fetched_at", 0) or 0)
    # Older than the title by more than a rendered age step would show. The
    # threshold is the age formatter's own resolution: below a minute
    # `format_age` says `just now` for both stamps, so a note there would
    # contradict a title that is telling the same truth.
    behind_header = bool(
        header_ms is not None and fetched_at and (header_ms - fetched_at) >= _STALE_BEHIND_MS
    )
    if not unavailable and failures <= 0 and not behind_header:
        return ""
    age = ""
    if fetched_at and report.limits:
        age = format_age(max(0.0, now_ms - fetched_at))
    if unavailable:
        if age:
            return f"usage unavailable — last known {age}"
        return "usage unavailable"
    if age:
        return f"last known {age}"
    return "last known"


def _fit_status_note(note: str, width: int) -> str:
    """The status note, shortened rather than clipped when the card is narrow.

    ``usage unavailable — last known 2h ago`` is 37 cells and truncates from the
    right, which drops the AGE — the more actionable half — while keeping the
    label. On a 40-column card the reader is left with
    ``usage unavailable — last known 2h a…``. The short form states both facts in
    20 cells, so it fits where the long one cannot; the long form is preferred
    whenever there is room for it, since it reads as a sentence.
    """
    if cell_len(note) <= width or " — " not in note:
        return note
    label, _, tail = note.partition(" — ")
    # `last known 2h ago` -> `2h ago`; the `·` keeps the two facts separable.
    short = f"{label.replace('usage ', '')} · {tail.replace('last known ', '')}"
    return short if cell_len(short) <= width else note


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
    if getattr(report, "usage_unavailable", False) and not report.limits:
        # Last-known meters already occupy the heading's binding slot —
        # naming unavailable here duplicates the note and clips the
        # window on a 72-col card. With no meters there is no binding
        # to protect, so the heading may say the probe failed.
        row.append("  ·  usage unavailable", style=dim)
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


def build_usage_body(  # noqa: ANN001
    reports, width: int, now_ms: float, header_ms: float | None = None
) -> UsageBody:
    """The scrolling half of the panel: one block per report.

    Column widths are measured over EVERY row rather than per provider, so the
    bars of two providers line up and the whole panel can be read as one table.

    Each block is heading → (notes) → blank → meters. The blank row is what
    makes the heading read as a heading: without it the identity line, the
    account note and the first meter run together as three equal rows, which is
    the "tight" the card was reported for. The blank is also why the returned
    :class:`UsageBody` carries its cut points: those three rows are one unit,
    and a window may not stop part way through them.

    ``header_ms`` is the stamp the title is displaying, passed down so a block
    the title does not speak for can say so (see :func:`_account_status_note`).
    """
    lines: list[Text] = []
    cuts: set[int] = set()
    blocks: list[tuple[int, int]] = []
    if not reports:
        return UsageBody(lines, frozenset({0}))

    columns = _measure_columns(reports, width, now_ms)

    dim = Style(color=theme_mod.semantic_color("dim"))
    # A degraded account's note is the only thing correcting the title for that
    # block, so it is painted in the theme's `warning` (7.09:1 on the card)
    # rather than `dim` (3.43:1 — the same colour as `resets in 4d` and the
    # footer, i.e. the panel's "this is decoration" ramp).
    warning = Style(color=theme_mod.semantic_color("warning"))
    for index, report in enumerate(reports):
        if index:
            lines.append(Text())
        block_start = len(lines)
        header = _provider_header(report, now_ms)
        # Provider/account identifiers are user data, not geometry. A long
        # identity must lose its tail rather than widen or clip the whole card.
        header.truncate(max(1, width), overflow="ellipsis")
        lines.append(header)
        account_note = _account_status_note(report, now_ms, header_ms)
        if account_note:
            # The two-cell indent is part of the budget the note has to fit in.
            account_note = _fit_status_note(account_note, max(1, width - 2))
            note = Text(f"  {account_note}", style=warning)
            note.truncate(max(1, width), overflow="ellipsis")
            lines.append(note)
            # The note qualifies the meters below it, so it survives compaction
            # with them. `_window_rows` keeps only rows whose index is a cut
            # point when a block outruns a short viewport; without this the
            # layout drops `last known 2h ago` while keeping the stale meter it
            # describes, under a title that now reads `1m ago` — measured at 18
            # of 40 swept sizes, every one of them a frame whose only visible
            # numbers were the stale ones.
            cuts.add(len(lines))
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
            lines.append(_limit_row(limit, columns, now_ms, degraded=bool(account_note)))
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
        #: A background refresh is running while cached reports are already on
        #: screen. The body renders the reports (their age stated in the title)
        #: rather than "fetching…": the numbers the user came for are present,
        #: and the fetch only confirms or replaces them.
        self._refreshing = False
        #: The last refresh over these reports came back empty-handed; a note
        #: under the title says so (see :meth:`settle_refresh`).
        self._refresh_failed = False
        self._error = ""
        self._target = ""
        self._fetched_ms: float | None = None
        # A monotonic identity for the network request allowed to paint this
        # surface. Textual messages and workers run on separate queues, so worker
        # cancellation alone cannot close the gap between `close()` and the app
        # receiving `UsageDismissed`.
        self._request_generation = 0
        self._clock: float = 0.0
        # Screen size plus the live bottom-dock ceiling used for the last
        # paint. The dock can grow without resizing this overlay, so Textual
        # emits no resize event for the panel itself.
        self._layout_shown: tuple[int, int, int] | None = None
        #: True between a mouse-down on the scrollbar and its release. Held so the
        #: thumb can light to ``accent`` while a drag is in flight (its only
        #: "active" affordance) and so a move only scrolls when it CONTINUES a
        #: grab — a bare hover over the column must not drag the report.
        self._dragging = False
        #: Where on the thumb the grab took hold (rows from the thumb's top), so
        #: the thumb tracks the pointer instead of jumping its top to it. A track
        #: click seeds this with half the thumb so the thumb centres on the click.
        self._drag_grab = 0
        self.display = False

    # -- state ---------------------------------------------------------------
    def start_fetch(self, target: str = "") -> int:
        """Show the panel in its loading state and return this request's identity.

        Opened BEFORE the request rather than after it: the fetch crosses the
        network for every logged-in provider, and a command that does nothing
        visible for two seconds reads as a command that did nothing.
        """
        self._request_generation += 1
        self._loading = True
        self._refreshing = False
        self._refresh_failed = False
        self._error = ""
        self._target = target
        # Clear any previous reports so a cold open reads "fetching…" rather
        # than the last session's numbers; the cached-first path repopulates
        # this immediately via `show_cached` when a row is on hand.
        self._reports = []
        self._fetched_ms = None
        self.display = True
        self._repaint()
        return self._request_generation

    def show_cached(
        self, reports, *, now_ms: float, keep_offset: bool = False
    ) -> None:  # noqa: ANN001
        """Paint cached reports immediately while the fetch runs behind them.

        The panel opened with a row already in the shared cache: showing
        "fetching…" would hide an answer that is on hand, so the reports render
        at once with their true age in the title and a ``refreshing…`` mark, and
        the fetch's result replaces them when it lands.

        ``keep_offset`` is the ``r`` path's flag: a refresh re-shows the rows
        the user is already READING, and snapping their scroll position to the
        top would lose their place in a long report for no reason. A fresh open
        starts at the top as before.
        """
        self._reports = list(reports)
        self._loading = False
        self._refreshing = True
        self._refresh_failed = False
        self._error = ""
        self._fetched_ms = now_ms
        if not keep_offset:
            self._offset = 0
        self.display = True
        self._repaint()

    def show_reports(self, reports, *, now_ms: float | None = None) -> None:  # noqa: ANN001
        """Install a finished fetch and paint it, keeping the reader's place.

        A refresh replaces numbers the user is already reading; snapping them
        to the top is jostling, not a new open. A brand-new open is already at
        0 (`start_fetch` clamps against the one-row loading body; `show_cached`
        without ``keep_offset`` still zeros). `_compose_rows` clamps if the new
        body is shorter.
        """
        self._reports = list(reports)
        self._loading = False
        self._refreshing = False
        self._refresh_failed = False
        self._error = ""
        self._fetched_ms = self._now() if now_ms is None else now_ms
        # Keep `_offset`. The finished-fetch path lands over an already-open
        # panel the user may have scrolled while the worker was in flight.
        self.display = True
        self._repaint()

    def show_error(self, message: str) -> None:
        """A failed fetch stays IN the panel, next to the key that retries it."""
        self._loading = False
        self._refreshing = False
        self._refresh_failed = False
        self._error = message
        self.display = True
        self._repaint()

    def settle_refresh(self, *, failed: bool = False) -> None:
        """A background refresh ended with the cached reports still on screen.

        The fetch path serves stale data on failure rather than blanking the
        panel, so the numbers stay and their age in the title already says how
        stale they are. ``failed`` additionally pins a one-row note under the
        title: without it the only signal that an EXPLICIT ``r`` came back
        empty-handed was the ``refreshing…`` mark silently disappearing —
        honest, but illegible as an answer to a key the user just pressed.
        """
        self._refreshing = False
        self._refresh_failed = failed
        self._repaint()

    def accepts_request(self, generation: int) -> bool:
        """Whether a worker result still belongs to the visible request."""
        return self.is_open and generation == self._request_generation

    def close(self) -> None:
        """Hide the panel and forget the scroll position."""
        self._request_generation += 1
        self._loading = False
        self._refreshing = False
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
    def has_reports(self) -> bool:
        """Whether any reports (live or cached) are on screen."""
        return bool(self._reports)

    @property
    def fetched_ms(self) -> float | None:
        """The epoch-ms clock reading the title's age is measured from.

        Exposed so the app's ``r`` handler can re-show the standing reports
        with their ORIGINAL age while the forced fetch runs — re-deriving it
        from the reports would silently reset the age of rows that carry no
        ``fetched_at`` of their own.
        """
        return self._fetched_ms

    @property
    def view_offset(self) -> int:
        return self._offset

    def set_view_offset(self, offset: int) -> None:
        """Restore a scroll position (clamped by the next repaint).

        Exists for the ``r`` path, which re-shows the standing reports after
        ``start_fetch`` clamped the offset against the one-row loading body.
        The finished fetch then keeps that restored place rather than snapping
        to the top.
        """
        self._offset = max(0, int(offset))
        self._repaint()

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

    # -- drag-scroll ---------------------------------------------------------
    # The scrollbar is grabbable. Every handler stops the event for the same
    # reason the wheel does: the card floats over the transcript, so a gesture
    # left to bubble would drag both the quota list and the conversation behind
    # it. Coordinates arrive widget-relative, but this widget (unlike the picker)
    # carries CSS padding, so `get_content_offset_capture` maps the pointer
    # through the widget's LIVE gutter to the content box the composed rows fill.
    # Reading the gutter rather than a padding constant is also what keeps the
    # hit-test correct in the CSS-less test host (gutter 0) and while `-squeezed`
    # (top padding dropped) without either case being special-cased here.
    def on_mouse_down(self, event) -> None:  # noqa: ANN001 - Textual event type
        """Begin a drag if the press landed in the scrollbar column.

        A press on the thumb grabs it where it was touched (so it tracks the
        pointer rather than snapping its top to the cursor); a press on the bare
        track jumps the thumb's centre there and then drags from that point. A
        press anywhere else in the card is left to the base widget.
        """
        hit = self._scrollbar_hit(event)
        if hit is None:
            return
        # Base ``Screen._forward_event`` arms a text selection on this MouseDown
        # BEFORE this handler runs, because the panel is a selectable ``Static``;
        # its base mouse-move handler would then extend that selection in parallel
        # with the drag (the messy highlight, on both a near-miss and an exact
        # hit). A scrollbar grab is not a selection, so clear whatever it armed as
        # the grab takes hold. Guarded because the CSS-less test host and any
        # pre-mount call have no screen to clear.
        if self.is_mounted:
            try:
                self.screen.clear_selection()
            except NoScreen:
                pass
        event.stop()
        row_in_body, thumb_top, thumb_len = hit
        if thumb_top <= row_in_body < thumb_top + thumb_len:
            self._drag_grab = row_in_body - thumb_top
        else:
            # Track click: centre the thumb on the pointer, then drag from there.
            self._drag_grab = thumb_len // 2
            self._apply_thumb_top(row_in_body - self._drag_grab)
        self._dragging = True
        self.capture_mouse()
        self._repaint()

    def on_mouse_move(self, event) -> None:  # noqa: ANN001 - Textual event type
        """While grabbed, map the pointer's body row to a new offset."""
        if not self._dragging:
            return
        event.stop()
        budget = self._body_budget()
        first_row, _ = self._body_region(budget)
        row_in_body = self._content_row(event) - first_row
        self._apply_thumb_top(row_in_body - self._drag_grab)

    def on_mouse_up(self, event) -> None:  # noqa: ANN001 - Textual event type
        """Release the grab; the offset stays where the drag left it."""
        if not self._dragging:
            return
        event.stop()
        self._dragging = False
        self.release_mouse()
        self._repaint()

    def _content_offset(self, event):  # noqa: ANN001, ANN201 - Textual event type
        """Pointer position in the content box (composed-row coordinates).

        ``get_content_offset_capture`` subtracts the widget's live gutter
        (padding here), so ``(0, 0)`` is the title cell regardless of the card's
        padding — which differs between the real stylesheet, the CSS-less test
        host, and the ``-squeezed`` state. Kept as one call so the down/move/hit
        paths cannot drift on how they read coordinates.
        """
        return event.get_content_offset_capture(self)

    def _content_row(self, event) -> int:  # noqa: ANN001 - Textual event type
        return self._content_offset(event).y

    def _scrollbar_hit(self, event) -> tuple[int, int, int] | None:  # noqa: ANN001
        """``(row_in_body, thumb_top, thumb_len)`` if the press is on the bar, or
        within :data:`SCROLLBAR_GRAB_PAD` cells left of it OVER A BLANK TAIL.

        ``None`` when the body is not scrollable, when the press is left of the
        pad band, when the pad cells for that row hold real content, or when it
        is above/below the track. Sharing the region and thumb maths with the
        painter is what keeps the grab target exactly under the glyph the user
        sees.
        """
        body = self._body()
        budget = self._body_budget()
        total = len(body.lines)
        if total <= budget:
            return None
        offset = self._content_offset(event)
        # The bar is the single content column just past the body's composed
        # width (the reserved gutter). Content x, so no padding term is needed.
        bar = self._body_content_width()
        if not (bar - SCROLLBAR_GRAB_PAD <= offset.x <= bar):
            return None
        first_row, count = self._body_region(budget)
        row_in_body = offset.y - first_row
        if not 0 <= row_in_body < count:
            return None
        # The exact bar column always grabs. A near-miss in the pad band grabs
        # ONLY when the cells from the press up to the bar are blank in THIS row.
        # Unlike the transcript (whose rows leave empty right padding the pad can
        # safely eat), this panel composes rows flush to `bar` and appends the
        # bar at that column, so the pad columns (`bar-1`, `bar-2`) can hold real
        # right-aligned data — a meter's "10%", a "resets in 5h" countdown, a
        # truncated identity. Grabbing over a visible character would steal a
        # legitimate click/selection, so forgiveness is gated on a blank tail:
        # it kicks in on the common blank-tail rows and never over a glyph.
        if offset.x < bar and not self._pad_tail_blank(row_in_body, offset.x, bar, budget):
            return None
        thumb_top, thumb_len = self._scrollbar_thumb(total, budget)
        return row_in_body, thumb_top, thumb_len

    def _pad_tail_blank(self, row_in_body: int, x: int, bar: int, budget: int) -> bool:
        """Whether composed columns ``[x, bar)`` are all whitespace in this row.

        Reads the same composed viewport row the painter overlays the bar onto
        (``_window_rows``), padded conceptually to ``bar`` exactly as
        :meth:`_paint_scrollbar` pads it — a line shorter than ``x`` therefore
        reads as an all-blank tail. The give-back blank rows below a short block
        (``row_in_body >= len(window)``) carry no data, so they are blank by
        construction. Indexing is by code point rather than cell because the
        panel's rows are single-width text (labels, numbers, bar glyphs); the
        painter and ``render_lines_for_test`` treat the two as equivalent here.
        """
        window, _ = self._window_rows(self._body(), budget)
        if row_in_body >= len(window):
            return True
        return window[row_in_body].plain[x:bar].strip() == ""

    def _apply_thumb_top(self, thumb_top: int) -> None:
        """Move the offset so the thumb's top lands at ``thumb_top`` and repaint."""
        body = self._body()
        budget = self._body_budget()
        target = self._offset_from_thumb_top(thumb_top, len(body.lines), budget)
        self._offset = max(0, min(self._max_offset(), target))
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
    # Shared with the aside card via ``widgets.overlay``: every card on the
    # toast layer has to answer "how wide", "how many rows above the dock" and
    # "where does the host go" identically, and two copies of that arithmetic
    # is how one card ends up centred against the screen and the other against
    # the ground. See that module for why each answer is what it is.
    def _screen_size(self) -> tuple[int, int]:
        return overlay.screen_size(self)

    def panel_width(self) -> int:
        width, _ = self._screen_size()
        return max(PANEL_MIN_WIDTH, min(PANEL_MAX_WIDTH, width - PANEL_WIDTH_MARGIN))

    def _content_width(self) -> int:
        return max(1, self.panel_width() - PANEL_PADDING_CELLS)

    def _body_content_width(self) -> int:
        """Width the SCROLLING rows are composed to — one column narrower.

        The rightmost body column is the scrollbar's, reserved in every state
        (see ``SCROLLBAR_GUTTER_CELLS``). Composing the report to this width
        rather than the full content width is what pins the report's
        right-aligned numbers: they measure the same available cells whether or
        not the bar is currently drawn, so toggling scrollable/not-scrollable
        never slides a column sideways. The chrome rows (title, rule, footer)
        keep the full width — the bar lives only in the body window.
        """
        return max(1, self._content_width() - SCROLLBAR_GUTTER_CELLS)

    def _rows_above_dock(self) -> int:
        return overlay.rows_above_dock(self)

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
        # The failed-refresh note is a CONDITIONAL chrome row (see
        # `_compose_rows`): when it is pinned it must come out of the body
        # budget like the title and rule do, or a full-height card grows one
        # row past the dock exactly when the note appears.
        note = 1 if (self._refresh_failed and self._reports) else 0
        return rows, gutter, max(1, rows - PANEL_HEIGHT_MARGIN - CHROME_ROWS - gutter - note)

    def sync_layout(self, *, force: bool = False) -> None:
        """Repaint when the screen or live dock changed around an open card.

        ``force`` bypasses the guard for the resize path, where the fingerprint
        is read before the dock has finished re-arranging and so compares two
        stale numbers. Same signature as the aside card's, because one caller
        drives both.
        """
        if not self.display or not self.is_mounted:
            return
        width, height = self._screen_size()
        fingerprint = (width, height, self._rows_above_dock())
        if force or fingerprint != self._layout_shown:
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

    # -- scrollbar -----------------------------------------------------------
    # ONE source of truth for "which composed rows are the scrolling viewport",
    # shared by the painter (_paint_scrollbar) and the hit-test (the mouse
    # handlers). Deriving it twice is how a bar ends up drawn one row off from
    # where a drag thinks it is.
    def _note_shown(self) -> bool:
        """Whether the pinned failed-refresh note occupies a chrome row.

        Mirrors the condition in :meth:`_compose_rows`; the note sits between the
        rule and the window, so it shifts where the body region begins.
        """
        return bool(self._refresh_failed and self._reports)

    def _body_region(self, budget: int) -> tuple[int, int]:
        """``(first_row, row_count)`` of the scrolling viewport in composed rows.

        ``first_row`` is the index of the first body row: title (0), rule (1),
        then the optional failed-refresh note. ``row_count`` is the viewport
        height — the budget, NOT ``len(window)``: the track is a fixed-height
        reference the thumb slides inside, so it spans the give-back blanks a
        short block leaves below the window as well as the window itself. The
        spacer row above the meta is deliberately outside this span, so it stays
        blank (the pinned-height tests read it).
        """
        return 2 + (1 if self._note_shown() else 0), budget

    def _scrollbar_thumb(self, total: int, budget: int) -> tuple[int, int]:
        """``(thumb_top, thumb_len)`` inside a ``budget``-tall track.

        Proportional both ways, the standard model: the thumb is the fraction of
        the track that the viewport is of the content (``budget/total``), and its
        top is that fraction of the free travel that the offset is of its range.
        Guards ``total`` (a scrolled body has ``total > budget >= 1``, but the
        division is defended anyway) and a zero travel range (a thumb as tall as
        the track never moves).
        """
        track = max(1, budget)
        thumb = max(1, min(track, round(track * budget / total))) if total > 0 else track
        span = track - thumb
        max_off = self._max_offset()
        top = round(span * self._offset / max_off) if (span > 0 and max_off > 0) else 0
        return max(0, min(span, top)), thumb

    def _offset_from_thumb_top(self, thumb_top: int, total: int, budget: int) -> int:
        """Invert :meth:`_scrollbar_thumb`: a thumb top → the offset that places it.

        The drag's whole job. Clamped to the thumb's travel and scaled back onto
        the offset range, so dragging the thumb to the track's bottom lands on
        ``_max_offset`` exactly and a track with no travel is a no-op.
        """
        _, thumb = self._scrollbar_thumb(total, budget)
        span = max(1, budget) - thumb
        max_off = self._max_offset()
        if span <= 0 or max_off <= 0:
            return 0
        return round(max(0, min(span, thumb_top)) / span * max_off)

    def _paint_scrollbar(self, rows: list[Text], budget: int, total: int) -> None:
        """Overlay the bar on the viewport rows' rightmost (reserved) column.

        Mutates ``rows`` in place: each viewport row is padded to the body width
        the report was composed to (``_body_content_width``) and then the track
        or thumb glyph is appended in the one gutter column beyond it, so the
        report's numbers never move and the bar never widens the card. Only the
        ``budget`` viewport rows are touched — the spacer and the meta below keep
        their full width and their emptiness.
        """
        first_row, count = self._body_region(budget)
        thumb_top, thumb_len = self._scrollbar_thumb(total, budget)
        body_width = self._body_content_width()
        edge = Style(color=theme_mod.semantic_color("edge"))
        # `muted` idle, `accent` while grabbed — the transcript scrollbar's
        # idle/active intent, minus the hover step a hand-painted Static cannot
        # afford per cell. The grab is the state worth signalling.
        thumb_idle = Style(color=theme_mod.semantic_color("muted"))
        thumb_active = Style(color=theme_mod.semantic_color("accent"))
        for i in range(count):
            index = first_row + i
            if index >= len(rows):
                break
            row = rows[index]
            row.pad_right(max(0, body_width - row.cell_len))
            on_thumb = thumb_top <= i < thumb_top + thumb_len
            glyph = SCROLLBAR_THUMB if on_thumb else SCROLLBAR_TRACK
            style = (thumb_active if self._dragging else thumb_idle) if on_thumb else edge
            row.append(glyph, style=style)

    # -- rendering -----------------------------------------------------------
    def _body(self) -> UsageBody:
        dim = Style(color=theme_mod.semantic_color("dim"))
        # The body is one column narrower than the chrome: its rightmost cell is
        # the scrollbar gutter, reserved so the report's numbers do not reflow
        # when the bar appears. Chrome rows (title/rule/footer) span the full
        # width — the bar is painted only over the body window.
        width = self._body_content_width()
        if self._error:
            danger = Style(color=theme_mod.semantic_color("danger"))
            # Error text commonly comes from a provider and is unbounded. The
            # popup is intentionally single-line-per-row: wrap would consume
            # its footer/close receipt, while no truncation clips both.
            return UsageBody(
                [Text(truncate_cells(self._error, width), style=danger)],
                frozenset({1}),
            )
        if self._loading and not self._reports:
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
        return build_usage_body(self._reports, width, self._now(), self._fetched_ms)

    def _stale_account_count(self) -> int:
        """How many blocks the title's age does NOT speak for.

        Same predicate the body uses, so the count and the `last known` notes
        can never disagree about which accounts are stale.
        """
        return sum(
            1
            for report in self._reports
            if _account_status_note(report, self._now(), self._fetched_ms)
        )

    def _title_row(self) -> Text:
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        row = Text()
        row.append("Usage", style=Style(color=theme_mod.semantic_color("fg")))
        if self._target:
            row.append(f"  {self._target}", style=muted)
        if self._fetched_ms is not None and not self._loading and not self._error:
            # A `·` between the age and the refreshing mark so "2m ago" and
            # "refreshing…" read as two facts rather than one running phrase.
            row.append(f"  {format_age(self._now() - self._fetched_ms)}", style=dim)
            # The age is the NEWEST confirmation, so on a mixed set it does not
            # describe every block. A bare age never named its subject; unqualified
            # it reads as "all of this is one minute old", which is the same
            # misreading in the opposite direction from the one this panel was
            # reported for. The count makes the title self-correcting and points
            # at the blocks carrying their own `last known` line.
            stale = self._stale_account_count()
            if stale:
                row.append("  · ", style=faint)
                row.append(f"{stale} stale", style=warning)
        if self._refreshing and not self._error:
            row.append("  · ", style=faint)
            row.append("refreshing…", style=dim)
        # Truncated like every other composed row. The title grew suffixes
        # (target, age, refreshing) that can outrun a 32-cell card, and an
        # untruncated Text WRAPS — an extra visual row `_repaint`'s pinned
        # height never counted, which clipped the footer (and its `r refresh`
        # receipt) off the bottom exactly in the stale/narrow state where the
        # refresh key matters most.
        row.truncate(max(1, self._content_width()), overflow="ellipsis")
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
        # row fits. Close is never dropped.
        #
        # Stats go BEFORE the scroll hint, which only matters when the row is
        # actually scrolled — otherwise `↑↓` was never appended. In that state
        # the two say different kinds of thing: the tally summarises what is on
        # screen, while `↑↓ scroll` is the only affordance telling the reader
        # the rest is reachable at all. Shedding the affordance and keeping the
        # summary leaves a footer reporting a total the user cannot get to.
        # Multi-account reports made this bite: the body grew from 16 rows to
        # 23, so what is hidden at 120x30 is now two whole providers rather
        # than a short tail.
        for disposable in ("stats", "↑↓"):
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
        rows = [self._title_row(), rule]
        if self._refresh_failed and self._reports:
            # One dim row, pinned with the chrome rather than scrolled with the
            # body: it annotates the WHOLE report ("these numbers are what you
            # already had"), and it must stay visible next to the age it
            # qualifies while the user scrolls looking for fresher rows.
            note = Text(
                "refresh failed — showing last known numbers",
                style=Style(color=theme_mod.semantic_color("warning")),
            )
            note.truncate(max(1, self._content_width()), overflow="ellipsis")
            rows.append(note)
        rows.extend(window)
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
        # The bar is the last thing painted, over the reserved rightmost column
        # of the viewport rows, and ONLY when the body actually overflows: an
        # unscrollable report reserves the gutter (so nothing reflows) but shows
        # no bar, matching the `↑↓`/position chrome, which also appear only when
        # `scrolled`. It adds no row and changes no width, so `_repaint`'s pinned
        # height and `_body_budget` are untouched.
        if scrolled:
            self._paint_scrollbar(rows, budget, len(lines))
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
        return overlay.recentre(self, width, height)

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
