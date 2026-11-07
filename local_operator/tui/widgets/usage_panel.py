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

import bisect
import uuid
from dataclasses import dataclass, replace
from typing import Any

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.dom import NoScreen
from textual.message import Message
from textual.widgets import Static

from local_operator.i18n.keys import tui_usage as _x
from local_operator.i18n.text import tr
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import overlay
from local_operator.tui.widgets.tool_card import truncate_cells

_t = tr("tui.usage")

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


def _stale_behind_ms() -> int:
    """How far behind the title's stamp a row must fall to be called stale.

    Derived from the CACHE's freshness contract, not from :func:`format_age`'s
    display resolution. This panel renders cached data: a row lives
    ``USAGE_REPORT_TTL_MS`` (5 min, jittered ±25%) and the background warmer
    only refreshes the ACTIVE provider, so an idle provider's row is *supposed*
    to be minutes old — that is the cache working, not an account failing.

    Anchored to the formatter's one minute instead, a completely healthy panel
    reported ``5 stale`` and dimmed five healthy dots on most opens (measured at
    97% of opens for a four-provider user). That both states something false and
    burns the signal: marks the operator sees constantly stop meaning
    "degraded", and the genuinely stuck account this panel exists to surface
    ends up looking like every other block.

    A row is unremarkable until it is older than the freshness the system
    promised it, so the floor is one full TTL plus the ±25% jitter that TTL is
    spread by — a row at the top of its jitter band has not outlived its
    contract. Taken from the cache's own constant rather than a copied literal
    so the two cannot drift; the import is function-local for the same reason
    :func:`format_amount` keeps ``UNIT_LABELS`` off this module's import graph
    (``tui.app`` imports this widget, so importing back through it at module
    level would be a cycle).
    """
    from local_operator.providers.usage_cache import USAGE_REPORT_TTL_MS

    return int(USAGE_REPORT_TTL_MS * 1.25)


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

#: How many accounts the reset band lists AND makes selectable. Five digit keys
#: is the keyboard half of the choice, and the band states the remainder
#: (``…and N more``) rather than growing a scrollable list inside a transient
#: strip -- the remainder row survives the degrade ladder ahead of a candidate
#: row, so the band never counts an account it does not account for.
RESET_CHOOSE_MAX = 5

#: How long the result band stays up before returning to idle. Long enough to
#: read a sentence, short enough that a card the user reopened is not still
#: reporting a reset from a minute ago. Any key dismisses it sooner.
#:
#: Applies to the four DEFINITIVE outcomes only. A ``failed`` result states that
#: a credit's fate is unknown, and it is the only record of a possible spend, so
#: it stays until dismissed or until a refresh succeeds (D8/U9).
RESET_RESULT_SECONDS = 6.0

#: Cells a heading must keep for itself before a reset chip earns its place after
#: it. Below this the heading would be nothing but an ellipsis and a count, so the
#: chip is dropped and the footer's ``x reset`` is the discovery path instead.
RESET_CHIP_MIN_HEADING_CELLS = 12

#: Outcomes after which the redeem id is SPENT and may be forgotten. ``failed``
#: is deliberately absent: the credit's fate is unknown, so the same id must be
#: retried (the server answers ``already_redeemed`` rather than charging twice).
RESET_DEFINITIVE_OUTCOMES = frozenset(
    {"reset", "nothing_to_reset", "no_credit", "already_redeemed"}
)


class UsageDismissed(Message):
    """The panel closed itself (Esc/q). The app restores focus and hides it."""


class UsageRefreshRequested(Message):
    """``r`` was pressed. The app owns the fetch, so it owns the refresh."""


class UsageResetRequested(Message):
    """A confirmed reset: spend one credit on ONE named account.

    Raised only after the confirmation band named the account, because a spent
    credit cannot be refunded. The panel cannot send it itself — the provider
    controller belongs to the app — so it carries everything the request needs:

    - ``provider`` / ``identity``: the account the user confirmed, by the label
      the report carried rather than by registry order.
    - ``credential_ref``: the stable per-credential ref that label cannot carry
      (two stored logins may share an email). The controller resolves by it when
      present, and refuses an ambiguous label rather than guessing.
    - ``redeem_request_id``: the idempotency id minted for this ATTEMPT. The
      app echoes it back through :meth:`UsagePanel.show_reset_result`, and a
      failed attempt keeps it so a retry cannot charge twice.
    """

    def __init__(
        self,
        provider: str,
        identity: str,
        redeem_request_id: str,
        credential_ref: str | None = None,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.identity = identity
        self.redeem_request_id = redeem_request_id
        self.credential_ref = credential_ref


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
    balances: int = 0

    def describe(self) -> str:
        """``6 windows · 1 exhausted · 1 near limit`` — worst state first."""
        if not self.windows:
            return ""
        parts = [_t(_x.windows_count(count=self.windows, shown=str(self.windows)))]
        if self.exhausted:
            parts.append(_t(_x.exhausted(value=str(self.exhausted))))
        if self.warning:
            parts.append(_t(_x.near_limit(value=str(self.warning))))
        if self.balances:
            parts.append(_t(_x.reported_balance(count=self.balances, shown=str(self.balances))))
        if self.unknown:
            parts.append(_t(_x.not_reported(value=str(self.unknown))))
        return " · ".join(parts)


def collect_stats(
    reports,
) -> UsageStats:  # noqa: ANN001 — provider types stay off the TUI import graph
    """Tally every window across every report by its effective status."""
    windows = exhausted = warning = unknown = balances = 0
    for report in reports:
        for limit in report.limits:
            windows += 1
            status = limit.effective_status()
            if status == "exhausted":
                exhausted += 1
            elif status == "warning":
                warning += 1
            elif status == "unknown":
                if format_amount(limit.amount) or getattr(limit, "unlimited", False):
                    balances += 1
                else:
                    unknown += 1
    return UsageStats(windows, exhausted, warning, unknown, balances)


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

#: Prefixes the countdown column may use, longest first. A narrow card buys the
#: label its cells before it spends them on the words around the countdown: the
#: number is the measurement, the prefix is prose that a reader can infer.
_RESET_PREFIXES = (_RESET_PREFIX, "in ", "")


def _format_percent(value: float) -> str:
    """A percent for a row, without rounding a real reading down to nothing.

    Percent rows print integers because the vendors quote ``2.0`` and ``100.0``,
    and a trailing ``.0`` on every row costs cells to say nothing. Kimi's
    monthly window is the counter-example: it arrives as a ratio like ``0.0006``
    — 0.06% — and ``:.0f`` would render that ``0%``, which reads as "nothing
    used" on a window that has started. Anything strictly between 0 and 1
    prints ``<1`` so the number stays honest and narrow; zero and whole
    percentages keep the integer form.
    """
    if 0 < value < 1:
        return "<1"
    return f"{value:.0f}"


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
            text = _format_percent(value)
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
        return f"{_format_percent(amount.used_fraction * 100)}% used"
    return ""


def _display_amount(limit):  # noqa: ANN001, ANN201
    # A blocked paid wallet is still funded. Keep its observed remaining
    # amount visible without altering the backend's consumed/fraction facts.
    if getattr(limit, "paid_fallback", False) and limit.amount.remaining is not None:
        from dataclasses import replace

        return replace(limit.amount, used=None)
    return limit.amount


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
    reset_prefix: str = _RESET_PREFIX


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
    numbers_cells = [
        cell_len(format_amount(_display_amount(limit)) or "not reported") for limit in limits
    ]
    reset_cells = [cell_len(format_countdown(limit.resets_in_ms(now_ms))) for limit in limits]
    # The label truncates rather than pushing the numbers off the row: it is the
    # one column whose content the user can still identify from a prefix.
    capped = min(max(label_cells, default=0), max(12, width // 3))
    numbers = max(numbers_cells, default=0)
    longest_reset = max(reset_cells, default=0)

    def composed(label: int, prefix: str) -> tuple[int, int]:
        """``(countdown cell, total row width)`` for this label and prefix."""
        reset = (len(prefix) + longest_reset) if longest_reset else 0
        fixed = 2 + label + 2 + numbers + (2 + reset if reset else 0)
        return reset, fixed

    # Buy the words around the countdown last: the longest prefix that leaves
    # room for the label wins, and a row wider than its body is never emitted.
    label, reset_prefix, reset = capped, _RESET_PREFIX, 0
    for prefix in _RESET_PREFIXES:
        candidate_reset, fixed = composed(capped, prefix)
        if fixed <= width:
            reset_prefix, reset = prefix, candidate_reset
            break
    else:
        # No prefix fits the capped label. Spend the remaining cells on a
        # shorter label under the shortest prefix; drop the countdown only if
        # even one label cell cannot sit beside it.
        reset_prefix = _RESET_PREFIXES[-1]
        reset, fixed = composed(0, reset_prefix)
        label = max(1, min(capped, width - fixed))
        if fixed + label > width:
            reset_prefix, reset = "", 0
            label = max(1, min(capped, width - (2 + 2 + numbers)))

    fixed = 2 + label + 2 + numbers + (2 + reset if reset else 0)
    bar = max(0, min(BAR_MAX_CELLS, width - fixed - 2))
    return _Columns(
        label=label,
        numbers=numbers,
        reset=reset,
        bar=bar,
        reset_prefix=reset_prefix,
    )


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
    # The mark keys on whether the WINDOW's state is known, not on whether a
    # proportion could be computed. A balance-only row has no denominator by
    # construction, so keying the mark on `fraction` drew every reported
    # balance as a figure we do not have. The BAR still keys on the fraction:
    # the proportion genuinely is unknown there, and dots say so honestly.
    known = limit.effective_status() != "unknown"
    row.append(f"{MARK_KNOWN if known else MARK_UNKNOWN} ", style=mark_style)

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

    numbers = format_amount(_display_amount(limit)) or "not reported"
    row.append("  ")
    row.append(numbers.rjust(columns.numbers), style=dim)

    if columns.reset:
        countdown = format_countdown(limit.resets_in_ms(now_ms))
        # A window with no countdown still pads its cell, so the rows that do
        # have one stay in a column rather than starting wherever their numbers
        # happened to end.
        text = f"{columns.reset_prefix}{countdown}" if countdown else ""
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

    A probe miss after a fresh confirmation misdates nothing, which is why
    ``consecutive_failures > 0`` is deliberately NOT a trigger on its own. The
    reported frame — title ``just now · 5 stale`` over rows reading ``last known
    just now`` — is what pressing ``r`` during a rate-limit storm produces: the
    force round zeroes every streak and re-probes all accounts in one burst, and
    a 429 in that burst lands in ``_mark_account_failure`` with ``failures=1``
    while KEEPING the previous round's seconds-old ``fetched_at``. The numbers
    were confirmed seconds ago; one missed probe does not change what they say
    or when they were confirmed, so the title's age still describes the row and
    a note would contradict it. The row renders clean and is not counted stale.

    A dead grant outranks both: it is the only one of these states with a
    remedy the user can act on, so it names the command instead of an age.
    An exhausted 200 (100% weekly) is quota, not this path.
    """
    unavailable = bool(getattr(report, "usage_unavailable", False))
    fetched_at = int(getattr(report, "fetched_at", 0) or 0)
    if (
        report.provider == "meta-muse"
        and fetched_at
        and not getattr(report, "credential_invalid", False)
    ):
        from local_operator.providers.usage import (
            META_MUSE_ALLOWANCE_NOTE,
            META_MUSE_OBSERVATION_TTL_MS,
            META_MUSE_STALE_NOTE,
        )

        # Unknown Muse rows intentionally have no meters; their vintage is
        # still load-bearing, even beside a fresher provider's title stamp.
        if report.notes in {META_MUSE_ALLOWANCE_NOTE, META_MUSE_STALE_NOTE} or (
            now_ms - fetched_at > META_MUSE_OBSERVATION_TTL_MS
        ):
            return f"last known {format_age(max(0.0, now_ms - fetched_at))}"
    if getattr(report, "credential_invalid", False):
        # The provider is named because the panel lists several accounts and
        # the note sits under one block; `/login kimi` is runnable as printed,
        # which `sign in again` would not be. Kept generic -- any provider's
        # OAuth grant can die this way.
        #
        # `sign-in expired` deliberately does NOT reuse `_credential_state`'s
        # three-word vocabulary (`logged in` / `env key` / `needs login`,
        # app.py). That set answers "is there a usable credential", and
        # `needs login` covers never-logged-in and died-since alike. Here the
        # difference is the whole message: the user HAS logged in, and what
        # they need to know is that it stopped working. Do not "fix" this into
        # `needs login` -- the divergence is the information.
        #
        # The age rides along when there is room, for the same reason this
        # note exists at all: the block keeps rendering last-known meters, and
        # without a vintage a two-day-old 100% sits unmarked under a title
        # reading `just now`. `_fit_status_note` sheds it first on a narrow
        # card, since the command is the actionable half.
        note = f"sign-in expired \u2014 /login {report.provider}"
        if fetched_at and report.limits:
            return f"{note} \u00b7 numbers {format_age(max(0.0, now_ms - fetched_at))}"
        return note
    # Older than the title by more than a rendered age step would show. The
    # threshold is the age formatter's own resolution: below a minute
    # `format_age` says `just now` for both stamps, so a note there would
    # contradict a title that is telling the same truth.
    behind_header = bool(
        header_ms is not None and fetched_at and (header_ms - fetched_at) >= _stale_behind_ms()
    )
    if unavailable:
        # The age rides along only when it would misdate the numbers. A
        # sub-minute-old report renders `just now` — exactly what the title
        # already says — so appending it produced the self-contradictory
        # `usage unavailable — last known just now` this panel was reported
        # for. Past the formatter's minute the vintage is information again.
        age = format_age(max(0.0, now_ms - fetched_at)) if fetched_at and report.limits else ""
        if age and age != "just now":
            return f"usage unavailable — last known {age}"
        return "usage unavailable"
    if behind_header:
        # (header_ms - fetched_at) >= TTL*1.25 (~6.25 min) by construction, so
        # the age can never be `just now` here.
        age = format_age(max(0.0, now_ms - fetched_at)) if fetched_at and report.limits else ""
        if age:
            return f"last known {age}"
        return "last known"
    # A probe miss after a fresh confirmation misdates nothing: the numbers were
    # confirmed seconds ago and one 429/blip does not change what they say or
    # when they were confirmed, so the title's age still describes the row.
    # `consecutive_failures > 0` is deliberately NOT a trigger — see docstring.
    return ""


def _fit_status_note(note: str, width: int) -> str:
    """The status note, shortened rather than clipped when the card is narrow.

    ``usage unavailable — last known 2h ago`` is 37 cells and truncates from the
    right, which drops the AGE — the more actionable half — while keeping the
    label. On a 40-column card the reader is left with
    ``usage unavailable — last known 2h a…``. The short form states both facts in
    20 cells, so it fits where the long one cannot; the long form is preferred
    whenever there is room for it, since it reads as a sentence.

    The dead-grant note descends a further ladder, because its tail is a
    COMMAND and a truncated command is worse than no command: it still looks
    like an instruction and then fails when followed. The age yields first,
    then the verb — ``/login <id>`` is runnable as printed and survives to the
    32-col ``PANEL_MIN_WIDTH`` floor for every id in ``USAGE_PROVIDERS`` but
    the longest (``alibaba-token-plan-oauth`` needs 31 cells against the
    floor's 25 budget). Where even that does not fit, the STATE is kept and
    the command dropped: the id is already on the heading directly above, so
    a bare id repeated the line above it and said nothing about what happened
    (#618 D6). ``sign-in expired`` is 15 cells, fits the floor, and sends the
    reader to ``/login``, which the panel vocabulary already teaches.
    """
    if cell_len(note) <= width:
        return note
    dead_grant = "/login " in note
    if dead_grant:
        # Shed the age first: it qualifies the meters, which are still on
        # screen, while the command is the only actionable thing here.
        note = note.partition(" \u00b7 numbers ")[0]
        if cell_len(note) <= width:
            return note
    if " — " not in note:
        return note
    label, _, tail = note.partition(" — ")
    if dead_grant:
        # `sign-in expired — /login kimi` gains nothing from the generic
        # rewrite below (both halves are already minimal), so go straight to
        # the tail. Past that the label is what survives, NOT the bare id: the
        # heading renders the id whole at every width that reaches this rung,
        # while the state — that the sign-in died — is on no other row.
        if cell_len(tail) <= width:
            return tail
        return label
    # `last known 2h ago` -> `2h ago`; the `·` keeps the two facts separable.
    short = f"{label.replace('usage ', '')} · {tail.replace('last known ', '')}"
    return short if cell_len(short) <= width else note


@dataclass(frozen=True)
class _ResetCandidate:
    """One account that holds a spendable reset credit."""

    provider: str
    identity: str
    count: int
    #: The stable per-credential ref the report carried (``identity_key`` or
    #: ``cred:<row id>``), or ``None`` on a report fetched before the field
    #: existed. It -- not ``identity`` -- is what the spend resolves by, because
    #: ``identity`` is a display label two stored logins can share.
    ref: str | None = None

    @property
    def key(self) -> str:
        """What identifies THIS account: its ref, else the label.

        One spelling for the pending redeem id, the band's re-resolution and a
        chip click, so a receipt can never be filed under an account the request
        was not sent for.
        """
        return self.ref or self.identity


def _resets_available(report) -> int:  # noqa: ANN001
    """Spendable reset credits on this report, or 0 when none may be offered.

    Mirrors the backend's exact-int rule (``type(...) is int``, not
    ``isinstance``): ``bool`` is an int, so a ``true`` wire value would
    otherwise offer a reset that does not exist. ``None`` means "not reported"
    and ``0`` means "none left" — neither is spendable.

    A missing identity reads as 0 as well. The confirmation names the account,
    and the request is resolved BY that name; an account the panel cannot name
    is an account the user cannot confirm, so it must not be offered at all.
    """
    count = getattr(report, "resets_available", None)
    identity = getattr(report, "identity", None)
    if type(count) is not int or count <= 0:
        return 0
    if not isinstance(identity, str) or not identity.strip():
        return 0
    return count


def reset_candidates(reports) -> list[_ResetCandidate]:  # noqa: ANN001
    """Every report that can offer a reset, in BODY order.

    Body order rather than a sorted one: the band's ``1``-``5`` must select the
    account the reader is looking at, and the body is what they are reading.
    """
    candidates = []
    for report in reports:
        count = _resets_available(report)
        if count:
            ref = getattr(report, "credential_ref", None)
            candidates.append(
                _ResetCandidate(
                    report.provider,
                    report.identity,
                    count,
                    ref if isinstance(ref, str) and ref.strip() else None,
                )
            )
    return candidates


def _reset_chip_texts(count: int) -> tuple[str, str]:
    """The heading chip's two spellings: the full badge, then the datum alone.

    ONE derivation for the painter and the click hit-test. A region measured
    from a different spelling than the one painted would sit beside the glyph the
    user pressed, which is worse than no target at all.
    """
    return f"[{_t(_x.resets_badge(count=count))}]", f"[{count}]"


def _with_reset_chip(row: Text, count: int, width: int | None) -> Text:
    """``row`` with the reset chip appended WHOLE, or unchanged if it cannot fit.

    The chip is a click target as well as a datum (see
    :meth:`UsagePanel._chip_hits`), so a partial chip is worse than none: ``[2…``
    reads as a rendering fault and cannot be hit. The heading's own text yields
    the cells instead -- its identity is ellipsised so the chip keeps its room --
    and past that the noun is shed (``[2]``) rather than the number, because the
    number is the datum and the noun only labels it.

    ``width`` is the body's composed width. ``None`` means "let the caller
    truncate" (the pre-existing behaviour) and is only used by callers that pass
    no width at all.
    """
    long_chip, short_chip = _reset_chip_texts(count)
    accent = Style(color=theme_mod.semantic_color("accent"))
    if width is None:
        row.append("  ")
        row.append(long_chip, style=accent)
        return row
    for chip in (long_chip, short_chip):
        cells = cell_len(chip)
        room = width - 2 - cells
        if room < RESET_CHIP_MIN_HEADING_CELLS:
            continue
        trial = row.copy()
        trial.truncate(room, overflow="ellipsis")
        if trial.cell_len + 2 + cells > width:
            continue
        trial.append("  ")
        trial.append(chip, style=accent)
        return trial
    return row


def _provider_header(report, now_ms: float, width: int | None = None) -> Text:  # noqa: ANN001
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
    if (
        getattr(report, "usage_unavailable", False)
        and not getattr(report, "credential_invalid", False)
        and not report.limits
    ):
        # Last-known meters already occupy the heading's binding slot —
        # naming unavailable here duplicates the note and clips the
        # window on a 72-col card. With no meters there is no binding
        # to protect, so the heading may say the probe failed.
        #
        # A dead grant is excluded even with no meters: its note already
        # states the condition AND the remedy, so repeating a vaguer version
        # of it in the heading spends the same cells saying less.
        row.append(_t(_x.unavailable()), style=dim)
    payable = [limit for limit in report.limits if limit.paid_usable()]
    if payable:
        # Eligible credit replaces included shared quota, but only explicit
        # provider scope can replace a model-family cap.
        from dataclasses import replace

        remaining_gates = [
            limit
            for limit in report.limits
            if not getattr(limit, "paid_fallback", False)
            and limit.tier
            and not any(p.supersedes_tiers for p in payable)
        ]
        binding = binding_limit(replace(report, limits=remaining_gates))
    else:
        binding = binding_limit(report)
    if binding is not None:
        fraction = binding.amount.fraction() or 0.0
        tint = Style(color=_status_color(binding.effective_status()))
        row.append("  ·  ", style=dim)
        row.append(f"{binding.label} {_format_percent(fraction * 100)}%", style=tint)
        countdown = format_countdown(binding.resets_in_ms(now_ms))
        if countdown:
            row.append(_t(_x.resets_in(value=countdown)), style=dim)
    # The one idle-state affordance, and the only reason `x` exists: a chip at
    # the END of the heading, because the heading is where an account's numbers
    # already are and a row of its own would displace every row below it. It is
    # fitted HERE (see `_with_reset_chip`) so the chip is drawn whole or not at
    # all: a click target half-eaten by the row truncator is not a target.
    resets = _resets_available(report)
    if resets:
        row = _with_reset_chip(row, resets, width)
    return row


@dataclass(frozen=True)
class UsageBody:
    """The scrolling half of the panel and its semantic block boundaries.

    ``cuts`` holds row counts at which a window may end. ``blocks`` holds
    ``(start, end)`` ranges for provider groups, allowing a short viewport to
    remove decorative air without separating meters from their provider.

    ``notes`` holds the LINE INDICES of staleness notes. Compaction has to keep
    those ahead of the meters they qualify (see :meth:`UsagePanel._window_rows`),
    and identifying them by index rather than by matching the rendered string
    keeps that decision with the code that knows which row is which — a
    substring test would silently start matching an account whose identity
    happens to contain the copy.

    ``details`` holds the LINE INDICES of limit-detail rows, identified the same
    way and for the same reason. A detail is DECORATION attached to the meter
    row directly above it — strictly lower priority than any meter — so it lives
    in this body but not in the tree the window is chosen against
    (:meth:`skeleton`). ``100.00 USD paid · 20.00 USD granted`` stranded under
    some other provider's meter is worse than not showing the split, and a
    window that could have shown a number instead of the split is worse still.
    """

    lines: list[Text]
    cuts: frozenset[int]
    blocks: tuple[tuple[int, int], ...] = ()
    notes: frozenset[int] = frozenset()
    details: frozenset[int] = frozenset()

    def content_rows(self) -> tuple[int, ...]:
        """Composed rows that are not annotations, in order.

        The row map between this body and :meth:`skeleton`: ``content_rows()[i]``
        is the row showing the ``i``-th detail-free row. Derived rather than
        stored — a second copy of the row order is how a window ends up one row
        off from the data it is scrolling.
        """
        return tuple(row for row in range(len(self.lines)) if row not in self.details)

    def content_at(self, row: int) -> int:
        """Which detail-free position a composed row (or the body's end) sits at."""
        return bisect.bisect_left(self.content_rows(), row)

    def row_at(self, content_index: int) -> int:
        """The composed row showing the ``content_index``-th detail-free row.

        One past the last content row answers the body's end, so a window's tail
        position can be read back through the same map.
        """
        rows = self.content_rows()
        return rows[content_index] if content_index < len(rows) else len(self.lines)

    def skeleton(self) -> UsageBody:
        """This body with its annotations taken out — the tree the window is
        chosen against.

        Windowing in content space is what makes the detail droppable without
        ever stranding it: the rows a window shows are picked over the meters
        alone, and the annotations are painted afterwards out of the budget the
        meters did not need (see :meth:`UsagePanel._window_rows`). Cut points,
        blocks and notes come across so a block-aware window stops in the same
        place it always did.
        """
        if not self.details:
            # The identity is the common case: one provider can carry a split.
            return self
        rows = self.content_rows()
        return UsageBody(
            lines=[self.lines[row] for row in rows],
            cuts=frozenset(bisect.bisect_left(rows, cut) for cut in self.cuts),
            blocks=tuple(
                (bisect.bisect_left(rows, start), bisect.bisect_left(rows, end))
                for start, end in self.blocks
            ),
            notes=frozenset(bisect.bisect_left(rows, note) for note in self.notes),
        )


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
    notes: set[int] = set()
    details: set[int] = set()
    if not reports:
        return UsageBody(lines, frozenset({0}))

    # Usage owns HTTP fetchers as well as these fixed provenance strings.
    # Import only when painting actual reports: importing the panel at first
    # paint must not pull httpx/the provider stack onto every app startup.
    from local_operator.providers.usage import (
        META_MUSE_ALLOWANCE_NOTE,
        META_MUSE_OBSERVATION_TTL_MS,
        META_MUSE_PASSIVE_NOTE,
        META_MUSE_STALE_NOTE,
        META_MUSE_USAGE_NOTES,
    )

    # A panel can remain open past the passive receipt's lifetime. Re-evaluate
    # on paint too, so its clock never ages a verified 0% into apparently-current
    # free capacity. Replace locally; do not mutate shared cached report objects.
    visible_reports = []
    for report in reports:
        if report.provider == "meta-muse" and report.limits:
            if report.notes != META_MUSE_PASSIVE_NOTE:
                report = replace(report, limits=[], notes=META_MUSE_ALLOWANCE_NOTE)
            elif now_ms - report.fetched_at > META_MUSE_OBSERVATION_TTL_MS or any(
                limit.resets_at_ms is not None and limit.resets_at_ms <= now_ms
                for limit in report.limits
            ):
                report = replace(report, limits=[], notes=META_MUSE_STALE_NOTE)
        visible_reports.append(report)
    reports = visible_reports
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
        header = _provider_header(report, now_ms, width)
        # Provider/account identifiers are user data, not geometry. A long
        # identity must lose its tail rather than widen or clip the whole card.
        header.truncate(max(1, width), overflow="ellipsis")
        lines.append(header)
        account_note = _account_status_note(report, now_ms, header_ms)
        paid_text = _t
        # Reuse the backend's normalized eligibility, not wallet funding alone.
        paid = [limit for limit in report.limits if limit.paid_usable()]
        exhausted_plan = any(
            not getattr(limit, "paid_fallback", False)
            and getattr(limit, "shared", False)
            and not limit.tier
            and limit.amount.fraction() is not None
            and limit.amount.fraction() >= 1
            for limit in report.limits
        )
        retained_tiers = any(
            limit.tier and limit.amount.fraction() is not None and limit.amount.fraction() >= 1
            for limit in report.limits
        ) and not any(limit.supersedes_tiers for limit in paid)
        # Paid explanations annotate their meter, never consume the pinned
        # staleness-note budget. In a short viewport a number must win the tie.
        paid_notes: dict[str, str] = {}
        for allowance in report.limits:
            copy = []
            if (
                paid
                and exhausted_plan
                and allowance.shared
                and not allowance.tier
                and not allowance.paid_fallback
            ):
                copy.append(paid_text(_x.paid_continue()))
            if retained_tiers and allowance.tier:
                copy.append(paid_text(_x.paid_family_caps()))
            if allowance.paid_fallback:
                qualifier = (
                    paid_text(_x.paid_blocked())
                    if allowance.available is False
                    else (
                        paid_text(_x.paid_eligibility_unknown())
                        if allowance.available is None
                        else paid_text(_x.paid_unlimited()) if allowance.unlimited else ""
                    )
                )
                if qualifier:
                    copy.append(qualifier)
            paid_notes[allowance.id] = " ".join(copy)
        muse_note = (
            report.notes
            if report.provider == "meta-muse"
            and isinstance(report.notes, str)
            and report.notes in META_MUSE_USAGE_NOTES
            else None
        )
        if muse_note:
            # The title dates the newest provider, not this account. Carry
            # this receipt's own age at every width and for empty-limit rows;
            # passive age is useful BEFORE its five-minute lifetime expires.
            age = format_age(max(0.0, now_ms - report.fetched_at)) if report.fetched_at else ""
            vintage = f" ({age})" if age else ""
            if muse_note == META_MUSE_ALLOWANCE_NOTE:
                muse_note = (
                    f"Stale sign-in allowance{vintage}; current usage unknown; r won't query Muse"
                )
            elif muse_note == META_MUSE_STALE_NOTE:
                muse_note = (
                    f"Stale response allowance{vintage}; current usage unknown; r won't query Muse"
                )
            elif muse_note == META_MUSE_PASSIVE_NOTE:
                muse_note = f"Last response allowance{vintage}; r won't query Muse"
            else:
                muse_note = "Usage unknown; r won't query Muse; next Muse reply updates it"
            if account_note.startswith("last known"):
                account_note = ""  # The provenance row already carries that age.
            if cell_len(muse_note) + 2 > width:
                if report.notes == META_MUSE_PASSIVE_NOTE:
                    muse_note = f"Response{vintage}; r won't query Muse"
                else:
                    muse_note = f"Unknown{vintage}; r won't query Muse"
        if account_note:
            # The two-cell indent is part of the budget the note has to fit in.
            account_note = _fit_status_note(account_note, max(1, width - 2))
            note = Text(f"  {account_note}", style=warning)
            note.truncate(max(1, width), overflow="ellipsis")
            lines.append(note)
            # The note qualifies the meters below it, so it survives
            # compaction with them: a cut point makes it eligible for a short
            # viewport, and `notes` makes `_window_rows` keep it AHEAD of the
            # meters rather than dropping it as the first row of the tail slice.
            # Both are needed — with only the cut point, the layout still
            # rendered `last known 2h ago`'s meter under a title reading `1m
            # ago` with nothing correcting it.
            cuts.add(len(lines))
            notes.add(len(lines) - 1)
        if report.notes:
            # Muse's source/unknown explanation must travel with its meters
            # just like an age note. Generic endpoint notes remain decoration;
            # this does not retune another provider or the theme's dim palette.
            note = Text(
                f"  {muse_note or report.notes}",
                style=Style(color=theme_mod.semantic_color("fg")) if muse_note else dim,
            )
            note.truncate(max(1, width), overflow="ellipsis")
            lines.append(note)
            if muse_note:
                cuts.add(len(lines))
                notes.add(len(lines) - 1)
        if not report.limits:
            # Unknown quota is not evidence that the account has no limits.
            # The retained Muse qualifier already states the unknown outcome.
            if not muse_note:
                lines.append(Text(_t(_x.no_windows()), style=dim))
            cuts.add(len(lines))
            blocks.append((block_start, len(lines)))
            continue
        lines.append(Text())
        for limit in report.limits:
            meter = _limit_row(limit, columns, now_ms, degraded=bool(account_note))
            lines.append(meter)
            detail = getattr(limit, "detail", "")
            paid_note = paid_notes.get(limit.id, "")
            if paid_note:
                # The existing detail projection drops annotations before any
                # meter; primary ink meets contrast unlike the decorative ramp.
                row = Text(
                    f"{TIER_INDENT}{paid_note}",
                    style=Style(color=theme_mod.semantic_color("fg")),
                )
                row.truncate(max(1, width), overflow="ellipsis")
                lines.append(row)
                details.add(len(lines) - 1)
            if detail and not paid_note:
                # Indented under its meter and truncated to the row it
                # ANNOTATES — the meter's own ink, not the card's width. A
                # balance row's right edge is wherever its number ends, which at
                # 60 columns is 35 cells (the bar has collapsed to a single dot)
                # while a split is a fixed 37-cell string; measuring the split
                # against the card let it out-paint the row it annotates and take
                # the block's right edge with it.
                row = Text(f"{TIER_INDENT}{detail}", style=dim)
                row.truncate(max(1, cell_len(meter.plain.rstrip())), overflow="ellipsis")
                lines.append(row)
                details.add(len(lines) - 1)
            # The cut closes the meter's GROUP — it is where a block-aligned
            # window may stop. It does not make the pair indivisible: windows are
            # chosen over the meters, so a budget that cannot hold the split
            # keeps the meter and drops the annotation.
            cuts.add(len(lines))
        blocks.append((block_start, len(lines)))
    return UsageBody(lines, frozenset(cuts), tuple(blocks), frozenset(notes), frozenset(details))


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
        ("x", "reset", "Reset usage"),
        ("1", "select_reset(1)", "Choose 1"),
        ("2", "select_reset(2)", "Choose 2"),
        ("3", "select_reset(3)", "Choose 3"),
        ("4", "select_reset(4)", "Choose 4"),
        ("5", "select_reset(5)", "Choose 5"),
        ("enter", "confirm_reset", "Confirm reset"),
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
        self._muse_clock_signature: tuple[tuple[int, int, bool, bool], ...] | None = None
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
        #: Bumped only when the card CLOSES. An async reset result uses it to
        #: tell "the surface I started in is gone" (do not paint — the user has
        #: left) from "a refresh happened while I waited" (paint: same card,
        #: same question). `_request_generation` cannot answer that, because
        #: `start_fetch` bumps it on every refresh.
        self._surface = 0
        #: Which reset band is up: ``None`` (idle), ``choose``, ``confirm``,
        #: ``busy`` or ``result``. Panel-owned for the same reason the scroll
        #: offset is: it is transient interior state, and the app
        #: only ever hears about it as a `UsageResetRequested`.
        self._reset_state: str | None = None
        #: The candidates the choose band froze when it opened. Frozen rather
        #: than re-derived per paint so a background refresh landing under the
        #: band cannot renumber the accounts between the reader reading `2.`
        #: and pressing `2`.
        self._reset_choices: list[_ResetCandidate] = []
        #: Accounts beyond the five the band can list, stated as a remainder.
        self._reset_overflow = 0
        #: The account the confirmation band names (and the count it spends from).
        self._reset_account: _ResetCandidate | None = None
        #: The result band's sentence, rendered when the outcome arrived.
        self._reset_result = ""
        #: Whether that sentence reports an UNKNOWN outcome (``failed``). Drives
        #: two things that must not be guessed at paint time: the ``warning``
        #: tone, and the fact that only an unknown outcome outlives its timer.
        self._reset_failed = False
        #: The 6-second result timer, held so it can be stopped. A timer armed
        #: on one card would otherwise outlive `close()` and clear a band the
        #: reader opened on the NEXT card inside those six seconds.
        self._reset_timer: Any = None
        #: ``identity -> redeem_request_id`` for every attempt whose outcome is
        #: still unknown. A failed attempt keeps its id so a retry is the same
        #: request (the vendor answers ``already_redeemed`` rather than charging
        #: twice); a definitive outcome clears it. Deliberately NOT cleared by
        #: Esc or by closing the card: the ambiguity outlives the band.
        #:
        #: Keyed by :attr:`_ResetCandidate.key` (the credential ref, else the
        #: label), NOT by the label: two stored logins can share one email, and
        #: one shared key made two distinct accounts look like one repeat attempt.
        self._pending_redeem: dict[str, str] = {}
        #: Click regions for the headings' reset chips, recorded by the LAST
        #: paint: ``(composed_row, start_cell, end_cell, candidate key)``. Read
        #: back by :meth:`_chip_hit` so the pointer hits the glyphs the user can
        #: actually see -- including the truncation that decided the chip's shape.
        self._chip_hits: list[tuple[int, int, int, str]] = []
        #: Content-row indices of the body rows the last window painted, in
        #: painting order (see :meth:`_window_rows`).
        self._window_content_rows: list[int] = []
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
        self._revalidate_band()
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
        self._revalidate_band()
        if self._reset_state == "result" and self._reset_failed:
            # A refresh SUCCEEDED, which is exactly the truth the failure copy
            # said it could not establish: the receipt has done its job and
            # clears itself (D8/U9). It is the only result that outlived its
            # timer for this reason, so it is the only one that clears here.
            self.cancel_reset()
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
        self._surface += 1
        self._loading = False
        self._refreshing = False
        self._offset = 0
        # The band is transient interior state like the offset: a reopened card
        # presents the report, never the confirmation from last time. The
        # pending redeem ids stay (see `__init__`).
        self._reset_state = None
        self._reset_choices = []
        self._reset_overflow = 0
        self._reset_account = None
        self._reset_result = ""
        self._reset_failed = False
        self._chip_hits = []
        self._cancel_reset_timer()
        self.display = False

    @property
    def is_open(self) -> bool:
        return bool(self.display)

    @property
    def generation(self) -> int:
        """The live FETCH identity a worker result must still match.

        Bumped by every :meth:`start_fetch` and by :meth:`close`; used by
        :meth:`accepts_request` so a fetch that lands after the reader has moved
        on cannot paint over the newer one.
        """
        return self._request_generation

    @property
    def surface_generation(self) -> int:
        """The card's own identity: bumped by :meth:`close`, nothing else.

        A reset takes a network round trip and its answer belongs to the card it
        was asked from. A refresh in between changes ``generation`` — a fetch is
        a new request — but not this: the account, the question and the reader
        are still the ones on screen, so the result must still paint.
        """
        return self._surface

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
        """Esc/q: leave the reset band first, then the panel.

        A band is a question the user opened with a key; Esc answers "not now"
        and must not also throw away the report they were reading. Only with no
        band up does Esc mean close.

        Checked against the STATE rather than through :meth:`_reset_swallowed`,
        because ``busy`` swallows every other key: the request is already on the
        wire and leaving the band changes nothing about the spend, while
        refusing left a card whose every key was inert and only quitting the app
        recovered it (R3).
        """
        if self._reset_state is not None:
            self.cancel_reset()
            return
        self.close()
        self.post_message(UsageDismissed())

    def action_refresh(self) -> None:
        """``r``: re-run the fetch behind the numbers on screen.

        In the ``result`` state this DISMISSES and refetches in one press. The
        failure copy names ``r`` as the way to find out what happened to a
        credit, and treating it as a bare dismissal left the user with a
        vanished sentence and the stale numbers that raised the question
        (D3/U3) -- the panel telling them it did something it had not.
        """
        if self._reset_state == "result":
            self.cancel_reset()
            self.post_message(UsageRefreshRequested())
            return
        if self._reset_swallowed():
            return
        self.post_message(UsageRefreshRequested())

    # -- usage resets --------------------------------------------------------
    # The panel owns the STATES (choose → confirm → busy → result) because they
    # are interior to the card; it owns the redeem id too, because idempotency
    # is a property of the attempt, not of the app's fetch pipeline. The app owns
    # the call itself, and reports back through `show_reset_result`.
    def _reset_swallowed(self) -> bool:
        """Whether the band owns this keystroke, leaving the panel action inert.

        ``busy`` ignores every panel action: the request is already on the wire,
        the outcome is the server's to decide, and there is nothing left to
        cancel. Esc is the exception and does not come through here -- see
        :meth:`action_dismiss`.

        ``result`` treats any key as its dismiss, so the key pressed to clear a
        receipt must not also scroll or refresh behind it. ``r`` is the one key
        that acts INSTEAD of being swallowed, because the failure copy names it
        (:meth:`action_refresh`).
        """
        if self._reset_state == "busy":
            return True
        if self._reset_state == "result":
            self.cancel_reset()
            return True
        return False

    def action_reset(self) -> None:
        """``x``: open the band for the accounts that hold a credit.

        No candidates is a no-op rather than a message: the footer only offers
        ``x`` when one exists, so reaching this with none means the reports moved
        under the keypress, and a transient strip explaining its own absence
        would displace rows to say nothing actionable.
        """
        if self._reset_swallowed():
            return
        self._cancel_reset_timer()
        if self._reset_state is not None:
            self.cancel_reset()
            return
        candidates = reset_candidates(self._reports)
        if not candidates:
            return
        if len(candidates) == 1:
            self._ask_reset_confirm(candidates[0])
            return
        self._reset_choices = candidates[:RESET_CHOOSE_MAX]
        self._reset_overflow = max(0, len(candidates) - RESET_CHOOSE_MAX)
        self._reset_state = "choose"
        self._repaint()

    def action_select_reset(self, index: int) -> None:
        """A digit key in the choose band: confirm the account it numbered."""
        if self._reset_swallowed():
            return
        if self._reset_state != "choose":
            return
        # The LISTED prefix, not the frozen five: on a short card the band lists
        # fewer rows, and a digit that named an account the band did not show
        # would spend a credit the user never saw.
        listed, _ = self._reset_choose_plan()
        if 1 <= index <= len(listed):
            self._ask_reset_confirm(listed[index - 1])

    def action_confirm_reset(self) -> None:
        """``enter``: spend the credit the confirmation band named."""
        if self._reset_swallowed():
            return
        if self._reset_state != "confirm" or self._reset_account is None:
            return
        # Re-check that the question is PAINTED, not merely armed: a resize can
        # take the band's rows away between the question and the key that answers
        # it, and a credit is not refundable. Refusing here is the last of the
        # three renderability gates (`action_reset`, `_ask_reset_confirm`, this).
        if not self._reset_band_rows():
            self.cancel_reset()
            return
        account = self._reset_account
        # One id per ATTEMPT, minted here rather than in the app: a retry after
        # an ambiguous failure must repeat the SAME request, or the vendor could
        # charge a second credit. The id is kept until an outcome is definitive,
        # and it is filed under the CREDENTIAL REF when one is known, so two
        # logins sharing an email cannot share (or steal) one idempotency id.
        redeem_id = self._pending_redeem.get(account.key)
        if redeem_id is None:
            redeem_id = str(uuid.uuid4())
            self._pending_redeem[account.key] = redeem_id
        self._reset_failed = False
        self._reset_state = "busy"
        self.post_message(
            UsageResetRequested(account.provider, account.identity, redeem_id, account.ref)
        )
        self._repaint()

    def cancel_reset(self) -> None:
        """Drop the band without closing the card or spending anything.

        ANY state may be cancelled, ``busy`` included. The request is already
        sent, so leaving the band changes nothing about the spend -- and refusing
        left a card where every key was inert, with no way out but quitting the
        app (R3). The pending redeem id is never cleared here: an unknown outcome
        outlives the band that reported it, so a retry is still the same request.
        """
        self._reset_state = None
        self._reset_choices = []
        self._reset_overflow = 0
        self._reset_account = None
        self._reset_result = ""
        self._reset_failed = False
        self._cancel_reset_timer()
        self._repaint()

    @staticmethod
    def _reset_band_floor(count: int) -> int:
        """Rows the band must be able to paint before anything is armed.

        ``confirm`` needs the title AND the consequence -- the account, and what
        spending on it costs. ``choose`` needs the title, one candidate row and
        the remainder row that accounts for the rest. Below that the band states
        less than a user needs to spend safely, so it is not offered at all; the
        keys live in the footer and cost the band no row (see `_band_key_row`).
        """
        return 2 if count <= 1 else 3

    def _ask_reset_confirm(self, candidate: _ResetCandidate) -> None:
        # The same renderability gate for the paths that do not reach
        # `action_reset`: a digit key, a chip click, and a resize landing between
        # the choice and the paint.
        if self._band_room() < 2:
            return
        self._reset_account = candidate
        self._reset_choices = []
        self._reset_overflow = 0
        self._reset_failed = False
        self._reset_state = "confirm"
        self._repaint()

    def show_reset_result(
        self,
        result,
        *,
        identity: str = "",
        credential_ref: str | None = None,
    ) -> None:  # noqa: ANN001
        """Paint the outcome of a reset the app just performed.

        The library never raises, so anything but the four documented outcomes
        (including a ``failed`` the backend used for "transport error, HTTP 500,
        unparseable body") reads as ``failed`` -- and the failure copy deliberately
        does NOT claim the credit survived, because nobody knows that yet. It
        names the account and the key that fetches the truth (``r``, which
        `action_refresh` now actually performs in this state).

        EVERY sentence names the account it is about: the band that asked the
        question is gone by the time this is read, and a receipt that says only
        "2 windows cleared" leaves a two-account report ambiguous about whose
        credit went (D7). ``nothing_to_reset`` is the one exception -- its copy
        states a fact about the card's windows and carries no room for a label.

        ``identity`` / ``credential_ref`` select the pending redeem id to clear.
        Only a definitive outcome clears it: retrying a ``failed`` attempt with
        the same id is the only safe way to find out whether the credit went.
        """
        outcome = getattr(result, "outcome", "failed")
        windows = getattr(result, "windows_reset", 0)
        windows = windows if type(windows) is int and windows >= 0 else 0
        if outcome == "reset":
            text = _t(_x.resets_done(count=windows, identity=identity))
        elif outcome == "nothing_to_reset":
            text = _t(_x.resets_nothing())
        elif outcome == "no_credit":
            text = _t(_x.resets_no_credit(identity=identity))
        elif outcome == "already_redeemed":
            text = _t(_x.resets_already(identity=identity))
        else:
            text = _t(_x.resets_failed(identity=identity))
        if outcome in RESET_DEFINITIVE_OUTCOMES:
            self._pending_redeem.pop(credential_ref or identity, None)
        self._reset_state = "result"
        self._reset_choices = []
        self._reset_overflow = 0
        self._reset_account = None
        self._reset_result = text
        # An UNKNOWN outcome is the one receipt that must outlive the moment: it
        # is the only record that a credit may have gone. It stays until it is
        # dismissed or until a refresh succeeds (see `show_reports`), so it is
        # the only result without a timer (D8/U9).
        self._reset_failed = outcome not in RESET_DEFINITIVE_OUTCOMES
        self._repaint()
        self._cancel_reset_timer()
        if not self._reset_failed:
            self._arm_reset_timer()

    def _arm_reset_timer(self) -> None:
        """Start the result band's self-dismiss, when this result gets one."""
        self._cancel_reset_timer()
        if self.is_mounted:
            self._reset_timer = self.set_timer(RESET_RESULT_SECONDS, self._reset_result_expired)

    def _band_changed(self, identity: str) -> None:
        """Drop an armed band whose account's credits moved, and say so.

        The receipt is deliberate rather than a silent cancel: the user was
        reading a question that no longer exists, and "nothing was spent" is the
        one thing they need to know before deciding whether to retry (U4).
        """
        self._reset_choices = []
        self._reset_overflow = 0
        self._reset_account = None
        self._reset_result = _t(_x.resets_changed(identity=identity))
        self._reset_failed = False
        self._reset_state = "result"
        self._arm_reset_timer()

    def _revalidate_band(self) -> None:
        """Re-resolve an open band against reports that just landed.

        A refresh can change a credit count under a question the user is still
        reading (U4): the band said "Uses 1 of 2 reset credits" while the heading
        beneath it had already lost its chip, and ``enter`` then spent against an
        intent the panel knew was stale. Every armed account is re-resolved by
        ``ref or identity``; one that is gone -- which includes "down to zero",
        since a zero-credit account is no candidate at all -- drops the band for a
        receipt saying nothing was spent, because the number the user was
        answering does not exist any more.

        ``busy`` is skipped on purpose: that request is already on the wire, so
        "nothing was spent" is not a claim this method can make. Its outcome
        paints over the band when it lands.
        """
        state = self._reset_state
        if state not in ("choose", "confirm"):
            return
        candidates = {candidate.key: candidate for candidate in reset_candidates(self._reports)}
        armed = list(self._reset_choices) if state == "choose" else [self._reset_account]
        resolved: list[_ResetCandidate] = []
        for candidate in armed:
            if candidate is None:
                return
            fresh = candidates.get(candidate.key)
            if fresh is None:
                self._band_changed(candidate.identity)
                return
            resolved.append(fresh)
        if state == "choose":
            self._reset_choices = resolved
        else:
            self._reset_account = resolved[0]

    def _cancel_reset_timer(self) -> None:
        timer = self._reset_timer
        self._reset_timer = None
        if timer is not None:
            timer.stop()

    def _reset_result_expired(self) -> None:
        """The result band's own timeout: only ever clears ITS OWN state."""
        self._reset_timer = None
        if self._reset_state == "result":
            self.cancel_reset()

    def action_scroll_rows(self, delta: int) -> None:
        if self._reset_swallowed():
            return
        self._scroll_by(delta)

    def action_scroll_page(self, delta: int) -> None:
        if self._reset_swallowed():
            return
        # A page is what the card is SHOWING, not what it budgeted for: the
        # window stops at a block boundary, so paging by the budget would step
        # over the heading that the boundary held back. Measured over the meters
        # (see :meth:`UsageBody.skeleton`), because a detail row the window never
        # painted must not count as a row the reader is looking at.
        body = self._body()
        start = body.content_at(self._offset)
        shown = self._window_end(body.skeleton(), start, self._body_budget()) - start
        self._scroll_by(delta * max(1, shown))

    def action_scroll_home(self) -> None:
        if self._reset_swallowed():
            return
        self._offset = 0
        self._repaint()

    def action_scroll_end(self) -> None:
        if self._reset_swallowed():
            return
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
        """A press on the scrollbar starts a drag; a press on a reset chip asks.

        A press on the thumb grabs it where it was touched (so it tracks the
        pointer rather than snapping its top to the cursor); a press on the bare
        track jumps the thumb's centre there and then drags from that point.

        The reset chip is checked FIRST because it is a real target rather than a
        blank-tail near-miss: an accent chip in a mouse-live card that ignored the
        pointer was the other half of D1. A press anywhere else is left to the
        base widget.
        """
        chip = self._chip_hit(event)
        if chip is not None:
            # Same reason the drag clears it: the base handler armed a text
            # selection on this MouseDown, and clicking a control is not one.
            if self.is_mounted:
                try:
                    self.screen.clear_selection()
                except NoScreen:
                    pass
            event.stop()
            self._open_reset_for(chip)
            return
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

    def _chip_regions(
        self, body: UsageBody, window: list[Text], painted_rows: list[int], first_row: int
    ) -> list[tuple[int, int, int, str]]:  # noqa: ANN001
        """``(composed_row, start_cell, end_cell, candidate key)`` per painted chip.

        Recorded by the LAST paint and read back by :meth:`_chip_hit`, so the
        target is exactly the glyphs on screen: the same composition decides both
        whether the chip was drawn whole and how much heading it displaced.

        The account comes from the BLOCK the row belongs to -- ``blocks`` is one
        entry per report in body order, and a block's first row is its heading --
        rather than from matching the heading's text, which would start naming an
        account whose identity happens to contain another's.
        """
        regions: list[tuple[int, int, int, str]] = []
        if not self._chip_hits_enabled():
            return regions
        starts = {start: index for index, (start, _end) in enumerate(body.blocks)}
        for offset, content_row in enumerate(painted_rows):
            block = starts.get(content_row)
            if block is None or block >= len(self._reports):
                continue
            report = self._reports[block]
            count = _resets_available(report)
            if not count or not isinstance(report.identity, str):
                continue
            text = window[offset].plain
            for chip in _reset_chip_texts(count):
                if not text.endswith(chip):
                    continue
                start = cell_len(text[: -len(chip)])
                key = getattr(report, "credential_ref", None) or report.identity
                regions.append((first_row + offset, start, start + cell_len(chip), str(key)))
                break
        return regions

    def _chip_hits_enabled(self) -> bool:
        """Whether a chip click could open anything right now.

        The same gate as the keyboard's ``x``: a band already up owns the card's
        interaction, and a card too short to paint a confirmation must not offer a
        target that would spend into an invisible question (R4/U1).
        """
        if self._reset_state is not None:
            return False
        count = len(reset_candidates(self._reports))
        return count > 0 and self._band_room() >= self._reset_band_floor(count)

    def _chip_hit(self, event) -> str | None:  # noqa: ANN001 - Textual event type
        """The candidate key whose heading chip covers the press, or ``None``."""
        if not self._chip_hits:
            return None
        offset = self._content_offset(event)
        for row, start, end, key in self._chip_hits:
            if offset.y == row and start <= offset.x < end:
                return key
        return None

    def _open_reset_for(self, key: str) -> None:
        """A click on an account's chip asks about THAT account, directly.

        The keyboard's ``x`` still opens the chooser because it names no account;
        a click already named one, so repeating the list would ask the user to
        choose again what they just pointed at (D1).
        """
        candidate = next(
            (candidate for candidate in reset_candidates(self._reports) if candidate.key == key),
            None,
        )
        if candidate is None:
            return
        self._cancel_reset_timer()
        self._ask_reset_confirm(candidate)

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
        total = self._scrolling_rows(body)
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
        target = self._offset_from_thumb_top(thumb_top, self._scrolling_rows(body), budget)
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
        # The reset band is conditional chrome for the same reason, and it is
        # measured by the SAME method `_compose_rows` paints from, so the two
        # cannot disagree about how many rows the band costs.
        band = len(self._reset_band_rows())
        return rows, gutter, max(1, rows - PANEL_HEIGHT_MARGIN - CHROME_ROWS - gutter - note - band)

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
        """The furthest composed row the viewport may start at.

        Measured in content space (see :meth:`UsageBody.skeleton`): annotations
        are not rows the reader scrolled to, and clamping against the composed
        body would let a scroll position rest on a detail whose meter is behind
        it — a split floating under the title with nothing it could belong to.
        """
        body = self._body()
        budget = self._body_budget()
        content = body.skeleton()
        raw = max(0, len(content.lines) - budget)
        # If the raw tail begins inside a provider block, start at its heading.
        # `_window_rows` then removes notes/blank air and keeps the meters that
        # fit, so End never shows anonymous numbers.
        for start, end in content.blocks:
            if start <= raw < end and end - start > budget:
                return body.row_at(start)
        return body.row_at(raw)

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

    def _scrolling_rows(self, body: UsageBody) -> int:
        """How many rows the scroll chrome counts: the METERS, not the annotations.

        ONE source of truth for the three consumers — the painter, the hit test
        and the drag — for the reason the bar's own comment gives: a bar drawn
        from one total and dragged by another sits under the pointer in one place
        and moves from another. Annotations are droppable decoration, so they are
        not part of the list the reader is scrolling.
        """
        return len(body.skeleton().lines)

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
            return UsageBody([Text(_t(_x.fetching()), style=dim)], frozenset({1}))
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

    def _flagged_account_counts(self) -> tuple[int, int]:
        """``(stale, expired)`` — blocks the title's age does not speak for.

        Split because the two carry different instructions. A stale block is
        waiting on the network and will catch up on its own; an expired
        sign-in never will, and saying `1 stale` for it re-frames as a
        freshness problem the exact category error this panel now exists to
        correct. The title is the most-read row on the card, so a user who
        glances only there must not be told to wait.

        Same predicate the body uses, so the counts and the per-block notes
        can never disagree about which accounts are flagged.

        "Stale" here means "not confirmed by the round the title reports", which
        is very nearly but not exactly "older". The title takes the newest
        CONFIRMED stamp, so a failing account whose last-good happens to be more
        recent than that — an old confirmed reading beside a newer unconfirmed
        one — is still counted. The direction is conservative (the title
        under-claims freshness and the block states its own true age, so no
        number is misdated) and the alternative, counting only rows that are
        literally older, would leave an unconfirmed block unmarked.
        """
        stale = expired = 0
        for report in self._reports:
            if not _account_status_note(report, self._now(), self._fetched_ms):
                continue
            if getattr(report, "credential_invalid", False):
                expired += 1
            else:
                stale += 1
        return stale, expired

    def _title_row(self) -> Text:
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        row = Text()
        row.append(_t(_x.title()), style=Style(color=theme_mod.semantic_color("fg")))
        stale = expired = 0
        age = ""
        if self._fetched_ms is not None and not self._loading and not self._error:
            stale, expired = self._flagged_account_counts()
            age = f"  {format_age(self._now() - self._fetched_ms)}"
        # The target is dropped when the row cannot hold both it and the stale
        # count. The title truncates from the RIGHT, so whatever is appended
        # first survives — and on a narrow scoped panel (`/usage anthropic` in a
        # 40-column split) the target pushed `· N stale` off the end, restoring
        # the unmarked stale meter this panel was reported for. The target is
        # the one piece of the title the user already knows: they just typed it,
        # and every block header repeats it. The stale count is the thing they
        # did not know, so it is the target that yields.
        target = f"  {self._target}" if self._target else ""
        if target and (stale or expired):
            wanted = cell_len("Usage") + cell_len(target) + cell_len(age)
            if stale:
                wanted += cell_len(f"  · {stale} stale")
            if expired:
                wanted += cell_len(f"  · {expired} sign-in expired")
            if wanted > self._content_width():
                target = ""
        if target:
            row.append(target, style=muted)
        if age:
            # A `·` between the age and the refreshing mark so "2m ago" and
            # "refreshing…" read as two facts rather than one running phrase.
            row.append(age, style=dim)
            # The age is the NEWEST confirmation, so on a mixed set it does not
            # describe every block. A bare age never named its subject; unqualified
            # it reads as "all of this is one minute old", which is the same
            # misreading in the opposite direction from the one this panel was
            # reported for. The count makes the title self-correcting and points
            # at the blocks carrying their own `last known` line.
            #
            # A mixed set reads `1 sign-in expired · 1 stale`. Both counts are
            # kept rather than folded into one number, because they ask the
            # user for opposite things: act, versus wait. The ACTIONABLE one
            # goes first: the row truncates from the right, and with `stale`
            # leading, a mixed set below 56 cols read `1 stale ·…` — the
            # category error the split exists to correct, restored at exactly
            # the widths where the title is all the reader gets (#618 D7).
            # Same rule as the target above: what the user can do nothing
            # about is what yields. The `wanted` budget sums both clauses, so
            # the order changes nothing there.
            if expired:
                row.append("  · ", style=faint)
                row.append(_t(_x.sign_in_expired(value=str(expired))), style=warning)
            if stale:
                row.append("  · ", style=faint)
                row.append(_t(_x.stale(value=str(stale))), style=warning)
        if self._refreshing and not self._error:
            row.append("  · ", style=faint)
            row.append(_t(_x.refreshing()), style=dim)
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

        While a reset band is up this row is the BAND's key line and nothing else
        (see :meth:`_band_key_row`): the panel's own receipts contradicted it one
        row apart, and ``x`` there cancelled the question instead of opening it.
        """
        if self._reset_state is not None:
            return self._band_key_row()
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        width = self._content_width()
        stats = collect_stats(self._reports).describe() if self._reports else ""
        # Offered only when pressing it could do something: a footer advertising
        # a key that opens nothing is the dead receipt this row's shed order
        # exists to avoid. "Something" includes the card having the rows to show
        # the band at all -- `x` on a card too short for the question is a no-op
        # (`action_reset`), so it must not be advertised (R4).
        candidates = reset_candidates(self._reports)
        resettable = bool(candidates) and (
            self._band_room() >= self._reset_band_floor(len(candidates))
        )
        segments: list[tuple[str, str, str]] = []
        if stats:
            segments.append(("stats", stats, ""))
        for key, what in KEY_HINTS:
            if key == "↑↓" and not scrolled:
                continue
            segments.append((key, key, what))
            if key == "r" and resettable:
                # Immediately after `r`: both are actions on the report itself,
                # and the reset is the rarer one, so it must not push refresh.
                segments.append(("x", "x", _t(_x.resets_hint())))

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
        #
        # `x` goes last of the three, and that is a DECISION rather than an
        # ordering accident (D9/K): without it the reset is still reachable (the
        # chip on the account's heading is the affordance and is clickable, the
        # hint is only its receipt), while a footer with no `r` leaves a failed
        # probe with no stated remedy at all. The tally it displaces at 60 cols
        # is the panel's existing reference, so the "displaces nothing" constraint
        # outranks a rare affordance -- but only after the tally itself has
        # yielded to `↑↓` and `r`, which is the order above.
        for disposable in ("stats", "↑↓", "x"):
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

    # -- reset band ----------------------------------------------------------
    def _band_room(self) -> int:
        """Rows the band may take before the body would fall below one row.

        The same sum :meth:`_fit` solves, with the one-row body floor reserved.
        The band degrades rather than growing the card past ``rows_above_dock``,
        because a row over that ceiling lands on the docked input panel.
        """
        rows = self._rows_above_dock()
        gutter = PANEL_PADDING_ROWS if rows >= SQUEEZE_ROWS else 0
        note = 1 if (self._refresh_failed and self._reports) else 0
        return max(0, rows - PANEL_HEIGHT_MARGIN - CHROME_ROWS - gutter - note - 1)

    def _band_paints(self) -> bool:
        """Whether the armed band can paint the least it must show.

        Asked at the three gates that precede a spend (arming, selecting, and
        confirming) and by :meth:`_repaint`, which cancels a band a resize has
        taken the rows from. One predicate rather than a number repeated at each
        gate, so a band can never be armed by one measurement and painted by
        another.
        """
        return bool(self._reset_band_rows())

    def _band_keys(self, pairs: list[tuple[str, str]]) -> Text:
        """A key receipt line, composed the way :meth:`_hint_row` composes its own."""
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        row = Text()
        for key, what in pairs:
            if row.plain:
                row.append(" · ", style=faint)
            row.append(key, style=dim)
            if what:
                row.append(f" {what}", style=faint)
        return row

    def _band_key_row(self) -> Text:
        """The footer WHILE a band is up: that band's keys, and nothing else.

        One line claims each key (D4/U5). The panel's own receipts go quiet while
        a question is being answered, because they contradicted the band one row
        below it: ``esc close`` next to ``esc cancel``, and an ``x reset`` that
        cancelled the reset question instead of opening one. This costs no height
        -- the footer row is one the card already reserves (``CHROME_ROWS``).

        ``busy`` states no keys: the band's own row carries the in-flight
        sentence, and a second copy here would be exactly the duplication this
        takeover exists to remove. Esc still leaves that band (see
        :meth:`action_dismiss`); it is the universal way out and needs no receipt
        of its own to be found.

        ``result`` advertises ``r`` only for an UNKNOWN outcome, which is the one
        whose copy names it -- and `action_refresh` performs it rather than
        treating it as a dismissal.
        """
        if self._reset_state == "busy":
            return Text()
        if self._reset_state == "result":
            pairs = [("esc", _t(_x.resets_dismiss()))]
            if self._reset_failed:
                pairs.append(("r", _t(_x.resets_recheck())))
        elif self._reset_state == "confirm":
            pairs = [("esc", _t(_x.resets_cancel())), ("enter", _t(_x.resets_accept()))]
        else:
            shown, _ = self._reset_choose_plan()
            last = len(shown)
            # `1-1` reads as a range of one; a lone `1` is what the band means.
            pairs = [
                ("esc", _t(_x.resets_cancel())),
                ("1" if last == 1 else f"1-{last}", _t(_x.resets_choose())),
            ]
        row = self._band_keys(pairs)
        row.truncate(max(1, self._content_width()), overflow="ellipsis")
        return row

    def _reset_title(self, account: _ResetCandidate) -> str:
        """The confirmation's question, naming the account it spends on.

        `` · account N`` appears ONLY when another candidate shares the display
        label: a label is not unique (two stored logins can share an email), and
        a question naming the one thing both accounts have in common cannot be
        answered deliberately. ``N`` is 1-based among same-label candidates in
        body order -- the same order the chooser numbers them in.
        """
        same = [
            candidate
            for candidate in reset_candidates(self._reports)
            if candidate.identity == account.identity
        ]
        if len(same) < 2:
            return _t(_x.resets_confirm_title(identity=account.identity))
        index = next(
            (
                position
                for position, candidate in enumerate(same, start=1)
                if candidate.key == account.key
            ),
            1,
        )
        return _t(_x.resets_confirm_title_account(identity=account.identity, account=str(index)))

    def _reset_choose_plan(self, room: int | None = None) -> tuple[list[_ResetCandidate], int]:
        """``(accounts the band lists, accounts it could not list)``.

        The degrade ladder for ``choose``: candidate rows from the TAIL yield,
        and the remainder row is reserved BEFORE any of them. Inverted on purpose
        (D5/U2) -- shedding the remainder first let a 60x20 band say "5 accounts
        can reset", list one, and account for the other four nowhere, telling the
        reader about accounts it gave them no way to reach or even see. One row
        of honesty outranks one more name.

        The prefix is what the digit keys address -- the same list drives the
        paint, `action_select_reset` and the footer's ``1-N choose``, so a number
        can never name an account the band did not show. The keys live in the
        footer (see `_band_key_row`), so they cost the band no row.
        """
        room = self._band_room() if room is None else room
        choices = self._reset_choices
        total = len(choices) + self._reset_overflow
        fixed = 1  # the title
        if total <= max(0, room - fixed):
            shown_count = total
        else:
            # title + at least one candidate + the remainder row
            shown_count = max(1, min(total, room - fixed - 1))
        shown = choices[:shown_count]
        return shown, total - len(shown)

    def _reset_band_rows(self) -> list[Text]:
        """The band's rows for the current state -- none when idle or unfittable.

        ONE source of truth for the painter and the budget: :meth:`_fit`
        subtracts ``len(...)`` from the body budget, so whatever the band paints
        came out of the body rather than out of the card's surroundings.

        Row counts, settled here and asserted by the height-invariance test:

        - ``choose``: title + one row per listed account + ``…and N more`` when
          some could not be listed.
        - ``confirm``: title + consequence (the account, then its cost).
        - ``busy``: one row.
        - ``result``: one row.

        The key receipt that used to be the band's last row lives in the footer
        now (``_band_key_row``): the card reserves that row anyway, and two key
        lines one above the other contradicted each other about ``esc``.

        NOTHING is returned when the required rows do not fit. A band that
        paints a truncated question is the R4/U1 hole: an irreversible spend
        behind a confirmation the card did not show. :meth:`_repaint` cancels an
        armed band that stops fitting (a resize), ``action_reset`` and
        ``_ask_reset_confirm`` refuse to arm one, and ``action_confirm_reset``
        re-checks immediately before emitting.

        The consequence row and an UNKNOWN outcome are painted in the theme's
        ``warning`` (7.09:1 on the card) rather than ``dim`` (3.43:1 -- the
        panel's "this is decoration" ramp, see the degraded-note comment above).
        The four factual outcomes stay quiet but at ``fg`` weight: they are the
        result of a spend, not chrome.
        """
        state = self._reset_state
        if state is None:
            return []
        room = self._band_room()
        if room <= 0:
            return []
        accent = Style(color=theme_mod.semantic_color("accent"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        factual = Style(color=theme_mod.semantic_color("fg"))
        warning = Style(color=theme_mod.semantic_color("warning"))
        rows: list[Text] = []
        if state == "busy":
            identity = self._reset_account.identity if self._reset_account else ""
            rows = [Text(_t(_x.resets_busy(identity=identity)), style=muted)]
        elif state == "result":
            rows = [Text(self._reset_result, style=warning if self._reset_failed else factual)]
        elif state == "confirm":
            account = self._reset_account
            if account is not None:
                rows = [
                    Text(self._reset_title(account), style=accent),
                    Text(_t(_x.resets_confirm_detail(count=account.count)), style=warning),
                ]
        else:  # choose
            shown, more = self._reset_choose_plan(room)
            if not shown:
                return []
            total = len(self._reset_choices) + self._reset_overflow
            rows = [Text(_t(_x.resets_choose_title(count=total)), style=accent)]
            for index, candidate in enumerate(shown, start=1):
                row = Text()
                row.append(f"{index}. ", style=muted)
                # NOT accent: accent means "you can act here", and the name in
                # this list is not what the digit key acts on -- the number is.
                row.append(candidate.identity, style=factual)
                row.append("  ")
                row.append(f"[{_t(_x.resets_badge(count=candidate.count))}]", style=muted)
                rows.append(row)
            if more:
                rows.append(Text(_t(_x.resets_more(count=str(more))), style=muted))
        if not rows or len(rows) > room:
            return []
        width = max(1, self._content_width())
        for row in rows:
            row.truncate(width, overflow="ellipsis")
        return rows

    def _compose_rows(self) -> list[Text]:
        body = self._body()
        content = body.skeleton()
        lines = body.lines
        budget = self._body_budget()
        self._offset = max(0, min(self._offset, self._max_offset()))
        # "Scrolled" is about the METERS, not the annotations. The card's scroll
        # chrome — the position marker, the `↑↓ scroll` hint, the bar — is a
        # statement about the numbers, and an annotation the window chose to drop
        # is decoration the reader was never promised. Keyed off the composed
        # body instead, a card whose meters ALL fit would raise a marker and a
        # scroll hint that cannot move the moment a split had to be dropped,
        # which is the dead key the hint's own rule exists to avoid.
        scrolled = len(content.lines) > budget
        window, shown_end = self._window_rows(body, budget)
        first_row, _ = self._body_region(budget)
        self._chip_hits = self._chip_regions(body, window, self._window_content_rows, first_row)
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
                _t(_x.refresh_failed()),
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
            position.append(_t(_x.showing_prefix()), style=faint)
            position.append(str(shown_end), style=dim)
            position.append(_t(_x.of_separator()), style=faint)
            position.append(str(len(lines)), style=dim)
            rows.append(position)
        # The reset band sits between the report and the keys: it is a question
        # ABOUT the report, and the keys row that answers it is the band's own,
        # so it never separates the meta rows from each other. Its rows came off
        # the body budget in `_fit`, which is why appending them here leaves the
        # card's total height unchanged.
        rows.extend(self._reset_band_rows())
        rows.append(self._hint_row(scrolled))
        # The bar is the last thing painted, over the reserved rightmost column
        # of the viewport rows, and ONLY when the body actually overflows: an
        # unscrollable report reserves the gutter (so nothing reflows) but shows
        # no bar, matching the `↑↓`/position chrome, which also appear only when
        # `scrolled`. It adds no row and changes no width, so `_repaint`'s pinned
        # height and `_body_budget` are untouched.
        if scrolled:
            self._paint_scrollbar(rows, budget, len(content.lines))
        return rows

    def _window_rows(self, body: UsageBody, budget: int) -> tuple[list[Text], int]:
        """Visible rows plus the source position represented by their tail.

        A provider block taller than the short viewport is compacted
        semantically: heading first, then the last meters that fit. The
        decorative blank yields before identity or quota values.

        A staleness note is kept AHEAD of the meters rather than yielding with
        the other air. A cut point alone only made it *eligible*: the tail a
        window keeps is the end of the block and the note is its first row, so it
        was the first row discarded — which rendered a 169-minute-old meter with
        nothing on the card saying so, the exact frame this panel was reported
        for. The note qualifies the meters, so a budget that can show a meter can
        show the sentence that says the meter is old; dropping a *meter* to keep
        it is the honest trade, because an unlabelled stale number is worse than
        one fewer number. ``_select_rows`` therefore reserves the note's rows
        before it spends any on meters.

        Compaction works on the METERS: the window is chosen over the tree
        without the annotations (:meth:`UsageBody.skeleton`) and the details are
        painted afterwards, out of rows the meters did not need. Two rules
        follow, and the second one is what the first draft of this change got
        wrong — it made a meter and its detail one indivisible unit, so a budget
        that could not hold both kept neither, and the card announced a provider
        with no numbers under it:

        * a detail is never painted without its meter in the same window — not
          as the window's first row, not at any scroll offset, not compacted;
        * the meter always wins a tie. At every budget and every offset this
          window shows at least as many meters as the tree without annotations
          showed, and in fact exactly as many: the annotation is decoration and
          the number is the reason `/usage` was opened.
        """
        content = body.skeleton()
        kept = body.content_rows()
        start = bisect.bisect_left(kept, self._offset)
        selected, content_end = self._select_rows(content, start, budget)
        rows = [kept[index] for index in selected]
        room = budget - len(rows)
        if room > 0:
            # Only rows the meters did not need may go to an annotation, and
            # they go to the ones nearest the tail first: a compacted window is
            # anchored on its last rows and a scrolled one is read from the
            # bottom up, so the split next to the tail is the one being looked
            # for.
            for index in reversed(selected):
                if room <= 0:
                    break
                detail = kept[index] + 1
                if detail in body.details:
                    rows.append(detail)
                    room -= 1
            rows.sort()
        end = kept[content_end] if content_end < len(kept) else len(body.lines)
        # The content indices travel on the widget rather than in the return
        # value: the window has callers that only want the rows, and the chip
        # hit-test is the one consumer that must map a painted row back to the
        # block it belongs to. Recording them here (one derivation, the one the
        # paints use) is what keeps a click target from landing one row off the
        # glyph it belongs to.
        self._window_content_rows = rows
        return [body.lines[row] for row in rows], end

    def _select_rows(self, body: UsageBody, start: int, budget: int) -> tuple[list[int], int]:
        """Which rows the window shows, as indices into a detail-free body.

        Split out from :meth:`_window_rows` because the choice is made over the
        meters alone: an annotation must never be able to cost a window a
        number, so the selection cannot see the details at all.

        A block taller than the short viewport is compacted from its tail: its
        heading, then the staleness notes it has room for, then as many of the
        LAST meters as the budget still holds. Every row it takes is one row, so
        a budget that can hold a meter always shows it — the earlier version
        walked trailing units and broke on the first one too large for the room
        left, which kept nothing at all (a heading with no numbers, one row of
        the allowance unused) the moment a meter carried an annotation.
        """
        for block_start, block_end in body.blocks:
            if start == block_start and block_end - block_start > budget:
                if budget <= 1:
                    return [block_start], block_end
                tails = [
                    index for index in range(block_start + 1, block_end) if index + 1 in body.cuts
                ]
                notes = [index for index in tails if index in body.notes][: budget - 1]
                meters = [index for index in tails if index not in body.notes]
                room = budget - 1 - len(notes)
                kept = meters[max(0, len(meters) - room) :] if room > 0 else []
                return sorted([block_start, *notes, *kept]), block_end
        end = self._window_end(body, start, budget)
        return list(range(start, end)), end

    def _window_end(self, body: UsageBody, start: int, budget: int) -> int:
        """Where a window opened at ``start`` stops: the last row that
        COMPLETES a block.

        A budget that ran out mid-block left a provider heading — or a heading
        and the blank under it — as the last thing on the card, announcing a
        provider and then showing none of its numbers (80x24 landed exactly
        there). The cut moves back to the end of the last whole block instead.

        A budget too small to hold even one block has nothing to give back, so
        it keeps the raw cut: the head of a block reads better than a card with
        an empty body.
        """
        end = min(start + budget, len(body.lines))
        pulled = end
        while pulled > start and pulled not in body.cuts:
            pulled -= 1
        return pulled if pulled > start else end

    def _recentre(self, width: int, height: int) -> None:
        return overlay.recentre(self, width, height)

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        if self._reset_state is not None and not self._band_paints():
            # A resize (or a taller dock) took the band's required rows away while
            # the question was armed. Cancel rather than paint a partial band: a
            # confirmation that shows the account but not its cost is exactly the
            # spend R4/U1 caught. `cancel_reset` clears the state before it
            # repaints, so this cannot recurse.
            self.cancel_reset()
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

    def _refresh_muse_clock(self) -> None:
        """Repaint on a Muse age step/expiry, without fetching or repaint churn."""
        if not self.display:
            return
        from local_operator.providers.usage import META_MUSE_OBSERVATION_TTL_MS

        now = self._now()
        signature = tuple(
            (
                report.fetched_at,
                int(max(0, now - report.fetched_at) // 60_000),
                now - report.fetched_at > META_MUSE_OBSERVATION_TTL_MS,
                any(
                    limit.resets_at_ms is not None and limit.resets_at_ms <= now
                    for limit in report.limits
                ),
            )
            for report in self._reports
            if report.provider == "meta-muse"
        )
        if signature and signature != self._muse_clock_signature:
            self._muse_clock_signature = signature
            self._repaint()

    def on_mount(self) -> None:
        # The wall clock advancing emits no resize/scroll event. Poll only a
        # cheap signature; actual paint happens on minute steps or TTL/reset
        # crossings, and hidden/non-Muse panels do no work or network I/O.
        self.set_interval(1, self._refresh_muse_clock)
        if self.display:
            self._repaint()
