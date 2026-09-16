"""Time-of-use (peak/off-peak) tariff schedules for model pricing.

A schedule is named on the registry row that carries the prices it applies to
(``ModelInfo.time_of_use``), so the money path can scale a row's rates without
knowing which provider it came from, and a new tariff is DATA rather than a new
branch in the arithmetic.

WHY a name on the row rather than a rule inside the pricing function: a
``ModelInfo`` is copied (``model_copy``) through resolution, served over HTTP,
cached, and projected from discovery; a callable or a ``datetime`` would not
survive that and a bool could not say WHICH schedule applies. Deriving the
schedule from the provider id inside ``calculate_cost`` would be a second,
divergent statement of the rule — the mistake ``format_price_pair``'s docstring
already argues against at length.

WHY stdlib only: ``web_search.cost`` imports this at module scope and
``providers/clients.py`` stamps ``Usage.at_ms`` from it, so importing it must
not drag pydantic, the model registry or httpx onto those import graphs. That is
the same cheap-import discipline ``model/defaults.py`` documents.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DEEPSEEK_TOU",
    "TARIFFS",
    "TimeOfUseTariff",
    "get_tariff",
    "is_peak",
    "moment_for",
    "moment_from_ms",
    "now_ms",
    "now_utc",
    "scale_at",
    "scale_for",
    "window_label",
    "window_label_for",
]


@dataclass(frozen=True)
class TimeOfUseTariff:
    """A peak/off-peak schedule: which UTC hours cost full price, and the ratio.

    Windows are half-open ``[start, end)`` UTC hours, evaluated on weekdays only
    when ``weekdays_only`` — DeepSeek's published rule (off-peak rates are half
    the peak rates; peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday
    through Friday; every other hour, weekends included, is off-peak).

    The ratio lives here rather than on the row because it is a property of the
    SCHEDULE: a row states the peak list price, and every schedule that describes
    it states how that price moves through the day.
    """

    name: str
    peak_scale: float = 1.0
    off_peak_scale: float = 0.5
    peak_windows_utc: tuple[tuple[int, int], ...] = ()
    weekdays_only: bool = True
    peak_label: str = "peak"
    off_peak_label: str = "off-peak"
    #: Abbreviated forms, for the widths where the long word is paid for out of
    #: something scarcer than it is (design round 1, D4: at a 56-65-cell picker
    #: the 8-cell ``off-peak`` came out of the model id). ``peak`` is already one
    #: four-cell word, so only the off-peak side shortens — and ``off`` is not a
    #: second vocabulary: it is the same word, spelled as far as it fits, beside
    #: the number it qualifies.
    short_peak_label: str = "peak"
    short_off_peak_label: str = "off"

    def is_peak(self, moment: datetime | None = None) -> bool:
        """Whether full price is in force at ``moment`` (``None`` → the clock).

        A naive ``moment`` is read as UTC and an aware one is CONVERTED to UTC
        before the hour is tested. Evaluating the window in local time would
        mis-price every off-hours session by up to a whole window, which is the
        same argument ``web_search.cost`` has always made for its own copy of
        this rule.
        """
        when = _as_utc(moment)
        if self.weekdays_only and when.weekday() >= 5:  # Saturday/Sunday: off-peak all day
            return False
        hour = when.hour
        return any(start <= hour < end for start, end in self.peak_windows_utc)

    def scale_at(self, moment: datetime | None = None) -> float:
        """The multiplier to apply to this row's stored (peak) rates at ``moment``."""
        return self.peak_scale if self.is_peak(moment) else self.off_peak_scale

    def window_label(self, moment: datetime | None = None, *, short: bool = False) -> str:
        """The window in force at ``moment``, spelled for a human.

        The complete word rather than an abbreviation: a price that silently
        halves every few hours needs to say why it moved, and ``peak``/``off-peak``
        is the same lower-case hyphenated vocabulary the price column already
        prints (``free``, ``usage-based``). ``short=True`` is the narrow-width
        form the caller chooses when the long word would cost a model id cells
        it needs (see the class's short-label fields).
        """
        if self.is_peak(moment):
            return self.peak_label if not short else self.short_peak_label
        return self.off_peak_label if not short else self.short_off_peak_label


#: The only schedule shipped today. The name is the value stored on
#: ``ModelInfo.time_of_use``, so it is part of that row's serialized shape:
#: never rename it without migrating rows, and never make it a callable.
DEEPSEEK_TOU = "deepseek-tou"

TARIFFS: dict[str, TimeOfUseTariff] = {
    DEEPSEEK_TOU: TimeOfUseTariff(
        name=DEEPSEEK_TOU,
        peak_windows_utc=((1, 4), (6, 10)),
        weekdays_only=True,
    ),
}


def now_utc() -> datetime:
    """The current UTC time. THE clock seam — tests patch this, never the clock."""
    return datetime.now(timezone.utc)


def now_ms() -> int:
    """The current epoch milliseconds in UTC, for ``Usage.at_ms``.

    Derived from :func:`now_utc` rather than from ``time.time()`` so patching one
    seam moves both: a test that freezes the schedule's clock and then stamps a
    usage must get a stamp inside the window it froze.
    """
    return int(now_utc().timestamp() * 1000)


def moment_from_ms(ms: Any) -> datetime | None:
    """``ms`` (epoch milliseconds, UTC) as an aware datetime, else ``None``.

    ``None`` rather than an exception for anything unusable — a bool, a string, a
    negative, an infinite or absurd magnitude, or a value outside the range
    ``datetime`` can represent. A bad reading must not be able to take down the
    pricing path; the caller falls back to the wall clock, exactly as
    ``_usage_field``/``_usage_cost`` degrade a malformed wire value.

    Floats are accepted because a JSON round trip may hand one back; the value is
    only ever used as a moment, so an integral float is not worth rejecting.
    """
    if isinstance(ms, bool) or not isinstance(ms, (int, float)):
        return None
    try:
        seconds = float(ms) / 1000.0
    except (OverflowError, ValueError):  # an int too large to become a float
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _as_utc(moment: datetime | None) -> datetime:
    """``moment`` as an aware UTC datetime; ``None`` or a non-datetime uses the clock."""
    when = moment if isinstance(moment, datetime) else now_utc()
    if when.tzinfo is None:
        # Naive timestamps are UTC by convention here, which is the behaviour the
        # web-search route already had. The alternative — reading them as local
        # time — would price a naive stamp against the machine's timezone and
        # silently disagree with the same stamp interpreted by DeepSeek.
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def _stamp(usage: Any, name: str) -> Any:
    """One timestamp off a ``Usage`` or an equivalent mapping.

    Duck-typed for the same reason ``configure._usage_field`` is: ``usage``
    arrives either as the wire ``Usage`` model or as a plain mapping rehydrated
    from a serialized child event, and both must price identically.
    """
    if isinstance(usage, Mapping):
        return usage.get(name)
    return getattr(usage, name, None)


def get_tariff(name: Any) -> TimeOfUseTariff | None:
    """The schedule called ``name``, or ``None`` when nothing by that name exists.

    ``None`` is a real answer, not a lookup failure: a row with no schedule (every
    non-DeepSeek row) and a row naming one this build does not ship are both
    "we know nothing about this row's time structure".
    """
    if not isinstance(name, str):
        return None
    return TARIFFS.get(name)


def is_peak(name: Any, moment: datetime | None = None) -> bool:
    """Whether the schedule ``name`` is at full price at ``moment``.

    An unknown or absent schedule answers ``True`` — "the base rate is in force"
    — because that is the same fact :func:`scale_at` reports as ``1.0``, and
    because a schedule-aware caller must never be told "half price" about a row
    whose ratio nobody published.
    """
    schedule = get_tariff(name)
    if schedule is None:
        return True
    return schedule.is_peak(moment)


def scale_at(name: Any, moment: datetime | None = None) -> float:
    """The multiplier for the schedule ``name`` at ``moment``.

    An unknown or absent name scales by ``1.0`` — the stored base rate — so the
    failure runs toward "we charged the published peak price" and never toward an
    invented discount. It cannot raise: an unknown name is a registry bug, not a
    user situation, and the registry guard test is what keeps it unreachable.
    """
    schedule = get_tariff(name)
    if schedule is None:
        return 1.0
    return schedule.scale_at(moment)


def window_label(name: Any, moment: datetime | None = None, *, short: bool = False) -> str | None:
    """The window in force for the schedule ``name``, or ``None`` when there is none.

    ``None`` is what tells a renderer NOT to print a tag: a row whose schedule is
    unknown must not claim ``peak`` or ``off-peak``, because both are statements
    about a ratio we do not have. ``short`` selects the abbreviated spelling for
    a caller with a measured width budget.
    """
    schedule = get_tariff(name)
    if schedule is None:
        return None
    return schedule.window_label(moment, short=short)


def scale_for(model_info: Any, moment: datetime | None = None) -> float:
    """The multiplier for a registry row's schedule at ``moment``.

    Duck-typed (``getattr``) so the display layer and the money layer can share
    one implementation without importing ``ModelInfo``.
    """
    return scale_at(getattr(model_info, "time_of_use", None), moment)


def window_label_for(
    model_info: Any, moment: datetime | None = None, *, short: bool = False
) -> str | None:
    """The window in force for a registry row's schedule, or ``None`` when it has none."""
    return window_label(getattr(model_info, "time_of_use", None), moment, short=short)


def moment_for(model_info: Any, usage: Any, moment: datetime | None = None) -> datetime:
    """The moment a call's rates should be evaluated at, in a fixed order.

    Explicit ``moment`` wins; else the usage's OWN stamp — ``at_ms`` on a wire
    ``Usage``, ``ts_ms`` on an analytics ``CallSnapshot``, read duck-typed off an
    object or a mapping; else :func:`now_utc`.

    WHY the stamp matters: the window is a property of the CALL, not of when
    somebody later read the record. A restored session, an attached receipt or a
    replayed ledger priced at "now" is wrong by up to 2x in either direction, and
    every one of those surfaces has a better answer available — its own timestamp.

    ``model_info`` is accepted rather than merely ignored: every caller has the
    row in hand (it is what the moment is FOR), keeping a single resolution site,
    and a schedule whose clock is not the wall clock would need it without a
    signature change at every call site.
    """
    if moment is not None:
        return moment
    for field in ("at_ms", "ts_ms"):
        parsed = moment_from_ms(_stamp(usage, field))
        if parsed is not None:
            return parsed
    return now_utc()
