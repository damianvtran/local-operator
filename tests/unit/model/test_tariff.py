"""The time-of-use schedule: which UTC hours cost full price, and by how much.

The defect these cover: DeepSeek bills peak/off-peak, and every DeepSeek figure
this app produced was the PEAK rate — so the ~79% of the week that is off-peak
was overstated by exactly 2x, the model picker included (it only ever showed the
peak number).

Every test here passes an explicit ``moment`` or patches :func:`tariff.now_utc`.
That is not style. A tariff test that reads the wall clock passes on a Wednesday
afternoon and fails on a Saturday, which is worse than no test at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from local_operator.model import tariff
from local_operator.model.registry import deepseek_models

#: 2026-09-14 is a Monday, so ``_mon(...)`` states a time in the weekday window
#: and everything else is derived from it by day arithmetic.
MONDAY = datetime(2026, 9, 14, tzinfo=timezone.utc)


def _mon(hour: int, minute: int = 0) -> datetime:
    return MONDAY.replace(hour=hour, minute=minute)


def _plus_days(days: int, hour: int, minute: int = 0) -> datetime:
    return (MONDAY + timedelta(days=days)).replace(hour=hour, minute=minute)


#: ``(moment, is_peak, why)`` — the whole published rule, edge by edge. The
#: half-open windows are [01:00, 04:00) and [06:00, 10:00), so both ends of each
#: are covered: the closing edge is the one a ``<=`` gets wrong, and the 04:00
#: and 06:00 gap between the two windows is the one a single-range shortcut
#: (``1 <= hour < 10``) gets wrong.
BOUNDARY_MATRIX: list[tuple[datetime, bool, str]] = [
    (_mon(0, 59), False, "Monday 00:59 is before the first window"),
    (_mon(1), True, "Monday 01:00 opens the first window"),
    (_mon(2), True, "Monday 02:00 is inside the first window"),
    (_mon(3, 59), True, "Monday 03:59 is the last minute of the first window"),
    (_mon(4), False, "Monday 04:00 closes the first window"),
    (_mon(5, 59), False, "Monday 05:59 is in the gap between the windows"),
    (_mon(6), True, "Monday 06:00 opens the second window"),
    (_mon(7), True, "Monday 07:00 is inside the second window"),
    (_mon(9, 59), True, "Monday 09:59 is the last minute of the second window"),
    (_mon(10), False, "Monday 10:00 closes the second window"),
    (_mon(12), False, "Monday noon is off-peak"),
    (_mon(23, 59), False, "Monday 23:59 is off-peak"),
    # Weekend: off-peak all day, including inside the weekday windows.
    (_plus_days(5, 2), False, "Saturday 02:00 is off-peak despite the window"),
    (_plus_days(5, 7), False, "Saturday 07:00 is off-peak despite the window"),
    (_plus_days(6, 9, 59), False, "Sunday 09:59 is off-peak"),
    # The end of the working week, where an off-by-one in the weekday test shows
    # up: a Friday 10:00 that is still read as peak means the boundary is wrong
    # on exactly the day most likely to be exercised.
    (_plus_days(4, 9, 59), True, "Friday 09:59 is peak (last peak minute of the week)"),
    (_plus_days(4, 10), False, "Friday 10:00 is off-peak (the week's own boundary)"),
]


@pytest.mark.parametrize(("moment", "expected", "why"), BOUNDARY_MATRIX)
def test_is_peak_matches_the_published_window(moment: datetime, expected: bool, why: str) -> None:
    assert tariff.is_peak(tariff.DEEPSEEK_TOU, moment) is expected, why


@pytest.mark.parametrize(("moment", "expected", "why"), BOUNDARY_MATRIX)
def test_scale_is_the_ratio_the_window_names(moment: datetime, expected: bool, why: str) -> None:
    scale = tariff.scale_at(tariff.DEEPSEEK_TOU, moment)
    assert scale == (1.0 if expected else 0.5), why


def test_a_non_utc_moment_is_converted_not_reinterpreted() -> None:
    """09:00 at +08:00 IS 01:00 UTC, so it is peak.

    Evaluating the hour in the caller's own timezone would call this off-peak
    and halve a real bill — the argument ``web_search/cost.py`` has always made
    for its own copy of the rule.
    """
    shifted = datetime(2026, 9, 14, 9, 0, tzinfo=timezone(timedelta(hours=8)))
    assert shifted.utcoffset() == timedelta(hours=8)
    assert tariff.scale_at(tariff.DEEPSEEK_TOU, shifted) == 1.0
    assert tariff.scale_at(tariff.DEEPSEEK_TOU, shifted.astimezone(timezone.utc)) == 1.0


def test_a_naive_moment_is_read_as_utc() -> None:
    """Naive means UTC here, which is the behaviour the search route already had."""
    assert tariff.scale_at(tariff.DEEPSEEK_TOU, _mon(7).replace(tzinfo=None)) == 1.0
    assert tariff.scale_at(tariff.DEEPSEEK_TOU, _mon(12).replace(tzinfo=None)) == 0.5


def test_the_windows_are_evaluated_in_utc_not_local_time() -> None:
    """A moment whose LOCAL hour is peak but whose UTC hour is not.

    17:00 at -07:00 is 00:00 UTC — off-peak. This is the case a naive
    ``moment.hour`` test gets wrong on this very machine.
    """
    shifted = datetime(2026, 9, 14, 17, 0, tzinfo=timezone(timedelta(hours=-7)))
    assert shifted.astimezone(timezone.utc).hour == 0
    assert tariff.scale_at(tariff.DEEPSEEK_TOU, shifted) == 0.5


@pytest.mark.parametrize("name", ["nope", None, "", 7, object()])
def test_an_unknown_schedule_is_the_base_rate_never_a_discount(name: Any) -> None:
    """An unknown name is a registry bug, and it must fail toward the full price.

    Returning the off-peak half for a row whose ratio nobody published would
    under-bill real spend; returning the base rate is always the defensible
    reading. It must not raise either — a bad schedule name is not a reason to
    take down the pricing path.
    """
    assert tariff.get_tariff(name) is None
    assert tariff.scale_at(name, _mon(7)) == 1.0
    assert tariff.scale_at(name, _mon(12)) == 1.0
    assert tariff.is_peak(name, _mon(12)) is True
    assert tariff.window_label(name, _mon(12)) is None


def test_window_labels_are_the_words_the_price_column_prints() -> None:
    assert tariff.window_label(tariff.DEEPSEEK_TOU, _mon(7)) == "peak"
    assert tariff.window_label(tariff.DEEPSEEK_TOU, _mon(12)) == "off-peak"
    assert tariff.window_label_for(deepseek_models["deepseek-flash"], _mon(12)) == "off-peak"


def test_window_label_for_a_row_without_a_schedule_is_none() -> None:
    """``None`` is what stops a renderer tagging a row it knows nothing about."""
    assert tariff.window_label_for(deepseek_models["deepseek-chat"], _mon(12)) is None
    assert tariff.scale_for(deepseek_models["deepseek-chat"], _mon(12)) == 1.0


def test_the_clock_seam_moves_both_readings_together(monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _mon(7)
    monkeypatch.setattr(tariff, "now_utc", lambda: frozen)
    assert tariff.now_utc() is frozen
    assert tariff.now_ms() == int(frozen.timestamp() * 1000)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, datetime(1970, 1, 1, tzinfo=timezone.utc)),
        (int(datetime(2026, 9, 14, 7, 0, tzinfo=timezone.utc).timestamp() * 1000), _mon(7)),
        (None, None),
        (-1, None),
        ("x", None),
        (True, None),
        (float("inf"), None),
        (float("nan"), None),
        (1e300, None),
        (10**400, None),
    ],
)
def test_moment_from_ms_degrades_instead_of_raising(value: Any, expected: datetime | None) -> None:
    """A malformed stamp is "unknown moment", never an exception.

    ``1e300`` and a 400-digit int are both past what ``datetime`` can represent,
    and both are exactly what a corrupted sidecar or a hostile mapping looks
    like — the pricing path must fall back to the clock, not die.
    """
    assert tariff.moment_from_ms(value) == expected


def test_moment_for_prefers_the_explicit_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tariff, "now_utc", lambda: _mon(12))
    row = deepseek_models["deepseek-flash"]
    usage = {"input_tokens": 1, "at_ms": int(_mon(7).timestamp() * 1000)}
    assert tariff.moment_for(row, usage, _mon(3)) == _mon(3)


def test_moment_for_reads_the_usages_own_stamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """``at_ms`` on a wire usage, ``ts_ms`` on a ledger snapshot, either shape.

    Both are the same fact at different layers, which is why the ledger needs no
    change beyond passing its snapshot: its row is final and priced at the
    call's own moment, never at whenever the row is read.
    """
    monkeypatch.setattr(tariff, "now_utc", lambda: _mon(12))
    row = deepseek_models["deepseek-flash"]
    at = int(_mon(7).timestamp() * 1000)
    assert tariff.moment_for(row, {"at_ms": at}, None) == _mon(7)
    assert tariff.moment_for(row, {"ts_ms": at}, None) == _mon(7)
    assert tariff.moment_for(row, SimpleUsage(at_ms=at), None) == _mon(7)
    assert tariff.moment_for(row, SimpleUsage(ts_ms=at), None) == _mon(7)
    # at_ms wins over ts_ms when a record somehow carries both.
    assert tariff.moment_for(row, {"at_ms": at, "ts_ms": 0}, None) == _mon(7)
    # And the clock is the last resort.
    assert tariff.moment_for(row, {"input_tokens": 5}, None) == _mon(12)
    assert tariff.moment_for(row, None, None) == _mon(12)


@pytest.mark.parametrize("stamp", [None, -1, "x", True, float("nan"), 1e300])
def test_moment_for_falls_back_to_the_clock_on_a_garbage_stamp(
    stamp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tariff, "now_utc", lambda: _mon(12))
    row = deepseek_models["deepseek-flash"]
    assert tariff.moment_for(row, {"at_ms": stamp}, None) == _mon(12)


class SimpleUsage:
    """A usage-shaped object: the duck type ``configure`` promises to accept."""

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)
