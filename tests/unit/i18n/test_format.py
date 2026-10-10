"""`format.py` against the generated probes — the Python/`Intl` drift anchor.

Every table carries golden `probes` (input -> the exact string Node produced).
The loops below re-render ALL of them (plural samples for 8 locales, every
number/percent probe) and compare; the hardcoded goldens are the composed
outputs (dates, times, datetime joins, relative time) copied from the same
`Intl` calls, so a change to the tables or to the formatter has to survive
both the systematic and the spot-check shape.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from local_operator.i18n import format as fmt

DATA = Path(fmt.__file__).parent / "data"

FORMATS = json.loads((DATA / "formats.json").read_text(encoding="utf-8"))
PLURALS = json.loads((DATA / "plural_rules.json").read_text(encoding="utf-8"))

ISOS = (
    "2025-07-19T14:05:09+00:00",
    "2025-01-05T09:00:00+00:00",
    "1999-12-31T23:59:00+00:00",
)


def test_available_locales_are_the_wave_list() -> None:
    # Sorted: the emitter writes keys sorted for reproducibility; the WAVE
    # order that matters for pickers lives in `resolve.SUPPORTED_LOCALES`.
    assert fmt.available_locales() == ("en", "es", "fr", "hi", "ru", "ur", "vi", "zh-CN")


def test_plural_categories_match_every_generated_sample() -> None:
    mismatches = []
    for locale, data in PLURALS["locales"].items():
        for sample, category in data["samples"].items():
            value = float(sample) if "." in sample else int(sample)
            got = fmt.plural_category(locale, value)
            if got != category:
                mismatches.append((locale, sample, category, got))
    assert not mismatches, f"plural rules drifted from Intl: {mismatches[:10]}"


def test_number_probes_match() -> None:
    mismatches = []
    for locale, data in FORMATS["locales"].items():
        for raw, want in data["number"]["probes"].items():
            got = fmt.format_number(float(raw), locale)
            if got != want:
                mismatches.append((locale, raw, want, got))
    assert not mismatches, f"number formatting drifted: {mismatches[:10]}"


def test_percent_probes_match() -> None:
    mismatches = []
    for locale, data in FORMATS["locales"].items():
        for raw, want in data["percent"]["probes"].items():
            got = fmt.format_percent(float(raw), locale)
            if got != want:
                mismatches.append((locale, raw, want, got))
    assert not mismatches, f"percent formatting drifted: {mismatches[:10]}"


def test_dates_and_times_match_goldens() -> None:
    # Copied from `Intl.DateTimeFormat(locale, {dateStyle|timeStyle})` on the
    # same three instants the probe dates use; the systematic comparison lives
    # in the RFC's parity matrix, these pin the composed surfaces.
    goldens = {
        ("en", "date", "medium"): ["Jul 19, 2025", "Jan 5, 2025", "Dec 31, 1999"],
        ("fr", "date", "medium"): ["19 juil. 2025", "5 janv. 2025", "31 déc. 1999"],
        ("zh-CN", "date", "long"): ["2025年7月19日", "2025年1月5日", "1999年12月31日"],
        ("en", "time", "short"): ["2:05 PM", "9:00 AM", "11:59 PM"],
        ("fr", "time", "short"): ["14:05", "09:00", "23:59"],
        ("zh-CN", "time", "medium"): ["14:05:09", "09:00:00", "23:59:00"],
    }
    for (locale, kind, style), wants in goldens.items():
        for want, iso in zip(wants, ISOS):
            when = datetime.fromisoformat(iso)
            got = (
                fmt.format_date(when, locale, style=style)
                if kind == "date"
                else fmt.format_time(when, locale, style=style)
            )
            assert got == want, f"{locale} {kind} {style} {iso}: {got!r} != {want!r}"


def test_datetime_joins_are_per_locale() -> None:
    # The join is measured, not concatenated: vi leads with the time, zh-CN
    # joins with a bare space, en with ", ".
    goldens = {
        "en": ["Jul 19, 2025, 2:05 PM", "Jan 5, 2025, 9:00 AM"],
        "fr": ["19 juil. 2025, 14:05", "5 janv. 2025, 09:00"],
        "zh-CN": ["2025年7月19日 14:05", "2025年1月5日 09:00"],
        "vi": ["14:05 19 thg 7, 2025", "09:00 5 thg 1, 2025"],
        "ru": ["19 июл. 2025 г., 14:05", "5 янв. 2025 г., 09:00"],
    }
    for locale, wants in goldens.items():
        for want, iso in zip(wants, ISOS[:2]):
            got = fmt.format_datetime(datetime.fromisoformat(iso), locale)
            assert got == want, f"{locale} {iso}: {got!r} != {want!r}"


def test_relative_time_goldens() -> None:
    assert fmt.format_relative(-3, "day", "en") == "3 days ago"
    assert fmt.format_relative(3, "day", "en") == "in 3 days"
    assert fmt.format_relative(-3, "day", "ru") == "3 дня назад"
    assert fmt.format_relative(-11, "day", "ru") == "11 дней назад"
    assert fmt.format_relative(-1, "minute", "fr") == "il y a 1 minute"


def test_hi_and_ur_use_latin_digits() -> None:
    # The v1 digits policy (§2.8): Latin digits for hi/ur (what Intl returns
    # by default). Guards against a future `nu` extension sneaking in.
    for locale in ("hi", "ur"):
        text = fmt.format_number(12345678, locale)
        assert all(ch.isascii() for ch in text if ch.isdigit()), text
    assert fmt.format_number(12345678, "hi") == "1,23,45,678"


def test_unknown_locale_falls_back_to_english() -> None:
    assert fmt.format_number(1234, "de") == fmt.format_number(1234, "en")
    assert fmt.format_date(datetime(2025, 7, 19), "xx") == "Jul 19, 2025"
    assert fmt.plural_category("xx", 1) == "one"


def test_duration_formats() -> None:
    assert fmt.format_duration(0, "en") == "0s"
    assert fmt.format_duration(-5, "en") == "0s"
    assert fmt.format_duration(65, "en") == "1m 5s"
    assert fmt.format_duration(3725, "en") == "1h 2m 5s"
    assert fmt.format_duration(90061, "en") == "1d 1h 1m 1s"


def test_byte_sizes() -> None:
    assert fmt.format_bytes(0, "en") == "0 B"
    assert fmt.format_bytes(999, "en") == "999 B"
    assert fmt.format_bytes(1500, "en") == "1.5 KB"
    assert fmt.format_bytes(1234567, "en") == "1.2 MB"
    assert fmt.format_bytes(5_000_000_000, "en") == "5 GB"
    assert fmt.format_bytes(-1, "en") == "0 B"


def test_relative_time_rejects_unknown_units() -> None:
    with pytest.raises(ValueError):
        fmt.format_relative(1, "fortnight", "en")
