"""Locale formatting: numbers, dates, times, relative time, durations, byte sizes.

This module is the Python half of the RFC's "one formatter layer per runtime"
(§2.8). Every locale-specific value it needs — separators, grouping with `hi`'s
3;2 ladder, month/weekday names, date and time patterns, relative-time
templates, and the plural rules — is GENERATED at build time from Node `Intl`
by ``scripts/i18n/emit.mjs`` and committed under ``data/``. Nothing here is
hand-maintained per locale, and the wheel adds no dependency: the stdlib's
``locale`` module is deliberately NOT used (process-global, Unix-only — RFC
§2.3).

The generated tables also carry golden ``probes``: input -> the exact string
Node produced. ``tests/unit/i18n/test_format.py`` compares this module's output
against every probe, which is what turns "our reimplementation of three Intl
algorithms" into something a drift check can hold.

Signatures are deliberately small: every public function takes ``locale`` and
falls back to ``en`` for an unknown locale rather than raising, because
formatting sits on user-facing paths where a missing table must degrade, not
crash. The checker and the parity test are where a missing locale goes red.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_DATA_DIR = Path(__file__).with_name("data")

#: Fallback locale for every function here. `en` is the source locale and the
#: only one shipped in M0; it is also the ledger-independent fallback (§6).
FALLBACK_LOCALE = "en"

#: Maximum fraction digits for a default number — Node's `Intl` default, which
#: the probes pin (`0.12345` -> `0.123`). `integer` skeleton rounds instead.
_DEFAULT_FRACTION_DIGITS = 3

#: Byte-size units, base 1000 (SI). The style guide owns any per-locale revisit
#: (§6); v1 keeps the letters ASCII because they are technical identifiers.
_BYTE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


@lru_cache(maxsize=1)
def _formats() -> dict[str, Any]:
    """The generated ``formats.json``, loaded once per process."""
    return json.loads((_DATA_DIR / "formats.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _plural_rules() -> dict[str, Any]:
    """The generated ``plural_rules.json``, loaded once per process."""
    return json.loads((_DATA_DIR / "plural_rules.json").read_text(encoding="utf-8"))


def _locale_data(locale: str) -> Mapping[str, Any]:
    locales = _formats()["locales"]
    return locales.get(locale, locales[FALLBACK_LOCALE])


def available_locales() -> tuple[str, ...]:
    """Locales the generated tables cover, sorted (the emitter's key order)."""
    return tuple(_formats()["locales"].keys())


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


def _decimal(value: "int | float | Decimal") -> Decimal:
    """``value`` as a Decimal, normalised the way JS numbers are.

    JS has one number type, so ``2.0`` is the integer ``2`` and its plural
    operands report ``v=0``; Python floats keep a ``.0``. Integral floats are
    therefore collapsed here — a rule that checks ``v == 0`` must agree with
    `Intl.PluralRules.select(n)` for the same n, and the generated samples
    (which include fractional values like ``1.5``) are the proof.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float) and value.is_integer():
        return Decimal(int(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal(value)
    return Decimal(str(value))


def _round_half_up(value: Decimal, digits: int) -> Decimal:
    """Round the JS way. Python's ``round`` is banker's; `Intl` is half-up."""
    quantum = Decimal(1).scaleb(-digits)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _group(integer_digits: str, *, group: str, sizes: list[int], min_grouping: int) -> str:
    """Insert ``group`` separators per the generated grouping parameters.

    ``sizes`` is ``[primary, secondary]`` (``hi`` is ``[3, 2]``; everyone else
    ``[3, 3]``), walked from the RIGHT. ``min_grouping`` withholds grouping for
    the smallest magnitudes — it is MEASURED by the emitter, not assumed, so a
    locale whose 1000 renders bare stays bare here too.
    """
    primary = sizes[0]
    if len(integer_digits) <= primary + (min_grouping - 1):
        return integer_digits
    secondary = sizes[1] if len(sizes) > 1 else primary
    parts: list[str] = []
    rest = integer_digits
    first = rest[-primary:]
    parts.append(first)
    rest = rest[:-primary]
    while rest:
        block = rest[-secondary:]
        parts.append(block)
        rest = rest[:-secondary]
    return group.join(reversed(parts))


def format_number(
    value: "int | float | Decimal", locale: str, *, skeleton: str | None = None
) -> str:
    """Format a number: ``12345.67`` -> ``"12,345.67"`` (en), ``"12 345,67"`` (fr).

    ``skeleton`` is the ICU subset this runtime allows:
      - ``None``/``"decimal"`` — default: up to 3 fraction digits (Node's
        default), trailing zeros stripped;
      - ``"integer"`` — rounded half-up to 0 fraction digits;
      - ``"percent"`` — the value is a FRACTION (``0.25`` -> ``"25%"``),
        rounded to an integer like `Intl`'s default.
    """
    if skeleton == "percent":
        return format_percent(value, locale)
    data = _locale_data(locale)["number"]
    decimal = _decimal(value)
    sign = "-" if decimal < 0 else ""
    magnitude = abs(decimal)
    if skeleton == "integer":
        rounded = _round_half_up(magnitude, 0)
    else:
        rounded = _round_half_up(magnitude, _DEFAULT_FRACTION_DIGITS).normalize()
        # `.normalize()` strips trailing zeros but can reintroduce exponent
        # notation (`1E+3`); `format(..., "f")` expands it back to digits.
    text = format(rounded, "f")
    if "." in text:
        integer_part, fraction_part = text.split(".", 1)
    else:
        integer_part, fraction_part = text, ""
    integer_part = _group(
        integer_part,
        group=data["group"],
        sizes=data["groupSizes"],
        min_grouping=data["minimumGroupingDigits"],
    )
    if fraction_part:
        return f"{sign}{integer_part}{data['decimal']}{fraction_part}"
    return f"{sign}{integer_part}"


def format_percent(fraction: "int | float | Decimal", locale: str) -> str:
    """Format a fraction as a percent: ``0.25`` -> ``"25%"`` / ``"25 %"`` (fr).

    Rounding is to the nearest INTEGER, matching `Intl`'s percent default
    (`0.075` -> `"8%"`) — the probes pin it. The sign/space placement comes
    from the emitted pattern.
    """
    data = _locale_data(locale)["percent"]
    value = _decimal(fraction) * 100
    number_text = format_number(_round_half_up(value, 0), locale, skeleton="integer")
    return data["pattern"].replace("{n}", number_text)


# ---------------------------------------------------------------------------
# Dates and times
# ---------------------------------------------------------------------------


def _render_pattern(pattern: str, fields: Mapping[str, str]) -> str:
    """Substitute ``{token}`` placeholders in a generated date/time pattern.

    Patterns come from the emitter as literal text plus ``{token}`` slots
    (``"{mon} {d}, {yyyy}"``), so the separators — including CJK suffixes like
    ``年月日`` — are data, never re-derived here.
    """
    out = pattern
    for token, value in fields.items():
        out = out.replace("{" + token + "}", value)
    return out


def _year_fields(year: int) -> dict[str, str]:
    return {"yyyy": f"{year:04d}", "yy": f"{year % 100:02d}", "y": str(year)}


def _month_fields(dt: "date | datetime", names: Mapping[str, Any]) -> dict[str, str]:
    month = dt.month
    return {
        "m": str(month),
        "mm": f"{month:02d}",
        "mon": names["monthsShort"][month - 1],
        "mon_full": names["monthsWide"][month - 1],
    }


def format_date(when: "date | datetime", locale: str, *, style: str = "medium") -> str:
    """Format a date: ``style`` in ``short | medium | long`` (generated patterns).

    A ``datetime`` is accepted and its date part used — callers that want both
    call :func:`format_datetime`.
    """
    data = _locale_data(locale)["date"]
    fields = {
        **_year_fields(when.year),
        **_month_fields(when, data),
        "d": str(when.day),
        "dd": f"{when.day:02d}",
    }
    pattern = data.get(style)
    if pattern is None:
        raise ValueError(f"unknown date style {style!r} (have: short, medium, long)")
    return _render_pattern(pattern, fields)


def _time_fields(when: "datetime", data: Mapping[str, Any]) -> dict[str, str]:
    """Both hour-cycle spellings plus minutes/seconds; the pattern selects.

    Supplying h/hh AND H/HH in one mapping makes a single helper serve every
    time pattern and the composed datetime pattern whatever token it carries;
    ``ampm`` appears only when the locale HAS periods (a 24h locale emitting
    an ``{ampm}`` token would be a table bug the drift probe rejects).
    """
    hour = when.hour
    display = hour % 12 or 12
    fields = {
        "h": str(display),
        "hh": f"{display:02d}",
        "H": str(hour),
        "HH": f"{hour:02d}",
        "min": f"{when.minute:02d}",
        "ss": f"{when.second:02d}",
    }
    periods = data.get("periods")
    if periods:
        fields["ampm"] = periods["am"] if hour < 12 else periods["pm"]
    return fields


def format_time(when: "datetime", locale: str, *, style: str = "short") -> str:
    """Format a time: ``style`` in ``short | medium``; 12h/24h per the locale.

    The hour cycle and the AM/PM strings are measured (`en` -> ``"4:05 PM"``,
    ``fr`` -> ``"16:05"``, ``hi``/``ur`` -> lowercase ``am``/``pm`` where
    ``Intl`` produces them).
    """
    data = _locale_data(locale)["time"]
    pattern = data.get(style)
    if pattern is None:
        raise ValueError(f"unknown time style {style!r} (have: short, medium)")
    return _render_pattern(pattern, _time_fields(when, data))


def format_datetime(when: "datetime", locale: str) -> str:
    """``medium`` date + ``short`` time in the locale's OWN join pattern.

    The join is measured data, not a concatenation: `vi` puts the time FIRST
    (``"14:05 19 thg 7, 2025"``), `zh-CN` joins with a bare space and no
    comma (``"2025年7月19日 14:05"``), `en` with ``", "``. The pattern comes
    from `Intl` (dateStyle ``medium`` + timeStyle ``short``), so separators
    and field order cannot drift here.
    """
    data = _locale_data(locale)
    fields = {
        **_year_fields(when.year),
        **_month_fields(when, data["date"]),
        "d": str(when.day),
        "dd": f"{when.day:02d}",
        **_time_fields(when, data["time"]),
    }
    return _render_pattern(data["datetime"], fields)


# ---------------------------------------------------------------------------
# Plural categories
# ---------------------------------------------------------------------------
#
# The rule functions below implement CLDR's plural rules for exactly the eight
# wave locales. They are NOT free-standing: ``scripts/i18n/generate.py``
# validates each against every sample emitted from `Intl.PluralRules` (a dense
# 0..200 sweep plus magnitude specials and decimals, per locale), so a rule
# that disagrees with the JS the TS surfaces use fails generation loudly.
# Adding a locale means adding its rule here AND its entry in emit.mjs.
#
# Operands follow CLDR: `n` is the absolute value, `i` its integer part, `v`
# the count of visible fraction digits (per the JS-number normalisation in
# `_decimal`). `e` (compact exponent) never occurs in the emitted samples and
# is not modelled; nothing in the product formats compact notation in v1.


def _operands(value: "int | float | Decimal") -> tuple[Decimal, int, int]:
    decimal = abs(_decimal(value))
    exponent = decimal.as_tuple().exponent
    v = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    i = int(decimal)
    return decimal, i, v


def _plural_en(value) -> str:
    _, i, v = _operands(value)
    return "one" if i == 1 and v == 0 else "other"


def _plural_fr(value) -> str:
    _, i, v = _operands(value)
    if i in (0, 1):
        return "one"
    if v == 0 and i != 0 and i % 1_000_000 == 0:
        return "many"
    return "other"


def _plural_es(value) -> str:
    _, i, v = _operands(value)
    if i == 1 and v == 0:
        return "one"
    if v == 0 and i != 0 and i % 1_000_000 == 0:
        return "many"
    return "other"


def _plural_ru(value) -> str:
    _, i, v = _operands(value)
    if v != 0:
        return "other"
    if i % 10 == 1 and i % 100 != 11:
        return "one"
    if i % 10 in (2, 3, 4) and i % 100 not in (12, 13, 14):
        return "few"
    if i % 10 == 0 or i % 10 in (5, 6, 7, 8, 9) or i % 100 in (11, 12, 13, 14):
        return "many"
    return "other"


def _plural_hi(value) -> str:
    _, i, v = _operands(value)
    if i == 0 or (i == 1 and v == 0):
        return "one"
    return "other"


def _plural_one_other(value) -> str:
    """`ur` and any locale whose only distinction is a bare singular."""
    _, i, v = _operands(value)
    return "one" if i == 1 and v == 0 else "other"


def _plural_other_only(value) -> str:
    """`vi`, `zh-CN` — no plural distinction at all."""
    return "other"


_PLURAL_RULES = {
    "en": _plural_en,
    "fr": _plural_fr,
    "es": _plural_es,
    "ru": _plural_ru,
    "hi": _plural_hi,
    "ur": _plural_one_other,
    "vi": _plural_other_only,
    "zh-CN": _plural_other_only,
}


def plural_category(locale: str, value: "int | float | Decimal") -> str:
    """The CLDR plural category of ``value`` in ``locale`` (``"one"``, ...).

    Unknown locales fall back to the English rule rather than raising, for the
    same reason the formatters do: a user-facing path must degrade. The
    generated samples prove each rule; the catalogue parity check proves every
    message selects a category its locale can produce.
    """
    rule = _PLURAL_RULES.get(locale, _plural_en)
    return rule(value)


def plural_categories(locale: str) -> tuple[str, ...]:
    """The categories ``locale`` can produce, from the generated table."""
    locales = _plural_rules()["locales"]
    data = locales.get(locale, locales[FALLBACK_LOCALE])
    return tuple(data["categories"])


# ---------------------------------------------------------------------------
# Relative time
# ---------------------------------------------------------------------------


def format_relative(value: "int | float", unit: str, locale: str) -> str:
    """``-3, "day"`` -> ``"3 days ago"``; ``+3`` -> ``"in 3 days"`` (en).

    Templates are generated per (locale, unit, category, sign) with the number
    as ``{n}``; the category is selected by the same rules `Intl` validates,
    and the number is formatted by :func:`format_number`. ``numeric: "always"`
    keeps the templates regular — no ``yesterday`` specials in v1 (RFC-style
    guide entry if a wave wants them).
    """
    data = _locale_data(locale)["relative"]
    if unit not in data:
        raise ValueError(f"unknown relative unit {unit!r} (have: {', '.join(sorted(data))})")
    sign_key = "past" if value < 0 else "future"
    templates = data[unit][sign_key]
    category = plural_category(locale, abs(value))
    template = templates.get(category) or templates.get("other")
    if template is None:
        raise ValueError(f"no relative template {locale}/{unit}/{sign_key}/{category}")
    return template.replace("{n}", format_number(abs(value), locale))


# ---------------------------------------------------------------------------
# Durations and byte sizes
# ---------------------------------------------------------------------------


def format_duration(seconds: "int | float", locale: str) -> str:
    """A compact duration: ``3725`` -> ``"1h 2m 5s"`` (digits localized).

    Zero units are omitted; ``0`` renders ``"0s"``. The unit letters stay
    ASCII/technical for v1 (they are identifiers in the style guide's
    kept-English list); the DIGITS localize like every other number.
    """
    total = int(seconds)
    if total <= 0:
        return f"{format_number(0, locale, skeleton='integer')}s"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    for amount, suffix in ((days, "d"), (hours, "h"), (minutes, "m"), (secs, "s")):
        if amount:
            parts.append(f"{format_number(amount, locale, skeleton='integer')}{suffix}")
    return " ".join(parts)


def format_bytes(size: "int | float", locale: str) -> str:
    """A byte size in SI units, base 1000: ``1536`` -> ``"1.5 KB"``.

    One decimal below 10 units (trailing ``.0`` stripped), whole numbers above;
    ``0`` renders ``"0 B"``. The unit letters are ASCII for the same reason as
    durations'. Negative sizes clamp to ``0 B`` — a size is never negative.
    """
    value = float(size)
    if value < 0:
        value = 0.0
    unit_index = 0
    while value >= 1000 and unit_index < len(_BYTE_UNITS) - 1:
        value /= 1000
        unit_index += 1
    if value < 10:
        # `format_number` already strips trailing zeros (`1.0` -> `"1"`,
        # `1.50` -> `"1.5"`), so no post-strip is needed here.
        text = format_number(_round_half_up(Decimal(str(value)), 1), locale)
    else:
        text = format_number(_round_half_up(Decimal(str(value)), 0), locale, skeleton="integer")
    return f"{text} {_BYTE_UNITS[unit_index]}"
