"""The Python mirror of the prelude's ``LO.fmt`` (memo §2.10, round-1 design review D4).

WHY A MIRROR EXISTS. The honesty rule is not only about ``<data>``: a component can declare
honest data and then PRINT a number the data does not support -- the classic case is a
baked-in rounded label ("211.3" next to a 211.25 row). The rendered values are therefore
checked, and checking them means reproducing the prelude's own formatting rules exactly, in
Python, without a browser: the validator recomputes what the component's static text claims
and refuses a mismatch (repair turn, then drop).

WHAT IT MIRRORS (``prelude.src.js``, ``LO.fmt``)::

    fmt(n, o={}){ if(n==null||n==="")return "–";
      const v=+n; if(!isFinite(v))return String(n);
      const d=(String(v).split(".")[1]||"").length;
      const s = o.compact && Math.abs(v)>=1e4
        ? v.toLocaleString(undefined, {notation:"compact", maximumFractionDigits:1})
        : v.toLocaleString(undefined, {maximumFractionDigits:o.digits ?? d});
      return o.unit ? (o.unit==="%" ? s+"%" : s+" "+o.unit) : s }

So the three rules the mirror must get right are: the default precision is the number's OWN
shortest decimal form (never a fixed number of places); an explicit ``digits`` is the only
way fewer decimals appear; and grouping is en-US (commas, three-digit groups).

THE FIVE BEHAVIOURS, spelled out because each is a place a plausible re-implementation
diverges:

* ``None`` and ``""`` render as ``–`` (U+2013), the not-a-value mark.
* a value that is not a finite number renders as its own JS string form (``NaN``,
  ``Infinity``); a non-numeric STRING renders as itself (``+"abc"`` is NaN and
  ``String("abc")`` is ``"abc"``).
* the default precision is ``String(v)``'s decimal count. Python's ``repr`` is the same
  shortest-round-trip algorithm, with one difference that matters: an integral float prints
  as ``"120.0"`` in Python and ``"120"`` in JS, so the mirror strips that ``.0``.
* ``digits`` is ``maximumFractionDigits`` -- it can only REMOVE decimals; it never pads.
* ``compact`` applies only at ``|v| >= 1e4`` and prints one decimal: ``10K``, ``1.2M``.

KNOWN LIMITS, and why they are acceptable here. Exponent spelling differs between the two
languages at the extremes (``1e-07`` vs ``1e-7``) and the compact suffix table is en-US
only; a value in those ranges is outside what a supplement component plots (the evidence
datasets are tool output and tables), and a mismatch there fails CLOSED -- the validator asks
for a repair rather than accepting an unverifiable label. The mirror's own correctness is
pinned against the running prelude under node, on shared vectors, in
``tests/unit/supplements/test_validate.py``; that test is the contract between the two
implementations, not this docstring.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

#: The prelude's not-a-value mark.
NOT_A_VALUE: Final = "\u2013"

#: The compact suffixes the mirror knows (en-US ``notation:"compact"``). Base 1000.
_COMPACT_SUFFIXES: Final = ("", "K", "M", "B", "T", "P", "E")


def js_number(value: float) -> str:
    """``String(v)`` for a JS number, for the ranges a component can plot."""
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    if value == 0:
        return "0"  # JS prints both 0 and -0 as "0"
    text = repr(float(value))
    if text.endswith(".0"):
        return text[:-2]
    return text


def _decimals(value: float) -> int:
    """The default ``maximumFractionDigits``: ``String(v).split(".")[1].length``.

    Literally that, including its quirks: an exponent-form number has no ``"."`` at all
    (``1e-7`` -> 0, so ``LO.fmt(1e-7)`` really does print ``0``), while ``1.5e-7`` contributes
    the four characters ``"5e-7"``. Measured against the running prelude in
    ``tests/unit/supplements/test_validate.py`` -- the mirror follows the code, not the
    intent it looks like it had.
    """
    return len(js_number(value).partition(".")[2])


def _half_up(value: float, digits: int) -> Decimal:
    """``toLocaleString``'s rounding: half away from zero (ECMA-402 ``halfExpand``).

    Python's ``format`` rounds half to EVEN, which disagrees with the browser on the exact
    tie this feature checks for: ``LO.fmt(211.25, {digits: 1})`` prints ``211.3`` in the
    frame and ``211.2`` under a naive mirror (measured 2026-10-10 on the vendored pair).
    ``Decimal(repr(v))`` keeps the shortest decimal the float actually round-trips to, so the
    tie stays a tie.
    """
    return Decimal(repr(float(value))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


def _grouped(value: float, digits: int) -> str:
    """en-US grouping with at most ``digits`` decimals, trailing zeros trimmed.

    ``toLocaleString`` pads nothing: ``maximumFractionDigits`` is a ceiling. Formatting with
    exactly ``digits`` places and then trimming the trailing zeros reproduces that for every
    value whose shortest form is at least as precise as the ceiling asks for -- which the
    default path guarantees by construction (``digits`` IS that precision).
    """
    text = f"{_half_up(value, max(0, digits)):,.{max(0, digits)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _compact(value: float) -> str:
    """``toLocaleString(undefined,{notation:"compact",maximumFractionDigits:1})``, en-US."""
    magnitude = abs(value)
    index = 0
    scaled = magnitude
    while scaled >= 1000 and index < len(_COMPACT_SUFFIXES) - 1:
        scaled /= 1000.0
        index += 1
    rounded = float(_half_up(scaled, 1))
    if rounded >= 1000 and index < len(_COMPACT_SUFFIXES) - 1:
        rounded /= 1000.0
        index += 1
    body = f"{rounded:.1f}".rstrip("0").rstrip(".")
    return f"{'-' if value < 0 else ''}{body}{_COMPACT_SUFFIXES[index]}"


def fmt(value: Any, *, unit: str = "", digits: int | None = None, compact: bool = False) -> str:
    """The prelude's ``LO.fmt`` as a pure function (see the module docstring).

    ``value`` is whatever the data holds: a number, a numeric string, ``None``, or an
    identifier. Anything that is not a finite number is printed as JS would print it.
    """
    if value is None or value == "":
        return NOT_A_VALUE
    numeric: float | None
    if isinstance(value, bool):
        numeric = None
    elif isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            numeric = float(text)
        except ValueError:
            numeric = None
    else:
        numeric = None
    if numeric is None or not math.isfinite(numeric):
        # ``String(n)`` for a non-finite number, and the string itself for a non-numeric one.
        return js_number(numeric) if numeric is not None else str(value)
    precision = _decimals(numeric) if digits is None else max(0, int(digits))
    if compact and abs(numeric) >= 1e4:
        body = _compact(numeric)
    else:
        body = _grouped(numeric, precision)
    if not unit:
        return body
    return f"{body}%" if unit == "%" else f"{body} {unit}"
