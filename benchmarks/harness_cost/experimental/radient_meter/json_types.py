"""Validate untrusted wire JSON once; downstream code never depends on Any.

Charges are JSON numbers, not strings or booleans. Configuration uses Decimal
separately, so user-supplied budget strings cannot weaken wire validation.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TypeAlias, cast

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def _validate(value: object) -> JSONValue:
    """Narrow json.loads' dynamic boundary, rejecting non-JSON Python objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_validate(child) for child in cast(list[object], value)]
    if isinstance(value, dict):
        result: dict[str, JSONValue] = {}
        for key, child in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object key must be a string")
            result[key] = _validate(child)
        return result
    raise ValueError("not JSON")


def parse_json(raw: bytes) -> JSONValue:
    """Parse bytes without coercing numeric values; semantic validators reject NaN.

    Retaining nonfinite floats here lets a model parser preserve an earlier
    known partial bill when a later usage frame contains an invalid charge.
    Duplicate keys are ambiguous billing metadata and rejected instead of taking
    a silent last value.
    """

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        return _validate(cast(object, json.loads(raw, object_pairs_hook=pairs)))
    except RecursionError:
        # A byte bound alone does not bound JSON nesting. Normalize decoder and
        # validation recursion at the shared boundary so every admission caller
        # takes its existing invalid-JSON abort path, never an unlatched escape.
        raise ValueError("JSON nesting exceeds validated boundary") from None


def object_value(value: JSONValue) -> dict[str, JSONValue]:
    """Require a JSON object without a coercive empty-object fallback."""
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def amount(value: JSONValue) -> Decimal:
    """Accept only finite nonnegative JSON numeric charge statements."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid amount type")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("invalid amount value")
    return result


def count(value: JSONValue) -> int:
    """Token/count fields are nonnegative integers, never bool or float."""
    if type(value) is not int or value < 0:
        raise ValueError("invalid count")
    return value
