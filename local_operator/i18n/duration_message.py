"""Shared elapsed-time copy without choosing a caller's locale or presentation.

Only rendered strings leave this seam: deferred keys must never escape as wire
identities. The callback keeps backend config ownership separate from Info's
active locale and preserves str subclasses carrying paint-only provenance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from local_operator.i18n.keys import wire_session_status as keys
from local_operator.i18n.messages import Msg

T = TypeVar("T", bound=str)


def _checked(value: T) -> T:
    # A miswired callback returning Msg would bypass the deferred-key boundary.
    if not isinstance(value, str):
        raise TypeError("duration renderer must return str")
    return value


def format_duration(seconds: float, render: Callable[[Msg], T]) -> T:
    """Render the legacy four branches, clamping negatives and truncating fractions."""
    total = int(max(0.0, seconds))
    if total < 60:
        return _checked(render(keys.duration_seconds(count=total)))
    if total < 3600:
        return _checked(render(keys.duration_minutes(count=total // 60)))
    if total < 86400:
        return _checked(
            render(keys.duration_hours_minutes(hours=total // 3600, minutes=(total % 3600) // 60))
        )
    # ICU number grouping would change the legacy ungrouped 1000-day output.
    return _checked(
        render(keys.duration_days_hours(days=str(total // 86400), hours=(total % 86400) // 3600))
    )
