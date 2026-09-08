"""Human-facing wake timestamps; scheduling and JSON retain epoch milliseconds."""

from __future__ import annotations

from datetime import UTC, datetime

DEFAULT_TIME_FORMAT = "12h"


def format_wake_time(epoch_ms: int, *, now: datetime | None = None) -> str:
    """Render in the OS local zone at the due instant, including its DST offset.

    Convert the instant itself rather than attaching today's local tzinfo: the
    latter is a fixed offset and gives the wrong clock across a DST transition.
    A date is necessary off today's local date; the year disambiguates distant
    reminders. The zone stays visible even today so UTC is never implied.
    """
    from local_operator.tui.settings import settings_get

    due = datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).astimezone()
    today = (now or datetime.now(UTC)).astimezone().date()
    if settings_get("display.time_format", DEFAULT_TIME_FORMAT) == "24h":
        clock = f"{due.hour:02d}:{due.minute:02d}"
    else:
        # Explicit AM/PM, not locale-dependent %p (which can be empty).
        clock = f"{due.hour % 12 or 12}:{due.minute:02d} {'AM' if due.hour < 12 else 'PM'}"
    date = ""
    if due.date() != today:
        date = due.strftime("%b %d %Y " if due.year != today.year else "%b %d ")
    return f"{date}{clock} {due.tzname() or due.strftime('%z')}"
