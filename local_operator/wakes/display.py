"""Human-facing wake timestamps; scheduling and JSON retain epoch milliseconds."""

from __future__ import annotations

from datetime import UTC, datetime

DEFAULT_TIME_FORMAT = "12h"


def format_age(seconds: float) -> str:
    """Approximate status age: the CLI's single-unit 45s/12m/3h/2d ladder.

    Unlike schedule-duration serialization, a status slot should sacrifice
    precision before width. Retaining only its largest unit stays readable as
    poll ticks add seconds to multi-day ages, rather than falling back to an
    exact but unreadable millisecond count. Scheduling timestamps stay exact.
    """
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    return f"{hours // 24}d"


def format_wake_instant(epoch_ms: int) -> str:
    """The due instant as ISO-8601 WITH its offset, e.g. '2026-11-05T23:45:00-05:00'.

    Convert at the due instant (not today's tzinfo), so a wake across a DST
    boundary prints the offset it will actually fire under. A clock reading
    alone cannot say which day it belongs to (a bare HH:MM resolves to the NEXT
    occurrence, often tomorrow) and a zone abbreviation alone cannot say which
    offset was used; this form says both.
    """
    return (
        datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).astimezone().isoformat(timespec="seconds")
    )


def format_wake_at_echo(next_due_at: int, at_text: str | None, now_ms: int) -> str:
    """``<ISO instant> (from '<input>', read as LOCAL time on this machine)``.

    Empty string for anything that already read as itself (``+30m``, an ISO
    timestamp with an offset) or when the re-resolution does not reproduce
    ``next_due_at`` (never quote an instant this text did not produce).

    Exists because a clock reading and the instant it names are two different
    statements: a caller that wrote ``18:45`` meaning 18:45Z got the device's
    next 18:45, and nothing in the reply disagreed with what they thought they
    had asked for. The echo is the one place the resolved instant is put beside
    the text that produced it.
    """
    from local_operator.harness.wake import resolve_wake_at

    if not isinstance(at_text, str) or not at_text.strip():
        return ""
    moment = resolve_wake_at(at_text, now_ms)
    if moment is None or moment.kind != "local" or moment.epoch_ms != next_due_at:
        return ""
    return (
        f"{format_wake_instant(next_due_at)} (from {at_text.strip()!r}, "
        "read as LOCAL time on this machine)"
    )


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
