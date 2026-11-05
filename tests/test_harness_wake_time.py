"""A wake time is an instant PLUS the reading that produced it.

The incident these pin: a lane wrote ``at="18:45"`` meaning 18:45Z, ``HH:MM``
was read as the device's next 18:45 (23:45Z), and a 30-minute watch went two
hours without ticking. Nothing in the accepted form or the reply disagreed with
what the writer thought they had asked for, so the fix is to (a) keep the
reading as data, (b) accept an explicit ``local:`` spelling, and (c) let the
surfaces quote the resolved instant back beside the text that produced it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Callable, Iterator
from zoneinfo import ZoneInfo

import pytest

from local_operator.harness.wake import (
    build_wake_schedule,
    parse_wake_at,
    resolve_wake_at,
)
from local_operator.wakes.display import format_wake_at_echo, format_wake_instant


@pytest.fixture
def tz(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Set the process zone, the way ``tests/unit/wakes/test_display.py`` does.

    ``TZ`` alone changes nothing already cached in libc, so ``tzset`` is
    required; where it does not exist (Windows) the assertions would silently
    read the ambient zone instead, so those cases skip.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the process timezone requires tzset")

    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield _set
    # ``undo`` puts TZ back; the C library's copy only follows on a tzset.
    monkeypatch.undo()
    time.tzset()


def local_ms(*args: int, zone: str = "America/New_York") -> int:
    return int(datetime(*args, tzinfo=ZoneInfo(zone)).timestamp() * 1000)


def clock_ms(text: str, now: int) -> int:
    """The instant a clock reading resolves to, asserting it was readable."""
    moment = resolve_wake_at(text, now)
    assert moment is not None
    return moment.epoch_ms


def clock_kind(text: str, now: int) -> str:
    moment = resolve_wake_at(text, now)
    assert moment is not None
    return moment.kind


def test_bare_hh_mm_is_this_machines_local_zone(tz) -> None:
    """The ambiguity itself: a bare clock reading is the DEVICE's next 18:45."""
    tz("America/New_York")
    now = local_ms(2026, 11, 5, 12, 0)

    resolved = parse_wake_at("18:45", now)

    assert resolved == local_ms(2026, 11, 5, 18, 45)
    # The same text read as UTC is a different instant — which is the whole
    # reason the resolved instant has to be quoted back.
    assert resolved != int(datetime(2026, 11, 5, 18, 45, tzinfo=UTC).timestamp() * 1000)
    assert clock_kind("18:45", now) == "local"
    echo = format_wake_at_echo(clock_ms("18:45", now), "18:45", now)
    assert echo.startswith("2026-11-05T18:45:00-05:00")
    assert "'18:45'" in echo
    assert "LOCAL" in echo


def test_local_prefix_is_the_explicit_spelling_of_the_same_reading(tz) -> None:
    """``local:HH:MM`` says what the bare form only implies, and resolves the same."""
    tz("America/New_York")
    now = local_ms(2026, 11, 5, 12, 0)

    assert parse_wake_at("local:18:45", now) == parse_wake_at("18:45", now)
    # Case-insensitive, with an optional space either side of the colon: this
    # is a human spelling, not a token.
    assert parse_wake_at("LOCAL: 18:45", now) == parse_wake_at("18:45", now)
    assert clock_kind("LOCAL: 18:45", now) == "local"

    echo = format_wake_at_echo(clock_ms("local:18:45", now), "local:18:45", now)
    assert "'local:18:45'" in echo
    assert echo.startswith("2026-11-05T18:45:00-05:00")


def test_iso_with_an_offset_is_absolute_not_local(tz) -> None:
    """An offset names its zone; the device's clock must not enter the answer."""
    tz("America/New_York")
    now = local_ms(2026, 11, 5, 12, 0)
    utc_instant = int(datetime(2026, 11, 5, 18, 45, tzinfo=UTC).timestamp() * 1000)

    for text in ("2026-11-05T18:45:00Z", "2026-11-05T18:45:00+00:00"):
        moment = resolve_wake_at(text, now)
        assert moment is not None
        assert moment.kind == "absolute"
        assert moment.epoch_ms == utc_instant
        assert moment.epoch_ms != parse_wake_at("18:45", now)
        # Nothing to echo: the text already stated the instant it meant.
        assert format_wake_at_echo(moment.epoch_ms, text, now) == ""


def test_naive_iso_reads_as_local_and_is_echoed(tz) -> None:
    """Same ambiguity as ``HH:MM``: without an offset only the zone decides."""
    tz("America/New_York")
    now = local_ms(2026, 11, 5, 12, 0)

    moment = resolve_wake_at("2026-11-05T18:45:00", now)

    assert moment is not None
    assert moment.kind == "local"
    assert moment.epoch_ms == parse_wake_at("18:45", now)
    assert format_wake_at_echo(moment.epoch_ms, "2026-11-05T18:45:00", now) != ""


@pytest.mark.parametrize(
    ("now_args", "expected_offset", "expected_wall_clock"),
    [
        # Spring forward 2026-03-08: 23:00 EST on the 7th, `03:30` is the next
        # 03:30 — which exists only under EDT.
        ((2026, 3, 7, 23, 0), "-0400", "2026-03-08T03:30:00"),
        # Fall back 2026-11-01: 23:00 EDT on the 31st, the next 03:30 is EST.
        ((2026, 10, 31, 23, 0), "-0500", "2026-11-01T03:30:00"),
    ],
)
def test_a_dst_boundary_keeps_the_wall_clock_and_reports_the_offset_used(
    tz,
    now_args: tuple[int, int, int, int, int],
    expected_offset: str,
    expected_wall_clock: str,
) -> None:
    """Assert the offset ACTUALLY USED, not merely "another day".

    Date arithmetic (``date + 1 day``) is what keeps the requested wall-clock
    time across the transition; ``+24h`` would land an hour off. The echo is the
    only surface that reports which offset the wake will fire under.
    """
    tz("America/New_York")
    now = local_ms(*now_args)

    resolved = clock_ms("03:30", now)

    fired = datetime.fromtimestamp(resolved / 1000).astimezone()
    assert fired.strftime("%z") == expected_offset
    iso_offset = f"{expected_offset[:3]}:{expected_offset[3:]}"
    assert fired.isoformat(timespec="seconds") == f"{expected_wall_clock}{iso_offset}"
    echo = format_wake_at_echo(resolved, "03:30", now)
    assert echo.startswith(f"{expected_wall_clock}{iso_offset}")


def test_the_echo_is_never_quoted_for_an_instant_it_did_not_produce(tz) -> None:
    """A stale or mismatched instant must produce no claim at all."""
    tz("America/New_York")
    now = local_ms(2026, 11, 5, 12, 0)
    resolved = clock_ms("18:45", now)

    assert format_wake_at_echo(resolved, "+30m", now) == ""
    assert format_wake_at_echo(resolved, None, now) == ""
    assert format_wake_at_echo(resolved, "  ", now) == ""
    assert format_wake_at_echo(resolved + 60_000, "18:45", now) == ""


def test_an_invalid_at_is_refused_naming_every_form_and_the_local_rule(tz) -> None:
    """No fallback: an unreadable ``at`` is a sentence, not a silent guess."""
    tz("America/New_York")

    outcome = build_wake_schedule({"message": "m", "at": "tomorrow"}, [], local_ms(2026, 11, 5))

    assert "error" in outcome
    assert outcome["malformed"] is True
    error = outcome["error"]
    assert "+30m" in error
    assert "local:HH:MM" in error
    assert "ISO-8601" in error
    assert "offset" in error
    assert "LOCAL" in error


def test_local_prefix_refuses_rather_than_falling_through_to_iso(tz) -> None:
    """``local:`` followed by a non-clock is wrong, not an absolute timestamp."""
    tz("America/New_York")
    now = local_ms(2026, 11, 5, 12, 0)

    assert resolve_wake_at("local:banana", now) is None
    assert resolve_wake_at("local:25:00", now) is None
    assert resolve_wake_at("local:2026-11-05T18:45:00Z", now) is None


def test_instant_render_carries_the_offset(tz) -> None:
    """The echo's instant half is ISO-8601 with the offset, seconds precision."""
    tz("America/New_York")

    assert format_wake_instant(local_ms(2026, 11, 5, 18, 45)) == "2026-11-05T18:45:00-05:00"


def test_a_bare_clock_still_works_when_tz_is_inherited() -> None:
    """Backward compatibility: an existing ``at='07:15'`` schedule keeps parsing.

    The form is unchanged for every caller that already used it; only the
    READING is now named. No ``tz`` fixture, so this also covers whatever zone
    the host happens to run in.
    """
    now = int(datetime.now(UTC).timestamp() * 1000)

    assert parse_wake_at("07:15", now) is not None
