"""Wake tests: pure parsing/scheduling semantics + the live scheduler."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest

from local_operator.harness.wake import (
    LOAD_GRACE_MS,
    MAX_WAKE_MESSAGE_CHARS,
    MAX_WAKE_SCHEDULES,
    MIN_WAKE_INTERVAL_MS,
    WAKE_SCRATCH_CLAUSE,
    DueWake,
    WakeSchedule,
    WakeScheduler,
    advance_wake_schedule,
    build_wake_schedule,
    format_duration,
    format_wake_delivery_text,
    missed_occurrences,
    parse_wake_at,
    parse_wake_duration,
)

NOW = 1_700_000_000_000  # fixed epoch ms for pure tests


class TestParseDuration:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("45s", 45_000),
            ("30m", 30 * 60_000),
            ("2h", 2 * 3_600_000),
            ("7d", 7 * 86_400_000),
            ("1w", 604_800_000),
        ],
    )
    def test_units(self, text, expected):
        assert parse_wake_duration(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("8h30m", 8 * 3_600_000 + 30 * 60_000),
            ("1h30m15s", 3_600_000 + 30 * 60_000 + 15_000),
            ("1h 30m", 3_600_000 + 30 * 60_000),
            ("1w2d", 604_800_000 + 2 * 86_400_000),
            ("2h15m", 2 * 3_600_000 + 15 * 60_000),
            ("8H30M", 8 * 3_600_000 + 30 * 60_000),  # case-insensitive
            ("90m45s", 90 * 60_000 + 45_000),  # terms need not be unit-ordered
            (" 8h30m ", 8 * 3_600_000 + 30 * 60_000),  # surrounding whitespace
        ],
    )
    def test_compound_units_sum(self, text, expected):
        """Compound durations ("8h30m") are the natural way to say "8 and a
        half hours"; rejecting them made the most common phrasing an error."""
        assert parse_wake_duration(text) == expected

    def test_bare_number_rejected(self):
        """60 reads as both seconds and ms; guessing wrong is a runaway loop."""
        assert parse_wake_duration("60") is None
        assert parse_wake_duration("1000") is None

    def test_zero_sum_rejected(self):
        """A compound summing to zero ("0s") is the same runaway-loop risk as
        a bare number, one frame later."""
        assert parse_wake_duration("0s") is None
        assert parse_wake_duration("0h0m") is None

    def test_garbage_rejected(self):
        assert parse_wake_duration("") is None
        assert parse_wake_duration("soon") is None
        assert parse_wake_duration("-5m") is None
        assert parse_wake_duration("5x") is None
        assert parse_wake_duration("8h30") is None  # trailing bare number
        assert parse_wake_duration("h30m") is None  # leading bare unit


class TestParseAt:
    def test_plus_duration(self):
        assert parse_wake_at("+30m", NOW) == NOW + 30 * 60_000

    def test_plus_compound_duration(self):
        assert parse_wake_at("+8h30m", NOW) == NOW + 8 * 3_600_000 + 30 * 60_000

    def test_plus_bare_number_rejected(self):
        assert parse_wake_at("+60", NOW) is None

    def test_clock_today(self):
        base = int(datetime(2026, 8, 4, 9, 0, 0).timestamp() * 1000)
        result = parse_wake_at("10:30", base)
        assert result == int(datetime(2026, 8, 4, 10, 30).timestamp() * 1000)

    def test_clock_wraps_to_tomorrow(self):
        base = int(datetime(2026, 8, 4, 12, 0, 0).timestamp() * 1000)
        result = parse_wake_at("09:00", base)
        assert result == int(datetime(2026, 8, 5, 9, 0).timestamp() * 1000)

    def test_clock_is_date_arithmetic_not_24h(self):
        """Next occurrence keeps the wall-clock time across DST boundaries —
        computed via date+1day, not +24h. (Deterministic check in any zone:
        result is the next calendar day at that time.)"""
        base_dt = datetime(2026, 8, 4, 23, 0, 0)
        base = int(base_dt.timestamp() * 1000)
        result = parse_wake_at("00:30", base)
        expected = datetime(2026, 8, 5, 0, 30)
        assert result == int(expected.timestamp() * 1000)

    def test_clock_invalid(self):
        assert parse_wake_at("25:00", NOW) is None
        assert parse_wake_at("12:99", NOW) is None

    def test_iso_8601(self):
        target = datetime(2027, 1, 2, 3, 4, 5)
        result = parse_wake_at(target.isoformat(), NOW)
        assert result == int(target.timestamp() * 1000)

    def test_garbage(self):
        assert parse_wake_at("whenever", NOW) is None
        assert parse_wake_at("", NOW) is None


class TestBuildSchedule:
    def make_existing(self, n: int) -> list[WakeSchedule]:
        return [
            WakeSchedule(id=f"w{i + 1}", message="m", next_due_at=NOW + 1000, created_at=NOW)
            for i in range(n)
        ]

    def test_creates_with_first_free_id(self):
        existing = self.make_existing(1)  # w1 taken
        result = build_wake_schedule({"message": "check", "in": "5m"}, existing, NOW)
        assert "error" not in result
        schedule = result["schedule"]
        assert schedule.id == "w2"
        assert schedule.next_due_at == NOW + 5 * 60_000
        assert schedule.fired_count == 0
        assert schedule.created_at == NOW

    def test_missing_message_errors(self):
        result = build_wake_schedule({"in": "5m"}, [], NOW)
        assert "error" in result and "message" in result["error"]

    def test_missing_when_errors(self):
        result = build_wake_schedule({"message": "x"}, [], NOW)
        assert "error" in result

    def test_max_schedules(self):
        result = build_wake_schedule(
            {"message": "x", "in": "5m"}, self.make_existing(MAX_WAKE_SCHEDULES), NOW
        )
        assert "error" in result and str(MAX_WAKE_SCHEDULES) in result["error"]

    def test_message_too_long(self):
        result = build_wake_schedule(
            {"message": "x" * (MAX_WAKE_MESSAGE_CHARS + 1), "in": "5m"}, [], NOW
        )
        assert "error" in result

    def test_interval_too_small(self):
        result = build_wake_schedule({"message": "x", "in": "5m", "every": "30s"}, [], NOW)
        assert "error" in result and "60s" in result["error"]

    def test_min_interval_accepted(self):
        result = build_wake_schedule(
            {"message": "x", "in": "5m", "every": f"{MIN_WAKE_INTERVAL_MS // 1000}s"}, [], NOW
        )
        assert "error" not in result

    def test_past_at_rejected_beyond_grace(self):
        past = NOW - 60_000
        iso = datetime.fromtimestamp(past / 1000).isoformat()
        result = build_wake_schedule({"message": "x", "at": iso}, [], NOW)
        assert "error" in result and "past" in result["error"]

    def test_past_at_within_grace_clamps_to_now(self):
        grace_past = NOW - 3_000  # within PAST_AT_GRACE_MS
        iso = datetime.fromtimestamp(grace_past / 1000).isoformat()
        result = build_wake_schedule({"message": "x", "at": iso}, [], NOW)
        assert "error" not in result
        assert result["schedule"].next_due_at == NOW

    def test_at_via_clock(self):
        base = int(datetime(2026, 8, 4, 9, 0).timestamp() * 1000)
        result = build_wake_schedule({"message": "x", "at": "10:00"}, [], base)
        assert "error" not in result
        assert result["schedule"].next_due_at == int(datetime(2026, 8, 4, 10).timestamp() * 1000)

    def test_bad_every_until_limit(self):
        assert "error" in build_wake_schedule(
            {"message": "x", "in": "5m", "every": "bogus"}, [], NOW
        )
        assert "error" in build_wake_schedule(
            {"message": "x", "in": "5m", "until": "bogus"}, [], NOW
        )
        assert "error" in build_wake_schedule({"message": "x", "in": "5m", "limit": 0}, [], NOW)

    def test_notify_round_trips_and_an_edit_keeps_it(self):
        """§14.4: the request key lands on the schedule, and an edit keeps it
        unless the request moves it (a PRESENT ``notify: false`` clears it)."""
        from local_operator.harness.wake import build_wake_edit

        built = build_wake_schedule({"message": "x", "in": "5m", "notify": True}, [], NOW)
        assert "error" not in built
        schedule = built["schedule"]
        assert schedule.notify is True

        default = build_wake_schedule({"message": "x", "in": "5m"}, [], NOW)
        assert "error" not in default
        assert default["schedule"].notify is False

        kept = build_wake_edit({"message": "y"}, [schedule], schedule.id, NOW)
        assert "error" not in kept
        assert kept["schedule"].notify is True

        cleared = build_wake_edit({"notify": False}, [schedule], schedule.id, NOW)
        assert "error" not in cleared
        assert cleared["schedule"].notify is False


class TestAdvance:
    def one_shot(self, **kw) -> WakeSchedule:
        return WakeSchedule(id="w1", message="m", next_due_at=NOW, created_at=NOW - 1000, **kw)

    def test_one_shot_retires(self):
        result = advance_wake_schedule(self.one_shot(), NOW)
        assert result == {"retired": "one-shot"}

    def test_limit_retires_after_n(self):
        schedule = self.one_shot(every_ms=60_000, limit=2, fired_count=1)
        result = advance_wake_schedule(schedule, NOW)
        assert result == {"retired": "limit"}

    def test_recurring_advances(self):
        schedule = self.one_shot(every_ms=60_000)
        result = advance_wake_schedule(schedule, NOW)
        assert "next" in result
        next_schedule = result["next"]
        assert next_schedule.fired_count == 1
        assert next_schedule.next_due_at == NOW + 60_000

    def test_missed_occurrences_skipped(self):
        """Asleep six hours: owes ONE hourly fire, not six."""
        schedule = self.one_shot(every_ms=3_600_000)
        six_hours_late = NOW + 6 * 3_600_000 + 1
        result = advance_wake_schedule(schedule, six_hours_late)
        assert "next" in result
        next_schedule = result["next"]
        assert next_schedule.fired_count == 1  # one fire, not six
        # Next occurrence strictly after now, within one interval.
        assert next_schedule.next_due_at > six_hours_late
        assert next_schedule.next_due_at <= six_hours_late + 3_600_000

    def test_until_retires(self):
        schedule = self.one_shot(every_ms=60_000, until_at=NOW + 30_000)
        result = advance_wake_schedule(schedule, NOW)
        assert result == {"retired": "until"}


class TestFormatDuration:
    def test_shortest_exact_unit(self):
        assert format_duration(45_000) == "45s"
        assert format_duration(30 * 60_000) == "30m"
        assert format_duration(8 * 3_600_000) == "8h"

    def test_compound_rendered_as_compound(self):
        """Parse accepts compounds, so format must round-trip them: a wake
        created with 'every 1h30m' rendering as '90m' would read nothing like
        what was asked for."""
        assert format_duration(8 * 3_600_000 + 30 * 60_000) == "8h30m"
        assert format_duration(90 * 60_000) == "1h30m"
        assert format_duration(3_600_000 + 15_000) == "1h15s"
        assert format_duration(604_800_000 + 86_400_000) == "1w1d"

    def test_unrepresentable_falls_back_to_ms(self):
        assert format_duration(61_001) == "61001ms"


class TestMissedOccurrences:
    def recurring(self, **kw) -> WakeSchedule:
        defaults: dict[str, Any] = dict(
            id="w1", message="m", next_due_at=NOW, every_ms=3_600_000, created_at=NOW - 1000
        )
        defaults.update(kw)
        return WakeSchedule(**defaults)

    def test_not_due_yet(self):
        assert missed_occurrences(self.recurring(next_due_at=NOW + 1000), NOW) == 0

    def test_counts_past_occurrences(self):
        # Strictly-past due times at NOW+2h30m are NOW and NOW+1h; NOW+2h is
        # the current occurrence the resume is about to deliver, not a miss.
        assert missed_occurrences(self.recurring(), NOW + 2 * 3_600_000 + 30 * 60_000) == 2

    def test_one_shot_is_zero_or_one(self):
        one_shot = self.recurring(every_ms=None)
        assert missed_occurrences(one_shot, NOW + 3_600_000) == 1
        # Due exactly at ``now`` is the imminent fire, not a skip ("strictly
        # before", matching the recurring arm); only a strictly-past due time
        # counts.
        assert missed_occurrences(one_shot, NOW) == 0
        assert missed_occurrences(one_shot, NOW + 1) == 1

    def test_clamped_to_remaining_limit(self):
        """A limit-3 wake resumed a week late never claims more misses than it
        could ever have delivered."""
        limited = self.recurring(limit=3, fired_count=1)
        assert missed_occurrences(limited, NOW + 50 * 3_600_000) == 2

    def test_clamped_to_until(self):
        """Occurrences past until_at do not count — the schedule was already
        retired by then."""
        bounded = self.recurring(until_at=NOW + 3_600_000)
        assert missed_occurrences(bounded, NOW + 50 * 3_600_000) == 2


class TestDeliveryText:
    def test_envelope_with_handle_and_cancel_hint(self):
        schedule = WakeSchedule(
            id="w3",
            message="check the build",
            next_due_at=NOW,
            every_ms=3_600_000,
            limit=8,
            created_at=NOW,
        )
        due = DueWake(schedule=schedule, occurrence=3, planned_total=8, final=False)
        text = format_wake_delivery_text(due)
        assert "w3" in text
        assert "3/8" in text
        assert "every 1h" in text
        assert "cancel" in text.lower()
        assert text.endswith("check the build")

    def test_final_delivery_drops_cancel_hint(self):
        schedule = WakeSchedule(id="w1", message="m", next_due_at=NOW, created_at=NOW)
        due = DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)
        text = format_wake_delivery_text(due)
        assert "cancel" not in text.lower()

    def test_every_envelope_points_the_runs_own_files_at_the_scratchpad(self):
        """A wake is the turn with the LEAST context around it — no user message,
        no recent tool result naming a path — so it is exactly where a scheduled
        run's own files (logs, snapshots, bookkeeping) drift to ``/tmp`` by
        default. Both arms carry the clause, because the FINAL delivery is still
        a run whose files outlive the turn that wrote them.

        Asserted against the named constant rather than a copy of the sentence:
        the point of the pin is that the formatter writes the bytes the
        human-surface stripper knows how to drop, and a literal here would let
        the two drift apart while both stayed green.
        """
        recurring = DueWake(
            schedule=WakeSchedule(
                id="w3", message="check the build", next_due_at=NOW, every_ms=3_600_000
            ),
            occurrence=3,
            planned_total=8,
            final=False,
        )
        final = DueWake(
            schedule=WakeSchedule(id="w1", message="m", next_due_at=NOW, created_at=NOW),
            occurrence=1,
            planned_total=1,
            final=True,
        )

        for due in (recurring, final):
            text = format_wake_delivery_text(due)
            envelope, _, message = text.partition("\n\n")
            assert WAKE_SCRATCH_CLAUSE in envelope
            # In the ENVELOPE, never the message: the message is the schedule's
            # own words and is rendered verbatim on both human surfaces.
            assert message == due.schedule.message
            assert text.endswith(due.schedule.message)
            # The clause is model-facing markup, so it must not be something the
            # human surfaces have to know about twice — it stays short, and this
            # is the one line repeated in every delivery.
            assert len(WAKE_SCRATCH_CLAUSE) <= 80

    def test_the_clause_is_stripped_from_the_human_headline(self):
        """The receipt line a person reads is the envelope with the model-facing
        parts removed. A clause appended to the envelope therefore has to be
        stripped too, and the FINAL delivery is where that is not automatic: it
        carries no cancel hint, so a stripper that only knows the cancel phrase
        leaves the new clause sitting in the headline.
        """
        from local_operator.harness.rows import wake_receipt_headline

        live = "(alarm) Scheduled wake w-9 (1, every 6h) — cancel with "
        live += f'wake({{op:"cancel",id:"w-9"}}) once its goal is met. {WAKE_SCRATCH_CLAUSE}'
        assert wake_receipt_headline(f"{live}\n\nCheck the deploy") == "w-9 (1, every 6h)"

        final = f"(alarm) Scheduled wake w-9 (1). {WAKE_SCRATCH_CLAUSE}"
        assert wake_receipt_headline(f"{final}\n\nCheck the deploy") == "w-9 (1)."
        assert "scratchpad" not in wake_receipt_headline(f"{final}\n\nCheck the deploy")

    def test_engine_armed_rows_read_as_hers_not_as_raw_ids(self):
        """D2: the greeting's first frame must not be a schedule id.

        ``aida-greeting`` is the engine's handle, not words a person chose;
        the label map lives beside the ids (``aida/proactive.py``) and this
        function reads it lazily, so the two cannot drift. Everything the
        user created still names itself — pinned so the substitution can never
        earn the label onto somebody else's wake.
        """
        from local_operator.harness.rows import wake_receipt_headline

        greeting = f"(alarm) Scheduled wake aida-greeting (1/1). {WAKE_SCRATCH_CLAUSE}"
        assert wake_receipt_headline(f"{greeting}\n\nFirst-run greeting.") == (
            "Aida's introduction (1/1)."
        )
        extra = f"(alarm) Scheduled wake aida-extra-2 (1). {WAKE_SCRATCH_CLAUSE}"
        assert wake_receipt_headline(f"{extra}\n\nFollow up") == "Aida's follow-up (1)."
        mine = f"(alarm) Scheduled wake w-9 (1). {WAKE_SCRATCH_CLAUSE}"
        assert wake_receipt_headline(f"{mine}\n\nCheck the deploy") == "w-9 (1)."

    def test_the_label_follows_a_renamed_chief_of_staff(self, tmp_path, monkeypatch):
        """D4: the label is formatted at LOOKUP, not frozen at import time.

        Under the packaged default a literal label is invisible; a rename made
        the receipt lie — "Aida's introduction" above a body that signs off
        "Introduce yourself as Sovereign", one card, two names (design review
        round 3, D4). The lookup reads the same ``aida.name`` reader every
        other surface uses, so the receipt and the delivered body agree by
        construction; the default frames stay byte-identical.
        """
        from local_operator.harness.rows import wake_receipt_headline
        from tests.unit.aida.conftest import isolated_root_path, write_config

        root = isolated_root_path(tmp_path, monkeypatch)
        write_config(root, {"aida": {"name": "Sovereign"}})

        greeting = f"(alarm) Scheduled wake aida-greeting (1/1). {WAKE_SCRATCH_CLAUSE}"
        assert wake_receipt_headline(f"{greeting}\n\nFirst-run greeting.") == (
            "Sovereign's introduction (1/1)."
        )
        extra = f"(alarm) Scheduled wake aida-extra-2 (1). {WAKE_SCRATCH_CLAUSE}"
        assert wake_receipt_headline(f"{extra}\n\nFollow up") == "Sovereign's follow-up (1)."


# ---------------------------------------------------------------------------
# Live scheduler
# ---------------------------------------------------------------------------


class SchedulerHarness:
    """WakeScheduler wired to in-memory callbacks with an injectable clock.
    ``wall_clock=True`` backs the clock with real time — required when the
    scheduler's OWN timer drives pump() and only real time advances."""

    def __init__(self, start: int = NOW, wall_clock: bool = False):
        # Wall-clock mode compares against real time everywhere, so the
        # synthetic epoch must be real time too or every schedule looks
        # overdue at load.
        self.now_ms = int(time.time() * 1000) if wall_clock else start
        self.delivered: list[DueWake] = []
        self.persisted: list[list[WakeSchedule]] = []
        self.retired: list[tuple[WakeSchedule, str]] = []
        self.deliver_should_raise = False
        if wall_clock:
            now_fn: Callable[[], int] = lambda: int(time.time() * 1000)  # noqa: E731
        else:
            now_fn = lambda: self.now_ms  # noqa: E731
        self.scheduler = WakeScheduler(
            now=now_fn,
            deliver=self._deliver,
            persist=self._persist,
            on_retire=self._on_retire,
        )

    async def _deliver(self, due: DueWake) -> None:
        if self.deliver_should_raise:
            raise RuntimeError("delivery broken")
        self.delivered.append(due)

    async def _persist(self, schedules: list[WakeSchedule]) -> None:
        self.persisted.append(list(schedules))

    async def _on_retire(self, schedule: WakeSchedule, reason: str) -> None:
        self.retired.append((schedule, reason))

    def schedule(self, **kw) -> WakeSchedule:
        defaults: dict[str, Any] = dict(
            id="w1", message="m", next_due_at=self.now_ms + 60_000, created_at=self.now_ms
        )
        defaults.update(kw)
        return WakeSchedule(**defaults)


@pytest.mark.asyncio
async def test_scheduler_pump_fires_due_wakes():
    harness = SchedulerHarness()
    # Due in the future at load (no grace clamp); advance past it.
    harness.scheduler.load([harness.schedule(next_due_at=NOW + 100)])
    harness.now_ms = NOW + 50
    assert await harness.scheduler.pump() == 0  # not due yet
    harness.now_ms = NOW + 200
    fired = await harness.scheduler.pump()
    assert fired == 1
    assert len(harness.delivered) == 1
    assert harness.delivered[0].occurrence == 1
    # One-shot retires immediately.
    assert harness.retired and harness.retired[0][1] == "one-shot"
    assert harness.scheduler.schedules == ()
    # Persisted after fire.
    assert harness.persisted and harness.persisted[-1] == []
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_recurring_advances_and_keeps():
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule(every_ms=60_000, next_due_at=NOW + 10)])
    harness.now_ms = NOW + 20
    fired = await harness.scheduler.pump()
    assert fired == 1
    remaining = harness.scheduler.schedules
    assert len(remaining) == 1
    assert remaining[0].fired_count == 1
    assert remaining[0].next_due_at > NOW + 20
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_delivery_throw_still_advances():
    """One broken wake must not become a hot loop."""
    harness = SchedulerHarness()
    harness.deliver_should_raise = True
    harness.scheduler.load([harness.schedule(every_ms=60_000, next_due_at=NOW + 10)])
    harness.now_ms = NOW + 20
    fired = await harness.scheduler.pump()
    assert fired == 1
    remaining = harness.scheduler.schedules
    assert len(remaining) == 1
    assert remaining[0].fired_count == 1  # advanced despite the throw
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_load_grace_for_overdue():
    """An overdue wake adopted at load fires shortly AFTER load, not inside it."""
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule(next_due_at=NOW - 50_000)])
    adopted = harness.scheduler.schedules[0]
    assert adopted.next_due_at == NOW + LOAD_GRACE_MS
    # Not due yet at load time.
    assert await harness.scheduler.pump() == 0
    harness.now_ms = NOW + LOAD_GRACE_MS + 1
    assert await harness.scheduler.pump() == 1
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_load_reports_missed_wakes():
    """Overdue schedules adopted at load are recorded with their skipped
    counts so the caller can aggregate ONE catch-up delivery instead of
    letting each overdue wake fire its own turn."""
    harness = SchedulerHarness()
    overdue_one_shot = harness.schedule(id="w1", next_due_at=NOW - 5_000)
    overdue_recurring = harness.schedule(
        id="w2", every_ms=3_600_000, next_due_at=NOW - 10 * 3_600_000, created_at=NOW + 1
    )
    future = harness.schedule(id="w3", next_due_at=NOW + 60_000, created_at=NOW + 2)
    harness.scheduler.load([overdue_one_shot, overdue_recurring, future])

    missed = harness.scheduler.take_missed()
    by_id = {m["schedule"].id: m["occurrences"] for m in missed}
    assert by_id == {"w1": 1, "w2": 10}  # future wake is not a miss
    # The entry carries the ORIGINAL due time (needed for the "first missed
    # at" label), not the grace-shifted re-arm.
    w2 = next(m for m in missed if m["schedule"].id == "w2")
    assert w2["schedule"].next_due_at == NOW - 10 * 3_600_000
    # One-shot consumer: the second take is empty.
    assert harness.scheduler.take_missed() == []
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_load_with_no_overdue_records_no_missed():
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule(next_due_at=NOW + 60_000)])
    assert harness.scheduler.take_missed() == []
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_update_persists_full_list():
    harness = SchedulerHarness()
    s1 = harness.schedule(id="w1")
    s2 = harness.schedule(id="w2", created_at=NOW + 1)
    await harness.scheduler.update([s2, s1])  # re-sorted by created_at
    assert [s.id for s in harness.scheduler.schedules] == ["w1", "w2"]
    assert len(harness.persisted) == 1
    assert len(harness.persisted[0]) == 2
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_load_does_not_persist():
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule()])
    assert harness.persisted == []
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_missed_fire_skips_not_replays():
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule(every_ms=60_000, next_due_at=NOW)])
    harness.now_ms = NOW + 5 * 60_000 + 30_000  # five intervals missed
    fired = await harness.scheduler.pump()
    assert fired == 1  # ONE fire, not five
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_timer_arms_and_fires():
    """The armed asyncio timer actually fires the wake without a manual pump
    (wall clock, since the scheduler's own tick reads it)."""
    harness = SchedulerHarness(wall_clock=True)
    due_at = harness.now_ms + 50
    harness.scheduler.load([harness.schedule(next_due_at=due_at)])
    await asyncio.sleep(0.4)  # arm ~50ms (MIN_ARM_MS floor 25ms) + slack
    assert len(harness.delivered) == 1
    harness.scheduler.dispose()


@pytest.mark.asyncio
async def test_scheduler_dispose_cancels_timer():
    """dispose() must clear the scheduled handle so a pending wake never
    keeps the loop alive."""
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule(next_due_at=harness.now_ms + 60_000)])
    assert harness.scheduler._timer is not None
    harness.scheduler.dispose()
    assert harness.scheduler._timer is None
    assert harness.scheduler.disposed is True
    # No fire after dispose even past the due time.
    harness.now_ms += 120_000
    await asyncio.sleep(0.05)
    assert harness.delivered == []


@pytest.mark.asyncio
async def test_scheduler_max_arm_bounds_timer():
    """A wake a week out arms at most a MAX_ARM_MS re-check tick."""
    harness = SchedulerHarness()
    harness.scheduler.load([harness.schedule(next_due_at=harness.now_ms + 7 * 86_400_000)])
    timer = harness.scheduler._timer
    assert timer is not None
    when = timer.when()
    loop_now = asyncio.get_running_loop().time()
    assert when - loop_now <= 61.0  # ~MAX_ARM_MS + slack
    harness.scheduler.dispose()


# -- update()'s removal report (the wake-spool flood, part 2) --------------------


def _scheduler(**kw: Any) -> WakeScheduler:
    return WakeScheduler(
        now=lambda: NOW,
        deliver=lambda due: None,
        persist=lambda schedules: None,
        **kw,
    )


@pytest.mark.asyncio
async def test_update_reports_exactly_what_it_removed():
    """One site sees every removal, so the purge of a cancelled schedule's
    spooled tail lives here rather than on the wake tool's own path."""
    seen: list[list[WakeSchedule]] = []
    scheduler = _scheduler(on_removed=lambda rows: seen.append(list(rows)))
    a = WakeSchedule(id="a", message="m", next_due_at=NOW + 1, created_at=NOW)
    b = WakeSchedule(id="b", message="m", next_due_at=NOW + 1, created_at=NOW)

    await scheduler.update([a, b])
    assert seen == [], "an initial write removes nothing"

    await scheduler.update([a])
    assert [[row.id for row in rows] for rows in seen] == [["b"]]

    # An in-place rewrite of the SAME id is not a removal, which is what keeps
    # patience's armed_after flush from purging a live schedule's spool.
    await scheduler.update([a.model_copy(update={"message": "edited"})])
    assert len(seen) == 1
    scheduler.dispose()


@pytest.mark.asyncio
async def test_a_broken_removal_hook_cannot_break_scheduling():
    """Observation, not ownership: the schedules are already persisted, and the
    change notification must still go out when the purge throws."""
    changes: list[int] = []

    async def boom(rows: list[WakeSchedule]) -> None:
        raise RuntimeError("purge exploded")

    scheduler = _scheduler(on_removed=boom, on_change=lambda: changes.append(1))
    a = WakeSchedule(id="a", message="m", next_due_at=NOW + 1_000, created_at=NOW)
    await scheduler.update([a])
    await scheduler.update([])

    assert scheduler.schedules == ()
    assert changes == [1, 1], "the cancel still landed"
    scheduler.dispose()
