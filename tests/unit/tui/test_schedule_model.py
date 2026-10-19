"""The ``/schedule`` page's pure derivations, against the real builders.

Every validation case here ends in the sentence ``harness.wake`` would hand the
model, because the page paints that same sentence: a second vocabulary of
refusals is the defect this file exists to prevent.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from local_operator.harness.wake import format_duration
from local_operator.harness.wake_types import MAX_WAKE_MESSAGE_CHARS, WakeSchedule
from local_operator.tui import schedule
from local_operator.wakes.display import format_wake_time

NOW = 1_790_000_000_000  # epoch ms, an arbitrary fixed instant
MINUTE = 60_000
DAY = 86_400_000


def _schedule(
    wake_id: str,
    *,
    message: str = "check the build",
    next_due_at: int | None = None,
    every_ms: int | None = None,
    until_at: int | None = None,
    limit: int | None = None,
    fired_count: int = 0,
    notify: bool = False,
    kind: str = "scheduled",
    hidden: bool = False,
) -> WakeSchedule:
    return WakeSchedule(
        id=wake_id,
        message=message,
        next_due_at=NOW + 30 * MINUTE if next_due_at is None else next_due_at,
        every_ms=every_ms,
        until_at=until_at,
        limit=limit,
        fired_count=fired_count,
        created_at=NOW - DAY,
        notify=notify,
        kind=kind,  # type: ignore[arg-type]
        hidden=hidden,
    )


# ---------------------------------------------------------------------------
# The When field: one input, two request keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # A bare duration is the "in" spelling; the word is optional.
        ("30m", {"in": "30m"}),
        ("in 30m", {"in": "30m"}),
        ("IN 8h30m", {"in": "8h30m"}),
        # A bare clock, +duration or ISO timestamp is the "at" spelling.
        ("09:00", {"at": "09:00"}),
        ("at 09:00", {"at": "09:00"}),
        ("+2h", {"at": "+2h"}),
        ("2026-10-20 09:00", {"at": "2026-10-20 09:00"}),
        # Blank is a real value: keep the row's current time on EDIT.
        ("", {}),
        ("   ", {}),
    ],
)
def test_when_request_splits_a_bare_value_into_the_natural_key(text, expected) -> None:
    assert schedule.when_request(text) == expected


def test_when_request_passes_an_unparseable_value_to_the_builder() -> None:
    """No second parser: junk goes to the builder as an "at" and its sentence
    names what is accepted, which is the copy the page paints."""
    assert schedule.when_request("soonish") == {"at": "soonish"}


# ---------------------------------------------------------------------------
# Repeat and Ends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", {"every": None}),
        ("every 1d", {"every": "1d"}),
        ("EVery 1h30m", {"every": "1h30m"}),
        ("1d", {"every": "1d"}),
        # `every` on its own is an empty interval, which the builder refuses
        # with its own sentence rather than a page-specific one.
        ("every", {"every": ""}),
    ],
)
def test_repeat_request_accepts_the_word_and_the_bare_duration(text, expected) -> None:
    assert schedule.repeat_request(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", {"limit": None, "until": None}),
        # BOTH keys on every branch: the opposite side is a deliberate CLEAR,
        # not an omission, so switching the bound type cannot retain the old
        # one hidden beside it (round-1 review F1).
        ("limit 3", {"limit": 3, "until": None}),
        ("3", {"limit": 3, "until": None}),
        ("until 2026-10-20 09:00", {"limit": None, "until": "2026-10-20 09:00"}),
        ("2026-10-20T09:00:00+11:00", {"limit": None, "until": "2026-10-20T09:00:00+11:00"}),
        # Not a number: the builder's own "invalid 'limit'" sentence is the
        # refusal, so the page invents nothing.
        ("limit soon", {"limit": "soon", "until": None}),
    ],
)
def test_ends_request_routes_a_limit_or_an_until(text, expected) -> None:
    assert schedule.ends_request(text) == expected


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_builds_through_the_shared_validator_and_names_the_new_row() -> None:
    result = schedule.create(
        schedule.FormValues(message="water the plants", when="in 45m", notify=True),
        [],
        NOW,
    )
    assert result.ok and result.error == ""
    built = result.schedule
    assert built is not None
    assert built.id == "w1"
    assert built.message == "water the plants"
    assert built.next_due_at == NOW + 45 * MINUTE
    assert built.every_ms is None
    assert built.notify is True
    assert result.sentence == schedule.notice_created("w1", built.next_due_at)
    assert "w1" in result.sentence


def test_create_repeats_and_bounds_reach_the_row() -> None:
    result = schedule.create(
        schedule.FormValues(message="standup", when="at 09:00", repeat="every 1d", ends="limit 3"),
        [],
        NOW,
    )
    assert result.ok
    assert result.schedule is not None
    assert result.schedule.every_ms == DAY
    assert result.schedule.limit == 3


@pytest.mark.parametrize(
    "values,expected",
    [
        (schedule.FormValues(message="", when="in 30m"), "non-empty 'message'"),
        (schedule.FormValues(message="   ", when="in 30m"), "non-empty 'message'"),
        (schedule.FormValues(message="x" * (MAX_WAKE_MESSAGE_CHARS + 1), when="in 30m"), "at most"),
        (schedule.FormValues(message="x", when="soonish"), "invalid time"),
        (schedule.FormValues(message="x", when="in soonish"), "invalid duration"),
        (schedule.FormValues(message="x", when="+0s"), "invalid time"),
        (schedule.FormValues(message="x", when="30m", repeat="30s"), "at least 60s"),
        # A bound promises a repeat the schedule does not have.
        (schedule.FormValues(message="x", when="30m", ends="limit 3"), "bound a repeat"),
        (schedule.FormValues(message="x", when="30m", repeat="1d", ends="limit 0"), "positive"),
        (schedule.FormValues(message="x", when="30m", repeat="every"), "invalid 'every'"),
    ],
)
def test_create_refusals_are_the_builder_sentences(values, expected) -> None:
    result = schedule.create(values, [], NOW)
    assert not result.ok
    assert result.schedule is None
    assert expected in result.error
    # The sentence is what the form paints; there is no second copy.
    assert result.sentence == result.error


def test_create_refuses_a_time_older_than_the_grace_window() -> None:
    result = schedule.create(schedule.FormValues(message="x", when="2000-01-01 09:00"), [], NOW)
    assert not result.ok
    assert "in the past" in result.error


def test_create_refuses_the_seventeenth_visible_row_but_not_hidden_ones() -> None:
    full = [_schedule(f"w{i}") for i in range(1, 17)]
    refused = schedule.create(schedule.FormValues(message="x", when="in 30m"), full, NOW)
    assert not refused.ok
    assert "at most 16" in refused.error

    # Internal timers are not the user's budget: sixteen patience waits beside
    # the user's own rows must not shrink it (harness.wake's own subtraction).
    hidden = [_schedule(f"patience-{i}", kind="patience", hidden=True) for i in range(16)]
    allowed = schedule.create(schedule.FormValues(message="x", when="in 30m"), hidden, NOW)
    assert allowed.ok


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def test_edit_keeps_id_time_fired_count_and_creation_when_only_the_message_moves() -> None:
    row = _schedule("w5", message="old words", every_ms=DAY, limit=4, fired_count=2, notify=True)
    # The page sends the form's PREFILLED values, which is how "untouched" is
    # expressed: the repeat and the bound ride back exactly.
    values = schedule.prefilled(row)
    values = schedule.FormValues(
        message="new words",
        when=values.when,
        repeat=values.repeat,
        ends=values.ends,
        notify=values.notify,
    )
    result = schedule.edit(values, [row], "w5", NOW)
    assert result.ok
    edited = result.schedule
    assert edited is not None
    assert edited.id == "w5"
    assert edited.message == "new words"
    assert edited.next_due_at == row.next_due_at
    assert edited.fired_count == 2
    assert edited.created_at == row.created_at
    assert edited.every_ms == DAY
    assert edited.limit == 4
    assert edited.notify is True


def test_edit_moves_the_time_only_when_the_when_field_is_filled() -> None:
    row = _schedule("w5")
    moved = schedule.edit(schedule.FormValues(message="same", when="in 2h"), [row], "w5", NOW)
    assert moved.ok and moved.schedule is not None
    assert moved.schedule.next_due_at == NOW + 2 * 3_600_000


def test_edit_clears_the_repeat_and_the_bound_when_their_fields_are_blank() -> None:
    """Blank is a deliberate clear on EDIT: the form prefills these from the
    row, so an untouched field still says what the row already is."""
    row = _schedule("w5", every_ms=DAY, limit=4, fired_count=1)
    result = schedule.edit(
        schedule.FormValues(message="once now", repeat="", ends=""), [row], "w5", NOW
    )
    assert result.ok and result.schedule is not None
    assert result.schedule.every_ms is None
    assert result.schedule.limit is None
    assert result.schedule.until_at is None


def test_switching_the_ends_type_replaces_the_bound_instead_of_stacking_it() -> None:
    """Round-1 review F1: a limit becoming an until must not keep the limit.

    ``build_wake_edit`` treats an absent bound key as "leave that bound alone",
    so the old one survived hidden beside the visible one: the repeat still
    retired after three fires while the form showed only the new deadline.
    """
    until_at = NOW + 2 * DAY
    until_text = datetime.fromtimestamp(until_at / 1000.0).astimezone().isoformat()
    row = _schedule("w5", every_ms=DAY, limit=3, fired_count=1)
    switched = schedule.edit(
        schedule.FormValues(
            message=row.message,
            when="",
            repeat=schedule.prefilled(row).repeat,
            ends=f"until {until_text}",
        ),
        [row],
        "w5",
        NOW,
    )
    assert switched.ok and switched.schedule is not None
    assert switched.schedule.until_at == until_at
    assert switched.schedule.limit is None
    assert switched.schedule.every_ms == DAY
    # Reopening shows the bound that is actually armed, not the stale limit.
    assert schedule.prefilled(switched.schedule).ends == f"until {until_text}"

    bounded = _schedule("w6", every_ms=DAY, until_at=until_at)
    reverse = schedule.edit(
        schedule.FormValues(
            message=bounded.message,
            when="",
            repeat=schedule.prefilled(bounded).repeat,
            ends="limit 2",
        ),
        [bounded],
        "w6",
        NOW,
    )
    assert reverse.ok and reverse.schedule is not None
    assert reverse.schedule.limit == 2
    assert reverse.schedule.until_at is None


def test_an_untouched_ends_preserves_both_bounds_a_legacy_row_carries() -> None:
    """A row with BOTH bounds can only show one in the single Ends field.

    The form prefills ``limit``; leaving it untouched must therefore OMIT the
    bound request rather than clear the until the user never saw.
    """
    until_at = NOW + 4 * DAY
    row = _schedule("w7", every_ms=DAY, limit=2, until_at=until_at, fired_count=1)
    values = schedule.prefilled(row)
    assert values.ends == "limit 2"
    result = schedule.edit(
        schedule.FormValues(
            message="reworded",
            when=values.when,
            repeat=values.repeat,
            ends=values.ends,
            notify=values.notify,
        ),
        [row],
        "w7",
        NOW,
    )
    assert result.ok and result.schedule is not None
    assert result.schedule.limit == 2
    assert result.schedule.until_at == until_at


def test_edit_prefill_round_trips_the_row_into_the_form() -> None:
    until_at = NOW + 3 * DAY
    row = _schedule("w7", message="pack", next_due_at=NOW + MINUTE, every_ms=DAY, until_at=until_at)
    values = schedule.prefilled(row)
    assert values.message == "pack"
    assert values.when == ""  # blank keeps the row's instant
    assert values.repeat == format_duration(DAY)
    assert values.notify is False
    # The until instant survives a form round trip exactly (offset-bearing ISO,
    # so a DST fold cannot move it).
    result = schedule.edit(values, [row], "w7", NOW)
    assert result.ok and result.schedule is not None
    assert result.schedule.until_at == until_at

    one_shot = _schedule("w8", limit=2, fired_count=1, every_ms=DAY)
    assert schedule.prefilled(one_shot).ends == "limit 2"


def test_edit_reports_an_unknown_id_with_the_builder_sentence() -> None:
    result = schedule.edit(schedule.FormValues(message="x"), [], "w9", NOW)
    assert not result.ok
    assert "no wake schedule with id 'w9'" in result.error


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def test_rows_are_empty_for_an_empty_or_hidden_only_list() -> None:
    assert schedule.rows([], NOW) == []
    hidden = [_schedule("patience-1", kind="patience", hidden=True)]
    assert schedule.rows(hidden, NOW) == []


def test_rows_preserve_the_scheduler_order() -> None:
    first = _schedule("w2", next_due_at=NOW + DAY)
    second = _schedule("w1", next_due_at=NOW + MINUTE)
    assert [row.id for row in schedule.rows([first, second], NOW)] == ["w2", "w1"]


def test_row_text_carries_id_due_cadence_fired_and_notify() -> None:
    row = _schedule(
        "w3",
        message="check the deploy",
        next_due_at=NOW + 90 * MINUTE,
        every_ms=DAY,
        limit=3,
        fired_count=1,
        notify=True,
    )
    [built] = schedule.rows([row], NOW)
    assert built.head.startswith("w3  check the deploy")
    assert built.status == "scheduled"
    assert built.relative_due == "in 1h"  # format_age floors to one unit
    assert built.cadence == "every 1d"
    assert built.ends == "limit 3"
    assert built.fired == "1 of 3 fired"
    assert built.remaining == "2 left"
    assert built.notify == "notify"
    assert built.relative_due in built.meta
    assert built.cadence in built.meta and built.notify in built.meta
    labels = dict(built.detail)
    # The pane's headline is the message; a labelled row restating it is the
    # duplicate the first rendered frame showed.
    assert "Prompt" not in labels
    assert labels["Next"] == format_wake_time(row.next_due_at)
    assert labels["Cadence"] == "every 1d"
    assert labels["Ends"] == "limit 3"
    assert labels["Fired"] == "1 of 3"
    assert labels["Notify"] == "on"
    assert labels["Status"] == "scheduled"


def test_row_marks_a_one_shot_and_an_unbounded_repeat() -> None:
    [one_shot] = schedule.rows([_schedule("w1")], NOW)
    assert one_shot.cadence == "one-shot"
    assert one_shot.ends == ""
    assert one_shot.fired == "0 fired"
    assert one_shot.remaining == ""
    assert one_shot.notify == ""
    assert dict(one_shot.detail)["Notify"] == "off"

    until_at = NOW + 2 * DAY
    [bounded] = schedule.rows([_schedule("w2", every_ms=DAY, until_at=until_at)], NOW)
    assert bounded.ends == "until " + format_wake_time(until_at)
    assert bounded.remaining == "2d left"


def test_row_status_names_due_now_and_overdue() -> None:
    [due] = schedule.rows([_schedule("w1", next_due_at=NOW - 5_000)], NOW)
    assert due.status == "due now"
    assert due.relative_due == "due now"

    [late] = schedule.rows([_schedule("w2", next_due_at=NOW - 2 * MINUTE)], NOW)
    assert late.status == "overdue 2m"
    assert late.relative_due == "overdue 2m"


# ---------------------------------------------------------------------------
# Full-list assembly
# ---------------------------------------------------------------------------


def test_a_write_list_keeps_the_hidden_internal_timers() -> None:
    """A page write is a FULL-list write: dropping a patience wait would cancel
    a timer the user never asked about (the reason every function here takes
    and returns the scheduler's owned view)."""
    visible = [_schedule("w1"), _schedule("w2")]
    hidden = [_schedule("patience-1", kind="patience", hidden=True)]
    written = schedule.with_schedule(visible + hidden, _schedule("w3"))
    assert [row.id for row in written] == ["w1", "w2", "patience-1", "w3"]
    remaining = schedule.without_schedule(visible + hidden, "w1")
    assert [row.id for row in remaining] == ["w2", "patience-1"]


def test_with_and_without_schedule_keep_the_other_rows_in_place() -> None:
    rows = [_schedule("w1"), _schedule("w2")]
    replacement = _schedule("w2", message="reworded")
    updated = schedule.with_schedule(rows, replacement)
    assert [row.id for row in updated] == ["w1", "w2"]
    assert updated[1].message == "reworded"
    # The caller's list is not mutated; the writer gets a new view.
    assert rows[1].message == "check the build"

    appended = schedule.with_schedule(rows, _schedule("w3"))
    assert [row.id for row in appended] == ["w1", "w2", "w3"]
    assert [row.id for row in schedule.without_schedule(rows, "w1")] == ["w2"]
