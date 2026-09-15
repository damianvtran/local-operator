"""The supervisor's delivery ledger: what it records, for how long, and when it clears.

The property under test throughout is that a fire which could not be handed to a
runtime is OWED — recorded durably, retried with a bounded backoff, and readable
by the CLI — rather than being a WARNING line and nothing else.

The ledger is DERIVED bookkeeping: it is about the supervisor's own delivery
attempts, never about what a session's schedules are, so the failure modes it
has to survive are the index's own (a torn file, a foreign schema, a stale
record whose occurrence has since fired) plus the deliverability failure the
whole mechanism exists for (a filesystem that will not take the write must not
stop the engage, which is the thing that fires the wake).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from local_operator.wakes import deliveries

NOW_MS = int(time.time() * 1000)


def test_a_failed_attempt_is_recorded_with_a_bounded_backoff(tmp_path: Path) -> None:
    record = deliveries.note_failure(
        tmp_path, "owedsession1", NOW_MS - 5_000, error="could not reach a runtime", now_ms=NOW_MS
    )

    assert record is not None, "a failed attempt on a writable store must be recorded"
    assert record["state"] == deliveries.STATE_RETRYING
    assert record["attempts"] == 1
    assert record["first_attempt_ms"] == NOW_MS
    assert record["last_attempt_ms"] == NOW_MS
    assert record["next_attempt_ms"] == NOW_MS + int(deliveries.RETRY_BASE_S * 1000)
    # The record is readable back, which is what the sweep and the CLI both do.
    assert deliveries.read_delivery(tmp_path, "owedsession1") == record
    assert deliveries.read_deliveries(tmp_path) == {"owedsession1": record}


def test_attempts_accumulate_and_the_wait_grows_to_the_cap(tmp_path: Path) -> None:
    """The cap is the load-bearing half: it is what keeps a permanently failing
    session from spending a full engage deadline out of the fleet's slots on
    every pass — but it must never grow past the point where a late wake is
    indistinguishable from a lost one."""
    previous = 0
    for attempt in range(1, 9):
        record = deliveries.note_failure(
            tmp_path, "owedsession1", NOW_MS, error="still unreachable", now_ms=NOW_MS
        )
        assert record is not None
        assert record["attempts"] == attempt
        wait_s = (record["next_attempt_ms"] - NOW_MS) / 1000.0
        assert wait_s >= previous
        assert wait_s <= deliveries.RETRY_CAP_S
        previous = wait_s
    # Monotonic up to the cap, and flat at it.
    assert previous == deliveries.RETRY_CAP_S
    assert deliveries.backoff_s(1) == deliveries.RETRY_BASE_S
    assert deliveries.backoff_s(50) == deliveries.RETRY_CAP_S


def test_a_record_is_read_as_absent_when_it_is_unreadable_or_foreign(tmp_path: Path) -> None:
    path = deliveries.delivery_path(tmp_path, "owedsession1")
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text("{not json", encoding="utf-8")
    assert deliveries.read_delivery(tmp_path, "owedsession1") is None

    path.write_text(json.dumps({"schema": 99, "occurrence_ms": 1}), encoding="utf-8")
    assert deliveries.read_delivery(tmp_path, "owedsession1") is None

    # A record with no occurrence cannot name a fire, so it is not one.
    path.write_text(json.dumps({"schema": deliveries.DELIVERY_SCHEMA}), encoding="utf-8")
    assert deliveries.read_delivery(tmp_path, "owedsession1") is None

    assert deliveries.read_deliveries(tmp_path) == {}


def test_the_undelivered_state_is_a_report_and_not_a_surrender(tmp_path: Path) -> None:
    """The fire keeps being retried past this threshold.

    Crossing it must change what the OPERATOR is told (an ERROR, and a line on
    ``lop wake status``), not whether the supervisor keeps trying: giving up
    permanently is the defect this module exists to remove.
    """
    last = None
    for _ in range(deliveries.UNDELIVERED_AFTER_ATTEMPTS):
        last = deliveries.note_failure(
            tmp_path, "owedsession1", NOW_MS, error="unreachable", now_ms=NOW_MS
        )
    assert last is not None
    assert last["attempts"] == deliveries.UNDELIVERED_AFTER_ATTEMPTS
    assert last["state"] == deliveries.STATE_UNDELIVERED
    assert last["next_attempt_ms"] > NOW_MS, "an undelivered fire must still have a next attempt"


def test_a_record_for_a_different_occurrence_restarts_the_run(tmp_path: Path) -> None:
    """The count is CONSECUTIVE failures for one occurrence. A recurrence the
    supervisor moves on to is a different fire, and inheriting the old count
    would mark a fresh wake undelivered on its first failure."""
    deliveries.note_failure(tmp_path, "owedsession1", NOW_MS, error="one", now_ms=NOW_MS)
    record = deliveries.note_failure(
        tmp_path, "owedsession1", NOW_MS + 60_000, error="two", now_ms=NOW_MS
    )

    assert record is not None
    assert record["attempts"] == 1
    assert record["occurrence_ms"] == NOW_MS + 60_000


def test_delivering_clears_the_record_and_reports_the_attempts_it_took(tmp_path: Path) -> None:
    deliveries.note_failure(tmp_path, "owedsession1", NOW_MS, error="unreachable", now_ms=NOW_MS)
    deliveries.note_failure(tmp_path, "owedsession1", NOW_MS, error="unreachable", now_ms=NOW_MS)

    assert deliveries.note_delivered(tmp_path, "owedsession1", NOW_MS) == 2
    assert deliveries.read_delivery(tmp_path, "owedsession1") is None
    # Idempotent, and a first-try success reports zero failed attempts.
    assert deliveries.note_delivered(tmp_path, "owedsession1", NOW_MS) == 0


def test_a_failed_write_is_not_an_exception(tmp_path: Path) -> None:
    """The ledger is observability and retry bookkeeping; the ENGAGE is what
    fires the wake. A filesystem that will not take the record must never turn
    a wake that could have fired into one that did not."""
    blocker = tmp_path / "wakes"
    blocker.write_text("not a directory", encoding="utf-8")

    # No raise, and nothing recorded — the caller carries on.
    assert deliveries.read_deliveries(tmp_path) == {}
    assert deliveries.write_delivery(tmp_path, "owedsession1", {"occurrence_ms": 1}) is None


def test_no_staged_temp_file_is_left_behind(tmp_path: Path) -> None:
    deliveries.note_failure(tmp_path, "owedsession1", NOW_MS, error="unreachable", now_ms=NOW_MS)

    names = sorted(os.listdir(deliveries.deliveries_dir(tmp_path)))
    assert names == ["owedsession1.json"], names


def test_note_failure_reports_a_refused_write(tmp_path: Path) -> None:
    """Review round 1, MINOR 2: the caller must be able to tell that it failed.

    ``note_failure`` used to answer with the in-memory record whether or not the
    write landed, so a supervisor on an unwritable store escalated to the
    "STILL OWED … 'lop wake status' reports it" line, which was a lie in exactly
    the case the durability work exists for. ``None`` is that answer.
    """
    blocker = tmp_path / "wakes"
    blocker.write_text("not a directory", encoding="utf-8")

    refused = deliveries.note_failure(
        tmp_path, "owedsession1", NOW_MS, error="unreachable", now_ms=NOW_MS
    )
    assert refused is None
