"""A spooled message that asked for a turn: who raises the runtime that runs it.

THE DEFECT THESE TESTS ARE ABOUT, measured on 2026-09-21: a session retired for
a newer build with messages in its spool and NO successor. Nothing raised a
runtime, so nothing drained the spool, and the receipt the sender had been handed
(``inbox.SPOOL_RECEIPT_WAKE``, "held for the next runtime — it runs it") was true
of no process. The spool is drained only BY a runtime that exists
(``process._drain_inbox_into``), the draining runtime itself cannot start one (it
holds the transcript lease until it exits), and the one always-on process whose
job is "make a runtime exist for this session" — the wake supervisor — fired only
from the schedule index, which a spooled peer message never enters.

So the shape under test is the circuit: the spool writer records that a turn is
OWED (:mod:`local_operator.wakes.spooled`), the supervisor's due set includes it,
and the drain that empties the spool retires the record. Each half is pinned
here, and the end-to-end version (a real successor that actually exists) lives
in the reproduction rig on the PR.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from local_operator.wakes import spooled
from local_operator.wakes.supervisor import (
    _due_sessions,
    _has_fireable_wakes,
    _reconcile_spooled,
)
from local_operator.wakes.store import write_entry

NOW_MS = int(time.time() * 1000)


def _session(config_dir: Path, session_id: str, *, rows: list[dict[str, object]] | None = None):
    """A session directory, with an optional spool in it."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        with (directory / spooled.INBOX_NAME).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return directory


def _wake_row() -> dict[str, object]:
    return {"text": "check the deploy", "wake": True, "source": "peer", "written_at": 1.0}


def _note_row() -> dict[str, object]:
    return {"text": "no rush", "wake": False, "source": "peer", "written_at": 1.0}


def _owner_row() -> dict[str, object]:
    return {"text": "the prompt I typed", "wake": False, "source": "user", "written_at": 1.0}


# -- the store -----------------------------------------------------------------


def test_a_spooled_turn_round_trips(tmp_path: Path) -> None:
    config_dir = tmp_path / "store"
    assert spooled.note_spooled_turn(config_dir, "sess-a", cwd="/tmp/work") is True

    record = spooled.read_spooled_turn(config_dir, "sess-a")
    assert record is not None
    assert record["session_id"] == "sess-a"
    assert record["cwd"] == "/tmp/work"
    assert record["rows"] == 1
    assert record["attempts"] == 0
    assert spooled.read_spooled(config_dir).keys() == {"sess-a"}
    assert spooled.clear_spooled_turn(config_dir, "sess-a") is True
    assert spooled.read_spooled(config_dir) == {}


def test_a_second_row_does_not_restart_the_attempt_walk(tmp_path: Path) -> None:
    """Re-noting is not a new obligation: the cap has to stay reachable.

    A session that spools a second message while the first is still owed must not
    reset ``attempts`` — otherwise a busy session never reaches
    :data:`spooled.MAX_ATTEMPTS` and the supervisor raises it forever, which is
    the churn the cap exists to bound.
    """
    config_dir = tmp_path / "store"
    spooled.note_spooled_turn(config_dir, "sess-b")
    spooled.note_attempt(config_dir, "sess-b", error="engage failed")
    first = spooled.read_spooled_turn(config_dir, "sess-b")
    assert first is not None and first["attempts"] == 1
    assert first["next_attempt_ms"] > NOW_MS - 1000

    spooled.note_spooled_turn(config_dir, "sess-b")

    second = spooled.read_spooled_turn(config_dir, "sess-b")
    assert second is not None
    assert second["attempts"] == 1, "a second row must not reset the walk"
    assert second["next_attempt_ms"] == first["next_attempt_ms"]
    assert second["rows"] == 2, "the count of what is waiting still grows"
    assert second["noted_at_ms"] == first["noted_at_ms"]


def test_a_nameless_obligation_is_refused(tmp_path: Path) -> None:
    """An id-less record would be invisible to the reconciler and stay forever."""
    config_dir = tmp_path / "store"
    assert spooled.note_spooled_turn(config_dir, "") is False
    assert spooled.read_spooled(config_dir) == {}


def test_the_walk_stops_at_the_attempt_cap(tmp_path: Path) -> None:
    """``next_fireable_ms`` is the one place the cap and the backoff are read."""
    config_dir = tmp_path / "store"
    spooled.note_spooled_turn(config_dir, "sess-c")
    record = spooled.read_spooled_turn(config_dir, "sess-c")
    assert record is not None
    assert spooled.next_fireable_ms(record) == 0, "a fresh obligation is due now"

    for _ in range(spooled.MAX_ATTEMPTS):
        spooled.note_attempt(config_dir, "sess-c")
    exhausted = spooled.read_spooled_turn(config_dir, "sess-c")
    assert exhausted is not None
    assert exhausted["attempts"] == spooled.MAX_ATTEMPTS
    assert spooled.next_fireable_ms(exhausted) is None, "the walk is over, not merely delayed"

    # The record is KEPT when the walk gives up: the state stays legible rather
    # than being silently dropped, and the spool row is untouched, so an ordinary
    # engage still delivers it.
    assert spooled.read_spooled_turn(config_dir, "sess-c") is not None


def test_the_backoff_is_bounded_between_attempts(tmp_path: Path) -> None:
    assert spooled.backoff_s(1) == spooled.RETRY_BASE_S
    assert spooled.backoff_s(2) == spooled.RETRY_BASE_S * spooled.RETRY_FACTOR
    assert spooled.backoff_s(50) == spooled.RETRY_CAP_S


# -- what the spool says it owes ------------------------------------------------


def test_only_a_turn_asking_row_owes_a_turn(tmp_path: Path) -> None:
    """A wake row and the owner's own prompt owe one; a quiet note does not.

    The quiet note is the design's own trade (``peer_send.deliver_peer_message``:
    "``wake=False`` means 'read this on your next turn', not 'start one now'"),
    and a recall marker is not a message at all. Getting this wrong in the
    generous direction costs a runtime per note; in the mean direction it loses
    the turn.
    """
    config_dir = tmp_path / "store"
    assert spooled.spool_owes_turn(_session(config_dir, "w", rows=[_wake_row()])) is True
    assert spooled.spool_owes_turn(_session(config_dir, "u", rows=[_owner_row()])) is True
    assert spooled.spool_owes_turn(_session(config_dir, "n", rows=[_note_row()])) is False
    assert (
        spooled.spool_owes_turn(
            _session(config_dir, "r", rows=[{"source": "recall", "command_id": "c1"}])
        )
        is False
    )
    assert spooled.spool_owes_turn(
        _session(config_dir, "m", rows=[_note_row(), {"text": "ask", "wake": True}])
    ) is True, "one turn-asking row among notes is enough"
    assert spooled.spool_owes_turn(_session(config_dir, "empty", rows=[])) is False
    assert spooled.spool_owes_turn(config_dir / "sessions" / "missing") is False


def test_a_malformed_spool_row_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    config_dir = tmp_path / "store"
    directory = _session(config_dir, "junk")
    (directory / spooled.INBOX_NAME).write_text(
        'not json\n{"wake": true}\n', encoding="utf-8"
    )
    assert spooled.spool_owes_turn(directory) is True


# -- the supervisor's two halves ------------------------------------------------


def test_a_spooled_turn_is_due_the_moment_it_is_recorded(tmp_path: Path) -> None:
    """The whole point: no schedule, no due time, still fireable."""
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-d", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-d", cwd="/tmp/here")

    due = _due_sessions({}, NOW_MS, spooled=spooled.read_spooled(config_dir))

    assert due == [("sess-d", "/tmp/here", spooled.read_spooled(config_dir)["sess-d"]["noted_at_ms"])]
    assert _has_fireable_wakes({}, spooled=spooled.read_spooled(config_dir)) is True


def test_a_dormant_session_is_not_raised_for_a_spooled_turn(tmp_path: Path) -> None:
    """The kill switch outranks an owed turn, exactly as it outranks a schedule."""
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-e", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-e")
    write_entry(
        config_dir,
        "sess-e",
        cwd="/tmp",
        schedules=[{"id": "w", "message": "m", "next_due_at": NOW_MS + 60_000}],
        preserve={"stopped_at": NOW_MS},
    )

    from local_operator.wakes.store import read_index

    assert (
        _due_sessions(read_index(config_dir), NOW_MS, spooled=spooled.read_spooled(config_dir)) == []
    )


def test_a_session_due_both_ways_is_engaged_once(tmp_path: Path) -> None:
    """The index row is the more specific occurrence; the spool does not double it."""
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-f", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-f")
    index = {"sess-f": {"schedules": [{"id": "w1", "next_due_at": NOW_MS - 1}]}}

    due = _due_sessions(index, NOW_MS, spooled=spooled.read_spooled(config_dir))

    assert len(due) == 1, due
    assert due[0][0] == "sess-f"


def test_a_backing_off_turn_is_not_raised_again_yet(tmp_path: Path) -> None:
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-g", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-g")
    spooled.note_attempt(config_dir, "sess-g", error="engage failed")

    record = spooled.read_spooled(config_dir)["sess-g"]
    assert _due_sessions({}, NOW_MS, spooled={"sess-g": record}) == []

    later = int(record["next_attempt_ms"]) + 1
    assert len(_due_sessions({}, later, spooled={"sess-g": record})) == 1


def test_reconciliation_drops_an_obligation_whose_spool_is_empty(tmp_path: Path) -> None:
    """The spool is the authority: delivered rows must not keep a raise loop alive."""
    config_dir = tmp_path / "store"
    directory = _session(config_dir, "sess-h", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-h")
    kept = spooled.read_spooled(config_dir)

    _reconcile_spooled(config_dir, kept)
    assert set(kept) == {"sess-h"}, "a spool that still owes keeps its record"

    (directory / spooled.INBOX_NAME).write_text("", encoding="utf-8")
    _reconcile_spooled(config_dir, kept)

    assert kept == {}
    assert spooled.read_spooled_turn(config_dir, "sess-h") is None, "the file goes too"


def test_an_exhausted_walk_is_not_fireable(tmp_path: Path) -> None:
    """A delivery that can never succeed must not make the supervisor immortal."""
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-i", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-i")
    for _ in range(spooled.MAX_ATTEMPTS):
        spooled.note_attempt(config_dir, "sess-i")

    records = spooled.read_spooled(config_dir)

    assert _due_sessions({}, NOW_MS, spooled=records) == []
    assert _has_fireable_wakes({}, spooled=records) is False


# -- discharge ------------------------------------------------------------------


def test_the_drain_that_empties_the_spool_retires_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``inbox.settle_owed_turn``, the discharge half of the circuit."""
    from local_operator.session.runtime import inbox

    config_dir = tmp_path / "store"
    directory = _session(config_dir, "sess-j", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-j")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config_dir)

    inbox.settle_owed_turn(directory)
    assert spooled.read_spooled_turn(config_dir, "sess-j") is not None, (
        "a spool that still holds a turn-asking row keeps the record"
    )

    (directory / spooled.INBOX_NAME).write_text("", encoding="utf-8")
    inbox.settle_owed_turn(directory)

    assert spooled.read_spooled_turn(config_dir, "sess-j") is None
