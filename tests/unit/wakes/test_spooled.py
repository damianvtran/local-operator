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
from local_operator.wakes.store import read_index, write_entry
from local_operator.wakes.supervisor import (
    _due_sessions,
    _has_fireable_wakes,
    _note_spooled_attempt,
    _reconcile_spooled,
)

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


def test_a_second_row_pulls_the_wait_back_without_resetting_the_walk(tmp_path: Path) -> None:
    """A fresh row is fresh work: it must not wait behind an hour-old backoff.

    ROUND 1 CHANGED THIS RULE (R1-2). The record keeps its ``attempts`` — a busy
    session must not get a new walk per message — but the ``next_attempt_ms`` it
    inherits belongs to the walk that previous rows started, and the receipt the
    new sender is about to be handed says a runtime will run their message. So the
    wait is pulled back to at most :data:`spooled.RETRY_BASE_S`, never pushed out,
    and never *restarted* outright: the attempts Figure stays where the walk had
    got to, which is what keeps the churn bounded when a session keeps failing.
    """
    config_dir = tmp_path / "store"
    spooled.note_spooled_turn(config_dir, "sess-b")
    for _ in range(4):
        spooled.note_attempt(config_dir, "sess-b", error="engage failed")
    first = spooled.read_spooled_turn(config_dir, "sess-b")
    assert first is not None and first["attempts"] == 4
    assert first["next_attempt_ms"] > NOW_MS + 100_000, "four failures walk to a long wait"

    spooled.note_spooled_turn(config_dir, "sess-b")

    second = spooled.read_spooled_turn(config_dir, "sess-b")
    assert second is not None
    assert second["attempts"] == 4, "the walk is not restarted by a later row"
    assert second["rows"] == 2, "the count of what is waiting still grows"
    assert second["noted_at_ms"] == first["noted_at_ms"], "the oldest row still leads the queue"
    ceiling = int(time.time() * 1000) + spooled.RETRY_BASE_S * 1000 + 2_000
    assert (
        second["next_attempt_ms"] <= ceiling
    ), "the new row's runtime is owed promptly, not after the old walk's hour"


def test_the_walk_never_ends(tmp_path: Path) -> None:
    """THERE IS NO ATTEMPT CAP, and that is round 1's finding (R1-1, R1-2).

    A capped walk gave up permanently. Two shapes then had no successor at all:
    a handover whose owner outlived the cap (a slow clean exit, or the wedged
    owner this change is about) and a row spooled after the walk ended. The walk
    therefore keeps going at the backoff ceiling — what bounds it is the spool
    emptying, the authority, not a counter.
    """
    config_dir = tmp_path / "store"
    spooled.note_spooled_turn(config_dir, "sess-c")
    record = spooled.read_spooled_turn(config_dir, "sess-c")
    assert record is not None
    assert spooled.next_attempt_at_ms(record) == 0, "a fresh obligation is due now"

    for _ in range(50):
        spooled.note_attempt(config_dir, "sess-c")

    walked = spooled.read_spooled_turn(config_dir, "sess-c")
    assert walked is not None
    assert walked["attempts"] == 50
    now = int(time.time() * 1000)
    later = spooled.next_attempt_at_ms(walked)
    assert later > now, "the walk is throttled to its ceiling, not stopped"
    assert later <= now + spooled.RETRY_CAP_S * 1000 + 2_000, "…and the ceiling holds"
    assert (
        _due_sessions({}, later + 1, spooled={"sess-c": walked}) != []
    ), "an hour later it is tried again rather than abandoned"


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
    assert (
        spooled.spool_owes_turn(
            _session(config_dir, "m", rows=[_note_row(), {"text": "ask", "wake": True}])
        )
        is True
    ), "one turn-asking row among notes is enough"
    assert spooled.spool_owes_turn(_session(config_dir, "empty", rows=[])) is False
    assert spooled.spool_owes_turn(config_dir / "sessions" / "missing") is False


def test_a_malformed_spool_row_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    config_dir = tmp_path / "store"
    directory = _session(config_dir, "junk")
    (directory / spooled.INBOX_NAME).write_text('not json\n{"wake": true}\n', encoding="utf-8")
    assert spooled.spool_owes_turn(directory) is True


# -- the supervisor's two halves ------------------------------------------------


def test_a_spooled_turn_is_due_the_moment_it_is_recorded(tmp_path: Path) -> None:
    """The whole point: no schedule, no due time, still fireable."""
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-d", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-d", cwd="/tmp/here")

    due = _due_sessions({}, NOW_MS, spooled=spooled.read_spooled(config_dir))

    recorded = spooled.read_spooled(config_dir)
    assert due == [("sess-d", "/tmp/here", recorded["sess-d"]["noted_at_ms"])]
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

    due = _due_sessions(read_index(config_dir), NOW_MS, spooled=spooled.read_spooled(config_dir))

    assert due == []


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


def test_a_backed_off_walk_is_still_fireable_work(tmp_path: Path) -> None:
    """An owed turn is work, so the resident supervisor stays up to do it.

    That is the deliberate price of dropping the cap (see :mod:`spooled`): one
    boot an hour past the ceiling, not a loop, and the process stops once the
    spool stops owing.
    """
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-i", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-i")
    for _ in range(6):
        spooled.note_attempt(config_dir, "sess-i")

    records = spooled.read_spooled(config_dir)

    assert _has_fireable_wakes({}, spooled=records) is True


def test_a_dormant_session_is_not_fireable_even_with_a_spooled_turn(tmp_path: Path) -> None:
    """R1-6: the kill switch has to reach BOTH doors into ``_has_fireable_wakes``.

    ``_due_sessions`` already refuses to fire a stopped session's spooled turn;
    if the retirement predicate disagreed, a stopped session with an owed turn
    would keep the supervisor resident forever, waiting to fire something it will
    never fire — the leak that function's own docstring says it was written to
    close for dormant index entries.
    """
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-z", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-z")
    write_entry(
        config_dir,
        "sess-z",
        cwd="/tmp",
        schedules=[{"id": "w", "message": "m", "next_due_at": NOW_MS + 60_000}],
        preserve={"stopped_at": NOW_MS},
    )
    index = read_index(config_dir)

    assert _due_sessions(index, NOW_MS, spooled=spooled.read_spooled(config_dir)) == []
    assert (
        _has_fireable_wakes(index, config_dir=config_dir, spooled=spooled.read_spooled(config_dir))
        is False
    )


def test_a_corrupt_record_cannot_raise_out_of_the_sweep(tmp_path: Path) -> None:
    """R1-5: the store is written by other processes and read on the hot path.

    ``_due_sessions``' own contract for the index ("a hand-edited or half-written
    entry must cost one session's wake, never the whole sweep") applies to this
    store for the same reason, so a non-numeric field reads as DUE rather than
    raising ``ValueError`` out of ``sweep``/``serve``.
    """
    junk = {"schema": 1, "session_id": "sess-junk", "attempts": "many", "next_attempt_ms": "soon"}

    assert spooled.next_attempt_at_ms(junk) == 0
    assert _due_sessions({}, NOW_MS, spooled={"sess-junk": junk}) != []
    assert _has_fireable_wakes({}, spooled={"sess-junk": junk}) is True


def test_only_a_tried_raise_moves_the_walk(tmp_path: Path) -> None:
    """R1-1, at the ledger: a SERVED session is not a failed attempt.

    ``live`` means the handover has not begun — nothing was tried — and counting
    those refused passes ended the walk ≈7.75 min after the record was written,
    which is defect (B) returning through the bookkeeping. ``wedged`` and
    ``failed`` are the shapes where a raise really did not happen and repeating it
    immediately is pointless.
    """
    config_dir = tmp_path / "store"
    _session(config_dir, "sess-live", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-live")

    _note_spooled_attempt(config_dir, "sess-live", reason="live")
    _note_spooled_attempt(config_dir, "sess-live", reason="started")
    _note_spooled_attempt(config_dir, "sess-live", reason="ghost")
    untouched = spooled.read_spooled_turn(config_dir, "sess-live")
    assert untouched is not None and untouched["attempts"] == 0
    assert spooled.next_attempt_at_ms(untouched) == 0, "a first row is due immediately"

    _note_spooled_attempt(config_dir, "sess-live", reason="wedged")
    counted = spooled.read_spooled_turn(config_dir, "sess-live")
    assert counted is not None and counted["attempts"] == 1
    assert counted["last_error"] == "wedged"


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
    assert (
        spooled.read_spooled_turn(config_dir, "sess-j") is not None
    ), "a spool that still holds a turn-asking row keeps the record"

    (directory / spooled.INBOX_NAME).write_text("", encoding="utf-8")
    inbox.settle_owed_turn(directory)

    assert spooled.read_spooled_turn(config_dir, "sess-j") is None


# -- the wire: the walk is only as good as the call site -------------------------
#
# ROUND 2'S MAJOR (R2-1) LIVED HERE AND NOWHERE ELSE. `_engage_one` grew an
# `on_outcome` callback in round 1 and the one caller that needs it never passed
# it, so every outcome word was dropped, `note_attempt` was never called, and a
# session the supervisor could not raise was re-engaged on every slice. No cell
# could see it: the walk's own cells call `_note_spooled_attempt` directly. These
# two drive `_Sweeper.engage` itself, which is the seam that was broken.


@pytest.mark.asyncio
async def test_the_sweeper_hands_every_outcome_to_the_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real engagement outcome reaches the store, THROUGH the sweeper."""
    from local_operator.wakes import supervisor

    config_dir = tmp_path / "store"
    _session(config_dir, "sess-wired", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-wired")
    fresh = spooled.read_spooled_turn(config_dir, "sess-wired")
    assert fresh is not None and fresh["attempts"] == 0

    seen: dict[str, object] = {}

    async def _fake_engage_one(
        cfg: Path,
        session_id: str,
        cwd: str,
        due_ms: int,
        moment: int,
        semaphore: object,
        *,
        on_outcome=None,
    ) -> bool:
        seen["on_outcome"] = on_outcome
        assert on_outcome is not None, "the sweeper must pass the walk's callback"
        on_outcome("wedged")  # what a wedged owner produces
        return False

    monkeypatch.setattr(supervisor, "_engage_one", _fake_engage_one)

    sweeper = supervisor._Sweeper()
    sweeper.engage(config_dir, "sess-wired", "/tmp", 0, NOW_MS)
    for task in list(sweeper._in_flight.values()):
        await task

    assert seen["on_outcome"] is not None
    counted = spooled.read_spooled_turn(config_dir, "sess-wired")
    assert counted is not None and counted["attempts"] == 1, (
        "an outcome the sweeper was handed must move the walk, or the churn bound "
        "the store documents does not exist at its call site"
    )
    assert counted["last_error"] == "wedged"


@pytest.mark.asyncio
async def test_an_unanticipated_raise_from_the_engage_is_still_an_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The belt is where a raise that never reported lands, so it must count."""
    from local_operator.wakes import supervisor

    config_dir = tmp_path / "store"
    _session(config_dir, "sess-belt", rows=[_wake_row()])
    spooled.note_spooled_turn(config_dir, "sess-belt")

    async def _boom(*_args: object, **_kwargs: object) -> bool:
        raise RuntimeError("engage exploded before it could report")

    monkeypatch.setattr(supervisor, "_engage_one", _boom)

    sweeper = supervisor._Sweeper()
    sweeper.engage(config_dir, "sess-belt", "/tmp", 0, NOW_MS)
    for task in list(sweeper._in_flight.values()):
        await task

    counted = spooled.read_spooled_turn(config_dir, "sess-belt")
    assert counted is not None and counted["attempts"] == 1
    assert counted["last_error"] == "raised", (
        "the belt's word must not claim the engage failed: it may have started a "
        "runtime before it raised (round 3, R3-1)"
    )


def test_dropping_a_deferral_still_judges_the_record_it_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-Q2: the third unlink passes the same compare-and-delete as its siblings.

    Checked at the call rather than through a race the isolated rig cannot build
    (the deferral runs pre-socket): the drop reads the record, then asks for the
    value it read to still be there.
    """
    from local_operator.session.runtime import inbox
    from local_operator.wakes import spooled as store

    config_dir = tmp_path / "store"
    directory = _session(config_dir, "sess-cas", rows=[_wake_row()])
    store.note_spooled_turn(config_dir, "sess-cas")
    judged = store.read_spooled_turn(config_dir, "sess-cas")
    assert judged is not None, "the fixture must have written a record"
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config_dir)

    seen: dict[str, int | None] = {}
    real_clear = store.clear_spooled_turn

    def _spy(root: Path, session_id: str, *, expected_updated_at_ms: int | None = None) -> bool:
        seen["expected_updated_at_ms"] = expected_updated_at_ms
        return real_clear(root, session_id, expected_updated_at_ms=expected_updated_at_ms)

    # ``drop_owed_turn`` imports the name function-locally, so patching the module
    # attribute is what the production call resolves.
    monkeypatch.setattr(store, "clear_spooled_turn", _spy)
    inbox.drop_owed_turn(directory)

    assert (
        seen.get("expected_updated_at_ms") == judged["updated_at_ms"]
    ), "the drop must not unlink a record on a judgement it cannot still see"


def test_dropping_a_deferral_spares_the_owners_own_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-2: the drop refuses when the spool holds the OWNER's own words.

    A ``SOURCE_USER`` row is dischargeable by exactly the raise the deferral is
    declining — the successor's boot drain RUNS it, durable history or not — so a
    record that arrived for one must survive; the deferral judgement is about the
    peer rows beside it. Pinned because it is the one guard this round added whose
    removal nothing else noticed (round 3 proved that by mutation).
    """
    from local_operator.session.runtime import inbox

    config_dir = tmp_path / "store"
    directory = _session(config_dir, "sess-owner-row", rows=[_owner_row(), _note_row()])
    spooled.note_spooled_turn(config_dir, "sess-owner-row")
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: config_dir)
    assert spooled.spool_has_owner_row(directory) is True

    inbox.drop_owed_turn(directory)

    assert (
        spooled.read_spooled_turn(config_dir, "sess-owner-row") is not None
    ), "the owner's message is deliverable by a raise; its record must survive"


def test_the_spool_stores_agree_on_what_the_owners_words_are(tmp_path: Path) -> None:
    """R3-3: two modules, one token, and a rename must not desynchronise them.

    ``spooled.SOURCE_USER`` and ``inbox.SOURCE_USER`` are the same wire value
    spelled in two places, and ``spool_owes_turn``/``spool_has_owner_row`` decide
    whether a message has an owner by comparing against it. A rename on one side
    alone would fail in the UNSAFE direction: rows would stop counting as
    turn-asking, and the message would lose its owner silently.
    """
    from local_operator.session.runtime.inbox import SOURCE_USER as INBOX_SOURCE_USER

    assert spooled.SOURCE_USER == INBOX_SOURCE_USER == "user"
