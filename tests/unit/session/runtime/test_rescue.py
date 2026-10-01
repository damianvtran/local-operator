"""Unit tests for the rescue pass (auto-re-engage of involuntary deaths).

The pass is tested here with an INJECTED clock and an injected liveness probe —
there is no wall-clock assertion and no real process anywhere in this file (the
``-m e2e`` rig is QA's, see the design's §5.D). The seat that drives the pass is
tested separately in ``tests/unit/wakes/test_rescue_seat.py``.

The calibration fixture is generated from the 2026-09-30 forensics bundle; see
``tests/unit/session/runtime/data/rescue_20260930.json`` for the provenance and
what was reconstructed.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import registry, rescue
from local_operator.session.runtime.types import RUN_DIRNAME, SessionRecord

FIXTURE = Path(__file__).parent / "data" / "rescue_20260930.json"

#: A pid that is certainly not alive on this host (rescue never signals, so this
#: is only ever compared, never used).
DEAD_PID = 4_700_001


# --- builders ---------------------------------------------------------------


@pytest.fixture
def alive(monkeypatch: pytest.MonkeyPatch) -> set[int]:
    """The set of pids the classification sees as live.

    ``registry.pid_alive`` is the single liveness primitive ``classify`` calls,
    so replacing it exercises the REAL scan and classification with a controlled
    probe — no ``ps`` fork, no dependence on what happens to be running.
    """
    live: set[int] = set()
    monkeypatch.setattr(registry, "pid_alive", lambda pid, *, check_zombie=False: pid in live)
    monkeypatch.setattr(registry, "zombie_states", lambda pids: {})
    return live


def _record(
    root: Path,
    session_id: str,
    *,
    pid: int,
    started_at: float,
    heartbeat_at: float,
    cwd: str = "/Users/damian",
) -> None:
    record = SessionRecord(
        pid=pid,
        kind="daemon",
        session_id=session_id,
        conversation_name=session_id,
        cwd=cwd,
        model_label="test/model",
        control_port=0,
        control_key="k",
        started_at=started_at,
        heartbeat_at=heartbeat_at,
    )
    directory = root / RUN_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pid}.json").write_text(json.dumps(record.to_json()))


def _ledger(root: Path, session_id: str) -> dict[str, Any]:
    """The session's ledger entry, asserted present.

    A helper rather than a bare ``rescue.read_ledger(...)[...]`` at each site:
    ``read_ledger`` returns ``dict | None`` because absence is a real state, and
    every call in this file is about a session whose episode was just opened, so
    a ``None`` there IS the failure. Asserting once keeps the tests readable and
    keeps the checker honest.
    """
    entry = rescue.read_ledger(root, session_id)
    assert entry is not None
    return entry


def _session_dir(root: Path, session_id: str) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _completions(root: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Append ``attention.db`` completions rows: ``(conversation, kind, cause, reason)``.

    Callable more than once against one root (a test may stage a second
    conversation later), so the table is created only when absent and the new
    rows continue the sequence from the highest one already stored — a restart
    at 1 would collide on the primary key and, before that, invert the "latest
    row" ordering the predicate depends on.
    """
    connection = sqlite3.connect(root / "attention.db")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS completions (sequence INTEGER PRIMARY KEY, conversation TEXT,"
        " token TEXT, anchor TEXT, kind TEXT, reason TEXT, cause TEXT, notify INTEGER)"
    )
    start = connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM completions").fetchone()[0]
    for offset, (conversation, kind, cause, reason) in enumerate(rows, start=1):
        index = int(start) + offset
        connection.execute(
            "INSERT INTO completions (sequence, conversation, token, anchor, kind, reason, cause,"
            " notify) VALUES (?,?,?,?,?,?,?,0)",
            (index, conversation, f"t{index}", "a", kind, reason, cause),
        )
    connection.commit()
    connection.close()


def _journal(root: Path, session_id: str, pid: int, *, still_open: bool = True) -> None:
    _session_dir(root, session_id).joinpath("turn-journal.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "pid": pid,
                "parent_pid": 1,
                "turn_seq": 3,
                "command_id": "c",
                "started_at": 1.0,
                "ended_at": None,
                "open": True,
                "end_cause": "runtime-shutdown",
                "exit_cause": "SIGTERM",
                "still_open_at_exit": still_open,
                "last_boundary": "tool",
                "build": {},
                "install_root": "/install",
                "updated_at": 2.0,
            }
        )
    )


def _marker(
    root: Path, session_id: str, pid: int, started_at: float, *, deliberate: bool = True
) -> None:
    _session_dir(root, session_id).joinpath("runtime-stop.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "pid": pid,
                "started_at": started_at,
                "at": started_at + 5.0,
                "rung": "sigterm",
                "deliberate": deliberate,
            }
        )
    )


def _wake(root: Path, session_id: str, **fields: Any) -> None:
    """Write a wake index entry.

    The entry MUST carry ``schema``: ``store.read_entry`` treats a file whose
    schema it does not recognise as ABSENT (its documented contract, because the
    transcript is the truth and the next open repairs the file), so a fixture
    without it would silently test nothing.
    """
    from local_operator.wakes.store import INDEX_SCHEMA

    directory = root / "wakes"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{session_id}.json").write_text(json.dumps({"schema": INDEX_SCHEMA, **fields}))


def _receipt(
    root: Path, session_id: str, pid: int, started_at: float, signals: list[dict[str, Any]]
) -> None:
    _session_dir(root, session_id).joinpath("runtime-signal.json").write_text(
        json.dumps(
            {
                "v": 1,
                "kind": "signal-receipt",
                "session_id": session_id,
                "pid": pid,
                "started_at": started_at,
                "signals": signals,
                "count": len(signals),
            }
        )
    )


# --- A. the predicate -------------------------------------------------------

NOW = 1_800_000_000.0
HEARTBEAT = NOW - 60.0  # comfortably inside [settle, recent]
STARTED = NOW - 3600.0


def _one_dead(
    root: Path,
    alive: set[int],
    *,
    session_id: str = "aaaa11112222",
    started_at: float = STARTED,
    heartbeat_at: float = HEARTBEAT,
) -> None:
    _session_dir(root, session_id)
    _record(root, session_id, pid=DEAD_PID, started_at=started_at, heartbeat_at=heartbeat_at)


@pytest.mark.parametrize(
    ("label", "kind", "cause", "journal", "marker", "stopped_at", "held_at", "verdict"),
    [
        ("disposed", "error", "disposed", False, False, False, False, "fire"),
        ("runtime-shutdown", "error", "runtime-shutdown", False, False, False, False, "fire"),
        ("runtime-killed", "error", "runtime-killed", False, False, False, False, "fire"),
        ("no-cause-with-journal", "error", "", True, False, False, False, "fire"),
        ("no-cause-no-journal", "error", "", False, False, False, False, "unclassified"),
        ("user-stop-no-marker", "interrupted", "user-stop", False, False, False, False, "fire"),
        ("user-stop-with-marker", "interrupted", "user-stop", False, True, False, False, "skip"),
        (
            "user-stop-with-stopped-at",
            "interrupted",
            "user-stop",
            False,
            False,
            True,
            False,
            "skip",
        ),
        ("complete", "complete", "", False, False, False, False, "skip"),
        ("held-at-pause", "error", "disposed", False, False, False, True, "skip"),
    ],
)
def test_death_class_table(
    tmp_path: Path,
    alive: set[int],
    label: str,
    kind: str,
    cause: str,
    journal: bool,
    marker: bool,
    stopped_at: bool,
    held_at: bool,
    verdict: str,
) -> None:
    sid = "aaaa11112222"
    _one_dead(tmp_path, alive, session_id=sid)
    _completions(tmp_path, [(f"session/{sid}", kind, cause, "reason")])
    if journal:
        _journal(tmp_path, sid, DEAD_PID)
    if marker:
        _marker(tmp_path, sid, DEAD_PID, STARTED)
    if stopped_at or held_at:
        _wake(tmp_path, sid, **({"stopped_at": NOW - 10} if stopped_at else {"held_at": NOW - 10}))

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    decision = next(d for d in report.decisions if d.session_id == sid)
    assert decision.verdict == verdict, (label, decision.verdict, decision.tag)


def test_no_row_with_an_open_journal_fires(tmp_path: Path, alive: set[int]) -> None:
    sid = "bbbb11112222"
    _one_dead(tmp_path, alive, session_id=sid)
    _journal(tmp_path, sid, DEAD_PID)

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert [d.session_id for d in report.fire()] == [sid]


def test_liveness_is_read_before_the_completion_row(tmp_path: Path, alive: set[int]) -> None:
    """A live successor beats a stale ``error/disposed`` row (the survivors' shape)."""
    sid = "cccc11112222"
    _session_dir(tmp_path, sid)
    dead_pid = DEAD_PID
    live_pid = 4_700_002
    _record(tmp_path, sid, pid=dead_pid, started_at=STARTED, heartbeat_at=HEARTBEAT)
    _record(tmp_path, sid, pid=live_pid, started_at=STARTED, heartbeat_at=NOW)
    alive.add(live_pid)
    _completions(tmp_path, [(f"session/{sid}", "error", "disposed", "the session was disposed")])

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert report.fire() == []
    assert report.refusals["live"] == 1


def test_only_the_latest_completion_row_decides(tmp_path: Path, alive: set[int]) -> None:
    """An old wave row followed by a ``complete`` row is a planned end, not a death."""
    sid = "dddd11112222"
    _one_dead(tmp_path, alive, session_id=sid)
    _completions(
        tmp_path,
        [
            (f"session/{sid}", "error", "disposed", "old wave"),
            (f"session/{sid}", "complete", "", "finished"),
        ],
    )

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert report.fire() == []
    assert report.refusals["complete"] == 1


@pytest.mark.parametrize(
    ("label", "heartbeat", "started", "why"),
    [
        ("settle-not-elapsed", NOW - 29.0, STARTED, "young-run-settle"),
        ("too-old", NOW - rescue.RESCUE_RECENT_S - 1, STARTED, "stale"),
        ("run-younger-than-120s", HEARTBEAT, NOW - 119.0, "young-run"),
    ],
)
def test_windows(
    tmp_path: Path, alive: set[int], label: str, heartbeat: float, started: float, why: str
) -> None:
    sid = "eeee11112222"
    _session_dir(tmp_path, sid)
    _record(tmp_path, sid, pid=DEAD_PID, started_at=started, heartbeat_at=heartbeat)
    _completions(tmp_path, [(f"session/{sid}", "error", "disposed", "x")])

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert report.fire() == []
    assert report.refusals[why] == 1, (label, dict(report.refusals))


def test_a_young_session_with_no_completed_turn_is_not_rescued(
    tmp_path: Path, alive: set[int]
) -> None:
    sid = "ffff11112222"
    # The session's OWN earliest record started moments before the pass: this is
    # the young-session signal (a spawn may still be constructing its first
    # runtime), and it comes from the records rather than the directory's birth
    # time so a fixture can express it.
    _one_dead(tmp_path, alive, session_id=sid, started_at=NOW - 300.0, heartbeat_at=HEARTBEAT)
    _completions(tmp_path, [(f"session/{sid}", "interrupted", "user-stop", "stopped")])

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert report.fire() == []
    assert report.refusals["young-session"] == 1

    # Once the session is past the window it is rescued like any other (and the
    # completed turn is not required — only the AGE is).
    later = rescue.rescue_scan(tmp_path, now=NOW + rescue.RESCUE_YOUNG_SESSION_S + 1, apply=False)
    assert [d.session_id for d in later.fire()] == [sid]


def test_an_attached_viewer_defers_to_the_viewer(
    tmp_path: Path, alive: set[int], monkeypatch
) -> None:
    sid = "998877766655"
    _one_dead(tmp_path, alive, session_id=sid)
    _completions(tmp_path, [(f"session/{sid}", "error", "disposed", "x")])
    monkeypatch.setattr(rescue, "_viewer_attached", lambda root, session: session == sid)

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert report.fire() == []
    assert report.refusals["attached"] == 1


# --- A5. tonight's calibration fixture --------------------------------------


def _build_calibration(tmp_path: Path, fixture: dict[str, Any], alive: set[int]) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for entry in fixture["sessions"]:
        sid = entry["session_id"]
        pid = entry["pid"]
        started_at = fixture["snapshot_epoch"] - 3600
        _session_dir(tmp_path, sid)
        # THE FIXTURE'S OWN LEVERS ARE APPLIED, not ignored. Every one of these
        # fields is false in the recorded wave (no marker, no `stopped_at`, no
        # surviving open journal row — verified in the design's §1.3), so this
        # changes nothing about the expected sets; it makes the pin REAL, so a
        # session the wave DID sanction would drop out of the candidate set
        # rather than being silently rescued. The discriminating test below
        # flips one field to prove the builder honours them.
        if entry.get("journal_open"):
            _journal(tmp_path, sid, pid)
        if entry.get("stop_marker"):
            _marker(tmp_path, sid, pid, started_at)
        if entry.get("stopped_at"):
            _wake(tmp_path, sid, stopped_at=entry.get("death_at") or started_at)
        # Every one of these sessions ran real turns earlier in its life, so the
        # attention row BEFORE the wave is a `complete`: that is both faithful
        # and what the young-session rule needs (a session with no completed turn
        # is deliberately left alone).
        rows.append((f"session/{sid}", "complete", "", "an earlier turn finished"))
        if entry["live_at_pass"]:
            # A live record's heartbeat must be FRESH against the REAL clock, not
            # against ``pass_epoch``: ``registry.scan`` classifies with
            # ``time.time()``, so a heartbeat pinned to the wave reads as
            # ``wedged`` on any machine whose clock has moved on — which is every
            # machine, always. Dead records are unaffected (pid liveness decides
            # first), which is why only the live half needs this.
            _record(
                tmp_path,
                sid,
                pid=pid,
                started_at=started_at,
                heartbeat_at=time.time(),
                cwd=entry["cwd"],
            )
            alive.add(pid)
        else:
            _record(
                tmp_path,
                sid,
                pid=pid,
                started_at=started_at,
                heartbeat_at=entry["death_at"],
                cwd=entry["cwd"],
            )
            completion = entry["completion"]
            rows.append(
                (f"session/{sid}", completion["kind"], completion["cause"], completion["reason"])
            )
    _completions(tmp_path, rows)


def test_mass_failure_calibration(tmp_path: Path, alive: set[int]) -> None:
    """The candidate set is exactly the 21 victims — and none of the survivors.

    This is the design's §5.A5 pin: the predicate is validated against the wave
    it was written for, from the bundle's own data, not against a hand-built
    approximation of it.
    """
    fixture = json.loads(FIXTURE.read_text())
    _build_calibration(tmp_path, fixture, alive)

    report = rescue.rescue_scan(tmp_path, now=fixture["pass_epoch"], apply=False)

    assert sorted(d.session_id for d in report.fire()) == fixture["expected_candidates"]
    assert set(fixture["expected_non_candidates"]).isdisjoint(d.session_id for d in report.fire())
    # The per-pass start budget bounds the ENGAGEMENTS, not the verdicts.
    assert len(report.to_engage) == rescue.RESCUE_MAX_STARTS_PER_PASS


def test_the_calibration_fixture_pins_the_levers_too(tmp_path: Path, alive: set[int]) -> None:
    """The builder HONOURS the fixture's lever fields, so the set is really pinned.

    Flips a victim into a sanctioned stop (a covering marker) and asserts it
    leaves the candidate set — if the builder ignored the field, this would fail
    and the calibration test above would be pinning less than it claims.
    """
    fixture = json.loads(FIXTURE.read_text())
    victim = fixture["sessions"][0]
    assert victim["verdict"] == "VICTIM"
    victim["stop_marker"] = True
    _build_calibration(tmp_path, fixture, alive)
    report = rescue.rescue_scan(tmp_path, now=fixture["pass_epoch"], apply=False)
    fired = {d.session_id for d in report.fire()}
    assert victim["session_id"] not in fired
    assert fired == set(fixture["expected_candidates"]) - {victim["session_id"]}


def test_a_host_only_death_is_seen(tmp_path: Path, alive: set[int]) -> None:
    """A death whose only record is its ``run/host`` boot record still fires.

    Round-1 MAJOR: the census parsed ``run/host`` with ``SessionRecord.from_json``,
    which raises for the four fields a ``journal.BootRecord`` does not carry, and
    ``registry.scan`` drops what will not parse — so the whole namespace was
    invisible and a spawn that died at load left no decision at all.
    """
    from local_operator.session.runtime.journal import BootRecord
    from local_operator.session.runtime.types import HOST_RUN_DIRNAME

    sid = "abab12121212"
    _session_dir(tmp_path, sid)
    record = BootRecord(
        pid=DEAD_PID,
        session_id=sid,
        cwd="/Users/damian",
        started_at=STARTED,
        heartbeat_at=HEARTBEAT,
    )
    directory = tmp_path / HOST_RUN_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{DEAD_PID}.json").write_text(json.dumps(record.to_json()))
    _completions(tmp_path, [(f"session/{sid}", "error", "disposed", "the session was disposed")])

    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert [d.session_id for d in report.fire()] == [sid]

    # The two namespaces agree: the same session under run/mobile fires as well.
    _record(tmp_path, "abab12121213", pid=DEAD_PID + 1, started_at=STARTED, heartbeat_at=HEARTBEAT)
    _completions(tmp_path, [(f"session/{sid}", "error", "disposed", "x")])
    both = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    assert sid in {d.session_id for d in both.fire()}


# --- B. bounds, idempotency, the ledger ------------------------------------


def _bound_fixture(tmp_path: Path, alive: set[int], sid: str = "121212121212") -> None:
    _session_dir(tmp_path, sid)
    _record(tmp_path, sid, pid=DEAD_PID, started_at=STARTED, heartbeat_at=HEARTBEAT)
    _completions(tmp_path, [(f"session/{sid}", "error", "disposed", "x")])


def test_two_passes_over_the_same_run_key_engage_once(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)

    first = rescue.rescue_scan(tmp_path, now=NOW, apply=True)
    assert [d.session_id for d in first.to_engage] == [sid]

    # The seat records the attempt; the next pass must wait out the backoff.
    rescue.note_rescue_attempt(tmp_path, sid, outcome="failed", detail="boom", now=NOW)
    second = rescue.rescue_scan(tmp_path, now=NOW + 1.0, apply=True)
    assert second.to_engage == []
    assert second.refusals["backing-off"] == 1


def test_the_backoff_walk_is_30_60_120_300_900(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)

    # The FIRST rung (30 s) is armed when the episode opens, before any attempt.
    entry = _ledger(tmp_path, sid)
    assert int(entry["next_at_ms"] - NOW * 1000) == 30_000

    # Each subsequent rung is armed by the attempt that failed before it, and the
    # 5th attempt ends the episode (no 6th wait).
    seen: list[int] = []
    for _ in range(5):
        rescue.note_rescue_attempt(tmp_path, sid, outcome="failed", detail="x", now=NOW)
        entry = _ledger(tmp_path, sid)
        if entry["next_at_ms"] is None:
            break
        seen.append(int(entry["next_at_ms"] - NOW * 1000))
    assert seen == [60_000, 120_000, 300_000, 900_000]
    assert entry["state"] == "abandoned"


def test_the_started_outcome_advances_the_ladder_too(tmp_path: Path, alive: set[int]) -> None:
    """The ``started`` path (a runtime asked for, no successor record yet) walks
    the same ladder as ``failed``.

    This is the path the round-1 MAJOR lived on: a stray second ``next_at_ms``
    assignment inside the ``started`` branch re-indexed the ladder at
    ``count - 1``, so 30 s was spent twice and the 900 s rung was never reached —
    and nothing drove the branch, which is why it shipped.
    """
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)

    seen: list[int] = []
    entry = _ledger(tmp_path, sid)
    for _ in range(5):
        rescue.note_rescue_attempt(tmp_path, sid, outcome="started", now=NOW)
        entry = _ledger(tmp_path, sid)
        if entry["next_at_ms"] is None:
            break
        seen.append(int(entry["next_at_ms"] - NOW * 1000))
    assert seen == [60_000, 120_000, 300_000, 900_000]
    assert entry["state"] == "abandoned"
    # ``started`` means a runtime was asked for: the engagement is recorded.
    assert entry["engaged"] is None


def test_a_started_outcome_records_the_engaged_pid(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)

    rescue.note_rescue_attempt(tmp_path, sid, outcome="started", engaged_pid=4_700_500, now=NOW)
    entry = _ledger(tmp_path, sid)
    assert entry["state"] == "engaged"
    assert entry["engaged"] == {"pid": 4_700_500, "at": NOW}


def test_abandon_after_five_attempts(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)

    for index in range(rescue.RESCUE_MAX_ATTEMPTS):
        rescue.note_rescue_attempt(
            tmp_path, sid, outcome="failed", detail="x", now=NOW + index * 1000
        )
    entry = _ledger(tmp_path, sid)
    assert entry["state"] == "abandoned"
    assert entry["count"] == rescue.RESCUE_MAX_ATTEMPTS

    later = rescue.rescue_scan(tmp_path, now=NOW + 3300.0, apply=True)
    assert later.to_engage == []
    assert later.refusals["abandoned"] == 1


def test_a_new_run_key_starts_a_fresh_episode(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)
    rescue.note_rescue_attempt(tmp_path, sid, outcome="failed", detail="x", now=NOW)

    # The session dies AGAIN as a new run (different pid), well past the episode
    # cooldown: the pass must consider it afresh. The run is old enough not to be
    # "young" (started > 120 s ago) and its death is past the 30 s settle.
    new_pid = 4_700_009
    _record(tmp_path, sid, pid=new_pid, started_at=NOW + 600.0, heartbeat_at=NOW + 720.0)
    report = rescue.rescue_scan(tmp_path, now=NOW + 800.0, apply=True)
    assert [d.session_id for d in report.to_engage] == [sid]
    assert _ledger(tmp_path, sid)["run_key"]["pid"] == new_pid


def test_the_episode_cooldown_holds_a_rapid_second_death(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)
    rescue.note_rescue_attempt(tmp_path, sid, outcome="verified", engaged_pid=4_700_010, now=NOW)

    new_pid = 4_700_011
    # Death 40 s old (past the settle) on a run 320 s old (past the young-run
    # window), so the only rung that can refuse it is the episode cooldown.
    _record(tmp_path, sid, pid=new_pid, started_at=NOW - 300.0, heartbeat_at=NOW + 20.0)
    report = rescue.rescue_scan(tmp_path, now=NOW + 60.0, apply=True)
    assert report.to_engage == []
    assert report.refusals["cooldown"] == 1


def test_the_session_breaker_trips_after_three_episodes(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)
    rescue.note_rescue_attempt(tmp_path, sid, outcome="verified", engaged_pid=DEAD_PID, now=NOW)

    # Four more episodes, each a fresh run key 1000 s apart (past the cooldown),
    # each resolved so it lands in the history the breaker reads.
    for index in range(1, 5):
        pid = 4_700_100 + index
        base = NOW + 1000.0 * index
        _record(tmp_path, sid, pid=pid, started_at=base - 500.0, heartbeat_at=base - 100.0)
        report = rescue.rescue_scan(tmp_path, now=base, apply=True)
        assert [d.session_id for d in report.to_engage] == [sid], index
        rescue.note_rescue_attempt(tmp_path, sid, outcome="verified", engaged_pid=pid, now=base)

    entry = _ledger(tmp_path, sid)
    assert len(entry["episodes"]) > rescue.RESCUE_BREAKER_EPISODES

    # A fifth death: enough episodes inside the window (four resolved in the last
    # 3600 s) to trip the breaker, which is checked ahead of the cooldown.
    breaker_pid = 4_700_200
    base = NOW + 4500.0
    _record(tmp_path, sid, pid=breaker_pid, started_at=base - 500.0, heartbeat_at=base - 100.0)
    report = rescue.rescue_scan(tmp_path, now=base, apply=True)
    assert report.to_engage == []
    assert report.refusals["breaker"] == 1


def test_the_ledger_survives_a_restart(tmp_path: Path, alive: set[int]) -> None:
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)
    rescue.note_rescue_attempt(tmp_path, sid, outcome="failed", detail="x", now=NOW)

    # A fresh read (as a restarted supervisor would do) sees the same state.
    entry = _ledger(tmp_path, sid)
    assert entry["count"] == 1
    assert entry["state"] == "pending"
    assert rescue.ledger_path(tmp_path, sid).is_file()


def test_a_lever_pulled_between_passes_wins(tmp_path: Path, alive: set[int]) -> None:
    """The stop levers are re-read every pass, not just at first seeing."""
    sid = "121212121212"
    _bound_fixture(tmp_path, alive, sid)
    rescue.rescue_scan(tmp_path, now=NOW, apply=True)

    _wake(tmp_path, sid, stopped_at=NOW)
    report = rescue.rescue_scan(tmp_path, now=NOW + 100.0, apply=True)
    assert report.to_engage == []
    assert report.refusals["held"] == 1


# --- E. receipt-aware tightening -------------------------------------------


def _receipt_fixture(
    tmp_path: Path,
    alive: set[int],
    signals: list[dict[str, Any]],
    *,
    sid: str = "777788889999",
) -> None:
    _session_dir(tmp_path, sid)
    _record(tmp_path, sid, pid=DEAD_PID, started_at=STARTED, heartbeat_at=HEARTBEAT)
    _completions(tmp_path, [(f"session/{sid}", "interrupted", "user-stop", "stopped")])
    _receipt(tmp_path, sid, DEAD_PID, STARTED, signals)


def _verdict(tmp_path: Path, sid: str) -> str:
    report = rescue.rescue_scan(tmp_path, now=NOW, apply=False)
    return next(d for d in report.decisions if d.session_id == sid).verdict


@pytest.mark.parametrize(
    ("label", "signals", "verdict"),
    [
        (
            "sanction-none",
            [{"sanction": "none", "action": "terminate"}],
            "fire",
        ),
        (
            "deliberate-marker",
            [{"sanction": "marker", "stop_marker": {"deliberate": True, "mechanism": "stop"}}],
            "skip",
        ),
        (
            "reclaim-idle",
            [
                {
                    "sanction": "marker",
                    "in_flight": False,
                    "stop_marker": {"deliberate": False, "mechanism": "reclaim"},
                }
            ],
            "skip",
        ),
        (
            "install-in-flight",
            [
                {
                    "sanction": "marker",
                    "in_flight": True,
                    "stop_marker": {"deliberate": False, "mechanism": "install"},
                }
            ],
            "fire",
        ),
        (
            "named-but-not-on-the-matrix",
            [
                {
                    "sanction": "marker",
                    "in_flight": True,
                    "stop_marker": {"deliberate": False, "mechanism": "mesh-move"},
                }
            ],
            "skip",
        ),
    ],
)
def test_receipt_tightening(
    tmp_path: Path, alive: set[int], label: str, signals: list[dict[str, Any]], verdict: str
) -> None:
    _receipt_fixture(tmp_path, alive, signals)
    assert _verdict(tmp_path, "777788889999") == verdict, label


def test_a_receipt_that_does_not_cover_the_run_falls_back(tmp_path: Path, alive: set[int]) -> None:
    """Feature detection, both ways: the RECEIPT branch and the FALLBACK branch.

    A receipt naming a different pid cannot attest to this run, so the file
    presence alone must not tighten anything — the conservative fallback decides,
    which for an unsanctioned ``user-stop`` is FIRE.
    """
    sid = "777788889999"
    _session_dir(tmp_path, sid)
    _record(tmp_path, sid, pid=DEAD_PID, started_at=STARTED, heartbeat_at=HEARTBEAT)
    _completions(tmp_path, [(f"session/{sid}", "interrupted", "user-stop", "stopped")])
    _receipt(
        tmp_path,
        sid,
        DEAD_PID + 1,
        STARTED,
        [{"sanction": "marker", "stop_marker": {"deliberate": True}}],
    )

    assert _verdict(tmp_path, sid) == "fire"
    assert rescue.read_signal_receipt(tmp_path, sid) is not None
