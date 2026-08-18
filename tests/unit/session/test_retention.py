"""Retention sweep over the ephemeral session store.

The behaviour under test is a deletion, so the tests care as much about what
survives as about what goes: the live session and the newest history must be
there afterwards, on every path, including the ones that fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from local_operator.session.retention import (
    CLAIM_TRUST_S,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_SESSIONS,
    LIVE_MARKER_NAME,
    NEW_SESSION_GRACE_S,
    _is_claimed,
    claim_session,
    release_session,
    sweep_from_config,
    sweep_sessions,
)


def _dead_pid() -> int:
    """A process id nothing owns.

    Spawned and reaped so the id is genuinely unused rather than merely large:
    a hardcoded number could belong to a live process on a busy machine and
    would make these tests flake in the "nothing was evicted" direction.

    ``subprocess`` rather than ``os.fork()``: pytest runs with threads (the
    TUI suites start them), and forking a threaded process can deadlock the
    child in the allocator. The pid is reaped by ``wait()`` before it is
    returned, so ``os.kill(pid, 0)`` reports it gone.
    """
    proc = subprocess.Popen([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def _session(root, name: str, *, size: int = 1024, age_days: float = 0.0):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("x" * size)
    when = time.time() - age_days * 86400
    for path in (directory / "transcript.jsonl", directory):
        os.utime(path, (when, when))
    return directory


def test_missing_directory_is_a_no_op(tmp_path):
    """First run of a fresh install: nothing to sweep is not an error."""
    result = sweep_sessions(tmp_path / "never-created")
    assert result == sweep_sessions(tmp_path / "never-created")
    assert result.evicted == 0 and result.errors == 0


def test_count_ceiling_evicts_oldest_first(tmp_path):
    sessions = tmp_path / "sessions"
    for i in range(10):
        _session(sessions, f"s{i:02d}", age_days=10 - i)

    result = sweep_sessions(sessions, max_sessions=4, max_bytes=0, max_age_days=0)

    survivors = sorted(p.name for p in sessions.iterdir())
    assert survivors == ["s06", "s07", "s08", "s09"]
    assert result.evicted == 6


def test_byte_ceiling_holds_even_when_count_does(tmp_path):
    """One session that dumps megabytes must not be able to blow the budget
    just because there are only a handful of directories."""
    sessions = tmp_path / "sessions"
    _session(sessions, "old", size=900_000, age_days=3)
    _session(sessions, "mid", size=900_000, age_days=2)
    live_survivor = _session(sessions, "new", size=900_000, age_days=1)

    result = sweep_sessions(sessions, max_sessions=100, max_bytes=1_000_000, max_age_days=0)

    assert result.bytes_remaining <= 1_000_000
    assert live_survivor.exists()
    assert not (sessions / "old").exists()


def test_age_ceiling(tmp_path):
    sessions = tmp_path / "sessions"
    _session(sessions, "stale", age_days=45)
    _session(sessions, "fresh", age_days=1)

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)

    assert not (sessions / "stale").exists()
    assert (sessions / "fresh").exists()


def test_live_session_is_never_evicted(tmp_path):
    """The live directory is both the oldest and, on its own, over every
    ceiling — and still must survive, because evicting it takes out resume
    and compaction replay for the run that is currently writing to it."""
    sessions = tmp_path / "sessions"
    live = _session(sessions, "live", size=5_000_000, age_days=400)
    for i in range(5):
        _session(sessions, f"other{i}", size=1000, age_days=1)

    result = sweep_sessions(
        sessions,
        live_dir=live,
        max_sessions=1,
        max_bytes=1000,
        max_age_days=1,
    )

    assert live.exists()
    assert (live / "transcript.jsonl").read_text().startswith("x")
    assert result.evicted == 5


def test_empty_directories_are_reaped_once_past_the_grace_period(tmp_path):
    """Left behind by runs that built a session and exited before writing a
    turn; 23 of 147 directories on a real install were exactly this.

    Aged past ``NEW_SESSION_GRACE_S``, because an empty directory younger than
    that belongs to a session that is still starting up (see the test below).
    """
    sessions = tmp_path / "sessions"
    hollow = sessions / "hollow"
    hollow.mkdir(parents=True)
    old = time.time() - NEW_SESSION_GRACE_S - 60
    os.utime(hollow, (old, old))
    _session(sessions, "real")

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert not hollow.exists()
    assert (sessions / "real").exists()


def test_a_session_that_has_not_written_its_first_turn_yet_survives(tmp_path):
    """The regression that broke real sessions.

    A run creates its directory before it writes the first transcript line. A
    concurrent session's startup sweep landing in that window used to reap the
    directory as "empty", and the starting run then died on
    ``FileNotFoundError: .../transcript.jsonl`` — on that turn and on every
    turn after it, because nothing recreated the directory.
    """
    sessions = tmp_path / "sessions"
    starting = sessions / "starting"
    starting.mkdir(parents=True)

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert starting.exists()


def test_a_concurrent_live_session_is_not_evicted(tmp_path):
    """The other half of the same regression.

    ``live_dir`` only names the sweeping session's OWN directory, so every
    other running session was an ordinary candidate. Claiming is what tells
    this sweep that a directory belongs to a run that is still writing to it.
    """
    sessions = tmp_path / "sessions"
    concurrent = _session(sessions, "concurrent", size=512, age_days=9)
    claim_session(concurrent)
    for i in range(4):
        _session(sessions, f"other{i}", size=512, age_days=8 - i)
    mine = _session(sessions, "mine", size=512)

    result = sweep_sessions(sessions, live_dir=mine, max_sessions=1, max_bytes=0, max_age_days=0)

    assert concurrent.exists(), "a session still writing to this directory lost it"
    assert mine.exists()
    # The ceiling applies to evictable history only: the claimed and live
    # directories are exempt, so one of the four ordinary ones survives.
    assert result.evicted == 3
    assert (sessions / "other3").exists()


def test_a_claim_from_a_dead_process_does_not_protect_the_directory(tmp_path):
    """Otherwise a crashed run would make its directory immortal and quietly
    disable every ceiling."""
    sessions = tmp_path / "sessions"
    orphan = _session(sessions, "orphan", size=512, age_days=90)
    claim_session(orphan, pid=_dead_pid())

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)

    assert not orphan.exists()


def test_a_corrupt_claim_does_not_protect_the_directory(tmp_path):
    sessions = tmp_path / "sessions"
    corrupt = _session(sessions, "corrupt", size=512, age_days=90)
    (corrupt / LIVE_MARKER_NAME).write_text("not-a-pid")

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)

    assert not corrupt.exists()


def test_releasing_a_claim_makes_the_directory_evictable_again(tmp_path):
    sessions = tmp_path / "sessions"
    finished = _session(sessions, "finished", size=512, age_days=90)
    claim_session(finished)

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)
    assert finished.exists(), "still claimed"

    release_session(finished)
    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)
    assert not finished.exists()


def test_a_claim_alone_does_not_count_as_session_content(tmp_path):
    """A directory holding only a released claim is empty history, and the
    reap must see it that way rather than counting our own bookkeeping."""
    sessions = tmp_path / "sessions"
    hollow = sessions / "hollow"
    hollow.mkdir(parents=True)
    claim_session(hollow, pid=_dead_pid())
    old = time.time() - NEW_SESSION_GRACE_S - 60
    for path in (hollow / LIVE_MARKER_NAME, hollow):
        os.utime(path, (old, old))

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert not hollow.exists()


def test_eviction_ranks_by_last_activity_not_by_directory_age(tmp_path):
    """A directory's own mtime is effectively its BIRTH time: it moves when an
    entry is created inside it, and a session creates ``transcript.jsonl`` once
    and then appends to it for hours. Ranking by it evicted the long-running
    session an operator had been talking to all afternoon ahead of one-shot
    runs that had started since and done nothing.
    """
    sessions = tmp_path / "sessions"
    veteran = _session(sessions, "veteran", size=512, age_days=5)
    # Started long ago, but written to a minute ago: the busiest session here.
    recent = time.time() - 60
    os.utime(veteran / "transcript.jsonl", (recent, recent))
    for i in range(3):
        _session(sessions, f"drive-by{i}", size=512, age_days=1)

    sweep_sessions(sessions, max_sessions=2, max_bytes=0, max_age_days=0)

    assert veteran.exists(), "the most active session was evicted as the oldest"


def test_sweep_is_idempotent(tmp_path):
    sessions = tmp_path / "sessions"
    for i in range(8):
        _session(sessions, f"s{i}", age_days=8 - i)

    first = sweep_sessions(sessions, max_sessions=3, max_bytes=0, max_age_days=0)
    second = sweep_sessions(sessions, max_sessions=3, max_bytes=0, max_age_days=0)

    assert first.evicted == 5
    assert second.evicted == 0
    assert len(list(sessions.iterdir())) == 3


def test_all_ceilings_disabled_keeps_everything_but_empties(tmp_path):
    sessions = tmp_path / "sessions"
    for i in range(6):
        _session(sessions, f"s{i}", age_days=500)

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert len(list(sessions.iterdir())) == 6


def test_sibling_stores_under_the_config_dir_are_untouched(tmp_path):
    """Only ``sessions/`` is swept.

    The spill store (``<config>/spill``) holds the full text of large tool
    outputs behind the ``spill://`` handles the truncation footers advertise.
    It runs its OWN LRU eviction under its own ceiling and protects the live
    session's entries inside a grace window; a second sweeper racing it could
    evict a handle whose footer is still sitting in the live transcript, and
    the agent would be told to expand an output that no longer exists.
    """
    sessions = tmp_path / "sessions"
    for i in range(6):
        _session(sessions, f"s{i}", age_days=6 - i)
    spill = tmp_path / "spill"
    spill.mkdir()
    (spill / "deadbeef.txt").write_text("the full tool output")
    agents = tmp_path / "agents"
    (agents / "an-agent").mkdir(parents=True)
    (agents / "an-agent" / "transcript.jsonl").write_text("named agent history")

    sweep_from_config(_Config({"session_retention_max_sessions": 1}), tmp_path, None)

    assert (spill / "deadbeef.txt").read_text() == "the full tool output"
    assert (agents / "an-agent" / "transcript.jsonl").exists()
    assert len(list(sessions.iterdir())) == 1


class _Config:
    def __init__(self, values):
        self._values = values

    def get_config_value(self, key, default=None):
        return self._values.get(key, default)


def test_sweep_from_config_reads_the_settings(tmp_path):
    sessions = tmp_path / "sessions"
    for i in range(6):
        _session(sessions, f"s{i}", age_days=6 - i)

    config = _Config({"session_retention_max_sessions": 2})
    result = sweep_from_config(config, tmp_path, live_dir=None)

    assert result.evicted == 4
    assert len(list(sessions.iterdir())) == 2


def test_unparseable_setting_falls_back_to_the_default(tmp_path):
    """A typo must not silently restore the unbounded behaviour."""
    sessions = tmp_path / "sessions"
    _session(sessions, "s0")

    config = _Config({"session_retention_max_sessions": "not-a-number"})
    sweep_from_config(config, tmp_path, live_dir=None)

    assert (sessions / "s0").exists()
    assert DEFAULT_MAX_SESSIONS > 0
    assert DEFAULT_MAX_BYTES > 0
    assert DEFAULT_MAX_AGE_DAYS > 0


def test_undeletable_directory_is_reported_not_raised(tmp_path, monkeypatch):
    """Reclaiming disk must never be the reason a session fails to start."""
    sessions = tmp_path / "sessions"
    for i in range(4):
        _session(sessions, f"s{i}", age_days=4 - i)

    def boom(_path):
        raise OSError("read-only file system")

    monkeypatch.setattr("local_operator.session.retention.shutil.rmtree", boom)
    result = sweep_sessions(sessions, max_sessions=1, max_bytes=0, max_age_days=0)

    assert result.errors == 3
    assert result.evicted == 0
    assert len(list(sessions.iterdir())) == 4


@pytest.mark.parametrize("ceiling", [1, 5, 25])
def test_directory_stays_under_budget_however_many_sessions_arrive(tmp_path, ceiling):
    """The property the whole module exists for, exercised by exceeding it."""
    sessions = tmp_path / "sessions"
    live = _session(sessions, "live", size=512)
    for i in range(120):
        _session(sessions, f"s{i:03d}", size=512, age_days=(120 - i) / 24)
        sweep_sessions(sessions, live_dir=live, max_sessions=ceiling, max_bytes=0, max_age_days=0)
        # +1 for the live directory, which is exempt from the ceiling.
        assert len(list(sessions.iterdir())) <= ceiling + 1
    assert live.exists()


def test_live_bytes_are_reported_separately_from_governed_bytes(tmp_path):
    """A claimed session is exempt from the ceilings, so ``bytes_remaining``
    (what the ceilings govern) is NOT the size of the store. Reporting only
    that figure made the sweep look compliant while the disk said otherwise;
    a caller measuring the footprint needs ``bytes_on_disk``."""
    sessions = tmp_path / "sessions"
    running = _session(sessions, "running", size=50_000)
    claim_session(running)
    _session(sessions, "history", size=1_000, age_days=1)

    result = sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert result.bytes_remaining == 1_000
    assert result.bytes_live == 50_000
    assert result.bytes_on_disk == 51_000


def test_live_sessions_over_the_byte_ceiling_are_logged(tmp_path, caplog):
    """The overshoot is the correct trade but must not be silent: it is the
    only signal that the configured bound is not currently being honoured,
    and without it a full volume looks like a clean sweep log."""
    import logging

    sessions = tmp_path / "sessions"
    running = _session(sessions, "running", size=200_000)
    claim_session(running)

    with caplog.at_level(logging.WARNING, logger="local_operator.session.retention"):
        result = sweep_sessions(sessions, max_sessions=0, max_bytes=1_000, max_age_days=0)

    assert running.exists()
    assert result.bytes_live == 200_000
    assert any("above the" in record.message for record in caplog.records)


def test_a_released_claim_does_not_refresh_an_empty_directory(tmp_path):
    """Claim churn used to bump the directory's mtime, so a session that
    started, wrote nothing and exited looked freshly created and re-earned the
    startup grace period on every sweep — exempting exactly the population the
    empty-directory reap exists to collect."""
    sessions = tmp_path / "sessions"
    hollow = sessions / "hollow"
    hollow.mkdir(parents=True)
    claim_session(hollow)
    release_session(hollow)
    # The run happened well before the grace window; only the marker records it.
    old = time.time() - NEW_SESSION_GRACE_S - 600
    os.utime(hollow / LIVE_MARKER_NAME, (old, old))

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert not hollow.exists()


def test_a_released_claim_reads_as_unclaimed_without_a_liveness_probe(tmp_path):
    """``release_session`` overwrites rather than unlinks (to preserve the age
    record), so the released marker must not read as a live claim — including
    where the pid probe is unavailable."""
    sessions = tmp_path / "sessions"
    finished = _session(sessions, "finished", size=512, age_days=90)
    claim_session(finished)
    release_session(finished)

    assert not _is_claimed(finished)

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)
    assert not finished.exists()


def test_an_unverifiable_claim_expires_instead_of_lasting_forever(tmp_path, monkeypatch):
    """Where liveness cannot be probed (Windows), an unbounded claim would
    exempt a directory from all three ceilings permanently, so every crash
    would switch the module off one directory at a time."""
    monkeypatch.setattr("local_operator.session.retention._LIVENESS_IS_VERIFIABLE", False)
    monkeypatch.setattr("local_operator.session.retention._process_alive", lambda pid: True)

    sessions = tmp_path / "sessions"
    stale = _session(sessions, "stale", size=512, age_days=90)
    claim_session(stale)
    ancient = time.time() - CLAIM_TRUST_S - 60
    os.utime(stale / LIVE_MARKER_NAME, (ancient, ancient))

    fresh = _session(sessions, "fresh", size=512, age_days=90)
    claim_session(fresh)

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)

    assert not stale.exists(), "an unverifiable claim protected a directory forever"
    assert fresh.exists(), "a recent claim must still protect a running session"
