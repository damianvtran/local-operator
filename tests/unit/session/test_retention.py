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


def test_empty_directories_are_always_reaped(tmp_path):
    """Left behind by runs that built a session and exited before writing a
    turn; 23 of 147 directories on a real install were exactly this.

    Reaped whatever their apparent age: a session that is still starting up is
    protected by its CLAIM, not by looking new (see the test below).
    """
    sessions = tmp_path / "sessions"
    hollow = sessions / "hollow"
    hollow.mkdir(parents=True)
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

    The CLAIM is what protects it, which is why ``_prepare`` claims before it
    sweeps. Nothing here depends on the directory looking recently created:
    an empty directory with no live owner is still reaped on sight.
    """
    sessions = tmp_path / "sessions"
    starting = sessions / "starting"
    starting.mkdir(parents=True)
    claim_session(starting)

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
    # Live sessions holding most of the budget: eviction cannot bring the store
    # back under the ceiling, however much history it reclaims. Note the live
    # bytes do NOT exceed the ceiling by themselves — the shape the first
    # version of this warning stayed silent for.
    running = _session(sessions, "running", size=280_000)
    claim_session(running)
    _session(sessions, "history", size=200_000, age_days=1)

    with caplog.at_level(logging.WARNING, logger="local_operator.session.retention"):
        result = sweep_sessions(sessions, max_sessions=0, max_bytes=300_000, max_age_days=0)

    assert running.exists()
    assert result.bytes_live == 280_000
    assert result.bytes_live < 300_000
    assert any("exempt from eviction" in record.message for record in caplog.records)


def test_a_hollow_run_leaves_nothing_behind_once_it_is_past_the_grace_period(tmp_path):
    """A session that started, wrote nothing and exited must be reaped, marker
    and all — that population is 23 of 147 directories on a real install.

    Claim/release churn touches the directory, so such a directory keeps its
    grace period for one sweep cycle longer than its age warrants; the reap
    collects it on the next pass, which is the tradeoff ``_activity_mtime``
    documents.
    """
    sessions = tmp_path / "sessions"
    hollow = sessions / "hollow"
    hollow.mkdir(parents=True)
    claim_session(hollow)
    release_session(hollow)
    assert not (hollow / LIVE_MARKER_NAME).exists(), "release must remove the marker"

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert not hollow.exists()


def test_a_released_claim_reads_as_unclaimed_on_every_platform(tmp_path, monkeypatch):
    """Release must stop protecting the directory even where the pid probe is
    unavailable, since there nothing else can disprove a stale claim.

    Today this holds on every platform BY CONSTRUCTION rather than by this
    test: release removes the marker, so ``_is_claimed`` returns at its
    missing-file guard before any platform branch is reached. The patch below
    is kept anyway — it is what would catch a future release that blanks the
    marker instead of deleting it, which is exactly the shape a previous
    revision of this fix had.
    """
    # BOTH names, always: ``_LIVENESS_IS_VERIFIABLE`` is derived from
    # ``_PLATFORM`` at import, so patching one alone builds a combination the
    # real code never reaches and quietly stops testing the platform it names.
    monkeypatch.setattr("local_operator.session.retention._PLATFORM", "win32")
    monkeypatch.setattr("local_operator.session.retention._LIVENESS_IS_VERIFIABLE", False)

    sessions = tmp_path / "sessions"
    finished = _session(sessions, "finished", size=512, age_days=90)
    claim_session(finished)
    release_session(finished)

    assert not _is_claimed(finished, time.time())

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)
    assert not finished.exists()


def test_an_unverifiable_claim_expires_once_the_session_stops_writing(tmp_path, monkeypatch):
    """Where liveness cannot be probed (Windows), an unbounded claim would
    exempt a directory from all three ceilings permanently, so every crash
    would switch the module off one directory at a time.

    ``_PLATFORM`` is patched rather than ``_process_alive``, so the real
    branch runs — including the ``pid <= 0`` guard ahead of it, which is what
    makes a bogus pid read as unclaimed everywhere.
    """
    monkeypatch.setattr("local_operator.session.retention._PLATFORM", "win32")
    monkeypatch.setattr("local_operator.session.retention._LIVENESS_IS_VERIFIABLE", False)

    sessions = tmp_path / "sessions"
    # Claimed, and nothing has written for far longer than the trust window
    # (90 days, so the age ceiling has something to act on once the claim
    # stops protecting it).
    abandoned = _session(sessions, "abandoned", size=512, age_days=90)
    claim_session(abandoned)
    ancient = time.time() - 90 * 86400
    os.utime(abandoned / LIVE_MARKER_NAME, (ancient, ancient))
    assert time.time() - ancient > CLAIM_TRUST_S

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=30)

    assert not abandoned.exists(), "an unverifiable claim protected a directory forever"


def test_an_unverifiable_claim_survives_while_the_session_is_still_writing(tmp_path, monkeypatch):
    """The regression guard for the trust bound itself.

    Measuring the bound against the MARKER would delete the transcript of a
    session that is alive and actively writing, once it had simply been
    running longer than the window — this module's original bug, on a timer.
    It is measured against the last write instead, which a live session keeps
    moving.
    """
    monkeypatch.setattr("local_operator.session.retention._PLATFORM", "win32")
    monkeypatch.setattr("local_operator.session.retention._LIVENESS_IS_VERIFIABLE", False)

    sessions = tmp_path / "sessions"
    long_running = _session(sessions, "long-running", size=512)
    claim_session(long_running)
    # Started long ago — the marker is never refreshed — but writing right now.
    ancient = time.time() - CLAIM_TRUST_S * 3
    os.utime(long_running / LIVE_MARKER_NAME, (ancient, ancient))
    now = time.time()
    os.utime(long_running / "transcript.jsonl", (now, now))

    # A byte ceiling nothing can satisfy: every CANDIDATE is evicted, so the
    # directory survives only if the claim still exempts it from the sweep.
    sweep_sessions(sessions, max_sessions=0, max_bytes=1, max_age_days=0)

    assert long_running.exists(), "a live, actively-writing session was evicted"


def test_an_unverifiable_claim_survives_a_resume_of_an_older_transcript(tmp_path, monkeypatch):
    """`--resume` is the case where the claim is fresh and the content is old.

    Measuring the trust bound against activity ALONE discarded the marker, so a
    session resumed from yesterday's transcript read as abandoned from the
    moment it was claimed until its first turn landed — a live session's
    transcript deleted out from under it, which is this module's original bug.
    The bound takes the later of the two clocks for that reason.
    """
    monkeypatch.setattr("local_operator.session.retention._PLATFORM", "win32")
    monkeypatch.setattr("local_operator.session.retention._LIVENESS_IS_VERIFIABLE", False)

    sessions = tmp_path / "sessions"
    # Yesterday's transcript, well past the trust window; claimed just now.
    resumed = _session(sessions, "resumed", size=512)
    stale = time.time() - CLAIM_TRUST_S * 2
    os.utime(resumed / "transcript.jsonl", (stale, stale))
    claim_session(resumed)

    assert _is_claimed(resumed, time.time()), "a just-resumed session read as abandoned"

    sweep_sessions(sessions, max_sessions=0, max_bytes=1, max_age_days=0)

    assert resumed.exists(), "a live resumed session was evicted before its first turn"


def test_a_store_inside_its_ceiling_does_not_warn(tmp_path, caplog):
    """The overshoot warning must not fire on a healthy store.

    Eviction leaves governed bytes just under the ceiling by construction, so
    any live session pushes the total fractionally over it. A strict comparison
    fired on roughly half of normal startups, which is how a real 1.8x warning
    ends up ignored.
    """
    import logging

    sessions = tmp_path / "sessions"
    running = _session(sessions, "running", size=8_000)
    claim_session(running)
    _session(sessions, "history", size=100_000, age_days=1)

    with caplog.at_level(logging.WARNING, logger="local_operator.session.retention"):
        result = sweep_sessions(sessions, max_sessions=0, max_bytes=100_000, max_age_days=0)

    # Over the ceiling, but only by the live session — the healthy resting state.
    assert result.bytes_on_disk > 100_000
    assert not [r for r in caplog.records if "exempt from eviction" in r.message]


def test_directories_that_could_not_be_deleted_are_still_counted_and_reported(
    tmp_path, monkeypatch, caplog
):
    """A sweep that can delete nothing must not report an empty store.

    Bytes selected for eviction but still on disk belong to neither the live
    nor the surviving-history bucket, so they used to vanish from the
    accounting: a read-only store holding 10 MB over its ceiling reported
    ``bytes_on_disk=0`` and logged nothing at warning level, which is the blind
    spot the byte accounting exists to close.
    """
    import logging

    sessions = tmp_path / "sessions"
    for i in range(3):
        _session(sessions, f"stuck{i}", size=100_000, age_days=90)

    def refuse(_path):
        raise OSError("read-only file system")

    monkeypatch.setattr("local_operator.session.retention.shutil.rmtree", refuse)

    with caplog.at_level(logging.WARNING, logger="local_operator.session.retention"):
        result = sweep_sessions(sessions, max_sessions=0, max_bytes=1_000, max_age_days=30)

    assert result.errors == 3
    assert result.evicted == 0
    assert result.bytes_remaining == 300_000, "stranded bytes vanished from the accounting"
    assert result.bytes_on_disk == 300_000
    assert any("could not delete" in record.message for record in caplog.records)


def test_an_aborted_child_leaves_nothing_worth_a_retention_slot(tmp_path):
    """A session is stamped with its origin BEFORE its transcript exists, so a
    run that aborts in between leaves a directory holding only the marker.

    Such a directory used to be empty, and empty directories are always reaped
    regardless of the ceilings — they carry nothing to lose. Counting the
    marker's 43 bytes turned each one into an ordinary keep candidate holding a
    slot: measured with a count ceiling of 3, two aborted children evicted two
    of the user's real transcripts to keep two empty markers.
    """
    import os

    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

    sessions = tmp_path / "sessions"
    # The user's real work, older than the children that abort after it.
    for i in range(3):
        _session(sessions, f"user{i}", age_days=3 - i)
    for i in range(2):
        hollow = sessions / f"hollow{i}"
        hollow.mkdir(parents=True)
        mark_session_origin(hollow, ORIGIN_SUBAGENT, label="review")
        now = time.time()
        os.utime(hollow, (now, now))

    sweep_sessions(sessions, max_sessions=3, max_bytes=0, max_age_days=0)

    survivors = sorted(path.name for path in sessions.iterdir())
    assert survivors == ["user0", "user1", "user2"], survivors


def test_the_marker_is_not_charged_against_the_byte_ceiling(tmp_path):
    """The marker is bookkeeping ABOUT a session, never session content, so it
    is not what the ceilings are budgeting."""
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

    sessions = tmp_path / "sessions"
    directory = _session(sessions, "one", size=100)
    before = sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)
    mark_session_origin(directory, ORIGIN_SUBAGENT, label="review")
    after = sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert before.bytes_remaining == after.bytes_remaining == 100


def test_claiming_creates_the_directory_so_there_is_no_unclaimed_window(tmp_path):
    """A session directory must never exist unclaimed, even for an instant.

    Empty directories are reaped on sight, so any gap between creating a
    session directory and claiming it is a window in which a concurrent
    session's startup sweep can delete it — which is the original bug. Callers
    close that gap by claiming FIRST, which only works because the claim
    creates the directory rather than requiring one.
    """
    sessions = tmp_path / "sessions"
    fresh = sessions / "fresh"

    claim_session(fresh)

    assert fresh.is_dir(), "claim_session must create the directory it claims"
    assert (fresh / LIVE_MARKER_NAME).is_file()

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert fresh.exists(), "a claimed, empty, brand-new session directory was reaped"
