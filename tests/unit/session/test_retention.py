"""Retention over the ephemeral session store: nothing is ever deleted.

A session transcript is the only durable record of a run, so the suite's
central property is not what goes but what survives: every directory with a
byte of content must still be there after a sweep, whatever ceiling-like
arguments the caller passes, however old or large or numerous the sessions
are. The only thing a sweep removes is a directory that contains NOTHING —
no transcript, no content — which is definitionally not a session.
"""

from __future__ import annotations

import os
import time

from local_operator.session.retention import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_SESSIONS,
    EMPTY_DIR_GRACE_SECONDS,
    sweep_from_config,
    sweep_sessions,
)


def _session(root, name: str, *, size: int = 1024, age_days: float = 0.0):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("x" * size)
    when = time.time() - age_days * 86400
    for path in (directory / "transcript.jsonl", directory):
        os.utime(path, (when, when))
    return directory


def _hollow(root, name: str, *, age_seconds: float = EMPTY_DIR_GRACE_SECONDS + 60.0):
    """An EMPTY session directory old enough for the reaper to take it.

    Aged past the grace window by default because that is what "abandoned"
    means now: a fresh empty directory is indistinguishable from a sibling
    process's session awaiting its first message, and must survive.
    """
    directory = root / name
    directory.mkdir(parents=True)
    when = time.time() - age_seconds
    os.utime(directory, (when, when))
    return directory


def test_missing_directory_is_a_no_op(tmp_path):
    """First run of a fresh install: nothing to sweep is not an error."""
    result = sweep_sessions(tmp_path / "never-created")
    assert result == sweep_sessions(tmp_path / "never-created")
    assert result.evicted == 0 and result.errors == 0


def test_no_ceiling_combination_ever_deletes_a_transcript(tmp_path):
    """The regression this module now exists to prevent.

    Every one of these arguments used to doom the directories below:
    10 directories over a count ceiling of 1, 10 MB over a byte ceiling of
    1 KB, and mtimes two years past a 1-day age horizon. A heavy install
    with several concurrent sessions hit exactly this, and the eviction
    took out a running session's transcript — the run's next turn died on
    FileNotFoundError and the session's work was gone. Under the current
    policy every one of them survives.
    """
    sessions = tmp_path / "sessions"
    made = [_session(sessions, f"s{i:02d}", size=1_000_000, age_days=700 - i) for i in range(10)]

    result = sweep_sessions(sessions, max_sessions=1, max_bytes=1024, max_age_days=1)

    assert result.evicted == 0
    for directory in made:
        assert (directory / "transcript.jsonl").read_text() == "x" * 1_000_000


def test_live_dir_is_never_reaped_even_when_empty(tmp_path):
    """The caller just created this directory and has not written a turn.
    It is empty by construction; ``live_dir`` is the belt that keeps the
    sweep from rmtree'ing it in the same call that created it."""
    sessions = tmp_path / "sessions"
    live = _hollow(sessions, "live")
    other = _hollow(sessions, "other")

    result = sweep_sessions(sessions, live_dir=live)

    assert live.exists()
    assert not other.exists()
    assert result.evicted == 1


def test_empty_directories_are_reaped(tmp_path):
    """Left behind by runs that built a session and exited before writing a
    turn; 23 of 147 directories on a real install were exactly this. A
    directory that contains nothing holds nothing to lose."""
    sessions = tmp_path / "sessions"
    _hollow(sessions, "hollow")
    _session(sessions, "real")

    result = sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert not (sessions / "hollow").exists()
    assert (sessions / "real").exists()
    assert result.evicted == 1


def test_a_marker_alone_protects_the_directory(tmp_path):
    """A session is stamped with its origin BEFORE its transcript exists.
    Treating that marker as invisible made the directory look empty, and
    a concurrent process's startup sweep rmtree'd it — the child's first
    append then died on FileNotFoundError, the exact kill this module
    exists to prevent. A marker is a claim: the directory stays."""
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

    sessions = tmp_path / "sessions"
    hollow = _hollow(sessions, "hollow")
    mark_session_origin(hollow, ORIGIN_SUBAGENT, label="review")

    result = sweep_sessions(sessions)

    assert hollow.exists()
    assert result.evicted == 0


def test_any_content_at_all_protects_the_directory(tmp_path):
    """One byte of anything that is not bookkeeping is a session's work."""
    sessions = tmp_path / "sessions"
    directory = sessions / "almost-empty"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("x")

    result = sweep_sessions(sessions, max_sessions=1, max_bytes=1, max_age_days=1)

    assert result.evicted == 0
    assert (directory / "transcript.jsonl").read_text() == "x"


def test_sweep_is_idempotent(tmp_path):
    sessions = tmp_path / "sessions"
    _hollow(sessions, "hollow")
    _session(sessions, "real")

    first = sweep_sessions(sessions)
    second = sweep_sessions(sessions)

    assert first.evicted == 1
    assert second.evicted == 0
    assert sorted(p.name for p in sessions.iterdir()) == ["real"]


def test_sibling_stores_under_the_config_dir_are_untouched(tmp_path):
    """Only ``sessions/`` is swept, and even there only empty directories.

    The spill store (``<config>/spill``) holds the full text of large tool
    outputs behind the ``spill://`` handles the truncation footers
    advertise; the named-agent store holds real conversations. Neither is
    this module's business.
    """
    sessions = tmp_path / "sessions"
    _hollow(sessions, "hollow")
    _session(sessions, "real")
    spill = tmp_path / "spill"
    spill.mkdir()
    (spill / "deadbeef.txt").write_text("the full tool output")
    agents = tmp_path / "agents"
    (agents / "an-agent").mkdir(parents=True)
    (agents / "an-agent" / "transcript.jsonl").write_text("named agent history")

    sweep_from_config(_Config({"session_retention_max_sessions": 1}), tmp_path, None)

    assert (spill / "deadbeef.txt").read_text() == "the full tool output"
    assert (agents / "an-agent" / "transcript.jsonl").exists()
    assert (sessions / "real").exists()
    assert not (sessions / "hollow").exists()


class _Config:
    def __init__(self, values):
        self._values = values

    def get_config_value(self, key, default=None):
        return self._values.get(key, default)


def test_sweep_from_config_never_deletes_even_with_aggressive_settings(tmp_path):
    """A config still carrying the retired ceilings — 200/128MiB/30d were
    the SHIPPED defaults, so most existing config files carry them — must
    not delete anything. The settings are retired; the sweep reaps empties
    and nothing more."""
    sessions = tmp_path / "sessions"
    for i in range(6):
        _session(sessions, f"s{i}", size=5_000_000, age_days=500)

    config = _Config(
        {
            "session_retention_max_sessions": 1,
            "session_retention_max_bytes": 1,
            "session_retention_max_age_days": 1,
        }
    )
    result = sweep_from_config(config, tmp_path, live_dir=None)

    assert result.evicted == 0
    assert len(list(sessions.iterdir())) == 6


def test_retired_ceilings_still_in_config_produce_one_honest_warning(tmp_path, caplog):
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True)
    config = _Config({"session_retention_max_sessions": 200})

    import logging

    with caplog.at_level(logging.WARNING, logger="local_operator.session.retention"):
        sweep_from_config(config, tmp_path, live_dir=None)

    assert "session_retention_max_sessions" in caplog.text
    assert "never deleted automatically" in caplog.text


def test_unparseable_retired_setting_is_ignored_without_warning(tmp_path, caplog):
    """A typo in a retired key changes nothing either way; it is not worth a
    warning now that the value cannot cause a deletion."""
    import logging

    sessions = tmp_path / "sessions"
    _session(sessions, "s0")

    config = _Config({"session_retention_max_sessions": "not-a-number"})
    with caplog.at_level(logging.WARNING, logger="local_operator.session.retention"):
        result = sweep_from_config(config, tmp_path, live_dir=None)

    assert (sessions / "s0").exists()
    assert result.evicted == 0
    assert "not-a-number" not in caplog.text


def test_the_retired_defaults_are_all_zero(tmp_path):
    """If any default ceiling were nonzero it would imply a sweep that still
    evicts by policy. All three are 0 because the ceilings no longer exist."""
    assert DEFAULT_MAX_SESSIONS == 0
    assert DEFAULT_MAX_BYTES == 0
    assert DEFAULT_MAX_AGE_DAYS == 0


def test_undeletable_empty_directory_is_reported_not_raised(tmp_path, monkeypatch):
    """Reclaiming disk must never be the reason a session fails to start."""
    sessions = tmp_path / "sessions"
    for i in range(3):
        _hollow(sessions, f"hollow{i}")
    _session(sessions, "real")

    def boom(_path):
        raise OSError("read-only file system")

    monkeypatch.setattr("local_operator.session.retention.shutil.rmtree", boom)
    result = sweep_sessions(sessions)

    assert result.errors == 3
    assert result.evicted == 0
    assert len(list(sessions.iterdir())) == 4


def test_a_fresh_empty_directory_survives_a_sibling_sweep(tmp_path):
    """THE regression this grace window exists for. ``_prepare`` creates the
    session directory before the user has typed a word; a sibling process
    starting up in that window used to reap it as "empty", and the session's
    first append died on FileNotFoundError — observed on a real install
    running ~12 concurrent sessions. ``live_dir`` cannot help: it names the
    SWEEPING process's session, not the victim's."""
    sessions = tmp_path / "sessions"
    victim = sessions / "just-created-by-another-process"
    victim.mkdir(parents=True)  # fresh mtime — exactly what _prepare leaves

    result = sweep_sessions(sessions, live_dir=None)

    assert victim.exists()
    assert result.evicted == 0


def test_grace_window_boundary(tmp_path):
    """Inside the window survives; past it is reaped. ``now`` moves the
    clock so the boundary is tested without sleeping or utime races."""
    sessions = tmp_path / "sessions"
    fresh = _hollow(sessions, "fresh", age_seconds=0.0)
    aged = _hollow(sessions, "aged", age_seconds=0.0)
    base = time.time()
    for path, when in ((fresh, base), (aged, base - EMPTY_DIR_GRACE_SECONDS - 1)):
        os.utime(path, (when, when))

    result = sweep_sessions(sessions, now=base)

    assert fresh.exists()
    assert not aged.exists()
    assert result.evicted == 1


def test_a_transcript_is_never_deleted_even_when_a_failure_cascade_hits(tmp_path, monkeypatch):
    """The worst case the old design could produce: a scan error, an
    undeletable directory, AND every ceiling tripped at once. Content
    survives regardless."""
    sessions = tmp_path / "sessions"
    real = _session(sessions, "real", size=100, age_days=900)

    def boom(_path):
        raise OSError("everything is broken")

    monkeypatch.setattr("local_operator.session.retention.shutil.rmtree", boom)
    result = sweep_sessions(sessions, max_sessions=1, max_bytes=1, max_age_days=1)

    assert (real / "transcript.jsonl").read_text() == "x" * 100
    assert result.evicted == 0


# --- Claim marker: the startup race and the never-leak counterpart ----------
#
# The never-delete model reaps EMPTY directories, but every LIVE session is
# empty for the instant between its ``mkdir`` and its first append. A claim
# marker (:func:`claim_session`) is what lets a concurrent startup sweep tell
# that live-but-empty directory from a dead run's leftover, so it does not
# delete a session out from under its owner (the original FileNotFoundError
# kill). The marker is LIVENESS, not content: a dead owner's marker must not
# make its empty directory immortal, or the module would stop reaping.


def test_a_live_claim_protects_an_empty_directory(tmp_path):
    """The startup window: a directory a live process owns survives the reap
    even while it holds nothing but the claim marker."""
    import os

    from local_operator.session.retention import claim_session

    sessions = tmp_path / "sessions"
    starting = sessions / "starting"
    # Claim with THIS process's pid, which is provably alive.
    claim_session(starting, pid=os.getpid())
    assert (starting / ".session.pid").exists()

    result = sweep_sessions(sessions)

    assert starting.exists()
    assert result.evicted == 0


def test_a_dead_runs_leftover_marker_does_not_make_its_dir_immortal(tmp_path):
    """A hard-killed session leaves its marker on an otherwise-empty
    directory. Because the marker is liveness (not content) and its pid is
    dead, that corpse is reaped — otherwise every crash would leak a directory
    the sweep could never reclaim."""
    sessions = tmp_path / "sessions"
    corpse = sessions / "corpse"
    corpse.mkdir(parents=True)
    # A pid that does not belong to any live process.
    (corpse / ".session.pid").write_text("2147483646")

    result = sweep_sessions(sessions)

    assert not corpse.exists()
    assert result.evicted == 1


def test_a_claim_landing_mid_sweep_is_honoured_before_the_rmtree(tmp_path):
    """L2: scan and delete are one pass, and the claim is re-checked
    immediately before each ``rmtree``. A claim written after the loop began
    but before this directory's deletion must save it — the window a
    concurrent session's startup claim lands in."""
    import os

    import local_operator.session.retention as retention

    sessions = tmp_path / "sessions"
    victim = sessions / "victim"
    victim.mkdir(parents=True)  # empty, first in line to be reaped
    other = _session(sessions, "other", size=100)

    # A live session claims ``victim`` while the sweep is busy sizing ``other``.
    original = retention._dir_size

    def claim_mid_sweep(directory):
        if directory.name == "other" and not (victim / ".session.pid").exists():
            retention.claim_session(victim, pid=os.getpid())
        return original(directory)

    retention._dir_size = claim_mid_sweep
    try:
        sweep_sessions(sessions)
    finally:
        retention._dir_size = original

    assert victim.exists(), "a claim that landed mid-sweep was ignored"
    assert (other / "transcript.jsonl").exists()


def test_the_liveness_marker_is_not_charged_as_bytes(tmp_path):
    """The marker must not count toward a directory's size, or a dead run's
    marker-only directory would read as non-empty and never be reaped."""
    from local_operator.session.retention import _dir_size

    directory = tmp_path / "sessions" / "sess"
    directory.mkdir(parents=True)
    (directory / ".session.pid").write_text("12345")

    assert _dir_size(directory) == 0


def test_origin_marker_is_still_charged_as_content(tmp_path):
    """#154/#192 protect an aborted child stamped with only ``origin.json``.
    That marker is content, not a sidecar, so it must still count — the claim
    work does not narrow that protection."""
    from local_operator.session.retention import _dir_size

    directory = tmp_path / "sessions" / "child"
    directory.mkdir(parents=True)
    (directory / "origin.json").write_text('{"origin": "subagent"}')

    assert _dir_size(directory) > 0
    result = sweep_sessions(tmp_path / "sessions")
    assert result.evicted == 0
    assert directory.exists()


def test_on_an_unverifiable_platform_a_stale_claim_expires(tmp_path, monkeypatch):
    """Where liveness cannot be probed (Windows), a claim cannot be trusted
    forever or a leaked marker would disable the reap one crash at a time. It
    is bounded by ``CLAIM_TRUST_S`` measured from the later of the marker and
    the last write. A marker older than the bound on an otherwise-empty
    directory reads as abandoned and is reaped."""
    import os

    import local_operator.session.retention as retention

    # Force the unverifiable branch and make every pid look alive, so ONLY the
    # age bound can decide — this isolates the bound from the pid probe.
    monkeypatch.setattr(retention, "_PLATFORM", "win32")
    monkeypatch.setattr(retention, "_LIVENESS_IS_VERIFIABLE", False)

    sessions = tmp_path / "sessions"
    stale = sessions / "stale"
    stale.mkdir(parents=True)
    marker = stale / ".session.pid"
    marker.write_text("12345")
    # Age the marker well past the trust window.
    old = time.time() - retention.CLAIM_TRUST_S - 3600
    os.utime(marker, (old, old))

    result = sweep_sessions(sessions)

    assert not stale.exists()
    assert result.evicted == 1


def test_on_an_unverifiable_platform_a_fresh_claim_is_kept(tmp_path, monkeypatch):
    """The counterpart: on the same unverifiable platform a claim written
    within the trust window protects its empty directory, so a session that
    just started (resume of an old transcript, marker fresh) is not reaped
    before it writes its first turn."""
    import local_operator.session.retention as retention

    monkeypatch.setattr(retention, "_PLATFORM", "win32")
    monkeypatch.setattr(retention, "_LIVENESS_IS_VERIFIABLE", False)

    sessions = tmp_path / "sessions"
    fresh = sessions / "fresh"
    fresh.mkdir(parents=True)
    (fresh / ".session.pid").write_text("12345")  # written now, inside the window

    result = sweep_sessions(sessions)

    assert fresh.exists()
    assert result.evicted == 0
