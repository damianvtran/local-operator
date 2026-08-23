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
