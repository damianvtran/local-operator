"""Retention sweep over the ephemeral session store.

The behaviour under test is a deletion, so the tests care as much about what
survives as about what goes: the live session and the newest history must be
there afterwards, on every path, including the ones that fail.
"""

from __future__ import annotations

import time

import pytest

from local_operator.session.retention import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_SESSIONS,
    sweep_from_config,
    sweep_sessions,
)


def _session(root, name: str, *, size: int = 1024, age_days: float = 0.0):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("x" * size)
    when = time.time() - age_days * 86400
    for path in (directory / "transcript.jsonl", directory):
        import os

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
    turn; 23 of 147 directories on a real install were exactly this."""
    sessions = tmp_path / "sessions"
    (sessions / "hollow").mkdir(parents=True)
    _session(sessions, "real")

    sweep_sessions(sessions, max_sessions=0, max_bytes=0, max_age_days=0)

    assert not (sessions / "hollow").exists()
    assert (sessions / "real").exists()


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
