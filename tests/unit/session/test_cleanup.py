"""The session cleanup policy: OFF does nothing, ON obeys every hard guard,
and every removal is recorded.

These are mostly NEGATIVE controls, because the asymmetry is total: a kept
directory costs disk, a removed one costs somebody their conversation. The
first test is the one the incident demands — every limit set to its most
aggressive value, fifty sessions, ``enabled: false``, fifty remain.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.session import cleanup
from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    STORE_MARKER_NAME,
    CleanupPolicy,
    cleanup_from_config,
    mark_store,
    policy_from_config,
    remove_session_dir,
    run_cleanup,
)
from local_operator.session.retention import LIVE_MARKER_NAME

DAY = 86400.0
NOW = time.time()


def _session(
    root: Path,
    name: str,
    *,
    transcript: bool = True,
    age_days: float = 30.0,
    size: int = 100,
) -> Path:
    directory = root / "sessions" / name
    directory.mkdir(parents=True, exist_ok=True)
    stamp = NOW - age_days * DAY
    if transcript:
        (directory / "transcript.jsonl").write_text("x" * size, encoding="utf-8")
        os.utime(directory / "transcript.jsonl", (stamp, stamp))
    os.utime(directory, (stamp, stamp))
    return directory


def _store(
    root: Path,
    count: int,
    *,
    transcript: bool = True,
    size: int = 100,
    age_days: float | None = None,
) -> list[Path]:
    """``count`` marked sessions, ``s000`` oldest. With ``age_days`` unset the
    ages are distinct (``count+1`` .. ``2`` days) so activity order is total."""
    mark_store(root / "sessions")
    made = []
    for index in range(count):
        age = age_days if age_days is not None else count - index + 1
        made.append(_session(root, f"s{index:03d}", transcript=transcript, age_days=age, size=size))
    return made


def _names(root: Path) -> set[str]:
    return {p.name for p in (root / "sessions").iterdir() if p.is_dir()}


AGGRESSIVE = CleanupPolicy(
    enabled=False,
    max_sessions=1,
    max_inactive_days=1,
    max_total_bytes=1,
    remove_empty=True,
)


# ---------------------------------------------------------------------------
# OFF means off
# ---------------------------------------------------------------------------


def test_disabled_policy_removes_nothing_whatever_the_limits(tmp_path: Path) -> None:
    _store(tmp_path, 25, transcript=True)
    for index in range(25):
        _session(tmp_path, f"empty{index:02d}", transcript=False, age_days=400)
    assert len(_names(tmp_path)) == 50

    result = run_cleanup(tmp_path, AGGRESSIVE, now=NOW)

    assert result.skipped == "disabled"
    assert result.removed == [] and result.chosen == []
    assert len(_names(tmp_path)) == 50
    assert not (tmp_path / "sessions" / CLEANUP_LOG_NAME).exists()


def test_cleanup_from_config_with_default_config_does_nothing(tmp_path: Path) -> None:
    """The startup entry point on a fresh config: five reads, no listing."""
    _store(tmp_path, 50, transcript=False, age_days=400)
    manager = ConfigManager(tmp_path)
    result = cleanup_from_config(manager, tmp_path)
    assert result.skipped == "disabled"
    assert result.scanned == 0
    assert len(_names(tmp_path)) == 50


def test_cleanup_from_config_ignores_limits_when_enabled_is_false(tmp_path: Path) -> None:
    _store(tmp_path, 50, transcript=False, age_days=400)
    manager = ConfigManager(tmp_path)
    manager.update_config(
        {
            "session": {
                "cleanup": {
                    "enabled": False,
                    "max_sessions": 1,
                    "max_inactive_days": 1,
                    "max_total_bytes": 1,
                    "remove_empty": True,
                }
            }
        }
    )
    assert cleanup_from_config(ConfigManager(tmp_path), tmp_path).skipped == "disabled"
    assert len(_names(tmp_path)) == 50


def test_enabled_with_no_limits_removes_nothing(tmp_path: Path) -> None:
    _store(tmp_path, 20, transcript=False, age_days=400)
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True), now=NOW)
    assert result.skipped == "no limits configured"
    assert len(_names(tmp_path)) == 20


# ---------------------------------------------------------------------------
# Reading the policy through the SAME path settings_io writes
# ---------------------------------------------------------------------------


def test_policy_is_read_through_the_nested_path(tmp_path: Path) -> None:
    """The #576 bug: the toggle was written nested and read flat. The policy
    reader walks the nested path, so a value written by ``settings_io`` is
    the value the consumer sees."""
    from local_operator import settings_io

    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, settings_io.BY_KEY["session.cleanup.enabled"], True)
    settings_io.write_setting(manager, settings_io.BY_KEY["session.cleanup.max_sessions"], 7)
    policy = policy_from_config(ConfigManager(tmp_path))
    assert policy.enabled is True
    assert policy.max_sessions == 7


def test_a_flat_dotted_key_is_not_honoured(tmp_path: Path) -> None:
    """A stray ``"session.cleanup.enabled": true`` at the top level (the shape
    of the old bug) must not enable anything."""
    manager = ConfigManager(tmp_path)
    manager.update_config({"session.cleanup.enabled": True, "session.cleanup.remove_empty": True})
    assert policy_from_config(ConfigManager(tmp_path)).enabled is False


def test_garbage_values_fall_back_to_defaults(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    manager.update_config({"session": {"cleanup": {"enabled": True, "max_sessions": "lots"}}})
    policy = policy_from_config(ConfigManager(tmp_path))
    assert policy.enabled is True
    assert policy.max_sessions == 0


def test_a_manager_without_the_nested_reader_yields_the_disabled_default() -> None:
    class Old:
        def get_config_value(self, key: str, default: object = None) -> object:
            return True  # would have enabled everything under the old accessor

    assert policy_from_config(Old()) == CleanupPolicy()


# ---------------------------------------------------------------------------
# Enabled: the limits, and the hard guards on each
# ---------------------------------------------------------------------------


def test_remove_empty_takes_only_transcriptless_directories(tmp_path: Path) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    empty = _session(tmp_path, "empty", transcript=False, age_days=5)
    (empty / "attachment.json").write_text("{}", encoding="utf-8")
    zero = _session(tmp_path, "zero", transcript=False, age_days=5)
    (zero / "transcript.jsonl").write_text("", encoding="utf-8")
    # Writing the files above restamped "last activity" to now, which would
    # put both inside the ten-most-recent guard; age them back down.
    for path in (empty, zero):
        for entry in (path, *path.iterdir()):
            os.utime(entry, (NOW - 5 * DAY, NOW - 5 * DAY))

    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW)

    assert {c.session for c in result.removed} == {"empty", "zero"}
    assert all(c.policy == "remove_empty" for c in result.removed)
    assert len(_names(tmp_path)) == 12
    assert not empty.exists() and not zero.exists()


def test_max_inactive_days_uses_last_activity_not_creation(tmp_path: Path) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    old = _session(tmp_path, "old", age_days=40)
    # A sidecar written yesterday must NOT count as activity.
    (old / ".session.pid").write_text("0", encoding="utf-8")
    touched = _session(tmp_path, "touched", age_days=40)
    (touched / "notes.txt").write_text("recent", encoding="utf-8")

    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_inactive_days=30), now=NOW)

    assert {c.session for c in result.removed} == {"old"}
    assert result.removed[0].policy == "max_inactive_days"
    assert touched.exists()


def test_max_sessions_keeps_the_most_recently_active(tmp_path: Path) -> None:
    made = _store(tmp_path, 30, transcript=True)
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=15), now=NOW)
    survivors = _names(tmp_path)
    assert len(survivors) == 15
    # s000 is the oldest (age 31 d), s029 the newest.
    assert survivors == {p.name for p in made[15:]}
    assert all(c.policy == "max_sessions" for c in result.removed)


def test_max_total_bytes_trims_least_recent_first(tmp_path: Path) -> None:
    _store(tmp_path, 20, transcript=True, size=1000)  # 20 kB
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_total_bytes=12_000), now=NOW)
    assert len(_names(tmp_path)) == 12
    assert {c.session for c in result.removed} == {f"s{i:03d}" for i in range(8)}
    assert all(c.policy == "max_total_bytes" for c in result.removed)


def test_the_ten_most_recent_are_never_candidates(tmp_path: Path) -> None:
    _store(tmp_path, 10, transcript=False, age_days=400)
    result = run_cleanup(tmp_path, dataclasses.replace(AGGRESSIVE, enabled=True), now=NOW)
    assert result.removed == []
    assert len(_names(tmp_path)) == 10
    assert all(guard.startswith("one of the 10") for _, guard in result.protected)


@pytest.mark.parametrize(
    "shape",
    ["claimed", "leased", "wake", "mail", "live_dir"],
)
def test_hard_guards_hold_when_enabled(tmp_path: Path, shape: str) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    victim = _session(tmp_path, "victim", transcript=False, age_days=400)
    live_dir = None
    if shape == "claimed":
        (victim / LIVE_MARKER_NAME).write_text(str(os.getpid()), encoding="utf-8")
    elif shape == "leased":
        (victim / ".execution-lease").write_text(
            json.dumps({"generation": "g", "pid": os.getpid()}), encoding="utf-8"
        )
    elif shape == "wake":
        from local_operator.wakes.store import wakes_dir

        wakes_dir(tmp_path).mkdir(parents=True, exist_ok=True)
        (wakes_dir(tmp_path) / "victim.json").write_text("{}", encoding="utf-8")
    elif shape == "mail":
        (victim / "inbox.jsonl").write_text('{"from":"peer"}\n', encoding="utf-8")
    elif shape == "live_dir":
        live_dir = victim
    os.utime(victim, (NOW - 400 * DAY, NOW - 400 * DAY))

    policy = CleanupPolicy(enabled=True, remove_empty=True, max_inactive_days=1)
    result = run_cleanup(tmp_path, policy, now=NOW, live_dir=live_dir)

    assert victim.exists(), shape
    assert "victim" not in {c.session for c in result.removed}
    assert any(name == "victim" for name, _ in result.protected), result.protected


def test_a_dead_claim_does_not_protect(tmp_path: Path) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    victim = _session(tmp_path, "victim", transcript=False, age_days=400)
    (victim / LIVE_MARKER_NAME).write_text("999999999", encoding="utf-8")
    os.utime(victim, (NOW - 400 * DAY, NOW - 400 * DAY))
    run_cleanup(tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW)
    assert not victim.exists()


# ---------------------------------------------------------------------------
# Dry run, logging, the record
# ---------------------------------------------------------------------------


def test_dry_run_removes_nothing_but_reports_and_records(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        result = run_cleanup(
            tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW, dry_run=True
        )
    assert [c.session for c in result.removed] == ["doomed"]
    assert (tmp_path / "sessions" / "doomed").exists()
    assert any("dry run" in r.message and "doomed" in r.message for r in caplog.records)
    rows = [
        json.loads(line)
        for line in (tmp_path / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
    ]
    assert rows == [
        {
            "at": rows[0]["at"],
            "session": "doomed",
            "policy": "remove_empty",
            "reason": "no transcript",
            "dry_run": True,
        }
    ]


def test_every_removal_is_logged_at_warning_with_policy_and_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        run_cleanup(tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW)
    lines = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "removing doomed" in m and "policy=remove_empty" in m and "reason=no transcript" in m
        for m in lines
    ), lines
    row = json.loads((tmp_path / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()[0])
    assert row["session"] == "doomed" and row["dry_run"] is False


def test_explicit_run_does_not_need_enabled(tmp_path: Path) -> None:
    """The CLI path: the command is the consent, the guards still apply."""
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    result = run_cleanup(
        tmp_path, CleanupPolicy(enabled=False, remove_empty=True), now=NOW, explicit=True
    )
    assert [c.session for c in result.removed] == ["doomed"]


# ---------------------------------------------------------------------------
# The remover refuses anything that is not a marked store
# ---------------------------------------------------------------------------


def test_remover_refuses_an_unmarked_store(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "sessions" / "abc"
    target.mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        done = remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r")
    assert done is False and target.exists()
    assert any("REFUSED" in r.message and STORE_MARKER_NAME in r.message for r in caplog.records)


def test_remover_refuses_a_directory_not_under_sessions(tmp_path: Path) -> None:
    target = tmp_path / "agents" / "abc"
    target.mkdir(parents=True)
    (tmp_path / "agents" / STORE_MARKER_NAME).write_text("x")
    assert remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r") is False
    assert target.exists()


def test_remover_refuses_another_config_dirs_store(tmp_path: Path) -> None:
    other = tmp_path / "other"
    target = other / "sessions" / "abc"
    target.mkdir(parents=True)
    mark_store(other / "sessions")
    mine = tmp_path / "mine"
    (mine / "sessions").mkdir(parents=True)
    assert remove_session_dir(target, config_dir=mine, policy="p", reason="r") is False
    assert target.exists()


def test_remover_refuses_a_symlink_into_a_store(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    victim = real_root / "sessions" / "abc"
    victim.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    mark_store(scratch / "sessions")
    link = scratch / "sessions" / "abc"
    link.symlink_to(victim)
    assert remove_session_dir(link, config_dir=scratch, policy="p", reason="r") is False
    assert victim.exists()


def test_remover_removes_a_marked_store_entry(tmp_path: Path) -> None:
    mark_store(tmp_path / "sessions")
    target = tmp_path / "sessions" / "abc"
    target.mkdir()
    (target / "transcript.jsonl").write_text("row\n")
    assert remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r") is True
    assert not target.exists()


def test_mark_store_is_idempotent(tmp_path: Path) -> None:
    mark_store(tmp_path / "sessions")
    first = (tmp_path / "sessions" / STORE_MARKER_NAME).read_text()
    mark_store(tmp_path / "sessions")
    assert (tmp_path / "sessions" / STORE_MARKER_NAME).read_text() == first


def test_cleanup_never_marks_a_store_itself(tmp_path: Path) -> None:
    """A store without the marker stays out of reach even when enabled: the
    policy must not be able to authorise its own target."""
    (tmp_path / "sessions").mkdir()
    for index in range(12):
        _session(tmp_path, f"s{index}", transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW)
    assert result.removed == []
    assert (tmp_path / "sessions" / "doomed").exists()
    assert not (tmp_path / "sessions" / STORE_MARKER_NAME).exists()


def test_module_has_exactly_one_rmtree() -> None:
    import inspect

    source = inspect.getsource(cleanup)
    assert source.count("shutil.rmtree(") == 1
