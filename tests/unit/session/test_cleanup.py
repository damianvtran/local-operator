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
    (touched / "transcript.jsonl").write_text("x" * 100, encoding="utf-8")  # written now
    # A backfill sentinel written yesterday must NOT count as activity either.
    stale = _session(tmp_path, "stale", age_days=40)
    (stale / "title-scan.json").write_text("{}", encoding="utf-8")
    (stale / "notes.txt").write_text("unknown file, also not activity", encoding="utf-8")

    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_inactive_days=30), now=NOW)

    assert {c.session for c in result.removed} == {"old", "stale"}
    assert all(c.policy == "max_inactive_days" for c in result.removed)
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
    _store(tmp_path, 10, transcript=True, age_days=400)
    result = run_cleanup(tmp_path, dataclasses.replace(AGGRESSIVE, enabled=True), now=NOW)
    assert result.removed == []
    assert len(_names(tmp_path)) == 10
    assert all(guard.startswith("one of the 10") for _, guard in result.protected)


def test_never_active_directories_are_outside_the_ranked_set(tmp_path: Path) -> None:
    """UX round 2, U11 — the reviewer's scenario, with enough conversations
    that BOTH the recent-10 guard and ``max_sessions`` bite. 14
    conversations, ``max_sessions: 5``, then eleven idle open-and-quit
    launches (empty directories, each newer than every conversation). The
    old fallback ranked each launch as the MOST recent session, so eleven
    launches filled the recent-10 with empties and emptied the store of
    conversations. Now: the 10 most recently ACTIVE conversations survive
    (the recent-N guard outranks the cap, as before), and every launch
    directory is untouched unless ``remove_empty`` — it has no activity and
    no rank."""
    from local_operator.resume import recent_sessions

    mark_store(tmp_path / "sessions")
    for index in range(14):
        _session(tmp_path, f"conv{index:02d}", transcript=True, age_days=30 - index)
    for index in range(11):
        directory = tmp_path / "sessions" / f"launch{index:02d}"
        directory.mkdir()
        os.utime(directory, (NOW - index, NOW - index))  # all newer than any conversation
    policy = CleanupPolicy(enabled=True, max_sessions=5)
    result = run_cleanup(tmp_path, policy, now=NOW)
    # conv13 newest. Recent-10 = conv04..conv13 (protected); the cap takes
    # the rest — conv00..conv03 — and NOT a single launch directory.
    assert {c.session for c in result.removed} == {f"conv{i:02d}" for i in range(4)}
    survivors = _names(tmp_path)
    assert {f"conv{i:02d}" for i in range(4, 14)} <= survivors
    assert {f"launch{i:02d}" for i in range(11)} <= survivors
    # The picker and the policy agree: the picker's rows are exactly the
    # ranked set, in the same order, and no launch directory has a row.
    picker = [name for name, _stamp in recent_sessions(tmp_path)]
    assert picker == [f"conv{i:02d}" for i in range(13, 3, -1)]
    # A second wave of launches changes nothing.
    for index in range(11, 22):
        (tmp_path / "sessions" / f"launch{index:02d}").mkdir()
    again = run_cleanup(tmp_path, policy, now=NOW)
    assert again.removed == []
    assert {f"conv{i:02d}" for i in range(4, 14)} <= _names(tmp_path)
    # ``remove_empty`` is the ONLY policy that takes them.
    swept = run_cleanup(tmp_path, dataclasses.replace(policy, remove_empty=True), now=NOW)
    assert {c.session for c in swept.removed} == {f"launch{i:02d}" for i in range(22)}
    assert all(c.policy == "remove_empty" for c in swept.removed)
    assert {f"conv{i:02d}" for i in range(4, 14)} == _names(tmp_path)


def test_the_recent_guard_is_the_pickers_first_page(tmp_path: Path) -> None:
    """QA round 2, Q8. The 10 newest directories are subagent-origin or
    empty; the 10 rows the user sees on ``/resume`` are OLDER user-origin
    transcripts. ``max_sessions: 12`` must keep every picker row: the
    recent-N guard is ``recent_sessions(limit=10)`` itself, not a ranking
    of every directory. Subagent-origin transcripts still rank under the
    cap by activity — they are real transcripts, never "empty" — so the
    12 kept are the 10 picker rows plus the 2 most recent subagent runs."""
    from local_operator.resume import (
        ORIGIN_SUBAGENT,
        mark_session_origin,
        recent_sessions,
    )

    mark_store(tmp_path / "sessions")
    for index in range(10):  # user-origin, 30..21 d old
        _session(tmp_path, f"user{index:02d}", transcript=True, age_days=30 - index)
    for index in range(7):  # subagent-origin, 7..1 d old — newer than every user row
        directory = _session(tmp_path, f"sub{index:02d}", transcript=True, age_days=7 - index)
        mark_session_origin(directory, ORIGIN_SUBAGENT)
    for index in range(3):  # empty, newest of all
        (tmp_path / "sessions" / f"empty{index:02d}").mkdir()
    picker = [name for name, _stamp in recent_sessions(tmp_path, limit=10)]
    assert picker == [f"user{i:02d}" for i in range(9, -1, -1)]
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=12), now=NOW)
    survivors = _names(tmp_path)
    assert {f"user{i:02d}" for i in range(10)} <= survivors, "a picker row was removed"
    assert {f"empty{i:02d}" for i in range(3)} <= survivors
    # Ranked set = 17 transcripts; cap 12; guard protects the 10 user rows;
    # the 5 oldest subagent runs go (sub00..sub04), sub05/sub06 stay.
    assert {c.session for c in result.removed} == {f"sub{i:02d}" for i in range(5)}
    assert all(c.policy == "max_sessions" for c in result.removed)
    assert {"sub05", "sub06"} <= survivors
    # The guard rows are reported as the picker's, by name.
    guarded = {name for name, guard in result.protected if guard.startswith("one of the 10")}
    assert guarded <= set(picker)


def test_equal_stamps_are_stable_across_consecutive_runs(tmp_path: Path) -> None:
    """QA round 2, Q10: consecutive launches must not each shave one more
    equal-stamped session. With a name tie-break the ranked order is a
    property of the store, so a second run over the survivors removes
    nothing and a third run removes nothing."""
    mark_store(tmp_path / "sessions")
    for index in range(15):
        _session(tmp_path, f"t{index:02d}", transcript=True, age_days=30)  # identical stamps
    policy = CleanupPolicy(enabled=True, max_sessions=12)
    first = run_cleanup(tmp_path, policy, now=NOW)
    assert [c.session for c in first.removed] == ["t14", "t13", "t12"]
    for _ in range(3):
        assert run_cleanup(tmp_path, policy, now=NOW).removed == []
    assert len(_names(tmp_path)) == 12


def test_picker_and_policy_share_one_ranking(tmp_path: Path) -> None:
    """The picker's order IS the policy's order, by construction: both call
    ``session_activity``. A sidecar newer than every transcript and an
    inbox spool newer still must move the row on both surfaces or neither."""
    from local_operator.resume import recent_sessions
    from local_operator.session.retention import session_activity

    made = _store(tmp_path, 6, transcript=True)  # s005 newest
    (made[1] / "title-scan.json").write_text("{}")  # bookkeeping: no effect
    (made[0] / "inbox.jsonl").write_text('{"from":"peer"}\n')  # unread mail: activity
    picker = [name for name, _stamp in recent_sessions(tmp_path)]
    by_clock = sorted(
        (p.name for p in made), key=lambda n: -(session_activity(tmp_path / "sessions" / n) or 0)
    )
    assert picker == by_clock == ["s000", "s005", "s004", "s003", "s002", "s001"]


def test_apply_removes_exactly_the_previewed_set(tmp_path: Path) -> None:
    """Review round 2, R2-2 — reproduced: a session created between the
    preview and the confirmation shifted the recent-10 window and a row
    shown as KEPT was removed. ``apply_cleanup(plan)`` removes the plan's
    set and nothing else; only the hard guards are re-checked."""
    from local_operator.session.cleanup import apply_cleanup

    _store(tmp_path, 15, transcript=True)  # s014 newest; recent-10 = s005..s014
    policy = CleanupPolicy(enabled=True, max_sessions=10)
    plan = run_cleanup(tmp_path, policy, now=NOW, dry_run=True)
    assert {c.session for c in plan.removed} == {f"s{i:03d}" for i in range(5)}
    assert ("s005", "one of the 10 most recent") not in plan.protected  # not even a candidate
    # Meanwhile: a brand-new conversation lands, newer than everything.
    _session(tmp_path, "fresh", transcript=True, age_days=0.5)
    # And one planned directory acquires unread mail before the removal.
    (tmp_path / "sessions" / "s004" / "inbox.jsonl").write_text('{"from":"peer"}\n')
    result = apply_cleanup(tmp_path, plan, actor="cli", now=NOW)
    assert {c.session for c in result.removed} == {f"s{i:03d}" for i in range(4)}
    assert ("s004", "has unread spooled mail (since the preview)") in result.protected
    # s005 — shown as kept — is still there, even though a re-scan would
    # now rank it 11th.
    assert (tmp_path / "sessions" / "s005").is_dir()
    assert (tmp_path / "sessions" / "fresh").is_dir()


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


def test_dry_run_removes_nothing_and_records_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A rehearsal is reported through the result, never through the log:
    a WARNING per dry-run line doubled the CLI's output (U3) and rehearsals
    in the jsonl made the record of real losses unreadable (U6/R1-12)."""
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        result = run_cleanup(
            tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW, dry_run=True
        )
    assert [c.session for c in result.removed] == ["doomed"]
    assert (tmp_path / "sessions" / "doomed").exists()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not (tmp_path / "sessions" / CLEANUP_LOG_NAME).exists()


def test_dry_run_with_the_switch_off_lists_but_reports_disabled(tmp_path: Path) -> None:
    """The safe, useful half of the old CLI bypass: a dry run may show what
    the config's limits WOULD take while the switch is off, and says so."""
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    result = run_cleanup(
        tmp_path, CleanupPolicy(enabled=False, remove_empty=True), now=NOW, dry_run=True
    )
    assert [c.session for c in result.removed] == ["doomed"]
    assert result.skipped == "disabled"
    assert (tmp_path / "sessions" / "doomed").exists()


def test_candidates_carry_title_age_and_size(tmp_path: Path) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    old = _session(tmp_path, "old", age_days=40, size=2048)
    row = json.dumps(
        {
            "type": "message",
            "payload": {
                "role": "user",
                "content": [{"type": "text", "text": "Research thread seven"}],
            },
        }
    )
    (old / "transcript.jsonl").write_text(row + "\n" + "x" * 2000, encoding="utf-8")
    os.utime(old / "transcript.jsonl", (NOW - 40 * DAY, NOW - 40 * DAY))
    result = run_cleanup(
        tmp_path, CleanupPolicy(enabled=True, max_inactive_days=30), now=NOW, dry_run=True
    )
    (candidate,) = result.removed
    assert candidate.session == "old"
    assert "Research thread seven" in candidate.title
    assert 39.9 < candidate.idle_days < 40.1
    assert candidate.size_bytes > 2000


def test_every_removal_is_logged_at_warning_with_policy_and_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        run_cleanup(tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW)
    lines = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "removing doomed" in m
        and "policy=remove_empty" in m
        and "reason=no transcript" in m
        and "actor=startup" in m
        and CLEANUP_LOG_NAME in m
        for m in lines
    ), lines
    row = json.loads((tmp_path / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()[0])
    assert row["session"] == "doomed"
    assert row["actor"] == "startup" and row["pid"] == os.getpid()
    assert "dry_run" not in row


def test_the_switch_governs_every_caller_including_the_cli(tmp_path: Path) -> None:
    """Round 1 let a caller pass ``explicit=True`` to bypass the switch; that
    was the incident's shape (Q1/U2/R1-5). There is no such flag now: with
    ``enabled: false`` a non-dry run removes nothing, whoever calls."""
    import inspect

    from local_operator.session import cleanup as cleanup_mod

    assert "explicit" not in inspect.signature(run_cleanup).parameters
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    result = run_cleanup(
        tmp_path, CleanupPolicy(enabled=False, remove_empty=True), now=NOW, actor="cli"
    )
    assert result.skipped == "disabled" and result.removed == []
    assert (tmp_path / "sessions" / "doomed").exists()
    assert not (tmp_path / "sessions" / cleanup_mod.CLEANUP_LOG_NAME).exists()


def test_force_overrides_the_switch_and_is_recorded_as_the_actor(tmp_path: Path) -> None:
    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    result = run_cleanup(
        tmp_path,
        CleanupPolicy(enabled=False, remove_empty=True),
        now=NOW,
        force=True,
        actor="cli",
    )
    assert [c.session for c in result.removed] == ["doomed"]
    row = json.loads((tmp_path / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()[0])
    assert row["actor"] == "cli"


def test_a_boot_does_not_move_the_activity_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 1, Q2: the startup backfills write ``title-scan.json`` /
    ``origin-scan.json`` into every transcript directory, and the old clock
    (newest non-sidecar file) read a 40-day session as active NOW after one
    boot. Run the REAL maintenance pass, then assert every session's activity
    is unchanged and ``max_inactive_days`` still finds the old ones."""
    import asyncio

    from local_operator import session_factory
    from local_operator.session.retention import _activity_mtime

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    made = _store(tmp_path, 15, transcript=True)  # ages 16d .. 2d
    row = json.dumps({"type": "message", "payload": {"role": "user", "content": "hello"}})
    for path in made:
        (path / "transcript.jsonl").write_text(row + "\n", encoding="utf-8")
        stamp = path.stat().st_mtime
        os.utime(path / "transcript.jsonl", (stamp, stamp))
    before = {p.name: _activity_mtime(p, 0.0) for p in made}
    manager = ConfigManager(tmp_path)

    async def no_wait() -> None:
        return None

    monkeypatch.setattr(session_factory, "_wait_for_store_maintenance_idle_window", no_wait)
    asyncio.run(session_factory._run_store_maintenance(manager, tmp_path, live_dir=None))

    stamped = [p for p in made if (p / "title-scan.json").exists() or (p / "origin.json").exists()]
    assert stamped, "the backfills did not run; the test proves nothing"
    after = {p.name: _activity_mtime(p, 0.0) for p in made}
    assert after == before
    result = run_cleanup(
        tmp_path, CleanupPolicy(enabled=True, max_inactive_days=10), now=NOW, dry_run=True
    )
    # Ages 16..11 d are older than 10 d: s000..s005; the 10 most recent are
    # s005..s014 by the transcript clock, so s000..s004 are chosen.
    assert {c.session for c in result.removed} == {f"s{i:03d}" for i in range(5)}


def test_ties_break_on_name_so_dry_run_and_real_run_agree(tmp_path: Path) -> None:
    mark_store(tmp_path / "sessions")
    for index in range(15):
        _session(tmp_path, f"t{index:02d}", age_days=30)  # all the same stamp
    dry = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=10), now=NOW, dry_run=True)
    assert [c.session for c in dry.removed] == ["t14", "t13", "t12", "t11", "t10"]
    real = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=10), now=NOW)
    assert [c.session for c in real.removed] == [c.session for c in dry.removed]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (True, True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("yes", True),
        (1, True),
        (0, False),
        ("banana", False),
        ([], False),
        ({"x": 1}, False),
        (2, False),
    ],
)
def test_the_master_switch_is_parsed_strictly(tmp_path: Path, raw: object, expected: bool) -> None:
    """``enabled: "false"`` in a hand-edited YAML must not enable cleanup
    (R1-6): anything that is not a recognisable boolean reads as OFF."""
    manager = ConfigManager(tmp_path)
    manager.update_config({"session": {"cleanup": {"enabled": raw}}})
    assert policy_from_config(ConfigManager(tmp_path)).enabled is expected


# ---------------------------------------------------------------------------
# The remover refuses anything that is not a marked store
# ---------------------------------------------------------------------------


def test_remover_refuses_an_unmarked_store(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    target = tmp_path / "sessions" / "abc"
    target.mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger="local_operator.session.cleanup"):
        done = remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")
    assert done is False and target.exists()
    assert any("REFUSED" in r.message and STORE_MARKER_NAME in r.message for r in caplog.records)


def test_remover_refuses_a_directory_not_under_sessions(tmp_path: Path) -> None:
    target = tmp_path / "agents" / "abc"
    target.mkdir(parents=True)
    (tmp_path / "agents" / STORE_MARKER_NAME).write_text("x")
    assert (
        remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")
        is False
    )
    assert target.exists()


def test_remover_refuses_another_config_dirs_store(tmp_path: Path) -> None:
    other = tmp_path / "other"
    target = other / "sessions" / "abc"
    target.mkdir(parents=True)
    mark_store(other / "sessions")
    mine = tmp_path / "mine"
    (mine / "sessions").mkdir(parents=True)
    assert (
        remove_session_dir(target, config_dir=mine, policy="p", reason="r", actor="test") is False
    )
    assert target.exists()


def test_remover_refuses_a_symlink_into_a_store(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    victim = real_root / "sessions" / "abc"
    victim.mkdir(parents=True)
    scratch = tmp_path / "scratch"
    mark_store(scratch / "sessions")
    link = scratch / "sessions" / "abc"
    link.symlink_to(victim)
    assert (
        remove_session_dir(link, config_dir=scratch, policy="p", reason="r", actor="test") is False
    )
    assert victim.exists()


def test_remover_removes_a_marked_store_entry(tmp_path: Path) -> None:
    mark_store(tmp_path / "sessions")
    target = tmp_path / "sessions" / "abc"
    target.mkdir()
    (target / "transcript.jsonl").write_text("row\n")
    assert (
        remove_session_dir(target, config_dir=tmp_path, policy="p", reason="r", actor="test")
        is True
    )
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


# ---------------------------------------------------------------------------
# The launch notice (UX round 1, U1 second half)
# ---------------------------------------------------------------------------


def test_a_removing_startup_pass_leaves_a_last_cleanup_record(tmp_path: Path) -> None:
    from local_operator.session.cleanup import (
        LAST_CLEANUP_NAME,
        cleanup_from_config,
        take_unannounced_cleanup,
    )

    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    manager = ConfigManager(tmp_path)
    manager.update_config({"session": {"cleanup": {"enabled": True, "remove_empty": True}}})
    result = cleanup_from_config(ConfigManager(tmp_path), tmp_path)
    assert [c.session for c in result.removed] == ["doomed"]
    record = json.loads((tmp_path / "sessions" / LAST_CLEANUP_NAME).read_text())
    assert record["removed"] == 1 and record["policies"] == {"remove_empty": 1}
    assert record["actor"] == "startup" and record["acknowledged"] is False
    # First reader takes it; second reader gets nothing — one announcement.
    taken = take_unannounced_cleanup(tmp_path / "sessions")
    assert taken is not None and taken["removed"] == 1
    assert take_unannounced_cleanup(tmp_path / "sessions") is None
    assert json.loads((tmp_path / "sessions" / LAST_CLEANUP_NAME).read_text())["acknowledged"]


def test_a_pass_that_removed_nothing_writes_no_record(tmp_path: Path) -> None:
    from local_operator.session.cleanup import LAST_CLEANUP_NAME, cleanup_from_config

    _store(tmp_path, 12, transcript=True, age_days=1)
    ConfigManager(tmp_path).update_config(
        {"session": {"cleanup": {"enabled": True, "remove_empty": True}}}
    )
    result = cleanup_from_config(ConfigManager(tmp_path), tmp_path)
    assert result.removed == []
    assert not (tmp_path / "sessions" / LAST_CLEANUP_NAME).exists()


def test_the_notice_names_count_policy_record_and_preview() -> None:
    from local_operator.session.cleanup import format_cleanup_notice

    text = format_cleanup_notice(
        {
            "removed": 9,
            "policies": {"max_inactive_days": 7, "remove_empty": 2},
            "record": os.path.expanduser("~/.local-operator/sessions/.cleanup-log.jsonl"),
        }
    )
    assert text.startswith("session cleanup removed 9 sessions at launch")
    assert "7 by max_inactive_days" in text and "2 by remove_empty" in text
    assert "~/.local-operator/sessions/.cleanup-log.jsonl" in text
    assert "lop sessions cleanup --dry-run" in text and "/settings" in text
