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
from typing import Any

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
    """QA round 2, Q8 + UX round 3, U15. The 10 newest directories are
    subagent-origin or empty; the 10 rows the user sees on ``/resume`` are
    OLDER user-origin transcripts. ``max_sessions: 12`` must keep every
    picker row — the recent-N guard is ``recent_sessions(limit=10)``
    itself — and, since the cap counts in the picker's unit, must touch
    NO subagent-origin directory at all: 10 conversations is under 12."""
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
    assert result.removed == []
    assert len(_names(tmp_path)) == 20
    # The guard rows are reported as the picker's, by name — none needed here.
    guarded = {name for name, guard in result.protected if guard.startswith("one of the 10")}
    assert guarded <= set(picker)


def _operator_shaped_store(tmp_path: Path) -> tuple[list[str], list[str]]:
    """UX round 3's store: 210 directories, 31 user conversations, 179
    subagent-origin runs interleaved by age so the subagent runs are mostly
    NEWER than the conversations (as on the operator's real store)."""
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

    mark_store(tmp_path / "sessions")
    users: list[str] = []
    subs: list[str] = []
    for index in range(31):  # 62..2 d old, every other day
        users.append(_session(tmp_path, f"user{index:02d}", age_days=62 - 2 * index).name)
    for index in range(179):  # 0.1..17.9 d old — the newest 179 directories
        directory = _session(tmp_path, f"sub{index:03d}", age_days=0.1 * (index + 1))
        mark_session_origin(directory, ORIGIN_SUBAGENT)
        subs.append(directory.name)
    return users, subs


def test_max_sessions_counts_in_the_pickers_unit(tmp_path: Path) -> None:
    """UX round 3, U15: ``/resume`` says "31 sessions"; ``max_sessions: 50``
    must therefore remove NOTHING — 31 is under 50 — and no subagent-origin
    directory is a ``max_sessions`` candidate at any N. Before: 159 removed,
    21 of them the user's own conversations, the picker held at 10 only by
    the guard floor."""
    from local_operator.resume import recent_sessions

    users, subs = _operator_shaped_store(tmp_path)
    assert len(recent_sessions(tmp_path)) == 31
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=50), now=NOW)
    assert result.removed == []
    assert len(_names(tmp_path)) == 210

    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=20), now=NOW)
    # The 11 OLDEST conversations go; the 20 newest survive; every subagent
    # run is untouched whatever its age.
    assert {c.session for c in result.removed} == set(users[:11])
    assert all(c.policy == "max_sessions" and c.origin == "user" for c in result.removed)
    survivors = _names(tmp_path)
    assert set(users[11:]) <= survivors
    assert set(subs) <= survivors
    assert [name for name, _ in recent_sessions(tmp_path)] == list(reversed(users[11:]))


def test_age_and_byte_limits_still_consider_subagent_runs(tmp_path: Path) -> None:
    """U15 narrows ``max_sessions`` only: staleness and disk are about the
    directory, not about whose it is, so ``max_inactive_days`` takes an old
    subagent run and labels its origin on the row."""
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

    mark_store(tmp_path / "sessions")
    _session(tmp_path, "conv", age_days=2)
    old = _session(tmp_path, "oldsub", age_days=90)
    mark_session_origin(old, ORIGIN_SUBAGENT)
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_inactive_days=30), now=NOW)
    assert [(c.session, c.origin) for c in result.removed] == [("oldsub", "subagent")]


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
    _session(tmp_path, "doomed", transcript=True, age_days=400)
    manager = ConfigManager(tmp_path)
    manager.update_config({"session": {"cleanup": {"enabled": True, "max_inactive_days": 30}}})
    result = cleanup_from_config(ConfigManager(tmp_path), tmp_path)
    assert [c.session for c in result.removed] == ["doomed"]
    record = json.loads((tmp_path / "sessions" / LAST_CLEANUP_NAME).read_text())
    assert record["removed"] == 1 and record["policies"] == {"max_inactive_days": 1}
    assert record["origins"] == {"user": 1} and record["quiet"] is None
    assert record["actor"] == "startup" and record["acknowledged"] is False
    # The writer is THIS process, so this viewer takes it; a second reader
    # gets nothing — one announcement.
    taken = take_unannounced_cleanup(tmp_path / "sessions")
    assert taken is not None and taken["removed"] == 1
    assert take_unannounced_cleanup(tmp_path / "sessions") is None
    assert json.loads((tmp_path / "sessions" / LAST_CLEANUP_NAME).read_text())["acknowledged"]


def test_a_never_active_only_pass_is_recorded_but_quiet(tmp_path: Path) -> None:
    """Design round 3, N6: the record is written (the jsonl and the WARNING
    still say what went) but pre-acknowledged, with the reason, so no viewer
    announces an idle open-and-quit directory's removal."""
    from local_operator.session.cleanup import (
        LAST_CLEANUP_NAME,
        cleanup_from_config,
        take_unannounced_cleanup,
    )

    _store(tmp_path, 12, transcript=True, age_days=1)
    _session(tmp_path, "doomed", transcript=False, age_days=400)
    ConfigManager(tmp_path).update_config(
        {"session": {"cleanup": {"enabled": True, "remove_empty": True}}}
    )
    result = cleanup_from_config(ConfigManager(tmp_path), tmp_path)
    assert [(c.session, c.active) for c in result.removed] == [("doomed", False)]
    record = json.loads((tmp_path / "sessions" / LAST_CLEANUP_NAME).read_text())
    assert record["removed"] == 1 and record["acknowledged"] is True
    assert record["quiet"] == "only never-active directories were removed"
    assert take_unannounced_cleanup(tmp_path / "sessions") is None


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


# ---------------------------------------------------------------------------
# Every guard fails CLOSED (review round 3, R3-1)
# ---------------------------------------------------------------------------


def _aggressive_store(tmp_path: Path, count: int = 15) -> None:
    """A store every limit would gut: old, small, transcripted, unguarded."""
    _store(tmp_path, count, transcript=True, age_days=400)


@pytest.mark.parametrize(
    "target",
    ["_claimed", "_lease_owner_alive", "_has_wake", "_has_spooled_mail"],
)
def test_a_guard_that_raises_keeps_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Each hard guard, made to raise something it did not anticipate: the
    directory is KEPT, the refusal names the guard, nothing is removed."""
    _aggressive_store(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> bool:
        raise MemoryError("probe exploded")

    monkeypatch.setattr(cleanup, target, boom)
    result = run_cleanup(tmp_path, dataclasses.replace(AGGRESSIVE, enabled=True), now=NOW)
    assert result.removed == []
    assert len(_names(tmp_path)) == 15
    # The 10 picker rows are kept by the recent guard first; the other 5 reach
    # the hard guards and must be kept BY the failure, named.
    failed = {name for name, guard in result.protected if guard == "guard failed: MemoryError"}
    assert failed == {f"s{i:03d}" for i in range(10, 15)}  # s000..s009 tie-break as recent


def test_an_unreadable_claim_marker_keeps_the_directory(tmp_path: Path) -> None:
    """A PRESENT marker that cannot be read is "claimed", not "unclaimed":
    ``retention._is_claimed`` answers False to EACCES because it serves
    liveness questions where "assume dead" is right; the cleanup wrapper
    must not inherit that default."""
    import stat

    _aggressive_store(tmp_path, 12)  # s010/s011 sit outside the recent 10
    marker = tmp_path / "sessions" / "s011" / LIVE_MARKER_NAME
    marker.write_text("1", encoding="utf-8")
    marker.chmod(0)
    if os.access(marker, os.R_OK):  # root, or a filesystem without modes
        pytest.skip("cannot make the marker unreadable here")
    try:
        result = run_cleanup(tmp_path, dataclasses.replace(AGGRESSIVE, enabled=True), now=NOW)
    finally:
        marker.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert "s011" in _names(tmp_path)
    assert ("s011", "claimed by a live process") in result.protected
    assert {c.session for c in result.removed} == {"s010"}


def test_a_picker_that_cannot_be_listed_refuses_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-1's reproduction: ``recent_sessions`` raising used to yield an
    EMPTY recent set and the run proceeded with zero recent protection
    (5 -> 12 of 15 removable, nothing logged). Now the run is SKIPPED with
    the cause named, and nothing is removed — in a dry run or a real one."""
    from unittest import mock

    from local_operator import resume

    _store(tmp_path, 15, transcript=True)
    with mock.patch.object(resume, "recent_sessions", side_effect=RuntimeError("readdir")):
        preview = run_cleanup(
            tmp_path, CleanupPolicy(enabled=True, max_sessions=3), now=NOW, dry_run=True
        )
        real = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=3), now=NOW)
    for result in (preview, real):
        assert result.removed == [] and result.chosen == []
        assert result.skipped is not None and result.skipped.startswith("guard unavailable")
        assert "RuntimeError" in result.skipped and result.errors == 1
    assert len(_names(tmp_path)) == 15


def test_a_picker_import_failure_refuses_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lazy ``import local_operator.resume`` failing is the same shape."""
    import builtins
    import sys

    _store(tmp_path, 15, transcript=True)
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "local_operator.resume":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "local_operator.resume", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, max_sessions=3), now=NOW)
    assert result.removed == []
    assert result.skipped is not None and "ImportError" in result.skipped
    assert len(_names(tmp_path)) == 15


def test_an_unresolvable_live_dir_still_protects_nothing_extra_but_removes_nothing_live(
    tmp_path: Path,
) -> None:
    """``live_dir`` that cannot be resolved: the current session is found by
    identity, so an unresolvable path means "no current session here" — the
    hard guards (a live claim) still protect the directory the caller meant."""
    _aggressive_store(tmp_path, 12)
    (tmp_path / "sessions" / "s011" / LIVE_MARKER_NAME).write_text(str(os.getpid()))
    result = run_cleanup(
        tmp_path,
        dataclasses.replace(AGGRESSIVE, enabled=True),
        live_dir=tmp_path / "sessions" / "does-not-exist",
        now=NOW,
    )
    assert "s011" in _names(tmp_path)
    assert ("s011", "claimed by a live process") in result.protected
    assert {c.session for c in result.removed} == {"s010"}


# ---------------------------------------------------------------------------
# The notice record is read defensively (review round 3, R3-2 / QA Q11)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"removed": "many", "acknowledged": False},
        {"removed": 2, "policies": "remove_empty"},
        {"removed": 2, "policies": {"remove_empty": "two"}},
        {"removed": 2, "policies": {3: 1, None: 2}},
        {"removed": 2, "origins": ["user"]},
        {"removed": 2.5, "record": 12},
        {"removed": None},
        {"removed": True, "policies": {"x": True}},
        {"removed": [1, 2]},
        "not even a dict",
        None,
        [],
    ],
)
def test_the_notice_formats_any_record_shape(payload: object) -> None:
    """A hand-edited or newer-schema record must format, never raise: the
    formatter runs at boot and took the first frame down on
    ``"removed": "many"`` (R3-2)."""
    from local_operator.session.cleanup import format_cleanup_notice

    text = format_cleanup_notice(payload)
    assert text.startswith("session cleanup removed ")
    assert "lop sessions cleanup --dry-run" in text


def test_the_notice_labels_origins() -> None:
    """U15: the notice says whose sessions went, not only how many."""
    from local_operator.session.cleanup import format_cleanup_notice

    text = format_cleanup_notice(
        {"removed": 5, "policies": {"max_sessions": 5}, "origins": {"user": 2, "subagent": 3}}
    )
    assert "5 by max_sessions" in text and "3 subagent" in text and "2 user" in text


def test_the_record_is_written_atomically_and_a_stuck_one_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R3-4: tmp + replace, so a viewer never reads a torn record; a record
    that cannot be flipped announces again and says so at debug."""
    from local_operator.session.cleanup import (
        LAST_CLEANUP_NAME,
        Candidate,
        CleanupResult,
        take_unannounced_cleanup,
        write_last_cleanup,
    )

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    result = CleanupResult(scanned=3)
    result.removed = [Candidate("a", "remove_empty", "no transcript")]
    write_last_cleanup(sessions, result, actor="startup")
    assert not (sessions / (LAST_CLEANUP_NAME + ".tmp")).exists()
    payload = json.loads((sessions / LAST_CLEANUP_NAME).read_text())
    assert payload["origins"] == {"user": 1} and payload["acknowledged"] is False
    sessions.chmod(0o500)
    if os.access(sessions, os.W_OK):
        sessions.chmod(0o700)
        pytest.skip("cannot make the store read-only here")
    try:
        with caplog.at_level(logging.DEBUG, logger="local_operator.session.cleanup"):
            first = take_unannounced_cleanup(sessions)
            second = take_unannounced_cleanup(sessions)
    finally:
        sessions.chmod(0o700)
    assert first is not None and second is not None, "still announced: the facts are true"
    assert "will repeat" in caplog.text


# ---------------------------------------------------------------------------
# The picker, ``@latest`` and ``resume_dir`` agree on membership (R3-5)
# ---------------------------------------------------------------------------


def test_an_inbox_only_session_is_listed_latest_and_resumable(tmp_path: Path) -> None:
    """A peer's message spooled into an idle launch: the picker ranks it
    first (a spooled message is a reason to come back), ``@latest`` picks
    the same row, and ``resume_dir`` opens it rather than refusing."""
    from local_operator.resume import RESUME_LATEST, recent_sessions, resume_dir
    from local_operator.session.runtime.inbox import INBOX_NAME

    mark_store(tmp_path / "sessions")
    _session(tmp_path, "conv", age_days=3)
    mailonly = tmp_path / "sessions" / "mailonly"
    mailonly.mkdir()
    (mailonly / INBOX_NAME).write_text('{"from": "peer"}\n', encoding="utf-8")
    stamp = NOW - DAY
    os.utime(mailonly / INBOX_NAME, (stamp, stamp))

    picker = [name for name, _ in recent_sessions(tmp_path)]
    assert picker == ["mailonly", "conv"]
    assert resume_dir(tmp_path, RESUME_LATEST).name == "mailonly"
    assert resume_dir(tmp_path, "mailonly") == mailonly
    # And it is kept by the policy: a picker row first, and behind that a
    # hard-guarded one (never a remove_empty candidate while mail is spooled).
    result = run_cleanup(tmp_path, CleanupPolicy(enabled=True, remove_empty=True), now=NOW)
    assert result.removed == []
    assert ("mailonly", "one of the 10 most recent") in result.protected
    assert cleanup._guard(mailonly, tmp_path, NOW) == "has unread spooled mail"


def test_latest_breaks_equal_stamps_the_way_the_picker_does(tmp_path: Path) -> None:
    """Same clock, same tie-break: with equal stamps ``@latest`` is the
    picker's first row (ascending id), not ``max``'s largest id."""
    from local_operator.resume import RESUME_LATEST, recent_sessions, resume_dir

    mark_store(tmp_path / "sessions")
    for name in ("b", "c", "a"):
        _session(tmp_path, name, age_days=3)
    assert [n for n, _ in recent_sessions(tmp_path)] == ["a", "b", "c"]
    assert resume_dir(tmp_path, RESUME_LATEST).name == "a"


# ---------------------------------------------------------------------------
# The removing runtime's viewer announces (UX round 3, U14)
# ---------------------------------------------------------------------------


def _record(sessions: Path, pid: int) -> None:
    from local_operator.session.cleanup import LAST_CLEANUP_NAME

    sessions.mkdir(exist_ok=True)
    (sessions / LAST_CLEANUP_NAME).write_text(
        json.dumps({"removed": 2, "pid": pid, "acknowledged": False, "policies": {}})
    )


def test_the_viewer_of_the_removing_runtime_takes_the_record(tmp_path: Path) -> None:
    """While the removing runtime lives, only a viewer dialed into THAT pid
    announces; every other viewer (another terminal's runtime, a cold
    viewer) defers and leaves the record unacknowledged."""
    import subprocess
    import sys

    from local_operator.session.cleanup import take_unannounced_cleanup

    sessions = tmp_path / "sessions"
    writer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _record(sessions, writer.pid)
        assert take_unannounced_cleanup(sessions, runtime_pid=None) is None, "cold viewer"
        assert take_unannounced_cleanup(sessions, runtime_pid=writer.pid + 100_000) is None
        taken = take_unannounced_cleanup(sessions, runtime_pid=writer.pid)
        assert taken is not None and taken["removed"] == 2
        assert take_unannounced_cleanup(sessions, runtime_pid=writer.pid) is None, "once"
    finally:
        writer.kill()
        writer.wait()


def test_any_viewer_takes_the_record_once_the_writer_is_gone(tmp_path: Path) -> None:
    """A runtime that exited before its viewer looked (headless, crashed): the
    fact must still reach a screen, so the first reader announces."""
    import subprocess
    import sys

    from local_operator.session.cleanup import take_unannounced_cleanup

    sessions = tmp_path / "sessions"
    writer = subprocess.Popen([sys.executable, "-c", "pass"])
    writer.wait()
    _record(sessions, writer.pid)
    taken = take_unannounced_cleanup(sessions, runtime_pid=None)
    assert taken is not None and taken["removed"] == 2
