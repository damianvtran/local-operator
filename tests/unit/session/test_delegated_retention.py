"""The DELEGATED class of ``session.cleanup``: what may go, and every reason it must not.

The asymmetry is the one ``test_cleanup.py`` opens with: a kept directory costs
bytes, a removed one costs somebody a result. So most of these are NEGATIVE
controls, one per protection rule, each built so that deleting the rule from
``session/delegated_retention.py`` turns it red (the two that guard the most —
the parent-alive rule and the scratchpad git guard — were mutated by hand and
seen failing; see the PR body).

Every store here is a ``tmp_path`` config root with a store marker; nothing
touches the operator's real one.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import delegated_retention as dr
from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    CleanupPolicy,
    clamp_delegated_hours,
    mark_store,
    policy_from_config,
    remove_scratchpad_entry,
    run_cleanup,
)

HOUR = 3600.0
NOW = time.time()
ON = CleanupPolicy(delegated_enabled=True, delegated_max_age_hours=48)


def _provider(policy: CleanupPolicy = ON):
    return lambda: policy


def _mk(
    root: Path,
    name: str,
    origin: str | None = "subagent",
    age_h: float | None = 100.0,
    *,
    roster: list[str] | None = None,
    **details: object,
) -> Path:
    directory = root / "sessions" / name
    directory.mkdir(parents=True, exist_ok=True)
    if origin is not None:
        (directory / "origin.json").write_text(json.dumps({"origin": origin, **details}))
    if age_h is not None:
        transcript = directory / "transcript.jsonl"
        transcript.write_text('{"type":"message"}\n')
        os.utime(transcript, (NOW - age_h * HOUR, NOW - age_h * HOUR))
    if roster is not None:
        records = [{"session_dir": str(root / "sessions" / child)} for child in roster]
        (directory / dr.ROSTER_NAME).write_text(json.dumps({"version": 1, "records": records}))
    return directory


@pytest.fixture
def root(tmp_path: Path) -> Path:
    mark_store(tmp_path / "sessions")
    return tmp_path


def _names(root: Path) -> set[str]:
    return {p.name for p in (root / "sessions").iterdir() if p.is_dir()}


def _pass(root: Path, **kwargs: Any) -> dr.DelegatedResult:
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("removal_pause_s", 0)
    return dr.run_delegated_pass(root, kwargs.pop("policy", _provider()), **kwargs)


def _kept(result: dr.DelegatedResult) -> dict[str, str]:
    return dict(result.protected)


# --------------------------------------------------------------------------
# The class split: one predicate, both directions
# --------------------------------------------------------------------------


def test_only_hidden_origins_are_candidates_and_user_origins_never_are(root: Path) -> None:
    for name, origin in (
        ("sub", "subagent"),
        ("shell", "agent-shell"),
        ("config", "agent-config"),
        ("future", "some-origin-minted-next-year"),
    ):
        _mk(root, name, origin)
    for name, origin in (("user", None), ("fork", "fork"), ("ws", "agent-workstream")):
        _mk(root, name, origin, age_h=24 * 300)
    result = _pass(root)
    assert {c.session for c in result.removed} == {"sub", "shell", "config", "future"}
    assert _names(root) == {"user", "fork", "ws"}


def test_an_agent_workstream_ten_times_past_the_cutoff_is_never_delegated(root: Path) -> None:
    """The sidebar lists it, so it is the parent class (USER_ORIGINS)."""
    _mk(root, "ws", "agent-workstream", age_h=48 * 10)
    result = _pass(root, dry_run=True)
    assert result.removed == [] and "ws" not in _kept(result)
    assert _pass(root).removed == [] and "ws" in _names(root)


def test_an_unreadable_origin_reads_as_the_users_own(root: Path) -> None:
    directory = _mk(root, "garbled", None)
    (directory / "origin.json").write_text("{not json")
    assert _pass(root).removed == []
    assert "garbled" in _names(root)


def test_the_parent_class_scan_no_longer_sees_delegated_sessions(root: Path) -> None:
    """max_total_bytes / max_inactive_days / remove_empty iterate parent dirs only."""
    for index in range(5):
        _mk(
            root,
            f"user{index}",
            None,
            age_h=24 * 40 + index,
        )
    big = _mk(root, "bigsub", "subagent", age_h=24 * 90)
    (big / "blob").write_bytes(b"x" * 500_000)
    empty_sub = _mk(root, "emptysub", "subagent", age_h=None)
    policy = CleanupPolicy(
        enabled=True,
        max_total_bytes=1,
        max_inactive_days=1,
        remove_empty=True,
        delegated_enabled=False,
    )
    result = run_cleanup(root, policy, now=NOW, dry_run=True)
    seen = {c.session for c in result.chosen} | {n for n, _ in result.protected}
    assert "bigsub" not in seen and "emptysub" not in seen
    assert big.exists() and empty_sub.exists()
    assert result.scanned == 5  # the five user sessions, nothing else


# --------------------------------------------------------------------------
# Age: the window, the 2h floor, and the clock of a never-active directory
# --------------------------------------------------------------------------


def test_a_session_inside_the_window_is_kept_and_one_outside_goes(root: Path) -> None:
    _mk(root, "fresh", age_h=47)
    _mk(root, "stale", age_h=49)
    result = _pass(root)
    assert [c.session for c in result.removed] == ["stale"]
    assert "fresh" in _names(root)


def test_a_one_hour_old_session_survives_even_the_minimum_window(root: Path) -> None:
    _mk(root, "young", age_h=1)
    assert _pass(root, policy=_provider(CleanupPolicy(delegated_max_age_hours=2))).removed == []
    assert "young" in _names(root)


def test_the_window_is_configurable_up_and_down(root: Path) -> None:
    _mk(root, "a", age_h=3)
    _mk(root, "b", age_h=100)
    assert [
        c.session
        for c in _pass(root, policy=_provider(CleanupPolicy(delegated_max_age_hours=720))).removed
    ] == []
    removed = _pass(root, policy=_provider(CleanupPolicy(delegated_max_age_hours=2))).removed
    assert {c.session for c in removed} == {"a", "b"}


def test_an_abandoned_directory_is_clocked_by_created_at_then_mtime(root: Path) -> None:
    old = _mk(root, "old", age_h=None)
    (old / "created_at.json").write_text(str(NOW - 200 * HOUR))
    young = _mk(root, "young", age_h=None)
    (young / "created_at.json").write_text(str(NOW - 1 * HOUR))
    bare = _mk(root, "bare", age_h=None)  # no created_at, brand-new mtime => inside the window
    removed = _pass(root).removed
    assert [c.session for c in removed] == ["old"]
    assert young.exists() and bare.exists()


def test_a_directory_with_no_clock_is_kept(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mk(root, "noclock", age_h=None)
    monkeypatch.setattr(dr, "_clock_without_activity", lambda _d: None)
    assert _pass(root).removed == []
    assert "noclock" in _names(root)


def test_a_sidecar_write_does_not_move_the_clock(root: Path) -> None:
    directory = _mk(root, "s", age_h=100)
    (directory / "title-scan.json").write_text("{}")  # a boot's sentinel
    (directory / "turn-journal.json").write_text("{}")
    assert [c.session for c in _pass(root).removed] == ["s"]


# --------------------------------------------------------------------------
# The hard guards
# --------------------------------------------------------------------------


def test_a_live_claim_keeps_the_session(root: Path) -> None:
    directory = _mk(root, "live")
    (directory / ".session.pid").write_text(str(os.getpid()))
    result = _pass(root)
    assert result.removed == [] and "live" in _names(root)
    assert _kept(result)["live"] == "claimed by a live process"


def test_a_dead_claim_does_not_keep_it(root: Path) -> None:
    directory = _mk(root, "dead")
    (directory / ".session.pid").write_text("999999999")
    assert [c.session for c in _pass(root).removed] == ["dead"]


def test_an_armed_wake_keeps_it_and_a_stopped_one_does_not(root: Path) -> None:
    _mk(root, "armed")
    _mk(root, "dormant")
    wakes = root / "wakes"
    wakes.mkdir()
    (wakes / "armed.json").write_text(json.dumps({"schedules": [{"id": "w1"}]}))
    (wakes / "dormant.json").write_text(json.dumps({"stopped_at": 1, "schedules": [{"id": "w1"}]}))
    result = _pass(root)
    assert _kept(result) == {"armed": "has an armed wake"}
    assert [c.session for c in result.removed] == ["dormant"]


def test_an_armed_monitor_keeps_it(root: Path) -> None:
    _mk(root, "watched")
    (root / "monitors").mkdir()
    (root / "monitors" / "watched.json").write_text(
        json.dumps({"monitors": [{"id": "m1", "every_ms": 60000}]})
    )
    result = _pass(root)
    assert result.removed == [] and _kept(result) == {"watched": "has an armed monitor"}


def test_a_corrupt_monitor_entry_keeps_it(root: Path) -> None:
    _mk(root, "watched")
    (root / "monitors").mkdir()
    (root / "monitors" / "watched.json").write_text("{nope")
    assert _pass(root).removed == []


def test_unread_spooled_mail_keeps_it(root: Path) -> None:
    directory = _mk(root, "mail")
    inbox = directory / "inbox.jsonl"
    inbox.write_text('{"x":1}\n')
    os.utime(inbox, (NOW - 100 * HOUR, NOW - 100 * HOUR))
    result = _pass(root)
    assert result.removed == [] and _kept(result) == {"mail": "has unread spooled mail"}


def test_the_current_session_is_kept(root: Path) -> None:
    directory = _mk(root, "me")
    result = _pass(root, live_dir=directory)
    assert result.removed == [] and _kept(result) == {"me": "the current session"}


def test_a_session_that_moves_after_the_scan_is_kept(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _mk(root, "wakes-up")
    real = dr._evaluate

    def touch_then_evaluate(cand, **kw):  # type: ignore[no-untyped-def]
        os.utime(directory / "transcript.jsonl", (NOW, NOW))
        return real(cand, **kw)

    monkeypatch.setattr(dr, "_evaluate", touch_then_evaluate)
    result = _pass(root)
    assert result.removed == [] and _kept(result) == {"wakes-up": "active since the scan"}


def test_a_guard_that_raises_keeps_the_session(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mk(root, "boom")
    monkeypatch.setattr(
        "local_operator.session.cleanup._has_spooled_mail",
        lambda _d: (_ for _ in ()).throw(RuntimeError("x")),
    )
    result = _pass(root)
    assert result.removed == [] and "boom" in _names(root)


def test_an_unmarked_store_removes_nothing(tmp_path: Path) -> None:
    _mk(tmp_path, "s")  # no store marker
    result = _pass(tmp_path)
    assert result.removed == [] and "s" in _names(tmp_path)


# --------------------------------------------------------------------------
# THE PARENT-ALIVE RULE
# --------------------------------------------------------------------------


def test_a_child_of_a_parent_active_inside_the_window_is_protected(root: Path) -> None:
    _mk(root, "parent", None, age_h=10, roster=["kid"])
    _mk(root, "kid")
    result = _pass(root)
    assert result.removed == [] and _kept(result) == {"kid": "parent session still active"}


def test_a_child_of_a_parent_idle_past_the_window_is_reapable(root: Path) -> None:
    _mk(root, "parent", None, age_h=24 * 5, roster=["kid"])
    _mk(root, "kid")
    assert [c.session for c in _pass(root).removed] == ["kid"]


def test_a_child_of_a_live_parent_is_protected_whatever_its_age(root: Path) -> None:
    parent = _mk(root, "parent", None, age_h=24 * 30, roster=["kid"])
    (parent / ".session.pid").write_text(str(os.getpid()))
    _mk(root, "kid")
    assert _pass(root).removed == []


def test_a_child_with_no_roster_edge_is_reapable(root: Path) -> None:
    _mk(root, "orphan")
    assert [c.session for c in _pass(root).removed] == ["orphan"]


def test_any_active_parent_is_enough(root: Path) -> None:
    _mk(root, "idle-parent", None, age_h=24 * 9, roster=["kid"])
    _mk(root, "active-parent", None, age_h=3, roster=["kid"])
    _mk(root, "kid")
    assert _pass(root).removed == []


def test_protection_is_transitive_through_a_protected_child(root: Path) -> None:
    _mk(root, "parent", None, age_h=5, roster=["child"])
    _mk(root, "child", age_h=100, roster=["grandchild"])
    _mk(root, "grandchild")
    result = _pass(root)
    assert result.removed == []
    assert set(_kept(result)) == {"child", "grandchild"}


def test_a_hidden_parent_inside_the_window_protects_its_children(root: Path) -> None:
    _mk(root, "orchestrator", "subagent", age_h=5, roster=["worker"])
    _mk(root, "worker")
    assert _pass(root).removed == []


def test_a_corrupt_roster_of_a_recent_parent_skips_the_whole_pass(root: Path) -> None:
    parent = _mk(root, "parent", None, age_h=5)
    (parent / dr.ROSTER_NAME).write_text("{truncated")
    _mk(root, "kid")
    result = _pass(root)
    assert result.removed == [] and result.skipped and result.skipped.startswith("skipped:")
    assert result.errors == 1 and "kid" in _names(root)


def test_a_corrupt_roster_of_an_idle_parent_is_not_consulted(root: Path) -> None:
    """Only live/recent parents' rosters are read (the 6 MB rosters are never opened)."""
    parent = _mk(root, "parent", None, age_h=24 * 9)
    (parent / dr.ROSTER_NAME).write_text("{truncated")
    _mk(root, "kid")
    assert [c.session for c in _pass(root).removed] == ["kid"]


def test_rosters_of_idle_parents_are_never_opened(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(20):
        _mk(root, f"idle{index}", None, age_h=24 * 9, roster=["kid"])
    _mk(root, "kid")
    opened: list[str] = []
    real = dr._read_json
    monkeypatch.setattr(dr, "_read_json", lambda p: (opened.append(str(p)), real(p))[1])
    dr._FILE_CACHE.clear()
    _pass(root)
    assert opened == []


def test_the_roster_cache_is_keyed_on_mtime_and_size(root: Path) -> None:
    parent = _mk(root, "parent", None, age_h=5, roster=["a"])
    _mk(root, "a")
    dr._FILE_CACHE.clear()
    assert dr.roster_children(parent / dr.ROSTER_NAME) == {"a"}
    (parent / dr.ROSTER_NAME).write_text(
        json.dumps({"version": 1, "records": [{"session_dir": "/x/a"}, {"session_dir": "/x/b"}]})
    )
    assert dr.roster_children(parent / dr.ROSTER_NAME) == {"a", "b"}


# --------------------------------------------------------------------------
# Open projects
# --------------------------------------------------------------------------


def _project(root: Path, name: str, status: str, sessions: list[str]) -> None:
    (root / "projects").mkdir(exist_ok=True)
    (root / "projects" / f"{name}.json").write_text(
        json.dumps({"id": name, "status": status, "sessions": sessions})
    )


def test_a_session_of_an_open_project_is_kept_and_a_closed_one_is_not(root: Path) -> None:
    _mk(root, "open")
    _mk(root, "closed")
    _mk(root, "archived")
    _project(root, "p1", "active", ["open"])
    _project(root, "p2", "done", ["closed"])
    _project(root, "p3", "archived", ["archived"])
    result = _pass(root)
    assert _kept(result) == {"open": "linked to an open project"}
    assert {c.session for c in result.removed} == {"closed", "archived"}


def test_a_project_with_no_status_counts_as_open(root: Path) -> None:
    _mk(root, "s")
    (root / "projects").mkdir()
    (root / "projects" / "p.json").write_text(json.dumps({"sessions": ["s"]}))
    assert _pass(root).removed == []


def test_children_of_a_project_linked_parent_are_not_additionally_pinned(root: Path) -> None:
    """Documented: only the parent-alive rule protects them."""
    _mk(root, "parent", None, age_h=24 * 9, roster=["kid"])
    _mk(root, "kid")
    _project(root, "p", "active", ["parent"])
    assert [c.session for c in _pass(root).removed] == ["kid"]


def test_an_unreadable_project_file_skips_the_pass(root: Path) -> None:
    _mk(root, "s")
    (root / "projects").mkdir()
    (root / "projects" / "p.json").write_text("{nope")
    result = _pass(root)
    assert result.removed == [] and result.skipped and "projects/p.json" in result.skipped


# --------------------------------------------------------------------------
# Scratchpad git
# --------------------------------------------------------------------------

_GIT = {
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _git(directory: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        capture_output=True,
        env={**os.environ, **_GIT},
    )


def _repo(
    root: Path, session: str, *, commit: bool, dirty: bool, pushed: bool, sub: str = "repo"
) -> Path:
    repo = root / "sessions" / session / "scratchpad" / sub
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "f").write_text("x")
    if commit:
        _git(repo, "add", "f")
        _git(repo, "commit", "-qm", "c")
    if pushed:
        remote = root / f"remote-{session}.git"
        subprocess.run(
            ["git", "init", "-q", "--bare", str(remote)], check=True, env={**os.environ, **_GIT}
        )
        _git(repo, "remote", "add", "origin", str(remote))
        _git(repo, "push", "-q", "origin", "HEAD")
    if not dirty and commit:
        pass
    elif dirty and commit:
        (repo / "g").write_text("y")
    return repo


def test_a_dirty_scratchpad_repo_keeps_the_session_and_warns(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _mk(root, "dirty")
    _repo(root, "dirty", commit=False, dirty=True, pushed=False)
    with caplog.at_level("WARNING"):
        result = _pass(root)
    assert result.removed == []
    assert _kept(result) == {"dirty": "scratchpad git repo has uncommitted/unpushed work"}
    assert any(
        "keeping dirty: scratchpad repo" in r.message and "uncommitted/unpushed work" in r.message
        for r in caplog.records
    )


def test_an_unpushed_commit_keeps_the_session(root: Path) -> None:
    _mk(root, "unpushed")
    _repo(root, "unpushed", commit=True, dirty=False, pushed=False)
    assert _pass(root).removed == []


def test_a_clean_fully_pushed_repo_does_not_keep_the_session(root: Path) -> None:
    _mk(root, "clean")
    _repo(root, "clean", commit=True, dirty=False, pushed=True)
    assert [c.session for c in _pass(root).removed] == ["clean"]


def test_dirty_after_a_push_keeps_the_session(root: Path) -> None:
    _mk(root, "later")
    _repo(root, "later", commit=True, dirty=True, pushed=True)
    assert _pass(root).removed == []


def test_a_nested_linked_worktree_git_file_counts(root: Path) -> None:
    _mk(root, "wt")
    nested = root / "sessions" / "wt" / "scratchpad" / "a" / "b" / "checkout"
    nested.mkdir(parents=True)
    (nested / ".git").write_text("gitdir: /nonexistent/place\n")
    assert _pass(root).removed == []  # git cannot inspect it => keep


def test_git_missing_or_timing_out_keeps_the_session(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mk(root, "nogit")
    _repo(root, "nogit", commit=True, dirty=False, pushed=True)
    monkeypatch.setattr(dr, "_git", lambda *a: (_ for _ in ()).throw(FileNotFoundError("git")))
    assert _pass(root).removed == []
    monkeypatch.setattr(
        dr, "_git", lambda *a: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 5))
    )
    assert _pass(root).removed == []


def test_a_repo_below_the_depth_limit_is_not_searched_but_a_shallow_one_is(root: Path) -> None:
    _mk(root, "deep")
    deep = root / "sessions" / "deep" / "scratchpad" / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (deep / ".git").mkdir()
    assert dr.scratchpad_git_hazard(root / "sessions" / "deep") is None  # depth 5 > 4
    shallow = root / "sessions" / "deep" / "scratchpad" / "a" / "b" / "c" / "d"
    (shallow / ".git").write_text("gitdir: /nonexistent\n")
    assert dr.scratchpad_git_hazard(root / "sessions" / "deep") is not None


def test_dependency_trees_are_skipped(root: Path) -> None:
    _mk(root, "deps")
    inner = root / "sessions" / "deps" / "scratchpad" / "node_modules" / "pkg"
    inner.mkdir(parents=True)
    (inner / ".git").write_text("gitdir: /nonexistent\n")
    assert dr.scratchpad_git_hazard(root / "sessions" / "deps") is None


def test_exceeding_the_entry_cap_keeps_the_session(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mk(root, "huge")
    pad = root / "sessions" / "huge" / "scratchpad"
    pad.mkdir()
    for index in range(30):
        (pad / f"f{index}").write_text("x")
    monkeypatch.setattr(dr, "SCRATCH_MAX_ENTRIES", 10)
    assert dr.scratchpad_git_hazard(root / "sessions" / "huge") is not None
    assert _pass(root).removed == []


def test_a_scratchpad_without_repos_costs_no_git_call(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mk(root, "plain")
    (root / "sessions" / "plain" / "scratchpad").mkdir()
    (root / "sessions" / "plain" / "scratchpad" / "notes.md").write_text("hi")
    monkeypatch.setattr(dr, "_git", lambda *a: pytest.fail("git must not run"))
    assert [c.session for c in _pass(root).removed] == ["plain"]


def test_git_runs_with_a_scrubbed_environment() -> None:
    env = dr._git_env()
    assert "XPC_FLAGS" not in env
    assert env["GIT_CONFIG_SYSTEM"] == "/dev/null" and env["GIT_TERMINAL_PROMPT"] == "0"


# --------------------------------------------------------------------------
# Draining: batches, budget, stop, switch-off, idempotence, logging
# --------------------------------------------------------------------------


def test_removals_are_logged_with_the_delegated_policy_and_an_actor(root: Path) -> None:
    _mk(root, "a")
    _mk(root, "b")
    _pass(root, actor="startup")
    rows = [
        json.loads(line) for line in (root / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
    ]
    assert {r["session"] for r in rows} == {"a", "b"}
    assert all(
        r["policy"] == "delegated_max_age"
        and r["reason"] == "idle over 48h"
        and r["actor"] == "startup"
        for r in rows
    )


def test_a_second_pass_removes_nothing(root: Path) -> None:
    for index in range(5):
        _mk(root, f"s{index}")
    assert len(_pass(root).removed) == 5
    again = _pass(root)
    assert again.removed == [] and again.remaining == 0


def test_a_budget_that_expires_during_the_scan_does_not_abandon_the_drain(root: Path) -> None:
    for index in range(3):
        _mk(root, f"s{index}")
    result = _pass(root, budget_s=0.0, batch_size=2)
    # Not skipped, and still one full batch of progress: a pass can be slowed by
    # its budget but never stalled into removing nothing.
    assert result.skipped is None and result.budget_exhausted
    assert len(result.removed) == 2 and result.remaining == 1


def test_a_budget_stops_the_pass_and_the_next_one_continues(root: Path) -> None:
    for index in range(12):
        _mk(root, f"s{index:02d}", age_h=100 + index)
    first = _pass(root, batch_size=5, budget_s=0.0)
    assert first.budget_exhausted and len(first.removed) == 5 and first.remaining == 7
    drained = [c.session for c in first.removed]
    passes = 1
    while _names(root):
        result = _pass(root, batch_size=5, budget_s=0.0)
        assert result.removed, "a pass made no progress"
        drained += [c.session for c in result.removed]
        passes += 1
    assert len(drained) == 12 and len(set(drained)) == 12 and passes == 3


def test_permanent_keeps_at_the_head_of_the_queue_do_not_stall_the_drain(root: Path) -> None:
    """Oldest-first order puts the sessions that are kept for good FIRST (a live
    claim, an armed wake). A batch counted in candidates examined was spent on them
    and the drain removed nothing, pass after pass; a batch is N REMOVALS."""
    for index in range(6):
        wake = _mk(root, f"keep{index}", age_h=500 + index)
        (root / "wakes").mkdir(exist_ok=True)
        (root / "wakes" / f"{wake.name}.json").write_text(json.dumps({"schedules": [{"id": "w"}]}))
    for index in range(12):
        _mk(root, f"go{index:02d}", age_h=100 + index)
    first = _pass(root, batch_size=5, budget_s=0.0)
    assert len(first.removed) == 5 and first.budget_exhausted
    assert {n for n, _ in first.protected} == {f"keep{i}" for i in range(6)}
    second = _pass(root, batch_size=5, budget_s=0.0)
    assert len(second.removed) == 5


def test_a_pass_whose_removals_are_all_refused_stops_at_the_budget(tmp_path: Path) -> None:
    """Review F2: a foreign/read-only store must not walk the whole candidate list.

    An unmarked store refuses every removal (``remove_session_dir`` fails
    closed), so no batch of removals ever completes — with the deadline checked
    only at batch boundaries the pass walked all 250 candidates with
    ``budget_exhausted`` false and ``remaining`` 0, stamping the store clean.
    A full batch of refused attempts is the failure-direction evidence that lets
    the deadline end the pass instead.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for index in range(250):  # NOT marked: every removal attempt is refused
        _mk(tmp_path, f"c{index:04d}")
    attempts = {"n": 0}
    real_remove = dr.remove_session_dir

    def counting_remove(*args: Any, **kwargs: Any) -> bool:
        attempts["n"] += 1
        return real_remove(*args, **kwargs)

    dr.remove_session_dir = counting_remove  # type: ignore[assignment]
    try:
        result = _pass(tmp_path, batch_size=10, budget_s=0.001)
    finally:
        dr.remove_session_dir = real_remove  # type: ignore[assignment]
    assert result.scanned == 250 and result.removed == []
    assert result.budget_exhausted and result.remaining > 0
    # The walk is bounded by the refusal window: one full batch of refused
    # attempts (10) plus at most the one in flight when the deadline is seen.
    assert attempts["n"] <= 11, f"walked {attempts['n']} candidates instead of stopping"


def test_a_marked_store_with_a_generous_budget_still_drains_normally(root: Path) -> None:
    """The F2 control: bounding the unproductive walk changes no real drain."""
    for index in range(250):
        _mk(root, f"c{index:04d}")
    result = _pass(root, batch_size=10, budget_s=30.0)
    assert result.budget_exhausted is False
    assert len(result.removed) == 250 and result.remaining == 0
    assert _names(root) == set()


def test_a_pass_stops_between_batches_on_the_stop_event(root: Path) -> None:
    for index in range(12):
        _mk(root, f"s{index:02d}")
    ticks = {"n": 0}

    def stop() -> bool:
        ticks["n"] += 1
        return len(_names(root)) <= 7  # stop once 5 are gone

    result = _pass(root, batch_size=5, should_stop=stop)
    assert len(result.removed) == 5 and result.remaining == 7
    assert len(_names(root)) == 7


def test_turning_the_switch_off_stops_the_drain_at_the_next_batch(root: Path) -> None:
    for index in range(12):
        _mk(root, f"s{index:02d}")
    state = {"on": True}

    def policy() -> CleanupPolicy:
        return CleanupPolicy(delegated_enabled=state["on"])

    calls = {"n": 0}
    real_remove = dr.remove_session_dir

    def flip_after_first_batch(*a: Any, **k: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 5:
            state["on"] = False
        return real_remove(*a, **k)

    dr.remove_session_dir = flip_after_first_batch  # restored below
    try:
        result = dr.run_delegated_pass(root, policy, batch_size=5, now=NOW, removal_pause_s=0)
    finally:
        dr.remove_session_dir = real_remove
    assert len(result.removed) == 5 and result.skipped == "disabled during the pass"
    assert len(_names(root)) == 7


def test_a_disabled_policy_removes_nothing_without_force(root: Path) -> None:
    _mk(root, "s")
    off = CleanupPolicy(delegated_enabled=False)
    result = _pass(root, policy=_provider(off))
    assert result.removed == [] and result.skipped == "disabled" and "s" in _names(root)


def test_a_dry_run_lists_everything_and_removes_nothing(root: Path) -> None:
    for index in range(7):
        _mk(root, f"s{index}")
    _mk(root, "fresh", age_h=1)
    result = _pass(root, dry_run=True, batch_size=2)
    assert len(result.removed) == 7 and len(_names(root)) == 8
    assert _kept(result)["fresh"].startswith("active within")
    assert not (root / "sessions" / CLEANUP_LOG_NAME).exists()


def test_a_dry_run_with_the_switch_off_still_lists_and_says_so(root: Path) -> None:
    _mk(root, "s")
    result = _pass(root, policy=_provider(CleanupPolicy(delegated_enabled=False)), dry_run=True)
    assert [c.session for c in result.removed] == ["s"] and result.skipped == "disabled"


def test_only_restricts_a_confirmed_removal_to_the_previewed_rows(root: Path) -> None:
    _mk(root, "shown")
    _mk(root, "appeared-after-the-preview")
    result = _pass(root, only=["shown"])
    assert [c.session for c in result.removed] == ["shown"]
    assert "appeared-after-the-preview" in _names(root)


def test_sizes_are_computed_for_removed_rows_only(root: Path) -> None:
    gone = _mk(root, "gone")
    (gone / "payload").write_bytes(b"x" * 1000)
    keep = _mk(root, "keep", age_h=1)
    (keep / "payload").write_bytes(b"x" * 1000)
    seen: list[str] = []
    real = dr.bounded_dir_bytes

    def spy(directory: Path, *args: Any) -> int:
        seen.append(Path(directory).name)
        return real(directory, *args)

    dr.bounded_dir_bytes = spy  # type: ignore[assignment]
    try:
        result = _pass(root)
    finally:
        dr.bounded_dir_bytes = real  # type: ignore[assignment]
    assert seen == ["gone"] and result.removed[0].size_bytes >= 1000


# --------------------------------------------------------------------------
# The one-time notice
# --------------------------------------------------------------------------


def test_the_first_removal_writes_the_notice_record_and_later_ones_stay_silent(root: Path) -> None:
    sessions = root / "sessions"
    assert dr.take_unannounced_delegated_notice(sessions) is None  # nothing yet
    _mk(root, "a")
    _pass(root)
    state = dr.read_state(sessions)
    assert state["removed_total"] == 1 and state["notice_acknowledged"] is False
    assert state["first_removal_at"] and state["freed_bytes_estimate"] is not None
    first = dr.take_unannounced_delegated_notice(sessions, defer_to_writer=False)
    assert first is not None and first["removed_total"] == 1
    # Steady state: more removals, NO second announcement.
    _mk(root, "b")
    _pass(root)
    assert dr.read_state(sessions)["removed_total"] == 2
    assert dr.take_unannounced_delegated_notice(sessions, defer_to_writer=False) is None


def test_the_removing_runtimes_own_viewer_announces_first(root: Path) -> None:
    _mk(root, "a")
    _pass(root)
    sessions = root / "sessions"
    state = dr.read_state(sessions)
    state["first_removal_pid"] = os.getppid()  # a live process that is not us
    (sessions / dr.STATE_NAME).write_text(json.dumps(state))
    assert dr.take_unannounced_delegated_notice(sessions, runtime_pid=1) is None
    assert dr.take_unannounced_delegated_notice(sessions, runtime_pid=os.getppid()) is not None


def test_the_peek_returns_the_notice_without_consuming_it(root: Path) -> None:
    """The desktop list route's read half: repeatable until someone ACKNOWLEDGES."""
    _mk(root, "a")
    _pass(root)
    sessions = root / "sessions"
    first = dr.peek_unannounced_delegated_notice(sessions)
    assert first is not None and first["removed_total"] == 1
    second = dr.peek_unannounced_delegated_notice(sessions)
    assert second == first  # nothing was flipped
    assert dr.read_state(sessions)["notice_acknowledged"] is False


def test_acknowledge_flips_the_flag_once_and_repeats_cleanly(root: Path) -> None:
    """The desktop ack route's write half: idempotent, and it silences take too."""
    _mk(root, "a")
    _pass(root)
    sessions = root / "sessions"
    assert dr.acknowledge_delegated_notice(sessions) is True
    assert dr.read_state(sessions)["notice_acknowledged"] is True
    assert dr.peek_unannounced_delegated_notice(sessions) is None
    assert dr.take_unannounced_delegated_notice(sessions, defer_to_writer=False) is None
    assert dr.acknowledge_delegated_notice(sessions) is True  # idempotent


def test_the_notice_reads_tolerate_no_record(root: Path) -> None:
    """No record: the peek answers nothing and the ack invents no file."""
    sessions = root / "sessions"
    assert dr.peek_unannounced_delegated_notice(sessions) is None
    assert dr.acknowledge_delegated_notice(sessions) is True
    assert not (sessions / dr.STATE_NAME).exists()


def test_the_parent_class_record_is_untouched_by_a_delegated_removal(root: Path) -> None:
    _mk(root, "a")
    _pass(root)
    assert not (root / "sessions" / "last-cleanup.json").exists()


def test_the_notice_text_and_the_so_far_clause() -> None:
    text = dr.format_delegated_notice({"removed_total": 17579, "max_age_hours": 48})
    assert "Cleaned up 17,579 delegated sessions (subagents and background runs)" in text
    assert "older than 48 hours" in text and "Your own conversations were not touched." in text
    assert "Settings > Delegated work" in text and ".cleanup-log.jsonl" in text
    assert "so far" not in text
    assert "so far" in dr.format_delegated_notice(
        {"removed_total": 3, "max_age_hours": 48, "drain_remaining": 40}
    )
    assert "1 delegated session " in dr.format_delegated_notice({"removed_total": 1})


@pytest.mark.parametrize(
    "payload", [None, [], {}, {"removed_total": "many"}, {"max_age_hours": "x"}]
)
def test_the_notice_formats_any_shape(payload: object) -> None:
    assert isinstance(dr.format_delegated_notice(payload), str)


def test_a_malformed_state_file_announces_nothing(root: Path) -> None:
    (root / "sessions" / dr.STATE_NAME).write_text("{nope")
    sessions = root / "sessions"
    assert dr.take_unannounced_delegated_notice(sessions) is None
    assert dr.peek_unannounced_delegated_notice(sessions) is None
    assert dr.acknowledge_delegated_notice(sessions) is True  # nothing to flip, no error


# --------------------------------------------------------------------------
# The sweep loop: hourly, drain-continues, off-stops, lock
# --------------------------------------------------------------------------


class _Clock:
    """Records the waits the loop asks for and stops it after ``limit`` of them."""

    def __init__(self, limit: int) -> None:
        self.waits: list[float] = []
        self.limit = limit

    def wait(self, seconds: float) -> bool:
        self.waits.append(seconds)
        return len(self.waits) >= self.limit


def _sweeps(root: Path, policy, clock: _Clock, **kw: Any) -> str:  # type: ignore[no-untyped-def]
    return dr.run_sweeps(
        root,
        policy,
        live_dir=None,
        should_stop=lambda: len(clock.waits) >= clock.limit,
        wait=clock.wait,
        **kw,
    )


def test_a_clean_store_is_re_swept_hourly(root: Path) -> None:
    clock = _Clock(1)
    _sweeps(root, _provider(), clock)
    assert clock.waits == [dr.STEADY_SWEEP_S]


def test_an_unfinished_drain_comes_back_after_a_short_pause(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(12):
        _mk(root, f"s{index:02d}")
    monkeypatch.setattr(dr, "BATCH_SIZE", 5)
    monkeypatch.setattr(dr, "PASS_BUDGET_S", 0.0)
    monkeypatch.setattr(dr, "REMOVAL_PAUSE_S", 0)
    clock = _Clock(1)
    _sweeps(root, _provider(), clock)
    assert clock.waits == [dr.DRAIN_RESUME_S]


def test_the_loop_ends_when_the_switch_is_off(root: Path) -> None:
    _mk(root, "s")
    clock = _Clock(5)
    why = _sweeps(root, _provider(CleanupPolicy(delegated_enabled=False)), clock)
    assert why == "disabled" and clock.waits == [] and "s" in _names(root)


def test_a_fresh_sweep_stamp_skips_the_launch_scan(root: Path) -> None:
    _mk(root, "s")
    dr._stamp_sweep(root / "sessions", 0)
    clock = _Clock(1)
    _sweeps(root, _provider(), clock)
    assert "s" in _names(root)  # stamp fresh and backlog-free: no pass ran


def test_a_stamp_with_a_backlog_does_not_skip(root: Path) -> None:
    _mk(root, "s")
    dr._stamp_sweep(root / "sessions", 5)
    _sweeps(root, _provider(), _Clock(1))
    assert "s" not in _names(root)


def test_a_held_lock_defers_the_pass(root: Path) -> None:
    _mk(root, "s")
    held = dr._acquire_sweep_lock(root)
    assert held is not None
    try:
        clock = _Clock(1)
        _sweeps(root, _provider(), clock)
        assert clock.waits == [dr.LOCK_BUSY_RETRY_S] and "s" in _names(root)
    finally:
        dr._release_sweep_lock(held)


def test_a_failing_pass_does_not_kill_the_loop(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dr, "run_delegated_pass", lambda *a, **k: 1 / 0)
    clock = _Clock(1)
    _sweeps(root, _provider(), clock)
    assert clock.waits == [dr.STEADY_SWEEP_S]


# --------------------------------------------------------------------------
# Policy reading
# --------------------------------------------------------------------------


class _Cfg:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get_nested_value(self, path: tuple[str, ...], default: Any = None) -> Any:
        node: Any = self.values
        for part in path:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _hours(value: Any) -> int:
    return policy_from_config(
        _Cfg({"session": {"cleanup": {"delegated": {"max_age_hours": value}}}})
    ).delegated_max_age_hours


@pytest.mark.parametrize(
    "raw,expected",
    [
        (48, 48),
        (2, 2),
        (720, 720),
        (1, 2),
        (0, 48),
        (-5, 48),
        (721, 720),
        (10**9, 720),
        ("96", 96),
        ("abc", 48),
        (True, 48),
        (False, 48),
        (None, 48),
        (2.0, 2),
        (2.5, 48),
        ([], 48),
    ],
)
def test_the_reader_clamps_and_never_returns_less_than_two_hours(
    raw: object, expected: int
) -> None:
    assert _hours(raw) == expected
    assert clamp_delegated_hours(raw) >= 2


def test_the_default_policy_is_delegated_on_parent_off() -> None:
    policy = policy_from_config(_Cfg({}))
    assert policy.delegated_enabled is True and policy.delegated_max_age_hours == 48
    assert policy.enabled is False and not policy.has_any_limit


@pytest.mark.parametrize(
    "raw,expected",
    [
        (False, False),
        ("false", False),
        ("off", False),
        ("banana", False),
        (True, True),
        ("yes", True),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_the_delegated_switch_fails_closed_on_garbage(raw: object, expected: bool) -> None:
    cfg = _Cfg({"session": {"cleanup": {"delegated": {"enabled": raw}}}})
    assert policy_from_config(cfg).delegated_enabled is expected


def test_a_manager_that_raises_disables_the_delegated_class() -> None:
    class Broken:
        def get_nested_value(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("x")

    assert policy_from_config(Broken()).delegated_enabled is False


def test_an_out_of_range_hand_edit_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    from local_operator.session import cleanup

    cleanup._WARNED_HOURS.clear()
    with caplog.at_level("WARNING"):
        for _ in range(3):
            _hours(1)
    assert sum("outside 2..720" in r.message for r in caplog.records) == 1


def test_the_live_provider_follows_the_watcher_snapshot_then_the_manager(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _Cfg({"session": {"cleanup": {"delegated": {"max_age_hours": 96}}}})
    provider = dr.live_policy_provider(manager, root)
    assert provider().delegated_max_age_hours == 96

    class Watcher:
        values = {"session": {"cleanup": {"delegated": {"enabled": False}}}}

    monkeypatch.setattr("local_operator.config_watch.existing_watcher", lambda _d: Watcher())
    assert provider().delegated_enabled is False


# --------------------------------------------------------------------------
# Wire form
# --------------------------------------------------------------------------


def test_the_desktop_wire_notice_carries_the_message_and_the_numbers() -> None:
    wire = dr.notice_wire({"removed_total": 12, "max_age_hours": 48, "drain_remaining": 3})
    assert (
        wire["removed"] == 12 and wire["in_progress"] is True and "Cleaned up 12" in wire["message"]
    )
    assert dr.notice_wire({"removed_total": 12})["in_progress"] is False


# --------------------------------------------------------------------------
# THE CONTENT LAYER (the carve-out): what a KEPT record's pad still releases
# --------------------------------------------------------------------------
#
# These fixtures build REAL repositories with the real toolchain: the rules
# are about what git says (ancestry, ref reachability), and a mocked git
# would only restate the mock. Everything happens under ``tmp_path`` stores
# with a fake HOME; nothing touches an operator's directories.

_GIT_TEST_ENV = {key: value for key, value in os.environ.items() if key != "XPC_FLAGS"}
_GIT_TEST_ENV.update(_GIT)


def _git_stdout(*args: str) -> str:
    """Run one git command outside any repository (``bundle list-heads``)."""
    done = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        env=_GIT_TEST_ENV,
        timeout=60,
    )
    return done.stdout


def _kept_pad(root: Path, name: str) -> Path:
    """A delegated session whose RECORD the pass keeps, with a scratchpad/.

    The keep comes from an active parent (a roster edge) — the carve-out's
    case: the record stays, the content rules still run.
    """
    _mk(root, f"parent-{name}", None, age_h=10, roster=[name])
    directory = _mk(root, name)
    pad = directory / "scratchpad"
    pad.mkdir()
    return pad


def _stale_file(path: Path, *, size_mb: int = 11, age_h: float = 24) -> Path:
    """A sparse file of roughly ``size_mb`` MiB, last modified ``age_h`` ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.seek(size_mb * 1024 * 1024 - 1)
        handle.write(b"\0")
    stamp = NOW - age_h * HOUR
    os.utime(path, (stamp, stamp))
    return path


def _merged_tree(pad: Path, name: str, root: Path) -> Path:
    """A clone that is clean, pushed, and whose HEAD is on origin/main."""
    origin = root / "origins" / f"{name}.git"
    origin.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "--bare", "-b", "main", os.fspath(origin))
    tree = pad / name
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "remote", "add", "origin", os.fspath(origin))
    (tree / "a.txt").write_text("a\n")
    _git(tree, "add", "a.txt")
    _git(tree, "commit", "-qm", "one")
    _git(tree, "push", "-q", "origin", "main")
    _git(tree, "fetch", "-q", "origin")
    _git(tree, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    return tree


def _worktree_owner(root: Path, name: str) -> Path:
    """A repository with a bare origin and a pushed ``main`` — a worktree source.

    The owner lives OUTSIDE any pad (a worktree's objects are its shared
    repository's, so the fixtures need one that is not itself a candidate),
    and the linked worktree it hands out is created with ``git worktree add``
    so the real ``.git`` FILE and the real registration exist.
    """
    origin = root / "worktrees" / f"{name}.git"
    origin.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "--bare", "-b", "main", os.fspath(origin))
    owner = root / "worktrees" / name
    _git(root, "clone", "-q", os.fspath(origin), os.fspath(owner))
    (owner / "a.txt").write_text("a\n")
    _git(owner, "add", "a.txt")
    _git(owner, "commit", "-qm", "one")
    _git(owner, "push", "-q", "origin", "main")
    _git(owner, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    return owner


def _linked_worktree(owner: Path, target: Path) -> Path:
    """``git worktree add --detach`` ``target`` at the owner's ``main``."""
    _git(owner, "worktree", "add", "--detach", os.fspath(target), "main")
    return target


def test_a_merged_clean_clone_past_the_window_is_reclaimed_whole(root: Path) -> None:
    pad = _kept_pad(root, "reapme")
    tree = _merged_tree(pad, "clone", root)
    result = _pass(root)
    assert (root / "sessions" / "reapme").exists(), "only content goes; the record stays"
    assert not tree.exists()
    assert [(row.session, row.reason) for row in result.content] == [
        ("reapme", "merged-clean clone past the window")
    ]
    row = result.content[0]
    assert row.classes == ("merged-clean clone",) and row.entries >= 1 and row.bytes > 0
    assert row.rescue_bundle == "", "every local ref is on the remote: no bundle needed"
    assert not (pad / "reap-rescue-clone.bundle").exists()
    assert list(pad.iterdir()) == [], "the pad root itself stays, its content goes"


def test_a_pushed_but_unmerged_clone_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "unmerged")
    tree = _merged_tree(pad, "clone", root)
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-qm", "two")
    _git(tree, "push", "-q", "origin", "HEAD:feature")
    result = _pass(root)
    assert tree.exists() and (tree / "b.txt").exists()
    assert result.content == []
    assert result.content_kept == [("unmerged", "clone: HEAD is not merged into the remote trunk")]


def test_a_dirty_clone_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "dirtyone")
    tree = _merged_tree(pad, "clone", root)
    (tree / "a.txt").write_text("changed\n")
    result = _pass(root)
    assert tree.exists()
    assert result.content_kept == [("dirtyone", "clone: uncommitted changes")]


def test_a_clone_with_no_remote_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "noremote")
    tree = pad / "clone"
    tree.mkdir()
    _git(tree, "init", "-q", "-b", "main")
    (tree / "a.txt").write_text("a\n")
    _git(tree, "add", "a.txt")
    _git(tree, "commit", "-qm", "one")
    result = _pass(root)
    assert tree.exists()
    assert result.content_kept == [
        (
            "noremote",
            "clone: no remote trunk (origin/HEAD, origin/main or origin/master) to compare",
        )
    ]


def test_a_broken_gitdir_link_keeps_the_directory_whole(root: Path) -> None:
    pad = _kept_pad(root, "broken")
    tree = pad / "clone"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /nonexistent/elsewhere.git\n")
    (tree / "notes.txt").write_text("evidence\n")
    result = _pass(root)
    assert tree.exists() and (tree / "notes.txt").exists()
    assert result.content_kept == [
        ("broken", "clone: the gitdir link does not name a linked worktree")
    ]


def test_a_stale_big_log_is_reclaimed_and_small_or_fresh_files_stay(root: Path) -> None:
    pad = _kept_pad(root, "logs")
    big = _stale_file(pad / "desktop-suite.log")
    big_size = big.stat().st_size
    small = _stale_file(pad / "fd-run.log", size_mb=1)
    fresh = _stale_file(pad / "today.log", age_h=1)
    (pad / "notes.md").write_text("what happened\n")
    (pad / "drive.py").write_text("print('x')\n")
    evidence = pad / "evidence"
    evidence.mkdir()
    (evidence / "frame.png").write_bytes(b"\x89PNG")
    result = _pass(root)
    assert not big.exists()
    assert small.exists() and fresh.exists()
    assert (pad / "notes.md").exists() and (pad / "drive.py").exists()
    assert (evidence / "frame.png").exists()
    assert [(row.session, row.bytes, row.entries) for row in result.content] == [
        ("logs", big_size, 1)
    ]


def test_a_loose_node_modules_is_reclaimed_whole(root: Path) -> None:
    pad = _kept_pad(root, "mods")
    (pad / "node_modules").mkdir()
    (pad / "node_modules" / "pkg.js").write_text("x\n")
    result = _pass(root)
    assert not (pad / "node_modules").exists()
    assert [(row.session, row.reason, row.classes) for row in result.content] == [
        ("mods", "build shapes past the window", ("build shapes",))
    ]


def test_content_inside_a_kept_tree_is_never_touched(root: Path) -> None:
    pad = _kept_pad(root, "held")
    tree = _merged_tree(pad, "clone", root)
    (tree / ".gitignore").write_text("node_modules/\n")
    _git(tree, "add", ".gitignore")
    _git(tree, "commit", "-qm", "ignore")
    _git(tree, "push", "-q", "origin", "HEAD:feature")
    (tree / "node_modules").mkdir()
    (tree / "node_modules" / "pkg.js").write_text("x\n")
    result = _pass(root)
    assert result.content == []
    assert (tree / "node_modules" / "pkg.js").exists()
    assert result.content_kept == [("held", "clone: HEAD is not merged into the remote trunk")]


def test_a_dot_git_entry_is_never_taken_by_shape(root: Path) -> None:
    pad = _kept_pad(root, "gitroot")
    (pad / ".git").mkdir()
    (pad / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (pad / "notes.md").write_text("kept\n")
    result = _pass(root)
    assert (pad / ".git" / "HEAD").exists() and (pad / "notes.md").exists()
    assert result.content == []
    assert result.content_kept == [("gitroot", "the scratchpad root is itself a git tree")]


def test_a_build_named_directory_with_a_gitdir_is_judged_as_a_tree(root: Path) -> None:
    pad = _kept_pad(root, "treeshape")
    build = pad / "build"
    build.mkdir()
    (build / ".git").mkdir()
    (build / "junk.txt").write_text("x\n")
    loose = pad / "out"
    loose.mkdir()
    (loose / "payload.bin").write_text("y\n")
    result = _pass(root)
    assert build.exists() and (build / ".git").exists() and (build / "junk.txt").exists()
    assert not loose.exists()
    assert [row.reason for row in result.content] == ["build shapes past the window"]
    assert (
        "treeshape",
        "build: HEAD does not resolve (or the gitdir link is broken)",
    ) in result.content_kept


def test_a_symlink_entry_is_skipped_not_followed(root: Path) -> None:
    pad = _kept_pad(root, "linked")
    outside = root / "outside"
    outside.mkdir()
    _stale_file(outside / "big.log")
    os.symlink(os.fspath(outside), os.fspath(pad / "node_modules"))
    os.symlink(os.fspath(outside / "big.log"), os.fspath(pad / "copy.log"))
    result = _pass(root)
    assert result.content == []
    assert (outside / "big.log").exists()
    assert (pad / "node_modules").is_symlink() and (pad / "copy.log").is_symlink()


def test_a_read_only_tree_is_reclaimed_through_the_widen_path(root: Path) -> None:
    pad = _kept_pad(root, "husk")
    tree = pad / "node_modules"
    tree.mkdir()
    (tree / "pkg.js").write_text("x\n")
    os.chmod(tree / "pkg.js", 0o444)
    os.chmod(tree, 0o555)
    result = _pass(root)
    assert not tree.exists()
    assert len(result.content) == 1 and result.errors == 0


def test_an_unreadable_tree_is_reclaimed_through_the_pre_widen_walk(root: Path) -> None:
    pad = _kept_pad(root, "husk0")
    tree = pad / "node_modules"
    tree.mkdir()
    (tree / "pkg.js").write_text("x\n")
    os.chmod(tree, 0o000)
    result = _pass(root)
    assert not tree.exists()
    assert len(result.content) == 1 and result.errors == 0


def test_a_failed_file_removal_keeps_the_file_and_counts_an_error(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pad = _kept_pad(root, "stuck")
    holder = pad / "holder"
    holder.mkdir()
    big = _stale_file(holder / "big.log")
    os.chmod(holder, 0o555)  # readable, not writable: the unlink fails
    with caplog.at_level(logging.WARNING):
        result = _pass(root)
    assert big.exists()
    assert result.content == [] and result.errors == 1
    assert ("stuck", "scratchpad/holder/big.log: cannot be removed") in result.content_kept
    assert any("cannot remove" in message for message in caplog.messages)


def test_the_content_drain_batches_and_the_second_pass_is_idempotent(root: Path) -> None:
    for name in ("c1", "c2", "c3"):
        _stale_file(_kept_pad(root, name) / "big.log")
    first = _pass(root, batch_size=1, budget_s=0.0)
    assert [row.session for row in first.content] == ["c1"]
    assert first.content_remaining == 2 and first.content_budget_exhausted
    second = _pass(root, batch_size=1, budget_s=60.0)
    assert [row.session for row in second.content] == ["c2", "c3"]
    assert second.content_remaining == 0 and not second.content_budget_exhausted
    third = _pass(root)
    assert third.content == [] and third.content_remaining == 0


def test_a_content_reclaim_writes_one_log_row_per_pad(root: Path) -> None:
    pad = _kept_pad(root, "rowy")
    big = _stale_file(pad / "desktop-suite.log")
    size = big.stat().st_size
    _pass(root)
    rows = [
        json.loads(line) for line in (root / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["session"] == "rowy"
    assert row["policy"] == "delegated_scratchpad"
    assert row["reason"] == "stale output past the window"
    assert row["bytes"] == size and row["entries"] == 1
    assert row["title"] == "" and row["actor"] == "startup"
    assert row["classes"] == ["stale output"]


def test_content_totals_land_in_the_state_file_without_touching_the_notice_keys(
    root: Path,
) -> None:
    state = root / "sessions" / dr.STATE_NAME
    state.write_text(
        json.dumps(
            {
                "removed_total": 5,
                "first_removal_at": "2026-10-01T00:00:00+0000",
                "notice_acknowledged": False,
            }
        )
    )
    _stale_file(_kept_pad(root, "stated") / "big.log")
    _pass(root)
    data = json.loads(state.read_text())
    assert data["removed_total"] == 5, "content reclaims remove no sessions"
    assert data["first_removal_at"] == "2026-10-01T00:00:00+0000"
    assert data["content_reclaimed_total"] == 1
    assert data["content_freed_bytes_estimate"] >= 11 * 1024 * 1024
    assert "content_last_removal_at" in data


def test_a_dry_run_lists_the_content_it_would_take_and_writes_nothing(root: Path) -> None:
    pad = _kept_pad(root, "drypad")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "stray")
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-qm", "two")
    _git(tree, "checkout", "-q", "main")
    _stale_file(pad / "desktop-suite.log")
    result = _pass(root, dry_run=True)
    assert [row.session for row in result.content] == ["drypad"]
    row = result.content[0]
    assert row.classes == ("merged-clean clone", "stale output")
    assert row.rescue_bundle.endswith("reap-rescue-clone.bundle")
    assert tree.exists() and (pad / "desktop-suite.log").exists()
    assert not (pad / "reap-rescue-clone.bundle").exists(), "a dry run writes no bundle"
    assert not (root / "sessions" / CLEANUP_LOG_NAME).exists()
    assert not (root / "sessions" / dr.STATE_NAME).exists()


def test_unique_sidebar_refs_are_rescued_into_a_bundle_before_the_tree_goes(
    root: Path,
) -> None:
    pad = _kept_pad(root, "strayrefs")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "stray")
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-qm", "two")
    (tree / "a.txt").write_text("held aside\n")
    _git(tree, "stash", "push", "-q", "-m", "held")
    _git(tree, "checkout", "-q", "main")
    result = _pass(root)
    assert not tree.exists()
    row = result.content[0]
    assert row.reason == "merged-clean clone past the window"
    bundle = Path(row.rescue_bundle)
    assert bundle == pad / "reap-rescue-clone.bundle"
    assert bundle.exists()
    assert row.rescue_bundle_bytes == bundle.stat().st_size and row.rescue_bundle_bytes > 0
    heads = _git_stdout("bundle", "list-heads", os.fspath(bundle))
    assert "refs/heads/stray" in heads and "refs/stash" in heads
    assert "refs/heads/main" not in heads, "a pushed ref needs no rescue"
    log_row = json.loads((root / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()[-1])
    assert log_row["rescue_bundle"] == os.fspath(bundle)


def test_a_tree_over_the_uniqueness_cap_is_kept(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _kept_pad(root, "capped")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "stray")
    for name in ("b", "c"):
        (tree / f"{name}.txt").write_text(f"{name}\n")
        _git(tree, "add", f"{name}.txt")
        _git(tree, "commit", "-qm", name)
    _git(tree, "checkout", "-q", "main")
    monkeypatch.setattr(dr, "CONTENT_MAX_UNIQUE_COMMITS", 1)
    result = _pass(root)
    assert tree.exists() and result.content == []
    assert result.content_kept == [
        ("capped", "clone: more unique commits than the rescue check will list")
    ]


def test_a_bundle_failure_keeps_the_tree_and_counts_an_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _kept_pad(root, "rescuefail")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "stray")
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-qm", "two")
    _git(tree, "checkout", "-q", "main")

    def _fail(repo: Path, bundle_path: Path, refnames: list[str]) -> str:
        return "git bundle create failed for clone"

    monkeypatch.setattr(dr, "_write_rescue_bundle", _fail)
    result = _pass(root)
    assert tree.exists()
    assert result.content == [] and result.errors == 1
    assert result.content_kept == [("rescuefail", "clone: git bundle create failed for clone")]


def test_a_stale_bundle_lock_does_not_block_the_next_pass(root: Path) -> None:
    pad = _kept_pad(root, "locky")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "stray")
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-qm", "two")
    _git(tree, "checkout", "-q", "main")
    # A killed earlier attempt leaves git's lock file behind; without the
    # self-heal, "File exists" blocks every retry (measured on the acceptance
    # copy: a 5 s bound firing mid-create did exactly that).
    (pad / "reap-rescue-clone.bundle.lock").write_text("")
    result = _pass(root)
    assert not tree.exists()
    bundle = pad / "reap-rescue-clone.bundle"
    assert bundle.exists() and not (pad / "reap-rescue-clone.bundle.lock").exists()
    assert result.content and result.errors == 0


def test_a_merged_clean_linked_worktree_is_removed_not_bundled(root: Path) -> None:
    pad = _kept_pad(root, "wtpad")
    owner = _worktree_owner(root, "owner1")
    tree = _linked_worktree(owner, pad / "lo-before")
    result = _pass(root)
    assert not tree.exists(), "a merged-clean worktree is reclaimed through the owner"
    listed = _git_stdout("-C", os.fspath(owner), "worktree", "list", "--porcelain")
    assert "lo-before" not in listed, "the registration is pruned by the removal itself"
    assert [(row.session, row.reason) for row in result.content] == [
        ("wtpad", "merged-clean clone past the window")
    ]
    row = result.content[0]
    assert row.method == "worktree-remove" and row.rescue_bundle == ""
    assert not (
        pad / "reap-rescue-lo-before.bundle"
    ).exists(), "a worktree's objects are the shared store's; nothing is bundled"
    assert list(pad.iterdir()) == [], "no bundle, no leftovers"


def test_a_worktree_copy_whose_pointer_escapes_is_kept(root: Path) -> None:
    # A cp copy keeps the .git FILE, so it still names the ORIGINAL's gitdir
    # while its own path is not registered anywhere: the belt must refuse, and
    # the registered original must come out of the pass untouched.
    pad = _kept_pad(root, "copied")
    owner = _worktree_owner(root, "owner2")
    source = _linked_worktree(owner, root / "worktrees" / "outside")
    copy = pad / "lo-before"
    subprocess.run(["cp", "-Rc", os.fspath(source), os.fspath(copy)], check=True)
    result = _pass(root)
    assert copy.exists() and (copy / "a.txt").exists()
    assert source.exists(), "the registered original is untouched"
    listed = _git_stdout("-C", os.fspath(owner), "worktree", "list", "--porcelain")
    assert "outside" in listed, "the registration is not acted on through a copy"
    assert result.content == []
    assert result.content_kept == [
        ("copied", "lo-before: not registered in the shared repository's worktree list")
    ]


def test_a_dirty_linked_worktree_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "wtdirty")
    owner = _worktree_owner(root, "owner3")
    tree = _linked_worktree(owner, pad / "wt")
    (tree / "a.txt").write_text("changed\n")
    result = _pass(root)
    assert tree.exists() and (tree / "a.txt").read_text() == "changed\n"
    assert result.content_kept == [("wtdirty", "wt: uncommitted changes")]


def test_a_dry_run_does_not_remove_a_linked_worktree(root: Path) -> None:
    pad = _kept_pad(root, "wtdry")
    owner = _worktree_owner(root, "owner4")
    _linked_worktree(owner, pad / "wt")
    before = _git_stdout("-C", os.fspath(owner), "worktree", "list", "--porcelain")
    result = _pass(root, dry_run=True)
    assert (pad / "wt" / "a.txt").exists()
    after = _git_stdout("-C", os.fspath(owner), "worktree", "list", "--porcelain")
    assert after == before, "a dry run performs no worktree operation"
    assert [row.session for row in result.content] == ["wtdry"]
    assert result.content[0].method == "worktree-remove"
    assert result.content[0].rescue_bundle == ""
    assert not (pad / "reap-rescue-wt.bundle").exists()


def test_a_submodule_gitdir_link_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "submod")
    tree = pad / "wtsub"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /elsewhere/.git/modules/foo/worktrees/wtsub\n")
    (tree / "keep.txt").write_text("x\n")
    result = _pass(root)
    assert tree.exists() and (tree / "keep.txt").exists()
    assert result.content_kept == [("submod", "wtsub: a submodule pointer (never touched)")]


def test_a_dangling_worktree_pointer_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "dangling")
    tree = pad / "gone"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: /nonexistent/repo/.git/worktrees/gone\n")
    result = _pass(root)
    assert tree.exists()
    reason = dict(result.content_kept)["dangling"]
    assert "HEAD does not resolve" in reason


def test_a_malformed_git_file_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "malformed")
    tree = pad / "weird"
    tree.mkdir()
    (tree / ".git").write_text("not a pointer\n")
    result = _pass(root)
    assert tree.exists()
    assert result.content_kept == [("malformed", "weird: the .git link is malformed")]


def test_a_symlinked_git_entry_is_kept(root: Path) -> None:
    pad = _kept_pad(root, "symgit")
    owner = _worktree_owner(root, "owner5")
    tree = pad / "linked"
    tree.mkdir()
    (tree / ".git").symlink_to(owner / ".git")
    result = _pass(root)
    assert tree.exists()
    assert result.content_kept == [("symgit", "linked: the .git entry is a symlink")]


def test_evidence_paths_are_never_taken_by_the_shape_arm(root: Path) -> None:
    pad = _kept_pad(root, "evidence")
    cache = pad / "r3" / "be" / "docs" / "evidence" / "session-load-central-cache"
    cache.mkdir(parents=True)
    (cache / "README.md").write_text("evidence\n")
    (cache / "bench_ab.json").write_text("{}\n")
    _stale_file(cache / "session-load.log")
    archive = pad / "docs" / "evidence"
    archive.mkdir(parents=True)
    (archive / "bundle.tar.gz").write_text("x\n")
    own_name = pad / "r3" / "evidence-build"
    own_name.mkdir()
    (own_name / "keep.bin").write_text("x\n")
    sibling = pad / "r3" / "deep-cache"
    sibling.mkdir()
    (sibling / "blob.bin").write_text("x\n")
    result = _pass(root)
    assert not sibling.exists(), "a plain shape match still goes"
    assert cache.exists() and (cache / "README.md").exists()
    assert not (
        cache / "session-load.log"
    ).exists(), "an exempted dir is still walked: rule (a) carries no evidence exemption"
    assert (archive / "bundle.tar.gz").exists()
    assert own_name.exists() and (own_name / "keep.bin").exists()
    assert [(row.session, row.reason) for row in result.content] == [
        ("evidence", "build shapes past the window + stale output past the window")
    ]
    reasons = [reason for session, reason in result.content_kept if session == "evidence"]
    assert any("session-load-central-cache: evidence path" in r for r in reasons)
    assert any("evidence-build: evidence path" in r for r in reasons)
    assert any("docs/evidence/bundle.tar.gz: evidence path" in r for r in reasons)


def test_content_of_a_young_session_is_never_evaluated(root: Path) -> None:
    young = _mk(root, "young", age_h=1)
    pad = young / "scratchpad"
    pad.mkdir()
    _stale_file(pad / "big.log")
    result = _pass(root)
    assert result.content == [] and result.content_kept == []
    assert (pad / "big.log").exists()


def test_content_of_a_live_claimed_session_is_untouched(root: Path) -> None:
    pad = _kept_pad(root, "livemod")
    _stale_file(pad / "big.log")
    (root / "sessions" / "livemod" / ".session.pid").write_text(str(os.getpid()))
    result = _pass(root)
    assert result.content == []
    assert result.content_kept == [("livemod", "claimed by a live process")]
    assert (pad / "big.log").exists()


def test_content_of_a_wake_armed_session_is_untouched(root: Path) -> None:
    pad = _kept_pad(root, "woken")
    _stale_file(pad / "big.log")
    wakes = root / "wakes"
    wakes.mkdir()
    (wakes / "woken.json").write_text(json.dumps({"schedules": [{"id": "w1"}]}))
    result = _pass(root)
    assert result.content == []
    assert result.content_kept == [("woken", "armed wake for this session")]
    assert (pad / "big.log").exists()


def test_content_of_a_monitored_session_is_untouched(root: Path) -> None:
    pad = _kept_pad(root, "watched")
    _stale_file(pad / "big.log")
    (root / "monitors").mkdir()
    (root / "monitors" / "watched.json").write_text(
        json.dumps({"monitors": [{"id": "m1", "every_ms": 60000}]})
    )
    result = _pass(root)
    assert result.content == []
    assert result.content_kept == [("watched", "armed monitor for this session")]
    assert (pad / "big.log").exists()


def test_record_guards_do_not_block_content_the_project_case(root: Path) -> None:
    directory = _mk(root, "kid")
    pad = directory / "scratchpad"
    pad.mkdir()
    _stale_file(pad / "big.log")
    _project(root, "proj", "active", ["kid"])
    result = _pass(root)
    assert "kid" in _kept(result), "the record is kept by the open project"
    assert [row.session for row in result.content] == ["kid"]


def test_the_scratchpad_remover_refuses_outside_paths_and_keeps_its_contract(
    root: Path,
) -> None:
    pad = _kept_pad(root, "guarded")
    sessions = root / "sessions"
    entry = pad / "junk"
    entry.mkdir()
    (entry / "x.txt").write_text("x\n")
    assert remove_scratchpad_entry(entry, config_dir=None, sessions_dir=sessions) is True
    assert not entry.exists()
    piece = pad / "keepme"
    piece.mkdir()
    assert remove_scratchpad_entry(pad, config_dir=None, sessions_dir=sessions) is False
    assert (
        remove_scratchpad_entry(sessions / "guarded", config_dir=None, sessions_dir=sessions)
        is False
    )
    victim = root / "victim.txt"
    victim.write_text("x")
    assert remove_scratchpad_entry(victim, config_dir=None, sessions_dir=sessions) is False
    symlinked = pad / "link"
    os.symlink(os.fspath(victim), os.fspath(symlinked))
    assert remove_scratchpad_entry(symlinked, config_dir=None, sessions_dir=sessions) is False
    assert piece.exists() and victim.exists() and symlinked.is_symlink()
