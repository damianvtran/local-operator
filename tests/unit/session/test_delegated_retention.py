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
import shutil
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


def test_only_the_reapable_origins_go_and_every_other_class_survives(root: Path) -> None:
    """Survival, not predicate values: each directory is still on disk.

    The reap set is ``cleanup.deletable_origins()``, an ALLOW-LIST — deliberately
    not the negation of the visibility predicate. That inversion is the defect
    this pins: it made every origin minted by a later code path deletable by
    default, and reaped a hidden-but-operator-owned origin nobody had decided
    to reap.
    """
    for name, origin in (("sub", "subagent"), ("shell", "agent-shell")):
        _mk(root, name, origin)
    # Hidden, but never reapable: the operator's own configuration run and a
    # value this build does not know. Both fail CLOSED.
    _mk(root, "config", "agent-config")
    _mk(root, "future", "some-origin-minted-next-year")
    for name, origin in (("user", None), ("fork", "fork"), ("ws", "agent-workstream")):
        _mk(root, name, origin, age_h=24 * 300)
    result = _pass(root)
    assert {c.session for c in result.removed} == {"sub", "shell"}
    assert _names(root) == {"config", "future", "user", "fork", "ws"}
    assert result.kept_by_origin == 2
    assert set(result.kept_by_origin_names) == {"config", "future"}


def test_an_agent_config_run_is_never_reaped(root: Path) -> None:
    """A stated DISPOSITION, not an inherited one.

    The operator starts configuration runs from the Agents page, so the run is
    the operator's own chore: reaping it is irreversible, keeping it costs a few
    kilobytes. It is hidden (not in ``resume.USER_ORIGINS``, so no listing shows
    it) AND kept (not in ``deletable_origins()``) — two answers on purpose.
    """
    _mk(root, "config", "agent-config", age_h=24 * 400)
    assert _pass(root, dry_run=True).removed == []
    assert _pass(root).removed == []
    assert "config" in _names(root)


def test_an_agent_config_run_is_not_the_parent_class_either(root: Path) -> None:
    """Kept by BOTH policies: the parent scan stays blind to it.

    A kept-but-hidden origin must not fall through to the parent class, whose
    limits could remove it instead. It belongs to neither removal path.
    """
    _mk(root, "config", "agent-config", age_h=24 * 400)
    parent = run_cleanup(
        root,
        CleanupPolicy(enabled=True, max_sessions=1, max_total_bytes=1, remove_empty=True),
        now=NOW,
        dry_run=True,
    )
    assert parent.scanned == 0
    assert {c.session for c in parent.chosen} == set()


def test_an_unknown_origin_is_kept_because_ambiguity_fails_closed(root: Path) -> None:
    """A value present in ``origin.json`` but not in the reap set is NOT deletable.

    Same direction as an empty marker: it may cost disk, never a session nobody
    can recover. It is also invisible to the parent class (hidden), so neither
    policy removes it.
    """
    _mk(root, "future", "agent-future-thing", age_h=24 * 400)
    assert _pass(root).removed == []
    assert "future" in _names(root)
    parent = run_cleanup(
        root, CleanupPolicy(enabled=True, max_total_bytes=1), now=NOW, dry_run=True
    )
    assert parent.scanned == 0


def test_a_missing_origin_marker_is_the_users_own_and_survives(root: Path) -> None:
    _mk(root, "plain", origin=None, age_h=24 * 400)
    assert _pass(root).removed == []
    assert "plain" in _names(root)


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
    monkeypatch.setattr(dr, "_clock_without_activity", lambda *_args: None)
    assert _pass(root).removed == []
    assert "noclock" in _names(root)


def test_a_sane_birth_stamp_is_believed(root: Path) -> None:
    """Positive control for the implausible cases below: the stamp IS the clock."""
    directory = _mk(root, "sane", "subagent", age_h=None)
    marker = directory / "created_at.json"
    marker.write_text(json.dumps(NOW - 400 * HOUR))
    old = NOW - 400 * HOUR
    os.utime(marker, (old, old))
    os.utime(directory, (old, old))
    assert [c.session for c in _pass(root).removed] == ["sane"]
    assert "sane" not in _names(root)


@pytest.mark.parametrize(
    "stamp",
    [0.0, -1.0, 1.7e8, 1_000_000_000_000.0, 10**400, float("nan"), float("inf")],
    ids=[
        "zero",
        "negative",
        "truncated-epoch",
        "far-future",
        "too-large-for-float",
        "nan",
        "inf",
    ],
)
def test_an_implausible_birth_stamp_is_no_clock_and_the_session_survives(
    root: Path, stamp: float
) -> None:
    """A positive-but-garbage ``created_at`` used to rank as MAXIMALLY OLD.

    So a truncated epoch (``1.7e8`` is 1975) was the first thing reaped — the
    inverse of safe. The stamp is now believed only inside a plausible window
    (:data:`dr.MIN_PLAUSIBLE_BIRTH` .. now + ``MAX_BIRTH_SKEW_S``); outside it the
    directory has no clock and is kept.

    The mtimes are pushed OLD deliberately: that is what makes this a real
    control rather than a coincidence. A fallback to mtime would remove the
    session, so survival proves the stamp was REJECTED, not merely outranked.
    """
    directory = _mk(root, "odd", "subagent", age_h=None)
    marker = directory / "created_at.json"
    marker.write_text(json.dumps(stamp))
    old = NOW - 400 * HOUR
    os.utime(marker, (old, old))
    os.utime(directory, (old, old))
    assert _pass(root).removed == []
    assert "odd" in _names(root)


def test_an_unreadable_birth_marker_still_takes_the_mtime_clock(root: Path) -> None:
    """Only a PRESENT-but-implausible value is rejected; a corrupt marker is not it.

    A marker that does not parse holds no stamp to contradict the fallback, so
    the mtime clock stands (an old one still reaps, which is the pre-existing
    behaviour this change deliberately leaves alone).
    """
    directory = _mk(root, "garbled", "subagent", age_h=None)
    marker = directory / "created_at.json"
    marker.write_text("{not a number")
    old = NOW - 400 * HOUR
    os.utime(marker, (old, old))
    os.utime(directory, (old, old))
    assert [c.session for c in _pass(root).removed] == ["garbled"]


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
    # The kept row names its bucket honestly: "inside the window" and "no usable
    # clock" share it (a missing marker and an implausible birth stamp both land
    # there), so it must not claim the session was recently active.
    assert _kept(result)["fresh"].startswith("inside the 48h window")
    assert not (root / "sessions" / CLEANUP_LOG_NAME).exists()


def test_a_dry_run_names_the_sessions_kept_by_origin(root: Path) -> None:
    """A count the operator cannot act on is not a usable preview.

    These rows are hidden from every listing, so the dry run's kept list is the
    only place they can be found; the name must ride beside ``kept_by_origin``
    exactly as ``window_names`` does for the in-window group.
    """
    _mk(root, "config", "agent-config")
    result = _pass(root, dry_run=True)
    assert result.kept_by_origin == 1
    assert _kept(result)["config"] == "origin not reapable"


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
        ("reapme", "merged-clean clone")
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
        ("mods", "build shapes", ("build shapes",))
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
    assert [row.reason for row in result.content] == ["build shapes"]
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
    assert row["reason"] == "stale output"
    assert row["bytes"] == size and row["entries"] == 1
    assert row["title"] == "" and row["actor"] == "startup"
    assert row["classes"] == ["stale output"]
    # Provenance: the aggregate row points at a manifest that names the path.
    assert row["trigger"] == "startup"
    assert row["paths"] == 1
    assert 99.0 < row["activity_age_hours"] < 101.0
    manifest = Path(row["manifest"])
    assert manifest.parent == root / "sessions" / dr.CONTENT_MANIFEST_DIRNAME
    assert manifest.is_file() and manifest.name.endswith("-rowy.json")
    body = json.loads(manifest.read_text())
    assert body["schema"] == dr.CONTENT_MANIFEST_SCHEMA
    assert body["session"] == "rowy" and body["origin"] == "subagent"
    assert body["actor"] == "startup" and body["trigger"] == "startup"
    assert body["policy"] == "delegated_scratchpad"
    assert body["planned"] == [
        {"path": str(big), "bytes": size, "class": "stale output", "method": "remove"}
    ]
    assert body["completed"] == body["planned"]
    assert body["bytes"] == size and body["entries"] == 1


def test_a_manifest_write_failure_never_fails_the_removal(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest is best-effort: the removal the operator asked for proceeds,
    and the log row records ``manifest_error`` instead of a path."""
    pad = _kept_pad(root, "nomanifest")
    big = _stale_file(pad / "desktop-suite.log")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("manifest store is read-only")

    monkeypatch.setattr(dr, "_write_cleanup_manifest", boom)
    _pass(root)
    assert not big.exists(), "a manifest failure must not keep the content"
    row = json.loads((root / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()[0])
    assert row["manifest"] == ""
    assert row["manifest_error"].startswith("planned: OSError")
    # No manifest to point at: the row carries path-level provenance itself.
    assert row["paths"] == [str(big)]
    assert row["paths_total"] == 1
    assert row["trigger"] == "startup"


def test_a_failed_completed_rewrite_still_names_the_taken_paths(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review m1: planned write OK, completed rewrite fails.

    The manifest exists, but its ``completed`` list is empty while the files are
    gone. Pointing the log row at it alone would lose path-level provenance, so
    the row must carry the taken paths itself.
    """
    pad = _kept_pad(root, "halfmanifest")
    big = _stale_file(pad / "desktop-suite.log")
    real = dr._write_cleanup_manifest
    calls: list[int] = []

    def flaky(path: Path, payload: dict[str, Any]) -> None:
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        real(path, payload)

    monkeypatch.setattr(dr, "_write_cleanup_manifest", flaky)
    _pass(root)
    assert not big.exists(), "the completed-rewrite failure must not undo the removal"
    row = json.loads((root / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()[0])
    assert row["manifest"] != "" and row["manifest_error"].startswith("completed: OSError")
    assert row["paths"] == [str(big)] and row["paths_total"] == 1
    body = json.loads(Path(row["manifest"]).read_text())
    assert body["planned"] and body["completed"] == []


def test_an_unreadable_ask_log_keeps_the_record_and_the_content(root: Path) -> None:
    """QA Q-1: ``chmod 000`` on asks.jsonl must fail closed, not read as no asks.

    ``asks.store.read_events`` deliberately swallows an unreadable log into
    ``[]``; without the explicit readability probe, both passes would treat this
    as "no asks" and delete an ask the user can still answer.
    """
    from local_operator.asks import store as ask_store

    directory = _mk(root, "asklocked", age_h=100)  # no roster: the RECORD is at risk
    pad = directory / "scratchpad"
    pad.mkdir()
    shaped = pad / "build"
    shaped.mkdir()
    (shaped / "out.o").write_bytes(b"x" * 4096)
    stamp = ask_store.now_ms()
    assert ask_store.append_event(
        directory,
        {
            "ask_id": "ask-locked",
            "kind": ask_store.EVENT_QUEUED,
            "at": stamp,
            "expires_at": stamp + 600_000,
            "timeout_s": 600,
        },
    )
    asks = directory / "asks.jsonl"
    asks.chmod(0o000)
    try:
        result = _pass(root)
    finally:
        asks.chmod(0o600)
    assert result.removed == [], "the unreadable ask must keep the RECORD"
    assert result.content == []
    assert ("asklocked", "has open asks") in result.content_kept
    assert shaped.exists() and directory.exists()


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
    assert row.paths_total == 2
    assert set(row.paths) == {str(tree), str(pad / "desktop-suite.log")}
    assert row.manifest == "", "a dry run writes no manifest"
    assert not (root / "sessions" / dr.CONTENT_MANIFEST_DIRNAME).exists()
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
    assert row.reason == "merged-clean clone"
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
        ("wtpad", "merged-clean clone")
    ]
    row = result.content[0]
    assert row.method == "worktree-remove" and row.rescue_bundle == ""
    assert not (
        pad / "reap-rescue-lo-before.bundle"
    ).exists(), "a worktree's objects are the shared store's; nothing is bundled"
    assert list(pad.iterdir()) == [], "no bundle, no leftovers"


def test_a_worktree_copy_whose_pointer_escapes_is_kept(root: Path) -> None:
    # A recursive copy keeps the .git FILE, so it still names the ORIGINAL's
    # gitdir while its own path is not registered anywhere: the belt must
    # refuse, and the registered original must come out of the pass untouched.
    # ``shutil.copytree``, not ``cp -Rc``: the reflink flag is macOS-only and
    # fails on the Linux CI runners (no reflink support).
    pad = _kept_pad(root, "copied")
    owner = _worktree_owner(root, "owner2")
    source = _linked_worktree(owner, root / "worktrees" / "outside")
    copy = pad / "lo-before"
    shutil.copytree(os.fspath(source), os.fspath(copy), symlinks=True)
    git_file = copy / ".git"
    assert git_file.is_file(), "the copy keeps the .git FILE, not a directory"
    recorded = git_file.read_text().strip()
    target = Path(recorded.removeprefix("gitdir:").strip())
    assert target == owner / ".git" / "worktrees" / "outside", recorded
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
        ("evidence", "build shapes + stale output")
    ]
    reasons = [reason for session, reason in result.content_kept if session == "evidence"]
    assert any("session-load-central-cache: evidence path" in r for r in reasons)
    assert any("evidence-build: evidence path" in r for r in reasons)
    assert any("docs/evidence/bundle.tar.gz: evidence path" in r for r in reasons)


def _head_sha(repo: Path) -> str:
    """The commit ``repo``'s HEAD points at (for tip-recovery assertions)."""
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT},
        timeout=60,
    )
    return done.stdout.strip()


def test_two_reclaimable_clones_get_one_bundle_each_and_both_tips_recover(
    root: Path,
) -> None:
    # Review F1: with ONE bundle slot per pad, the second tree's unique refs
    # were silently dropped and both trees removed. Per-tree bundles make each
    # tip recoverable from its own bundle, and each tree goes only after ITS
    # bundle verifies.
    pad = _kept_pad(root, "two-clones")
    tips: dict[str, str] = {}
    for name in ("clone-a", "clone-b"):
        tree = _merged_tree(pad, name, root)
        _git(tree, "checkout", "-qb", "stray")
        (tree / f"{name}.txt").write_text(f"{name}\n")
        _git(tree, "add", f"{name}.txt")
        _git(tree, "commit", "-qm", "stray")
        tips[name] = _head_sha(tree)
        _git(tree, "checkout", "-q", "main")
    result = _pass(root)
    assert not (pad / "clone-a").exists() and not (pad / "clone-b").exists()
    a = pad / "reap-rescue-clone-a.bundle"
    b = pad / "reap-rescue-clone-b.bundle"
    assert a.exists() and b.exists(), sorted(entry.name for entry in pad.iterdir())
    assert result.errors == 0
    assert result.content and len(result.content[0].rescue_bundles) == 2
    for name, bundle in (("clone-a", a), ("clone-b", b)):
        heads = _git_stdout("bundle", "list-heads", os.fspath(bundle))
        assert "refs/heads/stray" in heads and tips[name] in heads
        dest = root / f"recovered-{name}"
        _git_stdout("clone", "-q", "-b", "stray", os.fspath(bundle), os.fspath(dest))
        assert (dest / f"{name}.txt").read_text() == f"{name}\n"


def test_a_tag_only_unique_commit_is_bundled_before_the_tree_goes(root: Path) -> None:
    # Review F2: tags are durable refs. A tagged checkpoint whose branch was
    # deleted was invisible to the walk, and the tree went with no bundle.
    pad = _kept_pad(root, "tagged")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "checkpoint")
    (tree / "keepme.txt").write_text("checkpoint\n")
    _git(tree, "add", "keepme.txt")
    _git(tree, "commit", "-qm", "checkpoint")
    _git(tree, "tag", "keepme")
    _git(tree, "checkout", "-q", "main")
    _git(tree, "branch", "-qD", "checkpoint")
    result = _pass(root)
    assert not tree.exists()
    bundle = pad / "reap-rescue-clone.bundle"
    assert bundle.exists()
    heads = _git_stdout("bundle", "list-heads", os.fspath(bundle))
    assert "refs/tags/keepme" in heads
    dest = root / "tag-recovered"
    _git_stdout("clone", "-q", "-b", "keepme", os.fspath(bundle), os.fspath(dest))
    assert (dest / "keepme.txt").read_text() == "checkpoint\n"
    assert result.errors == 0


def _blob(tree: Path, payload: str = "local-only blob\n") -> str:
    """A blob written into ``tree``'s object store and referenced by nothing."""
    done = subprocess.run(
        ["git", "-C", str(tree), "hash-object", "-w", "--stdin"],
        input=payload,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT},
        timeout=60,
    )
    return done.stdout.strip()


def test_a_lone_non_commit_tag_ref_keeps_the_tree(root: Path) -> None:
    # Review F2b / QA Q1: the commit walk silently ignores a tag on a blob or
    # tree, so with that ref as the tree's ONLY local-unique one the walk was
    # empty and the tree was reclaimed with NO bundle (the blob lost). The
    # listing now runs BEFORE the empty-walk return.
    shapes = ("light", "annotated", "tree", "mixed")
    for name in shapes:
        pad = _kept_pad(root, f"nc-{name}")
        # One origin per pad: ``_merged_tree`` keys the bare origin on the tree
        # name, so a shared name would make the second pad's push a
        # non-fast-forward.
        tree = _merged_tree(pad, f"clone-{name}", root)
        if name == "tree":
            tree_sha = subprocess.run(
                ["git", "-C", str(tree), "rev-parse", "HEAD^{tree}"],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, **_GIT},
                timeout=60,
            ).stdout.strip()
            _git(tree, "tag", "snap", tree_sha)
        elif name == "annotated":
            _git(tree, "tag", "-a", "-m", "snap", "snap", _blob(tree))
        else:
            _git(tree, "tag", "snap", _blob(tree))
        if name == "mixed":
            _git(tree, "checkout", "-qb", "stray")
            (tree / "b.txt").write_text("b\n")
            _git(tree, "add", "b.txt")
            _git(tree, "commit", "-qm", "two")
            _git(tree, "checkout", "-q", "main")
    result = _pass(root)
    kept = dict(result.content_kept)
    for name in shapes:
        assert kept[f"nc-{name}"] == (f"clone-{name}: a local ref points at a non-commit object")
        assert (root / "sessions" / f"nc-{name}" / "scratchpad" / f"clone-{name}").exists()
    assert result.content == []


def test_a_clone_with_registered_worktrees_is_kept_whole(root: Path) -> None:
    # Review F3: the clone arm ignored worktree registrations, so removing the
    # store destroyed the worktrees' objects (including a worktree the same
    # pass kept). Fail closed: keep the clone, touch nothing.
    pad = _kept_pad(root, "ownerpad")
    owner = _merged_tree(pad, "owner", root)
    inside = pad / "inside-wt"
    outside = root / "worktrees" / "outside-wt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    for target, label in ((inside, "inside"), (outside, "outside")):
        _git(owner, "checkout", "-qb", f"tmp-{label}")
        (owner / f"{label}.txt").write_text(f"{label}\n")
        _git(owner, "add", f"{label}.txt")
        _git(owner, "commit", "-qm", label)
        sha = _head_sha(owner)
        _git(owner, "checkout", "-q", "main")
        _git(owner, "branch", "-qD", f"tmp-{label}")
        _git(owner, "worktree", "add", "--detach", str(target), sha)
    result = _pass(root)
    assert owner.exists() and (owner / "a.txt").exists()
    assert (inside / "inside.txt").exists() and (outside / "outside.txt").exists()
    kept = [reason for session, reason in result.content_kept if session == "ownerpad"]
    assert any(reason.startswith("owner: has registered worktrees") for reason in kept), kept
    assert any(reason.startswith("inside-wt: HEAD is not merged") for reason in kept), kept
    assert result.content == []


def test_content_only_confines_the_phase_to_the_previewed_names(root: Path) -> None:
    # Review F4: the apply re-runs the pass, so without confinement a pad that
    # became eligible after the preview would be reclaimed unseen.
    for name in ("pad-a", "pad-b"):
        pad = _kept_pad(root, name)
        _stale_file(pad / "desktop-suite.log")
    preview = _pass(root, dry_run=True)
    assert sorted(row.session for row in preview.content) == ["pad-a", "pad-b"]
    result = _pass(root, content_only=["pad-a"])
    assert [row.session for row in result.content] == ["pad-a"]
    assert not (root / "sessions" / "pad-a" / "scratchpad" / "desktop-suite.log").exists()
    assert (root / "sessions" / "pad-b" / "scratchpad" / "desktop-suite.log").exists()


def test_an_entry_cap_abort_keeps_the_pad_and_counts_an_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _kept_pad(root, "capped")
    log = _stale_file(pad / "desktop-suite.log")
    for index in range(5):
        (pad / f"note-{index}.txt").write_text("x\n")
    monkeypatch.setattr(dr, "CONTENT_MAX_ENTRIES", 3)
    result = _pass(root)
    assert log.exists(), "an aborted pad takes nothing"
    assert result.errors == 1
    assert any("more than 3 entries" in reason for _, reason in result.content_kept)


def test_a_repo_cap_abort_keeps_the_pad_and_counts_an_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pad = _kept_pad(root, "repocap")
    tree = _merged_tree(pad, "clone", root)
    _git(tree, "checkout", "-qb", "stray")
    (tree / "b.txt").write_text("b\n")
    _git(tree, "add", "b.txt")
    _git(tree, "commit", "-qm", "two")
    _git(tree, "checkout", "-q", "main")
    monkeypatch.setattr(dr, "CONTENT_MAX_REPOS", 0)
    result = _pass(root)
    assert tree.exists()
    assert result.errors == 1
    assert any("more than 0 git repositories" in reason for _, reason in result.content_kept)


def test_an_unreadable_subtree_aborts_the_pad_and_takes_nothing(root: Path) -> None:
    pad = _kept_pad(root, "lockedpad")
    log = _stale_file(pad / "desktop-suite.log")
    locked = pad / "locked-dir"
    locked.mkdir()
    (locked / "inner.txt").write_text("x\n")
    os.chmod(locked, 0)
    try:
        result = _pass(root)
    finally:
        os.chmod(locked, 0o755)
    assert log.exists(), "an aborted pad takes nothing"
    assert result.errors == 1
    assert any("cannot be read" in reason for _, reason in result.content_kept)


def test_a_spent_budget_stops_the_content_loop_between_batches(root: Path) -> None:
    for name in ("budget-a", "budget-b"):
        pad = _kept_pad(root, name)
        _stale_file(pad / "desktop-suite.log")
    result = _pass(root, batch_size=1, budget_s=0.0)
    assert [row.session for row in result.content] == ["budget-a"]
    assert result.content_budget_exhausted is True
    assert result.content_remaining == 1
    assert not (root / "sessions" / "budget-a" / "scratchpad" / "desktop-suite.log").exists()
    assert (root / "sessions" / "budget-b" / "scratchpad" / "desktop-suite.log").exists()


def test_the_content_guard_re_stat_mirrors_rule_six(root: Path) -> None:
    # Review N1: a session that moved AFTER the scan is inside the window
    # again — the content guard must re-read the clock like the record pass's
    # rule 6, not trust the scan's snapshot.
    directory = _mk(root, "moved", origin="subagent", age_h=100)
    transcript = directory / "transcript.jsonl"
    os.utime(transcript, (NOW, NOW))
    cand = dr._Cand(directory, clock=NOW - 100 * HOUR, has_transcript=True)
    guard = dr._content_guard(
        cand, config_dir=root, now=NOW, cutoff=NOW - 48 * HOUR, live_resolved=None
    )
    assert guard.reason == "active since the scan"
    assert guard.age_hours(NOW) == 0.0


def test_content_of_a_young_session_is_kept_whole_inside_the_retention_window(
    root: Path,
) -> None:
    """The age gate is back (2026-10-19): live-adjacent content is not ours.

    The reversed policy reclaimed a build shape from a transcript touched an
    hour ago as long as no claim/lease/wake was visible. A pad whose activity is
    inside the record window is being worked in; its build shape survives, the
    guard's kept reason says why, and the RECORD stays too.
    """
    young = _mk(root, "young", age_h=1)
    pad = young / "scratchpad"
    pad.mkdir()
    shaped = pad / "build"
    shaped.mkdir()
    (shaped / "out.o").write_bytes(b"x" * 4096)
    notes = pad / "notes.md"
    notes.write_text("keep me\n")
    result = _pass(root)
    assert result.content == []
    assert ("young", "inside the retention window") in result.content_kept
    assert shaped.exists(), "a build shape inside the window is NOT reclaimed"
    assert notes.read_text() == "keep me\n"
    assert young.exists(), "the RECORD stays: the age window governs it"
    assert result.removed == []


def test_content_of_a_pad_with_no_activity_clock_is_kept(root: Path) -> None:
    """No clock is not old: the content guard fails closed like rule 1."""
    directory = _mk(root, "clockless", age_h=None)
    pad = directory / "scratchpad"
    pad.mkdir()
    shaped = pad / "build"
    shaped.mkdir()
    (shaped / "out.o").write_bytes(b"x" * 4096)
    result = _pass(root)
    assert result.content == []
    assert ("clockless", "no activity clock") in result.content_kept
    assert shaped.exists()


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


def test_content_of_a_pad_with_unread_spooled_mail_is_untouched(root: Path) -> None:
    """A question the user has not seen outranks reclaiming cache bytes."""
    pad = _kept_pad(root, "mailed")
    _stale_file(pad / "big.log")
    (pad.parent / "inbox.jsonl").write_text('{"x":1}\n')
    # Backdate the spool: a mailbox touched NOW is "active since the scan"
    # (mail is part of the activity clock), which would keep the pad before the
    # dedicated mail guard is consulted.
    os.utime(pad.parent / "inbox.jsonl", (NOW - 100 * HOUR, NOW - 100 * HOUR))
    result = _pass(root)
    assert result.content == []
    assert result.content_kept == [("mailed", "has unread spooled mail")]
    assert (pad / "big.log").exists()


def test_content_of_a_pad_with_an_open_ask_is_untouched(root: Path) -> None:
    """An answerable ask is live conversation, whatever the transcript age."""
    from local_operator.asks import store as ask_store

    pad = _kept_pad(root, "asking")
    _stale_file(pad / "big.log")
    stamp = ask_store.now_ms()
    assert ask_store.append_event(
        pad.parent,
        {
            "ask_id": "ask-1",
            "kind": ask_store.EVENT_QUEUED,
            "at": stamp,
            "expires_at": stamp + 600_000,
            "timeout_s": 600,
        },
    )
    result = _pass(root)
    assert result.content == []
    assert result.content_kept == [("asking", "has open asks")]
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


def test_a_parent_users_pad_is_never_content_reclaimed(root: Path) -> None:
    """The audit reports both classes; the CONTENT layer takes from one.

    A parent (user) directory is not in the delegated pool at all, so its build
    tree survives however old and however shaped — and that is the promise the
    module docstring makes about the user's own conversations.
    """
    user = _mk(root, "mine", origin=None, age_h=100)
    pad = user / "scratchpad"
    pad.mkdir()
    shaped = pad / "build"
    shaped.mkdir()
    (shaped / "bundle.js").write_bytes(b"x" * 4096)
    notes = pad / "notes.md"
    notes.write_text("keep me\n")

    result = _pass(root)

    assert result.content == []
    assert shaped.exists() and notes.exists()
    assert user.exists()


def test_an_agent_workstream_pad_is_never_in_the_content_pool(root: Path) -> None:
    """Visible parent-class work is audit-only, however shaped and however old.

    ``agent-workstream`` is the user-facing parent class; the content pool is
    built from ``_is_deletable_dir`` candidates only, so its build tree is not
    even a content candidate.
    """
    directory = _mk(root, "flowpad", origin="agent-workstream", age_h=100)
    pad = directory / "scratchpad"
    pad.mkdir()
    shaped = pad / "build"
    shaped.mkdir()
    (shaped / "out.o").write_bytes(b"x" * 4096)
    result = _pass(root)
    assert result.content == []
    assert shaped.exists()


def test_a_live_subagents_scratchpad_and_record_survive_the_full_pass(root: Path) -> None:
    """Liveness and the age window are the content gates — so the full pass
    (record AND content) must leave a live child's pad and its transcript
    exactly where they were, even though the record is well past the window."""
    directory = _mk(root, "livemod", age_h=100)
    (directory / ".session.pid").write_text(str(os.getpid()))
    pad = directory / "scratchpad"
    pad.mkdir()
    shaped = pad / "build"
    shaped.mkdir()
    (shaped / "out.o").write_bytes(b"x" * 4096)
    notes = pad / "notes.md"
    notes.write_text("keep me\n")

    result = _pass(root)

    assert result.removed == []
    assert directory.exists() and (directory / dr.TRANSCRIPT_FILENAME).exists()
    assert shaped.exists() and notes.exists()
    assert ("livemod", "claimed by a live process") in result.content_kept


def test_a_spent_budget_stops_even_when_no_pad_has_reclaimable_content(root: Path) -> None:
    """Minor 1: the pre-fix deadline check ran only at a 100-removal/100-failure
    batch boundary, so a pool of pads with nothing to reclaim walked to the end
    and never set ``content_budget_exhausted``. The per-pad check must stop after
    the one-pad progress guarantee."""
    for name in ("plain-a", "plain-b", "plain-c"):
        pad = _kept_pad(root, name)
        (pad / "notes.md").write_text("keep\n")

    result = _pass(root, batch_size=100, budget_s=0.0)

    assert result.content == []
    assert result.content_budget_exhausted is True
    assert result.content_remaining == 2


# --------------------------------------------------------------------------
# The non-git COPY arm (content rule (d))
# --------------------------------------------------------------------------

#: The candidate floor is 50 MiB and the proof hash-reads every byte, so these
#: tests build a real 55 MiB file rather than patching the constant: the module
#: binds ``min_bytes`` as a default argument, and a patched global would prove
#: nothing about the shipped floor.
COPY_MB = 55


def _big_file(path: Path, *, fill: bytes = b"\0", size_mb: int = COPY_MB) -> Path:
    """A real file of ``size_mb``; ``fill`` changes the last byte (and the blob)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.seek(size_mb * 1024 * 1024 - 1)
        handle.write(fill[-1:])
    return path


def _copy_reference(root: Path, name: str = "source") -> Path:
    """A reference repo with a bare origin, kept OUTSIDE every pad."""
    origin = root / "copy-origins" / f"{name}.git"
    origin.parent.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "--bare", "-b", "main", os.fspath(origin))
    repo = root / "copy-refs" / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "remote", "add", "origin", os.fspath(origin))
    return repo


def _publish(repo: Path) -> None:
    """Commit everything and publish ``main`` as the merged trunk."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "content")
    _git(repo, "push", "-q", "origin", "HEAD:main")
    _git(repo, "fetch", "-q", "origin")
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")


def _copy_tree(source: Path, dest: Path) -> Path:
    """Copy every working-tree file of ``source`` to ``dest`` (never ``.git``)."""
    dest.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".git" in relative.parts:
            continue
        if path.is_dir():
            (dest / relative).mkdir(parents=True, exist_ok=True)
        else:
            (dest / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest / relative)
    return dest


def _copy_policy(references: tuple[Path, ...] = (), **kwargs: Any) -> CleanupPolicy:
    return CleanupPolicy(
        delegated_enabled=True,
        delegated_scratchpad_enabled=True,
        delegated_non_git_copies_enabled=True,
        delegated_non_git_copy_references=tuple(os.fspath(path) for path in references),
        **kwargs,
    )


def _copy_provider(*references: Path, **kwargs: Any):
    """``run_delegated_pass`` takes a provider, not a policy instance."""
    policy = _copy_policy(references, **kwargs)
    return lambda: policy


def _published_source(root: Path) -> Path:
    source = _copy_reference(root)
    _big_file(source / "payload")
    (source / "sub").mkdir()
    (source / "sub" / "a.txt").write_text("alpha\n")
    _publish(source)
    return source


def test_a_fully_reproducible_copy_is_reclaimed_file_by_file(root: Path) -> None:
    """Every byte exists at the same path in a merged tree: the files go."""
    source = _published_source(root)
    pad = _kept_pad(root, "copied")
    copy = _copy_tree(source, pad / "copy")
    total = sum(path.stat().st_size for path in copy.rglob("*") if path.is_file())

    result = _pass(root, policy=_copy_provider(source))

    assert not (copy / "payload").exists()
    assert not (copy / "sub" / "a.txt").exists()
    assert copy.exists(), "only matched FILES are removed; the directory stays"
    assert [(row.session, row.classes, row.entries) for row in result.content] == [
        ("copied", ("non-git copy",), 2)
    ]
    assert result.content[0].bytes == total
    assert result.copy_candidates == 1
    assert result.copy_candidate_bytes == total
    assert result.copy_reclaimed_bytes == total
    assert result.copy_skipped is None
    assert result.errors == 0


def test_a_copy_with_unique_files_keeps_only_them(root: Path) -> None:
    """Path+blob mismatch is a keep: session artifacts are never proof."""
    source = _published_source(root)
    pad = _kept_pad(root, "partial")
    copy = _copy_tree(source, pad / "copy")
    artifact = copy / "artifact.log"
    artifact.write_text("only this run produced it\n")

    result = _pass(root, policy=_copy_provider(source))

    assert artifact.exists()
    assert not (copy / "payload").exists() and not (copy / "sub" / "a.txt").exists()
    assert [(name, "unmatched file(s)" in reason) for name, reason in result.content_kept] == [
        ("partial", True)
    ]
    assert "artifact.log" in result.content_kept[0][1]


def test_a_copy_below_the_share_threshold_keeps_everything(root: Path) -> None:
    """0% matched is a refusal of the WHOLE candidate, with the share named."""
    source = _published_source(root)
    pad = _kept_pad(root, "foreign")
    copy = pad / "copy"
    _big_file(copy / "payload", fill=b"x")  # same name, different bytes
    (copy / "sub").mkdir()
    (copy / "sub" / "a.txt").write_text("alpha\n")

    result = _pass(root, policy=_copy_provider(source))

    assert (copy / "payload").exists() and (copy / "sub" / "a.txt").exists()
    assert result.content == []
    assert result.copy_reclaimed_bytes == 0
    reason = dict(result.content_kept)["foreign"]
    # A nonzero share that truncates to zero reads as "<1%", never "0%" beside
    # 6 bytes (design C4): the earlier spelling made the instrument look broken.
    assert "only <1% of its bytes" in reason and "match a merged tree" in reason


def test_a_copy_with_no_reference_repository_is_kept(root: Path) -> None:
    """No configured reference and none in the pad: nothing can be proved."""
    pad = _kept_pad(root, "unprovable")
    copy = pad / "copy"
    _big_file(copy / "payload")

    result = _pass(root, policy=lambda: _copy_policy())

    assert (copy / "payload").exists()
    assert result.content == []
    assert result.copy_candidates == 0
    assert result.copy_skipped == "no reference repository"


def test_a_nested_git_refuses_the_copy_and_the_git_arm_still_sees_it(root: Path) -> None:
    """The copy arm must never delete a repository arm (c) would keep."""
    source = _published_source(root)
    pad = _kept_pad(root, "nested")
    copy = _copy_tree(source, pad / "code")
    nested = copy / "project"
    nested.mkdir()
    _git(nested, "init", "-q", "-b", "main")
    (nested / "f").write_text("x")
    _git(nested, "add", "f")
    _git(nested, "commit", "-qm", "c")  # no remote: unmerged

    result = _pass(root, policy=_copy_provider(source))

    assert (copy / "payload").exists(), "a nested repository blocks this arm below it"
    reasons = [reason for name, reason in result.content_kept if name == "nested"]
    assert any("contains a git repository" in reason for reason in reasons)
    assert any("project:" in reason for reason in reasons), reasons
    assert result.copy_reclaimed_bytes == 0


def test_an_unmerged_reference_does_not_qualify(root: Path) -> None:
    """A repo with no remote trunk proves nothing, however matching it looks."""
    source = _published_source(root)
    unmerged = root / "copy-refs" / "unmerged"
    unmerged.mkdir(parents=True)
    _git(unmerged, "init", "-q", "-b", "main")
    _big_file(unmerged / "payload")
    _git(unmerged, "add", "payload")
    _git(unmerged, "commit", "-qm", "c")
    pad = _kept_pad(root, "nostrunk")
    copy = _copy_tree(source, pad / "copy")

    result = _pass(root, policy=_copy_provider(unmerged))

    assert (copy / "payload").exists()
    assert result.copy_candidates == 0
    reasons = [reason for name, reason in result.content_kept if name == "nostrunk"]
    assert any("no qualifying reference repository" in reason for reason in reasons), reasons


def test_an_empty_directory_scaffold_is_not_an_error(root: Path) -> None:
    """Measured: a scratch HOME's empty ``Library`` has entries and zero bytes."""
    pad = _kept_pad(root, "scaffold")
    (pad / "review-home" / "Library" / "Caches").mkdir(parents=True)

    report = dr.scan_non_git_copies(root / "sessions", min_store_bytes=1)

    assert report.errors == 0 and report.candidates == []

    result = _pass(root, policy=_copy_provider(_published_source(root)))
    assert result.errors == 0
    assert not any("zero files" in reason for _, reason in result.content_kept)


def test_zero_files_with_real_bytes_is_still_an_impossible_result(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard needs a SECOND source; its contradiction is what it reports."""
    pad = _kept_pad(root, "vacuouscopy")
    scaffold = pad / "link-only"
    scaffold.mkdir()
    os.symlink("/nonexistent", scaffold / "link")
    # The walk sees one entry and no regular files; the independent measure
    # reports bytes, which is the measured "0 files in a 1.0 GB copy" shape.
    monkeypatch.setattr(dr, "_bounded_dir_stats", lambda *a, **k: (4096, 1))

    with pytest.raises(dr.NonGitCopyScanError, match="zero files from a non-empty tree"):
        dr._prove_copy(scaffold, trees=())

    report = dr.scan_non_git_copies(root / "sessions", min_store_bytes=1)
    assert report.errors == 1
    assert report.candidates[0].error is True
    assert "zero files" in report.candidates[0].reason

    result = _pass(root, policy=_copy_provider(_published_source(root)))
    assert result.errors == 1
    assert any("zero files" in reason for _, reason in result.content_kept)


def test_the_store_guard_means_could_not_see_the_store(root: Path) -> None:
    """Both directions of the vacuous guard, measured (review B3b).

    A store whose PADS WERE SCANNED and hold no plain copies is an honest
    empty report; only a store with no reachable scratchpad directory over the
    floor is the instrument failing.
    """
    session = root / "sessions" / "fileonly"
    (session / "scratchpad").mkdir(parents=True)
    (session / "scratchpad" / "notes.txt").write_text("x")
    honest = dr.scan_non_git_copies(root / "sessions", min_store_bytes=1)
    assert honest.candidates == [] and honest.count == 0

    # No scratchpad directory at all, over a store that holds bytes: the scan
    # could not see the class, so it must not read as "nothing to reclaim".
    blind = root / "blind-store" / "sessions"
    (blind / "transcript-only").mkdir(parents=True)
    (blind / "transcript-only" / "transcript.jsonl").write_text("x" * 64)
    with pytest.raises(dr.NonGitCopyScanError, match="could not see any scratchpad"):
        dr.scan_non_git_copies(blind, min_store_bytes=1)

    empty = root / "empty-store" / "sessions"
    empty.mkdir(parents=True)
    assert dr.scan_non_git_copies(empty, min_store_bytes=1).candidates == []


def test_an_unreadable_store_raises_instead_of_reporting_empty(root: Path) -> None:
    """An unlistable store is never "nothing to reclaim" (review B3a)."""
    not_a_store = root / "sessions-is-a-file"
    not_a_store.write_text("not a directory")
    with pytest.raises(dr.NonGitCopyScanError, match="store cannot be read"):
        dr.scan_non_git_copies(not_a_store)


def test_an_unreadable_file_refuses_the_candidate_without_aborting_the_pad(
    root: Path,
) -> None:
    """Measured: a mode-000 fixture inside a real candidate must not raise."""
    source = _published_source(root)
    # The reference must hold the SAME path, or the proof would never open the
    # file (a path no tree has is unmatched without a read) and the test would
    # prove nothing about the unreadable case.
    (source / "sub" / "private.txt").write_text("published\n")
    _publish(source)
    pad = _kept_pad(root, "private")
    copy = _copy_tree(source, pad / "copy")
    private = copy / "sub" / "private.txt"
    private.write_text("secret\n")
    os.chmod(private, 0o000)
    other = _stale_file(pad / "big.log")  # the pad's other arms must still run

    try:
        result = _pass(root, policy=_copy_provider(source))
    finally:
        os.chmod(private, 0o600)

    assert (copy / "payload").exists(), "the whole candidate is kept"
    reasons = [reason for name, reason in result.content_kept if name == "private"]
    assert any("cannot read" in reason for reason in reasons), reasons
    assert not other.exists(), "the stale-output arm on the rest of the pad still ran"
    assert result.errors == 0


def test_a_report_lists_both_provable_and_withheld_copies(root: Path) -> None:
    """``would reclaim now`` and ``proof ok, inside the window`` are separate."""
    source = _published_source(root)
    old_pad = _kept_pad(root, "oldcopy")
    _copy_tree(source, old_pad / "copy")
    # A KEPT record that is nevertheless young: the parent's roster keeps the
    # session, while its own 1 h transcript clock puts it inside the window.
    _mk(root, "parent-youngcopy", None, age_h=10, roster=["youngcopy"])
    young_pad = _mk(root, "youngcopy", age_h=1) / "scratchpad"
    young_pad.mkdir()
    _copy_tree(source, young_pad / "copy")

    report = dr.scan_non_git_copies(
        root / "sessions", references=(os.fspath(source),), now=NOW, window_hours=48
    )

    rows = {row.session: row for row in report.candidates}
    assert rows["oldcopy"].proof_ok and rows["oldcopy"].would_reclaim
    assert rows["youngcopy"].proof_ok and not rows["youngcopy"].would_reclaim
    assert "inside the 48h window" in rows["youngcopy"].reason
    assert report.reclaimable_sessions == 1 and report.within_window_sessions == 1
    assert rows["youngcopy"].waiting is True and rows["oldcopy"].waiting is False
    assert report.reclaimable_bytes == rows["oldcopy"].matched_bytes
    assert rows["oldcopy"].proved_by == (os.path.realpath(source),)


def test_the_report_counts_sessions_not_provable_subtrees(root: Path) -> None:
    """One session with two provable siblings is ONE session (design D2)."""
    source = _published_source(root)
    pad = _kept_pad(root, "twocopies")
    _copy_tree(source, pad / "a")
    _copy_tree(source, pad / "b")

    report = dr.scan_non_git_copies(
        root / "sessions", references=(os.fspath(source),), now=NOW, window_hours=48
    )

    assert report.count == 2, "both top-level directories are candidates"
    assert report.reclaimable_sessions == 1
    assert report.reclaimable_bytes == sum(
        row.matched_bytes for row in report.candidates if row.would_reclaim
    )


def test_a_dry_run_lists_the_copy_and_removes_nothing(root: Path) -> None:
    source = _published_source(root)
    pad = _kept_pad(root, "drycopy")
    copy = _copy_tree(source, pad / "copy")
    total = sum(path.stat().st_size for path in copy.rglob("*") if path.is_file())

    result = _pass(root, policy=_copy_provider(source), dry_run=True)

    assert (copy / "payload").exists()
    assert [(row.classes, row.bytes) for row in result.content] == [(("non-git copy",), total)]
    assert result.copy_reclaimed_bytes == total


def test_the_hash_budget_defers_a_candidate(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _published_source(root)
    pad = _kept_pad(root, "budgeted")
    copy = _copy_tree(source, pad / "copy")
    monkeypatch.setattr(dr, "NON_GIT_COPY_HASH_BUDGET_BYTES", 1)

    result = _pass(root, policy=_copy_provider(source))

    assert (copy / "payload").exists()
    assert result.copy_budget_exhausted is True
    assert result.copy_candidates == 1 and result.copy_reclaimed_bytes == 0
    assert "hash budget" in dict(result.content_kept)["budgeted"]


def test_the_copy_arm_off_leaves_the_copy_alone(root: Path) -> None:
    """The switch is the gate: no proof runs and nothing under the copy moves."""
    source = _published_source(root)
    pad = _kept_pad(root, "offcopy")
    copy = _copy_tree(source, pad / "copy")

    result = _pass(
        root,
        policy=lambda: CleanupPolicy(delegated_enabled=True, delegated_scratchpad_enabled=True),
    )

    assert (copy / "payload").exists()
    assert result.content == []
    assert result.copy_candidates == 0
    assert result.copy_skipped == "disabled"


def test_a_refused_removal_keeps_the_file_and_counts_an_error(root: Path) -> None:
    source = _published_source(root)
    pad = _kept_pad(root, "stuckcopy")
    copy = _copy_tree(source, pad / "copy")
    os.chmod(copy, 0o555)  # readable, not writable: the unlink fails
    try:
        result = _pass(root, policy=_copy_provider(source))
    finally:
        os.chmod(copy, 0o755)

    assert (copy / "payload").exists(), "the refused file is kept"
    assert result.errors == 1
    assert all(row.bytes == 6 for row in result.content), "only sub/a.txt was removable"
    # THE PASS COUNTER IS WHAT WAS TAKEN, NOT WHAT WAS PLANNED (review B2):
    # the planned 55 MB payload never left, so only its sibling's 6 bytes count.
    assert result.copy_reclaimed_bytes == 6
    assert any("cannot be removed" in reason for _, reason in result.content_kept)


def test_a_polluted_git_dir_cannot_redirect_the_proof(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduced: ``GIT_DIR=<repoB>/.git`` made ``_merged_tree(repoA)`` list B."""
    first = _published_source(root)
    second = _copy_reference(root, "other")
    (second / "payload").write_text("a different payload\n")
    _publish(second)

    monkeypatch.setenv("GIT_DIR", os.fspath(second / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", os.fspath(second))
    merged = dr._merged_tree(os.fspath(first))

    assert merged is not None
    payload = first / "payload"
    assert merged[1]["payload"] == dr._blob_sha1(payload, payload.stat().st_size)
    assert merged[1]["payload"] != dr._blob_sha1(
        second / "payload", (second / "payload").stat().st_size
    )


def test_in_pad_references_are_capped_and_sorted(root: Path) -> None:
    """A pad with many repos contributes at most CONTENT_MAX_REPOS, in name order."""
    pad = _kept_pad(root, "manyrepos")
    for index in range(10):
        repo = pad / f"repo-{index}"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")

    repos = dr._repos_in(pad)

    assert len(repos) == dr.CONTENT_MAX_REPOS
    assert [path.name for path in repos] == sorted(path.name for path in repos)


def test_a_pad_with_no_copy_shaped_directory_resolves_no_references(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arm must not walk for repos in a pad it could never prove anything in."""
    pad = _kept_pad(root, "nothingtoprove")
    (pad / "notes.txt").write_text("not a copy\n")
    source = _published_source(root)
    calls: list[Path] = []

    def counted(path: Path) -> list[Path]:
        calls.append(path)
        return []

    monkeypatch.setattr(dr, "_repos_in", counted)
    result = _pass(root, policy=_copy_provider(source))

    assert calls == []
    assert result.content == [] and result.copy_candidates == 0


def test_a_hash_cache_hit_is_not_charged_to_the_budget(root: Path) -> None:
    """A refused parent re-offers files its child will hash; only reads cost."""
    source = _published_source(root)
    pad = _kept_pad(root, "cached")
    copy = _copy_tree(source, pad / "copy")
    merged = dr._merged_tree(os.fspath(source))
    assert merged is not None
    trees = ((os.path.realpath(source), merged[1]),)
    cache = dr._HashCache()

    first = dr._prove_copy(copy, trees=trees, cache=cache)
    second = dr._prove_copy(copy, trees=trees, cache=cache)

    assert first.ok and second.ok
    assert first.hashed_bytes > 0
    assert second.hashed_bytes == 0


def test_the_report_keeps_a_copy_the_pass_would_refuse(root: Path) -> None:
    """Both pad-root guards the pass applies must reach the report (review B1)."""
    source = _published_source(root)
    root_git = _kept_pad(root, "rootgit")
    _copy_tree(source, root_git / "copy")
    (root_git / ".git").write_text("gitdir: elsewhere\n")
    session_git = _kept_pad(root, "sessiongit")
    _copy_tree(source, session_git / "copy")
    (session_git.parent / ".git").write_text("gitdir: elsewhere\n")

    report = dr.scan_non_git_copies(
        root / "sessions",
        references=(os.fspath(source),),
        now=NOW,
        window_hours=48,
    )

    rows = {row.session: row for row in report.candidates}
    assert rows["rootgit"].would_reclaim is False
    assert rows["rootgit"].reason == "the scratchpad root is itself a git tree"
    assert rows["sessiongit"].would_reclaim is False
    assert rows["sessiongit"].reason == "the session directory is itself a git tree"
    assert report.reclaimable_sessions == 0
    assert report.kept_candidates == 2


def test_a_sha256_reference_is_refused_with_its_object_format(root: Path) -> None:
    """A SHA-256 repo silently matches nothing; say WHY it is unusable (D4)."""
    source = _published_source(root)
    sha = root / "copy-refs" / "sha256"
    sha.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", "--object-format=sha256", "-b", "main", str(sha)],
        check=True,
        capture_output=True,
        env={**os.environ, **_GIT},
    )
    _big_file(sha / "payload")
    origin = root / "copy-origins" / "sha256.git"
    subprocess.run(
        [
            "git",
            "init",
            "-q",
            "--bare",
            "--object-format=sha256",
            "-b",
            "main",
            str(origin),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, **_GIT},
    )
    _git(sha, "remote", "add", "origin", str(origin))
    _git(sha, "add", "-A")
    _git(sha, "commit", "-qm", "content")
    _git(sha, "push", "-q", "origin", "HEAD:main")
    _git(sha, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    pad = _kept_pad(root, "sha256ref")
    copy = _copy_tree(source, pad / "copy")

    report = dr.scan_non_git_copies(
        root / "sessions", references=(os.fspath(sha),), now=NOW, window_hours=48
    )

    rows = [row for row in report.candidates if row.session == "sha256ref"]
    assert rows, "the candidate must still be reported"
    assert all("reference object format is sha256" in row.reason for row in rows)
    assert not any(row.would_reclaim for row in rows)
    assert (copy / "payload").exists()


def test_the_early_refusal_uses_the_one_merged_tree_phrase(root: Path) -> None:
    """One refusal phrase in both bounds; no "at a merged-tree path" variant."""
    source = _published_source(root)
    pad = _kept_pad(root, "earlybound")
    copy = _copy_tree(source, pad / "copy")
    _big_file(copy / "run-only.bin", fill=b"y")  # a path no merged tree has

    result = _pass(root, policy=_copy_provider(source))

    reason = dict(result.content_kept)["earlybound"]
    assert "at most" in reason and "match a merged tree" in reason
    assert "merged-tree path" not in reason
    assert (copy / "payload").exists(), "the whole candidate is kept"


def test_candidates_are_recursive_and_the_largest_provable_subtree_wins(root: Path) -> None:
    """``qa-1232/`` matches nothing; its ``base-repo/`` child matches nearly all."""
    source = _published_source(root)
    pad = _kept_pad(root, "qacopy")
    run = pad / "qa-1232"
    run.mkdir()
    (run / "notes.md").write_text("session artifacts\n")
    copy = _copy_tree(source, run / "base-repo")

    report = dr.scan_non_git_copies(
        root / "sessions", references=(os.fspath(source),), now=NOW, window_hours=48
    )
    rows = {row.path: row for row in report.candidates}
    parent = rows["qacopy/scratchpad/qa-1232"]
    child = rows["qacopy/scratchpad/qa-1232/base-repo"]
    assert parent.top_level and not parent.proof_ok
    assert not child.top_level and child.proof_ok
    assert report.count == 1, "the summary counts the top-most attempt only"

    result = _pass(root, policy=_copy_provider(source))
    assert not (copy / "payload").exists()
    assert result.copy_candidates == 1
    assert result.copy_reclaimed_bytes > 0


def test_the_proof_is_a_union_across_reference_repositories(root: Path) -> None:
    """A file proves when ANY reference has it: one repo per half is enough."""
    first = _copy_reference(root, "first")
    _big_file(first / "payload")
    _publish(first)
    second = _copy_reference(root, "second")
    (second / "sub").mkdir()
    (second / "sub" / "a.txt").write_text("alpha\n")
    _publish(second)
    pad = _kept_pad(root, "unioned")
    copy = _copy_tree(first, pad / "copy")
    (copy / "sub").mkdir()
    (copy / "sub" / "a.txt").write_text("alpha\n")

    report = dr.scan_non_git_copies(
        root / "sessions",
        references=(os.fspath(first), os.fspath(second)),
        now=NOW,
        window_hours=48,
    )
    row = report.candidates[0]
    assert row.proof_ok and row.matched_bytes >= COPY_MB * 1024 * 1024
    assert set(row.proved_by) == {
        os.path.realpath(first),
        os.path.realpath(second),
    }


def test_an_operator_hold_refuses_a_provable_copy_in_both_surfaces(root: Path) -> None:
    """A hold outranks proof: the held pad takes nothing, its sibling still goes."""
    source = _published_source(root)
    held_pad = _kept_pad(root, "heldcopy")
    held_copy = _copy_tree(source, held_pad / "copy")
    free_pad = _kept_pad(root, "freecopy")
    free_copy = _copy_tree(source, free_pad / "copy")

    report = dr.scan_non_git_copies(
        root / "sessions",
        references=(os.fspath(source),),
        holds=("heldcopy",),
        now=NOW,
        window_hours=48,
    )
    rows = {row.session: row for row in report.candidates}
    assert rows["heldcopy"].would_reclaim is False
    assert rows["heldcopy"].reason == "operator hold"
    assert rows["freecopy"].would_reclaim is True and rows["freecopy"].reason == ""

    result = _pass(
        root,
        policy=_copy_provider(source, delegated_non_git_copy_holds=("heldcopy",)),
    )

    assert (held_copy / "payload").exists(), "the held copy is untouched"
    assert not (free_copy / "payload").exists(), "its unheld sibling is reclaimed"
    assert any(
        name == "heldcopy" and "operator hold" in reason for name, reason in result.content_kept
    )
    assert result.copy_candidates == 1, "only the unheld sibling was attempted"


def test_unreadable_holds_skip_the_arm_entirely(root: Path) -> None:
    """An unparseable hold list is "I cannot tell", never "nothing is held"."""
    source = _published_source(root)
    pad = _kept_pad(root, "unreadableholds")
    copy = _copy_tree(source, pad / "copy")
    policy = _copy_policy((source,), delegated_non_git_copy_holds_unreadable=True)

    result = _pass(root, policy=lambda: policy)

    assert (copy / "payload").exists(), "a skipped arm takes nothing"
    assert result.copy_skipped == "holds unreadable"
    assert result.copy_candidates == 0

    report = dr.scan_non_git_copies(
        root / "sessions", references=(os.fspath(source),), holds=None, now=NOW
    )
    assert report.skipped == "holds unreadable" and report.candidates == []


def test_a_sub_threshold_parent_is_descended_into(root: Path) -> None:
    """The measured shape: the copy sits under a small run dir (``r3/normal-prev``)."""
    source = _published_source(root)
    pad = _kept_pad(root, "nestedrun")
    run = pad / "r3"  # a few bytes of artifacts, far below the 50 MiB floor
    run.mkdir()
    (run / "notes.md").write_text("run artifacts\n")
    copy = _copy_tree(source, run / "normal-prev")

    result = _pass(root, policy=_copy_provider(source))

    assert not (copy / "payload").exists()
    assert result.copy_reclaimed_bytes > 0
