"""Own cleanup on a Linux mesh node: the three gaps behind node-2's refusals.

Operator directive (2026-11-14, verbatim, relayed by Aida): "the safety guards
should never refuse being able to clean up uncommitted files that look like
scratch, safe to delete, etc. it needs to be tuned … this sort of ask should
never come up again."

Field cases (lop-remote-node-2, Amazon Linux, session 99ffc68d3d05, and the
radient-site marketing session 9ad475b6f5d9 on macOS), each pinned here in its
exact shape:

1a. ``.worktrees/github-default-repos``: the session's ledger row was still on
    the node (same owner, path, dev, ino) but its ctime ``stamp`` had moved
    with the session's own checkouts, and the row predated the ``.git``
    companion row, so nothing re-matched it -> the own tree read unrecorded.
1b. ``gh`` could not answer, so a long-merged branch read "PR state unknown".
1c / 2. Ten same-uid Electron ("Local Operator") processes and ``(sd-pam)``
    had a ``/proc/<pid>/cwd`` this uid may not read; the fresh holder scan
    failed closed on the first one, so EVERY own tree read ``busy=unknown``.
3.  ``radient-site/.worktrees/copy-overhaul`` (PR #32 merged): clean but for
    an ignored ``dist-ssr/`` (Vite's SSR output) -> "dirty".

Every test builds real repositories in a tmp HOME; nothing touches a live
session, the operator's ledger or the network (origins are local bare repos).
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails import spawned
from local_operator.guardrails.classify import Risk, classify_bash

S = "99ffc68d3d05"  # the field session (synthetic store here)
S2 = "58de2f8ff79b"  # case 3's creating session, unrelated to the deleter
BRANCH = "feat/github-default-pushable-repos"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (S, S2):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    monkeypatch.setattr(prov, "_cwd_entries", lambda *, fresh=False: ())
    # raising=False: the same file runs against the pre-fix base to prove each
    # test fails there (PR evidence), where these names do not exist yet.
    monkeypatch.setattr(prov, "_cwd_unreadable", {}, raising=False)
    monkeypatch.setattr(prov, "_github_repo", lambda _: "damianvtran/local-operator")
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid: ())
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    return home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_recorded(home: Path, command: str, *, sid: str = S) -> None:
    """Run ``command`` the way the harness loop does: plan, execute, commit."""
    env = {**_env(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    pending = prov.plan_creations(command, cwd=str(home), env=env)
    subprocess.run(["bash", "-c", command], cwd=str(home), env={**os.environ, **env}, check=True)
    assert pending is not None, command
    prov.commit_creations(pending, session_id=sid, env=env, scratchpad=None)


@pytest.fixture
def repo(home: Path) -> Path:
    """``~/local-operator`` cloned from a local bare origin; ``main`` + a merged branch."""
    origin = home / "origin.git"
    seed = home / "seed"
    seed.mkdir()
    _git("init", "-q", "-b", "main", cwd=seed)
    (seed / ".gitignore").write_text(".worktrees/\ndist-ssr/\nnode_modules\n")
    (seed / "a.py").write_text("a = 1\n")
    _git("add", ".", cwd=seed)
    _git("commit", "-qm", "init", cwd=seed)
    _git("clone", "-q", "--bare", str(seed), str(origin), cwd=home)
    repo = home / "local-operator"
    _git("clone", "-q", str(origin), str(repo), cwd=home)
    (repo / ".worktrees").mkdir()
    return repo


def _merged_branch(repo: Path, branch: str = BRANCH) -> None:
    """``branch`` pushed, then merged into origin's ``main`` with a merge commit."""
    _git("checkout", "-qb", branch, cwd=repo)
    (repo / "b.py").write_text("b = 1\n")
    _git("add", "b.py", cwd=repo)
    _git("commit", "-qm", "feature", cwd=repo)
    _git("push", "-q", "origin", branch, cwd=repo)
    _git("checkout", "-q", "main", cwd=repo)
    _git("merge", "-q", "--no-ff", "-m", "Merge pull request #3842", branch, cwd=repo)
    _git("push", "-q", "origin", "main", cwd=repo)


def _rank(home: Path, command: str, *, sid: str = S) -> tuple[Risk, str]:
    v = classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)
    return v.risk, " | ".join(t.reason for t in v.targets)


def _ledger(home: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(home / ".local-operator" / "guardrails" / "created.sqlite"))


def _age_row_like_node2(home: Path, path: Path, *, prestamp: bool = False) -> None:
    """Make the row look like node-2's: no ``.git`` companion, a moved stamp.

    node-2: row stamp 1794449817716387514, directory ctime now
    1794683305395946140, and no ``worktree-gitfile`` row (the runtime that
    wrote it predates the companion). ``prestamp``: a row written before the
    ``stamp`` column existed (NULL), the other ledger age the brief names.
    """
    real = os.path.realpath(path)
    with _ledger(home) as conn:
        conn.execute("DELETE FROM created WHERE kind = ?", (prov.WORKTREE_GITFILE_KIND,))
        conn.execute(
            "UPDATE created SET stamp = "
            + ("NULL" if prestamp else "stamp + 233487679558")
            + " WHERE path = ? AND kind = 'worktree'",
            (real,),
        )


def _github_default_repos(home: Path, repo: Path) -> Path:
    """Field case 1: a CLEAN worktree on a branch whose PR merged."""
    _merged_branch(repo)
    _run_recorded(
        home, f"cd ~/local-operator && git worktree add .worktrees/github-default-repos {BRANCH}"
    )
    wt = repo / ".worktrees" / "github-default-repos"
    _age_row_like_node2(home, wt)
    return wt


FIELD_1 = "cd ~/local-operator && git worktree remove --force .worktrees/github-default-repos"


# -- 1a: rows are NOT lost, they stop MATCHING ---------------------------------


@pytest.mark.parametrize("prestamp", [False, True], ids=["ctime-drift", "pre-stamp-row"])
def test_1a_drifted_row_without_companion_still_owns_the_tree(
    home: Path, repo: Path, prestamp: bool
) -> None:
    _merged_branch(repo)
    _run_recorded(
        home, f"cd ~/local-operator && git worktree add .worktrees/github-default-repos {BRANCH}"
    )
    wt = repo / ".worktrees" / "github-default-repos"
    _age_row_like_node2(home, wt, prestamp=prestamp)
    record = prov.owned_record(str(wt), S)
    assert record is not None and record.kind == "worktree"
    risk, reason = _rank(home, FIELD_1)
    assert risk is Risk.LOW, reason
    assert "created by this session" in reason


def test_1a_a_fresh_process_same_session_reads_the_same_row(home: Path, repo: Path) -> None:
    """Restart / model switch: a NEW interpreter, same session id, same verdict."""
    wt = _github_default_repos(home, repo)
    code = (
        "import sys; from local_operator.guardrails import provenance as p; "
        "r = p.owned_record(sys.argv[1], sys.argv[2]); print(r.kind if r else None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(wt), S],
        env={**os.environ, **_env(home)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == "worktree"


def test_1a_a_git_file_newer_than_the_row_is_another_generation(home: Path, repo: Path) -> None:
    """Negative: a ``.git`` written after the row (a re-add at the reused inode) fails closed."""
    wt = _github_default_repos(home, repo)
    with _ledger(home) as conn:
        conn.execute(
            "UPDATE created SET created_at = created_at - 3600 WHERE path = ?",
            (os.path.realpath(wt),),
        )
    assert prov.owned_record(str(wt), S) is None
    assert _rank(home, FIELD_1)[0] is Risk.HIGH


def test_1a_another_sessions_drifted_row_is_still_foreign(home: Path, repo: Path) -> None:
    """The witness re-matches rows for their OWNER only: S2 gains nothing from S's row."""
    wt = _github_default_repos(home, repo)
    assert prov.owned_record(str(wt), S2) is None
    assert _rank(home, f"rm -rf {wt}", sid=S2)[0] is Risk.HIGH


# -- 1b: gh cannot answer -> git decides, and only on proof ----------------------


def test_1b_gh_down_merged_branch_reads_merged_by_git(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)  # no auth / no network
    wt = _github_default_repos(home, repo)
    activity = prov.worktree_activity(str(wt), S2)
    assert activity.pr_state == "none", activity
    assert "is in origin/main (read from origin)" in activity.facts()
    # Unrecorded for S2 (another session), clean and idle: the clean-tree LOW.
    risk, reason = _rank(
        home, "git worktree remove ~/local-operator/.worktrees/github-default-repos", sid=S2
    )
    assert risk is Risk.LOW, reason
    assert "state from git" in reason


def test_1b_unmerged_branch_stays_unknown_and_high(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    _run_recorded(home, "cd ~/local-operator && git worktree add -b feat/wip .worktrees/wip")
    wt = repo / ".worktrees" / "wip"
    (wt / "c.py").write_text("c = 1\n")
    _git("add", "c.py", cwd=wt)
    _git("commit", "-qm", "wip", cwd=wt)
    activity = prov.worktree_activity(str(wt), S2)
    assert activity.pr_state == "unknown"
    assert "is not in origin/main" in activity.git_unproven
    risk, _reason = _rank(home, "git worktree remove ~/local-operator/.worktrees/wip", sid=S2)
    assert risk is Risk.HIGH


def test_1b_remote_branch_ahead_of_main_is_not_merged(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-11-09 shape: local HEAD in main, but the PR's newer commits on origin."""
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    wt = _github_default_repos(home, repo)
    other = repo.parent / "other"
    _git("clone", "-q", str(repo.parent / "origin.git"), str(other), cwd=repo.parent)
    _git("checkout", "-q", BRANCH, cwd=other)
    (other / "d.py").write_text("d = 1\n")
    _git("add", "d.py", cwd=other)
    _git("commit", "-qm", "more review fixes", cwd=other)
    _git("push", "-q", "origin", BRANCH, cwd=other)
    merged, why = prov.git_merged(str(wt), BRANCH)
    assert not merged and "not proven" in why, why
    assert prov.worktree_activity(str(wt), S2).pr_state == "unknown"


def test_1b_fetch_failure_uses_existing_refs_only_when_they_prove_it(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    wt = _github_default_repos(home, repo)
    _git("fetch", "-q", "origin", cwd=repo)  # origin/main already holds the merge
    _git("remote", "set-url", "origin", str(repo.parent / "gone.git"), cwd=repo)
    merged, why = prov.git_merged(str(wt), BRANCH)
    assert merged and "already-fetched refs" in why, why


def test_1b_fetch_failure_with_stale_refs_never_claims_merged(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merged on origin AFTER the last fetch, then origin unreachable: not proven."""
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    _git("checkout", "-qb", BRANCH, cwd=repo)
    (repo / "b.py").write_text("b = 1\n")
    _git("add", "b.py", cwd=repo)
    _git("commit", "-qm", "feature", cwd=repo)
    _git("push", "-q", "origin", BRANCH, cwd=repo)
    _git("checkout", "-q", "main", cwd=repo)
    # The merge lands on origin from another clone; this clone never fetches it.
    other = repo.parent / "other"
    _git("clone", "-q", str(repo.parent / "origin.git"), str(other), cwd=repo.parent)
    _git("merge", "-q", "--no-ff", "-m", "merge", f"origin/{BRANCH}", cwd=other)
    _git("push", "-q", "origin", "main", cwd=other)
    _run_recorded(home, f"cd ~/local-operator && git worktree add .worktrees/stale {BRANCH}")
    wt = repo / ".worktrees" / "stale"
    _git("remote", "set-url", "origin", str(repo.parent / "gone.git"), cwd=repo)
    merged, why = prov.git_merged(str(wt), BRANCH)
    assert not merged and "is not in origin/main" in why, why


# -- 1c / 2: EACCES on a same-uid process's cwd ----------------------------------


@pytest.fixture
def fake_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A ``/proc`` stand-in: ``self`` plus pids added by :func:`_add_pid`."""
    root = tmp_path / "proc"
    (root / "self").mkdir(parents=True)
    monkeypatch.setattr(prov, "_PROC_ROOT", str(root), raising=False)
    monkeypatch.setattr(prov, "_cwd_entries", _REAL_CWD_ENTRIES)
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    return root


_REAL_CWD_ENTRIES = prov._cwd_entries


def _add_pid(
    root: Path,
    pid: int,
    cwd: Path | None,
    *,
    fds: tuple[Path, ...] = (),
    maps: tuple[Path, ...] = (),
) -> None:
    d = root / str(pid)
    (d / "fd").mkdir(parents=True)
    (d / "cwd").symlink_to(cwd or "/")
    for n, target in enumerate(fds):
        (d / "fd" / str(n)).symlink_to(target)
    (d / "maps").write_text(
        "".join(f"7f00-7f01 r-xp 00000000 fd:01 {n} {p}\n" for n, p in enumerate(maps))
    )


def _deny_cwd(monkeypatch: pytest.MonkeyPatch, root: Path, *pids: int) -> None:
    """``os.readlink`` of these pids' ``cwd`` raises EACCES, as on node-2."""
    denied = {os.path.join(str(root), str(p), "cwd") for p in pids}
    real = os.readlink

    def readlink(path: Any, *a: Any, **kw: Any) -> str:
        if os.fspath(path) in denied:
            raise PermissionError(13, "Permission denied")
        return real(path, *a, **kw)

    monkeypatch.setattr(prov.os, "readlink", readlink)


def test_1c_unrelated_unreadable_cwd_no_longer_fails_the_scan(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _github_default_repos(home, repo)
    _add_pid(fake_proc, 4242, None)  # an Electron zygote: nothing readable
    _deny_cwd(monkeypatch, fake_proc, 4242)
    assert prov._cwd_entries(fresh=True) == ()
    assert prov.cwd_holders(os.path.realpath(wt), S) == ()
    note = prov.cwd_scan_note(str(wt))
    assert note == (
        "1 same-user process had an unreadable cwd (pid 4242); "
        "ignored: no evidence they hold the tree"
    )
    risk, reason = _rank(home, FIELD_1)
    assert risk is Risk.LOW, reason
    assert note in reason


@pytest.mark.parametrize("via", ["fd", "maps"])
def test_1c_unreadable_cwd_with_evidence_inside_is_a_foreign_holder(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch, via: str
) -> None:
    wt = _github_default_repos(home, repo)
    inside = wt / "a.py"
    _add_pid(
        fake_proc,
        4343,
        None,
        fds=(inside,) if via == "fd" else (),
        maps=(inside,) if via == "maps" else (),
    )
    _deny_cwd(monkeypatch, fake_proc, 4343)
    holders = prov.cwd_holders(os.path.realpath(wt), S)
    assert holders is not None and [h.pid for h in holders] == [4343]
    assert not holders[0].family
    assert prov.cwd_scan_note(str(wt)) == ""  # a holder, not an ignored pid
    risk, reason = _rank(home, FIELD_1)
    assert risk is Risk.HIGH, reason
    assert "pid 4343" in reason


def test_1c_unreadable_proc_still_fails_closed(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _github_default_repos(home, repo)
    real_scandir = os.scandir

    def scandir(path: Any = ".") -> Any:
        if not isinstance(path, int) and os.fspath(path) == str(fake_proc):
            raise PermissionError(13, "Permission denied")
        return real_scandir(path)

    monkeypatch.setattr(prov.os, "scandir", scandir)
    assert prov._cwd_entries(fresh=True) is None
    assert "/proc unreadable" in prov._cwd_inventory_error
    assert _rank(home, FIELD_1)[0] is Risk.HIGH


def test_1c_other_cwd_errors_still_fail_closed(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only EACCES is narrowed: an EIO-like failure keeps the whole scan unknown."""
    _github_default_repos(home, repo)
    _add_pid(fake_proc, 4545, None)
    target = os.path.join(str(fake_proc), "4545", "cwd")
    real = os.readlink

    def readlink(path: Any, *a: Any, **kw: Any) -> str:
        if os.fspath(path) == target:
            raise OSError(5, "Input/output error")
        return real(path, *a, **kw)

    monkeypatch.setattr(prov.os, "readlink", readlink)
    assert prov._cwd_entries(fresh=True) is None


def test_2_attr3880_detached_dirty_own_tree_on_a_linux_node_is_low(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Field case 2 end to end: drifted row, no companion, EACCES pids, dirty detached."""
    _run_recorded(home, "cd ~/local-operator && git worktree add --detach .worktrees/attr3880-ci")
    wt = repo / ".worktrees" / "attr3880-ci"
    (wt / "a.py").write_text("a = 'scratch'\n")
    _age_row_like_node2(home, wt)
    for pid in (3076, 3550, 3553):
        _add_pid(fake_proc, pid, None)
    _deny_cwd(monkeypatch, fake_proc, 3076, 3550, 3553)
    risk, reason = _rank(
        home, "cd ~/local-operator && git worktree remove --force .worktrees/attr3880-ci"
    )
    assert risk is Risk.LOW, reason
    assert "own-created detached worktree" in reason
    risk, reason = _rank(home, "rm -rf ~/local-operator/.worktrees/attr3880-ci")
    assert risk is Risk.LOW, reason


def test_1_github_default_repos_field_shape_end_to_end(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1a + 1b + 1c together, as on node-2: drifted row, gh down, EACCES pids."""
    monkeypatch.setattr(prov, "_gh_prs", lambda repo, branch: None)
    _github_default_repos(home, repo)
    for pid in range(12807, 12810):
        _add_pid(fake_proc, pid, None)
    _deny_cwd(monkeypatch, fake_proc, *range(12807, 12810))
    for command in (
        FIELD_1,
        "git worktree remove ~/local-operator/.worktrees/github-default-repos",
        "rm -rf ~/local-operator/.worktrees/github-default-repos",
    ):
        risk, reason = _rank(home, command)
        assert risk is Risk.LOW, (command, reason)


# -- 3: copy-overhaul, a merged tree "dirty" only by Vite's ignored SSR output ----


def test_3_ignored_dist_ssr_is_regenerable(home: Path, repo: Path) -> None:
    _merged_branch(repo, "feat/site-copy-overhaul")
    # Created by an unrelated session (58de2f8ff79b in the field), not running.
    _run_recorded(
        home,
        "cd ~/local-operator && git worktree add .worktrees/copy-overhaul feat/site-copy-overhaul",
        sid=S2,
    )
    wt = repo / ".worktrees" / "copy-overhaul"
    (wt / "dist-ssr").mkdir()
    (wt / "dist-ssr" / "entry-server.js").write_text("export {}\n")
    assert prov.worktree_state(str(wt)) == "clean"
    risk, reason = _rank(
        home, "git -C ~/local-operator worktree remove .worktrees/copy-overhaul", sid=S
    )
    assert risk is Risk.LOW, reason
    # ``--force`` and ``rm -rf`` of another session's tree still go to the judge.
    assert (
        _rank(home, "git -C ~/local-operator worktree remove --force .worktrees/copy-overhaul")[0]
        is Risk.HIGH
    )
    assert _rank(home, f"rm -rf {wt}")[0] is Risk.HIGH


def test_3_nested_repo_inside_dist_ssr_is_not_regenerable(home: Path, repo: Path) -> None:
    _run_recorded(home, "cd ~/local-operator && git worktree add -b x .worktrees/x", sid=S2)
    wt = repo / ".worktrees" / "x"
    nested = wt / "dist-ssr" / "keep"
    nested.mkdir(parents=True)
    _git("init", "-q", cwd=nested)
    assert prov.worktree_state(str(wt)) == "dirty"


# -- minors folded in: Q3 (spawn window), R2-m1 (rg/sessions), Q4 (dangling HEAD) --


def test_q3_unrecorded_harness_child_is_foreign_while_a_spawn_is_in_flight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    me, child = os.getpid(), 987_654
    real = str(tmp_path)
    monkeypatch.setattr(prov, "_cwd_entries", lambda *, fresh=False: ((child, real),))
    monkeypatch.setattr(prov, "_process_table", lambda: {child: (me, "bash"), me: (1, "lop")})
    monkeypatch.setattr(prov, "_session_pids", lambda sid: ({me: "this session"}, {}))
    holders = prov.cwd_holders(real, S)
    assert holders is not None and holders[0].family  # base attribution, nothing in flight
    with spawned.spawning():
        holders = prov.cwd_holders(real, S)
    assert holders is not None and not holders[0].family
    assert "unrecorded" in holders[0].owner
    assert spawned.pending() == 0


def test_q3_spawning_counter_is_released_when_the_spawn_fails() -> None:
    with pytest.raises(OSError):
        with spawned.spawning():
            assert spawned.pending() == 1
            raise OSError("no such shell")
    assert spawned.pending() == 0


@pytest.mark.asyncio
async def test_r2m1_sessions_launch_records_its_child(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.tools import builtin

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(builtin.guard_spawned, "record", lambda *a: seen.append(a))

    class _Proc:
        pid = 31337
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_exec(*_a: Any, **_kw: Any) -> _Proc:
        assert spawned.pending() == 1  # recorded inside the in-flight window
        return _Proc()

    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", fake_exec)
    assert await builtin._sessions_launch_as(S, ["--version"], {}, None) == (0, "")
    assert [(a[0], a[1]) for a in seen] == [(31337, S)]


@pytest.mark.asyncio
async def test_r2m1_grep_rg_records_its_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.tools import builtin

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(builtin.guard_spawned, "record", lambda *a: seen.append(a))
    real_exec = asyncio.create_subprocess_exec

    async def spy_exec(*a: Any, **kw: Any) -> Any:
        assert spawned.pending() == 1
        return await real_exec(*a, **kw)

    monkeypatch.setattr(builtin.asyncio, "create_subprocess_exec", spy_exec)
    (tmp_path / "f.txt").write_text("needle\n")
    if builtin.shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    scan = await builtin._ripgrep_scan(
        "needle", tmp_path, tmp_path, None, True, 0, None, session_id=S
    )
    assert scan is not None and scan.records
    assert len(seen) == 1 and seen[0][1] == S and isinstance(seen[0][0], int)


def test_q4_dangling_head_fact_is_in_the_agent_visible_reason(home: Path, repo: Path) -> None:
    _run_recorded(home, "cd ~/local-operator && git worktree add --detach .worktrees/dangle")
    wt = repo / ".worktrees" / "dangle"
    (wt / "e.py").write_text("e = 1\n")
    _git("add", "e.py", cwd=wt)
    _git("commit", "-qm", "only HEAD names this", cwd=wt)
    (wt / "a.py").write_text("a = 'scratch'\n")
    risk, reason = _rank(
        home, "cd ~/local-operator && git worktree remove --force .worktrees/dangle"
    )
    assert risk is Risk.HIGH  # never loosened: the commit would be lost
    assert "reachable from NO ref" in reason, reason
