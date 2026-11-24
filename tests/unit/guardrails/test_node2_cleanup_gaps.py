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
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _, **_kw: False)
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
    (seed / ".gitignore").write_text(".worktrees/\ndist-ssr/\nlocal-data/\nnode_modules\n")
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
_REAL_LIVE_CWD_INSIDE = prov._live_cwd_inside


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
    assert prov._cwd_entries(fresh=True) == ((4242, None),)
    assert prov.cwd_holders(os.path.realpath(wt), S) == ()
    note = prov.cwd_scan_note(str(wt))
    assert note == (
        "1 same-user process had an unreadable cwd (pid 4242); "
        "ignored: no evidence they hold the tree"
    )
    risk, reason = _rank(home, FIELD_1)
    assert risk is Risk.LOW, reason
    assert note in reason


def test_overlap_readable_foreign_holder_is_named_before_unrelated_hidden_pid(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#3987 m1: a hidden background pid must not conceal the definite holder."""
    wt = _github_default_repos(home, repo)
    _add_pid(fake_proc, 4242, None)
    _add_pid(fake_proc, 4343, wt)
    _deny_cwd(monkeypatch, fake_proc, 4242)
    monkeypatch.setattr(prov, "_process_table", lambda: {4343: (1, "foreign-shell")})
    phrase = prov.busy_phrase(str(wt), S)
    assert "pid 4343 foreign-shell" in phrase
    # Field report 2026-11-17: a hidden pid with no fd/maps evidence inside is
    # no holder on any judged path; the facts' scan note names it instead.
    assert "pid 4242" not in phrase
    assert "pid 4242" in prov.cwd_scan_note(str(wt))
    assert prov.worktree_state(str(wt), own=True, session_id=S) == "busy"
    risk, reason = _rank(home, FIELD_1)
    assert risk is Risk.HIGH, reason
    assert "pid 4343 foreign-shell" in reason
    # Only the judge-free interrupted-add settle keeps #3980's conservative
    # hidden-pid policy (``hidden_blocks``); judged removals need evidence.
    elsewhere = os.path.realpath(repo / "elsewhere")
    assert _REAL_LIVE_CWD_INSIDE(elsewhere) is False
    assert _REAL_LIVE_CWD_INSIDE(elsewhere, hidden_blocks=True) is None


def test_overlap_fresh_scan_finds_foreign_holder_after_cached_miss(
    home: Path, repo: Path, fake_proc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = _github_default_repos(home, repo)
    _add_pid(fake_proc, 4242, None)
    _deny_cwd(monkeypatch, fake_proc, 4242)
    assert prov.cwd_holders(os.path.realpath(wt), S) == ()
    _add_pid(fake_proc, 4343, wt)
    monkeypatch.setattr(prov, "_process_table", lambda: {4343: (1, "foreign-shell")})
    assert prov.worktree_state(str(wt), own=True, session_id=S) == "busy"
    risk, reason = _rank(home, FIELD_1)
    assert risk is Risk.HIGH and "pid 4343 foreign-shell" in reason


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


# -- 3: copy-overhaul, a merged tree with and without ignored SSR output ----


@pytest.mark.parametrize("ignored", [False, True], ids=["clean", "ignored-dist-ssr"])
def test_3_merged_plain_cleanup_sheds_build_output(
    home: Path, repo: Path, ignored: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The module stub takes no ``strict``; the finished-tree proof needs the
    # strict signature (no running owner: S2 is not running).
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid, **_: ())
    _merged_branch(repo, "feat/site-copy-overhaul")
    # Created by an unrelated session (58de2f8ff79b in the field), not running.
    _run_recorded(
        home,
        "cd ~/local-operator && git worktree add .worktrees/copy-overhaul feat/site-copy-overhaul",
        sid=S2,
    )
    wt = repo / ".worktrees" / "copy-overhaul"
    if ignored:
        (wt / "dist-ssr").mkdir()
        (wt / "dist-ssr" / "entry-server.js").write_text("export {}\n")
    # 2026-11-24 (fleet census, operator directive): a FINISHED tree (merged
    # on origin, clean, no open PR or running owner) sheds regenerable-named
    # build output whoever made it, so plain removal is LOW either way.
    assert prov.worktree_state(str(wt)) == "clean"
    command = "git -C ~/local-operator worktree remove .worktrees/copy-overhaul"
    verdict = classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=S)
    assert verdict.risk is Risk.LOW, verdict.reasons
    # ``--force`` and ``rm -rf`` over another session's ignored payload still
    # go to the judge (the name rule is plain-only). A clean, finished tree's
    # ``--force``/``rm -rf`` is LOW by operator decision a-891c.
    if ignored:
        forced = "git -C ~/local-operator worktree remove --force .worktrees/copy-overhaul"
        assert _rank(home, forced)[0] is Risk.HIGH
        assert _rank(home, f"rm -rf {wt}")[0] is Risk.HIGH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "exact",
        "brace",
        "ancestor",
        "glob",
        "remote",
        "pending",
        "dirty",
        "busy",
        "compound",
        "compound-unapproved",
        "owned",
        "owned-no-ask",
        "keep",
        "backup",
        "conditional",
        "field-shape",
        "unverified",
        "unknown-forge",
        "git-fallback",
        "open-forge",
        "owner",
        "unreadable-owner",
        "late-holder",
        "late-forge",
        "late-revision",
        "late-pending",
        "label-keep",
        "label-wrong-verb",
        "label-condition",
        "typed",
        "field-brace",
        "mixed-question",
        "field-conditional",
        "field-after",
        "field-reversed",
        "newer-keep-yes",
        "newer-backup-yes",
        "ambiguous-safe",
        "early-holder",
        "early-revision",
        "hidden-holder",
        "fresh-holder",
        "late-fresh-holder",
        "reworded",
        "excluding",
        "narrowing-pick",
    ],
)
async def test_ignored_cleanup_approval_at_real_checkpoint(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Real Git inventory and ask-store rows; the destructive command is never run."""
    import json
    import time

    from local_operator.guardrails.check import (
        CallContext,
        GuardSettings,
        check_tool_call,
    )
    from local_operator.guardrails.preview import preview_tool_call

    monkeypatch.setattr(prov, "_running_owners", lambda real, sid, **kwargs: ())
    monkeypatch.setattr(prov, "cwd_holders", lambda *a, **kw: ())
    if mode in ("hidden-holder", "fresh-holder"):
        monkeypatch.setattr(
            prov, "cwd_holders", lambda *a, **kw: None if mode == "hidden-holder" else (object(),)
        )
    wt = repo / ".worktrees" / "consent"
    if mode.startswith("owned"):
        _run_recorded(home, f"git -C {repo} worktree add -qb consent {wt}")
    else:
        _git("worktree", "add", "-qb", "consent", ".worktrees/consent", cwd=repo)
    (wt / "local-data").mkdir()
    (wt / "local-data" / "payload").write_text("preserve")
    if mode == "dirty":
        (wt / "untracked").write_text("work")
    if mode.startswith("compound") or mode in ("field-brace", "mixed-question"):
        _git("worktree", "add", "-qb", "other", ".worktrees/other", cwd=repo)
        (wt.parent / "other" / "local-data").mkdir()
        (wt.parent / "other" / "local-data" / "payload").write_text("preserve")
    if mode == "busy":
        monkeypatch.setattr(prov, "_live_cwd_inside", lambda *a, **kw: True)
    if mode in ("unknown-forge", "git-fallback", "open-forge", "owner"):
        activity = prov.worktree_activity

        def override_activity(path, sid=""):
            from dataclasses import replace

            result = activity(path, sid)
            if mode == "unknown-forge":
                return replace(result, pr_state="unknown")
            if mode == "git-fallback":
                return replace(result, pr_state="none", git_merged="old local refs")
            if mode == "open-forge":
                return replace(result, pr_state="open", pr_number=1)
            return replace(result, owners_running=("other-session",))

        monkeypatch.setattr(prov, "worktree_activity", override_activity)
    if mode == "unreadable-owner":

        def unreadable(real, sid, **kwargs):
            if kwargs.get("strict"):
                raise OSError("registry unreadable")
            return ()

        monkeypatch.setattr(prov, "_running_owners", unreadable)
    named = str(wt)
    if mode in ("brace", "compound", "field-brace"):
        named = str(wt.parent / "{consent,other}")
    elif mode == "ancestor":
        named = str(wt.parent)
    elif mode == "glob":
        named = str(wt.parent / "consent*")
    now = int(time.time() * 1000)
    rows = [
        {
            "v": 1,
            "kind": "queued",
            "ask_id": "a-cafe",
            "at": now - 2000,
            "expires_at": now + 60000,
            "timeout_s": 60,
            "questions": [
                {
                    "id": "q",
                    "question": f"Delete {named}?",
                    "options": [{"label": "Approve"}, {"label": "Keep"}],
                    "multi": False,
                    "question_id": "",
                }
            ],
        }
    ]
    question = rows[0]["questions"][0]
    if mode == "ambiguous-safe":
        question["question"] = f"Delete {named} when safe?"
    if mode in ("keep", "backup", "conditional"):
        question["question"] = {
            "keep": f"Keep {named} instead of deleting it?",
            "backup": f"Back up {named} before deleting?",
            "conditional": f"Delete {named} after backing it up?",
        }[mode]
        question["options"] = [{"label": "Yes"}, {"label": "No"}]
    if mode in (
        "field-shape",
        "field-brace",
        "mixed-question",
        "field-conditional",
        "field-after",
        "field-reversed",
    ):
        question["question"] = (
            f"Approve removing 33 clean merged/closed worktrees with ignored files "
            f"(re-verified for 0 live holders before removal): {named}?"
        )
        if mode == "field-conditional":
            question["question"] = f"Approve removing {named} if safe?"
        elif mode == "field-after":
            question["question"] = f"Approve removing {named} after backing up?"
        elif mode == "field-reversed":
            question["question"] = f"Approve removing the backup of {named}?"
        question["options"] = [
            {"label": "Approve removing all 33 clean merged/closed worktrees (~17.3 GiB)"},
            {"label": "Keep everything"},
        ]
        # Independent questions must never borrow each other's deletion picks.
        rows[0]["questions"].extend(
            [
                {
                    "id": "q2",
                    "question": f"Keep {wt.parent / 'other'}?",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                    "multi": False,
                },
                {
                    "id": "q3",
                    "question": "Approve deleting old clones?",
                    "options": [
                        {"label": "Approve deleting all old clones"},
                        {"label": "Keep everything"},
                    ],
                    "multi": False,
                },
            ]
        )
    # Review round 2: approval is a rule on the pick, not a vocabulary tuned to
    # one field ask. An unqualified ask in other words must pass; a list that
    # spares a path (the field Q3 shape, "excluding ~/x") must fail closed.
    if mode == "narrowing-pick":
        # QA Q-R3-1: the pick narrows the field question to a glob of
        # ignored payload; it must not approve removing the whole trees.
        question["question"] = (
            f"Approve removing 33 clean merged/closed worktrees with ignored files: {named}?"
        )
        question["options"] = [
            {"label": f"Approve removing only {wt.parent}/*/node_modules"},
            {"label": "Keep everything"},
        ]
    if mode in ("reworded", "excluding"):
        question["question"] = (
            f"OK to clear out these finished worktrees now: {named}?"
            if mode == "reworded"
            else f"Approve deleting old trees, excluding {wt.parent / 'other'}: {named}?"
        )
        question["options"] = [{"label": "Remove them"}, {"label": "Leave them"}]
    if mode in ("label-keep", "label-wrong-verb", "label-condition", "typed"):
        question["question"] = (
            f"Keep {named}?"
            if mode == "label-keep"
            else f"Back up {named}?" if mode == "label-wrong-verb" else f"Delete {named}?"
        )
        question["options"] = [
            {"label": "Delete after it is safe" if mode == "label-condition" else "Delete"},
            {"label": "Keep"},
        ]
    if mode != "pending":
        picked = (
            "Yes"
            if mode in ("keep", "backup", "conditional", "typed")
            else (
                "Delete"
                if mode in ("label-keep", "label-wrong-verb")
                else (
                    "Delete after it is safe"
                    if mode == "label-condition"
                    else (
                        "Remove them"
                        if mode in ("reworded", "excluding")
                        else (
                            f"Approve removing only {wt.parent}/*/node_modules"
                            if mode == "narrowing-pick"
                            else (
                                "Approve removing all 33 clean merged/closed worktrees (~17.3 GiB)"
                                if mode.startswith("field-") or mode == "mixed-question"
                                else "Approve"
                            )
                        )
                    )
                )
            )
        )
        answer = {"q": [picked]}
        if mode.startswith("field-") or mode == "mixed-question":
            answer.update({"q2": ["Yes"], "q3": ["Approve deleting all old clones"]})
        rows.append(
            {
                "v": 1,
                "kind": "answered",
                "ask_id": "a-cafe",
                "at": now - 1000,
                "by": {
                    "surface": (
                        "remote"
                        if mode == "remote"
                        else "unverified" if mode == "unverified" else "desktop"
                    )
                },
                "answers": answer,
            }
        )
    if mode in ("newer-keep-yes", "newer-backup-yes"):
        contrary = (
            f"Keep {wt} instead of deleting it?"
            if mode == "newer-keep-yes"
            else f"Back up {wt} before deleting?"
        )
        rows.extend(
            [
                {
                    "v": 1,
                    "kind": "queued",
                    "ask_id": "a-beef",
                    "at": now - 500,
                    "expires_at": now + 60000,
                    "questions": [
                        {
                            "id": "q",
                            "question": contrary,
                            "options": [{"label": "Yes"}, {"label": "No"}],
                            "multi": False,
                        }
                    ],
                },
                {
                    "v": 1,
                    "kind": "answered",
                    "ask_id": "a-beef",
                    "at": now - 100,
                    "by": {"surface": "desktop"},
                    "answers": {"q": ["Yes"]},
                },
            ]
        )
    path = home / ".local-operator" / "sessions" / S / "asks.jsonl"
    path.write_text(
        "" if mode == "owned-no-ask" else "".join(json.dumps(row) + "\n" for row in rows)
    )
    command = f"git -C {repo} worktree remove -- {wt}"
    if mode.startswith("compound") or mode in ("field-brace", "mixed-question"):
        command += f" && git -C {repo} worktree remove -- {wt.parent / 'other'}"
    ctx = CallContext(cwd=str(home), env=_env(home), scratchpad=None, session_id=S)
    settings = GuardSettings()
    calls = []
    if mode.startswith("late-"):
        from local_operator.guardrails.check import DeleteCase

        monkeypatch.setattr(DeleteCase, "evidence_basis", lambda self, *args: "")

    async def judge(system: str, prompt: str) -> str:
        calls.append(prompt)
        if mode == "late-holder":
            monkeypatch.setattr(prov, "_live_cwd_inside", lambda *a, **kw: True)
        elif mode == "late-fresh-holder":
            monkeypatch.setattr(prov, "cwd_holders", lambda *a, **kw: (object(),))
        elif mode == "late-forge":
            monkeypatch.setattr(
                prov,
                "worktree_activity",
                lambda *a, **kw: prov.WorktreeActivity("consent", "unknown"),
            )
        elif mode == "late-revision":
            with path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "v": 1,
                            "kind": "revised",
                            "ask_id": "a-cafe",
                            "at": now + 100,
                            "by": {"surface": "desktop"},
                            "answers": {"q": ["Keep"]},
                        }
                    )
                    + "\n"
                )
        elif mode == "late-pending":
            with path.open("a") as fh:
                fh.write(
                    json.dumps(
                        {
                            "v": 1,
                            "kind": "queued",
                            "ask_id": "a-beef",
                            "at": now + 100,
                            "expires_at": now + 60000,
                            "questions": rows[0]["questions"],
                        }
                    )
                    + "\n"
                )
        return "ALLOW ask=a-cafe"

    if mode.startswith("early-"):
        from local_operator.guardrails import check as check_module

        real_build = check_module.build_delete_case

        def build_then_mutate(verdict, context):
            case = real_build(verdict, context)
            if mode == "early-holder":
                monkeypatch.setattr(prov, "_live_cwd_inside", lambda *a, **kw: True)
            else:
                with path.open("a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "v": 1,
                                "kind": "revised",
                                "ask_id": "a-cafe",
                                "at": now + 100,
                                "by": {"surface": "desktop"},
                                "answers": {"q": ["Keep"]},
                            }
                        )
                        + "\n"
                    )
            return case

        monkeypatch.setattr(check_module, "build_delete_case", build_then_mutate)
    preview = preview_tool_call("bash", {"command": command}, ctx, settings=settings)
    result = await check_tool_call(
        "bash", {"command": command}, ctx, settings=settings, judge=judge
    )
    if mode.startswith("early-"):
        assert preview.builtin_effect == "pass", preview.render()
        assert result is not None and result.blocked and not calls
    elif mode in ("exact", "brace", "compound", "owned", "field-shape", "field-brace", "reworded"):
        assert preview.builtin_effect == "pass", preview.render()
        assert result is not None and not result.blocked, result and result.text
        assert not calls
    elif mode.startswith("late-"):
        assert preview.builtin_effect == "judge", preview.render()
        assert result is not None and result.blocked and calls
    else:
        assert preview.builtin_effect in ("hold-cleanup", "refuse-critical"), preview.render()
        assert result is not None and result.blocked and not calls
    assert (wt / "local-data" / "payload").read_text() == "preserve"


def test_3_nested_repo_inside_dist_ssr_is_not_regenerable(home: Path, repo: Path) -> None:
    _run_recorded(home, "cd ~/local-operator && git worktree add -b x .worktrees/x", sid=S2)
    wt = repo / ".worktrees" / "x"
    nested = wt / "local-data" / "keep"
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
    # ``child`` is synthetic yet models a LIVE process: without this, the
    # gone-holder re-check (``_still_exists``) would rightly drop it.
    monkeypatch.setattr(prov, "_still_exists", lambda pid: True)
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


# -- PR #4493: cache cleanup inside a worktree, spelled find | xargs rm --------------

#: The field command (session ed88c11e4952, backlog-gates, 2026-11-23), refused
#: on guard 0.112.5 as an opaque delete of the whole worktree.
FIELD_XARGS = "find . -maxdepth 4 -name '__pycache__' -type d -prune -print0 | xargs -0 rm -rf"


@pytest.mark.parametrize(
    ("command", "risk"),
    [
        (FIELD_XARGS, Risk.LOW),
        # The spelling S2-3 already ranked LOW: both must agree.
        ("find . -name __pycache__ -type d -prune -exec rm -rf {} +", Risk.LOW),
        # No selecting predicate: every entry goes, so it removes the tree.
        ("find . -print0 | xargs -0 rm -rf", Risk.HIGH),
        # A filter between the stages can rewrite paths: stays opaque.
        ("find . -name __pycache__ | sed s/x/y/ | xargs rm -rf", Risk.HIGH),
        # The find stage itself runs a program: its output is not find's list.
        ("find . -name a -exec echo {} + | xargs rm -rf", Risk.HIGH),
        # Review SEC-R2-B1 (PR #4493): every way rm's list can differ from the walk.
        ("find . -name __pycache__ -print0 | xargs -0 -a list rm -rf", Risk.HIGH),
        ("find . -name __pycache__ -print0 | xargs -0 --arg-file=list rm -rf", Risk.HIGH),
        ("find . -name __pycache__ -print0 | xargs -0 rm -rf < list", Risk.HIGH),
        ("find . -name x -printf '/etc\\0' | xargs -0 rm -rf", Risk.HIGH),
        ("find . -name x -fprint /dev/stdout | xargs rm -rf", Risk.HIGH),
        ("find . -name x -ls | xargs rm -rf", Risk.HIGH),
        ("find . -name x -print0 | xargs -0 -I@ rm -rf ../precious/@", Risk.HIGH),
        ("find . -name x -print0 | xargs -0 -i rm -rf {}", Risk.HIGH),
        ("find -H . -name x -print0 | xargs -0 rm -rf", Risk.HIGH),
        ("find -files0-from list -name x -print0 | xargs -0 rm -rf", Risk.HIGH),
        ("find . -name x -print0 | xargs -0 rm -rf ../precious", Risk.HIGH),
        # Newline-separated output (SEC-R2-m2): only -print0 | xargs -0 is find's list.
        ("find . -name __pycache__ -print | xargs rm -rf", Risk.HIGH),
        # Review SEC-R3-B1: a grouped stage feeds rm more than find's list.
        (
            "(printf '%s\\0' ../precious; find . -name __pycache__ -print0) | xargs -0 rm -rf",
            Risk.HIGH,
        ),
        ("{ true && find . -name __pycache__ -print0; } | xargs -0 rm -rf", Risk.HIGH),
    ],
    ids=[
        "field-xargs",
        "find-exec",
        "no-predicate",
        "filtered",
        "find-runs-exec",
        "xargs-a",
        "xargs-arg-file",
        "stdin-redirect",
        "printf",
        "fprint",
        "ls",
        "xargs-I",
        "xargs-i",
        "find-H",
        "files0-from",
        "extra-operand",
        "newline-print",
        "grouped-subshell",
        "grouped-brace",
    ],
)
def test_find_fed_xargs_delete_ranks_like_find(
    home: Path, repo: Path, command: str, risk: Risk
) -> None:
    _git("worktree", "add", "-q", "-b", "feat/caches", ".worktrees/backlog-gates", cwd=repo)
    wt = repo / ".worktrees" / "backlog-gates"
    (wt / "__pycache__").mkdir()
    (wt / "__pycache__" / "a.pyc").write_text("x")
    verdict = classify_bash(
        f"cd {wt} && {command}", cwd=str(home), env=_env(home), scratchpad=None, session_id=S
    )
    assert verdict.risk is risk, verdict.reasons
    if risk is Risk.LOW:
        assert all(not t.target.opaque for t in verdict.targets), verdict.reasons


def test_find_fed_xargs_never_lowers_the_state_root(home: Path) -> None:
    """The selecting-find rule never covers a state root, whichever spelling."""
    state = home / ".local-operator"
    command = f"find {state} -name '*.json' -print0 | xargs -0 rm -f"
    verdict = classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=S)
    assert verdict.risk is Risk.CRITICAL, verdict.reasons


# -- PR #4493 QA Q-R2-1: an exact approval may settle an unattributed tree ----------


def test_unattributed_tree_has_no_recorded_owner_under_exact_approval(
    home: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tree made in a terminal has no ledger row. The approval waiver reads
    that as "nobody on record" (the live-cwd registry scan still runs); every
    other strict caller still raises, and an UNREADABLE ledger still raises."""
    monkeypatch.undo()  # the real _running_owners, not the module fixture's stub
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(home / ".local-operator"))
    (home / ".local-operator" / "run" / "mobile").mkdir(parents=True, exist_ok=True)
    _git("worktree", "add", "-q", "-b", "feat/terminal", ".worktrees/terminal", cwd=repo)
    real = os.path.realpath(repo / ".worktrees" / "terminal")
    # A ledger exists (another tree's row) but holds nothing for this one.
    _run_recorded(home, "mkdir -p ~/rig-elsewhere")
    with pytest.raises(prov.UnattributedWorktree):
        prov._running_owners(real, S, strict=True)
    assert prov._running_owners(real, S, strict=True, allow_unattributed=True) == ()
    # Unreadable ledger: not "unattributed", so the waiver still fails closed.
    ledger = prov.ledger_path()
    ledger.write_bytes(b"not a sqlite database")
    with pytest.raises((OSError, sqlite3.Error)) as raised:
        prov._running_owners(real, S, strict=True, allow_unattributed=True)
    assert not isinstance(raised.value, prov.UnattributedWorktree)
