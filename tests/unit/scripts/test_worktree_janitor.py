"""``scripts/worktree_janitor.py``: nothing in flight is ever offered for removal.

The script deletes directories under ``--apply``, so the classification matrix
is pinned against real (synthetic) git repositories: a bare ``origin`` plus a
clone under ``tmp_path``. ``gh`` is never called - ``_pr_records`` is
monkeypatched to canned PR records. Git runs with the same keychain-safe
environment as ``test_rebase_proof.py`` (``GIT_CONFIG_SYSTEM=/dev/null``).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "worktree_janitor.py"
_spec = importlib.util.spec_from_file_location("worktree_janitor", SCRIPT)
assert _spec and _spec.loader
janitor = importlib.util.module_from_spec(_spec)
sys.modules["worktree_janitor"] = janitor
_spec.loader.exec_module(janitor)


@pytest.fixture(autouse=True)
def _git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith(("GIT_", "CMUX_"))]:
        monkeypatch.delenv(key)
    for key, value in dict(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    ).items():
        monkeypatch.setenv(key, value)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def commit(cwd: Path, name: str) -> str:
    (cwd / name).write_text(name)
    git(cwd, "add", name)
    git(cwd, "commit", "-qm", name)
    return git(cwd, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A clone of a bare origin with one commit on ``main`` pushed."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    primary = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", str(origin), str(primary)], check=True)
    git(primary, "checkout", "-q", "-b", "main")
    commit(primary, "base")
    git(primary, "push", "-q", "origin", "main")
    return primary


def add(repo: Path, name: str, *args: str, detach: str = "") -> Path:
    path = repo / ".worktrees" / name
    if detach:  # `--detach <path> <commit>`: flags precede the path, the commit follows
        git(repo, "worktree", "add", "-q", "--detach", str(path), detach)
    else:
        git(repo, "worktree", "add", "-q", *args, str(path))
    return path


def verdict(repo: Path, path: Path, **kwargs) -> tuple[str, str]:
    trees = {t["worktree"]: t for t in janitor._worktrees(str(repo))}
    tree = trees[str(path.resolve())] if str(path.resolve()) in trees else trees[str(path)]
    kwargs.setdefault("min_age_hours", 0.0)
    return janitor.classify(tree, str(repo.resolve()), kwargs.pop("use_gh", False), **kwargs)


def no_prs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(janitor, "_pr_records", lambda path, branch: [])


def test_merged_fast_forward_is_removed(repo: Path) -> None:
    wt = add(repo, "ff", "-b", "ff")  # at origin/main, aged by min_age_hours=0
    assert verdict(repo, wt) == ("REMOVE", "HEAD is on origin/main")


def test_fresh_lane_with_no_commits_is_kept_by_default_age_gate(repo: Path) -> None:
    wt = add(repo, "fresh", "-b", "fresh")
    state, reason = verdict(repo, wt, min_age_hours=janitor.DEFAULT_MIN_AGE_HOURS)
    assert state == "KEEP" and "fresh lane" in reason


def test_old_lane_at_origin_main_passes_the_age_gate(repo: Path) -> None:
    wt = add(repo, "old", "-b", "old")
    gitdir = Path(git(wt, "rev-parse", "--absolute-git-dir"))
    old = 1_000_000_000
    for name in ("HEAD", "logs/HEAD"):
        if (gitdir / name).exists():
            os.utime(gitdir / name, (old, old))
    assert verdict(repo, wt, min_age_hours=janitor.DEFAULT_MIN_AGE_HOURS)[0] == "REMOVE"


def test_detached_at_open_pr_head_is_kept(repo: Path) -> None:
    """Contained in a remote ref (the PR branch) but not merged to origin/main."""
    feature = add(repo, "feature", "-b", "feature")
    sha = commit(feature, "work")
    git(feature, "push", "-q", "origin", "feature")
    qa = add(repo, "qa", detach=sha)
    assert git(qa, "branch", "-r", "--contains", "HEAD")  # the old rule would have fired
    state, reason = verdict(repo, qa)
    assert state == "KEEP" and "no merged/closed PR" in reason


def test_squash_merged_with_deleted_remote_branch_is_removed(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = add(repo, "squash", "-b", "squash")
    sha = commit(wt, "work")  # never pushed: the remote branch is gone after merge
    monkeypatch.setattr(janitor, "_pr_records", lambda p, b: [("MERGED", sha)])
    state, reason = verdict(repo, wt, use_gh=True)
    assert state == "REMOVE" and "at this head" in reason


def test_reused_branch_name_with_new_commits_is_kept(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wt = add(repo, "release-next", "-b", "release-next")
    old = commit(wt, "old")
    new = commit(wt, "new")  # commit after the old PR's head, not on any remote
    monkeypatch.setattr(janitor, "_pr_records", lambda p, b: [("MERGED", old)])
    state, reason = verdict(repo, wt, use_gh=True)
    assert state == "KEEP" and "commit(s) on no remote" in reason and new != old


def test_open_pr_is_kept(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wt = add(repo, "open", "-b", "open")
    sha = commit(wt, "work")
    monkeypatch.setattr(janitor, "_pr_records", lambda p, b: [("OPEN", sha), ("MERGED", "x")])
    assert verdict(repo, wt, use_gh=True) == ("KEEP", "PR still open")


def test_dirty_unpushed_and_locked_are_kept(repo: Path) -> None:
    dirty = add(repo, "dirty", "-b", "dirty")
    (dirty / "scratch").write_text("x")
    unpushed = add(repo, "unpushed", "-b", "unpushed")
    commit(unpushed, "wip")
    locked = add(repo, "locked", "-b", "locked")
    git(repo, "worktree", "lock", "--reason", "in use", str(locked))
    assert verdict(repo, dirty)[1].startswith("dirty")
    assert verdict(repo, unpushed)[1].endswith("commit(s) on no remote")
    assert verdict(repo, locked) == ("KEEP", "locked")


def test_initializing_lock_and_missing_dir_are_reported_stale(repo: Path) -> None:
    stuck = add(repo, "stuck", "-b", "stuck")
    git(repo, "worktree", "lock", "--reason", "initializing", str(stuck))
    gone = add(repo, "gone", "-b", "gone")
    subprocess.run(["rm", "-rf", str(gone)], check=True)
    state, reason = verdict(repo, stuck)
    assert state == "STALE" and f"git worktree remove -f -f {stuck.resolve()}" in reason
    state, reason = verdict(repo, gone)
    assert state == "STALE" and "git worktree prune" in reason


def test_apply_removes_only_remove_rows_and_fails_without_force_on_dirtied_tree(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    merged = add(repo, "merged", "-b", "merged")
    keep = add(repo, "keep", "-b", "keep")
    commit(keep, "wip")
    assert janitor.main(["--repo", str(repo), "--no-gh", "--min-age-hours", "0", "--apply"]) == 0
    assert not merged.exists() and keep.exists()

    # TOCTOU: the tree turns dirty after classification; git must refuse (no --force).
    racing = add(repo, "racing", "-b", "racing")
    real_classify = janitor.classify

    def dirtying(*a, **k):
        result = real_classify(*a, **k)
        if a[0]["worktree"] == str(racing.resolve()):
            (racing / "late").write_text("x")
        return result

    monkeypatch.setattr(janitor, "classify", dirtying)
    code = janitor.main(["--repo", str(repo), "--no-gh", "--min-age-hours", "0", "--apply"])
    captured = capsys.readouterr()
    assert code == 1 and (racing / "late").exists()
    assert "remove failed:" in captured.err and "contains modified or untracked" in captured.err
