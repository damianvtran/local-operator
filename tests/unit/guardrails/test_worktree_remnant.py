"""A failed ``git worktree remove`` remnant ranks as an ordinary directory.

Field case 2026-11-24 (disk recovery, guard 0.112.8): Finder wrote a
``.DS_Store`` while Git was deleting a clean, merged tree, so ``git worktree
remove`` unregistered the tree, deleted its admin directory and then failed
with "Directory not empty". The leftover folder still held its ``.git`` pointer
file and ranked as "a git repository root" (sent to the judge), while an
identical leftover without the pointer was an ordinary path. See
``classify._removed_worktree_remnant``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from local_operator.guardrails.classify import _removed_worktree_remnant

ENV = {**os.environ, "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=ENV)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "i",
        cwd=root,
    )
    return root


def _tree(repo: Path, name: str) -> Path:
    wt = repo / ".worktrees" / name
    _git("worktree", "add", "-q", "-b", f"fix/{name}", str(wt), cwd=repo)
    (wt / "kept.txt").write_text("tracked leftover\n")
    return wt


def test_a_live_linked_worktree_is_not_a_remnant(repo: Path) -> None:
    assert not _removed_worktree_remnant(str(_tree(repo, "live")))


def test_admin_dir_gone_and_unregistered_is_a_remnant(repo: Path) -> None:
    wt = _tree(repo, "half-removed")
    # What a failed ``git worktree remove`` leaves: admin dir gone, folder kept.
    shutil.rmtree(repo / ".git" / "worktrees" / "half-removed")
    _git("worktree", "prune", cwd=repo)
    assert _removed_worktree_remnant(str(wt))


def test_a_moved_tree_whose_admin_name_differs_is_not_a_remnant(repo: Path) -> None:
    wt = _tree(repo, "orig")
    moved = repo / ".worktrees" / "moved"
    wt.rename(moved)  # manual move: the pointer still names admin ``orig``
    shutil.rmtree(repo / ".git" / "worktrees" / "orig")
    assert not _removed_worktree_remnant(str(moved))


@pytest.mark.parametrize("shape", ["git-dir", "two-lines", "relative-outside", "admin-present"])
def test_any_other_shape_keeps_the_repository_rank(repo: Path, shape: str) -> None:
    wt = _tree(repo, f"shape-{shape}")
    marker = wt / ".git"
    pointer = marker.read_text()
    (
        shutil.rmtree(repo / ".git" / "worktrees" / f"shape-{shape}")
        if shape != "admin-present"
        else None
    )
    if shape == "git-dir":
        marker.unlink()
        marker.mkdir()
    elif shape == "two-lines":
        marker.write_text(pointer + "extra\n")
    elif shape == "relative-outside":
        marker.write_text(f"gitdir: ../../elsewhere/.git/worktrees/shape-{shape}\n")
    assert not _removed_worktree_remnant(str(wt))


def test_rm_rf_rank_follows_the_remnant_rule(repo: Path) -> None:
    """Classification level: a live linked tree is still held (HIGH, judged),
    the remnant is an ordinary path and is never 'a git repository root'."""
    from local_operator.guardrails.classify import Risk, classify_bash

    live = _tree(repo, "still-live")
    gone = _tree(repo, "remnant")
    shutil.rmtree(repo / ".git" / "worktrees" / "remnant")
    env = {"HOME": str(repo.parent)}

    def rank(path: Path):
        return classify_bash(
            f"rm -rf {path}", cwd=str(repo), env=env, scratchpad=None, session_id="aaaaaaaaaaa1"
        )

    assert rank(live).risk is Risk.HIGH
    remnant = rank(gone)
    reasons = " ".join([t.reason for t in remnant.targets] + list(remnant.reasons))
    assert "a git repository root" not in reasons and "linked worktree" not in reasons, reasons
