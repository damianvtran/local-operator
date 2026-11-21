"""Snapshot → bundle → restore reproduces a worktree on another device (design §5, §14).

Real git in ``tmp_path``. The source and destination homes differ, so the
home-relative remap is exercised rather than assumed. Every observable the
operator would check after a move — HEAD, staged, unstaged, untracked, exec
bits, symlinks, branch + upstream, ``git worktree list`` — must match.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.network import workset_transfer as wt
from tests.unit.network.workset_git_fixtures import (  # noqa: F401
    Rig,
    copy_artifact,
    facts,
    git,
    git_env,
    make_rig,
)

# Every test here runs real git; none may read system/global config or prompt
# (``git_env`` is imported for this mark, hence the F401 noqa on the import).
pytestmark = pytest.mark.usefixtures("git_env")

SID = "abcdef0123456789"


def move(
    rig: Rig, tmp_path: Path, *, excludes: tuple[str, ...] = ()
) -> tuple[wt.SnapshotRefs, wt.RestoreReport]:
    snap = wt.snapshot(rig.worktree, SID, 0, excludes)
    upstream_tip = git(rig.worktree, "rev-parse", "@{u}")
    have = wt.compute_have(rig.dst_primary, [snap.head, upstream_tip], remote="origin")
    artifact = wt.stage_bundle(rig.src_primary, snap.refs, have, tmp_path / "out" / SID)
    incoming = copy_artifact(artifact.path, tmp_path / "in" / SID)
    plan = wt.RestorePlan(
        session_id=SID,
        n=0,
        bundle_digest=artifact.digest,
        path=wt.home_relative(rig.worktree, home=rig.src_home) or "",
        primary=wt.home_relative(rig.src_primary, home=rig.src_home) or "",
        remote_url=git(rig.src_primary, "remote", "get-url", "origin"),
        head=snap.head,
        index=snap.index,
        worktree=snap.worktree,
        branch=snap.branch,
        upstream_merge="refs/heads/feat",
        src_device="laptop",
    )
    return snap, wt.restore(incoming, plan, home=rig.dst_home)


def test_round_trip_reproduces_every_worktree_observable(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    before = facts(rig.worktree)
    snap, report = move(rig, tmp_path)

    assert report.state == "restored", report
    assert report.mode == "new"
    assert report.dest_path == "~/proj/.worktrees/feat"
    dest = rig.dst_home / "proj" / ".worktrees" / "feat"
    assert facts(dest) == before
    # The concrete states, not just equality with a possibly-empty source.
    assert before["cached"].splitlines() == [
        "D\tdel_staged.txt",
        "M\tmod_staged.txt",
        "A\tnew_staged.txt",
    ]
    assert before["unstaged"].splitlines() == ["D\tdel_unstaged.txt", "M\tmod_unstaged.txt"]
    assert sorted(before["untracked"].splitlines()) == ["newlink", "tool.sh", "untracked.txt"]
    # The unpushed commit arrived and HEAD is it.
    assert git(dest, "log", "-1", "--format=%s") == "feat unpushed"
    # Contents, exec bits and symlinks as on the source.
    assert (dest / "mod_unstaged.txt").read_text() == "two\n"
    assert not (dest / "del_unstaged.txt").exists()
    assert os.access(dest / "run.sh", os.X_OK) and os.access(dest / "tool.sh", os.X_OK)
    assert not os.access(dest / "keep.txt", os.X_OK)
    assert (dest / "link").is_symlink() and os.readlink(dest / "link") == "keep.txt"
    assert (dest / "newlink").is_symlink() and os.readlink(dest / "newlink") == "keep.txt"
    # Ignored dependency trees are left behind (recipes, §9), never carried.
    assert not (dest / "node_modules").exists()
    # Branch + upstream configured; @{u} resolves after the destination's fetch.
    assert git(dest, "symbolic-ref", "--short", "HEAD") == "feat"
    assert git(rig.dst_primary, "config", "branch.feat.remote") == "origin"
    assert git(rig.dst_primary, "config", "branch.feat.merge") == "refs/heads/feat"
    assert git(dest, "rev-parse", "@{u}") == git(rig.worktree, "rev-parse", "@{u}")
    # Registered in the destination primary, at its final (not staging) path.
    listed = git(rig.dst_primary, "worktree", "list", "--porcelain")
    assert f"worktree {os.path.realpath(dest)}" in listed.splitlines()
    assert ".lop-restore-" not in listed
    assert not list(dest.parent.glob(".lop-restore-*"))
    assert git(dest, "status", "--porcelain") == git(rig.worktree, "status", "--porcelain")


def test_round_trip_withholds_excluded_secret_and_keeps_head_version(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    (rig.worktree / ".env.local").write_text("TOKEN=not-a-real-value\n")
    # A tracked file the manifest excluded, edited AND staged: neither edit may ride.
    (rig.worktree / "keep.txt").write_text("staged edit\n")
    git(rig.worktree, "add", "keep.txt")
    (rig.worktree / "keep.txt").write_text("unstaged edit\n")
    _, report = move(rig, tmp_path, excludes=(".env.local", "keep.txt"))
    assert report.state == "restored", report
    dest = rig.dst_home / "proj" / ".worktrees" / "feat"
    assert not (dest / ".env.local").exists()
    assert (dest / "keep.txt").read_text() == "keep\n"
    assert "keep.txt" not in git(dest, "diff", "--cached", "--name-only")


def test_snapshot_leaves_the_source_index_byte_identical(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    index = Path(git(rig.worktree, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    before = index.read_bytes()
    wt.snapshot(rig.worktree, SID, 0)
    assert index.read_bytes() == before
    assert not list(index.parent.glob("lop-move-index-*"))


def test_snapshot_is_idempotent_for_the_same_tree(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    first = wt.snapshot(rig.worktree, SID, 0)
    second = wt.snapshot(rig.worktree, SID, 0)
    assert first == second


def test_unmerged_worktree_refuses_snapshot(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, dirty=False)
    git(rig.worktree, "checkout", "-q", "-b", "side", "HEAD~1")
    (rig.worktree / "feat.txt").write_text("conflicting\n")
    git(rig.worktree, "commit", "-qam", "side")
    git(rig.worktree, "merge", "-q", "feat", check=False)
    with pytest.raises(wt.MoveWorksetError) as info:
        wt.snapshot(rig.worktree, SID, 0)
    assert info.value.reason == "unmerged"


def test_snapshot_sees_a_same_size_edit_that_only_racy_git_detects(tmp_path: Path) -> None:
    """Regression: copying the index with a fresh mtime defeated git's racy-clean check.

    The edit keeps the size and mtime git recorded, and ctime is not trusted, so
    only "entry mtime >= index file mtime" tells git to re-hash the file. A temp
    index stamped "now" made the edit look clean and the snapshot carried the
    old bytes (found by the recall test under a slow scratch filesystem).
    """
    rig = make_rig(tmp_path, dirty=False)
    git(rig.worktree, "config", "core.trustctime", "false")
    target = rig.worktree / "keep.txt"
    stamp = 1_700_000_000
    os.utime(target, (stamp, stamp))
    git(rig.worktree, "update-index", "--refresh")
    target.write_text("KEEP\n")  # same size as "keep\n"
    os.utime(target, (stamp, stamp))
    index = Path(git(rig.worktree, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    os.utime(index, (stamp, stamp))
    assert git(rig.worktree, "diff", "--name-only") == "keep.txt"  # git itself sees it
    snap = wt.snapshot(rig.worktree, SID, 0)
    assert git(rig.worktree, "diff", "--name-only", snap.index, snap.worktree) == "keep.txt"
