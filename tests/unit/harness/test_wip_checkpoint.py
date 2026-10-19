"""Real Git proof that checkpointing never borrows the user's mutable index."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from local_operator.harness.wip_checkpoint import WipCheckpointer


def git(repo: Path, *args: str) -> str:
    env = os.environ.copy()
    env.update(GIT_CONFIG_SYSTEM=os.devnull, GIT_TERMINAL_PROMPT="0")
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=env, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def make_repo(directory: Path) -> Path:
    directory.mkdir()
    git(directory, "init")
    git(directory, "config", "user.name", "Synthetic Test")
    git(directory, "config", "user.email", "test@localhost")
    (directory / "tracked").write_text("initial\n")
    (directory / ".gitignore").write_text("ignored\n")
    git(directory, "add", ".")
    git(directory, "commit", "-m", "initial")
    return directory


def checkpoint(repo: Path, tmp_path: Path, **kwargs) -> WipCheckpointer:
    child = tmp_path / "child-session"
    child.mkdir(exist_ok=True)
    return WipCheckpointer(repo, child, "job-test", "coder", **kwargs)


def test_snapshot_is_a_real_commit_without_touching_head_index_or_files(tmp_path):
    repo = make_repo(tmp_path / "repo")
    (repo / "tracked").write_text("staged\n")
    git(repo, "add", "tracked")
    (repo / "tracked").write_text("unstaged\n")
    (repo / "untracked").write_text("new\n")
    (repo / "ignored").write_text("not recoverable\n")
    head = git(repo, "rev-parse", "HEAD")
    index = (repo / ".git" / "index").read_bytes()
    files = {path.name: path.read_bytes() for path in repo.iterdir() if path.is_file()}
    cp = checkpoint(repo, tmp_path)

    result = cp.snapshot(force=True)

    assert "skipped" not in result
    assert git(repo, "rev-parse", cp.ref) == result["commit"]
    assert git(repo, "show", f"{cp.ref}:tracked") == "unstaged"
    assert git(repo, "show", f"{cp.ref}:untracked") == "new"
    assert "ignored" not in git(repo, "ls-tree", "--name-only", cp.ref).splitlines()
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / ".git" / "index").read_bytes() == index
    assert {path.name: path.read_bytes() for path in repo.iterdir() if path.is_file()} == files
    assert git(repo, "show", "-s", "--format=%an <%ae>|%cn <%ce>", cp.ref) == (
        "Local Operator WIP <local-operator-wip@localhost>|"
        "Local Operator WIP <local-operator-wip@localhost>"
    )
    payload = json.loads(cp.path.read_text())
    assert payload["child_session_id"] == "child-session"
    assert payload["tree"] == git(repo, "rev-parse", f"{cp.ref}^{{tree}}")
    assert set(payload["files"]) == {"tracked", "untracked"}
    assert stat.S_IMODE(cp.path.stat().st_mode) == 0o600
    assert not list(cp.child_dir.glob("lop-wip-index-*"))


def test_force_does_not_make_an_empty_commit_and_clear_releases_artifacts(tmp_path):
    repo = make_repo(tmp_path / "repo")
    cp = checkpoint(repo, tmp_path)
    assert cp.snapshot(force=True)["skipped"] == "clean worktree"
    assert git(repo, "for-each-ref", "refs/lop/wip") == ""
    (repo / "tracked").write_text("changed")
    first = cp.snapshot(force=True)["commit"]
    assert cp.snapshot(force=True)["commit"] == first
    cp.clear()
    assert not cp.path.exists()
    assert git(repo, "for-each-ref", "refs/lop/wip") == ""


def test_restore_hint_restores_tracked_and_untracked_files_after_restart(tmp_path):
    repo = make_repo(tmp_path / "repo with spaces")
    (repo / "tracked").write_text("saved tracked\n")
    (repo / "new file").write_text("saved new\n")
    cp = checkpoint(repo, tmp_path)
    saved = cp.snapshot(force=True)
    # Clone contains a clean HEAD and no worktree changes. Copy the recovery
    # ref explicitly: ordinary clones intentionally do not fetch private refs.
    clone = tmp_path / "clean clone"
    subprocess.run(["git", "clone", str(repo), str(clone)], check=True, capture_output=True)
    git(clone, "fetch", str(repo), f"{cp.ref}:{cp.ref}")
    restored = WipCheckpointer(repo, cp.child_dir, cp.job_id, cp.label)
    assert restored.result["commit"] == saved["commit"]
    restored.root = clone
    subprocess.run(restored.restore_hint(), shell=True, check=True)
    assert (clone / "tracked").read_text() == "saved tracked\n"
    assert (clone / "new file").read_text() == "saved new\n"


def test_non_repository_is_an_answerable_skip(tmp_path):
    cp = checkpoint(tmp_path, tmp_path)
    result = cp.snapshot(force=True)
    assert result["skipped"]
    assert not result.get("commit")
    assert json.loads(cp.path.read_text())["skipped"]


def test_unborn_head_is_captured(tmp_path):
    repo = tmp_path / "unborn"
    repo.mkdir()
    git(repo, "init")
    (repo / "new").write_text("first work")
    cp = checkpoint(repo, tmp_path)
    result = cp.snapshot(force=True)
    assert result["head"] == ""
    assert git(repo, "show", f"{cp.ref}:new") == "first work"
    assert git(repo, "rev-list", "--parents", "-1", cp.ref) == result["commit"]


@pytest.mark.parametrize("kind", ["bare", "missing-git"])
def test_unsupported_hosts_skip_without_failure(tmp_path, monkeypatch, kind):
    repo = tmp_path / "repo"
    repo.mkdir()
    if kind == "bare":
        git(repo, "init", "--bare")
    else:
        monkeypatch.setenv("PATH", "")
    cp = checkpoint(repo, tmp_path)
    assert cp.snapshot(force=True).get("skipped")


def test_step_receipts_and_debounce(tmp_path):
    repo = make_repo(tmp_path / "repo")
    cp = checkpoint(repo, tmp_path, interval_s=3600)
    cp.note_step("bash", "Writing checkpoint test", 100.0)
    before = cp.path.read_bytes()
    cp.note_step("write", "Adding recovery case", 101.0)
    assert cp.path.read_bytes() == before  # within the five-second receipt bound
    (repo / "tracked").write_text("changed")
    result = cp.snapshot(force=True)
    receipt = json.loads(cp.path.read_text())
    assert receipt["step"] == "write: Adding recovery case"
    assert receipt["last_step_at"] == 101.0
    (repo / "tracked").write_text("later")
    assert cp.snapshot()["commit"] == result["commit"]
    assert cp.snapshot(force=True)["commit"] != result["commit"]


def test_tracked_deletion_is_saved_without_touching_the_worktree(tmp_path):
    repo = make_repo(tmp_path / "repo")
    (repo / "tracked").unlink()
    cp = checkpoint(repo, tmp_path)
    result = cp.snapshot(force=True)
    assert result["files"] == ["tracked"]
    assert "tracked" not in git(repo, "ls-tree", "--name-only", cp.ref).splitlines()
    assert not (repo / "tracked").exists()
