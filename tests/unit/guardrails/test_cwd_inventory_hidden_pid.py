"""One process that hides its cwd no longer voids the cwd inventory (2026-11-14).

Found by QA on PR #3975: on Linux the kernel answers EACCES to
``readlink /proc/<pid>/cwd`` for a non-dumpable same-uid process — ``(sd-pam)``
on every host with a systemd user session — and ``_cwd_inventory`` returned
``None`` for the whole inventory. Every caller read that as "unknown", so
``interrupted_worktree_add.settle_after_kill`` never settled a killed
``git worktree add`` on such hosts.

The fix keeps the inventory and reports the hidden pid; a hidden pid still
counts as "might be inside" (fail closed), because PR #3980 rounds 1-3 found
no sound proof that such a process is outside a tree. So the settle deletes
nothing on its account: it only unlocks the half-made tree and the bash result
tells the agent to remove it. These tests drive the real scan code against a
fake ``/proc`` (one readable, one hidden and two vanishing pids) and the real
settle against real git.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import interrupted_worktree_add as iwa
from local_operator.guardrails import provenance as prov

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX /proc semantics")

READABLE, HIDDEN, GONE_ENOENT, GONE_ESRCH = 101, 102, 103, 104


class _FakePath:
    """``os.path`` with ``/proc/self`` present, so the Linux branch runs anywhere."""

    def __getattr__(self, name: str) -> Any:
        return getattr(os.path, name)

    @staticmethod
    def isdir(path: Any) -> bool:
        return True if path == "/proc/self" else os.path.isdir(path)


class _FakeProcOs:
    """``os`` whose ``/proc`` is a temp dir; ``readlink`` of a pid's cwd obeys ``state``."""

    path = _FakePath()

    def __init__(self, root: Path, state: dict[int, Any]) -> None:
        self.root = root
        self.state = state

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)

    def scandir(self, path: Any = ".") -> Iterator[os.DirEntry[str]]:
        for pid in self.state:
            (self.root / str(pid)).mkdir(exist_ok=True)
        return os.scandir(self.root if path == "/proc" else path)

    def readlink(self, path: Any, *args: Any, **kwargs: Any) -> str:
        parent, leaf = os.path.split(str(path))
        if leaf == "cwd" and os.path.dirname(parent) == str(self.root):
            outcome = self.state.get(int(os.path.basename(parent)), FileNotFoundError)
            if isinstance(outcome, type) and issubclass(outcome, BaseException):
                raise outcome(path)
            return str(outcome)
        return os.readlink(path, *args, **kwargs)


@pytest.fixture
def fake_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "proc"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    state: dict[int, Any] = {
        READABLE: str(elsewhere),
        HIDDEN: PermissionError,  # what (sd-pam) answers
        GONE_ENOENT: FileNotFoundError,  # exited between scandir and readlink
        GONE_ESRCH: ProcessLookupError,
    }
    fake_os = _FakeProcOs(root, state)
    monkeypatch.setattr(prov, "os", fake_os)
    monkeypatch.setattr(iwa, "os", fake_os)
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    return {"state": state, "elsewhere": elsewhere}


def test_one_hidden_cwd_keeps_the_inventory_and_vanished_pids_are_ignored(fake_proc):
    inventory = prov._cwd_inventory()
    assert inventory == prov.CwdInventory(
        cwds=(os.path.realpath(fake_proc["elsewhere"]),), hidden=(HIDDEN,)
    )


def test_a_hidden_pid_counts_as_maybe_inside(fake_proc, tmp_path):
    tree = os.path.realpath(tmp_path / "tree")
    assert prov._live_cwd_inside(tree) is None  # fail closed: busy to worktree_state
    # A readable cwd inside is a definite yes.
    assert prov._live_cwd_inside(os.path.realpath(fake_proc["elsewhere"])) is True
    fake_proc["state"][HIDDEN] = ProcessLookupError  # it exited: nothing hidden
    prov._cwd_inventory_cache = None
    assert prov._live_cwd_inside(tree) is False


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_"))]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", str(root))
    for i in range(5):
        (root / f"f{i}.txt").write_text(f"{i}\n")
    _git("add", "-A", cwd=root)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "init", cwd=root)
    return root


def _killed_add(repo: Path, name: str) -> Path:
    """The shape a SIGKILL mid-checkout leaves: locked ``initializing``, no index."""
    tree = repo / ".worktrees" / name
    _git("-C", str(repo), "worktree", "add", "-q", "--detach", str(tree), "HEAD")
    admin = repo / ".git" / "worktrees" / name
    (admin / "locked").write_text("initializing")
    (admin / "index").rename(admin / "index.lock")
    return tree


def _cmd(repo: Path, name: str) -> str:
    return f"git -C {repo} worktree add --detach {repo}/.worktrees/{name} HEAD"


def test_a_hidden_pid_never_lets_the_settle_delete_it_only_unlocks(fake_proc, repo):
    """The (sd-pam) host: the killed add is unlocked, kept whole, and reported."""
    env = dict(os.environ)
    started = time.time() - 1
    tree = _killed_add(repo, "a")
    admin = repo / ".git" / "worktrees" / "a"
    files = sorted(p.name for p in tree.iterdir())
    done = iwa.settle_after_kill(_cmd(repo, "a"), cwd=None, env=env, spawned_at=started)
    assert done == [f"unlocked, left in place: {tree}"]
    assert sorted(p.name for p in tree.iterdir()) == files
    assert all((tree / f"f{i}.txt").read_text() == f"{i}\n" for i in range(5))
    assert (admin / "gitdir").is_file() and (admin / "index.lock").is_file()
    assert not (admin / "locked").exists()
    porcelain = _git("-C", str(repo), "worktree", "list", "--porcelain")
    assert f"worktree {tree}\n" in porcelain and "locked" not in porcelain
    notice = iwa.left_in_place_notice(done)
    assert notice is not None and f"git worktree remove --force {tree}" in notice


def test_a_pid_that_vanishes_before_the_settle_does_not_block_removal(fake_proc, repo):
    # No hidden pid left: the pre-existing removal path, unchanged.
    env = dict(os.environ)
    fake_proc["state"][HIDDEN] = ProcessLookupError  # exited: holds no cwd
    started = time.time() - 1
    tree = _killed_add(repo, "c")
    done = iwa.settle_after_kill(_cmd(repo, "c"), cwd=None, env=env, spawned_at=started)
    assert done == [f"removed {tree}"]
    assert not tree.exists()
    assert iwa.left_in_place_notice(done) is None
