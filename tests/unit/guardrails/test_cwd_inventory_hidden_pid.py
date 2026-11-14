"""One process that hides its cwd no longer voids the cwd inventory (2026-11-14).

Found by QA on PR #3975: on Linux the kernel answers EACCES to
``readlink /proc/<pid>/cwd`` for a non-dumpable same-uid process — ``(sd-pam)``
on every host with a systemd user session — and ``_cwd_inventory`` returned
``None`` for the whole inventory. Every caller read that as "unknown", so
``interrupted_worktree_add.settle_after_kill`` never settled a killed
``git worktree add`` on such hosts.

The fix keeps the inventory and reports the hidden pid; a hidden pid still
counts as "might be inside" (fail closed) unless the spawn-time snapshot
proves it has not run since before the tree existed. These tests drive the
real scan code against a fake ``/proc`` (one readable, one hidden and two
vanishing pids) and the real settle against real git; the last tests pin the
kernel facts the proof relies on, against the real ``/proc``.
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
    marks = {HIDDEN: prov.ProcMark(start=359, switches=1, on_cpu=False)}
    monkeypatch.setattr(prov, "_proc_mark", lambda pid: marks.get(pid))
    return {"state": state, "marks": marks, "elsewhere": elsewhere}


def test_one_hidden_cwd_keeps_the_inventory_and_vanished_pids_are_ignored(fake_proc):
    inventory = prov._cwd_inventory()
    assert inventory == prov.CwdInventory(
        cwds=(os.path.realpath(fake_proc["elsewhere"]),), hidden=(HIDDEN,)
    )


def test_a_hidden_pid_counts_as_maybe_inside_unless_cleared(fake_proc, tmp_path):
    tree = os.path.realpath(tmp_path / "tree")
    assert prov._live_cwd_inside(tree) is None  # fail closed: busy to worktree_state
    assert prov._live_cwd_inside(tree, lambda pid: pid != HIDDEN) is None
    assert prov._live_cwd_inside(tree, lambda pid: pid == HIDDEN) is False
    # A readable cwd inside wins whatever the hidden pid's proof says.
    assert prov._live_cwd_inside(str(fake_proc["elsewhere"]), lambda _pid: True) is True


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


def test_a_hidden_pid_that_has_not_run_since_the_spawn_lets_the_add_settle(fake_proc, repo):
    env = dict(os.environ)
    snapshot = iwa.snapshot_before_spawn(_cmd(repo, "a"), cwd=None, env=env)
    assert snapshot is not None and HIDDEN in snapshot.marks
    started = time.time() - 1
    tree = _killed_add(repo, "a")
    done = iwa.settle_after_kill(
        _cmd(repo, "a"), cwd=None, env=env, spawned_at=started, snapshot=snapshot
    )
    assert done == [f"removed {tree}"]
    assert not tree.exists()


@pytest.mark.parametrize(
    "case",
    ["ran-since", "on-cpu", "pid-reused", "new-since-spawn", "target-existed", "no-snapshot"],
)
def test_a_hidden_pid_that_might_be_inside_blocks_the_settle(fake_proc, repo, case):
    env = dict(os.environ)
    marks: dict[int, prov.ProcMark] = fake_proc["marks"]
    if case == "new-since-spawn":
        fake_proc["state"][HIDDEN] = FileNotFoundError  # not there at the snapshot
    if case == "target-existed":
        # An empty directory git may add into: a cwd parked there needed no chdir.
        (repo / ".worktrees" / "b").mkdir(parents=True)
    snapshot = iwa.snapshot_before_spawn(_cmd(repo, "b"), cwd=None, env=env)
    if case == "target-existed":
        assert snapshot is None
    fake_proc["state"][HIDDEN] = PermissionError
    before = marks[HIDDEN]
    marks[HIDDEN] = {
        "ran-since": before._replace(switches=before.switches + 1),
        "on-cpu": before._replace(on_cpu=True),
        "pid-reused": before._replace(start=before.start + 1),
    }.get(case, before)
    started = time.time() - 1
    tree = _killed_add(repo, "b")
    done = iwa.settle_after_kill(
        _cmd(repo, "b"),
        cwd=None,
        env=env,
        spawned_at=started,
        snapshot=None if case == "no-snapshot" else snapshot,
    )
    assert done == []
    assert tree.is_dir()
    assert (repo / ".git" / "worktrees" / "b" / "locked").read_text() == "initializing"


def test_a_pid_that_vanishes_before_the_settle_does_not_block_it(fake_proc, repo):
    env = dict(os.environ)
    snapshot = iwa.snapshot_before_spawn(_cmd(repo, "c"), cwd=None, env=env)
    fake_proc["state"][HIDDEN] = ProcessLookupError  # exited: holds no cwd
    started = time.time() - 1
    tree = _killed_add(repo, "c")
    done = iwa.settle_after_kill(
        _cmd(repo, "c"), cwd=None, env=env, spawned_at=started, snapshot=snapshot
    )
    assert done == [f"removed {tree}"]


# -- the kernel facts the proof relies on (real /proc) ------------------------------

needs_proc = pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="Linux /proc only")


@needs_proc
def test_proc_mark_of_a_sleeping_process_is_stable_and_off_cpu():
    child = subprocess.Popen(["sleep", "30"])
    try:
        deadline = time.monotonic() + 5
        first = prov._proc_mark(child.pid)
        while first is not None and first.on_cpu and time.monotonic() < deadline:
            time.sleep(0.02)
            first = prov._proc_mark(child.pid)
        time.sleep(0.1)
        second = prov._proc_mark(child.pid)
        assert first is not None and second is not None
        assert not first.on_cpu and first == second
        with open(f"/proc/{child.pid}/stat", encoding="utf-8") as fh:
            stat_line = fh.read()
        assert first.start == int(stat_line.rsplit(")", 1)[1].split()[19])
    finally:
        child.kill()
        child.wait()


@needs_proc
def test_proc_mark_sees_the_reader_itself_running():
    mark = prov._proc_mark(os.getpid())
    assert mark is not None and mark.on_cpu
    assert prov._proc_mark(2**22 + 12345) is None  # no such pid: no mark, no proof
