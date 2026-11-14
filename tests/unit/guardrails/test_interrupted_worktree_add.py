"""A SIGKILLed ``git worktree add`` gets git's own cleanup from the harness (2026-11-13).

Operator request: "the guardrails are not allowing agents to clean up after
themselves on obviously safe directories and worktree folders". One cause: the
bash tool's stop paths SIGKILL the command's group, which skips git's
``remove_junk`` and strands the worktree locked ``initializing`` (HIGH to the
guard, so its creator could not remove it). These tests pin the narrow settle
in ``guardrails/interrupted_worktree_add.py`` against real git; the killed shapes
are built directly (no index + ``index.lock`` for a mid-checkout kill) so the
unit lane stays fast and deterministic. The real kill path is covered by the
bash-tool test at the end.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from local_operator.guardrails import interrupted_worktree_add as iwa

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_"))]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    # No live process ever holds a cwd in these fixtures; the lsof inventory
    # is exercised by its own provenance tests.
    monkeypatch.setattr(iwa, "_live_cwd_inside", lambda _tree: False)
    return dict(os.environ)


def _git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path, env: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git("init", "-q", str(root))
    for i in range(20):
        (root / f"f{i}.txt").write_text(f"{i}\n")
    _git("add", "-A", cwd=root)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "init", cwd=root)
    return root


def _add(repo: Path, name: str, *, killed: bool, reason: str = "initializing") -> Path:
    """A linked tree in the shape a SIGKILL leaves: locked, and (killed) index-less."""
    tree = repo / ".worktrees" / name
    _git("-C", str(repo), "worktree", "add", "-q", "--detach", str(tree), "HEAD")
    admin = repo / ".git" / "worktrees" / name
    (admin / "locked").write_text(reason)
    if killed:
        (admin / "index").rename(admin / "index.lock")
        (tree / "f3.txt").unlink()  # a checkout cut short
    return tree


def _cmd(repo: Path, name: str) -> str:
    return f"git -C {repo} worktree add --detach {repo}/.worktrees/{name} HEAD"


def _porcelain(repo: Path) -> str:
    return _git("-C", str(repo), "worktree", "list", "--porcelain")


def test_killed_mid_checkout_is_removed_with_its_registration(repo, env):
    started = time.time() - 1
    tree = _add(repo, "a", killed=True)
    done = iwa.settle_after_kill(_cmd(repo, "a"), cwd=None, env=env, spawned_at=started)
    assert done == [f"removed {tree}"]
    assert not tree.exists()
    assert not (repo / ".git" / "worktrees" / "a").exists()
    assert "locked" not in _porcelain(repo)


def test_complete_checkout_is_only_unlocked(repo, env):
    started = time.time() - 1
    tree = _add(repo, "b", killed=False)
    done = iwa.settle_after_kill(_cmd(repo, "b"), cwd=None, env=env, spawned_at=started)
    assert done == [f"unlocked {tree}"]
    assert (tree / "f3.txt").is_file()
    assert _git("-C", str(tree), "status", "--porcelain") == ""
    assert "locked" not in _porcelain(repo)


@pytest.mark.parametrize(
    "case",
    [
        "lock-older-than-call",
        "other-reason",
        "non-literal",
        "keep-lock-flag",
        "other-path",
    ],
)
def test_anything_not_provably_this_calls_add_is_untouched(repo, env, case):
    started = time.time() - 1
    reason = "keep me" if case == "other-reason" else "initializing"
    tree = _add(repo, "c", killed=True, reason=reason)
    command = {
        "non-literal": f"cd {repo} && sh ./make-tree.sh",
        "keep-lock-flag": _cmd(repo, "c").replace("--detach", "--detach --lock"),
        "other-path": _cmd(repo, "other"),
    }.get(case, _cmd(repo, "c"))
    if case == "lock-older-than-call":
        started = time.time() + 60
    assert iwa.settle_after_kill(command, cwd=None, env=env, spawned_at=started) == []
    assert tree.is_dir()
    assert (repo / ".git" / "worktrees" / "c" / "locked").read_text() == reason


def test_a_lock_written_after_the_spawn_stamp_is_never_read_as_older(repo, env):
    # Regression (2026-11-14): with a time.time() stamp the kernel's coarser
    # file clock made 39 of 40 locks git wrote after the spawn read as older,
    # so the settle stranded its own lock. The marker stamp shares the file
    # clock, so every lock written after it is admitted.
    for n in range(10):
        command = _cmd(repo, f"s{n}")
        spawned_at = iwa.spawn_stamp(command)
        tree = _add(repo, f"s{n}", killed=True)
        assert iwa.settle_after_kill(command, cwd=None, env=env, spawned_at=spawned_at) == [
            f"removed {tree}"
        ]


def test_spawn_stamp_marks_only_worktree_commands_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(iwa.tempfile, "tempdir", str(tmp_path))
    before = time.time()
    assert iwa.spawn_stamp("ls -la") >= before
    assert iwa.spawn_stamp("git worktree add x") >= before - 1
    assert list(tmp_path.iterdir()) == []  # the marker never outlives the call

    def refuse(*_a, **_k):
        raise OSError("read-only temp dir")

    monkeypatch.setattr(iwa.tempfile, "mkstemp", refuse)
    assert iwa.spawn_stamp("git worktree add x") >= before


@pytest.mark.parametrize("live", [True, None])
def test_a_live_or_unknown_cwd_inside_the_tree_blocks_removal(repo, env, monkeypatch, live):
    # An escaped (setsid/nohup) add would still be checking out; unknown fails closed.
    monkeypatch.setattr(iwa, "_live_cwd_inside", lambda _tree: live)
    tree = _add(repo, "d", killed=True)
    assert iwa.settle_after_kill(_cmd(repo, "d"), cwd=None, env=env, spawned_at=0) == []
    assert tree.is_dir()


def test_a_registration_not_pointing_back_is_untouched(repo, env):
    tree = _add(repo, "e", killed=True)
    (repo / ".git" / "worktrees" / "e" / "gitdir").write_text("/elsewhere/.git\n")
    assert iwa.settle_after_kill(_cmd(repo, "e"), cwd=None, env=env, spawned_at=0) == []
    assert tree.is_dir()


def test_group_gone_waits_out_macos_eperm_for_reaped_zombies(monkeypatch):
    answers = iter([PermissionError(), PermissionError(), ProcessLookupError()])

    def fake_killpg(_pgid, _sig):
        raise next(answers)

    monkeypatch.setattr(iwa.os, "killpg", fake_killpg)
    assert iwa._group_gone(4242, 2.0) is True
    monkeypatch.setattr(iwa.os, "killpg", lambda _p, _s: None)  # never dies
    assert iwa._group_gone(4242, 0.1) is False


def _hold_pid_gone(pid: int, within: float) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_bash_timeout_kill_of_a_real_add_leaves_no_lock(tmp_path, env, monkeypatch):
    """The real stop path: execute_bash's timeout SIGKILLs a checkout in flight.

    Why a smudge filter holds the add (2026-11-14): this test used to race a
    4,000-file checkout against a 0.15 s timeout, and on a fast CI runner git
    finished first (``exit code: 0``, no ``TIMEOUT``; #3937 gate run
    43586098128). Instead, one tracked file carries a ``filter=hold`` attribute
    whose smudge driver records its pid and then blocks in ``sleep``. git runs
    smudge filters inside the add's ``reset --hard`` checkout -- after it wrote
    ``locked`` = ``initializing`` and while ``index.lock`` is held -- so the add
    can never complete on its own and the timeout always lands mid-checkout.
    (``post-checkout`` would not do: git runs it after dropping the lock.)
    The add stays a real ``git worktree add``; only its pace is controlled.

    Measured on Linux (git 2.50.1, n=40): the filter starts 5-12 ms (median
    8 ms) after the add is spawned, every time with the registration locked
    ``initializing`` and ``index.lock`` present. The 2 s timeout leaves ample
    margin on slow runners, and the pid file assertion below fails loudly
    rather than passing vacuously if the kill ever beats the filter.
    """
    from local_operator.tools.builtin import execute_bash

    big = tmp_path / "big"
    big.mkdir()
    _git("init", "-q", str(big))
    for i in range(20):
        (big / f"f{i}.txt").write_text(f"{i}\n")
    (big / "hold.bin").write_text("held\n")
    (big / ".gitattributes").write_text("hold.bin filter=hold\n")
    _git("add", "-A", cwd=big)
    _git("-c", "user.name=t", "-c", "user.email=t@x", "commit", "-qm", "init", cwd=big)
    # Configured only after the commit, so the add above never runs a filter.
    pid_file = tmp_path / "hold.pid"
    hold = tmp_path / "hold.sh"
    hold.write_text('#!/bin/sh\necho $$ > "$1"\nexec sleep 300\n')
    hold.chmod(0o755)
    _git("-C", str(big), "config", "filter.hold.smudge", f"{hold} {pid_file}")
    started: list[threading.Thread | None] = []
    real = iwa.settle_in_background
    monkeypatch.setattr(
        iwa,
        "settle_in_background",
        lambda *a, **k: started.append(real(*a, **k)) or started[-1],
    )
    hold_pid: int | None = None
    reaped = False
    try:
        result = await execute_bash(
            "t", {"command": _cmd(big, "x"), "timeout": 2.0}, None, None, None
        )
        assert "TIMEOUT" in "".join(getattr(c, "text", "") for c in result.content)
        # The filter ran, so the kill landed while the checkout was in flight.
        hold_pid = int(pid_file.read_text())
        # The filter is in the command's process group: the timeout kill reaps it.
        reaped = _hold_pid_gone(hold_pid, 5.0)
        assert reaped
        thread = started[0] if started else None
        assert thread is not None
        await asyncio.to_thread(thread.join, 30)
        # Killed mid-checkout, so the settle removes the tree and its registration.
        assert not (big / ".worktrees" / "x").exists()
        assert "locked" not in _porcelain(big)
    finally:
        # Never leak the 300 s sleep if an assertion fired before the reap was
        # confirmed; once it was, the pid is not ours to signal any more.
        if hold_pid is None and pid_file.exists():
            hold_pid = int(pid_file.read_text() or 0) or None
        if hold_pid is not None and not reaped:
            try:
                os.kill(hold_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
