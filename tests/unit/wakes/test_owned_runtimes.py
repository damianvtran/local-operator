"""The wake supervisor's ownership-scoped SIGTERM sweep (#3028 Q-S1)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.session.runtime.reclaim import RUNTIME_MODULE
from local_operator.wakes import owned_runtimes


def _row(pid: int, root: Path, owner: str, session_id: str, age: float = 30.0):
    env = (
        f"LOCAL_OPERATOR_CONFIG_DIR={root} LOP_MOBILE_CHILD_RESUME={session_id} "
        f"{owned_runtimes.OWNER_ENV}={owner}"
    )
    return SimpleNamespace(
        pid=pid,
        age_s=age,
        command=f"{sys.executable} -m {RUNTIME_MODULE} {env}",
    )


def test_two_roots_only_the_sweeping_roots_runtime_is_signalled(
    tmp_path: Path, monkeypatch
) -> None:
    """A rig supervisor must not terminate the live install's runtimes.

    The ownership token and root are both required; equal session ids do not
    make the other store's process ours. The signal-time process row is also
    rechecked, not just the census row.
    """
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    rows = [_row(111, ours, "owner-a", "shared-id"), _row(222, theirs, "owner-b", "shared-id")]
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: next(row for row in rows if row.pid == pid),
    )
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert owned_runtimes.sweep(ours, owner="owner-a") == [111]
    assert sent == [(111, signal.SIGTERM)]


def test_a_changed_pid_token_or_younger_process_is_never_signalled(
    tmp_path: Path, monkeypatch
) -> None:
    """The exact-pid signal-time guard refuses a recycled or substituted target."""
    root = tmp_path / "root"
    old = _row(111, root, "owner-a", "s1", age=30)
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: [old])
    sent: list[int] = []
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append(pid))

    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: _row(pid, root, "owner-a", "s1", age=1),
    )
    assert owned_runtimes.sweep(root, owner="owner-a") == []
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: _row(pid, root, "different-owner", "s1"),
    )
    assert owned_runtimes.sweep(root, owner="owner-a") == []
    assert sent == []


def test_a_sweep_without_a_token_or_permission_targets_nothing(tmp_path: Path, monkeypatch) -> None:
    """No token and no ``any_owner`` is NO target — never an unnamed sweep."""
    rows = [_row(111, tmp_path, "owner-a", "s1")]
    monkeypatch.setattr("local_operator.session.runtime.reclaim.runtime_processes", lambda: rows)
    monkeypatch.setattr(
        "local_operator.session.runtime.reclaim.process_row",
        lambda pid: rows[0] if pid == 111 else None,
    )
    sent: list[int] = []
    monkeypatch.setattr(owned_runtimes.os, "kill", lambda pid, sig: sent.append(pid))
    assert owned_runtimes.sweep(tmp_path) == []
    assert sent == []
    # The uninstall fallback sweeps EVERY wake-owned runtime of THIS store.
    assert owned_runtimes.sweep(tmp_path, any_owner=True) == [111]


def test_a_real_owned_process_is_signalled_by_exact_pid(tmp_path: Path, monkeypatch) -> None:
    """A real process proves the census/argv/env parsing, not just the mocks."""
    root = tmp_path / "root"
    # The host-wide census is a separately tested reclaim seam; use the one-pid
    # production probe here so this cell measures ownership and SIGTERM, not
    # the loaded host's ps latency.
    from local_operator.session.runtime import reclaim

    monkeypatch.setattr(
        reclaim,
        "runtime_processes",
        lambda: [row for pid in [child.pid] if (row := reclaim.process_row(pid)) is not None],
    )
    env = os.environ.copy()
    env.update(
        LOCAL_OPERATOR_CONFIG_DIR=str(root),
        LOP_MOBILE_CHILD_RESUME="r1gs1gnal01",
        **{owned_runtimes.OWNER_ENV: "owner-real"},
    )
    # A tiny real ``-m`` module with the runtime's argv shape; the production
    # runtime is exercised by the isolated SIGTERM rig, not by this unit cell.
    child = subprocess.Popen(
        # ``-m <runtime>`` is only the argv identity: no model runtime boots.
        # The child SLEEPS rather than SIGSTOPping itself. A stopped process
        # holds SIGTERM pending until SIGCONT, so whenever the stop landed
        # before the sweep (Linux CI) the wait below timed out and the cell
        # hung (#3028 shard 0); whether it landed first was a startup race.
        [
            sys.executable,
            "-c",
            "import time; time.sleep(120)",
            "-m",
            RUNTIME_MODULE,
        ],
        env={**env, "PYTHONPATH": str(tmp_path)},
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 15
        signalled: list[int] = []
        while time.monotonic() < deadline and not signalled:
            signalled = owned_runtimes.sweep(root, owner="owner-real")
            if not signalled:
                time.sleep(0.1)
        assert signalled == [child.pid]
        try:
            returncode = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pytest.fail(f"pid {child.pid} was signalled but did not exit within 15 s")
        assert returncode == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
