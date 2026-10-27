"""``proc.spawn_detached`` reaps what it spawns (D-2d).

The measured defect: a headless worker's session runtime, parked on an approval,
carried a defunct ``sh`` — the clickable ``notify-send`` waiter that
``tui/notify`` launches through ``spawn_detached``. CPython reaps an abandoned
``Popen`` only at the NEXT ``Popen`` construction, so a long-lived runtime that
spawns one toast keeps that zombie for its whole life.

The cells spawn a REAL short-lived child through the real helper and ask the
kernel. ``os.waitpid(pid, WNOHANG)`` on a pid that is no longer our child
raises ``ChildProcessError``; on an unreaped zombie it returns ``(pid,
status)`` (and reaps it, which is why the control below is a separate cell).
Portable to macOS, which has no ``os.waitid``/``WNOWAIT`` to peek with.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from local_operator import proc

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX zombies only")


def _spawn_exit0(monkeypatch: pytest.MonkeyPatch) -> "subprocess.Popen[bytes]":
    """Spawn ``sh -c 'exit 0'`` through the real helper; return its Popen."""
    seen: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    def _recording(*args: object, **kwargs: object) -> "subprocess.Popen[bytes]":
        child = real(*args, **kwargs)  # type: ignore[call-overload]
        seen.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", _recording)
    try:
        assert proc.spawn_detached(["/bin/sh", "-c", "exit 0"]) is True
    finally:
        monkeypatch.setattr(subprocess, "Popen", real)
    assert len(seen) == 1
    return seen[0]


def test_a_detached_child_is_reaped_once_it_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    child = _spawn_exit0(monkeypatch)
    # The reaper's own wait records the status on the Popen it was handed: the
    # event to wait on, rather than a sleep (AGENTS.md "wait on the event").
    end = time.monotonic() + 10.0
    while child.returncode is None and time.monotonic() < end:
        time.sleep(0.02)
    assert child.returncode == 0, "the reaper never collected the child"
    with pytest.raises(ChildProcessError):
        os.waitpid(child.pid, os.WNOHANG)


def test_without_the_reaper_the_same_spawn_leaves_a_zombie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TEETH: neutralise the reaper and the kernel still holds the exited child."""
    monkeypatch.setattr(proc, "_reap_in_background", lambda child: None)
    child = _spawn_exit0(monkeypatch)
    end = time.monotonic() + 10.0
    got = 0
    while time.monotonic() < end:
        got, _status = os.waitpid(child.pid, os.WNOHANG)  # reaps it: no zombie left behind
        if got:
            break
        time.sleep(0.02)
    assert got == child.pid, "the exited child was not left for its parent to reap"
