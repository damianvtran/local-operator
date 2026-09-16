"""The shared liveness probe: a zombie is not a live process.

Three modules act on this one answer, so the cost of getting it wrong is not a
misleading log line but a disagreement between them: discovery reaping a record
whose owner is gone while the lease keeps reporting that same owner as live is
a session that `lop sessions` calls "stored" and no interface will open. The
last test in this file asserts that agreement directly, because that, not the
probe's return value, is what the incident was made of.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator import procstate
from tests.unreaped import unreaped_child


def test_a_live_process_is_not_a_zombie() -> None:
    assert procstate.is_zombie(os.getpid()) is False
    # pid 1 is this platform's init (launchd/systemd) and never a zombie.
    assert procstate.is_zombie(1) is False


def test_zero_and_negative_pids_are_not_zombies() -> None:
    """Guarded rather than probed: ``ps -p 0`` would answer about a whole group."""
    assert procstate.is_zombie(0) is False
    assert procstate.is_zombie(-1) is False


def test_a_pid_that_does_not_exist_is_not_a_zombie() -> None:
    """Fails closed with no exception — the caller decides, the probe does not."""
    assert procstate.is_zombie(2_147_483_600) is False


def test_a_killed_child_is_a_zombie_until_it_is_reaped() -> None:
    """Signal 0 cannot see the difference; this probe is what does.

    The first assertion is the whole reason the module exists: it passes for a
    process that has already exited.
    """
    with unreaped_child() as pid:
        os.kill(pid, 0)  # succeeds against a corpse: no exception
        assert procstate.is_zombie(pid) is True
    # Reaped: the pid is genuinely gone, and the probe says so.
    assert procstate.is_zombie(pid) is False


def test_an_unprobeable_pid_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doubt must read as "alive".

    A probe that answered "zombie" because it could not look would let a second
    writer take a transcript a working runtime is still appending to. That trade
    is the one this module must never make.
    """

    def _boom(*args: object, **kwargs: object) -> object:
        raise OSError("cannot probe")

    monkeypatch.setattr(procstate.subprocess, "run", _boom)
    # A pid with no /proc entry (and none on macOS), so the POSIX fork is the
    # path taken into the failure.
    assert procstate.is_zombie(2_147_483_600) is False


def test_the_lease_the_scan_and_the_attach_guard_agree_about_a_zombie(
    tmp_path: Path,
) -> None:
    """THE REGRESSION, at the level it hurt: three callers, one answer.

    Each of these knew a different thing about the same corpse. Discovery
    already probed for zombies, so it reaped the record and `lop sessions`
    showed the session as merely "stored"; the lease did not, so its claim
    stayed held; and the attach guard read the claim marker and told the user
    the session was "already open in pid N". The session was therefore
    un-attachable from the TUI, from `lop exec --resume` and from the phone,
    while appearing idle everywhere else.
    """
    from local_operator.resume import live_runtime_pid
    from local_operator.session.runtime.registry import pid_alive
    from local_operator.session_lease import _pid_state

    session_id = "agreeszombie"
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text("")
    with unreaped_child() as pid:
        (session_dir / ".session.pid").write_text(str(pid), encoding="utf-8")
        (session_dir / ".execution-lease").write_text(
            json.dumps(
                {"schema": 1, "session_id": session_id, "generation": "zombie", "pid": pid},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        # Discovery: gone, so the record is reapable and the row is "stored".
        assert pid_alive(pid, check_zombie=True) is False
        # The lease: proven dead, so the claim may be taken over.
        assert _pid_state(pid) == "dead"
        # The attach guard: nobody is running it, so offer the session.
        assert live_runtime_pid(tmp_path, session_id) is None
